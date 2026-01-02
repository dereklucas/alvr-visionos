//
//  GestureClutchRecognizer.swift
//

import Foundation
import ARKit
import simd

enum GestureClutchSlot: Int, CaseIterable, Hashable {
    case forward
    case down
    case left
    case right
}

struct GestureClutchSafeZone: Equatable {
    var x: ClosedRange<Float>
    var y: ClosedRange<Float>
    var z: ClosedRange<Float>
}

struct GestureClutchConfig {
    // Tunables: adjust thresholds to trade off speed vs. false positives.
    var allowedHand: HandAnchor.Chirality = .left
    var palmUpDotEnter: Float = 0.7
    var palmUpDotExit: Float = 0.4
    var entryMaxLinearSpeed: Float = 0.6
    var entryMaxAngularSpeed: Float = 2.2
    var cancelLinearSpeed: Float = 1.6
    var cancelAngularSpeed: Float = 3.6
    var slotDistance: Float = 0.10
    var slotHysteresisExitFactor: Float = 0.5
    var slotHysteresisSwitchFactor: Float = 1.2
    var cooldownSeconds: Double = 0.35
    var panicWindowSeconds: Double = 0.32
    var panicForwardDistance: Float = 0.06
    var panicReturnDistance: Float = 0.03
    var safeZone = GestureClutchSafeZone(
        x: (-0.25)...(0.25),
        y: (-0.57)...(-0.17),
        z: (0.15)...(0.55)
    )
}

struct GestureClutchUpdate {
    var hand: HandAnchor.Chirality
    var isActive: Bool
    var selectedSlot: GestureClutchSlot?
    var committedSlot: GestureClutchSlot?
    var didCancel: Bool
    var didPanicToggle: Bool
    var deltaForward: Float
    var deltaRight: Float
    var deltaUp: Float
}

final class GestureClutchRecognizer {
    private struct HandState {
        var isActive = false
        var startTime: Double = 0.0
        var startPosWorld = simd_float3()
        var startPalmNormal = simd_float3()
        var selectedSlot: GestureClutchSlot?
        var lastPosWorld = simd_float3()
        var lastPalmNormal = simd_float3()
        var lastUpdateTime: Double = 0.0
        var cooldownUntil: Double = 0.0
        var panicForwardCrossed = false
        var panicReturned = false
    }

    struct HandSample {
        var positionWorld: simd_float3
        var palmNormalWorld: simd_float3
        var isTracked: Bool
    }

    var config = GestureClutchConfig()
    private var handStates: [HandAnchor.Chirality: HandState] = [:]

    static func isPalmUp(palmNormal: simd_float3, worldUp: simd_float3, threshold: Float) -> Bool {
        return simd_dot(palmNormal, worldUp) >= threshold
    }

    static func isInSafeZone(handPositionHead: simd_float3, safeZone: GestureClutchSafeZone) -> Bool {
        return safeZone.x.contains(handPositionHead.x)
            && safeZone.y.contains(handPositionHead.y)
            && safeZone.z.contains(handPositionHead.z)
    }

    static func classifySlot(deltaForward: Float, deltaRight: Float, deltaUp: Float, threshold: Float) -> (GestureClutchSlot, Float)? {
        var candidates: [(GestureClutchSlot, Float)] = []
        if deltaForward > threshold {
            candidates.append((.forward, deltaForward))
        }
        if deltaRight > threshold {
            candidates.append((.right, deltaRight))
        }
        if deltaRight < -threshold {
            candidates.append((.left, -deltaRight))
        }
        if deltaUp < -threshold {
            candidates.append((.down, -deltaUp))
        }
        return candidates.max(by: { $0.1 < $1.1 })
    }

    static func applySlotHysteresis(current: GestureClutchSlot?, candidate: GestureClutchSlot?, candidateStrength: Float, distanceFromCenter: Float, threshold: Float, exitFactor: Float, switchFactor: Float) -> GestureClutchSlot? {
        if let current {
            if distanceFromCenter < threshold * exitFactor {
                return nil
            }
            if let candidate, candidate != current, candidateStrength >= threshold * switchFactor {
                return candidate
            }
            return current
        }
        return candidate
    }

    func update(hand: HandAnchor?, headTransform: simd_float4x4, torsoYaw: Float, steamVrFromApple: simd_float4x4, now: Double) -> GestureClutchUpdate? {
        guard let hand else {
            return cancelIfActive(handedness: config.allowedHand, now: now)
        }
        if hand.chirality != config.allowedHand {
            return nil
        }

        guard let sample = makeHandSample(hand: hand, steamVrFromApple: steamVrFromApple) else {
            return cancelIfActive(handedness: hand.chirality, now: now)
        }

        var state = handStates[hand.chirality] ?? HandState()

        _ = torsoYaw
        let headPosition = headTransform.columns.3.asFloat3()
        var yawForward = -headTransform.columns.2.asFloat3()
        yawForward.y = 0.0
        if simd_length(yawForward) < 0.001 {
            yawForward = simd_float3(0.0, 0.0, -1.0)
        } else {
            yawForward = simd_normalize(yawForward)
        }
        let worldUp = simd_float3(0.0, 1.0, 0.0)
        let yawRight = simd_normalize(simd_cross(worldUp, yawForward))
        let handRel = sample.positionWorld - headPosition
        let handPosHead = simd_float3(simd_dot(handRel, yawRight),
                                      simd_dot(handRel, worldUp),
                                      simd_dot(handRel, yawForward))

        var linearSpeed: Float = 0.0
        var angularSpeed: Float = 0.0
        if state.lastUpdateTime > 0.0 {
            let dt = Float(max(now - state.lastUpdateTime, 0.0001))
            linearSpeed = simd_length(sample.positionWorld - state.lastPosWorld) / dt
            let dotN = max(-1.0, min(1.0, simd_dot(sample.palmNormalWorld, state.lastPalmNormal)))
            angularSpeed = acos(dotN) / dt
        }

        let palmUpEnter = GestureClutchRecognizer.isPalmUp(palmNormal: sample.palmNormalWorld, worldUp: worldUp, threshold: config.palmUpDotEnter)
        let palmUpExit = simd_dot(sample.palmNormalWorld, worldUp) < config.palmUpDotExit

        var update = GestureClutchUpdate(hand: hand.chirality,
                                         isActive: state.isActive,
                                         selectedSlot: state.selectedSlot,
                                         committedSlot: nil,
                                         didCancel: false,
                                         didPanicToggle: false,
                                         deltaForward: 0.0,
                                         deltaRight: 0.0,
                                         deltaUp: 0.0)

        if state.isActive {
            if !sample.isTracked || linearSpeed > config.cancelLinearSpeed || angularSpeed > config.cancelAngularSpeed {
                state.isActive = false
                state.selectedSlot = nil
                state.cooldownUntil = now + config.cooldownSeconds
                update.isActive = false
                update.didCancel = true
            } else {
                let deltaWorld = sample.positionWorld - state.startPosWorld
                let deltaForward = simd_dot(deltaWorld, yawForward)
                let deltaRight = simd_dot(deltaWorld, yawRight)
                let deltaUp = simd_dot(deltaWorld, worldUp)
                let distanceFromCenter = simd_length(deltaWorld)
                let candidate = GestureClutchRecognizer.classifySlot(deltaForward: deltaForward,
                                                                     deltaRight: deltaRight,
                                                                     deltaUp: deltaUp,
                                                                     threshold: config.slotDistance)
                state.selectedSlot = GestureClutchRecognizer.applySlotHysteresis(current: state.selectedSlot,
                                                                                  candidate: candidate?.0,
                                                                                  candidateStrength: candidate?.1 ?? 0.0,
                                                                                  distanceFromCenter: distanceFromCenter,
                                                                                  threshold: config.slotDistance,
                                                                                  exitFactor: config.slotHysteresisExitFactor,
                                                                                  switchFactor: config.slotHysteresisSwitchFactor)
                update.selectedSlot = state.selectedSlot
                update.deltaForward = deltaForward
                update.deltaRight = deltaRight
                update.deltaUp = deltaUp

                if now - state.startTime <= config.panicWindowSeconds {
                    if deltaForward > config.panicForwardDistance {
                        state.panicForwardCrossed = true
                    }
                    if state.panicForwardCrossed && simd_length(deltaWorld) < config.panicReturnDistance {
                        state.panicReturned = true
                    }
                }

                if palmUpExit {
                    let panicTriggered = state.panicForwardCrossed && state.panicReturned && (now - state.startTime) <= config.panicWindowSeconds
                    if panicTriggered {
                        update.didPanicToggle = true
                    } else if let committed = state.selectedSlot {
                        update.committedSlot = committed
                    } else {
                        update.didCancel = true
                    }
                    state.isActive = false
                    state.selectedSlot = nil
                    state.cooldownUntil = now + config.cooldownSeconds
                }
            }
        } else {
            if now >= state.cooldownUntil && palmUpEnter && GestureClutchRecognizer.isInSafeZone(handPositionHead: handPosHead, safeZone: config.safeZone)
                && linearSpeed <= config.entryMaxLinearSpeed && angularSpeed <= config.entryMaxAngularSpeed {
                state.isActive = true
                state.startTime = now
                state.startPosWorld = sample.positionWorld
                state.startPalmNormal = sample.palmNormalWorld
                state.selectedSlot = nil
                state.panicForwardCrossed = false
                state.panicReturned = false
                update.isActive = true
            }
        }

        state.lastPosWorld = sample.positionWorld
        state.lastPalmNormal = sample.palmNormalWorld
        state.lastUpdateTime = now
        handStates[hand.chirality] = state

        update.isActive = state.isActive
        update.selectedSlot = state.selectedSlot
        return update
    }

    private func cancelIfActive(handedness: HandAnchor.Chirality, now: Double) -> GestureClutchUpdate? {
        var state = handStates[handedness] ?? HandState()
        guard state.isActive else {
            return nil
        }
        state.isActive = false
        state.selectedSlot = nil
        state.cooldownUntil = now + config.cooldownSeconds
        handStates[handedness] = state
        return GestureClutchUpdate(hand: handedness,
                                   isActive: false,
                                   selectedSlot: nil,
                                   committedSlot: nil,
                                   didCancel: true,
                                   didPanicToggle: false,
                                   deltaForward: 0.0,
                                   deltaRight: 0.0,
                                   deltaUp: 0.0)
    }

    private func makeHandSample(hand: HandAnchor, steamVrFromApple: simd_float4x4) -> HandSample? {
        guard hand.isTracked, let skeleton = hand.handSkeleton else {
            return nil
        }
        guard let wristPos = jointWorldPosition(hand: hand, joint: .wrist, skeleton: skeleton, steamVrFromApple: steamVrFromApple),
              let indexPos = jointWorldPosition(hand: hand, joint: .indexFingerMetacarpal, skeleton: skeleton, steamVrFromApple: steamVrFromApple),
              let littlePos = jointWorldPosition(hand: hand, joint: .littleFingerMetacarpal, skeleton: skeleton, steamVrFromApple: steamVrFromApple),
              let middleMetacarpalPos = jointWorldPosition(hand: hand, joint: .middleFingerMetacarpal, skeleton: skeleton, steamVrFromApple: steamVrFromApple),
              let middleKnucklePos = jointWorldPosition(hand: hand, joint: .middleFingerKnuckle, skeleton: skeleton, steamVrFromApple: steamVrFromApple) else {
            return nil
        }

        let palmRight = simd_normalize(indexPos - littlePos)
        let fingerDir = simd_normalize(middleKnucklePos - wristPos)
        var palmNormal = simd_normalize(simd_cross(palmRight, fingerDir))
        if hand.chirality == .left {
            palmNormal *= -1.0
        }
        let palmCenter = (middleMetacarpalPos + middleKnucklePos) * 0.5
        return HandSample(positionWorld: palmCenter, palmNormalWorld: palmNormal, isTracked: true)
    }

    private func jointWorldPosition(hand: HandAnchor, joint: HandSkeleton.JointName, skeleton: HandSkeleton, steamVrFromApple: simd_float4x4) -> simd_float3? {
        let jointData = skeleton.joint(joint)
        let jointTransform = steamVrFromApple * hand.originFromAnchorTransform * jointData.anchorFromJointTransform
        return jointTransform.columns.3.asFloat3()
    }
}
