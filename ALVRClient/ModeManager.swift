//
//  ModeManager.swift
//

import Foundation
import ARKit

enum InputMode: Int, CaseIterable, Hashable {
    case handsSafe
    case fullController
    case lockedHold
    case mixedOffhandUi

    var displayName: String {
        switch self {
        case .handsSafe:
            return "Hands Safe"
        case .fullController:
            return "Full"
        case .lockedHold:
            return "Locked Hold"
        case .mixedOffhandUi:
            return "Mixed UI"
        }
    }
}

final class ModeManager {
    static let shared = ModeManager()

    private(set) var currentMode: InputMode = .handsSafe
    var dominantHand: HandAnchor.Chirality = .right
    var mixedModeEnabled = true

    private var leftGripLatched = false
    private var rightGripLatched = false

    func modeForSlot(_ slot: GestureClutchSlot) -> InputMode {
        switch slot {
        case .forward:
            return .fullController
        case .down:
            return .handsSafe
        case .left:
            return .lockedHold
        case .right:
            return mixedModeEnabled ? .mixedOffhandUi : .handsSafe
        }
    }

    func slotMapping() -> [GestureClutchSlot: InputMode] {
        var mapping: [GestureClutchSlot: InputMode] = [
            .forward: .fullController,
            .down: .handsSafe,
            .left: .lockedHold
        ]
        if mixedModeEnabled {
            mapping[.right] = .mixedOffhandUi
        }
        return mapping
    }

    func setMode(_ mode: InputMode) {
        if currentMode == mode {
            return
        }
        currentMode = mode
        leftGripLatched = false
        rightGripLatched = false
    }

    func toggleSafeFull() {
        if currentMode == .fullController {
            setMode(.handsSafe)
        } else {
            setMode(.fullController)
        }
    }

    func perHandMode(for hand: HandAnchor.Chirality) -> InputMode {
        if currentMode == .mixedOffhandUi {
            return hand == dominantHand ? .fullController : .handsSafe
        }
        return currentMode
    }

    func updateGripLatch(for hand: HandAnchor.Chirality, gripValue: Float, latchThreshold: Float, releaseThreshold: Float) -> Bool {
        switch hand {
        case .left:
            if leftGripLatched {
                if gripValue < releaseThreshold {
                    leftGripLatched = false
                }
            } else {
                if gripValue >= latchThreshold {
                    leftGripLatched = true
                }
            }
            return leftGripLatched
        case .right:
            if rightGripLatched {
                if gripValue < releaseThreshold {
                    rightGripLatched = false
                }
            } else {
                if gripValue >= latchThreshold {
                    rightGripLatched = true
                }
            }
            return rightGripLatched
        }
    }
}
