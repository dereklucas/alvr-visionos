//
//  ModeSwitchOverlayRenderer.swift
//

import Foundation
import SwiftUI
import Metal
import MetalKit
import os
import simd

struct ModeSwitchOverlayState: Equatable {
    var isActive: Bool
    var selectedSlot: GestureClutchSlot?
    var currentMode: InputMode
    var slotMapping: [GestureClutchSlot: InputMode]
    var showDebugQuad: Bool
    var safeZone: GestureClutchSafeZone
    var didCommit: Bool
    var didCancel: Bool
    var showIdleLabel: Bool
    var debugAnchorWorldPosition: simd_float3
    var debugAnchorValid: Bool

    static func inactive(currentMode: InputMode) -> ModeSwitchOverlayState {
        return ModeSwitchOverlayState(isActive: false,
                                      selectedSlot: nil,
                                      currentMode: currentMode,
                                      slotMapping: [:],
                                      showDebugQuad: false,
                                      safeZone: GestureClutchConfig().safeZone,
                                      didCommit: false,
                                      didCancel: false,
                                      showIdleLabel: false,
                                      debugAnchorWorldPosition: simd_float3(),
                                      debugAnchorValid: false)
    }

    func signature() -> String {
        let slotPart = slotMapping
            .sorted { $0.key.rawValue < $1.key.rawValue }
            .map { "\($0.key.rawValue):\($0.value.rawValue)" }
            .joined(separator: ",")
        let selected = selectedSlot?.rawValue ?? -1
        let safeZonePart = "\(safeZone.x.lowerBound),\(safeZone.x.upperBound),\(safeZone.y.lowerBound),\(safeZone.y.upperBound),\(safeZone.z.lowerBound),\(safeZone.z.upperBound)"
        return "\(isActive ? 1 : 0)|\(selected)|\(currentMode.rawValue)|\(slotPart)|\(showDebugQuad ? 1 : 0)|\(didCommit ? 1 : 0)|\(didCancel ? 1 : 0)|\(showIdleLabel ? 1 : 0)|\(safeZonePart)"
    }
}

private enum HudPhase: String {
    case idle
    case clutchIntro
    case selecting
    case commitFlash
    case cancelRetract
}

private struct ModeSwitchOverlayRenderState: Equatable {
    var phase: HudPhase
    var phaseT: Float
    var selectedSlot: GestureClutchSlot?
    var currentMode: InputMode
    var slotMapping: [GestureClutchSlot: InputMode]
    var showIdleLabel: Bool
}

private struct ModeSwitchOverlayView: View {
    let state: ModeSwitchOverlayRenderState

    private func slotPosition(_ slot: GestureClutchSlot, center: CGPoint, offset: CGFloat) -> CGPoint {
        switch slot {
        case .forward:
            return CGPoint(x: center.x, y: center.y - offset)
        case .down:
            return CGPoint(x: center.x, y: center.y + offset)
        case .left:
            return CGPoint(x: center.x - offset, y: center.y)
        case .right:
            return CGPoint(x: center.x + offset, y: center.y)
        }
    }

    var body: some View {
        GeometryReader { geo in
            let size = geo.size
            let center = CGPoint(x: size.width * 0.5, y: size.height * 0.5)
            let offset = min(size.width, size.height) * 0.32
            let diamond = min(size.width, size.height) * 0.36
            let introT = CGFloat(min(max(state.phaseT, 0.0), 1.0))
            let baseOffsetY: CGFloat = 44.0
            let clusterOffsetY = state.phase == .idle ? baseOffsetY : (baseOffsetY * (1.0 - introT))
            let clusterAlpha: CGFloat = (state.phase == .idle && !state.showIdleLabel) ? 0.0 : (state.phase == .idle ? 0.35 : CGFloat(0.85 * introT + 0.15))
            let slotsAlpha: CGFloat = (state.phase == .clutchIntro || state.phase == .selecting || state.phase == .commitFlash) ? CGFloat(0.9 * introT + 0.1) : 0.0
            let labelAlpha: CGFloat = state.showIdleLabel ? 0.8 : 0.0

            ZStack {
                Path { path in
                    path.move(to: CGPoint(x: center.x, y: center.y - diamond))
                    path.addLine(to: CGPoint(x: center.x + diamond, y: center.y))
                    path.addLine(to: CGPoint(x: center.x, y: center.y + diamond))
                    path.addLine(to: CGPoint(x: center.x - diamond, y: center.y))
                    path.closeSubpath()
                }
                .fill(Color.white.opacity(state.phase == .idle ? 0.0 : 0.12))
                .overlay(
                    Path { path in
                        path.move(to: CGPoint(x: center.x, y: center.y - diamond))
                        path.addLine(to: CGPoint(x: center.x + diamond, y: center.y))
                        path.addLine(to: CGPoint(x: center.x, y: center.y + diamond))
                        path.addLine(to: CGPoint(x: center.x - diamond, y: center.y))
                        path.closeSubpath()
                    }
                    .stroke(Color.white.opacity(state.phase == .idle ? 0.0 : 0.5), lineWidth: 1)
                )

                Path { path in
                    path.move(to: center)
                    path.addLine(to: CGPoint(x: center.x - diamond, y: center.y))
                    path.addLine(to: CGPoint(x: center.x, y: center.y - diamond))
                    path.closeSubpath()
                }
                .fill(Color.white.opacity(state.phase == .idle ? 0.0 : 0.05))

                Path { path in
                    path.move(to: center)
                    path.addLine(to: CGPoint(x: center.x, y: center.y - diamond))
                    path.addLine(to: CGPoint(x: center.x + diamond, y: center.y))
                    path.closeSubpath()
                }
                .fill(Color.white.opacity(state.phase == .idle ? 0.0 : 0.05))

                Path { path in
                    path.move(to: center)
                    path.addLine(to: CGPoint(x: center.x + diamond, y: center.y))
                    path.addLine(to: CGPoint(x: center.x, y: center.y + diamond))
                    path.closeSubpath()
                }
                .fill(Color.white.opacity(state.phase == .idle ? 0.0 : 0.05))

                Path { path in
                    path.move(to: center)
                    path.addLine(to: CGPoint(x: center.x, y: center.y + diamond))
                    path.addLine(to: CGPoint(x: center.x - diamond, y: center.y))
                    path.closeSubpath()
                }
                .fill(Color.white.opacity(state.phase == .idle ? 0.0 : 0.05))

                Path { path in
                    path.move(to: CGPoint(x: center.x, y: center.y - offset * 0.6))
                    path.addLine(to: CGPoint(x: center.x, y: center.y + offset * 0.6))
                    path.move(to: CGPoint(x: center.x - offset * 0.6, y: center.y))
                    path.addLine(to: CGPoint(x: center.x + offset * 0.6, y: center.y))
                }
                .stroke(.white.opacity(state.phase == .idle ? 0.0 : 0.5), lineWidth: 2)

                Circle()
                    .fill(state.selectedSlot == nil ? Color.yellow.opacity(0.9) : Color.white.opacity(0.35))
                    .frame(width: 10, height: 10)
                    .opacity(state.phase == .idle ? 0.0 : 1.0)

                ForEach(state.slotMapping.keys.sorted { $0.rawValue < $1.rawValue }, id: \.self) { slot in
                    let mode = state.slotMapping[slot] ?? .handsSafe
                    let isSelected = state.selectedSlot == slot
                    Text(mode.displayName)
                        .font(.system(size: 12, weight: .semibold, design: .rounded))
                        .foregroundStyle(isSelected ? .black : .white)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .background(isSelected ? Color.yellow : Color.black.opacity(0.55))
                        .clipShape(RoundedRectangle(cornerRadius: 6))
                        .position(slotPosition(slot, center: center, offset: offset))
                        .opacity(slotsAlpha)
                        .scaleEffect(isSelected && state.phase == .commitFlash ? 1.15 : 1.0)
                }

                VStack(spacing: 2) {
                    Text("Mode")
                        .font(.system(size: 9, weight: .medium, design: .monospaced))
                        .foregroundStyle(Color.white.opacity(0.7))
                    Text(state.currentMode.displayName)
                        .font(.system(size: 14, weight: .bold, design: .rounded))
                        .foregroundStyle(Color.white)
                }
                .opacity(labelAlpha)
                .position(x: center.x, y: center.y + clusterOffsetY)
            }
            .frame(width: size.width, height: size.height)
            .background(Color.clear)
            .opacity(clusterAlpha)
        }
    }
}

final class ModeSwitchOverlayRenderer {
    private let device: MTLDevice
    private let textureLoader: MTKTextureLoader
    private let lock = OSAllocatedUnfairLock()
    private var texture: MTLTexture?
    private var lastSignature = ""
    private var lastUpdateNs: UInt64 = 0
    private var isUpdating = false
    private let updateIntervalNs: UInt64 = 16_000_000
    private var phase: HudPhase = .idle
    private var phaseStartTime: Double = 0.0

    private let clutchIntroDuration: Double = 0.18
    private let commitFlashDuration: Double = 0.22
    private let cancelDuration: Double = 0.2

    init(device: MTLDevice) {
        self.device = device
        self.textureLoader = MTKTextureLoader(device: device)
    }

    func updateIfNeeded(state: ModeSwitchOverlayState) -> MTLTexture? {
        let now = CACurrentMediaTime()
        if state.didCommit {
            phase = .commitFlash
            phaseStartTime = now
        } else if state.didCancel {
            phase = .cancelRetract
            phaseStartTime = now
        } else if state.isActive {
            if phase == .idle || phase == .cancelRetract || phase == .commitFlash {
                phase = .clutchIntro
                phaseStartTime = now
            }
        } else if phase != .idle {
            phase = .idle
            phaseStartTime = now
        }

        if phase == .clutchIntro && (now - phaseStartTime) >= clutchIntroDuration {
            phase = .selecting
            phaseStartTime = now
        }
        if phase == .commitFlash && (now - phaseStartTime) >= commitFlashDuration {
            phase = .idle
            phaseStartTime = now
        }
        if phase == .cancelRetract && (now - phaseStartTime) >= cancelDuration {
            phase = .idle
            phaseStartTime = now
        }

        if !state.isActive && phase == .idle && !state.showIdleLabel {
            return nil
        }
        let nowNs = PerfClock.nowNs()
        let signature = state.signature()
        let phaseT: Float
        switch phase {
        case .clutchIntro:
            phaseT = Float(min(max((now - phaseStartTime) / clutchIntroDuration, 0.0), 1.0))
        case .commitFlash:
            phaseT = Float(min(max((now - phaseStartTime) / commitFlashDuration, 0.0), 1.0))
        case .cancelRetract:
            phaseT = Float(min(max((now - phaseStartTime) / cancelDuration, 0.0), 1.0))
        case .selecting:
            phaseT = 1.0
        case .idle:
            phaseT = 0.0
        }
        var shouldUpdate = false
        lock.withLock {
            if !isUpdating && (signature != lastSignature || nowNs &- lastUpdateNs >= updateIntervalNs) {
                isUpdating = true
                lastUpdateNs = nowNs
                lastSignature = signature
                shouldUpdate = true
            }
        }
        if shouldUpdate {
            DispatchQueue.main.async { [weak self] in
                guard let self else { return }
                let renderState = ModeSwitchOverlayRenderState(phase: self.phase,
                                                             phaseT: phaseT,
                                                             selectedSlot: state.selectedSlot,
                                                             currentMode: state.currentMode,
                                                             slotMapping: state.slotMapping,
                                                             showIdleLabel: state.showIdleLabel)
                let view = ModeSwitchOverlayView(state: renderState)
                let renderer = ImageRenderer(content: view)
                renderer.proposedSize = ProposedViewSize(width: 220, height: 220)
                renderer.scale = 2
                renderer.isOpaque = false
                guard let cgImage = renderer.cgImage else {
                    self.lock.withLock { self.isUpdating = false }
                    return
                }
                let options: [MTKTextureLoader.Option: Any] = [
                    .SRGB: false,
                    .origin: MTKTextureLoader.Origin.topLeft
                ]
                let newTexture = try? self.textureLoader.newTexture(cgImage: cgImage, options: options)
                self.lock.withLock {
                    self.texture = newTexture
                    self.isUpdating = false
                }
            }
        }
        return lock.withLock { texture }
    }
}
