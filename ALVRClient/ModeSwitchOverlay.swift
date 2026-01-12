//
//  ModeSwitchOverlay.swift
//
//  State machine for the Half-Life Alyx style mode-switch radial menu.
//  Triggered by palm-up gesture within a bounded "shifting box" region.
//

import Foundation
import simd

/// State machine states for the mode switch overlay
enum ModeSwitchOverlayState: Equatable {
    case hidden
    case appearing(startTime: Double)
    case visible
    case selecting(hoveredIndex: Int)
    case disappearing(startTime: Double, selectedIndex: Int?)

    static func == (lhs: ModeSwitchOverlayState, rhs: ModeSwitchOverlayState) -> Bool {
        switch (lhs, rhs) {
        case (.hidden, .hidden): return true
        case (.visible, .visible): return true
        case (.appearing(let t1), .appearing(let t2)): return t1 == t2
        case (.selecting(let i1), .selecting(let i2)): return i1 == i2
        case (.disappearing(let t1, let s1), .disappearing(let t2, let s2)): return t1 == t2 && s1 == s2
        default: return false
        }
    }
}

/// Manages the mode-switch radial menu overlay
class ModeSwitchOverlay: ObservableObject {
    static let shared = ModeSwitchOverlay()

    // MARK: - Published State

    /// Current state of the overlay
    @Published private(set) var state: ModeSwitchOverlayState = .hidden

    /// Position of the menu in world space
    @Published var menuPosition: simd_float3 = .zero

    /// Orientation of the menu (faces the user)
    @Published var menuOrientation: simd_quatf = simd_quatf(ix: 0, iy: 0, iz: 0, r: 1)

    /// Currently hovered segment index (-1 = none/center)
    @Published private(set) var hoveredSegment: Int = -1

    // MARK: - Configuration

    /// Duration of appear animation in seconds
    let appearDuration: Double = 0.15

    /// Duration of disappear animation in seconds
    let disappearDuration: Double = 0.1

    /// Radius of the menu in meters
    let menuRadius: Float = 0.12

    /// Inner dead zone radius ratio (0-1)
    let innerRadiusRatio: Float = 0.25

    /// Number of segments in the radial menu
    var segmentCount: Int {
        InputLayerManager.shared.availableLayers.count
    }

    // MARK: - Gesture Detection Configuration

    /// How long palm must be held up before menu appears (seconds)
    let palmUpThreshold: Double = 0.25

    /// Minimum dot product for palm-up detection (0.7 = ~45 degrees)
    let palmUpDotThreshold: Float = 0.7

    /// Distance from center required to select a segment (meters)
    let selectionDistanceThreshold: Float = 0.03

    // MARK: - Shifting Box Bounds (relative to head position)

    /// Minimum bounds for the shifting box
    let shiftingBoxMin: simd_float3 = simd_float3(-0.4, -0.3, -0.6)

    /// Maximum bounds for the shifting box
    let shiftingBoxMax: simd_float3 = simd_float3(0.4, 0.1, -0.2)

    // MARK: - Private State

    private var palmUpStartTime: Double = 0
    private var lastUpdateTime: Double = 0
    private var triggeringHand: HandChirality = .right

    /// Which hand triggered the menu
    enum HandChirality {
        case left
        case right
    }

    // MARK: - Computed Properties

    /// Whether the overlay is currently visible (in any state except hidden)
    var isVisible: Bool {
        switch state {
        case .hidden: return false
        default: return true
        }
    }

    /// Whether the overlay is fully visible and interactive
    var isInteractive: Bool {
        switch state {
        case .visible, .selecting: return true
        default: return false
        }
    }

    /// Inner radius in meters
    var innerRadius: Float {
        menuRadius * innerRadiusRatio
    }

    // MARK: - Initialization

    private init() {}

    // MARK: - Public Methods

    /// Update the overlay state based on gesture input
    /// - Parameters:
    ///   - leftPalmUp: Whether left palm is facing up
    ///   - rightPalmUp: Whether right palm is facing up
    ///   - leftWristPosition: World position of left wrist
    ///   - rightWristPosition: World position of right wrist
    ///   - leftPalmNormal: Normal vector of left palm
    ///   - rightPalmNormal: Normal vector of right palm
    ///   - headPosition: Current head/device position
    ///   - currentTime: Current timestamp (CACurrentMediaTime)
    func update(
        leftPalmUp: Bool,
        rightPalmUp: Bool,
        leftWristPosition: simd_float3,
        rightWristPosition: simd_float3,
        leftPalmNormal: simd_float3,
        rightPalmNormal: simd_float3,
        headPosition: simd_float3,
        currentTime: Double
    ) {
        lastUpdateTime = currentTime

        // Determine which hand (if any) is triggering
        let leftInBox = leftPalmUp && isInShiftingBox(leftWristPosition, headPosition: headPosition)
        let rightInBox = rightPalmUp && isInShiftingBox(rightWristPosition, headPosition: headPosition)

        // Prefer right hand, but use left if only left is active
        let palmUp = leftInBox || rightInBox
        let activeHand: HandChirality = rightInBox ? .right : .left
        let activeWristPosition = activeHand == .right ? rightWristPosition : leftWristPosition
        let activePalmNormal = activeHand == .right ? rightPalmNormal : leftPalmNormal

        switch state {
        case .hidden:
            handleHiddenState(
                palmUp: palmUp,
                activeHand: activeHand,
                wristPosition: activeWristPosition,
                palmNormal: activePalmNormal,
                headPosition: headPosition,
                currentTime: currentTime
            )

        case .appearing(let startTime):
            handleAppearingState(
                palmUp: palmUp,
                startTime: startTime,
                currentTime: currentTime
            )

        case .visible, .selecting:
            handleVisibleState(
                palmUp: palmUp,
                wristPosition: activeWristPosition,
                currentTime: currentTime
            )

        case .disappearing(let startTime, let selectedIndex):
            handleDisappearingState(
                startTime: startTime,
                selectedIndex: selectedIndex,
                currentTime: currentTime
            )
        }
    }

    /// Calculate the animation progress (0 to 1) for rendering
    func appearProgress(currentTime: Double) -> Float {
        switch state {
        case .appearing(let startTime):
            let progress = (currentTime - startTime) / appearDuration
            return Float(min(1.0, max(0.0, progress)))

        case .disappearing(let startTime, _):
            let progress = 1.0 - (currentTime - startTime) / disappearDuration
            return Float(min(1.0, max(0.0, progress)))

        case .visible, .selecting:
            return 1.0

        case .hidden:
            return 0.0
        }
    }

    /// Force hide the overlay (e.g., when streaming stops)
    func forceHide() {
        state = .hidden
        hoveredSegment = -1
        palmUpStartTime = 0
    }

    // MARK: - Private Methods

    private func handleHiddenState(
        palmUp: Bool,
        activeHand: HandChirality,
        wristPosition: simd_float3,
        palmNormal: simd_float3,
        headPosition: simd_float3,
        currentTime: Double
    ) {
        if palmUp {
            if palmUpStartTime == 0 {
                // Start tracking palm-up duration
                palmUpStartTime = currentTime
            } else if currentTime - palmUpStartTime > palmUpThreshold {
                // Palm held long enough - trigger appearance
                triggeringHand = activeHand

                // Position menu slightly above the wrist
                menuPosition = wristPosition + simd_float3(0, 0.08, 0)

                // Orient menu to face the user
                menuOrientation = calculateMenuOrientation(headPosition: headPosition)

                state = .appearing(startTime: currentTime)
                palmUpStartTime = 0

                print("[ModeSwitchOverlay] Menu appearing at \(menuPosition)")
            }
        } else {
            // Palm lowered - reset timer
            palmUpStartTime = 0
        }
    }

    private func handleAppearingState(
        palmUp: Bool,
        startTime: Double,
        currentTime: Double
    ) {
        if currentTime - startTime > appearDuration {
            // Animation complete
            state = .visible
        }

        if !palmUp {
            // Palm lowered during appearance - cancel
            state = .hidden
            palmUpStartTime = 0
        }
    }

    private func handleVisibleState(
        palmUp: Bool,
        wristPosition: simd_float3,
        currentTime: Double
    ) {
        if !palmUp {
            // Palm lowered - confirm selection and begin disappearing
            let selected = hoveredSegment >= 0 ? hoveredSegment : nil
            state = .disappearing(startTime: currentTime, selectedIndex: selected)
            print("[ModeSwitchOverlay] Menu closing, selected: \(selected?.description ?? "none")")
        } else {
            // Update hover based on hand position relative to menu center
            let newHovered = calculateHoveredSegment(wristPosition: wristPosition)

            if newHovered != hoveredSegment {
                hoveredSegment = newHovered
                state = .selecting(hoveredIndex: newHovered)
            }
        }
    }

    private func handleDisappearingState(
        startTime: Double,
        selectedIndex: Int?,
        currentTime: Double
    ) {
        if currentTime - startTime > disappearDuration {
            // Animation complete - apply selection
            if let index = selectedIndex {
                confirmSelection(index)
            }

            state = .hidden
            hoveredSegment = -1
        }
    }

    /// Check if a position is within the shifting box bounds
    private func isInShiftingBox(_ position: simd_float3, headPosition: simd_float3) -> Bool {
        let relativePos = position - headPosition

        return relativePos.x >= shiftingBoxMin.x && relativePos.x <= shiftingBoxMax.x &&
               relativePos.y >= shiftingBoxMin.y && relativePos.y <= shiftingBoxMax.y &&
               relativePos.z >= shiftingBoxMin.z && relativePos.z <= shiftingBoxMax.z
    }

    /// Calculate menu orientation to face the user
    private func calculateMenuOrientation(headPosition: simd_float3) -> simd_quatf {
        // Menu should face the user (look at head position)
        let toHead = simd_normalize(headPosition - menuPosition)

        // Project onto XZ plane for horizontal rotation only
        let forward = simd_normalize(simd_float3(toHead.x, 0, toHead.z))

        // Create rotation from default forward (-Z) to face the user
        if simd_length(forward) > 0.001 {
            return simd_quatf(from: simd_float3(0, 0, -1), to: forward)
        }

        return simd_quatf(ix: 0, iy: 0, iz: 0, r: 1)
    }

    /// Calculate which segment the hand is hovering over
    private func calculateHoveredSegment(wristPosition: simd_float3) -> Int {
        // Get offset from menu center in menu-local space
        let worldOffset = wristPosition - menuPosition

        // Rotate offset into menu local space (inverse of menu orientation)
        let inverseOrientation = menuOrientation.inverse
        let localOffset = inverseOrientation.act(worldOffset)

        // Check distance from center (in XZ plane)
        let distance = simd_length(simd_float2(localOffset.x, localOffset.z))

        // If in dead zone (center), no segment selected
        if distance < selectionDistanceThreshold {
            return -1
        }

        // Calculate angle and map to segment
        let angle = atan2(localOffset.x, -localOffset.z)  // Note: -z because menu faces +z
        let normalizedAngle = (angle + .pi) / (2 * .pi)   // 0 to 1
        let segment = Int(normalizedAngle * Float(segmentCount)) % segmentCount

        return segment
    }

    /// Apply the selected layer
    private func confirmSelection(_ index: Int) {
        if let layer = InputLayerManager.shared.layer(at: index) {
            InputLayerManager.shared.selectLayer(layer)
        }
    }
}
