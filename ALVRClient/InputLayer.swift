//
//  InputLayer.swift
//
//  Defines input layer types for the mode-switch overlay.
//  Each layer represents a different hand-tracking input interpretation.
//

import Foundation
import simd

/// Input layer types that define how hand tracking is interpreted
enum InputLayerType: Int, CaseIterable, Codable {
    case joystickMode = 0      // Hands act as virtual joysticks
    case gripTriggerMode = 1   // Hands provide grip & trigger inputs
    case gestureMode = 2       // Full gesture recognition mode
    case pointerMode = 3       // Laser pointer style interaction

    var displayName: String {
        switch self {
        case .joystickMode: return "Joystick"
        case .gripTriggerMode: return "Grip/Trigger"
        case .gestureMode: return "Gestures"
        case .pointerMode: return "Pointer"
        }
    }

    var shortName: String {
        switch self {
        case .joystickMode: return "JOY"
        case .gripTriggerMode: return "G/T"
        case .gestureMode: return "GES"
        case .pointerMode: return "PTR"
        }
    }

    /// SF Symbol name for the layer icon
    var iconName: String {
        switch self {
        case .joystickMode: return "gamecontroller"
        case .gripTriggerMode: return "hand.raised"
        case .gestureMode: return "hand.wave"
        case .pointerMode: return "arrow.up.right"
        }
    }

    /// Color associated with this layer (RGBA)
    var color: simd_float4 {
        switch self {
        case .joystickMode: return simd_float4(0.2, 0.6, 1.0, 0.85)    // Blue
        case .gripTriggerMode: return simd_float4(1.0, 0.4, 0.2, 0.85) // Orange
        case .gestureMode: return simd_float4(0.4, 1.0, 0.4, 0.85)     // Green
        case .pointerMode: return simd_float4(0.8, 0.4, 1.0, 0.85)     // Purple
        }
    }
}

/// Manages the current input layer state
class InputLayerManager: ObservableObject {
    static let shared = InputLayerManager()

    /// Currently active input layer
    @Published var currentLayer: InputLayerType = .gripTriggerMode

    /// Layers available for selection in the radial menu
    @Published var availableLayers: [InputLayerType] = InputLayerType.allCases

    /// Timestamp of last layer change
    var lastLayerChangeTime: Double = 0

    private init() {}

    /// Select a new input layer
    func selectLayer(_ layer: InputLayerType) {
        guard layer != currentLayer else { return }

        let previousLayer = currentLayer
        currentLayer = layer
        lastLayerChangeTime = CACurrentMediaTime()

        // Post notification for systems that need to respond to layer changes
        NotificationCenter.default.post(
            name: .inputLayerChanged,
            object: nil,
            userInfo: [
                "newLayer": layer,
                "previousLayer": previousLayer
            ]
        )

        print("[InputLayerManager] Layer changed: \(previousLayer.displayName) -> \(layer.displayName)")
    }

    /// Get the layer at a specific index (for radial menu selection)
    func layer(at index: Int) -> InputLayerType? {
        guard index >= 0 && index < availableLayers.count else { return nil }
        return availableLayers[index]
    }

    /// Get the index of the current layer
    var currentLayerIndex: Int {
        availableLayers.firstIndex(of: currentLayer) ?? 0
    }
}

extension Notification.Name {
    static let inputLayerChanged = Notification.Name("inputLayerChanged")
}
