//
//  DebugSettings.swift
//
// Centralized debug/runtime toggles for compositor configuration and logging.
//

import Foundation
import Metal

enum DebugLayerLayoutOverride: String {
    case auto
    case layered
    case dedicated
}

enum DebugColorFormatOverride: String {
    case `default`
    case bgra8UnormSrgb
    case rgba16Float
}

struct DebugSettings {
    private static let defaults = UserDefaults.standard

    private static func bool(forKey key: String, default defaultValue: Bool) -> Bool {
        if defaults.object(forKey: key) != nil {
            return defaults.bool(forKey: key)
        }
        return defaultValue
    }

    private static func string(forKey key: String) -> String? {
        defaults.string(forKey: key)
    }

    static var debugLoggingEnabled: Bool {
#if DEBUG
        return bool(forKey: "ALVRDebugLoggingEnabled", default: true)
#else
        return bool(forKey: "ALVRDebugLoggingEnabled", default: false)
#endif
    }

    static var forceFoveationOff: Bool {
        bool(forKey: "ALVRForceFoveationOff", default: false)
    }

    static var enableFoveationMaps: Bool {
        bool(forKey: "ALVREnableFoveationMaps", default: true)
    }

    static var forceLayout: DebugLayerLayoutOverride {
        guard let raw = string(forKey: "ALVRForceLayout"),
              let value = DebugLayerLayoutOverride(rawValue: raw) else {
            return .auto
        }
        return value
    }

    static var forceColorFormat: DebugColorFormatOverride {
        guard let raw = string(forKey: "ALVRForceColorFormat"),
              let value = DebugColorFormatOverride(rawValue: raw) else {
            return .default
        }
        return value
    }

    static var maxRenderQualityOverride: Double? {
        guard defaults.object(forKey: "ALVRMaxRenderQuality") != nil else {
            return nil
        }
        return defaults.double(forKey: "ALVRMaxRenderQuality")
    }

    static func resolvedColorFormat(default defaultFormat: MTLPixelFormat) -> MTLPixelFormat {
        switch forceColorFormat {
        case .default:
            return defaultFormat
        case .bgra8UnormSrgb:
            return .bgra8Unorm_srgb
        case .rgba16Float:
            return .rgba16Float
        }
    }
}
