/*
Abstract:
Controls that allow entry into the ALVR environment.
*/

import SwiftUI

/// Controls that allow entry into the ALVR environment.
struct EntryControls: View {
    @Environment(ViewModel.self) private var model
    @Environment(\.openImmersiveSpace) private var openImmersiveSpace
    @Environment(\.dismissImmersiveSpace) private var dismissImmersiveSpace
    @Environment(\.dismissWindow) private var dismissWindow
    @ObservedObject var eventHandler = EventHandler.shared
    @EnvironmentObject var gStore: GlobalSettingsStore
    
    @State private var showImmersiveSpace = false
    @State private var immersiveSpaceIsShown = false
    
    let saveAction: ()->Void

    var body: some View {
        @Bindable var model = model
        
        HStack(spacing: 17) {
            if eventHandler.connectionState == .connected {
                Toggle(isOn: $model.isShowingClient) {
                    Label("Enter", systemImage: "visionpro")
                        .labelStyle(.titleAndIcon)
                        .padding(15)
                }
            } else {
                Label("Connecting...", systemImage: "visionpro")
                    .labelStyle(.titleOnly)
                    .padding(15)
            }
        }
        .toggleStyle(.button)
        .buttonStyle(.borderless)
        .glassBackgroundEffect(in: .rect(cornerRadius: 50))

        //Enable Client
        .onChange(of: model.isShowingClient) { _, isShowing in
            Task {
                if isShowing {

                    saveAction()
                    print("Opening Immersive Space")
                    if gStore.settings.experimental40ppd {
                        if !DummyMetalRenderer.haveRenderInfo {
                            var dummySpaceIsOpened = false
                            while !dummySpaceIsOpened {
                                switch await openImmersiveSpace(id: "DummyImmersiveSpace") {
                                case .opened:
                                    dummySpaceIsOpened = true
                                case .error, .userCancelled:
                                    fallthrough
                                @unknown default:
                                    dummySpaceIsOpened = false
                                }
                            }
                            
                            while dummySpaceIsOpened && !DummyMetalRenderer.haveRenderInfo {
                                try! await Task.sleep(nanoseconds: 1_000_000)
                            }
                            
                            await dismissImmersiveSpace()
                            try! await Task.sleep(nanoseconds: 1_000_000_000)
                        }
                        
                        if !DummyMetalRenderer.haveRenderInfo {
                            print("MISSING VIEW INFO!!")
                        }
                        
                        WorldTracker.shared.worldTrackingAddedOriginAnchor = false
                        
                        print("Open real immersive space")
                        
                        switch await openImmersiveSpace(id: "RealityKitClient") {
                        case .opened:
                            immersiveSpaceIsShown = true
                        case .error, .userCancelled:
                            fallthrough
                        @unknown default:
                            immersiveSpaceIsShown = false
                            showImmersiveSpace = false
                        }
                    }
                    else {
                        await openImmersiveSpace(id: "MetalClient")
                    }
                    VideoHandler.applyRefreshRate(videoFormat: EventHandler.shared.videoFormat)
                    if gStore.settings.dismissWindowOnEnter {
                        dismissWindow(id: "Entry")
                    }
                }
            }
        }
    }
}

struct DebugControlsView: View {
#if DEBUG
    private static let defaultDebugLoggingEnabledValue = true
#else
    private static let defaultDebugLoggingEnabledValue = false
#endif
    
    @State private var didLoadDebugDefaults = false
    @State private var enableMaxRenderQualityOverride = false
    @State private var maxRenderQualityText = "1.0"

    @AppStorage("ALVRDebugLoggingEnabled") private var debugLoggingEnabled = defaultDebugLoggingEnabledValue
    @AppStorage("ALVRForceFoveationOff") private var forceFoveationOff = false
    @AppStorage("ALVREnableFoveationMaps") private var enableFoveationMaps = true
    @AppStorage("ALVRForceLayout") private var forceLayout = "auto"
    @AppStorage("ALVRForceColorFormat") private var forceColorFormat = "default"

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Toggle("Debug Logging", isOn: $debugLoggingEnabled)
            Toggle("Force Foveation Off", isOn: $forceFoveationOff)
            Toggle("Use Foveation Maps", isOn: $enableFoveationMaps)

            Picker("Layout", selection: $forceLayout) {
                Text("Auto").tag("auto")
                Text("Layered").tag("layered")
                Text("Dedicated").tag("dedicated")
            }
            .pickerStyle(.segmented)

            Picker("Color Format", selection: $forceColorFormat) {
                Text("Default").tag("default")
                Text("BGRA8 sRGB").tag("bgra8UnormSrgb")
                Text("RGBA16F").tag("rgba16Float")
            }
            .pickerStyle(.segmented)

            HStack {
                Toggle("Max Render Quality Override", isOn: $enableMaxRenderQualityOverride)
                TextField("1.0", text: $maxRenderQualityText)
                    .frame(width: 60)
                    .textInputAutocapitalization(.never)
            }
        }
        .onAppear {
            if !didLoadDebugDefaults {
                didLoadDebugDefaults = true
                loadMaxRenderQualityDefaults()
            }
        }
        .onChange(of: enableMaxRenderQualityOverride) { _, _ in
            updateMaxRenderQualityOverride()
        }
        .onChange(of: maxRenderQualityText) { _, _ in
            updateMaxRenderQualityOverride()
        }
    }

    private func loadMaxRenderQualityDefaults() {
        let defaults = UserDefaults.standard
        if let value = defaults.object(forKey: "ALVRMaxRenderQuality") as? Double {
            enableMaxRenderQualityOverride = true
            maxRenderQualityText = String(value)
        } else {
            enableMaxRenderQualityOverride = false
            maxRenderQualityText = "1.0"
        }
    }

    private func updateMaxRenderQualityOverride() {
        let defaults = UserDefaults.standard
        if !enableMaxRenderQualityOverride {
            defaults.removeObject(forKey: "ALVRMaxRenderQuality")
            return
        }
        if let value = Double(maxRenderQualityText) {
            defaults.set(value, forKey: "ALVRMaxRenderQuality")
        }
    }
}
