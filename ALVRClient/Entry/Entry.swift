/*
Abstract:
The Entry content for a volume.
*/

import RealityKit
import SwiftUI
import UIKit

struct Entry: View {
    @ObservedObject var eventHandler = EventHandler.shared
    @EnvironmentObject var gStore: GlobalSettingsStore
    @Binding var chromaKeyColor: Color
    @Environment(\.scenePhase) private var scenePhase
    @Environment(\.self) var environment
    let saveAction: ()->Void
    
    let refreshRatesPost20 = ["Default", "90", "96", "100", "120"]
    let refreshRatesPre20 = ["Default", "90", "96"]
    let entryControlsInset: CGFloat = 90
    
    let chromaFormatter: NumberFormatter = {
        let formatter = NumberFormatter()
        formatter.numberStyle = .decimal
        return formatter
    }()
    
    @State private var chromaRangeMaximum: Float = 1.0
    func applyRangeSettings() {
        if gStore.settings.chromaKeyDistRangeMax < 0.001 {
            gStore.settings.chromaKeyDistRangeMax = 0.001
        }
        if gStore.settings.chromaKeyDistRangeMax > 1.0 {
            gStore.settings.chromaKeyDistRangeMax = 1.0
        }
        if gStore.settings.chromaKeyDistRangeMin < 0.0 {
            gStore.settings.chromaKeyDistRangeMin = 0.0
        }
        if gStore.settings.chromaKeyDistRangeMin > 1.0 {
            gStore.settings.chromaKeyDistRangeMin = 1.0
        }
        
        if gStore.settings.chromaKeyDistRangeMin > gStore.settings.chromaKeyDistRangeMax {
            gStore.settings.chromaKeyDistRangeMin = gStore.settings.chromaKeyDistRangeMax - 0.001
        }
        chromaRangeMaximum = gStore.settings.chromaKeyDistRangeMax
        saveAction()
    }
    
    func applyStreamHz() {
        VideoHandler.applyRefreshRate(videoFormat: EventHandler.shared.videoFormat)
    }

    private func connectionInfoView() -> some View {
        GroupBox("Connection") {
            if eventHandler.hostname != "" && eventHandler.IP != "" {
                let columns = [
                    GridItem(.fixed(140), alignment: .trailing),
                    GridItem(.fixed(200), alignment: .leading)
                ]

                LazyVGrid(columns: columns, spacing: 8) {
                    Text("Host")
                        .foregroundStyle(.secondary)
                    Text(eventHandler.hostname)
                    Text("IP Address")
                        .foregroundStyle(.secondary)
                    Text(eventHandler.IP)
                    Text("Protocol")
                        .foregroundStyle(.secondary)
                    Text(eventHandler.getMdnsProtocolId())
                    Text("Streamer")
                        .foregroundStyle(.secondary)
                    if eventHandler.hostAlvrVersion != "" {
                        Text(eventHandler.hostAlvrVersion)
                    }
                    else {
                        Text("Disconnected")
                            .foregroundStyle(.secondary)
                    }
                }
                .frame(maxWidth: .infinity, alignment: .center)
            }
            else {
                Text("Searching for a streamer.")
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
    }

    private func logo3DView() -> some View {
        RealityView { content in
            let orb = ModelEntity(
                mesh: .generateSphere(radius: 0.06),
                materials: [SimpleMaterial(color: .white, roughness: 0.1, isMetallic: true)]
            )
            let plate = ModelEntity(
                mesh: .generateBox(size: 0.12),
                materials: [SimpleMaterial(color: UIColor.systemTeal, roughness: 0.2, isMetallic: true)]
            )
            plate.transform = Transform(pitch: 0.1, yaw: 0.6, roll: 0.2)

            let root = Entity()
            root.addChild(orb)
            root.addChild(plate)
            content.add(root)
        }
        .frame(width: 220, height: 220)
        .accessibilityLabel("ALVR 3D logo")
    }

    var body: some View {
        VStack(spacing: 16) {
            TabView {
                VStack(spacing: 16) {
                    logo3DView()

                    Image(.alvrCombinedLogoHqLight)
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                        .accessibilityLabel("ALVR logo")
                        .frame(maxWidth: 280, maxHeight: 80)

                    GroupBox("Stream Portal") {
                        ZStack {
                            RoundedRectangle(cornerRadius: 24)
                                .fill(.thinMaterial)
                                .overlay(
                                    RoundedRectangle(cornerRadius: 24)
                                        .strokeBorder(.white.opacity(0.2), lineWidth: 1)
                                )
                            VStack(spacing: 6) {
                                Text("Portal preview")
                                    .font(.system(size: 14, weight: .semibold))
                                Text("Placeholder for live stream entry")
                                    .font(.system(size: 11))
                                    .foregroundStyle(.secondary)
                            }
                        }
                        .frame(height: 160)
                    }

                    connectionInfoView()

                    VStack(spacing: 6) {
                        if eventHandler.alvrVersion != "" {
                            Text(eventHandler.alvrVersion)
                                .font(.system(size: 18, weight: .bold))
                        }
                        else {
                            Text("Loading settings...")
                                .font(.system(size: 18, weight: .bold))
                        }

                        Text(eventHandler.connectionFlavorText)
                            .font(.system(size: 14))
                            .foregroundStyle(.secondary)
                            .frame(maxWidth: .infinity, alignment: .center)
                    }
                }
                .frame(minWidth: 450)
                .padding()
                .tabItem {
                    Label("Join", systemImage: "sparkles")
                }
                ScrollView {
                    VStack(alignment: .leading, spacing: 16) {
                        GroupBox("Streaming") {
                            VStack(alignment: .leading, spacing: 12) {
                                HStack {
                                    Text("Stream refresh rate")
                                    Picker("Stream refresh rate", selection: $gStore.settings.streamFPS) {
                                        if #unavailable(visionOS 2.0) {
                                            ForEach(refreshRatesPre20, id: \.self) {
                                                Text($0)
                                            }
                                        }
                                        else {
                                            ForEach(refreshRatesPost20, id: \.self) {
                                                Text($0)
                                            }
                                        }
                                    }
                                    .pickerStyle(.menu)
                                    .onChange(of: gStore.settings.streamFPS) {
                                        applyStreamHz()
                                    }
                                    .frame(maxWidth: .infinity, alignment: .trailing)
                                }

                                if #unavailable(visionOS 2.0) {
                                    Text("Higher refresh rates can stutter 30P content and passthrough.")
                                        .font(.system(size: 10))
                                        .frame(maxWidth: .infinity, alignment: .leading)
                                }
                                else {
                                    Text("Higher refresh rates can stutter 30P content.")
                                        .font(.system(size: 10))
                                        .frame(maxWidth: .infinity, alignment: .leading)
                                }
                            }
                        }

                        GroupBox("Overlays") {
                            VStack(alignment: .leading, spacing: 12) {
                                Toggle(isOn: $gStore.settings.showHandsOverlaid) {
                                    Text("Overlay hands")
                                }
                                .toggleStyle(.switch)

                                Toggle(isOn: $gStore.settings.disablePersistentSystemOverlays) {
                                    Text("Disable system overlays (palm)")
                                }
                                .toggleStyle(.switch)

                                Toggle(isOn: $gStore.settings.showPerformanceHud) {
                                    Text("Performance HUD")
                                    Text("Pinned to your left forearm.")
                                        .font(.system(size: 10))
                                }
                                .toggleStyle(.switch)
                                .onChange(of: gStore.settings.showPerformanceHud) {
                                    saveAction()
                                }
                            }
                        }

                        GroupBox("Input") {
                            VStack(alignment: .leading, spacing: 12) {
                                Toggle(isOn: $gStore.settings.keepSteamVRCenter) {
                                    Text("Ignore Crown long-press in SteamVR")
                                }
                                .toggleStyle(.switch)

                                Toggle(isOn: $gStore.settings.enableDoubleTapForHands) {
                                    Text("Double-tap controllers to switch to hands")
                                }
                                .toggleStyle(.switch)

                                Toggle(isOn: $gStore.settings.emulatedPinchInteractions) {
                                    Text("Gaze-pinch as controller input")
                                }
                                .toggleStyle(.switch)
                            }
                        }

                        GroupBox("Rendering") {
                            VStack(alignment: .leading, spacing: 12) {
                                Toggle(isOn: $gStore.settings.experimental40ppd) {
                                    Text("RealityKit renderer")
                                    Text("Deprecated. May judder or cause discomfort.")
                                        .font(.system(size: 10))
                                }
                                .toggleStyle(.switch)
                            }
                        }

                        if gStore.settings.experimental40ppd {
                            GroupBox("Chroma Key") {
                                VStack(alignment: .leading, spacing: 12) {
                                    Toggle(isOn: $gStore.settings.chromaKeyEnabled) {
    #if XCODE_BETA_16
                                        if #unavailable(visionOS 2.0) {
                                            Text("Chroma-keyed passthrough")
                                            Text("Requires RealityKit renderer.")
                                                .font(.system(size: 10))
                                        }
                                        else {
                                            Text("Chroma-keyed passthrough")
                                        }
    #else
                                        Text("Chroma-keyed passthrough")
                                        Text("Requires RealityKit renderer.")
                                            .font(.system(size: 10))
    #endif
                                    }
                                    .toggleStyle(.switch)
                                    .onChange(of: gStore.settings.chromaKeyEnabled) {
                                        saveAction()
                                    }

                                    if gStore.settings.chromaKeyEnabled {
                                        ColorPicker("Key color", selection: $chromaKeyColor)
                                            .onChange(of: chromaKeyColor) {
                                                gStore.settings.chromaKeyColorR = Float((chromaKeyColor.cgColor?.components ?? [0.0, 1.0, 0.0])[0])
                                                gStore.settings.chromaKeyColorG = Float((chromaKeyColor.cgColor?.components ?? [0.0, 1.0, 0.0])[1])
                                                gStore.settings.chromaKeyColorB = Float((chromaKeyColor.cgColor?.components ?? [0.0, 1.0, 0.0])[2])
                                                saveAction()
                                           }

                                       Text("Blend distance").frame(maxWidth: .infinity, alignment: .leading)
                                       HStack {
                                           Slider(value: $gStore.settings.chromaKeyDistRangeMin,
                                                  in: 0...chromaRangeMaximum,
                                                  step: 0.01) {
                                               Text("Blend Distance Min")
                                           }
                                           .onChange(of: gStore.settings.chromaKeyDistRangeMin) {
                                               applyRangeSettings()

                                           }
                                           TextField("Min", value: $gStore.settings.chromaKeyDistRangeMin, formatter: chromaFormatter)
                                           .textFieldStyle(RoundedBorderTextFieldStyle())
                                           .onChange(of: gStore.settings.chromaKeyDistRangeMin) {
                                               applyRangeSettings()
                                           }
                                           .frame(width: 90)
                                       }
                                       HStack {
                                           Slider(value: $gStore.settings.chromaKeyDistRangeMax,
                                                  in: 0.001...1,
                                                  step: 0.01) {
                                               Text("Blend Distance Max")
                                           }
                                           .onChange(of: gStore.settings.chromaKeyDistRangeMax) {
                                               applyRangeSettings()
                                           }
                                           TextField("Max", value: $gStore.settings.chromaKeyDistRangeMax, formatter: chromaFormatter)
                                           .textFieldStyle(RoundedBorderTextFieldStyle())
                                           .onChange(of: gStore.settings.chromaKeyDistRangeMax) {
                                               applyRangeSettings()
                                           }
                                           .frame(width: 90)
                                       }
                                    }
                                }
                            }
                        }

                        GroupBox("Tracking") {
                            VStack(alignment: .leading, spacing: 12) {
                                Toggle(isOn: $gStore.settings.targetHandsAtRoundtripLatency) {
                                    Text("Predict hands at round-trip latency")
                                    Text("May increase jitter.")
                                        .font(.system(size: 10))
                                }
                                .toggleStyle(.switch)

                                if gStore.settings.experimental40ppd {
                                    Toggle(isOn: $gStore.settings.enablePersonaFaceTracking) {
                                        Text("Spatial Persona face tracking (beta)")
                                        Text("Requires RealityKit renderer.")
                                            .font(.system(size: 10))
                                    }
                                    .toggleStyle(.switch)

                                    Toggle(isOn: $gStore.settings.showFaceTrackingDebug) {
                                        Text("Face tracking debug overlay")
                                        Text("Requires RealityKit renderer.")
                                            .font(.system(size: 10))
                                    }
                                    .toggleStyle(.switch)
                                }
                            }
                        }

                        GroupBox("Comfort & Safety") {
                            VStack(alignment: .leading, spacing: 12) {
                                Text("FoV scale").frame(maxWidth: .infinity, alignment: .leading)
                                HStack {
                                   Slider(value: $gStore.settings.fovRenderScale,
                                          in: 0.2...1.6,
                                          step: 0.1) {
                                       Text("FoV Scale")
                                   }
                                   .onChange(of: gStore.settings.fovRenderScale) {
                                       applyRangeSettings()
                                   }
                                   TextField("Scale", value: $gStore.settings.fovRenderScale, formatter: chromaFormatter)
                                   .textFieldStyle(RoundedBorderTextFieldStyle())
                                   .onChange(of: gStore.settings.fovRenderScale) {
                                       applyRangeSettings()
                                   }
                                   .frame(width: 90)
                                }
                                Text("Wider FoV improves comfort; narrower FoV improves sharpness.")
                                    .font(.system(size: 10))
                                    .frame(maxWidth: .infinity, alignment: .leading)

                                Toggle(isOn: Binding(
                                    get: { gStore.settings.chaperoneDistanceCm > 0 },
                                    set: { isOn in
                                        gStore.settings.chaperoneDistanceCm = isOn ? 30 : 0
                                        saveAction()
                                    }
                                )) {
                                    Text("Proximity chaperone")
                                    Text("Highlights nearby objects.")
                                        .font(.system(size: 10))
                                }
                                .toggleStyle(.switch)

                                if gStore.settings.chaperoneDistanceCm > 0 {
                                    Text("Chaperone distance").frame(maxWidth: .infinity, alignment: .leading)
                                    HStack {
                                       Slider(value: Binding(
                                           get: { Double(gStore.settings.chaperoneDistanceCm) },
                                           set: { newValue in
                                               let rounded = Int(newValue.rounded())
                                               // Snap invalid values: if between 1-29, snap to 30
                                               if rounded > 0 && rounded < 30 {
                                                   gStore.settings.chaperoneDistanceCm = 30
                                               } else {
                                                   gStore.settings.chaperoneDistanceCm = rounded
                                               }
                                           }
                                       ),
                                       in: 30...60,
                                       step: 5) {
                                           Text("Chaperone Distance")
                                       }
                                       .onChange(of: gStore.settings.chaperoneDistanceCm) {
                                           saveAction()
                                       }

                                       Text("\(gStore.settings.chaperoneDistanceCm) cm")
                                           .frame(width: 60, alignment: .trailing)
                                           .foregroundColor(.secondary)
                                    }
                                }
                            }
                        }

                        GroupBox("Window") {
                            VStack(alignment: .leading, spacing: 12) {
                                Toggle(isOn: $gStore.settings.dismissWindowOnEnter) {
                                    Text("Close window on entry")
                                }
                                .toggleStyle(.switch)
                            }
                        }

#if IS_ALVR_TESTFLIGHT
                        if gStore.settings.experimental40ppd {
                            GroupBox("TestFlight") {
                                VStack(alignment: .leading, spacing: 12) {
                                   Toggle(isOn: $gStore.settings.forceMipmapEyeTracking) {
                                        Text("Force visionOS 1.x eye tracking")
                                        Text("Requires RealityKit renderer. Faster, but blocks left eye FoV.")
                                            .font(.system(size: 10))
                                        Text("Use Control Center > View Recording to select ALVR.")
                                            .font(.system(size: 10))
                                    }
                                    .toggleStyle(.switch)
                                }
                            }
                        }
#endif
                    }
                    .frame(minWidth: 450)
                    .padding()
                }
                .tabItem {
                    Label("Settings", systemImage: "gearshape")
                }

                ScrollView {
                    VStack(alignment: .leading, spacing: 16) {
                        GroupBox("Help and Information") {
                            VStack(alignment: .leading, spacing: 12) {
                                Text("Need help setting up your PC for ALVR? Check out our getting started guide at:")
                                    .frame(maxWidth: .infinity, alignment: .leading)
                                Link("https://github.com/alvr-org/ALVR/wiki/Installation-guide", destination: URL(string: "https://github.com/alvr-org/ALVR/wiki/Installation-guide")!)
                                    .frame(maxWidth: .infinity, alignment: .leading)
                                Text("Having trouble connecting? Framerate issues or stuttering? Check out our troubleshooting guide at:")
                                    .frame(maxWidth: .infinity, alignment: .leading)
                                Link("https://github.com/alvr-org/ALVR/wiki/Troubleshooting", destination: URL(string: "https://github.com/alvr-org/ALVR/wiki/Troubleshooting")!)
                                    .frame(maxWidth: .infinity, alignment: .leading)
                            }
                        }

                        GroupBox("Community") {
                            VStack(alignment: .leading, spacing: 12) {
                                Text("We also have Discord and Matrix chats for more complex troubleshooting and development discussion:")
                                    .frame(maxWidth: .infinity, alignment: .leading)
                                Link("https://discord.gg/ALVR", destination: URL(string: "https://discord.gg/ALVR")!)
                                    .frame(maxWidth: .infinity, alignment: .leading)
                                Link("https://matrix.to/#/#alvr:ckie.dev?via=ckie.dev", destination: URL(string: "https://matrix.to/#/#alvr:ckie.dev?via=ckie.dev")!)
                                    .frame(maxWidth: .infinity, alignment: .leading)
                            }
                        }

                        GroupBox("License") {
                            VStack(alignment: .leading, spacing: 12) {
                                Text("ALVR is licensed under the MIT license.\nCopyright (c) 2018-2019 polygraphene\nCopyright (c) 2020-2024 alvr-org")
                                    .frame(maxWidth: .infinity, alignment: .leading)
                                Link("Click here for full license information", destination: URL(string: "https://raw.githubusercontent.com/alvr-org/ALVR/master/LICENSE")!)
                                    .frame(maxWidth: .infinity, alignment: .leading)
                            }
                        }
                    }
                    .frame(width: 450)
                    .padding()
                }
                .tabItem {
                    Label("Help and Information", systemImage: "questionmark.circle")
                }
            }
            .frame(minHeight: 600)
            .padding(.horizontal)
        }
        .frame(minWidth: 650, maxWidth: 650)
        .glassBackgroundEffect()
        .onChange(of: scenePhase) {
            switch scenePhase {
            case .background:
                saveAction()
                break
            case .inactive:
                saveAction()
                break
            case .active:
                break
            @unknown default:
                break
            }
        }
        .task({
            applyRangeSettings()
        })
        .safeAreaInset(edge: .bottom) {
            EntryControls(saveAction: saveAction)
                .padding(.bottom, 12)
        }
    }
}
