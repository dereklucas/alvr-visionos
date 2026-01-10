/*
Abstract:
The Entry content for a volume.
*/

import OSLog
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

    private let logoLogger = Logger(subsystem: "alvr.client", category: "Entry")
    
    @State private var chromaRangeMaximum: Float = 1.0
    @State private var portalGlowOn = false
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

    private var connectionStatusText: String {
        if eventHandler.hostname != "" && eventHandler.IP != "" {
            return eventHandler.hostAlvrVersion != "" ? "Connected" : "Searching"
        }
        return "Searching"
    }

    private var connectionStatusColor: Color {
        connectionStatusText == "Connected" ? .green : .secondary
    }

    private func sessionInfoView() -> some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Session")
                    .font(.system(size: 14, weight: .semibold))
                Spacer()
                Text(connectionStatusText)
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundStyle(connectionStatusColor)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 4)
                    .background(.thinMaterial)
                    .clipShape(Capsule())
            }

            if eventHandler.hostname != "" && eventHandler.IP != "" {
                let columns = [
                    GridItem(.fixed(140), alignment: .trailing),
                    GridItem(.fixed(200), alignment: .leading)
                ]

                LazyVGrid(columns: columns, spacing: 8) {
                    Text("Streamer")
                        .foregroundStyle(.secondary)
                    Text(eventHandler.hostname)

                    Text("IP Address")
                        .foregroundStyle(.secondary)
                    Text(eventHandler.IP)
                        .font(.system(.body, design: .monospaced))

                    Text("Protocol")
                        .foregroundStyle(.secondary)
                    Text(eventHandler.getMdnsProtocolId())
                        .font(.system(.body, design: .monospaced))

                    Text("Streamer Version")
                        .foregroundStyle(.secondary)
                    if eventHandler.hostAlvrVersion != "" {
                        Text(eventHandler.hostAlvrVersion)
                    }
                    else {
                        Text("Not connected")
                            .foregroundStyle(.secondary)
                    }
                }
                .frame(maxWidth: .infinity, alignment: .center)
            }
            else {
                Text("No streamer found.")
                    .font(.system(size: 14, weight: .semibold))
                Text("Make sure ALVR is running on your PC.")
                    .font(.system(size: 11))
                    .foregroundStyle(.secondary)
            }

            Divider()

            HStack {
                Text("Client")
                    .foregroundStyle(.secondary)
                Spacer()
                if eventHandler.alvrVersion != "" {
                    Text(eventHandler.alvrVersion)
                        .font(.system(.body, design: .monospaced))
                }
                else {
                    Text("Loading")
                        .foregroundStyle(.secondary)
                }
            }

            Text(eventHandler.connectionFlavorText)
                .font(.system(size: 13))
                .foregroundStyle(.secondary)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding(16)
        .glassBackgroundEffect(in: RoundedRectangle(cornerRadius: 20))
        .overlay(
            RoundedRectangle(cornerRadius: 20)
                .strokeBorder(.white.opacity(0.12), lineWidth: 1)
        )
    }

    private func logo3DView() -> some View {
        RealityView { content in
            if let logo = try? await Entity(named: "alvr", in: .main) {
                content.add(logo)
            }
            else {
                logoLogger.warning("Failed to load USDZ asset named 'alvr'.")
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
        }
        .frame(width: 220, height: 220)
        .accessibilityLabel("ALVR 3D logo")
    }

    var body: some View {
        let canShowStreamPreview = eventHandler.streamingActive && Settings.getAlvrSettings() != nil
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
                            RoundedRectangle(cornerRadius: 28)
                                .fill(.ultraThinMaterial)
                                .overlay(
                                    RoundedRectangle(cornerRadius: 28)
                                        .strokeBorder(.white.opacity(0.15), lineWidth: 1)
                                )

                            if canShowStreamPreview {
                                RealityKitClientView(isPreview: true)
                                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                                    .clipShape(RoundedRectangle(cornerRadius: 24))
                                    .padding(8)
                                    .allowsHitTesting(false)
                            }

                            TimelineView(.animation) { context in
                                let angle = Angle.degrees(context.date.timeIntervalSinceReferenceDate * 20)
                                RoundedRectangle(cornerRadius: 24)
                                    .fill(
                                        AngularGradient(
                                            gradient: Gradient(colors: [
                                                Color.teal.opacity(0.35),
                                                Color.white.opacity(0.15),
                                                Color.teal.opacity(0.5),
                                                Color.white.opacity(0.08),
                                                Color.teal.opacity(0.35)
                                            ]),
                                            center: .center,
                                            angle: angle
                                        )
                                    )
                                    .blendMode(.screen)
                                    .padding(10)
                                    .opacity(canShowStreamPreview ? 0.25 : 1)
                            }

                            RoundedRectangle(cornerRadius: 22)
                                .strokeBorder(
                                    LinearGradient(
                                        colors: [
                                            Color.white.opacity(0.3),
                                            Color.teal.opacity(0.6),
                                            Color.white.opacity(0.15)
                                        ],
                                        startPoint: .topLeading,
                                        endPoint: .bottomTrailing
                                    ),
                                    lineWidth: 1.5
                                )
                                .padding(14)
                                .blur(radius: portalGlowOn ? 0.2 : 0.8)
                                .opacity(portalGlowOn ? 0.9 : 0.6)

                            if !canShowStreamPreview {
                                VStack(spacing: 8) {
                                    ProgressView()
                                    Text(eventHandler.streamingActive ? "Starting stream…" : "Waiting for stream…")
                                        .font(.system(size: 13, weight: .semibold))
                                        .foregroundStyle(.secondary)
                                }
                            }
                        }
                        .frame(height: 170)
                        .onAppear {
                            withAnimation(.easeInOut(duration: 2.6).repeatForever(autoreverses: true)) {
                                portalGlowOn = true
                            }
                        }
                    }
                    .glassBackgroundEffect(in: RoundedRectangle(cornerRadius: 22))

                    sessionInfoView()
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

                                Divider()

                                Toggle(isOn: $gStore.settings.dismissWindowOnEnter) {
                                    Text("Close window on entry")
                                }
                                .toggleStyle(.switch)
                            }
                        }

                        GroupBox("Overlays") {
                            VStack(alignment: .leading, spacing: 12) {
                                Toggle(isOn: $gStore.settings.showHandsOverlaid) {
                                    Text("Overlay hands")
                                }
                                .toggleStyle(.switch)

                                Toggle(isOn: $gStore.settings.disablePersistentSystemOverlays) {
                                    VStack(alignment: .leading, spacing: 4) {
                                        Text("Hide system overlays")
                                        Text("Hides the palm-gesture system UI.")
                                            .font(.system(size: 10))
                                            .foregroundStyle(.secondary)
                                    }
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
                                    Text("Clack controllers twice while looking at them to switch to hand tracking")
                                }
                                .toggleStyle(.switch)

                                Toggle(isOn: $gStore.settings.emulatedPinchInteractions) {
                                    Text("Gaze-pinch as controller input")
                                }
                                .toggleStyle(.switch)

                                Toggle(isOn: $gStore.settings.showPinchRayDebug) {
                                    VStack(alignment: .leading, spacing: 4) {
                                        Text("Show pinch ray debug")
                                        Text("Draws the resolved hit point and ray in the debug overlay.")
                                            .font(.system(size: 10))
                                            .foregroundStyle(.secondary)
                                    }
                                }
                                .toggleStyle(.switch)
                            }
                        }

                        GroupBox("RealityKit Renderer") {
                            VStack(alignment: .leading, spacing: 12) {
                                Text("Deprecated. May judder or cause discomfort.")
                                    .font(.system(size: 10))
                                    .foregroundStyle(.secondary)
                                Toggle(isOn: $gStore.settings.experimental40ppd) {
                                    Text("Enabled")
                                }
                                .toggleStyle(.switch)
                            }
                            if gStore.settings.experimental40ppd {
                                Divider()
                                VStack(alignment: .leading, spacing: 12) {
                                    Toggle(isOn: $gStore.settings.chromaKeyEnabled) {
    #if XCODE_BETA_16
                                        if #unavailable(visionOS 2.0) {
                                            Text("Chroma-keyed passthrough")
                                        }
                                        else {
                                            Text("Chroma-keyed passthrough")
                                        }
    #else
                                        Text("Chroma-keyed passthrough")
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
                                Divider()
                                Toggle(isOn: $gStore.settings.targetHandsAtRoundtripLatency) {
                                    Text("Predict hands at round-trip latency")
                                    Text("May increase jitter.")
                                        .font(.system(size: 10))
                                }
                                .toggleStyle(.switch)

                                if gStore.settings.experimental40ppd {
                                    Divider()
                                    Toggle(isOn: $gStore.settings.enablePersonaFaceTracking) {
                                        VStack(alignment: .leading, spacing: 4) {
                                            Text("Spatial Persona face tracking")
                                            Text("Beta")
                                                .font(.system(size: 10))
                                                .foregroundStyle(.secondary)
                                        }
                                    }
                                    .toggleStyle(.switch)

                                    if gStore.settings.enablePersonaFaceTracking {
                                        Toggle(isOn: $gStore.settings.showFaceTrackingDebug) {
                                            Text("Face tracking debug overlay")
                                        }
                                        .toggleStyle(.switch)
                                    }
                                }

#if IS_ALVR_TESTFLIGHT
                                Divider()
                                Toggle(isOn: $gStore.settings.forceMipmapEyeTracking) {
                                    Text("Force visionOS 1.x eye tracking")
                                    Text("Requires RealityKit renderer. Faster, but blocks left eye FoV.")
                                        .font(.system(size: 10))
                                    Text("Use Control Center > View Recording to select ALVR.")
                                        .font(.system(size: 10))
                                }
                                .toggleStyle(.switch)
#endif
                            }
                        }

                        GroupBox("Comfort & Safety") {
                            VStack(alignment: .leading, spacing: 12) {
                                VStack(alignment: .leading, spacing: 4) {
                                    Text("FoV scale")
                                    Text("Adjusts field of view for comfort vs sharpness.")
                                        .font(.system(size: 10))
                                        .foregroundStyle(.secondary)
                                }
                                .frame(maxWidth: .infinity, alignment: .leading)
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

                                Divider()

                                Toggle(isOn: Binding(
                                    get: { gStore.settings.chaperoneDistanceCm > 0 },
                                    set: { isOn in
                                        gStore.settings.chaperoneDistanceCm = isOn ? 30 : 0
                                        saveAction()
                                    }
                                )) {
                                    VStack(alignment: .leading, spacing: 4) {
                                        Text("Proximity chaperone")
                                        Text("Highlights nearby real-world objects as you approach them.")
                                            .font(.system(size: 10))
                                            .foregroundStyle(.secondary)
                                    }
                                }
                                .toggleStyle(.switch)

                                if gStore.settings.chaperoneDistanceCm > 0 {
                                    Text("Personal bubble size")
                                        .frame(maxWidth: .infinity, alignment: .leading)
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

                    }
                    .frame(minWidth: 450)
                    .padding()
                }
                .glassBackgroundEffect(in: RoundedRectangle(cornerRadius: 24))
                .padding(.horizontal)
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
                .glassBackgroundEffect(in: RoundedRectangle(cornerRadius: 24))
                .padding(.horizontal)
                .tabItem {
                    Label("Help and Information", systemImage: "questionmark.circle")
                }
            }
            .frame(minHeight: 600)
            .padding(.horizontal)
        }
        .frame(minWidth: 650, maxWidth: 650)
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
