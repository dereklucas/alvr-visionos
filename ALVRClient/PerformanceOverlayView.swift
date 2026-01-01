//
//  PerformanceOverlayView.swift
//

#if PERF_HUD
import SwiftUI
import Metal
import MetalKit
import os

private struct GraphView: View {
    let samples: [Double]
    let lineColor: Color
    let referenceLines: [Double]

    var body: some View {
        GeometryReader { geo in
            Canvas { context, size in
                let width = size.width
                let height = size.height
                let maxSample = max(samples.max() ?? 0, referenceLines.max() ?? 0, 1)

                for ref in referenceLines {
                    let y = height - (CGFloat(ref) / CGFloat(maxSample)) * height
                    var path = Path()
                    path.move(to: CGPoint(x: 0, y: y))
                    path.addLine(to: CGPoint(x: width, y: y))
                    context.stroke(path, with: .color(.white.opacity(0.2)), lineWidth: 1)
                }

                guard samples.count > 1 else { return }
                let step = width / CGFloat(max(samples.count - 1, 1))
                var path = Path()
                for (i, value) in samples.enumerated() {
                    let x = CGFloat(i) * step
                    let y = height - (CGFloat(value) / CGFloat(maxSample)) * height
                    if i == 0 {
                        path.move(to: CGPoint(x: x, y: y))
                    } else {
                        path.addLine(to: CGPoint(x: x, y: y))
                    }
                }
                context.stroke(path, with: .color(lineColor), lineWidth: 2)
            }
        }
    }
}

private struct MetricRow: View {
    let label: String
    let value: String

    var body: some View {
        HStack {
            Text(label)
            Spacer()
            Text(value)
        }
    }
}

private func streamFpsCap(from value: String) -> Double {
    return value == "Default" ? 90.0 : (Double(value) ?? 90.0)
}

private struct PerformanceOverlayContent: View {
    let snapshot: PerformanceTracker.Snapshot
    let streamFpsCap: Double
    let isExpanded: Bool
    let showToggle: Bool
    let onToggle: (() -> Void)?

    var body: some View {
        let latest = snapshot.latest
        let frameStats = snapshot.stats.frame
        let decodeStats = snapshot.stats.decode
        let renderStats = snapshot.stats.render
        let gpuStats = snapshot.stats.gpu
        let latestFrameMs = latest?.gpuMs ?? latest?.frameIntervalMs ?? 0
        let fps = latestFrameMs > 0 ? 1000.0 / latestFrameMs : 0
        let cappedFps = min(fps, streamFpsCap)
        let fpsRatio = streamFpsCap > 0 ? (cappedFps / streamFpsCap) : 0
        let fpsColor: Color = fpsRatio >= 0.95 ? .green : (fpsRatio >= 0.8 ? .yellow : .red)
        let frameTargetMs = streamFpsCap > 0 ? (1000.0 / streamFpsCap) : 11.11
        let frameRatio = frameTargetMs > 0 ? (frameTargetMs / max(latestFrameMs, 0.001)) : 0
        let frameColor: Color = frameRatio >= 0.95 ? .green : (frameRatio >= 0.8 ? .yellow : .red)

        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 10) {
                VStack(spacing: 2) {
                    Text("FPS")
                        .font(.system(size: 10, weight: .medium, design: .rounded))
                        .foregroundStyle(.secondary)
                    Text(latest != nil ? String(format: "%3.0f", min(cappedFps, 300)) : "--")
                        .font(.system(size: 16, weight: .bold, design: .rounded))
                        .foregroundStyle(fpsColor)
                        .monospacedDigit()
                        .frame(width: 40, alignment: .trailing)
                }
                
                Divider()
                    .frame(height: 30)
                
                VStack(alignment: .leading, spacing: 2) {
                    Text("FRAME")
                        .font(.system(size: 10, weight: .medium, design: .rounded))
                        .foregroundStyle(.secondary)
                    HStack(spacing: 4) {
                        Text(latest != nil ? String(format: "%.1f", latestFrameMs) : "--")
                            .font(.system(size: 16, weight: .bold, design: .rounded))
                            .foregroundStyle(frameColor)
                        Text("ms")
                            .font(.system(size: 8, weight: .regular, design: .rounded))
                            .foregroundStyle(.secondary)
                    }
                }
                
                Spacer()
                
                if showToggle {
                    Button {
                        onToggle?()
                    } label: {
                        Image(systemName: isExpanded ? "chevron.up.circle.fill" : "chart.xyaxis.line")
                            .font(.system(size: 14))
                            .foregroundStyle(.secondary)
                    }
                    .buttonStyle(.plain)
                }
            }

            GraphView(samples: snapshot.frameTimes, lineColor: .green, referenceLines: [11.11, 16.67])
                .frame(height: 48)

            MetricRow(label: "avg", value: String(format: "%.2fms", frameStats.avg))
                .font(.system(size: 10, weight: .regular, design: .monospaced))

            if isExpanded {
                VStack(alignment: .leading, spacing: 6) {
                    Text("Stages")
                        .font(.system(size: 11, weight: .semibold, design: .monospaced))

                    GraphView(samples: snapshot.decodeTimes, lineColor: .yellow, referenceLines: [])
                        .frame(height: 32)
                    MetricRow(label: "decode", value: String(format: "%.2fms", decodeStats.avg))
                        .font(.system(size: 10, weight: .regular, design: .monospaced))

                    GraphView(samples: snapshot.renderTimes, lineColor: .orange, referenceLines: [])
                        .frame(height: 32)
                    MetricRow(label: "render", value: String(format: "%.2fms", renderStats.avg))
                        .font(.system(size: 10, weight: .regular, design: .monospaced))

                    GraphView(samples: snapshot.gpuTimes, lineColor: .pink, referenceLines: [])
                        .frame(height: 32)
                    MetricRow(label: "gpu", value: String(format: "%.2fms", gpuStats.avg))
                        .font(.system(size: 10, weight: .regular, design: .monospaced))
                }
            }
        }
        .padding(8)
        .frame(width: 240)
        .background(.black.opacity(0.6))
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }
}

struct PerformanceOverlayPanel: View {
    @EnvironmentObject private var gStore: GlobalSettingsStore
    @State private var isExpanded = false
    
    var body: some View {
        TimelineView(.periodic(from: .now, by: 1.0 / 30.0)) { _ in
            let snapshot = PerformanceTracker.shared.snapshot()
            PerformanceOverlayContent(
                snapshot: snapshot,
                streamFpsCap: streamFpsCap(from: gStore.settings.streamFPS),
                isExpanded: isExpanded,
                showToggle: true,
                onToggle: {
                    withAnimation(.easeInOut(duration: 0.2)) {
                        isExpanded.toggle()
                    }
                }
            )
            .onTapGesture {
                isExpanded.toggle()
            }
        }
    }
}

struct PerformanceOverlaySnapshotView: View {
    let snapshot: PerformanceTracker.Snapshot
    let streamFpsCap: Double
    let isExpanded: Bool

    var body: some View {
        PerformanceOverlayContent(
            snapshot: snapshot,
            streamFpsCap: streamFpsCap,
            isExpanded: isExpanded,
            showToggle: false,
            onToggle: nil
        )
    }
}

final class PerformanceHudRenderer {
    private let device: MTLDevice
    private let textureLoader: MTKTextureLoader
    private let queue = DispatchQueue(label: "PerfHUD.Render")
    private let lock = OSAllocatedUnfairLock()
    private var texture: MTLTexture?
    private var lastUpdateNs: UInt64 = 0
    private var isUpdating = false
    private let updateIntervalNs: UInt64 = 33_000_000

    init(device: MTLDevice) {
        self.device = device
        self.textureLoader = MTKTextureLoader(device: device)
    }

    func updateIfNeeded() -> MTLTexture? {
        let now = PerfClock.nowNs()
        var shouldUpdate = false
        lock.withLock {
            if !isUpdating && now &- lastUpdateNs >= updateIntervalNs {
                isUpdating = true
                lastUpdateNs = now
                shouldUpdate = true
            }
        }
        if shouldUpdate {
            let snapshot = PerformanceTracker.shared.snapshot()
            let fpsCap = streamFpsCap(from: ALVRClientApp.gStore.settings.streamFPS)
            queue.async { [weak self] in
                guard let self else { return }
                DispatchQueue.main.async {
                    let view = PerformanceOverlaySnapshotView(snapshot: snapshot, streamFpsCap: fpsCap, isExpanded: true)
                    let renderer = ImageRenderer(content: view)
                    renderer.proposedSize = ProposedViewSize(width: 240, height: 250)
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
        }
        return lock.withLock { texture }
    }
}

#endif
