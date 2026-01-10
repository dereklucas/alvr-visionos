//
//  PerformanceOverlayView.swift
//

import SwiftUI
import Metal
import os

@_silgen_name("alvr_get_last_client_statistics")
private func alvr_get_last_client_statistics(_ outStats: UnsafeMutablePointer<AlvrClientStatisticsC>) -> Bool

private struct AlvrClientStatisticsC {
    var target_timestamp_ns: UInt64 = 0
    var frame_interval_ns: UInt64 = 0
    var video_decode_ns: UInt64 = 0
    var video_decoder_queue_ns: UInt64 = 0
    var rendering_ns: UInt64 = 0
    var vsync_queue_ns: UInt64 = 0
    var total_pipeline_latency_ns: UInt64 = 0
}

private struct AlvrClientStatisticsSnapshot {
    let targetTimestampNs: UInt64
    let frameIntervalMs: Double
    let videoDecodeMs: Double
    let videoDecoderQueueMs: Double
    let renderingMs: Double
    let vsyncQueueMs: Double
    let totalPipelineLatencyMs: Double
}

private func ms(from ns: UInt64) -> Double {
    return Double(ns) / 1_000_000.0
}

private func averageMs(_ values: [Double]) -> Double {
    let filtered = values.filter { $0 > 0 }
    guard !filtered.isEmpty else { return 0 }
    return filtered.reduce(0, +) / Double(filtered.count)
}

private func tailSeries(_ values: [Double], count: Int) -> [Double] {
    guard count > 0 else { return [] }
    if values.count <= count {
        return values
    }
    return Array(values.suffix(count))
}

private struct AlvrStatsSeriesSnapshot {
    let frameTimes: [Double]
    let pipelineTimes: [Double]
    let decodeTimes: [Double]
    let renderTimes: [Double]
    let vsyncQueueTimes: [Double]
    let decoderQueueTimes: [Double]
    let latest: AlvrClientStatisticsSnapshot?
}

private final class AlvrStatsTracker {
    static let shared = AlvrStatsTracker()

    private let lock = OSAllocatedUnfairLock()
    private var ring: [AlvrClientStatisticsSnapshot?] = Array(repeating: nil, count: 60)
    private var ringIndex = 0
    private var lastTargetTimestampNs: UInt64 = 0

    private init() {}

    func recordIfNew(_ stats: AlvrClientStatisticsSnapshot) {
        guard stats.targetTimestampNs != 0 else { return }
        lock.withLock {
            if stats.targetTimestampNs == lastTargetTimestampNs {
                return
            }
            lastTargetTimestampNs = stats.targetTimestampNs
            ring[ringIndex] = stats
            ringIndex = (ringIndex + 1) % ring.count
        }
    }

    func snapshot() -> AlvrStatsSeriesSnapshot {
        lock.withLock {
            var ordered: [AlvrClientStatisticsSnapshot] = []
            ordered.reserveCapacity(ring.count)
            for offset in 0..<ring.count {
                let idx = (ringIndex + offset) % ring.count
                if let sample = ring[idx] {
                    ordered.append(sample)
                }
            }
            return AlvrStatsSeriesSnapshot(
                frameTimes: ordered.map { $0.frameIntervalMs },
                pipelineTimes: ordered.map { $0.totalPipelineLatencyMs },
                decodeTimes: ordered.map { $0.videoDecodeMs },
                renderTimes: ordered.map { $0.renderingMs },
                vsyncQueueTimes: ordered.map { $0.vsyncQueueMs },
                decoderQueueTimes: ordered.map { $0.videoDecoderQueueMs },
                latest: ordered.last
            )
        }
    }
}

private func latestAlvrStatistics() -> AlvrClientStatisticsSnapshot? {
    var stats = AlvrClientStatisticsC()
    guard alvr_get_last_client_statistics(&stats) else {
        return nil
    }
    return AlvrClientStatisticsSnapshot(
        targetTimestampNs: stats.target_timestamp_ns,
        frameIntervalMs: ms(from: stats.frame_interval_ns),
        videoDecodeMs: ms(from: stats.video_decode_ns),
        videoDecoderQueueMs: ms(from: stats.video_decoder_queue_ns),
        renderingMs: ms(from: stats.rendering_ns),
        vsyncQueueMs: ms(from: stats.vsync_queue_ns),
        totalPipelineLatencyMs: ms(from: stats.total_pipeline_latency_ns)
    )
}

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
    let alvrSeries: AlvrStatsSeriesSnapshot
    let streamFpsCap: Double

    var body: some View {
        let latest = snapshot.latest
        let alvrLatest = alvrSeries.latest
        let trackerFrameMs = latest?.frameIntervalMs ?? 0
        let alvrFrameMs = alvrLatest?.frameIntervalMs ?? 0
        let latestFrameMs = alvrFrameMs > 0 ? alvrFrameMs : trackerFrameMs
        let fps = latestFrameMs > 0 ? 1000.0 / latestFrameMs : 0
        let cappedFps = min(fps, streamFpsCap)
        let fpsRatio = streamFpsCap > 0 ? (cappedFps / streamFpsCap) : 0
        let fpsColor: Color = fpsRatio >= 0.95 ? .green : (fpsRatio >= 0.8 ? .yellow : .red)
        let frameTargetMs = streamFpsCap > 0 ? (1000.0 / streamFpsCap) : 11.11
        let primaryText = Color.white
        let secondaryText = Color.white.opacity(0.7)
        let gpuSeriesCount = {
            if !alvrSeries.frameTimes.isEmpty {
                return alvrSeries.frameTimes.count
            }
            if !alvrSeries.renderTimes.isEmpty {
                return alvrSeries.renderTimes.count
            }
            return snapshot.gpuTimes.count
        }()
        let gpuSeries = tailSeries(snapshot.gpuTimes, count: gpuSeriesCount)

        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 10) {
                VStack(spacing: 2) {
                    Text("FPS")
                        .font(.system(size: 10, weight: .medium, design: .rounded))
                        .foregroundStyle(secondaryText)
                    Text(latest != nil || alvrFrameMs > 0 ? String(format: "%3.0f", min(cappedFps, 300)) : "--")
                        .font(.system(size: 16, weight: .bold, design: .rounded))
                        .foregroundStyle(fpsColor)
                        .monospacedDigit()
                        .frame(width: 40, alignment: .trailing)
                }

                Spacer()
            }
            .foregroundStyle(primaryText)
            
            VStack(alignment: .leading, spacing: 6) {
                HStack {
                    Text("FRAME")
                        .font(.system(size: 9, weight: .semibold, design: .rounded))
                        .foregroundStyle(secondaryText)
                    Spacer()
                    Text(String(format: "%.2fms", averageMs(alvrSeries.frameTimes)))
                        .font(.system(size: 9, weight: .regular, design: .monospaced))
                        .foregroundStyle(secondaryText)
                }
                GraphView(samples: alvrSeries.frameTimes, lineColor: .green, referenceLines: [frameTargetMs])
                    .frame(height: 32)

                HStack {
                    Text("PIPELINE")
                        .font(.system(size: 9, weight: .semibold, design: .rounded))
                        .foregroundStyle(secondaryText)
                    Spacer()
                    Text(String(format: "%.2fms", averageMs(alvrSeries.pipelineTimes)))
                        .font(.system(size: 9, weight: .regular, design: .monospaced))
                        .foregroundStyle(secondaryText)
                }
                GraphView(samples: alvrSeries.pipelineTimes, lineColor: .teal, referenceLines: [frameTargetMs])
                    .frame(height: 32)

                HStack {
                    Text("DECODE")
                        .font(.system(size: 9, weight: .semibold, design: .rounded))
                        .foregroundStyle(secondaryText)
                    Spacer()
                    Text(String(format: "%.2fms", averageMs(alvrSeries.decodeTimes)))
                        .font(.system(size: 9, weight: .regular, design: .monospaced))
                        .foregroundStyle(secondaryText)
                }
                GraphView(samples: alvrSeries.decodeTimes, lineColor: .yellow, referenceLines: [frameTargetMs])
                    .frame(height: 32)
                if let alvrLatest {
                    MetricRow(label: "decoder q", value: String(format: "%.2fms", alvrLatest.videoDecoderQueueMs))
                        .font(.system(size: 10, weight: .regular, design: .monospaced))
                        .foregroundStyle(secondaryText)
                }

                HStack {
                    Text("RENDER")
                        .font(.system(size: 9, weight: .semibold, design: .rounded))
                        .foregroundStyle(secondaryText)
                    Spacer()
                    Text(String(format: "%.2fms", averageMs(alvrSeries.renderTimes)))
                        .font(.system(size: 9, weight: .regular, design: .monospaced))
                        .foregroundStyle(secondaryText)
                }
                GraphView(samples: alvrSeries.renderTimes, lineColor: .orange, referenceLines: [frameTargetMs])
                    .frame(height: 32)
                if let alvrLatest {
                    MetricRow(label: "vsync q", value: String(format: "%.2fms", alvrLatest.vsyncQueueMs))
                        .font(.system(size: 10, weight: .regular, design: .monospaced))
                        .foregroundStyle(secondaryText)
                }

                HStack {
                    Text("GPU")
                        .font(.system(size: 9, weight: .semibold, design: .rounded))
                        .foregroundStyle(secondaryText)
                    Spacer()
                    Text(String(format: "%.2fms", averageMs(gpuSeries)))
                        .font(.system(size: 9, weight: .regular, design: .monospaced))
                        .foregroundStyle(secondaryText)
                }
                GraphView(samples: gpuSeries, lineColor: .pink, referenceLines: [frameTargetMs])
                    .frame(height: 32)
                if alvrLatest == nil {
                    MetricRow(label: "alvr stats", value: "--")
                        .font(.system(size: 10, weight: .regular, design: .monospaced))
                        .foregroundStyle(secondaryText)
                }
            }
        }
        .padding(8)
        .frame(width: 240)
        .background(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .fill(Color.black)
        )
        .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
    }
}

private struct PerformanceOverlayPanel: View {
    @EnvironmentObject private var gStore: GlobalSettingsStore
    var body: some View {
        TimelineView(.periodic(from: .now, by: 1.0 / 30.0)) { _ in
            let snapshot = PerformanceTracker.shared.snapshot()
            let alvrSeries: AlvrStatsSeriesSnapshot = {
                if let alvrStats = latestAlvrStatistics() {
                    AlvrStatsTracker.shared.recordIfNew(alvrStats)
                }
                return AlvrStatsTracker.shared.snapshot()
            }()
            PerformanceOverlayContent(
                snapshot: snapshot,
                alvrSeries: alvrSeries,
                streamFpsCap: streamFpsCap(from: gStore.settings.streamFPS)
            )
        }
    }
}

private struct PerformanceOverlaySnapshotView: View {
    let snapshot: PerformanceTracker.Snapshot
    let alvrSeries: AlvrStatsSeriesSnapshot
    let streamFpsCap: Double

    var body: some View {
        PerformanceOverlayContent(
            snapshot: snapshot,
            alvrSeries: alvrSeries,
            streamFpsCap: streamFpsCap
        )
    }
}

final class PerformanceHudRenderer {
    private let device: MTLDevice
    private let queue = DispatchQueue(label: "PerfHUD.Render")
    private let lock = OSAllocatedUnfairLock()
    private var texture: MTLTexture?
    private var lastUpdateNs: UInt64 = 0
    private var isUpdating = false
    private let updateIntervalNs: UInt64 = 33_000_000

    init(device: MTLDevice) {
        self.device = device
    }

    private func makeTexture(cgImage: CGImage) -> MTLTexture? {
        let width = cgImage.width
        let height = cgImage.height
        let bytesPerPixel = 4
        let bytesPerRow = width * bytesPerPixel
        let dataSize = bytesPerRow * height

        var data = Data(count: dataSize)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue)
        let created = data.withUnsafeMutableBytes { rawBuffer -> Bool in
            guard let baseAddress = rawBuffer.baseAddress else {
                return false
            }
            guard let context = CGContext(data: baseAddress,
                                          width: width,
                                          height: height,
                                          bitsPerComponent: 8,
                                          bytesPerRow: bytesPerRow,
                                          space: colorSpace,
                                          bitmapInfo: bitmapInfo.rawValue) else {
                return false
            }
            context.clear(CGRect(x: 0, y: 0, width: width, height: height))
            context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
            return true
        }
        guard created else {
            return nil
        }

        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .bgra8Unorm,
                                                                   width: width,
                                                                   height: height,
                                                                   mipmapped: false)
        descriptor.usage = [.shaderRead]
        descriptor.storageMode = .shared
        guard let texture = device.makeTexture(descriptor: descriptor) else {
            return nil
        }
        data.withUnsafeBytes { rawBuffer in
            if let baseAddress = rawBuffer.baseAddress {
                texture.replace(region: MTLRegionMake2D(0, 0, width, height),
                                mipmapLevel: 0,
                                withBytes: baseAddress,
                                bytesPerRow: bytesPerRow)
            }
        }
        return texture
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
            if let alvrStats = latestAlvrStatistics() {
                AlvrStatsTracker.shared.recordIfNew(alvrStats)
            }
            let alvrSeries = AlvrStatsTracker.shared.snapshot()
            let fpsCap = streamFpsCap(from: ALVRClientApp.gStore.settings.streamFPS)
            queue.async { [weak self] in
                guard let self else { return }
                DispatchQueue.main.async {
                    let view = PerformanceOverlaySnapshotView(snapshot: snapshot, alvrSeries: alvrSeries, streamFpsCap: fpsCap)
                    let renderer = ImageRenderer(content: view)
                    renderer.proposedSize = ProposedViewSize(width: 240, height: 300)
                    renderer.scale = 2
                    renderer.isOpaque = false
                    guard let cgImage = renderer.cgImage else {
                        self.lock.withLock { self.isUpdating = false }
                        return
                    }
                    let newTexture = self.makeTexture(cgImage: cgImage)
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
