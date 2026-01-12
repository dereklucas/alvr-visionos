//
//  RadialMenuRenderer.swift
//
//  Metal renderer for the radial mode-switch overlay.
//  Renders a Half-Life Alyx style pie menu with segments for each input layer.
//

import Metal
import simd

class RadialMenuRenderer {
    private let device: MTLDevice
    private var segmentPipelineState: MTLRenderPipelineState!
    private var centerPipelineState: MTLRenderPipelineState!
    private var uniformBuffer: MTLBuffer!

    /// Number of triangles per segment for smooth arcs
    private let trianglesPerSegment = 16

    /// Number of triangles for the center circle
    private let centerTriangles = 32

    init(device: MTLDevice, library: MTLLibrary, colorFormat: MTLPixelFormat, depthFormat: MTLPixelFormat, viewCount: Int) throws {
        self.device = device

        // Create segment pipeline
        segmentPipelineState = try RadialMenuRenderer.buildPipeline(
            device: device,
            library: library,
            colorFormat: colorFormat,
            depthFormat: depthFormat,
            viewCount: viewCount,
            vertexFunction: "radialMenuVertexShader",
            fragmentFunction: "radialMenuFragmentShader"
        )

        // Create center indicator pipeline
        centerPipelineState = try RadialMenuRenderer.buildPipeline(
            device: device,
            library: library,
            colorFormat: colorFormat,
            depthFormat: depthFormat,
            viewCount: viewCount,
            vertexFunction: "radialMenuCenterVertexShader",
            fragmentFunction: "radialMenuCenterFragmentShader"
        )

        // Create uniform buffer (256-byte aligned)
        let alignedSize = (MemoryLayout<RadialMenuUniforms>.size + 0xFF) & ~0xFF
        uniformBuffer = device.makeBuffer(length: alignedSize, options: .storageModeShared)
        uniformBuffer.label = "RadialMenuUniforms"
    }

    private static func buildPipeline(
        device: MTLDevice,
        library: MTLLibrary,
        colorFormat: MTLPixelFormat,
        depthFormat: MTLPixelFormat,
        viewCount: Int,
        vertexFunction: String,
        fragmentFunction: String
    ) throws -> MTLRenderPipelineState {
        let pipelineDescriptor = MTLRenderPipelineDescriptor()
        pipelineDescriptor.label = "RadialMenu_\(vertexFunction)"

        guard let vertexFunc = library.makeFunction(name: vertexFunction),
              let fragmentFunc = library.makeFunction(name: fragmentFunction) else {
            throw RendererError.badVertexDescriptor
        }

        pipelineDescriptor.vertexFunction = vertexFunc
        pipelineDescriptor.fragmentFunction = fragmentFunc

        // Color attachment with alpha blending
        pipelineDescriptor.colorAttachments[0].pixelFormat = colorFormat
        pipelineDescriptor.colorAttachments[0].isBlendingEnabled = true
        pipelineDescriptor.colorAttachments[0].rgbBlendOperation = .add
        pipelineDescriptor.colorAttachments[0].alphaBlendOperation = .add
        pipelineDescriptor.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
        pipelineDescriptor.colorAttachments[0].sourceAlphaBlendFactor = .sourceAlpha
        pipelineDescriptor.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
        pipelineDescriptor.colorAttachments[0].destinationAlphaBlendFactor = .oneMinusSourceAlpha

        pipelineDescriptor.depthAttachmentPixelFormat = depthFormat
        pipelineDescriptor.maxVertexAmplificationCount = viewCount

        return try device.makeRenderPipelineState(descriptor: pipelineDescriptor)
    }

    /// Render the radial menu overlay
    /// - Parameters:
    ///   - encoder: Active render command encoder
    ///   - overlay: The ModeSwitchOverlay state machine
    ///   - viewUniforms: Buffer containing view/projection matrices
    ///   - currentTime: Current timestamp for animation
    ///   - viewports: Viewports for stereo rendering
    func render(
        encoder: MTLRenderCommandEncoder,
        overlay: ModeSwitchOverlay,
        viewUniforms: MTLBuffer,
        currentTime: Double,
        viewports: [MTLViewport]
    ) {
        guard overlay.isVisible else { return }

        // Build model matrix from overlay position/orientation
        var modelMatrix = matrix_identity_float4x4

        // Apply rotation (menu orientation)
        let rotationMatrix = simd_float4x4(overlay.menuOrientation)
        modelMatrix = rotationMatrix

        // Apply translation
        modelMatrix.columns.3 = simd_float4(overlay.menuPosition, 1.0)

        // Fill uniforms
        var uniforms = RadialMenuUniforms()
        uniforms.modelMatrix = modelMatrix
        uniforms.radius = overlay.menuRadius
        uniforms.innerRadius = overlay.innerRadius
        uniforms.segmentCount = Int32(overlay.segmentCount)
        uniforms.hoveredSegment = Int32(overlay.hoveredSegment)
        uniforms.selectedSegment = Int32(InputLayerManager.shared.currentLayerIndex)
        uniforms.animationProgress = overlay.appearProgress(currentTime: currentTime)

        // Set segment colors from InputLayerManager
        let layers = InputLayerManager.shared.availableLayers
        if layers.count > 0 { uniforms.segmentColors.0 = layers[0].color }
        if layers.count > 1 { uniforms.segmentColors.1 = layers[1].color }
        if layers.count > 2 { uniforms.segmentColors.2 = layers[2].color }
        if layers.count > 3 { uniforms.segmentColors.3 = layers[3].color }
        if layers.count > 4 { uniforms.segmentColors.4 = layers[4].color }
        if layers.count > 5 { uniforms.segmentColors.5 = layers[5].color }
        if layers.count > 6 { uniforms.segmentColors.6 = layers[6].color }
        if layers.count > 7 { uniforms.segmentColors.7 = layers[7].color }

        // Copy uniforms to buffer
        memcpy(uniformBuffer.contents(), &uniforms, MemoryLayout<RadialMenuUniforms>.size)

        // Setup vertex amplification for stereo rendering
        if viewports.count > 1 {
            var viewMappings = (0..<viewports.count).map {
                MTLVertexAmplificationViewMapping(
                    viewportArrayIndexOffset: UInt32($0),
                    renderTargetArrayIndexOffset: UInt32($0)
                )
            }
            encoder.setVertexAmplificationCount(viewports.count, viewMappings: &viewMappings)
        }

        encoder.pushDebugGroup("RadialMenu")

        // Draw center indicator first (behind segments)
        encoder.setRenderPipelineState(centerPipelineState)
        encoder.setVertexBuffer(uniformBuffer, offset: 0, index: 0)
        encoder.setVertexBuffer(viewUniforms, offset: 0, index: BufferIndex.uniforms.rawValue)
        encoder.setFragmentBuffer(uniformBuffer, offset: 0, index: 0)

        // Center: 32 triangles * 3 vertices
        encoder.drawPrimitives(
            type: .triangle,
            vertexStart: 0,
            vertexCount: centerTriangles * 3
        )

        // Draw segment ring
        encoder.setRenderPipelineState(segmentPipelineState)
        encoder.setVertexBuffer(uniformBuffer, offset: 0, index: 0)
        encoder.setVertexBuffer(viewUniforms, offset: 0, index: BufferIndex.uniforms.rawValue)
        encoder.setFragmentBuffer(uniformBuffer, offset: 0, index: 0)

        // Each segment: 16 triangles * 3 vertices, instanced per segment
        encoder.drawPrimitives(
            type: .triangle,
            vertexStart: 0,
            vertexCount: trianglesPerSegment * 3,
            instanceCount: overlay.segmentCount
        )

        encoder.popDebugGroup()
    }
}
