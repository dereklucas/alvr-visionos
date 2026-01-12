//
//  Shaders.metal
//

// File for Metal kernel and shader functions

#include <metal_stdlib>
#include <simd/simd.h>

// Including header shared between this Metal shader code and Swift/C code executing Metal API commands
#import "ShaderTypes.h"

using namespace metal;

typedef struct
{
    float3 position [[attribute(VertexAttributePosition)]];
    float2 texCoord [[attribute(VertexAttributeTexcoord)]];
} Vertex;

typedef struct
{
    float3 position [[attribute(VertexAttributePosition)]];
} VertexPosOnly;

typedef struct
{
    float4 position [[position]];
    float4 viewPosition;
    float4 color;
    float2 texCoord;
    float planeDoProximity;
} ColorInOutPlane;

typedef struct
{
    float4 position [[position]];
    float2 texCoord;
} ColorInOut;

vertex ColorInOutPlane vertexShader(Vertex in [[stage_in]],
                               ushort amp_id [[amplification_id]],
                               constant UniformsArray & uniformsArray [[ buffer(BufferIndexUniforms) ]],
                               constant PlaneUniform & planeUniform [[ buffer(BufferIndexPlaneUniforms) ]])
{
    ColorInOutPlane out;

    Uniforms uniforms = uniformsArray.uniforms[amp_id];
    
    float4 position = float4(in.position, 1.0);
    out.position = uniforms.projectionMatrix * uniforms.modelViewMatrix * planeUniform.planeTransform * position;
    out.viewPosition = uniforms.modelViewMatrix * planeUniform.planeTransform * position;
    out.texCoord = in.texCoord;
    out.color = planeUniform.planeColor;
    out.planeDoProximity = planeUniform.planeDoProximity;

    return out;
}

fragment float4 fragmentShader(ColorInOutPlane in [[stage_in]])
{
    float4 color = in.color;
    if (in.planeDoProximity >= 0.5) {
        float cameraDistance = ((-in.viewPosition.z / in.viewPosition.w));
        float cameraX = (in.viewPosition.x);
        float cameraY = (in.viewPosition.y);
        float distFromCenterOfCamera = clamp((2.0 - sqrt(cameraX*cameraX+cameraY*cameraY)) / 2.0, 0.0, 0.9);
        cameraDistance = clamp((1.5 - sqrt(cameraDistance))/1.5, 0.0, 1.0);
        
        color *= pow(distFromCenterOfCamera * cameraDistance, 2.2);
        color.a = in.color.a;
    }
    
    if (color.a <= 0.0) {
        discard_fragment();
        return float4(0.0, 0.0, 0.0, 0.0);
    }
    return color;
}

// from ALVR

// FFR_COMMON_SHADER_FORMAT

constant bool FFR_ENABLED [[ function_constant(ALVRFunctionConstantFfrEnabled) ]];
constant uint2 TARGET_RESOLUTION [[ function_constant(ALVRFunctionConstantFfrCommonShaderTargetResolution) ]];
constant uint2 OPTIMIZED_RESOLUTION [[ function_constant(ALVRFunctionConstantFfrCommonShaderOptimizedResolution) ]];
constant float2 EYE_SIZE_RATIO [[ function_constant(ALVRFunctionConstantFfrCommonShaderEyeSizeRatio) ]];
constant float2 CENTER_SIZE [[ function_constant(ALVRFunctionConstantFfrCommonShaderCenterSize) ]];
constant float2 CENTER_SHIFT [[ function_constant(ALVRFunctionConstantFfrCommonShaderCenterShift) ]];
constant float2 EDGE_RATIO [[ function_constant(ALVRFunctionConstantFfrCommonShaderEdgeRatio) ]];
constant bool CHROMAKEY_ENABLED [[ function_constant(ALVRFunctionConstantChromaKeyEnabled) ]];
constant float3 CHROMAKEY_COLOR [[ function_constant(ALVRFunctionConstantChromaKeyColor) ]];
constant float2 CHROMAKEY_LERP_DIST_RANGE [[ function_constant(ALVRFunctionConstantChromaKeyLerpDistRange) ]];
constant bool REALITYKIT_ENABLED [[ function_constant(ALVRFunctionConstantRealityKitEnabled) ]];
constant float2 VRR_SCREEN_SIZE [[ function_constant(ALVRFunctionConstantVRRScreenSize) ]];
constant float2 VRR_PHYS_SIZE [[ function_constant(ALVRFunctionConstantVRRPhysSize) ]];
constant float ENCODING_GAMMA [[ function_constant(ALVRFunctionConstantEncodingGamma) ]];
constant float4 ENCODING_YUV_TRANSFORM_0 [[ function_constant(ALVRFunctionConstantEncodingYUVTransform0) ]];
constant float4 ENCODING_YUV_TRANSFORM_1 [[ function_constant(ALVRFunctionConstantEncodingYUVTransform1) ]];
constant float4 ENCODING_YUV_TRANSFORM_2 [[ function_constant(ALVRFunctionConstantEncodingYUVTransform2) ]];
constant float4 ENCODING_YUV_TRANSFORM_3 [[ function_constant(ALVRFunctionConstantEncodingYUVTransform3) ]];

float2 TextureToEyeUV(float2 textureUV, bool isRightEye) {
    // flip distortion horizontally for right eye
    // left: x * 2; right: (1 - x) * 2
    return float2((textureUV.x + float(isRightEye) * (1. - 2. * textureUV.x)) * 2., textureUV.y);
}

float2 EyeToTextureUV(float2 eyeUV, bool isRightEye) {
    // left: x / 2; right 1 - (x / 2)
    return float2(eyeUV.x * 0.5 + float(isRightEye) * (1. - eyeUV.x), eyeUV.y);
}

// DECOMPRESS_AXIS_ALIGNED_FRAGMENT_SHADER
float2 decompressAxisAlignedCoord(float2 uv) {
    bool isRightEye = uv.x > 0.5;
    float2 eyeUV = TextureToEyeUV(uv, isRightEye);

    const float2 c0 = (1. - CENTER_SIZE) * 0.5;
    const float2 c1 = (EDGE_RATIO - 1.) * c0 * (CENTER_SHIFT + 1.) / EDGE_RATIO;
    const float2 c2 = (EDGE_RATIO - 1.) * CENTER_SIZE + 1.;

    const float2 loBound = c0 * (CENTER_SHIFT + 1.);
    const float2 hiBound = c0 * (CENTER_SHIFT - 1.) + 1.;
    float2 underBound = float2(eyeUV.x < loBound.x, eyeUV.y < loBound.y);
    float2 inBound = float2(loBound.x < eyeUV.x && eyeUV.x < hiBound.x,
                        loBound.y < eyeUV.y && eyeUV.y < hiBound.y);
    float2 overBound = float2(eyeUV.x > hiBound.x, eyeUV.y > hiBound.y);

    float2 center = (eyeUV - c1) * EDGE_RATIO / c2;

    const float2 loBoundC = c0 * (CENTER_SHIFT + 1.) / c2;
    const float2 hiBoundC = c0 * (CENTER_SHIFT - 1.) / c2 + 1.;

    float2 leftEdge = (-(c1 + c2 * loBoundC) / loBoundC +
                    sqrt(((c1 + c2 * loBoundC) / loBoundC) * ((c1 + c2 * loBoundC) / loBoundC) +
                        4. * c2 * (1. - EDGE_RATIO) / (EDGE_RATIO * loBoundC) * eyeUV)) /
                    (2. * c2 * (1. - EDGE_RATIO)) * (EDGE_RATIO * loBoundC);
    float2 rightEdge =
        (-(c2 - EDGE_RATIO * c1 - 2. * EDGE_RATIO * c2 + c2 * EDGE_RATIO * (1. - hiBoundC) +
        EDGE_RATIO) /
            (EDGE_RATIO * (1. - hiBoundC)) +
        sqrt(((c2 - EDGE_RATIO * c1 - 2. * EDGE_RATIO * c2 + c2 * EDGE_RATIO * (1. - hiBoundC) +
                EDGE_RATIO) /
            (EDGE_RATIO * (1. - hiBoundC))) *
                ((c2 - EDGE_RATIO * c1 - 2. * EDGE_RATIO * c2 +
                    c2 * EDGE_RATIO * (1. - hiBoundC) + EDGE_RATIO) /
                (EDGE_RATIO * (1. - hiBoundC))) -
            4. * ((c2 * EDGE_RATIO - c2) * (c1 - hiBoundC + hiBoundC * c2) /
                        (EDGE_RATIO * (1. - hiBoundC) * (1. - hiBoundC)) -
                    eyeUV * (c2 * EDGE_RATIO - c2) / (EDGE_RATIO * (1. - hiBoundC))))) /
        (2. * c2 * (EDGE_RATIO - 1.)) * (EDGE_RATIO * (1. - hiBoundC));

    // todo: idk why these clamps are necessary
    float2 uncompressedUV = clamp(underBound * leftEdge, float2(0, 0), float2(1, 1)) + clamp(inBound * center, float2(0, 0), float2(1, 1)) + clamp(overBound * rightEdge, float2(0, 0), float2(1, 1));
    return EyeToTextureUV(uncompressedUV * EYE_SIZE_RATIO, isRightEye);
}

// VERTEX_SHADER

ColorInOut videoFrameVertexShaderCommon(ushort vertexID [[vertex_id]],
                               int which,
                               matrix_float4x4 projectionMatrix,
                               matrix_float4x4 modelViewMatrixFrame,
                               simd_float4 tangents)
{
    ColorInOut out;

    float2 uv = float2(float((vertexID << ushort(1)) & 2u) * 0.5, 1.0 - (float(vertexID & ushort(2)) * 0.5));
    float4 position = float4((uv * float2(2.0, -2.0)) + float2(-1.0, 1.0), -500.0, 1.0);

    if (position.x < 1.0) {
        position.x *= tangents[0] * 500.0;
    }
    else {
        position.x *= tangents[1] * 500.0;
    }
    if (position.y < 1.0) {
        position.y *= tangents[3] * 500.0;
    }
    else {
        position.y *= tangents[2] * 500.0;
    }
    out.position = projectionMatrix * modelViewMatrixFrame * position;
    if (which == 0) {
        out.texCoord = float2((uv.x * 0.5), uv.y);
    } else {
        out.texCoord = float2((uv.x * 0.5) + 0.5,  uv.y);
    }

    return out;
}

vertex ColorInOut videoFrameVertexShader(ushort vertexID [[vertex_id]],
                               ushort amp_id [[amplification_id]],
                               constant UniformsArray & uniformsArray [[ buffer(BufferIndexUniforms) ]])
{
    int which = (vertexID >= 4 ? 1 : 0) + amp_id;
    Uniforms uniforms = uniformsArray.uniforms[which];
    
    return videoFrameVertexShaderCommon(vertexID, which, uniforms.projectionMatrix, uniforms.modelViewMatrixFrame, uniforms.tangents);
}

float3 NonlinearToLinearRGB(float3 color) {
    const float DIV12 = 1. / 12.92;
    const float DIV1 = 1. / 1.055;
    const float THRESHOLD = 0.04045;
    const float3 GAMMA = float3(2.4);
        
    float3 condition = float3(color.r < THRESHOLD, color.g < THRESHOLD, color.b < THRESHOLD);
    float3 lowValues = color * DIV12;
    float3 highValues = pow((color + 0.055) * DIV1, GAMMA);
    return condition * lowValues + (1.0 - condition) * highValues;
}

float3 EncodingNonlinearToLinearRGB(float3 color, float gamma) {
    float3 ret;
    ret.r = color.r < 0.0 ? color.r : pow(color.r, gamma);
    ret.g = color.g < 0.0 ? color.g : pow(color.g, gamma);
    ret.b = color.b < 0.0 ? color.b : pow(color.b, gamma);
    return ret;
}

half3 NonlinearToLinearRGB_half(half3 color) {
    const half DIV12 = 1. / 12.92;
    const half DIV1 = 1. / 1.055;
    const half THRESHOLD = 0.04045;
    const half GAMMA = 2.4;
      
    return half3(color.r < THRESHOLD ? (color.r * DIV12) : pow((color.r + 0.055h) * DIV1, GAMMA), color.g < THRESHOLD ? (color.g * DIV12) : pow((color.g + 0.055h) * DIV1, GAMMA), color.b < THRESHOLD ? (color.b * DIV12) : pow((color.b + 0.055h) * DIV1, GAMMA));
}

half3 EncodingNonlinearToLinearRGB_half(half3 color, half gamma) {
    half3 ret;
    ret.r = color.r < 0.0 ? color.r : pow(color.r, gamma);
    ret.g = color.g < 0.0 ? color.g : pow(color.g, gamma);
    ret.b = color.b < 0.0 ? color.b : pow(color.b, gamma);
    return ret;
}

half colorclose_hsv(half3 hsv, half3 keyHsv, half2 tol)
{
    // Saturation or value too low, don't consider it at all.
    if (hsv.b < 0.001 || hsv.g < 0.001) {
        return 1.0;
    }
    half3 weights = half3(4., 1., 2.);
    half tmp = length(weights * (keyHsv - hsv));
    if (tmp < tol.x)
      return 0.0;
   	else if (tmp < tol.y)
      return (tmp - tol.x)/(tol.y - tol.x);
   	else
      return 1.0;
}

half3 rgb2hsv(half3 rgb) {
    half Cmax = max(rgb.r, max(rgb.g, rgb.b));
    half Cmin = min(rgb.r, min(rgb.g, rgb.b));
    half delta = Cmax - Cmin;
    half3 hsv = half3(0., 0., Cmax);

    if(Cmax > Cmin) {
        hsv.y = delta / Cmax;

        if(rgb.r == Cmax) {
            hsv.x = (rgb.g - rgb.b) / delta;
        } else {
            if (rgb.g == Cmax) {
                hsv.x = 2. + (rgb.b - rgb.r) / delta;
            } else {
                hsv.x = 4. + (rgb.r - rgb.g) / delta;
            }
        }
        hsv.x = fract(hsv.x / 6.);
    }
    
    return hsv;
}

half4 videoFrameFragmentShader_common(half3 color_in) {
    half3 color = EncodingNonlinearToLinearRGB_half(color_in, ENCODING_GAMMA);
    
    half4 colorOut = half4(color.rgb, 1.0);
    if (CHROMAKEY_ENABLED) {
        half4 chromaKeyHSV = half4(rgb2hsv(half3(CHROMAKEY_COLOR)), 1.0);
        half4 newHSV = half4(rgb2hsv(color.rgb), 1.0);
        half mask = colorclose_hsv(newHSV.rgb, chromaKeyHSV.rgb, half2(CHROMAKEY_LERP_DIST_RANGE));
        if (!REALITYKIT_ENABLED && mask <= 0.0) {
            discard_fragment();
        }
        
        colorOut = half4((colorOut.rgb * mask) - (half3(CHROMAKEY_COLOR) * (1.0 - mask)), colorOut.a * mask);
        return colorOut;
        //return float4(color.rgb, mask);
    }
    else {
        return colorOut;
    }
    
    // Brighten the scene to examine blocking artifacts/smearing
    //color = pow(color, 1.0 / 2.4);

    /*const float3x3 linearToDisplayP3 = {
        float3(1.2249, -0.0420, -0.0197),
        float3(-0.2247, 1.0419, -0.0786),
        float3(0.0, 0.0, 1.0979),
    };*/
    
    //technically not accurate, since sRGB is below 1.0, but it makes colors pop a bit
    //color = linearToDisplayP3 * color;
}

fragment half4 videoFrameFragmentShader_YpCbCrBiPlanar(ColorInOut in [[stage_in]], texture2d<half> in_tex_y, texture2d<half> in_tex_uv) {
    
    float2 sampleCoord;
    if (FFR_ENABLED) {
        sampleCoord = decompressAxisAlignedCoord(in.texCoord);
    } else {
        sampleCoord = in.texCoord;
    }
    
    constexpr sampler colorSampler(mip_filter::none,
                                   mag_filter::linear,
                                   min_filter::linear);
    half4 ySample = in_tex_y.sample(colorSampler, sampleCoord);
    half4 uvSample = in_tex_uv.sample(colorSampler, sampleCoord);
    half4 ycbcr = half4(ySample.r, uvSample.rg, 1.0f);
    
    const matrix_half4x4 transform = matrix_half4x4(
        half4(ENCODING_YUV_TRANSFORM_0),
        half4(ENCODING_YUV_TRANSFORM_1),
        half4(ENCODING_YUV_TRANSFORM_2),
        half4(ENCODING_YUV_TRANSFORM_3)
    );
    
    half3 rgb_uncorrect = half3((transform * ycbcr).rgb);
    half3 color = NonlinearToLinearRGB_half(rgb_uncorrect);
    
    return videoFrameFragmentShader_common(color);
}

fragment half4 videoFrameFragmentShader_SecretYpCbCrFormats(ColorInOut in [[stage_in]], texture2d<half> in_tex_y) {
    
    float2 sampleCoord;
    if (FFR_ENABLED) {
        sampleCoord = decompressAxisAlignedCoord(in.texCoord);
    } else {
        sampleCoord = in.texCoord;
    }
    
    constexpr sampler colorSampler(mip_filter::none,
                                   mag_filter::linear,
                                   min_filter::linear);
    
    half3 color = in_tex_y.sample(colorSampler, sampleCoord).rgb;
    
    return videoFrameFragmentShader_common(color);
}

fragment float4 videoFrameDepthFragmentShader(ColorInOut in [[stage_in]], texture2d<float> in_tex_y, texture2d<float> in_tex_uv) {
    return float4(0.0, 0.0, 0.0, 0.0);
}

struct CopyVertexOut {
    float4 position [[position]];
    float2 uv;
};

vertex CopyVertexOut copyVertexShader(constant VertexPosOnly* inArr [[buffer(BufferIndexMeshPositions)]], uint vertexID [[vertex_id]], constant rasterization_rate_map_data &vrr [[buffer(BufferIndexVRR)]]) {
    CopyVertexOut out;
    rasterization_rate_map_decoder map(vrr);
    
    // generate vertices in-shader, not sure if fast or slow tbh
    /*const uint gridSize = 128;

    ushort x = (vertexID >> 1) % gridSize;
    ushort y = (vertexID & 1) + (((vertexID >> 1) / gridSize) % (gridSize-1));
    ushort idx = (vertexID >= (gridSize-1)*gridSize*2) ? 1 : 0;
    float otherEye = idx ? 1.0 : 0.0;

    // Normalize coordinates to the range [0, 1]
    float2 uv = float2(
        (float(x) / float(gridSize - 1)),
        (float(y) / float(gridSize - 1))
    );

    // Normalize coordinates to the range [-1, 1]ish
    out.position = float4((uv * float2(2.0, -1.0)) + float2(-1.0, otherEye), otherEye, 1.0);
    out.uv = map.map_screen_to_physical_coordinates(uv * VRR_SCREEN_SIZE, idx);*/
    
    float2 uv = inArr[vertexID].position.xy;
    float otherEye = inArr[vertexID].position.z;
    ushort idx = (otherEye != 0.0) ? 1.0 : 0.0;
    out.position = float4((uv * float2(2.0, -1.0)) + float2(-1.0, otherEye), otherEye, 1.0);
    out.uv = map.map_screen_to_physical_coordinates(uv * VRR_SCREEN_SIZE, idx);
    
    return out;
}

//constant rasterization_rate_map_data &vrr [[buffer(BufferIndexVRR)]]
fragment half4 copyFragmentShader(CopyVertexOut in [[stage_in]], texture2d_array<half> in_tex) {
    constexpr sampler colorSampler(coord::pixel,
                    address::clamp_to_edge,
                    filter::linear);
    ushort idx = in.position.z != 0.0 ? 1 : 0;

    half4 color = in_tex.sample(colorSampler, in.uv, idx);
    //half4 color = half4(in.uv.x, in.uv.y, 0.0, 1.0);

    return color;
}

// MARK: - Radial Menu Overlay Shaders

struct RadialMenuVertexOut {
    float4 position [[position]];
    float2 localPos;      // Position relative to menu center (-1 to 1)
    float segmentAngle;   // Angle of segment center
    int segmentIndex;     // Which segment this vertex belongs to
    float distanceFromCenter;  // Normalized distance from center (0-1)
    uint renderTargetIndex [[render_target_array_index]];
};

// Generates radial menu geometry procedurally
// Each segment is rendered as a pie slice using instancing
vertex RadialMenuVertexOut radialMenuVertexShader(
    uint vertexID [[vertex_id]],
    uint instanceID [[instance_id]],
    ushort amp_id [[amplification_id]],
    constant RadialMenuUniforms& menuUniforms [[buffer(0)]],
    constant UniformsArray& viewUniforms [[buffer(BufferIndexUniforms)]]
) {
    RadialMenuVertexOut out;
    Uniforms uniforms = viewUniforms.uniforms[amp_id];

    int segmentIndex = instanceID;
    int segmentCount = menuUniforms.segmentCount;

    // Each segment is a triangle fan with vertices:
    // 0 = center, 1-N = outer edge points
    // We'll generate smooth arcs with 16 triangles per segment
    const int TRIS_PER_SEGMENT = 16;
    const int VERTS_PER_SEGMENT = TRIS_PER_SEGMENT * 3;

    int triIndex = vertexID / 3;
    int vertexInTri = vertexID % 3;

    float segmentAngle = 2.0 * M_PI_F / float(segmentCount);
    float startAngle = float(segmentIndex) * segmentAngle - M_PI_F / 2.0;

    // Calculate angle for this vertex within the segment
    float triStartAngle = startAngle + (float(triIndex) / float(TRIS_PER_SEGMENT)) * segmentAngle;
    float triEndAngle = startAngle + (float(triIndex + 1) / float(TRIS_PER_SEGMENT)) * segmentAngle;

    float3 localPos;
    float2 normPos;  // For fragment shader

    if (vertexInTri == 0) {
        // Center vertex (at inner radius for donut shape)
        float midAngle = (triStartAngle + triEndAngle) / 2.0;
        localPos = float3(
            cos(midAngle) * menuUniforms.innerRadius,
            0,
            sin(midAngle) * menuUniforms.innerRadius
        );
        normPos = float2(cos(midAngle), sin(midAngle)) * (menuUniforms.innerRadius / menuUniforms.radius);
    } else if (vertexInTri == 1) {
        // First outer edge vertex
        localPos = float3(
            cos(triStartAngle) * menuUniforms.radius,
            0,
            sin(triStartAngle) * menuUniforms.radius
        );
        normPos = float2(cos(triStartAngle), sin(triStartAngle));
    } else {
        // Second outer edge vertex
        localPos = float3(
            cos(triEndAngle) * menuUniforms.radius,
            0,
            sin(triEndAngle) * menuUniforms.radius
        );
        normPos = float2(cos(triEndAngle), sin(triEndAngle));
    }

    // Transform to world space
    float4 worldPos = menuUniforms.modelMatrix * float4(localPos, 1.0);

    // Transform to clip space
    out.position = uniforms.projectionMatrix * uniforms.modelViewMatrix * worldPos;
    out.localPos = normPos;
    out.segmentAngle = startAngle + segmentAngle / 2.0;
    out.segmentIndex = segmentIndex;
    out.distanceFromCenter = length(normPos);
    out.renderTargetIndex = amp_id;

    return out;
}

fragment float4 radialMenuFragmentShader(
    RadialMenuVertexOut in [[stage_in]],
    constant RadialMenuUniforms& menuUniforms [[buffer(0)]]
) {
    int segmentIndex = in.segmentIndex;

    // Get base color for this segment
    float4 baseColor = menuUniforms.segmentColors[segmentIndex % 8];

    // Highlight hovered segment
    if (segmentIndex == menuUniforms.hoveredSegment) {
        baseColor = mix(baseColor, float4(1.0, 1.0, 1.0, 1.0), 0.4);
    }

    // Dim the selected/current segment slightly differently
    if (segmentIndex == menuUniforms.selectedSegment) {
        baseColor = mix(baseColor, float4(0.8, 0.8, 0.8, 1.0), 0.2);
    }

    // Calculate segment borders for visual separation
    float segmentAngle = 2.0 * M_PI_F / float(menuUniforms.segmentCount);
    float angle = atan2(in.localPos.y, in.localPos.x) + M_PI_F / 2.0;  // Offset to match vertex shader
    if (angle < 0) angle += 2.0 * M_PI_F;

    float angleInSegment = fmod(angle, segmentAngle);
    float edgeDist = min(angleInSegment, segmentAngle - angleInSegment);

    // Add subtle border between segments
    float borderWidth = 0.03;
    if (edgeDist < borderWidth) {
        float borderAlpha = 1.0 - (edgeDist / borderWidth);
        baseColor = mix(baseColor, float4(1.0, 1.0, 1.0, 0.8), borderAlpha * 0.5);
    }

    // Soft outer edge
    float outerSoftness = smoothstep(1.0, 0.95, in.distanceFromCenter);

    // Soft inner edge
    float innerRatio = menuUniforms.innerRadius / menuUniforms.radius;
    float innerSoftness = smoothstep(innerRatio, innerRatio + 0.05, in.distanceFromCenter);

    // Apply edge softness
    baseColor.a *= outerSoftness * innerSoftness;

    // Apply animation progress (fade in/out)
    baseColor.a *= menuUniforms.animationProgress;

    // Discard fully transparent pixels
    if (baseColor.a < 0.01) {
        discard_fragment();
    }

    return baseColor;
}

// Center indicator shader - shows current selection in the middle
vertex RadialMenuVertexOut radialMenuCenterVertexShader(
    uint vertexID [[vertex_id]],
    ushort amp_id [[amplification_id]],
    constant RadialMenuUniforms& menuUniforms [[buffer(0)]],
    constant UniformsArray& viewUniforms [[buffer(BufferIndexUniforms)]]
) {
    RadialMenuVertexOut out;
    Uniforms uniforms = viewUniforms.uniforms[amp_id];

    // Generate a small circle in the center
    const int NUM_SEGMENTS = 32;
    int triIndex = vertexID / 3;
    int vertexInTri = vertexID % 3;

    float angle1 = float(triIndex) / float(NUM_SEGMENTS) * 2.0 * M_PI_F;
    float angle2 = float(triIndex + 1) / float(NUM_SEGMENTS) * 2.0 * M_PI_F;

    float centerRadius = menuUniforms.innerRadius * 0.7;
    float3 localPos;

    if (vertexInTri == 0) {
        localPos = float3(0, 0, 0);
    } else if (vertexInTri == 1) {
        localPos = float3(cos(angle1) * centerRadius, 0, sin(angle1) * centerRadius);
    } else {
        localPos = float3(cos(angle2) * centerRadius, 0, sin(angle2) * centerRadius);
    }

    float4 worldPos = menuUniforms.modelMatrix * float4(localPos, 1.0);
    out.position = uniforms.projectionMatrix * uniforms.modelViewMatrix * worldPos;
    out.localPos = float2(localPos.x, localPos.z) / centerRadius;
    out.segmentAngle = 0;
    out.segmentIndex = -1;
    out.distanceFromCenter = length(out.localPos);
    out.renderTargetIndex = amp_id;

    return out;
}

fragment float4 radialMenuCenterFragmentShader(
    RadialMenuVertexOut in [[stage_in]],
    constant RadialMenuUniforms& menuUniforms [[buffer(0)]]
) {
    // Dark semi-transparent center with soft edges
    float4 color = float4(0.1, 0.1, 0.1, 0.7);

    // Soft edge
    float softness = smoothstep(1.0, 0.8, in.distanceFromCenter);
    color.a *= softness;

    // Apply animation
    color.a *= menuUniforms.animationProgress;

    if (color.a < 0.01) {
        discard_fragment();
    }

    return color;
}
