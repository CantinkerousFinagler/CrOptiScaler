#pragma once

// The foveal crop's passes, ONE SOURCE for D3D12 and Vulkan (as PerEyeFitShader_Dx12.h): PerEyeDlss_Dx12 compiles these
// literals at runtime with fxc (cs_5_0, no VK_MODE: the #ifdef VK_MODE blocks fall away), and
// tools/xr_observer/test/vk_fovea_shaders.py compiles them with the fork's dxc -spirv -D VK_MODE into FoveaShader_Vk.h
// (FoveaPasses_Vk). The Vulkan bindings: set 0, the constants as
// push constants, the storage images with an explicit rgba16f format (the output format DLSS writes in the Vulkan game it was written for:
// R16G16B16A16_SFLOAT; FoveaCrop_Vk checks the game's output format against it before the crop is allowed).
// Moved here out of PerEyeDlss_Dx12.cpp; the HLSL the D3D12 path compiles is unchanged.

// COMPOSITE: our crop texture (ow x oh) blended over the output's box at (Base) with a feathered edge; Outline = F8.
constexpr char kCompositeShader[] = R"(
#ifdef VK_MODE
[[vk::binding(0, 0)]] Texture2D<float4> Crop;
[[vk::binding(1, 0)]] [[vk::image_format("rgba16f")]] RWTexture2D<float4> Output;
struct ConstantsVk
{
    uint2 Base;
    uint2 Size;
    uint Feather;
    uint Outline;
    uint AgeFrames; // D3D12 only (AGE_FADE); the Vulkan path hands 0
    uint Unused;
};
[[vk::push_constant]] ConstantsVk PC;
#define Base PC.Base
#define Size PC.Size
#define Feather PC.Feather
#define Outline PC.Outline
#else
Texture2D<float4> Crop : register(t0);
#ifdef AGE_FADE
Texture2D<uint> Age : register(t1); // frames each crop pixel has been in the box (PerEyeDlss_Dx12's age pass)
#endif
RWTexture2D<float4> Output : register(u0);

cbuffer Constants : register(b0)
{
    uint2 Base;
    uint2 Size;
    uint Feather;
    uint Outline;
    uint AgeFrames; // with AGE_FADE: a pixel younger than this shows the output under it, in proportion; 0 = off
    uint Unused;
};
#endif

[numthreads(8, 8, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= Size.x || id.y >= Size.y)
        return;
    const uint2 at = Base + id.xy;
    const float d = (float) min(min(id.x, id.y), min(Size.x - 1 - id.x, Size.y - 1 - id.y));
    float4 c = Crop[id.xy];
    if (Outline != 0 && d < 3.0)
    {
        Output[at] = float4(4.0, 0.0, 4.0, 1.0);
        return;
    }
#ifdef AGE_FADE
    // The picture that came into the box at a move has no history in the crop's DLSS yet and takes frames to
    // settle; the periphery under it (its own history over the whole eye) is settled all along: show that through
    // until the pixel has been in the box AgeFrames frames (otherwise the image visibly settles when
    // the crop moves from one far side to the other).
    if (AgeFrames > 0)
        c = lerp(Output[at], c, saturate((float) Age[id.xy] / (float) AgeFrames));
#endif
    if (Feather > 0 && d < (float) Feather)
    {
        const float w = smoothstep(0.0, 1.0, (d + 0.5) / (float) Feather);
        c = lerp(Output[at], c, w);
    }
    Output[at] = c;
}
)";

// AGE (D3D12 only, PerEyeDlss_Dx12::UpdateAge; fit12_test case 16): how many frames each crop pixel has been in the
// box, carried along with the box's move in OUTPUT pixels (a pixel at box coordinate p sat at p + Shift in the old
// box: Shift = new origin - old origin), 0 where it came in from outside, 0 everywhere on a reset; saturates at 255.
constexpr char kAgeShader[] = R"(
Texture2D<uint> Age : register(t0);
RWTexture2D<uint> AgeOut : register(u0);
cbuffer Constants : register(b0)
{
    uint2 Size;
    int2 Shift;
    uint Reset;
    uint3 Unused;
};
[numthreads(8, 8, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= Size.x || id.y >= Size.y)
        return;
    uint age = 0;
    if (Reset == 0)
    {
        const int2 from = int2(id.xy) + Shift;
        if (from.x >= 0 && from.y >= 0 && from.x < (int) Size.x && from.y < (int) Size.y)
            age = min(Age[uint2(from)] + 1u, 255u);
    }
    AgeOut[id.xy] = age;
}
)";

// ENLARGE: the cheap periphery's render-size result over the whole eye, 9-tap Catmull-Rom through bilinear taps.
constexpr char kEnlargeShader[] = R"(
#ifdef VK_MODE
[[vk::binding(0, 0)]] Texture2D<float4> Src;
[[vk::binding(1, 0)]] SamplerState Linear;
[[vk::binding(2, 0)]] [[vk::image_format("rgba16f")]] RWTexture2D<float4> Output;
struct ConstantsVk
{
    uint2 OutSize;
    uint2 SrcSize;
    uint4 Unused;
};
[[vk::push_constant]] ConstantsVk PC;
#define OutSize PC.OutSize
#define SrcSize PC.SrcSize
#else
Texture2D<float4> Src : register(t0);
RWTexture2D<float4> Output : register(u0);
SamplerState Linear : register(s0);

cbuffer Constants : register(b0)
{
    uint2 OutSize;
    uint2 SrcSize;
    uint4 Unused;
};
#endif

// 9-tap Catmull-Rom through bilinear taps.
float4 CatmullRom(float2 uv, float2 size)
{
    const float2 samplePos = uv * size;
    const float2 texPos1 = floor(samplePos - 0.5) + 0.5;
    const float2 f = samplePos - texPos1;
    const float2 w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
    const float2 w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
    const float2 w2 = f * (0.5 + f * (2.0 - 1.5 * f));
    const float2 w3 = f * f * (-0.5 + 0.5 * f);
    const float2 w12 = w1 + w2;
    const float2 offset12 = w2 / w12;
    const float2 p0 = (texPos1 - 1.0) / size;
    const float2 p3 = (texPos1 + 2.0) / size;
    const float2 p12 = (texPos1 + offset12) / size;
    float4 r = 0;
    r += Src.SampleLevel(Linear, float2(p0.x, p0.y), 0) * w0.x * w0.y;
    r += Src.SampleLevel(Linear, float2(p12.x, p0.y), 0) * w12.x * w0.y;
    r += Src.SampleLevel(Linear, float2(p3.x, p0.y), 0) * w3.x * w0.y;
    r += Src.SampleLevel(Linear, float2(p0.x, p12.y), 0) * w0.x * w12.y;
    r += Src.SampleLevel(Linear, float2(p12.x, p12.y), 0) * w12.x * w12.y;
    r += Src.SampleLevel(Linear, float2(p3.x, p12.y), 0) * w3.x * w12.y;
    r += Src.SampleLevel(Linear, float2(p0.x, p3.y), 0) * w0.x * w3.y;
    r += Src.SampleLevel(Linear, float2(p12.x, p3.y), 0) * w12.x * w3.y;
    r += Src.SampleLevel(Linear, float2(p3.x, p3.y), 0) * w3.x * w3.y;
    return max(r, 0.0); // HDR: no negative ringing around highlights
}

[numthreads(8, 8, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= OutSize.x || id.y >= OutSize.y)
        return;
    const float2 uv = (float2(id.xy) + 0.5) / float2(OutSize);
    Output[id.xy] = CatmullRom(uv, float2(SrcSize));
}
)";

// TAA: the own TAA periphery (PerEyeDlss_Dx12 TaaPeriphery, FoveaCrop_Vk on Vulkan): one pass at the output size over the
// whole eye. Per output pixel: the current frame reconstructed from the jittered render samples (4x4 Catmull-Rom), this
// eye's history reprojected with the motion DLSS gets, dilated to the nearest depth of the 3x3, clamped to the 3x3's
// colour range (mean +- 1.25 sigma in YCoCg) and blended, all in a reversible tonemapped space (the history keeps it).
// Two history textures per eye, read one, write the other. The literal is the one PerEyeDlss_Dx12.cpp used to
// compile; the VK_MODE block is the only addition.
constexpr char kTaaShader[] = R"(
#ifdef VK_MODE
[[vk::binding(0, 0)]] Texture2D<float4> Colour;
[[vk::binding(1, 0)]] Texture2D<float2> Motion;
[[vk::binding(2, 0)]] Texture2D<float> Depth;
[[vk::binding(3, 0)]] Texture2D<float4> History;
[[vk::binding(4, 0)]] SamplerState Linear;
[[vk::binding(5, 0)]] [[vk::image_format("rgba16f")]] RWTexture2D<float4> Output;
[[vk::binding(6, 0)]] [[vk::image_format("rgba16f")]] RWTexture2D<float4> HistoryOut;
struct ConstantsVk
{
    uint2 OutSize;
    uint2 RenderSize;
    uint2 ColourBase;
    uint2 MotionBase;
    uint2 DepthBase;
    float2 Jitter;
    float2 MvScale;
    float Blend;
    uint Reset;
};
[[vk::push_constant]] ConstantsVk PC;
#define OutSize PC.OutSize
#define RenderSize PC.RenderSize
#define ColourBase PC.ColourBase
#define MotionBase PC.MotionBase
#define DepthBase PC.DepthBase
#define Jitter PC.Jitter
#define MvScale PC.MvScale
#define Blend PC.Blend
#define Reset PC.Reset
#else
Texture2D<float4> Colour : register(t0);
Texture2D<float2> Motion : register(t1);
Texture2D<float> Depth : register(t2);
Texture2D<float4> History : register(t3);
RWTexture2D<float4> Output : register(u0);
RWTexture2D<float4> HistoryOut : register(u1);
SamplerState Linear : register(s0);

cbuffer Constants : register(b0)
{
    uint2 OutSize;
    uint2 RenderSize;
    uint2 ColourBase;
    uint2 MotionBase;
    uint2 DepthBase;
    float2 Jitter; // render px, sign already applied
    float2 MvScale;
    float Blend;   // weight of the current frame
    uint Reset;
};
#endif

bool Finite(float v) { return (asuint(v) & 0x7F800000u) != 0x7F800000u; }
float Luma(float3 c) { return dot(c, float3(0.299, 0.587, 0.114)); }
float3 Tm(float3 c) { return c / (1.0 + Luma(c)); }
float3 Itm(float3 c) { return c / max(1.0 - Luma(c), 1e-3); }
float3 ToYCoCg(float3 c) { return float3(0.25 * c.r + 0.5 * c.g + 0.25 * c.b, 0.5 * c.r - 0.5 * c.b, -0.25 * c.r + 0.5 * c.g - 0.25 * c.b); }
float3 FromYCoCg(float3 c) { return float3(c.x + c.y - c.z, c.x + c.z, c.x - c.y - c.z); }

void CrWeights(float f, out float w[4])
{
    w[0] = f * (-0.5 + f * (1.0 - 0.5 * f));
    w[1] = 1.0 + f * f * (-2.5 + 1.5 * f);
    w[2] = f * (0.5 + f * (2.0 - 1.5 * f));
    w[3] = f * f * (-0.5 + 0.5 * f);
}

// 9-tap Catmull-Rom through bilinear taps (the history is ours, at the output size).
float3 SampleHistory(float2 uv, float2 size)
{
    const float2 samplePos = uv * size;
    const float2 texPos1 = floor(samplePos - 0.5) + 0.5;
    const float2 f = samplePos - texPos1;
    const float2 w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
    const float2 w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
    const float2 w2 = f * (0.5 + f * (2.0 - 1.5 * f));
    const float2 w3 = f * f * (-0.5 + 0.5 * f);
    const float2 w12 = w1 + w2;
    const float2 p0 = (texPos1 - 1.0) / size;
    const float2 p3 = (texPos1 + 2.0) / size;
    const float2 p12 = (texPos1 + w2 / w12) / size;
    float3 r = 0;
    r += History.SampleLevel(Linear, float2(p0.x, p0.y), 0).rgb * w0.x * w0.y;
    r += History.SampleLevel(Linear, float2(p12.x, p0.y), 0).rgb * w12.x * w0.y;
    r += History.SampleLevel(Linear, float2(p3.x, p0.y), 0).rgb * w3.x * w0.y;
    r += History.SampleLevel(Linear, float2(p0.x, p12.y), 0).rgb * w0.x * w12.y;
    r += History.SampleLevel(Linear, float2(p12.x, p12.y), 0).rgb * w12.x * w12.y;
    r += History.SampleLevel(Linear, float2(p3.x, p12.y), 0).rgb * w3.x * w12.y;
    r += History.SampleLevel(Linear, float2(p0.x, p3.y), 0).rgb * w0.x * w3.y;
    r += History.SampleLevel(Linear, float2(p12.x, p3.y), 0).rgb * w12.x * w3.y;
    r += History.SampleLevel(Linear, float2(p3.x, p3.y), 0).rgb * w3.x * w3.y;
    return max(r, 0.0);
}

[numthreads(8, 8, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= OutSize.x || id.y >= OutSize.y)
        return;
    const float2 R = float2(RenderSize);
    const int2 maxP = int2(RenderSize) - 1;
    const float2 uvOut = (float2(id.xy) + 0.5) / float2(OutSize);
    // This output pixel's place on the jittered render grid (texel centres at i + 0.5).
    const float2 t = uvOut * R + Jitter - 0.5;
    const int2 i0 = int2(floor(t));
    const float2 f = t - floor(t);
    const int2 nearest = i0 + int2(f.x >= 0.5 ? 1 : 0, f.y >= 0.5 ? 1 : 0);
    float wx[4], wy[4];
    CrWeights(f.x, wx);
    CrWeights(f.y, wy);

    float3 cur = 0;
    float3 m1 = 0, m2 = 0;
    float zBest = -1.0;
    int2 pBest = clamp(nearest, int2(0, 0), maxP);
    [unroll] for (int j = 0; j < 4; ++j)
    {
        [unroll] for (int k = 0; k < 4; ++k)
        {
            const int2 p = clamp(i0 + int2(k - 1, j - 1), int2(0, 0), maxP);
            float3 c = Colour[ColourBase + p].rgb;
            if (!Finite(c.r) || !Finite(c.g) || !Finite(c.b))
                c = 0;
            const float3 ct = Tm(max(c, 0.0));
            cur += ct * (wx[k] * wy[j]);
            const int2 d = (i0 + int2(k - 1, j - 1)) - nearest;
            if (abs(d.x) <= 1 && abs(d.y) <= 1)
            {
                const float3 y = ToYCoCg(ct);
                m1 += y;
                m2 += y * y;
                const float z = Depth[DepthBase + p];
                if (Finite(z) && z > zBest)
                {
                    zBest = z;
                    pBest = p;
                }
            }
        }
    }
    cur = max(cur, 0.0);

    float2 mv = Motion[MotionBase + pBest] * MvScale;
    if (!Finite(mv.x) || !Finite(mv.y))
        mv = 0;
    const float2 prevUv = (uvOut * R + mv) / R;
    const bool offscreen = any(prevUv < 0.0) || any(prevUv > 1.0);

    float3 result = cur;
    if (Reset == 0 && !offscreen)
    {
        const float3 mean = m1 / 9.0;
        const float3 sigma = sqrt(max(m2 / 9.0 - mean * mean, 0.0));
        float3 hist = ToYCoCg(SampleHistory(prevUv, float2(OutSize)));
        hist = clamp(hist, mean - 1.25 * sigma, mean + 1.25 * sigma);
        result = lerp(FromYCoCg(hist), cur, Blend);
    }
    result = max(result, 0.0);
    HistoryOut[id.xy] = float4(result, 1.0);
    Output[id.xy] = float4(Itm(result), 1.0);
}
)";
