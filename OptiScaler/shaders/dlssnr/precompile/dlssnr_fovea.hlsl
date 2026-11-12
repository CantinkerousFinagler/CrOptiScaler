#ifdef VK_MODE
[[vk::binding(0, 0)]]
cbuffer Params : register(b0, space0)
#else
cbuffer Params : register(b0)
#endif
{
    uint RoiX;
    uint RoiY;
    uint RoiW;
    uint RoiH;
    float Roundness;
    float Feather;
    uint ShowBorder;
    uint Pad;
};

#ifdef VK_MODE
[[vk::binding(1, 0)]] Texture2D<float4> Original : register(t0, space0);
[[vk::binding(2, 0)]] Texture2D<float4> Edited : register(t1, space0);
[[vk::binding(3, 0)]] RWTexture2D<float4> Dest : register(u0, space0);
#else
Texture2D<float4> Original : register(t0);
Texture2D<float4> Edited : register(t1);
RWTexture2D<float4> Dest : register(u0);
#endif

float Shape(uint2 dest)
{
    float2 centered = (float2(dest) - float2(RoiX, RoiY) + 0.5) /
        (0.5 * max(float2(RoiW, RoiH), 1.0)) - 1.0;
    float2 scaled = abs(centered);
    return lerp(max(scaled.x, scaled.y), length(scaled), saturate(Roundness));
}

[numthreads(8, 8, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= RoiW || id.y >= RoiH)
        return;
    uint2 dest = uint2(RoiX, RoiY) + id.xy;
    float distance = Shape(dest);
    float weight = Feather <= 0.0
        ? (distance <= 1.0 ? 1.0 : 0.0)
        : 1.0 - smoothstep(max(0.0, 1.0 - saturate(Feather)), 1.0, distance);
    float4 original = Original[id.xy];
    float4 edited = Edited[id.xy];
    float4 color = lerp(original, edited, saturate(weight));
    if (ShowBorder != 0 && distance <= 1.0 && distance >= 1.0 - 4.0 / max((float) RoiW, 1.0))
        color = float4(original.rgb * float3(0.05, 1.0, 0.05) + float3(0.0, 0.25, 0.0), original.a);
    Dest[id.xy] = color;
}
