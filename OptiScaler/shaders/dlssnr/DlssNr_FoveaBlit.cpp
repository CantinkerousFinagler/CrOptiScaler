#include "pch.h"

#include "DlssNr_FoveaBlit.h"

#include <d3dcompiler.h>

#include <algorithm>
#include <cstdint>

#include <Config.h>

namespace DlssNr
{
namespace
{
constexpr char kShader[] = R"(
Texture2D<float4> Edited : register(t0);
Texture2D<float4> Original : register(t1);
RWTexture2D<float4> Dest : register(u0);

cbuffer Params : register(b0)
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
    if (weight <= 0.0)
        return;
    float4 edited = Edited[id.xy];
    float4 original = Original[id.xy];
    float4 color = lerp(original, edited, saturate(weight));
    if (ShowBorder != 0 && distance <= 1.0 && distance >= 1.0 - 4.0 / max((float) RoiW, 1.0))
        color = float4(original.rgb * float3(0.05, 1.0, 0.05) + float3(0.0, 0.25, 0.0), original.a);
    Dest[dest] = color;
}
)";

struct Constants
{
    uint32_t roiX, roiY, roiW, roiH;
    float roundness, feather;
    uint32_t showBorder, pad;
};

ID3D12RootSignature* g_root = nullptr;
ID3D12PipelineState* g_pso = nullptr;
ID3D12DescriptorHeap* g_heap = nullptr;
ID3D12Resource* g_original = nullptr;
D3D12_RESOURCE_STATES g_originalState = D3D12_RESOURCE_STATE_COPY_DEST;
UINT g_originalW = 0;
UINT g_originalH = 0;
DXGI_FORMAT g_originalFormat = DXGI_FORMAT_UNKNOWN;
UINT g_descriptorSize = 0;
UINT g_slot = 0;
constexpr UINT kSlots = 8;
constexpr UINT kDescriptorsPerSlot = 3;

// A typeless texture cannot be viewed through a null view description; the device is removed with
// DXGI_ERROR_INVALID_CALL. FC5's TAA target (and so the D3D11 bridge's shared copy) is typeless.
DXGI_FORMAT TypedColorFormat(DXGI_FORMAT format)
{
    switch (format)
    {
    case DXGI_FORMAT_R32G32B32A32_TYPELESS:
        return DXGI_FORMAT_R32G32B32A32_FLOAT;
    case DXGI_FORMAT_R16G16B16A16_TYPELESS:
        return DXGI_FORMAT_R16G16B16A16_FLOAT;
    case DXGI_FORMAT_R10G10B10A2_TYPELESS:
        return DXGI_FORMAT_R10G10B10A2_UNORM;
    case DXGI_FORMAT_R8G8B8A8_TYPELESS:
        return DXGI_FORMAT_R8G8B8A8_UNORM;
    case DXGI_FORMAT_B8G8R8A8_TYPELESS:
        return DXGI_FORMAT_B8G8R8A8_UNORM;
    default:
        return format;
    }
}

bool IsKnownColorFormat(DXGI_FORMAT format)
{
    switch (TypedColorFormat(format))
    {
    case DXGI_FORMAT_R32G32B32A32_FLOAT:
    case DXGI_FORMAT_R16G16B16A16_FLOAT:
    case DXGI_FORMAT_R11G11B10_FLOAT:
    case DXGI_FORMAT_R10G10B10A2_UNORM:
    case DXGI_FORMAT_R8G8B8A8_UNORM:
    case DXGI_FORMAT_B8G8R8A8_UNORM:
        return true;
    default:
        return false;
    }
}

void Barrier(ID3D12GraphicsCommandList* commands, ID3D12Resource* resource, D3D12_RESOURCE_STATES before,
             D3D12_RESOURCE_STATES after)
{
    if (commands == nullptr || resource == nullptr || before == after)
        return;
    D3D12_RESOURCE_BARRIER barrier {};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = resource;
    barrier.Transition.StateBefore = before;
    barrier.Transition.StateAfter = after;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    commands->ResourceBarrier(1, &barrier);
}

bool EnsureOriginal(ID3D12Device* device, UINT width, UINT height, DXGI_FORMAT format)
{
    if (g_original != nullptr && g_originalW == width && g_originalH == height && g_originalFormat == format)
        return true;
    if (g_original != nullptr)
    {
        g_original->Release();
        g_original = nullptr;
    }
    D3D12_HEAP_PROPERTIES heap {};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC desc {};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = width;
    desc.Height = height;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = format;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    if (FAILED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COPY_DEST,
                                               nullptr, IID_PPV_ARGS(&g_original))))
        return false;
    g_originalW = width;
    g_originalH = height;
    g_originalFormat = format;
    g_originalState = D3D12_RESOURCE_STATE_COPY_DEST;
    return true;
}

bool EnsurePipeline(ID3D12Device* device)
{
    if (g_pso != nullptr)
        return true;

    ID3DBlob* shader = nullptr;
    ID3DBlob* errors = nullptr;
    if (FAILED(D3DCompile(kShader, sizeof(kShader) - 1, "DlssNr_FoveaBlit", nullptr, nullptr, "CSMain", "cs_5_0",
                          D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &shader, &errors)))
    {
        if (errors)
        {
            LOG_ERROR("DLSS-NR fovea blit compile failed: {}", (const char*) errors->GetBufferPointer());
            errors->Release();
        }
        return false;
    }
    if (errors)
        errors->Release();

    D3D12_DESCRIPTOR_RANGE ranges[2] {};
    ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    ranges[0].NumDescriptors = 2;
    ranges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    ranges[1].NumDescriptors = 1;
    ranges[1].OffsetInDescriptorsFromTableStart = 2;
    D3D12_ROOT_PARAMETER params[2] {};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[0].DescriptorTable.NumDescriptorRanges = 2;
    params[0].DescriptorTable.pDescriptorRanges = ranges;
    params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[1].Constants.Num32BitValues = 8;
    params[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    D3D12_ROOT_SIGNATURE_DESC rootDesc {};
    rootDesc.NumParameters = 2;
    rootDesc.pParameters = params;
    ID3DBlob* serialized = nullptr;
    if (FAILED(D3D12SerializeRootSignature(&rootDesc, D3D_ROOT_SIGNATURE_VERSION_1, &serialized, nullptr)))
    {
        shader->Release();
        return false;
    }
    if (FAILED(device->CreateRootSignature(0, serialized->GetBufferPointer(), serialized->GetBufferSize(),
                                           IID_PPV_ARGS(&g_root))))
    {
        serialized->Release();
        shader->Release();
        return false;
    }
    serialized->Release();
    D3D12_COMPUTE_PIPELINE_STATE_DESC pso {};
    pso.pRootSignature = g_root;
    pso.CS = { shader->GetBufferPointer(), shader->GetBufferSize() };
    const auto created = device->CreateComputePipelineState(&pso, IID_PPV_ARGS(&g_pso));
    shader->Release();
    if (FAILED(created))
        return false;

    D3D12_DESCRIPTOR_HEAP_DESC heap {};
    heap.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    heap.NumDescriptors = kSlots * kDescriptorsPerSlot;
    heap.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    if (FAILED(device->CreateDescriptorHeap(&heap, IID_PPV_ARGS(&g_heap))))
        return false;
    g_descriptorSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    LOG_INFO("DLSS-NR fovea blit compiled (ROI original SRV, write-only dest UAV)");
    return true;
}
} // namespace

bool BlitFovea(ID3D12Device* device, ID3D12GraphicsCommandList* commands, ID3D12Resource* dest,
               ID3D12Resource* edited, const FoveaBlitParams& params)
{
    if (device == nullptr || commands == nullptr || dest == nullptr || edited == nullptr)
        return false;
    if (params.roi.width == 0 || params.roi.height == 0)
        return false;
    if ((dest->GetDesc().Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS) == 0)
        return false;
    if (!EnsurePipeline(device))
        return false;

    const auto destDesc = dest->GetDesc();
    const auto editedFormat = edited->GetDesc().Format;
    // Anything we cannot name a view format for falls back to the caller's rectangle copy.
    if (!IsKnownColorFormat(destDesc.Format) || !IsKnownColorFormat(editedFormat))
    {
        static bool logged = false;
        if (!logged)
        {
            logged = true;
            LOG_WARN("DLSS-NR fovea blit: unsupported formats (dest {}, edited {}), using the rectangle copy",
                     (int) destDesc.Format, (int) editedFormat);
        }
        return false;
    }
    const auto destFormat = TypedColorFormat(destDesc.Format);
    if (!EnsureOriginal(device, params.roi.width, params.roi.height, destFormat))
        return false;

    Barrier(commands, dest, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
    Barrier(commands, g_original, g_originalState, D3D12_RESOURCE_STATE_COPY_DEST);
    g_originalState = D3D12_RESOURCE_STATE_COPY_DEST;

    D3D12_TEXTURE_COPY_LOCATION dstLoc {};
    dstLoc.pResource = g_original;
    dstLoc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    D3D12_TEXTURE_COPY_LOCATION srcLoc {};
    srcLoc.pResource = dest;
    srcLoc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    D3D12_BOX box {};
    box.left = params.roi.x;
    box.top = params.roi.y;
    box.front = 0;
    box.right = params.roi.x + params.roi.width;
    box.bottom = params.roi.y + params.roi.height;
    box.back = 1;
    commands->CopyTextureRegion(&dstLoc, 0, 0, 0, &srcLoc, &box);

    Barrier(commands, dest, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    Barrier(commands, g_original, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    g_originalState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

    const UINT slot = g_slot++ % kSlots;
    auto cpu = g_heap->GetCPUDescriptorHandleForHeapStart();
    cpu.ptr += (SIZE_T) slot * kDescriptorsPerSlot * g_descriptorSize;
    D3D12_SHADER_RESOURCE_VIEW_DESC srv {};
    srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srv.Texture2D.MipLevels = 1;
    srv.Format = TypedColorFormat(editedFormat);
    device->CreateShaderResourceView(edited, &srv, cpu);
    cpu.ptr += g_descriptorSize;
    srv.Format = destFormat;
    device->CreateShaderResourceView(g_original, &srv, cpu);
    cpu.ptr += g_descriptorSize;
    D3D12_UNORDERED_ACCESS_VIEW_DESC uav {};
    uav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    uav.Format = destFormat;
    device->CreateUnorderedAccessView(dest, nullptr, &uav, cpu);
    auto gpu = g_heap->GetGPUDescriptorHandleForHeapStart();
    gpu.ptr += (UINT64) slot * kDescriptorsPerSlot * g_descriptorSize;

    Constants constants { params.roi.x, params.roi.y, params.roi.width, params.roi.height,
                          std::clamp(params.roundness, 0.0f, 1.0f), std::clamp(params.transition, 0.0f, 0.5f),
                          params.showBorder ? 1u : 0u, 0u };

    ID3D12DescriptorHeap* heaps[] { g_heap };
    commands->SetDescriptorHeaps(1, heaps);
    commands->SetComputeRootSignature(g_root);
    commands->SetPipelineState(g_pso);
    commands->SetComputeRootDescriptorTable(0, gpu);
    commands->SetComputeRoot32BitConstants(1, 8, &constants, 0);
    commands->Dispatch((params.roi.width + 7) / 8, (params.roi.height + 7) / 8, 1);
    return true;
}

void ReleaseFoveaBlit()
{
    g_slot = 0;
    g_originalW = 0;
    g_originalH = 0;
    g_originalFormat = DXGI_FORMAT_UNKNOWN;
    g_originalState = D3D12_RESOURCE_STATE_COPY_DEST;
    if (g_original)
    {
        g_original->Release();
        g_original = nullptr;
    }
    if (g_heap)
    {
        g_heap->Release();
        g_heap = nullptr;
    }
    if (g_pso)
    {
        g_pso->Release();
        g_pso = nullptr;
    }
    if (g_root)
    {
        g_root->Release();
        g_root = nullptr;
    }
}
} // namespace DlssNr
