#pragma once

// The headset overlay's shared textures, DIRECT3D 12 twin of OverlayTextures.h (for D3D12 games).
// Header only and free of everything but Direct3D, so that tools/xr_observer/test/overlay_test makes them with THIS code
// on a real device and opens them on a D3D11 device as the layer does (ID3D11Device1::OpenSharedResource1 and
// ID3D11Device5::OpenSharedFence take D3D12's NT handles). The texture is R8G8B8A8_TYPELESS on a SHARED heap with a
// render target view in the format ImGui's D3D12 pipeline was made for (the game window's: UNORM or UNORM_SRGB); it is
// left in the COMMON state between uses, which cross-API sharing asks for.
#include <d3d12.h>

namespace OverlayTextures12
{
// The texture's TYPELESS family for the format ImGui's pipeline draws in (the game window's swapchain format): the
// layer copies the texture into a panel swapchain of the same family (some games' windows are not RGBA8).
// Three families: RGBA8, BGRA8, and 10-bit RGB with a 2-bit alpha (the panel's edges then have four alpha steps).
inline DXGI_FORMAT TypelessFor(DXGI_FORMAT f)
{
    switch (f)
    {
    case DXGI_FORMAT_R8G8B8A8_UNORM:
    case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
        return DXGI_FORMAT_R8G8B8A8_TYPELESS;
    case DXGI_FORMAT_B8G8R8A8_UNORM:
    case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
        return DXGI_FORMAT_B8G8R8A8_TYPELESS;
    case DXGI_FORMAT_R10G10B10A2_UNORM:
        return DXGI_FORMAT_R10G10B10A2_TYPELESS;
    default:
        return DXGI_FORMAT_UNKNOWN;
    }
}
inline bool RtvFormatOk(DXGI_FORMAT f) { return TypelessFor(f) != DXGI_FORMAT_UNKNOWN; }

// One width x height texture with its render target view written at `rtv`, and its NT shared handle. On failure
// nothing is left behind; *hr says why.
inline bool MakeShared(ID3D12Device* device, UINT width, UINT height, DXGI_FORMAT rtvFormat,
                       D3D12_CPU_DESCRIPTOR_HANDLE rtv, ID3D12Resource** texture, HANDLE* handle, HRESULT* hr)
{
    *texture = nullptr;
    *handle = nullptr;
    D3D12_HEAP_PROPERTIES heap {};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC desc {};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = width;
    desc.Height = height;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = TypelessFor(rtvFormat);
    if (desc.Format == DXGI_FORMAT_UNKNOWN)
    {
        *hr = E_INVALIDARG;
        return false;
    }
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS;
    *hr = device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_SHARED, &desc, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                          IID_PPV_ARGS(texture));
    if (FAILED(*hr) || *texture == nullptr)
    {
        *texture = nullptr;
        return false;
    }
    *hr = device->CreateSharedHandle(*texture, nullptr, GENERIC_ALL, nullptr, handle);
    if (FAILED(*hr) || *handle == nullptr)
    {
        (*texture)->Release();
        *texture = nullptr;
        *handle = nullptr;
        return false;
    }
    D3D12_RENDER_TARGET_VIEW_DESC view {};
    view.Format = rtvFormat;
    view.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
    device->CreateRenderTargetView(*texture, &view, rtv);
    return true;
}

// A shared fence (the game signals it with each image's sequence; the layer waits on it).
inline bool MakeSharedFence(ID3D12Device* device, ID3D12Fence** fence, HANDLE* handle, HRESULT* hr)
{
    *fence = nullptr;
    *handle = nullptr;
    *hr = device->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(fence));
    if (FAILED(*hr) || *fence == nullptr)
    {
        *fence = nullptr;
        return false;
    }
    *hr = device->CreateSharedHandle(*fence, nullptr, GENERIC_ALL, nullptr, handle);
    if (FAILED(*hr) || *handle == nullptr)
    {
        (*fence)->Release();
        *fence = nullptr;
        *handle = nullptr;
        return false;
    }
    return true;
}
} // namespace OverlayTextures12
