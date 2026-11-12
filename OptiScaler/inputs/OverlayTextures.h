#pragma once

// The headset overlay's shared textures (inputs/OverlayShare), header only and free of everything but Direct3D, so
// that tools/xr_observer/test/overlay_test makes and reads them with THIS code on a real device.
// Two kinds of shared handle: NT handles (IDXGIResource1::CreateSharedHandle, opened with
// ID3D11Device1::OpenSharedResource1), and the old kind (IDXGIResource::GetSharedHandle, opened with
// ID3D11Device::OpenSharedResource) for a game's device that is a wrapper knowing less than Direct3D 11.1. The old
// kind is for a short list of TYPED formats, so its textures are R8G8B8A8_UNORM; the NT kind's are typeless.
#include <d3d11.h>
#include <dxgi1_2.h>

namespace OverlayTextures
{
// One texture of width x height with its render target view (UNORM: an overlay's colours are written as they are)
// and its shared handle. On failure nothing is left behind.
inline bool MakeShared(ID3D11Device* device, UINT width, UINT height, bool oldKind, ID3D11Texture2D** texture,
                       ID3D11RenderTargetView** rtv, HANDLE* handle)
{
    *texture = nullptr;
    *rtv = nullptr;
    *handle = nullptr;
    D3D11_TEXTURE2D_DESC desc {};
    desc.Width = width;
    desc.Height = height;
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = oldKind ? DXGI_FORMAT_R8G8B8A8_UNORM : DXGI_FORMAT_R8G8B8A8_TYPELESS;
    desc.SampleDesc.Count = 1;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
    desc.MiscFlags =
        oldKind ? D3D11_RESOURCE_MISC_SHARED : (D3D11_RESOURCE_MISC_SHARED | D3D11_RESOURCE_MISC_SHARED_NTHANDLE);
    D3D11_RENDER_TARGET_VIEW_DESC view {};
    view.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    view.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2D;
    bool ok = SUCCEEDED(device->CreateTexture2D(&desc, nullptr, texture)) && *texture != nullptr &&
              SUCCEEDED(device->CreateRenderTargetView(*texture, &view, rtv)) && *rtv != nullptr;
    if (ok && oldKind)
    {
        IDXGIResource* dxgi = nullptr;
        ok = SUCCEEDED((*texture)->QueryInterface(IID_PPV_ARGS(&dxgi))) && dxgi != nullptr &&
             SUCCEEDED(dxgi->GetSharedHandle(handle)) && *handle != nullptr;
        if (dxgi != nullptr)
            dxgi->Release();
    }
    else if (ok)
    {
        IDXGIResource1* dxgi = nullptr;
        ok = SUCCEEDED((*texture)->QueryInterface(IID_PPV_ARGS(&dxgi))) && dxgi != nullptr &&
             SUCCEEDED(dxgi->CreateSharedHandle(nullptr, DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE,
                                                nullptr, handle)) &&
             *handle != nullptr;
        if (dxgi != nullptr)
            dxgi->Release();
    }
    if (!ok)
    {
        if (*rtv != nullptr)
            (*rtv)->Release();
        if (*texture != nullptr)
            (*texture)->Release();
        if (*handle != nullptr && !oldKind)
            CloseHandle(*handle);
        *texture = nullptr;
        *rtv = nullptr;
        *handle = nullptr;
    }
    return ok;
}
} // namespace OverlayTextures
