#pragma once
// The scene depth for the stale-eye layer (StaleEyeShareAbi.h), made on a game's D3D11 device that may have NO fences
// and NO NT shared handles (a game's device with no ID3D11Device5, NT handles refused -- the headset panel met both
// first). Header only and free of everything but Direct3D, so that taa_test runs THIS code on a real device and
// reads the result from a second device.
//
// The depth of one Present is converted into one of kStaleEyeDepthSlots shared R16_FLOAT textures by a compute pass in
// a SPARE device state (ID3D11DeviceContext1::SwapDeviceContextState, as per-eye DLSS does on the D3D11 path every Present):
// nothing of the game's state is touched. The game's depth texture is only read: a view is made for the pass and let
// go at once (nothing of the game's is kept). Shared handles are NT handles where the device makes them, else the old
// kind (IDXGIResource::GetSharedHandle, never closed: they end with their textures). No fences: the layer reads a
// slot only once a later Present has copied (StaleEyeShareAbi.h). Textures of a size gone are let go only
// kKeepPresents later (the layer may still read them on its own device).
//
// HALF PRECISION (~235 MB less at a 3920x3920 eye with 8 slots): the warp needs the depth roughly --
// SMOOTH averages it over coarse cells, its comparisons allow 5% -- and reversed Z keeps ~0.05% relative precision in
// 16 bits down to 6e-5 (one game: ~1300 game units at a near plane of 0.08), and stays under 1% to the far plane. The
// sign (the marks) and 0 (the sky) are exact. The layer views the textures in their own format (FC5's are R32_FLOAT).
//
// MARKS (depthMarks): the pixels of one stencil CLASS are written with their depth NEGATED. One game's classes are the
// stencil's bits 6 and 7 (as its TAA's dilation uses them): 1 characters and their weapons, 2 foliage, 3
// water, 0 the rest. The stencil is read from the same depth texture (its second plane), never kept.
#include <d3d11_1.h>
#include <d3dcompiler.h>
#include <dxgi1_2.h>

#include <cstdint>
#include <string>
#include <vector>

#ifndef STALE_EYE_DEPTH_SLOTS
#include "StaleEyeShareAbi.h"
#define STALE_EYE_DEPTH_SLOTS kStaleEyeDepthSlots
#endif

namespace StaleEyeDepth
{
constexpr int kSlots = STALE_EYE_DEPTH_SLOTS;
constexpr uint64_t kKeepPresents = 600;

constexpr char kShader[] = R"(
Texture2D<float> Depth : register(t0);
RWTexture2D<float> Out : register(u0);
[numthreads(8, 8, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    uint w, h;
    Out.GetDimensions(w, h);
    if (id.x < w && id.y < h)
        Out[id.xy] = Depth[id.xy];
}
)";
// The same with the marks: MARK_CLASS is defined when compiled (the class whose depth is written negated). A depth of
// 0 (the sky) stays 0: nothing there to mark.
constexpr char kMarkShader[] = R"(
Texture2D<float> Depth : register(t0);
Texture2D<uint2> Stencil : register(t1);
RWTexture2D<float> Out : register(u0);
[numthreads(8, 8, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    uint w, h;
    Out.GetDimensions(w, h);
    if (id.x < w && id.y < h)
    {
        const float d = Depth[id.xy];
        const uint kind = (Stencil[id.xy].y >> 6) & 3;
        Out[id.xy] = kind == MARK_CLASS && d > 0.0 ? -d : d;
    }
}
)";
// The stencil class of a game's characters (and their weapons).
constexpr int kCharacterClass = 1;

// The view that reads the depth of a typeless depth texture as a float (the depth part only), or UNKNOWN.
inline DXGI_FORMAT DepthViewFormat(DXGI_FORMAT texture)
{
    switch (texture)
    {
    case DXGI_FORMAT_R32G8X24_TYPELESS: // format 19
        return DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS;
    case DXGI_FORMAT_R32_TYPELESS:
        return DXGI_FORMAT_R32_FLOAT;
    case DXGI_FORMAT_R24G8_TYPELESS:
        return DXGI_FORMAT_R24_UNORM_X8_TYPELESS;
    case DXGI_FORMAT_R16_TYPELESS:
        return DXGI_FORMAT_R16_UNORM;
    case DXGI_FORMAT_R32_FLOAT:
    case DXGI_FORMAT_R16_UNORM:
        return texture;
    default:
        return DXGI_FORMAT_UNKNOWN;
    }
}

// The view that reads the STENCIL of a typeless depth texture (as uint2, the stencil in .y), or UNKNOWN.
inline DXGI_FORMAT StencilViewFormat(DXGI_FORMAT texture)
{
    switch (texture)
    {
    case DXGI_FORMAT_R32G8X24_TYPELESS: // format 19
        return DXGI_FORMAT_X32_TYPELESS_G8X24_UINT;
    case DXGI_FORMAT_R24G8_TYPELESS:
        return DXGI_FORMAT_X24_TYPELESS_G8_UINT;
    default:
        return DXGI_FORMAT_UNKNOWN;
    }
}

// Why the marks were not written with a copy (the depth is copied all the same, unmarked).
enum MarkResult
{
    kMarked = 0,
    kMarksNotAsked,     // no class asked for
    kMarksNoStencil,    // the depth's format has no stencil to read
    kMarksNoShader,     // the marking shader did not compile or could not be made (said once by Copy's caller)
    kMarksNoView,       // the stencil view could not be made
    kMarkResults
};
inline const char* MarkResultName(int r)
{
    static const char* const names[kMarkResults] = { "marked", "not asked", "the depth's format has no stencil",
                                                     "the marking shader could not be made",
                                                     "the stencil view could not be made" };
    return r >= 0 && r < kMarkResults ? names[r] : "?";
}

// Why a copy was not made. kCopied first.
enum Result
{
    kCopied = 0,
    kNotReady,      // Setup failed (said by Setup)
    kNoDepth,       // no texture handed
    kNoView,        // the depth's format is not one that can be read, or it has no shader-resource binding
    kNoTextures,    // the shared textures of this size could not be made (said by EnsureSize)
    kNoContext1,    // the context has no ID3D11DeviceContext1 (no spare state)
    kOtherSize,     // the depth is not the shared textures' size
    kResults
};
inline const char* ResultName(int r)
{
    static const char* const names[kResults] = { "copied", "not set up", "no depth texture",
                                                 "the depth cannot be read (format or binding)",
                                                 "no shared textures of this size", "no ID3D11DeviceContext1",
                                                 "the depth is not the shared textures' size" };
    return r >= 0 && r < kResults ? names[r] : "?";
}

class Share
{
  public:
    // Once, on the game's device. False with `why` said; nothing is left half made.
    bool Setup(ID3D11Device* device, std::string& why)
    {
        if (_ready)
            return true;
        if (_tried)
        {
            why = _why;
            return false;
        }
        _tried = true;
        ID3DBlob* code = nullptr;
        ID3DBlob* errors = nullptr;
        if (FAILED(D3DCompile(kShader, sizeof(kShader) - 1, "StaleEyeDepth", nullptr, nullptr, "CSMain", "cs_5_0",
                              D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &code, &errors)))
        {
            _why = std::string("the copy shader did not compile: ") +
                   (errors != nullptr ? (const char*) errors->GetBufferPointer() : "no message");
            if (errors != nullptr)
                errors->Release();
            why = _why;
            return false;
        }
        if (errors != nullptr)
            errors->Release();
        const bool made =
            SUCCEEDED(device->CreateComputeShader(code->GetBufferPointer(), code->GetBufferSize(), nullptr, &_shader));
        code->Release();
        ID3D11Device1* device1 = nullptr;
        if (made && SUCCEEDED(device->QueryInterface(IID_PPV_ARGS(&device1))) && device1 != nullptr)
        {
            const D3D_FEATURE_LEVEL level = device->GetFeatureLevel();
            const UINT flags = (device->GetCreationFlags() & D3D11_CREATE_DEVICE_SINGLETHREADED) != 0
                                   ? D3D11_1_CREATE_DEVICE_CONTEXT_STATE_SINGLETHREADED
                                   : 0;
            if (FAILED(device1->CreateDeviceContextState(flags, &level, 1, D3D11_SDK_VERSION, __uuidof(ID3D11Device),
                                                         nullptr, &_spare)))
                _spare = nullptr;
            device1->Release();
        }
        _ready = made && _shader != nullptr && _spare != nullptr;
        if (!_ready)
        {
            _why = !made ? "the copy shader could not be made"
                         : "no spare device state (ID3D11Device1::CreateDeviceContextState)";
            why = _why;
        }
        return _ready;
    }

    // The shared textures of width x height (made once for a size; the old size's are let go kKeepPresents later).
    // False with `why` when they cannot be made; a size refused is not tried again.
    bool EnsureSize(ID3D11Device* device, UINT width, UINT height, uint64_t present, std::string& why)
    {
        if (_width == width && _height == height && _slots[0].texture != nullptr)
            return true;
        if (_refusedW == width && _refusedH == height)
        {
            why = _why;
            return false;
        }
        // The old size's textures: let go kKeepPresents from now (the layer may be reading them on its device).
        if (_slots[0].texture != nullptr)
        {
            for (auto& slot : _slots)
                _retired.push_back({ slot, _legacy, present });
            for (auto& slot : _slots)
                slot = Slot {};
            _width = _height = 0;
        }
        for (int attempt = _legacyOnly ? 1 : 0; attempt < 2; ++attempt)
        {
            const bool oldKind = attempt == 1;
            bool ok = true;
            for (auto& slot : _slots)
                ok = ok && Make(device, width, height, oldKind, slot);
            if (ok)
            {
                _legacy = oldKind;
                _width = width;
                _height = height;
                ++_generation;
                return true;
            }
            DropNow(); // never used by anyone: made just now, handed to no one
            if (!oldKind)
                _legacyOnly = true; // NT handles are not tried again on this device
        }
        _refusedW = width;
        _refusedH = height;
        _why = "the shared R16_FLOAT textures could not be made, NT or old kind (" + std::to_string(width) + "x" +
               std::to_string(height) + ")";
        why = _why;
        return false;
    }

    // `depth` (the game's) into `slot`, in a spare state. The textures must be of the depth's size (EnsureSize).
    // `markClass` > 0: the pixels of that stencil class are written negated; `marks` says whether they were (a depth
    // that cannot be marked is copied unmarked).
    int Copy(ID3D11DeviceContext* ctx, ID3D11Texture2D* depth, int slot, int markClass = 0, int* marks = nullptr)
    {
        if (marks != nullptr)
            *marks = kMarksNotAsked;
        if (!_ready)
            return kNotReady;
        if (depth == nullptr)
            return kNoDepth;
        if (_slots[slot].uav == nullptr)
            return kNoTextures;
        D3D11_TEXTURE2D_DESC desc {};
        depth->GetDesc(&desc);
        D3D11_SHADER_RESOURCE_VIEW_DESC view {};
        view.Format = DepthViewFormat(desc.Format);
        if (desc.Width != _width || desc.Height != _height)
            return kOtherSize;
        if (view.Format == DXGI_FORMAT_UNKNOWN || (desc.BindFlags & D3D11_BIND_SHADER_RESOURCE) == 0 ||
            desc.SampleDesc.Count != 1)
            return kNoView;
        view.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
        view.Texture2D.MipLevels = 1;
        ID3D11Device* device = nullptr;
        ctx->GetDevice(&device);
        ID3D11ShaderResourceView* srv = nullptr;
        const bool viewed = device != nullptr && SUCCEEDED(device->CreateShaderResourceView(depth, &view, &srv));
        // The marks: the stencil plane of the same texture, with the marking shader of that class.
        ID3D11ShaderResourceView* stencil = nullptr;
        ID3D11ComputeShader* shader = _shader;
        int markResult = kMarksNotAsked;
        if (viewed && markClass > 0)
        {
            D3D11_SHADER_RESOURCE_VIEW_DESC sv = view;
            sv.Format = StencilViewFormat(desc.Format);
            ID3D11ComputeShader* marking = sv.Format == DXGI_FORMAT_UNKNOWN ? nullptr : MarkShader(device, markClass);
            if (sv.Format == DXGI_FORMAT_UNKNOWN)
                markResult = kMarksNoStencil;
            else if (marking == nullptr)
                markResult = kMarksNoShader;
            else if (FAILED(device->CreateShaderResourceView(depth, &sv, &stencil)) || stencil == nullptr)
                markResult = kMarksNoView;
            else
            {
                markResult = kMarked;
                shader = marking;
            }
        }
        if (marks != nullptr)
            *marks = markResult;
        if (device != nullptr)
            device->Release();
        if (!viewed || srv == nullptr)
            return kNoView;
        ID3D11DeviceContext1* ctx1 = nullptr;
        if (FAILED(ctx->QueryInterface(IID_PPV_ARGS(&ctx1))) || ctx1 == nullptr)
        {
            srv->Release();
            if (stencil != nullptr)
                stencil->Release();
            if (marks != nullptr)
                *marks = kMarksNotAsked;
            return kNoContext1;
        }
        ID3DDeviceContextState* gameState = nullptr;
        ctx1->SwapDeviceContextState(_spare, &gameState);
        ctx1->CSSetShader(shader, nullptr, 0);
        ID3D11ShaderResourceView* views[2] = { srv, stencil };
        ctx1->CSSetShaderResources(0, 2, views);
        ctx1->CSSetUnorderedAccessViews(0, 1, &_slots[slot].uav, nullptr);
        ctx1->Dispatch((_width + 7) / 8, (_height + 7) / 8, 1);
        ID3D11ShaderResourceView* noSrv[2] = {};
        ID3D11UnorderedAccessView* noUav = nullptr;
        ctx1->CSSetShaderResources(0, 2, noSrv);
        ctx1->CSSetUnorderedAccessViews(0, 1, &noUav, nullptr);
        ctx1->CSSetShader(nullptr, nullptr, 0);
        ID3DDeviceContextState* spare = nullptr;
        ctx1->SwapDeviceContextState(gameState, &spare);
        if (spare != nullptr)
            spare->Release(); // _spare keeps its own reference
        if (gameState != nullptr)
            gameState->Release();
        ctx1->Release();
        srv->Release(); // the dispatch is recorded; the views are not kept
        if (stencil != nullptr)
            stencil->Release();
        return kCopied;
    }

    // Textures of sizes gone, let go once kKeepPresents have passed since. Every Present.
    void ReleaseOld(uint64_t present)
    {
        for (size_t i = 0; i < _retired.size();)
        {
            if (present - _retired[i].since >= kKeepPresents)
            {
                Release(_retired[i].slot, _retired[i].legacy);
                _retired[i] = _retired.back();
                _retired.pop_back();
            }
            else
                ++i;
        }
    }
    size_t RetiredSizes() const { return _retired.size() / kSlots; }

    // The offline test: the old kind of handle as on a device that refuses NT handles.
    void ForceOldKind() { _legacyOnly = true; }
    bool Ready() const { return _ready; }
    bool Legacy() const { return _legacy; }
    uint64_t Generation() const { return _generation; }
    UINT Width() const { return _width; }
    UINT Height() const { return _height; }
    HANDLE Handle(int slot) const { return _slots[slot].handle; }
    // Why the marking shader could not be made (empty while it could, or was never asked for).
    const std::string& MarkError() const { return _markWhy; }
    ID3D11Texture2D* Texture(int slot) const { return _slots[slot].texture; } // for the offline test

  private:
    struct Slot
    {
        ID3D11Texture2D* texture = nullptr;
        ID3D11UnorderedAccessView* uav = nullptr;
        HANDLE handle = nullptr;
    };
    struct Retired
    {
        Slot slot;
        bool legacy = false;
        uint64_t since = 0;
    };

    // The marking shader of `markClass` (1..3), compiled once; a failed compile is final (not tried every Present).
    ID3D11ComputeShader* MarkShader(ID3D11Device* device, int markClass)
    {
        if (markClass < 1 || markClass > 3 || device == nullptr)
            return nullptr;
        if (_marking[markClass] != nullptr || _markTried[markClass])
            return _marking[markClass];
        _markTried[markClass] = true;
        const std::string value = std::to_string(markClass);
        const D3D_SHADER_MACRO defines[] = { { "MARK_CLASS", value.c_str() }, { nullptr, nullptr } };
        ID3DBlob* code = nullptr;
        ID3DBlob* errors = nullptr;
        if (FAILED(D3DCompile(kMarkShader, sizeof(kMarkShader) - 1, "StaleEyeDepthMarks", defines, nullptr, "CSMain",
                              "cs_5_0", D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &code, &errors)))
        {
            _markWhy = std::string("the marking shader did not compile: ") +
                       (errors != nullptr ? (const char*) errors->GetBufferPointer() : "no message");
            if (errors != nullptr)
                errors->Release();
            return nullptr;
        }
        if (errors != nullptr)
            errors->Release();
        if (FAILED(device->CreateComputeShader(code->GetBufferPointer(), code->GetBufferSize(), nullptr,
                                               &_marking[markClass])))
        {
            _marking[markClass] = nullptr;
            _markWhy = "the marking shader could not be made";
        }
        code->Release();
        return _marking[markClass];
    }

    static bool Make(ID3D11Device* device, UINT width, UINT height, bool oldKind, Slot& slot)
    {
        D3D11_TEXTURE2D_DESC desc {};
        desc.Width = width;
        desc.Height = height;
        desc.MipLevels = 1;
        desc.ArraySize = 1;
        desc.Format = DXGI_FORMAT_R16_FLOAT;
        desc.SampleDesc.Count = 1;
        desc.Usage = D3D11_USAGE_DEFAULT;
        desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
        desc.MiscFlags =
            oldKind ? D3D11_RESOURCE_MISC_SHARED : (D3D11_RESOURCE_MISC_SHARED | D3D11_RESOURCE_MISC_SHARED_NTHANDLE);
        if (FAILED(device->CreateTexture2D(&desc, nullptr, &slot.texture)) || slot.texture == nullptr ||
            FAILED(device->CreateUnorderedAccessView(slot.texture, nullptr, &slot.uav)) || slot.uav == nullptr)
            return false;
        bool ok = false;
        if (oldKind)
        {
            IDXGIResource* dxgi = nullptr;
            ok = SUCCEEDED(slot.texture->QueryInterface(IID_PPV_ARGS(&dxgi))) && dxgi != nullptr &&
                 SUCCEEDED(dxgi->GetSharedHandle(&slot.handle)) && slot.handle != nullptr;
            if (dxgi != nullptr)
                dxgi->Release();
        }
        else
        {
            IDXGIResource1* dxgi = nullptr;
            ok = SUCCEEDED(slot.texture->QueryInterface(IID_PPV_ARGS(&dxgi))) && dxgi != nullptr &&
                 SUCCEEDED(dxgi->CreateSharedHandle(nullptr, DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE,
                                                    nullptr, &slot.handle)) &&
                 slot.handle != nullptr;
            if (dxgi != nullptr)
                dxgi->Release();
        }
        return ok;
    }

    static void Release(Slot& slot, bool legacy)
    {
        if (slot.uav != nullptr)
            slot.uav->Release();
        if (slot.texture != nullptr)
            slot.texture->Release();
        if (slot.handle != nullptr && !legacy)
            CloseHandle(slot.handle);
        slot = Slot {};
    }

    // What an attempt made, let go at once: it was never handed to anyone or used by any pass.
    void DropNow()
    {
        for (auto& slot : _slots)
            Release(slot, _legacyOnly || slot.handle == nullptr);
    }

    Slot _slots[kSlots];
    std::vector<Retired> _retired;
    ID3D11ComputeShader* _shader = nullptr;
    ID3D11ComputeShader* _marking[4] = {}; // by class; kept for the process, as _shader
    bool _markTried[4] = {};
    std::string _markWhy;
    ID3DDeviceContextState* _spare = nullptr;
    bool _tried = false, _ready = false, _legacy = false, _legacyOnly = false;
    UINT _width = 0, _height = 0, _refusedW = 0, _refusedH = 0;
    uint64_t _generation = 0;
    std::string _why;
};
} // namespace StaleEyeDepth
