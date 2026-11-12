#include "pch.h"
#include "Fc5Taa_Dx11.h"
#include "StaleEyeShare.h"

#include "Config.h"
#include "Logger.h"
#include "NVNGX_Parameter.h"
#include "resource.h"
#include "State.h"
#include "Util.h"

#include <d3d11_1.h>
#include <d3d11_4.h>
#include <dxgi1_2.h>
#include <dxgi1_4.h>
#include <d3dcompiler.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <format>
#include <mutex>
#include <string>
#include <unordered_set>
#include <vector>

namespace
{
// One pass: this Present's motion (to the previous Present, the other eye under alternate-eye VR) into Motion, and
// the motion to two Presents back (the same eye) into Composed, both in render-size pixels, previous minus current.
//
// Camera motion is computed like FC5's own motion vector pass, but per pixel and in float: (ndc, depth, 1) through
// its current-to-previous matrix (copied from that pass's cb0, rows at +16..+64), previous NDC minus NDC. The game stores
// the result in 8 bits per axis on a curve (about 0.5 px steps at the usual eye-to-eye offsets, several px when
// turning), which made DLSS's reprojection wobble. Composed motion chains this Present's matrix with the previous
// Present's, which is exact for anything static. Pixels FC5 flags as moving objects (object-flag bit 3, depth not 0)
// and every pixel when a matrix is missing use the game's own vectors instead: its half-res R8G8 texture decoded (per
// channel d = v - 0.498039, f = |d| > 0.33437 ? d * 4.830047 -+ 1.415024 : sign(d) * (2d)^4, previous UV = UV + 0.1 f;
// tools/capture/fc5mvwarp.py), chained with the previous Present's Motion read where this Present's points, with
// Days Gone's occlusion handling (a landing point clearly nearer, reversed Z, takes the previous motion from the
// closest texel along the row at this pixel's depth, a few texels further out).
constexpr char kDecodeShader[] = R"(
Texture2D<float2> Packed : register(t0);
Texture2D<float2> Previous : register(t1);
Texture2D<float> Depth : register(t2);
Texture2D<float> PreviousDepth : register(t3);
Texture2D<float4> ShaderIndex : register(t4);
RWTexture2D<float2> Motion : register(u0);
RWTexture2D<float2> Composed : register(u1);

cbuffer Constants : register(b0)
{
    uint HasPrevious;
    uint Occlusion;
    uint UseMatrix;
    uint UsePreviousMatrix;
};
// The game's temporal constants: the current-to-previous matrix's rows in [1]..[4].
cbuffer CurrentToPrevious : register(b1)
{
    float4 A[8];
};
cbuffer PreviousToTwoBack : register(b2)
{
    float4 B[8];
};

static const int kSearch = 96;
static const int kMargin = 3;

bool SameDepth(float a, float b) { return abs(a - b) <= a * 0.05 + 1e-6; }

float Decode(float v)
{
    const float d = v - 0.498039;
    const float f = d > 0.33437 ? d * 4.830047 - 1.415024
                  : d < -0.33437 ? d * 4.830047 + 1.415024
                  : sign(d) * pow(2.0 * d, 4.0);
    return 0.1 * f;
}

float4 Transform(float4 p, float4 r1, float4 r2, float4 r3, float4 r4)
{
    return float4(dot(p, r1), dot(p, r2), dot(p, r3), dot(p, r4));
}

[numthreads(8, 8, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    uint w, h;
    Motion.GetDimensions(w, h);
    if (id.x >= w || id.y >= h)
        return;
    const float2 size = float2(w, h);
    const float2 uv = (float2(id.xy) + 0.5) / size;
    const float2 ndc = float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    const float z = Depth[id.xy];
    // UseMatrix 0: no matrix, the game's vectors everywhere. 1: the game's vectors on object-flagged pixels. 2: matrix
    // motion everywhere. 3 (default): object-flagged pixels use the game's vectors only where they disagree with the
    // matrix motion by more than the 8-bit curve's step (about 8 (|m| / 255)^0.75 px) plus 1 px, i.e. where the thing
    // really moves; static props FC5 flags as objects (indoors) otherwise wobbled with the quantised vectors.
    const bool flagged = z != 0.0 && (uint(ShaderIndex[id.xy].x * 255.0) & 8u) != 0;

    float2 matrixOffset = 0;
    float4 matrixPrevious = 0;
    if (UseMatrix != 0)
    {
        const float4 c = Transform(float4(ndc, z, 1.0), A[1], A[2], A[3], A[4]);
        matrixPrevious = c / c.w;
        matrixOffset = (matrixPrevious.xy - ndc) * float2(0.5, -0.5) * size;
    }
    float2 decoded = 0;
    if (UseMatrix == 0 || (flagged && (UseMatrix == 1 || UseMatrix == 3)))
    {
        uint pw, ph;
        Packed.GetDimensions(pw, ph);
        const float2 packed = Packed[min(id.xy / 2, uint2(pw - 1, ph - 1))];
        decoded = float2(Decode(packed.x), Decode(packed.y)) * size;
    }
    bool object = UseMatrix == 0 || (UseMatrix == 1 && flagged);
    if (UseMatrix == 3 && flagged)
    {
        const float2 tolerance = 1.0 + 8.0 * pow(abs(matrixOffset) / 255.0, 0.75);
        object = any(abs(decoded - matrixOffset) > tolerance);
    }

    float2 offset;
    float2 composed = 0;
    bool composedDone = false;
    if (object)
        offset = decoded;
    else
    {
        const float4 previous = matrixPrevious;
        offset = matrixOffset;
        if (HasPrevious != 0 && UsePreviousMatrix != 0)
        {
            const float4 c2 = Transform(previous, B[1], B[2], B[3], B[4]);
            composed = (c2.xy / c2.w - ndc) * float2(0.5, -0.5) * size;
            composedDone = true;
        }
    }
    Motion[id.xy] = offset;

    if (!composedDone)
    {
        composed = offset;
        if (HasPrevious != 0)
        {
            const int2 at = clamp(int2(floor(float2(id.xy) + 0.5 + offset)), int2(0, 0), int2(w - 1, h - 1));
            float2 previous = Previous[at];
            if (Occlusion != 0 && PreviousDepth[at] > z * 1.05 + 1e-5)
            {
                for (int k = 1; k <= kSearch; ++k)
                {
                    const int2 left = int2(max(at.x - k, 0), at.y);
                    const int2 right = int2(min(at.x + k, (int) w - 1), at.y);
                    if (SameDepth(z, PreviousDepth[left]))
                    {
                        previous = Previous[int2(max(left.x - kMargin, 0), at.y)];
                        break;
                    }
                    if (SameDepth(z, PreviousDepth[right]))
                    {
                        previous = Previous[int2(min(right.x + kMargin, (int) w - 1), at.y)];
                        break;
                    }
                }
            }
            composed += previous;
        }
    }
    Composed[id.xy] = composed;
}
)";

// DLSS's output (same encoding as the TAA's colour input) into the TAA output, alpha 1 as the game writes it.
constexpr char kCopyShader[] = R"(
Texture2D<float4> Source : register(t0);
RWTexture2D<float4> Output : register(u0);

[numthreads(8, 8, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    uint w, h;
    Output.GetDimensions(w, h);
    if (id.x >= w || id.y >= h)
        return;
    Output[id.xy] = float4(Source[id.xy].rgb, 1.0);
}
)";

// Diagnostics: the scene in grey, moving pixels tinted red/green by their X/Y offset (full at 16 px).
constexpr char kViewShader[] = R"(
Texture2D<float2> Motion : register(t0);
Texture2D<float4> Scene : register(t1);
RWTexture2D<float4> Output : register(u0);

[numthreads(8, 8, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    uint w, h;
    Output.GetDimensions(w, h);
    if (id.x >= w || id.y >= h)
        return;
    const float2 mv = Motion[id.xy];
    const float grey = 0.6 * dot(saturate(Scene[id.xy].rgb), float3(0.25, 0.5, 0.25));
    const float3 tint = float3(saturate(0.5 + mv / 32.0), 0.5);
    Output[id.xy] = float4(lerp(grey.xxx, tint, saturate(length(mv) / 2.0)), 1.0);
}
)";

// All on the thread driving the immediate context; the modes and the Present count cross threads.
thread_local int g_inPass = 0;

struct PassScope
{
    PassScope() { ++g_inPass; }
    ~PassScope() { --g_inPass; }
};

enum Mode : int
{
    GameTaa = 0,
    Dlaa = 1,
};
std::atomic<int> g_mode { GameTaa };
// 0 off, 1 decoded, 2 composed.
std::atomic<int> g_viewMode { 0 };
std::atomic<bool> g_occlusion { true };
// Object-flagged pixels (F9 cycles): 0 game's vectors where they disagree with the matrix motion (default), 1 matrix
// motion everywhere, 2 game's vectors on every flagged pixel (the old way).
std::atomic<int> g_objectMotion { 0 };
// DLSS jitter = signs * the image's shift (x right, y down).
constexpr float kJitterSigns[4][2] = { { 1, 1 }, { -1, -1 }, { 1, -1 }, { -1, 1 } };
std::atomic<int> g_jitterVariant { 0 };

std::atomic<uint64_t> g_present { 0 };

// The TAA's pixel shaders found so far (compared by address only; FC5 switches to another permutation while aiming),
// and pixel shaders that failed the check, so other draws are dismissed with one lookup.
constexpr int kMaxTaaShaders = 8;
const void* g_taaShaders[kMaxTaaShaders] = {};
int g_taaShaderCount = 0;
// Per thread: before setup any of FC5's devices' contexts can get here.
thread_local std::unordered_set<const void*> g_rejectedShaders;
constexpr size_t kMaxRejected = 4096;
// The TAA's two motion textures (t0 this Present's, t1 the previous one's), compared by address only: the motion
// vector pass is the compute dispatch writing one of them as u0.
const void* g_mvTextures[2] = {};

// Copies of that pass's constants (the current-to-previous matrix), made on the GPU at the pass, by Present parity, and its
// object-flag texture (t3), held.
struct MatrixCopy
{
    ID3D11Buffer* buffer = nullptr;
    UINT bytes = 0;
    uint64_t present = UINT64_MAX;
};
MatrixCopy g_matrices[2];
ID3D11ShaderResourceView* g_shaderIndex = nullptr;
uint64_t g_matrixCopies = 0;
uint64_t g_matrixFrames = 0;
uint64_t g_ownJitterFrames = 0;

// ---- G-buffer depth, from the render target binds ----

ID3D11Texture2D* g_depth = nullptr;
// The Present of the last G-buffer bind with g_depth: menus and loading screens draw no G-buffer, so "bound this
// Present" is what tells gameplay from them (their TAA-like draws come at other sizes, or at the eye's).
std::atomic<uint64_t> g_depthBindPresent { UINT64_MAX };
struct DeferredRelease
{
    IUnknown* texture;
    uint64_t present;
};
std::vector<DeferredRelease> g_deferred;
constexpr uint64_t kReleaseAfterPresents = 600;

void ReleaseDeferred()
{
    const uint64_t now = g_present.load();
    for (size_t i = 0; i < g_deferred.size();)
    {
        if (now - g_deferred[i].present >= kReleaseAfterPresents)
        {
            g_deferred[i].texture->Release();
            g_deferred[i] = g_deferred.back();
            g_deferred.pop_back();
        }
        else
        {
            ++i;
        }
    }
}

// Returns an AddRef'd Texture2D, or nullptr.
ID3D11Texture2D* TextureFromView(ID3D11View* view)
{
    if (view == nullptr)
        return nullptr;
    ID3D11Resource* resource = nullptr;
    view->GetResource(&resource);
    if (resource == nullptr)
        return nullptr;
    ID3D11Texture2D* texture = nullptr;
    resource->QueryInterface(IID_PPV_ARGS(&texture));
    resource->Release();
    return texture;
}

D3D11_TEXTURE2D_DESC Desc(ID3D11Texture2D* texture)
{
    D3D11_TEXTURE2D_DESC desc {};
    texture->GetDesc(&desc);
    return desc;
}

bool IsRgba8(DXGI_FORMAT format)
{
    return format == DXGI_FORMAT_R8G8B8A8_TYPELESS || format == DXGI_FORMAT_R8G8B8A8_UNORM ||
           format == DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
}

bool IsRg8(DXGI_FORMAT format) { return format == DXGI_FORMAT_R8G8_TYPELESS || format == DXGI_FORMAT_R8G8_UNORM; }

// ---- The TAA draw ----

constexpr UINT kMinRenderSize = 640;

// The game's bindings at the TAA draw. All AddRef'd.
struct TaaViews
{
    ID3D11ShaderResourceView* srvs[5] = {};
    ID3D11RenderTargetView* rtv = nullptr;
    ID3D11Texture2D* output = nullptr;
    ID3D11Texture2D* colour = nullptr;
    ID3D11Buffer* constants = nullptr;
    UINT width = 0;
    UINT height = 0;

    ~TaaViews()
    {
        for (auto* srv : srvs)
        {
            if (srv != nullptr)
                srv->Release();
        }
        if (rtv != nullptr)
            rtv->Release();
        if (output != nullptr)
            output->Release();
        if (colour != nullptr)
            colour->Release();
        if (constants != nullptr)
            constants->Release();
    }
};

bool SrvDesc(ID3D11ShaderResourceView* srv, D3D11_TEXTURE2D_DESC& desc)
{
    ID3D11Texture2D* texture = TextureFromView(srv);
    if (texture == nullptr)
        return false;
    desc = Desc(texture);
    texture->Release();
    return true;
}

bool KnownTaaShader(const void* shader)
{
    for (int i = 0; i < g_taaShaderCount; ++i)
    {
        if (g_taaShaders[i] == shader)
            return true;
    }
    return false;
}

void Reject(const void* shader)
{
    if (g_rejectedShaders.size() >= kMaxRejected)
        g_rejectedShaders.clear();
    g_rejectedShaders.insert(shader);
}

// A draw that looked like the TAA (rt0 RGBA8 of at least the minimum size, ps t0 R8G8) but failed a later check: once
// per pixel shader, why.
void LogNearMiss(const void* shader, const char* why)
{
    static int logged = 0;
    if (logged >= 16)
        return;
    ++logged;
    LOG_INFO("FC5 TAA: a TAA-like draw (pixel shader {:X}) doesn't match: {}", (size_t) shader, why);
}

// The TAA draw: rt0 RGBA8 (render size, also sampled later), ps t0/t1 half-size R8G8 (t0 viewed as R8G8_UNORM),
// t2..t4 RGBA8 of the rt0 size, ps cb0 at least 96 bytes. Any pixel shader passing this counts (FC5 has more than
// one TAA permutation); one that fails is not looked at again, unless it looked TAA-like (then it may be checked
// again with other bindings).
bool MatchTaa(ID3D11DeviceContext* ctx, TaaViews& views)
{
    ID3D11PixelShader* ps = nullptr;
    ctx->PSGetShader(&ps, nullptr, nullptr);
    if (ps == nullptr)
        return false;
    const void* shader = ps;
    ps->Release();
    const bool known = KnownTaaShader(shader);
    if (!known && g_rejectedShaders.count(shader) != 0)
        return false;

    ctx->OMGetRenderTargets(1, &views.rtv, nullptr);
    views.output = TextureFromView(views.rtv);
    if (views.output == nullptr)
    {
        if (!known)
            Reject(shader);
        return false;
    }
    const auto outDesc = Desc(views.output);
    if (!IsRgba8(outDesc.Format) || outDesc.SampleDesc.Count != 1 || outDesc.Width < kMinRenderSize ||
        outDesc.Height < kMinRenderSize || (outDesc.BindFlags & D3D11_BIND_SHADER_RESOURCE) == 0)
    {
        if (!known)
            Reject(shader);
        return false;
    }
    const UINT w = outDesc.Width;
    const UINT h = outDesc.Height;

    ctx->PSGetShaderResources(0, 5, views.srvs);
    D3D11_TEXTURE2D_DESC desc[5] {};
    const bool haveFirst = SrvDesc(views.srvs[0], desc[0]);
    // Only draws whose t0 is R8G8 are TAA-like; everything else with this shader is dismissed for good.
    if (!haveFirst || !IsRg8(desc[0].Format))
    {
        if (!known)
            Reject(shader);
        return false;
    }
    for (int i = 1; i < 5; ++i)
    {
        if (!SrvDesc(views.srvs[i], desc[i]))
        {
            if (!known)
                LogNearMiss(shader, "a texture in t1..t4 is missing");
            return false;
        }
    }
    const auto half = [&](const D3D11_TEXTURE2D_DESC& d)
    { return IsRg8(d.Format) && d.Width >= w / 2 && d.Width <= (w + 1) / 2 && d.Height >= h / 2 && d.Height <= (h + 1) / 2; };
    const auto full = [&](const D3D11_TEXTURE2D_DESC& d)
    { return IsRgba8(d.Format) && d.Width == w && d.Height == h && d.SampleDesc.Count == 1; };
    if (!half(desc[0]) || !half(desc[1]) || !full(desc[2]) || !full(desc[3]) || !full(desc[4]))
    {
        if (!known)
        {
            char why[256];
            snprintf(why, sizeof(why),
                     "rt0 %ux%u fmt %u; t0 %ux%u fmt %u, t1 %ux%u fmt %u, t2 %ux%u fmt %u, t3 %ux%u fmt %u, t4 %ux%u "
                     "fmt %u",
                     w, h, (UINT) outDesc.Format, desc[0].Width, desc[0].Height, (UINT) desc[0].Format, desc[1].Width,
                     desc[1].Height, (UINT) desc[1].Format, desc[2].Width, desc[2].Height, (UINT) desc[2].Format,
                     desc[3].Width, desc[3].Height, (UINT) desc[3].Format, desc[4].Width, desc[4].Height,
                     (UINT) desc[4].Format);
            LogNearMiss(shader, why);
        }
        return false;
    }
    D3D11_SHADER_RESOURCE_VIEW_DESC packedView {};
    views.srvs[0]->GetDesc(&packedView);
    if (packedView.Format != DXGI_FORMAT_R8G8_UNORM)
    {
        if (!known)
            LogNearMiss(shader, "t0 is not viewed as R8G8_UNORM");
        return false;
    }
    ctx->PSGetConstantBuffers(0, 1, &views.constants);
    if (views.constants == nullptr)
    {
        if (!known)
            LogNearMiss(shader, "no ps cb0");
        return false;
    }
    D3D11_BUFFER_DESC cbDesc {};
    views.constants->GetDesc(&cbDesc);
    if (cbDesc.ByteWidth < 96)
    {
        if (!known)
            LogNearMiss(shader, "ps cb0 smaller than 96 bytes");
        return false;
    }

    for (int i = 0; i < 2; ++i)
    {
        ID3D11Resource* resource = nullptr;
        views.srvs[i]->GetResource(&resource);
        g_mvTextures[i] = resource;
        if (resource != nullptr)
            resource->Release();
    }
    views.colour = TextureFromView(views.srvs[2]);
    views.width = w;
    views.height = h;
    if (!known && g_taaShaderCount < kMaxTaaShaders)
    {
        g_taaShaders[g_taaShaderCount++] = shader;
        LOG_INFO("FC5 TAA: TAA draw found (pixel shader {:X}, {} known) at {}x{}, motion {}x{}, ps cb0 {} bytes",
                 (size_t) shader, g_taaShaderCount, w, h, desc[0].Width, desc[0].Height, cbDesc.ByteWidth);
    }
    return views.colour != nullptr;
}

// ---- Shaders and state ----

bool g_setupTried = false;
bool g_ready = false;
// The context the TAA draw was first found on: FC5 creates several devices, and the spare state and every texture
// here belong to this one.
ID3D11DeviceContext* g_context = nullptr;
ID3D11ComputeShader* g_decode = nullptr;
ID3D11ComputeShader* g_copy = nullptr;
ID3D11ComputeShader* g_view = nullptr;
ID3D11Buffer* g_decodeConstants = nullptr;
ID3DDeviceContextState* g_spareState = nullptr;

ID3D11ComputeShader* Compile(ID3D11Device* device, const char* source, size_t length, const char* name)
{
    ID3DBlob* code = nullptr;
    ID3DBlob* errors = nullptr;
    if (FAILED(D3DCompile(source, length, name, nullptr, nullptr, "CSMain", "cs_5_0", D3DCOMPILE_OPTIMIZATION_LEVEL3,
                          0, &code, &errors)))
    {
        LOG_ERROR("FC5 TAA: {} compile failed: {}", name,
                  errors != nullptr ? (const char*) errors->GetBufferPointer() : "no message");
        if (errors != nullptr)
            errors->Release();
        return nullptr;
    }
    if (errors != nullptr)
        errors->Release();
    ID3D11ComputeShader* shader = nullptr;
    if (FAILED(device->CreateComputeShader(code->GetBufferPointer(), code->GetBufferSize(), nullptr, &shader)))
        shader = nullptr;
    code->Release();
    return shader;
}

// Once: shaders, the decode constants and a spare device context state (everything this module does runs inside
// it, so the game's pipeline state comes back exactly).
void Setup(ID3D11DeviceContext* ctx)
{
    g_setupTried = true;
    ctx->AddRef();
    g_context = ctx;
    ID3D11Device* device = nullptr;
    ctx->GetDevice(&device);
    if (device == nullptr)
        return;
    g_decode = Compile(device, kDecodeShader, sizeof(kDecodeShader) - 1, "Fc5Taa_Decode");
    g_copy = Compile(device, kCopyShader, sizeof(kCopyShader) - 1, "Fc5Taa_Copy");
    g_view = Compile(device, kViewShader, sizeof(kViewShader) - 1, "Fc5Taa_View");

    D3D11_BUFFER_DESC cbDesc {};
    cbDesc.ByteWidth = 16;
    cbDesc.Usage = D3D11_USAGE_DYNAMIC;
    cbDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    cbDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    if (FAILED(device->CreateBuffer(&cbDesc, nullptr, &g_decodeConstants)))
        g_decodeConstants = nullptr;

    ID3D11Device1* device1 = nullptr;
    if (SUCCEEDED(device->QueryInterface(IID_PPV_ARGS(&device1))) && device1 != nullptr)
    {
        const D3D_FEATURE_LEVEL level = device->GetFeatureLevel();
        const UINT flags = (device->GetCreationFlags() & D3D11_CREATE_DEVICE_SINGLETHREADED) != 0
                               ? D3D11_1_CREATE_DEVICE_CONTEXT_STATE_SINGLETHREADED
                               : 0;
        if (FAILED(device1->CreateDeviceContextState(flags, &level, 1, D3D11_SDK_VERSION, __uuidof(ID3D11Device),
                                                     nullptr, &g_spareState)))
            g_spareState = nullptr;
        device1->Release();
    }
    device->Release();

    g_ready = g_decode != nullptr && g_copy != nullptr && g_view != nullptr && g_decodeConstants != nullptr &&
              g_spareState != nullptr;
    if (!g_ready)
        LOG_ERROR("FC5 TAA: setup failed (shaders {}/{}/{}, spare state {}); staying off", g_decode != nullptr,
                  g_copy != nullptr, g_view != nullptr, g_spareState != nullptr);
}

// Runs `work` inside the spare device context state; the game's state comes back after.
template <typename Work> bool InSpareState(ID3D11DeviceContext* ctx, Work&& work)
{
    ID3D11DeviceContext1* ctx1 = nullptr;
    if (FAILED(ctx->QueryInterface(IID_PPV_ARGS(&ctx1))) || ctx1 == nullptr)
        return false;
    bool result = false;
    {
        PassScope scope;
        ID3DDeviceContextState* gameState = nullptr;
        ctx1->SwapDeviceContextState(g_spareState, &gameState);
        result = work();
        ID3DDeviceContextState* spare = nullptr;
        ctx1->SwapDeviceContextState(gameState, &spare);
        if (spare != nullptr)
            spare->Release(); // g_spareState keeps its own reference
        if (gameState != nullptr)
            gameState->Release();
    }
    ctx1->Release();
    return result;
}

void Dispatch(ID3D11DeviceContext* ctx, ID3D11ComputeShader* shader, UINT srvCount,
              ID3D11ShaderResourceView* const* srvs, UINT uavCount, ID3D11UnorderedAccessView* const* uavs,
              ID3D11Buffer* constants, UINT width, UINT height)
{
    ctx->CSSetShader(shader, nullptr, 0);
    ctx->CSSetShaderResources(0, srvCount, srvs);
    ctx->CSSetUnorderedAccessViews(0, uavCount, uavs, nullptr);
    if (constants != nullptr)
        ctx->CSSetConstantBuffers(0, 1, &constants);
    ctx->Dispatch((width + 7) / 8, (height + 7) / 8, 1);
    ID3D11ShaderResourceView* nullSrvs[5] = {};
    ID3D11UnorderedAccessView* nullUavs[2] = {};
    ctx->CSSetShaderResources(0, srvCount, nullSrvs);
    ctx->CSSetUnorderedAccessViews(0, uavCount, nullUavs, nullptr);
}

// ---- Motion textures (one set per settled size, parked and never released) ----

struct MotionTexture
{
    ID3D11Texture2D* texture = nullptr;
    ID3D11UnorderedAccessView* uav = nullptr;
    ID3D11ShaderResourceView* srv = nullptr;
};

struct MotionSet
{
    MotionTexture motion[2];
    MotionTexture composed;
    UINT width = 0;
    UINT height = 0;
};
MotionSet g_set;
std::vector<MotionSet> g_parked;
int g_current = 0;
int g_newSizes = 0;
// Sizes change during play now (Resolution Scale, R.E.A.L. VR's resolution): more of them in a session, and only the
// two parked last are kept for going back to; older ones are retired through the deferred list.
constexpr int kMaxNewSizes = 24;
constexpr size_t kMaxParked = 2;
constexpr uint64_t kSettlePresents = 45;
UINT g_candidateWidth = 0;
UINT g_candidateHeight = 0;
uint64_t g_candidateSince = 0;
uint64_t g_lastDecodePresent = UINT64_MAX;

bool CreateMotion(ID3D11Device* device, UINT width, UINT height, MotionTexture& out)
{
    D3D11_TEXTURE2D_DESC desc {};
    desc.Width = width;
    desc.Height = height;
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = DXGI_FORMAT_R16G16_FLOAT;
    desc.SampleDesc.Count = 1;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;
    return SUCCEEDED(device->CreateTexture2D(&desc, nullptr, &out.texture)) &&
           SUCCEEDED(device->CreateUnorderedAccessView(out.texture, nullptr, &out.uav)) &&
           SUCCEEDED(device->CreateShaderResourceView(out.texture, nullptr, &out.srv));
}

// Motion textures for this TAA size: the current set, a parked one, or new ones once the size has held
// kSettlePresents (the game steps through in-between sizes when the Resolution Scale changes).
bool EnsureMotion(ID3D11DeviceContext* ctx, UINT width, UINT height)
{
    if (g_set.width == width && g_set.height == height)
        return true;
    if (width != g_candidateWidth || height != g_candidateHeight)
    {
        g_candidateWidth = width;
        g_candidateHeight = height;
        g_candidateSince = g_present.load();
    }
    // The first size is taken straight away (nothing to settle from).
    if (g_set.width != 0 && g_present.load() - g_candidateSince < kSettlePresents)
        return false;

    MotionSet next;
    const auto parked = std::find_if(g_parked.begin(), g_parked.end(), [&](const MotionSet& set)
                                     { return set.width == width && set.height == height; });
    const bool reused = parked != g_parked.end();
    if (reused)
    {
        next = *parked;
        g_parked.erase(parked);
    }
    else
    {
        if (g_newSizes >= kMaxNewSizes)
            return false;
        ++g_newSizes;
        ID3D11Device* device = nullptr;
        ctx->GetDevice(&device);
        if (device == nullptr)
            return false;
        const bool ok = CreateMotion(device, width, height, next.motion[0]) &&
                        CreateMotion(device, width, height, next.motion[1]) &&
                        CreateMotion(device, width, height, next.composed);
        device->Release();
        if (!ok)
        {
            LOG_ERROR("FC5 TAA: motion textures at {}x{} failed", width, height);
            g_newSizes = kMaxNewSizes; // don't retry every frame (the partial set is leaked, not released live)
            return false;
        }
        next.width = width;
        next.height = height;
    }
    if (g_set.width != 0)
        g_parked.push_back(g_set);
    while (g_parked.size() > kMaxParked)
    {
        const uint64_t now = g_present.load();
        for (MotionTexture* t : { &g_parked.front().motion[0], &g_parked.front().motion[1], &g_parked.front().composed })
        {
            for (IUnknown* p : { (IUnknown*) t->srv, (IUnknown*) t->uav, (IUnknown*) t->texture })
                if (p != nullptr)
                    g_deferred.push_back({ p, now });
        }
        LOG_INFO("FC5 TAA: motion textures {}x{} retired (more than {} sizes parked)", g_parked.front().width,
                 g_parked.front().height, kMaxParked);
        g_parked.erase(g_parked.begin());
    }
    LOG_INFO("FC5 TAA: motion at {}x{} ({} textures{})", width, height, reused ? "earlier" : "new",
             g_set.width != 0 ? ", size changed" : "");
    g_set = next;
    g_lastDecodePresent = UINT64_MAX;
    return true;
}

// ---- Depth: SRVs on the game's depth, and copies of it for the previous Present ----

DXGI_FORMAT DepthViewFormat(DXGI_FORMAT format)
{
    switch (format)
    {
    case DXGI_FORMAT_R24G8_TYPELESS:
    case DXGI_FORMAT_D24_UNORM_S8_UINT:
        return DXGI_FORMAT_R24_UNORM_X8_TYPELESS;
    case DXGI_FORMAT_R32_TYPELESS:
    case DXGI_FORMAT_D32_FLOAT:
        return DXGI_FORMAT_R32_FLOAT;
    case DXGI_FORMAT_R32G8X24_TYPELESS:
    case DXGI_FORMAT_D32_FLOAT_S8X24_UINT:
        return DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS;
    default:
        return DXGI_FORMAT_UNKNOWN;
    }
}

ID3D11ShaderResourceView* CreateDepthSrv(ID3D11Device* device, ID3D11Texture2D* texture, DXGI_FORMAT format)
{
    D3D11_SHADER_RESOURCE_VIEW_DESC desc {};
    desc.Format = DepthViewFormat(format);
    desc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
    desc.Texture2D.MipLevels = 1;
    ID3D11ShaderResourceView* srv = nullptr;
    if (desc.Format == DXGI_FORMAT_UNKNOWN || FAILED(device->CreateShaderResourceView(texture, &desc, &srv)))
        return nullptr;
    return srv;
}

// One view per game depth texture, kept (each holds its texture).
struct DepthView
{
    ID3D11Texture2D* texture = nullptr;
    ID3D11ShaderResourceView* srv = nullptr;
};
constexpr int kMaxDepthViews = 12;
DepthView g_depthViews[kMaxDepthViews];

ID3D11ShaderResourceView* DepthSrvFor(ID3D11Device* device, ID3D11Texture2D* texture)
{
    for (auto& view : g_depthViews)
    {
        if (view.texture == texture)
            return view.srv;
    }
    const auto desc = Desc(texture);
    if ((desc.BindFlags & D3D11_BIND_SHADER_RESOURCE) == 0 || desc.SampleDesc.Count != 1)
        return nullptr;
    for (auto& view : g_depthViews)
    {
        if (view.texture != nullptr)
            continue;
        view.srv = CreateDepthSrv(device, texture, desc.Format);
        if (view.srv == nullptr)
            return nullptr;
        view.texture = texture;
        return view.srv;
    }
    return nullptr;
}

struct DepthCopy
{
    ID3D11Texture2D* texture = nullptr;
    ID3D11ShaderResourceView* srv = nullptr;
    UINT width = 0;
    UINT height = 0;
    DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;
    uint64_t present = UINT64_MAX;
};
DepthCopy g_depthCopies[2];
std::vector<DepthCopy> g_parkedDepth;

bool EnsureDepthCopy(ID3D11Device* device, DepthCopy& copy, ID3D11Texture2D* depth)
{
    const auto desc = Desc(depth);
    const auto matches = [&](const DepthCopy& c)
    { return c.texture != nullptr && c.width == desc.Width && c.height == desc.Height && c.format == desc.Format; };
    if (matches(copy))
        return true;
    const auto parked = std::find_if(g_parkedDepth.begin(), g_parkedDepth.end(), matches);
    if (parked != g_parkedDepth.end())
    {
        DepthCopy reused = *parked;
        g_parkedDepth.erase(parked);
        if (copy.texture != nullptr)
            g_parkedDepth.push_back(copy);
        copy = reused;
        copy.present = UINT64_MAX;
        return true;
    }
    static int created = 0;
    if (created >= 2 * (kMaxNewSizes + 1))
        return false;
    ++created;
    if (copy.texture != nullptr)
        g_parkedDepth.push_back(copy);
    copy = {};
    auto copyDesc = desc;
    copyDesc.CPUAccessFlags = 0;
    copyDesc.MiscFlags = 0;
    copyDesc.Usage = D3D11_USAGE_DEFAULT;
    if (FAILED(device->CreateTexture2D(&copyDesc, nullptr, &copy.texture)))
    {
        copy.texture = nullptr;
        return false;
    }
    copy.srv = CreateDepthSrv(device, copy.texture, desc.Format);
    copy.width = desc.Width;
    copy.height = desc.Height;
    copy.format = desc.Format;
    return copy.srv != nullptr;
}

// ---- Stale-eye depth share ([Inputs] StaleEyeCameraShare + StaleEyeDepthShare) ----
//
// Each Present's scene depth (the G-buffer depth at the TAA size), copied at the TAA draw into one of
// kStaleEyeDepthSlots shared R32_FLOAT textures (slot = the camera publication's sequence % kStaleEyeDepthSlots;
// several, because the submitted eye image lags the Present and the layer reads depth several Presents old) for the stale-eye warp in the OpenXR layer, which
// runs on another D3D11 device (not the game's). GPU-side sync only: this
// context signals the written fence with the sequence after the copy; the layer signals the read fence after its
// reads of a slot. Before reusing a slot this context waits on the read fence, but ONLY for a value the layer has
// reported queued (StaleEyeShare::ReportedRead), so a stopped or absent layer can never hang the game's GPU.
// Resources of an old size, and their NT handles, are retired after kReleaseAfterPresents, never at once: the layer
// may be opening a handle on its own thread, and a closed handle's value can be reused by another object.
constexpr char kDepthShareShader[] = R"(
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

struct DepthShareSet
{
    ID3D11Texture2D* textures[kStaleEyeDepthSlots] = {};
    ID3D11UnorderedAccessView* uavs[kStaleEyeDepthSlots] = {};
    HANDLE handles[kStaleEyeDepthSlots] = {};
    UINT width = 0;
    UINT height = 0;
};
DepthShareSet g_depthShare;
struct RetiredHandle
{
    HANDLE handle = nullptr;
    uint64_t present = 0;
};
std::vector<RetiredHandle> g_retiredHandles;
ID3D11Fence* g_writtenFence = nullptr;
ID3D11Fence* g_readFence = nullptr;
HANDLE g_writtenFenceHandle = nullptr;
HANDLE g_readFenceHandle = nullptr;
ID3D11ComputeShader* g_depthShareShader = nullptr;
bool g_depthShareSetupTried = false;
bool g_depthShareReady = false;
uint64_t g_depthShareGeneration = 0;
int g_depthShareSizes = 0;
uint64_t g_lastSharedSequence = 0;
uint64_t g_depthShares = 0;
uint64_t g_depthShareUnsyncedReuse = 0;
// The TAA size must have been the same for this many Presents before any depth is shared (start-up, loading and
// Resolution Scale changes step through other sizes; work in the first frames at a new size lost FC5's device).
constexpr uint64_t kDepthShareSettlePresents = 90;
UINT g_depthShareCandidate[2] = {};
uint64_t g_depthShareCandidateSince = 0;
// Noted at the TAA draw, done in AfterDraw: work queued BEFORE FC5's TAA draw that then runs loses the device
// (887A0005 18 ms after the first share, at a settled eye size -- the same as the DLSS-before-TAA
// trap, whose fix was "skip the game's pass, or run after it").
struct PendingShare
{
    bool set = false;
    UINT width = 0;
    UINT height = 0;
    uint64_t present = 0;
};
PendingShare g_pendingShare;

void DepthShareSkip(int reason, const char* text)
{
    static std::atomic<uint32_t> logged { 0 };
    const uint32_t bit = 1u << reason;
    if ((logged.fetch_or(bit) & bit) == 0)
        LOG_INFO("Stale-eye depth share: not shared this Present: {}", text);
}

void SetupDepthShare(ID3D11Device* device)
{
    g_depthShareSetupTried = true;
    ID3D11Device5* device5 = nullptr;
    if (FAILED(device->QueryInterface(IID_PPV_ARGS(&device5))) || device5 == nullptr)
    {
        LOG_ERROR("Stale-eye depth share: no ID3D11Device5 (fences); off");
        return;
    }
    bool ok = SUCCEEDED(device5->CreateFence(0, D3D11_FENCE_FLAG_SHARED, IID_PPV_ARGS(&g_writtenFence))) &&
              SUCCEEDED(device5->CreateFence(0, D3D11_FENCE_FLAG_SHARED, IID_PPV_ARGS(&g_readFence))) &&
              SUCCEEDED(g_writtenFence->CreateSharedHandle(nullptr, GENERIC_ALL, nullptr, &g_writtenFenceHandle)) &&
              SUCCEEDED(g_readFence->CreateSharedHandle(nullptr, GENERIC_ALL, nullptr, &g_readFenceHandle));
    device5->Release();
    g_depthShareShader = ok ? Compile(device, kDepthShareShader, sizeof(kDepthShareShader) - 1, "Fc5Taa_DepthShare")
                            : nullptr;
    g_depthShareReady = ok && g_depthShareShader != nullptr;
    if (g_depthShareReady)
        LOG_INFO("Stale-eye depth share: ready (fences {:X} / {:X})", (size_t) g_writtenFenceHandle,
                 (size_t) g_readFenceHandle);
    else
        LOG_ERROR("Stale-eye depth share: setup failed (fences {}, shader {}); off", ok, g_depthShareShader != nullptr);
}

bool EnsureDepthShareSize(ID3D11Device* device, UINT width, UINT height)
{
    if (g_depthShare.width == width && g_depthShare.height == height && g_depthShare.textures[0] != nullptr)
        return true;
    if (g_depthShareSizes >= 24) // the old size's textures are retired below, so this only bounds the churn
    {
        DepthShareSkip(7, "more than 24 sizes; no new shared textures (restart to reset)");
        return false;
    }
    ++g_depthShareSizes;
    // Retire the old size: textures and views through the deferred list, handles after the same delay.
    const uint64_t now = g_present.load();
    for (int i = 0; i < kStaleEyeDepthSlots; ++i)
    {
        if (g_depthShare.uavs[i] != nullptr)
            g_deferred.push_back({ g_depthShare.uavs[i], now });
        if (g_depthShare.textures[i] != nullptr)
            g_deferred.push_back({ g_depthShare.textures[i], now });
        if (g_depthShare.handles[i] != nullptr)
            g_retiredHandles.push_back({ g_depthShare.handles[i], now });
    }
    g_depthShare = {};
    D3D11_TEXTURE2D_DESC desc {};
    desc.Width = width;
    desc.Height = height;
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = DXGI_FORMAT_R32_FLOAT;
    desc.SampleDesc.Count = 1;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
    desc.MiscFlags = D3D11_RESOURCE_MISC_SHARED | D3D11_RESOURCE_MISC_SHARED_NTHANDLE;
    for (int i = 0; i < kStaleEyeDepthSlots; ++i)
    {
        IDXGIResource1* dxgi = nullptr;
        if (FAILED(device->CreateTexture2D(&desc, nullptr, &g_depthShare.textures[i])) ||
            FAILED(device->CreateUnorderedAccessView(g_depthShare.textures[i], nullptr, &g_depthShare.uavs[i])) ||
            FAILED(g_depthShare.textures[i]->QueryInterface(IID_PPV_ARGS(&dxgi))) ||
            FAILED(dxgi->CreateSharedHandle(nullptr, DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE, nullptr,
                                            &g_depthShare.handles[i])))
        {
            if (dxgi != nullptr)
                dxgi->Release();
            LOG_ERROR("Stale-eye depth share: shared texture {} ({}x{}) failed; off for this size", i, width, height);
            return false;
        }
        dxgi->Release();
    }
    g_depthShare.width = width;
    g_depthShare.height = height;
    StaleEyeShare::DepthShareInfo info;
    info.generation = ++g_depthShareGeneration;
    for (int i = 0; i < kStaleEyeDepthSlots; ++i)
        info.handles[i] = reinterpret_cast<uint64_t>(g_depthShare.handles[i]);
    info.width = width;
    info.height = height;
    info.writtenFence = reinterpret_cast<uint64_t>(g_writtenFenceHandle);
    info.readFence = reinterpret_cast<uint64_t>(g_readFenceHandle);
    StaleEyeShare::SetDepthShare(info);
    LOG_INFO("Stale-eye depth share: {} x {}x{} shared textures, generation {}", kStaleEyeDepthSlots, width, height,
             info.generation);
    return true;
}

// The video memory of the whole process (DXGI counts per process and adapter: the game, the VR mod, DLSS's own
// D3D12 device and the stale-eye layer together). False when it cannot be read. The ADAPTER is kept (it is DXGI's,
// not a resource of the game's: nothing the game frees or resizes depends on it); device may be null once it is.
IDXGIAdapter3* g_videoAdapter = nullptr;
bool VideoMemoryMb(ID3D11Device* device, double& usedMb, double& budgetMb)
{
    IDXGIDevice* dxgi = nullptr;
    if (g_videoAdapter == nullptr && device != nullptr && SUCCEEDED(device->QueryInterface(IID_PPV_ARGS(&dxgi))) &&
        dxgi != nullptr)
    {
        IDXGIAdapter* adapter = nullptr;
        if (SUCCEEDED(dxgi->GetAdapter(&adapter)) && adapter != nullptr)
        {
            adapter->QueryInterface(IID_PPV_ARGS(&g_videoAdapter));
            adapter->Release();
        }
        dxgi->Release();
    }
    DXGI_QUERY_VIDEO_MEMORY_INFO info {};
    if (g_videoAdapter == nullptr ||
        FAILED(g_videoAdapter->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &info)))
        return false;
    usedMb = (double) info.CurrentUsage / (1024.0 * 1024.0);
    budgetMb = (double) info.Budget / (1024.0 * 1024.0);
    return true;
}

// At the TAA draw, every Present (both F6 states): the G-buffer depth of this Present into the shared slot of this
// Present's camera publication. Nothing at all at other sizes (menus, loading, flat frames).
void ShareDepth(ID3D11DeviceContext* ctx, UINT width, UINT height)
{
    if (!StaleEyeShare::Enabled() || !Config::Instance()->StaleEyeDepthShare.value_or_default())
        return;
    // Size stability is tracked every Present, so it is known when the layer asks.
    if (width != g_depthShareCandidate[0] || height != g_depthShareCandidate[1])
    {
        g_depthShareCandidate[0] = width;
        g_depthShareCandidate[1] = height;
        g_depthShareCandidateSince = g_present.load();
    }
    if (!StaleEyeShare::DepthRequested())
        return DepthShareSkip(8, "the layer does not request depth (Pause to DEPTH in the headset)");
    if (g_present.load() - g_depthShareCandidateSince < kDepthShareSettlePresents)
        return DepthShareSkip(9, "the TAA size has not been stable for 90 Presents yet");
    for (size_t i = 0; i < g_retiredHandles.size();)
    {
        if (g_present.load() - g_retiredHandles[i].present >= kReleaseAfterPresents)
        {
            CloseHandle(g_retiredHandles[i].handle);
            g_retiredHandles[i] = g_retiredHandles.back();
            g_retiredHandles.pop_back();
        }
        else
        {
            ++i;
        }
    }
    uint32_t cameraWidth = 0, cameraHeight = 0;
    const uint64_t sequence = StaleEyeShare::LatestSequence(&cameraWidth, &cameraHeight);
    if (sequence == 0 || sequence == g_lastSharedSequence)
        return DepthShareSkip(0, "no new camera publication this Present (no motion pass yet?)");
    if (cameraWidth != width || cameraHeight != height)
        return DepthShareSkip(1, "the camera matrix is not at the TAA size");
    if (g_depth == nullptr)
        return DepthShareSkip(2, "no G-buffer depth seen");
    const auto depthDesc = Desc(g_depth);
    if (depthDesc.Width != width || depthDesc.Height != height)
        return DepthShareSkip(3, "the G-buffer depth is not at the TAA size");

    ID3D11Device* device = nullptr;
    ctx->GetDevice(&device);
    if (device == nullptr)
        return;
    if (!g_depthShareSetupTried)
        SetupDepthShare(device);
    ID3D11ShaderResourceView* depthSrv = g_depthShareReady ? DepthSrvFor(device, g_depth) : nullptr;
    const bool sized = g_depthShareReady && depthSrv != nullptr && EnsureDepthShareSize(device, width, height);
    device->Release();
    if (!g_depthShareReady)
        return;
    if (depthSrv == nullptr)
        return DepthShareSkip(4, "no SRV on the G-buffer depth");
    if (!sized)
        return;
    ID3D11DeviceContext4* ctx4 = nullptr;
    if (FAILED(ctx->QueryInterface(IID_PPV_ARGS(&ctx4))) || ctx4 == nullptr)
        return DepthShareSkip(5, "no ID3D11DeviceContext4 (fences)");

    const int slot = (int) (sequence % kStaleEyeDepthSlots);
    // The slot last held sequence - kStaleEyeDepthSlots; wait until the layer's reads of it are done, if it reported
    // them queued.
    if (sequence > (uint64_t) kStaleEyeDepthSlots)
    {
        const uint64_t lastUse = sequence - kStaleEyeDepthSlots;
        if (StaleEyeShare::ReportedRead() >= lastUse)
            ctx4->Wait(g_readFence, lastUse);
        else if (StaleEyeShare::ReportedRead() != 0)
            ++g_depthShareUnsyncedReuse;
    }
    ID3D11UnorderedAccessView* uav = g_depthShare.uavs[slot];
    InSpareState(ctx, [&] {
        Dispatch(ctx, g_depthShareShader, 1, &depthSrv, 1, &uav, nullptr, width, height);
        return true;
    });
    ctx4->Signal(g_writtenFence, sequence);
    ctx4->Release();
    StaleEyeShare::SetDepthSequence(sequence);
    g_lastSharedSequence = sequence;
    if (++g_depthShares == 1 || g_depthShares % 5400 == 0)
        LOG_INFO("Stale-eye depth share: {} Presents shared (sequence {}, slot {}, {}x{}); reused unsynced {} (layer "
                 "reported {})",
                 g_depthShares, sequence, slot, width, height, g_depthShareUnsyncedReuse, StaleEyeShare::ReportedRead());
}

// ---- UAVs on the game's TAA outputs (RGBA8, typeless), one per texture ----
// Kept while the game uses that texture. Sizes change during play, and each size brings its own outputs: a full table
// retires what is in it (through the deferred list; each entry also holds the game's old texture alive) instead of
// refusing the new one. Refusing was silent for DLSS below 100%, which does not write the TAA output, and at the next
// 100% DLAA could not write its picture: the draw fell through to the game's with our passes queued before it, and
// the device was lost 13 ms later.

struct TargetUav
{
    ID3D11Texture2D* texture = nullptr;
    ID3D11UnorderedAccessView* uav = nullptr;
};
constexpr int kMaxTargets = 8;
TargetUav g_targets[kMaxTargets];

ID3D11UnorderedAccessView* TargetUavFor(ID3D11Device* device, ID3D11Texture2D* texture)
{
    for (auto& target : g_targets)
    {
        if (target.texture == texture)
            return target.uav;
    }
    static bool loggedBad = false;
    const auto desc = Desc(texture);
    if ((desc.BindFlags & D3D11_BIND_UNORDERED_ACCESS) == 0 || desc.Format == DXGI_FORMAT_R8G8B8A8_UNORM_SRGB)
    {
        if (!loggedBad)
        {
            loggedBad = true;
            LOG_WARN("FC5 TAA: the TAA output (format {}, bind {:X}) can't take a UAV", (UINT) desc.Format,
                     desc.BindFlags);
        }
        return nullptr;
    }
    for (auto& target : g_targets)
    {
        if (target.texture != nullptr)
            continue;
        D3D11_UNORDERED_ACCESS_VIEW_DESC uavDesc {};
        uavDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        uavDesc.ViewDimension = D3D11_UAV_DIMENSION_TEXTURE2D;
        ID3D11UnorderedAccessView* uav = nullptr;
        if (FAILED(device->CreateUnorderedAccessView(texture, &uavDesc, &uav)))
        {
            if (!loggedBad)
            {
                loggedBad = true;
                LOG_WARN("FC5 TAA: UAV on the TAA output failed");
            }
            return nullptr;
        }
        texture->AddRef();
        target.texture = texture;
        target.uav = uav;
        return uav;
    }
    // Full: the outputs of earlier sizes. All of them go; the ones still in use come back on their next draw.
    {
        const uint64_t now = g_present.load();
        static int retirements = 0;
        if (retirements++ < 10)
            LOG_INFO("FC5 TAA: {} UAVs on TAA outputs of earlier sizes retired for {}x{}", kMaxTargets, desc.Width,
                     desc.Height);
        for (auto& target : g_targets)
        {
            if (target.uav != nullptr)
                g_deferred.push_back({ target.uav, now });
            if (target.texture != nullptr)
                g_deferred.push_back({ target.texture, now });
            target = {};
        }
    }
    D3D11_UNORDERED_ACCESS_VIEW_DESC uavDesc {};
    uavDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    uavDesc.ViewDimension = D3D11_UAV_DIMENSION_TEXTURE2D;
    ID3D11UnorderedAccessView* uav = nullptr;
    if (FAILED(device->CreateUnorderedAccessView(texture, &uavDesc, &uav)))
    {
        if (!loggedBad)
        {
            loggedBad = true;
            LOG_WARN("FC5 TAA: UAV on the TAA output failed");
        }
        return nullptr;
    }
    texture->AddRef();
    g_targets[0].texture = texture;
    g_targets[0].uav = uav;
    return uav;
}

// ---- Jitter: ps cb0 +80 (the jitter parameters) read back a few Presents late, by Present parity ----

constexpr UINT kJitterOffset = 80;
constexpr int kJitterSlots = 4;

struct JitterSlot
{
    ID3D11Buffer* staging = nullptr;
    UINT bytes = 0;
    uint64_t present = 0;
    bool pending = false;
};
JitterSlot g_jitterSlots[kJitterSlots];

// The image's shift in pixels (x right, y down) for even / odd Presents.
float g_jitter[2][2] = {};
bool g_jitterKnown[2] = {};
uint64_t g_jitterReads = 0;
uint64_t g_jitterChanges = 0;
uint64_t g_jitterInvalid = 0;

void ReadJitter(ID3D11DeviceContext* ctx, uint64_t present)
{
    for (auto& slot : g_jitterSlots)
    {
        if (!slot.pending || present < slot.present + 2)
            continue;
        D3D11_MAPPED_SUBRESOURCE map {};
        const HRESULT result = ctx->Map(slot.staging, 0, D3D11_MAP_READ, D3D11_MAP_FLAG_DO_NOT_WAIT, &map);
        if (result == DXGI_ERROR_WAS_STILL_DRAWING)
            continue;
        slot.pending = false;
        if (FAILED(result))
            continue;
        float values[3] = {};
        std::memcpy(values, static_cast<const char*>(map.pData) + kJitterOffset, sizeof(values));
        ctx->Unmap(slot.staging, 0);
        if (!(values[0] >= 0.0f && values[0] <= 1.0f && values[1] >= 0.0f && values[1] <= 1.0f))
        {
            if (g_jitterInvalid++ == 0)
                LOG_WARN("FC5 TAA: ps cb0 +{} isn't the jitter layout ({} {} {})", kJitterOffset, values[0], values[1],
                         values[2]);
            continue;
        }
        ++g_jitterReads;
        const int parity = (int) (slot.present & 1);
        const float x = 0.5f - values[0];
        const float y = 0.5f - values[1];
        if (g_jitterKnown[parity] && (x != g_jitter[parity][0] || y != g_jitter[parity][1]))
            ++g_jitterChanges;
        if (!g_jitterKnown[parity])
            LOG_INFO("FC5 TAA: jitter on {} Presents ({:+.3f}, {:+.3f}) px (phase {})", parity ? "odd" : "even", x, y,
                     values[2]);
        g_jitter[parity][0] = x;
        g_jitter[parity][1] = y;
        g_jitterKnown[parity] = true;
    }
}

void QueueJitterCopy(ID3D11DeviceContext* ctx, ID3D11Buffer* buffer, uint64_t present)
{
    D3D11_BUFFER_DESC desc {};
    buffer->GetDesc(&desc);
    auto& slot = g_jitterSlots[present % kJitterSlots];
    if (desc.ByteWidth < kJitterOffset + 16 || slot.pending)
        return;
    if (slot.staging != nullptr && slot.bytes != desc.ByteWidth)
    {
        slot.staging->Release(); // a CPU readback buffer, not used by the GPU unless pending
        slot.staging = nullptr;
    }
    if (slot.staging == nullptr)
    {
        D3D11_BUFFER_DESC stagingDesc {};
        stagingDesc.ByteWidth = desc.ByteWidth;
        stagingDesc.Usage = D3D11_USAGE_STAGING;
        stagingDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        ID3D11Device* device = nullptr;
        ctx->GetDevice(&device);
        if (device != nullptr)
        {
            if (FAILED(device->CreateBuffer(&stagingDesc, nullptr, &slot.staging)))
                slot.staging = nullptr;
            device->Release();
        }
        slot.bytes = desc.ByteWidth;
    }
    if (slot.staging != nullptr)
    {
        PassScope scope;
        ctx->CopyResource(slot.staging, buffer);
        slot.present = present;
        slot.pending = true;
    }
}

// ---- DLSS: two features (one per eye) per set, created once on first use: DLAA at the render size (100%), or
// upscaling from the render size to the eye size (below 100%) ----

struct Eye
{
    NVSDK_NGX_Handle* handle = nullptr;
    uint64_t lastPresent = UINT64_MAX;
};

struct Features
{
    const char* name;
    Eye eyes[2];
    bool createTried = false;
    bool ready = false;
    UINT width = 0; // input (render size)
    UINT height = 0;
    UINT outWidth = 0;
    UINT outHeight = 0;
    ID3D11Texture2D* output = nullptr;
    ID3D11ShaderResourceView* outputSrv = nullptr;
    int rebuilds = 0;
};
Features g_dlaa { "DLAA" };
Features g_upscale { "upscaling" };

// ---- Sizes that change during play (FC6 and Days Gone had it, FC5 said "restart to match") ----
// The Resolution Scale slider and R.E.A.L. VR's resolution change the render and the eye size. The features follow,
// but only a size that is GAMEPLAY's and has HELD: the same render and eye size for kSettlePasses Presents in a row,
// each with the G-buffer bound at the render size and the game's upscale draw seen the Present before. Menus (the
// eye's size, no G-buffer), loading screens and the flat frames of the switch into VR (other sizes, another shape)
// never settle, and nothing at all is done at their sizes -- work at a new size lost the device twice.
// Followed in every mode, so the first F6 finds the size settled already.
constexpr uint64_t kSettlePasses = 90;
struct WantedSize
{
    UINT size[4] = {}; // render width, height, output width, height
    uint64_t since = 0;
    uint64_t lastPresent = UINT64_MAX;
    bool refusalLogged = false, waitLogged = false, stuckLogged = false;
    uint64_t waited = 0;           // TAA draws that found this size not settled
    bool depthHere = false, eyeSeen = false; // the two halves of the gameplay signature at the last draw
};
WantedSize g_wanted;
// Set by the overlay ("Rebuild DLSS now", a preset change), taken at the next TAA draw on the game's context.
std::atomic<bool> g_rebuildRequested { false };
// DLSS switched off (F6): its features are freed once it has stayed off for kFreeAfterOff Presents (they hold
// 2.8 to 5.5 GB of video memory). Not at once: F6 back and forth to compare the two pictures stays instant, and
// each release is a hitch of a second. UINT64_MAX while DLSS is on.
constexpr uint64_t kFreeAfterOff = 900;
uint64_t g_offSince = UINT64_MAX;
std::mutex g_statusMutex;
std::string g_statusText = "DLSS: not started";

void SetStatus(std::string text)
{
    std::lock_guard lock(g_statusMutex);
    g_statusText = std::move(text);
}

// True once this Present's sizes have held, in gameplay, for kSettlePasses Presents.
bool SizeSettled(const UINT want[4], uint64_t present, bool gameplay)
{
    auto& w = g_wanted;
    if (!gameplay || std::memcmp(w.size, want, sizeof(w.size)) != 0 || w.lastPresent == UINT64_MAX ||
        w.lastPresent + 1 != present)
    {
        if (std::memcmp(w.size, want, sizeof(w.size)) != 0)
        {
            w.refusalLogged = w.waitLogged = w.stuckLogged = false;
            w.waited = 0;
        }
        std::memcpy(w.size, want, sizeof(w.size));
        w.since = present;
    }
    w.lastPresent = gameplay ? present : UINT64_MAX;
    return gameplay && present - w.since >= kSettlePasses;
}
// The render size of the features made (the own jitter only touches that viewport) and the number of jitter phases
// per eye: 8 for DLAA, 8 x the area ratio when upscaling (DLSS's guidance).
std::atomic<UINT> g_jitterWidth { 0 };
std::atomic<UINT> g_jitterHeight { 0 };
std::atomic<int> g_jitterSamples { 8 };

// ---- The game's upscale draw (the first draw of a Present reading its TAA output into a render-target-only
// target): its eye size, and whether this Present's upscaling output waits for it ----
const void* g_taaOutput = nullptr; // this Present's TAA output, compared by address only
uint64_t g_taaOutputPresent = UINT64_MAX;
uint64_t g_upscaleDrawPresent = UINT64_MAX;
uint64_t g_upscalePendingPresent = UINT64_MAX;
UINT g_eyeWidth = 0;
UINT g_eyeHeight = 0;
uint64_t g_drawsReplaced = 0;
// The upscaling output in the eye target's format (made on the first replacement), copied over the eye target.
ID3D11Texture2D* g_eyeCopy = nullptr;
ID3D11UnorderedAccessView* g_eyeCopyUav = nullptr;
bool g_eyeCopyTried = false;
// The eye target to overwrite right after the game's upscale draw (AddRef'd), set at that draw.
ID3D11Texture2D* g_overwriteTarget = nullptr;
NVSDK_NGX_Parameter* g_params = nullptr;

int g_nextEye = 0;
uint64_t g_lastReplacedPresent = UINT64_MAX;

bool InitNgx(ID3D11Device* device)
{
    auto& state = State::Instance();
    if (state.nvngxDx11Inited)
        return true;
    NVSDK_NGX_FeatureCommonInfo info {};
    const auto exePath = Util::ExePath().remove_filename();
    const auto result = NVSDK_NGX_D3D11_Init_with_ProjectID(
        OPTI_GUID, state.NVNGX_Engine, OPTI_VERSION, exePath.c_str(), device, &info,
        state.NVNGX_Version == 0 ? NVSDK_NGX_Version_API : state.NVNGX_Version);
    if (result != NVSDK_NGX_Result_Success)
    {
        LOG_ERROR("FC5 TAA: NGX init failed: {:X}", (UINT) result);
        return false;
    }
    return true;
}

NVSDK_NGX_PerfQuality_Value QualityFor(UINT in, UINT out)
{
    const double ratio = (double) in / (double) out;
    if (in >= out)
        return NVSDK_NGX_PerfQuality_Value_DLAA;
    if (ratio >= 0.62)
        return NVSDK_NGX_PerfQuality_Value_MaxQuality;
    if (ratio >= 0.55)
        return NVSDK_NGX_PerfQuality_Value_Balanced;
    if (ratio >= 0.45)
        return NVSDK_NGX_PerfQuality_Value_MaxPerf;
    return NVSDK_NGX_PerfQuality_Value_UltraPerformance;
}

// Frees a pair so that the next TAA draw makes it again at that draw's sizes. On the game's context only (the thread
// that creates and evaluates the features). The output texture goes through the deferred list: work recorded a moment
// ago may still read it. OptiScaler's ReleaseFeature waits for the GPU itself (about half a second per feature).
void ReleaseSet(Features& set, const char* why)
{
    PassScope scope;
    const auto start = std::chrono::steady_clock::now();
    int freed = 0;
    for (auto& eye : set.eyes)
    {
        if (eye.handle != nullptr)
        {
            NVSDK_NGX_D3D11_ReleaseFeature(eye.handle);
            ++freed;
        }
        eye.handle = nullptr;
        eye.lastPresent = UINT64_MAX;
    }
    const uint64_t now = g_present.load();
    if (set.outputSrv != nullptr)
        g_deferred.push_back({ set.outputSrv, now });
    if (set.output != nullptr)
        g_deferred.push_back({ set.output, now });
    LOG_INFO("FC5 TAA: {} features {}x{} -> {}x{} released ({} features, {:.0f} ms): {}", set.name, set.width,
             set.height, set.outWidth, set.outHeight, freed,
             std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count(), why);
    set.outputSrv = nullptr;
    set.output = nullptr;
    set.createTried = false;
    set.ready = false;
    set.width = set.height = set.outWidth = set.outHeight = 0;
}

// Inside the spare state.
void CreateDlss(ID3D11DeviceContext* ctx, Features& set, UINT width, UINT height, UINT outWidth, UINT outHeight)
{
    set.createTried = true;
    ID3D11Device* device = nullptr;
    ctx->GetDevice(&device);
    if (device == nullptr)
        return;
    D3D11_TEXTURE2D_DESC desc {};
    desc.Width = outWidth;
    desc.Height = outHeight;
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
    desc.SampleDesc.Count = 1;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;
    // COST: what the features take (read again after the first 900 evaluations, in LogStats: DLSS allocates at use).
    double usedBefore = 0, budgetBefore = 0;
    if (VideoMemoryMb(device, usedBefore, budgetBefore))
        LOG_INFO("FC5 TAA: video memory before creating the {} features ({}x{} -> {}x{}): the process uses {:.0f} MB of "
                 "a budget of {:.0f} MB",
                 set.name, width, height, outWidth, outHeight, usedBefore, budgetBefore);
    bool ok = SUCCEEDED(device->CreateTexture2D(&desc, nullptr, &set.output)) &&
              SUCCEEDED(device->CreateShaderResourceView(set.output, nullptr, &set.outputSrv)) && InitNgx(device);
    device->Release();
    if (ok && g_params == nullptr && NVSDK_NGX_D3D11_GetCapabilityParameters(&g_params) != NVSDK_NGX_Result_Success)
    {
        LOG_ERROR("FC5 TAA: GetCapabilityParameters failed");
        ok = false;
    }
    const bool inverted = Config::Instance()->DuniaTaaDepthInverted.value_or_default();
    const auto quality = QualityFor(width, outWidth);
    for (int eye = 0; ok && eye < 2; ++eye)
    {
        // The picture is tonemapped 8-bit: no IsHDR, so NR also leaves its colour transform off.
        UINT flags = NVSDK_NGX_DLSS_Feature_Flags_AutoExposure;
        // Motion vectors are in render pixels.
        if (width < outWidth)
            flags |= NVSDK_NGX_DLSS_Feature_Flags_MVLowRes;
        if (inverted)
            flags |= NVSDK_NGX_DLSS_Feature_Flags_DepthInverted;
        g_params->Set(NVSDK_NGX_Parameter_DLSS_Feature_Create_Flags, flags);
        g_params->Set(NVSDK_NGX_Parameter_Width, width);
        g_params->Set(NVSDK_NGX_Parameter_Height, height);
        g_params->Set(NVSDK_NGX_Parameter_OutWidth, outWidth);
        g_params->Set(NVSDK_NGX_Parameter_OutHeight, outHeight);
        g_params->Set(NVSDK_NGX_Parameter_PerfQualityValue, quality);
        // Our flags are facts about our inputs; the [InitFlags] overrides must not rewrite them (IFeature.cpp).
        // [InitFlags] AutoExposure = false took the flag away, the bridge found no exposure texture at the first
        // evaluation and created both features a second time: 2.8 s of the freeze at 4736x4864.
        g_params->Set("OptiScaler.HostInitFlags", 1);
        const auto result =
            NVSDK_NGX_D3D11_CreateFeature(ctx, NVSDK_NGX_Feature_SuperSampling, g_params, &set.eyes[eye].handle);
        if (result != NVSDK_NGX_Result_Success)
        {
            LOG_ERROR("FC5 TAA: {} CreateFeature for eye {} failed: {:X}", set.name, eye, (UINT) result);
            set.eyes[eye].handle = nullptr;
            ok = false;
        }
    }
    if (!ok)
    {
        SetStatus(std::format("DLSS: the features for {}x{} -> {}x{} could not be made (see the log)", width, height,
                              outWidth, outHeight));
        return;
    }
    set.width = width;
    set.height = height;
    set.outWidth = outWidth;
    set.outHeight = outHeight;
    set.ready = true;
    const double area = ((double) outWidth * outHeight) / ((double) width * height);
    g_jitterSamples = std::clamp((int) std::lround(8.0 * area), 8, 32);
    g_jitterWidth = width;
    g_jitterHeight = height;
    LOG_INFO("FC5 TAA: created two {} features {}x{} -> {}x{} (quality value {}, depth {}); own jitter {} phases per "
             "eye",
             set.name, width, height, outWidth, outHeight, (int) quality, inverted ? "inverted" : "normal",
             g_jitterSamples.load());
    // The preset OptiScaler hands DLSS for this quality level ([DLSS] RenderPreset*, set by the overlay's selector):
    // read at creation, so a change takes a rebuild. 0 = DLSS's own default.
    const auto* config = Config::Instance();
    const uint32_t preset =
        !config->RenderPresetOverride.value_or_default()              ? 0u
        : quality == NVSDK_NGX_PerfQuality_Value_DLAA                 ? config->RenderPresetDLAA.value_or_default()
        : quality == NVSDK_NGX_PerfQuality_Value_MaxQuality           ? config->RenderPresetQuality.value_or_default()
        : quality == NVSDK_NGX_PerfQuality_Value_Balanced             ? config->RenderPresetBalanced.value_or_default()
        : quality == NVSDK_NGX_PerfQuality_Value_MaxPerf              ? config->RenderPresetPerformance.value_or_default()
                                                                      : config->RenderPresetUltraPerformance.value_or_default();
    const std::string presetName = preset == 0 ? "DLSS default" : std::string(1, (char) ('A' + preset - 1));
    LOG_INFO("FC5 TAA: {} features requested preset {} ({}; override {})", set.name, preset, presetName,
             config->RenderPresetOverride.value_or_default() ? "on" : "off");
    SetStatus(width < outWidth
                  ? std::format("DLSS: upscaling {}x{} -> {}x{}, preset {}", width, height, outWidth, outHeight, presetName)
                  : std::format("DLSS: DLAA {}x{}, preset {}", width, height, presetName));
}

bool Evaluate(ID3D11DeviceContext* ctx, Features& set, Eye& eye, bool reset, ID3D11Texture2D* colour,
              ID3D11Texture2D* motion, ID3D11Texture2D* depth, float jitterX, float jitterY)
{
    g_params->Set(NVSDK_NGX_Parameter_Jitter_Offset_X, jitterX);
    g_params->Set(NVSDK_NGX_Parameter_Jitter_Offset_Y, jitterY);
    g_params->Set(NVSDK_NGX_Parameter_MV_Scale_X, 1.0f);
    g_params->Set(NVSDK_NGX_Parameter_MV_Scale_Y, 1.0f);
    g_params->Set(NVSDK_NGX_Parameter_DLSS_Exposure_Scale, 1.0f);
    g_params->Set(NVSDK_NGX_Parameter_Reset, reset ? 1 : 0);
    g_params->Set(NVSDK_NGX_Parameter_Width, set.width);
    g_params->Set(NVSDK_NGX_Parameter_Height, set.height);
    g_params->Set(NVSDK_NGX_Parameter_OutWidth, set.outWidth);
    g_params->Set(NVSDK_NGX_Parameter_OutHeight, set.outHeight);
    g_params->Set(NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Width, set.width);
    g_params->Set(NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Height, set.height);
    g_params->Set(NVSDK_NGX_Parameter_Depth, depth);
    g_params->Set(NVSDK_NGX_Parameter_Color, colour);
    g_params->Set(NVSDK_NGX_Parameter_MotionVectors, motion);
    g_params->Set(NVSDK_NGX_Parameter_Output, set.output);

    State::Instance().setInputApiName = ApiUpscalerInput::FSR2X_DX11;

    const auto result = NVSDK_NGX_D3D11_EvaluateFeature(ctx, eye.handle, g_params, nullptr);
    if (result != NVSDK_NGX_Result_Success)
    {
        static NVSDK_NGX_Result last = NVSDK_NGX_Result_Success;
        if (result != last)
        {
            last = result;
            LOG_ERROR("FC5 TAA: EvaluateFeature failed: {:X}", (UINT) result);
        }
        return false;
    }
    return true;
}

// ---- Stats ----

constexpr uint64_t kStatsWindow = 900;
uint64_t g_statsStart = 0;
uint64_t g_evaluations = 0;
uint64_t g_resets = 0;
uint64_t g_skips = 0;
uint64_t g_evalMicros = 0;
uint64_t g_evalMaxMicros = 0;

void SkipOnce(uint32_t reason, const char* message)
{
    static uint32_t logged = 0;
    ++g_skips;
    if ((logged & reason) != 0)
        return;
    logged |= reason;
    LOG_WARN("FC5 TAA: {}; the game's TAA shows this frame", message);
}

// ---- Frame pacing: the interval between Presents, for judder (reprojection shows as a spread, not a lower rate) ----

constexpr size_t kPacingSlots = 2048;
uint32_t g_frameMicros[kPacingSlots] {};
std::atomic<uint64_t> g_frameCount { 0 }; // intervals recorded so far; the slot is the count modulo kPacingSlots
uint64_t g_pacingStart = 0;               // g_frameCount at the last stats line
int64_t g_lastPresentCounter = 0;         // OnPresent only

void RecordPresent()
{
    LARGE_INTEGER now {};
    QueryPerformanceCounter(&now);
    const int64_t previous = g_lastPresentCounter;
    g_lastPresentCounter = now.QuadPart;
    if (previous == 0)
        return;
    LARGE_INTEGER frequency {};
    QueryPerformanceFrequency(&frequency);
    const int64_t micros = (now.QuadPart - previous) * 1000000 / frequency.QuadPart;
    const uint64_t count = g_frameCount.load(std::memory_order_relaxed);
    g_frameMicros[count % kPacingSlots] = (uint32_t) std::clamp<int64_t>(micros, 0, UINT32_MAX);
    g_frameCount.store(count + 1, std::memory_order_release);
}

// ---- Present pacing ----
//
// On DX11 in VR the game was measured free-running:
// below refresh that means frames reach the compositor out of step and reprojection judders. Drawing a
// weapon in FC5 only fixes it by accident, by adding work until the cadence settles (measured:
// 80/s free-running, 95th 17.4 ms, 2-4% long frames -> juddering; 75.4/s settled, 95th 14.2 ms, ~0% -> smooth).
// Holding each Present to a whole number of refreshes supplies the missing wait. Timing only: no GPU work.
uint64_t g_paceLate = 0; // Presents that missed their slot (the cadence is beyond what the frame costs allow)

void Pace()
{
    static const int refreshes = Config::Instance()->Fc5TaaPaceRefreshes.value_or_default();
    static const int hz = Config::Instance()->Fc5TaaPaceHz.value_or_default();
    if (refreshes <= 0 || hz <= 0)
        return;
    static LARGE_INTEGER frequency = []
    {
        LARGE_INTEGER value {};
        QueryPerformanceFrequency(&value);
        return value;
    }();
    if (frequency.QuadPart == 0)
        return;
    // A high-resolution timer where the OS has one (Windows 10 1803+), so the wait isn't rounded to the tick.
    static HANDLE timer = []
    {
        HANDLE handle = CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION,
                                               TIMER_ALL_ACCESS);
        if (handle == nullptr)
            handle = CreateWaitableTimerExW(nullptr, nullptr, 0, TIMER_ALL_ACCESS);
        return handle;
    }();
    const int64_t interval = frequency.QuadPart * refreshes / hz;
    static int64_t nextSlot = 0;
    LARGE_INTEGER now {};
    QueryPerformanceCounter(&now);
    if (nextSlot == 0 || now.QuadPart > nextSlot + interval)
    {
        // First Present, or a whole slot late (a hitch, a loading screen): start the schedule again from here.
        if (nextSlot != 0)
            ++g_paceLate;
        nextSlot = now.QuadPart + interval;
        return;
    }
    // Sleep to a little before the slot, then spin: sleeping the whole way overshoots and eats the next one.
    const int64_t spinMargin = frequency.QuadPart / 2000; // 0.5 ms
    if (timer != nullptr && nextSlot - now.QuadPart > spinMargin)
    {
        LARGE_INTEGER due {};
        due.QuadPart = -((nextSlot - spinMargin - now.QuadPart) * 10000000 / frequency.QuadPart); // 100 ns, relative
        if (SetWaitableTimer(timer, &due, 0, nullptr, nullptr, FALSE))
            WaitForSingleObject(timer, 100);
    }
    while (true)
    {
        QueryPerformanceCounter(&now);
        if (now.QuadPart >= nextSlot)
            break;
        YieldProcessor();
    }
    nextSlot += interval;
}

// Median, 95th percentile and the share of intervals longer than 1.5x the median, over the stats window.
void PacingStats(double& medianMs, double& p95Ms, double& longShare, uint64_t& samples)
{
    medianMs = p95Ms = longShare = 0.0;
    const uint64_t count = g_frameCount.load(std::memory_order_acquire);
    samples = std::min<uint64_t>(count - g_pacingStart, kPacingSlots);
    g_pacingStart = count;
    if (samples < 8)
        return;
    std::vector<uint32_t> window;
    window.reserve((size_t) samples);
    for (uint64_t i = count - samples; i < count; ++i)
        window.push_back(g_frameMicros[i % kPacingSlots]);
    std::sort(window.begin(), window.end());
    const auto median = window[window.size() / 2];
    medianMs = (double) median / 1000.0;
    p95Ms = (double) window[(size_t) ((double) window.size() * 0.95)] / 1000.0;
    const auto over = std::upper_bound(window.begin(), window.end(), (uint32_t) (median * 3 / 2));
    longShare = 100.0 * (double) std::distance(over, window.end()) / (double) window.size();
}

void LogStats(uint64_t present)
{
    if (g_statsStart == 0)
        g_statsStart = present;
    if (present - g_statsStart < kStatsWindow)
        return;
    const double avgMs = g_evaluations > 0 ? (double) g_evalMicros / (double) g_evaluations / 1000.0 : 0.0;
    double medianMs = 0.0;
    double p95Ms = 0.0;
    double longShare = 0.0;
    uint64_t pacingSamples = 0;
    PacingStats(medianMs, p95Ms, longShare, pacingSamples);
    const int paceRefreshes = Config::Instance()->Fc5TaaPaceRefreshes.value_or_default();
    if (pacingSamples >= 8)
    {
        LOG_INFO("FC5 TAA: Present pacing over {} intervals: {:.2f} ms median ({:.1f}/s), {:.2f} ms 95th, {:.1f}% "
                 "longer than 1.5x the median",
                 pacingSamples, medianMs, medianMs > 0.0 ? 1000.0 / medianMs : 0.0, p95Ms, longShare);
        if (paceRefreshes > 0)
            LOG_INFO("FC5 TAA: pacer on, one Present per {} refreshes of {} Hz ({:.2f} ms); {} slots missed",
                     paceRefreshes, Config::Instance()->Fc5TaaPaceHz.value_or_default(),
                     1000.0 * paceRefreshes / std::max(1, Config::Instance()->Fc5TaaPaceHz.value_or_default()),
                     g_paceLate);
    }
    g_paceLate = 0;
    LOG_INFO("FC5 TAA: {} evaluations ({} resets, {} frames skipped) over {} Presents, CPU {:.2f} ms average, "
             "{:.2f} ms worst; {} with matrix motion, {} with own jitter; {} upscale draws replaced; game jitter: {} "
             "reads, {} changes, {} invalid",
             g_evaluations, g_resets, g_skips, present - g_statsStart, avgMs, (double) g_evalMaxMicros / 1000.0,
             g_matrixFrames, g_ownJitterFrames, g_drawsReplaced, g_jitterReads, g_jitterChanges, g_jitterInvalid);
    double usedMb = 0, budgetMb = 0;
    if (VideoMemoryMb(nullptr, usedMb, budgetMb))
        LOG_INFO("FC5 TAA: COST: video memory with the features in use: the process uses {:.0f} MB of a budget of "
                 "{:.0f} MB{}",
                 usedMb, budgetMb, usedMb > budgetMb ? " -- OVER the budget: textures are being paged out" : "");
    g_statsStart = present;
    g_evaluations = g_resets = g_skips = g_evalMicros = g_evalMaxMicros = g_matrixFrames = g_ownJitterFrames = 0;
    g_drawsReplaced = 0;
}

void ResetStats()
{
    g_statsStart = 0;
    g_pacingStart = g_frameCount.load(std::memory_order_acquire);
    g_evaluations = g_resets = g_skips = g_evalMicros = g_evalMaxMicros = g_matrixFrames = g_ownJitterFrames = 0;
    g_drawsReplaced = 0;
}

bool Diagnostics() { return Config::Instance()->DuniaTaaDiagnostics.value_or_default(); }

// ---- Own per-eye jitter, patched into the game's view constants ----
//
// FC5's jitter is a 2-phase pattern advancing every Present, so under alternate-eye VR each eye always renders from
// the same sub-pixel offset and DLSS never gathers sub-pixel detail. While DLAA runs, the game's writes of its view
// constants (1984 bytes: inverse projection +0, projection +128,
// view-projection +256, view-rotation-projection +320, the previous view-rotation-projection +384, viewport size +640,
// motion vector offsets +816) are changed to our own jitter: per eye, Halton(2,3) over 8 samples, advancing every
// second Present. The jitter is the image's shift in pixels (x right, y down), as the motion vector offsets hold it; the
// matrices with clip-space output get row x += tx * row w, row y += ty * row w (tx = 2 dx / W, ty = -2 dy / H for the
// change dx, dy from the game's), the inverse projection the opposite on its w column, the previous-frame matrix the
// change that was made on the previous Present. Only buffers whose ViewportSize is the TAA size are touched (shadow
// and other viewports keep theirs). The composed same-eye motion needs no change: chaining the game's matrices over
// two Presents cancels its own jitter.
constexpr UINT kViewConstantsBytes = 1984;
std::atomic<bool> g_ownJitter { true };
// Diagnostics (F4's third and fourth states). 1: our jitter x20 patched in, DLSS still given the game's jitter, so if
// the patched constants reach the geometry the image shakes by up to ~10 px. 2: our jitter x4 patched in and given to
// DLSS (with F5's signs): when eye, timing and sign match, DLSS cancels it and the image stays still.
std::atomic<int> g_shakeTest { 0 };

struct AppliedJitter
{
    uint64_t present = UINT64_MAX;
    float game[2] = {};
    float ours[2] = {};
};
std::mutex g_jitterMutex;
AppliedJitter g_applied[2]; // by Present parity
uint64_t g_patchedWrites = 0;
uint64_t g_patchedMaps = 0;
uint64_t g_patchedUpdates = 0;

float HaltonSample(int index, int base)
{
    float f = 1.0f;
    float r = 0.0f;
    while (index > 0)
    {
        f /= (float) base;
        r += f * (float) (index % base);
        index /= base;
    }
    return r;
}

void OwnJitterFor(uint64_t present, float& x, float& y)
{
    const int index = (int) ((present >> 1) % (uint64_t) g_jitterSamples.load(std::memory_order_relaxed)) + 1;
    x = HaltonSample(index, 2) - 0.5f;
    y = HaltonSample(index, 3) - 0.5f;
    const int shake = g_shakeTest.load(std::memory_order_relaxed);
    if (shake != 0)
    {
        const float scale = shake == 1 ? 20.0f : 4.0f;
        x *= scale;
        y *= scale;
    }
}

bool IsViewConstants(ID3D11Resource* resource)
{
    if (resource == nullptr)
        return false;
    D3D11_RESOURCE_DIMENSION dim = D3D11_RESOURCE_DIMENSION_UNKNOWN;
    resource->GetType(&dim);
    if (dim != D3D11_RESOURCE_DIMENSION_BUFFER)
        return false;
    D3D11_BUFFER_DESC desc {};
    static_cast<ID3D11Buffer*>(resource)->GetDesc(&desc);
    return desc.ByteWidth == kViewConstantsBytes && (desc.BindFlags & D3D11_BIND_CONSTANT_BUFFER) != 0;
}

// Clip-space output: rows x, y, z, w at m[0], m[4], m[8], m[12].
void ShiftClip(float* m, float tx, float ty)
{
    for (int c = 0; c < 4; ++c)
    {
        m[c] += tx * m[12 + c];
        m[4 + c] += ty * m[12 + c];
    }
}

// The inverse of a clip-space matrix: its input's w column takes the opposite shift.
void ShiftInverse(float* m, float tx, float ty)
{
    for (int r = 0; r < 4; ++r)
        m[r * 4 + 3] -= tx * m[r * 4] + ty * m[r * 4 + 1];
}

// ---- Stale-eye camera share ([Inputs] StaleEyeCameraShare) ----
//
// The motion vector pass's constants (a 192-byte block) hold the current-to-previous matrix at +16..+64:
// this Present's clip space to the previous Present's, i.e. the other eye's render under alternate-eye VR. The game
// writes it every Present; it is published for the stale-eye OpenXR layer (StaleEyeShareAbi.h). Identified by its
// layout, not its size alone: +0 (1, -1), +96 (W, H, 1/W, 1/H), +112 (W/2, H/2) -- the TAA draw's own 192-byte
// block has (W, H) at +112. Checked against F7 captures.
constexpr UINT kTemporalConstantsBytes = 192;

bool IsTemporalConstants(ID3D11Resource* resource)
{
    if (resource == nullptr)
        return false;
    D3D11_RESOURCE_DIMENSION dim = D3D11_RESOURCE_DIMENSION_UNKNOWN;
    resource->GetType(&dim);
    if (dim != D3D11_RESOURCE_DIMENSION_BUFFER)
        return false;
    D3D11_BUFFER_DESC desc {};
    static_cast<ID3D11Buffer*>(resource)->GetDesc(&desc);
    return desc.ByteWidth == kTemporalConstantsBytes && (desc.BindFlags & D3D11_BIND_CONSTANT_BUFFER) != 0;
}

// The main view's projection, from the game's writes of its view constants (1984 bytes: the projection at +128, rows,
// column-vector; the viewport size at +640), as frustum tangents (left, right, up, down) in a camera with x right, y up,
// looking forward: the last row is (0, 0, s, 0), w = s z, so ndc = P00 t + s P02 for the forward tangent t. FC5 writes
// many views a frame (probes 2304x1024 at +-2.14, several 1920x1080 passes, the eye), so the latest projection is
// kept PER SIZE and the one at the motion pass's size is published. (Keeping only the latest one published almost
// never, and the layer fell back to the headset's FOV.) Checked offline on F7
// captures: the eye's is L-1.059 R+1.059 U+1.235 D-0.951, and with it the motion pass's matrix is a rotation to a
// residual of 0.0005.
std::mutex g_projectionMutex;
struct SizedProjection
{
    float size[2] = {};
    float tangents[4] = {};
    float matrix[16] = {};
    bool valid = false;
};
SizedProjection g_projections[16];
// The size the TAA draw ran at (width << 32 | height) and the Present it was last seen: the eye's render size.
// Written where the TAA draw is matched, read by the constants hooks on the game's recording threads.
std::atomic<uint64_t> g_taaSizeSeen { 0 }, g_taaSizePresent { 0 };
std::atomic<uint64_t> g_published { 0 }, g_publishOtherSize { 0 };

void NoteViewProjection(const float* w)
{
    const float* p = w + 32;
    const float s = p[14];
    if (p[12] != 0.0f || p[13] != 0.0f || std::fabs(std::fabs(s) - 1.0f) > 1e-4f || p[15] != 0.0f ||
        std::fabs(p[0]) < 1e-3f || std::fabs(p[5]) < 1e-3f || w[160] < 64.0f || w[161] < 64.0f)
        return;
    const float t[4] = { (-1.0f - s * p[2]) / p[0], (1.0f - s * p[2]) / p[0], (1.0f - s * p[6]) / p[5],
                         (-1.0f - s * p[6]) / p[5] };
    std::lock_guard lock(g_projectionMutex);
    SizedProjection* slot = nullptr;
    for (auto& p : g_projections)
    {
        if (p.valid && p.size[0] == w[160] && p.size[1] == w[161])
        {
            slot = &p;
            break;
        }
    }
    if (slot == nullptr)
    {
        for (auto& p : g_projections)
        {
            if (!p.valid)
            {
                slot = &p;
                break;
            }
        }
    }
    if (slot == nullptr)
        slot = &g_projections[15]; // more sizes than slots: the last one rotates
    const bool changed = !slot->valid || slot->size[0] != w[160] || slot->size[1] != w[161] ||
                         std::fabs(t[0] - slot->tangents[0]) > 1e-3f || std::fabs(t[2] - slot->tangents[2]) > 1e-3f ||
                         std::fabs(t[3] - slot->tangents[3]) > 1e-3f;
    // Logged per size, a few changes each: a size whose projection keeps changing is not a stable view.
    static std::atomic<int> changesLogged { 0 };
    if (changed && changesLogged.fetch_add(1) < 40)
        LOG_INFO("Stale-eye share: view projection {}x{} tangents L{:+.4f} R{:+.4f} U{:+.4f} D{:+.4f}", w[160], w[161],
                 t[0], t[1], t[2], t[3]);
    slot->size[0] = w[160];
    slot->size[1] = w[161];
    std::memcpy(slot->tangents, t, sizeof(t));
    std::memcpy(slot->matrix, p, sizeof(slot->matrix));
    slot->valid = true;
}

void PublishIfMotionConstants(const float* w)
{
    const float width = w[24];
    const float height = w[25];
    const char* reason = nullptr;
    if (w[0] != 1.0f || w[1] != -1.0f)
        reason = "+0 is not (1, -1)";
    else if (!(width >= 64.0f && height >= 64.0f && width <= 16384.0f && height <= 16384.0f) ||
             std::fabs(w[26] * width - 1.0f) > 1e-3f || std::fabs(w[27] * height - 1.0f) > 1e-3f)
        reason = "+96 is not (W, H, 1/W, 1/H)";
    else if (std::fabs(w[28] - width * 0.5f) > 0.5f || std::fabs(w[29] - height * 0.5f) > 0.5f)
        reason = "+112 is not (W/2, H/2) (the TAA draw's block, or another)";
    if (reason != nullptr)
    {
        // Every other 192-byte constant buffer lands here too: log what it nearly matched, once per reason.
        static std::atomic<uint32_t> logged { 0 };
        const uint32_t bit = reason[1] == '0' ? 1u : reason[1] == '9' ? 2u : 4u;
        if ((logged.fetch_or(bit) & bit) == 0 && bit != 1u)
            LOG_INFO("Stale-eye share: 192-byte constants not taken: {} (+96 {} {} {} {}, +112 {} {})", reason,
                     w[24], w[25], w[26], w[27], w[28], w[29]);
        return;
    }
    // Only the EYE's camera is published: the game writes the same constants for a pass at HALF the eye's size a few
    // times a minute (2368x2432 beside 4736x4864), and each of those counted as a Present of its own -- the eyes'
    // alternation in the publications broke there, and the layer's pairing with it (frequent hard
    // judder). The eye is the size the TAA draw ran at (a structural fact, known from the Present
    // before); while no TAA draw was seen lately (start, menus) everything is published, as before.
    {
        const uint64_t taa = g_taaSizeSeen.load(std::memory_order_relaxed);
        const uint64_t taaPresent = g_taaSizePresent.load(std::memory_order_relaxed);
        const uint64_t mine = ((uint64_t) (uint32_t) width << 32) | (uint64_t) (uint32_t) height;
        if (taa != 0 && g_present.load() - taaPresent <= 90 && mine != taa)
        {
            const uint64_t skipped = g_publishOtherSize.fetch_add(1) + 1;
            if (skipped == 1 || skipped % 100 == 0)
                LOG_INFO("Stale-eye share: motion constants at {}x{} are not the eye's ({}x{}, the TAA draw's): not "
                         "published ({} so far, {} published)",
                         width, height, (uint32_t) (taa >> 32), (uint32_t) taa, skipped, g_published.load());
            return;
        }
    }
    g_published.fetch_add(1);
    float tangents[4];
    float projection[16];
    bool haveTangents = false;
    {
        std::lock_guard lock(g_projectionMutex);
        for (const auto& p : g_projections)
        {
            if (p.valid && p.size[0] == width && p.size[1] == height)
            {
                std::memcpy(tangents, p.tangents, sizeof(tangents));
                std::memcpy(projection, p.matrix, sizeof(projection));
                haveTangents = true;
                break;
            }
        }
    }
    static std::atomic<int> loggedNoProjection { 0 };
    if (!haveTangents && loggedNoProjection.fetch_add(1) < 3)
        LOG_INFO("Stale-eye share: no view projection seen at the motion pass's size {}x{}; published without",
                 width, height);
    static std::atomic<bool> loggedFirstEye { false };
    if (haveTangents && width > 1920 && !loggedFirstEye.exchange(true))
        LOG_INFO("Stale-eye share: publishing {}x{} with its projection L{:+.4f} R{:+.4f} U{:+.4f} D{:+.4f}", width,
                 height, tangents[0], tangents[1], tangents[2], tangents[3]);
    StaleEyeShare::Publish(w + 4, (uint32_t) width, (uint32_t) height, true, haveTangents ? tangents : nullptr,
                           haveTangents ? projection : nullptr);
}

bool OwnJitterActive()
{
    return g_ownJitter.load(std::memory_order_relaxed) && g_mode.load() == Dlaa && g_jitterWidth.load() != 0;
}

// Patches one write of the view constants (a CPU copy, `data` 1984 bytes). True when it changed it.
bool PatchViewConstants(float* data)
{
    const float width = data[160];
    const float height = data[161];
    if (width != (float) g_jitterWidth.load() || height != (float) g_jitterHeight.load())
        return false;
    const float gameX = data[204];
    const float gameY = data[205];
    if (!(std::fabs(gameX) <= 0.5f && std::fabs(gameY) <= 0.5f))
        return false;

    const uint64_t present = g_present.load();
    float oursX = 0.0f;
    float oursY = 0.0f;
    OwnJitterFor(present, oursX, oursY);
    AppliedJitter previous;
    {
        std::lock_guard<std::mutex> lock(g_jitterMutex);
        auto& applied = g_applied[present & 1];
        if (applied.present != present)
        {
            applied.present = present;
            applied.game[0] = gameX;
            applied.game[1] = gameY;
            applied.ours[0] = oursX;
            applied.ours[1] = oursY;
        }
        previous = g_applied[(present + 1) & 1];
    }

    const float tx = 2.0f * (oursX - gameX) / width;
    const float ty = -2.0f * (oursY - gameY) / height;
    ShiftInverse(data + 0, tx, ty);
    ShiftClip(data + 32, tx, ty);
    ShiftClip(data + 64, tx, ty);
    ShiftClip(data + 80, tx, ty);
    // The previous frame's matrix: the change made on the previous Present, if any.
    if (previous.present + 1 == present)
    {
        const float ptx = 2.0f * (previous.ours[0] - previous.game[0]) / width;
        const float pty = -2.0f * (previous.ours[1] - previous.game[1]) / height;
        ShiftClip(data + 96, ptx, pty);
    }
    data[204] = oursX;
    data[205] = oursY;
    if (g_patchedWrites++ == 0)
        LOG_INFO("FC5 TAA: own jitter patched into the view constants ({}x{}): game ({:+.3f}, {:+.3f}) -> ours "
                 "({:+.3f}, {:+.3f}) px",
                 width, height, gameX, gameY, oursX, oursY);
    return true;
}

// Maps of the view constants in flight on this thread (Map and Unmap come in pairs on one context).
struct PendingMap
{
    ID3D11Resource* resource = nullptr;
    UINT subresource = 0;
    void* data = nullptr;
    bool temporal = false; // the motion pass's temporal constants (stale-eye share), not the view constants
};
thread_local PendingMap t_maps[4];

// The game's jitter and ours for this Present, if the view constants were patched on it.
bool AppliedFor(uint64_t present, float& x, float& y)
{
    if (g_shakeTest.load(std::memory_order_relaxed) == 1)
        return false;
    std::lock_guard<std::mutex> lock(g_jitterMutex);
    const auto& applied = g_applied[present & 1];
    if (applied.present != present)
        return false;
    x = applied.ours[0];
    y = applied.ours[1];
    return true;
}

// [Inputs] TaaDlaaSkipGameTaa (default on): the work runs just before the game's TAA draw and the draw is skipped
// whenever the TAA output was written; off, it runs after the draw and overwrites its output.
bool SkipGameTaa()
{
    static const bool skip = Config::Instance()->TaaDlaaSkipGameTaa.value_or_default();
    return skip;
}
} // namespace

bool Fc5Taa_Dx11::Enabled()
{
    static int cached = -1;
    if (cached < 0)
        cached = (_stricmp(State::Instance().gameExe.c_str(), "FarCry5.exe") == 0 &&
                  Config::Instance()->Fc5TaaDlss.value_or_default())
                     ? 1
                     : 0;
    return cached == 1;
}

bool Fc5Taa_Dx11::InPass() { return g_inPass > 0; }

void Fc5Taa_Dx11::OnBind(ID3D11DeviceContext* ctx, UINT numRtvs, ID3D11RenderTargetView* const* rtvs,
                         ID3D11DepthStencilView* dsv)
{
    // The G-buffer bind: four targets plus depth, all one size. Its depth is the scene depth.
    if (g_inPass > 0 || ctx == nullptr || numRtvs < 4 || numRtvs > D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT ||
        rtvs == nullptr || dsv == nullptr)
        return;
    for (UINT i = 0; i < 4; ++i)
    {
        if (rtvs[i] == nullptr)
            return;
    }
    if (StaleEyeShare::Enabled())
    {
        // The device the G-buffer is drawn on, for the stale-eye layer to compare with OpenXR's (identity only).
        ID3D11Device* device = nullptr;
        ctx->GetDevice(&device);
        if (device != nullptr)
        {
            StaleEyeShare::SetGameDevice(device);
            device->Release();
        }
    }
    ID3D11Texture2D* colour = TextureFromView(rtvs[0]);
    ID3D11Texture2D* depth = TextureFromView(dsv);
    if (depth != nullptr && depth == g_depth)
        g_depthBindPresent = g_present.load();
    if (colour != nullptr && depth != nullptr && depth != g_depth)
    {
        const auto colourDesc = Desc(colour);
        const auto depthDesc = Desc(depth);
        if (colourDesc.Width == depthDesc.Width && colourDesc.Height == depthDesc.Height &&
            colourDesc.Width >= kMinRenderSize && colourDesc.Height >= kMinRenderSize &&
            depthDesc.SampleDesc.Count == 1)
        {
            // Held until a while after the game moves on (reloads reallocate its targets).
            depth->AddRef();
            if (g_depth != nullptr)
                g_deferred.push_back({ g_depth, g_present.load() });
            g_depth = depth;
            g_depthBindPresent = g_present.load();
            static int logged = 0;
            if (logged++ < 5)
                LOG_INFO("FC5 TAA: G-buffer depth {}x{} format {}", depthDesc.Width, depthDesc.Height,
                         (UINT) depthDesc.Format);
        }
    }
    if (colour != nullptr)
        colour->Release();
    if (depth != nullptr)
        depth->Release();
}

void Fc5Taa_Dx11::BeforeDispatch(ID3D11DeviceContext* ctx)
{
    if (g_inPass > 0 || ctx == nullptr || ctx != g_context || !g_ready || g_mvTextures[0] == nullptr)
        return;
    if (g_mode.load(std::memory_order_relaxed) == GameTaa &&
        (!Diagnostics() || g_viewMode.load(std::memory_order_relaxed) == 0))
        return;
    ID3D11UnorderedAccessView* uav = nullptr;
    ctx->CSGetUnorderedAccessViews(0, 1, &uav);
    if (uav == nullptr)
        return;
    ID3D11Resource* written = nullptr;
    uav->GetResource(&written);
    uav->Release();
    if (written != nullptr)
        written->Release(); // compared by address only
    if (written == nullptr || (written != g_mvTextures[0] && written != g_mvTextures[1]))
        return;

    ID3D11Buffer* constants = nullptr;
    ctx->CSGetConstantBuffers(0, 1, &constants);
    ID3D11ShaderResourceView* shaderIndex = nullptr;
    ctx->CSGetShaderResources(3, 1, &shaderIndex);
    const uint64_t present = g_present.load();
    auto& copy = g_matrices[present & 1];
    D3D11_BUFFER_DESC desc {};
    if (constants != nullptr)
        constants->GetDesc(&desc);
    if (constants != nullptr && desc.ByteWidth >= 80 && desc.ByteWidth % 16 == 0 && shaderIndex != nullptr)
    {
        if (copy.buffer == nullptr || copy.bytes != desc.ByteWidth)
        {
            // A new size (not expected): the old copy is parked, never released under queued work.
            if (copy.buffer != nullptr)
                g_deferred.push_back({ copy.buffer, present });
            copy = {};
            D3D11_BUFFER_DESC copyDesc {};
            copyDesc.ByteWidth = desc.ByteWidth;
            copyDesc.Usage = D3D11_USAGE_DEFAULT;
            copyDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
            ID3D11Device* device = nullptr;
            ctx->GetDevice(&device);
            if (device != nullptr)
            {
                if (FAILED(device->CreateBuffer(&copyDesc, nullptr, &copy.buffer)))
                    copy.buffer = nullptr;
                device->Release();
            }
            copy.bytes = desc.ByteWidth;
        }
        if (copy.buffer != nullptr)
        {
            PassScope scope;
            ctx->CopyResource(copy.buffer, constants);
            copy.present = present;
            if (g_matrixCopies++ == 0)
                LOG_INFO("FC5 TAA: motion vector pass found ({} byte constants)", desc.ByteWidth);
        }
        if (shaderIndex != g_shaderIndex)
        {
            shaderIndex->AddRef();
            if (g_shaderIndex != nullptr)
                g_deferred.push_back({ g_shaderIndex, present });
            g_shaderIndex = shaderIndex;
        }
    }
    if (constants != nullptr)
        constants->Release();
    if (shaderIndex != nullptr)
        shaderIndex->Release();
}

// The TAA work, at the TAA draw: true when it wrote the TAA output (DLAA or the motion view).
// beforeGameDraw: called in front of the game's TAA draw, which runs if this returns false.
static bool RunTaa(ID3D11DeviceContext* ctx, bool beforeGameDraw)
{
    if (g_inPass > 0 || ctx == nullptr || (g_setupTried && (!g_ready || ctx != g_context)))
        return false;
    TaaViews views;
    if (!MatchTaa(ctx, views))
        return false;
    if (!g_setupTried)
        Setup(ctx);
    if (!g_ready)
        return false;

    const uint64_t present = g_present.load();
    g_taaOutput = views.output;
    g_taaOutputPresent = present;
    ReleaseDeferred();
    ReadJitter(ctx, present);
    QueueJitterCopy(ctx, views.constants, present);
    g_pendingShare = { true, views.width, views.height, present };
    g_taaSizeSeen.store(((uint64_t) views.width << 32) | (uint64_t) views.height, std::memory_order_relaxed);
    g_taaSizePresent.store(present, std::memory_order_relaxed);

    const int mode = g_mode.load(std::memory_order_relaxed);
    const int view = Diagnostics() ? g_viewMode.load(std::memory_order_relaxed) : 0;
    if (mode == GameTaa)
        ResetStats();
    else
        LogStats(present);
    // The size this draw asks for, and whether it is gameplay's and has held (followed in every mode: no GPU work).
    bool settled = false;
    {
        const bool upscalingNow = g_eyeWidth > views.width && g_eyeHeight > views.height;
        const UINT want[4] = { views.width, views.height, upscalingNow ? g_eyeWidth : views.width,
                               upscalingNow ? g_eyeHeight : views.height };
        bool depthHere = false;
        if (g_depth != nullptr && g_depthBindPresent.load(std::memory_order_relaxed) == present)
        {
            const auto depthDesc = Desc(g_depth);
            depthHere = depthDesc.Width == views.width && depthDesc.Height == views.height;
        }
        const bool eyeSeen = g_upscaleDrawPresent != UINT64_MAX && g_upscaleDrawPresent + 1 == present;
        g_wanted.depthHere = depthHere;
        g_wanted.eyeSeen = eyeSeen;
        settled = SizeSettled(want, present, depthHere && eyeSeen);
    }
    if (mode != GameTaa)
        g_offSince = UINT64_MAX;
    else if (g_offSince == UINT64_MAX)
        g_offSince = present;
    if (mode == GameTaa && (g_dlaa.ready || g_upscale.ready) && present - g_offSince >= kFreeAfterOff)
    {
        if (g_dlaa.ready)
            ReleaseSet(g_dlaa, "DLSS has been off for 900 Presents");
        if (g_upscale.ready)
            ReleaseSet(g_upscale, "DLSS has been off for 900 Presents");
        double usedMb = 0, budgetMb = 0;
        if (VideoMemoryMb(nullptr, usedMb, budgetMb))
            LOG_INFO("FC5 TAA: video memory with the features freed: the process uses {:.0f} MB of a budget of {:.0f} MB",
                     usedMb, budgetMb);
        SetStatus("DLSS: not started");
    }
    if (mode == GameTaa && view == 0)
    {
        // Nothing to do; composition restarts when DLAA or the view is switched on.
        g_lastDecodePresent = UINT64_MAX;
        g_lastReplacedPresent = UINT64_MAX;
        return false;
    }
    // "Rebuild DLSS now" / a preset picked in the overlay: both pairs are freed here and made again by this draw or the
    // next with the settings as they are now.
    if (g_rebuildRequested.exchange(false))
    {
        LOG_INFO("FC5 TAA: rebuild requested from the overlay (Present {})", present);
        if (g_dlaa.ready || g_dlaa.createTried)
            ReleaseSet(g_dlaa, "rebuild requested");
        if (g_upscale.ready || g_upscale.createTried)
            ReleaseSet(g_upscale, "rebuild requested");
        g_lastReplacedPresent = UINT64_MAX;
    }
    // The game's own frames (loading screens after a death, menus) come at another size, where the features can't
    // run. Do nothing at all then: no motion textures, no decode, no depth copy — those frames stay the game's.
    // Working at the new size after a reload lost the D3D11 device twice (both within
    // 0.3 s of the first frame at that size).
    {
        const bool upscalingNow = g_eyeWidth > views.width && g_eyeHeight > views.height;
        const Features& set = upscalingNow ? g_upscale : g_dlaa;
        const UINT outWidth = upscalingNow ? g_eyeWidth : views.width;
        const UINT outHeight = upscalingNow ? g_eyeHeight : views.height;
        Features& wanted = upscalingNow ? g_upscale : g_dlaa;
        Features& other = upscalingNow ? g_dlaa : g_upscale;
        const bool mismatch = set.ready && (views.width != set.width || views.height != set.height ||
                                            outWidth != set.outWidth || outHeight != set.outHeight);
        if ((mismatch || !set.ready) && mode == Dlaa)
        {
            // The shape of the picture must be the one the features had: R.E.A.L. VR's resolutions and the Resolution
            // Scale keep it; the flat frames of a reload or of the switch into VR do not.
            const Features* reference = set.ready ? &set : other.ready ? &other : nullptr;
            bool shapeOk = true;
            if (reference != nullptr)
            {
                const double before = (double) reference->outWidth / (double) reference->outHeight;
                const double now = (double) outWidth / (double) outHeight;
                shapeOk = std::fabs(now / before - 1.0) <= 0.03;
            }
            if (!settled || !shapeOk)
            {
                if (!shapeOk && !g_wanted.refusalLogged)
                {
                    g_wanted.refusalLogged = true;
                    LOG_INFO("FC5 TAA: {}x{} -> {}x{} is not followed: its shape is not the picture's the features "
                             "were made for ({}x{}); the game's own frames (loading, the desktop view)",
                             views.width, views.height, outWidth, outHeight, reference->outWidth, reference->outHeight);
                }
                else if (shapeOk && !g_wanted.waitLogged)
                {
                    g_wanted.waitLogged = true;
                    LOG_INFO("FC5 TAA: new size {}x{} -> {}x{}: waiting for it to hold for {} Presents of gameplay "
                             "(the game's TAA shows meanwhile)",
                             views.width, views.height, outWidth, outHeight, kSettlePasses);
                    SetStatus(std::format("DLSS: new size {}x{} -> {}x{}, waiting for it to settle", views.width,
                                          views.height, outWidth, outHeight));
                }
                // A wait that does not end says why: which half of the gameplay signature is missing.
                if (shapeOk && ++g_wanted.waited >= 600 && !g_wanted.stuckLogged)
                {
                    g_wanted.stuckLogged = true;
                    LOG_WARN("FC5 TAA: {}x{} -> {}x{} has not settled in {} TAA draws: G-buffer bound at the render "
                             "size this Present: {}; the game's upscale draw seen the Present before: {} (a menu or a "
                             "loading screen if both stay 'no'; DLSS stays off until gameplay at this size)",
                             views.width, views.height, outWidth, outHeight, g_wanted.waited,
                             g_wanted.depthHere ? "yes" : "NO", g_wanted.eyeSeen ? "yes" : "NO");
                }
                if (mismatch || !set.createTried)
                {
                    // Nothing at all at this size: no motion textures, no decode, no depth copy.
                    ++g_skips;
                    g_lastDecodePresent = UINT64_MAX;
                    g_lastReplacedPresent = UINT64_MAX;
                    return false;
                }
            }
            else
            {
                if (mismatch)
                {
                    ++wanted.rebuilds;
                    LOG_INFO("FC5 TAA: {} features {}x{} -> {}x{} rebuilt for {}x{} -> {}x{} (the size held for {} "
                             "Presents of gameplay; rebuild {})",
                             set.name, set.width, set.height, set.outWidth, set.outHeight, views.width, views.height,
                             outWidth, outHeight, kSettlePasses, wanted.rebuilds);
                    ReleaseSet(wanted, "the size changed");
                }
                // The pair for the other mode (100% against below 100%) is not kept beside this one: 2.8 GB and 5.5 GB
                // of video memory at a 4000x4128 eye. Going back makes it again.
                if (!wanted.createTried && other.ready)
                    ReleaseSet(other, "the Resolution Scale went to the other side of 100%");
                g_lastReplacedPresent = UINT64_MAX;
            }
        }
        else if (mismatch)
        {
            // Not DLAA mode (the motion view): nothing at this size either.
            g_lastDecodePresent = UINT64_MAX;
            g_lastReplacedPresent = UINT64_MAX;
            return false;
        }
    }
    if (!EnsureMotion(ctx, views.width, views.height))
        return false;

    ID3D11Device* device = nullptr;
    ctx->GetDevice(&device);
    if (device == nullptr)
        return false;

    // Depth: this frame's G-buffer depth at the TAA size, and the previous Present's copy of it.
    const int next = 1 - g_current;
    ID3D11ShaderResourceView* depthSrv = nullptr;
    ID3D11ShaderResourceView* previousDepthSrv = nullptr;
    ID3D11Texture2D* depth = g_depth;
    bool depthUsable = false;
    if (depth != nullptr)
    {
        const auto depthDesc = Desc(depth);
        depthUsable = depthDesc.Width == views.width && depthDesc.Height == views.height &&
                      (depthSrv = DepthSrvFor(device, depth)) != nullptr &&
                      EnsureDepthCopy(device, g_depthCopies[next], depth);
        const auto& previous = g_depthCopies[1 - next];
        if (depthUsable && previous.present != UINT64_MAX && previous.present + 1 == present &&
            previous.width == depthDesc.Width && previous.height == depthDesc.Height)
            previousDepthSrv = previous.srv;
    }
    const bool hasPrevious = g_lastDecodePresent != UINT64_MAX && g_lastDecodePresent + 1 == present;
    const bool occlusion = hasPrevious && previousDepthSrv != nullptr && g_occlusion.load(std::memory_order_relaxed);
    ID3D11UnorderedAccessView* targetUav = TargetUavFor(device, views.output);
    device->Release();

    // In front of the game's TAA draw, our passes may only be queued when this call is going to REPLACE that draw. The
    // decode pass and the depth copy queued before a draw that then ran lost the device (887A0005 13 ms later;
    // a plain depth copy before it did the same). So everything that can stop
    // DLSS is asked here, before anything is queued, and then nothing at all is done.
    if (beforeGameDraw && mode == Dlaa && view == 0)
    {
        const bool upscalingEarly = g_eyeWidth > views.width && g_eyeHeight > views.height;
        const Features& early = upscalingEarly ? g_upscale : g_dlaa;
        float x = 0.0f, y = 0.0f;
        uint32_t reason = 0;
        const char* why = nullptr;
        if (!depthUsable)
        {
            reason = 1;
            why = "no G-buffer depth at the TAA size this frame";
        }
        else if (!g_jitterKnown[present & 1] && !AppliedFor(present, x, y))
        {
            reason = 4;
            why = "jitter not known yet";
        }
        else if (!upscalingEarly && targetUav == nullptr)
        {
            reason = 8;
            why = "can't write the TAA output";
        }
        else if (early.createTried && !early.ready)
        {
            reason = 16;
            why = "the DLSS features for this size could not be made";
        }
        if (why != nullptr)
        {
            SkipOnce(reason, why);
            g_lastDecodePresent = UINT64_MAX;
            g_lastReplacedPresent = UINT64_MAX;
            return false;
        }
    }

    D3D11_MAPPED_SUBRESOURCE map {};
    if (FAILED(ctx->Map(g_decodeConstants, 0, D3D11_MAP_WRITE_DISCARD, 0, &map)))
        return false;
    // Matrix motion: this Present's copy (and the previous Present's, for composing), the frame depth and the object
    // flags at the TAA size.
    const auto& matrix = g_matrices[present & 1];
    const auto& previousMatrix = g_matrices[(present + 1) & 1];
    bool shaderIndexOk = false;
    if (ID3D11Texture2D* shaderIndex = TextureFromView(g_shaderIndex); shaderIndex != nullptr)
    {
        const auto indexDesc = Desc(shaderIndex);
        shaderIndexOk = indexDesc.Width == views.width && indexDesc.Height == views.height;
        shaderIndex->Release();
    }
    const bool useMatrix = depthUsable && shaderIndexOk && matrix.present == present;
    const bool usePreviousMatrix = useMatrix && hasPrevious && previousMatrix.present + 1 == present;
    static bool loggedMatrix = false;
    if (useMatrix && !loggedMatrix)
    {
        loggedMatrix = true;
        LOG_INFO("FC5 TAA: motion from depth and the game's current-to-previous matrix (float, full size)");
    }
    if (useMatrix)
        ++g_matrixFrames;
    static constexpr uint32_t kMatrixModes[3] = { 3u, 2u, 1u };
    const uint32_t matrixMode = useMatrix ? kMatrixModes[g_objectMotion.load(std::memory_order_relaxed)] : 0u;
    const uint32_t constants[4] = { hasPrevious ? 1u : 0u, occlusion ? 1u : 0u, matrixMode,
                                    usePreviousMatrix ? 1u : 0u };
    std::memcpy(map.pData, constants, sizeof(constants));
    ctx->Unmap(g_decodeConstants, 0);
    g_current = next;
    const int current = g_current;

    // DLSS inputs. Below 100% (the game's upscale draw writes a larger eye size) the upscaling features run and
    // their output is copied over that draw's target; the game's TAA draw is skipped as with DLAA (its output is
    // unused). Never let the TAA draw run after DLSS was called before it: that lost the D3D11 device every time.
    const bool upscaling = g_eyeWidth > views.width && g_eyeHeight > views.height;
    Features& set = upscaling ? g_upscale : g_dlaa;
    const UINT outWidth = upscaling ? g_eyeWidth : views.width;
    const UINT outHeight = upscaling ? g_eyeHeight : views.height;
    bool dlaa = mode == Dlaa;
    int eyeIndex = 0;
    bool reset = false;
    float jitterX = 0.0f;
    float jitterY = 0.0f;
    if (dlaa)
    {
        if (!depthUsable)
        {
            SkipOnce(1, "no G-buffer depth at the TAA size this frame");
            dlaa = false;
        }
        else if (set.ready && (views.width != set.width || views.height != set.height || outWidth != set.outWidth ||
                               outHeight != set.outHeight))
        {
            SkipOnce(2, "the render or eye size changed since the DLSS features were made; waiting for it to settle");
            dlaa = false;
        }
        else if (!g_jitterKnown[present & 1] && !AppliedFor(present, jitterX, jitterY))
        {
            SkipOnce(4, "jitter not known yet");
            dlaa = false;
        }
        else if (!upscaling && targetUav == nullptr)
        {
            SkipOnce(8, "can't write the TAA output");
            dlaa = false;
        }
        if (!dlaa)
            g_lastReplacedPresent = UINT64_MAX;
    }
    if (dlaa)
    {
        if (!hasPrevious || g_lastReplacedPresent == UINT64_MAX || g_lastReplacedPresent + 1 != present)
        {
            g_nextEye = 0;
            for (auto& eye : set.eyes)
                eye.lastPresent = UINT64_MAX;
        }
        eyeIndex = g_nextEye;
        reset = !hasPrevious || set.eyes[eyeIndex].lastPresent == UINT64_MAX ||
                set.eyes[eyeIndex].lastPresent + 2 != present;
        const int variant = g_jitterVariant.load(std::memory_order_relaxed);
        float shiftX = g_jitter[present & 1][0];
        float shiftY = g_jitter[present & 1][1];
        const bool own = AppliedFor(present, shiftX, shiftY);
        jitterX = kJitterSigns[variant][0] * shiftX;
        jitterY = kJitterSigns[variant][1] * shiftY;
        if (own)
            ++g_ownJitterFrames;
    }

    LARGE_INTEGER frequency {};
    LARGE_INTEGER start {};
    LARGE_INTEGER end {};
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&start);
    bool replaced = false;
    bool viewWritten = false;
    bool dlssCalled = false;
    InSpareState(ctx,
                 [&]
                 {
                     ID3D11ShaderResourceView* srvs[5] = { views.srvs[0], g_set.motion[1 - current].srv, depthSrv,
                                                           previousDepthSrv, useMatrix ? g_shaderIndex : nullptr };
                     ID3D11UnorderedAccessView* uavs[2] = { g_set.motion[current].uav, g_set.composed.uav };
                     // Unused when their flag is off; the decode constants stand in so nothing is unbound.
                     ID3D11Buffer* matrices[2] = { useMatrix ? matrix.buffer : g_decodeConstants,
                                                   usePreviousMatrix ? previousMatrix.buffer : g_decodeConstants };
                     ctx->CSSetConstantBuffers(1, 2, matrices);
                     Dispatch(ctx, g_decode, 5, srvs, 2, uavs, g_decodeConstants, views.width, views.height);
                     if (depthUsable)
                     {
                         ctx->CopyResource(g_depthCopies[current].texture, depth);
                         g_depthCopies[current].present = present;
                     }

                     if (dlaa)
                     {
                         if (!set.createTried)
                             CreateDlss(ctx, set, views.width, views.height, outWidth, outHeight);
                         ID3D11Texture2D* motion =
                             hasPrevious ? g_set.composed.texture : g_set.motion[current].texture;
                         dlssCalled = set.ready;
                         if (set.ready && Evaluate(ctx, set, set.eyes[eyeIndex], reset, views.colour, motion, depth,
                                                   jitterX, jitterY))
                         {
                             // Upscaling: shown at the game's upscale draw (ReplaceUpscaleDraw).
                             if (!upscaling)
                                 Dispatch(ctx, g_copy, 1, &set.outputSrv, 1, &targetUav, nullptr, views.width,
                                          views.height);
                             replaced = true;
                         }
                     }
                     if (view != 0 && targetUav != nullptr)
                     {
                         ID3D11ShaderResourceView* viewSrvs[2] = {
                             view == 1 || !hasPrevious ? g_set.motion[current].srv : g_set.composed.srv, views.srvs[2]
                         };
                         Dispatch(ctx, g_view, 2, viewSrvs, 1, &targetUav, nullptr, views.width, views.height);
                         viewWritten = true;
                     }
                     return true;
                 });
    QueryPerformanceCounter(&end);
    g_lastDecodePresent = present;

    if (!dlaa)
        return viewWritten;
    if (!replaced)
    {
        g_lastReplacedPresent = UINT64_MAX;
        // Our passes are queued by now: in front of the game's TAA draw that draw must not run after them, whatever
        // failed (the evaluation, logged there). The eye shows the previous TAA output for this frame.
        static bool loggedUnreplaced = false;
        if (beforeGameDraw && !viewWritten && !loggedUnreplaced)
        {
            loggedUnreplaced = true;
            LOG_WARN("FC5 TAA: DLSS did not write this frame's picture after its passes were queued (created: {}); the "
                     "game's TAA draw is skipped all the same, the previous picture shows",
                     dlssCalled);
        }
        return viewWritten || beforeGameDraw || (upscaling && dlssCalled);
    }
    g_lastReplacedPresent = present;
    set.eyes[eyeIndex].lastPresent = present;
    g_nextEye = 1 - eyeIndex;
    if (upscaling)
        g_upscalePendingPresent = present;
    const uint64_t micros = (uint64_t) ((end.QuadPart - start.QuadPart) * 1000000 / frequency.QuadPart);
    ++g_evaluations;
    if (reset)
        ++g_resets;
    g_evalMicros += micros;
    g_evalMaxMicros = std::max(g_evalMaxMicros, micros);
    static bool logged = false;
    if (!logged)
    {
        logged = true;
        LOG_INFO("FC5 TAA: first {} evaluation {}x{} -> {}x{}, eye {}, jitter ({:+.3f}, {:+.3f}), {} us; game TAA "
                 "draw {}",
                 set.name, views.width, views.height, outWidth, outHeight, eyeIndex, jitterX, jitterY, micros,
                 SkipGameTaa() ? "skipped" : "still runs first");
    }
    return true;
}

// The eye-format texture the upscaling output is converted into (same size and format as the eye target, so a
// plain copy replaces the game's upscale). nullptr if the target doesn't allow it.
static ID3D11Texture2D* EyeCopyFor(ID3D11DeviceContext* ctx, const D3D11_TEXTURE2D_DESC& target)
{
    if (g_eyeCopy != nullptr)
    {
        const auto desc = Desc(g_eyeCopy);
        if (desc.Width == target.Width && desc.Height == target.Height && desc.Format == target.Format)
            return g_eyeCopy;
        // Another eye size (R.E.A.L. VR's resolution changed; only reached once the upscaling pair was rebuilt for
        // it): the old copy is retired, never released at once.
        const uint64_t now = g_present.load();
        if (g_eyeCopyUav != nullptr)
            g_deferred.push_back({ g_eyeCopyUav, now });
        g_deferred.push_back({ g_eyeCopy, now });
        LOG_INFO("FC5 TAA: eye copy {}x{} retired for {}x{}", desc.Width, desc.Height, target.Width, target.Height);
        g_eyeCopy = nullptr;
        g_eyeCopyUav = nullptr;
        g_eyeCopyTried = false;
    }
    if (g_eyeCopyTried)
        return nullptr;
    g_eyeCopyTried = true;
    if ((target.Format != DXGI_FORMAT_R8G8B8A8_UNORM && target.Format != DXGI_FORMAT_R8G8B8A8_TYPELESS) ||
        target.SampleDesc.Count != 1 || target.ArraySize != 1)
    {
        LOG_WARN("FC5 TAA: the eye target (format {}, {} samples, {} layers) can't be copied to; the game's upscale "
                 "stays",
                 (UINT) target.Format, target.SampleDesc.Count, target.ArraySize);
        return nullptr;
    }
    ID3D11Device* device = nullptr;
    ctx->GetDevice(&device);
    if (device == nullptr)
        return nullptr;
    D3D11_TEXTURE2D_DESC desc {};
    desc.Width = target.Width;
    desc.Height = target.Height;
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = target.Format;
    desc.SampleDesc.Count = 1;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
    D3D11_UNORDERED_ACCESS_VIEW_DESC uavDesc {};
    uavDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    uavDesc.ViewDimension = D3D11_UAV_DIMENSION_TEXTURE2D;
    if (FAILED(device->CreateTexture2D(&desc, nullptr, &g_eyeCopy)) ||
        FAILED(device->CreateUnorderedAccessView(g_eyeCopy, &uavDesc, &g_eyeCopyUav)))
    {
        LOG_ERROR("FC5 TAA: eye copy texture {}x{} failed", target.Width, target.Height);
        if (g_eyeCopy != nullptr)
            g_eyeCopy->Release();
        g_eyeCopy = nullptr;
        g_eyeCopyUav = nullptr;
    }
    device->Release();
    return g_eyeCopy;
}

// The game's upscale draw: the first draw of this Present reading its TAA output, into a render-target-only target
// (the eye image). Learns the eye size (its viewport); when this Present's upscaling output is ready, the draw still
// runs (skipping it lost the D3D11 device, with a draw or a copy in its place) and AfterUpscaleDraw overwrites its
// target with the upscaling output, converted to the target's format. Never skips the draw (returns false).
static bool ReplaceUpscaleDraw(ID3D11DeviceContext* ctx)
{
    if (g_inPass > 0 || !g_ready || ctx != g_context || g_taaOutput == nullptr)
        return false;
    const uint64_t present = g_present.load();
    if (g_taaOutputPresent != present || g_upscaleDrawPresent == present)
        return false;

    constexpr UINT kSlots = 8;
    ID3D11ShaderResourceView* srvs[kSlots] = {};
    ctx->PSGetShaderResources(0, kSlots, srvs);
    int slot = -1;
    DXGI_FORMAT srvFormat = DXGI_FORMAT_UNKNOWN;
    for (UINT i = 0; i < kSlots; ++i)
    {
        if (srvs[i] == nullptr)
            continue;
        if (slot < 0)
        {
            ID3D11Resource* resource = nullptr;
            srvs[i]->GetResource(&resource);
            if (resource != nullptr && resource == g_taaOutput)
            {
                slot = (int) i;
                D3D11_SHADER_RESOURCE_VIEW_DESC viewDesc {};
                srvs[i]->GetDesc(&viewDesc);
                srvFormat = viewDesc.Format;
            }
            if (resource != nullptr)
                resource->Release();
        }
        srvs[i]->Release();
    }
    if (slot < 0)
        return false;

    ID3D11RenderTargetView* rtv = nullptr;
    ctx->OMGetRenderTargets(1, &rtv, nullptr);
    ID3D11Texture2D* target = TextureFromView(rtv);
    D3D11_RENDER_TARGET_VIEW_DESC rtvDesc {};
    if (rtv != nullptr)
    {
        rtv->GetDesc(&rtvDesc);
        rtv->Release();
    }
    if (target == nullptr)
        return false;
    const auto desc = Desc(target);
    if ((desc.BindFlags & D3D11_BIND_SHADER_RESOURCE) != 0)
    {
        target->Release();
        return false;
    }
    g_upscaleDrawPresent = present;

    UINT viewports = 1;
    D3D11_VIEWPORT vp {};
    ctx->RSGetViewports(&viewports, &vp);
    if (viewports == 0)
    {
        vp.Width = (float) desc.Width;
        vp.Height = (float) desc.Height;
    }
    const UINT eyeWidth = (UINT) vp.Width;
    const UINT eyeHeight = (UINT) vp.Height;
    if (eyeWidth != g_eyeWidth || eyeHeight != g_eyeHeight)
    {
        static int logged = 0;
        if (logged++ < 20)
            LOG_INFO("FC5 TAA: the game's upscale draw reads the TAA output (t{}, view format {}) into {}x{} "
                     "(viewport at {},{}; texture {}x{} format {}, view format {})",
                     slot, (UINT) srvFormat, eyeWidth, eyeHeight, vp.TopLeftX, vp.TopLeftY, desc.Width, desc.Height,
                     (UINT) desc.Format, (UINT) rtvDesc.Format);
        g_eyeWidth = eyeWidth;
        g_eyeHeight = eyeHeight;
    }

    static const bool show = Config::Instance()->Fc5TaaUpscaleShow.value_or_default();
    // A plain copy: the whole target, same encoding in and out (the game reads and writes it through linear views).
    const bool copyable = vp.TopLeftX == 0.0f && vp.TopLeftY == 0.0f && eyeWidth == desc.Width &&
                          eyeHeight == desc.Height && srvFormat == DXGI_FORMAT_R8G8B8A8_UNORM &&
                          rtvDesc.Format == DXGI_FORMAT_R8G8B8A8_UNORM;
    bool replaced = false;
    if (show && g_upscalePendingPresent == present && g_upscale.ready && eyeWidth == g_upscale.outWidth &&
        eyeHeight == g_upscale.outHeight)
    {
        static bool loggedNotCopyable = false;
        if (!copyable && !loggedNotCopyable)
        {
            loggedNotCopyable = true;
            LOG_WARN("FC5 TAA: the upscale draw isn't a whole-target draw with linear views; the game's upscale stays");
        }
        if (copyable && EyeCopyFor(ctx, desc) != nullptr && g_overwriteTarget == nullptr)
        {
            target->AddRef();
            g_overwriteTarget = target;
        }
    }
    target->Release();
    return replaced;
}

// Right after the game's upscale draw: our upscaling output over its target.
static void AfterUpscaleDraw(ID3D11DeviceContext* ctx)
{
    ID3D11Texture2D* target = g_overwriteTarget;
    if (target == nullptr || ctx != g_context || g_inPass > 0)
        return;
    g_overwriteTarget = nullptr;
    const auto desc = Desc(target);
    const bool copied = InSpareState(ctx,
                                     [&]
                                     {
                                         Dispatch(ctx, g_copy, 1, &g_upscale.outputSrv, 1, &g_eyeCopyUav, nullptr,
                                                  desc.Width, desc.Height);
                                         ctx->CopyResource(target, g_eyeCopy);
                                         return true;
                                     });
    target->Release();
    if (!copied)
        return;
    ++g_drawsReplaced;
    static bool logged = false;
    if (!logged)
    {
        logged = true;
        LOG_INFO("FC5 TAA: first upscale draw overwritten ({}x{}, converted and copied after the game's draw)",
                 desc.Width, desc.Height);
    }
}

bool Fc5Taa_Dx11::ReplaceDraw(ID3D11DeviceContext* ctx)
{
    if (ReplaceUpscaleDraw(ctx))
        return true;
    return SkipGameTaa() && RunTaa(ctx, true);
}

void Fc5Taa_Dx11::AfterDraw(ID3D11DeviceContext* ctx)
{
    AfterUpscaleDraw(ctx);
    if (!SkipGameTaa())
        RunTaa(ctx, false);
    // After the TAA draw has run (or, when DLAA replaced it, after the next draw of that Present).
    if (g_pendingShare.set && g_inPass == 0 && ctx == g_context)
    {
        g_pendingShare.set = false;
        if (g_pendingShare.present == g_present.load())
            ShareDepth(ctx, g_pendingShare.width, g_pendingShare.height);
    }
}

void Fc5Taa_Dx11::OnPresent()
{
    Pace(); // before the game's Present goes out
    ++g_present;
    RecordPresent();
    static bool initialised = false;
    if (!initialised)
    {
        initialised = true;
        g_ownJitter = Config::Instance()->Fc5TaaOwnJitter.value_or_default();
    }
}

void Fc5Taa_Dx11::CycleMode()
{
    g_mode = g_mode.load() == Dlaa ? GameTaa : Dlaa;
    LOG_INFO("FC5 TAA: mode {}", ModeName());
}

const char* Fc5Taa_Dx11::ModeName() { return g_mode.load() == Dlaa ? "DLSS (per eye)" : "game TAA"; }

void Fc5Taa_Dx11::RequestRebuild() { g_rebuildRequested = true; }

std::string Fc5Taa_Dx11::StatusText()
{
    if (g_mode.load() != Dlaa)
        return "DLSS: off, the game's TAA shows; its memory is freed after 10 s";
    std::lock_guard lock(g_statusMutex);
    return g_statusText;
}

void Fc5Taa_Dx11::CycleJitter()
{
    g_jitterVariant = (g_jitterVariant.load() + 1) % 4;
    LOG_INFO("FC5 TAA: jitter {}", JitterName());
}

const char* Fc5Taa_Dx11::JitterName()
{
    switch (g_jitterVariant.load())
    {
    case 1:
        return "(-x, -y)";
    case 2:
        return "(+x, -y)";
    case 3:
        return "(-x, +y)";
    default:
        return "(+x, +y)";
    }
}

void Fc5Taa_Dx11::CycleView()
{
    g_viewMode = (g_viewMode.load() + 1) % 3;
    LOG_INFO("FC5 TAA: motion view {}", ViewName());
}

const char* Fc5Taa_Dx11::ViewName()
{
    switch (g_viewMode.load())
    {
    case 1:
        return "decoded (to the previous Present)";
    case 2:
        return "composed (to two Presents back)";
    default:
        return "off";
    }
}

void Fc5Taa_Dx11::ToggleOcclusion()
{
    g_occlusion = !g_occlusion.load();
    LOG_INFO("FC5 TAA: occlusion-aware composition {}", g_occlusion.load() ? "on" : "off");
}

bool Fc5Taa_Dx11::OcclusionOn() { return g_occlusion.load(); }

void Fc5Taa_Dx11::OnMap(ID3D11Resource* resource, UINT subresource, D3D11_MAP type, void* data)
{
    if (data == nullptr || (type != D3D11_MAP_WRITE_DISCARD && type != D3D11_MAP_WRITE))
        return;
    const bool view = (OwnJitterActive() || StaleEyeShare::Enabled()) && IsViewConstants(resource);
    const bool temporal = !view && StaleEyeShare::Enabled() && IsTemporalConstants(resource);
    if (!view && !temporal)
        return;
    for (auto& map : t_maps)
    {
        if (map.resource == nullptr || (map.resource == resource && map.subresource == subresource))
        {
            map = { resource, subresource, data, temporal };
            return;
        }
    }
}

void Fc5Taa_Dx11::BeforeUnmap(ID3D11Resource* resource, UINT subresource)
{
    if (resource == nullptr)
        return;
    for (auto& map : t_maps)
    {
        if (map.resource != resource || map.subresource != subresource)
            continue;
        if (map.temporal)
        {
            alignas(16) float temporal[kTemporalConstantsBytes / 4];
            std::memcpy(temporal, map.data, kTemporalConstantsBytes);
            PublishIfMotionConstants(temporal);
            map = {};
            return;
        }
        // The mapped memory can be write-combined: patch a copy and write it back whole.
        alignas(16) float copy[kViewConstantsBytes / 4];
        std::memcpy(copy, map.data, kViewConstantsBytes);
        if (StaleEyeShare::Enabled())
            NoteViewProjection(copy);
        if (OwnJitterActive() && PatchViewConstants(copy))
        {
            std::memcpy(map.data, copy, kViewConstantsBytes);
            ++g_patchedMaps;
        }
        map = {};
        return;
    }
}

const void* Fc5Taa_Dx11::PatchUpdate(ID3D11Resource* resource, UINT subresource, const D3D11_BOX* box,
                                     const void* data)
{
    if (data != nullptr && box == nullptr && subresource == 0 && StaleEyeShare::Enabled() &&
        IsTemporalConstants(resource))
    {
        alignas(16) float temporal[kTemporalConstantsBytes / 4];
        std::memcpy(temporal, data, kTemporalConstantsBytes);
        PublishIfMotionConstants(temporal);
        return data;
    }
    if (data != nullptr && box == nullptr && subresource == 0 && StaleEyeShare::Enabled() && IsViewConstants(resource))
        NoteViewProjection(static_cast<const float*>(data));
    if (data == nullptr || box != nullptr || subresource != 0 || !OwnJitterActive() || !IsViewConstants(resource))
        return data;
    thread_local float copy[kViewConstantsBytes / 4];
    std::memcpy(copy, data, kViewConstantsBytes);
    if (!PatchViewConstants(copy))
        return data;
    ++g_patchedUpdates;
    return copy;
}

void Fc5Taa_Dx11::ToggleMatrixEverywhere()
{
    g_objectMotion = (g_objectMotion.load() + 1) % 3;
    LOG_INFO("FC5 TAA: motion for object-flagged pixels: {}", ObjectMotionName());
}

const char* Fc5Taa_Dx11::ObjectMotionName()
{
    switch (g_objectMotion.load())
    {
    case 1:
        return "matrix (object flag ignored)";
    case 2:
        return "the game's 8-bit vectors (old)";
    default:
        return "the game's vectors where they disagree with the matrix (default)";
    }
}

void Fc5Taa_Dx11::ToggleOwnJitter()
{
    // game -> own -> own x20 (DLSS given the game's) -> own x4 (DLSS given ours) -> game
    if (!g_ownJitter.load())
    {
        g_shakeTest = 0;
        g_ownJitter = true;
    }
    else if (g_shakeTest.load() < 2)
        ++g_shakeTest;
    else
    {
        g_shakeTest = 0;
        g_ownJitter = false;
    }
    const int shake = g_shakeTest.load();
    LOG_INFO("FC5 TAA: jitter {} (view constants patched {} times: {} maps, {} updates)",
             !g_ownJitter.load() ? "the game's"
             : shake == 1        ? "SHAKE TEST: own x20 patched, DLSS given the game's"
             : shake == 2        ? "SHAKE TEST 2: own x4 patched and given to DLSS (still = aligned)"
                                 : "own per-eye Halton(2,3) x8",
             g_patchedWrites, g_patchedMaps, g_patchedUpdates);
}

bool Fc5Taa_Dx11::OwnJitterOn() { return g_ownJitter.load(); }
