#include "pch.h"
#include "FinalImageNr_Dx11.h"

#include "Config.h"
#include "Logger.h"
#include "NVNGX_Parameter.h"
#include "resource.h"
#include "State.h"
#include "Util.h"
#include "inputs/TaaDlaa_Dx11.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <vector>

namespace
{
// Everything here is touched from the thread that drives the gameplay context and calls Present (the first
// gameplay bind and the first Present log their thread ids to confirm they match).

// Set while this backend evaluates or copies: DLSS runs NR synchronously and the D3D11 bridge binds and restores
// targets through the hooked context functions. Per thread.
thread_local int g_inEvaluate = 0;

struct EvaluateScope
{
    EvaluateScope() { ++g_inEvaluate; }
    ~EvaluateScope() { --g_inEvaluate; }
};

uint64_t g_present = 0;
uint64_t g_evaluations = 0;

// The context that draws gameplay, adopted on its first G-buffer bind.
ID3D11DeviceContext* g_context = nullptr;

// Gameplay signature: two or more targets bound with eye-sized depth of the same size (the G-buffer). Menus and
// loading screens don't draw it. Its depth is the scene depth DLSS reads: the largest such depth of the frame,
// taken at Present, since half-resolution passes match too once their size clears the eye-size limits (Days Gone
// at 4240x4240 binds two targets with 2120x2120 depth).
uint64_t g_gameplayPresent = 0;
ID3D11Texture2D* g_frameDepth = nullptr;
ID3D11Texture2D* g_sceneDepth = nullptr;
constexpr uint64_t kGameplayRecentPresents = 30;

// Candidates for the final image on the gameplay context: the bound colour target and the compute UAV in slot 0,
// kept only while they are eye-sized 8/10-bit colour. The last one a draw or dispatch wrote is the final image.
ID3D11Texture2D* g_boundColor = nullptr;
ID3D11Texture2D* g_csUav0 = nullptr;
ID3D11Texture2D* g_lastWrite = nullptr;

// Reloads reallocate the game's targets; references are dropped a while later rather than immediately.
struct DeferredRelease
{
    ID3D11Texture2D* texture;
    uint64_t present;
};
std::vector<DeferredRelease> g_deferred;
constexpr uint64_t kReleaseAfterPresents = 600;

NVSDK_NGX_Parameter* g_params = nullptr;
NVSDK_NGX_Handle* g_handle = nullptr;
ID3D11Texture2D* g_input = nullptr;
ID3D11Texture2D* g_output = nullptr;
ID3D11Texture2D* g_motion = nullptr;
bool g_createTried = false;
UINT g_width = 0;
UINT g_height = 0;
DXGI_FORMAT g_format = DXGI_FORMAT_UNKNOWN;

// Sizes are decided once, so only from a settled game.
constexpr ULONGLONG kStableSetupMs = 5000;
constexpr ULONGLONG kStableGapMs = 1000;
UINT g_candidateW = 0;
UINT g_candidateH = 0;
ULONGLONG g_candidateSince = 0;
ULONGLONG g_candidateLast = 0;

bool IsStableSetup(UINT width, UINT height)
{
    const ULONGLONG now = GetTickCount64();
    if (g_candidateW != width || g_candidateH != height || now - g_candidateLast > kStableGapMs)
    {
        g_candidateW = width;
        g_candidateH = height;
        g_candidateSince = now;
    }
    g_candidateLast = now;
    return now - g_candidateSince >= kStableSetupMs;
}

void Retain(ID3D11Texture2D** slot, ID3D11Texture2D* value)
{
    if (*slot == value)
        return;
    if (value != nullptr)
        value->AddRef();
    if (*slot != nullptr)
        (*slot)->Release();
    *slot = value;
}

void Take(ID3D11Texture2D** slot)
{
    if (*slot != nullptr)
        (*slot)->Release();
    *slot = nullptr;
}

void RetainDeferred(ID3D11Texture2D** slot, ID3D11Texture2D* value)
{
    if (*slot == value)
        return;
    if (value != nullptr)
        value->AddRef();
    if (*slot != nullptr)
        g_deferred.push_back({ *slot, g_present });
    *slot = value;
}

void ReleaseDeferred()
{
    for (size_t i = 0; i < g_deferred.size();)
    {
        if (g_present - g_deferred[i].present >= kReleaseAfterPresents)
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

// Returns an AddRef'd texture, or nullptr.
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

// A target passes on either axis ([Inputs] DuniaTaaMinEvalWidth / DuniaTaaMinEvalHeight, shared with FC5/FC6).
bool IsEyeSize(UINT width, UINT height)
{
    const auto minW = (UINT) std::max(0, Config::Instance()->DuniaTaaMinEvalWidth.value_or_default());
    const auto minH = (UINT) std::max(0, Config::Instance()->DuniaTaaMinEvalHeight.value_or_default());
    return width >= minW || height >= minH;
}

// The typed UNORM format for copies and DLSS of an 8/10-bit colour image, or UNKNOWN for anything else
// (HDR float targets, depth, integer buffers).
DXGI_FORMAT ColourUnorm(DXGI_FORMAT format)
{
    switch (format)
    {
    case DXGI_FORMAT_R8G8B8A8_TYPELESS:
    case DXGI_FORMAT_R8G8B8A8_UNORM:
    case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
        return DXGI_FORMAT_R8G8B8A8_UNORM;
    case DXGI_FORMAT_B8G8R8A8_TYPELESS:
    case DXGI_FORMAT_B8G8R8A8_UNORM:
    case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
        return DXGI_FORMAT_B8G8R8A8_UNORM;
    case DXGI_FORMAT_R10G10B10A2_TYPELESS:
    case DXGI_FORMAT_R10G10B10A2_UNORM:
        return DXGI_FORMAT_R10G10B10A2_UNORM;
    default:
        return DXGI_FORMAT_UNKNOWN;
    }
}

bool IsFinalCandidate(ID3D11Texture2D* texture)
{
    if (texture == nullptr)
        return false;
    const auto desc = Desc(texture);
    return ColourUnorm(desc.Format) != DXGI_FORMAT_UNKNOWN && IsEyeSize(desc.Width, desc.Height) &&
           desc.SampleDesc.Count == 1 && desc.ArraySize == 1;
}

bool IsGameplayBind(UINT numRtvs, ID3D11RenderTargetView* const* rtvs, ID3D11Texture2D* color,
                    ID3D11Texture2D* depth)
{
    if (numRtvs < 2 || numRtvs > D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT || rtvs == nullptr || rtvs[1] == nullptr ||
        color == nullptr || depth == nullptr)
        return false;
    const auto colorDesc = Desc(color);
    const auto depthDesc = Desc(depth);
    return colorDesc.Width == depthDesc.Width && colorDesc.Height == depthDesc.Height &&
           IsEyeSize(depthDesc.Width, depthDesc.Height) && depthDesc.SampleDesc.Count == 1;
}

bool CreateTexture(ID3D11Device* device, UINT width, UINT height, DXGI_FORMAT format, UINT bindFlags,
                   ID3D11Texture2D** out, const char* name)
{
    D3D11_TEXTURE2D_DESC desc {};
    desc.Width = width;
    desc.Height = height;
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = format;
    desc.SampleDesc.Count = 1;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = bindFlags;
    if (FAILED(device->CreateTexture2D(&desc, nullptr, out)))
    {
        LOG_ERROR("Final image NR: {} texture create failed (fmt {})", name, (UINT) format);
        return false;
    }
    return true;
}

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
        LOG_ERROR("Final image NR: NGX init failed: {:X}", (UINT) result);
        return false;
    }
    return true;
}

// Once per session: creation-time settings are startup-only, and a live rebuild is not attempted.
void CreateDlss(ID3D11DeviceContext* ctx, UINT width, UINT height, DXGI_FORMAT format)
{
    g_createTried = true;
    g_width = width;
    g_height = height;
    g_format = format;

    ID3D11Device* device = nullptr;
    ctx->GetDevice(&device);
    if (device == nullptr)
        return;

    EvaluateScope scope;
    constexpr UINT kUavSrv = D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;
    bool ok = CreateTexture(device, width, height, format, D3D11_BIND_SHADER_RESOURCE, &g_input, "input") &&
              CreateTexture(device, width, height, format, kUavSrv, &g_output, "output") &&
              CreateTexture(device, width, height, DXGI_FORMAT_R16G16_FLOAT, kUavSrv, &g_motion, "motion") &&
              InitNgx(device);
    if (ok)
    {
        ID3D11UnorderedAccessView* uav = nullptr;
        if (SUCCEEDED(device->CreateUnorderedAccessView(g_motion, nullptr, &uav)) && uav != nullptr)
        {
            const float zero[4] = {};
            ctx->ClearUnorderedAccessViewFloat(uav, zero);
            uav->Release();
        }
    }
    device->Release();

    if (ok && g_params == nullptr && NVSDK_NGX_D3D11_GetCapabilityParameters(&g_params) != NVSDK_NGX_Result_Success)
    {
        LOG_ERROR("Final image NR: GetCapabilityParameters failed");
        ok = false;
    }

    if (ok)
    {
        // The picture is tonemapped: no IsHDR, so NR also leaves its colour transform off.
        UINT flags = NVSDK_NGX_DLSS_Feature_Flags_AutoExposure;
        if (Config::Instance()->DuniaTaaDepthInverted.value_or_default())
            flags |= NVSDK_NGX_DLSS_Feature_Flags_DepthInverted;
        g_params->Set(NVSDK_NGX_Parameter_DLSS_Feature_Create_Flags, flags);
        g_params->Set(NVSDK_NGX_Parameter_Width, width);
        g_params->Set(NVSDK_NGX_Parameter_Height, height);
        g_params->Set(NVSDK_NGX_Parameter_OutWidth, width);
        g_params->Set(NVSDK_NGX_Parameter_OutHeight, height);
        g_params->Set(NVSDK_NGX_Parameter_PerfQualityValue, NVSDK_NGX_PerfQuality_Value_DLAA);

        const auto result = NVSDK_NGX_D3D11_CreateFeature(ctx, NVSDK_NGX_Feature_SuperSampling, g_params, &g_handle);
        if (result != NVSDK_NGX_Result_Success)
        {
            LOG_ERROR("Final image NR: CreateFeature DLAA failed: {:X}", (UINT) result);
            g_handle = nullptr;
            ok = false;
        }
    }

    if (ok)
        LOG_INFO("Final image NR: created DLAA {}x{} fmt {} on context {:X} (depth {})", width, height, (UINT) format,
                 (size_t) ctx, Config::Instance()->DuniaTaaDepthInverted.value_or_default() ? "inverted" : "normal");
}

bool Evaluate(ID3D11DeviceContext* ctx)
{
    EvaluateScope scope;
    g_params->Set(NVSDK_NGX_Parameter_Jitter_Offset_X, 0.0f);
    g_params->Set(NVSDK_NGX_Parameter_Jitter_Offset_Y, 0.0f);
    g_params->Set(NVSDK_NGX_Parameter_MV_Scale_X, 1.0f);
    g_params->Set(NVSDK_NGX_Parameter_MV_Scale_Y, 1.0f);
    g_params->Set(NVSDK_NGX_Parameter_DLSS_Exposure_Scale, 1.0f);
    g_params->Set(NVSDK_NGX_Parameter_Reset, 1);
    g_params->Set(NVSDK_NGX_Parameter_Width, g_width);
    g_params->Set(NVSDK_NGX_Parameter_Height, g_height);
    g_params->Set(NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Width, g_width);
    g_params->Set(NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Height, g_height);
    g_params->Set(NVSDK_NGX_Parameter_Depth, g_sceneDepth);
    g_params->Set(NVSDK_NGX_Parameter_Color, g_input);
    g_params->Set(NVSDK_NGX_Parameter_MotionVectors, g_motion);
    g_params->Set(NVSDK_NGX_Parameter_Output, g_output);

    State::Instance().setInputApiName = ApiUpscalerInput::FSR2X_DX11;

    const auto result = NVSDK_NGX_D3D11_EvaluateFeature(ctx, g_handle, g_params, nullptr);
    if (result != NVSDK_NGX_Result_Success)
    {
        static NVSDK_NGX_Result last = NVSDK_NGX_Result_Success;
        if (result != last)
        {
            last = result;
            LOG_ERROR("Final image NR: EvaluateFeature failed: {:X}", (UINT) result);
        }
        return false;
    }
    return true;
}

uint64_t g_evalMicros = 0;
uint64_t g_evalMaxMicros = 0;

void ProcessFinal(ID3D11DeviceContext* ctx, ID3D11Texture2D* picture)
{
    // Nothing to gain without NR: no motion vectors, so DLAA alone would only add its cost.
    if (!Config::Instance()->DlssNrEnabled.value_or_default())
        return;
    // TaaDlaa_Dx11 already ran DLSS on this frame in place of the game's TAA.
    if (TaaDlaa_Dx11::ReplacedThisPresent())
        return;
    if (g_gameplayPresent == 0 || g_present + 1 - g_gameplayPresent > kGameplayRecentPresents)
        return;

    const auto desc = Desc(picture);
    const DXGI_FORMAT format = ColourUnorm(desc.Format);
    if (!g_createTried)
    {
        if (IsStableSetup(desc.Width, desc.Height))
            CreateDlss(ctx, desc.Width, desc.Height, format);
        return;
    }
    if (g_handle == nullptr)
        return;
    if (desc.Width != g_width || desc.Height != g_height || format != g_format)
    {
        static bool logged = false;
        if (!logged)
        {
            logged = true;
            LOG_WARN("Final image NR: final image now {}x{} fmt {}, set up for {}x{} fmt {}; restart to match",
                     desc.Width, desc.Height, (UINT) desc.Format, g_width, g_height, (UINT) g_format);
        }
        return;
    }
    if (g_sceneDepth == nullptr)
        return;
    const auto depthDesc = Desc(g_sceneDepth);
    if (depthDesc.Width != g_width || depthDesc.Height != g_height)
    {
        static bool logged = false;
        if (!logged)
        {
            logged = true;
            LOG_WARN("Final image NR: scene depth is {}x{}, not {}x{}; skipping", depthDesc.Width, depthDesc.Height,
                     g_width, g_height);
        }
        return;
    }

    LARGE_INTEGER frequency {};
    LARGE_INTEGER start {};
    LARGE_INTEGER end {};
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&start);
    bool ok = false;
    {
        EvaluateScope scope;
        ctx->CopyResource(g_input, picture);
        ok = Evaluate(ctx);
        if (ok)
            ctx->CopyResource(picture, g_output);
    }
    QueryPerformanceCounter(&end);
    if (!ok)
        return;

    const uint64_t micros = (uint64_t) ((end.QuadPart - start.QuadPart) * 1000000 / frequency.QuadPart);
    g_evalMicros += micros;
    g_evalMaxMicros = std::max(g_evalMaxMicros, micros);
    ++g_evaluations;
    static bool logged = false;
    if (!logged)
    {
        logged = true;
        LOG_INFO("Final image NR: first evaluation at Present on {}x{} fmt {} (thread {})", desc.Width, desc.Height,
                 (UINT) desc.Format, GetCurrentThreadId());
    }
}
} // namespace

bool FinalImageNr_Dx11::Enabled()
{
    static int cached = -1;
    if (cached < 0)
        cached = Config::Instance()->FinalImageNr.value_or_default() ? 1 : 0;
    return cached == 1;
}

void FinalImageNr_Dx11::OnBind(ID3D11DeviceContext* ctx, UINT numRtvs, ID3D11RenderTargetView* const* rtvs,
                               ID3D11DepthStencilView* dsv)
{
    if (g_inEvaluate > 0 || ctx == nullptr)
        return;
    // Once DLSS exists on a context it stays there; before that, follow whichever context draws gameplay.
    if (ctx != g_context && (g_createTried || ctx->GetType() != D3D11_DEVICE_CONTEXT_IMMEDIATE))
        return;

    ID3D11Texture2D* color =
        (numRtvs > 0 && numRtvs <= D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT && rtvs != nullptr) ? TextureFromView(rtvs[0])
                                                                                              : nullptr;
    ID3D11Texture2D* depth = TextureFromView(dsv);

    if (IsGameplayBind(numRtvs, rtvs, color, depth))
    {
        if (ctx != g_context)
        {
            LOG_INFO("Final image NR: following context {:X} (thread {}), gameplay depth {}x{} fmt {}", (size_t) ctx,
                     GetCurrentThreadId(), Desc(depth).Width, Desc(depth).Height, (UINT) Desc(depth).Format);
            ctx->AddRef();
            if (g_context != nullptr)
                g_context->Release();
            g_context = ctx;
            Take(&g_boundColor);
            Take(&g_csUav0);
            Take(&g_lastWrite);
        }
        g_gameplayPresent = g_present + 1;
        if (g_frameDepth == nullptr ||
            (UINT64) Desc(depth).Width * Desc(depth).Height > (UINT64) Desc(g_frameDepth).Width * Desc(g_frameDepth).Height)
            Retain(&g_frameDepth, depth);
    }

    if (ctx == g_context)
        Retain(&g_boundColor, IsFinalCandidate(color) ? color : nullptr);

    if (color != nullptr)
        color->Release();
    if (depth != nullptr)
        depth->Release();
}

void FinalImageNr_Dx11::OnDraw(ID3D11DeviceContext* ctx)
{
    if (g_inEvaluate > 0 || ctx != g_context || g_boundColor == nullptr)
        return;
    Retain(&g_lastWrite, g_boundColor);
}

void FinalImageNr_Dx11::OnCsUavs(ID3D11DeviceContext* ctx, UINT start, UINT count,
                                 ID3D11UnorderedAccessView* const* uavs)
{
    if (g_inEvaluate > 0 || ctx != g_context || start != 0 || count == 0)
        return;
    ID3D11Texture2D* texture = uavs != nullptr ? TextureFromView(uavs[0]) : nullptr;
    Retain(&g_csUav0, IsFinalCandidate(texture) ? texture : nullptr);
    if (texture != nullptr)
        texture->Release();
}

void FinalImageNr_Dx11::OnDispatch(ID3D11DeviceContext* ctx)
{
    if (g_inEvaluate > 0 || ctx != g_context || g_csUav0 == nullptr)
        return;
    Retain(&g_lastWrite, g_csUav0);
}

ID3D11Texture2D* FinalImageNr_Dx11::FrameDepth() { return g_frameDepth; }

void FinalImageNr_Dx11::OnPresent()
{
    ++g_present;
    ReleaseDeferred();
    if (g_frameDepth != nullptr)
    {
        RetainDeferred(&g_sceneDepth, g_frameDepth);
        Take(&g_frameDepth);
    }

    static bool loggedThread = false;
    if (!loggedThread && g_context != nullptr)
    {
        loggedThread = true;
        LOG_INFO("Final image NR: first Present after gameplay on thread {}", GetCurrentThreadId());
    }

    if (g_lastWrite != nullptr && g_context != nullptr)
    {
        static bool logged = false;
        if (!logged)
        {
            const auto desc = Desc(g_lastWrite);
            logged = true;
            LOG_INFO("Final image NR: last eye-sized write before Present is {:X} {}x{} fmt {} bind {:X}",
                     (size_t) g_lastWrite, desc.Width, desc.Height, (UINT) desc.Format, desc.BindFlags);
        }
        ProcessFinal(g_context, g_lastWrite);
    }
    Take(&g_lastWrite);

    // One eye per Present, so one evaluation per Present is the expected ratio.
    static uint64_t windowStart = 0;
    constexpr uint64_t kWindow = 900;
    if (g_evaluations == 0)
    {
        windowStart = g_present;
        return;
    }
    if (g_present - windowStart >= kWindow)
    {
        const double avgMs = (double) g_evalMicros / (double) g_evaluations / 1000.0;
        LOG_INFO("Final image NR: {} evaluations over the last {} presents, CPU {:.2f} ms average, {:.2f} ms worst",
                 g_evaluations, g_present - windowStart, avgMs, (double) g_evalMaxMicros / 1000.0);
        g_evaluations = 0;
        g_evalMicros = 0;
        g_evalMaxMicros = 0;
        windowStart = g_present;
    }
}
