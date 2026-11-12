#include "pch.h"
#include "EyeImageNr_Dx11.h"

#include "Config.h"
#include "Logger.h"
#include "NVNGX_Parameter.h"
#include "resource.h"
#include "State.h"
#include "Util.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace
{
// Set while this backend evaluates or copies. DLSS runs NR synchronously and the D3D11 bridge binds and
// restores targets through the hooked context functions. Per thread.
thread_local int g_inEvaluate = 0;

struct EvaluateScope
{
    EvaluateScope() { ++g_inEvaluate; }
    ~EvaluateScope() { --g_inEvaluate; }
};

std::atomic<uint64_t> g_present { 0 };
std::atomic<uint64_t> g_evaluations { 0 };

// Recording state. Deferred contexts record on several threads at once, so everything shared between them and
// the immediate context sits behind g_mutex.
std::mutex g_mutex;

// Per deferred context, since its last FinishCommandList: the final picture it bound, and whether it drew into it.
struct Recording
{
    ID3D11Texture2D* target = nullptr;
    bool drew = false;
};
std::unordered_map<ID3D11DeviceContext*, Recording> g_recording;

// Finished lists that draw the final picture, waiting for the immediate context to run them. Holds a reference.
std::unordered_map<ID3D11CommandList*, ID3D11Texture2D*> g_marked;

// The deferred context whose final-picture bind is current on this thread, until its next bind or draw.
thread_local ID3D11DeviceContext* t_finalCtx = nullptr;

// Dark Souls III draws the final picture on the immediate context instead (the rest of the frame is still deferred).
// Bound there: held until the next bind on that context or Present, then processed if a draw went into it.
ID3D11DeviceContext* g_immCtx = nullptr;
ID3D11Texture2D* g_immFinal = nullptr;
bool g_immDrew = false;

// Gameplay signature: the G-buffer bind (four or more targets plus depth, eye-sized). Menus and loading screens
// don't draw it. Its depth is the scene depth DLSS reads.
std::atomic<uint64_t> g_gameplayPresent { 0 };
ID3D11Texture2D* g_sceneDepth = nullptr;
constexpr uint64_t kGameplayRecentPresents = 30;

// Reloads reallocate the game's targets; references are dropped a while later rather than immediately.
struct DeferredRelease
{
    ID3D11Texture2D* texture;
    uint64_t present;
};
std::vector<DeferredRelease> g_deferred;
constexpr uint64_t kReleaseAfterPresents = 600;

// Below here: only the immediate context that runs the final list touches these (the game serialises its use).
ID3D11DeviceContext* g_context = nullptr;
NVSDK_NGX_Parameter* g_params = nullptr;
NVSDK_NGX_Handle* g_handle = nullptr;
ID3D11Texture2D* g_input = nullptr;
ID3D11Texture2D* g_output = nullptr;
ID3D11Texture2D* g_motion = nullptr;
bool g_createTried = false;
UINT g_width = 0;
UINT g_height = 0;

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

// Called with g_mutex held.
void RetainDeferred(ID3D11Texture2D** slot, ID3D11Texture2D* value)
{
    if (*slot == value)
        return;
    if (value != nullptr)
        value->AddRef();
    if (*slot != nullptr)
        g_deferred.push_back({ *slot, g_present.load() });
    *slot = value;
}

// Called with g_mutex held.
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

bool IsRgba8(DXGI_FORMAT format)
{
    return format == DXGI_FORMAT_R8G8B8A8_TYPELESS || format == DXGI_FORMAT_R8G8B8A8_UNORM ||
           format == DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
}

bool IsGameplayBind(UINT numRtvs, ID3D11RenderTargetView* const* rtvs, ID3D11Texture2D* color,
                    ID3D11Texture2D* depth)
{
    if (numRtvs < 4 || numRtvs > D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT || rtvs == nullptr || color == nullptr ||
        depth == nullptr)
        return false;
    for (UINT i = 0; i < 4; ++i)
    {
        if (rtvs[i] == nullptr)
            return false;
    }
    const auto colorDesc = Desc(color);
    const auto depthDesc = Desc(depth);
    return colorDesc.Width == depthDesc.Width && colorDesc.Height == depthDesc.Height &&
           IsEyeSize(colorDesc.Width, colorDesc.Height) && colorDesc.SampleDesc.Count == 1;
}

// The final picture: eye-sized 8-bit colour that is only ever drawn into, never sampled, bound alone without depth.
bool IsFinalBind(UINT numRtvs, ID3D11Texture2D* color, ID3D11Texture2D* depth)
{
    if (numRtvs != 1 || color == nullptr || depth != nullptr)
        return false;
    const auto desc = Desc(color);
    return (desc.BindFlags & D3D11_BIND_SHADER_RESOURCE) == 0 && IsRgba8(desc.Format) &&
           IsEyeSize(desc.Width, desc.Height) && desc.SampleDesc.Count == 1 && desc.ArraySize == 1 &&
           desc.MipLevels == 1;
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
        LOG_ERROR("Eye image NR: {} texture create failed", name);
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
        LOG_ERROR("Eye image NR: NGX init failed: {:X}", (UINT) result);
        return false;
    }
    return true;
}

// Once per session: creation-time settings are startup-only, and a live rebuild is not attempted.
void CreateDlss(ID3D11DeviceContext* ctx, UINT width, UINT height)
{
    g_createTried = true;
    g_width = width;
    g_height = height;

    ID3D11Device* device = nullptr;
    ctx->GetDevice(&device);
    if (device == nullptr)
        return;

    EvaluateScope scope;
    // The game's picture is render-target only; DLSS reads a copy.
    constexpr UINT kUavSrv = D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;
    bool ok = CreateTexture(device, width, height, DXGI_FORMAT_R8G8B8A8_UNORM, D3D11_BIND_SHADER_RESOURCE, &g_input,
                            "input") &&
              CreateTexture(device, width, height, DXGI_FORMAT_R8G8B8A8_UNORM, kUavSrv, &g_output, "output") &&
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
        LOG_ERROR("Eye image NR: GetCapabilityParameters failed");
        ok = false;
    }

    if (ok)
    {
        // The picture is tonemapped 8-bit: no IsHDR, so NR also leaves its colour transform off.
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
            LOG_ERROR("Eye image NR: CreateFeature DLAA failed: {:X}", (UINT) result);
            g_handle = nullptr;
            ok = false;
        }
    }

    if (ok)
        LOG_INFO("Eye image NR: created DLAA {}x{} on context {:X} (depth {})", width, height, (size_t) ctx,
                 Config::Instance()->DuniaTaaDepthInverted.value_or_default() ? "inverted" : "normal");
}

bool Evaluate(ID3D11DeviceContext* ctx, ID3D11Texture2D* depth)
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
    g_params->Set(NVSDK_NGX_Parameter_Depth, depth);
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
            LOG_ERROR("Eye image NR: EvaluateFeature failed: {:X}", (UINT) result);
        }
        return false;
    }
    return true;
}

// CPU time on the game's render thread for the copies, DLSS + NR through the bridge and the copy back.
std::atomic<uint64_t> g_evalMicros { 0 };
std::atomic<uint64_t> g_evalMaxMicros { 0 };

void ProcessFinal(ID3D11DeviceContext* ctx, ID3D11Texture2D* picture)
{
    // Nothing to gain without NR: no motion vectors, so DLAA alone would only add its cost.
    if (!Config::Instance()->DlssNrEnabled.value_or_default())
        return;
    const uint64_t present = g_present.load() + 1;
    const uint64_t gameplay = g_gameplayPresent.load();
    if (gameplay == 0 || present - gameplay > kGameplayRecentPresents)
        return;

    const auto desc = Desc(picture);
    if (!g_createTried)
    {
        if (IsStableSetup(desc.Width, desc.Height))
        {
            g_context = ctx;
            CreateDlss(ctx, desc.Width, desc.Height);
        }
        return;
    }
    if (g_handle == nullptr)
        return;
    if (ctx != g_context || desc.Width != g_width || desc.Height != g_height)
    {
        static bool logged = false;
        if (!logged)
        {
            logged = true;
            LOG_WARN("Eye image NR: final picture now {}x{} on context {:X}, set up for {}x{} on {:X}; restart to match",
                     desc.Width, desc.Height, (size_t) ctx, g_width, g_height, (size_t) g_context);
        }
        return;
    }

    ID3D11Texture2D* depth = nullptr;
    {
        std::lock_guard lock(g_mutex);
        depth = g_sceneDepth;
        if (depth != nullptr)
            depth->AddRef();
    }
    if (depth == nullptr)
        return;
    const auto depthDesc = Desc(depth);
    if (depthDesc.Width != g_width || depthDesc.Height != g_height)
    {
        static bool logged = false;
        if (!logged)
        {
            logged = true;
            LOG_WARN("Eye image NR: scene depth is {}x{}, not {}x{}; skipping", depthDesc.Width, depthDesc.Height,
                     g_width, g_height);
        }
        depth->Release();
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
        ok = Evaluate(ctx, depth);
        if (ok)
            ctx->CopyResource(picture, g_output);
    }
    QueryPerformanceCounter(&end);
    depth->Release();
    if (!ok)
        return;

    const uint64_t micros = (uint64_t) ((end.QuadPart - start.QuadPart) * 1000000 / frequency.QuadPart);
    g_evalMicros += micros;
    if (micros > g_evalMaxMicros.load())
        g_evalMaxMicros = micros;
    ++g_evaluations;
    static bool logged = false;
    if (!logged)
    {
        logged = true;
        LOG_INFO("Eye image NR: first evaluation on the final picture {}x{} fmt {}", desc.Width, desc.Height,
                 (UINT) desc.Format);
    }
}
} // namespace

bool EyeImageNr_Dx11::Enabled()
{
    static int cached = -1;
    if (cached < 0)
        cached = Config::Instance()->EyeImageNr.value_or_default() ? 1 : 0;
    return cached == 1;
}

namespace
{
// Runs NR on the final picture drawn on the immediate context, if one was drawn since it was bound.
void FlushImmediateFinal()
{
    if (g_immFinal == nullptr)
        return;
    ID3D11Texture2D* picture = g_immFinal;
    ID3D11DeviceContext* ctx = g_immCtx;
    const bool drew = g_immDrew;
    g_immFinal = nullptr;
    g_immCtx = nullptr;
    g_immDrew = false;
    if (drew)
    {
        {
            std::lock_guard lock(g_mutex);
            ReleaseDeferred();
        }
        static bool logged = false;
        if (!logged)
        {
            logged = true;
            const auto desc = Desc(picture);
            LOG_INFO("Eye image NR: final picture drawn on the immediate context {:X}, {}x{} fmt {}", (size_t) ctx,
                     desc.Width, desc.Height, (UINT) desc.Format);
        }
        ProcessFinal(ctx, picture);
    }
    picture->Release();
}
} // namespace

void EyeImageNr_Dx11::OnImmediateBind(ID3D11DeviceContext* ctx, UINT numRtvs, ID3D11RenderTargetView* const* rtvs,
                                    ID3D11DepthStencilView* dsv)
{
    if (g_inEvaluate > 0 || ctx == nullptr)
        return;
    // The draw into the final picture has been recorded: process it before the game's next bind goes through.
    if (g_immFinal != nullptr && ctx == g_immCtx)
        FlushImmediateFinal();

    ID3D11Texture2D* color =
        (numRtvs > 0 && numRtvs <= D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT && rtvs != nullptr) ? TextureFromView(rtvs[0])
                                                                                              : nullptr;
    ID3D11Texture2D* depth = TextureFromView(dsv);
    if (g_immFinal == nullptr && IsFinalBind(numRtvs, color, depth))
    {
        color->AddRef();
        g_immFinal = color;
        g_immCtx = ctx;
        g_immDrew = false;
    }
    if (color != nullptr)
        color->Release();
    if (depth != nullptr)
        depth->Release();
}

void EyeImageNr_Dx11::OnImmediateDraw(ID3D11DeviceContext* ctx)
{
    if (g_inEvaluate == 0 && ctx == g_immCtx && g_immFinal != nullptr)
        g_immDrew = true;
}

void EyeImageNr_Dx11::OnDeferredBind(ID3D11DeviceContext* ctx, UINT numRtvs, ID3D11RenderTargetView* const* rtvs,
                                   ID3D11DepthStencilView* dsv)
{
    if (g_inEvaluate > 0 || ctx == nullptr)
        return;

    ID3D11Texture2D* color =
        (numRtvs > 0 && numRtvs <= D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT && rtvs != nullptr) ? TextureFromView(rtvs[0])
                                                                                              : nullptr;
    ID3D11Texture2D* depth = TextureFromView(dsv);

    if (IsGameplayBind(numRtvs, rtvs, color, depth))
    {
        if (g_gameplayPresent.exchange(g_present.load() + 1) == 0)
        {
            const auto desc = Desc(color);
            LOG_INFO("Eye image NR: gameplay seen, G-buffer {}x{}", desc.Width, desc.Height);
        }
        std::lock_guard lock(g_mutex);
        RetainDeferred(&g_sceneDepth, depth);
    }

    if (IsFinalBind(numRtvs, color, depth))
    {
        std::lock_guard lock(g_mutex);
        auto& recording = g_recording[ctx];
        if (recording.target != color)
        {
            color->AddRef();
            if (recording.target != nullptr)
                recording.target->Release();
            recording.target = color;
        }
        t_finalCtx = ctx;
    }
    else if (t_finalCtx == ctx)
    {
        t_finalCtx = nullptr;
    }

    if (color != nullptr)
        color->Release();
    if (depth != nullptr)
        depth->Release();
}

void EyeImageNr_Dx11::OnDeferredDraw(ID3D11DeviceContext* ctx)
{
    if (t_finalCtx != ctx || ctx == nullptr)
        return;
    t_finalCtx = nullptr;
    std::lock_guard lock(g_mutex);
    if (auto it = g_recording.find(ctx); it != g_recording.end())
        it->second.drew = true;
}

void EyeImageNr_Dx11::OnFinishCommandList(ID3D11DeviceContext* ctx, ID3D11CommandList* list)
{
    if (t_finalCtx == ctx)
        t_finalCtx = nullptr;
    if (list == nullptr)
        return;

    std::lock_guard lock(g_mutex);
    // A list address can be reused once the game releases the old list.
    if (auto it = g_marked.find(list); it != g_marked.end())
    {
        it->second->Release();
        g_marked.erase(it);
    }
    auto it = g_recording.find(ctx);
    if (it == g_recording.end())
        return;
    if (it->second.drew && it->second.target != nullptr)
    {
        static bool logged = false;
        if (!logged)
        {
            logged = true;
            const auto desc = Desc(it->second.target);
            LOG_INFO("Eye image NR: final picture list seen, {}x{} fmt {}", desc.Width, desc.Height, (UINT) desc.Format);
        }
        g_marked[list] = it->second.target;
    }
    else if (it->second.target != nullptr)
    {
        it->second.target->Release();
    }
    g_recording.erase(it);
}

void EyeImageNr_Dx11::AfterExecuteCommandList(ID3D11DeviceContext* ctx, ID3D11CommandList* list)
{
    if (g_inEvaluate > 0 || ctx == nullptr || list == nullptr)
        return;

    ID3D11Texture2D* picture = nullptr;
    {
        std::lock_guard lock(g_mutex);
        ReleaseDeferred();
        auto it = g_marked.find(list);
        if (it == g_marked.end())
            return;
        picture = it->second;
        g_marked.erase(it);
    }
    ProcessFinal(ctx, picture);
    picture->Release();
}

void EyeImageNr_Dx11::OnPresent()
{
    FlushImmediateFinal();
    const uint64_t present = ++g_present;

    // One eye per Present, so one evaluation per Present is the expected ratio.
    static uint64_t windowStart = 0;
    constexpr uint64_t kWindow = 900;
    if (g_evaluations.load() == 0)
    {
        windowStart = present;
        return;
    }
    if (present - windowStart >= kWindow)
    {
        const uint64_t evaluations = g_evaluations.exchange(0);
        const uint64_t micros = g_evalMicros.exchange(0);
        const uint64_t maxMicros = g_evalMaxMicros.exchange(0);
        const double avgMs = evaluations > 0 ? (double) micros / (double) evaluations / 1000.0 : 0.0;
        LOG_INFO("Eye image NR: {} evaluations over the last {} presents, CPU {:.2f} ms average, {:.2f} ms worst",
                 evaluations, present - windowStart, avgMs, (double) maxMicros / 1000.0);
        windowStart = present;
    }
}
