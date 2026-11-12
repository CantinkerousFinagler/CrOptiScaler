#include "pch.h"
#include "Fc5Dlss_Dx11.h"

#include "Config.h"
#include "Logger.h"
#include "NVNGX_Parameter.h"
#include "resource.h"
#include "State.h"
#include "Util.h"
#include "inputs/Fc5Taa_Dx11.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <vector>

namespace
{
// Everything below except the atomics is touched only from the game's render thread, inside the bind hooks
// on the device's immediate context (FC5 records its whole frame there).
ID3D11DeviceContext* g_context = nullptr;

// Set while this backend evaluates or copies. DLSS runs NR synchronously, and the D3D11 bridge unbinds and
// restores render targets through the same hooked OMSetRenderTargets; FC6 crashed on exactly this kind of
// re-entry. Per thread.
thread_local int g_inEvaluate = 0;

struct EvaluateScope
{
    EvaluateScope() { ++g_inEvaluate; }
    ~EvaluateScope() { --g_inEvaluate; }
};

std::atomic<uint64_t> g_present { 0 };
std::atomic<uint64_t> g_evaluations { 0 };

// The previous bind on the immediate context: the upscale draw's source is the target bound just before
// the eye target, with no depth either time.
ID3D11Texture2D* g_prevColor = nullptr;
bool g_prevHadDepth = false;

// Gameplay signature: the G-buffer MRT bind (four targets plus depth). Menus and loading screens never
// draw it, and its size is the render size. Its depth is the scene depth DLSS reads.
uint64_t g_gameplayPresent = 0;
UINT g_renderW = 0;
UINT g_renderH = 0;
ID3D11Texture2D* g_sceneDepth = nullptr;

// Reloads and settings changes reallocate the game's targets; references are dropped a while later rather
// than immediately, so nothing we hold is released under a frame that may still use it.
struct DeferredRelease
{
    ID3D11Texture2D* texture;
    uint64_t present;
};
std::vector<DeferredRelease> g_deferred;
constexpr uint64_t kReleaseAfterPresents = 600;

// Upscale draw seen at this bind: evaluate after the game's bind goes through.
ID3D11Texture2D* g_evalSrc = nullptr;
ID3D11Texture2D* g_evalDest = nullptr;
// Evaluated: copy the output over the game's draw at the next bind.
ID3D11Texture2D* g_copyDest = nullptr;
// Render-size NR (below 100%, unless [Inputs] DuniaTaaQuality=1): the TAA output NR was written into, restored from
// g_backup at the next bind.
bool g_renderSizeNr = false;
ID3D11Texture2D* g_backup = nullptr;
ID3D11Texture2D* g_restoreSrc = nullptr;

NVSDK_NGX_Parameter* g_params = nullptr;
NVSDK_NGX_Handle* g_handle = nullptr;
ID3D11Texture2D* g_output = nullptr;
ID3D11Texture2D* g_motion = nullptr;
bool g_createTried = false;

std::atomic<int> g_status { (int) Fc5Dlss_Dx11::Status::WaitingForGameplay };
std::atomic<unsigned> g_inW { 0 };
std::atomic<unsigned> g_inH { 0 };
std::atomic<unsigned> g_outW { 0 };
std::atomic<unsigned> g_outH { 0 };

constexpr uint64_t kGameplayRecentPresents = 30;

// Sizes are decided once, so only from a settled game. Entering VR, FC5 keeps drawing its flat 1920x1080
// frames for a few seconds beside the first eye-sized ones, which read as a render size below the eye.
constexpr ULONGLONG kStableSetupMs = 5000;
constexpr ULONGLONG kStableGapMs = 1000;
UINT g_candidate[4] = {};
ULONGLONG g_candidateSince = 0;
ULONGLONG g_candidateLast = 0;

// True once the same render and eye sizes have been seen on every upscale draw for kStableSetupMs.
bool IsStableSetup(UINT inW, UINT inH, UINT outW, UINT outH)
{
    const ULONGLONG now = GetTickCount64();
    const bool same = g_candidate[0] == inW && g_candidate[1] == inH && g_candidate[2] == outW &&
                      g_candidate[3] == outH;
    if (!same || now - g_candidateLast > kStableGapMs)
    {
        static int logged = 0;
        if (!same && g_candidateSince != 0 && logged++ < 10)
            LOG_INFO("FC5 DLSS: upscale draw now {}x{} -> {}x{} (was {}x{} -> {}x{}), waiting for it to settle", inW,
                     inH, outW, outH, g_candidate[0], g_candidate[1], g_candidate[2], g_candidate[3]);
        g_candidate[0] = inW;
        g_candidate[1] = inH;
        g_candidate[2] = outW;
        g_candidate[3] = outH;
        g_candidateSince = now;
    }
    g_candidateLast = now;
    return now - g_candidateSince >= kStableSetupMs;
}

void SetStatus(Fc5Dlss_Dx11::Status status) { g_status = (int) status; }

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
        g_deferred.push_back({ *slot, g_present.load() });
    *slot = value;
}

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

// Before VR starts, FC5 renders its first frames flat at the monitor size (1920x1080 here) with the whole
// pipeline, G-buffer included. Setting DLSS up then left it at the wrong size for the eyes. Same limits as
// the FC6 backend: a target passes on either axis ([Inputs] DuniaTaaMinEvalWidth / DuniaTaaMinEvalHeight).
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
    if (numRtvs < 4 || rtvs == nullptr || color == nullptr || depth == nullptr)
        return false;
    for (UINT i = 0; i < 4; ++i)
    {
        if (rtvs[i] == nullptr)
            return false;
    }
    const auto colorDesc = Desc(color);
    const auto depthDesc = Desc(depth);
    return colorDesc.Width == depthDesc.Width && colorDesc.Height == depthDesc.Height && colorDesc.Width >= 512 &&
           colorDesc.Height >= 512 && colorDesc.SampleDesc.Count == 1;
}

// FC5 creates several D3D11 devices, and the hooks patch d3d11.dll code shared by all of them, so binds
// arrive from every immediate context. Follow the one that draws gameplay, adopted on its first G-buffer
// bind. Once DLSS exists on a context it stays there: nothing is torn down or rebuilt live.
bool AdoptContext(ID3D11DeviceContext* ctx, UINT numRtvs, ID3D11RenderTargetView* const* rtvs,
                  ID3D11DepthStencilView* dsv)
{
    if (ctx->GetType() != D3D11_DEVICE_CONTEXT_IMMEDIATE)
        return false;

    ID3D11Texture2D* color =
        (numRtvs > 0 && numRtvs <= D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT && rtvs != nullptr) ? TextureFromView(rtvs[0])
                                                                                              : nullptr;
    ID3D11Texture2D* depth = TextureFromView(dsv);
    const bool gameplay = IsGameplayBind(numRtvs, rtvs, color, depth);
    if (color != nullptr)
        color->Release();
    if (depth != nullptr)
        depth->Release();
    if (!gameplay)
        return false;

    if (g_createTried)
    {
        static bool logged = false;
        if (!logged)
        {
            logged = true;
            LOG_WARN("FC5 DLSS: gameplay is now drawn on context {:X}, but DLSS was set up on {:X}; restart the game",
                     (size_t) ctx, (size_t) g_context);
        }
        return false;
    }

    LOG_INFO("FC5 DLSS: following context {:X}{}", (size_t) ctx, g_context != nullptr ? " (switched)" : "");
    ctx->AddRef();
    if (g_context != nullptr)
        g_context->Release();
    g_context = ctx;
    Take(&g_prevColor);
    g_prevHadDepth = false;
    Take(&g_evalSrc);
    Take(&g_evalDest);
    Take(&g_copyDest);
    return true;
}

bool CreateTextures(ID3D11Device* device, UINT inW, UINT inH, UINT outW, UINT outH)
{
    D3D11_TEXTURE2D_DESC desc {};
    desc.Width = outW;
    desc.Height = outH;
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.SampleDesc.Count = 1;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;
    if (FAILED(device->CreateTexture2D(&desc, nullptr, &g_output)))
    {
        LOG_ERROR("FC5 DLSS: output texture create failed");
        return false;
    }

    // FC5 has no motion vectors we use yet: zero vectors and a history reset every frame, as FC6.
    desc.Width = inW;
    desc.Height = inH;
    desc.Format = DXGI_FORMAT_R16G16_FLOAT;
    desc.BindFlags = D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;
    if (FAILED(device->CreateTexture2D(&desc, nullptr, &g_motion)))
    {
        LOG_ERROR("FC5 DLSS: motion texture create failed");
        return false;
    }
    ID3D11UnorderedAccessView* uav = nullptr;
    if (SUCCEEDED(device->CreateUnorderedAccessView(g_motion, nullptr, &uav)) && uav != nullptr)
    {
        const float zero[4] = {};
        g_context->ClearUnorderedAccessViewFloat(uav, zero);
        uav->Release();
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
        LOG_ERROR("FC5 DLSS: NGX init failed: {:X}", (UINT) result);
        return false;
    }
    return true;
}

// Once per session: creation-time settings are startup-only, and a live rebuild is not attempted.
void CreateDlss(ID3D11DeviceContext* ctx, UINT inW, UINT inH, UINT outW, UINT outH, DXGI_FORMAT srcFormat)
{
    g_createTried = true;
    g_inW = inW;
    g_inH = inH;
    g_outW = outW;
    g_outH = outH;

    // Below 100% Resolution Scale the game upscales in that same draw. By default the game's upscale stays: DLAA at
    // the render size carries NR, written into the TAA output the upscale draw reads. Without motion vectors DLSS
    // upscaling looks no better than the game's own and costs more (eye-size work and copies; measured lower fps),
    // so it is opt-in with [Inputs] DuniaTaaQuality=1. At 100% the draw is a 1:1 copy and DLSS
    // runs as DLAA either way.
    const bool scaled = inW != outW || inH != outH;
    const bool renderSizeNr = scaled && Config::Instance()->DuniaTaaQuality.value_or_default() != 1;
    const bool upscale = scaled && !renderSizeNr;
    const UINT featureOutW = upscale ? outW : inW;
    const UINT featureOutH = upscale ? outH : inH;
    const char* mode = upscale ? "upscaling" : (renderSizeNr ? "DLAA at render size (game upscales)" : "DLAA");

    ID3D11Device* device = nullptr;
    ctx->GetDevice(&device);
    if (device == nullptr)
    {
        SetStatus(Fc5Dlss_Dx11::Status::CreateFailed);
        return;
    }

    EvaluateScope scope;
    bool ok = CreateTextures(device, inW, inH, featureOutW, featureOutH) && InitNgx(device);
    if (ok && renderSizeNr)
    {
        // Holds the game's TAA output while NR's result stands in for it during the upscale draw. The TAA output
        // is also TAA history for later frames, so it is put back straight after that draw.
        D3D11_TEXTURE2D_DESC desc {};
        desc.Width = inW;
        desc.Height = inH;
        desc.MipLevels = 1;
        desc.ArraySize = 1;
        desc.Format = srcFormat;
        desc.SampleDesc.Count = 1;
        desc.Usage = D3D11_USAGE_DEFAULT;
        if (FAILED(device->CreateTexture2D(&desc, nullptr, &g_backup)))
        {
            LOG_ERROR("FC5 DLSS: TAA backup texture create failed (fmt {})", (UINT) srcFormat);
            ok = false;
        }
    }
    device->Release();

    if (ok && g_params == nullptr && NVSDK_NGX_D3D11_GetCapabilityParameters(&g_params) != NVSDK_NGX_Result_Success)
    {
        LOG_ERROR("FC5 DLSS: GetCapabilityParameters failed");
        ok = false;
    }

    if (ok)
    {
        // The picture is tonemapped 8-bit: no IsHDR, so NR also leaves its colour transform off.
        UINT flags = NVSDK_NGX_DLSS_Feature_Flags_AutoExposure;
        if (Config::Instance()->DuniaTaaDepthInverted.value_or_default())
            flags |= NVSDK_NGX_DLSS_Feature_Flags_DepthInverted;
        g_params->Set(NVSDK_NGX_Parameter_DLSS_Feature_Create_Flags, flags);
        g_params->Set(NVSDK_NGX_Parameter_Width, inW);
        g_params->Set(NVSDK_NGX_Parameter_Height, inH);
        g_params->Set(NVSDK_NGX_Parameter_OutWidth, featureOutW);
        g_params->Set(NVSDK_NGX_Parameter_OutHeight, featureOutH);
        g_params->Set(NVSDK_NGX_Parameter_PerfQualityValue,
                      upscale ? NVSDK_NGX_PerfQuality_Value_MaxQuality : NVSDK_NGX_PerfQuality_Value_DLAA);

        const auto result = NVSDK_NGX_D3D11_CreateFeature(ctx, NVSDK_NGX_Feature_SuperSampling, g_params, &g_handle);
        if (result != NVSDK_NGX_Result_Success)
        {
            LOG_ERROR("FC5 DLSS: CreateFeature {} failed: {:X}", mode, (UINT) result);
            g_handle = nullptr;
            ok = false;
        }
    }

    if (!ok)
    {
        SetStatus(Fc5Dlss_Dx11::Status::CreateFailed);
        return;
    }

    g_renderSizeNr = renderSizeNr;
    SetStatus(upscale ? Fc5Dlss_Dx11::Status::Upscaling
                      : (renderSizeNr ? Fc5Dlss_Dx11::Status::RenderSizeNr : Fc5Dlss_Dx11::Status::Dlaa));
    LOG_INFO("FC5 DLSS: created {} {}x{} -> {}x{} for a {}x{} eye (depth {})", mode, inW, inH, featureOutW,
             featureOutH, outW, outH,
             Config::Instance()->DuniaTaaDepthInverted.value_or_default() ? "inverted" : "normal");
}

bool Evaluate(ID3D11DeviceContext* ctx, ID3D11Texture2D* src)
{
    if (g_handle == nullptr || g_params == nullptr || g_output == nullptr || g_motion == nullptr ||
        g_sceneDepth == nullptr)
        return false;

    const UINT inW = g_inW.load();
    const UINT inH = g_inH.load();
    const auto depthDesc = Desc(g_sceneDepth);
    if (depthDesc.Width != inW || depthDesc.Height != inH)
    {
        static bool logged = false;
        if (!logged)
        {
            logged = true;
            LOG_WARN("FC5 DLSS: scene depth is {}x{}, not the {}x{} input; skipping", depthDesc.Width,
                     depthDesc.Height, inW, inH);
        }
        return false;
    }

    EvaluateScope scope;
    g_params->Set(NVSDK_NGX_Parameter_Jitter_Offset_X, 0.0f);
    g_params->Set(NVSDK_NGX_Parameter_Jitter_Offset_Y, 0.0f);
    g_params->Set(NVSDK_NGX_Parameter_MV_Scale_X, 1.0f);
    g_params->Set(NVSDK_NGX_Parameter_MV_Scale_Y, 1.0f);
    g_params->Set(NVSDK_NGX_Parameter_DLSS_Exposure_Scale, 1.0f);
    g_params->Set(NVSDK_NGX_Parameter_Reset, 1);
    g_params->Set(NVSDK_NGX_Parameter_Width, inW);
    g_params->Set(NVSDK_NGX_Parameter_Height, inH);
    g_params->Set(NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Width, inW);
    g_params->Set(NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Height, inH);
    g_params->Set(NVSDK_NGX_Parameter_Depth, g_sceneDepth);
    g_params->Set(NVSDK_NGX_Parameter_Color, src);
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
            LOG_ERROR("FC5 DLSS: EvaluateFeature failed: {:X}", (UINT) result);
        }
        return false;
    }
    return true;
}
} // namespace

bool Fc5Dlss_Dx11::Enabled()
{
    static int cached = -1;
    if (cached < 0)
        // Off when the per-eye DLAA at the TAA draw (Fc5Taa_Dx11) is on: F6 is then the only switch, and nothing is
        // set up on a timer.
        cached = (_stricmp(State::Instance().gameExe.c_str(), "FarCry5.exe") == 0 &&
                  Config::Instance()->DuniaTaaHost.value_or_default() && !Fc5Taa_Dx11::Enabled())
                     ? 1
                     : 0;
    return cached == 1;
}

void Fc5Dlss_Dx11::BeforeBind(ID3D11DeviceContext* ctx, UINT numRtvs, ID3D11RenderTargetView* const* rtvs,
                              ID3D11DepthStencilView* dsv)
{
    if (g_inEvaluate > 0 || ctx == nullptr)
        return;
    if (ctx != g_context && !AdoptContext(ctx, numRtvs, rtvs, dsv))
        return;

    ReleaseDeferred();

    // Straight after the upscale draw, while the eye target it drew into is still the one bound: put DLSS's
    // result over it before the eye-resolution passes draw on top.
    if (g_copyDest != nullptr)
    {
        if (g_prevColor == g_copyDest && g_output != nullptr)
        {
            EvaluateScope scope;
            ctx->CopyResource(g_copyDest, g_output);
            static bool logged = false;
            if (!logged)
            {
                logged = true;
                LOG_INFO("FC5 DLSS: output copied over the game's upscale draw");
            }
        }
        Take(&g_copyDest);
    }

    // Render-size NR: the upscale draw has read NR's result from the TAA output; give the game its own back.
    if (g_restoreSrc != nullptr)
    {
        if (g_backup != nullptr)
        {
            EvaluateScope scope;
            ctx->CopyResource(g_restoreSrc, g_backup);
            static bool logged = false;
            if (!logged)
            {
                logged = true;
                LOG_INFO("FC5 DLSS: game upscaled NR's result; TAA output restored");
            }
        }
        Take(&g_restoreSrc);
    }

    ID3D11Texture2D* color =
        (numRtvs > 0 && numRtvs <= D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT && rtvs != nullptr) ? TextureFromView(rtvs[0])
                                                                                              : nullptr;
    ID3D11Texture2D* depth = TextureFromView(dsv);
    const uint64_t now = g_present.load() + 1;

    if (IsGameplayBind(numRtvs, rtvs, color, depth))
    {
        const auto desc = Desc(color);
        if (g_gameplayPresent == 0)
            LOG_INFO("FC5 DLSS: gameplay seen, render size {}x{}", desc.Width, desc.Height);
        g_gameplayPresent = now;
        g_renderW = desc.Width;
        g_renderH = desc.Height;
        RetainDeferred(&g_sceneDepth, depth);
    }

    // The upscale draw: the eye target bound without depth, straight after a render-size 8-bit target that
    // was also bound without depth (the TAA output).
    if (color != nullptr && depth == nullptr && g_prevColor != nullptr && !g_prevHadDepth && g_prevColor != color &&
        g_gameplayPresent != 0 && now - g_gameplayPresent <= kGameplayRecentPresents)
    {
        const auto srcDesc = Desc(g_prevColor);
        const auto destDesc = Desc(color);
        // The eye target is only ever drawn into, never sampled, while every post-chain target is (tonemap ->
        // prep -> TAA, all render size and all bound without depth). Without this, each of those links matched
        // too: at 100% DLAA+NR ran several times per eye, overwriting the game's own passes.
        if ((destDesc.BindFlags & D3D11_BIND_SHADER_RESOURCE) == 0 && IsEyeSize(destDesc.Width, destDesc.Height) &&
            IsRgba8(srcDesc.Format) && IsRgba8(destDesc.Format) &&
            srcDesc.Width == g_renderW &&
            srcDesc.Height == g_renderH && destDesc.Width >= srcDesc.Width && destDesc.Height >= srcDesc.Height &&
            srcDesc.SampleDesc.Count == 1 && destDesc.SampleDesc.Count == 1)
        {
            if (!g_createTried)
            {
                if (IsStableSetup(srcDesc.Width, srcDesc.Height, destDesc.Width, destDesc.Height))
                {
                    Retain(&g_evalSrc, g_prevColor);
                    Retain(&g_evalDest, color);
                }
            }
            else if (srcDesc.Width == g_inW.load() && srcDesc.Height == g_inH.load() &&
                     destDesc.Width == g_outW.load() && destDesc.Height == g_outH.load())
            {
                IsStableSetup(srcDesc.Width, srcDesc.Height, destDesc.Width, destDesc.Height);
                // With NR off there is nothing to gain: no motion vectors, so DLSS alone matches the game's own
                // draw and would only add its cost. Leave the game's draw untouched.
                if (g_handle != nullptr && Config::Instance()->DlssNrEnabled.value_or_default())
                {
                    Retain(&g_evalSrc, g_prevColor);
                    Retain(&g_evalDest, color);
                }
            }
            else if (IsStableSetup(srcDesc.Width, srcDesc.Height, destDesc.Width, destDesc.Height) &&
                     g_status.load() != (int) Status::ScaleChanged)
            {
                SetStatus(Status::ScaleChanged);
                LOG_WARN("FC5 DLSS: game now renders {}x{} -> {}x{}, set up for {}x{} -> {}x{}; restart to match",
                         srcDesc.Width, srcDesc.Height, destDesc.Width, destDesc.Height, g_inW.load(), g_inH.load(),
                         g_outW.load(), g_outH.load());
            }
        }
    }

    Retain(&g_prevColor, color);
    g_prevHadDepth = depth != nullptr;
    if (color != nullptr)
        color->Release();
    if (depth != nullptr)
        depth->Release();
}

namespace
{
// CPU time spent inside the evaluate call on the render thread: the bridge's copies to D3D12, DLSS and NR
// recording, the waits for D3D12 and the copy back. Time here holds up the game's own frame.
std::atomic<uint64_t> g_evalMicros { 0 };
std::atomic<uint64_t> g_evalMaxMicros { 0 };

bool TimedEvaluate(ID3D11DeviceContext* ctx, ID3D11Texture2D* src)
{
    LARGE_INTEGER frequency {};
    LARGE_INTEGER start {};
    LARGE_INTEGER end {};
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&start);
    const bool ok = Evaluate(ctx, src);
    QueryPerformanceCounter(&end);
    const uint64_t micros = (uint64_t) ((end.QuadPart - start.QuadPart) * 1000000 / frequency.QuadPart);
    if (ok)
    {
        g_evalMicros += micros;
        if (micros > g_evalMaxMicros.load())
            g_evalMaxMicros = micros;
    }
    return ok;
}
} // namespace

void Fc5Dlss_Dx11::AfterBind(ID3D11DeviceContext* ctx)
{
    if (g_inEvaluate > 0 || ctx == nullptr || ctx != g_context || g_evalSrc == nullptr)
        return;

    ID3D11Texture2D* src = g_evalSrc;
    ID3D11Texture2D* dest = g_evalDest;
    g_evalSrc = nullptr;
    g_evalDest = nullptr;

    if (!g_createTried)
    {
        const auto srcDesc = Desc(src);
        const auto destDesc = Desc(dest);
        CreateDlss(ctx, srcDesc.Width, srcDesc.Height, destDesc.Width, destDesc.Height, srcDesc.Format);
    }
    else if (TimedEvaluate(ctx, src))
    {
        static bool logged = false;
        if (!logged)
        {
            logged = true;
            const auto srcDesc = Desc(src);
            LOG_INFO("FC5 DLSS: evaluating on the game's upscale draw, source {}x{} fmt {}", srcDesc.Width,
                     srcDesc.Height, (UINT) srcDesc.Format);
        }
        ++g_evaluations;
        if (g_renderSizeNr)
        {
            // Before the upscale draw runs: it samples the TAA output, so NR's render-size result goes there.
            EvaluateScope scope;
            ctx->CopyResource(g_backup, src);
            ctx->CopyResource(src, g_output);
            Retain(&g_restoreSrc, src);
        }
        else
        {
            Retain(&g_copyDest, dest);
        }
    }

    src->Release();
    dest->Release();
}

void Fc5Dlss_Dx11::OnPresent()
{
    const uint64_t present = ++g_present;

    // RealVR presents once per eye, so one evaluation per Present is the expected ratio. Logged for the
    // first few windows after DLSS starts, then now and then.
    static uint64_t windowStart = 0;
    static int windows = 0;
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
        LOG_INFO("FC5 DLSS: {} evaluations over the last {} presents, CPU {:.2f} ms average, {:.2f} ms worst",
                 evaluations, present - windowStart, avgMs, (double) maxMicros / 1000.0);
        ++windows;
        windowStart = present;
    }
}

Fc5Dlss_Dx11::Info Fc5Dlss_Dx11::GetInfo()
{
    Info info;
    info.status = (Status) g_status.load();
    info.inW = g_inW.load();
    info.inH = g_inH.load();
    info.outW = g_outW.load();
    info.outH = g_outH.load();
    return info;
}
