#include "pch.h"
#include "TaaDlaa_Dx11.h"

#include "Config.h"
#include "Logger.h"
#include "NVNGX_Parameter.h"
#include "resource.h"
#include "State.h"
#include "Util.h"
#include "inputs/FinalImageNr_Dx11.h"
#include "inputs/TaaMotion_Dx11.h"
#include <dlssnr/DlssNr.h>

#include <d3d11_1.h>
#include <d3dcompiler.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <format>
#include <string>
#include <cstring>

namespace
{
// Re-implements the game's finish pass after its TAA resolve (Days Gone, same in flat and VR):
// grain, square, vignette, grain, HUD blend, sRGB. Its constants are the game's cb0, still bound. The colour
// channels go through the game's float4 math with red duplicated into w, which also gives the alpha it writes.
// DXBC exp/log are base 2.
constexpr char kFinishShader[] = R"(
Texture2D<float4> Resolved : register(t0);
Texture2D<float4> Hud : register(t1);
RWTexture2D<float4> Output : register(u0);

cbuffer PassConstants : register(b0)
{
    float4 C[10];
};

// Our sharpening (overlay "Sharpening", 0 = off): an unsharp mask at 1 or 2 px on the DLSS output, in the colour
// input's encoding, before the game's post. The game's own TAA and upscale draw sharpen and DLSS does not; OptiScaler's
// RCAS was too weak to see in VR (it works per pixel and caps itself, and the headset resamples the eye image).
cbuffer SharpenConstants : register(b1)
{
    float SharpenStrength;
    float SharpenRadius;
    float2 SharpenPad;
};

float3 Sharpened(int2 p, int2 last, float3 c0)
{
    if (SharpenStrength <= 0.0)
        return c0;
    const int r = (int) SharpenRadius;
    const float3 cross = Resolved[clamp(p + int2(r, 0), 0, last)].rgb + Resolved[clamp(p - int2(r, 0), 0, last)].rgb +
                         Resolved[clamp(p + int2(0, r), 0, last)].rgb + Resolved[clamp(p - int2(0, r), 0, last)].rgb;
    const float3 diag = Resolved[clamp(p + int2(r, r), 0, last)].rgb + Resolved[clamp(p - int2(r, r), 0, last)].rgb +
                        Resolved[clamp(p + int2(r, -r), 0, last)].rgb + Resolved[clamp(p + int2(-r, r), 0, last)].rgb;
    const float3 blur = (4.0 * c0 + 2.0 * cross + diag) / 16.0;
    return max(c0 + SharpenStrength * (c0 - blur), 0.0);
}

float Grain(float2 pixel, float2 invSize)
{
    const float2 uv = pixel * invSize + C[0].xy;
    return frac(sin(uv.x + uv.y * 543.309998) * 493013.0) * C[1].x + C[1].y;
}

[numthreads(8, 8, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    uint w, h;
    Output.GetDimensions(w, h);
    if (id.x >= w || id.y >= h)
        return;
    const float2 pixel = float2(id.xy) + 0.5;
    const float3 rgb = Sharpened(int2(id.xy), int2(w - 1, h - 1), Resolved[id.xy].rgb);
    float4 c = float4(rgb, rgb.r) * Grain(pixel, C[5].zw);

    const float2 d = pixel * C[6].zw - 0.5;
    const float v = saturate(exp2(-(0.5 - dot(d, d)) * C[3].x) * C[3].y);
    c = c * c;
    c = c * (C[2].w - v * C[2].w) + C[2].xyzx;
    c = C[4].xyzx * (v * v * C[2].w) + c;
    c *= Grain(pixel, C[6].zw);

    const float4 hud = Hud[id.xy];
    c = c * (1.0 - hud.a) + hud.xyzx;
    const float4 linearPart = c.wyzw * 12.92;
    const float4 curve = exp2(log2(max(c, 0.003131)) * 0.416667) * 1.055 - 0.055;
    Output[id.xy] = min(curve, linearPart);
}
)";

// Re-implements the game's finish pass of the upscale draw at a Render Scale below 100% (Days Gone), without
// its bilinear sample, adaptive sharpen and YCoCg decode (DLSS already gives the eye-size picture in the colour
// input's encoding): square, vignette, one grain, HUD blend, sRGB, alpha 0. Its constants are the draw's ps cb0:
// [6] render size / inverse, [7] output size / inverse. The draw's texture coordinate is the output UV.
constexpr char kUpscaleFinishShader[] = R"(
Texture2D<float4> Resolved : register(t0);
Texture2D<float4> Hud : register(t1);
RWTexture2D<float4> Output : register(u0);

cbuffer DrawConstants : register(b0)
{
    float4 C[8];
};

// Our sharpening (overlay "Sharpening", 0 = off): an unsharp mask at 1 or 2 px on the DLSS output, in the colour
// input's encoding, before the game's post. The game's own TAA and upscale draw sharpen and DLSS does not; OptiScaler's
// RCAS was too weak to see in VR (it works per pixel and caps itself, and the headset resamples the eye image).
cbuffer SharpenConstants : register(b1)
{
    float SharpenStrength;
    float SharpenRadius;
    float2 SharpenPad;
};

float3 Sharpened(int2 p, int2 last, float3 c0)
{
    if (SharpenStrength <= 0.0)
        return c0;
    const int r = (int) SharpenRadius;
    const float3 cross = Resolved[clamp(p + int2(r, 0), 0, last)].rgb + Resolved[clamp(p - int2(r, 0), 0, last)].rgb +
                         Resolved[clamp(p + int2(0, r), 0, last)].rgb + Resolved[clamp(p - int2(0, r), 0, last)].rgb;
    const float3 diag = Resolved[clamp(p + int2(r, r), 0, last)].rgb + Resolved[clamp(p - int2(r, r), 0, last)].rgb +
                        Resolved[clamp(p + int2(r, -r), 0, last)].rgb + Resolved[clamp(p + int2(-r, r), 0, last)].rgb;
    const float3 blur = (4.0 * c0 + 2.0 * cross + diag) / 16.0;
    return max(c0 + SharpenStrength * (c0 - blur), 0.0);
}

[numthreads(8, 8, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    uint w, h;
    Output.GetDimensions(w, h);
    if (id.x >= w || id.y >= h)
        return;
    const float2 uv = (float2(id.xy) + 0.5) * C[7].zw;
    float3 c = Sharpened(int2(id.xy), int2(w - 1, h - 1), Resolved[id.xy].rgb);

    const float2 d = uv - 0.5;
    const float v = saturate(exp2(-(0.5 - dot(d, d)) * C[3].x) * C[3].y);
    c = c * c;
    c = c * (C[2].w - v * C[2].w) + C[2].xyz;
    c = C[4].xyz * (v * v * C[2].w) + c;
    const float2 g = uv + C[0].xy;
    c *= frac(sin(g.x + g.y * 543.309998) * 493013.0) * C[1].x + C[1].y;

    const float4 hud = Hud[id.xy];
    c = c * (1.0 - hud.a) + hud.rgb;
    const float3 linearPart = c * 12.92;
    const float3 curve = exp2(log2(max(c, 0.003131)) * 0.416667) * 1.055 - 0.055;
    Output[id.xy] = float4(min(curve, linearPart), 0.0);
}
)";

// All on the thread driving the immediate context and calling Present; the mode and jitter choice cross threads.
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

// DLSS jitter from the view constants' (x, y): the sign of each axis. (+x, -y) = the negated sample
// offset, confirmed in VR.
constexpr float kJitterSigns[4][2] = { { 1, -1 }, { -1, 1 }, { 1, 1 }, { -1, -1 } };
std::atomic<int> g_jitterVariant { 0 };

// ---- Jitter: read back from cs cb3 a few Presents late, predicted forward ----

constexpr UINT kJitterOffset = 464; // index, sample count, x, y
constexpr int kJitterSlots = 4;

struct JitterSlot
{
    ID3D11Buffer* staging = nullptr;
    UINT bytes = 0;
    uint64_t present = 0;
    bool pending = false;
};
JitterSlot g_jitterSlots[kJitterSlots];

bool g_jitterKnown = false;
int g_baseIndex = 0; // 1..g_sampleCount
// The game's sample count (r.TemporalAASamples, 8 by default), from the same constants. Odd counts matter under
// alternate-eye VR: with an even count each eye only ever gets every other sample.
int g_sampleCount = 8;
uint64_t g_basePresent = 0;
uint64_t g_jitterReads = 0;
uint64_t g_jitterMismatches = 0;
uint64_t g_jitterInvalid = 0;
bool g_jitterLoggedFirst = false;

float Halton(int index, int base)
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

int PredictIndex(uint64_t present)
{
    return (int) (((uint64_t) (g_baseIndex - 1) + (present - g_basePresent)) % (uint64_t) g_sampleCount) + 1;
}

// ---- DLSS ----

struct Eye
{
    NVSDK_NGX_Handle* handle = nullptr;
    uint64_t lastPresent = UINT64_MAX;
};

// Two DLSS features (one per eye) and the output they write. One set for DLAA at the eye-size pass, one for
// upscaling at the render-resolution pass; each is created on first use, and rebuilt when the game's Render Scale or
// R.E.A.L. VR's eye size changes (RebuildWhenSettled).
struct Features
{
    const char* name;
    Eye eyes[2];
    bool createTried = false;
    bool ready = false;
    UINT inWidth = 0;
    UINT inHeight = 0;
    UINT outWidth = 0;
    UINT outHeight = 0;
    ID3D11Texture2D* output = nullptr;
    ID3D11ShaderResourceView* outputSrv = nullptr;
    // A size other than the set's, and for how many passes in a row it has been seen.
    UINT pending[4] = {};
    int pendingCount = 0;
    int rebuilds = 0;
};
Features g_dlaa { "DLAA" };
Features g_upscale { "upscaling" };

// Eyes alternate while the chain of evaluated Presents is unbroken.
int g_nextEye = 0;
uint64_t g_lastReplacedPresent = UINT64_MAX;
// The Present whose displayed image came from DLSS (DLAA replaced the pass, or the upscale draw was replaced).
uint64_t g_shownPresent = UINT64_MAX;

NVSDK_NGX_Parameter* g_params = nullptr;
ID3D11ComputeShader* g_finish = nullptr;
ID3D11ComputeShader* g_upscaleFinish = nullptr;
bool g_finishTried = false;

// The render-resolution pass of this Present (its output, compared by address only) and the game's upscale draw
// that reads it: the eye size it draws to, and whether this Present's upscaling output waits for it.
const void* g_renderTaaOutput = nullptr;
uint64_t g_renderTaaPresent = UINT64_MAX;
uint64_t g_upscaleDrawPresent = UINT64_MAX;
uint64_t g_upscalePendingPresent = UINT64_MAX;
// The last Present whose upscale draw the finish pass replaced: the game's render-resolution pass is skipped only
// while the previous Present's draw was replaced, so a draw that isn't never shows a stale output for long.
uint64_t g_drawReplacedPresent = UINT64_MAX;
UINT g_eyeWidth = 0;
UINT g_eyeHeight = 0;

// (UAVs on the game's upscale target are made per draw: CreateTargetUav.)

// A spare device context state: DLSS and the finish pass run inside it, the game's state is swapped back after.
ID3DDeviceContextState* g_spareState = nullptr;
bool g_spareTried = false;

// Stats, logged every kStatsWindow Presents while DLAA is on.
constexpr uint64_t kStatsWindow = 900;
uint64_t g_statsStart = 0;
uint64_t g_evaluations = 0;
uint64_t g_resets = 0;
uint64_t g_skips = 0;
uint64_t g_evalMicros = 0;
uint64_t g_evalMaxMicros = 0;
uint64_t g_drawsReplaced = 0;

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

// Why DLSS was not shown this Present (TaaDlaa_Dx11::MissSlot): the SkipOnce reasons by bit index (0-7), then these.
enum MissSlotId : int
{
    kMissNoSpareState = 8,     // no spare context state or finish shaders
    kMissNoContext1 = 9,       // no ID3D11DeviceContext1
    kMissNoFeatures = 10,      // the features were never made (creation failed, logged)
    kMissEvaluateFailed = 11,  // EvaluateFeature failed (logged per new result)
    kMissDrawNotReplaced = 12, // the upscale draw was found but could not be replaced
    kMissNothing = 13,         // nothing recorded (at render resolution: no upscale draw read the pass's output)
};
static_assert(kMissNothing + 1 == TaaDlaa_Dx11::kMissSlots);
// Set by the overlay, taken by the next replaced pass (TaaDlaa_Dx11::RequestRebuild).
std::atomic<bool> g_rebuildRequested { false };
// Sharpening after DLSS (overlay, live, not saved): passed as the evaluate's sharpness, which OptiScaler's DX12 path
// turns into its RCAS pass on the DLSS output (DLSS 2.5.1+ has none of its own). 0 = off, as before. The game's own TAA
// pass sharpens (r.TemporalAASharpness); replacing it loses that, and distance looked softer than with its TAA.
std::atomic<float> g_sharpness { 0.0f };
std::atomic<int> g_sharpenRadius { 1 };
float g_sharpnessLogged = -1.0f; // render thread only
int g_sharpenRadiusLogged = -1;  // render thread only
ID3D11Buffer* g_sharpenConstants = nullptr; // b1 of both finish shaders, ours

// F12 marker (the 70% breaks: whole-image blur/aliasing for a moment with DLSS shown on every frame and no
// reset counted): what DLSS was fed on the last evaluations and jitter reads, dumped when a break is marked.
struct EvalRecord
{
    uint64_t present;
    int eye;
    bool reset;
    int jitterIndex;
    float jitterX, jitterY;
    bool upscale;
    bool composed;
};
constexpr int kMarkRing = 256;
EvalRecord g_evalRing[kMarkRing] = {};
uint64_t g_evalRingCount = 0;
struct JitterRead
{
    uint64_t present;
    int index;     // the game's, 0 when the read was invalid
    int predicted; // ours for that Present, 0 when not known yet
    int source;    // 0 cs cb3, 1 the view constants
};
constexpr int kJitterRing = 128;
JitterRead g_jitterRing[kJitterRing] = {};
uint64_t g_jitterRingCount = 0;
std::atomic<bool> g_markRequested { false };

void RecordEval(uint64_t present, int eye, bool reset, int jitterIndex, float jx, float jy, bool upscale, bool composed)
{
    g_evalRing[g_evalRingCount++ % kMarkRing] = { present, eye, reset, jitterIndex, jx, jy, upscale, composed };
}
uint64_t g_missPresent = UINT64_MAX;
int g_missSlot = kMissNothing;

void NoteMiss(int slot)
{
    g_missPresent = TaaMotion_Dx11::PresentIndex();
    g_missSlot = slot;
}

// Logs `message` the first time `reason` (a bit) comes up.
void SkipOnce(uint32_t reason, const char* message)
{
    static uint32_t logged = 0;
    ++g_skips;
    unsigned long bit = 0;
    _BitScanForward(&bit, reason);
    NoteMiss((int) bit);
    if ((logged & reason) != 0)
        return;
    logged |= reason;
    LOG_WARN("TAA DLAA: {}; the game's TAA shows this frame", message);
}

// Jitter sources: 0 = cs cb3 of the TAA pass (the original), 1 = our copy of UE4's view uniform buffer (TaaMotion's, the
// base pass's vs cb1), where the same TemporalAAParams sit. F12 marks showed the 70% breaks bunched where cs cb3
// held no valid jitter for seconds at a time (0 of 128 reads) and DLSS ran on the prediction alone.
JitterSlot g_viewJitterSlots[kJitterSlots];
uint64_t g_sourceValid[2] = {};
uint64_t g_sourceInvalid[2] = {};
bool g_viewJitterDead = false;  // the view copy never validated: not the layout, stop reading it
uint64_t g_invalidStreak = 0;   // reads in a row with no valid jitter from either source
uint64_t g_invalidSince = 0;
int g_streakLogs = 0;

void ReadJitter(ID3D11DeviceContext* ctx, uint64_t present, JitterSlot* slots, int source)
{
    for (int s = 0; s < kJitterSlots; ++s)
    {
        auto& slot = slots[s];
        if (!slot.pending || present < slot.present + 2)
            continue;
        D3D11_MAPPED_SUBRESOURCE map {};
        const HRESULT result = ctx->Map(slot.staging, 0, D3D11_MAP_READ, D3D11_MAP_FLAG_DO_NOT_WAIT, &map);
        if (result == DXGI_ERROR_WAS_STILL_DRAWING)
            continue;
        slot.pending = false;
        if (FAILED(result))
            continue;
        float values[4] = {};
        std::memcpy(values, static_cast<const char*>(map.pData) + kJitterOffset, sizeof(values));
        ctx->Unmap(slot.staging, 0);

        const int index = (int) values[0];
        const int count = (int) values[1];
        const bool valid = values[0] == (float) index && values[1] == (float) count && count >= 1 && count <= 255 &&
                           index >= 1 && index <= count &&
                           std::fabs(values[2] - (Halton(index, 2) - 0.5f)) < 1e-4f &&
                           std::fabs(values[3] - (Halton(index, 3) - 0.5f)) < 1e-4f;
        g_jitterRing[g_jitterRingCount++ % kJitterRing] = { slot.present, valid ? index : 0,
                                                            g_jitterKnown ? PredictIndex(slot.present) : 0, source };
        if (!valid)
        {
            ++g_sourceInvalid[source];
            if (source == 0 && g_jitterInvalid++ == 0)
                LOG_WARN("TAA DLAA: cs cb3 +{} isn't the jitter layout ({} {} {} {})", kJitterOffset, values[0],
                         values[1], values[2], values[3]);
            if (source == 1 && g_sourceValid[1] == 0 && g_sourceInvalid[1] >= 300 && !g_viewJitterDead)
            {
                g_viewJitterDead = true;
                LOG_WARN("TAA DLAA: the view constants' +{} is not the jitter either (300 reads, e.g. {} {} {} {}); "
                         "reading cs cb3 only",
                         kJitterOffset, values[0], values[1], values[2], values[3]);
            }
            // A run with no valid jitter from either source: DLSS gets the prediction alone. Logged when it passes
            // 60 reads (cap 40 lines); its end is logged by the next valid read.
            if (source == 0 && ++g_invalidStreak == 1)
                g_invalidSince = slot.present;
            if (source == 0 && g_invalidStreak == 60 && g_streakLogs < 40)
            {
                ++g_streakLogs;
                LOG_WARN("TAA DLAA: no valid jitter for 60 reads since Present {} (cs cb3 e.g. {} {} {} {}); DLSS runs "
                         "on the prediction",
                         g_invalidSince, values[0], values[1], values[2], values[3]);
            }
            continue;
        }
        ++g_sourceValid[source];
        if (source == 1 && g_sourceValid[1] == 1)
            LOG_INFO("TAA DLAA: jitter also found in the view constants (+{}), index {} of {}", kJitterOffset, index,
                     count);
        if (g_invalidStreak >= 60 && g_streakLogs < 40)
        {
            ++g_streakLogs;
            LOG_INFO("TAA DLAA: valid jitter again at Present {} from {} after {} reads without (prediction {} "
                     "vs {})",
                     slot.present, source == 0 ? "cs cb3" : "the view constants", g_invalidStreak,
                     g_jitterKnown ? PredictIndex(slot.present) : 0, index);
        }
        g_invalidStreak = 0;
        ++g_jitterReads;
        if (count != g_sampleCount)
        {
            LOG_INFO("TAA DLAA: the game uses {} jitter samples{}", count,
                     count % 2 == 0 ? " (even: each eye gets only every other one)" : " (odd: each eye gets all of them)");
            g_sampleCount = count;
        }
        else if (g_jitterKnown && PredictIndex(slot.present) != index)
            ++g_jitterMismatches;
        g_baseIndex = index;
        g_basePresent = slot.present;
        g_jitterKnown = true;
        if (!g_jitterLoggedFirst)
        {
            g_jitterLoggedFirst = true;
            LOG_INFO("TAA DLAA: jitter found, index {} of {} ({:+.3f}, {:+.3f}) at Present {}", index, count, values[2],
                     values[3], slot.present);
        }
    }
}

// Copies `buffer` into the source's staging slot for this Present, read back a few Presents later (ReadJitter).
void QueueJitterCopy(ID3D11DeviceContext* ctx, uint64_t present, ID3D11Buffer* buffer, JitterSlot* slots)
{
    D3D11_BUFFER_DESC desc {};
    buffer->GetDesc(&desc);
    auto& slot = slots[present % kJitterSlots];
    if (desc.ByteWidth >= kJitterOffset + 16 && !slot.pending)
    {
        if (slot.staging != nullptr && slot.bytes != desc.ByteWidth)
        {
            slot.staging->Release();
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
            ctx->CopyResource(slot.staging, buffer);
            slot.present = present;
            slot.pending = true;
        }
    }
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
        LOG_ERROR("TAA DLAA: NGX init failed: {:X}", (UINT) result);
        return false;
    }
    return true;
}

ID3D11ComputeShader* CompileFinish(ID3D11Device* device, const char* source, size_t length, const char* name)
{
    ID3DBlob* code = nullptr;
    ID3DBlob* errors = nullptr;
    if (FAILED(D3DCompile(source, length, name, nullptr, nullptr, "CSMain", "cs_5_0", D3DCOMPILE_OPTIMIZATION_LEVEL3, 0,
                          &code, &errors)))
    {
        LOG_ERROR("TAA DLAA: {} compile failed: {}", name,
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

// The spare state and the finish shaders: needed by both replacing modes.
bool SetupCommon(ID3D11DeviceContext* ctx)
{
    ID3D11Device* device = nullptr;
    ctx->GetDevice(&device);
    if (device == nullptr)
        return false;
    if (!g_spareTried)
    {
        g_spareTried = true;
        ID3D11Device1* device1 = nullptr;
        if (SUCCEEDED(device->QueryInterface(IID_PPV_ARGS(&device1))) && device1 != nullptr)
        {
            const D3D_FEATURE_LEVEL level = device->GetFeatureLevel();
            const UINT flags = (device->GetCreationFlags() & D3D11_CREATE_DEVICE_SINGLETHREADED) != 0
                                   ? D3D11_1_CREATE_DEVICE_CONTEXT_STATE_SINGLETHREADED
                                   : 0;
            if (FAILED(device1->CreateDeviceContextState(flags, &level, 1, D3D11_SDK_VERSION,
                                                         __uuidof(ID3D11Device), nullptr, &g_spareState)))
                g_spareState = nullptr;
            device1->Release();
        }
        if (g_spareState == nullptr)
            LOG_ERROR("TAA DLAA: no spare device context state (needs D3D11.1); staying off");
    }
    if (!g_finishTried)
    {
        g_finishTried = true;
        g_finish = CompileFinish(device, kFinishShader, sizeof(kFinishShader) - 1, "TaaDlaa_Finish");
        g_upscaleFinish =
            CompileFinish(device, kUpscaleFinishShader, sizeof(kUpscaleFinishShader) - 1, "TaaDlaa_UpscaleFinish");
        D3D11_BUFFER_DESC cbDesc {};
        cbDesc.ByteWidth = 16;
        cbDesc.Usage = D3D11_USAGE_DYNAMIC;
        cbDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        cbDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
        if (FAILED(device->CreateBuffer(&cbDesc, nullptr, &g_sharpenConstants)))
            g_sharpenConstants = nullptr;
    }
    device->Release();
    return g_spareState != nullptr && g_finish != nullptr && g_upscaleFinish != nullptr;
}

// The DLSS quality level for an input/output ratio, which picks the preset ([DLSS] RenderPreset*).
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

// Frees a set so the next pass creates it again (at that pass's sizes). Only here, on the game's immediate context --
// the thread that creates and evaluates the features, so nothing of ours is recorded anywhere else. The D3D11 runtime
// keeps a released texture alive until the GPU is done with it; OptiScaler's ReleaseFeature waits 500 ms before it
// frees a feature (a second-long hitch per rebuild, logged).
void ReleaseSet(Features& set)
{
    for (auto& eye : set.eyes)
    {
        if (eye.handle != nullptr)
            NVSDK_NGX_D3D11_ReleaseFeature(eye.handle);
        eye.handle = nullptr;
        eye.lastPresent = UINT64_MAX;
    }
    if (set.outputSrv != nullptr)
        set.outputSrv->Release();
    if (set.output != nullptr)
        set.output->Release();
    set.outputSrv = nullptr;
    set.output = nullptr;
    set.createTried = false;
    set.ready = false;
    set.inWidth = set.inHeight = set.outWidth = set.outHeight = 0;
}

// A set made for other sizes is rebuilt once the same new sizes have been seen in kRebuildAfter gameplay passes: the game
// renders other sizes for a frame or two while the Render Scale settles, and those must not each cost a rebuild.
// True when the set was freed (the caller then creates it at the new sizes); false while waiting.
constexpr int kRebuildAfter = 30;
bool RebuildWhenSettled(Features& set, UINT inWidth, UINT inHeight, UINT outWidth, UINT outHeight)
{
    const UINT want[4] = { inWidth, inHeight, outWidth, outHeight };
    if (std::memcmp(set.pending, want, sizeof(want)) != 0)
    {
        std::memcpy(set.pending, want, sizeof(want));
        set.pendingCount = 0;
    }
    if (++set.pendingCount < kRebuildAfter)
        return false;
    set.pendingCount = 0;
    ++set.rebuilds;
    const auto start = std::chrono::steady_clock::now();
    LOG_INFO("TAA DLAA: {} features {}x{} -> {}x{} rebuilt for {}x{} -> {}x{} (the size held for {} passes; rebuild "
             "{})",
             set.name, set.inWidth, set.inHeight, set.outWidth, set.outHeight, inWidth, inHeight, outWidth, outHeight,
             kRebuildAfter, set.rebuilds);
    ReleaseSet(set);
    LOG_INFO("TAA DLAA: {} features released in {:.0f} ms", set.name,
             std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
    return true;
}

// Once per set: two features from the input size to the output size, and the output they write. Inside the spare
// state.
void CreateDlss(ID3D11DeviceContext* ctx, Features& set, UINT inWidth, UINT inHeight, UINT outWidth, UINT outHeight)
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
    bool ok = SUCCEEDED(device->CreateTexture2D(&desc, nullptr, &set.output)) &&
              SUCCEEDED(device->CreateShaderResourceView(set.output, nullptr, &set.outputSrv)) && InitNgx(device);
    device->Release();

    if (ok && g_params == nullptr && NVSDK_NGX_D3D11_GetCapabilityParameters(&g_params) != NVSDK_NGX_Result_Success)
    {
        LOG_ERROR("TAA DLAA: GetCapabilityParameters failed");
        ok = false;
    }
    const bool inverted = Config::Instance()->TaaDlaaDepthInverted.value_or_default();
    const auto quality = QualityFor(inWidth, outWidth);
    for (int eye = 0; ok && eye < 2; ++eye)
    {
        UINT flags = NVSDK_NGX_DLSS_Feature_Flags_AutoExposure;
        // Motion vectors are in input pixels (the DLAA features were made without the flag; the same at 1:1).
        if (inWidth < outWidth)
            flags |= NVSDK_NGX_DLSS_Feature_Flags_MVLowRes;
        if (inverted)
            flags |= NVSDK_NGX_DLSS_Feature_Flags_DepthInverted;
        g_params->Set(NVSDK_NGX_Parameter_DLSS_Feature_Create_Flags, flags);
        g_params->Set(NVSDK_NGX_Parameter_Width, inWidth);
        g_params->Set(NVSDK_NGX_Parameter_Height, inHeight);
        g_params->Set(NVSDK_NGX_Parameter_OutWidth, outWidth);
        g_params->Set(NVSDK_NGX_Parameter_OutHeight, outHeight);
        g_params->Set(NVSDK_NGX_Parameter_PerfQualityValue, quality);
        const auto result =
            NVSDK_NGX_D3D11_CreateFeature(ctx, NVSDK_NGX_Feature_SuperSampling, g_params, &set.eyes[eye].handle);
        if (result != NVSDK_NGX_Result_Success)
        {
            LOG_ERROR("TAA DLAA: {} CreateFeature for eye {} failed: {:X}", set.name, eye, (UINT) result);
            set.eyes[eye].handle = nullptr;
            ok = false;
        }
    }
    if (!ok)
        return;
    set.inWidth = inWidth;
    set.inHeight = inHeight;
    set.outWidth = outWidth;
    set.outHeight = outHeight;
    set.ready = true;
    LOG_INFO("TAA DLAA: created two {} features {}x{} -> {}x{} (quality value {}, depth {})", set.name, inWidth,
             inHeight, outWidth, outHeight, (int) quality, inverted ? "inverted" : "normal");
    // The preset OptiScaler hands DLSS for this quality level (its overlay's Render Presets Override; 0 = DLSS's
    // default). Read at creation, so a change in the overlay takes effect on the next rebuild ("Rebuild DLSS now").
    const auto* config = Config::Instance();
    const uint32_t preset = !config->RenderPresetOverride.value_or_default() ? 0u
                            : quality == NVSDK_NGX_PerfQuality_Value_DLAA ? config->RenderPresetDLAA.value_or_default()
                            : quality == NVSDK_NGX_PerfQuality_Value_MaxQuality
                                ? config->RenderPresetQuality.value_or_default()
                            : quality == NVSDK_NGX_PerfQuality_Value_Balanced ? config->RenderPresetBalanced.value_or_default()
                            : quality == NVSDK_NGX_PerfQuality_Value_MaxPerf ? config->RenderPresetPerformance.value_or_default()
                                : config->RenderPresetUltraPerformance.value_or_default();
    LOG_INFO("TAA DLAA: {} features requested preset {} ({}; override {}, for-all {})", set.name, preset,
             preset == 0 ? "DLSS default" : std::string(1, (char) ('A' + preset - 1)),
             config->RenderPresetOverride.value_or_default() ? "on" : "off",
             config->RenderPresetForAll.value_or_default());
}

bool Evaluate(ID3D11DeviceContext* ctx, Features& set, Eye& eye, bool reset, ID3D11Texture2D* colour,
              ID3D11Texture2D* motion, ID3D11Texture2D* depth, float jitterX, float jitterY)
{
    g_params->Set(NVSDK_NGX_Parameter_Jitter_Offset_X, jitterX);
    g_params->Set(NVSDK_NGX_Parameter_Jitter_Offset_Y, jitterY);
    g_params->Set(NVSDK_NGX_Parameter_MV_Scale_X, 1.0f);
    g_params->Set(NVSDK_NGX_Parameter_MV_Scale_Y, 1.0f);
    g_params->Set(NVSDK_NGX_Parameter_DLSS_Exposure_Scale, 1.0f);
    // Our finish pass sharpens (Sharpened); OptiScaler's RCAS stays off.
    g_params->Set(NVSDK_NGX_Parameter_Sharpness, 0.0f);
    g_params->Set(NVSDK_NGX_Parameter_Reset, reset ? 1 : 0);
    g_params->Set(NVSDK_NGX_Parameter_Width, set.inWidth);
    g_params->Set(NVSDK_NGX_Parameter_Height, set.inHeight);
    g_params->Set(NVSDK_NGX_Parameter_OutWidth, set.outWidth);
    g_params->Set(NVSDK_NGX_Parameter_OutHeight, set.outHeight);
    // The input textures can be a few texels larger than the image (the render-resolution pass's are).
    g_params->Set(NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Width, set.inWidth);
    g_params->Set(NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Height, set.inHeight);
    g_params->Set(NVSDK_NGX_Parameter_Depth, depth);
    g_params->Set(NVSDK_NGX_Parameter_Color, colour);
    g_params->Set(NVSDK_NGX_Parameter_MotionVectors, motion);
    g_params->Set(NVSDK_NGX_Parameter_Output, set.output);
    // Neural Rendering runs inside this evaluate with a history of its own: name the eye, so NR pairs like the two
    // DLSS features do. Unset, NR's one history was reprojected between the eyes (FC6's "quicksand").
    g_params->Set(DlssNr::kNrEyeParam, (unsigned int) (&eye - &set.eyes[0]));
    // Disocclusion hint: where the history is known to be something else, DLSS favours the current colour.
    ID3D11Texture2D* bias = TaaMotion_Dx11::BiasMask();
    g_params->Set(NVSDK_NGX_Parameter_DLSS_Input_Bias_Current_Color_Mask, bias);
    static bool loggedBias = false;
    if (bias != nullptr && !loggedBias)
    {
        loggedBias = true;
        LOG_INFO("TAA DLAA: FIRST evaluate with the disocclusion mask ({} features)", set.name);
    }

    State::Instance().setInputApiName = ApiUpscalerInput::FSR2X_DX11;

    const auto result = NVSDK_NGX_D3D11_EvaluateFeature(ctx, eye.handle, g_params, nullptr);
    if (result != NVSDK_NGX_Result_Success)
    {
        static NVSDK_NGX_Result last = NVSDK_NGX_Result_Success;
        if (result != last)
        {
            last = result;
            LOG_ERROR("TAA DLAA: {} EvaluateFeature failed: {:X}", set.name, (UINT) result);
        }
        return false;
    }
    return true;
}

// Our sharpening's constants into b1 of the finish shaders (inside the spare state). Logged on each change.
void BindSharpen(ID3D11DeviceContext* ctx)
{
    const float strength = g_sharpness.load(std::memory_order_relaxed);
    const int radius = g_sharpenRadius.load(std::memory_order_relaxed);
    if (strength != g_sharpnessLogged || radius != g_sharpenRadiusLogged)
    {
        g_sharpnessLogged = strength;
        g_sharpenRadiusLogged = radius;
        LOG_INFO("TAA DLAA: sharpening {:.2f} at {} px in the finish pass{}", strength, radius,
                 g_sharpenConstants == nullptr ? " -- NOT APPLIED: no constant buffer" : "");
    }
    if (g_sharpenConstants == nullptr)
        return;
    D3D11_MAPPED_SUBRESOURCE map {};
    if (SUCCEEDED(ctx->Map(g_sharpenConstants, 0, D3D11_MAP_WRITE_DISCARD, 0, &map)))
    {
        const float values[4] = { strength, (float) radius, 0.0f, 0.0f };
        std::memcpy(map.pData, values, sizeof(values));
        ctx->Unmap(g_sharpenConstants, 0);
    }
    ctx->CSSetConstantBuffers(1, 1, &g_sharpenConstants);
}

// The finish pass: `source` (same encoding as the pass's colour input) to the game's output, with the game's HUD
// layer and constants. Inside the spare state.
void Finish(ID3D11DeviceContext* ctx, ID3D11ShaderResourceView* source, ID3D11ShaderResourceView* hud,
            ID3D11Buffer* constants, ID3D11UnorderedAccessView* output, UINT width, UINT height)
{
    ID3D11ShaderResourceView* srvs[2] = { source, hud };
    ctx->CSSetShader(g_finish, nullptr, 0);
    ctx->CSSetShaderResources(0, 2, srvs);
    ctx->CSSetConstantBuffers(0, 1, &constants);
    BindSharpen(ctx);
    ctx->CSSetUnorderedAccessViews(0, 1, &output, nullptr);
    ctx->Dispatch((width + 7) / 8, (height + 7) / 8, 1);
    ID3D11ShaderResourceView* nullSrvs[2] = {};
    ID3D11UnorderedAccessView* nullUav = nullptr;
    ctx->CSSetShaderResources(0, 2, nullSrvs);
    ctx->CSSetUnorderedAccessViews(0, 1, &nullUav, nullptr);
}

// F12: what DLSS was fed over the last 180 Presents (render thread, on the next replaced pass after the key).
void DumpMark(uint64_t present)
{
    const uint64_t since = present > 180 ? present - 180 : 0;
    const uint64_t n = std::min<uint64_t>(g_evalRingCount, kMarkRing);
    int evals = 0, upscaling = 0, resets = 0, gaps = 0, eyeRepeats = 0, uncomposed = 0;
    std::string resetsAt;
    const EvalRecord* prev = nullptr;
    for (uint64_t k = g_evalRingCount - n; k < g_evalRingCount; ++k)
    {
        const auto& r = g_evalRing[k % kMarkRing];
        if (r.present >= since)
        {
            ++evals;
            upscaling += r.upscale ? 1 : 0;
            uncomposed += r.composed ? 0 : 1;
            if (r.reset)
            {
                ++resets;
                resetsAt += std::format(" {}", r.present);
            }
            if (prev != nullptr && prev->present >= since)
            {
                if (r.present != prev->present + 1)
                    ++gaps;
                else if (r.eye == prev->eye)
                    ++eyeRepeats;
            }
        }
        prev = &r;
    }
    std::string tail;
    for (uint64_t k = g_evalRingCount - std::min<uint64_t>(n, 16); k < g_evalRingCount; ++k)
    {
        const auto& r = g_evalRing[k % kMarkRing];
        tail += std::format(" {}:e{}{}j{}{}", r.present, r.eye, r.reset ? "R" : "", r.jitterIndex, r.composed ? "" : "u");
    }
    int valid = 0, invalid = 0, mismatched = 0, viewValid = 0;
    std::string mismatches;
    const uint64_t m = std::min<uint64_t>(g_jitterRingCount, kJitterRing);
    for (uint64_t k = g_jitterRingCount - m; k < g_jitterRingCount; ++k)
    {
        const auto& j = g_jitterRing[k % kJitterRing];
        if (j.present < since)
            continue;
        if (j.index == 0)
        {
            ++invalid;
            continue;
        }
        ++valid;
        viewValid += j.source == 1 ? 1 : 0;
        if (j.predicted != 0 && j.predicted != j.index)
        {
            ++mismatched;
            mismatches += std::format(" {}:{}!={}", j.present, j.index, j.predicted);
        }
    }
    LOG_WARN("TAA DLAA: F12 MARK at Present {} ({}): last 180 Presents: {} evaluations ({} upscaling), {} resets{}, {} "
             "gaps, {} eye repeats, {} without same-eye motion; jitter reads {} valid ({} from the view constants), {} "
             "invalid, {} mismatched{} ({} samples); last evaluations (present:eye R=reset j=index u=no motion):{}",
             present, TaaDlaa_Dx11::ModeName(), evals, upscaling, resets, resetsAt, gaps, eyeRepeats, uncomposed, valid,
             viewValid, invalid, mismatched, mismatches, g_sampleCount, tail);
}

void LogStats(uint64_t present)
{
    if (g_statsStart == 0)
        g_statsStart = present;
    if (present - g_statsStart < kStatsWindow)
        return;
    const double avgMs = g_evaluations > 0 ? (double) g_evalMicros / (double) g_evaluations / 1000.0 : 0.0;
    LOG_INFO("TAA DLAA: {} evaluations ({} resets, {} frames skipped) over {} Presents, CPU {:.2f} ms average, "
             "{:.2f} ms worst; {} upscale draws replaced; jitter: {} reads, {} mispredicted, {} invalid",
             g_evaluations, g_resets, g_skips, present - g_statsStart, avgMs, (double) g_evalMaxMicros / 1000.0,
             g_drawsReplaced, g_jitterReads, g_jitterMismatches, g_jitterInvalid);
    g_statsStart = present;
    g_evaluations = g_resets = g_skips = g_evalMicros = g_evalMaxMicros = g_drawsReplaced = 0;
}

// Eye and history reset for this Present. Eyes alternate while the chain of Presents with an evaluation is unbroken;
// after a break both start over.
void PickEye(Features& set, bool composed, uint64_t present, int& eyeIndex, bool& reset)
{
    if (!composed || g_lastReplacedPresent == UINT64_MAX || g_lastReplacedPresent + 1 != present)
    {
        g_nextEye = 0;
        for (auto& eye : set.eyes)
            eye.lastPresent = UINT64_MAX;
    }
    eyeIndex = g_nextEye;
    reset = !composed || set.eyes[eyeIndex].lastPresent == UINT64_MAX || set.eyes[eyeIndex].lastPresent + 2 != present;
}

void JitterFor(uint64_t present, float& x, float& y)
{
    const int index = PredictIndex(present);
    const int variant = g_jitterVariant.load(std::memory_order_relaxed);
    x = kJitterSigns[variant][0] * (Halton(index, 2) - 0.5f);
    y = kJitterSigns[variant][1] * (Halton(index, 3) - 0.5f);
}

// Runs `work` inside the spare device context state; the game's state comes back after. False if it can't swap.
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

// A 10-bit UAV on the game's upscale target for this draw only: the caller releases it right after the finish pass.
// Never kept: in VR that target is the eye image the swap chain resizes, and a view kept on it (it holds the texture)
// made the game's ResizeBuffers fail with DXGI_ERROR_INVALID_CALL -- a crash on changing R.E.A.L. VR's resolution
// below 100% Render Scale. nullptr if the texture can't take one.
ID3D11UnorderedAccessView* CreateTargetUav(ID3D11DeviceContext* ctx, ID3D11Texture2D* texture)
{
    const auto desc = Desc(texture);
    static bool loggedBad = false;
    if ((desc.BindFlags & D3D11_BIND_UNORDERED_ACCESS) == 0 ||
        (desc.Format != DXGI_FORMAT_R10G10B10A2_TYPELESS && desc.Format != DXGI_FORMAT_R10G10B10A2_UNORM))
    {
        if (!loggedBad)
        {
            loggedBad = true;
            LOG_WARN("TAA DLAA: the upscale target (format {}, bind {:X}) can't take a UAV", (UINT) desc.Format,
                     desc.BindFlags);
        }
        return nullptr;
    }
    ID3D11Device* device = nullptr;
    ctx->GetDevice(&device);
    if (device == nullptr)
        return nullptr;
    D3D11_UNORDERED_ACCESS_VIEW_DESC uavDesc {};
    uavDesc.Format = DXGI_FORMAT_R10G10B10A2_UNORM;
    uavDesc.ViewDimension = D3D11_UAV_DIMENSION_TEXTURE2D;
    ID3D11UnorderedAccessView* uav = nullptr;
    if (FAILED(device->CreateUnorderedAccessView(texture, &uavDesc, &uav)))
    {
        uav = nullptr;
        static bool loggedFailed = false;
        if (!loggedFailed)
        {
            loggedFailed = true;
            LOG_WARN("TAA DLAA: UAV on the upscale target failed");
        }
    }
    device->Release();
    return uav;
}

// The upscale finish: the DLSS output to the game's upscale target, with the draw's HUD layer (t0) and ps cb0.
// Inside the spare state.
void FinishUpscale(ID3D11DeviceContext* ctx, ID3D11ShaderResourceView* hud, ID3D11Buffer* constants,
                   ID3D11UnorderedAccessView* target, UINT width, UINT height)
{
    ID3D11ShaderResourceView* srvs[2] = { g_upscale.outputSrv, hud };
    ctx->CSSetShader(g_upscaleFinish, nullptr, 0);
    ctx->CSSetShaderResources(0, 2, srvs);
    ctx->CSSetConstantBuffers(0, 1, &constants);
    BindSharpen(ctx);
    ctx->CSSetUnorderedAccessViews(0, 1, &target, nullptr);
    ctx->Dispatch((width + 7) / 8, (height + 7) / 8, 1);
    ID3D11ShaderResourceView* nullSrvs[2] = {};
    ID3D11UnorderedAccessView* nullUav = nullptr;
    ctx->CSSetShaderResources(0, 2, nullSrvs);
    ctx->CSSetUnorderedAccessViews(0, 1, &nullUav, nullptr);
}

// True when a texture holds an image of this size, with at most a few texels to spare (the render-resolution pass's
// textures are 2 larger than its output).
bool Covers(const D3D11_TEXTURE2D_DESC& outer, UINT width, UINT height)
{
    return outer.Width >= width && outer.Height >= height && outer.Width <= width + 16 && outer.Height <= height + 16;
}

// The game's bindings for the pass, gathered before the state swap. All AddRef'd.
struct GameViews
{
    ID3D11UnorderedAccessView* output = nullptr;
    ID3D11ShaderResourceView* srvs[3] = {};
    ID3D11Buffer* constants = nullptr;

    ~GameViews()
    {
        if (output != nullptr)
            output->Release();
        for (auto* srv : srvs)
        {
            if (srv != nullptr)
                srv->Release();
        }
        if (constants != nullptr)
            constants->Release();
    }
};
// At the render-resolution pass: DLSS from its colour to the eye size, left for the game's upscale draw to show
// (ReplaceDraw). True when DLSS evaluated.
bool EvaluateUpscale(ID3D11DeviceContext* ctx, uint64_t present, ID3D11Texture2D* composed)
{
    GameViews game;
    ctx->CSGetUnorderedAccessViews(0, 1, &game.output);
    ctx->CSGetShaderResources(0, 3, game.srvs);
    ID3D11Texture2D* output = TextureFromView(game.output);
    ID3D11Texture2D* colour = TextureFromView(game.srvs[1]);
    const auto release = [&]
    {
        if (output != nullptr)
            output->Release();
        if (colour != nullptr)
            colour->Release();
    };
    if (output == nullptr || colour == nullptr)
    {
        SkipOnce(1, "the pass's bindings are incomplete");
        release();
        return false;
    }
    const auto renderDesc = Desc(output);
    const UINT renderWidth = renderDesc.Width;
    const UINT renderHeight = renderDesc.Height;

    if (g_eyeWidth == 0)
    {
        SkipOnce(16, "the game's upscale draw hasn't been seen yet (eye size unknown)");
        release();
        return false;
    }
    // A rebuild counts gameplay passes only: the G-buffer's depth at the render size (as checked again below).
    ID3D11Texture2D* gameplayDepth = FinalImageNr_Dx11::FrameDepth();
    const bool gameplay = gameplayDepth != nullptr && Covers(Desc(gameplayDepth), renderWidth, renderHeight);
    if (g_upscale.ready && (renderWidth != g_upscale.inWidth || renderHeight != g_upscale.inHeight ||
                            g_eyeWidth != g_upscale.outWidth || g_eyeHeight != g_upscale.outHeight) &&
        (!gameplay || !RebuildWhenSettled(g_upscale, renderWidth, renderHeight, g_eyeWidth, g_eyeHeight)))
    {
        SkipOnce(32, "the render or eye size changed since the upscaling features were made (rebuilt once it holds)");
        g_lastReplacedPresent = UINT64_MAX;
        release();
        return false;
    }
    ID3D11Texture2D* depth = FinalImageNr_Dx11::FrameDepth();
    if (depth == nullptr || !Covers(Desc(depth), renderWidth, renderHeight) ||
        !Covers(Desc(colour), renderWidth, renderHeight))
    {
        SkipOnce(64, "no gameplay depth at the render size this frame");
        g_lastReplacedPresent = UINT64_MAX;
        release();
        return false;
    }
    if (!g_jitterKnown)
    {
        SkipOnce(4, "jitter not known yet");
        release();
        return false;
    }
    if (!SetupCommon(ctx))
    {
        NoteMiss(kMissNoSpareState);
        release();
        return false;
    }

    int eyeIndex = 0;
    bool reset = false;
    PickEye(g_upscale, composed != nullptr, present, eyeIndex, reset);
    float jitterX = 0.0f;
    float jitterY = 0.0f;
    JitterFor(present, jitterX, jitterY);
    // Without the previous Present's motion (a reset anyway) DLSS still needs a motion input.
    ID3D11Texture2D* motion = composed != nullptr ? composed : TaaMotion_Dx11::Decoded();

    LARGE_INTEGER frequency {};
    LARGE_INTEGER start {};
    LARGE_INTEGER end {};
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&start);
    const bool evaluated = InSpareState(ctx,
                                        [&]
                                        {
                                            if (!g_upscale.createTried)
                                            {
                                                CreateDlss(ctx, g_upscale, renderWidth, renderHeight, g_eyeWidth,
                                                           g_eyeHeight);
                                            }
                                            return g_upscale.ready &&
                                                   Evaluate(ctx, g_upscale, g_upscale.eyes[eyeIndex], reset, colour,
                                                            motion, depth, jitterX, jitterY);
                                        });
    QueryPerformanceCounter(&end);
    release();

    if (!evaluated)
    {
        NoteMiss(g_upscale.ready ? kMissEvaluateFailed : kMissNoFeatures);
        g_lastReplacedPresent = UINT64_MAX;
        return false;
    }
    g_lastReplacedPresent = present;
    g_upscalePendingPresent = present;
    g_upscale.eyes[eyeIndex].lastPresent = present;
    g_nextEye = 1 - eyeIndex;
    RecordEval(present, eyeIndex, reset, PredictIndex(present), jitterX, jitterY, true, composed != nullptr);
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
        LOG_INFO("TAA DLAA: first upscaling evaluation {}x{} -> {}x{}, eye {}, jitter ({:+.3f}, {:+.3f}), {} us",
                 renderWidth, renderHeight, g_eyeWidth, g_eyeHeight, eyeIndex, jitterX, jitterY, micros);
    }
    return true;
}
} // namespace

bool TaaDlaa_Dx11::Enabled()
{
    static int cached = -1;
    if (cached < 0)
        cached = Config::Instance()->TaaDlaa.value_or_default() ? 1 : 0;
    return cached == 1;
}

bool TaaDlaa_Dx11::InPass() { return g_inPass > 0; }

void TaaDlaa_Dx11::BeforeDispatch(ID3D11DeviceContext* ctx)
{
    // Only the pass TaaMotion_Dx11 just decoded.
    if (TaaMotion_Dx11::Decoded() == nullptr)
        return;
    const uint64_t present = TaaMotion_Dx11::PresentIndex();
    // The render-resolution pass's output, to find the game's upscale draw that reads it.
    if (TaaMotion_Dx11::RenderResolution())
    {
        ID3D11UnorderedAccessView* view = nullptr;
        ctx->CSGetUnorderedAccessViews(0, 1, &view);
        if (view != nullptr)
        {
            ID3D11Resource* resource = nullptr;
            view->GetResource(&resource);
            g_renderTaaOutput = resource;
            g_renderTaaPresent = present;
            if (resource != nullptr)
                resource->Release();
            view->Release();
        }
    }
    ReadJitter(ctx, present, g_jitterSlots, 0);
    if (!g_viewJitterDead)
        ReadJitter(ctx, present, g_viewJitterSlots, 1);
    ID3D11Buffer* cb3 = nullptr;
    ctx->CSGetConstantBuffers(3, 1, &cb3);
    if (cb3 != nullptr)
    {
        QueueJitterCopy(ctx, present, cb3, g_jitterSlots);
        cb3->Release();
    }
    if (!g_viewJitterDead)
    {
        if (ID3D11Buffer* view = TaaMotion_Dx11::ViewBuffer())
            QueueJitterCopy(ctx, present, view, g_viewJitterSlots);
    }
}

bool TaaDlaa_Dx11::Replace(ID3D11DeviceContext* ctx)
{
    ID3D11Texture2D* composed = TaaMotion_Dx11::Composed();
    if (TaaMotion_Dx11::Decoded() == nullptr)
        return false;
    const uint64_t present = TaaMotion_Dx11::PresentIndex();
    if (g_markRequested.exchange(false))
        DumpMark(present);
    const int mode = g_mode.load(std::memory_order_relaxed);
    if (mode == GameTaa)
    {
        g_statsStart = 0;
        g_evaluations = g_resets = g_skips = g_evalMicros = g_evalMaxMicros = g_drawsReplaced = 0;
        return false;
    }
    LogStats(present);
    // "Rebuild DLSS now" (overlay): both sets are freed here, on the thread that creates and evaluates them, and made
    // again by this pass with the current settings (e.g. a preset picked in OptiScaler's overlay). ~1 s per set.
    if (g_rebuildRequested.exchange(false))
    {
        LOG_INFO("TAA DLAA: rebuild requested from the overlay (Present {})", present);
        if (g_dlaa.ready || g_dlaa.createTried)
            ReleaseSet(g_dlaa);
        if (g_upscale.ready || g_upscale.createTried)
            ReleaseSet(g_upscale);
        g_lastReplacedPresent = UINT64_MAX;
    }
    // The render-resolution pass (Render Scale below 100%): DLSS upscaling, shown at the game's upscale draw. The
    // game's pass is skipped ([Inputs] TaaDlaaSkipGameTaa) once the draw has been replaced on the previous Present;
    // until then it runs, so the draw falls back to the game's own upscale if it isn't replaced.
    if (TaaMotion_Dx11::RenderResolution())
    {
        const bool evaluated = EvaluateUpscale(ctx, present, composed);
        const bool skip = evaluated && g_drawReplacedPresent != UINT64_MAX && g_drawReplacedPresent + 1 == present &&
                          Config::Instance()->TaaDlaaSkipGameTaa.value_or_default();
        static bool loggedSkip = false;
        if (skip && !loggedSkip)
        {
            loggedSkip = true;
            LOG_INFO("TAA DLAA: the game's render-resolution TAA pass is skipped while upscaling");
        }
        return skip;
    }

    GameViews game;
    ctx->CSGetUnorderedAccessViews(0, 1, &game.output);
    ctx->CSGetShaderResources(0, 3, game.srvs);
    ctx->CSGetConstantBuffers(0, 1, &game.constants);
    ID3D11Texture2D* output = TextureFromView(game.output);
    ID3D11Texture2D* colour = TextureFromView(game.srvs[2]);
    if (output == nullptr || colour == nullptr || game.srvs[0] == nullptr || game.constants == nullptr)
    {
        if (output != nullptr)
            output->Release();
        if (colour != nullptr)
            colour->Release();
        SkipOnce(1, "the pass's bindings are incomplete");
        return false;
    }
    const auto outDesc = Desc(output);
    output->Release();

    if (!SetupCommon(ctx))
    {
        NoteMiss(kMissNoSpareState);
        colour->Release();
        return false;
    }

    // DLAA inputs.
    ID3D11Texture2D* depth = nullptr;
    bool reset = false;
    int eyeIndex = 0;
    float jitterX = 0.0f;
    float jitterY = 0.0f;
    if (mode == Dlaa)
    {
        depth = FinalImageNr_Dx11::FrameDepth();
        // Gameplay only: the pass has the G-buffer's depth at its own size (menus and loading screens do not).
        const bool depthOk =
            depth != nullptr && Desc(depth).Width == outDesc.Width && Desc(depth).Height == outDesc.Height;
        const bool sizeOk = depthOk && (!g_dlaa.ready || (outDesc.Width == g_dlaa.inWidth &&
                                                          outDesc.Height == g_dlaa.inHeight) ||
                                        RebuildWhenSettled(g_dlaa, outDesc.Width, outDesc.Height, outDesc.Width,
                                                           outDesc.Height));
        if (!sizeOk)
        {
            if (depthOk)
                SkipOnce(128, "the pass changed size since the DLAA features were made (rebuilt once it holds)");
            else
                SkipOnce(2, "no gameplay depth at the pass's size this frame");
            g_lastReplacedPresent = UINT64_MAX;
            colour->Release();
            return false;
        }
        if (!g_jitterKnown)
        {
            SkipOnce(4, "jitter not known yet");
            colour->Release();
            return false;
        }
        PickEye(g_dlaa, composed != nullptr, present, eyeIndex, reset);
        JitterFor(present, jitterX, jitterY);
    }

    ID3D11DeviceContext1* ctx1 = nullptr;
    if (FAILED(ctx->QueryInterface(IID_PPV_ARGS(&ctx1))) || ctx1 == nullptr)
    {
        NoteMiss(kMissNoContext1);
        colour->Release();
        return false;
    }

    LARGE_INTEGER frequency {};
    LARGE_INTEGER start {};
    LARGE_INTEGER end {};
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&start);
    bool replaced = false;
    {
        PassScope scope;
        ID3DDeviceContextState* gameState = nullptr;
        ctx1->SwapDeviceContextState(g_spareState, &gameState);

        if (!g_dlaa.createTried)
            CreateDlss(ctx, g_dlaa, outDesc.Width, outDesc.Height, outDesc.Width, outDesc.Height);
        // Without the previous Present's motion (a reset anyway) DLSS still needs a motion input.
        ID3D11Texture2D* motion = composed != nullptr ? composed : TaaMotion_Dx11::Decoded();
        if (g_dlaa.ready &&
            Evaluate(ctx, g_dlaa, g_dlaa.eyes[eyeIndex], reset, colour, motion, depth, jitterX, jitterY))
        {
            Finish(ctx, g_dlaa.outputSrv, game.srvs[0], game.constants, game.output, outDesc.Width, outDesc.Height);
            replaced = true;
        }

        ID3DDeviceContextState* spare = nullptr;
        ctx1->SwapDeviceContextState(gameState, &spare);
        if (spare != nullptr)
            spare->Release(); // g_spareState keeps its own reference
        if (gameState != nullptr)
            gameState->Release();
    }
    ctx1->Release();
    colour->Release();
    QueryPerformanceCounter(&end);

    if (!replaced)
    {
        NoteMiss(g_dlaa.ready ? kMissEvaluateFailed : kMissNoFeatures);
        g_lastReplacedPresent = UINT64_MAX;
        return false;
    }
    g_lastReplacedPresent = present;
    g_shownPresent = present;

    g_dlaa.eyes[eyeIndex].lastPresent = present;
    g_nextEye = 1 - eyeIndex;
    RecordEval(present, eyeIndex, reset, PredictIndex(present), jitterX, jitterY, false, composed != nullptr);
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
        LOG_INFO("TAA DLAA: first evaluation, eye {}, jitter ({:+.3f}, {:+.3f}), {} us; game TAA pass {}", eyeIndex,
                 jitterX, jitterY, micros,
                 Config::Instance()->TaaDlaaSkipGameTaa.value_or_default() ? "skipped" : "still runs first");
    }
    return true;
}

bool TaaDlaa_Dx11::ReplaceDraw(ID3D11DeviceContext* ctx)
{
    const uint64_t present = TaaMotion_Dx11::PresentIndex();
    // Only the first draw after this Present's render-resolution pass that reads its output.
    if (g_renderTaaPresent != present || g_upscaleDrawPresent == present)
        return false;
    ID3D11ShaderResourceView* srvs[2] = {};
    ctx->PSGetShaderResources(0, 2, srvs);
    const auto releaseSrvs = [&]
    {
        for (auto* srv : srvs)
        {
            if (srv != nullptr)
                srv->Release();
        }
    };
    bool reads = false;
    if (srvs[1] != nullptr)
    {
        ID3D11Resource* resource = nullptr;
        srvs[1]->GetResource(&resource);
        reads = resource != nullptr && resource == g_renderTaaOutput;
        if (resource != nullptr)
            resource->Release();
    }
    ID3D11RenderTargetView* rtv = nullptr;
    if (reads)
        ctx->OMGetRenderTargets(1, &rtv, nullptr);
    ID3D11Texture2D* target = TextureFromView(rtv);
    if (rtv != nullptr)
        rtv->Release();
    if (target == nullptr)
    {
        releaseSrvs();
        return false;
    }
    g_upscaleDrawPresent = present;
    const auto desc = Desc(target);
    if (desc.Width != g_eyeWidth || desc.Height != g_eyeHeight)
    {
        LOG_INFO("TAA DLAA: the game's upscale draw writes {}x{} (format {})", desc.Width, desc.Height,
                 (UINT) desc.Format);
        g_eyeWidth = desc.Width;
        g_eyeHeight = desc.Height;
    }

    bool replaced = false;
    if (g_upscalePendingPresent == present && g_upscale.ready && desc.Width == g_upscale.outWidth &&
        desc.Height == g_upscale.outHeight && srvs[0] != nullptr)
    {
        ID3D11Buffer* constants = nullptr;
        ctx->PSGetConstantBuffers(0, 1, &constants);
        ID3D11UnorderedAccessView* uav = CreateTargetUav(ctx, target);
        if (constants != nullptr && uav != nullptr)
        {
            replaced = InSpareState(ctx,
                                    [&]
                                    {
                                        FinishUpscale(ctx, srvs[0], constants, uav, desc.Width, desc.Height);
                                        return true;
                                    });
        }
        if (constants != nullptr)
            constants->Release();
        // FinishUpscale unbound it; the D3D11 runtime keeps the texture only for the work already queued.
        if (uav != nullptr)
            uav->Release();
    }
    target->Release();
    releaseSrvs();
    // Found the draw for a pass DLSS evaluated, and could not replace it (size, HUD input, constants or UAV).
    if (!replaced && g_upscalePendingPresent == present)
        NoteMiss(kMissDrawNotReplaced);
    if (!replaced)
        return false;
    g_shownPresent = present;
    g_drawReplacedPresent = present;
    ++g_drawsReplaced;
    static bool logged = false;
    if (!logged)
    {
        logged = true;
        LOG_INFO("TAA DLAA: first upscale draw replaced ({}x{})", desc.Width, desc.Height);
    }
    return true;
}

bool TaaDlaa_Dx11::ReplacedThisPresent()
{
    return g_shownPresent != UINT64_MAX && g_shownPresent == TaaMotion_Dx11::PresentIndex();
}

bool TaaDlaa_Dx11::On() { return g_mode.load(std::memory_order_relaxed) == Dlaa; }

void TaaDlaa_Dx11::RequestRebuild() { g_rebuildRequested = true; }

float TaaDlaa_Dx11::Sharpness() { return g_sharpness.load(); }

void TaaDlaa_Dx11::SetSharpness(float sharpness) { g_sharpness = std::clamp(sharpness, 0.0f, 2.0f); }

int TaaDlaa_Dx11::SharpenRadius() { return g_sharpenRadius.load(); }

void TaaDlaa_Dx11::SetSharpenRadius(int radius) { g_sharpenRadius = std::clamp(radius, 1, 2); }

void TaaDlaa_Dx11::UserMark()
{
    LOG_WARN("TAA DLAA: F12 pressed at Present {} (the dump follows at the next DLSS pass)",
             TaaMotion_Dx11::PresentIndex());
    g_markRequested = true;
}

int TaaDlaa_Dx11::MissSlot(uint64_t present) { return g_missPresent == present ? g_missSlot : kMissNothing; }

const char* TaaDlaa_Dx11::MissSlotName(int slot)
{
    switch (slot)
    {
    case 0:
        return "DLSS: the pass's bindings are incomplete";
    case 1:
        return "DLSS: no gameplay depth at the pass's size";
    case 2:
        return "DLSS: jitter not known yet";
    case 4:
        return "DLSS: eye size unknown (no upscale draw seen yet)";
    case 5:
        return "DLSS: render or eye size changed, upscaling features wait to rebuild";
    case 6:
        return "DLSS: no gameplay depth at the render size";
    case 7:
        return "DLSS: pass size changed, DLAA features wait to rebuild";
    case kMissNoSpareState:
        return "DLSS: no spare context state or finish shaders";
    case kMissNoContext1:
        return "DLSS: no ID3D11DeviceContext1";
    case kMissNoFeatures:
        return "DLSS: the features were not made";
    case kMissEvaluateFailed:
        return "DLSS: EvaluateFeature failed";
    case kMissDrawNotReplaced:
        return "DLSS: the upscale draw was found but not replaced";
    case kMissNothing:
        return "DLSS: nothing recorded (at render resolution: no upscale draw read the pass's output)";
    default:
        return "DLSS: unknown reason";
    }
}

void TaaDlaa_Dx11::CycleMode()
{
    g_mode = g_mode.load() == Dlaa ? GameTaa : Dlaa;
    LOG_INFO("TAA DLAA: mode {}", ModeName());
}

const char* TaaDlaa_Dx11::ModeName()
{
    switch (g_mode.load())
    {
    case Dlaa:
        return "DLAA";
    default:
        return "game TAA";
    }
}

void TaaDlaa_Dx11::CycleJitter()
{
    g_jitterVariant = (g_jitterVariant.load() + 1) % 4;
    LOG_INFO("TAA DLAA: jitter {}", JitterName());
}

const char* TaaDlaa_Dx11::JitterName()
{
    switch (g_jitterVariant.load())
    {
    case 1:
        return "(-x, +y)";
    case 2:
        return "(+x, +y)";
    case 3:
        return "(-x, -y)";
    default:
        return "(+x, -y)";
    }
}
