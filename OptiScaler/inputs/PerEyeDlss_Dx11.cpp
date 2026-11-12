#include "pch.h"

#include "Config.h"
#include "Logger.h"
#include "NVNGX_Parameter.h"
#include <dlssnr/DlssNr.h>

#include "MotionMatrixProbe_Dx11.h"
#include "PerEyeDlss_Dx11.h"

#include <gpu_time/GpuTime_Dx11.h>
#include <gpu_time/StageTimes.h>

#include <d3d11_1.h>
#include <d3dcompiler.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace
{
// F8 views, drawn over DLSS's output: the picture in grey with motion as colour (render px: red = +x, blue = -x,
// green = |y|).
//   Mode 1, OBJECT MOTION: the game's vector minus the camera's motion to the previous Present (this Present's fitted
//           map), full colour at 4 px -- the static world stays grey whatever the head does; only what moves by itself
//           lights up. Magenta-grey = no fit this Present.
//   Mode 2, WHAT DLSS GETS: the vectors DLSS is handed (the game's in shared, same-eye ones per eye), full at 128 px.
constexpr char kViewShader[] = R"(
Texture2D<float4> Colour : register(t0);
Texture2D<float2> Motion : register(t1);
Texture2D<float2> Game : register(t2);
Texture2D<float> Depth : register(t3);
StructuredBuffer<float4> Fits : register(t4);
RWTexture2D<unorm float4> Output : register(u0);

cbuffer Constants : register(b0)
{
    float2 MvScale;
    uint Mode;
    uint Cur;
    float FullPx;
    uint HaveFit;
    float2 Unused;
};

static const float KZ = 100.0;

[numthreads(8, 8, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    uint ow, oh, rw, rh, dw, dh;
    Output.GetDimensions(ow, oh);
    Motion.GetDimensions(rw, rh);
    Depth.GetDimensions(dw, dh);
    if (id.x >= ow || id.y >= oh)
        return;
    const float2 uv = (float2(id.xy) + 0.5) / float2(ow, oh);
    const uint2 at = min(uint2(uv * float2(rw, rh)), uint2(rw - 1, rh - 1));
    const float grey = dot(Colour[at].rgb, float3(0.299, 0.587, 0.114)) * 0.4;
    float2 mv;
    if (Mode == 1)
    {
        if (HaveFit == 0 || Fits[Cur * 5 + 3].z < 0.5)
        {
            Output[id.xy] = float4(saturate(float3(grey + 0.2, grey, grey + 0.2)), 1.0);
            return;
        }
        // As the composition judges it: against the camera at this depth and at the nearest depth within 2 px (the
        // game's field is dilated), whichever the vector agrees with -- otherwise every silhouette shows a band of
        // the near object's parallax.
        const float2 xy = float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
        const int2 dat = int2(min(uint2(uv * float2(dw, dh)), uint2(dw - 1, dh - 1)));
        const float z = Depth[dat];
        float zn = z;
        for (int dy = -2; dy <= 2; ++dy)
            for (int dx = -2; dx <= 2; ++dx)
                zn = max(zn, Depth[clamp(dat + int2(dx, dy), int2(0, 0), int2(dw - 1, dh - 1))]);
        const float2 game = Game[at] * MvScale;
        float2 best = game;
        [unroll] for (uint k = 0; k < 2; ++k)
        {
            const float4 p = float4(xy, (k == 0 ? z : zn) * KZ, 1.0);
            const float wq = dot(Fits[Cur * 5 + 2], p);
            const float2 prev = float2(dot(Fits[Cur * 5 + 0], p), dot(Fits[Cur * 5 + 1], p)) / wq;
            const float2 r = game - (prev - xy) * float2(0.5, -0.5) * float2(rw, rh);
            if (k == 0 || dot(r, r) < dot(best, best))
                best = r;
        }
        mv = best;
    }
    else
        mv = Motion[at] * MvScale;
    const float3 tint = float3(saturate(mv.x / FullPx), saturate(abs(mv.y) / FullPx), saturate(-mv.x / FullPx));
    Output[id.xy] = float4(saturate(grey + tint), 1.0);
}
)";

// In-headset indicator (notifications only reach the flat window): a row of squares a little left of and below the
// centre -- [mode: green per eye / orange shared] [4 variant squares, the active one white] [view: off dark, cyan object
// motion, magenta what DLSS gets]. Shown while a view is on and for 3 s after F4 / F6 / F8.
constexpr char kIndicatorShader[] = R"(
RWTexture2D<unorm float4> Output : register(u0);

cbuffer Constants : register(b0)
{
    uint2 Origin;
    uint Cell;
    uint Routing;
    uint Variant;
    uint View;
    uint2 Unused;
};

[numthreads(8, 8, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    const uint slots = 8;
    if (id.x >= slots * Cell || id.y >= Cell)
        return;
    const uint slot = id.x / Cell;
    const uint2 inCell = uint2(id.x % Cell, id.y);
    const uint border = max(Cell / 8, 2u);
    float3 c = float3(0, 0, 0);
    const bool edge = inCell.x < border || inCell.y < border || inCell.x >= Cell - border || inCell.y >= Cell - border;
    if (slot == 1 || slot == 6)
        return; // gaps
    if (!edge)
    {
        if (slot == 0)
            c = Routing != 0 ? float3(0.1, 0.9, 0.2) : float3(1.0, 0.55, 0.05);
        else if (slot <= 5)
            c = (slot - 2) == Variant ? float3(1, 1, 1) : float3(0.18, 0.18, 0.18);
        else
            c = View == 1 ? float3(0.1, 0.9, 0.9) : View == 2 ? float3(0.9, 0.2, 0.9) : float3(0.18, 0.18, 0.18);
    }
    Output[Origin + id.xy] = float4(c, 1.0);
}
)";

// F7 frame strip: DLSS's output (after any F8 view) shrunk to 256x256, packed RGBA8.
constexpr char kStripShader[] = R"(
Texture2D<float4> Src : register(t0);
RWStructuredBuffer<uint> Out : register(u0);

[numthreads(8, 8, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    uint w, h;
    Src.GetDimensions(w, h);
    const float2 cell = float2(w, h) / 256.0;
    float3 sum = 0;
    [unroll] for (uint j = 0; j < 2; ++j)
        [unroll] for (uint i = 0; i < 2; ++i)
            sum += Src[min(uint2((float2(id.xy) + float2(i, j) * 0.5 + 0.25) * cell), uint2(w - 1, h - 1))].rgb;
    const uint3 c = uint3(saturate(sum * 0.25) * 255.0 + 0.5);
    Out[id.y * 256 + id.x] = c.r | (c.g << 8) | (c.b << 16) | (255u << 24);
}
)";

// Sums over a 64x64 grid of the motion (render px, fixed point x16): sum x, sum |x|, sum |y|, count.
constexpr char kStatsShader[] = R"(
Texture2D<float2> Motion : register(t0);
RWByteAddressBuffer Sums : register(u0);

cbuffer Constants : register(b0)
{
    float2 MvScale;
    float2 Unused;
};

[numthreads(8, 8, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    uint rw, rh;
    Motion.GetDimensions(rw, rh);
    const uint2 at = min(uint2((float2(id.xy) + 0.5) / 64.0 * float2(rw, rh)), uint2(rw - 1, rh - 1));
    const float2 mv = Motion[at] * MvScale;
    uint ignored;
    Sums.InterlockedAdd(0, (uint) (int) round(mv.x * 16.0), ignored);
    Sums.InterlockedAdd(4, (uint) (int) round(abs(mv.x) * 16.0), ignored);
    Sums.InterlockedAdd(8, (uint) (int) round(abs(mv.y) * 16.0), ignored);
    Sums.InterlockedAdd(12, 1u, ignored);
}
)";

// Matrix probe: a 32x32 grid of (depth, motion x px, motion y px) for offline fitting.
constexpr char kSampleShader[] = R"(
Texture2D<float> Depth : register(t0);
Texture2D<float2> Motion : register(t1);
Texture2D<float2> Given : register(t2);
RWStructuredBuffer<float4> Out : register(u0);

cbuffer Constants : register(b0)
{
    float2 MvScale;
    float2 Unused;
};

[numthreads(8, 8, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    uint rw, rh, dw, dh;
    Motion.GetDimensions(rw, rh);
    Depth.GetDimensions(dw, dh);
    const float2 uv = (float2(id.xy) + 0.5) / 32.0;
    const uint2 at = min(uint2(uv * float2(rw, rh)), uint2(rw - 1, rh - 1));
    const uint2 dat = min(uint2(uv * float2(dw, dh)), uint2(dw - 1, dh - 1));
    const uint i = id.y * 32 + id.x;
    Out[i * 2] = float4(Depth[dat], Motion[at] * MvScale, 0.0);
    Out[i * 2 + 1] = float4(Given[at] * MvScale, 0.0, 0.0);
}
)";


// ---- Same-eye motion from the game's own vectors ----
//
// The game's vectors point at the previous Present, the other eye. Each frame's static motion is one projective map
// M_N: clip (x, y, z, 1) -> previous clip (fitted offline on a D3D11 game's capture to 0.01 px median). FIT solves M_N from a 64x64
// grid of depth and motion (DLT with c.w = 1, 4 robust rounds, Cauchy weights at 4 / 1 / 0.3 px); COMPOSE chains
// M_{N-1} o M_N for the same-eye camera motion (two Presents back) and adds the object motion the matrix cannot see.
// Both work in render px with the DLSS convention (previous - current); z is scaled by 100 in the fit.
// The fit runs wide, in three stages and eleven dispatches (PerEyeFitShader.h): in one group it was most of what
// per eye cost (offline, fit_test: 0.60 ms a fit against 0.07).
#include "PerEyeFitShader.h"

constexpr char kComposeShader[] = R"(
Texture2D<float> Depth : register(t0);
Texture2D<float2> Motion : register(t1);
StructuredBuffer<float4> Fits : register(t2);
Texture2D<float2> PrevGame : register(t3);
RWTexture2D<float2> Out : register(u0);

cbuffer Constants : register(b0)
{
    float2 MvScale;
    uint Cur;
    uint Prev;
    float Knee0;
    float Knee1;
    float Gain;
    uint Radius;
    uint Variant;
    uint HavePrev;
    float2 Unused;
};

static const float KZ = 100.0;

float3 Apply(uint slot, float2 xy, float z)
{
    const float4 p = float4(xy, z * KZ, 1.0);
    const float wq = dot(Fits[slot * 5 + 2], p);
    return float3(dot(Fits[slot * 5 + 0], p) / wq, dot(Fits[slot * 5 + 1], p) / wq, z / wq);
}

[numthreads(8, 8, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    uint w, h, dw, dh;
    Out.GetDimensions(w, h);
    Depth.GetDimensions(dw, dh);
    if (id.x >= w || id.y >= h)
        return;
    const float2 size = float2(w, h);
    const float2 uv = (float2(id.xy) + 0.5) / size;
    const float2 xy = float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    const int2 dat = int2(min(uint2(uv * float2(dw, dh)), uint2(dw - 1, dh - 1)));
    const float z = Depth[dat];
    const float2 game = Motion[id.xy] * MvScale;
    // A slot is usable once it has held an accepted fit, while its eye has not been rejected more than twice in a row.
    if (Fits[Cur * 5 + 3].z < 0.5 || Fits[Prev * 5 + 3].z < 0.5 || Fits[Cur * 5 + 4].y > 2.0 ||
        Fits[Prev * 5 + 4].y > 2.0)
    {
        Out[id.xy] = Motion[id.xy];
        return;
    }
    const float2 toPx = float2(0.5, -0.5) * size;

    // Camera: this Present to the previous (the other eye), then on to the one before (this eye again).
    const float3 c1 = Apply(Cur, xy, z);
    const float3 c2 = Apply(Prev, c1.xy, c1.z);
    const float2 sameEye = (c2.xy - xy) * toPx;

    // Objects: what the game's vector has beyond camera motion to the other eye, judged at this depth and at the
    // nearest depth around it (the field is dilated), only where this pixel is itself that nearest surface.
    float zn = z;
    const int r = (int) Radius;
    for (int dy = -r; dy <= r; ++dy)
        for (int dx = -r; dx <= r; ++dx)
            zn = max(zn, Depth[clamp(dat + int2(dx, dy), int2(0, 0), int2(dw - 1, dh - 1))]);
    const float2 resHere = game - (c1.xy - xy) * toPx;
    const float2 resNear = game - (Apply(Cur, xy, zn).xy - xy) * toPx;
    const float2 res = dot(resNear, resNear) < dot(resHere, resHere) ? resNear : resHere;
    // F4 variants: 0 as validated (knee 0.1-0.4 px, nearest surface only, one Present of object motion x2); 1 knee
    // 0.02-0.1 px (the vectors are float, small motion is real); 2 = 1 + object pixels take the game's real motion over
    // both Presents (its vector, then the previous Present's vector where it lands); 3 = 2 without the nearest-surface
    // rule.
    const float nearest = (Variant == 3 || z >= zn * 0.9) ? 1.0 : 0.0;
    const float k0 = Variant == 0 ? Knee0 : 0.02;
    const float k1 = Variant == 0 ? Knee1 : 0.1;
    const float knee = smoothstep(k0, k1, length(res)) * nearest;

    float2 result = sameEye + Gain * knee * res;
    if (Variant >= 2 && HavePrev != 0)
    {
        const float2 landed = float2(id.xy) + 0.5 + game;
        const int2 q = clamp(int2(floor(landed)), int2(0, 0), int2(w - 1, h - 1));
        const float2 both = game + PrevGame[q] * MvScale;
        result = lerp(sameEye, both, knee);
    }
    Out[id.xy] = result / MvScale;
}
)";

std::mutex g_mutex;
std::atomic<uint64_t> g_present { 0 };

// Twins and routing.
std::unordered_map<unsigned int, unsigned int> g_twins;
bool g_routing = false; // starts as the game (mod) made it, F6 for per eye -- the same habit as FC5, FC6, Days Gone
bool g_resetPrimary = false;
bool g_resetTwin = false;

// Per window of Presents.
uint64_t g_windowStart = 0;
uint64_t g_evalsThisPresent = 0;
uint64_t g_toPrimary = 0, g_toTwin = 0, g_noTwin = 0;
uint64_t g_presentsWithout = 0, g_presentsWithSeveral = 0;
bool g_seenEvaluate = false;
constexpr uint64_t kWindow = 600;
// The last whole window, kept for the overlay's section. Written and read under g_mutex.
struct LastWindow
{
    uint64_t presents = 0, toGame = 0, toTwin = 0, sameEye = 0, chainBreaks = 0, fitsRead = 0, fitsValid = 0;
    double fitResidual = 0, fitInliers = 0;
    bool perEye = false; // per eye from its first Present to its last
};
LastWindow g_lastWindow;
bool g_windowSwitched = false; // the routing was switched within the window that is being counted

// Motion statistics by Present parity.
struct Parity
{
    double sumX = 0, sumAbsX = 0, sumAbsY = 0;
    uint64_t frames = 0;
};
Parity g_parity[2];

// GPU side.
bool g_setupTried = false;
bool g_ready = false;
ID3D11Device* g_device = nullptr; // compared only; the context's device, no reference kept
ID3D11ComputeShader* g_view = nullptr;
ID3D11ComputeShader* g_stats = nullptr;
ID3D11Buffer* g_constants = nullptr;
ID3D11Buffer* g_sums = nullptr;
ID3D11UnorderedAccessView* g_sumsUav = nullptr;
constexpr int kRing = 4;
ID3D11Buffer* g_staging[kRing] = {};
int g_stagingParity[kRing] = { -1, -1, -1, -1 };
uint64_t g_statsCount = 0;
ID3DDeviceContextState* g_spareState = nullptr;
ID3D11ComputeShader* g_sample = nullptr;
ID3D11Buffer* g_samples = nullptr;
ID3D11UnorderedAccessView* g_samplesUav = nullptr;
constexpr int kSampleRing = 8;
struct SampleSlot
{
    ID3D11Buffer* staging = nullptr;
    bool pending = false;
    uint64_t present = 0;
    int parity = 0;
    float mvScale[2] = {};
    float jitter[2] = {};
    unsigned int renderW = 0, renderH = 0;
};
SampleSlot g_sampleSlots[kSampleRing];
uint64_t g_sampleCount = 0;
bool g_loggedSampleFail = false;
// Same-eye motion.
bool g_motionReady = false;
bool g_loggedMotionFail = false;
ID3D11ComputeShader* g_fitInit = nullptr;
ID3D11ComputeShader* g_fitAccumulate = nullptr;
ID3D11ComputeShader* g_fitSolve = nullptr;
ID3D11ComputeShader* g_compose = nullptr;
ID3D11Buffer* g_motionConstants = nullptr;
ID3D11Buffer* g_fits = nullptr;
ID3D11UnorderedAccessView* g_fitsUav = nullptr;
ID3D11ShaderResourceView* g_fitsSrv = nullptr;
ID3D11Buffer* g_fitWork = nullptr; // the fit's partial sums and its solution between the dispatches; ours only
ID3D11UnorderedAccessView* g_fitWorkUav = nullptr;
ID3D11Texture2D* g_composed = nullptr;
ID3D11UnorderedAccessView* g_composedUav = nullptr;
UINT g_composedW = 0, g_composedH = 0;
uint64_t g_fitPresent[2] = { UINT64_MAX, UINT64_MAX };
ID3D11Resource* g_originalMotion = nullptr; // the game's, compared and handed back only
bool g_swapped = false;
constexpr int kFitRing = 4;
ID3D11Buffer* g_fitStaging[kFitRing] = {};
bool g_fitPending[kFitRing] = {};
int g_fitSlotOf[kFitRing] = {};
uint64_t g_fitPresentOf[kFitRing] = {};
uint64_t g_fitReasons[4] = {};
uint64_t g_fitKeptUsed = 0, g_fitStale = 0;
uint64_t g_fitRejectedOf[2] = {}; // by the parity of the Present: one eye locked out, or both with nothing to fit
uint64_t g_fitCount = 0;
// Per window.
uint64_t g_composedEvals = 0, g_chainBreaks = 0, g_fitsRead = 0, g_fitsValid = 0;
double g_fitResidualSum = 0, g_fitInlierSum = 0;
bool g_loggedFirstComposed = false, g_loggedFirstFit = false;
int g_viewMode = 0; // 0 off, 1 object motion, 2 what DLSS gets
// [Inputs] UpscalerGpuTiming. Made and used on the evaluate's thread only; the device is compared, not kept.
std::unique_ptr<GpuTime_Dx11> g_motionTime;
ID3D11Device* g_motionTimeDevice = nullptr;
int g_variant = 1;  // F4, per eye: object motion variant; 1 = x2 with the low knee (user pick, as FC6 found)
const char* const kVariantNames[4] = { "x2, knee 0.1-0.4 px (validated)", "x2, knee 0.02-0.1 px",
                                       "chained over both Presents, knee 0.02-0.1 px",
                                       "chained, knee 0.02-0.1 px, no nearest-surface rule" };
ID3D11Buffer* g_viewConstants = nullptr;
ID3D11ComputeShader* g_strip = nullptr;
ID3D11ComputeShader* g_indicator = nullptr;
ID3D11Buffer* g_indicatorConstants = nullptr;
uint64_t g_indicatorUntil = 0; // Present until which the indicator shows after a key
ID3D11Buffer* g_stripBuffer = nullptr;
ID3D11UnorderedAccessView* g_stripUav = nullptr;
constexpr int kStripRing = 24;
struct StripSlot
{
    ID3D11Buffer* staging = nullptr;
    bool pending = false;
    uint64_t present = 0;
    int parity = 0;
    int viewMode = 0;
    bool perEye = false;
};
StripSlot g_stripSlots[kStripRing];
uint64_t g_stripCount = 0;
ID3D11Texture2D* g_prevGame = nullptr;
ID3D11ShaderResourceView* g_prevGameSrv = nullptr;
uint64_t g_prevGamePresent = UINT64_MAX;
bool g_loggedViewFail = false;
bool g_loggedStatsFail = false;
bool g_loggedFirstView = false;

bool ProbeOn()
{
    static const bool on = Config::Instance()->UpscalerInputProbe.value_or_default();
    return on;
}

ID3D11ComputeShader* Compile(ID3D11Device* device, const char* source, size_t length, const char* name,
                             const char* stage = nullptr)
{
    const D3D_SHADER_MACRO macros[2] = { { stage, "1" }, { nullptr, nullptr } };
    ID3DBlob* code = nullptr;
    ID3DBlob* errors = nullptr;
    if (FAILED(D3DCompile(source, length, name, stage != nullptr ? macros : nullptr, nullptr, "CSMain", "cs_5_0",
                          D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &code, &errors)))
    {
        LOG_ERROR("Per-eye DLSS: {} compile failed: {}", name,
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

// Once, on the first evaluate: shaders, buffers and a spare device context state (our passes run inside it, so the
// game's pipeline state comes back exactly). A failure is final and logged once.
void Setup(ID3D11DeviceContext* ctx)
{
    g_setupTried = true;
    ID3D11Device* device = nullptr;
    ctx->GetDevice(&device);
    if (device == nullptr)
        return;
    g_device = device;
    g_view = Compile(device, kViewShader, sizeof(kViewShader) - 1, "PerEye_View");
    {
        D3D11_BUFFER_DESC vc {};
        vc.ByteWidth = 32;
        vc.Usage = D3D11_USAGE_DYNAMIC;
        vc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        vc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
        if (FAILED(device->CreateBuffer(&vc, nullptr, &g_viewConstants)))
            g_viewConstants = nullptr;
    }
    g_indicator = Compile(device, kIndicatorShader, sizeof(kIndicatorShader) - 1, "PerEye_Indicator");
    {
        D3D11_BUFFER_DESC ic {};
        ic.ByteWidth = 32;
        ic.Usage = D3D11_USAGE_DYNAMIC;
        ic.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        ic.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
        if (FAILED(device->CreateBuffer(&ic, nullptr, &g_indicatorConstants)))
            g_indicatorConstants = nullptr;
        if (g_indicator == nullptr || g_indicatorConstants == nullptr)
            LOG_ERROR("Per-eye DLSS: in-headset indicator unavailable (shader {}, constants {})", g_indicator != nullptr,
                      g_indicatorConstants != nullptr);
    }
    if (MotionMatrixProbe_Dx11::Enabled())
    {
        g_strip = Compile(device, kStripShader, sizeof(kStripShader) - 1, "PerEye_Strip");
        D3D11_BUFFER_DESC sb {};
        sb.ByteWidth = 256 * 256 * 4;
        sb.Usage = D3D11_USAGE_DEFAULT;
        sb.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
        sb.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
        sb.StructureByteStride = 4;
        if (SUCCEEDED(device->CreateBuffer(&sb, nullptr, &g_stripBuffer)) &&
            FAILED(device->CreateUnorderedAccessView(g_stripBuffer, nullptr, &g_stripUav)))
            g_stripUav = nullptr;
        D3D11_BUFFER_DESC st {};
        st.ByteWidth = 256 * 256 * 4;
        st.Usage = D3D11_USAGE_STAGING;
        st.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        for (auto& s : g_stripSlots)
            if (FAILED(device->CreateBuffer(&st, nullptr, &s.staging)))
                s.staging = nullptr;
        if (g_strip == nullptr || g_stripUav == nullptr || g_stripSlots[kStripRing - 1].staging == nullptr)
            LOG_ERROR("Per-eye DLSS: frame strip unavailable (shader {}, buffer {}, staging {})", g_strip != nullptr,
                      g_stripUav != nullptr, g_stripSlots[kStripRing - 1].staging != nullptr);
    }
    g_stats = Compile(device, kStatsShader, sizeof(kStatsShader) - 1, "PerEye_Stats");
    if (MotionMatrixProbe_Dx11::Enabled())
    {
        g_sample = Compile(device, kSampleShader, sizeof(kSampleShader) - 1, "PerEye_Sample");
        D3D11_BUFFER_DESC sb {};
        sb.ByteWidth = 2048 * 16;
        sb.Usage = D3D11_USAGE_DEFAULT;
        sb.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
        sb.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
        sb.StructureByteStride = 16;
        if (SUCCEEDED(device->CreateBuffer(&sb, nullptr, &g_samples)) &&
            FAILED(device->CreateUnorderedAccessView(g_samples, nullptr, &g_samplesUav)))
            g_samplesUav = nullptr;
        D3D11_BUFFER_DESC st {};
        st.ByteWidth = 2048 * 16;
        st.Usage = D3D11_USAGE_STAGING;
        st.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        for (auto& s : g_sampleSlots)
            if (FAILED(device->CreateBuffer(&st, nullptr, &s.staging)))
                s.staging = nullptr;
        if (g_sample == nullptr || g_samplesUav == nullptr || g_sampleSlots[kSampleRing - 1].staging == nullptr)
            LOG_ERROR("Per-eye DLSS: matrix probe sample grid unavailable (shader {}, buffer {}, staging {})",
                      g_sample != nullptr, g_samplesUav != nullptr, g_sampleSlots[kSampleRing - 1].staging != nullptr);
    }

    D3D11_BUFFER_DESC cb {};
    cb.ByteWidth = 16;
    cb.Usage = D3D11_USAGE_DYNAMIC;
    cb.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    cb.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    if (FAILED(device->CreateBuffer(&cb, nullptr, &g_constants)))
        g_constants = nullptr;

    D3D11_BUFFER_DESC sums {};
    sums.ByteWidth = 16;
    sums.Usage = D3D11_USAGE_DEFAULT;
    sums.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
    sums.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS;
    if (SUCCEEDED(device->CreateBuffer(&sums, nullptr, &g_sums)))
    {
        D3D11_UNORDERED_ACCESS_VIEW_DESC uav {};
        uav.Format = DXGI_FORMAT_R32_TYPELESS;
        uav.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
        uav.Buffer.NumElements = 4;
        uav.Buffer.Flags = D3D11_BUFFER_UAV_FLAG_RAW;
        if (FAILED(device->CreateUnorderedAccessView(g_sums, &uav, &g_sumsUav)))
            g_sumsUav = nullptr;
    }
    D3D11_BUFFER_DESC staging {};
    staging.ByteWidth = 16;
    staging.Usage = D3D11_USAGE_STAGING;
    staging.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    bool stagingOk = true;
    for (auto& s : g_staging)
        stagingOk = SUCCEEDED(device->CreateBuffer(&staging, nullptr, &s)) && stagingOk;

    g_fitInit = Compile(device, kFitShader, sizeof(kFitShader) - 1, "PerEye_FitInit", "FIT_INIT");
    g_fitAccumulate = Compile(device, kFitShader, sizeof(kFitShader) - 1, "PerEye_FitAccumulate", "FIT_ACCUMULATE");
    g_fitSolve = Compile(device, kFitShader, sizeof(kFitShader) - 1, "PerEye_FitSolve", "FIT_SOLVE");
    g_compose = Compile(device, kComposeShader, sizeof(kComposeShader) - 1, "PerEye_Compose");
    D3D11_BUFFER_DESC mc {};
    mc.ByteWidth = 48;
    mc.Usage = D3D11_USAGE_DYNAMIC;
    mc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    mc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    if (FAILED(device->CreateBuffer(&mc, nullptr, &g_motionConstants)))
        g_motionConstants = nullptr;
    D3D11_BUFFER_DESC fb {};
    fb.ByteWidth = 15 * 16;
    fb.Usage = D3D11_USAGE_DEFAULT;
    fb.BindFlags = D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;
    fb.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
    fb.StructureByteStride = 16;
    // Zeroed: no slot counts as holding an accepted fit until one is written.
    const float zeros[60] = {};
    D3D11_SUBRESOURCE_DATA fitsInit { zeros, 0, 0 };
    if (SUCCEEDED(device->CreateBuffer(&fb, &fitsInit, &g_fits)))
    {
        if (FAILED(device->CreateUnorderedAccessView(g_fits, nullptr, &g_fitsUav)))
            g_fitsUav = nullptr;
        if (FAILED(device->CreateShaderResourceView(g_fits, nullptr, &g_fitsSrv)))
            g_fitsSrv = nullptr;
    }
    D3D11_BUFFER_DESC fw {};
    fw.ByteWidth = kFitWorkFloats * sizeof(float);
    fw.Usage = D3D11_USAGE_DEFAULT;
    fw.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
    fw.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
    fw.StructureByteStride = sizeof(float);
    if (SUCCEEDED(device->CreateBuffer(&fw, nullptr, &g_fitWork)))
    {
        if (FAILED(device->CreateUnorderedAccessView(g_fitWork, nullptr, &g_fitWorkUav)))
            g_fitWorkUav = nullptr;
    }
    D3D11_BUFFER_DESC fs {};
    fs.ByteWidth = 15 * 16;
    fs.Usage = D3D11_USAGE_STAGING;
    fs.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    bool fitStagingOk = true;
    for (auto& s : g_fitStaging)
        fitStagingOk = SUCCEEDED(device->CreateBuffer(&fs, nullptr, &s)) && fitStagingOk;

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

    g_ready = g_view != nullptr && g_stats != nullptr && g_constants != nullptr && g_sumsUav != nullptr &&
              stagingOk && g_spareState != nullptr && g_viewConstants != nullptr;
    g_motionReady = g_fitInit != nullptr && g_fitAccumulate != nullptr && g_fitSolve != nullptr &&
                    g_compose != nullptr && g_motionConstants != nullptr && g_fitsUav != nullptr &&
                    g_fitsSrv != nullptr && g_fitWorkUav != nullptr && fitStagingOk && g_spareState != nullptr;
    if (!g_motionReady)
        LOG_ERROR("Per-eye DLSS: same-eye motion unavailable (fit {}/{}/{}, compose {}, constants {}, fits {}/{}, "
                  "fit work {}, staging {}, spare state {}); per eye will use the game's vectors",
                  g_fitInit != nullptr, g_fitAccumulate != nullptr, g_fitSolve != nullptr, g_compose != nullptr,
                  g_motionConstants != nullptr, g_fitsUav != nullptr, g_fitsSrv != nullptr, g_fitWorkUav != nullptr,
                  fitStagingOk, g_spareState != nullptr);
    else
        LOG_INFO("Per-eye DLSS: same-eye motion ready, the fit in 11 dispatches (64 groups a round)");
    if (g_ready)
        LOG_INFO("Per-eye DLSS: motion view and statistics ready");
    else
        LOG_ERROR("Per-eye DLSS: setup failed (view {}, stats {}, constants {}, sums {}, staging {}, spare state {}); "
                  "no motion view or statistics this session",
                  g_view != nullptr, g_stats != nullptr, g_constants != nullptr, g_sumsUav != nullptr, stagingOk,
                  g_spareState != nullptr);
}

template <typename Work> void InSpareState(ID3D11DeviceContext* ctx, Work&& work)
{
    ID3D11DeviceContext1* ctx1 = nullptr;
    if (FAILED(ctx->QueryInterface(IID_PPV_ARGS(&ctx1))) || ctx1 == nullptr)
        return;
    ID3DDeviceContextState* gameState = nullptr;
    ctx1->SwapDeviceContextState(g_spareState, &gameState);
    work();
    ID3DDeviceContextState* spare = nullptr;
    ctx1->SwapDeviceContextState(gameState, &spare);
    if (spare != nullptr)
        spare->Release(); // g_spareState keeps its own reference
    if (gameState != nullptr)
        gameState->Release();
    ctx1->Release();
}

ID3D11Resource* GetResource(NVSDK_NGX_Parameter* params, const char* name)
{
    ID3D11Resource* resource = nullptr;
    if (params->Get(name, &resource) != NVSDK_NGX_Result_Success)
        params->Get(name, (void**) &resource);
    return resource;
}

// Reads the oldest statistics slot that the GPU has finished, without waiting.
void CollectStats(ID3D11DeviceContext* ctx)
{
    for (int i = 0; i < kRing; ++i)
    {
        if (g_stagingParity[i] < 0)
            continue;
        D3D11_MAPPED_SUBRESOURCE mapped {};
        if (FAILED(ctx->Map(g_staging[i], 0, D3D11_MAP_READ, D3D11_MAP_FLAG_DO_NOT_WAIT, &mapped)))
            continue;
        int32_t v[4];
        std::memcpy(v, mapped.pData, sizeof(v));
        ctx->Unmap(g_staging[i], 0);
        if (v[3] > 0)
        {
            auto& p = g_parity[g_stagingParity[i]];
            p.sumX += v[0] / 16.0 / v[3];
            p.sumAbsX += v[1] / 16.0 / v[3];
            p.sumAbsY += v[2] / 16.0 / v[3];
            ++p.frames;
        }
        g_stagingParity[i] = -1;
    }
}

// The matrix probe's grids, oldest finished first, without waiting; each goes to the probe's file.
void CollectSamples(ID3D11DeviceContext* ctx)
{
    for (auto& s : g_sampleSlots)
    {
        if (!s.pending || s.staging == nullptr)
            continue;
        D3D11_MAPPED_SUBRESOURCE mapped {};
        if (FAILED(ctx->Map(s.staging, 0, D3D11_MAP_READ, D3D11_MAP_FLAG_DO_NOT_WAIT, &mapped)))
            continue;
        MotionMatrixProbe_Dx11::WriteSamples(s.present, s.parity, (const float*) mapped.pData, s.mvScale[0],
                                             s.mvScale[1], s.jitter[0], s.jitter[1], s.renderW, s.renderH);
        ctx->Unmap(s.staging, 0);
        s.pending = false;
    }
}

// Fit results of earlier Presents, without waiting (for the log only).
void CollectFits(ID3D11DeviceContext* ctx)
{
    for (int i = 0; i < kFitRing; ++i)
    {
        if (!g_fitPending[i])
            continue;
        D3D11_MAPPED_SUBRESOURCE mapped {};
        if (FAILED(ctx->Map(g_fitStaging[i], 0, D3D11_MAP_READ, D3D11_MAP_FLAG_DO_NOT_WAIT, &mapped)))
            continue;
        float f[60];
        std::memcpy(f, mapped.pData, sizeof(f));
        ctx->Unmap(g_fitStaging[i], 0);
        g_fitPending[i] = false;
        const float* m = f + 2 * 20; // this Present's own fit (the scratch slot)
        const float* kept = f + g_fitSlotOf[i] * 20;
        ++g_fitsRead;
        const int reason = std::clamp((int) m[16], 0, 3);
        ++g_fitReasons[reason];
        if (reason != 0)
        {
            ++(kept[17] > 2.0f ? g_fitStale : g_fitKeptUsed);
            ++g_fitRejectedOf[g_fitSlotOf[i] & 1];
        }
        if (m[14] > 0.5f)
        {
            ++g_fitsValid;
            g_fitResidualSum += m[12];
            g_fitInlierSum += m[13];
        }
        if (MotionMatrixProbe_Dx11::Capturing())
            MotionMatrixProbe_Dx11::WriteFit(g_fitPresentOf[i], g_fitSlotOf[i], m, kept[17]);
        if (!g_loggedFirstFit)
        {
            g_loggedFirstFit = true;
            LOG_INFO("Per-eye DLSS: FIRST fit read back: {} -- inlier residual {:.4f} px, {:.1f}% of {} samples within 1 "
                     "px; M rows x ({:.5f} {:.5f} {:.5f} {:.5f}) y ({:.5f} {:.5f} {:.5f} {:.5f}) w ({:.5f} {:.5f} {:.5f} 1)",
                     m[14] > 0.5f ? "VALID" : "REJECTED", m[12], m[13] * 100.0f, (int) m[15], m[0], m[1], m[2], m[3],
                     m[4], m[5], m[6], m[7], m[8], m[9], m[10]);
        }
    }
}

// Our same-eye motion texture at the game's motion size and format (made once per size; our own, never the game's).
bool EnsureComposed(ID3D11Device* device, const D3D11_TEXTURE2D_DESC& motion)
{
    if (g_composed != nullptr && g_prevGame != nullptr && g_composedW == motion.Width && g_composedH == motion.Height)
        return true;
    // Ours; D3D11 keeps them alive for work still in flight.
    if (g_composedUav != nullptr)
        g_composedUav->Release();
    if (g_composed != nullptr)
        g_composed->Release();
    if (g_prevGameSrv != nullptr)
        g_prevGameSrv->Release();
    if (g_prevGame != nullptr)
        g_prevGame->Release();
    g_composed = nullptr;
    g_composedUav = nullptr;
    g_prevGame = nullptr;
    g_prevGameSrv = nullptr;
    g_prevGamePresent = UINT64_MAX;
    D3D11_TEXTURE2D_DESC desc {};
    desc.Width = motion.Width;
    desc.Height = motion.Height;
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = DXGI_FORMAT_R16G16_FLOAT;
    desc.SampleDesc.Count = 1;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
    if (FAILED(device->CreateTexture2D(&desc, nullptr, &g_composed)) ||
        FAILED(device->CreateUnorderedAccessView(g_composed, nullptr, &g_composedUav)))
    {
        if (g_composed != nullptr)
            g_composed->Release();
        g_composed = nullptr;
        g_composedUav = nullptr;
        return false;
    }
    // The previous Present's own vectors (a copy of the game's, same size and format) for the chained object variants.
    D3D11_TEXTURE2D_DESC prev = desc;
    prev.Format = motion.Format;
    prev.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    if (FAILED(device->CreateTexture2D(&prev, nullptr, &g_prevGame)) ||
        FAILED(device->CreateShaderResourceView(g_prevGame, nullptr, &g_prevGameSrv)))
    {
        if (g_prevGame != nullptr)
            g_prevGame->Release();
        g_prevGame = nullptr;
        g_prevGameSrv = nullptr;
        LOG_WARN("Per-eye DLSS: no copy of the previous Present's vectors (format {}); chained variants act as 1",
                 (int) motion.Format);
    }
    g_composedW = motion.Width;
    g_composedH = motion.Height;
    LOG_INFO("Per-eye DLSS: same-eye motion texture {}x{} made", motion.Width, motion.Height);
    return true;
}

// Frame strips finished on the GPU, without waiting; each goes to the probe (a BMP per frame and its brightness).
void CollectStrips(ID3D11DeviceContext* ctx)
{
    for (auto& s : g_stripSlots)
    {
        if (!s.pending || s.staging == nullptr)
            continue;
        D3D11_MAPPED_SUBRESOURCE mapped {};
        if (FAILED(ctx->Map(s.staging, 0, D3D11_MAP_READ, D3D11_MAP_FLAG_DO_NOT_WAIT, &mapped)))
            continue;
        MotionMatrixProbe_Dx11::WriteStrip(s.present, s.parity, s.viewMode, s.perEye, (const uint32_t*) mapped.pData);
        ctx->Unmap(s.staging, 0);
        s.pending = false;
    }
}

// SRV format for a depth texture made typeless.
DXGI_FORMAT DepthSrvFormat(DXGI_FORMAT format)
{
    switch (format)
    {
    case DXGI_FORMAT_R32G8X24_TYPELESS:
    case DXGI_FORMAT_D32_FLOAT_S8X24_UINT:
        return DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS;
    case DXGI_FORMAT_R32_TYPELESS:
    case DXGI_FORMAT_D32_FLOAT:
        return DXGI_FORMAT_R32_FLOAT;
    case DXGI_FORMAT_R24G8_TYPELESS:
    case DXGI_FORMAT_D24_UNORM_S8_UINT:
        return DXGI_FORMAT_R24_UNORM_X8_TYPELESS;
    case DXGI_FORMAT_R16_TYPELESS:
    case DXGI_FORMAT_D16_UNORM:
        return DXGI_FORMAT_R16_UNORM;
    default:
        return format;
    }
}
} // namespace

namespace PerEyeDlss_Dx11
{
bool Enabled()
{
    // Once, before anything reads the routing (every reader asks Enabled() first): [Inputs] UpscalerPerEyeStart
    // starts per eye, so that a run does not depend on a key being remembered.
    static const bool enabled = []
    {
        const bool on = Config::Instance()->UpscalerPerEye.value_or_default();
        if (on && Config::Instance()->UpscalerPerEyeStart.value_or_default())
        {
            g_routing = true;
            LOG_INFO("Per-eye DLSS: routing starts PER EYE ([Inputs] UpscalerPerEyeStart)");
        }
        return on;
    }();
    return enabled;
}

void OnPresent()
{
    if (!Enabled())
        return;
    std::lock_guard lock(g_mutex);
    const uint64_t present = ++g_present;
    if (g_seenEvaluate)
    {
        if (g_evalsThisPresent == 0)
            ++g_presentsWithout;
        else if (g_evalsThisPresent > 1)
            ++g_presentsWithSeveral;
    }
    g_evalsThisPresent = 0;

    if (g_seenEvaluate && present - g_windowStart >= kWindow)
    {
        std::string motion;
        if (ProbeOn())
        {
            for (int i = 0; i < 2; ++i)
            {
                const auto& p = g_parity[i];
                motion += p.frames == 0 ? std::format("; {} Presents: no motion statistics", i ? "odd" : "even")
                                        : std::format("; {} Presents: mean motion x {:+.2f} px, |x| {:.2f}, |y| {:.2f} "
                                                      "({} frames)",
                                                      i ? "odd" : "even", p.sumX / p.frames, p.sumAbsX / p.frames,
                                                      p.sumAbsY / p.frames, p.frames);
            }
        }
        const std::string sameEye =
            g_composedEvals + g_chainBreaks == 0
                ? std::string()
                : std::format("; same-eye motion on {} evaluates, {} chain breaks (history reset); fits read {}, valid "
                              "{}, mean inlier residual {:.4f} px, mean inliers {:.1f}%; rejected: {} solve failed, {} "
                              "too few depth samples, {} too few within 1 px -> {} bridged by the eye's last fit, {} "
                              "past 2 in a row (the game's vectors); rejected on even Presents {}, on odd {}",
                              g_composedEvals, g_chainBreaks, g_fitsRead, g_fitsValid,
                              g_fitsValid ? g_fitResidualSum / g_fitsValid : 0.0,
                              g_fitsValid ? 100.0 * g_fitInlierSum / g_fitsValid : 0.0, g_fitReasons[1],
                              g_fitReasons[2], g_fitReasons[3], g_fitKeptUsed, g_fitStale, g_fitRejectedOf[0],
                              g_fitRejectedOf[1]);
        LOG_INFO("Per-eye DLSS: {} Presents, routing {} (objects: {}): {} evaluates to the game's feature, {} to the "
                 "twin, {} with no twin; {} Presents without an evaluate, {} with several{}{}",
                 present - g_windowStart, g_routing ? "PER EYE" : "shared (game's feature only)", kVariantNames[g_variant],
                 g_toPrimary, g_toTwin,
                 g_noTwin, g_presentsWithout, g_presentsWithSeveral, sameEye, motion);
        // The same window for the overlay's section (GetStatus).
        g_lastWindow.presents = present - g_windowStart;
        g_lastWindow.toGame = g_toPrimary;
        g_lastWindow.toTwin = g_toTwin;
        g_lastWindow.sameEye = g_composedEvals;
        g_lastWindow.chainBreaks = g_chainBreaks;
        g_lastWindow.fitsRead = g_fitsRead;
        g_lastWindow.fitsValid = g_fitsValid;
        g_lastWindow.fitResidual = g_fitsValid ? g_fitResidualSum / g_fitsValid : 0.0;
        g_lastWindow.fitInliers = g_fitsValid ? 100.0 * g_fitInlierSum / g_fitsValid : 0.0;
        g_lastWindow.perEye = g_routing && !g_windowSwitched;
        g_windowSwitched = false;
        g_composedEvals = g_chainBreaks = g_fitsRead = g_fitsValid = 0;
        g_fitReasons[0] = g_fitReasons[1] = g_fitReasons[2] = g_fitReasons[3] = g_fitKeptUsed = g_fitStale = 0;
        g_fitRejectedOf[0] = g_fitRejectedOf[1] = 0;
        g_fitResidualSum = g_fitInlierSum = 0;
        g_windowStart = present;
        g_toPrimary = g_toTwin = g_noTwin = g_presentsWithout = g_presentsWithSeveral = 0;
        g_parity[0] = g_parity[1] = Parity {};
    }
}

void RegisterTwin(unsigned int primary, unsigned int twin)
{
    std::lock_guard lock(g_mutex);
    g_twins[primary] = twin;
    g_resetPrimary = g_resetTwin = true;
    LOG_INFO("Per-eye DLSS: feature {} has twin {} (odd Presents go to the twin while routing is per eye)", primary,
             twin);
}

unsigned int TwinOf(unsigned int primary)
{
    std::lock_guard lock(g_mutex);
    auto it = g_twins.find(primary);
    return it == g_twins.end() ? 0 : it->second;
}

void Forget(unsigned int primary)
{
    std::lock_guard lock(g_mutex);
    g_twins.erase(primary);
}

unsigned int Route(unsigned int handle, NVSDK_NGX_Parameter* params)
{
    if (!Enabled())
        return handle;
    std::lock_guard lock(g_mutex);
    g_seenEvaluate = true;
    ++g_evalsThisPresent;
    auto it = g_twins.find(handle);
    // Neural Rendering runs inside this evaluate with a history of its own: name the eye, so NR keeps one history per
    // eye like the twin features (unnamed, NR shares one history between the eyes -- FC6's "quicksand").
    // Set on every call: the parameters are the game's and keep a value between evaluates. Shared routing = eye 0.
    const bool perEye = it != g_twins.end() && g_routing;
    if (params != nullptr)
        params->Set(DlssNr::kNrEyeParam, perEye ? (unsigned int) (g_present.load() & 1) : 0u);
    if (it == g_twins.end())
    {
        ++g_noTwin;
        return handle;
    }
    const bool toTwin = g_routing && (g_present.load() & 1) != 0;
    bool& reset = toTwin ? g_resetTwin : g_resetPrimary;
    if (reset)
    {
        params->Set(NVSDK_NGX_Parameter_Reset, 1);
        reset = false;
        LOG_INFO("Per-eye DLSS: history reset for the {} (routing {})", toTwin ? "twin" : "game's feature",
                 g_routing ? "per eye" : "shared");
    }
    if (toTwin)
    {
        ++g_toTwin;
        return it->second;
    }
    ++g_toPrimary;
    return handle;
}

void BeforeEvaluate(ID3D11DeviceContext* ctx, NVSDK_NGX_Parameter* params)
{
    g_swapped = false;
    // The fit runs per eye (it feeds the composition) and for the object-motion view (in either routing).
    if (!Enabled() || (!g_routing && g_viewMode != 1) || ctx == nullptr || params == nullptr)
        return;
    if (!g_setupTried)
        Setup(ctx);
    if (!g_motionReady)
        return;
    ID3D11Device* device = nullptr;
    ctx->GetDevice(&device);
    if (device == nullptr)
        return;
    device->Release(); // compared only
    if (device != g_device)
        return;

    ID3D11Resource* motion = GetResource(params, NVSDK_NGX_Parameter_MotionVectors);
    ID3D11Resource* depth = GetResource(params, NVSDK_NGX_Parameter_Depth);
    auto skip = [&](const char* why)
    {
        if (!g_loggedMotionFail)
        {
            g_loggedMotionFail = true;
            LOG_WARN("Per-eye DLSS: same-eye motion skipped ({}); per eye uses the game's vectors", why);
        }
    };
    if (motion == nullptr || depth == nullptr)
        return skip("no motion or no depth in the DLSS parameters");
    D3D11_TEXTURE2D_DESC md {}, dd {};
    ID3D11Texture2D* tex = nullptr;
    if (FAILED(motion->QueryInterface(IID_PPV_ARGS(&tex))) || tex == nullptr)
        return skip("the motion vectors are not a Texture2D");
    tex->GetDesc(&md);
    tex->Release();
    if (FAILED(depth->QueryInterface(IID_PPV_ARGS(&tex))) || tex == nullptr)
        return skip("the depth is not a Texture2D");
    tex->GetDesc(&dd);
    tex->Release();
    if (!EnsureComposed(device, md))
        return skip("could not make the same-eye motion texture");

    float scale[2] = { 0, 0 };
    params->Get(NVSDK_NGX_Parameter_MV_Scale_X, &scale[0]);
    params->Get(NVSDK_NGX_Parameter_MV_Scale_Y, &scale[1]);
    if (scale[0] == 0 || scale[1] == 0)
        return skip("MV scale is zero");

    const uint64_t present = g_present.load();
    const uint32_t cur = (uint32_t) (present & 1), prev = cur ^ 1;
    const bool chain = g_routing && g_fitPresent[prev] == present - 1;
    // Read once: the constants and the copy below must agree on it.
    const int variant = g_variant;
    // The copy of this Present's vectors is read only by the variants from 2 on. Made only for them, and its
    // Present noted only when it was made: after a switch into such a variant the first Present has no previous
    // vectors (HavePrev 0), never a stale copy.
    const bool keepPrev = g_routing && g_prevGame != nullptr && variant >= 2;
    const bool havePrev = g_prevGameSrv != nullptr && g_prevGamePresent == present - 1;

    struct
    {
        float mvScale[2];
        uint32_t cur, prev;
        float knee0, knee1, gain;
        uint32_t radius, variant, havePrev;
        float unused[2];
    } constants { { scale[0], scale[1] }, cur, prev, 0.1f, 0.4f, 2.0f, 2, (uint32_t) variant, havePrev ? 1u : 0u,
                  { 0, 0 } };
    D3D11_MAPPED_SUBRESOURCE mapped {};
    if (FAILED(ctx->Map(g_motionConstants, 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped)))
        return skip("could not write the constants");
    std::memcpy(mapped.pData, &constants, sizeof(constants));
    ctx->Unmap(g_motionConstants, 0);

    ID3D11ShaderResourceView* motionSrv = nullptr;
    ID3D11ShaderResourceView* depthSrv = nullptr;
    device->CreateShaderResourceView(motion, nullptr, &motionSrv);
    D3D11_SHADER_RESOURCE_VIEW_DESC dsd {};
    dsd.Format = DepthSrvFormat(dd.Format);
    dsd.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
    dsd.Texture2D.MipLevels = 1;
    device->CreateShaderResourceView(depth, &dsd, &depthSrv);
    if (motionSrv == nullptr || depthSrv == nullptr)
    {
        if (motionSrv != nullptr)
            motionSrv->Release();
        if (depthSrv != nullptr)
            depthSrv->Release();
        return skip("no view on the motion vectors or the depth");
    }

    // [Inputs] UpscalerGpuTiming (live): the fit, the composed motion and the two copies, by D3D11 timestamps. What
    // they read is said with its floor (the composed texture and the copy of the vectors are written: 8 bytes a
    // pixel), because D3D11 brackets have read next to nothing for full-eye copies before.
    const bool timing = Config::Instance()->UpscalerGpuTiming.value_or_default();
    if (timing)
    {
        if (g_motionTime == nullptr || g_motionTimeDevice != device)
        {
            g_motionTime = std::make_unique<GpuTime_Dx11>(device);
            g_motionTimeDevice = device;
        }
        if (const auto took = g_motionTime->ReadGpuTime(ctx); took.has_value())
            StageTimes::Global().Add(StageTimes::SameEyeMotion, took.value(),
                                     StageTimes::FloorMs((double) md.Width * (double) md.Height * 8.0));
        g_motionTime->Start(ctx);
    }
    InSpareState(ctx,
                 [&]
                 {
                     ID3D11ShaderResourceView* srvs[3] = { depthSrv, motionSrv, nullptr };
                     ID3D11UnorderedAccessView* nullUav = nullptr;
                     ID3D11UnorderedAccessView* fitUavs[2] = { g_fitsUav, g_fitWorkUav };
                     ID3D11UnorderedAccessView* nullUavs[2] = {};
                     ctx->CSSetConstantBuffers(0, 1, &g_motionConstants);
                     ctx->CSSetShaderResources(0, 2, srvs);
                     ctx->CSSetUnorderedAccessViews(0, 2, fitUavs, nullptr);
                     DispatchFit(ctx, g_fitInit, g_fitAccumulate, g_fitSolve);
                     ctx->CSSetUnorderedAccessViews(0, 2, nullUavs, nullptr);
                     if (chain)
                     {
                         ID3D11ShaderResourceView* composeSrvs[4] = { depthSrv, motionSrv, g_fitsSrv, g_prevGameSrv };
                         ctx->CSSetShader(g_compose, nullptr, 0);
                         ctx->CSSetShaderResources(0, 4, composeSrvs);
                         ctx->CSSetUnorderedAccessViews(0, 1, &g_composedUav, nullptr);
                         ctx->Dispatch((md.Width + 7) / 8, (md.Height + 7) / 8, 1);
                         ctx->CSSetUnorderedAccessViews(0, 1, &nullUav, nullptr);
                     }
                     ID3D11ShaderResourceView* nullSrvs[4] = {};
                     ctx->CSSetShaderResources(0, 4, nullSrvs);
                     // This Present's own vectors, for the next Present's chained object variants (a copy; nothing
                     // of the game's is kept).
                     if (keepPrev)
                         ctx->CopyResource(g_prevGame, motion);
                     const int slot = (int) (g_fitCount++ % kFitRing);
                     if (!g_fitPending[slot])
                     {
                         ctx->CopyResource(g_fitStaging[slot], g_fits);
                         g_fitPending[slot] = true;
                         g_fitSlotOf[slot] = (int) cur;
                         g_fitPresentOf[slot] = MotionMatrixProbe_Dx11::Present();
                     }
                 });
    if (timing && g_motionTime != nullptr)
        g_motionTime->End(ctx);
    motionSrv->Release();
    depthSrv->Release();
    g_fitPresent[cur] = present;
    if (keepPrev)
        g_prevGamePresent = present;

    std::lock_guard lock(g_mutex);
    if (!g_routing)
        return; // the object view's fit only; DLSS keeps the game's vectors
    if (chain)
    {
        g_originalMotion = motion;
        params->Set(NVSDK_NGX_Parameter_MotionVectors, (ID3D11Resource*) g_composed);
        g_swapped = true;
        ++g_composedEvals;
        if (!g_loggedFirstComposed)
        {
            g_loggedFirstComposed = true;
            LOG_INFO("Per-eye DLSS: FIRST same-eye motion handed to DLSS ({}x{}, present {}, fits of presents {} and {})",
                     md.Width, md.Height, present, present, present - 1);
        }
    }
    else
    {
        // No fit of the previous Present: nothing to chain, so this eye's history starts over rather than being fed
        // the other eye's vectors.
        params->Set(NVSDK_NGX_Parameter_Reset, 1);
        ++g_chainBreaks;
    }
}

void EndEvaluate(NVSDK_NGX_Parameter* params)
{
    if (g_swapped && params != nullptr)
        params->Set(NVSDK_NGX_Parameter_MotionVectors, g_originalMotion);
    g_swapped = false;
}

void AfterEvaluate(ID3D11DeviceContext* ctx, NVSDK_NGX_Parameter* params)
{
    if (!Enabled() || ctx == nullptr || params == nullptr)
        return;
    if (MotionMatrixProbe_Dx11::Enabled())
        MotionMatrixProbe_Dx11::SetMotionTexture(g_swapped ? g_originalMotion
                                                           : GetResource(params, NVSDK_NGX_Parameter_MotionVectors));
    const bool view = g_viewMode != 0;
    const bool indicator = view || g_present.load() < g_indicatorUntil;
    const bool stats = ProbeOn();
    const bool sample = MotionMatrixProbe_Dx11::Capturing();
    const bool sampling = MotionMatrixProbe_Dx11::Enabled();
    // Grids finished since the last evaluate go to the probe's file (also after its capture window has closed).
    if (sampling && g_ready)
    {
        CollectSamples(ctx);
        CollectStrips(ctx);
    }
    if (g_motionReady)
    {
        std::lock_guard lock(g_mutex);
        CollectFits(ctx);
    }
    if (!view && !stats && !sample && !indicator)
        return;
    if (!g_setupTried)
        Setup(ctx);
    if (!g_ready)
        return;

    ID3D11Device* device = nullptr;
    ctx->GetDevice(&device);
    if (device == nullptr)
        return;
    device->Release(); // compared only
    if (device != g_device)
        return; // another device's evaluate; our objects belong to the first

    float scale[4] = { 0, 0, 0, 0 };
    params->Get(NVSDK_NGX_Parameter_MV_Scale_X, &scale[0]);
    params->Get(NVSDK_NGX_Parameter_MV_Scale_Y, &scale[1]);

    ID3D11Resource* motion = GetResource(params, NVSDK_NGX_Parameter_MotionVectors);
    ID3D11Resource* colour = GetResource(params, NVSDK_NGX_Parameter_Color);
    ID3D11Resource* output = GetResource(params, NVSDK_NGX_Parameter_Output);
    ID3D11Resource* depth = GetResource(params, NVSDK_NGX_Parameter_Depth);
    if (motion == nullptr)
        return;

    // Views are made per evaluate and released straight after: no reference to the game's textures outlives this.
    ID3D11ShaderResourceView* motionSrv = nullptr;
    if (FAILED(device->CreateShaderResourceView(motion, nullptr, &motionSrv)) || motionSrv == nullptr)
    {
        if (!g_loggedStatsFail)
        {
            g_loggedStatsFail = true;
            LOG_WARN("Per-eye DLSS: no view on the motion vectors; motion view and statistics skipped");
        }
        return;
    }

    D3D11_MAPPED_SUBRESOURCE mapped {};
    if (SUCCEEDED(ctx->Map(g_constants, 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped)))
    {
        std::memcpy(mapped.pData, scale, sizeof(scale));
        ctx->Unmap(g_constants, 0);
    }

    const int parity = (int) (g_present.load() & 1);
    InSpareState(ctx,
                 [&]
                 {
                     if (sample && g_sample != nullptr && g_samplesUav != nullptr && depth != nullptr)
                     {
                         SampleSlot& s = g_sampleSlots[g_sampleCount % kSampleRing];
                         ID3D11Texture2D* depthTexture = nullptr;
                         D3D11_TEXTURE2D_DESC dd {};
                         if (SUCCEEDED(depth->QueryInterface(IID_PPV_ARGS(&depthTexture))) && depthTexture != nullptr)
                         {
                             depthTexture->GetDesc(&dd);
                             depthTexture->Release();
                         }
                         D3D11_SHADER_RESOURCE_VIEW_DESC dsd {};
                         dsd.Format = DepthSrvFormat(dd.Format);
                         dsd.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
                         dsd.Texture2D.MipLevels = 1;
                         ID3D11ShaderResourceView* depthSrv = nullptr;
                         device->CreateShaderResourceView(depth, &dsd, &depthSrv);
                         // The game's vectors (put back only after this pass, so taken from the saved pointer) and
                         // what DLSS got.
                         ID3D11ShaderResourceView* gameSrv = nullptr;
                         if (g_swapped && g_originalMotion != nullptr)
                             device->CreateShaderResourceView(g_originalMotion, nullptr, &gameSrv);
                         if (depthSrv != nullptr && !s.pending && s.staging != nullptr)
                         {
                             ID3D11ShaderResourceView* srvs[3] = { depthSrv, gameSrv != nullptr ? gameSrv : motionSrv,
                                                                   motionSrv };
                             ctx->CSSetShader(g_sample, nullptr, 0);
                             ctx->CSSetShaderResources(0, 3, srvs);
                             ctx->CSSetUnorderedAccessViews(0, 1, &g_samplesUav, nullptr);
                             ctx->CSSetConstantBuffers(0, 1, &g_constants);
                             ctx->Dispatch(4, 4, 1);
                             ID3D11ShaderResourceView* nullSrvs[3] = {};
                             ID3D11UnorderedAccessView* nullUav = nullptr;
                             ctx->CSSetShaderResources(0, 3, nullSrvs);
                             ctx->CSSetUnorderedAccessViews(0, 1, &nullUav, nullptr);
                             ctx->CopyResource(s.staging, g_samples);
                             s.pending = true;
                             s.present = MotionMatrixProbe_Dx11::Present();
                             s.parity = (int) (s.present & 1);
                             s.mvScale[0] = scale[0];
                             s.mvScale[1] = scale[1];
                             params->Get(NVSDK_NGX_Parameter_Jitter_Offset_X, &s.jitter[0]);
                             params->Get(NVSDK_NGX_Parameter_Jitter_Offset_Y, &s.jitter[1]);
                             params->Get(NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Width, &s.renderW);
                             params->Get(NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Height, &s.renderH);
                             ++g_sampleCount;
                         }
                         else if (depthSrv == nullptr && !g_loggedSampleFail)
                         {
                             g_loggedSampleFail = true;
                             LOG_WARN("Per-eye DLSS: matrix probe: no view on the depth (format {}); no sample grid",
                                      (int) dd.Format);
                         }
                         if (depthSrv != nullptr)
                             depthSrv->Release();
                         if (gameSrv != nullptr)
                             gameSrv->Release();
                     }

                     if (stats)
                     {
                         const UINT zeros[4] = {};
                         ctx->ClearUnorderedAccessViewUint(g_sumsUav, zeros);
                         ctx->CSSetShader(g_stats, nullptr, 0);
                         ctx->CSSetShaderResources(0, 1, &motionSrv);
                         ctx->CSSetUnorderedAccessViews(0, 1, &g_sumsUav, nullptr);
                         ctx->CSSetConstantBuffers(0, 1, &g_constants);
                         ctx->Dispatch(8, 8, 1);
                         ID3D11UnorderedAccessView* nullUav = nullptr;
                         ctx->CSSetUnorderedAccessViews(0, 1, &nullUav, nullptr);
                         const int slot = (int) (g_statsCount++ % kRing);
                         ctx->CopyResource(g_staging[slot], g_sums);
                         g_stagingParity[slot] = parity;
                     }

                     if (view && colour != nullptr && output != nullptr)
                     {
                         ID3D11ShaderResourceView* colourSrv = nullptr;
                         ID3D11UnorderedAccessView* outputUav = nullptr;
                         ID3D11ShaderResourceView* gameView = nullptr;
                         ID3D11ShaderResourceView* depthView = nullptr;
                         device->CreateShaderResourceView(colour, nullptr, &colourSrv);
                         device->CreateUnorderedAccessView(output, nullptr, &outputUav);
                         if (g_swapped && g_originalMotion != nullptr)
                             device->CreateShaderResourceView(g_originalMotion, nullptr, &gameView);
                         if (depth != nullptr)
                         {
                             ID3D11Texture2D* dt = nullptr;
                             D3D11_TEXTURE2D_DESC ddesc {};
                             if (SUCCEEDED(depth->QueryInterface(IID_PPV_ARGS(&dt))) && dt != nullptr)
                             {
                                 dt->GetDesc(&ddesc);
                                 dt->Release();
                             }
                             D3D11_SHADER_RESOURCE_VIEW_DESC vd {};
                             vd.Format = DepthSrvFormat(ddesc.Format);
                             vd.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
                             vd.Texture2D.MipLevels = 1;
                             device->CreateShaderResourceView(depth, &vd, &depthView);
                         }
                         const uint64_t present = g_present.load();
                         struct
                         {
                             float mvScale[2];
                             uint32_t mode, cur;
                             float fullPx;
                             uint32_t haveFit;
                             float unused[2];
                         } vc { { scale[0], scale[1] }, (uint32_t) g_viewMode, (uint32_t) (present & 1),
                                g_viewMode == 1 ? 4.0f : 128.0f,
                                (g_motionReady && depthView != nullptr && g_fitPresent[present & 1] == present) ? 1u : 0u,
                                { 0, 0 } };
                         D3D11_MAPPED_SUBRESOURCE vm {};
                         if (SUCCEEDED(ctx->Map(g_viewConstants, 0, D3D11_MAP_WRITE_DISCARD, 0, &vm)))
                         {
                             std::memcpy(vm.pData, &vc, sizeof(vc));
                             ctx->Unmap(g_viewConstants, 0);
                         }
                         if (colourSrv != nullptr && outputUav != nullptr)
                         {
                             ID3D11ShaderResourceView* srvs[5] = { colourSrv, motionSrv,
                                                                   gameView != nullptr ? gameView : motionSrv, depthView,
                                                                   g_fitsSrv };
                             ctx->CSSetShader(g_view, nullptr, 0);
                             ctx->CSSetShaderResources(0, 5, srvs);
                             ctx->CSSetUnorderedAccessViews(0, 1, &outputUav, nullptr);
                             ctx->CSSetConstantBuffers(0, 1, &g_viewConstants);
                             D3D11_TEXTURE2D_DESC desc {};
                             ID3D11Texture2D* texture = nullptr;
                             if (SUCCEEDED(output->QueryInterface(IID_PPV_ARGS(&texture))) && texture != nullptr)
                             {
                                 texture->GetDesc(&desc);
                                 texture->Release();
                             }
                             ctx->Dispatch((desc.Width + 7) / 8, (desc.Height + 7) / 8, 1);
                             ID3D11ShaderResourceView* nullSrvs[5] = {};
                             ID3D11UnorderedAccessView* nullUav = nullptr;
                             ctx->CSSetShaderResources(0, 5, nullSrvs);
                             ctx->CSSetUnorderedAccessViews(0, 1, &nullUav, nullptr);
                             if (!g_loggedFirstView)
                             {
                                 g_loggedFirstView = true;
                                 LOG_INFO("Per-eye DLSS: motion view drawn over the DLSS output ({}x{}), MV scale "
                                          "({}, {})",
                                          desc.Width, desc.Height, scale[0], scale[1]);
                             }
                         }
                         else if (!g_loggedViewFail)
                         {
                             g_loggedViewFail = true;
                             LOG_WARN("Per-eye DLSS: motion view skipped: no view on the colour ({}) or the output "
                                      "({})",
                                      colourSrv != nullptr, outputUav != nullptr);
                         }
                         if (colourSrv != nullptr)
                             colourSrv->Release();
                         if (outputUav != nullptr)
                             outputUav->Release();
                         if (gameView != nullptr)
                             gameView->Release();
                         if (depthView != nullptr)
                             depthView->Release();
                     }

                     if (indicator && output != nullptr && g_indicator != nullptr && g_indicatorConstants != nullptr)
                     {
                         ID3D11UnorderedAccessView* outUav = nullptr;
                         device->CreateUnorderedAccessView(output, nullptr, &outUav);
                         ID3D11Texture2D* ot = nullptr;
                         D3D11_TEXTURE2D_DESC od {};
                         if (SUCCEEDED(output->QueryInterface(IID_PPV_ARGS(&ot))) && ot != nullptr)
                         {
                             ot->GetDesc(&od);
                             ot->Release();
                         }
                         const uint32_t cell = std::max<uint32_t>(od.Width / 90, 8);
                         if (outUav != nullptr && od.Width >= cell * 8 && od.Height >= cell)
                         {
                             struct
                             {
                                 uint32_t origin[2];
                                 uint32_t cell, routing, variant, view;
                                 uint32_t unused[2];
                             } ic { { od.Width * 40 / 100, od.Height * 58 / 100 }, cell, g_routing ? 1u : 0u,
                                    (uint32_t) g_variant, (uint32_t) g_viewMode, { 0, 0 } };
                             D3D11_MAPPED_SUBRESOURCE im {};
                             if (SUCCEEDED(ctx->Map(g_indicatorConstants, 0, D3D11_MAP_WRITE_DISCARD, 0, &im)))
                             {
                                 std::memcpy(im.pData, &ic, sizeof(ic));
                                 ctx->Unmap(g_indicatorConstants, 0);
                             }
                             ctx->CSSetShader(g_indicator, nullptr, 0);
                             ctx->CSSetUnorderedAccessViews(0, 1, &outUav, nullptr);
                             ctx->CSSetConstantBuffers(0, 1, &g_indicatorConstants);
                             ctx->Dispatch((cell * 8 + 7) / 8, (cell + 7) / 8, 1);
                             ID3D11UnorderedAccessView* nullUav = nullptr;
                             ctx->CSSetUnorderedAccessViews(0, 1, &nullUav, nullptr);
                         }
                         if (outUav != nullptr)
                             outUav->Release();
                     }

                     // F7 frame strip: what DLSS's output now holds (with the view, if on), shrunk, per eye.
                     if (sample && g_strip != nullptr && g_stripUav != nullptr && output != nullptr)
                     {
                         StripSlot& s = g_stripSlots[g_stripCount % kStripRing];
                         ID3D11ShaderResourceView* outSrv = nullptr;
                         device->CreateShaderResourceView(output, nullptr, &outSrv);
                         if (outSrv != nullptr && !s.pending && s.staging != nullptr)
                         {
                             ctx->CSSetShader(g_strip, nullptr, 0);
                             ctx->CSSetShaderResources(0, 1, &outSrv);
                             ctx->CSSetUnorderedAccessViews(0, 1, &g_stripUav, nullptr);
                             ctx->Dispatch(32, 32, 1);
                             ID3D11ShaderResourceView* nullSrv = nullptr;
                             ID3D11UnorderedAccessView* nullUav = nullptr;
                             ctx->CSSetShaderResources(0, 1, &nullSrv);
                             ctx->CSSetUnorderedAccessViews(0, 1, &nullUav, nullptr);
                             ctx->CopyResource(s.staging, g_stripBuffer);
                             s.pending = true;
                             s.present = MotionMatrixProbe_Dx11::Present();
                             s.parity = (int) (s.present & 1);
                             s.viewMode = g_viewMode;
                             s.perEye = g_routing;
                             ++g_stripCount;
                         }
                         if (outSrv != nullptr)
                             outSrv->Release();
                     }
                 });
    motionSrv->Release();

    if (stats)
    {
        std::lock_guard lock(g_mutex);
        CollectStats(ctx);
    }
}

void ToggleRouting()
{
    std::lock_guard lock(g_mutex);
    g_routing = !g_routing;
    g_windowSwitched = true;
    g_resetPrimary = g_resetTwin = true;
    g_indicatorUntil = g_present.load() + 270;
    LOG_INFO("Per-eye DLSS: F6, routing now {}", g_routing ? "PER EYE" : "shared (game's feature only)");
}

bool RoutingOn() { return g_routing; }

void ToggleView()
{
    g_viewMode = (g_viewMode + 1) % 3;
    g_indicatorUntil = g_present.load() + 270;
    LOG_INFO("Per-eye DLSS: F8, view now {}", ViewName());
}

bool ViewOn() { return g_viewMode != 0; }

const char* ViewName()
{
    return g_viewMode == 1 ? "OBJECT MOTION (camera removed, full at 4 px)"
         : g_viewMode == 2 ? "WHAT DLSS GETS (full at 128 px)"
                           : "off";
}

void CycleVariant()
{
    g_variant = (g_variant + 1) % 4;
    g_indicatorUntil = g_present.load() + 270;
    LOG_INFO("Per-eye DLSS: F4, object motion now {}", kVariantNames[g_variant]);
}

const char* VariantName() { return kVariantNames[g_variant]; }

// The overlay's section: the same state the keys change, by value.
void SetView(int mode)
{
    if (mode < 0 || mode > 2 || mode == g_viewMode)
        return;
    g_viewMode = mode;
    g_indicatorUntil = g_present.load() + 270;
    LOG_INFO("Per-eye DLSS: overlay, view now {}", ViewName());
}

void SetVariant(int variant)
{
    if (variant < 0 || variant > 3 || variant == g_variant)
        return;
    g_variant = variant;
    g_indicatorUntil = g_present.load() + 270;
    LOG_INFO("Per-eye DLSS: overlay, object motion now {}", kVariantNames[g_variant]);
}

PerEyeMenu::Status GetStatus()
{
    PerEyeMenu::Status status;
    status.startPerEye = Config::Instance()->UpscalerPerEyeStart.value_or_default();
    for (int i = 0; i < PerEyeMenu::kVariants; ++i)
        status.variantNames[i] = kVariantNames[i];
    status.viewNames[0] = "Off";
    status.viewNames[1] = "Object motion (camera removed, full at 4 px)";
    status.viewNames[2] = "What DLSS gets (full at 128 px)";
    std::lock_guard lock(g_mutex);
    status.twin = !g_twins.empty();
    status.routing = g_routing;
    status.variant = g_variant;
    status.view = g_viewMode;
    status.windowPresents = g_lastWindow.presents;
    status.toGame = g_lastWindow.toGame;
    status.toTwin = g_lastWindow.toTwin;
    status.sameEye = g_lastWindow.sameEye;
    status.chainBreaks = g_lastWindow.chainBreaks;
    status.fitsRead = g_lastWindow.fitsRead;
    status.fitsValid = g_lastWindow.fitsValid;
    status.fitResidual = g_lastWindow.fitResidual;
    status.fitInliers = g_lastWindow.fitInliers;
    status.windowPerEye = g_lastWindow.perEye;
    status.timing = Config::Instance()->UpscalerGpuTiming.value_or_default();
    status.times = StageTimes::Global().Last();
    return status;
}
} // namespace PerEyeDlss_Dx11
