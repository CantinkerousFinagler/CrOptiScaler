#include "pch.h"

#include "Config.h"
#include "Logger.h"
#include "NVNGX_Parameter.h"
#include <dlssnr/DlssNr.h>
#include "State.h"

#include "PerEyeDlss_Dx12.h"
#include "VarjoGaze.h"
#include "FoveaCopyRule.h"
#include "StaleEyeShare.h"
#include "proxies/NVNGX_Proxy.h"

#include <shaders/Shader_Common.h>

#include <d3d12.h>
#include <dxgi1_4.h>

#include "Util.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <thread>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace
{
// The game's motion vectors over a 64x64 grid, in render px: mean x, mean |x|, mean |y|, and the stamp of the evaluate
// that wrote the slot. One group; each thread walks one column of the grid, then a tree sum.
constexpr char kStatsShader[] = R"(
Texture2D<float2> Motion : register(t0);
RWByteAddressBuffer Out : register(u0);

cbuffer Constants : register(b0)
{
    float2 MvScale;
    uint Slot;
    uint Stamp;
};

groupshared float3 Sums[64];

[numthreads(64, 1, 1)]
void CSMain(uint gi : SV_GroupIndex)
{
    uint w, h;
    Motion.GetDimensions(w, h);
    float3 acc = 0;
    for (uint j = 0; j < 64; ++j)
    {
        const uint2 at = min(uint2((float2(gi, j) + 0.5) / 64.0 * float2(w, h)), uint2(w - 1, h - 1));
        const float2 mv = Motion[at] * MvScale;
        acc += float3(mv.x, abs(mv.x), abs(mv.y));
    }
    Sums[gi] = acc;
    GroupMemoryBarrierWithGroupSync();
    for (uint k = 32; k > 0; k >>= 1)
    {
        if (gi < k)
            Sums[gi] += Sums[gi + k];
        GroupMemoryBarrierWithGroupSync();
    }
    if (gi == 0)
    {
        const uint o = Slot * 16;
        Out.Store3(o, asuint(Sums[0] / 4096.0));
        Out.Store(o + 12, Stamp);
    }
}
)";

// Evaluates arrive on the game's recording threads, Presents and keys on others: all state below is under g_mutex.
std::mutex g_mutex;
std::atomic<uint64_t> g_present { 0 };

std::unordered_map<unsigned int, unsigned int> g_twins;
bool g_routing = false; // starts as the game made it; F6 for per eye -- the same habit as FC5, FC6, Days Gone
bool g_resetPrimary = false;
bool g_resetTwin = false;

bool g_seenEvaluate = false;
uint64_t g_windowStart = 0;
constexpr uint64_t kWindow = 600;
unsigned int g_evalsThisPresent = 0;
uint64_t g_toPrimary = 0, g_toTwin = 0, g_noTwin = 0, g_presentsWithout = 0, g_presentsWithSeveral = 0;

// The statistics pass: our own root signature, pipeline, heap and buffers. Never a game resource beyond one SRV.
constexpr uint32_t kRing = 8;
bool g_setupTried = false;
bool g_ready = false;
ID3D12Device* g_device = nullptr; // compared only; the resources below hold what they need
ID3D12RootSignature* g_rootSignature = nullptr;
ID3D12PipelineState* g_pipeline = nullptr;
ID3D12DescriptorHeap* g_heap = nullptr;
UINT g_descriptorSize = 0;
ID3D12Resource* g_sums = nullptr;     // default heap, UAV; buffers decay to COMMON between lists
ID3D12Resource* g_readback = nullptr; // readback heap, persistently mapped
const uint8_t* g_readbackData = nullptr;
uint64_t g_statsSeq = 0;
int g_slotParity[kRing] = {};

struct Parity
{
    double sumX = 0, sumAbsX = 0, sumAbsY = 0;
    uint64_t frames = 0;
};
Parity g_parity[2];
uint64_t g_statsNotReady = 0;

// Skip reasons, each logged once.
bool g_loggedNrEyeByParity = false;
bool g_loggedSetupFail = false, g_loggedNoMotion = false, g_loggedOtherDevice = false,
     g_loggedFirstStats = false, g_loggedNotReady = false;

// ---- Same-eye motion from the game's own vectors (the D3D12 port of PerEyeDlss_Dx11's FIT and COMPOSE) ----
//
// What the compose adds beyond the fitted camera motion (UpscalerSameEyeObjectMotion; live, read at each compose):
// 0 nothing (gain 0), 1 the game's residual x2 through FC5's validated 0.1-0.4 px knee, 2 the same through the
// 0.02-0.1 px knee (as it used to be). In one D3D12 game the fit's mean inlier residual is ~0.09 px, inside mode 2's
// knee, so mode 2 doubled the fit's own noise everywhere and the picture warped in places; mode 0 was clean.
// 3 (for the smear of small animations and distant moving things seen under 1): the residual
// averaged over 3x3 taps in the shader, knee 0.03-0.15 px (the fit's ~0.09 px noise averages to ~0.03). Radius's high
// byte carries the mode to the shader (the 48-byte constant block has no spare slot).
int ObjectMotionMode() { return std::clamp(Config::Instance()->UpscalerSameEyeObjectMotion.value_or_default(), 0, 3); }
float ObjectGain() { return ObjectMotionMode() == 0 ? 0.0f : 2.0f; }
float ObjectKnee0() { return ObjectMotionMode() == 1 ? 0.1f : ObjectMotionMode() == 3 ? 0.03f : 0.02f; }
float ObjectKnee1() { return ObjectMotionMode() == 1 ? 0.4f : ObjectMotionMode() == 3 ? 0.15f : 0.1f; }
int BiasMaskMode() { return std::clamp(Config::Instance()->UpscalerSameEyeBiasMask.value_or_default(), 0, 2); }
// Radius's bytes: 0 the dilation radius, 1 the object-motion mode, 2 the bias-mask mode.
uint32_t ObjectRadiusWord() { return 2u | ((uint32_t) ObjectMotionMode() << 8) | ((uint32_t) BiasMaskMode() << 16); }
const char* BiasMaskName()
{
    switch (BiasMaskMode())
    {
    case 1:
        return "moving things (point residual, 0.1-0.4 px)";
    case 2:
        return "small movements (3x3 residual, 0.03-0.15 px)";
    default:
        return "off";
    }
}
const char* ObjectMotionName()
{
    switch (ObjectMotionMode())
    {
    case 1:
        return "camera + moving things (the game's object motion x2 through the 0.1-0.4 px knee)";
    case 2:
        return "camera + everything (the game's object motion x2 through the 0.02-0.1 px knee, as it used to be)";
    case 3:
        return "camera + small movements (the game's object motion averaged 3x3, x2 through the 0.03-0.15 px knee)";
    default:
        return "camera motion only";
    }
}
//
// The game's vectors point at the previous Present, the other eye (Crimson Desert: -10..-21 px on even Presents,
// +10..+22 on odd). Each frame's static motion is one projective map M_N: clip (x, y, z, 1) -> previous
// clip. FIT solves M_N from a 64x64 grid of depth and motion (DLT with c.w = 1, robust rounds from this eye's last
// accepted fit, a gentle pull toward it); COMPOSE chains M_{N-1} o M_N for the same-eye camera motion (two Presents
// back) and adds the object motion the matrix cannot see (x2 through the 0.02-0.1 px knee: the D3D11 path's validated variant
// 1). Both work in render px with the DLSS convention (previous - current); z is scaled by 100 in the fit.
// Non-finite samples are rejected by their bits: Crimson Desert hands DLSS NaN / inf vectors in some frames, and the
// fork builds with fast math, where a comparison is no NaN test.
// The fit runs wide instead of in one group of 64 threads (which measured ~1 ms in Crimson Desert, a visible drop in
// frame rate), one dispatch per round: 64 groups each weigh their samples (one per thread) against the current
// solution and write one partial set of normal equations; the LAST group to finish (a counter after the partials)
// sums them and solves. INIT (Round 100, one group) seeds the solution from this eye's last accepted fit. Four
// weighted rounds, then a measuring round (Round 4) whose solve writes the verdict. Same maths as PerEyeDlss_Dx11's
// fit; accumulate and solve used to be separate dispatches (11 per fit, ~0.083 ms, mostly waits between them).
// Fits layout (float4): slot 0 / 1 = parity 0 / 1 (rows x, y, w, stats, (reason, streak, -, -)), slot 2 = this
// Present's scratch fit (rows, stats, (reason, -, -, stamp)), slot 3 = the start (rows, -, (-, have, -, -)).
#include "PerEyeFitShader_Dx12.h"


// Same-eye motion state (ours). The fits buffer holds 4 slots of 5 float4: parity 0, parity 1, this Present's
// scratch fit, and the start; a default-heap committed buffer starts zeroed = "no accepted fit yet".
bool g_motionTried = false;
bool g_motionReady = false;
ID3D12RootSignature* g_motionRoot = nullptr;
ID3D12PipelineState* g_fitPso = nullptr;
ID3D12PipelineState* g_composePso = nullptr;
ID3D12DescriptorHeap* g_motionHeap = nullptr; // kRing x (depth SRV, motion SRV, composed UAV, mask UAV)
ID3D12Resource* g_fits = nullptr;             // UAV; buffers decay to COMMON between lists, we always leave it UAV
ID3D12Resource* g_partials = nullptr;         // 64 x 80 floats, the count, then the dump (kDumpBytes); always UAV
ID3D12Resource* g_fitReadback = nullptr;      // kRing x 256 bytes, persistently mapped
const uint8_t* g_fitReadbackData = nullptr;
// THE FIT'S SAMPLE DUMP (overlay 'Save the fit's samples'; for fits while driving in one D3D12 game, rejected or
// sloppy with the cabin left out too -- what the game hands while driving is the question). Armed, the next kDumpFits
// fits set KeepStale bit 3: the measuring round writes its 4096 samples (raw depth, raw motion x / y, fitted) and the
// fit's stamp into the partials buffer past the count; we copy that part into a readback ring slot beside the fit's,
// read it with the fit kRing evaluates later (the stamp must match), and a worker thread writes it to
// OptiFitDump\<armed at>\ beside the exe. No barrier on anything of the game's; our partials buffer only.
constexpr uint32_t kPartialWords = 64 * 80;                // the partial sums; the count is the next word
constexpr uint32_t kDumpBytes = (4 + 4096 * 4) * 4;        // the count, the stamp, 2 spare, then 4096 x 4 words
constexpr uint32_t kDumpStride = 65792;                    // a ring slot (>= kDumpBytes)
constexpr uint32_t kDumpFits = 16;                         // fits saved per press (8 per eye)
ID3D12Resource* g_dumpReadback = nullptr;                  // kRing x kDumpStride, persistently mapped
const uint8_t* g_dumpReadbackData = nullptr;
struct DumpMeta
{
    bool armed = false;
    uint64_t present = 0, seq = 0;
    uint32_t parity = 0, keepStale = 0, linear = 0;
    uint32_t mw = 0, mh = 0, dw = 0, dh = 0, mf = 0, df = 0;
    float scale[2] = { 0, 0 };
};
DumpMeta g_dumpMeta[kRing];
uint32_t g_dumpLeft = 0, g_dumpLost = 0;
std::atomic<uint32_t> g_dumpSaved { 0 };
std::string g_dumpFolder;
ID3D12Resource* g_composed = nullptr; // R16G16_FLOAT at the game's motion size; DLSS reads it in place of the game's
// The compose's second output: DLSS's current-colour bias mask (R8_UNORM, the composed size), 1 where the game's vector
// disagrees with the fitted camera. Made with g_composed, written at every compose, handed to DLSS only while
// UpscalerSameEyeBiasMask is on (DLSS.Input.Bias.Current.Color.Mask; the game's own value, if any, is put back after).
ID3D12Resource* g_mask = nullptr;
D3D12_RESOURCE_STATES g_maskState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
bool g_maskHanded = false;
void* g_maskWas = nullptr;
uint64_t g_maskEvals = 0;
bool g_loggedFirstMask = false;
D3D12_RESOURCE_STATES g_composedState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
UINT64 g_composedW = 0;
UINT g_composedH = 0;
uint32_t g_sizeMismatch = 0;
std::vector<std::pair<ID3D12Resource*, uint64_t>> g_retired; // ours, released 16 Presents after retirement
uint64_t g_fitSeq = 0;
uint32_t g_fitSlotParity[kRing] = {};
uint64_t g_fitPresent[2] = { UINT64_MAX, UINT64_MAX };
ID3D12Resource* g_swappedFrom = nullptr; // the game's motion texture while ours is in the parameters (compared only)
bool g_swapped = false;
// The full-size evaluate of a foveated frame (MainVectorsBefore / After): the vectors handed, and those found in the
// block after it when something below us changed them. Compared and read within the same
// evaluate only; never AddRef'd.
ID3D12Resource* g_mainMvHanded = nullptr;
ID3D12Resource* g_mainMvBelow = nullptr;
// A DLSS full-size pass (the game's or the cheap periphery) ran before this evaluate's crop: set by MainVectorsBefore,
// cleared when the crop has looked (BeginFovea). Without it g_mainMvBelow would be a texture of an earlier frame (the
// own TAA periphery runs no DLSS pass at all).
bool g_mainRan = false;
float g_mainScaleHanded[2] = { 0, 0 };
float g_mainScaleBelow[2] = { 0, 0 };  // the MV scale beside g_mainMvBelow (DL(S)SS has left it unchanged so far)
bool g_scaleHandedDown = false;        // the game's MV scale replaced by DL(S)SS's for this evaluate (RestoreInputs)
float g_scaleWas[2] = { 0, 0 };
uint32_t g_belowHandedDown = 0; // per window: periphery evaluates whose DL(S)SS vectors were handed down to the game's block
uint32_t g_mainChecked = 0, g_mainSwapped = 0;          // this window
uint32_t g_cropVecBelow = 0, g_cropVecOurs = 0, g_cropVecBelowMissing = 0; // this window: the crop's vectors (RR or SR)
// The RR crop direct on the snippet DLL: creates (ok / failed, the last failure's code) since start, evaluates per window.
std::atomic<int> g_cropDirectMade { 0 }, g_cropDirectFailed { 0 }, g_cropDirectLastFail { 0 };
std::atomic<uint32_t> g_cropDirectEvals { 0 }, g_cropDirectEvalFails { 0 }, g_cropCoreEvals { 0 };
std::atomic<int> g_srDirectMade { 0 }, g_srDirectFailed { 0 }, g_srDirectLastFail { 0 };
std::atomic<uint32_t> g_srDirectEvals { 0 }, g_srDirectEvalFails { 0 }, g_srCoreEvals { 0 };
bool g_loggedMainSwap = false, g_loggedMainSame = false, g_loggedCropVecMissing = false;
int g_cropVecSourceLast = -1;
std::string DescribeTexture(ID3D12Resource* r)
{
    if (r == nullptr)
        return "null";
    const D3D12_RESOURCE_DESC d = r->GetDesc();
    return std::format("{:p} {}x{} fmt {}", (void*) r, d.Width, d.Height, (int) d.Format);
}

uint64_t g_composedEvals = 0, g_chainBreaks = 0, g_sizeSkips = 0, g_fitsRead = 0, g_fitsValid = 0, g_fitsNotReady = 0;
// Fits that gave up the eye's stale start (UpscalerFitGiveUpStale), and of those the accepted ones; per window.
uint64_t g_fitGaveUp = 0, g_fitGaveUpAccepted = 0;
// Fits that restarted from no motion within the fit (UpscalerFitRestart: round 0 found the eye's last fit left
// behind), and of those the accepted ones; per window.
uint64_t g_fitRestarted = 0, g_fitRestartedAccepted = 0;
// The share of the samples with depth the fit left out beyond a depth gap (UpscalerFitLeaveOutNearest: a car around
// the camera), summed over the fits read this window, and the fits that left anything out.
double g_fitLeftOutSum = 0;
uint64_t g_fitLeftOutFits = 0;
bool g_loggedFirstRestart = false;
// Evaluates whose fit and compose took 1/depth (UpscalerFitDepth, or Auto: RR's DLSS.Use.HW.Depth = 0); per window.
uint64_t g_fitLinear = 0, g_fitHardware = 0;
int g_loggedDepthMode = -1;

// Whether this evaluate's depth is linear: the overlay's choice, or Auto = what the game tells RR. Logged when it
// changes, counted per window.
bool LinearDepthFor(NVSDK_NGX_Parameter* params)
{
    const int mode = Config::Instance()->UpscalerFitDepth.value_or_default(); // 0 auto, 1 hardware, 2 linear
    int hw = -1;
    if (params != nullptr)
        params->Get("DLSS.Use.HW.Depth", &hw);
    const bool linear = mode == 2 || (mode == 0 && hw == 0);
    ++(linear ? g_fitLinear : g_fitHardware);
    const int key = (linear ? 1 : 0) + 2 * mode + 8 * (hw + 1);
    if (key != g_loggedDepthMode)
    {
        g_loggedDepthMode = key;
        LOG_INFO("Per-eye DLSS DX12: same-eye fit on {} depth (setting {}; the game's DLSS.Use.HW.Depth = {})",
                 linear ? "LINEAR (1/depth)" : "hardware", mode == 0 ? "Auto" : mode == 1 ? "Hardware" : "Linear",
                 hw < 0 ? std::string("not set") : std::to_string(hw));
    }
    return linear;
}
bool g_loggedFirstGiveUp = false;
uint64_t g_fitReasons[4] = {}, g_fitKeptUsed = 0, g_fitStale = 0;
// Parity == eye check: this Present's fit maps it to the previous Present (the other eye), so the x row's depth term
// carries the eye-to-eye parallax, whose sign is fixed per eye. Counted per parity (accepted fits only).
uint64_t g_parallaxSign[2][2] = {}; // [parity][positive]
// The same signs for the whole session: which parity is the LEFT eye. This Present's fit maps it to the other eye; a
// left eye's partner sits to its right, so near points (reversed Z: larger z) land further left there -- the x row's
// depth term is negative on the left eye's Presents.
uint64_t g_eyeVotes[2][2] = {}; // [parity][positive]
double g_fitResidualSum = 0, g_fitInlierSum = 0;
// The REJECTED fits ('too few within 1 px', reason 3) per window: one D3D12 game had whole
// windows of 600 rejected while driving, DLSS on the eye's stale fit for 40 s at a time (a shimmer while
// driving); the summary showed the accepted fits' statistics only. What the rejected ones looked like decides whether
// the half-of-the-samples rule is too strict (inliers just under 50%) or the model is wrong in those frames.
double g_rejInlierSum = 0, g_rejResidualSum = 0, g_rejSampleSum = 0;
uint64_t g_rejCount = 0, g_rejHist[4] = {}; // inliers under 10%, 10-25%, 25-40%, 40-50%
bool g_loggedFirstNrOnly = false; // NR-only mode's first same-eye motion (under g_mutex)
bool g_loggedMotionFail = false, g_loggedFirstComposed = false, g_loggedFirstFit = false, g_loggedSizeSkip = false,
     g_loggedFitNotReady = false;

// GPU time per evaluate (probe only): timestamps on the evaluate's list before and after our same-eye passes and
// around the upscaler's evaluate, read back when the slot comes round again (kRing evaluates later). Split by routing
// mode and Present parity (= eye). Step 1 of foveated DLSS: how much there is to save.
bool g_timeTried = false;
bool g_timeReady = false;
ID3D12QueryHeap* g_queryHeap = nullptr;       // kRing x 4 timestamps
ID3D12Resource* g_timeReadback = nullptr;     // kRing x 32 bytes, persistently mapped
uint64_t* g_timeData = nullptr;
uint64_t g_tsFrequency = 0;                   // from the present queue (all direct queues of the GPU share it)
uint64_t g_timeSeq = 0;
bool g_timeFilled[kRing] = {};
int g_timeMode[kRing] = {};                   // 0 shared, 1 per eye
int g_timeParity[kRing] = {};
bool g_timingThis = false;                    // this evaluate wrote T0..T3; EndEvaluate writes T4
bool g_wroteFitTime = false;                  // SameEye wrote T1 (after the fit)
bool g_wroteUpscaleTime = false;              // AfterUpscale wrote T4 (after the upscaler, before the foveal crop)
uint32_t g_timeSlotCur = 0;
struct GpuTimes
{
    double fit = 0, compose = 0, dlss = 0, fovea = 0;
    uint64_t n = 0;
};
GpuTimes g_times[2][2];                       // [per eye][parity]
uint64_t g_timesInvalid = 0;
bool g_loggedTimeFail = false, g_loggedFirstTime = false;

bool ProbeOn()
{
    static const bool on = Config::Instance()->UpscalerInputProbe.value_or_default();
    return on;
}

// GPU timestamps around our passes and the upscaler: with the probe, or on their own ([Inputs] UpscalerGpuTiming, the
// menu's GPU timing switch, live) -- it used to be only with the probe, which also runs a statistics pass per evaluate.
bool TimingOn()
{
    return ProbeOn() || Config::Instance()->UpscalerGpuTiming.value_or_default();
}

void LogOnce(bool& logged, const std::string& what)
{
    if (logged)
        return;
    logged = true;
    LOG_WARN("Per-eye DLSS DX12: {}", what);
}

// Called once, on the first evaluate with the probe on. Failure is final (logged once), never retried per frame.
void Setup(ID3D12Device* device)
{
    g_setupTried = true;

    D3D12_DESCRIPTOR_RANGE range {};
    range.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    range.NumDescriptors = 1;
    range.BaseShaderRegister = 0;
    range.OffsetInDescriptorsFromTableStart = 0;

    D3D12_ROOT_PARAMETER params[3] {};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[0].Constants.ShaderRegister = 0;
    params[0].Constants.Num32BitValues = 4;
    params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[1].DescriptorTable.NumDescriptorRanges = 1;
    params[1].DescriptorTable.pDescriptorRanges = &range;
    params[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    params[2].Descriptor.ShaderRegister = 0;
    params[2].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    D3D12_ROOT_SIGNATURE_DESC rsDesc {};
    rsDesc.NumParameters = 3;
    rsDesc.pParameters = params;
    ID3DBlob* rsBlob = nullptr;
    ID3DBlob* rsError = nullptr;
    if (FAILED(D3D12SerializeRootSignature(&rsDesc, D3D_ROOT_SIGNATURE_VERSION_1, &rsBlob, &rsError)) ||
        rsBlob == nullptr)
    {
        LogOnce(g_loggedSetupFail, std::format("motion statistics off: root signature did not serialise ({})",
                                               rsError ? (const char*) rsError->GetBufferPointer() : "no message"));
        if (rsError)
            rsError->Release();
        return;
    }
    const HRESULT rsHr = device->CreateRootSignature(0, rsBlob->GetBufferPointer(), rsBlob->GetBufferSize(),
                                                     IID_PPV_ARGS(&g_rootSignature));
    rsBlob->Release();
    if (FAILED(rsHr))
        return LogOnce(g_loggedSetupFail, std::format("motion statistics off: CreateRootSignature {:#x}", (unsigned) rsHr));

    ID3DBlob* cs = CompileShader(kStatsShader, "CSMain", "cs_5_0");
    if (cs == nullptr)
        return LogOnce(g_loggedSetupFail, "motion statistics off: the statistics shader did not compile (see above)");
    D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc {};
    psoDesc.pRootSignature = g_rootSignature;
    psoDesc.CS = { cs->GetBufferPointer(), cs->GetBufferSize() };
    const HRESULT psoHr = device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&g_pipeline));
    cs->Release();
    if (FAILED(psoHr))
        return LogOnce(g_loggedSetupFail,
                       std::format("motion statistics off: CreateComputePipelineState {:#x}", (unsigned) psoHr));

    D3D12_DESCRIPTOR_HEAP_DESC heapDesc {};
    heapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    heapDesc.NumDescriptors = kRing;
    heapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    if (FAILED(device->CreateDescriptorHeap(&heapDesc, IID_PPV_ARGS(&g_heap))))
        return LogOnce(g_loggedSetupFail, "motion statistics off: no descriptor heap");
    g_descriptorSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

    D3D12_RESOURCE_DESC buf {};
    buf.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    buf.Width = kRing * 16;
    buf.Height = 1;
    buf.DepthOrArraySize = 1;
    buf.MipLevels = 1;
    buf.SampleDesc.Count = 1;
    buf.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    D3D12_HEAP_PROPERTIES heap {};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    buf.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    if (FAILED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &buf, D3D12_RESOURCE_STATE_COMMON,
                                               nullptr, IID_PPV_ARGS(&g_sums))))
        return LogOnce(g_loggedSetupFail, "motion statistics off: no sums buffer");
    heap.Type = D3D12_HEAP_TYPE_READBACK;
    buf.Flags = D3D12_RESOURCE_FLAG_NONE;
    if (FAILED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &buf, D3D12_RESOURCE_STATE_COPY_DEST,
                                               nullptr, IID_PPV_ARGS(&g_readback))))
        return LogOnce(g_loggedSetupFail, "motion statistics off: no readback buffer");
    void* mapped = nullptr;
    if (FAILED(g_readback->Map(0, nullptr, &mapped)) || mapped == nullptr)
        return LogOnce(g_loggedSetupFail, "motion statistics off: the readback buffer did not map");
    g_readbackData = (const uint8_t*) mapped;
    std::memset(mapped, 0xFF, kRing * 16); // stamps that no evaluate writes
    g_ready = true;
    LOG_INFO("Per-eye DLSS DX12: motion statistics ready (64x64 grid, read back {} evaluates later)", kRing);
}

ID3D12Resource* GetResource(NVSDK_NGX_Parameter* params, const char* name)
{
    ID3D12Resource* resource = nullptr;
    if (params->Get(name, &resource) != NVSDK_NGX_Result_Success)
        params->Get(name, (void**) &resource);
    return resource;
}

DXGI_FORMAT MotionSrvFormat(DXGI_FORMAT format)
{
    switch (format)
    {
    case DXGI_FORMAT_R16G16_TYPELESS:
        return DXGI_FORMAT_R16G16_FLOAT;
    case DXGI_FORMAT_R32G32_TYPELESS:
        return DXGI_FORMAT_R32G32_FLOAT;
    default:
        return format;
    }
}

// The slot about to be reused holds the statistics of the evaluate kRing back, if the GPU has written it by now.
void ReadSlot(uint32_t slot)
{
    if (g_statsSeq < kRing)
        return;
    const uint8_t* p = g_readbackData + slot * 16;
    uint32_t stamp = 0;
    std::memcpy(&stamp, p + 12, 4);
    if (stamp != (uint32_t) (g_statsSeq - kRing))
    {
        ++g_statsNotReady;
        LogOnce(g_loggedNotReady, std::format("a statistics slot was not written {} evaluates later (stamp {:#x}, "
                                              "expected {:#x}); counted per window as 'not ready'",
                                              kRing, stamp, (uint32_t) (g_statsSeq - kRing)));
        return;
    }
    float v[3];
    std::memcpy(v, p, 12);
    auto& par = g_parity[g_slotParity[slot]];
    par.sumX += v[0];
    par.sumAbsX += v[1];
    par.sumAbsY += v[2];
    ++par.frames;
    if (!g_loggedFirstStats)
    {
        g_loggedFirstStats = true;
        LOG_INFO("Per-eye DLSS DX12: FIRST motion statistics read back: mean x {:+.3f} px, |x| {:.3f}, |y| {:.3f} "
                 "({} Presents)",
                 v[0], v[1], v[2], g_slotParity[slot] ? "odd" : "even");
    }
}

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

void Transition(ID3D12GraphicsCommandList* cmdList, ID3D12Resource* resource, D3D12_RESOURCE_STATES before,
                D3D12_RESOURCE_STATES after)
{
    D3D12_RESOURCE_BARRIER barrier {};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = resource;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore = before;
    barrier.Transition.StateAfter = after;
    cmdList->ResourceBarrier(1, &barrier);
}

// Called once, on the first per-eye evaluate. Failure is final (logged once), never retried per frame.
void SetupMotion(ID3D12Device* device)
{
    g_motionTried = true;

    D3D12_DESCRIPTOR_RANGE ranges[3] {};
    ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    ranges[0].NumDescriptors = 2;
    ranges[0].BaseShaderRegister = 0;
    ranges[0].OffsetInDescriptorsFromTableStart = 0;
    ranges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    ranges[1].NumDescriptors = 1;
    ranges[1].BaseShaderRegister = 1;
    ranges[1].OffsetInDescriptorsFromTableStart = 2;
    ranges[2].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV; // the bias mask (u3)
    ranges[2].NumDescriptors = 1;
    ranges[2].BaseShaderRegister = 3;
    ranges[2].OffsetInDescriptorsFromTableStart = 3;

    D3D12_ROOT_PARAMETER params[5] {};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[0].Constants.ShaderRegister = 0;
    params[0].Constants.Num32BitValues = 12;
    params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[1].DescriptorTable.NumDescriptorRanges = 3;
    params[1].DescriptorTable.pDescriptorRanges = ranges;
    params[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    params[2].Descriptor.ShaderRegister = 0;
    params[2].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    params[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
    params[3].Descriptor.ShaderRegister = 2;
    params[3].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    params[4].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    params[4].Descriptor.ShaderRegister = 2;
    params[4].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    D3D12_ROOT_SIGNATURE_DESC rsDesc {};
    rsDesc.NumParameters = 5;
    rsDesc.pParameters = params;
    ID3DBlob* rsBlob = nullptr;
    ID3DBlob* rsError = nullptr;
    if (FAILED(D3D12SerializeRootSignature(&rsDesc, D3D_ROOT_SIGNATURE_VERSION_1, &rsBlob, &rsError)) ||
        rsBlob == nullptr)
    {
        LogOnce(g_loggedMotionFail, std::format("same-eye motion off: root signature did not serialise ({})",
                                                rsError ? (const char*) rsError->GetBufferPointer() : "no message"));
        if (rsError)
            rsError->Release();
        return;
    }
    const HRESULT rsHr = device->CreateRootSignature(0, rsBlob->GetBufferPointer(), rsBlob->GetBufferSize(),
                                                     IID_PPV_ARGS(&g_motionRoot));
    rsBlob->Release();
    if (FAILED(rsHr))
        return LogOnce(g_loggedMotionFail,
                       std::format("same-eye motion off: CreateRootSignature {:#x}", (unsigned) rsHr));

    auto makePso = [&](const char* source, const char* name, ID3D12PipelineState** pso)
    {
        ID3DBlob* cs = CompileShader(source, "CSMain", "cs_5_0");
        if (cs == nullptr)
        {
            LogOnce(g_loggedMotionFail, std::format("same-eye motion off: the {} shader did not compile (see above)", name));
            return false;
        }
        D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc {};
        psoDesc.pRootSignature = g_motionRoot;
        psoDesc.CS = { cs->GetBufferPointer(), cs->GetBufferSize() };
        const HRESULT hr = device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(pso));
        cs->Release();
        if (FAILED(hr))
        {
            LogOnce(g_loggedMotionFail,
                    std::format("same-eye motion off: CreateComputePipelineState ({}) {:#x}", name, (unsigned) hr));
            return false;
        }
        return true;
    };
    if (!makePso(kFitShader, "fit", &g_fitPso) || !makePso(kComposeShader, "compose", &g_composePso))
        return;

    D3D12_DESCRIPTOR_HEAP_DESC heapDesc {};
    heapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    heapDesc.NumDescriptors = kRing * 4;
    heapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    if (FAILED(device->CreateDescriptorHeap(&heapDesc, IID_PPV_ARGS(&g_motionHeap))))
        return LogOnce(g_loggedMotionFail, "same-eye motion off: no descriptor heap");
    if (g_descriptorSize == 0)
        g_descriptorSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

    D3D12_RESOURCE_DESC buf {};
    buf.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    buf.Width = 20 * 16;
    buf.Height = 1;
    buf.DepthOrArraySize = 1;
    buf.MipLevels = 1;
    buf.SampleDesc.Count = 1;
    buf.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    buf.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    D3D12_HEAP_PROPERTIES heap {};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    if (FAILED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &buf, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                               IID_PPV_ARGS(&g_fits))))
        return LogOnce(g_loggedMotionFail, "same-eye motion off: no fits buffer");
    buf.Width = kPartialWords * 4 + kDumpBytes; // the partial sums, the fit shader's count of groups done (starts
                                                // zeroed), its dump's stamp and samples (kDumpBytes from the count)
    if (FAILED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &buf, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                               IID_PPV_ARGS(&g_partials))))
        return LogOnce(g_loggedMotionFail, "same-eye motion off: no partial sums buffer");
    buf.Width = kRing * 256;
    buf.Flags = D3D12_RESOURCE_FLAG_NONE;
    heap.Type = D3D12_HEAP_TYPE_READBACK;
    if (FAILED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &buf, D3D12_RESOURCE_STATE_COPY_DEST,
                                               nullptr, IID_PPV_ARGS(&g_fitReadback))))
        return LogOnce(g_loggedMotionFail, "same-eye motion off: no fit readback buffer");
    void* mapped = nullptr;
    if (FAILED(g_fitReadback->Map(0, nullptr, &mapped)) || mapped == nullptr)
        return LogOnce(g_loggedMotionFail, "same-eye motion off: the fit readback buffer did not map");
    g_fitReadbackData = (const uint8_t*) mapped;
    std::memset(mapped, 0xFF, kRing * 256);
    buf.Width = kRing * kDumpStride; // the dump's ring (64 KB a slot); without it the overlay's dump says so
    void* dumpMapped = nullptr;
    if (SUCCEEDED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &buf, D3D12_RESOURCE_STATE_COPY_DEST,
                                                  nullptr, IID_PPV_ARGS(&g_dumpReadback))) &&
        SUCCEEDED(g_dumpReadback->Map(0, nullptr, &dumpMapped)) && dumpMapped != nullptr)
        g_dumpReadbackData = (const uint8_t*) dumpMapped;
    else
        LOG_WARN("Per-eye DLSS DX12: no readback buffer for the fit's sample dump; 'Save the fit's samples' will refuse");
    g_motionReady = true;
    LOG_INFO("Per-eye DLSS DX12: same-eye motion ready (fit 64x64 in 64 groups x 5 rounds, one dispatch each, from the previous fit; compose "
             "{})",
             ObjectMotionName());
}

// Our same-eye motion texture at the game's motion size. A different size is adopted only after it held for 90
// evaluates (menus and loading frames at other sizes must not rebuild it); the old texture is retired, not released:
// work recorded a moment ago can still reference it.
bool EnsureComposed(ID3D12Device* device, const D3D12_RESOURCE_DESC& md, uint64_t present)
{
    if (g_composed != nullptr && g_composedW == md.Width && g_composedH == md.Height)
    {
        g_sizeMismatch = 0;
        return true;
    }
    if (g_composed != nullptr && ++g_sizeMismatch < 90)
        return false;
    if (g_composed != nullptr)
    {
        g_retired.push_back({ g_composed, present });
        g_composed = nullptr;
    }
    if (g_mask != nullptr)
    {
        g_retired.push_back({ g_mask, present });
        g_mask = nullptr;
    }
    g_sizeMismatch = 0;
    D3D12_RESOURCE_DESC desc {};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = md.Width;
    desc.Height = md.Height;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = DXGI_FORMAT_R16G16_FLOAT;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    D3D12_HEAP_PROPERTIES heap {};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    if (FAILED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
                                               D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
                                               IID_PPV_ARGS(&g_composed))))
    {
        g_composed = nullptr;
        return false;
    }
    g_composed->SetName(L"PerEyeDlss_Dx12 same-eye motion");
    g_composedState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    desc.Format = DXGI_FORMAT_R8_UNORM;
    if (FAILED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
                                               D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&g_mask))))
    {
        g_mask = nullptr;
        g_retired.push_back({ g_composed, present });
        g_composed = nullptr;
        return false;
    }
    g_mask->SetName(L"PerEyeDlss_Dx12 bias mask");
    g_maskState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    g_composedW = md.Width;
    g_composedH = md.Height;
    LOG_INFO("Per-eye DLSS DX12: same-eye motion texture {}x{} made", md.Width, md.Height);
    return true;
}

// The fit slot about to be reused holds the fit of the evaluate kRing back, if the GPU has written it by now.
void ReadFitSlot(uint32_t slot)
{
    if (g_fitSeq < kRing)
        return;
    float f[60];
    std::memcpy(f, g_fitReadbackData + slot * 256, sizeof(f));
    const float* m = f + 2 * 20; // this Present's own fit (the scratch slot)
    const float* kept = f + g_fitSlotParity[slot] * 20;
    const float expected = (float) ((g_fitSeq - kRing) & 0xFFFFFF);
    if (!(m[19] == expected))
    {
        if (g_dumpMeta[slot].armed)
        {
            g_dumpMeta[slot].armed = false;
            ++g_dumpLost;
            LOG_WARN("Per-eye DLSS DX12: FIT DUMP: fit {} not read back in time (its slot was not written); not saved",
                     g_dumpMeta[slot].seq);
        }
        ++g_fitsNotReady;
        LogOnce(g_loggedFitNotReady, std::format("a fit slot was not written {} evaluates later (stamp {}, expected {}); "
                                                 "counted per window",
                                                 kRing, m[19], expected));
        return;
    }
    ++g_fitsRead;
    const int reason = std::clamp((int) m[16], 0, 3);
    if (g_dumpMeta[slot].armed)
    {
        g_dumpMeta[slot].armed = false;
        const uint8_t* d = g_dumpReadbackData + (size_t) slot * kDumpStride;
        uint32_t stamp = 0;
        std::memcpy(&stamp, d + 4, 4);
        if (stamp != ((uint32_t) (g_fitSeq - kRing) & 0xFFFFFFu))
        {
            ++g_dumpLost;
            LOG_WARN("Per-eye DLSS DX12: FIT DUMP: fit {}'s samples carry stamp {} (expected {}); not saved",
                     g_dumpMeta[slot].seq, stamp, (g_fitSeq - kRing) & 0xFFFFFF);
        }
        else
        {
            const DumpMeta meta = g_dumpMeta[slot];
            std::vector<uint8_t> samples(d + 16, d + kDumpBytes);
            std::vector<float> fit(f, f + 60);
            const std::string folder = g_dumpFolder;
            // File I/O off the game's thread; the thread holds copies only.
            std::thread(
                [meta, samples = std::move(samples), fit = std::move(fit), folder]
                {
                    std::error_code error;
                    std::filesystem::create_directories(folder, error);
                    const auto base = std::filesystem::path(folder) / std::format("fit_{:08}_p{}", meta.seq, meta.parity);
                    std::ofstream bin(base.string() + ".bin", std::ios::binary);
                    bin.write((const char*) samples.data(), (std::streamsize) samples.size());
                    const float* m = fit.data() + 2 * 20;
                    std::ofstream txt(base.string() + ".txt");
                    txt << std::format(
                        "Same-eye fit sample dump (PerEyeFitShader_Dx12). {}.bin: 4096 samples x 4 float32, row-major "
                        "over a 64x64 grid (sample s at uv ((s % 64) + 0.5) / 64, ((s / 64) + 0.5) / 64), each: raw depth "
                        "(as handed), raw motion x, raw motion y (before MV scale), 1.0 if fitted (0 = left out: no depth, "
                        "not finite, or the nearest half)\n"
                        "seq {} present {} parity {} keepStale {} linearDepth {}\n"
                        "motion {}x{} fmt {}, depth {}x{} fmt {}, MV scale {} {}\n"
                        "verdict: reason {} (0 accepted, 1 solve failed, 2 too few depth samples, 3 too few within 1 px), "
                        "inlier residual {:.4f} px, within 1 px {:.2f}%, samples counted {}\n"
                        "gave up the start {}, restarted/left-out field {:.4f}\n"
                        "this fit (slot 2) rows: x {:.7f} {:.7f} {:.7f} {:.7f} | y {:.7f} {:.7f} {:.7f} {:.7f} | w {:.7f} "
                        "{:.7f} {:.7f} {:.7f}\n"
                        "slot 0 rows: x {:.7f} {:.7f} {:.7f} {:.7f} | y {:.7f} {:.7f} {:.7f} {:.7f} | w {:.7f} {:.7f} "
                        "{:.7f}; stats {:.4f} {:.4f} {:.1f} {:.0f}; row4 {:.1f} {:.1f}\n"
                        "slot 1 rows: x {:.7f} {:.7f} {:.7f} {:.7f} | y {:.7f} {:.7f} {:.7f} {:.7f} | w {:.7f} {:.7f} "
                        "{:.7f}; stats {:.4f} {:.4f} {:.1f} {:.0f}; row4 {:.1f} {:.1f}\n",
                        base.filename().string(), meta.seq, meta.present, meta.parity, meta.keepStale, meta.linear,
                        meta.mw, meta.mh, meta.mf, meta.dw, meta.dh, meta.df, meta.scale[0], meta.scale[1], m[16], m[12],
                        m[13] * 100.0f, m[15], m[17], m[18], m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8], m[9],
                        m[10], m[11], fit[0], fit[1], fit[2], fit[3], fit[4], fit[5], fit[6], fit[7], fit[8], fit[9],
                        fit[10], fit[12], fit[13], fit[14], fit[15], fit[16], fit[17], fit[20], fit[21], fit[22], fit[23],
                        fit[24], fit[25], fit[26], fit[27], fit[28], fit[29], fit[30], fit[32], fit[33], fit[34], fit[35],
                        fit[36], fit[37]);
                    const bool ok = bin.good() && txt.good();
                    if (ok)
                        ++g_dumpSaved;
                    LOG_INFO("Per-eye DLSS DX12: FIT DUMP: fit {} (parity {}, reason {}, {:.1f}% within 1 px) {} {}",
                             meta.seq, meta.parity, (int) m[16], m[13] * 100.0f, ok ? "saved to" : "FAILED to write",
                             base.string());
                })
                .detach();
        }
    }
    ++g_fitReasons[reason];
    if (m[17] > 0.5f)
    {
        ++g_fitGaveUp;
        if (reason == 0)
            ++g_fitGaveUpAccepted;
        if (!g_loggedFirstGiveUp)
        {
            g_loggedFirstGiveUp = true;
            LOG_INFO("Per-eye DLSS DX12: FIRST stale start given up (parity {}): reason {} (0 = accepted), "
                     "residual {:.4f} px, {:.1f}% within 1 px",
                     g_fitSlotParity[slot], reason, m[12], m[13] * 100.0f);
        }
    }
    if (m[18] > 0.5f)
    {
        ++g_fitRestarted;
        if (reason == 0)
            ++g_fitRestartedAccepted;
        if (!g_loggedFirstRestart)
        {
            g_loggedFirstRestart = true;
            LOG_INFO("Per-eye DLSS DX12: FIRST fit restarted from no motion within the fit (parity {}): reason {} "
                     "(0 = accepted), residual {:.4f} px, {:.1f}% within 1 px",
                     g_fitSlotParity[slot], reason, m[12], m[13] * 100.0f);
        }
    }
    // Row 4's z: restarted (0 / 1) + 0.45 x the share left out beyond the depth gap.
    const float leftOut = (m[18] - (m[18] > 0.5f ? 1.0f : 0.0f)) / 0.45f;
    if (leftOut > 0.0f)
    {
        g_fitLeftOutSum += leftOut;
        ++g_fitLeftOutFits;
    }
    if (reason != 0)
        ++(kept[17] > 2.0f ? g_fitStale : g_fitKeptUsed);
    if (reason == 3)
    {
        ++g_rejCount;
        g_rejInlierSum += m[13];
        g_rejResidualSum += m[12];
        g_rejSampleSum += m[15];
        ++g_rejHist[m[13] < 0.10f ? 0 : m[13] < 0.25f ? 1 : m[13] < 0.40f ? 2 : 3];
    }
    if (m[14] > 0.5f)
    {
        ++g_parallaxSign[g_fitSlotParity[slot] & 1][m[2] > 0.0f ? 1 : 0];
        ++g_eyeVotes[g_fitSlotParity[slot] & 1][m[2] > 0.0f ? 1 : 0];
        ++g_fitsValid;
        g_fitResidualSum += m[12];
        g_fitInlierSum += m[13];
    }
    if (!g_loggedFirstFit)
    {
        g_loggedFirstFit = true;
        LOG_INFO("Per-eye DLSS DX12: FIRST fit read back: {} (reason {}) -- inlier residual {:.4f} px, {:.1f}% of {} "
                 "samples within 1 px; M rows x ({:.5f} {:.5f} {:.5f} {:.5f}) y ({:.5f} {:.5f} {:.5f} {:.5f}) w ({:.5f} "
                 "{:.5f} {:.5f} 1)",
                 m[14] > 0.5f ? "VALID" : "REJECTED", reason, m[12], m[13] * 100.0f, (int) m[15], m[0], m[1], m[2],
                 m[3], m[4], m[5], m[6], m[7], m[8], m[9], m[10]);
    }
}

// ---- Foveal crop (foveated DLSS, step 2) ----
//
// A second DLSS feature per eye covers a centred crop of the eye at its own preset (M by default), drawn over the full
// DLSS pass. Its inputs are the same colour, depth and motion (ours per eye) read through DLSS's input subrects; its
// output is our own texture, blended over the crop of the game's output by COMPOSITE with a feathered edge. F4 turns
// it on (the crop features are created at the first press, never at a size change) and off; F8 draws its outline.
// Step 2 saves nothing yet: it shows the crop's quality and the seam before the periphery is made cheaper.
#include "FoveaShader_Dx12.h" // kCompositeShader (the feather blend), kEnlargeShader (the periphery's upscale)

// Eye tracking, step 1: a crop-sized box (green, with a cross at its centre) where the eyes look, drawn with the
// outline (F8) while the crop itself stays put -- shows the gaze mapping before anything depends on it.
constexpr char kGazeMarkShader[] = R"(
RWTexture2D<float4> Output : register(u0);

cbuffer Constants : register(b0)
{
    uint2 Base;
    uint2 Size;
    uint Thick;
    uint Colour; // 0 green (gaze), 1 yellow (calibration target)
    uint2 Centre;
};

[numthreads(8, 8, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= Size.x || id.y >= Size.y)
        return;
    const uint d = min(min(id.x, id.y), min(Size.x - 1 - id.x, Size.y - 1 - id.y));
    const int2 p = int2(Base + id.xy) - int2(Centre);
    const bool cross = (abs(p.x) < 3 && abs(p.y) < 48) || (abs(p.y) < 3 && abs(p.x) < 48);
    if (d < Thick || cross)
        Output[Base + id.xy] = Colour == 1 ? float4(4.0, 4.0, 0.0, 1.0) : float4(0.0, 4.0, 0.0, 1.0);
}
)";

struct FoveaRect
{
    uint32_t rx = 0, ry = 0, rw = 0, rh = 0; // render subrect of the crop
    uint32_t ox = 0, oy = 0, ow = 0, oh = 0; // its place in the output
    uint32_t renderW = 0, renderH = 0, outW = 0, outH = 0;
    bool exact = false;
    bool gaze = false;          // where the eyes look is known for this evaluate
    uint32_t gazeX = 0, gazeY = 0; // in output pixels
    bool target = false;        // a calibration target to draw (VarjoGaze), output pixels
    uint32_t targetX = 0, targetY = 0;
};

bool g_foveaOn = false;
bool g_foveaOutline = false;
// The green box at the raw gaze is drawn with the outline only while the crop FOLLOWS the gaze (a box that follows
// the eye while the crop is still is only a distraction);
// this session-only switch (the Junk Drawer's) shows it with a still crop too, to check the tracker.
bool g_gazeBoxWhenStill = false;
bool g_foveaFailed = false; // creation or setup failed: final, logged once
std::atomic<int> g_presetForCreate { 0 }; // read from DLSSFeature's create, outside our lock
std::atomic<int> g_kindForCreate { 0 };   // 1 a crop, 2 a periphery, while its feature is created or evaluated
std::atomic<int> g_rrCropPreset { -1 };   // the RR preset the crop's last RR feature was made with (NoteRrCreate)
std::atomic<bool> g_cropIsRr { false };   // the crop's last evaluate was a ray-reconstruction one
// Shared routing: the crop, the cheap periphery and the TAA periphery are keyed per Present parity = per eye
// (UpscalerCropEyeByParity), so each eye keeps its own history like NR's; per-eye routing has one owner per eye already.
// Feature ids stay far below the bit. Forget* erase both keys of an owner.
constexpr unsigned int kOddEyeKey = 0x40000000u;
unsigned int EyeKey(unsigned int handle)
{
    if (g_routing || !Config::Instance()->UpscalerCropEyeByParity.value_or_default())
        return handle;
    return (g_present.load() & 1) ? (handle | kOddEyeKey) : handle;
}
// The cheap DLSS periphery's key: one feature for both eyes in shared routing (needed under the VR mod's own DLSS
// pass: one feature evaluated every Present, as the game's), else EyeKey. UpscalerPeripheryOneFeature: 1 always, 2 never,
// 0 auto = while g_belowSeen.
// g_belowSeen: something below us is seen changing the vectors of the full-size pass of a foveated frame (the game's
// or the cheap periphery; MainVectorsAfter: on after kOneOnAfter evaluates replaced in a row, off after kOneOffAfter
// not replaced in a row). It drives both autos: the periphery's one feature and the SR crop's vectors from below.
std::atomic<bool> g_belowSeen { false };
uint32_t g_periphBelowRun = 0, g_periphOursRun = 0;
uint32_t g_periphAutoFlips = 0; // this window
constexpr uint32_t kOneOnAfter = 10, kOneOffAfter = 60;
bool PeripheryOneFeature()
{
    if (g_routing)
        return false;
    const int mode = Config::Instance()->UpscalerPeripheryOneFeature.value_or_default();
    return mode == 1 || (mode == 0 && g_belowSeen.load());
}
// The SR crop takes DL(S)SS's vectors from below: UpscalerFoveaSrVectors 1 always, 0 never (ours), -1 auto = while
// g_belowSeen (no shimmer while driving, even in fast motion).
bool SrCropFromBelow()
{
    const int v = Config::Instance()->UpscalerFoveaSrVectors.value_or_default();
    return v == 1 || (v < 0 && g_belowSeen.load());
}
// Per window: same-eye fits not run because nothing would take them (DL(S)SS's vectors serve the crop and the periphery).
uint32_t g_fitSkippedBelow = 0;
bool g_loggedFitSkippedBelow = false;
uint32_t g_nrOnBelow = 0; // per window: evaluates whose NR (after SR) reads DL(S)SS's vectors, none of ours
// Per kind, since start: MB the creates took (0 the game, 1 crop, 2 periphery, 3 NR, 4 twin) and how many.
std::atomic<int64_t> g_vramMade[5] = {};
std::atomic<uint32_t> g_vramCreates[5] = {};
const char* g_nextCreateLabel = nullptr;
std::atomic<uint64_t> g_vramWindowStartMB { 0 };

unsigned int PeriphKey(unsigned int handle) { return PeripheryOneFeature() ? handle : EyeKey(handle); }
// Per window: cheap periphery evaluates begun on one feature (the game's vectors) / on a feature per eye.
std::atomic<unsigned int> g_periphOneEvals { 0 }, g_periphPerEyeEvals { 0 };
bool g_composedForShared = false; // this evaluate: our same-eye motion exists for NR / the crop while DLSS keeps the game's
std::unordered_map<unsigned int, unsigned int> g_crops;  // feature (EyeKey) -> its crop feature
std::unordered_map<unsigned int, FoveaRect> g_cropRects; // crop feature -> the geometry it was made for
std::unordered_map<unsigned int, bool> g_cropReset;      // crop feature -> reset its history on the next evaluate
FoveaRect g_foveaPlan;                                   // this evaluate's crop
std::unordered_map<unsigned int, std::pair<uint32_t, uint32_t>> g_cropPos; // crop feature -> its last render box origin
std::unordered_map<unsigned int, unsigned int> g_dlssRebuild;             // features we asked to rebuild as DLSS
bool g_loggedFoveaMove = false;
// Eye tracking, step 2: the crop follows the gaze. One centre for both eyes (fractions of the eye), recentred on the gaze
// when it leaves the inner part of the crop (the dead zone). A crop that moved gets the move added to its copied
// motion vectors (GATHER), so DLSS sees a pan and keeps its history, instead of the reset a move used to cost.
// The centre GLIDES there ([Inputs] UpscalerFoveaGazeGlideMs, the time constant; 0 = one jump, as before): a jump moved
// a whole block of the eye from the periphery to a crop with no history there, visibly, in one frame. Gliding, each
// step is a thin strip. It keeps gliding after the gaze until it is within a tenth of the half crop of it.
bool g_followInit = false;
float g_followX = 0.5f, g_followY = 0.5f; // the glide's centre (not rounded to pixels; kept where a crop can be)
bool g_followGliding = false;
std::chrono::steady_clock::time_point g_followLast {};
// The box PlanFovea placed last for each Present parity (= eye) and the Present it was placed on, for the NR fovea
// (FoveaBoxForNr): the box differs per eye once the nose offset is set (mirrored), so one 'latest' box would hand NR
// the other eye's place on every other frame (the NR fovea then did not follow the
// crop moved by the sliders -- it followed only the gaze's centre, never the size or the offsets).
FoveaRect g_foveaPlanEye[2] {};
uint64_t g_foveaPlanEyePresent[2] = { UINT64_MAX, UINT64_MAX };
uint64_t g_followMoves = 0, g_followShifts = 0, g_followResets = 0;
// Of the restarts: moves that could not be carried over because the crop reads through DLSS's subrects (the shift
// goes into COPIED vectors: `[Inputs] UpscalerFoveaCopyInputs` off), and moves whose copy failed at the move.
uint64_t g_followResetsNoCopy = 0, g_followResetsCopyFailed = 0;
bool g_loggedNoCopyMove = false;
// The game's current-colour bias mask (DLSS.Input.Bias.Current.Color.Mask) for an SR crop: some games hand one at
// the render size, and the crop's DLSS read it at (0, 0) of the FULL frame -- the wrong place -- in both input paths
// (RR copies every guide for the box; SR copied colour, motion and depth only). `[Inputs] UpscalerFoveaCropMask`:
// 0 none for the crop, 1 copied at the box (default), 2 as handed (the old way, a test).
uint64_t g_maskCopied = 0, g_maskDropped = 0, g_maskAsIs = 0;
bool g_loggedCropMask = false, g_loggedMaskSkip = false;
// PROBE: the crop's inputs read back AFTER its evaluate. Something below us
// replaces the game's vectors in the block; if it does that to the crop's evaluate too, the copied +
// shifted vectors never reach DLSS -- the shimmer
// while the crop moves that no change of ours touched.
uint64_t g_cropAfterChecked = 0, g_cropAfterMvSwapped = 0, g_cropAfterColourSwapped = 0, g_cropAfterDepthSwapped = 0,
         g_cropAfterBaseChanged = 0;
bool g_loggedCropAfter = false;
// The probe's answer: the VECTORS are swapped on every crop evaluate (600 of 600 a
// window) for a texture of the crop's own size; colour, depth and the bases come back untouched. So the crop's shift
// cannot travel in the vectors then. `[Inputs] UpscalerFoveaShiftByOffset` (overlay 'Crop move through MV
// offset (test)'): the shift goes into DLSS's MV.Offset.X/Y parameters instead (the copied vectors unshifted); whether
// nvngx_dlss applies that parameter is not documented -- the picture decides, and the probe says whether something below
// us clears it. The hook probe below says which module the NGX evaluate entry resolves into.
uint64_t g_offsetShifts = 0, g_cropAfterOffsetChanged = 0;
// Latched for the launch: a crop evaluate that read COPIES came back with other vectors (something below us). A
// still crop then reads through subrects, which came back untouched (FoveaCopyRule.h; the readback of a subrect
// evaluate reads 0 swapped, so the latch cannot be kept up by the probe itself).
bool g_copiesSwappedBelow = false;
int g_cropCopiesLast = -1; // what the last crop evaluate read (-1 none yet): a change restarts the crops' histories
bool g_loggedOffset = false, g_loggedHook = false;
double g_placeErrMax = 0.0; // the largest output-pixel rounding of a placed crop's origin this window (PlanFovea)
bool g_loggedFollow = false, g_loggedShift = false, g_loggedGlide = false;

// Copies or subrects for the SR crop's inputs (FoveaCopyRule.h). Called under g_mutex: raw flags only.
bool CropReadsCopies(bool rayReconstruction)
{
    const auto* cfg = Config::Instance();
    const auto& key = cfg->UpscalerFoveaCopyInputs;
    return FoveaCopyRule::ReadsCopies(rayReconstruction, key.has_value(), key.has_value() ? key.value() : false,
                                      key.value_or_default(), g_copiesSwappedBelow);
}

// Which of the crop's three main inputs reach DLSS through its subrects (1 colour, 2 depth, 4 vectors): all or none
// (the copy rule; an RR crop always copies). Under g_mutex: raw flags only.
int CropSubrectMask(bool rayReconstruction)
{
    return CropReadsCopies(rayReconstruction) ? 0 : 7;
}

// The parity whose votes say 'left' (neg, pos per parity): at least `least` of each parity, 90% of each one way, the two
// parities opposite; -1 if they do not.
int VotedLeft(const uint64_t votes[2][2], uint64_t least)
{
    for (int p = 0; p < 2; ++p)
    {
        const uint64_t neg = votes[p][0], pos = votes[p][1];
        const uint64_t oNeg = votes[p ^ 1][0], oPos = votes[p ^ 1][1];
        if (neg + pos >= least && oNeg + oPos >= least && neg * 10 >= (neg + pos) * 9 && oPos * 10 >= (oNeg + oPos) * 9)
            return p;
    }
    return -1;
}

// Which parity is the left eye (-1 unknown). Decided once from the launch's votes (20 accepted fits of each parity),
// then KEPT; only a window that clearly votes the other way changes it (ReviewEyeDecision). The
// cumulative test used to be asked every time, and ~20 s of fits in the game's menu with ONE sign for both parities ('even +5
// -197, odd +0 -201') left the even parity under 90% for the rest of the launch: undecided, the fit never skipped
// again.
int g_leftDecided = -1;
uint32_t g_eyeMonoWindows = 0; // since start: windows whose fits voted one sign for both parities (ignored)
int LeftParity()
{
    if (g_leftDecided >= 0)
        return g_leftDecided;
    const int p = VotedLeft(g_eyeVotes, 20);
    if (p >= 0)
    {
        g_leftDecided = p;
        LOG_INFO("Per-eye DLSS DX12: the left eye is the {} parity (decided from {} + {} votes; kept until a window "
                 "clearly votes the other way)",
                 p ? "odd" : "even", g_eyeVotes[p][0] + g_eyeVotes[p][1], g_eyeVotes[p ^ 1][0] + g_eyeVotes[p ^ 1][1]);
    }
    return p;
}

// At each window's end: a decided eye changes only on a window that votes the other way as clearly (50 fits of each
// parity); a window with one sign for both parities (a mono picture, the game's menu) is ignored and counted.
void ReviewEyeDecision()
{
    const uint64_t n0 = g_parallaxSign[0][0] + g_parallaxSign[0][1], n1 = g_parallaxSign[1][0] + g_parallaxSign[1][1];
    if (n0 < 50 || n1 < 50)
        return;
    const bool sameSign = (g_parallaxSign[0][0] * 10 >= n0 * 9 && g_parallaxSign[1][0] * 10 >= n1 * 9) ||
                          (g_parallaxSign[0][1] * 10 >= n0 * 9 && g_parallaxSign[1][1] * 10 >= n1 * 9);
    if (sameSign)
    {
        ++g_eyeMonoWindows;
        LOG_INFO("Per-eye DLSS DX12: a window whose fits voted ONE sign for both parities (even +{} -{}, odd +{} -{}): "
                 "not an eye vote, ignored ({} such windows since start)",
                 g_parallaxSign[0][1], g_parallaxSign[0][0], g_parallaxSign[1][1], g_parallaxSign[1][0],
                 g_eyeMonoWindows);
        return;
    }
    const int voted = VotedLeft(g_parallaxSign, 50);
    if (g_leftDecided >= 0 && voted >= 0 && voted != g_leftDecided)
    {
        LOG_INFO("Per-eye DLSS DX12: the left eye CHANGES to the {} parity: this window voted it clearly (even +{} -{}, "
                 "odd +{} -{})",
                 voted ? "odd" : "even", g_parallaxSign[0][1], g_parallaxSign[0][0], g_parallaxSign[1][1],
                 g_parallaxSign[1][0]);
        g_leftDecided = voted;
    }
}
unsigned int g_foveaCropNow = 0;
bool g_foveaSetupTried = false, g_foveaReady = false;
ID3D12RootSignature* g_compositeRoot = nullptr;
ID3D12PipelineState* g_compositePso = nullptr;
ID3D12PipelineState* g_gazeMarkPso = nullptr; // optional: no marker if it fails
bool g_loggedGazeSkip = false, g_loggedGazeFirst = false, g_loggedGazeProjection = false, g_loggedGazeEyeUnknown = false;
int g_lastGazeEye = -1; // 0 left, 1 right, 2 both (eye unknown)
ID3D12DescriptorHeap* g_compositeHeap = nullptr; // kRing x (crop SRV, age SRV, output UAV)
uint64_t g_compositeSeq = 0;
ID3D12Resource* g_cropTex = nullptr; // ours: the crop feature's output
D3D12_RESOURCE_STATES g_cropTexState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
UINT64 g_cropTexW = 0;
UINT g_cropTexH = 0;
DXGI_FORMAT g_cropTexFormat = DXGI_FORMAT_UNKNOWN;
uint64_t g_foveaEvals = 0, g_foveaSkips = 0, g_foveaFails = 0, g_foveaRebuildFrames = 0;
double g_lastUpscalerMs = 0, g_lastCropMs = 0; // the last window's means, for the overlay
bool g_loggedFoveaSkip = false, g_loggedFoveaFirst = false, g_loggedFoveaFail = false, g_loggedFoveaPlan = false;
bool g_loggedFoveaNotReady = false, g_loggedPeriphNotReady = false;

// The parameters a crop evaluate changes, saved and handed back after it.
struct SavedParams
{
    unsigned int colorX = 0, colorY = 0, mvX = 0, mvY = 0, depthX = 0, depthY = 0, subW = 0, subH = 0;
    unsigned int width = 0, height = 0, outW = 0, outH = 0, outX = 0, outY = 0;
    ID3D12Resource* output = nullptr;
    int reset = 0;
    bool gathered = false; // the crop read our copies: Color / MotionVectors / Depth below go back
    int sub = 0;           // which inputs were handed through subrects all the same (1 colour, 2 depth, 4 vectors)
    bool shifted = false;  // the crop read our composed vectors shifted by its move (g_gShifted, through subrects)
    bool rr = false;       // a ray reconstruction crop: its guides and the projection go back too
    ID3D12Resource* colour = nullptr;
    ID3D12Resource* motion = nullptr;
    ID3D12Resource* depth = nullptr;
    bool guides = false;             // an SR crop whose bias mask was copied through the guide path: goes back
    bool maskDropped = false;        // an SR crop handed no bias mask: the game's goes back
    ID3D12Resource* mask = nullptr;
    float mvOffX = 0.0f, mvOffY = 0.0f; // the block's MV offset before the crop's evaluate: goes back after
    float offHandedX = 0.0f, offHandedY = 0.0f; // what the crop's evaluate was handed (the probe compares)
};
SavedParams g_saved;

unsigned int Gcd(unsigned int a, unsigned int b)
{
    while (b != 0)
    {
        const unsigned int t = a % b;
        a = b;
        b = t;
    }
    return a;
}

// One axis of the crop. An exact size (the crop's render and output boxes in the same ratio as the whole eye, on
// whole pixels at both ends) is taken only within 0.02 of the requested fraction; otherwise the asked size, its width
// chosen within 1% of the eye so that it scales to nearly whole output pixels, edges up to half an output pixel off
// (logged). Exact sizes are few (2056 -> 3472: only 0.25, 0.5, 0.75 of the eye); within 0.15, as it used to be,
// every size from 0.20 to 0.37 became 0.25 in Crimson Desert (below 35x35 the crop stopped
// shrinking). A crop that follows the gaze -- which gives exactness up at its first move anyway -- does not snap
// at all (`snap` false).
bool CropAxis(unsigned int R, unsigned int O, float f, bool snap, uint32_t& rb, uint32_t& rw, uint32_t& ob,
              uint32_t& ow, bool& exact)
{
    if (R == 0 || O < R || f <= 0.05f || f >= 0.95f)
        return false;
    const unsigned int g = Gcd(R, O), ur = R / g, uo = O / g;
    float bestDiff = 1e9f;
    for (unsigned int k = 1; snap && k < g; ++k)
    {
        const unsigned int w = k * ur;
        if ((R - w) % (2 * ur) != 0)
            continue;
        const float diff = std::abs((float) w / (float) R - f);
        if (diff < bestDiff && diff <= 0.02f)
        {
            bestDiff = diff;
            rw = w;
            rb = (R - w) / 2;
            ow = k * uo;
            ob = (rb / ur) * uo;
        }
    }
    exact = bestDiff < 1e9f;
    if (exact)
        return true;
    rw = ((unsigned int) std::lround(f * (float) R)) & ~1u;
    // Not exact: the near-whole-output-pixel width, snapping or not (it used to be the gaze's only).
    {
        const long want = (long) rw, span = std::max(2L, (long) R / 100);
        double bestErr = 1e9;
        for (long w = std::max(16L, want - span) & ~1L; w <= std::min((long) R, want + span); w += 2)
        {
            const double o = (double) w * O / R, err = std::abs(o - std::round(o)) + 1e-4 * std::abs(w - want);
            if (err < bestErr)
            {
                bestErr = err;
                rw = (uint32_t) w;
            }
        }
    }
    rb = (R - rw) / 2;
    ow = (unsigned int) std::lround((double) rw * O / R);
    ob = (unsigned int) std::lround((double) rb * O / R);
    return rw >= 16;
}

bool SetupFovea(ID3D12Device* device, DXGI_FORMAT outputFormat)
{
    g_foveaSetupTried = true;
    // Blending the feathered edge reads the game's output through a typed UAV: an optional D3D12 capability.
    D3D12_FEATURE_DATA_FORMAT_SUPPORT fs { outputFormat };
    if (FAILED(device->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &fs, sizeof(fs))) ||
        !(fs.Support2 & D3D12_FORMAT_SUPPORT2_UAV_TYPED_LOAD) || !(fs.Support2 & D3D12_FORMAT_SUPPORT2_UAV_TYPED_STORE))
    {
        LogOnce(g_loggedFoveaFail, std::format("foveal crop off: the output format {} has no typed UAV load/store here",
                                               (int) outputFormat));
        return false;
    }
    D3D12_DESCRIPTOR_RANGE ranges[2] {};
    ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    ranges[0].NumDescriptors = 2; // the crop (t0) and its ages (t1, AGE_FADE)
    ranges[0].OffsetInDescriptorsFromTableStart = 0;
    ranges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    ranges[1].NumDescriptors = 1;
    ranges[1].OffsetInDescriptorsFromTableStart = 2;
    D3D12_ROOT_PARAMETER params[2] {};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[0].Constants.Num32BitValues = 8;
    params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[1].DescriptorTable.NumDescriptorRanges = 2;
    params[1].DescriptorTable.pDescriptorRanges = ranges;
    params[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    D3D12_ROOT_SIGNATURE_DESC rsDesc {};
    rsDesc.NumParameters = 2;
    rsDesc.pParameters = params;
    ID3DBlob* rsBlob = nullptr;
    ID3DBlob* rsError = nullptr;
    if (FAILED(D3D12SerializeRootSignature(&rsDesc, D3D_ROOT_SIGNATURE_VERSION_1, &rsBlob, &rsError)) ||
        rsBlob == nullptr)
    {
        if (rsError)
            rsError->Release();
        LogOnce(g_loggedFoveaFail, "foveal crop off: the composite root signature did not serialise");
        return false;
    }
    const HRESULT rsHr = device->CreateRootSignature(0, rsBlob->GetBufferPointer(), rsBlob->GetBufferSize(),
                                                     IID_PPV_ARGS(&g_compositeRoot));
    rsBlob->Release();
    if (FAILED(rsHr))
    {
        LogOnce(g_loggedFoveaFail, "foveal crop off: CreateRootSignature failed");
        return false;
    }
    // AGE_FADE: the D3D12 composite reads the crop's ages (t1); the Vulkan build of the same source does not.
    const std::string compositeSource = std::string("#define AGE_FADE 1\n") + kCompositeShader;
    ID3DBlob* cs = CompileShader(compositeSource.c_str(), "CSMain", "cs_5_0");
    if (cs == nullptr)
    {
        LogOnce(g_loggedFoveaFail, "foveal crop off: the composite shader did not compile (see above)");
        return false;
    }
    D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc {};
    psoDesc.pRootSignature = g_compositeRoot;
    psoDesc.CS = { cs->GetBufferPointer(), cs->GetBufferSize() };
    const HRESULT psoHr = device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&g_compositePso));
    cs->Release();
    if (FAILED(psoHr))
    {
        LogOnce(g_loggedFoveaFail, "foveal crop off: CreateComputePipelineState (composite) failed");
        return false;
    }
    if (VarjoGaze::Enabled())
    {
        if (ID3DBlob* mark = CompileShader(kGazeMarkShader, "CSMain", "cs_5_0"); mark != nullptr)
        {
            psoDesc.CS = { mark->GetBufferPointer(), mark->GetBufferSize() };
            if (FAILED(device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&g_gazeMarkPso))))
                g_gazeMarkPso = nullptr;
            mark->Release();
        }
        LOG_INFO("Per-eye DLSS DX12: gaze marker {}", g_gazeMarkPso ? "ready (drawn with the outline, F8)"
                                                                     : "FAILED to build; no marker");
    }
    D3D12_DESCRIPTOR_HEAP_DESC heapDesc {};
    heapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    heapDesc.NumDescriptors = kRing * 3; // crop SRV, age SRV, output UAV
    heapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    if (FAILED(device->CreateDescriptorHeap(&heapDesc, IID_PPV_ARGS(&g_compositeHeap))))
    {
        LogOnce(g_loggedFoveaFail, "foveal crop off: no composite descriptor heap");
        return false;
    }
    if (g_descriptorSize == 0)
        g_descriptorSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    g_foveaReady = true;
    LOG_INFO("Per-eye DLSS DX12: foveal crop composite ready (output format {})", (int) outputFormat);
    return true;
}

DXGI_FORMAT TypedFormat(DXGI_FORMAT format)
{
    switch (format)
    {
    case DXGI_FORMAT_R16G16B16A16_TYPELESS:
        return DXGI_FORMAT_R16G16B16A16_FLOAT;
    case DXGI_FORMAT_R8G8B8A8_TYPELESS:
        return DXGI_FORMAT_R8G8B8A8_UNORM;
    case DXGI_FORMAT_R10G10B10A2_TYPELESS:
        return DXGI_FORMAT_R10G10B10A2_UNORM;
    case DXGI_FORMAT_R32G32B32A32_TYPELESS:
        return DXGI_FORMAT_R32G32B32A32_FLOAT;
    default:
        return format;
    }
}

// Our crop output texture (the crop feature's output), at the crop's output size and the game's output format.
bool EnsureCropTex(ID3D12Device* device, uint32_t w, uint32_t h, DXGI_FORMAT format, uint64_t present)
{
    if (g_cropTex != nullptr && g_cropTexW == w && g_cropTexH == h && g_cropTexFormat == format)
        return true;
    if (g_cropTex != nullptr)
    {
        g_retired.push_back({ g_cropTex, present }); // work recorded a moment ago can still reference it
        g_cropTex = nullptr;
    }
    D3D12_RESOURCE_DESC desc {};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = w;
    desc.Height = h;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = format;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    D3D12_HEAP_PROPERTIES heap {};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    if (FAILED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
                                               D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&g_cropTex))))
    {
        g_cropTex = nullptr;
        return false;
    }
    g_cropTex->SetName(L"PerEyeDlss_Dx12 foveal crop output");
    g_cropTexState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    g_cropTexW = w;
    g_cropTexH = h;
    g_cropTexFormat = format;
    LOG_INFO("Per-eye DLSS DX12: foveal crop output texture {}x{} fmt {} made", w, h, (int) format);
    return true;
}

// GATHER: the crop's inputs copied into our own crop-sized textures, so the crop's DLSS reads them from (0, 0) like any
// ordinary call. Reading them through DLSS's input subrects at an offset worked for presets E, L and M but left K and J
// pixelated in Crimson Desert (history thrown away every frame -- as if one input were read at the wrong place).
constexpr char kGatherShader[] = R"(
Texture2D<float4> Colour : register(t0);
Texture2D<float2> Motion : register(t1);
Texture2D<float> Depth : register(t2);
RWTexture2D<float4> OutColour : register(u0);
RWTexture2D<float2> OutMotion : register(u1);
RWTexture2D<float> OutDepth : register(u2);

cbuffer Constants : register(b0)
{
    uint2 ColourBase;
    uint2 MotionBase;
    uint2 DepthBase;
    uint2 Size;
    float2 MotionShift; // a crop that moved since its last evaluate: the move, in the vectors' own units
};

[numthreads(8, 8, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= Size.x || id.y >= Size.y)
        return;
    OutColour[id.xy] = Colour[ColourBase + id.xy];
    OutMotion[id.xy] = Motion[MotionBase + id.xy] + MotionShift;
    OutDepth[id.xy] = Depth[DepthBase + id.xy];
}
)";

struct GatherTex
{
    ID3D12Resource* res = nullptr;
    D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    UINT64 w = 0;
    UINT h = 0;
    DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;
};
GatherTex g_gColour, g_gMotion, g_gDepth;
bool g_gatherTried = false, g_gatherReady = false, g_loggedGather = false, g_loggedGatherSkip = false;
ID3D12RootSignature* g_gatherRoot = nullptr;
ID3D12PipelineState* g_gatherPso = nullptr;
ID3D12DescriptorHeap* g_gatherHeap = nullptr; // kRing x (3 SRV, 3 UAV)
uint64_t g_gatherSeq = 0;

bool SetupGather(ID3D12Device* device)
{
    g_gatherTried = true;
    D3D12_DESCRIPTOR_RANGE ranges[2] {};
    ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    ranges[0].NumDescriptors = 3;
    ranges[0].OffsetInDescriptorsFromTableStart = 0;
    ranges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    ranges[1].NumDescriptors = 3;
    ranges[1].OffsetInDescriptorsFromTableStart = 3;
    D3D12_ROOT_PARAMETER params[2] {};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[0].Constants.Num32BitValues = 10;
    params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[1].DescriptorTable.NumDescriptorRanges = 2;
    params[1].DescriptorTable.pDescriptorRanges = ranges;
    params[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    D3D12_ROOT_SIGNATURE_DESC rsDesc {};
    rsDesc.NumParameters = 2;
    rsDesc.pParameters = params;
    ID3DBlob* rsBlob = nullptr;
    ID3DBlob* rsError = nullptr;
    if (FAILED(D3D12SerializeRootSignature(&rsDesc, D3D_ROOT_SIGNATURE_VERSION_1, &rsBlob, &rsError)) ||
        rsBlob == nullptr)
    {
        if (rsError)
            rsError->Release();
        LogOnce(g_loggedGatherSkip, "crop input copies off: the gather root signature did not serialise; subrects used");
        return false;
    }
    const HRESULT rsHr = device->CreateRootSignature(0, rsBlob->GetBufferPointer(), rsBlob->GetBufferSize(),
                                                     IID_PPV_ARGS(&g_gatherRoot));
    rsBlob->Release();
    if (FAILED(rsHr))
    {
        LogOnce(g_loggedGatherSkip, "crop input copies off: CreateRootSignature failed; subrects used");
        return false;
    }
    ID3DBlob* cs = CompileShader(kGatherShader, "CSMain", "cs_5_0");
    if (cs == nullptr)
    {
        LogOnce(g_loggedGatherSkip, "crop input copies off: the gather shader did not compile; subrects used");
        return false;
    }
    D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc {};
    psoDesc.pRootSignature = g_gatherRoot;
    psoDesc.CS = { cs->GetBufferPointer(), cs->GetBufferSize() };
    const HRESULT psoHr = device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&g_gatherPso));
    cs->Release();
    if (FAILED(psoHr))
    {
        LogOnce(g_loggedGatherSkip, "crop input copies off: CreateComputePipelineState (gather) failed; subrects used");
        return false;
    }
    D3D12_DESCRIPTOR_HEAP_DESC heapDesc {};
    heapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    heapDesc.NumDescriptors = kRing * 6;
    heapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    if (FAILED(device->CreateDescriptorHeap(&heapDesc, IID_PPV_ARGS(&g_gatherHeap))))
    {
        LogOnce(g_loggedGatherSkip, "crop input copies off: no gather descriptor heap; subrects used");
        return false;
    }
    g_gatherReady = true;
    return true;
}

bool EnsureGatherTex(ID3D12Device* device, GatherTex& t, UINT64 w, UINT h, DXGI_FORMAT format, const wchar_t* name,
                     uint64_t present)
{
    if (t.res != nullptr && t.w == w && t.h == h && t.format == format)
        return true;
    if (t.res != nullptr)
    {
        g_retired.push_back({ t.res, present }); // work recorded a moment ago can still reference it
        t.res = nullptr;
    }
    D3D12_RESOURCE_DESC desc {};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = w;
    desc.Height = h;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = format;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    D3D12_HEAP_PROPERTIES heap {};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    if (FAILED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
                                               D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&t.res))))
    {
        t.res = nullptr;
        return false;
    }
    t.res->SetName(name);
    t.state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    t.w = w;
    t.h = h;
    t.format = format;
    return true;
}

void GatherState(ID3D12GraphicsCommandList* cmdList, GatherTex& t, D3D12_RESOURCE_STATES state)
{
    if (t.state == state)
        return;
    Transition(cmdList, t.res, t.state, state);
    t.state = state;
}

// Copies the crop box of colour, motion and depth into our textures; false = use the subrects (reason logged once).
// `motionShift`: added to every copied vector (0, 0 unless the crop moved -- see BeginFovea).
bool GatherCropInputs(ID3D12GraphicsCommandList* cmdList, ID3D12Resource* colour, ID3D12Resource* motion,
                      ID3D12Resource* depth, const uint32_t colourBase[2], const uint32_t motionBase[2],
                      const uint32_t depthBase[2], uint32_t w, uint32_t h, const float motionShift[2])
{
    if (!g_gatherTried)
        SetupGather(g_device);
    if (!g_gatherReady)
        return false;
    if (colour == nullptr || motion == nullptr || depth == nullptr)
    {
        LogOnce(g_loggedGatherSkip, "crop input copies skipped: a missing input; subrects used");
        return false;
    }
    const D3D12_RESOURCE_DESC cd = colour->GetDesc(), md = motion->GetDesc(), dd = depth->GetDesc();
    auto fits = [](const D3D12_RESOURCE_DESC& d, const uint32_t base[2], uint32_t w, uint32_t h)
    { return d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D && base[0] + w <= d.Width && base[1] + h <= d.Height; };
    // Motion at the render size only (MVLowRes): full-size vectors would need their own box.
    if (!fits(cd, colourBase, w, h) || !fits(md, motionBase, w, h) || !fits(dd, depthBase, w, h))
    {
        LogOnce(g_loggedGatherSkip, std::format("crop input copies skipped: the box does not fit colour {}x{} / motion "
                                                "{}x{} / depth {}x{}; subrects used",
                                                cd.Width, cd.Height, md.Width, md.Height, dd.Width, dd.Height));
        return false;
    }
    const uint64_t present = g_present.load();
    if (!EnsureGatherTex(g_device, g_gColour, w, h, TypedFormat(cd.Format), L"PerEyeDlss_Dx12 crop colour", present) ||
        !EnsureGatherTex(g_device, g_gMotion, w, h, MotionSrvFormat(md.Format), L"PerEyeDlss_Dx12 crop motion",
                         present) ||
        !EnsureGatherTex(g_device, g_gDepth, w, h, DXGI_FORMAT_R32_FLOAT, L"PerEyeDlss_Dx12 crop depth", present))
    {
        LogOnce(g_loggedGatherSkip, "crop input copies skipped: could not make the copy textures; subrects used");
        return false;
    }
    GatherState(cmdList, g_gColour, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    GatherState(cmdList, g_gMotion, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    GatherState(cmdList, g_gDepth, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    const uint32_t slot = (uint32_t) (g_gatherSeq++ % kRing);
    D3D12_CPU_DESCRIPTOR_HANDLE cpu = g_gatherHeap->GetCPUDescriptorHandleForHeapStart();
    cpu.ptr += (SIZE_T) slot * 6 * g_descriptorSize;
    D3D12_GPU_DESCRIPTOR_HANDLE gpu = g_gatherHeap->GetGPUDescriptorHandleForHeapStart();
    gpu.ptr += (UINT64) slot * 6 * g_descriptorSize;
    D3D12_SHADER_RESOURCE_VIEW_DESC srv {};
    srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srv.Texture2D.MipLevels = 1;
    // The game's inputs (and per eye our own motion) are read as DLSS reads them: no barrier on them.
    srv.Format = TypedFormat(cd.Format);
    g_device->CreateShaderResourceView(colour, &srv, cpu);
    cpu.ptr += g_descriptorSize;
    srv.Format = MotionSrvFormat(md.Format);
    g_device->CreateShaderResourceView(motion, &srv, cpu);
    cpu.ptr += g_descriptorSize;
    srv.Format = DepthSrvFormat(dd.Format);
    g_device->CreateShaderResourceView(depth, &srv, cpu);
    cpu.ptr += g_descriptorSize;
    D3D12_UNORDERED_ACCESS_VIEW_DESC uav {};
    uav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    for (GatherTex* t : { &g_gColour, &g_gMotion, &g_gDepth })
    {
        uav.Format = t->format;
        g_device->CreateUnorderedAccessView(t->res, nullptr, &uav, cpu);
        cpu.ptr += g_descriptorSize;
    }
    uint32_t constants[10] = { colourBase[0], colourBase[1], motionBase[0], motionBase[1], depthBase[0],
                               depthBase[1],  w,             h,             0,             0 };
    std::memcpy(&constants[8], motionShift, 2 * sizeof(float));
    ID3D12DescriptorHeap* heaps[] = { g_gatherHeap };
    cmdList->SetDescriptorHeaps(1, heaps);
    cmdList->SetComputeRootSignature(g_gatherRoot);
    cmdList->SetPipelineState(g_gatherPso);
    cmdList->SetComputeRoot32BitConstants(0, 10, constants, 0);
    cmdList->SetComputeRootDescriptorTable(1, gpu);
    cmdList->Dispatch((w + 7) / 8, (h + 7) / 8, 1);
    // DLSS reads them as shader resources.
    GatherState(cmdList, g_gColour, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    GatherState(cmdList, g_gMotion, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    GatherState(cmdList, g_gDepth, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    if (!g_loggedGather)
    {
        g_loggedGather = true;
        LOG_INFO("Per-eye DLSS DX12: crop inputs COPIED into our own {}x{} textures (colour fmt {}, motion fmt {}, depth "
                 "R32F from fmt {}); the crop's DLSS reads them from (0, 0)",
                 w, h, (int) g_gColour.format, (int) g_gMotion.format, (int) dd.Format);
    }
    return true;
}

// ---- The crop's move carried over THROUGH SUBRECTS ----
//
// A crop that follows the gaze keeps its history in its own coordinates, so a move of the box is a pan DLSS cannot
// see: the move is added to the vectors (BeginFovea). That went into COPIED vectors alone, and something below us
// was seen changing the vectors of a copied, crop-sized evaluate (FoveaCopyRule.h) -- while
// a subrect evaluate came back untouched; without copies every glide step restarted the history (2416 restarts on
// 176 moves in one run, the crop shimmered while the gaze moved). Here: the box of OUR composed vectors (the crop reads them
// in shared routing and per eye alike), shifted, copied into a full-size texture of ours at the same place; the
// crop then reads that texture through the same subrect base. Only our own textures are barriered; the composed
// one is read in the state the compose left it for DLSS (a shader resource), and refused otherwise.
GatherTex g_gShifted;
bool g_shiftTried = false, g_shiftReady = false, g_loggedShiftSkip = false, g_loggedShiftSubrects = false;
ID3D12RootSignature* g_shiftRoot = nullptr;
ID3D12PipelineState* g_shiftPso = nullptr;
ID3D12DescriptorHeap* g_shiftHeap = nullptr; // kRing x (1 SRV, 1 UAV)
uint64_t g_shiftSeq = 0;

bool SetupShift(ID3D12Device* device)
{
    g_shiftTried = true;
    D3D12_DESCRIPTOR_RANGE ranges[2] {};
    ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    ranges[0].NumDescriptors = 1;
    ranges[0].OffsetInDescriptorsFromTableStart = 0;
    ranges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    ranges[1].NumDescriptors = 1;
    ranges[1].OffsetInDescriptorsFromTableStart = 1;
    D3D12_ROOT_PARAMETER params[2] {};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[0].Constants.Num32BitValues = 6;
    params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[1].DescriptorTable.NumDescriptorRanges = 2;
    params[1].DescriptorTable.pDescriptorRanges = ranges;
    params[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    D3D12_ROOT_SIGNATURE_DESC rsDesc {};
    rsDesc.NumParameters = 2;
    rsDesc.pParameters = params;
    ID3DBlob* rsBlob = nullptr;
    ID3DBlob* rsError = nullptr;
    if (FAILED(D3D12SerializeRootSignature(&rsDesc, D3D_ROOT_SIGNATURE_VERSION_1, &rsBlob, &rsError)) ||
        rsBlob == nullptr)
    {
        if (rsError)
            rsError->Release();
        LogOnce(g_loggedShiftSkip, "crop move through subrects off: the shift root signature did not serialise");
        return false;
    }
    const HRESULT rsHr = device->CreateRootSignature(0, rsBlob->GetBufferPointer(), rsBlob->GetBufferSize(),
                                                     IID_PPV_ARGS(&g_shiftRoot));
    rsBlob->Release();
    if (FAILED(rsHr))
    {
        LogOnce(g_loggedShiftSkip, "crop move through subrects off: CreateRootSignature (shift) failed");
        return false;
    }
    ID3DBlob* cs = CompileShader(kShiftShader, "CSMain", "cs_5_0");
    if (cs == nullptr)
    {
        LogOnce(g_loggedShiftSkip, "crop move through subrects off: the shift shader did not compile");
        return false;
    }
    D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc {};
    psoDesc.pRootSignature = g_shiftRoot;
    psoDesc.CS = { cs->GetBufferPointer(), cs->GetBufferSize() };
    const HRESULT psoHr = device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&g_shiftPso));
    cs->Release();
    if (FAILED(psoHr))
    {
        LogOnce(g_loggedShiftSkip, "crop move through subrects off: CreateComputePipelineState (shift) failed");
        return false;
    }
    D3D12_DESCRIPTOR_HEAP_DESC heapDesc {};
    heapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    heapDesc.NumDescriptors = kRing * 2;
    heapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    if (FAILED(device->CreateDescriptorHeap(&heapDesc, IID_PPV_ARGS(&g_shiftHeap))))
    {
        LogOnce(g_loggedShiftSkip, "crop move through subrects off: no shift descriptor heap");
        return false;
    }
    g_shiftReady = true;
    return true;
}

// The box (base, w x h) of `composed` (ours: g_composed, a shader resource since the compose) plus `shift` into
// g_gShifted at the same place; false = not done (reason logged once), the caller restarts the history instead.
bool ShiftComposed(ID3D12GraphicsCommandList* cmdList, ID3D12Resource* composed, const uint32_t base[2], uint32_t w,
                   uint32_t h, const float shift[2])
{
    if (!g_shiftTried)
        SetupShift(g_device);
    if (!g_shiftReady)
        return false;
    if (composed == nullptr || composed != g_composed)
    {
        LogOnce(g_loggedShiftSkip, "crop move through subrects skipped: the crop's vectors are not our composed ones");
        return false;
    }
    if (g_composedState != D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)
    {
        LogOnce(g_loggedShiftSkip, "crop move through subrects skipped: the composed vectors are not in the shader "
                                   "resource state the compose leaves them in");
        return false;
    }
    const D3D12_RESOURCE_DESC md = composed->GetDesc();
    if (md.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || base[0] + w > md.Width || base[1] + h > md.Height)
    {
        LogOnce(g_loggedShiftSkip, std::format("crop move through subrects skipped: the box ({}, {}) {}x{} does not fit "
                                               "the composed vectors {}x{}",
                                               base[0], base[1], w, h, md.Width, md.Height));
        return false;
    }
    if (!EnsureGatherTex(g_device, g_gShifted, md.Width, md.Height, MotionSrvFormat(md.Format),
                         L"PerEyeDlss_Dx12 crop motion shifted (subrects)", g_present.load()))
    {
        LogOnce(g_loggedShiftSkip, "crop move through subrects skipped: could not make the shifted texture");
        return false;
    }
    GatherState(cmdList, g_gShifted, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    const uint32_t slot = (uint32_t) (g_shiftSeq++ % kRing);
    D3D12_CPU_DESCRIPTOR_HANDLE cpu = g_shiftHeap->GetCPUDescriptorHandleForHeapStart();
    cpu.ptr += (SIZE_T) slot * 2 * g_descriptorSize;
    D3D12_GPU_DESCRIPTOR_HANDLE gpu = g_shiftHeap->GetGPUDescriptorHandleForHeapStart();
    gpu.ptr += (UINT64) slot * 2 * g_descriptorSize;
    D3D12_SHADER_RESOURCE_VIEW_DESC srv {};
    srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srv.Texture2D.MipLevels = 1;
    srv.Format = MotionSrvFormat(md.Format);
    g_device->CreateShaderResourceView(composed, &srv, cpu);
    cpu.ptr += g_descriptorSize;
    D3D12_UNORDERED_ACCESS_VIEW_DESC uav {};
    uav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    uav.Format = g_gShifted.format;
    g_device->CreateUnorderedAccessView(g_gShifted.res, nullptr, &uav, cpu);
    uint32_t constants[6] = { base[0], base[1], w, h, 0, 0 };
    std::memcpy(&constants[4], shift, 2 * sizeof(float));
    ID3D12DescriptorHeap* heaps[] = { g_shiftHeap };
    cmdList->SetDescriptorHeaps(1, heaps);
    cmdList->SetComputeRootSignature(g_shiftRoot);
    cmdList->SetPipelineState(g_shiftPso);
    cmdList->SetComputeRoot32BitConstants(0, 6, constants, 0);
    cmdList->SetComputeRootDescriptorTable(1, gpu);
    cmdList->Dispatch((w + 7) / 8, (h + 7) / 8, 1);
    GatherState(cmdList, g_gShifted, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE); // DLSS reads it
    return true;
}

// ---- The crop's AGE: the fade-in over the periphery where the crop's picture is new ----
//
// Carrying a move over keeps the crop's history for what was in the box; what came in from outside has none, and a
// far move brings most of the box in new: the crop takes frames to settle there while the periphery under it (its
// own history over the whole eye, never moved) is settled all along (otherwise the image visibly
// settles when the crop moves from one far side to the other). Per crop, a pair of R8_UINT textures at the
// crop's output size: each evaluate, kAgeShader writes the frames every pixel has been in the box (carried along
// with the move in output pixels; 0 where it came in from outside; 0 everywhere on a history reset), and the
// composite shows the output under the crop through a pixel younger than UpscalerFoveaAgeFrames, in proportion.
// Only our own textures are touched; the pass is recorded in EndFovea right before the composite.
struct AgeTex
{
    GatherTex tex[2];
    int cur = -1; // the texture holding this evaluate's ages (-1: none yet)
};
std::unordered_map<unsigned int, AgeTex> g_age;                                     // crop feature -> its ages
std::unordered_map<unsigned int, std::pair<uint32_t, uint32_t>> g_cropOutPos;       // crop feature -> last output box origin
int g_ageShiftOut[2] = { 0, 0 };                                                    // this evaluate's move, output px
bool g_ageResetNow = false;                                                         // this evaluate restarts the history
bool g_ageTried = false, g_ageReady = false, g_loggedAgeSkip = false, g_loggedAgeFirst = false;
ID3D12RootSignature* g_ageRoot = nullptr;
ID3D12PipelineState* g_agePso = nullptr;
ID3D12DescriptorHeap* g_ageHeap = nullptr; // kRing x (1 SRV, 1 UAV)
uint64_t g_ageSeq = 0;
uint64_t g_ageRuns = 0, g_ageResets = 0, g_ageFails = 0;

bool SetupAge(ID3D12Device* device)
{
    g_ageTried = true;
    D3D12_DESCRIPTOR_RANGE ranges[2] {};
    ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    ranges[0].NumDescriptors = 1;
    ranges[0].OffsetInDescriptorsFromTableStart = 0;
    ranges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    ranges[1].NumDescriptors = 1;
    ranges[1].OffsetInDescriptorsFromTableStart = 1;
    D3D12_ROOT_PARAMETER params[2] {};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[0].Constants.Num32BitValues = 8;
    params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[1].DescriptorTable.NumDescriptorRanges = 2;
    params[1].DescriptorTable.pDescriptorRanges = ranges;
    params[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    D3D12_ROOT_SIGNATURE_DESC rsDesc {};
    rsDesc.NumParameters = 2;
    rsDesc.pParameters = params;
    ID3DBlob* rsBlob = nullptr;
    ID3DBlob* rsError = nullptr;
    if (FAILED(D3D12SerializeRootSignature(&rsDesc, D3D_ROOT_SIGNATURE_VERSION_1, &rsBlob, &rsError)) ||
        rsBlob == nullptr)
    {
        if (rsError)
            rsError->Release();
        LogOnce(g_loggedAgeSkip, "crop fade-in off: the age root signature did not serialise");
        return false;
    }
    const HRESULT rsHr = device->CreateRootSignature(0, rsBlob->GetBufferPointer(), rsBlob->GetBufferSize(),
                                                     IID_PPV_ARGS(&g_ageRoot));
    rsBlob->Release();
    if (FAILED(rsHr))
    {
        LogOnce(g_loggedAgeSkip, "crop fade-in off: CreateRootSignature (age) failed");
        return false;
    }
    ID3DBlob* cs = CompileShader(kAgeShader, "CSMain", "cs_5_0");
    if (cs == nullptr)
    {
        LogOnce(g_loggedAgeSkip, "crop fade-in off: the age shader did not compile");
        return false;
    }
    D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc {};
    psoDesc.pRootSignature = g_ageRoot;
    psoDesc.CS = { cs->GetBufferPointer(), cs->GetBufferSize() };
    const HRESULT psoHr = device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&g_agePso));
    cs->Release();
    if (FAILED(psoHr))
    {
        LogOnce(g_loggedAgeSkip, "crop fade-in off: CreateComputePipelineState (age) failed");
        return false;
    }
    D3D12_DESCRIPTOR_HEAP_DESC heapDesc {};
    heapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    heapDesc.NumDescriptors = kRing * 2;
    heapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    if (FAILED(device->CreateDescriptorHeap(&heapDesc, IID_PPV_ARGS(&g_ageHeap))))
    {
        LogOnce(g_loggedAgeSkip, "crop fade-in off: no age descriptor heap");
        return false;
    }
    g_ageReady = true;
    return true;
}

// This evaluate's ages for `crop` at the crop's output size, from the last ones carried along the move (or 0 on a
// reset, or when there are none yet); null = no ages (reason logged once; the composite then fades nothing).
GatherTex* UpdateAge(ID3D12GraphicsCommandList* cmdList, unsigned int crop, UINT64 ow, UINT oh, const int shift[2],
                     bool reset)
{
    if (!g_ageTried)
        SetupAge(g_device);
    if (!g_ageReady)
        return nullptr;
    AgeTex& a = g_age[crop];
    const uint64_t present = g_present.load();
    bool fresh = a.cur < 0;
    for (GatherTex& t : a.tex)
    {
        const bool had = t.res != nullptr && t.w == ow && t.h == oh;
        if (!EnsureGatherTex(g_device, t, ow, oh, DXGI_FORMAT_R8_UINT, L"PerEyeDlss_Dx12 crop age", present))
        {
            LogOnce(g_loggedAgeSkip, "crop fade-in skipped: could not make the age textures");
            ++g_ageFails;
            return nullptr;
        }
        fresh = fresh || !had; // a new size: the old ages are gone
    }
    const int from = fresh ? 0 : a.cur, to = fresh ? 1 : 1 - a.cur;
    const bool resetNow = reset || fresh;
    GatherState(cmdList, a.tex[from], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    GatherState(cmdList, a.tex[to], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    const uint32_t slot = (uint32_t) (g_ageSeq++ % kRing);
    D3D12_CPU_DESCRIPTOR_HANDLE cpu = g_ageHeap->GetCPUDescriptorHandleForHeapStart();
    cpu.ptr += (SIZE_T) slot * 2 * g_descriptorSize;
    D3D12_GPU_DESCRIPTOR_HANDLE gpu = g_ageHeap->GetGPUDescriptorHandleForHeapStart();
    gpu.ptr += (UINT64) slot * 2 * g_descriptorSize;
    D3D12_SHADER_RESOURCE_VIEW_DESC srv {};
    srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srv.Texture2D.MipLevels = 1;
    srv.Format = DXGI_FORMAT_R8_UINT;
    g_device->CreateShaderResourceView(a.tex[from].res, &srv, cpu);
    cpu.ptr += g_descriptorSize;
    D3D12_UNORDERED_ACCESS_VIEW_DESC uav {};
    uav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    uav.Format = DXGI_FORMAT_R8_UINT;
    g_device->CreateUnorderedAccessView(a.tex[to].res, nullptr, &uav, cpu);
    const uint32_t constants[8] = { (uint32_t) ow, oh, (uint32_t) shift[0], (uint32_t) shift[1], resetNow ? 1u : 0u, 0, 0, 0 };
    ID3D12DescriptorHeap* heaps[] = { g_ageHeap };
    cmdList->SetDescriptorHeaps(1, heaps);
    cmdList->SetComputeRootSignature(g_ageRoot);
    cmdList->SetPipelineState(g_agePso);
    cmdList->SetComputeRoot32BitConstants(0, 8, constants, 0);
    cmdList->SetComputeRootDescriptorTable(1, gpu);
    cmdList->Dispatch((uint32_t) ((ow + 7) / 8), (oh + 7) / 8, 1);
    GatherState(cmdList, a.tex[to], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE); // the composite reads it
    a.cur = to;
    ++g_ageRuns;
    if (resetNow)
        ++g_ageResets;
    if (!g_loggedAgeFirst)
    {
        g_loggedAgeFirst = true;
        LOG_INFO("Per-eye DLSS DX12: FIRST crop age pass: crop {} at {}x{} (fade-in over {} frames)", crop, ow, oh,
                 Config::Instance()->UpscalerFoveaAgeFrames.value_or_default());
    }
    return &a.tex[to];
}

unsigned int GetUIntParam(NVSDK_NGX_Parameter* params, const char* name)
{
    unsigned int v = 0;
    params->Get(name, &v);
    return v;
}

// ---- Ray reconstruction crop: the guides and the projection ----
//
// An RR crop needs every guide the game hands RR copied for the box too (nvngx_dlssd reads each through its own subrect
// base, which is exactly what GATHER avoids), and the crop's own projection: RR is handed ViewToClipMatrix, and the
// crop sees only part of that view. Up to 8 guides; a guide that is not at the render size (the box would not fit)
// skips the crop for that evaluate.
constexpr char kGuideShader[] = R"(
Texture2D<float4> In0 : register(t0);
Texture2D<float4> In1 : register(t1);
Texture2D<float4> In2 : register(t2);
Texture2D<float4> In3 : register(t3);
Texture2D<float4> In4 : register(t4);
Texture2D<float4> In5 : register(t5);
Texture2D<float4> In6 : register(t6);
Texture2D<float4> In7 : register(t7);
RWTexture2D<float4> Out0 : register(u0);
RWTexture2D<float4> Out1 : register(u1);
RWTexture2D<float4> Out2 : register(u2);
RWTexture2D<float4> Out3 : register(u3);
RWTexture2D<float4> Out4 : register(u4);
RWTexture2D<float4> Out5 : register(u5);
RWTexture2D<float4> Out6 : register(u6);
RWTexture2D<float4> Out7 : register(u7);

cbuffer Constants : register(b0)
{
    uint4 B01;
    uint4 B23;
    uint4 B45;
    uint4 B67;
    uint2 Size;
    uint Count;
    uint Pad;
};

[numthreads(8, 8, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= Size.x || id.y >= Size.y)
        return;
    const uint2 p = id.xy;
    if (Count > 0) Out0[p] = In0[B01.xy + p];
    if (Count > 1) Out1[p] = In1[B01.zw + p];
    if (Count > 2) Out2[p] = In2[B23.xy + p];
    if (Count > 3) Out3[p] = In3[B23.zw + p];
    if (Count > 4) Out4[p] = In4[B45.xy + p];
    if (Count > 5) Out5[p] = In5[B45.zw + p];
    if (Count > 6) Out6[p] = In6[B67.xy + p];
    if (Count > 7) Out7[p] = In7[B67.zw + p];
}
)";

// RR's resource inputs (names from nvngx_dlssd.dll 310.9.1's strings) and the subrect base each is read at.
struct GuideName
{
    const char* name;
    const char* baseX; // nullptr: read at the colour's base
    const char* baseY;
};
constexpr GuideName kGuides[] = {
    { "DLSS.Input.DiffuseAlbedo", "DLSS.Input.DiffuseAlbedo.Subrect.Base.X", "DLSS.Input.DiffuseAlbedo.Subrect.Base.Y" },
    { "DLSS.Input.SpecularAlbedo", "DLSS.Input.SpecularAlbedo.Subrect.Base.X",
      "DLSS.Input.SpecularAlbedo.Subrect.Base.Y" },
    { "GBuffer.Normals", "DLSS.Input.Normals.Subrect.Base.X", "DLSS.Input.Normals.Subrect.Base.Y" },
    { "GBuffer.Roughness", "DLSS.Input.Roughness.Subrect.Base.X", "DLSS.Input.Roughness.Subrect.Base.Y" },
    { "DLSSD.SpecularHitDistance", "DLSSD.SpecularHitDistance.Subrect.Base.X",
      "DLSSD.SpecularHitDistance.Subrect.Base.Y" },
    { "DLSSD.ScreenSpaceSubsurfaceScatteringGuide", "DLSSD.ScreenSpaceSubsurfaceScatteringGuide.Subrect.Base.X",
      "DLSSD.ScreenSpaceSubsurfaceScatteringGuide.Subrect.Base.Y" },
    { "DLSS.DisocclusionMask", "DLSS.DisocclusionMask.Subrect.Base.X", "DLSS.DisocclusionMask.Subrect.Base.Y" },
    { "DLSSD.ReflectedAlbedo", nullptr, nullptr },
    { "DLSSD.ColorBeforeParticles", nullptr, nullptr },
    { "DLSSD.ColorBeforeTransparency", "DLSSD.ColorBeforeTransparency.Subrect.Base.X",
      "DLSSD.ColorBeforeTransparency.Subrect.Base.Y" },
    { "DLSSD.ColorAfterTransparency", "DLSSD.ColorAfterTransparency.Subrect.Base.X",
      "DLSSD.ColorAfterTransparency.Subrect.Base.Y" },
    { "DLSSD.ColorBeforeFog", "DLSSD.ColorBeforeFog.Subrect.Base.X", "DLSSD.ColorBeforeFog.Subrect.Base.Y" },
    { "DLSSD.DepthOfFieldGuide", "DLSSD.DepthOfFieldGuide.Subrect.Base.X", "DLSSD.DepthOfFieldGuide.Subrect.Base.Y" },
    { "DLSSD.ResponsivityMask", "DLSSD.ResponsivityMask.Subrect.Base.X", "DLSSD.ResponsivityMask.Subrect.Base.Y" },
    { "DLSS.Input.Bias.Current.Color.Mask", "DLSS.Input.Bias.Current.Color.Subrect.Base.X",
      "DLSS.Input.Bias.Current.Color.Subrect.Base.Y" },
    { "DLSS.TransparencyLayer", "DLSS.TransparencyLayer.Subrect.Base.X", "DLSS.TransparencyLayer.Subrect.Base.Y" },
    { "DLSS.TransparencyLayerOpacity", "DLSS.TransparencyLayerOpacity.Subrect.Base.X",
      "DLSS.TransparencyLayerOpacity.Subrect.Base.Y" },
    { "DLSS.TransparencyLayerMvecs", "DLSS.TransparencyLayerMvecs.Subrect.Base.X",
      "DLSS.TransparencyLayerMvecs.Subrect.Base.Y" },
    { "GBuffer.SpecularMvec", nullptr, nullptr },
    { "MotionVectorsReflection", nullptr, nullptr },
    { "IsParticleMask", nullptr, nullptr },
    { "AnimatedTextureMask", nullptr, nullptr },
    { "DLSSD.Alpha", "DLSSD.Alpha.Subrect.Base.X", "DLSSD.Alpha.Subrect.Base.Y" },
};
constexpr int kMaxGuides = 8;

struct GuidePlan
{
    int count = 0;
    const GuideName* names[kMaxGuides] = {};
    ID3D12Resource* src[kMaxGuides] = {};
    uint32_t oldBase[kMaxGuides][2] = {};
    DXGI_FORMAT format[kMaxGuides] = {};
};
GuidePlan g_guides;
GatherTex g_gGuide[kMaxGuides];
bool g_guideTried = false, g_guideReady = false, g_loggedGuides = false, g_loggedGuideSkip = false;
ID3D12RootSignature* g_guideRoot = nullptr;
ID3D12PipelineState* g_guidePso = nullptr;
ID3D12DescriptorHeap* g_guideHeap = nullptr; // kRing x (8 SRV, 8 UAV)
uint64_t g_guideSeq = 0;
// The crop's projection: handed to RR in place of the game's for the crop's evaluate only (RR reads it on the CPU).
float g_cropProj[16] = {};
void* g_savedProj = nullptr;
bool g_projSwapped = false, g_loggedProj = false;

DXGI_FORMAT GuideFormat(DXGI_FORMAT format)
{
    switch (format)
    {
    case DXGI_FORMAT_R32_TYPELESS:
        return DXGI_FORMAT_R32_FLOAT;
    case DXGI_FORMAT_R16_TYPELESS:
        return DXGI_FORMAT_R16_FLOAT;
    case DXGI_FORMAT_R8_TYPELESS:
        return DXGI_FORMAT_R8_UNORM;
    case DXGI_FORMAT_R16G16_TYPELESS:
        return DXGI_FORMAT_R16G16_FLOAT;
    default:
        return TypedFormat(format);
    }
}

bool SetupGuides(ID3D12Device* device)
{
    g_guideTried = true;
    D3D12_DESCRIPTOR_RANGE ranges[2] {};
    ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    ranges[0].NumDescriptors = kMaxGuides;
    ranges[0].OffsetInDescriptorsFromTableStart = 0;
    ranges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    ranges[1].NumDescriptors = kMaxGuides;
    ranges[1].OffsetInDescriptorsFromTableStart = kMaxGuides;
    D3D12_ROOT_PARAMETER params[2] {};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[0].Constants.Num32BitValues = 20;
    params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[1].DescriptorTable.NumDescriptorRanges = 2;
    params[1].DescriptorTable.pDescriptorRanges = ranges;
    params[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    D3D12_ROOT_SIGNATURE_DESC rsDesc {};
    rsDesc.NumParameters = 2;
    rsDesc.pParameters = params;
    ID3DBlob* rsBlob = nullptr;
    ID3DBlob* rsError = nullptr;
    if (FAILED(D3D12SerializeRootSignature(&rsDesc, D3D_ROOT_SIGNATURE_VERSION_1, &rsBlob, &rsError)) ||
        rsBlob == nullptr)
    {
        if (rsError)
            rsError->Release();
        LogOnce(g_loggedGuideSkip, "RR crop off: the guide copy root signature did not serialise");
        return false;
    }
    const HRESULT rsHr = device->CreateRootSignature(0, rsBlob->GetBufferPointer(), rsBlob->GetBufferSize(),
                                                     IID_PPV_ARGS(&g_guideRoot));
    rsBlob->Release();
    if (FAILED(rsHr))
    {
        LogOnce(g_loggedGuideSkip, "RR crop off: CreateRootSignature (guides) failed");
        return false;
    }
    ID3DBlob* cs = CompileShader(kGuideShader, "CSMain", "cs_5_0");
    if (cs == nullptr)
    {
        LogOnce(g_loggedGuideSkip, "RR crop off: the guide copy shader did not compile (see above)");
        return false;
    }
    D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc {};
    psoDesc.pRootSignature = g_guideRoot;
    psoDesc.CS = { cs->GetBufferPointer(), cs->GetBufferSize() };
    const HRESULT psoHr = device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&g_guidePso));
    cs->Release();
    if (FAILED(psoHr))
    {
        LogOnce(g_loggedGuideSkip, "RR crop off: CreateComputePipelineState (guides) failed");
        return false;
    }
    D3D12_DESCRIPTOR_HEAP_DESC heapDesc {};
    heapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    heapDesc.NumDescriptors = kRing * kMaxGuides * 2;
    heapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    if (FAILED(device->CreateDescriptorHeap(&heapDesc, IID_PPV_ARGS(&g_guideHeap))))
    {
        LogOnce(g_loggedGuideSkip, "RR crop off: no guide descriptor heap");
        return false;
    }
    g_guideReady = true;
    return true;
}

// Which guides this evaluate sets and whether the box fits each: nothing recorded, nothing changed. False = no RR crop.
bool PlanGuides(NVSDK_NGX_Parameter* params, const uint32_t colourBase[2], uint32_t bx, uint32_t by, uint32_t w,
                uint32_t h)
{
    GuidePlan plan {};
    for (const auto& g : kGuides)
    {
        ID3D12Resource* res = GetResource(params, g.name);
        if (res == nullptr)
            continue;
        if (plan.count == kMaxGuides)
        {
            LogOnce(g_loggedGuideSkip, std::format("RR crop skipped: more than {} RR guides set (counted as skips)",
                                                   kMaxGuides));
            return false;
        }
        const D3D12_RESOURCE_DESC d = res->GetDesc();
        uint32_t base[2] = { colourBase[0], colourBase[1] };
        if (g.baseX != nullptr)
        {
            base[0] = GetUIntParam(params, g.baseX);
            base[1] = GetUIntParam(params, g.baseY);
        }
        if (d.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || base[0] + bx + w > d.Width ||
            base[1] + by + h > d.Height)
        {
            LogOnce(g_loggedGuideSkip, std::format("RR crop skipped: guide {} is {}x{}, the render box does not fit it "
                                                   "(counted as skips)",
                                                   g.name, d.Width, d.Height));
            return false;
        }
        plan.names[plan.count] = &g;
        plan.src[plan.count] = res;
        plan.oldBase[plan.count][0] = base[0];
        plan.oldBase[plan.count][1] = base[1];
        plan.format[plan.count] = GuideFormat(d.Format);
        ++plan.count;
    }
    g_guides = plan;
    return true;
}

// Copies the planned guides' boxes into our textures and hands them to the crop's evaluate (bases 0).
bool GatherGuides(ID3D12GraphicsCommandList* cmdList, NVSDK_NGX_Parameter* params, uint32_t bx, uint32_t by,
                  uint32_t w, uint32_t h)
{
    if (!g_guideTried)
        SetupGuides(g_device);
    if (!g_guideReady)
        return false;
    auto& plan = g_guides;
    const uint64_t present = g_present.load();
    for (int i = 0; i < plan.count; ++i)
        if (!EnsureGatherTex(g_device, g_gGuide[i], w, h, plan.format[i], L"PerEyeDlss_Dx12 crop RR guide", present))
        {
            LogOnce(g_loggedGuideSkip, "RR crop skipped: could not make a guide copy texture");
            return false;
        }
    for (int i = 0; i < plan.count; ++i)
        GatherState(cmdList, g_gGuide[i], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    const uint32_t slot = (uint32_t) (g_guideSeq++ % kRing);
    D3D12_CPU_DESCRIPTOR_HANDLE cpu = g_guideHeap->GetCPUDescriptorHandleForHeapStart();
    cpu.ptr += (SIZE_T) slot * kMaxGuides * 2 * g_descriptorSize;
    D3D12_GPU_DESCRIPTOR_HANDLE gpu = g_guideHeap->GetGPUDescriptorHandleForHeapStart();
    gpu.ptr += (UINT64) slot * kMaxGuides * 2 * g_descriptorSize;
    for (int i = 0; i < kMaxGuides; ++i)
    {
        // Unused slots get null descriptors (never read: the shader stops at Count).
        D3D12_SHADER_RESOURCE_VIEW_DESC srv {};
        srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srv.Texture2D.MipLevels = 1;
        srv.Format = i < plan.count ? plan.format[i] : DXGI_FORMAT_R8G8B8A8_UNORM;
        g_device->CreateShaderResourceView(i < plan.count ? plan.src[i] : nullptr, &srv, cpu);
        cpu.ptr += g_descriptorSize;
    }
    for (int i = 0; i < kMaxGuides; ++i)
    {
        D3D12_UNORDERED_ACCESS_VIEW_DESC uav {};
        uav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
        uav.Format = i < plan.count ? plan.format[i] : DXGI_FORMAT_R8G8B8A8_UNORM;
        g_device->CreateUnorderedAccessView(i < plan.count ? g_gGuide[i].res : nullptr, nullptr, &uav, cpu);
        cpu.ptr += g_descriptorSize;
    }
    uint32_t constants[20] = {};
    for (int i = 0; i < plan.count; ++i)
    {
        constants[i * 2 + 0] = plan.oldBase[i][0] + bx;
        constants[i * 2 + 1] = plan.oldBase[i][1] + by;
    }
    constants[16] = w;
    constants[17] = h;
    constants[18] = (uint32_t) plan.count;
    ID3D12DescriptorHeap* heaps[] = { g_guideHeap };
    cmdList->SetDescriptorHeaps(1, heaps);
    cmdList->SetComputeRootSignature(g_guideRoot);
    cmdList->SetPipelineState(g_guidePso);
    cmdList->SetComputeRoot32BitConstants(0, 20, constants, 0);
    cmdList->SetComputeRootDescriptorTable(1, gpu);
    cmdList->Dispatch((w + 7) / 8, (h + 7) / 8, 1);
    for (int i = 0; i < plan.count; ++i)
    {
        GatherState(cmdList, g_gGuide[i], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        params->Set(plan.names[i]->name, g_gGuide[i].res);
        if (plan.names[i]->baseX != nullptr)
        {
            params->Set(plan.names[i]->baseX, 0u);
            params->Set(plan.names[i]->baseY, 0u);
        }
    }
    if (!g_loggedGuides)
    {
        g_loggedGuides = true;
        std::string list;
        for (int i = 0; i < plan.count; ++i)
            list += std::format("{} (fmt {}) ", plan.names[i]->name, (int) plan.format[i]);
        LOG_INFO("Per-eye DLSS DX12: RR crop: {} guide(s) copied for the box: {}", plan.count, list);
    }
    return true;
}

void RestoreGuides(NVSDK_NGX_Parameter* params)
{
    auto& plan = g_guides;
    for (int i = 0; i < plan.count; ++i)
    {
        params->Set(plan.names[i]->name, plan.src[i]);
        if (plan.names[i]->baseX != nullptr)
        {
            params->Set(plan.names[i]->baseX, plan.oldBase[i][0]);
            params->Set(plan.names[i]->baseY, plan.oldBase[i][1]);
        }
    }
    plan.count = 0;
}

// The crop's projection from the eye's: its clip x / y re-centred and re-scaled to the box. Both conventions are
// recognised by where w comes from (the view's z): row vectors (v * M, w = column 3) or column vectors (M * v, w = row
// 3). Anything else keeps the eye's projection (logged once).
bool CropProjection(const float* m, const FoveaRect& r, float* out)
{
    const float W = (float) r.renderW, H = (float) r.renderH;
    const float xc = (r.rx + r.rw * 0.5f) / W * 2.0f - 1.0f, hx = r.rw / W;
    const float yc = 1.0f - (r.ry + r.rh * 0.5f) / H * 2.0f, hy = r.rh / H;
    std::memcpy(out, m, 16 * sizeof(float));
    const bool rowVectors = std::abs(std::abs(m[11]) - 1.0f) < 1e-4f && m[3] == 0 && m[7] == 0 && m[15] == 0;
    const bool colVectors = std::abs(std::abs(m[14]) - 1.0f) < 1e-4f && m[12] == 0 && m[13] == 0 && m[15] == 0;
    if (rowVectors)
    {
        for (int k = 0; k < 4; ++k)
        {
            out[k * 4 + 0] = (m[k * 4 + 0] - xc * m[k * 4 + 3]) / hx;
            out[k * 4 + 1] = (m[k * 4 + 1] - yc * m[k * 4 + 3]) / hy;
        }
        return true;
    }
    if (colVectors)
    {
        for (int k = 0; k < 4; ++k)
        {
            out[0 * 4 + k] = (m[0 * 4 + k] - xc * m[3 * 4 + k]) / hx;
            out[1 * 4 + k] = (m[1 * 4 + k] - yc * m[3 * 4 + k]) / hy;
        }
        return true;
    }
    return false;
}

// ---- Cheap periphery (foveated DLSS, step 3) ----
//
// With the crop on, the game's full pass (render size -> eye size) is replaced by a DLSS pass at the render size in and
// out (its own preset, E by default) into our texture, ENLARGEd to the eye size with a Catmull-Rom filter; the crop is
// then drawn over the centre as before. DLSS's cost follows its output size, so the periphery costs about
// (render / eye)^2 of the full pass. A frame whose periphery cannot run (a rebuild, a failure) takes the game's full pass;
// the game's feature, skipped meanwhile, then starts its history over.

bool g_periphTried = false, g_periphReady = false, g_periphFailed = false;
ID3D12RootSignature* g_enlargeRoot = nullptr;
ID3D12PipelineState* g_enlargePso = nullptr;
ID3D12DescriptorHeap* g_enlargeHeap = nullptr; // kRing x (SRV, UAV)
uint64_t g_enlargeSeq = 0;
ID3D12Resource* g_periphTex = nullptr; // ours: the periphery feature's output, render size
D3D12_RESOURCE_STATES g_periphTexState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
UINT64 g_periphTexW = 0;
UINT g_periphTexH = 0;
DXGI_FORMAT g_periphTexFormat = DXGI_FORMAT_UNKNOWN;
std::unordered_map<unsigned int, unsigned int> g_periphs;                          // feature -> its periphery feature
std::unordered_map<unsigned int, std::pair<unsigned int, unsigned int>> g_periphSize; // periphery feature -> render size
std::unordered_map<unsigned int, bool> g_periphReset;
// A periphery feature whose evaluates fail from its FIRST one is remade ONCE through the live-rebuild path (the one
// the 'Periphery preset' row uses). Crimson Desert: in that RR game the first SR feature ever made
// (ours, the periphery) answered every evaluate with InvalidParameter (made with ~100 MB less than every later
// one), and the game's own pass on the same command list then failed too;
// every SR feature made after it was fine, and turning 'SR for RR' off and on remade it. The cause in the first SR
// create is not known: this is the workaround, counted, so the log says when it ran.
std::unordered_map<unsigned int, bool> g_periphEverOk; // a periphery that evaluated successfully at least once
std::unordered_map<unsigned int, int> g_periphRemade;  // remakes after a failed first evaluate, per periphery (at most 1)
uint64_t g_periphRemadeTotal = 0;                      // since start
std::unordered_map<unsigned int, bool> g_mainSkipped; // game features skipped for a periphery: reset on their next evaluate
SavedParams g_savedP;
unsigned int g_periphNow = 0;
unsigned int g_periphOwnerNow = 0;
uint64_t g_periphEvals = 0, g_periphFallbacks = 0;
bool g_loggedPeriphFirst = false, g_loggedPeriphFail = false, g_loggedPeriphSkip = false;

bool SetupPeriphery(ID3D12Device* device)
{
    g_periphTried = true;
    D3D12_DESCRIPTOR_RANGE ranges[2] {};
    ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    ranges[0].NumDescriptors = 1;
    ranges[0].OffsetInDescriptorsFromTableStart = 0;
    ranges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    ranges[1].NumDescriptors = 1;
    ranges[1].OffsetInDescriptorsFromTableStart = 1;
    D3D12_ROOT_PARAMETER params[2] {};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[0].Constants.Num32BitValues = 8;
    params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[1].DescriptorTable.NumDescriptorRanges = 2;
    params[1].DescriptorTable.pDescriptorRanges = ranges;
    params[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    D3D12_STATIC_SAMPLER_DESC sampler {};
    sampler.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
    sampler.AddressU = sampler.AddressV = sampler.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    sampler.MaxLOD = D3D12_FLOAT32_MAX;
    sampler.ShaderRegister = 0;
    sampler.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    D3D12_ROOT_SIGNATURE_DESC rsDesc {};
    rsDesc.NumParameters = 2;
    rsDesc.pParameters = params;
    rsDesc.NumStaticSamplers = 1;
    rsDesc.pStaticSamplers = &sampler;
    ID3DBlob* rsBlob = nullptr;
    ID3DBlob* rsError = nullptr;
    if (FAILED(D3D12SerializeRootSignature(&rsDesc, D3D_ROOT_SIGNATURE_VERSION_1, &rsBlob, &rsError)) ||
        rsBlob == nullptr)
    {
        if (rsError)
            rsError->Release();
        LogOnce(g_loggedPeriphFail, "cheap periphery off: the enlarge root signature did not serialise");
        return false;
    }
    const HRESULT rsHr = device->CreateRootSignature(0, rsBlob->GetBufferPointer(), rsBlob->GetBufferSize(),
                                                     IID_PPV_ARGS(&g_enlargeRoot));
    rsBlob->Release();
    if (FAILED(rsHr))
    {
        LogOnce(g_loggedPeriphFail, "cheap periphery off: CreateRootSignature (enlarge) failed");
        return false;
    }
    ID3DBlob* cs = CompileShader(kEnlargeShader, "CSMain", "cs_5_0");
    if (cs == nullptr)
    {
        LogOnce(g_loggedPeriphFail, "cheap periphery off: the enlarge shader did not compile (see above)");
        return false;
    }
    D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc {};
    psoDesc.pRootSignature = g_enlargeRoot;
    psoDesc.CS = { cs->GetBufferPointer(), cs->GetBufferSize() };
    const HRESULT psoHr = device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&g_enlargePso));
    cs->Release();
    if (FAILED(psoHr))
    {
        LogOnce(g_loggedPeriphFail, "cheap periphery off: CreateComputePipelineState (enlarge) failed");
        return false;
    }
    D3D12_DESCRIPTOR_HEAP_DESC heapDesc {};
    heapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    heapDesc.NumDescriptors = kRing * 2;
    heapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    if (FAILED(device->CreateDescriptorHeap(&heapDesc, IID_PPV_ARGS(&g_enlargeHeap))))
    {
        LogOnce(g_loggedPeriphFail, "cheap periphery off: no enlarge descriptor heap");
        return false;
    }
    g_periphReady = true;
    return true;
}

bool EnsurePeriphTex(ID3D12Device* device, UINT64 w, UINT h, DXGI_FORMAT format, uint64_t present)
{
    if (g_periphTex != nullptr && g_periphTexW == w && g_periphTexH == h && g_periphTexFormat == format)
        return true;
    if (g_periphTex != nullptr)
    {
        g_retired.push_back({ g_periphTex, present }); // work recorded a moment ago can still reference it
        g_periphTex = nullptr;
    }
    D3D12_RESOURCE_DESC desc {};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = w;
    desc.Height = h;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = format;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    D3D12_HEAP_PROPERTIES heap {};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    if (FAILED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
                                               D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
                                               IID_PPV_ARGS(&g_periphTex))))
    {
        g_periphTex = nullptr;
        return false;
    }
    g_periphTex->SetName(L"PerEyeDlss_Dx12 periphery output");
    g_periphTexState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    g_periphTexW = w;
    g_periphTexH = h;
    g_periphTexFormat = format;
    LOG_INFO("Per-eye DLSS DX12: periphery output texture {}x{} fmt {} made", w, h, (int) format);
    return true;
}

// ---- Own TAA periphery (foveated DLSS, step 4) ----
//
// Per eye or shared (it follows F6; see TaaPeriphery's history rule), in place of the cheap periphery's DLSS call and
// ENLARGE: one pass at the output size over the whole eye (the
// crop is drawn over the centre after it). Per output pixel: the current frame reconstructed from the jittered render
// samples (4x4 Catmull-Rom; the sample of render pixel i was taken at i + 0.5 - jitter, the DLSS / FSR2 convention),
// this eye's history reprojected with the motion DLSS gets (per eye our same-eye motion), dilated to the nearest depth
// of the 3x3 (reversed Z: the largest), clamped to the 3x3's colour range (mean +- 1.25 sigma in YCoCg), and blended.
// All in a reversible tonemapped space (c / (1 + luma)), which is what the history keeps. Two history textures per eye,
// read one, write the other.
// kTaaShader: in FoveaShader_Dx12.h (one source for D3D12 and Vulkan).

struct TaaHistory
{
    ID3D12Resource* tex[2] = {};
    D3D12_RESOURCE_STATES state[2] = { D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS };
    uint32_t read = 0;
    UINT64 w = 0;
    UINT h = 0;
    uint64_t lastPresent = UINT64_MAX; // this eye's last TAA evaluate; history is used only from two Presents back
};
std::unordered_map<unsigned int, TaaHistory> g_taa; // owner feature (the game's or its twin = one eye) -> its history
bool g_taaTried = false, g_taaReady = false, g_taaFailed = false;
ID3D12RootSignature* g_taaRoot = nullptr;
ID3D12PipelineState* g_taaPso = nullptr;
ID3D12DescriptorHeap* g_taaHeap = nullptr; // kRing x (4 SRV, 2 UAV)
uint64_t g_taaSeq = 0;
uint64_t g_taaEvals = 0, g_taaResets = 0, g_taaSkips = 0;
bool g_loggedTaaFirst[2] = {}, g_loggedTaaFail = false, g_loggedTaaSkip = false; // first draw: [shared, per eye]

bool SetupTaa(ID3D12Device* device)
{
    g_taaTried = true;
    D3D12_DESCRIPTOR_RANGE ranges[2] {};
    ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    ranges[0].NumDescriptors = 4;
    ranges[0].OffsetInDescriptorsFromTableStart = 0;
    ranges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    ranges[1].NumDescriptors = 2;
    ranges[1].OffsetInDescriptorsFromTableStart = 4;
    D3D12_ROOT_PARAMETER params[2] {};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[0].Constants.Num32BitValues = 16;
    params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[1].DescriptorTable.NumDescriptorRanges = 2;
    params[1].DescriptorTable.pDescriptorRanges = ranges;
    params[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    D3D12_STATIC_SAMPLER_DESC sampler {};
    sampler.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
    sampler.AddressU = sampler.AddressV = sampler.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    sampler.MaxLOD = D3D12_FLOAT32_MAX;
    sampler.ShaderRegister = 0;
    sampler.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    D3D12_ROOT_SIGNATURE_DESC rsDesc {};
    rsDesc.NumParameters = 2;
    rsDesc.pParameters = params;
    rsDesc.NumStaticSamplers = 1;
    rsDesc.pStaticSamplers = &sampler;
    ID3DBlob* rsBlob = nullptr;
    ID3DBlob* rsError = nullptr;
    if (FAILED(D3D12SerializeRootSignature(&rsDesc, D3D_ROOT_SIGNATURE_VERSION_1, &rsBlob, &rsError)) ||
        rsBlob == nullptr)
    {
        if (rsError)
            rsError->Release();
        LogOnce(g_loggedTaaFail, "own TAA periphery off: the root signature did not serialise");
        return false;
    }
    const HRESULT rsHr = device->CreateRootSignature(0, rsBlob->GetBufferPointer(), rsBlob->GetBufferSize(),
                                                     IID_PPV_ARGS(&g_taaRoot));
    rsBlob->Release();
    if (FAILED(rsHr))
    {
        LogOnce(g_loggedTaaFail, "own TAA periphery off: CreateRootSignature failed");
        return false;
    }
    ID3DBlob* cs = CompileShader(kTaaShader, "CSMain", "cs_5_0");
    if (cs == nullptr)
    {
        LogOnce(g_loggedTaaFail, "own TAA periphery off: the TAA shader did not compile (see above)");
        return false;
    }
    D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc {};
    psoDesc.pRootSignature = g_taaRoot;
    psoDesc.CS = { cs->GetBufferPointer(), cs->GetBufferSize() };
    const HRESULT psoHr = device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&g_taaPso));
    cs->Release();
    if (FAILED(psoHr))
    {
        LogOnce(g_loggedTaaFail, "own TAA periphery off: CreateComputePipelineState failed");
        return false;
    }
    D3D12_DESCRIPTOR_HEAP_DESC heapDesc {};
    heapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    heapDesc.NumDescriptors = kRing * 6;
    heapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    if (FAILED(device->CreateDescriptorHeap(&heapDesc, IID_PPV_ARGS(&g_taaHeap))))
    {
        LogOnce(g_loggedTaaFail, "own TAA periphery off: no descriptor heap");
        return false;
    }
    g_taaReady = true;
    LOG_INFO("Per-eye DLSS DX12: own TAA periphery ready");
    return true;
}

// This eye's two history textures at the output size; a new size retires the old ones (never released under work that
// may still reference them) and starts the history over.
bool EnsureTaaHistory(ID3D12Device* device, TaaHistory& hist, UINT64 w, UINT h, uint64_t present)
{
    if (hist.tex[0] != nullptr && hist.w == w && hist.h == h)
        return true;
    for (auto*& tex : hist.tex)
        if (tex != nullptr)
        {
            g_retired.push_back({ tex, present });
            tex = nullptr;
        }
    hist.lastPresent = UINT64_MAX;
    D3D12_RESOURCE_DESC desc {};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = w;
    desc.Height = h;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    D3D12_HEAP_PROPERTIES heap {};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    for (int i = 0; i < 2; ++i)
    {
        if (FAILED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
                                                   D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
                                                   IID_PPV_ARGS(&hist.tex[i]))))
        {
            hist.tex[i] = nullptr;
            if (i == 1 && hist.tex[0] != nullptr)
            {
                g_retired.push_back({ hist.tex[0], present });
                hist.tex[0] = nullptr;
            }
            return false;
        }
        hist.tex[i]->SetName(L"PerEyeDlss_Dx12 TAA periphery history");
        hist.state[i] = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    }
    hist.w = w;
    hist.h = h;
    LOG_INFO("Per-eye DLSS DX12: TAA periphery history textures {}x{} made (x2)", w, h);
    return true;
}

// The TAA's own output below TAA scale 1 (RGBA16F, ours), ENLARGEd over the eye right after. One for every eye: each
// evaluate writes and reads it on its own list, in queue order. A new size retires the old one, never releases it.
ID3D12Resource* g_taaMid = nullptr;
D3D12_RESOURCE_STATES g_taaMidState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
UINT g_taaMidW = 0, g_taaMidH = 0;
uint64_t g_taaEnlarged = 0;
bool g_loggedTaaScaleFail = false;

bool EnsureTaaMid(ID3D12Device* device, UINT w, UINT h, uint64_t present)
{
    if (g_taaMid != nullptr && g_taaMidW == w && g_taaMidH == h)
        return true;
    if (g_taaMid != nullptr)
    {
        g_retired.push_back({ g_taaMid, present });
        g_taaMid = nullptr;
    }
    D3D12_RESOURCE_DESC desc {};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = w;
    desc.Height = h;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    D3D12_HEAP_PROPERTIES heap {};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    if (FAILED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                               nullptr, IID_PPV_ARGS(&g_taaMid))))
    {
        g_taaMid = nullptr;
        return false;
    }
    g_taaMid->SetName(L"PerEyeDlss_Dx12 TAA periphery (scaled)");
    g_taaMidState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    g_taaMidW = w;
    g_taaMidH = h;
    LOG_INFO("Per-eye DLSS DX12: TAA periphery runs at {}x{} and is enlarged to the eye", w, h);
    return true;
}

void TaaState(ID3D12GraphicsCommandList* cmdList, TaaHistory& hist, int i, D3D12_RESOURCE_STATES state)
{
    if (hist.state[i] == state)
        return;
    Transition(cmdList, hist.tex[i], hist.state[i], state);
    hist.state[i] = state;
}

void SetupTiming(ID3D12Device* device)
{
    g_timeTried = true;
    D3D12_QUERY_HEAP_DESC qd {};
    qd.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
    qd.Count = kRing * 6;
    if (FAILED(device->CreateQueryHeap(&qd, IID_PPV_ARGS(&g_queryHeap))))
        return LogOnce(g_loggedTimeFail, "GPU timing off: no timestamp query heap");
    D3D12_RESOURCE_DESC buf {};
    buf.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    buf.Width = kRing * 48;
    buf.Height = 1;
    buf.DepthOrArraySize = 1;
    buf.MipLevels = 1;
    buf.SampleDesc.Count = 1;
    buf.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    D3D12_HEAP_PROPERTIES heap {};
    heap.Type = D3D12_HEAP_TYPE_READBACK;
    if (FAILED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &buf, D3D12_RESOURCE_STATE_COPY_DEST,
                                               nullptr, IID_PPV_ARGS(&g_timeReadback))))
        return LogOnce(g_loggedTimeFail, "GPU timing off: no timestamp readback buffer");
    void* mapped = nullptr;
    if (FAILED(g_timeReadback->Map(0, nullptr, &mapped)) || mapped == nullptr)
        return LogOnce(g_loggedTimeFail, "GPU timing off: the timestamp readback buffer did not map");
    g_timeData = (uint64_t*) mapped;
    std::memset(mapped, 0, kRing * 48);
    g_timeReady = true;
    LOG_INFO("Per-eye DLSS DX12: GPU timing ready (same-eye passes and the upscaler's evaluate, per eye)");
}

// The timestamps of the evaluate kRing back, if they are complete and ordered; the slot is cleared for its next use.
void ReadTimeSlot(uint32_t slot)
{
    if (!g_timeFilled[slot])
        return;
    g_timeFilled[slot] = false;
    uint64_t t[6];
    std::memcpy(t, g_timeData + slot * 6, sizeof(t));
    std::memset(g_timeData + slot * 6, 0, sizeof(t));
    if (g_tsFrequency == 0 || t[0] == 0 ||
        !(t[0] <= t[1] && t[1] <= t[2] && t[2] <= t[3] && t[3] <= t[4] && t[4] <= t[5]))
    {
        ++g_timesInvalid;
        return;
    }
    const double toMs = 1000.0 / (double) g_tsFrequency;
    auto& g = g_times[g_timeMode[slot]][g_timeParity[slot]];
    g.fit += (double) (t[1] - t[0]) * toMs;
    g.compose += (double) (t[2] - t[1]) * toMs;
    g.dlss += (double) (t[4] - t[3]) * toMs;
    g.fovea += (double) (t[5] - t[4]) * toMs;
    ++g.n;
    if (!g_loggedFirstTime)
    {
        g_loggedFirstTime = true;
        LOG_INFO("Per-eye DLSS DX12: FIRST GPU time read back: upscaler {:.3f} ms, fit {:.3f} ms, compose {:.3f} ms ({}, "
                 "{} Presents)",
                 (double) (t[4] - t[3]) * toMs, (double) (t[1] - t[0]) * toMs, (double) (t[2] - t[1]) * toMs,
                 g_timeMode[slot] ? "per eye" : "shared", g_timeParity[slot] ? "odd" : "even");
    }
}

// Per eye: fit this Present's map, chain it with the previous Present's, and hand DLSS the result in place of the
// game's vectors -- or reset this eye's history when there is nothing to chain. Under g_mutex.
// forNr (NR-only mode): DLSS stays the game's shared feature on the game's vectors and is never reset by us; the
// same-eye motion goes to NR's finished-picture copy only (kNrMotionParam), and a break resets NR's eye history.
void SameEye(ID3D12GraphicsCommandList* cmdList, ID3D12Device* device, NVSDK_NGX_Parameter* params, bool forNr = false)
{
    auto resetHistory = [&]
    {
        if (forNr)
            params->Set(DlssNr::kNrResetParam, 1u);
        else
            params->Set(NVSDK_NGX_Parameter_Reset, 1); // never the other eye's vectors into a per-eye history
    };
    if (!g_motionTried)
        SetupMotion(device);
    if (!g_motionReady)
    {
        resetHistory();
        return;
    }
    const uint64_t present = g_present.load();
    ID3D12Resource* motion = GetResource(params, NVSDK_NGX_Parameter_MotionVectors);
    ID3D12Resource* depth = GetResource(params, NVSDK_NGX_Parameter_Depth);
    float scale[2] = { 0, 0 };
    params->Get(NVSDK_NGX_Parameter_MV_Scale_X, &scale[0]);
    params->Get(NVSDK_NGX_Parameter_MV_Scale_Y, &scale[1]);
    auto skip = [&](const char* why)
    {
        LogOnce(g_loggedMotionFail, std::format("same-eye motion skipped ({}); this eye's history is reset instead", why));
        resetHistory();
        ++g_chainBreaks;
    };
    if (motion == nullptr || depth == nullptr)
        return skip("no motion or no depth in the DLSS parameters");
    if (scale[0] == 0 || scale[1] == 0)
        return skip("MV scale is zero");
    const D3D12_RESOURCE_DESC md = motion->GetDesc();
    const D3D12_RESOURCE_DESC dd = depth->GetDesc();
    if (md.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || dd.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D)
        return skip("motion or depth is not a 2D texture");
    if (!EnsureComposed(device, md, present))
    {
        ++g_sizeSkips;
        LogOnce(g_loggedSizeSkip, std::format("same-eye motion skipped at motion size {}x{} (texture {}x{}; a new size "
                                              "is adopted after 90 evaluates); counted per window",
                                              md.Width, md.Height, g_composedW, g_composedH));
        resetHistory();
        return;
    }

    const uint32_t cur = (uint32_t) (present & 1), prev = cur ^ 1;
    const bool chain = present > 0 && g_fitPresent[prev] == present - 1;
    const uint32_t slot = (uint32_t) (g_fitSeq % kRing);
    ReadFitSlot(slot);
    g_fitSlotParity[slot] = cur;

    // Views: depth, the game's vectors, our texture. The game's two are read as DLSS reads them -- no barrier.
    D3D12_CPU_DESCRIPTOR_HANDLE cpu = g_motionHeap->GetCPUDescriptorHandleForHeapStart();
    cpu.ptr += (SIZE_T) slot * 4 * g_descriptorSize;
    D3D12_GPU_DESCRIPTOR_HANDLE gpu = g_motionHeap->GetGPUDescriptorHandleForHeapStart();
    gpu.ptr += (UINT64) slot * 4 * g_descriptorSize;
    D3D12_SHADER_RESOURCE_VIEW_DESC srv {};
    srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srv.Texture2D.MipLevels = 1;
    srv.Format = DepthSrvFormat(dd.Format);
    device->CreateShaderResourceView(depth, &srv, cpu);
    cpu.ptr += g_descriptorSize;
    srv.Format = MotionSrvFormat(md.Format);
    device->CreateShaderResourceView(motion, &srv, cpu);
    cpu.ptr += g_descriptorSize;
    D3D12_UNORDERED_ACCESS_VIEW_DESC uav {};
    uav.Format = DXGI_FORMAT_R16G16_FLOAT;
    uav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    device->CreateUnorderedAccessView(g_composed, nullptr, &uav, cpu);
    cpu.ptr += g_descriptorSize;
    uav.Format = DXGI_FORMAT_R8_UNORM;
    device->CreateUnorderedAccessView(g_mask, nullptr, &uav, cpu);

    const bool dump = g_dumpLeft > 0 && g_dumpReadbackData != nullptr;
    struct
    {
        float mvScale[2];
        uint32_t cur, prev;
        float knee0, knee1, gain;
        uint32_t radius, stamp, round, keepStale, linearDepth;
    } constants { { scale[0], scale[1] }, cur, prev, ObjectKnee0(), ObjectKnee1(), ObjectGain(), ObjectRadiusWord(),
                  (uint32_t) g_fitSeq, 100,
                  (Config::Instance()->UpscalerFitGiveUpStale.value_or_default() ? 0u : 1u) |
                      (Config::Instance()->UpscalerFitRestart.value_or_default() ? 0u : 2u) |
                      (Config::Instance()->UpscalerFitLeaveOutNearest.value_or_default() ? 0u : 4u) |
                      (dump ? 8u : 0u),
                  LinearDepthFor(params) ? 1u : 0u };
    static_assert(sizeof(constants) == 48);

    ID3D12DescriptorHeap* heaps[] = { g_motionHeap };
    cmdList->SetDescriptorHeaps(1, heaps);
    cmdList->SetComputeRootSignature(g_motionRoot);
    cmdList->SetComputeRoot32BitConstants(0, 12, &constants, 0);
    cmdList->SetComputeRootDescriptorTable(1, gpu);
    cmdList->SetComputeRootUnorderedAccessView(2, g_fits->GetGPUVirtualAddress());
    cmdList->SetComputeRootShaderResourceView(3, g_fits->GetGPUVirtualAddress());
    cmdList->SetComputeRootUnorderedAccessView(4, g_partials->GetGPUVirtualAddress());
    // INIT (1 group), then 4 weighted rounds and the measuring round (64 groups, the last one solves). Both buffers
    // are written by every dispatch (the solution; the partials and their count), so both are barriered after each.
    // The UAV barriers are on our own buffers.
    D3D12_RESOURCE_BARRIER uavs[2] {};
    uavs[0].Type = uavs[1].Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    uavs[0].UAV.pResource = g_fits;
    uavs[1].UAV.pResource = g_partials;
    cmdList->SetPipelineState(g_fitPso);
    cmdList->Dispatch(1, 1, 1);
    cmdList->ResourceBarrier(2, uavs);
    for (uint32_t round = 0; round <= 4; ++round)
    {
        cmdList->SetComputeRoot32BitConstant(0, round, 9);
        cmdList->Dispatch(64, 1, 1);
        cmdList->ResourceBarrier(2, uavs);
    }
    if (g_timingThis)
    {
        cmdList->EndQuery(g_queryHeap, D3D12_QUERY_TYPE_TIMESTAMP, g_timeSlotCur * 6 + 1);
        g_wroteFitTime = true;
    }

    // Our fits buffer only: UAV (promoted from COMMON on a fresh list, or left so by us) -> read + copy -> UAV.
    const auto readStates = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_COPY_SOURCE;
    Transition(cmdList, g_fits, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, readStates);
    if (chain)
    {
        if (g_composedState != D3D12_RESOURCE_STATE_UNORDERED_ACCESS)
        {
            Transition(cmdList, g_composed, g_composedState, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            g_composedState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        }
        if (g_maskState != D3D12_RESOURCE_STATE_UNORDERED_ACCESS)
        {
            Transition(cmdList, g_mask, g_maskState, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            g_maskState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        }
        cmdList->SetPipelineState(g_composePso);
        cmdList->Dispatch((UINT) ((md.Width + 7) / 8), (md.Height + 7) / 8, 1);
        // DLSS reads both as shader resources in a compute pass.
        Transition(cmdList, g_composed, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                   D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        g_composedState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        Transition(cmdList, g_mask, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                   D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        g_maskState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    }
    cmdList->CopyBufferRegion(g_fitReadback, slot * 256, g_fits, 0, 15 * 16);
    Transition(cmdList, g_fits, readStates, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    if (dump)
    {
        // Our partials buffer only: UAV (its dispatches above) -> copy source -> UAV.
        Transition(cmdList, g_partials, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
        cmdList->CopyBufferRegion(g_dumpReadback, (UINT64) slot * kDumpStride, g_partials, kPartialWords * 4, kDumpBytes);
        Transition(cmdList, g_partials, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        DumpMeta& meta = g_dumpMeta[slot];
        meta.armed = true;
        meta.present = present;
        meta.seq = g_fitSeq;
        meta.parity = cur;
        meta.keepStale = constants.keepStale;
        meta.linear = constants.linearDepth;
        meta.mw = (uint32_t) md.Width;
        meta.mh = md.Height;
        meta.mf = (uint32_t) md.Format;
        meta.dw = (uint32_t) dd.Width;
        meta.dh = dd.Height;
        meta.df = (uint32_t) dd.Format;
        meta.scale[0] = scale[0];
        meta.scale[1] = scale[1];
        --g_dumpLeft;
    }
    ++g_fitSeq;
    g_fitPresent[cur] = present;

    if (chain && forNr)
    {
        // DLSS keeps the game's vectors; NR's copy takes ours, and so do the crop and the periphery (BeginFovea /
        // BeginPeriphery, through g_composedForShared).
        params->Set(DlssNr::kNrMotionParam, (void*) g_composed);
        g_composedForShared = true;
        ++g_composedEvals;
        if (!g_loggedFirstNrOnly)
        {
            g_loggedFirstNrOnly = true;
            LOG_INFO("Per-eye DLSS DX12: FIRST same-eye motion handed to NR only ({}x{}, Present {}); DLSS stays the "
                     "game's shared feature",
                     md.Width, md.Height, present);
        }
    }
    else if (chain)
    {
        g_swappedFrom = motion;
        params->Set(NVSDK_NGX_Parameter_MotionVectors, g_composed);
        g_swapped = true;
        ++g_composedEvals;
        if (BiasMaskMode() != 0)
        {
            g_maskWas = nullptr;
            params->Get(NVSDK_NGX_Parameter_DLSS_Input_Bias_Current_Color_Mask, &g_maskWas);
            params->Set(NVSDK_NGX_Parameter_DLSS_Input_Bias_Current_Color_Mask, (void*) g_mask);
            params->Set(NVSDK_NGX_Parameter_DLSS_Input_Bias_Current_Color_SubrectBase_X, 0u);
            params->Set(NVSDK_NGX_Parameter_DLSS_Input_Bias_Current_Color_SubrectBase_Y, 0u);
            g_maskHanded = true;
            ++g_maskEvals;
            if (!g_loggedFirstMask)
            {
                g_loggedFirstMask = true;
                LOG_INFO("Per-eye DLSS DX12: FIRST current-colour bias mask handed to DLSS ({}x{}, Present {}, {}; the "
                         "game had {})",
                         md.Width, md.Height, present, BiasMaskName(), g_maskWas ? "a mask of its own" : "none");
            }
        }
        if (!g_loggedFirstComposed)
        {
            g_loggedFirstComposed = true;
            LOG_INFO("Per-eye DLSS DX12: FIRST same-eye motion handed to DLSS ({}x{}, Present {}; depth {}x{} fmt {})",
                     md.Width, md.Height, present, dd.Width, dd.Height, (int) dd.Format);
        }
    }
    else
    {
        // No fit of the previous Present: nothing to chain, so this eye's history starts over rather than being fed
        // the other eye's vectors.
        resetHistory();
        ++g_chainBreaks;
    }
}

// NR-only mode (Hogwarts Legacy): per-eye NR on the finished picture while DLSS stays the game's own shared
// feature -- routing shared, finished-picture NR on, NR history per eye on. Per-eye DLSS looked worse there; NR on the
// finished image is what is wanted, with one history per eye and same-eye motion.
bool NrOnlyMode()
{
    const auto* config = Config::Instance();
    return !g_routing && config->DlssNrFinishedPicture.value_or_default() && config->DlssNrEnabled.value_or_default() &&
           DlssNr::PerEyeHistory();
}

// Shared routing with NR on and its history per eye: the same-eye fit and compose run for NR alone (kNrMotionParam);
// DLSS keeps the game's vectors and its one feature. Only the finished-picture mode used to do this, and
// NR before / after SR read the game's other-eye vectors against a per-eye history (flicker in shared routing in
// one D3D12 game).
bool NrWantsSameEye()
{
    const auto* config = Config::Instance();
    return !g_routing && config->DlssNrEnabled.value_or_default() && DlssNr::PerEyeHistory();
}
} // namespace

namespace PerEyeDlss_Dx12
{
bool Enabled()
{
    static const bool enabled = Config::Instance()->UpscalerPerEye.value_or_default();
    return enabled;
}

void OnPresent(ID3D12CommandQueue* queue)
{
    if (!Enabled())
        return;
    std::lock_guard lock(g_mutex);
    if (g_tsFrequency == 0 && queue != nullptr)
    {
        UINT64 frequency = 0;
        if (SUCCEEDED(queue->GetTimestampFrequency(&frequency)))
            g_tsFrequency = frequency;
    }
    const uint64_t present = ++g_present;
    if (g_seenEvaluate)
    {
        if (g_evalsThisPresent == 0)
            ++g_presentsWithout;
        else if (g_evalsThisPresent > 1)
            ++g_presentsWithSeveral;
    }
    g_evalsThisPresent = 0;
    // Our retired textures, once no recorded work can still reference them.
    for (auto it = g_retired.begin(); it != g_retired.end();)
    {
        if (present >= it->second + 16)
        {
            it->first->Release();
            it = g_retired.erase(it);
        }
        else
            ++it;
    }
    if (!g_seenEvaluate || present - g_windowStart < kWindow)
        return;

    std::string motion;
    // The window's counters print whether or not the probe runs (they had all sat behind ProbeOn(), so
    // the probe off would have silenced the crop / DL(S)SS lines); only the motion statistics are the probe's.
    {
        if (ProbeOn())
        {
            for (int i = 0; i < 2; ++i)
            {
                const auto& p = g_parity[i];
                motion += p.frames == 0 ? std::format("; {} Presents: no motion statistics", i ? "odd" : "even")
                                        : std::format("; {} Presents: mean motion x {:+.2f} px, |x| {:.2f}, |y| "
                                                      "{:.2f} ({} frames)",
                                                      i ? "odd" : "even", p.sumX / p.frames, p.sumAbsX / p.frames,
                                                      p.sumAbsY / p.frames, p.frames);
            }
            motion += std::format("; statistics not ready {}", g_statsNotReady);
        }
        std::string times;
        for (int mode = 0; mode < 2; ++mode)
            for (int parity = 0; parity < 2; ++parity)
            {
                const auto& g = g_times[mode][parity];
                if (g.n == 0)
                    continue;
                times += std::format("{} {}: upscaler {:.3f} ms, foveal crop {:.3f} ms, fit {:.3f} ms, compose "
                                     "{:.3f} ms ({}); ",
                                     mode ? "per eye" : "shared", parity ? "odd" : "even", g.dlss / g.n,
                                     g.fovea / g.n, g.fit / g.n, g.compose / g.n, g.n);
            }
        {
            double up = 0, crop = 0;
            uint64_t n = 0;
            for (auto& row : g_times)
                for (auto& g : row)
                {
                    up += g.dlss;
                    crop += g.fovea;
                    n += g.n;
                }
            g_lastUpscalerMs = n ? up / n : 0;
            g_lastCropMs = n ? crop / n : 0;
        }
        motion += TimingOn() ? std::format("; GPU per evaluate: {}{} invalid", times, g_timesInvalid)
                             : std::string("; GPU times off (the menu's GPU timing / [Inputs] UpscalerGpuTiming)");
        motion += std::format("; foveal crop {} (preset {}; as ray reconstruction {}, its RR feature made with RR "
                              "preset {} [-1 none made, 0 the DLL's default, 4 = D, 5 = E]): {} drawn, {} skipped, {} "
                              "failed, {} frames rebuilding",
                              g_foveaOn ? "ON" : "off", Config::Instance()->UpscalerFoveaPreset.value_or_default(),
                              g_cropIsRr.load() ? "YES" : "no",
                              g_rrCropPreset.load(), g_foveaEvals, g_foveaSkips, g_foveaFails, g_foveaRebuildFrames);
        g_foveaEvals = g_foveaSkips = g_foveaFails = g_foveaRebuildFrames = 0;
        motion += std::format("; the full-size pass of a foveated frame: {} checked, its vectors replaced below us on {}; "
                              "the crop's vectors (RR choice {}, SR choice {}): {} from below (DL(S)SS's), {} ours, {} "
                              "wanted from below but none there",
                              g_mainChecked, g_mainSwapped,
                              Config::Instance()->UpscalerFoveaRrVectors.value_or_default(),
                              Config::Instance()->UpscalerFoveaSrVectors.value_or_default(), g_cropVecBelow,
                              g_cropVecOurs, g_cropVecBelowMissing);
        g_mainChecked = g_mainSwapped = g_cropVecBelow = g_cropVecOurs = g_cropVecBelowMissing = 0;
        motion += std::format("; crop evaluates DIRECT on nvngx_dlssd.dll (past the NGX entry): {} ok, "
                              "{} failed; through the NGX entry {}; direct features made since start {}, creates failed {} "
                              "(last code {:#x})",
                              g_cropDirectEvals.load(), g_cropDirectEvalFails.load(), g_cropCoreEvals.load(),
                              g_cropDirectMade.load(), g_cropDirectFailed.load(),
                              (unsigned int) g_cropDirectLastFail.load());
        g_cropDirectEvals = g_cropDirectEvalFails = g_cropCoreEvals = 0;
        motion += std::format("; SR crop evaluates DIRECT on nvngx_dlss.dll (past the NGX entry): "
                              "{} ok, {} failed; SR crop evaluates through the NGX entry {}; direct SR features made since "
                              "start {}, creates failed {} (last code {:#x}); switch {} (a change rebuilds the crop)",
                              g_srDirectEvals.load(), g_srDirectEvalFails.load(), g_srCoreEvals.load(),
                              g_srDirectMade.load(), g_srDirectFailed.load(), (unsigned int) g_srDirectLastFail.load(),
                              Config::Instance()->UpscalerFoveaSrDirect.value_or_default() ? "ON" : "off");
        g_srDirectEvals = g_srDirectEvalFails = g_srCoreEvals = 0;
        {
            const int oneMode = Config::Instance()->UpscalerPeripheryOneFeature.value_or_default();
            const int srVec = Config::Instance()->UpscalerFoveaSrVectors.value_or_default();
            motion += std::format("; vectors seen replaced below us on the full-size pass: {} (changes this window {}); cheap periphery "
                                  "evaluates on ONE feature for both eyes (the game's vectors) {}, on a feature per eye {} "
                                  "(periphery features {}); SR crop's vectors {} (now {}); same-eye fit skipped (nothing "
                                  "takes it) on {} evaluates; NR after SR on DL(S)SS's vectors (none of ours) on {} evaluates; DL(S)SS's "
                                  "vectors handed down from the periphery's own block to the game's on {} evaluates",
                                  g_belowSeen.load() ? "YES" : "no", g_periphAutoFlips, g_periphOneEvals.load(),
                                  g_periphPerEyeEvals.load(), oneMode == 1 ? "one" : oneMode == 2 ? "per eye" : "auto",
                                  srVec < 0 ? "auto" : srVec == 1 ? "DL(S)SS's" : "ours",
                                  SrCropFromBelow() ? "from below" : "ours", g_fitSkippedBelow, g_nrOnBelow,
                                  g_belowHandedDown);
            g_periphOneEvals = g_periphPerEyeEvals = 0;
            g_periphAutoFlips = 0;
            g_fitSkippedBelow = 0;
            g_nrOnBelow = 0;
            g_belowHandedDown = 0;
            const uint64_t vramNow = VramUsedMB(g_device);
            motion += std::format("; VRAM now {} MB (window start {} MB); made since start (count, MB): the game's "
                                  "{} {:+}, twin {} {:+}, crop {} {:+}, periphery {} {:+}, NR {} {:+}",
                                  vramNow, g_vramWindowStartMB.load(), g_vramCreates[0].load(), g_vramMade[0].load(),
                                  g_vramCreates[4].load(), g_vramMade[4].load(), g_vramCreates[1].load(),
                                  g_vramMade[1].load(), g_vramCreates[2].load(), g_vramMade[2].load(),
                                  g_vramCreates[3].load(), g_vramMade[3].load());
            g_vramWindowStartMB = vramNow;
        }
        motion += std::format("; crop moves since start: {} begun, {} carried over, {} history restarts ({} with neither "
                              "copies nor vectors of ours to shift, {} because the copy or the shift failed; move mode {}); "
                              "crop origin off a whole output pixel by up to {:.3f} px this window (whole-pixel placing {})",
                              g_followMoves, g_followShifts, g_followResets, g_followResetsNoCopy,
                              g_followResetsCopyFailed,
                              Config::Instance()->UpscalerFoveaMoveMode.value_or_default(), g_placeErrMax,
                              Config::Instance()->UpscalerFoveaPlaceExact.value_or_default() ? "ON" : "off");
        g_placeErrMax = 0.0;
        motion += std::format("; crop age (fade-in {} frames): {} passes, {} resets, {} failed",
                              Config::Instance()->UpscalerFoveaAgeFrames.value_or_default(), g_ageRuns, g_ageResets,
                              g_ageFails);
        g_ageRuns = g_ageResets = g_ageFails = 0;
        motion += std::format("; the game's bias mask at the crop since start: {} copied at the box, {} dropped, {} as handed "
                              "(mask mode {})",
                              g_maskCopied, g_maskDropped, g_maskAsIs,
                              Config::Instance()->UpscalerFoveaCropMask.value_or_default());
        motion += std::format("; crop inputs read back AFTER the evaluate: {} checked, vectors swapped {}, colour swapped {}, "
                              "depth swapped {}, MV base changed {} (a swap = something below us, DL(S)SS?, replaced them)",
                              g_cropAfterChecked, g_cropAfterMvSwapped, g_cropAfterColourSwapped, g_cropAfterDepthSwapped,
                              g_cropAfterBaseChanged);
        g_cropAfterChecked = g_cropAfterMvSwapped = g_cropAfterColourSwapped = g_cropAfterDepthSwapped =
            g_cropAfterBaseChanged = 0;
        motion += std::format("; the crop reads {} ({}; copied vectors replaced below us this launch: {})",
                              g_cropCopiesLast < 0 ? "nothing yet" : g_cropCopiesLast != 0 ? "COPIES" : "SUBRECTS",
                              Config::Instance()->UpscalerFoveaCopyInputs.has_value() ? "set by hand" : "automatic",
                              g_copiesSwappedBelow ? "YES" : "no");
        motion += std::format("; crop shift through MV offset: {} evaluates this window, MV offset changed below us {} (switch {})",
                              g_offsetShifts, g_cropAfterOffsetChanged,
                              Config::Instance()->UpscalerFoveaShiftByOffset.value_or_default() ? "ON" : "off");
        g_offsetShifts = g_cropAfterOffsetChanged = 0;
        motion += std::format("; cheap periphery {}{} (preset {}): {} drawn, {} fell back to the full pass; remade after "
                              "a failed first evaluate {} since start",
                              Config::Instance()->UpscalerFoveaCheapPeriphery.value_or_default() ? "ON" : "off",
                              Config::Instance()->UpscalerFoveaPeripherySr.value_or_default() ? " as SR in RR mode" : "",
                              Config::Instance()->UpscalerFoveaPeripheryPreset.value_or_default(), g_periphEvals,
                              g_periphFallbacks, g_periphRemadeTotal);
        motion += std::format("; own TAA periphery {} (blend {:.2f}{}, resolution {:.2f}): {} drawn ({} with no history, {} "
                              "at {}x{} enlarged), {} skipped",
                              Config::Instance()->UpscalerFoveaPeripheryTaa.value_or_default() ? "ON" : "off",
                              Config::Instance()->UpscalerFoveaTaaBlend.value_or_default(),
                              Config::Instance()->UpscalerFoveaTaaFlipJitter.value_or_default() ? ", jitter flipped"
                                                                                                 : "",
                              Config::Instance()->UpscalerFoveaTaaScale.value_or_default(), g_taaEvals, g_taaResets,
                              g_taaEnlarged, g_taaMidW, g_taaMidH, g_taaSkips);
        g_periphEvals = g_periphFallbacks = 0;
        g_taaEvals = g_taaResets = g_taaSkips = g_taaEnlarged = 0;
        for (auto& row : g_times)
            for (auto& g : row)
                g = GpuTimes {};
        g_timesInvalid = 0;
    }
    const std::string sameEye =
        g_composedEvals + g_chainBreaks + g_sizeSkips == 0
            ? std::string()
            : std::format("; same-eye motion on {} evaluates, {} chain breaks (history reset), {} size skips; fits read "
                          "{} ({} not ready), valid {}, mean inlier residual {:.4f} px, mean inliers {:.1f}%; rejected: {} "
                          "solve failed, {} too few depth samples, {} too few within 1 px -> {} bridged by the eye's "
                          "last fit, {} past 2 in a row (still the eye's last fit); stale start given up {} ({} of "
                          "those accepted; switch {}); restarted within the fit {} ({} of those accepted; switch "
                          "{}); a depth gap found (the nearer side left out: a car) on {} fits, {:.1f}% of the "
                          "samples left out on average (switch {}); object "
                          "motion {}; bias mask {} on {} evaluates; vectors same-eye already {}; crop per eye in shared routing {}; depth linear {} / hardware {}; parallax sign of the fits: even "
                          "+{} -{}, odd +{} -{} (one sign per parity = parity is the eye)",
                          g_composedEvals, g_chainBreaks, g_sizeSkips, g_fitsRead, g_fitsNotReady, g_fitsValid,
                          g_fitsValid ? g_fitResidualSum / g_fitsValid : 0.0,
                          g_fitsValid ? 100.0 * g_fitInlierSum / g_fitsValid : 0.0, g_fitReasons[1], g_fitReasons[2],
                          g_fitReasons[3], g_fitKeptUsed, g_fitStale, g_fitGaveUp, g_fitGaveUpAccepted,
                          Config::Instance()->UpscalerFitGiveUpStale.value_or_default() ? "ON" : "OFF",
                          g_fitRestarted, g_fitRestartedAccepted,
                          Config::Instance()->UpscalerFitRestart.value_or_default() ? "ON" : "OFF",
                          g_fitLeftOutFits, g_fitLeftOutFits ? 100.0 * g_fitLeftOutSum / g_fitLeftOutFits : 0.0,
                          Config::Instance()->UpscalerFitLeaveOutNearest.value_or_default() ? "ON" : "OFF",
                          ObjectMotionName(), BiasMaskName(), g_maskEvals,
                          Config::Instance()->UpscalerVectorsSameEye.value_or_default() ? "ON (no fit of ours)" : "off",
                          Config::Instance()->UpscalerCropEyeByParity.value_or_default() ? "ON" : "off",
                          g_fitLinear, g_fitHardware,
                          g_parallaxSign[0][1], g_parallaxSign[0][0],
                          g_parallaxSign[1][1], g_parallaxSign[1][0]);
    LOG_INFO("Per-eye DLSS DX12: {} Presents, routing {}: {} evaluates to the game's feature, {} to the twin, {} with "
             "no twin; {} Presents without an evaluate, {} with several{}{} (statistics = what DLSS gets)",
             present - g_windowStart, g_routing ? "PER EYE" : "shared (game's feature only)", g_toPrimary, g_toTwin,
             g_noTwin, g_presentsWithout, g_presentsWithSeveral, sameEye, motion);
    if (g_rejCount > 0)
        LOG_INFO("Per-eye DLSS DX12:   the fits rejected for too few within 1 px this window: {} -- mean inliers {:.1f}%, "
                 "mean inlier residual {:.3f} px, mean samples {:.0f}; inliers under 10%: {}, 10-25%: {}, 25-40%: {}, "
                 "40-50%: {}",
                 g_rejCount, 100.0 * g_rejInlierSum / g_rejCount, g_rejResidualSum / g_rejCount,
                 g_rejSampleSum / g_rejCount, g_rejHist[0], g_rejHist[1], g_rejHist[2], g_rejHist[3]);
    g_rejCount = g_rejHist[0] = g_rejHist[1] = g_rejHist[2] = g_rejHist[3] = 0;
    g_rejInlierSum = g_rejResidualSum = g_rejSampleSum = 0;
    g_composedEvals = g_chainBreaks = g_sizeSkips = g_fitsRead = g_fitsValid = g_fitsNotReady = 0;
    g_maskEvals = 0;
    g_fitReasons[0] = g_fitReasons[1] = g_fitReasons[2] = g_fitReasons[3] = g_fitKeptUsed = g_fitStale = 0;
    g_fitGaveUp = g_fitGaveUpAccepted = 0;
    g_fitRestarted = g_fitRestartedAccepted = 0;
    g_fitLeftOutSum = 0;
    g_fitLeftOutFits = 0;
    g_fitLinear = g_fitHardware = 0;
    g_fitResidualSum = g_fitInlierSum = 0;
    ReviewEyeDecision();
    std::memset(g_parallaxSign, 0, sizeof(g_parallaxSign));
    g_windowStart = present;
    g_toPrimary = g_toTwin = g_noTwin = g_presentsWithout = g_presentsWithSeveral = 0;
    g_parity[0] = g_parity[1] = Parity {};
    g_statsNotReady = 0;
}

void RegisterTwin(unsigned int primary, unsigned int twin)
{
    std::lock_guard lock(g_mutex);
    g_twins[primary] = twin;
    g_resetPrimary = g_resetTwin = true;
    LOG_INFO("Per-eye DLSS DX12: feature {} has twin {} (odd Presents go to the twin while routing is per eye)",
             primary, twin);
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
    if (!g_seenEvaluate)
    {
        g_seenEvaluate = true;
        g_windowStart = g_present.load();
    }
    ++g_evalsThisPresent;
    auto it = g_twins.find(handle);
    // Neural Rendering runs inside this evaluate with a history of its own: name the eye, so NR keeps one history per
    // eye like the twin features (unnamed, NR shares one history between the eyes -- FC6's "quicksand").
    // Set on every call: the parameters are the game's and keep a value between evaluates. Shared routing used to name
    // eye 0: NR's one history then took the eyes in turn, and a game flickered in stereo and not in
    // mono. Now the parity names the eye whenever this module is on (UpscalerNrEyeByParity); in a mono picture that
    // only splits one stream into two half-rate histories. The NR summary counts the evaluates per eye.
    const bool byParity = Config::Instance()->UpscalerNrEyeByParity.value_or_default();
    const bool perEye = (it != g_twins.end() && g_routing) || NrOnlyMode() || byParity;
    if (params != nullptr)
        params->Set(DlssNr::kNrEyeParam, perEye ? (unsigned int) (g_present.load() & 1) : 0u);
    if (!g_loggedNrEyeByParity && params != nullptr)
    {
        g_loggedNrEyeByParity = true;
        LOG_INFO("Per-eye DLSS DX12: NR is told the eye by the Present's parity {} (UpscalerNrEyeByParity {})",
                 byParity ? "in every routing" : "only while the routing is per eye (shared = eye 0, as before)",
                 byParity ? "true" : "false");
    }
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
        LOG_INFO("Per-eye DLSS DX12: history reset for the {} (routing {})", toTwin ? "twin" : "game's feature",
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

// The motion statistics of what DLSS is about to get (the game's vectors, or ours per eye). Under g_mutex.
void RunStats(ID3D12GraphicsCommandList* cmdList, ID3D12Device* device, NVSDK_NGX_Parameter* params)
{
    if (!g_setupTried)
        Setup(device);
    if (!g_ready)
        return; // the reason was logged once by Setup

    ID3D12Resource* motion = GetResource(params, NVSDK_NGX_Parameter_MotionVectors);
    if (motion == nullptr)
        return LogOnce(g_loggedNoMotion, "motion statistics skipped: no motion vectors in the DLSS parameters");
    float scale[2] = { 0, 0 };
    params->Get(NVSDK_NGX_Parameter_MV_Scale_X, &scale[0]);
    params->Get(NVSDK_NGX_Parameter_MV_Scale_Y, &scale[1]);
    if (scale[0] == 0 || scale[1] == 0)
        return LogOnce(g_loggedNoMotion, "motion statistics skipped: MV scale is zero");

    const uint32_t slot = (uint32_t) (g_statsSeq % kRing);
    ReadSlot(slot);
    g_slotParity[slot] = (int) (g_present.load() & 1);

    const D3D12_RESOURCE_DESC md = motion->GetDesc();
    D3D12_SHADER_RESOURCE_VIEW_DESC srv {};
    srv.Format = MotionSrvFormat(md.Format);
    srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srv.Texture2D.MipLevels = 1;
    D3D12_CPU_DESCRIPTOR_HANDLE cpu = g_heap->GetCPUDescriptorHandleForHeapStart();
    cpu.ptr += (SIZE_T) slot * g_descriptorSize;
    D3D12_GPU_DESCRIPTOR_HANDLE gpu = g_heap->GetGPUDescriptorHandleForHeapStart();
    gpu.ptr += (UINT64) slot * g_descriptorSize;
    device->CreateShaderResourceView(motion, &srv, cpu);

    uint32_t c[4] = { 0, 0, slot, (uint32_t) g_statsSeq };
    std::memcpy(&c[0], &scale[0], 4);
    std::memcpy(&c[1], &scale[1], 4);

    // The vectors are read as DLSS is about to read them: no barrier on the game's resource. The upscaler binds its
    // own heap, root signature and pipeline after this, as OptiScaler's own passes on this list already do.
    ID3D12DescriptorHeap* heaps[] = { g_heap };
    cmdList->SetDescriptorHeaps(1, heaps);
    cmdList->SetComputeRootSignature(g_rootSignature);
    cmdList->SetPipelineState(g_pipeline);
    cmdList->SetComputeRoot32BitConstants(0, 4, c, 0);
    cmdList->SetComputeRootDescriptorTable(1, gpu);
    cmdList->SetComputeRootUnorderedAccessView(2, g_sums->GetGPUVirtualAddress());
    cmdList->Dispatch(1, 1, 1);

    // Our sums buffer only. Buffers decay to COMMON when a list completes, and the dispatch promotes COMMON -> UAV; a
    // second evaluate on the same list finds it in UAV because we always leave it there. Either way it is UAV here.
    D3D12_RESOURCE_BARRIER barrier {};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = g_sums;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    cmdList->ResourceBarrier(1, &barrier);
    cmdList->CopyBufferRegion(g_readback, slot * 16, g_sums, slot * 16, 16);
    std::swap(barrier.Transition.StateBefore, barrier.Transition.StateAfter);
    cmdList->ResourceBarrier(1, &barrier);
    ++g_statsSeq;
}

void BeforeEvaluate(ID3D12GraphicsCommandList* cmdList, NVSDK_NGX_Parameter* params)
{
    if (!Enabled() || cmdList == nullptr || params == nullptr)
        return;
    std::lock_guard lock(g_mutex);
    g_swapped = false;
    g_maskHanded = false;
    g_composedForShared = false;
    // The parameters are the game's and keep values between evaluates: NR's are cleared here, set below when used.
    params->Set(DlssNr::kNrMotionParam, (void*) nullptr);
    params->Set(DlssNr::kNrResetParam, 0u);
    // Under R.E.A.L. VR's DL(S)SS nothing of ours is fitted or composed; NR, the crop
    // and the periphery read them as they are. Otherwise, in shared routing, ours are made for NR and for the crop /
    // periphery (their histories are per eye since UpscalerCropEyeByParity) while DLSS keeps the game's.
    const bool vectorsSameEye = Config::Instance()->UpscalerVectorsSameEye.value_or_default();
    // g_mutex is held here: the raw flags, never FoveaOn() (it locks g_mutex itself; the recursive lock threw a C++
    // exception at every evaluate).
    // The crop needs our same-eye motion unless DL(S)SS's serve it AND the periphery takes none of ours (one feature on
    // the game's vectors, our own TAA, or no cheap periphery), and the eye is told apart already (LeftParity: the fits'
    // votes place the crop and the gaze per eye; once decided they stand).
    const auto* cfg = Config::Instance();
    const bool cropOn = !g_routing && g_foveaOn && !g_foveaFailed;
    const bool cropTakesBelow =
        g_cropIsRr.load() ? cfg->UpscalerFoveaRrVectors.value_or_default() == 1 : SrCropFromBelow();
    const bool periphTakesOurs = cfg->UpscalerFoveaCheapPeriphery.value_or_default() &&
                                 !cfg->UpscalerFoveaPeripheryTaa.value_or_default() && !PeripheryOneFeature();
    const bool fitNotNeeded = cropTakesBelow && !periphTakesOurs && LeftParity() >= 0;
    // NR after SR reads the block's vectors after the game's (or the periphery's) evaluate: under a per-eye DLSS
    // pass below us those are same-eye already, so NR needs none of ours (kNrMotionParam stays cleared). NR before SR runs before
    // anything below us touches them, and the finished-picture mode is left as it was: both keep our fit. Only with the crop on (the vote runs on
    // every evaluate; NR without the crop keeps our fit).
    const bool nrOnBelow = cropOn && NrWantsSameEye() && g_belowSeen.load() && LeftParity() >= 0 &&
                           !cfg->DlssNrRunBeforeSr.value_or_default() && !cfg->DlssNrFinishedPicture.value_or_default();
    const bool nrWantsOurs = NrWantsSameEye() && !nrOnBelow;
    if (nrOnBelow)
        ++g_nrOnBelow;
    const bool nrOnly = !vectorsSameEye && (nrWantsOurs || (cropOn && !fitNotNeeded));
    if (!vectorsSameEye && cropOn && fitNotNeeded && !nrWantsOurs)
    {
        ++g_fitSkippedBelow;
        LogOnce(g_loggedFitSkippedBelow, "same-eye fit skipped: DL(S)SS's vectors serve the crop (and NR after SR, if "
                                         "on) and the periphery takes none of ours, the eye is told apart (counted per "
                                         "window; it runs again when DL(S)SS is no longer seen, or NR runs before SR)");
    }
    if (!g_routing && !TimingOn() && !nrOnly)
        return;

    ID3D12Device* device = nullptr;
    if (FAILED(cmdList->GetDevice(IID_PPV_ARGS(&device))) || device == nullptr)
        return LogOnce(g_loggedOtherDevice, "passes skipped: no device from the evaluate's list");
    device->Release(); // compared only; the device outlives the list
    if (g_device == nullptr)
        g_device = device;
    if (device != g_device)
    {
        if (g_routing)
            params->Set(NVSDK_NGX_Parameter_Reset, 1);
        return LogOnce(g_loggedOtherDevice, "passes skipped: an evaluate on another device");
    }

    g_timingThis = false;
    if (TimingOn())
    {
        if (!g_timeTried)
            SetupTiming(device);
        if (g_timeReady)
        {
            g_timeSlotCur = (uint32_t) (g_timeSeq % kRing);
            ReadTimeSlot(g_timeSlotCur);
            g_timeMode[g_timeSlotCur] = g_routing ? 1 : 0;
            g_timeParity[g_timeSlotCur] = (int) (g_present.load() & 1);
            g_timingThis = true;
            cmdList->EndQuery(g_queryHeap, D3D12_QUERY_TYPE_TIMESTAMP, g_timeSlotCur * 6 + 0);
        }
    }

    // Only evaluates that went to a feature with a twin are per eye (Route counted the others as "no twin").
    g_wroteFitTime = false;
    g_wroteUpscaleTime = false;
    if (g_routing && !g_twins.empty() && !vectorsSameEye)
        SameEye(cmdList, device, params);
    else if (nrOnly)
        SameEye(cmdList, device, params, true);
    if (g_timingThis)
    {
        if (!g_wroteFitTime) // no fit ran: T1 = T2
            cmdList->EndQuery(g_queryHeap, D3D12_QUERY_TYPE_TIMESTAMP, g_timeSlotCur * 6 + 1);
        cmdList->EndQuery(g_queryHeap, D3D12_QUERY_TYPE_TIMESTAMP, g_timeSlotCur * 6 + 2);
    }
    if (ProbeOn())
        RunStats(cmdList, device, params);
    if (g_timingThis)
        cmdList->EndQuery(g_queryHeap, D3D12_QUERY_TYPE_TIMESTAMP, g_timeSlotCur * 6 + 3);
}

void MarkCropBlock(NVSDK_NGX_Parameter* cropParams)
{
    if (cropParams == nullptr)
        return;
    cropParams->Set("RealVR.IgnoreFeature", 1);
}

void MainVectorsBefore(NVSDK_NGX_Parameter* params)
{
    if (!Enabled() || params == nullptr)
        return;
    std::lock_guard lock(g_mutex);
    g_mainMvHanded = GetResource(params, NVSDK_NGX_Parameter_MotionVectors);
    g_mainMvBelow = nullptr;
    g_mainRan = true;
    g_mainScaleHanded[0] = g_mainScaleHanded[1] = 0;
    params->Get(NVSDK_NGX_Parameter_MV_Scale_X, &g_mainScaleHanded[0]);
    params->Get(NVSDK_NGX_Parameter_MV_Scale_Y, &g_mainScaleHanded[1]);
}

void MainVectorsAfter(NVSDK_NGX_Parameter* params, bool evaluated)
{
    if (!Enabled() || params == nullptr)
        return;
    std::lock_guard lock(g_mutex);
    ID3D12Resource* now = GetResource(params, NVSDK_NGX_Parameter_MotionVectors);
    // DL(S)SS's buffer might carry another MV scale than the game's: the crop hands the block's scale on with them.
    float scaleNow[2] = { 0, 0 };
    params->Get(NVSDK_NGX_Parameter_MV_Scale_X, &scaleNow[0]);
    params->Get(NVSDK_NGX_Parameter_MV_Scale_Y, &scaleNow[1]);
    ++g_mainChecked;
    // Every full-size evaluate votes for g_belowSeen (the periphery's and the SR crop's autos), in shared routing only:
    // per eye our own same-eye vectors are in the block.
    if (evaluated && !g_routing)
    {
        const bool below = now != nullptr && now != g_mainMvHanded;
        g_periphBelowRun = below ? g_periphBelowRun + 1 : 0;
        g_periphOursRun = below ? 0 : g_periphOursRun + 1;
        const bool was = g_belowSeen.load();
        if (!was && g_periphBelowRun >= kOneOnAfter)
            g_belowSeen = true;
        else if (was && g_periphOursRun >= kOneOffAfter)
            g_belowSeen = false;
        if (g_belowSeen.load() != was)
        {
            ++g_periphAutoFlips;
            const bool periphAuto = Config::Instance()->UpscalerPeripheryOneFeature.value_or_default() == 0;
            if (periphAuto && !g_routing)
                for (auto& entry : g_periphReset)
                    entry.second = true;
            LOG_INFO("Per-eye DLSS DX12: something below us {} the full-size pass's vectors on {} "
                     "evaluates in a row -> DL(S)SS {}: periphery features {}, SR crop's vectors {}",
                     was ? "left" : "replaced", was ? kOneOffAfter : kOneOnAfter, was ? "NOT SEEN" : "SEEN",
                     periphAuto ? (was ? "one per eye (auto; history restarts)" : "ONE for both eyes (auto; history "
                                                                                  "restarts)")
                                : "as chosen (not auto)",
                     Config::Instance()->UpscalerFoveaSrVectors.value_or_default() < 0
                         ? (was ? "ours (auto)" : "FROM BELOW (auto)")
                         : "as chosen (not auto)");
        }
    }
    if (evaluated && now != nullptr && now != g_mainMvHanded)
    {
        g_mainMvBelow = now;
        g_mainScaleBelow[0] = scaleNow[0];
        g_mainScaleBelow[1] = scaleNow[1];
        ++g_mainSwapped;
        if (!g_loggedMainSwap)
        {
            g_loggedMainSwap = true;
            LOG_INFO("Per-eye DLSS DX12: FIRST full-size evaluate whose vectors were CHANGED below us: "
                     "handed {} -> in the block after it {}; MV scale handed ({}, {}) -> after ({}, {}); a "
                     "crop can take these ([Inputs] UpscalerFoveaRrVectors / UpscalerFoveaSrVectors = 1)",
                     DescribeTexture(g_mainMvHanded), DescribeTexture(now), g_mainScaleHanded[0],
                     g_mainScaleHanded[1], scaleNow[0], scaleNow[1]);
        }
    }
    else if (!g_loggedMainSame)
    {
        g_loggedMainSame = true;
        LOG_INFO("Per-eye DLSS DX12: FIRST full-size evaluate whose vectors were LEFT AS HANDED ({}; evaluated {}; MV "
                 "scale handed ({}, {}) -> after ({}, {})): nothing below us replaced them, or it put them back",
                 DescribeTexture(g_mainMvHanded), evaluated, g_mainScaleHanded[0], g_mainScaleHanded[1], scaleNow[0],
                 scaleNow[1]);
    }
}

// The periphery evaluates on a parameter block of its own (needed under alternate-eye rendering, found in Crimson
// Desert: a periphery made on the game's block had its evaluates fail on alternate Presents). What the pass below us
// left in that block -- the vectors and their scale -- is handed down to the game's block here, so the crop's copy and NR after SR read exactly what
// they read when the periphery ran on the game's block; RestoreInputs puts the game's own back after NR.
void HandBelowVectorsToGame(NVSDK_NGX_Parameter* game)
{
    if (!Enabled() || game == nullptr)
        return;
    std::lock_guard lock(g_mutex);
    if (g_mainMvBelow == nullptr)
        return;
    if (!g_swapped)
    {
        g_swappedFrom = GetResource(game, NVSDK_NGX_Parameter_MotionVectors);
        g_swapped = true;
    }
    game->Set(NVSDK_NGX_Parameter_MotionVectors, g_mainMvBelow);
    // The scale only where the game's block HOLDS one (a key it never set cannot be put back as absent).
    float scaleWas[2] = { 0, 0 };
    const bool hasScale = game->Get(NVSDK_NGX_Parameter_MV_Scale_X, &scaleWas[0]) == NVSDK_NGX_Result_Success &&
                          game->Get(NVSDK_NGX_Parameter_MV_Scale_Y, &scaleWas[1]) == NVSDK_NGX_Result_Success;
    if (hasScale)
    {
        if (!g_scaleHandedDown)
        {
            g_scaleHandedDown = true;
            g_scaleWas[0] = scaleWas[0];
            g_scaleWas[1] = scaleWas[1];
        }
        game->Set(NVSDK_NGX_Parameter_MV_Scale_X, g_mainScaleBelow[0]);
        game->Set(NVSDK_NGX_Parameter_MV_Scale_Y, g_mainScaleBelow[1]);
    }
    ++g_belowHandedDown;
    static bool logged = false;
    if (!logged)
    {
        logged = true;
        LOG_INFO("Per-eye DLSS DX12: FIRST hand-down of DL(S)SS's vectors from the periphery's own block to the game's "
                 "({}; MV scale ({}, {})): the crop's copy and NR after SR read them there (counted per window)",
                 DescribeTexture(g_mainMvBelow), g_mainScaleBelow[0], g_mainScaleBelow[1]);
    }
}

void EndEvaluate(ID3D12GraphicsCommandList* cmdList, NVSDK_NGX_Parameter* params)
{
    if (!Enabled())
        return;
    std::lock_guard lock(g_mutex);
    if (g_timingThis && cmdList != nullptr)
    {
        if (!g_wroteUpscaleTime) // no AfterUpscale: T4 = T5
            cmdList->EndQuery(g_queryHeap, D3D12_QUERY_TYPE_TIMESTAMP, g_timeSlotCur * 6 + 4);
        cmdList->EndQuery(g_queryHeap, D3D12_QUERY_TYPE_TIMESTAMP, g_timeSlotCur * 6 + 5);
        cmdList->ResolveQueryData(g_queryHeap, D3D12_QUERY_TYPE_TIMESTAMP, g_timeSlotCur * 6, 6, g_timeReadback,
                                  (UINT64) g_timeSlotCur * 48);
        g_timeFilled[g_timeSlotCur] = true;
        ++g_timeSeq;
    }
    g_timingThis = false;
}

void RestoreInputs(NVSDK_NGX_Parameter* params)
{
    if (!Enabled())
        return;
    std::lock_guard lock(g_mutex);
    if (g_swapped && params != nullptr)
        params->Set(NVSDK_NGX_Parameter_MotionVectors, g_swappedFrom);
    g_swapped = false;
    g_swappedFrom = nullptr;
    if (g_scaleHandedDown && params != nullptr)
    {
        params->Set(NVSDK_NGX_Parameter_MV_Scale_X, g_scaleWas[0]);
        params->Set(NVSDK_NGX_Parameter_MV_Scale_Y, g_scaleWas[1]);
    }
    g_scaleHandedDown = false;
    if (g_maskHanded && params != nullptr)
        params->Set(NVSDK_NGX_Parameter_DLSS_Input_Bias_Current_Color_Mask, g_maskWas);
    g_maskHanded = false;
    g_maskWas = nullptr;
}

void ToggleRouting()
{
    std::lock_guard lock(g_mutex);
    g_routing = !g_routing;
    g_resetPrimary = g_resetTwin = true;
    // The crops and the DLSS peripheries too: their history was built from the other mode's motion (and, per eye,
    // from one eye only). The TAA periphery restarts by its own Present-gap rule.
    for (auto& entry : g_cropReset)
        entry.second = true;
    for (auto& entry : g_periphReset)
        entry.second = true;
    LOG_INFO("Per-eye DLSS DX12: routing {}; all histories reset (features, crops, peripheries){}",
             g_routing ? "PER EYE" : "shared",
             g_routing && g_twins.empty() ? "; no twin yet: made at the next evaluate" : "");
}

bool RoutingOn()
{
    std::lock_guard lock(g_mutex);
    return g_routing;
}

// ---- Foveal crop ----

bool FoveaOn()
{
    if (!Enabled())
        return false;
    std::lock_guard lock(g_mutex);
    return g_foveaOn && !g_foveaFailed;
}

int FoveaBoxForNr(float& cx, float& cy, float& w, float& h)
{
    if (!Enabled())
        return 1;
    // Called from the NR module's evaluate, outside our lock; a try-lock so that an unexpected nesting costs one
    // frame of following, never a recursive lock (a std::mutex locked twice throws).
    std::unique_lock lock(g_mutex, std::try_to_lock);
    if (!lock.owns_lock())
        return 3;
    if (!g_foveaOn || g_foveaFailed)
        return 1;
    const uint64_t present = g_present.load();
    const unsigned int e = (unsigned int) (present & 1);
    const FoveaRect& r = g_foveaPlanEye[e];
    if (g_foveaPlanEyePresent[e] == UINT64_MAX || present - g_foveaPlanEyePresent[e] > 4 || r.outW == 0 ||
        r.outH == 0 || r.ow == 0 || r.oh == 0)
        return 2;
    cx = ((float) r.ox + 0.5f * (float) r.ow) / (float) r.outW;
    cy = ((float) r.oy + 0.5f * (float) r.oh) / (float) r.outH;
    w = (float) r.ow / (float) r.outW;
    h = (float) r.oh / (float) r.outH;
    return 0;
}

void ToggleFovea()
{
    std::lock_guard lock(g_mutex);
    g_foveaOn = !g_foveaOn;
    if (g_foveaOn)
        for (auto& entry : g_cropReset)
            entry.second = true;
    LOG_INFO("Per-eye DLSS DX12: foveal crop {}{}", g_foveaOn ? "ON" : "off",
             g_foveaFailed ? " -- but it failed earlier and stays off" : "");
}

void ToggleFoveaOutline()
{
    std::lock_guard lock(g_mutex);
    g_foveaOutline = !g_foveaOutline;
    LOG_INFO("Per-eye DLSS DX12: foveal crop outline {}", g_foveaOutline ? "on" : "off");
}

const char* FoveaName()
{
    std::lock_guard lock(g_mutex);
    if (g_foveaFailed)
        return "Fovea failed (see log)";
    return g_foveaOn ? (g_foveaOutline ? "Fovea on (outline)" : "Fovea on") : "Fovea off";
}

int PresetForCreate()
{
    return g_presetForCreate.load();
}

void SetPresetForCreate(int preset)
{
    g_presetForCreate = preset;
}

int KindForCreate()
{
    return g_kindForCreate.load();
}

void SetKindForCreate(int kind)
{
    g_kindForCreate = kind;
}

void NoteRrCreate(int kind, int preset)
{
    if (kind == 1)
        g_rrCropPreset = preset;
}

void NoteCropDirectCreate(bool ok, int result)
{
    if (ok)
        ++g_cropDirectMade;
    else
    {
        ++g_cropDirectFailed;
        g_cropDirectLastFail = result;
    }
}

void NoteCropDirectEvaluate(bool ok)
{
    ++(ok ? g_cropDirectEvals : g_cropDirectEvalFails);
}

void NoteCropCoreEvaluate()
{
    ++g_cropCoreEvals;
}

void NoteSrCropDirectCreate(bool ok, int result)
{
    if (ok)
        ++g_srDirectMade;
    else
    {
        ++g_srDirectFailed;
        g_srDirectLastFail = result;
    }
}

void NoteSrCropDirectEvaluate(bool ok)
{
    ++(ok ? g_srDirectEvals : g_srDirectEvalFails);
}

void NoteSrCropCoreEvaluate()
{
    ++g_srCoreEvals;
}

unsigned int FoveaCropOf(unsigned int handle)
{
    std::lock_guard lock(g_mutex);
    auto it = g_crops.find(EyeKey(handle));
    return it == g_crops.end() ? 0 : it->second;
}

std::vector<unsigned int> ForgetFoveaFeatures(unsigned int owner)
{
    std::vector<unsigned int> ids;
    for (unsigned int key : { owner, owner | kOddEyeKey })
        for (unsigned int id : { ForgetFoveaCrop(key), ForgetPeriphery(key) })
            if (id != 0)
                ids.push_back(id);
    return ids;
}

unsigned int ForgetFoveaCrop(unsigned int handle)
{
    std::lock_guard lock(g_mutex);
    auto it = g_crops.find(handle);
    if (it == g_crops.end())
        return 0;
    const unsigned int crop = it->second;
    g_crops.erase(it);
    g_cropRects.erase(crop);
    g_cropReset.erase(crop);
    g_cropOutPos.erase(crop);
    if (auto a = g_age.find(crop); a != g_age.end())
    {
        for (GatherTex& t : a->second.tex)
            if (t.res != nullptr)
                g_retired.push_back({ t.res, g_present.load() }); // work recorded a moment ago may reference it
        g_age.erase(a);
    }
    g_cropPos.erase(crop);
    g_dlssRebuild.erase(crop);
    return crop;
}

// The projection the gaze and the calibration targets are placed with (row vectors: clip = v * M): the one the game
// hands RR (ViewToClipMatrix, the projection the eye was rendered with) or, in a game with DLSS SR
// alone, one built from the HEADSET's per-view FOV handed over by the OpenXR layer (StaleEyeShareAbi.h,
// XrObserverViewsV1): view 0 for the left eye, 1 for the right, 0 while the eye is not told apart yet. The VR mod may
// render with another FOV than the headset's; the in-game calibration absorbs an affine difference (the targets are
// drawn through this same projection). nullptr, logged once, when there is neither.
const float* GazeProjection(NVSDK_NGX_Parameter* params, int eyeSign, float built[16])
{
    void* proj = nullptr;
    if (params->Get("ViewToClipMatrix", &proj) == NVSDK_NGX_Result_Success && proj != nullptr)
        return (const float*) proj;
    XrObserverViewsV1 v { sizeof(v) };
    if (!StaleEyeShare::LayerViews(&v) || v.viewCount < 2)
    {
        LogOnce(g_loggedGazeSkip, "gaze marker skipped: no ViewToClipMatrix (only ray reconstruction hands one) and no "
                                  "view FOV from the OpenXR layer (xr_observer.dll not loaded, an older build, "
                                  "or no xrLocateViews yet)");
        return nullptr;
    }
    const int view = eyeSign < 0 ? 1 : 0;
    // Which FOV, and whether its offsets apply (UpscalerFoveaGazeProjection). The headset's located FOV is off-centre
    // toward the outside of each eye (e.g. -1.58..+0.71 for the left eye); the submitted FOV is not the
    // located one -- with the offsets applied, a straight gaze sat 38% off centre toward the nose in each
    // eye. Default: the scale alone, centred.
    // Measured: the SUBMITTED FOV is symmetric (a centred projection); the three
    // centred candidates agreed, what remained was the tracker's
    // own bias (pitch ~5 deg low), which the calibration removes. Default: the submitted FOV scaled to the eye's image.
    static const char* const kModeNames[] = { "centred, headset's scale", "headset's FOV as located",
                                              "submitted FOV (scaled to the eye)", "submitted FOV, centred (scaled)" };
    const int mode = std::clamp(Config::Instance()->UpscalerFoveaGazeProjection.value_or_default(), 0, 3);
    const bool submitted = mode >= 2;
    if (submitted && v.submits == 0)
    {
        LogOnce(g_loggedGazeSkip, "gaze marker skipped: 'Submitted FOV' is chosen but the layer has seen no submitted "
                                  "projection layer yet");
        return nullptr;
    }
    const float* f = submitted ? v.submittedFov[view] : v.fov[view];
    float l = f[0], rt = f[1], u = f[2], d = f[3];
    // The submitted FOV is the IMAGE RECT's (a part of the eye at the same pixel scale, the centre kept), so for the eye's full image its tangents are scaled by the sizes' ratio, the centre kept.
    if (submitted && v.imageRect[view][2] > 0 && v.imageRect[view][3] > 0 && g_foveaPlan.outW > 0 && g_foveaPlan.outH > 0)
    {
        const float sx = (float) g_foveaPlan.outW / (float) v.imageRect[view][2];
        const float sy = (float) g_foveaPlan.outH / (float) v.imageRect[view][3];
        const float cx = 0.5f * (l + rt), cy = 0.5f * (u + d);
        l = cx + (l - cx) * sx;
        rt = cx + (rt - cx) * sx;
        u = cy + (u - cy) * sy;
        d = cy + (d - cy) * sy;
    }
    if (!(rt > l + 1e-4f && u > d + 1e-4f))
    {
        LogOnce(g_loggedGazeSkip, std::format("gaze marker skipped: the layer's view {} FOV ({}) is degenerate (tangents "
                                              "{:.3f} {:.3f} {:.3f} {:.3f})",
                                              view, kModeNames[mode], l, rt, u, d));
        return nullptr;
    }
    if (mode == 0 || mode == 3)
    {
        const float hx = 0.5f * (rt - l), hy = 0.5f * (u - d);
        l = -hx;
        rt = hx;
        d = -hy;
        u = hy;
    }
    std::memset(built, 0, 16 * sizeof(float));
    built[0] = 2.0f / (rt - l);
    built[5] = 2.0f / (u - d);
    built[8] = -(rt + l) / (rt - l);
    built[9] = -(u + d) / (u - d);
    built[11] = 1.0f;
    static int loggedMode = -1;
    if (!g_loggedGazeProjection || loggedMode != mode)
    {
        g_loggedGazeProjection = true;
        loggedMode = mode;
        // Where a STRAIGHT gaze lands (ndc) under each candidate, per view: the number to check against the green box.
        const auto centre = [](const float t[4], float& nx, float& ny) {
            nx = -(t[1] + t[0]) / (t[1] - t[0]);
            ny = -(t[3] + t[2]) / (t[2] - t[3]);
        };
        float lx0, ly0, lx1, ly1, sx0 = 0, sy0 = 0, sx1 = 0, sy1 = 0;
        centre(v.fov[0], lx0, ly0);
        centre(v.fov[1], lx1, ly1);
        if (v.submits > 0)
        {
            centre(v.submittedFov[0], sx0, sy0);
            centre(v.submittedFov[1], sx1, sy1);
        }
        LOG_INFO("Per-eye DLSS DX12: gaze projection '{}' from the OpenXR layer (no ViewToClipMatrix: DLSS SR only): view "
                 "{} for the {} eye, tangents in use L {:.4f} R {:.4f} U {:.4f} D {:.4f}. Located FOV view 0 L {:.4f} R "
                 "{:.4f} U {:.4f} D {:.4f} (a straight gaze at ndc {:+.3f} {:+.3f}), view 1 L {:.4f} R {:.4f} U {:.4f} D "
                 "{:.4f} ({:+.3f} {:+.3f}); submitted FOV view 0 L {:.4f} R {:.4f} U {:.4f} D {:.4f} ({:+.3f} {:+.3f}), "
                 "view 1 L {:.4f} R {:.4f} U {:.4f} D {:.4f} ({:+.3f} {:+.3f}), image rects {}x{} at {},{} / {}x{} at "
                 "{},{} of swapchains {}x{} / {}x{} ({} submits); the eye here is {}x{}. The layer's gaze: {}",
                 kModeNames[mode], view, eyeSign > 0 ? "left" : eyeSign < 0 ? "right" : "unknown", l, rt, u, d,
                 v.fov[0][0], v.fov[0][1], v.fov[0][2], v.fov[0][3], lx0, ly0, v.fov[1][0], v.fov[1][1], v.fov[1][2],
                 v.fov[1][3], lx1, ly1, v.submittedFov[0][0], v.submittedFov[0][1], v.submittedFov[0][2],
                 v.submittedFov[0][3], sx0, sy0, v.submittedFov[1][0], v.submittedFov[1][1], v.submittedFov[1][2],
                 v.submittedFov[1][3], sx1, sy1, v.imageRect[0][2], v.imageRect[0][3], v.imageRect[0][0],
                 v.imageRect[0][1], v.imageRect[1][2], v.imageRect[1][3], v.imageRect[1][0], v.imageRect[1][1],
                 v.swapchainSize[0][0], v.swapchainSize[0][1], v.swapchainSize[1][0], v.swapchainSize[1][1], v.submits,
                 g_foveaPlan.outW, g_foveaPlan.outH, v.text);
    }
    if (eyeSign == 0 && !g_loggedGazeEyeUnknown)
    {
        g_loggedGazeEyeUnknown = true;
        LOG_INFO("Per-eye DLSS DX12: the eye of this Present is not told apart yet (the fits' parallax vote needs 20 "
                 "accepted fits of each parity): view 0's FOV and gaze serve both eyes until then");
    }
    return built;
}

// Where this evaluate's eye looks, in output pixels: this eye's gaze direction (x right, y up, the forward axis taken as
// positive -- live flips in the overlay if the signs are wrong; from the OpenXR layer in the eye's own frame, or Varjo's
// HMD frame for both eyes) through GazeProjection. `eyeSign` 1 = left eye, -1 = right, 0 = unknown (then view 0 / both
// eyes' gaze). Under g_mutex.
void PlaceGaze(NVSDK_NGX_Parameter* params, int eyeSign, FoveaRect& r)
{
    r.gaze = false;
    r.target = false;
    if (!VarjoGaze::Enabled())
        return;
    float built[16];
    const float* m = nullptr;
    // A calibration target: a direction of the rendered view itself, through the same projection.
    if (float ty = 0, tp = 0; VarjoGaze::CalibrationTarget(ty, tp))
    {
        if (m == nullptr)
            m = GazeProjection(params, eyeSign, built);
        if (m != nullptr)
        {
            const float yr = ty * 0.01745329f, pr = tp * 0.01745329f;
            const float x = std::sin(yr) * std::cos(pr), y = std::sin(pr), z = std::cos(yr) * std::cos(pr);
            const float w = x * m[3] + y * m[7] + z * m[11];
            if (w > 1e-4f)
            {
                const float nx = (x * m[0] + y * m[4] + z * m[8]) / w, ny = (x * m[1] + y * m[5] + z * m[9]) / w;
                r.targetX = (uint32_t) std::clamp((nx * 0.5f + 0.5f) * (float) r.outW, 0.0f, (float) r.outW - 1.0f);
                r.targetY = (uint32_t) std::clamp((0.5f - ny * 0.5f) * (float) r.outH, 0.0f, (float) r.outH - 1.0f);
                r.target = true;
            }
        }
    }
    const VarjoGaze::Sample g = VarjoGaze::Latest();
    if (!g.valid)
        return;
    if (m == nullptr)
        m = GazeProjection(params, eyeSign, built);
    if (m == nullptr)
        return;
    // The smoothed gaze for this eye: from the layer, in this eye's own frame (the headset's cant applied); from Varjo,
    // both eyes' gaze in the HMD frame for either eye's image -- the raw per-eye rays made the marker jumpy:
    // noise, plus two different rays on alternating frames; the parallax given up is a few pixels for anything but the
    // very near, well inside the crop. Then the flips and the overlay's offset.
    const float* d = g.viewDir[eyeSign < 0 ? 1 : 0];
    float c[3];
    VarjoGaze::Correct(d, c);
    const float x = c[0], y = c[1], z = c[2];
    const float w = x * m[3] + y * m[7] + z * m[11];
    if (z < 0.05f || w <= 1e-4f)
        return LogOnce(g_loggedGazeSkip, std::format("gaze marker skipped: gaze ({:.3f} {:.3f} {:.3f}) does not project "
                                                     "in front of the eye (w {:.4f})",
                                                     d[0], d[1], d[2], w));
    const float nx = (x * m[0] + y * m[4] + z * m[8]) / w;
    const float ny = (x * m[1] + y * m[5] + z * m[9]) / w;
    const float px = (nx * 0.5f + 0.5f) * (float) r.outW;
    const float py = (0.5f - ny * 0.5f) * (float) r.outH;
    r.gazeX = (uint32_t) std::clamp(px, 0.0f, (float) r.outW - 1.0f);
    r.gazeY = (uint32_t) std::clamp(py, 0.0f, (float) r.outH - 1.0f);
    r.gaze = true;
    g_lastGazeEye = eyeSign > 0 ? 0 : eyeSign < 0 ? 1 : 2;
    if (!g_loggedGazeFirst)
    {
        g_loggedGazeFirst = true;
        LOG_INFO("Per-eye DLSS DX12: FIRST gaze placed: {} eye, direction ({:.3f} {:.3f} {:.3f}) from {} -> ndc ({:.3f} "
                 "{:.3f}) -> pixel ({:.0f}, {:.0f}) of {}x{} (projection scale {:.4f} {:.4f}, offset {:.4f} {:.4f}, {})",
                 eyeSign > 0 ? "left" : eyeSign < 0 ? "right" : "unknown (both)", d[0], d[1], d[2],
                 g.source == 1 ? "the OpenXR layer" : "Varjo's reader", nx, ny, px, py, r.outW, r.outH, m[0], m[5], m[8],
                 m[9], m == built ? "built from the layer's view FOV" : "the game's ViewToClipMatrix");
    }
}

bool PlanFovea(ID3D12GraphicsCommandList* cmdList, NVSDK_NGX_Parameter* params, unsigned int handle,
               FoveaSize* create)
{
    if (!Enabled() || cmdList == nullptr || params == nullptr || create == nullptr)
        return false;
    std::lock_guard lock(g_mutex);
    if (!g_foveaOn || g_foveaFailed)
        return false;
    ID3D12Device* device = nullptr;
    if (FAILED(cmdList->GetDevice(IID_PPV_ARGS(&device))) || device == nullptr)
        return false;
    device->Release(); // compared only
    if (device != g_device)
    {
        ++g_foveaSkips;
        LogOnce(g_loggedFoveaSkip, "foveal crop skipped: an evaluate on another device (counted per window)");
        return false;
    }
    ID3D12Resource* output = GetResource(params, NVSDK_NGX_Parameter_Output);
    unsigned int rw = GetUIntParam(params, NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Width);
    unsigned int rh = GetUIntParam(params, NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Height);
    if (output == nullptr || rw == 0 || rh == 0)
    {
        ++g_foveaSkips;
        LogOnce(g_loggedFoveaSkip, std::format("foveal crop skipped: {} (counted per window)",
                                               output == nullptr ? "no output" : "no render subrect size"));
        return false;
    }
    const D3D12_RESOURCE_DESC od = output->GetDesc();
    if (!(od.Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS))
    {
        ++g_foveaSkips;
        LogOnce(g_loggedFoveaSkip, std::format("foveal crop skipped: the output (fmt {}) allows no UAV, so the crop "
                                               "cannot be blended into it (counted per window)",
                                               (int) od.Format));
        return false;
    }
    FoveaRect r {};
    r.renderW = rw;
    r.renderH = rh;
    r.outW = (uint32_t) od.Width;
    r.outH = od.Height;
    // Width and height fractions of the eye; a height of 0 (the default) follows the width.
    const float f = Config::Instance()->UpscalerFoveaCrop.value_or_default();
    const float fhSet = Config::Instance()->UpscalerFoveaCropHeight.value_or_default();
    const float fh = fhSet > 0.0f ? fhSet : f;
    bool ex = false, ey = false;
    const bool snap = !(VarjoGaze::Enabled() && Config::Instance()->UpscalerFoveaGazeFollow.value_or_default());
    if (!CropAxis(rw, r.outW, f, snap, r.rx, r.rw, r.ox, r.ow, ex) ||
        !CropAxis(rh, r.outH, fh, snap, r.ry, r.rh, r.oy, r.oh, ey))
    {
        ++g_foveaSkips;
        LogOnce(g_loggedFoveaSkip, std::format("foveal crop skipped: no crop of {:.2f} x {:.2f} fits render {}x{} -> "
                                               "output {}x{} (counted per window)",
                                               f, fh, rw, rh, r.outW, r.outH));
        return false;
    }
    r.exact = ex && ey;

    // Position: the horizontal offset is toward the nose, mirrored per eye; the vertical one is up. Any offset gives up
    // the exact alignment (edges up to half an output pixel off, inside the blended edge). The eye is known per eye,
    // and in shared routing once the crop keeps a history per eye (UpscalerCropEyeByParity: a crop feature
    // per parity may sit in another place per eye); with one crop feature for both eyes it must not alternate.
    const float offX = Config::Instance()->UpscalerFoveaOffsetX.value_or_default();
    const float offY = Config::Instance()->UpscalerFoveaOffsetY.value_or_default();
    // Sideways: the same for both eyes (+ = right), so it needs no eye told apart.
    const float offSide = Config::Instance()->UpscalerFoveaOffsetSideways.value_or_default();
    int eyeSign = 0;
    if (g_routing || Config::Instance()->UpscalerCropEyeByParity.value_or_default())
    {
        const int left = LeftParity();
        if (left >= 0)
            eyeSign = ((int) (g_present.load() & 1) == left) ? 1 : -1;
    }
    float cx = 0.5f + offX * (float) eyeSign + offSide;
    float cy = 0.5f - offY;
    PlaceGaze(params, eyeSign, r);
    const bool follow = VarjoGaze::Enabled() && Config::Instance()->UpscalerFoveaGazeFollow.value_or_default();
    if (offX != 0.0f && eyeSign == 0)
    {
        static bool loggedNoseNoEye = false;
        if (!loggedNoseNoEye)
            LogOnce(loggedNoseNoEye, std::format("crop's 'Toward the nose' {:.3f} not applied yet: the eyes are not told "
                                             "apart (sideways {:.3f} and up {:.3f} apply)",
                                             offX, offSide, offY));
    }
    if (follow && r.gaze)
    {
        const float gx = (float) r.gazeX / (float) r.outW, gy = (float) r.gazeY / (float) r.outH;
        const float dz = std::clamp(Config::Instance()->UpscalerFoveaGazeDeadZone.value_or_default(), 0.0f, 0.95f);
        const float hx = 0.5f * (float) r.ow / (float) r.outW, hy = 0.5f * (float) r.oh / (float) r.outH;
        const float glideMs = std::clamp(Config::Instance()->UpscalerFoveaGazeGlideMs.value_or_default(), 0.0f, 500.0f);
        const auto now = std::chrono::steady_clock::now();
        // Capped: after a pause (a menu, a loading screen) one step must not be a jump.
        const float dt = g_followInit
                             ? std::min(std::chrono::duration<float, std::milli>(now - g_followLast).count(), 50.0f)
                             : 0.0f;
        g_followLast = now;
        if (!g_followInit)
        {
            g_followX = gx;
            g_followY = gy;
            g_followInit = true;
            ++g_followMoves;
        }
        else if (!g_followGliding && (std::abs(gx - g_followX) > dz * hx || std::abs(gy - g_followY) > dz * hy))
        {
            g_followGliding = true;
            ++g_followMoves;
        }
        if (g_followGliding)
        {
            const float k = glideMs > 0.0f ? 1.0f - std::exp(-dt / glideMs) : 1.0f;
            g_followX += (gx - g_followX) * k;
            g_followY += (gy - g_followY) * k;
            // Where a crop can be: a centre past the eye's edge would glide on with nothing moving.
            g_followX = std::clamp(g_followX, hx, 1.0f - hx);
            g_followY = std::clamp(g_followY, hy, 1.0f - hy);
            const float sx = std::clamp(gx, hx, 1.0f - hx), sy = std::clamp(gy, hy, 1.0f - hy);
            if (std::abs(sx - g_followX) < 0.1f * hx && std::abs(sy - g_followY) < 0.1f * hy)
                g_followGliding = false;
            if (!g_loggedGlide && glideMs > 0.0f)
            {
                g_loggedGlide = true;
                LOG_INFO("Per-eye DLSS DX12: FIRST crop glide: step {:.3f} of the way ({:.1f} ms since the last, time "
                         "constant {:.0f} ms), centre now ({:.3f}, {:.3f}) toward the gaze ({:.3f}, {:.3f})",
                         k, dt, glideMs, g_followX, g_followY, gx, gy);
            }
        }
    }
    if (follow && g_followInit)
    {
        cx = g_followX;
        cy = g_followY;
        static bool loggedOffsetsFollow = false;
        if (!loggedOffsetsFollow && (offX != 0.0f || offY != 0.0f || offSide != 0.0f))
            LogOnce(loggedOffsetsFollow, std::format("crop's offsets (nose {:.3f}, sideways {:.3f}, up {:.3f}) give way "
                                                     "to the gaze while the crop follows it",
                                                     offX, offSide, offY));
    }
    if (cx != 0.5f || cy != 0.5f)
    {
        // The crop's output origin is the render origin scaled and ROUNDED: the rounding (up to half an output pixel)
        // displaced everything inside the crop against the full frame, and a glide changed it at every step --
        // a shimmer on objects while the crop moves, settling once still (SR at 1.5; the
        // move's carry-over sign and a history restart were ruled out). UpscalerFoveaPlaceExact (default):
        // the render origin within 3 px of the wanted one whose output origin is nearest a whole pixel (at 1.5: even
        // origins, exact; at 1.72: within ~0.1 px). The error is tracked for the window summary either way.
        const bool exactPlace = Config::Instance()->UpscalerFoveaPlaceExact.value_or_default();
        auto place = [exactPlace](float c, uint32_t R, uint32_t O, uint32_t rwid, uint32_t owid, uint32_t& rb,
                                  uint32_t& ob)
        {
            const long maxR = (long) R - (long) rwid;
            const long want = std::clamp(std::lround(c * (double) R - rwid / 2.0), 0L, maxR);
            long best = want;
            if (exactPlace)
            {
                double bestErr = 1e9;
                for (long cand = (std::max)(0L, want - 3); cand <= (std::min)(maxR, want + 3); ++cand)
                {
                    const double o = (double) cand * O / R;
                    const double e = std::abs(o - std::round(o)) + 0.02 * (double) std::abs(cand - want);
                    if (e < bestErr)
                    {
                        bestErr = e;
                        best = cand;
                    }
                }
            }
            rb = (uint32_t) best;
            const double o = (double) best * O / R;
            g_placeErrMax = (std::max)(g_placeErrMax, std::abs(o - std::round(o)));
            const long maxO = (long) O - (long) owid;
            ob = (uint32_t) std::clamp(std::lround(o), 0L, maxO);
        };
        place(cx, rw, r.outW, r.rw, r.ow, r.rx, r.ox);
        place(cy, rh, r.outH, r.rh, r.oh, r.ry, r.oy);
        r.exact = false;
    }
    if (follow && g_followInit)
    {
        // The glide's centre is not overwritten by the placed one: rounded to whole pixels, a step smaller than half a
        // pixel would be lost every frame and the glide would stall short of the gaze.
        if (!g_loggedFollow)
        {
            g_loggedFollow = true;
            LOG_INFO("Per-eye DLSS DX12: FIRST crop placed by the gaze: centre ({:.3f}, {:.3f}) -> render box ({}, {}) "
                     "{}x{}, dead zone {:.2f} of the half crop",
                     g_followX, g_followY, r.rx, r.ry, r.rw, r.rh,
                     Config::Instance()->UpscalerFoveaGazeDeadZone.value_or_default());
        }
    }
    if (!g_loggedFoveaPlan)
    {
        g_loggedFoveaPlan = true;
        LOG_INFO("Per-eye DLSS DX12: foveal crop for render {}x{} -> output {}x{} (asked {:.2f} x {:.2f}): render box ({}, {}) "
                 "{}x{} -> output box ({}, {}) {}x{} = {:.1f}% of the eye; {}",
                 rw, rh, r.outW, r.outH, f, fh, r.rx, r.ry, r.rw, r.rh, r.ox, r.oy, r.ow, r.oh,
                 100.0 * r.ow * r.oh / ((double) r.outW * r.outH),
                 r.exact ? "EXACT (same ratio, whole pixels)" : "approximate (edges up to half an output pixel off)");
    }
    if (auto it = g_crops.find(EyeKey(handle)); it != g_crops.end())
    {
        auto& made = g_cropRects[it->second];
        // A new crop size (the overlay's size slider or the ini): the crop feature is rebuilt at it, live, by the staged
        // rebuild; BeginFovea hands it the new sizes at every step.
        if (made.renderW == r.renderW && made.renderH == r.renderH && made.outW == r.outW && made.outH == r.outH &&
            (made.rw != r.rw || made.rh != r.rh || made.ow != r.ow || made.oh != r.oh))
        {
            LOG_INFO("Per-eye DLSS DX12: foveal crop {} resized {}x{} -> {}x{} to {}x{} -> {}x{}; rebuilding it live",
                     it->second, made.rw, made.rh, made.ow, made.oh, r.rw, r.rh, r.ow, r.oh);
            made = r;
            State::Instance().changeBackend[it->second] = true;
            g_dlssRebuild[it->second] = 1;
        }
        if (made.renderW != r.renderW || made.renderH != r.renderH || made.outW != r.outW || made.outH != r.outH)
        {
            ++g_foveaSkips;
            LogOnce(g_loggedFoveaSkip, std::format("foveal crop skipped: its feature was made for {}x{} -> {}x{}, this "
                                                   "evaluate is {}x{} -> {}x{} (counted per window)",
                                                   made.renderW, made.renderH, made.outW, made.outH, r.renderW,
                                                   r.renderH, r.outW, r.outH));
            return false;
        }
    }
    if (!g_foveaSetupTried && !SetupFovea(device, TypedFormat(od.Format)))
        g_foveaFailed = true;
    if (!g_foveaReady)
        return false;
    g_foveaPlan = r;
    {
        const uint64_t present = g_present.load();
        g_foveaPlanEye[present & 1] = r;
        g_foveaPlanEyePresent[present & 1] = present;
    }
    create->renderW = r.rw;
    create->renderH = r.rh;
    create->outW = r.ow;
    create->outH = r.oh;
    create->preset = Config::Instance()->UpscalerFoveaPreset.value_or_default();
    return true;
}

void FoveaCreated(unsigned int handle, unsigned int crop)
{
    std::lock_guard lock(g_mutex);
    g_crops[EyeKey(handle)] = crop;
    g_cropRects[crop] = g_foveaPlan;
    g_cropReset[crop] = true;
    LOG_INFO("Per-eye DLSS DX12: foveal crop feature {} made for feature {} ({}x{} -> {}x{}, preset {})", crop, handle,
             g_foveaPlan.rw, g_foveaPlan.rh, g_foveaPlan.ow, g_foveaPlan.oh,
             Config::Instance()->UpscalerFoveaPreset.value_or_default());
}

void FoveaCreateFailed()
{
    std::lock_guard lock(g_mutex);
    g_foveaFailed = true;
    LOG_WARN("Per-eye DLSS DX12: foveal crop feature could not be made; the crop stays off this session");
}

void FoveaNotReady()
{
    std::lock_guard lock(g_mutex);
    ++g_foveaSkips;
    LogOnce(g_loggedFoveaNotReady, "foveal crop not made yet: the game's feature is being rebuilt; asked again at the "
                                   "next evaluate (counted per window with the skipped crops)");
}

// The keys DLSS and RR read, by the type they are handed in (NGX's SDK defines, RR's names from nvngx_dlssd.dll
// 310.9.1's strings -- kGuides and the input probe's list). Only keys the game's block HOLDS are copied: a key it never
// set stays absent in ours as well.
constexpr const char* kCropUintKeys[] = {
    NVSDK_NGX_Parameter_Width, NVSDK_NGX_Parameter_Height, NVSDK_NGX_Parameter_OutWidth,
    NVSDK_NGX_Parameter_OutHeight, NVSDK_NGX_Parameter_Reset, NVSDK_NGX_Parameter_PerfQualityValue,
    NVSDK_NGX_Parameter_CreationNodeMask, NVSDK_NGX_Parameter_VisibilityNodeMask,
    NVSDK_NGX_Parameter_DLSS_Feature_Create_Flags, NVSDK_NGX_Parameter_DLSS_Enable_Output_Subrects,
    NVSDK_NGX_Parameter_DLSS_Checkerboard_Jitter_Hack, NVSDK_NGX_Parameter_TonemapperType,
    NVSDK_NGX_Parameter_FreeMemOnReleaseFeature, NVSDK_NGX_Parameter_DLSS_Input_Color_Subrect_Base_X,
    NVSDK_NGX_Parameter_DLSS_Input_Color_Subrect_Base_Y, NVSDK_NGX_Parameter_DLSS_Input_Depth_Subrect_Base_X,
    NVSDK_NGX_Parameter_DLSS_Input_Depth_Subrect_Base_Y, NVSDK_NGX_Parameter_DLSS_Input_MV_SubrectBase_X,
    NVSDK_NGX_Parameter_DLSS_Input_MV_SubrectBase_Y, NVSDK_NGX_Parameter_DLSS_Input_Translucency_SubrectBase_X,
    NVSDK_NGX_Parameter_DLSS_Input_Translucency_SubrectBase_Y, NVSDK_NGX_Parameter_DLSS_Output_Subrect_Base_X,
    NVSDK_NGX_Parameter_DLSS_Output_Subrect_Base_Y, NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Width,
    NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Height, NVSDK_NGX_Parameter_DLSS_Indicator_Invert_X_Axis,
    NVSDK_NGX_Parameter_DLSS_Indicator_Invert_Y_Axis, NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_DLAA,
    NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_UltraQuality, NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Quality,
    NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Balanced, NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Performance,
    NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_UltraPerformance, "RayReconstruction.Hint.Render.Preset.DLAA",
    "RayReconstruction.Hint.Render.Preset.UltraQuality", "RayReconstruction.Hint.Render.Preset.Quality",
    "RayReconstruction.Hint.Render.Preset.Balanced", "RayReconstruction.Hint.Render.Preset.Performance",
    "RayReconstruction.Hint.Render.Preset.UltraPerformance", "DLSS.Denoise.Mode", "DLSS.Roughness.Mode",
    "DLSS.Use.HW.Depth", DlssNr::kNrResetParam, DlssNr::kNrEyeParam
};
constexpr const char* kCropFloatKeys[] = {
    NVSDK_NGX_Parameter_Jitter_Offset_X,  NVSDK_NGX_Parameter_Jitter_Offset_Y,     NVSDK_NGX_Parameter_MV_Scale_X,
    NVSDK_NGX_Parameter_MV_Scale_Y,       NVSDK_NGX_Parameter_MV_Offset_X,         NVSDK_NGX_Parameter_MV_Offset_Y,
    NVSDK_NGX_Parameter_Sharpness,        NVSDK_NGX_Parameter_DLSS_Pre_Exposure,   NVSDK_NGX_Parameter_DLSS_Exposure_Scale,
    NVSDK_NGX_Parameter_FrameTimeDeltaInMsec
};
// Resources besides kGuides (whose names and subrect bases are copied from that table).
constexpr const char* kCropResourceKeys[] = {
    NVSDK_NGX_Parameter_Color, NVSDK_NGX_Parameter_Output, NVSDK_NGX_Parameter_Depth,
    NVSDK_NGX_Parameter_MotionVectors, NVSDK_NGX_Parameter_TransparencyMask, NVSDK_NGX_Parameter_ExposureTexture,
    NVSDK_NGX_Parameter_DepthHighRes, NVSDK_NGX_Parameter_Position_ViewSpace, NVSDK_NGX_Parameter_MotionVectors3D,
    NVSDK_NGX_Parameter_RayTracingHitDistance, "GBuffer.Albedo", "GBuffer.DiffuseAlbedo", "GBuffer.SpecularAlbedo",
    "GBuffer.IndirectAlbedo", "GBuffer.DisocclusionMask", "GBuffer.Metallic", "GBuffer.Specular", "GBuffer.Subsurface",
    "GBuffer.ShadingModelId", "GBuffer.MaterialId", "GBuffer.Attrib.8", "GBuffer.Attrib.9", "GBuffer.Attrib.10",
    "GBuffer.Attrib.11", "GBuffer.Attrib.12", "GBuffer.Attrib.13", "GBuffer.Attrib.14", "GBuffer.Attrib.15",
    "GBuffer.Emissive", "DLSSD.DiffuseHitDistance", "DLSSD.SpecularRayDirection",
    "DLSSD.SpecularRayDirectionHitDistance", "DLSSD.DiffuseRayDirection", "DLSSD.DiffuseRayDirectionHitDistance",
    "DLSSD.OutputAlpha", "DLSS.Input.Reduce.Ghost.Mask"
};
// Handed as pointers (the camera matrices are 16 floats each).
constexpr const char* kCropPointerKeys[] = { "WorldToViewMatrix",       "ViewToClipMatrix",   "ClipToPrevClipMatrix",
                                             "InvViewProjectionMatrix", "WorldToScreenMatrix", DlssNr::kNrMotionParam };

bool CopyCropParameters(NVSDK_NGX_Parameter* game, NVSDK_NGX_Parameter* own)
{
    static bool loggedFirst = false, loggedMissing = false;
    int numbers = 0, floats = 0, pointers = 0;
    std::string resources;
    auto copyUint = [&](const char* name) {
        unsigned int v = 0;
        if (name != nullptr && game->Get(name, &v) == NVSDK_NGX_Result_Success)
        {
            own->Set(name, v);
            ++numbers;
        }
    };
    auto copyResource = [&](const char* name) {
        ID3D12Resource* r = nullptr;
        if (game->Get(name, &r) != NVSDK_NGX_Result_Success && game->Get(name, (void**) &r) != NVSDK_NGX_Result_Success)
            return;
        own->Set(name, r); // identity only: an NGX block holds no reference
        if (!loggedFirst)
            resources += std::string(resources.empty() ? "" : " ") + name + (r == nullptr ? "(null)" : "");
    };
    for (const char* name : kCropUintKeys)
        copyUint(name);
    for (const char* name : kCropFloatKeys)
    {
        float v = 0.0f;
        if (game->Get(name, &v) == NVSDK_NGX_Result_Success)
        {
            own->Set(name, v);
            ++floats;
        }
    }
    for (const char* name : kCropResourceKeys)
        copyResource(name);
    for (const GuideName& g : kGuides)
    {
        copyResource(g.name);
        copyUint(g.baseX);
        copyUint(g.baseY);
    }
    for (const char* name : kCropPointerKeys)
    {
        void* p = nullptr;
        if (game->Get(name, &p) == NVSDK_NGX_Result_Success)
        {
            own->Set(name, p);
            ++pointers;
        }
    }
    if (GetResource(own, NVSDK_NGX_Parameter_Color) == nullptr || GetResource(own, NVSDK_NGX_Parameter_Output) == nullptr)
    {
        if (!loggedMissing)
        {
            loggedMissing = true;
            LOG_WARN("Per-eye DLSS DX12: crop skipped: the game's block holds no colour or output to copy into the "
                     "crop's own block");
        }
        return false;
    }
    if (!loggedFirst)
    {
        loggedFirst = true;
        LOG_INFO("Per-eye DLSS DX12: the crop / periphery runs on a parameter block of its own (the game's is never written): "
                 "copied {} numbers, {} floats, {} pointers; resources: {}",
                 numbers, floats, pointers, resources);
    }
    return true;
}

bool BeginFovea(ID3D12GraphicsCommandList* cmdList, NVSDK_NGX_Parameter* params, unsigned int crop,
                bool rayReconstruction)
{
    std::lock_guard lock(g_mutex);
    if (g_cropRects.find(crop) == g_cropRects.end() || g_device == nullptr)
        return false;
    const FoveaRect r = g_foveaPlan;
    // DLSS keeps the crop's history in the crop's own coordinates: a moved box is a pan it cannot see. Following the
    // gaze, the move is added to the crop's copied vectors (a pixel's previous place in the crop's own coordinates is
    // its full-frame previous place minus the previous box origin: shift = new origin - old origin, render pixels,
    // divided by the MV scale to be in the vectors' units). Otherwise the history starts over, as before. A jump is
    // carried over as long as the boxes OVERLAP (it used to be only up to half the crop, 'where most of it is
    // new anyway' -- but the part that is not new keeps its history this way, and the new part has none either way;
    // 26 restarts in one run were such jumps, each a whole crop settling from scratch).
    float motionShift[2] = { 0.0f, 0.0f };
    bool shiftWanted = false;
    if (auto pos = g_cropPos.find(crop); pos == g_cropPos.end() || pos->second != std::make_pair(r.rx, r.ry))
    {
        const bool follow = VarjoGaze::Enabled() && Config::Instance()->UpscalerFoveaGazeFollow.value_or_default();
        float scale[2] = { 0.0f, 0.0f };
        params->Get(NVSDK_NGX_Parameter_MV_Scale_X, &scale[0]);
        params->Get(NVSDK_NGX_Parameter_MV_Scale_Y, &scale[1]);
        const long dx = pos != g_cropPos.end() ? (long) r.rx - (long) pos->second.first : 0;
        const long dy = pos != g_cropPos.end() ? (long) r.ry - (long) pos->second.second : 0;
        const bool nearMove = std::abs(dx) < (long) r.rw && std::abs(dy) < (long) r.rh; // the boxes overlap
        // UpscalerFoveaMoveMode (overlay 'Crop move'): 0 carry the move over (as derived), 1 the same with the shift's
        // sign flipped, 2 restart the history at every move -- the A/B for a shimmer on objects while the crop moves
        // (SR; the crop settles once still).
        const int moveMode = std::clamp(Config::Instance()->UpscalerFoveaMoveMode.value_or_default(), 0, 2);
        // The shift goes into the COPIED vectors (GatherCropInputs) or into a shifted copy of the
        // box of OUR composed vectors read through the subrects (ShiftComposed) -- the crop reads those in shared
        // routing (made for NR and the crop) and per eye (set on the block by SameEye) alike; RR always copies.
        // Without either (the vectors as the game's: 'Vectors are same-eye already' ticked, or no compose) no move
        // can be carried over and every move restarts the history (one run: 2416
        // restarts on 176 moves through subrects -- the shimmer while the gaze moved).
        const bool copies = (CropSubrectMask(rayReconstruction) & 4) == 0; // the VECTORS are copies
        const bool motionIsOurs = g_composed != nullptr && (!rayReconstruction || !copies) &&
                                  ((g_composedForShared && !g_routing) ||
                                   GetResource(params, NVSDK_NGX_Parameter_MotionVectors) == g_composed);
        const bool shiftOnSubrects = !copies && motionIsOurs;
        if (pos != g_cropPos.end() && follow && nearMove && !copies && !motionIsOurs && moveMode != 2 &&
            !g_loggedNoCopyMove)
        {
            g_loggedNoCopyMove = true;
            LOG_WARN("Per-eye DLSS DX12: the crop follows the gaze, reads through DLSS's subrects and has no composed "
                     "vectors of ours to shift ('Vectors are same-eye already' ticked, or no compose): a move cannot be "
                     "carried over, so its history restarts at EVERY move");
        }
        if (pos != g_cropPos.end() && follow && nearMove && scale[0] != 0.0f && scale[1] != 0.0f && moveMode != 2 &&
            (copies || shiftOnSubrects))
        {
            motionShift[0] = (float) dx / scale[0];
            motionShift[1] = (float) dy / scale[1];
            if (moveMode == 1)
            {
                motionShift[0] = -motionShift[0];
                motionShift[1] = -motionShift[1];
            }
            shiftWanted = true;
            ++g_followShifts;
            if (!g_loggedShift)
            {
                g_loggedShift = true;
                LOG_INFO("Per-eye DLSS DX12: FIRST crop move carried over: crop {} moved ({}, {}) render px, vectors "
                         "shifted by ({:.6f}, {:.6f}) (MV scale {} {}); no history reset unless the copy fails",
                         crop, dx, dy, motionShift[0], motionShift[1], scale[0], scale[1]);
            }
        }
        else if (pos != g_cropPos.end())
        {
            if (follow)
            {
                ++g_followResets;
                if (nearMove && moveMode != 2 && !copies && !shiftOnSubrects)
                    ++g_followResetsNoCopy;
            }
            g_cropReset[crop] = true;
            if (!g_loggedFoveaMove)
            {
                g_loggedFoveaMove = true;
                LOG_INFO("Per-eye DLSS DX12: foveal crop {} moved to render ({}, {}) / output ({}, {}); its history "
                         "restarts at each move",
                         crop, r.rx, r.ry, r.ox, r.oy);
            }
        }
        g_cropPos[crop] = { r.rx, r.ry };
    }
    // For the age pass (EndFovea): by how much the box moved in OUTPUT pixels this evaluate (0, 0 when it stayed).
    {
        const auto op = g_cropOutPos.find(crop);
        g_ageShiftOut[0] = op != g_cropOutPos.end() ? (int) r.ox - (int) op->second.first : 0;
        g_ageShiftOut[1] = op != g_cropOutPos.end() ? (int) r.oy - (int) op->second.second : 0;
        g_cropOutPos[crop] = { r.ox, r.oy };
    }
    ID3D12Resource* output = GetResource(params, NVSDK_NGX_Parameter_Output);
    if (output == nullptr)
        return false;
    if (!EnsureCropTex(g_device, r.ow, r.oh, TypedFormat(output->GetDesc().Format), g_present.load()))
    {
        ++g_foveaFails;
        LogOnce(g_loggedFoveaFail, "foveal crop skipped: no crop output texture");
        return false;
    }
    auto& s = g_saved;
    s.colorX = GetUIntParam(params, NVSDK_NGX_Parameter_DLSS_Input_Color_Subrect_Base_X);
    s.colorY = GetUIntParam(params, NVSDK_NGX_Parameter_DLSS_Input_Color_Subrect_Base_Y);
    s.mvX = GetUIntParam(params, NVSDK_NGX_Parameter_DLSS_Input_MV_SubrectBase_X);
    s.mvY = GetUIntParam(params, NVSDK_NGX_Parameter_DLSS_Input_MV_SubrectBase_Y);
    s.depthX = GetUIntParam(params, NVSDK_NGX_Parameter_DLSS_Input_Depth_Subrect_Base_X);
    s.depthY = GetUIntParam(params, NVSDK_NGX_Parameter_DLSS_Input_Depth_Subrect_Base_Y);
    s.subW = GetUIntParam(params, NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Width);
    s.subH = GetUIntParam(params, NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Height);
    s.width = GetUIntParam(params, NVSDK_NGX_Parameter_Width);
    s.height = GetUIntParam(params, NVSDK_NGX_Parameter_Height);
    s.outW = GetUIntParam(params, NVSDK_NGX_Parameter_OutWidth);
    s.outH = GetUIntParam(params, NVSDK_NGX_Parameter_OutHeight);
    s.outX = GetUIntParam(params, NVSDK_NGX_Parameter_DLSS_Output_Subrect_Base_X);
    s.outY = GetUIntParam(params, NVSDK_NGX_Parameter_DLSS_Output_Subrect_Base_Y);
    s.output = output;
    s.reset = 0;
    params->Get(NVSDK_NGX_Parameter_Reset, &s.reset);
    s.rr = rayReconstruction;
    g_cropIsRr = rayReconstruction;
    // RR reads each guide through its own subrect base: an RR crop always reads copies, of the guides as well. Which
    // guides are set, and whether the box fits each, is settled before anything is recorded or changed.
    if (rayReconstruction)
    {
        const uint32_t colourBase[2] = { s.colorX, s.colorY };
        if (!PlanGuides(params, colourBase, r.rx, r.ry, r.rw, r.rh))
        {
            ++g_foveaSkips;
            return false;
        }
    }

    // The crop reads the same inputs at its box (the same colour, depth and -- per eye -- our same-eye motion), with
    // the same jitter and MV scale, and writes our texture from its origin. By default from our own copies of the box
    // (GATHER); otherwise through DLSS's input subrects.
    s.gathered = false;
    s.shifted = false;
    s.sub = 0;
    s.colour = GetResource(params, NVSDK_NGX_Parameter_Color);
    s.motion = GetResource(params, NVSDK_NGX_Parameter_MotionVectors);
    s.depth = GetResource(params, NVSDK_NGX_Parameter_Depth);
    if (g_composedForShared && g_composed != nullptr && !g_routing)
    {
        // Shared routing, our same-eye motion made for NR: the crop (a history per eye) takes it too; these parameters
        // are the crop's own copy, the game's feature keeps its vectors.
        s.motion = g_composed;
        params->Set(NVSDK_NGX_Parameter_MotionVectors, g_composed);
        static bool logged = false;
        if (!logged)
        {
            logged = true;
            LOG_INFO("Per-eye DLSS DX12: FIRST foveal crop on our same-eye motion in shared routing (DLSS keeps the game's)");
        }
    }
    // The crop's vectors from below: what the full-size pass of this same evaluate left in the block (same-eye
    // vectors from a per-eye DLSS pass below us), in the place of ours. Copied at the box like any other source; for RR
    // hand the DEPTH through a subrect with it, or the copy is replaced below us again (a marked SR crop
    // should be left alone). RR: UpscalerFoveaRrVectors; SR:
    // UpscalerFoveaSrVectors -- both read only a pass of THIS evaluate (g_mainRan), never an earlier frame's texture.
    // Wanted from below: RR on its choice (UpscalerFoveaRrVectors 1), SR on SrCropFromBelow (auto while DL(S)SS is seen).
    // Only a change of what is WANTED restarts the crops' histories; a frame with nothing from below (about one a
    // minute: the full-size pass 'left as handed') keeps the block's vectors as they are for that frame
    // (ours if the fit ran, else the game's) and restarts nothing -- before, each such frame restarted both histories
    // twice.
    const bool wantBelow = rayReconstruction ? Config::Instance()->UpscalerFoveaRrVectors.value_or_default() == 1
                                             : SrCropFromBelow();
    const bool mainRan = g_mainRan;
    ID3D12Resource* const mainBelow = mainRan ? g_mainMvBelow : nullptr;
    g_mainRan = false;
    g_mainMvBelow = nullptr;
    bool below = false;
    if (wantBelow)
    {
        const char* why = nullptr;
        if (!mainRan)
            why = "no DLSS full-size pass ran before the crop (the own TAA periphery? [Inputs] "
                  "UpscalerFoveaPeripheryTaa = false gives the pass below us a DLSS periphery)";
        else if (mainBelow == nullptr)
            why = "the full-size pass left the block's vectors as handed";
        else
        {
            const D3D12_RESOURCE_DESC d = mainBelow->GetDesc();
            if (d.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || s.mvX + r.rx + r.rw > d.Width ||
                s.mvY + r.ry + r.rh > d.Height)
                why = "the crop's box does not fit the texture left in the block";
        }
        if (why == nullptr)
        {
            s.motion = mainBelow;
            params->Set(NVSDK_NGX_Parameter_MotionVectors, mainBelow);
            below = true;
        }
        else
        {
            ++g_cropVecBelowMissing;
            LogOnce(g_loggedCropVecMissing,
                    std::format("{} crop: vectors from below wanted, but {}: the block's vectors as they are for that "
                                "frame ({}), no history restart (counted per window)",
                                rayReconstruction ? "RR" : "SR", why,
                                g_composedForShared ? "ours" : "the game's: our fit did not run"));
        }
    }
    ++(below ? g_cropVecBelow : g_cropVecOurs);
    if (g_cropVecSourceLast != (wantBelow ? 1 : 0))
    {
        LOG_INFO("Per-eye DLSS DX12: the {} crop's vectors are now {}: {}{}", rayReconstruction ? "RR" : "SR",
                 wantBelow ? "FROM BELOW (what the full-size pass left in the block: DL(S)SS's)" : "OURS",
                 DescribeTexture(s.motion), g_cropVecSourceLast >= 0 ? "; the crops' histories start over" : "");
        if (g_cropVecSourceLast >= 0)
            for (auto& entry : g_cropReset)
                entry.second = true;
        g_cropVecSourceLast = wantBelow ? 1 : 0;
    }
    const bool readsCopies = CropReadsCopies(rayReconstruction);
    const int subMask = CropSubrectMask(rayReconstruction);
    // 0 = all through subrects, 1 = copies.
    const int pathCode = readsCopies ? 1 : 0;
    if (g_cropCopiesLast != pathCode)
    {
        // Another input path than the last evaluate's: the crops' histories start over (as the overlay's switch does).
        if (g_cropCopiesLast >= 0)
        {
            for (auto& entry : g_cropReset)
                entry.second = true; // set on each crop's block at its next evaluate (below, for this one)
        }
        const auto& key = Config::Instance()->UpscalerFoveaCopyInputs;
        LOG_INFO("Per-eye DLSS DX12: the crop reads {} ({}){}",
                 readsCopies ? "COPIES of its inputs" : "the game's full-size inputs through DLSS's subrects",
                 rayReconstruction      ? "ray reconstruction always copies"
                 : key.has_value()      ? "[Inputs] UpscalerFoveaCopyInputs set by hand"
                 : g_copiesSwappedBelow ? (readsCopies ? "automatic: it follows the gaze, and a move is carried over in "
                                                         "copied vectors -- which something below us REPLACES: expect "
                                                         "shimmer in the crop"
                                                       : "automatic: something below us changes "
                                                         "copied vectors; read through subrects, a move carried over in "
                                                         "our own shifted vectors")
                                        : "automatic: the default",
                 g_cropCopiesLast >= 0 ? "; the crops' histories start over" : "");
        g_cropCopiesLast = pathCode;
    }
    if (readsCopies)
    {
        const uint32_t colourBase[2] = { s.colorX + r.rx, s.colorY + r.ry };
        const uint32_t motionBase[2] = { s.mvX + r.rx, s.mvY + r.ry };
        const uint32_t depthBase[2] = { s.depthX + r.rx, s.depthY + r.ry };
        // Shift by MV offset: the copied vectors stay unshifted, the shift is set on the block below.
        const bool byOffset = Config::Instance()->UpscalerFoveaShiftByOffset.value_or_default();
        const float gatherShift[2] = { byOffset ? 0.0f : motionShift[0], byOffset ? 0.0f : motionShift[1] };
        s.gathered = GatherCropInputs(cmdList, s.colour, s.motion, s.depth, colourBase, motionBase, depthBase, r.rw,
                                      r.rh, gatherShift);
    }
    if ((subMask & 4) != 0 && shiftWanted)
    {
        // Subrects: the box of our composed vectors, shifted by the move, into g_gShifted; the crop reads it through
        // the same subrect base as it read g_composed.
        const uint32_t motionBase[2] = { s.mvX + r.rx, s.mvY + r.ry };
        if (ShiftComposed(cmdList, s.motion, motionBase, r.rw, r.rh, motionShift))
        {
            params->Set(NVSDK_NGX_Parameter_MotionVectors, g_gShifted.res);
            s.shifted = true;
            if (!g_loggedShiftSubrects)
            {
                g_loggedShiftSubrects = true;
                LOG_INFO("Per-eye DLSS DX12: FIRST crop move carried over THROUGH SUBRECTS: the box ({}, {}) {}x{} of our "
                         "composed vectors shifted by ({:.3f}, {:.3f}) into a full-size texture of ours",
                         motionBase[0], motionBase[1], r.rw, r.rh, motionShift[0], motionShift[1]);
            }
        }
    }
    if (shiftWanted && ((subMask & 4) != 0 ? !s.shifted : !s.gathered))
    {
        g_cropReset[crop] = true; // the copy (or the shift) failed at this move, so the shift reached no vectors
        ++g_followResets;
        ++g_followResetsCopyFailed;
        --g_followShifts; // not carried over after all
    }
    // RR's guides are copied either way; its colour, depth and vectors only when it reads copies.
    if (rayReconstruction &&
        ((readsCopies && !s.gathered) || !GatherGuides(cmdList, params, r.rx, r.ry, r.rw, r.rh)))
    {
        ++g_foveaSkips;
        LogOnce(g_loggedGuideSkip, "RR crop skipped: its inputs could not be copied (counted as skips)");
        s.gathered = false;
        return false; // nothing handed over yet (GatherGuides sets its parameters only once all is recorded)
    }
    // An SR crop: the game's bias mask (if any) is the one guide left; see UpscalerFoveaCropMask above.
    s.guides = false;
    s.maskDropped = false;
    s.mask = nullptr;
    if (!rayReconstruction)
    {
        ID3D12Resource* mask = GetResource(params, NVSDK_NGX_Parameter_DLSS_Input_Bias_Current_Color_Mask);
        const int maskMode = std::clamp(Config::Instance()->UpscalerFoveaCropMask.value_or_default(), 0, 2);
        if (mask != nullptr)
        {
            if (!g_loggedCropMask)
            {
                g_loggedCropMask = true;
                const D3D12_RESOURCE_DESC md = mask->GetDesc();
                LOG_INFO("Per-eye DLSS DX12: the game hands DLSS a current-colour bias mask {}x{} (fmt {}); the crop takes "
                         "it: {}",
                         md.Width, md.Height, (int) md.Format,
                         maskMode == 0   ? "NONE (dropped for the crop's evaluates)"
                         : maskMode == 1 ? "COPIED at the crop's box"
                                         : "AS HANDED (read at (0, 0) of the full frame: the wrong place; a test)");
            }
            if (maskMode == 0)
            {
                s.mask = mask;
                s.maskDropped = true;
                params->Set(NVSDK_NGX_Parameter_DLSS_Input_Bias_Current_Color_Mask, (ID3D12Resource*) nullptr);
                ++g_maskDropped;
            }
            else if (maskMode == 1)
            {
                const uint32_t colourBase[2] = { s.colorX, s.colorY };
                if (PlanGuides(params, colourBase, r.rx, r.ry, r.rw, r.rh) &&
                    GatherGuides(cmdList, params, r.rx, r.ry, r.rw, r.rh))
                {
                    s.guides = true;
                    ++g_maskCopied;
                }
                else
                {
                    RestoreGuides(params); // a plan without its copy: everything back as handed
                    LogOnce(g_loggedMaskSkip, "crop: the game's bias mask could not be copied for the box (the reason is "
                                              "the 'RR crop skipped' line above); handed as is");
                    ++g_maskAsIs;
                }
            }
            else
                ++g_maskAsIs;
        }
    }
    if (rayReconstruction)
    {
        void* proj = nullptr;
        if (params->Get("ViewToClipMatrix", &proj) == NVSDK_NGX_Result_Success && proj != nullptr)
        {
            if (CropProjection((const float*) proj, r, g_cropProj))
            {
                g_savedProj = proj;
                params->Set("ViewToClipMatrix", (void*) g_cropProj);
                g_projSwapped = true;
                if (!g_loggedProj)
                {
                    g_loggedProj = true;
                    const float* m = (const float*) proj;
                    LOG_INFO("Per-eye DLSS DX12: RR crop projection: eye ({:.4f} {:.4f} | {:.4f} {:.4f}) -> crop ({:.4f} "
                             "{:.4f} | {:.4f} {:.4f}) [x/y scale | offset terms]",
                             m[0], m[5], m[8], m[9], g_cropProj[0], g_cropProj[5], g_cropProj[8], g_cropProj[9]);
                }
            }
            else
                LogOnce(g_loggedProj, "RR crop: ViewToClipMatrix in an unrecognised layout; the eye's projection is "
                                      "kept (reflections in the crop may be off)");
        }
    }
    // Input by input: our copy of the box at base 0, or the full-size texture at the box (all three alike).
    s.sub = s.gathered ? subMask : 7;
    if ((s.sub & 1) == 0)
    {
        params->Set(NVSDK_NGX_Parameter_Color, g_gColour.res);
        params->Set(NVSDK_NGX_Parameter_DLSS_Input_Color_Subrect_Base_X, 0u);
        params->Set(NVSDK_NGX_Parameter_DLSS_Input_Color_Subrect_Base_Y, 0u);
    }
    else
    {
        params->Set(NVSDK_NGX_Parameter_DLSS_Input_Color_Subrect_Base_X, s.colorX + r.rx);
        params->Set(NVSDK_NGX_Parameter_DLSS_Input_Color_Subrect_Base_Y, s.colorY + r.ry);
    }
    if ((s.sub & 4) == 0)
    {
        params->Set(NVSDK_NGX_Parameter_MotionVectors, g_gMotion.res);
        params->Set(NVSDK_NGX_Parameter_DLSS_Input_MV_SubrectBase_X, 0u);
        params->Set(NVSDK_NGX_Parameter_DLSS_Input_MV_SubrectBase_Y, 0u);
    }
    else
    {
        params->Set(NVSDK_NGX_Parameter_DLSS_Input_MV_SubrectBase_X, s.mvX + r.rx);
        params->Set(NVSDK_NGX_Parameter_DLSS_Input_MV_SubrectBase_Y, s.mvY + r.ry);
    }
    if ((s.sub & 2) == 0)
    {
        params->Set(NVSDK_NGX_Parameter_Depth, g_gDepth.res);
        params->Set(NVSDK_NGX_Parameter_DLSS_Input_Depth_Subrect_Base_X, 0u);
        params->Set(NVSDK_NGX_Parameter_DLSS_Input_Depth_Subrect_Base_Y, 0u);
    }
    else
    {
        params->Set(NVSDK_NGX_Parameter_DLSS_Input_Depth_Subrect_Base_X, s.depthX + r.rx);
        params->Set(NVSDK_NGX_Parameter_DLSS_Input_Depth_Subrect_Base_Y, s.depthY + r.ry);
    }
    params->Set(NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Width, r.rw);
    params->Set(NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Height, r.rh);
    // Creation sizes too: a live recreate of the crop (the overlay's preset Apply marks every feature) must make it at
    // the crop's size, with the crop's preset.
    params->Set(NVSDK_NGX_Parameter_Width, r.rw);
    params->Set(NVSDK_NGX_Parameter_Height, r.rh);
    params->Set(NVSDK_NGX_Parameter_OutWidth, r.ow);
    params->Set(NVSDK_NGX_Parameter_OutHeight, r.oh);
    params->Set(NVSDK_NGX_Parameter_DLSS_Output_Subrect_Base_X, 0u);
    params->Set(NVSDK_NGX_Parameter_DLSS_Output_Subrect_Base_Y, 0u);
    params->Set(NVSDK_NGX_Parameter_Output, g_cropTex);
    // The crop's shift through MV.Offset (see g_offsetShifts): the block's own offset is kept under it and put back.
    s.mvOffX = s.mvOffY = 0.0f;
    params->Get(NVSDK_NGX_Parameter_MV_Offset_X, &s.mvOffX);
    params->Get(NVSDK_NGX_Parameter_MV_Offset_Y, &s.mvOffY);
    s.offHandedX = s.mvOffX;
    s.offHandedY = s.mvOffY;
    if (shiftWanted && Config::Instance()->UpscalerFoveaShiftByOffset.value_or_default())
    {
        s.offHandedX = s.mvOffX + motionShift[0];
        s.offHandedY = s.mvOffY + motionShift[1];
        params->Set(NVSDK_NGX_Parameter_MV_Offset_X, s.offHandedX);
        params->Set(NVSDK_NGX_Parameter_MV_Offset_Y, s.offHandedY);
        ++g_offsetShifts;
        if (!g_loggedOffset)
        {
            g_loggedOffset = true;
            LOG_INFO("Per-eye DLSS DX12: FIRST crop shift handed through MV.Offset: ({:.3f}, {:.3f}) on top of the block's "
                     "({:.3f}, {:.3f}); the copied vectors carry no shift in this mode",
                     motionShift[0], motionShift[1], s.mvOffX, s.mvOffY);
        }
    }
    g_ageResetNow = s.reset != 0 || g_cropReset[crop]; // the history restarts: every crop pixel is new (the age pass)
    if (g_cropReset[crop])
    {
        params->Set(NVSDK_NGX_Parameter_Reset, 1);
        g_cropReset[crop] = false;
    }
    g_presetForCreate = Config::Instance()->UpscalerFoveaPreset.value_or_default();
    g_kindForCreate = 1;
    // DLSS writes our texture as a UAV.
    if (g_cropTexState != D3D12_RESOURCE_STATE_UNORDERED_ACCESS)
    {
        Transition(cmdList, g_cropTex, g_cropTexState, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        g_cropTexState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    }
    g_foveaCropNow = crop;
    return true;
}

void EndFovea(ID3D12GraphicsCommandList* cmdList, NVSDK_NGX_Parameter* params, bool evaluated, bool rebuilding)
{
    std::lock_guard lock(g_mutex);
    const auto& s = g_saved;
    if (evaluated && g_foveaCropNow != 0)
    {
        // PROBE: what the crop's block holds now, against what BeginFovea handed over (see g_cropAfter* above).
        const bool mvCopied = (s.sub & 4) == 0;
        ID3D12Resource* mvHanded = mvCopied ? g_gMotion.res : s.shifted ? g_gShifted.res : s.motion;
        ID3D12Resource* colHanded = (s.sub & 1) == 0 ? g_gColour.res : s.colour;
        ID3D12Resource* depHanded = (s.sub & 2) == 0 ? g_gDepth.res : s.depth;
        ID3D12Resource* mvNow = GetResource(params, NVSDK_NGX_Parameter_MotionVectors);
        ID3D12Resource* colNow = GetResource(params, NVSDK_NGX_Parameter_Color);
        ID3D12Resource* depNow = GetResource(params, NVSDK_NGX_Parameter_Depth);
        const unsigned int baseNowX = GetUIntParam(params, NVSDK_NGX_Parameter_DLSS_Input_MV_SubrectBase_X);
        const unsigned int baseNowY = GetUIntParam(params, NVSDK_NGX_Parameter_DLSS_Input_MV_SubrectBase_Y);
        const unsigned int baseHandedX = mvCopied ? 0u : s.mvX + g_foveaPlan.rx;
        const unsigned int baseHandedY = mvCopied ? 0u : s.mvY + g_foveaPlan.ry;
        ++g_cropAfterChecked;
        const bool mvSwapped = mvNow != mvHanded, colSwapped = colNow != colHanded, depSwapped = depNow != depHanded;
        const bool baseChanged = baseNowX != baseHandedX || baseNowY != baseHandedY;
        if (mvSwapped) ++g_cropAfterMvSwapped;
        if (mvSwapped && mvCopied && !g_copiesSwappedBelow)
        {
            g_copiesSwappedBelow = true;
            LOG_INFO("Per-eye DLSS DX12: something below us REPLACED the vectors of a crop evaluate that read copies "
                     "from here a crop that stands still reads through DLSS's subrects unless "
                     "[Inputs] UpscalerFoveaCopyInputs is set by hand (FoveaCopyRule.h)");
        }
        if (colSwapped) ++g_cropAfterColourSwapped;
        if (depSwapped) ++g_cropAfterDepthSwapped;
        if (baseChanged) ++g_cropAfterBaseChanged;
        float offNow[2] = { 0.0f, 0.0f };
        params->Get(NVSDK_NGX_Parameter_MV_Offset_X, &offNow[0]);
        params->Get(NVSDK_NGX_Parameter_MV_Offset_Y, &offNow[1]);
        if (offNow[0] != s.offHandedX || offNow[1] != s.offHandedY)
            ++g_cropAfterOffsetChanged;
        if (!g_loggedHook)
        {
            // Where does the NGX evaluate entry we call resolve? (Which module, and is its first instruction a jump:
            // an inline hook by another mod.) Read-only.
            g_loggedHook = true;
            const void* fn = (const void*) NVNGXProxy::D3D12_EvaluateFeature();
            auto moduleOf = [](const void* p) -> std::string
            {
                HMODULE m = nullptr;
                if (p == nullptr ||
                    !GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                                        (LPCSTR) p, &m))
                    return "(no module)";
                char name[MAX_PATH] = {};
                GetModuleFileNameA(m, name, MAX_PATH);
                const char* slash = strrchr(name, '\\');
                return slash ? slash + 1 : name;
            };
            std::string bytes;
            std::string target;
            if (fn != nullptr)
            {
                const unsigned char* b = (const unsigned char*) fn;
                for (int i = 0; i < 8; ++i)
                    bytes += std::format("{:02X} ", b[i]);
                if (b[0] == 0xE9)
                {
                    int32_t rel = 0;
                    std::memcpy(&rel, b + 1, 4);
                    const void* to = (const void*) (b + 5 + rel);
                    target = std::format(" -> JMP to {:p} in {}", to, moduleOf(to));
                }
                else if (b[0] == 0xFF && b[1] == 0x25)
                    target = " -> an indirect JMP (an import stub)";
            }
            LOG_INFO("Per-eye DLSS DX12: PROBE: the NGX D3D12 evaluate entry the crop calls is {:p} in {}; first bytes {}{}",
                     fn, moduleOf(fn), bytes, target);
            // Diagnostic: logs which module the NGX evaluate entry resolves into. Read-only; a module not loaded says so.
            for (const char* mod : { "nvngx_dlssd.dll", "nvngx_dlss.dll", "_nvngx.dll", "nvngx.dll" })
            {
                HMODULE h = GetModuleHandleA(mod);
                if (h == nullptr)
                {
                    LOG_INFO("Per-eye DLSS DX12: PROBE: {} is not loaded in this process", mod);
                    continue;
                }
                for (const char* exp : { "NVSDK_NGX_D3D12_EvaluateFeature", "NVSDK_NGX_D3D12_CreateFeature" })
                {
                    const unsigned char* b = (const unsigned char*) GetProcAddress(h, exp);
                    if (b == nullptr)
                    {
                        LOG_INFO("Per-eye DLSS DX12: PROBE: {} exports no {}", mod, exp);
                        continue;
                    }
                    std::string eb, et;
                    for (int i = 0; i < 8; ++i)
                        eb += std::format("{:02X} ", b[i]);
                    if (b[0] == 0xE9)
                    {
                        int32_t rel = 0;
                        std::memcpy(&rel, b + 1, 4);
                        const void* to = (const void*) (b + 5 + rel);
                        et = std::format(" -> JMP to {:p} in {} (intercepted)", to, moduleOf(to));
                    }
                    else if (b[0] == 0xFF && b[1] == 0x25)
                        et = " -> an indirect JMP (an import stub)";
                    else
                        et = " (no jump at the entry: not intercepted inline)";
                    LOG_INFO("Per-eye DLSS DX12: PROBE: {}'s own {} is {:p}; first bytes {}{}", mod, exp, (const void*) b,
                             eb, et);
                }
            }
        }
        if ((mvSwapped || colSwapped || depSwapped || baseChanged) && !g_loggedCropAfter)
        {
            g_loggedCropAfter = true;
            auto describe = [](ID3D12Resource* r) -> std::string
            {
                if (r == nullptr)
                    return "null";
                const D3D12_RESOURCE_DESC d = r->GetDesc();
                return std::format("{:p} {}x{} fmt {}", (void*) r, d.Width, d.Height, (int) d.Format);
            };
            float scale[2] = { 0.0f, 0.0f };
            params->Get(NVSDK_NGX_Parameter_MV_Scale_X, &scale[0]);
            params->Get(NVSDK_NGX_Parameter_MV_Scale_Y, &scale[1]);
            LOG_WARN("Per-eye DLSS DX12: PROBE: the crop's inputs CHANGED during its evaluate (something below us rewrote the "
                     "block -- something below us?): vectors handed {} -> now {}; colour handed {} -> now {}; depth handed "
                     "{} -> now {}; MV subrect base handed ({}, {}) -> now ({}, {}); MV scale now ({}, {}); crop {} ({})",
                     describe(mvHanded), describe(mvNow), describe(colHanded), describe(colNow), describe(depHanded),
                     describe(depNow), baseHandedX, baseHandedY, baseNowX, baseNowY, scale[0], scale[1], g_foveaCropNow,
                     s.sub == 0 ? "inputs copied" : s.sub == 7 ? "subrects" : "copies and subrects mixed");
        }
    }
    params->Set(NVSDK_NGX_Parameter_DLSS_Input_Color_Subrect_Base_X, s.colorX);
    params->Set(NVSDK_NGX_Parameter_DLSS_Input_Color_Subrect_Base_Y, s.colorY);
    params->Set(NVSDK_NGX_Parameter_DLSS_Input_MV_SubrectBase_X, s.mvX);
    params->Set(NVSDK_NGX_Parameter_DLSS_Input_MV_SubrectBase_Y, s.mvY);
    params->Set(NVSDK_NGX_Parameter_DLSS_Input_Depth_Subrect_Base_X, s.depthX);
    params->Set(NVSDK_NGX_Parameter_DLSS_Input_Depth_Subrect_Base_Y, s.depthY);
    params->Set(NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Width, s.subW);
    params->Set(NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Height, s.subH);
    params->Set(NVSDK_NGX_Parameter_Width, s.width);
    params->Set(NVSDK_NGX_Parameter_Height, s.height);
    params->Set(NVSDK_NGX_Parameter_OutWidth, s.outW);
    params->Set(NVSDK_NGX_Parameter_OutHeight, s.outH);
    params->Set(NVSDK_NGX_Parameter_DLSS_Output_Subrect_Base_X, s.outX);
    params->Set(NVSDK_NGX_Parameter_DLSS_Output_Subrect_Base_Y, s.outY);
    params->Set(NVSDK_NGX_Parameter_Output, s.output);
    params->Set(NVSDK_NGX_Parameter_Reset, s.reset);
    if (s.gathered)
    {
        params->Set(NVSDK_NGX_Parameter_Color, s.colour);
        params->Set(NVSDK_NGX_Parameter_MotionVectors, s.motion);
        params->Set(NVSDK_NGX_Parameter_Depth, s.depth);
        g_saved.gathered = false;
    }
    if (s.rr)
    {
        RestoreGuides(params);
        if (g_projSwapped)
            params->Set("ViewToClipMatrix", g_savedProj);
        g_projSwapped = false;
        g_saved.rr = false;
    }
    if (s.guides)
    {
        RestoreGuides(params);
        g_saved.guides = false;
    }
    if (g_foveaCropNow != 0)
    {
        params->Set(NVSDK_NGX_Parameter_MV_Offset_X, s.mvOffX);
        params->Set(NVSDK_NGX_Parameter_MV_Offset_Y, s.mvOffY);
    }
    if (s.maskDropped)
    {
        params->Set(NVSDK_NGX_Parameter_DLSS_Input_Bias_Current_Color_Mask, s.mask);
        g_saved.maskDropped = false;
        g_saved.mask = nullptr;
    }
    g_presetForCreate = 0;
    g_kindForCreate = 0;
    if (rebuilding)
    {
        // The staged rebuild returns without evaluating until its last step: nothing new in our texture to draw.
        ++g_foveaRebuildFrames;
        return;
    }
    if (!evaluated)
    {
        ++g_foveaFails;
        LogOnce(g_loggedFoveaFail, "foveal crop: the crop feature's evaluate failed (counted per window)");
        return;
    }
    const FoveaRect r = g_foveaPlan;

    // COMPOSITE: our crop over the crop box of the game's output. The output is a UAV at an upscaler's evaluate (the
    // DLSS contract; the full pass just wrote it on this list), so a UAV barrier orders our read-modify-write after
    // that write. Our texture: UAV (DLSS wrote it) -> shader resource.
    D3D12_RESOURCE_BARRIER uav {};
    uav.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    uav.UAV.pResource = s.output;
    cmdList->ResourceBarrier(1, &uav);
    Transition(cmdList, g_cropTex, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    g_cropTexState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

    // The crop's ages for this evaluate (the fade-in over the periphery where its picture is new), when wanted.
    const int ageFrames = std::clamp(Config::Instance()->UpscalerFoveaAgeFrames.value_or_default(), 0, 255);
    GatherTex* age = ageFrames > 0 ? UpdateAge(cmdList, g_foveaCropNow, r.ow, r.oh, g_ageShiftOut, g_ageResetNow)
                                   : nullptr;

    const uint32_t slot = (uint32_t) (g_compositeSeq++ % kRing);
    D3D12_CPU_DESCRIPTOR_HANDLE cpu = g_compositeHeap->GetCPUDescriptorHandleForHeapStart();
    cpu.ptr += (SIZE_T) slot * 3 * g_descriptorSize;
    D3D12_GPU_DESCRIPTOR_HANDLE gpu = g_compositeHeap->GetGPUDescriptorHandleForHeapStart();
    gpu.ptr += (UINT64) slot * 3 * g_descriptorSize;
    D3D12_SHADER_RESOURCE_VIEW_DESC srv {};
    srv.Format = g_cropTexFormat;
    srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srv.Texture2D.MipLevels = 1;
    g_device->CreateShaderResourceView(g_cropTex, &srv, cpu);
    cpu.ptr += g_descriptorSize;
    if (age != nullptr)
    {
        srv.Format = DXGI_FORMAT_R8_UINT;
        g_device->CreateShaderResourceView(age->res, &srv, cpu);
    }
    else
        g_device->CreateShaderResourceView(g_cropTex, &srv, cpu); // a valid descriptor; never read (AgeFrames 0)
    cpu.ptr += g_descriptorSize;
    D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc {};
    uavDesc.Format = TypedFormat(s.output->GetDesc().Format);
    uavDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    g_device->CreateUnorderedAccessView(s.output, nullptr, &uavDesc, cpu);

    const int feather = std::max(0, Config::Instance()->UpscalerFoveaFeather.value_or_default());
    const uint32_t constants[8] = { r.ox,  r.oy, r.ow, r.oh, (uint32_t) feather, g_foveaOutline ? 1u : 0u,
                                    age != nullptr ? (uint32_t) ageFrames : 0u, 0 };
    ID3D12DescriptorHeap* heaps[] = { g_compositeHeap };
    cmdList->SetDescriptorHeaps(1, heaps);
    cmdList->SetComputeRootSignature(g_compositeRoot);
    cmdList->SetPipelineState(g_compositePso);
    cmdList->SetComputeRoot32BitConstants(0, 8, constants, 0);
    cmdList->SetComputeRootDescriptorTable(1, gpu);
    cmdList->Dispatch((r.ow + 7) / 8, (r.oh + 7) / 8, 1);
    if (r.target && g_gazeMarkPso != nullptr)
    {
        constexpr uint32_t side = 64;
        const uint32_t bx = (uint32_t) std::clamp((long) r.targetX - (long) side / 2, 0L, (long) r.outW - (long) side);
        const uint32_t by = (uint32_t) std::clamp((long) r.targetY - (long) side / 2, 0L, (long) r.outH - (long) side);
        cmdList->ResourceBarrier(1, &uav);
        const uint32_t mark[8] = { bx, by, side, side, 6, 1, r.targetX, r.targetY };
        cmdList->SetPipelineState(g_gazeMarkPso);
        cmdList->SetComputeRoot32BitConstants(0, 8, mark, 0);
        cmdList->Dispatch((side + 7) / 8, (side + 7) / 8, 1);
    }
    const bool gazeBox = g_gazeBoxWhenStill ||
                         (VarjoGaze::Enabled() && Config::Instance()->UpscalerFoveaGazeFollow.value_or_default());
    if (g_foveaOutline && r.gaze && gazeBox && g_gazeMarkPso != nullptr)
    {
        // The crop-sized box centred on the gaze, kept inside the eye; same table (its UAV is the output).
        const uint32_t bx = (uint32_t) std::clamp((long) r.gazeX - (long) r.ow / 2, 0L, (long) r.outW - (long) r.ow);
        const uint32_t by = (uint32_t) std::clamp((long) r.gazeY - (long) r.oh / 2, 0L, (long) r.outH - (long) r.oh);
        cmdList->ResourceBarrier(1, &uav); // after the composite's writes to the same output
        const uint32_t mark[8] = { bx, by, r.ow, r.oh, 4, 0, r.gazeX, r.gazeY };
        cmdList->SetPipelineState(g_gazeMarkPso);
        cmdList->SetComputeRoot32BitConstants(0, 8, mark, 0);
        cmdList->Dispatch((r.ow + 7) / 8, (r.oh + 7) / 8, 1);
    }
    ++g_foveaEvals;
    if (!g_loggedFoveaFirst)
    {
        g_loggedFoveaFirst = true;
        LOG_INFO("Per-eye DLSS DX12: FIRST foveal crop drawn (crop feature {}, box ({}, {}) {}x{}, feather {} px)",
                 g_foveaCropNow, r.ox, r.oy, r.ow, r.oh, feather);
    }
}

void RequestFoveaPreset(int preset)
{
    std::lock_guard lock(g_mutex);
    Config::Instance()->UpscalerFoveaPreset = preset;
    // The crop features are rebuilt by OptiScaler's staged rebuild (the overlay's Apply path) at their next evaluates,
    // with the crop's sizes and this preset (BeginFovea sets both around every crop evaluate). Only the crops.
    for (auto& entry : g_cropRects)
        State::Instance().changeBackend[entry.first] = true;
    LOG_INFO("Per-eye DLSS DX12: foveal crop preset -> {} (overlay); {} crop feature(s) rebuild live", preset,
             g_cropRects.size());
}

void RequestFoveaDirect(bool rr, bool on)
{
    std::lock_guard lock(g_mutex);
    if (rr)
        Config::Instance()->UpscalerFoveaRrDirect = on;
    else
        Config::Instance()->UpscalerFoveaSrDirect = on;
    // The same staged rebuild as RequestFoveaPreset: the new feature is made inside the crop's evaluate, where BeginFovea
    // has set the kind (1) the direct path reads.
    for (auto& entry : g_cropRects)
        State::Instance().changeBackend[entry.first] = true;
    LOG_INFO("Per-eye DLSS DX12: {} crop direct on the snippet DLL -> {} (overlay); {} crop feature(s) rebuild live",
             rr ? "RR" : "SR", on ? "ON" : "off", g_cropRects.size());
}

void RequestPeripheryOneFeature(int mode)
{
    std::lock_guard lock(g_mutex);
    const bool before = PeripheryOneFeature();
    Config::Instance()->UpscalerPeripheryOneFeature = mode;
    // The feature now used for an eye had the other eye's history (or none): every periphery restarts. A per-eye
    // feature left unused stays until its owner is forgotten (crop off, the game's feature released).
    if (PeripheryOneFeature() != before)
        for (auto& entry : g_periphReset)
            entry.second = true;
    LOG_INFO("Per-eye DLSS DX12: periphery features -> {} (overlay); now {}",
             mode == 1 ? "one for both eyes" : mode == 2 ? "one per eye" : "auto", PeripheryOneFeature() ? "one" : "per eye");
}

bool PeripheryOneFeatureNow()
{
    // The menu's thread: the routing flag and the auto state are read without the lock (a label only).
    const int mode = Config::Instance()->UpscalerPeripheryOneFeature.value_or_default();
    return !g_routing && (mode == 1 || (mode == 0 && g_belowSeen.load()));
}

uint64_t VramUsedMB(ID3D12Device* device)
{
    // The game's adapter, found once by the device's LUID (our own factory: Release'd here; the adapter is ours and kept).
    static std::mutex m;
    static IDXGIAdapter3* adapter = nullptr;
    static bool tried = false;
    std::lock_guard lock(m);
    if (!tried && device != nullptr)
    {
        tried = true;
        IDXGIFactory4* factory = nullptr;
        if (SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&factory))) && factory != nullptr)
        {
            if (FAILED(factory->EnumAdapterByLuid(device->GetAdapterLuid(), IID_PPV_ARGS(&adapter))))
                adapter = nullptr;
            factory->Release();
        }
        LOG_INFO("Per-eye DLSS DX12: VRAM readings {}", adapter != nullptr ? "ready (the game's adapter, local memory)"
                                                                          : "UNAVAILABLE (no adapter by the device's LUID)");
    }
    if (adapter == nullptr)
        return 0;
    DXGI_QUERY_VIDEO_MEMORY_INFO info {};
    if (FAILED(adapter->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &info)))
        return 0;
    return info.CurrentUsage >> 20;
}

void LabelNextCreate(const char* label)
{
    g_nextCreateLabel = label;
}

void ReportCreateVram(NVSDK_NGX_Parameter* params, uint64_t beforeMB, uint64_t afterMB, bool direct)
{
    const int kind = g_kindForCreate.load();
    const bool twin = kind == 0 && g_nextCreateLabel != nullptr;
    const int slot = kind == 1 ? 1 : kind == 2 ? 2 : twin ? 4 : 0;
    static const char* names[5] = { "the game's feature", "foveal crop", "periphery", "NR model", "twin" };
    const int64_t took = beforeMB && afterMB ? (int64_t) afterMB - (int64_t) beforeMB : 0;
    g_vramMade[slot] += took;
    ++g_vramCreates[slot];
    unsigned int w = 0, h = 0, ow = 0, oh = 0;
    if (params != nullptr)
    {
        params->Get(NVSDK_NGX_Parameter_Width, &w);
        params->Get(NVSDK_NGX_Parameter_Height, &h);
        params->Get(NVSDK_NGX_Parameter_OutWidth, &ow);
        params->Get(NVSDK_NGX_Parameter_OutHeight, &oh);
    }
    LOG_INFO("Per-eye DLSS DX12: VRAM: {}{} {}x{} -> {}x{} made: {:+} MB (process {} -> {} MB)", names[slot],
             direct ? " (direct)" : "", w, h, ow, oh, took, beforeMB, afterMB);
}

void ReportNrCreateVram(unsigned int w, unsigned int h, uint64_t beforeMB, uint64_t afterMB)
{
    const int64_t took = beforeMB && afterMB ? (int64_t) afterMB - (int64_t) beforeMB : 0;
    g_vramMade[3] += took;
    ++g_vramCreates[3];
    LOG_INFO("Per-eye DLSS DX12: VRAM: NR model {}x{} made: {:+} MB (process {} -> {} MB)", w, h, took, beforeMB, afterMB);
}

void RequestOuterPreset(int preset)
{
    std::lock_guard lock(g_mutex);
    auto* cfg = Config::Instance();
    cfg->RenderPresetOverride = true;
    cfg->RenderPresetForAll = (uint32_t) preset;
    // Only the game's features and their twins: the crops keep their own preset.
    for (auto& [primary, twin] : g_twins)
        for (unsigned int h : { primary, twin })
        {
            State::Instance().changeBackend[h] = true;
            g_dlssRebuild[h] = 1;
        }
    LOG_INFO("Per-eye DLSS DX12: outer (full pass) preset -> {} (overlay); {} feature(s) rebuild live", preset,
             g_twins.size() * 2);
}

void RequestFoveaSize(float width, float height)
{
    std::lock_guard lock(g_mutex);
    Config::Instance()->UpscalerFoveaCrop = std::clamp(width, 0.1f, 0.9f);
    Config::Instance()->UpscalerFoveaCropHeight = std::clamp(height, 0.1f, 0.9f);
}

void RequestFoveaOffset(float x, float y, float sideways)
{
    std::lock_guard lock(g_mutex);
    Config::Instance()->UpscalerFoveaOffsetX = std::clamp(x, -0.4f, 0.4f);
    Config::Instance()->UpscalerFoveaOffsetY = std::clamp(y, -0.4f, 0.4f);
    Config::Instance()->UpscalerFoveaOffsetSideways = std::clamp(sideways, -0.4f, 0.4f);
}

void RequestGazeBoxWhenStill(bool on)
{
    std::lock_guard lock(g_mutex);
    g_gazeBoxWhenStill = on;
    LOG_INFO("Per-eye DLSS DX12: the green gaze box is drawn with the outline {} (overlay; not saved)",
             on ? "ALSO while the crop stands still" : "only while the crop follows the gaze");
}

void RequestFoveaCopyInputs(bool on)
{
    std::lock_guard lock(g_mutex);
    Config::Instance()->UpscalerFoveaCopyInputs = on;
    for (auto& entry : g_cropReset)
        entry.second = true; // a different input path: start the crops' histories over
    LOG_INFO("Per-eye DLSS DX12: crop inputs {} (overlay)", on ? "COPIED" : "through DLSS subrects");
}

void RequestFoveaCropMask(int mode)
{
    std::lock_guard lock(g_mutex);
    Config::Instance()->UpscalerFoveaCropMask = std::clamp(mode, 0, 2);
    for (auto& entry : g_cropReset)
        entry.second = true; // a different input: start the crops' histories over
    LOG_INFO("Per-eye DLSS DX12: the game's bias mask at the crop is now mode {} (overlay; 0 none, 1 copied at the box, "
             "2 as handed)",
             std::clamp(mode, 0, 2));
}

void RequestFitDepth(int mode)
{
    std::lock_guard lock(g_mutex);
    Config::Instance()->UpscalerFitDepth = std::clamp(mode, 0, 2);
    LOG_INFO("Per-eye DLSS DX12: same-eye fit depth set to {} (overlay)",
             mode == 0 ? "Auto" : mode == 1 ? "Hardware" : "Linear");
}

void RequestGiveUpStale(bool on)
{
    std::lock_guard lock(g_mutex);
    Config::Instance()->UpscalerFitGiveUpStale = on;
    LOG_INFO("Per-eye DLSS DX12: a stale start of the same-eye fit is {} (overlay)",
             on ? "GIVEN UP past two rejections in a row" : "KEPT however long (as it used to be)");
}

void RequestFitDump()
{
    std::lock_guard lock(g_mutex);
    if (!g_motionReady || g_dumpReadbackData == nullptr)
    {
        LOG_WARN("Per-eye DLSS DX12: FIT DUMP refused: {}",
                 !g_motionReady ? "same-eye motion is not running (per-eye routing, F6)" : "no dump readback buffer");
        return;
    }
    const auto now = std::time(nullptr);
    std::tm local {};
    localtime_s(&local, &now);
    g_dumpFolder = (Util::ExePath().parent_path() / "OptiFitDump" /
                    std::format("{:04}{:02}{:02}-{:02}{:02}{:02}", local.tm_year + 1900, local.tm_mon + 1, local.tm_mday,
                                local.tm_hour, local.tm_min, local.tm_sec))
                       .string();
    g_dumpLeft = kDumpFits;
    LOG_INFO("Per-eye DLSS DX12: FIT DUMP armed (overlay): the next {} same-eye fits go to {}", kDumpFits, g_dumpFolder);
}

void RequestFitLeaveOutNearest(bool on)
{
    std::lock_guard lock(g_mutex);
    Config::Instance()->UpscalerFitLeaveOutNearest = on;
    LOG_INFO("Per-eye DLSS DX12: the same-eye fit {} (overlay)",
             on ? "LEAVES OUT what lies nearer than a depth gap (a car around the camera), restarts at 8 px, steps 128/8/2 px"
                : "fits every sample (as it used to be)");
}

void RequestFitRestart(bool on)
{
    std::lock_guard lock(g_mutex);
    Config::Instance()->UpscalerFitRestart = on;
    LOG_INFO("Per-eye DLSS DX12: the same-eye fit {} (overlay)",
             on ? "RESTARTS from no motion within the fit when the eye's last fit is left behind"
                : "does NOT restart within the fit (as before: two rejected fits first)");
}

void RequestSameEyeObjectMotion(int mode)
{
    std::lock_guard lock(g_mutex);
    Config::Instance()->UpscalerSameEyeObjectMotion = std::clamp(mode, 0, 3);
    LOG_INFO("Per-eye DLSS DX12: the composed same-eye motion is now {} (overlay)", ObjectMotionName());
}

void RequestVectorsSameEye(bool on)
{
    std::lock_guard lock(g_mutex);
    Config::Instance()->UpscalerVectorsSameEye = on;
    LOG_INFO("Per-eye DLSS DX12: the parameters' vectors are {} (overlay)",
             on ? "left as they are (DL(S)SS on): no fit or compose of ours" : "the game's other-eye ones: ours are made from them");
}

void RequestSameEyeBiasMask(int mode)
{
    std::lock_guard lock(g_mutex);
    Config::Instance()->UpscalerSameEyeBiasMask = std::clamp(mode, 0, 2);
    LOG_INFO("Per-eye DLSS DX12: DLSS's current-colour bias mask is now {} (overlay)", BiasMaskName());
}

// ---- Cheap periphery ----

bool PeripheryOn()
{
    if (!Enabled())
        return false;
    std::lock_guard lock(g_mutex);
    return g_foveaOn && !g_foveaFailed && !g_periphFailed &&
           Config::Instance()->UpscalerFoveaCheapPeriphery.value_or_default();
}

unsigned int PeripheryOf(unsigned int handle)
{
    std::lock_guard lock(g_mutex);
    auto it = g_periphs.find(PeriphKey(handle));
    return it == g_periphs.end() ? 0 : it->second;
}

unsigned int ForgetPeriphery(unsigned int handle)
{
    std::lock_guard lock(g_mutex);
    g_mainSkipped.erase(handle);
    if (auto taa = g_taa.find(handle); taa != g_taa.end())
    {
        // The eye's TAA history goes with its feature, released once no recorded work can reference it.
        for (auto* tex : taa->second.tex)
            if (tex != nullptr)
                g_retired.push_back({ tex, g_present.load() });
        g_taa.erase(taa);
    }
    auto it = g_periphs.find(handle);
    if (it == g_periphs.end())
        return 0;
    const unsigned int periph = it->second;
    g_periphs.erase(it);
    g_periphSize.erase(periph);
    g_periphReset.erase(periph);
    g_periphEverOk.erase(periph);
    g_periphRemade.erase(periph);
    g_dlssRebuild.erase(periph);
    return periph;
}

bool PlanPeriphery(unsigned int handle, FoveaSize* create)
{
    std::lock_guard lock(g_mutex);
    // PlanFovea ran for this evaluate: the eye's render size is in the plan.
    const unsigned int rw = g_foveaPlan.renderW, rh = g_foveaPlan.renderH;
    if (rw == 0 || rh == 0 || g_device == nullptr)
        return false;
    if (auto it = g_periphs.find(PeriphKey(handle)); it != g_periphs.end())
    {
        const auto made = g_periphSize[it->second];
        if (made.first != rw || made.second != rh)
        {
            ++g_periphFallbacks;
            LogOnce(g_loggedPeriphSkip, std::format("cheap periphery skipped: its feature was made for {}x{}, this "
                                                    "evaluate renders {}x{} (the full pass runs; counted per window)",
                                                    made.first, made.second, rw, rh));
            return false;
        }
    }
    if (!g_periphTried && !SetupPeriphery(g_device))
        g_periphFailed = true;
    if (!g_periphReady)
        return false;
    create->renderW = create->outW = rw;
    create->renderH = create->outH = rh;
    create->preset = Config::Instance()->UpscalerFoveaPeripheryPreset.value_or_default();
    return true;
}

void PeripheryCreated(unsigned int handle, unsigned int periph)
{
    std::lock_guard lock(g_mutex);
    g_periphs[PeriphKey(handle)] = periph;
    g_periphSize[periph] = { g_foveaPlan.renderW, g_foveaPlan.renderH };
    g_periphReset[periph] = true;
    LOG_INFO("Per-eye DLSS DX12: periphery feature {} made for feature {} ({}x{} in and out, preset {})", periph, handle,
             g_foveaPlan.renderW, g_foveaPlan.renderH,
             Config::Instance()->UpscalerFoveaPeripheryPreset.value_or_default());
}

void PeripheryCreateFailed()
{
    std::lock_guard lock(g_mutex);
    g_periphFailed = true;
    LOG_WARN("Per-eye DLSS DX12: the periphery feature could not be made; the full pass runs this session");
}

void PeripheryNotReady()
{
    std::lock_guard lock(g_mutex);
    ++g_periphFallbacks;
    LogOnce(g_loggedPeriphNotReady, "cheap periphery not made yet: the game's feature is being rebuilt; the full pass "
                                    "runs and it is asked again at the next evaluate (counted per window with the "
                                    "fall-backs)");
}

bool BeginPeriphery(ID3D12GraphicsCommandList* cmdList, NVSDK_NGX_Parameter* params, unsigned int periph,
                    unsigned int owner)
{
    std::lock_guard lock(g_mutex);
    auto size = g_periphSize.find(periph);
    if (size == g_periphSize.end() || g_device == nullptr)
        return false;
    const unsigned int rw = size->second.first, rh = size->second.second;
    ID3D12Resource* output = GetResource(params, NVSDK_NGX_Parameter_Output);
    if (output == nullptr || !EnsurePeriphTex(g_device, rw, rh, TypedFormat(output->GetDesc().Format), g_present.load()))
    {
        ++g_periphFallbacks;
        LogOnce(g_loggedPeriphFail, "cheap periphery skipped: no output or no periphery texture (the full pass runs)");
        return false;
    }
    auto& s = g_savedP;
    s.width = GetUIntParam(params, NVSDK_NGX_Parameter_Width);
    s.height = GetUIntParam(params, NVSDK_NGX_Parameter_Height);
    s.outW = GetUIntParam(params, NVSDK_NGX_Parameter_OutWidth);
    s.outH = GetUIntParam(params, NVSDK_NGX_Parameter_OutHeight);
    s.outX = GetUIntParam(params, NVSDK_NGX_Parameter_DLSS_Output_Subrect_Base_X);
    s.outY = GetUIntParam(params, NVSDK_NGX_Parameter_DLSS_Output_Subrect_Base_Y);
    s.output = output;
    s.reset = 0;
    params->Get(NVSDK_NGX_Parameter_Reset, &s.reset);
    // One feature for both eyes (UpscalerPeripheryOneFeature): the game's own vectors, as the game's feature gets them;
    // under a DLSS pass below us they are read as they stand. A feature per eye takes our same-eye motion below.
    const bool oneFeature = PeripheryOneFeature();
    ++(oneFeature ? g_periphOneEvals : g_periphPerEyeEvals);
    if (!oneFeature && g_composedForShared && g_composed != nullptr && !g_routing && !g_swapped)
    {
        // Shared routing, our same-eye motion made for NR: the periphery (a history per eye) takes it; put back by
        // RestoreInputs after the evaluate like the per-eye swap.
        g_swappedFrom = GetResource(params, NVSDK_NGX_Parameter_MotionVectors);
        params->Set(NVSDK_NGX_Parameter_MotionVectors, g_composed);
        g_swapped = true;
        static bool logged = false;
        if (!logged)
        {
            logged = true;
            LOG_INFO("Per-eye DLSS DX12: FIRST cheap periphery on our same-eye motion in shared routing");
        }
    }
    // The whole eye's inputs as they are (render subrect, bases, jitter, motion -- per eye our same-eye motion); only
    // the output is ours, at the render size. Creation sizes too, for a live rebuild of it.
    params->Set(NVSDK_NGX_Parameter_Width, rw);
    params->Set(NVSDK_NGX_Parameter_Height, rh);
    params->Set(NVSDK_NGX_Parameter_OutWidth, rw);
    params->Set(NVSDK_NGX_Parameter_OutHeight, rh);
    params->Set(NVSDK_NGX_Parameter_DLSS_Output_Subrect_Base_X, 0u);
    params->Set(NVSDK_NGX_Parameter_DLSS_Output_Subrect_Base_Y, 0u);
    params->Set(NVSDK_NGX_Parameter_Output, g_periphTex);
    if (g_periphReset[periph])
    {
        params->Set(NVSDK_NGX_Parameter_Reset, 1);
        g_periphReset[periph] = false;
    }
    g_presetForCreate = Config::Instance()->UpscalerFoveaPeripheryPreset.value_or_default();
    g_kindForCreate = 2;
    if (g_periphTexState != D3D12_RESOURCE_STATE_UNORDERED_ACCESS)
    {
        Transition(cmdList, g_periphTex, g_periphTexState, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        g_periphTexState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    }
    g_periphNow = periph;
    g_periphOwnerNow = owner;
    return true;
}

bool EndPeriphery(ID3D12GraphicsCommandList* cmdList, NVSDK_NGX_Parameter* params, bool evaluated, bool rebuilding)
{
    std::lock_guard lock(g_mutex);
    const auto& s = g_savedP;
    params->Set(NVSDK_NGX_Parameter_Width, s.width);
    params->Set(NVSDK_NGX_Parameter_Height, s.height);
    params->Set(NVSDK_NGX_Parameter_OutWidth, s.outW);
    params->Set(NVSDK_NGX_Parameter_OutHeight, s.outH);
    params->Set(NVSDK_NGX_Parameter_DLSS_Output_Subrect_Base_X, s.outX);
    params->Set(NVSDK_NGX_Parameter_DLSS_Output_Subrect_Base_Y, s.outY);
    params->Set(NVSDK_NGX_Parameter_Output, s.output);
    params->Set(NVSDK_NGX_Parameter_Reset, s.reset);
    g_presetForCreate = 0;
    g_kindForCreate = 0;
    if (rebuilding || !evaluated)
    {
        // The game's full pass runs for this frame instead.
        ++g_periphFallbacks;
        if (!rebuilding)
        {
            LogOnce(g_loggedPeriphFail, "cheap periphery: its evaluate failed (the full pass runs; counted per window)");
            // Once per feature, on its first failure (the crossed feature's FIRST evaluate went
            // through and the next one failed, so 'never succeeded' was too narrow).
            if (g_periphRemade[g_periphNow] < 1)
            {
                ++g_periphRemade[g_periphNow];
                ++g_periphRemadeTotal;
                State::Instance().changeBackend[g_periphNow] = true;
                g_dlssRebuild[g_periphNow] = 1;
                g_periphReset[g_periphNow] = true;
                LOG_INFO("Per-eye DLSS DX12: periphery feature {} FAILED AN EVALUATE (ok before: {}): remade once through the live "
                         "rebuild (Crimson Desert: the first SR feature made in that RR game answered every "
                         "evaluate with InvalidParameter; the next one was fine); counted since start in the window line",
                         g_periphNow, g_periphEverOk[g_periphNow] ? "yes" : "no");
            }
        }
        return false;
    }
    g_periphEverOk[g_periphNow] = true;

    // ENLARGE our render-size picture over the whole eye (the crop is drawn over the centre next). The output is a UAV
    // at an upscaler's evaluate (the DLSS contract); our texture: UAV (DLSS wrote it) -> shader resource.
    D3D12_RESOURCE_BARRIER uav {};
    uav.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    uav.UAV.pResource = s.output;
    cmdList->ResourceBarrier(1, &uav);
    Transition(cmdList, g_periphTex, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
               D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    g_periphTexState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    const D3D12_RESOURCE_DESC od = s.output->GetDesc();
    const uint32_t slot = (uint32_t) (g_enlargeSeq++ % kRing);
    D3D12_CPU_DESCRIPTOR_HANDLE cpu = g_enlargeHeap->GetCPUDescriptorHandleForHeapStart();
    cpu.ptr += (SIZE_T) slot * 2 * g_descriptorSize;
    D3D12_GPU_DESCRIPTOR_HANDLE gpu = g_enlargeHeap->GetGPUDescriptorHandleForHeapStart();
    gpu.ptr += (UINT64) slot * 2 * g_descriptorSize;
    D3D12_SHADER_RESOURCE_VIEW_DESC srv {};
    srv.Format = g_periphTexFormat;
    srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srv.Texture2D.MipLevels = 1;
    g_device->CreateShaderResourceView(g_periphTex, &srv, cpu);
    cpu.ptr += g_descriptorSize;
    D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc {};
    uavDesc.Format = TypedFormat(od.Format);
    uavDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    g_device->CreateUnorderedAccessView(s.output, nullptr, &uavDesc, cpu);
    const uint32_t constants[8] = { (uint32_t) od.Width, od.Height, (uint32_t) g_periphTexW, g_periphTexH, 0, 0, 0, 0 };
    ID3D12DescriptorHeap* heaps[] = { g_enlargeHeap };
    cmdList->SetDescriptorHeaps(1, heaps);
    cmdList->SetComputeRootSignature(g_enlargeRoot);
    cmdList->SetPipelineState(g_enlargePso);
    cmdList->SetComputeRoot32BitConstants(0, 8, constants, 0);
    cmdList->SetComputeRootDescriptorTable(1, gpu);
    cmdList->Dispatch((UINT) ((od.Width + 7) / 8), (od.Height + 7) / 8, 1);
    g_mainSkipped[g_periphOwnerNow] = true;
    ++g_periphEvals;
    if (!g_loggedPeriphFirst)
    {
        g_loggedPeriphFirst = true;
        LOG_INFO("Per-eye DLSS DX12: FIRST cheap periphery drawn (feature {}, {}x{} -> enlarged to {}x{}); the game's "
                 "full pass is skipped while it runs",
                 g_periphNow, g_periphTexW, g_periphTexH, od.Width, od.Height);
    }
    return true;
}

// ---- Own TAA periphery ----

bool PeripheryTaa()
{
    if (!Enabled())
        return false;
    std::lock_guard lock(g_mutex);
    return Config::Instance()->UpscalerFoveaPeripheryTaa.value_or_default() && !g_taaFailed;
}

bool TaaPeriphery(ID3D12GraphicsCommandList* cmdList, NVSDK_NGX_Parameter* params, unsigned int owner)
{
    std::lock_guard lock(g_mutex);
    auto skip = [&](const std::string& why)
    {
        ++g_taaSkips;
        LogOnce(g_loggedTaaSkip, std::format("own TAA periphery skipped ({}); the DLSS periphery runs instead (counted "
                                             "per window)",
                                             why));
        return false;
    };
    const FoveaRect plan = g_foveaPlan; // PlanFovea ran for this evaluate: the eye's render and output sizes
    if (cmdList == nullptr || params == nullptr || g_device == nullptr || plan.renderW == 0 || plan.outW == 0)
        return skip("no plan for this evaluate");
    if (!g_taaTried && !SetupTaa(g_device))
        g_taaFailed = true;
    if (!g_taaReady)
        return false; // the reason was logged once by SetupTaa

    ID3D12Resource* output = GetResource(params, NVSDK_NGX_Parameter_Output);
    ID3D12Resource* colour = GetResource(params, NVSDK_NGX_Parameter_Color);
    ID3D12Resource* motion = GetResource(params, NVSDK_NGX_Parameter_MotionVectors);
    ID3D12Resource* depth = GetResource(params, NVSDK_NGX_Parameter_Depth);
    if (output == nullptr || colour == nullptr || motion == nullptr || depth == nullptr)
        return skip("a missing input or output");
    const uint32_t colourBase[2] = { GetUIntParam(params, NVSDK_NGX_Parameter_DLSS_Input_Color_Subrect_Base_X),
                                     GetUIntParam(params, NVSDK_NGX_Parameter_DLSS_Input_Color_Subrect_Base_Y) };
    const uint32_t motionBase[2] = { GetUIntParam(params, NVSDK_NGX_Parameter_DLSS_Input_MV_SubrectBase_X),
                                     GetUIntParam(params, NVSDK_NGX_Parameter_DLSS_Input_MV_SubrectBase_Y) };
    const uint32_t depthBase[2] = { GetUIntParam(params, NVSDK_NGX_Parameter_DLSS_Input_Depth_Subrect_Base_X),
                                    GetUIntParam(params, NVSDK_NGX_Parameter_DLSS_Input_Depth_Subrect_Base_Y) };
    if (GetUIntParam(params, NVSDK_NGX_Parameter_DLSS_Output_Subrect_Base_X) != 0 ||
        GetUIntParam(params, NVSDK_NGX_Parameter_DLSS_Output_Subrect_Base_Y) != 0)
        return skip("the output has a subrect offset");
    const D3D12_RESOURCE_DESC od = output->GetDesc(), cd = colour->GetDesc(), md = motion->GetDesc(),
                              dd = depth->GetDesc();
    if (od.Width != plan.outW || od.Height != plan.outH)
        return skip(std::format("output {}x{} is not the planned eye {}x{}", od.Width, od.Height, plan.outW, plan.outH));
    auto fits = [&](const D3D12_RESOURCE_DESC& d, const uint32_t base[2])
    {
        return d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D && base[0] + plan.renderW <= d.Width &&
               base[1] + plan.renderH <= d.Height;
    };
    // Motion at the render size only (MVLowRes), as GATHER assumes.
    if (!fits(cd, colourBase) || !fits(md, motionBase) || !fits(dd, depthBase))
        return skip(std::format("render {}x{} does not fit colour {}x{} / motion {}x{} / depth {}x{}", plan.renderW,
                                plan.renderH, cd.Width, cd.Height, md.Width, md.Height, dd.Width, dd.Height));
    float jitter[2] = { 0, 0 }, scale[2] = { 0, 0 };
    params->Get(NVSDK_NGX_Parameter_Jitter_Offset_X, &jitter[0]);
    params->Get(NVSDK_NGX_Parameter_Jitter_Offset_Y, &jitter[1]);
    params->Get(NVSDK_NGX_Parameter_MV_Scale_X, &scale[0]);
    params->Get(NVSDK_NGX_Parameter_MV_Scale_Y, &scale[1]);
    if (scale[0] == 0 || scale[1] == 0)
        return skip("MV scale is zero");
    const bool flip = Config::Instance()->UpscalerFoveaTaaFlipJitter.value_or_default();
    if (flip)
    {
        jitter[0] = -jitter[0];
        jitter[1] = -jitter[1];
    }

    const uint64_t present = g_present.load();
    // Below 1, the TAA (and its history) runs at a fraction of the eye into g_taaMid and ENLARGE fills the eye: the pass
    // was bandwidth-bound at the full eye (0.27-0.29 ms at 3472^2), so fewer pixels is the only saving there is.
    const float taaScale = std::clamp(Config::Instance()->UpscalerFoveaTaaScale.value_or_default(), 0.25f, 1.0f);
    bool scaled = taaScale < 0.99f;
    if (scaled && !g_periphTried)
        SetupPeriphery(g_device);
    if (scaled && !g_periphReady)
    {
        scaled = false;
        LogOnce(g_loggedTaaScaleFail, "TAA resolution below 1 needs the enlarge pass, which failed; the TAA runs at the eye size");
    }
    const UINT tw = scaled ? (UINT) std::max<double>(64.0, std::round((double) od.Width * taaScale)) : (UINT) od.Width;
    const UINT th = scaled ? (UINT) std::max<double>(64.0, std::round((double) od.Height * taaScale)) : od.Height;
    if (scaled && !EnsureTaaMid(g_device, tw, th, present))
        return skip(std::format("could not make the {}x{} TAA texture", tw, th));
    auto& hist = g_taa[EyeKey(owner)];
    if (!EnsureTaaHistory(g_device, hist, tw, th, present))
        return skip("could not make the history textures");
    int reset = 0;
    params->Get(NVSDK_NGX_Parameter_Reset, &reset);
    // Per eye the owner is one eye, whose previous evaluate was two Presents back (same-eye motion). Shared (F6 off) the
    // owner is the game's feature for both eyes: ONE history, reprojected by the game's own vectors, which point at the
    // previous Present -- the other eye -- so it follows one Present back, as the game's shared DLSS does. Anything else
    // (a gap, the first frame, a routing switch, a reset or a chain break -- SameEye sets Reset) starts it over.
    const uint64_t stride = g_routing ? 2 : 1;
    const bool useHistory = reset == 0 && hist.lastPresent != UINT64_MAX && present == hist.lastPresent + stride;
    const int rd = (int) hist.read, wr = rd ^ 1;
    TaaState(cmdList, hist, rd, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    TaaState(cmdList, hist, wr, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    const uint32_t slot = (uint32_t) (g_taaSeq++ % kRing);
    D3D12_CPU_DESCRIPTOR_HANDLE cpu = g_taaHeap->GetCPUDescriptorHandleForHeapStart();
    cpu.ptr += (SIZE_T) slot * 6 * g_descriptorSize;
    D3D12_GPU_DESCRIPTOR_HANDLE gpu = g_taaHeap->GetGPUDescriptorHandleForHeapStart();
    gpu.ptr += (UINT64) slot * 6 * g_descriptorSize;
    D3D12_SHADER_RESOURCE_VIEW_DESC srv {};
    srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srv.Texture2D.MipLevels = 1;
    // The game's inputs (and per eye our own motion) are read as DLSS reads them: no barrier on them.
    srv.Format = TypedFormat(cd.Format);
    g_device->CreateShaderResourceView(colour, &srv, cpu);
    cpu.ptr += g_descriptorSize;
    srv.Format = MotionSrvFormat(md.Format);
    g_device->CreateShaderResourceView(motion, &srv, cpu);
    cpu.ptr += g_descriptorSize;
    srv.Format = DepthSrvFormat(dd.Format);
    g_device->CreateShaderResourceView(depth, &srv, cpu);
    cpu.ptr += g_descriptorSize;
    srv.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
    g_device->CreateShaderResourceView(hist.tex[rd], &srv, cpu);
    cpu.ptr += g_descriptorSize;
    D3D12_UNORDERED_ACCESS_VIEW_DESC uav {};
    uav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    uav.Format = scaled ? DXGI_FORMAT_R16G16B16A16_FLOAT : TypedFormat(od.Format);
    g_device->CreateUnorderedAccessView(scaled ? g_taaMid : output, nullptr, &uav, cpu);
    cpu.ptr += g_descriptorSize;
    if (scaled && g_taaMidState != D3D12_RESOURCE_STATE_UNORDERED_ACCESS)
    {
        Transition(cmdList, g_taaMid, g_taaMidState, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        g_taaMidState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    }
    uav.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
    g_device->CreateUnorderedAccessView(hist.tex[wr], nullptr, &uav, cpu);

    // The output is a UAV at an upscaler's evaluate (the DLSS contract); the barrier orders our write after anything
    // earlier on this list that wrote it.
    D3D12_RESOURCE_BARRIER outBarrier {};
    outBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    outBarrier.UAV.pResource = output;
    cmdList->ResourceBarrier(1, &outBarrier);

    const float blend = std::clamp(Config::Instance()->UpscalerFoveaTaaBlend.value_or_default(), 0.01f, 1.0f);
    struct
    {
        uint32_t outSize[2], renderSize[2], colourBase[2], motionBase[2], depthBase[2];
        float jitter[2], mvScale[2];
        float blend;
        uint32_t reset;
    } constants { { (uint32_t) tw, th },
                  { plan.renderW, plan.renderH },
                  { colourBase[0], colourBase[1] },
                  { motionBase[0], motionBase[1] },
                  { depthBase[0], depthBase[1] },
                  { jitter[0], jitter[1] },
                  { scale[0], scale[1] },
                  blend,
                  useHistory ? 0u : 1u };
    static_assert(sizeof(constants) == 64);
    ID3D12DescriptorHeap* heaps[] = { g_taaHeap };
    cmdList->SetDescriptorHeaps(1, heaps);
    cmdList->SetComputeRootSignature(g_taaRoot);
    cmdList->SetPipelineState(g_taaPso);
    cmdList->SetComputeRoot32BitConstants(0, 16, &constants, 0);
    cmdList->SetComputeRootDescriptorTable(1, gpu);
    cmdList->Dispatch((tw + 7) / 8, (th + 7) / 8, 1);

    if (scaled)
    {
        // ENLARGE g_taaMid over the whole eye, as the cheap periphery does with its DLSS picture. Ours: UAV -> read.
        Transition(cmdList, g_taaMid, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        g_taaMidState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        const uint32_t eslot = (uint32_t) (g_enlargeSeq++ % kRing);
        D3D12_CPU_DESCRIPTOR_HANDLE ecpu = g_enlargeHeap->GetCPUDescriptorHandleForHeapStart();
        ecpu.ptr += (SIZE_T) eslot * 2 * g_descriptorSize;
        D3D12_GPU_DESCRIPTOR_HANDLE egpu = g_enlargeHeap->GetGPUDescriptorHandleForHeapStart();
        egpu.ptr += (UINT64) eslot * 2 * g_descriptorSize;
        srv.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
        g_device->CreateShaderResourceView(g_taaMid, &srv, ecpu);
        ecpu.ptr += g_descriptorSize;
        uav.Format = TypedFormat(od.Format);
        g_device->CreateUnorderedAccessView(output, nullptr, &uav, ecpu);
        const uint32_t econst[8] = { (uint32_t) od.Width, od.Height, tw, th, 0, 0, 0, 0 };
        ID3D12DescriptorHeap* eheaps[] = { g_enlargeHeap };
        cmdList->SetDescriptorHeaps(1, eheaps);
        cmdList->SetComputeRootSignature(g_enlargeRoot);
        cmdList->SetPipelineState(g_enlargePso);
        cmdList->SetComputeRoot32BitConstants(0, 8, econst, 0);
        cmdList->SetComputeRootDescriptorTable(1, egpu);
        cmdList->Dispatch((UINT) ((od.Width + 7) / 8), (od.Height + 7) / 8, 1);
        ++g_taaEnlarged;
    }

    hist.read = (uint32_t) wr;
    hist.lastPresent = present;
    ++g_taaEvals;
    if (!useHistory)
        ++g_taaResets;
    ++g_periphEvals;
    g_mainSkipped[owner] = true; // the game's feature, skipped meanwhile, starts over when it runs again
    if (!g_loggedTaaFirst[g_routing ? 1 : 0])
    {
        g_loggedTaaFirst[g_routing ? 1 : 0] = true;
        LOG_INFO("Per-eye DLSS DX12: FIRST own TAA periphery drawn {} (feature {}, render {}x{} -> {}x{}; colour fmt {} base "
                 "({}, {}), motion fmt {} {}x{}, depth fmt {}; jitter ({:+.3f}, {:+.3f}){}, MV scale ({:.1f}, {:.1f}), "
                 "blend {:.2f}, {})",
                 g_routing ? "PER EYE (same-eye motion, history 2 Presents back)"
                           : "SHARED (the game's vectors, history 1 Present back = the other eye)",
                 owner, plan.renderW, plan.renderH, od.Width, od.Height, (int) cd.Format, colourBase[0], colourBase[1],
                 (int) md.Format, md.Width, md.Height, (int) dd.Format, jitter[0], jitter[1],
                 flip ? " flipped" : "", scale[0], scale[1], blend, useHistory ? "with history" : "history reset");
    }
    return true;
}

void RequestPeripheryTaa(bool on)
{
    std::lock_guard lock(g_mutex);
    Config::Instance()->UpscalerFoveaPeripheryTaa = on;
    // Whichever periphery takes over has a stale history: the DLSS ones start over; ours restart by the Present gap.
    for (auto& entry : g_periphReset)
        entry.second = true;
    LOG_INFO("Per-eye DLSS DX12: own TAA periphery {} (overlay)", on ? "ON" : "off");
}

void RequestTaaBlend(float blend)
{
    std::lock_guard lock(g_mutex);
    Config::Instance()->UpscalerFoveaTaaBlend = std::clamp(blend, 0.01f, 1.0f);
}

void RequestTaaScale(float scale)
{
    std::lock_guard lock(g_mutex);
    Config::Instance()->UpscalerFoveaTaaScale = std::clamp(scale, 0.25f, 1.0f);
}

void RequestTaaFlipJitter(bool on)
{
    std::lock_guard lock(g_mutex);
    Config::Instance()->UpscalerFoveaTaaFlipJitter = on;
    LOG_INFO("Per-eye DLSS DX12: TAA periphery jitter sign {} (overlay)", on ? "FLIPPED" : "as the game gives it");
}

bool TakeMainSkipped(unsigned int handle)
{
    std::lock_guard lock(g_mutex);
    auto it = g_mainSkipped.find(handle);
    if (it == g_mainSkipped.end() || !it->second)
        return false;
    it->second = false;
    return true;
}

void RequestPeriphery(bool on)
{
    std::lock_guard lock(g_mutex);
    Config::Instance()->UpscalerFoveaCheapPeriphery = on;
    for (auto& entry : g_periphReset)
        entry.second = true;
    LOG_INFO("Per-eye DLSS DX12: cheap periphery {} (overlay)", on ? "ON" : "off");
}

bool PeripherySr()
{
    return Config::Instance()->UpscalerFoveaPeripherySr.value_or_default();
}

bool IsPeriphery(unsigned int handle)
{
    std::lock_guard lock(g_mutex);
    return g_periphSize.find(handle) != g_periphSize.end();
}

void RequestPeripherySr(bool on)
{
    std::lock_guard lock(g_mutex);
    Config::Instance()->UpscalerFoveaPeripherySr = on;
    // The periphery features are rebuilt as the other kind (RebuildBackend reads the setting); an SR game's are DLSS
    // either way, so the rebuild only changes an RR game's.
    for (auto& entry : g_periphSize)
    {
        State::Instance().changeBackend[entry.first] = true;
        g_dlssRebuild[entry.first] = 1;
        g_periphReset[entry.first] = true;
    }
    LOG_INFO("Per-eye DLSS DX12: SR periphery in RR mode {} (overlay); {} periphery feature(s) rebuild live",
             on ? "ON" : "off", g_periphSize.size());
}

void RequestPeripheryPreset(int preset)
{
    std::lock_guard lock(g_mutex);
    Config::Instance()->UpscalerFoveaPeripheryPreset = preset;
    for (auto& entry : g_periphSize)
    {
        State::Instance().changeBackend[entry.first] = true;
        g_dlssRebuild[entry.first] = 1;
    }
    LOG_INFO("Per-eye DLSS DX12: periphery preset -> {} (overlay); {} periphery feature(s) rebuild live", preset,
             g_periphSize.size());
}

void RequestFoveaFeather(int px)
{
    std::lock_guard lock(g_mutex);
    Config::Instance()->UpscalerFoveaFeather = std::clamp(px, 0, 256);
}

bool TakeDlssRebuild(unsigned int handle, bool stillChanging)
{
    std::lock_guard lock(g_mutex);
    auto it = g_dlssRebuild.find(handle);
    if (it == g_dlssRebuild.end())
        return false;
    if (!stillChanging)
        g_dlssRebuild.erase(it);
    return true;
}

PerEyeStatus GetStatus()
{
    std::lock_guard lock(g_mutex);
    PerEyeStatus st {};
    auto* cfg = Config::Instance();
    st.size = cfg->UpscalerFoveaCrop.value_or_default();
    st.sizeH = cfg->UpscalerFoveaCropHeight.value_or_default() > 0.0f ? cfg->UpscalerFoveaCropHeight.value_or_default()
                                                                     : st.size;
    st.offsetX = cfg->UpscalerFoveaOffsetX.value_or_default();
    st.offsetY = cfg->UpscalerFoveaOffsetY.value_or_default();
    st.offsetSideways = cfg->UpscalerFoveaOffsetSideways.value_or_default();
    st.feather = cfg->UpscalerFoveaFeather.value_or_default();
    st.copyInputs = cfg->UpscalerFoveaCopyInputs.value_or_default();
    st.copyInputsNow = g_cropCopiesLast >= 0 ? g_cropCopiesLast != 0 : st.copyInputs;
    st.copyInputsByHand = cfg->UpscalerFoveaCopyInputs.has_value();
    st.copiesSwappedBelow = g_copiesSwappedBelow;
    st.giveUpStale = cfg->UpscalerFitGiveUpStale.value_or_default();
    st.fitRestart = cfg->UpscalerFitRestart.value_or_default();
    st.fitLeaveOutNearest = cfg->UpscalerFitLeaveOutNearest.value_or_default();
    st.dumpLeft = g_dumpLeft;
    st.dumpSaved = g_dumpSaved.load();
    st.dumpLost = g_dumpLost;
    st.objectMotion = cfg->UpscalerSameEyeObjectMotion.value_or_default();
    st.biasMask = cfg->UpscalerSameEyeBiasMask.value_or_default();
    st.vectorsSameEye = cfg->UpscalerVectorsSameEye.value_or_default();
    st.fitDepth = cfg->UpscalerFitDepth.value_or_default();
    st.periphery = cfg->UpscalerFoveaCheapPeriphery.value_or_default();
    st.peripheryPreset = cfg->UpscalerFoveaPeripheryPreset.value_or_default();
    st.peripheryFailed = g_periphFailed;
    st.peripherySr = cfg->UpscalerFoveaPeripherySr.value_or_default();
    st.peripheryTaa = cfg->UpscalerFoveaPeripheryTaa.value_or_default();
    st.taaFailed = g_taaFailed;
    st.taaBlend = cfg->UpscalerFoveaTaaBlend.value_or_default();
    st.taaScale = cfg->UpscalerFoveaTaaScale.value_or_default();
    st.taaFlipJitter = cfg->UpscalerFoveaTaaFlipJitter.value_or_default();
    st.outerPreset = cfg->RenderPresetOverride.value_or_default() ? (int) cfg->RenderPresetForAll.value_or(0) : 0;
    st.leftParity = LeftParity();
    st.cropX = g_foveaPlan.ox;
    st.cropY = g_foveaPlan.oy;
    st.eyeW = g_foveaPlan.outW;
    st.eyeH = g_foveaPlan.outH;
    st.routing = g_routing;
    st.fovea = g_foveaOn;
    st.outline = g_foveaOutline;
    st.failed = g_foveaFailed;
    st.preset = Config::Instance()->UpscalerFoveaPreset.value_or_default();
    st.cropRr = g_cropIsRr.load();
    st.rrCropPreset = g_rrCropPreset.load();
    st.rrVectors = std::clamp(Config::Instance()->UpscalerFoveaRrVectors.value_or_default(), 0, 1);
    st.srVectors = std::clamp(Config::Instance()->UpscalerFoveaSrVectors.value_or_default(), -1, 1);
    st.srFromBelowNow = SrCropFromBelow();
    st.belowSeen = g_belowSeen.load();
    st.cropW = g_foveaPlan.rw;
    st.cropH = g_foveaPlan.rh;
    st.cropOutW = g_foveaPlan.ow;
    st.cropOutH = g_foveaPlan.oh;
    st.cropPercent = g_foveaPlan.outW ? 100.0f * g_foveaPlan.ow * g_foveaPlan.oh /
                                            ((float) g_foveaPlan.outW * g_foveaPlan.outH)
                                      : 0.0f;
    st.exact = g_foveaPlan.exact;
    st.upscalerMs = g_lastUpscalerMs;
    st.cropMs = g_lastCropMs;
    st.gazeOn = VarjoGaze::Enabled();
    st.gazeBoxWhenStill = g_gazeBoxWhenStill;
    if (st.gazeOn)
    {
        st.gazeText = VarjoGaze::StatusText();
        st.gazePlaced = g_foveaPlan.gaze;
        st.gazeX = g_foveaPlan.gazeX;
        st.gazeY = g_foveaPlan.gazeY;
        st.gazeEye = g_lastGazeEye;
        st.gazeFlipX = VarjoGaze::FlipX();
        st.gazeFlipY = VarjoGaze::FlipY();
        st.follow = Config::Instance()->UpscalerFoveaGazeFollow.value_or_default();
        st.deadZone = Config::Instance()->UpscalerFoveaGazeDeadZone.value_or_default();
        st.glideMs = Config::Instance()->UpscalerFoveaGazeGlideMs.value_or_default();
        st.ageFrames = Config::Instance()->UpscalerFoveaAgeFrames.value_or_default();
        st.followMoves = g_followMoves;
        st.followShifts = g_followShifts;
        st.followResets = g_followResets;
        st.followResetsNoCopy = g_followResetsNoCopy;
    }
    return st;
}

void AfterUpscale(ID3D12GraphicsCommandList* cmdList)
{
    if (!Enabled() || cmdList == nullptr)
        return;
    std::lock_guard lock(g_mutex);
    if (g_timingThis && !g_wroteUpscaleTime)
    {
        cmdList->EndQuery(g_queryHeap, D3D12_QUERY_TYPE_TIMESTAMP, g_timeSlotCur * 6 + 4);
        g_wroteUpscaleTime = true;
    }
}
} // namespace PerEyeDlss_Dx12
