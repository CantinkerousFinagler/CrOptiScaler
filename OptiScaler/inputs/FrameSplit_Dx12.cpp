#include "pch.h"

#include "Config.h"
#include "Logger.h"
#include "Util.h"

#include "FrameSplit_Dx12.h"
#include "GiCapture_Dx12.h"

#include <detours/detours.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <format>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

// Threads: every hook runs on the game's recording threads, OnPresent / Arm / GetStatus on the present thread.
// - A recording (Rec) is touched only by the thread recording its list (a D3D12 list is recorded by one thread at a
//   time), so its fields need no lock; `closed` is atomic because another thread's cache may look at it.
// - g_open (list -> open Rec) and g_recs are under g_mutex.
// - Recs are freed only while g_phase == Reading, after g_inHooks has drained to 0; every hook that touches a Rec
//   holds a HookGuard, which re-checks the phase after counting itself in. Thread caches carry the generation, which
//   is bumped at every free.
// - The query heap and the readback buffer are never released: a list recorded during a capture could in principle
//   run late and still write them.
namespace
{
constexpr uint32_t kCapturePresents = 64;
constexpr uint32_t kEdgePresents = 4; // Presents left out at each end (lists that straddle the start or the end)
constexpr uint32_t kDrainPresents = 32;
constexpr double kDelaySeconds = 5.0;
constexpr uint32_t kMaxQueries = 1u << 18;
constexpr uint32_t kChunk = 64;
constexpr uint32_t kNoQuery = 0xFFFFFFFFu;

enum Cat : uint16_t
{
    CatColour,
    CatDepthOnly,
    CatMesh,
    CatIndirectDraw,
    CatIndirectDepth,
    CatIndirectDispatch,
    CatIndirectRays,
    CatIndirectMesh,
    CatIndirectUnknown,
    CatCompute,
    CatRays,
    CatAccelBuild,
    CatCopy,
    CatBundle,
    CatRenderPass,
    CatOurs,
    CatAfterOurs,
    CatCount
};

const char* const kCatName[CatCount] = { "raster, colour targets",
                                         "raster, depth only",
                                         "mesh shaders",
                                         "indirect draws, colour targets",
                                         "indirect draws, depth only",
                                         "indirect dispatches (compute)",
                                         "indirect ray dispatches",
                                         "indirect mesh dispatches",
                                         "indirect, signature never seen created",
                                         "compute",
                                         "ray tracing (DispatchRays)",
                                         "BVH builds",
                                         "copies",
                                         "bundles",
                                         "render passes (BeginRenderPass)",
                                         "upscaler + our passes",
                                         "after the upscaler, before the next pass" };

enum Phase : int
{
    Idle = 0,
    Waiting = 1,
    Capturing = 2,
    Draining = 3,
    Reading = 4
};

struct Seg
{
    uint16_t cat = 0;
    uint16_t w = 0, h = 0;
    uint32_t q = kNoQuery;
    uint32_t draws = 0;
    uint32_t dispatches = 0;
};

struct Rec
{
    void* list = nullptr;
    bool compute = false;
    bool skip = false;
    bool overflow = false;
    std::atomic<bool> closed { false };
    bool inRenderPass = false;
    uint16_t vpW = 0, vpH = 0;
    uint16_t numRt = 1; // targets never seen set count as colour
    uint64_t present = 0;
    uint32_t endQ = kNoQuery;
    uint32_t chunkStart = kNoQuery;
    uint32_t chunkUsed = 0;
    std::vector<std::pair<uint32_t, uint32_t>> chunks;
    std::vector<Seg> segs;
};

std::atomic<int> g_phase { Idle };
std::atomic<int> g_inHooks { 0 };
std::atomic<uint32_t> g_gen { 1 };
std::atomic<uint32_t> g_nextQuery { 0 };
std::atomic<uint64_t> g_present { 0 };
std::atomic<bool> g_armRequested { false };

std::mutex g_mutex;
std::unordered_map<void*, Rec*> g_open;
std::vector<std::unique_ptr<Rec>> g_recs;
std::vector<std::string> g_lines;

ID3D12Device* g_device = nullptr; // identity only, never AddRef'd
ID3D12QueryHeap* g_heap = nullptr;
ID3D12Resource* g_readback = nullptr;
uint64_t* g_data = nullptr;
uint64_t g_frequency = 0;
bool g_installed = false;
bool g_setupTried = false;

uint64_t g_phaseStartPresent = 0;
uint64_t g_captureStart = 0;
std::chrono::steady_clock::time_point g_phaseStartTime;
std::chrono::steady_clock::time_point g_captureStartTime;
double g_captureSeconds = 0;

std::atomic<uint32_t> g_copyLists { 0 }, g_bundleLists { 0 }, g_otherLists { 0 }, g_overflowLists { 0 },
    g_renderPasses { 0 }, g_evaluates { 0 };
bool g_loggedProxy = false, g_loggedNative = false, g_loggedNoInstall = false, g_loggedSetupFail = false;

struct Cache
{
    void* list = nullptr;
    Rec* rec = nullptr;
    uint32_t gen = 0;
};
thread_local Cache t_cache;
// Inside NVSDK_NGX_D3D12_EvaluateFeature on this thread (the evaluate records synchronously on the game's recording
// thread). Per thread, not per list: in Crimson Desert the list the evaluate is handed is not a pointer the hooks ever
// see (a wrapper), while the calls made through it reach the hooks as the real list.
thread_local int t_oursDepth = 0;

// Counts the hook in while a capture or its drain runs; a Rec may be touched only while ok.
struct HookGuard
{
    bool ok = false;
    int phase = Idle;
    HookGuard()
    {
        if (int p = g_phase.load(); p != Capturing && p != Draining)
            return;
        g_inHooks.fetch_add(1);
        phase = g_phase.load();
        if (phase != Capturing && phase != Draining)
        {
            g_inHooks.fetch_sub(1);
            return;
        }
        ok = true;
    }
    ~HookGuard()
    {
        if (ok)
            g_inHooks.fetch_sub(1);
    }
    HookGuard(const HookGuard&) = delete;
    HookGuard& operator=(const HookGuard&) = delete;
};

inline bool Active()
{
    const int p = g_phase.load(std::memory_order_relaxed);
    return p == Capturing || p == Draining;
}

// New recordings start only while capturing; the drain only finishes the ones already open.
Rec* Find(void* list, bool create)
{
    const uint32_t gen = g_gen.load();
    if (t_cache.list == list && t_cache.gen == gen && t_cache.rec != nullptr && !t_cache.rec->closed.load())
        return t_cache.rec;
    Rec* r = nullptr;
    {
        std::lock_guard lock(g_mutex);
        if (auto it = g_open.find(list); it != g_open.end())
            r = it->second;
        else if (create)
        {
            auto rec = std::make_unique<Rec>();
            rec->list = list;
            switch (static_cast<ID3D12GraphicsCommandList*>(list)->GetType())
            {
            case D3D12_COMMAND_LIST_TYPE_DIRECT:
                break;
            case D3D12_COMMAND_LIST_TYPE_COMPUTE:
                rec->compute = true;
                break;
            case D3D12_COMMAND_LIST_TYPE_COPY: // a TIMESTAMP heap is not valid there
                rec->skip = true;
                ++g_copyLists;
                break;
            case D3D12_COMMAND_LIST_TYPE_BUNDLE: // no queries in bundles; their work is timed where they are executed
                rec->skip = true;
                ++g_bundleLists;
                break;
            default:
                rec->skip = true;
                ++g_otherLists;
                break;
            }
            r = rec.get();
            g_recs.push_back(std::move(rec));
            g_open[list] = r;
        }
    }
    t_cache = { list, r, gen };
    return r;
}

uint32_t AllocQuery(Rec* r)
{
    if (r->overflow)
        return kNoQuery;
    if (r->chunkStart == kNoQuery || r->chunkUsed == kChunk)
    {
        if (r->chunkStart != kNoQuery)
            r->chunks.emplace_back(r->chunkStart, r->chunkUsed);
        const uint32_t start = g_nextQuery.fetch_add(kChunk);
        if (start + kChunk > kMaxQueries)
        {
            r->overflow = true;
            r->chunkStart = kNoQuery;
            ++g_overflowLists;
            return kNoQuery;
        }
        r->chunkStart = start;
        r->chunkUsed = 0;
    }
    return r->chunkStart + r->chunkUsed++;
}

bool Stamp(void* list, Rec* r, uint32_t& q)
{
    q = AllocQuery(r);
    if (q == kNoQuery)
        return false;
    static_cast<ID3D12GraphicsCommandList*>(list)->EndQuery(g_heap, D3D12_QUERY_TYPE_TIMESTAMP, q);
    return true;
}

// `recordOn`: the list object to write the timestamp through, when it is not the one the Rec is keyed on (the
// evaluate's wrapper, which forwards to the real list).
void StartSeg(void* list, Rec* r, uint16_t cat, uint16_t w, uint16_t h, void* recordOn = nullptr)
{
    Seg s;
    s.cat = cat;
    s.w = w;
    s.h = h;
    if (Stamp(recordOn != nullptr ? recordOn : list, r, s.q))
        r->segs.push_back(s);
}

// One unit of GPU work of this kind. A new segment (and timestamp) starts when the kind or size changes; inside a
// render pass no timestamp is written (whether queries are allowed there is not certain), so the pass's segment
// carries everything until it ends.
void Note(void* list, uint16_t cat, uint16_t w, uint16_t h, uint32_t draws, uint32_t dispatches)
{
    HookGuard g;
    if (!g.ok)
        return;
    Rec* r = Find(list, g.phase == Capturing);
    if (r == nullptr || r->skip || r->overflow || r->closed.load())
        return;
    if (t_oursDepth > 0)
    {
        cat = CatOurs;
        w = h = 0;
    }
    const bool same = !r->segs.empty() && r->segs.back().cat == cat && r->segs.back().w == w && r->segs.back().h == h;
    if (!same && !(r->inRenderPass && !r->segs.empty()))
        StartSeg(list, r, cat, w, h);
    if (r->segs.empty())
        return;
    r->segs.back().draws += draws;
    r->segs.back().dispatches += dispatches;
}

void NoteDraw(void* list, uint16_t kind, uint32_t draws)
{
    // Read the list's current viewport and target count without starting a recording twice.
    uint16_t cat = kind;
    uint16_t w = 0, h = 0;
    {
        HookGuard g;
        if (!g.ok)
            return;
        if (Rec* r = Find(list, g.phase == Capturing); r != nullptr)
        {
            w = r->vpW;
            h = r->vpH;
            if (kind == CatColour && r->numRt == 0)
                cat = CatDepthOnly;
        }
    }
    Note(list, cat, w, h, draws, 0);
}

// Command signatures, by what their last argument does -- recorded at creation from startup, because an
// ExecuteIndirect call carries no hint of whether it draws or dispatches. Only a label: a freed signature's address
// reused for another kind would mislabel, never touch anything.
enum SigKind : uint8_t
{
    SigDraw,
    SigDispatch,
    SigRays,
    SigMesh,
    SigOther
};
std::mutex g_sigMutex;
std::unordered_map<void*, SigKind> g_sigs;
std::atomic<uint32_t> g_sigsCreated { 0 };

SigKind SignatureKind(void* sig)
{
    std::lock_guard lock(g_sigMutex);
    auto it = g_sigs.find(sig);
    return it == g_sigs.end() ? SigOther : it->second;
}

bool SignatureKnown(void* sig)
{
    std::lock_guard lock(g_sigMutex);
    return g_sigs.contains(sig);
}

using PFN_CreateCommandSignature = HRESULT(STDMETHODCALLTYPE*)(ID3D12Device*, const D3D12_COMMAND_SIGNATURE_DESC*,
                                                               ID3D12RootSignature*, REFIID, void**);
PFN_CreateCommandSignature o_CreateCommandSignature = nullptr;

using PFN_CreateConstantBufferView = void(STDMETHODCALLTYPE*)(ID3D12Device*, const D3D12_CONSTANT_BUFFER_VIEW_DESC*, D3D12_CPU_DESCRIPTOR_HANDLE);
PFN_CreateConstantBufferView o_CreateConstantBufferView = nullptr;

void STDMETHODCALLTYPE hkCreateConstantBufferView(ID3D12Device* This, const D3D12_CONSTANT_BUFFER_VIEW_DESC* desc, D3D12_CPU_DESCRIPTOR_HANDLE dest)
{
    GiCapture_Dx12::OnCreateCbv(desc, dest.ptr);
    o_CreateConstantBufferView(This, desc, dest);
}

using PFN_CreateShaderResourceView = void(STDMETHODCALLTYPE*)(ID3D12Device*, ID3D12Resource*, const D3D12_SHADER_RESOURCE_VIEW_DESC*, D3D12_CPU_DESCRIPTOR_HANDLE);
using PFN_CopyDescriptors = void(STDMETHODCALLTYPE*)(ID3D12Device*, UINT, const D3D12_CPU_DESCRIPTOR_HANDLE*, const UINT*, UINT,
                                                     const D3D12_CPU_DESCRIPTOR_HANDLE*, const UINT*, D3D12_DESCRIPTOR_HEAP_TYPE);
using PFN_CopyDescriptorsSimple = void(STDMETHODCALLTYPE*)(ID3D12Device*, UINT, D3D12_CPU_DESCRIPTOR_HANDLE, D3D12_CPU_DESCRIPTOR_HANDLE, D3D12_DESCRIPTOR_HEAP_TYPE);
PFN_CreateShaderResourceView o_CreateShaderResourceView = nullptr;
using PFN_CreateUnorderedAccessView = void(STDMETHODCALLTYPE*)(ID3D12Device*, ID3D12Resource*, ID3D12Resource*, const D3D12_UNORDERED_ACCESS_VIEW_DESC*, D3D12_CPU_DESCRIPTOR_HANDLE);
PFN_CreateUnorderedAccessView o_CreateUnorderedAccessView = nullptr;
PFN_CopyDescriptors o_CopyDescriptors = nullptr;
PFN_CopyDescriptorsSimple o_CopyDescriptorsSimple = nullptr;

void STDMETHODCALLTYPE hkCreateShaderResourceView(ID3D12Device* This, ID3D12Resource* resource, const D3D12_SHADER_RESOURCE_VIEW_DESC* desc,
                                                  D3D12_CPU_DESCRIPTOR_HANDLE dest)
{
    GiCapture_Dx12::OnCreateSrv(resource, desc, dest.ptr);
    o_CreateShaderResourceView(This, resource, desc, dest);
}

void STDMETHODCALLTYPE hkCreateUnorderedAccessView(ID3D12Device* This, ID3D12Resource* resource, ID3D12Resource* counter, const D3D12_UNORDERED_ACCESS_VIEW_DESC* desc,
                                                   D3D12_CPU_DESCRIPTOR_HANDLE dest)
{
    GiCapture_Dx12::OnCreateUav(resource, desc, dest.ptr);
    o_CreateUnorderedAccessView(This, resource, counter, desc, dest);
}

using PFN_CreateRootSignature = HRESULT(STDMETHODCALLTYPE*)(ID3D12Device*, UINT, const void*, SIZE_T, REFIID, void**);
PFN_CreateRootSignature o_CreateRootSignature = nullptr;

HRESULT STDMETHODCALLTYPE hkCreateRootSignature(ID3D12Device* This, UINT nodeMask, const void* blob, SIZE_T bytes, REFIID riid, void** out)
{
    const HRESULT made = o_CreateRootSignature(This, nodeMask, blob, bytes, riid, out);
    if (SUCCEEDED(made) && out != nullptr)
        GiCapture_Dx12::OnCreateRootSignature(*out, blob, bytes);
    return made;
}

void STDMETHODCALLTYPE hkCopyDescriptors(ID3D12Device* This, UINT destRanges, const D3D12_CPU_DESCRIPTOR_HANDLE* destStarts, const UINT* destSizes,
                                         UINT srcRanges, const D3D12_CPU_DESCRIPTOR_HANDLE* srcStarts, const UINT* srcSizes,
                                         D3D12_DESCRIPTOR_HEAP_TYPE type)
{
    if (type == D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV)
        GiCapture_Dx12::OnCopyDescriptors(This, destRanges, destStarts, destSizes, srcRanges, srcStarts, srcSizes);
    o_CopyDescriptors(This, destRanges, destStarts, destSizes, srcRanges, srcStarts, srcSizes, type);
}

void STDMETHODCALLTYPE hkCopyDescriptorsSimple(ID3D12Device* This, UINT count, D3D12_CPU_DESCRIPTOR_HANDLE dest, D3D12_CPU_DESCRIPTOR_HANDLE src,
                                               D3D12_DESCRIPTOR_HEAP_TYPE type)
{
    if (type == D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV)
        GiCapture_Dx12::OnCopyDescriptors(This, 1, &dest, &count, 1, &src, &count);
    o_CopyDescriptorsSimple(This, count, dest, src, type);
}

HRESULT STDMETHODCALLTYPE hkCreateCommandSignature(ID3D12Device* This, const D3D12_COMMAND_SIGNATURE_DESC* desc,
                                                   ID3D12RootSignature* root, REFIID riid, void** pp)
{
    const HRESULT hr = o_CreateCommandSignature(This, desc, root, riid, pp);
    if (SUCCEEDED(hr) && pp != nullptr && *pp != nullptr && desc != nullptr && desc->NumArgumentDescs > 0 &&
        desc->pArgumentDescs != nullptr)
    {
        SigKind kind = SigOther;
        switch (desc->pArgumentDescs[desc->NumArgumentDescs - 1].Type)
        {
        case D3D12_INDIRECT_ARGUMENT_TYPE_DRAW:
        case D3D12_INDIRECT_ARGUMENT_TYPE_DRAW_INDEXED:
            kind = SigDraw;
            break;
        case D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH:
            kind = SigDispatch;
            break;
        case D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH_RAYS:
            kind = SigRays;
            break;
        case D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH_MESH:
            kind = SigMesh;
            break;
        default:
            break;
        }
        // The object handed back may be for another interface of the same object; key on its IUnknown-free identity
        // as ExecuteIndirect receives it: the ID3D12CommandSignature pointer.
        ID3D12CommandSignature* sig = nullptr;
        if (SUCCEEDED(((IUnknown*) *pp)->QueryInterface(IID_PPV_ARGS(&sig))) && sig != nullptr)
        {
            {
                std::lock_guard lock(g_sigMutex);
                g_sigs[sig] = kind;
            }
            sig->Release();
            ++g_sigsCreated;
        }
    }
    return hr;
}

// ---- hooks ----
// A game's command lists need not share one vtable with a list we create ourselves: in Crimson Desert the first build,
// hooked from a scratch list only, saw no Draw, Dispatch, RSSetViewports or OMSetRenderTargets at all, and the lists
// the upscaler evaluates on never reached our Close. So every list vtable is registered -- ours at Install, the game's
// at CreateCommandList(1) -- and each method ADDRESS not hooked yet is hooked with its own trampoline (variant V of the
// hook). One address is hooked once, whichever vtables share it, so a hook never has to work out which one it is.
enum Slot : int
{
    SClose,
    SDraw,
    SDrawIndexed,
    SDispatch,
    SCopyBuffer,
    SCopyTexture,
    SCopyResource,
    SViewports,
    SBundle,
    STargets,
    SIndirect,
    SBeginRenderPass,
    SEndRenderPass,
    SBuildRtas,
    SRays,
    SMesh,
    SBarrier,
    SEnhancedBarrier,
    // Only with GiCapture_Dx12 on: the binds of its camera readers.
    SPipeline,
    SPipeline1,
    SComputeCbv,
    SGraphicsCbv,
    SComputeSrv,
    SGraphicsSrv,
    SHeaps,
    SComputeTable,
    SGraphicsTable,
    SComputeConstant,
    SGraphicsConstant,
    SComputeConstants,
    SGraphicsConstants,
    SComputeSignature,
    SGraphicsSignature,
    SCount
};
// d3d12.h 10.0.26100, ID3D12GraphicsCommandList7Vtbl. 68..76 need List4, 79 List6, 80 List7.
const int kSlotIndex[SCount] = { 9, 12, 13, 14, 15, 16, 17, 21, 27, 46, 59, 68, 69, 72, 76, 79, 26, 80, 25, 75, 37, 38, 39, 40, 28, 31, 32, 33, 34, 35, 36, 29, 30 };
const char* const kSlotName[SCount] = { "Close",         "DrawInstanced",   "DrawIndexedInstanced", "Dispatch",
                                        "CopyBufferRegion", "CopyTextureRegion", "CopyResource",     "RSSetViewports",
                                        "ExecuteBundle", "OMSetRenderTargets", "ExecuteIndirect",   "BeginRenderPass",
                                        "EndRenderPass", "BuildRaytracingAccelerationStructure", "DispatchRays",
                                        "DispatchMesh",  "ResourceBarrier", "Barrier",
                                        "SetPipelineState", "SetPipelineState1", "SetComputeRootCBV", "SetGraphicsRootCBV",
                                        "SetComputeRootSRV", "SetGraphicsRootSRV", "SetDescriptorHeaps", "SetComputeRootTable",
                                        "SetGraphicsRootTable", "SetComputeRoot32BitConstant", "SetGraphicsRoot32BitConstant",
                                        "SetComputeRoot32BitConstants", "SetGraphicsRoot32BitConstants", "SetComputeRootSignature", "SetGraphicsRootSignature" };
constexpr int kVariants = 8;
void* g_orig[kVariants][SCount] = {}; // trampolines, per variant and method
std::atomic<uint64_t> g_calls[SCount];  // calls seen while measuring, per method

using PFN_Close = HRESULT(STDMETHODCALLTYPE*)(ID3D12GraphicsCommandList*);
using PFN_DrawInstanced = void(STDMETHODCALLTYPE*)(ID3D12GraphicsCommandList*, UINT, UINT, UINT, UINT);
using PFN_DrawIndexedInstanced = void(STDMETHODCALLTYPE*)(ID3D12GraphicsCommandList*, UINT, UINT, UINT, INT, UINT);
using PFN_Dispatch = void(STDMETHODCALLTYPE*)(ID3D12GraphicsCommandList*, UINT, UINT, UINT);
using PFN_CopyBufferRegion = void(STDMETHODCALLTYPE*)(ID3D12GraphicsCommandList*, ID3D12Resource*, UINT64,
                                                      ID3D12Resource*, UINT64, UINT64);
using PFN_CopyTextureRegion = void(STDMETHODCALLTYPE*)(ID3D12GraphicsCommandList*, const D3D12_TEXTURE_COPY_LOCATION*,
                                                       UINT, UINT, UINT, const D3D12_TEXTURE_COPY_LOCATION*,
                                                       const D3D12_BOX*);
using PFN_CopyResource = void(STDMETHODCALLTYPE*)(ID3D12GraphicsCommandList*, ID3D12Resource*, ID3D12Resource*);
using PFN_RSSetViewports = void(STDMETHODCALLTYPE*)(ID3D12GraphicsCommandList*, UINT, const D3D12_VIEWPORT*);
using PFN_ExecuteBundle = void(STDMETHODCALLTYPE*)(ID3D12GraphicsCommandList*, ID3D12GraphicsCommandList*);
using PFN_OMSetRenderTargets = void(STDMETHODCALLTYPE*)(ID3D12GraphicsCommandList*, UINT,
                                                        const D3D12_CPU_DESCRIPTOR_HANDLE*, BOOL,
                                                        const D3D12_CPU_DESCRIPTOR_HANDLE*);
using PFN_ExecuteIndirect = void(STDMETHODCALLTYPE*)(ID3D12GraphicsCommandList*, ID3D12CommandSignature*, UINT,
                                                     ID3D12Resource*, UINT64, ID3D12Resource*, UINT64);
using PFN_BeginRenderPass = void(STDMETHODCALLTYPE*)(ID3D12GraphicsCommandList4*, UINT,
                                                     const D3D12_RENDER_PASS_RENDER_TARGET_DESC*,
                                                     const D3D12_RENDER_PASS_DEPTH_STENCIL_DESC*,
                                                     D3D12_RENDER_PASS_FLAGS);
using PFN_EndRenderPass = void(STDMETHODCALLTYPE*)(ID3D12GraphicsCommandList4*);
using PFN_BuildRtas = void(STDMETHODCALLTYPE*)(ID3D12GraphicsCommandList4*,
                                               const D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC*, UINT,
                                               const D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC*);
using PFN_DispatchRays = void(STDMETHODCALLTYPE*)(ID3D12GraphicsCommandList4*, const D3D12_DISPATCH_RAYS_DESC*);
using PFN_DispatchMesh = void(STDMETHODCALLTYPE*)(ID3D12GraphicsCommandList6*, UINT, UINT, UINT);
using PFN_ResourceBarrier = void(STDMETHODCALLTYPE*)(ID3D12GraphicsCommandList*, UINT, const D3D12_RESOURCE_BARRIER*);
using PFN_Barrier = void(STDMETHODCALLTYPE*)(ID3D12GraphicsCommandList7*, UINT32, const D3D12_BARRIER_GROUP*);
using PFN_SetPipelineState = void(STDMETHODCALLTYPE*)(ID3D12GraphicsCommandList*, ID3D12PipelineState*);
using PFN_SetPipelineState1 = void(STDMETHODCALLTYPE*)(ID3D12GraphicsCommandList4*, ID3D12StateObject*);
using PFN_SetRootView = void(STDMETHODCALLTYPE*)(ID3D12GraphicsCommandList*, UINT, D3D12_GPU_VIRTUAL_ADDRESS);
using PFN_SetDescriptorHeaps = void(STDMETHODCALLTYPE*)(ID3D12GraphicsCommandList*, UINT, ID3D12DescriptorHeap* const*);
using PFN_SetRootTable = void(STDMETHODCALLTYPE*)(ID3D12GraphicsCommandList*, UINT, D3D12_GPU_DESCRIPTOR_HANDLE);
using PFN_SetRootConstant = void(STDMETHODCALLTYPE*)(ID3D12GraphicsCommandList*, UINT, UINT, UINT);
using PFN_SetRootConstants = void(STDMETHODCALLTYPE*)(ID3D12GraphicsCommandList*, UINT, UINT, const void*, UINT);
using PFN_SetRootSignature = void(STDMETHODCALLTYPE*)(ID3D12GraphicsCommandList*, ID3D12RootSignature*);

template <typename T> inline T Orig(int v, Slot s) { return (T) g_orig[v][s]; }

uint16_t Clamp16(double v) { return (uint16_t) std::clamp(v, 0.0, 65535.0); }

inline bool Counted(Slot s)
{
    if (!Active())
        return false;
    g_calls[s].fetch_add(1, std::memory_order_relaxed);
    return true;
}

// Lists the game made before CreateCommandList was hooked (or by a path we do not see) have vtables we never
// registered: in Crimson Desert the list the upscaler evaluates on is one, and its Draw / Dispatch / RSSetViewports
// live at addresses of their own (D3D12 keeps a version of those per list type), while Close, OMSetRenderTargets and
// ExecuteIndirect are shared and reach us anyway. So every hooked call checks its list's vtable -- a thread-local
// compare while it is the same vtable as last time -- and registers a new one. Idle too: the first sightings happen
// at load, not while measuring.
void RegisterList(ID3D12GraphicsCommandList* list, const char* where);
constexpr int kMaxVtables = 1024;
std::atomic<void**> g_known[kMaxVtables] = {};
std::atomic<int> g_knownCount { 0 };
thread_local void** t_lastVtable = nullptr;

inline void Learn(void* list)
{
    void** vt = *(void***) list;
    if (vt == t_lastVtable)
        return;
    t_lastVtable = vt;
    const int n = std::min(g_knownCount.load(), kMaxVtables);
    for (int i = 0; i < n; ++i)
        if (g_known[i].load() == vt)
            return;
    RegisterList(static_cast<ID3D12GraphicsCommandList*>(list), "its first hooked call");
}

// The list's current viewport, for work that has no size of its own.
void ViewportOf(void* list, uint16_t& w, uint16_t& h)
{
    HookGuard g;
    if (!g.ok)
        return;
    if (Rec* r = Find(list, g.phase == Capturing); r != nullptr)
    {
        w = r->vpW;
        h = r->vpH;
    }
}

template <int V> void STDMETHODCALLTYPE hkDrawInstanced(ID3D12GraphicsCommandList* This, UINT a, UINT b, UINT c, UINT d)
{
    Learn(This);
    if (Counted(SDraw))
        NoteDraw(This, CatColour, 1);
    GiCapture_Dx12::OnWork(This, 0);
    Orig<PFN_DrawInstanced>(V, SDraw)(This, a, b, c, d);
}

template <int V>
void STDMETHODCALLTYPE hkDrawIndexedInstanced(ID3D12GraphicsCommandList* This, UINT a, UINT b, UINT c, INT d, UINT e)
{
    Learn(This);
    if (Counted(SDrawIndexed))
        NoteDraw(This, CatColour, 1);
    GiCapture_Dx12::OnWork(This, 0);
    Orig<PFN_DrawIndexedInstanced>(V, SDrawIndexed)(This, a, b, c, d, e);
}

template <int V> void STDMETHODCALLTYPE hkDispatch(ID3D12GraphicsCommandList* This, UINT x, UINT y, UINT z)
{
    Learn(This);
    if (Counted(SDispatch))
        Note(This, CatCompute, 0, 0, 0, 1);
    GiCapture_Dx12::OnWork(This, 1);
    Orig<PFN_Dispatch>(V, SDispatch)(This, x, y, z);
}

template <int V>
void STDMETHODCALLTYPE hkCopyBufferRegion(ID3D12GraphicsCommandList* This, ID3D12Resource* d, UINT64 dOff,
                                          ID3D12Resource* s, UINT64 sOff, UINT64 n)
{
    Learn(This);
    if (Counted(SCopyBuffer))
        Note(This, CatCopy, 0, 0, 0, 0);
    GiCapture_Dx12::OnCopyBuffer(This, d, dOff, s, sOff, n);
    GiCapture_Dx12::PatchCameraCopy(This, s, sOff, n, d, dOff);
    Orig<PFN_CopyBufferRegion>(V, SCopyBuffer)(This, d, dOff, s, sOff, n);
}

template <int V>
void STDMETHODCALLTYPE hkCopyTextureRegion(ID3D12GraphicsCommandList* This, const D3D12_TEXTURE_COPY_LOCATION* d,
                                           UINT x, UINT y, UINT z, const D3D12_TEXTURE_COPY_LOCATION* s,
                                           const D3D12_BOX* box)
{
    Learn(This);
    if (Counted(SCopyTexture))
        Note(This, CatCopy, 0, 0, 0, 0);
    if (d != nullptr && s != nullptr)
        GiCapture_Dx12::OnCopyTexture(This, d->pResource, s->pResource);
    Orig<PFN_CopyTextureRegion>(V, SCopyTexture)(This, d, x, y, z, s, box);
}

template <int V> void STDMETHODCALLTYPE hkCopyResource(ID3D12GraphicsCommandList* This, ID3D12Resource* d, ID3D12Resource* s)
{
    Learn(This);
    if (Counted(SCopyResource))
        Note(This, CatCopy, 0, 0, 0, 0);
    GiCapture_Dx12::OnCopyBuffer(This, d, 0, s, 0, 0);
    GiCapture_Dx12::OnCopyTexture(This, d, s);
    Orig<PFN_CopyResource>(V, SCopyResource)(This, d, s);
}

template <int V>
void STDMETHODCALLTYPE hkRSSetViewports(ID3D12GraphicsCommandList* This, UINT n, const D3D12_VIEWPORT* vp)
{
    Learn(This);
    if (Counted(SViewports) && n > 0 && vp != nullptr)
    {
        HookGuard g;
        if (g.ok)
            if (Rec* r = Find(This, g.phase == Capturing); r != nullptr)
            {
                r->vpW = Clamp16(vp[0].Width);
                r->vpH = Clamp16(vp[0].Height);
            }
    }
    Orig<PFN_RSSetViewports>(V, SViewports)(This, n, vp);
}

template <int V>
void STDMETHODCALLTYPE hkExecuteBundle(ID3D12GraphicsCommandList* This, ID3D12GraphicsCommandList* bundle)
{
    Learn(This);
    if (Counted(SBundle))
        Note(This, CatBundle, 0, 0, 1, 0);
    Orig<PFN_ExecuteBundle>(V, SBundle)(This, bundle);
}

template <int V>
void STDMETHODCALLTYPE hkOMSetRenderTargets(ID3D12GraphicsCommandList* This, UINT n,
                                            const D3D12_CPU_DESCRIPTOR_HANDLE* rtvs, BOOL single,
                                            const D3D12_CPU_DESCRIPTOR_HANDLE* dsv)
{
    Learn(This);
    if (Counted(STargets))
    {
        HookGuard g;
        if (g.ok)
            if (Rec* r = Find(This, g.phase == Capturing); r != nullptr)
                r->numRt = (uint16_t) n;
    }
    Orig<PFN_OMSetRenderTargets>(V, STargets)(This, n, rtvs, single, dsv);
}

template <int V>
void STDMETHODCALLTYPE hkExecuteIndirect(ID3D12GraphicsCommandList* This, ID3D12CommandSignature* sig, UINT maxCount,
                                         ID3D12Resource* args, UINT64 argsOff, ID3D12Resource* count, UINT64 countOff)
{
    Learn(This);
    if (Counted(SIndirect))
    {
        uint16_t w = 0, h = 0, numRt = 1;
        {
            HookGuard g;
            if (g.ok)
                if (Rec* r = Find(This, g.phase == Capturing); r != nullptr)
                {
                    w = r->vpW;
                    h = r->vpH;
                    numRt = r->numRt;
                }
        }
        uint16_t cat = CatIndirectUnknown;
        if (SignatureKnown(sig))
            switch (SignatureKind(sig))
            {
            case SigDraw:
                cat = numRt == 0 ? CatIndirectDepth : CatIndirectDraw;
                break;
            case SigDispatch:
                cat = CatIndirectDispatch;
                w = h = 0;
                break;
            case SigRays:
                cat = CatIndirectRays;
                break;
            case SigMesh:
                cat = CatIndirectMesh;
                break;
            default:
                break;
            }
        if (cat == CatIndirectDispatch || cat == CatIndirectRays)
            Note(This, cat, w, h, 0, 1);
        else
            Note(This, cat, w, h, 1, 0);
    }
    GiCapture_Dx12::OnWork(This, 3);
    Orig<PFN_ExecuteIndirect>(V, SIndirect)(This, sig, maxCount, args, argsOff, count, countOff);
}

template <int V>
void STDMETHODCALLTYPE hkBeginRenderPass(ID3D12GraphicsCommandList4* This, UINT n,
                                         const D3D12_RENDER_PASS_RENDER_TARGET_DESC* rts,
                                         const D3D12_RENDER_PASS_DEPTH_STENCIL_DESC* ds, D3D12_RENDER_PASS_FLAGS flags)
{
    Learn(This);
    if (Counted(SBeginRenderPass))
    {
        ++g_renderPasses;
        HookGuard g;
        if (g.ok)
            if (Rec* r = Find(This, g.phase == Capturing); r != nullptr && !r->skip && !r->closed.load())
            {
                r->numRt = (uint16_t) n;
                // Stamped outside the pass, before it begins; the pass is one segment.
                if (t_oursDepth == 0 && !r->overflow)
                    StartSeg(This, r, CatRenderPass, r->vpW, r->vpH);
                r->inRenderPass = true;
            }
    }
    Orig<PFN_BeginRenderPass>(V, SBeginRenderPass)(This, n, rts, ds, flags);
}

template <int V> void STDMETHODCALLTYPE hkEndRenderPass(ID3D12GraphicsCommandList4* This)
{
    Learn(This);
    Orig<PFN_EndRenderPass>(V, SEndRenderPass)(This);
    if (Counted(SEndRenderPass))
    {
        HookGuard g;
        if (g.ok)
            if (Rec* r = Find(This, false); r != nullptr)
                r->inRenderPass = false;
    }
}

template <int V>
void STDMETHODCALLTYPE hkBuildRtas(ID3D12GraphicsCommandList4* This,
                                   const D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC* desc, UINT n,
                                   const D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC* post)
{
    Learn(This);
    if (Counted(SBuildRtas))
        Note(This, CatAccelBuild, 0, 0, 0, 1);
    Orig<PFN_BuildRtas>(V, SBuildRtas)(This, desc, n, post);
}

template <int V> void STDMETHODCALLTYPE hkDispatchRays(ID3D12GraphicsCommandList4* This, const D3D12_DISPATCH_RAYS_DESC* desc)
{
    Learn(This);
    if (Counted(SRays))
        Note(This, CatRays, desc ? Clamp16(desc->Width) : 0, desc ? Clamp16(desc->Height) : 0, 0, 1);
    GiCapture_Dx12::OnWork(This, 2);
    Orig<PFN_DispatchRays>(V, SRays)(This, desc);
}

template <int V> void STDMETHODCALLTYPE hkDispatchMesh(ID3D12GraphicsCommandList6* This, UINT x, UINT y, UINT z)
{
    Learn(This);
    if (Counted(SMesh))
    {
        uint16_t w = 0, h = 0;
        ViewportOf(This, w, h);
        Note(This, CatMesh, w, h, 1, 0);
    }
    GiCapture_Dx12::OnWork(This, 0);
    Orig<PFN_DispatchMesh>(V, SMesh)(This, x, y, z);
}

// Not measured: the hooks GiCapture_Dx12 needs, here because this module already reaches every list vtable.
template <int V>
void STDMETHODCALLTYPE hkResourceBarrier(ID3D12GraphicsCommandList* This, UINT count, const D3D12_RESOURCE_BARRIER* barriers)
{
    Learn(This);
    Orig<PFN_ResourceBarrier>(V, SBarrier)(This, count, barriers);
    GiCapture_Dx12::OnBarrier(This, count, barriers);
}

template <int V>
void STDMETHODCALLTYPE hkBarrier(ID3D12GraphicsCommandList7* This, UINT32 count, const D3D12_BARRIER_GROUP* groups)
{
    Learn(This);
    GiCapture_Dx12::OnEnhancedBarrier(This, count, groups);
    Orig<PFN_Barrier>(V, SEnhancedBarrier)(This, count, groups);
}

template <int V> void STDMETHODCALLTYPE hkSetPipelineState(ID3D12GraphicsCommandList* This, ID3D12PipelineState* pipeline)
{
    Learn(This);
    GiCapture_Dx12::OnPipeline(This, pipeline);
    Orig<PFN_SetPipelineState>(V, SPipeline)(This, GiCapture_Dx12::SwapPipeline(pipeline));
}

template <int V> void STDMETHODCALLTYPE hkSetPipelineState1(ID3D12GraphicsCommandList4* This, ID3D12StateObject* state)
{
    Learn(This);
    GiCapture_Dx12::OnPipeline(This, state);
    Orig<PFN_SetPipelineState1>(V, SPipeline1)(This, state);
}

template <int V, Slot S, bool Graphics, bool Srv>
void STDMETHODCALLTYPE hkSetRootView(ID3D12GraphicsCommandList* This, UINT index, D3D12_GPU_VIRTUAL_ADDRESS address)
{
    Learn(This);
    GiCapture_Dx12::OnRootView(This, Graphics, Srv, index, address);
    Orig<PFN_SetRootView>(V, S)(This, index, address);
}

template <int V>
void STDMETHODCALLTYPE hkSetDescriptorHeaps(ID3D12GraphicsCommandList* This, UINT count, ID3D12DescriptorHeap* const* heaps)
{
    Learn(This);
    GiCapture_Dx12::OnSetHeaps(This, count, heaps);
    Orig<PFN_SetDescriptorHeaps>(V, SHeaps)(This, count, heaps);
}

template <int V, Slot S, bool Graphics>
void STDMETHODCALLTYPE hkSetRootTable(ID3D12GraphicsCommandList* This, UINT index, D3D12_GPU_DESCRIPTOR_HANDLE table)
{
    Learn(This);
    GiCapture_Dx12::OnRootTable(This, Graphics, index, table.ptr);
    Orig<PFN_SetRootTable>(V, S)(This, index, table);
}

template <int V, Slot S, bool Graphics>
void STDMETHODCALLTYPE hkSetRootConstant(ID3D12GraphicsCommandList* This, UINT index, UINT value, UINT offset)
{
    Learn(This);
    GiCapture_Dx12::OnRootConstants(This, Graphics, index, 1, &value, offset);
    Orig<PFN_SetRootConstant>(V, S)(This, index, value, offset);
}

template <int V, Slot S, bool Graphics>
void STDMETHODCALLTYPE hkSetRootConstants(ID3D12GraphicsCommandList* This, UINT index, UINT count, const void* values, UINT offset)
{
    Learn(This);
    GiCapture_Dx12::OnRootConstants(This, Graphics, index, count, values, offset);
    Orig<PFN_SetRootConstants>(V, S)(This, index, count, values, offset);
}

template <int V, Slot S, bool Graphics>
void STDMETHODCALLTYPE hkSetRootSignature(ID3D12GraphicsCommandList* This, ID3D12RootSignature* signature)
{
    Learn(This);
    GiCapture_Dx12::OnSetRootSignature(This, Graphics, signature);
    Orig<PFN_SetRootSignature>(V, S)(This, signature);
}

// The end timestamp and the resolve of every chunk this list used, recorded into the list itself (copy-like work).
template <int V> HRESULT STDMETHODCALLTYPE hkClose(ID3D12GraphicsCommandList* This)
{
    Learn(This);
    if (Counted(SClose))
    {
        HookGuard g;
        if (g.ok)
            if (Rec* r = Find(This, false); r != nullptr && !r->closed.load())
            {
                if (!r->skip && !r->overflow && !r->segs.empty())
                {
                    uint32_t q = kNoQuery;
                    if (Stamp(This, r, q))
                        r->endQ = q;
                    if (r->chunkStart != kNoQuery && r->chunkUsed > 0)
                        r->chunks.emplace_back(r->chunkStart, r->chunkUsed);
                    r->chunkStart = kNoQuery;
                    if (!r->overflow)
                        for (const auto& [start, used] : r->chunks)
                            This->ResolveQueryData(g_heap, D3D12_QUERY_TYPE_TIMESTAMP, start, used, g_readback,
                                                   (UINT64) start * sizeof(uint64_t));
                }
                r->present = g_present.load();
                r->closed.store(true);
                std::lock_guard lock(g_mutex);
                g_open.erase(This);
            }
    }
    return Orig<PFN_Close>(V, SClose)(This);
}

template <int V> void* HookOf(Slot s)
{
    switch (s)
    {
    case SClose:
        return (void*) &hkClose<V>;
    case SDraw:
        return (void*) &hkDrawInstanced<V>;
    case SDrawIndexed:
        return (void*) &hkDrawIndexedInstanced<V>;
    case SDispatch:
        return (void*) &hkDispatch<V>;
    case SCopyBuffer:
        return (void*) &hkCopyBufferRegion<V>;
    case SCopyTexture:
        return (void*) &hkCopyTextureRegion<V>;
    case SCopyResource:
        return (void*) &hkCopyResource<V>;
    case SViewports:
        return (void*) &hkRSSetViewports<V>;
    case SBundle:
        return (void*) &hkExecuteBundle<V>;
    case STargets:
        return (void*) &hkOMSetRenderTargets<V>;
    case SIndirect:
        return (void*) &hkExecuteIndirect<V>;
    case SBeginRenderPass:
        return (void*) &hkBeginRenderPass<V>;
    case SEndRenderPass:
        return (void*) &hkEndRenderPass<V>;
    case SBuildRtas:
        return (void*) &hkBuildRtas<V>;
    case SRays:
        return (void*) &hkDispatchRays<V>;
    case SMesh:
        return (void*) &hkDispatchMesh<V>;
    case SBarrier:
        return (void*) &hkResourceBarrier<V>;
    case SEnhancedBarrier:
        return (void*) &hkBarrier<V>;
    case SPipeline:
        return (void*) &hkSetPipelineState<V>;
    case SPipeline1:
        return (void*) &hkSetPipelineState1<V>;
    case SComputeCbv:
        return (void*) &hkSetRootView<V, SComputeCbv, false, false>;
    case SGraphicsCbv:
        return (void*) &hkSetRootView<V, SGraphicsCbv, true, false>;
    case SComputeSrv:
        return (void*) &hkSetRootView<V, SComputeSrv, false, true>;
    case SGraphicsSrv:
        return (void*) &hkSetRootView<V, SGraphicsSrv, true, true>;
    case SHeaps:
        return (void*) &hkSetDescriptorHeaps<V>;
    case SComputeTable:
        return (void*) &hkSetRootTable<V, SComputeTable, false>;
    case SGraphicsTable:
        return (void*) &hkSetRootTable<V, SGraphicsTable, true>;
    case SComputeConstant:
        return (void*) &hkSetRootConstant<V, SComputeConstant, false>;
    case SGraphicsConstant:
        return (void*) &hkSetRootConstant<V, SGraphicsConstant, true>;
    case SComputeConstants:
        return (void*) &hkSetRootConstants<V, SComputeConstants, false>;
    case SGraphicsConstants:
        return (void*) &hkSetRootConstants<V, SGraphicsConstants, true>;
    case SComputeSignature:
        return (void*) &hkSetRootSignature<V, SComputeSignature, false>;
    case SGraphicsSignature:
        return (void*) &hkSetRootSignature<V, SGraphicsSignature, true>;
    default:
        return nullptr;
    }
}

void* HookFor(int v, Slot s)
{
    switch (v)
    {
    case 0:
        return HookOf<0>(s);
    case 1:
        return HookOf<1>(s);
    case 2:
        return HookOf<2>(s);
    case 3:
        return HookOf<3>(s);
    case 4:
        return HookOf<4>(s);
    case 5:
        return HookOf<5>(s);
    case 6:
        return HookOf<6>(s);
    case 7:
        return HookOf<7>(s);
    default:
        return nullptr;
    }
}

std::mutex g_hookMutex;
std::vector<void**> g_vtables;   // every command-list vtable registered, ours first
std::set<void*> g_hookedCode;    // method addresses hooked (the code, not the vtable slot)
int g_variantsUsed = 0;
bool g_loggedVariantsFull = false;

const char* TypeName(D3D12_COMMAND_LIST_TYPE t)
{
    switch (t)
    {
    case D3D12_COMMAND_LIST_TYPE_DIRECT:
        return "direct";
    case D3D12_COMMAND_LIST_TYPE_COMPUTE:
        return "compute";
    case D3D12_COMMAND_LIST_TYPE_COPY:
        return "copy";
    case D3D12_COMMAND_LIST_TYPE_BUNDLE:
        return "bundle";
    default:
        return "other";
    }
}

// Hooks every method of this list's vtable whose code is not hooked yet. `list` is a real (not proxied) list.
void RegisterList(ID3D12GraphicsCommandList* list, const char* where)
{
    if (list == nullptr)
        return;
    void** vt = *(void***) list;
    std::lock_guard lock(g_hookMutex);
    if (std::find(g_vtables.begin(), g_vtables.end(), vt) != g_vtables.end())
        return;
    const size_t index = g_vtables.size();
    g_vtables.push_back(vt);
    if (const int n = g_knownCount.load(); n < kMaxVtables)
    {
        g_known[n].store(vt);
        g_knownCount.store(n + 1);
    }

    void* code[SCount] = {};
    for (int s = 0; s < SCount; ++s)
        if (kSlotIndex[s] < 60)
            code[s] = vt[kSlotIndex[s]];
    ID3D12GraphicsCommandList4* list4 = nullptr;
    if (SUCCEEDED(list->QueryInterface(IID_PPV_ARGS(&list4))) && list4 != nullptr)
    {
        void** vt4 = *(void***) list4;
        for (Slot s : { SBeginRenderPass, SEndRenderPass, SBuildRtas, SRays, SPipeline1 })
            code[s] = vt4[kSlotIndex[s]];
        list4->Release();
    }
    ID3D12GraphicsCommandList6* list6 = nullptr;
    if (SUCCEEDED(list->QueryInterface(IID_PPV_ARGS(&list6))) && list6 != nullptr)
    {
        code[SMesh] = (*(void***) list6)[kSlotIndex[SMesh]];
        list6->Release();
    }
    ID3D12GraphicsCommandList7* list7 = nullptr;
    if (SUCCEEDED(list->QueryInterface(IID_PPV_ARGS(&list7))) && list7 != nullptr)
    {
        code[SEnhancedBarrier] = (*(void***) list7)[kSlotIndex[SEnhancedBarrier]];
        list7->Release();
    }
    if (!GiCapture_Dx12::Enabled())
        for (Slot s : { SPipeline, SPipeline1, SComputeCbv, SGraphicsCbv, SComputeSrv, SGraphicsSrv, SHeaps, SComputeTable, SGraphicsTable,
                        SComputeConstant, SGraphicsConstant, SComputeConstants, SGraphicsConstants, SComputeSignature, SGraphicsSignature })
            code[s] = nullptr;

    std::string fresh, shared;
    int wanted = 0;
    for (int s = 0; s < SCount; ++s)
        if (code[s] != nullptr && !g_hookedCode.contains(code[s]))
            ++wanted;
    const char* type = TypeName(list->GetType());
    if (wanted == 0)
    {
        LOG_INFO("Frame split: command-list vtable #{} ({} list, from {}): every method already hooked", index, type,
                 where);
        return;
    }
    if (g_variantsUsed >= kVariants)
    {
        if (!g_loggedVariantsFull)
            LOG_ERROR("Frame split: command-list vtable #{} ({} list, from {}) has {} methods not hooked, and all {} hook "
                      "variants are used; its calls are NOT measured",
                      index, type, where, wanted, kVariants);
        g_loggedVariantsFull = true;
        return;
    }
    const int v = g_variantsUsed;
    DetourTransactionBegin();
    DetourUpdateThread(GetCurrentThread());
    std::vector<int> attached;
    for (int s = 0; s < SCount; ++s)
    {
        if (code[s] == nullptr || g_hookedCode.contains(code[s]))
        {
            if (code[s] != nullptr)
                shared += std::string(shared.empty() ? "" : ", ") + kSlotName[s];
            continue;
        }
        g_orig[v][s] = code[s];
        if (DetourAttach(&g_orig[v][s], HookFor(v, (Slot) s)) == NO_ERROR)
            attached.push_back(s);
        else
            g_orig[v][s] = nullptr;
    }
    const LONG result = DetourTransactionCommit();
    if (result != NO_ERROR)
    {
        for (int s : attached)
            g_orig[v][s] = nullptr;
        LOG_ERROR("Frame split: hooking command-list vtable #{} ({} list, from {}) failed ({:X}); its calls are NOT "
                  "measured",
                  index, type, where, result);
        return;
    }
    for (int s : attached)
    {
        g_hookedCode.insert(code[s]);
        fresh += std::string(fresh.empty() ? "" : ", ") + kSlotName[s];
    }
    ++g_variantsUsed;
    LOG_INFO("Frame split: command-list vtable #{} ({} list, from {}): hooked {} [{}]; already hooked through an "
             "earlier vtable: [{}]",
             index, type, where, attached.size(), fresh, shared);
}

using PFN_CreateCommandList = HRESULT(STDMETHODCALLTYPE*)(ID3D12Device*, UINT, D3D12_COMMAND_LIST_TYPE,
                                                          ID3D12CommandAllocator*, ID3D12PipelineState*, REFIID, void**);
using PFN_CreateCommandList1 = HRESULT(STDMETHODCALLTYPE*)(ID3D12Device4*, UINT, D3D12_COMMAND_LIST_TYPE,
                                                           D3D12_COMMAND_LIST_FLAGS, REFIID, void**);
PFN_CreateCommandList o_CreateCommandList = nullptr;
PFN_CreateCommandList1 o_CreateCommandList1 = nullptr;

void RegisterCreated(void** pp, const char* where)
{
    if (pp == nullptr || *pp == nullptr)
        return;
    ID3D12GraphicsCommandList* list = nullptr;
    if (SUCCEEDED(((IUnknown*) *pp)->QueryInterface(IID_PPV_ARGS(&list))) && list != nullptr)
    {
        RegisterList(list, where);
        list->Release();
    }
}

HRESULT STDMETHODCALLTYPE hkCreateCommandList(ID3D12Device* This, UINT mask, D3D12_COMMAND_LIST_TYPE type,
                                              ID3D12CommandAllocator* allocator, ID3D12PipelineState* pso, REFIID riid,
                                              void** pp)
{
    const HRESULT hr = o_CreateCommandList(This, mask, type, allocator, pso, riid, pp);
    if (SUCCEEDED(hr))
        RegisterCreated(pp, "CreateCommandList");
    return hr;
}

HRESULT STDMETHODCALLTYPE hkCreateCommandList1(ID3D12Device4* This, UINT mask, D3D12_COMMAND_LIST_TYPE type,
                                               D3D12_COMMAND_LIST_FLAGS flags, REFIID riid, void** pp)
{
    const HRESULT hr = o_CreateCommandList1(This, mask, type, flags, riid, pp);
    if (SUCCEEDED(hr))
        RegisterCreated(pp, "CreateCommandList1");
    return hr;
}

bool EnsureSetup()
{
    if (g_heap != nullptr)
        return true;
    if (g_setupTried || g_device == nullptr)
        return false;
    g_setupTried = true;
    D3D12_QUERY_HEAP_DESC qd {};
    qd.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
    qd.Count = kMaxQueries;
    ID3D12QueryHeap* heap = nullptr;
    if (FAILED(g_device->CreateQueryHeap(&qd, IID_PPV_ARGS(&heap))))
    {
        LOG_ERROR("Frame split: no timestamp query heap; nothing can be measured");
        return false;
    }
    D3D12_RESOURCE_DESC buf {};
    buf.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    buf.Width = (UINT64) kMaxQueries * sizeof(uint64_t);
    buf.Height = 1;
    buf.DepthOrArraySize = 1;
    buf.MipLevels = 1;
    buf.SampleDesc.Count = 1;
    buf.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    D3D12_HEAP_PROPERTIES hp {};
    hp.Type = D3D12_HEAP_TYPE_READBACK;
    ID3D12Resource* readback = nullptr;
    if (FAILED(g_device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &buf, D3D12_RESOURCE_STATE_COPY_DEST,
                                                 nullptr, IID_PPV_ARGS(&readback))))
    {
        heap->Release();
        LOG_ERROR("Frame split: no readback buffer; nothing can be measured");
        return false;
    }
    void* mapped = nullptr;
    if (FAILED(readback->Map(0, nullptr, &mapped)) || mapped == nullptr)
    {
        readback->Release();
        heap->Release();
        LOG_ERROR("Frame split: the readback buffer did not map; nothing can be measured");
        return false;
    }
    g_heap = heap;
    g_readback = readback;
    g_data = (uint64_t*) mapped;
    return true;
}

struct Agg
{
    double ms = 0;
    uint64_t segs = 0;
    uint64_t draws = 0;
    uint64_t dispatches = 0;
};

struct Line
{
    uint64_t t0 = 0;
    double ms = 0;
    bool compute = false;
    uint64_t present = 0;
    Seg seg;
};

std::string SizeText(const Seg& s)
{
    if (s.w == 0 && s.h == 0)
        return "";
    return std::format(" {}x{}", s.w, s.h);
}

// Under g_mutex, with no hook inside (phase Reading, g_inHooks drained).
void Report()
{
    std::vector<std::string> lines;
    if (g_frequency == 0)
    {
        LOG_ERROR("Frame split: no timestamp frequency from the present queue; nothing read");
        g_lines = { "No timestamp frequency; nothing read." };
        return;
    }
    const double toMs = 1000.0 / (double) g_frequency;
    const uint64_t first = g_captureStart + kEdgePresents;              // exclusive
    const uint64_t last = g_captureStart + kCapturePresents - kEdgePresents; // inclusive
    const double presents = (double) (last - first);
    const uint64_t timelinePresent = g_captureStart + kCapturePresents / 2;

    std::map<std::tuple<int, uint16_t, uint16_t, uint16_t>, Agg> bySize; // queue, cat, w, h
    Agg byKind[2][CatCount];
    double queueMs[2] = { 0, 0 };
    std::set<uint64_t> presentsWithOurs;
    std::vector<Line> timeline;
    uint32_t read = 0, notRun = 0, open = 0, overflow = 0, outside = 0, empty = 0, skipped = 0;

    for (const auto& rp : g_recs)
    {
        const Rec& r = *rp;
        if (r.skip)
        {
            ++skipped;
            continue;
        }
        if (!r.closed.load())
        {
            ++open;
            continue;
        }
        if (r.overflow || (r.endQ == kNoQuery && !r.segs.empty()))
        {
            ++overflow;
            continue;
        }
        if (r.segs.empty())
        {
            ++empty;
            continue;
        }
        if (r.present <= first || r.present > last)
        {
            ++outside;
            continue;
        }
        std::vector<uint64_t> t(r.segs.size() + 1);
        bool ok = true;
        for (size_t i = 0; i < r.segs.size(); ++i)
            t[i] = g_data[r.segs[i].q];
        t.back() = g_data[r.endQ];
        for (size_t i = 0; i < t.size() && ok; ++i)
            ok = t[i] != 0 && (i == 0 || t[i] >= t[i - 1]);
        if (!ok)
        {
            ++notRun;
            continue;
        }
        ++read;
        const int queue = r.compute ? 1 : 0;
        for (size_t i = 0; i < r.segs.size(); ++i)
        {
            const Seg& s = r.segs[i];
            const double ms = (double) (t[i + 1] - t[i]) * toMs;
            auto& a = bySize[{ queue, s.cat, s.w, s.h }];
            a.ms += ms;
            ++a.segs;
            a.draws += s.draws;
            a.dispatches += s.dispatches;
            auto& k = byKind[queue][s.cat];
            k.ms += ms;
            ++k.segs;
            k.draws += s.draws;
            k.dispatches += s.dispatches;
            queueMs[queue] += ms;
            if (s.cat == CatOurs)
                presentsWithOurs.insert(r.present);
            if (r.present == timelinePresent || r.present == timelinePresent + 1)
                timeline.push_back({ t[i], ms, r.compute, r.present, s });
        }
    }

    const double rate = g_captureSeconds > 0 ? (double) kCapturePresents / g_captureSeconds : 0.0;
    LOG_INFO("Frame split: {} Presents measured ({} of them with an upscaler evaluate), {:.1f} Presents/s during the "
             "capture ({:.2f} ms per Present). GPU work per Present: direct queue {:.2f} ms, async compute queue {:.2f} "
             "ms (runs beside the direct queue). {} command lists read; left out: {} not executed or incomplete, {} "
             "still open at the end, {} overflowed, {} outside the window, {} with no work, {} on queues that cannot "
             "be timed ({} copy-queue lists, {} bundles, {} other). {} render passes seen, {} upscaler evaluates.",
             (int) presents, presentsWithOurs.size(), rate, rate > 0 ? 1000.0 / rate : 0.0, queueMs[0] / presents,
             queueMs[1] / presents, read, notRun, open, overflow, outside, empty, skipped, g_copyLists.load(),
             g_bundleLists.load(), g_otherLists.load(), g_renderPasses.load(), g_evaluates.load());
    {
        std::string calls;
        for (int c = 0; c < SCount; ++c)
            calls += std::format("{}{} {}", c ? ", " : "", kSlotName[c], g_calls[c].load());
        calls += std::format("; command signatures recorded since startup: {}", g_sigsCreated.load());
        LOG_INFO("Frame split: calls seen during the capture and drain (a 0 is a blind spot unless the game never "
                 "calls it): {}",
                 calls);
    }
    LOG_INFO("Frame split: a segment runs from its first call to the next segment's first call in the same list, so "
             "barriers and clears count toward the segment before them, and overlapping work is split at the "
             "timestamps.");
    lines.push_back(std::format("{} Presents, {:.1f}/s: direct {:.2f} ms, async compute {:.2f} ms per Present",
                                (int) presents, rate, queueMs[0] / presents, queueMs[1] / presents));

    for (int queue = 0; queue < 2; ++queue)
    {
        if (queueMs[queue] <= 0)
            continue;
        std::vector<int> order;
        for (int c = 0; c < CatCount; ++c)
            if (byKind[queue][c].segs > 0)
                order.push_back(c);
        std::sort(order.begin(), order.end(),
                  [&](int a, int b) { return byKind[queue][a].ms > byKind[queue][b].ms; });
        LOG_INFO("Frame split, {} queue, by kind (per Present):", queue ? "async compute" : "direct");
        for (int c : order)
        {
            const auto& k = byKind[queue][c];
            LOG_INFO("  {:5.1f}%  {:6.3f} ms  {}: {:.1f} segments, {:.0f} draws, {:.0f} dispatches",
                     100.0 * k.ms / queueMs[queue], k.ms / presents, kCatName[c], k.segs / presents,
                     k.draws / presents, k.dispatches / presents);
            if (queue == 0)
                lines.push_back(std::format("{:4.1f}% {:5.2f} ms  {}", 100.0 * k.ms / queueMs[queue], k.ms / presents,
                                            kCatName[c]));
        }
    }

    std::vector<std::pair<std::tuple<int, uint16_t, uint16_t, uint16_t>, Agg>> sizes(bySize.begin(), bySize.end());
    std::sort(sizes.begin(), sizes.end(), [](const auto& a, const auto& b) { return a.second.ms > b.second.ms; });
    LOG_INFO("Frame split, by kind and size (viewport for raster, ray grid for DispatchRays), top 40 per Present:");
    for (size_t i = 0; i < sizes.size() && i < 40; ++i)
    {
        const Agg a = sizes[i].second;
        const int queue = std::get<0>(sizes[i].first);
        Seg s;
        s.cat = std::get<1>(sizes[i].first);
        s.w = std::get<2>(sizes[i].first);
        s.h = std::get<3>(sizes[i].first);
        LOG_INFO("  {:6.3f} ms  {}{}{}: {:.1f} segments, {:.0f} draws, {:.0f} dispatches", a.ms / presents,
                 kCatName[s.cat], SizeText(s), queue ? " [async compute]" : "", a.segs / presents,
                 a.draws / presents, a.dispatches / presents);
    }

    std::sort(timeline.begin(), timeline.end(), [](const Line& a, const Line& b) { return a.t0 < b.t0; });
    if (!timeline.empty())
    {
        size_t shown = 0, hidden = 0;
        double hiddenMs = 0;
        LOG_INFO("Frame split, timeline of Presents {} and {} (two eyes), segments of 0.03 ms or more, by GPU start "
                 "time:",
                 timelinePresent, timelinePresent + 1);
        const uint64_t t0 = timeline.front().t0;
        for (const auto& l : timeline)
        {
            if (l.ms < 0.03 || shown >= 200)
            {
                ++hidden;
                hiddenMs += l.ms;
                continue;
            }
            ++shown;
            LOG_INFO("  +{:7.3f} ms  {:6.3f} ms  P{}  {}{}{}  ({} draws, {} dispatches)",
                     (double) (l.t0 - t0) * toMs, l.ms, l.present - g_captureStart, kCatName[l.seg.cat],
                     SizeText(l.seg), l.compute ? " [async compute]" : "", l.seg.draws, l.seg.dispatches);
        }
        LOG_INFO("  ... and {} smaller segments, {:.3f} ms together", hidden, hiddenMs);
    }
    else
    {
        LOG_INFO("Frame split: no timeline -- nothing was read for Presents {} and {}", timelinePresent,
                 timelinePresent + 1);
    }
    g_lines = std::move(lines);
}

void FinishCapture()
{
    // No hook may be inside a Rec from here on: the phase is Reading, and every hook re-checks it after counting in.
    while (g_inHooks.load() != 0)
        std::this_thread::yield();
    std::lock_guard lock(g_mutex);
    Report();
    g_open.clear();
    g_recs.clear();
    g_gen.fetch_add(1);
}
} // namespace

namespace FrameSplit_Dx12
{
bool Enabled()
{
    static const bool enabled = Config::Instance()->FrameSplitProbe.value_or_default() || Config::Instance()->PerEyeLighting.value_or_default();
    return enabled;
}

void Install(ID3D12Device* device)
{
    static std::mutex installMutex;
    std::lock_guard lock(installMutex);
    if (!Enabled() || g_installed || device == nullptr)
        return;
    g_device = device;

    ID3D12Device* realDevice = nullptr;
    if (!Util::CheckForRealObject(__FUNCTION__, device, (IUnknown**) &realDevice))
        realDevice = device;

    // The game's lists: every one created from here on registers its vtable.
    void** dvt = *(void***) realDevice;
    o_CreateCommandList = (PFN_CreateCommandList) dvt[12];
    ID3D12Device4* device4 = nullptr;
    if (SUCCEEDED(realDevice->QueryInterface(IID_PPV_ARGS(&device4))) && device4 != nullptr)
    {
        o_CreateCommandList1 = (PFN_CreateCommandList1) (*(void***) device4)[54];
        device4->Release();
    }
    // Test switch: with this hook always on, Crimson Desert crashed ~3 s after start, before any gameplay.
    if (Config::Instance()->FrameSplitSignatures.value_or_default())
        o_CreateCommandSignature = (PFN_CreateCommandSignature) dvt[43];
    if (GiCapture_Dx12::Enabled())
    {
        o_CreateConstantBufferView = (PFN_CreateConstantBufferView) dvt[17];
        o_CreateShaderResourceView = (PFN_CreateShaderResourceView) dvt[18];
        o_CreateUnorderedAccessView = (PFN_CreateUnorderedAccessView) dvt[19];
        o_CreateRootSignature = (PFN_CreateRootSignature) dvt[16];
        o_CopyDescriptors = (PFN_CopyDescriptors) dvt[23];
        o_CopyDescriptorsSimple = (PFN_CopyDescriptorsSimple) dvt[24];
    }
    DetourTransactionBegin();
    DetourUpdateThread(GetCurrentThread());
    if (o_CreateCommandSignature != nullptr)
        DetourAttach(&(PVOID&) o_CreateCommandSignature, hkCreateCommandSignature);
    if (o_CreateConstantBufferView != nullptr)
    {
        DetourAttach(&(PVOID&) o_CreateConstantBufferView, hkCreateConstantBufferView);
        DetourAttach(&(PVOID&) o_CreateShaderResourceView, hkCreateShaderResourceView);
        DetourAttach(&(PVOID&) o_CreateUnorderedAccessView, hkCreateUnorderedAccessView);
        DetourAttach(&(PVOID&) o_CreateRootSignature, hkCreateRootSignature);
        DetourAttach(&(PVOID&) o_CopyDescriptors, hkCopyDescriptors);
        DetourAttach(&(PVOID&) o_CopyDescriptorsSimple, hkCopyDescriptorsSimple);
    }
    if (o_CreateCommandList != nullptr)
        DetourAttach(&(PVOID&) o_CreateCommandList, hkCreateCommandList);
    if (o_CreateCommandList1 != nullptr)
        DetourAttach(&(PVOID&) o_CreateCommandList1, hkCreateCommandList1);
    if (const LONG result = DetourTransactionCommit(); result != NO_ERROR)
    {
        LOG_ERROR("Frame split: hooking CreateCommandList failed ({:X}); the button will do nothing", result);
        return;
    }
    g_installed = true;
    LOG_INFO("Frame split: CreateCommandSignature {}, CreateCommandList{} hooked; each command-list vtable is registered at creation. Idle until "
             "the overlay's Measure button",
             o_CreateCommandSignature ? "hooked" : "NOT hooked (FrameSplitSignatures off: every ExecuteIndirect "
                                                   "reads 'signature never seen created')",
             o_CreateCommandList1 ? " and CreateCommandList1" : "");

    // Ours: the first vtable (through the hook just installed).
    ID3D12CommandAllocator* allocator = nullptr;
    ID3D12GraphicsCommandList* list = nullptr;
    if (SUCCEEDED(realDevice->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator))))
    {
        if (SUCCEEDED(realDevice->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator, nullptr,
                                                    IID_PPV_ARGS(&list))))
        {
            list->Close();
            list->Release();
        }
        allocator->Release();
    }
}

void OnPresent(ID3D12CommandQueue* queue)
{
    if (!Enabled())
        return;
    const uint64_t present = ++g_present;
    if (g_frequency == 0 && queue != nullptr)
    {
        UINT64 frequency = 0;
        if (SUCCEEDED(queue->GetTimestampFrequency(&frequency)))
            g_frequency = frequency;
    }
    const auto now = std::chrono::steady_clock::now();

    if (g_armRequested.exchange(false) && g_phase.load() == Idle)
    {
        if (!g_installed)
        {
            if (!g_loggedNoInstall)
                LOG_ERROR("Frame split: Measure pressed, but the hooks were never installed (no D3D12 device seen?)");
            g_loggedNoInstall = true;
        }
        else if (!EnsureSetup())
        {
            if (!g_loggedSetupFail)
                LOG_ERROR("Frame split: Measure pressed, but the query heap or readback could not be made");
            g_loggedSetupFail = true;
        }
        else
        {
            std::memset(g_data, 0, (size_t) kMaxQueries * sizeof(uint64_t));
            g_nextQuery.store(0);
            g_copyLists = 0;
            g_bundleLists = 0;
            g_otherLists = 0;
            g_overflowLists = 0;
            g_renderPasses = 0;
            g_evaluates = 0;
            for (auto& c : g_calls)
                c.store(0);
            {
                std::lock_guard lock(g_mutex);
                g_lines.clear();
            }
            g_phaseStartTime = now;
            g_phase.store(Waiting);
            LOG_INFO("Frame split: armed; measuring {} Presents in {:.0f} s", kCapturePresents, kDelaySeconds);
        }
    }

    switch (g_phase.load())
    {
    case Waiting:
        if (std::chrono::duration<double>(now - g_phaseStartTime).count() >= kDelaySeconds)
        {
            g_captureStart = present;
            g_captureStartTime = now;
            g_phase.store(Capturing);
            LOG_INFO("Frame split: capturing from Present {}", present);
        }
        break;
    case Capturing:
        if (present - g_captureStart >= kCapturePresents)
        {
            g_captureSeconds = std::chrono::duration<double>(now - g_captureStartTime).count();
            g_phaseStartPresent = present;
            g_phase.store(Draining);
        }
        break;
    case Draining:
        if (present - g_phaseStartPresent >= kDrainPresents)
        {
            g_phase.store(Reading);
            FinishCapture();
            g_phase.store(Idle);
        }
        break;
    default:
        break;
    }
}

// The list the evaluate's work lands in, as the hooks know it: the handed list itself (or Streamline's real list)
// when the hooks have seen it, else the list this thread last recorded on. A wrong guess puts the timestamps out of
// order in that list, which the report counts as "not executed or incomplete" -- never as a silent mislabel.
void* OursTarget(ID3D12GraphicsCommandList* cmdList, bool& fallback)
{
    static const IID streamlineIid = []
    {
        IID iid {};
        IIDFromString(L"{ADEC44E2-61F0-45C3-AD9F-1B37379284FF}", &iid);
        return iid;
    }();
    void* list = cmdList;
    IUnknown* real = nullptr;
    if (SUCCEEDED(cmdList->QueryInterface(streamlineIid, (void**) &real)) && real != nullptr)
    {
        real->Release();
        list = real;
    }
    fallback = false;
    {
        std::lock_guard lock(g_mutex);
        if (g_open.contains(list))
            return list;
    }
    fallback = true;
    if (t_cache.gen == g_gen.load() && t_cache.list != nullptr)
        return t_cache.list;
    return nullptr;
}

void BeginOurs(ID3D12GraphicsCommandList* cmdList)
{
    if (cmdList == nullptr || ++t_oursDepth != 1 || !Active())
        return;
    HookGuard g;
    if (!g.ok)
        return;
    ++g_evaluates;
    bool fallback = false;
    void* target = OursTarget(cmdList, fallback);
    {
        std::lock_guard lock(g_mutex);
        if (!g_loggedProxy)
        {
            g_loggedProxy = true;
            LOG_INFO("Frame split: the evaluate's list {} a list the hooks have seen{}", fallback ? "is NOT" : "is",
                     fallback ? (target != nullptr ? "; its time is booked to this thread's last recorded list, "
                                                     "stamped through the handed list"
                                                   : "; and this thread recorded nothing yet -- not timed")
                              : "");
        }
    }
    if (target == nullptr)
        return;
    Rec* r = Find(target, g.phase == Capturing);
    if (r == nullptr || r->skip || r->closed.load() || r->overflow || r->inRenderPass)
        return;
    StartSeg(target, r, CatOurs, 0, 0, cmdList);
}

void EndOurs(ID3D12GraphicsCommandList* cmdList)
{
    if (cmdList == nullptr || t_oursDepth == 0 || --t_oursDepth != 0 || !Active())
        return;
    HookGuard g;
    if (!g.ok)
        return;
    bool fallback = false;
    void* target = OursTarget(cmdList, fallback);
    if (target == nullptr)
        return;
    Rec* r = Find(target, false);
    if (r == nullptr || r->skip || r->closed.load() || r->overflow || r->inRenderPass)
        return;
    StartSeg(target, r, CatAfterOurs, 0, 0, cmdList);
}

void Arm() { g_armRequested.store(true); }

Status GetStatus()
{
    Status s;
    s.installed = g_installed;
    s.phase = std::min(g_phase.load(), (int) Draining);
    std::lock_guard lock(g_mutex);
    s.lines = g_lines;
    return s;
}
} // namespace FrameSplit_Dx12
