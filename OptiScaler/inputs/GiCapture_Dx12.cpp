#include "pch.h"

#include "GiCapture_Dx12.h"

#include "Config.h"
#include "ConstantCapture_Dx12.h"
#include "FrameSplit_Dx12.h"
#include "Logger.h"
#include "NVNGX_Parameter.h"
#include "Util.h"

#include <set>
#include <include/d3dx/d3dx12.h>
#include <d3dcompiler.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <deque>
#include <filesystem>
#include <format>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace
{
constexpr int kFrames = 6;          // three of each eye under alternate-eye rendering: enough to watch a fault form while the head moves
constexpr int kSurveyPresents = 4; // a texture the game flips is written every other Present: see it written, read and written again
constexpr UINT kMinSize = 128;
constexpr int kSizeTolerance = 8;   // the viewport a game rounds: 2312 for a render size of 2314
constexpr uint64_t kByteLimit = 12000ull << 20;
constexpr uint64_t kDrainPresents = 3;
constexpr int kSuperSampling = 1;
constexpr int kRayReconstruction = 13;
constexpr size_t kMaxLoggedSurvey = 300;
constexpr size_t kMaxEvents = 1500000;
constexpr size_t kMaxCameraLines = 64;
constexpr size_t kMaxCameraHits = 64;
constexpr size_t kMaxNearMisses = 16;
constexpr uint64_t kSweepLimitMs = 60000;
constexpr size_t kCameraBefore = 1024, kCameraBytes = 4096; // what is saved around each hit
constexpr uint64_t kBindReach = 16 * 1024;     // a root view carries no size: how far behind its address a block is looked for
constexpr uint64_t kScanBudget = 32ull << 20;  // per Present. Upload memory reads at about 100 MB/s, and the ring moves on
constexpr uint64_t kScanPiece = 64 * 1024, kScanSmallest = 256;
constexpr float kEyeTolerance = 0.025f;        // well under the eye step of 0.061: says which eye
constexpr size_t kMaxBlockLines = 2000;        // per Present

enum Phase : int
{
    Idle,
    Requested, // F7 pressed: the survey starts at the next Present
    Surveying,
    Allocating,
    Waiting,   // [Inputs] CaptureDelaySeconds
    Capturing,
    Draining,  // the last copies are on their way to the GPU
    Writing
};

// What the survey saw of one resource. The pointer is identity only and is never dereferenced from here.
struct Seen
{
    D3D12_RESOURCE_DESC desc {};
    std::string name;
    uint32_t order = 0;     // when it was first seen
    uint32_t writes = 0;    // transitions out of UAV / render target, over the whole survey
    bool fromUav = false;
    bool fromTarget = false;
    bool onCompute = false;
    uint32_t after = 0;     // the state, or the access, it went to
    uint32_t others = 0;    // enhanced barriers on it that did not end a write
    std::string barrier;    // the first enhanced barrier on it, for a texture that turns out never to end a write
};

// One line of timeline.txt: a texture barrier, or a submit. Formatted on the worker, never on the game's thread.
struct Event
{
    uint64_t present = 0;
    uint64_t list = 0;   // identity only, as Seen's pointer; a queue's lists follow as extra entries
    uint32_t order = 0;  // Seen::order, the oNNN of the file names
    uint32_t accessBefore = 0, accessAfter = 0, syncBefore = 0, syncAfter = 0, sub = 0;
    int layoutBefore = 0, layoutAfter = 0;
    uint32_t work[4] = {}; // draws, dispatches, rays, indirect on this list: since its last barrier, or for C S P its total so far
    uint64_t address = 0;  // C S: what was bound; V: what the view holds; P: the pipeline, identity only; T: the table; B: the source; H: the GPU start
    uint64_t address2 = 0; // B: the destination; H: the CPU start
    uint32_t values[4] = {}; // K: the first of them
    uint32_t extra = 0;      // K: the offset
    // E enhanced, L legacy (accesses hold states, layouts 0), X a submit, C root CBV, S root SRV, P pipeline, V CBV made,
    // T descriptor table, K root constants (order: root index, extra: offset, sub: count), B buffer copy (order: bytes), H heap set (order: descriptors)
    char kind = 'B';
    char queue = 'd';
    bool graphics = false; // C S T K: the root signature it went to
};

// What a descriptor slot holds, followed from launch. Latest only.
struct DescriptorEntry
{
    uint64_t address = 0, bytes = 0;
    char kind = 'V'; // V a CBV, S a buffer SRV, u a buffer UAV, t a texture SRV, U a texture UAV
    ID3D12Resource* resource = nullptr; // t U: identity only
    uint32_t width = 0, height = 0;
    int format = 0;
};

// A root signature's table parameter: which slots from the table's start it declares.
struct RootRange
{
    uint32_t offset = 0, count = 0;
    bool unbounded = false;
};
using RootParameters = std::vector<std::vector<RootRange>>; // by root index; empty for a parameter that is no table

// What a descriptor table held when it was bound.
struct TableItem
{
    uint32_t slot = 0;
    DescriptorEntry entry;
};

struct HeapRange
{
    uint64_t cpu = 0, gpu = 0;
    uint32_t count = 0;
};

// What the game told Streamline about its camera, one slSetConstants of the survey.
struct CameraLine
{
    uint64_t present = 0;
    uint32_t frame = 0;
    float viewToClip[16] {}, clipToPrevClip[16] {};
    float jitter[2] {}, pos[3] {}, up[3] {}, right[3] {}, fwd[3] {};
    float nearZ = 0, farZ = 0;
};

struct Item
{
    int frame = 0;
    std::string fileName;
    std::string note;
    DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint {};
    UINT rows = 0;
    UINT64 rowBytes = 0;
    UINT64 total = 0;
    ID3D12Resource* readback = nullptr;
    bool recorded = false;
};

// One planned texture: where its copies go, and how many writes this Present has seen.
struct Planned
{
    D3D12_RESOURCE_DESC desc {};
    uint32_t expected = 1;
    uint32_t writesSeen[kFrames] = {};
    int first[kFrames] = {};
    int last[kFrames] = {};
};

std::atomic<int> g_phase { Idle };
std::atomic<uint64_t> g_present { 0 };
std::atomic<uint64_t> g_captureStart { 0 };
std::atomic<uint64_t> g_enhancedBarriers { 0 };
thread_local bool t_inside = false; // our own barriers come back through the hook

std::mutex g_mutex; // everything below while a survey or a frame is recorded
std::unordered_map<ID3D12Resource*, Seen> g_seen;
std::unordered_map<ID3D12Resource*, Planned> g_planned;
std::vector<Item> g_items;
std::vector<std::string> g_frameInfo;
std::vector<Event> g_events;
uint64_t g_eventsDropped = 0, g_submits = 0;
struct Work
{
    uint32_t count[4] = {};
    uint32_t total[4] = {};
};
std::unordered_map<void*, Work> g_work;
std::vector<CameraLine> g_cameras;
CameraLine g_lastCamera, g_olderCamera; // under alternate eyes: one of each
bool g_haveLastCamera = false, g_haveOlderCamera = false;
std::atomic<uint64_t> g_cameraCalls { 0 };
std::atomic<bool> g_sweeping { false };
std::vector<uint64_t> g_bound;  // this Present's addresses, for its scan
std::string g_blockText;        // the scans' lines, for the camera file
std::atomic<int> g_scansRunning { 0 };
std::vector<HeapRange> g_heaps; // the shader-visible heaps the survey saw set
std::vector<std::pair<uint64_t, uint64_t>> g_copyDests; // the survey's buffer copies of a block's size: destination, bytes
std::vector<std::vector<TableItem>> g_tableContents;    // what each T event's table held at its bind: the Event's `extra` - 1
std::vector<int> g_tableExtents;                        // beside it: the slots declared, -1 unbounded, -2 no root signature known
std::unordered_map<ID3D12Resource*, int> g_seenOrders;  // the survey's texture numbers, kept past its end for the timeline
constexpr uint32_t kTableSlots = 64;                    // a table with no root signature known, or an unbounded range: this many from its start
constexpr uint32_t kMaxDeclaredSlots = 4096;            // a declared range past this is bindless in all but name: treated as unbounded
std::mutex g_rootMutex;                                 // on its own: from launch, on any thread
std::unordered_map<void*, RootParameters> g_rootSignatures; // identity only
std::unordered_map<void*, void*> g_listRootSignature[2];    // a list's current one: [0] compute, [1] graphics. Identity only
uint64_t g_rootSignaturesRead = 0, g_rootSignaturesUnread = 0;
std::atomic<uint64_t> g_tablesDeclared { 0 }, g_tablesUnbounded { 0 }, g_tablesUnknown { 0 };
std::string g_copyDumpText;                             // what those of the camera block's size held when the copy was recorded
size_t g_copyDumps = 0;

std::mutex g_descriptorMutex;   // on its own: these come from launch, on any thread, and are read by the camera hunt only
std::unordered_map<uint64_t, DescriptorEntry> g_descriptors;
uint64_t g_descriptorsCopied = 0, g_descriptorIncrement = 0;
constexpr size_t kMaxDescriptors = 4000000;
constexpr size_t kMaxCoverLines = 400;
constexpr uint64_t kMaxCopyToScan = 256 * 1024; // a copy's source, read as a bind's: not a texture upload's megabytes
constexpr uint64_t kMinCopyToCover = 2048;       // a copy worth naming its readers: a block, not a value
constexpr uint64_t kCopyToDump = 2768;           // the camera block's size: every copy of it over the survey, whole, into the camera file
constexpr size_t kMaxCopyDumps = 32;
// One sweep at a time. Written only by Accept, which ConstantCapture calls under its own lock.
float g_sweepSx = 0, g_sweepSy = 0, g_sweepPos[3] {};
std::vector<std::string> g_nearMisses;
uint64_t g_nearMissTotal = 0;
std::string g_surveyText;
uint32_t g_nextOrder = 0;
uint64_t g_barriersSeen = 0, g_buffersSeen = 0, g_bufferBytes = 0, g_globalBarriers = 0, g_textureBarriers = 0;
uint64_t g_surveyEnd = 0, g_startAtTick = 0, g_drainFrom = 0;
uint64_t g_lastEvaluatePresent = UINT64_MAX;
ID3D12Device* g_device = nullptr; // held from the survey's first barrier until the worker has made the buffers
ID3D12Fence* g_fence = nullptr;
HANDLE g_fenceEvent = nullptr;
std::string g_status;

// The upscaler's inputs, from the latest evaluate.
UINT g_renderW = 0, g_renderH = 0, g_colourX = 0, g_colourY = 0;
D3D12_RESOURCE_DESC g_colourDesc {}, g_depthDesc {};
bool g_haveInputs = false;

void SetStatus(const std::string& text)
{
    std::lock_guard lock(g_mutex);
    g_status = text;
}

ID3D12Resource* GetResource(NVSDK_NGX_Parameter* params, const char* name)
{
    ID3D12Resource* resource = nullptr;
    if (params->Get(name, &resource) != NVSDK_NGX_Result_Success)
        params->Get(name, (void**) &resource);
    return resource;
}

unsigned int GetUInt(NVSDK_NGX_Parameter* params, const char* name)
{
    unsigned int value = 0;
    params->Get(name, &value);
    return value;
}

// What a reader of the .dds should take the bytes as: the footprint of plane 0 is what was copied.
DXGI_FORMAT ReadableFormat(DXGI_FORMAT format)
{
    switch (format)
    {
    case DXGI_FORMAT_R32G8X24_TYPELESS:
    case DXGI_FORMAT_D32_FLOAT_S8X24_UINT:
    case DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS:
    case DXGI_FORMAT_R32_TYPELESS:
    case DXGI_FORMAT_D32_FLOAT:
        return DXGI_FORMAT_R32_FLOAT;
    case DXGI_FORMAT_R32G32B32A32_TYPELESS:
        return DXGI_FORMAT_R32G32B32A32_FLOAT;
    case DXGI_FORMAT_R16G16B16A16_TYPELESS:
        return DXGI_FORMAT_R16G16B16A16_FLOAT;
    case DXGI_FORMAT_R32G32_TYPELESS:
        return DXGI_FORMAT_R32G32_FLOAT;
    case DXGI_FORMAT_R16G16_TYPELESS:
        return DXGI_FORMAT_R16G16_FLOAT;
    case DXGI_FORMAT_R16_TYPELESS:
        return DXGI_FORMAT_R16_FLOAT;
    case DXGI_FORMAT_R8G8B8A8_TYPELESS:
        return DXGI_FORMAT_R8G8B8A8_UNORM;
    case DXGI_FORMAT_R10G10B10A2_TYPELESS:
        return DXGI_FORMAT_R10G10B10A2_UNORM;
    default:
        return format;
    }
}

bool IsBlockCompressed(DXGI_FORMAT format)
{
    const UINT f = (UINT) format;
    return (f >= 70 && f <= 84) || (f >= 94 && f <= 99);
}

std::string DebugName(ID3D12Resource* resource)
{
    wchar_t wide[128] = {};
    UINT size = sizeof(wide) - sizeof(wchar_t);
    if (FAILED(resource->GetPrivateData(WKPDID_D3DDebugObjectNameW, &size, wide)) || size == 0)
        return {};
    std::string name;
    for (const wchar_t* p = wide; *p != 0 && name.size() < 80; ++p)
        name += (*p >= 32 && *p < 127) ? (char) *p : '?';
    return name;
}

// A fraction of the render size? Then a pass of the frame wrote it, not a shadow atlas or a look-up table.
// Returns why not, or nullptr when it is wanted.
const char* WhyNot(const D3D12_RESOURCE_DESC& desc, UINT renderW, UINT renderH)
{
    if (desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D)
        return "not a 2D texture";
    if (desc.SampleDesc.Count > 1)
        return "multisampled";
    if (desc.Flags & D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS)
        return "simultaneous access";
    if (desc.Flags & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL)
        return "a depth target";
    if (IsBlockCompressed(desc.Format))
        return "block compressed";
    if (desc.Width < kMinSize || desc.Height < kMinSize)
        return "too small";
    for (const int k : { 1, 2, 4, 8 })
        if (std::abs((int) desc.Width - (int) renderW / k) <= kSizeTolerance &&
            std::abs((int) desc.Height - (int) renderH / k) <= kSizeTolerance)
            return nullptr;
    // Past the render size in 8 bits: the upscaled pictures, what the viewer sees. Larger float targets stay out: 96 MB each, every Present.
    if (desc.Width * 2 >= (UINT64) renderW * 3 && (desc.Format == DXGI_FORMAT_R8G8B8A8_UNORM || desc.Format == DXGI_FORMAT_R8G8B8A8_UNORM_SRGB || desc.Format == DXGI_FORMAT_R8G8B8A8_TYPELESS))
        return nullptr;
    return "not a fraction of the render size";
}

bool AddItem(ID3D12Device* device, std::vector<Item>& items, int frame, std::string fileName, std::string note,
             D3D12_RESOURCE_DESC desc, UINT width, UINT height)
{
    desc.Width = width;
    desc.Height = height;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    Item item;
    item.frame = frame;
    item.fileName = std::move(fileName);
    item.note = std::move(note);
    item.format = ReadableFormat(desc.Format);
    device->GetCopyableFootprints(&desc, 0, 1, 0, &item.footprint, &item.rows, &item.rowBytes, &item.total);
    if (item.total == 0 || item.total == UINT64_MAX)
    {
        LOG_WARN("GI capture: no footprint for {} ({}x{} format {})", item.fileName, width, height, (int) desc.Format);
        return false;
    }
    items.push_back(std::move(item));
    return true;
}

void AllocateOnWorker(ID3D12Device* device, std::vector<Item> items)
{
    uint64_t bytes = 0;
    bool ok = true;
    for (auto& item : items)
    {
        D3D12_HEAP_PROPERTIES heap = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK);
        auto bufferDesc = CD3DX12_RESOURCE_DESC::Buffer(item.total);
        if (FAILED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &bufferDesc,
                                                   D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                   IID_PPV_ARGS(&item.readback))) ||
            item.readback == nullptr)
        {
            LOG_WARN("GI capture: readback buffer of {} MB for {} failed", item.total >> 20, item.fileName);
            ok = false;
            break;
        }
        bytes += item.total;
    }
    if (ok && g_fence == nullptr)
    {
        ok = SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&g_fence)));
        if (ok)
            g_fenceEvent = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        ok = ok && g_fenceEvent != nullptr;
        if (!ok)
            LOG_WARN("GI capture: no fence");
    }
    device->Release();

    std::lock_guard lock(g_mutex);
    g_items = std::move(items);
    if (!ok)
    {
        for (auto& item : g_items)
            if (item.readback != nullptr)
                item.readback->Release();
        g_items.clear();
        g_planned.clear();
        g_status = "GI capture: failed making its buffers (see the log)";
        g_phase.store(Idle);
        return;
    }
    g_frameInfo.clear();
    g_lastEvaluatePresent = UINT64_MAX;
    const int delay = Config::Instance()->CaptureDelaySeconds.value_or_default();
    g_startAtTick = GetTickCount64() + (uint64_t) (delay > 0 ? delay : 0) * 1000;
    g_status = std::format("GI capture: starting in {} s, hold still", delay);
    LOG_INFO("GI capture: {} MB of readback made on the worker; starting in {} s", bytes >> 20, delay);
    g_phase.store(Waiting);
}

// The survey is over: keep what fits, say what did not and why, and make the buffers on a worker. Holds g_mutex.
void Plan()
{
    ID3D12Device* device = g_device;
    g_device = nullptr;
    if (device == nullptr || !g_haveInputs || g_renderW == 0)
    {
        LOG_WARN("GI capture: nothing to plan from: {} barriers seen, an evaluate seen {}: not taken", g_barriersSeen,
                 g_haveInputs);
        g_status = "GI capture: saw no barriers or no evaluate (see the log)";
        if (device != nullptr)
            device->Release();
        g_seen.clear();
        g_phase.store(Idle);
        return;
    }

    std::vector<std::pair<ID3D12Resource*, const Seen*>> ordered;
    for (const auto& entry : g_seen)
        ordered.emplace_back(entry.first, &entry.second);
    std::sort(ordered.begin(), ordered.end(), [](const auto& a, const auto& b) { return a.second->order < b.second->order; });

    // Everything the survey saw goes to the log and the index, kept or not: the buffer sought may be a near miss.
    std::string text = std::format("survey: render {}x{}, {} barriers, {} textures written, {} buffers written ({} MB), "
                                   "enhanced calls {} (texture barriers {}, global {})\n",
                                   g_renderW, g_renderH, g_barriersSeen, ordered.size(), g_buffersSeen,
                                   g_bufferBytes >> 20, g_enhancedBarriers.load(), g_textureBarriers, g_globalBarriers);
    LOG_INFO("GI capture: {}", text);
    std::vector<std::pair<ID3D12Resource*, const Seen*>> wanted;
    size_t logged = 0;
    for (const auto& [resource, seen] : ordered)
    {
        const char* why = seen->writes == 0 ? "no write ended" : WhyNot(seen->desc, g_renderW, g_renderH);
        const std::string line =
            std::format("order {} {}x{} format {} mips {} writes {} from {}{} queue {} after {:X} others {} first {} name '{}' -> {}",
                        seen->order, seen->desc.Width, seen->desc.Height, (int) seen->desc.Format, seen->desc.MipLevels,
                        seen->writes, seen->fromUav ? "UAV " : "", seen->fromTarget ? "RT" : "",
                        seen->onCompute ? "compute" : "direct", seen->after, seen->others,
                        seen->barrier.empty() ? "-" : seen->barrier, seen->name, why != nullptr ? why : "kept");
        text += line + "\n";
        if (logged++ < kMaxLoggedSurvey)
            LOG_INFO("GI capture: {}", line);
        if (why == nullptr)
            wanted.emplace_back(resource, seen);
    }

    // Under the byte limit: the small ones first, the lighting buffers are fractions of the render size.
    std::stable_sort(wanted.begin(), wanted.end(), [](const auto& a, const auto& b) {
        return a.second->desc.Width * a.second->desc.Height < b.second->desc.Width * b.second->desc.Height;
    });

    std::vector<Item> items;
    uint64_t total = 0;
    bool ok = true;
    for (int f = 0; f < kFrames && ok; ++f)
    {
        ok = AddItem(device, items, f, std::format("f{:02}_colour.dds", f), "colour", g_colourDesc, g_renderW, g_renderH) &&
             AddItem(device, items, f, std::format("f{:02}_depth.dds", f), "depth", g_depthDesc, (UINT) g_depthDesc.Width,
                     g_depthDesc.Height);
    }
    for (const auto& item : items)
        total += item.total;

    g_planned.clear();
    size_t dropped = 0;
    for (const auto& [resource, seen] : wanted)
    {
        // The survey ran over kSurveyPresents: writes per Present, rounded up.
        const uint32_t expected = std::max<uint32_t>(1, (seen->writes + kSurveyPresents - 1) / kSurveyPresents);
        std::vector<Item> mine;
        Planned planned;
        planned.desc = seen->desc;
        planned.expected = expected;
        bool fits = ok;
        for (int f = 0; f < kFrames && fits; ++f)
        {
            const std::string base = std::format("f{:02}_o{:03}_{}x{}_fmt{}", f, seen->order, seen->desc.Width,
                                                 seen->desc.Height, (int) seen->desc.Format);
            planned.first[f] = (int) (items.size() + mine.size());
            fits = AddItem(device, mine, f, base + "_first.dds", std::format("order {} first of {} '{}'", seen->order, expected, seen->name),
                           seen->desc, (UINT) seen->desc.Width, seen->desc.Height);
            planned.last[f] = -1;
            if (fits && expected > 1)
            {
                planned.last[f] = (int) (items.size() + mine.size());
                fits = AddItem(device, mine, f, base + "_last.dds", std::format("order {} last of {} '{}'", seen->order, expected, seen->name),
                               seen->desc, (UINT) seen->desc.Width, seen->desc.Height);
            }
        }
        uint64_t bytes = 0;
        for (const auto& item : mine)
            bytes += item.total;
        if (!fits || total + bytes > kByteLimit)
        {
            ++dropped;
            text += std::format("DROPPED order {} {}x{} ({} MB): {}\n", seen->order, seen->desc.Width, seen->desc.Height,
                                bytes >> 20, fits ? "over the byte limit" : "no footprint");
            LOG_WARN("GI capture: order {} {}x{} ({} MB) dropped: {}", seen->order, seen->desc.Width, seen->desc.Height,
                     bytes >> 20, fits ? "over the byte limit" : "no footprint");
            continue;
        }
        total += bytes;
        items.insert(items.end(), std::make_move_iterator(mine.begin()), std::make_move_iterator(mine.end()));
        g_planned.emplace(resource, planned);
    }
    g_surveyText = std::move(text);
    g_seenOrders.clear();
    for (const auto& [resource, seen] : g_seen)
        g_seenOrders[resource] = seen.order;
    g_seen.clear();

    if (!ok || g_planned.empty())
    {
        LOG_WARN("GI capture: {} textures kept of {} seen, inputs ok {}: not taken", g_planned.size(), ordered.size(), ok);
        g_status = "GI capture: nothing to take (see the log)";
        device->Release();
        g_planned.clear();
        g_phase.store(Idle);
        return;
    }
    LOG_INFO("GI capture: planned {} textures, {} dropped, {} files, {} MB, render {}x{}", g_planned.size(), dropped,
             items.size(), total >> 20, g_renderW, g_renderH);
    g_status = "GI capture: making its buffers";
    g_phase.store(Allocating);
    std::thread(AllocateOnWorker, device, std::move(items)).detach();
}

// The copy, right behind the game's barrier: `state` is what the game just put the texture in. Holds g_mutex.
void Record(ID3D12GraphicsCommandList* cmdList, ID3D12Resource* source, Item& item, D3D12_RESOURCE_STATES state,
            const D3D12_BOX* box)
{
    if (item.recorded)
        return;
    const D3D12_TEXTURE_COPY_LOCATION dst = CD3DX12_TEXTURE_COPY_LOCATION(item.readback, item.footprint);
    const D3D12_TEXTURE_COPY_LOCATION src = CD3DX12_TEXTURE_COPY_LOCATION(source, 0);
    const bool transition = (state & D3D12_RESOURCE_STATE_COPY_SOURCE) == 0;
    D3D12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::Transition(source, state, D3D12_RESOURCE_STATE_COPY_SOURCE, 0);
    if (transition)
        cmdList->ResourceBarrier(1, &barrier);
    cmdList->CopyTextureRegion(&dst, 0, 0, 0, &src, box);
    if (transition)
    {
        std::swap(barrier.Transition.StateBefore, barrier.Transition.StateAfter);
        cmdList->ResourceBarrier(1, &barrier);
    }
    item.recorded = true;
}

// The enhanced copy goes in front of the game's barrier, from the state that barrier declares and back to it, so a texture
// the barrier sends to no access (or discards) can still be copied. Only subresource 0. Holds g_mutex.
void RecordEnhanced(ID3D12GraphicsCommandList7* cmdList, const D3D12_TEXTURE_BARRIER& game, Item& item)
{
    if (item.recorded)
        return;
    const D3D12_TEXTURE_COPY_LOCATION dst = CD3DX12_TEXTURE_COPY_LOCATION(item.readback, item.footprint);
    const D3D12_TEXTURE_COPY_LOCATION src = CD3DX12_TEXTURE_COPY_LOCATION(game.pResource, 0);
    D3D12_TEXTURE_BARRIER barrier {};
    barrier.SyncBefore = game.SyncBefore;
    barrier.SyncAfter = D3D12_BARRIER_SYNC_COPY;
    barrier.AccessBefore = game.AccessBefore;
    barrier.AccessAfter = D3D12_BARRIER_ACCESS_COPY_SOURCE;
    barrier.LayoutBefore = game.LayoutBefore;
    barrier.LayoutAfter = D3D12_BARRIER_LAYOUT_COPY_SOURCE;
    barrier.pResource = game.pResource;
    barrier.Subresources.IndexOrFirstMipLevel = 0;
    barrier.Subresources.NumMipLevels = 0; // 0: the line above is a subresource index
    D3D12_BARRIER_GROUP group {};
    group.Type = D3D12_BARRIER_TYPE_TEXTURE;
    group.NumBarriers = 1;
    group.pTextureBarriers = &barrier;
    cmdList->Barrier(1, &group);
    cmdList->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    std::swap(barrier.SyncBefore, barrier.SyncAfter);
    std::swap(barrier.AccessBefore, barrier.AccessAfter);
    std::swap(barrier.LayoutBefore, barrier.LayoutAfter);
    cmdList->Barrier(1, &group);
    item.recorded = true;
}

// The timeline's next line: the barrier takes the work this list did since its last one. Holds g_mutex.
Event* AddEvent(ID3D12GraphicsCommandList* cmdList, char kind, const Seen& seen, uint64_t present)
{
    if (g_events.size() >= kMaxEvents)
    {
        ++g_eventsDropped;
        return nullptr;
    }
    Event& e = g_events.emplace_back();
    e.kind = kind;
    e.queue = cmdList->GetType() == D3D12_COMMAND_LIST_TYPE_COMPUTE ? 'c' : 'd';
    e.present = present;
    e.list = (uint64_t) cmdList;
    e.order = seen.order;
    if (const auto it = g_work.find(cmdList); it != g_work.end())
    {
        std::copy(std::begin(it->second.count), std::end(it->second.count), e.work);
        std::fill(std::begin(it->second.count), std::end(it->second.count), 0u);
    }
    return &e;
}

// The survey's entry of a texture. Holds g_mutex.
Seen& NoteTexture(ID3D12GraphicsCommandList* cmdList, ID3D12Resource* resource, const D3D12_RESOURCE_DESC& desc)
{
    auto [it, fresh] = g_seen.try_emplace(resource);
    Seen& seen = it->second;
    if (fresh)
    {
        seen.desc = desc;
        seen.name = DebugName(resource);
        seen.order = g_nextOrder++;
    }
    seen.onCompute |= cmdList->GetType() == D3D12_COMMAND_LIST_TYPE_COMPUTE;
    if (g_device == nullptr)
        cmdList->GetDevice(IID_PPV_ARGS(&g_device));
    return seen;
}

// A pass has finished writing `resource`. Surveying: count it. Capturing: the item to copy into, or -1. Holds g_mutex.
int OnWrite(ID3D12GraphicsCommandList* cmdList, ID3D12Resource* resource, const D3D12_RESOURCE_DESC& desc, int phase, int frame,
            bool fromUav, bool fromTarget, uint32_t after)
{
    if (phase == Surveying)
    {
        Seen& seen = NoteTexture(cmdList, resource, desc);
        ++seen.writes;
        seen.fromUav |= fromUav;
        seen.fromTarget |= fromTarget;
        seen.after = after;
        return -1;
    }
    if (frame < 0 || frame >= kFrames)
        return -1;
    const auto it = g_planned.find(resource);
    if (it == g_planned.end())
        return -1;
    Planned& planned = it->second;
    // The pointer may be another texture by now: identity is only good together with its shape.
    if (desc.Width != planned.desc.Width || desc.Height != planned.desc.Height || desc.Format != planned.desc.Format)
        return -1;
    const uint32_t write = planned.writesSeen[frame]++;
    int slot = -1;
    if (write == 0)
        slot = planned.first[frame];
    else if (write + 1 == planned.expected)
        slot = planned.last[frame];
    return slot >= 0 && slot < (int) g_items.size() ? slot : -1;
}

// Access COMMON means whatever the layout allows.
bool WriteAccess(D3D12_BARRIER_ACCESS access, D3D12_BARRIER_LAYOUT layout)
{
    if (access == D3D12_BARRIER_ACCESS_COMMON)
        return layout == D3D12_BARRIER_LAYOUT_UNORDERED_ACCESS || layout == D3D12_BARRIER_LAYOUT_RENDER_TARGET ||
               layout == D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_UNORDERED_ACCESS ||
               layout == D3D12_BARRIER_LAYOUT_COMPUTE_QUEUE_UNORDERED_ACCESS;
    return access != D3D12_BARRIER_ACCESS_NO_ACCESS &&
           (access & (D3D12_BARRIER_ACCESS_UNORDERED_ACCESS | D3D12_BARRIER_ACCESS_RENDER_TARGET)) != 0;
}

// By its bits: this module is built /fp:fast, where a NaN passes every comparison (ConstantCapture's raw match included: the first
// real run's 64 hits were all NaN and junk behind it, and used up the search).
bool AllNumbers(const float* v, size_t count)
{
    for (size_t i = 0; i < count; ++i)
    {
        uint32_t bits = 0;
        std::memcpy(&bits, &v[i], sizeof(bits));
        if ((bits & 0x7f800000u) == 0x7f800000u)
            return false;
    }
    return true;
}

bool Close(float a, float b, float rel) { return std::abs(a - b) <= std::abs(b) * rel; }

void NoteNearMiss(const float* v)
{
    ++g_nearMissTotal;
    if (g_nearMisses.size() < kMaxNearMisses)
        g_nearMisses.push_back(std::format("{} {} {} {} | {} {} {} {}", v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]));
}

// A projection's two first rows: (sx 0 0 0) (0 sy 0 0). Called with the mapping alive; 512 bytes before the quad and 128
// after it are inside it. Whatever fails this but starts with the right number is a near miss: its contents go in the file.
bool AcceptProjection(const uint8_t* quad)
{
    float v[8];
    std::memcpy(v, quad, sizeof(v));
    if (!AllNumbers(v, 8) || !Close(v[0], g_sweepSx, 1e-4f))
        return false;
    const bool ok = std::abs(v[1]) < 1e-6f && std::abs(v[2]) < 1e-6f && std::abs(v[3]) < 1e-6f && std::abs(v[4]) < 1e-6f &&
                    Close(v[5], g_sweepSy, 1e-4f) && std::abs(v[6]) < 1e-3f && std::abs(v[7]) < 1e-3f;
    if (!ok)
        NoteNearMiss(v);
    return ok;
}

// The camera's position as three floats in a row, to half a metre: either eye, and a head that moved a little since. The first
// number alone is common in a world's transforms, so only the first two matching make a near miss.
bool AcceptPosition(const uint8_t* quad)
{
    float v[8];
    std::memcpy(v, quad, sizeof(v));
    if (!AllNumbers(v, 8) || std::abs(v[0] - g_sweepPos[0]) > 0.5f)
        return false;
    if (std::abs(v[1] - g_sweepPos[1]) > 0.5f)
        return false;
    const bool ok = std::abs(v[2] - g_sweepPos[2]) <= 0.5f;
    if (!ok)
        NoteNearMiss(v);
    return ok;
}

// The two eyes side by side: each to half a metre of the camera, an eye step apart. The block the previous camera lives in, as read.
bool AcceptEyePair(const uint8_t* quad)
{
    float v[8];
    std::memcpy(v, quad, sizeof(v));
    if (!AllNumbers(v, 3) || std::abs(v[0] - g_sweepPos[0]) > 0.5f || std::abs(v[1] - g_sweepPos[1]) > 0.5f || std::abs(v[2] - g_sweepPos[2]) > 0.5f)
        return false;
    const float dx = v[4] - v[0], dy = v[5] - v[1], dz = v[6] - v[2];
    const bool ok = AllNumbers(v + 4, 3) && std::abs(dx) < 0.25f && std::abs(dy) < 0.25f && std::abs(dz) < 0.25f;
    const float step = std::sqrt(dx * dx + dy * dy + dz * dz);
    return ok && step >= 0.015f && step <= 0.2f;
}

// Every descriptor over each place: one pass over the slots, under their lock for its few milliseconds. Holds no other lock.
std::string CoversText(std::vector<uint64_t> places, const std::vector<HeapRange>& heaps)
{
    std::sort(places.begin(), places.end());
    std::string text;
    size_t lines = 0, over = 0, slots = 0;
    uint64_t copied = 0;
    {
        std::lock_guard lock(g_descriptorMutex);
        slots = g_descriptors.size();
        copied = g_descriptorsCopied;
        for (const auto& [dest, entry] : g_descriptors)
            for (auto it = std::lower_bound(places.begin(), places.end(), entry.address); it != places.end() && *it < entry.address + entry.bytes; ++it)
            {
                if (lines++ >= kMaxCoverLines)
                {
                    ++over;
                    continue;
                }
                std::string where = "in no heap the survey saw set";
                if (g_descriptorIncrement != 0)
                    for (const HeapRange& h : heaps)
                        if (dest >= h.cpu && dest < h.cpu + h.count * g_descriptorIncrement)
                            where = std::format("heap gpu {:X} index {}", h.gpu, (dest - h.cpu) / g_descriptorIncrement);
                text += std::format("covers {:X} dest {:X} {} base {:X} bytes {} offset {} {}\n", *it, dest, entry.kind, entry.address, entry.bytes,
                                    *it - entry.address, where);
            }
    }
    text = std::format("descriptors: {} slots known, {} copied since launch, {} places, {} covers, {} over the limit\n", slots, copied, places.size(), lines, over) + text;
    LOG_INFO("GI capture: camera readers: {} descriptor slots, {} copied since launch, {} eye-pair places, {} descriptors over them", slots, copied, places.size(), lines);
    return text;
}

// Every descriptor over each copy's destination: what the game copies into a GPU buffer is reached through these. Same lock, same few milliseconds.
std::string CopyCoversText(std::vector<std::pair<uint64_t, uint64_t>> dests, const std::vector<HeapRange>& heaps)
{
    std::sort(dests.begin(), dests.end());
    dests.erase(std::unique(dests.begin(), dests.end(), [](const auto& a, const auto& b) { return a.first == b.first; }), dests.end());
    std::string text;
    size_t lines = 0, over = 0;
    {
        std::lock_guard lock(g_descriptorMutex);
        for (const auto& [dest, entry] : g_descriptors)
        {
            const uint64_t end = entry.address + entry.bytes;
            auto it = std::lower_bound(dests.begin(), dests.end(), std::pair<uint64_t, uint64_t>(end, 0));
            while (it != dests.begin())
            {
                --it;
                if (it->first + it->second <= entry.address)
                    break;
                // Whatever else is dropped, not the camera block's: 400 lines in map order lost nearly all of them.
                if (lines++ >= kMaxCoverLines && it->second != kCopyToDump)
                {
                    ++over;
                    continue;
                }
                std::string where = "in no heap the survey saw set";
                if (g_descriptorIncrement != 0)
                    for (const HeapRange& h : heaps)
                        if (dest >= h.cpu && dest < h.cpu + h.count * g_descriptorIncrement)
                            where = std::format("heap gpu {:X} index {}", h.gpu, (dest - h.cpu) / g_descriptorIncrement);
                text += std::format("covers copy {:X} bytes {} dest {:X} {} base {:X} bytes {} {}\n", it->first, it->second, dest, entry.kind, entry.address, entry.bytes, where);
            }
        }
    }
    text = std::format("copy destinations: {} of at least {} bytes, {} covers, {} over the limit\n", dests.size(), kMinCopyToCover, lines, over) + text;
    LOG_INFO("GI capture: camera readers: {} copy destinations, {} descriptors over them", dests.size(), lines);
    return text;
}

std::string HitText(const std::vector<float>& words)
{
    std::string text;
    for (size_t i = 0; i + 4 <= words.size(); i += 4)
        text += std::format("{:+06}: {} {} {} {}\n", (int) (i * 4) - (int) kCameraBefore, words[i], words[i + 1], words[i + 2], words[i + 3]);
    return text;
}

// One Present's binds: which of the places they point at hold the camera. On a worker, straight away: the game's ring keeps these
// bytes for some frames only, and this reads uncached memory. The addresses of one frame's ring lie close together, so their
// ranges merge and the read stays near the size of that frame's constants.
void ScanBoundOnWorker(uint64_t present, std::vector<uint64_t> bound, std::vector<CameraLine> cameras)
{
    const uint64_t startTick = GetTickCount64();
    std::sort(bound.begin(), bound.end());
    const size_t binds = bound.size();
    bound.erase(std::unique(bound.begin(), bound.end()), bound.end());

    // The previous camera's row: its size is the eye step, either sign. In mono it is about 0 and matches half of any buffer: then no search.
    float rowX = 0;
    for (const CameraLine& c : cameras)
        rowX = std::max(rowX, std::abs(c.clipToPrevClip[8]));
    const bool seekRow = rowX > 0.01f;

    std::string text;
    uint64_t readBytes = 0, unreadable = 0, overBudget = 0, positions = 0, rows = 0, nearMisses = 0, overLines = 0, lines = 0;
    std::vector<std::string> misses;
    const auto scanPiece = [&](const uint8_t* data, uint64_t bytes, uint64_t gpu) {
        for (uint64_t i = 0; i + 16 <= bytes; i += 16)
        {
            float v[4];
            std::memcpy(v, data + i, sizeof(v));
            if (!AllNumbers(v, 3))
                continue;
            const CameraLine* eye = nullptr;
            bool close = false;
            for (const CameraLine& c : cameras)
            {
                const float dx = std::abs(v[0] - c.pos[0]), dy = std::abs(v[1] - c.pos[1]), dz = std::abs(v[2] - c.pos[2]);
                close = close || (dx <= 0.5f && dy <= 0.5f && dz <= 0.5f);
                if (eye == nullptr && dx <= kEyeTolerance && dy <= kEyeTolerance && dz <= kEyeTolerance)
                    eye = &c;
            }
            const bool row = seekRow && AllNumbers(v, 4) && std::abs(v[2] - 1.0f) < 1e-3f && std::abs(std::abs(v[0]) - rowX) < 0.01f &&
                             std::abs(v[1]) < 0.02f && std::abs(v[3]) < 0.02f;
            if (eye == nullptr && !row)
            {
                if (close)
                {
                    ++nearMisses;
                    if (misses.size() < kMaxNearMisses)
                        misses.push_back(std::format("{:X}: {} {} {} {}", gpu + i, v[0], v[1], v[2], v[3]));
                }
                continue;
            }
            positions += eye != nullptr ? 1 : 0;
            rows += row ? 1 : 0;
            if (lines++ >= kMaxBlockLines)
            {
                ++overLines;
                continue;
            }
            text += eye != nullptr
                        ? std::format("block present {} position {:X} {} {} {} {} camera present {} parity {}\n", present, gpu + i, v[0], v[1],
                                      v[2], v[3], eye->present, eye->present & 1)
                        : std::format("block present {} row {:X} {} {} {} {}\n", present, gpu + i, v[0], v[1], v[2], v[3]);
        }
    };

    std::vector<uint8_t> piece(kScanPiece);
    for (size_t b = 0; b < bound.size();)
    {
        // Sixteen bytes to a quad: the game's own places are on that grid.
        const uint64_t from = bound[b] & ~15ull;
        uint64_t to = bound[b] + kBindReach;
        for (++b; b < bound.size() && bound[b] <= to; ++b)
            to = bound[b] + kBindReach;
        if (readBytes >= kScanBudget)
        {
            ++overBudget;
            continue;
        }
        for (uint64_t at = from; at < to && readBytes < kScanBudget;)
        {
            uint64_t bytes = std::min(kScanPiece, to - at);
            bool ok = false;
            for (; !ok; bytes /= 2) // the buffer's end, or no mapping at all
            {
                ok = ConstantCapture_Dx12::ReadAtGpuAddress(at, piece.data(), (size_t) bytes);
                if (ok || bytes <= kScanSmallest)
                    break;
            }
            if (!ok)
            {
                // The bound address itself is in no mapping: a GPU buffer, perhaps what a copy (B) fills.
                if (at == from)
                    text += std::format("block unmapped present {} {:X}\n", present, from);
                ++unreadable;
                at += kScanSmallest;
                continue;
            }
            scanPiece(piece.data(), bytes, at);
            readBytes += bytes;
            at += std::max<uint64_t>(bytes & ~15ull, 16); // stays on the grid: a halved piece need not end on it
        }
    }

    LOG_INFO("GI capture: camera readers: present {}: {} binds, {} places, {} KB read, {} ms, {} positions, {} previous-camera rows, {} near misses, "
             "{} pieces not in a mapping, {} ranges over the budget, {} lines over the limit",
             present, binds, bound.size(), readBytes >> 10, GetTickCount64() - startTick, positions, rows, nearMisses, unreadable, overBudget, overLines);
    for (const auto& miss : misses)
        text += std::format("block near miss present {} {}\n", present, miss);
    text += std::format("block summary present {}: binds {} places {} KB {} positions {} rows {} near misses {} unreadable {} over budget {}\n", present, binds,
                        bound.size(), readBytes >> 10, positions, rows, nearMisses, unreadable, overBudget);
    {
        std::lock_guard lock(g_mutex);
        g_blockText += text;
    }
    g_scansRunning.fetch_sub(1);
}

// The camera hunt. The game's projection is known exactly from Streamline; every mapped buffer is searched for it, and what
// surrounds each copy is saved twice, half a second apart, for offline reading: which block holds the view, which the
// previous one. On a worker: the search reads uncached memory for seconds. It takes ConstantCapture's lock 1 ms at a time.
void SweepOnWorker(std::vector<CameraLine> cameras, size_t surveyed)
{
    const uint64_t startTick = GetTickCount64();
    std::string text = std::format("Camera hunt. slSetConstants since launch: {}. Seen in the survey: {} (kept up to {})\n", g_cameraCalls.load(),
                                   surveyed, kMaxCameraLines);
    for (const CameraLine& c : cameras)
    {
        text += std::format("camera present {} parity {} frame {} jitter {} {} pos {} {} {} right {} {} {} up {} {} {} fwd {} {} {} near {} far {}\n",
                            c.present, c.present & 1, c.frame, c.jitter[0], c.jitter[1], c.pos[0], c.pos[1], c.pos[2], c.right[0], c.right[1],
                            c.right[2], c.up[0], c.up[1], c.up[2], c.fwd[0], c.fwd[1], c.fwd[2], c.nearZ, c.farZ);
        const auto matrix = [&](const char* name, const float* m) {
            text += std::format("  {} ({} {} {} {}) ({} {} {} {}) ({} {} {} {}) ({} {} {} {})\n", name, m[0], m[1], m[2], m[3], m[4], m[5], m[6],
                                m[7], m[8], m[9], m[10], m[11], m[12], m[13], m[14], m[15]);
        };
        matrix("viewToClip", c.viewToClip);
        matrix("clipToPrevClip", c.clipToPrevClip);
    }

    g_sweepSx = cameras.empty() ? 0.0f : cameras.back().viewToClip[0];
    g_sweepSy = cameras.empty() ? 0.0f : cameras.back().viewToClip[5];
    if (!cameras.empty())
        std::memcpy(g_sweepPos, cameras.back().pos, sizeof(g_sweepPos));
    const bool haveCamera = g_sweepSx != 0.0f && g_sweepSy != 0.0f;
    if (!haveCamera)
    {
        text += "no projection from Streamline: nothing to search for\n";
        LOG_WARN("GI capture: camera hunt: no projection from Streamline ({} slSetConstants): no search", cameras.size());
    }

    struct Search
    {
        const char* name;
        float first;
        float rel;
        bool (*accept)(const uint8_t*);
        std::vector<const uint8_t*> hits;
        size_t found = 0;
    };
    // The position's raw match must let a head's half metre through: as a share of the number itself.
    const float posRel = std::abs(g_sweepPos[0]) > 1.0f ? 0.5f / std::abs(g_sweepPos[0]) : 1.0f;
    Search searches[] = { { "projection", g_sweepSx, 1e-4f, AcceptProjection, std::vector<const uint8_t*>(kMaxCameraHits, nullptr) },
                          { "position", g_sweepPos[0], posRel, AcceptPosition, std::vector<const uint8_t*>(kMaxCameraHits, nullptr) },
                          { "eyes", g_sweepPos[0], posRel, AcceptEyePair, std::vector<const uint8_t*>(kMaxCameraHits, nullptr) } };
    size_t totalHits = 0;
    bool finished = haveCamera;
    uint64_t sweptMb = 0, totalMb = 0;
    for (Search& s : searches)
    {
        g_nearMisses.clear();
        g_nearMissTotal = 0;
        ConstantCapture_Dx12::PatternScan scan;
        const float pattern[1] = { s.first };
        while (haveCamera && !scan.done && GetTickCount64() - startTick < kSweepLimitMs)
        {
            ConstantCapture_Dx12::FindPattern(pattern, 1, s.rel, s.hits.data(), s.hits.size(), s.accept, scan, 1.0);
            Sleep(1);
        }
        s.found = scan.found;
        totalHits += scan.found;
        finished = finished && scan.done;
        sweptMb = scan.swept >> 20;
        totalMb = scan.total >> 20;
        text += std::format("sweep {}: finished {}, swept {} of {} MB in {} buffers ({} mapped now, {} seen at Map), {} ms in the lock, {} slices, raw matches {}, "
                            "accepted {}, near misses {}\n",
                            s.name, scan.done, scan.swept >> 20, scan.total >> 20, scan.order.size(), ConstantCapture_Dx12::MappedCount(),
                            ConstantCapture_Dx12::BuffersSeenAtMap(), scan.ms, scan.slices, scan.raw, scan.found, g_nearMissTotal);
        for (const auto& near_miss : g_nearMisses)
            text += std::format("near miss {}: {}\n", s.name, near_miss);
    }

    // Before the snapshots' half second: what covers a place now, while its descriptors are the ones of about its frame.
    {
        std::vector<uint64_t> places;
        for (size_t i = 0; i < searches[2].found; ++i)
        {
            uint64_t gpu = 0;
            if (ConstantCapture_Dx12::GpuAddressOf(searches[2].hits[i], &gpu))
                places.push_back(gpu);
        }
        std::vector<HeapRange> heaps;
        std::vector<std::pair<uint64_t, uint64_t>> copyDests;
        {
            std::lock_guard lock(g_mutex);
            heaps = g_heaps;
            copyDests = g_copyDests;
        }
        text += CoversText(std::move(places), heaps);
        text += CopyCoversText(std::move(copyDests), heaps);
        {
            std::lock_guard lock(g_rootMutex);
            text += std::format("root signatures: {} read, {} not read; tables at their binds: {} declared, {} unbounded, {} no signature known\n", g_rootSignaturesRead, g_rootSignaturesUnread, g_tablesDeclared.load(),
                                g_tablesUnbounded.load(), g_tablesUnknown.load());
            LOG_INFO("GI capture: camera readers: root signatures {} read, {} not read; tables {} declared, {} unbounded, {} no signature known", g_rootSignaturesRead, g_rootSignaturesUnread, g_tablesDeclared.load(),
                     g_tablesUnbounded.load(), g_tablesUnknown.load());
        }
        {
            std::lock_guard lock(g_mutex);
            text += g_copyDumpText;
        }
    }

    // Twice: a ring's slot that changes between the two is being rewritten, and the two then show one camera against another.
    for (int snapshot = 0; snapshot < 2; ++snapshot)
    {
        if (snapshot == 1)
            Sleep(500);
        for (const Search& s : searches)
            for (size_t i = 0; i < s.found; ++i)
            {
                uint64_t gpu = 0;
                if (!ConstantCapture_Dx12::GpuAddressOf(s.hits[i], &gpu))
                {
                    text += std::format("hit {} {} snapshot {}: its buffer is gone\n", s.name, i, snapshot);
                    continue;
                }
                std::vector<float> words(kCameraBytes / sizeof(float));
                if (ConstantCapture_Dx12::ReadAtGpuAddress(gpu - kCameraBefore, words.data(), kCameraBytes))
                {
                    text += std::format("hit {} {} snapshot {} gpu {:X}:\n", s.name, i, snapshot, gpu);
                    text += HitText(words);
                    continue;
                }
                // At the end or the start of its buffer: what the search itself knows to be there.
                words.assign((512 + 128) / sizeof(float), 0.0f);
                if (!ConstantCapture_Dx12::ReadAtGpuAddress(gpu - 512, words.data(), 512 + 128))
                {
                    text += std::format("hit {} {} snapshot {} gpu {:X}: unreadable\n", s.name, i, snapshot, gpu);
                    continue;
                }
                text += std::format("hit {} {} snapshot {} gpu {:X} short (starts at -512):\n", s.name, i, snapshot, gpu);
                for (size_t k = 0; k + 4 <= words.size(); k += 4)
                    text += std::format("{:+06}: {} {} {} {}\n", (int) (k * 4) - 512, words[k], words[k + 1], words[k + 2], words[k + 3]);
            }
    }

    // The survey's last scan started just ahead of this worker, and the searches above take far longer than a scan.
    for (int i = 0; i < 500 && g_scansRunning.load() > 0; ++i)
        Sleep(10);
    if (g_scansRunning.load() > 0)
        text += "a scan of the binds was still running: its block lines are missing\n";
    {
        std::lock_guard lock(g_mutex);
        text += g_blockText;
        g_blockText.clear();
    }

    char stamp[32] {};
    const std::time_t now = std::time(nullptr);
    std::tm local {};
    localtime_s(&local, &now);
    std::strftime(stamp, sizeof(stamp), "%Y%m%d-%H%M%S", &local);
    const auto folder = Util::ExePath().parent_path() / "OptiCapture";
    std::error_code ec;
    std::filesystem::create_directories(folder, ec);
    std::ofstream(folder / (std::string(stamp) + "-camera.txt")) << text;
    LOG_INFO("GI capture: camera hunt: {} slSetConstants, projection {} hits, position {} hits, eye pairs {} hits, last sweep {} of {} MB, all finished {}, {} ms, to {}-camera.txt",
             cameras.size(), searches[0].found, searches[1].found, searches[2].found, sweptMb, totalMb, finished, GetTickCount64() - startTick, stamp);
    g_sweeping.store(false);
}

void WriteOnWorker(uint64_t fenceValue)
{
    const uint64_t startTick = GetTickCount64();
    bool completed = g_fence->GetCompletedValue() >= fenceValue;
    if (!completed && SUCCEEDED(g_fence->SetEventOnCompletion(fenceValue, g_fenceEvent)))
        completed = WaitForSingleObject(g_fenceEvent, 10000) == WAIT_OBJECT_0;
    // The Present's queue is not the only one the lists went to (async compute): a moment more, on a worker, costs nothing.
    Sleep(500);
    if (!completed)
        LOG_WARN("GI capture: the fence did not complete in 10 s; writing anyway, the frames may be incomplete");

    std::vector<Item> items;
    std::vector<std::string> info;
    std::string survey;
    std::vector<Event> events;
    std::vector<std::vector<TableItem>> tableContents;
    std::vector<int> tableExtents;
    std::unordered_map<ID3D12Resource*, int> seenOrders;
    uint64_t eventsDropped = 0, submits = 0;
    {
        std::lock_guard lock(g_mutex);
        events = std::move(g_events);
        g_events.clear();
        tableContents = std::move(g_tableContents);
        g_tableContents.clear();
        tableExtents = std::move(g_tableExtents);
        g_tableExtents.clear();
        seenOrders = g_seenOrders;
        g_work.clear();
        eventsDropped = g_eventsDropped;
        submits = g_submits;
        items = std::move(g_items);
        g_items.clear();
        g_planned.clear();
        info = g_frameInfo;
        survey = g_surveyText;
    }

    char stamp[32] {};
    const std::time_t now = std::time(nullptr);
    std::tm local {};
    localtime_s(&local, &now);
    std::strftime(stamp, sizeof(stamp), "%Y%m%d-%H%M%S", &local);
    const auto folder = Util::ExePath().parent_path() / "OptiCapture" / (std::string(stamp) + "-gi");
    std::error_code ec;
    std::filesystem::create_directories(folder, ec);

    std::ofstream index(folder / "index.txt");
    index << "GI buffer capture, " << kFrames << " Presents\n";
    for (const auto& line : info)
        index << line << "\n";
    index << survey;

    // In recording order, which is not execution order across lists: the X lines say the order they were submitted in.
    std::ofstream timeline(folder / "timeline.txt");
    timeline << "E/L present queue list oNNN access before->after layout before->after sync before->after sub | "
                "work since this list's last barrier: draws dispatches rays indirect\n"
                "X present queue list list ...: one submit, in order\n"
                "C/S present queue list g|c rN address | this list's work so far; P present queue list pipeline | the same; V present size address\n"
                "Y present queue list dest < source, a texture copied to a texture; D table n=declared slot:kind:what ... (what the T line's table held at its bind, the slots its root signature declares; unbounded or unknown: its first 64; V CBV, S buffer SRV, u buffer UAV, t texture SRV, U texture UAV); T present queue list g|c rN table | work; K present queue list g|c rN offset count v0 v1 v2 v3 | work; B present queue list source dest bytes; H present queue list gpu cpu descriptors\n";
    for (const Event& e : events)
    {
        if (e.kind == 'X')
        {
            if (e.order == 0)
                timeline << std::format("\nX {} {}", e.present, e.queue);
            timeline << std::format(" {:X}", e.list);
            continue;
        }
        if (e.kind == 'Y')
        {
            const auto name = [&](uint64_t r, uint32_t w, uint32_t h, uint32_t f) {
                const auto known = seenOrders.find((ID3D12Resource*) r);
                return known != seenOrders.end() ? std::format("o{:03}", known->second) : std::format("{}x{}f{}", w, h, f);
            };
            timeline << std::format("\nY {} {} {:X} {} < {}", e.present, e.queue, e.list, name(e.address, e.values[0], e.values[1], e.values[2]),
                                    name(e.address2, e.accessBefore, e.accessAfter, e.syncBefore));
            continue;
        }
        if (e.kind == 'T')
        {
            timeline << std::format("\nT {} {} {:X} {} r{} {:X} | {} {} {} {}", e.present, e.queue, e.list, e.graphics ? 'g' : 'c', e.order, e.address,
                                    e.work[0], e.work[1], e.work[2], e.work[3]);
            if (e.extra != 0 && e.extra <= tableContents.size())
            {
                const int extent = tableExtents[e.extra - 1];
                timeline << std::format("\nD {:X} n={}", e.address, extent == -1 ? std::string("unbounded") : extent == -2 ? std::string("unknown") : std::to_string(extent));
                for (const TableItem& item : tableContents[e.extra - 1])
                {
                    const DescriptorEntry& d = item.entry;
                    if (d.kind == 't' || d.kind == 'U')
                    {
                        const auto known = seenOrders.find(d.resource);
                        if (known != seenOrders.end())
                            timeline << std::format(" {}:{}:o{:03}", item.slot, d.kind, known->second);
                        else
                            timeline << std::format(" {}:{}:{}x{}f{}", item.slot, d.kind, d.width, d.height, d.format);
                    }
                    else
                    {
                        timeline << std::format(" {}:{}:{:X}+{}", item.slot, d.kind, d.address, d.bytes);
                    }
                }
            }
            continue;
        }
        if (e.kind == 'K')
        {
            timeline << std::format("\nK {} {} {:X} {} r{} {} {} {:X} {:X} {:X} {:X} | {} {} {} {}", e.present, e.queue, e.list, e.graphics ? 'g' : 'c', e.order,
                                    e.extra, e.sub, e.values[0], e.values[1], e.values[2], e.values[3], e.work[0], e.work[1], e.work[2], e.work[3]);
            continue;
        }
        if (e.kind == 'B')
        {
            timeline << std::format("\nB {} {} {:X} {:X} {:X} {}", e.present, e.queue, e.list, e.address, e.address2, e.order);
            continue;
        }
        if (e.kind == 'H')
        {
            timeline << std::format("\nH {} {} {:X} {:X} {:X} {}", e.present, e.queue, e.list, e.address, e.address2, e.order);
            continue;
        }
        if (e.kind == 'C' || e.kind == 'S')
        {
            timeline << std::format("\n{} {} {} {:X} {} r{} {:X} | {} {} {} {}", e.kind, e.present, e.queue, e.list, e.graphics ? 'g' : 'c', e.order,
                                    e.address, e.work[0], e.work[1], e.work[2], e.work[3]);
            continue;
        }
        if (e.kind == 'P')
        {
            timeline << std::format("\nP {} {} {:X} {:X} | {} {} {} {}", e.present, e.queue, e.list, e.address, e.work[0], e.work[1], e.work[2], e.work[3]);
            continue;
        }
        if (e.kind == 'V')
        {
            timeline << std::format("\nV {} {} {:X}", e.present, e.order, e.address);
            continue;
        }
        timeline << std::format("\n{} {} {} {:X} o{:03} {:X}->{:X} {}->{} {:X}->{:X} {:X} | {} {} {} {}", e.kind, e.present, e.queue, e.list,
                                e.order, e.accessBefore, e.accessAfter, e.layoutBefore, e.layoutAfter, e.syncBefore, e.syncAfter,
                                e.sub, e.work[0], e.work[1], e.work[2], e.work[3]);
    }
    timeline << "\n";
    timeline.close();
    LOG_INFO("GI capture: timeline {} lines, {} submits, {} lines dropped over the limit", events.size(), submits, eventsDropped);
    if (submits == 0)
        LOG_WARN("GI capture: timeline: no submit seen, so its lists cannot be put in execution order (no queue hook?)");

    int written = 0, skipped = 0;
    for (auto& item : items)
    {
        void* mapped = nullptr;
        const D3D12_RANGE range { 0, (SIZE_T) item.total };
        if (!item.recorded || FAILED(item.readback->Map(0, &range, &mapped)) || mapped == nullptr)
        {
            ++skipped;
            index << "SKIPPED " << item.fileName << " recorded " << item.recorded << " " << item.note << "\n";
        }
        else
        {
            const UINT width = item.footprint.Footprint.Width, height = item.footprint.Footprint.Height;
            uint32_t header[32] = {};
            header[0] = 0x20534444; // "DDS "
            header[1] = 124;
            header[2] = 0x1 | 0x2 | 0x4 | 0x8 | 0x1000; // caps, height, width, pitch, pixel format
            header[3] = height;
            header[4] = width;
            header[5] = (uint32_t) item.rowBytes;
            header[7] = 1;
            header[19] = 32;
            header[20] = 0x4;        // FOURCC
            header[21] = 0x30315844; // "DX10"
            header[27] = 0x1000;
            const uint32_t dx10[5] = { (uint32_t) item.format, 3 /* TEXTURE2D */, 0, 1, 0 };
            std::ofstream file(folder / item.fileName, std::ios::binary);
            file.write(reinterpret_cast<const char*>(header), sizeof(header));
            file.write(reinterpret_cast<const char*>(dx10), sizeof(dx10));
            const char* row = static_cast<const char*>(mapped) + item.footprint.Offset;
            for (UINT y = 0; y < item.rows; ++y, row += item.footprint.Footprint.RowPitch)
                file.write(row, (std::streamsize) item.rowBytes);
            const D3D12_RANGE none { 0, 0 };
            item.readback->Unmap(0, &none);
            index << item.fileName << " " << width << "x" << height << " format " << (int) item.format << " " << item.note << "\n";
            ++written;
        }
        item.readback->Release();
        item.readback = nullptr;
    }
    index.close();

    LOG_INFO("GI capture: {} files written, {} skipped, {} evaluates, to {} ({} ms on the worker)", written, skipped,
             info.size(), folder.string(), GetTickCount64() - startTick);
    SetStatus(std::format("GI capture: done, {} files in OptiCapture\\{}-gi", written, stamp));
    g_phase.store(Idle);
}
} // namespace

bool GiCapture_Dx12::Enabled()
{
    static const bool enabled = [] {
        const bool wanted = Config::Instance()->GiBufferCapture.value_or_default() || Config::Instance()->PerEyeLighting.value_or_default();
        if (wanted && !FrameSplit_Dx12::Enabled())
            LOG_WARN("GI capture: [Inputs] GiBufferCapture is on, but its hooks come with FrameSplitProbe, which is off: "
                     "it stays off");
        return wanted && FrameSplit_Dx12::Enabled();
    }();
    return enabled;
}

bool GiCapture_Dx12::Developer()
{
    static const bool developer = Enabled() && Config::Instance()->GiBufferCapture.value_or_default();
    return developer;
}

void GiCapture_Dx12::Arm()
{
    if (!Developer())
    {
        LOG_WARN("GI capture: pressed, but [Inputs] GiBufferCapture or FrameSplitProbe is off");
        return;
    }
    int expected = Idle;
    if (g_phase.compare_exchange_strong(expected, Requested))
    {
        SetStatus("GI capture: armed, surveying the next frames");
        LOG_INFO("GI capture: pressed, armed");
    }
    else
    {
        LOG_INFO("GI capture: pressed while one is running (phase {}): ignored", expected);
    }
}

bool GiCapture_Dx12::Busy() { return g_phase.load() != Idle; }

std::string GiCapture_Dx12::Status()
{
    std::lock_guard lock(g_mutex);
    return g_status;
}

void GiCapture_Dx12::OnWork(ID3D12GraphicsCommandList* cmdList, int kind)
{
    if (g_phase.load(std::memory_order_relaxed) != Surveying || kind < 0 || kind > 3)
        return;
    std::lock_guard lock(g_mutex);
    Work& work = g_work[cmdList];
    ++work.count[kind];
    ++work.total[kind];
}

void GiCapture_Dx12::OnRootView(ID3D12GraphicsCommandList* cmdList, bool graphics, bool srv, UINT rootIndex, D3D12_GPU_VIRTUAL_ADDRESS address)
{
    if (g_phase.load(std::memory_order_relaxed) != Surveying || cmdList == nullptr || address == 0)
        return;
    const char queue = cmdList->GetType() == D3D12_COMMAND_LIST_TYPE_COMPUTE ? 'c' : 'd';
    const uint64_t present = g_present.load();
    std::lock_guard lock(g_mutex);
    g_bound.push_back(address);
    if (g_events.size() >= kMaxEvents)
    {
        ++g_eventsDropped;
        return;
    }
    Event& e = g_events.emplace_back();
    e.kind = srv ? 'S' : 'C';
    e.queue = queue;
    e.present = present;
    e.list = (uint64_t) cmdList;
    e.order = rootIndex;
    e.address = address;
    e.graphics = graphics;
    if (const auto it = g_work.find(cmdList); it != g_work.end())
        std::copy(std::begin(it->second.total), std::end(it->second.total), e.work);
}

void GiCapture_Dx12::OnPipeline(ID3D12GraphicsCommandList* cmdList, void* pipeline)
{
    if (g_phase.load(std::memory_order_relaxed) != Surveying || cmdList == nullptr)
        return;
    const char queue = cmdList->GetType() == D3D12_COMMAND_LIST_TYPE_COMPUTE ? 'c' : 'd';
    const uint64_t present = g_present.load();
    std::lock_guard lock(g_mutex);
    if (g_events.size() >= kMaxEvents)
    {
        ++g_eventsDropped;
        return;
    }
    Event& e = g_events.emplace_back();
    e.kind = 'P';
    e.queue = queue;
    e.present = present;
    e.list = (uint64_t) cmdList;
    e.address = (uint64_t) pipeline;
    if (const auto it = g_work.find(cmdList); it != g_work.end())
        std::copy(std::begin(it->second.total), std::end(it->second.total), e.work);
}

namespace
{
// A bind of the survey's that carries no address of its own, with the list's work so far.
Event* AddListEvent(ID3D12GraphicsCommandList* cmdList, char kind) // holds g_mutex
{
    if (g_events.size() >= kMaxEvents)
    {
        ++g_eventsDropped;
        return nullptr;
    }
    Event& e = g_events.emplace_back();
    e.kind = kind;
    const D3D12_COMMAND_LIST_TYPE type = cmdList->GetType();
    e.queue = type == D3D12_COMMAND_LIST_TYPE_COMPUTE ? 'c' : type == D3D12_COMMAND_LIST_TYPE_COPY ? 'x' : 'd';
    e.present = g_present.load();
    e.list = (uint64_t) cmdList;
    if (const auto it = g_work.find(cmdList); it != g_work.end())
        std::copy(std::begin(it->second.total), std::end(it->second.total), e.work);
    return &e;
}

void SetDescriptor(uint64_t dest, uint64_t address, uint64_t bytes, char kind)
{
    std::lock_guard lock(g_descriptorMutex);
    if (g_descriptors.size() < kMaxDescriptors || g_descriptors.contains(dest))
    {
        DescriptorEntry entry;
        entry.address = address;
        entry.bytes = bytes;
        entry.kind = kind;
        g_descriptors[dest] = entry;
    }
}

void SetTextureDescriptor(uint64_t dest, ID3D12Resource* resource, char kind)
{
    const D3D12_RESOURCE_DESC desc = resource->GetDesc();
    DescriptorEntry entry;
    entry.kind = kind;
    entry.resource = resource;
    entry.width = (uint32_t) desc.Width;
    entry.height = desc.Height;
    entry.format = (int) desc.Format;
    std::lock_guard lock(g_descriptorMutex);
    if (g_descriptors.size() < kMaxDescriptors || g_descriptors.contains(dest))
        g_descriptors[dest] = entry;
}
} // namespace

void GiCapture_Dx12::OnSetHeaps(ID3D12GraphicsCommandList* cmdList, UINT count, ID3D12DescriptorHeap* const* heaps)
{
    if (g_phase.load(std::memory_order_relaxed) != Surveying || cmdList == nullptr || heaps == nullptr)
        return;
    for (UINT i = 0; i < count; ++i)
    {
        if (heaps[i] == nullptr)
            continue;
        const D3D12_DESCRIPTOR_HEAP_DESC desc = heaps[i]->GetDesc();
        if (desc.Type != D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV)
            continue;
        const HeapRange range { heaps[i]->GetCPUDescriptorHandleForHeapStart().ptr, heaps[i]->GetGPUDescriptorHandleForHeapStart().ptr, desc.NumDescriptors };
        {
            // A game that writes straight into this heap never copies a descriptor: the size of a slot from here too.
            std::lock_guard descriptors(g_descriptorMutex);
            ID3D12Device* device = nullptr;
            if (g_descriptorIncrement == 0 && SUCCEEDED(cmdList->GetDevice(IID_PPV_ARGS(&device))) && device != nullptr)
            {
                g_descriptorIncrement = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
                device->Release();
            }
        }
        std::lock_guard lock(g_mutex);
        const bool known = std::any_of(g_heaps.begin(), g_heaps.end(), [&](const HeapRange& h) { return h.cpu == range.cpu; });
        if (known)
            continue;
        g_heaps.push_back(range);
        if (Event* e = AddListEvent(cmdList, 'H'); e != nullptr)
        {
            e->address = range.gpu;
            e->address2 = range.cpu;
            e->order = range.count;
        }
    }
}

void GiCapture_Dx12::OnRootTable(ID3D12GraphicsCommandList* cmdList, bool graphics, UINT rootIndex, uint64_t table)
{
    if (g_phase.load(std::memory_order_relaxed) != Surveying || cmdList == nullptr)
        return;
    // What the table holds now: the game copies descriptors in just ahead of a bind and reuses the slots later, so at the bind or not at all.
    std::vector<HeapRange> heaps;
    {
        std::lock_guard lock(g_mutex);
        heaps = g_heaps;
    }
    // The slots this root parameter declares, as the list's root signature says.
    std::vector<RootRange> ranges;
    int extent = -2;
    {
        std::lock_guard lock(g_rootMutex);
        const auto& lists = g_listRootSignature[graphics ? 1 : 0];
        if (const auto list = lists.find(cmdList); list != lists.end())
            if (const auto sig = g_rootSignatures.find(list->second); sig != g_rootSignatures.end() && rootIndex < sig->second.size())
                ranges = sig->second[rootIndex];
    }
    if (!ranges.empty())
    {
        uint32_t last = 0;
        bool unbounded = false;
        for (const RootRange& r : ranges)
        {
            unbounded = unbounded || r.unbounded;
            last = std::max(last, r.offset + (r.unbounded ? 0 : r.count));
        }
        extent = unbounded ? -1 : (int) last;
    }
    (extent == -2 ? g_tablesUnknown : extent == -1 ? g_tablesUnbounded : g_tablesDeclared).fetch_add(1, std::memory_order_relaxed);
    const auto declared = [&](uint32_t slot) {
        if (extent < 0)
            return true;
        for (const RootRange& r : ranges)
            if (slot >= r.offset && slot < r.offset + r.count)
                return true;
        return false;
    };
    const uint32_t reach = extent >= 0 ? (uint32_t) extent : kTableSlots;
    std::vector<TableItem> items;
    {
        std::lock_guard lock(g_descriptorMutex);
        if (g_descriptorIncrement != 0)
            for (const HeapRange& h : heaps)
                if (table >= h.gpu && table < h.gpu + h.count * g_descriptorIncrement)
                {
                    const uint64_t first = (table - h.gpu) / g_descriptorIncrement;
                    for (uint64_t i = 0; i < reach && first + i < h.count; ++i)
                        if (declared((uint32_t) i))
                            if (const auto it = g_descriptors.find(h.cpu + (first + i) * g_descriptorIncrement); it != g_descriptors.end())
                                items.push_back(TableItem { (uint32_t) i, it->second });
                    break;
                }
    }
    std::lock_guard lock(g_mutex);
    if (Event* e = AddListEvent(cmdList, 'T'); e != nullptr)
    {
        e->graphics = graphics;
        e->order = rootIndex;
        e->address = table;
        if (!items.empty())
        {
            g_tableContents.push_back(std::move(items));
            g_tableExtents.push_back(extent);
            e->extra = (uint32_t) g_tableContents.size();
        }
    }
}

void GiCapture_Dx12::OnCreateRootSignature(void* signature, const void* blob, SIZE_T bytes)
{
    if (!Enabled() || signature == nullptr || blob == nullptr)
        return;
    // Already loaded: this finds it and waits for nothing.
    static const auto deserialize = reinterpret_cast<PFN_D3D12_CREATE_VERSIONED_ROOT_SIGNATURE_DESERIALIZER>(
        GetProcAddress(GetModuleHandleW(L"d3d12.dll"), "D3D12CreateVersionedRootSignatureDeserializer"));
    ID3D12VersionedRootSignatureDeserializer* reader = nullptr;
    const D3D12_VERSIONED_ROOT_SIGNATURE_DESC* desc = nullptr;
    const HRESULT made = deserialize != nullptr ? deserialize(blob, bytes, IID_PPV_ARGS(&reader)) : E_NOINTERFACE;
    const HRESULT read = SUCCEEDED(made) ? reader->GetRootSignatureDescAtVersion(D3D_ROOT_SIGNATURE_VERSION_1_1, &desc) : made;
    if (FAILED(read) || desc == nullptr)
    {
        if (reader != nullptr)
            reader->Release();
        std::lock_guard lock(g_rootMutex);
        if (g_rootSignaturesUnread++ == 0)
            LOG_WARN("GI capture: a root signature not read: reader {:X}, {} bytes, function found {}", (uint32_t) read, bytes, deserialize != nullptr);
        return;
    }
    RootParameters parameters(desc->Desc_1_1.NumParameters);
    for (UINT p = 0; p < desc->Desc_1_1.NumParameters; ++p)
    {
        const D3D12_ROOT_PARAMETER1& parameter = desc->Desc_1_1.pParameters[p];
        if (parameter.ParameterType != D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE)
            continue;
        uint32_t next = 0; // an append range starts where the one before it ended
        for (UINT r = 0; r < parameter.DescriptorTable.NumDescriptorRanges; ++r)
        {
            const D3D12_DESCRIPTOR_RANGE1& range = parameter.DescriptorTable.pDescriptorRanges[r];
            RootRange out;
            out.offset = range.OffsetInDescriptorsFromTableStart == D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND ? next : range.OffsetInDescriptorsFromTableStart;
            out.unbounded = range.NumDescriptors == UINT_MAX || range.NumDescriptors > kMaxDeclaredSlots;
            out.count = out.unbounded ? 0 : range.NumDescriptors;
            next = out.offset + out.count;
            parameters[p].push_back(out);
        }
    }
    reader->Release();
    std::lock_guard lock(g_rootMutex);
    g_rootSignatures[signature] = std::move(parameters);
    ++g_rootSignaturesRead;
}

void GiCapture_Dx12::OnSetRootSignature(ID3D12GraphicsCommandList* cmdList, bool graphics, void* signature)
{
    if (!Enabled() || cmdList == nullptr)
        return;
    std::lock_guard lock(g_rootMutex);
    g_listRootSignature[graphics ? 1 : 0][cmdList] = signature;
}

void GiCapture_Dx12::OnRootConstants(ID3D12GraphicsCommandList* cmdList, bool graphics, UINT rootIndex, UINT count, const void* values, UINT offset)
{
    if (g_phase.load(std::memory_order_relaxed) != Surveying || cmdList == nullptr || values == nullptr)
        return;
    std::lock_guard lock(g_mutex);
    if (Event* e = AddListEvent(cmdList, 'K'); e != nullptr)
    {
        e->graphics = graphics;
        e->order = rootIndex;
        e->extra = offset;
        e->sub = count;
        std::memcpy(e->values, values, std::min<size_t>(count, 4) * sizeof(uint32_t));
    }
}

void GiCapture_Dx12::OnCopyBuffer(ID3D12GraphicsCommandList* cmdList, ID3D12Resource* dest, UINT64 destOffset, ID3D12Resource* source, UINT64 sourceOffset,
                                  UINT64 bytes)
{
    if (g_phase.load(std::memory_order_relaxed) != Surveying || t_inside || cmdList == nullptr || dest == nullptr || source == nullptr)
        return;
    const D3D12_RESOURCE_DESC desc = source->GetDesc();
    if (desc.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER)
        return;
    const uint64_t size = bytes != 0 ? bytes : desc.Width;
    const uint64_t from = source->GetGPUVirtualAddress() + sourceOffset, to = dest->GetGPUVirtualAddress() + destOffset;
    // Read here, not on the worker: a ring's slot holds this frame's only until it comes round again.
    std::string dump;
    if (size == kCopyToDump)
    {
        std::vector<float> words(kCopyToDump / sizeof(float));
        const D3D12_COMMAND_LIST_TYPE type = cmdList->GetType();
        if (ConstantCapture_Dx12::ReadAtGpuAddress(from, words.data(), kCopyToDump))
        {
            dump = std::format("copy dump present {} queue {} source {:X} dest {:X}\n", g_present.load(), type == D3D12_COMMAND_LIST_TYPE_COMPUTE ? 'c' : 'd', from, to);
            for (size_t i = 0; i + 4 <= words.size(); i += 4)
                dump += std::format("{:05}: {} {} {} {}\n", i * sizeof(float), words[i], words[i + 1], words[i + 2], words[i + 3]);
        }
        else
        {
            dump = std::format("copy dump present {} source {:X} dest {:X}: not mapped\n", g_present.load(), from, to);
        }
    }
    std::lock_guard lock(g_mutex);
    if (!dump.empty() && g_copyDumps++ < kMaxCopyDumps)
        g_copyDumpText += dump;
    if (size <= kMaxCopyToScan)
        g_bound.push_back(from);
    if (size >= kMinCopyToCover)
        g_copyDests.emplace_back(to, size);
    if (Event* e = AddListEvent(cmdList, 'B'); e != nullptr)
    {
        e->address = from;
        e->address2 = to;
        e->order = (uint32_t) std::min<uint64_t>(size, UINT32_MAX);
    }
}

// The camera-block test: the keys are in the header. Holds no game resource and writes no game memory: the game's copy reads our slot instead.
namespace
{
constexpr uint64_t kCamBytes = 2768;
constexpr uint32_t kCamSlotBytes = 3072, kCamSlots = 64;
constexpr size_t kCamHistory = 6;
// The block, as read from a dump (OptiCapture 20261102-114706): the current camera's fields, then the previous camera's.
constexpr uint32_t kCamCurPos = 0x80, kCamPrevPos = 0x520, kCamPrevVpJ = 0x530, kCamPrevVpU = 0x570, kCamPrevInv = 0x5F0, kCamClipToPrev = 0x670, kCamClipToPrevTwin = 0x6B0, kCamPrevPlanes = 0x7B0,
                   kCamRow = 0x690, kCamJitter = 0x730, kCamPos2 = 0x870, kCamPrevPos2 = 0x880, kCamPos3 = 0xA50, kCamPrevPos3 = 0xA60, kCamPos4 = 0xA70, kCamPrevPos4 = 0xA80,
                   kCamDelta = 0xAB0;
struct CamField
{
    uint32_t current, previous, bytes;
};
constexpr CamField kCamFields[] = { { kCamCurPos, kCamPrevPos, 16 }, { 0x0E0, kCamPrevVpJ, 64 },     { 0x120, kCamPrevVpU, 64 },     { 0x160, 0x5B0, 64 },
                                    { 0x1A0, kCamPrevInv, 64 },      { 0x1E0, 0x630, 64 },           { 0x420, 0x460, 64 },           { 0x750, 0x7B0, 96 },
                                    { kCamPos2, kCamPrevPos2, 16 },  { kCamPos3, kCamPrevPos3, 16 }, { kCamPos4, kCamPrevPos4, 16 } };

std::atomic<int> g_camEffect { 0 }; // 0 off, 1 a metre on the row, 2 the whole previous camera a metre to its right, 3 the same eye's previous camera, 4 the same eye's clipToPrevClip only
std::atomic<int> g_camQueues { 0 }; // 0 both, 1 direct only, 2 compute only
std::atomic<bool> g_camUploadStarted { false }, g_camUploadReady { false };
ID3D12Resource* g_camUpload = nullptr; // ours, for the life of the game: a copy in flight may read it at any time
uint8_t* g_camCpu = nullptr;
std::mutex g_camMutex; // holds no other lock
std::deque<std::vector<uint8_t>> g_camSeen; // the distinct blocks as the game wrote them, newest first
uint32_t g_camNextSlot = 0;
uint64_t g_camSwapped[2] = {}, g_camOtherQueue = 0, g_camBehind = 0, g_camUnmapped = 0, g_camRejected = 0, g_camOld = 0, g_camNoHistory = 0, g_camChainBroke = 0, g_camNoBuffer = 0, g_camRowMatched = 0,
         g_camRowMissed = 0;
int g_camRejectLines[5] = {}, g_camChainLines = 0, g_camRowLines = 0;
uint64_t g_camRejectedBy[5] = {};
bool g_camLastNoStep = false, g_camLastTwice = false; // the last frame's block had no eye step in it; and the one before too
uint32_t g_camLastFrame = 0; // by the block's frame number: both queues copy each frame's
std::vector<float> g_camRows; // the game's row x of each block sent since the last Present
// The matrix apart from the game's: the head copy untouched, two slots made there, and each frame's copies read them. Dest is identity for this Present only: a stamp, never kept across one.
std::atomic<bool> g_camSplit { false };
// The Present left to the game: no same-eye camera for it when its first swap was reached. Decided once a Present, so no frame is part ours; read off the lock by the pipeline swap and the copy below.
std::atomic<uint64_t> g_histPlainPresent { UINT64_MAX };
uint64_t g_camBlockLate = 0; // a block that came after its Present had been left to the game: if this counts, the decision is too early
// F8's step 10: each eye its own frame count. _frameNumber (the block's 0x20, x) chooses the half-size sample positions of the lighting (its low two bits) and is read by the shadow and AO passes too.
// Under alternate eyes bit 0 is fixed for an eye, so each sees half the sequence. Halved, each eye counts 0, 1, 2, 3. The direct queue's block only.
constexpr uint32_t kCamFrameNumber = 0x20;
std::atomic<bool> g_camFrameOwn { false };
uint64_t g_camFrameOwnSent = 0;
ID3D12Resource* g_camSplitDest = nullptr;
UINT64 g_camSplitDestOffset = 0;
uint32_t g_camSplitSame = 0, g_camSplitGame = 0;
float g_camGameMatrix[16] = {}, g_camSameMatrix[16] = {}; // _projToPrevProj of this Present's block, the game's and the same eye's, 16 floats as in the block
// What the game itself does to that buffer: nothing has recorded it. Our barriers around the inserted copy are the conservative pair until this says otherwise.
std::atomic<ID3D12Resource*> g_camWatchDest { nullptr };
std::atomic<void*> g_camWatchList { nullptr }; // the list of the head copy, identity only
std::atomic<uint64_t> g_camWatchBuffer { 0 }, g_camWatchLegacy { 0 }, g_camWatchGlobal { 0 };
std::atomic<int> g_camWatchLines { 0 };
uint64_t g_camSplitPresent = UINT64_MAX, g_camSplitPrepared = 0, g_camSplitCompute = 0, g_camSplitNoDest = 0, g_camSplitNoBlock = 0, g_camSplitCopies[2] = {};

float* CamFloats(std::vector<uint8_t>& block, uint32_t offset) { return reinterpret_cast<float*>(block.data() + offset); }
const float* CamFloats(const std::vector<uint8_t>& block, uint32_t offset) { return reinterpret_cast<const float*>(block.data() + offset); }

float CamEyeStep(const std::vector<uint8_t>& block)
{
    const float* a = CamFloats(block, kCamPos3);
    const float* b = CamFloats(block, kCamPrevPos3);
    return std::sqrt((a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]) + (a[2] - b[2]) * (a[2] - b[2]));
}

constexpr const char* kCamRejectNames[5] = { "the row's third value is not 1", "its two current positions differ", "its two previous positions differ", "current and previous over 2 m apart", "no eye step, twice running" };

// -1: the camera block. The distance is one Present's motion and the eye step together: running at Max's frame rate it passed 0.2 m and the block was lost, and the lighting ran two Presents on the other
// eye's matrix (a layer off the ground). Sideways it can also cancel the eye step, in one eye only: so no step in one frame is nothing, in two running it is a flat game.
int CamReject(const std::vector<uint8_t>& block)
{
    if (std::abs(CamFloats(block, kCamRow)[2] - 1.0f) > 1e-4f)
        return 0;
    if (std::memcmp(block.data() + kCamCurPos, block.data() + kCamPos3, 12) != 0)
        return 1;
    if (std::memcmp(block.data() + kCamPrevPos, block.data() + kCamPrevPos3, 12) != 0)
        return 2;
    const float step = CamEyeStep(block);
    if (!(step <= 2.0f))
        return 3;
    uint32_t frame = 0;
    std::memcpy(&frame, block.data() + kCamFrameNumber, sizeof(frame));
    if (frame != g_camLastFrame)
    {
        const bool noStep = step < 0.015f;
        g_camLastTwice = noStep && g_camLastNoStep;
        g_camLastNoStep = noStep;
        g_camLastFrame = frame;
    }
    return g_camLastTwice ? 4 : -1;
}

void CamMultiply(float* out, const float* a, const float* b)
{
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
        {
            out[r * 4 + c] = 0;
            for (int k = 0; k < 4; ++k)
                out[r * 4 + c] += a[r * 4 + k] * b[k * 4 + c];
        }
}

void CamUploadOnWorker(ID3D12Device* device)
{
    D3D12_HEAP_PROPERTIES heap {};
    heap.Type = D3D12_HEAP_TYPE_UPLOAD;
    D3D12_RESOURCE_DESC desc {};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = uint64_t(kCamSlotBytes) * kCamSlots;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    ID3D12Resource* upload = nullptr;
    void* cpu = nullptr;
    const HRESULT made = device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&upload));
    device->Release();
    if (FAILED(made) || FAILED(upload->Map(0, nullptr, &cpu)))
    {
        LOG_WARN("GI capture: camera test: no upload buffer ({:X}): the test does nothing", (uint32_t) made);
        return;
    }
    g_camUpload = upload;
    g_camCpu = static_cast<uint8_t*>(cpu);
    g_camUploadReady.store(true);
    LOG_INFO("GI capture: camera test: upload buffer made");
}

// In this list's order: this Present's slot's 128 bytes into the block the game's head copy filled. Takes g_camMutex.
bool CamSplitCopy(ID3D12GraphicsCommandList7* cmdList, bool sameEye)
{
    std::lock_guard lock(g_camMutex);
    if (g_camSplitPresent != g_present.load() || g_camSplitDest == nullptr)
    {
        ++g_camSplitNoBlock;
        return false;
    }
    constexpr uint64_t kBytes = kCamClipToPrevTwin + 64 - kCamClipToPrev;
    D3D12_BUFFER_BARRIER barrier {};
    barrier.pResource = g_camSplitDest;
    barrier.Offset = 0;
    barrier.Size = UINT64_MAX;
    D3D12_BARRIER_GROUP group {};
    group.Type = D3D12_BARRIER_TYPE_BUFFER;
    group.NumBarriers = 1;
    group.pBufferBarriers = &barrier;
    barrier.SyncBefore = D3D12_BARRIER_SYNC_ALL;
    barrier.AccessBefore = D3D12_BARRIER_ACCESS_COMMON;
    barrier.SyncAfter = D3D12_BARRIER_SYNC_COPY;
    barrier.AccessAfter = D3D12_BARRIER_ACCESS_COPY_DEST;
    cmdList->Barrier(1, &group);
    cmdList->CopyBufferRegion(g_camSplitDest, g_camSplitDestOffset + kCamClipToPrev, g_camUpload, uint64_t(sameEye ? g_camSplitSame : g_camSplitGame) + kCamClipToPrev, kBytes);
    barrier.SyncBefore = D3D12_BARRIER_SYNC_COPY;
    barrier.AccessBefore = D3D12_BARRIER_ACCESS_COPY_DEST;
    barrier.SyncAfter = D3D12_BARRIER_SYNC_ALL;
    barrier.AccessAfter = D3D12_BARRIER_ACCESS_COMMON;
    cmdList->Barrier(1, &group);
    ++g_camSplitCopies[sameEye ? 0 : 1];
    return true;
}

void CamBeeps(int count)
{
    std::thread([count] {
        for (int i = 0; i < count; ++i)
            Beep(880, 120);
    }).detach();
}
} // namespace

void GiCapture_Dx12::PatchCameraCopy(ID3D12GraphicsCommandList* cmdList, ID3D12Resource*& source, UINT64& sourceOffset, UINT64 bytes, ID3D12Resource* dest, UINT64 destOffset)
{
    const int effect = g_camEffect.load(std::memory_order_relaxed);
    if (effect == 0 || t_inside || bytes != kCamBytes || cmdList == nullptr || source == nullptr) // t_inside: our own copy of the whole block comes back through the hook, with g_camMutex held
        return;
    std::vector<uint8_t> block(kCamBytes);
    const bool mapped = ConstantCapture_Dx12::ReadAtGpuAddress(source->GetGPUVirtualAddress() + sourceOffset, block.data(), kCamBytes);
    std::lock_guard lock(g_camMutex);
    if (!mapped)
    {
        ++g_camUnmapped;
        return;
    }
    if (const int reason = CamReject(block); reason >= 0)
    {
        ++g_camRejected;
        ++g_camRejectedBy[reason];
        if (g_camRejectLines[reason]++ < 4)
        {
            const float* row = CamFloats(block, kCamRow);
            const float* a = CamFloats(block, kCamPos3);
            const float* b = CamFloats(block, kCamPrevPos3);
            LOG_INFO("GI capture: camera test: a {} byte copy refused, {}: distance {} row {} {} {} {} position {} {} {} previous {} {} {}", kCamBytes, kCamRejectNames[reason], CamEyeStep(block), row[0], row[1], row[2], row[3],
                     a[0], a[1], a[2], b[0], b[1], b[2]);
        }
        return;
    }

    // Both queues copy the same frame's block: one place in the history. The current camera's fields are never ours, so they name the frame.
    size_t at = g_camSeen.size();
    for (size_t i = 0; i < g_camSeen.size(); ++i)
        if (std::memcmp(g_camSeen[i].data() + 0x0A0, block.data() + 0x0A0, 64) == 0 && std::memcmp(g_camSeen[i].data() + kCamPos3, block.data() + kCamPos3, 12) == 0)
        {
            at = i;
            break;
        }
    if (at == g_camSeen.size())
    {
        g_camSeen.push_front(block);
        if (g_camSeen.size() > kCamHistory)
            g_camSeen.pop_back();
        at = 0;
    }
    // One frame behind what has been seen is either the other queue, late, or a block the game had not written yet: sent, and counted.
    if (at == 1)
        ++g_camBehind;
    // Two frames behind what has been seen: the game has not written this one yet, and what would go out is old.
    if (at >= 2)
    {
        ++g_camOld;
        return;
    }

    const D3D12_COMMAND_LIST_TYPE type = cmdList->GetType();
    const int queues = g_camQueues.load(std::memory_order_relaxed);
    if ((queues == 1 && type == D3D12_COMMAND_LIST_TYPE_COMPUTE) || (queues == 2 && type != D3D12_COMMAND_LIST_TYPE_COMPUTE))
    {
        ++g_camOtherQueue;
        return;
    }
    if (!g_camUploadReady.load())
    {
        ++g_camNoBuffer;
        if (!g_camUploadStarted.exchange(true))
        {
            ID3D12Device* device = nullptr;
            if (SUCCEEDED(cmdList->GetDevice(IID_PPV_ARGS(&device))) && device != nullptr)
                std::thread(CamUploadOnWorker, device).detach();
        }
        return;
    }

    const float gameRow = CamFloats(block, kCamRow)[0];
    if (effect == 1)
    {
        // A metre, in the row's own scale: it holds the eye step's parallax.
        CamFloats(block, kCamRow)[0] += std::abs(gameRow) / CamEyeStep(block);
    }
    else if (effect == 2)
    {
        // One coherent camera a metre to its right, in every form the block holds it. Rows are row-vector: clip = world * M, so the
        // view-projection is T(d) * VP and its inverse Inv * T(-d). On the dump, Inv(0x5F0) * VP(0x530) is the identity, Inv(0x1A0) * VP(0x530) the row.
        float* jittered = CamFloats(block, kCamPrevVpJ);
        const float across = std::sqrt(jittered[0] * jittered[0] + jittered[4] * jittered[4] + jittered[8] * jittered[8]);
        const float d[3] = { jittered[0] / across, jittered[4] / across, jittered[8] / across };
        float* inv = CamFloats(block, kCamPrevInv);
        float inverse[16];
        std::memcpy(inverse, inv, sizeof(inverse));
        for (uint32_t vp : { kCamPrevVpJ, kCamPrevVpU })
        {
            float* m = CamFloats(block, vp);
            for (int c = 0; c < 4; ++c)
                m[12 + c] += d[0] * m[c] + d[1] * m[4 + c] + d[2] * m[8 + c];
        }
        // This clip > the old previous clip > the world > the new previous clip.
        float toMoved[16], moved[16];
        CamMultiply(toMoved, inverse, jittered);
        CamMultiply(moved, CamFloats(block, kCamClipToPrev), toMoved);
        std::memcpy(CamFloats(block, kCamClipToPrev), moved, sizeof(moved));
        for (int r = 0; r < 4; ++r)
            for (int k = 0; k < 3; ++k)
                inv[r * 4 + k] -= inv[r * 4 + 3] * d[k];
        for (uint32_t pos : { kCamPrevPos, kCamPrevPos2, kCamPrevPos3, kCamPrevPos4 })
            for (int k = 0; k < 3; ++k)
                CamFloats(block, pos)[k] -= d[k];
        for (int k = 0; k < 3; ++k)
            CamFloats(block, kCamDelta)[k] += d[k];
        // The six rows at 0x7B0: a normal and its plane through the camera, w = normal . position on the dump.
        for (int row = 0; row < 6; ++row)
        {
            float* plane = CamFloats(block, kCamPrevPlanes + row * 16);
            plane[3] -= plane[0] * d[0] + plane[1] * d[1] + plane[2] * d[2];
        }
    }
    else
    {
        if (at + 2 >= g_camSeen.size())
        {
            ++g_camNoHistory;
            return;
        }
        const std::vector<uint8_t>& older = g_camSeen[at + 1];
        const std::vector<uint8_t>& oldest = g_camSeen[at + 2];
        const auto chained = [](const std::vector<uint8_t>& later, const std::vector<uint8_t>& earlier) {
            const float* a = CamFloats(later, kCamPrevPos3);
            const float* b = CamFloats(earlier, kCamPos3);
            return std::abs(a[0] - b[0]) < 0.002f && std::abs(a[1] - b[1]) < 0.002f && std::abs(a[2] - b[2]) < 0.002f;
        };
        if (!chained(block, older) || !chained(older, oldest))
        {
            ++g_camChainBroke;
            if (g_camChainLines++ < 4)
                LOG_INFO("GI capture: camera test: chain broken: previous {} {} against the frame before's {} {}, its previous {} {} against {} {}", CamFloats(block, kCamPrevPos3)[0],
                         CamFloats(block, kCamPrevPos3)[2], CamFloats(older, kCamPos3)[0], CamFloats(older, kCamPos3)[2], CamFloats(older, kCamPrevPos3)[0],
                         CamFloats(older, kCamPrevPos3)[2], CamFloats(oldest, kCamPos3)[0], CamFloats(oldest, kCamPos3)[2]);
            return;
        }
        // This eye's last frame is two frames back: its current camera is this frame's previous one. Clip to clip chains: this > other eye > this.
        if (effect == 3)
        {
            for (const CamField& f : kCamFields)
                std::memcpy(block.data() + f.previous, oldest.data() + f.current, f.bytes);
            std::memcpy(block.data() + kCamJitter + 8, oldest.data() + kCamJitter, 8);
        }
        for (uint32_t matrix : { kCamClipToPrev, kCamClipToPrevTwin })
        {
            float chain[16];
            CamMultiply(chain, CamFloats(g_camSeen[at], matrix), CamFloats(older, matrix));
            std::memcpy(block.data() + matrix, chain, sizeof(chain));
        }
        if (effect == 3)
            for (int i = 0; i < 3; ++i)
                CamFloats(block, kCamDelta)[i] = CamFloats(block, kCamPos3)[i] - CamFloats(block, kCamPrevPos3)[i];
    }

    // The matrix apart from the game's: this copy goes out as the game wrote it. Only the direct queue's block changes later in the frame (F10: the compute copy changed nothing to see).
    if (effect == 4 && g_camSplit.load(std::memory_order_relaxed))
    {
        if (type == D3D12_COMMAND_LIST_TYPE_COMPUTE)
        {
            ++g_camSplitCompute;
            return;
        }
        if (dest == nullptr)
        {
            ++g_camSplitNoDest;
            return;
        }
        g_camSplitSame = (g_camNextSlot++ % kCamSlots) * kCamSlotBytes;
        g_camSplitGame = (g_camNextSlot++ % kCamSlots) * kCamSlotBytes;
        std::memcpy(g_camCpu + g_camSplitSame, block.data(), kCamBytes);
        std::memcpy(g_camCpu + g_camSplitGame, g_camSeen[at].data(), kCamBytes);
        std::memcpy(g_camGameMatrix, g_camSeen[at].data() + kCamClipToPrev, sizeof(g_camGameMatrix));
        std::memcpy(g_camSameMatrix, block.data() + kCamClipToPrev, sizeof(g_camSameMatrix));
        g_camSplitDest = dest;
        g_camSplitDestOffset = destOffset;
        g_camWatchDest.store(dest, std::memory_order_relaxed);
        g_camWatchList.store(cmdList, std::memory_order_relaxed);
        if (g_camWatchLines.load(std::memory_order_relaxed) < 24)
            LOG_INFO("GI capture: the block's buffer: the game's copy on list {:X}, barriers follow, none means no buffer barrier of its own", (uint64_t) cmdList);
        g_camSplitPresent = g_present.load();
        ++g_camSplitPrepared;
        if (g_histPlainPresent.load(std::memory_order_relaxed) == g_camSplitPresent)
            ++g_camBlockLate;
        g_camRows.push_back(gameRow);
        if (g_camFrameOwn.load(std::memory_order_relaxed))
        {
            std::vector<uint8_t> own = g_camSeen[at];
            uint32_t frame = 0;
            std::memcpy(&frame, own.data() + kCamFrameNumber, sizeof(frame));
            frame >>= 1;
            std::memcpy(own.data() + kCamFrameNumber, &frame, sizeof(frame));
            const uint32_t ownSlot = (g_camNextSlot++ % kCamSlots) * kCamSlotBytes;
            std::memcpy(g_camCpu + ownSlot, own.data(), kCamBytes);
            source = g_camUpload;
            sourceOffset = ownSlot;
            ++g_camFrameOwnSent;
        }
        return;
    }

    const uint32_t slot = (g_camNextSlot++ % kCamSlots) * kCamSlotBytes;
    std::memcpy(g_camCpu + slot, block.data(), kCamBytes);
    source = g_camUpload;
    sourceOffset = slot;
    ++g_camSwapped[type == D3D12_COMMAND_LIST_TYPE_COMPUTE ? 1 : 0];
    g_camRows.push_back(gameRow);
}

std::string GiCapture_Dx12::CameraTestStatus()
{
    std::lock_guard lock(g_camMutex);
    return std::format("effect {} queues {}: sent direct {} compute {}, one frame behind {}, other queue {}, no buffer yet {}, not mapped {}, refused {} (row {}, current positions {}, previous positions {}, over 2 m {}, no eye step {}), old {}, no history {}, chain broken {}, block after its frame was left {}, the game's row against "
                       "Streamline's: matched {}, missed {}; split: prepared {}, compute left the game's {}, no dest {}, no block for the Present {}, same eye copied {}, game's put back {}; the game's own on that buffer: buffer barriers {}, legacy {}, globals on the copy's list {}",
                       g_camEffect.load(), g_camQueues.load(), g_camSwapped[0], g_camSwapped[1], g_camBehind, g_camOtherQueue, g_camNoBuffer, g_camUnmapped, g_camRejected, g_camRejectedBy[0], g_camRejectedBy[1], g_camRejectedBy[2],
                       g_camRejectedBy[3], g_camRejectedBy[4], g_camOld, g_camNoHistory,
                       g_camChainBroke, g_camBlockLate, g_camRowMatched, g_camRowMissed, g_camSplitPrepared, g_camSplitCompute, g_camSplitNoDest, g_camSplitNoBlock, g_camSplitCopies[0], g_camSplitCopies[1], g_camWatchBuffer.load(), g_camWatchLegacy.load(), g_camWatchGlobal.load());
}

void GiCapture_Dx12::CycleCameraEffect()
{
    if (!Enabled())
        return;
    static const char* const names[] = { "off", "a metre on the previous camera's row", "the whole previous camera a metre to its right", "the same eye's previous camera", "the same eye's clipToPrevClip only" };
    LOG_INFO("GI capture: camera test: so far: {}", CameraTestStatus());
    const int effect = (g_camEffect.load() + 1) % 5;
    {
        std::lock_guard lock(g_camMutex);
        g_camSeen.clear();
    }
    g_camEffect.store(effect);
    LOG_INFO("GI capture: camera test: present {}: now {} ({})", g_present.load(), effect, names[effect]);
    CamBeeps(effect + 1);
}

void GiCapture_Dx12::CycleCameraQueues()
{
    if (!Enabled())
        return;
    static const char* const names[] = { "both queues", "the direct queue only", "the compute queue only" };
    LOG_INFO("GI capture: camera test: so far: {}", CameraTestStatus());
    const int queues = (g_camQueues.load() + 1) % 3;
    g_camQueues.store(queues);
    LOG_INFO("GI capture: camera test: present {}: now {}", g_present.load(), names[queues]);
    CamBeeps(queues + 1);
}

void GiCapture_Dx12::OnCreateSrv(ID3D12Resource* resource, const D3D12_SHADER_RESOURCE_VIEW_DESC* desc, uint64_t dest)
{
    if (!Enabled())
        return;
    if (resource == nullptr || desc == nullptr)
    {
        std::lock_guard lock(g_descriptorMutex);
        g_descriptors.erase(dest);
        return;
    }
    if (desc->ViewDimension != D3D12_SRV_DIMENSION_BUFFER)
    {
        SetTextureDescriptor(dest, resource, 't');
        return;
    }
    // A formatted buffer's element size is not worked out here: 16 at most, so the range may reach too far, never too short.
    const uint64_t element = desc->Buffer.StructureByteStride != 0 ? desc->Buffer.StructureByteStride : (desc->Buffer.Flags & D3D12_BUFFER_SRV_FLAG_RAW) != 0 ? 4 : 16;
    SetDescriptor(dest, resource->GetGPUVirtualAddress() + desc->Buffer.FirstElement * element, desc->Buffer.NumElements * element, 'S');
}

void GiCapture_Dx12::OnCopyTexture(ID3D12GraphicsCommandList* cmdList, ID3D12Resource* dest, ID3D12Resource* source)
{
    if (g_phase.load(std::memory_order_relaxed) != Surveying || t_inside || cmdList == nullptr || dest == nullptr || source == nullptr)
        return;
    const D3D12_RESOURCE_DESC to = dest->GetDesc(), from = source->GetDesc();
    if (to.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER || from.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER)
        return;
    std::lock_guard lock(g_mutex);
    if (Event* e = AddListEvent(cmdList, 'Y'); e != nullptr)
    {
        e->address = (uint64_t) dest; // identity only
        e->address2 = (uint64_t) source;
        e->values[0] = (uint32_t) to.Width;
        e->values[1] = to.Height;
        e->values[2] = (uint32_t) to.Format;
        e->accessBefore = (uint32_t) from.Width;
        e->accessAfter = from.Height;
        e->syncBefore = (uint32_t) from.Format;
    }
}

void GiCapture_Dx12::OnCreateUav(ID3D12Resource* resource, const D3D12_UNORDERED_ACCESS_VIEW_DESC* desc, uint64_t dest)
{
    if (!Enabled())
        return;
    if (resource == nullptr)
    {
        std::lock_guard lock(g_descriptorMutex);
        g_descriptors.erase(dest);
        return;
    }
    const D3D12_RESOURCE_DESC rd = resource->GetDesc();
    if (rd.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER)
    {
        SetTextureDescriptor(dest, resource, 'U');
        return;
    }
    if (desc == nullptr || desc->ViewDimension != D3D12_UAV_DIMENSION_BUFFER)
    {
        SetDescriptor(dest, resource->GetGPUVirtualAddress(), rd.Width, 'u');
        return;
    }
    const uint64_t element = desc->Buffer.StructureByteStride != 0 ? desc->Buffer.StructureByteStride : (desc->Buffer.Flags & D3D12_BUFFER_UAV_FLAG_RAW) != 0 ? 4 : 16;
    SetDescriptor(dest, resource->GetGPUVirtualAddress() + desc->Buffer.FirstElement * element, desc->Buffer.NumElements * element, 'u');
}

void GiCapture_Dx12::OnCopyDescriptors(ID3D12Device* device, UINT destRanges, const D3D12_CPU_DESCRIPTOR_HANDLE* destStarts, const UINT* destSizes, UINT srcRanges,
                                       const D3D12_CPU_DESCRIPTOR_HANDLE* srcStarts, const UINT* srcSizes)
{
    if (!Enabled() || device == nullptr || destStarts == nullptr || srcStarts == nullptr)
        return;
    std::lock_guard lock(g_descriptorMutex);
    if (g_descriptorIncrement == 0)
        g_descriptorIncrement = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    // A null size array means ranges of one. Both sides walk their own ranges, one slot at a time.
    UINT d = 0, s = 0, dAt = 0, sAt = 0;
    while (d < destRanges && s < srcRanges)
    {
        const UINT dSize = destSizes != nullptr ? destSizes[d] : 1, sSize = srcSizes != nullptr ? srcSizes[s] : 1;
        if (dAt >= dSize)
        {
            ++d;
            dAt = 0;
            continue;
        }
        if (sAt >= sSize)
        {
            ++s;
            sAt = 0;
            continue;
        }
        const uint64_t to = destStarts[d].ptr + dAt * g_descriptorIncrement, from = srcStarts[s].ptr + sAt * g_descriptorIncrement;
        if (const auto it = g_descriptors.find(from); it != g_descriptors.end())
        {
            if (g_descriptors.size() < kMaxDescriptors || g_descriptors.contains(to))
            {
                const DescriptorEntry entry = it->second; // the insert may move it
                g_descriptors[to] = entry;
                ++g_descriptorsCopied;
            }
        }
        else if (!g_descriptors.empty())
        {
            g_descriptors.erase(to);
        }
        ++dAt;
        ++sAt;
    }
}

void GiCapture_Dx12::OnCreateCbv(const D3D12_CONSTANT_BUFFER_VIEW_DESC* desc, uint64_t dest)
{
    if (!Enabled())
        return;
    if (desc == nullptr || desc->BufferLocation == 0)
    {
        std::lock_guard lock(g_descriptorMutex);
        g_descriptors.erase(dest);
        return;
    }
    SetDescriptor(dest, desc->BufferLocation, desc->SizeInBytes, 'V');
    if (g_phase.load(std::memory_order_relaxed) != Surveying)
        return;
    const uint64_t present = g_present.load();
    std::lock_guard lock(g_mutex);
    g_bound.push_back(desc->BufferLocation);
    if (g_events.size() >= kMaxEvents)
    {
        ++g_eventsDropped;
        return;
    }
    Event& e = g_events.emplace_back();
    e.kind = 'V';
    e.present = present;
    e.order = desc->SizeInBytes;
    e.address = desc->BufferLocation;
}

void GiCapture_Dx12::OnCameraConstants(const float* viewToClip, const float* clipToPrevClip, const float* jitter, const float* pos,
                                       const float* up, const float* right, const float* fwd, float nearZ, float farZ, uint32_t frame)
{
    if (!Enabled())
        return;
    g_cameraCalls.fetch_add(1, std::memory_order_relaxed);
    CameraLine c;
    c.present = g_present.load();
    c.frame = frame;
    std::memcpy(c.viewToClip, viewToClip, sizeof(c.viewToClip));
    std::memcpy(c.clipToPrevClip, clipToPrevClip, sizeof(c.clipToPrevClip));
    std::memcpy(c.jitter, jitter, sizeof(c.jitter));
    std::memcpy(c.pos, pos, sizeof(c.pos));
    std::memcpy(c.up, up, sizeof(c.up));
    std::memcpy(c.right, right, sizeof(c.right));
    std::memcpy(c.fwd, fwd, sizeof(c.fwd));
    c.nearZ = nearZ;
    c.farZ = farZ;
    std::lock_guard lock(g_mutex);
    g_olderCamera = g_lastCamera;
    g_haveOlderCamera = g_haveLastCamera;
    g_lastCamera = c;
    g_haveLastCamera = true;
    if (g_phase.load(std::memory_order_relaxed) == Surveying && g_cameras.size() < kMaxCameraLines)
        g_cameras.push_back(c);
}

void GiCapture_Dx12::OnExecute(ID3D12CommandQueue* queue, ID3D12CommandList* const* lists, UINT count)
{
    if (g_phase.load(std::memory_order_relaxed) != Surveying || queue == nullptr || lists == nullptr)
        return;
    const char type = queue->GetDesc().Type == D3D12_COMMAND_LIST_TYPE_COMPUTE ? 'c' : 'd';
    const uint64_t present = g_present.load();
    std::lock_guard lock(g_mutex);
    ++g_submits;
    for (UINT i = 0; i < count && g_events.size() < kMaxEvents; ++i)
    {
        Event& e = g_events.emplace_back();
        e.kind = 'X';
        e.queue = type;
        e.present = present;
        e.list = (uint64_t) lists[i];
        e.order = i; // 0 starts the line
    }
}

// The per-eye history test (F8). Each Present the game sends one texture of each flipping pair to SRV as history, in one list of the
// direct queue, and writes its partner. Under alternate eyes that history is the other eye's. Right in front of the game's own barrier: this
// texture > our copy of it (it holds the other eye's newest), then our copy of its partner (this eye's newest) > this texture. The game's
// barrier states the layout and the moment. Pointers are identity only, checked against the barrier's resource at every use.
namespace
{
constexpr int kHistSurveyPresents = 4, kHistRetryPresents = 60, kHistMaxAttempts = 20, kHistSlowRetryPresents = 450, kHistWarnAttempts = 5, kHistNoFlipPresents = 30, kHistOnStep = 18;
struct HistSeen
{
    uint64_t present = 0, list = 0;
    ID3D12Resource* resource = nullptr; // identity only
    D3D12_RESOURCE_DESC desc {};
    int layoutBefore = 0;
};
struct HistPair
{
    ID3D12Resource* a = nullptr; // identity only
    ID3D12Resource* b = nullptr; // identity only
    ID3D12Resource* c = nullptr; // identity only: the map's third, when it runs on three (read A, write B; read B, write C; read C, write A: launched at a size other than 3472, `08b2189`'s trace)
    ID3D12Resource* storeA = nullptr; // ours, for the life of the game: a list in flight may use them
    ID3D12Resource* storeB = nullptr;
    D3D12_RESOURCE_DESC desc {};
    bool lighting = false, validA = false, validB = false;
    int lightingIndex = -1; // 0 or 1: its place among the two lighting pairs of the batch; -1 the map and the rest
    bool snapshot = false; // written once a frame and read as this frame's and, through its partner, the last: swapped in front of the barrier that starts the write
    uint64_t swappedA = 0, swappedB = 0; // the Present of each side's last swap: one a Present
    // Steps 9 and 10: history the accumulation's neighbours read. Each is written one Present in two and read as previous through its partner, so swapped like a snapshot,
    // at the barrier that starts its write. 1 g_manyLightsMoments, 2 the half-size depth (g_depthStencilPrev), both for GenerateManyLightsMomentsCS; 3 g_stencil, for the accumulation.
    int extra = 0;
    uint64_t startedA = UINT64_MAX, startedB = UINT64_MAX; // the Present each side last started a write: seen alive by the game's own barrier
    D3D12_BARRIER_LAYOUT triggerBefore = D3D12_BARRIER_LAYOUT_COMMON, triggerAfter = D3D12_BARRIER_LAYOUT_COMMON;
};
struct HistStore
{
    DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;
    UINT64 width = 0;
    UINT height = 0;
    int ordinal = 0;
    ID3D12Resource* a = nullptr;
    ID3D12Resource* b = nullptr;
};
// 0 off, 1 the same eye's matrix + the lighting pairs and their map, 2 also every other pair of that list and the snapshots, 3 as 2 with the matrix in only from the lighting's history barrier on,
// 4 as 3 with the game's matrix put back after the lighting, 5 as 4 with no swaps, 6 as 4 with only the first lighting pair swapped, 7 with only the second, 8 as 4 with the motion same-eye for the accumulation, 9 as 8 with the moments and their half-size depth swapped, 10 as 9 with each eye its own frame count, 11 as 10 with the stencil too, 12 to 14 as 9 with the history limit edited, 15 as 13 with an object's own share of the motion doubled, 16 as 15 with the reflections' new frame by the whole vector, 17 as 15 with slow motion doubled too, 18 as 17 with the own motion averaged with the other eye's
std::atomic<int> g_histMode { 0 };
std::atomic<int> g_clipState { 0 }; // 0 idle, 1 finding the pairs and making the buffer, 2 running, 3 draining, 4 writing. The history hooks stay on for it, with no swap at step 0
std::atomic<UINT> g_histRenderW { 0 }, g_histRenderH { 0 };
std::mutex g_histMutex; // holds no other lock
std::vector<HistSeen> g_histSeen;
std::vector<HistSeen> g_histSnapSeen; // R32_UINT at any size, going from SRV to UAV
std::vector<HistSeen> g_histMomentsSeen, g_histStencilSeen; // for steps 9 and 10: their own barriers name them
uint64_t g_histExtraSaves = 0, g_histExtraPuts = 0, g_histMomentsLate = 0;
uint64_t g_histMomentsLooked = 0, g_histExtraPartnerGone = 0;
int g_histMomentsLines = 0;
std::vector<HistPair> g_histPairs;
std::vector<HistStore> g_histStores;
ID3D12Device* g_histDevice = nullptr; // a reference of ours, given to the worker
bool g_histSurveying = false, g_histStoresMaking = false;
std::atomic<bool> g_histStoresReady { false }; // THE switch: found, and the stores made. Everything of ours that changes the picture stands behind it: not found is the plain game
uint64_t g_histSurveyStart = 0, g_histSurveyEnd = 0, g_histNextAttempt = 0, g_histLastFlip = 0;
uint64_t g_histMapPresent = 0, g_histMapMissed = 0, g_histWatchMapMissed = 0; // a Present whose lighting flipped and whose map did not: the map has moved to textures not in the set
bool g_histWarnPending = false; // switched on and not yet found
std::unordered_map<ID3D12Resource*, uint32_t> g_histWritten; // the survey: to UAV or RT, a bit for each Present from its start; identity only
uint64_t g_histWatchNext = 0, g_histWatchSize = 0, g_histWatchGone = 0;
std::atomic<uint64_t> g_pipeSwaps { 0 };
constexpr uint64_t kHistUiPresents = 45;
uint64_t g_histUiNext = 0, g_histUiCorrected = 0, g_histUiSwaps = 0;
bool g_histUiCorrecting = false, g_histUiLimit = false;
int g_histAttempts = 0;
uint64_t g_histSaves = 0, g_histPuts = 0, g_histOnlySaved = 0, g_histSizeChanged = 0, g_histNoStore = 0, g_histRefused = 0, g_histResurveys = 0, g_histPhaseSame = 0, g_histPhaseSlip = 0, g_histTwice = 0, g_histSnapSaves = 0, g_histSnapPuts = 0;
int g_histLastSignA = 0;
// The bracket, from the recording, whose order is not the execution's: the list with the close is recorded before the list with the open. Each has its own stamp.
uint64_t g_histOpenPresent = UINT64_MAX, g_histClosePresent = UINT64_MAX, g_histBatchPresent = UINT64_MAX, g_histOpened = UINT64_MAX, g_histClosed = UINT64_MAX, g_histOpens = 0, g_histCloses = 0, g_histSecondOpens = 0;
ID3D12GraphicsCommandList7* g_histOpenList = nullptr; // identity only, this Present
uint64_t g_histWhole = 0, g_histOpenOnly = 0, g_histCloseOnly = 0, g_histNeither = 0, g_histBatchNoClose = 0;
ID3D12GraphicsCommandList7* g_histBatchList = nullptr; // identity only, this Present
std::vector<ID3D12Resource*> g_histBatch; // identity only, this Present: what the lighting's UAV > UAV batch holds
int g_histNoCloseLines = 0;

void HistBarrier(ID3D12GraphicsCommandList7* cmdList, ID3D12Resource* resource, D3D12_BARRIER_SYNC syncBefore, D3D12_BARRIER_ACCESS accessBefore, D3D12_BARRIER_LAYOUT layoutBefore,
                 D3D12_BARRIER_SYNC syncAfter, D3D12_BARRIER_ACCESS accessAfter, D3D12_BARRIER_LAYOUT layoutAfter)
{
    D3D12_TEXTURE_BARRIER barrier {};
    barrier.SyncBefore = syncBefore;
    barrier.SyncAfter = syncAfter;
    barrier.AccessBefore = accessBefore;
    barrier.AccessAfter = accessAfter;
    barrier.LayoutBefore = layoutBefore;
    barrier.LayoutAfter = layoutAfter;
    barrier.pResource = resource;
    barrier.Subresources.IndexOrFirstMipLevel = 0;
    barrier.Subresources.NumMipLevels = 0; // 0: the line above is a subresource index
    D3D12_BARRIER_GROUP group {};
    group.Type = D3D12_BARRIER_TYPE_TEXTURE;
    group.NumBarriers = 1;
    group.pTextureBarriers = &barrier;
    cmdList->Barrier(1, &group);
}

// Our textures keep the layout of their last copy: a barrier from no access is refused once they have been used. Holds g_histMutex.
std::unordered_map<ID3D12Resource*, D3D12_BARRIER_LAYOUT> g_histLayouts;

void HistStoreTo(ID3D12GraphicsCommandList7* cmdList, ID3D12Resource* store, D3D12_BARRIER_LAYOUT layout)
{
    const auto access = [](D3D12_BARRIER_LAYOUT l) { return l == D3D12_BARRIER_LAYOUT_COPY_DEST ? D3D12_BARRIER_ACCESS_COPY_DEST : D3D12_BARRIER_ACCESS_COPY_SOURCE; };
    const auto it = g_histLayouts.find(store);
    if (it == g_histLayouts.end())
        HistBarrier(cmdList, store, D3D12_BARRIER_SYNC_NONE, D3D12_BARRIER_ACCESS_NO_ACCESS, D3D12_BARRIER_LAYOUT_COMMON, D3D12_BARRIER_SYNC_COPY, access(layout), layout);
    else
        HistBarrier(cmdList, store, D3D12_BARRIER_SYNC_COPY, access(it->second), it->second, D3D12_BARRIER_SYNC_COPY, access(layout), layout);
    g_histLayouts[store] = layout;
}

void HistCopy(ID3D12GraphicsCommandList7* cmdList, ID3D12Resource* from, ID3D12Resource* to)
{
    const D3D12_TEXTURE_COPY_LOCATION dst = CD3DX12_TEXTURE_COPY_LOCATION(to, 0), src = CD3DX12_TEXTURE_COPY_LOCATION(from, 0);
    cmdList->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
}

ID3D12Resource* HistMakeStore(ID3D12Device* device, const D3D12_RESOURCE_DESC& like, D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE)
{
    D3D12_HEAP_PROPERTIES heap {};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC desc = like;
    desc.Flags = flags;
    desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    ID3D12Resource* store = nullptr;
    const HRESULT made = device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&store));
    if (FAILED(made))
        LOG_WARN("GI capture: history test: no store for {}x{} format {}: {:X}", desc.Width, desc.Height, (int) desc.Format, (uint32_t) made);
    return FAILED(made) ? nullptr : store;
}

// Makes what the pairs need, and keeps what a size has already made: nothing is released while a list may hold it.
void HistStoresOnWorker(ID3D12Device* device, std::vector<HistPair> pairs)
{
    bool all = true;
    std::vector<HistStore> stores;
    {
        std::lock_guard lock(g_histMutex);
        stores = g_histStores;
    }
    std::vector<int> used(stores.size(), 0);
    for (HistPair& pair : pairs)
    {
        int ordinal = 0;
        HistStore* found = nullptr;
        for (size_t i = 0; i < stores.size(); ++i)
            if (stores[i].format == pair.desc.Format && stores[i].width == pair.desc.Width && stores[i].height == pair.desc.Height)
            {
                if (used[i] == 0)
                {
                    found = &stores[i];
                    used[i] = 1;
                    break;
                }
                ++ordinal;
            }
        HistStore fresh {};
        if (found == nullptr)
        {
            fresh.format = pair.desc.Format;
            fresh.width = pair.desc.Width;
            fresh.height = pair.desc.Height;
            fresh.ordinal = ordinal;
            fresh.a = HistMakeStore(device, pair.desc);
            fresh.b = fresh.a != nullptr ? HistMakeStore(device, pair.desc) : nullptr;
            all = all && fresh.b != nullptr;
            found = &fresh;
        }
        pair.storeA = found->a;
        pair.storeB = found->b;
        if (found == &fresh && fresh.b != nullptr)
        {
            stores.push_back(fresh);
            used.push_back(1);
        }
    }
    device->Release();
    std::lock_guard lock(g_histMutex);
    g_histStores = std::move(stores);
    g_histStoresMaking = false;
    if (g_histPairs.size() != pairs.size())
        return; // a new survey has begun
    for (size_t i = 0; i < pairs.size(); ++i)
    {
        g_histPairs[i].storeA = pairs[i].storeA;
        g_histPairs[i].storeB = pairs[i].storeB;
    }
    g_histStoresReady = all;
    LOG_INFO("GI capture: history test: stores {} for {} pairs", all ? "ready" : "NOT all made", pairs.size());
}

// Four Presents of the barriers that send a texture to SRV: which flip, with which partner. Holds g_histMutex. Refuses anything but the expected set.
void HistFinish()
{
    g_histSurveying = false;
    const UINT renderW = g_histRenderW.load(), renderH = g_histRenderH.load();
    std::string nearly;
    struct Flip
    {
        const HistSeen* first = nullptr;
        std::vector<uint64_t> presents;
        std::vector<std::pair<uint64_t, uint64_t>> seen; // Present, list: one texture can go to SRV on two lists in a Present
    };
    std::vector<Flip> flips;
    for (const HistSeen& seen : g_histSeen)
    {
        auto it = std::find_if(flips.begin(), flips.end(), [&](const Flip& f) { return f.first->resource == seen.resource; });
        if (it == flips.end())
        {
            flips.push_back(Flip { &seen, { seen.present }, { { seen.present, seen.list } } });
            continue;
        }
        it->seen.emplace_back(seen.present, seen.list);
        if (std::find(it->presents.begin(), it->presents.end(), seen.present) == it->presents.end())
            it->presents.push_back(seen.present);
    }
    const auto period2 = [](const Flip& f) { return f.presents.size() == 2 && f.presents[1] == f.presents[0] + 2; };
    const auto isLighting = [&](const D3D12_RESOURCE_DESC& d) { return d.Format == DXGI_FORMAT_R16G16B16A16_FLOAT && d.Width == renderW && d.Height == renderH; };
    const auto isMap = [&](const D3D12_RESOURCE_DESC& d) { return d.Format == DXGI_FORMAT_R8_UNORM && d.Width == renderW && d.Height == renderH; };
    // History that flips is read before it is written in a frame. Written in a Present it is read in = that frame's, on whatever list (Max's third pair, the extras): no swap, its copy would be old.
    // Any order inside the Present: recording order is not the GPU's. 8 captures at 3 sizes and both lighting levels agree (rule_nofix.py). This rule is for the head's lighting and map only:
    // the snapshots, moments, half-size depth and stencil have their own routes below and do not come through here (the stencil reads as written first at the head too, and is in the set).
    const auto writtenThen = [&](const Flip& f) {
        const auto w = g_histWritten.find(f.first->resource);
        if (w == g_histWritten.end())
            return false;
        return std::any_of(f.presents.begin(), f.presents.end(), [&](uint64_t present) {
            const uint64_t at = present - g_histSurveyStart;
            return at < 32 && (w->second >> at & 1) != 0;
        });
    };
    // The map on three textures: each read in one Present, written in the one before, idle in the third, so 4 Presents see them in 2, 1 and 1, one a Present. Exactly that, or it is not the map.
    std::vector<const Flip*> mapRing;
    {
        std::vector<const Flip*> maps;
        std::vector<uint64_t> covered;
        bool pairToo = false, repeats = false;
        for (const Flip& f : flips)
            if (isMap(f.first->desc) && !writtenThen(f))
            {
                pairToo = pairToo || period2(f);
                maps.push_back(&f);
                repeats = repeats || (f.presents.size() == 2 && f.presents[1] == f.presents[0] + 3);
                covered.insert(covered.end(), f.presents.begin(), f.presents.end());
            }
        std::sort(covered.begin(), covered.end());
        if (!pairToo && repeats && maps.size() == 3 && covered.size() == 4 && std::adjacent_find(covered.begin(), covered.end()) == covered.end())
            mapRing = maps;
    }
    // The lighting's list in each Present: the one sending the R8 map to SRV, which goes nowhere else. Not "the list with the most": lists are recorded in parallel, so which saw a texture first is a race.
    // One map list a Present, or refused: the rule below must never run on the wrong list in silence.
    std::unordered_map<uint64_t, uint64_t> headOf;
    bool oneMap = true;
    for (const Flip& f : flips)
        if ((period2(f) || std::find(mapRing.begin(), mapRing.end(), &f) != mapRing.end()) && isMap(f.first->desc) && !writtenThen(f)) // a second R8 at render size, written where it is read: this frame's, not the map (a setting of the game's)
            for (const auto& [present, list] : f.seen)
            {
                const auto [it, added] = headOf.emplace(present, list);
                oneMap = oneMap && (added || it->second == list);
            }
    const auto onHead = [&](const Flip& f) {
        return std::all_of(f.presents.begin(), f.presents.end(), [&](uint64_t present) {
            const auto head = headOf.find(present);
            return head != headOf.end() && std::find(f.seen.begin(), f.seen.end(), std::make_pair(present, head->second)) != f.seen.end();
        });
    };
    const auto isHistory = [&](const Flip& f) { return period2(f) && onHead(f) && !writtenThen(f); };
    int lightingFlips = 0;
    for (const Flip& f : flips)
    {
        lightingFlips += isHistory(f) && isLighting(f.first->desc) ? 1 : 0;
        if (f.first->desc.Width * 2 >= renderW && f.first->desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D)
            nearly += std::format(" [{}x{} fmt {} list {:X} from {} presents {}{}{}{}]", f.first->desc.Width, f.first->desc.Height, (int) f.first->desc.Format, f.first->list, f.first->layoutBefore,
                                  f.presents.size(), period2(f) ? " every other" : "", onHead(f) ? " head" : "", writtenThen(f) ? " written" : "");
    }

    std::vector<HistPair> pairs;
    int lightingPairs = 0, mapPairs = 0;
    bool balanced = oneMap && !headOf.empty();
    std::vector<const Flip*> done;
    for (const Flip& f : flips)
    {
        if (!balanced || !isHistory(f) || std::find(done.begin(), done.end(), &f) != done.end())
            continue;
        const D3D12_RESOURCE_DESC& d = f.first->desc;
        if (d.Width * 2 < renderW || d.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || d.MipLevels != 1 || d.DepthOrArraySize != 1 || d.SampleDesc.Count != 1)
            continue;
        const auto group = [&](int phase) {
            std::vector<const Flip*> out;
            for (const Flip& o : flips)
                if (isHistory(o) && ((o.presents[0] - flips.front().first->present) & 1) == (uint64_t) phase && o.first->desc.Format == d.Format &&
                    o.first->desc.Width == d.Width && o.first->desc.Height == d.Height)
                    out.push_back(&o);
            return out;
        };
        const std::vector<const Flip*> one = group(0), other = group(1);
        if (std::find(one.begin(), one.end(), &f) == one.end() && std::find(other.begin(), other.end(), &f) == other.end())
            continue; // not history
        if (one.size() != other.size())
        {
            if (isLighting(d) || isMap(d))
                balanced = false;
            continue;
        }
        // Partners by their place among their kind in the batch: the numbers mislead (o130 / o410, o129 / o412).
        for (size_t i = 0; i < one.size(); ++i)
        {
            HistPair pair;
            pair.a = one[i]->first->resource;
            pair.b = other[i]->first->resource;
            pair.desc = d;
            pair.lighting = isLighting(d) || isMap(d);
            pair.lightingIndex = isLighting(d) ? lightingPairs : -1;
            lightingPairs += isLighting(d) ? 1 : 0;
            mapPairs += isMap(d) ? 1 : 0;
            pairs.push_back(pair);
            done.push_back(one[i]);
            done.push_back(other[i]);
        }
    }
    if (balanced && mapRing.size() == 3)
    {
        HistPair pair;
        pair.a = mapRing[0]->first->resource;
        pair.b = mapRing[1]->first->resource;
        pair.c = mapRing[2]->first->resource;
        pair.desc = mapRing[0]->first->desc;
        pair.lighting = true;
        ++mapPairs;
        pairs.push_back(pair);
    }
    // The snapshots, likely packed depth and packed normals (by their look): R32_UINT at render size, one to UAV every other Present. Partners by their place in the batch.
    // Two pairs or none: a wrong set here must not take the lighting's test with it.
    int snapshotPairs = 0;
    {
        std::vector<const HistSeen*> phases[2];
        std::vector<const HistSeen*> counted;
        std::unordered_map<ID3D12Resource*, std::vector<uint64_t>> times;
        for (const HistSeen& seen : g_histSnapSeen)
            if (seen.desc.Width == renderW && seen.desc.Height == renderH && seen.desc.MipLevels == 1 && seen.desc.DepthOrArraySize == 1)
                times[seen.resource].push_back(seen.present);
        const uint64_t base = g_histSnapSeen.empty() ? 0 : g_histSnapSeen.front().present;
        std::string snapNearly;
        for (const HistSeen& seen : g_histSnapSeen)
        {
            const auto it = times.find(seen.resource);
            if (it == times.end() || std::find_if(counted.begin(), counted.end(), [&](const HistSeen* c) { return c->resource == seen.resource; }) != counted.end())
                continue;
            counted.push_back(&seen);
            const std::vector<uint64_t>& t = it->second;
            snapNearly += std::format(" [{}x{} list {:X} {} times]", seen.desc.Width, seen.desc.Height, seen.list, t.size());
            if (t.size() == 2 && t[1] == t[0] + 2)
                phases[(t[0] - base) & 1].push_back(&seen);
        }
        if (phases[0].size() == 2 && phases[1].size() == 2)
            for (size_t i = 0; i < 2; ++i)
            {
                HistPair pair;
                pair.a = phases[0][i]->resource;
                pair.b = phases[1][i]->resource;
                pair.desc = phases[0][i]->desc;
                pair.snapshot = true;
                pairs.push_back(pair);
                ++snapshotPairs;
            }
        else
            LOG_WARN("GI capture: history test: no snapshot pairs: {} and {} of the 2 and 2 wanted; R32_UINT textures going to UAV:{}", phases[0].size(), phases[1].size(), snapNearly);
    }
    // Steps 9 and 10. Each one pair or none, said by name: a wrong set here must not take the rest with it. Written one Present in two, the partners by their phase.
    const auto addExtra = [&](int extra, const char* name, const std::vector<HistSeen>& seenList, D3D12_BARRIER_LAYOUT before, D3D12_BARRIER_LAYOUT after) {
        std::vector<const HistSeen*> phases[2];
        std::vector<ID3D12Resource*> counted;
        const uint64_t base = seenList.empty() ? 0 : seenList.front().present;
        std::string extraNearly;
        for (const HistSeen& seen : seenList)
        {
            if (std::find(counted.begin(), counted.end(), seen.resource) != counted.end())
                continue;
            counted.push_back(seen.resource);
            std::vector<uint64_t> t;
            for (const HistSeen& o : seenList)
                if (o.resource == seen.resource && std::find(t.begin(), t.end(), o.present) == t.end())
                    t.push_back(o.present);
            extraNearly += std::format(" [{}x{} fmt {} list {:X} {} Presents]", seen.desc.Width, seen.desc.Height, (int) seen.desc.Format, seen.list, t.size());
            if (t.size() == 2 && t[1] == t[0] + 2)
                phases[(t[0] - base) & 1].push_back(&seen);
        }
        if (phases[0].size() != 1 || phases[1].size() != 1)
        {
            LOG_INFO("GI capture: history test: no {} pair: {} and {} of the 1 and 1 wanted:{}", name, phases[0].size(), phases[1].size(), extraNearly);
            return;
        }
        HistPair pair;
        pair.a = phases[0][0]->resource;
        pair.b = phases[1][0]->resource;
        pair.desc = phases[0][0]->desc;
        pair.extra = extra;
        pair.triggerBefore = before;
        pair.triggerAfter = after;
        pairs.push_back(pair);
    };
    addExtra(1, "moments", g_histMomentsSeen, D3D12_BARRIER_LAYOUT_RENDER_TARGET, D3D12_BARRIER_LAYOUT_UNORDERED_ACCESS);
    {
        std::vector<HistSeen> halfDepth;
        for (const HistSeen& seen : g_histSnapSeen)
            if (seen.desc.Width == renderW / 2 && seen.desc.Height == renderH / 2 && seen.desc.MipLevels == 1 && seen.desc.DepthOrArraySize == 1)
                halfDepth.push_back(seen);
        addExtra(2, "half-size depth", halfDepth, D3D12_BARRIER_LAYOUT_SHADER_RESOURCE, D3D12_BARRIER_LAYOUT_UNORDERED_ACCESS);
    }
    addExtra(3, "stencil", g_histStencilSeen, D3D12_BARRIER_LAYOUT_SHADER_RESOURCE, D3D12_BARRIER_LAYOUT_RENDER_TARGET);
    if (!balanced || lightingPairs != 2 || mapPairs != 1)
    {
        ++g_histRefused;
        g_histPairs.clear();
        g_histStoresReady = false;
        if (g_histAttempts <= 3)
            LOG_WARN("GI capture: history test: refused, attempt {}: render {}x{}, one map list a Present {}, {} lighting flips, {} lighting pairs (2 wanted), {} map pairs (1 wanted); every texture sent to SRV:{}",
                     g_histAttempts, renderW, renderH, oneMap, lightingFlips, lightingPairs, mapPairs, nearly);
        // A long menu or a sleep has no lighting to find: never stop looking, only look less often. A step of ours left silently off is a report on the plain game.
        g_histNextAttempt = g_present.load() + (g_histAttempts < kHistMaxAttempts ? kHistRetryPresents : kHistSlowRetryPresents);
        if (g_histWarnPending && g_histAttempts >= kHistWarnAttempts)
        {
            g_histWarnPending = false;
            LOG_WARN("GI capture: history test: switched on and the textures are not found: {} tries: it is NOT on", g_histAttempts);
            std::thread([] {
                for (int i = 0; i < 3; ++i)
                    Beep(220, 300);
            }).detach();
        }
        return;
    }
    std::string list;
    for (const HistPair& p : pairs)
        list += std::format(" [{}x{} fmt {}{}]", p.desc.Width, p.desc.Height, (int) p.desc.Format, p.c != nullptr ? " map of three" : p.lighting ? " lighting" : p.snapshot ? " snapshot" : p.extra == 1 ? " moments" : p.extra == 2 ? " half-size depth" : p.extra == 3 ? " stencil" : "");
    LOG_INFO("GI capture: history test: found {} pairs:{}", pairs.size(), list);
    g_histPairs = pairs;
    g_histStoresReady = false;
    g_histWarnPending = false;
    g_histLastFlip = g_present.load();
    g_histNextAttempt = 0;
    if (g_histDevice != nullptr && !g_histStoresMaking)
    {
        g_histStoresMaking = true;
        ID3D12Device* device = g_histDevice;
        g_histDevice = nullptr;
        std::thread(HistStoresOnWorker, device, std::move(pairs)).detach();
    }
}

// The map's trace (launched at some sizes the render-size R8 shows as three textures in a survey of 4, and which Presents is not logged). Read only: nothing here changes a barrier or a texture.
// From the first survey of each F8 or look-again, kMapTracePresents of every render-size R8 and R16G16B16A16_FLOAT, one character a Present: S to SRV, W to UAV or RT, B both, . neither. Holds g_histMutex.
constexpr uint64_t kMapTracePresents = 300;
constexpr int kMapTraceMax = 16;
constexpr size_t kMapTraceTextures = 48;
struct MapTraceEntry
{
    D3D12_RESOURCE_DESC desc {};
    D3D12_HEAP_PROPERTIES heap {};
    D3D12_HEAP_FLAGS heapFlags = D3D12_HEAP_FLAG_NONE;
    HRESULT heapResult = S_OK;
    uint64_t first = 0;
    uint64_t lists[6] = {};
    int listCount = 0;
    std::string marks;
};
std::unordered_map<ID3D12Resource*, MapTraceEntry> g_mapTrace; // identity only, no reference held; asked about only on a barrier naming it, in the game's list, so alive
uint64_t g_mapTraceStart = UINT64_MAX, g_mapTraceDropped = 0;
int g_mapTraces = 0;

void MapTraceBarrier(ID3D12GraphicsCommandList7* cmdList, const D3D12_TEXTURE_BARRIER& b, uint64_t present)
{
    if (g_mapTraceStart == UINT64_MAX || present < g_mapTraceStart || present - g_mapTraceStart >= kMapTracePresents)
        return;
    const bool toSrv = b.LayoutAfter == D3D12_BARRIER_LAYOUT_SHADER_RESOURCE;
    const bool toWrite = b.LayoutAfter == D3D12_BARRIER_LAYOUT_UNORDERED_ACCESS || b.LayoutAfter == D3D12_BARRIER_LAYOUT_RENDER_TARGET;
    if (!toSrv && !toWrite)
        return;
    auto it = g_mapTrace.find(b.pResource);
    if (it == g_mapTrace.end())
    {
        const D3D12_RESOURCE_DESC desc = b.pResource->GetDesc();
        if ((desc.Format != DXGI_FORMAT_R8_UNORM && desc.Format != DXGI_FORMAT_R16G16B16A16_FLOAT) || desc.Width != g_histRenderW.load() || desc.Height != g_histRenderH.load())
            return;
        if (g_mapTrace.size() >= kMapTraceTextures)
        {
            ++g_mapTraceDropped;
            return;
        }
        MapTraceEntry entry;
        entry.desc = desc;
        entry.first = present;
        entry.heapResult = b.pResource->GetHeapProperties(&entry.heap, &entry.heapFlags);
        entry.marks.assign(kMapTracePresents, '.');
        it = g_mapTrace.emplace(b.pResource, std::move(entry)).first;
    }
    MapTraceEntry& entry = it->second;
    char& mark = entry.marks[present - g_mapTraceStart];
    const char now = toSrv ? 'S' : 'W';
    mark = mark == '.' || mark == now ? now : 'B';
    const uint64_t list = (uint64_t) cmdList;
    if (std::find(entry.lists, entry.lists + entry.listCount, list) == entry.lists + entry.listCount && entry.listCount < 6)
        entry.lists[entry.listCount++] = list;
}

void MapTraceFlush()
{
    LOG_INFO("GI capture: map trace {} of {}: from present {} ({}), {} Presents, render {}x{}, {} textures, {} not recorded; S to SRV, W to UAV or RT, B both, . neither", g_mapTraces, kMapTraceMax, g_mapTraceStart,
             (g_mapTraceStart & 1) != 0 ? "odd" : "even", kMapTracePresents, g_histRenderW.load(), g_histRenderH.load(), g_mapTrace.size(), g_mapTraceDropped);
    std::vector<const std::pair<ID3D12Resource* const, MapTraceEntry>*> sorted;
    for (const auto& one : g_mapTrace)
        sorted.push_back(&one);
    std::sort(sorted.begin(), sorted.end(), [](const auto* a, const auto* b) {
        return a->second.desc.Format != b->second.desc.Format ? a->second.desc.Format > b->second.desc.Format : a->second.first != b->second.first ? a->second.first < b->second.first : a->first < b->first;
    });
    for (const auto* one : sorted)
    {
        const MapTraceEntry& e = one->second;
        std::string lists;
        for (int i = 0; i < e.listCount; ++i)
            lists += std::format(" {:X}", e.lists[i]);
        LOG_INFO("GI capture: map trace: {:X} fmt {} flags {:X} heap type {} flags {:X} result {:X}, first at +{}, lists{}: {}", (uint64_t) one->first, (int) e.desc.Format, (int) e.desc.Flags, (int) e.heap.Type, (int) e.heapFlags,
                 (uint32_t) e.heapResult, e.first - g_mapTraceStart, lists, e.marks);
    }
    g_mapTrace.clear();
    g_mapTraceDropped = 0;
    g_mapTraceStart = UINT64_MAX;
}

void HistStartSurvey(uint64_t present)
{
    if (g_histAttempts == 0 && g_mapTraceStart == UINT64_MAX && g_mapTraces < kMapTraceMax)
    {
        ++g_mapTraces;
        g_mapTraceStart = present + 1;
    }
    g_histSeen.clear();
    g_histWritten.clear();
    g_histSnapSeen.clear();
    g_histMomentsSeen.clear();
    g_histStencilSeen.clear();
    g_histPairs.clear();
    g_histStoresReady = false;
    g_histSurveying = true;
    // From the next Present: this one's head batch may be past, and half a frame would read as no flip.
    g_histSurveyStart = present + 1;
    g_histSurveyEnd = g_histSurveyStart + kHistSurveyPresents;
    ++g_histAttempts;
}

// The clip (F11): 180 Presents, 90 an eye, of a box in the middle of the view: the upscaled picture, the colour going in, both lighting pairs as their readers get them and the map.
// A crop is small enough to copy every frame, so what forms while the head moves can be watched. One readback buffer, made on a worker, freed when written; nothing of the game's is held.
constexpr int kClipFrames = 180;
// The same fraction of the view in every stream: 1024 of 3472 is 606 of 2056. Other render sizes crop a little differently, and index.txt says what was cut.
constexpr UINT kClipOutputSize = 1024, kClipInputSize = 606;
enum ClipStream
{
    ClipOutput,
    ClipColour,
    ClipFirst,
    ClipSecond,
    ClipMap,
    ClipStreams
};
constexpr const char* kClipNames[ClipStreams] = { "output", "colour", "first", "second", "map" };
constexpr uint64_t kClipStride[ClipStreams] = { 4ull << 20, 2ull << 20, 3ull << 20, 3ull << 20, 1ull << 20 };
struct ClipFormat
{
    bool ready = false, refused = false;
    DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;
    UINT width = 0, height = 0, rows = 0;
    UINT64 rowBytes = 0, rowPitch = 0;
    int recorded = 0, wrongFormat = 0;
};
std::mutex g_clipMutex; // holds no other lock; taken under g_histMutex
ClipFormat g_clipFormats[ClipStreams];
bool g_clipRecorded[ClipStreams][kClipFrames] = {};
std::vector<std::string> g_clipInfo;
ID3D12Resource* g_clipReadback = nullptr; // ours, freed by the writer
ID3D12Fence* g_clipFence = nullptr;
HANDLE g_clipFenceEvent = nullptr;
std::atomic<bool> g_clipBufferAsked { false }, g_clipBufferReady { false };
uint64_t g_clipArmed = 0, g_clipStart = 0, g_clipDrainFrom = 0;

uint64_t ClipOffset(int stream, int frame)
{
    uint64_t perFrame = 0, before = 0;
    for (int s = 0; s < ClipStreams; ++s)
    {
        perFrame += kClipStride[s];
        if (s < stream)
            before += kClipStride[s];
    }
    return perFrame * frame + before;
}

void ClipBufferOnWorker(ID3D12Device* device)
{
    D3D12_HEAP_PROPERTIES heap = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK);
    const D3D12_RESOURCE_DESC desc = CD3DX12_RESOURCE_DESC::Buffer(ClipOffset(0, kClipFrames));
    ID3D12Resource* readback = nullptr;
    const HRESULT made = device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&readback));
    bool ok = SUCCEEDED(made) && readback != nullptr;
    if (ok && g_clipFence == nullptr)
    {
        ok = SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&g_clipFence)));
        g_clipFenceEvent = ok ? CreateEventW(nullptr, FALSE, FALSE, nullptr) : nullptr;
        ok = ok && g_clipFenceEvent != nullptr;
    }
    device->Release();
    if (!ok)
    {
        LOG_WARN("GI capture: clip: no readback buffer of {} MB ({:X}): not taken", ClipOffset(0, kClipFrames) >> 20, (uint32_t) made);
        if (readback != nullptr)
            readback->Release();
        g_clipState.store(0);
        CamBeeps(1);
        return;
    }
    g_clipReadback = readback;
    g_clipBufferReady.store(true);
    LOG_INFO("GI capture: clip: readback buffer of {} MB made", ClipOffset(0, kClipFrames) >> 20);
}

// A centred box of `source` into its slot, in the list's order. Holds no lock on entry; the format of each stream is settled once. Returns false when refused.
bool ClipPlace(ID3D12GraphicsCommandList* cmdList, ID3D12Resource* source, int stream, int frame, UINT x, UINT y, UINT w, UINT h, D3D12_TEXTURE_COPY_LOCATION& dst, D3D12_BOX& box)
{
    if (frame < 0 || frame >= kClipFrames || g_clipReadback == nullptr)
        return false;
    const UINT want = stream == ClipOutput ? kClipOutputSize : kClipInputSize;
    const UINT cw = std::min(want, w), ch = std::min(want, h);
    D3D12_RESOURCE_DESC desc = source->GetDesc();
    std::lock_guard lock(g_clipMutex);
    ClipFormat& f = g_clipFormats[stream];
    if (!f.ready && !f.refused)
    {
        ID3D12Device* device = nullptr;
        if (FAILED(cmdList->GetDevice(IID_PPV_ARGS(&device))) || device == nullptr)
            return false;
        D3D12_RESOURCE_DESC cut = desc;
        cut.Width = cw;
        cut.Height = ch;
        cut.DepthOrArraySize = 1;
        cut.MipLevels = 1;
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp {};
        UINT rows = 0;
        UINT64 rowBytes = 0, total = 0;
        device->GetCopyableFootprints(&cut, 0, 1, 0, &fp, &rows, &rowBytes, &total);
        device->Release();
        if (total == 0 || total == UINT64_MAX || total > kClipStride[stream])
        {
            f.refused = true;
            LOG_WARN("GI capture: clip: {} refused: {}x{} format {} is {} bytes, the slot {}", kClipNames[stream], cw, ch, (int) desc.Format, total, kClipStride[stream]);
            return false;
        }
        f.ready = true;
        f.format = desc.Format;
        f.width = cw;
        f.height = ch;
        f.rows = rows;
        f.rowBytes = rowBytes;
        f.rowPitch = fp.Footprint.RowPitch;
        LOG_INFO("GI capture: clip: {} is {}x{} format {}", kClipNames[stream], cw, ch, (int) desc.Format);
    }
    if (!f.ready)
        return false;
    if (desc.Format != f.format || cw != f.width || ch != f.height)
    {
        ++f.wrongFormat;
        return false;
    }
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp {};
    fp.Offset = ClipOffset(stream, frame);
    fp.Footprint.Format = f.format;
    fp.Footprint.Width = f.width;
    fp.Footprint.Height = f.height;
    fp.Footprint.Depth = 1;
    fp.Footprint.RowPitch = (UINT) f.rowPitch;
    dst = CD3DX12_TEXTURE_COPY_LOCATION(g_clipReadback, fp);
    box = { x + (w - cw) / 2, y + (h - ch) / 2, 0, x + (w - cw) / 2 + cw, y + (h - ch) / 2 + ch, 1 };
    if (!g_clipRecorded[stream][frame])
        ++f.recorded;
    g_clipRecorded[stream][frame] = true;
    return true;
}

// In front of the game's barrier, from the state it declares and back to it.
void ClipCopyEnhanced(ID3D12GraphicsCommandList7* cmdList, const D3D12_TEXTURE_BARRIER& b, int stream, int frame)
{
    const D3D12_RESOURCE_DESC desc = b.pResource->GetDesc();
    D3D12_TEXTURE_COPY_LOCATION dst {};
    D3D12_BOX box {};
    if (!ClipPlace(cmdList, b.pResource, stream, frame, 0, 0, (UINT) desc.Width, desc.Height, dst, box))
        return;
    HistBarrier(cmdList, b.pResource, b.SyncBefore, b.AccessBefore, b.LayoutBefore, D3D12_BARRIER_SYNC_COPY, D3D12_BARRIER_ACCESS_COPY_SOURCE, D3D12_BARRIER_LAYOUT_COPY_SOURCE);
    const D3D12_TEXTURE_COPY_LOCATION src = CD3DX12_TEXTURE_COPY_LOCATION(b.pResource, 0);
    cmdList->CopyTextureRegion(&dst, 0, 0, 0, &src, &box);
    HistBarrier(cmdList, b.pResource, D3D12_BARRIER_SYNC_COPY, D3D12_BARRIER_ACCESS_COPY_SOURCE, D3D12_BARRIER_LAYOUT_COPY_SOURCE, b.SyncBefore, b.AccessBefore, b.LayoutBefore);
}

// A lighting texture on its way to its readers, as they get it: behind our swap, if there is one.
void ClipOnFlip(ID3D12GraphicsCommandList7* cmdList, const D3D12_TEXTURE_BARRIER& b, const HistPair& pair)
{
    if (g_clipState.load(std::memory_order_relaxed) != 2 || !pair.lighting || pair.lightingIndex >= 2)
        return;
    const int stream = pair.lightingIndex == 0 ? ClipFirst : pair.lightingIndex == 1 ? ClipSecond : ClipMap;
    ClipCopyEnhanced(cmdList, b, stream, (int) (g_present.load() - g_clipStart));
}

// The upscaled picture: an 8-bit target past the render size, leaving its write. A Present may write it twice: the slot takes the last to run.
void ClipOnBarrier(ID3D12GraphicsCommandList7* cmdList, UINT32 count, const D3D12_BARRIER_GROUP* groups)
{
    if (g_clipState.load(std::memory_order_relaxed) != 2 || cmdList->GetType() != D3D12_COMMAND_LIST_TYPE_DIRECT)
        return;
    const UINT render = g_histRenderW.load();
    const int frame = (int) (g_present.load() - g_clipStart);
    for (UINT32 g = 0; g < count; ++g)
    {
        if (groups[g].Type != D3D12_BARRIER_TYPE_TEXTURE)
            continue;
        for (UINT32 i = 0; i < groups[g].NumBarriers; ++i)
        {
            const D3D12_TEXTURE_BARRIER& b = groups[g].pTextureBarriers[i];
            if (b.pResource == nullptr || b.LayoutBefore == b.LayoutAfter || (b.LayoutBefore != D3D12_BARRIER_LAYOUT_RENDER_TARGET && b.LayoutBefore != D3D12_BARRIER_LAYOUT_UNORDERED_ACCESS))
                continue;
            const D3D12_RESOURCE_DESC desc = b.pResource->GetDesc();
            if (render != 0 && desc.Width * 2 >= (UINT64) render * 3 && (desc.Format == DXGI_FORMAT_R8G8B8A8_UNORM || desc.Format == DXGI_FORMAT_R8G8B8A8_UNORM_SRGB || desc.Format == DXGI_FORMAT_R8G8B8A8_TYPELESS))
                ClipCopyEnhanced(cmdList, b, ClipOutput, frame);
        }
    }
}

void ClipOnEvaluate(ID3D12GraphicsCommandList* cmdList, int featureId, NVSDK_NGX_Parameter* params)
{
    const int state = g_clipState.load(std::memory_order_relaxed);
    if (state == 0 || cmdList == nullptr || params == nullptr || cmdList->GetType() != D3D12_COMMAND_LIST_TYPE_DIRECT)
        return;
    if (state == 1 && !g_clipBufferAsked.exchange(true))
    {
        ID3D12Device* device = nullptr;
        if (SUCCEEDED(cmdList->GetDevice(IID_PPV_ARGS(&device))) && device != nullptr)
            std::thread(ClipBufferOnWorker, device).detach();
    }
    if (state != 2)
        return;
    const uint64_t present = g_present.load();
    const int frame = (int) (present - g_clipStart);
    ID3D12Resource* colour = GetResource(params, NVSDK_NGX_Parameter_Color);
    if (colour == nullptr || frame < 0 || frame >= kClipFrames)
        return;
    {
        std::lock_guard lock(g_clipMutex);
        if (g_clipRecorded[ClipColour][frame])
            return;
    }
    const UINT w = GetUInt(params, NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Width), h = GetUInt(params, NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Height);
    const UINT x = GetUInt(params, NVSDK_NGX_Parameter_DLSS_Input_Color_Subrect_Base_X), y = GetUInt(params, NVSDK_NGX_Parameter_DLSS_Input_Color_Subrect_Base_Y);
    D3D12_TEXTURE_COPY_LOCATION dst {};
    D3D12_BOX box {};
    t_inside = true;
    if (w != 0 && h != 0 && ClipPlace(cmdList, colour, ClipColour, frame, x, y, w, h, dst, box))
    {
        const D3D12_TEXTURE_COPY_LOCATION src = CD3DX12_TEXTURE_COPY_LOCATION(colour, 0);
        D3D12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::Transition(colour, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_SOURCE, 0);
        cmdList->ResourceBarrier(1, &barrier);
        cmdList->CopyTextureRegion(&dst, 0, 0, 0, &src, &box);
        std::swap(barrier.Transition.StateBefore, barrier.Transition.StateAfter);
        cmdList->ResourceBarrier(1, &barrier);
    }
    t_inside = false;
    CameraLine camera;
    {
        std::lock_guard lock(g_mutex);
        camera = g_lastCamera;
    }
    LARGE_INTEGER qpc {}, frequency {};
    QueryPerformanceCounter(&qpc);
    QueryPerformanceFrequency(&frequency);
    const auto list = [](const float* v, int n) {
        std::string out;
        for (int i = 0; i < n; ++i)
            out += std::format("{}{}", i == 0 ? "" : " ", v[i]);
        return out;
    };
    std::lock_guard lock(g_clipMutex);
    g_clipInfo.push_back(std::format("frame {} present {} parity {} feature {} history step {} camera effect {} queues {} ms {:.3f} row {} jitter {} position {} forward {} up {} right {} near {} far {} viewToClip {} clipToPrevClip {}", frame, present, present & 1, featureId,
                                     g_histMode.load(), g_camEffect.load(), g_camQueues.load(), 1000.0 * (double) qpc.QuadPart / (double) frequency.QuadPart, camera.clipToPrevClip[8], list(camera.jitter, 2), list(camera.pos, 3), list(camera.fwd, 3), list(camera.up, 3),
                                     list(camera.right, 3), camera.nearZ, camera.farZ, list(camera.viewToClip, 16), list(camera.clipToPrevClip, 16)));
}

void ClipWriteOnWorker(uint64_t fenceValue)
{
    const uint64_t startTick = GetTickCount64();
    bool completed = g_clipFence->GetCompletedValue() >= fenceValue;
    if (!completed && SUCCEEDED(g_clipFence->SetEventOnCompletion(fenceValue, g_clipFenceEvent)))
        completed = WaitForSingleObject(g_clipFenceEvent, 10000) == WAIT_OBJECT_0;
    Sleep(500);
    if (!completed)
        LOG_WARN("GI capture: clip: the fence did not complete in 10 s; writing anyway");
    char stamp[32] {};
    const std::time_t now = std::time(nullptr);
    std::tm local {};
    localtime_s(&local, &now);
    std::strftime(stamp, sizeof(stamp), "%Y%m%d-%H%M%S", &local);
    const auto folder = Util::ExePath().parent_path() / "OptiCapture" / (std::string(stamp) + "-clip");
    std::error_code ec;
    std::filesystem::create_directories(folder, ec);
    std::ofstream index(folder / "index.txt");
    index << "GI clip, " << kClipFrames << " Presents\n";
    void* mapped = nullptr;
    const D3D12_RANGE range { 0, (SIZE_T) ClipOffset(0, kClipFrames) };
    int written = 0;
    std::string counts;
    {
        std::lock_guard lock(g_clipMutex);
        for (const std::string& line : g_clipInfo)
            index << line << "\n";
        if (SUCCEEDED(g_clipReadback->Map(0, &range, &mapped)) && mapped != nullptr)
        {
            for (int s = 0; s < ClipStreams; ++s)
            {
                const ClipFormat& f = g_clipFormats[s];
                counts += std::format(" {} {} of {}, wrong format {}{};", kClipNames[s], f.recorded, kClipFrames, f.wrongFormat, f.refused ? ", refused" : "");
                if (!f.ready)
                    continue;
                index << kClipNames[s] << " " << f.width << "x" << f.height << " format " << (int) f.format << " recorded " << f.recorded << "\n";
                for (int frame = 0; frame < kClipFrames; ++frame)
                {
                    if (!g_clipRecorded[s][frame])
                        continue;
                    uint32_t header[32] = {};
                    header[0] = 0x20534444; // "DDS "
                    header[1] = 124;
                    header[2] = 0x1 | 0x2 | 0x4 | 0x8 | 0x1000; // caps, height, width, pitch, pixel format
                    header[3] = f.height;
                    header[4] = f.width;
                    header[5] = (uint32_t) f.rowBytes;
                    header[7] = 1;
                    header[19] = 32;
                    header[20] = 0x4;        // FOURCC
                    header[21] = 0x30315844; // "DX10"
                    header[27] = 0x1000;
                    const uint32_t dx10[5] = { (uint32_t) ReadableFormat(f.format), 3 /* TEXTURE2D */, 0, 1, 0 };
                    std::ofstream file(folder / std::format("f{:03}_{}.dds", frame, kClipNames[s]), std::ios::binary);
                    file.write(reinterpret_cast<const char*>(header), sizeof(header));
                    file.write(reinterpret_cast<const char*>(dx10), sizeof(dx10));
                    const char* row = static_cast<const char*>(mapped) + ClipOffset(s, frame);
                    for (UINT y = 0; y < f.rows; ++y, row += f.rowPitch)
                        file.write(row, (std::streamsize) f.rowBytes);
                    ++written;
                }
            }
            const D3D12_RANGE none { 0, 0 };
            g_clipReadback->Unmap(0, &none);
        }
        else
            LOG_WARN("GI capture: clip: mapping the readback failed: nothing written");
        for (ClipFormat& f : g_clipFormats)
            f = ClipFormat {};
        std::memset(g_clipRecorded, 0, sizeof(g_clipRecorded));
        g_clipInfo.clear();
    }
    index.close();
    g_clipReadback->Release();
    g_clipReadback = nullptr;
    g_clipBufferReady.store(false);
    g_clipBufferAsked.store(false);
    LOG_INFO("GI capture: clip: {} files to {} ({} ms on the worker):{}", written, folder.string(), GetTickCount64() - startTick, counts);
    g_clipState.store(0);
    CamBeeps(3);
}

void ClipOnPresent(uint64_t present, ID3D12CommandQueue* queue)
{
    const int state = g_clipState.load();
    if (state == 1)
    {
        bool pairs = false;
        {
            std::lock_guard lock(g_histMutex);
            pairs = !g_histSurveying && !g_histPairs.empty();
        }
        // No pairs in a second: the picture and the colour are still worth having, and it says so.
        const bool late = present >= g_clipArmed + 90;
        if (g_clipBufferReady.load() && (pairs || late))
        {
            if (!pairs)
                LOG_WARN("GI capture: clip: no lighting pairs found: the picture and the colour only");
            g_clipStart = present;
            g_clipState.store(2);
            LOG_INFO("GI capture: clip: present {}: running", present);
            CamBeeps(2);
        }
        else if (present >= g_clipArmed + 600)
        {
            LOG_WARN("GI capture: clip: no readback buffer after 600 Presents (no evaluate seen?): not taken");
            g_clipBufferAsked.store(false);
            g_clipState.store(0);
            CamBeeps(1);
        }
    }
    else if (state == 2)
    {
        // Each frame's bracket, as recorded: a clip where it went missing on three frames is another clip.
        if (present > g_clipStart && present <= g_clipStart + kClipFrames)
        {
            bool opened = false, closed = false;
            {
                std::lock_guard lock(g_histMutex);
                opened = g_histOpened == present - 1;
                closed = g_histClosed == present - 1;
            }
            std::lock_guard lock(g_clipMutex);
            g_clipInfo.push_back(std::format("bracket {} opened {} closed {}", present - 1 - g_clipStart, opened ? 1 : 0, closed ? 1 : 0));
        }
        if (present >= g_clipStart + kClipFrames)
        {
            g_clipDrainFrom = present;
            g_clipState.store(3);
        }
    }
    else if (state == 3 && queue != nullptr && present >= g_clipDrainFrom + kDrainPresents)
    {
        static uint64_t fenceValue = 0;
        ++fenceValue;
        if (FAILED(queue->Signal(g_clipFence, fenceValue)))
            LOG_WARN("GI capture: clip: signalling the fence failed");
        g_clipState.store(4);
        std::thread(ClipWriteOnWorker, fenceValue).detach();
    }
}

// F8's step 8: the bracket, and the motion texture same-eye for the accumulation alone. The lighting pass reprojects some classes of pixel through g_velocity, which holds the game's previous camera:
// on same-eye history that drags them an eye step every frame. Our own list runs between the barrier list in front of the lighting's list and that list: the texture > our copy, then
// the copy less (the game's matrix - the same eye's) on the pixel's depth > the texture. The copy goes back behind the accumulation, in the game's list, before PrepareUpscaleCS: the reservoirs before it and DLSS after it see the game's.
// That copy back is recorded before we know whether our list can correct, so our list always makes the copy; with nothing to correct by, it does only that.
constexpr int kVelSlots = 8;
std::atomic<int> g_velState { 0 }; // 0 nothing, 1 being made, 2 ready, 3 failed for good
std::atomic<ID3D12GraphicsCommandList7*> g_velArmedList { nullptr }; // identity only: the list with the copy back, not yet submitted
std::atomic<uint64_t> g_velSubmitSeen { UINT64_MAX }; // the Present of the last ExecuteCommandLists seen: with none, a copy back would put an empty store over the game's motion
bool g_velStopped = false; // a copy back's list went unsubmitted: no more until the next F8
ID3D12RootSignature* g_velRoot = nullptr;
ID3D12PipelineState* g_velPso = nullptr;
ID3D12DescriptorHeap* g_velHeap = nullptr;
ID3D12CommandAllocator* g_velAllocators[kVelSlots] = {};
ID3D12GraphicsCommandList7* g_velList = nullptr;
ID3D12Fence* g_velFence = nullptr;
HANDLE g_velEvent = nullptr;
ID3D12Device* g_velDevice = nullptr; // a reference of ours
ID3D12Resource* g_velStore = nullptr; // ours, for the life of the game: a list in flight may use it
ID3D12Resource* g_velOwn[2] = {}; // each eye's own share of the motion, by the Present's parity; between frames in SRV
bool g_velOwnUsed[2] = {};
uint64_t g_velOwnPresent = UINT64_MAX;
D3D12_RESOURCE_DESC g_velStoreDesc {};
bool g_velStoreUsed = false;
uint64_t g_velFenceValue = 0, g_velUsedAt[kVelSlots] = {}, g_velRecorded = 0;
struct VelStamp
{
    uint64_t present = UINT64_MAX;
    ID3D12Resource* resource = nullptr; // identity only
    D3D12_RESOURCE_DESC desc {}; // as it was when seen: the pointer may be two Presents old
};
VelStamp g_velByParity[2];
ID3D12Resource* g_velTarget = nullptr; // identity only: the texture the copy back was recorded for
ID3D12Resource* g_velDepth = nullptr;  // identity only, this Present
uint64_t g_velDepthPresent = UINT64_MAX, g_velArmedPresent = UINT64_MAX;
int g_velDepthFirstLayout = 0;
uint64_t g_velStamps = 0, g_velArmed = 0, g_velByStamp = 0, g_velByFlip = 0, g_velNotArmed = 0, g_velCorrected = 0, g_velSaveOnly = 0, g_velWrongTarget = 0, g_velNoMatrix = 0, g_velNoDepth = 0, g_velNotFound = 0, g_velWaits = 0, g_velNoSubmits = 0, g_velOwnDoubled = 0, g_velOwnAveraged = 0;
int g_velReasonLines = 0;

void VelReason(const char* reason)
{
    if (g_velReasonLines++ < 12)
        LOG_INFO("GI capture: motion for the accumulation: {}", reason);
}

void VelMakeOnWorker(ID3D12Device* device, D3D12_RESOURCE_DESC like)
{
    const auto failed = [&](const char* what, HRESULT hr) {
        LOG_WARN("GI capture: motion for the accumulation: {} failed: {:X}", what, (uint32_t) hr);
        device->Release();
        g_velState.store(3);
    };
    bool firstTime;
    {
        std::lock_guard lock(g_histMutex);
        firstTime = g_velPso == nullptr;
    }
    if (firstTime)
    {
        // The vector stored is half the clip-space step: x = (previous clip x - clip x) / 2, y the same. Sign and scale as the game's own passes store and read it (x 2, then x .5 to uv).
        static const char shader[] = R"(
cbuffer Constants : register(b0)
{
    float4 game[4];
    float4 same[4];
    float2 invSize;
    float ownExtra;
    float ownFrom;
    float ownWidth;
    float ownAverage;
};
Texture2D<float2> Motion : register(t0);
Texture2D<float> Depth : register(t1);
Texture2D<float2> OwnPrev : register(t2);
RWTexture2D<float2> Out : register(u0);
RWTexture2D<float2> OwnOut : register(u1);
[numthreads(8, 8, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    const float x = (id.x + 0.5) * invSize.x * 2.0 - 1.0;
    const float y = 1.0 - (id.y + 0.5) * invSize.y * 2.0;
    const float d = Depth.Load(int3(id.xy, 0));
    const float4 g = game[0] * x + game[1] * y + game[2] * d + game[3];
    const float4 s = same[0] * x + same[1] * y + same[2] * d + same[3];
    const float2 motion = Motion.Load(int3(id.xy, 0));
    const float2 previous = g.xy / g.w;
    const float2 own = motion - 0.5 * (previous - float2(x, y)); // what the camera does not explain: the thing's own movement, one Present's
    OwnOut[id.xy] = own;
    float2 used = own;
    if (ownAverage > 0.5)
    {
        // Clothing's vector holds about 96% of the eye step, so its 'own' flips with the eye: real motion does not. With the other eye's, at where this pixel was, the flip cancels.
        const int2 last = int2(1.0 / invSize) - 1;
        const float2 other = OwnPrev.Load(int3(clamp(int2((previous.x * 0.5 + 0.5) / invSize.x, (0.5 - previous.y * 0.5) / invSize.y), int2(0, 0), last), 0));
        if (length(other - own) / invSize.x < 8.0) // another surface there: a hole
            used = 0.5 * (own + other);
    }
    Out[id.xy] = 0.5 * (s.xy / s.w - float2(x, y)) + used + ownExtra * used * saturate((length(used) / invSize.x - ownFrom) / ownWidth); // none below ownFrom px, all from ownFrom + ownWidth: still pixels' noise is not doubled
}
)";
        ID3DBlob* cs = nullptr;
        ID3DBlob* errors = nullptr;
        if (FAILED(D3DCompile(shader, sizeof(shader) - 1, "GiCaptureMotion", nullptr, nullptr, "CSMain", "cs_5_0", D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &cs, &errors)))
        {
            LOG_WARN("GI capture: motion for the accumulation: compile: {}", errors != nullptr ? (const char*) errors->GetBufferPointer() : "no message");
            device->Release();
            g_velState.store(3); // final: never tried again
            return;
        }
        D3D12_DESCRIPTOR_RANGE range[2] {};
        range[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        range[0].NumDescriptors = 3;
        range[0].OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;
        range[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
        range[1].NumDescriptors = 2;
        range[1].OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;
        D3D12_ROOT_PARAMETER params[2] {};
        params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        params[0].Constants.Num32BitValues = 38;
        params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        params[1].DescriptorTable.NumDescriptorRanges = 2;
        params[1].DescriptorTable.pDescriptorRanges = range;
        D3D12_ROOT_SIGNATURE_DESC rootDesc {};
        rootDesc.NumParameters = 2;
        rootDesc.pParameters = params;
        ID3DBlob* rootBlob = nullptr;
        HRESULT hr = D3D12SerializeRootSignature(&rootDesc, D3D_ROOT_SIGNATURE_VERSION_1, &rootBlob, &errors);
        if (FAILED(hr))
            return failed("root signature", hr);
        ID3D12RootSignature* root = nullptr;
        hr = device->CreateRootSignature(0, rootBlob->GetBufferPointer(), rootBlob->GetBufferSize(), IID_PPV_ARGS(&root));
        rootBlob->Release();
        if (FAILED(hr))
            return failed("root signature made", hr);
        D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc {};
        psoDesc.pRootSignature = root;
        psoDesc.CS = { cs->GetBufferPointer(), cs->GetBufferSize() };
        ID3D12PipelineState* pso = nullptr;
        hr = device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&pso));
        cs->Release();
        if (FAILED(hr))
            return failed("pipeline", hr);
        D3D12_DESCRIPTOR_HEAP_DESC heapDesc {};
        heapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        heapDesc.NumDescriptors = 5 * kVelSlots;
        heapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        ID3D12DescriptorHeap* heap = nullptr;
        hr = device->CreateDescriptorHeap(&heapDesc, IID_PPV_ARGS(&heap));
        if (FAILED(hr))
            return failed("heap", hr);
        ID3D12CommandAllocator* allocators[kVelSlots] = {};
        for (auto& allocator : allocators)
        {
            hr = device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator));
            if (FAILED(hr))
                return failed("allocator", hr);
        }
        ID3D12GraphicsCommandList7* list = nullptr;
        hr = device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocators[0], nullptr, IID_PPV_ARGS(&list));
        if (FAILED(hr))
            return failed("list", hr);
        list->Close();
        ID3D12Fence* fence = nullptr;
        hr = device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));
        if (FAILED(hr))
            return failed("fence", hr);
        std::lock_guard lock(g_histMutex);
        g_velRoot = root;
        g_velPso = pso;
        g_velHeap = heap;
        std::copy(std::begin(allocators), std::end(allocators), g_velAllocators);
        g_velList = list;
        g_velFence = fence;
        g_velEvent = CreateEventA(nullptr, FALSE, FALSE, nullptr);
        device->AddRef();
        g_velDevice = device;
    }
    ID3D12Resource* store = HistMakeStore(device, like);
    ID3D12Resource* ownA = HistMakeStore(device, like, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    ID3D12Resource* ownB = HistMakeStore(device, like, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    device->Release();
    if (store == nullptr || ownA == nullptr || ownB == nullptr)
    {
        g_velState.store(3);
        return;
    }
    {
        std::lock_guard lock(g_histMutex);
        g_velOwn[0] = ownA;
        g_velOwn[1] = ownB;
        g_velOwnUsed[0] = g_velOwnUsed[1] = false;
        g_velOwnPresent = UINT64_MAX;
        g_velStore = store; // an older one stays: nothing is released while a list may hold it
        g_velStoreDesc = like;
        g_velStoreUsed = false;
    }
    LOG_INFO("GI capture: motion for the accumulation: ready, {}x{} format {}", like.Width, like.Height, (int) like.Format);
    g_velState.store(2);
}

// From the barrier hook, on every game's barrier at step 8. Holds g_histMutex.
void VelOnBarrier(ID3D12GraphicsCommandList7* cmdList, const D3D12_TEXTURE_BARRIER& b, uint64_t present)
{
    if (b.LayoutBefore == D3D12_BARRIER_LAYOUT_RENDER_TARGET && b.LayoutAfter == D3D12_BARRIER_LAYOUT_UNORDERED_ACCESS)
    {
        const D3D12_RESOURCE_DESC desc = b.pResource->GetDesc();
        if (desc.Format != DXGI_FORMAT_R32G32_FLOAT || desc.Width != g_histRenderW.load() || desc.Height != g_histRenderH.load())
            return;
        VelStamp& stamp = g_velByParity[present & 1];
        stamp.present = present;
        stamp.resource = b.pResource;
        stamp.desc = desc;
        ++g_velStamps;
        int expected = 0;
        if (g_velStoreDesc.Width != desc.Width || g_velStoreDesc.Height != desc.Height)
            g_velState.compare_exchange_strong(expected = 2, 0);
        expected = 0;
        if (g_velState.compare_exchange_strong(expected, 1))
        {
            ID3D12Device* device = nullptr;
            if (SUCCEEDED(cmdList->GetDevice(IID_PPV_ARGS(&device))) && device != nullptr)
                std::thread(VelMakeOnWorker, device, desc).detach();
            else
                g_velState.store(3);
        }
        return;
    }
    // The depth this frame's lighting list goes on with: the frame's own, and its first barrier there says what it is when our list runs in front.
    if (g_histBatchPresent != present || g_histBatchList != cmdList || g_velDepthPresent == present)
        return;
    const auto isDepth = [](D3D12_BARRIER_LAYOUT l) { return l == D3D12_BARRIER_LAYOUT_DEPTH_STENCIL_WRITE || l == D3D12_BARRIER_LAYOUT_DEPTH_STENCIL_READ; };
    if (!isDepth(b.LayoutBefore) && !isDepth(b.LayoutAfter))
        return;
    const D3D12_RESOURCE_DESC desc = b.pResource->GetDesc();
    if ((desc.Flags & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL) == 0 || desc.Width != g_histRenderW.load() || desc.Height != g_histRenderH.load())
        return;
    g_velDepthPresent = present;
    g_velDepth = b.pResource;
    g_velDepthFirstLayout = (int) b.LayoutBefore;
}

// Behind the accumulation, on the lighting's list. Holds g_histMutex.
void VelCopyBack(ID3D12GraphicsCommandList7* cmdList, uint64_t present)
{
    if (g_velState.load() != 2 || g_velStore == nullptr || g_velStopped)
    {
        ++g_velNotArmed;
        return;
    }
    const uint64_t seen = g_velSubmitSeen.load(std::memory_order_relaxed);
    if (seen == UINT64_MAX || seen + 2 < present)
    {
        ++g_velNotArmed;
        ++g_velNoSubmits;
        VelReason("no ExecuteCommandLists seen: nothing touched");
        return;
    }
    // The list with the lighting may be recorded before the one that writes the motion: then this Present's is the one two back, by the flip.
    const VelStamp& stamp = g_velByParity[present & 1];
    if (stamp.resource == nullptr || (stamp.present != present && stamp.present + 2 != present))
    {
        ++g_velNotArmed;
        VelReason("no motion texture written this Present or two back: nothing touched");
        return;
    }
    const D3D12_RESOURCE_DESC& desc = stamp.desc;
    if (desc.Width != g_velStoreDesc.Width || desc.Height != g_velStoreDesc.Height || desc.Format != g_velStoreDesc.Format)
    {
        ++g_velNotArmed;
        VelReason("the motion texture's size changed: nothing touched");
        return;
    }
    ++(stamp.present == present ? g_velByStamp : g_velByFlip);
    HistBarrier(cmdList, stamp.resource, D3D12_BARRIER_SYNC_ALL, D3D12_BARRIER_ACCESS_SHADER_RESOURCE, D3D12_BARRIER_LAYOUT_SHADER_RESOURCE, D3D12_BARRIER_SYNC_COPY, D3D12_BARRIER_ACCESS_COPY_DEST,
                D3D12_BARRIER_LAYOUT_COPY_DEST);
    HistCopy(cmdList, g_velStore, stamp.resource);
    HistBarrier(cmdList, stamp.resource, D3D12_BARRIER_SYNC_COPY, D3D12_BARRIER_ACCESS_COPY_DEST, D3D12_BARRIER_LAYOUT_COPY_DEST, D3D12_BARRIER_SYNC_ALL, D3D12_BARRIER_ACCESS_SHADER_RESOURCE,
                D3D12_BARRIER_LAYOUT_SHADER_RESOURCE);
    g_velTarget = stamp.resource;
    g_velArmedPresent = present;
    ++g_velArmed;
    if (g_velArmedList.exchange(cmdList, std::memory_order_relaxed) != nullptr)
    {
        ++g_velNotFound;
        g_velStopped = true;
        VelReason("a list with the copy back was never submitted: stopped until the next F8");
    }
}

// At the submit. Holds g_histMutex. The store's layout between frames: copy source, which the copy back needs.
void VelRecord(ID3D12CommandQueue* queue)
{
    const uint64_t present = g_velArmedPresent;
    ID3D12Resource* target = g_velTarget;
    float game[16], same[16];
    bool matrices;
    {
        std::lock_guard lock(g_camMutex);
        matrices = g_camSplitPresent == present;
        if (matrices)
        {
            std::memcpy(game, g_camGameMatrix, sizeof(game));
            std::memcpy(same, g_camSameMatrix, sizeof(same));
        }
    }
    const VelStamp& stamp = g_velByParity[present & 1];
    bool correct = true;
    if (stamp.present != present || stamp.resource != target)
    {
        ++g_velWrongTarget;
        VelReason("the texture written this Present is not the one the flip said: only the copy");
        correct = false;
    }
    if (!matrices)
    {
        ++g_velNoMatrix;
        VelReason("no camera block for this Present: only the copy");
        correct = false;
    }
    ID3D12Resource* depth = g_velDepthPresent == present ? g_velDepth : nullptr;
    if (depth == nullptr || g_velDepthFirstLayout != (int) D3D12_BARRIER_LAYOUT_SHADER_RESOURCE)
    {
        ++g_velNoDepth;
        VelReason("the lighting's list holds no depth that starts in SRV: only the copy");
        correct = false;
    }

    // The value signalled here covers every list sent before it; a slot is free once the value after its use is reached.
    const int slot = (int) (g_velRecorded++ % kVelSlots);
    queue->Signal(g_velFence, ++g_velFenceValue);
    if (g_velFence->GetCompletedValue() < g_velUsedAt[slot])
    {
        ++g_velWaits;
        g_velFence->SetEventOnCompletion(g_velUsedAt[slot], g_velEvent);
        WaitForSingleObject(g_velEvent, 1000);
    }
    g_velUsedAt[slot] = g_velFenceValue + 1;
    g_velAllocators[slot]->Reset();
    ID3D12GraphicsCommandList7* list = g_velList;
    list->Reset(g_velAllocators[slot], g_velPso);
    HistBarrier(list, target, D3D12_BARRIER_SYNC_ALL, D3D12_BARRIER_ACCESS_SHADER_RESOURCE, D3D12_BARRIER_LAYOUT_SHADER_RESOURCE, D3D12_BARRIER_SYNC_COPY, D3D12_BARRIER_ACCESS_COPY_SOURCE,
                D3D12_BARRIER_LAYOUT_COPY_SOURCE);
    if (g_velStoreUsed)
        HistBarrier(list, g_velStore, D3D12_BARRIER_SYNC_ALL, D3D12_BARRIER_ACCESS_COPY_SOURCE, D3D12_BARRIER_LAYOUT_COPY_SOURCE, D3D12_BARRIER_SYNC_COPY, D3D12_BARRIER_ACCESS_COPY_DEST,
                    D3D12_BARRIER_LAYOUT_COPY_DEST);
    else
        HistBarrier(list, g_velStore, D3D12_BARRIER_SYNC_NONE, D3D12_BARRIER_ACCESS_NO_ACCESS, D3D12_BARRIER_LAYOUT_COMMON, D3D12_BARRIER_SYNC_COPY, D3D12_BARRIER_ACCESS_COPY_DEST,
                    D3D12_BARRIER_LAYOUT_COPY_DEST);
    g_velStoreUsed = true;
    HistCopy(list, target, g_velStore);
    if (correct)
    {
        HistBarrier(list, g_velStore, D3D12_BARRIER_SYNC_COPY, D3D12_BARRIER_ACCESS_COPY_DEST, D3D12_BARRIER_LAYOUT_COPY_DEST, D3D12_BARRIER_SYNC_COMPUTE_SHADING, D3D12_BARRIER_ACCESS_SHADER_RESOURCE,
                    D3D12_BARRIER_LAYOUT_SHADER_RESOURCE);
        HistBarrier(list, target, D3D12_BARRIER_SYNC_COPY, D3D12_BARRIER_ACCESS_COPY_SOURCE, D3D12_BARRIER_LAYOUT_COPY_SOURCE, D3D12_BARRIER_SYNC_COMPUTE_SHADING, D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
                    D3D12_BARRIER_LAYOUT_UNORDERED_ACCESS);
        const UINT step = g_velDevice->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        D3D12_CPU_DESCRIPTOR_HANDLE cpu = g_velHeap->GetCPUDescriptorHandleForHeapStart();
        D3D12_GPU_DESCRIPTOR_HANDLE gpu = g_velHeap->GetGPUDescriptorHandleForHeapStart();
        cpu.ptr += (SIZE_T) slot * 5 * step;
        gpu.ptr += (UINT64) slot * 5 * step;
        ID3D12Resource* ownWrite = g_velOwn[present & 1];
        ID3D12Resource* ownRead = g_velOwn[(present & 1) ^ 1];
        for (int i = 0; i < 2; ++i)
            if (!g_velOwnUsed[i])
            {
                HistBarrier(list, g_velOwn[i], D3D12_BARRIER_SYNC_NONE, D3D12_BARRIER_ACCESS_NO_ACCESS, D3D12_BARRIER_LAYOUT_COMMON, D3D12_BARRIER_SYNC_COMPUTE_SHADING, D3D12_BARRIER_ACCESS_SHADER_RESOURCE, D3D12_BARRIER_LAYOUT_SHADER_RESOURCE);
                g_velOwnUsed[i] = true;
            }
        HistBarrier(list, ownWrite, D3D12_BARRIER_SYNC_COMPUTE_SHADING, D3D12_BARRIER_ACCESS_SHADER_RESOURCE, D3D12_BARRIER_LAYOUT_SHADER_RESOURCE, D3D12_BARRIER_SYNC_COMPUTE_SHADING, D3D12_BARRIER_ACCESS_UNORDERED_ACCESS, D3D12_BARRIER_LAYOUT_UNORDERED_ACCESS);
        D3D12_SHADER_RESOURCE_VIEW_DESC srv {};
        srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srv.Texture2D.MipLevels = 1;
        srv.Format = DXGI_FORMAT_R32G32_FLOAT;
        g_velDevice->CreateShaderResourceView(g_velStore, &srv, cpu);
        cpu.ptr += step;
        const DXGI_FORMAT depthFormat = depth->GetDesc().Format;
        srv.Format = depthFormat == DXGI_FORMAT_R32G8X24_TYPELESS || depthFormat == DXGI_FORMAT_D32_FLOAT_S8X24_UINT ? DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS : DXGI_FORMAT_R32_FLOAT;
        g_velDevice->CreateShaderResourceView(depth, &srv, cpu);
        cpu.ptr += step;
        srv.Format = DXGI_FORMAT_R32G32_FLOAT;
        g_velDevice->CreateShaderResourceView(ownRead, &srv, cpu);
        cpu.ptr += step;
        D3D12_UNORDERED_ACCESS_VIEW_DESC uav {};
        uav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
        uav.Format = DXGI_FORMAT_R32G32_FLOAT;
        g_velDevice->CreateUnorderedAccessView(target, nullptr, &uav, cpu);
        cpu.ptr += step;
        g_velDevice->CreateUnorderedAccessView(ownWrite, nullptr, &uav, cpu);
        ID3D12DescriptorHeap* heaps[] = { g_velHeap };
        list->SetDescriptorHeaps(1, heaps);
        list->SetComputeRootSignature(g_velRoot);
        const float invSize[2] = { 1.0f / (float) g_velStoreDesc.Width, 1.0f / (float) g_velStoreDesc.Height };
        list->SetComputeRoot32BitConstants(0, 16, game, 0);
        list->SetComputeRoot32BitConstants(0, 16, same, 16);
        list->SetComputeRoot32BitConstants(0, 2, invSize, 32);
        // Step 15: the history is two Presents old and the object's share of the vector one: a straight line over both. The position two Presents back is not kept.
        const bool ownDoubled = g_histMode.load(std::memory_order_relaxed) >= 15;
        // Step 17: a slow sway too (.03 px a step, over a long history): far under step 15's start, so step 15 leaves it. The price, if any: noise on still pixels doubled.
        const bool ownAveraged = g_histMode.load(std::memory_order_relaxed) == 18 && g_velOwnPresent != UINT64_MAX && g_velOwnPresent + 1 == present;
        const bool ownSlow = g_histMode.load(std::memory_order_relaxed) >= 17;
        const float own[4] = { ownDoubled ? 1.0f : 0.0f, ownSlow ? 0.02f : 0.5f, ownSlow ? 0.08f : 1.0f, ownAveraged ? 1.0f : 0.0f };
        list->SetComputeRoot32BitConstants(0, 4, own, 34);
        if (ownAveraged)
            ++g_velOwnAveraged;
        if (ownDoubled)
            ++g_velOwnDoubled;
        list->SetComputeRootDescriptorTable(1, gpu);
        list->Dispatch((UINT) ((g_velStoreDesc.Width + 7) / 8), (g_velStoreDesc.Height + 7) / 8, 1);
        HistBarrier(list, target, D3D12_BARRIER_SYNC_COMPUTE_SHADING, D3D12_BARRIER_ACCESS_UNORDERED_ACCESS, D3D12_BARRIER_LAYOUT_UNORDERED_ACCESS, D3D12_BARRIER_SYNC_ALL, D3D12_BARRIER_ACCESS_SHADER_RESOURCE,
                    D3D12_BARRIER_LAYOUT_SHADER_RESOURCE);
        HistBarrier(list, g_velStore, D3D12_BARRIER_SYNC_COMPUTE_SHADING, D3D12_BARRIER_ACCESS_SHADER_RESOURCE, D3D12_BARRIER_LAYOUT_SHADER_RESOURCE, D3D12_BARRIER_SYNC_COPY, D3D12_BARRIER_ACCESS_COPY_SOURCE,
                    D3D12_BARRIER_LAYOUT_COPY_SOURCE);
        HistBarrier(list, ownWrite, D3D12_BARRIER_SYNC_COMPUTE_SHADING, D3D12_BARRIER_ACCESS_UNORDERED_ACCESS, D3D12_BARRIER_LAYOUT_UNORDERED_ACCESS, D3D12_BARRIER_SYNC_COMPUTE_SHADING, D3D12_BARRIER_ACCESS_SHADER_RESOURCE, D3D12_BARRIER_LAYOUT_SHADER_RESOURCE);
        g_velOwnPresent = present;
        ++g_velCorrected;
    }
    else
    {
        HistBarrier(list, target, D3D12_BARRIER_SYNC_COPY, D3D12_BARRIER_ACCESS_COPY_SOURCE, D3D12_BARRIER_LAYOUT_COPY_SOURCE, D3D12_BARRIER_SYNC_ALL, D3D12_BARRIER_ACCESS_SHADER_RESOURCE,
                    D3D12_BARRIER_LAYOUT_SHADER_RESOURCE);
        HistBarrier(list, g_velStore, D3D12_BARRIER_SYNC_COPY, D3D12_BARRIER_ACCESS_COPY_DEST, D3D12_BARRIER_LAYOUT_COPY_DEST, D3D12_BARRIER_SYNC_COPY, D3D12_BARRIER_ACCESS_COPY_SOURCE,
                    D3D12_BARRIER_LAYOUT_COPY_SOURCE);
        ++g_velSaveOnly;
    }
    list->Close();
}

// A barrier that starts this texture's write: its partner, in SRV and what this frame reads as previous, to our copy, then this eye's last into it. As the snapshots. Holds g_histMutex.
// This Present: ours whole, or the game's whole. Same-eye history on the other eye's matrix shows as a layer off the ground, and the next trigger will not be one anyone thought of. A Present left to the
// game still saves, so the next one puts this eye's last and not one four Presents old. Holds g_histMutex.
uint64_t g_histDecidedPresent = UINT64_MAX, g_histPlainFrames = 0;
bool g_histDecidedOurs = true;
bool HistFrameOurs(uint64_t present)
{
    if (g_histDecidedPresent != present)
    {
        g_histDecidedPresent = present;
        bool ours = true;
        if (g_camSplit.load(std::memory_order_relaxed))
        {
            std::lock_guard lock(g_camMutex);
            ours = g_camSplitPresent == present;
        }
        g_histDecidedOurs = ours;
        if (!ours)
        {
            ++g_histPlainFrames;
            g_histPlainPresent.store(present, std::memory_order_relaxed);
        }
    }
    return g_histDecidedOurs;
}

void HistSwapExtra(ID3D12GraphicsCommandList7* cmdList, HistPair& pair, const D3D12_TEXTURE_BARRIER& b, uint64_t present)
{
    const bool writesA = b.pResource == pair.a;
    if ((writesA ? pair.swappedA : pair.swappedB) == present)
        return; // the stencil starts its write several times a frame: the first is the turn
    (writesA ? pair.startedA : pair.startedB) = present;
    // The partner is only a pointer: the game may have freed it (a menu, a settings change, the way out: a crash in the driver from here). It started its own write the Present before, or it is not touched.
    const uint64_t partnerStarted = writesA ? pair.startedB : pair.startedA;
    if (partnerStarted == UINT64_MAX || partnerStarted + 1 != present)
    {
        ++g_histExtraPartnerGone;
        pair.validA = pair.validB = false;
        return;
    }
    const D3D12_RESOURCE_DESC now = b.pResource->GetDesc();
    if (now.Width != pair.desc.Width || now.Height != pair.desc.Height || now.Format != pair.desc.Format)
    {
        ++g_histSizeChanged;
        return;
    }
    if (!g_histStoresReady || pair.storeA == nullptr || pair.storeB == nullptr)
    {
        ++g_histNoStore;
        return;
    }
    ID3D12Resource* partner = writesA ? pair.b : pair.a;
    ID3D12Resource* partnersStore = writesA ? pair.storeB : pair.storeA;
    ID3D12Resource* thisEyes = writesA ? pair.storeA : pair.storeB;
    const bool restore = (writesA ? pair.validA : pair.validB) && HistFrameOurs(present);
    HistBarrier(cmdList, partner, D3D12_BARRIER_SYNC_ALL, D3D12_BARRIER_ACCESS_SHADER_RESOURCE, D3D12_BARRIER_LAYOUT_SHADER_RESOURCE, D3D12_BARRIER_SYNC_COPY, D3D12_BARRIER_ACCESS_COPY_SOURCE,
                D3D12_BARRIER_LAYOUT_COPY_SOURCE);
    HistStoreTo(cmdList, partnersStore, D3D12_BARRIER_LAYOUT_COPY_DEST);
    HistCopy(cmdList, partner, partnersStore);
    ++g_histExtraSaves;
    if (restore)
    {
        HistBarrier(cmdList, partner, D3D12_BARRIER_SYNC_COPY, D3D12_BARRIER_ACCESS_COPY_SOURCE, D3D12_BARRIER_LAYOUT_COPY_SOURCE, D3D12_BARRIER_SYNC_COPY, D3D12_BARRIER_ACCESS_COPY_DEST,
                    D3D12_BARRIER_LAYOUT_COPY_DEST);
        HistStoreTo(cmdList, thisEyes, D3D12_BARRIER_LAYOUT_COPY_SOURCE);
        HistCopy(cmdList, thisEyes, partner);
        HistBarrier(cmdList, partner, D3D12_BARRIER_SYNC_COPY, D3D12_BARRIER_ACCESS_COPY_DEST, D3D12_BARRIER_LAYOUT_COPY_DEST, D3D12_BARRIER_SYNC_ALL, D3D12_BARRIER_ACCESS_SHADER_RESOURCE,
                    D3D12_BARRIER_LAYOUT_SHADER_RESOURCE);
        ++g_histExtraPuts;
    }
    else
        HistBarrier(cmdList, partner, D3D12_BARRIER_SYNC_COPY, D3D12_BARRIER_ACCESS_COPY_SOURCE, D3D12_BARRIER_LAYOUT_COPY_SOURCE, D3D12_BARRIER_SYNC_ALL, D3D12_BARRIER_ACCESS_SHADER_RESOURCE,
                    D3D12_BARRIER_LAYOUT_SHADER_RESOURCE);
    (writesA ? pair.validB : pair.validA) = true;
    (writesA ? pair.swappedA : pair.swappedB) = present;
}

// Over the survey. The moments: half size, RGBA16F, going to SRV in the very barrier call that starts the accumulation's targets (the map's, or a render-size RGBA16F): the call in front of the accumulation (both frames of a frame capture, one texture each by either test). The size and format alone name a dozen.
void HistExtraSurvey(ID3D12GraphicsCommandList7* cmdList, const D3D12_BARRIER_GROUP& group, uint64_t present)
{
    const UINT renderW = g_histRenderW.load(), renderH = g_histRenderH.load();
    bool mapWrite = false;
    for (UINT32 i = 0; i < group.NumBarriers && !mapWrite; ++i)
    {
        const D3D12_TEXTURE_BARRIER& b = group.pTextureBarriers[i];
        if (b.pResource != nullptr && b.LayoutBefore == D3D12_BARRIER_LAYOUT_RENDER_TARGET && b.LayoutAfter == D3D12_BARRIER_LAYOUT_UNORDERED_ACCESS)
        {
            const D3D12_RESOURCE_DESC desc = b.pResource->GetDesc();
            mapWrite = (desc.Format == DXGI_FORMAT_R8_UNORM || desc.Format == DXGI_FORMAT_R16G16B16A16_FLOAT) && desc.Width == renderW && desc.Height == renderH;
        }
    }
    for (UINT32 i = 0; i < group.NumBarriers; ++i)
    {
        const D3D12_TEXTURE_BARRIER& b = group.pTextureBarriers[i];
        if (b.pResource == nullptr)
            continue;
        if (mapWrite && b.LayoutBefore == D3D12_BARRIER_LAYOUT_UNORDERED_ACCESS && b.LayoutAfter == D3D12_BARRIER_LAYOUT_SHADER_RESOURCE && g_histMomentsSeen.size() < 200)
        {
            const D3D12_RESOURCE_DESC desc = b.pResource->GetDesc();
            if (desc.Format == DXGI_FORMAT_R16G16B16A16_FLOAT && desc.Width == renderW / 2 && desc.Height == renderH / 2)
                g_histMomentsSeen.push_back(HistSeen { present, (uint64_t) cmdList, b.pResource, desc, (int) b.LayoutBefore });
        }
        if (b.LayoutBefore == D3D12_BARRIER_LAYOUT_SHADER_RESOURCE && b.LayoutAfter == D3D12_BARRIER_LAYOUT_RENDER_TARGET && g_histStencilSeen.size() < 400)
        {
            const D3D12_RESOURCE_DESC desc = b.pResource->GetDesc();
            if (desc.Format == DXGI_FORMAT_R8_UINT && desc.Width == renderW && desc.Height == renderH)
                g_histStencilSeen.push_back(HistSeen { present, (uint64_t) cmdList, b.pResource, desc, (int) b.LayoutBefore });
        }
    }
}

void HistOnBarrier(ID3D12GraphicsCommandList7* cmdList, UINT32 count, const D3D12_BARRIER_GROUP* groups)
{
    if (cmdList->GetType() != D3D12_COMMAND_LIST_TYPE_DIRECT)
        return;
    const int mode = g_histMode.load(std::memory_order_relaxed);
    std::lock_guard lock(g_histMutex);
    const uint64_t present = g_present.load();
    for (UINT32 g = 0; g < count; ++g)
    {
        if (groups[g].Type != D3D12_BARRIER_TYPE_TEXTURE)
            continue;
        const bool momentsMissing = mode >= 9 && !g_histSurveying && !g_histPairs.empty() && std::find_if(g_histPairs.begin(), g_histPairs.end(), [](const HistPair& p) { return p.extra == 1; }) == g_histPairs.end();
        if ((g_histSurveying && present >= g_histSurveyStart) || momentsMissing)
            HistExtraSurvey(cmdList, groups[g], present);
        if (momentsMissing && g_histDevice == nullptr)
            cmdList->GetDevice(IID_PPV_ARGS(&g_histDevice));
        for (UINT32 i = 0; i < groups[g].NumBarriers; ++i)
        {
            const D3D12_TEXTURE_BARRIER& b = groups[g].pTextureBarriers[i];
            if (b.pResource == nullptr)
                continue;
            MapTraceBarrier(cmdList, b, present);
            if (g_histSurveying && present >= g_histSurveyStart && present - g_histSurveyStart < 32 &&
                (b.LayoutAfter == D3D12_BARRIER_LAYOUT_UNORDERED_ACCESS || b.LayoutAfter == D3D12_BARRIER_LAYOUT_RENDER_TARGET))
                g_histWritten[b.pResource] |= 1u << (present - g_histSurveyStart);
            if (mode >= 8 && !g_histSurveying && g_histStoresReady)
                VelOnBarrier(cmdList, b, present);
            if (mode >= 9 && !g_histSurveying)
                for (HistPair& pair : g_histPairs)
                    if (pair.extra != 0 && (pair.extra != 3 || mode == 11) && b.LayoutBefore == pair.triggerBefore && b.LayoutAfter == pair.triggerAfter && (b.pResource == pair.a || b.pResource == pair.b))
                    {
                        HistSwapExtra(cmdList, pair, b, present);
                        break;
                    }
            if (mode >= 4)
            {
                // The lighting's batch: from its first lighting texture, on that list, once a Present. A texture of the batch going to SRV ends the lighting on that list.
                if (b.LayoutBefore == D3D12_BARRIER_LAYOUT_UNORDERED_ACCESS && b.LayoutAfter == D3D12_BARRIER_LAYOUT_UNORDERED_ACCESS)
                {
                    if (g_histBatchPresent != present &&
                        std::find_if(g_histPairs.begin(), g_histPairs.end(), [&](const HistPair& p) { return p.lighting && (p.a == b.pResource || p.b == b.pResource || p.c == b.pResource); }) != g_histPairs.end())
                    {
                        g_histBatchPresent = present;
                        g_histBatchList = cmdList;
                        g_histBatch.clear();
                    }
                    if (g_histBatchPresent == present && g_histBatchList == cmdList && g_histBatch.size() < 64)
                        g_histBatch.push_back(b.pResource);
                }
                else if (b.LayoutBefore == D3D12_BARRIER_LAYOUT_UNORDERED_ACCESS && b.LayoutAfter == D3D12_BARRIER_LAYOUT_SHADER_RESOURCE && g_histBatchPresent == present && g_histBatchList == cmdList &&
                         g_histClosePresent != present && std::find(g_histBatch.begin(), g_histBatch.end(), b.pResource) != g_histBatch.end())
                {
                    g_histClosePresent = present;
                    if (CamSplitCopy(cmdList, false))
                    {
                        g_histClosed = present;
                        ++g_histCloses;
                    }
                    if (mode >= 8)
                        VelCopyBack(cmdList, present);
                }
            }
            // About to be written, from a read: a snapshot's turn. Its partner holds the other eye's last and is what this frame reads as previous.
            if (b.LayoutBefore == D3D12_BARRIER_LAYOUT_SHADER_RESOURCE && b.LayoutAfter == D3D12_BARRIER_LAYOUT_UNORDERED_ACCESS)
            {
                if (g_histSurveying)
                {
                    if (present >= g_histSurveyStart && g_histSnapSeen.size() < 1000)
                    {
                        const D3D12_RESOURCE_DESC desc = b.pResource->GetDesc();
                        if (desc.Format == DXGI_FORMAT_R32_UINT)
                            g_histSnapSeen.push_back(HistSeen { present, (uint64_t) cmdList, b.pResource, desc, (int) b.LayoutBefore });
                    }
                    continue;
                }
                if (mode < 2 || mode == 5)
                    continue;
                for (HistPair& pair : g_histPairs)
                {
                    if (!pair.snapshot || (b.pResource != pair.a && b.pResource != pair.b))
                        continue;
                    const bool writesA = b.pResource == pair.a;
                    ID3D12Resource* partner = writesA ? pair.b : pair.a;
                    if ((writesA ? pair.swappedA : pair.swappedB) == present)
                    {
                        ++g_histTwice;
                        break;
                    }
                    const D3D12_RESOURCE_DESC now = b.pResource->GetDesc();
                    if (now.Width != pair.desc.Width || now.Height != pair.desc.Height || now.Format != pair.desc.Format)
                    {
                        ++g_histSizeChanged;
                        break;
                    }
                    if (!g_histStoresReady || pair.storeA == nullptr || pair.storeB == nullptr)
                    {
                        ++g_histNoStore;
                        break;
                    }
                    ID3D12Resource* partnersStore = writesA ? pair.storeB : pair.storeA;
                    ID3D12Resource* thisEyes = writesA ? pair.storeA : pair.storeB;
                    const bool restore = (writesA ? pair.validA : pair.validB) && HistFrameOurs(present);
                    // The partner, in SRV since its own write a Present ago: to our copy, then this eye's last into it. Back to exactly SRV.
                    HistBarrier(cmdList, partner, D3D12_BARRIER_SYNC_ALL, D3D12_BARRIER_ACCESS_SHADER_RESOURCE, D3D12_BARRIER_LAYOUT_SHADER_RESOURCE, D3D12_BARRIER_SYNC_COPY,
                                D3D12_BARRIER_ACCESS_COPY_SOURCE, D3D12_BARRIER_LAYOUT_COPY_SOURCE);
                    HistStoreTo(cmdList, partnersStore, D3D12_BARRIER_LAYOUT_COPY_DEST);
                    HistCopy(cmdList, partner, partnersStore);
                    ++g_histSnapSaves;
                    if (restore)
                    {
                        HistBarrier(cmdList, partner, D3D12_BARRIER_SYNC_COPY, D3D12_BARRIER_ACCESS_COPY_SOURCE, D3D12_BARRIER_LAYOUT_COPY_SOURCE, D3D12_BARRIER_SYNC_COPY, D3D12_BARRIER_ACCESS_COPY_DEST,
                                    D3D12_BARRIER_LAYOUT_COPY_DEST);
                        HistStoreTo(cmdList, thisEyes, D3D12_BARRIER_LAYOUT_COPY_SOURCE);
                        HistCopy(cmdList, thisEyes, partner);
                        HistBarrier(cmdList, partner, D3D12_BARRIER_SYNC_COPY, D3D12_BARRIER_ACCESS_COPY_DEST, D3D12_BARRIER_LAYOUT_COPY_DEST, D3D12_BARRIER_SYNC_ALL, D3D12_BARRIER_ACCESS_SHADER_RESOURCE,
                                    D3D12_BARRIER_LAYOUT_SHADER_RESOURCE);
                        ++g_histSnapPuts;
                    }
                    else
                    {
                        HistBarrier(cmdList, partner, D3D12_BARRIER_SYNC_COPY, D3D12_BARRIER_ACCESS_COPY_SOURCE, D3D12_BARRIER_LAYOUT_COPY_SOURCE, D3D12_BARRIER_SYNC_ALL, D3D12_BARRIER_ACCESS_SHADER_RESOURCE,
                                    D3D12_BARRIER_LAYOUT_SHADER_RESOURCE);
                    }
                    (writesA ? pair.validB : pair.validA) = true;
                    (writesA ? pair.swappedA : pair.swappedB) = present;
                    break;
                }
                continue;
            }
            // Written, now to be read. A texture coming to SRV from anything else (a copy) has not just been written: a second swap would overwrite our copy with what we put in.
            if (b.LayoutAfter != D3D12_BARRIER_LAYOUT_SHADER_RESOURCE ||
                (b.LayoutBefore != D3D12_BARRIER_LAYOUT_UNORDERED_ACCESS && b.LayoutBefore != D3D12_BARRIER_LAYOUT_RENDER_TARGET))
                continue;
            if (g_histSurveying)
            {
                if (present >= g_histSurveyStart && g_histSeen.size() < 4000)
                    g_histSeen.push_back(HistSeen { present, (uint64_t) cmdList, b.pResource, b.pResource->GetDesc(), (int) b.LayoutBefore });
                if (g_histDevice == nullptr)
                    cmdList->GetDevice(IID_PPV_ARGS(&g_histDevice));
                continue;
            }
            for (HistPair& pair : g_histPairs)
            {
                if (pair.snapshot || pair.extra != 0 || (b.pResource != pair.a && b.pResource != pair.b && b.pResource != pair.c))
                    continue;
                // The map's two copies are by eye, the Present's parity, not by texture: what it reads was written the Present before whichever texture holds it, on two or on three. The lighting's stay by texture.
                const bool isMapPair = pair.lighting && pair.lightingIndex < 0;
                const bool isA = isMapPair ? (present & 1) == 0 : b.pResource == pair.a;
                if (isMapPair)
                    g_histMapPresent = present;
                g_histLastFlip = present;
                // The head batch, in front of the game's barrier: the motion texture's last write is behind it, the lighting ahead (both captures, by the barriers' chain).
                if (mode >= 3 && pair.lighting)
                {
                    if (g_histOpenPresent != present)
                    {
                        g_histOpenPresent = present;
                        g_histOpenList = cmdList;
                        if (CamSplitCopy(cmdList, true))
                        {
                            g_histOpened = present;
                            ++g_histOpens;
                        }
                    }
                    else if (g_histOpenList != cmdList)
                        ++g_histSecondOpens;
                }
                if (mode == 0 || mode == 5 || (mode == 1 && !pair.lighting) || (mode == 6 && pair.lightingIndex == 1) || (mode == 7 && pair.lightingIndex == 0))
                {
                    ClipOnFlip(cmdList, b, pair);
                    break;
                }
                if ((isA ? pair.swappedA : pair.swappedB) == present)
                {
                    ++g_histTwice;
                    break;
                }
                const D3D12_RESOURCE_DESC now = b.pResource->GetDesc();
                if (now.Width != pair.desc.Width || now.Height != pair.desc.Height || now.Format != pair.desc.Format)
                {
                    ++g_histSizeChanged;
                    break;
                }
                if (!g_histStoresReady || pair.storeA == nullptr || pair.storeB == nullptr)
                {
                    ++g_histNoStore;
                    break;
                }
                ID3D12Resource* mine = isA ? pair.storeA : pair.storeB;
                ID3D12Resource* partners = isA ? pair.storeB : pair.storeA;
                const bool restore = (isA ? pair.validB : pair.validA) && HistFrameOurs(present);
                // The game's barrier says what this texture is now. Back to exactly that, so the barrier finds what it declared.
                HistBarrier(cmdList, b.pResource, b.SyncBefore, b.AccessBefore, b.LayoutBefore, D3D12_BARRIER_SYNC_COPY, D3D12_BARRIER_ACCESS_COPY_SOURCE, D3D12_BARRIER_LAYOUT_COPY_SOURCE);
                HistStoreTo(cmdList, mine, D3D12_BARRIER_LAYOUT_COPY_DEST);
                HistCopy(cmdList, b.pResource, mine);
                ++g_histSaves;
                if (restore)
                {
                    HistBarrier(cmdList, b.pResource, D3D12_BARRIER_SYNC_COPY, D3D12_BARRIER_ACCESS_COPY_SOURCE, D3D12_BARRIER_LAYOUT_COPY_SOURCE, D3D12_BARRIER_SYNC_COPY, D3D12_BARRIER_ACCESS_COPY_DEST,
                               D3D12_BARRIER_LAYOUT_COPY_DEST);
                    HistStoreTo(cmdList, partners, D3D12_BARRIER_LAYOUT_COPY_SOURCE);
                    HistCopy(cmdList, partners, b.pResource);
                    HistBarrier(cmdList, b.pResource, D3D12_BARRIER_SYNC_COPY, D3D12_BARRIER_ACCESS_COPY_DEST, D3D12_BARRIER_LAYOUT_COPY_DEST, b.SyncBefore, b.AccessBefore, b.LayoutBefore);
                    ++g_histPuts;
                }
                else
                {
                    HistBarrier(cmdList, b.pResource, D3D12_BARRIER_SYNC_COPY, D3D12_BARRIER_ACCESS_COPY_SOURCE, D3D12_BARRIER_LAYOUT_COPY_SOURCE, b.SyncBefore, b.AccessBefore, b.LayoutBefore);
                    ++g_histOnlySaved;
                }
                (isA ? pair.validA : pair.validB) = true;
                (isA ? pair.swappedA : pair.swappedB) = present;
                ClipOnFlip(cmdList, b, pair);
                break;
            }
        }
    }
}

// The survey's four Presents have missed one side of the moments: go on looking while a step wants them. Two textures, each only on its own parity. Holds g_histMutex.
void HistLookForMoments(uint64_t present)
{
    if (std::find_if(g_histPairs.begin(), g_histPairs.end(), [](const HistPair& p) { return p.extra == 1; }) != g_histPairs.end() || g_histPairs.empty() || g_histStoresMaking)
        return;
    if (++g_histMomentsLooked % 8 != 0)
        return;
    const HistSeen* sides[2] = {};
    bool mixed = false;
    for (const HistSeen& seen : g_histMomentsSeen)
    {
        const HistSeen*& side = sides[seen.present & 1];
        if (side != nullptr && side->resource != seen.resource)
            mixed = true;
        side = &seen;
    }
    if (sides[0] != nullptr && sides[1] != nullptr && sides[0]->resource == sides[1]->resource)
        mixed = true;
    if (mixed)
    {
        if (g_histMomentsLines++ < 4)
            LOG_INFO("GI capture: history test: looking for the moments: more than one texture a parity: looked again");
        g_histMomentsSeen.clear();
        return;
    }
    if (sides[0] == nullptr || sides[1] == nullptr || g_histDevice == nullptr)
    {
        if (g_histMomentsLooked % 512 == 0 && g_histMomentsLines++ < 8)
            LOG_INFO("GI capture: history test: looking for the moments at Present {}: parity 0 {}, parity 1 {}", present, sides[0] != nullptr ? "seen" : "none", sides[1] != nullptr ? "seen" : "none");
        return;
    }
    HistPair pair;
    pair.a = sides[0]->resource;
    pair.b = sides[1]->resource;
    pair.desc = sides[0]->desc;
    pair.extra = 1;
    pair.triggerBefore = D3D12_BARRIER_LAYOUT_RENDER_TARGET;
    pair.triggerAfter = D3D12_BARRIER_LAYOUT_UNORDERED_ACCESS;
    g_histPairs.push_back(pair);
    ++g_histMomentsLate;
    g_histStoresReady = false;
    g_histStoresMaking = true;
    ID3D12Device* device = g_histDevice;
    g_histDevice = nullptr;
    std::thread(HistStoresOnWorker, device, g_histPairs).detach();
    LOG_INFO("GI capture: history test: the moments found late, Present {}", present);
}

void HistOnPresent(uint64_t present)
{
    // The frame just recorded: what the lists carried, whole or not. A bracket that misses frames would flicker between states and read as something stranger.
    {
        const int mode = g_histMode.load(std::memory_order_relaxed);
        std::lock_guard lock(g_histMutex);
        const uint64_t frame = present - 1;
        if (mode >= 3 && !g_histSurveying && !g_histPairs.empty())
        {
            const bool opened = g_histOpened == frame, closed = g_histClosed == frame;
            if (mode == 3)
                ++(opened ? g_histWhole : g_histNeither);
            else
                ++(opened && closed ? g_histWhole : opened ? g_histOpenOnly : closed ? g_histCloseOnly : g_histNeither);
            if (mode != 3 && g_histBatchPresent == frame && g_histClosePresent != frame)
            {
                ++g_histBatchNoClose;
                if (g_histNoCloseLines++ < 4)
                    LOG_INFO("GI capture: history test: present {}: the lighting's batch of {} textures on list {:X} and none of them went to SRV on that list", frame, g_histBatch.size(), (uint64_t) g_histBatchList);
            }
        }
    }
    float row = 0;
    {
        std::lock_guard lock(g_mutex);
        row = g_lastCamera.clipToPrevClip[8];
    }
    bool report = false;
    {
        std::lock_guard lock(g_histMutex);
        if (present >= g_histUiNext)
        {
            const uint64_t swaps = g_pipeSwaps.load(std::memory_order_relaxed);
            g_histUiCorrecting = g_velCorrected != g_histUiCorrected;
            g_histUiLimit = swaps != g_histUiSwaps;
            g_histUiCorrected = g_velCorrected;
            g_histUiSwaps = swaps;
            g_histUiNext = present + kHistUiPresents;
        }
        if (g_mapTraceStart != UINT64_MAX && present >= g_mapTraceStart + kMapTracePresents)
            MapTraceFlush();
        if (g_histSurveying)
        {
            if (present >= g_histSurveyEnd)
                HistFinish();
        }
        else if (g_histPairs.empty())
        {
            if (g_histNextAttempt != 0 && present >= g_histNextAttempt)
            {
                HistStartSurvey(present);
            }
        }
        else if (g_histMode.load(std::memory_order_relaxed) >= 9)
            HistLookForMoments(present);
        if (!g_histSurveying && !g_histPairs.empty() && g_histOpenPresent == present - 1 && g_histMapPresent != present - 1)
            ++g_histMapMissed;
        // A settings change can remake textures with the frames still coming, so no gap: the swaps then go on with old pointers (an extra picture on top). What shows it: sizes that no longer match, partners that stopped starting their writes.
        if (present >= g_histWatchNext)
        {
            const bool stale = !g_histSurveying && !g_histPairs.empty() && (g_histSizeChanged > g_histWatchSize || g_histExtraPartnerGone > g_histWatchGone + 8 || g_histMapMissed > g_histWatchMapMissed + 2);
            g_histWatchSize = g_histSizeChanged;
            g_histWatchGone = g_histExtraPartnerGone;
            g_histWatchMapMissed = g_histMapMissed;
            g_histWatchNext = present + 30;
            if (stale)
            {
                LOG_INFO("GI capture: history test: the textures changed under it (size changed {}, partner not seen {}, map missed {}): looking again", g_histSizeChanged, g_histExtraPartnerGone, g_histMapMissed);
                ++g_histResurveys;
                g_histAttempts = 0;
                HistStartSurvey(present);
            }
        }
        if (!g_histSurveying && !g_histPairs.empty() && present > g_histLastFlip + kHistNoFlipPresents)
        {
            // A resolution change or a loading screen: the textures are others now.
            ++g_histResurveys;
            g_histAttempts = 0;
            HistStartSurvey(present);
        }
        // The flip and the eyes keep one phase: the sign of the eye step alternates with it. Counted, not acted on: a late camera would read as a slip.
        const int sign = row > 0 ? 1 : -1;
        const int expected = g_histLastSignA == 0 ? sign : -g_histLastSignA;
        if (sign == expected)
            ++g_histPhaseSame;
        else
            ++g_histPhaseSlip;
        g_histLastSignA = sign;
        report = present % 600 == 0;
    }
    if (report)
        LOG_INFO("GI capture: history test: {}", GiCapture_Dx12::HistoryTestStatus());
}
} // namespace

// F8's last steps: the game's limit on a moving pixel's history. The lighting pass multiplies the vector's length by 50 and takes that from the history length (15 x (1 - 50 |mv|)).
// Under alternate eyes the vector is twice mono's at the same head speed, so the limit comes at half the speed: the control (mono capped at 45) shows the same spots. Two more pipelines of each of its two shaders, made with
// that one multiply pointed at a constant the shader already holds, and the pipeline swapped when the game binds it, so the step can change in a running game. The exact double would be 25: not in the shader.
// Only the two shaders whose digest is in this table, and only if the bits read what the table says: a game patch just turns it off.
namespace
{
struct PipeOp
{
    uint32_t bit, oldOperand, operand31, operand16;
};
struct PipeShader
{
    const char* name;
    uint8_t digest[16];
    size_t size;
    size_t bitcode;
    PipeOp diffuse, specular;
    // How much new frame a reflection takes: camera only in the game: a helmet moving by itself keeps its old reflection. Level 4 points that multiply at |g_velocity|. Two chunks of VBR6.
    uint32_t reflectionBit, reflectionOld, reflectionNew; // 0: not found in this version
    bool maxLighting; // seen only at Max
};
constexpr PipeShader kPipeShaders[] = {
    { "5758dff2b0d1", { 0x63, 0x92, 0x1a, 0x9b, 0xdc, 0x0d, 0x2a, 0x67, 0x46, 0x0c, 0x7e, 0x27, 0x3c, 0x65, 0x05, 0x5d }, 67972, 17264, { 191560, 2603, 2544, 2588 }, { 230777, 3246, 0, 3231 }, 215594, 81, 386, false },
    { "635c2211bfa1", { 0xa6, 0x15, 0x5d, 0x6c, 0x48, 0xfb, 0x60, 0x77, 0xe8, 0x6e, 0xc4, 0x92, 0x52, 0xee, 0x02, 0x1c }, 49768, 16252, { 148937, 1924, 1887, 1907 }, { 184209, 2516, 0, 2499 }, 169321, 76, 341, false },
    { "45932 bytes", { 0x93, 0x3c, 0x29, 0x23, 0xdb, 0x83, 0xe1, 0x20, 0x17, 0xa5, 0x57, 0xac, 0x6b, 0x7d, 0x86, 0x5a }, 45932, 16280, { 117537, 1257, 1223, 1239 }, { 152495, 1847, 0, 1829 }, 0, 0, 0, true },
    { "63532 bytes", { 0xb7, 0xda, 0xc9, 0x4f, 0x76, 0xa9, 0x3b, 0xb1, 0xd0, 0x64, 0x33, 0xad, 0xcd, 0x87, 0x23, 0x9a }, 63532, 17212, { 155863, 1869, 1811, 1851 }, { 194861, 2512, 0, 2494 }, 0, 0, 0, true },
    { "67488 bytes", { 0xbf, 0xd0, 0x5b, 0x6b, 0x29, 0x8e, 0x69, 0x78, 0xc2, 0x18, 0xcb, 0x3c, 0x6d, 0x8a, 0xbc, 0xbf }, 67488, 17180, { 188078, 2546, 2486, 2530 }, { 227295, 3189, 0, 3173 }, 0, 0, 0, false },
};
constexpr int kPipeLevels = 4; // 1 the diffuse's limit at 31, 2 at 16, 3 at 16 with the specular's too, 4 at 16 with the reflection's new frame on the whole vector
struct PipeEntry
{
    ID3D12PipelineState* game = nullptr; // identity only
    ID3D12PipelineState* level[kPipeLevels] = {};
};
PipeEntry g_pipes[32];
std::atomic<int> g_pipeCount { 0 };
std::atomic<int> g_pipeLevel { 0 };
std::mutex g_pipeMutex;
uint64_t g_pipeMatched = 0, g_pipeOther = 0, g_pipeRefused = 0, g_pipeReused = 0;

uint32_t PipeRead(const uint8_t* bits, uint32_t bit, int chunks = 3)
{
    uint32_t value = 0;
    for (int chunk = 0; chunk < chunks; ++chunk)
        for (int i = 0; i < 5; ++i)
        {
            const uint32_t at = bit + chunk * 6 + i;
            value |= (uint32_t) ((bits[at >> 3] >> (at & 7)) & 1) << (chunk * 5 + i);
        }
    return value;
}

// In the same three chunks of a VBR6: the bits after it do not move.
void PipeWrite(uint8_t* bits, uint32_t bit, uint32_t value, int chunks = 3)
{
    for (int chunk = 0; chunk < chunks; ++chunk)
        for (int i = 0; i < 5; ++i)
        {
            const uint32_t at = bit + chunk * 6 + i;
            const uint8_t mask = (uint8_t) (1 << (at & 7));
            bits[at >> 3] = (uint8_t) ((bits[at >> 3] & ~mask) | (((value >> (chunk * 5 + i)) & 1) ? mask : 0));
        }
}

void PipeMd5Block(uint32_t state[4], const uint8_t* block)
{
    static const uint32_t k[64] = { 0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501, 0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be, 0x6b901122,
                                    0xfd987193, 0xa679438e, 0x49b40821, 0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa, 0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8, 0x21e1cde6, 0xc33707d6,
                                    0xf4d50d87, 0x455a14ed, 0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a, 0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c, 0xa4beea44, 0x4bdecfa9, 0xf6bb4b60,
                                    0xbebfbc70, 0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05, 0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665, 0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039,
                                    0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1, 0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1, 0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391 };
    static const uint8_t r[64] = { 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20,
                                   4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21 };
    uint32_t words[16];
    std::memcpy(words, block, 64);
    uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
    for (uint32_t i = 0; i < 64; ++i)
    {
        uint32_t f, g;
        if (i < 16)
        {
            f = (b & c) | (~b & d);
            g = i;
        }
        else if (i < 32)
        {
            f = (d & b) | (~d & c);
            g = (5 * i + 1) % 16;
        }
        else if (i < 48)
        {
            f = b ^ c ^ d;
            g = (3 * i + 5) % 16;
        }
        else
        {
            f = c ^ (b | ~d);
            g = (7 * i) % 16;
        }
        const uint32_t next = d;
        d = c;
        c = b;
        const uint32_t x = a + f + k[i] + words[g];
        b = b + ((x << r[i]) | (x >> (32 - r[i])));
        a = next;
    }
    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
}

// The container's digest over everything behind it: MD5's rounds with the lengths placed differently.
void PipeDigest(std::vector<uint8_t>& container)
{
    const uint8_t* data = container.data() + 20;
    const uint32_t size = (uint32_t) container.size() - 20;
    uint32_t state[4] = { 0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476 };
    uint32_t done = 0;
    for (; size - done >= 64; done += 64)
        PipeMd5Block(state, data + done);
    const uint32_t left = size - done;
    const uint32_t bitsLow = size * 8, bitsHigh = (bitsLow >> 2) | 1;
    uint8_t block[64] = {};
    if (left >= 56)
    {
        std::memcpy(block, data + done, left);
        block[left] = 0x80;
        PipeMd5Block(state, block);
        std::memset(block, 0, 64);
        std::memcpy(block, &bitsLow, 4);
        std::memcpy(block + 60, &bitsHigh, 4);
    }
    else
    {
        std::memcpy(block, &bitsLow, 4);
        std::memcpy(block + 4, data + done, left);
        block[4 + left] = 0x80;
        std::memcpy(block + 60, &bitsHigh, 4);
    }
    PipeMd5Block(state, block);
    std::memcpy(container.data() + 4, state, 16);
}

bool PipeMake(const PipeShader& shader, const void* bytes, int level, std::vector<uint8_t>& out)
{
    const uint8_t* original = static_cast<const uint8_t*>(bytes);
    out.assign(original, original + shader.size);
    uint8_t* bits = out.data() + shader.bitcode;
    if (PipeRead(bits, shader.diffuse.bit) != shader.diffuse.oldOperand || PipeRead(bits, shader.specular.bit) != shader.specular.oldOperand || (shader.reflectionBit != 0 && PipeRead(bits, shader.reflectionBit, 2) != shader.reflectionOld))
        return false;
    PipeWrite(bits, shader.diffuse.bit, level == 1 ? shader.diffuse.operand31 : shader.diffuse.operand16);
    if (level == 3)
        PipeWrite(bits, shader.specular.bit, shader.specular.operand16);
    if (level == 4 && shader.reflectionBit != 0)
        PipeWrite(bits, shader.reflectionBit, shader.reflectionNew, 2);
    PipeDigest(out);
    return true;
}
} // namespace

void GiCapture_Dx12::OnPipelineCreated(ID3D12PipelineState* pipeline, ID3D12RootSignature* root, const void* bytes, size_t length)
{
    if (!Enabled() || t_inside || pipeline == nullptr || bytes == nullptr || length < 20) // t_inside: our own creates below come back through the hook, with this lock held
        return;
    // The game freed one of its pipelines and this one has its address (a settings change): what we made for the old one must not follow.
    const int known = g_pipeCount.load(std::memory_order_acquire);
    for (int i = 0; i < known; ++i)
        if (g_pipes[i].game == pipeline)
        {
            std::lock_guard lock(g_pipeMutex);
            g_pipes[i].game = nullptr;
            ++g_pipeReused;
        }
    const PipeShader* shader = nullptr;
    for (const PipeShader& candidate : kPipeShaders)
        if (candidate.size == length && std::memcmp(static_cast<const uint8_t*>(bytes) + 4, candidate.digest, 16) == 0)
            shader = &candidate;
    static const char kEntry[] = "RenderDiffuseTiledCS";
    if (shader == nullptr)
    {
        // Another version of it: a game patch, or a setting this table has not seen. Named, once each.
        const char* start = static_cast<const char*>(bytes);
        if (std::search(start, start + length, kEntry, kEntry + sizeof(kEntry) - 1) != start + length)
        {
            std::lock_guard lock(g_pipeMutex);
            if (g_pipeOther++ < 8)
                LOG_INFO("GI capture: the history limit: a {} byte {} not in the table: left alone", length, kEntry);
            if (Developer())
            {
                // Each version once, to find the same edit in it offline. The digest names it.
                const std::string digest(static_cast<const char*>(bytes) + 4, 16);
                static std::set<std::string> saved;
                if (saved.insert(digest).second)
                {
                    std::string hex;
                    for (const char c : digest.substr(0, 6))
                        hex += std::format("{:02x}", (uint8_t) c);
                    const auto folder = Util::ExePath().parent_path() / "OptiCapture" / "shaders";
                    std::error_code ec;
                    std::filesystem::create_directories(folder, ec);
                    const auto path = folder / std::format("{}_{}.dxbc", length, hex);
                    std::ofstream out(path, std::ios::binary);
                    out.write(static_cast<const char*>(bytes), (std::streamsize) length);
                    LOG_INFO("GI capture: the history limit: saved it as {}", path.string());
                }
            }
        }
        return;
    }
    std::lock_guard lock(g_pipeMutex);
    const int at = g_pipeCount.load();
    if (root == nullptr || at >= (int) std::size(g_pipes))
    {
        ++g_pipeRefused;
        return;
    }
    ID3D12Device* device = nullptr;
    if (FAILED(pipeline->GetDevice(IID_PPV_ARGS(&device))) || device == nullptr)
    {
        ++g_pipeRefused;
        return;
    }
    PipeEntry entry;
    entry.game = pipeline;
    bool all = true;
    for (int level = 1; level <= kPipeLevels; ++level)
    {
        std::vector<uint8_t> edited;
        if (!PipeMake(*shader, bytes, level, edited))
        {
            all = false;
            LOG_WARN("GI capture: the history limit: {}: the bits are not what the table says", shader->name);
            break;
        }
        D3D12_COMPUTE_PIPELINE_STATE_DESC desc {};
        desc.pRootSignature = root;
        desc.CS = { edited.data(), edited.size() };
        t_inside = true;
        const HRESULT made = device->CreateComputePipelineState(&desc, IID_PPV_ARGS(&entry.level[level - 1]));
        t_inside = false;
        if (FAILED(made))
        {
            all = false;
            LOG_WARN("GI capture: the history limit: {} level {}: pipeline {:X}", shader->name, level, (uint32_t) made);
            break;
        }
    }
    device->Release();
    if (!all)
    {
        ++g_pipeRefused;
        return; // a pipeline already made stays: it is small, and nothing is released while a list may hold it
    }
    g_pipes[at] = entry;
    g_pipeCount.store(at + 1);
    ++g_pipeMatched;
    LOG_INFO("GI capture: the history limit: {}: {} edited pipelines ready", shader->name, kPipeLevels);
}

ID3D12PipelineState* GiCapture_Dx12::SwapPipeline(ID3D12PipelineState* pipeline)
{
    const int level = g_pipeLevel.load(std::memory_order_relaxed);
    if (level == 0 || pipeline == nullptr || !g_histStoresReady.load(std::memory_order_relaxed) || g_histPlainPresent.load(std::memory_order_relaxed) == g_present.load(std::memory_order_relaxed))
        return pipeline; // the edit on the other eye's history flickers (ground, roofs)
    const int count = g_pipeCount.load(std::memory_order_acquire);
    for (int i = 0; i < count; ++i)
        if (g_pipes[i].game == pipeline)
        {
            g_pipeSwaps.fetch_add(1, std::memory_order_relaxed);
            return g_pipes[i].level[level - 1];
        }
    return pipeline;
}

bool GiCapture_Dx12::PipelineForTest(int shader, int level, const void* bytes, std::vector<uint8_t>& out)
{
    return PipeMake(kPipeShaders[shader], bytes, level, out);
}

std::string GiCapture_Dx12::HistoryTestStatus()
{
    std::lock_guard lock(g_histMutex);
    return std::format("mode {}: pairs {}, attempts {}, refused {}, saves {}, puts {}, only saved {}, snapshot saves {}, snapshot puts {}, no store {}, size changed {}, a second flip in a Present {}, surveys again {}, the eye step's sign: alternating {}, not {}; bracket: opens {}, closes {}, lighting flips on another list {}, frames whole {}, open only {}, close only {}, neither {}, batch and no close {}; motion: state {}, written {}, copies back {} (by this Present {}, by the flip {}), none {}, corrected {}, only the copy {} (no block {}, no depth {}, not the flip's {}), its list not submitted {}, no submits seen {}, waits {}; extra saves {}, extra puts {}, moments found late {}, frame counts sent {}, partner not seen {}; limit: shaders edited {}, others seen {}, refused {}, swaps {}, own motion doubled {}, averaged {}, addresses reused {}, frames left to the game {}, map missed {}", g_histMode.load(),
                       g_histPairs.size(), g_histAttempts, g_histRefused, g_histSaves, g_histPuts, g_histOnlySaved, g_histSnapSaves, g_histSnapPuts, g_histNoStore, g_histSizeChanged, g_histTwice, g_histResurveys, g_histPhaseSame, g_histPhaseSlip,
                       g_histOpens, g_histCloses, g_histSecondOpens, g_histWhole, g_histOpenOnly, g_histCloseOnly, g_histNeither, g_histBatchNoClose, g_velState.load(), g_velStamps, g_velArmed, g_velByStamp, g_velByFlip, g_velNotArmed,
                       g_velCorrected, g_velSaveOnly, g_velNoMatrix, g_velNoDepth, g_velWrongTarget, g_velNotFound, g_velNoSubmits, g_velWaits, g_histExtraSaves, g_histExtraPuts, g_histMomentsLate, g_camFrameOwnSent, g_histExtraPartnerGone, g_pipeMatched, g_pipeOther, g_pipeRefused, g_pipeSwaps.load(), g_velOwnDoubled, g_velOwnAveraged, g_pipeReused, g_histPlainFrames, g_histMapMissed);
}

bool GiCapture_Dx12::SpliceExecute(ID3D12CommandQueue* queue, ID3D12CommandList* const* lists, UINT count, std::vector<ID3D12CommandList*>& out)
{
    g_velSubmitSeen.store(g_present.load(std::memory_order_relaxed), std::memory_order_relaxed);
    ID3D12GraphicsCommandList7* armed = g_velArmedList.load(std::memory_order_relaxed);
    if (armed == nullptr || queue == nullptr || lists == nullptr)
        return false;
    UINT at = count;
    for (UINT i = 0; i < count && at == count; ++i)
    {
        if ((uint64_t) lists[i] == (uint64_t) armed)
            at = i;
        else
        {
            // The hook's pointer came through QueryInterface: the same list may stand here under another.
            ID3D12GraphicsCommandList7* asked = nullptr;
            if (lists[i] != nullptr && SUCCEEDED(lists[i]->QueryInterface(IID_PPV_ARGS(&asked))) && asked != nullptr)
            {
                if (asked == armed)
                    at = i;
                asked->Release();
            }
        }
    }
    if (at == count)
        return false;
    std::lock_guard lock(g_histMutex);
    g_velArmedList.store(nullptr, std::memory_order_relaxed);
    t_inside = true; // our own barriers come back through the hook
    VelRecord(queue);
    t_inside = false;
    out.assign(lists, lists + count);
    out.insert(out.begin() + at, g_velList);
    return true;
}

// F8 in the game: off, the correction and the moments, then the limit at 31, at 16, at 16 with the specular's; beeps = the place + 1. The steps that found these are the offline test's.
static bool g_histAllSteps = false;
void GiCapture_Dx12::HistoryAllSteps(bool all)
{
    g_histAllSteps = all;
}

static void HistSetStep(int mode)
{
    static const char* const names[] = { "off",
                                         "the same eye's matrix and the lighting history",
                                         "also the depth and normal snapshots",
                                         "the swaps, the matrix in from the lighting on, not put back",
                                         "the swaps, the matrix bracketing the lighting",
                                         "the matrix bracketing the lighting, no swaps",
                                         "the bracket, the snapshots and map, only the first lighting pair swapped",
                                         "the bracket, the snapshots and map, only the second lighting pair swapped",
                                         "the swaps, the matrix bracketing the lighting, the motion same-eye for the accumulation",
                                         "as the step before, and the moments and their half-size depth swapped",
                                         "as the step before, and each eye its own frame count",
                                         "as the step before, and the stencil swapped",
                                         "the correction and the moments, the history limit at 31",
                                         "the correction and the moments, the history limit at 16",
                                         "the correction and the moments, the history limit at 16, the specular's too",
                                         "the correction and the moments, the history limit at 16, an object's own motion doubled",
                                         "as the step before, and the reflections' new frame on the whole vector",
                                         "the correction and the moments, the history limit at 16, an object's own motion doubled, slow motion too",
                                         "as the step before, and the eye-flipped part of the own motion cancelled" };
    LOG_INFO("GI capture: history test: so far: {}", GiCapture_Dx12::HistoryTestStatus());
    LOG_INFO("GI capture: history test: so far: {}", GiCapture_Dx12::CameraTestStatus());
    const int previous = g_histMode.load();
    {
        std::lock_guard lock(g_histMutex);
        for (HistPair& pair : g_histPairs)
            pair.validA = pair.validB = false;
        g_velStopped = false;
        g_velSubmitSeen.store(UINT64_MAX, std::memory_order_relaxed);
        if (previous == 0)
        {
            g_histAttempts = 0;
            g_histWarnPending = mode != 0;
            HistStartSurvey(g_present.load());
        }
    }
    g_camSplit.store(mode >= 3);
    g_camFrameOwn.store(mode == 10 || mode == 11);
    g_pipeLevel.store(mode == 16 ? 4 : mode == 15 || mode == 17 || mode == 18 ? 2 : mode >= 12 ? mode - 11 : 0);
    g_histMode.store(mode);
    // The matrix goes with the history: the swap alone would reproject its own eye's by an eye step.
    g_camQueues.store(0);
    if (mode == 0)
        g_camEffect.store(0);
    else if (g_camEffect.load() != 4)
    {
        {
            std::lock_guard lock(g_camMutex);
            g_camSeen.clear();
        }
        g_camEffect.store(4);
    }
    LOG_INFO("GI capture: history test: present {}: now {} ({})", g_present.load(), mode, names[mode]);
    CamBeeps(g_histAllSteps ? mode + 1 : (mode == 0 ? 1 : mode == 9 ? 2 : mode == 18 ? 7 : mode == 17 ? 6 : mode == 16 ? 8 : mode == 15 ? 5 : mode - 9));
}

void GiCapture_Dx12::CycleHistoryTest()
{
    if (!Enabled())
        return;
    const int previous = g_histMode.load();
    static constexpr int kSteps[] = { 0, 9, 12, 13, 15, 17, 18 };
    int place = 0;
    for (int i = 0; i < (int) std::size(kSteps); ++i)
        if (kSteps[i] == previous)
            place = (i + 1) % (int) std::size(kSteps);
    HistSetStep(g_histAllSteps ? (previous + 1) % 19 : kSteps[place]);
}

void GiCapture_Dx12::SetHistoryStep(int step)
{
    if (Enabled() && step != g_histMode.load())
        HistSetStep(step);
}

GiCapture_Dx12::HistoryUi GiCapture_Dx12::HistoryUiStatus()
{
    HistoryUi ui;
    ui.step = g_histMode.load(std::memory_order_relaxed);
    {
        std::lock_guard lock(g_histMutex);
        ui.textures = g_histSurveying ? 0 : (int) g_histPairs.size();
        ui.correcting = ui.step != 0 && g_histUiCorrecting;
        ui.limitRunning = ui.step != 0 && g_histUiLimit;
    }
    std::lock_guard lock(g_pipeMutex);
    ui.shadersEdited = g_pipeMatched;
    ui.otherVersions = g_pipeOther;
    return ui;
}

void GiCapture_Dx12::ToggleHistory()
{
    if (!Enabled())
        return;
    HistSetStep(g_histMode.load() == 0 ? kHistOnStep : 0);
}

void GiCapture_Dx12::ArmClip()
{
    if (!Developer())
        return;
    if (g_clipState.load() != 0 || g_phase.load() != Idle)
    {
        LOG_INFO("GI capture: clip: pressed and refused: busy (clip state {}, capture phase {})", g_clipState.load(), g_phase.load());
        CamBeeps(1);
        return;
    }
    const uint64_t present = g_present.load();
    g_clipArmed = present;
    {
        std::lock_guard lock(g_histMutex);
        if (g_histMode.load() == 0)
        {
            g_histAttempts = 0;
            HistStartSurvey(present);
        }
    }
    g_clipState.store(1);
    LOG_INFO("GI capture: clip: present {}: armed", present);
}

void GiCapture_Dx12::OnEnhancedBarrier(ID3D12GraphicsCommandList7* cmdList, UINT32 count, const D3D12_BARRIER_GROUP* groups)
{
    if ((g_histMode.load(std::memory_order_relaxed) != 0 || g_clipState.load(std::memory_order_relaxed) != 0) && !t_inside && cmdList != nullptr && groups != nullptr)
    {
        if (ID3D12Resource* watched = g_camWatchDest.load(std::memory_order_relaxed); watched != nullptr)
            for (UINT32 g = 0; g < count; ++g)
            {
                if (groups[g].Type == D3D12_BARRIER_TYPE_BUFFER)
                {
                    for (UINT32 i = 0; i < groups[g].NumBarriers; ++i)
                    {
                        const D3D12_BUFFER_BARRIER& b = groups[g].pBufferBarriers[i];
                        if (b.pResource != watched)
                            continue;
                        g_camWatchBuffer.fetch_add(1, std::memory_order_relaxed);
                        if (g_camWatchLines.fetch_add(1, std::memory_order_relaxed) < 24)
                            LOG_INFO("GI capture: the block's buffer: the game's barrier on list {:X}: sync {:X} > {:X} access {:X} > {:X}", (uint64_t) cmdList, (uint32_t) b.SyncBefore, (uint32_t) b.SyncAfter, (uint32_t) b.AccessBefore, (uint32_t) b.AccessAfter);
                    }
                }
                else if (groups[g].Type == D3D12_BARRIER_TYPE_GLOBAL && cmdList == g_camWatchList.load(std::memory_order_relaxed))
                {
                    for (UINT32 i = 0; i < groups[g].NumBarriers; ++i)
                    {
                        const D3D12_GLOBAL_BARRIER& b = groups[g].pGlobalBarriers[i];
                        g_camWatchGlobal.fetch_add(1, std::memory_order_relaxed);
                        if (g_camWatchLines.fetch_add(1, std::memory_order_relaxed) < 24)
                            LOG_INFO("GI capture: the block's buffer: a global barrier on the copy's list {:X}: sync {:X} > {:X} access {:X} > {:X}", (uint64_t) cmdList, (uint32_t) b.SyncBefore, (uint32_t) b.SyncAfter, (uint32_t) b.AccessBefore, (uint32_t) b.AccessAfter);
                    }
                }
            }
        t_inside = true; // our own barriers come back through this hook
        HistOnBarrier(cmdList, count, groups);
        ClipOnBarrier(cmdList, count, groups);
        t_inside = false;
    }
    const int phase = g_phase.load(std::memory_order_relaxed);
    if ((phase != Surveying && phase != Capturing) || t_inside || cmdList == nullptr || groups == nullptr)
        return;
    const D3D12_COMMAND_LIST_TYPE type = cmdList->GetType();
    if (type != D3D12_COMMAND_LIST_TYPE_DIRECT && type != D3D12_COMMAND_LIST_TYPE_COMPUTE)
        return;

    t_inside = true;
    std::lock_guard lock(g_mutex);
    if (phase == Surveying)
        g_enhancedBarriers.fetch_add(1, std::memory_order_relaxed);
    const uint64_t present = g_present.load();
    const int frame = phase == Capturing ? (int) (present - g_captureStart.load()) : 0;
    for (UINT32 g = 0; g < count; ++g)
    {
        const D3D12_BARRIER_GROUP& group = groups[g];
        if (group.Type == D3D12_BARRIER_TYPE_GLOBAL)
        {
            if (phase == Surveying)
                g_globalBarriers += group.NumBarriers;
        }
        else if (group.Type == D3D12_BARRIER_TYPE_BUFFER)
        {
            for (UINT32 i = 0; phase == Surveying && i < group.NumBarriers; ++i)
            {
                const D3D12_BUFFER_BARRIER& b = group.pBufferBarriers[i];
                if (b.pResource != nullptr && WriteAccess(b.AccessBefore, D3D12_BARRIER_LAYOUT_UNDEFINED) &&
                    !WriteAccess(b.AccessAfter, D3D12_BARRIER_LAYOUT_UNDEFINED))
                {
                    ++g_buffersSeen;
                    g_bufferBytes += b.pResource->GetDesc().Width;
                }
            }
        }
        else if (group.Type == D3D12_BARRIER_TYPE_TEXTURE)
        {
            for (UINT32 i = 0; i < group.NumBarriers; ++i)
            {
                const D3D12_TEXTURE_BARRIER& b = group.pTextureBarriers[i];
                if (b.pResource == nullptr)
                    continue;
                const D3D12_RESOURCE_DESC desc = b.pResource->GetDesc();
                const D3D12_BARRIER_SUBRESOURCE_RANGE& sub = b.Subresources;
                const bool hasZero = sub.IndexOrFirstMipLevel == 0xffffffff ||
                                     (sub.IndexOrFirstMipLevel == 0 &&
                                      (sub.NumMipLevels == 0 || (sub.FirstArraySlice == 0 && sub.FirstPlane == 0)));
                const bool split = ((b.SyncBefore | b.SyncAfter) & D3D12_BARRIER_SYNC_SPLIT) != 0;
                const bool ended = WriteAccess(b.AccessBefore, b.LayoutBefore) && !WriteAccess(b.AccessAfter, b.LayoutAfter);
                if (phase == Surveying)
                {
                    ++g_textureBarriers;
                    Seen& seen = NoteTexture(cmdList, b.pResource, desc);
                    if (seen.barrier.empty())
                        seen.barrier = std::format("{:X}->{:X}/{}->{}/sync{:X}->{:X}/sub{}", (unsigned) b.AccessBefore,
                                                   (unsigned) b.AccessAfter, (int) b.LayoutBefore, (int) b.LayoutAfter,
                                                   (unsigned) b.SyncBefore, (unsigned) b.SyncAfter, sub.IndexOrFirstMipLevel);
                    if (!ended || !hasZero || split)
                        ++seen.others;
                    if (Event* e = AddEvent(cmdList, 'E', seen, present); e != nullptr)
                    {
                        e->accessBefore = (uint32_t) b.AccessBefore;
                        e->accessAfter = (uint32_t) b.AccessAfter;
                        e->layoutBefore = (int) b.LayoutBefore;
                        e->layoutAfter = (int) b.LayoutAfter;
                        e->syncBefore = (uint32_t) b.SyncBefore;
                        e->syncAfter = (uint32_t) b.SyncAfter;
                        e->sub = sub.IndexOrFirstMipLevel;
                    }
                }
                if (!ended || !hasZero || split)
                    continue;
                const bool fromUav = (b.AccessBefore & D3D12_BARRIER_ACCESS_UNORDERED_ACCESS) != 0 ||
                                     (b.AccessBefore == D3D12_BARRIER_ACCESS_COMMON &&
                                      b.LayoutBefore != D3D12_BARRIER_LAYOUT_RENDER_TARGET);
                const int slot = OnWrite(cmdList, b.pResource, desc, phase, frame, fromUav, !fromUav, (uint32_t) b.AccessAfter);
                if (slot >= 0)
                    RecordEnhanced(cmdList, b, g_items[slot]);
            }
        }
    }
    t_inside = false;
}

void GiCapture_Dx12::OnBarrier(ID3D12GraphicsCommandList* cmdList, UINT count, const D3D12_RESOURCE_BARRIER* barriers)
{
    if (ID3D12Resource* watched = g_camWatchDest.load(std::memory_order_relaxed); watched != nullptr && !t_inside && barriers != nullptr)
        for (UINT i = 0; i < count; ++i)
            if (barriers[i].Type == D3D12_RESOURCE_BARRIER_TYPE_TRANSITION && barriers[i].Transition.pResource == watched)
            {
                g_camWatchLegacy.fetch_add(1, std::memory_order_relaxed);
                if (g_camWatchLines.fetch_add(1, std::memory_order_relaxed) < 24)
                    LOG_INFO("GI capture: the block's buffer: the game's legacy barrier on list {:X}: state {:X} > {:X}", (uint64_t) cmdList, (uint32_t) barriers[i].Transition.StateBefore, (uint32_t) barriers[i].Transition.StateAfter);
            }
    const int phase = g_phase.load(std::memory_order_relaxed);
    if ((phase != Surveying && phase != Capturing) || t_inside || cmdList == nullptr || barriers == nullptr)
        return;
    const D3D12_COMMAND_LIST_TYPE type = cmdList->GetType();
    if (type != D3D12_COMMAND_LIST_TYPE_DIRECT && type != D3D12_COMMAND_LIST_TYPE_COMPUTE)
        return;

    t_inside = true;
    std::lock_guard lock(g_mutex);
    if (phase == Surveying)
        g_barriersSeen += count;
    const uint64_t present = g_present.load();
    const int frame = phase == Capturing ? (int) (present - g_captureStart.load()) : 0;
    for (UINT i = 0; i < count; ++i)
    {
        const D3D12_RESOURCE_BARRIER& b = barriers[i];
        if (b.Type != D3D12_RESOURCE_BARRIER_TYPE_TRANSITION || b.Flags == D3D12_RESOURCE_BARRIER_FLAG_BEGIN_ONLY ||
            b.Transition.pResource == nullptr)
            continue;
        if (phase == Surveying)
        {
            const D3D12_RESOURCE_DESC seenDesc = b.Transition.pResource->GetDesc();
            if (seenDesc.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER)
                if (Event* e = AddEvent(cmdList, 'L', NoteTexture(cmdList, b.Transition.pResource, seenDesc), present); e != nullptr)
                {
                    e->accessBefore = (uint32_t) b.Transition.StateBefore;
                    e->accessAfter = (uint32_t) b.Transition.StateAfter;
                    e->sub = b.Transition.Subresource;
                }
        }
        const D3D12_RESOURCE_STATES written = D3D12_RESOURCE_STATE_UNORDERED_ACCESS | D3D12_RESOURCE_STATE_RENDER_TARGET;
        if ((b.Transition.StateBefore & written) == 0 || (b.Transition.StateAfter & written) != 0)
            continue;
        if (b.Transition.Subresource != D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES && b.Transition.Subresource != 0)
            continue;
        ID3D12Resource* resource = b.Transition.pResource;
        const D3D12_RESOURCE_DESC desc = resource->GetDesc();

        if (phase == Surveying)
        {
            if (desc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER)
            {
                ++g_buffersSeen;
                g_bufferBytes += desc.Width;
                continue;
            }
        }
        const int slot = OnWrite(cmdList, resource, desc, phase, frame,
                                 (b.Transition.StateBefore & D3D12_RESOURCE_STATE_UNORDERED_ACCESS) != 0,
                                 (b.Transition.StateBefore & D3D12_RESOURCE_STATE_RENDER_TARGET) != 0,
                                 (uint32_t) b.Transition.StateAfter);
        if (slot >= 0)
            Record(cmdList, resource, g_items[slot], b.Transition.StateAfter, nullptr);
    }
    t_inside = false;
}

void GiCapture_Dx12::OnEvaluate(ID3D12GraphicsCommandList* cmdList, int featureId, NVSDK_NGX_Parameter* params)
{
    if ((g_histMode.load(std::memory_order_relaxed) != 0 || g_clipState.load(std::memory_order_relaxed) != 0) && params != nullptr)
    {
        g_histRenderW.store(GetUInt(params, NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Width));
        g_histRenderH.store(GetUInt(params, NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Height));
    }
    if (featureId == kSuperSampling || featureId == kRayReconstruction)
        ClipOnEvaluate(cmdList, featureId, params);
    const int phase = g_phase.load(std::memory_order_relaxed);
    if ((phase != Surveying && phase != Capturing) || cmdList == nullptr || params == nullptr ||
        (featureId != kSuperSampling && featureId != kRayReconstruction) ||
        cmdList->GetType() != D3D12_COMMAND_LIST_TYPE_DIRECT)
        return;
    ID3D12Resource* colour = GetResource(params, NVSDK_NGX_Parameter_Color);
    ID3D12Resource* depth = GetResource(params, NVSDK_NGX_Parameter_Depth);
    if (colour == nullptr || depth == nullptr)
        return;
    UINT w = GetUInt(params, NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Width);
    UINT h = GetUInt(params, NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Height);
    const D3D12_RESOURCE_DESC cd = colour->GetDesc(), dd = depth->GetDesc();
    if (w == 0 || h == 0)
    {
        w = (UINT) cd.Width;
        h = cd.Height;
    }

    t_inside = true;
    std::lock_guard lock(g_mutex);
    if (phase == Surveying)
    {
        g_renderW = w;
        g_renderH = h;
        g_colourX = GetUInt(params, NVSDK_NGX_Parameter_DLSS_Input_Color_Subrect_Base_X);
        g_colourY = GetUInt(params, NVSDK_NGX_Parameter_DLSS_Input_Color_Subrect_Base_Y);
        g_colourDesc = cd;
        g_depthDesc = dd;
        g_haveInputs = g_colourX + w <= cd.Width && g_colourY + h <= cd.Height;
    }
    else
    {
        const uint64_t present = g_present.load();
        const int frame = (int) (present - g_captureStart.load());
        if (frame >= 0 && frame < kFrames && present != g_lastEvaluatePresent && w == g_renderW && h == g_renderH &&
            cd.Format == g_colourDesc.Format && dd.Format == g_depthDesc.Format)
        {
            g_lastEvaluatePresent = present;
            const std::string colourName = std::format("f{:02}_colour.dds", frame);
            const std::string depthName = std::format("f{:02}_depth.dds", frame);
            const D3D12_BOX colourBox { g_colourX, g_colourY, 0, g_colourX + w, g_colourY + h, 1 };
            for (auto& item : g_items)
            {
                if (item.fileName == colourName)
                    Record(cmdList, colour, item, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, &colourBox);
                else if (item.fileName == depthName)
                    Record(cmdList, depth, item, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, nullptr);
            }
            // The steps on, and the camera, in the capture itself: not to be read back from the log's key presses.
            g_frameInfo.push_back(std::format("frame {} present {} parity {} feature {} history step {} camera effect {} queues {} row {} jitter {} {} position {} {} {}", frame, present, present & 1, featureId,
                                              g_histMode.load(), g_camEffect.load(), g_camQueues.load(), g_lastCamera.clipToPrevClip[8], g_lastCamera.jitter[0], g_lastCamera.jitter[1], g_lastCamera.pos[0],
                                              g_lastCamera.pos[1], g_lastCamera.pos[2]));
        }
    }
    t_inside = false;
}

void GiCapture_Dx12::OnPresent(ID3D12CommandQueue* queue)
{
    if (!Enabled())
        return;
    const uint64_t present = g_present.fetch_add(1) + 1;
    if (g_histMode.load(std::memory_order_relaxed) != 0 || g_clipState.load(std::memory_order_relaxed) != 0)
        HistOnPresent(present);
    ClipOnPresent(present, queue);
    if (g_camEffect.load(std::memory_order_relaxed) != 0)
    {
        // By now Streamline has been told this frame's camera: was what went out the camera the game meant?
        float last = 0, older = 0;
        {
            std::lock_guard lock(g_mutex);
            last = g_lastCamera.clipToPrevClip[8];
            older = g_olderCamera.clipToPrevClip[8];
        }
        {
            std::lock_guard lock(g_camMutex);
            for (const float row : g_camRows)
            {
                if (std::abs(row - last) < 1e-5f || std::abs(row - older) < 1e-5f)
                    ++g_camRowMatched;
                else
                {
                    ++g_camRowMissed;
                    if (g_camRowLines++ < 4)
                        LOG_INFO("GI capture: camera test: present {}: the block's row {} is neither of Streamline's {} {}", present, row, last, older);
                }
            }
            g_camRows.clear();
        }
        if (present % 600 == 0)
            LOG_INFO("GI capture: camera test: {}", CameraTestStatus());
    }
    const int phase = g_phase.load();
    if (phase == Requested)
    {
        std::lock_guard lock(g_mutex);
        g_seen.clear();
        g_nextOrder = 0;
        g_barriersSeen = g_buffersSeen = g_bufferBytes = g_globalBarriers = g_textureBarriers = 0;
        g_haveInputs = false;
        g_enhancedBarriers.store(0);
        g_events.clear();
        g_events.reserve(40000);
        g_work.clear();
        g_eventsDropped = g_submits = 0;
        g_cameras.clear();
        g_bound.clear();
        g_blockText.clear();
        g_heaps.clear();
        g_copyDests.clear();
        g_tableContents.clear();
        g_tableExtents.clear();
        g_tablesDeclared.store(0);
        g_tablesUnbounded.store(0);
        g_tablesUnknown.store(0);
        g_seenOrders.clear();
        g_copyDumpText.clear();
        g_copyDumps = 0;
        g_surveyEnd = present + kSurveyPresents;
        g_phase.store(Surveying);
    }
    else if (phase == Surveying)
    {
        std::lock_guard lock(g_mutex);
        std::vector<CameraLine> eyes;
        if (g_haveLastCamera)
            eyes.push_back(g_lastCamera);
        if (g_haveOlderCamera)
            eyes.push_back(g_olderCamera);
        if (eyes.empty())
            LOG_WARN("GI capture: camera readers: no camera from Streamline yet, {} binds not looked at", g_bound.size());
        else if (g_bound.empty())
            LOG_WARN("GI capture: camera readers: present {}: no root CBV, root SRV or CBV made since the last Present", present - 1);
        else
        {
            g_scansRunning.fetch_add(1);
            std::thread(ScanBoundOnWorker, present - 1, std::move(g_bound), std::move(eyes)).detach();
        }
        g_bound.clear();
        if (present >= g_surveyEnd)
        {
            if (!g_sweeping.exchange(true))
            {
                // A game that calls it less than once a frame: the last one still names the projection, and its present says how old.
                std::vector<CameraLine> cameras = g_cameras;
                if (cameras.empty() && g_haveLastCamera)
                    cameras.push_back(g_lastCamera);
                std::thread(SweepOnWorker, std::move(cameras), g_cameras.size()).detach();
            }
            else
                LOG_WARN("GI capture: camera hunt: the last one is still running, none this time");
            Plan();
        }
    }
    else if (phase == Waiting)
    {
        bool start = false;
        {
            std::lock_guard lock(g_mutex);
            start = GetTickCount64() >= g_startAtTick;
            if (start)
                g_status = "GI capture: capturing, hold still";
        }
        if (start)
        {
            g_captureStart.store(present);
            g_phase.store(Capturing);
        }
    }
    else if (phase == Capturing)
    {
        if (present >= g_captureStart.load() + kFrames)
        {
            std::lock_guard lock(g_mutex);
            g_drainFrom = present;
            g_status = "GI capture: writing";
            g_phase.store(Draining);
        }
    }
    else if (phase == Draining && queue != nullptr)
    {
        uint64_t drainFrom = 0;
        {
            std::lock_guard lock(g_mutex);
            drainFrom = g_drainFrom;
        }
        if (present >= drainFrom + kDrainPresents)
        {
            static uint64_t fenceValue = 0;
            ++fenceValue;
            if (FAILED(queue->Signal(g_fence, fenceValue)))
                LOG_WARN("GI capture: signalling the fence failed");
            g_phase.store(Writing);
            std::thread(WriteOnWorker, fenceValue).detach();
        }
    }
}
