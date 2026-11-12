#include "pch.h"
#include "FrameCapture_Dx12.h"

#include "Config.h"
#include "Logger.h"
#include "Util.h"
#include "inputs/BindingCapture_Dx12.h"
#include "inputs/ConstantCapture_Dx12.h"

#include <include/d3dx/d3dx12.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <format>
#include <fstream>
#include <mutex>
#include <unordered_map>
#include <string>
#include <thread>
#include <vector>

namespace
{
// The capture runs on the threads that record the frame and call Present; only the arm request crosses threads
// (the overlay's hotkey handling). The candidate list has its own lock because the view-creation hooks offer
// resources from several threads.
std::atomic<bool> g_armRequested { false };
uint64_t g_armAt = 0; // tick count the delay expires at, 0 when not waiting
bool g_capturing = false;
int g_frame = 0;
bool g_eyeCapturedThisPresent = false;
int g_presentsWithoutEye = 0;
int g_repeatsLeft = 0;
// Six eye frames: a candidate copied at its bind holds the *previous* frame's finished content, so the frame it
// belongs to is one behind the eye images, and a same-eye pair needs two more on top of that.
constexpr int kCaptureFrames = 6;
constexpr int kMaxPresentsWithoutEye = 240;

// Readback is charged against a budget: an eye image is ~62 MB and a candidate ~31 MB, and a capture that quietly
// allocated gigabytes of readback heap in VR would be felt as a stall, not read as a number.
constexpr uint64_t kByteBudget = 1600ull << 20;
uint64_t g_bytes = 0;

// With the net widened past two-channel formats there can be more eye-sized targets than a capture can afford, so
// the ones actually copied are chosen once when the capture starts, by format priority, and then held for all six
// frames -- the warp test needs the same texture in every frame, not whichever bound first.
// Four rather than three: the two shapes worth testing now are R8G8 at full and half resolution, and half-res
// R8G8 is only ~6 MB a frame, so the extra slot costs almost nothing and covers a packed format as well.
constexpr size_t kMaxCopiedPerFrame = 4;
std::vector<ID3D12Resource*> g_chosen;

// The same choice as g_chosen, by shape instead of by pointer. The compute path needs this: FC6 keeps
// several instances of a target and ping-pongs between them, so the instance a dispatch writes this frame
// is often not the instance that was ranked when the capture started -- three of run 3's candidates were
// copied on every other frame or not at all for exactly this reason. The ranking is already deduplicated by
// shape and format, so matching that way loses nothing and holds no game pointer.
struct ChosenShape
{
    UINT64 width = 0;
    UINT height = 0;
    DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;
};
std::vector<ChosenShape> g_chosenShapes;

// Three at a time is not enough to see a whole G-buffer: the first run copied the three highest-priority formats
// and ruled them all out, leaving R11G11B10 and R10G10B10A2 -- the two most likely places for a packed motion
// pair -- untouched below the cutoff. So the ranked list is swept rather than topped: each capture within one arm
// takes the next group of kMaxCopiedPerFrame, wrapping at the end, and [Inputs] CaptureRepeat decides how much of
// the list one F7 covers. Reset on each arm so a fresh F7 is predictable and starts at the top.
size_t g_sweepOffset = 0;

// Guards everything a copy touches: g_copiedThisFrame, g_snapshots, g_bytes and g_chosen. ResTrack offers
// candidates from several recording threads at once, and without this the same resource was copied twice in a
// frame (30 textures read back where 24 were expected) because two threads passed the "already copied?" check
// together. Take it *after* StartIfDue, which reaches ChooseCandidates and takes it itself.
std::mutex g_captureLock;

// Candidates are offered by the view-creation hooks long before a capture is armed, so they are registered as they
// appear.
//
// The registry deliberately holds NO reference. It used to, back when the upstream score gate meant two textures
// reached it; once the net was widened past two-channel formats that turned into dozens of pinned render targets
// and broke display mode changes outright -- the game frees its render targets on a resolution or borderless
// switch, and a swapchain cannot resize while references are outstanding. So the descriptor is copied at
// registration and the pointer kept only as an identity token, never dereferenced outside a bind callback where
// the game itself guarantees the resource is alive. Nothing in this file holds a render target reference at any
// point, so nothing here can block a swapchain resize or a mode change.
struct Candidate
{
    ID3D12Resource* resource = nullptr;
    bool seenAsRenderTarget = false;
    // A texture a compute pass writes through a UAV is never bound by OMSetRenderTargets, so it could never be
    // chosen or copied. That is very likely why five captures of a game whose TAA is demonstrably running never
    // found a motion buffer: Dunia is compute-heavy, and everything copied so far has been a render target.
    bool seenAsUav = false;
    D3D12_RESOURCE_DESC desc {};
};
std::mutex g_candidateLock;
std::vector<Candidate> g_candidates;
// The first widened run filled a 64-entry registry and selection then ran on a truncated list. An entry is now a
// descriptor and a pointer and nothing else -- no reference, no resource kept alive -- so the cap can be generous.
constexpr size_t kMaxCandidates = 256;

// One copy per candidate per frame, at the first bind of that frame.
std::vector<ID3D12Resource*> g_copiedThisFrame;

// Per command list, for the compute path. Recorded order is execution order within one list, so a transition
// seen here before a dispatch is the state that dispatch will run with.
constexpr size_t kMaxUavsPerDispatch = 8;
constexpr size_t kMaxTrackedStates = 256;
struct ListState
{
    std::vector<ID3D12Resource*> uavsSinceDispatch;
    std::unordered_map<ID3D12Resource*, D3D12_RESOURCE_STATES> states;
};
// Lock order is g_captureLock then g_listLock, and never the reverse: the copies below go out through the
// game's own ResourceBarrier, which comes straight back into NoteBarriers while g_captureLock is held.
std::mutex g_listLock;
std::unordered_map<ID3D12GraphicsCommandList*, ListState> g_lists;

// Compute pipelines and their DXBC, by pipeline pointer. Lives from load, not just during a capture.
constexpr size_t kMaxShaderBytes = 64u << 20;
std::mutex g_shaderLock; // innermost: never takes another lock while held
std::unordered_map<ID3D12PipelineState*, std::vector<char>> g_shaders;
size_t g_shaderBytes = 0;
// What this capture has decided to write: pipeline -> its bytecode, filled when a candidate is copied.
std::unordered_map<ID3D12PipelineState*, std::vector<char>> g_shadersToWrite;

// The eye image's shape, remembered across captures so the bind-time filter has something to compare against
// before this capture's first eye image has been seen.
D3D12_RESOURCE_DESC g_lastEyeDesc {};

struct Snapshot
{
    int frame = 0;
    std::string kind;                 // "eye" or "mvbind"
    ID3D12Resource* source = nullptr; // not held; for the index only
    D3D12_RESOURCE_DESC desc {};
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint {};
    UINT rows = 0;
    UINT64 rowBytes = 0;
    ID3D12Resource* readback = nullptr;
    D3D12_RESOURCE_STATES assumedState = D3D12_RESOURCE_STATE_COMMON;
};
std::vector<Snapshot> g_snapshots;

ID3D12Fence* g_fence = nullptr;
HANDLE g_fenceEvent = nullptr;
UINT64 g_fenceValue = 0;

bool Diagnostics() { return Config::Instance()->DuniaTaaDiagnostics.value_or_default(); }

const char* StateName(D3D12_RESOURCE_STATES state)
{
    switch (state)
    {
    case D3D12_RESOURCE_STATE_RENDER_TARGET:
        return "render target";
    case D3D12_RESOURCE_STATE_UNORDERED_ACCESS:
        return "unordered access";
    case D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE:
        return "non-pixel shader resource";
    case D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE:
        return "pixel shader resource";
    case D3D12_RESOURCE_STATE_COPY_SOURCE:
        return "copy source";
    case D3D12_RESOURCE_STATE_COMMON:
        return "common";
    default:
        return "other";
    }
}

// The upstream candidate filter only asks for a two-channel format and a size above the menu threshold, which the
// game's shadow and lighting targets also pass. What gets copied is decided against the eye image: same shape, and
// no smaller than a quarter of it.
bool ShapeMatchesEye(const D3D12_RESOURCE_DESC& desc, const D3D12_RESOURCE_DESC& eye)
{
    if (desc.Height == 0 || eye.Height == 0 || eye.Width == 0)
        return true; // no eye image seen yet: let it through, the budget is the backstop

    const double aspect = (double) desc.Width / desc.Height;
    const double eyeAspect = (double) eye.Width / eye.Height;
    if (std::abs(aspect - eyeAspect) > eyeAspect * 0.05)
        return false;

    const double area = (double) desc.Width * desc.Height;
    const double eyeArea = (double) eye.Width * eye.Height;
    return area <= eyeArea * 1.05 && area >= eyeArea * 0.25;
}

// Typeless targets are picked up by whichever view the game makes of them; for deciding whether a texture is worth
// copying, the channel layout is all that matters.
DXGI_FORMAT CaptureColorFormat(DXGI_FORMAT format)
{
    switch (format)
    {
    case DXGI_FORMAT_R16G16_TYPELESS:
        return DXGI_FORMAT_R16G16_UNORM;
    case DXGI_FORMAT_R32G32_TYPELESS:
        return DXGI_FORMAT_R32G32_FLOAT;
    case DXGI_FORMAT_R8G8_TYPELESS:
        return DXGI_FORMAT_R8G8_UNORM;
    case DXGI_FORMAT_R8G8B8A8_TYPELESS:
        return DXGI_FORMAT_R8G8B8A8_UNORM;
    case DXGI_FORMAT_R16G16B16A16_TYPELESS:
        return DXGI_FORMAT_R16G16B16A16_FLOAT;
    case DXGI_FORMAT_R10G10B10A2_TYPELESS:
        return DXGI_FORMAT_R10G10B10A2_UNORM;
    default:
        return format;
    }
}

// A copy of an eye-sized target costs 31-129 MB a frame and six frames are taken, so the net is wide but ranked.
// Two-channel targets first -- FC5's motion turned out to be a half-res R8G8, and a dedicated pair is still the
// most likely shape. Then the packed 32-bit formats a G-buffer uses, where a motion pair can sit in two of the
// channels. The 8-byte-per-pixel ones are worth knowing about but not worth 129 MB a frame on spec, so they score
// 0: logged in the inventory, never copied.
int PriorityForFormat(DXGI_FORMAT format)
{
    switch (CaptureColorFormat(format))
    {
    // Reordered once there was evidence instead of guesswork. R8G8 is FC5's shape -- its motion vectors were a
    // half-res R8G8 -- and the binding capture found two half-res R8G8 targets read in 36 groups each that have
    // never been copied. Every R16G16 has now been tested three times and come back static, so it goes below the
    // packed G-buffer formats, which have never been tested at all.
    case DXGI_FORMAT_R8G8_UNORM:
    case DXGI_FORMAT_R8G8_SNORM:
        return 7;
    case DXGI_FORMAT_R11G11B10_FLOAT:
        return 6;
    case DXGI_FORMAT_R10G10B10A2_UNORM:
        return 5;
    case DXGI_FORMAT_R16G16_FLOAT:
    case DXGI_FORMAT_R16G16_UNORM:
    case DXGI_FORMAT_R16G16_SNORM:
        return 4;
    case DXGI_FORMAT_R8G8B8A8_UNORM:
    case DXGI_FORMAT_R8G8B8A8_SNORM:
        return 2;
    case DXGI_FORMAT_R32G32_FLOAT:
        return 1;
    case DXGI_FORMAT_R16G16B16A16_FLOAT:
    case DXGI_FORMAT_R16G16B16A16_SNORM:
    case DXGI_FORMAT_R16G16B16A16_UNORM:
        return 0; // inventory only
    default:
        return -1; // not a plausible carrier of a motion pair at all
    }
}

void Barrier(ID3D12GraphicsCommandList* cmd, ID3D12Resource* resource, D3D12_RESOURCE_STATES before,
             D3D12_RESOURCE_STATES after)
{
    if (cmd == nullptr || resource == nullptr || before == after)
        return;

    D3D12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::Transition(resource, before, after);
    cmd->ResourceBarrier(1, &barrier);
}

void ReleaseSnapshots()
{
    for (auto& snapshot : g_snapshots)
    {
        if (snapshot.readback != nullptr)
            snapshot.readback->Release();
    }
    g_snapshots.clear();
    g_bytes = 0;
}

void Stop()
{
    g_capturing = false;
    g_frame = 0;
    g_eyeCapturedThisPresent = false;
    g_presentsWithoutEye = 0;
    {
        std::lock_guard<std::mutex> lock(g_captureLock);
        g_copiedThisFrame.clear();
        g_chosen.clear();
        g_chosenShapes.clear();
        ReleaseSnapshots();
    }
    ConstantCapture_Dx12::Stop();
    BindingCapture_Dx12::Stop();
}

// Candidates register from the moment diagnostics are on, long before F7, so by the time a capture starts the
// inventory of this frame's eye-sized targets is already in hand and the best few can be picked from it. Only on a
// capture armed in the first seconds of a level will this come up short, and then the bind order decides.
void ChooseCandidates()
{
    std::lock_guard<std::mutex> chosenLock(g_captureLock);
    g_chosen.clear();
    g_chosenShapes.clear();

    // Ranked from the descriptors copied at registration: the registry holds no reference, so a pointer in it may
    // already be dead and must not be dereferenced here.
    std::vector<std::pair<int, Candidate>> ranked;
    {
        std::lock_guard<std::mutex> lock(g_candidateLock);
        for (const auto& candidate : g_candidates)
        {
            if (candidate.resource == nullptr || (!candidate.seenAsRenderTarget && !candidate.seenAsUav))
                continue;
            const int priority = PriorityForFormat(candidate.desc.Format);
            if (priority <= 0 || !ShapeMatchesEye(candidate.desc, g_lastEyeDesc))
                continue;
            ranked.push_back({ priority, candidate });
        }
    }

    std::stable_sort(ranked.begin(), ranked.end(),
                     [](const auto& a, const auto& b) { return a.first > b.first; });

    // One entry per distinct shape and format. The registry holds resources, and an engine keeps several of the
    // same kind around to ping-pong between, so without this the sweep would spend capture after capture on
    // different instances of a target it has already ruled out. The inventory in the log is per shape and format
    // too, so a swept list and that log line up.
    {
        std::vector<std::pair<int, Candidate>> unique;
        for (const auto& entry : ranked)
        {
            bool seen = false;
            for (const auto& kept : unique)
            {
                if (kept.second.desc.Width == entry.second.desc.Width &&
                    kept.second.desc.Height == entry.second.desc.Height &&
                    kept.second.desc.Format == entry.second.desc.Format)
                {
                    seen = true;
                    break;
                }
            }
            if (!seen)
                unique.push_back(entry);
        }
        ranked.swap(unique);
    }

    if (ranked.empty())
    {
        LOG_INFO("Frame capture: no candidate chosen yet, taking them in bind order this capture");
        return;
    }

    if (g_sweepOffset >= ranked.size())
        g_sweepOffset = 0;

    LOG_INFO("Frame capture: {} candidates ranked, this capture takes {} starting at {}", ranked.size(),
             kMaxCopiedPerFrame, g_sweepOffset);

    for (size_t i = g_sweepOffset; i < ranked.size() && g_chosen.size() < kMaxCopiedPerFrame; ++i)
    {
        const auto& entry = ranked[i];

        // Still no reference taken. The pointer is only ever compared, never dereferenced: what gets copied is the
        // live resource the bind callback hands over, and a candidate the game has since freed simply stops
        // matching. Nothing this file holds can keep a render target alive across a resolution change.
        g_chosen.push_back(entry.second.resource);
        g_chosenShapes.push_back(
            { entry.second.desc.Width, entry.second.desc.Height, entry.second.desc.Format });

        LOG_INFO("Frame capture: copying [{}] {:X} {}x{} fmt {} (priority {})", i, (size_t) entry.second.resource,
                 entry.second.desc.Width, entry.second.desc.Height, (UINT) entry.second.desc.Format, entry.first);
    }

    g_sweepOffset += kMaxCopiedPerFrame;
    if (g_sweepOffset >= ranked.size())
    {
        g_sweepOffset = 0;
        LOG_INFO("Frame capture: that was the end of the ranked list, the next capture starts over at the top");
    }
}

// True once the armed delay has expired. The capture starts on whichever callback gets here first: a candidate is
// bound well before the eye image is finished, so waiting for the splice would leave frame 0 without bind copies.
bool StartIfDue()
{
    if (g_armRequested.exchange(false))
    {
        const int delay = Config::Instance()->CaptureDelaySeconds.value_or_default();
        g_armAt = GetTickCount64() + (uint64_t) (delay > 0 ? delay : 0) * 1000;
        g_repeatsLeft = Config::Instance()->CaptureRepeat.value_or_default();
        if (g_repeatsLeft < 1)
            g_repeatsLeft = 1;
        LOG_INFO("Frame capture: armed, starting in {} s ({} capture(s))", delay, g_repeatsLeft);
    }

    if (g_capturing)
        return true;
    if (g_armAt == 0 || GetTickCount64() < g_armAt)
        return false;

    g_armAt = 0;
    g_capturing = true;
    g_frame = 0;
    g_presentsWithoutEye = 0;
    {
        std::lock_guard<std::mutex> lock(g_captureLock);
        g_bytes = 0;
        g_copiedThisFrame.clear();
    }
    ChooseCandidates();
    ConstantCapture_Dx12::Start();
    BindingCapture_Dx12::Start();
    return true;
}

// Copies mip 0 of `source` into a fresh readback buffer, recorded on the game's own list so the copy sits where the
// resource state is known rather than at some later point of our choosing.
//
// The caller must hold g_captureLock: this reads and writes g_bytes and g_snapshots, and candidates arrive on
// several recording threads at once.
bool Copy(ID3D12GraphicsCommandList* cmd, ID3D12Device* device, ID3D12Resource* source, D3D12_RESOURCE_STATES state,
          const char* kind)
{
    if (source == nullptr || device == nullptr)
        return false;

    const auto desc = source->GetDesc();
    if (desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || desc.SampleDesc.Count > 1)
        return false;

    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint {};
    UINT rows = 0;
    UINT64 rowBytes = 0;
    UINT64 total = 0;
    device->GetCopyableFootprints(&desc, 0, 1, 0, &footprint, &rows, &rowBytes, &total);
    if (total == 0)
        return false;

    if (g_bytes + total > kByteBudget)
    {
        static bool loggedBudget = false;
        if (!loggedBudget)
        {
            loggedBudget = true;
            LOG_WARN("Frame capture: {} MB budget reached, skipping the rest of this capture", kByteBudget >> 20);
        }
        return false;
    }

    ID3D12Resource* readback = nullptr;
    D3D12_HEAP_PROPERTIES heap = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK);
    auto bufferDesc = CD3DX12_RESOURCE_DESC::Buffer(total);
    if (FAILED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &bufferDesc,
                                               D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&readback))) ||
        readback == nullptr)
    {
        LOG_WARN("Frame capture: readback buffer for {} ({} MB) failed", kind, total >> 20);
        return false;
    }

    Barrier(cmd, source, state, D3D12_RESOURCE_STATE_COPY_SOURCE);
    const D3D12_TEXTURE_COPY_LOCATION dst = CD3DX12_TEXTURE_COPY_LOCATION(readback, footprint);
    const D3D12_TEXTURE_COPY_LOCATION src = CD3DX12_TEXTURE_COPY_LOCATION(source, 0);
    cmd->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    Barrier(cmd, source, D3D12_RESOURCE_STATE_COPY_SOURCE, state);

    g_bytes += total;
    g_snapshots.push_back({ g_frame, kind, source, desc, footprint, rows, rowBytes, readback, state });
    return true;
}

// Mip 0 of a read-back texture, rows packed.
struct Pixels
{
    std::filesystem::path path;
    DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;
    UINT width = 0;
    UINT height = 0;
    UINT rowBytes = 0;
    std::vector<char> rows;
};

void WriteDds(const Pixels& pixels)
{
    uint32_t header[32] = {};
    header[0] = 0x20534444;                     // "DDS "
    header[1] = 124;
    header[2] = 0x1 | 0x2 | 0x4 | 0x8 | 0x1000; // caps, height, width, pitch, pixel format
    header[3] = pixels.height;
    header[4] = pixels.width;
    header[5] = pixels.rowBytes;
    header[7] = 1;                              // mips
    header[19] = 32;                            // pixel format size
    header[20] = 0x4;                           // FOURCC
    header[21] = 0x30315844;                    // "DX10"
    header[27] = 0x1000;                        // TEXTURE
    const uint32_t dx10[5] = { (uint32_t) pixels.format, 3 /* TEXTURE2D */, 0, 1, 0 };

    std::ofstream file(pixels.path, std::ios::binary);
    file.write(reinterpret_cast<const char*>(header), sizeof(header));
    file.write(reinterpret_cast<const char*>(dx10), sizeof(dx10));
    file.write(pixels.rows.data(), (std::streamsize) pixels.rows.size());
}

bool EnsureFence(ID3D12Device* device)
{
    if (g_fence != nullptr)
        return true;
    if (device == nullptr)
        return false;
    if (FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&g_fence))))
        return false;
    g_fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    return g_fenceEvent != nullptr;
}

// Everything captured was recorded on the game's lists, so waiting on the queue that just presented covers it.
bool WaitForCopies(ID3D12CommandQueue* queue)
{
    if (queue == nullptr || g_fence == nullptr || g_fenceEvent == nullptr)
        return false;
    if (FAILED(queue->Signal(g_fence, ++g_fenceValue)))
        return false;
    if (g_fence->GetCompletedValue() >= g_fenceValue)
        return true;
    if (FAILED(g_fence->SetEventOnCompletion(g_fenceValue, g_fenceEvent)))
        return false;
    return WaitForSingleObject(g_fenceEvent, 5000) == WAIT_OBJECT_0;
}

void Flush(ID3D12CommandQueue* queue)
{
    // Closed before the snapshots are read: another thread can still be recording the next frame, and a Copy
    // landing in g_snapshots while this iterates it would be reading a vector as it reallocates. With this clear,
    // StartIfDue turns away any copy that has not already been recorded.
    {
        std::lock_guard<std::mutex> lock(g_captureLock);
        g_capturing = false;
    }

    if (g_snapshots.empty())
    {
        LOG_WARN("Frame capture: nothing was copied");
        Stop();
        return;
    }

    if (!WaitForCopies(queue))
    {
        LOG_ERROR("Frame capture: the GPU did not finish the copies, dropping them");
        Stop();
        return;
    }

    const auto now = std::time(nullptr);
    std::tm local {};
    localtime_s(&local, &now);
    const auto stamp = std::format("{:04}{:02}{:02}-{:02}{:02}{:02}", local.tm_year + 1900, local.tm_mon + 1,
                                   local.tm_mday, local.tm_hour, local.tm_min, local.tm_sec);
    const auto folder = Util::ExePath().parent_path() / "OptiCapture" / stamp;
    std::error_code error;
    std::filesystem::create_directories(folder, error);
    if (error)
    {
        LOG_ERROR("Frame capture: can't create {}: {}", folder.string(), error.message());
        Stop();
        return;
    }

    std::string index =
        "Far Cry 6 (D3D12) motion vector capture\n"
        "Consecutive eye frames. Whether n and n+2 are the same eye is ASSUMED, not measured -- the 20260919-211323\n"
        "capture looked like one continuous pan, so check the pairing before trusting a warp result.\n"
        "\"eye\" is the finished eye image at the splice point. \"mvbind\" is a candidate copied at the first\n"
        "render-target bind of that frame, so it holds what the PREVIOUS frame finished writing -- pair mvbind of\n"
        "frame n with eye frames n-1 and n-3. A bind needs no state guess: the game had to transition the resource\n"
        "to RENDER_TARGET to bind it there. \"mvcompute\" is a texture a compute pass writes, copied right after the\n"
        "Dispatch that wrote it -- the only way to reach one, since it never reaches OMSetRenderTargets at all. It\n"
        "holds THIS frame's writes, so pair mvcompute of frame n with eye frames n and n-2, not n-1 and n-3 like\n"
        "mvbind. (\"mvuav\", copied at the bind rather than the dispatch, is gone: a table bind says nothing about a\n"
        "resource's state, and barriering on that assumption hung the game.)\n"
        "cs_<pipeline>.dxbc is the compute shader that wrote a copied candidate. Dunia ships reflection names, so\n"
        "the shader names the texture.\n"
        "Candidates are no longer two-channel only: any eye-shaped target whose format could carry a motion pair is\n"
        "eligible, so a candidate may hold motion in two of its channels rather than in .xy. The few actually copied\n"
        "are chosen by format priority when the capture starts; the log lists the full inventory it chose from.\n\n"
        "file | kind | frame | source | width x height | format | flags | state\n";

    // Only the index is built here. Mapping and copying the readback used to happen on this thread too, and this
    // is the thread that Presents: a capture of around a gigabyte meant seconds of Map and memcpy between two
    // Presents, which in VR is a frozen image and then a black headset while the compositor gives up. All of it
    // now goes to the writer thread, which also releases the readback buffers when it is done with them. What is
    // left here is the fence wait, which is unavoidable and short.
    for (const auto& snapshot : g_snapshots)
    {
        const auto name = std::format("f{}_{}_{:X}.dds", snapshot.frame, snapshot.kind, (size_t) snapshot.source);
        index += std::format("{} | {} | {} | {:X} | {}x{} | {} | {:X} | {}\n", name, snapshot.kind, snapshot.frame,
                             (size_t) snapshot.source, snapshot.desc.Width, snapshot.desc.Height,
                             (UINT) snapshot.desc.Format, (UINT) snapshot.desc.Flags, StateName(snapshot.assumedState));
    }

    LOG_INFO("Frame capture: {} textures copied ({} MB), reading back and writing to {} in the background",
             g_snapshots.size(), g_bytes >> 20, folder.string());

    // Taken before Stop() clears it. Empty when nothing was recorded -- FC6 binds its constants through descriptor
    // tables, so the root CBV path sees nothing at all here.
    auto constants = ConstantCapture_Dx12::Take();
    auto bindings = BindingCapture_Dx12::Take();

    // Moved out, so the Stop() below has nothing left to release and the writer owns the readback buffers.
    std::vector<Snapshot> owned;
    owned.swap(g_snapshots);
    g_bytes = 0;

    std::unordered_map<ID3D12PipelineState*, std::vector<char>> shaders;
    shaders.swap(g_shadersToWrite);

    std::thread(
        [folder, index = std::move(index), constants = std::move(constants), bindings = std::move(bindings),
         owned = std::move(owned), shaders = std::move(shaders)]() mutable
        {
            size_t written = 0;
            for (auto& snapshot : owned)
            {
                const auto name =
                    std::format("f{}_{}_{:X}.dds", snapshot.frame, snapshot.kind, (size_t) snapshot.source);

                void* mapped = nullptr;
                // Null range = the whole resource. An explicit one has to stop at the buffer's real size, which is
                // RowPitch * (rows - 1) + rowBytes, and getting that wrong fails the Map and loses the capture.
                const HRESULT mapResult =
                    snapshot.readback == nullptr ? E_POINTER : snapshot.readback->Map(0, nullptr, &mapped);
                if (SUCCEEDED(mapResult) && mapped != nullptr)
                {
                    Pixels pixels { folder / name, snapshot.desc.Format, (UINT) snapshot.desc.Width,
                                    snapshot.desc.Height, (UINT) snapshot.rowBytes, {} };
                    pixels.rows.resize((size_t) snapshot.rowBytes * snapshot.rows);
                    for (UINT row = 0; row < snapshot.rows; ++row)
                    {
                        std::memcpy(pixels.rows.data() + (size_t) row * snapshot.rowBytes,
                                    (const char*) mapped + (size_t) row * snapshot.footprint.Footprint.RowPitch,
                                    (size_t) snapshot.rowBytes);
                    }
                    const D3D12_RANGE nothingWritten { 0, 0 };
                    snapshot.readback->Unmap(0, &nothingWritten);

                    WriteDds(pixels);
                    ++written;
                }
                else
                {
                    LOG_WARN("Frame capture: can't map the readback for {}: {:X}", name, (UINT) mapResult);
                }

                // Released as soon as this one is on disk, so peak memory is one image rather than the whole set.
                if (snapshot.readback != nullptr)
                {
                    snapshot.readback->Release();
                    snapshot.readback = nullptr;
                }
            }

            for (const auto& shader : shaders)
            {
                const auto name = std::format("cs_{:X}.dxbc", (size_t) shader.first);
                std::ofstream(folder / name, std::ios::binary)
                    .write(shader.second.data(), (std::streamsize) shader.second.size());
            }
            if (!shaders.empty())
                LOG_INFO("Frame capture: {} compute shaders written as cs_*.dxbc", shaders.size());

            std::ofstream(folder / "index.txt") << index;
            if (!constants.empty())
                std::ofstream(folder / "constants.txt") << constants;
            if (!bindings.empty())
                std::ofstream(folder / "bindings.txt") << bindings;
            LOG_INFO("Frame capture: {} of {} textures written to {}", written, owned.size(), folder.string());
        })
        .detach();

    Stop();
}
} // namespace

void FrameCapture_Dx12::Arm()
{
    // Said out loud, both ways. The toast never reaches the headset, so the log is the only feedback there is,
    // and until now a press left no trace at all until the capture actually started -- which meant a press that
    // went nowhere looked exactly like a press that never happened. That cost a run.
    if (!Diagnostics())
    {
        LOG_WARN("Frame capture: F7 pressed, but [Inputs] DuniaTaaDiagnostics is off, so there is nothing to arm");
        return;
    }
    LOG_INFO("Frame capture: F7 pressed, arming");
    // A fresh F7 starts the sweep at the top; the repeats within one arm walk down the ranked list.
    g_sweepOffset = 0;
    g_armRequested = true;
}

bool FrameCapture_Dx12::Busy() { return g_capturing || g_armAt != 0 || g_armRequested.load(); }

bool FrameCapture_Dx12::IsCaptureShape(const D3D12_RESOURCE_DESC& desc)
{
    if (desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || desc.SampleDesc.Count > 1)
        return false;
    if (desc.Width == desc.Height)
        return false; // shadow and lighting cubes; an eye is never square
    if (PriorityForFormat(desc.Format) < 0)
        return false;
    return ShapeMatchesEye(desc, g_lastEyeDesc);
}

int FrameCapture_Dx12::CapturePriority(const D3D12_RESOURCE_DESC& desc) { return PriorityForFormat(desc.Format); }

void FrameCapture_Dx12::NoteCandidate(ID3D12Resource* resource, bool boundAsRenderTarget,
                                      ID3D12GraphicsCommandList* cmd)
{
    if (!Diagnostics() || resource == nullptr)
        return;

    const auto desc = resource->GetDesc();
    if (desc.Width == desc.Height)
        return; // shadow and lighting cubes; an eye is never square, and there are enough of them to fill the list
    if (PriorityForFormat(desc.Format) < 0)
        return; // nothing that could hold a motion pair, whatever else it is
    // Eye-shaped, not merely non-square. Without this the registry fills with UI and post-process targets, which
    // is what the widened net did before this check was put back. Registration will not guess: until the splice
    // has reported an eye image there is nothing to be eye-shaped against, and the permissive "let it through"
    // that ShapeMatchesEye uses elsewhere is exactly wrong here.
    if (g_lastEyeDesc.Width == 0 || g_lastEyeDesc.Height == 0)
        return;
    if (!ShapeMatchesEye(desc, g_lastEyeDesc))
        return;

    {
        std::lock_guard<std::mutex> lock(g_candidateLock);
        bool known = false;
        for (auto& candidate : g_candidates)
        {
            if (candidate.resource == resource)
            {
                candidate.seenAsRenderTarget = candidate.seenAsRenderTarget || boundAsRenderTarget;
                candidate.seenAsUav = candidate.seenAsUav || !boundAsRenderTarget;
                known = true;
                break;
            }
        }

        if (!known)
        {
            if (g_candidates.size() >= kMaxCandidates)
                return;
            g_candidates.push_back({ resource, boundAsRenderTarget, !boundAsRenderTarget, desc });
            LOG_INFO("Frame capture: motion candidate {} registered: {:X} {}x{} fmt {} flags {:X}",
                     g_candidates.size(), (size_t) resource, desc.Width, desc.Height, (UINT) desc.Format,
                     (UINT) desc.Flags);
        }
    }

    // The capture itself. At the end of the frame, where the first version of this took its copies, D3D12's
    // transient allocators have long since handed that memory to other passes -- one candidate read back as
    // half-float noise with NaNs in it. Here the resource is being bound to be drawn into: it still holds what the
    // previous frame wrote, and the game had to put it in RENDER_TARGET state to bind it, so nothing is guessed.
    if (cmd == nullptr || !boundAsRenderTarget)
        return;
    if (!StartIfDue())
        return;
    if (!ShapeMatchesEye(desc, g_lastEyeDesc))
        return;
    if (PriorityForFormat(desc.Format) <= 0)
        return; // inventory only: the 8-byte-per-pixel formats are not worth 129 MB a frame on spec

    ID3D12Device* device = nullptr;
    if (FAILED(cmd->GetDevice(IID_PPV_ARGS(&device))) || device == nullptr)
        return;
    device->Release();
    EnsureFence(device);

    // Held across the whole check-and-copy. Splitting the two is what let two recording threads both decide this
    // resource had not been copied yet and each take one, which is where the duplicate copies came from.
    std::lock_guard<std::mutex> lock(g_captureLock);

    // The set was chosen when the capture started, and every frame copies the same textures so the warp test has
    // a series to work with. Empty only when nothing was registered in time, and then the cap does the limiting.
    if (!g_chosen.empty())
    {
        bool wanted = false;
        for (auto* chosen : g_chosen)
        {
            if (chosen == resource)
            {
                wanted = true;
                break;
            }
        }
        if (!wanted)
            return;
    }
    else if (g_copiedThisFrame.size() >= kMaxCopiedPerFrame)
    {
        return;
    }

    for (auto* copied : g_copiedThisFrame)
    {
        if (copied == resource)
            return;
    }

    if (Copy(cmd, device, resource, D3D12_RESOURCE_STATE_RENDER_TARGET, "mvbind"))
        g_copiedThisFrame.push_back(resource);
}

// Shared by NoteUavBound and AfterDispatch: is this a texture this capture would want a copy of?
namespace
{
bool WantedCandidate(ID3D12Resource* resource)
{
    if (resource == nullptr)
        return false;

    const auto desc = resource->GetDesc();
    if (desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || desc.SampleDesc.Count > 1)
        return false;
    if (desc.Width == desc.Height)
        return false;
    if (PriorityForFormat(desc.Format) <= 0)
        return false;
    return ShapeMatchesEye(desc, g_lastEyeDesc);
}
} // namespace

void FrameCapture_Dx12::NoteUavBound(ID3D12GraphicsCommandList* cmd, ID3D12Resource* resource)
{
    if (!g_capturing || cmd == nullptr || !WantedCandidate(resource))
        return;

    // Recorded as intent only. The copy waits for the Dispatch, which is the point D3D12 actually promises
    // something about the state of the UAVs a shader writes.
    std::lock_guard<std::mutex> lock(g_listLock);
    auto& list = g_lists[cmd];
    for (auto* seen : list.uavsSinceDispatch)
    {
        if (seen == resource)
            return;
    }
    if (list.uavsSinceDispatch.size() < kMaxUavsPerDispatch)
        list.uavsSinceDispatch.push_back(resource);
}

void FrameCapture_Dx12::NoteBarriers(ID3D12GraphicsCommandList* cmd, UINT count,
                                     const D3D12_RESOURCE_BARRIER* barriers)
{
    // Tracked from load, not only while a capture runs. A resource is usually transitioned into
    // UNORDERED_ACCESS well before the dispatch that writes it, and often before F7 was ever pressed -- the
    // first run skipped a candidate for exactly that reason.
    // And only with diagnostics: the barrier hook is also installed for real motion, and 500 barrier calls a
    // frame were each taking this lock for a record only a capture reads.
    if (cmd == nullptr || barriers == nullptr || count == 0 || !Diagnostics())
        return;

    std::lock_guard<std::mutex> lock(g_listLock);
    auto& list = g_lists[cmd];
    for (UINT i = 0; i < count; ++i)
    {
        const auto& barrier = barriers[i];
        if (barrier.Type != D3D12_RESOURCE_BARRIER_TYPE_TRANSITION)
            continue;
        // A split barrier only promises the state at its END half; in between the resource cannot be used.
        if ((barrier.Flags & D3D12_RESOURCE_BARRIER_FLAG_BEGIN_ONLY) != 0)
            continue;

        const auto& transition = barrier.Transition;
        if (transition.pResource == nullptr)
            continue;
        // Only whole-resource transitions are tracked: a per-subresource one leaves the rest of the resource in
        // another state, and the copy below takes subresource 0 of the whole thing.
        if (transition.Subresource != D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES)
            continue;
        if (list.states.size() < kMaxTrackedStates || list.states.count(transition.pResource) != 0)
            list.states[transition.pResource] = transition.StateAfter;
    }
}

void FrameCapture_Dx12::AfterDispatch(ID3D12GraphicsCommandList* cmd, ID3D12PipelineState* pipeline)
{
    if (!g_capturing || cmd == nullptr)
        return;

    std::vector<ID3D12Resource*> ready;
    {
        std::lock_guard<std::mutex> lock(g_listLock);
        auto it = g_lists.find(cmd);
        if (it == g_lists.end() || it->second.uavsSinceDispatch.empty())
            return;

        for (auto* resource : it->second.uavsSinceDispatch)
        {
            const auto state = it->second.states.find(resource);
            if (state == it->second.states.end())
            {
                // The game transitioned it on another list, or before this capture started listening. Not
                // knowing is a reason to do nothing: the barrier below would otherwise name a state the
                // resource is not in, which once hung the game.
                static int loggedUnknown = 0;
                if (loggedUnknown < 4)
                {
                    ++loggedUnknown;
                    const auto desc = resource->GetDesc();
                    LOG_INFO("Frame capture: UAV candidate {:X} {}x{} fmt {} was dispatched with no transition "
                             "recorded on its list; skipping it rather than assuming its state",
                             (size_t) resource, desc.Width, desc.Height, (UINT) desc.Format);
                }
                continue;
            }
            if (state->second != D3D12_RESOURCE_STATE_UNORDERED_ACCESS)
                continue;
            ready.push_back(resource);
        }
        it->second.uavsSinceDispatch.clear();
    }

    if (ready.empty())
        return;

    ID3D12Device* device = nullptr;
    if (FAILED(cmd->GetDevice(IID_PPV_ARGS(&device))) || device == nullptr)
        return;
    device->Release();
    EnsureFence(device);

    std::lock_guard<std::mutex> lock(g_captureLock);
    for (auto* resource : ready)
    {
        const auto desc = resource->GetDesc();
        if (!g_chosenShapes.empty())
        {
            bool wanted = false;
            for (const auto& shape : g_chosenShapes)
            {
                if (shape.width == desc.Width && shape.height == desc.Height && shape.format == desc.Format)
                {
                    wanted = true;
                    break;
                }
            }
            if (!wanted)
                continue;
        }
        else if (g_copiedThisFrame.size() >= kMaxCopiedPerFrame)
        {
            break;
        }

        bool already = false;
        for (auto* copied : g_copiedThisFrame)
        {
            if (copied == resource)
                already = true;
        }
        if (already)
            continue;

        // The dispatch has just been recorded and its writes are the whole point, so the copy has to wait for
        // them: a UAV barrier orders this list's later reads after that dispatch's writes.
        D3D12_RESOURCE_BARRIER uav {};
        uav.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        uav.UAV.pResource = resource;
        cmd->ResourceBarrier(1, &uav);

        if (Copy(cmd, device, resource, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, "mvcompute"))
        {
            g_copiedThisFrame.push_back(resource);

            // The shader that just wrote it. This is the point of the whole path: the picture says "smooth and
            // silhouette-sharp", the shader's reflection data says what the texture is for.
            if (pipeline == nullptr)
            {
                static bool loggedNoPipeline = false;
                if (!loggedNoPipeline)
                {
                    loggedNoPipeline = true;
                    LOG_INFO("Frame capture: the dispatch that wrote {:X} reported no pipeline -- the "
                             "SetPipelineState hook is not recording", (size_t) resource);
                }
            }
            else if (g_shadersToWrite.find(pipeline) == g_shadersToWrite.end())
            {
                std::lock_guard<std::mutex> shaders(g_shaderLock);
                const auto found = g_shaders.find(pipeline);
                if (found != g_shaders.end())
                {
                    g_shadersToWrite.emplace(pipeline, found->second);
                    LOG_INFO("Frame capture: kept the compute shader of pipeline {:X} ({} bytes) for {:X}",
                             (size_t) pipeline, found->second.size(), (size_t) resource);
                }
                else
                {
                    // Says outright that the writer's bytecode never reached us, rather than leaving an empty
                    // capture folder to be interpreted.
                    static int loggedUnknownPipeline = 0;
                    if (loggedUnknownPipeline < 4)
                    {
                        ++loggedUnknownPipeline;
                        LOG_INFO("Frame capture: the dispatch that wrote {:X} used pipeline {:X}, whose bytecode "
                                 "was never seen ({} known)",
                                 (size_t) resource, (size_t) pipeline, g_shaders.size());
                    }
                }
            }
            static bool loggedFirst = false;
            if (!loggedFirst)
            {
                loggedFirst = true;
                LOG_INFO("Frame capture: copying a compute-written candidate after its dispatch: {}x{} fmt {}",
                         desc.Width, desc.Height, (UINT) desc.Format);
            }
        }
    }
}

void FrameCapture_Dx12::NoteComputePipeline(ID3D12PipelineState* pipeline, const void* bytecode, size_t bytes)
{
    if (!Diagnostics() || pipeline == nullptr || bytecode == nullptr || bytes == 0)
        return;

    std::lock_guard<std::mutex> lock(g_shaderLock);
    static bool loggedFirst = false;
    if (!loggedFirst)
    {
        loggedFirst = true;
        LOG_INFO("Frame capture: keeping compute shader bytecode (first pipeline {:X}, {} bytes)",
                 (size_t) pipeline, bytes);
    }
    if (g_shaderBytes + bytes > kMaxShaderBytes || g_shaders.count(pipeline) != 0)
        return;

    const char* first = static_cast<const char*>(bytecode);
    g_shaders.emplace(pipeline, std::vector<char>(first, first + bytes));
    g_shaderBytes += bytes;
}

void FrameCapture_Dx12::NoteListReset(ID3D12GraphicsCommandList* cmd)
{
    if (cmd == nullptr)
        return;
    std::lock_guard<std::mutex> lock(g_listLock);
    g_lists.erase(cmd);
}

bool FrameCapture_Dx12::WantsComputeHooks() { return Diagnostics(); }

void FrameCapture_Dx12::CaptureAt(ID3D12GraphicsCommandList* cmd, ID3D12Resource* eye, D3D12_RESOURCE_STATES eyeState)
{
    if (!Diagnostics() || cmd == nullptr || eye == nullptr)
        return;

    // Learned every frame, capture or not. It used to be set only inside a running capture, which left the shape
    // filters with nothing to compare against until the first F7 -- and "no eye image yet" means "let it through",
    // so every eye-ish target in the game registered during that window.
    g_lastEyeDesc = eye->GetDesc();
    BindingCapture_Dx12::SetEyeShape((UINT) g_lastEyeDesc.Width, g_lastEyeDesc.Height);

    if (!StartIfDue())
        return;
    if (g_eyeCapturedThisPresent)
        return;
    g_eyeCapturedThisPresent = true;

    ID3D12Device* device = nullptr;
    if (FAILED(cmd->GetDevice(IID_PPV_ARGS(&device))) || device == nullptr)
        return;
    device->Release();
    EnsureFence(device);

    {
        std::lock_guard<std::mutex> lock(g_captureLock);
        Copy(cmd, device, eye, eyeState, "eye");
    }

    if (g_frame == 0)
    {
        size_t known = 0;
        {
            std::lock_guard<std::mutex> shaders(g_shaderLock);
            known = g_shaders.size();
        }
        LOG_INFO("Frame capture: capturing the {}x{} eye image over {} eye frames, candidates at their binds; "
                 "{} compute shaders known",
                 g_lastEyeDesc.Width, g_lastEyeDesc.Height, kCaptureFrames, known);
    }
}

void FrameCapture_Dx12::OnPresent(ID3D12CommandQueue* queue)
{
    if (!g_capturing)
        return;

    // A Present the splice did not run on captures no eye image; the frame counter only advances on the ones it
    // did, so the frames are eye frames and not wall-clock ones. If the splice stops altogether -- a menu, a
    // loading screen -- the capture would sit there holding its readback buffers, so it gives up after a while.
    if (!g_eyeCapturedThisPresent)
    {
        if (++g_presentsWithoutEye > kMaxPresentsWithoutEye)
        {
            LOG_WARN("Frame capture: no eye image for {} Presents, giving up on this capture", kMaxPresentsWithoutEye);
            Stop();
        }
        return;
    }

    g_eyeCapturedThisPresent = false;
    g_presentsWithoutEye = 0;
    {
        std::lock_guard<std::mutex> lock(g_captureLock);
        g_copiedThisFrame.clear();
    }

    if (++g_frame < kCaptureFrames)
    {
        ConstantCapture_Dx12::SetFrame(g_frame);
        BindingCapture_Dx12::SetFrame(g_frame);
        return;
    }

    Flush(queue);

    if (--g_repeatsLeft > 0)
    {
        const int spacing = Config::Instance()->CaptureSpacingSeconds.value_or_default();
        g_armAt = GetTickCount64() + (uint64_t) (spacing > 0 ? spacing : 0) * 1000;
        LOG_INFO("Frame capture: {} capture(s) left, the next in {} s", g_repeatsLeft, spacing);
    }
}
