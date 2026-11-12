#include "pch.h"
#include "DuniaTaa_Dx12.h"
#include "ConstantCapture_Dx12.h"
#include "CrashTrace.h"
#include "FrameCapture_Dx12.h"

#include "Config.h"
#include "Logger.h"
#include "NVNGX_Parameter.h"
#include "resource.h"
#include "State.h"
#include "SysUtils.h"
#include "Util.h"

#include <dlssnr/DlssNr.h>
#include <hudfix/Hudfix_Dx12.h>
#include <resource_tracking/ResTrack_Dx12.h>
#include <set>
#include <hooks/D3D12_Hooks.h>

#include <include/d3dx/d3dx12.h>
#include <atomic>
#include <d3dcompiler.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <format>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// ============================ GUARDRAILS =====================================================
// Two things in this file are deliberately awkward. They are not style; each one is a crash that happened
// in testing, and each was written down as a rule first and then repeated anyway. They are shaped
// so the mistake is a compile error rather than something to remember.
//
//   1. MvStagingPair  -- the two motion staging textures. The pointers are PRIVATE. The only way to read
//      the pair is Read(), which locks. AfterDispatch runs on the game's recording threads and can swap or
//      release these; the splice runs elsewhere and reads them. An unlocked read crashed on entering a
//      menu, because a menu is a size change and a size change is what queues a release.
//
//   2. DeferredReleaseList -- the vector of resources waiting to be released. Also PRIVATE, also locked.
//      It was a bare std::vector added to from the game's threads and erased from on the present thread.
//      push_back racing erase corrupts the heap. Three menu crashes; the first two "fixes" were plausible
//      guesses at other causes.
//
// IF ONE OF THESE IS IN YOUR WAY, read this before working around it:
//   * Needing a resource freed *now* rather than deferred almost always means the wait is missing, not the
//     defer. The game frees its targets on resolution and borderless changes, and the swapchain can be
//     resized -- work recorded seconds ago can still reference ours. ReleaseAllAfterGpuIdle() is the one
//     immediate path, and its name is the precondition: only after the GPU is idle.
//   * Needing the raw pointers outside the lock almost always means a new call site that should be taking
//     a Snapshot instead. Add a locked accessor to the class; do not add a second copy of the pointer.
//   * If you genuinely must relax one, say so in the commit message, because the
//     next person will otherwise read the comment above and assume the protection still holds.
//
// Why they exist as code rather than documentation: both were documented already and were repeated anyway.
// =============================================================================================

namespace
{
std::mutex g_lock;
ID3D12Resource* g_color = nullptr;
ID3D12Resource* g_depth = nullptr;

ID3D12CommandAllocator* g_allocator = nullptr;
ID3D12GraphicsCommandList* g_cmd = nullptr;
ID3D12Fence* g_fence = nullptr;
HANDLE g_fenceEvent = nullptr;
UINT64 g_fenceValue = 0;
ID3D12Device* g_device = nullptr;

ID3D12Resource* g_output = nullptr;
ID3D12Resource* g_motion = nullptr;
ID3D12Resource* g_gameMv = nullptr;
bool g_gameMvIsRt = true;

// ---- finding the motion vector pass ------------------------------------------------------------
// The game writes its motion vectors from a compute shader through a UAV, so they never reach
// OMSetRenderTargets and the render-target path that finds every other candidate cannot see them at
// all -- six captures missed them for exactly that reason. What identifies the pass is what it
// writes: it binds its three UAVs together in one table, the R8G8_UNORM motion pair, an R8_UNORM
// geometric edge and an R8_UINT stencil, all at half the eye size. Those are its u0, u1 and u2,
// and a capture's bindings.txt has that trio appearing nowhere else in a frame.
//
// Nothing is touched here: the table is read, the pointers are noted, and the search stops. A table
// bind carries no guarantee about a resource's state, and barriering on the assumption that it does
// hung the game once. No reference is taken either -- the game frees these on a resolution change,
// and an outstanding reference is what broke resolution changes before.
std::atomic<bool> g_mvSearchOn { false };
std::mutex g_mvSearchLock;
// Identity only: compared, logged, never dereferenced and never AddRef'd.
std::vector<ID3D12Resource*> g_mvFound;
uint64_t g_mvSearchUntil = 0;
UINT g_mvSearchPinnedW = 0;
bool g_mvSearchReported = false;
// One instance per eye: the game keeps its temporal history per eye under alternate-eye rendering,
// and the capture shows the two alternating with the Present.
constexpr size_t kMvInstances = 2;
// If the pass has not turned up in this many Presents since the eye shape last settled it is not going
// to, and the search is not worth paying for any longer. Generous on purpose: the cost is a few seconds
// of mirroring, and a window too short to reach gameplay would report a false negative.
constexpr uint64_t kMvSearchPresents = 3000;
// Which of the two instances is current when the splice runs has to be decided from something cheap --
// the Present's parity, the command list recording the pass, or the descriptor table's own handle -- and
// deciding it wrong would feed DLSS the other eye's motion, which is the exact artefact this port exists
// to remove. So the search reports all three for its first few matches and the choice is made from data.
// Diagnostic, and the only reason the search runs on past finding both instances.
constexpr int kMvProbes = 16;
int g_mvProbes = 0;

// ---- real motion --------------------------------------------------------------------------------
// The game's own vectors, fed to DLSS. Two steps, and the split between them is the whole point:
//   * right after the pass's Dispatch, on the game's own list, its texture is copied into g_mvStage.
//     A copy and barriers change no pipeline state, so injecting them there is safe -- which is why
//     the capture copy ran without incident. A compute pass of ours at that point would clobber
//     the game's descriptor heaps, root signature and PSO mid-frame.
//   * at the splice, the decode reads g_mvStage, whose state is ours, and writes g_motion.
// Copying also settles which of the two instances is current for free: it is whichever one the
// dispatch just wrote. No parity, no assumption.
// 0 off, 1 the game's vectors as they are (other-eye, so it halos), 2 chained over two Presents to
// reach the same eye. F4 cycles all three, because comparing across runs is meaningless here.
std::atomic<int> g_realMotionMode { 0 };
std::atomic<bool> g_realMotionOn { false };
// ---- the cheap route to the pass ---------------------------------------------------------------
// Resolving descriptor tables needs ResTrack's heap mirroring, and that turned out to cost about 45% of
// the frame rate: 90/s down to 49-53/s, measured with F4 inside one run. Narrowing the mirroring by heap
// achieved nothing, because the heaps the pass binds from are the game's two main per-eye shader-visible
// heaps -- every descriptor in the frame is copied into them.
//
// The pass does not need to be recognised by what it binds, though. It has its own compute pipeline --
// the one that writes the encoded vectors -- and hkSetPipelineState already
// tracks which pipeline each list last set. So: learn the pipeline once by the descriptor route, then
// switch the mirroring off for good and recognise the pass by one pointer compare per dispatch.
//
// Which of the two textures it wrote then comes from Present parity, which the probe run measured at
// 16/16 with exactly one bind per Present. If parity ever slips the motion is the other eye's for a
// frame, which degrades the picture and breaks nothing.
ID3D12PipelineState* g_mvPipeline = nullptr;   // identity only, never dereferenced, never referenced
// The game's motion texture per Present parity, learned once through the descriptor table and then used --
// barriered and copied -- at every dispatch of the learned pipeline. RAW POINTERS TO GAME RESOURCES, never
// referenced. A menu frees and reallocates those textures while the pipeline stays the same, so without a
// death notice AfterDispatch kept barriering a dead resource, and the driver dereferenced it: a crash on
// entering an FC6 menu, inside AfterDispatch under hkDispatch. The notice is
// DuniaTaa_Dx12::ForgetResource, from ResTrack's Release hook, exactly as ConstantCapture's buffers get one.
// Atomic because the Release hook runs on whichever thread frees the texture.
std::atomic<ID3D12Resource*> g_mvByParity[2] { nullptr, nullptr };
std::atomic<bool> g_mvPipelineReady { false };
bool g_mvPipelineLogged = false;
std::atomic<uint64_t> g_mvRelearns { 0 };

// ---- the game's own TAA resolve -----------------------------------------------------------------
// The step FC5 and Days Gone both took and FC6 never has: replace the game's temporal resolve and skip
// it, so DLSS *is* the TAA. Without it our DLAA is a second pass over an image the game has already
// resolved with its own full-precision motion, which is why real motion vectors changed so little here.
//
// Found offline in a capture's bindings.txt, and its binding set is the signature -- it maps onto FC5's
// TAA pixel shader slot for slot:
//   two half-eye R8G8_UNORM  = MotionVectorsTexture + MotionVectorsHistoryTexture (this eye, other eye)
//   three eye-sized RGBA16F  = SceneColor + SceneColorPreviousFrame + SceneColorAccumulation
//   two eye-sized R8_UNORM
// It is a graphics *draw*, so unlike the compute motion pass it is reachable from OMSetRenderTargets.
//
// Recognised in two stages, because a table bind does not say which pipeline the draw will use: the bind
// marks the list as a candidate, and the next draw on that list is the one that learns the pipeline.
// After that it is one pointer compare per draw -- the same trade that made the motion copy free, and
// for the same reason: resolving descriptor tables needs heap mirroring, which costs ~45% of the frame
// rate.
ID3D12PipelineState* g_taaPipeline = nullptr;   // identity only, never dereferenced, never referenced
std::atomic<bool> g_taaFound { false };
// F8, three states so the whole comparison lives inside one run: 0 the game's own resolve, 1 its draw
// skipped with nothing replacing it (the step-2 gate -- the picture must visibly change), 2 DLSS spliced
// in its place (step 3). Comparing frame rate or image quality across runs is meaningless here.
std::atomic<int> g_taaMode { 0 };
// Per-thread caches for the two per-draw checks. A command list is recorded by one thread at a time, so the
// pipeline this thread last set on this list, and the upscale target this thread last bound on it, are the
// list's -- and reading them costs no lock. Every draw in the frame (2,000 of them) used to take a global
// mutex and two map operations to learn that it was not the resolve. The maps stay as the fallback.
thread_local ID3D12GraphicsCommandList* t_pipeCmd = nullptr;
thread_local ID3D12PipelineState* t_pipeline = nullptr;
thread_local ID3D12GraphicsCommandList* t_upscaleCmd = nullptr;
std::atomic<uint64_t> g_taaSkipped { 0 };
std::atomic<uint64_t> g_taaSpliced { 0 };
std::atomic<bool> g_taaStatsRestart { false };
uint64_t g_taaStatsAt = 0;
constexpr uint64_t kTaaStatsPresents = 600;
// Set once the target->colour pairs have stopped changing, which is what lets the heap mirroring stop.
std::atomic<bool> g_taaPairsDone { false };
bool g_taaLogged = false;
UINT g_taaRootParameter = 0;
// The closest any graphics table came to the signature. A search that finds nothing has to say what it
// nearly matched, or widening it is guesswork -- the same lesson as every silent skip.
std::atomic<int> g_taaBestMotion { 0 };
std::atomic<int> g_taaBestHdr { 0 };
// Structural sizes, from facts no probe pass or menu has, logged beside what the size learners believe
// (LogSizeViews). A 4096^2 probe pass was once learned as the eye; these do not look at sizes to decide.
// The eye: the finished RGBA8 image copied into an sRGB texture every frame in VR (BeforeCopy).
// The scene: a graphics table with the TAA resolve's shape -- two R8G8_UNORM motion textures at half the size of two
// or more same-size RGBA16F -- at ANY size (NoteGraphicsTable). Written on the recording threads, read at Present.
std::atomic<unsigned> g_vrEyeW { 0 };
std::atomic<unsigned> g_vrEyeH { 0 };
std::atomic<unsigned> g_structSceneW { 0 };
std::atomic<unsigned> g_structSceneH { 0 };
std::atomic<uint64_t> g_structEyeSceneFirst { 0 }; // Present a resolve of the eye's shape was first seen (gameplay)
std::atomic<unsigned> g_structEyeSceneW { 0 };      // the last resolve-shaped table of the eye's aspect (not 16:9)
std::atomic<unsigned> g_structEyeSceneH { 0 };
std::atomic<uint64_t> g_structRefreshUntil { 0 };   // square-eye rules: heap mirroring held on until this Present
std::mutex g_structLock; // the logged-size sets below
std::unordered_set<uint64_t> g_structSceneLogged, g_vrEyeLogged;
// What the resolve reads, in table slot order, and what it writes. Step 3 has to put DLSS's output where
// the game's resolve put its own, and feed DLSS the same colour the resolve consumes -- neither can be
// guessed. FC5's pixel shader names the slots and FC6's table matches it: t0/t1 motion (this eye, other
// eye), t2 SceneColor, t3 SceneColorPreviousFrame, t4 SceneColorAccumulation, t5/t6 masks. Identity only.
constexpr int kTaaColourSlots = 3;
ID3D12Resource* g_taaColour[kTaaColourSlots] = { nullptr, nullptr, nullptr };
ID3D12Resource* g_taaTarget = nullptr;
ID3D12Resource* g_taaTargetDepth = nullptr;
// The render target each list last had bound, so the draw can say what the resolve writes to. Kept here
// rather than read out of CmdTargets, which several other paths maintain for their own purposes.
std::mutex g_taaTargetLock;
std::unordered_map<ID3D12GraphicsCommandList*, std::pair<ID3D12Resource*, ID3D12Resource*>> g_taaListTarget;
uint64_t g_taaSearchUntil = 0;
bool g_taaGaveUp = false;
// Lists whose last graphics table bind matched the signature, and the pipeline each list last set. Both
// touched from the game's recording threads, so both are locked.
std::mutex g_taaLock;
std::unordered_set<ID3D12GraphicsCommandList*> g_taaCandidate;
std::unordered_map<ID3D12GraphicsCommandList*, ID3D12PipelineState*> g_taaListPipeline;

// GUARDRAIL-shaped, for the same reason as MvStagingPair and DeferredReleaseList: every field here is
// written from the game's recording threads (the table bind, the render-target bind, the resolve draw)
// and read from the present thread (feature creation, the stats). The state is private and every accessor
// locks, so there is no call site that can read it unsynchronised -- a compile error, not a rule.
//
// What it is for. The splice has to feed DLSS the colour the resolve reads (t2) and write DLSS's output
// where the resolve writes (the render target bound at its draw). Neither may be latched: the game
// ping-pongs both per eye and frees them on a resolution change, and they are adjacent allocations, which
// is exactly what a per-eye pair looks like. The target is free at every draw -- OMSetRenderTargets hands
// it over. t2 costs a descriptor table walk, which needs ResTrack's heap mirroring, which costs ~45% of
// the frame rate. So the (target -> colour) pairs are LEARNED while the mirroring is on, the mirroring
// stops once they have settled, and from then on the splice is one hash lookup per resolve draw.
//
// Pointers are identity only: compared, never dereferenced here, never AddRef'd. The game frees these on
// a resolution change and an outstanding reference is what broke the custom resolutions before. The
// colour's shape is copied in at learn time rather than read back off the pointer later, for the same
// reason: at learn time the game has it bound, so it is alive; a Present later it may not be. A lookup
// that misses reopens learning instead of guessing -- that is what a reload looks like.
constexpr size_t kTaaMaxPairs = 8;
// How long the pair set must stop growing before the mirroring is switched off. ~1.5 s of gameplay at VR
// rates: long enough for both eyes and both halves of a ping-pong to have drawn, short enough that the
// 45% is paid for a moment rather than for the session.
constexpr uint64_t kTaaPairsStable = 120;

class TaaResolveMap
{
  public:
    // The colour the resolve reads, with its shape taken while the game still had it bound.
    struct Colour
    {
        ID3D12Resource* colour = nullptr;   // identity only
        UINT width = 0;
        UINT height = 0;
        DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;
    };

    // From the graphics table bind: what this list's last matching table had at t2.
    void NoteColour(ID3D12GraphicsCommandList* cmd, ID3D12Resource* colour, UINT width, UINT height,
                    DXGI_FORMAT format)
    {
        if (cmd == nullptr || colour == nullptr)
            return;
        std::lock_guard<std::mutex> lock(m_lock);
        m_pending[cmd] = { colour, width, height, format };
    }

    // At the resolve draw. Consumed, so a table bound for some other draw can never be reused later.
    bool TakeColour(ID3D12GraphicsCommandList* cmd, Colour* out)
    {
        if (cmd == nullptr || out == nullptr)
            return false;
        std::lock_guard<std::mutex> lock(m_lock);
        const auto it = m_pending.find(cmd);
        if (it == m_pending.end())
            return false;
        *out = it->second;
        m_pending.erase(it);
        return true;
    }

    // The eye depth this list last bound at the resolve's own size. Per list for the same reason the
    // tonemapped path keeps its own: the next eye's lists are recorded before this eye is presented, so
    // "the last eye depth" can belong to the other eye.
    void NoteEyeDepth(ID3D12GraphicsCommandList* cmd, ID3D12Resource* depth)
    {
        if (cmd == nullptr || depth == nullptr)
            return;
        std::lock_guard<std::mutex> lock(m_lock);
        m_depth[cmd] = depth;
    }

    ID3D12Resource* EyeDepth(ID3D12GraphicsCommandList* cmd) const
    {
        if (cmd == nullptr)
            return nullptr;
        std::lock_guard<std::mutex> lock(m_lock);
        const auto it = m_depth.find(cmd);
        return it == m_depth.end() ? nullptr : it->second;
    }

    // True when this pair was not already known, so the caller can log it once.
    bool Learn(ID3D12Resource* target, const Colour& colour, uint64_t presentIndex)
    {
        if (target == nullptr || colour.colour == nullptr || target == colour.colour)
            return false;
        std::lock_guard<std::mutex> lock(m_lock);
        const auto it = m_pairs.find(target);
        if (it != m_pairs.end() && it->second.colour == colour.colour)
            return false;
        if (it == m_pairs.end() && m_pairs.size() >= kTaaMaxPairs)
            return false;
        m_pairs[target] = colour;
        // Which eye this target is, by the order the targets are first seen -- the same identity trick
        // the upscale path uses with g_eyeSrc, and under AER the game keeps one resolve target per eye.
        // Both runs so far settled on exactly two.
        if (m_eye.find(target) == m_eye.end())
        {
            if (m_eyes < 2)
                m_eye[target] = m_eyes++;
            else
                m_eyeOverflow = true;
        }
        m_shape = colour;
        m_lastNew = presentIndex;
        return true;
    }

    // 0 or 1, or -1 when this target never got an eye. A third target would mean one-target-per-eye is
    // wrong for this game, so nothing is guessed: the caller logs it once and falls back to the single
    // shared history, which is exactly the behaviour before per-eye features existed.
    int EyeSlot(ID3D12Resource* target, bool* overflow) const
    {
        std::lock_guard<std::mutex> lock(m_lock);
        if (overflow != nullptr)
            *overflow = m_eyeOverflow;
        const auto it = m_eye.find(target);
        return it == m_eye.end() ? -1 : it->second;
    }

    ID3D12Resource* ColourFor(ID3D12Resource* target) const
    {
        if (target == nullptr)
            return nullptr;
        std::lock_guard<std::mutex> lock(m_lock);
        const auto it = m_pairs.find(target);
        return it == m_pairs.end() ? nullptr : it->second.colour;
    }

    // What the feature has to be created for: copied in at learn time, never read back off a pointer.
    bool ColourShape(UINT* width, UINT* height, DXGI_FORMAT* format) const
    {
        std::lock_guard<std::mutex> lock(m_lock);
        if (m_shape.width == 0)
            return false;
        *width = m_shape.width;
        *height = m_shape.height;
        *format = m_shape.format;
        return true;
    }

    bool Settled(uint64_t presentIndex) const
    {
        std::lock_guard<std::mutex> lock(m_lock);
        return !m_pairs.empty() && m_lastNew != 0 && presentIndex > m_lastNew + kTaaPairsStable;
    }

    size_t Size() const
    {
        std::lock_guard<std::mutex> lock(m_lock);
        return m_pairs.size();
    }

    // A reload reallocates the game's targets, so every pair goes stale at once. Dropping them and
    // learning again is the honest answer; carrying on would splice onto a pointer the game has freed.
    void Reopen(uint64_t presentIndex)
    {
        std::lock_guard<std::mutex> lock(m_lock);
        m_pairs.clear();
        m_pending.clear();
        // The eye assignment goes with them: the new targets are learned in whatever order they draw, so
        // the eyes can come back the other way round. The caller resets both DLSS histories for that.
        m_eye.clear();
        m_eyes = 0;
        m_eyeOverflow = false;
        m_lastNew = presentIndex;
    }

  private:
    mutable std::mutex m_lock;
    std::unordered_map<ID3D12Resource*, Colour> m_pairs;
    std::unordered_map<ID3D12GraphicsCommandList*, Colour> m_pending;
    std::unordered_map<ID3D12GraphicsCommandList*, ID3D12Resource*> m_depth;   // identity only
    std::unordered_map<ID3D12Resource*, int> m_eye;                            // identity only
    int m_eyes = 0;
    bool m_eyeOverflow = false;
    Colour m_shape;
    uint64_t m_lastNew = 0;
};
TaaResolveMap g_taaResolve;

// ---- the game's per-view constants, read from a CPU mapping ---------------------------------------
// What this replaces: the current-to-previous matrix was thought reachable only by resolving a CBV
// descriptor every frame, which needs ResTrack's heap mirroring, which costs ~45% of the frame rate. That was
// true of the *descriptor*. It is not true of the bytes: Dunia writes these constants from the CPU into a mapped
// upload buffer every frame, and hooking ID3D12Resource::Map gives us the mapping. No descriptor, no mirroring,
// no GPU work at all -- reading them is a pointer dereference.
//
// Layout, confirmed across four captures and two independent runs (offsets in bytes from the struct start):
//    +64  camera-to-world   (column-vector; its 4th column IS the camera position, which is what identifies it)
//   +128  projection        (P[0][0], P[1][1]; JITTER at P[0][2] and P[1][2]; 1/far at P[2][2], near at P[2][3])
//   +320  view-rotation-projection -- world->clip with the translation stripped (4th column is [0,0,near,0])
//   +384  the same, one Present back
//   +448  [near, far, far-near, 1/far]
//   +512  [w, h, 1/w, 1/h]  -- the fingerprint this is found by
//   +544  camera position          +576  the same, one Present back
//
// PROVEN from the captures, not assumed: +384/+576 are ONE Present back, which under alternate-eye rendering is
// the OTHER eye -- frame N's +576 equals frame N-1's +544 to 0.0000000. So the same-eye previous state is TWO
// Presents back and has to be kept here, by Present parity, exactly as FC5 keeps g_matrices[2].
//
// GUARDRAIL shape, for the usual reason: written from the splice (a game recording thread) and read from
// OnPresent. State private, accessors locked.
constexpr size_t kVcCameraToWorld = 64;
constexpr size_t kVcProjection = 128;
constexpr size_t kVcVrp = 320;
constexpr size_t kVcViewport = 512;
constexpr size_t kVcCamPos = 544;
constexpr size_t kVcMaxCandidates = 32;
// If nothing has been written for this long the set is stale: the scan ran before the live instances existed.
constexpr uint64_t kVcRescanPresents = 240;
// An empty scan is retried on a doubling timer up to this, because the sweep reads every mapped buffer the game
// has and a run where it keeps finding nothing was hitching every 3 s on the retry alone.
constexpr uint64_t kVcRescanMaxPresents = 240 * 16;
// How much of a Present the sweep may take. A full sweep measured ~1 s (write-combined memory, uncached reads),
// which was the hitch on leaving a menu; sliced at this it is spread over Presents and never felt -- once a
// set is held. While NOTHING is held, matrix motion is off anyway, so the sweep gets four times as much:
// a few seconds at ~60/s beats a run at 90/s with no motion.
constexpr double kVcScanMsPerPresent = 1.5;
constexpr double kVcScanMsPerPresentUrgent = 6.0;

// Everything a real per-view struct must be true of, checked structurally so a buffer that merely happens to
// carry the eye size is thrown out: a projection with a zeroed off-diagonal and w row, and a rotation-only
// view-projection whose 4th column is (0, 0, near, 0). Applied INSIDE the fingerprint sweep -- the raw quad
// matches 512+ buffers and validating a capped raw list afterwards made finding the real ones a matter of
// address order: one run found 6, the next 0.
// Why a raw fingerprint match was thrown out, so "0 real per-view structs" says whether the projection test
// or the view-rotation-projection test did it -- and a first rejected block is logged in the clear.
std::atomic<uint64_t> g_vcRejectProj { 0 }, g_vcRejectVrp { 0 }, g_vcAccepted { 0 };
std::atomic<int> g_vcRejectPeeks { 0 };
bool IsPerViewStruct(const uint8_t* quad)
{
    const uint8_t* base = quad - kVcViewport;
    float pr[16], vrp[16];
    std::memcpy(pr, base + kVcProjection, sizeof(pr));
    std::memcpy(vrp, base + kVcVrp, sizeof(vrp));
    const float nearZ = pr[11];
    const bool projOk = pr[0] > 0.01f && pr[5] > 0.01f && pr[1] == 0.0f && pr[4] == 0.0f && pr[12] == 0.0f &&
                        pr[13] == 0.0f && pr[15] == 0.0f && std::abs(pr[14] + 1.0f) < 1e-3f && nearZ > 0.0f &&
                        nearZ < 10.0f;
    const bool vrpOk = vrp[3] == 0.0f && vrp[7] == 0.0f && vrp[15] == 0.0f && std::abs(vrp[11] - nearZ) < 1e-3f;
    if (projOk && vrpOk)
    {
        g_vcAccepted.fetch_add(1, std::memory_order_relaxed);
        return true;
    }
    (projOk ? g_vcRejectVrp : g_vcRejectProj).fetch_add(1, std::memory_order_relaxed);
    if (g_vcRejectPeeks.load(std::memory_order_relaxed) < 3)
    {
        const int n = g_vcRejectPeeks.fetch_add(1, std::memory_order_relaxed);
        if (n < 3)
            LOG_INFO("DuniaTaa: view constants: raw match {} rejected by the {} test -- proj [{:.4g} {:.4g} {:.4g} "
                     "{:.4g} | {:.4g} {:.4g} {:.4g} {:.4g} | {:.4g} {:.4g} {:.4g} {:.4g} | {:.4g} {:.4g} {:.4g} "
                     "{:.4g}], vrp col3 [{:.4g} {:.4g} {:.4g} {:.4g}]",
                     n + 1, projOk ? "VRP" : "projection", pr[0], pr[1], pr[2], pr[3], pr[4], pr[5], pr[6], pr[7],
                     pr[8], pr[9], pr[10], pr[11], pr[12], pr[13], pr[14], pr[15], vrp[3], vrp[7], vrp[11],
                     vrp[15]);
    }
    return false;
}

// ---- The motion pass's own constants: its constant buffer, bound as 512 bytes, 432 used --------------------
// Register 7 is the SAME viewport quad the per-view struct carries, so the sweep already walks over these
// blocks; ObserveViewportQuad picks them out of its raw matches. Layout and the matrix's derivation were
// settled offline against two captures (tools/capture/fc6sameeye.py):
//   regs 2-5 (+32)  M = PrevVRP * Translate(cam - prevCam) * inverse(P_unjittered * ViewRot), rows x, y, (z), w
//   reg 7   (+112)  [eyeW, eyeH, 1/eyeW, 1/eyeH]
//   reg 8   (+128)  [eyeW/2, eyeH/2, 2/eyeW, 2/eyeH]
constexpr size_t kTbMatrix = 32;
constexpr size_t kTbViewport = 112;
constexpr size_t kTbBytes = 432;
constexpr size_t kTbMaxCandidates = 16;
// More per-view offsets, used only to rebuild that matrix: the view matrix (its 3x3 is the rotation) and the
// previous Present's unjittered view-rot-projection and camera position.
// The jitter as the image's shift in pixels, x right and y down: (+0.25, -0.25) / (-0.25, +0.25) alternating
// in every capture, and always equal to (-proj[0][2] * W/2, +proj[1][2] * H/2). As FC5's jitter offsets.
constexpr size_t kVcJitterPx = 704;
constexpr size_t kVcViewProjection = 256;
constexpr size_t kVcView = 192;
constexpr size_t kVcPrevVrp = 384;
constexpr size_t kVcPrevCamPos = 576;
// Found on whichever thread the sweep or the Unmap watch runs on, so they wait here under a lock of their own
// and ViewConstants takes them in under its own. Raw pointers into the game's memory: dropped on the
// buffer's death by ForgetIfWithin, exactly as the per-view candidates are, and never read to see if they
// are still good.
std::mutex g_tbPendingLock;
const uint8_t* g_tbPending[kTbMaxCandidates] {};
size_t g_tbPendingCount = 0;
std::atomic<uint64_t> g_tbObserved { 0 };

// The build uses fast floating point, where `x == x` and "a comparison with a NaN is false" are NOT things
// the compiler has to honour -- it may assume there are no NaNs at all. A pool slot full of NaNs "verified"
// to a residual of exactly 0 that way and was WRITTEN to (the log read "parallax term was -nan"). So
// anything read out of the game's memory is checked by its BITS before it is compared as a number.
bool FiniteBits(const float* v, size_t count)
{
    for (size_t i = 0; i < count; ++i)
    {
        uint32_t bits;
        std::memcpy(&bits, v + i, sizeof(bits));
        if ((bits & 0x7F800000u) == 0x7F800000u)
            return false;
    }
    return true;
}

// ---- Jitter ([Inputs] DuniaTaaJitter, live in the overlay) ------------------------------------------------
// 0: DLSS is told the jitter is zero -- how every validated FC6 state so far has run, and wrong since the
//    splice moved in front of the game's TAA: the colour DLSS gets IS jittered, by +-0.25 px.
// 1: DLSS is told the game's jitter. Still 2-phase, so under alternate-eye rendering each eye has the same
//    sub-pixel offset on every one of its frames and DLSS gathers nothing between the pixels.
// 2: our own jitter, per eye, Halton(2,3), patched into the per-view struct at the moment PatchPerView already
//    writes it, and reported to DLSS. FC5's fix, validated there.
// 3, 4: FC5's two SHAKE TESTS, because "is our jitter in the picture" and "does DLSS take it out again" cannot
//    be told apart by looking at mode 2 (first run: mode 2 shimmered over the whole image, which is what either
//    failure looks like). 3: our jitter x20 patched in, DLSS told ZERO -- if the patched matrices reach the
//    geometry the whole image shakes by several pixels; if it stands still, the geometry takes its projection
//    from somewhere else and mode 2 can never work as built. 4: our jitter x4 patched in AND told to DLSS --
//    when eye, timing and sign all match, DLSS cancels it and the image stands still; if it shakes, walk the
//    sign combo, and if no sign stills it the reported jitter belongs to another frame or another copy.
std::atomic<int> g_jitterMode { -1 }; // -1: not read from the ini yet
std::atomic<int> g_baselineOld { 0 }; // 1: the older other-eye baseline, for the halo A/B
std::atomic<int> g_jitterSign { 0 };  // diagnostics: which of FC5's four sign conventions DLSS is given
std::atomic<int> g_jitterPhases { 8 };
constexpr float kJitterSigns[4][2] = { { 1, 1 }, { -1, -1 }, { 1, -1 }, { -1, 1 } };
std::atomic<uint64_t> g_jitterPatched { 0 }, g_jitterUnverified { 0 }, g_jitterReported { 0 };
std::atomic<uint64_t> g_jitterDisagreed { 0 }; // +704 did not match the projection but was patched anyway
// Why a copy was refused, one counter each: the single 'refused' number hid which gate it was for a whole run
// (50% Resolution Scale: hundreds refused per window, edges shimmering, and no way to tell a
// jitter too large from a layout never proven). Plus the largest |jitter| a refused copy carried.
std::atomic<uint64_t> g_jitterRefNotFinite { 0 };
std::atomic<uint64_t> g_jitterRefViewport { 0 };
std::atomic<uint64_t> g_jitterRefTooLarge { 0 };
std::atomic<uint64_t> g_jitterRefUnproven { 0 };
std::atomic<int> g_jitterRefLogged { 0 };
std::atomic<uint64_t> g_vcDeadDrops { 0 }; // candidate sets dropped because a member no longer held a struct
std::atomic<float> g_jitterRefMaxPx { 0.0f };
// Is the jitter DLSS is told the jitter that was last written? The x of the last two written (as bits, so a
// pair of floats needs no lock), against what Update read for the frame being evaluated. A mismatch means the
// struct Update chose is not the one the frame was rendered with -- a copy, or another frame's.
std::atomic<uint32_t> g_jitterLastX[2] {};
std::atomic<uint64_t> g_jitterMismatch { 0 }, g_jitterNewFrames { 0 };

int JitterModeNow()
{
    int mode = g_jitterMode.load(std::memory_order_relaxed);
    if (mode < 0)
    {
        mode = std::clamp(Config::Instance()->DuniaTaaJitter.value_or_default(), 0, 4);
        g_jitterMode.store(mode, std::memory_order_relaxed);
    }
    return mode;
}

float Halton01(int index, int base)
{
    float f = 1.0f, r = 0.0f;
    while (index > 0)
    {
        f /= (float) base;
        r += f * (float) (index % base);
        index /= base;
    }
    return r;
}

// Clip-space output, rows x, y, z, w at m[0], m[4], m[8], m[12]: the image moves by (tx, ty) in NDC.
void ShiftClip(float* m, float tx, float ty)
{
    for (int c = 0; c < 4; ++c)
    {
        m[c] += tx * m[12 + c];
        m[4 + c] += ty * m[12 + c];
    }
}

// The inverse of such a matrix: its input's w column takes the opposite shift.
void ShiftInverse(float* m, float tx, float ty)
{
    for (int r = 0; r < 4; ++r)
        m[r * 4 + 3] -= tx * m[r * 4] + ty * m[r * 4 + 1];
}

void ObserveViewportQuad(const uint8_t* quad, uint64_t bytesBefore, uint64_t bytesFromQuad)
{
    if (bytesBefore < kTbViewport || bytesFromQuad < kTbBytes - kTbViewport)
        return;
    float q[4], half[4], m[16];
    std::memcpy(q, quad, sizeof(q));
    std::memcpy(half, quad + 16, sizeof(half));
    if (half[0] != q[0] * 0.5f || half[1] != q[1] * 0.5f)
        return;
    if (!(std::abs(half[2] - 2.0f * q[2]) < 1e-7f && std::abs(half[3] - 2.0f * q[3]) < 1e-7f))
        return;
    // A reprojection between two nearby views is near the identity on its diagonal.
    std::memcpy(m, quad - kTbViewport + kTbMatrix, sizeof(m));
    if (!FiniteBits(m, 16) || !FiniteBits(half, 4))
        return;
    if (!(std::abs(m[0] - 1.0f) < 0.25f && std::abs(m[5] - 1.0f) < 0.25f && std::abs(m[10] - 1.0f) < 0.01f &&
          std::abs(m[15] - 1.0f) < 0.25f))
        return;

    g_tbObserved.fetch_add(1, std::memory_order_relaxed);
    const uint8_t* base = quad - kTbViewport;
    std::lock_guard<std::mutex> lock(g_tbPendingLock);
    for (size_t i = 0; i < g_tbPendingCount; ++i)
        if (g_tbPending[i] == base)
            return;
    if (g_tbPendingCount < kTbMaxCandidates)
        g_tbPending[g_tbPendingCount++] = base;
}

// 4x4, row-major, double. The inverse is general Gauss-Jordan with partial pivoting: it runs once per frame
// on one matrix, and a closed form for "projection times rotation" is one more thing to get wrong.
void Mul4(const double a[16], const double b[16], double out[16])
{
    double r[16];
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
        {
            double v = 0.0;
            for (int k = 0; k < 4; ++k)
                v += a[i * 4 + k] * b[k * 4 + j];
            r[i * 4 + j] = v;
        }
    std::memcpy(out, r, sizeof(r));
}

bool Invert4(const double in[16], double out[16])
{
    double a[4][8];
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
        {
            a[i][j] = in[i * 4 + j];
            a[i][j + 4] = i == j ? 1.0 : 0.0;
        }
    for (int c = 0; c < 4; ++c)
    {
        int pivot = c;
        for (int r = c + 1; r < 4; ++r)
            if (std::abs(a[r][c]) > std::abs(a[pivot][c]))
                pivot = r;
        if (!(std::abs(a[pivot][c]) > 1e-300))
            return false;
        if (pivot != c)
            for (int j = 0; j < 8; ++j)
                std::swap(a[pivot][j], a[c][j]);
        const double inv = 1.0 / a[c][c];
        for (int j = 0; j < 8; ++j)
            a[c][j] *= inv;
        for (int r = 0; r < 4; ++r)
        {
            if (r == c)
                continue;
            const double f = a[r][c];
            if (f != 0.0)
                for (int j = 0; j < 8; ++j)
                    a[r][j] -= f * a[c][j];
        }
    }
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            out[i * 4 + j] = a[i][j + 4];
    return true;
}

// What one call of ViewConstants::PatchTemporal did. Every way of doing nothing has a name, because a patch
// that silently does not happen looks exactly like other-eye vectors, which look plausible.
enum class TbResult
{
    Busy,         // the present thread held the lock (a sweep slice); this frame keeps the game's matrix
    NoBlock,      // the sweep has not found the motion pass's constant block yet
    NoView,       // no per-view struct candidates
    AlreadyDone,  // the block still holds what we wrote: a second dispatch in the same frame
    Unverified,   // no block's matrix equals our other-eye build -- not the live block, or the derivation is off
    NoChain,      // verified, but the previous Present's state is not the one this frame follows on from
    VerifiedOnly, // verified and chained, patching is switched off (F4)
    Patched,
    Count
};
// What one call of ViewConstants::PatchPerView did -- the same idea for the per-view struct's +384/+576. It is
// called at EVERY ExecuteCommandLists, so "Idle" (nothing was rewritten since the last look) is the usual one.
enum class PvResult
{
    Idle,
    Busy,         // the present thread held the lock; looked at again at the next submit
    NoView,       // no per-view struct candidates yet
    NoChain,      // a struct was rewritten, but its +384/+576 are not the previous rewritten struct's state
    VerifiedOnly, // chained -- would have been written; the patch is switched off (F4)
    Patched,
    Count
};
// Test control (overlay): which of a frame's copies of the per-view struct get the same-eye previous view.
// 0 all (the validated behaviour), 1 the first written, 2 the second, 3 the third and later, 4 none.
std::atomic<int> g_pvCopyMask { 0 };
// Days Gone's composition knobs for the other-eye object branch; the validated FC6 values are the defaults.
std::atomic<float> g_objectScale { 1.0f };
std::atomic<float> g_objectFloor { 1.0f };
std::atomic<float> g_objectSpeedTerm { 0.0f };
std::atomic<bool> g_objectFastChain { false };
// The residual view (overlay checkbox): paints where the object branch adds motion, in place of the picture.
std::atomic<bool> g_residualView { false };
// Object residual only on the nearest surface (live A/B in the tuning node; on by default -- see MvConstants).
std::atomic<bool> g_objectNearOnly { true };
// No residual where the vector is the background's, carried onto a near pixel (live A/B; see MvConstants).
std::atomic<bool> g_objectBgReject { true };
// Their neighbourhood in px (live slider, 1..6). 3 is validated (at 70%): the silhouette outlines
// against the sky are gone at 3 and back at 2, a walking dog keeps its full residual, and the green (gated) grass
// interiors are identical at 2 and 3 -- that is the 8-bit tolerance at near range, not this test.
std::atomic<int> g_objectRadius { 3 };
// The private-copy feasibility probe (see PcWalk): 0 idle, 1 warming up, 2 counting; armed from the overlay.
std::atomic<int> g_pcState { 0 };
std::atomic<bool> g_pcArm { false };
void PcWalk(bool graphics, UINT root, D3D12_GPU_DESCRIPTOR_HANDLE base, ID3D12GraphicsCommandList* cmd);
void VelNoteSubmits(ID3D12CommandList* const* lists, UINT count);
std::atomic<uint64_t> g_pvCopySeen[4] {};
std::atomic<uint64_t> g_pvUnpatched { 0 }; // structs whose +384 was a view we wrote earlier (copies after a menu)
std::atomic<uint64_t> g_pvOtherView { 0 }; // candidate slots holding another view (viewport not the scene's): skipped
std::atomic<uint64_t> g_vcAmbiguous { 0 }; // Presents where no changed struct could be told to be the main view
// ---- the private copy --------------------------------------------------------------------------------------
// NPC idle motion is absent from the game's 8-bit OTHER-eye object vectors; same-eye vectors fix it, but patching
// the per-view struct in place (validated for NPCs) also hands the game's AO and temporal filters the
// same eye's history, and they shimmer. The private-copy probe settled that the G-buffer and those passes read the
// struct through the SAME descriptors, so the G-buffer is given its own: a mirror of the ring the struct lives in
// (our upload buffer, same size, same offsets), filled at every submit with the game's bytes plus the same-eye
// previous view, and a rebuilt copy of the G-buffer's 15-CBV table whose slot 0 points into the mirror (see
// RedirectGraphicsTable). The game's struct is never written in this mode.
constexpr size_t kPvMirrorStruct = 1024;
constexpr uint64_t kPvMirrorMaxRing = 64ull << 20;
struct PvMirrorEntry
{
    uint64_t offset = 0;    // of the struct in the ring
    bool sameEye = false;   // bytes carry the same-eye +384/+576
    bool haveBytes = false; // false: unchanged since its last fill this Present, the mirror already holds it
    uint8_t bytes[kPvMirrorStruct];
};
void RetireDeferred(ID3D12Pageable* object, uint64_t presentIndex);
std::atomic<bool> g_pvMirrorOn { false };
// Which graphics lists the private copy redirects (overlay, live): 0 motion target at slot 0 (ran, no
// change), 1 motion target at ANY slot with depth, 2 every graphics list (compute passes always keep the game's).
// 3 (the default): the G-buffer found by the pass map (an R8G8_UNORM target among 3+, with depth),
// and slot 0 of the GAME'S table rewritten in place -- ~1000 binds a Present, far past what a table region holds.
std::atomic<int> g_pvMirrorScope { 3 };
std::atomic<uint64_t> g_rwScopeSeen[3] {}; // qualifying binds per list class: slot 0 / any slot / neither
std::atomic<uint64_t> g_rwInPlace { 0 }, g_rwMrtSeen { 0 };
std::atomic<int> g_mirrorReady { 0 }; // structs the latest fill made redirectable

class PvMirror
{
  public:
    // Submit threads (BeforeExecute), after ViewConstants released its lock. Returns how many structs the G-buffer
    // may now be redirected to: exactly the ones filled for the current candidates, so a ring slot that stopped
    // being a candidate is never redirected to (its mirror would be stale).
    int Fill(const std::vector<PvMirrorEntry>& entries, uint64_t ringSize, uint64_t ringGpu, uint64_t presentIndex)
    {
        std::lock_guard<std::mutex> lock(m_mx);
        if ((m_buf == nullptr || ringGpu != m_ringGpu || ringSize != m_ringSize) &&
            !RecreateLocked(ringSize, ringGpu, presentIndex))
            return 0;
        m_readyCount = 0;
        for (const PvMirrorEntry& e : entries)
        {
            if (e.offset + kPvMirrorStruct > m_ringSize)
                continue;
            const size_t unit = (size_t) (e.offset / 256);
            if (e.haveBytes)
            {
                if (std::memcmp(m_shadow.data() + e.offset, e.bytes, kPvMirrorStruct) != 0)
                {
                    std::memcpy(m_cpu + e.offset, e.bytes, kPvMirrorStruct);
                    std::memcpy(m_shadow.data() + e.offset, e.bytes, kPvMirrorStruct);
                    m_writes++;
                }
                m_filled[unit] = 1;
            }
            else if (m_filled[unit] == 0)
                continue; // the buffer was recreated since this struct's last fill: nothing to point at yet
            if (m_readyCount < kMaxReady)
                m_ready[m_readyCount++] = m_ringGpu + e.offset;
            if (e.sameEye)
                m_sameEntries++;
            m_entries++;
            // A struct a G-buffer bind was redirected to since the last fill: did this fill give it same-eye
            // values? That is the only question that says whether the redirect can change the picture.
            for (int k = 0; k < m_pendingCount; ++k)
            {
                if (m_pending[k] != m_ringGpu + e.offset)
                    continue;
                (e.sameEye ? m_redirSame : m_redirPlain)++;
                m_pending[k] = m_pending[--m_pendingCount];
                break;
            }
        }
        m_redirStale += (uint64_t) m_pendingCount; // redirected, but not a candidate this fill: not refilled
        m_pendingCount = 0;
        return m_readyCount;
    }

    // Recording threads, at a G-buffer table bind: the mirror address for the game's per-view CBV at `address`,
    // if that struct was filled by the latest fill and the whole view fits the mirror.
    bool Redirect(uint64_t address, uint64_t size, uint64_t* mirrorAddress)
    {
        std::lock_guard<std::mutex> lock(m_mx);
        for (int k = 0; k < m_readyCount; ++k)
        {
            if (m_ready[k] != address)
                continue;
            if (address - m_ringGpu + size > m_ringSize)
                return false;
            *mirrorAddress = m_gpu + (address - m_ringGpu);
            bool known = false;
            for (int p = 0; p < m_pendingCount && !known; ++p)
                known = m_pending[p] == address;
            if (!known && m_pendingCount < kMaxReady)
                m_pending[m_pendingCount++] = address;
            return true;
        }
        return false;
    }

    // Nothing may be redirected to until the next fill: the candidates died (their buffer's Release, a menu or a
    // size change -- a new ring slot could even reuse a dead one's address) or the mode was switched off.
    void ClearReady()
    {
        std::lock_guard<std::mutex> lock(m_mx);
        m_readyCount = 0;
    }

    void Stats(uint64_t* writes, uint64_t* entries, uint64_t* sameEntries, int* ready, uint64_t* redirSame,
               uint64_t* redirPlain, uint64_t* redirStale)
    {
        std::lock_guard<std::mutex> lock(m_mx);
        *writes = m_writes;
        *entries = m_entries;
        *sameEntries = m_sameEntries;
        *ready = m_readyCount;
        *redirSame = m_redirSame;
        *redirPlain = m_redirPlain;
        *redirStale = m_redirStale;
        m_writes = m_entries = m_sameEntries = m_redirSame = m_redirPlain = m_redirStale = 0;
    }

  private:
    bool RecreateLocked(uint64_t ringSize, uint64_t ringGpu, uint64_t presentIndex)
    {
        if (g_device == nullptr || ringSize == 0 || ringSize > kPvMirrorMaxRing)
        {
            if (!m_loggedTooBig)
                LOG_WARN("DuniaTaa: private copy: the ring is {} bytes (limit {}), or no device -- NOT mirrored, the "
                         "G-buffer keeps the game's own tables",
                         ringSize, kPvMirrorMaxRing);
            m_loggedTooBig = true;
            return false;
        }
        D3D12_HEAP_PROPERTIES heap {};
        heap.Type = D3D12_HEAP_TYPE_UPLOAD;
        D3D12_RESOURCE_DESC desc {};
        desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        desc.Width = ringSize;
        desc.Height = 1;
        desc.DepthOrArraySize = 1;
        desc.MipLevels = 1;
        desc.SampleDesc.Count = 1;
        desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        ID3D12Resource* buf = nullptr;
        if (FAILED(g_device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
                                                     D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&buf))))
        {
            LOG_ERROR("DuniaTaa: private copy: could not create the {} byte mirror", ringSize);
            m_loggedTooBig = true;
            return false;
        }
        uint8_t* cpu = nullptr;
        D3D12_RANGE none { 0, 0 };
        if (FAILED(buf->Map(0, &none, (void**) &cpu)) || cpu == nullptr)
        {
            buf->Release();
            LOG_ERROR("DuniaTaa: private copy: could not map the mirror");
            return false;
        }
        buf->SetName(L"DuniaTaa per-view mirror");
        // A list recorded a moment ago may still point into the old one: the deferred list, never Release().
        if (m_buf != nullptr)
            RetireDeferred(m_buf, presentIndex);
        m_buf = buf;
        m_cpu = cpu;
        m_gpu = buf->GetGPUVirtualAddress();
        m_ringGpu = ringGpu;
        m_ringSize = ringSize;
        m_shadow.assign((size_t) ringSize, 0);
        m_filled.assign((size_t) (ringSize / 256 + 1), 0);
        m_readyCount = 0;
        LOG_INFO("DuniaTaa: private copy: mirror of the per-view ring created ({} bytes, ring at {:X}, mirror at {:X})",
                 ringSize, ringGpu, m_gpu);
        return true;
    }

    static constexpr int kMaxReady = 64;
    mutable std::mutex m_mx;
    ID3D12Resource* m_buf = nullptr; // ours
    uint8_t* m_cpu = nullptr;
    uint64_t m_gpu = 0, m_ringGpu = 0, m_ringSize = 0;
    std::vector<uint8_t> m_shadow; // what the mirror holds: compared instead of reading write-combined memory
    std::vector<uint8_t> m_filled; // per 256-byte unit: filled since this buffer was created
    uint64_t m_ready[kMaxReady] {};
    int m_readyCount = 0;
    uint64_t m_writes = 0, m_entries = 0, m_sameEntries = 0;
    uint64_t m_pending[kMaxReady] {}; // redirected to since the last fill
    int m_pendingCount = 0;
    uint64_t m_redirSame = 0, m_redirPlain = 0, m_redirStale = 0;
    bool m_loggedTooBig = false;
};
PvMirror g_pvMirror;

struct PvArgs
{
    bool write = false;       // rewrite +384/+576 to the same eye's previous state
    bool mirror = false;      // ...into OUR mirror of the ring (the private copy), never into the game's struct
    bool ownJitter = false;   // patch our own jitter in
    float jitterScale = 1.0f; // 1, or the shake tests' 20 and 4
};
struct PvInfo
{
    bool jitterWritten = false;
    float jitterGame[2] {};
    float jitterOurs[2] {};
    double chainResidual = 0.0; // how closely this struct's +384 matched the previous struct's unjittered P*V
    double otherEyeStep = 0.0;  // |cam - original prevCam|: about the IPD
    double sameEyeStep = 0.0;   // |cam - the prevCam written|: head motion over two Presents, near zero at rest
};
struct TbInfo
{
    double residual = 0.0;     // verified: how closely the block matched our other-eye build
    double nearMiss = 0.0;     // unverified: the closest any block came
    float otherDepthCol = 0.0f; // the parallax term that was in the block (about +-0.58 at a 6 cm IPD)
    uint64_t scanned = 0;       // pool slots looked at this call (0 = it was where it was last time)
    int64_t slotDelta = 0;      // how many slots past the last position it turned up
    uint64_t mapBytes = 0;      // the pool's size
};

struct ViewState
{
    float vrp[16] {};
    float camPos[3] {};
    bool valid = false;
};

struct ViewFrame
{
    float proj[16] {};
    float c2w[16] {};
    ViewState cur;
    ViewState prevSameEye;
    // One Present back -- under AER the OTHER eye, parallax included. Read straight from the buffer (+384 /
    // +576), no history needed. It is what the game computed its own vectors against, so it is the baseline
    // an object's motion has to be judged against; see the decode shader.
    ViewState prevOtherEye;
    // This frame's jitter as the image's shift in pixels (x right, y down), from the projection that was
    // really in the struct when the GPU read it -- the game's, or ours if PatchPerView wrote it.
    float jitterPx[2] {};
    bool ok = false;
};

class ViewConstants
{
  public:
    // Returns false until there is both a current instance and one from two Presents ago; the matrix path does
    // nothing at all until then rather than reprojecting against a state it has not got.
    bool Update(UINT eyeW, UINT eyeH, uint64_t presentIndex, ViewFrame* out, const PvArgs* pv = nullptr,
                PvResult* pvResult = nullptr, PvInfo* pvInfo = nullptr)
    {
        if (out == nullptr || eyeW == 0 || eyeH == 0)
            return false;

        std::lock_guard<std::mutex> lock(m_lock);

        // Rescan when the shape changes, and also when nothing has been written for a while: the first scan
        // ran before the live instances existed and came back with 32 stale copies -- the raw viewport quad
        // turned out to match dozens of buffers, not the handful predicted.
        // Note there is no "m_count > 0" here. With it, a scan that came back empty -- which is what the
        // Presents right after a Forget() look like, before the game has remapped its buffers -- could never
        // be retried, because nothing else would set m_forW to a different value. Matrix motion would be dead
        // for the rest of the run, silently. m_lastChange is set by the rescan itself, so the empty case
        // retries on the same timer as the stale one.
        // A shape change restarts a sweep in progress (it was matching the wrong size); the staleness timer
        // does not -- a sweep that takes longer than the timer would otherwise never finish.
        // The resweep (see m_resweep): too few copies patched per Present means a ring slot the set does not
        // know. Only while the per-view patch is being called every Present (real motion on) with own jitter.
        if (m_pvWindowStart == 0)
            m_pvWindowStart = presentIndex;
        else if (presentIndex - m_pvWindowStart >= 120)
        {
            const double presents = (double) (presentIndex - m_pvWindowStart);
            const double perPresent = (double) m_pvPatchedWindow / presents;
            const bool active = m_pvOwnJitter && m_pvCallsWindow >= (uint64_t) presents && m_count > 0;
            // The ring probe first, every window: a few hundred 16-byte compares, and it is where the copies are.
            const size_t probed = active ? ProbeRing(eyeW, eyeH, presentIndex) : 0;
            // A full sweep still running does not block this: one run at 60% had the initial sweep
            // stuck at 23 of 1209 MB throughout -- the mapped set churns faster than 1.5 ms a
            // Present can cover -- with one copy per Present patched and the resweep waiting for it to finish.
            // The resweep restarts the scan so the mappings the known copies live in are walked first
            // (NoteUsefulPointer), which is where the frame's other copies are if they are anywhere reachable.
            if (active && probed == 0 && perPresent < 2.5 && !m_resweep && presentIndex - m_lastResweep > 450)
            {
                m_scan = ConstantCapture_Dx12::PatternScan {};
                m_committed = 0;
                m_scanning = true;
                m_resweep = true;
                m_lastResweep = presentIndex;
                g_vcDeadDrops.fetch_add(1, std::memory_order_relaxed);
                if (++m_resweeps <= 5)
                {
                    std::string where;
                    for (size_t i = 0; i < m_count && i < 8; ++i)
                        where += std::format("{}{:X}", i ? " " : "", (size_t) m_candidates[i]);
                    LOG_INFO("DuniaTaa: view constants: {:.2f} copies patched per Present (want ~3) with {} candidates "
                             "at [{}] -- re-sweeping for the ring slots the set is missing, their mappings first; "
                             "the candidates in hand stay in use, motion keeps running",
                             perPresent, m_count, where);
                }
            }
            m_pvWindowStart = presentIndex;
            m_pvPatchedWindow = 0;
            m_pvCallsWindow = 0;
        }

        const bool stale = !m_scanning && m_lastChange != 0 && presentIndex > m_lastChange + m_retryPresents;
        const float pattern[4] = { (float) eyeW, (float) eyeH, 1.0f / (float) eyeW, 1.0f / (float) eyeH };
        if (m_forW != eyeW || m_forH != eyeH || stale)
        {
            // Start a sweep; it runs in slices over the Presents that follow. The candidates already in hand
            // stay in use meanwhile (after a Forget there are none, and the matrix path waits).
            m_scan = ConstantCapture_Dx12::PatternScan {};
            m_scanning = true;
            m_forW = eyeW;
            m_forH = eyeH;
            m_lastChange = presentIndex;
            // And the watch on the game's own writes, which is the route that does not search at all: if
            // Dunia passes a written range at Unmap, every rewrite of a per-view block is a hit.
            ConstantCapture_Dx12::SetWatchPattern(pattern, 4, 1e-4f, IsPerViewStruct);
            ConstantCapture_Dx12::SetRawMatchObserver(ObserveViewportQuad);
        }

        // Hits from the game's own writes come first, and they are authoritative: a block the game just
        // rewrote at the eye size and that validates as a per-view struct IS a live instance. They are added
        // as they arrive and used immediately.
        {
            const uint8_t* hits[kVcMaxCandidates] {};
            const size_t n = ConstantCapture_Dx12::TakeWrittenHits(hits, kVcMaxCandidates);
            size_t added = 0;
            for (size_t i = 0; i < n && m_count < kVcMaxCandidates; ++i)
            {
                bool have = false;
                for (size_t j = 0; j < m_count && !have; ++j)
                    have = m_candidates[j] == hits[i];
                if (have)
                    continue;
                m_candidates[m_count] = hits[i];
                m_sigValid[m_count] = false;
                ++m_count;
                ++added;
                ++m_writtenAdded;
            }
            if (added != 0)
            {
                m_lastChange = presentIndex;
                for (size_t i = m_count - added; i < m_count; ++i)
                    ConstantCapture_Dx12::NoteUsefulPointer(m_candidates[i]);
                if (!m_writtenLogged)
                {
                    m_writtenLogged = true;
                    const auto s = ConstantCapture_Dx12::WrittenRangeStats();
                    LOG_INFO("DuniaTaa: view constants: {} live per-view struct(s) caught at the game's own "
                             "Unmap -- no sweep needed. Unmap: {} calls, {} with a written range, {} ranges "
                             "checked, {} hits",
                             added, s.calls, s.withRange, s.checked, s.hits);
                }
                // The sweep has nothing to add that the game's writes will not also deliver.
                m_scanning = false;
            }
        }

        if (m_scanning)
        {
            ConstantCapture_Dx12::FindPattern(pattern, 4, 1e-4f, m_pending, kVcMaxCandidates, IsPerViewStruct,
                                              m_scan, m_count == 0 ? kVcScanMsPerPresentUrgent : kVcScanMsPerPresent);
            // Whatever the sweep has found so far is used now, not when it finishes: the live pair is often
            // among the first hits and the rest of a 1.3 GB set takes tens of seconds.
            if (m_scan.found > m_committed)
            {
                size_t added = 0;
                for (size_t i = m_committed; i < m_scan.found; ++i)
                {
                    // Never twice: a resweep walks the same mappings the set came from.
                    bool have = false;
                    for (size_t j = 0; j < m_count && !have; ++j)
                        have = m_candidates[j] == m_pending[i];
                    if (have)
                        continue;
                    size_t at = m_count;
                    if (at >= kVcMaxCandidates)
                    {
                        // Full: the slot most often seen holding something else gives way.
                        int worstDead = 0;
                        at = kVcMaxCandidates;
                        for (size_t j = 0; j < m_count; ++j)
                        {
                            if (m_pvDead[j] > worstDead)
                            {
                                worstDead = m_pvDead[j];
                                at = j;
                            }
                        }
                        if (at == kVcMaxCandidates)
                            break;
                    }
                    else
                        ++m_count;
                    m_candidates[at] = m_pending[i];
                    m_sigValid[at] = false;
                    m_pvSigValid[at] = false;
                    m_pvWrote[at] = false;
                    m_pvDead[at] = 0;
                    ConstantCapture_Dx12::NoteUsefulPointer(m_pending[i]);
                    ++added;
                }
                m_committed = m_scan.found;
                m_lastChange = presentIndex;
                if (m_resweep && added != 0)
                {
                    // Found what it was for; the rest of the mapped set is not worth the slices.
                    m_scanning = false;
                    m_resweep = false;
                    if (m_resweeps <= 5)
                        LOG_INFO("DuniaTaa: view constants: resweep added {} slot(s), the set holds {}", added, m_count);
                }
            }
            if (!m_scan.done && m_resweep && m_scan.slices >= 600)
            {
                // Nothing new in the mappings the known copies live in, nor in what followed: the frame's other
                // copies are not where this sweep can reach. Stop; the next trigger is 450 Presents away.
                m_scanning = false;
                m_resweep = false;
                if (m_resweeps <= 5)
                    LOG_WARN("DuniaTaa: view constants: resweep found nothing new in {} slices ({:.0f} ms, {:.1f} MB, "
                             "mapping {} of {}) -- the frame's other copies are outside the swept mappings",
                             m_scan.slices, m_scan.ms, (double) m_scan.swept / (1024.0 * 1024.0), m_scan.index,
                             m_scan.order.size());
            }
            else if (!m_scan.done)
            {
                // A sweep that does not finish is silent by construction -- the scan line only prints when it
                // does -- and that silence cost a run. Say so once, with the numbers that explain it.
                if (m_scan.slices == 200 && !m_scanStallLogged)
                {
                    m_scanStallLogged = true;
                    const auto s = ConstantCapture_Dx12::WrittenRangeStats();
                    LOG_WARN("DuniaTaa: view constants: the sweep has run {} slices ({:.0f} ms) and is not done "
                             "-- matching [{}, {}, 1/{}, 1/{}]: {} raw matches, {} found and in use, "
                             "validator rejected {} on projection and {} on VRP; {:.1f} of {:.1f} MB swept so "
                             "far, mapping {} of {} (largest {:.1f} MB), {} pending pointers dropped. Unmap: "
                             "{} calls, {} with a written range, {} checked ({} too big, {} no mapping, {} past "
                             "the end), {} raw matches, {} hits",
                             m_scan.slices, m_scan.ms, eyeW, eyeH, eyeW, eyeH, m_scan.raw, m_scan.found,
                             g_vcRejectProj.load(), g_vcRejectVrp.load(),
                             (double) m_scan.swept / (1024.0 * 1024.0), (double) m_scan.total / (1024.0 * 1024.0),
                             m_scan.index, m_scan.order.size(), (double) m_scan.largest / (1024.0 * 1024.0),
                             m_scanDropped, s.calls, s.withRange, s.checked, s.tooBig, s.noMapping, s.pastEnd,
                             s.rawMatches, s.hits);
                }
                if (m_count == 0)
                    return false;
            }
            else
            {
                m_scanning = false;
                m_resweep = false;
                m_committed = 0;
                // The mappings these live in lead the next sweep, and so do any of the same size after a
                // menu has replaced them.
                for (size_t i = 0; i < m_count; ++i)
                    ConstantCapture_Dx12::NoteUsefulPointer(m_candidates[i]);
                m_lastChange = presentIndex;
                // The timer backs off while nothing is found and resets the moment something is, so a run that
                // keeps coming back empty does not sweep for ever.
                m_retryPresents =
                    m_count > 0 ? kVcRescanPresents : std::min(m_retryPresents * 2, kVcRescanMaxPresents);
                // Everything that could explain a zero is on this one line: how many mappings there were to
                // read, how much was read, how long that took and over how many Presents (the hitch, or its
                // absence), how many raw hits the fingerprint gave, how many mapped buffers have died since
                // start, and how many addresses were re-mapped under a new resource. Read this before the picture.
                LOG_INFO("DuniaTaa: view constants: {} buffer(s) carry [{}, {}, 1/{}, 1/{}], {} of them are a "
                         "real per-view struct -- swept {} of {} mapped buffers, ring-sized largest first "
                         "(largest {:.1f} MB, set {:.1f} MB), {:.1f} MB in {:.1f} ms over {} Presents ({} pending pointers dropped to dying "
                         "buffers); {} mapped buffers released so far, {} addresses re-mapped under a new "
                         "resource; retry in {} Presents if nothing changes",
                         m_scan.raw, eyeW, eyeH, eyeW, eyeH, m_count, m_scan.index + 1, m_scan.order.size(),
                         (double) m_scan.largest / (1024.0 * 1024.0), (double) m_scan.total / (1024.0 * 1024.0),
                         (double) m_scan.swept / (1024.0 * 1024.0),
                         m_scan.ms, m_scan.slices, m_scanDropped,
                         ConstantCapture_Dx12::ForgetCount(), ConstantCapture_Dx12::RebindCount(),
                         m_retryPresents);
                m_scanDropped = 0;
                m_scanStallLogged = false;
                if (m_count == 0)
                    return false;
            }
        }

        // Patch first, read second. The game writes a frame's struct three times over, and a copy written after
        // the frame's last submit so far has not been looked at by BeforeExecute yet. Reading that copy here
        // told DLSS the GAME's jitter for a frame the GPU then rendered with OURS -- on exactly half the frames
        // (300 of 600 in the x4 shake test, where the image duly shook), because which copy comes first in the
        // list differs between the two eyes' slots. Nothing submitted can have read such a copy, so now is
        // still in time to write it.
        if (pv != nullptr && m_count != 0)
        {
            const PvResult r = PatchPerViewLocked(pv->write, pv->mirror, pv->ownJitter, pv->jitterScale, pvInfo);
            if (pvResult != nullptr)
                *pvResult = r;
        }

        // Which instance is THIS Present's. The game rings several and writes one per Present, so the one whose
        // bytes changed since we last looked is the one that was just written. Cheaper and less fragile than
        // guessing from parity, and it says so in the log when the assumption does not hold.
        int chosen = -1;
        int changed = 0;
        m_chosenMainFov = false;
        m_chosenRank = -1;
        for (size_t i = 0; i < m_count; ++i)
        {
            const uint8_t* base = m_candidates[i] - kVcViewport;
            float sig[7];
            std::memcpy(sig, base + kVcCamPos, 3 * sizeof(float));
            std::memcpy(sig + 3, base + kVcVrp, 4 * sizeof(float));
            const bool differs = !m_sigValid[i] || std::memcmp(sig, m_sig[i], sizeof(sig)) != 0;
            if (m_sigValid[i] && differs)
            {
                ++changed;
                // With a weapon out FC6 renders a second eye-sized view (the weapon / aim-down-sights camera,
                // another field of view) with a per-view struct of its own, and "the first that changed" was
                // sometimes that one: DLSS was then fed the weapon camera's matrices and jitter for the whole
                // eye -- shimmer and sideways-sliding ground exactly while aiming (the log: "3
                // instances changed in one Present"). The main view is the one whose projection (p00, p11)
                // matches what was chosen before; the first frame ever takes the first.
                // The decisive test is CONTINUITY, not the FOV (aiming down sights changes the main view's FOV,
                // and with a gun out every frame changes two structs, so a learned FOV went stale and stayed
                // stale -- the view broke again eventually): the main view's previous view
                // (+384, through ReadPrev) is the previous Present's chosen main view exactly; the weapon
                // view's is its own previous frame and matches nothing we hold. FOV is the fallback.
                float pv[16], pc[3];
                ReadPrev(i, pv, pc);
                const ViewState& lastMain = m_history[(presentIndex + 1) & 1];
                bool continues = lastMain.valid;
                for (int k = 0; k < 16 && continues; ++k)
                    continues = std::abs(pv[k] - lastMain.vrp[k]) < 1e-4f;
                float p[16];
                std::memcpy(p, base + kVcProjection, sizeof(p));
                const bool mainFov = !m_mainFovValid || (p[0] == m_mainP00 && p[5] == m_mainP11);
                const int rank = continues ? 2 : mainFov ? 1 : 0;
                if (chosen < 0 || rank > m_chosenRank)
                {
                    chosen = (int) i;
                    m_chosenRank = rank;
                    m_chosenMainFov = mainFov;
                }
            }
            std::memcpy(m_sig[i], sig, sizeof(sig));
            m_sigValid[i] = true;
        }
        if (chosen >= 0 && (changed == 1 || m_chosenRank == 2))
        {
            // Only one view was written: that is the main view, and its FOV is what the weapon view is told
            // apart from later. (A weapon view alone never happens; the main view is drawn every frame.)
            float p[16];
            std::memcpy(p, m_candidates[chosen] - kVcViewport + kVcProjection, sizeof(p));
            if (!m_mainFovValid || p[0] != m_mainP00 || p[5] != m_mainP11)
            {
                m_mainP00 = p[0];
                m_mainP11 = p[5];
                m_mainFovValid = true;
            }
        }
        else if (chosen >= 0 && changed > 1 && m_chosenRank == 0)
        {
            static bool logged = false;
            if (!logged)
            {
                logged = true;
                LOG_WARN("DuniaTaa: view constants: {} instances changed this Present and none continues the "
                         "previous main view or carries its field of view -- taking the first. Expected once "
                         "after a menu; if it repeats, the choice is wrong",
                         changed);
            }
            g_vcAmbiguous.fetch_add(1, std::memory_order_relaxed);
        }
        if (chosen < 0)
        {
            static bool logged = false;
            if (!logged)
            {
                logged = true;
                LOG_INFO("DuniaTaa: view constants: no instance changed this Present -- first frames, or the "
                         "ones found are stale copies. Rescanning after {} more Presents",
                         kVcRescanPresents);
            }
            return false;
        }
        m_lastChange = presentIndex;
        if (changed > 1)
        {
            static bool logged = false;
            if (!logged)
            {
                logged = true;
                LOG_WARN("DuniaTaa: view constants: {} instances changed in one Present, taking the first. If "
                         "the motion is wrong this is the first thing to doubt",
                         changed);
            }
        }

        const uint8_t* base = m_candidates[chosen] - kVcViewport;
        std::memcpy(out->proj, base + kVcProjection, sizeof(out->proj));
        std::memcpy(out->c2w, base + kVcCameraToWorld, sizeof(out->c2w));
        // The UNJITTERED view-rot-projection, rebuilt from the projection with its jitter terms zeroed and the
        // view rotation -- not +320, which carries this frame's jitter. With the game's 2-phase jitter the
        // two are interchangeable for same-eye motion (each eye's jitter never changes, so it cancels); with
        // our own jitter they are not, and the difference would go to DLSS as motion. BuildPrevClipMatrix puts
        // the CURRENT jitter back, which is what a jitter-free vector sampled on the jittered grid needs.
        {
            float viewM[16];
            std::memcpy(viewM, base + kVcView, sizeof(viewM));
            const float* p = out->proj;
            for (int c = 0; c < 3; ++c)
            {
                out->cur.vrp[0 * 4 + c] = p[0] * viewM[0 * 4 + c];
                out->cur.vrp[1 * 4 + c] = p[5] * viewM[1 * 4 + c];
                out->cur.vrp[2 * 4 + c] = p[10] * viewM[2 * 4 + c];
                out->cur.vrp[3 * 4 + c] = p[14] * viewM[2 * 4 + c];
            }
            out->cur.vrp[3] = 0.0f;
            out->cur.vrp[7] = 0.0f;
            out->cur.vrp[11] = p[11];
            out->cur.vrp[15] = p[15];
        }
        out->jitterPx[0] = -out->proj[2] * (float) eyeW * 0.5f;
        out->jitterPx[1] = out->proj[6] * (float) eyeH * 0.5f;
        std::memcpy(out->cur.camPos, base + kVcCamPos, sizeof(out->cur.camPos));
        out->cur.valid = true;
        // Through ReadPrev, never straight from +384/+576: with the per-view patch on those bytes are OURS.
        ReadPrev((size_t) chosen, out->prevOtherEye.vrp, out->prevOtherEye.camPos);
        out->prevOtherEye.valid = true;

        // The slot for this parity still holds what was stored two Presents ago -- the same eye. Read before
        // overwriting; that ordering is the whole trick, and it is FC5's.
        const size_t parity = (size_t) (presentIndex & 1);
        out->prevSameEye = m_history[parity];
        m_history[parity] = out->cur;

        out->ok = out->prevSameEye.valid;
        return out->ok;
    }

    // The dying buffer's CPU range. Returns true when a cached pointer was inside it, which is the only case
    // that costs anything: the caller sees one of these for every mapped buffer the game frees, and almost
    // none of them hold a per-view struct. Comparing the pointers is safe -- it is reading through them that
    // would be the crash, and nothing here dereferences one.
    // The ring probe. The frame's per-view copies are rung through ONE mapping on a 1 KB grid: a run at 60%
    // logged eight slots, every one 0x400-aligned to the first,
    // within 80 KB -- while the pattern sweep of the 1.2 GB mapped set never got past 64 MB. So the sweep is
    // the wrong instrument for a ring: this walks a few hundred slots either side of every known copy, inside
    // its own mapping, and adds any slot that holds the eye's viewport quad and validates as a struct. Under
    // m_lock, on the present thread; the reads are of mapped upload memory the registry says is alive, the
    // same trust the sweep has. Returns how many were added.
    size_t ProbeRing(UINT eyeW, UINT eyeH, uint64_t presentIndex)
    {
        constexpr ptrdiff_t kSlot = 0x400;
        constexpr ptrdiff_t kReach = 256; // slots either side: 256 KB each way, 512 compares per seed
        const float want[4] = { (float) eyeW, (float) eyeH, 1.0f / (float) eyeW, 1.0f / (float) eyeH };
        size_t added = 0;
        const size_t seeds = std::min<size_t>(m_count, 8);
        for (size_t s = 0; s < seeds && m_count < kVcMaxCandidates; ++s)
        {
            const uint8_t* seed = m_candidates[s];
            const uint8_t* mapBegin = nullptr;
            uint64_t mapSize = 0;
            if (!ConstantCapture_Dx12::MappingOf(seed, &mapBegin, &mapSize) || mapBegin == nullptr)
                continue;
            const uint8_t* lo = mapBegin + kVcViewport;          // the quad sits kVcViewport into the struct
            const uint8_t* hi = mapBegin + mapSize - (kVcJitterPx + 16); // the whole struct (through +704) must fit
            if (hi <= lo)
                continue;
            for (ptrdiff_t k = -kReach; k <= kReach && m_count < kVcMaxCandidates; ++k)
            {
                if (k == 0)
                    continue;
                const uint8_t* p = seed + k * kSlot;
                if (p < lo || p > hi)
                    continue;
                float q[4];
                std::memcpy(q, p, sizeof(q));
                bool match = FiniteBits(q, 4);
                for (int j = 0; j < 4 && match; ++j)
                    match = std::abs(q[j] - want[j]) <= std::abs(want[j]) * 1e-4f;
                if (!match || !IsPerViewStruct(p))
                    continue;
                bool have = false;
                for (size_t j = 0; j < m_count && !have; ++j)
                    have = m_candidates[j] == p;
                if (have)
                    continue;
                m_candidates[m_count] = p;
                m_sigValid[m_count] = false;
                m_pvSigValid[m_count] = false;
                m_pvWrote[m_count] = false;
                m_pvDead[m_count] = 0;
                ++m_count;
                ++added;
            }
        }
        if (added != 0)
        {
            m_lastChange = presentIndex;
            if (++m_ringProbeAdds <= 8)
                LOG_INFO("DuniaTaa: view constants: ring probe added {} slot(s) on the 1 KB grid around the known copies; "
                         "the set holds {}",
                         added, m_count);
        }
        return added;
    }

    bool ForgetIfWithin(const uint8_t* begin, const uint8_t* end)
    {
        if (begin == nullptr || end <= begin)
            return false;

        {
            // The motion pass's blocks first, wherever they are waiting. Only the ones in this buffer: they
            // are independent of each other, unlike the per-view set below.
            std::lock_guard<std::mutex> pending(g_tbPendingLock);
            size_t kept = 0;
            for (size_t i = 0; i < g_tbPendingCount; ++i)
                if (!(g_tbPending[i] >= begin && g_tbPending[i] < end))
                    g_tbPending[kept++] = g_tbPending[i];
            g_tbPendingCount = kept;
        }
        {
            std::lock_guard<std::mutex> lock(m_lock);
            size_t keptTb = 0;
            for (size_t i = 0; i < m_tbCount; ++i)
                if (!(m_tb[i] >= begin && m_tb[i] < end))
                    m_tb[keptTb++] = m_tb[i];
            m_tbCount = keptTb;
            // Everything else that points into the dying buffer: the last-found block, the one written, and
            // the cached pool range the scan walks (any overlap at all).
            if (m_tbLast >= begin && m_tbLast < end)
                m_tbLast = nullptr;
            for (int k = 0; k < kTbRecent; ++k)
                if (m_tbRecent[k] >= begin && m_tbRecent[k] < end)
                    m_tbRecent[k] = nullptr;
            if (m_tbWritten >= begin && m_tbWritten < end)
                m_tbWritten = nullptr;
            if (m_tbMapBegin != nullptr && m_tbMapBegin < end && begin < m_tbMapBegin + m_tbMapSize)
            {
                m_tbMapBegin = nullptr;
                m_tbMapSize = 0;
                m_tbLast = nullptr;
                m_tbWritten = nullptr;
                for (int k = 0; k < kTbRecent; ++k)
                    m_tbRecent[k] = nullptr;
            }
            bool hit = false;
            for (size_t i = 0; i < m_count && !hit; ++i)
                hit = m_candidates[i] >= begin && m_candidates[i] < end;
            // A sweep in progress may already hold a pointer into this buffer: DROP those entries and carry
            // on. The first version restarted the sweep instead, and with the game freeing ~50 mapped buffers
            // a second -- many of which briefly hold a valid-looking per-view struct -- it restarted for ever:
            // a whole run with no scan line, 0 decodes, and a sweep slice paid on every Present (21/s).
            if (m_scanning)
            {
                size_t kept = 0;
                for (size_t i = 0; i < m_scan.found; ++i)
                {
                    if (m_pending[i] >= begin && m_pending[i] < end)
                        ++m_scanDropped;
                    else
                        m_pending[kept++] = m_pending[i];
                }
                m_scan.found = kept;
            }
            if (!hit)
                return false;
        }

        // All of them, not just the ones in this buffer: the game rings these structs through one allocation
        // scheme, so a set that has lost a member is not a set to keep choosing "this Present's instance" out
        // of. A rescan is cheap next to being wrong about which matrix is current.
        Forget();
        return true;
    }

    // Drops every raw pointer into the game's buffers and everything derived from one. Must be called while
    // the memory behind them is still valid -- from the resource's Release, not from noticing bad contents.
    void Forget()
    {
        std::lock_guard<std::mutex> lock(m_lock);
        ForgetLocked();
    }

    // For the private-copy feasibility probe: every candidate slot's struct BASE (the viewport quad sits at
    // +kVcViewport) and the copy index it last held. Pointers are compared by the caller, never read through.
    int SnapshotCopies(const uint8_t** bases, int* copyIdx, int max) const
    {
        std::lock_guard<std::mutex> lock(m_lock);
        int n = 0;
        for (size_t i = 0; i < m_count && n < max; ++i)
        {
            if (m_candidates[i] == nullptr)
                continue;
            bases[n] = m_candidates[i] - kVcViewport;
            copyIdx[n] = m_copyIdx[i];
            ++n;
        }
        return n;
    }

    // The body, for a caller that already holds m_lock (PatchPerViewLocked drops a set with a dead member).
    void ForgetLocked()
    {
        m_forW = 0;
        m_forH = 0;
        m_count = 0;
        m_scanning = false;
        m_committed = 0;
        // The signatures are copies OF the candidates, so they are as stale as the pointers that made them;
        // leaving them behind would have the next scan compare a fresh instance against a dead one's bytes.
        for (size_t i = 0; i < kVcMaxCandidates; ++i)
        {
            m_sigValid[i] = false;
            m_mirValid[i] = false;
            m_mirFilledBase[i] = nullptr;
        }
        g_pvMirror.ClearReady(); // lock order m_lock -> PvMirror's; nothing takes them the other way
        // Not a rescan, so m_lastChange is left alone: the next Update sees m_forW == 0, rescans immediately,
        // and only then starts the retry timer -- from the short interval again, because a buffer dying is new
        // information and the replacement is usually mapped within a frame or two.
        m_retryPresents = kVcRescanPresents;
        m_history[0].valid = false;
        m_history[1].valid = false;
        m_mainFovValid = false;
        m_chosenMainFov = false;
        // The block pointers stay: they are dropped by range, when THEIR buffer dies.
        m_pvCurIndex = -1;
        // Everything PatchPerView keeps is indexed by candidate or chained from one.
        for (size_t i = 0; i < kVcMaxCandidates; ++i)
        {
            m_pvSigValid[i] = false;
            m_pvWrote[i] = false;
            m_pvDead[i] = 0;
        }
        for (int e = 0; e < kPvRing; ++e)
            m_pvRing[e].valid = false;
        m_pvSameValid = false;
    }

    // GUARDRAIL -- the one place this module WRITES into memory the game owns. Called from the game's
    // recording thread, right after the motion pass's own Dispatch is recorded: the game wrote the block
    // before it bound it (the capture read it at the bind and it already held the frame's matrix), and the
    // GPU reads it at execution, so this lands between the two.
    //
    // Everything -- reading the candidates, verifying, writing -- happens under m_lock, the lock
    // ForgetIfWithin takes from the buffer's Release while the buffer is still alive. Do not hand these
    // pointers out, and do not move the write outside the lock: that is the menu crash a fifth time.
    //
    // It writes ONLY into a block that has just proved itself: its matrix must equal, to float noise, the
    // other-eye matrix rebuilt from a per-view struct. That is the offline gate run in game every frame, and
    // it is a value only the real computation produces -- a stale block, a block of another pass, a wrong
    // per-view candidate or a broken derivation all fail it and nothing is written.
    //
    // try_lock, not lock: Update holds m_lock through a sweep slice (up to 6 ms) and a recording thread
    // must not wait on that. A busy frame keeps the game's other-eye matrix and the decode is told so.
    TbResult PatchTemporal(bool write, float sameOut[16], TbInfo* info)
    {
        std::unique_lock<std::mutex> lock(m_lock, std::try_to_lock);
        if (!lock.owns_lock())
            return TbResult::Busy;

        {
            std::lock_guard<std::mutex> pending(g_tbPendingLock);
            for (size_t i = 0; i < g_tbPendingCount; ++i)
            {
                bool have = false;
                for (size_t j = 0; j < m_tbCount && !have; ++j)
                    have = m_tb[j] == g_tbPending[i];
                if (!have && m_tbCount < kTbMaxCandidates)
                    m_tb[m_tbCount++] = g_tbPending[i];
            }
            g_tbPendingCount = 0;
        }
        if (m_tbCount == 0 && m_tbLast == nullptr)
            return TbResult::NoBlock;
        // This frame's per-view struct is the one PatchPerView last saw the game rewrite. Searching every
        // candidate against every block is how the first version "verified" two all-zero copies.
        if (m_count == 0 || m_pvCurIndex < 0 || (size_t) m_pvCurIndex >= m_count)
            return TbResult::NoView;


        const size_t v = (size_t) m_pvCurIndex;
        const uint8_t* viewBase = m_candidates[v] - kVcViewport;
        float proj[16], viewM[16], prevVrpF[16], cam[3], prevCam[3], quad[4];
        std::memcpy(proj, viewBase + kVcProjection, sizeof(proj));
        std::memcpy(viewM, viewBase + kVcView, sizeof(viewM));
        std::memcpy(cam, viewBase + kVcCamPos, sizeof(cam));
        std::memcpy(quad, viewBase + kVcViewport, sizeof(quad));
        // The game's ORIGINAL previous view, which is what it built the block's matrix from.
        ReadPrev(v, prevVrpF, prevCam);
        if (!FiniteBits(proj, 16) || !FiniteBits(viewM, 16) || !FiniteBits(prevVrpF, 16) || !FiniteBits(cam, 3) ||
            !FiniteBits(prevCam, 3))
            return TbResult::NoView;
        if (!(std::abs(prevVrpF[0]) + std::abs(prevVrpF[1]) + std::abs(prevVrpF[2]) > 0.1f) || !(proj[0] > 0.01f))
            return TbResult::NoView;

        // cur = P with the jitter zeroed, times the view rotation; camera-relative, so no translation.
        double p[16], rot[16] {}, cur[16], curInv[16], prevVrp[16];
        for (int i = 0; i < 16; ++i)
        {
            p[i] = proj[i];
            prevVrp[i] = prevVrpF[i];
        }
        p[2] = 0.0;
        p[6] = 0.0;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                rot[i * 4 + j] = viewM[i * 4 + j];
        rot[15] = 1.0;
        Mul4(p, rot, cur);
        if (!Invert4(cur, curInv))
            return TbResult::NoView;

        auto build = [&](const double prev[16], const double prevCamD[3], double out[16])
        {
            const double t[16] = { 1, 0, 0, (double) cam[0] - prevCamD[0],
                                   0, 1, 0, (double) cam[1] - prevCamD[1],
                                   0, 0, 1, (double) cam[2] - prevCamD[2],
                                   0, 0, 0, 1 };
            double pt[16];
            Mul4(prev, t, pt);
            Mul4(pt, curInv, out);
        };
        const double prevCamD[3] = { prevCam[0], prevCam[1], prevCam[2] };
        double other[16];
        build(prevVrp, prevCamD, other);

        // How far a block's matrix is from the game's own other-eye matrix for THIS frame. Rows x, y, w: the
        // shader never loads the z row. A slot handed to another pass since, or a NaN, is simply far away.
        float game[16];
        auto residualOf = [&](const uint8_t* block) -> double
        {
            // Register 7 must be THIS frame's viewport quad, bit for bit, for every candidate -- a seed slot
            // may have gone to another pass since it was seen -- and the matrix must be numbers.
            float q[4];
            std::memcpy(q, block + kTbViewport, sizeof(q));
            if (std::memcmp(q, quad, sizeof(q)) != 0)
                return 1e30;
            std::memcpy(game, block + kTbMatrix, sizeof(game));
            if (!FiniteBits(game, 16))
                return 1e30;
            double worst = 0.0;
            for (int r = 0; r < 4; ++r)
            {
                if (r == 2)
                    continue;
                for (int c = 0; c < 4; ++c)
                {
                    worst = std::max(worst, std::abs(other[r * 4 + c] - (double) game[r * 4 + c]));
                }
            }
            return worst;
        };

        // This frame's same-eye matrix, if PatchPerView chained this frame's struct.
        float sameRows[16] {};
        bool haveSame = false;
        if (m_pvSameValid && FiniteBits(m_pvSameVrp, 16) && FiniteBits(m_pvSameCam, 3))
        {
            haveSame = true;
            double sameVrpD[16], sameD[16];
            for (int i = 0; i < 16; ++i)
                sameVrpD[i] = m_pvSameVrp[i];
            const double sameCamD[3] = { m_pvSameCam[0], m_pvSameCam[1], m_pvSameCam[2] };
            build(sameVrpD, sameCamD, sameD);
            for (int i = 0; i < 16; ++i)
                sameRows[i] = (float) sameD[i];

            // A second dispatch in the same frame: the block we wrote holds THIS frame's same-eye rows. The
            // first version asked only "does it still hold what we wrote" -- and a block the game never
            // rewrites (it had moved on to another slot) answered yes for the rest of the run.
            if (m_tbWritten != nullptr)
            {
                float now[16];
                std::memcpy(now, m_tbWritten + kTbMatrix, sizeof(now));
                if (std::memcmp(now, sameRows, 8 * sizeof(float)) == 0 &&
                    std::memcmp(now + 12, sameRows + 12, 4 * sizeof(float)) == 0)
                    return TbResult::AlreadyDone;
            }
        }

        const uint8_t* found = nullptr;
        double nearMiss = 1e30, residual = 1e30;
        auto consider = [&](const uint8_t* block)
        {
            const double r = residualOf(block);
            nearMiss = std::min(nearMiss, r);
            if (r < kTbVerifyTolerance)
            {
                found = block;
                residual = r;
            }
        };

        // The last few places it was found first: if Dunia gives each eye a slot of its own (the captures
        // showed two addresses alternating), this is all that ever runs.
        for (int k = 0; k < kTbRecent && found == nullptr; ++k)
            if (m_tbRecent[k] != nullptr)
                consider(m_tbRecent[k]);
        for (size_t b = 0; b < m_tbCount && found == nullptr; ++b)
            consider(m_tb[b]);

        // The block MOVES: Dunia hands the pass a fresh 512-byte slot out of a constant pool every frame
        // (one run: 1235 quads looked like this block, the 16 kept were stale at once and the patch
        // never engaged). So look through the pool the known ones live in, slot by slot, starting just after
        // where it was last found -- a pool is handed out in order, so that is usually a short walk. One
        // 16-byte read per slot (register 7 must be this frame's viewport quad, exactly); only a slot that
        // passes costs the matrix read.
        uint64_t scanned = 0;
        int64_t slotDelta = 0;
        // A walk that keeps failing must not be paid for on every frame: after eight misses in a row it runs
        // on one call in thirty. The per-window line shows the walks and the slots read.
        const bool walk = m_tbMissStreak < 8 || m_tbMissStreak % 30 == 0;
        if (found == nullptr && walk)
        {
            const uint8_t* seed = m_tbLast != nullptr ? m_tbLast : m_tb[0];
            if (m_tbMapBegin == nullptr || seed < m_tbMapBegin || seed >= m_tbMapBegin + m_tbMapSize)
            {
                m_tbMapBegin = nullptr;
                m_tbMapSize = 0;
                const uint8_t* begin = nullptr;
                uint64_t size = 0;
                if (ConstantCapture_Dx12::MappingOf(seed, &begin, &size))
                {
                    m_tbMapBegin = begin;
                    m_tbMapSize = size;
                }
            }
            if (m_tbMapBegin != nullptr && m_tbMapSize >= kTbSlot)
            {
                const uint64_t slots = m_tbMapSize / kTbSlot;
                const uint64_t seedSlot = (uint64_t) (seed - m_tbMapBegin) / kTbSlot;
                const uint64_t limit = std::min<uint64_t>(slots, kTbMaxScanSlots);
                for (uint64_t k = 1; k <= limit && found == nullptr; ++k)
                {
                    const uint64_t slot = (seedSlot + k) % slots;
                    const uint8_t* block = m_tbMapBegin + slot * kTbSlot;
                    if (block + kTbBytes > m_tbMapBegin + m_tbMapSize)
                        continue;
                    ++scanned;
                    float q[4];
                    std::memcpy(q, block + kTbViewport, sizeof(q));
                    if (std::memcmp(q, quad, sizeof(q)) != 0)
                        continue;
                    consider(block);
                    if (found != nullptr)
                        slotDelta = (int64_t) k;
                }
            }
        }

        if (info != nullptr)
        {
            info->nearMiss = nearMiss;
            info->scanned = scanned;
            info->slotDelta = slotDelta;
            info->mapBytes = m_tbMapSize;
        }
        if (found == nullptr)
        {
            ++m_tbMissStreak;
            return TbResult::Unverified;
        }
        m_tbMissStreak = 0;

        m_tbLast = found;
        bool known = false;
        for (int k = 0; k < kTbRecent && !known; ++k)
            known = m_tbRecent[k] == found;
        if (!known)
        {
            m_tbRecent[m_tbRecentNext] = found;
            m_tbRecentNext = (m_tbRecentNext + 1) % kTbRecent;
        }
        residualOf(found); // leaves the block's matrix in `game`
        if (info != nullptr)
        {
            info->residual = residual;
            info->otherDepthCol = game[2];
        }
        // The same-eye previous state is PatchPerView's: what the struct before this one carried at
        // +384/+576, valid only if this frame's struct chained to it.
        if (!haveSame)
            return TbResult::NoChain;

        std::memcpy(sameOut, sameRows, 16 * sizeof(float));
        // Rows x, y, w are what the shader loads; the z row keeps the game's own, so anything else that
        // reads it sees what it always saw.
        std::memcpy(sameOut + 8, game + 8, 4 * sizeof(float));
        if (!write)
            return TbResult::VerifiedOnly;

        std::memcpy(const_cast<uint8_t*>(found) + kTbMatrix, sameOut, 16 * sizeof(float));
        m_tbWritten = found;
        std::memcpy(m_tbWrittenMatrix, sameOut, sizeof(m_tbWrittenMatrix));
        return TbResult::Patched;
    }

    // GUARDRAIL -- the second place this module writes into memory the game owns, and the one that matters:
    // the geometry passes bind the per-view struct itself, and its previous view (+384 view-rot-projection,
    // +576 camera) is what their velocity output is computed against. The game fills it with the previous
    // PRESENT -- under alternate-eye rendering the other eye -- so every object vector in the G-buffer carries
    // ~50 px of parallax, where the 8-bit curve's step is ~2.7 px. Rewritten to the same eye's previous state
    // those vectors sit near zero, where the step is a tenth of a pixel.
    //
    // Called before every ExecuteCommandLists: the game must have written the struct before it submits anything
    // that reads it, so the first submit after a rewrite is early enough whatever order Dunia records in. The
    // motion pass's Dispatch is NOT -- the G-buffer lists may be on the GPU by then.
    //
    // Same rules as PatchTemporal: everything under m_lock, try_lock only, pointers never leave. It writes only
    // into a struct that (a) the game has just rewritten -- its +384 is no longer what we wrote, or its
    // signature moved -- and (b) CHAINS: its +384/+576 equal the unjittered P*ViewRot and camera of the struct
    // rewritten before it (offline gate 2: 6e-8, 0). The same-eye state is then that previous struct's own
    // ORIGINAL +384/+576. A torn read mid-write heals itself: the game's write lands on top of ours and the
    // next submit sees a struct that is not what we wrote.
    //
    // Everything of ours that reads +384/+576 goes through ReadPrev, which hands back the saved originals.
    PvResult PatchPerView(bool write, bool mirror, bool ownJitter, float jitterScale, PvInfo* info)
    {
        std::unique_lock<std::mutex> lock(m_lock, std::try_to_lock);
        if (!lock.owns_lock())
            return PvResult::Busy;
        return PatchPerViewLocked(write, mirror, ownJitter, jitterScale, info);
    }

    // The body, for a caller that already holds m_lock: PatchPerView above, and Update.
    PvResult PatchPerViewLocked(bool write, bool mirror, bool ownJitter, float jitterScale, PvInfo* info)
    {
        m_pvOwnJitter = ownJitter;
        ++m_pvCallsWindow;
        if (m_count == 0)
            return PvResult::NoView;

        PvResult result = PvResult::Idle;
        for (size_t i = 0; i < m_count; ++i)
        {
            const uint8_t* base = m_candidates[i] - kVcViewport;
            float sig[7], prevNow[16];
            std::memcpy(sig, base + kVcCamPos, 3 * sizeof(float));
            std::memcpy(sig + 3, base + kVcVrp, 4 * sizeof(float));
            std::memcpy(prevNow, base + kVcPrevVrp, sizeof(prevNow));

            // The signature is stored AFTER anything this function writes (the jitter patch changes +320, which
            // is in it), so a change always means the game wrote the struct again.
            bool fresh = m_pvSigValid[i] && std::memcmp(sig, m_pvSig[i], sizeof(sig)) != 0;
            if (m_pvWrote[i] && std::memcmp(prevNow, m_pvWrittenVrp[i], sizeof(prevNow)) != 0)
                fresh = true;
            std::memcpy(m_pvSig[i], sig, sizeof(sig));
            m_pvSigValid[i] = true;
            if (!fresh)
                continue;
            m_pvWrote[i] = false;

            float proj[16], viewM[16], cam[3], prevCam[3];
            std::memcpy(proj, base + kVcProjection, sizeof(proj));
            std::memcpy(viewM, base + kVcView, sizeof(viewM));
            std::memcpy(cam, base + kVcCamPos, sizeof(cam));
            std::memcpy(prevCam, base + kVcPrevCamPos, sizeof(prevCam));
            // A copy the game built from patched memory carries our old same-eye view as its "previous view":
            // put the original back before anything below reads it, and count it.
            if (UnpatchPrev(prevNow, prevCam))
                g_pvUnpatched.fetch_add(1, std::memory_order_relaxed);
            // By their bits, not by comparing: see FiniteBits. A struct that is not all numbers is not chained,
            // not recorded and not written to.
            if (!FiniteBits(proj, 16) || !FiniteBits(viewM, 16) || !FiniteBits(cam, 3) || !FiniteBits(prevCam, 3) ||
                !FiniteBits(prevNow, 16))
                continue;
            // A rewrite whose viewport is not a viewport: this RING slot holds something else right now (the
            // runs at 50%: viewport 10x0, jitter of thousands of pixels). Not a dead address -- an
            // earlier build dropped the whole set for it five times in 45 s, re-found the same slots each time
            // and paid a second of motion per drop. Count it, skip it, and let the resweep in Update find the
            // slot the set is missing while motion keeps running; the count picks the eviction victim too.
            // And a viewport that is not the SCENE's: the slot now holds another view's struct -- one
            // run showed struct 19 at 256x256 and 127.5x126.5 after ~2 minutes, and from then on "patched
            // although +704 disagreed" climbed to 50 per window. Such a view with no jitter of its own passed
            // every test below, got OUR jitter written into its projection, and advanced the eye's jitter sequence.
            // Likely what baked a crawl into one road texture ("quicksand", reset by a menu, back later, F4 made no
            // difference). The main view and the weapon view both carry the scene viewport the sweep matched.
            {
                float whNow[2];
                std::memcpy(whNow, base + kVcViewport, sizeof(whNow));
                const bool isViewport = FiniteBits(whNow, 2) && whNow[0] > 16.0f && whNow[1] > 16.0f;
                const bool isScene = isViewport && m_forW != 0 && std::abs(whNow[0] - (float) m_forW) <= 1.0f &&
                                     std::abs(whNow[1] - (float) m_forH) <= 1.0f;
                if (isScene)
                    m_pvDead[i] = 0;
                else
                {
                    ++m_pvDead[i];
                    if (isViewport)
                    {
                        g_pvOtherView.fetch_add(1, std::memory_order_relaxed);
                        static int loggedOther = 0;
                        if (loggedOther < 8)
                        {
                            ++loggedOther;
                            LOG_INFO("DuniaTaa: view constants: struct {} of the candidates now has a {}x{} viewport, "
                                     "not the scene's {}x{} -- another view reuses the slot; skipped (no jitter, no "
                                     "chain), counted per window",
                                     i, whNow[0], whNow[1], m_forW, m_forH);
                        }
                    }
                    continue;
                }
            }
            double p[16], rot[16] {}, cur[16];
            for (int k = 0; k < 16; ++k)
                p[k] = proj[k];
            p[2] = 0.0;
            p[6] = 0.0;
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 3; ++c)
                    rot[r * 4 + c] = viewM[r * 4 + c];
            rot[15] = 1.0;
            Mul4(p, rot, cur);

            // Which recently rewritten struct is this one's previous frame? A small table rather than "the
            // last one": the game keeps copies of a frame's struct, and may well rewrite other eye-sized
            // views (a reflection, a second camera) between two frames of the main one. With a single slot,
            // any of those would break the chain for ever and the patch would never happen.
            int match = -1, same = -1;
            double residual = 1e30;
            for (int e = 0; e < kPvRing; ++e)
            {
                if (!m_pvRing[e].valid)
                    continue;
                double dPrev = 0.0, dCur = 0.0;
                for (int k = 0; k < 16; ++k)
                {
                    dPrev = std::max(dPrev, std::abs((double) prevNow[k] - m_pvRing[e].cur[k]));
                    dCur = std::max(dCur, std::abs(cur[k] - m_pvRing[e].cur[k]));
                }
                bool camPrev = true, camCur = true;
                for (int k = 0; k < 3; ++k)
                {
                    camPrev = camPrev && std::abs(prevCam[k] - m_pvRing[e].cam[k]) < 1e-4f;
                    camCur = camCur && cam[k] == m_pvRing[e].cam[k];
                }
                if (camPrev && dPrev < 1e-5 && dPrev < residual)
                {
                    match = e;
                    residual = dPrev;
                }
                if (camCur && dCur < 1e-7)
                    same = e; // a copy of a frame already in the table
            }
            // Taken out of the table BEFORE this struct goes in: the slot it lands on may be the match.
            m_pvSameValid = match >= 0;
            if (m_pvSameValid)
            {
                std::memcpy(m_pvSameVrp, m_pvRing[match].prevVrp, sizeof(m_pvSameVrp));
                std::memcpy(m_pvSameCam, m_pvRing[match].prevCam, sizeof(m_pvSameCam));
                m_pvLastResidual = residual;
            }
            const int entry = same >= 0 ? same : m_pvRingNext;
            if (same < 0)
            {
                PvState& slot = m_pvRing[m_pvRingNext];
                slot.jitterSet = false;
                slot.copies = 0;
                m_pvRingNext = (m_pvRingNext + 1) % kPvRing;
                std::memcpy(slot.cur, cur, sizeof(cur));
                std::memcpy(slot.cam, cam, sizeof(cam));
                std::memcpy(slot.prevVrp, prevNow, sizeof(prevNow));
                std::memcpy(slot.prevCam, prevCam, sizeof(prevCam));
                slot.valid = true;
            }

            m_pvCurIndex = (int) i;



            // ---- Own per-eye jitter. Independent of the chain: it needs only a struct the game has just
            // written, and one that proves the layout first -- the pixel jitter at +704 must be what the
            // projection's own jitter terms say it is, to a hundredth of a pixel, and within half a pixel.
            if (ownJitter)
            {
                float wh[2], pxNow[2];
                std::memcpy(wh, base + kVcViewport, sizeof(wh));
                std::memcpy(pxNow, base + kVcJitterPx, sizeof(pxNow));
                const float gameX = -proj[2] * wh[0] * 0.5f;
                const float gameY = proj[6] * wh[1] * 0.5f;
                // The layout proof -- +704 is the projection's own jitter in pixels -- has to hold ONCE. After
                // that it must not be a precondition of every write: after a MENU the game builds some copies
                // of a frame's struct from memory we already patched, so +704 holds our value while the
                // projection holds the game's (or the other way round), and such a copy was refused -- the
                // log of that run: ~1750 patched per 600 Presents and a dozen refused before the menu, 1200
                // patched and 140-300 refused after it. The refused copy kept the game's +-0.25 px while DLSS
                // was told ours: the per-eye 0.5 px shift, the halos and the black hand silhouettes
                // seen only after a menu, in the old baseline and the new alike. The shift below is computed
                // from the projection itself, so it is right whatever +704 says; +704 is simply overwritten.
                // Up to 1.5 px, not 0.5: after a menu the game builds some copies from patched memory and puts
                // its own +-0.25 on top of OUR jitter, so the projection can hold up to 0.75 px -- and those
                // copies were refused (250-770 per window in one run, while the view was broken). The
                // shift below sets the projection to ours whatever it held, so a wider gate costs nothing.
                const bool sane = FiniteBits(pxNow, 2) && FiniteBits(wh, 2) && wh[0] > 16.0f && wh[1] > 16.0f &&
                                  std::abs(gameX) <= 1.5f && std::abs(gameY) <= 1.5f;
                const bool agrees = sane && std::abs(gameX - pxNow[0]) < 0.01f && std::abs(gameY - pxNow[1]) < 0.01f;
                if (agrees)
                    m_pvJitterLayoutProven = true;
                else if (sane && m_pvJitterLayoutProven)
                    g_jitterDisagreed.fetch_add(1, std::memory_order_relaxed);
                const bool layoutOk = sane && m_pvJitterLayoutProven;
                if (!layoutOk)
                {
                    g_jitterUnverified.fetch_add(1, std::memory_order_relaxed);
                    const bool finite = FiniteBits(pxNow, 2) && FiniteBits(wh, 2);
                    const bool viewportOk = finite && wh[0] > 16.0f && wh[1] > 16.0f;
                    const float mag = finite ? std::max(std::abs(gameX), std::abs(gameY)) : 0.0f;
                    if (!finite)
                        g_jitterRefNotFinite.fetch_add(1, std::memory_order_relaxed);
                    else if (!viewportOk)
                        g_jitterRefViewport.fetch_add(1, std::memory_order_relaxed);
                    else if (mag > 1.5f)
                        g_jitterRefTooLarge.fetch_add(1, std::memory_order_relaxed);
                    else
                        g_jitterRefUnproven.fetch_add(1, std::memory_order_relaxed);
                    float seen = g_jitterRefMaxPx.load(std::memory_order_relaxed);
                    while (mag > seen && !g_jitterRefMaxPx.compare_exchange_weak(seen, mag, std::memory_order_relaxed))
                    {
                    }
                    if (g_jitterRefLogged.load(std::memory_order_relaxed) < 6 &&
                        g_jitterRefLogged.fetch_add(1, std::memory_order_relaxed) < 6)
                    {
                        LOG_INFO("DuniaTaa: own jitter: copy REFUSED -- struct {} of the candidates, viewport {}x{}, the "
                                 "projection's jitter ({:+.3f}, {:+.3f}) px, +704 holds ({:+.3f}, {:+.3f}) px; finite {}, "
                                 "viewport ok {}, within 1.5 px {}, layout proven {}. Our last jitter was ({:+.3f}, {:+.3f})",
                                 i, wh[0], wh[1], gameX, gameY, pxNow[0], pxNow[1], finite, viewportOk, mag <= 1.5f,
                                 m_pvJitterLayoutProven, m_pvJitterVal[0], m_pvJitterVal[1]);
                    }
                }
                else
                {
                    // Copies of one frame's struct take one frame's jitter; a new frame takes the next sample.
                    // Frames alternate eyes, so the sequence advances every second one: each eye walks all of
                    // it, which is the whole point.
                    PvState& e = m_pvRing[entry];
                    // One jitter per frame, keyed by the camera position, which every view of the frame shares
                    // (the weapon view has its own projection but the same camera). Two frames of one eye with
                    // the head perfectly still would share a value; that is only lost sample diversity.
                    if (!m_pvJitterCamValid || std::memcmp(m_pvJitterCam, cam, sizeof(cam)) != 0)
                    {
                        const int phases = std::max(1, g_jitterPhases.load(std::memory_order_relaxed));
                        const int index = (int) ((m_pvJitterFrames++ >> 1) % (uint64_t) phases) + 1;
                        m_pvJitterVal[0] = Halton01(index, 2) - 0.5f;
                        m_pvJitterVal[1] = Halton01(index, 3) - 0.5f;
                        std::memcpy(m_pvJitterCam, cam, sizeof(cam));
                        m_pvJitterCamValid = true;
                    }
                    if (!e.jitterSet)
                    {
                        std::memcpy(e.jitter, m_pvJitterVal, sizeof(e.jitter));
                        e.jitterSet = true;
                    }
                    const float ours[2] = { e.jitter[0] * jitterScale, e.jitter[1] * jitterScale };
                    const float tx = 2.0f * (ours[0] - gameX) / wh[0];
                    const float ty = -2.0f * (ours[1] - gameY) / wh[1];
                    uint8_t* dest = const_cast<uint8_t*>(base);
                    float m[16];
                    std::memcpy(m, base, sizeof(m)); // +0, the inverse projection
                    if (FiniteBits(m, 16))
                    {
                        ShiftInverse(m, tx, ty);
                        std::memcpy(dest, m, sizeof(m));
                    }
                    for (const size_t at : { kVcProjection, kVcViewProjection, kVcVrp })
                    {
                        std::memcpy(m, base + at, sizeof(m));
                        if (!FiniteBits(m, 16))
                            continue;
                        ShiftClip(m, tx, ty);
                        std::memcpy(dest + at, m, sizeof(m));
                    }
                    std::memcpy(dest + kVcJitterPx, ours, sizeof(ours));
                    g_jitterPatched.fetch_add(1, std::memory_order_relaxed);
                    ++m_pvPatchedWindow;
                    if (info != nullptr)
                    {
                        info->jitterWritten = true;
                        info->jitterGame[0] = gameX;
                        info->jitterGame[1] = gameY;
                        info->jitterOurs[0] = ours[0];
                        info->jitterOurs[1] = ours[1];
                    }
                    // +320 is in the signature: store it as it now stands, or the next look calls this fresh.
                    std::memcpy(m_pvSig[i] + 3, base + kVcVrp, 4 * sizeof(float));
                }
            }

            if (!m_pvSameValid)
            {
                result = std::max(result, PvResult::NoChain);
                continue;
            }
            if (info != nullptr)
            {
                info->chainResidual = m_pvLastResidual;
                double o = 0.0, s = 0.0;
                for (int k = 0; k < 3; ++k)
                {
                    o += ((double) cam[k] - prevCam[k]) * ((double) cam[k] - prevCam[k]);
                    s += ((double) cam[k] - m_pvSameCam[k]) * ((double) cam[k] - m_pvSameCam[k]);
                }
                info->otherEyeStep = std::sqrt(o);
                info->sameEyeStep = std::sqrt(s);
            }
            // Which of the frame's copies this is, in the order the game wrote them (as far as the submits and
            // the splice let us see). The game writes THREE a frame, and different passes may well read
            // different ones: the geometry passes need the same-eye previous view, while the game's own
            // temporal filters (AO, reflections) keep a history that is the previous PRESENT -- the other eye --
            // and flicker when the head moves if they are handed the same eye's. g_pvCopyMask picks which
            // copies are written, so one run can say which copy feeds what.
            const int copyIndex = m_pvRing[entry].copies++;
            m_copyIdx[i] = copyIndex; // for the private-copy feasibility probe: which copy each slot last was
            const int mask = mirror ? 0 : g_pvCopyMask.load(std::memory_order_relaxed);
            const bool copyWanted = mask == 0   ? true
                                    : mask == 1 ? copyIndex == 0
                                    : mask == 2 ? copyIndex == 1
                                    : mask == 3 ? copyIndex >= 2
                                                : false;
            if (copyIndex < 4)
                g_pvCopySeen[copyIndex].fetch_add(1, std::memory_order_relaxed);
            if (!write || !copyWanted)
            {
                result = std::max(result, PvResult::VerifiedOnly);
                continue;
            }
            // The private copy: the game's struct is left exactly as the game wrote it (its AO and temporal
            // filters keep their other-eye history), and the same-eye state is kept for MirrorFill, which writes
            // it into OUR copy of this ring slot -- the one the G-buffer's rebuilt table points at. Keyed on the
            // signature as it stands now (after the jitter write above), so a later rewrite by the game is copied
            // plain rather than given this frame's previous view.
            if (mirror)
            {
                std::memcpy(m_mirVrp[i], m_pvSameVrp, sizeof(m_pvSameVrp));
                std::memcpy(m_mirCam[i], m_pvSameCam, sizeof(m_pvSameCam));
                std::memcpy(m_mirSig[i], m_pvSig[i], sizeof(m_mirSig[i]));
                m_mirValid[i] = true;
                result = PvResult::Patched;
                continue;
            }

            std::memcpy(m_pvOrigVrp[i], prevNow, sizeof(prevNow));
            std::memcpy(m_pvOrigCam[i], prevCam, sizeof(prevCam));
            uint8_t* dest = const_cast<uint8_t*>(base);
            std::memcpy(dest + kVcPrevVrp, m_pvSameVrp, sizeof(m_pvSameVrp));
            std::memcpy(dest + kVcPrevCamPos, m_pvSameCam, sizeof(m_pvSameCam));
            std::memcpy(m_pvWrittenVrp[i], m_pvSameVrp, sizeof(m_pvSameVrp));
            {
                bool known = false;
                for (int k = 0; k < kPvWrittenRing && !known; ++k)
                    known = m_pvWritten[k].valid && std::memcmp(m_pvWritten[k].vrp, m_pvSameVrp, sizeof(m_pvSameVrp)) == 0;
                if (!known)
                {
                    PvWritten& w = m_pvWritten[m_pvWrittenNext];
                    m_pvWrittenNext = (m_pvWrittenNext + 1) % kPvWrittenRing;
                    std::memcpy(w.vrp, m_pvSameVrp, sizeof(w.vrp));
                    std::memcpy(w.origVrp, prevNow, sizeof(w.origVrp));
                    std::memcpy(w.origCam, prevCam, sizeof(w.origCam));
                    w.valid = true;
                }
            }
            m_pvWrote[i] = true;
            result = PvResult::Patched;
        }
        return result;
    }

    // The private copy (overlay "Same-eye NPC motion"): every candidate struct, copied into OUR mirror of its
    // ring slot, with the same-eye previous view where PatchPerView computed one for exactly these bytes. Called
    // at every submit, after PatchPerView. Blocking lock: a skipped fill would leave the G-buffer reading a stale
    // copy of the CURRENT view, not merely an old previous one. The game's bytes are read under m_lock like
    // everything else here and never written; only our buffer is (PvMirror, its own lock, taken after m_lock is
    // released). A struct whose signature and same-eye state are unchanged since its last fill in this Present
    // is not read again: the ring is write-combined upload memory, slow to read.
    int MirrorFill(uint64_t presentIndex)
    {
        std::vector<PvMirrorEntry> entries;
        uint64_t ringSize = 0, ringGpu = 0;
        {
            std::lock_guard<std::mutex> lock(m_lock);
            const uint8_t* ringBegin = nullptr;
            entries.reserve(m_count);
            for (size_t i = 0; i < m_count; ++i)
            {
                if (m_candidates[i] == nullptr)
                    continue;
                const uint8_t* base = m_candidates[i] - kVcViewport;
                const uint8_t* begin = nullptr;
                uint64_t size = 0;
                if (!ConstantCapture_Dx12::MappingOf(base, &begin, &size) || base + kPvMirrorStruct > begin + size)
                    continue;
                if (ringBegin == nullptr)
                {
                    if (!ConstantCapture_Dx12::GpuAddressOf(begin, &ringGpu))
                        return 0;
                    ringBegin = begin;
                    ringSize = size;
                }
                else if (begin != ringBegin)
                    continue; // one ring only; a copy elsewhere keeps the game's own table
                float sig[7];
                std::memcpy(sig, base + kVcCamPos, 3 * sizeof(float));
                std::memcpy(sig + 3, base + kVcVrp, 4 * sizeof(float));
                const bool same = m_mirValid[i] && std::memcmp(sig, m_mirSig[i], sizeof(sig)) == 0;
                PvMirrorEntry e;
                e.offset = (uint64_t) (base - begin);
                e.sameEye = same;
                e.haveBytes = !(m_mirFilledBase[i] == base && m_mirFilledPresent[i] == presentIndex &&
                                m_mirFilledSame[i] == same && std::memcmp(sig, m_mirFilledSig[i], sizeof(sig)) == 0);
                if (e.haveBytes)
                {
                    std::memcpy(e.bytes, base, kPvMirrorStruct);
                    if (same)
                    {
                        std::memcpy(e.bytes + kVcPrevVrp, m_mirVrp[i], sizeof(m_mirVrp[i]));
                        std::memcpy(e.bytes + kVcPrevCamPos, m_mirCam[i], sizeof(m_mirCam[i]));
                    }
                    m_mirFilledBase[i] = base;
                    m_mirFilledPresent[i] = presentIndex;
                    m_mirFilledSame[i] = same;
                    std::memcpy(m_mirFilledSig[i], sig, sizeof(sig));
                }
                entries.push_back(e);
            }
        }
        if (entries.empty())
            return 0;
        return g_pvMirror.Fill(entries, ringSize, ringGpu, presentIndex);
    }

    size_t TemporalBlockCount()
    {
        std::lock_guard<std::mutex> lock(m_lock);
        return m_tbCount;
    }

  private:
    // The game's own previous view for candidate i. Call with m_lock held. While +384 still holds what
    // PatchPerView wrote, the saved originals; otherwise the bytes themselves.
    void ReadPrev(size_t i, float vrp[16], float cam[3]) const
    {
        const uint8_t* base = m_candidates[i] - kVcViewport;
        std::memcpy(vrp, base + kVcPrevVrp, 16 * sizeof(float));
        std::memcpy(cam, base + kVcPrevCamPos, 3 * sizeof(float));
        UnpatchPrev(vrp, cam);
    }

    // If `vrp` is a previous view WE wrote (into any struct, any time recently), replace it and the camera with
    // the game's original. Values, not pointers: it survives Forget, which is the point -- after a menu (or a
    // death) the game builds some copies of the struct from memory we had patched, and a copy whose +384 is our
    // OLD same-eye value looked like a fresh struct with a strange previous view: the chain failed (no patch)
    // and the other-eye baseline was built from a same-eye view (full parallax in the residual: the halos,
    // hand silhouettes and per-eye edge shift seen in the same-eye state after menus).
    bool UnpatchPrev(float vrp[16], float cam[3]) const
    {
        for (int k = 0; k < kPvWrittenRing; ++k)
        {
            const PvWritten& w = m_pvWritten[k];
            if (w.valid && std::memcmp(vrp, w.vrp, 16 * sizeof(float)) == 0)
            {
                std::memcpy(vrp, w.origVrp, 16 * sizeof(float));
                std::memcpy(cam, w.origCam, 3 * sizeof(float));
                return true;
            }
        }
        return false;
    }

    struct PvWritten
    {
        float vrp[16] {};
        float origVrp[16] {};
        float origCam[3] {};
        bool valid = false;
    };
    static constexpr int kPvWrittenRing = 16;
    PvWritten m_pvWritten[kPvWrittenRing];
    int m_pvWrittenNext = 0;

    // Per candidate: PatchPerView's own signature (Update's m_sig is its own business), what was written and
    // what it replaced. Indices are stable: candidates are only appended, or all dropped by Forget.
    float m_pvSig[kVcMaxCandidates][7] {};
    bool m_pvSigValid[kVcMaxCandidates] {};
    bool m_pvWrote[kVcMaxCandidates] {};
    // Consecutive rewrites of a candidate whose viewport no longer reads as one. The game rings these structs
    // through one allocation scheme, and after a level change or a Resolution Scale change a cached address
    // can go on being written -- with something else. The set then patches two of the frame's three copies
    // and the third renders with the game's jitter: a run at 50% (300-600 refused per window,
    // edges shimmering). A set with a dead member is dropped and rescanned, as when a buffer is released.
    int m_pvDead[kVcMaxCandidates] {};
    // The resweep: with own jitter on, the game writes ~3 copies of the frame's struct and every one has to
    // be patched, or the unpatched one renders with the game's jitter (edge shimmer). When the copies patched
    // per Present fall below 2.5 over a 120-Present window, a sweep runs -- candidates stay in use -- and
    // stops as soon as it has added a slot. Counted per window; the candidates already held are kept.
    bool m_pvOwnJitter = false;
    uint64_t m_pvPatchedWindow = 0;
    uint64_t m_pvCallsWindow = 0;
    uint64_t m_pvWindowStart = 0;
    uint64_t m_lastResweep = 0;
    uint64_t m_resweeps = 0;
    bool m_resweep = false;
    uint64_t m_ringProbeAdds = 0;
    float m_pvWrittenVrp[kVcMaxCandidates][16] {};
    float m_pvOrigVrp[kVcMaxCandidates][16] {};
    float m_pvOrigCam[kVcMaxCandidates][3] {};
    // The chain, across structs: the last rewritten struct's unjittered P*ViewRot and camera, and its own
    // ORIGINAL +384/+576 -- which is the same-eye previous state of the struct that follows it.
    uint64_t m_pvJitterFrames = 0;
    bool m_pvJitterLayoutProven = false;
    // The main view's projection scale, to tell it from the weapon / ADS view (another FOV) when both
    // structs change in one Present.
    float m_mainP00 = 0.0f, m_mainP11 = 0.0f;
    bool m_mainFovValid = false;
    bool m_chosenMainFov = false;
    int m_chosenRank = -1;
    // The jitter is per FRAME, not per struct: the weapon view shares the frame's camera position with the
    // main view and must render with the same sub-pixel offset, or the gun and the world jitter apart.
    float m_pvJitterCam[3] {};
    float m_pvJitterVal[2] {};
    bool m_pvJitterCamValid = false;
    struct PvState
    {
        float jitter[2] {};
        bool jitterSet = false;
        int copies = 0;
        double cur[16] {};
        float cam[3] {};
        float prevVrp[16] {};
        float prevCam[3] {};
        bool valid = false;
    };
    static constexpr int kPvRing = 6;
    PvState m_pvRing[kPvRing];
    int m_pvRingNext = 0;
    bool m_pvSameValid = false;
    float m_pvSameVrp[16] {};
    float m_pvSameCam[3] {};
    // The private copy, per candidate: the same-eye previous state computed for the struct whose signature was
    // m_mirSig (PatchPerView, mirror mode), and what MirrorFill last copied into the mirror.
    float m_mirVrp[kVcMaxCandidates][16] {};
    float m_mirCam[kVcMaxCandidates][3] {};
    float m_mirSig[kVcMaxCandidates][7] {};
    bool m_mirValid[kVcMaxCandidates] {};
    const uint8_t* m_mirFilledBase[kVcMaxCandidates] {}; // compared, never read through
    uint64_t m_mirFilledPresent[kVcMaxCandidates] {};
    float m_mirFilledSig[kVcMaxCandidates][7] {};
    bool m_mirFilledSame[kVcMaxCandidates] {};
    double m_pvLastResidual = 0.0;

    // Float noise on this comparison measured 1.2e-7 .. 2.8e-7 offline; leaving the jitter in is 1.5e-4.
    static constexpr double kTbVerifyTolerance = 5e-6;
    const uint8_t* m_tb[kTbMaxCandidates] {};
    size_t m_tbCount = 0;
    const uint8_t* m_tbWritten = nullptr; // read only under m_lock; dropped with its buffer by ForgetIfWithin
    float m_tbWrittenMatrix[16] {};
    // Where the block was last found (the scan starts just after it), and the pool it lives in. All three
    // are dropped by ForgetIfWithin when that buffer dies.
    const uint8_t* m_tbLast = nullptr;
    static constexpr int kTbRecent = 4;
    const uint8_t* m_tbRecent[kTbRecent] {};
    int m_tbRecentNext = 0;
    uint64_t m_tbMissStreak = 0;
    const uint8_t* m_tbMapBegin = nullptr;
    uint64_t m_tbMapSize = 0;
    static constexpr uint64_t kTbSlot = 512;
    static constexpr uint64_t kTbMaxScanSlots = 16384; // 8 MB of pool: one 16-byte read each, worst case
    // The candidate PatchPerView last saw the game rewrite: this frame's per-view struct. -1 until then.
    int m_pvCurIndex = -1;
    mutable std::mutex m_lock;
    const uint8_t* m_candidates[kVcMaxCandidates] {};
    // The copy index (write order within its frame) each candidate slot last held, -1 if never seen as one.
    int m_copyIdx[kVcMaxCandidates] = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 };
    size_t m_count = 0;
    UINT m_forW = 0;
    UINT m_forH = 0;
    float m_sig[kVcMaxCandidates][7] {};
    bool m_sigValid[kVcMaxCandidates] {};
    uint64_t m_lastChange = 0;
    uint64_t m_retryPresents = kVcRescanPresents;
    // The sweep in progress, if any, and what it has found so far. Committed to m_candidates only when done.
    bool m_scanning = false;
    size_t m_committed = 0;       // how many of m_pending have already been copied into m_candidates
    uint64_t m_writtenAdded = 0;
    bool m_writtenLogged = false;
    uint64_t m_scanDropped = 0;
    bool m_scanStallLogged = false;
    ConstantCapture_Dx12::PatternScan m_scan;
    const uint8_t* m_pending[kVcMaxCandidates] {};
    ViewState m_history[2];
};
ViewConstants g_viewConstants;

// F4's states. Chained is NOT in the default cycle: it was rejected in both games -- smear and crawling
// textures, worse than Days Gone -- and leaving it one keypress away meant it kept being what was
// actually being looked at. It stays reachable behind [Inputs] DuniaTaaChainedMotion so the evidence is not deleted.
constexpr int kMotionOff = 0;
constexpr int kMotionGame = 1;
constexpr int kMotionMatrix = 2;
// Matrix motion plus the game's own per-object motion carried over onto it. Its own F4 state so the A/B
// against plain matrix motion is one keypress inside one run.
constexpr int kMotionMatrixObjects = 3;
constexpr int kMotionChained = 4;
bool MotionModeIsMatrix()
{
    const int mode = g_realMotionMode.load(std::memory_order_relaxed);
    return mode == kMotionMatrix || mode == kMotionMatrixObjects;
}
bool MotionObjectsOn() { return g_realMotionMode.load(std::memory_order_relaxed) == kMotionMatrixObjects; }
std::atomic<uint64_t> g_mvObjectDecodes { 0 };

// Same-eye vectors from the game's own pass ([Inputs] DuniaTaaSameEyeVectors, and its own F4 state). With the
// ini key on, the motion pass's block is found and VERIFIED in every state, so the log says whether the patch
// could work before it is ever switched on; the block is written only in the F4 state.
bool SameEyeWanted() { return Config::Instance()->DuniaTaaSameEyeVectors.value_or_default(); }
std::atomic<bool> g_sameEyePatchOn { false };
// The motion pass's own matrix as well, so the vectors it COMPUTES are same-eye too. MEASURED AND REJECTED
// (with the patch really running: 600/600 per window, matched to 1.3e-7, parallax
// term 0.578 -> -0.00008): no visible gain on NPC skin or motion -- the computed pixels are camera motion, which
// our own matrix already gives exactly -- and it ADDS an ambient-occlusion / shadow flicker in places. The
// block is the motion pass's constant buffer: the game's other temporal filters read the same matrix, and their
// history is the previous PRESENT, the other eye, not this eye's. So it is off in everyday play, costs nothing
// there (the block is not even looked for), and stays reachable as a diagnostics-only F4 state so the
// evidence is not deleted. Do not "finish" it.
std::atomic<bool> g_sameEyeBlockOn { false };
// Three A/Bs, each one press inside one run, settled what the patched state is:
//   object residual x2, not x1   -- the G-buffer's object motion covers one Present, the baseline two
//   knee floor 0.08 px, not 0.25 -- better for NPCs
//   the half-res field SMOOTHED (guarded bilinear among same-eye neighbours), not the nearest cell --
//     faces and skin better, and no quicksand once the blend moved into the same-eye branch
// So there is one patched state, and in everyday play it IS F4's on state.
constexpr float kToleranceFloorSameEyeFine = 0.08f;
std::atomic<uint64_t> g_pvCounts[(size_t) PvResult::Count] {};
// Describes the staging copy AfterDispatch has just committed: were those vectors computed with our matrix?
// Set before CommitCopy marks the copy fresh and read where TakeFresh consumes it, so it travels with the
// copy it describes rather than with a Present index two threads would have to agree on.
std::atomic<bool> g_mvCopySameEye { false };
std::atomic<uint64_t> g_tbCounts[(size_t) TbResult::Count] {};
// What finding the moving block costs: calls that had to walk the pool, and slots read, per stats window.
std::atomic<uint64_t> g_tbScans { 0 }, g_tbScannedSlots { 0 };
std::atomic<uint64_t> g_mvSameEyeDecodes { 0 };
// What the stats window that F4 is closing was IN, stored by the toggle before it flips anything. The old
// line inferred it by negating the new state, which is wrong for a press that changes only the patch.
std::atomic<bool> g_mvWindowWasOn { false };
std::atomic<bool> g_sameEyeWindowWas { false };
// The knee's floor, in pixels. 1.0 was chosen for OTHER-eye vectors, where the baseline is ~50 px of
// parallax and the 8-bit curve's step there is ~2.7 px. Same-eye vectors sit near zero, where the step is
// 0.137 * |v|^0.75 px (0.08 px at half a pixel), so the floor can come down to where a 2 px skin movement
// is all signal.
constexpr float kToleranceFloorSameEye = 0.25f;

// From BeforeExecute, on whichever thread submits. Counted per stats window; each outcome explained once,
// except Idle, which is what nearly every submit is.
PvArgs CurrentPvArgs()
{
    PvArgs args;
    // The private copy takes precedence: with it on, nothing is written into the game's struct.
    args.mirror = g_pvMirrorOn.load(std::memory_order_relaxed);
    args.write = g_sameEyePatchOn.load(std::memory_order_relaxed) || args.mirror;
    // Our jitter only where our motion is: the matrix modes keep their vectors jitter-free whatever the jitter
    // does from frame to frame; the game's-own and chained modes do not.
    const int jitterMode = JitterModeNow();
    args.ownJitter = jitterMode >= 2 && MotionModeIsMatrix();
    args.jitterScale = jitterMode == 3 ? 20.0f : jitterMode == 4 ? 4.0f : 1.0f;
    return args;
}

void NotePerViewResult(PvResult result, const PvInfo& info);

void PatchPerViewPrevious()
{
    PvInfo info;
    const PvArgs args = CurrentPvArgs();
    const PvResult result =
        g_viewConstants.PatchPerView(args.write, args.mirror, args.ownJitter, args.jitterScale, &info);
    NotePerViewResult(result, info);
}

void NotePerViewResult(PvResult result, const PvInfo& info)
{
    g_pvCounts[(size_t) result].fetch_add(1, std::memory_order_relaxed);
    if (info.jitterWritten)
    {
        uint32_t bits;
        std::memcpy(&bits, &info.jitterOurs[0], sizeof(bits));
        if (g_jitterLastX[0].load(std::memory_order_relaxed) != bits)
        {
            g_jitterLastX[1].store(g_jitterLastX[0].load(std::memory_order_relaxed), std::memory_order_relaxed);
            g_jitterLastX[0].store(bits, std::memory_order_relaxed);
            g_jitterNewFrames.fetch_add(1, std::memory_order_relaxed);
        }
        static std::atomic<bool> loggedJitter { false };
        if (!loggedJitter.exchange(true, std::memory_order_relaxed))
            LOG_INFO("DuniaTaa: own jitter: PATCHED into the per-view struct before a submit -- the game's ({:+.3f}, "
                     "{:+.3f}) px became ours ({:+.3f}, {:+.3f}) px; +704 agreed with the projection's own jitter "
                     "terms, which is what proves the layout. {} phases per eye",
                     info.jitterGame[0], info.jitterGame[1], info.jitterOurs[0], info.jitterOurs[1],
                     g_jitterPhases.load());
    }
    if (result == PvResult::Idle)
        return;

    static std::atomic<bool> logged[(size_t) PvResult::Count] {};
    if (logged[(size_t) result].exchange(true, std::memory_order_relaxed))
        return;
    switch (result)
    {
    case PvResult::Patched:
        LOG_INFO("DuniaTaa: same-eye vectors: PATCHED the per-view struct's previous view (+384/+576) before a "
                 "submit. It chained to the struct before it to {:.3g} (offline: 6e-8); the camera step to "
                 "the game's previous view was {:.4f} (about the IPD, ~0.06) and to the one written {:.4f} "
                 "(head motion over two Presents). If NPC vectors do not change in the motion view, the "
                 "velocity shaders read their previous view from somewhere else",
                 info.chainResidual, info.otherEyeStep, info.sameEyeStep);
        break;
    case PvResult::VerifiedOnly:
        LOG_INFO("DuniaTaa: same-eye vectors: a rewritten per-view struct was caught before a submit and "
                 "CHAINS to the one before it ({:.3g}); camera step to the game's previous view {:.4f}, to "
                 "the same eye's {:.4f}. Not written: the patch is an F4 state",
                 info.chainResidual, info.otherEyeStep, info.sameEyeStep);
        break;
    case PvResult::NoChain:
        LOG_INFO("DuniaTaa: same-eye vectors: a rewritten per-view struct does not follow on from the one "
                 "before it -- expected on the first frames and after a menu; not written. The per-window "
                 "line says how often");
        break;
    case PvResult::NoView:
        LOG_INFO("DuniaTaa: same-eye vectors: submits are being watched but there is no per-view struct yet");
        break;
    case PvResult::Busy:
        LOG_INFO("DuniaTaa: same-eye vectors: a submit found the view-constants lock held (a sweep slice); "
                 "that struct is looked at again at the next submit");
        break;
    default:
        break;
    }
}

// From AfterDispatch, on the game's recording thread. True when the vectors this dispatch will produce are
// same-eye. Every other outcome is counted per stats window and explained once.
bool PatchMotionPassMatrix()
{
    // The block is verified against THIS frame's per-view struct, which is whichever one PatchPerView last saw
    // rewritten. Normally a submit has already looked since the game wrote it; if none has, look now, or the
    // block would be compared with the other eye's frame and fail.
    PatchPerViewPrevious();

    float same[16] {};
    TbInfo info;
    const bool write =
        g_sameEyePatchOn.load(std::memory_order_relaxed) && g_sameEyeBlockOn.load(std::memory_order_relaxed);
    const TbResult result = g_viewConstants.PatchTemporal(write, same, &info);
    g_tbCounts[(size_t) result].fetch_add(1, std::memory_order_relaxed);
    g_tbScannedSlots.fetch_add(info.scanned, std::memory_order_relaxed);
    if (info.scanned != 0)
        g_tbScans.fetch_add(1, std::memory_order_relaxed);

    static std::atomic<bool> logged[(size_t) TbResult::Count] {};
    const bool first = !logged[(size_t) result].exchange(true, std::memory_order_relaxed);
    switch (result)
    {
    case TbResult::Patched:
        if (first)
            LOG_INFO("DuniaTaa: same-eye vectors: PATCHED the motion pass's matrix. The block matched our "
                     "other-eye build to {:.3g} (float noise is ~3e-7, the jitter would be 1.5e-4); its "
                     "parallax term was {:.6f} and is now {:.6f} -- about +-0.58 falling to ~1e-4 is the "
                     "real thing. Found {} slot(s) past the last known one after looking at {} of a {:.1f} MB "
                     "pool; {} seed block(s) known",
                     info.residual, info.otherDepthCol, same[2], info.slotDelta, info.scanned,
                     (double) info.mapBytes / (1024.0 * 1024.0), g_viewConstants.TemporalBlockCount());
        return true;
    case TbResult::AlreadyDone:
        return true;
    case TbResult::VerifiedOnly:
        if (first)
            LOG_INFO("DuniaTaa: same-eye vectors: the motion pass's block is found and VERIFIED (matches our "
                     "other-eye build to {:.3g}; parallax term {:.6f}, same-eye would be {:.6f}). Not "
                     "written: the patch is an F4 state",
                     info.residual, info.otherDepthCol, same[2]);
        return false;
    case TbResult::Unverified:
        if (first)
            LOG_WARN("DuniaTaa: same-eye vectors: no block matches our other-eye build for this frame -- {} "
                     "seed block(s), {} slots of a {:.1f} MB pool looked at, closest {:.3g}, needed < 5e-6. "
                     "~1.5e-4 means a jitter convention, ~0.5-1.2 means only the other eye's or stale blocks "
                     "were seen (the live one is outside the pool searched, or not on a 512-byte slot), "
                     "anything else doubts the derivation. Nothing written",
                     g_viewConstants.TemporalBlockCount(), info.scanned,
                     (double) info.mapBytes / (1024.0 * 1024.0), info.nearMiss);
        return false;
    case TbResult::NoChain:
        if (first)
            LOG_INFO("DuniaTaa: same-eye vectors: block verified ({:.3g}) but the previous Present's state is "
                     "not the one this frame follows on from -- expected on the first frame and after a "
                     "menu; nothing written this frame",
                     info.residual);
        return false;
    case TbResult::NoBlock:
        if (first)
            LOG_INFO("DuniaTaa: same-eye vectors: no motion pass block found yet ({} raw quads have looked "
                     "like one) -- the view-constants sweep finds them as it goes",
                     g_tbObserved.load());
        return false;
    case TbResult::NoView:
        if (first)
            LOG_INFO("DuniaTaa: same-eye vectors: a block is known but no per-view struct yet");
        return false;
    case TbResult::Busy:
        if (first)
            LOG_INFO("DuniaTaa: same-eye vectors: the present thread held the view-constants lock (a sweep "
                     "slice) -- this frame keeps the game's matrix. Counted per window as 'busy'");
        return false;
    default:
        return false;
    }
}

// Builds the matrix the decode shader wants: a projective 4x4 taking (ndc.x, ndc.y, 1/viewZ, 1) to the previous
// SAME-EYE clip position. The derivation, so it can be checked rather than trusted:
//
//   viewZ  = -near / (z_ndc + 1/far)                     so  1/viewZ is AFFINE in z_ndc, which is why the
//                                                        shader's unprojection is Unproject.x*z + Unproject.y
//   view   = viewZ * d,  d = ( -(ndc.x + jx)/P00, -(ndc.y + jy)/P11, 1 )
//   rel    = c2w3 * view + delta,   delta = camPos_now - camPos_twoPresentsAgo
//   clip'  = VRP3 * rel + VRPcol3   (VRP has no translation, so its 4th column carries only the near term)
//          = viewZ * (VRP3 * c2w3 * d) + (VRP3 * delta + VRPcol3)
//
// Dividing by viewZ leaves it linear in (ndc.x, ndc.y, 1/viewZ, 1), which is exactly the vector the existing
// shader already forms. So no new shader: only the unprojection changes shape.
// `prev` is the previous state to reproject to: f.prevSameEye for the motion DLSS is fed, f.prevOtherEye for
// the baseline the game's own vectors were computed against.
void BuildPrevClipMatrix(const ViewFrame& f, const ViewState& prev, float a[16], float unproject[2])
{
    const float p00 = f.proj[0];
    const float p11 = f.proj[5];
    const float jx = f.proj[2];
    const float jy = f.proj[6];
    const float invFar = f.proj[10];
    // Not `near`: windef.h defines that as a macro, which turns this into "const float = ..." and three
    // syntax errors a long way from the cause.
    const float nearZ = f.proj[11];

    unproject[0] = -1.0f / nearZ;
    unproject[1] = -invFar / nearZ;

    // G = VRP3 * c2w3, a 4x3: rows of the previous view-rot-projection against the camera basis.
    float g[4][3] {};
    for (int i = 0; i < 4; ++i)
        for (int k = 0; k < 3; ++k)
            for (int j = 0; j < 3; ++j)
                g[i][k] += prev.vrp[i * 4 + j] * f.c2w[j * 4 + k];

    const float delta[3] = { f.cur.camPos[0] - prev.camPos[0], f.cur.camPos[1] - prev.camPos[1],
                             f.cur.camPos[2] - prev.camPos[2] };

    for (int i = 0; i < 4; ++i)
    {
        const float colNdcX = g[i][0] * (-1.0f / p00);
        const float colNdcY = g[i][1] * (-1.0f / p11);
        const float colOne = g[i][0] * (-jx / p00) + g[i][1] * (-jy / p11) + g[i][2];
        float colInvZ = prev.vrp[i * 4 + 3];
        for (int j = 0; j < 3; ++j)
            colInvZ += prev.vrp[i * 4 + j] * delta[j];
        a[i * 4 + 0] = colNdcX;
        a[i * 4 + 1] = colNdcY;
        a[i * 4 + 2] = colInvZ;
        a[i * 4 + 3] = colOne;
    }

    // `prev` is UNJITTERED (both of them: +384 always was, and the same-eye history is now). The pixel this
    // is evaluated at sits on the JITTERED grid, and a jitter-free motion vector is prevUnjittered -
    // curUnjittered = prevUnjittered - (ndc + jx): so the current jitter goes onto the previous position.
    // The shader computes c.xy / c.w - ndc, hence rows x and y take -jx and -jy times row w. For the same-eye
    // matrix under the game's own jitter this is exactly what using the jittered +320 did; for the other-eye
    // baseline it removes a quarter-pixel error that had been hiding under the 1 px floor.
    for (int c = 0; c < 4; ++c)
    {
        a[0 * 4 + c] += -jx * a[3 * 4 + c];
        a[1 * 4 + c] += -jy * a[3 * 4 + c];
    }
}

// The root parameter the pass binds its UAV table at, learned while searching. -1 until then.
int g_mvRootParameter = -1;
// The Present a motion table was last resolved on, so the rest of that frame's binds cost one compare.
std::atomic<uint64_t> g_mvFoundThisPresent { UINT64_MAX };
// Collected while the search runs, handed to ResTrack when it ends. See NoteComputeTable for why they
// cannot be applied as they are found.
std::vector<std::pair<SIZE_T, SIZE_T>> g_mvHeapRanges;
// Defined further down, next to the rest of the resource helpers.
void RetainDeferred(ID3D12Resource** slot, ID3D12Resource* value, uint64_t presentIndex);
bool CreateTex(ID3D12Device* device, UINT width, UINT height, DXGI_FORMAT format, D3D12_RESOURCE_FLAGS flags,
               D3D12_RESOURCE_STATES state, ID3D12Resource** out);

// GUARDRAIL. The two staging textures live behind locked accessors and the pointers are private, so
// there is no way to read them without the lock -- see `GUARDRAILS` at the top of this file.
//
// Why: the splice used to read the pair unlocked while AfterDispatch, on the game's recording threads,
// could swap or release them. A menu size change does exactly that. Reading a consistent snapshot is not
// something to remember at each call site; it is the only thing the type allows.
//
// Two of them, written alternately: chaining needs the previous Present's vectors as well as this
// Present's, and the decode reads both. Ours, half the eye size, the game's R8G8_UNORM format.
class MvStagingPair
{
  public:
    struct Snapshot
    {
        ID3D12Resource* cur = nullptr;    // what the last copy wrote
        ID3D12Resource* prev = nullptr;   // the Present before it
        bool paired = false;              // both hold a real frame, so chaining has something to chain
    };

    Snapshot Read() const
    {
        std::lock_guard<std::mutex> lock(m_lock);
        return { m_tex[m_cur], m_tex[m_cur ^ 1], m_fills >= 2 };
    }

    // The buffer the next copy should write, creating or resizing as needed. nullptr means do nothing this
    // frame. Must be followed by CommitCopy() once the copy has been recorded onto the list.
    ID3D12Resource* PrepareCopy(ID3D12GraphicsCommandList* cmd, const D3D12_RESOURCE_DESC& desc,
                               uint64_t presentIndex)
    {
        std::lock_guard<std::mutex> lock(m_lock);

        if (m_tex[0] != nullptr && (m_w != (UINT) desc.Width || m_h != desc.Height))
        {
            // Deferred, never immediate: a list recorded moments ago may still reference these.
            RetainDeferred(&m_tex[0], nullptr, presentIndex);
            RetainDeferred(&m_tex[1], nullptr, presentIndex);
            m_w = m_h = 0;
            m_cur = 0;
            m_fills = 0;
            m_fresh.store(false, std::memory_order_relaxed);
        }

        if (m_tex[0] == nullptr || m_tex[1] == nullptr)
        {
            ID3D12Device* device = nullptr;
            if (cmd == nullptr || FAILED(cmd->GetDevice(IID_PPV_ARGS(&device))) || device == nullptr)
                return nullptr;
            device->Release();
            for (int i = 0; i < 2; ++i)
            {
                if (m_tex[i] != nullptr)
                    continue;
                if (!CreateTex(device, (UINT) desc.Width, desc.Height, desc.Format,
                               D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                               &m_tex[i]))
                {
                    if (!m_warned)
                    {
                        m_warned = true;
                        LOG_WARN("DuniaTaa: could not create motion staging texture {} at {}x{} fmt {}", i,
                                 desc.Width, desc.Height, (UINT) desc.Format);
                    }
                    return nullptr;
                }
            }
            m_w = (UINT) desc.Width;
            m_h = desc.Height;
            m_cur = 0;
            m_fills = 0;
            LOG_INFO("DuniaTaa: motion staging textures {}x{} fmt {} created (2, for chaining)", desc.Width,
                     desc.Height, (UINT) desc.Format);
        }

        m_next = m_cur ^ 1;
        return m_tex[m_next];
    }

    void CommitCopy()
    {
        std::lock_guard<std::mutex> lock(m_lock);
        if (m_fills < 2)
            ++m_fills;
        m_cur = m_next;
        m_fresh.store(true, std::memory_order_release);
    }

    // Teardown only: the caller must already have waited for the GPU, which ReleaseScratch does.
    void ReleaseAllAfterGpuIdle()
    {
        std::lock_guard<std::mutex> lock(m_lock);
        for (auto& tex : m_tex)
        {
            if (tex != nullptr)
            {
                tex->Release();
                tex = nullptr;
            }
        }
        m_w = m_h = 0;
        m_cur = 0;
        m_fills = 0;
        m_fresh.store(false, std::memory_order_relaxed);
    }

    // Each copy is consumed by exactly one decode, so nothing stale is ever decoded. Deliberately not a
    // Present-index comparison: Dunia records across Present boundaries, so the index at the dispatch and
    // at the splice need not agree, and requiring them to gave 0 decodes for 300 copies.
    bool TakeFresh() { return m_fresh.exchange(false, std::memory_order_acquire); }

  private:
    mutable std::mutex m_lock;
    ID3D12Resource* m_tex[2] = { nullptr, nullptr };
    int m_cur = 0;
    int m_next = 0;
    int m_fills = 0;
    UINT m_w = 0;
    UINT m_h = 0;
    bool m_warned = false;
    std::atomic<bool> m_fresh { false };
};
MvStagingPair g_mvStaging;

// The motion texture a list has bound and not yet had copied. Identity only.
std::mutex g_cmdMotionLock;
std::unordered_map<ID3D12GraphicsCommandList*, ID3D12Resource*> g_cmdMotion;

// Upscaling step 2, the last piece: FC6's upscale is a COMPUTE pass (FSR1's EASU), so its eye-sized output
// never reaches OMSetRenderTargets and the bind-tracking rule could not see it (two runs, 0 recognised). It
// is learned exactly as the motion pass is: while the heap mirroring is on, a compute table whose UAV is
// the eye-sized HDR target names the dispatch; the pipeline and the texture per Present parity are kept,
// and after that the dispatch is recognised by its pipeline and DLSS's output is copied over its output --
// the UAV state is a fact right after the dispatch on the same list. Dropped when a texture dies (a menu).
std::mutex g_cmdUpscaleLock;
std::unordered_map<ID3D12GraphicsCommandList*, ID3D12Resource*> g_cmdUpscale;
ID3D12PipelineState* g_upPipeline = nullptr;   // identity only
std::atomic<ID3D12Resource*> g_upByParity[2] { nullptr, nullptr };
std::atomic<bool> g_upPipelineReady { false };
std::atomic<uint64_t> g_upCopies { 0 };
std::atomic<int> g_upPeeks { 0 };
uint64_t g_upSearchUntil = 0;   // Present after which the mirroring stops waiting for the upscale dispatch
bool g_upGaveUp = false;
uint64_t g_resolveSplicedPresent = UINT64_MAX; // the Present whose resolve DLSS replaced (so the copy is fresh)
std::atomic<ID3D12GraphicsCommandList*> g_resolveSplicedCmd { nullptr }; // the list it was recorded on (identity)
// Stats for the F4 A/B: comparing frame rate across runs is meaningless here, so the numbers have to
// come from one run with the path switching on and off inside it.
std::atomic<uint64_t> g_mvCopies { 0 };
std::atomic<uint64_t> g_mvDecodes { 0 };
uint64_t g_mvStatsAt = 0;
std::atomic<bool> g_mvStatsRestart { false };
constexpr uint64_t kMvStatsPresents = 600;
ID3D12RootSignature* g_mvRs = nullptr;
ID3D12PipelineState* g_mvPso = nullptr;
// The decode pass's root constants, laid out to match its cbuffer word for word. Three SRVs because
// the matrix path needs our depth and the game's ShaderIndex beside its packed vectors; the two it
// does not use yet are bound as null descriptors rather than left unwritten.
constexpr UINT kMvSrvCount = 4;
// Which textures the cached views describe, so the swap cannot leave them pointing at the wrong pair.
ID3D12Resource* g_mvViewedSrc = nullptr;
ID3D12Resource* g_mvViewedPrev = nullptr;
ID3D12Resource* g_mvViewedDepth = nullptr;

// A depth resource cannot be read through its depth format; it needs the colour twin of the same layout. Returns
// UNKNOWN for anything that is not a depth format, which is the signal to leave the slot as a null descriptor
// rather than to guess.
DXGI_FORMAT DepthSrvFormat(DXGI_FORMAT format)
{
    switch (format)
    {
    case DXGI_FORMAT_D32_FLOAT:
    case DXGI_FORMAT_R32_TYPELESS:
    case DXGI_FORMAT_R32_FLOAT:
        return DXGI_FORMAT_R32_FLOAT;
    case DXGI_FORMAT_D32_FLOAT_S8X24_UINT:
    case DXGI_FORMAT_R32G8X24_TYPELESS:
        return DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS;
    case DXGI_FORMAT_D24_UNORM_S8_UINT:
    case DXGI_FORMAT_R24G8_TYPELESS:
        return DXGI_FORMAT_R24_UNORM_X8_TYPELESS;
    case DXGI_FORMAT_D16_UNORM:
    case DXGI_FORMAT_R16_TYPELESS:
        return DXGI_FORMAT_R16_UNORM;
    default:
        return DXGI_FORMAT_UNKNOWN;
    }
}
struct MvConstants
{
    UINT size[2];
    UINT useMatrix;
    UINT halfRes;
    float matrix[16];
    float unproject[2];
    UINT chain;
    // 1: read the half-res field with a guarded bilinear filter instead of the nearest cell. See the shader.
    UINT smoothField;
    // The same reprojection to the OTHER eye's previous frame, and whether the object path is on.
    float other[16];
    UINT objects;
    // 1: this copy's vectors MAY be same-eye (the motion pass ran with our matrix), so they are also judged
    // against the same-eye baseline, with a knee whose floor is sameEyeFloor pixels.
    UINT sameEye;
    float sameEyeFloor;
    // What a same-eye residual is multiplied by: the G-buffer's object motion covers ONE Present (the game's
    // previous object transforms), the same-eye camera baseline covers two.
    float sameEyeResidualScale;
    // Days Gone's composition, for the OTHER-eye object branch (F4 x2, the everyday state). Live from the overlay.
    float objectScale;      // the residual is one Present of object motion on a two-Present baseline: 2 (DG), or 1
    float objectFloor;      // the knee's floor in px: 1.0 (as validated) .. 0.1 (DG's fadeStart)
    float objectSpeedTerm;  // floor += this * |camera motion| in px (DG: 0.0015)
    UINT fastChain;         // 1: above ~1-2 px of object motion, take the CHAINED two-hop lookup instead of x2
    // 1: the RESIDUAL VIEW -- write what the object branch decided (gate weight, residual length) instead of
    // the motion, for VisualizeMv's mode 1. DLSS gets meaningless motion meanwhile; the view replaces the picture.
    UINT debugView;
    // 1: the other-eye residual only where this pixel IS the nearest surface around it (the same-eye branch's
    // validated halo rule). Behind a silhouette the dilated field holds the near surface's vector, and the
    // mismatch was carried as motion: the dotted yellow edges of the residual view.
    UINT objectNearOnly;
    // 1: no residual where the game's vector agrees best with the FARTHEST surface around the pixel -- a near
    // pixel carrying the background's vector at a silhouette (the yellow edges against the sky).
    UINT objectBgReject;
    // The neighbourhood (px, every pixel within it) both surface tests search for the nearest/farthest depth.
    UINT objectRadius;
};
static_assert(sizeof(MvConstants) == 208, "the root constants must match the decode shader's cbuffer");
constexpr UINT kMvConstants = (UINT) (sizeof(MvConstants) / sizeof(UINT));
ID3D12DescriptorHeap* g_mvHeap = nullptr;
UINT g_mvDescriptorSize = 0;
// Diagnostic motion vector view: draws the decoded vectors DLSS receives over the upscaled output.
ID3D12RootSignature* g_mvVizRs = nullptr;
ID3D12PipelineState* g_mvVizPso = nullptr;
ID3D12DescriptorHeap* g_mvVizHeap = nullptr;
bool g_mvVizViewsValid = false;
// The residual view's copy of DLSS's picture, read as an SRV. It could not be read through the output's UAV:
// d3dcompiler_47 (SM 5.0) refuses typed UAV loads of anything but single 32-bit elements, whatever the GPU
// supports -- the first build failed to compile in game (X3676) though the SDK's fxc had accepted it. Ours
// alone; at rest it is always COPY_DEST, so its state is a fact at every use.
ID3D12Resource* g_mvVizScene = nullptr;
bool g_mvVizFailed = false; // a compile/creation failure is final: never retried every frame

// Views into the decode heap survive between frames; only the resources they point at can change.
bool g_mvViewsValid = false;
ID3D12DescriptorHeap* g_uavCpu = nullptr;
ID3D12DescriptorHeap* g_uavGpu = nullptr;
UINT g_createdW = 0;
UINT g_createdH = 0;
UINT g_pinnedW = 0;
UINT g_pinnedH = 0;
DXGI_FORMAT g_pinnedFormat = DXGI_FORMAT_R16G16B16A16_FLOAT;
ID3D12Resource* g_dest = nullptr;
UINT g_destW = 0;
UINT g_taaSearchSceneW = 0; // the scene width the TAA resolve search was started against
UINT g_destH = 0;
DXGI_FORMAT g_destFormat = DXGI_FORMAT_R16G16B16A16_FLOAT;
ID3D12Resource* g_taa = nullptr;
ID3D12Resource* g_taaDepth = nullptr;
// The depth bound alongside the game's motion buffer: the main scene depth at render resolution. Other
// render-sized depths exist (a copy is used during lighting), so this is the one the upscale reads.
ID3D12Resource* g_mvDepth = nullptr;
// The scene depth the upscale reads, followed frame to frame. The game reallocates its targets on a
// reload or settings change; a depth grabbed once goes stale and DLSS reads dead data.
ID3D12Resource* g_renderDepth = nullptr;
// Last render-size depth bound with a scene colour target this frame. Identity only, no reference: the
// game has it bound, so it is alive while the frame is being recorded.
std::atomic<ID3D12Resource*> g_lastSceneDepth { nullptr };
UINT g_taaW = 0;
UINT g_taaH = 0;
int g_taaScore = -1;
uint64_t g_taaSeenPresent = 0; // last Present the learned scaled scene was bound with depth (present thread reads)
int g_destScore = -1;
UINT g_createdInW = 0;
UINT g_createdInH = 0;
UINT g_createdOutW = 0;
UINT g_createdOutH = 0;
DXGI_FORMAT g_createdFormat = DXGI_FORMAT_UNKNOWN;
int g_createdQuality = -1;
// The creation keys the features were (or are being) made with. Kept so the live rebuild can hand them to
// OptiScaler's staged recreate, which builds the new feature from g_params -- see RebuildForScene.
UINT g_createFlags = 0;
NVSDK_NGX_PerfQuality_Value g_createPerf = NVSDK_NGX_PerfQuality_Value_DLAA;
// Live rebuild on a Resolution Scale change. Everything the rebuild swaps (g_motion, g_uavCpu/g_uavGpu, the
// created sizes, g_mvViewsValid) is read in splice mode only inside SpliceTaa, which holds this lock for its
// whole body -- so the swap, also done inside it, cannot race a reader. The game's recording threads run
// SpliceTaa; the present thread never touches these while the splice owns the frame.
std::mutex g_spliceLock;
bool g_rebuildPending = false;       // under g_spliceLock: features marked, the new keys go into every evaluate
bool g_outputFresh = false;          // under g_spliceLock: the rebuild replaced g_output, which holds nothing yet

// ---- GPU timing of our passes at the splice (as in Crimson Desert) ------------------------------------------------
// [Inputs] DuniaTaaGpuTiming, live in the overlay. Four timestamps on the game's list around one splice: T0 before
// EvaluateOnList, T1 just before the DLSS evaluate (so T1-T0 = motion reconstruction: decode, matrix/object motion,
// edge search), T2 just after it (DLSS), T3 after the copy/downsample onto the resolve's target. Read back kTimeRing
// splices later. A splice that returns early fills the marks it skipped (never resolve an unwritten query) and is
// counted incomplete. The marks are thread-local and tied to the list: EvaluateOnList has other callers on other
// threads. Aggregates under g_timeLock (the splice's threads write, Present summarises).
constexpr uint32_t kTimeRing = 16;
std::mutex g_timeLock;
bool g_timeTried = false, g_timeReady = false;
ID3D12QueryHeap* g_timeHeap = nullptr;
ID3D12Resource* g_timeReadback = nullptr; // kTimeRing x 4 timestamps, persistently mapped
const uint64_t* g_timeData = nullptr;
std::atomic<uint64_t> g_tsFrequency { 0 }; // from the present queue
uint64_t g_timeSeq = 0;
bool g_timeFilled[kTimeRing] {};
bool g_timeComplete[kTimeRing] {};
int g_timeEye[kTimeRing] {};
struct PassTimes
{
    double motion = 0, dlss = 0, copy = 0;
    uint64_t n = 0;
};
PassTimes g_passTimes[2];            // per eye slot, this window
uint64_t g_timeIncomplete = 0, g_timeInvalid = 0;
std::atomic<unsigned> g_mirrorPresents { 0 };
DuniaTaa_Dx12::GpuTimes g_lastGpuTimes; // the last window, for the overlay (under g_timeLock)
thread_local ID3D12GraphicsCommandList* t_timeCmd = nullptr;
thread_local int t_timeSlot = -1;
thread_local unsigned t_timeMarks = 0; // bit i = T_i written

bool GpuTimingOn() { return Config::Instance()->DuniaTaaGpuTiming.value_or_default(); }

// Accumulates the splice kTimeRing back in `slot`, if it is complete and ordered. Under g_timeLock.
void ReadTimeSlot(uint32_t slot)
{
    if (!g_timeFilled[slot])
        return;
    g_timeFilled[slot] = false;
    const uint64_t* t = g_timeData + slot * 4;
    const uint64_t freq = g_tsFrequency.load(std::memory_order_relaxed);
    if (!g_timeComplete[slot])
    {
        ++g_timeIncomplete;
        return;
    }
    if (freq == 0 || t[0] == 0 || !(t[0] <= t[1] && t[1] <= t[2] && t[2] <= t[3]))
    {
        ++g_timeInvalid;
        return;
    }
    const double toMs = 1000.0 / (double) freq;
    auto& p = g_passTimes[g_timeEye[slot] & 1];
    p.motion += (double) (t[1] - t[0]) * toMs;
    p.dlss += (double) (t[2] - t[1]) * toMs;
    p.copy += (double) (t[3] - t[2]) * toMs;
    ++p.n;
    static bool loggedFirst = false;
    if (!loggedFirst)
    {
        loggedFirst = true;
        LOG_INFO("DuniaTaa: GPU timing: FIRST splice read back: motion {:.3f} ms, DLSS {:.3f} ms, copy {:.3f} ms (eye "
                 "slot {})",
                 (double) (t[1] - t[0]) * toMs, (double) (t[2] - t[1]) * toMs, (double) (t[3] - t[2]) * toMs,
                 g_timeEye[slot]);
    }
}

// T0, at the splice; the rest follow on this thread and list. Failure to set up is final and logged once.
void TimeBegin(ID3D12GraphicsCommandList* cmd, int eyeSlot)
{
    t_timeSlot = -1;
    if (!GpuTimingOn() || cmd == nullptr)
        return;
    std::lock_guard<std::mutex> lock(g_timeLock);
    if (!g_timeTried)
    {
        g_timeTried = true;
        ID3D12Device* device = nullptr;
        if (SUCCEEDED(cmd->GetDevice(IID_PPV_ARGS(&device))) && device != nullptr)
        {
            device->Release(); // the device outlives the list
            D3D12_QUERY_HEAP_DESC qd {};
            qd.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
            qd.Count = kTimeRing * 4;
            D3D12_HEAP_PROPERTIES heap = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK);
            auto desc = CD3DX12_RESOURCE_DESC::Buffer(kTimeRing * 4 * sizeof(uint64_t));
            void* mapped = nullptr;
            if (SUCCEEDED(device->CreateQueryHeap(&qd, IID_PPV_ARGS(&g_timeHeap))) &&
                SUCCEEDED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
                                                          D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                          IID_PPV_ARGS(&g_timeReadback))) &&
                SUCCEEDED(g_timeReadback->Map(0, nullptr, &mapped)) && mapped != nullptr)
            {
                g_timeData = (const uint64_t*) mapped;
                g_timeReady = true;
                LOG_INFO("DuniaTaa: GPU timing ready (motion / DLSS / copy per splice, per eye; heap mirroring counted)");
            }
        }
        if (!g_timeReady)
            LOG_WARN("DuniaTaa: GPU timing off: no timestamp query heap or readback buffer");
    }
    if (!g_timeReady)
        return;
    const uint32_t slot = (uint32_t) (g_timeSeq++ % kTimeRing);
    ReadTimeSlot(slot);
    g_timeEye[slot] = eyeSlot;
    t_timeCmd = cmd;
    t_timeSlot = (int) slot;
    t_timeMarks = 0;
    cmd->EndQuery(g_timeHeap, D3D12_QUERY_TYPE_TIMESTAMP, slot * 4 + 0);
    t_timeMarks |= 1u;
}

// T1 / T2 inside EvaluateOnList: only for the splice being timed on this thread and list.
void TimeMark(ID3D12GraphicsCommandList* cmd, int index)
{
    if (t_timeSlot < 0 || cmd != t_timeCmd || (t_timeMarks & (1u << index)))
        return;
    cmd->EndQuery(g_timeHeap, D3D12_QUERY_TYPE_TIMESTAMP, (UINT) t_timeSlot * 4 + index);
    t_timeMarks |= 1u << index;
}

// T3 and the resolve. Marks a splice skipped are written now, so nothing unwritten is resolved.
void TimeEnd(ID3D12GraphicsCommandList* cmd)
{
    if (t_timeSlot < 0 || cmd != t_timeCmd)
        return;
    const uint32_t slot = (uint32_t) t_timeSlot;
    const bool complete = (t_timeMarks & 0x7u) == 0x7u;
    for (int i = 1; i < 4; ++i)
        if (!(t_timeMarks & (1u << i)))
            cmd->EndQuery(g_timeHeap, D3D12_QUERY_TYPE_TIMESTAMP, slot * 4 + i);
    cmd->ResolveQueryData(g_timeHeap, D3D12_QUERY_TYPE_TIMESTAMP, slot * 4, 4, g_timeReadback,
                          (UINT64) slot * 4 * sizeof(uint64_t));
    {
        std::lock_guard<std::mutex> lock(g_timeLock);
        g_timeComplete[slot] = complete;
        g_timeFilled[slot] = true;
    }
    t_timeSlot = -1;
    t_timeCmd = nullptr;
}

struct TimeScope
{
    ID3D12GraphicsCommandList* cmd;
    ~TimeScope() { TimeEnd(cmd); }
};
uint64_t g_rebuildAt = 0;            // under g_spliceLock: Present of the last rebuild (one per 180, the deferred window)
UINT g_rebuildSeenW = 0, g_rebuildSeenH = 0; // under g_spliceLock: the scene size a mismatch was first seen at
uint64_t g_rebuildSeenAt = 0;
std::atomic<int> g_rebuilds { 0 };

NVSDK_NGX_Parameter* g_params = nullptr;
NVSDK_NGX_Handle* g_handle = nullptr;
// Upscaling keeps one DLSS feature per eye. R.E.A.L. VR renders the eyes alternately and the game keeps TAA
// history per eye; a single feature fed both eyes reprojects each one onto the other's viewpoint, rejects
// the history, and shows its noisy single-frame fallback. g_handle serves the first eye seen.
NVSDK_NGX_Handle* g_handleEye2 = nullptr;
// The game's per-eye upscale sources, as first seen. Identity only, never dereferenced.
ID3D12Resource* g_eyeSrc[2] = { nullptr, nullptr };
bool g_slotNeedsReset[2] = { true, true };
bool g_ngxInited = false;
bool g_loggedSkipMenu = false;
bool g_loggedActive = false;
bool g_needReset = true;
bool g_splicedThisFrame = false;
bool g_evalThisFrame = false;
bool g_allowSplice = false;
bool g_loggedSplice = false;
bool g_loggedCopySplice = false;
// Bumped from the present thread and, since the real-motion copy, from the game's recording threads too.
std::atomic<int> g_skipCopyHook { 0 };

// Set while this backend evaluates DLSS, which runs NR synchronously on the same thread. Every bind, copy
// and view NR makes in that time goes through the same hooks as the game's. Taken for the game's, NR's
// foveated crop copy of the eye colour triggered the DLAA fallback splice from inside NR, which ran NR
// again, recursively, until the game crashed. Per thread: other threads recording game lists still count.
thread_local int g_inEvaluate = 0;

struct EvaluateScope
{
    EvaluateScope() { ++g_inEvaluate; }
    ~EvaluateScope() { --g_inEvaluate; }
};

// Diagnostic: how splices map onto Presents and eyes, to find a reliable way to tell the eyes apart
// before giving each its own DLSS history. A fixed budget of lines, spent only while the splice runs.
std::atomic<uint64_t> g_presentIndex { 0 };

// Upscale health, for the overlay warning. Present indices of the last eye-sized scene bind, the last
// successful upscale, and the last upscale draw seen at sizes the feature was not created for.
std::atomic<uint64_t> g_lastEyeScenePresent { 0 };
std::atomic<uint64_t> g_firstEyeScenePresent { 0 };
std::atomic<uint64_t> g_lastUpscalePresent { 0 };
std::atomic<uint64_t> g_mismatchPresent { 0 };
std::atomic<unsigned> g_mismatchW { 0 };
std::atomic<unsigned> g_mismatchH { 0 };
std::atomic<int> g_upscaleProblem { 0 };
std::atomic<uint64_t> g_upscaleEpisode { 0 };

// Diagnostics for future motion vector work (split A/B, motion vector view, frame trace, game motion
// vector modes) only take effect with [Inputs] DuniaTaaDiagnostics=true. Off, a value left in the ini
// cannot change what players see.
bool DiagnosticsOn() { return Config::Instance()->DuniaTaaDiagnostics.value_or_default(); }

// FC6 exposes no camera motion vector buffer: the candidates it offers are not motion vectors, and using
// them makes DLSS noisy. Motion vectors stay off unless diagnostics are on.
int MvMode() { return DiagnosticsOn() ? Config::Instance()->DuniaTaaMotionVectors.value_or_default() : 0; }

// Diagnostic: log every render-target bind and copy for two Presents (one per eye), to see how the
// game gets from its render-resolution image to the eye buffer when resolution scale is below 100%.
// Armed from the overlay; recording order, not GPU execution order, so the list and thread are logged.
std::atomic<uint64_t> g_traceFrom { UINT64_MAX };
std::atomic<int> g_traceLines { 0 };

bool Tracing()
{
    const uint64_t from = g_traceFrom.load();
    const uint64_t now = g_presentIndex.load();
    return from != UINT64_MAX && now >= from && now < from + 2 && g_traceLines.load() < 4000;
}

// Draws recorded on each command list since its last traced bind or copy (draw hooks exist only with diagnostics).
std::mutex g_traceDrawLock;
std::unordered_map<ID3D12GraphicsCommandList*, int> g_traceDraws;

std::string TraceDrawsSince(ID3D12GraphicsCommandList* cmd)
{
    std::lock_guard<std::mutex> lock(g_traceDrawLock);
    const auto it = g_traceDraws.find(cmd);
    if (it == g_traceDraws.end() || it->second == 0)
        return {};
    const int draws = it->second;
    it->second = 0;
    return std::format(" after {} draws", draws);
}

std::string TraceDesc(ID3D12Resource* resource)
{
    if (resource == nullptr)
        return "none";
    const auto desc = resource->GetDesc();
    return std::format("{:X} {}x{} fmt {} flags {:X}", (size_t) resource, desc.Width, desc.Height,
                       (UINT) desc.Format, (UINT) desc.Flags);
}

struct CmdTargets
{
    ID3D12Resource* color = nullptr;
    ID3D12Resource* depth = nullptr;
    ID3D12Resource* lastPinned = nullptr;
    ID3D12Resource* lastPinnedDepth = nullptr;
    ID3D12Resource* spliceColor = nullptr;
    ID3D12Resource* spliceDepth = nullptr;

    // Upscale splice. The game upscales with one draw: a render-sized HDR target is released and the
    // eye-sized HDR target is bound with no depth. DLSS runs at that bind, while the source is still in
    // RENDER_TARGET state. The game's draw then writes its own result, and the next bind on the same
    // list, before any eye-resolution pass draws over it, copies DLSS's output on top.
    ID3D12Resource* upscaleSrc = nullptr;
    ID3D12Resource* copyDest = nullptr;
    ID3D12Resource* pendingDest = nullptr;
    uint64_t pendingPresent = 0;
    bool copyPending = false;
    // Splice mode, below 100%: the game's eye-sized upscale target while it is bound on this list. DLSS's
    // output is copied over it right after EACH draw on it (the last copy wins), because the list ends
    // there -- the "copy at the next bind" of the older mode never came (recognised, 0 copies).
    ID3D12Resource* upscaleDrawTarget = nullptr;
    // The private-copy probe: the list's current targets are the G-buffer (set at bind time, see OnSetTargets).
    bool gbuffer = false;
    // The private copy: the motion target is bound at ANY slot, with depth (NoteAllTargets). `gbuffer` above
    // sees slot 0 only, and a run that redirected every one of its binds with same-eye values in every one
    // and NPCs did not change: the pass that draws them must write motion at a later slot.
    bool gbufferAny = false;
    // Every bound render target's format and the depth format (NoteAllTargets, while the probe or the private copy
    // runs). The in-place test proved the NPC motion reads the FIRST copy of the per-view struct,
    // through no bind the motion-format tests picked: Dunia likely packs NPC velocity into an ordinary G-buffer
    // target, so the pass is identified by its whole target signature, not by one format.
    uint8_t rtCount = 0;
    uint16_t rtFmt[8] {};
    uint16_t dsFmt = 0;
    // THE G-buffer (pass map): targets [R8G8B8A8 x3, R8G8_UNORM] + D32_S8, ~1000 per-view binds a
    // Present on the FIRST copy. The R8G8_UNORM at slot 3 is the 8-bit velocity target -- not a format
    // IsMotionFormat knows, which is why every earlier scope missed this pass.
    bool gbufferMrt = false;
};

std::unordered_map<ID3D12GraphicsCommandList*, CmdTargets> g_cmdTargets;

std::unordered_set<uint64_t> g_loggedPairs;

// TEST ONLY ([Inputs] DuniaTaaTestFakeProbe, overlay; off by default). Reproduces a failure seen at one eye size on a
// setup that does not show it. FC6 renders a square environment-probe pass -- RGBA16F, R16G16, fmt 61 and 54 over
// its own depth -- whose size follows the render resolution: 1440^2 flat and 2288^2 at 70% on one eye size,
// but 4096^2 on a larger eye, where it outgrew the eye, was learned as the eye (and the pinned scene), and the
// TAA resolve was searched for at that size for ever: F6 did nothing, silently. With this on, the size-learning
// DECISIONS (Note's learners and the gameplay signature) see any square target of 1024+ px as 4096x4096. Never used
// for anything that creates, copies or binds: in splice mode nothing uses these sizes until the resolve is found.
bool FakeProbeOn()
{
    return Config::Instance()->DuniaTaaTestFakeProbe.value_or_default();
}

void FakeProbeSize(UINT& w, UINT& h)
{
    if (!FakeProbeOn() || w != h || w < 1024 || w == 4096)
        return;
    static std::atomic<bool> logged { false };
    if (!logged.exchange(true))
        LOG_WARN("DuniaTaa: TEST fake probe ON -- the size learners see the square {}x{} target (and every square one "
                 "of 1024+ px) as 4096x4096, as on a 3440x3760 headset. Test only ([Inputs] DuniaTaaTestFakeProbe)",
                 w, h);
    w = h = 4096;
}

bool IsEvalSize(UINT width, UINT height)
{
    // Separates scene/eye buffers from menu and UI targets: creating DLAA on the menu target and then
    // recreating it is what crashed boot. Menu/UI is 1920x1080 and eye buffers are usually well
    // above that, but a lower VR resolution would fall under a fixed threshold and
    // silently disable the whole path, so the limits are configurable.
    const auto minW = (UINT) std::max(0, Config::Instance()->DuniaTaaMinEvalWidth.value_or_default());
    const auto minH = (UINT) std::max(0, Config::Instance()->DuniaTaaMinEvalHeight.value_or_default());
    return width >= minW || height >= minH;
}

// Automatic mode decides from gameplay, not from the first frame: menus and loading screens draw at the
// full eye size whatever the Resolution Scale. Gameplay is recognised by the G-buffer's two-channel
// target (R16G16), which menus never draw and which is always at the scene's render resolution.
std::atomic<int> g_autoMode { -1 }; // -1 undecided, 0 DLAA, 1 upscaling
std::atomic<uint64_t> g_gameplayFullPresent { 0 };
std::atomic<uint64_t> g_gameplayFullSince { 0 };
std::atomic<uint64_t> g_gameplayScaledPresent { 0 };
std::atomic<unsigned> g_gameplayScaledW { 0 };
std::atomic<unsigned> g_gameplayScaledH { 0 };

// 0 = DLAA forced, 1 = upscaling forced, 2 = automatic.
int ConfiguredMode()
{
    const int value = Config::Instance()->DuniaTaaQuality.value_or_default();
    return value == 0 || value == 1 ? value : 2;
}

// The mode in effect: the created feature's once there is one, else what is forced or decided so far
// (-1 while automatic mode is still waiting for gameplay).
int EffectiveMode()
{
    if (g_createdQuality >= 0)
        return g_createdQuality;
    const int configured = ConfiguredMode();
    return configured == 2 ? g_autoMode.load() : configured;
}

int SelectedDuniaQuality() { return EffectiveMode() == 1 ? 1 : 0; }

const char* DuniaQualityName(int quality)
{
    return quality == 1 ? "Upscaling" : "DLAA";
}

bool NrOn() { return Config::Instance()->DlssNrEnabled.value_or_default(); }

// F6, as in FC5 and Days Gone: nothing is created or evaluated until the key is pressed, so the
// Resolution Scale and the other settings can be set first (the feature is created once and never
// rebuilt). Pressing it again hands the picture back to the game without releasing anything.
std::atomic<bool> g_started { false };
bool StartOnKey() { return Config::Instance()->DuniaTaaStartOnKey.value_or_default(); }
bool Started() { return !StartOnKey() || g_started.load(); }

// Logs once that everything is ready and only the key is missing, so the log explains an idle session.
bool WaitedForKey(const char* what)
{
    if (Started())
        return true;
    static const char* logged = nullptr;
    if (logged != what)
    {
        logged = what;
        LOG_INFO("DuniaTaa: {}: ready, waiting for F6 ([Inputs] DuniaTaaStartOnKey)", what);
    }
    return false;
}

// DLAA only runs to carry NR. Upscaling is its own reason to run: turning NR off would otherwise hand
// the frame back to the game's own upscale without any sign of it. Automatic mode stays on while it has
// not decided, so gameplay below 100% gets upscaling even with NR off.
// [Inputs] DuniaTaaSpliceTaa, read once: step 3 of the port. DLSS runs *at* the game's own temporal
// resolve and that draw is skipped, so DLSS is the temporal resolve rather than a second pass over one.
// Startup-only, because the feature it needs is created once, for one size and one format, and is never
// rebuilt live.
bool SpliceTaaMode()
{
    static const bool on = Config::Instance()->DuniaTaaSpliceTaa.value_or_default();
    return on;
}

// [Inputs] DuniaTaaNrOnTonemapped, read once: the feature it creates is never rebuilt live.
bool TonemappedMode()
{
    // The two modes compete for the one DLSS feature -- different size, different format -- so the splice
    // wins and the known-good fallback stands down. Said out loud: an ini with both set and only one of
    // them running is exactly the kind of silent choice that has cost runs here.
    static const bool on = []
    {
        const bool tonemapped = Config::Instance()->DuniaTaaNrOnTonemapped.value_or_default();
        if (tonemapped && SpliceTaaMode())
            LOG_INFO("DuniaTaa: DuniaTaaSpliceTaa is on, so DuniaTaaNrOnTonemapped stands down this session "
                     "-- one feature, one size, created once");
        return tonemapped && !SpliceTaaMode();
    }();
    return on;
}

// Tonemapped mode: the finished 8-bit eye image copied out in VR, and the eye-sized depth drawn with it.
std::atomic<unsigned> g_tonemapW { 0 };
std::atomic<unsigned> g_tonemapH { 0 };
std::atomic<int> g_tonemapFormat { 0 };
// Each finished image's own eye depth. The next eye's lists are recorded before the previous eye is copied out,
// so "the last eye depth" can belong to the other eye; the pairing is taken on the game's list, where the eye
// depth last bound before the tonemap bind is that image's. Under g_lock; depths retained.
struct TonemapDepth
{
    ID3D12Resource* image = nullptr; // identity only
    ID3D12Resource* depth = nullptr;
};
// Two eyes, plus room for an intermediate 8-bit pass (the trace showed a third eye-sized 8-bit target).
constexpr int kTonemapSlots = 4;
TonemapDepth g_tonemapDepth[kTonemapSlots];
int g_tonemapNextSlot = 0;
std::unordered_map<ID3D12GraphicsCommandList*, ID3D12Resource*> g_cmdEyeDepth; // identity only, per list

bool SplicePathOn()
{
    // Splice mode owns the feature and splices at the resolve instead; the upscale-bind and copy splices
    // must not also fire, or two paths would evaluate the same feature in one frame.
    if (SpliceTaaMode() || TonemappedMode())
        return false;
    return NrOn() || EffectiveMode() == 1 || (ConfiguredMode() == 2 && g_createdQuality < 0 && g_autoMode.load() < 0);
}

void WaitForGpu(ID3D12CommandQueue* queue);

void Barrier(ID3D12GraphicsCommandList* cmd, ID3D12Resource* resource, D3D12_RESOURCE_STATES before,
             D3D12_RESOURCE_STATES after)
{
    if (cmd == nullptr || resource == nullptr || before == after)
        return;

    D3D12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::Transition(resource, before, after);
    cmd->ResourceBarrier(1, &barrier);
}

void Retain(ID3D12Resource** slot, ID3D12Resource* value)
{
    if (slot == nullptr || *slot == value)
        return;
    if (value != nullptr)
        value->AddRef();
    if (*slot != nullptr)
        (*slot)->Release();
    *slot = value;
}

// For references a recorded frame may still use: the old one is released a couple of seconds later,
// once any command list that referenced it has long executed. Released immediately, a game-freed
// resource could disappear under work still in flight.
// ID3D12Pageable, not ID3D12Resource: the live rebuild on a Resolution Scale change also retires our own
// descriptor heaps, which a list recorded a moment ago may still reference in the same way.
struct DeferredRelease
{
    ID3D12Pageable* resource;
    uint64_t releaseAfter;
};
// Locked because the two ends are on different threads: RetainDeferred is called from the game's own
// recording threads (the tonemap depth pairing, the render depth, the motion staging buffers) and
// ReleaseDeferred from the present thread. A std::vector push_back racing an erase is heap corruption,
// and it only bites when something actually queues a release -- which is a size change, i.e. entering a
// menu. That is where three crashes in a row stopped the log, each time straight after the size change
// arrived. Pre-existing; the motion staging buffers just made it frequent enough to find.
// GUARDRAIL. The list itself is private, so it cannot be touched without the lock -- see the note at
// `GUARDRAILS` at the top of this file before relaxing this.
//
// Why: this was a bare std::vector, added to from the game's recording threads (RetainDeferred, called
// from the tonemap depth pairing, the render depth and the motion staging buffers) and erased from on the
// present thread (ReleaseDeferred, from OnPresent). push_back racing erase corrupts the heap, and it only
// bites when something actually queues a release -- a size change, i.e. entering a menu. That was three
// menu crashes, and the first two fixes were plausible guesses at other causes.
class DeferredReleaseList
{
  public:
    void Add(ID3D12Pageable* resource, uint64_t releaseAfter)
    {
        if (resource == nullptr)
            return;
        std::lock_guard<std::mutex> lock(m_lock);
        m_entries.push_back({ resource, releaseAfter });
    }

    void ReleaseDue(uint64_t presentIndex)
    {
        std::lock_guard<std::mutex> lock(m_lock);
        auto it = std::remove_if(m_entries.begin(), m_entries.end(),
                                 [&](const DeferredRelease& entry)
                                 {
                                     if (presentIndex < entry.releaseAfter)
                                         return false;
                                     entry.resource->Release();
                                     return true;
                                 });
        m_entries.erase(it, m_entries.end());
    }

  private:
    std::mutex m_lock;
    std::vector<DeferredRelease> m_entries;
};
DeferredReleaseList g_deferredReleases;

void RetainDeferred(ID3D12Resource** slot, ID3D12Resource* value, uint64_t presentIndex)
{
    if (slot == nullptr || *slot == value)
        return;
    if (value != nullptr)
        value->AddRef();
    g_deferredReleases.Add(*slot, presentIndex + 180);
    *slot = value;
}

void ReleaseDeferred(uint64_t presentIndex) { g_deferredReleases.ReleaseDue(presentIndex); }

// Hands one of OUR objects to the deferred list, taking over its reference: it is released after the same
// 180 Presents as a game resource, because a list recorded a moment ago may still reference it.
void RetireDeferred(ID3D12Pageable* object, uint64_t presentIndex) { g_deferredReleases.Add(object, presentIndex + 180); }

bool IsSwapchainBuffer(ID3D12Resource* resource)
{
    auto& buffers = State::Instance().scBuffers;
    for (auto* buffer : buffers)
    {
        if (buffer == resource)
            return true;
    }

    return false;
}

// The RTV/UAV creation hooks offer every view's resource as a candidate, which includes the scratch
// textures created here. g_motion is R16G16_FLOAT at the pinned size, so it outscores the game's own
// motion buffer and gets adopted as if it were the game's -- feeding our own stale output back in.
bool IsOwnedResource(ID3D12Resource* resource)
{
    return resource != nullptr && (resource == g_motion || resource == g_output);
}

DXGI_FORMAT UavColorFormat(DXGI_FORMAT format)
{
    switch (format)
    {
    case DXGI_FORMAT_R8G8B8A8_TYPELESS:
        return DXGI_FORMAT_R8G8B8A8_UNORM;
    case DXGI_FORMAT_B8G8R8A8_TYPELESS:
        return DXGI_FORMAT_B8G8R8A8_UNORM;
    case DXGI_FORMAT_R16G16B16A16_TYPELESS:
        return DXGI_FORMAT_R16G16B16A16_FLOAT;
    case DXGI_FORMAT_R10G10B10A2_TYPELESS:
        return DXGI_FORMAT_R10G10B10A2_UNORM;
    case DXGI_FORMAT_R16G16_TYPELESS:
        return DXGI_FORMAT_R16G16_FLOAT;
    case DXGI_FORMAT_R32G32_TYPELESS:
        return DXGI_FORMAT_R32G32_FLOAT;
    default:
        return format;
    }
}

bool IsHdrFormat(DXGI_FORMAT format)
{
    switch (UavColorFormat(format))
    {
    case DXGI_FORMAT_R16G16B16A16_FLOAT:
    case DXGI_FORMAT_R11G11B10_FLOAT:
    case DXGI_FORMAT_R32G32B32A32_FLOAT:
        return true;
    default:
        return false;
    }
}

int ColorScore(const D3D12_RESOURCE_DESC& desc)
{
    switch (UavColorFormat(desc.Format))
    {
    case DXGI_FORMAT_R16G16B16A16_FLOAT:
    case DXGI_FORMAT_R32G32B32A32_FLOAT:
        break;
    case DXGI_FORMAT_R11G11B10_FLOAT:
        break;
    case DXGI_FORMAT_R10G10B10A2_UNORM:
        break;
    case DXGI_FORMAT_R8G8B8A8_UNORM:
    case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
    case DXGI_FORMAT_B8G8R8A8_UNORM:
    case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
        break;
    default:
        return -1;
    }

    int score = 0;
    if (IsHdrFormat(desc.Format))
        score += 300;
    else
        score += 160;
    if (desc.Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS)
        score += 20;
    if (desc.Flags & D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET)
        score += 10;
    return score;
}

bool IsMotionFormat(DXGI_FORMAT format)
{
    switch (UavColorFormat(format))
    {
    case DXGI_FORMAT_R16G16_FLOAT:
    case DXGI_FORMAT_R16G16_UNORM:
    case DXGI_FORMAT_R16G16_SNORM:
    case DXGI_FORMAT_R32G32_FLOAT:
        return true;
    default:
        return false;
    }
}

int MotionScore(const D3D12_RESOURCE_DESC& desc)
{
    if (!IsMotionFormat(desc.Format) || !IsEvalSize((UINT) desc.Width, desc.Height))
        return -1;

    // Which of the two same-sized candidates is the real motion buffer is not knowable from the
    // descriptor alone, so the format preference is switchable rather than assumed.
    const bool preferUnorm = Config::Instance()->DuniaTaaMvPreferUnorm.value_or_default();

    int score = (int) (desc.Width * desc.Height / 1024);
    switch (UavColorFormat(desc.Format))
    {
    case DXGI_FORMAT_R16G16_FLOAT:
        score += preferUnorm ? 20 : 50;
        break;
    case DXGI_FORMAT_R32G32_FLOAT:
        score += 40;
        break;
    case DXGI_FORMAT_R16G16_SNORM:
        score += 30;
        break;
    case DXGI_FORMAT_R16G16_UNORM:
        score += preferUnorm ? 50 : 20;
        break;
    default:
        break;
    }
    return score;
}

DXGI_FORMAT MotionSrvFormat(DXGI_FORMAT format)
{
    switch (format)
    {
    case DXGI_FORMAT_R16G16_TYPELESS:
        return DXGI_FORMAT_R16G16_UNORM;
    case DXGI_FORMAT_R32G32_TYPELESS:
        return DXGI_FORMAT_R32G32_FLOAT;
    default:
        return UavColorFormat(format);
    }
}

bool NeedsMvDecode(DXGI_FORMAT format)
{
    return UavColorFormat(format) != DXGI_FORMAT_R16G16_FLOAT;
}

UINT SceneW();
UINT SceneH();

bool SamePinnedSize(UINT width, UINT height)
{
    return std::abs((int) width - (int) g_pinnedW) <= 32 && std::abs((int) height - (int) g_pinnedH) <= 32;
}

// Motion vectors come at the resolution the scene is rendered at: the eye size for DLAA, the scaled
// render size for upscaling. Accepting only the eye size discarded every render-sized buffer.
bool MvSizeOk(UINT width, UINT height)
{
    return std::abs((int) width - (int) SceneW()) <= 32 && std::abs((int) height - (int) SceneH()) <= 32;
}

bool SimilarAspect(UINT w1, UINT h1, UINT w2, UINT h2)
{
    if (w1 == 0 || h1 == 0 || w2 == 0 || h2 == 0)
        return false;
    const int64_t left = (int64_t) w1 * (int64_t) h2;
    const int64_t right = (int64_t) w2 * (int64_t) h1;
    const int64_t diff = left > right ? left - right : right - left;
    // Tight on purpose: every frame also renders square targets whose side is the eye buffer's height,
    // and a loose match took those for the scaled image. A genuine resolution scale keeps the aspect.
    return diff * 50 < left + right;
}

// The aspect test above cannot separate the square probe from a scaled scene when the eye itself is square (within
// its 2%). Then a scaled candidate needs the TAA resolve's confirmation (see Note). [Inputs] DuniaTaaTestSquareEye
// (read, never saved; overlay) forces these rules on any eye, to exercise that path on a non-square headset.
bool SquareEyeRules()
{
    if (Config::Instance()->DuniaTaaTestSquareEye.value_or_default())
        return true;
    const UINT eyeW = g_vrEyeW.load(std::memory_order_relaxed), eyeH = g_vrEyeH.load(std::memory_order_relaxed);
    return eyeW != 0 && SimilarAspect(eyeW, eyeH, 1, 1);
}

// The size the scene is RENDERED at: the scaled TAA target below 100% Resolution Scale, the eye size at
// 100%. Every gate in the resolve splice that says "eye size" means this: at 70% the G-buffer, the motion
// pass and the TAA resolve all run at render size (70% of the eye's width and height) and a search pinned to
// the eye size found "0 half-res motion + 1 eye HDR" for ever.
bool IsScaledTaa(UINT w, UINT h, UINT destW, UINT destH, bool depthBound = false);
bool SceneScaled()
{
    return g_taaW != 0 && g_destW != 0 && IsScaledTaa(g_taaW, g_taaH, g_destW, g_destH, g_taaDepth != nullptr);
}
UINT SceneW() { return SceneScaled() ? g_taaW : g_pinnedW; }
UINT SceneH() { return SceneScaled() ? g_taaH : g_pinnedH; }

// `depthBound`: the candidate was bound with a depth buffer. The exact-half-size exclusion exists for the
// game's half-resolution bloom composite at 100% scale, which is HDR and eye-aspect and looks exactly like a
// 50% scene -- but it is a post pass and never binds depth, while the scene at 50% (DLSS Performance) always
// does. Without the distinction a 50% Resolution Scale fell back to the game's own picture in silence.
bool IsScaledTaa(UINT w, UINT h, UINT destW, UINT destH, bool depthBound)
{
    if (destW == 0 || destH == 0 || !IsEvalSize(w, h))
        return false;
    if (std::abs((int) w - (int) destW) <= 32 && h + 64 < destH)
        return false;
    if (!depthBound && std::abs((int) w * 2 - (int) destW) <= 64 && std::abs((int) h * 2 - (int) destH) <= 64)
        return false;
    if (w + 32 >= destW && h + 32 >= destH)
        return false;
    if (w < destW * 45 / 100 || h < destH * 45 / 100)
        return false;
    return SimilarAspect(w, h, destW, destH);
}

bool IsUpscale() { return g_createdOutW > g_createdInW + 32 || g_createdOutH > g_createdInH + 32; }

void ReleaseOwnedResources()
{
    if (g_handle != nullptr)
    {
        NVSDK_NGX_D3D12_ReleaseFeature(g_handle);
        g_handle = nullptr;
    }

    if (g_handleEye2 != nullptr)
    {
        NVSDK_NGX_D3D12_ReleaseFeature(g_handleEye2);
        g_handleEye2 = nullptr;
    }
    g_eyeSrc[0] = nullptr;
    g_eyeSrc[1] = nullptr;

    if (g_output != nullptr)
    {
        g_output->Release();
        g_output = nullptr;
    }

    if (g_motion != nullptr)
    {
        g_motion->Release();
        g_motion = nullptr;
    }

    // Sized like g_output, and released under the same GPU wait the caller took for it.
    if (g_mvVizScene != nullptr)
    {
        g_mvVizScene->Release();
        g_mvVizScene = nullptr;
    }
    g_mvVizViewsValid = false;

    Retain(&g_gameMv, nullptr);
    g_mvViewsValid = false;

    g_mvStaging.ReleaseAllAfterGpuIdle();

    if (g_uavCpu != nullptr)
    {
        g_uavCpu->Release();
        g_uavCpu = nullptr;
    }

    if (g_uavGpu != nullptr)
    {
        g_uavGpu->Release();
        g_uavGpu = nullptr;
    }

    g_createdW = 0;
    g_createdH = 0;
    g_createdInW = 0;
    g_createdInH = 0;
    g_createdOutW = 0;
    g_createdOutH = 0;
    g_createdFormat = DXGI_FORMAT_UNKNOWN;
    g_createdQuality = -1;
    g_needReset = true;
}

bool CreateTex(ID3D12Device* device, UINT width, UINT height, DXGI_FORMAT format, D3D12_RESOURCE_FLAGS flags,
               D3D12_RESOURCE_STATES state, ID3D12Resource** out)
{
    D3D12_HEAP_PROPERTIES heap = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
    auto desc = CD3DX12_RESOURCE_DESC::Tex2D(format, width, height, 1, 1, 1, 0, flags);
    auto hr = device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr, IID_PPV_ARGS(out));
    if (FAILED(hr))
    {
        LOG_ERROR("DuniaTaa: CreateCommittedResource {}x{} fmt {} flags {:X} failed: {:X}", width, height, (UINT) format,
                  (UINT) flags, (UINT) hr);
        return false;
    }

    return true;
}

bool EnsureScratch(ID3D12Device* device, ID3D12CommandQueue* queue, UINT inW, UINT inH, UINT outW, UINT outH,
                   DXGI_FORMAT colorFormat)
{
    if (g_output != nullptr && g_motion != nullptr && g_createdInW == inW && g_createdInH == inH &&
        g_createdOutW == outW && g_createdOutH == outH)
        return true;

    if (g_output != nullptr || g_handle != nullptr)
    {
        WaitForGpu(queue);
        ReleaseOwnedResources();
    }

    const auto uavFormat = UavColorFormat(colorFormat);
    if (!CreateTex(device, outW, outH, uavFormat, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                   D3D12_RESOURCE_STATE_UNORDERED_ACCESS, &g_output))
        return false;

    if (!CreateTex(device, inW, inH, DXGI_FORMAT_R16G16_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                   D3D12_RESOURCE_STATE_UNORDERED_ACCESS, &g_motion))
        return false;

    D3D12_DESCRIPTOR_HEAP_DESC cpuDesc {};
    cpuDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    cpuDesc.NumDescriptors = 1;
    if (FAILED(device->CreateDescriptorHeap(&cpuDesc, IID_PPV_ARGS(&g_uavCpu))))
        return false;

    D3D12_DESCRIPTOR_HEAP_DESC gpuDesc = cpuDesc;
    gpuDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    if (FAILED(device->CreateDescriptorHeap(&gpuDesc, IID_PPV_ARGS(&g_uavGpu))))
        return false;

    device->CreateUnorderedAccessView(g_motion, nullptr, nullptr, g_uavCpu->GetCPUDescriptorHandleForHeapStart());
    device->CreateUnorderedAccessView(g_motion, nullptr, nullptr, g_uavGpu->GetCPUDescriptorHandleForHeapStart());

    g_createdW = inW;
    g_createdH = inH;
    g_createdInW = inW;
    g_createdInH = inH;
    g_createdOutW = outW;
    g_createdOutH = outH;
    g_createdFormat = colorFormat;
    return true;
}

bool EnsureCommands(ID3D12Device* device)
{
    if (g_device == device && g_allocator != nullptr && g_cmd != nullptr && g_fence != nullptr && g_fenceEvent != nullptr)
        return true;

    if (g_cmd != nullptr)
    {
        g_cmd->Release();
        g_cmd = nullptr;
    }
    if (g_allocator != nullptr)
    {
        g_allocator->Release();
        g_allocator = nullptr;
    }
    if (g_fence != nullptr)
    {
        g_fence->Release();
        g_fence = nullptr;
    }
    if (g_fenceEvent != nullptr)
    {
        CloseHandle(g_fenceEvent);
        g_fenceEvent = nullptr;
    }

    g_device = device;
    g_fenceValue = 0;

    if (FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&g_allocator))))
        return false;
    if (FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, g_allocator, nullptr, IID_PPV_ARGS(&g_cmd))))
        return false;
    g_cmd->Close();
    if (FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&g_fence))))
        return false;

    g_fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    return g_fenceEvent != nullptr;
}

void WaitForGpu(ID3D12CommandQueue* queue)
{
    if (g_fence == nullptr || g_fenceValue == 0)
        return;

    if (g_fence->GetCompletedValue() >= g_fenceValue)
        return;

    if (FAILED(g_fence->SetEventOnCompletion(g_fenceValue, g_fenceEvent)))
        return;

    WaitForSingleObject(g_fenceEvent, 2000);
}

bool InitNgx(ID3D12Device* device)
{
    auto& state = State::Instance();
    if (!state.nvngxDx12Inited)
    {
        NVSDK_NGX_FeatureCommonInfo info {};
        auto exePath = Util::ExePath().remove_filename();
        auto result = NVSDK_NGX_D3D12_Init_with_ProjectID(OPTI_GUID, state.NVNGX_Engine, OPTI_VERSION, exePath.c_str(),
                                                          device, &info,
                                                          state.NVNGX_Version == 0 ? NVSDK_NGX_Version_API
                                                                                   : state.NVNGX_Version);
        if (result != NVSDK_NGX_Result_Success)
        {
            LOG_ERROR("DuniaTaa: NGX init failed: {:X}", (UINT) result);
            return false;
        }
    }

    g_ngxInited = true;
    return true;
}

// The creation keys for a size pair, into g_createFlags / g_createPerf. Shared by CreateDlss and the live
// rebuild, so a feature rebuilt for a new Resolution Scale is made exactly as one created at startup.
void ComputeCreateKeys(UINT inW, UINT inH, UINT outW, UINT outH, DXGI_FORMAT colorFormat)
{
    UINT flags = NVSDK_NGX_DLSS_Feature_Flags_AutoExposure;
    if (Config::Instance()->DuniaTaaDepthInverted.value_or_default())
        flags |= NVSDK_NGX_DLSS_Feature_Flags_DepthInverted;
    if (IsHdrFormat(colorFormat))
        flags |= NVSDK_NGX_DLSS_Feature_Flags_IsHDR;

    const bool upscale = outW > inW + 32 || outH > inH + 32;
    // The motion texture is made at the INPUT size (EnsureScratch), so when upscaling DLSS has to be told
    // its vectors are low-res; without the flag it samples them as if they were eye-sized.
    if (upscale)
        flags |= NVSDK_NGX_DLSS_Feature_Flags_MVLowRes;
    // The quality flag from the ratio, the way a native game would ask: 50% is Performance, 58% Balanced,
    // everything from 62% up stays Max Quality (the validated 70% state is unchanged). With
    // RenderPresetForAll set the preset is the same whatever the flag; without it the flag picks the default.
    NVSDK_NGX_PerfQuality_Value quality = NVSDK_NGX_PerfQuality_Value_DLAA;
    if (upscale)
    {
        const float ratio = std::min((float) inW / (float) outW, (float) inH / (float) outH);
        quality = ratio < 0.54f   ? NVSDK_NGX_PerfQuality_Value_MaxPerf
                  : ratio < 0.62f ? NVSDK_NGX_PerfQuality_Value_Balanced
                                  : NVSDK_NGX_PerfQuality_Value_MaxQuality;
        LOG_INFO("DuniaTaa: DLSS quality flag for {}x{} -> {}x{} (ratio {:.2f}): {}", inW, inH, outW, outH, ratio,
                 quality == NVSDK_NGX_PerfQuality_Value_MaxPerf    ? "Performance"
                 : quality == NVSDK_NGX_PerfQuality_Value_Balanced ? "Balanced"
                                                                   : "Max Quality");
    }
    g_createFlags = flags;
    g_createPerf = quality;
}

// The creation keys on g_params. Width/Height are also the evaluate's render size, set per evaluate anyway.
void SetCreateKeys(UINT inW, UINT inH, UINT outW, UINT outH)
{
    g_params->Set(NVSDK_NGX_Parameter_DLSS_Feature_Create_Flags, g_createFlags);
    g_params->Set(NVSDK_NGX_Parameter_Width, inW);
    g_params->Set(NVSDK_NGX_Parameter_Height, inH);
    g_params->Set(NVSDK_NGX_Parameter_OutWidth, outW);
    g_params->Set(NVSDK_NGX_Parameter_OutHeight, outH);
    g_params->Set(NVSDK_NGX_Parameter_PerfQualityValue, g_createPerf);
    // Our flags are facts about our inputs; the [InitFlags] overrides must not rewrite them (IFeature.cpp).
    g_params->Set("OptiScaler.HostInitFlags", 1);
}

// FC5's rule: 8 jitter phases per eye at 100%, more as the render shrinks (16 at 70%, 32 at 50%) -- the
// sample positions have to cover an output pixel, which is (out/in)^2 render pixels.
void SetJitterPhasesFor(UINT inW, UINT inH, UINT outW, UINT outH)
{
    const double ratio = ((double) outW * (double) outH) / std::max(1.0, (double) inW * (double) inH);
    g_jitterPhases.store(std::clamp((int) std::lround(8.0 * ratio), 8, 32), std::memory_order_relaxed);
}

bool CreateDlss(ID3D12GraphicsCommandList* cmd, UINT inW, UINT inH, UINT outW, UINT outH, DXGI_FORMAT colorFormat)
{
    if (g_handle != nullptr)
        return true;

    if (g_params == nullptr)
    {
        if (NVSDK_NGX_D3D12_GetCapabilityParameters(&g_params) != NVSDK_NGX_Result_Success)
        {
            LOG_ERROR("DuniaTaa: GetCapabilityParameters failed");
            return false;
        }
    }

    ComputeCreateKeys(inW, inH, outW, outH, colorFormat);
    SetCreateKeys(inW, inH, outW, outH);
    const bool upscale = outW > inW + 32 || outH > inH + 32;

    auto result = NVSDK_NGX_D3D12_CreateFeature(cmd, NVSDK_NGX_Feature_SuperSampling, g_params, &g_handle);
    if (result != NVSDK_NGX_Result_Success)
    {
        LOG_ERROR("DuniaTaa: CreateFeature {} failed: {:X}", upscale ? "Upscaling" : "DLAA", (UINT) result);
        g_handle = nullptr;
        return false;
    }

    // A second eye only has something to keep apart when history accumulates. FC6 exposes no camera motion
    // vectors, so with them off DLSS resets every frame and a second feature would only hold output-sized
    // memory. Decided at creation; switching motion vectors on later shares one feature until restart.
    const bool perEyeHistory = MvMode() != 0;
    // Splice mode always takes the second feature. It *replaces* the game's temporal resolve, so DLSS is
    // the thing accumulating history, per eye, whether or not motion vectors are on yet -- and F4 turns
    // real motion on *after* creation, while a feature is never rebuilt live. Deciding this from
    // MvMode() at creation would therefore decide it wrong every time. Being wrong this way costs one
    // output-sized allocation; being wrong the other way is each eye's history reprojected onto the
    // other eye's viewpoint, which is what one shared feature does the moment motion arrives.
    const bool wantSecond = SpliceTaaMode() || (upscale && perEyeHistory);
    if (upscale && !perEyeHistory && !SpliceTaaMode())
        LOG_INFO("DuniaTaa: motion vectors off, one upscaling feature serves both eyes");

    if (wantSecond)
    {
        auto second = NVSDK_NGX_D3D12_CreateFeature(cmd, NVSDK_NGX_Feature_SuperSampling, g_params, &g_handleEye2);
        if (second != NVSDK_NGX_Result_Success)
        {
            g_handleEye2 = nullptr;
            LOG_WARN("DuniaTaa: second-eye feature failed ({:X}); both eyes will share one history, which "
                     "reprojects each eye onto the other's viewpoint as soon as motion vectors are on",
                     (UINT) second);
        }
        else
        {
            LOG_INFO("DuniaTaa: second DLSS feature created -- one history per eye");
        }
    }

    g_createdQuality = upscale ? 1 : 0;
    SetJitterPhasesFor(inW, inH, outW, outH);
    LOG_INFO("DuniaTaa: created {} {}x{} -> {}x{} fmt {} (depth {})", upscale ? "Upscaling" : "DLAA", inW, inH, outW,
             outH, (UINT) colorFormat,
             Config::Instance()->DuniaTaaDepthInverted.value_or_default() ? "inverted" : "normal");
    return true;
}

// LIVE REBUILD on a Resolution Scale change, from SpliceTaa under g_spliceLock. Nothing is torn down here:
// the "no live GPU teardown" rule (a direct rebuild from Present once crashed the driver) is about the
// direct path. This goes through the one the live preset combo already uses -- OptiScaler's staged
// recreate, which hands the old feature to a delayed destroy and builds the new one on the evaluated list
// over that handle's next three evaluates (each returns without writing: three frames of the previous
// output per eye). Passthrough DLSS creates from the evaluate's own parameter table, g_params, so the new
// size and keys reach it through EvaluateOnList, which sets them while g_rebuildPending.
//
// What is sized by the INPUT and so replaced: g_motion and the two heaps holding its UAV. The old ones go to
// the deferred list (a list recorded a moment ago may still reference them), the new ones are made here.
// The OUTPUT is the eye size at every scale; it changes only when the eye does -- the VR eye resolution changed
// mid-session (that used to need a restart). Then g_output (and the motion view's copy
// of it, g_mvVizScene) are replaced the same way, and until the features are rebuilt the splice hands the frame
// back to the game (g_outputFresh): the new texture holds nothing yet, where a kept one held the last frame.
bool RebuildForScene(ID3D12GraphicsCommandList* cmd, UINT inW, UINT inH, UINT outW, UINT outH)
{
    ID3D12Device* device = nullptr;
    if (FAILED(cmd->GetDevice(IID_PPV_ARGS(&device))) || device == nullptr)
        return false;
    device->Release();

    const bool newOutput = outW != g_createdOutW || outH != g_createdOutH;
    ID3D12Resource* output = nullptr;
    if (newOutput && !CreateTex(device, outW, outH, UavColorFormat(g_createdFormat),
                                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                &output))
    {
        g_rebuildAt = g_presentIndex.load(); // one try per deferred window
        LOG_ERROR("DuniaTaa: live rebuild: could not make the {}x{} output texture; the old feature stays (restart to "
                  "change the eye resolution)",
                  outW, outH);
        return false;
    }

    ID3D12Resource* motion = nullptr;
    ID3D12DescriptorHeap* uavCpu = nullptr;
    ID3D12DescriptorHeap* uavGpu = nullptr;
    D3D12_DESCRIPTOR_HEAP_DESC cpuDesc {};
    cpuDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    cpuDesc.NumDescriptors = 1;
    D3D12_DESCRIPTOR_HEAP_DESC gpuDesc = cpuDesc;
    gpuDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    if (!CreateTex(device, inW, inH, DXGI_FORMAT_R16G16_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                   D3D12_RESOURCE_STATE_UNORDERED_ACCESS, &motion) ||
        FAILED(device->CreateDescriptorHeap(&cpuDesc, IID_PPV_ARGS(&uavCpu))) ||
        FAILED(device->CreateDescriptorHeap(&gpuDesc, IID_PPV_ARGS(&uavGpu))))
    {
        // Never recorded anywhere, so these can go at once.
        if (output != nullptr)
            output->Release();
        if (motion != nullptr)
            motion->Release();
        if (uavCpu != nullptr)
            uavCpu->Release();
        if (uavGpu != nullptr)
            uavGpu->Release();
        g_rebuildAt = g_presentIndex.load(); // not every splice: one try per deferred window
        LOG_ERROR("DuniaTaa: live rebuild: could not make the {}x{} motion texture or its heaps; the old feature "
                  "stays (restart to change the Resolution Scale)",
                  inW, inH);
        return false;
    }
    device->CreateUnorderedAccessView(motion, nullptr, nullptr, uavCpu->GetCPUDescriptorHandleForHeapStart());
    device->CreateUnorderedAccessView(motion, nullptr, nullptr, uavGpu->GetCPUDescriptorHandleForHeapStart());

    const uint64_t now = g_presentIndex.load();
    RetireDeferred(g_motion, now);
    RetireDeferred(g_uavCpu, now);
    RetireDeferred(g_uavGpu, now);
    g_motion = motion;
    g_uavCpu = uavCpu;
    g_uavGpu = uavGpu;
    // The decode heap's UAV points at the old texture until rebuilt.
    g_mvViewsValid = false;
    g_mvVizViewsValid = false;
    if (newOutput)
    {
        RetireDeferred(g_output, now);
        g_output = output;
        if (g_mvVizScene != nullptr)
        {
            RetireDeferred(g_mvVizScene, now); // made again at the new size on its next use
            g_mvVizScene = nullptr;
        }
        g_outputFresh = true;
    }

    const UINT oldW = g_createdInW, oldH = g_createdInH;
    const UINT oldOutW = g_createdOutW, oldOutH = g_createdOutH;
    ComputeCreateKeys(inW, inH, outW, outH, g_createdFormat);
    g_createdW = inW;
    g_createdH = inH;
    g_createdInW = inW;
    g_createdInH = inH;
    g_createdOutW = outW;
    g_createdOutH = outH;
    const bool upscale = outW > inW + 32 || outH > inH + 32;
    g_createdQuality = upscale ? 1 : 0;
    SetJitterPhasesFor(inW, inH, outW, outH);

    // Only this module's two handles, exactly as RequestDlssPreset marks them.
    auto& state = State::Instance();
    int marked = 0;
    for (NVSDK_NGX_Handle* handle : { g_handle, g_handleEye2 })
    {
        if (handle == nullptr)
            continue;
        state.newBackend = Upscaler::DLSS;
        state.changeBackend[handle->Id] = true;
        ++marked;
    }
    g_rebuildPending = true;
    g_rebuildAt = now;
    g_needReset = true;
    g_rebuilds.fetch_add(1, std::memory_order_relaxed);
    LOG_INFO("DuniaTaa: live rebuild #{}: {} changed, {}x{} -> {}x{} becomes {} {}x{} -> {}x{} "
             "({} phases); {} feature(s) marked for OptiScaler's staged recreate, motion texture{} replaced",
             g_rebuilds.load(), newOutput ? "the eye resolution" : "Resolution Scale", oldW, oldH, oldOutW, oldOutH,
             upscale ? "Upscaling" : "DLAA", inW, inH, outW, outH, g_jitterPhases.load(), marked,
             newOutput ? " and output texture" : "");
    return true;
}

bool EnsureMvDecode(ID3D12Device* device)
{
    if (g_mvPso != nullptr && g_mvRs != nullptr && g_mvHeap != nullptr)
        return true;

    // Ported from FC5's Fc5Taa_Dx11.cpp CSMain. The two engines share the encoding, the 0.1 UV scale, the y-down
    // convention and the ShaderIndex bit 3 object rule; they differ only in where the matrix rows sit,
    // what the matrix's third input is, which channel carries ShaderIndex, and which depth value means
    // far plane. It replaces a placeholder from the hunt era that guessed the decode from the format.
    //
    // Packed is the game's half-res R8G8_UNORM motion texture. The pass runs at the full
    // eye size and samples it at id/2, as FC5's does. Motion is previous minus current, in pixels.
    //
    // UseMatrix 0: the game's own vectors everywhere -- the only mode that works until its
    // current-to-previous matrix can be read at runtime, because FC6 binds nothing as a root CBV. 1: camera
    // motion from our own depth through that matrix, in float, with the game's quantised vectors only
    // on object-flagged pixels where the two disagree by more than the 8-bit curve's step. That hybrid
    // is FC5's, and it is what stopped static props the game flags as objects from wobbling.
    static const char shader[] = R"(
Texture2D<float2> Packed : register(t0);
Texture2D<float> Depth : register(t1);
Texture2D<float4> ShaderIndex : register(t2);
Texture2D<float2> PackedPrev : register(t3);
RWTexture2D<float2> Motion : register(u0);
cbuffer Params : register(b0)
{
    uint2 Size;
    uint UseMatrix;
    uint HalfRes;
    // The motion pass's reprojection matrix, which lives at its constant buffer's
    // registers 2..5 (it loads rows 0, 1 and 3 -- x, y and w; the z row is unused), and the depth
    // unprojection pair from register 1 .xy. The matrix takes current NDC plus an unprojected depth,
    // not raw depth, so it carries the current inverse projection with it.
    float4 A[4];
    float2 Unproject;
    // 1: chain this Present's vectors with the previous Present's, to reach the same eye. The game's
    // vectors point one Present back, which under alternate-eye rendering is the *other* eye, parallax
    // included -- confirmed from the motion view, where the two eyes' fields have opposite horizontal
    // signs with the head still. Two hops land back on this eye.
    uint Chain;
    uint SmoothField;
    // The same matrix to the OTHER eye's previous frame (one Present back), which is what the game's own
    // vectors were computed against. Objects != 0 turns the object path on.
    float4 B[4];
    uint Objects;
    uint SameEye;
    float SameEyeFloor;
    float SameEyeResidualScale;
    float ObjectScale;
    float ObjectFloor;
    float ObjectSpeedTerm;
    uint FastChain;
    uint DebugView;
    uint ObjectNearOnly;
    uint ObjectBgReject;
    uint ObjectRadius;
};

// Dunia's 8-bit curve, inverted. It encodes sign(v) * |v|^0.25 inside |v| <= 0.2 and
// sign(v) * (0.4140746 |v| + 0.5859254) outside, then * 0.5 + 127/255. The pass scales the NDC delta
// by a literal 5.0 before encoding and NDC is twice UV, so one decoded unit is 0.1 of the screen --
// FC5's scale exactly. abs() inside the pow because pow() of a negative base is undefined in HLSL.
float Decode(float v)
{
    const float d = v - 0.498039;
    const float f = d > 0.33437 ? d * 4.830047 - 1.415024
                  : d < -0.33437 ? d * 4.830047 + 1.415024
                  : sign(d) * pow(abs(2.0 * d), 4.0);
    return 0.1 * f;
}

float4 Transform(float4 p, float4 r0, float4 r1, float4 r2, float4 r3)
{
    return float4(dot(p, r0), dot(p, r1), dot(p, r2), dot(p, r3));
}

[numthreads(8, 8, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= Size.x || id.y >= Size.y)
        return;
    const float2 size = float2(Size);
    const float2 uv = (float2(id.xy) + 0.5) / size;
    const float2 ndc = float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);

    uint pw, ph;
    Packed.GetDimensions(pw, ph);
    const uint2 at = min(HalfRes != 0 ? id.xy / 2 : id.xy, uint2(pw - 1, ph - 1));
    const float2 packed = Packed[at];
    // No y flip: the game negates the NDC y delta before encoding, so the stored pair is already
    // y-down -- the same as FC5's, and the same as the pixel offsets DLSS wants.
    const float2 decoded = float2(Decode(packed.x), Decode(packed.y)) * size;
    // The game's field is half resolution, and the nearest cell hands every 2x2 block of pixels one vector --
    // steps across a face or an arm once the vectors are fine enough to matter (same-eye, 0.08 px knee).
    // Bilinear over the four cells around this pixel, decoded first because the curve is not linear, and
    // The four cells are only GATHERED here; they are blended in the same-eye branch below, where the two
    // baselines are known. The first version blended here, guarded only by "within 4 px of the nearest
    // cell": on distant surfaces a same-eye object vector and the other-eye camera vector beside it are less
    // than 4 px apart, the blend matched neither baseline, and the difference was carried as motion -- the
    // crawl ("quicksand") on textures. Faces and skin were better, so the filter stays, for same-eye
    // vectors only and only among neighbours of the same kind.
    float2 tapV[4] = { decoded, decoded, decoded, decoded };
    float tapW[4] = { 0.0, 0.0, 0.0, 0.0 };
    if (SmoothField != 0 && HalfRes != 0)
    {
        const float2 cellPos = (float2(id.xy) + 0.5) * 0.5 - 0.5;
        const int2 cell = int2(floor(cellPos));
        const float2 f = cellPos - float2(cell);
        const int2 last = int2(pw, ph) - 1;
        [unroll]
        for (int ty = 0; ty <= 1; ++ty)
        {
            [unroll]
            for (int tx = 0; tx <= 1; ++tx)
            {
                const float2 tap = Packed[clamp(cell + int2(tx, ty), int2(0, 0), last)];
                tapV[ty * 2 + tx] = float2(Decode(tap.x), Decode(tap.y)) * size;
                tapW[ty * 2 + tx] = (tx == 0 ? 1.0 - f.x : f.x) * (ty == 0 ? 1.0 - f.y : f.y);
            }
        }
    }

    float2 offset = decoded;
    // The residual view: (gate weight, residual length in px) where the object branch judged the pixel,
    // (-1, 0) where it did not (sky, or no object data).
    float2 debugOut = float2(-1.0, 0.0);
    if (Chain != 0)
    {
        // Where this pixel was one Present ago, then that frame's own vector from there. Days Gone's
        // approach: exact for anything static, but the second lookup reads the other eye's field, so at
        // silhouettes it samples across a depth edge and shimmers. Occlusion handling only halved that
        // there; if it is bad here the matrix is the answer instead.
        const float2 back = uv + decoded / size;
        uint qw, qh;
        PackedPrev.GetDimensions(qw, qh);
        const uint2 at2 = min((uint2) (saturate(back) * float2(qw, qh)), uint2(qw - 1, qh - 1));
        const float2 second = PackedPrev[at2];
        offset = decoded + float2(Decode(second.x), Decode(second.y)) * size;
    }
    if (UseMatrix != 0)
    {
        const float z = Depth[id.xy];
        // 1/viewZ, which is AFFINE in the stored depth: viewZ = -near/(z + 1/far), so 1/viewZ = -z/near
        // - (1/far)/near. The old reciprocal form belonged to a matrix that was never obtained; nothing
        // used it, so nothing regresses. Unproject = (-1/near, -(1/far)/near).
        const float lin = Unproject.x * z + Unproject.y;
        const float4 p = float4(ndc, lin, 1.0);
        const float4 c = Transform(p, A[0], A[1], A[2], A[3]);
        const float2 matrixOffset = (c.xy / c.w - ndc) * float2(0.5, -0.5) * size;
        offset = matrixOffset;
        // Moving objects. The camera matrix knows nothing about them; the game's own vector does, but it
        // points at the OTHER eye's previous frame, parallax included. So the game's vector is judged
        // against the camera-only motion to THAT frame (B): where the two agree within the 8-bit curve's
        // step the pixel is static and the same-eye matrix is exact; where they disagree the pixel moved
        // by itself, and its own motion -- the difference -- is carried over onto the same-eye camera
        // motion. No ShaderIndex needed: the game's pass already wrote its object vectors only where it
        // had them, and everywhere else its vector IS camera motion to the other eye.
        //
        // This differs from FC5's rule, which compares the other-eye vector with the SAME-eye matrix: at
        // NPC range the parallax alone exceeds the tolerance, so every flagged pixel took the other-eye
        // vector with the interocular offset baked in. That is a plausible cause of the NPC shimmer FC5
        // and Days Gone share, and the reason this is not a straight port.
        if (Objects != 0 && z < 1.0)
        {
            // The game's field is DILATED: its pass writes each half-res cell's vector for the CLOSEST sample
            // in a 3x3 of cells, so at every silhouette the vector belongs to the near surface while this
            // pixel may be the far one. At the other eye the two surfaces' parallax differs by many pixels,
            // so judging the vector against this pixel's depth alone made every edge pixel "disagree" and
            // handed DLSS a difference-of-parallaxes in a band around every object -- the haloing of the
            // first object-motion run. So the baseline is also built at the nearest depth around here (in
            // 1/viewZ, which is what the matrix takes, so it is right whichever way the depth is stored:
            // closest = most negative), and the vector is judged against whichever surface it agrees with.
            // Every pixel within ObjectRadius (live; was every OTHER pixel within 2). The field's reach is its 3x3
            // dilation of half-res cells (+-2 px) plus our nearest-cell lookup (+1): a background pixel 3 px out
            // from a silhouette can carry the near vector with no near depth inside +-2, so neither the nearest-
            // surface test nor the background test below could see it (both toggles, no change).
            float linNear = lin;
            float linFar = lin;
            const int radius = (int) ObjectRadius;
            [loop]
            for (int dy = -radius; dy <= radius; ++dy)
            {
                [loop]
                for (int dx = -radius; dx <= radius; ++dx)
                {
                    const int2 at = clamp(int2(id.xy) + int2(dx, dy), int2(0, 0), int2(Size) - 1);
                    const float linAt = Unproject.x * Depth[at] + Unproject.y;
                    linNear = min(linNear, linAt);
                    linFar = max(linFar, linAt);
                }
            }
            const float4 pNear = float4(ndc, linNear, 1.0);
            const float4 oFar = Transform(p, B[0], B[1], B[2], B[3]);
            const float4 oNear = Transform(pNear, B[0], B[1], B[2], B[3]);
            const float4 cNear = Transform(pNear, A[0], A[1], A[2], A[3]);
            const float2 otherFar = (oFar.xy / oFar.w - ndc) * float2(0.5, -0.5) * size;
            const float2 otherNear = (oNear.xy / oNear.w - ndc) * float2(0.5, -0.5) * size;
            const float2 matrixNear = (cNear.xy / cNear.w - ndc) * float2(0.5, -0.5) * size;
)" // MSVC caps one string literal at 16 KB (C2026); adjacent literals are joined, and fc6_shader_check.py joins them too
                                 R"(
            const float2 rFar = decoded - otherFar;
            const float2 rNear = decoded - otherNear;
            const bool useNear = dot(rNear, rNear) < dot(rFar, rFar);
            const float2 r = useNear ? rNear : rFar;
            const float2 baseOther = useNear ? otherNear : otherFar;
            const float2 baseSame = useNear ? matrixNear : matrixOffset;
            // The 8-bit curve's step at this magnitude. A soft knee rather than a hard switch: nothing below
            // half a step (quantisation noise), all of the residual above one and a half (a moving object).
            // A hard switch strobed in Days Gone. At NPC range the other-eye parallax is ~50 px and the step
            // there ~2.7 px, so movements below that are lost in the encoding -- a limit of OTHER-eye
            // vectors, not of this rule; the way past it is same-eye vectors from the game's own pass.
            // Days Gone's composition (TaaMotion_Dx11), which reached "idle-animation smear much better" with
            // other-eye vectors alone: composed = matrixChain + 2*own*fade, the floor rising with camera speed
            // (quantisation noise grows with the vector's magnitude), and fast movers handed to the CHAINED
            // two-hop lookup -- the real second Present of their motion, where x2 extrapolation is wrong.
            const float2 tolerance = ObjectFloor + ObjectSpeedTerm * length(matrixOffset) +
                                     8.0 * pow(abs(baseOther) / 255.0, 0.75);
            // Behind a silhouette this pixel is the FAR surface while the dilated field carries the near one's
            // vector: nothing here moved by itself, so nothing is carried. The same test as the same-eye branch.
            const bool nearSurface = abs(lin - linNear) <= 0.1 * abs(linNear);
            float nearGate = (ObjectNearOnly != 0 && !nearSurface) ? 0.0 : 1.0;
            // The reverse case, which the rule above cannot see: this pixel IS the near surface (a frond, a roof
            // edge, a post) but the half-res field carries the BACKGROUND's vector, because thin near pixels are
            // missing from its depth. The residual view showed it: yellow along every silhouette
            // against the sky with nearest-surface on or off, larger for nearer objects -- the other-eye parallax
            // difference between the two surfaces. So the vector is also judged against the farthest surface
            // around here; when it agrees with that better than with this pixel's own baselines, it is a
            // background vector and nothing moved by itself. Only where a clearly farther surface is in reach.
            if (ObjectBgReject != 0 && abs(linFar - lin) > 0.1 * abs(lin))
            {
                const float4 oBg = Transform(float4(ndc, linFar, 1.0), B[0], B[1], B[2], B[3]);
                const float2 rBg = decoded - (oBg.xy / oBg.w - ndc) * float2(0.5, -0.5) * size;
                if (dot(rBg, rBg) < dot(r, r))
                    nearGate = 0.0;
            }
            const float2 w = smoothstep(0.5 * tolerance, 1.5 * tolerance, abs(r)) * nearGate;
            offset = baseSame + r * w * ObjectScale;
            debugOut = float2(max(w.x, w.y), length(r));
            if (FastChain != 0)
            {
                // Where this pixel was one Present ago, then that frame's own vector from there: two hops
                // land on this eye. Only trusted for a pixel that really moved, where the lookup's own error
                // (it reads the other eye's field across a depth edge) is small next to the motion.
                const float2 back = uv + decoded / size;
                uint qw, qh;
                PackedPrev.GetDimensions(qw, qh);
                const uint2 at2 = min((uint2) (saturate(back) * float2(qw, qh)), uint2(qw - 1, qh - 1));
                const float2 second = PackedPrev[at2];
                const float2 chained = decoded + float2(Decode(second.x), Decode(second.y)) * size;
                const float fast = smoothstep(1.0, 2.0, length(r)) * nearGate;
                offset = lerp(offset, chained, fast);
            }
            // With the motion pass's matrix patched, the vectors the pass COMPUTES are same-eye -- but the
            // ones it copies verbatim from the G-buffer for object-flagged pixels (ShaderIndex bit 3) are
            // not: the geometry passes wrote those against the game's own previous view, the other eye's.
            // One texture now holds both kinds and nothing here says which pixel is which, so the vector is
            // judged against both baselines and belongs to the one it sits closer to. A flagged pixel is
            // ~50 px of parallax from the same-eye baseline and so keeps the rule above, unchanged.
            if (SameEye != 0)
            {
                const float2 sFar = decoded - matrixOffset;
                const float2 sNear = decoded - matrixNear;
                const bool sUseNear = dot(sNear, sNear) < dot(sFar, sFar);
                const float2 rs = sUseNear ? sNear : sFar;
                if (dot(rs, rs) < dot(r, r))
                {
                    const float2 baseS = sUseNear ? matrixNear : matrixOffset;
                    const float2 tolS = SameEyeFloor + 8.0 * pow(abs(baseS) / 255.0, 0.75);
                    // The field is DILATED (the closest sample in a 3x3 of half-res cells), so in a band around
                    // every mover the vector is the mover's, not this pixel's. The other-eye rule never showed
                    // it for small motion because its 1 px floor threw small residuals away; with the floor at
                    // a quarter pixel the band carried the NPC's motion onto the background behind it -- the
                    // large halos of the first same-eye run. So the residual is carried only where
                    // this pixel IS the nearest surface around here; behind it, the camera matrix alone.
                    const bool pixelIsNear = abs(lin - linNear) <= 0.1 * abs(linNear);
                    // The smoothed residual: neighbours that are same-eye vectors too (nearer this baseline
                    // than the other-eye one) and within 4 px of the nearest cell's. All weights are zero
                    // unless SmoothField is on, and then the nearest cell alone is what is left.
                    float2 sum = 0.0;
                    float weight = 0.0;
                    [unroll]
                    for (int k = 0; k < 4; ++k)
                    {
                        const float2 ds = tapV[k] - baseS;
                        const float2 dO = tapV[k] - baseOther;
                        const bool sameKind = dot(ds, ds) < dot(dO, dO);
                        const float keep = (sameKind && all(abs(tapV[k] - decoded) < 4.0)) ? tapW[k] : 0.0;
                        sum += tapV[k] * keep;
                        weight += keep;
                    }
                    const float2 rsS = weight > 1e-4 ? sum / weight - baseS : rs;
                    const float2 wS = smoothstep(0.5 * tolS, 1.5 * tolS, abs(rsS));
                    offset = pixelIsNear ? baseS + rsS * wS * SameEyeResidualScale : matrixOffset;
                    // The residual view shows THIS branch's decision where it took the pixel: it used to
                    // only ever show the other-eye branch, so a working same-eye path looked unchanged in it.
                    debugOut = float2(pixelIsNear ? max(wS.x, wS.y) : 0.0, length(rsS));
                }
            }
        }
    }
    Motion[id.xy] = DebugView != 0 ? debugOut : offset;
}
)";

    ID3DBlob* cs = nullptr;
    ID3DBlob* errors = nullptr;
    if (FAILED(D3DCompile(shader, sizeof(shader) - 1, "DuniaTaaMv", nullptr, nullptr, "CSMain", "cs_5_0",
                          D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &cs, &errors)))
    {
        if (errors)
        {
            LOG_ERROR("DuniaTaa: MV decode compile failed: {}", (const char*) errors->GetBufferPointer());
            errors->Release();
        }
        return false;
    }
    if (errors)
        errors->Release();

    D3D12_DESCRIPTOR_RANGE ranges[2] {};
    ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    ranges[0].NumDescriptors = kMvSrvCount;
    ranges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    ranges[1].NumDescriptors = 1;
    ranges[1].OffsetInDescriptorsFromTableStart = kMvSrvCount;

    D3D12_ROOT_PARAMETER params[2] {};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[0].DescriptorTable.NumDescriptorRanges = 2;
    params[0].DescriptorTable.pDescriptorRanges = ranges;
    params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[1].Constants.Num32BitValues = kMvConstants;
    params[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    D3D12_ROOT_SIGNATURE_DESC rootDesc {};
    rootDesc.NumParameters = 2;
    rootDesc.pParameters = params;
    ID3DBlob* rsBlob = nullptr;
    if (FAILED(D3D12SerializeRootSignature(&rootDesc, D3D_ROOT_SIGNATURE_VERSION_1, &rsBlob, nullptr)))
    {
        cs->Release();
        return false;
    }
    if (FAILED(device->CreateRootSignature(0, rsBlob->GetBufferPointer(), rsBlob->GetBufferSize(),
                                           IID_PPV_ARGS(&g_mvRs))))
    {
        rsBlob->Release();
        cs->Release();
        return false;
    }
    rsBlob->Release();

    D3D12_COMPUTE_PIPELINE_STATE_DESC pso {};
    pso.pRootSignature = g_mvRs;
    pso.CS = { cs->GetBufferPointer(), cs->GetBufferSize() };
    const auto psoHr = device->CreateComputePipelineState(&pso, IID_PPV_ARGS(&g_mvPso));
    cs->Release();
    if (FAILED(psoHr))
        return false;

    D3D12_DESCRIPTOR_HEAP_DESC heap {};
    heap.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    heap.NumDescriptors = kMvSrvCount + 1;
    heap.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    if (FAILED(device->CreateDescriptorHeap(&heap, IID_PPV_ARGS(&g_mvHeap))))
        return false;

    g_mvDescriptorSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    return true;
}

// Draws g_motion (decoded, render-resolution pixel offsets) into g_output at output resolution:
// red = moving right, green = moving left, blue = vertical, full at 8 pixels. Same pattern as the
// decode pass, which runs on the game's list every frame. Built once; its views never change because
// g_motion and g_output are only recreated at startup.
// Upscaling step 2. At a Resolution Scale below 100% the feature is render -> eye and DLSS's output is
// eye-sized, but the game's post chain (bloom, tonemap) reads the resolve's RENDER-sized target and then
// its own upscale draw enlarges it. So the resolve's target gets a bilinear DOWNSAMPLE of DLSS's output
// (the chain stays coherent with what the eye finally sees), and DLSS's eye-sized output is copied over the
// game's upscale draw when that draw has run -- FC5's arrangement, with the old HDR-upscale splice's
// bind tracking doing the copy. One compute dispatch on the game's list, the shape VisualizeMv already
// proved there: descriptor heap, root signature and PSO restored by the splice's state save.
bool SameFormatFamily(DXGI_FORMAT a, DXGI_FORMAT b);
ID3D12RootSignature* g_dsRs = nullptr;
ID3D12PipelineState* g_dsPso = nullptr;
ID3D12DescriptorHeap* g_dsHeap = nullptr;
ID3D12Resource* g_dsViewsFor = nullptr; // the target the UAV in g_dsHeap was made for (identity only)
UINT g_dsDescriptorSize = 0;

bool DownsampleOutputTo(ID3D12GraphicsCommandList* cmd, ID3D12Resource* target, D3D12_RESOURCE_STATES targetBefore)
{
    if (cmd == nullptr || target == nullptr || g_output == nullptr)
        return false;
    const auto outDesc = g_output->GetDesc();
    const auto dstDesc = target->GetDesc();
    if ((dstDesc.Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS) == 0)
        return false;
    if (!SameFormatFamily(outDesc.Format, dstDesc.Format))
        return false;

    ID3D12Device* device = nullptr;
    if (FAILED(cmd->GetDevice(IID_PPV_ARGS(&device))) || device == nullptr)
        return false;
    device->Release();

    if (g_dsPso == nullptr)
    {
        static const char shader[] = R"(
Texture2D<float4> Src : register(t0);
SamplerState Linear : register(s0);
RWTexture2D<float4> Dst : register(u0);
cbuffer Params : register(b0)
{
    uint DstW;
    uint DstH;
};
[numthreads(8, 8, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= DstW || id.y >= DstH)
        return;
    const float2 uv = (float2(id.xy) + 0.5) / float2(DstW, DstH);
    Dst[id.xy] = Src.SampleLevel(Linear, uv, 0);
}
)";
        ID3DBlob* cs = nullptr;
        ID3DBlob* errors = nullptr;
        if (FAILED(D3DCompile(shader, sizeof(shader) - 1, "DuniaTaaDown", nullptr, nullptr, "CSMain", "cs_5_0",
                              D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &cs, &errors)))
        {
            if (errors)
            {
                LOG_ERROR("DuniaTaa: downsample compile failed: {}", (const char*) errors->GetBufferPointer());
                errors->Release();
            }
            return false;
        }
        if (errors)
            errors->Release();

        D3D12_DESCRIPTOR_RANGE ranges[2] {};
        ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        ranges[0].NumDescriptors = 1;
        ranges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
        ranges[1].NumDescriptors = 1;
        ranges[1].OffsetInDescriptorsFromTableStart = 1;
        D3D12_ROOT_PARAMETER params[2] {};
        params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        params[0].DescriptorTable.NumDescriptorRanges = 2;
        params[0].DescriptorTable.pDescriptorRanges = ranges;
        params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        params[1].Constants.Num32BitValues = 2;
        params[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        D3D12_STATIC_SAMPLER_DESC sampler {};
        sampler.Filter = D3D12_FILTER_MIN_MAG_LINEAR_MIP_POINT;
        sampler.AddressU = sampler.AddressV = sampler.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
        sampler.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        D3D12_ROOT_SIGNATURE_DESC rootDesc {};
        rootDesc.NumParameters = 2;
        rootDesc.pParameters = params;
        rootDesc.NumStaticSamplers = 1;
        rootDesc.pStaticSamplers = &sampler;
        ID3DBlob* rsBlob = nullptr;
        if (FAILED(D3D12SerializeRootSignature(&rootDesc, D3D_ROOT_SIGNATURE_VERSION_1, &rsBlob, nullptr)))
        {
            cs->Release();
            return false;
        }
        const auto rsHr =
            device->CreateRootSignature(0, rsBlob->GetBufferPointer(), rsBlob->GetBufferSize(), IID_PPV_ARGS(&g_dsRs));
        rsBlob->Release();
        if (FAILED(rsHr))
        {
            cs->Release();
            return false;
        }
        D3D12_COMPUTE_PIPELINE_STATE_DESC pso {};
        pso.pRootSignature = g_dsRs;
        pso.CS = { cs->GetBufferPointer(), cs->GetBufferSize() };
        const auto psoHr = device->CreateComputePipelineState(&pso, IID_PPV_ARGS(&g_dsPso));
        cs->Release();
        if (FAILED(psoHr))
            return false;
        D3D12_DESCRIPTOR_HEAP_DESC heap {};
        heap.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        heap.NumDescriptors = 2;
        heap.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        if (FAILED(device->CreateDescriptorHeap(&heap, IID_PPV_ARGS(&g_dsHeap))))
            return false;
        g_dsDescriptorSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        LOG_INFO("DuniaTaa: upscaling: downsample pass ready -- DLSS's {}x{} output feeds the game's {}x{} post "
                 "chain, and is copied over its upscale draw",
                 outDesc.Width, outDesc.Height, dstDesc.Width, dstDesc.Height);
    }

    // Views are rebuilt whenever the target differs from the one they were made for: the resolve alternates
    // between two targets (one per eye), and a menu replaces them.
    if (g_dsViewsFor != target)
    {
        D3D12_SHADER_RESOURCE_VIEW_DESC srv {};
        srv.Format = UavColorFormat(outDesc.Format);
        srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srv.Texture2D.MipLevels = 1;
        D3D12_UNORDERED_ACCESS_VIEW_DESC uav {};
        uav.Format = UavColorFormat(dstDesc.Format);
        uav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
        auto cpu = g_dsHeap->GetCPUDescriptorHandleForHeapStart();
        device->CreateShaderResourceView(g_output, &srv, cpu);
        cpu.ptr += g_dsDescriptorSize;
        device->CreateUnorderedAccessView(target, nullptr, &uav, cpu);
        g_dsViewsFor = target;
    }

    Barrier(cmd, g_output, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    Barrier(cmd, target, targetBefore, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    cmd->SetComputeRootSignature(g_dsRs);
    cmd->SetPipelineState(g_dsPso);
    cmd->SetDescriptorHeaps(1, &g_dsHeap);
    cmd->SetComputeRootDescriptorTable(0, g_dsHeap->GetGPUDescriptorHandleForHeapStart());
    const UINT constants[] = { (UINT) dstDesc.Width, dstDesc.Height };
    cmd->SetComputeRoot32BitConstants(1, 2, constants, 0);
    cmd->Dispatch(((UINT) dstDesc.Width + 7) / 8, (dstDesc.Height + 7) / 8, 1);
    Barrier(cmd, target, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, targetBefore);
    Barrier(cmd, g_output, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    return true;
}

bool VisualizeMv(ID3D12GraphicsCommandList* cmd)
{
    if (cmd == nullptr || g_motion == nullptr || g_output == nullptr || g_mvVizFailed)
        return false;

    ID3D12Device* device = nullptr;
    if (FAILED(cmd->GetDevice(IID_PPV_ARGS(&device))) || device == nullptr)
        return false;
    device->Release();

    if (g_mvVizPso == nullptr)
    {
        g_mvVizFailed = true; // cleared at the end of a successful setup
        static const char shader[] = R"(
Texture2D<float4> Src : register(t0);
Texture2D<float4> Scene : register(t1);
RWTexture2D<float4> Dst : register(u0);
cbuffer Params : register(b0)
{
    uint SrcW;
    uint SrcH;
    uint DstW;
    uint DstH;
    uint Scale;   // pixels of motion at full saturation
    uint Mode;    // 0: the motion field; 1: the residual view (the decode wrote gate weight, residual length)
};
[numthreads(8, 8, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= DstW || id.y >= DstH)
        return;
    uint2 s = min(uint2((id.x * SrcW) / DstW, (id.y * SrcH) / DstH), uint2(SrcW - 1, SrcH - 1));
    if (Mode == 1)
    {
        // Red = how much of the object residual is ADDED (the gate weight, before the scale slider).
        // Green = how large the residual is, full at Scale px, whether or not the gate let it through.
        // So: red specks on still walls/ground = noise getting through (shimmer); yellow/red on a walking
        // NPC or a car = real motion carried; green-only on a mover = real motion the gate threw away.
        // Over a dim grey copy of the picture, so it is clear what is being looked at.
        const float2 v = Src.Load(int3(s, 0)).rg;
        const float3 c = Scene.Load(int3(id.xy, 0)).rgb;
        const float l = max(dot(c, float3(0.2126, 0.7152, 0.0722)), 0.0);
        const float3 base = (l / (1.0 + l)) * 0.55;
        if (v.x < 0.0)
        {
            Dst[id.xy] = float4(base * 0.6, 1.0); // not judged: sky, or no object data
            return;
        }
        const float red = saturate(v.x);
        const float green = saturate(v.y / max(float(Scale), 1.0));
        const float a = saturate(max(red, green) * 0.9);
        Dst[id.xy] = float4(lerp(base, float3(red, green, 0.0), a), 1.0);
        return;
    }
    const float2 mv = Src.Load(int3(s, 0)).rg / max(float(Scale), 1.0);
    // Red right, green left, blue vertical. Flat grey at exactly zero, so a field that ought to be still
    // and is not -- which is what other-eye motion looks like with the head held still -- stands out.
    const float3 tint = float3(saturate(mv.x), saturate(-mv.x), saturate(abs(mv.y)));
    Dst[id.xy] = float4(all(mv == 0.0) ? float3(0.15, 0.15, 0.15) : tint, 1.0);
}
)";

        ID3DBlob* cs = nullptr;
        ID3DBlob* errors = nullptr;
        if (FAILED(D3DCompile(shader, sizeof(shader) - 1, "DuniaTaaMvViz", nullptr, nullptr, "CSMain", "cs_5_0",
                              D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &cs, &errors)))
        {
            if (errors)
            {
                LOG_ERROR("DuniaTaa: MV view compile failed (not retried): {}",
                          (const char*) errors->GetBufferPointer());
                errors->Release();
            }
            return false;
        }
        if (errors)
            errors->Release();

        D3D12_DESCRIPTOR_RANGE ranges[2] {};
        ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        ranges[0].NumDescriptors = 2;
        ranges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
        ranges[1].NumDescriptors = 1;
        ranges[1].OffsetInDescriptorsFromTableStart = 2;

        D3D12_ROOT_PARAMETER params[2] {};
        params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        params[0].DescriptorTable.NumDescriptorRanges = 2;
        params[0].DescriptorTable.pDescriptorRanges = ranges;
        params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        params[1].Constants.Num32BitValues = 6;
        params[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

        D3D12_ROOT_SIGNATURE_DESC rootDesc {};
        rootDesc.NumParameters = 2;
        rootDesc.pParameters = params;
        ID3DBlob* rsBlob = nullptr;
        if (FAILED(D3D12SerializeRootSignature(&rootDesc, D3D_ROOT_SIGNATURE_VERSION_1, &rsBlob, nullptr)))
        {
            cs->Release();
            return false;
        }
        const auto rsHr = device->CreateRootSignature(0, rsBlob->GetBufferPointer(), rsBlob->GetBufferSize(),
                                                      IID_PPV_ARGS(&g_mvVizRs));
        rsBlob->Release();
        if (FAILED(rsHr))
        {
            cs->Release();
            return false;
        }

        D3D12_COMPUTE_PIPELINE_STATE_DESC pso {};
        pso.pRootSignature = g_mvVizRs;
        pso.CS = { cs->GetBufferPointer(), cs->GetBufferSize() };
        const auto psoHr = device->CreateComputePipelineState(&pso, IID_PPV_ARGS(&g_mvVizPso));
        cs->Release();
        if (FAILED(psoHr))
            return false;

        D3D12_DESCRIPTOR_HEAP_DESC heap {};
        heap.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        heap.NumDescriptors = 3;
        heap.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        if (FAILED(device->CreateDescriptorHeap(&heap, IID_PPV_ARGS(&g_mvVizHeap))))
            return false;
        g_mvVizFailed = false;
        LOG_INFO("DuniaTaa: motion view pass ready (motion field, and the residual view over a copy of the picture)");
    }

    const bool residual = g_residualView.load(std::memory_order_relaxed);
    const auto motionDesc = g_motion->GetDesc();
    const auto outDesc = g_output->GetDesc();
    if (residual && g_mvVizScene == nullptr)
    {
        D3D12_HEAP_PROPERTIES heapProps = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
        auto sceneDesc = outDesc;
        sceneDesc.Flags = D3D12_RESOURCE_FLAG_NONE;
        if (FAILED(device->CreateCommittedResource(&heapProps, D3D12_HEAP_FLAG_NONE, &sceneDesc,
                                                   D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                   IID_PPV_ARGS(&g_mvVizScene))))
        {
            g_mvVizScene = nullptr;
            g_mvVizFailed = true;
            LOG_ERROR("DuniaTaa: residual view: could not create the {}x{} picture copy (not retried)", outDesc.Width,
                      outDesc.Height);
            return false;
        }
        g_mvVizViewsValid = false;
    }
    if (!g_mvVizViewsValid)
    {
        D3D12_SHADER_RESOURCE_VIEW_DESC srv {};
        srv.Format = DXGI_FORMAT_R16G16_FLOAT;
        srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srv.Texture2D.MipLevels = 1;
        D3D12_SHADER_RESOURCE_VIEW_DESC sceneSrv = srv;
        sceneSrv.Format = UavColorFormat(outDesc.Format);

        D3D12_UNORDERED_ACCESS_VIEW_DESC uav {};
        uav.Format = UavColorFormat(outDesc.Format);
        uav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;

        const UINT step = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        auto cpu = g_mvVizHeap->GetCPUDescriptorHandleForHeapStart();
        device->CreateShaderResourceView(g_motion, &srv, cpu);
        cpu.ptr += step;
        // A null view until the residual view first runs: mode 0 never reads it.
        device->CreateShaderResourceView(g_mvVizScene, &sceneSrv, cpu);
        cpu.ptr += step;
        device->CreateUnorderedAccessView(g_output, nullptr, &uav, cpu);
        g_mvVizViewsValid = true;
        LOG_INFO("DuniaTaa: motion vector view {}x{} -> {}x{}", motionDesc.Width, motionDesc.Height, outDesc.Width,
                 outDesc.Height);
    }

    if (residual)
    {
        // DLSS's picture, copied out before it is painted over. g_output is UNORDERED_ACCESS here -- DLSS has
        // just written it on this list -- and the copy is ours and at rest in COPY_DEST.
        Barrier(cmd, g_output, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
        cmd->CopyResource(g_mvVizScene, g_output);
        Barrier(cmd, g_output, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        Barrier(cmd, g_mvVizScene, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    }

    Barrier(cmd, g_motion, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    cmd->SetComputeRootSignature(g_mvVizRs);
    cmd->SetPipelineState(g_mvVizPso);
    cmd->SetDescriptorHeaps(1, &g_mvVizHeap);
    cmd->SetComputeRootDescriptorTable(0, g_mvVizHeap->GetGPUDescriptorHandleForHeapStart());
    // The residual view saturates green at 4 px of residual: the 8-bit step at NPC range is ~2.7 px.
    const UINT constants[] = { (UINT) motionDesc.Width, motionDesc.Height, (UINT) outDesc.Width, outDesc.Height,
                               residual ? 4u
                                        : (UINT) std::max(1, Config::Instance()->DuniaTaaMotionViewScale.value_or_default()),
                               residual ? 1u : 0u };
    cmd->SetComputeRoot32BitConstants(1, 6, constants, 0);
    cmd->Dispatch(((UINT) outDesc.Width + 7) / 8, (outDesc.Height + 7) / 8, 1);
    Barrier(cmd, g_motion, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    if (residual)
        Barrier(cmd, g_mvVizScene, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST);
    return true;
}

// `view` non-null selects the matrix path: motion is computed from our own depth and the game's own camera
// constants, and the game's packed vectors are not read at all -- which is why that path needs no motion-pass
// search, no dispatch copy and no heap mirroring. `src` may then be null.
// `objects`: with a view AND a fresh copy of the game's vectors in `src`, the decode carries the game's
// per-object motion over onto the same-eye camera motion (see the shader). Without a copy it is matrix-only.
bool DecodeGameMv(ID3D12GraphicsCommandList* cmd, ID3D12Resource* src, ID3D12Resource* prev, bool chain,
                  D3D12_RESOURCE_STATES srcBefore, ID3D12Resource* depth = nullptr,
                  const ViewFrame* view = nullptr, bool objects = false, bool sameEye = false)
{
    if (cmd == nullptr || g_motion == nullptr)
        return false;
    if (src == nullptr && view == nullptr)
        return false;
    if (view != nullptr && depth == nullptr)
        return false;
    if (chain && prev == nullptr)
        chain = false;

    const auto srcDesc = src != nullptr ? src->GetDesc() : g_motion->GetDesc();
    const auto motionDesc = g_motion->GetDesc();
    // The pass runs at the size DLSS reads, not at the source's: Dunia writes its vectors at half the
    // render size, and dispatching over those filled a quarter of the buffer and left the rest stale.
    const UINT width = (UINT) motionDesc.Width;
    const UINT height = motionDesc.Height;
    const UINT halfRes = (UINT) srcDesc.Width + 32 < width ? 1u : 0u;

    // Creating the views every frame is driver work for a result that never changes while the same
    // resources are in play. Invalidated when any of them is replaced.
    // The pair alternates every Present, so the views are rebuilt whenever it changes -- four descriptor
    // writes, against a wrong texture being read if they were cached across the swap.
    if (!g_mvViewsValid || g_mvViewedSrc != src || g_mvViewedPrev != prev || g_mvViewedDepth != depth)
    {
        ID3D12Device* device = nullptr;
        if (FAILED(cmd->GetDevice(IID_PPV_ARGS(&device))) || device == nullptr)
            return false;
        device->Release();
        if (!EnsureMvDecode(device))
            return false;

        D3D12_SHADER_RESOURCE_VIEW_DESC srv {};
        srv.Format = MotionSrvFormat(srcDesc.Format);
        srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srv.Texture2D.MipLevels = 1;

        D3D12_UNORDERED_ACCESS_VIEW_DESC uav {};
        uav.Format = DXGI_FORMAT_R16G16_FLOAT;
        uav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;

        D3D12_CPU_DESCRIPTOR_HANDLE cpu = g_mvHeap->GetCPUDescriptorHandleForHeapStart();
        // t0 is unused by the matrix path, and a null descriptor reads as zero rather than as undefined.
        device->CreateShaderResourceView(src, src != nullptr ? &srv : &srv, cpu);
        g_mvViewedSrc = src;
        g_mvViewedPrev = prev;
        g_mvViewedDepth = depth;
        // t1 is the depth the matrix path unprojects; t2 (the object flags) stays null, so every pixel takes
        // the matrix and nothing falls back to the game's quantised vectors. That is FC5's "matrix everywhere"
        // state, which is what proved the cause there, and it is the honest first version here.
        D3D12_SHADER_RESOURCE_VIEW_DESC nullSrv = srv;
        nullSrv.Format = DXGI_FORMAT_R32_FLOAT;
        cpu.ptr += g_mvDescriptorSize;
        const DXGI_FORMAT depthSrv = depth != nullptr ? DepthSrvFormat(depth->GetDesc().Format)
                                                      : DXGI_FORMAT_UNKNOWN;
        if (depth != nullptr && depthSrv != DXGI_FORMAT_UNKNOWN)
        {
            D3D12_SHADER_RESOURCE_VIEW_DESC dsrv {};
            dsrv.Format = depthSrv;
            dsrv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
            dsrv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            dsrv.Texture2D.MipLevels = 1;
            device->CreateShaderResourceView(depth, &dsrv, cpu);
        }
        else
        {
            if (depth != nullptr)
            {
                static bool logged = false;
                if (!logged)
                {
                    logged = true;
                    LOG_WARN("DuniaTaa: matrix motion: depth format {} has no readable twin, so the matrix path "
                             "cannot unproject it",
                             (UINT) depth->GetDesc().Format);
                }
            }
            device->CreateShaderResourceView(nullptr, &nullSrv, cpu);
        }
        nullSrv.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        cpu.ptr += g_mvDescriptorSize;
        device->CreateShaderResourceView(nullptr, &nullSrv, cpu);
        // t3: the previous Present's copy, for chaining.
        cpu.ptr += g_mvDescriptorSize;
        device->CreateShaderResourceView(prev != nullptr ? prev : src, &srv, cpu);
        cpu.ptr += g_mvDescriptorSize;
        device->CreateUnorderedAccessView(g_motion, nullptr, &uav, cpu);
        g_mvViewsValid = true;
    }

    MvConstants constants {};
    constants.size[0] = width;
    constants.size[1] = height;
    constants.halfRes = halfRes;
    constants.chain = chain ? 1u : 0u;
    constants.smoothField = sameEye ? 1u : 0u;
    constants.useMatrix = view != nullptr ? 1u : 0u;
    constants.objects = (view != nullptr && src != nullptr && objects) ? 1u : 0u;
    if (view != nullptr)
    {
        BuildPrevClipMatrix(*view, view->prevSameEye, constants.matrix, constants.unproject);
        float unusedUnproject[2];
        BuildPrevClipMatrix(*view, view->prevOtherEye, constants.other, unusedUnproject);
        // Live A/B (overlay). The jitter work changed the OTHER-eye baseline by 2*jitter = 0.5 px,
        // alternating in sign between the eyes: the derivation and the offline check say the new value
        // matches the game's own vectors, but the first daytime comparison against the older build found MORE
        // haloing, a black silhouette around the hands, and dirt edges moving right in the right eye and
        // left in the left -- a per-eye horizontal shift, which is exactly what this term is. Mode 1 puts
        // the old baseline back (+jx instead of -jx on the x and y rows); everything else stays new.
        if (g_baselineOld.load(std::memory_order_relaxed) != 0)
        {
            const float jx = view->proj[2], jy = view->proj[6];
            for (int c = 0; c < 4; ++c)
            {
                constants.other[0 * 4 + c] += 2.0f * jx * constants.other[3 * 4 + c];
                constants.other[1 * 4 + c] += 2.0f * jy * constants.other[3 * 4 + c];
            }
        }
    }
    // `other` stays the real other-eye baseline even then: object-flagged pixels are copied verbatim from the
    // G-buffer and are still other-eye. See the shader.
    constants.sameEye = sameEye ? 1u : 0u;
    constants.sameEyeFloor = kToleranceFloorSameEyeFine;
    constants.sameEyeResidualScale = 2.0f;
    constants.objectScale = g_objectScale.load(std::memory_order_relaxed);
    constants.objectFloor = g_objectFloor.load(std::memory_order_relaxed);
    constants.objectSpeedTerm = g_objectSpeedTerm.load(std::memory_order_relaxed);
    // The chained lookup needs the previous Present's copy; without one (first frame, after a menu) it is off.
    constants.fastChain = (g_objectFastChain.load(std::memory_order_relaxed) && prev != nullptr) ? 1u : 0u;
    constants.debugView = g_residualView.load(std::memory_order_relaxed) ? 1u : 0u;
    constants.objectNearOnly = g_objectNearOnly.load(std::memory_order_relaxed) ? 1u : 0u;
    constants.objectBgReject = g_objectBgReject.load(std::memory_order_relaxed) ? 1u : 0u;
    constants.objectRadius = (UINT) std::clamp(g_objectRadius.load(std::memory_order_relaxed), 1, 6);

    Barrier(cmd, src, srcBefore, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    cmd->SetComputeRootSignature(g_mvRs);
    cmd->SetPipelineState(g_mvPso);
    cmd->SetDescriptorHeaps(1, &g_mvHeap);
    cmd->SetComputeRootDescriptorTable(0, g_mvHeap->GetGPUDescriptorHandleForHeapStart());
    cmd->SetComputeRoot32BitConstants(1, kMvConstants, &constants, 0);
    cmd->Dispatch((width + 7) / 8, (height + 7) / 8, 1);
    Barrier(cmd, src, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, srcBefore);
    return true;
}

bool SameFormatFamily(DXGI_FORMAT a, DXGI_FORMAT b) { return UavColorFormat(a) == UavColorFormat(b); }

bool CopyOutputToDest(ID3D12GraphicsCommandList* cmd, ID3D12Resource* dest, D3D12_RESOURCE_STATES destBefore)
{
    // One read of g_output: an eye-resolution rebuild can swap it on another thread (the upscale-draw hook calls this
    // outside g_spliceLock), and the size checked must be the size copied. A swapped-out texture stays alive (retired).
    ID3D12Resource* const output = g_output;
    if (cmd == nullptr || dest == nullptr || output == nullptr)
        return false;

    const auto outDesc = output->GetDesc();
    const auto destDesc = dest->GetDesc();
    if (outDesc.Width != destDesc.Width || outDesc.Height != destDesc.Height)
        return false;
    if (!SameFormatFamily(outDesc.Format, destDesc.Format))
        return false;

    Barrier(cmd, output, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
    Barrier(cmd, dest, destBefore, D3D12_RESOURCE_STATE_COPY_DEST);
    g_skipCopyHook++;
    cmd->CopyResource(dest, output);
    g_skipCopyHook--;
    Barrier(cmd, dest, D3D12_RESOURCE_STATE_COPY_DEST, destBefore);
    Barrier(cmd, output, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    return true;
}

// The A/B split: DLSS output over the left half of the destination only, so the right half keeps the
// game's own upscale of the same frame.
bool CopyOutputLeftHalf(ID3D12GraphicsCommandList* cmd, ID3D12Resource* dest, D3D12_RESOURCE_STATES destBefore)
{
    ID3D12Resource* const output = g_output; // one read, as CopyOutputToDest
    if (cmd == nullptr || dest == nullptr || output == nullptr)
        return false;

    const auto outDesc = output->GetDesc();
    const auto destDesc = dest->GetDesc();
    if (outDesc.Width != destDesc.Width || outDesc.Height != destDesc.Height)
        return false;
    if (!SameFormatFamily(outDesc.Format, destDesc.Format))
        return false;

    Barrier(cmd, output, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
    Barrier(cmd, dest, destBefore, D3D12_RESOURCE_STATE_COPY_DEST);
    g_skipCopyHook++;
    const D3D12_TEXTURE_COPY_LOCATION dst = CD3DX12_TEXTURE_COPY_LOCATION(dest, 0);
    const D3D12_TEXTURE_COPY_LOCATION src = CD3DX12_TEXTURE_COPY_LOCATION(output, 0);
    const D3D12_BOX box { 0, 0, 0, (UINT) outDesc.Width / 2, outDesc.Height, 1 };
    cmd->CopyTextureRegion(&dst, 0, 0, 0, &src, &box);
    g_skipCopyHook--;
    Barrier(cmd, dest, D3D12_RESOURCE_STATE_COPY_DEST, destBefore);
    Barrier(cmd, output, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    return true;
}

bool EvaluateOnList(ID3D12GraphicsCommandList* cmd, ID3D12CommandQueue* queue, ID3D12Resource* color,
                    ID3D12Resource* depth, bool allowCreate, bool bindHeaps, bool copyBack,
                    D3D12_RESOURCE_STATES colorBefore,
                    D3D12_RESOURCE_STATES destBefore = D3D12_RESOURCE_STATE_RENDER_TARGET, int eyeSlot = 0)
{
    if (cmd == nullptr || color == nullptr || depth == nullptr || g_inEvaluate > 0)
        return false;
    EvaluateScope scope;
    (void) destBefore;
    (void) bindHeaps;

    ID3D12Device* device = nullptr;
    if (FAILED(cmd->GetDevice(IID_PPV_ARGS(&device))) || device == nullptr)
        return false;
    device->Release();

    const auto colorDesc = color->GetDesc();
    const UINT width = (UINT) colorDesc.Width;
    const UINT height = colorDesc.Height;

    if (!IsEvalSize(width, height))
        return false;
    if (!IsHdrFormat(colorDesc.Format) && !TonemappedMode())
        return false;

    if (allowCreate)
    {
        if (!EnsureScratch(device, queue, width, height, width, height, colorDesc.Format) || !InitNgx(device))
            return false;
    }
    else if (g_output == nullptr || g_handle == nullptr || g_createdInW != width || g_createdInH != height)
        return false;

    if (allowCreate)
    {
        if (!CreateDlss(cmd, width, height, width, height, colorDesc.Format) || g_handle == nullptr ||
            g_params == nullptr)
            return false;
    }
    else if (g_handle == nullptr || g_params == nullptr)
        return false;

    ID3D12Resource* motion = g_motion;
    const int mvMode = MvMode();
    const bool mvInverted = mvMode == 2 || mvMode == 4;
    const bool mvPixelSpace = mvMode >= 3;
    bool haveGameMv = false;
    bool mvDecoded = false;
    ID3D12Resource* mvRestore = nullptr;
    D3D12_RESOURCE_STATES mvRestoreState = D3D12_RESOURCE_STATE_COMMON;
    // The decode writes pixel-space vectors, so DLSS needs no further scaling. An already-float
    // buffer is handed over untouched and still normalised, so DLSS scales it by the frame size
    // instead -- the same arithmetic the decode shader would have done, for free.
    float mvScaleX = 1.0f;
    float mvScaleY = 1.0f;
    // What DLSS is told the jitter was, in pixels. Zero unless a jitter mode is on AND this frame's per-view
    // struct was read: a jitter reported for a frame it does not belong to is worse than none.
    float dlssJitter[2] = { 0.0f, 0.0f };

    // Real motion first: our own copy of the game's texture, taken right after the pass wrote it. Its
    // state is ours, so nothing is guessed here, and the decode writes pixel offsets so DLSS needs no
    // scaling. Each copy is consumed exactly once, so a stale one is never decoded.
    if (g_realMotionOn.load(std::memory_order_relaxed) && MotionModeIsMatrix())
    {
        // Matrix motion: our own depth through the game's own camera constants, chained over two Presents by
        // parity to reach the SAME eye. No packed vectors are read, so there is nothing to copy and nothing to
        // search for. Every way of doing nothing says so once.
        ViewFrame view {};
        const bool patching = SameEyeWanted() || JitterModeNow() >= 2;
        const PvArgs pvArgs = CurrentPvArgs();
        PvResult pvResult = PvResult::Idle;
        PvInfo pvInfo;
        const bool haveView = g_viewConstants.Update(SceneW(), SceneH(), g_presentIndex.load(), &view,
                                                     patching ? &pvArgs : nullptr, &pvResult, &pvInfo);
        if (patching)
            NotePerViewResult(pvResult, pvInfo);
        // The jitter of the frame DLSS is about to be handed, read from the projection the GPU rendered it
        // with. FC5's convention, validated there: DLSS jitter = the image's shift in pixels, x right, y down.
        if (view.cur.valid && JitterModeNow() != 0 && JitterModeNow() != 3)
        {
            const int sign = std::clamp(g_jitterSign.load(std::memory_order_relaxed), 0, 3);
            dlssJitter[0] = kJitterSigns[sign][0] * view.jitterPx[0];
            dlssJitter[1] = kJitterSigns[sign][1] * view.jitterPx[1];
            g_jitterReported.fetch_add(1, std::memory_order_relaxed);
            if (JitterModeNow() >= 2)
            {
                float last[2];
                for (int k = 0; k < 2; ++k)
                {
                    const uint32_t bits = g_jitterLastX[k].load(std::memory_order_relaxed);
                    std::memcpy(&last[k], &bits, sizeof(bits));
                }
                if (std::abs(view.jitterPx[0] - last[0]) > 2e-3f && std::abs(view.jitterPx[0] - last[1]) > 2e-3f)
                    g_jitterMismatch.fetch_add(1, std::memory_order_relaxed);
            }
            static std::atomic<bool> loggedJitter { false };
            if (!loggedJitter.exchange(true, std::memory_order_relaxed))
                LOG_INFO("DuniaTaa: jitter: DLSS is now told ({:+.3f}, {:+.3f}) px, from this frame's projection "
                         "(mode {}, sign variant {}). The game's own is +-0.25 px; ours is a Halton sample within "
                         "half a pixel",
                         dlssJitter[0], dlssJitter[1], JitterModeNow(), sign);
        }
        // The object path wants this Present's copy of the game's vectors beside the matrix. The copy is
        // taken at the dispatch exactly as in the game's-own mode (NeedsGameVectors includes this mode, so
        // the pipeline is learned and the copies flow); a Present with no fresh copy decodes matrix-only
        // and says so once, rather than reading a stale one.
        const bool objectsMode = MotionObjectsOn();
        ID3D12Resource* gameCopy = nullptr;
        ID3D12Resource* gameCopyPrev = nullptr; // the previous Present's copy, for the fast-mover chain
        bool sameEyeCopy = false;
        if (objectsMode)
        {
            const MvStagingPair::Snapshot stage = g_mvStaging.Read();
            if (stage.cur != nullptr && g_mvStaging.TakeFresh())
            {
                gameCopy = stage.cur;
                gameCopyPrev = stage.paired ? stage.prev : nullptr;
                sameEyeCopy = g_mvCopySameEye.load(std::memory_order_acquire);
            }
            else
            {
                static bool loggedNoCopy = false;
                if (!loggedNoCopy)
                {
                    loggedNoCopy = true;
                    LOG_WARN("DuniaTaa: object motion: no fresh copy of the game's vectors at this splice -- "
                             "matrix-only this Present. If this is the only object line in the log, the copies "
                             "never started: check for the 'motion pass pipeline ... learned' line");
                }
            }
        }
        const bool decoded = haveView && g_motion != nullptr && depth != nullptr &&
                             DecodeGameMv(cmd, gameCopy, gameCopyPrev, false,
                                          D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, depth, &view,
                                          gameCopy != nullptr, sameEyeCopy);
        if (decoded)
        {
            haveGameMv = true;
            mvDecoded = true;
            g_mvDecodes.fetch_add(1, std::memory_order_relaxed);
            if (gameCopy != nullptr)
                g_mvObjectDecodes.fetch_add(1, std::memory_order_relaxed);
            if (sameEyeCopy)
                g_mvSameEyeDecodes.fetch_add(1, std::memory_order_relaxed);
            static bool loggedOnce = false;
            if (!loggedOnce)
            {
                loggedOnce = true;
                LOG_INFO("DuniaTaa: matrix motion running -- unproject ({:.8g}, {:.8g}), camera moved "
                         "({:.4f}, {:.4f}, {:.4f}) since the same eye's last frame",
                         -1.0f / view.proj[11], -view.proj[10] / view.proj[11],
                         view.cur.camPos[0] - view.prevSameEye.camPos[0],
                         view.cur.camPos[1] - view.prevSameEye.camPos[1],
                         view.cur.camPos[2] - view.prevSameEye.camPos[2]);
            }
            // Logged on the first object decode with a value only the real computation produces: the
            // camera offset to the OTHER eye's frame is the interocular distance, a few centimetres,
            // sideways -- where the same-eye delta above is near zero with the head still.
            static bool loggedObjects = false;
            if (gameCopy != nullptr && !loggedObjects)
            {
                loggedObjects = true;
                LOG_INFO("DuniaTaa: object motion running -- the game's vectors are judged against camera "
                         "motion to the OTHER eye's frame, camera offset to it ({:.4f}, {:.4f}, {:.4f}); "
                         "|offset| should be about the IPD",
                         view.cur.camPos[0] - view.prevOtherEye.camPos[0],
                         view.cur.camPos[1] - view.prevOtherEye.camPos[1],
                         view.cur.camPos[2] - view.prevOtherEye.camPos[2]);
            }
        }
        else
        {
            // The `!logged` test belongs INSIDE the branch, never in its condition. With it in the condition
            // the chain falls through once a reason has been logged and reports the NEXT reason instead -- which
            // is how "the decode pass itself failed" appeared for a run whose only fault was missing view
            // constants, and sent the diagnosis one layer too deep.
            static bool loggedView = false, loggedDepth = false, loggedOut = false, loggedDec = false;
            if (!haveView)
            {
                if (!loggedView)
                {
                    loggedView = true;
                    LOG_WARN("DuniaTaa: matrix motion: no view constants yet -- nothing validated as a per-view "
                             "struct, or no state from two Presents ago");
                }
            }
            else if (depth == nullptr)
            {
                if (!loggedDepth)
                {
                    loggedDepth = true;
                    LOG_WARN("DuniaTaa: matrix motion: no eye depth at the splice");
                }
            }
            else if (g_motion == nullptr)
            {
                if (!loggedOut)
                {
                    loggedOut = true;
                    LOG_WARN("DuniaTaa: matrix motion: g_motion missing");
                }
            }
            else if (!loggedDec)
            {
                loggedDec = true;
                LOG_WARN("DuniaTaa: matrix motion: the decode pass itself failed");
            }
        }
    }
    else if (g_realMotionOn.load(std::memory_order_relaxed))
    {
        // Chaining needs both buffers filled; until then fall back to the single hop rather than reading a
        // buffer that has never been written.
        //
        // The snapshot is locked inside MvStagingPair, which is the only way to read the pair -- the
        // pointers are private. Reading them unlocked is what crashed on entering the menu.
        const MvStagingPair::Snapshot stage = g_mvStaging.Read();
        ID3D12Resource* cur = stage.cur;
        ID3D12Resource* prev = stage.prev;
        const bool chain = g_realMotionMode.load(std::memory_order_relaxed) == kMotionChained && stage.paired;
        const bool haveStage = cur != nullptr && prev != nullptr;
        const bool haveOutput = g_motion != nullptr;
        const bool fresh = haveStage && haveOutput && g_mvStaging.TakeFresh();
        bool decoded = false;
        if (fresh)
        {
            decoded = DecodeGameMv(cmd, cur, prev, chain, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            if (decoded)
            {
                haveGameMv = true;
                mvDecoded = true;
                g_mvDecodes.fetch_add(1, std::memory_order_relaxed);
            }
        }
        // 600 copies and 0 decodes twice over, with nothing in the log to say which of these it was.
        // Logged once per reason, so a silent skip can never cost another run.
        if (!decoded)
        {
            static bool loggedStage = false, loggedOutput = false, loggedFresh = false, loggedDecode = false;
            if (!haveStage && !loggedStage)
            {
                loggedStage = true;
                LOG_WARN("DuniaTaa: real motion: no staging texture yet at the splice");
            }
            else if (!haveOutput && !loggedOutput)
            {
                loggedOutput = true;
                LOG_WARN("DuniaTaa: real motion: g_motion missing at the splice");
            }
            else if (haveStage && haveOutput && !fresh && !loggedFresh)
            {
                loggedFresh = true;
                LOG_WARN("DuniaTaa: real motion: no fresh copy at the splice -- the dispatch copy and this "
                         "splice are not in the order assumed");
            }
            else if (fresh && !loggedDecode)
            {
                loggedDecode = true;
                LOG_WARN("DuniaTaa: real motion: the decode pass itself failed");
            }
        }
    }

    bool mvSizeMismatch = false;
    if (!haveGameMv && mvMode != 0 && g_gameMv != nullptr)
    {
        const auto mvDesc = g_gameMv->GetDesc();
        mvSizeMismatch = !MvSizeOk((UINT) mvDesc.Width, mvDesc.Height);
        // The tracked render-target flag latches and can outlive the binding it described, and a
        // resource without ALLOW_RENDER_TARGET can never legally be in RENDER_TARGET state, so the
        // guess is constrained to states this resource is actually capable of holding.
        const auto mvState =
            (g_gameMvIsRt && (mvDesc.Flags & D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET) != 0)
                ? D3D12_RESOURCE_STATE_RENDER_TARGET
                : ((mvDesc.Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS) != 0
                       ? D3D12_RESOURCE_STATE_UNORDERED_ACCESS
                       : D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        if (MvSizeOk((UINT) mvDesc.Width, mvDesc.Height))
        {
            if (NeedsMvDecode(mvDesc.Format))
            {
                haveGameMv = g_motion != nullptr && DecodeGameMv(cmd, g_gameMv, nullptr, false, mvState);
                mvDecoded = haveGameMv;
            }
            else
            {
                Barrier(cmd, g_gameMv, mvState, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                mvRestore = g_gameMv;
                mvRestoreState = mvState;
                motion = g_gameMv;
                // Scaling by the frame size assumes UV-space vectors. If the engine already stores
                // pixel offsets, that multiplies them by a few thousand and the result is noise.
                mvScaleX = mvPixelSpace ? 1.0f : (float) mvDesc.Width;
                mvScaleY = mvPixelSpace ? 1.0f : (float) mvDesc.Height;
                haveGameMv = true;
            }
        }
    }

    // Dunia's motion vector sign convention is unverified against DLSS's.
    if (mvInverted)
    {
        mvScaleX = -mvScaleX;
        mvScaleY = -mvScaleY;
    }

    if (!haveGameMv && g_uavGpu != nullptr && g_motion != nullptr)
    {
        cmd->SetDescriptorHeaps(1, &g_uavGpu);
        const float zero[4] = {};
        cmd->ClearUnorderedAccessViewFloat(g_uavGpu->GetGPUDescriptorHandleForHeapStart(),
                                           g_uavCpu->GetCPUDescriptorHandleForHeapStart(), g_motion, zero, 0, nullptr);
    }

    static int loggedMvState = -1;
    const int mvLogState = haveGameMv ? (mvDecoded ? 10 : 20) + mvMode : (mvSizeMismatch ? -1 : 0);
    if (loggedMvState != mvLogState)
    {
        loggedMvState = mvLogState;
        if (haveGameMv)
        {
            const auto mvDesc = g_gameMv->GetDesc();
            LOG_INFO("DuniaTaa: game motion vectors {}x{} fmt {} flags {:X} ({}, {} space), sign {}, keeping history",
                     mvDesc.Width, mvDesc.Height, (UINT) mvDesc.Format, (UINT) mvDesc.Flags,
                     mvDecoded ? "decoded" : "direct", mvPixelSpace ? "pixel" : "uv",
                     mvInverted ? "inverted" : "standard");
        }
        else if (mvSizeMismatch)
        {
            // Silent otherwise, and it looks exactly like the flicker coming and going by itself.
            const auto mvDesc = g_gameMv->GetDesc();
            LOG_INFO("DuniaTaa: motion buffer is {}x{} but the frame is {}x{}, falling back to dummy vectors",
                     mvDesc.Width, mvDesc.Height, g_pinnedW, g_pinnedH);
        }
        else
        {
            LOG_INFO("DuniaTaa: dummy motion vectors, resetting DLAA/NR history every frame");
        }
    }

    // The mode is switchable from the overlay mid-scene; drop the history so the comparison starts clean.
    static int lastMvMode = -1;
    const bool mvModeChanged = lastMvMode != mvMode;
    lastMvMode = mvMode;

    // A reset request applies to every eye's history, but each is only reset when that eye next runs.
    const int slot = (eyeSlot == 1 && g_handleEye2 != nullptr) ? 1 : 0;
    if (g_needReset || mvModeChanged)
    {
        g_slotNeedsReset[0] = true;
        g_slotNeedsReset[1] = true;
    }
    g_needReset = false;
    const bool resetHistory = !haveGameMv || g_slotNeedsReset[slot];
    g_slotNeedsReset[slot] = false;

    g_params->Set(NVSDK_NGX_Parameter_Jitter_Offset_X, dlssJitter[0]);
    g_params->Set(NVSDK_NGX_Parameter_Jitter_Offset_Y, dlssJitter[1]);
    g_params->Set(NVSDK_NGX_Parameter_MV_Scale_X, mvScaleX);
    g_params->Set(NVSDK_NGX_Parameter_MV_Scale_Y, mvScaleY);
    g_params->Set(NVSDK_NGX_Parameter_DLSS_Exposure_Scale, 1.0f);
    g_params->Set(NVSDK_NGX_Parameter_Reset, resetHistory ? 1 : 0);
    // Neural Rendering runs inside this evaluate and keeps a history of its own; this tells it which eye's
    // to use, so it pairs exactly like the DLSS feature chosen above (slot is 0 for both eyes when only one
    // feature exists). Without it NR's single history reprojected each eye onto the other: the road
    // "quicksand" with NR on, worse when aiming.
    g_params->Set(DlssNr::kNrEyeParam, (unsigned int) slot);
    // Live rebuild in progress: the staged recreate's first step writes the OLD feature's keys into this very
    // table (it is the create table for passthrough DLSS), and its second and third build the new feature
    // from it. Setting ours before every evaluate means the create always reads the new size and keys,
    // whichever eye's step ran last.
    if (g_rebuildPending)
        SetCreateKeys(g_createdInW, g_createdInH, g_createdOutW, g_createdOutH);
    g_params->Set(NVSDK_NGX_Parameter_Width, width);
    g_params->Set(NVSDK_NGX_Parameter_Height, height);
    g_params->Set(NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Width, width);
    g_params->Set(NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Height, height);
    g_params->Set(NVSDK_NGX_Parameter_Depth, depth);
    g_params->Set(NVSDK_NGX_Parameter_Color, color);
    g_params->Set(NVSDK_NGX_Parameter_MotionVectors, motion);
    g_params->Set(NVSDK_NGX_Parameter_Output, g_output);

    State::Instance().setInputApiName = ApiUpscalerInput::FSR3_DX12;

    Barrier(cmd, color, colorBefore, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

    TimeMark(cmd, 1); // GPU timing: everything above (motion reconstruction) ends here
    auto eval = NVSDK_NGX_D3D12_EvaluateFeature(cmd, slot == 1 ? g_handleEye2 : g_handle, g_params, nullptr);
    TimeMark(cmd, 2); // ... and DLSS here

    if (mvRestore != nullptr)
        Barrier(cmd, mvRestore, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, mvRestoreState);

    if (eval != NVSDK_NGX_Result_Success)
    {
        static NVSDK_NGX_Result lastEval = NVSDK_NGX_Result_Fail;
        if (eval != lastEval)
        {
            lastEval = eval;
            LOG_ERROR("DuniaTaa: EvaluateFeature failed: {:X}", (UINT) eval);
        }
        Barrier(cmd, color, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, colorBefore);
        return false;
    }

    // Diagnostics: draws the vectors DLSS has just been handed, over the output. It has to happen here
    // rather than after this function returns, because this path copies the output back immediately and
    // would overwrite it -- the upscale path defers its copy, which is why calling it from the outside
    // works there. Only for the paths that copy back, so the upscale path keeps its single outside call.
    //
    // Deliberately not behind DuniaTaaDiagnostics: setting DuniaTaaUpscaleView = 3 is already an explicit
    // request for it, and needing a second key to make a diagnostic appear has cost test runs
    // before. In tonemapped mode this view showed nothing at all before, whatever the keys were set to.
    if (copyBack && Config::Instance()->DuniaTaaUpscaleView.value_or_default() == 3)
        VisualizeMv(cmd);

    if (copyBack && g_output != nullptr && !IsUpscale())
        CopyOutputToDest(cmd, color, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

    Barrier(cmd, color, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, colorBefore);
    return true;
}

} // namespace

bool DuniaTaa_Dx12::Active()
{
    if (!Config::Instance()->DuniaTaaHost.value_or_default())
        return false;

    static int cached = -1;
    if (cached < 0)
        cached = (_stricmp(State::Instance().gameExe.c_str(), "FarCry6.exe") == 0) ? 1 : 0;

    return cached == 1;
}

bool DuniaTaa_Dx12::WantHooks() { return Active(); }

DuniaTaa_Dx12::GpuTimes DuniaTaa_Dx12::GetGpuTimes()
{
    std::lock_guard<std::mutex> lock(g_timeLock);
    GpuTimes g = g_lastGpuTimes;
    g.on = GpuTimingOn();
    return g;
}

DuniaTaa_Dx12::UpscaleStatus DuniaTaa_Dx12::GetUpscaleStatus()
{
    UpscaleStatus status;
    if (!Active())
        return status;
    status.problem = (UpscaleProblem) g_upscaleProblem.load();
    status.createdInW = g_createdInW;
    status.createdInH = g_createdInH;
    status.createdOutW = g_createdOutW;
    status.createdOutH = g_createdOutH;
    status.seenInW = g_mismatchW.load();
    status.seenInH = g_mismatchH.load();
    status.episode = g_upscaleEpisode.load();
    status.configuredMode = ConfiguredMode();
    status.activeMode = g_handle != nullptr ? g_createdQuality : -1;
    return status;
}

void DuniaTaa_Dx12::Note(ID3D12Resource* color, ID3D12Resource* depth)
{
    if (g_inEvaluate > 0)
        return;
    if (!Active() || color == nullptr || depth == nullptr)
        return;

    if (IsSwapchainBuffer(color) || IsOwnedResource(color) || IsOwnedResource(depth))
        return;

    const auto colorDesc = color->GetDesc();
    const auto depthDesc = depth->GetDesc();
    if (colorDesc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D ||
        depthDesc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D)
        return;

    if (colorDesc.SampleDesc.Count > 1 || depthDesc.SampleDesc.Count > 1)
        return;

    if (colorDesc.Width < 256 || colorDesc.Height < 256)
        return;

    if (std::abs((int) colorDesc.Width - (int) depthDesc.Width) > 32 ||
        std::abs((int) colorDesc.Height - (int) depthDesc.Height) > 32)
        return;

    const uint64_t key = ((uint64_t) colorDesc.Width << 32) | ((uint64_t) colorDesc.Height << 16) |
                         ((uint64_t) colorDesc.Format << 8) | (uint64_t) depthDesc.Format;
    if (g_loggedPairs.insert(key).second)
    {
        LOG_INFO("DuniaTaa: candidate color {}x{} fmt {} flags {:X} depth {}x{} fmt {}", colorDesc.Width,
                 colorDesc.Height, (UINT) colorDesc.Format, (UINT) colorDesc.Flags, depthDesc.Width, depthDesc.Height,
                 (UINT) depthDesc.Format);
    }

    std::lock_guard<std::mutex> lock(g_lock);
    const int newScore = ColorScore(colorDesc);
    if (newScore < 0)
        return;

    // The size every decision below is made on: the target's own, except under the test switch (FakeProbeSize).
    UINT w = (UINT) colorDesc.Width;
    UINT h = colorDesc.Height;
    FakeProbeSize(w, h);

    if (!IsEvalSize(w, h))
    {
        // Otherwise "nothing happens at all" is indistinguishable from "the threshold is too high for
        // this resolution". Only fires on a new largest, so it stays quiet once a scene target exists.
        static uint64_t largestRejected = 0;
        const uint64_t area = (uint64_t) w * h;
        if (area > largestRejected)
        {
            largestRejected = area;
            LOG_INFO("DuniaTaa: largest colour target so far is {}x{}, under the {}x{} scene threshold", w, h,
                     Config::Instance()->DuniaTaaMinEvalWidth.value_or_default(),
                     Config::Instance()->DuniaTaaMinEvalHeight.value_or_default());
        }
        return;
    }

    // The VR eye gates everything below. A square probe pass once rendered at 4096^2, larger than the
    // eye, and "largest HDR target" made it the eye and the pinned scene: F6 then waited for ever. Once
    // the eye copy-out is known (g_vrEyeW, a fact, no size heuristic), nothing larger than the eye is a
    // candidate, the eye learner takes only the eye's own size, and anything learned wrongly before it was known is
    // forgotten (deferred: a recorded list may still use it) and learned again. No copy-out seen: exactly as before.
    if (const UINT eyeW = g_vrEyeW.load(std::memory_order_relaxed), eyeH = g_vrEyeH.load(std::memory_order_relaxed);
        eyeW != 0)
    {
        auto isEye = [&](UINT cw, UINT ch)
        { return std::abs((int) cw - (int) eyeW) <= 32 && std::abs((int) ch - (int) eyeH) <= 32; };
        if ((g_destW != 0 && !isEye(g_destW, g_destH)) ||
            (g_pinnedW != 0 && (g_pinnedW > eyeW + 32 || g_pinnedH > eyeH + 32)))
        {
            static int forgets = 0;
            if (forgets++ < 8)
                LOG_WARN("DuniaTaa: forgetting learned eye {}x{} / scene {}x{} / scaled {}x{}: the VR eye is {}x{} "
                         "(the eye copy-out) -- a pass larger than the eye had been taken for it; learning again",
                         g_destW, g_destH, g_pinnedW, g_pinnedH, g_taaW, g_taaH, eyeW, eyeH);
            const uint64_t present = g_presentIndex.load();
            RetainDeferred(&g_dest, nullptr, present);
            g_destW = g_destH = 0;
            g_destFormat = DXGI_FORMAT_UNKNOWN;
            RetainDeferred(&g_color, nullptr, present);
            RetainDeferred(&g_depth, nullptr, present);
            g_pinnedW = g_pinnedH = 0;
            RetainDeferred(&g_gameMv, nullptr, present);
            g_mvViewsValid = false;
            RetainDeferred(&g_taa, nullptr, present);
            RetainDeferred(&g_taaDepth, nullptr, present);
            g_taaW = g_taaH = 0;
        }
        if (w > eyeW + 32 || h > eyeH + 32)
        {
            static std::unordered_set<uint64_t> loggedLarger;
            if (loggedLarger.size() < 8 && loggedLarger.insert(((uint64_t) w << 32) | h).second)
                LOG_INFO("DuniaTaa: {}x{} fmt {} ignored: larger than the VR eye {}x{} (a probe or other pass, not the "
                         "scene)",
                         w, h, (UINT) colorDesc.Format, eyeW, eyeH);
            return;
        }
    }

    const uint64_t area = (uint64_t) w * h;
    // With the VR eye known, only a target of the eye's own size can be the eye (below) or the pinned full-size scene
    // (at the end); smaller ones can still be the scaled scene, which is judged against that eye.
    const UINT vrEyeW = g_vrEyeW.load(std::memory_order_relaxed), vrEyeH = g_vrEyeH.load(std::memory_order_relaxed);
    const bool eyeSized =
        vrEyeW == 0 || (std::abs((int) w - (int) vrEyeW) <= 32 && std::abs((int) h - (int) vrEyeH) <= 32);

    if (eyeSized && (g_dest == nullptr || area > (uint64_t) g_destW * (uint64_t) g_destH))
    {
        if (IsHdrFormat(colorDesc.Format))
        {
            Retain(&g_dest, color);
            g_destW = w;
            g_destH = h;
            g_destFormat = colorDesc.Format;
            g_destScore = newScore;
        }
    }

    if (g_destW != 0 && IsScaledTaa(w, h, g_destW, g_destH, depth != nullptr))
    {
        if (!IsHdrFormat(colorDesc.Format))
            return;
        // The scene's HDR target is RGBA16F and RT|UAV at every size seen (at 100% and at 70%,
        // fmt 10 flags 5 in 30 logs) -- and RGBA16F is also what the resolve's own signature takes as colour.
        // A half-eye pass at 100% has exactly the 50% scene's shape (half the eye, with a depth of its own) and
        // two HDR targets: fmt 10 RT only, and fmt 26 (R11G11B10) RT|UAV. Since the half-size exclusion was
        // relaxed for bound depth the learner took one or the other for a 50% scene at 100% -- after
        // 70 -> 100 and at a fresh 100% start -- and every size-keyed search followed it.
        if (UavColorFormat(colorDesc.Format) != DXGI_FORMAT_R16G16B16A16_FLOAT ||
            (colorDesc.Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS) == 0)
        {
            // Under g_lock (taken above), so the set is safe; a pass has two such targets alternating.
            static std::unordered_set<uint64_t> loggedKeys;
            const uint64_t key = ((uint64_t) w << 40) | ((uint64_t) h << 16) | (uint64_t) colorDesc.Format;
            if (loggedKeys.size() < 16 && loggedKeys.insert(key).second)
            {
                LOG_INFO("DuniaTaa: scaled-looking HDR target {}x{} fmt {} flags {:X} with depth ignored -- the "
                         "scene's HDR target is always RGBA16F with UAV",
                         w, h, (UINT) colorDesc.Format, (UINT) colorDesc.Flags);
            }
            return;
        }
        const uint64_t now = g_presentIndex.load();
        if (color == g_taa)
            g_taaSeenPresent = now;
        // Square eye: FC6's square probe pass then has the scene's shape, so the aspect test above cannot
        // tell a smaller probe from a scaled scene. A scaled candidate must first be seen as a TAA resolve's colour at
        // its size (the structural scene of the eye's aspect, g_structEyeSceneW). Until then it is held, and the heap
        // mirroring is kept on for 600 Presents from the first hold of that size so the resolve's table can be seen --
        // a Resolution Scale change arrives here first. Non-square eyes: not reached (the aspect test already decides).
        if (SquareEyeRules() && color != g_taa)
        {
            const UINT sw = g_structEyeSceneW.load(std::memory_order_relaxed);
            const UINT sh = g_structEyeSceneH.load(std::memory_order_relaxed);
            if (std::abs((int) w - (int) sw) > 32 || std::abs((int) h - (int) sh) > 32)
            {
                static std::unordered_map<uint64_t, uint64_t> heldSince; // under g_lock
                const uint64_t key = ((uint64_t) w << 32) | h;
                auto [it, fresh] = heldSince.try_emplace(key, now);
                if (now < it->second + 600)
                    g_structRefreshUntil.store(it->second + 600, std::memory_order_relaxed);
                if (fresh && heldSince.size() <= 16)
                    LOG_INFO("DuniaTaa: scaled scene candidate {}x{} held: square-eye rules, no TAA resolve seen at that "
                             "size yet (the resolve of the eye's shape is {}x{}); heap mirroring on to look",
                             w, h, sw, sh);
                return;
            }
        }
        if (g_taa != nullptr && area == (uint64_t) g_taaW * (uint64_t) g_taaH && newScore < g_taaScore)
            return;
        // A scene that got SMALLER never replaced the learned one: the rule keeps the largest area, so a
        // Resolution Scale lowered mid-session (70% -> 50%) left g_taaW at the old size, F6 then
        // created the feature for it, and every size-keyed gate below missed for the rest of the run at 30 fps
        // (the motion search never matched, so its mirroring never stopped). The learned scene is stale once
        // it has not been bound for a second; a scaled candidate of another size then replaces it.
        // And the new size has to show up on three separate Presents first: a menu longer than a second makes
        // the learned scene stale, and one frame of some transient target must not swap it out and back.
        bool stale = g_taa != nullptr && g_taaSeenPresent != 0 && now - g_taaSeenPresent > 60 &&
                     (w != g_taaW || h != g_taaH);
        if (stale)
        {
            static UINT pendW = 0, pendH = 0;
            static uint64_t pendPresent = 0;
            static int pendCount = 0;
            if (w == pendW && h == pendH)
            {
                if (now != pendPresent)
                    ++pendCount;
            }
            else
            {
                pendW = w;
                pendH = h;
                pendCount = 1;
            }
            pendPresent = now;
            stale = pendCount >= 3;
        }
        if (g_taa == nullptr || area >= (uint64_t) g_taaW * (uint64_t) g_taaH || stale)
        {
            if (stale)
                LOG_INFO("DuniaTaa: scaled TAA changed {}x{} -> {}x{} (the old one has not been bound for {} Presents){}",
                         g_taaW, g_taaH, w, h, now - g_taaSeenPresent,
                         g_createdInW != 0 ? " -- the DLSS feature was created for the old size; the splice rebuilds "
                                             "it live once the resolve reads the new one"
                                           : "");
            Retain(&g_taa, color);
            Retain(&g_taaDepth, depth);
            g_taaW = w;
            g_taaH = h;
            g_taaScore = newScore;
            g_taaSeenPresent = now;
            static bool loggedTaa = false;
            if (!loggedTaa)
            {
                loggedTaa = true;
                LOG_INFO("DuniaTaa: scaled TAA {}x{} fmt {} (dest {}x{})", w, h, (UINT) colorDesc.Format, g_destW,
                         g_destH);
            }
        }
        // The feature exists for another render size: say so (overlay + toast) for as long as it lasts. This
        // used to come from the old upscale-draw rule, which no longer runs in splice mode.
        if (g_createdInW != 0 && IsUpscale() && (w != g_createdInW || h != g_createdInH) &&
            std::abs((int) w - (int) g_createdInW) > 32)
        {
            g_mismatchW = w;
            g_mismatchH = h;
            g_mismatchPresent = now + 1;
            static UINT warnedW = 0, warnedH = 0;
            if (warnedW != w || warnedH != h)
            {
                warnedW = w;
                warnedH = h;
                LOG_WARN("DuniaTaa: the game now renders {}x{} but DLSS was created for {}x{} -> {}x{}: the game's "
                         "own resolve runs until the live rebuild (look for 'live rebuild')",
                         w, h, g_createdInW, g_createdInH, g_createdOutW, g_createdOutH);
            }
        }
        return;
    }

    // The pinned full-size scene is the eye's own size once the eye is known (a 16:9 view or a probe is not).
    if (!eyeSized)
        return;

    if (g_color != nullptr)
    {
        const auto prevArea = (uint64_t) g_pinnedW * (uint64_t) g_pinnedH;
        if (area < prevArea)
            return;
        if (SamePinnedSize(w, h))
            return;
    }

    Retain(&g_color, color);
    Retain(&g_depth, depth);
    if (g_pinnedW != w || g_pinnedH != h)
    {
        Retain(&g_gameMv, nullptr);
        g_mvViewsValid = false;
    }
    g_pinnedW = w;
    g_pinnedH = h;
    g_pinnedFormat = colorDesc.Format;
}

bool DuniaTaa_Dx12::WantsRealMotion()
{
    return Active() && Config::Instance()->DuniaTaaRealMotion.value_or_default();
}

namespace
{
// Only the modes that read the game's own packed vectors need the motion pass found, its texture copied at the
// dispatch, and therefore ResTrack's heap mirroring. The matrix path reads our depth and the game's camera
// constants out of a CPU mapping, so it needs none of it -- which is why it costs no frame rate, where F4 used
// to cost a third of it.
bool NeedsGameVectors()
{
    const int mode = g_realMotionMode.load(std::memory_order_relaxed);
    // The object mode needs the game's vectors copied at the dispatch exactly as the game's-own mode does;
    // the mirroring it implies stops once the pipeline is learned, the same as there.
    return mode == kMotionGame || mode == kMotionChained || mode == kMotionMatrixObjects;
}
} // namespace

// The upscale dispatch below 100% is recognised by its pipeline too, so the pipeline is wanted whenever the
// splice may be upscaling -- SpliceTaaMode() alone, because the feature (and IsUpscale) may not exist yet.
bool DuniaTaa_Dx12::WantsDispatchPipeline() { return NeedsGameVectors() || SpliceTaaMode(); }

namespace
{
// Defined with the motion search below; both searches judge candidates against the same eye shape.
bool HalfEyeSized(UINT64 width, UINT height);

// Step 3 of the port: DLSS in place of the game's temporal resolve, recorded onto the game's own list
// immediately before the draw the caller is about to skip.
//
// The resource states here are contract facts at this point in the list, not assumptions:
//   * the target is bound through OMSetRenderTargets for a draw that is about to write it, so it is in
//     RENDER_TARGET;
//   * the colour is read by that same draw's pixel shader, so it is in PIXEL_SHADER_RESOURCE.
// The second has a known gap, named here rather than discovered in a headset: D3D12 wants a transition's
// "before" state to match exactly, and a resource the game left in
// PIXEL_SHADER_RESOURCE | NON_PIXEL_SHADER_RESOURCE would not. If this build's picture is wrong in a way
// that smells of state, that is the first suspect, and the remedy is the one the motion copy already
// uses: track the game's own ResourceBarrier calls per list and transition only out of a state that list
// itself set.
//
// Every way of doing nothing is logged once with its reason. A path that silently does nothing looks
// exactly like a path that was never built, and that has already cost two runs here. Whenever something
// is missing the game's own resolve is left to run -- the safe state, not a black frame.
bool SpliceTaa(ID3D12GraphicsCommandList* cmd, ID3D12Resource* target)
{
    if (!WaitedForKey("the TAA splice"))
        return false;

    // Held for the whole splice: the live rebuild swaps what the splice reads (see g_spliceLock).
    std::lock_guard<std::mutex> spliceLock(g_spliceLock);

    static bool loggedFeature = false, loggedTarget = false, loggedPair = false, loggedDepth = false;
    static bool loggedSize = false, loggedEval = false, loggedCopy = false, loggedOk = false;

    if (g_handle == nullptr)
    {
        if (!loggedFeature)
        {
            loggedFeature = true;
            LOG_INFO("DuniaTaa: TAA splice: no DLSS feature yet, the game's own resolve runs");
        }
        return false;
    }

    if (target == nullptr)
    {
        if (!loggedTarget)
        {
            loggedTarget = true;
            LOG_WARN("DuniaTaa: TAA splice: no render target recorded on this list at the resolve draw");
        }
        return false;
    }

    // Back up to 100%: the learned scaled scene never goes away by itself (the learner only ever replaces it
    // with another SCALED size), so SceneW() kept the old render size, and the pair search -- keyed on it --
    // could never learn the eye-sized pairs that would have said so. The fact that breaks the circle is the
    // resolve's own render target, bound right here and so alive: it is scene-sized by construction. When it
    // is the eye size and the scaled scene has not been bound for 60 Presents, forget the scaled scene. Menus
    // never draw the resolve, so a menu cannot do this and set every size-keyed learner relearning.
    if (SceneScaled())
    {
        const auto targetDesc = target->GetDesc();
        const uint64_t now = g_presentIndex.load();
        std::lock_guard<std::mutex> lock(g_lock);
        if ((UINT) targetDesc.Width == g_pinnedW && targetDesc.Height == g_pinnedH && g_taaSeenPresent != 0 &&
            now - g_taaSeenPresent > 60)
        {
            LOG_INFO("DuniaTaa: back to 100% Resolution Scale: the resolve writes the {}x{} eye size and the scaled "
                     "scene {}x{} has not been bound for {} Presents -- forgetting it, the scene is the eye size again",
                     g_pinnedW, g_pinnedH, g_taaW, g_taaH, now - g_taaSeenPresent);
            // Deferred, never an immediate Release: a list recorded a moment ago may still reference it.
            RetainDeferred(&g_taa, nullptr, now);
            RetainDeferred(&g_taaDepth, nullptr, now);
            g_taaW = 0;
            g_taaH = 0;
            g_taaScore = -1;
            g_taaSeenPresent = 0;
        }
    }

    // Which eye, from the target the pair lookup is about to use. Under AER the game keeps one resolve
    // target per eye, so this is free and exact -- no parity, no guess. A single shared feature would
    // reproject each eye's history onto the other eye's viewpoint the moment motion vectors are on.
    bool eyeOverflow = false;
    const int eyeSlot = g_taaResolve.EyeSlot(target, &eyeOverflow);
    if (eyeOverflow)
    {
        static bool loggedOverflow = false;
        if (!loggedOverflow)
        {
            loggedOverflow = true;
            LOG_WARN("DuniaTaa: TAA splice: more than two resolve targets, so one target per eye is wrong "
                     "for this game -- falling back to a single shared history");
        }
    }

    ID3D12Resource* colour = g_taaResolve.ColourFor(target);
    if (colour == nullptr)
    {
        // A reload or a resolution change frees the game's targets, so every learned pair goes stale at
        // once. Learn again rather than guess -- and say so, because "the splice quietly stopped" is
        // otherwise indistinguishable from "the splice was never on".
        g_taaResolve.Reopen(g_presentIndex.load());
        g_taaPairsDone.store(false, std::memory_order_relaxed);
        // The eye assignment went with them and can come back the other way round, so neither history
        // belongs to the eye that will next be handed it.
        g_needReset = true;
        if (!loggedPair)
        {
            loggedPair = true;
            LOG_WARN("DuniaTaa: TAA splice: target {:X} has no colour paired with it -- reopening the pair "
                     "search, which is what a reload looks like",
                     (size_t) target);
        }
        return false;
    }

    ID3D12Resource* depth = g_taaResolve.EyeDepth(cmd);
    if (depth == nullptr)
    {
        if (!loggedDepth)
        {
            loggedDepth = true;
            LOG_WARN("DuniaTaa: TAA splice: no eye depth bound on this list yet");
        }
        return false;
    }

    // The feature is made once, for one size: its INPUT must be what the resolve reads (the scene size).
    // Its output is the same (DLAA) or the eye size (upscaling, downsampled onto the resolve's target).
    // Anything else means the feature is not the one this path needs, and doing GPU work at a size a path
    // was not built for is what crashed FC5.
    if (g_createdInW == 0 || g_createdInW != SceneW() || g_createdInH != SceneH() ||
        !(g_createdOutW == g_createdInW || g_createdOutW == g_destW))
    {
        // A Resolution Scale change: rebuild live when the new size is a fact and has held. Every refusal
        // says why, once per reason -- a rebuild that silently never happens looks like the old restart rule.
        const uint64_t now = g_presentIndex.load();
        const UINT inW = SceneW(), inH = SceneH();
        const bool renderToEye = SceneScaled() && Config::Instance()->DuniaTaaSpliceUpscale.value_or_default();
        const UINT outW = renderToEye ? g_destW : inW;
        const UINT outH = renderToEye ? g_destH : inH;
        const auto colourDesc = colour->GetDesc();
        if (g_rebuildSeenW != inW || g_rebuildSeenH != inH)
        {
            g_rebuildSeenW = inW;
            g_rebuildSeenH = inH;
            g_rebuildSeenAt = now;
        }
        int refuse = 0;
        if (g_createdInW == 0)
            refuse = 1; // no feature sizes at all: not a scale change
        else if ((UINT) colourDesc.Width != inW || colourDesc.Height != inH)
            refuse = 2; // the resolve does not read the scene size (yet): pairs or scene still relearning
        // (3, "the output size would change", used to be refused here: RebuildForScene now replaces
        // g_output too, so an eye resolution changed in VR mid-session rebuilds live like a Scale change.)
        else if (g_pinnedFormat != g_createdFormat)
            refuse = 4;
        else if (now - g_rebuildSeenAt < 45)
            refuse = 5; // hold: the size has to stay put for ~half a second of gameplay
        else if (g_rebuildPending || (g_rebuildAt != 0 && now - g_rebuildAt < 180))
            refuse = 6; // one rebuild per deferred-release window

        if (refuse == 0 && RebuildForScene(cmd, inW, inH, outW, outH))
        {
            loggedSize = false;
            // Fall through: this evaluate is the staged recreate's first step.
        }
        else
        {
            static int loggedRefuse = -1;
            if (refuse != 5 && refuse != 6 && loggedRefuse != refuse)
            {
                loggedRefuse = refuse;
                LOG_WARN("DuniaTaa: TAA splice: the feature is {}x{} -> {}x{} and the scene is {}x{} (resolve reads "
                         "{}x{}, eye {}x{}); live rebuild refused: {}",
                         g_createdInW, g_createdInH, g_createdOutW, g_createdOutH, inW, inH, (UINT) colourDesc.Width,
                         colourDesc.Height, g_destW, g_destH,
                         refuse == 1   ? "no feature sizes"
                         : refuse == 2 ? "the resolve does not read the scene size yet (still relearning)"
                         : refuse == 4 ? "the colour format changed -- restart"
                                       : "the rebuild itself failed -- restart");
            }
            if (!loggedSize)
            {
                loggedSize = true;
                LOG_INFO("DuniaTaa: TAA splice: the feature ({}x{}) does not match the {}x{} scene -- the game's own "
                         "resolve runs until the live rebuild",
                         g_createdInW, g_createdInH, inW, inH);
            }
            return false;
        }
    }

    const int slot = (eyeOverflow || eyeSlot < 0) ? 0 : eyeSlot;
    // GPU timing (diagnostic): T0 here, T1/T2 around the DLSS evaluate, T3 when this function returns -- after the
    // copy/downsample below, or at any early return (then counted incomplete).
    TimeBegin(cmd, slot);
    const TimeScope timeScope { cmd };
    if (!EvaluateOnList(cmd, nullptr, colour, depth, false, false, false,
                        D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET, slot))
    {
        if (!loggedEval)
        {
            loggedEval = true;
            LOG_WARN("DuniaTaa: TAA splice: the evaluate did nothing -- size, format or feature mismatch at "
                     "the resolve (feature {}x{} fmt {})",
                     g_createdInW, g_createdInH, (UINT) g_createdFormat);
        }
        return false;
    }

    // The staged recreate clears a handle's flag once its new feature is initialised. Until both are, the
    // eyes show their previous output (the recreate's evaluates write nothing), which is expected.
    if (g_rebuildPending)
    {
        auto& state = State::Instance();
        bool busy = false;
        for (NVSDK_NGX_Handle* handle : { g_handle, g_handleEye2 })
        {
            if (handle == nullptr)
                continue;
            const auto it = state.changeBackend.find(handle->Id);
            busy = busy || (it != state.changeBackend.end() && it->second);
        }
        if (!busy)
        {
            g_rebuildPending = false;
            g_outputFresh = false; // this evaluate was the last step, and it wrote the new output
            LOG_INFO("DuniaTaa: live rebuild #{} done after {} Presents: {} {}x{} -> {}x{} for both eyes",
                     g_rebuilds.load(), g_presentIndex.load() - g_rebuildAt, IsUpscale() ? "Upscaling" : "DLAA",
                     g_createdInW, g_createdInH, g_createdOutW, g_createdOutH);
        }
    }
    // A rebuild that replaced g_output: the new texture holds nothing until the new features have evaluated, where a
    // kept one held the previous frame. Hand these frames to the game's own resolve (the draw runs) -- a few frames
    // of the game's TAA instead of uninitialised memory in the headset.
    if (g_rebuildPending && g_outputFresh)
        return false;

    // The residual view replaces DLSS's picture with the object branch's decisions; the downsample and the copy
    // over the game's upscale below carry it to the eye like any other output.
    if (g_residualView.load(std::memory_order_relaxed) && !VisualizeMv(cmd))
    {
        // The decode is writing the view's numbers instead of motion while the view is on, so a view that cannot
        // be drawn must not stay on: that was a blurry picture and nothing else.
        g_residualView = false;
        g_needReset = true;
        LOG_WARN("DuniaTaa: residual view: the view pass did not run -- switched off so DLSS gets real motion "
                 "again (see 'MV view compile failed' or 'picture copy' above)");
    }

    const bool written = IsUpscale() ? DownsampleOutputTo(cmd, target, D3D12_RESOURCE_STATE_RENDER_TARGET)
                                     : CopyOutputToDest(cmd, target, D3D12_RESOURCE_STATE_RENDER_TARGET);
    if (!written)
    {
        if (!loggedCopy)
        {
            loggedCopy = true;
            LOG_WARN("DuniaTaa: TAA splice: DLSS ran but its output would not {} onto the resolve's target",
                     IsUpscale() ? "downsample" : "copy");
        }
        return false;
    }

    g_resolveSplicedPresent = g_presentIndex.load();
    g_resolveSplicedCmd.store(cmd, std::memory_order_relaxed);
    if (!loggedOk)
    {
        loggedOk = true;
        LOG_INFO("DuniaTaa: TAA splice: DLSS is the temporal resolve now -- reads {}, writes {}, eye slot {}"
                 "{}",
                 TraceDesc(colour), TraceDesc(target), slot,
                 g_handleEye2 != nullptr ? " (one history per eye)" : " (ONE SHARED HISTORY)");
    }
    return true;
}
} // namespace

// Until this existed, every constant slot in every descriptor table resolved to nothing and was skipped in
// silence -- so no capture could contain a CBV, and the fact that none did said nothing whatever about the game.
// The hook is INSTALLED for FC6 always -- it can only be installed at startup, and a probe armed later without it
// would find no CBVs and read like a fact about the game.
// And CBVs are RECORDED whenever it is installed, from startup (ResTrack's CBV hook checks WantsCbvTracking). They
// used to be recorded only with diagnostics or while the private-copy probe ran, and the probe (everyday build,
// diagnostics off) came back "no per-view CBV in any table" twice: 56k-124k CBV slots resolved,
// 28-35 MILLION empty. Dunia creates its descriptors at load and copies them into the shader-visible heaps every
// frame, and a copy from a slot we never recorded CLEARS the destination -- so every per-view descriptor was
// invisible. The cost does not justify a gate: in gameplay the game creates about one CBV per Present (121 and
// 334 in two 240-Present windows).
// CbvRecordingOn keeps its old gate for the one call site left on it: BUFFER SRV recording, not wanted in
// everyday play.
bool DuniaTaa_Dx12::WantsCbvTracking() { return Active(); }
bool DuniaTaa_Dx12::CbvRecordingOn()
{
    return Active() && (DiagnosticsOn() || g_pcState.load(std::memory_order_relaxed) != 0);
}
void DuniaTaa_Dx12::ArmPrivateCopyProbe()
{
    g_pcArm.store(true, std::memory_order_relaxed);
    LOG_INFO("DuniaTaa: private-copy probe armed (starts at the next Present with DLSS spliced)");
}
const char* DuniaTaa_Dx12::PrivateCopyProbeState()
{
    switch (g_pcState.load(std::memory_order_relaxed))
    {
    case 1:
        return "warming up (frame rate drops)";
    case 2:
        return "counting (frame rate drops)";
    default:
        return g_pcArm.load(std::memory_order_relaxed) ? "armed" : "idle -- result in OptiScaler.log";
    }
}

bool DuniaTaa_Dx12::WantsBufferMappings()
{
    return Active() && (DiagnosticsOn() || Config::Instance()->DuniaTaaRealMotion.value_or_default());
}

void DuniaTaa_Dx12::ForgetResource(ID3D12Resource* resource)
{
    if (resource == nullptr)
        return;
    // The learned upscale output dies with a menu too: forget the pair, learn again.
    if (g_upByParity[0].load(std::memory_order_relaxed) == resource ||
        g_upByParity[1].load(std::memory_order_relaxed) == resource)
    {
        g_upPipelineReady.store(false, std::memory_order_release);
        g_upByParity[0].store(nullptr, std::memory_order_relaxed);
        g_upByParity[1].store(nullptr, std::memory_order_relaxed);
        std::lock_guard<std::mutex> pending(g_cmdUpscaleLock);
        for (auto it = g_cmdUpscale.begin(); it != g_cmdUpscale.end();)
            it = it->second == resource ? g_cmdUpscale.erase(it) : std::next(it);
    }

    // Two loads on every resource death in the game: cheap enough to be unconditional.
    const bool p0 = g_mvByParity[0].load(std::memory_order_relaxed) == resource;
    const bool p1 = g_mvByParity[1].load(std::memory_order_relaxed) == resource;
    if (!p0 && !p1)
        return;

    // Both go, not just the dead one: the pair is only meaningful as a pair, and the survivor is about to
    // be replaced too (the game reallocates them together). Ready goes false FIRST, so a dispatch racing
    // this sees "not ready" before it can load a cleared slot.
    g_mvPipelineReady.store(false, std::memory_order_release);
    g_mvByParity[0].store(nullptr, std::memory_order_relaxed);
    g_mvByParity[1].store(nullptr, std::memory_order_relaxed);
    g_mvPipelineLogged = false;
    {
        // A list may already have this texture queued for the copy after its next dispatch.
        std::lock_guard<std::mutex> pending(g_cmdMotionLock);
        for (auto it = g_cmdMotion.begin(); it != g_cmdMotion.end();)
            it = it->second == resource ? g_cmdMotion.erase(it) : std::next(it);
    }
    const auto n = g_mvRelearns.fetch_add(1, std::memory_order_relaxed);
    if (n == 0)
        LOG_INFO("DuniaTaa: real motion: the game released a learned motion texture ({:X}) -- forgetting the "
                 "pair; the heap mirroring comes back until the pass is learned again. This is what a menu "
                 "does; later ones are logged at debug level",
                 (size_t) resource);
    else
        LOG_DEBUG("DuniaTaa: real motion: learned motion texture {:X} released ({} so far), pair forgotten",
                  (size_t) resource, n + 1);
}

void DuniaTaa_Dx12::ForgetViewConstants(const uint8_t* begin, const uint8_t* end)
{
    if (!g_viewConstants.ForgetIfWithin(begin, end))
        return;

    // Said once at INFO because the first one answers the question this whole fix rests on -- does the death
    // of the view buffer actually reach us when a menu opens? -- and at DEBUG after that, because a level load
    // frees a great many buffers and by then it is not news. A silent invalidation is how the previous version
    // of this bug stayed a mystery for so long.
    static std::atomic<uint64_t> count { 0 };
    const auto n = count.fetch_add(1, std::memory_order_relaxed);
    if (n == 0)
        LOG_INFO("DuniaTaa: view constants: the buffer holding them was released -- candidates dropped, the set "
                 "will be rescanned. This is what a menu does; later ones are logged at debug level");
    else
        LOG_DEBUG("DuniaTaa: view constants: the buffer holding them was released ({} so far), candidates "
                  "dropped",
                  n + 1);
}

bool DuniaTaa_Dx12::WantsTaaSearch()
{
    // Splice mode cannot work without the resolve being found first, so it implies the search rather than
    // silently doing nothing when only one of the two keys is set.
    return Active() && (Config::Instance()->DuniaTaaFindTaa.value_or_default() || SpliceTaaMode());
}

void DuniaTaa_Dx12::NotePipeline(ID3D12GraphicsCommandList* cmd, ID3D12PipelineState* pipeline)
{
    // Not while we are evaluating: DLSS sets its own pipelines on this list and recording one of those as
    // "what this list last set" would lose the resolve for the draws that follow.
    if (g_inEvaluate > 0 || cmd == nullptr || !WantsTaaSearch())
        return;
    t_pipeCmd = cmd;
    t_pipeline = pipeline;
    std::lock_guard<std::mutex> lock(g_taaLock);
    g_taaListPipeline[cmd] = pipeline;
}

const char* DuniaTaa_Dx12::SkipTaaName()
{
    if (!Config::Instance()->DuniaTaaFindTaa.value_or_default())
        return "TAA skip off (startup)";
    if (!g_taaFound.load())
        return "TAA resolve not found yet";
    switch (g_taaMode.load())
    {
    case 1:
        return "Game TAA SKIPPED (nothing replacing it)";
    case 2:
        return "DLSS spliced at the game's TAA";
    default:
        return "Game TAA running";
    }
}

// ---- one-key F6 -----------------------------------------------------------------------------------
// Everyday play gets one key. With [Inputs] DuniaTaaDiagnostics off, F6 starts DLSS and, as each becomes ready,
// splices it at the game's TAA resolve (what F8 did) and turns the validated object-motion state on (what
// F4 did). The two readiness conditions are the ones the keys themselves checked: the resolve identified,
// and the feature created. Diagnostics on keeps the three keys separate for A/Bs.
bool OneKeyMode() { return !DiagnosticsOn(); }
std::atomic<bool> g_oneKeyArm { false };
void ApplyTaaMode(int mode);
void ApplyRealMotionMode(int mode);

void DuniaTaa_Dx12::ToggleSkipTaa()
{
    if (!WantsTaaSearch())
    {
        LOG_INFO("DuniaTaa: F8: the TAA search is off in the ini, nothing to toggle");
        return;
    }
    if (!g_taaFound.load())
    {
        LOG_INFO("DuniaTaa: F8: the TAA resolve has not been identified yet ({} half-res motion + {} eye HDR "
                 "was the closest table seen)",
                 g_taaBestMotion.load(), g_taaBestHdr.load());
        return;
    }
    // With the splice built in, F8 flips between the game's TAA and DLSS spliced in its place -- the middle
    // state (the draw skipped with nothing replacing it, a frozen image) was the gate for step 2 and has done
    // its job, so it is out of the everyday cycle. Without the splice it is still the only other
    // state, so the two-state cycle keeps it. Both comparisons inside one run, because FC6 costs a headset
    // run to test and comparing across runs is meaningless.
    const int current = g_taaMode.load();
    const int mode = SpliceTaaMode() ? (current == 0 ? 2 : 0) : (current == 0 ? 1 : 0);
    ApplyTaaMode(mode);
}

// The TAA state change itself, shared by F8 and by the one-key F6 (see DriveOneKey).
void ApplyTaaMode(int mode)
{
    g_taaMode = mode;
    // Closes the open stats window: one that straddles a toggle belongs to neither state.
    g_taaStatsRestart.store(true, std::memory_order_relaxed);
    // Entering or leaving the splice changes what DLSS is accumulating; start its history clean.
    g_needReset = true;
    switch (mode)
    {
    case 1:
        // The gate for step 2: with the draw skipped and nothing replacing it, the picture MUST change.
        // If it does not, the pass identified is not the temporal resolve and nothing built on it holds.
        LOG_INFO("DuniaTaa: F8: skipping the game's TAA draw with nothing replacing it -- the picture must "
                 "visibly change; that is the whole point of this state ({} draws skipped so far)",
                 g_taaSkipped.load());
        break;
    case 2:
        LOG_INFO("DuniaTaa: F8: DLSS spliced at the game's TAA resolve ({} splices so far)",
                 g_taaSpliced.load());
        break;
    default:
        LOG_INFO("DuniaTaa: F8: letting the game's TAA draw run ({} draws skipped, {} spliced so far)",
                 g_taaSkipped.load(), g_taaSpliced.load());
        break;
    }
}

bool DuniaTaa_Dx12::ShouldSkipDraw(ID3D12GraphicsCommandList* cmd)
{
    // The same recursion guard every other hook here has: the splice runs DLSS, which runs NR
    // synchronously on this thread, and whatever either of them records goes through these hooks too.
    if (g_inEvaluate > 0 || cmd == nullptr || !WantsTaaSearch())
        return false;

    // The game's upscale draw (FSR1's EASU, one full-screen draw at the eye size): its output was being
    // overwritten by our copy right after it, so the draw itself is skipped and the copy recorded in its
    // place. The target is bound as the render target for this very draw, so RENDER_TARGET is a fact.
    // Reached only on the thread that recorded the bind (thread-local), so no lock on the other 2,000 draws.
    if (t_upscaleCmd == cmd && SpliceTaaMode() && IsUpscale() && g_output != nullptr)
    {
        t_upscaleCmd = nullptr;
        ID3D12Resource* target = nullptr;
        {
            std::lock_guard<std::mutex> lock(g_lock);
            const auto it = g_cmdTargets.find(cmd);
            if (it != g_cmdTargets.end() && it->second.upscaleDrawTarget != nullptr &&
                it->second.color == it->second.upscaleDrawTarget)
            {
                target = it->second.upscaleDrawTarget;
                it->second.upscaleDrawTarget = nullptr;
            }
        }
        if (target != nullptr && CopyOutputToDest(cmd, target, D3D12_RESOURCE_STATE_RENDER_TARGET))
        {
            g_upCopies.fetch_add(1, std::memory_order_relaxed);
            static bool logged = false;
            if (!logged)
            {
                logged = true;
                LOG_INFO("DuniaTaa: upscaling: the game's upscale draw is SKIPPED and DLSS's eye image copied over "
                         "its target {:X} in its place -- the eye sees DLSS's upscale, and the game's FSR1 pass no "
                         "longer runs for nothing",
                         (size_t) target);
            }
            return true;
        }
    }

    ID3D12PipelineState* pipeline = nullptr;
    bool candidate = false;
    if (g_taaFound.load(std::memory_order_acquire) && t_pipeCmd == cmd)
    {
        // Found: the candidate set is no longer consulted, and this thread set this list's pipeline itself.
        pipeline = t_pipeline;
    }
    else
    {
        std::lock_guard<std::mutex> lock(g_taaLock);
        const auto it = g_taaListPipeline.find(cmd);
        if (it != g_taaListPipeline.end())
            pipeline = it->second;
        candidate = g_taaCandidate.erase(cmd) != 0;
    }
    if (pipeline == nullptr)
        return false;

    // Learning: this list's last graphics table matched the signature, so the pipeline it is drawing with
    // is the resolve's.
    if (candidate && !g_taaFound.load(std::memory_order_relaxed))
    {
        g_taaPipeline = pipeline;
        g_taaFound.store(true, std::memory_order_release);
        {
            std::lock_guard<std::mutex> targets(g_taaTargetLock);
            const auto it = g_taaListTarget.find(cmd);
            if (it != g_taaListTarget.end())
            {
                g_taaTarget = it->second.first;
                g_taaTargetDepth = it->second.second;
            }
        }
        if (!g_taaLogged)
        {
            g_taaLogged = true;
            LOG_INFO("DuniaTaa: game TAA resolve found -- pipeline {:X}, graphics root {}. F8 skips its draw",
                     (size_t) pipeline, g_taaRootParameter);
            // Everything step 3 needs, named once. If the target is one of the colour inputs the resolve
            // ping-pongs in place, and DLSS's output goes to that same texture.
            LOG_INFO("DuniaTaa: TAA resolve writes {} (depth {})", TraceDesc(g_taaTarget),
                     TraceDesc(g_taaTargetDepth));
            for (int i = 0; i < kTaaColourSlots; ++i)
            {
                LOG_INFO("DuniaTaa: TAA resolve colour t{} = {}{}", i + 2, TraceDesc(g_taaColour[i]),
                         g_taaColour[i] != nullptr && g_taaColour[i] == g_taaTarget ? "  <-- IS THE TARGET" : "");
            }
        }
    }

    if (pipeline != g_taaPipeline)
        return false;

    // The render target this list has bound is read at every resolve draw, never latched: it and the
    // colour the resolve reads are adjacent allocations the game ping-pongs per eye, and a latched
    // pointer is a pointer the game is free to release.
    ID3D12Resource* target = nullptr;
    {
        std::lock_guard<std::mutex> targets(g_taaTargetLock);
        const auto it = g_taaListTarget.find(cmd);
        if (it != g_taaListTarget.end())
            target = it->second.first;
    }

    // The colour is only resolvable while the heap mirroring is on, so the pairing is learned here and
    // used by target lookup afterwards. Consumed either way, so a table bound for some other draw can
    // never be picked up by a later one. Learning runs whatever F8 is set to, so switching the splice on
    // mid-run never has to wait for it.
    TaaResolveMap::Colour colour {};
    if (g_taaResolve.TakeColour(cmd, &colour) && SpliceTaaMode() && target != nullptr)
    {
        if (g_taaResolve.Learn(target, colour, g_presentIndex.load()))
        {
            // The eye slot is named here so the log alone says which target is which eye, and a wrong
            // pairing can be seen without a headset.
            bool overflow = false;
            LOG_INFO("DuniaTaa: TAA resolve pair {} (eye slot {}) -- reads {}, writes {}",
                     g_taaResolve.Size(), g_taaResolve.EyeSlot(target, &overflow), TraceDesc(colour.colour),
                     TraceDesc(target));
        }
    }

    // F7's capture is driven from here in splice mode, and it has to be: the other two modes drive it from
    // BeforeCopy, and *both* of those paths are switched off when the splice owns the frame -- TonemappedMode()
    // stands down and SplicePathOn() returns false. So an F7 press set the arm and nothing ever consumed it. No
    // capture, no file, and not one line in the log to say so, because NoteCandidate bails before StartIfDue
    // until an eye image has been reported and only CaptureAt reports one. Driving it from the resolve draw also
    // gives the shape filters the right eye shape, and it runs whatever F8 is set to, so a capture never needs
    // the splice switched on first.
    if (DiagnosticsOn())
    {
        ID3D12Resource* captureColour =
            colour.colour != nullptr ? colour.colour : g_taaResolve.ColourFor(target);
        if (captureColour != nullptr)
        {
            // The same guard the other two call sites take: the copies this records go back through our own
            // copy hooks otherwise.
            g_skipCopyHook++;
            FrameCapture_Dx12::CaptureAt(cmd, captureColour, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
            g_skipCopyHook--;
        }
    }

    const int mode = g_taaMode.load(std::memory_order_relaxed);
    if (mode == 0)
        return false;
    if (mode == 2)
    {
        // Nothing missing may turn into a skipped draw with no replacement: SpliceTaa logs its reason and
        // the game's own resolve runs.
        if (!SpliceTaa(cmd, target))
            return false;
        g_taaSpliced.fetch_add(1, std::memory_order_relaxed);
    }
    g_taaSkipped.fetch_add(1, std::memory_order_relaxed);
    return true;
}

// ---- upscale writer probe ------------------------------------------------------------------------
// Below 100% the game renders at the render size and something writes the eye-sized texture the post chain
// reads (eye-sized, fmt 10, seen ONLY as a compute-table SRV). Five builds
// guessed the writer and every one was wrong: the mode-1 'next bind' never came; the compute-UAV search saw
// only SRVs; an OMSet WITH depth was an in-world target with ~150 draws; with depth excluded no bind matched.
// Two blind spots were common to all of them. Every rule was gated on the resolve splice having been recorded
// EARLIER IN THIS PRESENT, and Dunia records lists in parallel, so a writer recorded on another thread before
// the splice was invisible; and OMSet resolved RTV slot 0 only.
// This is the diagnostic that replaces the guessing. For one bounded window it records EVERY touch of any
// eye-sized texture (any format, not ours) by every route the API offers for writing a texture -- OMSet at
// any slot, table UAV/SRV (graphics and compute, any slot), copy, resolve, clear, and every transition
// barrier (the game must declare RENDER_TARGET / UNORDERED_ACCESS / COPY_DEST before writing) -- ungated on
// the splice, with the phase (before/after this Present's splice, on its list or not) carried as data. Root
// UAV/SRV descriptors are GPU addresses and can bind only buffers, so they are not a route for a texture.
// Logged once per distinct (kind, slot/state, format, phase); then a summary that says which hooks actually
// saw calls (a hook at 0 is a blind spot, not a fact) and the first Presents' full event sequence with the
// submits inline. The heap mirroring is held on for the window and released with it.
namespace
{
enum class ProbeKind : int
{
    Rtv = 0,
    SrvGraphics,
    UavGraphics,
    SrvCompute,
    UavCompute,
    CopyDst,
    CopySrc,
    ResolveDst,
    ResolveSrc,
    ClearRtv,
    ClearUav,
    Barrier,
    Submit,
    Present,
    Count
};

const char* ProbeKindName(ProbeKind kind)
{
    switch (kind)
    {
    case ProbeKind::Rtv: return "OMSet render target";
    case ProbeKind::SrvGraphics: return "graphics table SRV";
    case ProbeKind::UavGraphics: return "graphics table UAV";
    case ProbeKind::SrvCompute: return "compute table SRV";
    case ProbeKind::UavCompute: return "compute table UAV";
    case ProbeKind::CopyDst: return "copy DESTINATION";
    case ProbeKind::CopySrc: return "copy source";
    case ProbeKind::ResolveDst: return "resolve DESTINATION";
    case ProbeKind::ResolveSrc: return "resolve source";
    case ProbeKind::ClearRtv: return "ClearRenderTargetView";
    case ProbeKind::ClearUav: return "ClearUnorderedAccessView";
    case ProbeKind::Barrier: return "barrier";
    case ProbeKind::Submit: return "submit";
    case ProbeKind::Present: return "present";
    default: return "?";
    }
}

enum class ProbeHook : int
{
    OmSet = 0,
    GraphicsTable,
    ComputeTable,
    Copy,
    Resolve,
    Clear,
    Barrier,
    Execute,
    Draw,
    Count
};

std::string ProbeStateName(int state)
{
    switch (state)
    {
    case D3D12_RESOURCE_STATE_COMMON: return "COMMON";
    case D3D12_RESOURCE_STATE_RENDER_TARGET: return "RENDER_TARGET";
    case D3D12_RESOURCE_STATE_UNORDERED_ACCESS: return "UNORDERED_ACCESS";
    case D3D12_RESOURCE_STATE_DEPTH_WRITE: return "DEPTH_WRITE";
    case D3D12_RESOURCE_STATE_DEPTH_READ: return "DEPTH_READ";
    case D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE: return "NON_PIXEL_SHADER_RESOURCE";
    case D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE: return "PIXEL_SHADER_RESOURCE";
    case D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE: return "ALL_SHADER_RESOURCE";
    case D3D12_RESOURCE_STATE_COPY_DEST: return "COPY_DEST";
    case D3D12_RESOURCE_STATE_COPY_SOURCE: return "COPY_SOURCE";
    case D3D12_RESOURCE_STATE_RESOLVE_DEST: return "RESOLVE_DEST";
    case D3D12_RESOURCE_STATE_RESOLVE_SOURCE: return "RESOLVE_SOURCE";
    case D3D12_RESOURCE_STATE_GENERIC_READ: return "GENERIC_READ";
    default: return std::format("state {:X}", (unsigned) state);
    }
}

std::atomic<bool> g_probeOn { false };
std::atomic<bool> g_probeArm { false }; // F10
bool g_probeDone = false;               // present thread only: auto-armed once per session
uint64_t g_probeFrom = 0;               // present thread only
uint64_t g_probeUntil = 0;              // present thread only
uint64_t g_probeSplicedPresents = 0;    // present thread only
std::atomic<uint64_t> g_probeSeqUntil { 0 };
std::atomic<uint64_t> g_probeSeen[(size_t) ProbeHook::Count] {};
std::atomic<uint64_t> g_probeUnresolvedRtvs { 0 };
std::atomic<uint64_t> g_probeTablesNoHeap { 0 };
std::atomic<uint64_t> g_probeSlotsUnresolved { 0 };
// Everything below the lock is touched from the recording threads and the present thread.
std::mutex g_probeLock;
uint64_t g_probeCounts[(size_t) ProbeKind::Count] {};
std::unordered_set<uint64_t> g_probeLogged;
std::unordered_map<ID3D12GraphicsCommandList*, uint32_t> g_probeDraws;

struct ProbeEvent
{
    uint64_t present;
    ID3D12GraphicsCommandList* cmd; // identity only
    ID3D12Resource* res;            // identity only
    ProbeKind kind;
    int a, b;
    bool spliced, spliceList;
    uint32_t draws;
    DWORD thread;
    UINT format;
};
std::vector<ProbeEvent> g_probeSeq;
constexpr size_t kProbeSeqMax = 700;
constexpr size_t kProbeLogMax = 64;
constexpr uint64_t kProbePresents = 240;
constexpr uint64_t kProbeSeqPresents = 3;

bool ProbeEyeSized(ID3D12Resource* res, D3D12_RESOURCE_DESC* out)
{
    if (res == nullptr || res == g_output || g_createdOutW == 0)
        return false;
    const auto desc = res->GetDesc();
    if (desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || (UINT) desc.Width != g_createdOutW ||
        desc.Height != g_createdOutH)
        return false;
    if (out != nullptr)
        *out = desc;
    return true;
}

std::string ProbeDetail(ProbeKind kind, int a, int b)
{
    switch (kind)
    {
    case ProbeKind::Rtv: return std::format("RTV slot {}, depth bound: {}", a, b != 0 ? "yes" : "NO");
    case ProbeKind::SrvGraphics:
    case ProbeKind::UavGraphics:
    case ProbeKind::SrvCompute:
    case ProbeKind::UavCompute: return std::format("root parameter {} slot {}", a, b);
    case ProbeKind::Barrier: return std::format("transition {} -> {}", ProbeStateName(a), ProbeStateName(b));
    case ProbeKind::Submit: return std::format("list {} of {}", a, b);
    default: return {};
    }
}

void ProbeSeen(ProbeHook hook) { g_probeSeen[(size_t) hook].fetch_add(1, std::memory_order_relaxed); }

void ProbeTouch(ID3D12GraphicsCommandList* cmd, ID3D12Resource* res, ProbeKind kind, int a, int b)
{
    D3D12_RESOURCE_DESC desc {};
    if (!ProbeEyeSized(res, &desc))
        return;
    const uint64_t present = g_presentIndex.load();
    const bool spliced = g_resolveSplicedPresent == present;
    const bool spliceList = spliced && cmd == g_resolveSplicedCmd.load(std::memory_order_relaxed);
    ProbeEvent ev { present, cmd, res, kind, a, b, spliced, spliceList, 0, GetCurrentThreadId(), (UINT) desc.Format };
    {
        std::lock_guard<std::mutex> lock(g_probeLock);
        g_probeCounts[(size_t) kind]++;
        ev.draws = g_probeDraws[cmd];
        if (present < g_probeSeqUntil.load(std::memory_order_relaxed) && g_probeSeq.size() < kProbeSeqMax)
            g_probeSeq.push_back(ev);
        const uint64_t key = ((uint64_t) kind << 58) | ((uint64_t) (a & 0xFFFFFF) << 34) |
                             ((uint64_t) (b & 0xFFFFFF) << 10) | ((uint64_t) (ev.format & 0xFF) << 2) |
                             ((uint64_t) spliced << 1) | (uint64_t) spliceList;
        if (g_probeLogged.size() >= kProbeLogMax || !g_probeLogged.insert(key).second)
            return;
    }
    LOG_INFO("DuniaTaa: upscale writer probe: {} -- {:X} {}x{} fmt {} flags {:X}; {}; present {}, {} this Present's "
             "resolve splice{}; list {:X} thread {}, {} draws on the list since its previous OMSet",
             ProbeKindName(kind), (size_t) res, desc.Width, desc.Height, (UINT) desc.Format, (UINT) desc.Flags,
             ProbeDetail(kind, a, b), present, spliced ? "AFTER" : "BEFORE (or without)",
             spliceList ? " on the splice's own list" : "", (size_t) cmd, ev.thread, ev.draws);
}

// A root descriptor table, either stage: every slot up to 16, any format. Needs the heap mirroring, which
// NeedsHeapTracking holds on while the probe runs.
void ProbeTable(bool graphics, UINT root, D3D12_GPU_DESCRIPTOR_HANDLE base, ID3D12GraphicsCommandList* cmd)
{
    ProbeSeen(graphics ? ProbeHook::GraphicsTable : ProbeHook::ComputeTable);
    if (base.ptr == 0 || cmd == nullptr || g_inEvaluate > 0)
        return;
    auto heap = ResTrack_Dx12::GetTableHeap(graphics, (SIZE_T) base.ptr);
    if (heap == nullptr || heap->increment == 0)
    {
        g_probeTablesNoHeap.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    for (UINT i = 0; i < 16; ++i)
    {
        const SIZE_T handle = (SIZE_T) base.ptr + (SIZE_T) i * heap->increment;
        if (handle >= heap->gpuEnd)
            break;
        ResourceInfo info {};
        if (!heap->GetByGpuHandle(handle, info))
        {
            g_probeSlotsUnresolved.fetch_add(1, std::memory_order_relaxed);
            continue;
        }
        if (info.buffer == nullptr || info.buffer == g_output || (UINT) info.width != g_createdOutW ||
            info.height != g_createdOutH)
            continue;
        if (info.type == UAV)
            ProbeTouch(cmd, info.buffer, graphics ? ProbeKind::UavGraphics : ProbeKind::UavCompute, (int) root,
                       (int) i);
        else if (info.type == SRV)
            ProbeTouch(cmd, info.buffer, graphics ? ProbeKind::SrvGraphics : ProbeKind::SrvCompute, (int) root,
                       (int) i);
    }
}

// Present thread, when the window closes. Says what was seen, what could not have been seen, and the order.
void ProbeSummary()
{
    std::lock_guard<std::mutex> lock(g_probeLock);
    auto seen = [](ProbeHook h) { return g_probeSeen[(size_t) h].load(std::memory_order_relaxed); };
    auto count = [](ProbeKind k) { return g_probeCounts[(size_t) k]; };
    LOG_INFO("DuniaTaa: upscale writer probe: OVER -- Presents {}..{}, {} of them with a resolve splice. Hook calls "
             "seen: OMSet {}, graphics tables {}, compute tables {} ({} tables with no mirrored heap, {} slots "
             "unresolved), copies {}, resolves {}, clears {}, barriers {}, submits {}, draws {}. A hook at 0 saw "
             "NOTHING and its kinds below are blind spots, not facts (barriers need DuniaTaaRealMotion or a "
             "capture; graphics tables need the TAA search; compute tables need the motion search; draws need the "
             "TAA search or diagnostics). {} OMSet RTV slots could not be resolved to a resource",
             g_probeFrom, g_probeUntil, g_probeSplicedPresents, seen(ProbeHook::OmSet), seen(ProbeHook::GraphicsTable),
             seen(ProbeHook::ComputeTable), g_probeTablesNoHeap.load(), g_probeSlotsUnresolved.load(),
             seen(ProbeHook::Copy), seen(ProbeHook::Resolve), seen(ProbeHook::Clear), seen(ProbeHook::Barrier),
             seen(ProbeHook::Execute), seen(ProbeHook::Draw), g_probeUnresolvedRtvs.load());
    LOG_INFO("DuniaTaa: upscale writer probe: touches of {}x{} textures -- OMSet {}, graphics SRV {}, graphics UAV {}, "
             "compute SRV {}, compute UAV {}, copy dst {}, copy src {}, resolve dst {}, resolve src {}, clear RTV {}, "
             "clear UAV {}, barriers {}. Root UAV/SRV descriptors bind by address and can only be buffers: not a "
             "route for a texture. The writer is whichever of OMSet / table UAV / copy dst / resolve dst / clear "
             "appears, and the barrier INTO its write state names the same route. If only SRVs and barriers "
             "appear, the write is on a path with no hook here (ExecuteIndirect, a bundle, or a list these hooks "
             "never saw)",
             g_createdOutW, g_createdOutH, count(ProbeKind::Rtv), count(ProbeKind::SrvGraphics),
             count(ProbeKind::UavGraphics), count(ProbeKind::SrvCompute), count(ProbeKind::UavCompute),
             count(ProbeKind::CopyDst), count(ProbeKind::CopySrc), count(ProbeKind::ResolveDst),
             count(ProbeKind::ResolveSrc), count(ProbeKind::ClearRtv), count(ProbeKind::ClearUav),
             count(ProbeKind::Barrier));
    LOG_INFO("DuniaTaa: upscale writer probe: {} events over the first {} Presents, in recording order with the "
             "submits inline{} -- read the eye-sized texture's touches top to bottom: the first write-state barrier "
             "or write bind before its first SRV read is the writer",
             g_probeSeq.size(), kProbeSeqPresents, g_probeSeq.size() >= kProbeSeqMax ? " (TRUNCATED)" : "");
    for (const auto& ev : g_probeSeq)
    {
        if (ev.kind == ProbeKind::Present)
            LOG_INFO("DuniaTaa: upscale writer probe:   ---- end of Present {} ({} resolve splice) ----", ev.present,
                     ev.a != 0 ? "with a" : "WITHOUT a");
        else if (ev.kind == ProbeKind::Submit)
            LOG_INFO("DuniaTaa: upscale writer probe:   [{}] submit list {:X} ({}) thread {}", ev.present,
                     (size_t) ev.cmd, ProbeDetail(ev.kind, ev.a, ev.b), ev.thread);
        else
            LOG_INFO("DuniaTaa: upscale writer probe:   [{}] {} {:X} fmt {} {} -- list {:X} {}{} thread {} "
                     "draws-since-OMSet {}",
                     ev.present, ProbeKindName(ev.kind), (size_t) ev.res, ev.format, ProbeDetail(ev.kind, ev.a, ev.b),
                     (size_t) ev.cmd, ev.spliced ? "after-splice" : "before-splice",
                     ev.spliceList ? " splice-list" : "", ev.thread, ev.draws);
    }
    // Clean for a re-arm (F10).
    g_probeSeq.clear();
    g_probeLogged.clear();
    g_probeDraws.clear();
    for (auto& c : g_probeCounts)
        c = 0;
    for (auto& s : g_probeSeen)
        s.store(0, std::memory_order_relaxed);
    g_probeUnresolvedRtvs.store(0);
    g_probeTablesNoHeap.store(0);
    g_probeSlotsUnresolved.store(0);
}
} // namespace

bool DuniaTaa_Dx12::ProbeOn() { return g_probeOn.load(std::memory_order_relaxed); }

void DuniaTaa_Dx12::ProbeOmSet(ID3D12GraphicsCommandList* cmd, ID3D12Resource* const* rtvs, UINT numRtvs,
                               UINT unresolved, ID3D12Resource* depth)
{
    if (!ProbeOn() || cmd == nullptr || cmd == g_cmd || g_inEvaluate > 0)
        return;
    ProbeSeen(ProbeHook::OmSet);
    if (unresolved != 0)
        g_probeUnresolvedRtvs.fetch_add(unresolved, std::memory_order_relaxed);
    for (UINT i = 0; i < numRtvs; ++i)
    {
        if (rtvs[i] != nullptr)
            ProbeTouch(cmd, rtvs[i], ProbeKind::Rtv, (int) i, depth != nullptr ? 1 : 0);
    }
    std::lock_guard<std::mutex> lock(g_probeLock);
    g_probeDraws[cmd] = 0;
}

void DuniaTaa_Dx12::ProbeBarriers(ID3D12GraphicsCommandList* cmd, UINT count, const D3D12_RESOURCE_BARRIER* barriers)
{
    if (!ProbeOn() || cmd == nullptr || cmd == g_cmd || barriers == nullptr || g_inEvaluate > 0)
        return;
    ProbeSeen(ProbeHook::Barrier);
    for (UINT i = 0; i < count; ++i)
    {
        if (barriers[i].Type == D3D12_RESOURCE_BARRIER_TYPE_TRANSITION)
            ProbeTouch(cmd, barriers[i].Transition.pResource, ProbeKind::Barrier,
                       (int) barriers[i].Transition.StateBefore, (int) barriers[i].Transition.StateAfter);
    }
}

void DuniaTaa_Dx12::ProbeResolve(ID3D12GraphicsCommandList* cmd, ID3D12Resource* dst, ID3D12Resource* src)
{
    if (!ProbeOn() || cmd == nullptr || cmd == g_cmd || g_inEvaluate > 0)
        return;
    ProbeSeen(ProbeHook::Resolve);
    ProbeTouch(cmd, dst, ProbeKind::ResolveDst, 0, 0);
    ProbeTouch(cmd, src, ProbeKind::ResolveSrc, 0, 0);
}

void DuniaTaa_Dx12::ProbeClear(ID3D12GraphicsCommandList* cmd, ID3D12Resource* resource, bool uav)
{
    if (!ProbeOn() || cmd == nullptr || cmd == g_cmd || g_inEvaluate > 0)
        return;
    ProbeSeen(ProbeHook::Clear);
    ProbeTouch(cmd, resource, uav ? ProbeKind::ClearUav : ProbeKind::ClearRtv, 0, 0);
}

void DuniaTaa_Dx12::NoteExecute(ID3D12CommandList* const* lists, UINT count)
{
    // The velocity-reader probe's submission order (its counting window only).
    if (lists != nullptr && g_pcState.load(std::memory_order_relaxed) == 2)
        VelNoteSubmits(lists, count);
    if (!ProbeOn() || lists == nullptr)
        return;
    ProbeSeen(ProbeHook::Execute);
    const uint64_t present = g_presentIndex.load();
    if (present >= g_probeSeqUntil.load(std::memory_order_relaxed))
        return;
    std::lock_guard<std::mutex> lock(g_probeLock);
    for (UINT i = 0; i < count && g_probeSeq.size() < kProbeSeqMax; ++i)
    {
        g_probeSeq.push_back(ProbeEvent { present, (ID3D12GraphicsCommandList*) lists[i], nullptr, ProbeKind::Submit,
                                          (int) i, (int) count, false, false, 0, GetCurrentThreadId(), 0 });
    }
}

// A table with the TAA resolve's shape at w x h (structural; see g_structSceneW). Every distinct size is logged once (up
// to 16), so a false match shows up as a second size rather than hiding behind the first.
static void StructNoteScene(UINT w, UINT h)
{
    g_structSceneW.store(w, std::memory_order_relaxed);
    g_structSceneH.store(h, std::memory_order_relaxed);
    // A TAA resolve of the VR eye's own shape (not the 16:9 view the pause menu also resolves): only
    // gameplay draws it, so from its first sighting a search that has not started is a fault to report (DriveTaaSearch).
    if (const UINT eyeW = g_vrEyeW.load(std::memory_order_relaxed), eyeH = g_vrEyeH.load(std::memory_order_relaxed);
        eyeW != 0 && w <= eyeW + 32 && h <= eyeH + 32 && SimilarAspect(w, h, eyeW, eyeH))
    {
        uint64_t expected = 0;
        g_structEyeSceneFirst.compare_exchange_strong(expected, g_presentIndex.load());
        g_structEyeSceneW.store(w, std::memory_order_relaxed);
        g_structEyeSceneH.store(h, std::memory_order_relaxed);
    }
    std::lock_guard<std::mutex> lock(g_structLock);
    if (g_structSceneLogged.size() < 16 && g_structSceneLogged.insert(((uint64_t) w << 32) | h).second)
        LOG_INFO("DuniaTaa: structural: a TAA-resolve-shaped table at {}x{} (2+ R8G8 motion at half that, 2+ RGBA16F) "
                 "-- the size learners have scene {}x{}, eye {}x{}",
                 w, h, SceneW(), SceneH(), g_destW, g_destH);
}

// The copy of the finished eye image in VR (structural; see g_vrEyeW). Distinct sizes logged once (up to 16).
// Adopted as the eye (g_vrEyeW/H, which gates Note's learners) at once the first time, and a new size only once it has
// been copied out on 3 separate Presents: one stray copy must not re-gate the learners. Flat before VR starts (the
// 1920x1080 copy-out) is harmless: nothing that small passes IsEvalSize, and the VR size replaces it within 3 Presents.
static void StructNoteEye(UINT w, UINT h)
{
    const uint64_t now = g_presentIndex.load();
    std::lock_guard<std::mutex> lock(g_structLock);
    if (g_vrEyeLogged.size() < 16 && g_vrEyeLogged.insert(((uint64_t) w << 32) | h).second)
        LOG_INFO("DuniaTaa: structural: an RGBA8 image {}x{} copied into an sRGB texture (the eye copy-out) "
                 "-- the size learners have eye {}x{}",
                 w, h, g_destW, g_destH);
    static UINT pendW = 0, pendH = 0;
    static uint64_t pendPresent = 0;
    static int pendCount = 0;
    const UINT curW = g_vrEyeW.load(std::memory_order_relaxed), curH = g_vrEyeH.load(std::memory_order_relaxed);
    if (w == curW && h == curH)
    {
        pendCount = 0;
        return;
    }
    if (w == pendW && h == pendH)
    {
        if (now != pendPresent)
        {
            ++pendCount;
            pendPresent = now;
        }
    }
    else
    {
        pendW = w;
        pendH = h;
        pendCount = 1;
        pendPresent = now;
    }
    if (curW != 0 && pendCount < 3)
        return;
    g_vrEyeW.store(w, std::memory_order_relaxed);
    g_vrEyeH.store(h, std::memory_order_relaxed);
    pendCount = 0;
    LOG_INFO("DuniaTaa: VR eye is now {}x{} (the eye copy-out, was {}x{}) -- it gates the size learners", w, h,
             curW, curH);
}

// Once per change of any of the four, at Present: the size learners' scene and eye beside the structural ones. The
// line to read when F6 does nothing -- "disagree" names which learner went wrong. Capped at 40 lines a session.
static void LogSizeViews()
{
    static unsigned last[6] {};
    static int lines = 0;
    const unsigned now[6] = { SceneW(), SceneH(), g_destW, g_destH, g_structSceneW.load(), g_structSceneH.load() };
    const unsigned eyeW = g_vrEyeW.load(), eyeH = g_vrEyeH.load();
    if (std::memcmp(now, last, sizeof(now)) == 0 || lines >= 40)
        return;
    std::memcpy(last, now, sizeof(now));
    ++lines;
    const bool sceneAgrees = now[4] == 0 || (now[0] == now[4] && now[1] == now[5]);
    const bool eyeAgrees = eyeW == 0 || (std::abs((int) now[2] - (int) eyeW) <= 32 &&
                                         std::abs((int) now[3] - (int) eyeH) <= 32);
    LOG_INFO("DuniaTaa: sizes: learned scene {}x{}{} eye {}x{} | structural scene {}x{} eye {}x{} -> scene {}, eye {}",
             now[0], now[1], SceneScaled() ? " (scaled)" : "", now[2], now[3], now[4], now[5], eyeW, eyeH,
             now[4] == 0 ? "not seen yet" : sceneAgrees ? "agree" : "DISAGREE",
             eyeW == 0 ? "not seen yet" : eyeAgrees ? "agree" : "DISAGREE");
}

void DuniaTaa_Dx12::NoteGraphicsTable(UINT rootParameterIndex, D3D12_GPU_DESCRIPTOR_HANDLE base,
                                      ID3D12GraphicsCommandList* cmd)
{
    if (g_probeOn.load(std::memory_order_relaxed))
        ProbeTable(true, rootParameterIndex, base, cmd);
    PcWalk(true, rootParameterIndex, base, cmd);
    if (base.ptr == 0 || cmd == nullptr || !WantsTaaSearch())
        return;
    // The search normally stops the moment the resolve is identified. Splice mode needs one thing more:
    // which colour buffer goes with which render target, for every instance the game ping-pongs between,
    // because the splice must read that colour at every draw and only the target is free there. So it
    // keeps walking tables until the pairs have stopped changing -- a second or so of the mirroring, not
    // a session of it.
    // A square-eye hold (Note) keeps the structural walk below running past that point, so the held size can be
    // confirmed; the search itself still stops.
    const bool searchDone = g_taaFound.load(std::memory_order_relaxed) &&
                            (!SpliceTaaMode() || g_taaPairsDone.load(std::memory_order_relaxed));
    if (searchDone && g_presentIndex.load() >= g_structRefreshUntil.load(std::memory_order_relaxed))
        return;
    auto heap = ResTrack_Dx12::GetTableHeap(true, (SIZE_T) base.ptr);
    if (heap == nullptr || heap->increment == 0)
        return;

    // Structural: the same signature, but at whatever size the table's RGBA16F have -- ahead of the pinned-size check
    // below, so it sees the resolve even when the size learners have the wrong scene.
    {
        UINT hdrW[12], hdrH[12], mvW[12], mvH[12];
        int hdr = 0, mv = 0;
        for (UINT i = 0; i < 12; ++i)
        {
            const SIZE_T handle = (SIZE_T) base.ptr + (SIZE_T) i * heap->increment;
            if (handle >= heap->gpuEnd)
                break;
            ResourceInfo info {};
            if (!heap->GetByGpuHandle(handle, info) || info.buffer == nullptr || info.width < 128 || info.height < 128)
                continue;
            if (info.format == DXGI_FORMAT_R16G16B16A16_FLOAT)
            {
                hdrW[hdr] = (UINT) info.width;
                hdrH[hdr++] = info.height;
            }
            else if (info.format == DXGI_FORMAT_R8G8_UNORM)
            {
                mvW[mv] = (UINT) info.width;
                mvH[mv++] = info.height;
            }
        }
        for (int a = 0; a < hdr && mv >= 2; ++a)
        {
            int same = 0, half = 0;
            for (int b = 0; b < hdr; ++b)
                same += hdrW[b] == hdrW[a] && hdrH[b] == hdrH[a];
            for (int m = 0; m < mv; ++m)
                half += std::abs((int) mvW[m] * 2 - (int) hdrW[a]) <= 32 && std::abs((int) mvH[m] * 2 - (int) hdrH[a]) <= 32;
            if (same >= 2 && half >= 2)
            {
                StructNoteScene(hdrW[a], hdrH[a]);
                break;
            }
        }
    }

    if (searchDone || g_pinnedW == 0 || g_pinnedH == 0)
        return;

    // The signature, counted rather than positional: slot order is the root signature's business, and
    // trailing entries may belong to whatever sits next in the heap.
    int halfResMotion = 0;
    int eyeHdr = 0;
    for (UINT i = 0; i < 12; ++i)
    {
        const SIZE_T handle = (SIZE_T) base.ptr + (SIZE_T) i * heap->increment;
        if (handle >= heap->gpuEnd)
            break;
        ResourceInfo info {};
        if (!heap->GetByGpuHandle(handle, info) || info.buffer == nullptr)
            continue;
        if (info.format == DXGI_FORMAT_R8G8_UNORM && HalfEyeSized(info.width, info.height))
            ++halfResMotion;
        else if (info.format == DXGI_FORMAT_R16G16B16A16_FLOAT && info.height == SceneH() &&
                 (UINT) info.width == SceneW())
            ++eyeHdr;
    }

    // Both motion instances and at least two of the three colour buffers. Nothing else in a Dunia frame
    // reads the two half-res motion textures together with eye-sized HDR colour.
    if (halfResMotion + eyeHdr > g_taaBestMotion.load(std::memory_order_relaxed) +
                                    g_taaBestHdr.load(std::memory_order_relaxed))
    {
        g_taaBestMotion.store(halfResMotion, std::memory_order_relaxed);
        g_taaBestHdr.store(eyeHdr, std::memory_order_relaxed);
    }
    if (halfResMotion < 2 || eyeHdr < 2)
        return;

    g_taaRootParameter = rootParameterIndex;
    // Re-walked rather than collected above, because only now is it worth keeping: the eye-sized RGBA16F
    // slots in table order are the resolve's colour inputs, and slot order is register order for a
    // contiguous SRV range.
    {
        TaaResolveMap::Colour found[kTaaColourSlots] {};
        int colour = 0;
        for (UINT i = 0; i < 12 && colour < kTaaColourSlots; ++i)
        {
            const SIZE_T handle = (SIZE_T) base.ptr + (SIZE_T) i * heap->increment;
            if (handle >= heap->gpuEnd)
                break;
            ResourceInfo info {};
            if (!heap->GetByGpuHandle(handle, info) || info.buffer == nullptr)
                continue;
            if (info.format == DXGI_FORMAT_R16G16B16A16_FLOAT && info.height == SceneH() &&
                (UINT) info.width == SceneW())
            {
                found[colour++] = { info.buffer, (UINT) info.width, info.height, info.format };
            }
        }
        // t2, for the splice: the first eye-sized RGBA16F in table order, which is register order for a
        // contiguous SRV range, and is SceneColor in FC5's pixel shader. Its shape is taken here, while
        // the game still has it bound and it is certainly alive, rather than read back off the pointer a
        // Present later when it may not be.
        if (colour > 0)
            g_taaResolve.NoteColour(cmd, found[0].colour, found[0].width, found[0].height, found[0].format);
        // The one-time log below names all three. Written only while the resolve is still unidentified,
        // so this stays a single-writer global even though the search now runs on past that point.
        if (!g_taaFound.load(std::memory_order_relaxed))
        {
            for (int i = 0; i < kTaaColourSlots; ++i)
                g_taaColour[i] = found[i].colour;
        }
    }

    std::lock_guard<std::mutex> lock(g_taaLock);
    g_taaCandidate.insert(cmd);
}

bool DuniaTaa_Dx12::WantsMotionSearch()
{
    // Real motion needs the pass found before it can copy anything, so it implies the search.
    return Active() && (Config::Instance()->DuniaTaaFindMotion.value_or_default() || WantsRealMotion());
}

const char* DuniaTaa_Dx12::RealMotionName()
{
    if (!Config::Instance()->DuniaTaaRealMotion.value_or_default())
        return "Real motion off (startup)";
    switch (g_realMotionMode.load())
    {
    case kMotionGame:
        return "Motion: game's own (other eye)";
    case kMotionMatrix:
        return "Motion: MATRIX, same eye (free)";
    case kMotionMatrixObjects:
        if (!g_sameEyePatchOn.load())
            return "Motion: MATRIX + the game's object motion (steady AO; small movers rougher)";
        return g_sameEyeBlockOn.load()
                   ? "Motion: MATRIX + SAME-EYE object motion + the motion pass's matrix patched too"
                   : "Motion: MATRIX + SAME-EYE object motion (better NPCs; AO shimmers when the head moves)";
    case kMotionChained:
        return "Motion: chained (known bad)";
    default:
        return "Motion off";
    }
}

void DuniaTaa_Dx12::ToggleRealMotion()
{
    if (!WantsRealMotion())
    {
        LOG_INFO("DuniaTaa: F4: real motion is off in the ini, nothing to toggle");
        return;
    }
    // Everyday: off <-> MATRIX + object motion, the validated state, one press each way.
    // With diagnostics on, the full cycle for A/Bs -- off -> the game's own -> matrix -> matrix + objects ->
    // chained only when asked for in the ini (the one both games rejected should not be a keypress away).
    // The object state learns the motion pass itself, so nothing needs the game's-own state first.
    // [Inputs] DuniaTaaSameEyeVectors: in everyday play the on state IS the patched one (validated)
    // -- off <-> MATRIX + same-eye object motion. With diagnostics on, the unpatched object state
    // stays in the cycle just before it, so the A/B is still one press inside one run. Leaving the patched
    // state stops the writes at once; the game rewrites its struct on that eye's next frame.
    g_mvWindowWasOn = g_realMotionOn.load();
    g_sameEyeWindowWas = g_sameEyePatchOn.load();
    // Diagnostics only: validated same-eye state -> the same plus the motion pass's matrix (rejected, see
    // g_sameEyeBlockOn) -> on to the next state.
    if (DiagnosticsOn() && SameEyeWanted() && g_realMotionMode.load() == kMotionMatrixObjects &&
        g_sameEyePatchOn.load() && !g_sameEyeBlockOn.load())
    {
        g_sameEyeBlockOn = true;
        g_mvStatsRestart.store(true, std::memory_order_relaxed);
        LOG_INFO("DuniaTaa: F4: {}", RealMotionName());
        return;
    }
    g_sameEyeBlockOn = false;
    // In BOTH cycles: the same-eye patch makes the game's ambient occlusion shimmer when the
    // head moves (its temporal filter reads the same previous view, and its history is the other eye's; there
    // is one struct per frame, so the two cannot be separated by copy). Better NPCs or steady AO is the player's
    // choice, per scene: off -> object motion (steady AO) -> same-eye object motion (better NPCs) -> off.
    if (SameEyeWanted() && g_realMotionMode.load() == kMotionMatrixObjects && !g_sameEyePatchOn.load())
    {
        g_sameEyePatchOn = true;
        g_mvStatsRestart.store(true, std::memory_order_relaxed);
        LOG_INFO("DuniaTaa: F4: {}", RealMotionName());
        return;
    }
    g_sameEyePatchOn = false;

    int mode;
    if (!DiagnosticsOn())
    {
        mode = g_realMotionMode.load() == kMotionOff ? kMotionMatrixObjects : kMotionOff;
    }
    else
    {
        const bool withChained = Config::Instance()->DuniaTaaChainedMotion.value_or_default();
        mode = g_realMotionMode.load() + 1;
        if (mode == kMotionChained && !withChained)
            mode = kMotionOff;
        else if (mode > kMotionChained)
            mode = kMotionOff;
    }
    ApplyRealMotionMode(mode);
}

// The motion state change itself, shared by F4 and by the one-key F6 (see DriveOneKey).
void ApplyRealMotionMode(int mode)
{
    g_realMotionMode = mode;
    const bool on = mode != kMotionOff;
    g_realMotionOn = on;
    // The mirroring is the bulk of the cost, so the toggle has to take it with it or the A/B measures
    // only half of what the path charges for.
    // The mirroring is decided in one place now, from every reason that might still need it.
    // The decode caches an SRV of whatever it last read, and the two paths read different textures.
    g_mvViewsValid = false;
    g_needReset = true;
    // Closes the open stats window: one that straddles a toggle belongs to neither state, and every window
    // in the first two runs did, which made the counts unreadable.
    g_mvStatsRestart.store(true, std::memory_order_relaxed);
    LOG_INFO("DuniaTaa: F4: {} (heap mirroring {})", DuniaTaa_Dx12::RealMotionName(),
             NeedsGameVectors() ? "on until the pass is found" : "off");
}

namespace
{
// Half the eye size, within the same slack the rest of this module allows for rounding.
bool HalfEyeSized(UINT64 width, UINT height)
{
    if (SceneW() == 0 || SceneH() == 0 || width == 0 || height == 0)
        return false;
    return std::abs((int) (width * 2) - (int) SceneW()) <= 32 &&
           std::abs((int) (height * 2) - (int) SceneH()) <= 32;
}

// Gameplay, by the same signature automatic mode uses: the G-buffer bind, which menus never draw. Both
// searches wait for it. Without this they ran during loading -- FC6's menus render at the *full eye size*,
// so the pinned shape is set early and nothing later changes it to restart a deadline on. A 3000-Present
// window then expired in 17 seconds at 200+ Presents/s, before the world was up, and neither search had
// seen a single gameplay frame. This is the "gate on a gameplay signature" lesson FC5 paid for.
bool GameplayNow()
{
    const uint64_t now = g_presentIndex.load();
    const uint64_t full = g_gameplayFullPresent.load();
    const uint64_t scaled = g_gameplayScaledPresent.load();
    return (full != 0 && now - full < 30) || (scaled != 0 && now - scaled < 30);
}

// One-key F6 (see OneKeyMode): once F6 is on and the feature exists, splice DLSS at the resolve the moment
// the resolve is identified, and turn object motion on. Each step is the same call the separate key made,
// and each says so in the log once per arming.
void DriveOneKey()
{
    if (!g_oneKeyArm.load(std::memory_order_relaxed) || !OneKeyMode() || !Started() || g_handle == nullptr)
        return;
    bool taaDone = true;
    if (SpliceTaaMode() && DuniaTaa_Dx12::WantsTaaSearch())
    {
        if (g_taaMode.load() == 2)
            taaDone = true;
        else if (g_taaFound.load(std::memory_order_relaxed))
        {
            LOG_INFO("DuniaTaa: F6 (one key): the TAA resolve is identified -- splicing DLSS in its place");
            ApplyTaaMode(2);
        }
        else
            taaDone = false;
    }
    bool motionDone = true;
    if (DuniaTaa_Dx12::WantsRealMotion() && g_realMotionMode.load() == kMotionOff)
    {
        LOG_INFO("DuniaTaa: F6 (one key): turning matrix + object motion on");
        g_sameEyeBlockOn = false;
        g_sameEyePatchOn = false;
        ApplyRealMotionMode(kMotionMatrixObjects);
    }
    if (taaDone && motionDone)
        g_oneKeyArm = false;
}

// The upscale writer probe's clock. Every Present, ahead of NeedsHeapTracking, which holds the mirroring on
// for it. Auto-armed once per session, the first time the resolve splice has run in gameplay below 100%;
// F10 re-arms it. The window is short and bounded because the mirroring costs about half the frame rate.
void DriveUpscaleWriterProbe()
{
    const uint64_t now = g_presentIndex.load();
    if (g_probeOn.load(std::memory_order_relaxed))
    {
        const bool spliced = g_resolveSplicedPresent == now;
        if (spliced)
            g_probeSplicedPresents++;
        if (now < g_probeSeqUntil.load(std::memory_order_relaxed))
        {
            std::lock_guard<std::mutex> lock(g_probeLock);
            if (g_probeSeq.size() < kProbeSeqMax)
                g_probeSeq.push_back(ProbeEvent { now, nullptr, nullptr, ProbeKind::Present, spliced ? 1 : 0, 0,
                                                  spliced, false, 0, GetCurrentThreadId(), 0 });
        }
        if (now >= g_probeUntil)
        {
            g_probeOn.store(false, std::memory_order_relaxed);
            ResTrack_Dx12::SetMirrorAllHeaps(false);
            ProbeSummary();
            if (g_probeSplicedPresents == 0)
                LOG_WARN("DuniaTaa: upscale writer probe: the window had NO resolve splice in it -- it ran in a menu "
                         "or with DLSS off, and everything it logged is about the wrong frames. Press F10 in "
                         "gameplay with DLSS spliced (600/600 in the 'TAA was DLSS spliced' line) to run it again");
        }
        return;
    }
    const bool arm = g_probeArm.exchange(false, std::memory_order_relaxed);
    if (!SpliceTaaMode() || !IsUpscale() || g_output == nullptr || g_createdOutW == 0)
    {
        if (arm)
            LOG_INFO("DuniaTaa: upscale writer probe: F10 ignored -- not upscaling in splice mode");
        return;
    }
    // Armed on the SPLICE having just run, never on the gameplay signature: the first run armed on
    // GameplayNow(), which was false for 100 s of 70% gameplay and became true the moment a menu opened
    // (menus render at the full eye size -- a known trap), so the window
    // recorded menu frames. A splice within the last Present is gameplay with DLSS on, by construction.
    const bool splicedNow = g_resolveSplicedPresent != UINT64_MAX && now - g_resolveSplicedPresent <= 1;
    // The probe found the upscale writer and nothing depends on it: the copy over the game's upscale
    // and the live rebuild both run without it. Auto-arming only with diagnostics, so an everyday first drop
    // below 100% is not ~5 s of every heap mirrored (one run: 71.5/s). F10 still runs it on request.
    if (!arm && (g_probeDone || !splicedNow || !DiagnosticsOn()))
        return;
    if (arm && !splicedNow)
        LOG_WARN("DuniaTaa: upscale writer probe: F10 with no splice in the last Present -- running anyway, but "
                 "the window will be about the wrong frames unless DLSS is spliced right now");
    g_probeDone = true;
    // Every heap, not the motion pass's two: once those are learned the mirroring is narrowed to them, and
    // the first run walked 70k compute tables it could not resolve for that reason.
    ResTrack_Dx12::SetMirrorAllHeaps(true);
    g_probeFrom = now;
    g_probeUntil = now + kProbePresents;
    g_probeSplicedPresents = 0;
    g_probeSeqUntil.store(now + kProbeSeqPresents, std::memory_order_relaxed);
    g_probeOn.store(true, std::memory_order_relaxed);
    LOG_INFO("DuniaTaa: upscale writer probe: ON for Presents {}..{} ({}) -- every OMSet, table SRV/UAV, copy, resolve, "
             "clear and barrier of any {}x{} texture (any format, not ours) is logged once per kind, the first {} "
             "Presents' event sequence is dumped at the end with the submits inline, and the heap mirroring is on "
             "meanwhile (the frame rate drops for the window). F10 re-arms it",
             now, g_probeUntil, arm ? "F10" : "auto-armed at the first gameplay splice", g_createdOutW, g_createdOutH,
             kProbeSeqPresents);
}

// Every reason the descriptor route might still be needed, in one place. Scattered on/off calls are what
// left the TAA search unable to resolve anything: the motion search had already switched the mirroring off
// before the key was ever pressed.
// ---- private-copy feasibility probe ---------------------------------------------------------------------
// The question: same-eye object vectors need the GEOMETRY passes to see a same-eye previous
// view, while the game's AO and other temporal filters keep theirs -- patching the shared struct made AO shimmer.
// Can the geometry alone be given a private copy? That depends on HOW the per-view struct reaches each pass:
// through which CBV descriptors, in which tables, and which of the frame's three copies. This probe records it
// for kPcWindow gameplay Presents and ends with a verdict. One-shot, armed from the overlay; mirrors every heap
// and records CBVs for its window only (the frame rate drops for ~4 s).
constexpr uint64_t kPcWarmup = 60;  // mirroring + CBV recording on before counting, so the descriptors are known
constexpr uint64_t kPcWindow = 240;
constexpr UINT kPcSlots = 32;
constexpr uint64_t kPcStructSize = 1024;
enum PcClass
{
    PcGbuffer = 0, // a graphics table bound while the G-buffer (R16G16 motion target + depth) is the list's target
    PcGraphics,    // any other graphics table
    PcCompute,
    PcClassCount
};
const char* const kPcClassNames[PcClassCount] = { "G-buffer", "other graphics", "compute" };
// g_pcState / g_pcArm are declared with the other view globals near the top (the CBV hook reads the state).
uint64_t g_pcFrom = 0, g_pcUntil = 0, g_pcSpliced = 0, g_pcCbvAtStart = 0, g_pcRecordedAtStart = 0; // present thread
struct PcRange
{
    uint64_t va;
    int copy;
};
struct PcStat
{
    uint64_t n[PcClassCount] {};
    int copy = -1;
    UINT root = 0;
    UINT slot = 0;
    uint64_t offset = 0;
};
// Everything below is touched from the recording threads (the walk) and the present thread (refresh, summary).
std::mutex g_pcLock;
PcRange g_pcRanges[kVcMaxCandidates] {};
int g_pcRangeCount = 0;
std::unordered_map<uint64_t, PcStat> g_pcDescs;   // by the GPU handle of the CBV's slot
std::unordered_map<uint64_t, uint32_t> g_pcTables; // table base -> mask of classes that bound it with a hit
uint64_t g_pcByCopy[5][PcClassCount] {};           // copy -1..3
uint64_t g_pcWalked[PcClassCount] {};
uint64_t g_pcCbvSlots = 0, g_pcHits = 0, g_pcEmpty = 0, g_pcNoHeap = 0;
// What the probe NEARLY matched, so a miss says why: CBVs per class, those of the struct's size (1 KB), those
// within 1 MB of a known copy (the same ring, a slot we do not know), and the G-buffer's own CBVs by address.
constexpr uint64_t kPcNearBytes = 1ull << 20;
constexpr size_t kPcGbufCbvsMax = 24;
uint64_t g_pcCbvByClass[PcClassCount] {}, g_pcKb[PcClassCount] {}, g_pcNear[PcClassCount] {};
struct PcGbufCbv
{
    uint64_t n[PcClassCount] {}; // binds by class: whether the G-buffer's buffers are ALSO read by other passes
    UINT size = 0;
    int64_t toCopy = INT64_MAX; // signed distance to the nearest known copy's start
};
// Every CBV the G-buffer binds, and every one within 1 MB of a known copy (the ring) whoever binds it -- so a ring
// slot another pass binds first is still counted for that pass. Bounded: the ring is ~1 KB slots.
constexpr size_t kPcCbvMapMax = 8192;
std::unordered_map<uint64_t, PcGbufCbv> g_pcGbufCbvs;
// Who reads which copy: [graphics, root, slot, copy+1, rtCount, dsFmt, rtFmt x8] -> binds. Under g_pcLock.
using PcPassKey = std::array<uint16_t, 14>;
std::map<PcPassKey, uint64_t> g_pcPasses;

// ---- velocity readers ------------------------------------------------------------------------------------------
// The in-place G-buffer rewrite made idle NPCs light up and the AO still shimmered, with ONLY the G-buffer given the
// same-eye copy: the AO must read the G-buffer's velocity (the R8G8_UNORM target) itself. The planned fix converts
// that target back to other-eye after our copy and before the AO reads it. This records what that needs: the
// texture (and whether compute may write it), every pass that reads it, and each read's place in SUBMISSION order
// relative to the motion pass. Control: the motion pass reads it too -- if it is not listed, the walk was blind.
std::atomic<ID3D12Resource*> g_velTex { nullptr }; // identity only, never AddRef'd, never read through
// The motion pass's inputs, named once each (under g_pcLock). g_velTex is now its t3: an earlier build took the
// G-buffer's R8G8_UNORM target for the velocity and its own control failed -- the motion pass never read it.
// Flagged pixels copy t3.xy, an R8G8B8A8_UNORM target.
std::set<uint64_t> g_mvInputsSeen;
std::atomic<ID3D12Resource*> g_velDepth { nullptr }; // the motion pass's t1; identity only
// Restorability at the motion pass dispatch (the conversion's injection point), under g_pcLock.
uint64_t g_rsDispatches = 0, g_rsComplete = 0, g_rsNoSig = 0, g_rsNotCompute = 0, g_rsNoParams = 0,
         g_rsInvalidParams = 0, g_rsNoHeaps = 0, g_rsNoPipeline = 0;
int g_rsLogged = 0;
struct VelEvent
{
    ID3D12GraphicsCommandList* list;
    uint64_t present;
    uint32_t seq;  // order within the list's recording
    uint16_t kind; // 0 a read of the velocity target, 1 the motion pass dispatch, 2 a G-buffer target bind,
                   // 3 a barrier on the velocity texture, 4 a barrier on the motion pass's depth
    uint32_t reader;
    uint32_t before = 0, after = 0; // barrier states (kinds 3 and 4)
};
struct VelSubmit
{
    ID3D12GraphicsCommandList* list;
    uint64_t present;
    uint32_t ordinal; // position among this Present's submitted lists
};
using VelKey = std::array<uint64_t, 3>; // [graphics<<48 | root<<32 | slot<<16 | rtCount, pipeline, rtFmt0..3]
std::vector<VelEvent> g_velEvents;       // under g_pcLock
std::vector<VelSubmit> g_velSubmits;     // under g_pcLock
std::unordered_map<ID3D12GraphicsCommandList*, uint32_t> g_velSeq;
std::map<VelKey, uint32_t> g_velReaderIds;
std::vector<VelKey> g_velReaderKeys;
std::vector<uint64_t> g_velReaderBinds;
uint64_t g_velSubmitPresent = UINT64_MAX;
uint32_t g_velSubmitOrdinal = 0;
constexpr size_t kVelMaxEvents = 400000, kVelMaxSubmits = 100000;
void VelNoteLocked(ID3D12GraphicsCommandList* cmd, uint16_t kind, uint32_t reader)
{
    if (g_velEvents.size() >= kVelMaxEvents)
        return;
    g_velEvents.push_back({ cmd, g_presentIndex.load(std::memory_order_relaxed), g_velSeq[cmd]++, kind, reader });
}
void VelNoteSubmits(ID3D12CommandList* const* lists, UINT count)
{
    std::lock_guard<std::mutex> lock(g_pcLock);
    const uint64_t present = g_presentIndex.load(std::memory_order_relaxed);
    if (present != g_velSubmitPresent)
    {
        g_velSubmitPresent = present;
        g_velSubmitOrdinal = 0;
    }
    for (UINT i = 0; i < count && g_velSubmits.size() < kVelMaxSubmits; ++i)
        g_velSubmits.push_back({ (ID3D12GraphicsCommandList*) lists[i], present, g_velSubmitOrdinal++ });
}

// ---- table-rewrite feasibility -------------------------------------------------------------------------------
// The probe said SHARED: the G-buffer and the AO/temporal passes read the per-view struct through the SAME
// descriptors, so a geometry-only copy means binding a table of our own at the G-buffer's per-view binds. Three
// facts decide whether that can be done without guessing, and a guess here is a GPU fault, not a bad picture:
//   1. the table's DECLARED layout (the root signature's ranges), so the replacement holds everything the shader
//      reads -- not what the walk happened to see;
//   2. whether every slot in it is a CBV we recorded (address + size), because a shader-visible heap cannot be
//      copied FROM, so the table must be rebuilt from what we know;
//   3. slots of the game's shader-visible heap the game never writes -- tracked FROM STARTUP -- to hold it.
struct RsRange
{
    D3D12_DESCRIPTOR_RANGE_TYPE type;
    UINT count; // UINT_MAX = unbounded
    UINT offset;
};
struct RsParam
{
    D3D12_ROOT_PARAMETER_TYPE type;
    std::vector<RsRange> ranges;
};
std::mutex g_rsLock;
std::unordered_map<ID3D12RootSignature*, std::vector<RsParam>> g_rootSigs; // identity only, never AddRef'd
std::unordered_map<ID3D12GraphicsCommandList*, ID3D12RootSignature*> g_listRootSig; // only while the probe runs
constexpr size_t kRootSigsMax = 8192;

constexpr int kSvHeapsMax = 32; // our own small heaps come through the same hook
constexpr int kSvBuckets = 64;
struct SvHeap
{
    ID3D12DescriptorHeap* heap = nullptr; // identity only
    SIZE_T cpuStart = 0, cpuEnd = 0;
    SIZE_T gpuStart = 0;
    UINT num = 0, inc = 0;
    std::atomic<int64_t> high { -1 };
    std::atomic<uint64_t> writes { 0 };
    std::atomic<uint64_t> bucket[kSvBuckets] {};
    // The private copy's tables. Big heaps only (the game's per-eye pair, 456192 each): per slot, the CPU
    // descriptor it was last COPIED from (0 = created in place, or never written) -- a shader-visible heap cannot
    // be copied from, so this is how a table is rebuilt; one relaxed store per descriptor the game copies. And our
    // region at the top, which the game has never written since startup (5800+ slots measured): if it ever does,
    // regionBad switches the redirect off for this heap for the rest of the session.
    std::atomic<SIZE_T>* srcMap = nullptr;
    UINT regionStart = 0;
    std::atomic<bool> regionBad { false };
    std::atomic<uint32_t> rwNext { 0 };
};
constexpr UINT kRwBigHeap = 16384;
// 256 tables of 16 in the top 4096 slots (the game's highest write since startup was ~450400 of 456192, so
// ~1700 slots of margin below the region). A table is reused after 256 binds; with at most kRwBudget redirects a
// Present that is at least 5 Presents, well past the GPU. The budget is what makes the wider scopes safe: one run's
// counters show ~2700 qualifying binds a Present outside the slot-0 lists, which would have recycled a table the
// GPU had not yet read, many times over.
constexpr UINT kRwRegionSlots = 4096;
constexpr uint32_t kRwBudget = 48;
std::atomic<uint32_t> g_rwThisPresent { 0 };
constexpr UINT kRwStride = 16;
thread_local bool t_ownWrite = false; // our own descriptor writes: not the game reaching our region
// The root signatures whose parameter 1 is ONE CBV range at offset 0 (the G-buffer's per-view table: 15 CBVs),
// and the table size. Written under g_rsLock, read lock-free at every root-1 bind.
constexpr int kRwSigsMax = 64;
std::atomic<ID3D12RootSignature*> g_rwSigs[kRwSigsMax] {};
std::atomic<UINT> g_rwSigSize[kRwSigsMax] {};
std::atomic<int> g_rwSigCount { 0 };
// Per LIST, not per thread: Dunia records through a job system, so a list's root signature can be set on one
// thread and its tables bound from another. An earlier build kept "this thread's last root signature" and redirected
// nothing, silently (the check was one of the cheap early returns). Sharded by list pointer so the recording
// threads rarely meet; only kept while the private copy is on. Value: the qualifying table size, 0 = not ours.
constexpr int kRsShards = 16;
struct RsShard
{
    std::mutex m;
    std::unordered_map<ID3D12GraphicsCommandList*, UINT> size;
};
RsShard g_rsShards[kRsShards];
RsShard& RsShardOf(const void* p) { return g_rsShards[((uintptr_t) p >> 6) % kRsShards]; }
// The cheap exits, counted too (while the mode is on): a silent early return is what hid that failure.
std::atomic<uint64_t> g_rwListUnseen { 0 }, g_rwNotOurs { 0 }, g_rwNotGbuffer { 0 }, g_rwCandidates { 0 };
UINT RwTableSize(ID3D12RootSignature* sig)
{
    const int n = g_rwSigCount.load(std::memory_order_acquire);
    for (int k = 0; k < n; ++k)
        if (g_rwSigs[k].load(std::memory_order_relaxed) == sig)
            return g_rwSigSize[k].load(std::memory_order_relaxed);
    return 0;
}
enum class RwSkip
{
    Budget,
    NoHeap,
    RegionWritten,
    OutOfHeap,
    UnknownSource,
    Slot0NotCbv,
    NotACopy,
    Count
};
const char* const kRwSkipNames[(int) RwSkip::Count] = {
    "this Present's redirect budget is spent (a table must outlive the GPU's use of it)",
    "the table is not in a big tracked shader-visible heap",
    "the game has written into our region of that heap -- redirect OFF for it",
    "the table runs past the end of its heap",
    "a slot's source descriptor is unknown (created in place, or copied before tracking)",
    "slot 0's source is not a CBV we recorded",
    "slot 0 is not a per-view struct the mirror holds this frame (normal for a copy not yet found)",
};
std::atomic<uint64_t> g_rwSkips[(int) RwSkip::Count] {};
std::atomic<uint64_t> g_rwRedirects { 0 };
void CountRwSkip(RwSkip why)
{
    static std::atomic<bool> logged[(int) RwSkip::Count] {};
    g_rwSkips[(int) why].fetch_add(1, std::memory_order_relaxed);
    if (!logged[(int) why].exchange(true, std::memory_order_relaxed))
        LOG_INFO("DuniaTaa: private copy: a G-buffer per-view bind kept the game's table -- {}",
                 kRwSkipNames[(int) why]);
}
SvHeap g_svHeaps[kSvHeapsMax];
std::atomic<int> g_svCount { 0 };
std::mutex g_svLock; // registration only; writers read g_svCount with acquire
std::atomic<uint64_t> g_svOverflow { 0 };

struct RwLayout
{
    uint64_t binds = 0, reconstructable = 0, nonCbv = 0, unrecorded = 0;
    UINT tableSize = 0, viewSlot = 0, rangeCount = 0;
    bool unbounded = false;
    std::string ranges;
    ID3D12DescriptorHeap* heap = nullptr;
    UINT heapNum = 0;
};
std::map<std::pair<ID3D12RootSignature*, UINT>, RwLayout> g_rwLayouts; // under g_pcLock
uint64_t g_rwBinds = 0, g_rwNoList = 0, g_rwUnknownSig = 0, g_rwNotTable = 0;

const char* RangeTypeName(D3D12_DESCRIPTOR_RANGE_TYPE t)
{
    switch (t)
    {
    case D3D12_DESCRIPTOR_RANGE_TYPE_CBV:
        return "CBV";
    case D3D12_DESCRIPTOR_RANGE_TYPE_SRV:
        return "SRV";
    case D3D12_DESCRIPTOR_RANGE_TYPE_UAV:
        return "UAV";
    default:
        return "SAMPLER";
    }
}

template <typename Param, typename Range>
std::vector<RsParam> ParseRootParams(const Param* params, UINT count)
{
    std::vector<RsParam> out(count);
    for (UINT p = 0; p < count; ++p)
    {
        out[p].type = params[p].ParameterType;
        if (params[p].ParameterType != D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE)
            continue;
        const auto& table = params[p].DescriptorTable;
        UINT next = 0;
        for (UINT r = 0; r < table.NumDescriptorRanges; ++r)
        {
            const Range& range = table.pDescriptorRanges[r];
            const UINT offset = range.OffsetInDescriptorsFromTableStart == D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND
                                    ? next
                                    : range.OffsetInDescriptorsFromTableStart;
            out[p].ranges.push_back({ range.RangeType, range.NumDescriptors, offset });
            next = range.NumDescriptors == UINT_MAX ? UINT_MAX : offset + range.NumDescriptors;
        }
    }
    return out;
}

// The G-buffer bound a table whose slot `viewSlot` is a per-view CBV: what is the table, and could we rebuild it?
// Called from PcWalk with g_pcLock held (lock order g_pcLock -> g_rsLock; nothing takes them the other way).
void RwAnalyse(UINT root, D3D12_GPU_DESCRIPTOR_HANDLE base, ID3D12GraphicsCommandList* cmd,
               const std::shared_ptr<HeapInfo>& heap, UINT viewSlot)
{
    g_rwBinds++;
    ID3D12RootSignature* sig = nullptr;
    std::vector<RsRange> ranges;
    {
        std::lock_guard<std::mutex> lock(g_rsLock);
        const auto l = g_listRootSig.find(cmd);
        if (l == g_listRootSig.end())
        {
            g_rwNoList++;
            return;
        }
        sig = l->second;
        const auto s = g_rootSigs.find(sig);
        if (s == g_rootSigs.end() || root >= s->second.size())
        {
            g_rwUnknownSig++;
            return;
        }
        if (s->second[root].type != D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE)
        {
            g_rwNotTable++;
            return;
        }
        ranges = s->second[root].ranges;
    }
    // The walk reads 32 slots, past the end of short tables: a "hit" at slot 16 of a 16-SRV table is the NEXT
    // table's per-view CBV (one run counted two such SRV tables and wrongly said NOT REBUILDABLE).
    bool inDeclaredCbv = false;
    for (const RsRange& r : ranges)
        inDeclaredCbv |= r.type == D3D12_DESCRIPTOR_RANGE_TYPE_CBV && viewSlot >= r.offset &&
                         (r.count == UINT_MAX || viewSlot < r.offset + r.count);
    if (!inDeclaredCbv)
        return;
    RwLayout& L = g_rwLayouts[{ sig, root }];
    if (L.binds == 0 && g_rwLayouts.size() > 32)
    {
        g_rwLayouts.erase({ sig, root });
        return;
    }
    if (L.binds == 0)
    {
        L.viewSlot = viewSlot;
        L.rangeCount = (UINT) ranges.size();
        L.heap = heap->heap;
        L.heapNum = heap->numDescriptors;
        for (const RsRange& r : ranges)
        {
            if (!L.ranges.empty())
                L.ranges += ", ";
            L.ranges += RangeTypeName(r.type);
            L.ranges += r.count == UINT_MAX ? std::string(" x unbounded") : " x" + std::to_string(r.count);
            L.ranges += " @" + std::to_string(r.offset);
            if (r.count == UINT_MAX)
                L.unbounded = true;
            else
                L.tableSize = std::max(L.tableSize, r.offset + r.count);
        }
    }
    L.binds++;
    bool ok = !L.unbounded;
    for (const RsRange& r : ranges)
    {
        if (r.count == UINT_MAX)
            continue;
        if (r.type != D3D12_DESCRIPTOR_RANGE_TYPE_CBV)
        {
            L.nonCbv++;
            ok = false;
            continue;
        }
        for (UINT i = 0; i < r.count; ++i)
        {
            const SIZE_T handle = (SIZE_T) base.ptr + (SIZE_T) (r.offset + i) * heap->increment;
            ResourceInfo info {};
            if (handle >= heap->gpuEnd || !heap->GetByGpuHandle(handle, info) || info.type != CBV ||
                info.cbvAddress == 0)
            {
                L.unrecorded++;
                ok = false;
            }
        }
    }
    if (ok)
        L.reconstructable++;
}

void PcWalk(bool graphics, UINT root, D3D12_GPU_DESCRIPTOR_HANDLE base, ID3D12GraphicsCommandList* cmd)
{
    if (g_pcState.load(std::memory_order_relaxed) != 2 || base.ptr == 0 || cmd == nullptr || cmd == g_cmd ||
        g_inEvaluate > 0)
        return;
    PcClass cls = graphics ? PcGraphics : PcCompute;
    PcPassKey key {};
    key[0] = graphics ? 1 : 0;
    key[1] = (uint16_t) root;
    if (graphics)
    {
        // Decided at bind time in OnSetTargets; the target pointers are NOT read here (they can be stale).
        std::lock_guard<std::mutex> lock(g_lock);
        const auto it = g_cmdTargets.find(cmd);
        if (it != g_cmdTargets.end() && it->second.gbuffer)
            cls = PcGbuffer;
        if (it != g_cmdTargets.end())
        {
            key[4] = it->second.rtCount;
            key[5] = it->second.dsFmt;
            for (int k = 0; k < 8; ++k)
                key[6 + k] = it->second.rtFmt[k];
        }
    }
    ID3D12PipelineState* listPipeline = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_taaLock);
        const auto p = g_taaListPipeline.find(cmd);
        if (p != g_taaListPipeline.end())
            listPipeline = p->second;
    }
    ID3D12Resource* const velTex = g_velTex.load(std::memory_order_relaxed);
    const auto heap = ResTrack_Dx12::GetTableHeap(graphics, (SIZE_T) base.ptr);
    std::lock_guard<std::mutex> lock(g_pcLock);
    g_pcWalked[cls]++;
    if (heap == nullptr || heap->increment == 0)
    {
        g_pcNoHeap++;
        return;
    }
    int viewSlot = -1; // the first slot of this table holding a per-view CBV
    for (UINT i = 0; i < kPcSlots; ++i)
    {
        const SIZE_T handle = (SIZE_T) base.ptr + (SIZE_T) i * heap->increment;
        if (handle >= heap->gpuEnd)
            break;
        ResourceInfo info {};
        if (!heap->GetByGpuHandle(handle, info))
        {
            g_pcEmpty++;
            continue;
        }
        if (!graphics && info.buffer != nullptr && listPipeline != nullptr && listPipeline == g_mvPipeline)
        {
            const uint64_t slotKey = ((uint64_t) root << 16) | i;
            // What ResTrack recorded when the view was made -- never the resource itself: a pointer from a
            // descriptor record may be stale, and reading through it is how a freed page gets touched.
            if (g_mvInputsSeen.insert(slotKey).second && g_mvInputsSeen.size() <= 48)
                LOG_INFO("DuniaTaa: velocity readers: MOTION PASS input root {} slot {}: {:X} {}x{} format {} flags {:X}{}",
                         root, i, (size_t) info.buffer, info.width, info.height, (int) info.format,
                         (uint32_t) info.flags,
                         (info.flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS) ? " (unordered access ALLOWED)" : "");
            if (i == 1 && root == 2)
                g_velDepth.store(info.buffer, std::memory_order_relaxed);
            // t3: the RGBA8 target the flagged pixels' vectors are copied from, still encoded.
            if (i == 3 && info.format == DXGI_FORMAT_R8G8B8A8_UNORM &&
                g_velTex.exchange(info.buffer, std::memory_order_relaxed) != info.buffer)
                LOG_INFO("DuniaTaa: velocity readers: THE velocity texture is the motion pass's t3 {:X} (root {} slot 3)",
                         (size_t) info.buffer, root);
        }
        if (velTex != nullptr && info.buffer == velTex)
        {
            VelKey vk {};
            vk[0] = ((uint64_t) (graphics ? 1 : 0) << 48) | ((uint64_t) root << 32) | ((uint64_t) i << 16) | key[4];
            vk[1] = (uint64_t) listPipeline;
            vk[2] = (uint64_t) key[6] | ((uint64_t) key[7] << 16) | ((uint64_t) key[8] << 32) | ((uint64_t) key[9] << 48);
            auto id = g_velReaderIds.find(vk);
            if (id == g_velReaderIds.end() && g_velReaderKeys.size() < 512)
            {
                id = g_velReaderIds.emplace(vk, (uint32_t) g_velReaderKeys.size()).first;
                g_velReaderKeys.push_back(vk);
                g_velReaderBinds.push_back(0);
            }
            if (id != g_velReaderIds.end())
            {
                g_velReaderBinds[id->second]++;
                VelNoteLocked(cmd, 0, id->second);
            }
            continue;
        }
        if (info.buffer != nullptr || info.type != CBV || info.cbvAddress == 0)
            continue;
        g_pcCbvSlots++;
        g_pcCbvByClass[cls]++;
        if (info.width == kPcStructSize)
            g_pcKb[cls]++;
        int64_t toCopy = INT64_MAX;
        for (int r = 0; r < g_pcRangeCount; ++r)
        {
            const int64_t d = (int64_t) (info.cbvAddress - g_pcRanges[r].va);
            if (std::llabs(d) < std::llabs(toCopy))
                toCopy = d;
        }
        const bool inRing = toCopy != INT64_MAX && (uint64_t) std::llabs(toCopy) < kPcNearBytes;
        if (inRing)
            g_pcNear[cls]++;
        {
            const auto it = g_pcGbufCbvs.find(info.cbvAddress);
            if (it != g_pcGbufCbvs.end())
                it->second.n[cls]++;
            else if ((cls == PcGbuffer || inRing) && g_pcGbufCbvs.size() < kPcCbvMapMax)
            {
                PcGbufCbv& c = g_pcGbufCbvs[info.cbvAddress];
                c.n[cls] = 1;
                c.size = (UINT) info.width;
                c.toCopy = toCopy;
            }
        }
        for (int r = 0; r < g_pcRangeCount; ++r)
        {
            const PcRange& range = g_pcRanges[r];
            if (info.cbvAddress < range.va || info.cbvAddress >= range.va + kPcStructSize)
                continue;
            g_pcHits++;
            PcStat& s = g_pcDescs[(uint64_t) handle];
            s.n[cls]++;
            s.copy = range.copy;
            s.root = root;
            s.slot = i;
            s.offset = info.cbvAddress - range.va;
            g_pcByCopy[std::clamp(range.copy + 1, 0, 4)][cls]++;
            g_pcTables[(uint64_t) base.ptr] |= 1u << cls;
            {
                PcPassKey k = key;
                k[2] = (uint16_t) i;
                k[3] = (uint16_t) (range.copy + 1);
                if (g_pcPasses.size() < 4096 || g_pcPasses.count(k) != 0)
                    g_pcPasses[k]++;
            }
            if (viewSlot < 0)
                viewSlot = (int) i;
            break;
        }
    }
    if (cls == PcGbuffer && viewSlot >= 0)
        RwAnalyse(root, base, cmd, heap, (UINT) viewSlot);
}

void PcSummary(uint64_t now)
{
    std::lock_guard<std::mutex> lock(g_pcLock);
    LOG_INFO("DuniaTaa: private-copy probe: OVER -- {} Presents ({} with the splice), CBV creates {} / recorded {} in "
             "the probe's life, per-view copies known {}; tables walked: G-buffer {}, other graphics {}, compute {}; "
             "CBV slots {}, per-view hits {}, empty/unresolved slots {}, tables with no tracked heap {}",
             now - g_pcFrom, g_pcSpliced, ResTrack_Dx12::CbvCreateCount() - g_pcCbvAtStart,
             ResTrack_Dx12::CbvRecordedCount() - g_pcRecordedAtStart, g_pcRangeCount, g_pcWalked[PcGbuffer],
             g_pcWalked[PcGraphics], g_pcWalked[PcCompute], g_pcCbvSlots, g_pcHits, g_pcEmpty, g_pcNoHeap);
    LOG_INFO("DuniaTaa: private-copy probe: near misses -- CBVs resolved (G-buffer / other graphics / compute): "
             "{} / {} / {}; of them 1 KB (the struct's size): {} / {} / {}; within 1 MB of a known copy: {} / {} / {}; "
             "CBVs recorded since startup {} (descriptors the game created before that are invisible)",
             g_pcCbvByClass[PcGbuffer], g_pcCbvByClass[PcGraphics], g_pcCbvByClass[PcCompute], g_pcKb[PcGbuffer],
             g_pcKb[PcGraphics], g_pcKb[PcCompute], g_pcNear[PcGbuffer], g_pcNear[PcGraphics], g_pcNear[PcCompute],
             ResTrack_Dx12::CbvRecordedCount());
    {
        std::vector<std::pair<uint64_t, PcGbufCbv>> gb;
        for (const auto& e : g_pcGbufCbvs)
            if (e.second.n[PcGbuffer] > 0)
                gb.push_back(e);
        std::sort(gb.begin(), gb.end(),
                  [](const auto& a, const auto& b) { return a.second.n[PcGbuffer] > b.second.n[PcGbuffer]; });
        for (size_t k = 0; k < gb.size() && k < kPcGbufCbvsMax; ++k)
        {
            const PcGbufCbv& c = gb[k].second;
            LOG_INFO("DuniaTaa: private-copy probe: G-buffer CBV {:X} ({} bytes) bound by G-buffer {}x, other graphics "
                     "{}x, compute {}x; nearest known copy {}",
                     gb[k].first, c.size, c.n[PcGbuffer], c.n[PcGraphics], c.n[PcCompute],
                     c.toCopy == INT64_MAX ? std::string("none known") : std::to_string(c.toCopy) + " bytes away");
        }
        // The G-buffer's hottest ring buffer beside the nearest known per-view copy, row by row (16 bytes each).
        // Whether it IS a per-view struct the fingerprint missed, and whether it carries the previous view (+384)
        // and previous camera (+576) the geometry's vectors are built from, decides the NPC route. Read now, under
        // ConstantCapture's lock -- never through a kept pointer.
        for (const auto& [addr, c] : gb)
        {
            if (c.toCopy == INT64_MAX || c.toCopy == 0 || (uint64_t) std::llabs(c.toCopy) >= kPcNearBytes ||
                c.size != kPcStructSize)
                continue;
            float hot[256] {}, known[256] {};
            const uint64_t knownVa = addr - (uint64_t) c.toCopy;
            if (!ConstantCapture_Dx12::ReadAtGpuAddress(addr, hot, sizeof(hot)) ||
                !ConstantCapture_Dx12::ReadAtGpuAddress(knownVa, known, sizeof(known)))
            {
                LOG_INFO("DuniaTaa: private-copy probe: dump of {:X} vs {:X}: not readable now (mapping gone)", addr,
                         knownVa);
                break;
            }
            std::string differ;
            for (int row = 0; row < 64; ++row)
                if (std::memcmp(hot + row * 4, known + row * 4, 16) != 0)
                    differ += (differ.empty() ? "+" : ",+") + std::to_string(row * 16);
            LOG_INFO("DuniaTaa: private-copy probe: dump: G-buffer's {:X} vs known copy {:X}; rows that differ: {}", addr,
                     knownVa, differ.empty() ? std::string("none (identical)") : differ);
            const int rows[] = { 8, 12, 20, 24, 25, 26, 27, 32, 34, 36 }; // +128 proj, +192, +320 VRP, +384 prev
                                                                        // VRP, +512 [w h 1/w 1/h], +544/+576 cams
            for (const int row : rows)
            {
                const float* h = hot + row * 4;
                const float* k = known + row * 4;
                LOG_INFO("DuniaTaa: private-copy probe: dump +{}: G-buffer's [{:.6g} {:.6g} {:.6g} {:.6g}]  known "
                         "[{:.6g} {:.6g} {:.6g} {:.6g}]",
                         row * 16, h[0], h[1], h[2], h[3], k[0], k[1], k[2], k[3]);
            }
            break; // the hottest one only
        }
    }
    for (int c = 0; c < 5; ++c)
    {
        const uint64_t* n = g_pcByCopy[c];
        if (n[0] + n[1] + n[2] == 0)
            continue;
        LOG_INFO("DuniaTaa: private-copy probe: copy {}: G-buffer {}, other graphics {}, compute {}",
                 c == 0 ? std::string("unknown") : std::to_string(c - 1), n[PcGbuffer], n[PcGraphics], n[PcCompute]);
    }
    std::vector<std::pair<uint64_t, PcStat>> descs(g_pcDescs.begin(), g_pcDescs.end());
    std::sort(descs.begin(), descs.end(),
              [](const auto& a, const auto& b)
              {
                  return a.second.n[0] + a.second.n[1] + a.second.n[2] > b.second.n[0] + b.second.n[1] + b.second.n[2];
              });
    int gbufOnly = 0, otherOnly = 0, shared = 0;
    uint32_t gbufCopies = 0, otherCopies = 0; // bit (copy + 1)
    for (const auto& [handle, s] : descs)
    {
        const bool g = s.n[PcGbuffer] > 0;
        const bool o = s.n[PcGraphics] + s.n[PcCompute] > 0;
        gbufOnly += g && !o;
        otherOnly += o && !g;
        shared += g && o;
        const uint32_t bit = 1u << std::clamp(s.copy + 1, 0, 4);
        if (g)
            gbufCopies |= bit;
        if (o)
            otherCopies |= bit;
    }
    for (size_t k = 0; k < descs.size() && k < 24; ++k)
    {
        const PcStat& s = descs[k].second;
        LOG_INFO("DuniaTaa: private-copy probe: descriptor {:X} (root {}, slot {}) -> copy {} +{}: G-buffer {}, other "
                 "graphics {}, compute {}",
                 descs[k].first, s.root, s.slot, s.copy, s.offset, s.n[PcGbuffer], s.n[PcGraphics], s.n[PcCompute]);
    }
    int tables[8] {};
    for (const auto& [base, mask] : g_pcTables)
        tables[mask & 7]++;
    LOG_INFO("DuniaTaa: private-copy probe: tables holding a per-view CBV: {} G-buffer only, {} other graphics only, "
             "{} compute only, {} bound by the G-buffer AND another pass",
             tables[1], tables[2], tables[4], tables[3] + tables[5] + tables[7]);
    auto copyList = [](uint32_t mask)
    {
        std::string s;
        for (int c = 0; c < 5; ++c)
            if (mask & (1u << c))
                s += (s.empty() ? "" : ",") + (c == 0 ? std::string("?") : std::to_string(c - 1));
        return s.empty() ? std::string("none") : s;
    };
    const char* verdict =
        g_pcHits == 0 ? "NO per-view CBV was seen in any table -- a blind spot until proven otherwise: check the CBV "
                        "create/recorded counts and the empty/unresolved slots above before believing it"
        : (gbufCopies & otherCopies) == 0 && gbufCopies != 0
            ? "SEPARABLE BY COPY: the G-buffer and the other passes read different copies of the frame's struct -- "
              "patch only the G-buffer's copy (the copy mask already does this)"
        : shared == 0 && gbufOnly > 0
            ? "SEPARABLE BY DESCRIPTOR: the G-buffer's per-view CBVs are descriptors no other pass binds -- a private "
              "copy can be reached by retargeting only those"
        : gbufOnly == 0 && shared == 0 ? "NO G-buffer pass bound a per-view CBV in the window -- was the view in gameplay?"
                                       : "SHARED: the G-buffer and other passes bind the SAME per-view descriptors -- "
                                         "a private copy needs the G-buffer's table rewritten per bind";
    LOG_INFO("DuniaTaa: private-copy probe: VERDICT -- {} (descriptors: {} G-buffer only, {} other only, {} shared; "
             "copies read by the G-buffer: {}, by other passes: {})",
             verdict, gbufOnly, otherOnly, shared, copyList(gbufCopies), copyList(otherCopies));

    // ---- who reads which copy: the passes, by render-target signature ----
    {
        std::vector<std::pair<PcPassKey, uint64_t>> passes(g_pcPasses.begin(), g_pcPasses.end());
        std::sort(passes.begin(), passes.end(), [](const auto& a, const auto& b)
                  {
                      if (a.first[3] != b.first[3]) // the first copy (the one that lit the NPCs) first
                          return a.first[3] == 1;
                      return a.second > b.second;
                  });
        const uint64_t presents = std::max<uint64_t>(1, now - g_pcFrom);
        LOG_INFO("DuniaTaa: pass map: {} distinct (pass, root, slot, copy) keys; the FIRST copy's readers first. "
                 "Formats are DXGI numbers (10 R16G16B16A16_FLOAT, 24 R10G10B10A2_UNORM, 26 R11G11B10_FLOAT, 28 "
                 "R8G8B8A8_UNORM, 34/35 R16G16_FLOAT/UNORM, 40 D32_FLOAT, 20 D32_FLOAT_S8X24, 45 D24_UNORM_S8)",
                 passes.size());
        int shown = 0;
        for (const auto& [k, n] : passes)
        {
            if (++shown > 48)
                break;
            std::string rts;
            for (int f = 0; f < k[4] && f < 8; ++f)
                rts += (rts.empty() ? "" : ",") + std::to_string(k[6 + f]);
            LOG_INFO("DuniaTaa: pass map: copy {} -- {} root {} slot {}{}: {} binds ({:.1f}/Present)",
                     k[3] == 0 ? std::string("?") : std::to_string(k[3]), k[0] ? "GRAPHICS" : "COMPUTE", k[1], k[2],
                     k[0] ? " targets [" + rts + "] depth " + std::to_string(k[5]) : std::string(), n,
                     (double) n / (double) presents);
        }
    }

    // ---- velocity readers: who reads the G-buffer's R8G8 velocity, before or after the motion pass ----
    {
        // Each event's place in submission order: the list's first submit at or after the Present it was recorded
        // in, then its order within that list. A list never seen submitted stays unresolved.
        std::unordered_map<ID3D12GraphicsCommandList*, std::vector<std::pair<uint64_t, uint32_t>>> subs;
        for (const VelSubmit& s : g_velSubmits)
            subs[s.list].push_back({ s.present, s.ordinal });
        auto resolve = [&](const VelEvent& e, uint64_t* present, uint32_t* ordinal)
        {
            const auto it = subs.find(e.list);
            if (it == subs.end())
                return false;
            for (const auto& [p, o] : it->second)
                if (p >= e.present)
                {
                    *present = p;
                    *ordinal = o;
                    return true;
                }
            return false;
        };
        struct Order
        {
            uint32_t ordinal, seq;
            ID3D12GraphicsCommandList* list;
        };
        std::unordered_map<uint64_t, Order> motionAt; // submit Present -> the motion pass (first one)
        uint64_t motionEvents = 0, gbufBefore = 0, gbufAfter = 0, gbufUnresolved = 0;
        for (const VelEvent& e : g_velEvents)
        {
            if (e.kind != 1)
                continue;
            motionEvents++;
            uint64_t p;
            uint32_t o;
            if (resolve(e, &p, &o) && motionAt.find(p) == motionAt.end())
                motionAt[p] = { o, e.seq, e.list };
        }
        auto relation = [&](const VelEvent& e) -> int // 0 unresolved, 1 before, 2 after, 3 same list before, 4 same after
        {
            uint64_t p;
            uint32_t o;
            if (!resolve(e, &p, &o))
                return 0;
            const auto m = motionAt.find(p);
            if (m == motionAt.end())
                return 0;
            if (e.list == m->second.list)
                return e.seq < m->second.seq ? 3 : 4;
            return o < m->second.ordinal ? 1 : 2;
        };
        std::vector<std::array<uint64_t, 5>> rel(g_velReaderKeys.size());
        for (const VelEvent& e : g_velEvents)
        {
            if (e.kind == 0 && e.reader < rel.size())
                rel[e.reader][relation(e)]++;
            else if (e.kind == 2)
            {
                const int r = relation(e);
                (r == 1 || r == 3 ? gbufBefore : r == 0 ? gbufUnresolved : gbufAfter)++;
            }
        }
        ID3D12Resource* const velTex = g_velTex.load();
        LOG_INFO("DuniaTaa: velocity readers: target {:X}; {} events, {} submits recorded; motion pass dispatches {} "
                 "(in {} frames resolved); CONTROL -- G-buffer binds before the motion pass {}, after {}, unresolved {} "
                 "(must be almost all before)",
                 (size_t) velTex, g_velEvents.size(), g_velSubmits.size(), motionEvents, motionAt.size(), gbufBefore,
                 gbufAfter, gbufUnresolved);
        std::vector<uint32_t> order(g_velReaderKeys.size());
        for (uint32_t k = 0; k < order.size(); ++k)
            order[k] = k;
        std::sort(order.begin(), order.end(),
                  [](uint32_t a, uint32_t b) { return g_velReaderBinds[a] > g_velReaderBinds[b]; });
        const uint64_t presents = std::max<uint64_t>(1, now - g_pcFrom);
        int shown = 0;
        for (const uint32_t k : order)
        {
            if (++shown > 24)
                break;
            const VelKey& vk = g_velReaderKeys[k];
            const bool gfx = (vk[0] >> 48) != 0;
            const bool isMotion = !gfx && vk[1] != 0 && vk[1] == (uint64_t) g_mvPipeline;
            std::string rts;
            for (int f = 0; f < (int) (vk[0] & 0xFFFF) && f < 4; ++f)
                rts += (rts.empty() ? "" : ",") + std::to_string((vk[2] >> (16 * f)) & 0xFFFF);
            LOG_INFO("DuniaTaa: velocity readers: {}{} root {} slot {} pipeline {:X}{}: {} binds ({:.1f}/Present) -- vs "
                     "the motion pass: BEFORE {} (same list {}), AFTER {} (same list {}), unresolved {}",
                     gfx ? "GRAPHICS" : "COMPUTE", isMotion ? " [THE MOTION PASS -- the control]" : "",
                     (vk[0] >> 32) & 0xFFFF, (vk[0] >> 16) & 0xFFFF, vk[1],
                     gfx ? " targets [" + rts + "]" : std::string(), g_velReaderBinds[k],
                     (double) g_velReaderBinds[k] / (double) presents, rel[k][1] + rel[k][3], rel[k][3],
                     rel[k][2] + rel[k][4], rel[k][4], rel[k][0]);
        }
        // Barrier facts: per texture, the transitions in submission order around the motion pass.
        auto stateName = [](uint32_t s)
        {
            if (s == 0)
                return std::string("COMMON");
            std::string n;
            const std::pair<uint32_t, const char*> bits[] = {
                { D3D12_RESOURCE_STATE_RENDER_TARGET, "RT" }, { D3D12_RESOURCE_STATE_UNORDERED_ACCESS, "UAV" },
                { D3D12_RESOURCE_STATE_DEPTH_WRITE, "DEPTH_WRITE" }, { D3D12_RESOURCE_STATE_DEPTH_READ, "DEPTH_READ" },
                { D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, "NPSR" },
                { D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, "PSR" }, { D3D12_RESOURCE_STATE_COPY_DEST, "COPY_DEST" },
                { D3D12_RESOURCE_STATE_COPY_SOURCE, "COPY_SOURCE" } };
            for (const auto& [bit, name] : bits)
                if ((s & bit) == bit)
                    n += (n.empty() ? "" : "|") + std::string(name);
            return n.empty() ? std::to_string(s) : n;
        };
        for (int tex = 0; tex < 2; ++tex)
        {
            const uint16_t kind = tex == 0 ? 3 : 4;
            std::map<std::tuple<int, uint32_t, uint32_t>, uint64_t> transitions; // (relation, before, after)
            // the LAST barrier before each frame's motion pass: its after-state is the state AT the motion pass
            std::unordered_map<uint64_t, std::pair<std::pair<uint32_t, uint32_t>, uint32_t>> lastBefore; // present -> ((ordinal, seq), after)
            uint64_t total = 0;
            for (const VelEvent& e : g_velEvents)
            {
                if (e.kind != kind)
                    continue;
                total++;
                const int r = relation(e);
                transitions[{ r, e.before, e.after }]++;
                uint64_t p;
                uint32_t o;
                if ((r == 1 || r == 3) && resolve(e, &p, &o))
                {
                    const auto key = std::make_pair(o, e.seq);
                    auto& slot = lastBefore[p];
                    if (slot.first <= key)
                        slot = { key, e.after };
                }
            }
            std::map<uint32_t, uint64_t> stateAtMotion;
            for (const auto& [p, v] : lastBefore)
                stateAtMotion[v.second]++;
            std::string atMotion;
            for (const auto& [s, n] : stateAtMotion)
                atMotion += (atMotion.empty() ? "" : ", ") + stateName(s) + " x" + std::to_string(n);
            LOG_INFO("DuniaTaa: conversion probe: {} -- {} transitions in the window; its state AT the motion pass "
                     "(the last transition before it, per frame): {} (frames with none before: {})",
                     tex == 0 ? "VELOCITY (t3)" : "DEPTH (t1)", total, atMotion.empty() ? std::string("none") : atMotion,
                     motionAt.size() - lastBefore.size());
            int shown = 0;
            for (const auto& [k, n] : transitions)
            {
                if (++shown > 12)
                    break;
                const char* rel[] = { "unresolved", "before (other list)", "AFTER (other list)", "before (same list)",
                                      "AFTER (same list)" };
                LOG_INFO("DuniaTaa: conversion probe:   {} {} -> {} x{} ({})", tex == 0 ? "t3" : "t1",
                         stateName(std::get<1>(k)), stateName(std::get<2>(k)), n, rel[std::get<0>(k)]);
            }
        }
        LOG_INFO("DuniaTaa: conversion probe: RESTORABILITY at {} motion pass dispatches -- complete {}; missing: root "
                 "signature {}, compute signature {}, root parameters {}, unknown parameters {}, heaps {}, pipeline {}",
                 g_rsDispatches, g_rsComplete, g_rsNoSig, g_rsNotCompute, g_rsNoParams, g_rsInvalidParams, g_rsNoHeaps,
                 g_rsNoPipeline);

        if (g_velReaderKeys.empty())
            LOG_INFO("DuniaTaa: velocity readers: NONE seen -- {}", velTex == nullptr
                                                                        ? "the G-buffer's R8G8 target was never identified"
                                                                        : "not even the motion pass: the walk is blind to "
                                                                          "this texture's descriptors");
    }

    // ---- table rewrite: can the G-buffer's per-view binds be given a table of our own? ----
    size_t sigCount = 0;
    {
        std::lock_guard<std::mutex> lock(g_rsLock);
        sigCount = g_rootSigs.size();
    }
    LOG_INFO("DuniaTaa: table rewrite: G-buffer binds holding a per-view CBV {}; skipped: list's root signature not "
             "seen {}, root signature unknown {}, root parameter not a table {}; root signatures recorded since "
             "startup {}",
             g_rwBinds, g_rwNoList, g_rwUnknownSig, g_rwNotTable, sigCount);
    bool allRebuildable = !g_rwLayouts.empty();
    std::vector<ID3D12DescriptorHeap*> tableHeaps;
    for (const auto& [key, L] : g_rwLayouts)
    {
        LOG_INFO("DuniaTaa: table rewrite: root signature {:X} param {}: {} range(s) [{}], table size {}, per-view CBV "
                 "at slot {}; binds {}, rebuildable {}, non-CBV ranges {}, unrecorded CBV slots {}; heap {:X} ({} "
                 "descriptors)",
                 (size_t) key.first, key.second, L.rangeCount, L.ranges, L.unbounded ? std::string("UNBOUNDED")
                                                                                      : std::to_string(L.tableSize),
                 L.viewSlot, L.binds, L.reconstructable, L.nonCbv, L.unrecorded, (size_t) L.heap, L.heapNum);
        if (L.reconstructable != L.binds)
            allRebuildable = false;
        if (std::find(tableHeaps.begin(), tableHeaps.end(), L.heap) == tableHeaps.end())
            tableHeaps.push_back(L.heap);
    }
    // Where our tables could live: slots of the SAME heap (only one CBV/SRV/UAV heap can be bound) that the game
    // has never written since startup. The map shows the heap in 64 parts: '.' never written, '#' written -- a
    // ring that wraps the whole heap shows no dots anywhere. Writes are counted in the probe window only.
    uint64_t minFreeTop = UINT64_MAX;
    const int n = g_svCount.load(std::memory_order_acquire);
    for (int h = 0; h < n; ++h)
    {
        const SvHeap& s = g_svHeaps[h];
        std::string map;
        for (int b = 0; b < kSvBuckets; ++b)
            map += s.bucket[b].load(std::memory_order_relaxed) == 0 ? '.' : '#';
        const int64_t high = s.high.load(std::memory_order_relaxed);
        const uint64_t freeTop = (uint64_t) ((int64_t) s.num - 1 - high);
        const bool usedByGbuffer = std::find(tableHeaps.begin(), tableHeaps.end(), s.heap) != tableHeaps.end();
        if (usedByGbuffer)
            minFreeTop = std::min(minFreeTop, freeTop);
        LOG_INFO("DuniaTaa: table rewrite: shader-visible heap {:X}{}: {} descriptors, highest slot written since "
                 "startup {}, never-written top {}, slots written in the probe window {} [{}]",
                 (size_t) s.heap, usedByGbuffer ? " (holds the G-buffer's tables)" : "", s.num, high, freeTop,
                 s.writes.load(std::memory_order_relaxed), map);
    }
    if (g_svOverflow.load() > 0)
        LOG_INFO("DuniaTaa: table rewrite: {} shader-visible heap(s) beyond the {} tracked were not watched",
                 g_svOverflow.load(), kSvHeapsMax);
    const char* rwVerdict =
        g_rwBinds == 0              ? "NO G-buffer per-view bind was analysed -- see the skip counts above"
        : !allRebuildable           ? "NOT REBUILDABLE as CBVs alone: a range holds SRVs/UAVs/samplers or slots we never "
                                      "recorded -- would need the game's CPU source descriptors (copy tracking)"
        : minFreeTop == UINT64_MAX  ? "REBUILDABLE, but the G-buffer's heap was not among the watched shader-visible heaps"
        : minFreeTop < 4096         ? "REBUILDABLE, but the G-buffer's heap has under 4096 never-written slots at the top "
                                      "-- no safe place for our tables"
                                    : "FEASIBLE: every G-buffer per-view table is CBVs we recorded, and its heap has a "
                                      "never-written region at the top to hold our copies";
    LOG_INFO("DuniaTaa: table rewrite: VERDICT -- {} (smallest never-written top of a G-buffer heap: {})", rwVerdict,
             minFreeTop == UINT64_MAX ? std::string("n/a") : std::to_string(minFreeTop));
}

void DrivePrivateCopyProbe()
{
    const uint64_t now = g_presentIndex.load();
    const int state = g_pcState.load(std::memory_order_relaxed);
    if (state == 0)
    {
        if (!g_pcArm.exchange(false, std::memory_order_relaxed))
            return;
        const bool splicedNow = g_resolveSplicedPresent != UINT64_MAX && now - g_resolveSplicedPresent <= 1;
        if (!splicedNow || g_probeOn.load(std::memory_order_relaxed))
        {
            LOG_WARN("DuniaTaa: private-copy probe: not started -- {}", splicedNow ? "the writer probe is running"
                                                                                 : "needs gameplay with DLSS on");
            return;
        }
        {
            std::lock_guard<std::mutex> lock(g_pcLock);
            g_pcDescs.clear();
            g_pcTables.clear();
            std::memset(g_pcByCopy, 0, sizeof(g_pcByCopy));
            std::memset(g_pcWalked, 0, sizeof(g_pcWalked));
            g_pcCbvSlots = g_pcHits = g_pcEmpty = g_pcNoHeap = 0;
            std::memset(g_pcCbvByClass, 0, sizeof(g_pcCbvByClass));
            std::memset(g_pcKb, 0, sizeof(g_pcKb));
            std::memset(g_pcNear, 0, sizeof(g_pcNear));
            g_pcGbufCbvs.clear();
            g_rwLayouts.clear();
            g_rwBinds = g_rwNoList = g_rwUnknownSig = g_rwNotTable = 0;
            g_pcPasses.clear();
            g_velEvents.clear();
            g_velSubmits.clear();
            g_mvInputsSeen.clear();
            g_velTex.store(nullptr, std::memory_order_relaxed);
            g_velDepth.store(nullptr, std::memory_order_relaxed);
            g_rsDispatches = g_rsComplete = g_rsNoSig = g_rsNotCompute = g_rsNoParams = g_rsInvalidParams =
                g_rsNoHeaps = g_rsNoPipeline = 0;
            g_rsLogged = 0;
            g_velSeq.clear();
            g_velReaderIds.clear();
            g_velReaderKeys.clear();
            g_velReaderBinds.clear();
            g_velSubmitPresent = UINT64_MAX;
            g_velSubmitOrdinal = 0;
        }
        {
            std::lock_guard<std::mutex> lock(g_rsLock);
            g_listRootSig.clear();
        }
        for (int h = 0; h < g_svCount.load(std::memory_order_acquire); ++h)
            g_svHeaps[h].writes.store(0, std::memory_order_relaxed);
        g_pcCbvAtStart = ResTrack_Dx12::CbvCreateCount();
        g_pcRecordedAtStart = ResTrack_Dx12::CbvRecordedCount();
        g_pcSpliced = 0;
        g_pcFrom = now + kPcWarmup;
        g_pcUntil = g_pcFrom + kPcWindow;
        ResTrack_Dx12::SetMirrorAllHeaps(true);
        g_pcState.store(1, std::memory_order_relaxed);
        LOG_INFO("DuniaTaa: private-copy probe: ON -- heap mirroring and CBV recording on; counting Presents {}..{}",
                 g_pcFrom, g_pcUntil);
    }

    // The struct copies move around a ring: refreshed every Present, as GPU addresses a CBV would hold.
    {
        const uint8_t* bases[kVcMaxCandidates] {};
        int copies[kVcMaxCandidates] {};
        const int n = g_viewConstants.SnapshotCopies(bases, copies, (int) kVcMaxCandidates);
        std::lock_guard<std::mutex> lock(g_pcLock);
        g_pcRangeCount = 0;
        for (int i = 0; i < n; ++i)
        {
            uint64_t va = 0;
            if (ConstantCapture_Dx12::GpuAddressOf(bases[i], &va))
                g_pcRanges[g_pcRangeCount++] = { va, copies[i] };
        }
    }

    const int nowState = g_pcState.load(std::memory_order_relaxed);
    if (nowState == 1 && now >= g_pcFrom)
    {
        g_pcState.store(2, std::memory_order_relaxed);
        LOG_INFO("DuniaTaa: private-copy probe: counting ({} per-view copy addresses known, {} CBVs recorded so far)",
                 g_pcRangeCount, ResTrack_Dx12::CbvRecordedCount() - g_pcRecordedAtStart);
    }
    else if (nowState == 2)
    {
        if (g_resolveSplicedPresent == now)
            g_pcSpliced++;
        if (now >= g_pcUntil)
        {
            g_pcState.store(0, std::memory_order_relaxed);
            ResTrack_Dx12::SetMirrorAllHeaps(false);
            PcSummary(now);
            std::lock_guard<std::mutex> lock(g_rsLock);
            g_listRootSig.clear();
        }
    }
}
} // namespace

void DuniaTaa_Dx12::NoteRootSignature(ID3D12RootSignature* rootSignature,
                                      const D3D12_VERSIONED_ROOT_SIGNATURE_DESC* desc)
{
    if (rootSignature == nullptr || desc == nullptr || !Active())
        return;
    std::vector<RsParam> params;
    if (desc->Version == D3D_ROOT_SIGNATURE_VERSION_1_0)
        params = ParseRootParams<D3D12_ROOT_PARAMETER, D3D12_DESCRIPTOR_RANGE>(desc->Desc_1_0.pParameters,
                                                                               desc->Desc_1_0.NumParameters);
    else if (desc->Version == D3D_ROOT_SIGNATURE_VERSION_1_1)
        params = ParseRootParams<D3D12_ROOT_PARAMETER1, D3D12_DESCRIPTOR_RANGE1>(desc->Desc_1_1.pParameters,
                                                                                 desc->Desc_1_1.NumParameters);
    else if (desc->Version == D3D_ROOT_SIGNATURE_VERSION_1_2)
        params = ParseRootParams<D3D12_ROOT_PARAMETER1, D3D12_DESCRIPTOR_RANGE1>(desc->Desc_1_2.pParameters,
                                                                                 desc->Desc_1_2.NumParameters);
    else
        return;
    std::lock_guard<std::mutex> lock(g_rsLock);
    if (params.size() > 1 && params[1].type == D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE &&
        params[1].ranges.size() == 1 && params[1].ranges[0].type == D3D12_DESCRIPTOR_RANGE_TYPE_CBV &&
        params[1].ranges[0].offset == 0 && params[1].ranges[0].count >= 1 && params[1].ranges[0].count <= kRwStride)
    {
        const int n = g_rwSigCount.load(std::memory_order_relaxed);
        if (n < kRwSigsMax)
        {
            g_rwSigs[n].store(rootSignature, std::memory_order_relaxed);
            g_rwSigSize[n].store(params[1].ranges[0].count, std::memory_order_relaxed);
            g_rwSigCount.store(n + 1, std::memory_order_release);
            LOG_INFO("DuniaTaa: private copy: root signature {:X} qualifies -- parameter 1 is one table of {} CBVs",
                     (size_t) rootSignature, params[1].ranges[0].count);
        }
    }
    if (g_rootSigs.size() < kRootSigsMax || g_rootSigs.count(rootSignature) != 0)
        g_rootSigs[rootSignature] = std::move(params);
}

void DuniaTaa_Dx12::NoteGraphicsRootSignature(ID3D12GraphicsCommandList* cmd, ID3D12RootSignature* rootSignature)
{
    // Only while the probe runs: everyday play pays one relaxed load. A list re-sets its root signature after
    // every Reset, so the probe's 60-Present warm-up is enough to know every list's.
    // Always, and free: which root signature this thread's list has, for the private copy's bind check. A thread
    // records one list at a time; a bind whose list is not this thread's last is simply left alone.
    if (cmd != nullptr && g_pvMirrorOn.load(std::memory_order_relaxed))
    {
        const UINT size = RwTableSize(rootSignature);
        RsShard& shard = RsShardOf(cmd);
        std::lock_guard<std::mutex> lock(shard.m);
        shard.size[cmd] = size;
    }
    if (g_pcState.load(std::memory_order_relaxed) == 0 || cmd == nullptr)
        return;
    std::lock_guard<std::mutex> lock(g_rsLock);
    g_listRootSig[cmd] = rootSignature;
}

void DuniaTaa_Dx12::NoteShaderVisibleHeap(ID3D12DescriptorHeap* heap, SIZE_T cpuStart, SIZE_T gpuStart,
                                          UINT numDescriptors, UINT increment)
{
    if (heap == nullptr || numDescriptors == 0 || increment == 0 || !Active())
        return;
    std::lock_guard<std::mutex> lock(g_svLock);
    const int n = g_svCount.load(std::memory_order_relaxed);
    if (n >= kSvHeapsMax)
    {
        g_svOverflow.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    SvHeap& s = g_svHeaps[n];
    s.heap = heap;
    s.cpuStart = cpuStart;
    s.cpuEnd = cpuStart + (SIZE_T) numDescriptors * increment;
    s.num = numDescriptors;
    s.inc = increment;
    s.gpuStart = gpuStart;
    if (numDescriptors >= kRwBigHeap)
    {
        s.srcMap = new std::atomic<SIZE_T>[numDescriptors](); // lives as long as the process; ~3.6 MB a heap
        s.regionStart = numDescriptors - kRwRegionSlots;
    }
    g_svCount.store(n + 1, std::memory_order_release);
    LOG_INFO("DuniaTaa: shader-visible CBV/SRV/UAV heap {:X}: {} descriptors -- writes into it are counted from now",
             (size_t) heap, numDescriptors);
}

void DuniaTaa_Dx12::NoteDescriptorWrite(SIZE_T cpuHandle, UINT count, SIZE_T srcHandle)
{
    const int n = g_svCount.load(std::memory_order_acquire);
    for (int h = 0; h < n; ++h)
    {
        SvHeap& s = g_svHeaps[h];
        if (cpuHandle < s.cpuStart || cpuHandle >= s.cpuEnd)
            continue;
        // Every descriptor copy the game makes comes through here, from all its recording threads, so this stays
        // READ-MOSTLY: the high-water mark is only exchanged when it rises, a bucket is a touched-once flag, and
        // write counts are kept only while the probe runs. No contended read-modify-write in everyday play.
        const int64_t first = (int64_t) ((cpuHandle - s.cpuStart) / s.inc);
        const int64_t last = std::min<int64_t>(first + (int64_t) std::max(count, 1u) - 1, (int64_t) s.num - 1);
        int64_t seen = s.high.load(std::memory_order_relaxed);
        while (last > seen && !s.high.compare_exchange_weak(seen, last, std::memory_order_relaxed))
        {
        }
        const int b0 = (int) (first * kSvBuckets / s.num), b1 = (int) (last * kSvBuckets / s.num);
        for (int b = b0; b <= b1; ++b)
            if (s.bucket[b].load(std::memory_order_relaxed) == 0)
                s.bucket[b].store(1, std::memory_order_relaxed);
        if (g_pcState.load(std::memory_order_relaxed) != 0)
            s.writes.fetch_add((uint64_t) (last - first + 1), std::memory_order_relaxed);
        if (s.srcMap != nullptr)
        {
            for (int64_t k = first; k <= last; ++k)
                s.srcMap[k].store(srcHandle != 0 ? srcHandle + (SIZE_T) (k - first) * s.inc : 0,
                                  std::memory_order_relaxed);
            if (!t_ownWrite && last >= (int64_t) s.regionStart && !s.regionBad.exchange(true))
                LOG_WARN("DuniaTaa: private copy: the GAME wrote slot {} of heap {:X}, inside the region the private "
                         "copy uses (from {}) -- the redirect is OFF for this heap from now on",
                         last, (size_t) s.heap, s.regionStart);
        }
        return;
    }
}

// The private copy, at a graphics table bind on the game's recording thread. Everything cheap is checked first:
// root parameter 1, the mode on, this thread's root signature one of the 15-CBV layouts, the list drawing the
// G-buffer. Past those it is a real per-view bind of the G-buffer, and every way out is counted and named once.
// The rebuilt table is the game's own 15 descriptors, copied from the CPU descriptors the game itself copied them
// from, with slot 0 re-created to point at our mirror of the same struct. It lives in the top of the SAME heap
// (only one CBV/SRV/UAV heap can be bound), in a region the game has never written.
D3D12_GPU_DESCRIPTOR_HANDLE DuniaTaa_Dx12::RedirectGraphicsTable(ID3D12GraphicsCommandList* cmd, UINT root,
                                                                 D3D12_GPU_DESCRIPTOR_HANDLE base)
{
    if (root != 1 || base.ptr == 0 || cmd == nullptr || !g_pvMirrorOn.load(std::memory_order_relaxed) ||
        cmd == g_cmd || g_inEvaluate > 0)
        return base;
    UINT n = 0;
    {
        RsShard& shard = RsShardOf(cmd);
        std::lock_guard<std::mutex> lock(shard.m);
        const auto it = shard.size.find(cmd);
        if (it == shard.size.end())
        {
            g_rwListUnseen.fetch_add(1, std::memory_order_relaxed);
            return base;
        }
        n = it->second;
    }
    if (n == 0)
    {
        g_rwNotOurs.fetch_add(1, std::memory_order_relaxed);
        return base;
    }
    {
        const int scope = g_pvMirrorScope.load(std::memory_order_relaxed);
        std::lock_guard<std::mutex> lock(g_lock);
        const auto it = g_cmdTargets.find(cmd);
        const bool slot0 = it != g_cmdTargets.end() && it->second.gbuffer;
        const bool any = it != g_cmdTargets.end() && it->second.gbufferAny;
        const bool mrt = it != g_cmdTargets.end() && it->second.gbufferMrt;
        g_rwScopeSeen[slot0 ? 0 : any ? 1 : 2].fetch_add(1, std::memory_order_relaxed);
        if (mrt)
            g_rwMrtSeen.fetch_add(1, std::memory_order_relaxed);
        const bool wanted = scope == 0 ? slot0 : scope == 1 ? (slot0 || any) : scope == 3 ? mrt : true;
        if (!wanted)
        {
            g_rwNotGbuffer.fetch_add(1, std::memory_order_relaxed);
            return base;
        }
    }
    if (g_rwCandidates.fetch_add(1, std::memory_order_relaxed) == 0)
        LOG_INFO("DuniaTaa: private copy: first root-1 bind of a qualifying table on a G-buffer list ({} CBVs)", n);

    SvHeap* s = nullptr;
    const int heaps = g_svCount.load(std::memory_order_acquire);
    for (int h = 0; h < heaps && s == nullptr; ++h)
    {
        SvHeap& c = g_svHeaps[h];
        if (base.ptr >= c.gpuStart && base.ptr < c.gpuStart + (SIZE_T) c.num * c.inc)
            s = &c;
    }
    if (s == nullptr || s->srcMap == nullptr)
    {
        CountRwSkip(RwSkip::NoHeap);
        return base;
    }
    // The in-place scope builds no table of ours, so the region and its guard are not its business. One build ran a
    // whole session with the guard tripped at load -- the game DID write slot 452096, the
    // start of the 4096-slot region -- and skipped every in-place bind as "region written": the box did nothing,
    // and the steady AO seen that session measured nothing. In place it needs only slot 0's source.
    const bool inPlace = g_pvMirrorScope.load(std::memory_order_relaxed) == 3;
    if (!inPlace && s->regionBad.load(std::memory_order_relaxed))
    {
        CountRwSkip(RwSkip::RegionWritten);
        return base;
    }
    const UINT idx = (UINT) ((base.ptr - s->gpuStart) / s->inc);
    if (idx + n > s->num || (!inPlace && idx + n > s->regionStart))
    {
        CountRwSkip(RwSkip::OutOfHeap);
        return base;
    }
    SIZE_T src[kRwStride] {};
    for (UINT i = 0; i < (inPlace ? 1u : n); ++i)
    {
        src[i] = s->srcMap[idx + i].load(std::memory_order_relaxed);
        if (src[i] == 0)
        {
            CountRwSkip(RwSkip::UnknownSource);
            return base;
        }
    }
    uint64_t viewAddress = 0, viewSize = 0;
    if (!ResTrack_Dx12::CbvAtCpuHandle(src[0], &viewAddress, &viewSize) || viewSize == 0)
    {
        CountRwSkip(RwSkip::Slot0NotCbv);
        return base;
    }
    uint64_t mirrorAddress = 0;
    if (!g_pvMirror.Redirect(viewAddress, viewSize, &mirrorAddress))
    {
        CountRwSkip(RwSkip::NotACopy);
        return base;
    }

    // Scope 3: no table of ours. Slot 0 of the game's own table -- a per-draw table Dunia writes fresh each frame
    // for this draw -- is re-created to point at the mirror, and the game's handle is bound unchanged. One CBV
    // write a bind; no region, no budget. If some other pass binds the same table, it sees the same-eye copy too:
    // that would show as the AO shimmer this is meant to avoid.
    if (g_pvMirrorScope.load(std::memory_order_relaxed) == 3)
    {
        D3D12_CONSTANT_BUFFER_VIEW_DESC view {};
        view.BufferLocation = mirrorAddress;
        view.SizeInBytes = (UINT) viewSize;
        t_ownWrite = true;
        g_device->CreateConstantBufferView(&view, D3D12_CPU_DESCRIPTOR_HANDLE { s->cpuStart + (SIZE_T) idx * s->inc });
        t_ownWrite = false;
        g_rwInPlace.fetch_add(1, std::memory_order_relaxed);
        static std::atomic<bool> loggedInPlace { false };
        if (!loggedInPlace.exchange(true, std::memory_order_relaxed))
            LOG_INFO("DuniaTaa: private copy: FIRST G-buffer (R8G8 target) bind rewritten IN PLACE -- heap slot {} "
                     "(heap {:X}) now reads the mirror at {:X} instead of {:X}",
                     idx, (size_t) s->heap, mirrorAddress, viewAddress);
        return base;
    }

    if (g_rwThisPresent.fetch_add(1, std::memory_order_relaxed) >= kRwBudget)
    {
        CountRwSkip(RwSkip::Budget);
        return base;
    }
    const UINT table = s->rwNext.fetch_add(1, std::memory_order_relaxed) % (kRwRegionSlots / kRwStride);
    const UINT dest = s->regionStart + table * kRwStride;
    const SIZE_T cpu = s->cpuStart + (SIZE_T) dest * s->inc;
    t_ownWrite = true;
    for (UINT i = 0; i < n; ++i)
        g_device->CopyDescriptorsSimple(1, D3D12_CPU_DESCRIPTOR_HANDLE { cpu + (SIZE_T) i * s->inc },
                                        D3D12_CPU_DESCRIPTOR_HANDLE { src[i] },
                                        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    D3D12_CONSTANT_BUFFER_VIEW_DESC cbv {};
    cbv.BufferLocation = mirrorAddress;
    cbv.SizeInBytes = (UINT) viewSize;
    g_device->CreateConstantBufferView(&cbv, D3D12_CPU_DESCRIPTOR_HANDLE { cpu });
    t_ownWrite = false;

    g_rwRedirects.fetch_add(1, std::memory_order_relaxed);
    static std::atomic<bool> loggedFirst { false };
    if (!loggedFirst.exchange(true, std::memory_order_relaxed))
        LOG_INFO("DuniaTaa: private copy: FIRST G-buffer per-view bind redirected -- table of {} CBVs at heap slot {} "
                 "rebuilt at slot {} (heap {:X}); slot 0 was {:X} ({} bytes), now the mirror at {:X}",
                 n, idx, dest, (size_t) s->heap, viewAddress, viewSize, mirrorAddress);
    return D3D12_GPU_DESCRIPTOR_HANDLE { s->gpuStart + (SIZE_T) dest * s->inc };
}

bool DuniaTaa_Dx12::PrivateCopyOn() { return g_pvMirrorOn.load(); }
void DuniaTaa_Dx12::NoteBarrierFacts(ID3D12GraphicsCommandList* cmd, UINT count, const D3D12_RESOURCE_BARRIER* barriers)
{
    if (g_pcState.load(std::memory_order_relaxed) != 2 || cmd == nullptr || barriers == nullptr || cmd == g_cmd ||
        g_inEvaluate > 0)
        return;
    ID3D12Resource* const vel = g_velTex.load(std::memory_order_relaxed);
    ID3D12Resource* const depth = g_velDepth.load(std::memory_order_relaxed);
    if (vel == nullptr && depth == nullptr)
        return;
    for (UINT i = 0; i < count; ++i)
    {
        const D3D12_RESOURCE_BARRIER& b = barriers[i];
        if (b.Type != D3D12_RESOURCE_BARRIER_TYPE_TRANSITION)
            continue;
        const uint16_t kind = b.Transition.pResource == vel ? 3 : b.Transition.pResource == depth ? 4 : 0;
        if (kind == 0)
            continue;
        std::lock_guard<std::mutex> lock(g_pcLock);
        VelNoteLocked(cmd, kind, 0);
        if (!g_velEvents.empty())
        {
            g_velEvents.back().before = (uint32_t) b.Transition.StateBefore;
            g_velEvents.back().after = (uint32_t) b.Transition.StateAfter;
        }
    }
}

bool DuniaTaa_Dx12::WantsAllTargets()
{
    return g_pvMirrorOn.load(std::memory_order_relaxed) || g_pcState.load(std::memory_order_relaxed) != 0;
}
int DuniaTaa_Dx12::PrivateCopyScope() { return g_pvMirrorScope.load(); }
void DuniaTaa_Dx12::SetPrivateCopyScope(int scope)
{
    scope = std::clamp(scope, 0, 3);
    if (g_pvMirrorScope.exchange(scope) == scope)
        return;
    LOG_INFO("DuniaTaa: private copy scope: {}",
             scope == 0   ? "lists with the motion target at slot 0"
             : scope == 1 ? "lists with the motion target at ANY slot (and depth)"
             : scope == 2 ? "EVERY graphics list (compute keeps the game's)"
                          : "the G-buffer (R8G8 target), slot 0 rewritten IN PLACE");
}

// From ResTrack's OMSetRenderTargets, only while the private copy is on: every slot resolved, not slot 0 alone.
void DuniaTaa_Dx12::NoteAllTargets(ID3D12GraphicsCommandList* cmd, ID3D12Resource* const* targets, UINT count,
                                   ID3D12Resource* depth)
{
    if (cmd == nullptr || cmd == g_cmd || g_inEvaluate > 0)
        return;
    bool motion = false;
    uint16_t fmt[8] {};
    const UINT n = std::min<UINT>(count, 8);
    for (UINT i = 0; i < n; ++i)
    {
        if (targets[i] == nullptr)
            continue;
        const DXGI_FORMAT f = targets[i]->GetDesc().Format;
        fmt[i] = (uint16_t) f;
        motion |= IsMotionFormat(f);
    }
    const uint16_t ds = depth != nullptr ? (uint16_t) depth->GetDesc().Format : 0;
    // The velocity target of THE G-buffer: its R8G8_UNORM slot. Described once per new resource (it is bound, so
    // alive, right now); the pointer is kept for identity only.
    bool isGbuffer = false;
    if (n >= 3 && depth != nullptr)
    {
        for (UINT i = 0; i < n; ++i)
        {
            if (fmt[i] != (uint16_t) DXGI_FORMAT_R8G8_UNORM)
                continue;
            isGbuffer = true;
            static std::atomic<ID3D12Resource*> lastDescribed { nullptr }; // identity only; several threads
            if (lastDescribed.exchange(targets[i], std::memory_order_relaxed) != targets[i])
            {
                const D3D12_RESOURCE_DESC d = targets[i]->GetDesc();
                static std::atomic<int> described { 0 };
                if (described.fetch_add(1, std::memory_order_relaxed) < 6)
                    LOG_INFO("DuniaTaa: G-buffer target: its R8G8_UNORM at slot {} is {:X}, {}x{}, flags {:X} "
                             "(unordered access {}, render target {})",
                             i, (size_t) targets[i], d.Width, d.Height, (uint32_t) d.Flags,
                             (d.Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS) ? "ALLOWED" : "not allowed",
                             (d.Flags & D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET) ? "allowed" : "not allowed");
            }
            break;
        }
    }
    if (isGbuffer && g_pcState.load(std::memory_order_relaxed) == 2)
    {
        std::lock_guard<std::mutex> lock(g_pcLock);
        VelNoteLocked(cmd, 2, 0);
    }
    std::lock_guard<std::mutex> lock(g_lock);
    CmdTargets& t = g_cmdTargets[cmd];
    t.gbufferAny = motion && depth != nullptr;
    bool r8g8 = false;
    for (UINT i = 0; i < n; ++i)
        r8g8 |= fmt[i] == (uint16_t) DXGI_FORMAT_R8G8_UNORM;
    t.gbufferMrt = r8g8 && n >= 3 && depth != nullptr;
    t.rtCount = (uint8_t) n;
    std::memcpy(t.rtFmt, fmt, sizeof(fmt));
    t.dsFmt = ds;
}
void DuniaTaa_Dx12::SetPrivateCopyOn(bool on)
{
    // Switched on: forget every list's recorded root signature first, so only ones set while ON are trusted -- an
    // entry from an earlier recording could name a layout the list no longer has, and a wrong table size is a GPU
    // fault. A list whose signature was set before this is left alone until it sets one again.
    if (on)
    {
        for (RsShard& shard : g_rsShards)
        {
            std::lock_guard<std::mutex> lock(shard.m);
            shard.size.clear();
        }
    }
    if (g_pvMirrorOn.exchange(on) == on)
        return;
    g_pvMirror.ClearReady();
    LOG_INFO("DuniaTaa: private copy (same-eye NPC motion) {} -- {}", on ? "ON" : "off",
             on ? "the G-buffer's per-view tables are rebuilt to read a same-eye mirror of the struct; the game's "
                  "own struct and every other pass are untouched"
                : "the G-buffer reads the game's own tables again");
}

// Once per stats window, from the Present driver: whether the private copy is really running. Counters, not the
// picture: redirects about 3 per Present, and the mirror entries carrying same-eye values.
void DriveMirrorStats(uint64_t presentIndex)
{
    // Every 600 Presents while on, AND when it is switched off: toggled with the overlay open (~52/s), an on
    // window of 10 s never reached 600 and one run printed nothing at all.
    static uint64_t windowStart = 0;
    static bool wasOn = false;
    g_rwThisPresent.store(0, std::memory_order_relaxed); // the per-Present budget
    const bool on = g_pvMirrorOn.load(std::memory_order_relaxed);
    if (!on && !wasOn)
    {
        windowStart = presentIndex;
        return;
    }
    const bool closing = wasOn && !on;
    wasOn = on;
    if (!closing && presentIndex - windowStart < 600)
        return;
    const uint64_t presents = presentIndex - windowStart;
    windowStart = presentIndex;
    uint64_t writes = 0, entries = 0, same = 0, redirSame = 0, redirPlain = 0, redirStale = 0;
    int ready = 0;
    g_pvMirror.Stats(&writes, &entries, &same, &ready, &redirSame, &redirPlain, &redirStale);
    uint64_t skips[(int) RwSkip::Count];
    for (int k = 0; k < (int) RwSkip::Count; ++k)
        skips[k] = g_rwSkips[k].exchange(0, std::memory_order_relaxed);
    LOG_INFO("DuniaTaa: private copy: {} Presents{} (scope {}; binds on the R8G8 G-buffer {}, rewritten in place {}; "
             "qualifying binds on lists with motion at slot 0 {}, "
             "at a later slot {}, neither {}) -- root-1 binds: list's root signature not seen {}, not a "
             "qualifying table {}, outside the scope {}, candidates {}; redirected {}; kept (over budget {}, no heap {}, "
             "region written {}, out of heap {}, unknown source {}, slot 0 not a CBV {}, not a mirrored copy {}); "
             "mirror: {} struct entries filled, {} with same-eye values, {} writes, {} redirectable now; redirected "
             "structs at their next fill: SAME-EYE {}, plain {}, not refilled {}",
             presents, closing ? " (switched off)" : "", g_pvMirrorScope.load(),
             g_rwMrtSeen.exchange(0, std::memory_order_relaxed), g_rwInPlace.exchange(0, std::memory_order_relaxed),
             g_rwScopeSeen[0].exchange(0, std::memory_order_relaxed),
             g_rwScopeSeen[1].exchange(0, std::memory_order_relaxed),
             g_rwScopeSeen[2].exchange(0, std::memory_order_relaxed),
             g_rwListUnseen.exchange(0, std::memory_order_relaxed),
             g_rwNotOurs.exchange(0, std::memory_order_relaxed), g_rwNotGbuffer.exchange(0, std::memory_order_relaxed),
             g_rwCandidates.load(std::memory_order_relaxed), g_rwRedirects.exchange(0, std::memory_order_relaxed),
             skips[0], skips[1], skips[2], skips[3], skips[4], skips[5], skips[6], entries, same, writes, ready, redirSame,
             redirPlain, redirStale);
}

namespace
{

bool NeedsHeapTracking();

// GPU timing's window: every 600 Presents while it is on, the per-eye means of the splices read back, and how many of
// those Presents had the heap mirroring on (ResTrack tracks every descriptor while it is; it has cost up to
// half the frame rate). Also what the overlay shows.
void DriveGpuTiming(ID3D12CommandQueue* queue, bool mirroring)
{
    static unsigned presents = 0;
    if (!GpuTimingOn())
    {
        presents = 0;
        g_mirrorPresents.store(0, std::memory_order_relaxed);
        return;
    }
    if (g_tsFrequency.load(std::memory_order_relaxed) == 0 && queue != nullptr)
    {
        UINT64 frequency = 0;
        if (SUCCEEDED(queue->GetTimestampFrequency(&frequency)))
            g_tsFrequency.store(frequency, std::memory_order_relaxed); // every direct queue of the GPU shares it
    }
    if (mirroring)
        g_mirrorPresents.fetch_add(1, std::memory_order_relaxed);
    if (++presents < 600)
        return;
    std::lock_guard<std::mutex> lock(g_timeLock);
    PassTimes all {};
    for (const auto& p : g_passTimes)
    {
        all.motion += p.motion;
        all.dlss += p.dlss;
        all.copy += p.copy;
        all.n += p.n;
    }
    DuniaTaa_Dx12::GpuTimes g {};
    g.on = true;
    g.samples = all.n;
    g.windowPresents = presents;
    g.mirrorPresents = g_mirrorPresents.load(std::memory_order_relaxed);
    if (all.n > 0)
    {
        g.motionMs = all.motion / all.n;
        g.dlssMs = all.dlss / all.n;
        g.copyMs = all.copy / all.n;
    }
    g_lastGpuTimes = g;
    auto eyeMs = [](const PassTimes& p) { return p.n ? (p.motion + p.dlss + p.copy) / p.n : 0.0; };
    LOG_INFO("DuniaTaa: GPU per splice (mean per eye, {} Presents): motion {:.3f} ms + DLSS {:.3f} ms ({} {}x{} -> "
             "{}x{}) + copy {:.3f} ms = {:.3f} ms; eye 0 {} splices {:.3f} ms, eye 1 {} splices {:.3f} ms; {} incomplete, "
             "{} invalid; heap mirroring on in {} of {} Presents",
             presents, g.motionMs, g.dlssMs, IsUpscale() ? "Upscaling" : "DLAA", g_createdInW, g_createdInH,
             g_createdOutW, g_createdOutH, g.copyMs, g.motionMs + g.dlssMs + g.copyMs, g_passTimes[0].n,
             eyeMs(g_passTimes[0]), g_passTimes[1].n, eyeMs(g_passTimes[1]), g_timeIncomplete, g_timeInvalid,
             g.mirrorPresents, presents);
    g_passTimes[0] = g_passTimes[1] = PassTimes {};
    g_timeIncomplete = g_timeInvalid = 0;
    g_mirrorPresents.store(0, std::memory_order_relaxed);
    presents = 0;
}

bool NeedsHeapTracking()
{
    // Square-eye rules: a held scaled-scene candidate waits for the resolve's table to be seen (see Note).
    if (g_presentIndex.load() < g_structRefreshUntil.load(std::memory_order_relaxed))
        return true;
    if (g_pcState.load(std::memory_order_relaxed) != 0)
        return true;
    // The upscale writer probe walks descriptor tables for its window and no longer.
    if (g_probeOn.load(std::memory_order_relaxed))
        return true;
    if (g_mvSearchOn.load(std::memory_order_relaxed))
        return true;
    // Real motion needs it until it has learned the pipeline; after that it recognises the pass by pointer.
    // Matrix motion never needs it at all.
    if (NeedsGameVectors() && !g_mvPipelineReady.load(std::memory_order_relaxed))
        return true;
    // The TAA search needs it until it has found the resolve, or given up.
    if (DuniaTaa_Dx12::WantsTaaSearch() && !g_taaFound.load(std::memory_order_relaxed) && !g_taaGaveUp)
        return true;
    // The upscale-dispatch search is kept but no longer holds the mirroring on: the upscale is a
    // draw (see OnSetTargets), and the mirroring cost half the frame rate for nothing.
    if (false && SpliceTaaMode() && IsUpscale() && !g_upPipelineReady.load(std::memory_order_relaxed) && !g_upGaveUp)
    {
        const uint64_t now = g_presentIndex.load();
        if (g_upSearchUntil == 0)
            g_upSearchUntil = now + 900;
        if (now < g_upSearchUntil)
            return true;
        g_upGaveUp = true;
        LOG_WARN("DuniaTaa: upscaling: the game's upscale dispatch was not learned in 900 Presents -- the eye "
                 "shows DLSS's output downsampled and FSR1'd back up (still better than TAA). Read the "
                 "'eye-sized entry in a compute table' lines above for what to match instead");
    }
    // And splice mode needs it a little longer still, until the target -> colour pairs have settled.
    // Gated on the resolve having been found, so a search that gives up does not leave it on for ever.
    if (SpliceTaaMode() && g_taaFound.load(std::memory_order_relaxed) &&
        !g_taaPairsDone.load(std::memory_order_relaxed))
        return true;
    return false;
}

// Runs from the first Present and stops as soon as both instances are in hand, so the heap mirroring is
// paid for a few seconds rather than for the session.
// The TAA resolve search: give it a generous window, then say what it nearly matched rather than
// leaving the mirroring on for the session.
void DriveTaaSearch()
{
    if (!DuniaTaa_Dx12::WantsTaaSearch() || g_taaGaveUp || g_taaFound.load(std::memory_order_relaxed))
        return;
    if (SceneW() == 0 || !GameplayNow())
    {
        // Never silent: at one eye size the size learners had the wrong eye, so neither held, the
        // search never started, F6 waited for ever and the heap mirroring ran all session. A resolve of the eye's
        // shape has been drawn, so this is gameplay: say why the search is not running, and after a minute of it give
        // up -- which releases the mirroring -- rather than cost frame rate for the rest of the session.
        const uint64_t first = g_structEyeSceneFirst.load(std::memory_order_relaxed);
        const uint64_t at = g_presentIndex.load();
        if (first != 0 && at > first + 600)
        {
            static bool loggedWhy = false;
            if (!loggedWhy)
            {
                loggedWhy = true;
                LOG_WARN("DuniaTaa: the TAA resolve search has not started 600 Presents into gameplay: scene {}x{}, "
                         "gameplay signature {} (learned eye {}x{}, VR eye {}x{}, structural scene {}x{}). It gives up "
                         "in 3000 more Presents",
                         SceneW(), SceneH(), GameplayNow() ? "seen" : "NOT seen", g_destW, g_destH, g_vrEyeW.load(),
                         g_vrEyeH.load(), g_structSceneW.load(), g_structSceneH.load());
            }
            if (at > first + 3600)
            {
                g_taaGaveUp = true;
                LOG_WARN("DuniaTaa: TAA resolve search given up without starting (see the line above); DLSS cannot "
                         "splice this session, and the heap mirroring stops");
            }
        }
        return;
    }
    const uint64_t now = g_presentIndex.load();
    // Restarted when the scene size changes (a Resolution Scale change mid-session): the shape it matched
    // against was wrong for every Present since.
    if (g_taaSearchUntil != 0 && g_taaSearchSceneW != SceneW())
        g_taaSearchUntil = 0;
    if (g_taaSearchUntil == 0)
    {
        g_taaSearchUntil = now + kMvSearchPresents;
        g_taaSearchSceneW = SceneW();
        g_taaBestMotion.store(0, std::memory_order_relaxed);
        g_taaBestHdr.store(0, std::memory_order_relaxed);
        LOG_INFO("DuniaTaa: looking for the game's TAA resolve -- a graphics table with both half-res "
                 "R8G8_UNORM motion textures and scene-sized RGBA16F colour, against {}x{}{}",
                 SceneW(), SceneH(), SceneScaled() ? " (the render size; the eye is larger)" : "");
        return;
    }
    if (now < g_taaSearchUntil)
        return;
    g_taaGaveUp = true;
    LOG_WARN("DuniaTaa: TAA resolve not found in {} Presents. Closest graphics table had {} half-res "
             "R8G8_UNORM and {} eye-sized RGBA16F of the 2 + 2 wanted -- widen the signature from a "
             "capture's bindings.txt rather than guessing",
             kMvSearchPresents, g_taaBestMotion.load(), g_taaBestHdr.load());
}

// The second half of the TAA search, for splice mode only: the pipeline identifies the draw, but the
// splice also needs to know which colour each render target is paired with. Those are learned at the
// resolve draws themselves; this just decides when they have stopped arriving, which is what releases
// the heap mirroring. Reopened by the splice if a target ever turns up unpaired -- that is a reload.
//
// Deliberately NOT gated on GameplayNow(), unlike the two searches. The rule those follow -- no search
// starts its clock until gameplay is recognised -- exists because they hunt for something that does not
// exist yet during loading, and a wall-clock window expires at 200-1000 Presents/s before the world is
// up. Nothing here hunts: the pairs are learned from the resolve's OWN draws, and the resolve drawing
// *is* gameplay, more directly than the G-buffer proxy the signature uses. Asking for both cost the
// first run its numbers -- the pairs were in hand 55 seconds before this released the mirroring,
// so ~45% of the frame rate was paid for 55 seconds for information already collected, and
// every rate in that log was mirroring-bound rather than a measurement of the splice.
//
// (That GameplayNow() ran 55 s late at all is a separate question, and it matters beyond this file,
// because that signature is the gate for every search. Not answered by that log.)
void DriveTaaPairs()
{
    if (!SpliceTaaMode() || g_taaPairsDone.load(std::memory_order_relaxed))
        return;
    if (!g_taaFound.load(std::memory_order_relaxed))
        return;
    if (!g_taaResolve.Settled(g_presentIndex.load()))
        return;
    g_taaPairsDone.store(true, std::memory_order_relaxed);
    LOG_INFO("DuniaTaa: TAA resolve pairs settled at {} -- the heap mirroring can stop; the splice reads "
             "the colour by the render target bound at each draw from here on",
             g_taaResolve.Size());
}

void DriveMotionSearch()
{
    if (!DuniaTaa_Dx12::WantsMotionSearch() || g_mvSearchReported)
        return;
    // Nothing needs the game's vectors right now, so do not pay the mirroring to go looking for them. This is
    // what made F4 cost a third of the frame rate for as long as the search took -- and the search waits on
    // GameplayNow(), which has been seen to take the best part of a minute.
    //
    // Switching it OFF again matters as much as not starting it. A search begun while passing through the
    // game's-own mode otherwise stays on for ever, because nothing clears g_mvSearchOn -- which is why the
    // matrix mode that needs none of this still ran at 45/s instead of 90.
    if (!NeedsGameVectors() && !(Config::Instance()->DuniaTaaFindMotion.value_or_default() && DiagnosticsOn()))
    {
        if (g_mvSearchOn.exchange(false, std::memory_order_relaxed))
            LOG_INFO("DuniaTaa: motion vector search stopped -- nothing wants the game's vectors any more, so "
                     "the heap mirroring goes with it");
        return;
    }

    // Once the pass is recognised by its pipeline there is nothing left to look for, and the search must not
    // be re-armed by a later mode change: entering the object state did exactly that, kept the heap mirroring
    // on for the whole state (90/s down to 44/s), and had the search resolving descriptor tables through the
    // next menu -- which is where the game then died.
    if (g_mvPipelineReady.load(std::memory_order_acquire))
    {
        if (g_mvSearchOn.exchange(false, std::memory_order_relaxed))
            LOG_INFO("DuniaTaa: motion vector search stopped -- the pass is already recognised by its pipeline, "
                     "so there is nothing to look for and the heap mirroring goes with it");
        return;
    }

    if (!g_mvSearchOn.load(std::memory_order_relaxed))
    {
        // Turned on here rather than when the eye shape arrives: ResTrack only mirrors the descriptor
        // copies it actually sees, so tracking that starts late can miss a heap the game filled once at
        // load and then never resolve anything. Matching waits for the eye shape by itself -- until
        // there is one, nothing is half of anything -- so starting early costs only the mirroring.
        g_mvSearchOn.store(true, std::memory_order_relaxed);
        LOG_INFO("DuniaTaa: looking for the motion vector pass -- R8G8_UNORM + R8_UNORM + R8_UINT bound "
                 "together as UAVs at half the eye size");
    }

    // Nothing counts until gameplay is actually rendering; see GameplayNow.
    if (!GameplayNow())
        return;

    // The deadline only starts once there is an eye shape to match against, and it starts again
    // whenever that shape changes. The pinned size only grows, and menus and loading screens render at
    // sizes of their own, so the first shape the search sees may well not be the gameplay one -- a
    // deadline that ran from it could expire before the world is even up. Anything matched against the
    // old shape was matched against the wrong half, so it goes too.
    if (SceneW() == 0)
        return;
    const uint64_t now = g_presentIndex.load();
    if (g_mvSearchPinnedW != SceneW())
    {
        g_mvSearchPinnedW = SceneW();
        g_mvSearchUntil = now + kMvSearchPresents;
        std::lock_guard<std::mutex> lock(g_mvSearchLock);
        g_mvFound.clear();
        LOG_INFO("DuniaTaa: motion vector search matching against half of {}x{}", SceneW(), SceneH());
    }

    size_t found = 0;
    int probes = 0;
    {
        std::lock_guard<std::mutex> lock(g_mvSearchLock);
        found = g_mvFound.size();
        probes = g_mvProbes;
    }
    // Runs on until the probes are in as well, so the pairing question is answered in the same run.
    if ((found < kMvInstances || probes < kMvProbes) && now < g_mvSearchUntil)
        return;

    g_mvSearchOn.store(false, std::memory_order_relaxed);
    // Not as they were found: narrowing mid-search hides the heaps still to be found. Skipped entirely
    // while the TAA search is still running, since that one reads *graphics* tables and may well bind out
    // of heaps the motion pass never touches.
    if (!DuniaTaa_Dx12::WantsTaaSearch() || g_taaFound.load(std::memory_order_relaxed) || g_taaGaveUp)
    {
        std::lock_guard<std::mutex> lock(g_mvSearchLock);
        for (const auto& range : g_mvHeapRanges)
            ResTrack_Dx12::AddTrackedHeapRange(range.first, range.second);
    }
    g_mvSearchReported = true;
    if (found == 0)
    {
        LOG_WARN("DuniaTaa: motion vector pass not found in {} Presents; heap tracking off again",
                 kMvSearchPresents);
    }
    else
    {
        LOG_INFO("DuniaTaa: motion vector pass found, {} instance(s), {} probes, {} heap(s) of interest; "
                 "heap tracking off again",
                 found, probes, ResTrack_Dx12::TrackedHeapCount());
    }
}
} // namespace

void DuniaTaa_Dx12::NoteComputeTable(UINT rootParameterIndex, D3D12_GPU_DESCRIPTOR_HANDLE base,
                                     ID3D12GraphicsCommandList* cmd)
{
    if (g_probeOn.load(std::memory_order_relaxed))
        ProbeTable(false, rootParameterIndex, base, cmd);
    PcWalk(false, rootParameterIndex, base, cmd);
    // The game's upscale dispatch, learned from its UAV while the mirroring is on. Independent of the
    // motion search below.
    if (base.ptr != 0 && SpliceTaaMode() && IsUpscale() && !g_upPipelineReady.load(std::memory_order_relaxed) &&
        g_createdOutW != 0)
    {
        if (auto heap = ResTrack_Dx12::GetTableHeap(false, (SIZE_T) base.ptr); heap != nullptr && heap->increment != 0)
        {
            for (UINT i = 0; i < 8; ++i)
            {
                const SIZE_T handle = (SIZE_T) base.ptr + (SIZE_T) i * heap->increment;
                if (handle >= heap->gpuEnd)
                    break;
                ResourceInfo info {};
                if (!heap->GetByGpuHandle(handle, info) || info.buffer == nullptr)
                    continue;
                if ((UINT) info.width != g_createdOutW || info.height != g_createdOutH || info.buffer == g_output)
                    continue;
                // Every eye-sized entry seen while learning, a few times: the first build looked only for an
                // RGBA16F UAV in slots 0-3, learned nothing, and left the mirroring on for the session
                // (38-45 fps). The near misses say what the upscale actually binds.
                if (g_upPeeks.load(std::memory_order_relaxed) < 12)
                {
                    const int n = g_upPeeks.fetch_add(1, std::memory_order_relaxed);
                    if (n < 12)
                        LOG_INFO("DuniaTaa: upscaling: eye-sized entry in a compute table while learning -- root {} "
                                 "slot {} {:X} {}x{} fmt {} {}",
                                 rootParameterIndex, i, (size_t) info.buffer, info.width, info.height,
                                 (UINT) info.format, info.type == UAV ? "UAV" : "SRV");
                }
                if (info.type == UAV && IsHdrFormat(info.format))
                {
                    std::lock_guard<std::mutex> pending(g_cmdUpscaleLock);
                    g_cmdUpscale[cmd] = info.buffer;
                    break;
                }
            }
        }
    }

    // The search stops once it has what it needs; real motion needs this to keep resolving, because the
    // texture it copies is whichever one the pass bound this frame.
    const bool searching = g_mvSearchOn.load(std::memory_order_relaxed);
    if (base.ptr == 0 || !(searching || g_realMotionOn.load(std::memory_order_relaxed)))
        return;
    // Once the pipeline is known the descriptor route is finished with, and with it the mirroring.
    if (g_mvPipelineReady.load(std::memory_order_relaxed))
        return;

    // Two filters that cost an integer compare each, ahead of a walk that costs three ResTrack lookups.
    // Without them this ran on every compute root table bind in the frame -- thousands of them -- and the
    // first measured run lost about a third of the frame rate to it (90/s down to 55/s).
    if (!searching)
    {
        // The pass binds its UAV table at one root parameter, learned while searching. Anything else is
        // some other pass.
        if (g_mvRootParameter >= 0 && (int) rootParameterIndex != g_mvRootParameter)
            return;
        // And it runs once a frame: past it, nothing in the rest of the frame needs resolving.
        if (g_mvFoundThisPresent.load(std::memory_order_relaxed) == g_presentIndex.load())
            return;
    }

    // Resolved outside our lock: ResTrack takes its own, and holding both would invite a lock order
    // problem. This is the same route BindingCapture_Dx12 uses.
    auto heap = ResTrack_Dx12::GetTableHeap(false, (SIZE_T) base.ptr);
    if (heap == nullptr || heap->increment == 0)
        return;

    // Exactly three descriptors: the signature is only meaningful as the pass's own u0, u1 and u2, and
    // walking further would drag in whatever sits next in the heap.
    ID3D12Resource* motion = nullptr;
    UINT64 motionW = 0;
    UINT motionH = 0;
    bool edge = false;
    bool stencil = false;
    for (UINT i = 0; i < 3; ++i)
    {
        const SIZE_T handle = (SIZE_T) base.ptr + (SIZE_T) i * heap->increment;
        if (handle >= heap->gpuEnd)
            return;

        ResourceInfo info {};
        if (!heap->GetByGpuHandle(handle, info) || info.buffer == nullptr || info.type != UAV)
            return;
        if (!HalfEyeSized(info.width, info.height))
            return;

        switch (info.format)
        {
        case DXGI_FORMAT_R8G8_UNORM:
            if (motion != nullptr)
                return;
            motion = info.buffer;
            motionW = info.width;
            motionH = info.height;
            break;
        case DXGI_FORMAT_R8_UNORM:
            if (edge)
                return;
            edge = true;
            break;
        case DXGI_FORMAT_R8_UINT:
            if (stencil)
                return;
            stencil = true;
            break;
        default:
            return;
        }
    }
    if (motion == nullptr || !edge || !stencil)
        return;

    g_mvRootParameter = (int) rootParameterIndex;
    g_mvFoundThisPresent.store(g_presentIndex.load(), std::memory_order_relaxed);

    // What the copy after this list's next Dispatch will take. Outside the search's lock and before its
    // early-outs, because this has to keep working long after the search has finished.
    if (g_realMotionOn.load(std::memory_order_relaxed) && cmd != nullptr)
    {
        std::lock_guard<std::mutex> pending(g_cmdMotionLock);
        g_cmdMotion[cmd] = motion;
    }

    std::lock_guard<std::mutex> lock(g_mvSearchLock);

    // Before the dedup: the probe is about every matched bind, not just the first sight of an instance.
    if (g_mvProbes < kMvProbes)
    {
        const uint64_t present = g_presentIndex.load();
        ++g_mvProbes;
        LOG_INFO("DuniaTaa: mv probe {}: present {} parity {}, cmd {:X}, table {:X}, texture {:X}", g_mvProbes,
                 present, (unsigned) (present & 1), (size_t) cmd, (size_t) base.ptr, (size_t) motion);
    }

    // The heaps this pass binds out of, so the mirroring can skip every other heap in the game. Only
    // collected here; applied when the search ends. Applying one immediately narrows the mirroring to it,
    // which stops the other eye's heap being mirrored, which stops its table ever resolving -- the first
    // run of this learned one heap instead of two and copied on every other Present as a result.
    if (g_mvHeapRanges.size() < ResTrack_Dx12::kMaxTrackedHeaps)
    {
        const std::pair<SIZE_T, SIZE_T> range { heap->cpuStart, heap->cpuEnd };
        if (std::find(g_mvHeapRanges.begin(), g_mvHeapRanges.end(), range) == g_mvHeapRanges.end())
            g_mvHeapRanges.push_back(range);
    }

    if (std::find(g_mvFound.begin(), g_mvFound.end(), motion) != g_mvFound.end())
        return;
    if (g_mvFound.size() >= kMvInstances)
        return;
    g_mvFound.push_back(motion);
    // The texture's own size, not half the eye's: a near-miss inside the slack should be visible here
    // rather than hidden behind the number it was matched against.
    LOG_INFO("DuniaTaa: motion vector texture {:X} ({} of {}), {}x{} R8G8_UNORM written by compute, eye {}x{}",
             (size_t) motion, g_mvFound.size(), kMvInstances, motionW, motionH, g_pinnedW, g_pinnedH);
}

int DuniaTaa_Dx12::DlssPreset()
{
    const auto config = Config::Instance();
    return config->RenderPresetOverride.value_or_default() ? (int) config->RenderPresetForAll.value_or_default() : 0;
}

void DuniaTaa_Dx12::RequestDlssPreset(int preset)
{
    if (preset < 0 || preset > 15)
        return;
    auto config = Config::Instance();
    config->RenderPresetOverride = preset != 0;
    config->RenderPresetForAll = (uint32_t) preset;

    // A preset is baked in when the feature is created, and this module never rebuilds a feature itself: a
    // live teardown from Present once crashed the driver. OptiScaler's NGX layer -- which these
    // features were created through -- has its own staged change: the old feature goes to a delayed destroy,
    // the new one is created and initialised on the list being evaluated, over three evaluates. Those
    // evaluates return without writing, so each eye shows its previous output for three of its frames.
    // Only this module's two handles are marked, not every feature in the process.
    auto& state = State::Instance();
    int marked = 0;
    for (NVSDK_NGX_Handle* handle : { g_handle, g_handleEye2 })
    {
        if (handle == nullptr)
            continue;
        state.newBackend = Upscaler::DLSS;
        state.changeBackend[handle->Id] = true;
        ++marked;
    }
    g_needReset = true;
    LOG_INFO("DuniaTaa: DLSS preset {} requested live -- {} feature(s) marked for OptiScaler's staged recreate{}",
             preset == 0 ? std::string("DEFAULT") : std::string(1, (char) ('A' + preset - 1)), marked,
             marked == 0 ? "; none exists yet, so it applies when they are created" : "");
}

int DuniaTaa_Dx12::SameEyeCopies() { return g_pvCopyMask.load(); }
bool DuniaTaa_Dx12::ObjectNearOnly() { return g_objectNearOnly.load(); }
void DuniaTaa_Dx12::SetObjectNearOnly(bool on)
{
    if (g_objectNearOnly.exchange(on) == on)
        return;
    LOG_INFO("DuniaTaa: object motion {} (composition x{:.2f}, floor {:.2f} px, scene {}x{})",
             on ? "only on the NEAREST surface (silhouette noise off)" : "on every judged pixel (old behaviour)",
             g_objectScale.load(), g_objectFloor.load(), SceneW(), SceneH());
}

bool DuniaTaa_Dx12::ObjectBgReject() { return g_objectBgReject.load(); }
void DuniaTaa_Dx12::SetObjectBgReject(bool on)
{
    if (g_objectBgReject.exchange(on) == on)
        return;
    LOG_INFO("DuniaTaa: background vectors on near pixels {} (composition x{:.2f}, floor {:.2f} px, scene {}x{})",
             on ? "REJECTED (silhouettes against the sky off)" : "carried (the older behaviour)",
             g_objectScale.load(), g_objectFloor.load(), SceneW(), SceneH());
}

int DuniaTaa_Dx12::ObjectRadius() { return g_objectRadius.load(); }
void DuniaTaa_Dx12::SetObjectRadius(int px)
{
    px = std::clamp(px, 1, 6);
    if (g_objectRadius.exchange(px) == px)
        return;
    LOG_INFO("DuniaTaa: surface tests search +-{} px (every pixel; nearest surface {}, background rejection {})", px,
             g_objectNearOnly.load() ? "on" : "off", g_objectBgReject.load() ? "on" : "off");
}

bool DuniaTaa_Dx12::ResidualView() { return g_residualView.load(); }
void DuniaTaa_Dx12::SetResidualView(bool on)
{
    if (g_residualView.exchange(on) == on)
        return;
    // DLSS was handed the view's numbers as motion while it was on; its history is not worth keeping.
    if (!on)
        g_needReset = true;
    LOG_INFO("DuniaTaa: residual view {} -- red = object motion added, green = residual size (full at 4 px), grey "
             "= not judged; composition now x{:.2f}, floor {:.2f} px + {:.4f}/px, fast movers {}, scene {}x{}",
             on ? "ON" : "off", g_objectScale.load(), g_objectFloor.load(), g_objectSpeedTerm.load(),
             g_objectFastChain.load() ? "chained" : "extrapolated", SceneW(), SceneH());
}

void DuniaTaa_Dx12::GetObjectComposition(float* scale, float* floor, float* speedTerm, bool* fastChain)
{
    *scale = g_objectScale.load();
    *floor = g_objectFloor.load();
    *speedTerm = g_objectSpeedTerm.load();
    *fastChain = g_objectFastChain.load();
}
void DuniaTaa_Dx12::SetObjectComposition(float scale, float floor, float speedTerm, bool fastChain, bool log)
{
    g_objectScale = std::clamp(scale, 0.0f, 3.0f);
    g_objectFloor = std::clamp(floor, 0.0f, 4.0f);
    g_objectSpeedTerm = std::clamp(speedTerm, 0.0f, 0.01f);
    g_objectFastChain = fastChain;
    // Logged once per settled change (a slider drag applies live but logs on release), with the scale it was
    // judged at: the point of the sliders is a value per Resolution Scale.
    if (log)
        LOG_INFO("DuniaTaa: object composition: residual x{:.2f}, floor {:.2f} render px + {:.4f}/px of camera "
                 "motion, fast movers {} -- at a {}x{} scene ({}% of the eye) (values only, not saved)",
                 g_objectScale.load(), g_objectFloor.load(), g_objectSpeedTerm.load(),
                 fastChain ? "CHAINED" : "extrapolated", SceneW(), SceneH(),
                 g_destW != 0 ? (int) std::lround(100.0 * SceneW() / g_destW) : 0);
}

void DuniaTaa_Dx12::SetSameEyeCopies(int mask)
{
    g_pvCopyMask = std::clamp(mask, 0, 4);
    g_mvStatsRestart.store(true, std::memory_order_relaxed);
    LOG_INFO("DuniaTaa: same-eye patch now written to copies: mode {}", g_pvCopyMask.load());
}

int DuniaTaa_Dx12::JitterMode() { return JitterModeNow(); }
int DuniaTaa_Dx12::JitterSign() { return g_jitterSign.load(); }
int DuniaTaa_Dx12::BaselineOld() { return g_baselineOld.load(); }
void DuniaTaa_Dx12::SetBaselineOld(int old)
{
    g_baselineOld = old != 0 ? 1 : 0;
    g_needReset = true;
    LOG_INFO("DuniaTaa: other-eye baseline: {} (values only; A/B for the haloing seen with the new one)",
             old != 0 ? "OLD, as in the older build (+jitter on rows x, y)" : "NEW, as derived (-jitter)");
}

void DuniaTaa_Dx12::SetJitterMode(int mode)
{
    mode = std::clamp(mode, 0, 4);
    if (mode == JitterModeNow())
        return;
    g_jitterMode = mode;
    // The shake tests are never written to the ini: a game that starts up shaking is not a setting.
    if (mode <= 2)
        Config::Instance()->DuniaTaaJitter = mode;
    // Values only -- nothing is created or destroyed -- but DLSS's history was gathered under the other
    // assumption, so it starts again. Leaving mode 2 stops the writes; the game's next write of the struct
    // carries its own jitter again.
    g_needReset = true;
    g_mvStatsRestart.store(true, std::memory_order_relaxed);
    LOG_INFO("DuniaTaa: jitter mode {} -- {}", mode,
             mode == 0   ? "DLSS told zero (the old behaviour)"
             : mode == 1 ? "DLSS told the game's own 2-phase jitter"
             : mode == 2 ? "own per-eye Halton jitter patched into the view constants and reported"
             : mode == 3 ? "SHAKE TEST A: our jitter x20 patched in, DLSS told zero -- the image should SHAKE"
                         : "SHAKE TEST B: our jitter x4 patched in and reported -- the image should stand STILL");
}

void DuniaTaa_Dx12::SetJitterSign(int variant)
{
    g_jitterSign = std::clamp(variant, 0, 3);
    g_needReset = true;
    LOG_INFO("DuniaTaa: jitter sign variant {} (x {:+.0f}, y {:+.0f})", g_jitterSign.load(),
             kJitterSigns[g_jitterSign.load()][0], kJitterSigns[g_jitterSign.load()][1]);
}

void DuniaTaa_Dx12::BeforeExecute()
{
    // Every submit of every queue comes through here, so the way out is one atomic load and an ini read.
    // Only while real motion is on, which already means gameplay at the eye size: the structs this touches
    // were found by the eye-sized viewport quad, so a menu's are never candidates.
    const bool mirror = g_pvMirrorOn.load(std::memory_order_relaxed);
    if (!g_realMotionOn.load(std::memory_order_relaxed) || (!SameEyeWanted() && JitterModeNow() < 2 && !mirror))
        return;
    PatchPerViewPrevious();
    if (mirror)
        g_mirrorReady.store(g_viewConstants.MirrorFill(g_presentIndex.load()), std::memory_order_relaxed);
}

void DuniaTaa_Dx12::AfterDispatch(ID3D12GraphicsCommandList* cmd, ID3D12PipelineState* pipeline)
{
    if (cmd == nullptr)
        return;

    // The game's upscale dispatch: learn it, then copy DLSS's eye-sized output over what it just wrote.
    // OFF: the writer probe showed the upscale is a DRAW (see ShouldSkipDraw); this route never
    // learned anything and took a lock on every dispatch in the frame. Kept for the record, never entered.
    if (false && SpliceTaaMode() && IsUpscale() && g_output != nullptr)
    {
        ID3D12Resource* target = nullptr;
        bool viaTable = false;
        {
            std::lock_guard<std::mutex> pending(g_cmdUpscaleLock);
            const auto it = g_cmdUpscale.find(cmd);
            if (it != g_cmdUpscale.end())
            {
                target = it->second;
                viaTable = true;
                g_cmdUpscale.erase(it);
            }
        }
        const size_t parity = (size_t) (g_presentIndex.load() & 1);
        if (viaTable)
        {
            if (pipeline != nullptr)
                g_upPipeline = pipeline;
            g_upByParity[parity].store(target, std::memory_order_relaxed);
            ID3D12Resource* const p0 = g_upByParity[0].load(std::memory_order_relaxed);
            ID3D12Resource* const p1 = g_upByParity[1].load(std::memory_order_relaxed);
            if (g_upPipeline != nullptr && p0 != nullptr && p1 != nullptr)
            {
                g_upPipelineReady.store(true, std::memory_order_release);
                static bool logged = false;
                if (!logged)
                {
                    logged = true;
                    LOG_INFO("DuniaTaa: upscaling: the game's upscale DISPATCH learned -- pipeline {:X}, output "
                             "parity 0 -> {:X}, parity 1 -> {:X}; DLSS's eye image is copied over it from here on",
                             (size_t) g_upPipeline, (size_t) p0, (size_t) p1);
                }
            }
        }
        else if (g_upPipelineReady.load(std::memory_order_acquire) && pipeline != nullptr && pipeline == g_upPipeline)
        {
            target = g_upByParity[parity].load(std::memory_order_relaxed);
        }
        if (target != nullptr && g_resolveSplicedPresent == g_presentIndex.load())
        {
            // The UAV a dispatch just wrote is UNORDERED_ACCESS by contract, on this same list.
            D3D12_RESOURCE_BARRIER uav {};
            uav.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
            uav.UAV.pResource = target;
            cmd->ResourceBarrier(1, &uav);
            if (CopyOutputToDest(cmd, target, D3D12_RESOURCE_STATE_UNORDERED_ACCESS))
            {
                g_upCopies.fetch_add(1, std::memory_order_relaxed);
                g_splicedThisFrame = true;
            }
        }
    }

    if (!g_realMotionOn.load(std::memory_order_relaxed))
        return;

    // Once the pass is known by its pipeline the table route no longer fills the map, so every other
    // dispatch in the frame (1,500 of them) can leave without the lock.
    if (g_mvPipelineReady.load(std::memory_order_acquire) && pipeline != g_mvPipeline)
        return;
    // The velocity-reader probe's reference point: the motion pass, in this list's recording order.
    if (g_mvPipelineReady.load(std::memory_order_acquire) && g_pcState.load(std::memory_order_relaxed) == 2)
    {
        bool computeSig = false, heaps = false, havePipeline = false;
        int params = -1, invalid = 0;
        const bool sig = D3D12Hooks::DescribeRestore(cmd, &computeSig, &params, &invalid, &heaps, &havePipeline);
        std::lock_guard<std::mutex> lock(g_pcLock);
        VelNoteLocked(cmd, 1, 0);
        g_rsDispatches++;
        const bool complete = sig && computeSig && params > 0 && invalid == 0 && heaps && havePipeline;
        g_rsComplete += complete ? 1 : 0;
        g_rsNoSig += sig ? 0 : 1;
        g_rsNotCompute += (sig && !computeSig) ? 1 : 0;
        g_rsNoParams += params <= 0 ? 1 : 0;
        g_rsInvalidParams += invalid > 0 ? 1 : 0;
        g_rsNoHeaps += heaps ? 0 : 1;
        g_rsNoPipeline += havePipeline ? 0 : 1;
        if (g_rsLogged < 3)
        {
            g_rsLogged++;
            LOG_INFO("DuniaTaa: conversion probe: at the motion pass dispatch the list's restorable state is -- root "
                     "signature {}{}, root parameters recorded {} ({} unknown), descriptor heaps {}, pipeline {} -> {}",
                     sig ? "recorded" : "NOT recorded", sig ? (computeSig ? " (compute)" : " (GRAPHICS, not compute)") : "",
                     params, invalid, heaps ? "recorded" : "NOT recorded", havePipeline ? "recorded" : "NOT recorded",
                     complete ? "COMPLETE" : "INCOMPLETE");
        }
    }

    ID3D12Resource* motion = nullptr;
    bool viaTable = false;
    {
        std::lock_guard<std::mutex> pending(g_cmdMotionLock);
        const auto it = g_cmdMotion.find(cmd);
        if (it != g_cmdMotion.end())
        {
            motion = it->second;
            viaTable = true;
            g_cmdMotion.erase(it);
        }
    }

    const size_t parity = (size_t) (g_presentIndex.load() & 1);

    if (viaTable)
    {
        // Learning. Remember the pipeline and which texture this parity uses; once both parities are
        // mapped the descriptor route -- and the mirroring that pays for it -- can be switched off.
        if (pipeline != nullptr)
            g_mvPipeline = pipeline;
        g_mvByParity[parity].store(motion, std::memory_order_relaxed);
        ID3D12Resource* const p0 = g_mvByParity[0].load(std::memory_order_relaxed);
        ID3D12Resource* const p1 = g_mvByParity[1].load(std::memory_order_relaxed);
        if (g_mvPipeline != nullptr && p0 != nullptr && p1 != nullptr && p0 != p1)
        {
            g_mvPipelineReady.store(true, std::memory_order_release);
            if (!g_mvPipelineLogged)
            {
                g_mvPipelineLogged = true;
                LOG_INFO("DuniaTaa: motion pass pipeline {:X} learned{}; parity 0 -> {:X}, parity 1 -> {:X}. "
                         "Heap mirroring off -- the pass is recognised by its pipeline until one of these "
                         "textures dies (a menu), when this is learned again",
                         (size_t) g_mvPipeline,
                         g_mvRelearns.load() ? std::format(" (re-learned, {} so far)", g_mvRelearns.load()) : "",
                         (size_t) p0, (size_t) p1);
            }
        }
    }
    else if (g_mvPipelineReady.load(std::memory_order_acquire) && pipeline != nullptr &&
             pipeline == g_mvPipeline)
    {
        motion = g_mvByParity[parity].load(std::memory_order_relaxed);
    }

    if (motion == nullptr)
        return;

    // Menus, loading screens and pre-VR frames render at sizes of their own, and the motion pass runs
    // for them too. Doing work at a size this is not for is what crashed FC5, and it is what
    // crashed here on entering the menu: when the size is not the one expected, do nothing at all.
    const auto desc = motion->GetDesc();
    if (!HalfEyeSized(desc.Width, desc.Height))
    {
        static bool logged = false;
        if (!logged)
        {
            logged = true;
            LOG_INFO("DuniaTaa: real motion: skipping a {}x{} motion texture -- not half of the {}x{} eye",
                     desc.Width, desc.Height, g_pinnedW, g_pinnedH);
        }
        return;
    }

    // Same-eye vectors: rewrite the pass's reprojection matrix now -- after the game wrote it, before the
    // GPU reads it. Below the size gate on purpose: a menu's motion pass is none of our business. No GPU
    // work and nothing recorded; see ViewConstants::PatchTemporal for what makes the write safe.
    // The copy is described as "may hold same-eye vectors" whenever either patch is writing: the G-buffer's
    // object vectors turn same-eye through the per-view patch even on a frame where this block did not
    // verify. The decode judges against both baselines, so saying so for other-eye vectors costs nothing.
    const bool blockPatched =
        SameEyeWanted() && g_sameEyeBlockOn.load(std::memory_order_relaxed) && PatchMotionPassMatrix();
    // The private copy makes the G-buffer's object vectors same-eye too (only for the binds it redirected, which
    // is why the decode judges every pixel against both baselines).
    const bool sameEye = blockPatched || (SameEyeWanted() && g_sameEyePatchOn.load(std::memory_order_relaxed)) ||
                         g_pvMirrorOn.load(std::memory_order_relaxed);

    // Creation, resizing and the deferred release all happen inside PrepareCopy, under its own lock.
    // nullptr means there is nothing to copy into this frame.
    ID3D12Resource* target = g_mvStaging.PrepareCopy(cmd, desc, g_presentIndex.load());
    if (target == nullptr)
        return;

    // The one state fact available here, and the reason the copy goes at the dispatch rather than at the
    // splice: D3D12 requires a UAV a dispatch writes to be in UNORDERED_ACCESS, and this is the same list,
    // where recorded order is execution order. At the splice, on another list, the state would be a guess.
    // The UAV barrier first, so the dispatch's writes have landed before the copy reads them.
    D3D12_RESOURCE_BARRIER uav {};
    uav.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    uav.UAV.pResource = motion;
    cmd->ResourceBarrier(1, &uav);

    // target is the buffer not holding the previous Present's copy, so both are readable at the splice.
    // The previous frame's decode has already been submitted by then, and submission order is execution
    // order on one queue, so overwriting it here cannot race with a read of it.
    Barrier(cmd, motion, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
    Barrier(cmd, target, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST);
    g_skipCopyHook++;
    cmd->CopyResource(target, motion);
    g_skipCopyHook--;
    Barrier(cmd, target, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    Barrier(cmd, motion, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    // Advances the pair and marks the copy fresh for exactly one decode. What kind of vectors it holds goes
    // first, so the decode never sees a fresh copy with the previous copy's description.
    g_mvCopySameEye.store(sameEye, std::memory_order_release);
    g_mvStaging.CommitCopy();
    g_mvCopies.fetch_add(1, std::memory_order_relaxed);
}

void DuniaTaa_Dx12::NoteMotion(ID3D12Resource* motion, bool boundAsRenderTarget, ID3D12GraphicsCommandList* cmd)
{
    if (g_inEvaluate > 0)
        return;
    if (!Active() || motion == nullptr || IsSwapchainBuffer(motion) || IsOwnedResource(motion))
        return;

    const auto desc = motion->GetDesc();
    if (desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || desc.SampleDesc.Count > 1)
        return;

    // The capture's net is deliberately wider than the scorer's, and is offered first. FC6 ships TAA as its main
    // anti-aliasing, and TAA reprojection needs per-pixel motion, so motion exists somewhere in this frame; the
    // renderer is described as keeping it in the G-buffer. MotionScore only ever accepts a two-channel format, so
    // a motion pair packed into a wider G-buffer target is dropped before it can ever become a candidate -- which
    // is exactly the blind spot that left the first three captures looking at two textures nothing writes.
    static std::unordered_set<uint64_t> loggedTarget;
    const uint64_t targetKey =
        ((uint64_t) desc.Width << 32) | ((uint64_t) desc.Height << 16) | (uint64_t) desc.Format;
    if (FrameCapture_Dx12::IsCaptureShape(desc) && loggedTarget.insert(targetKey).second)
    {
        LOG_INFO("DuniaTaa: eye-sized render target {}x{} fmt {} flags {:X} (capture priority {})", desc.Width,
                 desc.Height, (UINT) desc.Format, (UINT) desc.Flags, FrameCapture_Dx12::CapturePriority(desc));
    }

    // Which candidate holds the real motion is what the F7 capture is there to answer, so it is offered all of
    // them, not just the ones this function's score would adopt.
    FrameCapture_Dx12::NoteCandidate(motion, boundAsRenderTarget, cmd);

    const int score = MotionScore(desc);
    if (score < 0)
        return;

    static std::unordered_set<uint64_t> loggedMv;
    if (loggedMv.insert(targetKey).second)
    {
        LOG_INFO("DuniaTaa: candidate motion {}x{} fmt {} flags {:X}", desc.Width, desc.Height, (UINT) desc.Format,
                 (UINT) desc.Flags);
    }

    std::lock_guard<std::mutex> lock(g_lock);

    // Scores are only compared against the incumbent, so flipping the preference has no effect until
    // the current pick is dropped and selection runs again.
    static bool lastPreferUnorm = Config::Instance()->DuniaTaaMvPreferUnorm.value_or_default();
    const bool preferUnorm = Config::Instance()->DuniaTaaMvPreferUnorm.value_or_default();
    if (lastPreferUnorm != preferUnorm)
    {
        lastPreferUnorm = preferUnorm;
        Retain(&g_gameMv, nullptr);
        g_mvViewsValid = false;
        g_needReset = true;
    }

    if (g_color != nullptr && g_pinnedW != 0 && !MvSizeOk((UINT) desc.Width, desc.Height))
        return;

    if (g_gameMv != nullptr)
    {
        if (g_gameMv == motion)
        {
            if (boundAsRenderTarget)
                g_gameMvIsRt = true;
            return;
        }
        if (MvSizeOk((UINT) g_gameMv->GetDesc().Width, g_gameMv->GetDesc().Height) &&
            MotionScore(g_gameMv->GetDesc()) >= score)
            return;
    }

    // Held with a reference: the evaluate path reads this every frame, and the game frees its
    // render targets on resolution and settings changes.
    Retain(&g_gameMv, motion);
    g_mvViewsValid = false;
    g_gameMvIsRt = boundAsRenderTarget;
    g_needReset = true;
}

void DuniaTaa_Dx12::OnSetTargets(ID3D12GraphicsCommandList* cmd, ID3D12Resource* color, ID3D12Resource* depth)
{
    if (g_inEvaluate > 0)
        return;
    if (Active() && Tracing())
    {
        g_traceLines++;
        LOG_INFO("DuniaTaa trace: present {} bind cmd {:X} thread {}{} color {} depth {}", g_presentIndex.load(),
                 (size_t) cmd, GetCurrentThreadId(), TraceDrawsSince(cmd), TraceDesc(color), TraceDesc(depth));
    }
    Note(color, depth);
    NoteMotion(color, true);
    if (cmd == nullptr || cmd == g_cmd)
        return;

    // The TAA resolve's own target, for step 3: whatever this list has bound when the resolve draws is
    // where the game's resolve writes, and therefore where DLSS's output has to go.
    if (WantsTaaSearch())
    {
        std::lock_guard<std::mutex> targets(g_taaTargetLock);
        g_taaListTarget[cmd] = { color, depth };
    }

    // The depth the splice feeds DLSS, kept per list for the same reason the tonemapped path keeps its
    // own: the next eye's lists are recorded before this eye is presented, so "the last eye depth" can
    // belong to the other eye. The depth last bound on THIS list at the resolve's own size is this eye's.
    // The resolve itself binds no DSV, which is why it has to come from an earlier bind.
    if (SpliceTaaMode() && depth != nullptr && SceneW() != 0)
    {
        const auto depthDesc = depth->GetDesc();
        if ((UINT) depthDesc.Width == SceneW() && depthDesc.Height == SceneH())
            g_taaResolve.NoteEyeDepth(cmd, depth);
    }

    // Tonemapped mode: pair each finished 8-bit eye image with the eye-sized depth its list drew with before it.
    if (TonemappedMode())
    {
        if (depth != nullptr)
        {
            const auto depthDesc = depth->GetDesc();
            if (IsEvalSize((UINT) depthDesc.Width, depthDesc.Height))
            {
                std::lock_guard<std::mutex> lock(g_lock);
                g_cmdEyeDepth[cmd] = depth;
            }
        }
        else if (color != nullptr)
        {
            const auto colorDesc = color->GetDesc();
            if (colorDesc.Format == DXGI_FORMAT_R8G8B8A8_UNORM && IsEvalSize((UINT) colorDesc.Width, colorDesc.Height))
            {
                std::lock_guard<std::mutex> lock(g_lock);
                const auto it = g_cmdEyeDepth.find(cmd);
                if (it != g_cmdEyeDepth.end() && it->second != nullptr)
                {
                    const auto depthDesc = it->second->GetDesc();
                    if (depthDesc.Width == colorDesc.Width && depthDesc.Height == colorDesc.Height)
                    {
                        int slot = -1;
                        for (int i = 0; i < kTonemapSlots; ++i)
                        {
                            if (g_tonemapDepth[i].image == color)
                                slot = i;
                        }
                        if (slot < 0)
                        {
                            slot = g_tonemapNextSlot;
                            g_tonemapNextSlot = (g_tonemapNextSlot + 1) % kTonemapSlots;
                        }
                        g_tonemapDepth[slot].image = color;
                        RetainDeferred(&g_tonemapDepth[slot].depth, it->second, g_presentIndex.load());
                    }
                }
            }
        }
    }

    std::lock_guard<std::mutex> lock(g_lock);
    auto& state = g_cmdTargets[cmd];
    state.spliceColor = nullptr;
    state.spliceDepth = nullptr;
    // The private-copy probe's "is this the G-buffer pass": decided HERE, while the targets are bound and so alive.
    // The stored pointers are never read later -- entries outlive list resets, and a menu frees the targets.
    // Only while the probe runs (its warm-up refreshes every list's flag before it counts) -- AND while the
    // private copy is on, which redirects only on G-buffer lists: an earlier build reused this flag with the probe-only
    // gate still here, and turned away every one of ~2000 qualifying binds a Present as "not a G-buffer
    // list". A list's flag refreshes at its next bind after switching on; until then it reads as not G-buffer.
    if (g_pcState.load(std::memory_order_relaxed) != 0 || g_pvMirrorOn.load(std::memory_order_relaxed))
        state.gbuffer = color != nullptr && depth != nullptr && IsMotionFormat(color->GetDesc().Format);

    if (color != nullptr && depth != nullptr && color == g_gameMv)
        Retain(&g_mvDepth, depth);

    if (IsUpscale() && color != nullptr && depth != nullptr)
    {
        const auto colorDesc = color->GetDesc();
        const auto depthDesc = depth->GetDesc();
        if (IsHdrFormat(colorDesc.Format) && colorDesc.Width == g_createdInW && colorDesc.Height == g_createdInH &&
            depthDesc.Width == g_createdInW && depthDesc.Height == g_createdInH)
            g_lastSceneDepth = depth;
    }

    // Gameplay signature for automatic mode and for the scale-change warnings.
    if (color != nullptr && depth != nullptr && g_destW != 0)
    {
        const auto desc = color->GetDesc();
        if (IsMotionFormat(desc.Format))
        {
            const uint64_t now = g_presentIndex.load() + 1;
            UINT w = (UINT) desc.Width;
            UINT h = desc.Height;
            FakeProbeSize(w, h); // the probe pass draws an R16G16 target too; at one eye size it matched the eye
            if (std::abs((int) w - (int) g_destW) <= 32 && std::abs((int) h - (int) g_destH) <= 32)
            {
                if (now - g_gameplayFullPresent.load() > 30)
                    g_gameplayFullSince = now;
                g_gameplayFullPresent = now;
            }
            else if (IsScaledTaa(w, h, g_destW, g_destH, true))
            {
                g_gameplayScaledW = w;
                g_gameplayScaledH = h;
                g_gameplayScaledPresent = now;
            }
        }
    }

    // The eye itself still being drawn is what separates "upscaling stopped" from a menu or loading screen.
    if (EffectiveMode() != 0 && color != nullptr && depth != nullptr && g_destW != 0)
    {
        const auto desc = color->GetDesc();
        if (desc.Width == g_destW && desc.Height == g_destH)
        {
            const uint64_t now = g_presentIndex.load() + 1;
            g_lastEyeScenePresent = now;
            uint64_t unset = 0;
            g_firstEyeScenePresent.compare_exchange_strong(unset, now);
        }
    }

    if (IsUpscale())
    {
        state.upscaleSrc = nullptr;
        state.copyDest = nullptr;

        if (SpliceTaaMode())
        {
            // The game's upscale draw, from the writer probe's facts: on the list
            // that spliced this Present, right after the resolve, an eye-sized HDR target (flags RT|UAV) is
            // bound with NO depth and gets exactly ONE draw; the same target is then re-bound WITH depth for
            // ~127 in-world draws, then read as a graphics SRV by the tonemap. So: the first eye-sized HDR
            // bind without depth on a list that spliced this Present is the upscale, and AfterDraw copies
            // DLSS's eye image over it after that one draw (the target is in RENDER_TARGET state there).
            // The strict rule below (previous bind = render-size source) belongs to the old HDR-upscale mode
            // and ran AHEAD of the splice-mode rule; whether it matched or not, the splice-mode rule never
            // logged in five runs. In splice mode it no longer runs at all, and DLSS's own splice keeps the
            // size checks (PendingSplice), so the mismatch detection it carried is not needed here.
            // A bind that is eye-sized HDR but NOT taken says why, a few times: "no bind at all" and "the
            // rule never ran" were indistinguishable for two runs.
            if (g_handle != nullptr && color != nullptr && color != g_output)
            {
                const auto destDesc = color->GetDesc();
                if (IsHdrFormat(destDesc.Format) && destDesc.Width == g_createdOutW && destDesc.Height == g_createdOutH)
                {
                    const uint64_t present = g_presentIndex.load();
                    const bool fresh = g_resolveSplicedPresent == present;
                    if (fresh && depth == nullptr)
                    {
                        state.upscaleDrawTarget = color;
                        t_upscaleCmd = cmd;
                        static bool logged = false;
                        if (!logged)
                        {
                            logged = true;
                            LOG_INFO("DuniaTaa: upscaling: the game's upscale draw -- eye-sized HDR {:X} {}x{} fmt {} "
                                     "bound with no depth on the list that spliced this Present ({}); DLSS's eye image "
                                     "is copied over it right after its one draw (see 'copied over the game's upscale "
                                     "target')",
                                     (size_t) color, destDesc.Width, destDesc.Height, (UINT) destDesc.Format, present);
                        }
                    }
                    else
                    {
                        static std::atomic<int> nearMisses { 0 };
                        if (nearMisses.fetch_add(1, std::memory_order_relaxed) < 6)
                            LOG_INFO("DuniaTaa: upscaling: eye-sized HDR {:X} fmt {} bound but NOT taken as the upscale "
                                     "draw -- spliced this Present: {} (present {}, splice marker {}), depth bound: {}",
                                     (size_t) color, (UINT) destDesc.Format, fresh ? "yes" : "NO", present,
                                     g_resolveSplicedPresent, depth != nullptr ? "yes" : "no");
                    }
                }
            }
        }
        else if (state.copyPending)
        {
            // Only straight after the upscale draw, onto the target it drew into. Not tied to the
            // Present count: the next eye's lists are recorded before the previous eye is presented,
            // and a per-Present gate silently skipped one eye every frame.
            state.copyPending = false;
            if (state.color != nullptr && state.color == state.pendingDest)
                state.copyDest = state.color;
        }
        else if (g_handle != nullptr && color != nullptr && depth == nullptr &&
                 state.color != nullptr && state.depth == nullptr && state.color != color)
        {
            const auto destDesc = color->GetDesc();
            const auto srcDesc = state.color->GetDesc();
            if (IsHdrFormat(destDesc.Format) && IsHdrFormat(srcDesc.Format) && destDesc.Width == g_createdOutW &&
                destDesc.Height == g_createdOutH && srcDesc.Width == g_createdInW && srcDesc.Height == g_createdInH)
            {
                state.upscaleSrc = state.color;
            }
            else if (IsHdrFormat(destDesc.Format) && IsHdrFormat(srcDesc.Format) &&
                     IsScaledTaa((UINT) srcDesc.Width, srcDesc.Height, (UINT) destDesc.Width, destDesc.Height))
            {
                // The same upscale draw at different sizes: the resolution scale (or the eye size) changed
                // after the feature was created. Exact half-size sources are excluded by IsScaledTaa, which
                // keeps the half-resolution bloom composite at 100% scale from matching.
                g_mismatchW = (unsigned) srcDesc.Width;
                g_mismatchH = srcDesc.Height;
                g_mismatchPresent = g_presentIndex.load() + 1;
            }
        }
        if (color != state.upscaleDrawTarget)
        {
            state.upscaleDrawTarget = nullptr;
            if (t_upscaleCmd == cmd)
                t_upscaleCmd = nullptr;
        }
        state.color = color;
        state.depth = depth;
        return;
    }

    ID3D12Resource* evalColor = IsUpscale() && g_taa != nullptr ? g_taa : g_color;
    ID3D12Resource* evalDepth = IsUpscale() && g_taaDepth != nullptr ? g_taaDepth : g_depth;

    if (state.color != nullptr && state.color == evalColor)
    {
        state.lastPinned = state.color;
        state.lastPinnedDepth = state.depth != nullptr ? state.depth : evalDepth;
    }

    if (!g_splicedThisFrame && g_handle != nullptr && evalColor != nullptr && evalDepth != nullptr &&
        state.color == evalColor && color != evalColor)
    {
        state.spliceColor = evalColor;
        state.spliceDepth = evalDepth;
    }

    state.color = color;
    state.depth = depth;
}

void DuniaTaa_Dx12::ToggleStart()
{
    const bool on = !g_started.load();
    g_started = on;
    LOG_INFO("DuniaTaa: F6: {}", on ? (g_handle == nullptr ? "starting (creating the feature, a few seconds)"
                                                           : "ours again")
                                    : "the game's own picture");
    // One key for everyday play ([Inputs] DuniaTaaDiagnostics off): F6 on also splices DLSS at the TAA resolve and
    // turns object motion on, as soon as each is ready (DriveOneKey, at Present); F6 off takes both down
    // at once, so the picture really is the game's own. With diagnostics on, F6/F8/F4 stay three keys.
    if (!OneKeyMode())
        return;
    if (on)
    {
        g_oneKeyArm = true;
        return;
    }
    g_oneKeyArm = false;
    if (g_taaMode.load() != 0)
        ApplyTaaMode(0);
    if (g_realMotionMode.load() != kMotionOff)
    {
        g_sameEyeBlockOn = false;
        g_sameEyePatchOn = false;
        ApplyRealMotionMode(kMotionOff);
    }
}

bool DuniaTaa_Dx12::StartsOnKey() { return StartOnKey(); }

const char* DuniaTaa_Dx12::StartName()
{
    if (!Started())
        return "Game picture";
    if (g_handle == nullptr)
        return "Starting";
    // The feature was created for one render size and the game now renders another (the slider was moved
    // after F6, or its change landed a beat after it: once the feature was at 60% and the game at 70%,
    // nothing spliced for the run and no toast). Said in the state text, which the overlay and the F6
    // toast both show, so it cannot be missed whatever the status machinery decides.
    if (IsUpscale() && g_taaW != 0 &&
        (std::abs((int) g_taaW - (int) g_createdInW) > 32 || std::abs((int) g_taaH - (int) g_createdInH) > 32))
        return "Resolution Scale changed -- DLSS is rebuilding for the new size (if this stays, restart the game)";
    return TonemappedMode() ? "DLAA + NR" : DuniaQualityName(EffectiveMode());
}

bool DuniaTaa_Dx12::PendingSplice(ID3D12GraphicsCommandList* cmd)
{
    if (g_inEvaluate > 0)
        return false;
    if (cmd == nullptr || g_handle == nullptr || !SplicePathOn() || !g_allowSplice || !Started())
        return false;
    // DLAA splices once per Present. Upscaling runs on every upscale draw, one per eye, and those are
    // recorded across Present boundaries.
    if (g_splicedThisFrame && !IsUpscale())
        return false;
    if (IsUpscale())
    {
        if (g_taaW != g_createdInW || g_taaH != g_createdInH)
            return false;
        if (g_destW != g_createdOutW || g_destH != g_createdOutH)
            return false;
    }
    else if (g_createdOutW != 0 && (SceneW() != g_createdOutW || SceneH() != g_createdOutH))
        return false;

    std::lock_guard<std::mutex> lock(g_lock);
    auto it = g_cmdTargets.find(cmd);
    if (it == g_cmdTargets.end())
        return false;
    if (IsUpscale())
        return it->second.upscaleSrc != nullptr || it->second.copyDest != nullptr;
    return it->second.spliceColor != nullptr && it->second.spliceDepth != nullptr;
}

void DuniaTaa_Dx12::AfterOMSet(ID3D12GraphicsCommandList* cmd)
{
    if (g_inEvaluate > 0)
        return;
    if (cmd == nullptr || cmd == g_cmd)
        return;

    if (IsUpscale())
    {
        ID3D12Resource* src = nullptr;
        ID3D12Resource* dest = nullptr;
        ID3D12Resource* depth = nullptr;
        ID3D12Resource* boundDest = nullptr;
        {
            std::lock_guard<std::mutex> lock(g_lock);
            auto it = g_cmdTargets.find(cmd);
            if (it == g_cmdTargets.end())
                return;
            src = it->second.upscaleSrc;
            dest = it->second.copyDest;
            boundDest = it->second.color;
            it->second.upscaleSrc = nullptr;
            it->second.copyDest = nullptr;
            // Follow the depth this frame actually drew with; fall back to older picks only before one is seen.
            const auto sceneDepth = g_lastSceneDepth.load();
            if (sceneDepth != nullptr)
                RetainDeferred(&g_renderDepth, sceneDepth, g_presentIndex.load());
            depth = g_renderDepth != nullptr ? g_renderDepth : (g_mvDepth != nullptr ? g_mvDepth : g_taaDepth);
        }

        const int view = DiagnosticsOn() ? Config::Instance()->DuniaTaaUpscaleView.value_or_default() : 0;

        if (dest != nullptr)
        {
            const bool copied = view == 2 ? CopyOutputLeftHalf(cmd, dest, D3D12_RESOURCE_STATE_RENDER_TARGET)
                                          : CopyOutputToDest(cmd, dest, D3D12_RESOURCE_STATE_RENDER_TARGET);
            if (copied)
            {
                g_splicedThisFrame = true;
                static bool logged = false;
                if (!logged)
                {
                    logged = true;
                    LOG_INFO("DuniaTaa: upscaled output copied over the game's upscale into {:X}", (size_t) dest);
                }
            }
            return;
        }

        // Coming back from the game's upscale, the history is as old as the switch was long.
        static int lastView = view;
        if (lastView != view)
        {
            static const char* viewNames[] = { "DLSS", "game upscale", "split (DLSS left, game right)",
                                               "motion vectors" };
            LOG_INFO("DuniaTaa: upscale A/B view is now {}", view >= 0 && view <= 3 ? viewNames[view] : "unknown");
        }
        if (lastView == 1 && view != 1)
            g_needReset = true;
        lastView = view;

        // Game upscale only: skip the evaluation as well, so the framerate comparison is fair too.
        if (src == nullptr || view == 1)
            return;

        // Splice mode: DLSS already ran at this eye's TAA resolve and its eye-sized output is in g_output;
        // the game's upscale draw is about to write, and the next bind copies our output over it.
        if (SpliceTaaMode())
        {
            static bool loggedSplice = false;
            if (!loggedSplice)
            {
                loggedSplice = true;
                LOG_INFO("DuniaTaa: upscaling: the game's upscale draw recognised; DLSS's {}x{} output from the "
                         "resolve will be copied over it from the next bind on",
                         g_createdOutW, g_createdOutH);
            }
            g_lastUpscalePresent = g_presentIndex.load() + 1;
            std::lock_guard<std::mutex> lock(g_lock);
            auto& state = g_cmdTargets[cmd];
            state.copyPending = true;
            state.pendingDest = boundDest;
            state.pendingPresent = g_presentIndex.load();
            return;
        }
        if (depth == nullptr)
            return;

        static bool loggedEval = false;
        if (!loggedEval)
        {
            loggedEval = true;
            const auto srcDesc = src->GetDesc();
            LOG_INFO("DuniaTaa: upscaling {}x{} fmt {} -> {}x{} on the game's upscale draw (depth {:X})",
                     srcDesc.Width, srcDesc.Height, (UINT) srcDesc.Format, g_createdOutW, g_createdOutH,
                     (size_t) depth);
        }

        // Which eye, by the game's own per-eye source buffer. A third buffer means the game reallocated
        // (resolution or settings change), so the mapping starts over with fresh histories.
        int eyeSlot = 0;
        {
            std::lock_guard<std::mutex> lock(g_lock);
            if (src == g_eyeSrc[0])
                eyeSlot = 0;
            else if (src == g_eyeSrc[1])
                eyeSlot = 1;
            else if (g_eyeSrc[0] == nullptr)
                g_eyeSrc[0] = src;
            else if (g_eyeSrc[1] == nullptr)
            {
                g_eyeSrc[1] = src;
                eyeSlot = 1;
            }
            else
            {
                LOG_INFO("DuniaTaa: new upscale source {:X}, restarting the per-eye mapping", (size_t) src);
                g_eyeSrc[0] = src;
                g_eyeSrc[1] = nullptr;
                g_needReset = true;
            }
        }

        if (eyeSlot == 1 && g_handleEye2 == nullptr && MvMode() != 0)
        {
            static bool logged = false;
            if (!logged)
            {
                logged = true;
                LOG_INFO("DuniaTaa: motion vectors on but only one upscaling feature was created at startup; both "
                         "eyes share its history until the game is restarted");
            }
        }

        const bool evaluated = EvaluateOnList(cmd, nullptr, src, depth, false, false, false,
                                              D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_RENDER_TARGET,
                                              eyeSlot);
        if (evaluated && view == 3)
            VisualizeMv(cmd);

        if (evaluated)
        {
            g_lastUpscalePresent = g_presentIndex.load() + 1;
            g_evalThisFrame = true;
            std::lock_guard<std::mutex> lock(g_lock);
            auto& state = g_cmdTargets[cmd];
            state.copyPending = true;
            state.pendingDest = boundDest;
            state.pendingPresent = g_presentIndex.load();
        }
        return;
    }

    ID3D12Resource* color = nullptr;
    ID3D12Resource* depth = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_lock);
        auto it = g_cmdTargets.find(cmd);
        if (it == g_cmdTargets.end())
            return;
        color = it->second.spliceColor;
        depth = it->second.spliceDepth;
        it->second.spliceColor = nullptr;
        it->second.spliceDepth = nullptr;
        if (color == nullptr || depth == nullptr || g_handle == nullptr)
            return;
        if (g_splicedThisFrame)
            return;
    }

    if (!g_loggedSplice)
    {
        g_loggedSplice = true;
        const auto desc = color->GetDesc();
        LOG_INFO("DuniaTaa: splicing {}+NR on eye unbind {}x{} fmt {}", DuniaQualityName(g_createdQuality), desc.Width,
                 desc.Height, (UINT) desc.Format);
    }

    if (EvaluateOnList(cmd, nullptr, color, depth, false, false, !IsUpscale(), D3D12_RESOURCE_STATE_RENDER_TARGET))
    {
        if (IsUpscale())
            g_evalThisFrame = true;
        else
            g_splicedThisFrame = true;
    }
}

void DuniaTaa_Dx12::AfterBound(ID3D12GraphicsCommandList* cmd)
{
    // Unused: the upscaled result is copied after the game's upscale draw (see CmdTargets). A copy made
    // when the destination is bound lands before that draw, which paints over it.
    (void) cmd;
}

void DuniaTaa_Dx12::BeforeCopy(ID3D12GraphicsCommandList* cmd, ID3D12Resource* dst, ID3D12Resource* src)
{
    if (g_inEvaluate > 0)
        return;
    if (Active() && g_skipCopyHook == 0 && Tracing())
    {
        g_traceLines++;
        LOG_INFO("DuniaTaa trace: present {} copy cmd {:X} thread {}{} src {} dst {}", g_presentIndex.load(),
                 (size_t) cmd, GetCurrentThreadId(), TraceDrawsSince(cmd), TraceDesc(src), TraceDesc(dst));
    }
    if (g_skipCopyHook > 0 || cmd == nullptr || cmd == g_cmd || src == nullptr || dst == nullptr || src == dst)
        return;

    if (g_probeOn.load(std::memory_order_relaxed))
    {
        ProbeSeen(ProbeHook::Copy);
        ProbeTouch(cmd, dst, ProbeKind::CopyDst, 0, 0);
        ProbeTouch(cmd, src, ProbeKind::CopySrc, 0, 0);
    }

    // Structural eye size (log only for now), in every mode: the finished RGBA8 eye image copied into an
    // sRGB texture in VR, the same copy tonemapped mode below works on. No size threshold -- that is the point.
    {
        const auto srcDesc = src->GetDesc();
        if (srcDesc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D && srcDesc.Format == DXGI_FORMAT_R8G8B8A8_UNORM &&
            srcDesc.Width >= 256 && srcDesc.Height >= 256 && !IsSwapchainBuffer(src) &&
            dst->GetDesc().Format == DXGI_FORMAT_R8G8B8A8_UNORM_SRGB)
            StructNoteEye((UINT) srcDesc.Width, srcDesc.Height);
    }

    // Tonemapped mode: in VR the game's finished 8-bit eye image (render-target only) is copied into
    // an sRGB texture. Everything the game draws is done by then, so DLAA+NR runs on it here, on the copy's
    // list, and writes the result back before the copy reads it. Once per eye image, not per Present.
    if (TonemappedMode())
    {
        const auto srcDesc = src->GetDesc();
        const auto dstDesc = dst->GetDesc();
        if (srcDesc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || srcDesc.Format != DXGI_FORMAT_R8G8B8A8_UNORM ||
            dstDesc.Format != DXGI_FORMAT_R8G8B8A8_UNORM_SRGB || !IsEvalSize((UINT) srcDesc.Width, srcDesc.Height) ||
            IsSwapchainBuffer(src))
            return;

        if (g_tonemapW.load() != (unsigned) srcDesc.Width || g_tonemapH.load() != srcDesc.Height)
        {
            LOG_INFO("DuniaTaa: tonemapped mode: finished eye image {}x{} fmt {} copied out to {}x{} fmt {}",
                     srcDesc.Width, srcDesc.Height, (UINT) srcDesc.Format, dstDesc.Width, dstDesc.Height,
                     (UINT) dstDesc.Format);
            g_tonemapW = (unsigned) srcDesc.Width;
            g_tonemapH = srcDesc.Height;
            g_tonemapFormat = (int) srcDesc.Format;
        }

        // Ahead of the evaluate below, which writes DLAA+NR over this image: a capture wants the game's own picture.
        g_skipCopyHook++;
        FrameCapture_Dx12::CaptureAt(cmd, src, D3D12_RESOURCE_STATE_COPY_SOURCE);
        g_skipCopyHook--;

        // NR deliberately not required. FC5's F6 runs DLAA whether or not NR is on -- NR is an orthogonal
        // toggle there -- and this path requiring it made F6 look dead whenever [DlssNr] Enabled was off.
        // Which was often: OptiScaler auto-saves the ini, so that key does not survive a session reliably,
        // and several runs were spent on "F6 does nothing" because of it. DLAA is worth having on its own.
        if (g_handle == nullptr || !g_allowSplice || !Started())
            return;

        // Says which of the two is running, once per combination, so "F6 does nothing" is answerable from
        // the log instead of from the ini.
        {
            const bool nr = NrOn();
            static int loggedWith = -1;
            if (loggedWith != (int) nr)
            {
                loggedWith = (int) nr;
                LOG_INFO("DuniaTaa: tonemapped mode: running {}", nr ? "DLAA + NR" : "DLAA only (NR off)");
            }
        }

        ID3D12Resource* depth = nullptr;
        {
            std::lock_guard<std::mutex> lock(g_lock);
            for (const auto& pair : g_tonemapDepth)
            {
                if (pair.image == src)
                    depth = pair.depth;
            }
        }
        if (depth == nullptr)
        {
            static bool loggedNoDepth = false;
            if (!loggedNoDepth)
            {
                loggedNoDepth = true;
                LOG_INFO("DuniaTaa: tonemapped mode: no eye depth paired with image {:X} yet", (size_t) src);
            }
            return;
        }

        static bool logged = false;
        if (!logged)
        {
            logged = true;
            LOG_INFO("DuniaTaa: tonemapped mode: DLAA+NR on the finished {}x{} eye image before the eye copy-out",
                     srcDesc.Width, srcDesc.Height);
        }
        if (EvaluateOnList(cmd, nullptr, src, depth, false, false, true, D3D12_RESOURCE_STATE_COPY_SOURCE))
            g_lastUpscalePresent = g_presentIndex.load() + 1;
        return;
    }

    if (g_handle == nullptr || g_splicedThisFrame || !SplicePathOn() || !g_allowSplice || !Started())
        return;
    if (IsUpscale())
        return;

    ID3D12Resource* color = nullptr;
    ID3D12Resource* depth = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_lock);
        color = g_color;
        depth = g_depth;
    }
    if (src != color || depth == nullptr)
        return;
    if (IsSwapchainBuffer(src) || IsSwapchainBuffer(dst))
        return;

    const auto srcDesc = src->GetDesc();
    const auto dstDesc = dst->GetDesc();
    if (dstDesc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D)
        return;
    if (!IsEvalSize((UINT) dstDesc.Width, dstDesc.Height) &&
        dstDesc.Width * dstDesc.Height < (srcDesc.Width * srcDesc.Height) / 3)
        return;

    if (!g_loggedCopySplice)
    {
        g_loggedCopySplice = true;
        LOG_INFO("DuniaTaa: splicing DLAA+NR before copy {}x{} fmt {} -> {}x{} fmt {}", srcDesc.Width, srcDesc.Height,
                 (UINT) srcDesc.Format, dstDesc.Width, dstDesc.Height, (UINT) dstDesc.Format);
    }

    g_skipCopyHook++;
    FrameCapture_Dx12::CaptureAt(cmd, color, D3D12_RESOURCE_STATE_COPY_SOURCE);
    g_skipCopyHook--;

    if (EvaluateOnList(cmd, nullptr, color, depth, false, false, true, D3D12_RESOURCE_STATE_COPY_SOURCE))
        g_splicedThisFrame = true;
}

void DuniaTaa_Dx12::AfterCopy(ID3D12GraphicsCommandList* cmd, ID3D12Resource* dst, ID3D12Resource* src)
{
    if (g_inEvaluate > 0)
        return;
    if (g_skipCopyHook > 0 || cmd == nullptr || cmd == g_cmd || src == nullptr || dst == nullptr || src == dst)
        return;
    if (g_handle == nullptr || g_splicedThisFrame || !SplicePathOn() || !IsUpscale())
        return;

    ID3D12Resource* taa = nullptr;
    ID3D12Resource* depth = nullptr;
    ID3D12Resource* dest = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_lock);
        taa = g_taa;
        depth = g_taaDepth;
        dest = g_dest;
    }
    if (src != taa || dst != dest || taa == nullptr || dest == nullptr || depth == nullptr)
        return;
    if (IsSwapchainBuffer(src) || IsSwapchainBuffer(dst))
        return;

    if (g_evalThisFrame)
    {
        if (CopyOutputToDest(cmd, dest, D3D12_RESOURCE_STATE_COPY_DEST))
            g_splicedThisFrame = true;
        return;
    }

    if (!g_loggedCopySplice)
    {
        g_loggedCopySplice = true;
        const auto srcDesc = src->GetDesc();
        const auto dstDesc = dst->GetDesc();
        LOG_INFO("DuniaTaa: splicing upscale+NR after copy {}x{} -> {}x{}", srcDesc.Width, srcDesc.Height, dstDesc.Width,
                 dstDesc.Height);
    }

    if (EvaluateOnList(cmd, nullptr, taa, depth, false, false, false, D3D12_RESOURCE_STATE_COPY_SOURCE) &&
        CopyOutputToDest(cmd, dest, D3D12_RESOURCE_STATE_COPY_DEST))
        g_splicedThisFrame = true;
}

void DuniaTaa_Dx12::OnDraw(ID3D12GraphicsCommandList* cmd)
{
    if (g_inEvaluate > 0 || cmd == nullptr || !Active() || !Tracing())
        return;
    std::lock_guard<std::mutex> lock(g_traceDrawLock);
    ++g_traceDraws[cmd];
}

void DuniaTaa_Dx12::AfterDraw(ID3D12GraphicsCommandList* cmd)
{
    if (g_probeOn.load(std::memory_order_relaxed) && cmd != nullptr && g_inEvaluate == 0)
    {
        ProbeSeen(ProbeHook::Draw);
        std::lock_guard<std::mutex> lock(g_probeLock);
        ++g_probeDraws[cmd];
    }
    if (g_inEvaluate > 0 || cmd == nullptr || !SpliceTaaMode() || !IsUpscale() || g_output == nullptr)
        return;
    ID3D12Resource* target = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_lock);
        const auto it = g_cmdTargets.find(cmd);
        if (it == g_cmdTargets.end() || it->second.upscaleDrawTarget == nullptr ||
            it->second.color != it->second.upscaleDrawTarget)
            return;
        target = it->second.upscaleDrawTarget;
        // One copy per bind: the upscale is a single draw, and copying after every draw on a target with
        // many was 150 copies a frame.
        it->second.upscaleDrawTarget = nullptr;
    }
    // Bound as a render target for the draw that just recorded: RENDER_TARGET is a fact here.
    if (CopyOutputToDest(cmd, target, D3D12_RESOURCE_STATE_RENDER_TARGET))
    {
        g_upCopies.fetch_add(1, std::memory_order_relaxed);
        static bool logged = false;
        if (!logged)
        {
            logged = true;
            LOG_INFO("DuniaTaa: upscaling: DLSS's eye image copied over the game's upscale target {:X} right after "
                     "its draw -- the eye sees DLSS's upscale now, not FSR1's",
                     (size_t) target);
        }
    }
}

namespace
{
// Far Cry 6 dies without a WER record, a dump, or a line in this log, so a crash has been indistinguishable
// from a kill. This logs, for fatal exception codes only and at most three times, the faulting address and a
// short return-address trace as module+offset -- the .map beside the build turns an offset into a function.
// First-chance exceptions the game handles itself (sl.interposer's benign faults, C++ throws) are not fatal
// codes and are never logged. Installed once, from the present thread, when this module first presents.
std::string DescribeAddress(const void* addr)
{
    HMODULE mod = nullptr;
    if (addr != nullptr &&
        GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           (LPCSTR) addr, &mod) &&
        mod != nullptr)
    {
        char path[MAX_PATH] {};
        GetModuleFileNameA(mod, path, MAX_PATH);
        const char* name = std::strrchr(path, '\\');
        name = name != nullptr ? name + 1 : path;
        return std::format("{}+0x{:X}", name, (size_t) addr - (size_t) mod);
    }
    return std::format("{:X}", (size_t) addr);
}

LONG CALLBACK CrashReporter(EXCEPTION_POINTERS* ep)
{
    if (ep == nullptr || ep->ExceptionRecord == nullptr)
        return EXCEPTION_CONTINUE_SEARCH;
    const DWORD code = ep->ExceptionRecord->ExceptionCode;
    const bool fatal = code == EXCEPTION_ACCESS_VIOLATION || code == EXCEPTION_ILLEGAL_INSTRUCTION ||
                       code == EXCEPTION_STACK_OVERFLOW || code == EXCEPTION_INT_DIVIDE_BY_ZERO ||
                       code == EXCEPTION_ARRAY_BOUNDS_EXCEEDED || code == EXCEPTION_PRIV_INSTRUCTION ||
                       code == 0xC0000374 /* heap corruption */ || code == 0xC0000409 /* fail fast */;
    if (!fatal)
        return EXCEPTION_CONTINUE_SEARCH;
    static std::atomic<int> reported { 0 };
    if (reported.fetch_add(1) >= 3)
        return EXCEPTION_CONTINUE_SEARCH;

    const auto& rec = *ep->ExceptionRecord;
    std::string detail;
    if (code == EXCEPTION_ACCESS_VIOLATION && rec.NumberParameters >= 2)
        detail = std::format(" ({} of {:X})", rec.ExceptionInformation[0] == 0 ? "read"
                                              : rec.ExceptionInformation[0] == 1 ? "write"
                                                                                 : "execute",
                             (size_t) rec.ExceptionInformation[1]);
    LOG_ERROR("CRASH: exception {:X} at {}{} on thread {}", code, DescribeAddress(rec.ExceptionAddress), detail,
              GetCurrentThreadId());

    // The faulting thread's own frames, walked from the exception's context (CrashTrace.h). The handler's trace
    // below often stops at the dispatcher: a call through a null pointer gave only the reporter and three ntdll
    // frames (Cyberpunk 2077).
    void* fault[24] {};
    const int nf = WalkFaultStack(ep->ContextRecord, fault, 24);
    std::string stack;
    for (int i = 0; i < nf; ++i)
        stack += (i ? " < " : "") + DescribeAddress(fault[i]);
    LOG_ERROR("CRASH: faulting stack {}", stack);
    void* frames[24] {};
    const USHORT n = CaptureStackBackTrace(0, 24, frames, nullptr);
    std::string trace;
    for (USHORT i = 0; i < n; ++i)
        trace += (i ? " < " : "") + DescribeAddress(frames[i]);
    LOG_ERROR("CRASH: trace {}", trace);
    if (spdlog::default_logger() != nullptr)
        spdlog::default_logger()->flush();
    return EXCEPTION_CONTINUE_SEARCH;
}

void InstallCrashReporter()
{
    static std::atomic<bool> installed { false };
    if (installed.exchange(true))
        return;
    const bool ok = AddVectoredExceptionHandler(1, CrashReporter) != nullptr;
    LOG_INFO("DuniaTaa: crash reporter {} -- a fatal exception logs module+offset and a trace; map with the "
             ".map beside the build",
             ok ? "installed" : "NOT installed");
}
} // namespace

void DuniaTaa_Dx12::OnPresent(ID3D12CommandQueue* queue, IDXGISwapChain* swapChain)
{
    InstallCrashReporter();
    if (!Active() || queue == nullptr || swapChain == nullptr)
        return;
    (void) swapChain;

    // Reads back and writes out an armed capture once its frames are in. This function returns early in several
    // places below, so it sits at the top.
    FrameCapture_Dx12::OnPresent(queue);

    // Starts and stops the two searches, then decides the mirroring from both. Also at the top, for the
    // same reason.
    DriveMotionSearch();
    DriveTaaSearch();
    DriveTaaPairs();
    DriveUpscaleWriterProbe();
    DrivePrivateCopyProbe();
    DriveMirrorStats(g_presentIndex.load());
    DriveOneKey();
    LogSizeViews();
    const bool mirroring = NeedsHeapTracking();
    ResTrack_Dx12::SetForcedHeapTracking(mirroring);
    DriveGpuTiming(queue, mirroring);

    // The F4 A/B's numbers. Comparing frame rate across runs is meaningless here -- settings change
    // between and during them -- so the only useful comparison is one run with the path switching on and
    // off inside it, which needs the two states' rates side by side in the log.
    if (WantsRealMotion())
    {
        const uint64_t now = g_presentIndex.load();
        static uint64_t lastTick = 0;
        static uint64_t windowStart = 0;
        const bool restart = g_mvStatsRestart.exchange(false, std::memory_order_relaxed);
        if (g_mvStatsAt == 0)
        {
            g_mvStatsAt = now + kMvStatsPresents;
            windowStart = now;
            lastTick = GetTickCount64();
        }
        else if (restart || now >= g_mvStatsAt)
        {
            const uint64_t tick = GetTickCount64();
            const uint64_t ms = tick > lastTick ? tick - lastTick : 1;
            const uint64_t presents = now > windowStart ? now - windowStart : 1;
            // The state the window just ended in. On a restart that is the state being left, because the
            // toggle sets the flag after flipping -- so it is reported as the state it is flipping to,
            // which would be wrong. Hence "was".
            LOG_INFO("DuniaTaa: real motion was {} -- {} Presents in {} ms ({:.1f}/s), {} copies, {} decodes ({} with object motion){}",
                     (restart ? g_mvWindowWasOn.load() : g_realMotionOn.load()) ? "on" : "off",
                     presents, ms, (double) presents * 1000.0 / (double) ms, g_mvCopies.exchange(0),
                     g_mvDecodes.exchange(0), g_mvObjectDecodes.exchange(0), restart ? " (window closed by F4)" : "");
            if (SameEyeWanted())
                LOG_INFO("DuniaTaa: same-eye vectors, copies of the struct per frame in write order: {} first, {} "
                         "second, {} third, {} fourth; written: {}",
                         g_pvCopySeen[0].exchange(0), g_pvCopySeen[1].exchange(0), g_pvCopySeen[2].exchange(0),
                         g_pvCopySeen[3].exchange(0),
                         g_pvCopyMask.load() == 0   ? "all (validated)"
                         : g_pvCopyMask.load() == 1 ? "FIRST only"
                         : g_pvCopyMask.load() == 2 ? "SECOND only"
                         : g_pvCopyMask.load() == 3 ? "THIRD and later only"
                                                    : "NONE");
            if (JitterModeNow() != 0)
                LOG_INFO("DuniaTaa: jitter mode {}: {} frames reported to DLSS, {} structs patched with our own over "
                         "{} changes of value (about the Presents: one new jitter per frame, its copies sharing "
                         "it), {} refused (layout not proven or not sane), {} patched although +704 disagreed with "
                         "the projection (copies built from patched memory; common after a menu), {} frames where "
                         "what DLSS was told is NOT one of the last two jitters written -- refused and that one "
                         "should be zero",
                         JitterModeNow(), g_jitterReported.exchange(0), g_jitterPatched.exchange(0),
                         g_jitterNewFrames.exchange(0), g_jitterUnverified.exchange(0), g_jitterDisagreed.exchange(0),
                         g_jitterMismatch.exchange(0));
            if (JitterModeNow() != 0)
                LOG_INFO("DuniaTaa: jitter refusals by reason: {} not finite, {} viewport too small, {} jitter over 1.5 px "
                         "(largest {:.2f} px), {} layout not yet proven; {} resweep(s) for a missing ring slot",
                         g_jitterRefNotFinite.exchange(0), g_jitterRefViewport.exchange(0),
                         g_jitterRefTooLarge.exchange(0), g_jitterRefMaxPx.exchange(0.0f), g_jitterRefUnproven.exchange(0),
                         g_vcDeadDrops.exchange(0));
            // Read this before the picture: with the patch on, 'patched' and 'decodes with same-eye vectors'
            // should both be about the Presents; anything else in this line is why they are not.
            if (SameEyeWanted())
            {
                auto take = [](TbResult r) { return g_tbCounts[(size_t) r].exchange(0, std::memory_order_relaxed); };
                const uint64_t patched = take(TbResult::Patched), again = take(TbResult::AlreadyDone),
                               verified = take(TbResult::VerifiedOnly), unverified = take(TbResult::Unverified),
                               noChain = take(TbResult::NoChain), noBlock = take(TbResult::NoBlock),
                               noView = take(TbResult::NoView), busy = take(TbResult::Busy);
                auto takePv = [](PvResult r) { return g_pvCounts[(size_t) r].exchange(0, std::memory_order_relaxed); };
                const uint64_t pvPatched = takePv(PvResult::Patched), pvVerified = takePv(PvResult::VerifiedOnly),
                               pvNoChain = takePv(PvResult::NoChain), pvNoView = takePv(PvResult::NoView),
                               pvBusy = takePv(PvResult::Busy), pvIdle = takePv(PvResult::Idle);
                if (IsUpscale())
                    LOG_INFO("DuniaTaa: upscaling: {} copies of DLSS's eye image over the game's upscale dispatch "
                             "(want ~ the Presents; 0 = the dispatch is not learned, see 'upscale DISPATCH learned')",
                             g_upCopies.exchange(0));
                LOG_INFO("DuniaTaa: view constants: {} Presents where the main view could not be told from the "
                         "weapon view (should be 0 outside menus); {} rewrites of a candidate slot holding another "
                         "view (viewport not the scene's) skipped",
                         g_vcAmbiguous.exchange(0), g_pvOtherView.exchange(0));
                LOG_INFO("DuniaTaa: same-eye vectors, per-view previous view: {} submits PATCHED a struct, {} "
                         "verified only, {} no chain, {} no view, {} busy, {} had nothing new; {} structs carried a "
                         "previous view WE had written (copies built from patched memory, common after a menu) "
                         "and had the original put back -- patched should be about the Presents (or a small "
                         "multiple: the game keeps copies)",
                         pvPatched, pvVerified, pvNoChain, pvNoView, pvBusy, pvIdle, g_pvUnpatched.exchange(0));
                LOG_INFO("DuniaTaa: same-eye vectors (patch was {}): {} patched (+{} already done), {} verified "
                         "only, {} UNVERIFIED, {} no chain, {} no block, {} no view, {} busy; {} decodes with "
                         "same-eye vectors; {} pool walks reading {} slots in all; {} seed block(s), {} raw "
                         "quads looked like one",
                         (restart ? g_sameEyeWindowWas.load() : g_sameEyePatchOn.load()) ? "ON" : "off", patched, again, verified, unverified, noChain,
                         noBlock, noView, busy, g_mvSameEyeDecodes.exchange(0), g_tbScans.exchange(0),
                         g_tbScannedSlots.exchange(0), g_viewConstants.TemporalBlockCount(), g_tbObserved.load());
            }
            g_mvStatsAt = now + kMvStatsPresents;
            windowStart = now;
            lastTick = tick;
        }
    }

    // The F8 A/B's numbers, the same shape as F4's and for the same reason: the three states have to be
    // comparable inside one run, and a window that straddles a toggle belongs to neither state.
    if (SpliceTaaMode())
    {
        const uint64_t now = g_presentIndex.load();
        static uint64_t lastTick = 0;
        static uint64_t windowStart = 0;
        static int windowMode = 0;
        const bool restart = g_taaStatsRestart.exchange(false, std::memory_order_relaxed);
        if (g_taaStatsAt == 0)
        {
            g_taaStatsAt = now + kTaaStatsPresents;
            windowStart = now;
            windowMode = g_taaMode.load();
            lastTick = GetTickCount64();
        }
        else if (restart || now >= g_taaStatsAt)
        {
            const uint64_t tick = GetTickCount64();
            const uint64_t ms = tick > lastTick ? tick - lastTick : 1;
            const uint64_t presents = now > windowStart ? now - windowStart : 1;
            static const char* const kStates[] = { "the game's TAA", "TAA skipped", "DLSS spliced" };
            LOG_INFO("DuniaTaa: TAA was {} -- {} Presents in {} ms ({:.1f}/s), {} draws skipped, {} spliced{}",
                     kStates[windowMode < 0 || windowMode > 2 ? 0 : windowMode], presents, ms,
                     (double) presents * 1000.0 / (double) ms, g_taaSkipped.exchange(0),
                     g_taaSpliced.exchange(0), restart ? " (window closed by F8)" : "");
            g_taaStatsAt = now + kTaaStatsPresents;
            windowStart = now;
            windowMode = g_taaMode.load();
            lastTick = tick;
        }
    }

    if (!g_loggedActive)
    {
        g_loggedActive = true;
        LOG_INFO("DuniaTaa: Far Cry 6 TAA host enabled (DLSS on TAA unbind, create at Present)");
    }

    // Why the splice is or is not running. Logged on change only, so it costs nothing per frame but
    // makes "I turned it on and nothing happened" answerable from the log instead of by guesswork.
    {
        const char* reason = SpliceTaaMode()
                                 ? "not used -- splice mode replaces the game's TAA resolve instead"
                             : TonemappedMode()
                                 ? "not used -- tonemapped mode splices at the eye copy instead"
                             : !SplicePathOn() ? "off (NR off and mode not upscaling)"
                             : g_handle == nullptr ? "waiting for the DLSS feature"
                                                   : "running";
        static const char* lastReason = nullptr;
        if (lastReason != reason)
        {
            lastReason = reason;
            LOG_INFO("DuniaTaa: splice {}", reason);
        }
    }

    {
        // Roughly 3 seconds at VR rates before a stop is reported, and half a second of eye rendering
        // before anything is: loading and menu transitions pass through both states briefly.
        const uint64_t now = g_presentIndex.load() + 1;
        const uint64_t lastScene = g_lastEyeScenePresent.load();
        const bool sceneActive = lastScene != 0 && now - lastScene < 45 &&
                                 now - g_firstEyeScenePresent.load() > 45;
        DuniaTaa_Dx12::UpscaleProblem problem = DuniaTaa_Dx12::UpscaleProblem::None;

        const int configured = ConfiguredMode();
        const uint64_t scaledSeen = g_gameplayScaledPresent.load();
        const uint64_t fullSeen = g_gameplayFullPresent.load();
        const bool scaledNow = scaledSeen != 0 && now - scaledSeen < 30;
        const bool fullNow = fullSeen != 0 && now - fullSeen < 30 && now - g_gameplayFullSince.load() > 90;

        if (g_handle != nullptr && configured != 2 && g_createdQuality != configured)
            problem = DuniaTaa_Dx12::UpscaleProblem::RestartForMode;
        // Gameplay drawn at a size the feature was not made for, in either direction.
        else if (g_handle != nullptr && g_createdQuality == 0 && scaledNow)
        {
            g_mismatchW = g_gameplayScaledW.load();
            g_mismatchH = g_gameplayScaledH.load();
            problem = DuniaTaa_Dx12::UpscaleProblem::ScaleChanged;
        }
        else if (IsUpscale() && fullNow)
        {
            g_mismatchW = g_destW;
            g_mismatchH = g_destH;
            problem = DuniaTaa_Dx12::UpscaleProblem::ScaleChanged;
        }
        else if (EffectiveMode() != 0 && sceneActive && SplicePathOn())
        {
            const uint64_t lastUpscale = g_lastUpscalePresent.load();
            const uint64_t mismatch = g_mismatchPresent.load();
            if (configured == 1 && g_handle == nullptr && now - g_firstEyeScenePresent.load() > 270)
                problem = DuniaTaa_Dx12::UpscaleProblem::ScaleNotReduced;
            else if (IsUpscale() && mismatch != 0 && now - mismatch < 90)
                problem = DuniaTaa_Dx12::UpscaleProblem::ScaleChanged;
            // Loading screens and menus also draw the eye at full size without the game's upscale, so this
            // is only a hint in the overlay, not an announcement.
            else if (IsUpscale() && lastUpscale != 0 && now - lastUpscale > 270)
                problem = DuniaTaa_Dx12::UpscaleProblem::UpscaleStopped;
        }

        // The scale checks describe the HDR splice; tonemapped mode works at any Resolution Scale.
        if (TonemappedMode())
            problem = DuniaTaa_Dx12::UpscaleProblem::None;

        const int previous = g_upscaleProblem.exchange((int) problem);
        if (previous != (int) problem && problem != DuniaTaa_Dx12::UpscaleProblem::None)
        {
            g_upscaleEpisode++;
            LOG_WARN("DuniaTaa: upscale problem {} (created {}x{} -> {}x{}, game now renders {}x{})", (int) problem,
                     g_createdInW, g_createdInH, g_createdOutW, g_createdOutH, g_mismatchW.load(), g_mismatchH.load());
        }
    }

    if (Config::Instance()->DuniaTaaTraceFrame.value_or_default())
    {
        Config::Instance()->DuniaTaaTraceFrame = false;
        // F10 also re-arms the upscale writer probe (below 100%, splice mode); it needs no diagnostics.
        g_probeArm.store(true, std::memory_order_relaxed);
        if (DiagnosticsOn())
        {
            g_traceLines = 0;
            g_traceFrom = g_presentIndex.load() + 1;
            LOG_INFO("DuniaTaa trace: armed for presents {} and {} (game scene {}x{}, TAA candidate {}x{})",
                     g_presentIndex.load() + 1, g_presentIndex.load() + 2, g_pinnedW, g_pinnedH, g_taaW, g_taaH);
        }
    }

    {
        std::lock_guard<std::mutex> lock(g_lock);
        ReleaseDeferred(g_presentIndex.load());
    }
    // A frame's depth pointer is only trusted within that frame; a reload can free it right after.
    g_lastSceneDepth = nullptr;

    const bool spliced = g_splicedThisFrame;
    g_presentIndex++;
    g_splicedThisFrame = false;
    g_evalThisFrame = false;
    if (g_handle != nullptr)
        g_allowSplice = true;
    if (spliced && g_handle != nullptr)
        return;

    // Tonemapped mode: one DLAA feature for the finished 8-bit eye image, created once gameplay is drawn and the
    // image and its depth have been seen. The game's G-buffer signature keeps menus from creating it.
    //
    // NR deliberately not required, the same as the evaluate gate and the same as FC5's F6: DLAA is worth having
    // on its own, and requiring NR here is the second half of what made F6 look dead. Removing it from the
    // evaluate alone was not enough -- without a feature nothing can evaluate, so the key still did nothing.
    // Splice mode: one DLAA feature at the resolve's own size and format, created once the resolve has
    // been identified and a pair has been learned from one of its draws. Nothing can be created from a
    // menu by accident: the resolve only draws in gameplay, and the pair only exists because it drew.
    if (SpliceTaaMode())
    {
        if (g_handle != nullptr || !WaitedForKey("the TAA splice"))
            return;

        UINT w = 0, h = 0;
        DXGI_FORMAT fmt = DXGI_FORMAT_UNKNOWN;
        const bool haveShape = g_taaResolve.ColourShape(&w, &h, &fmt);
        if (!haveShape || !GameplayNow())
        {
            static int loggedWait = -1;
            const int reason = !g_taaFound.load(std::memory_order_relaxed) ? 0 : (!haveShape ? 1 : 2);
            if (loggedWait != reason)
            {
                loggedWait = reason;
                LOG_INFO("DuniaTaa: TAA splice: waiting for {}",
                         reason == 0   ? "the game's TAA resolve to be identified"
                         : reason == 1 ? "one of its draws, to learn what it reads and writes"
                                       : "gameplay");
            }
            return;
        }

        // The splice is mid-frame and the game keeps drawing on that list afterwards, while DLSS sets its
        // own root signature, PSO and descriptor heaps. Without this the first suspect for any
        // wrong-looking frame is state corruption, so it is worth saying once rather than rediscovering.
        if (!Config::Instance()->ExtendedStateRestore.value_or_default())
            LOG_WARN("DuniaTaa: TAA splice: [Hotfix] ExtendedStateRestore is off. This splice is mid-frame "
                     "and the game keeps recording on that list afterwards -- set it true, or treat a "
                     "wrong-looking frame as state corruption first");

        ID3D12Device* device = nullptr;
        if (FAILED(queue->GetDevice(IID_PPV_ARGS(&device))) || device == nullptr)
            return;
        device->Release();
        if (!EnsureCommands(device))
            return;
        WaitForGpu(queue);
        if (FAILED(g_allocator->Reset()) || FAILED(g_cmd->Reset(g_allocator, nullptr)))
            return;

        // Below 100% Resolution Scale the feature is render -> eye (upscaling step 2): DLSS's
        // eye-sized output is downsampled onto the resolve's target for the post chain and copied over the
        // game's upscale draw. The first step-2 run still created DLAA because THIS path, not the general
        // one, is what the splice uses -- the log said "created DLAA 2553x2284 -> 2553x2284".
        const bool renderToEye = SceneScaled() && Config::Instance()->DuniaTaaSpliceUpscale.value_or_default();
        const UINT outW = renderToEye ? g_destW : w;
        const UINT outH = renderToEye ? g_destH : h;
        const bool ok = EnsureScratch(device, queue, w, h, outW, outH, fmt) && InitNgx(device) &&
                        CreateDlss(g_cmd, w, h, outW, outH, fmt);
        if (FAILED(g_cmd->Close()) || !ok)
            return;
        ID3D12CommandList* lists[] = { g_cmd };
        queue->ExecuteCommandLists(1, lists);
        queue->Signal(g_fence, ++g_fenceValue);
        if (renderToEye)
            LOG_INFO("DuniaTaa: TAA splice: UPSCALING {}x{} -> {}x{} created -- DLSS makes the eye image from the "
                     "render-size resolve inputs; the game's post chain gets a downsample, its upscale draw is "
                     "overwritten",
                     w, h, outW, outH);
        LOG_INFO("DuniaTaa: TAA splice: DLAA {}x{} fmt {} created to stand in for the game's temporal "
                 "resolve -- press F8 twice to put it in",
                 w, h, (UINT) fmt);
        return;
    }

    if (TonemappedMode())
    {
        if (g_handle != nullptr || !WaitedForKey("tonemapped mode"))
            return;
        const uint64_t now = g_presentIndex.load();
        const uint64_t fullSeen = g_gameplayFullPresent.load();
        const uint64_t scaledSeen = g_gameplayScaledPresent.load();
        const bool gameplay = (fullSeen != 0 && now - fullSeen < 30) || (scaledSeen != 0 && now - scaledSeen < 30);
        const UINT w = g_tonemapW.load();
        const UINT h = g_tonemapH.load();
        ID3D12Resource* eyeDepth = nullptr;
        {
            std::lock_guard<std::mutex> lock(g_lock);
            eyeDepth = g_tonemapDepth[0].depth;
        }
        if (!gameplay || w == 0 || eyeDepth == nullptr)
        {
            static int loggedWait = -1;
            const int reason = !gameplay ? 0 : (w == 0 ? 1 : 2);
            if (loggedWait != reason)
            {
                loggedWait = reason;
                LOG_INFO("DuniaTaa: tonemapped mode: waiting for {}",
                         reason == 0 ? "gameplay" : (reason == 1 ? "the finished eye image" : "its depth"));
            }
            return;
        }

        ID3D12Device* device = nullptr;
        if (FAILED(queue->GetDevice(IID_PPV_ARGS(&device))) || device == nullptr)
            return;
        device->Release();
        if (!EnsureCommands(device))
            return;
        WaitForGpu(queue);
        if (FAILED(g_allocator->Reset()) || FAILED(g_cmd->Reset(g_allocator, nullptr)))
            return;

        const auto format = (DXGI_FORMAT) g_tonemapFormat.load();
        const bool ok = EnsureScratch(device, queue, w, h, w, h, format) && InitNgx(device) &&
                        CreateDlss(g_cmd, w, h, w, h, format);
        if (FAILED(g_cmd->Close()) || !ok)
            return;
        ID3D12CommandList* lists[] = { g_cmd };
        queue->ExecuteCommandLists(1, lists);
        queue->Signal(g_fence, ++g_fenceValue);
        LOG_INFO("DuniaTaa: tonemapped mode: DLAA {}x{} created for NR on the finished eye image", w, h);
        return;
    }

    ID3D12Resource* color = nullptr;
    ID3D12Resource* depth = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_lock);
        color = g_color;
        depth = g_depth;
    }

    if (color == nullptr || depth == nullptr)
    {
        if (!g_loggedSkipMenu)
        {
            g_loggedSkipMenu = true;
            LOG_INFO("DuniaTaa: waiting for a large scene/REAL colour target before DLSS");
        }
        return;
    }

    if (!IsEvalSize(SceneW(), SceneH()))
        return;

    if (g_handle != nullptr)
        return;

    if (!SplicePathOn())
        return;

    if (!WaitedForKey("the upscale splice"))
        return;

    // Automatic mode: decide once gameplay is seen. Nothing is created before then, because a created
    // feature is never rebuilt live and menus render at the full eye size whatever the scale.
    if (ConfiguredMode() == 2 && g_autoMode.load() < 0)
    {
        const uint64_t now = g_presentIndex.load() + 1;
        const uint64_t scaledSeen = g_gameplayScaledPresent.load();
        const uint64_t fullSeen = g_gameplayFullPresent.load();
        const UINT destW = g_destW != 0 ? g_destW : g_pinnedW;
        const UINT destH = g_destH != 0 ? g_destH : g_pinnedH;
        const bool scaled = scaledSeen != 0 && now - scaledSeen < 30 && g_taaW != 0 && g_taaDepth != nullptr &&
                            IsScaledTaa(g_taaW, g_taaH, destW, destH, true);
        const bool full = fullSeen != 0 && now - fullSeen < 30 && now - g_gameplayFullSince.load() > 90;

        if (scaled)
        {
            g_autoMode = 1;
            LOG_INFO("DuniaTaa: automatic mode: gameplay renders {}x{} below the {}x{} eye, using DLSS upscaling",
                     g_taaW, g_taaH, destW, destH);
        }
        else if (full && NrOn())
        {
            g_autoMode = 0;
            LOG_INFO("DuniaTaa: automatic mode: gameplay renders at the full {}x{} eye, using DLAA to carry NR",
                     destW, destH);
        }
        else
        {
            static int loggedWait = -1;
            const int reason = full ? 1 : 0;
            if (loggedWait != reason)
            {
                loggedWait = reason;
                if (full)
                    LOG_INFO("DuniaTaa: automatic mode: full resolution gameplay with NR off, nothing to do");
                else
                    LOG_INFO("DuniaTaa: automatic mode: waiting for gameplay to pick upscaling or DLAA");
            }
            return;
        }
    }

    // Upscaling must wait for the scaled render target. Frames before it is seen would otherwise
    // create DLAA at the eye size, and a created feature is never rebuilt live.
    if (SelectedDuniaQuality() == 1)
    {
        const UINT destW = g_destW != 0 ? g_destW : g_pinnedW;
        const UINT destH = g_destH != 0 ? g_destH : g_pinnedH;
        if (g_taaW == 0 || g_taaDepth == nullptr || !IsScaledTaa(g_taaW, g_taaH, destW, destH, true))
        {
            static bool logged = false;
            if (!logged)
            {
                logged = true;
                LOG_INFO("DuniaTaa: upscaling selected, waiting for a scaled render target (game resolution "
                         "scale below 100 percent)");
            }
            return;
        }
    }

    ID3D12Device* device = nullptr;
    if (FAILED(queue->GetDevice(IID_PPV_ARGS(&device))) || device == nullptr)
        return;
    device->Release();

    if (!EnsureCommands(device))
        return;

    WaitForGpu(queue);
    if (FAILED(g_allocator->Reset()) || FAILED(g_cmd->Reset(g_allocator, nullptr)))
        return;

    ID3D12Device* scratchDevice = nullptr;
    if (FAILED(g_cmd->GetDevice(IID_PPV_ARGS(&scratchDevice))) || scratchDevice == nullptr)
        return;
    scratchDevice->Release();

    UINT inW = g_pinnedW;
    UINT inH = g_pinnedH;
    UINT outW = g_destW != 0 ? g_destW : g_pinnedW;
    UINT outH = g_destH != 0 ? g_destH : g_pinnedH;
    DXGI_FORMAT fmt = g_destW != 0 ? g_destFormat : g_pinnedFormat;
    const bool wantUpscale = SelectedDuniaQuality() == 1 && g_taaW != 0 && g_taaDepth != nullptr &&
                             IsScaledTaa(g_taaW, g_taaH, outW, outH, true);
    if (wantUpscale)
    {
        inW = g_taaW;
        inH = g_taaH;
    }
    // The resolve splice: DLSS reads what the game's TAA resolve read, at the scene size. At 100% it is
    // DLAA and the output is copied over the resolve's target. Below 100% it is render -> eye: the output
    // is downsampled onto the resolve's target for the game's post chain and copied over the game's
    // upscale draw ([Inputs] DuniaTaaSpliceUpscale = false keeps step 1, DLAA at render size).
    if (SpliceTaaMode())
    {
        inW = SceneW();
        inH = SceneH();
        const bool renderToEye = SceneScaled() && Config::Instance()->DuniaTaaSpliceUpscale.value_or_default();
        outW = renderToEye ? g_destW : inW;
        outH = renderToEye ? g_destH : inH;
        fmt = g_pinnedFormat;
    }

    const bool ok = EnsureScratch(scratchDevice, queue, inW, inH, outW, outH, fmt) && InitNgx(scratchDevice) &&
                    CreateDlss(g_cmd, inW, inH, outW, outH, fmt);
    if (FAILED(g_cmd->Close()))
        return;
    if (!ok)
        return;

    ID3D12CommandList* lists[] = { g_cmd };
    queue->ExecuteCommandLists(1, lists);
    queue->Signal(g_fence, ++g_fenceValue);
}
