#pragma once

#include <d3d12.h>
#include <dxgi.h>
#include <cstdint>

struct DuniaTaa_Dx12
{
    // Why DLSS upscaling is not running, for the overlay.
    enum class UpscaleProblem
    {
        None,
        ScaleChanged,     // the game upscales again, but at sizes the feature was not created for
        UpscaleStopped,   // the eye is still rendered but the game no longer upscales (e.g. back to 100%)
        ScaleNotReduced,  // upscaling forced in the ini, but gameplay has never rendered below the eye size
        RestartForMode,   // the forced DLSS mode differs from the one created
    };

    struct UpscaleStatus
    {
        UpscaleProblem problem = UpscaleProblem::None;
        unsigned createdInW = 0, createdInH = 0, createdOutW = 0, createdOutH = 0;
        unsigned seenInW = 0, seenInH = 0;
        uint64_t episode = 0; // changes each time a problem starts, so it is announced once
        int configuredMode = 2; // [Inputs] DuniaTaaQuality: 0 forces DLAA, 1 forces upscaling, 2 automatic
        int activeMode = -1;    // created feature: 0 DLAA, 1 upscaling, -1 none yet
    };

    static UpscaleStatus GetUpscaleStatus();

    // GPU timing of our own passes at the splice ([Inputs] DuniaTaaGpuTiming; overlay "Developer tests"), per 600
    // Presents: motion reconstruction, the DLSS evaluate, the copy/downsample onto the game's target (per-eye means);
    // and how many of those Presents had ResTrack's heap mirroring on (its cost). The last window's, 0 if none.
    struct GpuTimes
    {
        bool on = false;
        double motionMs = 0, dlssMs = 0, copyMs = 0;
        unsigned mirrorPresents = 0, windowPresents = 0;
        uint64_t samples = 0;
    };
    static GpuTimes GetGpuTimes();

    static bool Active();
    // F6 ([Inputs] DuniaTaaStartOnKey, on by default): the first press creates the feature and starts
    // DLAA+NR (a few seconds' freeze), later presses switch between the game's picture and ours.
    static void ToggleStart();
    static bool StartsOnKey();
    static const char* StartName();
    static bool WantHooks();
    static void Note(ID3D12Resource* color, ID3D12Resource* depth);
    static void NoteMotion(ID3D12Resource* motion, bool boundAsRenderTarget,
                           ID3D12GraphicsCommandList* cmd = nullptr);
    // [Inputs] DuniaTaaFindMotion: whether the compute descriptor table hook is worth installing at all,
    // asked once at hook time by D3D12_Hooks.
    static bool WantsMotionSearch();
    // Every compute root descriptor table bind, while the search is running. Reads the table and nothing
    // else -- no barriers, no copies, no references taken. A table bind says nothing about a resource's
    // state, and acting on one as though it did is what hung the game once.
    static void NoteComputeTable(UINT rootParameterIndex, D3D12_GPU_DESCRIPTOR_HANDLE base,
                                 ID3D12GraphicsCommandList* cmd);
    // [Inputs] DuniaTaaRealMotion: whether the Dispatch hook is worth installing, asked once at hook time.
    static bool WantsRealMotion();
    // F4: turns the whole real-motion path on and off live -- the heap mirroring, the copy and the decode --
    // so its cost can be judged inside one run rather than across two.
    static void ToggleRealMotion();
    static const char* RealMotionName();
    // After the original Dispatch. Copies the motion texture this list just wrote, and nothing else: a copy
    // and barriers change no pipeline state, which is why this is safe to inject into the game's list where
    // a compute pass of ours would not be. The decode happens later, at the splice, from our own copy.
    static void AfterDispatch(ID3D12GraphicsCommandList* cmd, ID3D12PipelineState* pipeline);
    // Before every ExecuteCommandLists, from the queue hook in ResTrack: the per-view struct previous view is
    // rewritten to the same eye here ([Inputs] DuniaTaaSameEyeVectors). Returns at once otherwise.
    static void BeforeExecute();
    // The DLSS render preset, switchable live from the overlay. 0 = the driver's default, 1 = A ... 13 = M.
    // It does NOT tear anything down itself: it asks OptiScaler's own staged feature change (the path the stock
    // menu's Apply uses) to recreate this module's two features over their next three evaluates.
    // Jitter: 0 DLSS told zero (old), 1 told the game's, 2 our own per-eye Halton patched in and told. Live.
    // Test: which of a frame's copies of the per-view struct get the same-eye previous view (0 all ... 4 none).
    // Days Gone's object composition for the other-eye branch (F4 x2), live: residual scale, knee floor,
    // speed term on the floor, chained two-hop lookup for fast movers.
    static void GetObjectComposition(float* scale, float* floor, float* speedTerm, bool* fastChain);
    // `log`: false while a slider is being dragged (applied live, logged once on release).
    static void SetObjectComposition(float scale, float floor, float speedTerm, bool fastChain, bool log = true);
    // The residual view: paints where the object branch adds motion in place of the picture (live).
    static bool ResidualView();
    // Object residual only where the pixel is the nearest surface around it (silhouette noise); live, default on.
    // The private-copy feasibility probe (overlay button): one window, then a verdict in the log.
    static void ArmPrivateCopyProbe();
    // For the probe's table-rewrite question: every root signature's table layout, from
    // CreateRootSignature (identity only, never AddRef'd); the graphics root signature each list sets (only while
    // the probe runs); and, FROM STARTUP, which slots of each shader-visible CBV/SRV/UAV heap the game writes --
    // where a rewritten table could live without the game ever overwriting it.
    static void NoteRootSignature(ID3D12RootSignature* rootSignature, const D3D12_VERSIONED_ROOT_SIGNATURE_DESC* desc);
    static void NoteGraphicsRootSignature(ID3D12GraphicsCommandList* cmd, ID3D12RootSignature* rootSignature);
    static void NoteShaderVisibleHeap(ID3D12DescriptorHeap* heap, SIZE_T cpuStart, SIZE_T gpuStart, UINT numDescriptors,
                                      UINT increment);
    // srcHandle: the CPU descriptor copied from (0 for a view created in place).
    static void NoteDescriptorWrite(SIZE_T cpuHandle, UINT count, SIZE_T srcHandle = 0);
    // The private copy (overlay, live, off by default): at the G-buffer's per-view table binds, returns a rebuilt
    // table whose per-view CBV reads a same-eye mirror of the struct; otherwise `base` unchanged.
    static D3D12_GPU_DESCRIPTOR_HANDLE RedirectGraphicsTable(ID3D12GraphicsCommandList* cmd, UINT root,
                                                             D3D12_GPU_DESCRIPTOR_HANDLE base);
    static bool PrivateCopyOn();
    // Whether OMSetRenderTargets should resolve every slot for NoteAllTargets (private copy or probe running).
    static bool WantsAllTargets();
    // The conversion probe: transitions of the velocity texture (the motion pass's t3) and its depth (t1).
    static void NoteBarrierFacts(ID3D12GraphicsCommandList* cmd, UINT count, const D3D12_RESOURCE_BARRIER* barriers);
    static void SetPrivateCopyOn(bool on);
    // Which graphics lists it redirects: 0 motion target at slot 0, 1 at ANY slot (with depth), 2 every one.
    static int PrivateCopyScope();
    static void SetPrivateCopyScope(int scope);
    // Every render target of an OMSetRenderTargets, resolved (slot 0 alone hides a G-buffer's later slots).
    static void NoteAllTargets(ID3D12GraphicsCommandList* cmd, ID3D12Resource* const* targets, UINT count,
                               ID3D12Resource* depth);
    static const char* PrivateCopyProbeState();
    // Buffer-SRV recording in ResTrack's view hooks (diagnostics or the probe). CBVs are recorded whenever
    // WantsCbvTracking() is true -- from startup, so descriptors created at load are known.
    static bool CbvRecordingOn();
    static bool ObjectNearOnly();
    static void SetObjectNearOnly(bool on);
    static bool ObjectBgReject();
    static void SetObjectBgReject(bool on);
    static int ObjectRadius();
    static void SetObjectRadius(int px);
    static void SetResidualView(bool on);
    static int SameEyeCopies();
    static void SetSameEyeCopies(int mask);
    static int JitterMode();
    static void SetJitterMode(int mode);
    static int JitterSign();
    // Live A/B for the halo regression: 1 = the other-eye baseline as it was before the jitter work.
    static int BaselineOld();
    static void SetBaselineOld(int old);
    static void SetJitterSign(int variant);
    static int DlssPreset();
    static void RequestDlssPreset(int preset);

    // ---- the game's own TAA resolve ([Inputs] DuniaTaaFindTaa) ----------------------------------
    // Whether the graphics table hook and the draw hooks are worth installing, asked once at hook time.
    static bool WantsTaaSearch();
    // Whether ResTrack INSTALLS its CreateConstantBufferView hook (asked once, at hook time): FC6 always, since a
    // probe armed mid-session could not install it and would read as "no CBVs". It also RECORDS whenever this is
    // true, from startup: the game creates its descriptors at load, and a probe that only records during its own
    // window sees none of them (two probes once read 0 per-view hits, 28M+ empty slots).
    static bool WantsCbvTracking();
    // Whether ID3D12Resource::Map must be hooked so the per-view constants can be read from a CPU mapping.
    // WIDER than WantsCbvTracking: matrix motion needs the mappings in everyday play, with diagnostics off.
    // Gating this on diagnostics is what would have made the whole matrix path work only in the mode it was
    // developed in -- the same shape of mistake that took F7's capture out.
    static bool WantsBufferMappings();
    // The buffer a cached per-view pointer lives in has died. Called from ConstantCapture_Dx12::ForgetResource,
    // which ResTrack's Release hook drives on the last release of every resource.
    //
    // The candidates are RAW CPU POINTERS into memory the game owns. A menu is a size change, a size change is
    // when Dunia frees and reallocates, and the pointers outlive the resource -- Map is reference counted and
    // does NOT keep it alive. That is the FC6 bug where matrix motion broke after a menu and F4 then crashed.
    // The invalidation has to happen on the resource's DEATH: checking whether the contents still look right
    // means reading a freed page, and that read is the crash.
    // The range is the dead mapping's CPU range, and it is a filter, not decoration: dropping the candidates
    // on every released buffer would re-run the fingerprint sweep over every mapping the game has -- a byte
    // scan of megabytes -- for buffers that never held a candidate. Only a death that actually takes one of
    // the cached pointers with it costs anything.
    static void ForgetViewConstants(const uint8_t* begin, const uint8_t* end);
    // A resource is about to die (ResTrack_Dx12::hkRelease, before the final release). If it is one of the
    // learned motion textures the pair is forgotten and learned again -- see g_mvByParity. Never AddRef'd.
    static void ForgetResource(ID3D12Resource* resource);
    // Every graphics root descriptor table bind, while the resolve is still being looked for. Reads the
    // table and nothing else.
    static void NoteGraphicsTable(UINT rootParameterIndex, D3D12_GPU_DESCRIPTOR_HANDLE base,
                                  ID3D12GraphicsCommandList* cmd);
    // Which pipeline each list last set, from the SetPipelineState hook: a table bind does not say what
    // the draw will use, so the pipeline is learned at the draw that follows the matching bind.
    static void NotePipeline(ID3D12GraphicsCommandList* cmd, ID3D12PipelineState* pipeline);
    // True when this draw is the game's TAA resolve and the skip is on, in which case the caller must not
    // call the original. The gate for step 2: if the picture does not change, the pass was not the one.
    static bool ShouldSkipDraw(ID3D12GraphicsCommandList* cmd);
    // F8: cycles the game's own TAA / its draw skipped / DLSS spliced in its place. Never F5 -- that is
    // Far Cry 6's own photo mode. The third state exists only with [Inputs] DuniaTaaSpliceTaa.
    static void ToggleSkipTaa();
    static const char* SkipTaaName();
    // Whether the Dispatch hook needs the list's current pipeline resolved. True only while real motion is
    // running, so the hook pays for the lookup in no other configuration.
    static bool WantsDispatchPipeline();
    static void OnSetTargets(ID3D12GraphicsCommandList* cmd, ID3D12Resource* color, ID3D12Resource* depth);
    static bool PendingSplice(ID3D12GraphicsCommandList* cmd);
    static void AfterOMSet(ID3D12GraphicsCommandList* cmd);
    static void AfterBound(ID3D12GraphicsCommandList* cmd);
    static void BeforeCopy(ID3D12GraphicsCommandList* cmd, ID3D12Resource* dst, ID3D12Resource* src);
    static void AfterCopy(ID3D12GraphicsCommandList* cmd, ID3D12Resource* dst, ID3D12Resource* src);
    // Diagnostics only: counts draws per command list while a frame trace runs.
    static void OnDraw(ID3D12GraphicsCommandList* cmd);
    // After a draw was recorded (not skipped): copies DLSS's eye image over the game's upscale target below 100%.
    static void AfterDraw(ID3D12GraphicsCommandList* cmd);
    static void OnPresent(ID3D12CommandQueue* queue, IDXGISwapChain* swapChain);

    // ---- upscale writer probe (below 100%, splice mode) --------------------------------------------
    // One bounded window that logs, once per kind, EVERY way any eye-sized texture (any format, not ours) is
    // touched: OMSet at any RTV slot with or without depth, graphics and compute table SRV/UAV at any slot, copy
    // and resolve source/destination, clears, and every transition barrier. Five builds guessed the game's
    // upscale writer and every one was wrong; this replaces the guess with the set of facts. Auto-armed once,
    // when the upscaling feature's resolve splice first runs in gameplay; F10 re-arms it.
    // True only inside the window: every hook below gates on this one atomic load.
    static bool ProbeOn();
    // Every OMSetRenderTargets: all RTV slots resolved (a writer bound at slot 1+ was invisible to OnSetTargets,
    // which takes slot 0 only), plus how many slots could not be resolved -- an unresolved slot is a blind spot
    // and the summary says so. Also resets the list's draws-since-bind counter.
    static void ProbeOmSet(ID3D12GraphicsCommandList* cmd, ID3D12Resource* const* rtvs, UINT numRtvs,
                           UINT unresolved, ID3D12Resource* depth);
    // Every ResourceBarrier (the hook exists only with real motion or a capture): the game must declare the
    // state it writes a texture in, so the transition INTO that state names the writer's route.
    static void ProbeBarriers(ID3D12GraphicsCommandList* cmd, UINT count, const D3D12_RESOURCE_BARRIER* barriers);
    static void ProbeResolve(ID3D12GraphicsCommandList* cmd, ID3D12Resource* dst, ID3D12Resource* src);
    static void ProbeClear(ID3D12GraphicsCommandList* cmd, ID3D12Resource* resource, bool uav);
    // Every ExecuteCommandLists: the submit order goes into the event sequence, so recording order (Dunia
    // records in parallel) can be read as execution order.
    static void NoteExecute(ID3D12CommandList* const* lists, UINT count);
};
