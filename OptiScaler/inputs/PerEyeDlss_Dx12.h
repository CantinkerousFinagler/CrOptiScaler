#pragma once

#include <string>

struct ID3D12CommandQueue;
struct ID3D12Device;
struct ID3D12GraphicsCommandList;
struct NVSDK_NGX_Parameter;

// Per-eye DLSS on OptiScaler's own D3D12 feature path, for a game that calls DLSS (SR or ray reconstruction) with ONE
// feature under a VR mod that renders one eye per Present. First user: Crimson Desert (Streamline). The D3D12 port of
// PerEyeDlss_Dx11: the twin feature, the routing, and same-eye motion fitted from the game's own vectors.
//
// At the first per-eye evaluate (F6; or at the game's create with UpscalerTwinAtCreate), a twin is created from the
// game's creation parameters (SR or RR, the same backend). At
// evaluate, odd Presents go to the twin while routing is per eye, so each eye keeps a history of its own. Routing
// starts SHARED -- the game's picture untouched -- and F6 switches (live; both histories reset once). The twin, the
// crops and the peripheries are each created live at their first need; all are released with the game's feature.
//
// With [Inputs] UpscalerInputProbe as well: a small compute pass on the evaluate's list sums the game's motion
// vectors over a 64x64 grid (our own root signature, heap and buffers; the vectors are read as DLSS reads them, no
// barrier on any game resource), read back 8 evaluates later and validated by a stamp. Per 600 Presents the log gives
// the mean motion x per Present parity -- opposite signs with the head still = the vectors point at the other eye.
namespace PerEyeDlss_Dx12
{
// [Inputs] UpscalerPerEye (startup only, default on). Shared with the D3D11 module; each acts on its own API.
bool Enabled();

// Every Present; the queue gives the GPU timestamp frequency for the probe's timing.
void OnPresent(ID3D12CommandQueue* queue);

// Feature lifetime, from NVSDK_NGX_D3D12_CreateFeature / ReleaseFeature.
void RegisterTwin(unsigned int primary, unsigned int twin);
unsigned int TwinOf(unsigned int primary); // 0 if none
void Forget(unsigned int primary);

// Which feature this evaluate goes to (the game's or its twin). May set the reset flag once after a routing switch.
unsigned int Route(unsigned int handle, NVSDK_NGX_Parameter* params);

// Before the upscaler's evaluate, on its list. Per eye: fits this Present's camera map from the game's vectors and
// depth and, chained with the previous Present's fit, hands DLSS same-eye motion in place of the game's vectors (or
// resets this eye's history when there is nothing to chain). With the probe: the motion statistics of what DLSS gets.
// EndEvaluate, after the upscaler, hands the game's texture back to its parameters.
void BeforeEvaluate(ID3D12GraphicsCommandList* cmdList, NVSDK_NGX_Parameter* params);
void EndEvaluate(ID3D12GraphicsCommandList* cmdList, NVSDK_NGX_Parameter* params);
// Puts the game's own motion vectors (and its bias mask) back into the parameters. Called after NR's run over the
// upscaled picture, so NR reads our same-eye vectors too (it used to be part of EndEvaluate).
void RestoreInputs(NVSDK_NGX_Parameter* params);

// With [Inputs] UpscalerInputProbe, per 600 Presents: the GPU time of the upscaler's evaluate and of our same-eye
// passes, per routing mode and eye (timestamps on the evaluate's list).

// F6.
void ToggleRouting();
bool RoutingOn();

// ---- Foveal crop (foveated DLSS, step 2) ----
// A second DLSS feature per eye (created at the first F4) upscales a centred crop -- [Inputs] UpscalerFoveaCrop of the
// eye's width and height, an exactly aligned size nearby preferred -- at its own preset ([Inputs] UpscalerFoveaPreset,
// 13 = M), reading the same inputs through DLSS's input subrects, into our texture; COMPOSITE blends it over the crop
// box of the game's output with a [Inputs] UpscalerFoveaFeather px edge. F4 on/off, F8 outline. The full pass still
// runs, so this step saves nothing: it shows the crop and the seam.
struct FoveaSize
{
    unsigned int renderW = 0, renderH = 0, outW = 0, outH = 0;
    int preset = 0;
};
bool FoveaOn();
// The crop's box for the eye of THIS Present (its parity) while the crop is on: centre and size as fractions of the
// eye (the same in render and output: the crop keeps the ratio), with the gaze, the size sliders and the offsets in
// it. The NR fovea takes it ([DlssNr] FoveaFollowSr). It is the last box planned for this parity, at most 4 Presents
// old: NR after SR gets this evaluate's, NR before SR (run before the crop is planned) the same eye's previous one.
// Returns 0 with a box, else why not: 1 the crop is off or failed, 2 no box for this eye lately, 3 the lock was busy
// (one frame, an unexpected nesting). Never blocks (try-lock).
int FoveaBoxForNr(float& cx, float& cy, float& w, float& h);
void ToggleFovea();        // F4
void ToggleFoveaOutline(); // F8
const char* FoveaName();
// The preset a feature being created now must use (0 = none): set while a crop feature is created or evaluated, read by
// DLSSFeatureDx12's create after OptiScaler's own preset overrides.
int PresetForCreate();
void SetPresetForCreate(int preset);
// What is being created now (0 = nothing of ours, 1 = a crop, 2 = a periphery), and the RR preset a ray-reconstruction
// feature of ours was made with: DLSSDFeatureDx12's create reads the hint back from the block after every override
// (0 = default, the DLL chooses). Atomics: the create runs inside our evaluate, never take our lock there.
int KindForCreate();
void SetKindForCreate(int kind); // CreateFoveaCrop: around the create itself (BeginFovea sets it for rebuilds)
void NoteRrCreate(int kind, int preset);
// An RR crop's feature made direct on the snippet DLL (DLSSDFeatureDx12, [Inputs] UpscalerFoveaRrDirect): how the
// create went, and per window how many crop evaluates went that way or through the NGX entry. Atomics, as above.
void NoteCropDirectCreate(bool ok, int result);
void NoteCropDirectEvaluate(bool ok);
void NoteCropCoreEvaluate();
// The same for an SR crop (DLSSFeatureDx12, [Inputs] UpscalerFoveaSrDirect, on nvngx_dlss.dll): its own counters; the
// NGX-entry count is the SR crop's evaluates only (the RR one above counts every RR feature's).
void NoteSrCropDirectCreate(bool ok, int result);
void NoteSrCropDirectEvaluate(bool ok);
void NoteSrCropCoreEvaluate();
// The crop's own parameter block, before its create / evaluate: marked as a crop of ours. Never the game's block.
void MarkCropBlock(NVSDK_NGX_Parameter* cropParams);
// Around the full-size evaluate of a foveated frame (the game's pass or the cheap periphery): which vectors the block
// held before it and holds after it. Something below us may change them; the crop can take those
// ([Inputs] UpscalerFoveaRrVectors / UpscalerFoveaSrVectors = 1). Identity only, used within the same evaluate, never
// AddRef'd.
void MainVectorsBefore(NVSDK_NGX_Parameter* params);
void MainVectorsAfter(NVSDK_NGX_Parameter* params, bool evaluated);
// After the cheap periphery's evaluate on ITS OWN block (needed under alternate-eye rendering: a periphery on the
// game's block had its evaluates fail): the vectors the pass below us left there, and their scale, put into the game's
// block for the crop's copy and NR after SR; RestoreInputs
// puts the game's own back. Nothing to hand down = nothing done. Counted per window.
void HandBelowVectorsToGame(NVSDK_NGX_Parameter* game);
unsigned int FoveaCropOf(unsigned int handle); // 0 if none (this Present's eye in shared routing)
unsigned int ForgetFoveaCrop(unsigned int handle); // the crop feature to release, 0 if none
// Every crop and periphery feature made for this owner (both eyes' in shared routing), forgotten; the ids to release.
std::vector<unsigned int> ForgetFoveaFeatures(unsigned int owner);
// After the full pass: the crop for this evaluate (false = none; each reason logged once, counted per window). A handle
// without a crop feature gets one made from `create` (FoveaCreated / FoveaCreateFailed).
bool PlanFovea(ID3D12GraphicsCommandList* cmdList, NVSDK_NGX_Parameter* params, unsigned int handle, FoveaSize* create);
void FoveaCreated(unsigned int handle, unsigned int crop);
void FoveaCreateFailed();
// The game's feature is not ready to make a crop from (OptiScaler is rebuilding it): no crop this evaluate, asked again
// at the next one. Never final. Counted with the window's skipped crops, said once.
void FoveaNotReady();
// The crop is made and evaluated on a parameter block of ITS OWN, never on the game's: a key the game never set cannot
// be put back as absent (an NGX block has no erase), and Hogwarts' full pass, handed back an explicit render subrect,
// did crop-sized work for the rest of the session. Copies every key DLSS / RR reads that the game's block holds into
// `own`; false (logged once) if the game's block holds no colour or output.
bool CopyCropParameters(NVSDK_NGX_Parameter* game, NVSDK_NGX_Parameter* own);
// Around the crop feature's evaluate: its parameters in, then the game's back and the composite.
// A ray reconstruction crop (`rayReconstruction`) also gets copies of RR's guides and the crop's own projection.
bool BeginFovea(ID3D12GraphicsCommandList* cmdList, NVSDK_NGX_Parameter* params, unsigned int crop,
                bool rayReconstruction);
// `rebuilding`: the crop's staged rebuild is still under way (it did not evaluate), so nothing is drawn.
void EndFovea(ID3D12GraphicsCommandList* cmdList, NVSDK_NGX_Parameter* params, bool evaluated, bool rebuilding);
// Overlay: a new crop preset, applied by rebuilding only the crop features live (OptiScaler's staged rebuild).
void RequestFoveaPreset(int preset);
// The crop direct on the snippet DLL switched in the overlay ([Inputs] UpscalerFoveaSrDirect / UpscalerFoveaRrDirect): the
// crop features are rebuilt live, as for a preset change (F4 off / on keeps the crop's feature).
void RequestFoveaDirect(bool rr, bool on);
// The cheap DLSS periphery as one feature for both eyes in shared routing ([Inputs] UpscalerPeripheryOneFeature: 0 auto,
// 1 always, 2 never), live: every periphery's history restarts on a change. Now: what the periphery runs as (the menu).
void RequestPeripheryOneFeature(int mode);
// VRAM: this process's use of the adapter's local memory in MB (DXGI QueryVideoMemoryInfo; 0 if unknown), and what
// each feature create took (DLSSFeatureDx12::InitDLSS, the NR model): logged per create, totals per kind in the window
// summary (one D3D12 game once showed +8.8 GB in the second the crop came on, owner unknown).
uint64_t VramUsedMB(ID3D12Device* device);
void LabelNextCreate(const char* label); // a name for the next create of kind 0 (the twin); nullptr clears it
void ReportCreateVram(NVSDK_NGX_Parameter* params, uint64_t beforeMB, uint64_t afterMB, bool direct);
void ReportNrCreateVram(unsigned int w, unsigned int h, uint64_t beforeMB, uint64_t afterMB);
bool PeripheryOneFeatureNow();
struct PerEyeStatus
{
    bool routing = false, fovea = false, outline = false, failed = false, exact = false;
    int preset = 0;
    // Ray reconstruction: the crop's last evaluate was an RR one, the RR preset its last RR feature was made with
    // (-1: none made yet, 0: default = the DLL chooses).
    bool cropRr = false;
    int rrVectors = 0;  // 0 ours, 1 the ones left in the block below us by the full-size pass
    int srVectors = -1; // the same for an SR crop (UpscalerFoveaSrVectors: -1 auto)
    bool srFromBelowNow = false; // what the SR crop takes now (auto resolved)
    bool belowSeen = false;      // DL(S)SS is seen at work on the full-size pass (the vote of the last evaluates)
    int rrCropPreset = -1;
    unsigned int cropW = 0, cropH = 0, cropOutW = 0, cropOutH = 0;
    float cropPercent = 0;
    double upscalerMs = 0, cropMs = 0; // the last 600-Present window (probe timing), 0 if none
    float size = 0, sizeH = 0, offsetX = 0, offsetY = 0; // crop width / height fractions, offsets
    float offsetSideways = 0;                            // the same for both eyes, + = right
    int feather = 0, outerPreset = 0;
    bool gazeBoxWhenStill = false; // the green gaze box with the outline also while the crop does not follow
    bool copyInputs = true;
    // What the crop's last evaluate read (FoveaCopyRule.h), whether the switch is set by hand, and whether something
    // below us was seen to change copied vectors this launch.
    bool copyInputsNow = true, copyInputsByHand = false, copiesSwappedBelow = false;
    bool giveUpStale = true; // the same-eye fit gives up a stale start (UpscalerFitGiveUpStale)
    bool fitRestart = true;  // the same-eye fit restarts within the fit (UpscalerFitRestart)
    bool fitLeaveOutNearest = false; // the same-eye fit leaves out what lies nearer than a depth gap (a car)
    uint32_t dumpLeft = 0, dumpSaved = 0, dumpLost = 0; // the fit's sample dump (RequestFitDump)
    int objectMotion = 0; // UpscalerSameEyeObjectMotion: 0 camera only, 1 moving things (knee 0.1-0.4 px), 2 all
                          // (0.02-0.1), 3 small movements (residual averaged 3x3, knee 0.03-0.15)
    int biasMask = 0;     // UpscalerSameEyeBiasMask: 0 off, 1 moving things, 2 small movements (DLSS's current-colour mask)
    bool vectorsSameEye = false; // UpscalerVectorsSameEye: no same-eye work of ours (DL(S)SS on)
    int fitDepth = 0;        // UpscalerFitDepth: 0 Auto, 1 Hardware, 2 Linear (1/depth)
    bool periphery = false, peripheryFailed = false, peripherySr = false;
    bool peripheryTaa = false, taaFailed = false, taaFlipJitter = false;
    float taaBlend = 0.1f;
    float taaScale = 1.0f;
    int peripheryPreset = 5;
    int leftParity = -1; // which Present parity is the left eye (-1: not known yet -- needs per eye)
    unsigned int cropX = 0, cropY = 0, eyeW = 0, eyeH = 0; // this crop's output box origin, and the eye size
    // Eye tracking ([Inputs] UpscalerFoveaGaze, VarjoGaze.h): step 1 only marks the gaze with the outline (F8).
    bool gazeOn = false, gazePlaced = false, gazeFlipX = false, gazeFlipY = false;
    std::string gazeText;
    unsigned int gazeX = 0, gazeY = 0;
    int gazeEye = -1; // 0 left, 1 right, 2 both
    bool follow = false; // step 2: the crop follows the gaze
    float deadZone = 0.5f;
    float glideMs = 0.0f; // the crop glides after the gaze with this time constant (0: one jump)
    int ageFrames = 8;    // the crop fades in over the periphery where its picture is younger than this (0: off)
    uint64_t followMoves = 0, followShifts = 0, followResets = 0, followResetsNoCopy = 0;
};
PerEyeStatus GetStatus();
// Overlay, "Foveated SR": the full pass's preset (the game's features and their twins rebuild live; the crops keep
// theirs), the crop's size (fraction of the eye; the crops rebuild live at it), its offset (x toward the nose, mirrored
// per eye when per eye is on; y up; fractions of the eye, applied at once, each move restarts the crop's history) and
// the blended edge.
void RequestOuterPreset(int preset);
void RequestFoveaSize(float width, float height); // fractions of the eye's width and height
void RequestFoveaOffset(float x, float y, float sideways); // toward the nose (mirrored), up, sideways (both eyes)
void RequestFoveaFeather(int px);
// The crop reads copies of its box (default) or the full inputs through DLSS's input subrects (K and J pixelated that way).
void RequestFoveaCopyInputs(bool on);
void RequestGazeBoxWhenStill(bool on); // session only
void RequestFoveaCropMask(int mode); // the game's bias mask at an SR crop: 0 none, 1 copied at the box, 2 as handed
// Per eye: the same-eye fit gives up a stale start past two rejections in a row (default), or keeps it (as before).
void RequestGiveUpStale(bool on);
// Per eye: the same-eye fit restarts from no motion within the fit when round 0 finds the eye's last fit left behind
// (default), or not (as before: the eye's last fit handed on until two fits in a row were rejected).
void RequestFitRestart(bool on);
// Per eye: the same-eye fit leaves out what lies nearer than a gap in its depth histogram (a car around the camera),
// or fits every sample (default; as it used to be).
void RequestFitLeaveOutNearest(bool on);
// Per eye: save the next 16 same-eye fits' samples (raw depth and motion at the fit's 64x64 grid, and the verdict) to
// OptiFitDump\<time>\ beside the exe, for offline analysis.
void RequestFitDump();
// Per eye: what the compose adds beyond the fitted camera motion: 0 nothing (default), 1 the game's
// object motion x2 through FC5's 0.1-0.4 px knee (moving things only), 2 the same through the 0.02-0.1 px knee (as
// before).
void RequestSameEyeObjectMotion(int mode);
// Per eye: DLSS's current-colour bias mask from the compose's residual: 0 off, 1 moving things, 2 small movements.
void RequestSameEyeBiasMask(int mode);
// Per eye under R.E.A.L. VR's DL(S)SS: no fit or compose of ours.
void RequestVectorsSameEye(bool on);
// Per eye: the depth the same-eye fit takes: 0 Auto (RR's DLSS.Use.HW.Depth), 1 Hardware, 2 Linear (1/depth).
void RequestFitDepth(int mode);

// ---- Cheap periphery (foveated DLSS, step 3) ----
// With the crop on: a DLSS feature per eye at the render size in and out ([Inputs] UpscalerFoveaPeripheryPreset, 5 = E)
// into our texture, ENLARGEd (Catmull-Rom) over the whole eye, in place of the game's full pass; the crop is drawn over
// it as before. A frame it cannot run (rebuild, failure) takes the full pass, and a game feature skipped meanwhile starts
// its history over (TakeMainSkipped). [Inputs] UpscalerFoveaCheapPeriphery, overlay.
bool PeripheryOn();
unsigned int PeripheryOf(unsigned int handle);
unsigned int ForgetPeriphery(unsigned int handle); // the periphery feature to release, 0 if none
bool PlanPeriphery(unsigned int handle, FoveaSize* create); // after PlanFovea for the same evaluate
void PeripheryCreated(unsigned int handle, unsigned int periph);
void PeripheryCreateFailed();
// As FoveaNotReady, for the periphery: the full pass runs this evaluate (counted with the window's fall-backs).
void PeripheryNotReady();
bool BeginPeriphery(ID3D12GraphicsCommandList* cmdList, NVSDK_NGX_Parameter* params, unsigned int periph,
                    unsigned int owner);
bool EndPeriphery(ID3D12GraphicsCommandList* cmdList, NVSDK_NGX_Parameter* params, bool evaluated, bool rebuilding);
bool TakeMainSkipped(unsigned int handle);
void RequestPeriphery(bool on);
void RequestPeripheryPreset(int preset);
// RR mode: the periphery as an SR feature over RR's inputs (SR does not denoise: the periphery shows the raw ray traced
// noise RR would remove -- the experiment is whether peripheral vision minds). [Inputs] UpscalerFoveaPeripherySr.
bool PeripherySr();
bool IsPeriphery(unsigned int handle);
void RequestPeripherySr(bool on);

// ---- Own TAA periphery (foveated DLSS, step 4) ----
// With the cheap periphery on: our own TAA pass at the output size over the whole eye (the crop drawn over it after) in
// place of the periphery's DLSS call and ENLARGE -- no DLSS call outside the crop. Its history is ours, two textures per
// owner feature. It follows the routing (F6): per eye, one history per eye with same-eye motion (two Presents back);
// shared, one history for both eyes moved by the game's vectors (one Present back, the other eye), as the game's shared
// DLSS is. An input it cannot read (logged once, counted per window) falls back to the DLSS periphery.
// [Inputs] UpscalerFoveaPeripheryTaa / UpscalerFoveaTaaBlend / UpscalerFoveaTaaFlipJitter, all live in the overlay.
bool PeripheryTaa(); // on and not failed
// True = drawn (the full pass and the DLSS periphery are skipped for this evaluate).
bool TaaPeriphery(ID3D12GraphicsCommandList* cmdList, NVSDK_NGX_Parameter* params, unsigned int owner);
void RequestPeripheryTaa(bool on);
void RequestTaaBlend(float blend);
void RequestTaaScale(float scale); // [Inputs] UpscalerFoveaTaaScale: the TAA's size as a fraction of the eye, enlarged after
void RequestTaaFlipJitter(bool on);
// A feature this module asked to rebuild must come back as DLSS (OptiScaler resets its global new-backend choice after
// every finished rebuild; its D3D12 default is XeSS). True while it is ours; forgotten once `stillChanging` is false.
bool TakeDlssRebuild(unsigned int handle, bool stillChanging);
// Right after the full pass (timing).
void AfterUpscale(ID3D12GraphicsCommandList* cmdList);
} // namespace PerEyeDlss_Dx12
