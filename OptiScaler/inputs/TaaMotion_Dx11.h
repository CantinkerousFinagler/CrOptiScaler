#pragma once

// Diagnostics ([Inputs] TaaMotionCheck, startup only): decodes the motion vectors a D3D11 game's TAA compute pass
// reads into DLSS form, to check them in game before DLSS is fed with them. The pass is the immediate-context
// dispatch writing an eye-sized 10-bit image (UAV 0) from packed R32_UINT motion (t1) and RGBA16F colour (t2), as
// in Days Gone's tonemap + TAA. Just before it runs, a small compute pass unpacks t1 (X bits 0-13 over 16383, Y
// bits 14-26 over 8191, (value - 0.5) * 0.25 in UV towards the previous frame) into R16G16_FLOAT pixel offsets,
// previous minus current (a zero word, nothing written, becomes zero).
//
// With the game's Render Scale below 100% the pass is a TAA-only one at render resolution instead (the game upscales
// its output in a later draw): packed motion in t0, colour in t1 (both a few texels larger than the output), the
// 10-bit YCoCg history in t2 and the render-size output in u0. The same decode runs there, scaled by the output's
// size; RenderResolution() tells the two apart. Each size keeps its own motion textures, so switching back and forth
// (menus draw at eye size) costs nothing after the first time.
//
// Under alternate-eye VR the previous frame is the other eye, and these offsets include the eye parallax. So the
// same pass also composes them with the previous Present's decoded motion, read where this Present's points: the
// offset to two Presents back, the same eye, with the parallax cancelled. Without a decode on the previous Present
// the composed motion is just this Present's.
//
// F8 cycles a view that overwrites the pass's output: off, decoded, composed (the scene in grey, moving pixels
// tinted red/green by their X/Y offset; eye-size pass only). The F7 capture saves "decoded motion", "composed motion" and "output 2
// presents ago", a copy of the pass's output from two Presents back (kept only while a capture is armed).
namespace TaaMotion_Dx11
{
// [Inputs] TaaMotionCheck or TaaDlaa.
bool Enabled();

// True while this module's own passes run: the D3D11 hooks pass those calls straight through.
bool InPass();

// Immediate context, around the game's Dispatch.
void BeforeDispatch(ID3D11DeviceContext* ctx);
void AfterDispatch(ID3D11DeviceContext* ctx);

// Before the real Present.
void OnPresent();
// Presents so far (the frame being recorded until the next OnPresent).
uint64_t PresentIndex();
// This Present's copy of UE4's view uniform buffer (the base pass's vs cb1, 2048 bytes), or nullptr if none was taken
// yet. Ours, not the game's; not AddRef'd.
ID3D11Buffer* ViewBuffer();

// True when this Dispatch is the pass and it is the render-resolution TAA-only layout.
bool RenderResolution();

// This pass's decoded / composed motion, or nullptr outside the pass (composed: also when there was no previous
// Present's motion). Not AddRef'd.
ID3D11Texture2D* Decoded();
ID3D11Texture2D* Composed();

// The copy of the pass's output from `presents` Presents back (1 or 2), or nullptr. Not AddRef'd.
ID3D11Texture2D* OutputPresentsAgo(int presents);
// The same for the pass's colour input (the raw frame, no TAA in it).
ID3D11Texture2D* ColourPresentsAgo(int presents);

// For the capture, during this pass (nullptr otherwise or when not used): the previous Present's decoded motion, the
// frame depth the composition read, and its copy of the previous Present's depth. Not AddRef'd.
ID3D11Texture2D* PreviousDecoded();
ID3D11Texture2D* FrameDepth();
ID3D11Texture2D* PreviousDepth();

void CycleView();
const char* ViewName();

// Same-eye motion (F4 cycles it for testing; the occlusion-aware chaining below is the fallback in every mode):
// "matrix hybrid": static pixels move by UE4's view matrices of this Present and the previous one, applied
// to the pixel's depth (exact, no lookup in the other eye's image); pixels whose game vector disagrees with the
// matrix motion by more than 1 px (moving objects) are chained. "matrix, halo as static" (default, the one to use):
// the same, but pixels that merely carry a nearer moving object's vector (the halo around it) use the matrix too. "chained through the other
// eye": the previous behaviour for everything. Chaining is occlusion-aware: beside nearer objects, where the other
// eye's frame has the background hidden, the chained step is read from the nearest point along the row at the
// pixel's depth (both Presents' depth; the previous one is copied after each decode).
void CycleMotionMode();
const char* MotionModeName();
// The Days Gone overlay: the same modes by number (0 hybrid, 1 halo as static, 2 chained, 3 halo as static with the object
// motion chained over both Presents where the same surface is found), and how much of each
// pixel's own (object) motion is added on top of the matrix motion (1 = as validated, 0 = the matrix alone; live,
// not saved; the decode logs each new value once).
int MotionMode();
void SetMotionMode(int mode);
float OwnMotionScale();
void SetOwnMotionScale(float scale);
// The Days Gone overlay, live, not saved (the decode logs each new set): the object-motion knee -- own motion fades in
// from KneeStart + KneeRise x |camera motion| (px) to KneeRatio x that -- and the blend to the chained composition,
// from ChainStart to ChainEnd px of own motion. Defaults 0.1, 0.0015, x3, 1, 2 (as validated).
enum Tuning
{
    KneeStart,
    KneeRise,
    KneeRatio,
    ChainStart,
    ChainEnd,
    BusyScale, // object motion kept where the depth is busy (foliage), 1 = all
    kTuningCount
};
float TuningValue(int which);
void SetTuningValue(int which, float value);
void ResetTuning();
// The disocclusion mask for DLSS's "bias current colour" input, during the pass (nullptr when off or without the same
// eye's depth from two Presents back): pixels whose history two Presents back was covered by something nearer. Not
// AddRef'd. Strength = the mask's value there (0 = off); Nearer = how much nearer (reversed-Z ratio) counts as a cover.
ID3D11Texture2D* BiasMask();
float BiasStrength();
void SetBiasStrength(float strength);
float BiasNearer();
void SetBiasNearer(float ratio);
// What the mask marks: 0 off, 1 covers (depth), 2 covers + moved shadows (history darker, smoothed), 3 TEST: the
// whole frame (if DLSS follows the mask, the picture turns aliased and jittery). ShadowDrop: how much darker
// (tonemapped luma) the history must be.
int MaskMode();
void SetMaskMode(int mode);
float ShadowDrop();
void SetShadowDrop(float drop);

// Immediate context, before each of the game's draws: the first draw of each Present that writes a texture like the
// pass's packed motion (the base pass) has its vs cb1 (UE4's view uniform buffer) copied for the matrix motion.
void BeforeDraw(ID3D11DeviceContext* ctx);
} // namespace TaaMotion_Dx11
