#pragma once

// DLAA in place of a D3D11 game's TAA compute pass ([Inputs] TaaDlaa, startup only), for alternate-eye VR: one
// DLSS feature per eye, alternating every Present. Built on TaaMotion_Dx11 (the same pass, its composed motion to
// two Presents back) and on Days Gone's tonemap + TAA pass layout:
//   t0 HUD layer (RGBA8), t2 colour (RGBA16F, tonemapped, square-root-like encoding), cb0 the pass's constants,
//   u0 the final image (R10G10B10A2).
// After the game's own dispatch, DLSS runs on t2 with the composed motion, this frame's depth (the gameplay depth
// FinalImageNr_Dx11 tracks) and the frame's jitter, and a finish pass redoes what the game's shader does after its
// TAA (film grain, squaring, vignette, second grain, HUD blend, sRGB) from the DLSS output into u0. The game's TAA
// history (u1) is still written, so switching back is seamless.
//
// Jitter: the view constants left bound in cs cb3 hold the sample index (+464) and the Halton(2,3) - 0.5 offsets
// (+472/+476) for this Present; they are read back a few Presents late and the index is predicted forward (it
// advances by one per Present). DLSS gets (+x, -y) by default, the negated sample offset (confirmed in VR); F5
// cycles the signs.
//
// With the game's Render Scale below 100% the pass is TaaMotion_Dx11's render-resolution layout. Then a second pair
// of features upscales its colour (t1) to the eye size with the same motion, depth and jitter, and the game's
// upscale draw (the first draw reading that pass's output) is replaced by a finish pass from the DLSS output (the
// draw's steps after its sample and YCoCg decode: square, vignette, grain, HUD t0, sRGB) into the draw's target.
// The game's render-resolution pass is skipped ([Inputs] TaaDlaaSkipGameTaa) only while the previous Present's draw was
// replaced, so a frame DLSS can't do shows the game's own upscale. The eye
// size comes from that draw, so upscaling starts once it has been seen.
//
// Everything runs inside a swap to a spare device context state, so the game's whole pipeline state comes back
// exactly (UE4 caches bindings). F6 toggles game TAA (the default at start) and DLAA. The features are
// created on the first switch to DLAA and never rebuilt; turn it on only once the game renders the VR eye.
namespace TaaDlaa_Dx11
{
bool Enabled();

// True while this module runs its passes or DLSS: the D3D11 hooks pass those calls straight through.
bool InPass();

// Immediate context. BeforeDispatch runs before the game's Dispatch (after TaaMotion_Dx11::BeforeDispatch). Replace
// runs DLAA and writes the pass's output, in place of the game's Dispatch ([Inputs] TaaDlaaSkipGameTaa) or after
// it; true when it replaced the output this time (else the game's Dispatch must run).
void BeforeDispatch(ID3D11DeviceContext* ctx);
bool Replace(ID3D11DeviceContext* ctx);

// Immediate context, before one of the game's draws: true when it was the game's upscale draw and the finish pass
// wrote its target from DLSS instead (skip the draw). Also learns the eye size from that draw in any mode.
bool ReplaceDraw(ID3D11DeviceContext* ctx);

// True when the frame now being presented shows DLSS's image (the TAA pass's output or the upscale draw replaced).
bool ReplacedThisPresent();

// True while F6 has DLSS on (not the game's TAA).
bool On();
// Frees both feature sets at the next replaced pass; they are made again with the current settings (a preset picked
// in OptiScaler's overlay). A hitch of about a second per set.
void RequestRebuild();
// Our sharpening in the finish pass (an unsharp mask on the DLSS output), strength 0..2 and radius 1 or 2 px, live,
// not saved; 0 = off.
float Sharpness();
void SetSharpness(float sharpness);
int SharpenRadius();
void SetSharpenRadius(int radius);
// F12: logs the moment, and at the next DLSS pass what DLSS was fed over the last 180 Presents.
void UserMark();
// Why DLSS was not shown in `present` although the pass was decoded: a slot below kMissSlots, the last one is "nothing
// recorded". TaaMotion_Dx11 counts and logs these at Present.
constexpr int kMissSlots = 14;
int MissSlot(uint64_t present);
const char* MissSlotName(int slot);

void CycleMode();
const char* ModeName();
void CycleJitter();
const char* JitterName();
} // namespace TaaDlaa_Dx11
