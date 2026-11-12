#pragma once

#include <string>

// FC5: per-eye DLAA in place of the game's TAA draw ([Inputs] Fc5TaaDlss, startup only), for alternate-eye VR.
// Modelled on Days Gone's TaaMotion_Dx11 + TaaDlaa_Dx11, but FC5's TAA is a pixel-shader draw at render size:
//   ps t0 the motion vectors (half-res R8G8_UNORM, this Present's motion, to the previous Present = the other eye),
//   t1 its history, t2 the scene colour (current tonemapped 8-bit colour), t3/t4 previous colour / accumulation,
//   ps cb0 the temporal constants (+80: the jitter parameters), rt0 the TAA output (RGBA8, also next Present's t4).
// After the game's draw, a compute pass makes R16G16_FLOAT render-size pixel offsets (previous minus current) from
// depth and the game's current-to-previous matrix (copied at its motion vector pass), and chains this Present's matrix with
// the previous one's for motion to two Presents back, the same eye. Moving objects (and frames without a matrix) use
// t0 decoded and chained instead, occlusion-aware as in Days Gone. See kDecodeShader. DLSS then runs on t2 with
// that motion, the G-buffer depth (reversed Z) and the jitter, one feature per eye, and its output is copied over
// rt0 (100% Resolution Scale). Below 100% (the game's upscale draw, the first draw reading the TAA output into a
// render-target-only eye target, is larger) two upscaling features go from the render size to the eye size and their
// output is converted and copied over its target right after that draw (ReplaceUpscaleDraw / AfterUpscaleDraw); the
// game's TAA draw is skipped as with DLAA. Never run DLSS before the game's TAA draw and then let that draw run: it
// lost the D3D11 device every time (a GPU fault; cause unknown). [Inputs] Fc5TaaUpscaleShow = false keeps the game's
// upscale (isolation test).
// NR runs inside the DLSS evaluation (before SR with [DlssNr] RunBeforeSR). With this on, the older upscale-draw
// path (Fc5Dlss_Dx11) is off.
//
// Jitter: (0.5, 0.5) - the jitter parameters' xy = the motion vector offsets = the image's shift in pixels (x right, y down), a
// 2-phase pattern advancing every Present (so constant per eye). Read back from ps cb0 a few Presents late and
// predicted by Present parity. By default it is replaced by our own per-eye sequence (see OnMap below); F4 toggles
// back to the game's, F5 (diagnostics) cycles the signs DLSS gets.
//
// F6 toggles game TAA (the default at start) / DLAA; the features are created on the first switch (a few seconds'
// freeze). They follow the Resolution Scale and R.E.A.L. VR's resolution during play: rebuilt once a
// new size has held for a second of gameplay (see "Sizes that change during play" in the .cpp). With
// [Inputs] DuniaTaaDiagnostics: F8 cycles a motion view (decoded / composed) over the TAA output, F4 toggles
// occlusion-aware composition.
namespace Fc5Taa_Dx11
{
bool Enabled();

// True while this module runs its passes or DLSS: the D3D11 hooks pass those calls straight through.
bool InPass();

// Immediate context, from the render target bind hooks: tracks the G-buffer depth.
void OnBind(ID3D11DeviceContext* ctx, UINT numRtvs, ID3D11RenderTargetView* const* rtvs,
            ID3D11DepthStencilView* dsv);
// Immediate context, before each of the game's dispatches: at FC5's motion vector pass, copies its constants
// (the current-to-previous matrix) and keeps its object flags.
void BeforeDispatch(ID3D11DeviceContext* ctx);
// Immediate context, around each of the game's draws; act when it is the TAA draw. ReplaceDraw runs before it and
// returns true when it wrote the TAA output in its place (skip the game's draw; [Inputs] TaaDlaaSkipGameTaa, default
// on); otherwise AfterDraw does the work after the game's draw.
bool ReplaceDraw(ID3D11DeviceContext* ctx);
void AfterDraw(ID3D11DeviceContext* ctx);
// Before the real Present.
void OnPresent();

void CycleMode();
const char* ModeName();
// The overlay: frees both feature pairs at the next TAA draw; they are made again with the settings as they are then
// (the DLSS preset). What DLSS is doing now, in words.
void RequestRebuild();
std::string StatusText();
void CycleJitter();
const char* JitterName();
void CycleView();
const char* ViewName();
void ToggleOcclusion();
bool OcclusionOn();

// Own per-eye jitter ([Inputs] Fc5TaaOwnJitter, default on; F4 toggles): while DLAA runs, the game's writes of its
// view constants are patched to our jitter (see the .cpp). From the Map / Unmap / UpdateSubresource hooks, any context.
void OnMap(ID3D11Resource* resource, UINT subresource, D3D11_MAP type, void* data);
void BeforeUnmap(ID3D11Resource* resource, UINT subresource);
// Returns the data to pass on: `data`, or a patched copy (valid until the next call on this thread).
const void* PatchUpdate(ID3D11Resource* resource, UINT subresource, const D3D11_BOX* box, const void* data);
void ToggleOwnJitter();
bool OwnJitterOn();
// F9 diagnostic: cycles the motion for FC5's object-flagged pixels (default: the game's vectors only where they
// disagree with the depth + camera motion; matrix everywhere; the game's vectors on every flagged pixel).
void ToggleMatrixEverywhere();
const char* ObjectMotionName();
} // namespace Fc5Taa_Dx11
