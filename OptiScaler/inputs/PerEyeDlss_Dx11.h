#pragma once

#include "PerEyeMenuState.h"

struct ID3D11DeviceContext;
struct NVSDK_NGX_Parameter;

// Per-eye DLSS for a game (or mod) that calls DLSS through OptiScaler's own D3D11 feature path with ONE feature, under a
// VR mod that renders one eye per Present (alternate eye rendering). First user: a D3D11 game with a mod calling DLSS.
//
// When a DLSS feature is created, a twin is created with the same parameters. At evaluate, Presents of one parity
// keep the game's feature and the other parity goes to the twin, so each eye keeps a history of its own. Routing
// starts SHARED -- the game's (or mod's) picture untouched -- and F6 switches to per eye, as F6 starts DLSS in FC5,
// FC6 and Days Gone. Per eye (user-validated, object variant 1) needs the fitted same-eye motion below: the
// game's own vectors point at the other eye (+-25..40 render px by parity). F6 (live) resets both histories once. Nothing is created or released live except at the game's
// own create/release. A small in-headset indicator shows the mode, the F4 variant and the F8 view while a view is on
// and for 3 s after a key.
//
// Per eye, the game's vectors (which point at the other eye) are replaced by same-eye motion: each frame's camera map is
// fitted from the vectors and depth (one projective 4x4), the last two are chained, and object motion is carried over.
//
// F8: a motion view in place of the picture (the scene in grey, DLSS's motion vectors as colour: red = moving right,
// blue = moving left, green = vertical, full at 32 render px). With [Inputs] UpscalerInputProbe it also logs, per
// Present parity, the mean horizontal motion vector over a sparse grid -- opposite signs for the two parities with the
// head still = the vectors point at the other eye.
namespace PerEyeDlss_Dx11
{
// [Inputs] UpscalerPerEye (startup only, default on).
bool Enabled();

void OnPresent();

// Feature lifetime, from NVSDK_NGX_D3D11_CreateFeature / ReleaseFeature.
void RegisterTwin(unsigned int primary, unsigned int twin);
unsigned int TwinOf(unsigned int primary); // 0 if none
void Forget(unsigned int primary);

// Which feature this evaluate goes to (the game's or its twin). May set the reset flag once after a routing switch.
unsigned int Route(unsigned int handle, NVSDK_NGX_Parameter* params);

// Per eye only: fits this Present's camera map from the game's vectors and depth and, with the previous Present's fit,
// hands DLSS same-eye motion in place of the game's vectors (or resets this eye's history when there is nothing to
// chain). EndEvaluate hands the game's texture back to its parameters.
void BeforeEvaluate(ID3D11DeviceContext* ctx, NVSDK_NGX_Parameter* params);
void EndEvaluate(NVSDK_NGX_Parameter* params);

// After DLSS wrote its output: the motion view and the motion statistics, in a spare device context state.
void AfterEvaluate(ID3D11DeviceContext* ctx, NVSDK_NGX_Parameter* params);

// Keys (menu_common).
void ToggleRouting();
bool RoutingOn();
void ToggleView(); // F8: off -> object motion -> what DLSS gets -> off
bool ViewOn();
const char* ViewName();
void CycleVariant(); // F4, per eye: the object motion variant
const char* VariantName();

// The overlay's section (PerEyeMenu.h draws it): the state, and the same switches by value.
PerEyeMenu::Status GetStatus();
void SetView(int mode);       // 0 off, 1 object motion, 2 what DLSS gets
void SetVariant(int variant); // 0..3
} // namespace PerEyeDlss_Dx11
