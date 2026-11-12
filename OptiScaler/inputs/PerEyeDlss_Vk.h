#pragma once

#include <vulkan/vulkan.h>

struct NVSDK_NGX_Parameter;

// Per-eye DLSS on OptiScaler's own VULKAN feature path: the port of PerEyeDlss_Dx12's core (for a Vulkan
// game under R.E.A.L. VR, which renders one eye per frame). When the game's DLSS
// feature is created, a twin is made from the same parameters and backend; at evaluate, while routing is per eye (F6,
// starts shared), every other frame goes to the twin so each eye keeps a DLSS history of its own, and the game's
// other-eye vectors are replaced by same-eye motion: SameEyeFit_Vk fits this frame's camera map from the game's
// vectors and depth (the very shader of D3D12, as SPIR-V) and composes it with the other eye's previous fit.
//
// The eye is the parity of a clock: the Presents (vkQueuePresentKHR) once they are seen to advance between evaluates,
// the evaluates themselves otherwise (logged once which). Not yet on Vulkan (D3D12 only): the foveal crop, the cheap
// periphery, our TAA periphery, the gaze, the bias mask, NR per eye.
namespace PerEyeDlss_Vk
{
// [Inputs] UpscalerPerEye (startup only), shared with the D3D11 / D3D12 modules; each acts on its own API.
bool Enabled();
// Every vkQueuePresentKHR.
void OnPresent();

// Feature lifetime (NVSDK_NGX_VULKAN_CreateFeature1 / ReleaseFeature).
void RegisterTwin(unsigned int primary, unsigned int twin);
unsigned int TwinOf(unsigned int primary); // 0 if none
void Forget(unsigned int primary);

// The feature this evaluate goes to (the game's or its twin); may set the reset flag after a routing switch.
unsigned int Route(unsigned int handle, NVSDK_NGX_Parameter* params);
// Before the upscaler (and NR) on the evaluate's command buffer: per eye, the fit and the compose, and our same-eye
// motion in the parameters (or this eye's history reset when there is nothing to chain).
void BeforeEvaluate(VkCommandBuffer cmd, NVSDK_NGX_Parameter* params, VkDevice device, VkPhysicalDevice physical);
// After the upscaler and NR: the game's own vectors back in its parameters.
void RestoreInputs(NVSDK_NGX_Parameter* params);

// F6.
void ToggleRouting();
bool RoutingOn();
// This evaluate's eye by the clock's parity (the Presents', once they advance between evaluates): the foveal crop's
// eye key in shared routing (FoveaCrop_Vk).
unsigned int EyeParity();
// Overlay: save the next 16 fits' samples to OptiFitDump\<time>\ beside the exe (as D3D12's 'Save the fit's samples').
void RequestFitDump();
struct Status
{
    bool routing = false, twin = false, presentClock = false;
    unsigned long long fitsValid = 0, fitsRead = 0; // the last whole window
    unsigned int dumpLeft = 0, dumpSaved = 0;
};
Status GetStatus();
} // namespace PerEyeDlss_Vk
