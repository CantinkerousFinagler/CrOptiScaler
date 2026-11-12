#pragma once

#include <vulkan/vulkan.h>

#include "FoveaMenuState_Vk.h"

struct NVSDK_NGX_Parameter;
class IFeature_Vk;

// The foveal SR crop of PerEyeDlss_Dx12 on OptiScaler's VULKAN feature path (a Vulkan
// game), stage (a): a fixed box at the centre of the eye, rendered by a crop-sized DLSS feature of its own at a
// higher preset (UpscalerFoveaPreset) and blended over the game's full pass with a feathered edge (COMPOSITE,
// FoveaPasses_Vk). The box is PerEyeDlss_Dx12's (FoveaBox.h); the keys are the D3D12 ones ([Inputs] UpscalerFoveaCrop,
// UpscalerFoveaCropHeight, UpscalerFoveaPreset, UpscalerFoveaFeather, UpscalerCropEyeByParity, and
// UpscalerFoveaStart = on from the launch). Switched in the OVERLAY (FoveaMenu_Vk.h, the "Foveated SR" block of
// "Per-eye DLSS (Vulkan)"): F4 is one Vulkan game's own key (pressed twice it left VR) and does
// nothing on Vulkan; F8 still toggles the outline.
//
// How the crop reads its inputs here: through DLSS's own input subrects (the D3D12 "subrects used" path): the game's
// colour, depth and motion stay where they are and the crop's block names base + box. A fixed box at a whole
// render-pixel origin needs no copy; the gather copy (the moving crop, the gaze, the bias mask at the box) is stage (b).
// The crop runs on a parameter block of its own (NVSDK_NGX_VULKAN_AllocateParameters), filled from the game's at every
// evaluate (the key lists of PerEyeDlss_Dx12); the game's block is never written.
//
// Stage A2, THE CHEAP PERIPHERY = the saving: with [Inputs] UpscalerFoveaCheapPeriphery on, a periphery
// feature per eye key at the RENDER size in and out (UpscalerFoveaPeripheryPreset, E) runs on a block of its own
// filled from the game's (Output = our periphery image), its result is ENLARGEd over the whole eye (Catmull-Rom,
// FoveaPasses_Vk), and the game's full pass is SKIPPED for that evaluate (NVNGX_DLSS_Vk: Periphery() in its place);
// the crop is then composited over the centre as before. A game feature skipped meanwhile has a stale history: when
// its full pass runs again (periphery off, a fallback), TakeMainSkipped says so and the hook asks DLSS for a reset.
//
// Not here yet (D3D12 only): the gaze, the bias mask at the box, the TAA periphery, RR crops (the projection).
namespace FoveaCrop_Vk
{
// [Inputs] UpscalerPerEye (the module), as PerEyeDlss_Vk.
bool Enabled();
// Nonzero while a crop feature is being created ON THIS THREAD (the game's for an inline create, the worker's for
// VkCreateWorker's): DLSSFeature_Vk hands DLSS that preset (as DLSSFeature_Dx12 does).
int PresetForCreate();

// After the owner's (the game's or the twin's) evaluate succeeded, on the same command buffer, before NR after-upscale:
// plans the box, keeps the crop feature and image, evaluates the crop and composites it into the game's output.
void AfterUpscale(VkCommandBuffer cmd, unsigned int ownerHandle, NVSDK_NGX_Parameter* gameParams, IFeature_Vk* owner,
                  VkDevice device, VkPhysicalDevice physical);
// In the PLACE of the owner's full pass (before it would be evaluated; NR's colour swap already made): with the
// crop and the cheap periphery on, evaluates the eye's periphery feature at the render size and enlarges it over
// the game's output. True = drawn, the owner's full pass is to be skipped this evaluate. False = the full pass runs.
bool Periphery(VkCommandBuffer cmd, unsigned int ownerHandle, NVSDK_NGX_Parameter* gameParams, IFeature_Vk* owner,
               VkDevice device, VkPhysicalDevice physical);
// The owner's full pass was skipped for the periphery since it last ran: its history is stale (once per return).
bool TakeMainSkipped(unsigned int ownerHandle);
// The owner (or its twin) is released: its crops and peripheries go with it.
void Forget(unsigned int ownerHandle);
// Every vkQueuePresentKHR (retired images).
void OnPresent();

void ToggleFovea();   // the overlay (F4 on D3D12 only)
void ToggleOutline(); // F8, and the overlay
bool FoveaOn();
const char* FoveaName();

// The overlay's block (FoveaMenu_Vk.h): what to show, and what was asked for applied (the Config keys written,
// the crop features remade at their next evaluate where the box or the preset changed).
FoveaMenuVk::Status GetStatus();
void Apply(const FoveaMenuVk::Answer& answer);
} // namespace FoveaCrop_Vk
