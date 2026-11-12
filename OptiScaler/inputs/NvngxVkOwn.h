#pragma once

#include <vulkan/vulkan.h>

#include <upscalers/IFeature_Vk.h>

#include <memory>

// Features of OUR OWN on OptiScaler's Vulkan feature path (NVNGX_DLSS_Vk.cpp owns the context map): the foveal crop's
// features (FoveaCrop_Vk), made as the per-eye twin is made (FeatureProvider_Vk + Init with the game's Vulkan
// handles, OptiScaler's current feature put back to what it WAS), and retired as a game's feature is (VkRetire).
namespace NvngxVkOwn
{
// A new feature of `backend` from `params` (the creation keys set by the caller); 0 on failure.
unsigned int Create(VkCommandBuffer cmd, Upscaler backend, NVSDK_NGX_Parameter* params);
// The feature of an id, or null.
IFeature_Vk* Get(unsigned int id);
// Released as the game's features are ([Hotfix] VulkanNoStalls / VulkanRetireLater).
void Retire(unsigned int id);

// ON A WORKER THREAD (VkCreateWorker): a feature made and initialised on `cmd` (the worker's command buffer) but put
// in no map -- the context map is the game thread's. `id` is from IFeature::GetNextHandleId() on the game thread.
bool MakeDetached(VkCommandBuffer cmd, Upscaler backend, NVSDK_NGX_Parameter* params, unsigned int id,
                  std::unique_ptr<IFeature_Vk>& out);
// On the game thread: a detached feature put in the map under its id (Get / Retire work from here on).
void Adopt(unsigned int id, std::unique_ptr<IFeature_Vk> feature);
} // namespace NvngxVkOwn
