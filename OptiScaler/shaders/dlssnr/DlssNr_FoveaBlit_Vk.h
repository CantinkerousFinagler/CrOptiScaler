#pragma once

#include <shaders/Shader_Vk.h>
#include <shaders/dlssnr/DlssNr_ActiveColor.h>

namespace DlssNr
{
class FoveaBlit_Vk : public Shader_Vk
{
    VkDeviceSize _slotStride = 0;
    uint32_t _slot = 0;
    static constexpr uint32_t kSlots = 8;

  public:
    FoveaBlit_Vk(VkDevice device, VkPhysicalDevice physicalDevice);
    bool Blend(VkCommandBuffer cmd, VkImageView original, VkImageView edited, VkImageView dest, ColorExtent roi,
               float roundness, float feather, bool showBorder);
};
} // namespace DlssNr
