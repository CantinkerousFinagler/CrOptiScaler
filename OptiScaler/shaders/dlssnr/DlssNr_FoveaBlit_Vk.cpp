#include "pch.h"

#include "DlssNr_FoveaBlit_Vk.h"
#include "precompile/dlssnr_fovea_Shader_Vk.h"

#include <algorithm>
#include <cstring>

namespace DlssNr
{
namespace
{
struct Constants
{
    uint32_t roiX, roiY, roiW, roiH;
    float roundness, feather;
    uint32_t showBorder, pad;
};
} // namespace

FoveaBlit_Vk::FoveaBlit_Vk(VkDevice device, VkPhysicalDevice physicalDevice)
    : Shader_Vk("DLSS-NR fovea blit", device, physicalDevice)
{
    if (device == VK_NULL_HANDLE || physicalDevice == VK_NULL_HANDLE)
        return;

    _maxFramesInFlight = kSlots;
    std::vector<VkDescriptorSetLayoutBinding> bindings = {
        CreateBinding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER),
        CreateBinding(1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE),
        CreateBinding(2, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE),
        CreateBinding(3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE),
    };
    CreateLayouts(bindings);
    std::vector<VkDescriptorPoolSize> poolSizes = {
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, kSlots },
        { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 2 * kSlots },
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, kSlots },
    };
    CreateDescriptorPool(poolSizes, kSlots);
    CreateDescriptorSets(_descriptorSetLayout, _descriptorPool, _descriptorSets);

    VkPhysicalDeviceProperties props {};
    vkGetPhysicalDeviceProperties(_physicalDevice, &props);
    const VkDeviceSize alignment = std::max<VkDeviceSize>(props.limits.minUniformBufferOffsetAlignment, 1);
    _slotStride = ((sizeof(Constants) + alignment - 1) / alignment) * alignment;
    CreateConstantBuffer(_slotStride * kSlots);
    if (_mappedConstantBuffer == nullptr)
        return;

    std::vector<char> shaderCode(dlssnr_fovea_spv, dlssnr_fovea_spv + sizeof(dlssnr_fovea_spv));
    if (!CreateComputePipeline(_device, _pipelineLayout, &_pipeline, shaderCode))
        return;

    _init = true;
}

bool FoveaBlit_Vk::Blend(VkCommandBuffer cmd, VkImageView original, VkImageView edited, VkImageView dest,
                         ColorExtent roi, float roundness, float feather, bool showBorder)
{
    if (!_init || cmd == VK_NULL_HANDLE || original == VK_NULL_HANDLE || edited == VK_NULL_HANDLE ||
        dest == VK_NULL_HANDLE || roi.width == 0 || roi.height == 0)
        return false;

    const uint32_t slot = _slot++ % kSlots;
    Constants constants { roi.x, roi.y, roi.width, roi.height, std::clamp(roundness, 0.0f, 1.0f),
                          std::clamp(feather, 0.0f, 0.5f), showBorder ? 1u : 0u, 0u };
    std::memcpy(static_cast<unsigned char*>(_mappedConstantBuffer) + slot * _slotStride, &constants,
                sizeof(constants));

    VkDescriptorSet set = _descriptorSets[slot];
    VkDescriptorBufferInfo ubo { _constantBuffer, slot * _slotStride, sizeof(Constants) };
    VkDescriptorImageInfo orig { VK_NULL_HANDLE, original, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
    VkDescriptorImageInfo edit { VK_NULL_HANDLE, edited, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
    VkDescriptorImageInfo out { VK_NULL_HANDLE, dest, VK_IMAGE_LAYOUT_GENERAL };
    VkWriteDescriptorSet writes[] = {
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, 0, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr,
          &ubo, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, 1, 0, 1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &orig,
          nullptr, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, 2, 0, 1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &edit,
          nullptr, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, 3, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &out,
          nullptr, nullptr },
    };
    vkUpdateDescriptorSets(_device, 4, writes, 0, nullptr);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _pipelineLayout, 0, 1, &set, 0, nullptr);
    vkCmdDispatch(cmd, (roi.width + 7) / 8, (roi.height + 7) / 8, 1);
    return true;
}
} // namespace DlssNr
