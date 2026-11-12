// No pch: tools/xr_observer/test/vk_fovea_test compiles this file as it is (the DLL's project turns the pch off for it).
#include "FoveaPasses_Vk.h"
#include "FoveaShader_Vk.h"

#include <cstring>

namespace FoveaPasses_Vk
{
void CopyBox(VkCommandBuffer cmd, VkImage src, VkImageAspectFlags aspect, uint32_t sx, uint32_t sy, VkImage dst,
             uint32_t w, uint32_t h)
{
    if (cmd == VK_NULL_HANDLE || src == VK_NULL_HANDLE || dst == VK_NULL_HANDLE || w == 0 || h == 0)
        return;
    VkMemoryBarrier before { VK_STRUCTURE_TYPE_MEMORY_BARRIER };
    before.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT;
    before.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &before, 0,
                         nullptr, 0, nullptr);
    VkImageCopy region {};
    region.srcSubresource = { aspect, 0, 0, 1 };
    region.srcOffset = { (int32_t) sx, (int32_t) sy, 0 };
    region.dstSubresource = { aspect, 0, 0, 1 };
    region.dstOffset = { 0, 0, 0 };
    region.extent = { w, h, 1 };
    vkCmdCopyImage(cmd, src, VK_IMAGE_LAYOUT_GENERAL, dst, VK_IMAGE_LAYOUT_GENERAL, 1, &region);
    VkMemoryBarrier after { VK_STRUCTURE_TYPE_MEMORY_BARRIER };
    after.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    after.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &after, 0,
                         nullptr, 0, nullptr);
}

namespace
{
// Everything is in GENERAL: a memory barrier orders the writes before our reads (and ours before the next reader's)
// without a layout change. ALL_COMMANDS on both sides: the writer before us is DLSS (any stage) and the reader after
// us is NR, the VR mod or the game (any stage).
void Barrier(VkCommandBuffer cmd)
{
    VkMemoryBarrier b { VK_STRUCTURE_TYPE_MEMORY_BARRIER };
    b.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &b, 0,
                         nullptr, 0, nullptr);
}
} // namespace

bool Passes::Init(VkDevice device, VkPhysicalDevice physicalDevice, std::string& error)
{
    Destroy();
    _device = device;
    _physical = physicalDevice;
    if (device == VK_NULL_HANDLE)
    {
        error = "no device";
        return false;
    }

    auto setLayout = [&](const VkDescriptorSetLayoutBinding* b, uint32_t count, VkDescriptorSetLayout& out,
                         const char* what)
    {
        VkDescriptorSetLayoutCreateInfo sl { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        sl.bindingCount = count;
        sl.pBindings = b;
        if (vkCreateDescriptorSetLayout(_device, &sl, nullptr, &out) != VK_SUCCESS)
        {
            error = std::string("vkCreateDescriptorSetLayout (") + what + ")";
            return false;
        }
        return true;
    };
    auto pipelineLayout = [&](VkDescriptorSetLayout set, uint32_t pushBytes, VkPipelineLayout& out, const char* what)
    {
        VkPushConstantRange pc { VK_SHADER_STAGE_COMPUTE_BIT, 0, pushBytes };
        VkPipelineLayoutCreateInfo pl { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        pl.setLayoutCount = 1;
        pl.pSetLayouts = &set;
        pl.pushConstantRangeCount = 1;
        pl.pPushConstantRanges = &pc;
        if (vkCreatePipelineLayout(_device, &pl, nullptr, &out) != VK_SUCCESS)
        {
            error = std::string("vkCreatePipelineLayout (") + what + ")";
            return false;
        }
        return true;
    };
    auto pipeline = [&](const uint32_t* code, size_t bytes, VkPipelineLayout layout, VkPipeline& out, const char* what)
    {
        VkShaderModuleCreateInfo mi { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
        mi.codeSize = bytes;
        mi.pCode = code;
        VkShaderModule module = VK_NULL_HANDLE;
        if (vkCreateShaderModule(_device, &mi, nullptr, &module) != VK_SUCCESS)
        {
            error = std::string("vkCreateShaderModule (") + what + ")";
            return false;
        }
        VkComputePipelineCreateInfo ci { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
        ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        ci.stage.module = module;
        ci.stage.pName = "CSMain";
        ci.layout = layout;
        const VkResult r = vkCreateComputePipelines(_device, VK_NULL_HANDLE, 1, &ci, nullptr, &out);
        vkDestroyShaderModule(_device, module, nullptr);
        if (r != VK_SUCCESS)
        {
            error = std::string("vkCreateComputePipelines (") + what + ")";
            return false;
        }
        return true;
    };

    // COMPOSITE: 0 the crop (sampled, read by Load), 1 the output (storage, read and written).
    const VkDescriptorSetLayoutBinding cb[] = {
        { 0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
    };
    // ENLARGE: 0 the source (sampled), 1 the sampler, 2 the output (storage).
    const VkDescriptorSetLayoutBinding eb[] = {
        { 0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 1, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
    };
    // TAA: 0 colour, 1 motion, 2 depth, 3 history (sampled), 4 the sampler, 5 the output, 6 the history out (storage).
    const VkDescriptorSetLayoutBinding tb[] = {
        { 0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 2, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 3, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 4, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 5, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 6, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
    };
    if (!setLayout(cb, 2, _compositeSetLayout, "composite") || !setLayout(eb, 3, _enlargeSetLayout, "enlarge") ||
        !setLayout(tb, 7, _taaSetLayout, "taa") ||
        !pipelineLayout(_compositeSetLayout, sizeof(CompositeConstants), _compositeLayout, "composite") ||
        !pipelineLayout(_enlargeSetLayout, sizeof(EnlargeConstants), _enlargeLayout, "enlarge") ||
        !pipelineLayout(_taaSetLayout, sizeof(TaaConstants), _taaLayout, "taa") ||
        !pipeline(kCompositeShaderSpv, sizeof(kCompositeShaderSpv), _compositeLayout, _compositePipeline,
                  "composite") ||
        !pipeline(kEnlargeShaderSpv, sizeof(kEnlargeShaderSpv), _enlargeLayout, _enlargePipeline, "enlarge") ||
        !pipeline(kTaaShaderSpv, sizeof(kTaaShaderSpv), _taaLayout, _taaPipeline, "taa"))
        return false;

    const VkDescriptorPoolSize sizes[] = {
        { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 6 * kRing },
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 4 * kRing },
        { VK_DESCRIPTOR_TYPE_SAMPLER, 2 * kRing },
    };
    VkDescriptorPoolCreateInfo pi { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    pi.maxSets = 3 * kRing;
    pi.poolSizeCount = 3;
    pi.pPoolSizes = sizes;
    if (vkCreateDescriptorPool(_device, &pi, nullptr, &_pool) != VK_SUCCESS)
    {
        error = "vkCreateDescriptorPool";
        return false;
    }
    auto allocate = [&](VkDescriptorSetLayout layout, VkDescriptorSet* sets, const char* what)
    {
        VkDescriptorSetLayout layouts[kRing];
        for (uint32_t i = 0; i < kRing; ++i)
            layouts[i] = layout;
        VkDescriptorSetAllocateInfo ai { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        ai.descriptorPool = _pool;
        ai.descriptorSetCount = kRing;
        ai.pSetLayouts = layouts;
        if (vkAllocateDescriptorSets(_device, &ai, sets) != VK_SUCCESS)
        {
            error = std::string("vkAllocateDescriptorSets (") + what + ")";
            return false;
        }
        return true;
    };
    if (!allocate(_compositeSetLayout, _compositeSets, "composite") ||
        !allocate(_enlargeSetLayout, _enlargeSets, "enlarge") || !allocate(_taaSetLayout, _taaSets, "taa"))
        return false;

    VkSamplerCreateInfo si { VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
    si.magFilter = si.minFilter = VK_FILTER_LINEAR;
    si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.maxLod = 0.0f;
    if (vkCreateSampler(_device, &si, nullptr, &_sampler) != VK_SUCCESS)
    {
        error = "vkCreateSampler";
        return false;
    }
    _ready = true;
    return true;
}

void Passes::Destroy()
{
    if (_device == VK_NULL_HANDLE)
        return;
    if (_sampler != VK_NULL_HANDLE)
        vkDestroySampler(_device, _sampler, nullptr);
    if (_pool != VK_NULL_HANDLE)
        vkDestroyDescriptorPool(_device, _pool, nullptr); // frees the sets
    if (_compositePipeline != VK_NULL_HANDLE)
        vkDestroyPipeline(_device, _compositePipeline, nullptr);
    if (_enlargePipeline != VK_NULL_HANDLE)
        vkDestroyPipeline(_device, _enlargePipeline, nullptr);
    if (_taaPipeline != VK_NULL_HANDLE)
        vkDestroyPipeline(_device, _taaPipeline, nullptr);
    if (_taaLayout != VK_NULL_HANDLE)
        vkDestroyPipelineLayout(_device, _taaLayout, nullptr);
    if (_taaSetLayout != VK_NULL_HANDLE)
        vkDestroyDescriptorSetLayout(_device, _taaSetLayout, nullptr);
    if (_compositeLayout != VK_NULL_HANDLE)
        vkDestroyPipelineLayout(_device, _compositeLayout, nullptr);
    if (_enlargeLayout != VK_NULL_HANDLE)
        vkDestroyPipelineLayout(_device, _enlargeLayout, nullptr);
    if (_compositeSetLayout != VK_NULL_HANDLE)
        vkDestroyDescriptorSetLayout(_device, _compositeSetLayout, nullptr);
    if (_enlargeSetLayout != VK_NULL_HANDLE)
        vkDestroyDescriptorSetLayout(_device, _enlargeSetLayout, nullptr);
    _sampler = VK_NULL_HANDLE;
    _pool = VK_NULL_HANDLE;
    _compositePipeline = _enlargePipeline = _taaPipeline = VK_NULL_HANDLE;
    _compositeLayout = _enlargeLayout = _taaLayout = VK_NULL_HANDLE;
    _compositeSetLayout = _enlargeSetLayout = _taaSetLayout = VK_NULL_HANDLE;
    for (uint32_t i = 0; i < kRing; ++i)
        _compositeSets[i] = _enlargeSets[i] = _taaSets[i] = VK_NULL_HANDLE;
    _ready = false;
    _device = VK_NULL_HANDLE;
}

void Passes::Composite(VkCommandBuffer cmd, uint32_t slot, VkImageView crop, VkImageView output,
                       const CompositeConstants& c)
{
    if (!_ready || cmd == VK_NULL_HANDLE || crop == VK_NULL_HANDLE || output == VK_NULL_HANDLE || c.sizeX == 0 ||
        c.sizeY == 0)
        return;
    const VkDescriptorSet set = _compositeSets[slot % kRing];
    const VkDescriptorImageInfo cropInfo { VK_NULL_HANDLE, crop, VK_IMAGE_LAYOUT_GENERAL };
    const VkDescriptorImageInfo outInfo { VK_NULL_HANDLE, output, VK_IMAGE_LAYOUT_GENERAL };
    const VkWriteDescriptorSet writes[] = {
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, 0, 0, 1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &cropInfo,
          nullptr, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &outInfo,
          nullptr, nullptr },
    };
    vkUpdateDescriptorSets(_device, 2, writes, 0, nullptr);
    Barrier(cmd);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _compositePipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _compositeLayout, 0, 1, &set, 0, nullptr);
    vkCmdPushConstants(cmd, _compositeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(c), &c);
    vkCmdDispatch(cmd, (c.sizeX + 7) / 8, (c.sizeY + 7) / 8, 1);
    Barrier(cmd);
}

void Passes::Enlarge(VkCommandBuffer cmd, uint32_t slot, VkImageView src, VkImageView output, const EnlargeConstants& c)
{
    if (!_ready || cmd == VK_NULL_HANDLE || src == VK_NULL_HANDLE || output == VK_NULL_HANDLE || c.outW == 0 ||
        c.outH == 0 || c.srcW == 0 || c.srcH == 0)
        return;
    const VkDescriptorSet set = _enlargeSets[slot % kRing];
    const VkDescriptorImageInfo srcInfo { VK_NULL_HANDLE, src, VK_IMAGE_LAYOUT_GENERAL };
    const VkDescriptorImageInfo samplerInfo { _sampler, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED };
    const VkDescriptorImageInfo outInfo { VK_NULL_HANDLE, output, VK_IMAGE_LAYOUT_GENERAL };
    const VkWriteDescriptorSet writes[] = {
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, 0, 0, 1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &srcInfo,
          nullptr, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, 1, 0, 1, VK_DESCRIPTOR_TYPE_SAMPLER, &samplerInfo,
          nullptr, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &outInfo,
          nullptr, nullptr },
    };
    vkUpdateDescriptorSets(_device, 3, writes, 0, nullptr);
    Barrier(cmd);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _enlargePipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _enlargeLayout, 0, 1, &set, 0, nullptr);
    vkCmdPushConstants(cmd, _enlargeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(c), &c);
    vkCmdDispatch(cmd, (c.outW + 7) / 8, (c.outH + 7) / 8, 1);
    Barrier(cmd);
}

void Passes::Taa(VkCommandBuffer cmd, uint32_t slot, VkImageView colour, VkImageView motion, VkImageView depth,
                 VkImageView history, VkImageView output, VkImageView historyOut, const TaaConstants& c)
{
    if (!_ready || cmd == VK_NULL_HANDLE || colour == VK_NULL_HANDLE || motion == VK_NULL_HANDLE ||
        depth == VK_NULL_HANDLE || history == VK_NULL_HANDLE || output == VK_NULL_HANDLE ||
        historyOut == VK_NULL_HANDLE || c.outSize[0] == 0 || c.outSize[1] == 0)
        return;
    const VkDescriptorSet set = _taaSets[slot % kRing];
    const VkDescriptorImageInfo colourInfo { VK_NULL_HANDLE, colour, VK_IMAGE_LAYOUT_GENERAL };
    const VkDescriptorImageInfo motionInfo { VK_NULL_HANDLE, motion, VK_IMAGE_LAYOUT_GENERAL };
    const VkDescriptorImageInfo depthInfo { VK_NULL_HANDLE, depth, VK_IMAGE_LAYOUT_GENERAL };
    const VkDescriptorImageInfo historyInfo { VK_NULL_HANDLE, history, VK_IMAGE_LAYOUT_GENERAL };
    const VkDescriptorImageInfo samplerInfo { _sampler, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED };
    const VkDescriptorImageInfo outInfo { VK_NULL_HANDLE, output, VK_IMAGE_LAYOUT_GENERAL };
    const VkDescriptorImageInfo histOutInfo { VK_NULL_HANDLE, historyOut, VK_IMAGE_LAYOUT_GENERAL };
    const VkWriteDescriptorSet writes[] = {
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, 0, 0, 1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &colourInfo,
          nullptr, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, 1, 0, 1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &motionInfo,
          nullptr, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, 2, 0, 1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &depthInfo,
          nullptr, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, 3, 0, 1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &historyInfo,
          nullptr, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, 4, 0, 1, VK_DESCRIPTOR_TYPE_SAMPLER, &samplerInfo,
          nullptr, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, 5, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &outInfo,
          nullptr, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, 6, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &histOutInfo,
          nullptr, nullptr },
    };
    vkUpdateDescriptorSets(_device, 7, writes, 0, nullptr);
    Barrier(cmd);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _taaPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _taaLayout, 0, 1, &set, 0, nullptr);
    vkCmdPushConstants(cmd, _taaLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(c), &c);
    vkCmdDispatch(cmd, (c.outSize[0] + 7) / 8, (c.outSize[1] + 7) / 8, 1);
    Barrier(cmd);
}
} // namespace FoveaPasses_Vk
