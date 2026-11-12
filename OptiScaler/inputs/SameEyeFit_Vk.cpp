// No pch: tools/xr_observer/test/vk_fit_test compiles this file as it is (the DLL's project turns the pch off for it).
#include "SameEyeFit_Vk.h"
#include "PerEyeFitShader_Vk.h"

#include <cstring>

namespace SameEyeFit_Vk
{
namespace
{
constexpr VkDeviceSize kFitsBytes = 20 * 16;                           // slots 0-3, 5 float4 each
constexpr VkDeviceSize kPartialsBytes = (VkDeviceSize) kPartialWords * 4 + kDumpBytes; // as the D3D12 buffer
constexpr VkDeviceSize kFitReadSlot = 256;
constexpr VkDeviceSize kFitReadBytes = 15 * 16; // slots 0-2, as D3D12 copies

void Barrier(VkCommandBuffer cmd, VkPipelineStageFlags srcStage, VkAccessFlags srcAccess, VkPipelineStageFlags dstStage,
             VkAccessFlags dstAccess)
{
    VkMemoryBarrier b { VK_STRUCTURE_TYPE_MEMORY_BARRIER };
    b.srcAccessMask = srcAccess;
    b.dstAccessMask = dstAccess;
    vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 1, &b, 0, nullptr, 0, nullptr);
}

void ImageBarrier(VkCommandBuffer cmd, VkImage image, VkImageLayout from, VkPipelineStageFlags srcStage,
                  VkAccessFlags srcAccess, VkPipelineStageFlags dstStage, VkAccessFlags dstAccess,
                  VkImageLayout to = VK_IMAGE_LAYOUT_GENERAL)
{
    VkImageMemoryBarrier b { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
    b.srcAccessMask = srcAccess;
    b.dstAccessMask = dstAccess;
    b.oldLayout = from;
    b.newLayout = to;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = image;
    b.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &b);
}
} // namespace

bool Fitter::MakeBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags props, VkBuffer& buffer,
                        VkDeviceMemory& memory, void** mapped, std::string& error)
{
    VkBufferCreateInfo bi { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bi.size = size;
    bi.usage = usage;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(_device, &bi, nullptr, &buffer) != VK_SUCCESS)
    {
        error = "vkCreateBuffer";
        return false;
    }
    VkMemoryRequirements req {};
    vkGetBufferMemoryRequirements(_device, buffer, &req);
    VkPhysicalDeviceMemoryProperties mp {};
    vkGetPhysicalDeviceMemoryProperties(_physical, &mp);
    uint32_t type = UINT32_MAX;
    for (uint32_t i = 0; i < mp.memoryTypeCount && type == UINT32_MAX; ++i)
        if ((req.memoryTypeBits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & props) == props)
            type = i;
    if (type == UINT32_MAX)
    {
        error = "no memory type for a buffer";
        return false;
    }
    VkMemoryAllocateInfo ai { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = type;
    if (vkAllocateMemory(_device, &ai, nullptr, &memory) != VK_SUCCESS ||
        vkBindBufferMemory(_device, buffer, memory, 0) != VK_SUCCESS)
    {
        error = "vkAllocateMemory / vkBindBufferMemory";
        return false;
    }
    if (mapped != nullptr && vkMapMemory(_device, memory, 0, VK_WHOLE_SIZE, 0, mapped) != VK_SUCCESS)
    {
        error = "vkMapMemory";
        return false;
    }
    return true;
}

bool Fitter::Init(VkDevice device, VkPhysicalDevice physicalDevice, std::string& error)
{
    _device = device;
    _physical = physicalDevice;

    // One set for both shaders: 0 depth, 1 motion (sampled), 2 the fits read (compose), 3 composed, 4 mask (storage
    // images), 5 the fits written (fit), 6 the partials. A binding a pipeline does not use may stay unwritten.
    VkDescriptorSetLayoutBinding b[7] {};
    const VkDescriptorType types[7] = { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                                        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER };
    for (uint32_t i = 0; i < 7; ++i)
    {
        b[i].binding = i;
        b[i].descriptorType = types[i];
        b[i].descriptorCount = 1;
        b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo sl { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    sl.bindingCount = 7;
    sl.pBindings = b;
    if (vkCreateDescriptorSetLayout(_device, &sl, nullptr, &_setLayout) != VK_SUCCESS)
    {
        error = "vkCreateDescriptorSetLayout";
        return false;
    }
    VkPushConstantRange pc { VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Constants) };
    VkPipelineLayoutCreateInfo pl { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    pl.setLayoutCount = 1;
    pl.pSetLayouts = &_setLayout;
    pl.pushConstantRangeCount = 1;
    pl.pPushConstantRanges = &pc;
    if (vkCreatePipelineLayout(_device, &pl, nullptr, &_layout) != VK_SUCCESS)
    {
        error = "vkCreatePipelineLayout";
        return false;
    }
    auto pipeline = [&](const uint32_t* code, size_t bytes, VkPipeline& out, const char* what)
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
        ci.layout = _layout;
        const VkResult r = vkCreateComputePipelines(_device, VK_NULL_HANDLE, 1, &ci, nullptr, &out);
        vkDestroyShaderModule(_device, module, nullptr);
        if (r != VK_SUCCESS)
        {
            error = std::string("vkCreateComputePipelines (") + what + ")";
            return false;
        }
        return true;
    };
    if (!pipeline(kFitShaderSpv, sizeof(kFitShaderSpv), _fitPipeline, "fit") ||
        !pipeline(kComposeShaderSpv, sizeof(kComposeShaderSpv), _composePipeline, "compose"))
        return false;

    VkDescriptorPoolSize sizes[3] = { { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 2 * kRing },
                                      { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3 * kRing },
                                      { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2 * kRing } };
    VkDescriptorPoolCreateInfo dp { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    dp.maxSets = kRing;
    dp.poolSizeCount = 3;
    dp.pPoolSizes = sizes;
    if (vkCreateDescriptorPool(_device, &dp, nullptr, &_pool) != VK_SUCCESS)
    {
        error = "vkCreateDescriptorPool";
        return false;
    }
    VkDescriptorSetLayout layouts[kRing];
    for (auto& l : layouts)
        l = _setLayout;
    VkDescriptorSetAllocateInfo da { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    da.descriptorPool = _pool;
    da.descriptorSetCount = kRing;
    da.pSetLayouts = layouts;
    if (vkAllocateDescriptorSets(_device, &da, _sets) != VK_SUCCESS)
    {
        error = "vkAllocateDescriptorSets";
        return false;
    }

    const auto storage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                         VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    const auto host = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    if (!MakeBuffer(kFitsBytes, storage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, _fits, _fitsMem, nullptr, error) ||
        !MakeBuffer(kPartialsBytes, storage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, _partials, _partialsMem, nullptr,
                    error) ||
        !MakeBuffer(kRing * kFitReadSlot, VK_BUFFER_USAGE_TRANSFER_DST_BIT, host, _fitRead, _fitReadMem, &_fitReadMapped,
                    error) ||
        !MakeBuffer((VkDeviceSize) kRing * kDumpStride, VK_BUFFER_USAGE_TRANSFER_DST_BIT, host, _dumpRead, _dumpReadMem,
                    &_dumpReadMapped, error))
        return false;
    std::memset(_fitReadMapped, 0xFF, kRing * kFitReadSlot); // no stamp matches before the GPU writes one
    std::memset(_dumpReadMapped, 0xFF, (size_t) kRing * kDumpStride);
    _ready = true;
    return true;
}

void Fitter::Destroy()
{
    if (_device == VK_NULL_HANDLE)
        return;
    VkBuffer buffers[] = { _fits, _partials, _fitRead, _dumpRead };
    VkDeviceMemory memories[] = { _fitsMem, _partialsMem, _fitReadMem, _dumpReadMem };
    for (auto b : buffers)
        if (b != VK_NULL_HANDLE)
            vkDestroyBuffer(_device, b, nullptr);
    for (auto m : memories)
        if (m != VK_NULL_HANDLE)
            vkFreeMemory(_device, m, nullptr);
    if (_pool != VK_NULL_HANDLE)
        vkDestroyDescriptorPool(_device, _pool, nullptr);
    if (_fitPipeline != VK_NULL_HANDLE)
        vkDestroyPipeline(_device, _fitPipeline, nullptr);
    if (_composePipeline != VK_NULL_HANDLE)
        vkDestroyPipeline(_device, _composePipeline, nullptr);
    if (_layout != VK_NULL_HANDLE)
        vkDestroyPipelineLayout(_device, _layout, nullptr);
    if (_setLayout != VK_NULL_HANDLE)
        vkDestroyDescriptorSetLayout(_device, _setLayout, nullptr);
    *this = Fitter();
}

void Fitter::Record(VkCommandBuffer cmd, uint32_t slot, const Inputs& in, Constants c)
{
    slot %= kRing;
    const VkDescriptorSet set = _sets[slot];

    VkDescriptorImageInfo depth { VK_NULL_HANDLE, in.depth, in.depthLayout };
    VkDescriptorImageInfo motion { VK_NULL_HANDLE, in.motion, in.motionLayout };
    VkDescriptorImageInfo composed { VK_NULL_HANDLE, in.composed, VK_IMAGE_LAYOUT_GENERAL };
    VkDescriptorImageInfo mask { VK_NULL_HANDLE, in.mask, VK_IMAGE_LAYOUT_GENERAL };
    VkDescriptorBufferInfo fits { _fits, 0, kFitsBytes };
    VkDescriptorBufferInfo partials { _partials, 0, kPartialsBytes };
    VkWriteDescriptorSet w[7] {};
    uint32_t n = 0;
    auto image = [&](uint32_t binding, VkDescriptorType type, VkDescriptorImageInfo* info)
    {
        w[n] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        w[n].dstSet = set;
        w[n].dstBinding = binding;
        w[n].descriptorCount = 1;
        w[n].descriptorType = type;
        w[n].pImageInfo = info;
        ++n;
    };
    auto buffer = [&](uint32_t binding, VkDescriptorBufferInfo* info)
    {
        w[n] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        w[n].dstSet = set;
        w[n].dstBinding = binding;
        w[n].descriptorCount = 1;
        w[n].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w[n].pBufferInfo = info;
        ++n;
    };
    image(0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &depth);
    image(1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &motion);
    buffer(2, &fits);
    buffer(5, &fits);
    buffer(6, &partials);
    const bool compose = in.composed != VK_NULL_HANDLE && in.mask != VK_NULL_HANDLE;
    if (compose)
    {
        image(3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &composed);
        image(4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &mask);
    }
    vkUpdateDescriptorSets(_device, n, w, 0, nullptr);

    Prepare(cmd);
    // Our buffers after the previous evaluate's dispatches and readback copies (earlier submissions included).
    Barrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _fitPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _layout, 0, 1, &set, 0, nullptr);
    c.round = 100;
    vkCmdPushConstants(cmd, _layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(c), &c);
    vkCmdDispatch(cmd, 1, 1, 1);
    for (uint32_t round = 0; round <= 4; ++round)
    {
        Barrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
        c.round = round;
        vkCmdPushConstants(cmd, _layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(c), &c);
        vkCmdDispatch(cmd, 64, 1, 1);
    }
    Barrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT);

    if (compose)
    {
        // Ours. The first use throws away whatever was there; after it, the composed motion comes back from DLSS's
        // SHADER_READ_ONLY_OPTIMAL and the mask stays in GENERAL.
        const bool firstC = _knownComposed != in.composedImage, firstM = _knownMask != in.maskImage;
        ImageBarrier(cmd, in.composedImage, firstC ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, firstC ? 0 : VK_ACCESS_SHADER_READ_BIT,
                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT);
        ImageBarrier(cmd, in.maskImage, firstM ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_GENERAL,
                     VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, firstM ? 0 : VK_ACCESS_SHADER_READ_BIT,
                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT);
        _knownComposed = in.composedImage;
        _knownMask = in.maskImage;
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _composePipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _layout, 0, 1, &set, 0, nullptr);
        vkCmdPushConstants(cmd, _layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(c), &c);
        vkCmdDispatch(cmd, (in.width + 7) / 8, (in.height + 7) / 8, 1);
        // DLSS reads the composed motion next (any stage), in its inputs' layout.
        ImageBarrier(cmd, in.composedImage, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                     VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_SHADER_READ_BIT,
                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        ImageBarrier(cmd, in.maskImage, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                     VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_SHADER_READ_BIT);
    }

    VkBufferCopy fitCopy { 0, slot * kFitReadSlot, kFitReadBytes };
    vkCmdCopyBuffer(cmd, _fits, _fitRead, 1, &fitCopy);
    if (c.keepStale & 8u)
    {
        VkBufferCopy dumpCopy { (VkDeviceSize) kPartialWords * 4, (VkDeviceSize) slot * kDumpStride, kDumpBytes };
        vkCmdCopyBuffer(cmd, _partials, _dumpRead, 1, &dumpCopy);
    }
    Barrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_HOST_BIT,
            VK_ACCESS_HOST_READ_BIT);
}

void Fitter::Prepare(VkCommandBuffer cmd)
{
    if (_buffersCleared)
        return;
    // D3D12's committed buffers start zeroed and the fit relies on it (no stored fit; the count at 0).
    vkCmdFillBuffer(cmd, _fits, 0, VK_WHOLE_SIZE, 0);
    vkCmdFillBuffer(cmd, _partials, 0, VK_WHOLE_SIZE, 0);
    Barrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
    _buffersCleared = true;
}

const float* Fitter::FitSlot(uint32_t slot) const
{
    return (const float*) ((const uint8_t*) _fitReadMapped + (slot % kRing) * kFitReadSlot);
}

const uint8_t* Fitter::DumpSlot(uint32_t slot) const
{
    return (const uint8_t*) _dumpReadMapped + (size_t) (slot % kRing) * kDumpStride;
}
} // namespace SameEyeFit_Vk
