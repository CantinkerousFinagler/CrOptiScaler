#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <string>

// The same-eye fit and compose of PerEyeDlss_Dx12, on Vulkan:
// the very shaders (PerEyeFitShader_Dx12.h compiled to SPIR-V with VK_MODE, PerEyeFitShader_Vk.h), the same sequence
// of dispatches (INIT with Round 100 in one group, rounds 0-4 in 64 groups each, the compose over the motion's size)
// and the same buffers (the fits: 20 float4; the partial sums, the count and the dump). Pure Vulkan, no OptiScaler
// code, so tools/xr_observer/test/vk_fit_test runs this very recorder on a real device.
//
// Rules (as D3D12): nothing of the game's is barriered -- its depth and motion are read as DLSS reads them,
// in the layout the caller names (DLSS inputs: SHADER_READ_ONLY_OPTIMAL, as DlssNrFeature_Vk assumes); our own buffers
// and images get every barrier. The mask stays in GENERAL from its first use; the composed motion is written in GENERAL
// and left in SHADER_READ_ONLY_OPTIMAL for DLSS (its inputs' layout), moved back to GENERAL by the next compose.
namespace SameEyeFit_Vk
{
constexpr uint32_t kRing = 8;                     // fit readback slots (read kRing evaluates later, by stamp)
constexpr uint32_t kPartialWords = 64 * 80;       // the partial sums; the count is the next word
constexpr uint32_t kDumpBytes = (4 + 4096 * 4) * 4; // the count, the stamp, 2 spare, then 4096 samples x 4 words
constexpr uint32_t kDumpStride = 65792;            // a dump ring slot

struct Constants // the shaders' 48 bytes of push constants, in their order
{
    float mvScale[2];
    uint32_t cur, prev;
    float knee0, knee1, gain;
    uint32_t radius, stamp, round, keepStale, linearDepth;
};
static_assert(sizeof(Constants) == 48);

struct Inputs
{
    VkImageView depth = VK_NULL_HANDLE; // the game's, sampled (a depth-aspect view)
    VkImageLayout depthLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkImageView motion = VK_NULL_HANDLE; // the game's R16G16 vectors, sampled
    VkImageLayout motionLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    // The compose's outputs (ours, R16G16_SFLOAT and R8_UNORM storage images of the motion's size), or null for a fit
    // without the compose. Their images are named for the barriers.
    VkImage composedImage = VK_NULL_HANDLE, maskImage = VK_NULL_HANDLE;
    VkImageView composed = VK_NULL_HANDLE, mask = VK_NULL_HANDLE;
    uint32_t width = 0, height = 0; // the compose's size (the motion's)
};

class Fitter
{
  public:
    // Pipelines, layouts, a descriptor set per ring slot, the buffers and the mapped readbacks. On failure `error` says
    // which step; nothing is half made (Destroy is safe).
    bool Init(VkDevice device, VkPhysicalDevice physicalDevice, std::string& error);
    void Destroy(); // only once the GPU is done with everything recorded
    bool Ready() const { return _ready; }

    // Records the fit for ring slot `slot` (the descriptors of that slot are rewritten: it must not be in flight --
    // kRing evaluates apart), the compose when `in.composed` is set, the fit's readback and (with KeepStale bit 3) the
    // dump's. The caller's constants: round is set here.
    void Record(VkCommandBuffer cmd, uint32_t slot, const Inputs& in, Constants constants);

    // The one-time clear of our buffers (D3D12's start zeroed): recorded by the first Record, or earlier by a caller that
    // writes into the fits buffer before any fit (the test's starts).
    void Prepare(VkCommandBuffer cmd);

    // The fit readback of a slot: 15 float4 (the fits buffer's slots 0-2), host coherent. The DLL checks the stamp
    // (row 4's w of slot 2) before believing it.
    const float* FitSlot(uint32_t slot) const;
    // The dump's readback of a slot: kDumpBytes (the count, the stamp, 2 spare, 4096 x 4 words).
    const uint8_t* DumpSlot(uint32_t slot) const;
    // The fits buffer itself (the test writes starts into it).
    VkBuffer FitsBuffer() const { return _fits; }

  private:
    bool MakeBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags props, VkBuffer& buffer,
                    VkDeviceMemory& memory, void** mapped, std::string& error);

    VkDevice _device = VK_NULL_HANDLE;
    VkPhysicalDevice _physical = VK_NULL_HANDLE;
    bool _ready = false;
    VkDescriptorSetLayout _setLayout = VK_NULL_HANDLE;
    VkPipelineLayout _layout = VK_NULL_HANDLE;
    VkPipeline _fitPipeline = VK_NULL_HANDLE, _composePipeline = VK_NULL_HANDLE;
    VkDescriptorPool _pool = VK_NULL_HANDLE;
    VkDescriptorSet _sets[kRing] = {};
    VkBuffer _fits = VK_NULL_HANDLE, _partials = VK_NULL_HANDLE, _fitRead = VK_NULL_HANDLE, _dumpRead = VK_NULL_HANDLE;
    VkDeviceMemory _fitsMem = VK_NULL_HANDLE, _partialsMem = VK_NULL_HANDLE, _fitReadMem = VK_NULL_HANDLE,
                   _dumpReadMem = VK_NULL_HANDLE;
    void* _fitReadMapped = nullptr;
    void* _dumpReadMapped = nullptr;
    bool _buffersCleared = false;
    VkImage _knownComposed = VK_NULL_HANDLE, _knownMask = VK_NULL_HANDLE; // already initialised by a compose
};
} // namespace SameEyeFit_Vk
