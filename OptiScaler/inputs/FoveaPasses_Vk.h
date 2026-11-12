#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <string>

// The foveal crop's two compute passes of PerEyeDlss_Dx12, on Vulkan:
// COMPOSITE (our crop blended over the box of the game's output with a feathered edge) and ENLARGE (the
// cheap periphery's render-size result over the whole eye, Catmull-Rom). The very shaders (FoveaShader_Dx12.h compiled
// to SPIR-V with VK_MODE, FoveaShader_Vk.h). Pure Vulkan, no OptiScaler code, so tools/xr_observer/test/vk_fovea_test
// runs this very recorder on a real device.
//
// Layouts: everything these passes touch is in VK_IMAGE_LAYOUT_GENERAL -- the game's output (a DLSS output, written in
// GENERAL), our crop image and our periphery image (DLSS outputs of ours, GENERAL) -- so no pass changes a layout. The
// storage images are declared rgba16f in the shaders (the output format DLSS writes in the Vulkan game it was written for); the caller checks the
// game's output format is VK_FORMAT_R16G16B16A16_SFLOAT before using these.
namespace FoveaPasses_Vk
{
constexpr uint32_t kRing = 8; // descriptor sets per pass, rewritten per record (a slot must not be in flight)

// THE GATHER COPY: the box (sx, sy, w x h) of `src` copied to (0, 0) of `dst`, both in GENERAL (an upscaler's inputs
// and outputs are, by DLSS's Vulkan contract), one aspect (colour or depth), same format. A memory barrier before
// (whatever wrote the source, and whatever read the destination last, made visible to the copy) and after (the copy
// visible to the DLSS evaluate that reads it). No pipeline: a free function.
void CopyBox(VkCommandBuffer cmd, VkImage src, VkImageAspectFlags aspect, uint32_t sx, uint32_t sy, VkImage dst,
             uint32_t w, uint32_t h);

struct CompositeConstants // the shader's push constants, in its order (32 bytes)
{
    uint32_t baseX, baseY; // the box in the output
    uint32_t sizeX, sizeY;
    uint32_t feather; // output px
    uint32_t outline; // 1 = a magenta frame instead of the blend (F8)
    uint32_t unused0, unused1;
};
static_assert(sizeof(CompositeConstants) == 32);

struct EnlargeConstants // (32 bytes)
{
    uint32_t outW, outH;
    uint32_t srcW, srcH;
    uint32_t unused[4];
};
static_assert(sizeof(EnlargeConstants) == 32);

struct TaaConstants // the TAA shader's push constants, in its order (64 bytes)
{
    uint32_t outSize[2], renderSize[2], colourBase[2], motionBase[2], depthBase[2];
    float jitter[2], mvScale[2];
    float blend;
    uint32_t reset;
};
static_assert(sizeof(TaaConstants) == 64);

class Passes
{
  public:
    // Pipelines, layouts, a descriptor set per ring slot for each pass, a linear clamp sampler. On failure `error`
    // says which step; nothing is half made (Destroy is safe).
    bool Init(VkDevice device, VkPhysicalDevice physicalDevice, std::string& error);
    void Destroy(); // only once the GPU is done with everything recorded
    bool Ready() const { return _ready; }

    // COMPOSITE: `crop` (ours, ow x oh, GENERAL, written by the crop's DLSS) over `output` (the game's, GENERAL, written
    // by the game's DLSS) at the box. Barriers first (the two DLSS writes made visible to our dispatch) and after
    // (our write visible to whoever reads the output next). The descriptors of `slot` are rewritten.
    void Composite(VkCommandBuffer cmd, uint32_t slot, VkImageView crop, VkImageView output,
                   const CompositeConstants& c);

    // ENLARGE: `src` (ours, srcW x srcH, GENERAL) sampled over the whole of `output` (outW x outH, GENERAL).
    void Enlarge(VkCommandBuffer cmd, uint32_t slot, VkImageView src, VkImageView output, const EnlargeConstants& c);

    // TAA (the own TAA periphery): the game's colour, motion and depth (GENERAL, read by Load at the bases), this
    // eye's history (ours, outSize, GENERAL, sampled) -> `output` (outSize, GENERAL, rgba16f storage: the game's
    // output or our mid image) and `historyOut` (ours, outSize). Barriers before and after.
    void Taa(VkCommandBuffer cmd, uint32_t slot, VkImageView colour, VkImageView motion, VkImageView depth,
             VkImageView history, VkImageView output, VkImageView historyOut, const TaaConstants& c);

  private:
    VkDevice _device = VK_NULL_HANDLE;
    VkPhysicalDevice _physical = VK_NULL_HANDLE;
    VkDescriptorSetLayout _compositeSetLayout = VK_NULL_HANDLE, _enlargeSetLayout = VK_NULL_HANDLE,
                          _taaSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout _compositeLayout = VK_NULL_HANDLE, _enlargeLayout = VK_NULL_HANDLE, _taaLayout = VK_NULL_HANDLE;
    VkPipeline _compositePipeline = VK_NULL_HANDLE, _enlargePipeline = VK_NULL_HANDLE, _taaPipeline = VK_NULL_HANDLE;
    VkDescriptorPool _pool = VK_NULL_HANDLE;
    VkDescriptorSet _compositeSets[kRing] = {}, _enlargeSets[kRing] = {}, _taaSets[kRing] = {};
    VkSampler _sampler = VK_NULL_HANDLE;
    bool _ready = false;
};
} // namespace FoveaPasses_Vk
