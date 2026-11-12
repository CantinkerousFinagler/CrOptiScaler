#include "pch.h"

#include "Config.h"
#include "Logger.h"
#include "State.h"

#include "FoveaCrop_Vk.h"
#include "FoveaBox.h"
#include "FoveaPasses_Vk.h"
#include "NvngxVkOwn.h"
#include "PerEyeDlss_Vk.h"
#include "VkCreateWorker.h"
#include "FrameTime_Vk.h"
#include <dlssnr/DlssNr.h>
#include <upscalers/IFeature_Vk.h>

#include <nvsdk_ngx_vk.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <memory>
#include <cstring>
#include <format>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace FoveaCrop_Vk
{
namespace
{
constexpr unsigned int kOddEyeKey = 0x40000000u; // shared routing: the odd Present's crop (as PerEyeDlss_Dx12)
constexpr uint64_t kWindow = 600;                 // evaluates per summary
constexpr VkFormat kShaderFormat = VK_FORMAT_R16G16B16A16_SFLOAT; // the storage images' declared format

std::mutex g_mutex;
bool g_foveaOn = false, g_outline = false, g_foveaFailed = false;
bool g_periphFailed = false; // the periphery could not be set up or made: the full pass runs this session
// Read by DLSSFeature_Vk INSIDE our own feature's creation on the thread that creates it (the game's inline under
// g_mutex, or the worker's): per thread, lock-free (a std::mutex locked twice
// throws).
thread_local int t_presetForCreate = 0;

VkDevice g_device = VK_NULL_HANDLE;
VkPhysicalDevice g_physical = VK_NULL_HANDLE;
FoveaPasses_Vk::Passes g_passes;
bool g_passesTried = false;
uint32_t g_slot = 0;
uint64_t g_evals = 0;
std::atomic<uint64_t> g_presents { 0 };

// Our crop image: the crop's DLSS output (ow x oh, the game's output format), in GENERAL from its first use.
struct OwnImage
{
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    NVSDK_NGX_Resource_VK ngx {};
    uint32_t w = 0, h = 0;
    VkFormat format = VK_FORMAT_UNDEFINED;
    bool laidOut = false; // UNDEFINED -> GENERAL recorded
};
OwnImage g_cropImage;
OwnImage g_periphImage; // the periphery's DLSS output (render size, the game's output format)
// THE GATHER COPY ([Inputs] UpscalerFoveaCopyInputs, default on; the D3D12 crop's default too): the crop's colour,
// depth, motion and (UpscalerFoveaCropMask = 1) the game's bias mask copied at the box into crop-sized images of the
// game's own formats, so the crop's DLSS reads them like any ordinary call at base (0, 0). Off: the subrect path (the
// game's images at base + box), on which presets K and J turned pixelated (the whole crop
// shimmered at K). Both eyes share the images: the evaluates are in queue order and the copy is barriered.
OwnImage g_gatherColour, g_gatherDepth, g_gatherMotion, g_gatherMask;
// THE OWN TAA PERIPHERY ([Inputs] UpscalerFoveaPeripheryTaa): in the place of the periphery's DLSS call, our TAA
// (FoveaPasses_Vk::Taa, the D3D12 shader) over the whole eye at UpscalerFoveaTaaScale of the eye (below 1: into
// g_taaMid, then ENLARGEd), the crop over it. A history per eye key (two images, read one, write the other), used
// only when this key's last TAA evaluate was `stride` Presents back: 2 per eye (per-eye routing, or shared with the
// eye by parity under R.E.A.L. VR's DLSS pass), 1 shared without the parity.
struct TaaHistory
{
    OwnImage tex[2];
    uint32_t read = 0;
    uint64_t lastPresent = UINT64_MAX;
};
std::unordered_map<unsigned int, TaaHistory> g_taa;
OwnImage g_taaMid;
uint64_t g_taaEvals = 0, g_taaResets = 0, g_taaSkips = 0, g_lastTaaEvals = 0, g_lastTaaResets = 0, g_lastTaaSkips = 0;
bool g_loggedTaaFirst = false, g_loggedTaaSkip = false, g_loggedTaaFail = false, g_taaFailed = false;
bool g_loggedGather = false, g_loggedGatherFail = false;
uint64_t g_gathered = 0, g_lastGathered = 0; // evaluates whose inputs were copied, per window
std::vector<std::pair<OwnImage, uint64_t>> g_retiredImages; // destroyed 32 Presents later

// One crop per eye key: the owner's handle, or the owner's handle | kOddEyeKey in shared routing on odd Presents.
struct Crop
{
    unsigned int feature = 0; // the crop feature's id in VkContexts
    unsigned int owner = 0;
    FoveaBox::Rect rect {};
    uint32_t eyeR = 0, eyeO = 0; // the render and output widths it was made for (a size change = made for another)
    uint32_t eyeRh = 0, eyeOh = 0;
    int preset = 0;
    bool reset = true;
    NVSDK_NGX_Parameter* block = nullptr; // the worker's creation block, if made there (destroyed with the entry)
};
std::unordered_map<unsigned int, Crop> g_crops;

// One periphery per eye key (as the crops): a feature at the render size in and out.
struct Periph
{
    unsigned int feature = 0;
    unsigned int owner = 0;
    uint32_t rw = 0, rh = 0; // made for this render size
    int preset = 0;
    bool reset = true;
    NVSDK_NGX_Parameter* block = nullptr; // the worker's creation block, if made there (destroyed with the entry)
};
std::unordered_map<unsigned int, Periph> g_periphs;

// Features of our own being made on the worker (VkCreateWorker), one per eye key and kind: the full pass (or no
// crop) runs until the feature is ready. The block is the one the worker's create read (sizes, flags, preset);
// destroyed with the feature's entry.
struct Pending
{
    unsigned int id = 0, owner = 0;
    NVSDK_NGX_Parameter* block = nullptr;
    FoveaBox::Rect rect {};
    uint32_t R = 0, Rh = 0, O = 0, Oh = 0;
    int preset = 0;
    std::chrono::steady_clock::time_point asked;
};
std::unordered_map<unsigned int, Pending> g_pendingCrops, g_pendingPeriphs;
// Re-asks per key ("made for a box since changed"): past kMaxReasks the key fails for the session -- a loop of
// make / retire / ask again must never run free (859 times in 20 s crashed a game).
constexpr int kMaxReasks = 3;
std::unordered_map<unsigned int, int> g_reasksCrops, g_reasksPeriphs;
// What the worker finished (its thread writes under g_doneMutex); Collect() moves them under g_mutex.
struct Done
{
    unsigned int id = 0;
    std::unique_ptr<IFeature_Vk> feature;
    bool ok = false;
    double ms = 0.0;
};
std::mutex g_doneMutex;
std::vector<Done> g_doneByWorker;
std::unordered_map<unsigned int, Done> g_ready; // by id, under g_mutex
std::vector<unsigned int> g_orphans;            // asked for, owner gone meanwhile: retired when ready
uint64_t g_skippedMaking = 0, g_periphMaking = 0; // per window: waited for the worker
std::unordered_map<unsigned int, bool> g_mainSkipped; // game features skipped for a periphery: reset when they run again

NVSDK_NGX_Parameter* g_cropParams = nullptr;
NVSDK_NGX_Parameter* g_periphParams = nullptr;

// Per window, and the last whole window (the overlay shows it).
uint64_t g_drawn = 0, g_skippedSize = 0, g_skippedPlan = 0, g_skippedParams = 0, g_failed = 0, g_windowStart = 0;
uint64_t g_lastWindowEvals = 0, g_lastDrawn = 0, g_lastSkipped = 0, g_lastFailed = 0;
uint64_t g_periphDrawn = 0, g_periphFallbacks = 0, g_lastPeriphDrawn = 0, g_lastPeriphFallbacks = 0;
// Presents per second, per window of kPresentWindow Presents (the Vulkan path's only frame-rate reading).
constexpr uint64_t kPresentWindow = 600;
std::chrono::steady_clock::time_point g_presentWindowStart;
uint64_t g_presentWindowEvals = 0;
double g_lastPresentsPerSec = 0.0, g_lastEvalsPerPresent = 0.0;
FrameTime_Vk::Window g_lastGpu; // the GPU times of the last window (FrameTime_Vk)
bool g_loggedPeriphFirst = false, g_loggedPeriphFail = false, g_loggedPeriphCreateFail = false,
     g_loggedPeriphImageFail = false, g_loggedPeriphSkip = false;
// The box last planned and the eye it was planned for (the overlay shows it).
FoveaBox::Rect g_statusRect {};
uint32_t g_statusEyeW = 0, g_statusEyeH = 0;
bool g_startApplied = false; // [Inputs] UpscalerFoveaStart read once, at the first evaluate
bool g_loggedPlan = false, g_loggedFirst = false, g_loggedFormat = false, g_loggedNoOutput = false,
     g_loggedCopy = false, g_loggedMissing = false, g_loggedCreateFail = false, g_loggedEvalFail = false,
     g_loggedOtherDevice = false, g_loggedImageFail = false, g_loggedPasses = false;
FoveaBox::Rect g_loggedRect {};

void LogOnce(bool& logged, const std::string& text)
{
    if (logged)
        return;
    logged = true;
    LOG_WARN("Per-eye DLSS VK: {}", text);
}

uint32_t MemoryType(uint32_t bits, VkMemoryPropertyFlags props)
{
    VkPhysicalDeviceMemoryProperties mp {};
    vkGetPhysicalDeviceMemoryProperties(g_physical, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
        if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & props) == props)
            return i;
    return UINT32_MAX;
}

void DestroyImage(OwnImage& im)
{
    if (im.view != VK_NULL_HANDLE)
        vkDestroyImageView(g_device, im.view, nullptr);
    if (im.image != VK_NULL_HANDLE)
        vkDestroyImage(g_device, im.image, nullptr);
    if (im.memory != VK_NULL_HANDLE)
        vkFreeMemory(g_device, im.memory, nullptr);
    im = {};
}

// `aspect`: colour for our outputs; a gathered depth is a depth image. Storage usage only where the format offers
// it (a depth format does not).
bool MakeImage(OwnImage& im, uint32_t w, uint32_t h, VkFormat format,
               VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT)
{
    VkFormatProperties fp {};
    vkGetPhysicalDeviceFormatProperties(g_physical, format, &fp);
    VkImageCreateInfo ci { VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
    ci.imageType = VK_IMAGE_TYPE_2D;
    ci.format = format;
    ci.extent = { w, h, 1 };
    ci.mipLevels = 1;
    ci.arrayLayers = 1;
    ci.samples = VK_SAMPLE_COUNT_1_BIT;
    ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    ci.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    if (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT)
        ci.usage |= VK_IMAGE_USAGE_STORAGE_BIT;
    ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (vkCreateImage(g_device, &ci, nullptr, &im.image) != VK_SUCCESS)
        return false;
    VkMemoryRequirements req {};
    vkGetImageMemoryRequirements(g_device, im.image, &req);
    VkMemoryAllocateInfo ai { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = MemoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (ai.memoryTypeIndex == UINT32_MAX || vkAllocateMemory(g_device, &ai, nullptr, &im.memory) != VK_SUCCESS ||
        vkBindImageMemory(g_device, im.image, im.memory, 0) != VK_SUCCESS)
    {
        DestroyImage(im);
        return false;
    }
    VkImageViewCreateInfo vi { VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
    vi.image = im.image;
    vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vi.format = format;
    vi.subresourceRange = { aspect, 0, 1, 0, 1 };
    if (vkCreateImageView(g_device, &vi, nullptr, &im.view) != VK_SUCCESS)
    {
        DestroyImage(im);
        return false;
    }
    im.w = w;
    im.h = h;
    im.format = format;
    im.laidOut = false;
    im.ngx = {};
    im.ngx.Type = NVSDK_NGX_RESOURCE_VK_TYPE_VK_IMAGEVIEW;
    im.ngx.Resource.ImageViewInfo.ImageView = im.view;
    im.ngx.Resource.ImageViewInfo.Image = im.image;
    im.ngx.Resource.ImageViewInfo.SubresourceRange = vi.subresourceRange;
    im.ngx.Resource.ImageViewInfo.Format = format;
    im.ngx.Resource.ImageViewInfo.Width = w;
    im.ngx.Resource.ImageViewInfo.Height = h;
    im.ngx.ReadWrite = (ci.usage & VK_IMAGE_USAGE_STORAGE_BIT) != 0;
    return true;
}

// Our image into GENERAL once (its first use); it stays there: DLSS writes it as a storage image, we read it.
void LayOut(VkCommandBuffer cmd, OwnImage& im)
{
    if (im.laidOut || im.image == VK_NULL_HANDLE)
        return;
    VkImageMemoryBarrier b { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
    b.srcAccessMask = 0;
    b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = im.image;
    b.subresourceRange = im.ngx.Resource.ImageViewInfo.SubresourceRange;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &b);
    im.laidOut = true;
}

void RetireImage(OwnImage& im)
{
    if (im.image == VK_NULL_HANDLE)
        return;
    g_retiredImages.push_back({ im, g_presents.load() });
    im = {};
}

// `im` at w x h in `format` (remade when it differs; the old one retired). False = it could not be made (logged
// once under `logged`).
bool EnsureImage(OwnImage& im, uint32_t w, uint32_t h, VkFormat format, const char* what, bool& logged,
                 VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT)
{
    if (im.image != VK_NULL_HANDLE && im.w == w && im.h == h && im.format == format)
        return true;
    RetireImage(im);
    if (!MakeImage(im, w, h, format, aspect))
    {
        LogOnce(logged, std::format("{} off: our {}x{} image could not be made", what, w, h));
        return false;
    }
    LOG_INFO("Per-eye DLSS VK: {} output image {}x{} fmt {} made", what, w, h, (int) format);
    return true;
}

bool EnsureCropImage(uint32_t w, uint32_t h, VkFormat format)
{
    if (EnsureImage(g_cropImage, w, h, format, "foveal crop", g_loggedImageFail))
        return true;
    g_foveaFailed = true;
    return false;
}

// The keys DLSS reads, by the type they are handed in -- PerEyeDlss_Dx12's lists (kCropUintKeys, kCropFloatKeys,
// kCropResourceKeys, kCropPointerKeys, kGuides), the resources as void* here (NVSDK_NGX_Resource_VK*). Only keys the
// game's block HOLDS are copied.
constexpr const char* kUintKeys[] = {
    NVSDK_NGX_Parameter_Width, NVSDK_NGX_Parameter_Height, NVSDK_NGX_Parameter_OutWidth,
    NVSDK_NGX_Parameter_OutHeight, NVSDK_NGX_Parameter_Reset, NVSDK_NGX_Parameter_PerfQualityValue,
    NVSDK_NGX_Parameter_CreationNodeMask, NVSDK_NGX_Parameter_VisibilityNodeMask,
    NVSDK_NGX_Parameter_DLSS_Feature_Create_Flags, NVSDK_NGX_Parameter_DLSS_Enable_Output_Subrects,
    NVSDK_NGX_Parameter_DLSS_Checkerboard_Jitter_Hack, NVSDK_NGX_Parameter_TonemapperType,
    NVSDK_NGX_Parameter_FreeMemOnReleaseFeature, NVSDK_NGX_Parameter_DLSS_Input_Color_Subrect_Base_X,
    NVSDK_NGX_Parameter_DLSS_Input_Color_Subrect_Base_Y, NVSDK_NGX_Parameter_DLSS_Input_Depth_Subrect_Base_X,
    NVSDK_NGX_Parameter_DLSS_Input_Depth_Subrect_Base_Y, NVSDK_NGX_Parameter_DLSS_Input_MV_SubrectBase_X,
    NVSDK_NGX_Parameter_DLSS_Input_MV_SubrectBase_Y, NVSDK_NGX_Parameter_DLSS_Input_Translucency_SubrectBase_X,
    NVSDK_NGX_Parameter_DLSS_Input_Translucency_SubrectBase_Y, NVSDK_NGX_Parameter_DLSS_Output_Subrect_Base_X,
    NVSDK_NGX_Parameter_DLSS_Output_Subrect_Base_Y, NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Width,
    NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Height, NVSDK_NGX_Parameter_DLSS_Indicator_Invert_X_Axis,
    NVSDK_NGX_Parameter_DLSS_Indicator_Invert_Y_Axis, NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_DLAA,
    NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_UltraQuality, NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Quality,
    NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Balanced, NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Performance,
    NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_UltraPerformance, "DLSS.Use.HW.Depth", DlssNr::kNrResetParam,
    DlssNr::kNrEyeParam
};
constexpr const char* kFloatKeys[] = {
    NVSDK_NGX_Parameter_Jitter_Offset_X,  NVSDK_NGX_Parameter_Jitter_Offset_Y,     NVSDK_NGX_Parameter_MV_Scale_X,
    NVSDK_NGX_Parameter_MV_Scale_Y,       NVSDK_NGX_Parameter_MV_Offset_X,         NVSDK_NGX_Parameter_MV_Offset_Y,
    NVSDK_NGX_Parameter_Sharpness,        NVSDK_NGX_Parameter_DLSS_Pre_Exposure,   NVSDK_NGX_Parameter_DLSS_Exposure_Scale,
    NVSDK_NGX_Parameter_FrameTimeDeltaInMsec
};
constexpr const char* kResourceKeys[] = {
    NVSDK_NGX_Parameter_Color,           NVSDK_NGX_Parameter_Output,
    NVSDK_NGX_Parameter_Depth,           NVSDK_NGX_Parameter_MotionVectors,
    NVSDK_NGX_Parameter_TransparencyMask, NVSDK_NGX_Parameter_ExposureTexture,
    NVSDK_NGX_Parameter_DepthHighRes,    "DLSS.Input.Bias.Current.Color.Mask",
    "DLSS.Input.Reduce.Ghost.Mask",      DlssNr::kNrMotionParam
};
constexpr const char* kMaskKey = "DLSS.Input.Bias.Current.Color.Mask";
constexpr const char* kMaskBaseX = "DLSS.Input.Bias.Current.Color.Subrect.Base.X";
constexpr const char* kMaskBaseY = "DLSS.Input.Bias.Current.Color.Subrect.Base.Y";
constexpr const char* kResourceBaseKeys[] = { kMaskBaseX, kMaskBaseY };

bool CopyParameters(NVSDK_NGX_Parameter* game, NVSDK_NGX_Parameter* own)
{
    int numbers = 0, floats = 0;
    std::string resources;
    for (const char* name : kUintKeys)
    {
        unsigned int v = 0;
        if (game->Get(name, &v) == NVSDK_NGX_Result_Success)
        {
            own->Set(name, v);
            ++numbers;
        }
    }
    for (const char* name : kResourceBaseKeys)
    {
        unsigned int v = 0;
        if (game->Get(name, &v) == NVSDK_NGX_Result_Success)
            own->Set(name, v);
    }
    for (const char* name : kFloatKeys)
    {
        float v = 0.0f;
        if (game->Get(name, &v) == NVSDK_NGX_Result_Success)
        {
            own->Set(name, v);
            ++floats;
        }
    }
    void* colour = nullptr;
    void* output = nullptr;
    for (const char* name : kResourceKeys)
    {
        void* p = nullptr;
        if (game->Get(name, &p) != NVSDK_NGX_Result_Success)
            continue;
        own->Set(name, p); // identity only: an NGX block holds no reference
        if (std::strcmp(name, NVSDK_NGX_Parameter_Color) == 0)
            colour = p;
        if (std::strcmp(name, NVSDK_NGX_Parameter_Output) == 0)
            output = p;
        if (!g_loggedCopy)
            resources += std::string(resources.empty() ? "" : " ") + name + (p == nullptr ? "(null)" : "");
    }
    if (colour == nullptr || output == nullptr)
    {
        LogOnce(g_loggedMissing, "crop skipped: the game's block holds no colour or output to copy into the crop's own block");
        return false;
    }
    if (!g_loggedCopy)
    {
        g_loggedCopy = true;
        LOG_INFO("Per-eye DLSS VK: the crop runs on a parameter block of its own (the game's is never written): copied "
                 "{} numbers, {} floats; resources the game's block holds: {}",
                 numbers, floats, resources);
    }
    return true;
}

unsigned int GetUint(NVSDK_NGX_Parameter* p, const char* name, unsigned int fallback = 0)
{
    unsigned int v = fallback;
    return p->Get(name, &v) == NVSDK_NGX_Result_Success ? v : fallback;
}

void WindowSummary()
{
    if (g_evals - g_windowStart < kWindow)
        return;
    g_windowStart = g_evals;
    g_lastWindowEvals = kWindow;
    g_lastDrawn = g_drawn;
    g_lastSkipped = g_skippedSize + g_skippedPlan + g_skippedParams;
    g_lastFailed = g_failed;
    g_lastPeriphDrawn = g_periphDrawn;
    g_lastPeriphFallbacks = g_periphFallbacks;
    g_lastGathered = g_gathered;
    g_lastTaaEvals = g_taaEvals;
    g_lastTaaResets = g_taaResets;
    g_lastTaaSkips = g_taaSkips;
    if (g_taaEvals != 0 || g_taaSkips != 0)
        LOG_INFO("Per-eye DLSS VK: own TAA periphery: {} evaluates ({} history resets), {} skipped to the DLSS "
                 "periphery; blend {:.2f}, resolution {:.2f} of the eye, {} histories",
                 g_taaEvals, g_taaResets, g_taaSkips, Config::Instance()->UpscalerFoveaTaaBlend.value_or_default(),
                 Config::Instance()->UpscalerFoveaTaaScale.value_or_default(), g_taa.size());
    LOG_INFO("Per-eye DLSS VK: foveal crop {} (preset {}, feather {} px, {} crops): {} drawn, {} skipped ({} made for "
             "another eye size, {} no box, {} no block), {} failed; cheap periphery {} (preset {}, {} features): {} "
             "drawn in place of the full pass, {} fell back to the full pass",
             g_foveaOn ? "ON" : "off", Config::Instance()->UpscalerFoveaPreset.value_or_default(),
             Config::Instance()->UpscalerFoveaFeather.value_or_default(), g_crops.size(), g_drawn,
             g_skippedSize + g_skippedPlan + g_skippedParams, g_skippedSize, g_skippedPlan, g_skippedParams, g_failed,
             g_periphFailed ? "FAILED"
             : Config::Instance()->UpscalerFoveaCheapPeriphery.value_or_default() ? "on"
                                                                                   : "off",
             Config::Instance()->UpscalerFoveaPeripheryPreset.value_or_default(), g_periphs.size(), g_periphDrawn,
             g_periphFallbacks);
    if (g_drawn != 0)
        LOG_INFO("Per-eye DLSS VK: the crop's inputs {} this window ({} of {} evaluates copied at the box; the bias "
                 "mask {})",
                 g_gathered != 0 ? "COPIED" : "read through subrects", g_gathered, g_drawn,
                 Config::Instance()->UpscalerFoveaCropMask.value_or_default() == 0   ? "left out"
                 : Config::Instance()->UpscalerFoveaCropMask.value_or_default() == 2 ? "as handed (full frame)"
                                                                                     : "at the box");
    if (g_skippedMaking != 0 || g_periphMaking != 0 || !g_pendingCrops.empty() || !g_pendingPeriphs.empty())
        LOG_INFO("Per-eye DLSS VK: the worker: {} crop evaluates and {} periphery evaluates waited for a feature being "
                 "made this window; {} crop and {} periphery features still being made",
                 g_skippedMaking, g_periphMaking, g_pendingCrops.size(), g_pendingPeriphs.size());
    g_drawn = g_skippedSize = g_skippedPlan = g_skippedParams = g_failed = 0;
    g_periphDrawn = g_periphFallbacks = g_skippedMaking = g_periphMaking = g_gathered = 0;
    g_taaEvals = g_taaResets = g_taaSkips = 0;
}

void ForgetTaa(unsigned int key)
{
    auto it = g_taa.find(key);
    if (it == g_taa.end())
        return;
    RetireImage(it->second.tex[0]);
    RetireImage(it->second.tex[1]);
    g_taa.erase(it);
}

// Under g_mutex, in Periphery(): our TAA in the place of the DLSS periphery for `key`. True = the output holds the
// periphery (the crop goes over it; the game's full pass is skipped). False = not run (the DLSS periphery runs).
bool TaaPeriphery(VkCommandBuffer cmd, NVSDK_NGX_Parameter* params, unsigned int key, uint32_t R, uint32_t Rh,
                  const NVSDK_NGX_Resource_VK& out)
{
    auto skip = [&](const std::string& why)
    {
        ++g_taaSkips;
        LogOnce(g_loggedTaaSkip, std::format("own TAA periphery skipped ({}); the DLSS periphery runs instead (counted "
                                             "per window)",
                                             why));
        return false;
    };
    if (g_taaFailed)
        return false;
    const auto& ov = out.Resource.ImageViewInfo;
    NVSDK_NGX_Resource_VK *colour = nullptr, *motion = nullptr, *depth = nullptr;
    params->Get(NVSDK_NGX_Parameter_Color, (void**) &colour);
    params->Get(NVSDK_NGX_Parameter_MotionVectors, (void**) &motion);
    params->Get(NVSDK_NGX_Parameter_Depth, (void**) &depth);
    if (colour == nullptr || motion == nullptr || depth == nullptr ||
        colour->Type != NVSDK_NGX_RESOURCE_VK_TYPE_VK_IMAGEVIEW || motion->Type != NVSDK_NGX_RESOURCE_VK_TYPE_VK_IMAGEVIEW ||
        depth->Type != NVSDK_NGX_RESOURCE_VK_TYPE_VK_IMAGEVIEW)
        return skip("a missing colour, motion or depth");
    const auto &cv = colour->Resource.ImageViewInfo, &mv = motion->Resource.ImageViewInfo,
               &dv = depth->Resource.ImageViewInfo;
    const uint32_t colourBase[2] = { GetUint(params, NVSDK_NGX_Parameter_DLSS_Input_Color_Subrect_Base_X),
                                     GetUint(params, NVSDK_NGX_Parameter_DLSS_Input_Color_Subrect_Base_Y) };
    const uint32_t motionBase[2] = { GetUint(params, NVSDK_NGX_Parameter_DLSS_Input_MV_SubrectBase_X),
                                     GetUint(params, NVSDK_NGX_Parameter_DLSS_Input_MV_SubrectBase_Y) };
    const uint32_t depthBase[2] = { GetUint(params, NVSDK_NGX_Parameter_DLSS_Input_Depth_Subrect_Base_X),
                                    GetUint(params, NVSDK_NGX_Parameter_DLSS_Input_Depth_Subrect_Base_Y) };
    if (GetUint(params, NVSDK_NGX_Parameter_DLSS_Output_Subrect_Base_X) != 0 ||
        GetUint(params, NVSDK_NGX_Parameter_DLSS_Output_Subrect_Base_Y) != 0)
        return skip("the output has a subrect offset");
    auto fits = [&](const NVSDK_NGX_ImageViewInfo_VK& v, const uint32_t base[2])
    { return base[0] + R <= v.Width && base[1] + Rh <= v.Height; };
    if (!fits(cv, colourBase) || !fits(mv, motionBase) || !fits(dv, depthBase))
        return skip(std::format("render {}x{} does not fit colour {}x{} / motion {}x{} / depth {}x{}", R, Rh, cv.Width,
                                cv.Height, mv.Width, mv.Height, dv.Width, dv.Height));
    float jitter[2] = { 0, 0 }, scale[2] = { 0, 0 };
    params->Get(NVSDK_NGX_Parameter_Jitter_Offset_X, &jitter[0]);
    params->Get(NVSDK_NGX_Parameter_Jitter_Offset_Y, &jitter[1]);
    params->Get(NVSDK_NGX_Parameter_MV_Scale_X, &scale[0]);
    params->Get(NVSDK_NGX_Parameter_MV_Scale_Y, &scale[1]);
    if (scale[0] == 0 || scale[1] == 0)
        return skip("MV scale is zero");
    const bool flip = Config::Instance()->UpscalerFoveaTaaFlipJitter.value_or_default();
    if (flip)
    {
        jitter[0] = -jitter[0];
        jitter[1] = -jitter[1];
    }
    const float taaScale = std::clamp(Config::Instance()->UpscalerFoveaTaaScale.value_or_default(), 0.25f, 1.0f);
    const bool scaled = taaScale < 0.99f;
    const uint32_t tw = scaled ? (uint32_t) std::max(64.0, std::round((double) ov.Width * taaScale)) : ov.Width;
    const uint32_t th = scaled ? (uint32_t) std::max(64.0, std::round((double) ov.Height * taaScale)) : ov.Height;
    if (scaled && !EnsureImage(g_taaMid, tw, th, kShaderFormat, "TAA mid", g_loggedTaaFail))
    {
        g_taaFailed = true;
        return skip("could not make the TAA image");
    }
    auto& hist = g_taa[key];
    for (auto& t : hist.tex)
        if (!EnsureImage(t, tw, th, kShaderFormat, "TAA history", g_loggedTaaFail))
        {
            g_taaFailed = true;
            return skip("could not make the history images");
        }
    unsigned int reset = 0;
    params->Get(NVSDK_NGX_Parameter_Reset, &reset);
    const uint64_t present = g_presents.load();
    const bool twoBack = PerEyeDlss_Vk::RoutingOn() || Config::Instance()->UpscalerCropEyeByParity.value_or_default();
    const uint64_t stride = twoBack ? 2 : 1;
    const bool useHistory = reset == 0 && hist.lastPresent != UINT64_MAX && present == hist.lastPresent + stride;
    const int rd = (int) hist.read, wr = rd ^ 1;
    if (scaled)
        LayOut(cmd, g_taaMid);
    LayOut(cmd, hist.tex[0]);
    LayOut(cmd, hist.tex[1]);
    const float blend = std::clamp(Config::Instance()->UpscalerFoveaTaaBlend.value_or_default(), 0.01f, 1.0f);
    FoveaPasses_Vk::TaaConstants c {};
    c.outSize[0] = tw;
    c.outSize[1] = th;
    c.renderSize[0] = R;
    c.renderSize[1] = Rh;
    c.colourBase[0] = colourBase[0];
    c.colourBase[1] = colourBase[1];
    c.motionBase[0] = motionBase[0];
    c.motionBase[1] = motionBase[1];
    c.depthBase[0] = depthBase[0];
    c.depthBase[1] = depthBase[1];
    c.jitter[0] = jitter[0];
    c.jitter[1] = jitter[1];
    c.mvScale[0] = scale[0];
    c.mvScale[1] = scale[1];
    c.blend = blend;
    c.reset = useHistory ? 0u : 1u;
    g_passes.Taa(cmd, g_slot++, cv.ImageView, mv.ImageView, dv.ImageView, hist.tex[rd].view,
                 scaled ? g_taaMid.view : ov.ImageView, hist.tex[wr].view, c);
    if (scaled)
    {
        FoveaPasses_Vk::EnlargeConstants e {};
        e.outW = ov.Width;
        e.outH = ov.Height;
        e.srcW = tw;
        e.srcH = th;
        g_passes.Enlarge(cmd, g_slot++, g_taaMid.view, ov.ImageView, e);
    }
    hist.read = (uint32_t) wr;
    hist.lastPresent = present;
    ++g_taaEvals;
    if (!useHistory)
        ++g_taaResets;
    if (!g_loggedTaaFirst)
    {
        g_loggedTaaFirst = true;
        LOG_INFO("Per-eye DLSS VK: FIRST own TAA periphery drawn (eye key {:#x}, history {} Present(s) back; render "
                 "{}x{} -> {}x{}{}; colour fmt {} base ({}, {}), motion fmt {} {}x{}, depth fmt {} aspect {}; jitter "
                 "({:+.3f}, {:+.3f}){}, MV scale ({:.1f}, {:.1f}), blend {:.2f}, {})",
                 key, stride, R, Rh, ov.Width, ov.Height, scaled ? std::format(" via {}x{}", tw, th) : "",
                 (int) cv.Format, colourBase[0], colourBase[1], (int) mv.Format, mv.Width, mv.Height, (int) dv.Format,
                 (unsigned) dv.SubresourceRange.aspectMask, jitter[0], jitter[1], flip ? " flipped" : "", scale[0],
                 scale[1], blend, useHistory ? "with history" : "history reset");
    }
    return true;
}

// One input gathered: the game's image `src` (its format, size and aspect) copied at the render box into `im`, and
// the crop's block pointed at our copy at base (0, 0). False = not copied (logged once): the caller keeps the
// subrect path for this input.
bool GatherOne(VkCommandBuffer cmd, NVSDK_NGX_Parameter* game, NVSDK_NGX_Parameter* cp, const char* key,
               const char* baseX, const char* baseY, OwnImage& im, const FoveaBox::Rect& r, const char* what,
               std::string& list)
{
    NVSDK_NGX_Resource_VK* src = nullptr;
    if (game->Get(key, (void**) &src) != NVSDK_NGX_Result_Success || src == nullptr ||
        src->Type != NVSDK_NGX_RESOURCE_VK_TYPE_VK_IMAGEVIEW)
        return false;
    const auto& sv = src->Resource.ImageViewInfo;
    const unsigned int bx = GetUint(game, baseX), by = GetUint(game, baseY);
    if (bx + r.rx + r.rw > sv.Width || by + r.ry + r.rh > sv.Height)
    {
        LogOnce(g_loggedGatherFail, std::format("the crop's {} is not copied: the box ({}, {}) {}x{} at base ({}, {}) "
                                                "lies outside the game's {}x{} image (subrects used for it)",
                                                what, r.rx, r.ry, r.rw, r.rh, bx, by, sv.Width, sv.Height));
        return false;
    }
    const VkImageAspectFlags aspect =
        sv.SubresourceRange.aspectMask != 0 ? sv.SubresourceRange.aspectMask : VK_IMAGE_ASPECT_COLOR_BIT;
    if (!EnsureImage(im, r.rw, r.rh, sv.Format, what, g_loggedGatherFail, aspect))
        return false;
    LayOut(cmd, im);
    FoveaPasses_Vk::CopyBox(cmd, sv.Image, aspect, bx + r.rx, by + r.ry, im.image, r.rw, r.rh);
    cp->Set(key, (void*) &im.ngx);
    cp->Set(baseX, 0u);
    cp->Set(baseY, 0u);
    if (!g_loggedGather)
        list += std::format("{}{} fmt {} {}x{}", list.empty() ? "" : ", ", what, (int) sv.Format, sv.Width, sv.Height);
    return true;
}

void ForgetPeriphKey(unsigned int key)
{
    auto it = g_periphs.find(key);
    if (it == g_periphs.end())
        return;
    LOG_INFO("Per-eye DLSS VK: periphery feature {} of feature {} released", it->second.feature, it->second.owner);
    NvngxVkOwn::Retire(it->second.feature);
    if (it->second.block != nullptr)
        NVSDK_NGX_VULKAN_DestroyParameters(it->second.block);
    g_periphs.erase(it);
}

// ---- The worker's creates (VkCreateWorker) ----
// Under g_mutex: what the worker finished since the last call, orphans retired at once.
void Collect()
{
    std::vector<Done> done;
    {
        std::lock_guard lock(g_doneMutex);
        done.swap(g_doneByWorker);
    }
    for (auto& d : done)
    {
        const unsigned int id = d.id;
        if (auto it = std::find(g_orphans.begin(), g_orphans.end(), id); it != g_orphans.end())
        {
            g_orphans.erase(it);
            if (d.ok)
            {
                NvngxVkOwn::Adopt(id, std::move(d.feature));
                NvngxVkOwn::Retire(id);
            }
            LOG_INFO("Per-eye DLSS VK: feature {} made on the worker after its owner was released: retired", id);
            continue;
        }
        g_ready[id] = std::move(d);
    }
}

// Only the numbers of `from` (sizes, flags, presets, scales) and the output image: what a create reads. The game's
// resources are NOT copied (their descriptors are the game's, of this evaluate only).
NVSDK_NGX_Parameter* CreateBlockFrom(NVSDK_NGX_Parameter* from)
{
    NVSDK_NGX_Parameter* block = nullptr;
    if (NVSDK_NGX_VULKAN_AllocateParameters(&block) != NVSDK_NGX_Result_Success || block == nullptr)
        return nullptr;
    for (const char* name : kUintKeys)
    {
        unsigned int v = 0;
        if (from->Get(name, &v) == NVSDK_NGX_Result_Success)
            block->Set(name, v);
    }
    for (const char* name : kFloatKeys)
    {
        float v = 0.0f;
        if (from->Get(name, &v) == NVSDK_NGX_Result_Success)
            block->Set(name, v);
    }
    void* output = nullptr;
    if (from->Get(NVSDK_NGX_Parameter_Output, &output) == NVSDK_NGX_Result_Success && output != nullptr)
        block->Set(NVSDK_NGX_Parameter_Output, output);
    block->Set(NVSDK_NGX_Parameter_Reset, 1u);
    return block;
}

// A feature of our own through the worker, for `key`: 1 = ready, adopted, its id in *made (the caller keeps
// `block` with its entry); 0 = being made (asked now, or still on the worker, or made for a box since changed and
// asked again); -1 = the worker's create failed (the caller falls back for the session).
int WorkerFeature(std::unordered_map<unsigned int, Pending>& pending, unsigned int key, unsigned int owner,
                  NVSDK_NGX_Parameter* from, Upscaler backend, int preset, const FoveaBox::Rect& rect, uint32_t R,
                  uint32_t Rh, uint32_t O, uint32_t Oh, const char* what, unsigned int* made,
                  NVSDK_NGX_Parameter** block)
{
    auto it = pending.find(key);
    if (it == pending.end())
    {
        Pending p;
        p.id = IFeature::GetNextHandleId();
        p.owner = owner;
        p.block = CreateBlockFrom(from);
        if (p.block == nullptr)
            return -1;
        p.rect = rect;
        p.R = R;
        p.Rh = Rh;
        p.O = O;
        p.Oh = Oh;
        p.preset = preset;
        p.asked = std::chrono::steady_clock::now();
        auto holder = std::make_shared<std::unique_ptr<IFeature_Vk>>();
        const unsigned int id = p.id;
        NVSDK_NGX_Parameter* const b = p.block;
        const auto started = p.asked;
        VkCreateWorker::Job job;
        job.ticket = id;
        job.record = [id, b, backend, preset, holder](VkCommandBuffer cmd)
        {
            t_presetForCreate = preset;
            const bool ok = NvngxVkOwn::MakeDetached(cmd, backend, b, id, *holder);
            t_presetForCreate = 0;
            return ok;
        };
        job.done = [id, holder, started](uint64_t, bool ok)
        {
            Done d;
            d.id = id;
            d.ok = ok && *holder != nullptr;
            d.feature = std::move(*holder);
            d.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
            std::lock_guard lock(g_doneMutex);
            g_doneByWorker.push_back(std::move(d));
        };
        VkCreateWorker::Submit(std::move(job));
        LOG_INFO("Per-eye DLSS VK: the worker is asked for a {} feature {} for feature {} (eye key {:#x}; {}x{} -> "
                 "{}x{}, preset {}); the full pass runs until it is ready",
                 what, id, owner, key, rect.rw, rect.rh, rect.ow, rect.oh, preset);
        pending.emplace(key, p);
        return 0;
    }
    auto ready = g_ready.find(it->second.id);
    if (ready == g_ready.end())
        return 0;
    Done d = std::move(ready->second);
    g_ready.erase(ready);
    Pending p = it->second;
    pending.erase(it);
    if (!d.ok)
    {
        NVSDK_NGX_VULKAN_DestroyParameters(p.block);
        LOG_WARN("Per-eye DLSS VK: the worker could not make the {} feature {} ({:.0f} ms)", what, p.id, d.ms);
        return -1;
    }
    NvngxVkOwn::Adopt(p.id, std::move(d.feature));
    if (!FoveaBox::Same(p.rect, rect) || p.R != R || p.Rh != Rh || p.O != O || p.Oh != Oh ||
        p.preset != preset)
    {
        NvngxVkOwn::Retire(p.id);
        NVSDK_NGX_VULKAN_DestroyParameters(p.block);
        auto& reasks = &pending == &g_pendingCrops ? g_reasksCrops : g_reasksPeriphs;
        const int n = ++reasks[key];
        if (n > kMaxReasks)
        {
            LOG_WARN("Per-eye DLSS VK: the worker's {} feature {} was made for a box or preset since changed for the "
                     "{}th time (asked {}x{} -> {}x{} preset {}, made {}x{} -> {}x{} preset {}): this eye key gives up "
                     "for the session",
                     what, p.id, n, rect.rw, rect.rh, rect.ow, rect.oh, preset, p.rect.rw, p.rect.rh, p.rect.ow,
                     p.rect.oh, p.preset);
            return -1;
        }
        LOG_INFO("Per-eye DLSS VK: the worker's {} feature {} was made for a box or preset since changed ({} of {}): "
                 "retired, asked again",
                 what, p.id, n, kMaxReasks);
        return 0;
    }
    LOG_INFO("Per-eye DLSS VK: {} feature {} made ON THE WORKER in {:.0f} ms for feature {} (eye key {:#x}; {}x{} -> "
             "{}x{}, preset {}): no Present waited",
             what, p.id, d.ms, p.owner, key, rect.rw, rect.rh, rect.ow, rect.oh, preset);
    *made = p.id;
    *block = p.block;
    return 1;
}

// The owner is gone: what the worker still makes for its keys is retired when it comes.
void OrphanPending(std::unordered_map<unsigned int, Pending>& pending, unsigned int key)
{
    auto it = pending.find(key);
    if (it == pending.end())
        return;
    if (auto ready = g_ready.find(it->second.id); ready != g_ready.end())
    {
        if (ready->second.ok)
        {
            NvngxVkOwn::Adopt(it->second.id, std::move(ready->second.feature));
            NvngxVkOwn::Retire(it->second.id);
        }
        g_ready.erase(ready);
    }
    else
        g_orphans.push_back(it->second.id);
    NVSDK_NGX_VULKAN_DestroyParameters(it->second.block);
    pending.erase(it);
}

// What both passes need of an evaluate, under the lock: the device (the first one seen), the game's output image view
// in the composite's format, the composite and enlarge passes set up once. False = this evaluate is skipped (logged).
bool Setup(VkDevice device, VkPhysicalDevice physical, NVSDK_NGX_Parameter* params, NVSDK_NGX_Resource_VK*& output)
{
    if (g_device == VK_NULL_HANDLE)
    {
        g_device = device;
        g_physical = physical;
    }
    if (device != g_device)
    {
        LogOnce(g_loggedOtherDevice, "foveal crop skipped: an evaluate on another device");
        return false;
    }
    output = nullptr;
    params->Get(NVSDK_NGX_Parameter_Output, (void**) &output);
    if (output == nullptr || output->Type != NVSDK_NGX_RESOURCE_VK_TYPE_VK_IMAGEVIEW)
    {
        LogOnce(g_loggedNoOutput, "foveal crop skipped: the game's block holds no output image view");
        return false;
    }
    const auto& ov = output->Resource.ImageViewInfo;
    if (ov.Format != kShaderFormat)
    {
        if (!g_loggedFormat)
        {
            g_loggedFormat = true;
            LOG_WARN("Per-eye DLSS VK: foveal crop OFF: the game's output is format {} and the composite's storage "
                     "images are declared rgba16f (format {}) -- a shader variant is needed for this game",
                     (int) ov.Format, (int) kShaderFormat);
        }
        g_foveaFailed = true;
        return false;
    }
    if (!g_passesTried)
    {
        g_passesTried = true;
        std::string error;
        if (!g_passes.Init(g_device, g_physical, error))
        {
            LOG_WARN("Per-eye DLSS VK: foveal crop OFF: the composite could not be set up ({})", error);
            g_foveaFailed = true;
            return false;
        }
        LOG_INFO("Per-eye DLSS VK: foveal crop composite ready (output format {})", (int) ov.Format);
    }
    return g_passes.Ready();
}

void ForgetKey(unsigned int key)
{
    auto it = g_crops.find(key);
    if (it == g_crops.end())
        return;
    LOG_INFO("Per-eye DLSS VK: foveal crop feature {} of feature {} released", it->second.feature, it->second.owner);
    NvngxVkOwn::Retire(it->second.feature);
    if (it->second.block != nullptr)
        NVSDK_NGX_VULKAN_DestroyParameters(it->second.block);
    g_crops.erase(it);
}
} // namespace

bool Enabled() { return Config::Instance()->UpscalerPerEye.value_or_default(); }

int PresetForCreate() { return t_presetForCreate; }

void OnPresent()
{
    const uint64_t now = ++g_presents;
    std::lock_guard lock(g_mutex);
    // The GPU timer's ring holds 64 evaluates: read at EVERY Present (once per window read nothing at all).
    if (FrameTime_Vk::Ready())
        FrameTime_Vk::Collect();
    // Presents per second, per window (the Vulkan path's frame-rate reading; the D3D12 path has one of its own).
    if (now == 1)
    {
        g_presentWindowStart = std::chrono::steady_clock::now();
        g_presentWindowEvals = g_evals;
    }
    else if (now % kPresentWindow == 0)
    {
        const auto t = std::chrono::steady_clock::now();
        const double s = std::chrono::duration<double>(t - g_presentWindowStart).count();
        if (s > 0.0)
        {
            g_lastPresentsPerSec = kPresentWindow / s;
            g_lastEvalsPerPresent = (double) (g_evals - g_presentWindowEvals) / kPresentWindow;
            LOG_INFO("Per-eye DLSS VK: {} Presents in {:.1f} s = {:.1f} Presents/s; {:.2f} DLSS evaluates per Present; "
                     "crop {}, periphery {}",
                     kPresentWindow, s, g_lastPresentsPerSec, g_lastEvalsPerPresent,
                     g_foveaOn && !g_foveaFailed ? "ON" : "off",
                     g_foveaOn && !g_foveaFailed && !g_periphFailed &&
                             Config::Instance()->UpscalerFoveaCheapPeriphery.value_or_default()
                         ? "on"
                         : "off");
        }
        // The GPU times of the window (the frame as a whole, the game's full pass, our passes).
        if (FrameTime_Vk::Ready())
        {
            g_lastGpu = FrameTime_Vk::Take();
            if (g_lastGpu.frames == 0)
                LOG_INFO("Per-eye DLSS VK: GPU per frame: NOTHING measured this window ({} evaluates stamped, {} lost to "
                         "the ring before they were read, {} whose stamps never came)",
                         g_lastGpu.begun, g_lastGpu.lost, g_lastGpu.unfinished);
            else
                LOG_INFO("Per-eye DLSS VK: GPU per frame: {:.2f} ms the whole frame ({} frames); the game's DLSS full pass "
                         "{:.2f} ms on {} of them; ours {:.2f} ms = {:.2f} before the full pass's place (periphery or "
                         "TAA, the copies) + {:.2f} after (the crop, the composite); {} evaluates measured of {} stamped ({} lost "
                         "to the ring, {} whose stamps never came)",
                         g_lastGpu.frameMs, g_lastGpu.frames, g_lastGpu.gameMs, g_lastGpu.gameRuns,
                         g_lastGpu.beforeMs + g_lastGpu.afterMs, g_lastGpu.beforeMs, g_lastGpu.afterMs, g_lastGpu.ours,
                         g_lastGpu.begun, g_lastGpu.lost, g_lastGpu.unfinished);
        }
        g_presentWindowStart = t;
        g_presentWindowEvals = g_evals;
    }
    for (auto it = g_retiredImages.begin(); it != g_retiredImages.end();)
    {
        if (now - it->second > 32)
        {
            DestroyImage(it->first);
            it = g_retiredImages.erase(it);
        }
        else
            ++it;
    }
}

void AfterUpscale(VkCommandBuffer cmd, unsigned int ownerHandle, NVSDK_NGX_Parameter* params, IFeature_Vk* owner,
                  VkDevice device, VkPhysicalDevice physical)
{
    if (!Enabled() || cmd == VK_NULL_HANDLE || params == nullptr || owner == nullptr)
        return;
    std::lock_guard lock(g_mutex);
    ++g_evals;
    WindowSummary();
    Collect();
    if (!g_startApplied)
    {
        // The saved start, once: the overlay's "Crop on from the launch".
        g_startApplied = true;
        if (Config::Instance()->UpscalerFoveaStart.value_or_default())
        {
            g_foveaOn = true;
            LOG_INFO("Per-eye DLSS VK: foveal crop ON from the launch ([Inputs] UpscalerFoveaStart = true)");
        }
        else
            LOG_INFO("Per-eye DLSS VK: foveal crop off at the launch ([Inputs] UpscalerFoveaStart = false); the "
                     "overlay's 'Crop on' switches it (F4 is the game's key on Vulkan)");
    }
    if (!g_foveaOn || g_foveaFailed)
        return;
    // The game's output and the eye.
    NVSDK_NGX_Resource_VK* output = nullptr;
    if (!Setup(device, physical, params, output))
    {
        ++g_skippedPlan;
        return;
    }
    const auto& ov = output->Resource.ImageViewInfo;
    const unsigned int R = GetUint(params, NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Width,
                                   GetUint(params, NVSDK_NGX_Parameter_Width));
    const unsigned int Rh = GetUint(params, NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Height,
                                    GetUint(params, NVSDK_NGX_Parameter_Height));
    const unsigned int O = ov.Width, Oh = ov.Height;

    // The box.
    FoveaBox::Rect r;
    const float f = Config::Instance()->UpscalerFoveaCrop.value_or_default();
    const float fh = Config::Instance()->UpscalerFoveaCropHeight.value_or_default();
    if (!FoveaBox::Plan(R, Rh, O, Oh, f, fh, true, r) || r.ow == 0 || r.oh == 0)
    {
        LogOnce(g_loggedPlan, std::format("foveal crop skipped: no crop fits render {}x{} -> output {}x{} at {:.2f}", R,
                                          Rh, O, Oh, f));
        ++g_skippedPlan;
        return;
    }
    g_statusRect = r;
    g_statusEyeW = O;
    g_statusEyeH = Oh;
    if (!g_loggedFirst || !FoveaBox::Same(r, g_loggedRect))
    {
        g_loggedRect = r;
        LOG_INFO("Per-eye DLSS VK: foveal crop for render {}x{} -> output {}x{} (asked {:.2f} x {:.2f}): render box "
                 "({}, {}) {}x{} -> output box ({}, {}) {}x{} = {:.1f}% of the eye; {}",
                 R, Rh, O, Oh, f, fh > 0.0f ? fh : f, r.rx, r.ry, r.rw, r.rh, r.ox, r.oy, r.ow, r.oh,
                 100.0 * r.ow * r.oh / ((double) O * Oh),
                 r.exact ? "EXACT (same ratio, whole pixels)" : "approximate (edges up to half an output pixel off)");
    }

    // The eye key: the routed owner per eye; in shared routing the Present's parity names the eye.
    unsigned int key = ownerHandle;
    if (!PerEyeDlss_Vk::RoutingOn() && Config::Instance()->UpscalerCropEyeByParity.value_or_default() &&
        PerEyeDlss_Vk::EyeParity() != 0)
        key |= kOddEyeKey;
    const int preset = Config::Instance()->UpscalerFoveaPreset.value_or_default();
    auto it = g_crops.find(key);
    if (it != g_crops.end() && (it->second.eyeR != R || it->second.eyeRh != Rh || it->second.eyeO != O ||
                                it->second.eyeOh != Oh))
    {
        // Made for another eye size (a resize): it goes, a new one is made below.
        LOG_INFO("Per-eye DLSS VK: foveal crop feature {} was made for render {}x{} -> output {}x{}; this evaluate "
                 "is {}x{} -> {}x{}: remade",
                 it->second.feature, it->second.eyeR, it->second.eyeRh, it->second.eyeO, it->second.eyeOh, R, Rh, O,
                 Oh);
        ForgetKey(key);
        it = g_crops.end();
    }
    if (it != g_crops.end() &&
        (!FoveaBox::Same(it->second.rect, r) || it->second.preset != preset))
    {
        LOG_INFO("Per-eye DLSS VK: foveal crop feature {} remade (the box or the preset changed)", it->second.feature);
        ForgetKey(key);
        it = g_crops.end();
    }

    if (!EnsureCropImage(r.ow, r.oh, ov.Format))
        return;
    LayOut(cmd, g_cropImage);

    // The crop's own block, filled from the game's.
    if (g_cropParams == nullptr &&
        (NVSDK_NGX_VULKAN_AllocateParameters(&g_cropParams) != NVSDK_NGX_Result_Success || g_cropParams == nullptr))
    {
        g_cropParams = nullptr;
        LogOnce(g_loggedMissing, "crop skipped: no parameter block of its own could be allocated");
        ++g_skippedParams;
        return;
    }
    if (!CopyParameters(params, g_cropParams))
    {
        ++g_skippedParams;
        return;
    }
    NVSDK_NGX_Parameter* const cp = g_cropParams;
    // The subrect path first: DLSS reads the game's colour, depth, motion and bias mask at base + box (the mask's
    // base too: left at the game's, the crop read the mask at the top left of the frame).
    cp->Set(NVSDK_NGX_Parameter_DLSS_Input_Color_Subrect_Base_X,
            GetUint(params, NVSDK_NGX_Parameter_DLSS_Input_Color_Subrect_Base_X) + r.rx);
    cp->Set(NVSDK_NGX_Parameter_DLSS_Input_Color_Subrect_Base_Y,
            GetUint(params, NVSDK_NGX_Parameter_DLSS_Input_Color_Subrect_Base_Y) + r.ry);
    cp->Set(NVSDK_NGX_Parameter_DLSS_Input_Depth_Subrect_Base_X,
            GetUint(params, NVSDK_NGX_Parameter_DLSS_Input_Depth_Subrect_Base_X) + r.rx);
    cp->Set(NVSDK_NGX_Parameter_DLSS_Input_Depth_Subrect_Base_Y,
            GetUint(params, NVSDK_NGX_Parameter_DLSS_Input_Depth_Subrect_Base_Y) + r.ry);
    cp->Set(NVSDK_NGX_Parameter_DLSS_Input_MV_SubrectBase_X,
            GetUint(params, NVSDK_NGX_Parameter_DLSS_Input_MV_SubrectBase_X) + r.rx);
    cp->Set(NVSDK_NGX_Parameter_DLSS_Input_MV_SubrectBase_Y,
            GetUint(params, NVSDK_NGX_Parameter_DLSS_Input_MV_SubrectBase_Y) + r.ry);
    const int maskMode = std::clamp(Config::Instance()->UpscalerFoveaCropMask.value_or_default(), 0, 2);
    if (maskMode == 0)
        cp->Set(kMaskKey, (void*) nullptr);
    else if (maskMode == 1)
    {
        cp->Set(kMaskBaseX, GetUint(params, kMaskBaseX) + r.rx);
        cp->Set(kMaskBaseY, GetUint(params, kMaskBaseY) + r.ry);
    }
    // Then the gather copy over it, where it is on: each input copied at the box and read at (0, 0).
    if (Config::Instance()->UpscalerFoveaCopyInputs.value_or_default())
    {
        std::string list;
        bool any = false;
        any |= GatherOne(cmd, params, cp, NVSDK_NGX_Parameter_Color, NVSDK_NGX_Parameter_DLSS_Input_Color_Subrect_Base_X,
                         NVSDK_NGX_Parameter_DLSS_Input_Color_Subrect_Base_Y, g_gatherColour, r, "colour", list);
        any |= GatherOne(cmd, params, cp, NVSDK_NGX_Parameter_Depth, NVSDK_NGX_Parameter_DLSS_Input_Depth_Subrect_Base_X,
                         NVSDK_NGX_Parameter_DLSS_Input_Depth_Subrect_Base_Y, g_gatherDepth, r, "depth", list);
        any |= GatherOne(cmd, params, cp, NVSDK_NGX_Parameter_MotionVectors, NVSDK_NGX_Parameter_DLSS_Input_MV_SubrectBase_X,
                         NVSDK_NGX_Parameter_DLSS_Input_MV_SubrectBase_Y, g_gatherMotion, r, "motion", list);
        if (maskMode == 1)
            GatherOne(cmd, params, cp, kMaskKey, kMaskBaseX, kMaskBaseY, g_gatherMask, r, "bias mask", list);
        if (any)
            ++g_gathered;
        if (!g_loggedGather && any)
        {
            g_loggedGather = true;
            LOG_INFO("Per-eye DLSS VK: crop inputs COPIED at the box into crop-sized images of the game's formats: {}",
                     list);
        }
    }
    cp->Set(NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Width, r.rw);
    cp->Set(NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Height, r.rh);
    cp->Set(NVSDK_NGX_Parameter_Width, r.rw);
    cp->Set(NVSDK_NGX_Parameter_Height, r.rh);
    cp->Set(NVSDK_NGX_Parameter_OutWidth, r.ow);
    cp->Set(NVSDK_NGX_Parameter_OutHeight, r.oh);
    cp->Set(NVSDK_NGX_Parameter_DLSS_Output_Subrect_Base_X, 0u);
    cp->Set(NVSDK_NGX_Parameter_DLSS_Output_Subrect_Base_Y, 0u);
    cp->Set(NVSDK_NGX_Parameter_Output, (void*) &g_cropImage.ngx);

    // The crop feature, per eye key: on the worker where there is one (no Present waits for a cold create),
    // inline otherwise.
    if (it == g_crops.end())
    {
        Crop crop;
        crop.owner = ownerHandle;
        crop.rect = r;
        crop.eyeR = R;
        crop.eyeRh = Rh;
        crop.eyeO = O;
        crop.eyeOh = Oh;
        crop.preset = preset;
        crop.reset = true;
        if (VkCreateWorker::Available())
        {
            const int state = WorkerFeature(g_pendingCrops, key, ownerHandle, cp, owner->GetUpscalerType(), preset, r,
                                            R, Rh, O, Oh, "foveal crop", &crop.feature, &crop.block);
            if (state < 0)
            {
                LogOnce(g_loggedCreateFail, "foveal crop feature could not be made; the crop stays off this session");
                g_foveaFailed = true;
                return;
            }
            if (state == 0)
            {
                ++g_skippedMaking;
                return;
            }
        }
        else
        {
            t_presetForCreate = preset;
            cp->Set(NVSDK_NGX_Parameter_Reset, 1u);
            crop.feature = NvngxVkOwn::Create(cmd, owner->GetUpscalerType(), cp);
            t_presetForCreate = 0;
            if (crop.feature == 0)
            {
                LogOnce(g_loggedCreateFail, "foveal crop feature could not be made; the crop stays off this session");
                g_foveaFailed = true;
                return;
            }
            LOG_INFO("Per-eye DLSS VK: foveal crop feature {} made for feature {} (eye key {:#x}; {}x{} -> {}x{}, "
                     "preset {}) on the game's thread",
                     crop.feature, ownerHandle, key, r.rw, r.rh, r.ow, r.oh, preset);
        }
        it = g_crops.emplace(key, crop).first;
    }
    Crop& crop = it->second;
    cp->Set(NVSDK_NGX_Parameter_Reset, crop.reset ? 1u : 0u);
    crop.reset = false;

    IFeature_Vk* feature = NvngxVkOwn::Get(crop.feature);
    if (feature == nullptr)
    {
        ++g_failed;
        return;
    }
    // Whatever was current before the crop ran is what stays current (never set it to what it should be).
    IFeature* const currentBefore = State::Instance().currentFeature;
    const bool ok = feature->Evaluate(cmd, cp);
    State::Instance().currentFeature = currentBefore;
    if (!ok)
    {
        ++g_failed;
        LogOnce(g_loggedEvalFail, "the crop's DLSS evaluate failed (counted per window; the full pass stands)");
        crop.reset = true;
        return;
    }

    FoveaPasses_Vk::CompositeConstants c {};
    c.baseX = r.ox;
    c.baseY = r.oy;
    c.sizeX = r.ow;
    c.sizeY = r.oh;
    c.feather = (uint32_t) std::max(0, Config::Instance()->UpscalerFoveaFeather.value_or_default());
    c.outline = g_outline ? 1u : 0u;
    g_passes.Composite(cmd, g_slot++, g_cropImage.view, ov.ImageView, c);
    ++g_drawn;
    if (!g_loggedFirst)
    {
        g_loggedFirst = true;
        LOG_INFO("Per-eye DLSS VK: FIRST foveal crop drawn (crop feature {}, box ({}, {}) {}x{}, feather {} px)",
                 crop.feature, r.ox, r.oy, r.ow, r.oh, c.feather);
    }
}

void Forget(unsigned int ownerHandle)
{
    if (!Enabled())
        return;
    std::lock_guard lock(g_mutex);
    ForgetKey(ownerHandle);
    ForgetKey(ownerHandle | kOddEyeKey);
    ForgetPeriphKey(ownerHandle);
    ForgetPeriphKey(ownerHandle | kOddEyeKey);
    ForgetTaa(ownerHandle);
    ForgetTaa(ownerHandle | kOddEyeKey);
    Collect();
    OrphanPending(g_pendingCrops, ownerHandle);
    OrphanPending(g_pendingCrops, ownerHandle | kOddEyeKey);
    OrphanPending(g_pendingPeriphs, ownerHandle);
    OrphanPending(g_pendingPeriphs, ownerHandle | kOddEyeKey);
    g_mainSkipped.erase(ownerHandle);
}

bool Periphery(VkCommandBuffer cmd, unsigned int ownerHandle, NVSDK_NGX_Parameter* params, IFeature_Vk* owner,
               VkDevice device, VkPhysicalDevice physical)
{
    if (!Enabled() || cmd == VK_NULL_HANDLE || params == nullptr || owner == nullptr)
        return false;
    std::lock_guard lock(g_mutex);
    Collect();
    // Only with the crop drawn over it: alone, the periphery would soften the whole eye.
    if (!g_foveaOn || g_foveaFailed || g_periphFailed ||
        !Config::Instance()->UpscalerFoveaCheapPeriphery.value_or_default())
        return false;
    NVSDK_NGX_Resource_VK* output = nullptr;
    if (!Setup(device, physical, params, output))
        return false;
    const auto& ov = output->Resource.ImageViewInfo;
    const unsigned int R = GetUint(params, NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Width,
                                   GetUint(params, NVSDK_NGX_Parameter_Width));
    const unsigned int Rh = GetUint(params, NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Height,
                                    GetUint(params, NVSDK_NGX_Parameter_Height));
    const unsigned int O = ov.Width, Oh = ov.Height;
    if (R == 0 || Rh == 0 || O == 0 || Oh == 0 || R > O || Rh > Oh)
    {
        ++g_periphFallbacks;
        LogOnce(g_loggedPeriphSkip, std::format("cheap periphery skipped: render {}x{} -> output {}x{} is no upscale "
                                                "(the full pass runs; counted per window)",
                                                R, Rh, O, Oh));
        return false;
    }

    // The eye key, as the crop's.
    unsigned int key = ownerHandle;
    if (!PerEyeDlss_Vk::RoutingOn() && Config::Instance()->UpscalerCropEyeByParity.value_or_default() &&
        PerEyeDlss_Vk::EyeParity() != 0)
        key |= kOddEyeKey;
    // Our own TAA in the place of the DLSS periphery (any reason it cannot run falls through to that).
    if (Config::Instance()->UpscalerFoveaPeripheryTaa.value_or_default() && TaaPeriphery(cmd, params, key, R, Rh, *output))
    {
        g_mainSkipped[ownerHandle] = true;
        ++g_periphDrawn;
        return true;
    }
    const int preset = Config::Instance()->UpscalerFoveaPeripheryPreset.value_or_default();
    auto it = g_periphs.find(key);
    if (it != g_periphs.end() && (it->second.rw != R || it->second.rh != Rh || it->second.preset != preset))
    {
        LOG_INFO("Per-eye DLSS VK: periphery feature {} was made for {}x{} preset {}; this evaluate renders {}x{} "
                 "preset {}: remade",
                 it->second.feature, it->second.rw, it->second.rh, it->second.preset, R, Rh, preset);
        ForgetPeriphKey(key);
        it = g_periphs.end();
    }

    if (!EnsureImage(g_periphImage, R, Rh, ov.Format, "cheap periphery", g_loggedPeriphImageFail))
    {
        g_periphFailed = true;
        ++g_periphFallbacks;
        return false;
    }
    LayOut(cmd, g_periphImage);

    // The periphery's own block: the game's inputs as they are (render subrect, bases, jitter, motion, NR's colour
    // swap), the sizes at the render size in and out, the output ours.
    if (g_periphParams == nullptr &&
        (NVSDK_NGX_VULKAN_AllocateParameters(&g_periphParams) != NVSDK_NGX_Result_Success || g_periphParams == nullptr))
    {
        g_periphParams = nullptr;
        g_periphFailed = true;
        LogOnce(g_loggedPeriphFail, "cheap periphery OFF: no parameter block of its own could be allocated");
        ++g_periphFallbacks;
        return false;
    }
    if (!CopyParameters(params, g_periphParams))
    {
        ++g_periphFallbacks;
        return false;
    }
    NVSDK_NGX_Parameter* const pp = g_periphParams;
    pp->Set(NVSDK_NGX_Parameter_Width, R);
    pp->Set(NVSDK_NGX_Parameter_Height, Rh);
    pp->Set(NVSDK_NGX_Parameter_OutWidth, R);
    pp->Set(NVSDK_NGX_Parameter_OutHeight, Rh);
    pp->Set(NVSDK_NGX_Parameter_DLSS_Output_Subrect_Base_X, 0u);
    pp->Set(NVSDK_NGX_Parameter_DLSS_Output_Subrect_Base_Y, 0u);
    pp->Set(NVSDK_NGX_Parameter_Output, (void*) &g_periphImage.ngx);

    if (it == g_periphs.end())
    {
        Periph p;
        p.owner = ownerHandle;
        p.rw = R;
        p.rh = Rh;
        p.preset = preset;
        p.reset = true;
        if (VkCreateWorker::Available())
        {
            FoveaBox::Rect whole {};
            whole.rw = whole.ow = R;
            whole.rh = whole.oh = Rh;
            const int state = WorkerFeature(g_pendingPeriphs, key, ownerHandle, pp, owner->GetUpscalerType(), preset,
                                            whole, R, Rh, O, Oh, "periphery", &p.feature, &p.block);
            if (state < 0)
            {
                g_periphFailed = true;
                LogOnce(g_loggedPeriphCreateFail,
                        "the periphery feature could not be made; the full pass runs this session");
                ++g_periphFallbacks;
                return false;
            }
            if (state == 0)
            {
                ++g_periphMaking;
                ++g_periphFallbacks;
                return false;
            }
        }
        else
        {
            t_presetForCreate = preset;
            pp->Set(NVSDK_NGX_Parameter_Reset, 1u);
            p.feature = NvngxVkOwn::Create(cmd, owner->GetUpscalerType(), pp);
            t_presetForCreate = 0;
            if (p.feature == 0)
            {
                g_periphFailed = true;
                LogOnce(g_loggedPeriphCreateFail,
                        "the periphery feature could not be made; the full pass runs this session");
                ++g_periphFallbacks;
                return false;
            }
            LOG_INFO("Per-eye DLSS VK: periphery feature {} made for feature {} (eye key {:#x}; {}x{} in and out, "
                     "preset {}) on the game's thread",
                     p.feature, ownerHandle, key, R, Rh, preset);
        }
        it = g_periphs.emplace(key, p).first;
    }
    Periph& p = it->second;
    if (p.reset)
    {
        pp->Set(NVSDK_NGX_Parameter_Reset, 1u);
        p.reset = false;
    }
    IFeature_Vk* feature = NvngxVkOwn::Get(p.feature);
    if (feature == nullptr)
    {
        ++g_periphFallbacks;
        return false;
    }
    IFeature* const currentBefore = State::Instance().currentFeature;
    const bool ok = feature->Evaluate(cmd, pp);
    State::Instance().currentFeature = currentBefore;
    if (!ok)
    {
        ++g_periphFallbacks;
        LogOnce(g_loggedPeriphFail, "the periphery's DLSS evaluate failed (the full pass runs; counted per window)");
        p.reset = true;
        return false;
    }

    // ENLARGE our render-size picture over the whole eye; the crop is composited over the centre next (AfterUpscale).
    FoveaPasses_Vk::EnlargeConstants e {};
    e.outW = O;
    e.outH = Oh;
    e.srcW = R;
    e.srcH = Rh;
    g_passes.Enlarge(cmd, g_slot++, g_periphImage.view, ov.ImageView, e);
    g_mainSkipped[ownerHandle] = true;
    ++g_periphDrawn;
    if (!g_loggedPeriphFirst)
    {
        g_loggedPeriphFirst = true;
        LOG_INFO("Per-eye DLSS VK: FIRST cheap periphery drawn in the place of the full pass (periphery feature {}, "
                 "{}x{} enlarged over {}x{})",
                 p.feature, R, Rh, O, Oh);
    }
    return true;
}

bool TakeMainSkipped(unsigned int ownerHandle)
{
    if (!Enabled())
        return false;
    std::lock_guard lock(g_mutex);
    auto it = g_mainSkipped.find(ownerHandle);
    if (it == g_mainSkipped.end() || !it->second)
        return false;
    it->second = false;
    LOG_INFO("Per-eye DLSS VK: feature {} runs its full pass again after the periphery: its history is reset once",
             ownerHandle);
    return true;
}

void ToggleFovea()
{
    std::lock_guard lock(g_mutex);
    g_foveaOn = !g_foveaOn;
    if (g_foveaOn)
        for (auto& entry : g_crops)
            entry.second.reset = true;
    LOG_INFO("Per-eye DLSS VK: foveal crop {} (switched){}", g_foveaOn ? "ON" : "off",
             g_foveaFailed ? " -- but it failed earlier and stays off" : "");
}

void ToggleOutline()
{
    std::lock_guard lock(g_mutex);
    g_outline = !g_outline;
    LOG_INFO("Per-eye DLSS VK: foveal crop outline {}", g_outline ? "on" : "off");
}

FoveaMenuVk::Status GetStatus()
{
    FoveaMenuVk::Status st;
    const auto* cfg = Config::Instance();
    std::lock_guard lock(g_mutex);
    st.on = g_foveaOn && !g_foveaFailed;
    st.outline = g_outline;
    st.failed = g_foveaFailed;
    st.startOn = cfg->UpscalerFoveaStart.value_or_default();
    st.preset = cfg->UpscalerFoveaPreset.value_or_default();
    st.width = cfg->UpscalerFoveaCrop.value_or_default();
    const float h = cfg->UpscalerFoveaCropHeight.value_or_default();
    st.height = h > 0.0f ? h : st.width;
    st.feather = cfg->UpscalerFoveaFeather.value_or_default();
    st.cropW = g_statusRect.rw;
    st.cropH = g_statusRect.rh;
    st.cropOutW = g_statusRect.ow;
    st.cropOutH = g_statusRect.oh;
    st.cropX = g_statusRect.ox;
    st.cropY = g_statusRect.oy;
    st.eyeW = g_statusEyeW;
    st.eyeH = g_statusEyeH;
    st.exact = g_statusRect.exact;
    st.windowEvals = g_lastWindowEvals;
    st.drawn = g_lastDrawn;
    st.skipped = g_lastSkipped;
    st.failedEvals = g_lastFailed;
    st.crops = (uint32_t) g_crops.size();
    st.eyeByParity = !PerEyeDlss_Vk::RoutingOn() && cfg->UpscalerCropEyeByParity.value_or_default();
    st.periphery = cfg->UpscalerFoveaCheapPeriphery.value_or_default();
    st.peripheryPreset = cfg->UpscalerFoveaPeripheryPreset.value_or_default();
    st.peripheryFailed = g_periphFailed;
    st.peripheryDrawn = g_lastPeriphDrawn;
    st.peripheryFallbacks = g_lastPeriphFallbacks;
    st.peripheries = (uint32_t) g_periphs.size();
    st.presentsPerSec = g_lastPresentsPerSec;
    st.evalsPerPresent = g_lastEvalsPerPresent;
    st.gpuFrames = g_lastGpu.frames;
    st.gpuFrameMs = g_lastGpu.frameMs;
    st.gpuGameMs = g_lastGpu.gameMs;
    st.gpuGameRuns = g_lastGpu.gameRuns;
    st.gpuOursMs = g_lastGpu.beforeMs + g_lastGpu.afterMs;
    st.gpuTiming = cfg->UpscalerGpuTiming.value_or_default();
    st.making = (uint32_t) (g_pendingCrops.size() + g_pendingPeriphs.size());
    st.worker = VkCreateWorker::Available();
    st.copyInputs = cfg->UpscalerFoveaCopyInputs.value_or_default();
    st.cropMask = std::clamp(cfg->UpscalerFoveaCropMask.value_or_default(), 0, 2);
    st.gathered = g_lastGathered;
    st.peripheryTaa = cfg->UpscalerFoveaPeripheryTaa.value_or_default();
    st.taaBlend = cfg->UpscalerFoveaTaaBlend.value_or_default();
    st.taaScale = cfg->UpscalerFoveaTaaScale.value_or_default();
    st.taaFlipJitter = cfg->UpscalerFoveaTaaFlipJitter.value_or_default();
    st.taaFailed = g_taaFailed;
    st.taaEvals = g_lastTaaEvals;
    st.taaResets = g_lastTaaResets;
    st.taaSkips = g_lastTaaSkips;
    return st;
}

void Apply(const FoveaMenuVk::Answer& answer)
{
    // The toggles take the module's lock themselves (never under it: a std::mutex locked twice throws).
    if (answer.toggleCrop)
        ToggleFovea();
    if (answer.toggleOutline)
        ToggleOutline();
    auto* cfg = Config::Instance();
    std::lock_guard lock(g_mutex);
    if (answer.startOn >= 0)
    {
        cfg->UpscalerFoveaStart = answer.startOn != 0;
        LOG_INFO("Per-eye DLSS VK: [Inputs] UpscalerFoveaStart = {} (overlay; saved when the overlay closes)",
                 answer.startOn != 0);
    }
    if (answer.preset >= 0 && answer.preset != cfg->UpscalerFoveaPreset.value_or_default())
    {
        cfg->UpscalerFoveaPreset = answer.preset;
        // AfterUpscale sees the preset differ from the crop's and remakes it at that eye's next evaluate.
        LOG_INFO("Per-eye DLSS VK: foveal crop preset -> {} (overlay); {} crop feature(s) remade at their next "
                 "evaluate",
                 answer.preset, g_crops.size());
    }
    if (answer.width > 0.0f && answer.height > 0.0f)
    {
        cfg->UpscalerFoveaCrop = std::clamp(answer.width, 0.1f, 0.9f);
        cfg->UpscalerFoveaCropHeight = std::clamp(answer.height, 0.1f, 0.9f);
        LOG_INFO("Per-eye DLSS VK: foveal crop size -> {:.2f} x {:.2f} of the eye (overlay); the box is planned anew "
                 "and {} crop feature(s) remade where it changed",
                 cfg->UpscalerFoveaCrop.value_or_default(), cfg->UpscalerFoveaCropHeight.value_or_default(),
                 g_crops.size());
    }
    if (answer.feather >= 0 && answer.feather != cfg->UpscalerFoveaFeather.value_or_default())
        cfg->UpscalerFoveaFeather = std::clamp(answer.feather, 0, 256);
    if (answer.periphery >= 0)
    {
        const bool on = answer.periphery != 0;
        cfg->UpscalerFoveaCheapPeriphery = on;
        // Back on: the peripheries' histories are stale. Off: the game's features' are (TakeMainSkipped).
        for (auto& entry : g_periphs)
            entry.second.reset = true;
        LOG_INFO("Per-eye DLSS VK: cheap periphery {} (overlay){}", on ? "ON" : "off",
                 g_periphFailed ? " -- but it failed earlier and the full pass runs" : "");
    }
    if (answer.peripheryTaa >= 0)
    {
        cfg->UpscalerFoveaPeripheryTaa = answer.peripheryTaa != 0;
        // Whichever periphery takes over has a stale history: the DLSS ones start over; ours restart by the Present gap.
        for (auto& entry : g_periphs)
            entry.second.reset = true;
        LOG_INFO("Per-eye DLSS VK: own TAA periphery {} (overlay)", answer.peripheryTaa != 0 ? "ON" : "off");
    }
    if (answer.taaFlipJitter >= 0)
    {
        cfg->UpscalerFoveaTaaFlipJitter = answer.taaFlipJitter != 0;
        LOG_INFO("Per-eye DLSS VK: TAA jitter {} (overlay)", answer.taaFlipJitter != 0 ? "FLIPPED" : "as handed");
    }
    if (answer.taaBlend > 0.0f)
        cfg->UpscalerFoveaTaaBlend = std::clamp(answer.taaBlend, 0.01f, 1.0f);
    if (answer.taaScale > 0.0f)
    {
        cfg->UpscalerFoveaTaaScale = std::clamp(answer.taaScale, 0.25f, 1.0f);
        LOG_INFO("Per-eye DLSS VK: TAA resolution -> {:.2f} of the eye (overlay; the histories are remade)",
                 cfg->UpscalerFoveaTaaScale.value_or_default());
    }
    if (answer.copyInputs >= 0)
    {
        cfg->UpscalerFoveaCopyInputs = answer.copyInputs != 0;
        for (auto& entry : g_crops)
            entry.second.reset = true; // the inputs change place: the histories start over
        LOG_INFO("Per-eye DLSS VK: the crop's inputs {} (overlay)",
                 answer.copyInputs != 0 ? "COPIED at the box" : "read through subrects");
    }
    if (answer.cropMask >= 0)
    {
        cfg->UpscalerFoveaCropMask = std::clamp(answer.cropMask, 0, 2);
        LOG_INFO("Per-eye DLSS VK: the crop's bias mask mode -> {} (overlay)", answer.cropMask);
    }
    if (answer.peripheryPreset >= 0 && answer.peripheryPreset != cfg->UpscalerFoveaPeripheryPreset.value_or_default())
    {
        cfg->UpscalerFoveaPeripheryPreset = answer.peripheryPreset;
        LOG_INFO("Per-eye DLSS VK: periphery preset -> {} (overlay); {} periphery feature(s) remade at their next "
                 "evaluate",
                 answer.peripheryPreset, g_periphs.size());
    }
}

bool FoveaOn()
{
    std::lock_guard lock(g_mutex);
    return g_foveaOn && !g_foveaFailed;
}

const char* FoveaName()
{
    std::lock_guard lock(g_mutex);
    if (g_foveaFailed)
        return "Fovea failed (see log)";
    return g_foveaOn ? (g_outline ? "Fovea on (outline)" : "Fovea on") : "Fovea off";
}
} // namespace FoveaCrop_Vk
