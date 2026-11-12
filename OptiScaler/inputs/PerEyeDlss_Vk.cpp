#include "pch.h"

#include "PerEyeDlss_Vk.h"
#include "SameEyeFit_Vk.h"

#include "Config.h"
#include "Logger.h"
#include "Util.h"

#include <nvsdk_ngx.h>
#include <nvsdk_ngx_vk.h>
#include <dlssnr/DlssNrFeature_Dx12.h> // kNrEyeParam

#include <algorithm>
#include <atomic>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace PerEyeDlss_Vk
{
namespace
{
using SameEyeFit_Vk::kRing;

std::mutex g_mutex;
std::atomic<uint64_t> g_present { 0 };

// The eye's clock (see the header): the Presents once seen to advance between evaluates (30 times), else the evaluates.
uint64_t g_evals = 0;
uint64_t g_presentAtLastEval = UINT64_MAX;
uint32_t g_presentTicks = 0;
bool g_presentClock = false;
bool g_loggedClock = false;
bool g_loggedNrEyeByParity = false;
uint64_t Clock() { return g_presentClock ? g_present.load() : g_evals; }

std::unordered_map<unsigned int, unsigned int> g_twins;
bool g_routing = false;
bool g_resetPrimary = false, g_resetTwin = false;

// The fit.
SameEyeFit_Vk::Fitter g_fitter;
bool g_fitTried = false;
VkDevice g_device = VK_NULL_HANDLE;
VkPhysicalDevice g_physical = VK_NULL_HANDLE;
uint64_t g_fitSeq = 0;
uint32_t g_fitSlotParity[kRing] = {};
uint64_t g_fitClock[2] = { UINT64_MAX, UINT64_MAX }; // the clock of each parity's last fit

// The compose's outputs, ours (GENERAL from their first use), and the NGX description DLSS gets.
struct OwnImage
{
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
};
OwnImage g_composed, g_mask;
uint32_t g_composedW = 0, g_composedH = 0;
uint32_t g_pendingW = 0, g_pendingH = 0, g_pendingCount = 0;
NVSDK_NGX_Resource_VK g_composedNgx {};
// A depth-aspect view of the game's depth image when its own view also holds stencil (not sampleable).
VkImage g_depthViewOf = VK_NULL_HANDLE;
VkImageView g_depthView = VK_NULL_HANDLE;
std::vector<std::pair<VkImageView, uint64_t>> g_retiredViews; // destroyed kRing * 2 evaluates later

// The parameters' vectors while ours are in them.
void* g_swappedFrom = nullptr;
bool g_swapped = false;

// Per window (600 evaluates).
uint64_t g_windowStartEvals = 0, g_windowStartPresent = 0;
uint64_t g_toPrimary = 0, g_toTwin = 0, g_noTwin = 0, g_composedEvals = 0, g_chainBreaks = 0, g_sizeSkips = 0;
uint64_t g_fitsRead = 0, g_fitsValid = 0, g_fitsNotReady = 0, g_fitReasons[4] = {}, g_fitGaveUp = 0, g_fitRestarted = 0;
uint64_t g_fitLeftOutFits = 0;
double g_fitLeftOutSum = 0, g_fitResidualSum = 0, g_fitInlierSum = 0;
uint64_t g_parallaxSign[2][2] = {};
uint64_t g_lastWindowRead = 0, g_lastWindowValid = 0;

// The dump.
uint32_t g_dumpLeft = 0;
std::atomic<uint32_t> g_dumpSaved { 0 };
std::string g_dumpFolder;
struct DumpMeta
{
    bool armed = false;
    uint64_t seq = 0, clock = 0;
    uint32_t parity = 0, keepStale = 0, mw = 0, mh = 0, mf = 0, df = 0;
    float scale[2] = {};
};
DumpMeta g_dumpMeta[kRing];

bool g_loggedFirstFit = false, g_loggedFirstComposed = false, g_loggedNoInputs = false, g_loggedFitFail = false,
     g_loggedSize = false, g_loggedStencilView = false, g_loggedBias = false, g_loggedNotReady = false,
     g_loggedOtherDevice = false, g_loggedImageFail = false;

void LogOnce(bool& logged, const std::string& text)
{
    if (logged)
        return;
    logged = true;
    LOG_WARN("Per-eye DLSS VK: {}", text);
}

int ObjectMotionMode() { return std::clamp(Config::Instance()->UpscalerSameEyeObjectMotion.value_or_default(), 0, 3); }
int BiasMaskMode() { return std::clamp(Config::Instance()->UpscalerSameEyeBiasMask.value_or_default(), 0, 2); }

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

bool MakeImage(OwnImage& im, uint32_t w, uint32_t h, VkFormat format)
{
    VkImageCreateInfo ci { VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
    ci.imageType = VK_IMAGE_TYPE_2D;
    ci.format = format;
    ci.extent = { w, h, 1 };
    ci.mipLevels = 1;
    ci.arrayLayers = 1;
    ci.samples = VK_SAMPLE_COUNT_1_BIT;
    ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    ci.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
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
        return false;
    VkImageViewCreateInfo vi { VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
    vi.image = im.image;
    vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vi.format = format;
    vi.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    return vkCreateImageView(g_device, &vi, nullptr, &im.view) == VK_SUCCESS;
}

// Our two images at the motion's size. The first size is taken at once; a new size only after 90 evaluates at it (a
// menu, a resize), then with the GPU idle (our images may be in flight) -- the evaluates in between reset the history.
bool EnsureComposed(uint32_t w, uint32_t h)
{
    if (g_composed.image != VK_NULL_HANDLE && w == g_composedW && h == g_composedH)
    {
        g_pendingCount = 0;
        return true;
    }
    if (g_composed.image != VK_NULL_HANDLE)
    {
        if (w != g_pendingW || h != g_pendingH)
        {
            g_pendingW = w;
            g_pendingH = h;
            g_pendingCount = 0;
        }
        if (++g_pendingCount < 90)
            return false;
        vkDeviceWaitIdle(g_device);
        DestroyImage(g_composed);
        DestroyImage(g_mask);
    }
    if (!MakeImage(g_composed, w, h, VK_FORMAT_R16G16_SFLOAT) || !MakeImage(g_mask, w, h, VK_FORMAT_R8_UNORM))
    {
        DestroyImage(g_composed);
        DestroyImage(g_mask);
        LogOnce(g_loggedImageFail, std::format("same-eye motion off: our {}x{} images could not be made", w, h));
        return false;
    }
    g_composedW = w;
    g_composedH = h;
    g_pendingCount = 0;
    g_composedNgx = {};
    g_composedNgx.Type = NVSDK_NGX_RESOURCE_VK_TYPE_VK_IMAGEVIEW;
    g_composedNgx.Resource.ImageViewInfo.ImageView = g_composed.view;
    g_composedNgx.Resource.ImageViewInfo.Image = g_composed.image;
    g_composedNgx.Resource.ImageViewInfo.SubresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    g_composedNgx.Resource.ImageViewInfo.Format = VK_FORMAT_R16G16_SFLOAT;
    g_composedNgx.Resource.ImageViewInfo.Width = w;
    g_composedNgx.Resource.ImageViewInfo.Height = h;
    g_composedNgx.ReadWrite = true;
    LOG_INFO("Per-eye DLSS VK: same-eye motion images {}x{} made", w, h);
    return true;
}

// The depth view the fit samples: the game's own, or one of the depth aspect alone when its view also holds stencil.
VkImageView DepthView(const NVSDK_NGX_Resource_VK* depth)
{
    const auto& info = depth->Resource.ImageViewInfo;
    if ((info.SubresourceRange.aspectMask & VK_IMAGE_ASPECT_STENCIL_BIT) == 0)
        return info.ImageView;
    if (g_depthView != VK_NULL_HANDLE && g_depthViewOf == info.Image)
        return g_depthView;
    if (g_depthView != VK_NULL_HANDLE)
        g_retiredViews.push_back({ g_depthView, g_evals });
    g_depthView = VK_NULL_HANDLE;
    VkImageViewCreateInfo vi { VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
    vi.image = info.Image;
    vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vi.format = info.Format;
    vi.subresourceRange = info.SubresourceRange;
    vi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    if (vkCreateImageView(g_device, &vi, nullptr, &g_depthView) != VK_SUCCESS)
    {
        g_depthView = VK_NULL_HANDLE;
        return VK_NULL_HANDLE;
    }
    g_depthViewOf = info.Image;
    if (!g_loggedStencilView)
    {
        g_loggedStencilView = true;
        LOG_INFO("Per-eye DLSS VK: the game's depth view holds stencil too (format {}); the fit reads a depth-aspect "
                 "view of its own",
                 (int) info.Format);
    }
    return g_depthView;
}

void RetireViews()
{
    for (auto it = g_retiredViews.begin(); it != g_retiredViews.end();)
    {
        if (g_evals - it->second > kRing * 2)
        {
            vkDestroyImageView(g_device, it->first, nullptr);
            it = g_retiredViews.erase(it);
        }
        else
            ++it;
    }
}

void WriteDump(const DumpMeta& meta, const uint8_t* d, const float* f)
{
    std::vector<uint8_t> samples(d + 16, d + SameEyeFit_Vk::kDumpBytes);
    std::vector<float> fit(f, f + 60);
    const std::string folder = g_dumpFolder;
    std::thread(
        [meta, samples = std::move(samples), fit = std::move(fit), folder]
        {
            std::error_code error;
            std::filesystem::create_directories(folder, error);
            const auto base = std::filesystem::path(folder) / std::format("fit_{:08}_p{}", meta.seq, meta.parity);
            std::ofstream bin(base.string() + ".bin", std::ios::binary);
            bin.write((const char*) samples.data(), (std::streamsize) samples.size());
            const float* m = fit.data() + 2 * 20;
            std::ofstream txt(base.string() + ".txt");
            txt << std::format("Vulkan same-eye fit sample dump (SameEyeFit_Vk). {}.bin: 4096 samples x 4 float32 over a "
                               "64x64 grid (as the D3D12 dump): raw depth, raw motion x, y (before MV scale), 1.0 if "
                               "fitted\nseq {} clock {} parity {} keepStale {}\nmotion {}x{} fmt {}, depth fmt {}, MV "
                               "scale {} {}\nverdict: reason {}, inlier residual {:.4f} px, within 1 px {:.2f}%, samples "
                               "counted {}\nthis fit rows: x {:.7f} {:.7f} {:.7f} {:.7f} | y {:.7f} {:.7f} {:.7f} "
                               "{:.7f} | w {:.7f} {:.7f} {:.7f}\n",
                               base.filename().string(), meta.seq, meta.clock, meta.parity, meta.keepStale, meta.mw,
                               meta.mh, meta.mf, meta.df, meta.scale[0], meta.scale[1], m[16], m[12], m[13] * 100.0f,
                               m[15], m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8], m[9], m[10]);
            const bool ok = bin.good() && txt.good();
            if (ok)
                ++g_dumpSaved;
            LOG_INFO("Per-eye DLSS VK: FIT DUMP: fit {} (parity {}, reason {}, {:.1f}% within 1 px) {} {}", meta.seq,
                     meta.parity, (int) m[16], m[13] * 100.0f, ok ? "saved to" : "FAILED to write", base.string());
        })
        .detach();
}

// The fit of the slot about to be reused, kRing evaluates after it was recorded (its stamp must match).
void ReadFitSlot(uint32_t slot)
{
    if (g_fitSeq < kRing)
        return;
    float f[60];
    std::memcpy(f, g_fitter.FitSlot(slot), sizeof(f));
    const float* m = f + 2 * 20;
    const float expected = (float) ((g_fitSeq - kRing) & 0xFFFFFF);
    if (!(m[19] == expected))
    {
        ++g_fitsNotReady;
        if (g_dumpMeta[slot].armed)
            g_dumpMeta[slot].armed = false;
        LogOnce(g_loggedNotReady, std::format("a fit slot was not written {} evaluates later (stamp {}, expected {}); "
                                              "counted per window",
                                              kRing, m[19], expected));
        return;
    }
    ++g_fitsRead;
    const int reason = std::clamp((int) m[16], 0, 3);
    ++g_fitReasons[reason];
    if (m[17] > 0.5f)
        ++g_fitGaveUp;
    const float restarted = m[18] > 0.5f ? 1.0f : 0.0f;
    if (restarted > 0)
        ++g_fitRestarted;
    const float leftOut = (m[18] - restarted) / 0.45f;
    if (leftOut > 0.0f)
    {
        g_fitLeftOutSum += leftOut;
        ++g_fitLeftOutFits;
    }
    if (m[14] > 0.5f)
    {
        ++g_fitsValid;
        g_fitResidualSum += m[12];
        g_fitInlierSum += m[13];
        ++g_parallaxSign[g_fitSlotParity[slot] & 1][m[2] > 0.0f ? 1 : 0];
    }
    if (g_dumpMeta[slot].armed)
    {
        g_dumpMeta[slot].armed = false;
        const uint8_t* d = g_fitter.DumpSlot(slot);
        uint32_t stamp = 0;
        std::memcpy(&stamp, d + 4, 4);
        if (stamp == ((uint32_t) (g_fitSeq - kRing) & 0xFFFFFFu))
            WriteDump(g_dumpMeta[slot], d, f);
        else
            LOG_WARN("Per-eye DLSS VK: FIT DUMP: fit {} carries stamp {}; not saved", g_dumpMeta[slot].seq, stamp);
    }
    if (!g_loggedFirstFit)
    {
        g_loggedFirstFit = true;
        LOG_INFO("Per-eye DLSS VK: FIRST fit read back: {} (reason {}) -- inlier residual {:.4f} px, {:.1f}% of {} "
                 "samples within 1 px, {:.0f}% left out at a depth gap; M rows x ({:.5f} {:.5f} {:.5f} {:.5f}) y ({:.5f} "
                 "{:.5f} {:.5f} {:.5f}) w ({:.5f} {:.5f} {:.5f} 1)",
                 m[14] > 0.5f ? "VALID" : "REJECTED", reason, m[12], m[13] * 100.0f, (int) m[15], leftOut * 100.0f,
                 m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8], m[9], m[10]);
    }
}

void WindowSummary()
{
    if (g_evals - g_windowStartEvals < 600)
        return;
    const uint64_t presents = g_present.load() - g_windowStartPresent;
    LOG_INFO("Per-eye DLSS VK: {} evaluates, {} Presents (eye clock: {}), routing {}: {} to the game's feature, {} to "
             "the twin, {} with no twin; same-eye motion on {} evaluates, {} chain breaks (history reset), {} size skips; "
             "fits read {} ({} not ready), valid {}, mean inlier residual {:.4f} px, mean inliers {:.1f}%; rejected: {} "
             "solve failed, {} too few depth samples, {} too few within 1 px; stale start given up {}, restarted within "
             "the fit {}; a depth gap found on {} fits, {:.1f}% left out on average (switch {}); parallax sign even +{} "
             "-{}, odd +{} -{}",
             g_evals - g_windowStartEvals, presents, g_presentClock ? "Presents" : "evaluates",
             g_routing ? "PER EYE" : "shared", g_toPrimary, g_toTwin, g_noTwin, g_composedEvals, g_chainBreaks,
             g_sizeSkips, g_fitsRead, g_fitsNotReady, g_fitsValid, g_fitsValid ? g_fitResidualSum / g_fitsValid : 0.0,
             g_fitsValid ? 100.0 * g_fitInlierSum / g_fitsValid : 0.0, g_fitReasons[1], g_fitReasons[2],
             g_fitReasons[3], g_fitGaveUp, g_fitRestarted, g_fitLeftOutFits,
             g_fitLeftOutFits ? 100.0 * g_fitLeftOutSum / g_fitLeftOutFits : 0.0,
             Config::Instance()->UpscalerFitLeaveOutNearest.value_or_default() ? "ON" : "OFF", g_parallaxSign[0][1],
             g_parallaxSign[0][0], g_parallaxSign[1][1], g_parallaxSign[1][0]);
    g_lastWindowRead = g_fitsRead;
    g_lastWindowValid = g_fitsValid;
    g_windowStartEvals = g_evals;
    g_windowStartPresent = g_present.load();
    g_toPrimary = g_toTwin = g_noTwin = g_composedEvals = g_chainBreaks = g_sizeSkips = 0;
    g_fitsRead = g_fitsValid = g_fitsNotReady = g_fitGaveUp = g_fitRestarted = g_fitLeftOutFits = 0;
    std::memset(g_fitReasons, 0, sizeof(g_fitReasons));
    g_fitLeftOutSum = g_fitResidualSum = g_fitInlierSum = 0;
    std::memset(g_parallaxSign, 0, sizeof(g_parallaxSign));
}

// Under g_mutex: the fit (and the compose when the other eye's fit is the previous tick's) and the hand-over.
void SameEye(VkCommandBuffer cmd, NVSDK_NGX_Parameter* params)
{
    auto reset = [&] { params->Set(NVSDK_NGX_Parameter_Reset, 1); };
    if (!g_fitTried)
    {
        g_fitTried = true;
        std::string error;
        if (!g_fitter.Init(g_device, g_physical, error))
        {
            g_fitter.Destroy();
            LogOnce(g_loggedFitFail, "same-eye motion off for the session: the fit could not be set up (" + error + ")");
        }
        else
            LOG_INFO("Per-eye DLSS VK: same-eye motion ready (SameEyeFit_Vk: the D3D12 fit and compose as SPIR-V)");
    }
    if (!g_fitter.Ready())
        return reset();

    void* motionPtr = nullptr;
    void* depthPtr = nullptr;
    params->Get(NVSDK_NGX_Parameter_MotionVectors, &motionPtr);
    params->Get(NVSDK_NGX_Parameter_Depth, &depthPtr);
    float scale[2] = { 0, 0 };
    params->Get(NVSDK_NGX_Parameter_MV_Scale_X, &scale[0]);
    params->Get(NVSDK_NGX_Parameter_MV_Scale_Y, &scale[1]);
    auto skip = [&](const char* why)
    {
        LogOnce(g_loggedNoInputs, std::format("same-eye motion skipped ({}); this eye's history is reset instead", why));
        ++g_chainBreaks;
        reset();
    };
    const auto* motion = (const NVSDK_NGX_Resource_VK*) motionPtr;
    const auto* depth = (const NVSDK_NGX_Resource_VK*) depthPtr;
    if (motion == nullptr || depth == nullptr)
        return skip("no motion or no depth in the DLSS parameters");
    if (motion->Type != NVSDK_NGX_RESOURCE_VK_TYPE_VK_IMAGEVIEW || depth->Type != NVSDK_NGX_RESOURCE_VK_TYPE_VK_IMAGEVIEW)
        return skip("motion or depth is not an image view");
    if (scale[0] == 0 || scale[1] == 0)
        return skip("MV scale is zero");
    const auto& mi = motion->Resource.ImageViewInfo;
    const auto& di = depth->Resource.ImageViewInfo;
    if (!EnsureComposed(mi.Width, mi.Height))
    {
        ++g_sizeSkips;
        LogOnce(g_loggedSize, std::format("same-eye motion skipped at motion size {}x{} (images {}x{}; a new size is "
                                          "adopted after 90 evaluates); counted per window",
                                          mi.Width, mi.Height, g_composedW, g_composedH));
        return reset();
    }
    const VkImageView depthView = DepthView(depth);
    if (depthView == VK_NULL_HANDLE)
        return skip("no depth-aspect view of the game's depth could be made");
    if (BiasMaskMode() != 0)
        LogOnce(g_loggedBias, "the bias mask (UpscalerSameEyeBiasMask) is not handed to DLSS on Vulkan yet");

    const uint64_t clock = Clock();
    const uint32_t cur = (uint32_t) (clock & 1), prev = cur ^ 1;
    const bool chain = clock > 0 && g_fitClock[prev] == clock - 1;
    const uint32_t slot = (uint32_t) (g_fitSeq % kRing);
    ReadFitSlot(slot);
    g_fitSlotParity[slot] = cur;

    const int objectMode = ObjectMotionMode();
    SameEyeFit_Vk::Constants c {};
    c.mvScale[0] = scale[0];
    c.mvScale[1] = scale[1];
    c.cur = cur;
    c.prev = prev;
    c.knee0 = objectMode == 1 ? 0.1f : objectMode == 3 ? 0.03f : 0.02f;
    c.knee1 = objectMode == 1 ? 0.4f : objectMode == 3 ? 0.15f : 0.1f;
    c.gain = objectMode == 0 ? 0.0f : 2.0f;
    c.radius = 2u | ((uint32_t) objectMode << 8) | ((uint32_t) BiasMaskMode() << 16);
    c.stamp = (uint32_t) g_fitSeq;
    c.keepStale = (Config::Instance()->UpscalerFitGiveUpStale.value_or_default() ? 0u : 1u) |
                  (Config::Instance()->UpscalerFitRestart.value_or_default() ? 0u : 2u) |
                  (Config::Instance()->UpscalerFitLeaveOutNearest.value_or_default() ? 0u : 4u) |
                  (g_dumpLeft > 0 ? 8u : 0u);
    c.linearDepth = Config::Instance()->UpscalerFitDepth.value_or_default() == 2 ? 1u : 0u; // SR: hardware unless set

    SameEyeFit_Vk::Inputs in;
    in.depth = depthView;
    in.motion = mi.ImageView;
    if (chain)
    {
        in.composedImage = g_composed.image;
        in.composed = g_composed.view;
        in.maskImage = g_mask.image;
        in.mask = g_mask.view;
        in.width = mi.Width;
        in.height = mi.Height;
    }
    g_fitter.Record(cmd, slot, in, c);
    if (c.keepStale & 8u)
    {
        DumpMeta& meta = g_dumpMeta[slot];
        meta.armed = true;
        meta.seq = g_fitSeq;
        meta.clock = clock;
        meta.parity = cur;
        meta.keepStale = c.keepStale;
        meta.mw = mi.Width;
        meta.mh = mi.Height;
        meta.mf = (uint32_t) mi.Format;
        meta.df = (uint32_t) di.Format;
        meta.scale[0] = scale[0];
        meta.scale[1] = scale[1];
        --g_dumpLeft;
    }
    ++g_fitSeq;
    g_fitClock[cur] = clock;

    if (!chain)
    {
        // No fit of the previous tick: nothing to chain, so this eye's history starts over rather than being fed the
        // other eye's vectors.
        ++g_chainBreaks;
        return reset();
    }
    g_swappedFrom = motionPtr;
    params->Set(NVSDK_NGX_Parameter_MotionVectors, (void*) &g_composedNgx);
    g_swapped = true;
    ++g_composedEvals;
    if (!g_loggedFirstComposed)
    {
        g_loggedFirstComposed = true;
        LOG_INFO("Per-eye DLSS VK: FIRST same-eye motion handed to DLSS ({}x{} fmt {}, clock {}; depth {}x{} fmt {}, "
                 "aspect 0x{:x}; MV scale {} {})",
                 mi.Width, mi.Height, (int) mi.Format, clock, di.Width, di.Height, (int) di.Format,
                 di.SubresourceRange.aspectMask, scale[0], scale[1]);
    }
}
} // namespace

bool Enabled()
{
    static const bool enabled = Config::Instance()->UpscalerPerEye.value_or_default();
    return enabled;
}

void OnPresent() { ++g_present; }

void RegisterTwin(unsigned int primary, unsigned int twin)
{
    std::lock_guard lock(g_mutex);
    g_twins[primary] = twin;
}

unsigned int TwinOf(unsigned int primary)
{
    std::lock_guard lock(g_mutex);
    auto it = g_twins.find(primary);
    return it == g_twins.end() ? 0 : it->second;
}

void Forget(unsigned int primary)
{
    std::lock_guard lock(g_mutex);
    g_twins.erase(primary);
}

unsigned int Route(unsigned int handle, NVSDK_NGX_Parameter* params)
{
    if (!Enabled())
        return handle;
    std::lock_guard lock(g_mutex);
    ++g_evals;
    const uint64_t present = g_present.load();
    if (!g_presentClock)
    {
        if (g_presentAtLastEval != UINT64_MAX && present != g_presentAtLastEval)
            ++g_presentTicks;
        if (g_presentTicks >= 30)
            g_presentClock = true;
    }
    g_presentAtLastEval = present;
    if (!g_loggedClock && (g_presentClock || g_evals >= 300))
    {
        g_loggedClock = true;
        LOG_INFO("Per-eye DLSS VK: the eye is the parity of the {} ({} of the first {} evaluates came after a new "
                 "Present)",
                 g_presentClock ? "Presents (vkQueuePresentKHR)" : "EVALUATES (the Presents do not advance between "
                                                                   "them)",
                 g_presentTicks, g_evals);
    }
    auto it = g_twins.find(handle);
    // NR (DlssNrFeature_Vk) keeps one model history per eye when told the eye: set on every call (the parameters are
    // the game's and keep a value between evaluates) -- the parity while the routing is per eye (NR then reads our
    // same-eye vectors, swapped in by BeforeEvaluate), eye 0 otherwise (one history, the game's vectors, as before).
    // Ported from the D3D12 twin: with the routing SHARED, eye 0 named one history that took
    // the eyes in turn = the stereo flicker (9 of 9 'pass built' lines for eye 0). Now the parity
    // names the eye whenever this module is on (UpscalerNrEyeByParity), so NR keeps one history per eye under the
    // VR mod's DLSS pass (DL(S)SS) with no routing of ours -- the preferred configuration.
    const bool byParity = Config::Instance()->UpscalerNrEyeByParity.value_or_default();
    const bool perEyeNr = (it != g_twins.end() && g_routing) || byParity;
    if (params != nullptr)
        params->Set(DlssNr::kNrEyeParam, perEyeNr ? (unsigned int) (Clock() & 1) : 0u);
    if (!g_loggedNrEyeByParity && params != nullptr)
    {
        g_loggedNrEyeByParity = true;
        LOG_INFO("Per-eye DLSS VK: NR is told the eye by the Present's parity {} (UpscalerNrEyeByParity {})",
                 byParity ? "in every routing" : "only while the routing is per eye (shared = eye 0, as before)",
                 byParity ? "true" : "false");
    }
    if (it == g_twins.end())
    {
        ++g_noTwin;
        return handle;
    }
    const bool toTwin = g_routing && (Clock() & 1) != 0;
    bool& reset = toTwin ? g_resetTwin : g_resetPrimary;
    if (reset && params != nullptr)
    {
        params->Set(NVSDK_NGX_Parameter_Reset, 1);
        reset = false;
        LOG_INFO("Per-eye DLSS VK: history reset for the {} (routing {})", toTwin ? "twin" : "game's feature",
                 g_routing ? "per eye" : "shared");
    }
    if (toTwin)
    {
        ++g_toTwin;
        return it->second;
    }
    ++g_toPrimary;
    return handle;
}

void BeforeEvaluate(VkCommandBuffer cmd, NVSDK_NGX_Parameter* params, VkDevice device, VkPhysicalDevice physical)
{
    if (!Enabled() || cmd == VK_NULL_HANDLE || params == nullptr)
        return;
    std::lock_guard lock(g_mutex);
    g_swapped = false;
    if (g_device == VK_NULL_HANDLE)
    {
        g_device = device;
        g_physical = physical;
    }
    if (device != g_device)
    {
        if (g_routing)
            params->Set(NVSDK_NGX_Parameter_Reset, 1);
        LogOnce(g_loggedOtherDevice, "passes skipped: an evaluate on another device");
        return;
    }
    RetireViews();
    if (g_routing && !g_twins.empty())
        SameEye(cmd, params);
    WindowSummary();
}

void RestoreInputs(NVSDK_NGX_Parameter* params)
{
    if (!Enabled())
        return;
    std::lock_guard lock(g_mutex);
    if (g_swapped && params != nullptr)
        params->Set(NVSDK_NGX_Parameter_MotionVectors, g_swappedFrom);
    g_swapped = false;
    g_swappedFrom = nullptr;
}

void ToggleRouting()
{
    std::lock_guard lock(g_mutex);
    g_routing = !g_routing;
    g_resetPrimary = g_resetTwin = true;
    g_fitClock[0] = g_fitClock[1] = UINT64_MAX;
    LOG_INFO("Per-eye DLSS VK: routing {}; both histories reset", g_routing ? "PER EYE" : "shared");
}

bool RoutingOn()
{
    std::lock_guard lock(g_mutex);
    return g_routing;
}

unsigned int EyeParity()
{
    std::lock_guard lock(g_mutex);
    return (unsigned int) (Clock() & 1);
}

void RequestFitDump()
{
    std::lock_guard lock(g_mutex);
    if (!g_fitter.Ready())
    {
        LOG_WARN("Per-eye DLSS VK: FIT DUMP refused: same-eye motion is not running (per-eye routing, F6)");
        return;
    }
    const auto now = std::time(nullptr);
    std::tm local {};
    localtime_s(&local, &now);
    g_dumpFolder = (Util::ExePath().parent_path() / "OptiFitDump" /
                    std::format("{:04}{:02}{:02}-{:02}{:02}{:02}", local.tm_year + 1900, local.tm_mon + 1, local.tm_mday,
                                local.tm_hour, local.tm_min, local.tm_sec))
                       .string();
    g_dumpLeft = 16;
    LOG_INFO("Per-eye DLSS VK: FIT DUMP armed: the next 16 same-eye fits go to {}", g_dumpFolder);
}

Status GetStatus()
{
    std::lock_guard lock(g_mutex);
    Status st;
    st.routing = g_routing;
    st.twin = !g_twins.empty();
    st.presentClock = g_presentClock;
    st.fitsRead = g_lastWindowRead;
    st.fitsValid = g_lastWindowValid;
    st.dumpLeft = g_dumpLeft;
    st.dumpSaved = g_dumpSaved.load();
    return st;
}
} // namespace PerEyeDlss_Vk
