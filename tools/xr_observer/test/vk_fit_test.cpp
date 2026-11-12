// Offline test of the VULKAN same-eye fit and compose (OptiScaler/inputs/SameEyeFit_Vk, for PerEyeDlss_Vk): the very
// recorder the DLL uses, on a real Vulkan device, with the SPIR-V compiled from the D3D12 source
// (vk_fit_shaders.py -> PerEyeFitShader_Vk.h). The cases are fit12_test's, as far as they test the shader rather than
// D3D12 plumbing; the pictures and expectations are the same:
//   1  no start: accepted, the map found
//   2  a start 5 px off: accepted at once, no restart
//   8  the chain: a fit for each eye (two maps), then the COMPOSE: the same-eye motion on static pixels within 0.2 px of
//      the true chained maps (the compose's output read back from our R16G16_SFLOAT image)
//   13 the dump (KeepStale bit 3): 4096 samples bit for bit as uploaded, and the stamp
// The depth is a D32_SFLOAT image read through a depth-aspect view, as a game hands DLSS; the motion R16G16_SFLOAT.
// Control, must FAIL:
//   VK_FIT_TEST_NO_DUMP=1     the dump's bit not set: case 13 fails
#include <windows.h>

#include "../../../OptiScaler/inputs/SameEyeFit_Vk.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace
{
using SameEyeFit_Vk::Constants;
using SameEyeFit_Vk::Fitter;
using SameEyeFit_Vk::Inputs;

constexpr uint32_t kSize = 1024;
constexpr float kKZ = 100.0f;
constexpr float kLinearK = 37.0f;

struct Map
{
    float r0[4], r1[4], r2[3];
};
const Map kTrue = { { 1.0f, 0.0003f, -0.008f, -0.0004f }, { -0.0003f, 1.0f, 0.0f, 0.0002f }, { 0.0001f, -0.0002f, 0.0f } };
const Map kOther = { { 1.0f, 0.0002f, 0.006f, 0.0005f }, { -0.0002f, 1.0f, 0.0f, -0.0001f }, { -0.0001f, 0.0001f, 0.0f } };

uint32_t g_random = 0x2468ace1u;
float Random()
{
    g_random ^= g_random << 13;
    g_random ^= g_random >> 17;
    g_random ^= g_random << 5;
    return (float) (g_random & 0xffffff) / (float) 0x1000000;
}

uint16_t Half(float value)
{
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    const int exponent = (int) ((bits >> 23) & 0xff) - 127 + 15;
    const uint32_t mantissa = bits & 0x7fffffu;
    if (exponent <= 0)
    {
        if (exponent < -10)
            return (uint16_t) sign;
        const uint32_t m = (mantissa | 0x800000u) >> (1 - exponent);
        return (uint16_t) (sign | ((m + 0x1000u) >> 13));
    }
    if (exponent >= 31)
        return (uint16_t) (sign | 0x7c00u);
    return (uint16_t) (sign | (((uint32_t) exponent << 10) + ((mantissa + 0x1000u) >> 13)));
}

float FromHalf(uint16_t h)
{
    const uint32_t sign = (uint32_t) (h & 0x8000u) << 16;
    const int exponent = (h >> 10) & 0x1f;
    const uint32_t mantissa = h & 0x3ffu;
    uint32_t bits;
    if (exponent == 0)
    {
        const float v = (float) mantissa / 1024.0f / 16384.0f;
        return (h & 0x8000u) ? -v : v;
    }
    if (exponent == 31)
        bits = sign | 0x7f800000u | (mantissa << 13);
    else
        bits = sign | ((uint32_t) (exponent - 15 + 127) << 23) | (mantissa << 13);
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

// ---- Vulkan ----
VkInstance g_instance = VK_NULL_HANDLE;
VkPhysicalDevice g_physical = VK_NULL_HANDLE;
VkDevice g_device = VK_NULL_HANDLE;
VkQueue g_queue = VK_NULL_HANDLE;
VkCommandPool g_pool = VK_NULL_HANDLE;
VkCommandBuffer g_cmd = VK_NULL_HANDLE;
VkFence g_fence = VK_NULL_HANDLE;
std::string g_gpuName;

bool Check(VkResult r, const char* what)
{
    if (r != VK_SUCCESS)
        std::printf("FAILED: %s (VkResult %d)\n", what, (int) r);
    return r == VK_SUCCESS;
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

bool Setup()
{
    VkApplicationInfo app { VK_STRUCTURE_TYPE_APPLICATION_INFO };
    app.pApplicationName = "vk_fit_test";
    app.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo ii { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
    ii.pApplicationInfo = &app;
    if (!Check(vkCreateInstance(&ii, nullptr, &g_instance), "vkCreateInstance"))
        return false;
    uint32_t count = 0;
    vkEnumeratePhysicalDevices(g_instance, &count, nullptr);
    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(g_instance, &count, devices.data());
    for (auto d : devices)
    {
        VkPhysicalDeviceProperties p {};
        vkGetPhysicalDeviceProperties(d, &p);
        if (g_physical == VK_NULL_HANDLE || p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
        {
            g_physical = d;
            g_gpuName = p.deviceName;
        }
    }
    if (g_physical == VK_NULL_HANDLE)
    {
        std::printf("FAILED: no Vulkan device\n");
        return false;
    }
    uint32_t families = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(g_physical, &families, nullptr);
    std::vector<VkQueueFamilyProperties> fp(families);
    vkGetPhysicalDeviceQueueFamilyProperties(g_physical, &families, fp.data());
    uint32_t family = UINT32_MAX;
    for (uint32_t i = 0; i < families && family == UINT32_MAX; ++i)
        if ((fp[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && (fp[i].queueFlags & VK_QUEUE_COMPUTE_BIT))
            family = i;
    float priority = 1.0f;
    VkDeviceQueueCreateInfo qi { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
    qi.queueFamilyIndex = family;
    qi.queueCount = 1;
    qi.pQueuePriorities = &priority;
    VkPhysicalDeviceFeatures features {};
    VkDeviceCreateInfo di { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
    di.queueCreateInfoCount = 1;
    di.pQueueCreateInfos = &qi;
    di.pEnabledFeatures = &features;
    if (!Check(vkCreateDevice(g_physical, &di, nullptr, &g_device), "vkCreateDevice"))
        return false;
    vkGetDeviceQueue(g_device, family, 0, &g_queue);
    VkCommandPoolCreateInfo pi { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
    pi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pi.queueFamilyIndex = family;
    if (!Check(vkCreateCommandPool(g_device, &pi, nullptr, &g_pool), "vkCreateCommandPool"))
        return false;
    VkCommandBufferAllocateInfo ai { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    ai.commandPool = g_pool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    if (!Check(vkAllocateCommandBuffers(g_device, &ai, &g_cmd), "vkAllocateCommandBuffers"))
        return false;
    VkFenceCreateInfo fi { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    return Check(vkCreateFence(g_device, &fi, nullptr, &g_fence), "vkCreateFence");
}

void Begin()
{
    vkResetCommandBuffer(g_cmd, 0);
    VkCommandBufferBeginInfo bi { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(g_cmd, &bi);
}

bool Submit()
{
    vkEndCommandBuffer(g_cmd);
    VkSubmitInfo si { VK_STRUCTURE_TYPE_SUBMIT_INFO };
    si.commandBufferCount = 1;
    si.pCommandBuffers = &g_cmd;
    vkResetFences(g_device, 1, &g_fence);
    if (!Check(vkQueueSubmit(g_queue, 1, &si, g_fence), "vkQueueSubmit"))
        return false;
    return Check(vkWaitForFences(g_device, 1, &g_fence, VK_TRUE, 10'000'000'000ull), "vkWaitForFences (device lost?)");
}

struct Buffer
{
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void* mapped = nullptr;
};

Buffer HostBuffer(VkDeviceSize size, VkBufferUsageFlags usage)
{
    Buffer b;
    VkBufferCreateInfo bi { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bi.size = size;
    bi.usage = usage;
    vkCreateBuffer(g_device, &bi, nullptr, &b.buffer);
    VkMemoryRequirements req {};
    vkGetBufferMemoryRequirements(g_device, b.buffer, &req);
    VkMemoryAllocateInfo ai { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    ai.allocationSize = req.size;
    ai.memoryTypeIndex =
        MemoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    vkAllocateMemory(g_device, &ai, nullptr, &b.memory);
    vkBindBufferMemory(g_device, b.buffer, b.memory, 0);
    vkMapMemory(g_device, b.memory, 0, VK_WHOLE_SIZE, 0, &b.mapped);
    return b;
}

struct Image
{
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED;
    VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT;
    uint32_t w = 0, h = 0;
};

Image MakeImage(uint32_t w, uint32_t h, VkFormat format, VkImageUsageFlags usage, VkImageAspectFlags aspect)
{
    Image im;
    im.format = format;
    im.aspect = aspect;
    im.w = w;
    im.h = h;
    VkImageCreateInfo ci { VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
    ci.imageType = VK_IMAGE_TYPE_2D;
    ci.format = format;
    ci.extent = { w, h, 1 };
    ci.mipLevels = 1;
    ci.arrayLayers = 1;
    ci.samples = VK_SAMPLE_COUNT_1_BIT;
    ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    ci.usage = usage;
    ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    Check(vkCreateImage(g_device, &ci, nullptr, &im.image), "vkCreateImage");
    VkMemoryRequirements req {};
    vkGetImageMemoryRequirements(g_device, im.image, &req);
    VkMemoryAllocateInfo ai { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = MemoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    Check(vkAllocateMemory(g_device, &ai, nullptr, &im.memory), "vkAllocateMemory (image)");
    vkBindImageMemory(g_device, im.image, im.memory, 0);
    VkImageViewCreateInfo vi { VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
    vi.image = im.image;
    vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vi.format = format;
    vi.subresourceRange = { aspect, 0, 1, 0, 1 };
    Check(vkCreateImageView(g_device, &vi, nullptr, &im.view), "vkCreateImageView");
    return im;
}

void Transition(const Image& im, VkImageLayout from, VkImageLayout to, VkAccessFlags srcAccess, VkAccessFlags dstAccess)
{
    VkImageMemoryBarrier b { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
    b.srcAccessMask = srcAccess;
    b.dstAccessMask = dstAccess;
    b.oldLayout = from;
    b.newLayout = to;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = im.image;
    b.subresourceRange = { im.aspect, 0, 1, 0, 1 };
    vkCmdPipelineBarrier(g_cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &b);
}

// 4 bytes a texel (depth float, motion half2), into SHADER_READ_ONLY_OPTIMAL as the game hands DLSS its inputs.
void Upload(const Image& im, const void* data)
{
    const VkDeviceSize bytes = (VkDeviceSize) im.w * im.h * 4;
    Buffer staging = HostBuffer(bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    std::memcpy(staging.mapped, data, (size_t) bytes);
    Begin();
    Transition(im, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, VK_ACCESS_TRANSFER_WRITE_BIT);
    VkBufferImageCopy region {};
    region.imageSubresource = { im.aspect, 0, 0, 1 };
    region.imageExtent = { im.w, im.h, 1 };
    vkCmdCopyBufferToImage(g_cmd, staging.buffer, im.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    Transition(im, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
               VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
    Submit();
    vkDestroyBuffer(g_device, staging.buffer, nullptr);
    vkFreeMemory(g_device, staging.memory, nullptr);
}

// ---- the fit ----
Fitter g_fitter;
uint32_t g_slot = 0;
uint32_t g_stamp = 0;
uint32_t g_keepStale = 0;
uint32_t g_linear = 0;
float g_mvScale[2] = { -(float) kSize, -(float) kSize };
Image* g_depthImg = nullptr;
Image* g_motionImg = nullptr;
Image g_composed, g_mask;
const Map* g_fitMap = &kTrue;
std::vector<float> g_depthData;
std::vector<uint16_t> g_lastMotion;
bool g_depthLinear = false;

void WriteFits(const float* fits80)
{
    Begin();
    g_fitter.Prepare(g_cmd);
    vkCmdUpdateBuffer(g_cmd, g_fitter.FitsBuffer(), 0, 320, fits80);
    VkMemoryBarrier b { VK_STRUCTURE_TYPE_MEMORY_BARRIER };
    b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(g_cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &b, 0,
                         nullptr, 0, nullptr);
    Submit();
}

struct Present
{
    int reason = -1;
    float streak = -1, residual = 99, inliers = 0, mapError = 99, gaveUp = -1, restarted = -1, leftOut = -1;
    float f[60] = {};
};

// One fit (and with `compose`, the compose) for parity `cur`, as the DLL records it; the fit's readback read.
Present RunFit(uint32_t cur = 0, bool compose = false)
{
    Constants c {};
    c.mvScale[0] = g_mvScale[0];
    c.mvScale[1] = g_mvScale[1];
    c.cur = cur;
    c.prev = cur ^ 1u;
    c.knee0 = 0.02f;
    c.knee1 = 0.1f;
    c.gain = 2.0f;
    c.radius = 2;
    c.stamp = ++g_stamp;
    c.keepStale = g_keepStale;
    c.linearDepth = g_linear;
    Inputs in;
    in.depth = g_depthImg->view;
    in.motion = g_motionImg->view;
    if (compose)
    {
        in.composedImage = g_composed.image;
        in.composed = g_composed.view;
        in.maskImage = g_mask.image;
        in.mask = g_mask.view;
        in.width = g_motionImg->w;
        in.height = g_motionImg->h;
    }
    const uint32_t slot = g_slot++ % SameEyeFit_Vk::kRing;
    Begin();
    g_fitter.Record(g_cmd, slot, in, c);
    Present p;
    if (!Submit())
        return p;
    std::memcpy(p.f, g_fitter.FitSlot(slot), sizeof(p.f));
    const float* m = p.f + 2 * 20;
    p.reason = (int) m[16];
    p.gaveUp = m[17];
    p.restarted = m[18] > 0.5f ? 1.0f : 0.0f;
    p.leftOut = (m[18] - p.restarted) / 0.45f;
    p.streak = p.f[cur * 20 + 17];
    p.residual = m[12];
    p.inliers = m[13];
    if (m[19] != (float) (c.stamp & 0xFFFFFFu))
        std::printf("   the fit readback carries stamp %.0f, expected %u\n", m[19], c.stamp & 0xFFFFFFu);
    const float zs = g_depthLinear ? kLinearK / 10.0f : 1.0f;
    const Map& t = *g_fitMap;
    const float want[11] = { t.r0[0], t.r0[1], t.r0[2] * zs, t.r0[3], t.r1[0], t.r1[1],
                             t.r1[2] * zs, t.r1[3], t.r2[0], t.r2[1], t.r2[2] * zs };
    float worst = 0;
    for (int i = 0; i < 11; ++i)
        worst = (std::max)(worst, std::fabs(p.f[cur * 20 + i] - want[i]));
    p.mapError = worst;
    return p;
}

int g_failures = 0, g_checked = 0;
void Report(const char* what, bool ok, const Present& p)
{
    std::printf("%s: reason %d (0 = accepted), gave up %.0f, restarted %.0f, %.0f%% left out at a depth gap, rejections "
                "in a row %.0f, residual %.3f px, within 1 px %.0f%%, the eye's map off the true one by %.5f -> %s\n",
                what, p.reason, p.gaveUp, p.restarted, p.leftOut * 100.0f, p.streak, p.residual, p.inliers * 100.0f,
                p.mapError, ok ? "OK" : "FAIL");
    ++g_checked;
    if (!ok)
        ++g_failures;
}
bool found(const Present& p) { return p.reason == 0 && p.residual < 0.2f && p.mapError < 0.002f; }

void WritePicture(const Map& m)
{
    std::vector<uint16_t> motion(kSize * kSize * 2);
    for (uint32_t py = 0; py < kSize; ++py)
        for (uint32_t px = 0; px < kSize; ++px)
        {
            const float u = (px + 0.5f) / kSize, v = (py + 0.5f) / kSize;
            const float x = u * 2.0f - 1.0f, y = 1.0f - v * 2.0f;
            const float zz = g_depthData[py * kSize + px] * kKZ;
            const float w = m.r2[0] * x + m.r2[1] * y + m.r2[2] * zz + 1.0f;
            const float xp = (m.r0[0] * x + m.r0[1] * y + m.r0[2] * zz + m.r0[3]) / w;
            const float yp = (m.r1[0] * x + m.r1[1] * y + m.r1[2] * zz + m.r1[3]) / w;
            float mx = (xp - x) * kSize * 0.5f + (Random() - 0.5f) * 0.1f;
            float my = -(yp - y) * kSize * 0.5f + (Random() - 0.5f) * 0.1f;
            if (px >= kSize / 4 && px < kSize * 3 / 4 && py >= kSize / 4 && py < kSize * 3 / 4)
            {
                mx += 60.0f;
                my -= 25.0f;
            }
            motion[(py * kSize + px) * 2 + 0] = Half(mx / -(float) kSize);
            motion[(py * kSize + px) * 2 + 1] = Half(my / -(float) kSize);
        }
    Upload(*g_motionImg, motion.data());
    g_lastMotion = motion;
}

void WriteDepth(bool linear)
{
    std::vector<float> d(g_depthData);
    if (linear)
        for (auto& v : d)
            v = kLinearK / v;
    Upload(*g_depthImg, d.data());
    g_depthLinear = linear;
}

void ApplyTrue(const Map& m, float x, float y, float z, float& ox, float& oy, float& oz)
{
    const float zz = z * kKZ;
    const float w = m.r2[0] * x + m.r2[1] * y + m.r2[2] * zz + 1.0f;
    ox = (m.r0[0] * x + m.r0[1] * y + m.r0[2] * zz + m.r0[3]) / w;
    oy = (m.r1[0] * x + m.r1[1] * y + m.r1[2] * zz + m.r1[3]) / w;
    oz = z / w;
}

// Case 8: the other eye's fit into slot 1 (kOther), this eye's into slot 0 (kTrue), then the compose; the compose's
// output read back from our image and checked against the true chained maps on static pixels.
float ChainError(bool& fitsOk, int& checked)
{
    float zero[80] = {};
    WriteFits(zero);
    WritePicture(kOther);
    Present p1 = RunFit(1);
    WritePicture(kTrue);
    Present p0 = RunFit(0, true);
    fitsOk = p1.f[1 * 20 + 14] > 0.5f && p0.f[0 * 20 + 14] > 0.5f;
    Buffer read = HostBuffer((VkDeviceSize) kSize * kSize * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    // The recorder leaves the composed motion in SHADER_READ_ONLY_OPTIMAL (DLSS's inputs' layout): read it as a copy
    // would after DLSS, and hand it back in that layout (the next compose expects it so).
    Begin();
    Transition(g_composed, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
               VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_TRANSFER_READ_BIT);
    VkBufferImageCopy region {};
    region.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    region.imageExtent = { kSize, kSize, 1 };
    vkCmdCopyImageToBuffer(g_cmd, g_composed.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, read.buffer, 1, &region);
    Transition(g_composed, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
               VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_SHADER_READ_BIT);
    Submit();
    const uint16_t* h = (const uint16_t*) read.mapped;
    float worst = 0;
    checked = 0;
    for (uint32_t py = 8; py < kSize - 8; py += 7)
        for (uint32_t px = 8; px < kSize - 8; px += 7)
        {
            if (px >= kSize / 4 - 8 && px < kSize * 3 / 4 + 8 && py >= kSize / 4 - 8 && py < kSize * 3 / 4 + 8)
                continue;
            const float x = (px + 0.5f) / kSize * 2.0f - 1.0f, y = 1.0f - (py + 0.5f) / kSize * 2.0f;
            float x1, y1, z1, x2, y2, z2;
            ApplyTrue(kTrue, x, y, g_depthData[py * kSize + px], x1, y1, z1);
            ApplyTrue(kOther, x1, y1, z1, x2, y2, z2);
            const float ex = (x2 - x) * kSize * 0.5f, ey = -(y2 - y) * kSize * 0.5f;
            const float ox = FromHalf(h[(py * kSize + px) * 2]) * -(float) kSize;
            const float oy = FromHalf(h[(py * kSize + px) * 2 + 1]) * -(float) kSize;
            worst = (std::max)(worst, std::hypot(ox - ex, oy - ey));
            ++checked;
        }
    vkDestroyBuffer(g_device, read.buffer, nullptr);
    vkFreeMemory(g_device, read.memory, nullptr);
    return worst;
}

bool Env(const char* name)
{
    char value[8] = {};
    return GetEnvironmentVariableA(name, value, sizeof(value)) > 0 && value[0] == '1';
}
} // namespace

int main()
{
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
    const bool nearKept = Env("VK_FIT_TEST_NEAR_KEPT");
    const bool noDump = Env("VK_FIT_TEST_NO_DUMP");
    const bool oldGap = Env("VK_FIT_TEST_OLD_GAP"); // the depth gap by the older valley rule
    g_keepStale = (nearKept ? 4u : 0u) | (oldGap ? 16u : 0u);
    if (!Setup())
        return 1;
    std::string error;
    if (!g_fitter.Init(g_device, g_physical, error))
    {
        std::printf("FAILED: SameEyeFit_Vk::Init: %s\n", error.c_str());
        return 1;
    }
    std::printf("the Vulkan fit (SameEyeFit_Vk, SPIR-V from the D3D12 source) on %s%s\n", g_gpuName.c_str(),
                nearKept ? " -- CONTROL: nothing left out at a depth gap" : noDump ? " -- CONTROL: no dump bit" : "");

    Image depth = MakeImage(kSize, kSize, VK_FORMAT_D32_SFLOAT,
                            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_IMAGE_ASPECT_DEPTH_BIT);
    Image motion = MakeImage(kSize, kSize, VK_FORMAT_R16G16_SFLOAT,
                             VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
    g_composed = MakeImage(kSize, kSize, VK_FORMAT_R16G16_SFLOAT,
                           VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                           VK_IMAGE_ASPECT_COLOR_BIT);
    g_mask = MakeImage(kSize, kSize, VK_FORMAT_R8_UNORM, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                       VK_IMAGE_ASPECT_COLOR_BIT);
    g_depthImg = &depth;
    g_motionImg = &motion;

    g_depthData.resize(kSize * kSize);
    for (uint32_t py = 0; py < kSize; ++py)
        for (uint32_t px = 0; px < kSize; ++px)
            g_depthData[py * kSize + px] =
                0.002f + 0.05f * (0.5f + 0.5f * std::sin(px * 0.013f) * std::cos(py * 0.017f)) + 0.058f * Random();
    WriteDepth(false);
    WritePicture(kTrue);

    float fits[80] = {};
    WriteFits(fits);
    Present p = RunFit();
    Report("1 no start", found(p) && p.streak == 0.0f, p);

    std::memset(fits, 0, sizeof(fits));
    std::memcpy(fits + 0, kTrue.r0, sizeof(kTrue.r0));
    std::memcpy(fits + 4, kTrue.r1, sizeof(kTrue.r1));
    std::memcpy(fits + 8, kTrue.r2, sizeof(kTrue.r2));
    fits[3] += 5.0f * 2.0f / kSize;
    fits[11] = 1.0f;
    fits[12] = 0.05f;
    fits[13] = 0.9f;
    fits[14] = 1.0f;
    fits[15] = 4096.0f;
    WriteFits(fits);
    p = RunFit();
    Report("2 a start 5 px off (no restart)", found(p) && p.streak == 0.0f && p.restarted == 0.0f, p);

    bool fitsOk = false;
    int checkedPx = 0;
    const float err = ChainError(fitsOk, checkedPx);
    std::printf("8 the chain + COMPOSE: both fits %s; same-eye motion off the true chained maps by %.3f px at the most "
                "over %d static pixels -> %s\n",
                fitsOk ? "accepted" : "NOT accepted", err, checkedPx, fitsOk && err < 0.2f ? "OK" : "FAIL");
    ++g_checked;
    if (!(fitsOk && err < 0.2f))
        ++g_failures;

    // 13: the dump, on the plain picture.
    WritePicture(kTrue);
    std::memset(fits, 0, sizeof(fits));
    WriteFits(fits);
    {
        const uint32_t keep = g_keepStale;
        if (!noDump)
            g_keepStale |= 8u;
        const uint32_t slot = g_slot % SameEyeFit_Vk::kRing;
        p = RunFit();
        g_keepStale = keep;
        const uint8_t* d = g_fitter.DumpSlot(slot);
        uint32_t stamp = 0;
        std::memcpy(&stamp, d + 4, 4);
        const float* s = (const float*) (d + 16);
        int wrong = 0;
        for (uint32_t si = 0; si < 4096; ++si)
        {
            const uint32_t tx = (uint32_t) (((si % 64) + 0.5f) / 64.0f * kSize), ty = (uint32_t) (((si / 64) + 0.5f) / 64.0f * kSize);
            const float wantDepth = g_depthData[ty * kSize + tx];
            const float wantX = FromHalf(g_lastMotion[(ty * kSize + tx) * 2]);
            const float wantY = FromHalf(g_lastMotion[(ty * kSize + tx) * 2 + 1]);
            if (std::memcmp(&s[si * 4], &wantDepth, 4) != 0 || std::memcmp(&s[si * 4 + 1], &wantX, 4) != 0 ||
                std::memcmp(&s[si * 4 + 2], &wantY, 4) != 0)
                ++wrong;
        }
        const bool ok13 = wrong == 0 && stamp == (g_stamp & 0xFFFFFFu);
        std::printf("13 the dump: %d of 4096 samples not as uploaded, stamp %u (fit %u) -> %s\n", wrong, stamp,
                    g_stamp & 0xFFFFFFu, ok13 ? "OK" : "FAIL");
        ++g_checked;
        if (!ok13)
            ++g_failures;
    }

    vkDeviceWaitIdle(g_device);
    g_fitter.Destroy();
    std::printf("%s: %d checks, %d failures\n", g_failures == 0 ? "PASSED" : "FAILED", g_checked, g_failures);
    return g_failures == 0 ? 0 : 1;
}
