// Offline test of the VULKAN foveal crop's passes (OptiScaler/inputs/FoveaPasses_Vk, for FoveaCrop_Vk) and of the
// box maths (FoveaBox.h, PerEyeDlss_Dx12's CropAxis): the very recorder the DLL uses, on a real Vulkan device, with
// the SPIR-V compiled from the D3D12 source (vk_fovea_shaders.py -> FoveaShader_Vk.h). RGBA16F images, as the game's
// output in a Vulkan game (the shaders' declared storage format).
//   1  COMPOSITE: a 32x32 crop of 3.0 over a 64x64 output of 1.0 at (16, 16), feather 4: outside the box untouched
//      (1.0), the centre 3.0, the first edge pixel and the pixel 3 in from the edge blended as the shader's smoothstep
//      says (within a half-precision step)
//   2  COMPOSITE with the outline: the box's first three pixels magenta (4, 0, 4, 1), the centre still the crop
//   3  ENLARGE: a 16x16 constant 2.0 over 64x64 -> 2.0 everywhere; a 16x16 horizontal ramp -> monotonic along x, the
//      middle near 0.5
//   4  the box: R = O = 1000 at 0.4 is EXACT (400 at 300, both ends); a game's eye 2314 -> 3472 at 0.4 fits, is even,
//      stays inside, and lands within 0.15 of 0.4
// Controls, each must FAIL:
//   VK_FOVEA_TEST_NO_FEATHER=1   the composite recorded with feather 0 while the feathered values are expected
//   VK_FOVEA_TEST_BOX_OFF=1      the composite recorded at (0, 0) while the box at (16, 16) is expected
#include <windows.h>

#include "../../../OptiScaler/inputs/FoveaBox.h"
#include "../../../OptiScaler/inputs/FoveaPasses_Vk.h"
#include "../../../OptiScaler/inputs/VkCreateWorker.h"
#include "../../../OptiScaler/inputs/FrameTime_Vk.h"

#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace
{
VkInstance g_instance = VK_NULL_HANDLE;
VkPhysicalDevice g_physical = VK_NULL_HANDLE;
VkDevice g_device = VK_NULL_HANDLE;
VkQueue g_queue = VK_NULL_HANDLE;
bool g_workerMade = false;
bool Env(const char* name);
VkCommandPool g_pool = VK_NULL_HANDLE;
VkCommandBuffer g_cmd = VK_NULL_HANDLE;
VkFence g_fence = VK_NULL_HANDLE;
int g_checked = 0, g_failures = 0;

bool Check(VkResult r, const char* what)
{
    if (r != VK_SUCCESS)
        std::printf("FAILED: %s (VkResult %d)\n", what, (int) r);
    return r == VK_SUCCESS;
}

void Expect(bool ok, const char* what)
{
    ++g_checked;
    if (!ok)
        ++g_failures;
    std::printf("  %s: %s\n", what, ok ? "OK" : "FAIL");
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
    app.pApplicationName = "vk_fovea_test";
    app.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo ii { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
    ii.pApplicationInfo = &app;
    if (!Check(vkCreateInstance(&ii, nullptr, &g_instance), "vkCreateInstance"))
        return false;
    uint32_t count = 0;
    vkEnumeratePhysicalDevices(g_instance, &count, nullptr);
    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(g_instance, &count, devices.data());
    std::string name;
    for (auto d : devices)
    {
        VkPhysicalDeviceProperties p {};
        vkGetPhysicalDeviceProperties(d, &p);
        if (g_physical == VK_NULL_HANDLE || p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
        {
            g_physical = d;
            name = p.deviceName;
        }
    }
    if (g_physical == VK_NULL_HANDLE)
    {
        std::printf("FAILED: no Vulkan device\n");
        return false;
    }
    std::printf("device: %s\n", name.c_str());
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
    // ---- 5: the create worker's queue, asked for as the game DLL's vkCreateDevice hook asks (control:
    // VK_FOVEA_TEST_NO_WORKER=1 asks for none, and the worker must then be unavailable and the job refused).
    std::vector<VkDeviceQueueCreateInfo> workerInfos;
    std::vector<float> workerPriorities;
    std::string why;
    const bool asked = !Env("VK_FOVEA_TEST_NO_WORKER") &&
                       VkCreateWorker::AskQueue(g_physical, di, workerInfos, workerPriorities, why);
    std::printf("5 the create worker's queue: %s (%s)\n", asked ? "asked" : "NOT asked", why.c_str());
    if (!Check(vkCreateDevice(g_physical, &di, nullptr, &g_device), "vkCreateDevice"))
        return false;
    vkGetDeviceQueue(g_device, family, 0, &g_queue);
    g_workerMade = asked && VkCreateWorker::DeviceMade(g_device, vkGetDeviceQueue, why);
    std::printf("  worker: %s (%s)\n", g_workerMade ? "ready" : "NOT available", why.c_str());
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

void FreeBuffer(Buffer& b)
{
    vkDestroyBuffer(g_device, b.buffer, nullptr);
    vkFreeMemory(g_device, b.memory, nullptr);
    b = {};
}

struct Image
{
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    uint32_t w = 0, h = 0;
};

Image MakeImage(uint32_t w, uint32_t h)
{
    Image im;
    im.w = w;
    im.h = h;
    VkImageCreateInfo ci { VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
    ci.imageType = VK_IMAGE_TYPE_2D;
    ci.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    ci.extent = { w, h, 1 };
    ci.mipLevels = 1;
    ci.arrayLayers = 1;
    ci.samples = VK_SAMPLE_COUNT_1_BIT;
    ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    ci.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
               VK_IMAGE_USAGE_TRANSFER_DST_BIT;
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
    vi.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    vi.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    Check(vkCreateImageView(g_device, &vi, nullptr, &im.view), "vkCreateImageView");
    return im;
}

void Transition(const Image& im, VkImageLayout from, VkImageLayout to)
{
    VkImageMemoryBarrier b { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
    b.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT;
    b.dstAccessMask = VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT;
    b.oldLayout = from;
    b.newLayout = to;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = im.image;
    b.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    vkCmdPipelineBarrier(g_cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &b);
}

uint16_t Half(float value)
{
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    const int exponent = (int) ((bits >> 23) & 0xff) - 127 + 15;
    const uint32_t mantissa = bits & 0x7fffffu;
    if (exponent <= 0)
        return (uint16_t) sign;
    if (exponent >= 31)
        return (uint16_t) (sign | 0x7c00u);
    return (uint16_t) (sign | ((uint32_t) exponent << 10) | (mantissa >> 13));
}

float Float(uint16_t h)
{
    const uint32_t sign = (h & 0x8000u) << 16;
    const uint32_t exponent = (h >> 10) & 0x1fu;
    const uint32_t mantissa = h & 0x3ffu;
    uint32_t bits;
    if (exponent == 0)
        bits = sign; // denormals read as 0 here (not expected)
    else if (exponent == 31)
        bits = sign | 0x7f800000u | (mantissa << 13);
    else
        bits = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

// Fills an image with per-texel RGBA (a lambda of x, y), through a staging buffer, and leaves it in GENERAL.
template <typename F> void Fill(const Image& im, F texel)
{
    const VkDeviceSize bytes = (VkDeviceSize) im.w * im.h * 8;
    Buffer staging = HostBuffer(bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    uint16_t* p = (uint16_t*) staging.mapped;
    for (uint32_t y = 0; y < im.h; ++y)
        for (uint32_t x = 0; x < im.w; ++x)
        {
            float rgba[4];
            texel(x, y, rgba);
            for (int c = 0; c < 4; ++c)
                *p++ = Half(rgba[c]);
        }
    Begin();
    Transition(im, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    VkBufferImageCopy region {};
    region.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    region.imageExtent = { im.w, im.h, 1 };
    vkCmdCopyBufferToImage(g_cmd, staging.buffer, im.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    Transition(im, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);
    Submit();
    FreeBuffer(staging);
}

// Reads an image (in GENERAL) back as floats, RGBA per texel.
std::vector<float> Read(const Image& im)
{
    const VkDeviceSize bytes = (VkDeviceSize) im.w * im.h * 8;
    Buffer staging = HostBuffer(bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    Begin();
    VkBufferImageCopy region {};
    region.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    region.imageExtent = { im.w, im.h, 1 };
    vkCmdCopyImageToBuffer(g_cmd, im.image, VK_IMAGE_LAYOUT_GENERAL, staging.buffer, 1, &region);
    Submit();
    std::vector<float> out((size_t) im.w * im.h * 4);
    const uint16_t* p = (const uint16_t*) staging.mapped;
    for (size_t i = 0; i < out.size(); ++i)
        out[i] = Float(p[i]);
    FreeBuffer(staging);
    return out;
}

float At(const std::vector<float>& px, uint32_t w, uint32_t x, uint32_t y, int c = 0) { return px[((size_t) y * w + x) * 4 + c]; }

bool Near(float a, float b, float tol) { return std::abs(a - b) <= tol; }

bool Env(const char* name)
{
    char buf[8] = {};
    return GetEnvironmentVariableA(name, buf, sizeof(buf)) > 0 && buf[0] == '1';
}

float SmoothStep(float e0, float e1, float x)
{
    const float t = std::fmin(std::fmax((x - e0) / (e1 - e0), 0.0f), 1.0f);
    return t * t * (3.0f - 2.0f * t);
}
} // namespace

int main()
{
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
    const bool noFeather = Env("VK_FOVEA_TEST_NO_FEATHER"), boxOff = Env("VK_FOVEA_TEST_BOX_OFF");
    if (!Setup())
        return 2;
    FoveaPasses_Vk::Passes passes;
    std::string error;
    if (!passes.Init(g_device, g_physical, error))
    {
        std::printf("FAILED: FoveaPasses_Vk::Init: %s\n", error.c_str());
        return 2;
    }

    // ---- 1 + 2: COMPOSITE ----
    const uint32_t W = 64, CW = 32, BX = 16, BY = 16, FEATHER = 4;
    Image output = MakeImage(W, W), crop = MakeImage(CW, CW);
    auto reset = [&]()
    {
        Fill(output, [](uint32_t, uint32_t, float* v) { v[0] = v[1] = v[2] = 1.0f; v[3] = 1.0f; });
        Fill(crop, [](uint32_t, uint32_t, float* v) { v[0] = v[1] = v[2] = 3.0f; v[3] = 1.0f; });
    };
    reset();
    FoveaPasses_Vk::CompositeConstants c {};
    c.baseX = boxOff ? 0 : BX;
    c.baseY = boxOff ? 0 : BY;
    c.sizeX = c.sizeY = CW;
    c.feather = noFeather ? 0 : FEATHER;
    c.outline = 0;
    Begin();
    passes.Composite(g_cmd, 0, crop.view, output.view, c);
    Submit();
    auto px = Read(output);
    std::printf("1 COMPOSITE (box (%u, %u) %ux%u over %ux%u, feather %u)\n", BX, BY, CW, CW, W, W, FEATHER);
    Expect(Near(At(px, W, 2, 2), 1.0f, 1e-3f) && Near(At(px, W, 60, 60), 1.0f, 1e-3f) &&
               Near(At(px, W, BX - 1, BY + 10), 1.0f, 1e-3f),
           "outside the box untouched (1.0)");
    Expect(Near(At(px, W, BX + CW / 2, BY + CW / 2), 3.0f, 1e-3f), "the box's centre is the crop (3.0)");
    const float edge0 = 1.0f + 2.0f * SmoothStep(0.0f, 1.0f, 0.5f / FEATHER); // d = 0
    const float edge3 = 1.0f + 2.0f * SmoothStep(0.0f, 1.0f, 3.5f / FEATHER); // d = 3
    Expect(Near(At(px, W, BX, BY + 10), edge0, 0.01f), "the first edge pixel blended (smoothstep at d = 0)");
    Expect(Near(At(px, W, BX + 3, BY + 10), edge3, 0.01f), "the pixel 3 in from the edge blended (d = 3)");
    Expect(Near(At(px, W, BX + FEATHER, BY + 10), 3.0f, 1e-3f), "past the feather the crop (3.0)");

    reset();
    c.outline = 1;
    Begin();
    passes.Composite(g_cmd, 1, crop.view, output.view, c);
    Submit();
    px = Read(output);
    std::printf("2 COMPOSITE with the outline\n");
    Expect(Near(At(px, W, BX, BY + 10, 0), 4.0f, 1e-3f) && Near(At(px, W, BX, BY + 10, 1), 0.0f, 1e-3f) &&
               Near(At(px, W, BX + 2, BY + 10, 2), 4.0f, 1e-3f),
           "the frame's three pixels magenta (4, 0, 4)");
    Expect(Near(At(px, W, BX + CW / 2, BY + CW / 2), 3.0f, 1e-3f), "the centre still the crop");

    // ---- 3: ENLARGE ----
    const uint32_t SW = 16;
    Image src = MakeImage(SW, SW);
    Fill(src, [](uint32_t, uint32_t, float* v) { v[0] = v[1] = v[2] = 2.0f; v[3] = 1.0f; });
    Fill(output, [](uint32_t, uint32_t, float* v) { v[0] = v[1] = v[2] = 0.0f; v[3] = 0.0f; });
    FoveaPasses_Vk::EnlargeConstants e {};
    e.outW = e.outH = W;
    e.srcW = e.srcH = SW;
    Begin();
    passes.Enlarge(g_cmd, 2, src.view, output.view, e);
    Submit();
    px = Read(output);
    std::printf("3 ENLARGE (%ux%u -> %ux%u)\n", SW, SW, W, W);
    bool allTwo = true;
    for (uint32_t y = 0; y < W && allTwo; ++y)
        for (uint32_t x = 0; x < W; ++x)
            if (!Near(At(px, W, x, y), 2.0f, 0.01f))
            {
                allTwo = false;
                break;
            }
    Expect(allTwo, "a constant source enlarges to the constant everywhere");
    Fill(src, [](uint32_t x, uint32_t, float* v) { v[0] = v[1] = v[2] = (float) x / 15.0f; v[3] = 1.0f; });
    Begin();
    passes.Enlarge(g_cmd, 3, src.view, output.view, e);
    Submit();
    px = Read(output);
    bool monotonic = true;
    for (uint32_t x = 1; x < W; ++x)
        if (At(px, W, x, 32) < At(px, W, x - 1, 32) - 0.01f)
            monotonic = false;
    Expect(monotonic, "a ramp stays monotonic along x");
    Expect(Near(At(px, W, 32, 32), 0.5f, 0.06f), "the ramp's middle is near 0.5");

    // ---- 4: the box ----
    std::printf("4 the box (FoveaBox::Plan)\n");
    FoveaBox::Rect r;
    Expect(FoveaBox::Plan(1000, 1000, 1000, 1000, 0.4f, 0.0f, true, r) && r.exact && r.rw == 400 && r.rx == 300 &&
               r.ow == 400 && r.ox == 300 && r.rh == 400 && r.ry == 300,
           "R = O = 1000 at 0.4: EXACT, 400 at 300");
    const bool vkGame = FoveaBox::Plan(2314, 2314, 3472, 3472, 0.4f, 0.0f, true, r);
    Expect(vkGame && (r.rw % 2) == 0 && r.rx + r.rw <= 2314 && r.ox + r.ow <= 3472 &&
               std::abs((float) r.rw / 2314.0f - 0.4f) <= 0.15f && r.ow > 0 && r.oh > 0,
           "a game's eye 2314 -> 3472 at 0.4: a box that fits");
    std::printf("  the game's box: render (%u, %u) %ux%u -> output (%u, %u) %ux%u, %s\n", r.rx, r.ry, r.rw, r.rh, r.ox,
                r.oy, r.ow, r.oh, r.exact ? "exact" : "approximate");

    // ---- 6: the gather copy (FoveaPasses_Vk::CopyBox): the box (5, 7) 8x6 of a 32x32 image whose texel is (x, y)
    // lands at (0, 0) of an 8x6 image; outside the box nothing of the source is read (the destination's own content
    // beyond the box is untouched: an 8x8 destination keeps its last two rows). Control: VK_FOVEA_TEST_COPY_OFF=1
    // skips the copy and the checks must fail.
    std::printf("6 the gather copy (CopyBox)\n");
    {
        Image src = MakeImage(32, 32), dst = MakeImage(8, 8);
        Fill(src, [](uint32_t x, uint32_t y, float* v) { v[0] = (float) x; v[1] = (float) y; v[2] = 0.0f; v[3] = 1.0f; });
        Fill(dst, [](uint32_t, uint32_t, float* v) { v[0] = v[1] = -1.0f; v[2] = 0.0f; v[3] = 1.0f; });
        Begin();
        if (!Env("VK_FOVEA_TEST_COPY_OFF"))
            FoveaPasses_Vk::CopyBox(g_cmd, src.image, VK_IMAGE_ASPECT_COLOR_BIT, 5, 7, dst.image, 8, 6);
        Submit();
        const auto px = Read(dst);
        Expect(Near(At(px, 8, 0, 0, 0), 5.0f, 0.01f) && Near(At(px, 8, 0, 0, 1), 7.0f, 0.01f),
               "the box's top left (5, 7) lands at (0, 0)");
        Expect(Near(At(px, 8, 7, 5, 0), 12.0f, 0.01f) && Near(At(px, 8, 7, 5, 1), 12.0f, 0.01f),
               "the box's bottom right (12, 12) lands at (7, 5)");
        Expect(Near(At(px, 8, 3, 7, 0), -1.0f, 0.01f), "beyond the box the destination is untouched");
    }

    // ---- 7: the TAA pass (FoveaPasses_Vk::Taa, the D3D12 shader as SPIR-V): a 16x16 render of a vertical edge
    // (dark left of x = 8, bright right) at zero jitter and zero motion, 32x32 out. With Reset = 1 the output is the
    // reconstruction alone: flat dark well left of the edge, flat bright well right, the edge within two output pixels
    // of x = 16, and the history written equals the tonemapped output. With Reset = 0, the history the flat grey of
    // 0.5 and blend 0.1: the result is clamped to the current 3x3's range, so a flat region stays its own colour.
    // Control: VK_FOVEA_TEST_TAA_OFF=1 skips the pass and the checks must fail.
    std::printf("7 the TAA pass\n");
    {
        const uint32_t RW = 16, OW = 32;
        Image colour = MakeImage(RW, RW), motion = MakeImage(RW, RW), depth = MakeImage(RW, RW),
              history = MakeImage(OW, OW), output = MakeImage(OW, OW), historyOut = MakeImage(OW, OW);
        Fill(colour, [](uint32_t x, uint32_t, float* v) { v[0] = v[1] = v[2] = x < 8 ? 0.1f : 0.9f; v[3] = 1.0f; });
        Fill(motion, [](uint32_t, uint32_t, float* v) { v[0] = v[1] = v[2] = v[3] = 0.0f; });
        Fill(depth, [](uint32_t, uint32_t, float* v) { v[0] = 0.5f; v[1] = v[2] = v[3] = 0.0f; });
        Fill(history, [](uint32_t, uint32_t, float* v) { v[0] = v[1] = v[2] = 0.5f; v[3] = 1.0f; });
        Fill(output, [](uint32_t, uint32_t, float* v) { v[0] = v[1] = v[2] = -1.0f; v[3] = 1.0f; });
        Fill(historyOut, [](uint32_t, uint32_t, float* v) { v[0] = v[1] = v[2] = -1.0f; v[3] = 1.0f; });
        FoveaPasses_Vk::TaaConstants t {};
        t.outSize[0] = t.outSize[1] = OW;
        t.renderSize[0] = t.renderSize[1] = RW;
        t.mvScale[0] = t.mvScale[1] = 1.0f;
        t.blend = 0.1f;
        t.reset = 1;
        const bool taaOff = Env("VK_FOVEA_TEST_TAA_OFF");
        Begin();
        if (!taaOff)
            passes.Taa(g_cmd, 4, colour.view, motion.view, depth.view, history.view, output.view, historyOut.view, t);
        Submit();
        auto px = Read(output);
        const auto hx = Read(historyOut);
        Expect(Near(At(px, OW, 4, 10, 0), 0.1f, 0.02f), "reset: flat dark left of the edge (0.1)");
        Expect(Near(At(px, OW, 27, 10, 0), 0.9f, 0.02f), "reset: flat bright right of the edge (0.9)");
        int edge = -1;
        for (uint32_t x = 1; x < OW && edge < 0; ++x)
            if (At(px, OW, x, 10, 0) > 0.5f && At(px, OW, x - 1, 10, 0) <= 0.5f)
                edge = (int) x;
        Expect(edge >= 14 && edge <= 18, "reset: the edge lands within two output pixels of x = 16");
        const float tm = 0.9f / (1.0f + 0.9f);
        Expect(Near(At(hx, OW, 27, 10, 0), tm, 0.02f), "reset: the history written is the tonemapped output");
        t.reset = 0;
        Begin();
        if (!taaOff)
            passes.Taa(g_cmd, 5, colour.view, motion.view, depth.view, history.view, output.view, historyOut.view, t);
        Submit();
        px = Read(output);
        Expect(Near(At(px, OW, 4, 10, 0), 0.1f, 0.03f) && Near(At(px, OW, 27, 10, 0), 0.9f, 0.03f),
               "with a grey history and blend 0.1: a flat region keeps its own colour (the clamp)");
    }

    // ---- 8: the GPU frame timer (FrameTime_Vk): a slot with an ENLARGE between stamps 1 and 2 (the "game's pass"),
    // then ten empty slots so the first is old enough to read; the window must hold at least nine frames of a
    // positive length under 200 ms, one game run of a positive length, and our passes measured. Control:
    // VK_FOVEA_TEST_TIME_OFF=1 never begins a slot and the window stays empty.
    std::printf("8 the GPU frame timer\n");
    {
        std::string why;
        const bool timeOff = Env("VK_FOVEA_TEST_TIME_OFF");
        Expect(FrameTime_Vk::Init(g_device, g_physical, why), ("the timer's pool made (" + why + ")").c_str());
        Image src = MakeImage(16, 16), dst = MakeImage(64, 64);
        Fill(src, [](uint32_t, uint32_t, float* v) { v[0] = v[1] = v[2] = 0.5f; v[3] = 1.0f; });
        for (int i = 0; i < 12; ++i)
        {
            Begin();
            if (!timeOff)
            {
                FrameTime_Vk::Begin(g_cmd);
                FrameTime_Vk::Mark(g_cmd, 1, true);
                FoveaPasses_Vk::EnlargeConstants e {};
                e.outW = e.outH = 64;
                e.srcW = e.srcH = 16;
                passes.Enlarge(g_cmd, (uint32_t) (6 + i), src.view, dst.view, e);
                FrameTime_Vk::Mark(g_cmd, 2, true);
                FrameTime_Vk::Mark(g_cmd, 3, true);
            }
            Submit();
        }
        FrameTime_Vk::Collect(2);
        const FrameTime_Vk::Window w = FrameTime_Vk::Take();
        std::printf("  window: %u frames %.4f ms, game %u runs %.4f ms, ours %u (%.4f + %.4f ms)\n", w.frames,
                    w.frameMs, w.gameRuns, w.gameMs, w.ours, w.beforeMs, w.afterMs);
        Expect(w.frames >= 9 && w.frameMs > 0.0 && w.frameMs < 200.0, "frames measured, a positive length under 200 ms");
        Expect(w.gameRuns >= 9 && w.gameMs > 0.0, "the game's pass (the enlarge here) measured, a positive length");
        Expect(w.ours >= 9, "our passes measured on every slot");
        // THE GAME'S CASE (a Vulkan game: 45 windows without one reading): far more evaluates than the ring holds
        // between two reads. The read must go on from the oldest slot left and say how many were lost. Control:
        // VK_FOVEA_TEST_TIME_NO_SKIP=1 is the old read, which stops for good at the first overrun.
        if (!timeOff)
        {
            for (int i = 0; i < 200; ++i)
            {
                Begin();
                FrameTime_Vk::Begin(g_cmd);
                FrameTime_Vk::Mark(g_cmd, 1, true);
                FrameTime_Vk::Mark(g_cmd, 2, true);
                FrameTime_Vk::Mark(g_cmd, 3, true);
                Submit();
            }
            FrameTime_Vk::Collect(2, !Env("VK_FOVEA_TEST_TIME_NO_SKIP"));
            const FrameTime_Vk::Window w2 = FrameTime_Vk::Take();
            std::printf("  after 200 evaluates and one read: %u frames %.4f ms; %llu stamped, %llu lost, %llu unfinished\n",
                        w2.frames, w2.frameMs, (unsigned long long) w2.begun, (unsigned long long) w2.lost,
                        (unsigned long long) w2.unfinished);
            Expect(w2.frames >= 50, "a ring overrun: the read goes on from the oldest slot left (50 frames or more)");
            Expect(w2.begun == 200 && w2.lost >= 100 && w2.lost <= 150, "the slots lost to the ring are counted");
        }
        FrameTime_Vk::Shutdown();
    }

    // ---- 5: the create worker: a job records a clear of an image of ours on the worker's command buffer and
    // queue; the game's queue (ours here) reads it back after; a failing job (record returns false) is reported as
    // not ok and nothing of it submitted; jobs come back in order.
    std::printf("5 the create worker\n");
    Expect(g_workerMade, "the worker is available (a queue of its own, the thread started)");
    {
        Image im = MakeImage(16, 16);
        Fill(im, [](uint32_t, uint32_t, float* v) { v[0] = v[1] = v[2] = 0.0f; v[3] = 1.0f; });
        std::mutex m;
        std::condition_variable cv;
        std::vector<std::pair<uint64_t, bool>> results;
        auto done = [&](uint64_t ticket, bool ok)
        {
            std::lock_guard lock(m);
            results.push_back({ ticket, ok });
            cv.notify_all();
        };
        VkCreateWorker::Job clear;
        clear.ticket = 1;
        clear.record = [&](VkCommandBuffer cmd)
        {
            VkClearColorValue colour {};
            colour.float32[0] = 0.25f;
            colour.float32[1] = 0.5f;
            colour.float32[2] = 0.75f;
            colour.float32[3] = 1.0f;
            VkImageSubresourceRange range { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
            vkCmdClearColorImage(cmd, im.image, VK_IMAGE_LAYOUT_GENERAL, &colour, 1, &range);
            return true;
        };
        clear.done = done;
        VkCreateWorker::Job failing;
        failing.ticket = 2;
        failing.record = [](VkCommandBuffer) { return false; };
        failing.done = done;
        VkCreateWorker::Submit(std::move(clear));
        VkCreateWorker::Submit(std::move(failing));
        {
            std::unique_lock lock(m);
            cv.wait_for(lock, std::chrono::seconds(10), [&] { return results.size() >= 2; });
        }
        Expect(results.size() == 2, "both jobs reported");
        Expect(results.size() == 2 && results[0].first == 1 && results[0].second, "the clear job ran and completed OK");
        Expect(results.size() == 2 && results[1].first == 2 && !results[1].second,
               "the failing job came second and was reported NOT ok");
        const auto px = Read(im);
        Expect(Near(At(px, 16, 3, 5, 0), 0.25f, 0.01f) && Near(At(px, 16, 3, 5, 1), 0.5f, 0.01f) &&
                   Near(At(px, 16, 12, 9, 2), 0.75f, 0.01f),
               "the worker's clear is seen from the game's queue (0.25, 0.5, 0.75)");
        Expect(VkCreateWorker::Pending() == 0, "nothing pending after");
        VkCreateWorker::Shutdown();
        Expect(!VkCreateWorker::Available(), "shut down");
    }

    passes.Destroy();
    std::printf("%s: %d checks, %d failures\n", g_failures == 0 ? "PASSED" : "FAILED", g_checked, g_failures);
    return g_failures == 0 ? 0 : 1;
}
