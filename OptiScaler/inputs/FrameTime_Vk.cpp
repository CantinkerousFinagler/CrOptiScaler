#include "FrameTime_Vk.h"

#include <mutex>

namespace FrameTime_Vk
{
namespace
{
std::mutex g_mutex;
VkDevice g_device = VK_NULL_HANDLE;
VkQueryPool g_pool = VK_NULL_HANDLE;
double g_nsPerTick = 0.0;
bool g_ready = false;
uint64_t g_seq = 0;          // slots begun
uint64_t g_read = 0;         // slots read back (all below this)
struct Slot
{
    uint64_t seq = 0;
    bool began = false, ran[kStamps] = {};
    uint64_t t[kStamps] = {};
    bool done = false;
};
Slot g_slots[kSlots];
uint64_t g_prevT0 = 0, g_prevSeq = UINT64_MAX;
// Accumulated since the last Take.
uint32_t g_frames = 0, g_gameRuns = 0, g_ours = 0;
double g_frameNs = 0.0, g_gameNs = 0.0, g_beforeNs = 0.0, g_afterNs = 0.0;
uint64_t g_begun = 0, g_lost = 0, g_unfinished = 0;
} // namespace

bool Init(VkDevice device, VkPhysicalDevice physical, std::string& why)
{
    std::lock_guard lock(g_mutex);
    if (g_ready)
    {
        why = "already";
        return true;
    }
    if (device == VK_NULL_HANDLE || physical == VK_NULL_HANDLE)
    {
        why = "no device";
        return false;
    }
    VkPhysicalDeviceProperties props {};
    vkGetPhysicalDeviceProperties(physical, &props);
    if (props.limits.timestampPeriod <= 0.0f)
    {
        why = "the device has no timestamps";
        return false;
    }
    VkQueryPoolCreateInfo qi { VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO };
    qi.queryType = VK_QUERY_TYPE_TIMESTAMP;
    qi.queryCount = kSlots * kStamps;
    if (vkCreateQueryPool(device, &qi, nullptr, &g_pool) != VK_SUCCESS)
    {
        why = "vkCreateQueryPool failed";
        return false;
    }
    g_device = device;
    g_nsPerTick = props.limits.timestampPeriod;
    g_ready = true;
    why = "a pool of " + std::to_string(kSlots) + " slots, " + std::to_string(g_nsPerTick) + " ns per tick";
    return true;
}

bool Ready()
{
    std::lock_guard lock(g_mutex);
    return g_ready;
}

uint64_t Begin(VkCommandBuffer cmd)
{
    std::lock_guard lock(g_mutex);
    if (!g_ready || cmd == VK_NULL_HANDLE)
        return 0;
    const uint64_t seq = g_seq++;
    ++g_begun;
    Slot& s = g_slots[seq % kSlots];
    s = {};
    s.seq = seq;
    s.began = true;
    const uint32_t first = (uint32_t) (seq % kSlots) * kStamps;
    vkCmdResetQueryPool(cmd, g_pool, first, kStamps);
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, g_pool, first);
    s.ran[0] = true;
    return seq;
}

void Mark(VkCommandBuffer cmd, uint32_t stage, bool ran)
{
    std::lock_guard lock(g_mutex);
    if (!g_ready || cmd == VK_NULL_HANDLE || g_seq == 0 || stage == 0 || stage >= kStamps)
        return;
    const uint64_t seq = g_seq - 1;
    Slot& s = g_slots[seq % kSlots];
    if (!s.began)
        return;
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, g_pool, (uint32_t) (seq % kSlots) * kStamps + stage);
    s.ran[stage] = ran;
}

void Collect(uint64_t age, bool skipOverrun)
{
    std::lock_guard lock(g_mutex);
    if (!g_ready)
        return;
    if (g_seq - g_read > kSlots)
    {
        // More than a ring since the last read: the oldest slots are overwritten. Go on from the oldest one left.
        if (!skipOverrun)
            return;
        g_lost += g_seq - kSlots - g_read;
        g_read = g_seq - kSlots;
        g_prevSeq = UINT64_MAX;
    }
    while (g_read + age < g_seq)
    {
        Slot& s = g_slots[g_read % kSlots];
        if (!s.began || s.seq != g_read)
        {
            ++g_read;
            continue;
        }
        uint64_t data[kStamps * 2] = {};
        const VkResult r = vkGetQueryPoolResults(g_device, g_pool, (uint32_t) (g_read % kSlots) * kStamps, kStamps,
                                                 sizeof(data), data, 2 * sizeof(uint64_t),
                                                 VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
        if (r != VK_SUCCESS && r != VK_NOT_READY)
        {
            ++g_read;
            continue;
        }
        bool all = true;
        for (uint32_t i = 0; i < kStamps; ++i)
            all = all && data[i * 2 + 1] != 0;
        if (!all)
        {
            // Not finished yet (later slots are not either) -- unless it is old: a stamp never written (the evaluate
            // returned early) would block the ring for good.
            if (g_seq - g_read > kSlots / 2)
            {
                ++g_unfinished;
                ++g_read;
                continue;
            }
            break;
        }
        for (uint32_t i = 0; i < kStamps; ++i)
            s.t[i] = data[i * 2];
        s.done = true;
        // The frame: this slot's stamp 0 against the previous consecutive slot's.
        if (g_prevSeq != UINT64_MAX && s.seq == g_prevSeq + 1 && s.t[0] > g_prevT0)
        {
            const double ns = (double) (s.t[0] - g_prevT0) * g_nsPerTick;
            if (ns < 200.0e6)
            {
                g_frameNs += ns;
                ++g_frames;
            }
        }
        g_prevT0 = s.t[0];
        g_prevSeq = s.seq;
        if (s.ran[1] && s.ran[2] && s.t[2] >= s.t[1])
        {
            g_gameNs += (double) (s.t[2] - s.t[1]) * g_nsPerTick;
            ++g_gameRuns;
        }
        if (s.ran[3] && s.t[1] >= s.t[0] && s.t[3] >= s.t[2])
        {
            g_beforeNs += (double) (s.t[1] - s.t[0]) * g_nsPerTick;
            g_afterNs += (double) (s.t[3] - s.t[2]) * g_nsPerTick;
            ++g_ours;
        }
        ++g_read;
    }
}

Window Take()
{
    std::lock_guard lock(g_mutex);
    Window w;
    w.frames = g_frames;
    w.frameMs = g_frames ? g_frameNs / g_frames / 1.0e6 : 0.0;
    w.gameRuns = g_gameRuns;
    w.gameMs = g_gameRuns ? g_gameNs / g_gameRuns / 1.0e6 : 0.0;
    w.ours = g_ours;
    w.beforeMs = g_ours ? g_beforeNs / g_ours / 1.0e6 : 0.0;
    w.afterMs = g_ours ? g_afterNs / g_ours / 1.0e6 : 0.0;
    w.begun = g_begun;
    w.lost = g_lost;
    w.unfinished = g_unfinished;
    g_begun = g_lost = g_unfinished = 0;
    g_frames = g_gameRuns = g_ours = 0;
    g_frameNs = g_gameNs = g_beforeNs = g_afterNs = 0.0;
    return w;
}

void Shutdown()
{
    std::lock_guard lock(g_mutex);
    if (g_pool != VK_NULL_HANDLE && g_device != VK_NULL_HANDLE)
        vkDestroyQueryPool(g_device, g_pool, nullptr);
    g_pool = VK_NULL_HANDLE;
    g_device = VK_NULL_HANDLE;
    g_ready = false;
    g_seq = g_read = 0;
    g_prevSeq = UINT64_MAX;
    g_begun = g_lost = g_unfinished = 0;
}
} // namespace FrameTime_Vk
