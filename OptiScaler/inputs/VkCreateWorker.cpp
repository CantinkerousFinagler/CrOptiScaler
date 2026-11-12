#include "VkCreateWorker.h"

#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>

namespace VkCreateWorker
{
namespace
{
std::mutex g_mutex;
std::condition_variable g_wake;
std::deque<Job> g_jobs;
std::thread g_thread;
bool g_stop = false;
bool g_available = false;
size_t g_running = 0;

VkDevice g_device = VK_NULL_HANDLE;
VkQueue g_queue = VK_NULL_HANDLE;
uint32_t g_family = UINT32_MAX, g_index = 0;
VkCommandPool g_pool = VK_NULL_HANDLE;
VkCommandBuffer g_cmd = VK_NULL_HANDLE;
VkFence g_fence = VK_NULL_HANDLE;

void Run()
{
    for (;;)
    {
        Job job;
        {
            std::unique_lock lock(g_mutex);
            g_wake.wait(lock, [] { return g_stop || !g_jobs.empty(); });
            if (g_stop)
                return;
            job = std::move(g_jobs.front());
            g_jobs.pop_front();
            ++g_running;
        }
        bool ok = false;
        if (vkResetCommandPool(g_device, g_pool, 0) == VK_SUCCESS)
        {
            VkCommandBufferBeginInfo bi { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
            bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            if (vkBeginCommandBuffer(g_cmd, &bi) == VK_SUCCESS)
            {
                const bool recorded = job.record ? job.record(g_cmd) : false;
                // Whatever the job's work wrote is made available to the game's queue (the same family: no
                // ownership transfer; the fence wait below orders it before anything the caller submits after).
                VkMemoryBarrier mb { VK_STRUCTURE_TYPE_MEMORY_BARRIER };
                mb.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
                mb.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
                vkCmdPipelineBarrier(g_cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0,
                                     1, &mb, 0, nullptr, 0, nullptr);
                const bool ended = vkEndCommandBuffer(g_cmd) == VK_SUCCESS;
                if (recorded && ended)
                {
                    VkSubmitInfo si { VK_STRUCTURE_TYPE_SUBMIT_INFO };
                    si.commandBufferCount = 1;
                    si.pCommandBuffers = &g_cmd;
                    if (vkResetFences(g_device, 1, &g_fence) == VK_SUCCESS &&
                        vkQueueSubmit(g_queue, 1, &si, g_fence) == VK_SUCCESS)
                        ok = vkWaitForFences(g_device, 1, &g_fence, VK_TRUE, 30'000'000'000ull) == VK_SUCCESS;
                }
            }
        }
        if (job.done)
            job.done(job.ticket, ok);
        {
            std::lock_guard lock(g_mutex);
            --g_running;
        }
    }
}
} // namespace

bool AskQueue(VkPhysicalDevice physical, VkDeviceCreateInfo& info, std::vector<VkDeviceQueueCreateInfo>& infos,
              std::vector<float>& priorities, std::string& why)
{
    uint32_t families = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &families, nullptr);
    std::vector<VkQueueFamilyProperties> fp(families);
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &families, fp.data());
    infos.assign(info.pQueueCreateInfos, info.pQueueCreateInfos + info.queueCreateInfoCount);
    for (size_t i = 0; i < infos.size(); ++i)
    {
        const uint32_t f = infos[i].queueFamilyIndex;
        if (f >= families || (fp[f].queueFlags & VK_QUEUE_GRAPHICS_BIT) == 0 ||
            (fp[f].queueFlags & VK_QUEUE_COMPUTE_BIT) == 0)
            continue;
        if (infos[i].queueCount >= fp[f].queueCount)
        {
            why = "the game asks for every queue of family " + std::to_string(f) + " (" +
                  std::to_string(fp[f].queueCount) + "): none to spare";
            return false;
        }
        priorities.assign(infos[i].pQueuePriorities, infos[i].pQueuePriorities + infos[i].queueCount);
        priorities.push_back(0.5f);
        g_family = f;
        g_index = infos[i].queueCount; // ours is the one after the game's
        infos[i].queueCount += 1;
        infos[i].pQueuePriorities = priorities.data();
        info.pQueueCreateInfos = infos.data();
        info.queueCreateInfoCount = (uint32_t) infos.size();
        why = "one more queue of family " + std::to_string(f) + " asked for (the game's " + std::to_string(g_index) +
              ", the family has " + std::to_string(fp[f].queueCount) + ")";
        return true;
    }
    why = "the game asks for no graphics + compute queue";
    return false;
}

bool DeviceMade(VkDevice device, PFN_vkGetDeviceQueue getDeviceQueue, std::string& why)
{
    std::lock_guard lock(g_mutex);
    if (g_available)
    {
        why = "already made";
        return true;
    }
    if (g_family == UINT32_MAX || device == VK_NULL_HANDLE || getDeviceQueue == nullptr)
    {
        why = "no queue was asked for";
        return false;
    }
    g_device = device;
    getDeviceQueue(device, g_family, g_index, &g_queue);
    if (g_queue == VK_NULL_HANDLE)
    {
        why = "vkGetDeviceQueue handed no queue";
        return false;
    }
    VkCommandPoolCreateInfo pi { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
    pi.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    pi.queueFamilyIndex = g_family;
    if (vkCreateCommandPool(device, &pi, nullptr, &g_pool) != VK_SUCCESS)
    {
        why = "vkCreateCommandPool failed";
        return false;
    }
    VkCommandBufferAllocateInfo ai { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    ai.commandPool = g_pool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VkFenceCreateInfo fi { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    if (vkAllocateCommandBuffers(device, &ai, &g_cmd) != VK_SUCCESS ||
        vkCreateFence(device, &fi, nullptr, &g_fence) != VK_SUCCESS)
    {
        why = "the command buffer or the fence could not be made";
        return false;
    }
    g_stop = false;
    g_thread = std::thread(Run);
    g_available = true;
    why = "queue " + std::to_string(g_index) + " of family " + std::to_string(g_family) + ", the thread started";
    return true;
}

bool Available()
{
    std::lock_guard lock(g_mutex);
    return g_available;
}

VkDevice Device() { return g_device; }
uint32_t QueueFamily() { return g_family; }

void Submit(Job job)
{
    {
        std::lock_guard lock(g_mutex);
        if (!g_available)
        {
            if (job.done)
                job.done(job.ticket, false);
            return;
        }
        g_jobs.push_back(std::move(job));
    }
    g_wake.notify_one();
}

size_t Pending()
{
    std::lock_guard lock(g_mutex);
    return g_jobs.size() + g_running;
}

void Shutdown()
{
    {
        std::lock_guard lock(g_mutex);
        if (!g_available)
            return;
        g_stop = true;
        g_available = false;
        g_jobs.clear();
    }
    g_wake.notify_all();
    if (g_thread.joinable())
        g_thread.join();
    if (g_queue != VK_NULL_HANDLE)
        vkQueueWaitIdle(g_queue);
    if (g_fence != VK_NULL_HANDLE)
        vkDestroyFence(g_device, g_fence, nullptr);
    if (g_pool != VK_NULL_HANDLE)
        vkDestroyCommandPool(g_device, g_pool, nullptr);
    g_fence = VK_NULL_HANDLE;
    g_pool = VK_NULL_HANDLE;
    g_cmd = VK_NULL_HANDLE;
    g_queue = VK_NULL_HANDLE;
    g_device = VK_NULL_HANDLE;
    g_family = UINT32_MAX;
}
} // namespace VkCreateWorker
