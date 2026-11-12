#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

// A WORKER THREAD WITH A QUEUE OF ITS OWN for GPU work that must not sit on the game's thread: a
// DLSS feature's first create at a new size compiles pipelines for over a second
// (1.30 s for a large cheap periphery at preset E), and a Present a second late restarts the headset session.
// The game's own features are the game's; the features of OUR OWN (the foveal crop's, the periphery's) are made here:
// their NGX create records into a command buffer of ours and is submitted on a queue of ours, so the game presents
// meanwhile and the full pass runs until the feature is ready.
//
// Pure Vulkan, no OptiScaler code: tools/xr_observer/test/vk_fovea_test runs it on a real device (a job that records
// a clear on the worker's command buffer, its result read back; the controls: no queue asked, a failing job).
//
// How the queue is got: AskQueue() in the vkCreateDevice hook asks for ONE MORE queue of the family the game's first
// graphics + compute queue is in (if the family has one to spare); DeviceMade() takes it, makes a command pool and a
// fence and starts the thread. Without a spare queue the worker is unavailable and the caller creates inline.
namespace VkCreateWorker
{
// Edits `info` (the game's create info, copied by the caller) so that the graphics + compute family gets one more
// queue: `infos` and `priorities` are the caller's scratch (they must outlive the vkCreateDevice call). False =
// nothing changed (no such family, or no queue to spare); `why` says which.
bool AskQueue(VkPhysicalDevice physical, VkDeviceCreateInfo& info, std::vector<VkDeviceQueueCreateInfo>& infos,
              std::vector<float>& priorities, std::string& why);

// After vkCreateDevice succeeded with the edited info: our queue (through `getDeviceQueue`, the device's own
// function so the loader's chain is honoured), a pool, a fence, the thread. False = unavailable (`why`).
bool DeviceMade(VkDevice device, PFN_vkGetDeviceQueue getDeviceQueue, std::string& why);
bool Available();
VkDevice Device();
uint32_t QueueFamily();

// A job: `record` runs ON THE WORKER THREAD with the command buffer in recording state (it may record, or make
// things that record into it); on true the buffer is submitted on our queue and the fence waited, then `done`
// runs on the worker with the outcome (submitted and completed, or not). On false nothing is submitted. Jobs run one
// at a time, in order.
struct Job
{
    uint64_t ticket = 0; // the caller's
    std::function<bool(VkCommandBuffer)> record;
    std::function<void(uint64_t ticket, bool ok)> done;
};
void Submit(Job job);
// Jobs waiting or running.
size_t Pending();
// Ends the thread (jobs left are dropped, `done` not called); the pool and fence destroyed once the queue is idle.
void Shutdown();
} // namespace VkCreateWorker
