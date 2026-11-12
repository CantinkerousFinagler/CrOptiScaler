#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <string>

// GPU TIME PER FRAME ON THE VULKAN PATH: under the headset's
// refresh cap a saving never shows as frames per second, only as milliseconds. Four timestamps in the GAME's command
// buffer at every DLSS evaluate (one evaluate per Present here): 0 at the evaluate's start, 1 after our periphery (or
// nothing), 2 after the game's full DLSS pass (or nothing), 3 after our crop and composite. The frame = timestamp 0
// of one evaluate to timestamp 0 of the next (the whole GPU frame, the game's raster included); the game's full pass
// = 2 - 1 on the evaluates where it ran; ours = (1 - 0) + (3 - 2). A ring of slots, read back without waiting once
// they are old enough. Pure Vulkan: tools/xr_observer/test/vk_fovea_test runs it on a real device.
namespace FrameTime_Vk
{
constexpr uint32_t kSlots = 64;
constexpr uint32_t kStamps = 4;

bool Init(VkDevice device, VkPhysicalDevice physical, std::string& why);
bool Ready();
// A new slot: its four queries reset and stamp 0 written. Returns the slot's sequence number.
uint64_t Begin(VkCommandBuffer cmd);
// Stamp `stage` (1..3) of the current slot on the same command buffer; `ran` says whether the work before it ran
// (the game's full pass for stage 2).
void Mark(VkCommandBuffer cmd, uint32_t stage, bool ran);
// Reads back every slot that is at least `age` evaluates old and not read yet; accumulates into the window. CALL IT
// AT EVERY PRESENT: the ring holds kSlots evaluates. Slots begun more than a ring ago are gone (overwritten): they are
// skipped and counted as lost. (An earlier version called it once per 600 Presents and stopped for good at the first overrun:
// 45 windows without one reading and without a word. `skipOverrun` false is that behaviour, for the
// offline test's control only.)
void Collect(uint64_t age = 8, bool skipOverrun = true);
struct Window
{
    uint32_t frames = 0;    // consecutive evaluate pairs measured
    double frameMs = 0.0;   // their mean GPU time, stamp 0 to the next stamp 0
    uint32_t gameRuns = 0;  // evaluates on which the game's full pass ran
    double gameMs = 0.0;    // its mean (2 - 1)
    uint32_t ours = 0;      // evaluates measured for our passes
    double beforeMs = 0.0;  // mean (1 - 0): the periphery (DLSS or TAA + enlarge) and the gather copies
    double afterMs = 0.0;   // mean (3 - 2): the crop's DLSS and the composite
    uint64_t begun = 0;     // slots begun since the last Take
    uint64_t lost = 0;      // slots overwritten before they were read (Collect came too rarely)
    uint64_t unfinished = 0; // slots given up: their stamps never came (the evaluate returned early)
};
// What was accumulated since the last Take (means made), then cleared.
Window Take();
void Shutdown();
} // namespace FrameTime_Vk
