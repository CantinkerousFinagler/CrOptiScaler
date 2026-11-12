#pragma once

// Vulkan DLSS features released without a stall ([Hotfix] VulkanNoStalls).
// OptiScaler's Vulkan release waited for the device to go idle and slept 1 s; its create slept 0.5 s more,
// twice for a game that makes two features. A long gap in Presents restarts the headset session, and
// vkDeviceWaitIdle mid-frame is not safe in VR (a device lost followed a resolution change). With the switch on, a released feature is kept and destroyed kAfterPresents Presents
// later, when no frame in flight can still use it.
namespace VkRetire
{
// From the vkQueuePresentKHR hook: destroys what was released long enough ago.
void OnPresent();
} // namespace VkRetire
