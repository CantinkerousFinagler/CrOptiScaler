#pragma once
#include "SysUtils.h"

class VulkanHooks
{
  public:
    static PFN_vkCreateSemaphore o_vkCreateSemaphore;
    static PFN_vkSignalSemaphore o_vkSignalSemaphore;
    static PFN_vkAntiLagUpdateAMD o_vkAntiLagUpdateAMD;

    static void Hook(HMODULE vulkan1);
    static void Unhook();
    // One line at exit: were the device hooks placed by Detours or by the fallback, and did a Present reach us.
    static void LogDeviceHookSummary();
};
