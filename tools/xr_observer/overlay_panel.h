#pragma once

// The game DLL's overlay as a flat panel in the headset (D3D11 sessions). The overlay is drawn on the game's flat
// window, which no headset shows; the game DLL (OptiScaler/inputs/OverlayShare) draws it a second time into shared
// textures while this layer asks, and this module copies the newest one into a swapchain of its own and adds it to
// the frame as a quad layer in front of the eyes (VIEW space: it stays in front of the head). With the overlay
// closed the frame is submitted as it came. Standard OpenXR only. XR_OBSERVER_PANEL=0 switches it off (tests).
#include <d3d11.h>

#define XR_USE_GRAPHICS_API_D3D11
#include "third_party/openxr/openxr.h"
#include "third_party/openxr/openxr_platform.h"

#include "../../OptiScaler/inputs/StaleEyeShareAbi.h"

#include <vector>

namespace OverlayPanel
{
struct Next
{
    PFN_xrCreateSwapchain createSwapchain = nullptr;
    PFN_xrDestroySwapchain destroySwapchain = nullptr;
    PFN_xrEnumerateSwapchainImages enumerateImages = nullptr;
    PFN_xrAcquireSwapchainImage acquire = nullptr;
    PFN_xrWaitSwapchainImage wait = nullptr;
    PFN_xrReleaseSwapchainImage release = nullptr;
    PFN_xrEnumerateSwapchainFormats enumerateFormats = nullptr;
    PFN_xrCreateReferenceSpace createReferenceSpace = nullptr;
    PFN_xrDestroySpace destroySpace = nullptr;
};

using LogFn = void (*)(const char* format, ...);
void SetLog(LogFn log);
void OnSession(XrSession session, ID3D11Device* device, const Next& next);
void OnDestroySession(XrSession session);

// The copies a frame with the panel is submitted from: they must live until the next xrEndFrame has returned.
struct Storage
{
    XrFrameEndInfo info {};
    std::vector<const XrCompositionLayerBaseHeader*> layers;
    XrCompositionLayerQuad quad {};
};
// The frame to submit: `info` itself, or a copy (in `storage`) with the panel added as its last layer. Called from
// xrEndFrame on the application's thread, with the layer's lock held.
const XrFrameEndInfo* Append(const XrFrameEndInfo* info, Storage& storage);

void Summary();
} // namespace OverlayPanel
