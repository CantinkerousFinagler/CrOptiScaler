// The headset menu layer: an OpenXR API layer that shows the game DLL's overlay as a flat panel in the headset, and
// does nothing else.
//
// The overlay is drawn on the game's flat window, which no headset shows. While the overlay is open the game DLL
// (OptiScaler/inputs/OverlayShare) draws it a second time into shared textures, and overlay_panel.cpp adds the newest
// one to the frame as a quad layer in front of the eyes. With the overlay closed every frame is submitted exactly as
// the application handed it in. Standard OpenXR only: nothing here is headset-specific. The panel needs the
// application's session on Direct3D 11; any other session is passed through untouched (said once in the log).
//
// Three calls are intercepted: xrCreateSession, xrDestroySession, xrEndFrame. No extension is added to the instance,
// nothing is exported besides the loader's entry, and there are no settings: where the panel sits comes from the
// game DLL with each image.
//
// The log goes beside this DLL as headset_menu.log, written anew at each start: what happens once (loading, the
// session, the first panel, a reason the panel could not be shown) and one line of counts each time the overlay is
// closed. Nothing is written per frame.
//
// The game DLL offers the layer to the process's OpenXR loader from <the OptiScaler folder>\xr_observer (this DLL as
// xr_observer.dll beside its manifest); it is not registered anywhere.
#include <Windows.h>
#include <d3d11.h>

#define XR_USE_GRAPHICS_API_D3D11
#define XR_USE_PLATFORM_WIN32
#include "third_party/openxr/openxr.h"
#include "third_party/openxr/openxr_platform.h"
#include "third_party/openxr/openxr_loader_negotiation.h"

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <type_traits>

#include "overlay_panel.h"

namespace
{
constexpr const char* kLayerName = "XR_APILAYER_OPTISCALER_RVR_xr_observer";

std::mutex g_mutex;
FILE* g_log = nullptr;
bool g_logTried = false;

void Log(const char* format, ...)
{
    if (g_log == nullptr)
    {
        if (g_logTried)
            return;
        g_logTried = true;
        wchar_t path[MAX_PATH] = {};
        HMODULE module = nullptr;
        GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           reinterpret_cast<LPCWSTR>(&Log), &module);
        GetModuleFileNameW(module, path, MAX_PATH);
        std::wstring file(path);
        file = file.substr(0, file.find_last_of(L"\\/") + 1) + L"headset_menu.log";
        g_log = _wfopen(file.c_str(), L"w");
        if (g_log == nullptr)
            return;
    }
    SYSTEMTIME t {};
    GetLocalTime(&t);
    std::fprintf(g_log, "[%02d:%02d:%02d.%03d] ", t.wHour, t.wMinute, t.wSecond, t.wMilliseconds);
    va_list args;
    va_start(args, format);
    std::vfprintf(g_log, format, args);
    va_end(args);
    std::fputc('\n', g_log);
    std::fflush(g_log);
}

struct Dispatch
{
    PFN_xrGetInstanceProcAddr gipa = nullptr;
    PFN_xrCreateSession createSession = nullptr;
    PFN_xrDestroySession destroySession = nullptr;
    PFN_xrEndFrame endFrame = nullptr;
    OverlayPanel::Next panel; // any of them missing switches the panel off (overlay_panel.cpp says so)
};
Dispatch g_next;

XrSession g_panelSession = XR_NULL_HANDLE; // the D3D11 session the panel was set up for
bool g_panelWasShown = false;              // the previous frame went out with the panel
} // namespace

extern "C" {

XRAPI_ATTR XrResult XRAPI_CALL MenuLayer_xrCreateSession(XrInstance instance, const XrSessionCreateInfo* info,
                                                         XrSession* session)
{
    const XrResult result = g_next.createSession(instance, info, session);
    std::lock_guard lock(g_mutex);
    ID3D11Device* device = nullptr;
    for (auto* next = info != nullptr ? static_cast<const XrBaseInStructure*>(info->next) : nullptr; next != nullptr;
         next = next->next)
    {
        if (next->type == XR_TYPE_GRAPHICS_BINDING_D3D11_KHR)
            device = reinterpret_cast<const XrGraphicsBindingD3D11KHR*>(next)->device;
    }
    if (XR_FAILED(result) || session == nullptr)
    {
        Log("xrCreateSession failed in the runtime (%d): nothing to do", (int) result);
        return result;
    }
    if (device == nullptr)
    {
        Log("xrCreateSession: the session is NOT on Direct3D 11: no panel for this session, every frame is passed "
            "through as it is");
        return result;
    }
    Log("xrCreateSession: a Direct3D 11 session");
    g_panelSession = *session;
    g_panelWasShown = false;
    OverlayPanel::OnSession(*session, device, g_next.panel);
    return result;
}

XRAPI_ATTR XrResult XRAPI_CALL MenuLayer_xrDestroySession(XrSession session)
{
    {
        std::lock_guard lock(g_mutex);
        if (session == g_panelSession)
        {
            OverlayPanel::Summary();
            OverlayPanel::OnDestroySession(session);
            g_panelSession = XR_NULL_HANDLE;
            g_panelWasShown = false;
            Log("xrDestroySession: the panel's session ended");
        }
    }
    return g_next.destroySession(session);
}

XRAPI_ATTR XrResult XRAPI_CALL MenuLayer_xrEndFrame(XrSession session, const XrFrameEndInfo* info)
{
    // The copy a frame with the panel is submitted from lives until the runtime's xrEndFrame has returned.
    OverlayPanel::Storage storage;
    const XrFrameEndInfo* submit = info;
    {
        std::lock_guard lock(g_mutex);
        if (session == g_panelSession && session != XR_NULL_HANDLE)
        {
            submit = OverlayPanel::Append(info, storage);
            const bool shown = submit != info;
            if (g_panelWasShown && !shown)
                OverlayPanel::Summary(); // the overlay was closed: one line of counts for the time it was open
            g_panelWasShown = shown;
        }
    }
    return g_next.endFrame(session, submit);
}

XRAPI_ATTR XrResult XRAPI_CALL MenuLayer_xrGetInstanceProcAddr(XrInstance instance, const char* name,
                                                               PFN_xrVoidFunction* function)
{
    if (function == nullptr)
        return XR_ERROR_VALIDATION_FAILURE;
    *function = nullptr;
    if (g_next.gipa == nullptr) // before our instance exists there is nothing to forward to
        return XR_ERROR_HANDLE_INVALID;
    const XrResult result = g_next.gipa(instance, name, function);
    if (XR_FAILED(result) || name == nullptr)
        return result;
#define INTERCEPT(fn)                                                                                                 \
    if (std::strcmp(name, #fn) == 0)                                                                                   \
    {                                                                                                                  \
        *function = reinterpret_cast<PFN_xrVoidFunction>(MenuLayer_##fn);                                             \
        return result;                                                                                                 \
    }
    INTERCEPT(xrGetInstanceProcAddr)
    INTERCEPT(xrCreateSession)
    INTERCEPT(xrDestroySession)
    INTERCEPT(xrEndFrame)
#undef INTERCEPT
    return result;
}

XRAPI_ATTR XrResult XRAPI_CALL MenuLayer_xrCreateApiLayerInstance(const XrInstanceCreateInfo* info,
                                                                  const XrApiLayerCreateInfo* layerInfo,
                                                                  XrInstance* instance)
{
    if (info == nullptr || layerInfo == nullptr || layerInfo->nextInfo == nullptr)
        return XR_ERROR_INITIALIZATION_FAILED;
    XrApiLayerCreateInfo nextLayerInfo = *layerInfo;
    nextLayerInfo.nextInfo = layerInfo->nextInfo->next;
    const PFN_xrGetInstanceProcAddr gipa = layerInfo->nextInfo->nextGetInstanceProcAddr;
    // The instance exactly as the application asked for it: this layer adds no extension.
    const XrResult result = layerInfo->nextInfo->nextCreateApiLayerInstance(info, &nextLayerInfo, instance);
    if (XR_FAILED(result))
        return result;

    std::lock_guard lock(g_mutex);
    g_next.gipa = gipa;
    auto load = [&](const char* fn, auto& out) {
        PFN_xrVoidFunction f = nullptr;
        gipa(*instance, fn, &f);
        out = reinterpret_cast<std::remove_reference_t<decltype(out)>>(f);
    };
    load("xrCreateSession", g_next.createSession);
    load("xrDestroySession", g_next.destroySession);
    load("xrEndFrame", g_next.endFrame);
    load("xrCreateSwapchain", g_next.panel.createSwapchain);
    load("xrDestroySwapchain", g_next.panel.destroySwapchain);
    load("xrEnumerateSwapchainImages", g_next.panel.enumerateImages);
    load("xrAcquireSwapchainImage", g_next.panel.acquire);
    load("xrWaitSwapchainImage", g_next.panel.wait);
    load("xrReleaseSwapchainImage", g_next.panel.release);
    load("xrEnumerateSwapchainFormats", g_next.panel.enumerateFormats);
    load("xrCreateReferenceSpace", g_next.panel.createReferenceSpace);
    load("xrDestroySpace", g_next.panel.destroySpace);
    OverlayPanel::SetLog(&Log);
    Log("headset menu layer: instance made for '%s' (API %u.%u), as the application asked", info->applicationInfo.applicationName,
        XR_VERSION_MAJOR(info->applicationInfo.apiVersion), XR_VERSION_MINOR(info->applicationInfo.apiVersion));
    if (g_next.createSession == nullptr || g_next.destroySession == nullptr || g_next.endFrame == nullptr)
        Log("headset menu layer: the runtime lacks xrCreateSession / xrDestroySession / xrEndFrame ?! nothing can work");
    return result;
}

__declspec(dllexport) XRAPI_ATTR XrResult XRAPI_CALL xrNegotiateLoaderApiLayerInterface(
    const XrNegotiateLoaderInfo* loaderInfo, const char* layerName, XrNegotiateApiLayerRequest* request)
{
    if (loaderInfo == nullptr || request == nullptr || loaderInfo->structType != XR_LOADER_INTERFACE_STRUCT_LOADER_INFO ||
        request->structType != XR_LOADER_INTERFACE_STRUCT_API_LAYER_REQUEST ||
        loaderInfo->minInterfaceVersion > XR_CURRENT_LOADER_API_LAYER_VERSION ||
        loaderInfo->maxInterfaceVersion < XR_CURRENT_LOADER_API_LAYER_VERSION || layerName == nullptr ||
        std::strcmp(layerName, kLayerName) != 0)
    {
        std::lock_guard lock(g_mutex);
        Log("headset menu layer: negotiation REFUSED: asked for as '%s' (this layer is '%s'), loader interface %u..%u "
            "(this layer speaks %u)",
            layerName != nullptr ? layerName : "(null)", kLayerName,
            loaderInfo != nullptr ? loaderInfo->minInterfaceVersion : 0,
            loaderInfo != nullptr ? loaderInfo->maxInterfaceVersion : 0, (unsigned) XR_CURRENT_LOADER_API_LAYER_VERSION);
        return XR_ERROR_INITIALIZATION_FAILED;
    }
    request->layerInterfaceVersion = XR_CURRENT_LOADER_API_LAYER_VERSION;
    request->layerApiVersion = (std::min)(loaderInfo->maxApiVersion, XR_CURRENT_API_VERSION);
    request->getInstanceProcAddr = MenuLayer_xrGetInstanceProcAddr;
    request->createApiLayerInstance = MenuLayer_xrCreateApiLayerInstance;
    {
        std::lock_guard lock(g_mutex);
        Log("headset menu layer: loaded by the OpenXR loader");
    }
    return XR_SUCCESS;
}

BOOL WINAPI DllMain(HINSTANCE module, DWORD reason, LPVOID)
{
    if (reason == DLL_PROCESS_ATTACH)
        DisableThreadLibraryCalls(module);
    return TRUE;
}
}
