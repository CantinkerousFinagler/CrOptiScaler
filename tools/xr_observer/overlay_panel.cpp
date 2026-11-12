#include <Windows.h>
#include <d3d11_4.h>

#include "overlay_panel.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace
{
OverlayPanel::LogFn g_log = nullptr;
#define PLOG(...)                                                                                                     \
    do                                                                                                                \
    {                                                                                                                 \
        if (g_log != nullptr)                                                                                         \
            g_log(__VA_ARGS__);                                                                                       \
    } while (false)

XrSession g_session = XR_NULL_HANDLE;
ID3D11Device* g_device = nullptr; // AddRef'd while the session lives
ID3D11DeviceContext* g_context = nullptr;
ID3D11DeviceContext4* g_context4 = nullptr;
OverlayPanel::Next g_next;
bool g_off = false, g_envRead = false;

PFN_RvrTools_GetOverlayV1 g_getOverlay = nullptr;
ULONGLONG g_nextResolveMs = 0;
bool g_loggedMissing = false;

// The game's shared textures, opened on the session's device.
ID3D11Texture2D* g_shared[kOverlaySlots] = {};
ID3D11Fence* g_fence = nullptr;
uint64_t g_fenceHandle = 0;
uint64_t g_generation = 0;      // the generation opened, or the one that failed to open
bool g_sharedOk = false;
uint32_t g_sharedWidth = 0, g_sharedHeight = 0;
DXGI_FORMAT g_sharedFamily = DXGI_FORMAT_R8G8B8A8_TYPELESS; // the opened texture's family (see OpenShared)
DXGI_FORMAT FamilyOf(DXGI_FORMAT f)
{
    switch (f)
    {
    case DXGI_FORMAT_R8G8B8A8_TYPELESS:
    case DXGI_FORMAT_R8G8B8A8_UNORM:
    case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
        return DXGI_FORMAT_R8G8B8A8_TYPELESS;
    case DXGI_FORMAT_B8G8R8A8_TYPELESS:
    case DXGI_FORMAT_B8G8R8A8_UNORM:
    case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
        return DXGI_FORMAT_B8G8R8A8_TYPELESS;
    case DXGI_FORMAT_R10G10B10A2_TYPELESS:
    case DXGI_FORMAT_R10G10B10A2_UNORM:
        return DXGI_FORMAT_R10G10B10A2_TYPELESS;
    default:
        return DXGI_FORMAT_UNKNOWN;
    }
}

// The panel's own swapchain.
struct Panel
{
    XrSwapchain handle = XR_NULL_HANDLE;
    std::vector<ID3D11Texture2D*> images; // the runtime's, not owned
    uint32_t width = 0, height = 0;
    int64_t format = 0;
    DXGI_FORMAT family = DXGI_FORMAT_UNKNOWN; // the shared texture's family it was made for
    bool released = false; // an image has been released: there is something to show
};
Panel g_panel;
struct RetiredPanel
{
    XrSwapchain handle = XR_NULL_HANDLE;
    uint64_t frame = 0;
};
std::vector<RetiredPanel> g_retired;
int g_panelSizes = 0;
XrSpace g_space = XR_NULL_HANDLE;
bool g_spaceTried = false;
uint64_t g_lastSequence = 0;
uint64_t g_frame = 0;

// Counters (since the last summary) and what is said once.
uint64_t g_shown = 0, g_copied = 0, g_hidden = 0, g_old = 0, g_failed = 0, g_noProjection = 0;
uint64_t g_shownTotal = 0;
uint32_t g_reasons = 0;

void Reason(int reason, const char* text)
{
    ++g_failed;
    const uint32_t bit = 1u << reason;
    if ((g_reasons & bit) != 0)
        return;
    g_reasons |= bit;
    PLOG("overlay panel: NOT shown: %s", text);
}

void CloseShared()
{
    for (auto*& texture : g_shared)
    {
        if (texture != nullptr)
            texture->Release();
        texture = nullptr;
    }
    g_sharedOk = false;
}

void CloseFence()
{
    if (g_fence != nullptr)
        g_fence->Release();
    g_fence = nullptr;
    g_fenceHandle = 0;
}

bool OpenShared(const RvrToolsOverlayV1& o)
{
    if (o.generation == g_generation)
        return g_sharedOk;
    CloseShared();
    g_generation = o.generation; // a generation that fails is not tried again at every frame
    // Each kind of handle with the interface that opens it; the fence only when the game's device made one.
    ID3D11Device1* device1 = nullptr;
    ID3D11Device5* device5 = nullptr;
    const bool fenced = o.writtenFenceHandle != 0;
    bool ok = true;
    const char* lacking = nullptr;
    if (o.legacyHandles == 0 && (FAILED(g_device->QueryInterface(IID_PPV_ARGS(&device1))) || device1 == nullptr))
        lacking = "the session's device has no ID3D11Device1 (NT shared handles)";
    else if (fenced && (FAILED(g_device->QueryInterface(IID_PPV_ARGS(&device5))) || device5 == nullptr ||
                        g_context4 == nullptr))
        lacking = "the session's device has no ID3D11Device5 / ID3D11DeviceContext4 (the game's fence)";
    if (lacking != nullptr)
    {
        PLOG("overlay panel: %s", lacking);
        ok = false;
    }
    for (int i = 0; ok && i < kOverlaySlots; ++i)
    {
        const HANDLE handle = reinterpret_cast<HANDLE>(o.handles[i]);
        ok = o.legacyHandles != 0 ? SUCCEEDED(g_device->OpenSharedResource(handle, IID_PPV_ARGS(&g_shared[i])))
                                  : SUCCEEDED(device1->OpenSharedResource1(handle, IID_PPV_ARGS(&g_shared[i])));
    }
    if (ok && !fenced)
        CloseFence();
    if (ok && fenced && (g_fence == nullptr || g_fenceHandle != o.writtenFenceHandle))
    {
        CloseFence();
        ok = SUCCEEDED(device5->OpenSharedFence(reinterpret_cast<HANDLE>(o.writtenFenceHandle), IID_PPV_ARGS(&g_fence)));
        g_fenceHandle = ok ? o.writtenFenceHandle : 0;
    }
    if (device1 != nullptr)
        device1->Release();
    if (device5 != nullptr)
        device5->Release();
    if (ok)
    {
        D3D11_TEXTURE2D_DESC desc {};
        g_shared[0]->GetDesc(&desc);
        // The texture's family decides the panel swapchain's format (the image is copied, not drawn): RGBA8 (D3D11
        // games), or BGRA8 / 10-bit RGB (a D3D12 game's window format, which ImGui's pipeline draws in).
        const DXGI_FORMAT family = FamilyOf(desc.Format);
        ok = desc.Width == o.width && desc.Height == o.height && desc.MipLevels == 1 && desc.SampleDesc.Count == 1 &&
             family != DXGI_FORMAT_UNKNOWN;
        if (!ok)
            PLOG("overlay panel: the game's shared texture is %ux%u, format %d, %u levels, %u samples: not the %ux%u "
                 "RGBA8 / BGRA8 / R10G10B10A2 that the panel takes",
                 desc.Width, desc.Height, (int) desc.Format, desc.MipLevels, desc.SampleDesc.Count, o.width, o.height);
        else
            g_sharedFamily = family;
    }
    if (!ok)
    {
        CloseShared();
        PLOG("overlay panel: opening the game's shared overlay (generation %llu, %ux%u) FAILED",
             (unsigned long long) o.generation, o.width, o.height);
        return false;
    }
    g_sharedOk = true;
    g_sharedWidth = o.width;
    g_sharedHeight = o.height;
    g_lastSequence = 0;
    PLOG("overlay panel: opened the game's shared overlay, generation %llu, %ux%u (%s handles, %s)",
         (unsigned long long) o.generation, o.width, o.height, o.legacyHandles != 0 ? "OLD-kind" : "NT",
         fenced ? "with its fence" : "NO fence: copied without waiting, one Present late");
    return true;
}

bool EnsurePanel(uint32_t width, uint32_t height)
{
    if (g_panel.handle != XR_NULL_HANDLE && g_panel.width == width && g_panel.height == height &&
        g_panel.family == g_sharedFamily)
        return true;
    if (g_panelSizes >= 24)
    {
        Reason(6, "more than 24 sizes of the overlay; no new panel (restart to reset)");
        return false;
    }
    if (g_panel.handle != XR_NULL_HANDLE)
        g_retired.push_back({ g_panel.handle, g_frame }); // it was in the frame just submitted
    g_panel = {};
    ++g_panelSizes;
    // The overlay's values are sRGB-encoded: an sRGB format shows them as the flat window does. RGBA only: the image
    // is copied, not drawn.
    uint32_t count = 0;
    std::vector<int64_t> formats;
    if (XR_SUCCEEDED(g_next.enumerateFormats(g_session, 0, &count, nullptr)) && count > 0)
    {
        formats.resize(count);
        if (XR_FAILED(g_next.enumerateFormats(g_session, count, &count, formats.data())))
            formats.clear();
    }
    // The swapchain in the shared texture's own family (the copy needs it): the sRGB variant first where there is one.
    std::vector<int64_t> wantedList;
    if (g_sharedFamily == DXGI_FORMAT_B8G8R8A8_TYPELESS)
        wantedList = { (int64_t) DXGI_FORMAT_B8G8R8A8_UNORM_SRGB, (int64_t) DXGI_FORMAT_B8G8R8A8_UNORM };
    else if (g_sharedFamily == DXGI_FORMAT_R10G10B10A2_TYPELESS)
        wantedList = { (int64_t) DXGI_FORMAT_R10G10B10A2_UNORM };
    else
        wantedList = { (int64_t) DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, (int64_t) DXGI_FORMAT_R8G8B8A8_UNORM };
    int64_t format = 0;
    for (const int64_t wanted : wantedList)
    {
        for (const int64_t offered : formats)
            if (format == 0 && offered == wanted)
                format = wanted;
    }
    if (format == 0)
    {
        std::string all;
        for (const int64_t offered : formats)
            all += std::to_string(offered) + " ";
        PLOG("overlay panel: the runtime offers no swapchain format of the game's texture family %d (it offers: %s)",
             (int) g_sharedFamily, all.c_str());
        Reason(7, "no swapchain format of the game's texture family");
        return false;
    }
    XrSwapchainCreateInfo info { XR_TYPE_SWAPCHAIN_CREATE_INFO };
    info.usageFlags = XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT | XR_SWAPCHAIN_USAGE_SAMPLED_BIT;
    info.format = format;
    info.sampleCount = 1;
    info.width = width;
    info.height = height;
    info.faceCount = 1;
    info.arraySize = 1;
    info.mipCount = 1;
    XrSwapchain handle = XR_NULL_HANDLE;
    const XrResult created = g_next.createSwapchain(g_session, &info, &handle);
    if (XR_FAILED(created) || handle == XR_NULL_HANDLE)
    {
        PLOG("overlay panel: xrCreateSwapchain %ux%u format %lld FAILED (%d)", width, height, (long long) format,
             (int) created);
        Reason(8, "the panel's swapchain could not be created");
        return false;
    }
    uint32_t images = 0;
    g_next.enumerateImages(handle, 0, &images, nullptr);
    std::vector<XrSwapchainImageD3D11KHR> list(images, { XR_TYPE_SWAPCHAIN_IMAGE_D3D11_KHR });
    if (images == 0 || XR_FAILED(g_next.enumerateImages(handle, images, &images,
                                                        reinterpret_cast<XrSwapchainImageBaseHeader*>(list.data()))))
    {
        g_next.destroySwapchain(handle); // never submitted
        Reason(9, "the panel's swapchain has no images");
        return false;
    }
    g_panel.handle = handle;
    g_panel.width = width;
    g_panel.height = height;
    g_panel.format = format;
    g_panel.family = g_sharedFamily;
    for (const auto& image : list)
        g_panel.images.push_back(image.texture);
    D3D11_TEXTURE2D_DESC desc {};
    g_panel.images[0]->GetDesc(&desc);
    PLOG("overlay panel: swapchain %ux%u, format %lld, %u images (their textures: format %d, %u levels)", width,
         height, (long long) format, images, (int) desc.Format, desc.MipLevels);
    return true;
}

bool EnsureSpace()
{
    if (g_space != XR_NULL_HANDLE)
        return true;
    if (g_spaceTried)
        return false;
    g_spaceTried = true;
    XrReferenceSpaceCreateInfo info { XR_TYPE_REFERENCE_SPACE_CREATE_INFO };
    info.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_VIEW;
    info.poseInReferenceSpace.orientation.w = 1.0f;
    const XrResult result = g_next.createReferenceSpace(g_session, &info, &g_space);
    if (XR_FAILED(result) || g_space == XR_NULL_HANDLE)
    {
        g_space = XR_NULL_HANDLE;
        PLOG("overlay panel: xrCreateReferenceSpace (VIEW) FAILED (%d)", (int) result);
        return false;
    }
    PLOG("overlay panel: VIEW reference space created");
    return true;
}
} // namespace

void OverlayPanel::SetLog(LogFn log) { g_log = log; }

void OverlayPanel::OnSession(XrSession session, ID3D11Device* device, const Next& next)
{
    if (g_device != nullptr)
        OnDestroySession(g_session);
    if (!g_envRead)
    {
        g_envRead = true;
        char value[8] = {};
        g_off = GetEnvironmentVariableA("XR_OBSERVER_PANEL", value, sizeof(value)) > 0 && value[0] == '0';
    }
    g_session = session;
    g_next = next;
    g_device = device;
    g_device->AddRef();
    g_device->GetImmediateContext(&g_context);
    g_context->QueryInterface(IID_PPV_ARGS(&g_context4));
    const bool complete = next.createSwapchain != nullptr && next.destroySwapchain != nullptr &&
                          next.enumerateImages != nullptr && next.acquire != nullptr && next.wait != nullptr &&
                          next.release != nullptr && next.enumerateFormats != nullptr &&
                          next.createReferenceSpace != nullptr && next.destroySpace != nullptr;
    if (!complete)
    {
        g_off = true;
        PLOG("overlay panel: OFF for this session (the runtime lacks a function it needs)");
    }
    else
    {
        PLOG("overlay panel: D3D11 session%s%s", g_off ? ", switched OFF (XR_OBSERVER_PANEL=0)" : " (idle until the "
                                                                                                "overlay draws)",
             g_context4 == nullptr ? "; this device has no ID3D11DeviceContext4: only an overlay without a fence can "
                                     "be shown"
                                   : "");
    }
}

void OverlayPanel::OnDestroySession(XrSession session)
{
    if (session != g_session || g_device == nullptr)
        return;
    if (g_panel.handle != XR_NULL_HANDLE && g_next.destroySwapchain != nullptr)
        g_next.destroySwapchain(g_panel.handle);
    g_panel = {};
    for (const auto& retired : g_retired)
        g_next.destroySwapchain(retired.handle);
    g_retired.clear();
    if (g_space != XR_NULL_HANDLE && g_next.destroySpace != nullptr)
        g_next.destroySpace(g_space);
    g_space = XR_NULL_HANDLE;
    g_spaceTried = false;
    CloseShared();
    CloseFence();
    g_generation = 0;
    if (g_context4 != nullptr)
        g_context4->Release();
    g_context4 = nullptr;
    if (g_context != nullptr)
        g_context->Release();
    g_context = nullptr;
    g_device->Release();
    g_device = nullptr;
    g_session = XR_NULL_HANDLE;
}

const XrFrameEndInfo* OverlayPanel::Append(const XrFrameEndInfo* info, Storage& storage)
{
    ++g_frame;
    if (g_off || g_device == nullptr || info == nullptr)
        return info;
    for (size_t i = 0; i < g_retired.size();)
    {
        if (g_frame - g_retired[i].frame >= 90)
        {
            g_next.destroySwapchain(g_retired[i].handle);
            g_retired[i] = g_retired.back();
            g_retired.pop_back();
        }
        else
        {
            ++i;
        }
    }
    if (g_getOverlay == nullptr && GetTickCount64() >= g_nextResolveMs)
    {
        g_nextResolveMs = GetTickCount64() + 2000;
        // Tests set XR_OBSERVER_PROVIDER to the stand-in's full path: there the system winmm.dll is loaded too.
        wchar_t provider[MAX_PATH] = L"winmm.dll";
        GetEnvironmentVariableW(L"XR_OBSERVER_PROVIDER", provider, MAX_PATH);
        if (HMODULE winmm = GetModuleHandleW(provider))
            g_getOverlay =
                reinterpret_cast<PFN_RvrTools_GetOverlayV1>(GetProcAddress(winmm, "RvrTools_GetOverlayV1"));
        if (g_getOverlay != nullptr)
            PLOG("overlay panel: the game's overlay (RvrTools_GetOverlayV1) found in winmm.dll");
        else if (!g_loggedMissing)
        {
            g_loggedMissing = true;
            PLOG("overlay panel: no overlay to show (no winmm.dll export RvrTools_GetOverlayV1); nothing changes");
        }
    }
    if (g_getOverlay == nullptr)
        return info;
    RvrToolsOverlayV1 o {};
    o.structSize = sizeof(o);
    if (g_getOverlay(&o) != 1)
    {
        Reason(0, "the game DLL's overlay struct is another size (winmm.dll and the layer are different builds)");
        return info;
    }
    if (o.visible == 0 || o.generation == 0 || o.sequence == 0)
    {
        ++g_hidden;
        return info;
    }
    LARGE_INTEGER now {}, frequency {};
    QueryPerformanceCounter(&now);
    QueryPerformanceFrequency(&frequency);
    if ((double) (now.QuadPart - o.qpc) / (double) frequency.QuadPart > 0.5)
    {
        ++g_old; // the game has stopped presenting: a panel that no longer follows the mouse is not shown
        return info;
    }
    if (o.width < 16 || o.height < 16 || o.width > 8192 || o.height > 8192 || o.usedWidth < 8 || o.usedHeight < 8 ||
        o.usedWidth > o.width || o.usedHeight > o.height)
    {
        Reason(1, "the overlay's size or its used part is not one a panel can have");
        return info;
    }
    // Only beside the application's picture: a frame without one (loading, a lost focus) is left alone.
    bool projection = false;
    for (uint32_t l = 0; l < info->layerCount; ++l)
        projection |= info->layers[l] != nullptr && info->layers[l]->type == XR_TYPE_COMPOSITION_LAYER_PROJECTION;
    if (!projection)
    {
        ++g_noProjection;
        return info;
    }
    if (!OpenShared(o))
    {
        Reason(2, "the game's shared overlay could not be opened");
        return info;
    }
    if (!EnsurePanel(o.width, o.height) || !EnsureSpace())
    {
        Reason(3, "no swapchain or no reference space for the panel");
        return info;
    }
    if (o.sequence != g_lastSequence)
    {
        uint32_t index = 0;
        XrSwapchainImageWaitInfo wait { XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO };
        wait.timeout = XR_INFINITE_DURATION;
        if (XR_FAILED(g_next.acquire(g_panel.handle, nullptr, &index)) || index >= g_panel.images.size())
        {
            Reason(4, "the panel's image could not be acquired");
            return info;
        }
        const bool waited = XR_SUCCEEDED(g_next.wait(g_panel.handle, &wait));
        if (waited)
        {
            if (g_fence != nullptr)
                g_context4->Wait(g_fence, o.sequence); // GPU-side: the game's draw of this image is done
            // The used part only: the rest of the texture is not the panel (and need not be moved).
            const D3D11_BOX used = { 0, 0, 0, o.usedWidth, o.usedHeight, 1 };
            g_context->CopySubresourceRegion(g_panel.images[index], 0, 0, 0, 0, g_shared[o.sequence % kOverlaySlots],
                                             0, &used);
            g_context->Flush(); // queued before the image is the runtime's
        }
        const bool released = XR_SUCCEEDED(g_next.release(g_panel.handle, nullptr));
        if (!waited || !released)
        {
            Reason(5, "the panel's image could not be waited for or released");
            return info;
        }
        g_panel.released = true;
        g_lastSequence = o.sequence;
        ++g_copied;
    }
    if (!g_panel.released)
        return info;

    storage.quad = { XR_TYPE_COMPOSITION_LAYER_QUAD };
    // The colour is premultiplied by its alpha (ImGui blended onto transparent black): blended, not "unpremultiplied".
    storage.quad.layerFlags = XR_COMPOSITION_LAYER_BLEND_TEXTURE_SOURCE_ALPHA_BIT;
    storage.quad.space = g_space;
    storage.quad.eyeVisibility = XR_EYE_VISIBILITY_BOTH;
    storage.quad.subImage.swapchain = g_panel.handle;
    storage.quad.subImage.imageRect = { { 0, 0 }, { (int32_t) o.usedWidth, (int32_t) o.usedHeight } };
    storage.quad.subImage.imageArrayIndex = 0;
    storage.quad.pose.orientation = { 0, 0, 0, 1 };
    storage.quad.pose.position = { 0.0f, -o.panelDown, -o.panelDistance }; // VIEW space: -z is ahead
    storage.quad.size.width = o.panelWidth;
    storage.quad.size.height = o.panelWidth * (float) o.usedHeight / (float) o.usedWidth;
    storage.layers.assign(info->layers, info->layers + info->layerCount);
    storage.layers.push_back(reinterpret_cast<const XrCompositionLayerBaseHeader*>(&storage.quad));
    storage.info = *info;
    storage.info.layerCount = (uint32_t) storage.layers.size();
    storage.info.layers = storage.layers.data();
    ++g_shown;
    if (++g_shownTotal == 1)
        PLOG("overlay panel: FIRST panel submitted: %ux%u of a %ux%u image, %.2f x %.2f m, %.2f m ahead, %.2f m down, "
             "as layer %u of %u",
             o.usedWidth, o.usedHeight, g_panel.width, g_panel.height, storage.quad.size.width,
             storage.quad.size.height, o.panelDistance, o.panelDown, storage.info.layerCount - 1,
             storage.info.layerCount);
    return &storage.info;
}

void OverlayPanel::Summary()
{
    if (g_off || g_getOverlay == nullptr)
        return;
    PLOG("OVERLAY PANEL: %llu frames with the panel (%llu new images copied); without it: %llu overlay closed, %llu "
         "its image older than 0.5 s, %llu no projection layer, %llu could not (each reason is said once)",
         (unsigned long long) g_shown, (unsigned long long) g_copied, (unsigned long long) g_hidden,
         (unsigned long long) g_old, (unsigned long long) g_noProjection, (unsigned long long) g_failed);
    g_shown = g_copied = g_hidden = g_old = g_noProjection = g_failed = 0;
}
