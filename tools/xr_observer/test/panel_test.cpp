// Offline test of the headset menu layer (menu_layer.cpp + overlay_panel.cpp) on real D3D11: device A plays the game
// (the overlay's shared images and their fence, as the game DLL's OverlayShare makes them), device B plays the OpenXR
// application's device, and stub "next" functions play the runtime (swapchains on device B that the test reads back).
//   panel_test <layer dll> <fake winmm dll> <mode 1|2|3>
// Mode 1: three shared RGBA8 images of 96 x 40, each with its own pattern and alpha, the newest named by a sequence
//         and a fence. While the overlay is visible the frame must carry one more layer, a quad in VIEW space of the
//         layer's own swapchain, blended by its alpha, at the size and place the game DLL gives, holding exactly the
//         newest image; once it is not visible the frame must be the application's own, handed on untouched.
//         Control, set by hand: XR_OBSERVER_PANEL=0 (no panel).
// Mode 2: the same WITHOUT a fence (a game's device that has none): fence handle 0, nothing signalled; the layer must
//         copy without waiting.
// Mode 3: without a fence AND with the old kind of shared handle (typed RGBA8, IDXGIResource::GetSharedHandle), opened
//         with ID3D11Device::OpenSharedResource.
#include <windows.h>
#include <d3d11_4.h>
#include <dxgi1_2.h>

#define XR_USE_GRAPHICS_API_D3D11
#define XR_USE_PLATFORM_WIN32
#include "../third_party/openxr/openxr.h"
#include "../third_party/openxr/openxr_platform.h"
#include "../third_party/openxr/openxr_loader_negotiation.h"
#include "../../../OptiScaler/inputs/StaleEyeShareAbi.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <deque>
#include <map>
#include <vector>

namespace
{
ID3D11Device* g_b = nullptr;
ID3D11DeviceContext* g_bctx = nullptr;
constexpr UINT kW = 64, kH = 48;

struct StubSwapchain
{
    std::vector<ID3D11Texture2D*> images;
    uint32_t next = 0;
    std::deque<uint32_t> acquired; // FIFO, as OpenXR releases
    // What the compositor took at the last release. The stub then WIPES the released image: once released it is the
    // runtime's, and reading it a frame later gives 0 0 0 0.
    ID3D11Texture2D* shown = nullptr;
    ID3D11Texture2D* zero = nullptr;
};
std::map<uint64_t, StubSwapchain> g_swapchains;
uint64_t g_nextHandle = 100;
struct Captured
{
    const XrFrameEndInfo* info = nullptr; // the very pointer the runtime was handed
    uint32_t layerCount = 0;
    int quads = 0;
    XrCompositionLayerQuad quad {};
};
constexpr uint64_t kStubSpace = 777;
int g_spacesCreated = 0, g_spacesDestroyed = 0;
Captured g_captured;

XrResult XRAPI_CALL StubCreateSession(XrInstance, const XrSessionCreateInfo*, XrSession* s)
{
    *s = (XrSession) 1;
    return XR_SUCCESS;
}
XrResult XRAPI_CALL StubDestroySession(XrSession) { return XR_SUCCESS; }
XrResult XRAPI_CALL StubCreateSwapchain(XrSession, const XrSwapchainCreateInfo* info, XrSwapchain* out)
{
    StubSwapchain sc;
    for (int i = 0; i < 3; ++i)
    {
        D3D11_TEXTURE2D_DESC d {};
        d.Width = info->width;
        d.Height = info->height;
        d.MipLevels = info->mipCount;
        d.ArraySize = 1;
        d.Format = DXGI_FORMAT_R8G8B8A8_TYPELESS; // as runtimes create sRGB swapchains
        d.SampleDesc.Count = 1;
        d.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
        ID3D11Texture2D* t = nullptr;
        g_b->CreateTexture2D(&d, nullptr, &t);
        sc.images.push_back(t);
    }
    const uint64_t h = ++g_nextHandle;
    g_swapchains[h] = sc;
    *out = (XrSwapchain) h;
    return XR_SUCCESS;
}
XrResult XRAPI_CALL StubDestroySwapchain(XrSwapchain) { return XR_SUCCESS; }
XrResult XRAPI_CALL StubEnumerate(XrSwapchain s, uint32_t cap, uint32_t* count, XrSwapchainImageBaseHeader* images)
{
    auto& sc = g_swapchains[(uint64_t) s];
    *count = (uint32_t) sc.images.size();
    if (cap == 0)
        return XR_SUCCESS;
    auto* d3d = reinterpret_cast<XrSwapchainImageD3D11KHR*>(images);
    for (uint32_t i = 0; i < *count && i < cap; ++i)
        d3d[i].texture = sc.images[i];
    return XR_SUCCESS;
}
XrResult XRAPI_CALL StubAcquire(XrSwapchain s, const XrSwapchainImageAcquireInfo*, uint32_t* index)
{
    auto& sc = g_swapchains[(uint64_t) s];
    sc.acquired.push_back(sc.next);
    *index = sc.next;
    sc.next = (sc.next + 1) % (uint32_t) sc.images.size();
    return XR_SUCCESS;
}
XrResult XRAPI_CALL StubWait(XrSwapchain, const XrSwapchainImageWaitInfo*) { return XR_SUCCESS; }
XrResult XRAPI_CALL StubRelease(XrSwapchain s, const XrSwapchainImageReleaseInfo*)
{
    auto& sc = g_swapchains[(uint64_t) s];
    if (!sc.acquired.empty())
    {
        const uint32_t released = sc.acquired.front();
        sc.acquired.pop_front();
        ID3D11Texture2D* image = sc.images[released];
        if (sc.shown == nullptr)
        {
            D3D11_TEXTURE2D_DESC d {};
            image->GetDesc(&d);
            g_b->CreateTexture2D(&d, nullptr, &sc.shown);
            std::vector<uint8_t> zeros(d.Width * d.Height * 4, 0);
            std::vector<D3D11_SUBRESOURCE_DATA> init(d.MipLevels);
            for (UINT m = 0; m < d.MipLevels; ++m)
                init[m] = { zeros.data(), (std::max)(1u, d.Width >> m) * 4, 0 };
            g_b->CreateTexture2D(&d, init.data(), &sc.zero);
        }
        g_bctx->CopyResource(sc.shown, image);
        g_bctx->CopyResource(image, sc.zero);
    }
    return XR_SUCCESS;
}
XrResult XRAPI_CALL StubEndFrame(XrSession, const XrFrameEndInfo* info)
{
    g_captured.info = info;
    g_captured.layerCount = info->layerCount;
    g_captured.quads = 0;
    for (uint32_t l = 0; l < info->layerCount; ++l)
        if (info->layers[l] != nullptr && info->layers[l]->type == XR_TYPE_COMPOSITION_LAYER_QUAD)
        {
            ++g_captured.quads;
            g_captured.quad = *reinterpret_cast<const XrCompositionLayerQuad*>(info->layers[l]);
        }
    return XR_SUCCESS;
}
XrResult XRAPI_CALL StubEnumerateFormats(XrSession, uint32_t capacity, uint32_t* count, int64_t* formats)
{
    // As runtimes list them: what they prefer first. RGBA8 sRGB is not the first on purpose.
    const int64_t offered[] = { DXGI_FORMAT_B8G8R8A8_UNORM_SRGB, DXGI_FORMAT_R8G8B8A8_UNORM_SRGB,
                                DXGI_FORMAT_R8G8B8A8_UNORM };
    *count = 3;
    for (uint32_t i = 0; i < capacity && i < 3 && formats != nullptr; ++i)
        formats[i] = offered[i];
    return XR_SUCCESS;
}
XrResult XRAPI_CALL StubCreateReferenceSpace(XrSession, const XrReferenceSpaceCreateInfo* info, XrSpace* space)
{
    if (info == nullptr || info->referenceSpaceType != XR_REFERENCE_SPACE_TYPE_VIEW)
        return XR_ERROR_REFERENCE_SPACE_UNSUPPORTED;
    ++g_spacesCreated;
    *space = (XrSpace) kStubSpace;
    return XR_SUCCESS;
}
XrResult XRAPI_CALL StubDestroySpace(XrSpace)
{
    ++g_spacesDestroyed;
    return XR_SUCCESS;
}

XrResult XRAPI_CALL StubGipa(XrInstance, const char* name, PFN_xrVoidFunction* f)
{
    struct
    {
        const char* n;
        void* p;
    } table[] = { { "xrCreateSession", (void*) &StubCreateSession },
                  { "xrDestroySession", (void*) &StubDestroySession },
                  { "xrCreateSwapchain", (void*) &StubCreateSwapchain },
                  { "xrDestroySwapchain", (void*) &StubDestroySwapchain },
                  { "xrEnumerateSwapchainImages", (void*) &StubEnumerate },
                  { "xrAcquireSwapchainImage", (void*) &StubAcquire },
                  { "xrWaitSwapchainImage", (void*) &StubWait },
                  { "xrReleaseSwapchainImage", (void*) &StubRelease },
                  { "xrEndFrame", (void*) &StubEndFrame },
                  { "xrEnumerateSwapchainFormats", (void*) &StubEnumerateFormats },
                  { "xrCreateReferenceSpace", (void*) &StubCreateReferenceSpace },
                  { "xrDestroySpace", (void*) &StubDestroySpace } };
    for (auto& e : table)
        if (std::strcmp(name, e.n) == 0)
        {
            *f = (PFN_xrVoidFunction) e.p;
            return XR_SUCCESS;
        }
    *f = nullptr;
    return XR_ERROR_FUNCTION_UNSUPPORTED;
}
XrResult XRAPI_CALL StubCreateInstance(const XrInstanceCreateInfo*, const XrApiLayerCreateInfo*, XrInstance* i)
{
    *i = (XrInstance) 0x1234;
    return XR_SUCCESS;
}

std::vector<uint8_t> ReadBack(ID3D11Texture2D* t)
{
    D3D11_TEXTURE2D_DESC d {};
    t->GetDesc(&d);
    d.MipLevels = 1;
    d.Usage = D3D11_USAGE_STAGING;
    d.BindFlags = 0;
    d.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    d.MiscFlags = 0;
    ID3D11Texture2D* s = nullptr;
    g_b->CreateTexture2D(&d, nullptr, &s);
    g_bctx->CopySubresourceRegion(s, 0, 0, 0, 0, t, 0, nullptr);
    D3D11_MAPPED_SUBRESOURCE m {};
    g_bctx->Map(s, 0, D3D11_MAP_READ, 0, &m);
    std::vector<uint8_t> out(d.Width * d.Height * 4);
    for (UINT y = 0; y < d.Height; ++y)
        std::memcpy(out.data() + y * d.Width * 4, (uint8_t*) m.pData + y * m.RowPitch, d.Width * 4);
    g_bctx->Unmap(s, 0);
    s->Release();
    return out;
}
} // namespace

int main(int argc, char** argv)
{
    if (argc < 4)
    {
        std::printf("usage: panel_test <layer dll> <fake winmm dll> <mode 1|2|3>\n");
        return 2;
    }
    const int arg = std::atoi(argv[3]);
    if (arg < 1 || arg > 3)
    {
        std::printf("FAIL: mode %d (1 = with a fence, 2 = no fence, 3 = no fence and old-kind handles)\n", arg);
        return 2;
    }
    const bool panelNoFence = arg == 2 || arg == 3, panelLegacy = arg == 3;

    // Devices FIRST: the GPU driver imports the system winmm.dll and must not get the stand-in. Then the layer is
    // told the stand-in's full path (XR_OBSERVER_PROVIDER), as two winmm.dll modules are loaded.
    // Devices: A = the game, B = the XR session's.
    ID3D11Device* a = nullptr;
    ID3D11DeviceContext* actx = nullptr;
    const HRESULT ha =
        D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, nullptr, 0, D3D11_SDK_VERSION, &a, nullptr, &actx);
    const HRESULT hb = D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, nullptr, 0, D3D11_SDK_VERSION,
                                         &g_b, nullptr, &g_bctx);
    if (a == nullptr || g_b == nullptr)
    {
        std::printf("FAIL: devices (%08lX, %08lX)\n", (unsigned long) ha, (unsigned long) hb);
        return 1;
    }
    SetEnvironmentVariableA("XR_OBSERVER_PROVIDER", argv[2]);
    HMODULE winmm = LoadLibraryA(argv[2]);
    HMODULE layer = LoadLibraryA(argv[1]);
    if (winmm == nullptr || layer == nullptr)
    {
        std::printf("FAIL: load (winmm %p, layer %p)\n", winmm, layer);
        return 1;
    }
    ID3D11Device5* a5 = nullptr;
    a->QueryInterface(IID_PPV_ARGS(&a5));
    ID3D11DeviceContext4* actx4 = nullptr;
    actx->QueryInterface(IID_PPV_ARGS(&actx4));

    // The overlay's shared images (device A, as the game DLL makes them) and their fence.
    constexpr UINT kPanelW = 96, kPanelH = 40;
    std::vector<uint8_t> panelPattern[kOverlaySlots];
    ID3D11Fence* panelFence = nullptr;
    auto setOverlay =
        (void (*)(uint64_t, const uint64_t*, uint32_t, uint32_t, uint64_t)) GetProcAddress(winmm, "SetOverlay");
    auto setOverlayFrame = (void (*)(uint64_t, int)) GetProcAddress(winmm, "SetOverlayFrame");
    auto setOverlayLegacy = (void (*)(int)) GetProcAddress(winmm, "SetOverlayLegacy");
    auto overlayAsked = (int (*)()) GetProcAddress(winmm, "GetOverlayAsked");
    HANDLE hf = nullptr;
    if (!panelNoFence && a5 != nullptr)
    {
        a5->CreateFence(0, D3D11_FENCE_FLAG_SHARED, IID_PPV_ARGS(&panelFence));
        if (panelFence != nullptr)
            panelFence->CreateSharedHandle(nullptr, GENERIC_ALL, nullptr, &hf);
    }
    uint64_t panelHandles[kOverlaySlots] = {};
    for (int i = 0; i < kOverlaySlots; ++i)
    {
        panelPattern[i].resize(kPanelW * kPanelH * 4);
        for (UINT y = 0; y < kPanelH; ++y)
            for (UINT x = 0; x < kPanelW; ++x)
            {
                uint8_t* p = &panelPattern[i][(y * kPanelW + x) * 4];
                p[0] = (uint8_t) (x * 2 + i * 40);
                p[1] = (uint8_t) (y * 6);
                p[2] = (uint8_t) (200 - i * 60);
                p[3] = (uint8_t) ((x * 3 + y * 5 + i * 17) % 256); // every alpha, 0 included
            }
        D3D11_TEXTURE2D_DESC d {};
        d.Width = kPanelW;
        d.Height = kPanelH;
        d.MipLevels = 1;
        d.ArraySize = 1;
        d.Format = panelLegacy ? DXGI_FORMAT_R8G8B8A8_UNORM : DXGI_FORMAT_R8G8B8A8_TYPELESS;
        d.SampleDesc.Count = 1;
        d.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
        d.MiscFlags =
            panelLegacy ? D3D11_RESOURCE_MISC_SHARED : (D3D11_RESOURCE_MISC_SHARED | D3D11_RESOURCE_MISC_SHARED_NTHANDLE);
        D3D11_SUBRESOURCE_DATA init { panelPattern[i].data(), kPanelW * 4, 0 };
        ID3D11Texture2D* t = nullptr;
        a->CreateTexture2D(&d, &init, &t);
        IDXGIResource1* r = nullptr;
        IDXGIResource* old = nullptr;
        HANDLE h = nullptr;
        if (panelLegacy && t != nullptr && SUCCEEDED(t->QueryInterface(IID_PPV_ARGS(&old))))
        {
            old->GetSharedHandle(&h);
            old->Release();
        }
        else if (!panelLegacy && t != nullptr && SUCCEEDED(t->QueryInterface(IID_PPV_ARGS(&r))))
        {
            r->CreateSharedHandle(nullptr, DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE, nullptr, &h);
            r->Release();
        }
        panelHandles[i] = (uint64_t) h;
    }
    if (setOverlay == nullptr || setOverlayFrame == nullptr || overlayAsked == nullptr || setOverlayLegacy == nullptr ||
        panelHandles[0] == 0 || panelHandles[1] == 0 || panelHandles[2] == 0 || (hf == nullptr && !panelNoFence) ||
        (actx4 == nullptr && !panelNoFence))
    {
        std::printf("FAIL: the overlay's shared images could not be set up\n");
        return 1;
    }
    actx->Flush(); // the initial data must be on the GPU before the layer reads
    setOverlay(1, panelHandles, kPanelW, kPanelH, (uint64_t) hf);
    setOverlayLegacy(panelLegacy ? 1 : 0);
    std::printf("overlay: 3 shared images %ux%u, %s, %s handles\n", kPanelW, kPanelH,
                panelNoFence ? "NO fence" : "with a fence", panelLegacy ? "old-kind" : "NT");

    // The layer, on stub "next" functions.
    auto negotiate = (PFN_xrNegotiateLoaderApiLayerInterface) GetProcAddress(layer, "xrNegotiateLoaderApiLayerInterface");
    XrNegotiateLoaderInfo li { XR_LOADER_INTERFACE_STRUCT_LOADER_INFO, XR_LOADER_INFO_STRUCT_VERSION,
                               sizeof(XrNegotiateLoaderInfo), 1, 1, XR_MAKE_VERSION(1, 0, 0), XR_MAKE_VERSION(1, 0, 0xFFF) };
    XrNegotiateApiLayerRequest req { XR_LOADER_INTERFACE_STRUCT_API_LAYER_REQUEST, XR_API_LAYER_INFO_STRUCT_VERSION,
                                     sizeof(XrNegotiateApiLayerRequest) };
    if (negotiate == nullptr || XR_FAILED(negotiate(&li, "XR_APILAYER_OPTISCALER_RVR_xr_observer", &req)))
    {
        std::printf("FAIL: negotiate\n");
        return 1;
    }
    XrApiLayerNextInfo next { XR_LOADER_INTERFACE_STRUCT_API_LAYER_NEXT_INFO, XR_API_LAYER_NEXT_INFO_STRUCT_VERSION,
                              sizeof(XrApiLayerNextInfo) };
    std::strcpy(next.layerName, "stub");
    next.nextGetInstanceProcAddr = StubGipa;
    next.nextCreateApiLayerInstance = StubCreateInstance;
    XrApiLayerCreateInfo lci { XR_LOADER_INTERFACE_STRUCT_API_LAYER_CREATE_INFO, XR_API_LAYER_CREATE_INFO_STRUCT_VERSION,
                               sizeof(XrApiLayerCreateInfo) };
    lci.nextInfo = &next;
    XrInstanceCreateInfo ici { XR_TYPE_INSTANCE_CREATE_INFO };
    std::strcpy(ici.applicationInfo.applicationName, "panel_test");
    const char* appExtensions[] = { "XR_KHR_D3D11_enable" };
    ici.enabledExtensionCount = 1;
    ici.enabledExtensionNames = appExtensions;
    ici.applicationInfo.apiVersion = XR_MAKE_VERSION(1, 0, 0);
    XrInstance instance = XR_NULL_HANDLE;
    req.createApiLayerInstance(&ici, &lci, &instance);
    auto get = [&](const char* n) {
        PFN_xrVoidFunction f = nullptr;
        req.getInstanceProcAddr(instance, n, &f);
        return f;
    };
    auto createSession = (PFN_xrCreateSession) get("xrCreateSession");
    auto createSwapchain = (PFN_xrCreateSwapchain) get("xrCreateSwapchain");
    auto endFrame = (PFN_xrEndFrame) get("xrEndFrame");
    auto destroySession = (PFN_xrDestroySession) get("xrDestroySession");
    if (createSession == nullptr || createSwapchain == nullptr || endFrame == nullptr || destroySession == nullptr)
    {
        std::printf("FAIL: the layer hands out no xrCreateSession / xrCreateSwapchain / xrEndFrame / xrDestroySession\n");
        return 1;
    }

    XrGraphicsBindingD3D11KHR binding { XR_TYPE_GRAPHICS_BINDING_D3D11_KHR };
    binding.device = g_b;
    XrSessionCreateInfo sci { XR_TYPE_SESSION_CREATE_INFO, &binding };
    XrSession session = XR_NULL_HANDLE;
    createSession(instance, &sci, &session);

    // The application's two eye swapchains and its one projection layer. The layer under test never touches them.
    XrSwapchain app[2] = {};
    for (int v = 0; v < 2; ++v)
    {
        XrSwapchainCreateInfo info { XR_TYPE_SWAPCHAIN_CREATE_INFO };
        info.usageFlags = XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT | XR_SWAPCHAIN_USAGE_SAMPLED_BIT;
        info.format = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
        info.sampleCount = 1;
        info.width = kW;
        info.height = kH;
        info.faceCount = info.arraySize = 1;
        info.mipCount = 1;
        createSwapchain(session, &info, &app[v]);
    }
    XrCompositionLayerProjectionView views[2] = { { XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW },
                                                  { XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW } };
    for (int v = 0; v < 2; ++v)
    {
        views[v].pose.orientation.w = 1;
        views[v].fov = { -0.7853982f, 0.7853982f, 0.7853982f, -0.7853982f };
        views[v].subImage.swapchain = app[v];
        views[v].subImage.imageRect = { { 0, 0 }, { (int32_t) kW, (int32_t) kH } };
    }
    XrCompositionLayerProjection projection { XR_TYPE_COMPOSITION_LAYER_PROJECTION };
    projection.viewCount = 2;
    projection.views = views;
    const XrCompositionLayerBaseHeader* layers[1] = { reinterpret_cast<const XrCompositionLayerBaseHeader*>(&projection) };
    XrFrameEndInfo fei { XR_TYPE_FRAME_END_INFO };
    fei.layerCount = 1;
    fei.layers = layers;

    int failures = 0, checked = 0;
    for (uint64_t k = 1; k <= 8; ++k)
    {
        // The overlay's image k is drawn (slot k % 3) and visible for six frames, then closed.
        const bool panelVisible = k <= 6;
        setOverlayFrame(k, panelVisible ? 1 : 0);
        if (panelFence != nullptr)
            actx4->Signal(panelFence, k);
        actx->Flush();
        endFrame(session, &fei);
        bool ok = true;
        if (!panelVisible)
        {
            // Closed: the application's own frame, the very structure it handed in.
            ok = g_captured.layerCount == 1 && g_captured.quads == 0 && g_captured.info == &fei;
            std::printf("frame %llu: overlay closed: %u layers, %d quads, the application's frame %s -> %s\n", k,
                        g_captured.layerCount, g_captured.quads,
                        g_captured.info == &fei ? "handed on untouched" : "REPLACED", ok ? "OK" : "FAIL");
        }
        else
        {
            const auto& q = g_captured.quad;
            const uint64_t handle = (uint64_t) q.subImage.swapchain;
            const bool own = g_captured.quads == 1 && g_swapchains.count(handle) != 0 && handle != (uint64_t) app[0] &&
                             handle != (uint64_t) app[1];
            // The used part (the stand-in's: 16 and 8 less than the image) must hold the newest image exactly; the
            // panel is that part, and has its shape.
            const UINT usedW = kPanelW - 16, usedH = kPanelH - 8;
            int differing = -1;
            if (own && g_swapchains[handle].shown != nullptr)
            {
                const auto shown = ReadBack(g_swapchains[handle].shown);
                const auto& want = panelPattern[k % kOverlaySlots];
                differing = shown.size() == want.size() ? 0 : -1;
                for (UINT y = 0; differing >= 0 && y < usedH; ++y)
                    for (UINT x = 0; x < usedW * 4; ++x)
                        differing += shown[y * kPanelW * 4 + x] != want[y * kPanelW * 4 + x] ? 1 : 0;
            }
            const bool placed =
                own && g_captured.layerCount == 2 && (uint64_t) q.space == kStubSpace &&
                q.layerFlags == XR_COMPOSITION_LAYER_BLEND_TEXTURE_SOURCE_ALPHA_BIT &&
                q.eyeVisibility == XR_EYE_VISIBILITY_BOTH && q.subImage.imageRect.offset.x == 0 &&
                q.subImage.imageRect.offset.y == 0 && q.subImage.imageRect.extent.width == (int32_t) usedW &&
                q.subImage.imageRect.extent.height == (int32_t) usedH && std::fabs(q.size.width - 1.6f) < 1e-4f &&
                std::fabs(q.size.height - 1.6f * usedH / usedW) < 1e-4f && std::fabs(q.pose.position.z + 1.5f) < 1e-4f &&
                std::fabs(q.pose.position.y + 0.25f) < 1e-4f && std::fabs(q.pose.position.x) < 1e-6f &&
                std::fabs(q.pose.orientation.w - 1.0f) < 1e-6f;
            ok = placed && differing == 0;
            std::printf("frame %llu: overlay visible: %u layers, %d quads, %s swapchain, %.3f x %.3f m at %.2f %.2f "
                        "%.2f, flags 0x%llX, image %d values differing from slot %llu's -> %s\n",
                        k, g_captured.layerCount, g_captured.quads, own ? "the layer's own" : "NOT the layer's",
                        q.size.width, q.size.height, q.pose.position.x, q.pose.position.y, q.pose.position.z,
                        (unsigned long long) q.layerFlags, differing, k % kOverlaySlots, ok ? "OK" : "FAIL");
        }
        ++checked;
        if (!ok)
            ++failures;
    }
    destroySession(session);
    {
        // The layer asked at every frame, made one VIEW space and gave it back.
        const bool ok = overlayAsked() >= 8 && g_spacesCreated == 1 && g_spacesDestroyed == 1;
        std::printf("overlay: asked for %d times (must be >= 8), %d spaces created, %d destroyed (1 and 1) -> %s\n",
                    overlayAsked(), g_spacesCreated, g_spacesDestroyed, ok ? "OK" : "FAIL");
        ++checked;
        if (!ok)
            ++failures;
    }
    std::printf("%s: %d checks, %d failures\n", failures == 0 && checked > 0 ? "PASSED" : "FAILED", checked, failures);
    return failures == 0 && checked > 0 ? 0 : 1;
}
