#include "pch.h"
#include "OverlayShare.h"
#include "StaleEyeShareAbi.h"

#include "Config.h"
#include "Logger.h"
#include "resource.h" // VER_PRODUCT_NAME: the menu window's name begins with it

#include <imgui/imgui.h>
#include <imgui/imgui_impl_dx11.h>
#include <imgui/imgui_impl_dx12.h>

#include "OverlayCrop.h" // after imgui.h
#include "OverlayTextures.h"
#include "OverlayTextures12.h"

#include <d3d11_4.h>
#include <d3d12.h>
#include <dxgi1_2.h>

#include <algorithm>
#include <atomic>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <mutex>
#include <vector>

namespace
{
// What the layer reads. Written on the Present thread, read on the layer's (xrEndFrame): behind this lock only.
class Published
{
  public:
    void Set(const RvrToolsOverlayV1& value)
    {
        std::lock_guard lock(_mutex);
        _value = value;
    }
    RvrToolsOverlayV1 Get()
    {
        std::lock_guard lock(_mutex);
        return _value;
    }
    void Hide()
    {
        std::lock_guard lock(_mutex);
        _value.visible = 0;
    }

  private:
    std::mutex _mutex;
    RvrToolsOverlayV1 _value {};
};
Published g_published;
std::atomic<ULONGLONG> g_lastAsked { 0 }; // GetTickCount64 at the layer's last call; 0 = never

// ---- The Present thread's own state, both APIs (one of them runs in a process) ----
uint64_t g_present = 0, g_sequence = 0, g_generation = 0, g_drawn = 0;
int g_sizes = 0;
UINT g_candidate[2] = {};
uint64_t g_candidateSince = 0;
ULONGLONG g_candidateSinceMs = 0;
bool g_wasVisible = false;
// Without a fence (D3D11 only): the image drawn at the last Present, named to the layer at this one (by then it is done).
bool g_heldBack = false;
RvrToolsOverlayV1 g_held {};
// The overlay's own rectangle in the window (OverlayCrop.h): the menu's while the menu is up.
OverlayCrop::State g_crop;
// A size must have held this long before textures are made for it (FC5: work at a new size lost the device) -- in
// Presents AND in time: the first run counted its 90 Presents in 34 ms of a loading screen. A retired set is kept
// for kRetirePresents (the layer's copy of its last image may still be queued).
constexpr uint64_t kSettlePresents = 90, kRetirePresents = 600;
constexpr ULONGLONG kSettleMs = 1000;
// The textures: 4096 x 4096 from the first (64 MB each, three of them; never grown in steps -- see
// Decide). Never more: FC5's window is 8192 x 8192, and three of those plus the layer's three were 1.5 GB for a menu
// of a fraction of it.
constexpr int kMaxSide = 4096, kMargin = 16;

void Skip(int reason, const char* text)
{
    static std::atomic<uint32_t> logged { 0 };
    const uint32_t bit = 1u << reason;
    if ((logged.fetch_or(bit) & bit) == 0)
        LOG_INFO("Headset overlay: not drawn for the headset: {}", text);
}

void Hide()
{
    if (g_wasVisible)
        g_published.Hide();
    g_wasVisible = false;
    g_crop.valid = false; // the next time it is up, its rectangle starts anew
    g_heldBack = false;
}

// What one Present's draw for the headset needs, decided the same way for both APIs.
struct Plan
{
    UINT width = 0, height = 0;               // the window (ImGui's framebuffer)
    UINT usedWidth = 0, usedHeight = 0;       // the panel's part of the texture
    UINT textureWidth = 0, textureHeight = 0; // the textures to have (grown in steps, never smaller)
    int cropWidth = 0;                        // the overlay's rectangle in canvas units (the panel's width follows it)
    float fit = 1.0f;
    ImDrawData cropped {};                    // the draw lists seen through the rectangle
};

// Every Present: the size's settling, the layer asking, the overlay's rectangle. False = nothing is drawn for the
// headset this Present, and the panel is hidden. setWidth / setHeight: the current textures' size (0 = none yet).
bool Decide(ImDrawData* drawData, bool haveDevice, UINT setWidth, UINT setHeight, Plan& p)
{
    if (!haveDevice)
    {
        Hide();
        return false;
    }
    if (!Config::Instance()->HeadsetOverlay.value_or_default())
    {
        Skip(0, "[Inputs] HeadsetOverlay is off");
        Hide();
        return false;
    }
    // The size is followed at every Present, so that it is known to have held when something is to be drawn.
    p.width = drawData != nullptr ? (UINT) (drawData->DisplaySize.x * drawData->FramebufferScale.x) : 0;
    p.height = drawData != nullptr ? (UINT) (drawData->DisplaySize.y * drawData->FramebufferScale.y) : 0;
    if (drawData != nullptr && (p.width != g_candidate[0] || p.height != g_candidate[1]))
    {
        g_candidate[0] = p.width;
        g_candidate[1] = p.height;
        g_candidateSince = g_present;
        g_candidateSinceMs = GetTickCount64();
    }
    const ULONGLONG asked = g_lastAsked.load(std::memory_order_relaxed);
    if (asked == 0 || GetTickCount64() - asked > 1000)
    {
        Hide(); // no layer (every game without it, every Present): said by the layer's side, not here
        return false;
    }
    if (drawData == nullptr || drawData->TotalVtxCount <= 0)
    {
        Hide(); // the overlay is closed and no message is up
        return false;
    }
    if (p.width < 16 || p.height < 16)
    {
        Skip(1, "the game's window is under 16 pixels");
        Hide();
        return false;
    }
    if (g_present - g_candidateSince < kSettlePresents || GetTickCount64() - g_candidateSinceMs < kSettleMs)
    {
        Skip(2, "the game window's size has not held for 90 Presents and a second yet");
        Hide();
        return false;
    }
    // The overlay's own rectangle: the menu's while the menu is up, wherever it is on the canvas.
    const float scaleX = drawData->FramebufferScale.x > 0 ? drawData->FramebufferScale.x : 1.0f,
                scaleY = drawData->FramebufferScale.y > 0 ? drawData->FramebufferScale.y : 1.0f;
    OverlayCrop::Rules rules;
    rules.primary = VER_PRODUCT_NAME;
    p.fit = 1.0f;
    if (!OverlayCrop::Grow(drawData, g_crop, rules, kMargin, (int) (kMaxSide / (std::max)(scaleX, scaleY)), &p.fit))
    {
        Hide(); // vertices, but nothing of them inside their clip rectangles
        return false;
    }
    p.cropWidth = g_crop.crop[2] - g_crop.crop[0];
    const int cropHeight = g_crop.crop[3] - g_crop.crop[1];
    p.usedWidth = (UINT) (std::min)((float) kMaxSide, std::floor((float) p.cropWidth * scaleX * p.fit));
    p.usedHeight = (UINT) (std::min)((float) kMaxSide, std::floor((float) cropHeight * scaleY * p.fit));
    if (p.usedWidth < 8 || p.usedHeight < 8)
    {
        Hide();
        return false;
    }
    // The textures: full size (kMaxSide) from the first, so a menu that grows never remakes them -- nor the layer's
    // swapchain, which follows their size inside the VR session. In one D3D12 game the menu dragged
    // taller remade them (3584x4096, generation 4); the game stopped
    // presenting ~13 s after (one run).
    (void) setWidth;
    (void) setHeight;
    p.textureWidth = (UINT) kMaxSide;
    p.textureHeight = (UINT) kMaxSide;
    // The same draw lists, seen through the overlay's own rectangle: the renderer maps DisplayPos .. DisplayPos +
    // DisplaySize onto a viewport of that size at the target's top left corner, and clips to it.
    p.cropped = OverlayCrop::Through(drawData, g_crop.crop, p.fit);
    return true;
}

// The publication for one image, and its log line (the first image, one in 5400, the first twenty rectangles of a
// new size).
RvrToolsOverlayV1 Value(const Plan& p, const uint64_t handles[kOverlaySlots], UINT textureWidth, UINT textureHeight,
                           bool legacy, uint64_t fenceHandle, uint64_t sequence, float* inViewOut)
{
    auto* config = Config::Instance();
    LARGE_INTEGER now {};
    QueryPerformanceCounter(&now);
    RvrToolsOverlayV1 value {};
    value.structSize = sizeof(value);
    value.visible = 1;
    value.generation = g_generation;
    for (int i = 0; i < kOverlaySlots; ++i)
        value.handles[i] = handles[i];
    value.width = textureWidth;
    value.height = textureHeight;
    value.usedWidth = (std::min)(p.usedWidth, textureWidth);
    value.usedHeight = (std::min)(p.usedHeight, textureHeight);
    value.writtenFenceHandle = fenceHandle;
    value.legacyHandles = legacy ? 1u : 0u;
    value.sequence = sequence;
    value.qpc = now.QuadPart;
    // Every pixel of the overlay the same size in the headset (the setting is the width of 2048 of them, whatever
    // scale they were drawn at) -- but never a panel larger than the view takes in: about 70 by 55 degrees.
    value.panelDistance = std::clamp(config->HeadsetOverlayDistance.value_or_default(), 0.3f, 10.0f);
    value.panelDown = std::clamp(config->HeadsetOverlayDown.value_or_default(), -2.0f, 2.0f);
    float panelWidth =
        std::clamp(config->HeadsetOverlayWidth.value_or_default(), 0.2f, 5.0f) * (float) p.cropWidth / 2048.0f;
    const float panelHeight = panelWidth * (float) value.usedHeight / (float) value.usedWidth;
    const float inView = (std::min)({ 1.0f, 1.4f * value.panelDistance / panelWidth,
                                      1.04f * value.panelDistance / panelHeight });
    panelWidth *= inView;
    value.panelWidth = panelWidth;
    *inViewOut = inView;
    return value;
}

void LogDrawn(const Plan& p, const RvrToolsOverlayV1& value, UINT textureWidth, UINT textureHeight, int vertices,
              float inView, const char* api)
{
    static UINT saidUsed[2] = {};
    static int rectanglesSaid = 0;
    const bool newRectangle =
        (saidUsed[0] != value.usedWidth || saidUsed[1] != value.usedHeight) && rectanglesSaid < 20;
    if (++g_drawn == 1 || g_drawn % 5400 == 0 || newRectangle)
    {
        if (newRectangle)
            ++rectanglesSaid;
        saidUsed[0] = value.usedWidth;
        saidUsed[1] = value.usedHeight;
        const int cropHeight = g_crop.crop[3] - g_crop.crop[1];
        LOG_INFO("Headset overlay ({}): {} images drawn for the headset (sequence {}; {}: {}x{} at {},{} of the {}x{} "
                 "window, drawn at {:.2f} as {}x{} in {}x{} textures; {} vertices; panel {:.2f} m wide{}, {:.2f} m "
                 "away, {:.2f} m down)",
                 api, g_drawn, value.sequence, g_crop.menu ? "the MENU's rectangle" : "no menu, everything drawn",
                 p.cropWidth, cropHeight, g_crop.crop[0], g_crop.crop[1], p.width, p.height, p.fit, value.usedWidth,
                 value.usedHeight, textureWidth, textureHeight, vertices, value.panelWidth,
                 inView < 1.0f ? " (made smaller to stay in view)" : "", value.panelDistance, value.panelDown);
    }
}

// ---- D3D11 ----
struct TextureSet
{
    ID3D11Texture2D* textures[kOverlaySlots] = {};
    ID3D11RenderTargetView* rtvs[kOverlaySlots] = {};
    HANDLE handles[kOverlaySlots] = {};
    UINT width = 0, height = 0;
    bool legacy = false; // the handles are the old kind: not closed, they end with their textures
};
struct Retired
{
    TextureSet set;
    uint64_t present = 0;
};
TextureSet g_set;
std::vector<Retired> g_retired;
ID3D11Fence* g_fence = nullptr;
HANDLE g_fenceHandle = nullptr;
const void* g_deviceIdentity = nullptr; // compared only
bool g_setupTried = false, g_ready = false;

void ReleaseSet(TextureSet& set)
{
    for (int i = 0; i < kOverlaySlots; ++i)
    {
        if (set.rtvs[i] != nullptr)
            set.rtvs[i]->Release();
        if (set.textures[i] != nullptr)
            set.textures[i]->Release();
        if (set.handles[i] != nullptr && !set.legacy)
            CloseHandle(set.handles[i]);
    }
    set = {};
}

void Retire()
{
    if (g_set.textures[0] != nullptr)
        g_retired.push_back({ g_set, g_present });
    g_set = {};
}

void Setup(ID3D11Device* device)
{
    g_setupTried = true;
    g_ready = true;
    ID3D11Device5* device5 = nullptr;
    const HRESULT asked = device->QueryInterface(IID_PPV_ARGS(&device5));
    if (SUCCEEDED(asked) && device5 != nullptr)
    {
        const bool ok = SUCCEEDED(device5->CreateFence(0, D3D11_FENCE_FLAG_SHARED, IID_PPV_ARGS(&g_fence))) &&
                        SUCCEEDED(g_fence->CreateSharedHandle(nullptr, GENERIC_ALL, nullptr, &g_fenceHandle));
        device5->Release();
        if (ok)
        {
            LOG_INFO("Headset overlay: ready (fence {:X})", (size_t) g_fenceHandle);
            return;
        }
        if (g_fence != nullptr)
            g_fence->Release();
        g_fence = nullptr;
        g_fenceHandle = nullptr;
    }
    // Some games' devices are (the DX11-to-DX12 bridge failed on the same question).
    LOG_INFO("Headset overlay: ready WITHOUT a fence (the game's device has no ID3D11Device5 or made none: {:X}): "
             "each image is named to the layer one Present after it was drawn",
             (unsigned long) asked);
}

bool EnsureSize(ID3D11Device* device, UINT width, UINT height)
{
    if (g_set.width == width && g_set.height == height && g_set.textures[0] != nullptr)
        return true;
    if (g_sizes >= 24) // the sides only grow, in steps of 512 up to 4096: 16 sizes at most on one device
    {
        Skip(5, "more than 24 sizes of shared textures; no new ones (restart to reset)");
        return false;
    }
    ++g_sizes;
    g_published.Hide(); // the layer must not take a retired set for the current one
    Retire();
    TextureSet made;
    static bool legacyOnly = false; // once NT handles have failed on this device they are not tried again
    for (int attempt = legacyOnly ? 1 : 0; attempt < 2; ++attempt)
    {
        made.legacy = attempt == 1;
        bool ok = true;
        for (int i = 0; ok && i < kOverlaySlots; ++i)
            ok = OverlayTextures::MakeShared(device, width, height, made.legacy, &made.textures[i], &made.rtvs[i],
                                             &made.handles[i]);
        if (ok)
            break;
        ReleaseSet(made); // nothing was drawn into them and no handle was given out
        if (attempt == 0)
        {
            legacyOnly = true;
            LOG_INFO("Headset overlay: the game's device makes no NT shared handles for {}x{} textures; trying the old "
                     "kind of shared handle",
                     width, height);
            continue;
        }
        LOG_ERROR("Headset overlay: shared textures ({}x{}) failed with both kinds of handle; off for this size", width,
                  height);
        return false;
    }
    made.width = width;
    made.height = height;
    g_set = made;
    ++g_generation;
    LOG_INFO("Headset overlay: {} x {}x{} shared textures ({} handles), generation {}", kOverlaySlots, width, height,
             g_set.legacy ? "OLD-kind" : "NT", g_generation);
    return true;
}

// ---- D3D12 ----
struct TextureSet12
{
    ID3D12Resource* textures[kOverlaySlots] = {};
    HANDLE handles[kOverlaySlots] = {};
    ID3D12DescriptorHeap* rtvHeap = nullptr; // kOverlaySlots render target views
    D3D12_CPU_DESCRIPTOR_HANDLE rtvs[kOverlaySlots] = {};
    UINT width = 0, height = 0;
};
struct Retired12
{
    TextureSet12 set;
    uint64_t present = 0;
};
TextureSet12 g_set12;
std::vector<Retired12> g_retired12;
ID3D12Fence* g_fence12 = nullptr;
HANDLE g_fenceHandle12 = nullptr;
const void* g_device12 = nullptr; // compared only
bool g_setupTried12 = false, g_ready12 = false;
// Recorded this Present, signalled and published after the list's execute (AfterExecuteDx12).
bool g_pending12 = false;
uint64_t g_pendingSequence12 = 0;
RvrToolsOverlayV1 g_pendingValue12 {};

void ReleaseSet12(TextureSet12& set)
{
    for (int i = 0; i < kOverlaySlots; ++i)
    {
        if (set.textures[i] != nullptr)
            set.textures[i]->Release();
        if (set.handles[i] != nullptr)
            CloseHandle(set.handles[i]);
    }
    if (set.rtvHeap != nullptr)
        set.rtvHeap->Release();
    set = {};
}

void Retire12()
{
    if (g_set12.textures[0] != nullptr)
        g_retired12.push_back({ g_set12, g_present });
    g_set12 = {};
}

void Setup12(ID3D12Device* device)
{
    g_setupTried12 = true;
    HRESULT hr = S_OK;
    g_ready12 = OverlayTextures12::MakeSharedFence(device, &g_fence12, &g_fenceHandle12, &hr);
    if (g_ready12)
        LOG_INFO("Headset overlay (D3D12): ready (shared fence {:X})", (size_t) g_fenceHandle12);
    else
        LOG_ERROR("Headset overlay (D3D12): no shared fence ({:X}); the panel is off", (unsigned long) hr);
}

bool EnsureSize12(ID3D12Device* device, UINT width, UINT height, DXGI_FORMAT rtvFormat)
{
    if (g_set12.width == width && g_set12.height == height && g_set12.textures[0] != nullptr)
        return true;
    if (g_sizes >= 24)
    {
        Skip(5, "more than 24 sizes of shared textures; no new ones (restart to reset)");
        return false;
    }
    ++g_sizes;
    g_published.Hide();
    Retire12();
    TextureSet12 made;
    D3D12_DESCRIPTOR_HEAP_DESC heap {};
    heap.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    heap.NumDescriptors = kOverlaySlots;
    HRESULT hr = device->CreateDescriptorHeap(&heap, IID_PPV_ARGS(&made.rtvHeap));
    bool ok = SUCCEEDED(hr) && made.rtvHeap != nullptr;
    if (ok)
    {
        const UINT step = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
        D3D12_CPU_DESCRIPTOR_HANDLE at = made.rtvHeap->GetCPUDescriptorHandleForHeapStart();
        for (int i = 0; ok && i < kOverlaySlots; ++i)
        {
            made.rtvs[i] = at;
            ok = OverlayTextures12::MakeShared(device, width, height, rtvFormat, at, &made.textures[i], &made.handles[i],
                                               &hr);
            at.ptr += step;
        }
    }
    if (!ok)
    {
        ReleaseSet12(made);
        LOG_ERROR("Headset overlay (D3D12): shared textures ({}x{}) failed ({:X}); off for this size", width, height,
                  (unsigned long) hr);
        return false;
    }
    made.width = width;
    made.height = height;
    g_set12 = made;
    ++g_generation;
    LOG_INFO("Headset overlay (D3D12): {} x {}x{} shared textures (NT handles, typeless format {}, views of the window's "
             "format {}), generation {}",
             kOverlaySlots, width, height, (int) OverlayTextures12::TypelessFor(rtvFormat), (int) rtvFormat, g_generation);
    return true;
}
} // namespace

void OverlayShare::AfterDraw(ID3D11Device* device, ID3D11DeviceContext* context, ImDrawData* drawData)
{
    ++g_present;
    for (size_t i = 0; i < g_retired.size();)
    {
        if (g_present - g_retired[i].present >= kRetirePresents)
        {
            ReleaseSet(g_retired[i].set);
            g_retired[i] = g_retired.back();
            g_retired.pop_back();
        }
        else
        {
            ++i;
        }
    }
    Plan p;
    if (!Decide(drawData, device != nullptr && context != nullptr, g_set.width, g_set.height, p))
        return;
    if (g_deviceIdentity != device)
    {
        // Another device: everything made on the old one is retired, the fence included.
        if (g_deviceIdentity != nullptr)
            LOG_INFO("Headset overlay: the game's device changed; the shared textures are made again");
        g_deviceIdentity = device;
        g_published.Hide();
        Retire();
        if (g_fence != nullptr)
            g_fence->Release();
        g_fence = nullptr;
        if (g_fenceHandle != nullptr)
            CloseHandle(g_fenceHandle);
        g_fenceHandle = nullptr;
        g_setupTried = g_ready = false;
    }
    if (!g_setupTried)
        Setup(device);
    if (!g_ready || !EnsureSize(device, p.textureWidth, p.textureHeight))
        return Hide();
    ID3D11DeviceContext4* context4 = nullptr;
    if (g_fence != nullptr && (FAILED(context->QueryInterface(IID_PPV_ARGS(&context4))) || context4 == nullptr))
    {
        Skip(3, "a fence, but no ID3D11DeviceContext4 to signal it with");
        return Hide();
    }

    const uint64_t sequence = ++g_sequence;
    ID3D11RenderTargetView* target = g_set.rtvs[sequence % kOverlaySlots];
    // The flat window's target is bound here (the overlay has just been drawn onto it) and must be again after.
    ID3D11RenderTargetView* before[D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT] = {};
    ID3D11DepthStencilView* depthBefore = nullptr;
    context->OMGetRenderTargets(D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT, before, &depthBefore);
    const float transparent[4] = {};
    context->OMSetRenderTargets(1, &target, nullptr);
    context->ClearRenderTargetView(target, transparent);
    ImGui_ImplDX11_RenderDrawData(&p.cropped);
    context->OMSetRenderTargets(D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT, before, depthBefore);
    for (auto* view : before)
        if (view != nullptr)
            view->Release();
    if (depthBefore != nullptr)
        depthBefore->Release();
    if (context4 != nullptr)
    {
        context4->Signal(g_fence, sequence);
        context4->Release();
    }

    uint64_t handles[kOverlaySlots];
    for (int i = 0; i < kOverlaySlots; ++i)
        handles[i] = reinterpret_cast<uint64_t>(g_set.handles[i]);
    float inView = 1.0f;
    const RvrToolsOverlayV1 value = Value(p, handles, g_set.width, g_set.height, g_set.legacy,
                                             reinterpret_cast<uint64_t>(g_fenceHandle), sequence, &inView);
    if (g_fence != nullptr)
    {
        g_published.Set(value);
        g_wasVisible = true;
    }
    else
    {
        // No fence: what the layer is given now is the image of the LAST Present (of the same set of textures),
        // whose draw this Present's own flush has long sent; this one waits its turn.
        if (g_heldBack && g_held.generation == value.generation)
        {
            g_held.qpc = value.qpc; // it is what is on show now
            g_held.panelWidth = value.panelWidth;
            g_held.panelDistance = value.panelDistance;
            g_held.panelDown = value.panelDown;
            g_published.Set(g_held);
            g_wasVisible = true;
        }
        g_held = value;
        g_heldBack = true;
    }
    LogDrawn(p, value, g_set.width, g_set.height, drawData->TotalVtxCount, inView, "D3D11");
}

void OverlayShare::AfterDrawDx12(ID3D12Device* device, ID3D12GraphicsCommandList* list, DXGI_FORMAT rtvFormat,
                                 D3D12_CPU_DESCRIPTOR_HANDLE backBuffer, ImDrawData* drawData)
{
    ++g_present;
    g_pending12 = false;
    for (size_t i = 0; i < g_retired12.size();)
    {
        // 600 Presents on: nothing of the game's or the layer's still reads them.
        if (g_present - g_retired12[i].present >= kRetirePresents)
        {
            ReleaseSet12(g_retired12[i].set);
            g_retired12[i] = g_retired12.back();
            g_retired12.pop_back();
        }
        else
        {
            ++i;
        }
    }
    Plan p;
    if (!Decide(drawData, device != nullptr && list != nullptr, g_set12.width, g_set12.height, p))
        return;
    if (!OverlayTextures12::RtvFormatOk(rtvFormat))
    {
        // ImGui's one D3D12 pipeline draws in the window's format, and the layer copies the texture into a panel of the
        // same family: RGBA8, BGRA8 or 10-bit. Another window format would need a pipeline of its own: not built.
        static bool said = false;
        if (!said)
        {
            said = true;
            LOG_INFO("Headset overlay (D3D12): not drawn for the headset: the window's swapchain format is {} -- not RGBA8, "
                     "BGRA8 or R10G10B10A2, the families the panel can take (ImGui's D3D12 pipeline draws in the window's "
                     "format; another needs a pipeline of its own, not built)",
                     (int) rtvFormat);
        }
        return Hide();
    }
    if (g_device12 != device)
    {
        if (g_device12 != nullptr)
            LOG_INFO("Headset overlay (D3D12): the game's device changed; the shared textures are made again");
        g_device12 = device;
        g_published.Hide();
        Retire12();
        if (g_fence12 != nullptr)
            g_fence12->Release();
        g_fence12 = nullptr;
        if (g_fenceHandle12 != nullptr)
            CloseHandle(g_fenceHandle12);
        g_fenceHandle12 = nullptr;
        g_setupTried12 = g_ready12 = false;
    }
    if (!g_setupTried12)
        Setup12(device);
    if (!g_ready12 || !EnsureSize12(device, p.textureWidth, p.textureHeight, rtvFormat))
        return Hide();

    const uint64_t sequence = ++g_sequence;
    const int slot = (int) (sequence % kOverlaySlots);
    // Into our texture: COMMON (where cross-API sharing leaves it) -> render target -> COMMON; then the back buffer
    // is the list's target again, as the menu's code left it.
    D3D12_RESOURCE_BARRIER barrier {};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = g_set12.textures[slot];
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_RENDER_TARGET;
    list->ResourceBarrier(1, &barrier);
    const float transparent[4] = {};
    list->OMSetRenderTargets(1, &g_set12.rtvs[slot], FALSE, nullptr);
    list->ClearRenderTargetView(g_set12.rtvs[slot], transparent, 0, nullptr);
    ImGui_ImplDX12_RenderDrawData(&p.cropped, list);
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COMMON;
    list->ResourceBarrier(1, &barrier);
    list->OMSetRenderTargets(1, &backBuffer, FALSE, nullptr);

    uint64_t handles[kOverlaySlots];
    for (int i = 0; i < kOverlaySlots; ++i)
        handles[i] = reinterpret_cast<uint64_t>(g_set12.handles[i]);
    float inView = 1.0f;
    g_pendingValue12 = Value(p, handles, g_set12.width, g_set12.height, false,
                             reinterpret_cast<uint64_t>(g_fenceHandle12), sequence, &inView);
    g_pendingSequence12 = sequence;
    g_pending12 = true; // published once the fence's signal is queued behind the list (AfterExecuteDx12)
    LogDrawn(p, g_pendingValue12, g_set12.width, g_set12.height, drawData->TotalVtxCount, inView, "D3D12");
}

void OverlayShare::AfterExecuteDx12(ID3D12CommandQueue* queue)
{
    if (!g_pending12)
        return;
    g_pending12 = false;
    if (queue == nullptr || g_fence12 == nullptr)
    {
        Skip(7, "no command queue to signal the shared fence on");
        return Hide();
    }
    queue->Signal(g_fence12, g_pendingSequence12);
    g_published.Set(g_pendingValue12);
    g_wasVisible = true;
}

bool OverlayShare::LayerAsks()
{
    const ULONGLONG asked = g_lastAsked.load(std::memory_order_relaxed);
    return asked != 0 && GetTickCount64() - asked <= 1000;
}

bool OverlayShare::Latest(RvrToolsOverlayV1* out)
{
    if (out == nullptr)
        return false;
    *out = g_published.Get();
    return out->generation != 0;
}

extern "C" __declspec(dllexport) int __cdecl RvrTools_GetOverlayV1(RvrToolsOverlayV1* out)
{
    if (out == nullptr || out->structSize != sizeof(RvrToolsOverlayV1))
        return 0;
    if (g_lastAsked.exchange(GetTickCount64(), std::memory_order_relaxed) == 0)
        LOG_INFO("Headset overlay: the layer asks for the overlay");
    *out = g_published.Get();
    out->structSize = sizeof(RvrToolsOverlayV1);
    return 1;
}
