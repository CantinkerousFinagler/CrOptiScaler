#include "pch.h"
#include "Fc5Trace_Dx11.h"

#include "Config.h"
#include "State.h"
#include "inputs/Fc5Dlss_Dx11.h"
#include "inputs/Fc5Taa_Dx11.h"
#include "inputs/EyeImageNr_Dx11.h"
#include "inputs/FinalImageNr_Dx11.h"
#include "inputs/FrameCapture_Dx11.h"
#include "inputs/TaaMotion_Dx11.h"
#include "inputs/TaaDlaa_Dx11.h"
#include "inputs/MotionMatrixProbe_Dx11.h"

#include <detours/detours.h>

#include <atomic>
#include <cstdint>
#include <format>
#include <string>

// Recording order on the CPU. For D3D11's immediate context that is also submission order; deferred contexts
// record elsewhere and run at ExecuteCommandList. Deferred contexts can use their own function table (some
// games' do), so when they differ the same hooks are attached to it separately: K selects the originals to call.
namespace
{
using PSSetShaderResources_t = void(STDMETHODCALLTYPE*)(ID3D11DeviceContext*, UINT, UINT,
                                                         ID3D11ShaderResourceView* const*);
using DrawIndexed_t = void(STDMETHODCALLTYPE*)(ID3D11DeviceContext*, UINT, UINT, INT);
using Draw_t = void(STDMETHODCALLTYPE*)(ID3D11DeviceContext*, UINT, UINT);
using DrawIndexedInstanced_t = void(STDMETHODCALLTYPE*)(ID3D11DeviceContext*, UINT, UINT, UINT, INT, UINT);
using DrawInstanced_t = void(STDMETHODCALLTYPE*)(ID3D11DeviceContext*, UINT, UINT, UINT, UINT);
using OMSetRenderTargets_t = void(STDMETHODCALLTYPE*)(ID3D11DeviceContext*, UINT, ID3D11RenderTargetView* const*,
                                                       ID3D11DepthStencilView*);
using OMSetRenderTargetsAndUAVs_t =
    void(STDMETHODCALLTYPE*)(ID3D11DeviceContext*, UINT, ID3D11RenderTargetView* const*, ID3D11DepthStencilView*,
                             UINT, UINT, ID3D11UnorderedAccessView* const*, const UINT*);
using Dispatch_t = void(STDMETHODCALLTYPE*)(ID3D11DeviceContext*, UINT, UINT, UINT);
using RSSetViewports_t = void(STDMETHODCALLTYPE*)(ID3D11DeviceContext*, UINT, const D3D11_VIEWPORT*);
using RSSetScissorRects_t = void(STDMETHODCALLTYPE*)(ID3D11DeviceContext*, UINT, const D3D11_RECT*);
using ClearDepthStencilView_t = void(STDMETHODCALLTYPE*)(ID3D11DeviceContext*, ID3D11DepthStencilView*, UINT, FLOAT,
                                                          UINT8);
using CopySubresourceRegion_t = void(STDMETHODCALLTYPE*)(ID3D11DeviceContext*, ID3D11Resource*, UINT, UINT, UINT,
                                                          UINT, ID3D11Resource*, UINT, const D3D11_BOX*);
using CopyResource_t = void(STDMETHODCALLTYPE*)(ID3D11DeviceContext*, ID3D11Resource*, ID3D11Resource*);
using Map_t = HRESULT(STDMETHODCALLTYPE*)(ID3D11DeviceContext*, ID3D11Resource*, UINT, D3D11_MAP, UINT,
                                          D3D11_MAPPED_SUBRESOURCE*);
using Unmap_t = void(STDMETHODCALLTYPE*)(ID3D11DeviceContext*, ID3D11Resource*, UINT);
using UpdateSubresource_t = void(STDMETHODCALLTYPE*)(ID3D11DeviceContext*, ID3D11Resource*, UINT, const D3D11_BOX*,
                                                      const void*, UINT, UINT);
using ResolveSubresource_t = void(STDMETHODCALLTYPE*)(ID3D11DeviceContext*, ID3D11Resource*, UINT, ID3D11Resource*,
                                                       UINT, DXGI_FORMAT);
using ExecuteCommandList_t = void(STDMETHODCALLTYPE*)(ID3D11DeviceContext*, ID3D11CommandList*, BOOL);
using CSSetUnorderedAccessViews_t = void(STDMETHODCALLTYPE*)(ID3D11DeviceContext*, UINT, UINT,
                                                              ID3D11UnorderedAccessView* const*);
using FinishCommandList_t = HRESULT(STDMETHODCALLTYPE*)(ID3D11DeviceContext*, BOOL, ID3D11CommandList**);
using CreateDeferredContext_t = HRESULT(STDMETHODCALLTYPE*)(ID3D11Device*, UINT, ID3D11DeviceContext**);
using CreateShader_t = HRESULT(STDMETHODCALLTYPE*)(ID3D11Device*, const void*, SIZE_T, ID3D11ClassLinkage*,
                                                   void**);
using CreateSamplerState_t = HRESULT(STDMETHODCALLTYPE*)(ID3D11Device*, const D3D11_SAMPLER_DESC*,
                                                         ID3D11SamplerState**);

// ID3D11DeviceContext / ID3D11Device vtable slots (checked against the SDK's d3d11.h).
constexpr int kPSSetShaderResources = 8;
constexpr int kDrawIndexed = 12;
constexpr int kDraw = 13;
constexpr int kDrawIndexedInstanced = 20;
constexpr int kDrawInstanced = 21;
constexpr int kOMSetRenderTargets = 33;
constexpr int kOMSetRenderTargetsAndUAVs = 34;
constexpr int kDispatch = 41;
constexpr int kRSSetViewports = 44;
constexpr int kRSSetScissorRects = 45;
constexpr int kCopySubresourceRegion = 46;
constexpr int kCopyResource = 47;
constexpr int kMap = 14;
constexpr int kUnmap = 15;
constexpr int kUpdateSubresource = 48;
constexpr int kClearDepthStencilView = 53;
constexpr int kResolveSubresource = 57;
constexpr int kExecuteCommandList = 58;
constexpr int kCSSetShaderResources = 67;
constexpr int kCSSetUnorderedAccessViews = 68;
constexpr int kFinishCommandList = 114;
constexpr int kCreateDeferredContext = 27;
constexpr int kCreateVertexShader = 12;
constexpr int kCreatePixelShader = 15;
constexpr int kCreateComputeShader = 18;
constexpr int kCreateSamplerState = 23;
constexpr int kSlots = 115;

constexpr int kImmediate = 0;
constexpr int kDeferred = 1;

// Originals per function table: [kImmediate] for the immediate context, [kDeferred] for deferred contexts
// when theirs differs.
PVOID g_orig[2][kSlots] {};
PVOID o_CreateDeferredContext = nullptr;
PVOID o_CreateVertexShader = nullptr;
PVOID o_CreatePixelShader = nullptr;
PVOID o_CreateComputeShader = nullptr;

template <typename T> T Orig(int k, int slot) { return (T) g_orig[k][slot]; }

PVOID g_immediateVtable33 = nullptr;
bool g_installed = false;
// EyeImageNr_Dx11 is on: deferred binds, draws and finished lists feed it, immediate ExecuteCommandList runs it.
bool g_eyeNr = false;
// FinalImageNr_Dx11 is on: immediate binds, draws, compute UAVs and dispatches feed it.
bool g_finalNr = false;
// Diagnostics on: FrameCapture_Dx11 watches immediate dispatches.
bool g_capture = false;
// TaaMotion_Dx11 is on: it decodes motion at the TAA dispatch.
bool g_taaMotion = false;
// TaaDlaa_Dx11 is on: it can replace the TAA dispatch's output with DLAA.
bool g_taaDlaa = false;
// [Inputs] TaaDlaaSkipGameTaa: DLAA runs in place of the game's TAA dispatch instead of after it.
bool g_taaDlaaSkip = false;
// Fc5Taa_Dx11 is on (FC5): it tracks the G-buffer depth at binds and replaces the TAA draw's output.
bool g_fc5Taa = false;
// MotionMatrixProbe_Dx11 is on: binds, draws and constant buffer writes on every context feed it (it acts only
// during an F7 capture).
bool g_matrixProbe = false;
std::atomic<bool> g_deferredHooked { false };

constexpr int kMaxLines = 60000;
constexpr ULONGLONG kArmDelayMs = 3000;

std::atomic<uint64_t> g_present { 0 };
std::atomic<bool> g_tracing { false };
std::atomic<int> g_lines { 0 };
std::atomic<int> g_traceGeneration { 0 };
std::atomic<int> g_deferredCreated { 0 };
uint64_t g_traceEnd = 0;
ULONGLONG g_armAt = 0;

// Draws are counted per thread: deferred contexts record on several threads at once.
thread_local int t_draws = 0;
thread_local int t_drawsGeneration = -1;

bool Tracing() { return g_tracing.load(std::memory_order_relaxed); }

// True when this line may be written; counts it against the cap.
bool TakeLine()
{
    const int line = g_lines.fetch_add(1);
    if (line == kMaxLines)
        LOG_WARN("FC5 trace: line cap {} reached, the rest of this trace is dropped", kMaxLines);
    return line < kMaxLines;
}

void CountDraw()
{
    const int generation = g_traceGeneration.load(std::memory_order_relaxed);
    if (t_drawsGeneration != generation)
    {
        t_drawsGeneration = generation;
        t_draws = 0;
    }
    ++t_draws;
}

std::string DrawsSince()
{
    if (t_drawsGeneration != g_traceGeneration.load(std::memory_order_relaxed))
        return {};
    const int draws = t_draws;
    t_draws = 0;
    return draws > 0 ? std::format(" after {} draws", draws) : std::string {};
}

const char* CtxKind(int k) { return k == kDeferred ? "deferred " : ""; }

// Deferred contexts that share the immediate context's functions arrive through the kImmediate hooks.
template <int K> bool IsDeferred(ID3D11DeviceContext* ctx)
{
    return K == kDeferred || ctx->GetType() == D3D11_DEVICE_CONTEXT_DEFERRED;
}

// Our own passes (motion decode, DLAA and the D3D11 bridge's work inside it) aren't the game's: backends ignore them.
bool OwnPass()
{
    return TaaMotion_Dx11::InPass() || TaaDlaa_Dx11::InPass() || Fc5Taa_Dx11::InPass();
}

template <int K> void CountDrawFor(ID3D11DeviceContext* ctx)
{
    if (g_eyeNr)
    {
        if (IsDeferred<K>(ctx))
            EyeImageNr_Dx11::OnDeferredDraw(ctx);
        else
            EyeImageNr_Dx11::OnImmediateDraw(ctx);
    }
    if (g_finalNr && K == kImmediate && !OwnPass())
        FinalImageNr_Dx11::OnDraw(ctx);
    if (g_taaMotion && K == kImmediate && !OwnPass())
        TaaMotion_Dx11::BeforeDraw(ctx);
    if (g_capture && K == kImmediate && !OwnPass())
        FrameCapture_Dx11::BeforeDraw(ctx);
    if (g_matrixProbe && !OwnPass())
        MotionMatrixProbe_Dx11::OnDraw(ctx);
    if (Tracing())
        CountDraw();
}

// The game's upscale draw at a Render Scale below 100% is replaced while DLSS upscaling runs: skip the draw.
template <int K> bool ReplaceDrawFor(ID3D11DeviceContext* ctx)
{
    if (K != kImmediate || OwnPass())
        return false;
    // FC5: the per-eye DLAA writes the TAA output in place of the game's TAA draw.
    if (g_fc5Taa && Fc5Taa_Dx11::ReplaceDraw(ctx))
        return true;
    return g_taaDlaa && TaaDlaa_Dx11::ReplaceDraw(ctx);
}

// After the game's draw: lets the frame capture copy the draw's output.
template <int K> void AfterDrawFor(ID3D11DeviceContext* ctx)
{
    // Before the capture, so it sees the image that is shown.
    if (g_fc5Taa && K == kImmediate && !OwnPass())
        Fc5Taa_Dx11::AfterDraw(ctx);
    if (g_capture && K == kImmediate && !OwnPass())
        FrameCapture_Dx11::AfterDraw(ctx);
}

std::string DescResource(ID3D11Resource* resource)
{
    if (resource == nullptr)
        return "none";
    D3D11_RESOURCE_DIMENSION dim = D3D11_RESOURCE_DIMENSION_UNKNOWN;
    resource->GetType(&dim);
    if (dim != D3D11_RESOURCE_DIMENSION_TEXTURE2D)
        return std::format("{:X} dim {}", (size_t) resource, (int) dim);
    D3D11_TEXTURE2D_DESC desc {};
    static_cast<ID3D11Texture2D*>(resource)->GetDesc(&desc);
    std::string text = std::format("{:X} {}x{} fmt {} bind {:X}", (size_t) resource, desc.Width, desc.Height,
                                   (UINT) desc.Format, desc.BindFlags);
    if (desc.ArraySize > 1)
        text += std::format(" array {}", desc.ArraySize);
    if (desc.MipLevels > 1)
        text += std::format(" mips {}", desc.MipLevels);
    if (desc.SampleDesc.Count > 1)
        text += std::format(" msaa {}", desc.SampleDesc.Count);
    return text;
}

// Texture2D at least 256 on both sides; the scene, post and eye targets, not shadow cascades' small bits
// or lookup tables.
bool IsLarge(ID3D11Resource* resource)
{
    if (resource == nullptr)
        return false;
    D3D11_RESOURCE_DIMENSION dim = D3D11_RESOURCE_DIMENSION_UNKNOWN;
    resource->GetType(&dim);
    if (dim != D3D11_RESOURCE_DIMENSION_TEXTURE2D)
        return false;
    D3D11_TEXTURE2D_DESC desc {};
    static_cast<ID3D11Texture2D*>(resource)->GetDesc(&desc);
    return desc.Width >= 256 && desc.Height >= 256;
}

std::string DescView(ID3D11View* view)
{
    if (view == nullptr)
        return "none";
    ID3D11Resource* resource = nullptr;
    view->GetResource(&resource);
    std::string text = DescResource(resource);
    if (resource != nullptr)
        resource->Release();
    return text;
}

std::string DescRtv(ID3D11RenderTargetView* rtv)
{
    if (rtv == nullptr)
        return "none";
    D3D11_RENDER_TARGET_VIEW_DESC desc {};
    rtv->GetDesc(&desc);
    std::string text = DescView(rtv);
    if (desc.ViewDimension == D3D11_RTV_DIMENSION_TEXTURE2DARRAY)
        text += std::format(" slice {}", desc.Texture2DArray.FirstArraySlice);
    else if (desc.ViewDimension == D3D11_RTV_DIMENSION_TEXTURE2D && desc.Texture2D.MipSlice != 0)
        text += std::format(" mip {}", desc.Texture2D.MipSlice);
    return text;
}

// Large textures only; an unbind or a set of small inputs is not logged.
void LogSrvs(int k, const char* stage, ID3D11DeviceContext* ctx, UINT start, UINT count,
             ID3D11ShaderResourceView* const* views)
{
    if (views == nullptr)
        return;
    for (UINT i = 0; i < count; ++i)
    {
        if (views[i] == nullptr)
            continue;
        ID3D11Resource* resource = nullptr;
        views[i]->GetResource(&resource);
        const bool large = IsLarge(resource);
        if (large && TakeLine())
        {
            LOG_INFO("FC5 trace: present {} {}ctx {:X} {} srv t{} {}", g_present.load(), CtxKind(k), (size_t) ctx,
                     stage, start + i, DescResource(resource));
        }
        if (resource != nullptr)
            resource->Release();
    }
}

void LogTargets(int k, const char* kind, ID3D11DeviceContext* ctx, UINT numRtvs,
                ID3D11RenderTargetView* const* rtvs, ID3D11DepthStencilView* dsv)
{
    if (!TakeLine())
        return;
    std::string colors;
    for (UINT i = 0; i < numRtvs && rtvs != nullptr; ++i)
    {
        if (rtvs[i] == nullptr)
            continue;
        colors += std::format(" rt{} [{}]", i, DescRtv(rtvs[i]));
    }
    if (colors.empty())
        colors = " no color";

    D3D11_VIEWPORT viewport {};
    UINT numViewports = 1;
    ctx->RSGetViewports(&numViewports, &viewport);
    LOG_INFO("FC5 trace: present {} {}ctx {:X} thread {} {}{}{} depth [{}] (viewport before bind {}x{} at {},{})",
             g_present.load(), CtxKind(k), (size_t) ctx, GetCurrentThreadId(), kind, DrawsSince(), colors,
             DescView(dsv), viewport.Width, viewport.Height, viewport.TopLeftX, viewport.TopLeftY);
}

template <int K>
void STDMETHODCALLTYPE hkPSSetShaderResources(ID3D11DeviceContext* ctx, UINT start, UINT count,
                                              ID3D11ShaderResourceView* const* views)
{
    if (Tracing())
        LogSrvs(K, "ps", ctx, start, count, views);
    Orig<PSSetShaderResources_t>(K, kPSSetShaderResources)(ctx, start, count, views);
}

template <int K>
void STDMETHODCALLTYPE hkCSSetShaderResources(ID3D11DeviceContext* ctx, UINT start, UINT count,
                                              ID3D11ShaderResourceView* const* views)
{
    if (Tracing())
        LogSrvs(K, "cs", ctx, start, count, views);
    Orig<PSSetShaderResources_t>(K, kCSSetShaderResources)(ctx, start, count, views);
}

template <int K> void STDMETHODCALLTYPE hkDrawIndexed(ID3D11DeviceContext* ctx, UINT a, UINT b, INT c)
{
    CountDrawFor<K>(ctx);
    if (!ReplaceDrawFor<K>(ctx))
        Orig<DrawIndexed_t>(K, kDrawIndexed)(ctx, a, b, c);
    AfterDrawFor<K>(ctx);
}

template <int K> void STDMETHODCALLTYPE hkDraw(ID3D11DeviceContext* ctx, UINT a, UINT b)
{
    CountDrawFor<K>(ctx);
    if (!ReplaceDrawFor<K>(ctx))
        Orig<Draw_t>(K, kDraw)(ctx, a, b);
    AfterDrawFor<K>(ctx);
}

template <int K>
void STDMETHODCALLTYPE hkDrawIndexedInstanced(ID3D11DeviceContext* ctx, UINT a, UINT b, UINT c, INT d, UINT e)
{
    CountDrawFor<K>(ctx);
    if (!ReplaceDrawFor<K>(ctx))
        Orig<DrawIndexedInstanced_t>(K, kDrawIndexedInstanced)(ctx, a, b, c, d, e);
    AfterDrawFor<K>(ctx);
}

template <int K> void STDMETHODCALLTYPE hkDrawInstanced(ID3D11DeviceContext* ctx, UINT a, UINT b, UINT c, UINT d)
{
    CountDrawFor<K>(ctx);
    if (!ReplaceDrawFor<K>(ctx))
        Orig<DrawInstanced_t>(K, kDrawInstanced)(ctx, a, b, c, d);
    AfterDrawFor<K>(ctx);
}

template <int K>
void STDMETHODCALLTYPE hkOMSetRenderTargets(ID3D11DeviceContext* ctx, UINT numRtvs,
                                            ID3D11RenderTargetView* const* rtvs, ID3D11DepthStencilView* dsv)
{
    if (Tracing())
        LogTargets(K, "bind", ctx, numRtvs, rtvs, dsv);
    if (g_eyeNr)
    {
        if (IsDeferred<K>(ctx))
            EyeImageNr_Dx11::OnDeferredBind(ctx, numRtvs, rtvs, dsv);
        else
            EyeImageNr_Dx11::OnImmediateBind(ctx, numRtvs, rtvs, dsv);
    }
    if (g_finalNr && K == kImmediate && !OwnPass())
        FinalImageNr_Dx11::OnBind(ctx, numRtvs, rtvs, dsv);
    if (g_fc5Taa && K == kImmediate && !OwnPass())
        Fc5Taa_Dx11::OnBind(ctx, numRtvs, rtvs, dsv);
    if (g_matrixProbe && !OwnPass())
        MotionMatrixProbe_Dx11::OnBind(ctx, numRtvs, rtvs);
    const bool backend = K == kImmediate && Fc5Dlss_Dx11::Enabled() && !OwnPass();
    if (backend)
        Fc5Dlss_Dx11::BeforeBind(ctx, numRtvs, rtvs, dsv);
    Orig<OMSetRenderTargets_t>(K, kOMSetRenderTargets)(ctx, numRtvs, rtvs, dsv);
    if (backend)
        Fc5Dlss_Dx11::AfterBind(ctx);
}

template <int K>
void STDMETHODCALLTYPE hkOMSetRenderTargetsAndUAVs(ID3D11DeviceContext* ctx, UINT numRtvs,
                                                   ID3D11RenderTargetView* const* rtvs, ID3D11DepthStencilView* dsv,
                                                   UINT uavStart, UINT numUavs,
                                                   ID3D11UnorderedAccessView* const* uavs, const UINT* initialCounts)
{
    // D3D11_KEEP_RENDER_TARGETS_AND_DEPTH_STENCIL leaves the targets as they are.
    if (Tracing() && numRtvs != D3D11_KEEP_RENDER_TARGETS_AND_DEPTH_STENCIL)
        LogTargets(K, "bind+uav", ctx, numRtvs, rtvs, dsv);
    if (Tracing() && numUavs != D3D11_KEEP_UNORDERED_ACCESS_VIEWS && uavs != nullptr)
    {
        for (UINT i = 0; i < numUavs; ++i)
        {
            if (uavs[i] != nullptr && TakeLine())
                LOG_INFO("FC5 trace: present {} {}ctx {:X} ps uav u{} {}", g_present.load(), CtxKind(K),
                         (size_t) ctx, uavStart + i, DescView(uavs[i]));
        }
    }
    if (g_eyeNr && numRtvs != D3D11_KEEP_RENDER_TARGETS_AND_DEPTH_STENCIL)
    {
        if (IsDeferred<K>(ctx))
            EyeImageNr_Dx11::OnDeferredBind(ctx, numRtvs, rtvs, dsv);
        else
            EyeImageNr_Dx11::OnImmediateBind(ctx, numRtvs, rtvs, dsv);
    }
    if (g_finalNr && K == kImmediate && numRtvs != D3D11_KEEP_RENDER_TARGETS_AND_DEPTH_STENCIL && !OwnPass())
        FinalImageNr_Dx11::OnBind(ctx, numRtvs, rtvs, dsv);
    if (g_fc5Taa && K == kImmediate && numRtvs != D3D11_KEEP_RENDER_TARGETS_AND_DEPTH_STENCIL && !OwnPass())
        Fc5Taa_Dx11::OnBind(ctx, numRtvs, rtvs, dsv);
    if (g_matrixProbe && numRtvs != D3D11_KEEP_RENDER_TARGETS_AND_DEPTH_STENCIL && !OwnPass())
        MotionMatrixProbe_Dx11::OnBind(ctx, numRtvs, rtvs);
    const bool setsTargets = K == kImmediate && numRtvs != D3D11_KEEP_RENDER_TARGETS_AND_DEPTH_STENCIL &&
                             Fc5Dlss_Dx11::Enabled() && !OwnPass();
    if (setsTargets)
        Fc5Dlss_Dx11::BeforeBind(ctx, numRtvs, rtvs, dsv);
    Orig<OMSetRenderTargetsAndUAVs_t>(K, kOMSetRenderTargetsAndUAVs)(ctx, numRtvs, rtvs, dsv, uavStart, numUavs,
                                                                     uavs, initialCounts);
    if (setsTargets)
        Fc5Dlss_Dx11::AfterBind(ctx);
}

template <int K>
void STDMETHODCALLTYPE hkCSSetUnorderedAccessViews(ID3D11DeviceContext* ctx, UINT start, UINT count,
                                                   ID3D11UnorderedAccessView* const* uavs)
{
    if (Tracing() && uavs != nullptr)
    {
        for (UINT i = 0; i < count; ++i)
        {
            if (uavs[i] != nullptr && TakeLine())
                LOG_INFO("FC5 trace: present {} {}ctx {:X} cs uav u{} {}", g_present.load(), CtxKind(K),
                         (size_t) ctx, start + i, DescView(uavs[i]));
        }
    }
    if (g_finalNr && K == kImmediate && !OwnPass())
        FinalImageNr_Dx11::OnCsUavs(ctx, start, count, uavs);
    Orig<CSSetUnorderedAccessViews_t>(K, kCSSetUnorderedAccessViews)(ctx, start, count, uavs);
}

template <int K> void STDMETHODCALLTYPE hkDispatch(ID3D11DeviceContext* ctx, UINT x, UINT y, UINT z)
{
    if (K == kImmediate && OwnPass())
    {
        Orig<Dispatch_t>(K, kDispatch)(ctx, x, y, z);
        return;
    }
    if (Tracing() && TakeLine())
        LOG_INFO("FC5 trace: present {} {}ctx {:X} dispatch {}x{}x{}{}", g_present.load(), CtxKind(K), (size_t) ctx,
                 x, y, z, DrawsSince());
    if (g_finalNr && K == kImmediate)
        FinalImageNr_Dx11::OnDispatch(ctx);
    if (g_fc5Taa && K == kImmediate)
        Fc5Taa_Dx11::BeforeDispatch(ctx);
    // The motion decode runs first so the capture sees this Dispatch's result.
    if (g_taaMotion && K == kImmediate)
        TaaMotion_Dx11::BeforeDispatch(ctx);
    if (g_taaDlaa && K == kImmediate)
        TaaDlaa_Dx11::BeforeDispatch(ctx);
    if (g_capture && K == kImmediate)
        FrameCapture_Dx11::BeforeDispatch(ctx, x, y, z);
    // DLAA replaces the pass's output before the motion view and the capture, so they see the displayed image. It
    // runs in place of the game's pass when that is skipped (the game's pass still runs if DLAA can't this frame),
    // otherwise over its output.
    const bool dlaa = g_taaDlaa && K == kImmediate;
    const bool replaced = dlaa && g_taaDlaaSkip && TaaDlaa_Dx11::Replace(ctx);
    if (!replaced)
        Orig<Dispatch_t>(K, kDispatch)(ctx, x, y, z);
    if (dlaa && !g_taaDlaaSkip)
        TaaDlaa_Dx11::Replace(ctx);
    if (g_taaMotion && K == kImmediate)
        TaaMotion_Dx11::AfterDispatch(ctx);
    if (g_capture && K == kImmediate)
        FrameCapture_Dx11::AfterDispatch(ctx);
}

template <int K>
void STDMETHODCALLTYPE hkRSSetViewports(ID3D11DeviceContext* ctx, UINT count, const D3D11_VIEWPORT* viewports)
{
    if (Tracing() && count > 0 && viewports != nullptr && TakeLine())
        LOG_INFO("FC5 trace: present {} {}ctx {:X} viewport {}x{} at {},{}", g_present.load(), CtxKind(K),
                 (size_t) ctx, viewports[0].Width, viewports[0].Height, viewports[0].TopLeftX, viewports[0].TopLeftY);
    Orig<RSSetViewports_t>(K, kRSSetViewports)(ctx, count, viewports);
}

template <int K>
void STDMETHODCALLTYPE hkRSSetScissorRects(ID3D11DeviceContext* ctx, UINT count, const D3D11_RECT* rects)
{
    if (Tracing() && count > 0 && rects != nullptr && TakeLine())
        LOG_INFO("FC5 trace: present {} {}ctx {:X} scissor {},{}-{},{}", g_present.load(), CtxKind(K), (size_t) ctx,
                 rects[0].left, rects[0].top, rects[0].right, rects[0].bottom);
    Orig<RSSetScissorRects_t>(K, kRSSetScissorRects)(ctx, count, rects);
}

template <int K>
void STDMETHODCALLTYPE hkCopySubresourceRegion(ID3D11DeviceContext* ctx, ID3D11Resource* dst, UINT dstSub, UINT x,
                                               UINT y, UINT z, ID3D11Resource* src, UINT srcSub,
                                               const D3D11_BOX* box)
{
    if (Tracing() && TakeLine())
    {
        std::string region = box == nullptr ? std::string("whole")
                                            : std::format("box {},{}-{},{}", box->left, box->top, box->right,
                                                          box->bottom);
        LOG_INFO("FC5 trace: present {} {}ctx {:X} copyregion{} src [{}] sub {} {} dst [{}] sub {} at {},{}",
                 g_present.load(), CtxKind(K), (size_t) ctx, DrawsSince(), DescResource(src), srcSub, region,
                 DescResource(dst), dstSub, x, y);
    }
    Orig<CopySubresourceRegion_t>(K, kCopySubresourceRegion)(ctx, dst, dstSub, x, y, z, src, srcSub, box);
}

template <int K> void STDMETHODCALLTYPE hkCopyResource(ID3D11DeviceContext* ctx, ID3D11Resource* dst, ID3D11Resource* src)
{
    if (Tracing() && TakeLine())
        LOG_INFO("FC5 trace: present {} {}ctx {:X} copy{} src [{}] dst [{}]", g_present.load(), CtxKind(K),
                 (size_t) ctx, DrawsSince(), DescResource(src), DescResource(dst));
    Orig<CopyResource_t>(K, kCopyResource)(ctx, dst, src);
}

template <int K>
void STDMETHODCALLTYPE hkResolveSubresource(ID3D11DeviceContext* ctx, ID3D11Resource* dst, UINT dstSub,
                                            ID3D11Resource* src, UINT srcSub, DXGI_FORMAT format)
{
    if (Tracing() && TakeLine())
        LOG_INFO("FC5 trace: present {} {}ctx {:X} resolve{} src [{}] dst [{}] fmt {}", g_present.load(),
                 CtxKind(K), (size_t) ctx, DrawsSince(), DescResource(src), DescResource(dst), (UINT) format);
    Orig<ResolveSubresource_t>(K, kResolveSubresource)(ctx, dst, dstSub, src, srcSub, format);
}

template <int K>
void STDMETHODCALLTYPE hkExecuteCommandList(ID3D11DeviceContext* ctx, ID3D11CommandList* list, BOOL restore)
{
    if (Tracing() && TakeLine())
        LOG_INFO("FC5 trace: present {} {}ctx {:X} thread {} execute command list {:X}{}", g_present.load(),
                 CtxKind(K), (size_t) ctx, GetCurrentThreadId(), (size_t) list, DrawsSince());
    Orig<ExecuteCommandList_t>(K, kExecuteCommandList)(ctx, list, restore);
    if (g_eyeNr && !IsDeferred<K>(ctx))
        EyeImageNr_Dx11::AfterExecuteCommandList(ctx, list);
}

// Deferred contexts only: pairs what a context recorded with the ExecuteCommandList that later runs it.
HRESULT STDMETHODCALLTYPE hkFinishCommandList(ID3D11DeviceContext* ctx, BOOL restore, ID3D11CommandList** out)
{
    const HRESULT result = Orig<FinishCommandList_t>(kDeferred, kFinishCommandList)(ctx, restore, out);
    if (g_eyeNr)
        EyeImageNr_Dx11::OnFinishCommandList(ctx, (SUCCEEDED(result) && out != nullptr) ? *out : nullptr);
    if (Tracing() && TakeLine())
        LOG_INFO("FC5 trace: present {} deferred ctx {:X} thread {} finished command list {:X}{}", g_present.load(),
                 (size_t) ctx, GetCurrentThreadId(), (SUCCEEDED(result) && out != nullptr) ? (size_t) *out : 0,
                 DrawsSince());
    return result;
}

void Attach(PVOID* vtable, int k, int slot, PVOID hook)
{
    g_orig[k][slot] = vtable[slot];
    DetourAttach(&g_orig[k][slot], hook);
}

// FC5 per-eye jitter: the game's writes of its view constants are patched (Fc5Taa_Dx11).
template <int K>
HRESULT STDMETHODCALLTYPE hkMap(ID3D11DeviceContext* ctx, ID3D11Resource* resource, UINT subresource, D3D11_MAP type,
                                UINT flags, D3D11_MAPPED_SUBRESOURCE* mapped)
{
    const HRESULT result = Orig<Map_t>(K, kMap)(ctx, resource, subresource, type, flags, mapped);
    if (g_fc5Taa && SUCCEEDED(result) && mapped != nullptr && !OwnPass())
        Fc5Taa_Dx11::OnMap(resource, subresource, type, mapped->pData);
    if (g_matrixProbe && SUCCEEDED(result) && mapped != nullptr && !OwnPass())
        MotionMatrixProbe_Dx11::OnMap(ctx, resource, subresource, (int) type, mapped->pData);
    return result;
}

template <int K> void STDMETHODCALLTYPE hkUnmap(ID3D11DeviceContext* ctx, ID3D11Resource* resource, UINT subresource)
{
    if (g_fc5Taa)
        Fc5Taa_Dx11::BeforeUnmap(resource, subresource);
    if (g_matrixProbe)
        MotionMatrixProbe_Dx11::BeforeUnmap(ctx, resource, subresource);
    Orig<Unmap_t>(K, kUnmap)(ctx, resource, subresource);
}

template <int K>
void STDMETHODCALLTYPE hkUpdateSubresource(ID3D11DeviceContext* ctx, ID3D11Resource* resource, UINT subresource,
                                           const D3D11_BOX* box, const void* data, UINT rowPitch, UINT depthPitch)
{
    if (g_matrixProbe && !OwnPass())
        MotionMatrixProbe_Dx11::OnUpdateSubresource(ctx, resource, subresource, box, data);
    if (g_fc5Taa && !OwnPass())
        data = Fc5Taa_Dx11::PatchUpdate(resource, subresource, box, data);
    Orig<UpdateSubresource_t>(K, kUpdateSubresource)(ctx, resource, subresource, box, data, rowPitch, depthPitch);
}

template <int K> void AttachJitterHooks(PVOID* vtable)
{
    Attach(vtable, K, kMap, hkMap<K>);
    Attach(vtable, K, kUnmap, hkUnmap<K>);
    Attach(vtable, K, kUpdateSubresource, hkUpdateSubresource<K>);
}

// The trace-only hooks, for either function table.
template <int K> void AttachTraceHooks(PVOID* vtable)
{
    Attach(vtable, K, kPSSetShaderResources, hkPSSetShaderResources<K>);
    Attach(vtable, K, kDrawIndexed, hkDrawIndexed<K>);
    Attach(vtable, K, kDraw, hkDraw<K>);
    Attach(vtable, K, kDrawIndexedInstanced, hkDrawIndexedInstanced<K>);
    Attach(vtable, K, kDrawInstanced, hkDrawInstanced<K>);
    Attach(vtable, K, kDispatch, hkDispatch<K>);
    Attach(vtable, K, kRSSetViewports, hkRSSetViewports<K>);
    Attach(vtable, K, kRSSetScissorRects, hkRSSetScissorRects<K>);
    Attach(vtable, K, kCopySubresourceRegion, hkCopySubresourceRegion<K>);
    Attach(vtable, K, kCopyResource, hkCopyResource<K>);
    Attach(vtable, K, kResolveSubresource, hkResolveSubresource<K>);
    Attach(vtable, K, kExecuteCommandList, hkExecuteCommandList<K>);
    Attach(vtable, K, kCSSetShaderResources, hkCSSetShaderResources<K>);
    Attach(vtable, K, kCSSetUnorderedAccessViews, hkCSSetUnorderedAccessViews<K>);
}

// Once, for the first deferred context whose function table is not the immediate context's. Every deferred
// context of that implementation shares it.
void HookDeferredTable(ID3D11DeviceContext* ctx)
{
    bool expected = false;
    if (!g_deferredHooked.compare_exchange_strong(expected, true))
        return;

    PVOID* vtable = *(PVOID**) ctx;
    DetourTransactionBegin();
    DetourUpdateThread(GetCurrentThread());
    Attach(vtable, kDeferred, kOMSetRenderTargets, hkOMSetRenderTargets<kDeferred>);
    Attach(vtable, kDeferred, kOMSetRenderTargetsAndUAVs, hkOMSetRenderTargetsAndUAVs<kDeferred>);
    AttachTraceHooks<kDeferred>(vtable);
    Attach(vtable, kDeferred, kFinishCommandList, hkFinishCommandList);
    if (g_fc5Taa || g_matrixProbe)
        AttachJitterHooks<kDeferred>(vtable);
    const auto result = DetourTransactionCommit();
    if (result != NO_ERROR)
        LOG_ERROR("FC5 hooks: hooking the deferred context functions failed: {:X}", (UINT) result);
    else
        LOG_INFO("FC5 hooks: deferred context functions hooked");
}

// Diagnostics: shader bytecode for the frame capture.
HRESULT STDMETHODCALLTYPE hkCreateVertexShader(ID3D11Device* device, const void* code, SIZE_T length,
                                               ID3D11ClassLinkage* linkage, void** out)
{
    const HRESULT result = ((CreateShader_t) o_CreateVertexShader)(device, code, length, linkage, out);
    if (SUCCEEDED(result) && out != nullptr)
        FrameCapture_Dx11::RememberShader(*out, code, length);
    return result;
}

HRESULT STDMETHODCALLTYPE hkCreatePixelShader(ID3D11Device* device, const void* code, SIZE_T length,
                                              ID3D11ClassLinkage* linkage, void** out)
{
    const HRESULT result = ((CreateShader_t) o_CreatePixelShader)(device, code, length, linkage, out);
    if (SUCCEEDED(result) && out != nullptr)
        FrameCapture_Dx11::RememberShader(*out, code, length);
    return result;
}

HRESULT STDMETHODCALLTYPE hkCreateComputeShader(ID3D11Device* device, const void* code, SIZE_T length,
                                                ID3D11ClassLinkage* linkage, void** out)
{
    const HRESULT result = ((CreateShader_t) o_CreateComputeShader)(device, code, length, linkage, out);
    if (SUCCEEDED(result) && out != nullptr)
        FrameCapture_Dx11::RememberShader(*out, code, length);
    return result;
}

HRESULT STDMETHODCALLTYPE hkCreateDeferredContext(ID3D11Device* device, UINT flags, ID3D11DeviceContext** out)
{
    const HRESULT result = ((CreateDeferredContext_t) o_CreateDeferredContext)(device, flags, out);
    if (SUCCEEDED(result) && out != nullptr && *out != nullptr)
    {
        const int count = ++g_deferredCreated;
        const PVOID fn = (*(PVOID**) *out)[kOMSetRenderTargets];
        const bool sameTable = fn == g_immediateVtable33;
        if (count <= 8)
            LOG_INFO("FC5 trace: deferred context {} created {:X} (bind function {} the immediate context's)",
                     count, (size_t) *out, sameTable ? "same as" : "differs from");
        if (!sameTable)
            HookDeferredTable(*out);
        else if (g_eyeNr && count == 1)
            LOG_WARN("Eye image NR: deferred contexts share the immediate context's functions; finished lists "
                     "aren't hooked, so NR won't run");
    }
    return result;
}
} // namespace

void Fc5Trace_Dx11::Install(ID3D11Device* device)
{
    if (g_installed || device == nullptr)
        return;
    // FC5: its DLSS backend and/or the trace. EyeImageNr_Dx11 (Dark Souls III) and FinalImageNr_Dx11 (Days
    // Gone) need the full set of hooks on both function tables. Any D3D11 game: the trace, with diagnostics on.
    const char* exe = State::Instance().gameExe.c_str();
    const bool fc5 = _stricmp(exe, "FarCry5.exe") == 0;
    const bool diagnostics = Config::Instance()->DuniaTaaDiagnostics.value_or_default();
    const bool eyeNr = EyeImageNr_Dx11::Enabled();
    const bool finalNr = FinalImageNr_Dx11::Enabled();
    const bool taaMotion = TaaMotion_Dx11::Enabled();
    const bool fc5Taa = fc5 && Fc5Taa_Dx11::Enabled();
    const bool matrixProbe = MotionMatrixProbe_Dx11::Enabled();
    if (!diagnostics && !eyeNr && !finalNr && !taaMotion && !fc5Taa && !matrixProbe &&
        !(fc5 && Fc5Dlss_Dx11::Enabled()))
        return;
    const bool allHooks = diagnostics || eyeNr || finalNr || taaMotion || fc5Taa || matrixProbe;

    ID3D11DeviceContext* context = nullptr;
    device->GetImmediateContext(&context);
    if (context == nullptr)
        return;

    PVOID* ctxVtable = *(PVOID**) context;
    PVOID* devVtable = *(PVOID**) device;
    g_immediateVtable33 = ctxVtable[kOMSetRenderTargets];
    g_eyeNr = eyeNr;
    g_finalNr = finalNr;
    g_capture = diagnostics;
    g_taaMotion = taaMotion;
    g_taaDlaa = TaaDlaa_Dx11::Enabled();
    g_taaDlaaSkip = Config::Instance()->TaaDlaaSkipGameTaa.value_or_default();
    g_fc5Taa = fc5Taa;
    g_matrixProbe = matrixProbe;

    DetourTransactionBegin();
    DetourUpdateThread(GetCurrentThread());
    Attach(ctxVtable, kImmediate, kOMSetRenderTargets, hkOMSetRenderTargets<kImmediate>);
    Attach(ctxVtable, kImmediate, kOMSetRenderTargetsAndUAVs, hkOMSetRenderTargetsAndUAVs<kImmediate>);
    if (allHooks)
    {
        AttachTraceHooks<kImmediate>(ctxVtable);
    if (fc5Taa || matrixProbe)
        AttachJitterHooks<kImmediate>(ctxVtable);
        o_CreateDeferredContext = devVtable[kCreateDeferredContext];
        DetourAttach(&o_CreateDeferredContext, hkCreateDeferredContext);
    }
    // The shaders' bytecode is kept for the frame capture.
    if (diagnostics)
    {
        o_CreateVertexShader = devVtable[kCreateVertexShader];
        DetourAttach(&o_CreateVertexShader, hkCreateVertexShader);
        o_CreatePixelShader = devVtable[kCreatePixelShader];
        DetourAttach(&o_CreatePixelShader, hkCreatePixelShader);
        o_CreateComputeShader = devVtable[kCreateComputeShader];
        DetourAttach(&o_CreateComputeShader, hkCreateComputeShader);
    }
    const auto result = DetourTransactionCommit();

    if (result != NO_ERROR)
    {
        context->Release();
        LOG_ERROR("FC5 hooks: hooking failed: {:X}", (UINT) result);
        return;
    }
    context->Release();
    g_installed = true;
    LOG_INFO("FC5 hooks: installed on device {:X} (FC5 DLSS backend {}, eye image NR {}, final image NR "
             "{}, TAA motion check {}, TAA DLAA {}, FC5 TAA DLSS {}, trace {})",
             (size_t) device, Fc5Dlss_Dx11::Enabled() ? "on" : "off", eyeNr ? "on" : "off",
             finalNr ? "on" : "off", taaMotion ? "on" : "off",
             TaaDlaa_Dx11::Enabled() ? "on" : "off", fc5Taa ? "on" : "off", diagnostics ? "on" : "off");
}

bool Fc5Trace_Dx11::Installed() { return g_installed; }

void Fc5Trace_Dx11::OnPresent()
{
    if (!g_installed)
        return;

    const uint64_t present = ++g_present;

    if (Config::Instance()->DuniaTaaTraceFrame.value_or_default())
    {
        Config::Instance()->DuniaTaaTraceFrame = false;
        if (!Tracing() && g_armAt == 0)
        {
            g_armAt = GetTickCount64() + kArmDelayMs;
            LOG_INFO("FC5 trace: armed, starts in {} s (close the overlay)", kArmDelayMs / 1000);
        }
    }

    if (g_armAt != 0 && GetTickCount64() >= g_armAt)
    {
        g_armAt = 0;
        g_lines = 0;
        ++g_traceGeneration;
        g_traceEnd = present + 2;
        LOG_INFO("FC5 trace: begin at present {} ({} deferred contexts created so far)", present,
                 g_deferredCreated.load());
        g_tracing = true;
    }
    else if (Tracing() && present >= g_traceEnd)
    {
        g_tracing = false;
        LOG_INFO("FC5 trace: end at present {}, {} lines{}", present, g_lines.load(), DrawsSince());
    }
}
