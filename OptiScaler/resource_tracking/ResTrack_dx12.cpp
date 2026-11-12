#include "pch.h"

#include <dlssnr/DlssNr_ExposureScan.h>
#include <dlssnr/DlssNr.h>

#include "ResTrack_dx12.h"

#include <Config.h>
#include <State.h>
#include <Util.h>
#include <inputs/ConstantCapture_Dx12.h>
#include <inputs/DuniaTaa_Dx12.h>
#include <inputs/GiCapture_Dx12.h>

#include <menu/menu_overlay_dx.h>

#include <algorithm>
#include <future>

#include <magic_enum_utility.hpp>
#include <include/d3dx/d3dx12.h>
#include <detours/detours.h>

#ifndef STDMETHODCALLTYPE
#include <Unknwn.h> // or <objbase.h> to get STDMETHODCALLTYPE
#endif

#ifdef USE_SPINLOCK_MUTEX
#define LOCK_GUARD(mutex) std::lock_guard<SpinLock> name(mutex)
#else
#define LOCK_GUARD(mutex) std::lock_guard<std::mutex> name(mutex)
#endif

// Device hooks for FG
typedef void(STDMETHODCALLTYPE* PFN_CreateRenderTargetView)(ID3D12Device* This, ID3D12Resource* pResource,
                                                            D3D12_RENDER_TARGET_VIEW_DESC* pDesc,
                                                            D3D12_CPU_DESCRIPTOR_HANDLE DestDescriptor);
typedef void(STDMETHODCALLTYPE* PFN_CreateShaderResourceView)(ID3D12Device* This, ID3D12Resource* pResource,
                                                              D3D12_SHADER_RESOURCE_VIEW_DESC* pDesc,
                                                              D3D12_CPU_DESCRIPTOR_HANDLE DestDescriptor);
typedef void(STDMETHODCALLTYPE* PFN_CreateUnorderedAccessView)(ID3D12Device* This, ID3D12Resource* pResource,
                                                               ID3D12Resource* pCounterResource,
                                                               D3D12_UNORDERED_ACCESS_VIEW_DESC* pDesc,
                                                               D3D12_CPU_DESCRIPTOR_HANDLE DestDescriptor);
typedef void(STDMETHODCALLTYPE* PFN_CreateDepthStencilView)(ID3D12Device* This, ID3D12Resource* pResource,
                                                            const D3D12_DEPTH_STENCIL_VIEW_DESC* pDesc,
                                                            D3D12_CPU_DESCRIPTOR_HANDLE DestDescriptor);
typedef void(STDMETHODCALLTYPE* PFN_CreateConstantBufferView)(ID3D12Device* This,
                                                              const D3D12_CONSTANT_BUFFER_VIEW_DESC* pDesc,
                                                              D3D12_CPU_DESCRIPTOR_HANDLE DestDescriptor);

typedef void(STDMETHODCALLTYPE* PFN_CreateSampler)(ID3D12Device* This, const D3D12_SAMPLER_DESC* pDesc,
                                                   D3D12_CPU_DESCRIPTOR_HANDLE DestDescriptor);

typedef HRESULT(STDMETHODCALLTYPE* PFN_CreateDescriptorHeap)(ID3D12Device* This,
                                                             D3D12_DESCRIPTOR_HEAP_DESC* pDescriptorHeapDesc,
                                                             REFIID riid, void** ppvHeap);
typedef ULONG(STDMETHODCALLTYPE* PFN_HeapRelease)(ID3D12DescriptorHeap* This);
typedef void(STDMETHODCALLTYPE* PFN_CopyDescriptors)(ID3D12Device* This, UINT NumDestDescriptorRanges,
                                                     D3D12_CPU_DESCRIPTOR_HANDLE* pDestDescriptorRangeStarts,
                                                     UINT* pDestDescriptorRangeSizes, UINT NumSrcDescriptorRanges,
                                                     D3D12_CPU_DESCRIPTOR_HANDLE* pSrcDescriptorRangeStarts,
                                                     UINT* pSrcDescriptorRangeSizes,
                                                     D3D12_DESCRIPTOR_HEAP_TYPE DescriptorHeapsType);
typedef void(STDMETHODCALLTYPE* PFN_CopyDescriptorsSimple)(ID3D12Device* This, UINT NumDescriptors,
                                                           D3D12_CPU_DESCRIPTOR_HANDLE DestDescriptorRangeStart,
                                                           D3D12_CPU_DESCRIPTOR_HANDLE SrcDescriptorRangeStart,
                                                           D3D12_DESCRIPTOR_HEAP_TYPE DescriptorHeapsType);

// Command list hooks for FG
typedef void(STDMETHODCALLTYPE* PFN_OMSetRenderTargets)(ID3D12GraphicsCommandList* This,
                                                        UINT NumRenderTargetDescriptors,
                                                        D3D12_CPU_DESCRIPTOR_HANDLE* pRenderTargetDescriptors,
                                                        BOOL RTsSingleHandleToDescriptorRange,
                                                        D3D12_CPU_DESCRIPTOR_HANDLE* pDepthStencilDescriptor);
typedef void(STDMETHODCALLTYPE* PFN_SetGraphicsRootDescriptorTable)(ID3D12GraphicsCommandList* This,
                                                                    UINT RootParameterIndex,
                                                                    D3D12_GPU_DESCRIPTOR_HANDLE BaseDescriptor);
typedef void(STDMETHODCALLTYPE* PFN_SetComputeRootDescriptorTable)(ID3D12GraphicsCommandList* This,
                                                                   UINT RootParameterIndex,
                                                                   D3D12_GPU_DESCRIPTOR_HANDLE BaseDescriptor);
typedef void(STDMETHODCALLTYPE* PFN_DrawIndexedInstanced)(ID3D12GraphicsCommandList* This, UINT IndexCountPerInstance,
                                                          UINT InstanceCount, UINT StartIndexLocation,
                                                          INT BaseVertexLocation, UINT StartInstanceLocation);
typedef void(STDMETHODCALLTYPE* PFN_DrawInstanced)(ID3D12GraphicsCommandList* This, UINT VertexCountPerInstance,
                                                   UINT InstanceCount, UINT StartVertexLocation,
                                                   UINT StartInstanceLocation);
typedef void(STDMETHODCALLTYPE* PFN_Dispatch)(ID3D12GraphicsCommandList* This, UINT ThreadGroupCountX,
                                              UINT ThreadGroupCountY, UINT ThreadGroupCountZ);
typedef void(STDMETHODCALLTYPE* PFN_ExecuteBundle)(ID3D12GraphicsCommandList* This,
                                                   ID3D12GraphicsCommandList* pCommandList);
typedef HRESULT(STDMETHODCALLTYPE* PFN_Close)(ID3D12GraphicsCommandList* This);
typedef void(STDMETHODCALLTYPE* PFN_CopyResource)(ID3D12GraphicsCommandList* This, ID3D12Resource* pDstResource,
                                                  ID3D12Resource* pSrcResource);
typedef void(STDMETHODCALLTYPE* PFN_CopyTextureRegion)(ID3D12GraphicsCommandList* This,
                                                       const D3D12_TEXTURE_COPY_LOCATION* pDst, UINT DstX, UINT DstY,
                                                       UINT DstZ, const D3D12_TEXTURE_COPY_LOCATION* pSrc,
                                                       const D3D12_BOX* pSrcBox);

typedef void(STDMETHODCALLTYPE* PFN_ResolveSubresource)(ID3D12GraphicsCommandList* This, ID3D12Resource* pDstResource,
                                                        UINT DstSubresource, ID3D12Resource* pSrcResource,
                                                        UINT SrcSubresource, DXGI_FORMAT Format);
typedef void(STDMETHODCALLTYPE* PFN_ClearRenderTargetView)(ID3D12GraphicsCommandList* This,
                                                           D3D12_CPU_DESCRIPTOR_HANDLE RenderTargetView,
                                                           const FLOAT ColorRGBA[4], UINT NumRects,
                                                           const D3D12_RECT* pRects);
typedef void(STDMETHODCALLTYPE* PFN_ClearUnorderedAccessViewUint)(
    ID3D12GraphicsCommandList* This, D3D12_GPU_DESCRIPTOR_HANDLE ViewGPUHandleInCurrentHeap,
    D3D12_CPU_DESCRIPTOR_HANDLE ViewCPUHandle, ID3D12Resource* pResource, const UINT Values[4], UINT NumRects,
    const D3D12_RECT* pRects);
typedef void(STDMETHODCALLTYPE* PFN_ClearUnorderedAccessViewFloat)(
    ID3D12GraphicsCommandList* This, D3D12_GPU_DESCRIPTOR_HANDLE ViewGPUHandleInCurrentHeap,
    D3D12_CPU_DESCRIPTOR_HANDLE ViewCPUHandle, ID3D12Resource* pResource, const FLOAT Values[4], UINT NumRects,
    const D3D12_RECT* pRects);

typedef void(STDMETHODCALLTYPE* PFN_ExecuteCommandLists)(ID3D12CommandQueue* This, UINT NumCommandLists,
                                                         ID3D12CommandList* const* ppCommandLists);

typedef ULONG(STDMETHODCALLTYPE* PFN_Release)(ID3D12Resource* This);

// Original method calls for device
static PFN_CreateRenderTargetView o_CreateRenderTargetView = nullptr;
static PFN_CreateShaderResourceView o_CreateShaderResourceView = nullptr;
static PFN_CreateUnorderedAccessView o_CreateUnorderedAccessView = nullptr;
static PFN_CreateDepthStencilView o_CreateDepthStencilView = nullptr;
static PFN_CreateConstantBufferView o_CreateConstantBufferView = nullptr;
static PFN_CreateSampler o_CreateSampler = nullptr;

static PFN_CreateDescriptorHeap o_CreateDescriptorHeap = nullptr;
static PFN_HeapRelease o_HeapRelease = nullptr;
static PFN_CopyDescriptors o_CopyDescriptors = nullptr;
static PFN_CopyDescriptorsSimple o_CopyDescriptorsSimple = nullptr;

// Original method calls for command list
static PFN_Dispatch o_Dispatch = nullptr;
static PFN_DrawInstanced o_DrawInstanced = nullptr;
static PFN_DrawIndexedInstanced o_DrawIndexedInstanced = nullptr;
static PFN_ExecuteBundle o_ExecuteBundle = nullptr;
static PFN_Close o_Close = nullptr;
static PFN_CopyResource o_CopyResource = nullptr;
static PFN_CopyTextureRegion o_CopyTextureRegion = nullptr;
static PFN_ResolveSubresource o_ResolveSubresource = nullptr;
static PFN_ClearRenderTargetView o_ClearRenderTargetView = nullptr;
static PFN_ClearUnorderedAccessViewUint o_ClearUnorderedAccessViewUint = nullptr;
static PFN_ClearUnorderedAccessViewFloat o_ClearUnorderedAccessViewFloat = nullptr;
using PFN_LateReset = HRESULT(STDMETHODCALLTYPE*)(ID3D12GraphicsCommandList*, ID3D12CommandAllocator*, ID3D12PipelineState*);
static PFN_LateReset o_LateReset = nullptr;
static HRESULT STDMETHODCALLTYPE hkLateReset(ID3D12GraphicsCommandList* cmd, ID3D12CommandAllocator* allocator,
                                             ID3D12PipelineState* pipeline)
{
    const auto result = o_LateReset(cmd, allocator, pipeline);
    if (SUCCEEDED(result)) DlssNr::FinishedPictureResetCommandList(cmd);
    return result;
}

static PFN_ExecuteCommandLists o_ExecuteCommandLists = nullptr;
static PFN_Release o_Release = nullptr;

// ID3D12Resource::Map, vtable 8. Taken off the same temporary upload buffer that gives us Release, and
// installed only for the FC6 constant diagnostic.
typedef HRESULT(STDMETHODCALLTYPE* PFN_ResourceMap)(ID3D12Resource* This, UINT Subresource,
                                                    const D3D12_RANGE* pReadRange, void** ppData);
static PFN_ResourceMap o_ResourceMap = nullptr;

static HRESULT STDMETHODCALLTYPE hkResourceMap(ID3D12Resource* This, UINT Subresource,
                                               const D3D12_RANGE* pReadRange, void** ppData)
{
    const auto result = o_ResourceMap(This, Subresource, pReadRange, ppData);
    // Whatever created the buffer, it has to be mapped to be written, so this catches the constants the
    // creation hooks miss entirely. Subresource 0 only: a buffer has no others.
    if (SUCCEEDED(result) && Subresource == 0)
        ConstantCapture_Dx12::NoteMappedBuffer(This);
    return result;
}

// ID3D12Resource::Unmap, vtable 9. Its pWrittenRange is the game telling us exactly which bytes it just
// wrote -- for a per-view constant block, the block itself. Checking that range for the fingerprint costs
// a few dozen compares per Unmap and replaces a sweep of the whole 1.3 GB mapped set, which at ~40 MB/s of
// uncached reads never finished inside a run, four orderings in a row.
typedef void(STDMETHODCALLTYPE* PFN_ResourceUnmap)(ID3D12Resource* This, UINT Subresource,
                                                  const D3D12_RANGE* pWrittenRange);
static PFN_ResourceUnmap o_ResourceUnmap = nullptr;

static void STDMETHODCALLTYPE hkResourceUnmap(ID3D12Resource* This, UINT Subresource,
                                              const D3D12_RANGE* pWrittenRange)
{
    // Before the original: the mapping is certainly still valid while the game still holds it.
    if (Subresource == 0)
        ConstantCapture_Dx12::NoteWrittenRange(This, pWrittenRange);
    o_ResourceUnmap(This, Subresource, pWrittenRange);
}

static PFN_OMSetRenderTargets o_OMSetRenderTargets = nullptr;
static PFN_SetGraphicsRootDescriptorTable o_SetGraphicsRootDescriptorTable = nullptr;
static PFN_SetComputeRootDescriptorTable o_SetComputeRootDescriptorTable = nullptr;

static std::mutex _hudlessTrackMutex;
static ankerl::unordered_dense::map<ID3D12GraphicsCommandList*,
                                    ankerl::unordered_dense::map<ID3D12Resource*, ResourceInfo>>
    fgPossibleHudless[BUFFER_COUNT];

// heaps section

static std::shared_mutex _heapRegistryMutex;
static std::vector<std::shared_ptr<HeapInfo>> fgHeaps;

static std::set<void*> _notFoundCmdLists;
static std::unordered_map<FG_ResourceType, void*> _resCmdList[BUFFER_COUNT];

struct HeapCacheTLS
{
    unsigned genSeen = 0;
    std::shared_ptr<HeapInfo> heap;
    uint64_t heapVersion = 0;
};

static thread_local HeapCacheTLS cache;
static thread_local HeapCacheTLS cacheRTV;
static thread_local HeapCacheTLS cacheCBV;
static thread_local HeapCacheTLS cacheSRV;
static thread_local HeapCacheTLS cacheUAV;
static std::atomic<unsigned> gHeapGeneration { 1 };

static thread_local HeapCacheTLS cacheGR;
static thread_local HeapCacheTLS cacheCR;

bool ResTrack_Dx12::CheckResource(ID3D12Resource* resource)
{
    if (State::Instance().isShuttingDown)
        return false;

    auto resDesc = resource->GetDesc();

    if (resDesc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D)
        return false;

    if (DuniaTaa_Dx12::Active() && resDesc.Width >= 256 && resDesc.Height >= 256)
        return true;

    auto& s = State::Instance();

    if (resDesc.Height != s.currentSwapchainDesc.BufferDesc.Height ||
        resDesc.Width != s.currentSwapchainDesc.BufferDesc.Width)
    {
        auto result = Config::Instance()->FGRelaxedResolutionCheck.value_or_default() &&
                      resDesc.Height >= s.currentSwapchainDesc.BufferDesc.Height - 32 &&
                      resDesc.Height <= s.currentSwapchainDesc.BufferDesc.Height + 32 &&
                      resDesc.Width >= s.currentSwapchainDesc.BufferDesc.Width - 32 &&
                      resDesc.Width <= s.currentSwapchainDesc.BufferDesc.Width + 32;

        // LOG_TRACK("Resource: {}x{} ({}), Swapchain: {}x{} ({}), Relaxed Result: {}", resDesc.Width, resDesc.Height,
        //           (UINT) resDesc.Format, scDesc.BufferDesc.Width, scDesc.BufferDesc.Height,
        //           (UINT) scDesc.BufferDesc.Format, result);

        return result;
    }

    return true;
}

inline static IID streamlineRiid {};
inline static std::once_flag streamlineRiidInitFlag;

bool ResTrack_Dx12::CheckForRealObject(const std::string functionName, IUnknown* pObject, IUnknown** ppRealObject)
{
    std::call_once(streamlineRiidInitFlag,
                   []() { IIDFromString(L"{ADEC44E2-61F0-45C3-AD9F-1B37379284FF}", &streamlineRiid); });

    auto qResult = pObject->QueryInterface(streamlineRiid, (void**) ppRealObject);

    if (qResult == S_OK && *ppRealObject != nullptr)
    {
        LOG_INFO("{} Streamline proxy found!", functionName);
        (*ppRealObject)->Release();
        return true;
    }

    return false;
}

#pragma region Resource methods

bool ResTrack_Dx12::CreateBufferResource(ID3D12Device* InDevice, ResourceInfo* InSource, D3D12_RESOURCE_STATES InState,
                                         ID3D12Resource** OutResource)
{
    if (InDevice == nullptr || InSource == nullptr || InSource->buffer == nullptr)
        return false;

    if (*OutResource != nullptr)
    {
        auto bufDesc = (*OutResource)->GetDesc();

        if (bufDesc.Width != (UINT64) (InSource->width) || bufDesc.Height != (UINT) (InSource->height) ||
            bufDesc.Format != InSource->format)
        {
            (*OutResource)->Release();
            (*OutResource) = nullptr;
        }
        else
            return true;
    }

    D3D12_HEAP_PROPERTIES heapProperties;
    D3D12_HEAP_FLAGS heapFlags;
    HRESULT hr = InSource->buffer->GetHeapProperties(&heapProperties, &heapFlags);

    if (hr != S_OK)
    {
        LOG_ERROR("GetHeapProperties result: {0:X}", (UINT64) hr);
        return false;
    }

    D3D12_RESOURCE_DESC texDesc = InSource->buffer->GetDesc();
    texDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

    hr = InDevice->CreateCommittedResource(&heapProperties, D3D12_HEAP_FLAG_NONE, &texDesc, InState, nullptr,
                                           IID_PPV_ARGS(OutResource));

    if (hr != S_OK)
    {
        LOG_ERROR("CreateCommittedResource result: {0:X}", (UINT64) hr);
        return false;
    }

    (*OutResource)->SetName(L"fgHudlessSCBufferCopy");
    return true;
}

void ResTrack_Dx12::ResourceBarrier(ID3D12GraphicsCommandList* InCommandList, ID3D12Resource* InResource,
                                    D3D12_RESOURCE_STATES InBeforeState, D3D12_RESOURCE_STATES InAfterState)
{
    if (InBeforeState == InAfterState)
        return;

    D3D12_RESOURCE_BARRIER barrier = {};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = InResource;
    barrier.Transition.StateBefore = InBeforeState;
    barrier.Transition.StateAfter = InAfterState;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    InCommandList->ResourceBarrier(1, &barrier);
}

#pragma endregion

#pragma region Heap helpers

SIZE_T ResTrack_Dx12::GetGPUHandle(ID3D12Device* This, SIZE_T cpuHandle, D3D12_DESCRIPTOR_HEAP_TYPE type)
{
    std::shared_lock lock(_heapRegistryMutex);
    for (const auto& heap : fgHeaps)
    {
        if (heap != nullptr && heap->active.load(std::memory_order_acquire) && heap->cpuStart <= cpuHandle &&
            heap->cpuEnd > cpuHandle && heap->gpuStart != 0)
        {
            auto incSize = This->GetDescriptorHandleIncrementSize(type);
            auto addr = cpuHandle - heap->cpuStart;
            auto index = addr / incSize;
            return heap->gpuStart + (index * incSize);
        }
    }

    return NULL;
}

SIZE_T ResTrack_Dx12::GetCPUHandle(ID3D12Device* This, SIZE_T gpuHandle, D3D12_DESCRIPTOR_HEAP_TYPE type)
{
    std::shared_lock lock(_heapRegistryMutex);
    for (const auto& heap : fgHeaps)
    {
        if (heap != nullptr && heap->active.load(std::memory_order_acquire) && heap->gpuStart <= gpuHandle &&
            heap->gpuEnd > gpuHandle && heap->cpuStart != 0)
        {
            auto incSize = This->GetDescriptorHandleIncrementSize(type);
            auto addr = gpuHandle - heap->gpuStart;
            auto index = addr / incSize;
            return heap->cpuStart + (index * incSize);
        }
    }

    return NULL;
}

static std::shared_ptr<HeapInfo> FindHeapByCpuHandle(SIZE_T cpuHandle, HeapCacheTLS& heapCache)
{
    const auto currentGen = gHeapGeneration.load(std::memory_order_acquire);
    auto* cachedHeap = heapCache.heap.get();
    if (heapCache.genSeen == currentGen && cachedHeap != nullptr &&
        cachedHeap->version.load(std::memory_order_relaxed) == heapCache.heapVersion &&
        cachedHeap->active.load(std::memory_order_acquire) && cachedHeap->cpuStart <= cpuHandle &&
        cpuHandle < cachedHeap->cpuEnd)
    {
        return heapCache.heap;
    }

    std::shared_lock lock(_heapRegistryMutex);
    const auto registryGen = gHeapGeneration.load(std::memory_order_acquire);
    for (const auto& heap : fgHeaps)
    {
        if (heap != nullptr && heap->active.load(std::memory_order_acquire) && heap->cpuStart <= cpuHandle &&
            cpuHandle < heap->cpuEnd)
        {
            heapCache.genSeen = registryGen;
            heapCache.heap = heap;
            heapCache.heapVersion = heap->version.load(std::memory_order_relaxed);
            return heap;
        }
    }

    heapCache.genSeen = registryGen;
    heapCache.heapVersion = 0;
    heapCache.heap.reset();
    return nullptr;
}

static std::shared_ptr<HeapInfo> FindHeapByGpuHandle(SIZE_T gpuHandle, HeapCacheTLS& heapCache)
{
    if (gpuHandle == NULL)
        return nullptr;

    const auto currentGen = gHeapGeneration.load(std::memory_order_acquire);
    auto* cachedHeap = heapCache.heap.get();
    if (heapCache.genSeen == currentGen && cachedHeap != nullptr &&
        cachedHeap->version.load(std::memory_order_relaxed) == heapCache.heapVersion &&
        cachedHeap->active.load(std::memory_order_acquire) && cachedHeap->gpuStart <= gpuHandle &&
        gpuHandle < cachedHeap->gpuEnd)
    {
        return heapCache.heap;
    }

    std::shared_lock lock(_heapRegistryMutex);
    const auto registryGen = gHeapGeneration.load(std::memory_order_acquire);
    for (const auto& heap : fgHeaps)
    {
        if (heap != nullptr && heap->active.load(std::memory_order_acquire) && heap->gpuStart <= gpuHandle &&
            gpuHandle < heap->gpuEnd)
        {
            heapCache.genSeen = registryGen;
            heapCache.heap = heap;
            heapCache.heapVersion = heap->version.load(std::memory_order_relaxed);
            return heap;
        }
    }

    heapCache.genSeen = registryGen;
    heapCache.heapVersion = 0;
    heapCache.heap.reset();
    return nullptr;
}

std::shared_ptr<HeapInfo> ResTrack_Dx12::GetHeapByCpuHandleCBV(SIZE_T cpuHandle)
{
    return FindHeapByCpuHandle(cpuHandle, cacheCBV);
}

std::shared_ptr<HeapInfo> ResTrack_Dx12::GetHeapByCpuHandleRTV(SIZE_T cpuHandle)
{
    return FindHeapByCpuHandle(cpuHandle, cacheRTV);
}

std::shared_ptr<HeapInfo> ResTrack_Dx12::GetHeapByCpuHandleSRV(SIZE_T cpuHandle)
{
    return FindHeapByCpuHandle(cpuHandle, cacheSRV);
}

std::shared_ptr<HeapInfo> ResTrack_Dx12::GetHeapByCpuHandleUAV(SIZE_T cpuHandle)
{
    return FindHeapByCpuHandle(cpuHandle, cacheUAV);
}

std::shared_ptr<HeapInfo> ResTrack_Dx12::GetHeapByCpuHandle(SIZE_T cpuHandle)
{
    return FindHeapByCpuHandle(cpuHandle, cache);
}

std::shared_ptr<HeapInfo> ResTrack_Dx12::GetHeapByGpuHandleGR(SIZE_T gpuHandle)
{
    return FindHeapByGpuHandle(gpuHandle, cacheGR);
}

std::shared_ptr<HeapInfo> ResTrack_Dx12::GetHeapByGpuHandleCR(SIZE_T gpuHandle)
{
    return FindHeapByGpuHandle(gpuHandle, cacheCR);
}

#pragma endregion

#pragma region Hudless methods

void ResTrack_Dx12::FillResourceInfo(ID3D12Resource* resource, ResourceInfo* info)
{
    auto desc = resource->GetDesc();
    info->buffer = resource;
    info->width = desc.Width;
    info->height = desc.Height;
    info->format = desc.Format;
    info->flags = desc.Flags;
}

bool ResTrack_Dx12::IsHudFixActive()
{
    if (!Config::Instance()->FGEnabled.value_or_default() || !Config::Instance()->FGHUDFix.value_or_default())
    {
        LOG_TRACK(
            "!Config::Instance()->FGEnabled.value_or_default() || !Config::Instance()->FGHUDFix.value_or_default()");
        return false;
    }

    if (State::Instance().currentFG == nullptr || State::Instance().currentFeature == nullptr ||
        State::Instance().fgChanged)
    {
        LOG_TRACK("State::Instance().currentFG == nullptr || State::Instance().currentFeature == nullptr || "
                  "State::Instance().fgChanged");
        return false;
    }

    if (!State::Instance().currentFG->IsActive())
    {
        LOG_TRACK("!State::Instance().currentFG->IsActive()");
        return false;
    }

    if (!_presentDone)
    {
        LOG_TRACK("!_presentDone");
        return false;
    }

    if (Hudfix_Dx12::SkipHudlessChecks())
    {
        LOG_TRACK("!Hudfix_Dx12::SkipHudlessChecks()");
        return false;
    }

    if (!Hudfix_Dx12::IsResourceCheckActive())
    {
        // LOG_TRACK("!Hudfix_Dx12::IsResourceCheckActive()");
        return false;
    }

    return true;
}

#pragma endregion

#pragma region Resource input hooks

void ResTrack_Dx12::hkCreateRenderTargetView(ID3D12Device* This, ID3D12Resource* pResource,
                                             D3D12_RENDER_TARGET_VIEW_DESC* pDesc,
                                             D3D12_CPU_DESCRIPTOR_HANDLE DestDescriptor)
{
    // force hdr for swapchain buffer
    if (pResource != nullptr && pDesc != nullptr && Config::Instance()->ForceHDR.value_or_default())
    {
        for (size_t i = 0; i < State::Instance().scBuffers.size(); i++)
        {
            if (State::Instance().scBuffers[i] == pResource)
            {
                if (Config::Instance()->UseHDR10.value_or_default())
                    pDesc->Format = DXGI_FORMAT_R10G10B10A2_UNORM;
                else
                    pDesc->Format = DXGI_FORMAT_R16G16B16A16_FLOAT;

                break;
            }
        }
    }

    o_CreateRenderTargetView(This, pResource, pDesc, DestDescriptor);

    if (Config::Instance()->FGHudfixDisableRTV.value_or_default())
        return;

    if (pResource == nullptr || !CheckResource(pResource))
    {
        auto heap = GetHeapByCpuHandleRTV(DestDescriptor.ptr);

        if (heap != nullptr)
            heap->ClearByCpuHandle(DestDescriptor.ptr);

        return;
    }

    // if (!CheckResource(pResource))
    //     return;

    auto heap = GetHeapByCpuHandleRTV(DestDescriptor.ptr);
    if (heap != nullptr)
    {
        ResourceInfo resInfo {};
        FillResourceInfo(pResource, &resInfo);
        resInfo.type = RTV;
        resInfo.captureInfo = CaptureInfo::CreateRTV;
        heap->SetByCpuHandle(DestDescriptor.ptr, resInfo);
    }

    if (DuniaTaa_Dx12::Active())
        DuniaTaa_Dx12::NoteMotion(pResource, true);
}

void ResTrack_Dx12::hkCreateDepthStencilView(ID3D12Device* This, ID3D12Resource* pResource,
                                             const D3D12_DEPTH_STENCIL_VIEW_DESC* pDesc,
                                             D3D12_CPU_DESCRIPTOR_HANDLE DestDescriptor)
{
    o_CreateDepthStencilView(This, pResource, pDesc, DestDescriptor);

    if (!DuniaTaa_Dx12::Active() || pResource == nullptr || !CheckResource(pResource))
    {
        auto heap = GetHeapByCpuHandle(DestDescriptor.ptr);
        if (heap != nullptr)
            heap->ClearByCpuHandle(DestDescriptor.ptr);
        return;
    }

    auto heap = GetHeapByCpuHandle(DestDescriptor.ptr);
    if (heap != nullptr)
    {
        ResourceInfo resInfo {};
        FillResourceInfo(pResource, &resInfo);
        resInfo.type = DSV;
        heap->SetByCpuHandle(DestDescriptor.ptr, resInfo);
    }
}

void ResTrack_Dx12::hkCreateShaderResourceView(ID3D12Device* This, ID3D12Resource* pResource,
                                               D3D12_SHADER_RESOURCE_VIEW_DESC* pDesc,
                                               D3D12_CPU_DESCRIPTOR_HANDLE DestDescriptor)
{
    // FC6: which slots of the shader-visible heaps the game ever writes (a range compare; nothing when inactive).
    DuniaTaa_Dx12::NoteDescriptorWrite(DestDescriptor.ptr, 1);

    // force hdr for swapchain buffer
    if (pResource != nullptr && pDesc != nullptr && Config::Instance()->ForceHDR.value_or_default())
    {
        for (size_t i = 0; i < State::Instance().scBuffers.size(); i++)
        {
            if (State::Instance().scBuffers[i] == pResource)
            {
                if (Config::Instance()->UseHDR10.value_or_default())
                    pDesc->Format = DXGI_FORMAT_R10G10B10A2_UNORM;
                else
                    pDesc->Format = DXGI_FORMAT_R16G16B16A16_FLOAT;

                break;
            }
        }
    }

    o_CreateShaderResourceView(This, pResource, pDesc, DestDescriptor);

    if (Config::Instance()->FGHudfixDisableSRV.value_or_default())
        return;

    // The second blind spot of the same kind: a structured or raw buffer read as an SRV is how plenty of engines
    // hand a shader its constants, and CheckResource turns away anything that is not a TEXTURE2D -- so those have
    // been exactly as invisible as CBVs were, for as long, and for the same reason. Recorded only for the FC6
    // diagnostic, so nothing else sees a difference: height stays 0, which is how the rest of this code already
    // tells a buffer from a texture, and every filter that wants a texture shape keeps skipping them.
    // CbvRecordingOn, not WantsCbvTracking: the latter is now true for FC6 always (hook installation), and
    // recording buffer SRVs in everyday play is not wanted.
    if (pResource != nullptr && DuniaTaa_Dx12::CbvRecordingOn())
    {
        const auto bufDesc = pResource->GetDesc();
        if (bufDesc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER && bufDesc.Width >= 64)
        {
            auto bufHeap = GetHeapByCpuHandleSRV(DestDescriptor.ptr);
            if (bufHeap != nullptr)
            {
                ResourceInfo resInfo {};
                resInfo.buffer = pResource;
                resInfo.width = bufDesc.Width;
                resInfo.height = 0;
                resInfo.format = DXGI_FORMAT_UNKNOWN;
                resInfo.type = SRV;
                resInfo.captureInfo = CaptureInfo::CreateSRV;
                bufHeap->SetByCpuHandle(DestDescriptor.ptr, resInfo);
            }
            return;
        }
    }

    if (pResource == nullptr || !CheckResource(pResource))
    {
        auto heap = GetHeapByCpuHandleSRV(DestDescriptor.ptr);

        if (heap != nullptr)
            heap->ClearByCpuHandle(DestDescriptor.ptr);

        return;
    }

    // if (!CheckResource(pResource))
    //     return;

    auto heap = GetHeapByCpuHandleSRV(DestDescriptor.ptr);
    if (heap != nullptr)
    {
        ResourceInfo resInfo {};
        FillResourceInfo(pResource, &resInfo);
        resInfo.type = SRV;
        resInfo.captureInfo = CaptureInfo::CreateSRV;
        heap->SetByCpuHandle(DestDescriptor.ptr, resInfo);
    }
    // else
    //{
    //     LOG_TRACK("Heap not found for SRV: {:X}", DestDescriptor.ptr);
    // }
}

// A constant buffer view names no resource: the descriptor is a GPU virtual address plus a size. Turning that
// back into bytes is ConstantCapture_Dx12's job -- it already maps every CPU-visible buffer by address range --
// so all that is needed here is to stop throwing the descriptor away.
static std::atomic<uint64_t> g_cbvCreates { 0 };
static std::atomic<uint64_t> g_cbvRecorded { 0 };

uint64_t ResTrack_Dx12::CbvCreateCount() { return g_cbvCreates.load(std::memory_order_relaxed); }
uint64_t ResTrack_Dx12::CbvRecordedCount() { return g_cbvRecorded.load(std::memory_order_relaxed); }

void ResTrack_Dx12::hkCreateConstantBufferView(ID3D12Device* This, const D3D12_CONSTANT_BUFFER_VIEW_DESC* pDesc,
                                               D3D12_CPU_DESCRIPTOR_HANDLE DestDescriptor)
{
    o_CreateConstantBufferView(This, pDesc, DestDescriptor);
    DuniaTaa_Dx12::NoteDescriptorWrite(DestDescriptor.ptr, 1);

    // Installed AND recording for FC6 always, from startup: the game creates its descriptors at load and copies
    // them into the shader-visible heaps each frame, so a record started later (it used to wait for diagnostics
    // or the probe) knows none of them. About one CBV per Present in gameplay.
    if (!DuniaTaa_Dx12::WantsCbvTracking())
        return;

    if (g_cbvCreates.fetch_add(1, std::memory_order_relaxed) == 0)
        LOG_INFO("ResTrack: CreateConstantBufferView is reaching the hook, so the game does create CBVs");

    auto heap = GetHeapByCpuHandleCBV(DestDescriptor.ptr);
    if (heap == nullptr)
    {
        // The other half of the ambiguity: called, but into a heap that was never tracked -- created before our
        // hooks went on, or of a type CreateDescriptorHeap turns away. Said once, so it can never be mistaken
        // for the game not creating any.
        static std::atomic<bool> announced { false };
        if (!announced.exchange(true))
            LOG_WARN("ResTrack: a constant buffer view was created into an untracked heap, so it cannot be "
                     "resolved later (handle {:X})",
                     (size_t) DestDescriptor.ptr);
        return;
    }

    // A null desc creates a null descriptor: the slot now holds nothing, and a stale record of what used to be
    // there would answer a later walk with a buffer the game has moved on from.
    if (pDesc == nullptr || pDesc->BufferLocation == 0)
    {
        heap->ClearByCpuHandle(DestDescriptor.ptr);
        return;
    }

    ResourceInfo resInfo {};
    resInfo.buffer = nullptr;
    resInfo.cbvAddress = pDesc->BufferLocation;
    resInfo.width = pDesc->SizeInBytes;
    resInfo.height = 0;
    resInfo.format = DXGI_FORMAT_UNKNOWN;
    resInfo.type = CBV;
    heap->SetByCpuHandle(DestDescriptor.ptr, resInfo);
    g_cbvRecorded.fetch_add(1, std::memory_order_relaxed);
}

void ResTrack_Dx12::hkCreateUnorderedAccessView(ID3D12Device* This, ID3D12Resource* pResource,
                                                ID3D12Resource* pCounterResource,
                                                D3D12_UNORDERED_ACCESS_VIEW_DESC* pDesc,
                                                D3D12_CPU_DESCRIPTOR_HANDLE DestDescriptor)
{
    DuniaTaa_Dx12::NoteDescriptorWrite(DestDescriptor.ptr, 1);

    if (pResource != nullptr && pDesc != nullptr && Config::Instance()->ForceHDR.value_or_default())
    {
        for (size_t i = 0; i < State::Instance().scBuffers.size(); i++)
        {
            if (State::Instance().scBuffers[i] == pResource)
            {
                if (Config::Instance()->UseHDR10.value_or_default())
                    pDesc->Format = DXGI_FORMAT_R10G10B10A2_UNORM;
                else
                    pDesc->Format = DXGI_FORMAT_R16G16B16A16_FLOAT;

                break;
            }
        }
    }

    // Every unordered access view the game makes passes through here, which is the one place a
    // buffer shaped like an exposure can be spotted without knowing anything about the game. Silent
    // and cheap for everything that does not match, and inert unless the scan is switched on.
    DlssNr::ExposureScan::NoteUav(pResource, pDesc);

    o_CreateUnorderedAccessView(This, pResource, pCounterResource, pDesc, DestDescriptor);

    if (Config::Instance()->FGHudfixDisableUAV.value_or_default())
        return;

    if (pResource == nullptr || !CheckResource(pResource))
    {
        auto heap = GetHeapByCpuHandleUAV(DestDescriptor.ptr);

        if (heap != nullptr)
            heap->ClearByCpuHandle(DestDescriptor.ptr);

        return;
    }

    // if (!CheckResource(pResource))
    //     return;

    auto heap = GetHeapByCpuHandleUAV(DestDescriptor.ptr);
    if (heap != nullptr)
    {
        ResourceInfo resInfo {};
        FillResourceInfo(pResource, &resInfo);
        resInfo.type = UAV;
        resInfo.captureInfo = CaptureInfo::CreateUAV;
        heap->SetByCpuHandle(DestDescriptor.ptr, resInfo);
    }

    if (DuniaTaa_Dx12::Active())
        DuniaTaa_Dx12::NoteMotion(pResource, false);
}

#pragma endregion

void ResTrack_Dx12::hkExecuteCommandLists(ID3D12CommandQueue* This, UINT NumCommandLists,
                                          ID3D12CommandList* const* ppCommandLists)
{
    // FC6 same-eye vectors: constants the lists about to be submitted will read are patched here, the last
    // moment that is certainly before the GPU sees them. One atomic load when the feature is off.
    DuniaTaa_Dx12::BeforeExecute();
    // The probes' submit order (the upscale writer probe and the velocity-reader probe). NoteExecute returns on an
    // atomic load when neither runs; gating it here on the writer probe alone left the velocity-reader probe with
    // 0 submits and every before/after unresolved (4d4b5c7).
    DuniaTaa_Dx12::NoteExecute(ppCommandLists, NumCommandLists);
    GiCapture_Dx12::OnExecute(This, ppCommandLists, NumCommandLists);
    thread_local std::vector<ID3D12CommandList*> spliced;
    if (GiCapture_Dx12::SpliceExecute(This, ppCommandLists, NumCommandLists, spliced))
    {
        ppCommandLists = spliced.data();
        NumCommandLists = (UINT) spliced.size();
    }

    auto fg = State::Instance().currentFG;

    if (fg != nullptr && fg->IsActive() && !fg->IsPaused())
    {
        LOG_TRACK("NumCommandLists: {}", NumCommandLists);

        std::vector<FG_ResourceType> found;
        auto fIndex = fg->GetIndex();

        do
        {
            std::lock_guard<std::mutex> lock2(_resourceCommandListMutex);

            if (!_notFoundCmdLists.empty())
            {
                for (size_t i = 0; i < NumCommandLists; i++)
                {
                    if (_notFoundCmdLists.contains(ppCommandLists[i]))
                    {
                        LOG_WARN("Found last frames cmdList: {:X}", (size_t) ppCommandLists[i]);
                        _notFoundCmdLists.erase(ppCommandLists[i]);
                    }
                }
            }

            if (_resCmdList[fIndex].empty())
                break;

            for (size_t i = 0; i < NumCommandLists; i++)
            {
                LOG_TRACK("ppCommandLists[{}]: {:X}", i, (size_t) ppCommandLists[i]);

                for (const auto& pair : _resCmdList[fIndex])
                {
                    if (pair.second == ppCommandLists[i])
                    {
                        LOG_DEBUG("found {} cmdList: {:X}, queue: {:X}", (UINT) pair.first, (size_t) pair.second,
                                  (size_t) This);
                        fg->SetResourceReady(pair.first);
                        found.push_back(pair.first);
                    }
                }

                for (size_t i = 0; i < found.size(); i++)
                {
                    _resCmdList[fIndex].erase(found[i]);
                }

                if (_resCmdList[fIndex].empty())
                    break;
            }

        } while (false);

        if (!found.empty())
        {
            o_ExecuteCommandLists(This, NumCommandLists, ppCommandLists);
            DlssNr::FinishedPictureSubmitted(This, NumCommandLists, ppCommandLists);

            for (size_t i = 0; i < found.size(); i++)
            {
                fg->SetCommandQueue(found[i], This);
            }

            return;
        }
    }

    LOG_TRACK("Done NumCommandLists: {}", NumCommandLists);

    o_ExecuteCommandLists(This, NumCommandLists, ppCommandLists);
    DlssNr::FinishedPictureSubmitted(This, NumCommandLists, ppCommandLists);
}

#pragma region Heap hooks

static ULONG STDMETHODCALLTYPE hkHeapRelease(ID3D12DescriptorHeap* This)
{
    if (State::Instance().isShuttingDown)
        return o_HeapRelease(This);

    std::shared_ptr<HeapInfo> heapInfo;
    {
        std::shared_lock lock(_heapRegistryMutex);
        for (const auto& heap : fgHeaps)
        {
            if (heap != nullptr && heap->heap == This && heap->active.load(std::memory_order_acquire))
            {
                heapInfo = heap;
                break;
            }
        }
    }

    if (heapInfo == nullptr)
        return o_HeapRelease(This);

    This->AddRef();
    if (o_HeapRelease(This) <= 1)
    {
        bool deactivated = false;
        {
            std::unique_lock lock(_heapRegistryMutex);
            deactivated = heapInfo->DeactivateAndClear();
        }

        if (deactivated)
        {
            LOG_INFO("Heap released: {:X}", (size_t) This);
            gHeapGeneration.fetch_add(1, std::memory_order_release);
        }
    }

    return o_HeapRelease(This);
}

HRESULT ResTrack_Dx12::hkCreateDescriptorHeap(ID3D12Device* This, D3D12_DESCRIPTOR_HEAP_DESC* pDescriptorHeapDesc,
                                              REFIID riid, void** ppvHeap)
{
    auto result = o_CreateDescriptorHeap(This, pDescriptorHeapDesc, riid, ppvHeap);

    if (State::Instance().skipHeapCapture)
        return result;

    // try to calculate handle ranges for heap
    if (result == S_OK && (pDescriptorHeapDesc->Type == D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV ||
                           pDescriptorHeapDesc->Type == D3D12_DESCRIPTOR_HEAP_TYPE_RTV ||
                           (DuniaTaa_Dx12::Active() && pDescriptorHeapDesc->Type == D3D12_DESCRIPTOR_HEAP_TYPE_DSV)))
    {
        auto heap = (ID3D12DescriptorHeap*) (*ppvHeap);

        if (!o_HeapRelease)
        {
            PVOID* vtbl = *(PVOID**) heap;
            o_HeapRelease = (PFN_HeapRelease) vtbl[2];
            DetourTransactionBegin();
            DetourUpdateThread(GetCurrentThread());
            DetourAttach(&(PVOID&) o_HeapRelease, hkHeapRelease);
            auto detourResult = DetourTransactionCommit();
            if (detourResult != NO_ERROR)
            {
                LOG_ERROR("Failed to hook Heap Release: {:X}", detourResult);
                o_HeapRelease = nullptr;
            }
        }

        auto increment = This->GetDescriptorHandleIncrementSize(pDescriptorHeapDesc->Type);
        auto numDescriptors = pDescriptorHeapDesc->NumDescriptors;
        auto cpuStart = (SIZE_T) (heap->GetCPUDescriptorHandleForHeapStart().ptr);
        auto cpuEnd = cpuStart + (increment * numDescriptors);
        auto gpuStart = (SIZE_T) (heap->GetGPUDescriptorHandleForHeapStart().ptr);
        auto gpuEnd = gpuStart + (increment * numDescriptors);
        auto type = (UINT) pDescriptorHeapDesc->Type;

        if (pDescriptorHeapDesc->Type == D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV &&
            (pDescriptorHeapDesc->Flags & D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE) != 0)
            DuniaTaa_Dx12::NoteShaderVisibleHeap(heap, cpuStart, gpuStart, numDescriptors, increment);

        LOG_TRACE("Heap: {:X}, Heap type: {}, Cpu: {}-{}, Gpu: {}-{}, Desc count: {}", (size_t) *ppvHeap, type,
                  cpuStart, cpuEnd, gpuStart, gpuEnd, numDescriptors);
        {
            std::unique_lock lock(_heapRegistryMutex);
            size_t count = fgHeaps.size();
            bool foundEmpty = false;
            for (size_t i = 0; i < count; i++)
            {
                if (fgHeaps[i] != nullptr && !fgHeaps[i]->active.load(std::memory_order_acquire))
                {

                    fgHeaps[i] = std::make_shared<HeapInfo>(heap, cpuStart, cpuEnd, gpuStart, gpuEnd, numDescriptors,
                                                            increment, type);

                    gHeapGeneration.fetch_add(1, std::memory_order_release);
                    foundEmpty = true;
                    LOG_DEBUG("Reusing empty heap slot: {}", i);
                    break;
                }
            }

            if (!foundEmpty)
            {
                // Reallocate vector if needed
                if (fgHeaps.capacity() == fgHeaps.size())
                    fgHeaps.reserve(fgHeaps.size() + 65536);

                fgHeaps.push_back(std::make_shared<HeapInfo>(heap, cpuStart, cpuEnd, gpuStart, gpuEnd, numDescriptors,
                                                             increment, type));

                gHeapGeneration.fetch_add(1, std::memory_order_release);
                LOG_DEBUG("Adding new heap slot: {}", fgHeaps.size() - 1);
            }
        }
    }
    else
    {
        if (ppvHeap != nullptr && *ppvHeap != nullptr)
        {
            auto heap = (ID3D12DescriptorHeap*) (*ppvHeap);
            LOG_TRACE("Skipping, Heap type: {}, Cpu: {}, Gpu: {}", (UINT) pDescriptorHeapDesc->Type,
                      heap->GetCPUDescriptorHandleForHeapStart().ptr, heap->GetGPUDescriptorHandleForHeapStart().ptr);
        }
    }

    return result;
}

ULONG ResTrack_Dx12::hkRelease(ID3D12Resource* This)
{
    if (State::Instance().isShuttingDown)
        return o_Release(This);

    std::vector<TrackedResourceSlot> toClean;
    bool dying = false;
    {
        std::lock_guard lock(_trackedResourcesMutex);

        This->AddRef();
        auto refCount = o_Release(This);

        if (refCount <= 1)
        {
            dying = true;

            if (auto it = _trackedResources.find(This); it != _trackedResources.end())
            {
                toClean = std::move(it->second);
                _trackedResources.erase(it);
            }

            State::Instance().capturedHudlesses.erase(This);
        }
    }

    // Clean descriptor slots outside the reverse-index lock.
    for (const auto& slot : toClean)
    {
        if (auto heap = slot.heap.lock())
            heap->ClearSlotIfMatches(slot.index, This);
    }

    // A descriptor slot is not the only thing that can outlive a resource: ConstantCapture keeps a persistent
    // CPU MAPPING of every buffer the game maps, and Far Cry 6's matrix motion caches raw pointers into those
    // mappings. Map is reference counted and does not keep the resource alive, so both outlive it -- which is
    // why matrix motion broke after a menu and F4 then crashed. A menu is a size change, and a size change is
    // when the game frees and reallocates.
    //
    // This runs BEFORE the release below, while the resource is still alive, so the pointers are gone before
    // the pages are. The alternative -- checking whether the cached contents still look right -- cannot work:
    // that read is the crash. Nothing here takes a reference to keep the resource alive; the point is to
    // notice the death, not to prevent it.
    if (dying)
    {
        ConstantCapture_Dx12::ForgetResource(This);
        // And the learned motion textures, which are the same class of raw pointer: FC6's menu crash of
        // One crash was AfterDispatch barriering one of these after the game had freed it.
        DuniaTaa_Dx12::ForgetResource(This);
    }

    return o_Release(This);
}

void ResTrack_Dx12::hkCopyDescriptors(ID3D12Device* This, UINT NumDestDescriptorRanges,
                                      D3D12_CPU_DESCRIPTOR_HANDLE* pDestDescriptorRangeStarts,
                                      UINT* pDestDescriptorRangeSizes, UINT NumSrcDescriptorRanges,
                                      D3D12_CPU_DESCRIPTOR_HANDLE* pSrcDescriptorRangeStarts,
                                      UINT* pSrcDescriptorRangeSizes, D3D12_DESCRIPTOR_HEAP_TYPE DescriptorHeapsType)
{
    o_CopyDescriptors(This, NumDestDescriptorRanges, pDestDescriptorRangeStarts, pDestDescriptorRangeSizes,
                      NumSrcDescriptorRanges, pSrcDescriptorRangeStarts, pSrcDescriptorRangeSizes, DescriptorHeapsType);

    // FC6, before every early exit below: which shader-visible slots the game writes, from startup, and which CPU
    // descriptor each was copied from. Destination and source ranges split independently, so walk them in step
    // and hand over each run that is contiguous on both sides.
    if (DescriptorHeapsType == D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV && pDestDescriptorRangeStarts != nullptr)
    {
        const UINT inc = This->GetDescriptorHandleIncrementSize(DescriptorHeapsType);
        UINT d = 0, dOff = 0, s = 0, sOff = 0;
        while (d < NumDestDescriptorRanges)
        {
            const UINT dSize = pDestDescriptorRangeSizes != nullptr ? pDestDescriptorRangeSizes[d] : 1;
            if (dOff >= dSize)
            {
                ++d;
                dOff = 0;
                continue;
            }
            UINT run = dSize - dOff;
            SIZE_T src = 0;
            if (pSrcDescriptorRangeStarts != nullptr && s < NumSrcDescriptorRanges)
            {
                const UINT sSize = pSrcDescriptorRangeSizes != nullptr ? pSrcDescriptorRangeSizes[s] : 1;
                if (sOff >= sSize)
                {
                    ++s;
                    sOff = 0;
                    continue;
                }
                run = std::min(run, sSize - sOff);
                src = pSrcDescriptorRangeStarts[s].ptr + (SIZE_T) sOff * inc;
                sOff += run;
            }
            DuniaTaa_Dx12::NoteDescriptorWrite(pDestDescriptorRangeStarts[d].ptr + (SIZE_T) dOff * inc, run, src);
            dOff += run;
        }
    }

    // Early exit conditions - consistent validation
    if (DescriptorHeapsType != D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV &&
        DescriptorHeapsType != D3D12_DESCRIPTOR_HEAP_TYPE_RTV)
        return;

    if (NumDestDescriptorRanges == 0 || pDestDescriptorRangeStarts == nullptr)
        return;

    if (!Config::Instance()->FGAlwaysTrackHeaps.value_or_default() && !IsHudFixActive() &&
        !ResTrack_Dx12::ForcedHeapTracking())
        return;

    const UINT inc = This->GetDescriptorHandleIncrementSize(DescriptorHeapsType);

    // Skipped before any heap lookup: the lookup is the cost. A call whose destinations all miss the heaps
    // of interest is not mirrored at all.
    {
        bool wanted = false;
        for (UINT r = 0; r < NumDestDescriptorRanges && !wanted; ++r)
        {
            const SIZE_T begin = pDestDescriptorRangeStarts[r].ptr;
            const UINT count = pDestDescriptorRangeSizes != nullptr ? pDestDescriptorRangeSizes[r] : 1;
            wanted = ResTrack_Dx12::HeapRangeWanted(begin, begin + (SIZE_T) count * inc);
        }
        if (!wanted)
            return;
    }

    // Validate that we have source descriptors to copy
    bool haveSources = (NumSrcDescriptorRanges > 0 && pSrcDescriptorRangeStarts != nullptr);

    // Track positions in both source and destination ranges
    UINT srcRangeIndex = 0;
    UINT srcOffsetInRange = 0;
    UINT destRangeIndex = 0;
    UINT destOffsetInRange = 0;

    // Cache for heap lookups to avoid repeated lookups within the same range
    std::shared_ptr<HeapInfo> cachedDestHeap;
    SIZE_T cachedDestRangeStart = 0;
    UINT cachedDestRangeSize = 0;
    std::shared_ptr<HeapInfo> cachedSrcHeap;
    SIZE_T cachedSrcRangeStart = 0;
    UINT cachedSrcRangeSize = 0;

    // Process all destination descriptors
    while (destRangeIndex < NumDestDescriptorRanges)
    {
        // Update destination heap cache if we've moved to a new range
        if (destOffsetInRange == 0)
        {
            cachedDestRangeStart = pDestDescriptorRangeStarts[destRangeIndex].ptr;
            cachedDestRangeSize =
                (pDestDescriptorRangeSizes == nullptr) ? 1 : pDestDescriptorRangeSizes[destRangeIndex];
            cachedDestHeap = GetHeapByCpuHandle(cachedDestRangeStart);
        }

        // Calculate current destination handle
        const SIZE_T destHandle = cachedDestRangeStart + (static_cast<SIZE_T>(destOffsetInRange) * inc);

        // Get or update source information
        ResourceInfo srcInfo {};
        bool haveSrcInfo = false;
        if (haveSources && srcRangeIndex < NumSrcDescriptorRanges)
        {
            // Update source heap cache if we've moved to a new range
            if (srcOffsetInRange == 0)
            {
                cachedSrcRangeStart = pSrcDescriptorRangeStarts[srcRangeIndex].ptr;
                cachedSrcRangeSize =
                    (pSrcDescriptorRangeSizes == nullptr) ? 1 : pSrcDescriptorRangeSizes[srcRangeIndex];
                cachedSrcHeap = GetHeapByCpuHandle(cachedSrcRangeStart);
            }

            // Calculate current source handle
            const SIZE_T srcHandle = cachedSrcRangeStart + (static_cast<SIZE_T>(srcOffsetInRange) * inc);

            if (cachedSrcHeap != nullptr)
                haveSrcInfo = cachedSrcHeap->GetByCpuHandle(srcHandle, srcInfo);

            // Advance source position
            srcOffsetInRange++;
            if (srcOffsetInRange >= cachedSrcRangeSize)
            {
                srcOffsetInRange = 0;
                srcRangeIndex++;
            }
        }

        if (cachedDestHeap != nullptr)
        {
            if (haveSrcInfo)
                cachedDestHeap->SetByCpuHandle(destHandle, srcInfo);
            else
                cachedDestHeap->ClearByCpuHandle(destHandle);
        }

        // Advance destination position
        destOffsetInRange++;
        if (destOffsetInRange >= cachedDestRangeSize)
        {
            destOffsetInRange = 0;
            destRangeIndex++;
        }
    }
}

void ResTrack_Dx12::hkCopyDescriptorsSimple(ID3D12Device* This, UINT NumDescriptors,
                                            D3D12_CPU_DESCRIPTOR_HANDLE DestDescriptorRangeStart,
                                            D3D12_CPU_DESCRIPTOR_HANDLE SrcDescriptorRangeStart,
                                            D3D12_DESCRIPTOR_HEAP_TYPE DescriptorHeapsType)
{
    o_CopyDescriptorsSimple(This, NumDescriptors, DestDescriptorRangeStart, SrcDescriptorRangeStart,
                            DescriptorHeapsType);

    if (DescriptorHeapsType == D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV)
        DuniaTaa_Dx12::NoteDescriptorWrite(DestDescriptorRangeStart.ptr, NumDescriptors, SrcDescriptorRangeStart.ptr);

    if (DescriptorHeapsType != D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV &&
        DescriptorHeapsType != D3D12_DESCRIPTOR_HEAP_TYPE_RTV)
        return;

    if (!Config::Instance()->FGAlwaysTrackHeaps.value_or_default() && !IsHudFixActive() &&
        !ResTrack_Dx12::ForcedHeapTracking())
        return;

    auto size = This->GetDescriptorHandleIncrementSize(DescriptorHeapsType);

    // One contiguous destination range, so one check covers the call. Before any heap lookup, as above.
    if (!ResTrack_Dx12::HeapRangeWanted(DestDescriptorRangeStart.ptr,
                                       DestDescriptorRangeStart.ptr + (SIZE_T) NumDescriptors * size))
        return;

    for (size_t i = 0; i < NumDescriptors; i++)
    {
        std::shared_ptr<HeapInfo> srcHeap;
        SIZE_T srcHandle = 0;

        // source
        if (SrcDescriptorRangeStart.ptr != 0)
        {
            srcHandle = SrcDescriptorRangeStart.ptr + i * size;
            srcHeap = GetHeapByCpuHandle(srcHandle);
        }

        auto destHandle = DestDescriptorRangeStart.ptr + i * size;
        auto dstHeap = GetHeapByCpuHandle(destHandle);

        // destination
        if (dstHeap == nullptr)
            continue;

        if (srcHeap == nullptr)
        {
            dstHeap->ClearByCpuHandle(destHandle);
            continue;
        }

        ResourceInfo buffer {};
        if (!srcHeap->GetByCpuHandle(srcHandle, buffer))
        {
            dstHeap->ClearByCpuHandle(destHandle);
            continue;
        }

        dstHeap->SetByCpuHandle(destHandle, buffer);
    }
}

#pragma endregion

#pragma region Shader input hooks

void ResTrack_Dx12::hkSetGraphicsRootDescriptorTable(ID3D12GraphicsCommandList* This, UINT RootParameterIndex,
                                                     D3D12_GPU_DESCRIPTOR_HANDLE BaseDescriptor)
{
    // Consistent early exit - always call original function
    auto shouldTrack = !Config::Instance()->FGHudfixDisableSGR.value_or_default() && BaseDescriptor.ptr != 0 &&
                       IsHudFixActive() && !Hudfix_Dx12::SkipHudlessChecks() &&
                       This != MenuOverlayDx::MenuCommandList();

    if (!shouldTrack)
    {
        o_SetGraphicsRootDescriptorTable(This, RootParameterIndex, BaseDescriptor);
        return;
    }

    auto heap = GetHeapByGpuHandleGR(BaseDescriptor.ptr);
    if (heap == nullptr)
    {
        LOG_DEBUG_ONLY("No heap for handle: {:X}", BaseDescriptor.ptr);
        o_SetGraphicsRootDescriptorTable(This, RootParameterIndex, BaseDescriptor);
        return;
    }

    ResourceInfo capturedBuffer {};
    if (!heap->GetByGpuHandle(BaseDescriptor.ptr, capturedBuffer) || capturedBuffer.buffer == nullptr)
    {
        LOG_DEBUG_ONLY("No resource at RootParameterIndex: {}, CommandList: {:X}, gpuHandle: {:X}", RootParameterIndex,
                       (SIZE_T) This, BaseDescriptor.ptr);
        o_SetGraphicsRootDescriptorTable(This, RootParameterIndex, BaseDescriptor);
        return;
    }

    LOG_DEBUG_ONLY("CommandList: {:X}, Resource: {:X}", (size_t) This, (size_t) capturedBuffer.buffer);

    // Only proceed with tracking if we have a valid buffer
    capturedBuffer.state = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
    capturedBuffer.captureInfo = CaptureInfo::SetGR;

    // Track the resource
    bool capturedImmediately = false;
    if (Config::Instance()->FGImmediateCapture.value_or_default())
    {
        capturedImmediately = Hudfix_Dx12::CheckForHudless(This, &capturedBuffer, capturedBuffer.state);
    }

    if (!capturedImmediately)
    {
        auto fIndex = Hudfix_Dx12::ActivePresentFrame() % BUFFER_COUNT;

        if (!_useShards)
        {
            std::lock_guard<std::mutex> lock(_hudlessTrackMutex);

            if (!fgPossibleHudless[fIndex].contains(This))
            {
                ankerl::unordered_dense::map<ID3D12Resource*, ResourceInfo> newMap;
                newMap.reserve(32);
                fgPossibleHudless[fIndex].insert_or_assign(This, std::move(newMap));
            }

            LOG_TRACK("Tracking Resource: {:X}, Desc: {:X}", (size_t) capturedBuffer.buffer, BaseDescriptor.ptr);
            fgPossibleHudless[fIndex][This].insert_or_assign(capturedBuffer.buffer, capturedBuffer);
        }
        else
        {
            size_t shardIdx = GetShardIndex(This);
            auto& shard = _hudlessShards[fIndex][shardIdx];

            LOCK_GUARD(shard.mutex);

            if (!shard.map.contains(This))
            {
                ankerl::unordered_dense::map<ID3D12Resource*, ResourceInfo> newMap;
                newMap.reserve(32);
                shard.map.insert_or_assign(This, std::move(newMap));
            }

            LOG_TRACK("CmdList: {:X}, Tracking Resource: {:X}, Desc: {:X}, Format: {}", (size_t) This,
                      (size_t) capturedBuffer.buffer, BaseDescriptor.ptr, (UINT) capturedBuffer.format);

            shard.map[This].insert_or_assign(capturedBuffer.buffer, capturedBuffer);
        }
    }

    o_SetGraphicsRootDescriptorTable(This, RootParameterIndex, BaseDescriptor);
}

#pragma endregion

#pragma region Shader output hooks

void ResTrack_Dx12::hkOMSetRenderTargets(ID3D12GraphicsCommandList* This, UINT NumRenderTargetDescriptors,
                                         D3D12_CPU_DESCRIPTOR_HANDLE* pRenderTargetDescriptors,
                                         BOOL RTsSingleHandleToDescriptorRange,
                                         D3D12_CPU_DESCRIPTOR_HANDLE* pDepthStencilDescriptor)
{
    if (DuniaTaa_Dx12::Active())
    {
        static bool loggedOmSet = false;
        if (!loggedOmSet)
        {
            loggedOmSet = true;
            LOG_INFO("DuniaTaa: OMSetRenderTargets hooked (rtvs={} depth={})", NumRenderTargetDescriptors,
                     pDepthStencilDescriptor != nullptr);
        }

        ID3D12Resource* color = nullptr;
        ID3D12Resource* depth = nullptr;

        auto resolveRtv = [&](UINT index) -> ID3D12Resource* {
            if (pRenderTargetDescriptors == nullptr || index >= NumRenderTargetDescriptors)
                return nullptr;

            D3D12_CPU_DESCRIPTOR_HANDLE handle {};
            std::shared_ptr<HeapInfo> heap;
            if (RTsSingleHandleToDescriptorRange)
            {
                heap = GetHeapByCpuHandleRTV(pRenderTargetDescriptors[0].ptr);
                if (heap == nullptr)
                    return nullptr;
                handle.ptr = pRenderTargetDescriptors[0].ptr + (index * heap->increment);
            }
            else
            {
                handle = pRenderTargetDescriptors[index];
                heap = GetHeapByCpuHandleRTV(handle.ptr);
            }

            ResourceInfo info {};
            if (heap != nullptr && heap->GetByCpuHandle(handle.ptr, info))
                return info.buffer;
            return nullptr;
        };

        if (NumRenderTargetDescriptors > 0)
            color = resolveRtv(0);

        if (pDepthStencilDescriptor != nullptr)
        {
            auto heap = GetHeapByCpuHandle(pDepthStencilDescriptor->ptr);
            ResourceInfo info {};
            if (heap != nullptr && heap->GetByCpuHandle(pDepthStencilDescriptor->ptr, info))
                depth = info.buffer;
        }

        DuniaTaa_Dx12::OnSetTargets(This, color, depth);
        for (UINT i = 0; i < NumRenderTargetDescriptors; i++)
            DuniaTaa_Dx12::NoteMotion(resolveRtv(i), true, This);

        // FC6's private copy decides "is this the G-buffer" from EVERY slot, not slot 0 alone, and the probe maps
        // passes by their whole target signature. Only while either runs.
        if (DuniaTaa_Dx12::WantsAllTargets())
        {
            ID3D12Resource* targets[D3D12_SIMULTANEOUS_RENDER_TARGET_COUNT] {};
            const UINT count = std::min<UINT>(NumRenderTargetDescriptors, D3D12_SIMULTANEOUS_RENDER_TARGET_COUNT);
            for (UINT i = 0; i < count; i++)
                targets[i] = resolveRtv(i);
            DuniaTaa_Dx12::NoteAllTargets(This, targets, count, depth);
        }

        // The upscale writer probe wants EVERY slot, not slot 0 alone, and wants to know about slots it
        // could not resolve. One atomic load when the probe is off.
        if (DuniaTaa_Dx12::ProbeOn())
        {
            ID3D12Resource* rtvs[8] {};
            UINT unresolved = 0;
            const UINT n = NumRenderTargetDescriptors < 8 ? NumRenderTargetDescriptors : 8;
            for (UINT i = 0; i < n; i++)
            {
                rtvs[i] = resolveRtv(i);
                if (rtvs[i] == nullptr)
                    unresolved++;
            }
            DuniaTaa_Dx12::ProbeOmSet(This, rtvs, n, unresolved, depth);
        }
    }

    // Consistent early exit validation
    auto shouldTrack = !Config::Instance()->FGHudfixDisableOM.value_or_default() && NumRenderTargetDescriptors > 0 &&
                       pRenderTargetDescriptors != nullptr && IsHudFixActive() && !Hudfix_Dx12::SkipHudlessChecks() &&
                       This != MenuOverlayDx::MenuCommandList();

    if (!shouldTrack)
    {
        if (DuniaTaa_Dx12::PendingSplice(This))
        {
            o_OMSetRenderTargets(This, 0, nullptr, FALSE, nullptr);
            DuniaTaa_Dx12::AfterOMSet(This);
        }
        o_OMSetRenderTargets(This, NumRenderTargetDescriptors, pRenderTargetDescriptors,
                             RTsSingleHandleToDescriptorRange, pDepthStencilDescriptor);
        DuniaTaa_Dx12::AfterBound(This);
        return;
    }

    LOG_DEBUG_ONLY("NumRenderTargetDescriptors: {}", NumRenderTargetDescriptors);

    auto fIndex = Hudfix_Dx12::ActivePresentFrame() % BUFFER_COUNT;

    // Process render targets
    for (size_t i = 0; i < NumRenderTargetDescriptors; i++)
    {
        std::shared_ptr<HeapInfo> heap;
        D3D12_CPU_DESCRIPTOR_HANDLE handle {};

        // Get the appropriate handle
        if (RTsSingleHandleToDescriptorRange)
        {
            heap = GetHeapByCpuHandleRTV(pRenderTargetDescriptors[0].ptr);
            if (heap == nullptr)
            {
                LOG_DEBUG_ONLY("No heap at index: {}", i);
                continue;
            }

            handle.ptr = pRenderTargetDescriptors[0].ptr + (i * heap->increment);
        }
        else
        {
            handle = pRenderTargetDescriptors[i];
            heap = GetHeapByCpuHandleRTV(handle.ptr);
            if (heap == nullptr)
            {
                LOG_DEBUG_ONLY("No heap at index: {}", i);
                continue;
            }
        }

        ResourceInfo capturedBuffer {};
        if (!heap->GetByCpuHandle(handle.ptr, capturedBuffer) || capturedBuffer.buffer == nullptr)
        {
            LOG_DEBUG_ONLY("No resource at index: {}, cpu: {:X}", i, handle.ptr);
            continue;
        }

        // Valid resource found, update state
        capturedBuffer.state = D3D12_RESOURCE_STATE_RENDER_TARGET;
        capturedBuffer.captureInfo = CaptureInfo::OMSetRTV;

        // Check for immediate capture
        bool capturedImmediately = false;
        if (Config::Instance()->FGImmediateCapture.value_or_default())
        {
            capturedImmediately = Hudfix_Dx12::CheckForHudless(This, &capturedBuffer, capturedBuffer.state);
            if (capturedImmediately)
                break; // Early exit if captured
        }

        // Track for later processing
        if (!capturedImmediately)
        {
            if (!_useShards)
            {
                std::lock_guard<std::mutex> lock(_hudlessTrackMutex);

                if (!fgPossibleHudless[fIndex].contains(This))
                {
                    ankerl::unordered_dense::map<ID3D12Resource*, ResourceInfo> newMap;
                    newMap.reserve(32);
                    fgPossibleHudless[fIndex].insert_or_assign(This, std::move(newMap));
                }

                LOG_TRACK("Tracking Resource: {:X}, Desc: {:X}", (size_t) capturedBuffer.buffer, handle.ptr);
                fgPossibleHudless[fIndex][This].insert_or_assign(capturedBuffer.buffer, capturedBuffer);
            }
            else
            {
                size_t shardIdx = GetShardIndex(This);
                auto& shard = _hudlessShards[fIndex][shardIdx];

                LOCK_GUARD(shard.mutex);

                if (!shard.map.contains(This))
                {
                    ankerl::unordered_dense::map<ID3D12Resource*, ResourceInfo> newMap;
                    newMap.reserve(32);
                    shard.map.insert_or_assign(This, std::move(newMap));
                }

                LOG_TRACK("CmdList: {:X}, Tracking Resource: {:X}, Desc: {:X}, Format: {}", (size_t) This,
                          (size_t) capturedBuffer.buffer, handle.ptr, (UINT) capturedBuffer.format);

                shard.map[This].insert_or_assign(capturedBuffer.buffer, capturedBuffer);
            }
        }
    }

    if (DuniaTaa_Dx12::PendingSplice(This))
    {
        o_OMSetRenderTargets(This, 0, nullptr, FALSE, nullptr);
        DuniaTaa_Dx12::AfterOMSet(This);
    }
    o_OMSetRenderTargets(This, NumRenderTargetDescriptors, pRenderTargetDescriptors, RTsSingleHandleToDescriptorRange,
                         pDepthStencilDescriptor);
    DuniaTaa_Dx12::AfterBound(This);
}

#pragma endregion

#pragma region Compute paramter hooks

void ResTrack_Dx12::hkSetComputeRootDescriptorTable(ID3D12GraphicsCommandList* This, UINT RootParameterIndex,
                                                    D3D12_GPU_DESCRIPTOR_HANDLE BaseDescriptor)
{
    // Consistent early exit - always call original function
    auto shouldTrack = !Config::Instance()->FGHudfixDisableSCR.value_or_default() && BaseDescriptor.ptr != 0 &&
                       IsHudFixActive() && !Hudfix_Dx12::SkipHudlessChecks() &&
                       This != MenuOverlayDx::MenuCommandList();

    if (!shouldTrack)
    {
        o_SetComputeRootDescriptorTable(This, RootParameterIndex, BaseDescriptor);
        return;
    }

    auto heap = GetHeapByGpuHandleCR(BaseDescriptor.ptr);
    if (heap == nullptr)
    {
        LOG_DEBUG_ONLY("No heap for handle: {:X}", BaseDescriptor.ptr);
        o_SetComputeRootDescriptorTable(This, RootParameterIndex, BaseDescriptor);
        return;
    }

    ResourceInfo capturedBuffer {};
    if (!heap->GetByGpuHandle(BaseDescriptor.ptr, capturedBuffer) || capturedBuffer.buffer == nullptr)
    {
        LOG_DEBUG_ONLY("No resource at RootParameterIndex: {}, CommandList: {:X}, gpuHandle: {:X}", RootParameterIndex,
                       (SIZE_T) This, BaseDescriptor.ptr);
        o_SetComputeRootDescriptorTable(This, RootParameterIndex, BaseDescriptor);
        return;
    }

    LOG_DEBUG_ONLY("CommandList: {:X}, Resource: {:X}", (size_t) This, (size_t) capturedBuffer.buffer);

    // Only proceed with tracking if we have a valid buffer
    if (capturedBuffer.type == UAV)
        capturedBuffer.state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    else
        capturedBuffer.state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

    capturedBuffer.captureInfo = CaptureInfo::SetCR;

    // Track the resource
    bool capturedImmediately = false;
    if (Config::Instance()->FGImmediateCapture.value_or_default())
    {
        capturedImmediately = Hudfix_Dx12::CheckForHudless(This, &capturedBuffer, capturedBuffer.state);
    }

    if (!capturedImmediately)
    {
        auto fIndex = Hudfix_Dx12::ActivePresentFrame() % BUFFER_COUNT;

        if (!_useShards)
        {
            std::lock_guard<std::mutex> lock(_hudlessTrackMutex);

            if (!fgPossibleHudless[fIndex].contains(This))
            {
                ankerl::unordered_dense::map<ID3D12Resource*, ResourceInfo> newMap;
                newMap.reserve(32);
                fgPossibleHudless[fIndex].insert_or_assign(This, std::move(newMap));
            }

            LOG_TRACK("Tracking Resource: {:X}, Desc: {:X}", (size_t) capturedBuffer.buffer, BaseDescriptor.ptr);
            fgPossibleHudless[fIndex][This].insert_or_assign(capturedBuffer.buffer, capturedBuffer);
        }
        else
        {
            size_t shardIdx = GetShardIndex(This);
            auto& shard = _hudlessShards[fIndex][shardIdx];

            LOCK_GUARD(shard.mutex);

            if (!shard.map.contains(This))
            {
                ankerl::unordered_dense::map<ID3D12Resource*, ResourceInfo> newMap;
                newMap.reserve(32);
                shard.map.insert_or_assign(This, std::move(newMap));
            }

            LOG_TRACK("CmdList: {:X}, Tracking Resource: {:X}, Desc: {:X}, Format: {}", (size_t) This,
                      (size_t) capturedBuffer.buffer, BaseDescriptor.ptr, (UINT) capturedBuffer.format);

            shard.map[This].insert_or_assign(capturedBuffer.buffer, capturedBuffer);
        }
    }

    o_SetComputeRootDescriptorTable(This, RootParameterIndex, BaseDescriptor);
}

#pragma endregion

#pragma region Shader finalizer hooks

// Capture if render target matches, wait for DrawIndexed
void ResTrack_Dx12::hkDrawInstanced(ID3D12GraphicsCommandList* This, UINT VertexCountPerInstance, UINT InstanceCount,
                                    UINT StartVertexLocation, UINT StartInstanceLocation)
{
    o_DrawInstanced(This, VertexCountPerInstance, InstanceCount, StartVertexLocation, StartInstanceLocation);

    if (!IsHudFixActive())
    {
        LOG_TRACK("Skipping {:X}", (size_t) This);
        return;
    }

    LOG_TRACK("CmdList: {:X}", (size_t) This);

    auto fIndex = Hudfix_Dx12::ActivePresentFrame() % BUFFER_COUNT;

    if (!_useShards)
    {
        if (This == MenuOverlayDx::MenuCommandList())
        {
            std::lock_guard<std::mutex> lock(_hudlessTrackMutex);
            fgPossibleHudless[fIndex].erase(This);
            return;
        }

        ankerl::unordered_dense::map<ID3D12Resource*, ResourceInfo> val0;
        {
            std::lock_guard<std::mutex> lock(_hudlessTrackMutex);

            if (fgPossibleHudless[fIndex].size() == 0 || !fgPossibleHudless[fIndex].contains(This))
                return;

            val0 = std::move(fgPossibleHudless[fIndex][This]);
            fgPossibleHudless[fIndex].erase(This);
        }

        do
        {
            // if this command list does not have entries skip
            if (val0.size() == 0)
                break;

            if (Config::Instance()->FGHudfixDisableDI.value_or_default())
                break;

            std::lock_guard<std::mutex> lock(_drawMutex);
            for (auto& [key, val] : val0)
            {
                val.captureInfo |= CaptureInfo::DrawInstanced;

                if (Hudfix_Dx12::CheckForHudless(This, &val, val.state))
                    break;
            }

        } while (false);
    }
    else
    {
        size_t shardIdx = GetShardIndex(This);
        auto& shard = _hudlessShards[fIndex][shardIdx];

        if (This == MenuOverlayDx::MenuCommandList() && shard.map.contains(This))
        {
            LOCK_GUARD(shard.mutex);

            shard.map.erase(This);
            return;
        }

        ankerl::unordered_dense::map<ID3D12Resource*, ResourceInfo> val0;
        {

            LOCK_GUARD(shard.mutex);

            // if can't find output skip
            if (shard.map.size() == 0)
            {
                LOG_DEBUG_ONLY("Early exit");
                return;
            }

            if (!shard.map.contains(This))
                return;

            val0 = std::move(shard.map[This]);
            shard.map.erase(This);
        }

        do
        {
            // if this command list does not have entries skip
            if (val0.size() == 0)
                break;

            if (Config::Instance()->FGHudfixDisableDI.value_or_default())
                break;

            for (auto& [key, val] : val0)
            {
                std::lock_guard<std::mutex> lock(_drawMutex);

                val.captureInfo |= CaptureInfo::DrawInstanced;

                if (Hudfix_Dx12::CheckForHudless(This, &val, val.state))
                    break;
            }

        } while (false);
    }
}

void ResTrack_Dx12::hkDrawIndexedInstanced(ID3D12GraphicsCommandList* This, UINT IndexCountPerInstance,
                                           UINT InstanceCount, UINT StartIndexLocation, INT BaseVertexLocation,
                                           UINT StartInstanceLocation)
{
    o_DrawIndexedInstanced(This, IndexCountPerInstance, InstanceCount, StartIndexLocation, BaseVertexLocation,
                           StartInstanceLocation);

    if (!IsHudFixActive())
    {
        LOG_TRACK("Skipping CmdList: {:X}", (size_t) This);
        return;
    }

    LOG_TRACK("CmdList: {:X}", (size_t) This);

    auto fIndex = Hudfix_Dx12::ActivePresentFrame() % BUFFER_COUNT;

    if (!_useShards)
    {
        if (This == MenuOverlayDx::MenuCommandList())
        {
            std::lock_guard<std::mutex> lock(_hudlessTrackMutex);
            fgPossibleHudless[fIndex].erase(This);
            return;
        }

        ankerl::unordered_dense::map<ID3D12Resource*, ResourceInfo> val0;
        {
            std::lock_guard<std::mutex> lock(_hudlessTrackMutex);

            if (fgPossibleHudless[fIndex].size() == 0 || !fgPossibleHudless[fIndex].contains(This))
                return;

            val0 = std::move(fgPossibleHudless[fIndex][This]);
            fgPossibleHudless[fIndex].erase(This);
        }

        do
        {
            // if this command list does not have entries skip
            if (val0.size() == 0)
                break;

            if (Config::Instance()->FGHudfixDisableDII.value_or_default())
                break;

            std::lock_guard<std::mutex> lock(_drawMutex);
            for (auto& [key, val] : val0)
            {
                val.captureInfo |= CaptureInfo::DrawIndexedInstanced;

                if (Hudfix_Dx12::CheckForHudless(This, &val, val.state))
                    break;
            }

        } while (false);
    }
    else
    {
        size_t shardIdx = GetShardIndex(This);
        auto& shard = _hudlessShards[fIndex][shardIdx];

        if (This == MenuOverlayDx::MenuCommandList() && shard.map.contains(This))
        {
            LOCK_GUARD(shard.mutex);

            shard.map.erase(This);
            return;
        }

        ankerl::unordered_dense::map<ID3D12Resource*, ResourceInfo> val0;
        {
            LOCK_GUARD(shard.mutex);

            // if can't find output skip
            if (shard.map.size() == 0)
            {
                LOG_DEBUG_ONLY("Early exit");
                return;
            }

            if (!shard.map.contains(This))
                return;

            val0 = std::move(shard.map[This]);
            shard.map.erase(This);
        }

        do
        {
            // if this command list does not have entries skip
            if (val0.size() == 0)
                break;

            if (Config::Instance()->FGHudfixDisableDII.value_or_default())
                break;

            for (auto& [key, val] : val0)
            {
                // LOG_DEBUG("Waiting _drawMutex {:X}", (size_t)val.buffer);
                std::lock_guard<std::mutex> lock(_drawMutex);

                val.captureInfo |= CaptureInfo::DrawIndexedInstanced;

                if (Hudfix_Dx12::CheckForHudless(This, &val, val.state))
                    break;
            }

        } while (false);
    }
}

void ResTrack_Dx12::hkExecuteBundle(ID3D12GraphicsCommandList* This, ID3D12GraphicsCommandList* pCommandList)
{
    LOG_FUNC();

    IFGFeature_Dx12* fg = State::Instance().currentFG;
    auto index = fg != nullptr ? fg->GetIndex() : 0;

    {
        std::lock_guard<std::mutex> lock(_resourceCommandListMutex);

        if (fg != nullptr && fg->IsActive() && (_resourceCommandList[index].size() > 0 || !_resCmdList[index].empty()))
        {
            if (_notFoundCmdLists.contains(pCommandList))
                LOG_WARN("Found last frames cmdList: {:X}", (size_t) This);

            auto& frameCmdList = _resourceCommandList[index];
            for (std::unordered_map<FG_ResourceType, ID3D12GraphicsCommandList*>::iterator it = frameCmdList.begin();
                 it != frameCmdList.end(); ++it)
            {
                if (it->second == pCommandList)
                    it->second = This;
            }

            for (std::unordered_map<FG_ResourceType, void*>::iterator it = _resCmdList[index].begin();
                 it != _resCmdList[index].end(); ++it)
            {
                if (it->second == pCommandList)
                    it->second = This;
            }
        }
    }

    o_ExecuteBundle(This, pCommandList);
}

// DuniaTaa: the frame trace's draw counts ([Inputs] DuniaTaaDiagnostics), and the TAA resolve skip
// ([Inputs] DuniaTaaFindTaa). ShouldSkipDraw returning true means the original must NOT be called -- that
// is how the game's own temporal resolve gets replaced rather than merely post-processed.
static void STDMETHODCALLTYPE hkDuniaDrawInstanced(ID3D12GraphicsCommandList* This, UINT VertexCountPerInstance,
                                                   UINT InstanceCount, UINT StartVertexLocation,
                                                   UINT StartInstanceLocation)
{
    DuniaTaa_Dx12::OnDraw(This);
    if (DuniaTaa_Dx12::ShouldSkipDraw(This))
        return;
    o_DrawInstanced(This, VertexCountPerInstance, InstanceCount, StartVertexLocation, StartInstanceLocation);
    DuniaTaa_Dx12::AfterDraw(This);
}

static void STDMETHODCALLTYPE hkDuniaDrawIndexedInstanced(ID3D12GraphicsCommandList* This, UINT IndexCountPerInstance,
                                                          UINT InstanceCount, UINT StartIndexLocation,
                                                          INT BaseVertexLocation, UINT StartInstanceLocation)
{
    DuniaTaa_Dx12::OnDraw(This);
    if (DuniaTaa_Dx12::ShouldSkipDraw(This))
        return;
    o_DrawIndexedInstanced(This, IndexCountPerInstance, InstanceCount, StartIndexLocation, BaseVertexLocation,
                           StartInstanceLocation);
    DuniaTaa_Dx12::AfterDraw(This);
}

static void STDMETHODCALLTYPE hkCopyResource(ID3D12GraphicsCommandList* This, ID3D12Resource* pDstResource,
                                             ID3D12Resource* pSrcResource)
{
    if (DuniaTaa_Dx12::Active())
        DuniaTaa_Dx12::BeforeCopy(This, pDstResource, pSrcResource);
    o_CopyResource(This, pDstResource, pSrcResource);
    if (DuniaTaa_Dx12::Active())
        DuniaTaa_Dx12::AfterCopy(This, pDstResource, pSrcResource);
}

static void STDMETHODCALLTYPE hkCopyTextureRegion(ID3D12GraphicsCommandList* This,
                                                  const D3D12_TEXTURE_COPY_LOCATION* pDst, UINT DstX, UINT DstY,
                                                  UINT DstZ, const D3D12_TEXTURE_COPY_LOCATION* pSrc,
                                                  const D3D12_BOX* pSrcBox)
{
    if (DuniaTaa_Dx12::Active() && pDst != nullptr && pSrc != nullptr)
        DuniaTaa_Dx12::BeforeCopy(This, pDst->pResource, pSrc->pResource);
    o_CopyTextureRegion(This, pDst, DstX, DstY, DstZ, pSrc, pSrcBox);
    if (DuniaTaa_Dx12::Active() && pDst != nullptr && pSrc != nullptr)
        DuniaTaa_Dx12::AfterCopy(This, pDst->pResource, pSrc->pResource);
}

// DuniaTaa's upscale writer probe: the remaining ways a texture can be written that nothing above hooks --
// an MSAA resolve and the clears. Each is one atomic load when the probe is off, which is nearly always.
static void STDMETHODCALLTYPE hkResolveSubresource(ID3D12GraphicsCommandList* This, ID3D12Resource* pDstResource,
                                                   UINT DstSubresource, ID3D12Resource* pSrcResource,
                                                   UINT SrcSubresource, DXGI_FORMAT Format)
{
    if (DuniaTaa_Dx12::ProbeOn())
        DuniaTaa_Dx12::ProbeResolve(This, pDstResource, pSrcResource);
    o_ResolveSubresource(This, pDstResource, DstSubresource, pSrcResource, SrcSubresource, Format);
}

static void STDMETHODCALLTYPE hkClearRenderTargetView(ID3D12GraphicsCommandList* This,
                                                      D3D12_CPU_DESCRIPTOR_HANDLE RenderTargetView,
                                                      const FLOAT ColorRGBA[4], UINT NumRects, const D3D12_RECT* pRects)
{
    if (DuniaTaa_Dx12::ProbeOn())
    {
        if (auto* resource = ResTrack_Dx12::ResourceOfRtv(RenderTargetView.ptr); resource != nullptr)
            DuniaTaa_Dx12::ProbeClear(This, resource, false);
    }
    o_ClearRenderTargetView(This, RenderTargetView, ColorRGBA, NumRects, pRects);
}

static void STDMETHODCALLTYPE hkClearUnorderedAccessViewUint(ID3D12GraphicsCommandList* This,
                                                             D3D12_GPU_DESCRIPTOR_HANDLE ViewGPUHandleInCurrentHeap,
                                                             D3D12_CPU_DESCRIPTOR_HANDLE ViewCPUHandle,
                                                             ID3D12Resource* pResource, const UINT Values[4],
                                                             UINT NumRects, const D3D12_RECT* pRects)
{
    if (DuniaTaa_Dx12::ProbeOn())
        DuniaTaa_Dx12::ProbeClear(This, pResource, true);
    o_ClearUnorderedAccessViewUint(This, ViewGPUHandleInCurrentHeap, ViewCPUHandle, pResource, Values, NumRects, pRects);
}

static void STDMETHODCALLTYPE hkClearUnorderedAccessViewFloat(ID3D12GraphicsCommandList* This,
                                                              D3D12_GPU_DESCRIPTOR_HANDLE ViewGPUHandleInCurrentHeap,
                                                              D3D12_CPU_DESCRIPTOR_HANDLE ViewCPUHandle,
                                                              ID3D12Resource* pResource, const FLOAT Values[4],
                                                              UINT NumRects, const D3D12_RECT* pRects)
{
    if (DuniaTaa_Dx12::ProbeOn())
        DuniaTaa_Dx12::ProbeClear(This, pResource, true);
    o_ClearUnorderedAccessViewFloat(This, ViewGPUHandleInCurrentHeap, ViewCPUHandle, pResource, Values, NumRects,
                                    pRects);
}

HRESULT ResTrack_Dx12::hkClose(ID3D12GraphicsCommandList* This)
{
    auto fg = State::Instance().currentFG;
    auto index = fg != nullptr ? fg->GetIndex() : 0;

    if (fg != nullptr && fg->IsActive() && !fg->IsPaused() && _resourceCommandList[index].size() > 0)
    {
        LOG_TRACK("CmdList: {:X}", (size_t) This);

        std::lock_guard<std::mutex> lock(_resourceCommandListMutex);

        if (_notFoundCmdLists.contains(This))
            LOG_WARN("Found last frames cmdList: {:X}", (size_t) This);

        std::vector<FG_ResourceType> found;

        for (const auto& pair : _resourceCommandList[index])
        {
            if (This == pair.second)
            {
                if (!fg->IsResourceReady(pair.first))
                {
                    LOG_DEBUG("{} cmdList: {:X}", (UINT) pair.first, (size_t) This);
                    _resCmdList[index][pair.first] = pair.second;
                    found.push_back(pair.first);
                }
            }
        }

        for (size_t i = 0; i < found.size(); i++)
        {
            _resourceCommandList[index].erase(found[i]);
        }
    }

    return o_Close(This);
}

void ResTrack_Dx12::hkDispatch(ID3D12GraphicsCommandList* This, UINT ThreadGroupCountX, UINT ThreadGroupCountY,
                               UINT ThreadGroupCountZ)
{
    o_Dispatch(This, ThreadGroupCountX, ThreadGroupCountY, ThreadGroupCountZ);

    if (!IsHudFixActive())
    {
        LOG_TRACK("Skipping {:X}", (size_t) This);
        return;
    }

    LOG_TRACK("CmdList: {:X}", (size_t) This);

    auto fIndex = Hudfix_Dx12::ActivePresentFrame() % BUFFER_COUNT;

    if (!_useShards)
    {
        if (This == MenuOverlayDx::MenuCommandList())
        {
            std::lock_guard<std::mutex> lock(_hudlessTrackMutex);
            fgPossibleHudless[fIndex].erase(This);
            return;
        }

        ankerl::unordered_dense::map<ID3D12Resource*, ResourceInfo> val0;
        {
            std::lock_guard<std::mutex> lock(_hudlessTrackMutex);

            if (fgPossibleHudless[fIndex].size() == 0 || !fgPossibleHudless[fIndex].contains(This))
                return;

            val0 = std::move(fgPossibleHudless[fIndex][This]);
            fgPossibleHudless[fIndex].erase(This);
        }

        do
        {
            // if this command list does not have entries skip
            if (val0.size() == 0)
                break;

            if (Config::Instance()->FGHudfixDisableDispatch.value_or_default())
                break;

            std::lock_guard<std::mutex> lock(_drawMutex);
            for (auto& [key, val] : val0)
            {
                val.captureInfo |= CaptureInfo::Dispatch;

                if (Hudfix_Dx12::CheckForHudless(This, &val, val.state))
                    break;
            }
        } while (false);
    }
    else
    {
        size_t shardIdx = GetShardIndex(This);
        auto& shard = _hudlessShards[fIndex][shardIdx];

        if (This == MenuOverlayDx::MenuCommandList() && shard.map.contains(This))
        {
            LOCK_GUARD(shard.mutex);

            shard.map.erase(This);
            return;
        }

        ankerl::unordered_dense::map<ID3D12Resource*, ResourceInfo> val0;
        {

            LOCK_GUARD(shard.mutex);

            // if can't find output skip
            if (shard.map.size() == 0)
            {
                LOG_DEBUG_ONLY("Early exit");
                return;
            }

            if (!shard.map.contains(This))
                return;

            val0 = std::move(shard.map[This]);
            shard.map.erase(This);
        }

        do
        {
            // if this command list does not have entries skip
            if (val0.size() == 0)
                break;

            if (Config::Instance()->FGHudfixDisableDispatch.value_or_default())
                break;

            for (auto& [key, val] : val0)
            {
                // LOG_DEBUG("Waiting _drawMutex {:X}", (size_t)val.buffer);
                std::lock_guard<std::mutex> lock(_drawMutex);

                val.captureInfo |= CaptureInfo::Dispatch;
                if (Hudfix_Dx12::CheckForHudless(This, &val, val.state))
                {
                    break;
                }
            }
        } while (false);
    }
}

#pragma endregion

void ResTrack_Dx12::HookResource(ID3D12Device* InDevice)
{
    if (o_Release != nullptr)
        return;

    ID3D12Resource* tmp = nullptr;
    auto d = CD3DX12_RESOURCE_DESC::Buffer(4);
    auto heapProps = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);

    HRESULT hr = InDevice->CreateCommittedResource(&heapProps, D3D12_HEAP_FLAG_NONE, &d,
                                                   D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&tmp));

    if (hr == S_OK)
    {
        PVOID* pVTable = *(PVOID**) tmp;
        o_Release = (PFN_Release) pVTable[2];

        // Not only for the capture: matrix motion reads the game's per-view constants out of these mappings
        // every frame, with diagnostics off. Map is far from free, so it still goes on for nothing else.
        if (DuniaTaa_Dx12::WantsBufferMappings() || Config::Instance()->GiBufferCapture.value_or_default() || Config::Instance()->PerEyeLighting.value_or_default())
        {
            o_ResourceMap = (PFN_ResourceMap) pVTable[8];
            if (o_ResourceMap != nullptr)
            {
                DetourTransactionBegin();
                DetourUpdateThread(GetCurrentThread());
                DetourAttach(&(PVOID&) o_ResourceMap, hkResourceMap);
                const auto mapHookResult = DetourTransactionCommit();
                LOG_INFO("ResTrack: ID3D12Resource::Map hook {}",
                         mapHookResult == NO_ERROR ? "attached" : "FAILED");
                if (mapHookResult != NO_ERROR)
                    o_ResourceMap = nullptr;
            }
            o_ResourceUnmap = (PFN_ResourceUnmap) pVTable[9];
            if (o_ResourceUnmap != nullptr)
            {
                DetourTransactionBegin();
                DetourUpdateThread(GetCurrentThread());
                DetourAttach(&(PVOID&) o_ResourceUnmap, hkResourceUnmap);
                const auto unmapHookResult = DetourTransactionCommit();
                LOG_INFO("ResTrack: ID3D12Resource::Unmap hook {}",
                         unmapHookResult == NO_ERROR ? "attached" : "FAILED");
                if (unmapHookResult != NO_ERROR)
                    o_ResourceUnmap = nullptr;
            }
        }

        if (o_Release != nullptr)
        {
            DetourTransactionBegin();
            DetourUpdateThread(GetCurrentThread());
            DetourAttach(&(PVOID&) o_Release, hkRelease);
            auto detourResult = DetourTransactionCommit();

            if (detourResult != NO_ERROR)
            {
                LOG_ERROR("Failed to hook Heap Release: {:X}", detourResult);
                o_Release = nullptr;
                tmp->Release();
            }
            else
            {
                o_Release(tmp); // drop temp
            }
        }
        else
        {
            tmp->Release();
        }
    }
}

void ResTrack_Dx12::HookCommandList(ID3D12Device* InDevice)
{

    if (o_OMSetRenderTargets != nullptr)
        return;

    ID3D12GraphicsCommandList* commandList = nullptr;
    ID3D12CommandAllocator* commandAllocator = nullptr;

    if (InDevice->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&commandAllocator)) == S_OK)
    {
        if (InDevice->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, commandAllocator, nullptr,
                                        IID_PPV_ARGS(&commandList)) == S_OK)
        {
            ID3D12GraphicsCommandList* realCL = nullptr;
            if (!CheckForRealObject(__FUNCTION__, commandList, (IUnknown**) &realCL))
                realCL = commandList;

            // Get the vtable pointer
            PVOID* pVTable = *(PVOID**) realCL;

            // hudless shader
            o_OMSetRenderTargets = (PFN_OMSetRenderTargets) pVTable[46];
            o_SetGraphicsRootDescriptorTable = (PFN_SetGraphicsRootDescriptorTable) pVTable[32];

            o_DrawInstanced = (PFN_DrawInstanced) pVTable[12];
            o_DrawIndexedInstanced = (PFN_DrawIndexedInstanced) pVTable[13];
            o_Dispatch = (PFN_Dispatch) pVTable[14];
            o_Close = (PFN_Close) pVTable[9];

            // hudless compute
            o_SetComputeRootDescriptorTable = (PFN_SetComputeRootDescriptorTable) pVTable[31];

            o_ExecuteBundle = (PFN_ExecuteBundle) pVTable[27];

            if (o_OMSetRenderTargets != nullptr)
            {
                DetourTransactionBegin();
                DetourUpdateThread(GetCurrentThread());

                // Hudfix needs the full draw/dispatch set. Dunia TAA only needs OMSet.
                if (State::Instance().activeFgInput == FGInput::Upscaler)
                {
                    if (o_OMSetRenderTargets != nullptr)
                        DetourAttach(&(PVOID&) o_OMSetRenderTargets, hkOMSetRenderTargets);

                    if (o_SetGraphicsRootDescriptorTable != nullptr)
                        DetourAttach(&(PVOID&) o_SetGraphicsRootDescriptorTable, hkSetGraphicsRootDescriptorTable);

                    if (o_SetComputeRootDescriptorTable != nullptr)
                        DetourAttach(&(PVOID&) o_SetComputeRootDescriptorTable, hkSetComputeRootDescriptorTable);

                    if (o_DrawIndexedInstanced != nullptr)
                        DetourAttach(&(PVOID&) o_DrawIndexedInstanced, hkDrawIndexedInstanced);

                    if (o_DrawInstanced != nullptr)
                        DetourAttach(&(PVOID&) o_DrawInstanced, hkDrawInstanced);

                    if (o_Dispatch != nullptr)
                        DetourAttach(&(PVOID&) o_Dispatch, hkDispatch);
                }
                else if (DuniaTaa_Dx12::WantHooks() && o_OMSetRenderTargets != nullptr)
                {
                    DetourAttach(&(PVOID&) o_OMSetRenderTargets, hkOMSetRenderTargets);
                    o_CopyTextureRegion = (PFN_CopyTextureRegion) pVTable[16];
                    o_CopyResource = (PFN_CopyResource) pVTable[17];
                    if (o_CopyResource != nullptr)
                        DetourAttach(&(PVOID&) o_CopyResource, hkCopyResource);
                    if (o_CopyTextureRegion != nullptr)
                        DetourAttach(&(PVOID&) o_CopyTextureRegion, hkCopyTextureRegion);
                    // The upscale writer probe's extra routes (see DuniaTaa_Dx12::ProbeOn): one atomic load each
                    // when it is off. Slots confirmed against d3d12.h 10.0.26100: 19, 48, 49, 50.
                    o_ResolveSubresource = (PFN_ResolveSubresource) pVTable[19];
                    o_ClearRenderTargetView = (PFN_ClearRenderTargetView) pVTable[48];
                    o_ClearUnorderedAccessViewUint = (PFN_ClearUnorderedAccessViewUint) pVTable[49];
                    o_ClearUnorderedAccessViewFloat = (PFN_ClearUnorderedAccessViewFloat) pVTable[50];
                    if (o_ResolveSubresource != nullptr)
                        DetourAttach(&(PVOID&) o_ResolveSubresource, hkResolveSubresource);
                    if (o_ClearRenderTargetView != nullptr)
                        DetourAttach(&(PVOID&) o_ClearRenderTargetView, hkClearRenderTargetView);
                    if (o_ClearUnorderedAccessViewUint != nullptr)
                        DetourAttach(&(PVOID&) o_ClearUnorderedAccessViewUint, hkClearUnorderedAccessViewUint);
                    if (o_ClearUnorderedAccessViewFloat != nullptr)
                        DetourAttach(&(PVOID&) o_ClearUnorderedAccessViewFloat, hkClearUnorderedAccessViewFloat);
                    // The TAA resolve skip needs the draw hooks as well, not just the frame trace.
                    if (Config::Instance()->DuniaTaaDiagnostics.value_or_default() ||
                        DuniaTaa_Dx12::WantsTaaSearch())
                    {
                        if (o_DrawInstanced != nullptr)
                            DetourAttach(&(PVOID&) o_DrawInstanced, hkDuniaDrawInstanced);
                        if (o_DrawIndexedInstanced != nullptr)
                            DetourAttach(&(PVOID&) o_DrawIndexedInstanced, hkDuniaDrawIndexedInstanced);
                    }
                }

                if (o_Close != nullptr)
                    DetourAttach(&(PVOID&) o_Close, hkClose);

                if (o_ExecuteBundle != nullptr)
                    DetourAttach(&(PVOID&) o_ExecuteBundle, hkExecuteBundle);

                auto detourResult = DetourTransactionCommit();
                if (detourResult != NO_ERROR)
                {
                    LOG_ERROR("Failed to hook CommandList methods: {:X}", detourResult);
                    o_OMSetRenderTargets = nullptr;
                    o_SetGraphicsRootDescriptorTable = nullptr;
                    o_DrawInstanced = nullptr;
                    o_DrawIndexedInstanced = nullptr;
                    o_Dispatch = nullptr;
                    o_Close = nullptr;
                    o_SetComputeRootDescriptorTable = nullptr;
                    o_ExecuteBundle = nullptr;
                }
            }

            commandList->Close();
            commandList->Release();
        }

        commandAllocator->Reset();
        commandAllocator->Release();
    }
}

void ResTrack_Dx12::HookLateQueue(ID3D12Device* device)
{
    static std::mutex hookMutex;
    std::lock_guard<std::mutex> lock(hookMutex);
    HookToQueue(device);
    LOG_INFO("ResTrack: ExecuteCommandLists hook {}", o_ExecuteCommandLists != nullptr ? "attached" : "NOT attached");
}

void ResTrack_Dx12::HookLateNrQueue(ID3D12Device* device)
{
    static std::mutex hookMutex;
    std::lock_guard<std::mutex> lock(hookMutex);
    HookToQueue(device);
    if (o_LateReset) return;
    ID3D12CommandAllocator* allocator = nullptr;
    ID3D12GraphicsCommandList* cmd = nullptr;
    if (SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator))))
    {
        if (SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator, nullptr, IID_PPV_ARGS(&cmd))))
        {
            ID3D12GraphicsCommandList* real = nullptr;
            if (!CheckForRealObject(__FUNCTION__, cmd, (IUnknown**)&real)) real = cmd;
            o_LateReset = (PFN_LateReset)(*(void***)real)[10];
            DetourTransactionBegin();
            DetourUpdateThread(GetCurrentThread());
            DetourAttach(&(PVOID&)o_LateReset, hkLateReset);
            if (DetourTransactionCommit() != NO_ERROR) o_LateReset = nullptr;
            cmd->Close();
            cmd->Release();
        }
        allocator->Release();
    }
}

void ResTrack_Dx12::HookToQueue(ID3D12Device* InDevice)
{
    if (o_ExecuteCommandLists != nullptr)
        return;

    ID3D12CommandQueue* queue = nullptr;
    D3D12_COMMAND_QUEUE_DESC queueDesc = {};
    queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    queueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
    queueDesc.NodeMask = 0;
    queueDesc.Priority = D3D12_COMMAND_QUEUE_PRIORITY_NORMAL;

    auto hr = InDevice->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&queue));

    if (hr == S_OK)
    {
        ID3D12CommandQueue* realQueue = nullptr;
        if (!CheckForRealObject(__FUNCTION__, queue, (IUnknown**) &realQueue))
            realQueue = queue;

        // Get the vtable pointer
        PVOID* pVTable = *(PVOID**) realQueue;

        o_ExecuteCommandLists = (PFN_ExecuteCommandLists) pVTable[10];

        DetourTransactionBegin();
        DetourUpdateThread(GetCurrentThread());

        if (o_ExecuteCommandLists != nullptr)
            DetourAttach(&(PVOID&) o_ExecuteCommandLists, hkExecuteCommandLists);

        auto detourResult = DetourTransactionCommit();
        if (detourResult != NO_ERROR)
        {
            LOG_ERROR("Failed to hook CommandList methods: {:X}", detourResult);
            o_ExecuteCommandLists = nullptr;
        }

        queue->Release();
    }
}

void ResTrack_Dx12::HookDevice(ID3D12Device* device)
{
    if (o_CreateDescriptorHeap != nullptr || State::Instance().activeFgInput == FGInput::NvngxFG)
        return;

    if (device == nullptr)
        return;

    bool initializeTracking = false;
    {
        std::unique_lock lock(_heapRegistryMutex);
        if (fgHeaps.capacity() < 65536)
        {
            fgHeaps.reserve(65536);
            initializeTracking = true;
        }
    }

    if (initializeTracking)
    {
        _useShards = Config::Instance()->FGUseShards.value_or_default();
        std::scoped_lock lock(_trackedResourcesMutex);
        _trackedResources.reserve(1024);
    }

    LOG_FUNC();
    if (DuniaTaa_Dx12::Active())
        LOG_INFO("DuniaTaa: installing D3D12 RTV/DSV/OMSet tracking");

    ID3D12Device* realDevice = nullptr;
    if (!CheckForRealObject(__FUNCTION__, device, (IUnknown**) &realDevice))
        realDevice = device;

    // Get the vtable pointer
    PVOID* pVTable = *(PVOID**) realDevice;

    // Hudfix
    o_CreateDescriptorHeap = (PFN_CreateDescriptorHeap) pVTable[14];
    o_CreateShaderResourceView = (PFN_CreateShaderResourceView) pVTable[18];
    o_CreateUnorderedAccessView = (PFN_CreateUnorderedAccessView) pVTable[19];
    o_CreateRenderTargetView = (PFN_CreateRenderTargetView) pVTable[20];
    if (DuniaTaa_Dx12::Active())
        o_CreateDepthStencilView = (PFN_CreateDepthStencilView) pVTable[21];
    o_CreateSampler = (PFN_CreateSampler) pVTable[22];
    o_CopyDescriptors = (PFN_CopyDescriptors) pVTable[23];
    o_CopyDescriptorsSimple = (PFN_CopyDescriptorsSimple) pVTable[24];

    // o_CreateDepthStencilView = (PFN_CreateDepthStencilView) pVTable[21];

    // Only for the FC6 diagnostic that needs it: this is one of the hottest calls in a Dunia frame, and every
    // other game here has no use for the record.
    if (DuniaTaa_Dx12::WantsCbvTracking())
        o_CreateConstantBufferView = (PFN_CreateConstantBufferView) pVTable[17];

    // Apply the detour

    if (o_CreateDescriptorHeap != nullptr)
    {
        DetourTransactionBegin();
        DetourUpdateThread(GetCurrentThread());

        if (o_CreateDescriptorHeap != nullptr)
            DetourAttach(&(PVOID&) o_CreateDescriptorHeap, hkCreateDescriptorHeap);

        if (o_CreateRenderTargetView != nullptr)
            DetourAttach(&(PVOID&) o_CreateRenderTargetView, hkCreateRenderTargetView);

        if (o_CreateDepthStencilView != nullptr)
            DetourAttach(&(PVOID&) o_CreateDepthStencilView, hkCreateDepthStencilView);

        if (o_CreateShaderResourceView != nullptr)
            DetourAttach(&(PVOID&) o_CreateShaderResourceView, hkCreateShaderResourceView);

        if (o_CreateUnorderedAccessView != nullptr)
            DetourAttach(&(PVOID&) o_CreateUnorderedAccessView, hkCreateUnorderedAccessView);

        if (o_CreateConstantBufferView != nullptr)
            DetourAttach(&(PVOID&) o_CreateConstantBufferView, hkCreateConstantBufferView);

        if (o_CopyDescriptors != nullptr)
            DetourAttach(&(PVOID&) o_CopyDescriptors, hkCopyDescriptors);

        if (o_CopyDescriptorsSimple != nullptr)
            DetourAttach(&(PVOID&) o_CopyDescriptorsSimple, hkCopyDescriptorsSimple);

        auto detourResult = DetourTransactionCommit();
        LOG_INFO("ResTrack: descriptor hooks -- CreateConstantBufferView {}, buffer SRVs {}",
                 o_CreateConstantBufferView != nullptr ? "attached" : "NOT installed (WantsCbvTracking was false)",
                 DuniaTaa_Dx12::WantsCbvTracking() ? "recorded on demand (diagnostics or a probe)" : "skipped");
        if (detourResult != NO_ERROR)
        {
            LOG_ERROR("Failed to hook Descriptor methods: {:X}", detourResult);
            o_CreateDescriptorHeap = nullptr;
            o_CreateRenderTargetView = nullptr;
            o_CreateDepthStencilView = nullptr;
            o_CreateShaderResourceView = nullptr;
            o_CreateUnorderedAccessView = nullptr;
            o_CopyDescriptors = nullptr;
            o_CopyDescriptorsSimple = nullptr;
        }
    }

    HookToQueue(device);
    HookCommandList(device);
    HookResource(device);
}

void ResTrack_Dx12::ReleaseDeviceHooks()
{
    LOG_DEBUG("");

    DetourTransactionBegin();
    DetourUpdateThread(GetCurrentThread());

    if (o_CreateDescriptorHeap != nullptr)
        DetourDetach(&(PVOID&) o_CreateDescriptorHeap, hkCreateDescriptorHeap);

    if (o_CreateRenderTargetView != nullptr)
        DetourDetach(&(PVOID&) o_CreateRenderTargetView, hkCreateRenderTargetView);

    if (o_CreateDepthStencilView != nullptr)
        DetourDetach(&(PVOID&) o_CreateDepthStencilView, hkCreateDepthStencilView);

    if (o_CreateShaderResourceView != nullptr)
        DetourDetach(&(PVOID&) o_CreateShaderResourceView, hkCreateShaderResourceView);

    if (o_CreateUnorderedAccessView != nullptr)
        DetourDetach(&(PVOID&) o_CreateUnorderedAccessView, hkCreateUnorderedAccessView);

    if (o_CopyDescriptors != nullptr)
        DetourDetach(&(PVOID&) o_CopyDescriptors, hkCopyDescriptors);

    if (o_CopyDescriptorsSimple != nullptr)
        DetourDetach(&(PVOID&) o_CopyDescriptorsSimple, hkCopyDescriptorsSimple);

    // Queue
    if (o_ExecuteCommandLists != nullptr)
        DetourDetach(&(PVOID&) o_ExecuteCommandLists, hkExecuteCommandLists);

    // CommandList
    if (o_OMSetRenderTargets != nullptr)
        DetourDetach(&(PVOID&) o_OMSetRenderTargets, hkOMSetRenderTargets);

    if (o_SetGraphicsRootDescriptorTable != nullptr)
        DetourDetach(&(PVOID&) o_SetGraphicsRootDescriptorTable, hkSetGraphicsRootDescriptorTable);

    if (o_SetComputeRootDescriptorTable != nullptr)
        DetourDetach(&(PVOID&) o_SetComputeRootDescriptorTable, hkSetComputeRootDescriptorTable);

    if (o_DrawIndexedInstanced != nullptr)
        DetourDetach(&(PVOID&) o_DrawIndexedInstanced, hkDrawIndexedInstanced);

    if (o_DrawInstanced != nullptr)
        DetourDetach(&(PVOID&) o_DrawInstanced, hkDrawInstanced);

    if (o_Dispatch != nullptr)
        DetourDetach(&(PVOID&) o_Dispatch, hkDispatch);

    if (o_LateReset != nullptr)
        DetourDetach(&(PVOID&) o_LateReset, hkLateReset);

    if (o_Close != nullptr)
        DetourDetach(&(PVOID&) o_Close, hkClose);

    if (o_CopyResource != nullptr)
        DetourDetach(&(PVOID&) o_CopyResource, hkCopyResource);

    if (o_CopyTextureRegion != nullptr)
        DetourDetach(&(PVOID&) o_CopyTextureRegion, hkCopyTextureRegion);

    if (o_ResolveSubresource != nullptr)
        DetourDetach(&(PVOID&) o_ResolveSubresource, hkResolveSubresource);

    if (o_ClearRenderTargetView != nullptr)
        DetourDetach(&(PVOID&) o_ClearRenderTargetView, hkClearRenderTargetView);

    if (o_ClearUnorderedAccessViewUint != nullptr)
        DetourDetach(&(PVOID&) o_ClearUnorderedAccessViewUint, hkClearUnorderedAccessViewUint);

    if (o_ClearUnorderedAccessViewFloat != nullptr)
        DetourDetach(&(PVOID&) o_ClearUnorderedAccessViewFloat, hkClearUnorderedAccessViewFloat);

    if (o_ExecuteBundle != nullptr)
        DetourDetach(&(PVOID&) o_ExecuteBundle, hkExecuteBundle);

    // Resource
    if (o_Release != nullptr)
        DetourDetach(&(PVOID&) o_Release, hkRelease);

    auto detourResult = DetourTransactionCommit();
    if (detourResult != NO_ERROR)
    {
        LOG_ERROR("Failed to unhook Resource methods: {:X}", detourResult);
    }
    else
    {
        // Device
        o_CreateDescriptorHeap = nullptr;
        o_CreateRenderTargetView = nullptr;
        o_CreateShaderResourceView = nullptr;
        o_CreateUnorderedAccessView = nullptr;
        o_CopyDescriptors = nullptr;
        o_CopyDescriptorsSimple = nullptr;

        // Queue
        o_ExecuteCommandLists = nullptr;

        // CommandList
        o_OMSetRenderTargets = nullptr;
        o_SetGraphicsRootDescriptorTable = nullptr;
        o_SetComputeRootDescriptorTable = nullptr;
        o_DrawIndexedInstanced = nullptr;
        o_DrawInstanced = nullptr;
        o_Dispatch = nullptr;
        o_Close = nullptr;
        o_LateReset = nullptr;
        o_ExecuteBundle = nullptr;

        // Resource
        o_Release = nullptr;
    }
}

void ResTrack_Dx12::ReleaseHooks()
{
    LOG_DEBUG("");

    DetourTransactionBegin();
    DetourUpdateThread(GetCurrentThread());

    // if (o_CreateDescriptorHeap != nullptr)
    //     DetourDetach(&(PVOID&) o_CreateDescriptorHeap, hkCreateDescriptorHeap);

    // if (o_CreateRenderTargetView != nullptr)
    //     DetourDetach(&(PVOID&) o_CreateRenderTargetView, hkCreateRenderTargetView);

    // if (o_CreateShaderResourceView != nullptr)
    //     DetourDetach(&(PVOID&) o_CreateShaderResourceView, hkCreateShaderResourceView);

    // if (o_CreateUnorderedAccessView != nullptr)
    //     DetourDetach(&(PVOID&) o_CreateUnorderedAccessView, hkCreateUnorderedAccessView);

    // if (o_CopyDescriptors != nullptr)
    //     DetourDetach(&(PVOID&) o_CopyDescriptors, hkCopyDescriptors);

    // if (o_CopyDescriptorsSimple != nullptr)
    //     DetourDetach(&(PVOID&) o_CopyDescriptorsSimple, hkCopyDescriptorsSimple);

    // o_CreateDescriptorHeap = nullptr;
    // o_CreateRenderTargetView = nullptr;
    // o_CreateShaderResourceView = nullptr;
    // o_CreateUnorderedAccessView = nullptr;
    // o_CopyDescriptors = nullptr;
    // o_CopyDescriptorsSimple = nullptr;

    // if (o_ExecuteCommandLists != nullptr)
    //     DetourAttach(&(PVOID&) o_ExecuteCommandLists, hkExecuteCommandLists);

    // o_ExecuteCommandLists = nullptr;

    // if (o_Release != nullptr)
    //     DetourAttach(&(PVOID&) o_Release, hkRelease);

    // o_Release = nullptr;

    if (o_OMSetRenderTargets != nullptr)
        DetourDetach(&(PVOID&) o_OMSetRenderTargets, hkOMSetRenderTargets);

    if (o_SetGraphicsRootDescriptorTable != nullptr)
        DetourDetach(&(PVOID&) o_SetGraphicsRootDescriptorTable, hkSetGraphicsRootDescriptorTable);

    if (o_SetComputeRootDescriptorTable != nullptr)
        DetourDetach(&(PVOID&) o_SetComputeRootDescriptorTable, hkSetComputeRootDescriptorTable);

    if (o_DrawIndexedInstanced != nullptr)
        DetourDetach(&(PVOID&) o_DrawIndexedInstanced, hkDrawIndexedInstanced);

    if (o_DrawInstanced != nullptr)
        DetourDetach(&(PVOID&) o_DrawInstanced, hkDrawInstanced);

    if (o_Dispatch != nullptr)
        DetourDetach(&(PVOID&) o_Dispatch, hkDispatch);

    if (o_LateReset != nullptr)
        DetourDetach(&(PVOID&) o_LateReset, hkLateReset);

    if (o_Close != nullptr)
        DetourDetach(&(PVOID&) o_Close, hkClose);

    if (o_CopyResource != nullptr)
        DetourDetach(&(PVOID&) o_CopyResource, hkCopyResource);

    if (o_CopyTextureRegion != nullptr)
        DetourDetach(&(PVOID&) o_CopyTextureRegion, hkCopyTextureRegion);

    if (o_ResolveSubresource != nullptr)
        DetourDetach(&(PVOID&) o_ResolveSubresource, hkResolveSubresource);

    if (o_ClearRenderTargetView != nullptr)
        DetourDetach(&(PVOID&) o_ClearRenderTargetView, hkClearRenderTargetView);

    if (o_ClearUnorderedAccessViewUint != nullptr)
        DetourDetach(&(PVOID&) o_ClearUnorderedAccessViewUint, hkClearUnorderedAccessViewUint);

    if (o_ClearUnorderedAccessViewFloat != nullptr)
        DetourDetach(&(PVOID&) o_ClearUnorderedAccessViewFloat, hkClearUnorderedAccessViewFloat);

    if (o_ExecuteBundle != nullptr)
        DetourDetach(&(PVOID&) o_ExecuteBundle, hkExecuteBundle);

    auto detourResult = DetourTransactionCommit();
    if (detourResult != NO_ERROR)
    {
        LOG_ERROR("Failed to unhook CommandList methods: {:X}", detourResult);
    }
    else
    {
        o_OMSetRenderTargets = nullptr;
        o_SetGraphicsRootDescriptorTable = nullptr;
        o_SetComputeRootDescriptorTable = nullptr;
        o_DrawIndexedInstanced = nullptr;
        o_DrawInstanced = nullptr;
        o_Dispatch = nullptr;
        o_Close = nullptr;
        o_LateReset = nullptr;
        o_ExecuteBundle = nullptr;
    }
}

void ResTrack_Dx12::ClearPossibleHudless()
{
    LOG_DEBUG("");

    auto hfIndex = Hudfix_Dx12::ActivePresentFrame() % BUFFER_COUNT;

    if (!_useShards)
    {
        std::lock_guard<std::mutex> lock(_hudlessTrackMutex);
        fgPossibleHudless[hfIndex].clear();
    }
    else
    {
        for (size_t i = 0; i < SHARD_COUNT; i++)
        {
            auto& shard = _hudlessShards[hfIndex][i];

            LOCK_GUARD(shard.mutex);

            shard.map.clear();
        }
    }

    std::lock_guard<std::mutex> lock2(_resourceCommandListMutex);

    auto fg = State::Instance().currentFG;
    if (fg != nullptr)
    {
        auto fIndex = fg->GetIndex();

        if (_notFoundCmdLists.size() > 10)
            _notFoundCmdLists.clear();

        for (const auto& pair : _resourceCommandList[fIndex])
        {
            LOG_WARN("{} cmdList: {:X}, not closed!", (UINT) pair.first, (size_t) pair.second);
            _notFoundCmdLists.insert(pair.second);
        }

        _resourceCommandList[fIndex].clear();

        for (const auto& pair : _resCmdList[fIndex])
        {
            LOG_WARN("{} cmdList: {:X}, not executed!", (UINT) pair.first, (size_t) pair.second);
            _notFoundCmdLists.insert(pair.second);
        }

        _resCmdList[fIndex].clear();
    }
}

void ResTrack_Dx12::SetResourceCmdList(FG_ResourceType type, ID3D12GraphicsCommandList* cmdList)
{
    auto fg = State::Instance().currentFG;
    if (fg != nullptr && fg->IsActive())
    {
        auto index = fg->GetIndex();

        ID3D12GraphicsCommandList* realCmdList = nullptr;
        if (!CheckForRealObject(__FUNCTION__, cmdList, (IUnknown**) &realCmdList))
            realCmdList = cmdList;

        _resourceCommandList[index][type] = realCmdList;
        LOG_DEBUG("_resourceCommandList[{}][{}]: {:X}", index, magic_enum::enum_name(type), (size_t) realCmdList);
    }
}
