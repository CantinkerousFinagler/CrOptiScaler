#include <pch.h>
#include "IFeature_Dx11wDx12.h"
#include "NgxOptionalDx12Inputs.h"

#include <dlssnr/DlssNr.h>

#include <Config.h>

#include <proxies/DXGI_Proxy.h>
#include <proxies/D3D12_Proxy.h>
#include <misc/IdentifyGpu.h>

#include <with_dx12/with_dx12.h>
#include <gpu_time/StageTimes.h>

#include <array>

namespace
{
// Diagnostics ([Inputs] DuniaTaaDiagnostics): where the render thread's time goes inside Evaluate, averaged over
// 900 evaluations. Stages: state backup, input copies to D3D12 (of which the D3D11 Flush), NR before the
// upscaler, the upscaler, NR after it, D3D12 submit, output copy back, state restore.
enum BridgeStage
{
    StageBackup,
    StagePrepare,
    StageNrBefore,
    StageUpscaler,
    StageNrAfter,
    StageSubmit,
    StageCopyBack,
    StageRestore,
    StageCount
};

std::array<UINT64, StageCount> g_stageMicros {};
UINT64 g_stageFlushMicros = 0;
// Inside prepare: the CPU wait for the GPU to finish the D3D12 work that last used this frame's command allocator.
UINT64 g_stageAllocatorWaitMicros = 0;
UINT64 g_stageEvaluations = 0;

LONGLONG NowMicros()
{
    static LARGE_INTEGER frequency = []
    {
        LARGE_INTEGER value {};
        QueryPerformanceFrequency(&value);
        return value;
    }();
    LARGE_INTEGER now {};
    QueryPerformanceCounter(&now);
    return now.QuadPart * 1000000 / frequency.QuadPart;
}

void ReportStages(const std::array<LONGLONG, StageCount + 1>& marks)
{
    if (!Config::Instance()->DuniaTaaDiagnostics.value_or_default() &&
        !Config::Instance()->UpscalerGpuTiming.value_or_default())
        return;
    for (int i = 0; i < StageCount; ++i)
        g_stageMicros[i] += (UINT64) std::max<LONGLONG>(0, marks[i + 1] - marks[i]);
    g_stageFlushMicros += Dx11WithDx12::TakeFlushMicros();
    if (++g_stageEvaluations < 900)
        return;
    const double n = (double) g_stageEvaluations;
    auto ms = [n](UINT64 micros) { return (double) micros / n / 1000.0; };
    LOG_INFO("Dx11wDx12 bridge, render thread per evaluation (ms): backup {:.2f}, prepare {:.2f} (flush {:.2f}, "
             "GPU wait {:.2f}), NR before {:.2f}, upscaler {:.2f}, NR after {:.2f}, submit {:.2f}, copy back {:.2f}, "
             "restore {:.2f}",
             ms(g_stageMicros[StageBackup]), ms(g_stageMicros[StagePrepare]), ms(g_stageFlushMicros),
             ms(g_stageAllocatorWaitMicros),
             ms(g_stageMicros[StageNrBefore]), ms(g_stageMicros[StageUpscaler]), ms(g_stageMicros[StageNrAfter]),
             ms(g_stageMicros[StageSubmit]), ms(g_stageMicros[StageCopyBack]), ms(g_stageMicros[StageRestore]));
    g_stageMicros = {};
    g_stageFlushMicros = 0;
    g_stageAllocatorWaitMicros = 0;
    g_stageEvaluations = 0;
}
} // namespace

void IFeature_Dx11wDx12::ResourceBarrier(ID3D12GraphicsCommandList* commandList, ID3D12Resource* resource,
                                         D3D12_RESOURCE_STATES beforeState, D3D12_RESOURCE_STATES afterState)
{
    if (beforeState == afterState)
        return;

    D3D12_RESOURCE_BARRIER barrier = {};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = resource;
    barrier.Transition.StateBefore = beforeState;
    barrier.Transition.StateAfter = afterState;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    commandList->ResourceBarrier(1, &barrier);
}

bool IFeature_Dx11wDx12::CreateD3D12Objects()
{
    HRESULT result;

    for (size_t i = 0; i < DX11WDX12_NUM_OF_BUFFERS; i++)
    {
        if (Dx12CommandAllocator[i] == nullptr)
        {
            result =
                _dx11on12Device->CreateCommandAllocator(Dx12CommandListType, IID_PPV_ARGS(&Dx12CommandAllocator[i]));

            if (result != S_OK)
            {
                LOG_ERROR("CreateCommandAllocator error: {:X}", result);
                return false;
            }
        }

        if (Dx12CommandList[i] == nullptr && Dx12CommandAllocator[i] != nullptr)
        {
            // CreateCommandList
            result = _dx11on12Device->CreateCommandList(0, Dx12CommandListType, Dx12CommandAllocator[i], nullptr,
                                                        IID_PPV_ARGS(&Dx12CommandList[i]));

            if (result != S_OK)
            {
                LOG_ERROR("CreateCommandList error: {:X}", result);
                return false;
            }

            Dx12CommandList[i]->Close();
        }
    }

    if (Dx12Fence == nullptr)
    {
        result = _dx11on12Device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&Dx12Fence));

        if (result != S_OK)
        {
            LOG_ERROR("CreateFence error: {0:X}", result);
            return false;
        }

        Dx12FenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);

        if (Dx12FenceEvent == nullptr)
        {
            LOG_ERROR("CreateEvent error!");
            return false;
        }
    }

    return true;
}

void IFeature_Dx11wDx12::ReleaseSharedResources()
{
    for (size_t i = 0; i < DX11WDX12_NUM_OF_BUFFERS; i++)
    {
        SAFE_RELEASE(Dx12CommandList[i]);
        SAFE_RELEASE(Dx12CommandAllocator[i]);
        Dx12CommandAllocatorFenceValue[i] = 0;
    }

    SAFE_RELEASE(Dx12Fence);
    Dx12FenceValue = 0;

    if (Dx12FenceEvent)
    {
        CloseHandle(Dx12FenceEvent);
        Dx12FenceEvent = nullptr;
    }

    // SAFE_RELEASE(Dx12Device);
}

bool IFeature_Dx11wDx12::ProcessDx11Textures(const NVSDK_NGX_Parameter* InParameters)
{
    HRESULT result;

    auto frame = _frameCount % DX11WDX12_NUM_OF_BUFFERS;
    const auto cacheFrameKey = Dx11WithDx12::NextUpscalerFrameId();
    Dx11WithDx12::SetUpscalerFrameIndex((UINT) frame);

    auto mask = Dx11WithDx12::ResourceMask::Color | Dx11WithDx12::ResourceMask::Mv | Dx11WithDx12::ResourceMask::Depth |
                Dx11WithDx12::ResourceMask::Output;

    if (!AutoExposure())
        mask |= Dx11WithDx12::ResourceMask::Exposure;
    else
        LOG_DEBUG("AutoExposure enabled!");

    const bool reactiveDisabled = Config::Instance()->DisableReactiveMask.value_or(false);
    const bool reactiveRequired = Config::Instance()->Dx11Upscaler.value_or_default() == Upscaler::XeSS ||
                                  Config::Instance()->Dx11Upscaler.value_or_default() == Upscaler::XeSS_on12;

    if (!reactiveDisabled)
        mask |= Dx11WithDx12::ResourceMask::Reactive;
    else
        LOG_DEBUG("ReactiveMask disabled!");

    const auto prepareResult = Dx11WithDx12::PrepareUpscalerResources(
        InParameters, mask, (UINT) frame, cacheFrameKey, Config::Instance()->DontUseNTShared.value_or_default(),
        reactiveRequired, true);

    if (!prepareResult.Success)
    {
        if (prepareResult.MissingExposure)
        {
            LOG_WARN("AutoExposure disabled but ExposureTexture does not exist, enabling auto exposure and changing "
                     "backend");
            State::Instance().autoExposure = true;
            State::Instance().changeBackend[Handle()->Id] = true;
            return true;
        }

        if (prepareResult.MissingReactive && reactiveRequired)
        {
            LOG_WARN("Bias mask does not exist and is required by the current DX11 upscaler, disabling reactive mask");
            Config::Instance()->DisableReactiveMask.set_volatile_value(true);
            State::Instance().changeBackend[Handle()->Id] = true;
            return true;
        }

        LOG_ERROR("Dx11wDx12 resource cache preparation failed");
        return false;
    }

    const auto allocatorFenceValue = Dx12CommandAllocatorFenceValue[frame];
    if (allocatorFenceValue != 0 && Dx12Fence->GetCompletedValue() < allocatorFenceValue)
    {
        result = Dx12Fence->SetEventOnCompletion(allocatorFenceValue, Dx12FenceEvent);
        if (result != S_OK)
        {
            LOG_ERROR("SetEventOnCompletion error for allocator {} fence {}: {:X}", frame, allocatorFenceValue,
                      (UINT) result);
            return false;
        }

        const auto waitStart = NowMicros();
        const auto waitResult = WaitForSingleObject(Dx12FenceEvent, INFINITE);
        g_stageAllocatorWaitMicros += (UINT64) std::max<LONGLONG>(0, NowMicros() - waitStart);
        if (waitResult != WAIT_OBJECT_0)
        {
            LOG_ERROR("WaitForSingleObject failed for allocator {} fence {}: {:X}", frame, allocatorFenceValue,
                      (UINT) waitResult);
            return false;
        }
    }

    result = Dx12CommandAllocator[frame]->Reset();
    if (result != S_OK)
    {
        LOG_ERROR("CommandAllocator Reset error for frame {}, allocator fence {}, completed {}: {:X}", frame,
                  allocatorFenceValue, Dx12Fence->GetCompletedValue(), (UINT) result);
        return false;
    }

    result = Dx12CommandList[frame]->Reset(Dx12CommandAllocator[frame], nullptr);
    if (result != S_OK)
    {
        LOG_ERROR("CommandList Reset error: {:X}", (UINT) result);
        return false;
    }

    LOG_DEBUG("Shared handles prepared and synchronized by Dx11WithDx12 cache, frameKey: {}", cacheFrameKey);
    return true;
}

bool IFeature_Dx11wDx12::CopyBackOutput()
{
    const auto frame = (UINT) (_frameCount % DX11WDX12_NUM_OF_BUFFERS);
    return Dx11WithDx12::CopyUpscalerOutputToDx11(frame);
}

bool IFeature_Dx11wDx12::Init(ID3D11Device* InDevice, ID3D11DeviceContext* InContext, NVSDK_NGX_Parameter* InParameters)
{
    LOG_FUNC();

    if (IsInited())
        return true;

    if (State::Instance().NVNGX_Engine == NVSDK_NGX_ENGINE_TYPE_UNREAL ||
        State::Instance().gameEngine == GameEngineType::Unreal ||
        State::Instance().gameQuirks & GameQuirk::ForceUnrealEngine)
    {
        LOG_INFO("Dx11 detected, disabling UE resource barrier overrides");
        Config::Instance()->ColorResourceBarrier.set_volatile_value(D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        Config::Instance()->MVResourceBarrier.set_volatile_value(D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    }

    Device = InDevice;
    DeviceContext = InContext;

    if (!BaseInit(Device, InContext, InParameters))
    {
        LOG_DEBUG("BaseInit failed!");
        return false;
    }

    SetInitParameters(InParameters);

    // The list has to be recording, and what is recorded on it has to run.
    //
    // Every command list here is closed the moment it is created, and the note this replaces said
    // non-DLSS upscalers do not use the list during Init -- which was true, and stopped being true the
    // day DLSS got a bridge variant. DLSS and Ray Reconstruction are the only features whose
    // InitInternal touches the argument at all: NVSDK_NGX_D3D12_CreateFeature records the model's
    // weight upload and history initialisation onto it, and documents that the caller must submit it
    // afterwards.
    //
    // Recorded onto a closed list, all of that is discarded. ID3D12GraphicsCommandList methods return
    // void, so nothing fails, nothing logs, and CreateFeature still answers Success -- the model then
    // runs against state that was never uploaded. That is the posterised, flat-blocked picture, and it
    // gets worse with model size: the old CNN degraded to soft, the transformer collapses to blocks.
    //
    // So: open the list, let Init record into it, submit it, and wait. The wait is not optional --
    // the first Evaluate resets this same allocator, and doing that under work still in flight is a
    // device removal rather than a bad picture.
    HRESULT prep = Dx12CommandAllocator[0]->Reset();

    if (prep != S_OK)
        LOG_WARN("Init: allocator reset before feature creation failed: {:X}", (UINT) prep);

    prep = Dx12CommandList[0]->Reset(Dx12CommandAllocator[0], nullptr);

    if (prep != S_OK)
        LOG_WARN("Init: command list reset before feature creation failed: {:X}", (UINT) prep);

    const bool initialised = dx12Feature->Init(_dx11on12Device, Dx12CommandList[0], InParameters);

    SetInit(initialised);

    if (Dx12CommandList[0]->Close() == S_OK && Dx12CommandQueue != nullptr)
    {
        ID3D12CommandList* lists[] = { Dx12CommandList[0] };
        Dx12CommandQueue->ExecuteCommandLists(1, lists);

        // Recorded against allocator 0, so allocator 0 must not be reset until this has retired. That
        // is what Dx12CommandAllocatorFenceValue is for, and ProcessDx11Textures already honours it.
        const UINT64 signalled = ++Dx12FenceValue;

        if (Dx12CommandQueue->Signal(Dx12Fence, signalled) == S_OK)
        {
            Dx12CommandAllocatorFenceValue[0] = signalled;

            if (Dx12Fence->GetCompletedValue() < signalled &&
                Dx12Fence->SetEventOnCompletion(signalled, Dx12FenceEvent) == S_OK)
            {
                WaitForSingleObject(Dx12FenceEvent, INFINITE);
            }
        }

        LOG_INFO("Init: feature creation work submitted and waited on (fence {})", signalled);
    }
    else
    {
        LOG_WARN("Init: could not submit the feature creation work; a feature that records during "
                 "creation will be missing it");
    }

    return IsInited();
}

bool IFeature_Dx11wDx12::Evaluate(ID3D11DeviceContext* InDeviceContext, NVSDK_NGX_Parameter* InParameters)
{
    LOG_FUNC();

    if (!IsInited())
        return false;

    std::array<LONGLONG, StageCount + 1> marks {};
    marks[StageBackup] = NowMicros();

    ID3D11DeviceContext4* dc;
    auto result = InDeviceContext->QueryInterface(IID_PPV_ARGS(&dc));

    if (result != S_OK)
    {
        LOG_ERROR("QueryInterface error: {0:x}", result);
        return false;
    }

    if (dc != Dx11DeviceContext)
    {
        LOG_WARN("Dx11DeviceContext changed!");
        ReleaseSharedResources();
        Dx11DeviceContext = dc;

        if (!CreateD3D12Objects())
        {
            LOG_ERROR("Failed to recreate Dx11wDx12 D3D12 objects after context change");
            if (dc != nullptr)
                dc->Release();
            return false;
        }
    }

    if (dc != nullptr)
        dc->Release();

    auto frame = _frameCount % DX11WDX12_NUM_OF_BUFFERS;
    auto cmdList = Dx12CommandList[frame];

    // [Inputs] UpscalerGpuTiming (live): what the lists before this one took on the GPU, as far as they are done.
    const bool timing = Config::Instance()->UpscalerGpuTiming.value_or_default();
    if (timing)
    {
        if (!_listTimesTried)
        {
            _listTimesTried = true;
            if (!_listTimes.Init(_dx11on12Device))
                LOG_ERROR("GPU timing: the timestamps for the bridge's list could not be made; its stages are not timed");
            else
                LOG_INFO("GPU timing: timestamps on the bridge's list of feature {}", Handle()->Id);
        }
        const bool nr = Config::Instance()->DlssNrEnabled.value_or_default();
        // DLSS writes its output: 4 bytes a pixel at the least. NR's own size is not known here: no floor.
        const double upscalerFloor = StageTimes::FloorMs((double) DisplayWidth() * (double) DisplayHeight() * 4.0);
        ListTimes_Dx12::Intervals took;
        while (_listTimes.Collect(&took))
        {
            // NR that did not run leaves an interval of a ten-thousandth of a millisecond: not a sample.
            if (took.have[0] && nr && took.ms[0] >= 0.01)
                StageTimes::Global().Add(StageTimes::NrBefore, took.ms[0], 0.0);
            if (took.have[1])
                StageTimes::Global().Add(StageTimes::Upscaler, took.ms[1], upscalerFloor);
            if (took.have[2] && nr && took.ms[2] >= 0.01)
                StageTimes::Global().Add(StageTimes::NrAfter, took.ms[2], 0.0);
        }
    }

    auto& cache = Dx11WithDx12::GetUpscalerResourceCache();
    auto& dx11Color = cache.Color;
    auto& dx11Mv = cache.Mv;
    auto& dx11Depth = cache.Depth;
    auto& dx11Reactive = cache.Reactive;
    auto& dx11Exp = cache.Exposure;
    auto& dx11Out = cache.Output[frame];

    auto getOriginalNgxResource = [](NVSDK_NGX_Parameter* parameters, const char* name, ID3D11Resource** outResource)
    {
        if (parameters == nullptr || name == nullptr || outResource == nullptr)
            return false;

        *outResource = nullptr;

        if (parameters->Get(name, outResource) != NVSDK_NGX_Result_Success)
            parameters->Get(name, (void**) outResource);

        return *outResource != nullptr;
    };

    ID3D11Resource* restoreParamColor = nullptr;
    ID3D11Resource* restoreParamMv = nullptr;
    ID3D11Resource* restoreParamOutput = nullptr;
    ID3D11Resource* restoreParamDepth = nullptr;
    ID3D11Resource* restoreParamExposure = nullptr;
    ID3D11Resource* restoreParamReactive = nullptr;

    const bool hasRestoreParamColor =
        getOriginalNgxResource(InParameters, NVSDK_NGX_Parameter_Color, &restoreParamColor);
    const bool hasRestoreParamMv =
        getOriginalNgxResource(InParameters, NVSDK_NGX_Parameter_MotionVectors, &restoreParamMv);
    const bool hasRestoreParamOutput =
        getOriginalNgxResource(InParameters, NVSDK_NGX_Parameter_Output, &restoreParamOutput);
    const bool hasRestoreParamDepth =
        getOriginalNgxResource(InParameters, NVSDK_NGX_Parameter_Depth, &restoreParamDepth);
    getOriginalNgxResource(InParameters, NVSDK_NGX_Parameter_ExposureTexture, &restoreParamExposure);
    getOriginalNgxResource(InParameters, NVSDK_NGX_Parameter_DLSS_Input_Bias_Current_Color_Mask, &restoreParamReactive);

    ID3D11ShaderResourceView* restoreSRVs[D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT] = {};
    ID3D11SamplerState* restoreSamplerStates[D3D11_COMMONSHADER_SAMPLER_SLOT_COUNT] = {};
    ID3D11Buffer* restoreCBVs[D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT] = {};
    ID3D11UnorderedAccessView* restoreUAVs[D3D11_1_UAV_SLOT_COUNT] = {};
    ID3D11RenderTargetView* restoreRTVs[D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT] = {};
    ID3D11DepthStencilView* restoreDSV = nullptr;

    // backup compute shader resources
    for (UINT i = 0; i < D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT; i++)
    {
        restoreSRVs[i] = nullptr;
        InDeviceContext->CSGetShaderResources(i, 1, &restoreSRVs[i]);

        if (restoreSRVs[i] != nullptr)
            restoreSRVs[i]->Release();
    }

    for (UINT i = 0; i < D3D11_COMMONSHADER_SAMPLER_SLOT_COUNT; i++)
    {
        restoreSamplerStates[i] = nullptr;
        InDeviceContext->CSGetSamplers(i, 1, &restoreSamplerStates[i]);

        if (restoreSamplerStates[i] != nullptr)
            restoreSamplerStates[i]->Release();
    }

    for (UINT i = 0; i < D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT; i++)
    {
        restoreCBVs[i] = nullptr;
        InDeviceContext->CSGetConstantBuffers(i, 1, &restoreCBVs[i]);

        if (restoreCBVs[i] != nullptr)
            restoreCBVs[i]->Release();
    }

    for (UINT i = 0; i < D3D11_1_UAV_SLOT_COUNT; i++)
    {
        restoreUAVs[i] = nullptr;
        InDeviceContext->CSGetUnorderedAccessViews(i, 1, &restoreUAVs[i]);

        if (restoreUAVs[i] != nullptr)
            restoreUAVs[i]->Release();
    }

    InDeviceContext->OMGetRenderTargets(D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT, restoreRTVs, &restoreDSV);

    for (UINT i = 0; i < D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT; i++)
    {
        if (restoreRTVs[i] != nullptr)
            restoreRTVs[i]->Release();
    }

    if (restoreDSV != nullptr)
        restoreDSV->Release();

    // Unbind RenderTargets
    ID3D11RenderTargetView* nullRTVs[D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT] = {};
    InDeviceContext->OMSetRenderTargets(D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT, nullRTVs, nullptr);

    bool dx12EvalResult = false;
    bool commandListRecording = false;
    bool commandListExecuted = false;
    marks[StagePrepare] = NowMicros();
    do
    {
        if (!ProcessDx11Textures(InParameters))
        {
            LOG_ERROR("Can't process Dx11 textures!");
            break;
        }

        if (State::Instance().changeBackend[Handle()->Id])
        {
            break;
        }

        commandListRecording = true;

        InParameters->Set(NVSDK_NGX_Parameter_Color, (void*) dx11Color.Dx12Resource);
        InParameters->Set(NVSDK_NGX_Parameter_MotionVectors, (void*) dx11Mv.Dx12Resource);
        InParameters->Set(NVSDK_NGX_Parameter_Output, (void*) dx11Out.Dx12Resource);
        InParameters->Set(NVSDK_NGX_Parameter_Depth, (void*) dx11Depth.Dx12Resource);

        SetOptionalDx12Inputs(InParameters, dx11Exp.Dx12Resource, dx11Reactive.Dx12Resource, AutoExposure(),
                              Config::Instance()->DisableReactiveMask.value_or(false));

        LOG_DEBUG("Dispatch!!");
        marks[StageNrBefore] = NowMicros();
        if (timing)
            _listTimes.Begin(cmdList);
        DlssNr::EvaluateBeforeUpscale(cmdList, InParameters, Dx12CommandQueue, _frameCount,
                                      dx12Feature->GetUpscalerType() == Upscaler::DLSSD);
        marks[StageUpscaler] = NowMicros();
        _listTimes.Mark(cmdList, 1);
        dx12EvalResult = dx12Feature->Evaluate(cmdList, InParameters);
        marks[StageNrAfter] = NowMicros();
        _listTimes.Mark(cmdList, 2);

        // DLSS 5 Neural Rendering rides the bridge: at this moment the block carries the D3D12 copies
        // of every input, the list is still recording, and the model's edit lands on the D3D12 output
        // before it is copied back to the game's D3D11 texture. This one call is what makes the pass
        // work in DirectX 11 games, whatever upscaler carried it here.
        static bool reportedNrOffer = false;

        if (!reportedNrOffer)
        {
            reportedNrOffer = true;
            LOG_INFO("DLSS-NR: the D3D11 bridge reached the hand-off (upscale ok: {}, enabled: {})",
                     dx12EvalResult, Config::Instance()->DlssNrEnabled.value_or_default());

        }

        if (dx12EvalResult && Config::Instance()->DlssNrEnabled.value_or_default())
        {
            DlssNr::EvaluateAfterUpscale(cmdList, InParameters, Dx12CommandQueue,
                                         dx12Feature->GetUpscalerType() == Upscaler::DLSSD,
                                         _frameCount);

            // Asked only after the D3D12 path has had its turn. Probing first would have made a D3D11
            // init the very first thing to ever touch the snippet, and if that had left its core
            // holding a D3D11 device the D3D12 create would have failed -- killing the feature in
            // exactly the games the probe was written to help. Off by default regardless.
            DlssNr::ProbeD3D11(Dx11Device);
        }

    } while (false);

    if (hasRestoreParamColor)
        InParameters->Set(NVSDK_NGX_Parameter_Color, (void*) restoreParamColor);

    if (hasRestoreParamMv)
        InParameters->Set(NVSDK_NGX_Parameter_MotionVectors, (void*) restoreParamMv);

    if (hasRestoreParamOutput)
        InParameters->Set(NVSDK_NGX_Parameter_Output, (void*) restoreParamOutput);

    if (hasRestoreParamDepth)
        InParameters->Set(NVSDK_NGX_Parameter_Depth, (void*) restoreParamDepth);

    // Restore nulls too: no D3D12 pointer may escape back into the DX11 caller.
    InParameters->Set(NVSDK_NGX_Parameter_ExposureTexture, (void*) restoreParamExposure);
    InParameters->Set(NVSDK_NGX_Parameter_DLSS_Input_Bias_Current_Color_Mask, (void*) restoreParamReactive);

    marks[StageSubmit] = NowMicros();
    if (commandListRecording)
    {
        // The last mark and the resolve; nothing is written when no timing began on this list.
        _listTimes.Mark(cmdList, 3);
        _listTimes.End(cmdList);
        const auto closeResult = cmdList->Close();
        if (closeResult != S_OK)
        {
            LOG_ERROR("CommandList Close error: {:X}", (UINT) closeResult);
            dx12EvalResult = false;
        }
    }

    if (dx12EvalResult)
    {
        ID3D12CommandList* ppCommandLists[] = { cmdList };
        Dx12CommandQueue->ExecuteCommandLists(1, ppCommandLists);
        commandListExecuted = true;
        _listTimes.Submitted(Dx12CommandQueue, cmdList);

        const auto fenceValue = ++Dx12FenceValue;
        result = Dx12CommandQueue->Signal(Dx12Fence, fenceValue);
        if (result != S_OK)
        {
            LOG_ERROR("Dx12CommandQueue Signal failed for feature fence {}: {:X}", fenceValue, (UINT) result);
            dx12EvalResult = false;
        }
        else
        {
            Dx12CommandAllocatorFenceValue[frame] = fenceValue;
        }
    }

    auto evalResult = false;
    marks[StageCopyBack] = NowMicros();

    do
    {
        if (!dx12EvalResult || !commandListExecuted)
            break;

        if (!CopyBackOutput())
        {
            LOG_ERROR("Can't copy output texture back!");
            break;
        }

        evalResult = true;

    } while (false);

    if (evalResult)
        _frameCount++;
    else
        Dx11WithDx12::ClearLastPreparedUpscalerFrameState();

    marks[StageRestore] = NowMicros();

    // restore compute shader resources
    for (UINT i = 0; i < D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT; i++)
    {
        InDeviceContext->CSSetShaderResources(i, 1, &restoreSRVs[i]);
    }

    for (UINT i = 0; i < D3D11_COMMONSHADER_SAMPLER_SLOT_COUNT; i++)
    {
        InDeviceContext->CSSetSamplers(i, 1, &restoreSamplerStates[i]);
    }

    for (UINT i = 0; i < D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT; i++)
    {
        InDeviceContext->CSSetConstantBuffers(i, 1, &restoreCBVs[i]);
    }

    for (UINT i = 0; i < D3D11_1_UAV_SLOT_COUNT; i++)
    {
        InDeviceContext->CSSetUnorderedAccessViews(i, 1, &restoreUAVs[i], 0);
    }

    InDeviceContext->OMSetRenderTargets(D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT, restoreRTVs, restoreDSV);

    marks[StageCount] = NowMicros();
    if (evalResult)
        ReportStages(marks);
    if (evalResult && timing)
        StageTimes::Global().Add(StageTimes::RenderThread, (double) (marks[StageCount] - marks[StageBackup]) / 1000.0,
                                 0.0);

    return evalResult;
}

bool IFeature_Dx11wDx12::BaseInit(ID3D11Device* InDevice, ID3D11DeviceContext* InContext,
                                  NVSDK_NGX_Parameter* InParameters)
{
    LOG_FUNC();

    if (IsInited())
        return true;

    if (!InContext)
    {
        LOG_ERROR("context is null!");
        return false;
    }

    auto contextResult = InContext->QueryInterface(IID_PPV_ARGS(&Dx11DeviceContext));
    if (contextResult != S_OK)
    {
        LOG_ERROR("QueryInterface ID3D11DeviceContext4 result: {0:x}", contextResult);
        return false;
    }
    else
    {
        Dx11DeviceContext->Release();
    }

    if (!InDevice)
        Dx11DeviceContext->GetDevice(&InDevice);

    auto dx11DeviceResult = InDevice->QueryInterface(IID_PPV_ARGS(&Dx11Device));

    if (dx11DeviceResult != S_OK)
    {
        LOG_ERROR("QueryInterface ID3D11Device5 result: {0:x}", dx11DeviceResult);
        return false;
    }
    else
    {
        Dx11Device->Release();
    }

    if (!WithDx12::PrepareD3D12ForD3D11(InDevice, D3D_FEATURE_LEVEL_11_0))
    {
        LOG_ERROR("Cannot resolve D3D12 device/queue from WithDx12!");
        return false;
    }

    _dx11on12Device = WithDx12::GetD3D12Device();
    if (_dx11on12Device == nullptr)
    {
        LOG_ERROR("Cannot get D3D12 device from WithDx12!");
        return false;
    }

    Dx12CommandQueue = WithDx12::GetD3D12CommandQueue();
    if (Dx12CommandQueue == nullptr)
    {
        LOG_ERROR("Cannot get D3D12 command queue from WithDx12!");
        return false;
    }

    Dx12CommandListType = WithDx12::GetD3D12CommandListType();

    if (!CreateD3D12Objects())
    {
        LOG_ERROR("Failed to create D3D12 objects!");
        return false;
    }

    Dx11WithDx12::Init(Dx11Device, Dx11DeviceContext);

    return true;
}

IFeature_Dx11wDx12::IFeature_Dx11wDx12(unsigned int InHandleId, NVSDK_NGX_Parameter* InParameters)
    : IFeature(InHandleId, InParameters), IFeature_Dx11(InHandleId, InParameters)
{
}

IFeature_Dx11wDx12::~IFeature_Dx11wDx12()
{
    if (State::Instance().isShuttingDown)
    {
        _listTimes.Abandon();
        return;
    }

    // The timestamps' heap and buffer first, after the lists that name them are done on the GPU.
    _listTimes.Free_();
    ReleaseSharedResources();
}
