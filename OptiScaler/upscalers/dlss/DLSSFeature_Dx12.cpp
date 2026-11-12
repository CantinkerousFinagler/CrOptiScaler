#include <pch.h>
#include "DLSSFeature_Dx12.h"
#include <inputs/PerEyeDlss_Dx12.h>
#include <dxgi1_4.h>
#include <Config.h>
#include <upscalers/NgxDirect_Dx12.h>

bool DLSSFeatureDx12::InitInternal(ID3D12GraphicsCommandList* InCommandList, NVSDK_NGX_Parameter* InParameters)
{
    if (IsInited())
        return true;

    return InitDLSS(InCommandList, InParameters);
}

bool DLSSFeatureDx12::InitDLSS(ID3D12GraphicsCommandList* InCommandList, NVSDK_NGX_Parameter* InParameters)
{
    if (NVNGXProxy::NVNGXModule() == nullptr)
    {
        LOG_ERROR("nvngx.dll not loaded!");
        return false;
    }

    if (!_dlssInitedDx12)
    {
        _dlssInitedDx12 = NVNGXProxy::InitDx12(Device);

        if (!_dlssInitedDx12)
            return false;

        _moduleLoaded =
            (NVNGXProxy::D3D12_Init_ProjectID() != nullptr || NVNGXProxy::D3D12_Init_Ext() != nullptr) &&
            (NVNGXProxy::D3D12_Shutdown() != nullptr || NVNGXProxy::D3D12_Shutdown1() != nullptr) &&
            (NVNGXProxy::D3D12_GetParameters() != nullptr || NVNGXProxy::D3D12_AllocateParameters() != nullptr) &&
            NVNGXProxy::D3D12_DestroyParameters() != nullptr && NVNGXProxy::D3D12_CreateFeature() != nullptr &&
            NVNGXProxy::D3D12_ReleaseFeature() != nullptr && NVNGXProxy::D3D12_EvaluateFeature() != nullptr;

        // delay between init and create feature
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    LOG_INFO("Creating DLSS feature");

    if (NVNGXProxy::D3D12_CreateFeature() != nullptr)
    {
        ProcessInitParams(InParameters);

        // Per-eye DLSS's foveal crop has its own preset, after OptiScaler's overrides for the game's features.
        if (const int preset = PerEyeDlss_Dx12::PresetForCreate(); preset > 0)
        {
            for (const char* name : { NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_DLAA,
                                      NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_UltraQuality,
                                      NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Quality,
                                      NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Balanced,
                                      NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Performance,
                                      NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_UltraPerformance })
                InParameters->Set(name, (unsigned int) preset);
            LOG_INFO("Per-eye DLSS DX12: this feature is a foveal crop, preset {}", preset);
        }

        _p_dlssHandle = &_dlssHandle;
        const uint64_t vramBefore = PerEyeDlss_Dx12::VramUsedMB(Device);

        NVSDK_NGX_Result nvResult = NVSDK_NGX_Result_Fail;
        _crop = PerEyeDlss_Dx12::KindForCreate() == 1;
        _direct = false;
        // A foveal crop past the driver core (the SR twin of DLSSDFeatureDx12's): made on
        // nvngx_dlss.dll's own CreateFeature through the forwarder, as needed while DL(S)SS runs. A failure is logged
        // with its code and the crop is made through the core as before.
        if (_crop && Config::Instance()->UpscalerFoveaSrDirect.value_or_default())
        {
            NgxDirect::Calls& d = NgxDirect::Get();
            if (d.create == nullptr)
            {
                LOG_WARN("SR crop: a feature past the driver core was asked for ([Inputs] UpscalerFoveaSrDirect), but {}: "
                         "made through the core instead (through the core)",
                         d.reason);
                PerEyeDlss_Dx12::NoteSrCropDirectCreate(false, 0);
            }
            else
            {
                ScopedSkipHeapCapture skipHeapCapture {};
                void* handle = nullptr;
                const int populated = d.populate != nullptr ? d.populate(L"nvngx_dlss.dll", InParameters) : -9;
                const int r = d.create(L"nvngx_dlss.dll", InCommandList, (int) NVSDK_NGX_Feature_SuperSampling,
                                       InParameters, &handle);
                if (r == 1 && handle != nullptr)
                {
                    _p_dlssHandle = (NVSDK_NGX_Handle*) handle;
                    _direct = true;
                    nvResult = NVSDK_NGX_Result_Success;
                    LOG_INFO("SR crop: feature made DIRECTLY on nvngx_dlss.dll's own CreateFeature through the forwarder, "
                             "past the driver core (past the core): populate {}, "
                             "the snippet's handle id {}",
                             populated, _p_dlssHandle->Id);
                }
                else
                {
                    LOG_WARN("SR crop: the DIRECT create on nvngx_dlss.dll FAILED (result {:#x}, populate {}, handle {}): "
                             "made through the core instead (through the core)",
                             (unsigned int) r, populated, handle);
                }
                PerEyeDlss_Dx12::NoteSrCropDirectCreate(_direct, r);
            }
        }
        if (!_direct)
        {
            ScopedSkipHeapCapture skipHeapCapture {};

            nvResult = NVNGXProxy::D3D12_CreateFeature()(InCommandList, NVSDK_NGX_Feature_SuperSampling, InParameters,
                                                         &_p_dlssHandle);
        }

        if (nvResult != NVSDK_NGX_Result_Success)
        {
            LOG_ERROR("_CreateFeature result: {0:X}", (unsigned int) nvResult);
            return false;
        }
        else
        {
            LOG_INFO("_CreateFeature result: NVSDK_NGX_Result_Success");
            PerEyeDlss_Dx12::ReportCreateVram(InParameters, vramBefore, PerEyeDlss_Dx12::VramUsedMB(Device), _direct);
        }
    }
    else
    {
        LOG_ERROR("_CreateFeature is nullptr");
        return false;
    }

    ReadVersion();

    SetInit(true);
    return true;
}

bool DLSSFeatureDx12::EvaluateInternal(ID3D12GraphicsCommandList* InCommandList, NVSDK_NGX_Parameter* InParameters)
{
    if (!_moduleLoaded)
    {
        LOG_ERROR("nvngx.dll or _nvngx.dll is not loaded!");
        return false;
    }

    NVSDK_NGX_Result nvResult;

    if (NVNGXProxy::D3D12_EvaluateFeature() != nullptr)
    {
        ProcessEvaluateParams(InParameters);

        if (_direct)
        {
            // The snippet's own EvaluateFeature through the forwarder (see InitDLSS): never the core's.
            const int r = NgxDirect::Get().evaluate(InCommandList, _p_dlssHandle, InParameters);
            PerEyeDlss_Dx12::NoteSrCropDirectEvaluate(r == 1);
            if (r != 1)
            {
                LOG_ERROR("SR crop: the DIRECT _EvaluateFeature on nvngx_dlss.dll result: {0:X}", (unsigned int) r);
                return false;
            }
            _frameCount++;
            return true;
        }
        if (_crop)
            PerEyeDlss_Dx12::NoteSrCropCoreEvaluate();

        nvResult = NVNGXProxy::D3D12_EvaluateFeature()(InCommandList, _p_dlssHandle, InParameters, NULL);

        if (nvResult != NVSDK_NGX_Result_Success)
        {
            LOG_ERROR("_EvaluateFeature result: {0:X}", (unsigned int) nvResult);
            return false;
        }
    }
    else
    {
        LOG_ERROR("_EvaluateFeature is nullptr");
        return false;
    }

    _frameCount++;

    return true;
}

void DLSSFeatureDx12::Shutdown(ID3D12Device* InDevice)
{
    if (_dlssInitedDx12)
    {
        if (NVNGXProxy::D3D12_Shutdown() != nullptr)
            NVNGXProxy::D3D12_Shutdown()();
        else if (NVNGXProxy::D3D12_Shutdown1() != nullptr)
            NVNGXProxy::D3D12_Shutdown1()(InDevice);
    }

    DLSSFeature::Shutdown();
}

DLSSFeatureDx12::DLSSFeatureDx12(unsigned int InHandleId, NVSDK_NGX_Parameter* InParameters)
    : IFeature(InHandleId, InParameters), IFeature_Dx12(InHandleId, InParameters), DLSSFeature(InHandleId, InParameters)
{
    if (NVNGXProxy::NVNGXModule() == nullptr)
    {
        LOG_INFO("nvngx.dll not loaded, now loading");
        NVNGXProxy::InitNVNGX();
    }

    LOG_INFO("binding complete!");
}

DLSSFeatureDx12::~DLSSFeatureDx12()
{
    if (State::Instance().isShuttingDown)
        return;

    if (_direct)
    {
        if (_p_dlssHandle != nullptr)
        {
            const int r = NgxDirect::Get().release(_p_dlssHandle);
            LOG_INFO("SR crop: the DIRECT feature released on nvngx_dlss.dll (result {})", r);
        }
        return;
    }
    if (NVNGXProxy::D3D12_ReleaseFeature() != nullptr && _p_dlssHandle != nullptr)
        NVNGXProxy::D3D12_ReleaseFeature()(_p_dlssHandle);
}
