#include <pch.h>
#include "DLSSDFeature_Dx12.h"
#include <inputs/PerEyeDlss_Dx12.h>
#include <dxgi1_4.h>
#include <Config.h>
#include <upscalers/NgxDirect_Dx12.h>

using DirectNgx = NgxDirect::Calls;

static DirectNgx& Direct() { return NgxDirect::Get(); }

bool DLSSDFeatureDx12::InitInternal(ID3D12GraphicsCommandList* InCommandList, NVSDK_NGX_Parameter* InParameters)
{
    if (IsInited())
        return true;

    return InitDLSSD(InCommandList, InParameters);
}

bool DLSSDFeatureDx12::InitDLSSD(ID3D12GraphicsCommandList* InCommandList, NVSDK_NGX_Parameter* InParameters)
{
    if (NVNGXProxy::NVNGXModule() == nullptr)
    {
        LOG_ERROR("nvngx.dll not loaded!");
        return false;
    }

    if (!_dlssdInitedDx12)
    {
        _dlssdInitedDx12 = NVNGXProxy::InitDx12(Device);

        if (!_dlssdInitedDx12)
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

    LOG_INFO("Creating DLSSD feature");

    if (NVNGXProxy::D3D12_CreateFeature() != nullptr)
    {
        ProcessInitParams(InParameters);

        // Which RR preset this feature is made with: the hints read back from the block after every override, for
        // the game's features and for ours (a crop of per-eye DLSS has no RR preset of its own: it takes these).
        {
            static const char* const kHints[6] = { "RayReconstruction.Hint.Render.Preset.Performance",
                                                   "RayReconstruction.Hint.Render.Preset.Balanced",
                                                   "RayReconstruction.Hint.Render.Preset.Quality",
                                                   "RayReconstruction.Hint.Render.Preset.UltraPerformance",
                                                   "RayReconstruction.Hint.Render.Preset.UltraQuality",
                                                   "RayReconstruction.Hint.Render.Preset.DLAA" };
            unsigned int hint[6] = {};
            for (int i = 0; i < 6; ++i)
                InParameters->Get(kHints[i], &hint[i]);
            unsigned int mode = 0; // NVSDK_NGX_PerfQuality_Value: the index of kHints
            InParameters->Get(NVSDK_NGX_Parameter_PerfQualityValue, &mode);
            const unsigned int used = mode < 6 ? hint[mode] : 0;
            const int kind = PerEyeDlss_Dx12::KindForCreate();
            LOG_INFO("RR render presets handed to the DLL (0 = its default, 4 = D, 5 = E): Performance {}, Balanced "
                     "{}, Quality {}, UltraPerformance {}, UltraQuality {}, DLAA {}; quality mode {} -> this feature "
                     "is made with RR preset {} (OptiScaler's RR override {}; {})",
                     hint[0], hint[1], hint[2], hint[3], hint[4], hint[5], mode, used,
                     Config::Instance()->DLSSDRenderPresetOverride.value_or_default() ? "ON" : "off",
                     kind == 1   ? "a foveal crop of per-eye DLSS"
                     : kind == 2 ? "a periphery of per-eye DLSS"
                                 : "the game's feature");
            PerEyeDlss_Dx12::NoteRrCreate(kind, (int) used);
        }

        _p_dlssdHandle = &_dlssdHandle;

        NVSDK_NGX_Result nvResult = NVSDK_NGX_Result_Fail;
        _direct = false;
        // A foveal crop past the driver core (see Direct() above): on the snippet's own CreateFeature, through the
        // forwarder. A failure is logged with its code and the crop is made through the core as before.
        if (PerEyeDlss_Dx12::KindForCreate() == 1 && Config::Instance()->UpscalerFoveaRrDirect.value_or_default())
        {
            DirectNgx& d = Direct();
            if (d.create == nullptr)
            {
                LOG_WARN("RR crop: a feature past the driver core was asked for ([Inputs] UpscalerFoveaRrDirect), but {}: "
                         "made through the core instead (through the core)",
                         d.reason);
                PerEyeDlss_Dx12::NoteCropDirectCreate(false, 0);
            }
            else
            {
                ScopedSkipHeapCapture skipHeapCapture {};
                void* handle = nullptr;
                const int populated = d.populate != nullptr ? d.populate(L"nvngx_dlssd.dll", InParameters) : -9;
                const int r = d.create(L"nvngx_dlssd.dll", InCommandList, (int) NVSDK_NGX_Feature_RayReconstruction,
                                       InParameters, &handle);
                if (r == 1 && handle != nullptr)
                {
                    _p_dlssdHandle = (NVSDK_NGX_Handle*) handle;
                    _direct = true;
                    nvResult = NVSDK_NGX_Result_Success;
                    LOG_INFO("RR crop: feature made DIRECTLY on nvngx_dlssd.dll's own CreateFeature through the forwarder, "
                             "past the driver core (past the core): populate {}, "
                             "the snippet's handle id {}",
                             populated, _p_dlssdHandle->Id);
                }
                else
                {
                    LOG_WARN("RR crop: the DIRECT create on nvngx_dlssd.dll FAILED (result {:#x}, populate {}, handle {}): "
                             "made through the core instead (through the core)",
                             (unsigned int) r, populated, handle);
                }
                PerEyeDlss_Dx12::NoteCropDirectCreate(_direct, r);
            }
        }
        if (!_direct)
        {
            ScopedSkipHeapCapture skipHeapCapture {};

            nvResult = NVNGXProxy::D3D12_CreateFeature()(InCommandList, NVSDK_NGX_Feature_RayReconstruction,
                                                         InParameters, &_p_dlssdHandle);
        }

        if (nvResult != NVSDK_NGX_Result_Success)
        {
            LOG_ERROR("_CreateFeature result: {0:X}", (unsigned int) nvResult);
            return false;
        }
        else
        {
            LOG_INFO("_CreateFeature result: NVSDK_NGX_Result_Success, HandleId: {0}", _p_dlssdHandle->Id);
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

bool DLSSDFeatureDx12::EvaluateInternal(ID3D12GraphicsCommandList* InCommandList, NVSDK_NGX_Parameter* InParameters)
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
            // The snippet's own EvaluateFeature through the forwarder (see InitDLSSD): never the core's.
            const int r = Direct().evaluate(InCommandList, _p_dlssdHandle, InParameters);
            PerEyeDlss_Dx12::NoteCropDirectEvaluate(r == 1);
            if (r != 1)
            {
                LOG_ERROR("RR crop: the DIRECT _EvaluateFeature on nvngx_dlssd.dll result: {0:X}", (unsigned int) r);
                return false;
            }
            _frameCount++;
            return true;
        }
        PerEyeDlss_Dx12::NoteCropCoreEvaluate();

        nvResult = NVNGXProxy::D3D12_EvaluateFeature()(InCommandList, _p_dlssdHandle, InParameters, NULL);

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

DLSSDFeatureDx12::DLSSDFeatureDx12(unsigned int InHandleId, NVSDK_NGX_Parameter* InParameters)
    : IFeature(InHandleId, InParameters), IFeature_Dx12(InHandleId, InParameters),
      DLSSDFeature(InHandleId, InParameters)
{
    if (NVNGXProxy::NVNGXModule() == nullptr)
    {
        LOG_INFO("nvngx.dll not loaded, now loading");
        NVNGXProxy::InitNVNGX();
    }

    LOG_INFO("binding complete!");
}

DLSSDFeatureDx12::~DLSSDFeatureDx12()
{
    if (State::Instance().isShuttingDown)
        return;

    if (_direct)
    {
        if (_p_dlssdHandle != nullptr)
        {
            const int r = Direct().release(_p_dlssdHandle);
            LOG_INFO("RR crop: the DIRECT feature released on nvngx_dlssd.dll (result {})", r);
        }
        return;
    }
    if (NVNGXProxy::D3D12_ReleaseFeature() != nullptr && _p_dlssdHandle != nullptr)
        NVNGXProxy::D3D12_ReleaseFeature()(_p_dlssdHandle);
}
