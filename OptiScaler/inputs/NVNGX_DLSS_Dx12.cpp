#include "pch.h"
#include "Util.h"
#include "Config.h"

#include "NVNGX_DLSS.h"
#include "NgxFeatureRegistry.h"
#include "NVNGX_Parameter.h"
#include "proxies/NVNGX_Proxy.h"
#include "dlssnr/DlssNr.h"
#include "dlssnr/DlssNr_ExposureScan.h"
#include "UpscalerInputProbe_Dx12.h"
#include "EvalCapture_Dx12.h"
#include "GiCapture_Dx12.h"
#include "FrameSplit_Dx12.h"
#include "PerEyeDlss_Dx12.h"
#include <upscalers/dlss/DLSSFeature_Dx12.h>
#include <shaders/output_scaling/OS_Dx12.h>

#include <upscalers/FeatureProvider_Dx12.h>
#include "upscalers/dlss/DLSSFeature_Dx12.h"

#include <framegen/nvngx/Nvngx_FG.h>
#include "FG/FSR3_Dx12_FG.h"
#include "FG/Upscaler_Inputs_Dx12.h"

#include <imgui/ImGuiNotify.hpp>

#include <hooks/D3D12_Hooks.h>

#include <dxgi1_4.h>
#include <shared_mutex>
#include <unordered_map>
#include <unordered_set>
#include "detours/detours.h"
#include <ankerl/unordered_dense.h>
#include <misc/IdentifyGpu.h>

static ankerl::unordered_dense::map<unsigned int, ContextData<IFeature_Dx12>> Dx12Contexts;
// Per-eye DLSS: the backend each feature we made (twins, crops, peripheries) was made as, so a rebuild comes back as it.
static std::unordered_map<unsigned int, Upscaler> OwnBackend;
// Per-eye DLSS: the game's features whose twin failed to create at first need (tried once per feature; forgotten with it).
static std::unordered_set<unsigned int> TwinCreateFailed;
static NgxFeatureRegistry HandleToFeature;

static ID3D12Device* D3D12Device = nullptr;
static int evalCounter = 0;
static bool shutdown = false;
static bool _skipInit = false;
static wchar_t const** paths;

class ScopedInitDx12
{
  private:
    bool previousState;

  public:
    ScopedInitDx12()
    {
        previousState = _skipInit;
        _skipInit = true;
    }

    ~ScopedInitDx12() { _skipInit = previousState; }
};

static void UpdateInitPaths(NVSDK_NGX_FeatureCommonInfo* InFeatureInfo)
{
    State::Instance().NVNGX_FeatureInfo_Paths.clear();

    if (InFeatureInfo != nullptr)
    {
        auto exePath = Util::ExePath().remove_filename();

        std::optional<std::filesystem::path> nvngxDlssPath = std::nullopt;
        std::optional<std::filesystem::path> nvngxDlssDPath = std::nullopt;
        std::optional<std::filesystem::path> nvngxDlssGPath = std::nullopt;

        // Check DLSS path
        if (State::Instance().NVNGX_DLSS_Path.has_value())
        {
            nvngxDlssPath = std::filesystem::path(State::Instance().NVNGX_DLSS_Path.value());
        }
        else
        {
            auto path = Util::FindFilePath(exePath, "nvngx_dlss.dll");

            if (path.has_value())
                nvngxDlssPath = path.value();
        }

        // Check DLSS-D path
        if (State::Instance().NVNGX_DLSSD_Path.has_value())
        {
            nvngxDlssDPath = std::filesystem::path(State::Instance().NVNGX_DLSSD_Path.value());
        }
        else
        {
            auto path = Util::FindFilePath(exePath, "nvngx_dlssd.dll");

            if (path.has_value())
                nvngxDlssDPath = path.value();
        }

        // Check DLSS-G path
        if (State::Instance().NVNGX_DLSSG_Path.has_value())
        {
            nvngxDlssGPath = std::filesystem::path(State::Instance().NVNGX_DLSSG_Path.value());
        }
        else
        {
            auto path = Util::FindFilePath(exePath, "nvngx_dlssg.dll");

            if (path.has_value())
                nvngxDlssGPath = path.value();
        }

        // Override locations
        if (Config::Instance()->DLSSFeaturePath.has_value())
            State::Instance().NVNGX_FeatureInfo_Paths.push_back(Config::Instance()->DLSSFeaturePath.value());

        // If DLSS path is overriden
        if (Config::Instance()->NVNGX_DLSS_Library.has_value() && nvngxDlssPath.has_value())
            State::Instance().NVNGX_FeatureInfo_Paths.push_back(nvngxDlssPath.value().parent_path().wstring());

        // OptiDll Path
        State::Instance().NVNGX_FeatureInfo_Paths.push_back(Config::Instance()->MainDllPath.value());

        // Original paths from NVNGX
        for (size_t i = 0; i < InFeatureInfo->PathListInfo.Length; i++)
        {
            const wchar_t* path = InFeatureInfo->PathListInfo.Path[i];
            State::Instance().NVNGX_FeatureInfo_Paths.push_back(std::wstring(path));
        }

        // Exe path
        State::Instance().NVNGX_FeatureInfo_Paths.push_back(exePath.wstring());

        // If DLSS path is not overriden
        if (!Config::Instance()->NVNGX_DLSS_Library.has_value() && nvngxDlssPath.has_value())
            State::Instance().NVNGX_FeatureInfo_Paths.push_back(nvngxDlssPath.value().parent_path().wstring());

        // Add found locations
        if (nvngxDlssDPath.has_value())
            State::Instance().NVNGX_FeatureInfo_Paths.push_back(nvngxDlssDPath.value().parent_path().wstring());

        if (nvngxDlssGPath.has_value())
            State::Instance().NVNGX_FeatureInfo_Paths.push_back(nvngxDlssGPath.value().parent_path().wstring());

        // Build pointer array
        paths = new const wchar_t*[State::Instance().NVNGX_FeatureInfo_Paths.size()];
        for (size_t i = 0; i < State::Instance().NVNGX_FeatureInfo_Paths.size(); ++i)
        {
            paths[i] = State::Instance().NVNGX_FeatureInfo_Paths[i].c_str();
            LOG_DEBUG("Feature Path [{}]: {}", i, wstring_to_string(State::Instance().NVNGX_FeatureInfo_Paths[i]));
        }

        InFeatureInfo->PathListInfo.Path = paths;
        InFeatureInfo->PathListInfo.Length = (int) State::Instance().NVNGX_FeatureInfo_Paths.size();
    }
}

#pragma region DLSS Init Calls

NVSDK_NGX_API NVSDK_NGX_Result NVSDK_NGX_D3D12_Init_Ext(unsigned long long InApplicationId,
                                                        const wchar_t* InApplicationDataPath, ID3D12Device* InDevice,
                                                        NVSDK_NGX_Version InSDKVersion,
                                                        const NVSDK_NGX_FeatureCommonInfo* InFeatureInfo)
{
    LOG_FUNC();

    NVSDK_NGX_FeatureCommonInfo localFeatureInfo = {};

    if (InFeatureInfo != nullptr)
        std::memcpy(&localFeatureInfo, InFeatureInfo, sizeof(NVSDK_NGX_FeatureCommonInfo));

    if (!_skipInit)
        UpdateInitPaths(&localFeatureInfo);

    State::Instance().NVNGX_ApplicationId = InApplicationId;
    State::Instance().NVNGX_ApplicationDataPath = std::wstring(InApplicationDataPath);
    State::Instance().NVNGX_Version = InSDKVersion;
    State::Instance().NVNGX_FeatureInfo = &localFeatureInfo;
    State::Instance().NVNGX_Version = InSDKVersion;

    if (Config::Instance()->DLSSEnabled.value_or_default() && !_skipInit)
    {
        if (Config::Instance()->UseGenericAppIdWithDlss.value_or_default())
            InApplicationId = app_id_override;

        if (NVNGXProxy::NVNGXModule() == nullptr)
            NVNGXProxy::InitNVNGX();

        if (NVNGXProxy::NVNGXModule() != nullptr && NVNGXProxy::D3D12_Init_Ext() != nullptr)
        {
            LOG_INFO("calling NVNGXProxy::D3D12_Init_Ext");

            auto result = NVNGXProxy::D3D12_Init_Ext()(InApplicationId, InApplicationDataPath, InDevice, InSDKVersion,
                                                       &localFeatureInfo);
            LOG_INFO("calling NVNGXProxy::D3D12_Init_Ext result: {0:X}", (UINT) result);

            if (result == NVSDK_NGX_Result_Success)
                NVNGXProxy::SetDx12Inited(true);
        }
        else
        {
            LOG_WARN("NVNGXProxy::NVNGXModule or NVNGXProxy::D3D12_Init_Ext is nullptr!");
        }
    }

    if (InFeatureInfo != nullptr && InSDKVersion > 0x0000013)
        State::Instance().NVNGX_Logger = InFeatureInfo->LoggingInfo;

    if (State::Instance().nvngxDx12Inited && InDevice == D3D12Device)
    {
        LOG_WARN("NVNGX already inited");
        return NVSDK_NGX_Result_Success;
    }

    if (State::Instance().activeFgNvngx != FGNvngxReplacement::None)
    {
        Nvngx_FG::D3D12_Init_Ext(InApplicationId, InApplicationDataPath, InDevice, InSDKVersion, &localFeatureInfo);
    }

    LOG_INFO("AppId: {0}", InApplicationId);
    LOG_INFO("SDK: {0:x}", (unsigned int) InSDKVersion);
    LOG_INFO(L"InApplicationDataPath {0}", std::wstring(InApplicationDataPath));

    D3D12Device = InDevice;
    State::Instance().currentD3D12Device = InDevice;
    D3D12Hooks::HookDevice(InDevice);

    State::Instance().nvngxDx12Inited = true;

    UpscalerInputsDx12::Init(InDevice);

    return NVSDK_NGX_Result_Success;
}

NVSDK_NGX_API NVSDK_NGX_Result NVSDK_NGX_D3D12_Init(unsigned long long InApplicationId,
                                                    const wchar_t* InApplicationDataPath, ID3D12Device* InDevice,
                                                    const NVSDK_NGX_FeatureCommonInfo* InFeatureInfo,
                                                    NVSDK_NGX_Version InSDKVersion)
{
    LOG_FUNC();

    NVSDK_NGX_FeatureCommonInfo localFeatureInfo = {};

    if (InFeatureInfo != nullptr)
        std::memcpy(&localFeatureInfo, InFeatureInfo, sizeof(NVSDK_NGX_FeatureCommonInfo));

    if (!_skipInit)
        UpdateInitPaths(&localFeatureInfo);

    if (Config::Instance()->DLSSEnabled.value_or_default() && !_skipInit)
    {
        if (Config::Instance()->UseGenericAppIdWithDlss.value_or_default())
            InApplicationId = app_id_override;

        if (NVNGXProxy::NVNGXModule() == nullptr)
            NVNGXProxy::InitNVNGX();

        if (NVNGXProxy::NVNGXModule() != nullptr && NVNGXProxy::D3D12_Init() != nullptr)
        {
            LOG_INFO("calling NVNGXProxy::D3D12_Init");

            auto result = NVNGXProxy::D3D12_Init()(InApplicationId, InApplicationDataPath, InDevice, &localFeatureInfo,
                                                   InSDKVersion);

            LOG_INFO("calling NVNGXProxy::D3D12_Init result: {0:X}", (UINT) result);

            if (result == NVSDK_NGX_Result_Success)
                NVNGXProxy::SetDx12Inited(true);
        }
    }

    if (State::Instance().nvngxDx12Inited && InDevice == D3D12Device)
    {
        LOG_WARN("NVNGX already inited");
        return NVSDK_NGX_Result_Success;
    }

    // if (State::Instance().activeFgInput == FGInput::NvngxFG)
    //{
    //     Nvngx_FG::D3D12_Init(InApplicationId, InApplicationDataPath, InDevice, InFeatureInfo, InSDKVersion);
    // }

    ScopedInitDx12 scopedInit {};
    auto result =
        NVSDK_NGX_D3D12_Init_Ext(InApplicationId, InApplicationDataPath, InDevice, InSDKVersion, &localFeatureInfo);

    LOG_DEBUG("was called NVSDK_NGX_D3D12_Init_Ext");
    return result;
}

NVSDK_NGX_API NVSDK_NGX_Result NVSDK_NGX_D3D12_Init_ProjectID(const char* InProjectId,
                                                              NVSDK_NGX_EngineType InEngineType,
                                                              const char* InEngineVersion,
                                                              const wchar_t* InApplicationDataPath,
                                                              ID3D12Device* InDevice, NVSDK_NGX_Version InSDKVersion,
                                                              const NVSDK_NGX_FeatureCommonInfo* InFeatureInfo)
{
    LOG_FUNC();

    NVSDK_NGX_FeatureCommonInfo localFeatureInfo = {};

    if (InFeatureInfo != nullptr)
        std::memcpy(&localFeatureInfo, InFeatureInfo, sizeof(NVSDK_NGX_FeatureCommonInfo));

    if (!_skipInit)
        UpdateInitPaths(&localFeatureInfo);

    if (Config::Instance()->DLSSEnabled.value_or_default() && !_skipInit)
    {
        if (Config::Instance()->UseGenericAppIdWithDlss.value_or_default())
            InProjectId = project_id_override;

        if (NVNGXProxy::NVNGXModule() == nullptr)
            NVNGXProxy::InitNVNGX();

        if (NVNGXProxy::NVNGXModule() != nullptr && NVNGXProxy::D3D12_Init_ProjectID() != nullptr)
        {
            LOG_INFO("calling NVNGXProxy::D3D12_Init_ProjectID");

            auto result =
                NVNGXProxy::D3D12_Init_ProjectID()(InProjectId, InEngineType, InEngineVersion, InApplicationDataPath,
                                                   InDevice, InSDKVersion, &localFeatureInfo);

            LOG_INFO("calling NVNGXProxy::D3D12_Init_ProjectID result: {0:X}", (UINT) result);

            if (result == NVSDK_NGX_Result_Success)
                NVNGXProxy::SetDx12Inited(true);
        }
    }

    LOG_INFO("InProjectId: {0}", InProjectId);
    LOG_INFO("InEngineType: {0}", (int) InEngineType);
    LOG_INFO("InEngineVersion: {0}", InEngineVersion);

    State::Instance().NVNGX_ProjectId = std::string(InProjectId);
    State::Instance().NVNGX_Engine = InEngineType;
    State::Instance().NVNGX_EngineVersion = std::string(InEngineVersion);

    if (State::Instance().nvngxDx12Inited && InDevice == D3D12Device)
    {
        LOG_WARN("NVNGX already inited");
        return NVSDK_NGX_Result_Success;
    }

    ScopedInitDx12 scopedInit {};
    auto result = NVSDK_NGX_D3D12_Init_Ext(0x1337, InApplicationDataPath, InDevice, InSDKVersion, &localFeatureInfo);
    return result;
}

// Not sure about this one, original nvngx does not export this method
NVSDK_NGX_API NVSDK_NGX_Result NVSDK_NGX_D3D12_Init_with_ProjectID(
    const char* InProjectId, NVSDK_NGX_EngineType InEngineType, const char* InEngineVersion,
    const wchar_t* InApplicationDataPath, ID3D12Device* InDevice, const NVSDK_NGX_FeatureCommonInfo* InFeatureInfo,
    NVSDK_NGX_Version InSDKVersion)
{
    LOG_FUNC();

    LOG_INFO("InProjectId: {0}", InProjectId);
    LOG_INFO("InEngineType: {0}", (int) InEngineType);
    LOG_INFO("InEngineVersion: {0}", InEngineVersion);

    State::Instance().NVNGX_ProjectId = std::string(InProjectId);
    State::Instance().NVNGX_Engine = InEngineType;
    State::Instance().NVNGX_EngineVersion = std::string(InEngineVersion);

    if (State::Instance().nvngxDx12Inited)
    {
        LOG_WARN("NVNGX already inited");
        return NVSDK_NGX_Result_Success;
    }

    auto result = NVSDK_NGX_D3D12_Init_Ext(0x1337, InApplicationDataPath, InDevice, InSDKVersion, InFeatureInfo);

    return result;
}

#pragma endregion

#pragma region DLSS Shutdown Calls

static void ReleaseCropParams(bool destroy = true);
static void ReleasePeriphParams(bool destroy = true);

NVSDK_NGX_API NVSDK_NGX_Result NVSDK_NGX_D3D12_Shutdown(void)
{
    // The crop's own parameter block belongs to the NGX instance going away (destroyed before it, or only forgotten while
    // the process exits).
    ReleaseCropParams(!State::Instance().isShuttingDown);
    ReleasePeriphParams(!State::Instance().isShuttingDown);
    shutdown = true;
    State::Instance().nvngxDx12Inited = false;

    D3D12Device = nullptr;

    State::Instance().currentFeature = nullptr;

    // Unhooking and cleaning stuff causing issues during shutdown.
    // Disabled for now to check if it cause any issues
    // UnhookAll();
    DLSSFeatureDx12::Shutdown(D3D12Device);

    // Added `&& !State::Instance().isShuttingDown` hack for crash on exit
    if (Config::Instance()->DLSSEnabled.value_or_default() && NVNGXProxy::IsDx12Inited() &&
        NVNGXProxy::D3D12_Shutdown() != nullptr && !State::Instance().isShuttingDown)
    {
        auto result = NVNGXProxy::D3D12_Shutdown()();
        NVNGXProxy::SetDx12Inited(false);
    }

    // Unhooking and cleaning stuff causing issues during shutdown.
    // Disabled for now to check if it cause any issues
    // HooksDx::UnHook();

    // Disabled to prevent crash
    if (State::Instance().currentFG != nullptr && State::Instance().activeFgInput == FGInput::Upscaler)
    {
        if (State::Instance().isShuttingDown)
            State::Instance().currentFG->Shutdown();
        else
            State::Instance().currentFG->DestroyFGContext();

        State::Instance().clearCapturedHudlesses = true;
    }

    shutdown = false;

    if (State::Instance().activeFgNvngx != FGNvngxReplacement::None)
    {
        Nvngx_FG::D3D12_Shutdown();
    }

    State::Instance().nvngxDx12Inited = false;

    return NVSDK_NGX_Result_Success;
}

NVSDK_NGX_API NVSDK_NGX_Result NVSDK_NGX_D3D12_Shutdown1(ID3D12Device* InDevice)
{
    shutdown = true;
    State::Instance().nvngxDx12Inited = false;

    if (State::Instance().activeFgNvngx != FGNvngxReplacement::None)
    {
        Nvngx_FG::D3D12_Shutdown1(InDevice);
    }

    // Added `&& !State::Instance().isShuttingDown` hack for crash on exit
    if (Config::Instance()->DLSSEnabled.value_or_default() && NVNGXProxy::IsDx12Inited() &&
        NVNGXProxy::D3D12_Shutdown1() != nullptr && !State::Instance().isShuttingDown)
    {
        auto result = NVNGXProxy::D3D12_Shutdown1()(InDevice);
        NVNGXProxy::SetDx12Inited(false);
    }

    return NVSDK_NGX_D3D12_Shutdown();
}

#pragma endregion

#pragma region DLSS Parameter Calls

/**
 * @brief [Deprecated NGX API] Superceeded by NVSDK_NGX_AllocateParameters and NVSDK_NGX_GetCapabilityParameters.
 *
 * Retrieves a common NVSDK parameter map for providing params to the SDK. The lifetime of this
 * map is NOT managed by the application. It is expected to be managed internally by the SDK.
 */
NVSDK_NGX_API NVSDK_NGX_Result NVSDK_NGX_D3D12_GetParameters(NVSDK_NGX_Parameter** OutParameters)
{
    LOG_FUNC();

    if (OutParameters == nullptr)
        return NVSDK_NGX_Result_FAIL_InvalidParameter;

    // If DLSS is enabled and the real DLSS module is loaded, get native NGX table
    if (Config::Instance()->DLSSEnabled.value_or_default() && NVNGXProxy::NVNGXModule() != nullptr &&
        NVNGXProxy::D3D12_GetParameters() != nullptr)
    {
        LOG_INFO("Calling NVNGXProxy::D3D12_GetParameters");
        auto result = NVNGXProxy::D3D12_GetParameters()(OutParameters);
        LOG_INFO("Calling NVNGXProxy::D3D12_GetParameters result: {0:X}, ptr: {1:X}", (UINT) result,
                 (UINT64) *OutParameters);

        // Copy OptiScaler config to real NGX param table
        if (result == NVSDK_NGX_Result_Success)
        {
            InitNGXParameters(*OutParameters, API::DX12);
            SetNGXParamAllocType(*(*OutParameters), NGX_AllocTypes::NVPersistent);
            return NVSDK_NGX_Result_Success;
        }
    }

    // Get custom parameters if using custom backend
    static NVNGX_Parameters oldParams = NVNGX_Parameters(API::DX12, true);
    *OutParameters = &oldParams;
    InitNGXParameters(*OutParameters, API::DX12);

    LOG_DEBUG("Returning custom Opti parameters");

    return NVSDK_NGX_Result_Success;
}

/**
 * @brief Allocates a new NVSDK parameter map pre-populated with NGX capabilities and information about available
 * features. The output parameter map may also be used in the same ways as a parameter map allocated with
 * AllocateParameters(). The lifetime of this map is managed by the calling application with DestroyParameters().
 */
NVSDK_NGX_API NVSDK_NGX_Result NVSDK_NGX_D3D12_GetCapabilityParameters(NVSDK_NGX_Parameter** OutParameters)
{
    LOG_FUNC();

    if (OutParameters == nullptr)
        return NVSDK_NGX_Result_FAIL_InvalidParameter;

    // Get native DLSS params if DLSS is enabled and the module is loaded
    if (Config::Instance()->DLSSEnabled.value_or_default() && NVNGXProxy::NVNGXModule() != nullptr &&
        NVNGXProxy::IsDx12Inited() && NVNGXProxy::D3D12_GetCapabilityParameters() != nullptr)
    {
        LOG_INFO("Calling NVNGXProxy::D3D12_GetCapabilityParameters");
        auto result = NVNGXProxy::D3D12_GetCapabilityParameters()(OutParameters);
        LOG_INFO("Calling NVNGXProxy::D3D12_GetCapabilityParameters result: {0:X}, ptr: {1:X}", (UINT) result,
                 (UINT64) *OutParameters);

        if (result == NVSDK_NGX_Result_Success)
        {
            // Init external NGX table with current configuration and mark as dynamic+external
            InitNGXParameters(*OutParameters, API::DX12);
            SetNGXParamAllocType(*(*OutParameters), NGX_AllocTypes::NVDynamic);
            return NVSDK_NGX_Result_Success;
        }
    }

    // Get custom parameters if using custom backend
    auto& params = *(new NVNGX_Parameters(API::DX12, false));
    InitNGXParameters(&params, API::DX12);
    *OutParameters = &params;

    LOG_DEBUG("Returning custom Opti parameters");

    return NVSDK_NGX_Result_Success;
}

/**
 * @brief Allocates a new parameter map used to provide parameters needed by the DLSS API. The lifetime of this map
 * is managed by the calling application with DestroyParameters().
 */
NVSDK_NGX_API NVSDK_NGX_Result NVSDK_NGX_D3D12_AllocateParameters(NVSDK_NGX_Parameter** OutParameters)
{
    LOG_FUNC();

    if (Config::Instance()->DLSSEnabled.value_or_default() && NVNGXProxy::NVNGXModule() != nullptr &&
        NVNGXProxy::D3D12_AllocateParameters() != nullptr)
    {
        LOG_INFO("Calling NVNGXProxy::D3D12_AllocateParameters");
        auto result = NVNGXProxy::D3D12_AllocateParameters()(OutParameters);
        LOG_INFO("Calling NVNGXProxy::D3D12_AllocateParameters result: {0:X}, ptr: {1:X}", (UINT) result,
                 (UINT64) *OutParameters);

        if (result == NVSDK_NGX_Result_Success)
        {
            SetNGXParamAllocType(*(*OutParameters), NGX_AllocTypes::NVDynamic);
            return result;
        }
    }

    auto* params = new NVNGX_Parameters(API::DX12, false);
    *OutParameters = params;

    return NVSDK_NGX_Result_Success;
}

NVSDK_NGX_API NVSDK_NGX_Result NVSDK_NGX_D3D12_PopulateParameters_Impl(NVSDK_NGX_Parameter* InParameters)
{
    LOG_FUNC();

    if (InParameters == nullptr)
        return NVSDK_NGX_Result_Fail;

    InitNGXParameters(InParameters, API::DX12);

    if (State::Instance().activeFgNvngx != FGNvngxReplacement::None)
    {
        Nvngx_FG::D3D12_PopulateParameters_Impl(InParameters);
    }

    return NVSDK_NGX_Result_Success;
}

/**
 * @brief Destroys a given input parameter map created with AllocateParameters or GetCapabilityParameters.
 Must not be called on maps returned by GetParameters(). Unsupported tables will not be freed.
 */
NVSDK_NGX_API NVSDK_NGX_Result NVSDK_NGX_D3D12_DestroyParameters(NVSDK_NGX_Parameter* InParameters)
{
    LOG_FUNC();

    if (InParameters == nullptr)
        return NVSDK_NGX_Result_Fail;

    const bool isUsingDlss = Config::Instance()->DLSSEnabled.value_or_default() && NVNGXProxy::NVNGXModule();
    const bool success = TryDestroyNGXParameters(InParameters, NVNGXProxy::D3D12_DestroyParameters());

    if (isUsingDlss)
        UpscalerInputsDx12::Reset();

    return success ? NVSDK_NGX_Result_Success : NVSDK_NGX_Result_Fail;
}

#pragma endregion

#pragma region DLSS Feature Calls

static Upscaler GetUpscalerBackend()
{
    Upscaler upscaler = Upscaler::XeSS; // Default

    auto primaryGpu = IdentifyGpu::getPrimaryGpu();

    if (NVNGXProxy::IsDx12Inited() && primaryGpu.dlssCapable)
        upscaler = Upscaler::DLSS;

    if (primaryGpu.fsr4Support != FSR4Support::None)
        upscaler = Upscaler::FFX;

    if (Config::Instance()->Dx12Upscaler.has_value())
        upscaler = Config::Instance()->Dx12Upscaler.value();

    return upscaler;
}

static bool EnsureD3D12Device(ID3D12GraphicsCommandList* cmdList)
{
    if (D3D12Device)
        return true;

    LOG_DEBUG("Get D3D12 device from InCmdList!");

    if (FAILED(cmdList->GetDevice(IID_PPV_ARGS(&D3D12Device))) || !D3D12Device)
    {
        LOG_ERROR("Can't get Dx12Device from InCmdList!");
        return false;
    }

    return true;
}

static NVSDK_NGX_Result TryEvaluateOptiFeature(ID3D12GraphicsCommandList* InCmdList,
                                               const NVSDK_NGX_Handle* InFeatureHandle,
                                               NVSDK_NGX_Parameter* InParameters,
                                               PFN_NVSDK_NGX_ProgressCallback InCallback);

static NVSDK_NGX_Result TryCreateOptiFeature(ID3D12GraphicsCommandList* InCmdList, NVSDK_NGX_Feature InFeatureID,
                                             NVSDK_NGX_Parameter* InParameters, NVSDK_NGX_Handle** OutHandle)
{
    State& state = State::Instance();
    const Config& cfg = *Config::Instance();

    state.api = DX12;

    const uint32_t handleId = IFeature::GetNextHandleId();
    LOG_INFO("Creating OptiScaler feature, HandleId: {}", handleId);

    // Determine backend name
    Upscaler upscalerBackend;
    if (InFeatureID == NVSDK_NGX_Feature_SuperSampling)
    {
        upscalerBackend = GetUpscalerBackend();
        LOG_INFO("Creating {} upscaler feature", UpscalerDisplayName(upscalerBackend));
    }
    else
    {
        upscalerBackend = Upscaler::DLSSD;
        LOG_INFO("Creating DLSSD (Ray Reconstruction) feature");
    }

    // Root signature restoration setup
    const bool restoreCompute = cfg.RestoreComputeSignature.value_or_default();
    const bool restoreGraphics = cfg.RestoreGraphicSignature.value_or_default();
    const bool shouldRestoreSigs = restoreCompute || restoreGraphics;

    // To avoid capturing the upscaler creation
    D3D12Hooks::SetRootSignatureTracking(false);

    if (shouldRestoreSigs)
        D3D12Hooks::HookToCommandListLate(InCmdList);

    // Create context entry
    Dx12Contexts[handleId] = {};

    // Retrieve feature implementation
    if (!FeatureProvider_Dx12::GetFeature(upscalerBackend, handleId, InParameters, &Dx12Contexts[handleId].feature))
    {
        LOG_ERROR("Failed to retrieve feature implementation for '{}'", UpscalerDisplayName(upscalerBackend));

        D3D12Hooks::SetRootSignatureTracking(true);

        Dx12Contexts.erase(handleId);
        return NVSDK_NGX_Result_Fail;
    }

    // Ensure D3D12 device
    if (!EnsureD3D12Device(InCmdList))
    {
        LOG_ERROR("Failed to acquire D3D12 device");

        D3D12Hooks::SetRootSignatureTracking(true);

        // Partial cleanup � handle is allocated but context is incomplete
        Dx12Contexts.erase(handleId);
        return NVSDK_NGX_Result_Fail;
    }

    // Assign handle
    if (*OutHandle == nullptr)
        *OutHandle = new NVSDK_NGX_Handle { handleId };
    else
        (*OutHandle)->Id = handleId;

    state.autoExposure.reset();

    IFeature_Dx12* feature = Dx12Contexts[handleId].feature.get();

    // Initialize feature
    if (feature->Init(D3D12Device, InCmdList, InParameters))
    {
        state.currentFeature = feature;
        evalCounter = 0;
        UpscalerInputsDx12::Reset();
    }
    else
    {
        LOG_ERROR("Feature '{}' initialization failed falling back to FSR 2.1.2", UpscalerDisplayName(upscalerBackend));
        state.newBackend = Upscaler::FSR21;
        state.changeBackend[handleId] = true;
    }

    // Restore root signatures
    if (shouldRestoreSigs)
        D3D12Hooks::RestoreRoot(InCmdList);

    D3D12Hooks::SetRootSignatureTracking(true);

    if (state.activeFgInput == FGInput::Upscaler)
        state.fgChanged = true;

    return NVSDK_NGX_Result_Success;
}

// Per-eye DLSS: a second feature made from the same creation parameters and backend, for the other eye's Presents.
// Made at the first per-eye evaluate (F6; the game's parameter block keeps its creation fields), or at the game's own
// create with [Inputs] UpscalerTwinAtCreate = true (as it used to be: 2.9 GB unused all run under DL(S)SS).
static void CreatePerEyeTwin(ID3D12GraphicsCommandList* InCmdList, unsigned int primary,
                             NVSDK_NGX_Parameter* InParameters)
{
    auto primaryIt = Dx12Contexts.find(primary);
    if (primaryIt == Dx12Contexts.end() || primaryIt->second.feature == nullptr || !primaryIt->second.feature->IsInited())
    {
        LOG_WARN("Per-eye DLSS DX12: feature {} is not initialised; no twin, both eyes share it", primary);
        return;
    }
    const Upscaler backend = primaryIt->second.feature->GetUpscalerType();
    IFeature* const currentBefore = State::Instance().currentFeature;

    const uint32_t twinId = IFeature::GetNextHandleId();
    D3D12Hooks::SetRootSignatureTracking(false);
    Dx12Contexts[twinId] = {};
    PerEyeDlss_Dx12::LabelNextCreate("twin");
    if (!FeatureProvider_Dx12::GetFeature(backend, twinId, InParameters, &Dx12Contexts[twinId].feature) ||
        Dx12Contexts[twinId].feature == nullptr)
    {
        LOG_ERROR("Per-eye DLSS DX12: can't create the twin of feature {} ({}); both eyes share it", primary,
                  UpscalerDisplayName(backend));
        Dx12Contexts.erase(twinId);
        PerEyeDlss_Dx12::LabelNextCreate(nullptr);
        D3D12Hooks::SetRootSignatureTracking(true);
        return;
    }
    if (!Dx12Contexts[twinId].feature->Init(D3D12Device, InCmdList, InParameters))
    {
        LOG_ERROR("Per-eye DLSS DX12: the twin of feature {} failed to initialise; both eyes share it", primary);
        Dx12Contexts.erase(twinId);
        PerEyeDlss_Dx12::LabelNextCreate(nullptr);
        D3D12Hooks::SetRootSignatureTracking(true);
        State::Instance().currentFeature = currentBefore;
        return;
    }
    D3D12Hooks::SetRootSignatureTracking(true);
    State::Instance().currentFeature = currentBefore;
    PerEyeDlss_Dx12::LabelNextCreate(nullptr);
    OwnBackend[twinId] = backend;
    LOG_INFO("Per-eye DLSS DX12: created twin {} of feature {} ({})", twinId, primary, UpscalerDisplayName(backend));
    PerEyeDlss_Dx12::RegisterTwin(primary, twinId);
}

// Per-eye DLSS, foveal crop: a DLSS feature for the crop of `primary`'s picture, at the crop's sizes and preset, made
// with the evaluate's parameters (their creation fields set for it and handed back after). On first F4 only.
// The game's feature a crop / periphery is made from: there and initialised. It is not while OptiScaler rebuilds it (a
// preset applied from the overlay). That is no failure of the create: the caller draws nothing of ours for this
// evaluate and asks again at the next one (Crimson Desert: a periphery asked for in that moment was refused without a
// word and stayed off for the session).
static bool FoveaPrimaryReady(unsigned int primary)
{
    auto it = Dx12Contexts.find(primary);
    return it != Dx12Contexts.end() && it->second.feature != nullptr && it->second.feature->IsInited();
}

static unsigned int CreateFoveaCrop(ID3D12GraphicsCommandList* InCmdList, unsigned int primary,
                                    NVSDK_NGX_Parameter* InParameters, const PerEyeDlss_Dx12::FoveaSize& size,
                                    bool asSr = false, bool periphery = false)
{
    // The callers ask FoveaPrimaryReady first, so this is not expected: said every time it happens.
    if (!FoveaPrimaryReady(primary))
    {
        LOG_WARN("Per-eye DLSS DX12: feature {} is not there or not initialised; no crop / periphery made from it",
                 primary);
        return 0;
    }
    auto primaryIt = Dx12Contexts.find(primary);
    IFeature_Dx12* primaryFeature = primaryIt->second.feature.get();
    IFeature* const currentBefore = State::Instance().currentFeature;
    const Upscaler backend = primaryFeature->GetUpscalerType();
    if (backend != Upscaler::DLSS && backend != Upscaler::DLSSD)
    {
        LOG_WARN("Per-eye DLSS DX12: feature {} is {}, not DLSS; no crop / periphery", primary,
                 UpscalerDisplayName(backend));
        return 0;
    }

    unsigned int flags = 0;
    if (primaryFeature->DepthInverted())
        flags |= NVSDK_NGX_DLSS_Feature_Flags_DepthInverted;
    if (primaryFeature->AutoExposure())
        flags |= NVSDK_NGX_DLSS_Feature_Flags_AutoExposure;
    if (primaryFeature->IsHdr())
        flags |= NVSDK_NGX_DLSS_Feature_Flags_IsHDR;
    if (primaryFeature->JitteredMV())
        flags |= NVSDK_NGX_DLSS_Feature_Flags_MVJittered;
    if (primaryFeature->LowResMV())
        flags |= NVSDK_NGX_DLSS_Feature_Flags_MVLowRes;

    const char* sizeNames[4] = { NVSDK_NGX_Parameter_Width, NVSDK_NGX_Parameter_Height, NVSDK_NGX_Parameter_OutWidth,
                                 NVSDK_NGX_Parameter_OutHeight };
    const char* presetNames[6] = { NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_DLAA,
                                   NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_UltraQuality,
                                   NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Quality,
                                   NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Balanced,
                                   NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Performance,
                                   NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_UltraPerformance };
    unsigned int savedSizes[4] {}, savedPresets[6] {}, savedFlags = 0;
    for (int i = 0; i < 4; ++i)
        InParameters->Get(sizeNames[i], &savedSizes[i]);
    for (int i = 0; i < 6; ++i)
        InParameters->Get(presetNames[i], &savedPresets[i]);
    InParameters->Get(NVSDK_NGX_Parameter_DLSS_Feature_Create_Flags, &savedFlags);

    InParameters->Set(NVSDK_NGX_Parameter_Width, size.renderW);
    InParameters->Set(NVSDK_NGX_Parameter_Height, size.renderH);
    InParameters->Set(NVSDK_NGX_Parameter_OutWidth, size.outW);
    InParameters->Set(NVSDK_NGX_Parameter_OutHeight, size.outH);
    InParameters->Set(NVSDK_NGX_Parameter_DLSS_Feature_Create_Flags, flags);
    PerEyeDlss_Dx12::SetPresetForCreate(size.preset);
    // What this create is (1 crop, 2 periphery): DLSSDFeatureDx12's and DLSSFeatureDx12's creates read it (the RR preset
    // line; the direct path on the snippet DLL for a crop). It used to be `asSr ? 2 : 1`, so a periphery of the
    // game's own kind was made as a CROP: with the direct path on it would have gone that way too.
    PerEyeDlss_Dx12::SetKindForCreate(periphery ? 2 : 1);

    const uint32_t cropId = IFeature::GetNextHandleId();
    D3D12Hooks::SetRootSignatureTracking(false);
    Dx12Contexts[cropId] = {};
    // `asSr`: an SR feature over an RR game's inputs (the SR periphery in RR mode); recorded as the game's kind, so the
    // rebuild rule can switch it back.
    bool ok = FeatureProvider_Dx12::GetFeature(asSr ? Upscaler::DLSS : backend, cropId, InParameters,
                                               &Dx12Contexts[cropId].feature) &&
              Dx12Contexts[cropId].feature != nullptr &&
              Dx12Contexts[cropId].feature->Init(D3D12Device, InCmdList, InParameters);
    D3D12Hooks::SetRootSignatureTracking(true);

    PerEyeDlss_Dx12::SetPresetForCreate(0);
    PerEyeDlss_Dx12::SetKindForCreate(0);
    for (int i = 0; i < 4; ++i)
        InParameters->Set(sizeNames[i], savedSizes[i]);
    for (int i = 0; i < 6; ++i)
        InParameters->Set(presetNames[i], savedPresets[i]);
    InParameters->Set(NVSDK_NGX_Parameter_DLSS_Feature_Create_Flags, savedFlags);
    State::Instance().currentFeature = currentBefore;

    if (!ok)
    {
        Dx12Contexts.erase(cropId);
        LOG_ERROR("Per-eye DLSS DX12: the foveal crop / periphery feature for {} could not be made", primary);
        return 0;
    }
    OwnBackend[cropId] = backend;
    return cropId;
}

static NVSDK_NGX_Result TryEvaluateOptiFeature(ID3D12GraphicsCommandList* InCmdList,
                                               const NVSDK_NGX_Handle* InFeatureHandle,
                                               NVSDK_NGX_Parameter* InParameters,
                                               PFN_NVSDK_NGX_ProgressCallback InCallback);

// What a feature the per-eye path rebuilds must come back as: its own record, else the game's feature kind.
static Upscaler RebuildBackend(unsigned int handle)
{
    if (auto it = OwnBackend.find(handle); it != OwnBackend.end())
    {
        // A periphery of an RR game is an SR feature while the overlay's SR periphery is on.
        if (it->second == Upscaler::DLSSD && PerEyeDlss_Dx12::IsPeriphery(handle) && PerEyeDlss_Dx12::PeripherySr())
            return Upscaler::DLSS;
        return it->second;
    }
    const auto tracked = HandleToFeature.Read(handle);
    if (tracked.feature && *tracked.feature == NVSDK_NGX_Feature_RayReconstruction)
        return Upscaler::DLSSD;
    return Upscaler::DLSS;
}

// The periphery's own parameter block (needed under alternate-eye rendering, found in Crimson Desert: a periphery made
// on the GAME's block -- an SR feature on the RR game's block -- had its evaluates fail on alternate Presents). Like
// the crop's: filled from the game's block each evaluate, the game's only read; what the pass below us leaves in it
// is handed down.
static NVSDK_NGX_Parameter* PeriphParams = nullptr;
// In an RR game the FIRST SR feature made (ours: the periphery) misbehaved under alternate-eye rendering: one parity of its
// evaluates and the game's pass on the same command list failed, shimmer everywhere (Crimson Desert, four
// runs). Every SR feature made AFTER the first one ran clean. So in an RR game a THROW-AWAY SR feature is made
// first, never evaluated and kept for the process (it takes the first slot; ~113 MB), and the periphery gets the next
// one. Not needed in an SR game (the game's own feature is the SR snippet's first).
static unsigned int ThrowAwaySrFirst = 0;
static bool ThrowAwaySrTried = false;

// Per-eye DLSS, cheap periphery: `handle`'s picture from a render-size DLSS feature (made at the first need) enlarged
// over the eye, in place of the full pass. False = it did not draw (the caller runs the full pass).
static bool EvaluatePeriphery(ID3D12GraphicsCommandList* InCmdList, NVSDK_NGX_Parameter* InParameters,
                              unsigned int handle)
{
    // Per eye: our own TAA in place of the DLSS periphery (any reason it cannot run falls through to that).
    if (PerEyeDlss_Dx12::PeripheryTaa() && PerEyeDlss_Dx12::TaaPeriphery(InCmdList, InParameters, handle))
        return true;
    PerEyeDlss_Dx12::FoveaSize size {};
    if (!PerEyeDlss_Dx12::PlanPeriphery(handle, &size))
        return false;
    // From here the periphery's create, its evaluate and the restores are on OUR block (see PeriphParams above).
    if (PeriphParams == nullptr && (NVSDK_NGX_D3D12_AllocateParameters(&PeriphParams) != NVSDK_NGX_Result_Success ||
                                    PeriphParams == nullptr))
    {
        PeriphParams = nullptr;
        static bool logged = false;
        if (!logged)
        {
            logged = true;
            LOG_WARN("Per-eye DLSS DX12: cheap periphery skipped: no parameter block of its own could be allocated (the "
                     "full pass runs)");
        }
        return false;
    }
    if (!PerEyeDlss_Dx12::CopyCropParameters(InParameters, PeriphParams))
        return false;
    NVSDK_NGX_Parameter* const GameParameters = InParameters;
    InParameters = PeriphParams;
    unsigned int periph = PerEyeDlss_Dx12::PeripheryOf(handle);
    if (periph == 0)
    {
        if (!FoveaPrimaryReady(handle))
        {
            PerEyeDlss_Dx12::PeripheryNotReady();
            return false;
        }
        // The throw-away SR feature first (see ThrowAwaySrFirst): an SR periphery in an RR game, once per process.
        if (!ThrowAwaySrTried && PerEyeDlss_Dx12::PeripherySr())
        {
            ThrowAwaySrTried = true;
            auto gameIt = Dx12Contexts.find(handle);
            const bool gameIsRr = gameIt != Dx12Contexts.end() && gameIt->second.feature != nullptr &&
                                  gameIt->second.feature->GetUpscalerType() == Upscaler::DLSSD;
            if (gameIsRr)
            {
                ThrowAwaySrFirst = CreateFoveaCrop(InCmdList, handle, InParameters, size, true, true);
                if (ThrowAwaySrFirst != 0)
                    LOG_INFO("Per-eye DLSS DX12: a THROW-AWAY SR feature {} made before the periphery in this RR game: it "
                             "takes the first SR slot, which the periphery must not take in an RR game; never "
                             "evaluated, kept for the process (its VRAM line above is this one, not the periphery's)",
                             ThrowAwaySrFirst);
                else
                    LOG_WARN("Per-eye DLSS DX12: the throw-away SR feature could not be made; the periphery is made as "
                             "before (the first SR feature may not work in an RR game)");
            }
        }
        periph = CreateFoveaCrop(InCmdList, handle, InParameters, size, PerEyeDlss_Dx12::PeripherySr(), true);
        if (periph == 0)
        {
            PerEyeDlss_Dx12::PeripheryCreateFailed();
            return false;
        }
        PerEyeDlss_Dx12::PeripheryCreated(handle, periph);
    }
    if (!PerEyeDlss_Dx12::BeginPeriphery(InCmdList, InParameters, periph, handle))
        return false;
    auto& state = State::Instance();
    IFeature* const currentBefore = state.currentFeature; // restored as it was, never set (see EvaluateFoveaCrop)
    if (state.changeBackend[periph])
        state.newBackend = RebuildBackend(periph);
    NVSDK_NGX_Handle periphHandle { periph };
    PerEyeDlss_Dx12::MainVectorsBefore(InParameters);
    const NVSDK_NGX_Result result = TryEvaluateOptiFeature(InCmdList, &periphHandle, InParameters, nullptr);
    PerEyeDlss_Dx12::MainVectorsAfter(InParameters, result == NVSDK_NGX_Result_Success);
    // The vectors from below are in OUR block now: the crop's copy and NR after SR read the game's.
    if (result == NVSDK_NGX_Result_Success)
        PerEyeDlss_Dx12::HandBelowVectorsToGame(GameParameters);
    PerEyeDlss_Dx12::TakeDlssRebuild(periph, state.changeBackend[periph]);
    const bool drawn = PerEyeDlss_Dx12::EndPeriphery(InCmdList, InParameters, result == NVSDK_NGX_Result_Success,
                                                     state.changeBackend[periph]);
    state.currentFeature = currentBefore;
    return drawn;
}

// Per-eye DLSS, foveal crop: after the full pass (or the cheap periphery) of `handle`, its crop feature over the centre,
// at the crop PlanFovea made for this evaluate.
// The crop's own parameter block (PerEyeDlss_Dx12::CopyCropParameters): made at the first crop, destroyed with the crops.
static NVSDK_NGX_Parameter* CropParams = nullptr;

static void ReleaseCropParams(bool destroy)
{
    if (CropParams == nullptr)
        return;
    if (destroy)
        TryDestroyNGXParameters(CropParams, NVNGXProxy::D3D12_DestroyParameters());
    CropParams = nullptr;
    LOG_INFO("Per-eye DLSS DX12: the crop's own parameter block released");
}

static void ReleasePeriphParams(bool destroy)
{
    if (PeriphParams == nullptr)
        return;
    if (destroy)
        TryDestroyNGXParameters(PeriphParams, NVNGXProxy::D3D12_DestroyParameters());
    PeriphParams = nullptr;
    LOG_INFO("Per-eye DLSS DX12: the periphery's own parameter block released");
}

static void EvaluateFoveaCrop(ID3D12GraphicsCommandList* InCmdList, NVSDK_NGX_Parameter* GameParameters,
                              unsigned int handle, const PerEyeDlss_Dx12::FoveaSize& size, bool rayReconstruction)
{
    // Everything below -- the crop's creation, its evaluate, the restores -- is on our own block; the game's is only read.
    if (CropParams == nullptr && (NVSDK_NGX_D3D12_AllocateParameters(&CropParams) != NVSDK_NGX_Result_Success ||
                                  CropParams == nullptr))
    {
        CropParams = nullptr;
        static bool logged = false;
        if (!logged)
        {
            logged = true;
            LOG_WARN("Per-eye DLSS DX12: crop skipped: no parameter block of its own could be allocated");
        }
        return;
    }
    if (!PerEyeDlss_Dx12::CopyCropParameters(GameParameters, CropParams))
        return;
    PerEyeDlss_Dx12::MarkCropBlock(CropParams);
    NVSDK_NGX_Parameter* const InParameters = CropParams;
    unsigned int crop = PerEyeDlss_Dx12::FoveaCropOf(handle);
    if (crop == 0)
    {
        if (!FoveaPrimaryReady(handle))
        {
            PerEyeDlss_Dx12::FoveaNotReady();
            return;
        }
        crop = CreateFoveaCrop(InCmdList, handle, InParameters, size);
        if (crop == 0)
        {
            PerEyeDlss_Dx12::FoveaCreateFailed();
            return;
        }
        PerEyeDlss_Dx12::FoveaCreated(handle, crop);
    }
    if (!PerEyeDlss_Dx12::BeginFovea(InCmdList, InParameters, crop, rayReconstruction))
        return;
    NVSDK_NGX_Handle cropHandle { crop };
    // A crop being rebuilt (a live preset change) must come back as DLSS: OptiScaler resets its global new-backend
    // choice after every finished rebuild, and its default for DX12 is XeSS. The rebuild's steps return without
    // evaluating until the last one, which evaluates: while the flag is still set afterwards, draw nothing.
    auto& state = State::Instance();
    // Whatever was current before the crop ran is what stays current: while a feature is mid-rebuild OptiScaler keeps
    // this null on purpose (the half-built feature has no GPU timer yet), and Present reads it -- setting it to the
    // game's feature unconditionally here crashed Present during the overlay's rebuild of every feature.
    IFeature* const currentBefore = state.currentFeature;
    if (state.changeBackend[crop])
        state.newBackend = RebuildBackend(crop);
    const NVSDK_NGX_Result result = TryEvaluateOptiFeature(InCmdList, &cropHandle, InParameters, nullptr);
    PerEyeDlss_Dx12::TakeDlssRebuild(crop, state.changeBackend[crop]);
    PerEyeDlss_Dx12::EndFovea(InCmdList, InParameters, result == NVSDK_NGX_Result_Success, state.changeBackend[crop]);
    state.currentFeature = currentBefore;
}

/**
 * @brief Instantiates a new feature based on the given unique feature ID and param table and
 * provides a handle used to reference the feature elsewhere in the API. Currently supports
 * various TSR and Frame Generation algorithms, including a special case for DLSS-RR passthrough.
 */
NVSDK_NGX_API NVSDK_NGX_Result NVSDK_NGX_D3D12_CreateFeature(ID3D12GraphicsCommandList* InCmdList,
                                                             NVSDK_NGX_Feature InFeatureID,
                                                             NVSDK_NGX_Parameter* InParameters,
                                                             NVSDK_NGX_Handle** OutHandle)
{
    LOG_FUNC();

    if (!InCmdList)
    {
        LOG_ERROR("InCmdList is null");
        return NVSDK_NGX_Result_Fail;
    }

    if (!OutHandle)
    {
        LOG_ERROR("OutHandle is null");
        return NVSDK_NGX_Result_Fail;
    }

    const State& state = State::Instance();
    const Config& cfg = *Config::Instance();

    // DLSSG replacements passthrough
    if (State::Instance().activeFgNvngx != FGNvngxReplacement::None && Nvngx_FG::isDx12Available() &&
        InFeatureID == NVSDK_NGX_Feature_FrameGeneration)
    {
        LOG_INFO("Passthrough to DLSSG Replacement's CreateFeature for FrameGeneration");

        NVSDK_NGX_Result res = Nvngx_FG::D3D12_CreateFeature(InCmdList, InFeatureID, InParameters, OutHandle);

        if (res == NVSDK_NGX_Result_Success && *OutHandle)
        {
            LOG_INFO("Created modded DLSSG feature with HandleId: {}", (*OutHandle)->Id);
            HandleToFeature.Record((*OutHandle)->Id, InFeatureID);
        }

        return res;
    }

    // Native DLSS passthrough (exclude SuperSampling and RayReconstruction)
    if (InFeatureID != NVSDK_NGX_Feature_SuperSampling && InFeatureID != NVSDK_NGX_Feature_RayReconstruction)
    {
        if (cfg.DLSSEnabled.value_or_default() && NVNGXProxy::InitDx12(D3D12Device) &&
            NVNGXProxy::D3D12_CreateFeature() != nullptr)
        {
            LOG_INFO("Passthrough to native NGX CreateFeature for feature {}", (int) InFeatureID);

            NVSDK_NGX_Result res = NVNGXProxy::D3D12_CreateFeature()(InCmdList, InFeatureID, InParameters, OutHandle);

            if (res == NVSDK_NGX_Result_Success && *OutHandle)
            {
                LOG_INFO("Native CreateFeature success, HandleId: {}", (*OutHandle)->Id);
                HandleToFeature.Record((*OutHandle)->Id, InFeatureID);
                UpscalerInputProbe_Dx12::OnCreate((int) InFeatureID, (*OutHandle)->Id, false, InParameters);
            }
            else
            {
                LOG_INFO("Native CreateFeature failed: 0x{:X}", (uint32_t) res);
            }

            return res;
        }

        LOG_WARN("Native DLSS passthrough not available for feature {}", (int) InFeatureID);
        return NVSDK_NGX_Result_FAIL_FeatureNotSupported;
    }

    // OptiScaler internal handling (SuperSampling or RayReconstruction)
    auto tryResult = TryCreateOptiFeature(InCmdList, InFeatureID, InParameters, OutHandle);

    if (tryResult == NVSDK_NGX_Result_Success && *OutHandle)
    {
        HandleToFeature.Record((*OutHandle)->Id, InFeatureID);
        UpscalerInputProbe_Dx12::OnCreate((int) InFeatureID, (*OutHandle)->Id, true, InParameters);
        if (PerEyeDlss_Dx12::Enabled())
        {
            if (Config::Instance()->UpscalerTwinAtCreate.value_or_default())
                CreatePerEyeTwin(InCmdList, (*OutHandle)->Id, InParameters);
            else
                LOG_INFO("Per-eye DLSS DX12: no twin at the game's create of feature {} (UpscalerTwinAtCreate false): "
                         "made at the first per-eye evaluate",
                         (*OutHandle)->Id);
        }
    }

    return tryResult;
}

NVSDK_NGX_API NVSDK_NGX_Result NVSDK_NGX_D3D12_ReleaseFeature(NVSDK_NGX_Handle* InHandle)
{
    LOG_FUNC();

    if (!InHandle)
        return NVSDK_NGX_Result_Success;

    // Before any feature's resources are freed, drop the exposure scan's references to whatever it
    // captured. The scan AddRef's candidates and never released them; a Streamline/DLSS-D resource it
    // pinned would otherwise be used after its heap is freed here -- the Cyberpunk device-removal.
    // Capture is gated on NR being enabled (not the scan source), so this drops whatever was captured
    // whenever NR is on; a no-op only when NR is off.
    DlssNr::ExposureScan::ReleaseTrackedResources();

    auto handleId = InHandle->Id;

    // Clean up framegen
    if (State::Instance().currentFG != nullptr && State::Instance().activeFgInput == FGInput::Upscaler)
    {
        State::Instance().fgChanged = true;
        State::Instance().currentFG->DestroyFGContext();
        State::Instance().clearCapturedHudlesses = true;
        UpscalerInputsDx12::Reset();
    }

    if (!shutdown)
        LOG_INFO("releasing feature with id {0}", handleId);

    // OptiScaler handles start after this offset. If it's outside this range, it doesn't belong to OptiScaler.
    if (handleId < DLSS_MOD_ID_OFFSET)
    {
        if (Config::Instance()->DLSSEnabled.value_or_default() && NVNGXProxy::D3D12_ReleaseFeature() != nullptr)
        {
            if (!shutdown)
                LOG_INFO("calling D3D12_ReleaseFeature for ({0})", handleId);

            // Clean up real DLSS feature
            auto result = NVNGXProxy::D3D12_ReleaseFeature()(InHandle);

            if (!shutdown)
                LOG_INFO("D3D12_ReleaseFeature result for ({0}): {1:X}", handleId, (UINT) result);

            if (result == NVSDK_NGX_Result_Success)
                HandleToFeature.Forget(handleId);
            return result;
        }
        else
        {
            if (!shutdown)
                LOG_INFO("D3D12_ReleaseFeature not available for ({0})", handleId);

            return NVSDK_NGX_Result_FAIL_FeatureNotFound;
        }
    }
    // Clean up OptiScaler feature with framegen
    else if (State::Instance().activeFgNvngx != FGNvngxReplacement::None && handleId >= NVNGX_PROVIDER_ID_OFFSET)
    {
        LOG_INFO("D3D12_ReleaseFeature modded DLSSG with HandleId: {0}", handleId);
        const auto result = Nvngx_FG::D3D12_ReleaseFeature(InHandle);
        if (result == NVSDK_NGX_Result_Success)
            HandleToFeature.Forget(handleId);
        return result;
    }

    // Per-eye DLSS: the foveal crops go with the features they were made for.
    for (unsigned int owner : { handleId, PerEyeDlss_Dx12::TwinOf(handleId) })
    {
        if (owner == 0)
            continue;
        for (unsigned int ownId : PerEyeDlss_Dx12::ForgetFoveaFeatures(owner))
        {
            if (ownId == 0)
                continue;
            if (auto ownIt = Dx12Contexts.find(ownId); ownIt != Dx12Contexts.end())
            {
                if (ownIt->second.feature.get() == State::Instance().currentFeature)
                    State::Instance().currentFeature = nullptr;
                LOG_INFO("Per-eye DLSS DX12: releasing our feature {} (crop / periphery) with feature {}", ownId, owner);
                Dx12Contexts.erase(ownIt);
                OwnBackend.erase(ownId);
                ReleaseCropParams(); // made again at the next crop (a shared block between two crops is refilled each time)
                ReleasePeriphParams(); // likewise the periphery's (refilled from the game's block each evaluate)
            }
        }
    }

    // Per-eye DLSS: the twin goes with the feature it was made for (and a failed first-need create is forgotten with it).
    TwinCreateFailed.erase(handleId);
    if (auto twinId = PerEyeDlss_Dx12::TwinOf(handleId); twinId != 0)
    {
        PerEyeDlss_Dx12::Forget(handleId);
        if (auto twinIt = Dx12Contexts.find(twinId); twinIt != Dx12Contexts.end())
        {
            if (twinIt->second.feature.get() == State::Instance().currentFeature)
                State::Instance().currentFeature = nullptr;
            LOG_INFO("Per-eye DLSS DX12: releasing twin {} with feature {}", twinId, handleId);
            Dx12Contexts.erase(twinIt);
        }
    }

    // Remove feature from context map
    if (auto it = Dx12Contexts.find(handleId); it != Dx12Contexts.end())
    {
        auto& entry = it->second;

        if (auto* deviceContext = entry.feature.get())
        {
            // Clear global reference if it matches
            if (deviceContext == State::Instance().currentFeature)
                State::Instance().currentFeature = nullptr;

            // Erase from map (smart pointer reset is implicit on erase)
            Dx12Contexts.erase(it);
        }
    }
    else
    {
        // Fallback Error Handling
        if (!shutdown)
            LOG_ERROR("can't release feature with id {0}!", handleId);
    }

    HandleToFeature.Forget(handleId);
    return NVSDK_NGX_Result_Success;
}

/**
 * @brief Used by the client application to check for feature support.
 * @param Adapter Device the feature is for.
 * @param FeatureDiscoveryInfo Specifies the feature being queried.
 * @param OutSupported Used to indicate whether a feature is supported and its requirements.
 */
NVSDK_NGX_API NVSDK_NGX_Result NVSDK_NGX_D3D12_GetFeatureRequirements(
    IDXGIAdapter* Adapter, const NVSDK_NGX_FeatureDiscoveryInfo* FeatureDiscoveryInfo,
    NVSDK_NGX_FeatureRequirement* OutSupported)
{
    LOG_DEBUG("for ({0})", (int) FeatureDiscoveryInfo->FeatureID);

    const bool isUpscaling = FeatureDiscoveryInfo->FeatureID == NVSDK_NGX_Feature_SuperSampling;
    const bool isFG = FeatureDiscoveryInfo->FeatureID == NVSDK_NGX_Feature_FrameGeneration;
    const bool dlssgAdjacent = Nvngx_FG::isDx12Available() || State::Instance().activeFgInput == FGInput::DLSSG;

    if (isUpscaling || (isFG && dlssgAdjacent))
    {
        if (OutSupported == nullptr)
        {
            static auto tmp = NVSDK_NGX_FeatureRequirement();
            OutSupported = &tmp;
        }

        OutSupported->FeatureSupported = NVSDK_NGX_FeatureSupportResult_Supported;
        OutSupported->MinHWArchitecture = 0;

        // Some old windows 10 os version
        strcpy_s(OutSupported->MinOSVersion, "10.0.10240.16384");
        return NVSDK_NGX_Result_Success;
    }

    if (Config::Instance()->DLSSEnabled.value_or_default() && IdentifyGpu::getPrimaryGpu().dlssCapable &&
        NVNGXProxy::NVNGXModule() == nullptr)
    {
        NVNGXProxy::InitNVNGX();
    }

    if (Config::Instance()->DLSSEnabled.value_or_default() && IdentifyGpu::getPrimaryGpu().dlssCapable &&
        NVNGXProxy::D3D12_GetFeatureRequirements() != nullptr)
    {
        LOG_DEBUG("D3D12_GetFeatureRequirements for ({0})", (int) FeatureDiscoveryInfo->FeatureID);
        auto result = NVNGXProxy::D3D12_GetFeatureRequirements()(Adapter, FeatureDiscoveryInfo, OutSupported);
        LOG_DEBUG("D3D12_GetFeatureRequirements result for ({0}): {1:X}", (int) FeatureDiscoveryInfo->FeatureID,
                  (UINT) result);

        return result;
    }
    else
    {
        LOG_DEBUG("D3D12_GetFeatureRequirements not available for ({0})", (int) FeatureDiscoveryInfo->FeatureID);
    }

    OutSupported->FeatureSupported = NVSDK_NGX_FeatureSupportResult_AdapterUnsupported;
    return NVSDK_NGX_Result_FAIL_FeatureNotSupported;
}

static NVSDK_NGX_Result TryEvaluateOptiFeature(ID3D12GraphicsCommandList* InCmdList,
                                               const NVSDK_NGX_Handle* InFeatureHandle,
                                               NVSDK_NGX_Parameter* InParameters,
                                               PFN_NVSDK_NGX_ProgressCallback InCallback)
{
    State& state = State::Instance();
    const Config& cfg = *Config::Instance();
    const uint32_t handleId = InFeatureHandle->Id;

    auto ctxIt = Dx12Contexts.find(handleId);

    if (ctxIt == Dx12Contexts.end())
    {
        LOG_WARN("No context found for handle {}", handleId);
        return NVSDK_NGX_Result_FAIL_FeatureNotFound;
    }

    ContextData<IFeature_Dx12>& ctxData = ctxIt->second;
    IFeature_Dx12* feature = ctxData.feature.get();

    if (feature == nullptr) // Prevent source api name flicker when dlssg is active
        state.setInputApiName = state.currentInputApiName;

    const auto targetApiName =
        !state.setInputApiName.has_value() ? ApiUpscalerInput::DLSS_DX12 : state.setInputApiName.value();

    if (state.currentInputApiName != targetApiName)
        state.currentInputApiName = targetApiName;

    state.setInputApiName.reset();
    evalCounter++;

    // Skip evaluation for the first N frames if configured
    if (cfg.SkipFirstFrames.has_value() && evalCounter < cfg.SkipFirstFrames.value())
        return NVSDK_NGX_Result_Success;

    // Root signature restoration setup
    const bool restoreCompute = cfg.RestoreComputeSignature.value_or_default();
    const bool restoreGraphics = cfg.RestoreGraphicSignature.value_or_default();
    const bool shouldRestoreSigs = restoreCompute || restoreGraphics;

    if (shouldRestoreSigs)
    {
        D3D12Hooks::HookToCommandListLate(InCmdList);

        if (!D3D12Hooks::CanRestoreRootSignature(InCmdList))
        {
            LOG_DEBUG("Skipping upscaling because can't restore root signature");
            return NVSDK_NGX_Result_Success;
        }
    }

    if (InCallback)
        LOG_INFO("Progress callback provided but unused in synchronous OptiScaler path");

    // Resolution change detection (only for upscalers that may require recreation)
    if (feature != nullptr)
    {
        const bool isFFX =
            feature->GetUpscalerType() == Upscaler::FFX || feature->GetUpscalerType() == Upscaler::FFX_on12;
        const bool isFSR31OrLater = isFFX && feature->Version() >= feature_version { 3, 1, 0 };

        // FSR 3.1 supports upscaleSize that doesn't need reinit to change output resolution
        if (!isFSR31OrLater && feature->UpdateOutputResolution(InParameters))
            state.changeBackend[handleId] = true;
    }

    // To avoid capturing potential upscaler change (creation) and then upscaling itself
    D3D12Hooks::SetRootSignatureTracking(false);

    // Backend change or recreation requested
    if (state.changeBackend[handleId])
    {
        UpscalerInputsDx12::Reset();

        auto successfulPhase = FeatureProvider_Dx12::ChangeFeature(state.newBackend, D3D12Device, InCmdList, handleId,
                                                                   InParameters, &ctxData);
        feature = ctxData.feature.get();

        evalCounter = 0;

        if (ctxData.changeBackendCounter != 0 || !successfulPhase)
        {
            D3D12Hooks::SetRootSignatureTracking(true);
            return NVSDK_NGX_Result_Success;
        }
    }

    // Fallback to FSR 2.1.2 if feature failed to initialize and user didn't explicitly request it
    if (!feature->IsInited() && cfg.Dx12Upscaler.value_or_default() != Upscaler::FSR21)
    {
        LOG_WARN("Feature '{}' failed to initialize. Falling back to FSR 2.1.2", feature->Name());
        ImGui::InsertNotification({ ImGuiToastType::Warning, 10000, "Falling back to FSR 2.1.2" });

        state.newBackend = Upscaler::FSR21;
        state.changeBackend[handleId] = true;

        D3D12Hooks::SetRootSignatureTracking(true);

        return NVSDK_NGX_Result_Success;
    }

    state.currentFeature = feature;

    // Prepare upscaling inputs
    UpscalerInputsDx12::UpscaleStart(InCmdList, InParameters, feature);
    FSR3FG::SetUpscalerInputs(InCmdList, InParameters, feature);

    // Evaluate the feature
    bool evalSuccess = false;
    {
        // Resource tracking
        UpscalerInputsDx12::UpscaleEnd(InCmdList, InParameters, feature);

        ScopedSkipHeapCapture skip {};
        evalSuccess = feature->Evaluate(InCmdList, InParameters);
    }

    if (!evalSuccess)
    {
        LOG_ERROR("Feature evaluation failed for '{}'", feature->Name());
        ImGui::InsertNotification({ ImGuiToastType::Error, 10000, "Upscaler failed to run!" });
    }

    // Restore root signatures
    if (shouldRestoreSigs)
        D3D12Hooks::RestoreRoot(InCmdList);

    D3D12Hooks::SetRootSignatureTracking(true);

    return evalSuccess ? NVSDK_NGX_Result_Success : NVSDK_NGX_Result_Fail;
}

/**
 * @brief Per-frame feature execution. Runs a feature (upscaler, framegen, etc.) on a given command list using a
 * preexisting feature instance referenced by a unique handle.
 */
NVSDK_NGX_API NVSDK_NGX_Result NVSDK_NGX_D3D12_EvaluateFeature(ID3D12GraphicsCommandList* InCmdList,
                                                               const NVSDK_NGX_Handle* InFeatureHandle,
                                                               NVSDK_NGX_Parameter* InParameters,
                                                               PFN_NVSDK_NGX_ProgressCallback InCallback)
{
    if (!InFeatureHandle)
    {
        LOG_DEBUG("InFeatureHandle is null");
        return NVSDK_NGX_Result_FAIL_FeatureNotFound;
    }

    if (!InCmdList)
    {
        LOG_ERROR("InCmdList is null");
        return NVSDK_NGX_Result_Fail;
    }

    const uint32_t handleId = InFeatureHandle->Id;
    LOG_DEBUG("EvaluateFeature - Handle: {}, CmdList: {:p}", handleId, (void*) InCmdList);

    // Frame split: everything this evaluate records (the upscaler and our per-eye passes) is timed as one segment.
    FrameSplit_Dx12::BeginOurs(InCmdList);
    struct FrameSplitEnd
    {
        ID3D12GraphicsCommandList* list;
        ~FrameSplitEnd() { FrameSplit_Dx12::EndOurs(list); }
    } frameSplitEnd { InCmdList };

    const State& state = State::Instance();
    const Config& cfg = *Config::Instance();

    const auto tracked = HandleToFeature.Read(handleId);
    const auto feature = tracked.feature;
    bool nrUpscale = feature == NVSDK_NGX_Feature_SuperSampling || feature == NVSDK_NGX_Feature_RayReconstruction;
    bool rayReconstruction = feature == NVSDK_NGX_Feature_RayReconstruction;
    // A live OptiScaler backend is additional evidence, especially if the
    // creation record is absent or an SR handle has switched to DLSSD.
    if (auto it = Dx12Contexts.find(handleId); it != Dx12Contexts.end() && it->second.feature)
    {
        nrUpscale = true;
        rayReconstruction |= it->second.feature->GetUpscalerType() == Upscaler::DLSSD;
    }
    if (!tracked.feature && !nrUpscale)
        LOG_DEBUG("DLSS-NR: skipping untracked NGX handle {}; original evaluate is preserved", handleId);
    LOG_DEBUG("DLSS-NR route: handle {}, NGX feature {}, upscaler {}, RR {}", handleId,
              tracked.feature ? (int) *tracked.feature : -1, nrUpscale, rayReconstruction);
    UpscalerInputProbe_Dx12::OnEvaluate(InCmdList, handleId, tracked.feature ? (int) *tracked.feature : -1,
                                        InParameters);
    GiCapture_Dx12::OnEvaluate(InCmdList, tracked.feature ? (int) *tracked.feature : -1, InParameters);
    // Runs last, as frameSplitEnd does: the upscaler's output is on the list by then.
    struct EvalCaptureEnd
    {
        ID3D12GraphicsCommandList* list;
        bool took;
        ~EvalCaptureEnd()
        {
            if (took)
                EvalCapture_Dx12::AfterEvaluate(list);
        }
    } evalCaptureEnd { InCmdList, EvalCapture_Dx12::OnEvaluate(InCmdList, tracked.feature ? (int) *tracked.feature : -1,
                                                               InParameters) };

    static size_t evalWithoutFG = 0;
    const bool fgCreated = tracked.frameGenerationCreated;

    static std::optional<float> lastDlssgCameraNear {};
    static std::optional<float> lastDlssgCameraFar {};

    if (feature == NVSDK_NGX_Feature_FrameGeneration)
    {
        evalWithoutFG = 0;

        int frameCount = 0;
        InParameters->Get("DLSSG.MultiFrameCount", &frameCount);
        State::Instance().dlssgDetectedInterpolationCount = frameCount;
        ReflexHooks::setDlssgFrameCount(frameCount);

        float dlssgCameraNear = 0.0f;
        float dlssgCameraFar = 0.0f;

        if (InParameters->Get("DLSSG.CameraNear", &dlssgCameraNear) == NVSDK_NGX_Result_Success)
            lastDlssgCameraNear = dlssgCameraNear;

        if (InParameters->Get("DLSSG.CameraFar", &dlssgCameraFar) == NVSDK_NGX_Result_Success)
            lastDlssgCameraFar = dlssgCameraFar;
    }
    else if (fgCreated)
    {
        evalWithoutFG++;

        if (evalWithoutFG == 6)
        {
            // Report FG as disabled
            State::Instance().dlssgDetectedInterpolationCount = 0;
            ReflexHooks::setDlssgFrameCount(0);
        }
    }

    // Native DLSS passthrough
    if (handleId < DLSS_MOD_ID_OFFSET)
    {
        if (cfg.DLSSEnabled.value_or_default() && NVNGXProxy::D3D12_EvaluateFeature() != nullptr)
        {
            LOG_DEBUG("Passthrough to native DLSS EvaluateFeature for handle {}", handleId);

            if (nrUpscale)
                DlssNr::EvaluateBeforeUpscale(InCmdList, InParameters, nullptr, 0, rayReconstruction);

            NVSDK_NGX_Result result =
                NVNGXProxy::D3D12_EvaluateFeature()(InCmdList, InFeatureHandle, InParameters, InCallback);
            LOG_DEBUG("Native DLSS EvaluateFeature result: 0x{:X}", (uint32_t) result);

            // Neural Rendering runs over what the upscaler just wrote, on the same list, so frame
            // generation interpolates from enhanced frames and the model still costs one run per
            // rendered frame. The feature check is the point: frame generation is handed depth and
            // motion vectors too, and its handle can reach here because the branch above does not
            // return, so filtering on the parameter block alone would run the model twice a frame.
            if (result == NVSDK_NGX_Result_Success && nrUpscale)
                DlssNr::EvaluateAfterUpscale(InCmdList, InParameters, nullptr, rayReconstruction);

            return result;
        }

        LOG_DEBUG("Native DLSS EvaluateFeature not available for handle {}", handleId);
        return NVSDK_NGX_Result_FAIL_FeatureNotFound;
    }

    // DLSSG replacements passthrough
    if (State::Instance().activeFgNvngx != FGNvngxReplacement::None && handleId >= NVNGX_PROVIDER_ID_OFFSET)
    {
        LOG_DEBUG("Passthrough to DLSSG Replacement's EvaluateFeature for handle {}", handleId);
        return Nvngx_FG::D3D12_EvaluateFeature(InCmdList, InFeatureHandle, InParameters, InCallback);
    }

    if (lastDlssgCameraNear.has_value())
        InParameters->Set("DLSSG.CameraNear", lastDlssgCameraNear.value());

    if (lastDlssgCameraFar.has_value())
        InParameters->Set("DLSSG.CameraFar", lastDlssgCameraFar.value());

    // Per-eye DLSS: the other eye's Presents go to the twin feature, and BeforeEvaluate puts our same-eye vectors into
    // the parameters. Neural Rendering runs AFTER that: its model history is per eye, so it needs
    // same-eye vectors like DLSS does. Before, it ran first and read the game's other-eye vectors against a same-eye
    // history: a D3D12 game flickered under our per eye and not under the VR mod's own DLSS pass.
    // The restore of the game's vectors is after NR's second run below.
    NVSDK_NGX_Handle routedHandle { handleId };
    if (nrUpscale && PerEyeDlss_Dx12::Enabled())
    {
        // The twin at first need: the routing went per eye (F6) and this feature has none yet. Tried once per feature.
        if (PerEyeDlss_Dx12::RoutingOn() && PerEyeDlss_Dx12::TwinOf(handleId) == 0 && !TwinCreateFailed.contains(handleId))
        {
            LOG_INFO("Per-eye DLSS DX12: routing per eye and feature {} has no twin yet: making it now (first need)", handleId);
            CreatePerEyeTwin(InCmdList, handleId, InParameters);
            if (PerEyeDlss_Dx12::TwinOf(handleId) == 0)
            {
                TwinCreateFailed.insert(handleId);
                LOG_WARN("Per-eye DLSS DX12: no twin for feature {}: both eyes share it for its lifetime (not tried again)",
                         handleId);
            }
        }
        routedHandle.Id = PerEyeDlss_Dx12::Route(handleId, InParameters);
        PerEyeDlss_Dx12::BeforeEvaluate(InCmdList, InParameters);
    }

    if (nrUpscale)
        DlssNr::EvaluateBeforeUpscale(InCmdList, InParameters, nullptr, 0, rayReconstruction);

    // A rebuild the per-eye overlay asked for (the outer preset) comes back as DLSS, whatever OptiScaler's default.
    const bool perEyeRebuild = PerEyeDlss_Dx12::Enabled() && State::Instance().changeBackend[routedHandle.Id] &&
                               PerEyeDlss_Dx12::TakeDlssRebuild(routedHandle.Id, true);
    if (perEyeRebuild)
        State::Instance().newBackend = RebuildBackend(routedHandle.Id);

    // Foveated DLSS: the crop is planned first (SR, or RR with its guides and projection); with the cheap periphery on,
    // a render-size pass of the same kind enlarged over the eye replaces the game's full pass.
    PerEyeDlss_Dx12::FoveaSize cropSize {};
    const bool fovea = nrUpscale && PerEyeDlss_Dx12::Enabled() && PerEyeDlss_Dx12::FoveaOn() &&
                       PerEyeDlss_Dx12::PlanFovea(InCmdList, InParameters, routedHandle.Id, &cropSize);
    const bool periphery =
        fovea && PerEyeDlss_Dx12::PeripheryOn() && EvaluatePeriphery(InCmdList, InParameters, routedHandle.Id);
    // A game feature skipped for the periphery has a stale history: start it over.
    if (!periphery && PerEyeDlss_Dx12::Enabled() && PerEyeDlss_Dx12::TakeMainSkipped(routedHandle.Id))
        InParameters->Set(NVSDK_NGX_Parameter_Reset, 1);

    // OptiScaler internal handling
    // Without the cheap periphery the game's own pass is the full-size one (see MainVectorsBefore): checked on EVERY
    // evaluate, not only foveated ones, so the pass below us is known to replace the vectors before the crop comes on and
    // the periphery is made once, as one feature, instead of a pair per eye first.
    if (!periphery)
        PerEyeDlss_Dx12::MainVectorsBefore(InParameters);
    const NVSDK_NGX_Result optiResult =
        periphery ? NVSDK_NGX_Result_Success
                  : TryEvaluateOptiFeature(InCmdList, &routedHandle, InParameters, InCallback);
    if (!periphery)
        PerEyeDlss_Dx12::MainVectorsAfter(InParameters, optiResult == NVSDK_NGX_Result_Success);
    else if (auto* stoodInFor = Dx12Contexts[routedHandle.Id].feature.get())
        stoodInFor->NoteStandIn(); // the overlay must not take the game's feature for unused
    if (perEyeRebuild)
        PerEyeDlss_Dx12::TakeDlssRebuild(routedHandle.Id, State::Instance().changeBackend[routedHandle.Id]);
    if (nrUpscale && PerEyeDlss_Dx12::Enabled())
    {
        PerEyeDlss_Dx12::AfterUpscale(InCmdList);
        if (fovea && optiResult == NVSDK_NGX_Result_Success)
            EvaluateFoveaCrop(InCmdList, InParameters, routedHandle.Id, cropSize, rayReconstruction);
    }
    PerEyeDlss_Dx12::EndEvaluate(InCmdList, InParameters);

    // Same pass, for OptiScaler's own upscalers rather than native DLSS. With our same-eye vectors still in the
    // parameters (see above); the game's are put back after.
    if (optiResult == NVSDK_NGX_Result_Success && nrUpscale)
        DlssNr::EvaluateAfterUpscale(InCmdList, InParameters, nullptr, rayReconstruction);
    if (nrUpscale && PerEyeDlss_Dx12::Enabled())
        PerEyeDlss_Dx12::RestoreInputs(InParameters);

    return optiResult;
}

#pragma endregion

#pragma region DLSS Buffer Size Call

NVSDK_NGX_API NVSDK_NGX_Result NVSDK_NGX_D3D12_GetScratchBufferSize(NVSDK_NGX_Feature InFeatureId,
                                                                    const NVSDK_NGX_Parameter* InParameters,
                                                                    size_t* OutSizeInBytes)
{
    if (OutSizeInBytes == nullptr)
        return NVSDK_NGX_Result_FAIL_InvalidParameter;

    if (State::Instance().activeFgNvngx != FGNvngxReplacement::None && InFeatureId == NVSDK_NGX_Feature_FrameGeneration)
    {
        return Nvngx_FG::D3D12_GetScratchBufferSize(InFeatureId, InParameters, OutSizeInBytes);
    }

    LOG_WARN("-> 52428800");
    *OutSizeInBytes = 52428800;
    return NVSDK_NGX_Result_Success;
}

#pragma endregion
