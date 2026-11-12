#include "pch.h"
#include <dlssnr/DlssNrFeature_Vk.h>
#include "Util.h"
#include "Config.h"
#include "resource.h"

#include "NVNGX_DLSS.h"
#include <framegen/nvngx/Nvngx_FG.h>
#include "NVNGX_Parameter.h"
#include "proxies/NVNGX_Proxy.h"

#include "upscalers/FeatureProvider_Vk.h"
#include "inputs/PerEyeDlss_Vk.h"
#include "inputs/FoveaCrop_Vk.h"
#include "inputs/FrameTime_Vk.h"
#include "inputs/NvngxVkOwn.h"
#include "inputs/VkRetire.h"

#include <upscaler_time/UpscalerTime_Vk.h>

#include <vulkan/vulkan.hpp>
#include <ankerl/unordered_dense.h>
#include <imgui/ImGuiNotify.hpp>
#include <misc/IdentifyGpu.h>

VkInstance vkInstance;
VkPhysicalDevice vkPD;
VkDevice vkDevice;
PFN_vkGetInstanceProcAddr vkGIPA;
PFN_vkGetDeviceProcAddr vkGDPA;

static ankerl::unordered_dense::map<unsigned int, ContextData<IFeature_Vk>> VkContexts;
// Per-eye DLSS: a game feature whose first-need twin could not be made (tried once per feature; forgotten at release).
static ankerl::unordered_dense::set<unsigned int> TwinCreateFailedVk;

// VkRetire (see its header): features released with [Hotfix] VulkanNoStalls, destroyed kAfterPresents Presents later.
namespace
{
constexpr uint64_t kAfterPresents = 8;
std::mutex g_retireMutex;
std::vector<std::pair<std::unique_ptr<IFeature_Vk>, uint64_t>> g_retired;
std::atomic<uint64_t> g_retirePresents { 0 };

bool NoStalls() { return Config::Instance()->VulkanNoStalls.value_or_default(); }

// The feature leaves its context now. Stock (VulkanNoStalls off): destroyed behind vkDeviceWaitIdle + 0.5 s. NoStalls:
// destroyed at once in the game's call with no wait (as the game's own release would be without OptiScaler), or with
// VulkanRetireLater 8 Presents later. Logged every time: a game releases a handful of features a session.
void RetireFeature(std::unique_ptr<IFeature_Vk>& feature, unsigned int id)
{
    if (!feature)
        return;
    if (feature.get() == State::Instance().currentFeature)
        State::Instance().currentFeature = nullptr;
    if (!NoStalls())
    {
        vkDeviceWaitIdle(vkDevice);
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        feature.reset();
        return;
    }
    if (!Config::Instance()->VulkanRetireLater.value_or_default())
    {
        const auto t0 = std::chrono::steady_clock::now();
        feature.reset();
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        LOG_INFO("Vulkan: feature {} destroyed at once in the game's release, no wait ({:.2f} ms; [Hotfix] "
                 "VulkanNoStalls, VulkanRetireLater off)",
                 id, ms);
        return;
    }
    std::lock_guard lock(g_retireMutex);
    g_retired.push_back({ std::move(feature), g_retirePresents.load() });
    LOG_INFO("Vulkan: feature {} released without a stall; destroyed {} Presents later ([Hotfix] VulkanRetireLater)", id,
             kAfterPresents);
}
} // namespace

void VkRetire::OnPresent()
{
    const uint64_t now = ++g_retirePresents;
    std::vector<std::unique_ptr<IFeature_Vk>> doomed;
    {
        std::lock_guard lock(g_retireMutex);
        for (auto it = g_retired.begin(); it != g_retired.end();)
        {
            if (now - it->second >= kAfterPresents)
            {
                doomed.push_back(std::move(it->first));
                it = g_retired.erase(it);
            }
            else
                ++it;
        }
    }
    if (!doomed.empty())
        LOG_INFO("Vulkan: {} retired feature(s) destroyed at Present {}", doomed.size(), now);
    doomed.clear(); // destroyed outside the lock
}

// Per-eye DLSS (PerEyeDlss_Vk): a second feature made from the same creation parameters and backend, for the other
// eye's frames. Made at the FIRST PER-EYE EVALUATE (the routing switched per eye, F6), as on D3D12, or at the game's
// create with [Inputs] UpscalerTwinAtCreate. Not at the create by default: in shared routing the twin takes no evaluate,
// and every DLSS feature costs memory and descriptor pools. OptiScaler's current feature is put back to what it WAS (a
// stock path may have nulled it on purpose; Crimson Desert's Present crash).
static void CreatePerEyeTwinVk(VkDevice InDevice, VkCommandBuffer InCmdList, unsigned int primary,
                               NVSDK_NGX_Parameter* InParameters)
{
    auto primaryIt = VkContexts.find(primary);
    if (primaryIt == VkContexts.end() || primaryIt->second.feature == nullptr || !primaryIt->second.feature->IsInited())
    {
        LOG_WARN("Per-eye DLSS VK: feature {} is not initialised; no twin, both eyes share it", primary);
        return;
    }
    const Upscaler backend = primaryIt->second.feature->GetUpscalerType();
    IFeature* const currentBefore = State::Instance().currentFeature;
    const uint32_t twinId = IFeature::GetNextHandleId();
    VkContexts[twinId] = {};
    if (!FeatureProvider_Vk::GetFeature(backend, twinId, InParameters, &VkContexts[twinId].feature) ||
        VkContexts[twinId].feature == nullptr)
    {
        LOG_ERROR("Per-eye DLSS VK: can't create the twin of feature {} ({}); both eyes share it", primary,
                  UpscalerDisplayName(backend));
        VkContexts.erase(twinId);
        State::Instance().currentFeature = currentBefore;
        return;
    }
    bool inited = false;
    {
        ScopedSkipSpoofingGlobal skipSpoofingGlobal {};
        inited = VkContexts[twinId].feature->Init(vkInstance, vkPD, InDevice, InCmdList, vkGIPA, vkGDPA, InParameters);
    }
    State::Instance().currentFeature = currentBefore;
    if (!inited)
    {
        LOG_ERROR("Per-eye DLSS VK: the twin of feature {} failed to initialise; both eyes share it", primary);
        VkContexts.erase(twinId);
        return;
    }
    LOG_INFO("Per-eye DLSS VK: created twin {} of feature {} ({})", twinId, primary, UpscalerDisplayName(backend));
    PerEyeDlss_Vk::RegisterTwin(primary, twinId);
}

// Features of our own (the foveal crop's, FoveaCrop_Vk): made as the twin is made, retired as the game's are.
unsigned int NvngxVkOwn::Create(VkCommandBuffer cmd, Upscaler backend, NVSDK_NGX_Parameter* params)
{
    if (params == nullptr || vkDevice == VK_NULL_HANDLE)
        return 0;
    IFeature* const currentBefore = State::Instance().currentFeature;
    const uint32_t id = IFeature::GetNextHandleId();
    VkContexts[id] = {};
    if (!FeatureProvider_Vk::GetFeature(backend, id, params, &VkContexts[id].feature) ||
        VkContexts[id].feature == nullptr)
    {
        LOG_ERROR("Per-eye DLSS VK: can't create a feature of our own ({})", UpscalerDisplayName(backend));
        VkContexts.erase(id);
        State::Instance().currentFeature = currentBefore;
        return 0;
    }
    bool inited = false;
    {
        ScopedSkipSpoofingGlobal skipSpoofingGlobal {};
        inited = VkContexts[id].feature->Init(vkInstance, vkPD, vkDevice, cmd, vkGIPA, vkGDPA, params);
    }
    State::Instance().currentFeature = currentBefore;
    if (!inited)
    {
        LOG_ERROR("Per-eye DLSS VK: a feature of our own failed to initialise ({})", UpscalerDisplayName(backend));
        VkContexts.erase(id);
        return 0;
    }
    return id;
}

bool NvngxVkOwn::MakeDetached(VkCommandBuffer cmd, Upscaler backend, NVSDK_NGX_Parameter* params, unsigned int id,
                              std::unique_ptr<IFeature_Vk>& out)
{
    out.reset();
    if (params == nullptr || vkDevice == VK_NULL_HANDLE || cmd == VK_NULL_HANDLE)
        return false;
    if (!FeatureProvider_Vk::GetFeature(backend, id, params, &out) || out == nullptr)
    {
        LOG_ERROR("Per-eye DLSS VK: can't create a feature of our own on the worker ({})", UpscalerDisplayName(backend));
        out.reset();
        return false;
    }
    bool inited = false;
    {
        // This thread's spoofing skipped, as the inline create skips it globally (the worker is not the game's thread).
        ScopedSkipSpoofingThread skipSpoofingThread {};
        inited = out->Init(vkInstance, vkPD, vkDevice, cmd, vkGIPA, vkGDPA, params);
    }
    if (!inited)
    {
        LOG_ERROR("Per-eye DLSS VK: a feature of our own failed to initialise on the worker ({})",
                  UpscalerDisplayName(backend));
        out.reset();
        return false;
    }
    return true;
}

void NvngxVkOwn::Adopt(unsigned int id, std::unique_ptr<IFeature_Vk> feature)
{
    if (feature == nullptr)
        return;
    VkContexts[id] = {};
    VkContexts[id].feature = std::move(feature);
}

IFeature_Vk* NvngxVkOwn::Get(unsigned int id)
{
    auto it = VkContexts.find(id);
    return it == VkContexts.end() ? nullptr : it->second.feature.get();
}

void NvngxVkOwn::Retire(unsigned int id)
{
    auto it = VkContexts.find(id);
    if (it == VkContexts.end())
        return;
    RetireFeature(it->second.feature, id);
    VkContexts.erase(it);
}
static int evalCounter = 0;
static bool shutdown = false;
static bool _skipInit = false;
static wchar_t const** paths;

class ScopedInitVk
{
  private:
    bool previousState;

  public:
    ScopedInitVk()
    {
        previousState = _skipInit;
        _skipInit = true;
    }

    ~ScopedInitVk() { _skipInit = previousState; }
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

        // WAR: Doom Ethernal is sending junk data
        if (InFeatureInfo->PathListInfo.Length < 10)
        {
            // Original paths from NVNGX
            for (size_t i = 0; i < InFeatureInfo->PathListInfo.Length; i++)
            {
                const wchar_t* path = InFeatureInfo->PathListInfo.Path[i];
                State::Instance().NVNGX_FeatureInfo_Paths.push_back(std::wstring(path));
            }
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

NVSDK_NGX_API NVSDK_NGX_Result NVSDK_NGX_VULKAN_Init_Ext2(
    unsigned long long InApplicationId, const wchar_t* InApplicationDataPath, VkInstance InInstance,
    VkPhysicalDevice InPD, VkDevice InDevice, PFN_vkGetInstanceProcAddr InGIPA, PFN_vkGetDeviceProcAddr InGDPA,
    NVSDK_NGX_Version InSDKVersion, const NVSDK_NGX_FeatureCommonInfo* InFeatureInfo)
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

        if (NVNGXProxy::NVNGXModule() != nullptr && NVNGXProxy::VULKAN_Init_Ext2() != nullptr)
        {
            LOG_INFO("calling NVNGXProxy::VULKAN_Init_Ext2");
            auto result = NVNGXProxy::VULKAN_Init_Ext2()(InApplicationId, InApplicationDataPath, InInstance, InPD,
                                                         InDevice, InGIPA, InGDPA, InSDKVersion, &localFeatureInfo);
            LOG_INFO("NVNGXProxy::VULKAN_Init_Ext2 result: {0:X}", (UINT) result);

            if (result == NVSDK_NGX_Result_Success)
                NVNGXProxy::SetVulkanInited(true);
        }
    }

    if (InFeatureInfo != nullptr && InSDKVersion > 0x0000013)
        State::Instance().NVNGX_Logger = InFeatureInfo->LoggingInfo;

    if (State::Instance().nvngxVkInited && InInstance == vkInstance && InDevice == vkDevice && InPD == vkPD)
    {
        LOG_WARN("NVNGX already inited");
        return NVSDK_NGX_Result_Success;
    }

    if (State::Instance().activeFgInput == FGInput::NvngxFG)
    {
        Nvngx_FG::VULKAN_Init_Ext2(InApplicationId, InApplicationDataPath, InInstance, InPD, InDevice, InGIPA, InGDPA,
                                   InSDKVersion, &localFeatureInfo);
    }

    LOG_INFO("AppId: {0}", InApplicationId);
    LOG_INFO("SDK: {0:x}", (unsigned int) InSDKVersion);
    LOG_INFO(L"InApplicationDataPath {0}", std::wstring(InApplicationDataPath));

    if (InInstance)
    {
        LOG_INFO("InInstance exist!");
        vkInstance = InInstance;
    }

    if (InPD)
    {
        LOG_INFO("InPD exist!");
        vkPD = InPD;
    }

    if (InDevice)
    {
        LOG_INFO("InDevice exist!");
        vkDevice = InDevice;
    }

    if (InGDPA)
    {
        LOG_INFO("InGDPA exist!");
        vkGDPA = InGDPA;
    }
    else
    {
        LOG_INFO("InGDPA does not exist!");
        vkGDPA = vkGetDeviceProcAddr;
    }

    if (InGIPA)
    {
        LOG_INFO("InGIPA exist!");
        vkGIPA = InGIPA;
    }
    else
    {
        LOG_INFO("InGIPA does not exist!");
        vkGIPA = vkGetInstanceProcAddr;
    }

    State::Instance().currentVkDevice = InDevice;

    UpscalerTimeVk::Init(InDevice, InPD);

    State::Instance().nvngxVkInited = true;

    return NVSDK_NGX_Result_Success;
}

NVSDK_NGX_API NVSDK_NGX_Result NVSDK_NGX_VULKAN_Init_Ext(unsigned long long InApplicationId,
                                                         const wchar_t* InApplicationDataPath, VkInstance InInstance,
                                                         VkPhysicalDevice InPD, VkDevice InDevice,
                                                         NVSDK_NGX_Version InSDKVersion,
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
            InApplicationId = app_id_override;

        if (NVNGXProxy::NVNGXModule() == nullptr)
            NVNGXProxy::InitNVNGX();

        if (NVNGXProxy::NVNGXModule() != nullptr && NVNGXProxy::VULKAN_Init_Ext() != nullptr)
        {
            LOG_INFO("calling NVNGXProxy::VULKAN_Init_Ext");
            auto result = NVNGXProxy::VULKAN_Init_Ext()(InApplicationId, InApplicationDataPath, InInstance, InPD,
                                                        InDevice, InSDKVersion, &localFeatureInfo);
            LOG_INFO("NVNGXProxy::VULKAN_Init_Ext result: {0:X}", (UINT) result);

            if (result == NVSDK_NGX_Result_Success)
                NVNGXProxy::SetVulkanInited(true);
        }
    }

    Nvngx_FG::VULKAN_Init_Ext(InApplicationId, InApplicationDataPath, InInstance, InPD, InDevice, InSDKVersion,
                              &localFeatureInfo);

    ScopedInitVk scopedInit {};
    return NVSDK_NGX_VULKAN_Init_Ext2(InApplicationId, InApplicationDataPath, InInstance, InPD, InDevice,
                                      vkGetInstanceProcAddr, vkGetDeviceProcAddr, InSDKVersion, InFeatureInfo);
}

NVSDK_NGX_API NVSDK_NGX_Result NVSDK_NGX_VULKAN_Init_ProjectID_Ext(
    const char* InProjectId, NVSDK_NGX_EngineType InEngineType, const char* InEngineVersion,
    const wchar_t* InApplicationDataPath, VkInstance InInstance, VkPhysicalDevice InPD, VkDevice InDevice,
    PFN_vkGetInstanceProcAddr InGIPA, PFN_vkGetDeviceProcAddr InGDPA, NVSDK_NGX_Version InSDKVersion,
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
        if (NVNGXProxy::NVNGXModule() == nullptr)
            NVNGXProxy::InitNVNGX();

        if (NVNGXProxy::NVNGXModule() != nullptr && NVNGXProxy::VULKAN_Init_ProjectID_Ext() != nullptr)
        {
            LOG_INFO("calling NVNGXProxy::VULKAN_Init_ProjectID_Ext");
            auto result = NVNGXProxy::VULKAN_Init_ProjectID_Ext()(InProjectId, InEngineType, InEngineVersion,
                                                                  InApplicationDataPath, InInstance, InPD, InDevice,
                                                                  InGIPA, InGDPA, InSDKVersion, &localFeatureInfo);
            LOG_INFO("NVNGXProxy::VULKAN_Init_ProjectID_Ext result: {0:X}", (UINT) result);

            if (result == NVSDK_NGX_Result_Success)
                NVNGXProxy::SetVulkanInited(true);
        }
    }

    ScopedInitVk scopedInit {};
    auto result = NVSDK_NGX_VULKAN_Init_Ext2(0x1337, InApplicationDataPath, InInstance, InPD, InDevice, InGIPA, InGDPA,
                                             InSDKVersion, &localFeatureInfo);

    LOG_DEBUG("InProjectId: {0}", InProjectId);
    LOG_DEBUG("InEngineType: {0}", (int) InEngineType);
    LOG_DEBUG("InEngineVersion: {0}", InEngineVersion);

    State::Instance().NVNGX_ProjectId = std::string(InProjectId);
    State::Instance().NVNGX_Engine = InEngineType;
    State::Instance().NVNGX_EngineVersion = std::string(InEngineVersion);

    return result;
}

NVSDK_NGX_API NVSDK_NGX_Result NVSDK_NGX_VULKAN_Init(unsigned long long InApplicationId,
                                                     const wchar_t* InApplicationDataPath, VkInstance InInstance,
                                                     VkPhysicalDevice InPD, VkDevice InDevice,
                                                     PFN_vkGetInstanceProcAddr InGIPA, PFN_vkGetDeviceProcAddr InGDPA,
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

        if (NVNGXProxy::NVNGXModule() != nullptr && NVNGXProxy::VULKAN_Init() != nullptr)
        {
            LOG_INFO("calling NVNGXProxy::VULKAN_Init");
            auto result = NVNGXProxy::VULKAN_Init()(InApplicationId, InApplicationDataPath, InInstance, InPD, InDevice,
                                                    InGIPA, InGDPA, &localFeatureInfo, InSDKVersion);
            LOG_INFO("NVNGXProxy::VULKAN_Init result: {0:X}", (UINT) result);

            if (result == NVSDK_NGX_Result_Success)
                NVNGXProxy::SetVulkanInited(true);
        }
    }

    Nvngx_FG::VULKAN_Init(InApplicationId, InApplicationDataPath, InInstance, InPD, InDevice, InGIPA, InGDPA,
                          &localFeatureInfo, InSDKVersion);

    ScopedInitVk scopedInit {};
    return NVSDK_NGX_VULKAN_Init_Ext2(InApplicationId, InApplicationDataPath, InInstance, InPD, InDevice, InGIPA,
                                      InGDPA, InSDKVersion, InFeatureInfo);
}

NVSDK_NGX_API NVSDK_NGX_Result
NVSDK_NGX_VULKAN_Init_ProjectID(const char* InProjectId, NVSDK_NGX_EngineType InEngineType, const char* InEngineVersion,
                                const wchar_t* InApplicationDataPath, VkInstance InInstance, VkPhysicalDevice InPD,
                                VkDevice InDevice, PFN_vkGetInstanceProcAddr InGIPA, PFN_vkGetDeviceProcAddr InGDPA,
                                NVSDK_NGX_Version InSDKVersion, const NVSDK_NGX_FeatureCommonInfo* InFeatureInfo)
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

        if (NVNGXProxy::NVNGXModule() != nullptr && NVNGXProxy::VULKAN_Init_ProjectID() != nullptr)
        {
            LOG_INFO("calling NVNGXProxy::VULKAN_Init_ProjectID");
            auto result = NVNGXProxy::VULKAN_Init_ProjectID()(InProjectId, InEngineType, InEngineVersion,
                                                              InApplicationDataPath, InInstance, InPD, InDevice, InGIPA,
                                                              InGDPA, InSDKVersion, &localFeatureInfo);
            LOG_INFO("NVNGXProxy::VULKAN_Init_ProjectID result: {0:X}", (UINT) result);

            if (result == NVSDK_NGX_Result_Success)
                NVNGXProxy::SetVulkanInited(true);
        }
    }

    ScopedInitVk scopedInit {};
    return NVSDK_NGX_VULKAN_Init_ProjectID_Ext(InProjectId, InEngineType, InEngineVersion, InApplicationDataPath,
                                               InInstance, InPD, InDevice, InGIPA, InGDPA, InSDKVersion,
                                               &localFeatureInfo);
}

/**
 * @brief [Deprecated NGX API] Superceeded by NVSDK_NGX_AllocateParameters and NVSDK_NGX_GetCapabilityParameters.
 *
 * Retrieves a common NVSDK parameter map for providing params to the SDK. The lifetime of this
 * map is NOT managed by the application. It is expected to be managed internally by the SDK.
 */
NVSDK_NGX_API NVSDK_NGX_Result NVSDK_NGX_VULKAN_GetParameters(NVSDK_NGX_Parameter** OutParameters)
{
    LOG_FUNC();

    if (OutParameters == nullptr)
        return NVSDK_NGX_Result_FAIL_InvalidParameter;

    if (Config::Instance()->DLSSEnabled.value_or_default() && NVNGXProxy::NVNGXModule() != nullptr &&
        NVNGXProxy::VULKAN_GetParameters() != nullptr)
    {
        LOG_INFO("calling NVNGXProxy::VULKAN_GetParameters");
        auto result = NVNGXProxy::VULKAN_GetParameters()(OutParameters);
        LOG_INFO("NVNGXProxy::VULKAN_GetParameters result: {0:X}", (UINT) result);

        if (result == NVSDK_NGX_Result_Success)
        {
            InitNGXParameters(*OutParameters, API::Vulkan);
            SetNGXParamAllocType(*(*OutParameters), NGX_AllocTypes::NVPersistent);
            return result;
        }
    }

    // Get custom parameters if using custom backend
    static NVNGX_Parameters oldParams = NVNGX_Parameters(API::Vulkan, true);
    *OutParameters = &oldParams;
    InitNGXParameters(*OutParameters, API::Vulkan);

    LOG_DEBUG("Returning custom Opti parameters");

    return NVSDK_NGX_Result_Success;
}

NVSDK_NGX_API NVSDK_NGX_Result NVSDK_NGX_VULKAN_GetFeatureInstanceExtensionRequirements(
    const NVSDK_NGX_FeatureDiscoveryInfo* FeatureDiscoveryInfo, uint32_t* OutExtensionCount,
    VkExtensionProperties** OutExtensionProperties)
{
    LOG_DEBUG("FeatureID: {0}", (UINT) FeatureDiscoveryInfo->FeatureID);

    if (State::Instance().activeFgInput == FGInput::NvngxFG && Nvngx_FG::isVulkanAvailable() &&
        FeatureDiscoveryInfo->FeatureID == NVSDK_NGX_Feature_FrameGeneration)
    {
        return NVSDK_NGX_Result_Success;
    }

    if (Config::Instance()->DLSSEnabled.value_or_default() && NVNGXProxy::NVNGXModule() != nullptr &&
        NVNGXProxy::VULKAN_GetFeatureInstanceExtensionRequirements() != nullptr)
    {
        LOG_INFO("calling NVNGXProxy::VULKAN_GetFeatureInstanceExtensionRequirements");
        auto result = NVNGXProxy::VULKAN_GetFeatureInstanceExtensionRequirements()(
            FeatureDiscoveryInfo, OutExtensionCount, OutExtensionProperties);
        LOG_INFO("NVNGXProxy::VULKAN_GetFeatureInstanceExtensionRequirements result: {0:X}", (UINT) result);

        if (result == NVSDK_NGX_Result_Success)
        {
            if (*OutExtensionCount > 0)
                LOG_DEBUG("required extensions: {0}", *OutExtensionCount);

            return result;
        }
    }

    if (NVNGXProxy::NVNGXModule() != nullptr && NVNGXProxy::VULKAN_GetFeatureInstanceExtensionRequirements() != nullptr)
    {
        LOG_INFO("returning original needed extensions");
        return NVNGXProxy::VULKAN_GetFeatureInstanceExtensionRequirements()(FeatureDiscoveryInfo, OutExtensionCount,
                                                                            OutExtensionProperties);
    }
    else
    {
        LOG_DEBUG("OutExtensionCount != nullptr: {}", OutExtensionCount != nullptr);

        if (FeatureDiscoveryInfo->FeatureID == NVSDK_NGX_Feature_FrameGeneration)
            return NVSDK_NGX_Result_FAIL_InvalidParameter;

        if (FeatureDiscoveryInfo->FeatureID == NVSDK_NGX_Feature_SuperSampling && OutExtensionCount != nullptr)
        {
            if (OutExtensionProperties == nullptr)
            {
                if (Config::Instance()->VulkanExtensionSpoofing.value_or_default())
                {
                    LOG_INFO("returning 3 extensions are needed");
                    *OutExtensionCount = 3;
                }
                else
                {
                    LOG_INFO("returning no extensions are needed");
                    *OutExtensionCount = 0;
                }
            }
            else if (*OutExtensionCount == 3 && Config::Instance()->VulkanExtensionSpoofing.value_or_default())
            {
                LOG_INFO("returning extension infos");

                std::memset((*OutExtensionProperties)[0].extensionName, 0,
                            sizeof(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME));
                std::strcpy((*OutExtensionProperties)[0].extensionName,
                            VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
                (*OutExtensionProperties)[0].specVersion = VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_SPEC_VERSION;

                std::memset((*OutExtensionProperties)[1].extensionName, 0,
                            sizeof(VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME));
                std::strcpy((*OutExtensionProperties)[1].extensionName,
                            VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME);
                (*OutExtensionProperties)[1].specVersion = VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_SPEC_VERSION;

                std::memset((*OutExtensionProperties)[2].extensionName, 0,
                            sizeof(VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME));
                std::strcpy((*OutExtensionProperties)[2].extensionName,
                            VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME);
                (*OutExtensionProperties)[2].specVersion = VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_SPEC_VERSION;
            }

            return NVSDK_NGX_Result_Success;
        }
    }

    LOG_INFO("returning no extensions are needed");

    if (OutExtensionCount != nullptr)
        *OutExtensionCount = 0;

    return NVSDK_NGX_Result_Success;
}

NVSDK_NGX_API NVSDK_NGX_Result NVSDK_NGX_VULKAN_GetFeatureDeviceExtensionRequirements(
    VkInstance Instance, VkPhysicalDevice PhysicalDevice, const NVSDK_NGX_FeatureDiscoveryInfo* FeatureDiscoveryInfo,
    uint32_t* OutExtensionCount, VkExtensionProperties** OutExtensionProperties)
{
    LOG_DEBUG("FeatureID: {0}", (UINT) FeatureDiscoveryInfo->FeatureID);

    if (State::Instance().activeFgInput == FGInput::NvngxFG && Nvngx_FG::isVulkanAvailable() &&
        FeatureDiscoveryInfo->FeatureID == NVSDK_NGX_Feature_FrameGeneration)
    {
        return NVSDK_NGX_Result_Success;
    }

    if (Config::Instance()->DLSSEnabled.value_or_default() && NVNGXProxy::NVNGXModule() != nullptr &&
        NVNGXProxy::VULKAN_GetFeatureDeviceExtensionRequirements() != nullptr)
    {
        LOG_INFO("calling NVNGXProxy::VULKAN_GetFeatureDeviceExtensionRequirements");
        auto result = NVNGXProxy::VULKAN_GetFeatureDeviceExtensionRequirements()(
            Instance, PhysicalDevice, FeatureDiscoveryInfo, OutExtensionCount, OutExtensionProperties);
        LOG_INFO("NVNGXProxy::VULKAN_GetFeatureDeviceExtensionRequirements result: {0:X}", (UINT) result);

        if (result == NVSDK_NGX_Result_Success)
        {
            if (*OutExtensionCount > 0)
                LOG_DEBUG("required extensions: {0}", *OutExtensionCount);

            return result;
        }
    }

    if (NVNGXProxy::NVNGXModule() != nullptr && NVNGXProxy::VULKAN_GetFeatureDeviceExtensionRequirements() != nullptr)
    {
        LOG_INFO("returning original needed extensions");
        return NVNGXProxy::VULKAN_GetFeatureDeviceExtensionRequirements()(
            Instance, PhysicalDevice, FeatureDiscoveryInfo, OutExtensionCount, OutExtensionProperties);
    }
    else
    {
        if (FeatureDiscoveryInfo->FeatureID == NVSDK_NGX_Feature_FrameGeneration)
            return NVSDK_NGX_Result_FAIL_InvalidParameter;

        LOG_DEBUG("OutExtensionCount != nullptr: {}", OutExtensionCount != nullptr);

        if (FeatureDiscoveryInfo->FeatureID == NVSDK_NGX_Feature_SuperSampling && OutExtensionCount != nullptr)
        {
            if (OutExtensionProperties == nullptr)
            {
                if (Config::Instance()->VulkanExtensionSpoofing.value_or_default())
                {
                    LOG_INFO("returning 4 extensions are needed!");
                    *OutExtensionCount = 4;
                }
                else
                {
                    LOG_INFO("returning no extensions are needed!");
                    *OutExtensionCount = 0;
                }
            }
            else if (*OutExtensionCount == 4 && Config::Instance()->VulkanExtensionSpoofing.value_or_default())
            {
                LOG_INFO("returning extension infos");

                std::memset((*OutExtensionProperties)[0].extensionName, 0, sizeof(VK_NVX_BINARY_IMPORT_EXTENSION_NAME));
                std::strcpy((*OutExtensionProperties)[0].extensionName, VK_NVX_BINARY_IMPORT_EXTENSION_NAME);
                (*OutExtensionProperties)[0].specVersion = VK_NVX_BINARY_IMPORT_SPEC_VERSION;

                std::memset((*OutExtensionProperties)[1].extensionName, 0,
                            sizeof(VK_NVX_IMAGE_VIEW_HANDLE_EXTENSION_NAME));
                std::strcpy((*OutExtensionProperties)[1].extensionName, VK_NVX_IMAGE_VIEW_HANDLE_EXTENSION_NAME);
                (*OutExtensionProperties)[1].specVersion = VK_NVX_IMAGE_VIEW_HANDLE_SPEC_VERSION;

                std::memset((*OutExtensionProperties)[2].extensionName, 0,
                            sizeof(VK_EXT_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME));
                std::strcpy((*OutExtensionProperties)[2].extensionName, VK_EXT_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
                (*OutExtensionProperties)[2].specVersion = VK_EXT_BUFFER_DEVICE_ADDRESS_SPEC_VERSION;

                std::memset((*OutExtensionProperties)[3].extensionName, 0,
                            sizeof(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME));
                std::strcpy((*OutExtensionProperties)[3].extensionName, VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME);
                (*OutExtensionProperties)[3].specVersion = VK_KHR_PUSH_DESCRIPTOR_SPEC_VERSION;
            }

            return NVSDK_NGX_Result_Success;
        }
    }

    LOG_INFO("returning no extensions are needed!");

    if (OutExtensionCount != nullptr)
        *OutExtensionCount = 0;

    return NVSDK_NGX_Result_Success;
}

/**
 * @brief Allocates a new parameter map used to provide parameters needed by the DLSS API. The lifetime of this map
 * is managed by the calling application with DestroyParameters().
 */
NVSDK_NGX_API NVSDK_NGX_Result NVSDK_NGX_VULKAN_AllocateParameters(NVSDK_NGX_Parameter** OutParameters)
{
    LOG_FUNC();

    if (Config::Instance()->DLSSEnabled.value_or_default() && NVNGXProxy::NVNGXModule() != nullptr &&
        NVNGXProxy::VULKAN_AllocateParameters() != nullptr)
    {
        LOG_INFO("calling NVNGXProxy::VULKAN_AllocateParameters");
        auto result = NVNGXProxy::VULKAN_AllocateParameters()(OutParameters);
        LOG_INFO("NVNGXProxy::VULKAN_AllocateParameters result: {0:X}", (UINT) result);

        if (result == NVSDK_NGX_Result_Success)
        {
            SetNGXParamAllocType(*(*OutParameters), NGX_AllocTypes::NVDynamic);
            return result;
        }
    }

    auto* params = new NVNGX_Parameters(API::Vulkan, false);
    *OutParameters = params;

    return NVSDK_NGX_Result_Success;
}

NVSDK_NGX_API NVSDK_NGX_Result NVSDK_NGX_VULKAN_GetFeatureRequirements(
    VkInstance VulkanInstance, VkPhysicalDevice PhysicalDevice,
    const NVSDK_NGX_FeatureDiscoveryInfo* FeatureDiscoveryInfo, NVSDK_NGX_FeatureRequirement* OutSupported)
{
    LOG_DEBUG("for FeatureID: {0}", (int) FeatureDiscoveryInfo->FeatureID);

    if (FeatureDiscoveryInfo->FeatureID == NVSDK_NGX_Feature_SuperSampling ||
        (State::Instance().activeFgInput == FGInput::NvngxFG && Nvngx_FG::isVulkanAvailable() &&
         FeatureDiscoveryInfo->FeatureID == NVSDK_NGX_Feature_FrameGeneration))
    {
        if (OutSupported == nullptr)
            OutSupported = new NVSDK_NGX_FeatureRequirement();

        OutSupported->FeatureSupported = NVSDK_NGX_FeatureSupportResult_Supported;
        OutSupported->MinHWArchitecture = 0;

        // Some old windows 10 os version
        strcpy_s(OutSupported->MinOSVersion, "10.0.10240.16384");
        return NVSDK_NGX_Result_Success;
    }

    if (Config::Instance()->DLSSEnabled.value_or_default() && NVNGXProxy::NVNGXModule() == nullptr)
        NVNGXProxy::InitNVNGX();

    if (Config::Instance()->DLSSEnabled.value_or_default() && NVNGXProxy::D3D12_GetFeatureRequirements() != nullptr)
    {
        LOG_DEBUG("calling NVNGXProxy::VULKAN_GetFeatureRequirements");
        auto result = NVNGXProxy::VULKAN_GetFeatureRequirements()(VulkanInstance, PhysicalDevice, FeatureDiscoveryInfo,
                                                                  OutSupported);
        LOG_DEBUG("NVNGXProxy::VULKAN_GetFeatureRequirements result {0:X}", (UINT) result);

        if (result == NVSDK_NGX_Result_Success)
            LOG_DEBUG("FeatureSupported: {0}", (UINT) OutSupported->FeatureSupported);

        return result;
    }
    else
    {
        LOG_DEBUG("VULKAN_GetFeatureRequirements not available for FeatureID: {0}",
                  (int) FeatureDiscoveryInfo->FeatureID);
    }

    OutSupported->FeatureSupported = NVSDK_NGX_FeatureSupportResult_AdapterUnsupported;
    return NVSDK_NGX_Result_FAIL_FeatureNotSupported;
}

/**
 * @brief Allocates a new NVSDK parameter map pre-populated with NGX capabilities and information about available
 * features. The output parameter map may also be used in the same ways as a parameter map allocated with
 * AllocateParameters(). The lifetime of this map is managed by the calling application with DestroyParameters().
 */
NVSDK_NGX_API NVSDK_NGX_Result NVSDK_NGX_VULKAN_GetCapabilityParameters(NVSDK_NGX_Parameter** OutParameters)
{
    LOG_FUNC();

    if (OutParameters == nullptr)
        return NVSDK_NGX_Result_FAIL_InvalidParameter;

    if (Config::Instance()->DLSSEnabled.value_or_default() && NVNGXProxy::NVNGXModule() != nullptr &&
        NVNGXProxy::IsVulkanInited() && NVNGXProxy::VULKAN_GetCapabilityParameters() != nullptr)
    {
        LOG_INFO("calling NVNGXProxy::VULKAN_GetCapabilityParameters");
        auto result = NVNGXProxy::VULKAN_GetCapabilityParameters()(OutParameters);
        LOG_INFO("calling NVNGXProxy::VULKAN_GetCapabilityParameters result: {0:X}", (UINT) result);

        if (result == NVSDK_NGX_Result_Success)
        {
            // Init external NGX table with current configuration and mark as dynamic+external
            InitNGXParameters(*OutParameters, API::Vulkan);
            SetNGXParamAllocType(*(*OutParameters), NGX_AllocTypes::NVDynamic);
            return result;
        }
    }

    // Get custom parameters if using custom backend
    auto& params = *(new NVNGX_Parameters(API::Vulkan, false));
    InitNGXParameters(&params, API::Vulkan);
    *OutParameters = &params;

    return NVSDK_NGX_Result_Success;
}

NVSDK_NGX_API NVSDK_NGX_Result NVSDK_NGX_VULKAN_PopulateParameters_Impl(NVSDK_NGX_Parameter* InParameters)
{
    LOG_FUNC();

    if (InParameters == nullptr)
        return NVSDK_NGX_Result_Fail;

    InitNGXParameters(InParameters, API::Vulkan);

    Nvngx_FG::VULKAN_PopulateParameters_Impl(InParameters);

    return NVSDK_NGX_Result_Success;
}

/**
 * @brief Destroys a given input parameter map created with AllocateParameters or GetCapabilityParameters.
 Must not be called on maps returned by GetParameters(). Unsupported tables will not be freed.
 */
NVSDK_NGX_API NVSDK_NGX_Result NVSDK_NGX_VULKAN_DestroyParameters(NVSDK_NGX_Parameter* InParameters)
{
    LOG_FUNC();

    if (InParameters == nullptr)
        return NVSDK_NGX_Result_Fail;

    const bool success = TryDestroyNGXParameters(InParameters, NVNGXProxy::VULKAN_DestroyParameters());

    return success ? NVSDK_NGX_Result_Success : NVSDK_NGX_Result_Fail;
}

NVSDK_NGX_API NVSDK_NGX_Result NVSDK_NGX_VULKAN_GetScratchBufferSize(NVSDK_NGX_Feature InFeatureId,
                                                                     const NVSDK_NGX_Parameter* InParameters,
                                                                     size_t* OutSizeInBytes)
{
    if (Nvngx_FG::isVulkanAvailable() && InFeatureId == NVSDK_NGX_Feature_FrameGeneration)
    {
        return Nvngx_FG::VULKAN_GetScratchBufferSize(InFeatureId, InParameters, OutSizeInBytes);
    }

    LOG_WARN("-> 52428800");

    *OutSizeInBytes = 52428800;
    return NVSDK_NGX_Result_Success;
}

NVSDK_NGX_API NVSDK_NGX_Result NVSDK_NGX_VULKAN_CreateFeature1(VkDevice InDevice, VkCommandBuffer InCmdList,
                                                               NVSDK_NGX_Feature InFeatureID,
                                                               NVSDK_NGX_Parameter* InParameters,
                                                               NVSDK_NGX_Handle** OutHandle)
{
    if (Nvngx_FG::isVulkanAvailable() && InFeatureID == NVSDK_NGX_Feature_FrameGeneration)
    {
        auto result = Nvngx_FG::VULKAN_CreateFeature1(InDevice, InCmdList, InFeatureID, InParameters, OutHandle);
        LOG_INFO("Creating new modded DLSSG feature with HandleId: {0}", (*OutHandle)->Id);
        return result;
    }
    else if (InFeatureID != NVSDK_NGX_Feature_SuperSampling && InFeatureID != NVSDK_NGX_Feature_RayReconstruction)
    {
        if (Config::Instance()->DLSSEnabled.value_or_default() &&
            NVNGXProxy::InitVulkan(vkInstance, vkPD, vkDevice, vkGIPA, vkGDPA) &&
            NVNGXProxy::VULKAN_CreateFeature1() != nullptr)
        {
            auto result =
                NVNGXProxy::VULKAN_CreateFeature1()(InDevice, InCmdList, InFeatureID, InParameters, OutHandle);
            LOG_INFO("VULKAN_CreateFeature1 result for ({0}): {1:X}", (int) InFeatureID, (UINT) result);
            return result;
        }
        else
        {
            LOG_ERROR("Can't create this feature ({0})!", (int) InFeatureID);
            return NVSDK_NGX_Result_Fail;
        }
    }

    // Create feature
    auto handleId = IFeature::GetNextHandleId();
    LOG_INFO("HandleId: {0}", handleId);

    if (InFeatureID == NVSDK_NGX_Feature_SuperSampling)
    {
        Upscaler upscalerChoice = Upscaler::FSR22; // Default FSR 2.2.1

        // If original NVNGX available use DLSS as base upscaler
        if (IdentifyGpu::getPrimaryGpu().dlssCapable && NVNGXProxy::IsVulkanInited())
            upscalerChoice = Upscaler::DLSS;

        if (Config::Instance()->VulkanUpscaler.has_value())
            upscalerChoice = Config::Instance()->VulkanUpscaler.value();

        LOG_INFO("Creating new {} upscaler", UpscalerDisplayName(upscalerChoice));

        VkContexts[handleId] = {};

        if (!FeatureProvider_Vk::GetFeature(upscalerChoice, handleId, InParameters, &VkContexts[handleId].feature))
        {
            LOG_ERROR("Upscaler can't created");
            return NVSDK_NGX_Result_Fail;
        }
    }
    else if (InFeatureID == NVSDK_NGX_Feature_RayReconstruction)
    {
        LOG_INFO("creating new DLSSD feature");

        VkContexts[handleId] = {};

        if (!FeatureProvider_Vk::GetFeature(Upscaler::DLSSD, handleId, InParameters, &VkContexts[handleId].feature))
        {
            LOG_ERROR("DLSSD can't created");
            return NVSDK_NGX_Result_Fail;
        }
    }

    State::Instance().api = Vulkan;
    auto deviceContext = VkContexts[handleId].feature.get();

    if (*OutHandle == nullptr)
        *OutHandle = new NVSDK_NGX_Handle { handleId };
    else
        (*OutHandle)->Id = handleId;

    State::Instance().autoExposure.reset();

    bool initedPrimary = false;
    {
        ScopedSkipSpoofingGlobal skipSpoofingGlobal {};
        if (deviceContext->Init(vkInstance, vkPD, InDevice, InCmdList, vkGIPA, vkGDPA, InParameters))
        {
            State::Instance().currentFeature = deviceContext;
            if (!NoStalls())
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
            evalCounter = 0;
            initedPrimary = true;
        }
    }
    if (initedPrimary)
    {
        if (PerEyeDlss_Vk::Enabled())
        {
            if (Config::Instance()->UpscalerTwinAtCreate.value_or_default())
                CreatePerEyeTwinVk(InDevice, InCmdList, handleId, InParameters);
            else
                LOG_INFO("Per-eye DLSS VK: no twin at the game's create of feature {} (UpscalerTwinAtCreate false): "
                         "made at the first per-eye evaluate",
                         handleId);
        }
        return NVSDK_NGX_Result_Success;
    }

    LOG_ERROR("CreateFeature failed");
    return NVSDK_NGX_Result_FAIL_PlatformError;
}

NVSDK_NGX_API NVSDK_NGX_Result NVSDK_NGX_VULKAN_CreateFeature(VkCommandBuffer InCmdBuffer,
                                                              NVSDK_NGX_Feature InFeatureID,
                                                              NVSDK_NGX_Parameter* InParameters,
                                                              NVSDK_NGX_Handle** OutHandle)
{
    LOG_FUNC();

    if (Nvngx_FG::isVulkanAvailable() && InFeatureID == NVSDK_NGX_Feature_FrameGeneration)
    {
        auto result = Nvngx_FG::VULKAN_CreateFeature(InCmdBuffer, InFeatureID, InParameters, OutHandle);
        LOG_INFO("Creating new modded DLSSG feature with HandleId: {0}", (*OutHandle)->Id);
        return result;
    }
    else if (InFeatureID != NVSDK_NGX_Feature_SuperSampling && InFeatureID != NVSDK_NGX_Feature_RayReconstruction)
    {
        if (Config::Instance()->DLSSEnabled.value_or_default() &&
            NVNGXProxy::InitVulkan(vkInstance, vkPD, vkDevice, vkGIPA, vkGDPA) &&
            NVNGXProxy::VULKAN_CreateFeature() != nullptr)
        {
            auto result = NVNGXProxy::VULKAN_CreateFeature()(InCmdBuffer, InFeatureID, InParameters, OutHandle);
            LOG_INFO("VULKAN_CreateFeature result for ({0}): {1:X}", (int) InFeatureID, (UINT) result);
            return result;
        }
    }

    return NVSDK_NGX_VULKAN_CreateFeature1(vkDevice, InCmdBuffer, InFeatureID, InParameters, OutHandle);
}

NVSDK_NGX_API NVSDK_NGX_Result NVSDK_NGX_VULKAN_ReleaseFeature(NVSDK_NGX_Handle* InHandle)
{
    if (!InHandle)
        return NVSDK_NGX_Result_Success;

    auto handleId = InHandle->Id;
    if (handleId < DLSS_MOD_ID_OFFSET)
    {
        if (Config::Instance()->DLSSEnabled.value_or_default() && NVNGXProxy::VULKAN_ReleaseFeature() != nullptr)
        {
            auto result = NVNGXProxy::VULKAN_ReleaseFeature()(InHandle);

            if (!shutdown)
                LOG_INFO("VULKAN_ReleaseFeature result for ({0}): {1:X}", handleId, (UINT) result);

            return result;
        }
        else
        {
            return NVSDK_NGX_Result_FAIL_FeatureNotFound;
        }
    }
    else if (handleId >= NVNGX_PROVIDER_ID_OFFSET)
    {
        LOG_INFO("VULKAN_ReleaseFeature modded DLSSG with HandleId: {0}", handleId);
        return Nvngx_FG::VULKAN_ReleaseFeature(InHandle);
    }

    if (!shutdown)
        LOG_INFO("releasing feature with id {0}", handleId);

    // Per-eye DLSS: the twin goes with the feature it was made for, and the foveal crops of both (and a failed
    // first-need create is forgotten with it).
    TwinCreateFailedVk.erase(handleId);
    FoveaCrop_Vk::Forget(handleId);
    if (auto twinId = PerEyeDlss_Vk::TwinOf(handleId); twinId != 0)
    {
        PerEyeDlss_Vk::Forget(handleId);
        FoveaCrop_Vk::Forget(twinId);
        if (auto twinIt = VkContexts.find(twinId); twinIt != VkContexts.end())
        {
            LOG_INFO("Per-eye DLSS VK: releasing twin {} with feature {}", twinId, handleId);
            RetireFeature(twinIt->second.feature, twinId);
            VkContexts.erase(twinIt);
        }
    }

    if (auto deviceContext = VkContexts[handleId].feature.get(); deviceContext)
    {
        RetireFeature(VkContexts[handleId].feature, handleId);
        auto it = std::find_if(VkContexts.begin(), VkContexts.end(),
                               [&handleId](const auto& p) { return p.first == handleId; });
        VkContexts.erase(it);
    }

    if (!NoStalls())
        std::this_thread::sleep_for(std::chrono::milliseconds(500));

    return NVSDK_NGX_Result_Success;
}

NVSDK_NGX_API NVSDK_NGX_Result NVSDK_NGX_VULKAN_EvaluateFeature(VkCommandBuffer InCmdList,
                                                                const NVSDK_NGX_Handle* InFeatureHandle,
                                                                NVSDK_NGX_Parameter* InParameters,
                                                                PFN_NVSDK_NGX_ProgressCallback InCallback)
{
    State& state = State::Instance();

    if (InFeatureHandle == nullptr)
    {
        LOG_DEBUG("InFeatureHandle is null");
        return NVSDK_NGX_Result_FAIL_FeatureNotFound;
    }
    else
    {
        LOG_DEBUG("Handle: {0}", InFeatureHandle->Id);
    }

    if (InCmdList == nullptr)
    {
        LOG_ERROR("InCmdList is null!!!");
        return NVSDK_NGX_Result_Fail;
    }

    auto handleId = InFeatureHandle->Id;
    if (VkContexts[handleId].feature == nullptr) // prevent source api name flicker when dlssg is active
        state.setInputApiName = state.currentInputApiName;

    const auto targetApiName =
        !state.setInputApiName.has_value() ? ApiUpscalerInput::DLSS_VK : state.setInputApiName.value();

    if (state.currentInputApiName != targetApiName)
        state.currentInputApiName = targetApiName;

    state.setInputApiName.reset();

    if (handleId < DLSS_MOD_ID_OFFSET)
    {
        if (Config::Instance()->DLSSEnabled.value_or_default() && NVNGXProxy::VULKAN_EvaluateFeature() != nullptr)
        {
            LOG_DEBUG("VULKAN_EvaluateFeature for ({0})", handleId);
            auto result = NVNGXProxy::VULKAN_EvaluateFeature()(InCmdList, InFeatureHandle, InParameters, InCallback);
            LOG_INFO("VULKAN_EvaluateFeature result for ({0}): {1:X}", handleId, (UINT) result);

            // SR/RR creates are owned by VkContexts. These forwarded handles are other NGX
            // features, not an upscaler seam; never run NR on a frame-generation evaluate.

            return result;
        }
        else
        {
            return NVSDK_NGX_Result_FAIL_FeatureNotFound;
        }
    }
    else if (handleId >= NVNGX_PROVIDER_ID_OFFSET)
    {
        return Nvngx_FG::VULKAN_EvaluateFeature(InCmdList, InFeatureHandle, InParameters, InCallback);
    }

    evalCounter++;
    if (Config::Instance()->SkipFirstFrames.has_value() && evalCounter < Config::Instance()->SkipFirstFrames.value())
        return NVSDK_NGX_Result_Success;

    if (InCallback)
        LOG_WARN("callback exist");

    // Per-eye DLSS: the other eye's frames go to the twin feature (Route), and BeforeEvaluate (below, before NR) puts
    // our same-eye vectors into the parameters; the guard puts the game's back on every way out.
    const bool perEye = PerEyeDlss_Vk::Enabled();
    if (perEye)
    {
        // The twin at first need: the routing went per eye (F6) and this feature has none yet. Tried once per feature.
        if (PerEyeDlss_Vk::RoutingOn() && PerEyeDlss_Vk::TwinOf(handleId) == 0 && !TwinCreateFailedVk.contains(handleId))
        {
            LOG_INFO("Per-eye DLSS VK: routing per eye and feature {} has no twin yet: making it now (first need)",
                     handleId);
            CreatePerEyeTwinVk(vkDevice, InCmdList, handleId, InParameters);
            if (PerEyeDlss_Vk::TwinOf(handleId) == 0)
            {
                TwinCreateFailedVk.insert(handleId);
                LOG_WARN("Per-eye DLSS VK: no twin for feature {}: both eyes share it for its lifetime (not tried "
                         "again)",
                         handleId);
            }
        }
        handleId = PerEyeDlss_Vk::Route(handleId, InParameters);
    }
    struct RestoreGuard
    {
        NVSDK_NGX_Parameter* params;
        bool armed = false;
        ~RestoreGuard()
        {
            if (armed)
                PerEyeDlss_Vk::RestoreInputs(params);
        }
    } restoreGuard { InParameters };

    IFeature_Vk* deviceContext = nullptr;
    auto contextData = &VkContexts[handleId];

    if (state.changeBackend[handleId])
    {
        auto successfulPhase =
            FeatureProvider_Vk::ChangeFeature(state.newBackend, vkInstance, vkPD, vkDevice, InCmdList, vkGIPA, vkGDPA,
                                              handleId, InParameters, contextData);
        evalCounter = 0;

        if (contextData->changeBackendCounter != 0 || !successfulPhase)
        {
            return NVSDK_NGX_Result_Success;
        }
    }

    deviceContext = VkContexts[handleId].feature.get();
    state.currentFeature = deviceContext;

    UpscalerTimeVk::UpscaleStart(InCmdList);

    const auto backend = deviceContext->GetUpscalerType();
    const bool bridged = backend == Upscaler::XeSS_on12 || backend == Upscaler::FSR21_on12 ||
                         backend == Upscaler::FSR22_on12 || backend == Upscaler::FFX_on12;
    const bool rayReconstruction = backend == Upscaler::DLSSD;
    // GPU time per frame ([Inputs] UpscalerGpuTiming; FrameTime_Vk): stamp 0 here, 1 after our periphery, 2 after the
    // game's full pass, 3 after our crop; read back per window by FoveaCrop_Vk's Presents/s line.
    const bool frameTiming = !bridged && Config::Instance()->UpscalerGpuTiming.value_or_default();
    if (frameTiming)
    {
        static bool tried = false;
        if (!FrameTime_Vk::Ready() && !tried)
        {
            tried = true;
            std::string why;
            const bool ok = FrameTime_Vk::Init(vkDevice, vkPD, why);
            LOG_INFO("Per-eye DLSS VK: GPU frame timing {}: {}", ok ? "ON" : "OFF", why);
        }
        if (FrameTime_Vk::Ready())
            FrameTime_Vk::Begin(InCmdList);
    }
    if (perEye)
    {
        PerEyeDlss_Vk::BeforeEvaluate(InCmdList, InParameters, vkDevice, vkPD);
        restoreGuard.armed = true;
    }
    void* originalColor = nullptr;
    InParameters->Get(NVSDK_NGX_Parameter_Color, &originalColor);
    bool nrHandled = false;
    auto nrColor = !bridged
                       ? DlssNr::EvaluateBeforeUpscaleVk(InCmdList, InParameters, vkInstance, vkPD, vkDevice, nrHandled, rayReconstruction)
                       : nullptr;
    if (nrColor)
        InParameters->Set(NVSDK_NGX_Parameter_Color, (void*) nrColor);
    // The cheap periphery (FoveaCrop_Vk, stage A2): with the crop and it on, a render-size pass of this eye enlarged
    // over the eye replaces the game's full pass (the saving). A game feature skipped for it has a stale history
    // when it runs again: DLSS is asked for a reset that once (the game's own Reset put back after).
    const bool periphery =
        perEye && !bridged && FoveaCrop_Vk::Periphery(InCmdList, handleId, InParameters, deviceContext, vkDevice, vkPD);
    if (frameTiming)
        FrameTime_Vk::Mark(InCmdList, 1, true);
    unsigned int resetBefore = 0;
    const bool resetForSkip = !periphery && perEye && FoveaCrop_Vk::TakeMainSkipped(handleId);
    if (resetForSkip)
    {
        InParameters->Get(NVSDK_NGX_Parameter_Reset, &resetBefore);
        InParameters->Set(NVSDK_NGX_Parameter_Reset, 1u);
    }
    auto upscaleResult = periphery || deviceContext->Evaluate(InCmdList, InParameters);
    if (periphery)
        deviceContext->NoteStandIn(); // the overlay must not take the game's feature for unused
    if (frameTiming)
        FrameTime_Vk::Mark(InCmdList, 2, !periphery);
    if (resetForSkip)
        InParameters->Set(NVSDK_NGX_Parameter_Reset, resetBefore);
    // NR before SR: the game's colour is put back only AFTER the foveal crop (below), which copies its colour at the
    // box from these parameters -- NR's box is the crop's, so a crop that read the game's colour drew over every
    // denoised pixel. Every way out puts it back.
    if (!upscaleResult)
        ImGui::InsertNotification({ ImGuiToastType::Error, 10000, "Upscaler failed to run!" });

    if ((!upscaleResult || !deviceContext->IsInited()) &&
        Config::Instance()->VulkanUpscaler.value_or_default() != Upscaler::FSR22)
    {
        state.newBackend = Upscaler::FSR22;
        state.changeBackend[handleId] = true;
        if (nrColor)
            InParameters->Set(NVSDK_NGX_Parameter_Color, originalColor);
        return NVSDK_NGX_Result_Success;
    }

    UpscalerTimeVk::UpscaleEnd(InCmdList);

    // The foveal SR crop (FoveaCrop_Vk, F4): over the full pass just evaluated, on the same command buffer, before NR
    // after-upscale (which then sees the composited picture, as on D3D12).
    if (perEye && upscaleResult && !bridged)
        FoveaCrop_Vk::AfterUpscale(InCmdList, handleId, InParameters, deviceContext, vkDevice, vkPD);
    if (nrColor)
        InParameters->Set(NVSDK_NGX_Parameter_Color, originalColor);
    if (frameTiming)
        FrameTime_Vk::Mark(InCmdList, 3, true);

    // Same pass, for OptiScaler's own upscalers rather than native DLSS. Natively on Vulkan now: the
    // model exports a complete Vulkan surface and OptiScaler's vkCreateDevice hook already adds the
    // two NVX extensions NGX needs, so there is nothing left for the D3D12 bridge to do here.
    //
    // Except when the upscaler itself is bridged. A backend ending in _on12 does its work on a D3D12
    // device and IFeature_VkwDx12 already runs the model there, on the D3D12 command list, over the
    // D3D12 copies of these surfaces. Running it here as well would evaluate the model twice per
    // frame -- once on each side of the bridge -- at double the cost, with the second pass composing
    // over a frame the first had already edited.
    //
    // Asked of the live feature rather than of the config: the config is what was requested and the
    // feature is what is actually running, and they differ for a frame after any backend change and
    // permanently after a fallback.
    if (upscaleResult && !bridged)
        DlssNr::EvaluateAfterUpscaleVk(InCmdList, InParameters, vkInstance, vkPD, vkDevice,
                                      rayReconstruction, nrHandled);

    return upscaleResult ? NVSDK_NGX_Result_Success : NVSDK_NGX_Result_Fail;
}

NVSDK_NGX_API NVSDK_NGX_Result NVSDK_NGX_VULKAN_Shutdown(void)
{
    shutdown = true;

    // Features still waiting in VkRetire are left alive on purpose: NGX is going away, and destroying them after it
    // (at process exit) would call into it. The live ones in VkContexts are not released here either.
    {
        std::lock_guard lock(g_retireMutex);
        for (auto& entry : g_retired)
            (void) entry.first.release();
        g_retired.clear();
    }

    // for (auto const& [key, val] : VkContexts) {
    //     if (val.feature)
    //         NVSDK_NGX_VULKAN_ReleaseFeature(val.feature->Handle());
    // }

    // VkContexts.clear();

    vkInstance = nullptr;
    vkPD = nullptr;
    vkDevice = nullptr;

    State::Instance().currentFeature = nullptr;

    if (Config::Instance()->DLSSEnabled.value_or_default() && NVNGXProxy::IsVulkanInited() &&
        NVNGXProxy::VULKAN_Shutdown() != nullptr)
    {
        auto result = NVNGXProxy::VULKAN_Shutdown()();
        NVNGXProxy::SetVulkanInited(false);
    }

    // Unhooking and cleaning stuff causing issues during shutdown.
    // Disabled for now to check if it cause any issues
    // MenuOverlayVk::UnHookVk();

    Nvngx_FG::VULKAN_Shutdown();

    shutdown = false;
    State::Instance().nvngxVkInited = false;

    return NVSDK_NGX_Result_Success;
}

NVSDK_NGX_API NVSDK_NGX_Result NVSDK_NGX_VULKAN_Shutdown1(VkDevice InDevice)
{
    shutdown = true;

    if (Config::Instance()->DLSSEnabled.value_or_default() && NVNGXProxy::IsVulkanInited() &&
        NVNGXProxy::VULKAN_Shutdown1() != nullptr)
    {
        auto result = NVNGXProxy::VULKAN_Shutdown1()(InDevice);
        NVNGXProxy::SetVulkanInited(false);
    }

    Nvngx_FG::VULKAN_Shutdown1(InDevice);

    shutdown = false;

    return NVSDK_NGX_VULKAN_Shutdown();
}
