#include "pch.h"
#include "StaleEyeShare.h"
#include "StaleEyeSettingsSync.h"

#include "Config.h"
#include "Logger.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <filesystem>
#include <format>
#include <string>
#include <mutex>

namespace
{
std::mutex g_mutex;
// When the overlay may copy the layer's state into the Config (StaleEyeSettingsSync.h); guarded by g_gateMutex.
StaleEyeSettingsSync::Gate g_settingsGate;
std::mutex g_gateMutex;
RvrToolsStaleEyeCameraV1 g_latest {};
// Every publication, by sequence (guarded by g_mutex): the layer asks for the one an image belongs to. It used to
// keep its own history by reading the latest once per xrEndFrame, which runs on another clock: 93 frames of a 10 s
// window found no new one and 93 found two, so its history had holes.
RvrToolsStaleEyeCameraV1 g_history[kStaleEyeCameraHistory] = {};
bool g_loggedFirst = false;
float g_lastTangents[4] = {};
bool g_haveTangents = false;
float g_lastProjection[16] = {};
std::atomic<uint64_t> g_reportedRead { 0 };
std::atomic<int> g_depthRequested { 0 };
std::atomic<uint64_t> g_gameDevice { 0 };
} // namespace

bool StaleEyeShare::Enabled() { return Config::Instance()->StaleEyeCameraShare.value_or_default(); }

namespace
{
struct LayerApi
{
    PFN_XrObserver_GetStaleSettingsV1 get = nullptr;
    PFN_XrObserver_SetStaleSettingsV1 set = nullptr;
    PFN_XrObserver_GetStaleStatusV1 status = nullptr;
    PFN_XrObserver_GetViewsV1 views = nullptr; // may be absent (an older layer)
};
// The layer is loaded by the OpenXR loader, after this DLL: looked up (by name, not held) until found.
LayerApi Layer()
{
    static LayerApi api;
    static ULONGLONG nextTry = 0;
    if (api.get != nullptr)
        return api;
    const ULONGLONG now = GetTickCount64();
    if (now < nextTry)
        return api;
    nextTry = now + 2000;
    if (HMODULE layer = GetModuleHandleW(L"xr_observer.dll"))
    {
        LayerApi found;
        found.get = reinterpret_cast<PFN_XrObserver_GetStaleSettingsV1>(GetProcAddress(layer, "XrObserver_GetStaleSettingsV1"));
        found.set = reinterpret_cast<PFN_XrObserver_SetStaleSettingsV1>(GetProcAddress(layer, "XrObserver_SetStaleSettingsV1"));
        found.status = reinterpret_cast<PFN_XrObserver_GetStaleStatusV1>(GetProcAddress(layer, "XrObserver_GetStaleStatusV1"));
        found.views = reinterpret_cast<PFN_XrObserver_GetViewsV1>(GetProcAddress(layer, "XrObserver_GetViewsV1"));
        if (found.get != nullptr && found.set != nullptr && found.status != nullptr)
        {
            api = found;
            wchar_t from[MAX_PATH] = {};
            GetModuleFileNameW(layer, from, MAX_PATH);
            LOG_INFO("Stale-eye layer: controls found in xr_observer.dll, loaded from {}; the views/gaze export "
                     "XrObserver_GetViewsV1 {}",
                     wstring_to_string(from),
                     found.views != nullptr ? "found" : "NOT found (an older layer: no FOV, no gaze from it)");
        }
        else
        {
            static bool logged = false;
            if (!logged)
            {
                logged = true;
                LOG_INFO("Stale-eye layer: xr_observer.dll is loaded but has no controls (an older build)");
            }
        }
    }
    return api;
}
} // namespace

void StaleEyeShare::OfferLayerFromFolder()
{
    static constexpr wchar_t kLayerName[] = L"XR_APILAYER_OPTISCALER_RVR_xr_observer";
    auto* config = Config::Instance();
    if (!Enabled())
        return;
    if (!config->StaleEyeLayerInFolder.value_or_default())
    {
        LOG_INFO("Stale-eye layer: [Inputs] StaleEyeLayerInFolder is off; only a registered layer can load");
        return;
    }
    if (!config->MainDllPath.has_value())
    {
        LOG_INFO("Stale-eye layer: the OptiScaler folder is not known; only a registered layer can load");
        return;
    }
    const std::filesystem::path folder = std::filesystem::path(config->MainDllPath.value()) / L"xr_observer";
    const auto manifest = folder / (std::wstring(kLayerName) + L".json");
    const auto dll = folder / L"xr_observer.dll";
    std::error_code ec;
    if (!std::filesystem::exists(manifest, ec) || !std::filesystem::exists(dll, ec))
    {
        LOG_INFO("Stale-eye layer: not in the game's folder ({} with its manifest); only a registered layer can load",
                 dll.string());
        return;
    }
    // The loader reads its layer variables only in a process that is not elevated (its "secure" getenv): said here,
    // so that a layer that does not load has its reason in the log.
    bool elevated = false;
    HANDLE token = nullptr;
    if (OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &token))
    {
        TOKEN_ELEVATION elevation {};
        DWORD size = 0;
        if (GetTokenInformation(token, TokenElevation, &elevation, sizeof(elevation), &size))
            elevated = elevation.TokenIsElevated != 0;
        CloseHandle(token);
    }
    const auto append = [](const wchar_t* name, const std::wstring& value, bool first)
    {
        wchar_t existing[4096] = {};
        const DWORD length = GetEnvironmentVariableW(name, existing, (DWORD) std::size(existing));
        std::wstring all = value;
        if (length > 0 && length < std::size(existing) && std::wstring(existing).find(value) == std::wstring::npos)
            all = first ? value + L";" + existing : std::wstring(existing) + L";" + value;
        else if (length > 0 && length < std::size(existing))
            all = existing; // already in it
        return SetEnvironmentVariableW(name, all.c_str()) != 0;
    };
    const bool path = append(L"XR_API_LAYER_PATH", folder.wstring(), true);
    const bool enable = append(L"XR_ENABLE_API_LAYERS", kLayerName, false);
    const bool disable = SetEnvironmentVariableW(L"XR_OBSERVER_DISABLE", L"1") != 0;
    // The layer's eye gaze (XR_EXT_eye_gaze_interaction, gaze_share.cpp) is on by default; off, the layer still hands the
    // views on and the gaze comes from Varjo's reader. A test switch: in one game the layer brought up a notice in the
    // headset and a short hang while loading.
    if (!config->StaleEyeLayerGaze.value_or_default())
    {
        SetEnvironmentVariableW(L"XR_OBSERVER_GAZE", L"0");
        LOG_INFO("Stale-eye layer: [Inputs] StaleEyeLayerGaze is off: the layer adds no eye gaze extension and no action "
                 "set (XR_OBSERVER_GAZE=0); the views are still handed on");
    }
    LOG_INFO("Stale-eye layer: offered to this process's OpenXR loader from {} (XR_API_LAYER_PATH {}, "
             "XR_ENABLE_API_LAYERS {}; a registered copy switched off with XR_OBSERVER_DISABLE {}). The process is {}",
             folder.string(), path ? "set" : "FAILED", enable ? "set" : "FAILED", disable ? "set" : "FAILED",
             elevated ? "ELEVATED: the loader ignores the first two, so NO layer will load (run the game without "
                        "administrator rights, or register the layer)"
                      : "not elevated");
    LOG_INFO("Stale-eye layer: it has loaded when its own log appears beside it (xr_observer_<pid>.log in that "
             "folder) and 'controls found in xr_observer.dll' follows here");
}

bool StaleEyeShare::LayerSettings(XrObserverStaleSettingsV1* out)
{
    const auto api = Layer();
    out->structSize = sizeof(*out);
    return api.get != nullptr && api.get(out) == 1;
}

bool StaleEyeShare::SetLayerSettings(const XrObserverStaleSettingsV1& in)
{
    const auto api = Layer();
    return api.set != nullptr && api.set(&in) == 1;
}

bool StaleEyeShare::LayerStatus(XrObserverStaleStatusV1* out)
{
    const auto api = Layer();
    out->structSize = sizeof(*out);
    return api.status != nullptr && api.status(out) == 1;
}

bool StaleEyeShare::LayerViews(XrObserverViewsV1* out)
{
    const auto api = Layer();
    out->structSize = sizeof(*out);
    return api.views != nullptr && api.views(out) == 1;
}

void StaleEyeShare::SyncLayer()
{
    // Called by the camera's Publish (gameplay) AND by the overlay (menus before gameplay): once, whichever is first.
    static std::atomic<bool> pushed { false };
    static std::atomic<int> failures { 0 };
    if (pushed.load())
        return;
    XrObserverStaleSettingsV1 current {};
    if (!LayerSettings(&current)) // absent, or no frame submitted yet
        return;
    if (pushed.exchange(true))
        return;
    auto* config = Config::Instance();
    XrObserverStaleSettingsV1 s {};
    s.structSize = sizeof(s);
    s.mode = std::clamp(config->StaleEyeMode.value_or_default(), 0, 3);
    s.pairingOffset = std::clamp(config->StaleEyePairing.value_or_default(), -1, 4); // -1 = AUTO
    s.nearThreshold = std::clamp(config->StaleEyeNear.value_or_default(), 0.0f, 8.0f);
    s.bothEyes = config->StaleEyeBothEyes.value_or_default() ? 1 : 0;
    s.headMode = std::clamp(config->StaleEyeHead.value_or_default(), 0, 1);
    s.treatMode = std::clamp(config->StaleEyeTreat.value_or_default(), 0, 2);
    s.treatStrength = std::clamp(config->StaleEyeTreatStrength.value_or_default(), 0.0f, 1.0f);
    s.treatGated = config->StaleEyeTreatGated.value_or_default() ? 1 : 0;
    s.treatFullAt = std::clamp(config->StaleEyeTreatFullAt.value_or_default(), 0.001f, 1.0f);
    s.strafeOnly = config->StaleEyeStrafeOnly.value_or_default() ? 1 : 0;
    s.depthMode = std::clamp(config->StaleEyeDepthMode.value_or_default(), 0, 3);
    s.attachDistanceOnly = config->StaleEyeAttachDistanceOnly.value_or_default() ? 1 : 0;
    s.nearWithBody = config->StaleEyeNearWithBody.value_or_default() ? 1 : 0;
    s.fillFromOtherEye = config->StaleEyeFillOtherEye.value_or_default() ? 1 : 0;
    s.diagnostics = config->StaleEyeDiagnostics.value_or_default() ? 1 : 0;
    s.clampMetres = std::clamp(config->StaleEyeClamp.value_or_default(), 0.0f, 50.0f);
    s.singleMetres = std::clamp(config->StaleEyeSingle.value_or_default(), 0.1f, 1000.0f);
    const bool ok = SetLayerSettings(s);
    if (ok)
    {
        std::lock_guard lock(g_gateMutex);
        g_settingsGate.Pushed(s, GetTickCount64());
    }
    else
    {
        pushed = false; // tried again at the next call
        if (failures++ != 0)
            return; // said once
    }
    LOG_INFO("Stale-eye layer: ini settings pushed ({}): mode {}, pairing {}, near {:.2f}, both eyes {}, head {}; "
             "strafe only {}, depth mode {}, nearest corrected {:.2f} m, single distance {:.2f} m; near things with "
             "the {}; behind the player {}",
             ok ? "ok" : "FAILED", s.mode, s.pairingOffset, s.nearThreshold, s.bothEyes, s.headMode, s.strafeOnly,
             s.depthMode, s.clampMetres, s.singleMetres, s.nearWithBody ? "body" : "head",
             s.fillFromOtherEye ? "from the other eye" : "stretched");
}

bool StaleEyeShare::LayerSettingsReady(const XrObserverStaleSettingsV1& layerNow)
{
    std::lock_guard lock(g_gateMutex);
    const bool wasReady = g_settingsGate.WasPushed() && g_settingsGate.Ready(layerNow, GetTickCount64());
    static std::atomic<bool> logged { false };
    if (wasReady && !logged.exchange(true))
        LOG_INFO("Stale-eye layer: it has the saved settings (mode {}, near {:.2f}, depth mode {}): the overlay shows "
                 "its controls and keeps the ini in step from now on",
                 layerNow.mode, layerNow.nearThreshold, layerNow.depthMode);
    return wasReady;
}

void StaleEyeShare::Publish(const float currentToPrevious[16], uint32_t width, uint32_t height, bool reversedDepth,
                            const float* tangents, const float* projection)
{
    SyncLayer(); // outside g_mutex: it calls into the layer
    LARGE_INTEGER now {};
    QueryPerformanceCounter(&now);
    std::lock_guard lock(g_mutex);
    g_latest.structSize = sizeof(RvrToolsStaleEyeCameraV1);
    g_latest.version = 1;
    g_latest.magic = kStaleEyeCameraMagicV1;
    ++g_latest.sequence;
    g_latest.qpc = now.QuadPart;
    std::memcpy(g_latest.currentToPrevious, currentToPrevious, sizeof(g_latest.currentToPrevious));
    g_latest.width = width;
    g_latest.height = height;
    g_latest.reversedDepth = reversedDepth ? 1u : 0u;
    g_latest.gameDevice = g_gameDevice.load(std::memory_order_relaxed);
    // The previous publication's projection is the stale eye's; both must be known for the tangents to be valid.
    std::memcpy(g_latest.staleTangents, g_lastTangents, sizeof(g_lastTangents));
    std::memcpy(g_latest.staleProjection, g_lastProjection, sizeof(g_lastProjection));
    if (tangents != nullptr)
        std::memcpy(g_latest.freshTangents, tangents, sizeof(g_latest.freshTangents));
    if (projection != nullptr)
    {
        std::memcpy(g_latest.freshProjection, projection, sizeof(g_latest.freshProjection));
        std::memcpy(g_lastProjection, projection, sizeof(g_lastProjection));
    }
    g_latest.tangentsValid = tangents != nullptr && g_haveTangents ? 1u : 0u;
    if (tangents != nullptr)
        std::memcpy(g_lastTangents, tangents, sizeof(g_lastTangents));
    g_haveTangents = tangents != nullptr;
    g_history[g_latest.sequence % kStaleEyeCameraHistory] = g_latest;
    if (!g_loggedFirst)
    {
        g_loggedFirst = true;
        LOG_INFO("Stale-eye share: first camera published ({}x{}, row 0 {:.5f} {:.5f} {:.5f} {:.5f}), projection {}",
                 width, height, currentToPrevious[0], currentToPrevious[1], currentToPrevious[2], currentToPrevious[3],
                 tangents != nullptr ? std::format("tangents L{:+.4f} R{:+.4f} U{:+.4f} D{:+.4f}", tangents[0],
                                                   tangents[1], tangents[2], tangents[3])
                                     : std::string("unknown"));
    }
}

uint64_t StaleEyeShare::LatestSequence(uint32_t* width, uint32_t* height)
{
    std::lock_guard lock(g_mutex);
    if (width != nullptr)
        *width = g_latest.width;
    if (height != nullptr)
        *height = g_latest.height;
    return g_latest.sequence;
}

void StaleEyeShare::SetDepthShare(const DepthShareInfo& info)
{
    std::lock_guard lock(g_mutex);
    g_latest.depthGeneration = info.generation;
    std::memcpy(g_latest.depthHandles, info.handles, sizeof(g_latest.depthHandles));
    g_latest.depthWidth = info.width;
    g_latest.depthHeight = info.height;
    g_latest.writtenFenceHandle = info.writtenFence;
    g_latest.readFenceHandle = info.readFence;
    g_latest.depthLegacyHandles = info.legacyHandles ? 1u : 0u;
}

void StaleEyeShare::SetDepthSequence(uint64_t sequence)
{
    std::lock_guard lock(g_mutex);
    g_latest.depthSequence = sequence;
}

void StaleEyeShare::SetDepthMarks(bool marks)
{
    std::lock_guard lock(g_mutex);
    g_latest.depthMarks = marks ? 1u : 0u;
}

uint64_t StaleEyeShare::ReportedRead() { return g_reportedRead.load(std::memory_order_acquire); }

bool StaleEyeShare::DepthRequested() { return g_depthRequested.load(std::memory_order_acquire) != 0; }

extern "C" __declspec(dllexport) void __cdecl RvrTools_RequestDepthShareV1(int on)
{
    if (g_depthRequested.exchange(on != 0 ? 1 : 0) != (on != 0 ? 1 : 0))
        LOG_INFO("Stale-eye depth share: the layer {} depth", on != 0 ? "REQUESTS" : "no longer requests");
}

extern "C" __declspec(dllexport) void __cdecl RvrTools_ReportDepthReadV1(uint64_t sequence)
{
    uint64_t current = g_reportedRead.load(std::memory_order_relaxed);
    while (sequence > current && !g_reportedRead.compare_exchange_weak(current, sequence, std::memory_order_release))
    {
    }
}

void StaleEyeShare::SetGameDevice(const void* device)
{
    const uint64_t value = reinterpret_cast<uint64_t>(device);
    if (g_gameDevice.exchange(value, std::memory_order_relaxed) != value)
        LOG_INFO("Stale-eye share: game device {:X}", value);
}

extern "C" __declspec(dllexport) int __cdecl RvrTools_GetStaleEyeCameraV1(RvrToolsStaleEyeCameraV1* out)
{
    if (out == nullptr || out->structSize != sizeof(RvrToolsStaleEyeCameraV1))
        return 0;
    std::lock_guard lock(g_mutex);
    if (g_latest.sequence == 0)
        return 0;
    *out = g_latest;
    return 1;
}

extern "C" __declspec(dllexport) int __cdecl RvrTools_GetStaleEyeCameraAtV1(uint64_t sequence,
                                                                               RvrToolsStaleEyeCameraV1* out)
{
    if (out == nullptr || out->structSize != sizeof(RvrToolsStaleEyeCameraV1) || sequence == 0)
        return 0;
    std::lock_guard lock(g_mutex);
    const auto& entry = g_history[sequence % kStaleEyeCameraHistory];
    if (entry.sequence != sequence)
        return 0; // not published yet, or overwritten
    *out = entry;
    // The depth share's state is the latest's (the handles, and which sequence was copied last).
    out->depthGeneration = g_latest.depthGeneration;
    std::memcpy(out->depthHandles, g_latest.depthHandles, sizeof(out->depthHandles));
    out->depthWidth = g_latest.depthWidth;
    out->depthHeight = g_latest.depthHeight;
    out->writtenFenceHandle = g_latest.writtenFenceHandle;
    out->readFenceHandle = g_latest.readFenceHandle;
    out->depthSequence = g_latest.depthSequence;
    out->depthLegacyHandles = g_latest.depthLegacyHandles;
    out->depthMarks = g_latest.depthMarks;
    out->gameDevice = g_latest.gameDevice;
    return 1;
}
