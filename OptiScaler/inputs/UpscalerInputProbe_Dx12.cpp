#include "pch.h"

#include "Config.h"
#include "Logger.h"
#include "NVNGX_Parameter.h"

#include "UpscalerInputProbe_Dx12.h"

#include <d3d12.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace
{
// Evaluates arrive on the game's recording threads, Presents on the present thread: everything below is under this.
std::mutex g_mutex;

std::set<unsigned int> g_loggedFirst;
uint64_t g_presentIndex = 0;
unsigned int g_evaluatesThisPresent = 0;
float g_jitterXThisPresent = 0.0f;

// Per 600 Presents that had at least one evaluate. Presents without one are counted, not windowed (menus, loading).
struct Window
{
    uint64_t presents = 0; // with >= 1 evaluate
    uint64_t emptyPresents = 0;
    uint64_t multiPresents = 0; // with >= 2
    uint64_t evaluations = 0;
    uint64_t resets = 0;
    uint64_t zeroJitter = 0;
    std::map<std::pair<unsigned int, int>, uint64_t> perHandle; // (handle, feature) -> evaluates
    std::set<std::pair<int, int>> positions;
    float minX = 1e9f, maxX = -1e9f, minY = 1e9f, maxY = -1e9f;
    double sumX[2] = { 0, 0 };
    uint64_t countX[2] = { 0, 0 };
    float mvScaleX = 0, mvScaleY = 0;
    bool mvScaleChanged = false;
    unsigned int renderW = 0, renderH = 0;
    bool renderChanged = false;
    std::set<int> listTypes;
    // Per parity (even / odd Present), from each Present's first evaluate: what else a mod acting on the DLSS call
    // could change; our foveal crop does not get its effect. Identities are compared only, never referenced.
    std::set<void*> colour[2], motion[2], depth[2], output[2];
    uint64_t firsts[2] = { 0, 0 };
    uint64_t jitterRepeat[2] = { 0, 0 }; // the same jitter as the Present before (both eyes of a pair share it)
    double jitterStep[2] = { 0, 0 };     // |this jitter - the Present before's|
    uint64_t matrices[2] = { 0, 0 };     // Presents whose evaluate handed RR's WorldToView
    double viewT[2][3] = {};             // WorldToView translation
    double viewStep[2] = { 0, 0 };       // |translation - the Present before's| (eye separation + motion)
    double clip[2][4] = {};              // ViewToClip [0], [5], [8], [9] (scale x, y; off-centre x, y)
    double preExposure[2] = { 0, 0 };
    double frameMs[2] = { 0, 0 };
};
Window g_window;
float g_prevJx = 0, g_prevJy = 0;
float g_prevT[3] = {};
bool g_havePrevJitter = false, g_havePrevT = false;

std::vector<std::pair<float, float>> g_sequence;
constexpr size_t kSequence = 48;

const char* FeatureName(int feature)
{
    switch (feature)
    {
    case NVSDK_NGX_Feature_SuperSampling:
        return "SuperSampling";
    case NVSDK_NGX_Feature_RayReconstruction:
        return "RayReconstruction";
    case NVSDK_NGX_Feature_FrameGeneration:
        return "FrameGeneration";
    case -1:
        return "UNRECORDED";
    default:
        return "other";
    }
}

ID3D12Resource* GetResource(NVSDK_NGX_Parameter* params, const char* name)
{
    ID3D12Resource* resource = nullptr;
    if (params->Get(name, &resource) != NVSDK_NGX_Result_Success)
        params->Get(name, (void**) &resource);
    return resource;
}

// Size and format for the log. Described only; no reference is kept.
std::string Describe(NVSDK_NGX_Parameter* params, const char* name)
{
    ID3D12Resource* resource = GetResource(params, name);
    if (resource == nullptr)
        return "none";
    const D3D12_RESOURCE_DESC desc = resource->GetDesc();
    return std::format("{}x{} fmt {} dim {} flags {:#x} ({:p})", desc.Width, desc.Height, (int) desc.Format,
                       (int) desc.Dimension, (unsigned) desc.Flags, (void*) resource);
}

float GetFloat(NVSDK_NGX_Parameter* params, const char* name, float fallback = NAN)
{
    float value = fallback;
    params->Get(name, &value);
    return value;
}

unsigned int GetUInt(NVSDK_NGX_Parameter* params, const char* name, unsigned int fallback = 0)
{
    unsigned int value = fallback;
    params->Get(name, &value);
    return value;
}

int GetInt(NVSDK_NGX_Parameter* params, const char* name, int fallback = -1)
{
    int value = fallback;
    params->Get(name, &value);
    return value;
}

std::string CreateFlags(int flags)
{
    if (flags < 0)
        return "not set";
    std::string s = std::format("{:#x}", flags);
    if (flags & NVSDK_NGX_DLSS_Feature_Flags_IsHDR)
        s += " IsHDR";
    if (flags & NVSDK_NGX_DLSS_Feature_Flags_MVLowRes)
        s += " MVLowRes";
    if (flags & NVSDK_NGX_DLSS_Feature_Flags_MVJittered)
        s += " MVJittered";
    if (flags & NVSDK_NGX_DLSS_Feature_Flags_DepthInverted)
        s += " DepthInverted";
    if (flags & NVSDK_NGX_DLSS_Feature_Flags_AutoExposure)
        s += " AutoExposure";
    if (flags & NVSDK_NGX_DLSS_Feature_Flags_DoSharpening)
        s += " DoSharpening";
    return s;
}

void LogFirst(ID3D12GraphicsCommandList* cmdList, unsigned int handle, int feature, NVSDK_NGX_Parameter* params)
{
    LOG_INFO("Input probe DX12: FIRST evaluate of handle {} ({}), list type {}, Present {}", handle,
             FeatureName(feature), (int) cmdList->GetType(), g_presentIndex);
    LOG_INFO("Input probe DX12:   colour  {}", Describe(params, NVSDK_NGX_Parameter_Color));
    LOG_INFO("Input probe DX12:   motion  {}", Describe(params, NVSDK_NGX_Parameter_MotionVectors));
    LOG_INFO("Input probe DX12:   depth   {}", Describe(params, NVSDK_NGX_Parameter_Depth));
    LOG_INFO("Input probe DX12:   output  {}", Describe(params, NVSDK_NGX_Parameter_Output));
    LOG_INFO("Input probe DX12:   exposure {}, transparency mask {}, bias mask {}",
             Describe(params, NVSDK_NGX_Parameter_ExposureTexture),
             Describe(params, NVSDK_NGX_Parameter_TransparencyMask),
             Describe(params, NVSDK_NGX_Parameter_DLSS_Input_Bias_Current_Color_Mask));
    if (feature == NVSDK_NGX_Feature_RayReconstruction)
    {
        // Every resource input nvngx_dlssd.dll (310.9.1) looks up by name -- read out of the DLL's own strings -- and
        // which of them this game sets. Each logged with its size and format, or skipped when absent.
        static const char* const kRrResources[] = {
            "DLSS.Input.DiffuseAlbedo", "DLSS.Input.SpecularAlbedo", "GBuffer.Normals", "GBuffer.Roughness",
            "GBuffer.SpecularMvec", "GBuffer.Emissive", "GBuffer.IndirectAlbedo", "MotionVectorsReflection",
            "DLSSD.SpecularHitDistance", "DLSSD.DiffuseHitDistance", "DLSSD.ReflectedAlbedo",
            "DLSSD.SpecularRayDirection", "DLSSD.SpecularRayDirectionHitDistance", "DLSSD.DiffuseRayDirection",
            "DLSSD.DiffuseRayDirectionHitDistance", "DLSSD.ColorBeforeParticles", "DLSSD.ColorBeforeTransparency",
            "DLSSD.ColorAfterTransparency", "DLSSD.ColorBeforeFog", "DLSSD.ScreenSpaceSubsurfaceScatteringGuide",
            "DLSSD.DepthOfFieldGuide", "DLSSD.ResponsivityMask", "DLSSD.Alpha", "DLSSD.OutputAlpha",
            "DLSS.TransparencyLayer", "DLSS.TransparencyLayerOpacity", "DLSS.TransparencyLayerMvecs",
            "DLSS.DisocclusionMask", "DLSS.Input.Reduce.Ghost.Mask", "IsParticleMask", "AnimatedTextureMask",
            "RayTracingHitDistance" };
        int present = 0;
        for (const char* name : kRrResources)
        {
            const std::string d = Describe(params, name);
            if (d == "none")
                continue;
            ++present;
            LOG_INFO("Input probe DX12:   RR input {} = {}", name, d);
        }
        LOG_INFO("Input probe DX12:   RR: {} of {} optional inputs set; modes: denoise {}, roughness {}, HW depth {}",
                 present, std::size(kRrResources), GetInt(params, "DLSS.Denoise.Mode"),
                 GetInt(params, "DLSS.Roughness.Mode"), GetInt(params, "DLSS.Use.HW.Depth"));
        // The camera matrices are handed as pointers to 16 floats.
        for (const char* name : { "WorldToViewMatrix", "ViewToClipMatrix", "ClipToPrevClipMatrix",
                                  "InvViewProjectionMatrix", "WorldToScreenMatrix" })
        {
            void* ptr = nullptr;
            if (params->Get(name, &ptr) != NVSDK_NGX_Result_Success || ptr == nullptr)
                continue;
            float m[16] {};
            std::memcpy(m, ptr, sizeof(m));
            LOG_INFO("Input probe DX12:   RR matrix {}: rows ({:.4f} {:.4f} {:.4f} {:.4f}) ({:.4f} {:.4f} {:.4f} {:.4f}) "
                     "({:.4f} {:.4f} {:.4f} {:.4f}) ({:.4f} {:.4f} {:.4f} {:.4f})",
                     name, m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8], m[9], m[10], m[11], m[12], m[13],
                     m[14], m[15]);
        }
    }
    LOG_INFO("Input probe DX12:   MV scale ({}, {}), jitter ({}, {}), reset {}, pre-exposure {}, exposure scale {}, "
             "sharpness {}",
             GetFloat(params, NVSDK_NGX_Parameter_MV_Scale_X), GetFloat(params, NVSDK_NGX_Parameter_MV_Scale_Y),
             GetFloat(params, NVSDK_NGX_Parameter_Jitter_Offset_X),
             GetFloat(params, NVSDK_NGX_Parameter_Jitter_Offset_Y), GetUInt(params, NVSDK_NGX_Parameter_Reset),
             GetFloat(params, NVSDK_NGX_Parameter_DLSS_Pre_Exposure),
             GetFloat(params, NVSDK_NGX_Parameter_DLSS_Exposure_Scale),
             GetFloat(params, NVSDK_NGX_Parameter_Sharpness));
    LOG_INFO("Input probe DX12:   render subrect {}x{}, colour subrect base ({}, {}), MV subrect base ({}, {}), depth "
             "subrect base ({}, {})",
             GetUInt(params, NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Width),
             GetUInt(params, NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Height),
             GetUInt(params, NVSDK_NGX_Parameter_DLSS_Input_Color_Subrect_Base_X),
             GetUInt(params, NVSDK_NGX_Parameter_DLSS_Input_Color_Subrect_Base_Y),
             GetUInt(params, NVSDK_NGX_Parameter_DLSS_Input_MV_SubrectBase_X),
             GetUInt(params, NVSDK_NGX_Parameter_DLSS_Input_MV_SubrectBase_Y),
             GetUInt(params, NVSDK_NGX_Parameter_DLSS_Input_Depth_Subrect_Base_X),
             GetUInt(params, NVSDK_NGX_Parameter_DLSS_Input_Depth_Subrect_Base_Y));
}

void LogWindow()
{
    auto& w = g_window;
    std::string handles;
    for (auto& [key, n] : w.perHandle)
        handles += std::format("{} ({}) x{}, ", key.first, FeatureName(key.second), n);
    std::string lists;
    for (auto t : w.listTypes)
        lists += std::format("{} ", t);
    const double meanEven = w.countX[0] ? w.sumX[0] / w.countX[0] : 0.0;
    const double meanOdd = w.countX[1] ? w.sumX[1] / w.countX[1] : 0.0;
    LOG_INFO("Input probe DX12: {} Presents with an evaluate ({} with 2+; {} Presents WITHOUT one in between), {} "
             "evaluations: {}list types {}",
             w.presents, w.multiPresents, w.emptyPresents, w.evaluations, handles, lists);
    LOG_INFO("Input probe DX12:   jitter: {} distinct positions, x [{:.4f}, {:.4f}] y [{:.4f}, {:.4f}], {} zero; mean x "
             "on even Presents {:.4f}, odd {:.4f} (opposite signs = the cycle splits by eye under AER); {} resets; MV "
             "scale ({}, {}){}; render {}x{}{}",
             w.positions.size(), w.minX, w.maxX, w.minY, w.maxY, w.zeroJitter, meanEven, meanOdd, w.resets, w.mvScaleX,
             w.mvScaleY, w.mvScaleChanged ? " (CHANGED in the window)" : "", w.renderW, w.renderH,
             w.renderChanged ? " (CHANGED in the window)" : "");
    std::string perEye;
    for (int p = 0; p < 2; ++p)
    {
        const double n = (double) std::max<uint64_t>(w.firsts[p], 1);
        const double m = (double) std::max<uint64_t>(w.matrices[p], 1);
        perEye += std::format("{}: textures colour {} motion {} depth {} output {} distinct; jitter repeats the Present "
                              "before {} of {}, step {:.3f}; RR matrices {} -- view translation ({:.3f} {:.3f} "
                              "{:.3f}), step {:.4f}, clip scale ({:.4f} {:.4f}) off-centre ({:.4f} {:.4f}); "
                              "pre-exposure {:.3f}; frame ms {:.2f}{}",
                              p == 0 ? "even" : "odd", w.colour[p].size(), w.motion[p].size(), w.depth[p].size(),
                              w.output[p].size(), w.jitterRepeat[p], w.firsts[p], w.jitterStep[p] / n, w.matrices[p],
                              w.viewT[p][0] / m, w.viewT[p][1] / m, w.viewT[p][2] / m, w.viewStep[p] / m,
                              w.clip[p][0] / m, w.clip[p][1] / m, w.clip[p][2] / m, w.clip[p][3] / m,
                              w.preExposure[p] / n, w.frameMs[p] / n, p == 0 ? " | " : "");
    }
    LOG_INFO("Input probe DX12:   per eye -- {}", perEye);
    w = Window {};
}
} // namespace

namespace UpscalerInputProbe_Dx12
{
bool Enabled()
{
    static const bool enabled = Config::Instance()->UpscalerInputProbe.value_or_default();
    return enabled;
}

void OnCreate(int featureId, unsigned int handle, bool optiPath, NVSDK_NGX_Parameter* params)
{
    if (!Enabled() || params == nullptr)
        return;
    std::lock_guard lock(g_mutex);
    LOG_INFO("Input probe DX12: CREATE handle {} feature {} ({}) via {}: render {}x{}, output {}x{}, quality {}, flags "
             "{}, Present {}",
             handle, featureId, FeatureName(featureId), optiPath ? "OptiScaler's own feature" : "NATIVE passthrough",
             GetUInt(params, NVSDK_NGX_Parameter_Width), GetUInt(params, NVSDK_NGX_Parameter_Height),
             GetUInt(params, NVSDK_NGX_Parameter_OutWidth), GetUInt(params, NVSDK_NGX_Parameter_OutHeight),
             GetInt(params, NVSDK_NGX_Parameter_PerfQualityValue),
             CreateFlags(GetInt(params, NVSDK_NGX_Parameter_DLSS_Feature_Create_Flags)), g_presentIndex);
}

void OnEvaluate(ID3D12GraphicsCommandList* cmdList, unsigned int handle, int featureId, NVSDK_NGX_Parameter* params)
{
    if (!Enabled() || cmdList == nullptr || params == nullptr)
        return;
    // Frame generation is handed depth and motion too; only upscaler evaluates are the ones per eye.
    if (featureId == NVSDK_NGX_Feature_FrameGeneration)
        return;
    std::lock_guard lock(g_mutex);
    if (g_loggedFirst.insert(handle).second)
        LogFirst(cmdList, handle, featureId, params);

    auto& w = g_window;
    ++w.evaluations;
    ++w.perHandle[{ handle, featureId }];
    w.listTypes.insert((int) cmdList->GetType());
    if (GetUInt(params, NVSDK_NGX_Parameter_Reset) != 0)
        ++w.resets;

    const float jx = GetFloat(params, NVSDK_NGX_Parameter_Jitter_Offset_X, 0.0f);
    const float jy = GetFloat(params, NVSDK_NGX_Parameter_Jitter_Offset_Y, 0.0f);
    if (jx == 0.0f && jy == 0.0f)
        ++w.zeroJitter;
    w.positions.insert({ (int) std::lround(jx * 1024.0f), (int) std::lround(jy * 1024.0f) });
    w.minX = std::min(w.minX, jx);
    w.maxX = std::max(w.maxX, jx);
    w.minY = std::min(w.minY, jy);
    w.maxY = std::max(w.maxY, jy);
    if (g_evaluatesThisPresent++ == 0)
    {
        g_jitterXThisPresent = jx;
        const int p = (int) (g_presentIndex & 1);
        ++w.firsts[p];
        w.colour[p].insert(GetResource(params, NVSDK_NGX_Parameter_Color));
        w.motion[p].insert(GetResource(params, NVSDK_NGX_Parameter_MotionVectors));
        w.depth[p].insert(GetResource(params, NVSDK_NGX_Parameter_Depth));
        w.output[p].insert(GetResource(params, NVSDK_NGX_Parameter_Output));
        if (g_havePrevJitter)
        {
            if (jx == g_prevJx && jy == g_prevJy)
                ++w.jitterRepeat[p];
            w.jitterStep[p] += std::hypot(jx - g_prevJx, jy - g_prevJy);
        }
        g_prevJx = jx;
        g_prevJy = jy;
        g_havePrevJitter = true;
        void* view = nullptr;
        void* proj = nullptr;
        if (params->Get("WorldToViewMatrix", &view) == NVSDK_NGX_Result_Success && view != nullptr &&
            params->Get("ViewToClipMatrix", &proj) == NVSDK_NGX_Result_Success && proj != nullptr)
        {
            float v[16], c[16];
            std::memcpy(v, view, sizeof(v));
            std::memcpy(c, proj, sizeof(c));
            ++w.matrices[p];
            for (int i = 0; i < 3; ++i)
                w.viewT[p][i] += v[12 + i];
            if (g_havePrevT)
                w.viewStep[p] += std::sqrt((v[12] - g_prevT[0]) * (v[12] - g_prevT[0]) +
                                           (v[13] - g_prevT[1]) * (v[13] - g_prevT[1]) +
                                           (v[14] - g_prevT[2]) * (v[14] - g_prevT[2]));
            std::memcpy(g_prevT, v + 12, sizeof(g_prevT));
            g_havePrevT = true;
            w.clip[p][0] += c[0];
            w.clip[p][1] += c[5];
            w.clip[p][2] += c[8];
            w.clip[p][3] += c[9];
        }
        w.preExposure[p] += GetFloat(params, NVSDK_NGX_Parameter_DLSS_Pre_Exposure, 0.0f);
        w.frameMs[p] += GetFloat(params, NVSDK_NGX_Parameter_FrameTimeDeltaInMsec, 0.0f);
    }
    if (g_sequence.size() < kSequence)
    {
        g_sequence.push_back({ jx, jy });
        if (g_sequence.size() == kSequence)
        {
            std::string s;
            for (auto& [x, y] : g_sequence)
                s += std::format("({:.3f},{:.3f}) ", x, y);
            LOG_INFO("Input probe DX12: the first {} jitter values in order: {}", kSequence, s);
        }
    }

    const float sx = GetFloat(params, NVSDK_NGX_Parameter_MV_Scale_X, 0.0f);
    const float sy = GetFloat(params, NVSDK_NGX_Parameter_MV_Scale_Y, 0.0f);
    if (w.evaluations > 1 && (sx != w.mvScaleX || sy != w.mvScaleY))
        w.mvScaleChanged = true;
    w.mvScaleX = sx;
    w.mvScaleY = sy;

    unsigned int rw = GetUInt(params, NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Width);
    unsigned int rh = GetUInt(params, NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Height);
    if (w.evaluations > 1 && (rw != w.renderW || rh != w.renderH))
        w.renderChanged = true;
    w.renderW = rw;
    w.renderH = rh;
}

void OnPresent()
{
    if (!Enabled())
        return;
    std::lock_guard lock(g_mutex);
    auto& w = g_window;
    if (g_evaluatesThisPresent == 0)
    {
        // Only between gameplay Presents; before the first evaluate nothing is windowed at all.
        if (w.presents > 0)
            ++w.emptyPresents;
    }
    else
    {
        ++w.presents;
        if (g_evaluatesThisPresent >= 2)
            ++w.multiPresents;
        const int parity = (int) (g_presentIndex & 1);
        w.sumX[parity] += g_jitterXThisPresent;
        ++w.countX[parity];
    }
    g_evaluatesThisPresent = 0;
    ++g_presentIndex;
    if (w.presents >= 600)
        LogWindow();
}
} // namespace UpscalerInputProbe_Dx12
