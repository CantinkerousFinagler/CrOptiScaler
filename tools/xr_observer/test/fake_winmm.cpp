// Stand-in for the game's winmm.dll in layer tests: publishes the stale-eye camera ABI
// (OptiScaler/inputs/StaleEyeShareAbi.h) under test control. Built as winmm.dll so the layer finds it by name.
#include <windows.h>
#include <cmath>
#include <cstring>
#include "../../../OptiScaler/inputs/StaleEyeShareAbi.h"

static RvrToolsStaleEyeCameraV1 g_cam {};
static RvrToolsStaleEyeCameraV1 g_history[kStaleEyeCameraHistory] = {}; // every publication, as the game keeps them
static double g_yawDeg = 0.0;
static float g_tangents[4] = { -1, 1, 1, -1 };
static std::uint64_t g_reported = 0;

extern "C" __declspec(dllexport) void SetYaw(double deg) { g_yawDeg = deg; }
extern "C" __declspec(dllexport) void SetTangents(float l, float r, float u, float d)
{
    g_tangents[0] = l;
    g_tangents[1] = r;
    g_tangents[2] = u;
    g_tangents[3] = d;
}
extern "C" __declspec(dllexport) void SetDepth(std::uint64_t generation, const std::uint64_t* handles,
                                               std::uint32_t width, std::uint32_t height, std::uint64_t written,
                                               std::uint64_t read)
{
    g_cam.depthGeneration = generation;
    for (int i = 0; i < kStaleEyeDepthSlots; ++i)
        g_cam.depthHandles[i] = handles[i];
    g_cam.depthWidth = width;
    g_cam.depthHeight = height;
    g_cam.writtenFenceHandle = written;
    g_cam.readFenceHandle = read;
}
extern "C" __declspec(dllexport) void SetDepthSequence(std::uint64_t sequence) { g_cam.depthSequence = sequence; }
// 1: the depth handles are the old kind (IDXGIResource::GetSharedHandle), as a device without NT handles shares them.
extern "C" __declspec(dllexport) void SetDepthLegacy(int legacy) { g_cam.depthLegacyHandles = legacy != 0 ? 1u : 0u; }
// 1: the depth carries the marks of the player's kind (negated depth), as some games' do.
extern "C" __declspec(dllexport) void SetDepthMarks(int marks) { g_cam.depthMarks = marks != 0 ? 1u : 0u; }
extern "C" __declspec(dllexport) std::uint64_t GetReportedRead() { return g_reported; }
static int g_requested = -1;
extern "C" __declspec(dllexport) int GetDepthRequested() { return g_requested; }
extern "C" __declspec(dllexport) void __cdecl RvrTools_RequestDepthShareV1(int on) { g_requested = on; }

// Publishes the next Present: fresh -> stale is a yaw of g_yawDeg with the current tangents (the same for both).
extern "C" __declspec(dllexport) void Advance()
{
    const double a = g_yawDeg * 3.14159265358979 / 180.0;
    const double R[3][3] = { { std::cos(a), 0, std::sin(a) }, { 0, 1, 0 }, { -std::sin(a), 0, std::cos(a) } };
    const float l = g_tangents[0], r = g_tangents[1], u = g_tangents[2], d = g_tangents[3];
    const double S[3][3] = { { 2 / (r - l), 0, -(r + l) / (r - l) }, { 0, 2 / (u - d), -(u + d) / (u - d) }, { 0, 0, 1 } };
    const double Si[3][3] = { { (r - l) / 2, 0, (r + l) / 2 }, { 0, (u - d) / 2, (u + d) / 2 }, { 0, 0, 1 } };
    // H = S R S^-1 (ndc fresh -> ndc stale for directions)
    double T[3][3] = {}, H[3][3] = {};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k)
                T[i][j] += R[i][k] * Si[k][j];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k)
                H[i][j] += S[i][k] * T[k][j];
    float m[16] = {};
    const int idx[3] = { 0, 1, 3 };
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            m[idx[i] * 4 + idx[j]] = (float) H[i][j];
    m[2 * 4 + 2] = 1.0f;
    LARGE_INTEGER now {};
    QueryPerformanceCounter(&now);
    g_cam.structSize = sizeof(g_cam);
    g_cam.version = 1;
    g_cam.magic = kStaleEyeCameraMagicV1;
    ++g_cam.sequence;
    g_cam.qpc = now.QuadPart;
    std::memcpy(g_cam.currentToPrevious, m, sizeof(m));
    g_cam.width = g_cam.depthWidth != 0 ? g_cam.depthWidth : 1000;
    g_cam.height = g_cam.depthHeight != 0 ? g_cam.depthHeight : 1000;
    g_cam.reversedDepth = 1;
    g_cam.tangentsValid = 1;
    std::memcpy(g_cam.freshTangents, g_tangents, sizeof(g_tangents));
    std::memcpy(g_cam.staleTangents, g_tangents, sizeof(g_tangents));
    g_history[g_cam.sequence % kStaleEyeCameraHistory] = g_cam;
}

// Publishes the next Present with a full camera (WARP tests): the clip map fresh -> stale and the game's projection
// (the same for both eyes); tangents are taken from the projection as Fc5Taa does.
extern "C" __declspec(dllexport) void AdvanceWith(const float* currentToPrevious, const float* projection)
{
    const float s = projection[14];
    const float t[4] = { (-1.0f - s * projection[2]) / projection[0], (1.0f - s * projection[2]) / projection[0],
                         (1.0f - s * projection[6]) / projection[5], (-1.0f - s * projection[6]) / projection[5] };
    LARGE_INTEGER now {};
    QueryPerformanceCounter(&now);
    g_cam.structSize = sizeof(g_cam);
    g_cam.version = 1;
    g_cam.magic = kStaleEyeCameraMagicV1;
    ++g_cam.sequence;
    g_cam.qpc = now.QuadPart;
    std::memcpy(g_cam.currentToPrevious, currentToPrevious, sizeof(g_cam.currentToPrevious));
    g_cam.width = g_cam.depthWidth;
    g_cam.height = g_cam.depthHeight;
    g_cam.reversedDepth = 1;
    g_cam.tangentsValid = 1;
    std::memcpy(g_cam.freshTangents, t, sizeof(t));
    std::memcpy(g_cam.staleTangents, t, sizeof(t));
    std::memcpy(g_cam.freshProjection, projection, sizeof(g_cam.freshProjection));
    std::memcpy(g_cam.staleProjection, projection, sizeof(g_cam.staleProjection));
    g_history[g_cam.sequence % kStaleEyeCameraHistory] = g_cam;
}

extern "C" __declspec(dllexport) int __cdecl RvrTools_GetStaleEyeCameraV1(RvrToolsStaleEyeCameraV1* out)
{
    if (out == nullptr || out->structSize != sizeof(RvrToolsStaleEyeCameraV1) || g_cam.sequence == 0)
        return 0;
    *out = g_cam;
    return 1;
}

extern "C" __declspec(dllexport) int __cdecl RvrTools_GetStaleEyeCameraAtV1(std::uint64_t sequence,
                                                                               RvrToolsStaleEyeCameraV1* out)
{
    if (out == nullptr || out->structSize != sizeof(RvrToolsStaleEyeCameraV1) || sequence == 0)
        return 0;
    const auto& entry = g_history[sequence % kStaleEyeCameraHistory];
    if (entry.sequence != sequence)
        return 0;
    *out = entry;
    out->depthGeneration = g_cam.depthGeneration;
    std::memcpy(out->depthHandles, g_cam.depthHandles, sizeof(out->depthHandles));
    out->depthWidth = g_cam.depthWidth;
    out->depthHeight = g_cam.depthHeight;
    out->writtenFenceHandle = g_cam.writtenFenceHandle;
    out->readFenceHandle = g_cam.readFenceHandle;
    out->depthSequence = g_cam.depthSequence;
    out->depthLegacyHandles = g_cam.depthLegacyHandles;
    out->depthMarks = g_cam.depthMarks;
    return 1;
}

// The overlay for the headset (RvrToolsOverlayV1), under test control. Nothing is visible until a test says so.
static RvrToolsOverlayV1 g_overlay {};
static int g_overlayAsked = 0;
extern "C" __declspec(dllexport) void SetOverlay(std::uint64_t generation, const std::uint64_t* handles,
                                                 std::uint32_t width, std::uint32_t height, std::uint64_t written)
{
    g_overlay.generation = generation;
    for (int i = 0; i < kOverlaySlots; ++i)
        g_overlay.handles[i] = handles[i];
    g_overlay.width = width;
    g_overlay.height = height;
    // The used part is smaller than the image, as the game DLL's is (its textures grow in steps).
    g_overlay.usedWidth = width - 16;
    g_overlay.usedHeight = height - 8;
    g_overlay.writtenFenceHandle = written;
    g_overlay.panelWidth = 1.6f;
    g_overlay.panelDistance = 1.5f;
    g_overlay.panelDown = 0.25f;
}
extern "C" __declspec(dllexport) void SetOverlayLegacy(int legacy) { g_overlay.legacyHandles = legacy != 0 ? 1u : 0u; }
extern "C" __declspec(dllexport) void SetOverlayFrame(std::uint64_t sequence, int visible)
{
    LARGE_INTEGER now {};
    QueryPerformanceCounter(&now);
    g_overlay.sequence = sequence;
    g_overlay.visible = visible != 0 ? 1u : 0u;
    g_overlay.qpc = now.QuadPart;
}
extern "C" __declspec(dllexport) int GetOverlayAsked() { return g_overlayAsked; }
extern "C" __declspec(dllexport) int __cdecl RvrTools_GetOverlayV1(RvrToolsOverlayV1* out)
{
    if (out == nullptr || out->structSize != sizeof(RvrToolsOverlayV1))
        return 0;
    ++g_overlayAsked;
    *out = g_overlay;
    out->structSize = sizeof(RvrToolsOverlayV1);
    return 1;
}

extern "C" __declspec(dllexport) void __cdecl RvrTools_ReportDepthReadV1(std::uint64_t sequence)
{
    if (sequence > g_reported)
        g_reported = sequence;
}
