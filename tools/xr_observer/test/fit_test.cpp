// Offline test of the same-eye FIT (OptiScaler/inputs/PerEyeFitShader.h), the very shader the game DLL compiles and
// the very sequence of dispatches it runs, on a real D3D11 device with made-up depth and motion: one projective map
// between the eyes (the parallax about 45 px at the nearest depth, as in a third-person game), noise of 0.05 px, motion in
// R16G16_FLOAT divided by an MV scale of -size as the game hands it over, and a block of a quarter of the picture
// that moves by itself (the player character).
//   fit_test
// A "Present" is one fit for the same eye.
//   1  no start at all: accepted, the map found
//   2  a start 5 px off: accepted at once (the ordinary case must not change)
//   3  a STALE start, 205 px off (what a cut or a menu leaves behind): rejected twice, then the start is given up
//      and the third Present is accepted with the map found
//   4  a picture without any one map (random motion) over a good start: rejected every time, the start untouched;
//      the picture good again: accepted at once
//   5  a start 3000 px off: rejected twice, the third Present is accepted
//   6  the WIDE fit (one sample per thread, 11 dispatches) against the fit as it was in ONE GROUP, from the same
//      starts on the same pictures: an accepted fit is the same bits, and so is all that the game keeps of a
//      rejected one (the eyes' slots, the verdict)
//   7  the cost on the GPU, by timestamps around 100 and 200 fits of each kind: the time follows the count (a timer
//      that reads a constant does not pass), and a wide fit costs under half of a fit in one group
//   8  a picture WITHOUT DEPTH (a menu without the world behind it) over a good start: rejected as "too few
//      samples with depth", not as "the solve failed" (the fit in one group says the latter, shown beside it), the
//      start untouched; the depth back: accepted at once
// Controls, each must FAIL:
//   FIT_TEST_OLD=1        the start as it was (KEEP_STALE_START): case 3 stays rejected for good -- in a game one
//                         eye was locked out for 7 to 15 s at a time
//   FIT_TEST_ONE_GROUP=1  the fit in one group in the wide fit's place: case 7 finds nothing saved (and case 8 the
//                         old name of the reason)
#include <windows.h>
#include <d3d11.h>
#include <d3dcompiler.h>

#define FIT_SHADER_WITH_ONE_GROUP
#include "../../../OptiScaler/inputs/PerEyeFitShader.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace
{
constexpr UINT kSize = 1024;
constexpr float kKZ = 100.0f;

struct Map
{
    float r0[4], r1[4], r2[3];
};
const Map kTrue = { { 1.0f, 0.0003f, -0.008f, -0.0004f }, { -0.0003f, 1.0f, 0.0f, 0.0002f }, { 0.0001f, -0.0002f, 0.0f } };

uint32_t g_random = 0x2468ace1u;
float Random() // 0 .. 1
{
    g_random ^= g_random << 13;
    g_random ^= g_random >> 17;
    g_random ^= g_random << 5;
    return (float) (g_random & 0xffffff) / (float) 0x1000000;
}

uint16_t Half(float value)
{
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    const int exponent = (int) ((bits >> 23) & 0xff) - 127 + 15;
    const uint32_t mantissa = bits & 0x7fffffu;
    if (exponent <= 0)
    {
        if (exponent < -10)
            return (uint16_t) sign;
        const uint32_t m = (mantissa | 0x800000u) >> (1 - exponent);
        return (uint16_t) (sign | ((m + 0x1000u) >> 13));
    }
    if (exponent >= 31)
        return (uint16_t) (sign | 0x7c00u);
    return (uint16_t) (sign | ((uint32_t) exponent << 10) | ((mantissa + 0x1000u) >> 13));
}

ID3D11Device* g_device = nullptr;
ID3D11DeviceContext* g_context = nullptr;
ID3D11Texture2D* g_depth = nullptr;
ID3D11Texture2D* g_motion = nullptr;
ID3D11ShaderResourceView* g_depthSrv = nullptr;
ID3D11ShaderResourceView* g_motionSrv = nullptr;
ID3D11Buffer* g_fits = nullptr;
ID3D11Buffer* g_fitsRead = nullptr;
ID3D11UnorderedAccessView* g_fitsUav = nullptr;
ID3D11Buffer* g_work = nullptr;
ID3D11UnorderedAccessView* g_workUav = nullptr;
ID3D11Buffer* g_constants = nullptr;
ID3D11ComputeShader* g_init = nullptr;
ID3D11ComputeShader* g_accumulate = nullptr;
ID3D11ComputeShader* g_solve = nullptr;
ID3D11ComputeShader* g_oneGroup = nullptr;
ID3D11Query* g_disjoint = nullptr;
ID3D11Query* g_timeStart = nullptr;
ID3D11Query* g_timeEnd = nullptr;
bool g_oneGroupInPlace = false; // the control FIT_TEST_ONE_GROUP
std::vector<float> g_depthData(kSize * kSize);

// The picture: the map's motion everywhere, or no one map at all.
void WritePicture(bool oneMap)
{
    std::vector<uint16_t> motion(kSize * kSize * 2);
    for (UINT py = 0; py < kSize; ++py)
    {
        for (UINT px = 0; px < kSize; ++px)
        {
            const float u = ((float) px + 0.5f) / kSize, v = ((float) py + 0.5f) / kSize;
            const float x = u * 2.0f - 1.0f, y = 1.0f - v * 2.0f;
            const float zz = g_depthData[py * kSize + px] * kKZ;
            const float w = kTrue.r2[0] * x + kTrue.r2[1] * y + kTrue.r2[2] * zz + 1.0f;
            const float xp = (kTrue.r0[0] * x + kTrue.r0[1] * y + kTrue.r0[2] * zz + kTrue.r0[3]) / w;
            const float yp = (kTrue.r1[0] * x + kTrue.r1[1] * y + kTrue.r1[2] * zz + kTrue.r1[3]) / w;
            float mx = (xp - x) * kSize * 0.5f + (Random() - 0.5f) * 0.1f;
            float my = -(yp - y) * kSize * 0.5f + (Random() - 0.5f) * 0.1f;
            // The player character: the middle half by the middle half, moving by itself.
            if (px >= kSize / 4 && px < kSize * 3 / 4 && py >= kSize / 4 && py < kSize * 3 / 4)
            {
                mx += 60.0f;
                my -= 25.0f;
            }
            if (!oneMap)
            {
                mx = (Random() - 0.5f) * 400.0f;
                my = (Random() - 0.5f) * 400.0f;
            }
            // As the game hands it over: divided by the MV scale, -size.
            motion[(py * kSize + px) * 2 + 0] = Half(mx / -(float) kSize);
            motion[(py * kSize + px) * 2 + 1] = Half(my / -(float) kSize);
        }
    }
    g_context->UpdateSubresource(g_motion, 0, nullptr, motion.data(), kSize * 4, 0);
}

void WriteFits(const float* fits) { g_context->UpdateSubresource(g_fits, 0, nullptr, fits, 0, 0); }

bool ReadFits(float* fits)
{
    g_context->CopyResource(g_fitsRead, g_fits);
    D3D11_MAPPED_SUBRESOURCE mapped {};
    if (FAILED(g_context->Map(g_fitsRead, 0, D3D11_MAP_READ, 0, &mapped)))
        return false;
    std::memcpy(fits, mapped.pData, 60 * sizeof(float));
    g_context->Unmap(g_fitsRead, 0);
    return true;
}

void StartOf(float* fits, float pixelsOff)
{
    std::memset(fits, 0, 60 * sizeof(float));
    std::memcpy(fits + 0, kTrue.r0, sizeof(kTrue.r0));
    std::memcpy(fits + 4, kTrue.r1, sizeof(kTrue.r1));
    std::memcpy(fits + 8, kTrue.r2, sizeof(kTrue.r2));
    fits[3] += pixelsOff * 2.0f / kSize; // the x row's constant, in clip units
    fits[11] = 1.0f;
    fits[12] = 0.05f;
    fits[13] = 0.9f;
    fits[14] = 1.0f; // accepted
    fits[15] = 4096.0f;
}

// One fit, bound and dispatched as the game DLL does it (PerEyeDlss_Dx11.cpp), or the fit as it was in one group.
void RunFit(bool oneGroup)
{
    struct
    {
        float mvScale[2];
        uint32_t cur, prev;
        float knee0, knee1, gain;
        uint32_t radius;
    } constants { { -(float) kSize, -(float) kSize }, 0, 1, 0.1f, 0.4f, 2.0f, 2 };
    g_context->UpdateSubresource(g_constants, 0, nullptr, &constants, 0, 0);
    ID3D11ShaderResourceView* srvs[2] = { g_depthSrv, g_motionSrv };
    ID3D11UnorderedAccessView* uavs[2] = { g_fitsUav, g_workUav };
    ID3D11UnorderedAccessView* nullUavs[2] = {};
    g_context->CSSetConstantBuffers(0, 1, &g_constants);
    g_context->CSSetShaderResources(0, 2, srvs);
    g_context->CSSetUnorderedAccessViews(0, 2, uavs, nullptr);
    if (oneGroup)
    {
        g_context->CSSetShader(g_oneGroup, nullptr, 0);
        g_context->Dispatch(1, 1, 1);
    }
    else
        DispatchFit(g_context, g_init, g_accumulate, g_solve);
    g_context->CSSetUnorderedAccessViews(0, 2, nullUavs, nullptr);
}

struct Present
{
    int reason = -1;
    float streak = -1, residual = 99, inliers = 0, mapError = 99; // of the eye's slot, against the true map
    float slotXConstant = 0;
};

Present Dispatch()
{
    RunFit(g_oneGroupInPlace);
    float f[60] = {};
    Present p;
    if (!ReadFits(f))
        return p;
    p.reason = (int) f[2 * 20 + 16];
    p.streak = f[0 * 20 + 17];
    p.residual = f[2 * 20 + 12];
    p.inliers = f[2 * 20 + 13];
    p.slotXConstant = f[3];
    float worst = 0;
    const float want[11] = { kTrue.r0[0], kTrue.r0[1], kTrue.r0[2], kTrue.r0[3], kTrue.r1[0], kTrue.r1[1],
                             kTrue.r1[2], kTrue.r1[3], kTrue.r2[0], kTrue.r2[1], kTrue.r2[2] };
    for (int i = 0; i < 11; ++i)
        worst = (std::max)(worst, std::fabs(f[i] - want[i]));
    p.mapError = worst;
    return p;
}

int g_failures = 0, g_checked = 0;
void Check(const char* what, bool ok, const Present& p)
{
    std::printf("%s: reason %d (0 = accepted), rejections in a row %.0f, residual %.3f px, within 1 px %.0f%%, the eye's "
                "map off the true one by %.5f at the most -> %s\n",
                what, p.reason, p.streak, p.residual, p.inliers * 100.0f, p.mapError, ok ? "OK" : "FAIL");
    ++g_checked;
    if (!ok)
        ++g_failures;
}

void CheckPlain(const char* what, bool ok)
{
    std::printf("%s -> %s\n", what, ok ? "OK" : "FAIL");
    ++g_checked;
    if (!ok)
        ++g_failures;
}

// Case 6: both kinds of fit from the same start on the picture of now.
// What the game keeps must be the SAME BITS: the two eyes' slots (the maps, their statistics, the reason and the
// count of rejections) and, of this Present's own fit, the verdict and the count of samples. An ACCEPTED fit is the
// same bits altogether. The numbers of a REJECTED fit (kept for the log only) may differ a little (measured: 0.0004 px
// of the residual, 0.0001 of a row). The sums add up in the same order in both kinds; what fits the measurements is
// that the two shaders do not round the products of a FRACTIONAL weight alike, which a rejected fit's nearly
// singular equations enlarge. An accepted fit has whole weights (0 or 1) in every round but the first, and its sums
// then no longer depend on the last bits of the round before. Not a race: the wide fit run twice is the same bits.
bool SameFits(const float* start, const char* what)
{
    float one[60] = {}, wide[60] = {}, again[60] = {};
    WriteFits(start);
    RunFit(true);
    const bool readOne = ReadFits(one);
    WriteFits(start);
    RunFit(g_oneGroupInPlace);
    const bool readAgain = ReadFits(again);
    WriteFits(start);
    RunFit(g_oneGroupInPlace);
    const bool readWide = ReadFits(wide) && readAgain;
    const bool twice = std::memcmp(wide, again, sizeof(wide)) == 0;
    float worst = 0;
    int differing = 0, keptDiffering = 0;
    for (int i = 0; i < 60; ++i)
    {
        if (std::memcmp(&one[i], &wide[i], sizeof(float)) == 0)
            continue;
        ++differing;
        // Slots 0 and 1, and of slot 2 the accepted flag, the count of samples and the reason.
        if (i < 40 || i == 2 * 20 + 14 || i == 2 * 20 + 15 || i >= 2 * 20 + 16)
            ++keptDiffering;
        const float d = std::fabs(one[i] - wide[i]);
        worst = d == d ? (std::max)(worst, d) : 1e30f;
    }
    const bool accepted = one[2 * 20 + 16] == 0.0f;
    const bool ok = readOne && readWide && twice && keptDiffering == 0 && (accepted ? differing == 0 : worst < 0.01f);
    std::printf("6 %s: one group reason %.0f residual %.4f px, wide reason %.0f residual %.4f px; %d of 60 numbers "
                "differ (%d of those the game keeps), by %.3g at the most; the wide fit run twice %s -> %s\n",
                what, one[2 * 20 + 16], one[2 * 20 + 12], wide[2 * 20 + 16], wide[2 * 20 + 12], differing,
                keptDiffering, worst, twice ? "is the same bits" : "DIFFERS FROM ITSELF", ok ? "ok" : "NOT THE SAME");
    return ok;
}

// Case 7: the GPU's time for `count` fits between two timestamps, ms; under 0 when the timer gave nothing.
double TimeFits(int count, bool oneGroup)
{
    g_context->Begin(g_disjoint);
    g_context->End(g_timeStart);
    for (int i = 0; i < count; ++i)
        RunFit(oneGroup);
    g_context->End(g_timeEnd);
    g_context->End(g_disjoint);
    g_context->Flush();
    D3D11_QUERY_DATA_TIMESTAMP_DISJOINT disjoint {};
    UINT64 start = 0, end = 0;
    for (int tries = 0; tries < 5000; ++tries)
    {
        if (g_context->GetData(g_disjoint, &disjoint, sizeof(disjoint), 0) == S_OK &&
            g_context->GetData(g_timeStart, &start, sizeof(start), 0) == S_OK &&
            g_context->GetData(g_timeEnd, &end, sizeof(end), 0) == S_OK)
        {
            if (disjoint.Disjoint || disjoint.Frequency == 0 || end < start)
                return -1.0;
            return (double) (end - start) / (double) disjoint.Frequency * 1000.0;
        }
        Sleep(1);
    }
    return -1.0;
}

double MedianTime(int count, bool oneGroup)
{
    std::vector<double> times;
    for (int i = 0; i < 7; ++i)
    {
        const double t = TimeFits(count, oneGroup);
        if (t >= 0.0)
            times.push_back(t);
    }
    if (times.size() < 4)
        return -1.0;
    std::sort(times.begin(), times.end());
    return times[times.size() / 2];
}

ID3D11ComputeShader* CompileStage(const char* source, size_t length, const char* name, const char* stage, bool keepStale)
{
    D3D_SHADER_MACRO macros[3] = {};
    int n = 0;
    if (stage != nullptr)
        macros[n++] = { stage, "1" };
    if (keepStale)
        macros[n++] = { "KEEP_STALE_START", "1" };
    ID3DBlob* code = nullptr;
    ID3DBlob* errors = nullptr;
    ID3D11ComputeShader* shader = nullptr;
    // As the game DLL compiles it (PerEyeDlss_Dx11.cpp, Compile).
    if (FAILED(D3DCompile(source, length, name, macros, nullptr, "CSMain", "cs_5_0", D3DCOMPILE_OPTIMIZATION_LEVEL3, 0,
                          &code, &errors)) ||
        FAILED(g_device->CreateComputeShader(code->GetBufferPointer(), code->GetBufferSize(), nullptr, &shader)))
    {
        std::printf("FAILED: %s does not compile: %s\n", name,
                    errors != nullptr ? (const char*) errors->GetBufferPointer() : "no message");
        return nullptr;
    }
    return shader;
}
} // namespace

int main()
{
    char value[8] = {};
    const bool old = GetEnvironmentVariableA("FIT_TEST_OLD", value, sizeof(value)) > 0 && value[0] == '1';
    g_oneGroupInPlace = GetEnvironmentVariableA("FIT_TEST_ONE_GROUP", value, sizeof(value)) > 0 && value[0] == '1';
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
    if (FAILED(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, nullptr, 0, D3D11_SDK_VERSION, &g_device,
                                 nullptr, &g_context)))
    {
        std::printf("FAILED: no D3D11 device\n");
        return 1;
    }
    g_init = CompileStage(kFitShader, sizeof(kFitShader) - 1, "PerEye_FitInit", "FIT_INIT", old);
    g_accumulate = CompileStage(kFitShader, sizeof(kFitShader) - 1, "PerEye_FitAccumulate", "FIT_ACCUMULATE", old);
    g_solve = CompileStage(kFitShader, sizeof(kFitShader) - 1, "PerEye_FitSolve", "FIT_SOLVE", old);
    g_oneGroup = CompileStage(kFitShaderOneGroup, sizeof(kFitShaderOneGroup) - 1, "PerEye_FitOneGroup", nullptr, false);
    if (g_init == nullptr || g_accumulate == nullptr || g_solve == nullptr || g_oneGroup == nullptr)
        return 1;
    wchar_t compiler[MAX_PATH] = {};
    GetModuleFileNameW(GetModuleHandleW(L"d3dcompiler_47.dll"), compiler, MAX_PATH);
    std::printf("the fit %s, %s, compiled by %ls\n", g_oneGroupInPlace ? "IN ONE GROUP (the control)" : "wide",
                old ? "the start AS IT WAS (KEEP_STALE_START)" : "with the start given up", compiler);

    D3D11_TEXTURE2D_DESC td {};
    td.Width = td.Height = kSize;
    td.MipLevels = td.ArraySize = 1;
    td.SampleDesc.Count = 1;
    td.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    td.Format = DXGI_FORMAT_R32_FLOAT;
    bool ok = SUCCEEDED(g_device->CreateTexture2D(&td, nullptr, &g_depth));
    td.Format = DXGI_FORMAT_R16G16_FLOAT;
    ok = ok && SUCCEEDED(g_device->CreateTexture2D(&td, nullptr, &g_motion));
    ok = ok && SUCCEEDED(g_device->CreateShaderResourceView(g_depth, nullptr, &g_depthSrv));
    ok = ok && SUCCEEDED(g_device->CreateShaderResourceView(g_motion, nullptr, &g_motionSrv));
    D3D11_BUFFER_DESC bd {};
    bd.ByteWidth = 60 * sizeof(float);
    bd.BindFlags = D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;
    bd.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
    bd.StructureByteStride = 16;
    ok = ok && SUCCEEDED(g_device->CreateBuffer(&bd, nullptr, &g_fits));
    ok = ok && SUCCEEDED(g_device->CreateUnorderedAccessView(g_fits, nullptr, &g_fitsUav));
    D3D11_BUFFER_DESC rd = bd;
    rd.Usage = D3D11_USAGE_STAGING;
    rd.BindFlags = 0;
    rd.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    ok = ok && SUCCEEDED(g_device->CreateBuffer(&rd, nullptr, &g_fitsRead));
    // As the game DLL makes it (PerEyeDlss_Dx11.cpp, Setup).
    D3D11_BUFFER_DESC wd {};
    wd.ByteWidth = kFitWorkFloats * sizeof(float);
    wd.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
    wd.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
    wd.StructureByteStride = sizeof(float);
    ok = ok && SUCCEEDED(g_device->CreateBuffer(&wd, nullptr, &g_work));
    ok = ok && SUCCEEDED(g_device->CreateUnorderedAccessView(g_work, nullptr, &g_workUav));
    D3D11_BUFFER_DESC cd {};
    cd.ByteWidth = 32;
    cd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    ok = ok && SUCCEEDED(g_device->CreateBuffer(&cd, nullptr, &g_constants));
    D3D11_QUERY_DESC qd {};
    qd.Query = D3D11_QUERY_TIMESTAMP_DISJOINT;
    ok = ok && SUCCEEDED(g_device->CreateQuery(&qd, &g_disjoint));
    qd.Query = D3D11_QUERY_TIMESTAMP;
    ok = ok && SUCCEEDED(g_device->CreateQuery(&qd, &g_timeStart));
    ok = ok && SUCCEEDED(g_device->CreateQuery(&qd, &g_timeEnd));
    if (!ok)
    {
        std::printf("FAILED: the test's textures, buffers and queries\n");
        return 1;
    }
    // Depth with relief: a smooth field and a rough one, 0.002 .. 0.11 (far .. near).
    for (UINT py = 0; py < kSize; ++py)
        for (UINT px = 0; px < kSize; ++px)
            g_depthData[py * kSize + px] =
                0.002f + 0.05f * (0.5f + 0.5f * std::sin(px * 0.013f) * std::cos(py * 0.017f)) + 0.058f * Random();
    g_context->UpdateSubresource(g_depth, 0, nullptr, g_depthData.data(), kSize * 4, 0);

    float fits[60];
    const auto found = [](const Present& p) { return p.reason == 0 && p.residual < 0.2f && p.mapError < 0.002f; };

    // 1: no start at all.
    std::memset(fits, 0, sizeof(fits));
    WriteFits(fits);
    WritePicture(true);
    Present p = Dispatch();
    Check("1 no start", found(p) && p.streak == 0.0f, p);

    // 2: a start 5 px off.
    StartOf(fits, 5.0f);
    WriteFits(fits);
    p = Dispatch();
    Check("2 a start 5 px off", found(p) && p.streak == 0.0f, p);

    // 3: a stale start.
    StartOf(fits, 205.0f);
    WriteFits(fits);
    const float staleConstant = fits[3];
    p = Dispatch();
    Check("3 a STALE start, Present 1 (rejected, the start kept)",
          p.reason != 0 && p.streak == 1.0f && p.slotXConstant == staleConstant, p);
    p = Dispatch();
    Check("3 a STALE start, Present 2 (rejected, the start kept)",
          p.reason != 0 && p.streak == 2.0f && p.slotXConstant == staleConstant, p);
    p = Dispatch();
    Check("3 a STALE start, Present 3 (the start given up: accepted)", found(p) && p.streak == 0.0f, p);
    p = Dispatch();
    Check("3 a STALE start, Present 4 (from the new start)", found(p) && p.streak == 0.0f, p);

    // 4: a picture without any one map, over the good start that case 3 left.
    float before[60] = {};
    ReadFits(before);
    WritePicture(false);
    bool rejectedAll = true, untouched = true;
    for (int i = 0; i < 5; ++i)
    {
        p = Dispatch();
        float now[60] = {};
        ReadFits(now);
        rejectedAll = rejectedAll && p.reason != 0 && p.streak == (float) (i + 1);
        untouched = untouched && std::memcmp(now, before, 16 * sizeof(float)) == 0;
    }
    Check("4 no one map, 5 Presents (all rejected, the start untouched)", rejectedAll && untouched, p);
    WritePicture(true);
    p = Dispatch();
    Check("4 the picture good again (accepted at once)", found(p) && p.streak == 0.0f, p);

    // 5: a start far beyond anything a game leaves behind. (Meant to make the SOLVE fail, the game's "solve failed",
    // 300 of 600 fits for 13 s: it does not here, the rejection is "too few within 1 px" again. That kind of the
    // game's is NOT reproduced offline; the way out is the same count of rejections.)
    StartOf(fits, 3000.0f);
    WriteFits(fits);
    p = Dispatch();
    Check("5 a start 3000 px off, Present 1 (rejected)", p.reason != 0 && p.streak == 1.0f, p);
    p = Dispatch();
    Check("5 a start 3000 px off, Present 2 (rejected)", p.reason != 0 && p.streak == 2.0f, p);
    p = Dispatch();
    Check("5 a start 3000 px off, Present 3 (the start given up: accepted)", found(p) && p.streak == 0.0f, p);

    // 6: the wide fit against the fit in one group (SameFits says what "the same" is).
    bool same = true;
    std::memset(fits, 0, sizeof(fits));
    same = SameFits(fits, "no start") && same;
    StartOf(fits, 5.0f);
    same = SameFits(fits, "a start 5 px off") && same;
    StartOf(fits, 205.0f);
    same = SameFits(fits, "a stale start, rejected") && same;
    fits[16] = 3.0f;
    fits[17] = 2.0f;
    same = SameFits(fits, "a stale start, given up") && same;
    StartOf(fits, 0.0f);
    WritePicture(false);
    same = SameFits(fits, "no one map") && same;
    WritePicture(true);
    CheckPlain("6 the wide fit and the fit in one group: what the game keeps is the same bits", same);

    // 7: the cost. A good start on the good picture, so every fit is the ordinary accepted one.
    StartOf(fits, 0.0f);
    WriteFits(fits);
    TimeFits(200, true); // the card awake, the shaders run once
    TimeFits(200, g_oneGroupInPlace);
    const double nothing = MedianTime(0, false);
    const double wide100 = MedianTime(100, g_oneGroupInPlace), wide200 = MedianTime(200, g_oneGroupInPlace);
    const double one100 = MedianTime(100, true), one200 = MedianTime(200, true);
    std::printf("7 GPU time by timestamps (medians of 7): nothing %.4f ms; wide 100 fits %.3f ms, 200 fits %.3f ms = "
                "%.4f ms a fit; one group 100 fits %.3f ms, 200 fits %.3f ms = %.4f ms a fit\n",
                nothing, wide100, wide200, wide200 / 200.0, one100, one200, one200 / 200.0);
    const bool timed = nothing >= 0.0 && wide100 > 0.0 && wide200 > 0.0 && one100 > 0.0 && one200 > 0.0;
    const double wideRatio = timed ? wide200 / wide100 : 0.0, oneRatio = timed ? one200 / one100 : 0.0;
    std::printf("7 twice the fits take %.2f times as long (wide) and %.2f times (one group); a wide fit costs %.0f%% of "
                "a fit in one group\n",
                wideRatio, oneRatio, timed ? 100.0 * wide200 / one200 : 0.0);
    CheckPlain("7 the timer follows the count of fits (1.6 to 2.4 for twice the fits, both kinds)",
               timed && wideRatio > 1.6 && wideRatio < 2.4 && oneRatio > 1.6 && oneRatio < 2.4);
    CheckPlain("7 a wide fit costs under half of a fit in one group", timed && wide200 < 0.5 * one200);

    // 8: no depth at all.
    StartOf(fits, 0.0f);
    WriteFits(fits);
    p = Dispatch(); // an accepted fit first, so the slot holds what a game would hold
    float held[60] = {};
    ReadFits(held);
    const std::vector<float> noDepth(kSize * kSize, 0.0f);
    g_context->UpdateSubresource(g_depth, 0, nullptr, noDepth.data(), kSize * 4, 0);
    RunFit(true);
    float oneGroup[60] = {};
    ReadFits(oneGroup);
    WriteFits(held);
    p = Dispatch();
    float after[60] = {};
    ReadFits(after);
    std::printf("8 the fit in one group names this picture: reason %.0f (1 = the solve failed)\n", oneGroup[2 * 20 + 16]);
    Check("8 no depth (rejected as too few samples with depth = reason 2, the start untouched)",
          p.reason == 2 && p.streak == 1.0f && std::memcmp(after, held, 16 * sizeof(float)) == 0, p);
    g_context->UpdateSubresource(g_depth, 0, nullptr, g_depthData.data(), kSize * 4, 0);
    p = Dispatch();
    Check("8 the depth back (accepted at once)", found(p) && p.streak == 0.0f, p);

    std::printf("%s: %d checks, %d failures\n", g_failures == 0 ? "PASSED" : "FAILED", g_checked, g_failures);
    return g_failures == 0 ? 0 : 1;
}
