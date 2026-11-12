// Offline test of the D3D12 same-eye FIT (OptiScaler/inputs/PerEyeFitShader_Dx12.h), the very shader PerEyeDlss_Dx12
// compiles, bound by its root signature and run by its sequence of dispatches (INIT with Round 100 in one group, then
// rounds 0-4 in 64 groups each, UAV barriers between), on a real D3D12 device with the made-up picture of fit_test:
// one projective map between the eyes, noise of 0.05 px, motion in R16G16_FLOAT over an MV scale of -size, and a
// quarter of the picture moving by itself. A "Present" is one fit for the same eye.
//   fit12_test
//   1  no start at all: accepted, the map found, nothing given up
//   2  a start 5 px off: accepted at once (the ordinary case must not change)
//   3  a STALE start, 205 px off: rejected twice with the start kept, the third Present gives the start up and is
//      accepted (the counter's flag set), the fourth starts from the new fit
//   4  a start 45 px off (just past the 40 px within which a start's samples count at all; what fast motion leaves
//      behind, Cyberpunk 2077 sprinting and driving): the same way out on the third Present
//   5  a picture without any one map over a good start: rejected every time, the start untouched (giving up must not
//      replace it); the picture good again: accepted at once
//   6  a start 3000 px off: accepted on the third Present
//   7  LINEAR depth (d = 37 / hardware z, 336 .. 18500: centimetre-like, as Hogwarts Legacy hands RR with
//      DLSS.Use.HW.Depth = 0) with LinearDepth = 1: accepted from no start, the map found (its z column x3.7: the
//      shaders take 10/d)
//   8  the whole chain, hardware depth: a fit for each eye (two different maps), then COMPOSE: the same-eye motion
//      on static pixels within 0.2 px of the true chained maps
//   9  the same chain on linear depth with LinearDepth = 1
//   10 a FLICK (Star Wars Outlaws: a flick of the stick, the picture shimmers and settles): a good start,
//      then the whole picture 60 px to the side: accepted in the SAME Present (restarted within the fit), the next
//      Present from the new fit without a restart, and the flick's end (the picture back) accepted at once again
//   11 the flick the other way: the quarter that moves by itself lands ON the old start (a quarter of the samples
//      within 40 px of it, as a body that turns with the camera): restarted all the same, the world's map found
//   12 THE CABIN (a game driven in first person): a cabin NEAR (hardware z 0.15 .. 0.25 against
//      the world's 0.002 .. 0.11) and following its own map, which is also the start (the eye's fit at a stop). It
//      used to be that the start's fit held the cabin (49% within 1 px, rejected), and from no motion the fit bent into
//      a RAMP that half-fits both (x' = x + a y; 17-39% within 1 px), rejected for as long as the drive (in the game:
//      40-50% within 1 px in the car, 10-25% on the bike); nine rounds and 1/depth weights did not help. Now the fit
//      leaves out what lies nearer than a GAP in its depth histogram (UpscalerFitLeaveOutNearest; in the game: the
//      car at hardware z 0.125-0.7, 69-78% of the samples, the world below 0.044, three empty bins between), restarts
//      when at most half of the rest lie within 8 px of the start, and steps 128, 8, 2 px after a restart: the
//      world's map at the FIRST Present, held from then on. The cases' world lies at 0.35 x the base depth here
//      (0.0007 .. 0.039) and the cabin at 0.15 .. 0.25: a gap as the game's.
//      12  a turn: the world 15 px to the side of a cabin without the eyes' parallax (the first model), cabin 45%
//      12b the same, cabin 55%
//      12c going STRAIGHT: the world expands with its nearness (the far world barely moves, as the cabin; only depth
//          tells the two apart), cabin = start = kTrue
//      12d a cabin of 62% over a wider depth (0.12 .. 0.5: four bins)
// Cases 3, 4 and 6 run with the restart off (KeepStale bit 1): the older way out must go on working behind it.
// Controls, each must FAIL:
//   FIT12_TEST_OLD=1           both switches off (KeepStale = 3, the start as it was): cases 3, 4, 6, 10 and 11 fail
//   FIT12_TEST_NO_RESTART=1    the restart off (KeepStale = 2, the fit as it was before the restart): cases 10 and 11 fail
//                              (rejected, as the game's log: solve failed)
//   FIT12_TEST_LINEAR_OFF=1    linear depth handed with LinearDepth = 0 (as Hogwarts once had it: 22873 fits, none
//                              accepted): cases 7 and 9 fail
//   FIT12_TEST_NEAR_KEPT=1     nothing left out at a gap (KeepStale bit 2, the fit as it was before the gap): cases
//                              12-12d fail
//   FIT12_TEST_OLD_GAP=1       the depth gap by the older valley rule (a run of bins each under 0.5%; KeepStale bit 4);
//                              no case here tells it apart, so it is no control any more
//   FIT12_TEST_NO_DUMP=1       the dump's bit not set: case 13 fails (the samples are not where the DLL copies them)
//   FIT12_TEST_SHIFT_OFF=1     case 15's second pass shifts by nothing: the box reads wrong, case 15 fails
//   FIT12_TEST_AGE_OFF=1       case 16's move carries nothing along: the ages read wrong, case 16 fails
//   FIT12_TEST_NO_SNAP=1       case 17 judges the filter WITHOUT the snap (the gaze eased in, as it used to be): it
//                              needs more than 2 polls, case 17 fails
// 17 THE GAZE FILTER'S SACCADE (GazeFilter.h): polls at 90 Hz, a 20 deg saccade over four polls; with the
//    snap the smoothed gaze is within 0.1 deg of the landing point by the second poll there (the one that confirms
//    the jump); without, later; a single-sample glitch of 5 deg moves neither.
// 16 THE CROP'S AGE (kAgeShader, PerEyeDlss_Dx12::UpdateAge): 32x32, a reset (0 everywhere), three frames
//    in place (3), a move by (5, -2): 4 where the pixel's old place p + (5, -2) lay in the box, 0 where it came in.
// 15 THE CROP'S MOVE THROUGH SUBRECTS (kShiftShader, PerEyeDlss_Dx12::ShiftComposed): the box (8, 8) 16x16
//    of a 64x64 vector texture holding (x, y), shifted by (3, -2) into another texture that first took a plain copy;
//    every texel must read (x + 3, y - 2) in the box and (x, y) outside, exactly (half floats, integers).
// 13 THE DUMP (KeepStale bit 3, the overlay's 'Save the fit's samples'): the measuring round writes the 4096 samples it
//    read (raw depth, raw motion x / y, fitted or not) and the fit's stamp into the partials buffer past the counter;
//    each must equal the texel uploaded, bit for bit.
#include <windows.h>
#include <d3d12.h>
#include <d3dcompiler.h>

#include "../../../OptiScaler/inputs/PerEyeFitShader_Dx12.h"
#include "../../../OptiScaler/inputs/FoveaShader_Dx12.h" // kAgeShader (case 16)
#include "../../../OptiScaler/inputs/GazeFilter.h"       // the gaze filter (case 17)

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace
{
constexpr UINT kSize = 1024;
constexpr float kKZ = 100.0f;
constexpr UINT kFitFloats = 20 * 4; // 20 float4: slots 0-3, 5 rows each

struct Map
{
    float r0[4], r1[4], r2[3];
};
const Map kTrue = { { 1.0f, 0.0003f, -0.008f, -0.0004f }, { -0.0003f, 1.0f, 0.0f, 0.0002f }, { 0.0001f, -0.0002f, 0.0f } };
// The other eye's map for the chain (cases 8, 9): parallax the other way, a little turn.
const Map kOther = { { 1.0f, 0.0002f, 0.006f, 0.0005f }, { -0.0002f, 1.0f, 0.0f, -0.0001f }, { -0.0001f, 0.0001f, 0.0f } };
constexpr float kLinearK = 37.0f; // linear depth = kLinearK / hardware z
// kTrue after a flick of the stick: the whole picture 60 px to the side and 20 px up or down (cases 10, 11).
constexpr float kPx = 2.0f / 1024.0f;
const Map kFlick = { { 1.0f, 0.0003f, -0.008f, -0.0004f + 60.0f * kPx }, { -0.0003f, 1.0f, 0.0f, 0.0002f + 20.0f * kPx },
                     { 0.0001f, -0.0002f, 0.0f } };
// The other way, by what the moving quarter moves by itself (+60, -25 px in the motion's own signs): after it the
// quarter's vectors are those of kTrue's static picture, i.e. exactly on the old start.
const Map kFlickBack = { { 1.0f, 0.0003f, -0.008f, -0.0004f - 60.0f * kPx }, { -0.0003f, 1.0f, 0.0f, 0.0002f - 25.0f * kPx },
                         { 0.0001f, -0.0002f, 0.0f } };

uint32_t g_random = 0x2468ace1u;
float Random()
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
    // ADD, not OR: a mantissa that rounds up carries into the exponent (the OR halved such values: 3 of 15111
    // pixels of the chain read as objects moving by themselves).
    return (uint16_t) (sign | (((uint32_t) exponent << 10) + ((mantissa + 0x1000u) >> 13)));
}

ID3D12Device* g_device = nullptr;
ID3D12CommandQueue* g_queue = nullptr;
ID3D12CommandAllocator* g_alloc = nullptr;
ID3D12GraphicsCommandList* g_list = nullptr;
ID3D12Fence* g_fence = nullptr;
UINT64 g_fenceValue = 0;
HANDLE g_event = nullptr;
ID3D12RootSignature* g_root = nullptr;
ID3D12PipelineState* g_pso = nullptr;
ID3D12PipelineState* g_composePso = nullptr;
ID3D12Resource* g_out = nullptr;      // the compose's output, R16G16_FLOAT, kSize x kSize
ID3D12Resource* g_mask = nullptr;     // the compose's bias mask, R8_UNORM, kSize x kSize (written, not read here)
ID3D12Resource* g_outRead = nullptr;  // its readback
uint32_t g_linear = 0;                // the LinearDepth constant handed to the shaders
bool g_depthLinear = false;           // what the depth texture holds now
const Map* g_fitMap = &kTrue;         // the map the fit is checked against
ID3D12DescriptorHeap* g_heap = nullptr;
ID3D12Resource* g_depth = nullptr;
ID3D12Resource* g_motion = nullptr;
ID3D12Resource* g_fits = nullptr;
ID3D12Resource* g_partials = nullptr;
ID3D12Resource* g_upload = nullptr;   // kSize * kSize * 4 bytes, persistently mapped
ID3D12Resource* g_readback = nullptr; // 320 bytes
ID3D12Resource* g_dumpRead = nullptr; // the dump's part of the partials buffer
constexpr UINT kDumpWords = 64 * 80 + 4 + 4096 * 4; // the partials buffer as the DLL makes it since the dump
std::vector<uint16_t> g_lastMotion;  // the motion texture as last uploaded (half pairs)
uint8_t* g_uploadData = nullptr;
uint32_t g_keepStale = 0;
float g_mvScale[2] = { -(float) kSize, -(float) kSize }; // the MV scale handed to the shaders
uint32_t g_stamp = 0;
std::vector<float> g_depthData(kSize * kSize);

void Barrier(ID3D12Resource* r, D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after)
{
    D3D12_RESOURCE_BARRIER b {};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = r;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    b.Transition.StateBefore = before;
    b.Transition.StateAfter = after;
    g_list->ResourceBarrier(1, &b);
}

bool Submit()
{
    if (FAILED(g_list->Close()))
        return false;
    ID3D12CommandList* lists[] = { g_list };
    g_queue->ExecuteCommandLists(1, lists);
    g_queue->Signal(g_fence, ++g_fenceValue);
    if (g_fence->GetCompletedValue() < g_fenceValue)
    {
        g_fence->SetEventOnCompletion(g_fenceValue, g_event);
        WaitForSingleObject(g_event, 10000);
    }
    if (g_device->GetDeviceRemovedReason() != S_OK)
        return false;
    g_alloc->Reset();
    g_list->Reset(g_alloc, nullptr);
    return true;
}

// One texture's content through the upload buffer (both textures are 4 bytes a pixel: rows of 4096 bytes, aligned).
void UploadTexture(ID3D12Resource* tex, DXGI_FORMAT format, const void* data)
{
    std::memcpy(g_uploadData, data, kSize * kSize * 4);
    Barrier(tex, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST);
    D3D12_TEXTURE_COPY_LOCATION dst {}, src {};
    dst.pResource = tex;
    dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    src.pResource = g_upload;
    src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    src.PlacedFootprint.Footprint = { format, kSize, kSize, 1, kSize * 4 };
    g_list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    Barrier(tex, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    Submit();
}

// Depth as hardware z (0.002 .. 0.11) or as linear depth kLinearK / z.
void WriteDepth(bool linear)
{
    std::vector<float> d(g_depthData);
    if (linear)
        for (auto& v : d)
            v = kLinearK / v;
    UploadTexture(g_depth, DXGI_FORMAT_R32_FLOAT, d.data());
    g_depthLinear = linear;
}

void WritePicture(bool oneMap, const Map& kTrue = ::kTrue)
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
            motion[(py * kSize + px) * 2 + 0] = Half(mx / -(float) kSize);
            motion[(py * kSize + px) * 2 + 1] = Half(my / -(float) kSize);
        }
    }
    UploadTexture(g_motion, DXGI_FORMAT_R16G16_FLOAT, motion.data());
    g_lastMotion = motion;
}

// The cabin picture (case 12): rows from `cabinFrom` of the height follow `cabin` (the start's map), the rest `world`;
// no quarter moving by itself.
void WriteCabinPicture(const Map& world, const Map& cabin, float cabinFrom)
{
    std::vector<uint16_t> motion(kSize * kSize * 2);
    for (UINT py = 0; py < kSize; ++py)
    {
        const Map& m = py >= (UINT) (cabinFrom * kSize) ? cabin : world;
        for (UINT px = 0; px < kSize; ++px)
        {
            const float u = ((float) px + 0.5f) / kSize, v = ((float) py + 0.5f) / kSize;
            const float x = u * 2.0f - 1.0f, y = 1.0f - v * 2.0f;
            const float zz = g_depthData[py * kSize + px] * kKZ;
            const float w = m.r2[0] * x + m.r2[1] * y + m.r2[2] * zz + 1.0f;
            const float xp = (m.r0[0] * x + m.r0[1] * y + m.r0[2] * zz + m.r0[3]) / w;
            const float yp = (m.r1[0] * x + m.r1[1] * y + m.r1[2] * zz + m.r1[3]) / w;
            const float mx = (xp - x) * kSize * 0.5f + (Random() - 0.5f) * 0.1f;
            const float my = -(yp - y) * kSize * 0.5f + (Random() - 0.5f) * 0.1f;
            motion[(py * kSize + px) * 2 + 0] = Half(mx / -(float) kSize);
            motion[(py * kSize + px) * 2 + 1] = Half(my / -(float) kSize);
        }
    }
    UploadTexture(g_motion, DXGI_FORMAT_R16G16_FLOAT, motion.data());
}

void WriteFits(const float* fits)
{
    std::memcpy(g_uploadData, fits, kFitFloats * 4);
    Barrier(g_fits, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST);
    g_list->CopyBufferRegion(g_fits, 0, g_upload, 0, kFitFloats * 4);
    Barrier(g_fits, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    Submit();
}

bool ReadFits(float* fits)
{
    Barrier(g_fits, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
    g_list->CopyBufferRegion(g_readback, 0, g_fits, 0, kFitFloats * 4);
    Barrier(g_fits, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    if (!Submit())
        return false;
    void* mapped = nullptr;
    D3D12_RANGE range { 0, kFitFloats * 4 };
    if (FAILED(g_readback->Map(0, &range, &mapped)))
        return false;
    std::memcpy(fits, mapped, kFitFloats * 4);
    D3D12_RANGE none { 0, 0 };
    g_readback->Unmap(0, &none);
    return true;
}

void StartOf(float* fits, float pixelsOff)
{
    std::memset(fits, 0, kFitFloats * sizeof(float));
    std::memcpy(fits + 0, kTrue.r0, sizeof(kTrue.r0));
    std::memcpy(fits + 4, kTrue.r1, sizeof(kTrue.r1));
    std::memcpy(fits + 8, kTrue.r2, sizeof(kTrue.r2));
    fits[3] += pixelsOff * 2.0f / kSize;
    fits[11] = 1.0f;
    fits[12] = 0.05f;
    fits[13] = 0.9f;
    fits[14] = 1.0f; // accepted
    fits[15] = 4096.0f;
}

// One fit, recorded as PerEyeDlss_Dx12.cpp (SameEye) records it; with `compose`, the compose after it as well.
void RunFit(uint32_t cur = 0, bool compose = false)
{
    struct
    {
        float mvScale[2];
        uint32_t cur, prev;
        float knee0, knee1, gain;
        uint32_t radius, stamp, round, keepStale, linearDepth;
    } constants { { g_mvScale[0], g_mvScale[1] }, cur, cur ^ 1u, 0.02f, 0.1f, 2.0f, 2, ++g_stamp, 100, g_keepStale,
                  g_linear };
    static_assert(sizeof(constants) == 48);
    ID3D12DescriptorHeap* heaps[] = { g_heap };
    g_list->SetDescriptorHeaps(1, heaps);
    g_list->SetComputeRootSignature(g_root);
    g_list->SetComputeRoot32BitConstants(0, 12, &constants, 0);
    g_list->SetComputeRootDescriptorTable(1, g_heap->GetGPUDescriptorHandleForHeapStart());
    g_list->SetComputeRootUnorderedAccessView(2, g_fits->GetGPUVirtualAddress());
    g_list->SetComputeRootShaderResourceView(3, g_fits->GetGPUVirtualAddress());
    g_list->SetComputeRootUnorderedAccessView(4, g_partials->GetGPUVirtualAddress());
    D3D12_RESOURCE_BARRIER uavs[2] {};
    uavs[0].Type = uavs[1].Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    uavs[0].UAV.pResource = g_fits;
    uavs[1].UAV.pResource = g_partials;
    g_list->SetPipelineState(g_pso);
    g_list->Dispatch(1, 1, 1);
    g_list->ResourceBarrier(2, uavs);
    for (uint32_t round = 0; round <= 4; ++round)
    {
        g_list->SetComputeRoot32BitConstant(0, round, 9);
        g_list->Dispatch(64, 1, 1);
        g_list->ResourceBarrier(2, uavs);
    }
    if (compose)
    {
        // As the DLL: the fits buffer to a shader resource, the compose over the motion's size, back to UAV.
        Barrier(g_fits, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        g_list->SetPipelineState(g_composePso);
        g_list->Dispatch((kSize + 7) / 8, (kSize + 7) / 8, 1);
        Barrier(g_fits, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    }
    Submit();
}

float FromHalf(uint16_t h)
{
    const uint32_t sign = (uint32_t) (h & 0x8000u) << 16;
    const int exponent = (h >> 10) & 0x1f;
    const uint32_t mantissa = h & 0x3ffu;
    uint32_t bits;
    if (exponent == 0)
    {
        const float v = (float) mantissa / 1024.0f / 16384.0f;
        return (h & 0x8000u) ? -v : v;
    }
    if (exponent == 31)
        bits = sign | 0x7f800000u | (mantissa << 13);
    else
        bits = sign | ((uint32_t) (exponent - 15 + 127) << 23) | (mantissa << 13);
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

// The compose's output in px (it writes result / MvScale), read back.
bool ReadOut(std::vector<float>& px)
{
    Barrier(g_out, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
    D3D12_TEXTURE_COPY_LOCATION dst {}, src {};
    src.pResource = g_out;
    src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    dst.pResource = g_outRead;
    dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    dst.PlacedFootprint.Footprint = { DXGI_FORMAT_R16G16_FLOAT, kSize, kSize, 1, kSize * 4 };
    g_list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    Barrier(g_out, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    if (!Submit())
        return false;
    void* mapped = nullptr;
    if (FAILED(g_outRead->Map(0, nullptr, &mapped)))
        return false;
    const uint16_t* h = (const uint16_t*) mapped;
    px.resize(kSize * kSize * 2);
    for (size_t i = 0; i < px.size(); ++i)
        px[i] = FromHalf(h[i]) * -(float) kSize;
    D3D12_RANGE none { 0, 0 };
    g_outRead->Unmap(0, &none);
    return true;
}

// Where a map takes clip (x, y) at hardware z (the shader's Apply: z scaled by 100, w from row 3).
void ApplyTrue(const Map& m, float x, float y, float z, float& ox, float& oy, float& oz)
{
    const float zz = z * kKZ;
    const float w = m.r2[0] * x + m.r2[1] * y + m.r2[2] * zz + 1.0f;
    ox = (m.r0[0] * x + m.r0[1] * y + m.r0[2] * zz + m.r0[3]) / w;
    oy = (m.r1[0] * x + m.r1[1] * y + m.r1[2] * zz + m.r1[3]) / w;
    oz = z / w;
}

// Cases 8, 9: the other eye's fit into slot 1 (its picture: kOther), this eye's into slot 0 (kTrue), then the
// compose. The same-eye motion of a static pixel is kTrue then kOther; checked away from the moving block (its own
// motion is added on purpose) and from the edges. Returns the worst error in px over the checked pixels.
char g_worstAt[200] = {};
int g_bad = 0, g_checkedPx = 0;
float ChainError(bool& fitsOk)
{
    g_bad = g_checkedPx = 0;
    float zero[kFitFloats] = {};
    WriteFits(zero);
    WritePicture(true, kOther);
    RunFit(1);
    float f[kFitFloats] = {};
    ReadFits(f);
    fitsOk = f[1 * 20 + 14] > 0.5f;
    WritePicture(true, kTrue);
    RunFit(0, true);
    ReadFits(f);
    fitsOk = fitsOk && f[0 * 20 + 14] > 0.5f;
    std::vector<float> out;
    if (!ReadOut(out))
        return 1e9f;
    float worst = 0;
    for (UINT py = 8; py < kSize - 8; py += 7)
        for (UINT px = 8; px < kSize - 8; px += 7)
        {
            if (px >= kSize / 4 - 8 && px < kSize * 3 / 4 + 8 && py >= kSize / 4 - 8 && py < kSize * 3 / 4 + 8)
                continue;
            const float x = (px + 0.5f) / kSize * 2.0f - 1.0f, y = 1.0f - (py + 0.5f) / kSize * 2.0f;
            float x1, y1, z1, x2, y2, z2;
            ApplyTrue(kTrue, x, y, g_depthData[py * kSize + px], x1, y1, z1);
            ApplyTrue(kOther, x1, y1, z1, x2, y2, z2);
            const float ex = (x2 - x) * kSize * 0.5f, ey = -(y2 - y) * kSize * 0.5f;
            const float* o = &out[(py * kSize + px) * 2];
            const float e = std::hypot(o[0] - ex, o[1] - ey);
            if (e > 1.0f)
            {
                ++g_bad;
                if (g_bad <= 6)
                {
                    float zmax = 0, zmin = 1;
                    for (int dy = -2; dy <= 2; ++dy)
                        for (int dx = -2; dx <= 2; ++dx)
                        {
                            const float zq = g_depthData[(py + dy) * kSize + px + dx];
                            zmax = (std::max)(zmax, zq);
                            zmin = (std::min)(zmin, zq);
                        }
                    std::printf("   bad (%u, %u): compose (%.2f, %.2f) true (%.2f, %.2f); z %.4f, 5x5 z [%.4f, %.4f]\n", px,
                                py, o[0], o[1], ex, ey, g_depthData[py * kSize + px], zmin, zmax);
                }
            }
            ++g_checkedPx;
            if (e > worst)
            {
                worst = e;
                std::snprintf(g_worstAt, sizeof(g_worstAt), "(%u, %u): compose (%.3f, %.3f), true (%.3f, %.3f), depth %.4f",
                              px, py, o[0], o[1], ex, ey, g_depthData[py * kSize + px]);
            }
            if (py == 8 + 7 * 10 && (px == 8 || px == 8 + 7 * 140))
            {
                float gx, gy, gz;
                ApplyTrue(kTrue, x, y, g_depthData[py * kSize + px], gx, gy, gz);
                std::printf("   pixel (%u, %u): compose (%.3f, %.3f); true same-eye (%.3f, %.3f); the game's other-eye "
                            "vector (%.3f, %.3f)\n",
                            px, py, o[0], o[1], ex, ey, (gx - x) * kSize * 0.5f, -(gy - y) * kSize * 0.5f);
            }
        }
    return worst;
}

struct Present
{
    int reason = -1;
    float streak = -1, residual = 99, inliers = 0, mapError = 99, gaveUp = -1, restarted = -1, leftOut = -1;
    float slotXConstant = 0;
};

Present Dispatch()
{
    RunFit();
    float f[kFitFloats] = {};
    Present p;
    if (!ReadFits(f))
        return p;
    p.reason = (int) f[2 * 20 + 16];
    p.gaveUp = f[2 * 20 + 17];
    // Row 4's z: restarted within the fit (0 / 1) + half the share of the samples left out as the nearest third.
    p.restarted = f[2 * 20 + 18] > 0.5f ? 1.0f : 0.0f;
    p.leftOut = (f[2 * 20 + 18] - p.restarted) / 0.45f;
    p.streak = f[0 * 20 + 17];
    p.residual = f[2 * 20 + 12];
    p.inliers = f[2 * 20 + 13];
    p.slotXConstant = f[3];
    float worst = 0;
    // On linear depth the fit sees 10/d = (10 / kLinearK) z: its z column comes out kLinearK / 10 times larger.
    const float zs = g_depthLinear ? kLinearK / 10.0f : 1.0f;
    const Map& t = *g_fitMap;
    const float want[11] = { t.r0[0], t.r0[1], t.r0[2] * zs, t.r0[3], t.r1[0], t.r1[1],
                             t.r1[2] * zs, t.r1[3], t.r2[0], t.r2[1], t.r2[2] * zs };
    for (int i = 0; i < 11; ++i)
        worst = (std::max)(worst, std::fabs(f[i] - want[i]));
    p.mapError = worst;
    return p;
}

int g_failures = 0, g_checked = 0;
void Check(const char* what, bool ok, const Present& p)
{
    std::printf("%s: reason %d (0 = accepted), gave up the start %.0f, restarted %.0f, %.0f%% left out at a depth gap, "
                "rejections in a row %.0f, residual %.3f px, within 1 px %.0f%%, the eye's map off the true one by %.5f "
                "at the most -> %s\n",
                what, p.reason, p.gaveUp, p.restarted, p.leftOut * 100.0f, p.streak, p.residual, p.inliers * 100.0f,
                p.mapError,
                ok ? "OK" : "FAIL");
    ++g_checked;
    if (!ok)
        ++g_failures;
}

bool found(const Present& p) { return p.reason == 0 && p.residual < 0.2f && p.mapError < 0.002f; }
// A fit that restarted far from no motion has three rounds, the last at 0.5 px: accepted and the map found, a little
// coarser (0.16 - 0.22 px here against 0.04; the next Present starts from it and is as fine as ever, case 10).
bool foundCoarse(const Present& p) { return p.reason == 0 && p.residual < 0.3f && p.mapError < 0.002f; }

// A stale start `off` px away: rejected twice with the start kept, given up and accepted on the third Present.
// With the restart off (the caller sets KeepStale's bit 1): the older way out behind the restart.
void StaleCase(const char* name, float off)
{
    float fits[kFitFloats];
    StartOf(fits, off);
    WriteFits(fits);
    const float staleConstant = fits[3];
    char what[160];
    Present p = Dispatch();
    std::snprintf(what, sizeof(what), "%s, Present 1 (rejected, the start kept)", name);
    Check(what, p.reason != 0 && p.streak == 1.0f && p.slotXConstant == staleConstant && p.gaveUp == 0.0f, p);
    p = Dispatch();
    std::snprintf(what, sizeof(what), "%s, Present 2 (rejected, the start kept)", name);
    Check(what, p.reason != 0 && p.streak == 2.0f && p.slotXConstant == staleConstant && p.gaveUp == 0.0f, p);
    p = Dispatch();
    std::snprintf(what, sizeof(what), "%s, Present 3 (the start given up: accepted, counted)", name);
    Check(what, found(p) && p.streak == 0.0f && p.gaveUp == 1.0f, p);
    p = Dispatch();
    std::snprintf(what, sizeof(what), "%s, Present 4 (from the new start)", name);
    Check(what, found(p) && p.streak == 0.0f && p.gaveUp == 0.0f, p);
}

// The cabin (case 12): rows from `cabinFrom` of the height are a cabin, NEAR (hardware z 0.15 .. 0.25 against the
// world's 0.002 .. 0.11) and following `kCabin`, which is also the start (the eye's fit at a stop); the world follows
// `kDrive`. The world's map must be found at the first Present and held (the nearest third left out: the cabin is no
// sample of the fit at all); the control FIT12_TEST_NEAR_KEPT=1 holds the cabin, or a ramp between the two.
void CabinCase(const char* name, float cabinFrom, const Map& kDrive, const Map& kCabin, float zFrom = 0.15f,
               float zSpan = 0.1f)
{
    const std::vector<float> savedDepth = g_depthData;
    for (UINT py = 0; py < kSize; ++py)
        for (UINT px = 0; px < kSize; ++px)
            g_depthData[py * kSize + px] =
                py >= (UINT) (cabinFrom * kSize) ? zFrom + zSpan * Random() : 0.35f * g_depthData[py * kSize + px];
    UploadTexture(g_depth, DXGI_FORMAT_R32_FLOAT, g_depthData.data());
    WriteCabinPicture(kDrive, kCabin, cabinFrom);
    float fits[kFitFloats];
    StartOf(fits, 0.0f);
    std::memcpy(fits + 0, kCabin.r0, sizeof(kCabin.r0));
    std::memcpy(fits + 4, kCabin.r1, sizeof(kCabin.r1));
    std::memcpy(fits + 8, kCabin.r2, sizeof(kCabin.r2));
    WriteFits(fits);
    g_fitMap = &kDrive;
    char what[200];
    Present p = Dispatch();
    std::snprintf(what, sizeof(what), "%s, Present 1 (the world's map at once)", name);
    Check(what, foundCoarse(p) && p.streak == 0.0f, p);
    for (int i = 2; i <= 4; ++i)
    {
        p = Dispatch();
        std::snprintf(what, sizeof(what), "%s, Present %d (from the world's fit: held, no restart)", name, i);
        Check(what, found(p) && p.streak == 0.0f && p.restarted == 0.0f, p);
    }
    g_depthData = savedDepth;
    UploadTexture(g_depth, DXGI_FORMAT_R32_FLOAT, g_depthData.data());
    g_fitMap = &kTrue;
    WritePicture(true);
}

} // namespace

int main()
{
    char value[8] = {};
    g_keepStale = GetEnvironmentVariableA("FIT12_TEST_OLD", value, sizeof(value)) > 0 && value[0] == '1' ? 3u : 0u;
    if (GetEnvironmentVariableA("FIT12_TEST_NO_RESTART", value, sizeof(value)) > 0 && value[0] == '1')
        g_keepStale |= 2u;
    if (GetEnvironmentVariableA("FIT12_TEST_NEAR_KEPT", value, sizeof(value)) > 0 && value[0] == '1')
        g_keepStale |= 4u;
    if (GetEnvironmentVariableA("FIT12_TEST_OLD_GAP", value, sizeof(value)) > 0 && value[0] == '1')
        g_keepStale |= 16u;
    const uint32_t asked = g_keepStale;
    const bool linearOff =
        GetEnvironmentVariableA("FIT12_TEST_LINEAR_OFF", value, sizeof(value)) > 0 && value[0] == '1';
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
    if (FAILED(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&g_device))))
    {
        std::printf("FAILED: no D3D12 device\n");
        return 1;
    }
    D3D12_COMMAND_QUEUE_DESC qd {};
    qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    bool ok = SUCCEEDED(g_device->CreateCommandQueue(&qd, IID_PPV_ARGS(&g_queue)));
    ok = ok && SUCCEEDED(g_device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&g_alloc)));
    ok = ok && SUCCEEDED(g_device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, g_alloc, nullptr,
                                                     IID_PPV_ARGS(&g_list)));
    ok = ok && SUCCEEDED(g_device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&g_fence)));
    g_event = CreateEventW(nullptr, FALSE, FALSE, nullptr);

    // The root signature of PerEyeDlss_Dx12.cpp (SetupMotion).
    D3D12_DESCRIPTOR_RANGE ranges[3] {};
    ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    ranges[0].NumDescriptors = 2;
    ranges[0].OffsetInDescriptorsFromTableStart = 0;
    ranges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    ranges[1].NumDescriptors = 1;
    ranges[1].BaseShaderRegister = 1;
    ranges[1].OffsetInDescriptorsFromTableStart = 2;
    ranges[2].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV; // the bias mask (u3), as the DLL
    ranges[2].NumDescriptors = 1;
    ranges[2].BaseShaderRegister = 3;
    ranges[2].OffsetInDescriptorsFromTableStart = 3;
    D3D12_ROOT_PARAMETER params[5] {};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[0].Constants.Num32BitValues = 12;
    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[1].DescriptorTable.NumDescriptorRanges = 3;
    params[1].DescriptorTable.pDescriptorRanges = ranges;
    params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    params[2].Descriptor.ShaderRegister = 0;
    params[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
    params[3].Descriptor.ShaderRegister = 2;
    params[4].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    params[4].Descriptor.ShaderRegister = 2;
    D3D12_ROOT_SIGNATURE_DESC rs {};
    rs.NumParameters = 5;
    rs.pParameters = params;
    ID3DBlob* rsBlob = nullptr;
    ok = ok && SUCCEEDED(D3D12SerializeRootSignature(&rs, D3D_ROOT_SIGNATURE_VERSION_1, &rsBlob, nullptr));
    ok = ok && SUCCEEDED(g_device->CreateRootSignature(0, rsBlob->GetBufferPointer(), rsBlob->GetBufferSize(),
                                                       IID_PPV_ARGS(&g_root)));
    // Compiled as Shader_Common.cpp's CompileShader does it.
    ID3DBlob* code = nullptr;
    ID3DBlob* errors = nullptr;
    if (FAILED(D3DCompile(kFitShader, std::strlen(kFitShader), nullptr, nullptr, nullptr, "CSMain", "cs_5_0",
                          D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &code, &errors)))
    {
        std::printf("FAILED: the fit does not compile: %s\n",
                    errors != nullptr ? (const char*) errors->GetBufferPointer() : "no message");
        return 1;
    }
    D3D12_COMPUTE_PIPELINE_STATE_DESC pd {};
    pd.pRootSignature = g_root;
    pd.CS = { code->GetBufferPointer(), code->GetBufferSize() };
    ok = ok && SUCCEEDED(g_device->CreateComputePipelineState(&pd, IID_PPV_ARGS(&g_pso)));
    ID3DBlob* composeCode = nullptr;
    if (FAILED(D3DCompile(kComposeShader, std::strlen(kComposeShader), nullptr, nullptr, nullptr, "CSMain", "cs_5_0",
                          D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &composeCode, &errors)))
    {
        std::printf("FAILED: the compose does not compile: %s\n",
                    errors != nullptr ? (const char*) errors->GetBufferPointer() : "no message");
        return 1;
    }
    pd.CS = { composeCode->GetBufferPointer(), composeCode->GetBufferSize() };
    ok = ok && SUCCEEDED(g_device->CreateComputePipelineState(&pd, IID_PPV_ARGS(&g_composePso)));

    D3D12_HEAP_PROPERTIES heap {};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC td {};
    td.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    td.Width = td.Height = kSize;
    td.DepthOrArraySize = td.MipLevels = 1;
    td.SampleDesc.Count = 1;
    td.Format = DXGI_FORMAT_R32_FLOAT;
    ok = ok && SUCCEEDED(g_device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &td,
                                                           D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, nullptr,
                                                           IID_PPV_ARGS(&g_depth)));
    td.Format = DXGI_FORMAT_R16G16_FLOAT;
    ok = ok && SUCCEEDED(g_device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &td,
                                                           D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, nullptr,
                                                           IID_PPV_ARGS(&g_motion)));
    td.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS; // the compose's output, as the DLL makes it
    ok = ok && SUCCEEDED(g_device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &td,
                                                           D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
                                                           IID_PPV_ARGS(&g_out)));
    td.Format = DXGI_FORMAT_R8_UNORM;
    ok = ok && SUCCEEDED(g_device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &td,
                                                           D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
                                                           IID_PPV_ARGS(&g_mask)));
    td.Format = DXGI_FORMAT_R16G16_FLOAT;
    td.Flags = D3D12_RESOURCE_FLAG_NONE;
    D3D12_RESOURCE_DESC bd {};
    bd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    bd.Height = bd.DepthOrArraySize = bd.MipLevels = 1;
    bd.SampleDesc.Count = 1;
    bd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    bd.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    bd.Width = 20 * 16; // as the DLL makes them
    ok = ok && SUCCEEDED(g_device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &bd,
                                                           D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
                                                           IID_PPV_ARGS(&g_fits)));
    bd.Width = kDumpWords * 4;
    ok = ok && SUCCEEDED(g_device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &bd,
                                                           D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
                                                           IID_PPV_ARGS(&g_partials)));
    bd.Flags = D3D12_RESOURCE_FLAG_NONE;
    bd.Width = kSize * kSize * 4;
    heap.Type = D3D12_HEAP_TYPE_UPLOAD;
    ok = ok && SUCCEEDED(g_device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &bd,
                                                           D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                           IID_PPV_ARGS(&g_upload)));
    bd.Width = kFitFloats * 4;
    heap.Type = D3D12_HEAP_TYPE_READBACK;
    ok = ok && SUCCEEDED(g_device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &bd,
                                                           D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                           IID_PPV_ARGS(&g_readback)));
    bd.Width = kSize * kSize * 4;
    ok = ok && SUCCEEDED(g_device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &bd,
                                                           D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                           IID_PPV_ARGS(&g_outRead)));
    bd.Width = kDumpWords * 4;
    ok = ok && SUCCEEDED(g_device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &bd,
                                                           D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                           IID_PPV_ARGS(&g_dumpRead)));
    ok = ok && SUCCEEDED(g_upload->Map(0, nullptr, (void**) &g_uploadData));
    D3D12_DESCRIPTOR_HEAP_DESC hd {};
    hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    hd.NumDescriptors = 4;
    hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    ok = ok && SUCCEEDED(g_device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&g_heap)));
    if (!ok)
    {
        std::printf("FAILED: the test's device, pipeline, textures or buffers\n");
        return 1;
    }
    const UINT inc = g_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    D3D12_CPU_DESCRIPTOR_HANDLE cpu = g_heap->GetCPUDescriptorHandleForHeapStart();
    D3D12_SHADER_RESOURCE_VIEW_DESC srv {};
    srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srv.Texture2D.MipLevels = 1;
    srv.Format = DXGI_FORMAT_R32_FLOAT;
    g_device->CreateShaderResourceView(g_depth, &srv, cpu);
    cpu.ptr += inc;
    srv.Format = DXGI_FORMAT_R16G16_FLOAT;
    g_device->CreateShaderResourceView(g_motion, &srv, cpu);
    cpu.ptr += inc;
    D3D12_UNORDERED_ACCESS_VIEW_DESC uav {};
    uav.Format = DXGI_FORMAT_R16G16_FLOAT;
    uav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    g_device->CreateUnorderedAccessView(g_out, nullptr, &uav, cpu); // the compose's output (u1); the fit leaves it
    cpu.ptr += inc;
    uav.Format = DXGI_FORMAT_R8_UNORM;
    g_device->CreateUnorderedAccessView(g_mask, nullptr, &uav, cpu); // the bias mask (u3)

    wchar_t compiler[MAX_PATH] = {};
    GetModuleFileNameW(GetModuleHandleW(L"d3dcompiler_47.dll"), compiler, MAX_PATH);
    std::printf("the D3D12 fit, %s, compiled by %ls\n",
                g_keepStale == 3   ? "both switches OFF: the start as it was (KeepStale = 3, a control)"
                : g_keepStale == 2 ? "the restart OFF: the fit as it was before the restart (KeepStale = 2, a control)"
                                   : "the switches ON: a start left behind restarts the fit; a stale start given up",
                compiler);

    for (UINT py = 0; py < kSize; ++py)
        for (UINT px = 0; px < kSize; ++px)
            g_depthData[py * kSize + px] =
                0.002f + 0.05f * (0.5f + 0.5f * std::sin(px * 0.013f) * std::cos(py * 0.017f)) + 0.058f * Random();
    UploadTexture(g_depth, DXGI_FORMAT_R32_FLOAT, g_depthData.data());
    WritePicture(true);

    float fits[kFitFloats];
    // 1: no start at all.
    std::memset(fits, 0, sizeof(fits));
    WriteFits(fits);
    Present p = Dispatch();
    Check("1 no start", found(p) && p.streak == 0.0f && p.gaveUp == 0.0f, p);

    // 2: a start 5 px off.
    StartOf(fits, 5.0f);
    WriteFits(fits);
    p = Dispatch();
    Check("2 a start 5 px off (no restart)", found(p) && p.streak == 0.0f && p.gaveUp == 0.0f && p.restarted == 0.0f, p);

    // 10: a flick of the stick, from a good start; then its end.
    StartOf(fits, 0.0f);
    WriteFits(fits);
    g_fitMap = &kFlick;
    WritePicture(true, kFlick);
    p = Dispatch();
    Check("10 a flick 60 px to the side, Present 1 (restarted within the fit: accepted)",
          foundCoarse(p) && p.streak == 0.0f && p.restarted == 1.0f && p.gaveUp == 0.0f, p);
    p = Dispatch();
    Check("10 the turn goes on, Present 2 (from the new fit, no restart)",
          found(p) && p.streak == 0.0f && p.restarted == 0.0f, p);
    g_fitMap = &kTrue;
    WritePicture(true);
    p = Dispatch();
    Check("10 the flick's end, Present 3 (restarted again: accepted)",
          found(p) && p.streak == 0.0f && p.restarted == 1.0f, p);

    // 11: the flick the other way: the moving quarter lands on the old start.
    g_fitMap = &kFlickBack;
    WritePicture(true, kFlickBack);
    p = Dispatch();
    Check("11 a flick that leaves the moving quarter ON the old start (restarted: the world's map, accepted)",
          foundCoarse(p) && p.streak == 0.0f && p.restarted == 1.0f, p);
    p = Dispatch();
    Check("11 the Present after it (from the new fit, no restart, as fine as ever)",
          found(p) && p.streak == 0.0f && p.restarted == 0.0f, p);
    g_fitMap = &kTrue;
    WritePicture(true);

    // 12: the cabin (see the top). A turn: the world 15 px to the side of a cabin that turns with the head only (kTrue
    // without its parallax terms), the cabin 45% and then 55% of the picture (the game's rejected fits: 40-50% within 1 px).
    // 12c going STRAIGHT: the world expands from the middle with its nearness (row 3's z term: 20 px at the nearest
    // world, half way out; the far world barely moves, as the cabin), the cabin and the start = kTrue.
    {
        const Map kTurn = { { 1.0f, 0.0003f, -0.008f, -0.0004f + 15.0f * kPx }, { -0.0003f, 1.0f, 0.0f, 0.0002f },
                            { 0.0001f, -0.0002f, 0.0f } };
        const Map kTurnCabin = { { 1.0f, 0.0003f, 0.0f, -0.0004f }, { -0.0003f, 1.0f, 0.0f, 0.0002f },
                                 { 0.0001f, -0.0002f, 0.0f } };
        const Map kStraight = { { 1.0f, 0.0003f, -0.008f, -0.0004f }, { -0.0003f, 1.0f, 0.0f, 0.0002f },
                                { 0.0001f, -0.0002f, -0.0071f } };
        CabinCase("12 a turn, the cabin 45% of the picture", 0.55f, kTurn, kTurnCabin);
        CabinCase("12b a turn, the cabin 55% of the picture", 0.45f, kTurn, kTurnCabin);
        CabinCase("12c going straight, the cabin 45% of the picture", 0.55f, kStraight, kTrue);
        // 12d: a cabin of 62% over a wider depth (0.12 .. 0.5: four bins): the cut stops at half, and the part of
        // the cabin left in holds a start on it unless the 8 px gate restarts the fit. The cabin here has the eyes'
        // parallax (kTrue: under alternate eyes the swap moves the cabin by its depth's parallax as everything else,
        // only the car's motion is missing from it); with kTurnCabin, whose parallax-free map crosses the world's
        // mid depths within 8 px, a cabin over 60% is NOT solved (17% within 1 px, rejected).
        // With the earlier half cut ~9% of this cabin stayed in and bent the fit into a ramp (35% within 1 px);
        // the gap takes the whole cabin, however large.
        CabinCase("12d a turn, the cabin 62% of the picture over a wider depth", 0.38f, kTurn, kTrue, 0.12f, 0.38f);
    }

    // 3, 4: stale starts, with the restart off (the older way out behind it).
    g_keepStale = asked | 2u;
    StaleCase("3 a STALE start 205 px off", 205.0f);
    StaleCase("4 a start 45 px off (fast motion)", 45.0f);
    g_keepStale = asked;

    // 5: no one map, over the good start case 4 left: giving up must never replace the start.
    float before[kFitFloats] = {};
    ReadFits(before);
    WritePicture(false);
    bool rejectedAll = true, untouched = true;
    for (int i = 0; i < 6; ++i)
    {
        p = Dispatch();
        float now[kFitFloats] = {};
        ReadFits(now);
        rejectedAll = rejectedAll && p.reason != 0 && p.streak == (float) (i + 1);
        untouched = untouched && std::memcmp(now, before, 16 * sizeof(float)) == 0;
    }
    Check("5 no one map, 6 Presents (all rejected, the start untouched, two of them gave it up)",
          rejectedAll && untouched, p);
    WritePicture(true);
    p = Dispatch();
    Check("5 the picture good again (accepted at once)", found(p) && p.streak == 0.0f, p);

    // 6: a start far beyond anything a game leaves behind (the restart off, as cases 3 and 4).
    g_keepStale = asked | 2u;
    StartOf(fits, 3000.0f);
    WriteFits(fits);
    p = Dispatch();
    Check("6 a start 3000 px off, Present 1 (rejected)", p.reason != 0 && p.streak == 1.0f, p);
    p = Dispatch();
    Check("6 a start 3000 px off, Present 2 (rejected)", p.reason != 0 && p.streak == 2.0f, p);
    p = Dispatch();
    Check("6 a start 3000 px off, Present 3 (the start given up: accepted)", found(p) && p.streak == 0.0f, p);
    g_keepStale = asked;
    StartOf(fits, 3000.0f);
    WriteFits(fits);
    p = Dispatch();
    Check("6 a start 3000 px off with the restart on (accepted at once)",
          found(p) && p.streak == 0.0f && p.restarted == 1.0f, p);

    // 8: the chain on hardware depth.
    bool fitsOk = false;
    float err = ChainError(fitsOk);
    std::printf("8 the chain, hardware depth: both fits %s; same-eye motion off the true chained maps by %.3f px at "
                "the most (static pixels) -> %s\n",
                fitsOk ? "accepted" : "NOT accepted", err, fitsOk && err < 0.2f ? "OK" : "FAIL");
    std::printf("   worst at %s; %d of %d checked pixels off by more than 1 px\n", g_worstAt, g_bad, g_checkedPx);
    ++g_checked;
    if (!(fitsOk && err < 0.2f))
        ++g_failures;

    // 7 and 9: linear depth, as Hogwarts hands it (the control hands it with LinearDepth = 0).
    WriteDepth(true);
    g_linear = linearOff ? 0u : 1u;
    std::printf("-- linear depth (%.0f / z) with LinearDepth = %u%s\n", kLinearK, g_linear,
                linearOff ? " (the control: Hogwarts as it was)" : "");
    WritePicture(true);
    std::memset(fits, 0, sizeof(fits));
    WriteFits(fits);
    p = Dispatch();
    Check("7 linear depth, no start (accepted, the map found with its z column x3.7)", found(p) && p.streak == 0.0f, p);
    err = ChainError(fitsOk);
    std::printf("9 the chain, linear depth: both fits %s; same-eye motion off the true chained maps by %.3f px at "
                "the most (static pixels) -> %s\n",
                fitsOk ? "accepted" : "NOT accepted", err, fitsOk && err < 0.2f ? "OK" : "FAIL");
    std::printf("   worst at %s\n", g_worstAt);
    ++g_checked;
    if (!(fitsOk && err < 0.2f))
        ++g_failures;

    // 13: the dump, on hardware depth and the plain picture, from no start.
    {
        g_linear = 0;
        WriteDepth(false);
        WritePicture(true);
        std::memset(fits, 0, sizeof(fits));
        WriteFits(fits);
        const bool noDump = GetEnvironmentVariableA("FIT12_TEST_NO_DUMP", value, sizeof(value)) > 0 && value[0] == '1';
        const uint32_t keep = g_keepStale;
        if (!noDump)
            g_keepStale |= 8u;
        p = Dispatch();
        g_keepStale = keep;
        Barrier(g_partials, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
        g_list->CopyBufferRegion(g_dumpRead, 0, g_partials, 0, kDumpWords * 4);
        Barrier(g_partials, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        Submit();
        void* mapped = nullptr;
        int wrong = 0, fitted = 0;
        uint32_t stamp = 0;
        if (SUCCEEDED(g_dumpRead->Map(0, nullptr, &mapped)))
        {
            const uint32_t* w = (const uint32_t*) mapped;
            stamp = w[64 * 80 + 1];
            const float* d = (const float*) (w + 64 * 80 + 4);
            for (UINT si = 0; si < 4096; ++si)
            {
                const UINT tx = (UINT) (((si % 64) + 0.5f) / 64.0f * kSize), ty = (UINT) (((si / 64) + 0.5f) / 64.0f * kSize);
                const float wantDepth = g_depthData[ty * kSize + tx];
                const float wantX = FromHalf(g_lastMotion[(ty * kSize + tx) * 2 + 0]);
                const float wantY = FromHalf(g_lastMotion[(ty * kSize + tx) * 2 + 1]);
                if (std::memcmp(&d[si * 4 + 0], &wantDepth, 4) != 0 || std::memcmp(&d[si * 4 + 1], &wantX, 4) != 0 ||
                    std::memcmp(&d[si * 4 + 2], &wantY, 4) != 0)
                {
                    if (++wrong <= 3)
                        std::printf("   dump sample %u: depth %.6f x %.6f y %.6f, uploaded %.6f %.6f %.6f\n", si, d[si * 4],
                                    d[si * 4 + 1], d[si * 4 + 2], wantDepth, wantX, wantY);
                }
                fitted += d[si * 4 + 3] == 1.0f ? 1 : 0;
            }
            D3D12_RANGE none { 0, 0 };
            g_dumpRead->Unmap(0, &none);
        }
        const bool ok13 = wrong == 0 && stamp == (g_stamp & 0xFFFFFFu) && fitted == 4096;
        std::printf("13 the dump: %d of 4096 samples not as uploaded, stamp %u (fit %u), %d marked fitted (no depth "
                    "gap here: all) -> %s\n",
                    wrong, stamp, g_stamp & 0xFFFFFFu, fitted, ok13 ? "OK" : "FAIL");
        ++g_checked;
        if (!ok13)
            ++g_failures;
    }

    // 15: the crop's move carried over through subrects (kShiftShader, PerEyeDlss_Dx12::ShiftComposed): the box of a
    // full-size vector texture, shifted, into another at the same place; outside the box the other keeps what it had.
    {
        bool ok15 = true;
        ID3DBlob* shiftCode = nullptr;
        ID3DBlob* shiftErr = nullptr;
        ID3D12RootSignature* shiftRoot = nullptr;
        ID3D12PipelineState* shiftPso = nullptr;
        ID3D12DescriptorHeap* shiftHeap = nullptr;
        D3D12_DESCRIPTOR_RANGE sr[2] {};
        sr[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        sr[0].NumDescriptors = 1;
        sr[0].OffsetInDescriptorsFromTableStart = 0;
        sr[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
        sr[1].NumDescriptors = 1;
        sr[1].OffsetInDescriptorsFromTableStart = 1;
        D3D12_ROOT_PARAMETER sp[2] {};
        sp[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        sp[0].Constants.Num32BitValues = 6;
        sp[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        sp[1].DescriptorTable.NumDescriptorRanges = 2;
        sp[1].DescriptorTable.pDescriptorRanges = sr;
        D3D12_ROOT_SIGNATURE_DESC srs {};
        srs.NumParameters = 2;
        srs.pParameters = sp;
        ID3DBlob* srsBlob = nullptr;
        ok15 = SUCCEEDED(D3D12SerializeRootSignature(&srs, D3D_ROOT_SIGNATURE_VERSION_1, &srsBlob, nullptr)) &&
               SUCCEEDED(g_device->CreateRootSignature(0, srsBlob->GetBufferPointer(), srsBlob->GetBufferSize(),
                                                       IID_PPV_ARGS(&shiftRoot)));
        if (ok15 && FAILED(D3DCompile(kShiftShader, std::strlen(kShiftShader), nullptr, nullptr, nullptr, "CSMain",
                                      "cs_5_0", D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &shiftCode, &shiftErr)))
        {
            std::printf("15 the shift shader does not compile: %s\n",
                        shiftErr != nullptr ? (const char*) shiftErr->GetBufferPointer() : "no message");
            ok15 = false;
        }
        if (ok15)
        {
            D3D12_COMPUTE_PIPELINE_STATE_DESC spd {};
            spd.pRootSignature = shiftRoot;
            spd.CS = { shiftCode->GetBufferPointer(), shiftCode->GetBufferSize() };
            ok15 = SUCCEEDED(g_device->CreateComputePipelineState(&spd, IID_PPV_ARGS(&shiftPso)));
        }
        D3D12_DESCRIPTOR_HEAP_DESC shd {};
        shd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        shd.NumDescriptors = 2;
        shd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        ok15 = ok15 && SUCCEEDED(g_device->CreateDescriptorHeap(&shd, IID_PPV_ARGS(&shiftHeap)));
        if (ok15)
        {
            // In = the motion texture holding (x, y) as halfs; Out = the compose's output: first a plain copy of all of
            // it (base 0, no shift), then the box (8, 8) 16x16 shifted by (3, -2). Exact in half floats.
            std::vector<uint16_t> in(kSize * kSize * 2);
            for (UINT y = 0; y < kSize; ++y)
                for (UINT x = 0; x < kSize; ++x)
                {
                    in[(y * kSize + x) * 2] = Half((float) x);
                    in[(y * kSize + x) * 2 + 1] = Half((float) y);
                }
            UploadTexture(g_motion, DXGI_FORMAT_R16G16_FLOAT, in.data());
            const UINT inc = g_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
            D3D12_CPU_DESCRIPTOR_HANDLE cpu = shiftHeap->GetCPUDescriptorHandleForHeapStart();
            D3D12_SHADER_RESOURCE_VIEW_DESC srv {};
            srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
            srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            srv.Texture2D.MipLevels = 1;
            srv.Format = DXGI_FORMAT_R16G16_FLOAT;
            g_device->CreateShaderResourceView(g_motion, &srv, cpu);
            cpu.ptr += inc;
            D3D12_UNORDERED_ACCESS_VIEW_DESC uav {};
            uav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
            uav.Format = DXGI_FORMAT_R16G16_FLOAT;
            g_device->CreateUnorderedAccessView(g_out, nullptr, &uav, cpu);
            // The control FIT12_TEST_SHIFT_OFF=1: the second pass shifts by nothing, the box must then read wrong.
            const bool shiftOff =
                GetEnvironmentVariableA("FIT12_TEST_SHIFT_OFF", value, sizeof(value)) > 0 && value[0] == '1';
            const float shift[2] = { shiftOff ? 0.0f : 3.0f, shiftOff ? 0.0f : -2.0f };
            auto run = [&](UINT bx, UINT by, UINT bw, UINT bh, const float* sh)
            {
                uint32_t c[6] = { bx, by, bw, bh, 0, 0 };
                std::memcpy(&c[4], sh, 2 * sizeof(float));
                ID3D12DescriptorHeap* heaps[] = { shiftHeap };
                g_list->SetDescriptorHeaps(1, heaps);
                g_list->SetComputeRootSignature(shiftRoot);
                g_list->SetPipelineState(shiftPso);
                g_list->SetComputeRoot32BitConstants(0, 6, c, 0);
                g_list->SetComputeRootDescriptorTable(1, shiftHeap->GetGPUDescriptorHandleForHeapStart());
                g_list->Dispatch((bw + 7) / 8, (bh + 7) / 8, 1);
                D3D12_RESOURCE_BARRIER u {};
                u.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
                u.UAV.pResource = g_out;
                g_list->ResourceBarrier(1, &u);
            };
            const float none[2] = { 0.0f, 0.0f };
            run(0, 0, kSize, kSize, none);
            run(8, 8, 16, 16, shift);
            std::vector<float> px;
            ok15 = ReadOut(px); // scaled by -kSize there (the compose's units): undone below, exact
            int wrong = 0;
            for (UINT y = 0; y < kSize && ok15; ++y)
                for (UINT x = 0; x < kSize; ++x)
                {
                    const bool inBox = x >= 8 && x < 24 && y >= 8 && y < 24;
                    const float ex = (float) x + (inBox ? 3.0f : 0.0f), ey = (float) y + (inBox ? -2.0f : 0.0f);
                    const float gx = px[(y * kSize + x) * 2] / -(float) kSize;
                    const float gy = px[(y * kSize + x) * 2 + 1] / -(float) kSize;
                    if (gx != ex || gy != ey)
                        ++wrong;
                }
            ok15 = ok15 && wrong == 0;
            std::printf("15 the crop's move through subrects (kShiftShader): the box (8, 8) 16x16 shifted by (%.0f, %.0f), "
                        "%d of %u texels wrong -> %s\n",
                        shift[0], shift[1], wrong, kSize * kSize, ok15 ? "OK" : "FAIL");
        }
        else
            std::printf("15 FAILED: the shift pass could not be made\n");
        ++g_checked;
        if (!ok15)
            ++g_failures;
    }

    // 16: the crop's age (kAgeShader, PerEyeDlss_Dx12::UpdateAge): a reset, three frames in place, then a move.
    {
        bool ok16 = true;
        constexpr UINT kAge = 32;
        ID3DBlob* ageCode = nullptr;
        ID3DBlob* ageErr = nullptr;
        ID3D12RootSignature* ageRoot = nullptr;
        ID3D12PipelineState* agePso = nullptr;
        ID3D12DescriptorHeap* ageHeap = nullptr;
        ID3D12Resource* ageTex[2] = { nullptr, nullptr };
        D3D12_DESCRIPTOR_RANGE ar[2] {};
        ar[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        ar[0].NumDescriptors = 1;
        ar[0].OffsetInDescriptorsFromTableStart = 0;
        ar[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
        ar[1].NumDescriptors = 1;
        ar[1].OffsetInDescriptorsFromTableStart = 1;
        D3D12_ROOT_PARAMETER ap[2] {};
        ap[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        ap[0].Constants.Num32BitValues = 8;
        ap[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        ap[1].DescriptorTable.NumDescriptorRanges = 2;
        ap[1].DescriptorTable.pDescriptorRanges = ar;
        D3D12_ROOT_SIGNATURE_DESC ars {};
        ars.NumParameters = 2;
        ars.pParameters = ap;
        ID3DBlob* arsBlob = nullptr;
        ok16 = SUCCEEDED(D3D12SerializeRootSignature(&ars, D3D_ROOT_SIGNATURE_VERSION_1, &arsBlob, nullptr)) &&
               SUCCEEDED(g_device->CreateRootSignature(0, arsBlob->GetBufferPointer(), arsBlob->GetBufferSize(),
                                                       IID_PPV_ARGS(&ageRoot)));
        if (ok16 && FAILED(D3DCompile(kAgeShader, std::strlen(kAgeShader), nullptr, nullptr, nullptr, "CSMain", "cs_5_0",
                                      D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &ageCode, &ageErr)))
        {
            std::printf("16 the age shader does not compile: %s\n",
                        ageErr != nullptr ? (const char*) ageErr->GetBufferPointer() : "no message");
            ok16 = false;
        }
        if (ok16)
        {
            D3D12_COMPUTE_PIPELINE_STATE_DESC apd {};
            apd.pRootSignature = ageRoot;
            apd.CS = { ageCode->GetBufferPointer(), ageCode->GetBufferSize() };
            ok16 = SUCCEEDED(g_device->CreateComputePipelineState(&apd, IID_PPV_ARGS(&agePso)));
        }
        D3D12_DESCRIPTOR_HEAP_DESC ahd {};
        ahd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        ahd.NumDescriptors = 4; // two slots of (SRV, UAV): A -> B and B -> A
        ahd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        ok16 = ok16 && SUCCEEDED(g_device->CreateDescriptorHeap(&ahd, IID_PPV_ARGS(&ageHeap)));
        D3D12_HEAP_PROPERTIES aheap {};
        aheap.Type = D3D12_HEAP_TYPE_DEFAULT;
        D3D12_RESOURCE_DESC atd {};
        atd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        atd.Width = atd.Height = kAge;
        atd.DepthOrArraySize = atd.MipLevels = 1;
        atd.SampleDesc.Count = 1;
        atd.Format = DXGI_FORMAT_R8_UINT;
        atd.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        for (int i = 0; i < 2 && ok16; ++i)
            ok16 = SUCCEEDED(g_device->CreateCommittedResource(&aheap, D3D12_HEAP_FLAG_NONE, &atd,
                                                               D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
                                                               IID_PPV_ARGS(&ageTex[i])));
        if (ok16)
        {
            const UINT inc = g_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
            D3D12_CPU_DESCRIPTOR_HANDLE cpu = ageHeap->GetCPUDescriptorHandleForHeapStart();
            D3D12_SHADER_RESOURCE_VIEW_DESC srv {};
            srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
            srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            srv.Texture2D.MipLevels = 1;
            srv.Format = DXGI_FORMAT_R8_UINT;
            D3D12_UNORDERED_ACCESS_VIEW_DESC uav {};
            uav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
            uav.Format = DXGI_FORMAT_R8_UINT;
            for (int from = 0; from < 2; ++from) // slot `from`: reads ageTex[from], writes the other
            {
                g_device->CreateShaderResourceView(ageTex[from], &srv, cpu);
                cpu.ptr += inc;
                g_device->CreateUnorderedAccessView(ageTex[1 - from], nullptr, &uav, cpu);
                cpu.ptr += inc;
            }
            int cur = 0; // the texture holding the latest ages
            auto run = [&](int sx, int sy, bool reset)
            {
                const int from = cur, to = 1 - cur;
                Barrier(ageTex[from], D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                const uint32_t c[8] = { kAge, kAge, (uint32_t) sx, (uint32_t) sy, reset ? 1u : 0u, 0, 0, 0 };
                ID3D12DescriptorHeap* heaps[] = { ageHeap };
                g_list->SetDescriptorHeaps(1, heaps);
                g_list->SetComputeRootSignature(ageRoot);
                g_list->SetPipelineState(agePso);
                g_list->SetComputeRoot32BitConstants(0, 8, c, 0);
                D3D12_GPU_DESCRIPTOR_HANDLE gpu = ageHeap->GetGPUDescriptorHandleForHeapStart();
                gpu.ptr += (UINT64) from * 2 * inc;
                g_list->SetComputeRootDescriptorTable(1, gpu);
                g_list->Dispatch((kAge + 7) / 8, (kAge + 7) / 8, 1);
                Barrier(ageTex[from], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                D3D12_RESOURCE_BARRIER u {};
                u.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
                u.UAV.pResource = ageTex[to];
                g_list->ResourceBarrier(1, &u);
                cur = to;
            };
            // The control FIT12_TEST_AGE_OFF=1: the move's pass carries nothing along (shift 0): the ages read wrong.
            const bool ageOff = GetEnvironmentVariableA("FIT12_TEST_AGE_OFF", value, sizeof(value)) > 0 && value[0] == '1';
            run(0, 0, true);  // a reset: 0 everywhere
            run(0, 0, false); // 1
            run(0, 0, false); // 2
            run(0, 0, false); // 3
            run(ageOff ? 0 : 5, ageOff ? 0 : -2, false); // the box moved by (5, -2): a pixel's old place is p + (5, -2)
            // Read the latest ages back.
            Barrier(ageTex[cur], D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
            D3D12_TEXTURE_COPY_LOCATION dst {}, src {};
            src.pResource = ageTex[cur];
            src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            dst.pResource = g_outRead;
            dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
            dst.PlacedFootprint.Footprint = { DXGI_FORMAT_R8_UINT, kAge, kAge, 1, 256 };
            g_list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
            Barrier(ageTex[cur], D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            ok16 = Submit();
            int wrong = 0;
            if (ok16)
            {
                void* mapped = nullptr;
                ok16 = SUCCEEDED(g_outRead->Map(0, nullptr, &mapped));
                if (ok16)
                {
                    const uint8_t* ages = (const uint8_t*) mapped;
                    for (UINT y = 0; y < kAge; ++y)
                        for (UINT x = 0; x < kAge; ++x)
                        {
                            const int fx = (int) x + 5, fy = (int) y - 2;
                            const uint8_t expect = fx < (int) kAge && fy >= 0 ? 4 : 0;
                            if (ages[y * 256 + x] != expect)
                                ++wrong;
                        }
                    D3D12_RANGE none { 0, 0 };
                    g_outRead->Unmap(0, &none);
                }
            }
            ok16 = ok16 && wrong == 0;
            std::printf("16 the crop's age (kAgeShader): a reset, three frames in place, a move by (5, -2): %d of %u texels "
                        "wrong (4 where the pixel was in the box before, 0 where it came in) -> %s\n",
                        wrong, kAge * kAge, ok16 ? "OK" : "FAIL");
        }
        else
            std::printf("16 FAILED: the age pass could not be made\n");
        ++g_checked;
        if (!ok16)
            ++g_failures;
    }

    // 17: the gaze filter's saccade (GazeFilter.h): how many polls after the eyes land the smoothed gaze is there.
    {
        // Polls 11.1 ms apart (90 Hz), steadiness 0.6 (the default): still at 0 for 20 polls, a 20 deg saccade over
        // four polls (5, 10, 15, 20), then still at 20. The eyes' landing = the first poll at 20.
        const bool noSnap = GetEnvironmentVariableA("FIT12_TEST_NO_SNAP", value, sizeof(value)) > 0 && value[0] == '1';
        auto settle = [&](bool snap, int& glitchMoved)
        {
            GazeFilter::State f;
            const double dt = 1.0 / 90.0;
            for (int i = 0; i < 20; ++i)
                GazeFilter::Step(f, 0.0, 0.0, dt, 0.6, snap);
            // A single-sample glitch of 5 deg: the filter must not move (within 0.05 deg).
            GazeFilter::Step(f, 5.0, 0.0, dt, 0.6, snap);
            glitchMoved = std::abs(f.yaw) > 0.05 ? 1 : 0;
            GazeFilter::Step(f, 0.0, 0.0, dt, 0.6, snap);
            for (double raw : { 5.0, 10.0, 15.0 })
                GazeFilter::Step(f, raw, 0.0, dt, 0.6, snap);
            int polls = 0;
            for (; polls < 30; ++polls)
            {
                GazeFilter::Step(f, 20.0, 0.0, dt, 0.6, snap);
                if (std::abs(f.yaw - 20.0) < 0.1)
                    break;
            }
            return polls + 1; // polls at the landing point until the filter is within 0.1 deg of it
        };
        int glitchSnap = 0, glitchOld = 0;
        const int withSnap = settle(true, glitchSnap), without = settle(false, glitchOld);
        const int tested = noSnap ? without : withSnap;
        const bool ok17 = tested <= 2 && glitchSnap == 0 && glitchOld == 0 && without > withSnap;
        std::printf("17 the gaze filter's saccade: within 0.1 deg of the landing point after %d polls with the snap, %d "
                    "without (the tested one: %d, need <= 2); a one-sample glitch moved the filter: %d / %d (need 0) "
                    "-> %s\n",
                    withSnap, without, tested, glitchSnap, glitchOld, ok17 ? "OK" : "FAIL");
        ++g_checked;
        if (!ok17)
            ++g_failures;
    }

    std::printf("%s: %d checks, %d failures\n", g_failures == 0 ? "PASSED" : "FAILED", g_checked, g_failures);
    return g_failures == 0 ? 0 : 1;
}
