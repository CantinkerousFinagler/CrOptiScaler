#pragma once

// The fit and the compose of PerEyeDlss_Dx12 (see there), in a header of their own so that the offline test
// (tools/xr_observer/test/fit12_test) compiles and runs this very shader, by this very sequence of dispatches (INIT
// with Round 100 in one group, then rounds 0-4 in 64 groups each), on made-up depth and motion. No includes: the game
// DLL includes it inside its anonymous namespace.

constexpr char kFitShader[] = R"(
// VK_MODE: the same source compiled to SPIR-V for PerEyeDlss_Vk (tools\xr_observer\test\vk_fit_shaders.py writes
// PerEyeFitShader_Vk.h); one descriptor set for the fit and the compose, the constants as push constants.
#ifdef VK_MODE
[[vk::binding(0, 0)]] Texture2D<float> Depth;
[[vk::binding(1, 0)]] Texture2D<float2> Motion;
[[vk::binding(5, 0)]] RWStructuredBuffer<float4> Fits;
[[vk::binding(6, 0)]] globallycoherent RWStructuredBuffer<uint> Partials;
struct ConstantsVk // the 48 bytes of the D3D12 root constants, as push constants (a round = one vkCmdPushConstants)
{
    float2 MvScale;
    uint Cur;
    uint Prev;
    float Knee0;
    float Knee1;
    float Gain;
    uint Radius;
    uint Stamp;
    uint Round;
    uint KeepStale;
    uint LinearDepth;
};
[[vk::push_constant]] ConstantsVk PC;
#define MvScale PC.MvScale
#define Cur PC.Cur
#define Prev PC.Prev
#define Knee0 PC.Knee0
#define Knee1 PC.Knee1
#define Gain PC.Gain
#define Radius PC.Radius
#define Stamp PC.Stamp
#define LinearDepth PC.LinearDepth
#define Round PC.Round
#define KeepStale PC.KeepStale
#else
Texture2D<float> Depth : register(t0);
Texture2D<float2> Motion : register(t1);
RWStructuredBuffer<float4> Fits : register(u0);
// 64 x 80 partial sums (float bits), then the count of groups done this round (the last group sets it back to 0);
// then (the dump, KeepStale bit 3) the fit's stamp at kDump - 3 and from kDump the 4096 samples of the measuring round,
// 4 words each: the raw depth, the raw motion x and y (as handed, before MV scale), 1.0 if the sample was fitted.
globallycoherent RWStructuredBuffer<uint> Partials : register(u2);

cbuffer Constants : register(b0)
{
    float2 MvScale;
    uint Cur;
    uint Prev;
    float Knee0;
    float Knee1;
    float Gain;
    uint Radius;
    uint Stamp;
    uint Round;
    uint KeepStale; // bit 0: a stale start is never given up (as it used to be); bit 1: no restart within the fit;
                    // bit 2: nothing is left out at a depth gap (every sample with depth is fitted and counted; the
                    // restart's gate 40 px only and its rounds 128, 2, 0.5 px: the fit as it used to be);
                    // bit 3: DUMP this fit's samples into Partials (see there; the overlay's 'Save the fit's samples')
    uint LinearDepth; // 1 = the depth is linear: 1/depth (FitDepth)
};
#endif

static const float KZ = 100.0;
static const uint kCounter = 64 * 80;
static const uint kDump = kCounter + 4;
groupshared float S[64][80];
groupshared float T[80];
groupshared uint Done;
groupshared uint H[64];

uint Idx(uint i, uint j) { return i * 11 - (i * (i - 1)) / 2 + (j - i); }
bool Finite(float v) { return (asuint(v) & 0x7F800000u) != 0x7F800000u; }
// The fit and the compose work on hardware depth (z ~ 1/w: a projective map in (x, y, z)). A game that hands DLSS
// LINEAR depth d (RR's DLSS.Use.HW.Depth = 0: Hogwarts Legacy, 22873 fits and none accepted) gets 1/d,
// which is again ~ 1/w; larger is nearer either way, as the compose's "nearest" assumes. 0 = no depth there.
// 10/d, not 1/d: of the size of hardware depth for centimetres (Unreal's near plane is 10), so the solve's pull
// toward its start does not bend the depth column (fit12_test: 0.10 px residual on 1/d, see case 7).
float FitDepth(float d) { return LinearDepth != 0 ? (Finite(d) && d > 0.0 ? 10.0 / d : 0.0) : d; }
)" // MSVC caps one string literal at 16 KB (C2026): the shader is three adjacent raw literals
R"(
[numthreads(64, 1, 1)]
void CSMain(uint gi : SV_GroupIndex, uint3 g : SV_GroupID)
{
    if (Round == 100)
    {
        // Start from this eye's last accepted fit (it maps the same pair of eyes, two Presents earlier), or from no
        // motion (D3D11 path: 55 of 55 rejected running frames recovered from the previous fit, 3 of 55 unweighted).
        // Past two rejections in a row every second try starts from no motion instead (as PerEyeDlss_Dx11
        // does): weights are zero past 40 px of the start, and only an accepted fit replaces it, so a start left
        // behind by fast motion (Cyberpunk 2077, sprinting and driving: 0 of 600 fits valid for 7 s) was never
        // given up. Slot 3's row 4: x = this fit gave up the start, y = it starts from the eye's last fit.
        //
        // WHAT LIES NEARER THAN A DEPTH GAP IS LEFT OUT (KeepStale bit 2 clear). Slot 3's row 3: x = the fit depth at
        // or above which a sample is left out of the fit, the restart's gate and the verdict alike (larger is nearer),
        // y = the samples with depth, z = those left out. In one D3D12 game, driving in first person (the
        // fit's own sample dumps): the CAR -- roof, pillars, dashboard, body -- is 69-78% of the samples with depth, at
        // hardware z 0.125-0.7, and the world through the windows lies below 0.044; between them three half-octave
        // bins hold 0-1 samples. The world alone fits one camera map to 0.002 px (100% within 1 px) at a slow drive,
        // 83-92% at speed (a passed car, the hood's edge); with the car in, the fit sat on the car or between the two.
        // Standing, the histogram has no such gap and nothing is left out. A histogram of log2 of the fit depth, half
        // an octave a bin, from 2^-24 to 2^8 (hardware depth and 10/d alike); THE GAP = the sparsest bin with at least
        // 10% of the samples on either side, if it holds under 2% of them; everything from that bin up is left out
        // (the bin's own stragglers are edges). The gap used to be a VALLEY: a run of bins each under 0.5%,
        // the widest run -- and sitting against a car's side window (the same game's dumps) the
        // only sparse bin between the door and the street held 0.35-0.74%: one eye's fits lost the gap, took the
        // door in and came out at 0.2-0.5 px (the crop shimmered out of that window); pulled back from the window
        // the bin held 0.1-0.2% and both eyes were fine. The 10% on either side still keep a standing picture
        // (bins 1-5% each, nothing nearer) uncut, and the drives' gaps (0-1 samples) are the same bins as before.
        // KeepStale bit 4 (16) = the valley rule as it was: the offline test's control (FIT12_TEST_OLD_GAP).
        H[gi] = 0;
        GroupMemoryBarrierWithGroupSync();
        uint idw, idh;
        Depth.GetDimensions(idw, idh);
        for (uint k = 0; k < 64; ++k)
        {
            const uint si = k * 64 + gi;
            const float2 suv = (float2(si % 64, si / 64) + 0.5) / 64.0;
            const float sz = FitDepth(Depth[min(uint2(suv * float2(idw, idh)), uint2(idw - 1, idh - 1))]);
            if (Finite(sz) && sz > 0.0)
                InterlockedAdd(H[(uint) clamp(floor((log2(sz) + 24.0) * 2.0), 0.0, 63.0)], 1);
        }
        GroupMemoryBarrierWithGroupSync();
        if (gi == 0)
        {
            uint total = 0;
            for (uint b0 = 0; b0 < 64; ++b0)
                total += H[b0];
            uint lo = 64, hi = 0;
            for (uint b1 = 0; b1 < 64; ++b1)
                if (H[b1] > 0)
                {
                    lo = min(lo, b1);
                    hi = max(hi, b1);
                }
            uint below = 0, best = 64, bestWidth = 0, bestBelow = 0;
            if ((KeepStale & 16u) != 0)
            {
                // The test's control: the valley rule as it used to be.
                uint b2 = lo;
                while (b2 <= hi && lo < 64)
                {
                    if (H[b2] * 200 < total)
                    {
                        uint e2 = b2, inRun = H[b2];
                        while (e2 + 1 <= hi && H[e2 + 1] * 200 < total)
                        {
                            ++e2;
                            inRun += H[e2];
                        }
                        const uint above = total - below - inRun;
                        if (below * 10 >= total && above * 10 >= total && e2 - b2 + 1 > bestWidth)
                        {
                            best = b2;
                            bestWidth = e2 - b2 + 1;
                            bestBelow = below;
                        }
                        below += inRun;
                        b2 = e2 + 1;
                    }
                    else
                    {
                        below += H[b2];
                        ++b2;
                    }
                }
            }
            else
            {
                // The sparsest bin with at least 10% of the samples on either side, holding under 2% of them; of
                // equal bins the farthest (more of the world kept, the near side's stragglers out).
                uint bestCount = 0;
                for (uint b3 = 0; b3 < 64; ++b3)
                {
                    const uint above = total - below - H[b3];
                    if (below * 10 >= total && above * 10 >= total && H[b3] * 50 < total &&
                        (best == 64 || H[b3] < bestCount))
                    {
                        best = b3;
                        bestCount = H[b3];
                        bestBelow = below;
                    }
                    below += H[b3];
                }
            }
            const bool on = (KeepStale & 4u) == 0 && best < 64;
            Fits[3 * 5 + 3] = float4(on ? exp2(best * 0.5 - 24.0) : 3.0e38, (float) total,
                                     on ? (float) (total - bestBelow) : 0.0, 0);
        }
        if (gi == 0)
        {
            const bool stored = Fits[Cur * 5 + 3].z > 0.5;
            const uint streak = (uint) max(Fits[Cur * 5 + 4].y, 0.0);
            const bool giveUp = (KeepStale & 1u) == 0 && stored && streak >= 2 && (streak & 1) == 0;
            const bool have = stored && !giveUp;
            const float4 r0 = have ? Fits[Cur * 5 + 0] : float4(1, 0, 0, 0);
            const float4 r1 = have ? Fits[Cur * 5 + 1] : float4(0, 1, 0, 0);
            const float4 r2 = have ? float4(Fits[Cur * 5 + 2].xyz, 1.0) : float4(0, 0, 0, 1);
            Fits[2 * 5 + 0] = r0;
            Fits[2 * 5 + 1] = r1;
            Fits[2 * 5 + 2] = r2;
            Fits[2 * 5 + 3] = float4(0, 0, 0, 0);
            Fits[3 * 5 + 0] = r0;
            Fits[3 * 5 + 1] = r1;
            Fits[3 * 5 + 2] = r2;
            Fits[3 * 5 + 4] = float4(giveUp ? 1.0 : 0.0, have ? 1.0 : 0.0, 0, 0);
            Partials[kCounter] = 0; // a count left out of step can cost one fit, never every fit after it
        }
        return;
    }

    for (uint c0 = 0; c0 < 80; ++c0)
        S[gi][c0] = 0;
    uint w, h, dw, dh;
    Motion.GetDimensions(w, h);
    Depth.GetDimensions(dw, dh);
    const float2 size = float2(w, h);
    const uint s = g.x * 64 + gi;
    const float2 uv = (float2(s % 64, s / 64) + 0.5) / 64.0;
    const float rawDepth = Depth[min(uint2(uv * float2(dw, dh)), uint2(dw - 1, dh - 1))];
    const float z = FitDepth(rawDepth);
    const float2 raw = Motion[min(uint2(uv * size), uint2(w - 1, h - 1))];
    // A sample nearer than the depth gap (INIT) is no sample at all: not weighed, not counted by the restart or the verdict.
    const bool fitted = Finite(z) && z > 0.0 && z < Fits[3 * 5 + 3].x && Finite(raw.x) && Finite(raw.y);
    if (Round == 4 && (KeepStale & 8u) != 0)
    {
        Partials[kDump + s * 4 + 0] = asuint(rawDepth);
        Partials[kDump + s * 4 + 1] = asuint(raw.x);
        Partials[kDump + s * 4 + 2] = asuint(raw.y);
        Partials[kDump + s * 4 + 3] = asuint(fitted ? 1.0 : 0.0);
    }
    if (fitted)
    {
        const float4 r0 = Fits[2 * 5 + 0];
        const float4 r1 = Fits[2 * 5 + 1];
        const float4 r2 = Fits[2 * 5 + 2];
        const bool have = Fits[3 * 5 + 4].y > 0.5;
        // 2, 0.5, 0.3, 0.3 px from the eye's last fit, or 32, 2, 0.5, 0.3 px from no motion. When round 0 restarted
        // the fit (see the solve) the three rounds left take 128, 2, 0.5 px: the first weighs every sample within
        // 128 px of no motion alike, so the most of the picture wins over a part that happens to lie AT no motion (a
        // body that turns with the camera; at 32 px that part held the fit, fit12_test case 11). With the nearest
        // gap rule on (KeepStale bit 2 clear) 128, 8, 2 px: a gentler way down from 128 (cases 10, 11: 0.12-0.13
        // px against 0.16-0.22 after a restart).
        const bool restarted = Fits[3 * 5 + 4].z > 0.5;
        const bool nearRule = (KeepStale & 4u) == 0;
        const float tau = have        ? (Round == 0 ? 2.0 : Round == 1 ? 0.5 : 0.3)
                          : restarted ? (Round == 1 ? 128.0 : Round == 2 ? (nearRule ? 8.0 : 2.0) : (nearRule ? 2.0 : 0.5))
                                      : (Round == 0 ? 32.0 : Round == 1 ? 2.0 : Round == 2 ? 0.5 : 0.3);
        const float2 mv = raw * MvScale;
        const float x = uv.x * 2.0 - 1.0;
        const float y = 1.0 - uv.y * 2.0;
        const float xp = x + mv.x * 2.0 / size.x;
        const float yp = y - mv.y * 2.0 / size.y;
        const float zz = z * KZ;
        const float4 p = float4(x, y, zz, 1.0);
        const float wq = dot(r2, p);
        const float ex = (dot(r0, p) / wq - xp) * size.x * 0.5;
        const float ey = (dot(r1, p) / wq - yp) * size.y * 0.5;
        const float e = sqrt(ex * ex + ey * ey);
        S[gi][79] = 1.0;
        if (Round == 4)
        {
            if (e < 1.0)
            {
                S[gi][77] = e;
                S[gi][78] = 1.0;
            }
        }
        else
        {
            const float wgt = e < 20.0 * tau ? 1.0 / max(e / tau, 1.0) : 0.0;
            S[gi][78] = wgt > 0.0 ? 1.0 : 0.0; // samples that count at all this round (round 0 asks: restart?)
            S[gi][77] = e < 8.0 ? 1.0 : 0.0;   // samples within 8 px of the start (round 0 asks too: bit 2 clear)
            const float rx[11] = { x, y, zz, 1.0, 0, 0, 0, 0, -xp * x, -xp * y, -xp * zz };
            const float ry[11] = { 0, 0, 0, 0, x, y, zz, 1.0, -yp * x, -yp * y, -yp * zz };
            [unroll] for (uint i = 0; i < 11; ++i)
            {
                [unroll] for (uint j = i; j < 11; ++j)
                    S[gi][Idx(i, j)] = wgt * (rx[i] * rx[j] + ry[i] * ry[j]);
                S[gi][66 + i] = wgt * (rx[i] * xp + ry[i] * yp);
            }
        }
    }
    GroupMemoryBarrierWithGroupSync();
    for (uint col = gi; col < 80; col += 64)
    {
        float sum = 0;
        for (uint r = 0; r < 64; ++r)
            sum += S[r][col];
        Partials[g.x * 80 + col] = asuint(sum);
    }

    // This group's partials are visible to every group before it counts itself done; the group that brings the count
    // to 64 sees all of them. Every barrier below is reached by the whole group (Done is one value per group).
    DeviceMemoryBarrierWithGroupSync();
    if (gi == 0)
    {
        uint before;
        InterlockedAdd(Partials[kCounter], 1, before);
        Done = before;
    }
    GroupMemoryBarrierWithGroupSync();
    const bool last = Done == 63;
    if (last)
        for (uint col2 = gi; col2 < 80; col2 += 64)
        {
            float sum = 0;
            for (uint p = 0; p < 64; ++p)
                sum += asfloat(Partials[p * 80 + col2]);
            T[col2] = sum;
        }
    GroupMemoryBarrierWithGroupSync();
    if (!last || gi != 0)
        return;
    Partials[kCounter] = 0; // the next round counts from 0 (ordered before it by the UAV barrier between dispatches)
)"
R"(

    if (Round < 4)
    {
        // Cholesky on the normal equations, a small ridge, and a gentle pull toward the start (keeps a near-planar
        // grid -- a wall or the ground filling the view -- solvable).
        float init[11];
        const float4 s0 = Fits[3 * 5 + 0], s1 = Fits[3 * 5 + 1], s2 = Fits[3 * 5 + 2];
        init[0] = s0.x; init[1] = s0.y; init[2] = s0.z; init[3] = s0.w;
        init[4] = s1.x; init[5] = s1.y; init[6] = s1.z; init[7] = s1.w;
        init[8] = s2.x; init[9] = s2.y; init[10] = s2.z;
        float L[11][11];
        float trace = 0;
        for (uint i = 0; i < 11; ++i)
            trace += T[Idx(i, i)];
        const float ridge = 1e-7 * trace / 11.0;
        const float prior = 1e-4 * trace / 11.0;
        bool ok = trace > 0;
        for (uint a = 0; a < 11; ++a)
        {
            for (uint b = 0; b <= a; ++b)
            {
                float v = T[Idx(b, a)] + (a == b ? ridge + prior : 0.0);
                for (uint q = 0; q < b; ++q)
                    v -= L[a][q] * L[b][q];
                if (a == b)
                {
                    ok = ok && v > 0;
                    L[a][a] = sqrt(max(v, 1e-30));
                }
                else
                    L[a][b] = v / L[b][b];
            }
        }
        float yv[11];
        for (uint a1 = 0; a1 < 11; ++a1)
        {
            float v = T[66 + a1] + prior * init[a1];
            for (uint q = 0; q < a1; ++q)
                v -= L[a1][q] * yv[q];
            yv[a1] = v / L[a1][a1];
        }
        float sol[11];
        for (int a2 = 10; a2 >= 0; --a2)
        {
            float v = yv[a2];
            for (uint q = a2 + 1; q < 11; ++q)
                v -= L[q][a2] * sol[q];
            sol[a2] = v / L[a2][a2];
        }
        bool finite = true;
        for (uint f = 0; f < 11; ++f)
            finite = finite && Finite(sol[f]);
        const bool valid = ok && finite;
        // Restart within the fit (KeepStale bit 1 clear): round 0 found the eye's last fit left behind -- its solve
        // failed, or at most half of the samples lie within 40 px of it, and a fit needs more than half within 1 px
        // to be accepted. The three rounds left start from no motion (32, 2, 0.5 px). A flick of the stick (Star
        // Wars Outlaws) otherwise costs each eye two rejected fits at the turn's start and two at its
        // end, each handed the eye's last fit: the picture shimmers and settles. Slot 3's row 4: z = restarted.
        // With the gap rule on (KeepStale bit 2 clear) the gate is 8 px, not 40: a start that sat on a cabin
        // keeps the world within 40 px at a moderate speed, and the few near samples left in (the cut is by whole
        // bins) are enough to hold it there -- at 2 px the world 15 px away weighs 0.13 a sample (fit12_test 12b, 12c).
        if (Round == 0 && (KeepStale & 2u) == 0 && Fits[3 * 5 + 4].y > 0.5 &&
            (!valid || T[78] <= 0.5 * T[79] || ((KeepStale & 4u) == 0 && T[77] <= 0.5 * T[79])))
        {
            Fits[2 * 5 + 0] = float4(1, 0, 0, 0);
            Fits[2 * 5 + 1] = float4(0, 1, 0, 0);
            Fits[2 * 5 + 2] = float4(0, 0, 0, 1);
            Fits[2 * 5 + 3] = float4(0, 0, 0, 0);
            Fits[3 * 5 + 0] = float4(1, 0, 0, 0);
            Fits[3 * 5 + 1] = float4(0, 1, 0, 0);
            Fits[3 * 5 + 2] = float4(0, 0, 0, 1);
            Fits[3 * 5 + 4] = float4(Fits[3 * 5 + 4].x, 0.0, 1.0, 0.0);
            return;
        }
        // A failed solve keeps the previous round's solution for the next round's weights; the verdict still says
        // "solve failed" if the last round failed.
        if (valid)
        {
            Fits[2 * 5 + 0] = float4(sol[0], sol[1], sol[2], sol[3]);
            Fits[2 * 5 + 1] = float4(sol[4], sol[5], sol[6], sol[7]);
            Fits[2 * 5 + 2] = float4(sol[8], sol[9], sol[10], 1.0);
        }
        Fits[2 * 5 + 3] = float4(0, 0, valid ? 1.0 : 0.0, 0);
        return;
    }

    // Round 4: the verdict. Reason 0 accepted, 1 the last solve failed, 2 too few samples with depth, 3 too few
    // samples within 1 px. Only an accepted fit replaces this parity's slot; a rejected one keeps the eye's last
    // accepted fit and counts the streak. The stamp is an exact float value (raw bits could be flushed as a denormal).
    // n and the inliers count the samples beyond the depth gap only (when there is one).
    const float n = T[79];
    const float inliers = T[78];
    const bool solved = Fits[2 * 5 + 3].z > 0.5;
    const float reason = !solved ? 1.0 : n <= 256.0 ? 2.0 : inliers <= 0.5 * n ? 3.0 : 0.0;
    const float4 st = float4(inliers > 0 ? T[77] / inliers : 99.0, n > 0 ? inliers / n : 0.0, reason == 0.0 ? 1.0 : 0.0,
                             n);
    Fits[2 * 5 + 3] = st;
    // Row 4: x reason, y gave up the start, z restarted within the fit (0 / 1) + 0.45 x the share of the samples with
    // depth left out beyond the gap (0 .. 0.45 for 0 .. 100%: "> 0.5" still reads the restart), w the stamp.
    const float leftOut = Fits[3 * 5 + 3].y > 0.0 ? Fits[3 * 5 + 3].z / Fits[3 * 5 + 3].y : 0.0;
    Fits[2 * 5 + 4] = float4(reason, Fits[3 * 5 + 4].x, (Fits[3 * 5 + 4].z > 0.5 ? 1.0 : 0.0) + 0.45 * saturate(leftOut),
                             (float) (Stamp & 0xFFFFFFu));
    if ((KeepStale & 8u) != 0)
        Partials[kDump - 3] = Stamp & 0xFFFFFFu; // the dump's samples belong to this fit
    if (reason == 0.0)
    {
        Fits[Cur * 5 + 0] = Fits[2 * 5 + 0];
        Fits[Cur * 5 + 1] = Fits[2 * 5 + 1];
        Fits[Cur * 5 + 2] = Fits[2 * 5 + 2];
        Fits[Cur * 5 + 3] = st;
        Fits[Cur * 5 + 4] = float4(0, 0, 0, 0);
    }
    else
    {
        const float4 keep = Fits[Cur * 5 + 4];
        Fits[Cur * 5 + 4] = float4(reason, keep.y + 1.0, 0, 0);
    }
}
)";

// THE CROP'S MOVE THROUGH SUBRECTS (PerEyeDlss_Dx12::ShiftComposed): the box of our composed vectors,
// shifted by the crop's move, into a full-size texture of ours that the crop reads through the same subrect base.
// (A crop that follows the gaze used to need COPIES to carry a move over, and with a DLSS pass below us a copied,
// crop-sized evaluate lost its move; a subrect evaluate keeps it.) D3D12 only; fit12_test case 15.
constexpr char kShiftShader[] = R"(
Texture2D<float2> In : register(t0);
RWTexture2D<float2> Out : register(u0);
cbuffer Constants : register(b0)
{
    uint2 Base;
    uint2 Size;
    float2 Shift;
};
[numthreads(8, 8, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= Size.x || id.y >= Size.y)
        return;
    const uint2 p = Base + id.xy;
    Out[p] = In[p] + Shift;
}
)";

constexpr char kComposeShader[] = R"(
#ifdef VK_MODE
[[vk::binding(0, 0)]] Texture2D<float> Depth;
[[vk::binding(1, 0)]] Texture2D<float2> Motion;
[[vk::binding(2, 0)]] StructuredBuffer<float4> Fits;
[[vk::binding(3, 0)]] [[vk::image_format("rg16f")]] RWTexture2D<float2> Out;
[[vk::binding(4, 0)]] [[vk::image_format("r8")]] RWTexture2D<float> Mask;
struct ConstantsVk // the 48 bytes of the D3D12 root constants, as push constants (a round = one vkCmdPushConstants)
{
    float2 MvScale;
    uint Cur;
    uint Prev;
    float Knee0;
    float Knee1;
    float Gain;
    uint Radius;
    uint Stamp;
    uint Round;
    uint KeepStale;
    uint LinearDepth;
};
[[vk::push_constant]] ConstantsVk PC;
#define MvScale PC.MvScale
#define Cur PC.Cur
#define Prev PC.Prev
#define Knee0 PC.Knee0
#define Knee1 PC.Knee1
#define Gain PC.Gain
#define Radius PC.Radius
#define Stamp PC.Stamp
#define LinearDepth PC.LinearDepth
#else
Texture2D<float> Depth : register(t0);
Texture2D<float2> Motion : register(t1);
StructuredBuffer<float4> Fits : register(t2);
RWTexture2D<float2> Out : register(u1);
RWTexture2D<float> Mask : register(u3); // DLSS's current-colour bias: 1 where the game's vector is not the camera's

cbuffer Constants : register(b0)
{
    float2 MvScale;
    uint Cur;
    uint Prev;
    float Knee0;
    float Knee1;
    float Gain;
    uint Radius;
    uint Stamp;
    uint2 Unused;
    uint LinearDepth; // 1 = the depth is linear: 1/depth (FitDepth)
};
#endif

static const float KZ = 100.0;

bool Finite(float v) { return (asuint(v) & 0x7F800000u) != 0x7F800000u; }
// The fit and the compose work on hardware depth (z ~ 1/w: a projective map in (x, y, z)). A game that hands DLSS
// LINEAR depth d (RR's DLSS.Use.HW.Depth = 0: Hogwarts Legacy, 22873 fits and none accepted) gets 1/d,
// which is again ~ 1/w; larger is nearer either way, as the compose's "nearest" assumes. 0 = no depth there.
// 10/d, not 1/d: of the size of hardware depth for centimetres (Unreal's near plane is 10), so the solve's pull
// toward its start does not bend the depth column (fit12_test: 0.10 px residual on 1/d, see case 7).
float FitDepth(float d) { return LinearDepth != 0 ? (Finite(d) && d > 0.0 ? 10.0 / d : 0.0) : d; }

float3 Apply(uint slot, float2 xy, float z)
{
    const float4 p = float4(xy, z * KZ, 1.0);
    const float wq = dot(Fits[slot * 5 + 2], p);
    return float3(dot(Fits[slot * 5 + 0], p) / wq, dot(Fits[slot * 5 + 1], p) / wq, z / wq);
}

[numthreads(8, 8, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    uint w, h, dw, dh;
    Out.GetDimensions(w, h);
    Depth.GetDimensions(dw, dh);
    if (id.x >= w || id.y >= h)
        return;
    const float2 size = float2(w, h);
    const float2 uv = (float2(id.xy) + 0.5) / size;
    const float2 xy = float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    const int2 dat = int2(min(uint2(uv * float2(dw, dh)), uint2(dw - 1, dh - 1)));
    float z = FitDepth(Depth[dat]);
    if (!Finite(z))
        z = 0.0;
    const float2 raw = Motion[id.xy];
    const bool gameOk = Finite(raw.x) && Finite(raw.y);
    const float2 game = gameOk ? raw * MvScale : float2(0, 0);
    // A slot is usable once it has held an accepted fit. A rejected fit leaves the eye's last accepted one in its slot,
    // and that is used however long the streak: the game's own vectors (the D3D11 path's fallback after 2) point at the other
    // eye, and in Crimson Desert 20 such frames in a row were a burst of the pixelation per-eye history shows with them.
    if (Fits[Cur * 5 + 3].z < 0.5 || Fits[Prev * 5 + 3].z < 0.5)
    {
        Out[id.xy] = gameOk ? raw : float2(0, 0);
        Mask[id.xy] = 0.0;
        return;
    }
    const float2 toPx = float2(0.5, -0.5) * size;

    // Camera: this Present to the previous (the other eye), then on to the one before (this eye again).
    const float3 c1 = Apply(Cur, xy, z);
    const float3 c2 = Apply(Prev, c1.xy, c1.z);
    const float2 sameEye = (c2.xy - xy) * toPx;

    // Objects: what the game's vector has beyond camera motion to the other eye, judged at this depth and at the
    // nearest depth around it (the field is dilated), only where this pixel is itself that nearest surface.
    float zn = z;
    const int r = (int) (Radius & 0xFF); // the high byte carries the object-motion mode
    for (int dy = -r; dy <= r; ++dy)
        for (int dx = -r; dx <= r; ++dx)
        {
            const float zq = FitDepth(Depth[clamp(dat + int2(dx, dy), int2(0, 0), int2(dw - 1, dh - 1))]);
            if (Finite(zq))
                zn = max(zn, zq);
        }
    const float2 resHere = game - (c1.xy - xy) * toPx;
    const float2 resNear = game - (Apply(Cur, xy, zn).xy - xy) * toPx;
    float2 res = dot(resNear, resNear) < dot(resHere, resHere) ? resNear : resHere;
    if (!gameOk)
        res = float2(0, 0);
    // Mode 3 (Radius's high byte; UpscalerSameEyeObjectMotion): the residual averaged over 3x3 taps, each against the
    // camera at its own depth. The fit's own residual (~0.09 px in one D3D12 game) is noise that averages away; a small
    // coherent motion (a face, a distant car) does not, so the knee can sit lower than the noise.
    const float2 resPoint = res;
    float2 resAvg = res;
    const uint objectMode = (Radius >> 8) & 0xFF;
    const uint maskMode = (Radius >> 16) & 0xFF;
    if (objectMode == 3 || maskMode == 2)
    {
        float2 sum = float2(0, 0);
        float n = 0.0;
        for (int ty = -1; ty <= 1; ++ty)
            for (int tx = -1; tx <= 1; ++tx)
            {
                const int2 at = clamp(int2(id.xy) + int2(tx, ty), int2(0, 0), int2(w - 1, h - 1));
                const float2 rawq = Motion[at];
                const int2 datq = int2(min(uint2((float2(at) + 0.5) / size * float2(dw, dh)), uint2(dw - 1, dh - 1)));
                const float zq = FitDepth(Depth[datq]);
                if (!Finite(rawq.x) || !Finite(rawq.y) || !Finite(zq))
                    continue;
                const float2 uvq = (float2(at) + 0.5) / size;
                const float2 xyq = float2(uvq.x * 2.0 - 1.0, 1.0 - uvq.y * 2.0);
                sum += rawq * MvScale - (Apply(Cur, xyq, zq).xy - xyq) * toPx;
                n += 1.0;
            }
        resAvg = n > 0.0 ? sum / n : float2(0, 0);
        if (objectMode == 3)
            res = resAvg;
    }
    const float nearest = z >= zn * 0.9 ? 1.0 : 0.0;
    const float knee = smoothstep(Knee0, Knee1, length(res)) * nearest;
    float2 result = sameEye + Gain * knee * res;
    if (!Finite(result.x) || !Finite(result.y))
        result = float2(0, 0);
    Out[id.xy] = result / MvScale;
    // The bias mask: how surely this pixel moves by itself. Mode 1 from the point residual, mode 2 from the average.
    const float maskRes = maskMode == 2 ? length(resAvg) : length(resPoint);
    const float mask = maskMode == 1 ? smoothstep(0.1, 0.4, maskRes) : maskMode == 2 ? smoothstep(0.03, 0.15, maskRes) : 0.0;
    Mask[id.xy] = gameOk ? mask * nearest : 0.0;
}
)";
