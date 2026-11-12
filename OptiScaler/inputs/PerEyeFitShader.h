#pragma once

#include <d3d11.h>

// The fit of PerEyeDlss_Dx11 (see there), in a header of its own so that the offline test
// (tools/xr_observer/test/fit_test) compiles and runs this very shader, by this very sequence of dispatches, on
// made-up depth and motion.
//
// The fit runs WIDE, one sample per thread: as one group of 64 threads, each walking 64 samples through
// five rounds, it was most of the 0.6-1.0 ms that per eye cost on the D3D11 path (Crimson Desert's D3D12 fit measured the
// same ~1 ms in that form and ~0.1 ms wide). One source, three stages picked by a macro:
//   FIT_INIT        1 group: the start (this eye's last accepted fit, or no motion) and round 0
//   FIT_ACCUMULATE  64 groups: each weighs its 64 samples against the solution of the round before and writes one
//                   partial set of normal equations (rounds 0-3), or of the measure (round 4)
//   FIT_SOLVE       1 group: sums the partials and solves (rounds 0-3) or writes the verdict (round 4); counts the round
// Same maths and the same order of every sum as the one group had: an accepted fit is the same bits, and so is all
// that is kept of a rejected one (fit_test, case 6, says what may differ and why).
// KEEP_STALE_START is the test's control only: the start as it was before.
// FIT_SHADER_WITH_ONE_GROUP (the test only) keeps the shader as it was in one group, to compare and to time against.

// Work (floats): 64 x 80 partial sums, the solution (12), the start (11), whether there is a start, the round.
constexpr UINT kFitWorkFloats = 64 * 80 + 32;

constexpr char kFitShader[] = R"(
Texture2D<float> Depth : register(t0);
Texture2D<float2> Motion : register(t1);
RWStructuredBuffer<float4> Fits : register(u0);
RWStructuredBuffer<float> Work : register(u1);

cbuffer Constants : register(b0)
{
    float2 MvScale;
    uint Cur;
    uint Prev;
    float Knee0;
    float Knee1;
    float Gain;
    uint Radius;
};

static const float KZ = 100.0;
static const uint kSol = 64 * 80;
static const uint kInit = kSol + 12;
static const uint kHave = kInit + 11;
static const uint kRound = kHave + 1;

uint Idx(uint i, uint j) { return i * 11 - (i * (i - 1)) / 2 + (j - i); }

#if defined(FIT_INIT)

[numthreads(1, 1, 1)]
void CSMain()
{
    // Start from this eye's last accepted fit (it maps the same pair of eyes, two Presents earlier), or from no
    // motion. An unweighted first round let the player character's large near motion pull the solve into the
    // wrong answer: 3 of 55 rejected running frames recovered that way offline, 55 of 55 from the previous fit.
    // A start that has been rejected twice in a row is GIVEN UP on every second try from then on: the weights
    // are zero past 40 px of the start, so a start far from the motion of now rejects every sample, the solve
    // fails, the start stays (only an accepted fit replaces it) and the next fit fails the same way. In one D3D11 game
    // that locked ONE eye out for 7 to 15 s at a time (300 of 600 fits "solve failed"), and that
    // eye had the game's other-eye vectors meanwhile. Every second try, not every try: the old start is the
    // better one when the picture itself was the fault.
    const bool stored = Fits[Cur * 5 + 3].z > 0.5;
    const uint streak = (uint) max(Fits[Cur * 5 + 4].y, 0.0);
#ifdef KEEP_STALE_START
    const bool have = stored; // the offline test's control: the shader as it was
#else
    const bool have = stored && !(streak >= 2 && (streak & 1) == 0);
#endif
    const float4 r0 = have ? Fits[Cur * 5 + 0] : float4(1, 0, 0, 0);
    const float4 r1 = have ? Fits[Cur * 5 + 1] : float4(0, 1, 0, 0);
    const float4 r2 = have ? Fits[Cur * 5 + 2] : float4(0, 0, 0, 1);
    const float sol[12] = { r0.x, r0.y, r0.z, r0.w, r1.x, r1.y, r1.z, r1.w, r2.x, r2.y, r2.z, 1.0 };
    for (uint k = 0; k < 12; ++k)
        Work[kSol + k] = sol[k];
    for (uint k0 = 0; k0 < 11; ++k0)
        Work[kInit + k0] = sol[k0];
    Work[kHave] = have ? 1.0 : 0.0;
    Work[kRound] = 0.0;
}

#elif defined(FIT_ACCUMULATE)

groupshared float S[64][80];

[numthreads(64, 1, 1)]
void CSMain(uint gi : SV_GroupIndex, uint3 g : SV_GroupID)
{
    for (uint c0 = 0; c0 < 80; ++c0)
        S[gi][c0] = 0;
    uint w, h, dw, dh;
    Motion.GetDimensions(w, h);
    Depth.GetDimensions(dw, dh);
    const float2 size = float2(w, h);
    // Four weighted rounds from the start -- 2, 0.5, 0.3, 0.3 px from the eye's last fit, or 32, 2, 0.5, 0.3 px
    // from no motion -- then a round that only measures.
    const uint it = (uint) Work[kRound];
    const float tau = Work[kHave] > 0.5 ? (it == 0 ? 2.0 : it == 1 ? 0.5 : 0.3)
                                        : (it == 0 ? 32.0 : it == 1 ? 2.0 : it == 2 ? 0.5 : 0.3);
    // The group is the grid's row, the thread its column: one sample each.
    const uint s = g.x * 64 + gi;
    const float2 uv = (float2(s % 64, s / 64) + 0.5) / 64.0;
    const float z = Depth[min(uint2(uv * float2(dw, dh)), uint2(dw - 1, dh - 1))];
    if (z > 0.0)
    {
        const float2 mv = Motion[min(uint2(uv * size), uint2(w - 1, h - 1))] * MvScale;
        const float x = uv.x * 2.0 - 1.0;
        const float y = 1.0 - uv.y * 2.0;
        const float xp = x + mv.x * 2.0 / size.x;
        const float yp = y - mv.y * 2.0 / size.y;
        const float zz = z * KZ;
        const float4 p = float4(x, y, zz, 1.0);
        const float wq = dot(float4(Work[kSol + 8], Work[kSol + 9], Work[kSol + 10], 1.0), p);
        const float ex =
            (dot(float4(Work[kSol + 0], Work[kSol + 1], Work[kSol + 2], Work[kSol + 3]), p) / wq - xp) * size.x * 0.5;
        const float ey =
            (dot(float4(Work[kSol + 4], Work[kSol + 5], Work[kSol + 6], Work[kSol + 7]), p) / wq - yp) * size.y * 0.5;
        const float e = sqrt(ex * ex + ey * ey);
        const float wgt = e < 20.0 * tau ? 1.0 / max(e / tau, 1.0) : 0.0;
        if (it >= 4)
        {
            S[gi][79] = 1.0;
            if (e < 1.0)
            {
                S[gi][77] = e;
                S[gi][78] = 1.0;
            }
        }
        else
        {
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
        Work[g.x * 80 + col] = sum;
    }
}

#else // FIT_SOLVE

groupshared float T[80];

[numthreads(64, 1, 1)]
void CSMain(uint gi : SV_GroupIndex)
{
    for (uint col = gi; col < 80; col += 64)
    {
        float sum = 0;
        for (uint r = 0; r < 64; ++r)
            sum += Work[r * 80 + col];
        T[col] = sum;
    }
    GroupMemoryBarrierWithGroupSync();
    if (gi == 0)
    {
        const uint it = (uint) Work[kRound];
        if (it < 4)
        {
            // Cholesky on the normal equations, with a small ridge.
            float L[11][11];
            float trace = 0;
            for (uint i = 0; i < 11; ++i)
                trace += T[Idx(i, i)];
            const float ridge = 1e-7 * trace / 11.0;
            // A gentle pull toward the starting fit: keeps the system solvable when the grid is nearly one plane (a
            // wall or the ground fills the view and the depth column becomes degenerate), and changes nothing measurable
            // otherwise (offline, same acceptance with and without).
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
                float v = T[66 + a1] + prior * Work[kInit + a1];
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
            for (uint k = 0; k < 11; ++k)
                Work[kSol + k] = sol[k];
            Work[kSol + 11] = 1.0;
            Fits[2 * 5 + 3] = float4(0, 0, ok ? 1.0 : 0.0, 0);
            Work[kRound] = (float) (it + 1);
        }
        else
        {
            // This Present's fit goes to the scratch slot 2 with its reason (0 accepted, 1 the solve failed, 2 too
            // few samples with depth, 3 too few samples within 1 px). Only an accepted fit replaces this parity's
            // slot; a rejected one keeps the last accepted fit of this eye (two Presents old) and counts the streak.
            // Too few samples is asked FIRST: a picture without depth (a menu without the world behind it) has
            // nothing to solve either, and as "the solve failed" it read like a fault of the fit (the D3D11 path:
            // 600 of 600 in both eyes for half a minute). What is accepted does not change.
            const float n = T[79];
            const float inliers = T[78];
            const bool solved = Fits[2 * 5 + 3].z > 0.5;
            const float reason = n <= 256.0 ? 2.0 : !solved ? 1.0 : inliers <= 0.5 * n ? 3.0 : 0.0;
            const float4 r0 = float4(Work[kSol + 0], Work[kSol + 1], Work[kSol + 2], Work[kSol + 3]);
            const float4 r1 = float4(Work[kSol + 4], Work[kSol + 5], Work[kSol + 6], Work[kSol + 7]);
            const float4 r2 = float4(Work[kSol + 8], Work[kSol + 9], Work[kSol + 10], 1.0);
            const float4 st = float4(inliers > 0 ? T[77] / inliers : 99.0, n > 0 ? inliers / n : 0.0,
                                     reason == 0.0 ? 1.0 : 0.0, n);
            Fits[2 * 5 + 0] = r0;
            Fits[2 * 5 + 1] = r1;
            Fits[2 * 5 + 2] = r2;
            Fits[2 * 5 + 3] = st;
            Fits[2 * 5 + 4] = float4(reason, 0, 0, 0);
            if (reason == 0.0)
            {
                Fits[Cur * 5 + 0] = r0;
                Fits[Cur * 5 + 1] = r1;
                Fits[Cur * 5 + 2] = r2;
                Fits[Cur * 5 + 3] = st;
                Fits[Cur * 5 + 4] = float4(0, 0, 0, 0);
            }
            else
            {
                const float4 keep = Fits[Cur * 5 + 4];
                Fits[Cur * 5 + 4] = float4(reason, keep.y + 1.0, 0, 0);
            }
        }
    }
}

#endif
)";

// One fit: INIT, then five rounds of ACCUMULATE (64 groups) and SOLVE. The caller has bound the constants at b0,
// depth and motion at t0 and t1, and the fits and the work buffer at u0 and u1.
inline void DispatchFit(ID3D11DeviceContext* ctx, ID3D11ComputeShader* init, ID3D11ComputeShader* accumulate,
                        ID3D11ComputeShader* solve)
{
    ctx->CSSetShader(init, nullptr, 0);
    ctx->Dispatch(1, 1, 1);
    for (int round = 0; round <= 4; ++round)
    {
        ctx->CSSetShader(accumulate, nullptr, 0);
        ctx->Dispatch(64, 1, 1);
        ctx->CSSetShader(solve, nullptr, 0);
        ctx->Dispatch(1, 1, 1);
    }
}

#ifdef FIT_SHADER_WITH_ONE_GROUP
// The fit as it used to be, one dispatch of one group (the start given up after two rejections included).
constexpr char kFitShaderOneGroup[] = R"(
Texture2D<float> Depth : register(t0);
Texture2D<float2> Motion : register(t1);
RWStructuredBuffer<float4> Fits : register(u0);

cbuffer Constants : register(b0)
{
    float2 MvScale;
    uint Cur;
    uint Prev;
    float Knee0;
    float Knee1;
    float Gain;
    uint Radius;
};

static const float KZ = 100.0;
groupshared float S[64][80];
groupshared float Sol[12];
groupshared float Init[11];
groupshared float Have;

uint Idx(uint i, uint j) { return i * 11 - (i * (i - 1)) / 2 + (j - i); }

[numthreads(64, 1, 1)]
void CSMain(uint gi : SV_GroupIndex)
{
    uint w, h, dw, dh;
    Motion.GetDimensions(w, h);
    Depth.GetDimensions(dw, dh);
    const float2 size = float2(w, h);
    for (uint c0 = 0; c0 < 80; ++c0)
        S[gi][c0] = 0;
    if (gi == 0)
    {
        const bool stored = Fits[Cur * 5 + 3].z > 0.5;
        const uint streak = (uint) max(Fits[Cur * 5 + 4].y, 0.0);
        const bool have = stored && !(streak >= 2 && (streak & 1) == 0);
        const float4 r0 = have ? Fits[Cur * 5 + 0] : float4(1, 0, 0, 0);
        const float4 r1 = have ? Fits[Cur * 5 + 1] : float4(0, 1, 0, 0);
        const float4 r2 = have ? Fits[Cur * 5 + 2] : float4(0, 0, 0, 1);
        Sol[0] = r0.x; Sol[1] = r0.y; Sol[2] = r0.z; Sol[3] = r0.w;
        Sol[4] = r1.x; Sol[5] = r1.y; Sol[6] = r1.z; Sol[7] = r1.w;
        Sol[8] = r2.x; Sol[9] = r2.y; Sol[10] = r2.z; Sol[11] = 1.0;
        for (uint k0 = 0; k0 < 11; ++k0)
            Init[k0] = Sol[k0];
        Have = have ? 1.0 : 0.0;
    }
    GroupMemoryBarrierWithGroupSync();

    float valid = 0;
    for (uint it = 0; it < 5; ++it)
    {
        const float tau = Have > 0.5 ? (it == 0 ? 2.0 : it == 1 ? 0.5 : 0.3)
                                     : (it == 0 ? 32.0 : it == 1 ? 2.0 : it == 2 ? 0.5 : 0.3);
        for (uint k = 0; k < 64; ++k)
        {
            const uint s = gi * 64 + k;
            const float2 uv = (float2(s % 64, s / 64) + 0.5) / 64.0;
            const float z = Depth[min(uint2(uv * float2(dw, dh)), uint2(dw - 1, dh - 1))];
            if (!(z > 0.0))
                continue;
            const float2 mv = Motion[min(uint2(uv * size), uint2(w - 1, h - 1))] * MvScale;
            const float x = uv.x * 2.0 - 1.0;
            const float y = 1.0 - uv.y * 2.0;
            const float xp = x + mv.x * 2.0 / size.x;
            const float yp = y - mv.y * 2.0 / size.y;
            const float zz = z * KZ;
            float wgt = 1.0;
            float e = 0.0;
            {
                const float4 p = float4(x, y, zz, 1.0);
                const float wq = dot(float4(Sol[8], Sol[9], Sol[10], 1.0), p);
                const float ex = (dot(float4(Sol[0], Sol[1], Sol[2], Sol[3]), p) / wq - xp) * size.x * 0.5;
                const float ey = (dot(float4(Sol[4], Sol[5], Sol[6], Sol[7]), p) / wq - yp) * size.y * 0.5;
                e = sqrt(ex * ex + ey * ey);
                wgt = e < 20.0 * tau ? 1.0 / max(e / tau, 1.0) : 0.0;
                if (it == 4)
                {
                    S[gi][79] += 1.0;
                    if (e < 1.0)
                    {
                        S[gi][77] += e;
                        S[gi][78] += 1.0;
                    }
                    continue;
                }
            }
            const float rx[11] = { x, y, zz, 1.0, 0, 0, 0, 0, -xp * x, -xp * y, -xp * zz };
            const float ry[11] = { 0, 0, 0, 0, x, y, zz, 1.0, -yp * x, -yp * y, -yp * zz };
            [unroll] for (uint i = 0; i < 11; ++i)
            {
                [unroll] for (uint j = i; j < 11; ++j)
                    S[gi][Idx(i, j)] += wgt * (rx[i] * rx[j] + ry[i] * ry[j]);
                S[gi][66 + i] += wgt * (rx[i] * xp + ry[i] * yp);
            }
        }
        GroupMemoryBarrierWithGroupSync();
        for (uint col = gi; col < 80; col += 64)
        {
            float sum = 0;
            for (uint r = 0; r < 64; ++r)
                sum += S[r][col];
            S[0][col] = sum;
        }
        GroupMemoryBarrierWithGroupSync();
        if (gi == 0 && it < 4)
        {
            float L[11][11];
            float trace = 0;
            for (uint i = 0; i < 11; ++i)
                trace += S[0][Idx(i, i)];
            const float ridge = 1e-7 * trace / 11.0;
            const float prior = 1e-4 * trace / 11.0;
            bool ok = trace > 0;
            for (uint a = 0; a < 11; ++a)
            {
                for (uint b = 0; b <= a; ++b)
                {
                    float v = S[0][Idx(b, a)] + (a == b ? ridge + prior : 0.0);
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
                float v = S[0][66 + a1] + prior * Init[a1];
                for (uint q = 0; q < a1; ++q)
                    v -= L[a1][q] * yv[q];
                yv[a1] = v / L[a1][a1];
            }
            for (int a2 = 10; a2 >= 0; --a2)
            {
                float v = yv[a2];
                for (uint q = a2 + 1; q < 11; ++q)
                    v -= L[q][a2] * Sol[q];
                Sol[a2] = v / L[a2][a2];
            }
            Sol[11] = 1.0;
            valid = ok ? 1.0 : 0.0;
            Fits[2 * 5 + 3] = float4(0, 0, valid, 0);
        }
        GroupMemoryBarrierWithGroupSync();
        if (it < 4)
        {
            for (uint c1 = 0; c1 < 80; ++c1)
                S[gi][c1] = 0;
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (gi == 0)
    {
        const float n = S[0][79];
        const float inliers = S[0][78];
        const bool solved = Fits[2 * 5 + 3].z > 0.5;
        const float reason = !solved ? 1.0 : n <= 256.0 ? 2.0 : inliers <= 0.5 * n ? 3.0 : 0.0;
        const float4 r0 = float4(Sol[0], Sol[1], Sol[2], Sol[3]);
        const float4 r1 = float4(Sol[4], Sol[5], Sol[6], Sol[7]);
        const float4 r2 = float4(Sol[8], Sol[9], Sol[10], 1.0);
        const float4 st = float4(inliers > 0 ? S[0][77] / inliers : 99.0, n > 0 ? inliers / n : 0.0,
                                 reason == 0.0 ? 1.0 : 0.0, n);
        Fits[2 * 5 + 0] = r0;
        Fits[2 * 5 + 1] = r1;
        Fits[2 * 5 + 2] = r2;
        Fits[2 * 5 + 3] = st;
        Fits[2 * 5 + 4] = float4(reason, 0, 0, 0);
        if (reason == 0.0)
        {
            Fits[Cur * 5 + 0] = r0;
            Fits[Cur * 5 + 1] = r1;
            Fits[Cur * 5 + 2] = r2;
            Fits[Cur * 5 + 3] = st;
            Fits[Cur * 5 + 4] = float4(0, 0, 0, 0);
        }
        else
        {
            const float4 keep = Fits[Cur * 5 + 4];
            Fits[Cur * 5 + 4] = float4(reason, keep.y + 1.0, 0, 0);
        }
    }
}
)";
#endif
