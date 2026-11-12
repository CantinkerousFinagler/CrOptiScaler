#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

// The foveal crop's box along one axis, VERBATIM from PerEyeDlss_Dx12.cpp (CropAxis; kept in step by hand --
// a change there is a change here), so the Vulkan port (FoveaCrop_Vk) places its box exactly as D3D12 does and the
// offline test (vk_fovea_test) checks the maths without the DLL.
//
// R = the render subrect's size, O = the output's, f = the wanted fraction. `snap` asks for an EXACT box: a render width
// that is a whole multiple of R/gcd(R, O) and an output width of the same multiple of O/gcd, centred on whole pixels at
// both ends (the same ratio as the eye); otherwise an approximate one (edges up to half an output pixel off). rb/rw
// are render pixels relative to the render subrect's origin, ob/ow output pixels.
namespace FoveaBox
{
inline unsigned int Gcd(unsigned int a, unsigned int b)
{
    while (b != 0)
    {
        const unsigned int t = a % b;
        a = b;
        b = t;
    }
    return a;
}

inline bool CropAxis(unsigned int R, unsigned int O, float f, bool snap, uint32_t& rb, uint32_t& rw, uint32_t& ob,
                     uint32_t& ow, bool& exact)
{
    if (R == 0 || O < R || f <= 0.05f || f >= 0.95f)
        return false;
    const unsigned int g = Gcd(R, O), ur = R / g, uo = O / g;
    float bestDiff = 1e9f;
    for (unsigned int k = 1; snap && k < g; ++k)
    {
        const unsigned int w = k * ur;
        if ((R - w) % (2 * ur) != 0)
            continue;
        const float diff = std::abs((float) w / (float) R - f);
        if (diff < bestDiff && diff <= 0.15f)
        {
            bestDiff = diff;
            rw = w;
            rb = (R - w) / 2;
            ow = k * uo;
            ob = (rb / ur) * uo;
        }
    }
    exact = bestDiff < 1e9f;
    if (exact)
        return true;
    rw = ((unsigned int) std::lround(f * (float) R)) & ~1u;
    if (!snap)
    {
        const long want = (long) rw, span = std::max(2L, (long) R / 100);
        double bestErr = 1e9;
        for (long w = std::max(16L, want - span) & ~1L; w <= std::min((long) R, want + span); w += 2)
        {
            const double o = (double) w * O / R, err = std::abs(o - std::round(o)) + 1e-4 * std::abs(w - want);
            if (err < bestErr)
            {
                bestErr = err;
                rw = (uint32_t) w;
            }
        }
    }
    rb = (R - rw) / 2;
    ow = (unsigned int) std::lround((double) rw * O / R);
    ob = (unsigned int) std::lround((double) rb * O / R);
    return rw >= 16;
}

// Both axes, centred: the crop's width fraction f and height fraction fh (0 = f).
struct Rect
{
    uint32_t rx = 0, ry = 0, rw = 0, rh = 0; // render pixels, relative to the render subrect's origin
    uint32_t ox = 0, oy = 0, ow = 0, oh = 0; // output pixels
    bool exact = false;
};

// Field by field, NEVER memcmp: the struct has padding after `exact`, and a Rect made on the stack carries
// whatever lay there -- byte-compared, two equal boxes read as different, and the Vulkan periphery's feature was
// made, retired and asked again 859 times in 20 s until the game crashed.
inline bool Same(const Rect& a, const Rect& b)
{
    return a.rx == b.rx && a.ry == b.ry && a.rw == b.rw && a.rh == b.rh && a.ox == b.ox && a.oy == b.oy &&
           a.ow == b.ow && a.oh == b.oh && a.exact == b.exact;
}

inline bool Plan(unsigned int renderW, unsigned int renderH, unsigned int outW, unsigned int outH, float f, float fh,
                 bool snap, Rect& r)
{
    bool exactX = false, exactY = false;
    if (fh <= 0.0f)
        fh = f;
    if (!CropAxis(renderW, outW, f, snap, r.rx, r.rw, r.ox, r.ow, exactX) ||
        !CropAxis(renderH, outH, fh, snap, r.ry, r.rh, r.oy, r.oh, exactY))
        return false;
    r.exact = exactX && exactY;
    return true;
}
} // namespace FoveaBox
