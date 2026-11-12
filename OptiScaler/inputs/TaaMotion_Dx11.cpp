#include "pch.h"
#include "TaaMotion_Dx11.h"

#include "Config.h"
#include "Logger.h"
#include "inputs/FinalImageNr_Dx11.h"
#include "inputs/FrameCapture_Dx11.h"
#include "inputs/TaaDlaa_Dx11.h"

#include <d3dcompiler.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <format>
#include <string>
#include <vector>

namespace
{
// One pass: unpacks this Present's motion (to the previous Present, the other eye under alternate-eye VR) into
// Motion, and chains it with the previous Present's decoded motion into Composed (to two Presents back, the same
// eye). The previous motion is read where this Present's points (nearest texel, clamped to the image).
//
// Occlusion: beside a nearer object the other eye can't see the background this eye sees (the eyes look past the
// object at different angles), so the chained step lands on the object and would take its motion. With both
// Presents' depth (reversed Z: larger is nearer), a landing point clearly nearer than this pixel is taken as
// occluded, and the previous motion is read instead from the closest texel along the row (eye parallax is
// horizontal) at this pixel's depth, if one is within reach, a few texels further out (past the motion's halo).
//
// Matrix motion (UseMatrix, both Presents' view constants copied): the chained step above still reads the other
// eye's image, which sees other surfaces at silhouettes. UE4's view uniform buffer holds, at +1728, a row-vector
// matrix taking (ndc, depth, 1) of this Present to clip space of the previous Present (the other eye, parallax
// included; F7 captures fit the game's own vectors to 0.04 px). Through this Present's matrix and then the previous
// Present's, a static pixel lands exactly where it was two Presents back, without looking at the other eye's image.
// Pixels whose game vector disagrees with this Present's matrix motion by more than its rounding (~0.16 px) move by
// themselves (characters, idle animation included; vehicles, foliage) and keep the chained composition. UseMatrix 2
// also treats as static the halo around nearer moving objects: a pixel whose vector equals that of a clearly nearer texel within 3 px carries that
// object's motion, not its own.
constexpr char kDecodeShader[] = R"(
Texture2D<uint> Packed : register(t0);
Texture2D<float2> Previous : register(t1);
Texture2D<float> Depth : register(t2);
Texture2D<float> PreviousDepth : register(t3);
// The same eye's depth from two Presents back (alternate-eye rendering): where this Present's history comes from.
Texture2D<float> SameEyeDepth : register(t4);
// The pass's colour input: the same eye's from two Presents back, and this Present's (for the shadow test).
Texture2D<float4> SameEyeColour : register(t5);
Texture2D<float4> Colour : register(t6);
RWTexture2D<float2> Motion : register(u0);
RWTexture2D<float2> Composed : register(u1);
// Which path each pixel took, for the F8 classification view: 0 chained (no matrix), 1 halo (matrix chain only),
// 2 matrix (own motion below the fade), 3 matrix + own faded in, 4 blending to the chained composition, 5 chained
// while the matrix was available.
RWTexture2D<float> Class : register(u2);
// DLSS's "bias current colour" mask: 1 where this pixel's history is known to be something else (disoccluded).
RWTexture2D<float> Bias : register(u3);

cbuffer Constants : register(b0)
{
    uint HasPrevious;
    uint Occlusion;
    uint UseMatrix;
    // How much of the pixel's own (object) motion is added, 1 = as validated. 0 = the matrix alone: FC6 found its swim
    // and sky smear came entirely from this term, under game-driven camera motion only.
    float OwnScale;
    uint HasSameEyeDepth;
    float BiasStrength; // 0 = off
    float BiasNearer;   // how much nearer (reversed-Z ratio) the history's surface must be to count as a cover
    uint HasSameEyeColour;
    uint MaskMode;      // 0 off, 1 covers (depth), 2 covers + shadows (brightness), 3 TEST: the whole frame
    float ShadowDrop;   // how much darker (tonemapped luma, smoothed) the history must be to count as a moved shadow
    uint YCoCg;         // the render-resolution pass's colour is YCoCg (luma in x)
    // The object-motion knee and the chained blend, live from the Days Gone overlay (defaults = as
    // validated). Knee: own motion fades in from KneeStart + KneeRise x |camera motion| to KneeRatio x that.
    float KneeStart;
    float KneeRise;
    float KneeRatio;
    // From ChainStart to ChainEnd px of own motion the result blends over to the chained composition.
    float ChainStart;
    float ChainEnd;
    // Object motion kept on busy depth (foliage), 1 = all (as before): see where it is applied.
    float BusyScale;
    float Pad3;
    float Pad4;
    float Pad5;
};

bool FiniteF(float v) { return (asuint(v) & 0x7F800000u) != 0x7F800000u; }
// Tonemapped luma, smoothed over a 5x5 footprint (3x3 taps, 2 px apart): shadows and AO are soft, ground texture is
// not, so texture detail and jitter mostly average out and a moved shadow does not.
float SmoothLuma(Texture2D<float4> tex, int2 at, int2 size)
{
    float sum = 0;
    for (int dy = -2; dy <= 2; dy += 2)
    {
        for (int dx = -2; dx <= 2; dx += 2)
        {
            const float4 c = tex[clamp(at + int2(dx, dy), int2(0, 0), size - 1)];
            float l = YCoCg != 0 ? c.x : dot(c.rgb, float3(0.299, 0.587, 0.114));
            l = FiniteF(l) ? clamp(l, 0.0, 65504.0) : 0.0;
            sum += l / (1.0 + l);
        }
    }
    return sum / 9.0;
}
// UE4 view uniform buffers (2048 bytes) of this Present and the previous one: the matrix rows at [108]..[111].
cbuffer ThisView : register(b1)
{
    float4 A[128];
};
cbuffer PreviousView : register(b2)
{
    float4 B[128];
};

static const int kSearch = 96;
// The motion field reaches about 2 px past the depth edge (a nearer object's motion on the first background texels),
// so the motion is read this far beyond the first texel at the pixel's depth.
static const int kMargin = 3;
static const int kHalo = 3;
// The halo test's neighbour must be this much nearer in reversed-Z (about 40% closer in distance): a real silhouette.
// At 5% it fired inside moving objects themselves — a bike and its rider vary by more than that within 3 px, so the
// whole mover was held to the matrix chain and dragged its old position forward (the dark trail).
static const float kHaloNearer = 1.4;

// Same surface within 5% of depth (sky, at 0, only matches sky).
bool SameDepth(float a, float b) { return abs(a - b) <= a * 0.05 + 1e-6; }

float2 Unpack(uint v, float2 size)
{
    if (v == 0)
        return 0;
    const float2 packed = float2(v & 0x3FFF, (v >> 14) & 0x1FFF) * float2(1.0 / 16383.0, 1.0 / 8191.0);
    return (packed - 0.5) * 0.25 * size;
}

float4 Transform(float4 p, float4 r0, float4 r1, float4 r2, float4 r3)
{
    return p.x * r0 + p.y * r1 + p.z * r2 + p.w * r3;
}

[numthreads(8, 8, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    uint w, h;
    Motion.GetDimensions(w, h);
    if (id.x >= w || id.y >= h)
        return;
    const float2 size = float2(w, h);
    const float2 offset = Unpack(Packed[id.xy], size);
    Motion[id.xy] = offset;
    Bias[id.xy] = MaskMode == 3 ? BiasStrength : 0;

    // Matrix result and how much of the chained composition to blend in (1: chained only).
    float2 matrixResult = 0;
    float chainWeight = 1.0;
    float classCode = 0;
    if (UseMatrix != 0 && HasPrevious != 0)
    {
        const float z = Depth[id.xy];
        const float2 ndc = float2((id.x + 0.5) / w * 2.0 - 1.0, 1.0 - (id.y + 0.5) / h * 2.0);
        const float4 c1 = Transform(float4(ndc, z, 1.0), A[108], A[109], A[110], A[111]);
        const float4 p1 = c1 / c1.w;
        const float2 matrixOffset = (p1.xy - ndc) * float2(0.5, -0.5) * size;
        // The pixel's own motion (object motion; the game's vector minus the camera's). Static pixels show only noise
        // (median 0.04 px, tail to ~0.14 px); idle animation (faces, hair) 0.1-1 px per Present; real movers more. A
        // hard threshold anywhere in that range made pixels flip between paths from Present to Present (strobing,
        // ghosting when the camera moves), so the own motion fades in instead: none below 0.1 px, full from 0.3 px,
        // added twice (two Presents back) on top of the matrix motion; from 1 px to 2 px the result blends over to
        // the chained composition, which follows fast movers best.
        float2 own = offset - matrixOffset;
        const float ownLength = length(own);
        // The game's vectors only agree with the matrix to a noise floor that grows with the camera's motion (F7
        // captures while riding: median 0.04 px everywhere, but the 90th percentile rises from 0.11 px
        // when still to 0.18 px at 160 px of camera motion). A fixed 0.1 px start therefore treated a fifth to a
        // quarter of the near ground as moving at speed and added its noise twice — the dark trail. The start scales
        // with the motion instead: 0.1 px standing (idle faces keep their motion), ~0.34 px at 160 px/Present, which
        // puts 90% of a riding frame back on pure matrix motion (was 68%).
        // Idle NPCs smear with DLSS but not with the game's TAA, the same in both eyes, and Object motion
        // x0/x1/x2 changed nothing while "Chained" (no knee) was better: idle motion sits mostly under this knee, and
        // pixels either side of it make a patchy field inside a character. Hence live knee and chain controls.
        const float fadeStart = KneeStart + KneeRise * length(matrixOffset);
        const float fadeEnd = max(KneeRatio * fadeStart, fadeStart + 1e-3);
        bool halo = false;
        if (ownLength > fadeStart && UseMatrix >= 2)
        {
            for (int dy = -kHalo; dy <= kHalo && !halo; ++dy)
            {
                for (int dx = -kHalo; dx <= kHalo; ++dx)
                {
                    const int2 at = clamp(int2(id.xy) + int2(dx, dy), int2(0, 0), int2(w - 1, h - 1));
                    if (Depth[at] > z * kHaloNearer + 1e-5 && all(abs(Unpack(Packed[at], size) - offset) < 0.5))
                    {
                        halo = true;
                        break;
                    }
                }
            }
        }
        const float4 c2 = Transform(p1, B[108], B[109], B[110], B[111]);
        const float2 matrixComposed = (c2.xy / c2.w - ndc) * float2(0.5, -0.5) * size;
        // Disocclusion (the dark trail behind the bike): a static pixel's history is where the matrix puts
        // it two Presents back -- in the same eye. If that eye's depth there was NEARER than this point was then,
        // something covered it (the wheel, its shadow's caster, a tree against the sky) and the history shows that
        // instead. The motion is right; DLSS is told to trust the current colour there. Pixels with their own motion
        // are left alone: their history position is not the matrix one.
        // Moved shadows and AO (the trail is worst where there are shadows): the bike's shadow
        // travels with it over ground that stays visible, so no depth test sees it. Its history there is DARKER than
        // now, smoothly: that is flagged too (mode 2).
        bool disoccluded = false;
        bool shadowed = false;
        if (MaskMode == 1 || MaskMode == 2)
        {
            if (BiasStrength > 0 && (halo || ownLength <= fadeStart) && c2.w > 1e-6)
            {
                const float2 past = float2(id.xy) + 0.5 + matrixComposed;
                if (all(past >= 0) && all(past < size))
                {
                    if (HasSameEyeDepth != 0)
                        disoccluded = SameEyeDepth[int2(past)] > c2.z / c2.w * BiasNearer + 1e-5;
                    if (!disoccluded && MaskMode == 2 && HasSameEyeColour != 0)
                    {
                        const int2 isize = int2(w, h);
                        const float now = SmoothLuma(Colour, int2(id.xy), isize);
                        const float before = SmoothLuma(SameEyeColour, int2(past), isize);
                        shadowed = now - before > ShadowDrop && before < now * 0.85;
                    }
                    if (disoccluded || shadowed)
                        Bias[id.xy] = BiasStrength;
                }
            }
        }
        if (halo)
        {
            Composed[id.xy] = matrixComposed;
            Class[id.xy] = disoccluded ? 6 : shadowed ? 7 : 1;
            return;
        }
        // Object motion over the two Presents. By default twice this Present's (constant velocity). UseMatrix 3:
        // this Present's plus the previous Present's own motion, read where this point of the object
        // was then (p + the game's vector), minus the camera's motion there through the previous Present's matrix.
        // Idle NPCs smeared with x2 whatever the scale or knee, while chaining them helped a lot: UE4 may animate a
        // character less often than every Present, so one Present carries the motion and the next little -- a sum
        // follows that, a doubling cannot. Only where that point shows the same surface (its depth in the previous
        // Present matches this point's): at a silhouette the lookup lands on the other side of the edge (the old
        // chained mode's shimmer), and the pixel keeps x2 instead.
        float2 ownTotal = 2.0 * own;
        bool ownChained = false;
        if (UseMatrix == 3 && Occlusion != 0)
        {
            const float2 q = float2(id.xy) + 0.5 + offset;
            if (all(q >= 0) && all(q < size))
            {
                const int2 qi = int2(q);
                const float zq = PreviousDepth[qi];
                if (SameDepth(p1.z, zq))
                {
                    const float2 ndcq = float2(q.x / w * 2.0 - 1.0, 1.0 - q.y / h * 2.0);
                    const float4 cq = Transform(float4(ndcq, zq, 1.0), B[108], B[109], B[110], B[111]);
                    if (cq.w > 1e-6)
                    {
                        const float2 matrixPrevious = (cq.xy / cq.w - ndcq) * float2(0.5, -0.5) * size;
                        ownTotal = own + (Previous[qi] - matrixPrevious);
                        ownChained = true;
                    }
                }
            }
        }
        // The knee judges the larger of the two Presents' own motion: a Present the animation skipped has almost none.
        const float kneeLength = max(ownLength, 0.5 * length(ownTotal));
        // Busy depth: Object motion x0.1 removed the last shimmer on grass and foliage but brought NPC
        // smear back. Vegetation is a clutter of depth edges (blades over ground, leaves over sky); a character's body
        // is one smooth surface. Where most of the 8 neighbours sit at another depth, the object motion is scaled by
        // BusyScale; smooth surfaces keep all of it.
        float busyWeight = 1.0;
        if (BusyScale < 1.0 && kneeLength > fadeStart)
        {
            int edges = 0;
            for (int by = -1; by <= 1; ++by)
            {
                for (int bx = -1; bx <= 1; ++bx)
                {
                    const int2 at = clamp(int2(id.xy) + int2(bx, by), int2(0, 0), int2(w - 1, h - 1));
                    edges += (bx != 0 || by != 0) && !SameDepth(z, Depth[at]) ? 1 : 0;
                }
            }
            busyWeight = lerp(1.0, BusyScale, smoothstep(1.5, 4.5, (float) edges));
        }
        const float2 withOwn =
            matrixComposed + OwnScale * busyWeight * ownTotal * smoothstep(fadeStart, fadeEnd, kneeLength);
        // The chained composition carries the game's vectors too: at scale 0 nothing of them may reach the result.
        chainWeight = smoothstep(ChainStart, max(ChainEnd, ChainStart + 1e-3), ownLength) * saturate(OwnScale);
        if (chainWeight <= 0.0)
        {
            Composed[id.xy] = withOwn;
            Class[id.xy] = disoccluded ? 6 : shadowed ? 7 : kneeLength <= fadeStart ? 2 : busyWeight < 0.5 ? 9
                                                                                         : ownChained ? 8 : 3;
            return;
        }
        matrixResult = withOwn;
        classCode = chainWeight >= 1.0 ? 5 : 4;
    }

    float2 composed = offset;
    if (HasPrevious != 0)
    {
        const int2 at = clamp(int2(floor(float2(id.xy) + 0.5 + offset)), int2(0, 0), int2(w - 1, h - 1));
        float2 previous = Previous[at];
        if (Occlusion != 0)
        {
            const float z = Depth[id.xy];
            if (PreviousDepth[at] > z * 1.05 + 1e-5)
            {
                for (int k = 1; k <= kSearch; ++k)
                {
                    const int2 left = int2(max(at.x - k, 0), at.y);
                    const int2 right = int2(min(at.x + k, (int) w - 1), at.y);
                    if (SameDepth(z, PreviousDepth[left]))
                    {
                        previous = Previous[int2(max(left.x - kMargin, 0), at.y)];
                        break;
                    }
                    if (SameDepth(z, PreviousDepth[right]))
                    {
                        previous = Previous[int2(min(right.x + kMargin, (int) w - 1), at.y)];
                        break;
                    }
                }
            }
        }
        composed += previous;
    }
    Composed[id.xy] = lerp(matrixResult, composed, chainWeight);
    Class[id.xy] = classCode;
}
)";

constexpr char kViewShader[] = R"(
Texture2D<float2> Motion : register(t0);
Texture2D<float4> Scene : register(t1);
RWTexture2D<float4> Output : register(u0);

[numthreads(8, 8, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    uint w, h;
    Output.GetDimensions(w, h);
    if (id.x >= w || id.y >= h)
        return;
    const float2 mv = Motion[id.xy];
    const float grey = 0.6 * dot(saturate(Scene[id.xy].rgb), float3(0.25, 0.5, 0.25));
    // Red = X, green = Y, grey at zero, full at 16 px per frame; the tint fades in over the first 2 px.
    const float3 tint = float3(saturate(0.5 + mv / 32.0), 0.5);
    Output[id.xy] = float4(lerp(grey.xxx, tint, saturate(length(mv) / 2.0)), 1.0);
}
)";

// The F8 classification view: which path each pixel's motion came from, over the dimmed scene.
constexpr char kClassShader[] = R"(
Texture2D<float> Class : register(t0);
Texture2D<float4> Scene : register(t1);
RWTexture2D<float4> Output : register(u0);

cbuffer Constants : register(b0)
{
    uint YCoCg; // the render-resolution pass's colour is YCoCg (luma in x), not RGB
};

[numthreads(8, 8, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    uint w, h;
    Output.GetDimensions(w, h);
    if (id.x >= w || id.y >= h)
        return;
    const float4 scene = Scene[id.xy];
    const float grey = 0.5 * saturate(YCoCg != 0 ? scene.x : dot(scene.rgb, float3(0.25, 0.5, 0.25)));
    const int code = (int) (Class[id.xy] + 0.5);
    float3 tint;
    switch (code)
    {
    case 1: tint = float3(1.0, 0.0, 1.0); break; // halo held static: magenta
    case 2: tint = float3(0.0, 0.5, 1.0); break; // matrix alone: blue
    case 3: tint = float3(0.0, 1.0, 0.0); break; // matrix + own motion: green
    case 4: tint = float3(1.0, 0.8, 0.0); break; // blending to the chained composition: amber
    case 5: tint = float3(1.0, 0.0, 0.0); break; // chained, matrix available: red
    case 6: tint = float3(0.0, 1.0, 1.0); break; // disoccluded, DLSS told to trust the current colour: cyan
    case 7: tint = float3(1.0, 1.0, 1.0); break; // a moved shadow (history darker), the same: white
    case 8: tint = float3(0.0, 0.6, 0.3); break; // matrix + object motion chained over both Presents: dark green
    case 9: tint = float3(1.0, 0.5, 0.0); break; // object motion damped on busy depth (foliage): orange
    default: tint = float3(0.6, 0.6, 0.6); break; // chained, no matrix this Present: grey
    }
    Output[id.xy] = float4(lerp(grey.xxx, tint, 0.75), 1.0);
}
)";

// All on the thread driving the immediate context and calling Present; only the view mode crosses threads.
thread_local int g_inPass = 0;

struct PassScope
{
    PassScope() { ++g_inPass; }
    ~PassScope() { --g_inPass; }
};

// 0 off, 1 decoded motion (to the other eye), 2 composed motion (to the same eye).
std::atomic<int> g_viewMode { 0 };
// Same-eye motion (F4 cycles it for testing): 0 matrix motion for static pixels, 1 (default; tested best)
// the same with the halo around moving objects taken as static, 2 chained through the other eye only
// (occlusion-aware).
std::atomic<int> g_motionMode { 1 };
// Days Gone overlay: the object-motion scale (OwnScale in the decode shader), live, not saved. 1 = as validated.
std::atomic<float> g_ownScale { 1.0f };
// The disocclusion hint (DLSS's bias-current-colour mask): its value where set (0 = off), and the cover threshold.
std::atomic<float> g_biasStrength { 1.0f };
std::atomic<float> g_biasNearer { 1.15f };
// Off by default: DLSS 310.8 ignored the mask even over the whole frame (TEST mode, no change).
std::atomic<int> g_maskMode { 0 };           // 0 off, 1 covers, 2 covers + shadows, 3 TEST whole frame
// The knee and the chained blend (TaaMotion_Dx11::Tuning), live, not saved; the decode logs each new set once.
constexpr float kTuningDefaults[TaaMotion_Dx11::kTuningCount] = { 0.1f, 0.0015f, 3.0f, 1.0f, 2.0f, 1.0f };
constexpr float kTuningMin[TaaMotion_Dx11::kTuningCount] = { 0.0f, 0.0f, 1.5f, 0.05f, 0.1f, 0.0f };
constexpr float kTuningMax[TaaMotion_Dx11::kTuningCount] = { 0.5f, 0.005f, 6.0f, 4.0f, 6.0f, 1.0f };
std::atomic<float> g_tuning[TaaMotion_Dx11::kTuningCount] = { 0.1f, 0.0015f, 3.0f, 1.0f, 2.0f, 1.0f };
float g_tuningUsed[TaaMotion_Dx11::kTuningCount] = { -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f }; // render thread only
std::atomic<float> g_shadowDrop { 0.04f };   // tonemapped luma
uint64_t g_shadowPasses = 0, g_shadowMissingColour = 0;
// Copies of the pass's colour input, ping-ponging like the depth copies: during the decode the slot about to be
// written still holds the same eye's from two Presents back. Kept at their size; a size change parks them (never
// freed, like the motion textures).
struct ColourCopy
{
    ID3D11Texture2D* texture = nullptr;
    ID3D11ShaderResourceView* srv = nullptr;
    UINT width = 0;
    UINT height = 0;
    DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;
    uint64_t present = UINT64_MAX;
};
ColourCopy g_colourRing[2];
std::vector<ColourCopy> g_parkedColour;
bool g_colourCopyFailed = false;
bool g_biasThisDispatch = false; // the decode wrote a usable mask this pass (render thread only)
uint64_t g_biasPasses = 0, g_biasMissingDepth = 0;
float g_ownScaleUsed = -1.0f; // the value the last decode logged (render thread only)

uint64_t g_present = 0;

// Copies of UE4's view uniform buffer (2048 bytes, the matrix at +1728), one per Present by parity: taken from vs cb1
// of the first draw of each Present that writes a texture like the pass's packed motion (the base pass).
constexpr UINT kViewBytes = 2048;
struct ViewCopy
{
    ID3D11Buffer* buffer = nullptr;
    uint64_t present = UINT64_MAX;
};
ViewCopy g_views[2];
bool g_viewCopyFailed = false;
uint64_t g_viewCopies = 0;
uint64_t g_matrixPasses = 0;
// The last pass's packed motion texture (size and format; zero until a pass was seen).
UINT g_packedWidth = 0;
UINT g_packedHeight = 0;
DXGI_FORMAT g_packedFormat = DXGI_FORMAT_UNKNOWN;

struct MotionTexture
{
    ID3D11Texture2D* texture = nullptr;
    ID3D11UnorderedAccessView* uav = nullptr;
    ID3D11ShaderResourceView* srv = nullptr;
};

// Copies of the frame depth, ping-ponging like the motion: [g_current] gets this Present's after the decode, the
// other holds the previous Present's. Made with the depth's own description (so CopyResource matches) and kept;
// replaced (the old one parked) if the depth's size or format changes.
struct DepthCopy
{
    ID3D11Texture2D* texture = nullptr;
    ID3D11ShaderResourceView* srv = nullptr;
    UINT width = 0;
    UINT height = 0;
    DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;
    uint64_t present = UINT64_MAX;
};
DepthCopy g_depthCopies[2];
std::vector<DepthCopy> g_parkedDepth;

// SRVs on the game's depth textures, made once per texture (each holds its texture).
struct DepthView
{
    ID3D11Texture2D* texture = nullptr;
    ID3D11ShaderResourceView* srv = nullptr;
};
constexpr int kMaxDepthViews = 12;
DepthView g_depthViews[kMaxDepthViews];

bool g_setupTried = false;
bool g_ready = false;
ID3D11ComputeShader* g_decode = nullptr;
ID3D11ComputeShader* g_viewShader = nullptr;
ID3D11ComputeShader* g_classShader = nullptr;
ID3D11Buffer* g_classConstants = nullptr;
ID3D11Buffer* g_constants = nullptr;
// Decoded motion ping-pongs: [g_current] is this Present's, the other the previous Present's.
MotionTexture g_motion[2];
MotionTexture g_composed;
MotionTexture g_classes;
MotionTexture g_bias; // the disocclusion mask (R8), at the pass size
int g_current = 0;
UINT g_width = 0;
UINT g_height = 0;

// Present of the last decode, to tell whether the other texture holds the previous Present's motion.
uint64_t g_lastDecodePresent = UINT64_MAX;
bool g_hasPrevious = false;
uint64_t g_uncomposed = 0;

// The two shapes of the pass (see the header). The layout of the last decode, to restart composition on a switch.
enum class Layout
{
    None,
    EyeSize,
    RenderResolution,
};
Layout g_layout = Layout::None;
Layout g_lastDecodeLayout = Layout::None;

// This Dispatch is the pass and its motion was decoded: AfterDispatch copies the output and may draw the view.
bool g_decodedThisDispatch = false;
uint64_t g_decodes = 0;

// Copies of the pass's output from the last two Presents, made only while a frame capture is armed or running, so
// the capture can check the composed motion against the same eye's picture from two Presents back.
ID3D11Texture2D* g_finalCopies[2] = {};
// The pass's colour input from the same Presents (the raw, jittered frame without any TAA in it).
ID3D11Texture2D* g_colourCopies[2] = {};
uint64_t g_finalCopyPresent[2] = { UINT64_MAX, UINT64_MAX };
// What this pass's composition read, for the capture: the frame depth and the previous Present's copy (nullptr when
// not used).
ID3D11Texture2D* g_depthThisDispatch = nullptr;
ID3D11Texture2D* g_previousDepthThisDispatch = nullptr;
bool g_finalCopyFailed = false;

// Same eye-size limits as the NR backends ([Inputs] DuniaTaaMinEvalWidth / DuniaTaaMinEvalHeight).
bool IsEyeSize(UINT width, UINT height)
{
    const auto minW = (UINT) std::max(0, Config::Instance()->DuniaTaaMinEvalWidth.value_or_default());
    const auto minH = (UINT) std::max(0, Config::Instance()->DuniaTaaMinEvalHeight.value_or_default());
    return width >= minW || height >= minH;
}

// Returns an AddRef'd Texture2D, or nullptr.
ID3D11Texture2D* TextureFromView(ID3D11View* view)
{
    if (view == nullptr)
        return nullptr;
    ID3D11Resource* resource = nullptr;
    view->GetResource(&resource);
    if (resource == nullptr)
        return nullptr;
    ID3D11Texture2D* texture = nullptr;
    resource->QueryInterface(IID_PPV_ARGS(&texture));
    resource->Release();
    return texture;
}

D3D11_TEXTURE2D_DESC Desc(ID3D11Texture2D* texture)
{
    D3D11_TEXTURE2D_DESC desc {};
    texture->GetDesc(&desc);
    return desc;
}

// The game's views of the pass: UAV 0 (output) and t0-t2. All AddRef'd. `packed` and `colour` index srvs for the
// matched layout.
struct PassViews
{
    ID3D11UnorderedAccessView* output = nullptr;
    ID3D11ShaderResourceView* srvs[3] = {};
    int packed = 1;
    int colour = 2;

    ~PassViews()
    {
        if (output != nullptr)
            output->Release();
        for (auto* srv : srvs)
        {
            if (srv != nullptr)
                srv->Release();
        }
    }
};

bool Is10Bit(DXGI_FORMAT format)
{
    return format == DXGI_FORMAT_R10G10B10A2_UNORM || format == DXGI_FORMAT_R10G10B10A2_TYPELESS;
}

bool IsPackedMotionView(ID3D11ShaderResourceView* view)
{
    if (view == nullptr)
        return false;
    D3D11_SHADER_RESOURCE_VIEW_DESC desc {};
    view->GetDesc(&desc);
    return desc.Format == DXGI_FORMAT_R32_UINT && desc.ViewDimension == D3D11_SRV_DIMENSION_TEXTURE2D;
}

// The game's render targets for the render-resolution pass are a few texels larger than its output (2556 for 2554).
constexpr UINT kRenderTargetSlack = 16;
// Below this the render-resolution layout isn't trusted (a small pass with the same bindings).
constexpr UINT kMinRenderSize = 640;

// Width x height of srvs[index]'s texture and its format; false if it isn't a texture.
bool SrvSize(PassViews& views, int index, D3D11_TEXTURE2D_DESC& desc)
{
    ID3D11Texture2D* texture = TextureFromView(views.srvs[index]);
    if (texture == nullptr)
        return false;
    desc = Desc(texture);
    texture->Release();
    return true;
}

// Why no pass was decoded in a gameplay Present, the furthest stage reached that Present (a later value wins). With F6
// on, a gameplay Present without DLSS is logged with its reason (AccountPresent), so a drop to the game's TAA names
// itself.
enum Miss : int
{
    kMissNoPass,     // no dispatch looked like the pass
    kMissNearMiss,   // one had a 10-bit UAV 0 and packed motion bound but failed a check (logged once per check)
    kMissNotReady,   // the decode's setup failed
    kMissSettling,   // the pass is at a new size that has not held kSettlePresents yet
    kMissSizeCap,    // too many new pass sizes this session (kMaxNewSizes); textures are never freed
    kMissSizeFailed, // textures at the new size could not be made
    kMissMapFailed,  // the decode's constants could not be mapped
    kMissCount
};
const char* const kMissNames[kMissCount] = { "no TAA pass this Present",
                                             "a TAA-like pass failed a check",
                                             "the decode is not set up",
                                             "the pass's new size is still settling",
                                             "too many pass sizes this session",
                                             "motion textures at the new size failed",
                                             "the decode's constants did not map" };
std::atomic<int> g_missReason { kMissNoPass };
std::atomic<bool> g_decodedThisPresent { false };
std::atomic<bool> g_gameplayThisPresent { false };

void NoteMiss(Miss reason)
{
    if (reason > g_missReason.load(std::memory_order_relaxed))
        g_missReason.store(reason, std::memory_order_relaxed);
}

// A dispatch with the pass's shape that failed one check: logged once per check, with the sizes it had.
void NearMiss(const char* why, const D3D11_TEXTURE2D_DESC& out, const D3D11_TEXTURE2D_DESC& packed,
              const D3D11_TEXTURE2D_DESC& colour)
{
    NoteMiss(kMissNearMiss);
    static const char* logged[16] = {};
    for (auto*& seen : logged)
    {
        if (seen == why)
            return;
        if (seen == nullptr)
        {
            seen = why;
            LOG_WARN("TAA motion: a dispatch like the TAA pass was not taken: {} (output {}x{} format {}, motion {}x{}, "
                     "colour {}x{} format {}; Present {}); logged once per reason",
                     why, out.Width, out.Height, (UINT) out.Format, packed.Width, packed.Height, colour.Width,
                     colour.Height, (UINT) colour.Format, g_present);
            return;
        }
    }
}

// Which TAA pass is bound, if any. Eye size: an eye-sized 10-bit UAV 0 with t1 R32_UINT and t2 RGBA16F of the same
// size. Render resolution: a 10-bit UAV 0 with t0 R32_UINT and t1 RGBA16F of that size or a few texels larger, and
// t2 the 10-bit history of the same size. Sets width and height to the output's size, the size the motion is in.
// A dispatch with a 10-bit UAV 0 and packed motion that fails a later check is a near miss (NearMiss).
Layout MatchPass(ID3D11DeviceContext* ctx, PassViews& views, UINT& width, UINT& height)
{
    ctx->CSGetUnorderedAccessViews(0, 1, &views.output);
    ID3D11Texture2D* output = TextureFromView(views.output);
    if (output == nullptr)
        return Layout::None;
    const auto outDesc = Desc(output);
    output->Release();
    if (!Is10Bit(outDesc.Format))
        return Layout::None;
    width = outDesc.Width;
    height = outDesc.Height;

    ctx->CSGetShaderResources(0, 3, views.srvs);
    if (views.srvs[1] == nullptr || views.srvs[2] == nullptr)
        return Layout::None;
    D3D11_TEXTURE2D_DESC packedDesc {};
    D3D11_TEXTURE2D_DESC colourDesc {};

    if (IsPackedMotionView(views.srvs[1]))
    {
        views.packed = 1;
        views.colour = 2;
        const bool packedOk = SrvSize(views, 1, packedDesc);
        const bool colourOk = SrvSize(views, 2, colourDesc);
        const char* why = !IsEyeSize(outDesc.Width, outDesc.Height) ? "eye-size layout, output below the eye-size limits"
                          : !packedOk || !colourOk                  ? "eye-size layout, t1 or t2 is not a texture"
                          : packedDesc.Width != outDesc.Width || packedDesc.Height != outDesc.Height
                              ? "eye-size layout, motion size differs from the output"
                          : colourDesc.Width != outDesc.Width || colourDesc.Height != outDesc.Height
                              ? "eye-size layout, colour size differs from the output"
                          : colourDesc.Format != DXGI_FORMAT_R16G16B16A16_FLOAT ? "eye-size layout, colour is not RGBA16F"
                                                                                : nullptr;
        if (why == nullptr)
            return Layout::EyeSize;
        NearMiss(why, outDesc, packedDesc, colourDesc);
        return Layout::None;
    }

    if (IsPackedMotionView(views.srvs[0]))
    {
        views.packed = 0;
        views.colour = 1;
        D3D11_TEXTURE2D_DESC historyDesc {};
        const auto covers = [&](const D3D11_TEXTURE2D_DESC& desc)
        {
            return desc.Width >= outDesc.Width && desc.Height >= outDesc.Height &&
                   desc.Width <= outDesc.Width + kRenderTargetSlack && desc.Height <= outDesc.Height + kRenderTargetSlack;
        };
        const bool texturesOk =
            SrvSize(views, 0, packedDesc) && SrvSize(views, 1, colourDesc) && SrvSize(views, 2, historyDesc);
        const char* why =
            outDesc.Width < kMinRenderSize || outDesc.Height < kMinRenderSize ? "render layout, output below 640"
            : !texturesOk                                                     ? "render layout, t0-t2 are not all textures"
            : !covers(packedDesc)                                             ? "render layout, motion does not cover the output"
            : !covers(colourDesc)                                             ? "render layout, colour does not cover the output"
            : colourDesc.Format != DXGI_FORMAT_R16G16B16A16_FLOAT             ? "render layout, colour is not RGBA16F"
            : !Is10Bit(historyDesc.Format)                                    ? "render layout, history is not 10-bit"
            : historyDesc.Width != outDesc.Width || historyDesc.Height != outDesc.Height
                ? "render layout, history size differs from the output"
                : nullptr;
        if (why == nullptr)
            return Layout::RenderResolution;
        NearMiss(why, outDesc, packedDesc, colourDesc);
        return Layout::None;
    }
    return Layout::None;
}

const char* LayoutName(Layout layout)
{
    return layout == Layout::RenderResolution ? "render-resolution TAA pass (upscaled afterwards)"
                                              : "eye-size tonemap + TAA pass";
}

ID3D11ComputeShader* Compile(ID3D11Device* device, const char* source, size_t length, const char* name)
{
    ID3DBlob* code = nullptr;
    ID3DBlob* errors = nullptr;
    if (FAILED(D3DCompile(source, length, name, nullptr, nullptr, "CSMain", "cs_5_0", D3DCOMPILE_OPTIMIZATION_LEVEL3,
                          0, &code, &errors)))
    {
        LOG_ERROR("TAA motion: {} compile failed: {}", name,
                  errors != nullptr ? (const char*) errors->GetBufferPointer() : "no message");
        if (errors != nullptr)
            errors->Release();
        return nullptr;
    }
    if (errors != nullptr)
        errors->Release();
    ID3D11ComputeShader* shader = nullptr;
    if (FAILED(device->CreateComputeShader(code->GetBufferPointer(), code->GetBufferSize(), nullptr, &shader)))
    {
        LOG_ERROR("TAA motion: {} CreateComputeShader failed", name);
        shader = nullptr;
    }
    code->Release();
    return shader;
}

bool CreateMotion(ID3D11Device* device, UINT width, UINT height, MotionTexture& out,
                  DXGI_FORMAT format = DXGI_FORMAT_R16G16_FLOAT)
{
    D3D11_TEXTURE2D_DESC desc {};
    desc.Width = width;
    desc.Height = height;
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = format;
    desc.SampleDesc.Count = 1;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;
    return SUCCEEDED(device->CreateTexture2D(&desc, nullptr, &out.texture)) &&
           SUCCEEDED(device->CreateUnorderedAccessView(out.texture, nullptr, &out.uav)) &&
           SUCCEEDED(device->CreateShaderResourceView(out.texture, nullptr, &out.srv));
}

// Motion textures from earlier pass sizes, one set per size. Kept, never released: DLSS or queued work may still use
// them. A switch back to a parked size takes its set again, so only new sizes count against the limit.
struct MotionSet
{
    MotionTexture motion[2];
    MotionTexture composed;
    MotionTexture classes; // the F8 classification view's per-pixel path
    MotionTexture bias;    // the disocclusion mask for DLSS
    UINT width = 0;
    UINT height = 0;
};
std::vector<MotionSet> g_parked;
std::vector<MotionTexture> g_failed;
std::vector<ID3D11Texture2D*> g_parkedCopies;
int g_newSizes = 0;
constexpr int kMaxNewSizes = 6;
// A new pass size is taken once it has held this many Presents (about half a second).
constexpr uint64_t kSettlePresents = 45;
UINT g_candidateWidth = 0;
UINT g_candidateHeight = 0;
uint64_t g_candidateSince = 0;

// The pass changed size (Days Gone draws at one eye size, then switches; with a Render Scale below 100% it switches
// between the eye-size pass in menus and the render-resolution pass in play): motion textures at the new size, from
// the parked sets or new, and the current ones parked. Composition restarts. Returns false if new textures couldn't
// be made or too many sizes were seen.
bool Resize(ID3D11DeviceContext* ctx, UINT width, UINT height)
{
    MotionSet next;
    const auto parked = std::find_if(g_parked.begin(), g_parked.end(), [&](const MotionSet& set)
                                     { return set.width == width && set.height == height; });
    const bool reused = parked != g_parked.end();
    if (reused)
    {
        next = *parked;
        g_parked.erase(parked);
    }
    else
    {
        if (g_newSizes >= kMaxNewSizes)
            return false;
        ++g_newSizes;
        ID3D11Device* device = nullptr;
        ctx->GetDevice(&device);
        if (device == nullptr)
            return false;
        const bool ok = CreateMotion(device, width, height, next.motion[0]) &&
                        CreateMotion(device, width, height, next.motion[1]) &&
                        CreateMotion(device, width, height, next.composed) &&
                        CreateMotion(device, width, height, next.classes, DXGI_FORMAT_R16_FLOAT) &&
                        CreateMotion(device, width, height, next.bias, DXGI_FORMAT_R8_UNORM);
        device->Release();
        if (!ok)
        {
            LOG_ERROR("TAA motion: textures at {}x{} failed; staying at {}x{}", width, height, g_width, g_height);
            g_newSizes = kMaxNewSizes; // don't retry every frame
            g_failed.push_back(next.motion[0]);
            g_failed.push_back(next.motion[1]);
            g_failed.push_back(next.composed);
            g_failed.push_back(next.classes);
            g_failed.push_back(next.bias);
            return false;
        }
        next.width = width;
        next.height = height;
    }

    g_parked.push_back({ { g_motion[0], g_motion[1] }, g_composed, g_classes, g_bias, g_width, g_height });
    g_motion[0] = next.motion[0];
    g_motion[1] = next.motion[1];
    g_composed = next.composed;
    g_classes = next.classes;
    g_bias = next.bias;
    // The capture copies are the old size too; they're made again while a capture needs them.
    for (int i = 0; i < 2; ++i)
    {
        if (g_finalCopies[i] != nullptr)
            g_parkedCopies.push_back(g_finalCopies[i]);
        if (g_colourCopies[i] != nullptr)
            g_parkedCopies.push_back(g_colourCopies[i]);
        g_finalCopies[i] = nullptr;
        g_colourCopies[i] = nullptr;
        g_finalCopyPresent[i] = UINT64_MAX;
    }
    LOG_INFO("TAA motion: pass changed size {}x{} -> {}x{}; decoding at the new size ({} textures)", g_width,
             g_height, width, height, reused ? "earlier" : "new");
    g_width = width;
    g_height = height;
    g_lastDecodePresent = UINT64_MAX;
    return true;
}

// Once: the motion textures are sized to the first pass found; a later size change goes through Resize (a failed
// setup is simply left unused).
void Setup(ID3D11DeviceContext* ctx, UINT width, UINT height)
{
    g_setupTried = true;
    ID3D11Device* device = nullptr;
    ctx->GetDevice(&device);
    if (device == nullptr)
        return;

    g_decode = Compile(device, kDecodeShader, sizeof(kDecodeShader) - 1, "TaaMotion_Decode");
    g_viewShader = Compile(device, kViewShader, sizeof(kViewShader) - 1, "TaaMotion_View");
    g_classShader = Compile(device, kClassShader, sizeof(kClassShader) - 1, "TaaMotion_Class");

    D3D11_BUFFER_DESC cbDesc {};
    cbDesc.ByteWidth = 80;
    cbDesc.Usage = D3D11_USAGE_DYNAMIC;
    cbDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    cbDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;

    g_ready = g_decode != nullptr && g_viewShader != nullptr && g_classShader != nullptr &&
              SUCCEEDED(device->CreateBuffer(&cbDesc, nullptr, &g_constants)) &&
              SUCCEEDED(device->CreateBuffer(&cbDesc, nullptr, &g_classConstants)) &&
              CreateMotion(device, width, height, g_motion[0]) && CreateMotion(device, width, height, g_motion[1]) &&
              CreateMotion(device, width, height, g_composed) &&
              CreateMotion(device, width, height, g_classes, DXGI_FORMAT_R16_FLOAT) &&
              CreateMotion(device, width, height, g_bias, DXGI_FORMAT_R8_UNORM);
    // The view copies are optional: without them the composition is chained through the other eye only.
    D3D11_BUFFER_DESC viewDesc {};
    viewDesc.ByteWidth = kViewBytes;
    viewDesc.Usage = D3D11_USAGE_DEFAULT;
    viewDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    for (auto& view : g_views)
    {
        if (FAILED(device->CreateBuffer(&viewDesc, nullptr, &view.buffer)))
            view.buffer = nullptr;
        g_viewCopyFailed |= view.buffer == nullptr;
    }
    if (g_viewCopyFailed)
        LOG_WARN("TAA motion: view constant copies failed; matrix motion off");
    device->Release();

    if (!g_ready)
    {
        LOG_ERROR("TAA motion: setup at {}x{} failed; decoding stays off", width, height);
        return;
    }
    g_width = width;
    g_height = height;
    LOG_INFO("TAA motion: pass found, decoding and composing {}x{} motion (F8: view, third press = which path each "
             "pixel took)",
             width, height);
}

// The single-channel view format for a depth texture, or UNKNOWN.
DXGI_FORMAT DepthViewFormat(DXGI_FORMAT format)
{
    switch (format)
    {
    case DXGI_FORMAT_R24G8_TYPELESS:
    case DXGI_FORMAT_D24_UNORM_S8_UINT:
        return DXGI_FORMAT_R24_UNORM_X8_TYPELESS;
    case DXGI_FORMAT_R32_TYPELESS:
    case DXGI_FORMAT_D32_FLOAT:
        return DXGI_FORMAT_R32_FLOAT;
    case DXGI_FORMAT_R32G8X24_TYPELESS:
    case DXGI_FORMAT_D32_FLOAT_S8X24_UINT:
        return DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS;
    default:
        return DXGI_FORMAT_UNKNOWN;
    }
}

ID3D11ShaderResourceView* CreateDepthSrv(ID3D11Device* device, ID3D11Texture2D* texture, DXGI_FORMAT format)
{
    D3D11_SHADER_RESOURCE_VIEW_DESC desc {};
    desc.Format = DepthViewFormat(format);
    desc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
    desc.Texture2D.MipLevels = 1;
    ID3D11ShaderResourceView* srv = nullptr;
    if (desc.Format == DXGI_FORMAT_UNKNOWN || FAILED(device->CreateShaderResourceView(texture, &desc, &srv)))
        return nullptr;
    return srv;
}

// A view on the game's depth texture, cached. nullptr if it can't be read.
ID3D11ShaderResourceView* DepthSrvFor(ID3D11Device* device, ID3D11Texture2D* texture)
{
    for (auto& view : g_depthViews)
    {
        if (view.texture == texture)
            return view.srv;
    }
    const auto desc = Desc(texture);
    if ((desc.BindFlags & D3D11_BIND_SHADER_RESOURCE) == 0 || desc.SampleDesc.Count != 1)
        return nullptr;
    for (auto& view : g_depthViews)
    {
        if (view.texture != nullptr)
            continue;
        view.srv = CreateDepthSrv(device, texture, desc.Format);
        if (view.srv == nullptr)
            return nullptr;
        view.texture = texture;
        return view.srv;
    }
    return nullptr;
}

// The copy slot for `depth`, (re)made to match it. False if it can't be.
bool EnsureDepthCopy(ID3D11Device* device, DepthCopy& copy, ID3D11Texture2D* depth)
{
    const auto desc = Desc(depth);
    const auto matches = [&](const DepthCopy& c)
    { return c.texture != nullptr && c.width == desc.Width && c.height == desc.Height && c.format == desc.Format; };
    if (matches(copy))
        return true;
    // A copy parked at an earlier size is taken again (menus and play switch sizes).
    const auto parked = std::find_if(g_parkedDepth.begin(), g_parkedDepth.end(), matches);
    if (parked != g_parkedDepth.end())
    {
        DepthCopy reused = *parked;
        g_parkedDepth.erase(parked);
        if (copy.texture != nullptr)
            g_parkedDepth.push_back(copy);
        copy = reused;
        copy.present = UINT64_MAX;
        return true;
    }
    // Two per settled size, for as many sizes as the motion textures follow.
    static int created = 0;
    if (created >= 2 * (kMaxNewSizes + 1))
        return false; // sizes keep changing; stop making copies
    ++created;
    if (copy.texture != nullptr)
        g_parkedDepth.push_back(copy);
    copy = {};
    auto copyDesc = desc;
    copyDesc.CPUAccessFlags = 0;
    copyDesc.MiscFlags = 0;
    copyDesc.Usage = D3D11_USAGE_DEFAULT;
    if (FAILED(device->CreateTexture2D(&copyDesc, nullptr, &copy.texture)))
    {
        copy.texture = nullptr;
        return false;
    }
    copy.srv = CreateDepthSrv(device, copy.texture, desc.Format);
    copy.width = desc.Width;
    copy.height = desc.Height;
    copy.format = desc.Format;
    return copy.srv != nullptr;
}

// Runs `shader` over the image with the given SRVs in t0.., UAVs in u0.. and constant buffers in b0.. (up to 3),
// then puts the game's compute shader and those bindings back.
void RunPass(ID3D11DeviceContext* ctx, ID3D11ComputeShader* shader, UINT srvCount,
             ID3D11ShaderResourceView* const* srvs, UINT uavCount, ID3D11UnorderedAccessView* const* uavs,
             UINT cbCount = 0, ID3D11Buffer* const* constants = nullptr)
{
    ID3D11ComputeShader* savedShader = nullptr;
    ID3D11ShaderResourceView* savedSrvs[7] = {};  // the decode binds seven
    ID3D11UnorderedAccessView* savedUavs[4] = {}; // the decode binds four: motion, composed, classification, mask
    ID3D11Buffer* savedCbs[3] = {};
    ctx->CSGetShader(&savedShader, nullptr, nullptr);
    ctx->CSGetShaderResources(0, srvCount, savedSrvs);
    ctx->CSGetUnorderedAccessViews(0, uavCount, savedUavs);
    if (cbCount != 0)
        ctx->CSGetConstantBuffers(0, cbCount, savedCbs);

    ctx->CSSetShader(shader, nullptr, 0);
    ctx->CSSetShaderResources(0, srvCount, srvs);
    ctx->CSSetUnorderedAccessViews(0, uavCount, uavs, nullptr);
    if (cbCount != 0)
        ctx->CSSetConstantBuffers(0, cbCount, constants);
    ctx->Dispatch((g_width + 7) / 8, (g_height + 7) / 8, 1);

    ctx->CSSetUnorderedAccessViews(0, uavCount, savedUavs, nullptr);
    ctx->CSSetShaderResources(0, srvCount, savedSrvs);
    if (cbCount != 0)
        ctx->CSSetConstantBuffers(0, cbCount, savedCbs);
    ctx->CSSetShader(savedShader, nullptr, 0);

    if (savedShader != nullptr)
        savedShader->Release();
    for (UINT i = 0; i < srvCount; ++i)
    {
        if (savedSrvs[i] != nullptr)
            savedSrvs[i]->Release();
    }
    for (UINT i = 0; i < uavCount; ++i)
    {
        if (savedUavs[i] != nullptr)
            savedUavs[i]->Release();
    }
    for (auto* cb : savedCbs)
    {
        if (cb != nullptr)
            cb->Release();
    }
}

// A plain copy texture like `source` (shader resource only), or nullptr.
ID3D11Texture2D* CreateCopyOf(ID3D11DeviceContext* ctx, ID3D11Texture2D* source)
{
    auto desc = Desc(source);
    desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    desc.CPUAccessFlags = 0;
    desc.MiscFlags = 0;
    ID3D11Device* device = nullptr;
    ctx->GetDevice(&device);
    ID3D11Texture2D* copy = nullptr;
    if (device == nullptr || FAILED(device->CreateTexture2D(&desc, nullptr, &copy)))
        copy = nullptr;
    if (device != nullptr)
        device->Release();
    return copy;
}

// The colour ring slot at this colour's size and format (made, or taken back from the parked ones). False when it
// cannot be made; the shadow test is then skipped (logged).
bool EnsureColourCopy(ID3D11DeviceContext* ctx, ColourCopy& copy, ID3D11Texture2D* colour)
{
    const auto desc = Desc(colour);
    const auto matches = [&](const ColourCopy& c)
    { return c.texture != nullptr && c.width == desc.Width && c.height == desc.Height && c.format == desc.Format; };
    if (matches(copy))
        return true;
    const auto parked = std::find_if(g_parkedColour.begin(), g_parkedColour.end(), matches);
    if (parked != g_parkedColour.end())
    {
        ColourCopy reused = *parked;
        g_parkedColour.erase(parked);
        if (copy.texture != nullptr)
            g_parkedColour.push_back(copy);
        copy = reused;
        copy.present = UINT64_MAX;
        return true;
    }
    static int created = 0;
    if (g_colourCopyFailed || created >= 2 * (kMaxNewSizes + 1))
        return false;
    ++created;
    if (copy.texture != nullptr)
        g_parkedColour.push_back(copy);
    copy = {};
    copy.texture = CreateCopyOf(ctx, colour);
    ID3D11Device* device = nullptr;
    ctx->GetDevice(&device);
    const bool ok = copy.texture != nullptr && device != nullptr &&
                    SUCCEEDED(device->CreateShaderResourceView(copy.texture, nullptr, &copy.srv));
    if (device != nullptr)
        device->Release();
    if (!ok)
    {
        LOG_WARN("TAA motion: colour copy {}x{} fmt {} failed; the shadow mask stays off", desc.Width, desc.Height,
                 (int) desc.Format);
        g_colourCopyFailed = true;
        return false;
    }
    copy.width = desc.Width;
    copy.height = desc.Height;
    copy.format = desc.Format;
    return true;
}

// Keeps a copy of this Present's output and colour input in the older of the two copy slots. The copies are
// created on first use and kept.
void CopyFinal(ID3D11DeviceContext* ctx, ID3D11UnorderedAccessView* outputView, ID3D11ShaderResourceView* colourView)
{
    if (g_finalCopyFailed)
        return;
    ID3D11Texture2D* output = TextureFromView(outputView);
    if (output == nullptr)
        return;
    ID3D11Texture2D* colour = TextureFromView(colourView);
    const int slot = g_finalCopyPresent[0] == UINT64_MAX || (g_finalCopyPresent[1] != UINT64_MAX &&
                                                             g_finalCopyPresent[0] < g_finalCopyPresent[1])
                         ? 0
                         : 1;
    if (g_finalCopies[slot] == nullptr)
    {
        g_finalCopies[slot] = CreateCopyOf(ctx, output);
        if (g_finalCopies[slot] == nullptr)
        {
            LOG_WARN("TAA motion: output copy texture failed; captures won't include it");
            g_finalCopyFailed = true;
        }
    }
    if (colour != nullptr && g_colourCopies[slot] == nullptr)
        g_colourCopies[slot] = CreateCopyOf(ctx, colour);
    if (g_finalCopies[slot] != nullptr)
    {
        ctx->CopyResource(g_finalCopies[slot], output);
        if (colour != nullptr && g_colourCopies[slot] != nullptr &&
            Desc(colour).Width == Desc(g_colourCopies[slot]).Width &&
            Desc(colour).Height == Desc(g_colourCopies[slot]).Height)
            ctx->CopyResource(g_colourCopies[slot], colour);
        g_finalCopyPresent[slot] = g_present;
    }
    if (colour != nullptr)
        colour->Release();
    output->Release();
}

// Drop accounting, while F6 has DLSS on: every gameplay Present either shows DLSS or counts one reason, and each run of
// Presents without it is logged with its Present range when it ends. The TAA DLAA stats line could not show a drop: it
// is written only from a decoded pass, so Presents where the pass was not recognised were never in it.
constexpr int kDropSlots = kMissCount + TaaDlaa_Dx11::kMissSlots;
constexpr uint64_t kDropWindow = 900;
constexpr int kMaxStreakLogs = 300;
struct DropCounts
{
    uint64_t presents = 0;
    uint64_t gameplay = 0;
    uint64_t shown = 0;
    uint64_t slots[kDropSlots] = {};
};
DropCounts g_dropWindow;
uint64_t g_streakStart = UINT64_MAX; // the first Present of the current run without DLSS
uint64_t g_streakSlots[kDropSlots] = {};
int g_streaksLogged = 0;

const char* DropSlotName(int slot)
{
    return slot < kMissCount ? kMissNames[slot] : TaaDlaa_Dx11::MissSlotName(slot - kMissCount);
}

std::string DropReasons(const uint64_t (&slots)[kDropSlots])
{
    std::string text;
    for (int i = 0; i < kDropSlots; ++i)
    {
        if (slots[i] != 0)
            text += std::format("{}{} x{}", text.empty() ? "" : "; ", DropSlotName(i), slots[i]);
    }
    return text.empty() ? std::string("none") : text;
}

// Logs the current run without DLSS, if any, as ending before `present`.
void EndStreak(uint64_t present, const char* how)
{
    if (g_streakStart == UINT64_MAX)
        return;
    uint64_t count = 0;
    for (const auto n : g_streakSlots)
        count += n;
    if (g_streaksLogged < kMaxStreakLogs)
    {
        ++g_streaksLogged;
        LOG_WARN("TAA DLSS: {} gameplay Presents without DLSS (Presents {}-{}), {}: {}{}", count, g_streakStart,
                 present - 1, how, DropReasons(g_streakSlots),
                 g_streaksLogged == kMaxStreakLogs ? "; runs are not logged again (still counted per window)" : "");
    }
    g_streakStart = UINT64_MAX;
    std::memset(g_streakSlots, 0, sizeof(g_streakSlots));
}

// At Present, before the index advances: what this Present showed.
void AccountPresent()
{
    const bool gameplay = g_gameplayThisPresent.exchange(false, std::memory_order_relaxed);
    const bool decoded = g_decodedThisPresent.exchange(false, std::memory_order_relaxed);
    const int reason = g_missReason.exchange(kMissNoPass, std::memory_order_relaxed);
    if (!TaaDlaa_Dx11::Enabled() || !TaaDlaa_Dx11::On())
    {
        EndStreak(g_present, "ended by switching to the game's TAA");
        g_dropWindow = {};
        return;
    }
    auto& window = g_dropWindow;
    ++window.presents;
    if (!gameplay)
        EndStreak(g_present, "ended by a Present without gameplay (menu or loading)");
    else if (TaaDlaa_Dx11::ReplacedThisPresent())
    {
        ++window.gameplay;
        ++window.shown;
        EndStreak(g_present, "then DLSS again");
    }
    else
    {
        ++window.gameplay;
        const int slot = decoded ? kMissCount + TaaDlaa_Dx11::MissSlot(g_present) : reason;
        ++window.slots[slot];
        if (g_streakStart == UINT64_MAX)
            g_streakStart = g_present;
        ++g_streakSlots[slot];
        // A run that does not end is reported once per window length as well.
        if (g_present + 1 - g_streakStart >= kDropWindow)
            EndStreak(g_present + 1, "still going (reported every 900 Presents)");
    }
    if (window.presents >= kDropWindow)
    {
        LOG_INFO("TAA DLSS: shown in {} of {} gameplay Presents ({} Presents); without it: {}", window.shown,
                 window.gameplay, window.presents, DropReasons(window.slots));
        window = {};
    }
}
} // namespace

bool TaaMotion_Dx11::Enabled()
{
    static int cached = -1;
    if (cached < 0)
        cached = Config::Instance()->TaaMotionCheck.value_or_default() || Config::Instance()->TaaDlaa.value_or_default()
                     ? 1
                     : 0;
    return cached == 1;
}

bool TaaMotion_Dx11::InPass() { return g_inPass > 0; }

void TaaMotion_Dx11::BeforeDispatch(ID3D11DeviceContext* ctx)
{
    g_decodedThisDispatch = false;
    if (ctx == nullptr)
        return;
    // Gameplay: the G-buffer's depth has been bound this Present (the drop accounting counts only these Presents).
    if (FinalImageNr_Dx11::FrameDepth() != nullptr)
        g_gameplayThisPresent.store(true, std::memory_order_relaxed);
    if (g_setupTried && !g_ready)
    {
        NoteMiss(kMissNotReady);
        return;
    }

    PassViews views;
    UINT width = 0;
    UINT height = 0;
    const Layout layout = MatchPass(ctx, views, width, height);
    if (layout == Layout::None)
        return;

    {
        D3D11_TEXTURE2D_DESC packedDesc {};
        if (SrvSize(views, views.packed, packedDesc))
        {
            g_packedWidth = packedDesc.Width;
            g_packedHeight = packedDesc.Height;
            g_packedFormat = packedDesc.Format;
        }
    }
    if (!g_setupTried)
        Setup(ctx, width, height);
    if (!g_ready)
        return;
    if (width != g_width || height != g_height)
    {
        // The game steps through in-between sizes when the Render Scale or the eye size changes (90%, 80%, ... for a
        // few dozen Presents each); textures are only made once a size has held, so those steps cost nothing and
        // don't use up the limits. Until then the game's pass runs as it is.
        if (width != g_candidateWidth || height != g_candidateHeight)
        {
            // Logged so a size that never holds (two sizes taking turns resets the wait each time) shows itself.
            static int loggedCandidates = 0;
            if (loggedCandidates < 40)
            {
                ++loggedCandidates;
                LOG_INFO("TAA motion: pass at {}x{} ({}), set up for {}x{}; waiting {} Presents for the size to hold "
                         "(was waiting on {}x{}; Present {}){}",
                         width, height, LayoutName(layout), g_width, g_height, kSettlePresents, g_candidateWidth,
                         g_candidateHeight, g_present, loggedCandidates == 40 ? "; not logged again" : "");
            }
            g_candidateWidth = width;
            g_candidateHeight = height;
            g_candidateSince = g_present;
        }
        if (g_present - g_candidateSince < kSettlePresents)
        {
            NoteMiss(kMissSettling);
            return;
        }
        const bool atCap = g_newSizes >= kMaxNewSizes;
        if (!Resize(ctx, width, height))
        {
            NoteMiss(atCap ? kMissSizeCap : kMissSizeFailed);
            static bool loggedCap = false;
            static bool loggedFailed = false;
            bool& logged = atCap ? loggedCap : loggedFailed;
            if (!logged)
            {
                logged = true;
                LOG_WARN("TAA motion: pass at {}x{}, set up for {}x{}; skipping passes at this size: {} (Present {})",
                         width, height, g_width, g_height,
                         atCap ? "the limit of new pass sizes is reached (textures are kept, never freed)"
                               : "textures at the new size could not be made",
                         g_present);
            }
            return;
        }
    }
    if (layout != g_layout)
    {
        LOG_INFO("TAA motion: {} at {}x{}", LayoutName(layout), width, height);
        g_layout = layout;
    }

    // The other texture holds the previous Present's motion only if the pass ran then, once, at this size and in
    // this layout.
    const bool hasPrevious = g_lastDecodePresent != UINT64_MAX && g_lastDecodePresent + 1 == g_present &&
                             g_lastDecodeLayout == layout;

    // This frame's depth (covering the motion) and the previous Present's copy of it, for the occlusion test.
    ID3D11ShaderResourceView* depthSrv = nullptr;
    ID3D11ShaderResourceView* previousDepthSrv = nullptr;
    ID3D11Texture2D* depth = FinalImageNr_Dx11::FrameDepth();
    const int next = 1 - g_current;
    bool depthUsable = false;
    // The slot this Present's depth is copied into after the decode still holds the copy from two Presents back:
    // the same eye's, whose depth the disocclusion test reads.
    ID3D11ShaderResourceView* sameEyeDepthSrv = nullptr;
    if (depth != nullptr)
    {
        const auto depthDesc = Desc(depth);
        ID3D11Device* device = nullptr;
        ctx->GetDevice(&device);
        depthUsable = device != nullptr && depthDesc.Width >= width && depthDesc.Height >= height &&
                      depthDesc.Width <= width + 16 && depthDesc.Height <= height + 16 &&
                      (depthSrv = DepthSrvFor(device, depth)) != nullptr &&
                      EnsureDepthCopy(device, g_depthCopies[next], depth);
        if (device != nullptr)
            device->Release();
        const auto& previous = g_depthCopies[1 - next];
        if (depthUsable && previous.present != UINT64_MAX && previous.present + 1 == g_present &&
            previous.width == depthDesc.Width && previous.height == depthDesc.Height)
            previousDepthSrv = previous.srv;
        const auto& sameEye = g_depthCopies[next];
        if (depthUsable && sameEye.present != UINT64_MAX && sameEye.present + 2 == g_present &&
            sameEye.width == depthDesc.Width && sameEye.height == depthDesc.Height)
            sameEyeDepthSrv = sameEye.srv;
    }
    // The colour input and the same eye's copy of it from two Presents back (the shadow test).
    ID3D11ShaderResourceView* colourSrv = views.srvs[views.colour];
    ID3D11ShaderResourceView* sameEyeColourSrv = nullptr;
    ID3D11Texture2D* colourTex = TextureFromView(colourSrv);
    bool colourUsable = false;
    if (colourTex != nullptr)
    {
        colourUsable = g_maskMode.load(std::memory_order_relaxed) == 2 &&
                       EnsureColourCopy(ctx, g_colourRing[next], colourTex);
        const auto& sameEye = g_colourRing[next];
        if (colourUsable && sameEye.present != UINT64_MAX && sameEye.present + 2 == g_present)
            sameEyeColourSrv = sameEye.srv;
    }
    const int motionMode = g_motionMode.load(std::memory_order_relaxed);
    const bool occlusion = hasPrevious && previousDepthSrv != nullptr;
    // Matrix motion needs this Present's and the previous Present's view constants and this frame's depth.
    const auto& thisView = g_views[g_present & 1];
    const auto& previousView = g_views[(g_present + 1) & 1];
    const bool useMatrix = motionMode != 2 && hasPrevious && depthUsable && thisView.present == g_present &&
                           previousView.present != UINT64_MAX && previousView.present + 1 == g_present;
    if (useMatrix && g_matrixPasses++ == 0)
        LOG_INFO("TAA motion: same-eye motion from depth and the view matrices ({})",
                 motionMode == 1 ? "halo as static" : "hybrid");
    g_depthThisDispatch = depthUsable ? depth : nullptr;
    g_previousDepthThisDispatch = previousDepthSrv != nullptr ? g_depthCopies[1 - next].texture : nullptr;
    static bool loggedDepth = false;
    if (!loggedDepth && depthUsable)
    {
        loggedDepth = true;
        LOG_INFO("TAA motion: depth {}x{} format {} used for occlusion-aware composition", Desc(depth).Width,
                 Desc(depth).Height, (UINT) Desc(depth).Format);
    }

    D3D11_MAPPED_SUBRESOURCE map {};
    if (FAILED(ctx->Map(g_constants, 0, D3D11_MAP_WRITE_DISCARD, 0, &map)))
    {
        NoteMiss(kMissMapFailed);
        if (colourTex != nullptr)
            colourTex->Release();
        return;
    }
    const float ownScale = g_ownScale.load(std::memory_order_relaxed);
    uint32_t ownScaleBits = 0;
    std::memcpy(&ownScaleBits, &ownScale, sizeof(ownScaleBits));
    const float biasStrength = std::clamp(g_biasStrength.load(std::memory_order_relaxed), 0.0f, 1.0f);
    const float biasNearer = g_biasNearer.load(std::memory_order_relaxed);
    uint32_t biasBits = 0, nearerBits = 0;
    std::memcpy(&biasBits, &biasStrength, sizeof(biasBits));
    std::memcpy(&nearerBits, &biasNearer, sizeof(nearerBits));
    const int maskMode = g_maskMode.load(std::memory_order_relaxed);
    const float shadowDrop = g_shadowDrop.load(std::memory_order_relaxed);
    uint32_t dropBits = 0;
    std::memcpy(&dropBits, &shadowDrop, sizeof(dropBits));
    const bool biasOn = maskMode != 0 && biasStrength > 0.0f && (useMatrix || maskMode == 3);
    g_biasThisDispatch = biasOn && (maskMode == 3 || sameEyeDepthSrv != nullptr || sameEyeColourSrv != nullptr);
    if (biasOn && maskMode == 2)
    {
        if (sameEyeColourSrv != nullptr)
        {
            if (g_shadowPasses++ == 0)
                LOG_INFO("TAA motion: FIRST shadow mask (same-eye colour from Present {}, drop {:.3f}, {})",
                         g_present - 2, shadowDrop, g_layout == Layout::RenderResolution ? "YCoCg" : "RGB");
        }
        else if (g_shadowMissingColour++ == 0)
            LOG_WARN("TAA motion: shadow mask skipped: no same-eye colour from two Presents back ({}); counted per "
                     "900 decodes",
                     colourTex == nullptr ? "no colour input" : colourUsable ? "not yet" : "copy unavailable");
    }
    static int loggedMode = -1;
    if (maskMode != loggedMode)
    {
        loggedMode = maskMode;
        LOG_INFO("TAA motion: DLSS hint mask mode {} ({})", maskMode,
                 maskMode == 0   ? "off"
                 : maskMode == 1 ? "covers"
                 : maskMode == 2 ? "covers + shadows"
                                 : "TEST whole frame");
    }
    if (biasOn)
    {
        if (sameEyeDepthSrv != nullptr)
        {
            if (g_biasPasses++ == 0)
                LOG_INFO("TAA motion: FIRST disocclusion mask (same-eye depth from Present {}, strength {:.2f}, cover "
                         "{:.2f}x nearer)",
                         g_present - 2, biasStrength, biasNearer);
        }
        else if (g_biasMissingDepth++ == 0)
            LOG_WARN("TAA motion: disocclusion mask skipped: no same-eye depth from two Presents back (slot Present "
                     "{}, now {}); counted per 900 decodes",
                     g_depthCopies[next].present, g_present);
    }
    float tuning[TaaMotion_Dx11::kTuningCount] = {};
    bool tuningChanged = false;
    for (int i = 0; i < TaaMotion_Dx11::kTuningCount; ++i)
    {
        tuning[i] = g_tuning[i].load(std::memory_order_relaxed);
        tuningChanged |= tuning[i] != g_tuningUsed[i];
    }
    uint32_t constants[20] = { hasPrevious ? 1u : 0u,
                               occlusion ? 1u : 0u,
                               useMatrix ? (motionMode == 1 ? 2u : motionMode == 3 ? 3u : 1u) : 0u,
                               ownScaleBits,
                               sameEyeDepthSrv != nullptr ? 1u : 0u,
                               biasOn ? biasBits : 0u,
                               nearerBits,
                               sameEyeColourSrv != nullptr ? 1u : 0u,
                               biasOn ? (uint32_t) maskMode : 0u,
                               dropBits,
                               layout == Layout::RenderResolution ? 1u : 0u };
    std::memcpy(&constants[11], tuning, sizeof(tuning));
    if (useMatrix && ownScale != g_ownScaleUsed)
    {
        g_ownScaleUsed = ownScale;
        LOG_INFO("TAA motion: decoding with object motion x{:.2f} (Present {}, mode {})", ownScale, g_present,
                 TaaMotion_Dx11::MotionModeName());
    }
    if (useMatrix && tuningChanged)
    {
        std::memcpy(g_tuningUsed, tuning, sizeof(tuning));
        LOG_INFO("TAA motion: knee {:.3f} px + {:.4f} x camera motion, full at x{:.1f}; chained from {:.2f} to {:.2f} px; "
                 "object motion on busy depth x{:.2f} (Present {})",
                 tuning[0], tuning[1], tuning[2], tuning[3], tuning[4], tuning[5], g_present);
    }
    std::memcpy(map.pData, constants, sizeof(constants));
    ctx->Unmap(g_constants, 0);
    g_hasPrevious = hasPrevious;
    if (!hasPrevious)
        ++g_uncomposed;
    g_current = 1 - g_current;

    PassScope scope;
    ID3D11ShaderResourceView* srvs[7] = { views.srvs[views.packed], g_motion[1 - g_current].srv, depthSrv,
                                          previousDepthSrv, sameEyeDepthSrv, sameEyeColourSrv, colourSrv };
    ID3D11UnorderedAccessView* uavs[4] = { g_motion[g_current].uav, g_composed.uav, g_classes.uav, g_bias.uav };
    // Unused (UseMatrix 0) without both views; the placeholder only keeps the slots bound.
    ID3D11Buffer* cbs[3] = { g_constants, useMatrix ? thisView.buffer : g_constants,
                             useMatrix ? previousView.buffer : g_constants };
    RunPass(ctx, g_decode, 7, srvs, 4, uavs, 3, cbs);
    // Keep this Present's depth for the next one.
    if (depthUsable)
    {
        ctx->CopyResource(g_depthCopies[g_current].texture, depth);
        g_depthCopies[g_current].present = g_present;
    }
    // And its colour input, for the shadow test two Presents on (the slot the decode just read from).
    if (colourUsable)
    {
        ctx->CopyResource(g_colourRing[g_current].texture, colourTex);
        g_colourRing[g_current].present = g_present;
    }
    if (colourTex != nullptr)
        colourTex->Release();
    g_lastDecodePresent = g_present;
    g_lastDecodeLayout = layout;
    g_decodedThisDispatch = true;
    g_decodedThisPresent.store(true, std::memory_order_relaxed);
    if (++g_decodes % 900 == 1)
        LOG_INFO("TAA motion: {} passes decoded, {} without the previous Present's motion, {} with matrix motion; "
                 "{} view copies; disocclusion mask {} passes, {} without the same-eye depth; shadow mask {} "
                 "passes, {} without the same-eye colour",
                 g_decodes, g_uncomposed, g_matrixPasses, g_viewCopies, g_biasPasses, g_biasMissingDepth,
                 g_shadowPasses, g_shadowMissingColour);
}

void TaaMotion_Dx11::AfterDispatch(ID3D11DeviceContext* ctx)
{
    if (!g_decodedThisDispatch)
        return;
    const int mode = g_viewMode.load(std::memory_order_relaxed);
    const bool copy = FrameCapture_Dx11::Busy();
    if (mode == 0 && !copy)
        return;

    // The game's output and colour, still bound after its Dispatch.
    PassViews views;
    ctx->CSGetUnorderedAccessViews(0, 1, &views.output);
    ctx->CSGetShaderResources(0, 3, views.srvs);
    if (views.output == nullptr)
        return;

    PassScope scope;
    // Before the view draws over it.
    if (copy)
        CopyFinal(ctx, views.output, views.srvs[g_layout == Layout::RenderResolution ? 1 : 2]);
    // The render-resolution pass writes YCoCg for the game's upscale, so the motion views would show wrong colours
    // there; the classification view runs anyway and reads that pass's colour as luma.
    const bool renderRes = g_layout == Layout::RenderResolution;
    ID3D11ShaderResourceView* colour = views.srvs[renderRes ? 1 : 2];
    if (mode != 0 && mode != 3 && renderRes)
    {
        static bool logged = false;
        if (!logged)
        {
            logged = true;
            LOG_INFO("TAA motion: the motion views are off for the render-resolution pass (F8 again for the "
                     "classification view, which works there)");
        }
    }
    else if (mode != 0 && colour != nullptr)
    {
        ID3D11ShaderResourceView* srvs[2] = { mode == 3 ? g_classes.srv
                                              : mode == 1 ? g_motion[g_current].srv
                                                          : g_composed.srv,
                                              colour };
        if (mode == 3 && g_classConstants != nullptr)
        {
            D3D11_MAPPED_SUBRESOURCE map {};
            if (SUCCEEDED(ctx->Map(g_classConstants, 0, D3D11_MAP_WRITE_DISCARD, 0, &map)))
            {
                const uint32_t constants[4] = { renderRes ? 1u : 0u, 0, 0, 0 };
                std::memcpy(map.pData, constants, sizeof(constants));
                ctx->Unmap(g_classConstants, 0);
            }
        }
        if (mode == 3)
            RunPass(ctx, g_classShader, 2, srvs, 1, &views.output, 1, &g_classConstants);
        else
            RunPass(ctx, g_viewShader, 2, srvs, 1, &views.output);
    }
}

void TaaMotion_Dx11::BeforeDraw(ID3D11DeviceContext* ctx)
{
    auto& copy = g_views[g_present & 1];
    if (ctx == nullptr || !g_ready || g_viewCopyFailed || g_packedWidth == 0 || copy.present == g_present)
        return;
    // Cheap test first: the base pass's vertex shaders have the view buffer in cb1.
    ID3D11Buffer* view = nullptr;
    ctx->VSGetConstantBuffers(1, 1, &view);
    if (view == nullptr)
        return;
    D3D11_BUFFER_DESC viewDesc {};
    view->GetDesc(&viewDesc);
    bool writesMotion = false;
    if (viewDesc.ByteWidth == kViewBytes)
    {
        ID3D11RenderTargetView* rtvs[D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT] = {};
        ctx->OMGetRenderTargets(D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT, rtvs, nullptr);
        for (auto* rtv : rtvs)
        {
            if (rtv == nullptr)
                continue;
            if (ID3D11Texture2D* texture = TextureFromView(rtv))
            {
                const auto desc = Desc(texture);
                writesMotion |= desc.Width == g_packedWidth && desc.Height == g_packedHeight &&
                                desc.Format == g_packedFormat;
                texture->Release();
            }
            rtv->Release();
        }
    }
    if (writesMotion)
    {
        PassScope scope;
        ctx->CopyResource(copy.buffer, view);
        copy.present = g_present;
        if (g_viewCopies++ == 0)
            LOG_INFO("TAA motion: view constants found (vs cb1 of the base pass, {}x{} motion target)", g_packedWidth,
                     g_packedHeight);
    }
    view->Release();
}

void TaaMotion_Dx11::OnPresent()
{
    AccountPresent();
    ++g_present;
}

uint64_t TaaMotion_Dx11::PresentIndex() { return g_present; }

ID3D11Buffer* TaaMotion_Dx11::ViewBuffer()
{
    const auto& copy = g_views[g_present & 1];
    return copy.present == g_present ? copy.buffer : nullptr;
}

bool TaaMotion_Dx11::RenderResolution() { return g_decodedThisDispatch && g_layout == Layout::RenderResolution; }

ID3D11Texture2D* TaaMotion_Dx11::Decoded() { return g_decodedThisDispatch ? g_motion[g_current].texture : nullptr; }

ID3D11Texture2D* TaaMotion_Dx11::Composed()
{
    return g_decodedThisDispatch && g_hasPrevious ? g_composed.texture : nullptr;
}

ID3D11Texture2D* TaaMotion_Dx11::PreviousDecoded()
{
    return g_decodedThisDispatch && g_hasPrevious ? g_motion[1 - g_current].texture : nullptr;
}

ID3D11Texture2D* TaaMotion_Dx11::FrameDepth() { return g_decodedThisDispatch ? g_depthThisDispatch : nullptr; }

ID3D11Texture2D* TaaMotion_Dx11::PreviousDepth()
{
    return g_decodedThisDispatch ? g_previousDepthThisDispatch : nullptr;
}

ID3D11Texture2D* TaaMotion_Dx11::ColourPresentsAgo(int presents)
{
    for (int i = 0; i < 2; ++i)
    {
        if (g_colourCopies[i] != nullptr && g_finalCopyPresent[i] + presents == g_present)
            return g_colourCopies[i];
    }
    return nullptr;
}

ID3D11Texture2D* TaaMotion_Dx11::OutputPresentsAgo(int presents)
{
    for (int i = 0; i < 2; ++i)
    {
        if (g_finalCopies[i] != nullptr && g_finalCopyPresent[i] + presents == g_present)
            return g_finalCopies[i];
    }
    return nullptr;
}

int TaaMotion_Dx11::MotionMode() { return g_motionMode.load(); }

void TaaMotion_Dx11::SetMotionMode(int mode)
{
    g_motionMode = std::clamp(mode, 0, 3);
    LOG_INFO("TAA motion: same-eye motion {} (overlay)", MotionModeName());
}

float TaaMotion_Dx11::OwnMotionScale() { return g_ownScale.load(); }

ID3D11Texture2D* TaaMotion_Dx11::BiasMask()
{
    return g_decodedThisDispatch && g_biasThisDispatch ? g_bias.texture : nullptr;
}

float TaaMotion_Dx11::BiasStrength() { return g_biasStrength.load(); }
void TaaMotion_Dx11::SetBiasStrength(float strength) { g_biasStrength = std::clamp(strength, 0.0f, 1.0f); }
float TaaMotion_Dx11::BiasNearer() { return g_biasNearer.load(); }
void TaaMotion_Dx11::SetBiasNearer(float ratio) { g_biasNearer = std::clamp(ratio, 1.0f, 3.0f); }
int TaaMotion_Dx11::MaskMode() { return g_maskMode.load(); }
void TaaMotion_Dx11::SetMaskMode(int mode) { g_maskMode = std::clamp(mode, 0, 3); }
float TaaMotion_Dx11::ShadowDrop() { return g_shadowDrop.load(); }
void TaaMotion_Dx11::SetShadowDrop(float drop) { g_shadowDrop = std::clamp(drop, 0.005f, 0.5f); }

float TaaMotion_Dx11::TuningValue(int which)
{
    return which >= 0 && which < kTuningCount ? g_tuning[which].load() : 0.0f;
}

void TaaMotion_Dx11::SetTuningValue(int which, float value)
{
    if (which >= 0 && which < kTuningCount)
        g_tuning[which] = std::clamp(value, kTuningMin[which], kTuningMax[which]);
}

void TaaMotion_Dx11::ResetTuning()
{
    for (int i = 0; i < kTuningCount; ++i)
        g_tuning[i] = kTuningDefaults[i];
}
void TaaMotion_Dx11::SetOwnMotionScale(float scale)
{
    g_ownScale = std::clamp(scale, 0.0f, 2.0f);
}

void TaaMotion_Dx11::CycleMotionMode()
{
    g_motionMode = (g_motionMode.load() + 1) % 4;
    LOG_INFO("TAA motion: same-eye motion {}", MotionModeName());
}

const char* TaaMotion_Dx11::MotionModeName()
{
    switch (g_motionMode.load())
    {
    case 1:
        return "matrix, halo as static";
    case 2:
        return "chained through the other eye";
    case 3:
        return "matrix + object motion chained";
    default:
        return "matrix hybrid";
    }
}

void TaaMotion_Dx11::CycleView()
{
    const int mode = (g_viewMode.load() + 1) % 4;
    g_viewMode = mode;
    LOG_INFO("TAA motion: view {}", ViewName());
}

const char* TaaMotion_Dx11::ViewName()
{
    switch (g_viewMode.load())
    {
    case 1:
        return "decoded (to the previous Present)";
    case 2:
        return "composed (to two Presents back)";
    case 3:
        return "classification (grey/blue matrix, green own motion, amber blending, red chained, magenta halo)";
    default:
        return "off";
    }
}
