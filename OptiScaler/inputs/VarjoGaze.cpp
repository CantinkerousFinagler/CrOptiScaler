#include "pch.h"

#include "Config.h"
#include "Logger.h"

#include "VarjoGaze.h"
#include "GazeFilter.h"
#include "StaleEyeShare.h"

#include <atomic>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <format>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace
{
std::atomic<bool> g_flipX { false }, g_flipY { false };
// varjo_Gaze, as remembered from the Varjo SDK (Varjo_types.h), all fields 8 bytes:
//   varjo_Ray leftEye, rightEye, gaze;          // each: double origin[3], double forward[3]
//   double focusDistance, stability;
//   varjo_Nanoseconds captureTime;              // int64
//   varjo_GazeEyeStatus leftStatus, rightStatus; // int64: 0 invalid, 1 visible, 2 compensated, 3 tracked
//   varjo_GazeStatus status;                    // int64: 0 invalid, 1 adjust, 2 valid
//   int64_t frameNumber;
//   double leftPupilSize, rightPupilSize;
// varjo_GetGaze returns it by value: on x64 through a buffer the caller passes. Ours is 1 KB, so a newer, larger struct
// cannot write past it.
struct GazeBuffer
{
    uint8_t bytes[1024];
};
constexpr size_t kLeftOrigin = 0, kLeftForward = 24, kRightOrigin = 48, kRightForward = 72, kBothOrigin = 96,
                 kBothForward = 120, kFocusDistance = 144, kCaptureTime = 160, kLeftStatus = 168, kRightStatus = 176,
                 kStatus = 184, kFrameNumber = 192;

using PFN_SessionInit = void* (*) ();
using PFN_IsGazeAllowed = int32_t (*)(void*);
using PFN_GazeInit = void (*)(void*);
using PFN_GetGaze = GazeBuffer (*)(void*);
using PFN_GetError = int64_t (*)(void*);
using PFN_GetErrorDesc = const char* (*) (int64_t);
using PFN_GetCurrentTime = int64_t (*)(void*);

PFN_SessionInit p_SessionInit = nullptr;
PFN_IsGazeAllowed p_IsGazeAllowed = nullptr;
PFN_GazeInit p_GazeInit = nullptr;
PFN_GetGaze p_GetGaze = nullptr;
PFN_GetError p_GetError = nullptr;
PFN_GetErrorDesc p_GetErrorDesc = nullptr;
PFN_GetCurrentTime p_GetCurrentTime = nullptr;

std::mutex g_mutex; // the sample and the counters: written on the present thread, read on recording threads
void* g_session = nullptr;
bool g_failed = false; // final: logged once
uint64_t g_polls = 0;
int g_attempts = 0;
bool g_layoutChecked = false, g_layoutOk = false;
bool g_loggedInvalid = false, g_loggedFirstValid = false;
uint64_t g_validPolls = 0, g_invalidPolls = 0, g_freshPolls = 0;
int64_t g_lastFrame = -1;
int64_t g_lastStatus = -1; // for logging each change of the tracker's status (the first 40)
int g_statusChanges = 0;
double g_ageSum = 0;
VarjoGaze::Sample g_latest;
VarjoGaze::Sample g_lastValid;
std::chrono::steady_clock::time_point g_lastValidAt;
uint64_t g_heldPolls = 0, g_lostPolls = 0, g_rebuiltEyes = 0;
// The One Euro filter over the combined gaze's angles (GazeFilter.h: pure, tested by fit12_test case 17) with its
// clock here. Reset after a loss longer than the hold.
struct OneEuro
{
    GazeFilter::State f;
    std::chrono::steady_clock::time_point at;
};
OneEuro g_filter;
uint64_t g_spikes = 0;
bool g_loggedSnap = false;

void Smooth(const float in[3], float out[3], std::chrono::steady_clock::time_point now)
{
    const double yaw = std::atan2((double) in[0], std::abs((double) in[2])) * 57.29578;
    const double pitch = std::atan2((double) in[1], std::sqrt((double) in[0] * in[0] + (double) in[2] * in[2])) * 57.29578;
    const double dt = g_filter.f.primed ? std::chrono::duration<double>(now - g_filter.at).count() : 0.0;
    GazeFilter::Step(g_filter.f, yaw, pitch, dt, (double) Config::Instance()->UpscalerFoveaGazeSteadiness.value_or_default(),
                     Config::Instance()->UpscalerFoveaGazeSnap.value_or_default());
    g_filter.at = now;
    g_spikes = g_filter.f.spikes;
    if (g_filter.f.snaps > 0 && !g_loggedSnap)
    {
        g_loggedSnap = true;
        LOG_INFO("Gaze: FIRST snap to where the eyes landed after a saccade (yaw {:.1f}, pitch {:.1f} deg; "
                 "[Inputs] UpscalerFoveaGazeSnap)",
                 g_filter.f.yaw, g_filter.f.pitch);
    }
    const double y = g_filter.f.yaw / 57.29578, p = g_filter.f.pitch / 57.29578;
    out[0] = (float) (std::sin(y) * std::cos(p));
    out[1] = (float) std::sin(p);
    out[2] = (float) (std::cos(y) * std::cos(p));
}

float g_eyeOrigin[2][3] = { { -0.032f, 0.0f, 0.0f }, { 0.032f, 0.0f, 0.0f } }; // last tracked, metres (IPD 64 mm first)
std::string g_status = "not started";

// ---- The OpenXR layer's gaze (XR_EXT_eye_gaze_interaction through tools/xr_observer; VarjoGaze.h: Source) ----
void CalibrationStep(bool haveSample, double rawYaw, double rawPitch); // below
void RawAngles(const float in[3], double& yaw, double& pitch);
bool g_layerSource = false;     // the layer has given a valid gaze: it is THE source from then on (Varjo's samples ignored)
bool g_layerOffersGaze = false; // the layer says the extension is in the instance: in AUTO, Varjo's start waits for it
uint64_t g_layerLocates = 0, g_layerValid = 0, g_layerInvalid = 0;
bool g_loggedLayerWait = false, g_loggedLayerGiveUp = false;
constexpr uint64_t kLayerWaitPolls = 2700; // ~30 s at 90 Presents/s: then AUTO starts Varjo's reader after all

// q v q^-1 (q = x y z w, unit)
void QuatRotate(const float q[4], const float v[3], float out[3])
{
    const float tx = 2.0f * (q[1] * v[2] - q[2] * v[1]), ty = 2.0f * (q[2] * v[0] - q[0] * v[2]),
                tz = 2.0f * (q[0] * v[1] - q[1] * v[0]);
    out[0] = v[0] + q[3] * tx + (q[1] * tz - q[2] * ty);
    out[1] = v[1] + q[3] * ty + (q[2] * tx - q[0] * tz);
    out[2] = v[2] + q[3] * tz + (q[0] * ty - q[1] * tx);
}
// A direction given in view 0's frame, into view 1's: d1 = conj(q1) q0 d0, with q0 / q1 the views' orientations in one
// space (OpenXR frames look along -z; ours along +z: flipped on the way in and out). The headset's cant between its
// two displays, nothing else.
void IntoOtherView(const float q0[4], const float q1[4], const float in[3], float out[3])
{
    const float xr[3] = { in[0], in[1], -in[2] };
    float world[3], other[3];
    QuatRotate(q0, xr, world);
    const float conj1[4] = { -q1[0], -q1[1], -q1[2], q1[3] };
    QuatRotate(conj1, world, other);
    out[0] = other[0];
    out[1] = other[1];
    out[2] = -other[2];
}

// A poll with no valid gaze: the last one is held for kHoldMs (a blink, a dropout), then the gaze is lost.
void HoldOrLose(const std::string& why)
{
    CalibrationStep(false, 0, 0);
    ++g_invalidPolls;
    const double sinceValid =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - g_lastValidAt).count();
    if (g_lastValid.valid && sinceValid <= VarjoGaze::kHoldMs)
    {
        ++g_heldPolls;
        g_latest = g_lastValid;
        g_latest.held = true;
        g_latest.ageMs = g_lastValid.ageMs + sinceValid;
        return;
    }
    ++g_lostPolls;
    g_latest.valid = false;
    g_status = std::format("no valid gaze ({}; {} valid of {} polls)", why, g_validPolls, g_validPolls + g_invalidPolls);
}

// The layer's gaze of this poll taken as the sample: smoothed in view 0's frame (the left eye's), turned into view 1's
// by the views' orientations, and fed to the calibration as Varjo's would be.
void TakeLayerSample(const XrObserverViewsV1& v)
{
    const auto now = std::chrono::steady_clock::now();
    VarjoGaze::Sample s;
    s.valid = true;
    s.source = 1;
    s.status = 2;
    s.leftStatus = s.rightStatus = 2;
    std::memcpy(s.both, v.gazeDirection[0], sizeof(s.both));
    std::memcpy(s.left, v.gazeDirection[0], sizeof(s.left));
    std::memcpy(s.right, v.gazeDirection[1], sizeof(s.right));
    s.ageMs = (double) v.gazeAgeNs / 1e6;
    if (!g_layerSource)
    {
        g_layerSource = true;
        g_filter = OneEuro {}; // another frame than Varjo's: the filter starts over
        LOG_INFO("Gaze: FIRST valid gaze from the OpenXR layer (XR_EXT_eye_gaze_interaction): in the left eye's frame "
                 "({:.3f} {:.3f} {:.3f}), right eye's ({:.3f} {:.3f} {:.3f}), {:.1f} ms before display; the layer is "
                 "the gaze source from now on (Varjo's reader {}). Layer: {}",
                 s.both[0], s.both[1], s.both[2], s.right[0], s.right[1], s.right[2], s.ageMs,
                 g_session != nullptr ? "was running and is ignored" : "is not started", v.text);
    }
    Smooth(s.both, s.smooth, now);
    std::memcpy(s.viewDir[0], s.smooth, sizeof(s.smooth));
    IntoOtherView(v.orientation[0], v.orientation[1], s.smooth, s.viewDir[1]);
    {
        double ry, rp;
        RawAngles(s.both, ry, rp);
        CalibrationStep(true, ry, rp);
    }
    ++g_validPolls;
    ++g_layerValid;
    g_ageSum += s.ageMs;
    g_latest = s;
    g_lastValid = s;
    g_lastValidAt = now;
    const double all = (double) (g_validPolls + g_invalidPolls);
    g_status = std::format("OpenXR layer: tracking, {:.0f}% valid, {:.0f}% held over a dropout, {:.0f}% lost; {:.1f} ms "
                           "old on average; {} glitches ignored",
                           100.0 * g_validPolls / all, 100.0 * g_heldPolls / all, 100.0 * g_lostPolls / all,
                           g_ageSum / (double) g_validPolls, g_spikes);
    if (g_validPolls % 3000 == 0)
        LOG_INFO("Gaze: {}", g_status);
}

template <typename T> T Field(const GazeBuffer& b, size_t at)
{
    T v;
    std::memcpy(&v, b.bytes + at, sizeof(T));
    return v;
}

void Direction(const GazeBuffer& b, size_t at, float out[3])
{
    for (int i = 0; i < 3; ++i)
        out[i] = (float) Field<double>(b, at + i * sizeof(double));
}

double Length(const float v[3]) { return std::sqrt((double) v[0] * v[0] + (double) v[1] * v[1] + (double) v[2] * v[2]); }

std::string LastError()
{
    if (p_GetError == nullptr || g_session == nullptr)
        return "no error call";
    const int64_t e = p_GetError(g_session);
    const char* d = (p_GetErrorDesc != nullptr && e != 0) ? p_GetErrorDesc(e) : nullptr;
    return std::format("error {}{}{}", e, d ? ": " : "", d ? d : "");
}

// ---- In-game calibration ----
constexpr int kCalPoints = 9;
const float kCalYaw[kCalPoints] = { 0, -18, 18, 0, 0, -18, 18, -18, 18 };
const float kCalPitch[kCalPoints] = { 0, 0, 0, 14, -14, 14, 14, -14, -14 };
constexpr double kSettleS = 0.7, kCollectS = 0.9;
// Centring alone settles longer: the overlay is a panel in the headset now, and closing it after the press took the
// whole 0.7 s -- a one-target reading with the shorter settle sampled the eyes still on the panel (8 deg off).
constexpr double kCentreSettleS = 3.0;
struct Calibration
{
    bool active = false;
    int index = 0;
    std::chrono::steady_clock::time_point phaseStart;
    std::vector<std::pair<double, double>> samples;
    double rawYaw[kCalPoints] {}, rawPitch[kCalPoints] {};
    bool ok[kCalPoints] {};
    std::string text = "not calibrated";
    bool centreOnly = false; // StartCentring: point 1 alone, and only the offset is set from it
};
Calibration g_cal;

double Median(std::vector<double> v)
{
    std::sort(v.begin(), v.end());
    return v.empty() ? 0.0 : v[v.size() / 2];
}

// Solves the 3x3 normal equations of y ~ c0 + c1 a + c2 b. False if singular.
bool FitAffine(const std::vector<double>& a, const std::vector<double>& b, const std::vector<double>& y, double c[3])
{
    double m[3][4] {};
    for (size_t i = 0; i < y.size(); ++i)
    {
        const double row[3] = { 1.0, a[i], b[i] };
        for (int r = 0; r < 3; ++r)
        {
            for (int k = 0; k < 3; ++k)
                m[r][k] += row[r] * row[k];
            m[r][3] += row[r] * y[i];
        }
    }
    for (int col = 0; col < 3; ++col)
    {
        int piv = col;
        for (int r = col + 1; r < 3; ++r)
            if (std::abs(m[r][col]) > std::abs(m[piv][col]))
                piv = r;
        if (std::abs(m[piv][col]) < 1e-9)
            return false;
        for (int k = 0; k < 4; ++k)
            std::swap(m[col][k], m[piv][k]);
        for (int r = 0; r < 3; ++r)
        {
            if (r == col)
                continue;
            const double f = m[r][col] / m[col][col];
            for (int k = col; k < 4; ++k)
                m[r][k] -= f * m[col][k];
        }
    }
    for (int r = 0; r < 3; ++r)
        c[r] = m[r][3] / m[r][r];
    return true;
}

void FinishCalibration()
{
    g_cal.active = false;
    std::vector<double> ry, rp, ty, tp;
    for (int i = 0; i < kCalPoints; ++i)
        if (g_cal.ok[i])
        {
            ry.push_back(g_cal.rawYaw[i]);
            rp.push_back(g_cal.rawPitch[i]);
            ty.push_back(kCalYaw[i]);
            tp.push_back(kCalPitch[i]);
        }
    double cy[3], cp[3];
    if (ry.size() < 6 || !FitAffine(ry, rp, ty, cy) || !FitAffine(ry, rp, tp, cp))
    {
        g_cal.text = std::format("calibration FAILED: only {} of {} points had enough tracked samples", ry.size(),
                                 kCalPoints);
        LOG_WARN("Varjo gaze: {}", g_cal.text);
        return;
    }
    // Least squares takes one wild point (a glance away, a blink held through the window) and spreads it over all
    // nine: in one run, eight points within 2 deg of 'pitch reads 5 deg low' and one 25 deg off -> 10 deg
    // typical, rejected. So: while the fit is worse than kCalAcceptDeg and more than 6 points remain, the worst point
    // is dropped and the rest fitted again. Each drop is logged.
    constexpr double kCalAcceptDeg = 4.0;
    double sq = 0, worst = 0, rms = 0;
    int dropped = 0;
    std::string droppedText;
    for (;;)
    {
        sq = 0;
        worst = 0;
        size_t worstAt = 0;
        for (size_t i = 0; i < ry.size(); ++i)
        {
            const double ey = cy[0] + cy[1] * ry[i] + cy[2] * rp[i] - ty[i];
            const double ep = cp[0] + cp[1] * ry[i] + cp[2] * rp[i] - tp[i];
            sq += ey * ey + ep * ep;
            if (std::hypot(ey, ep) > worst)
            {
                worst = std::hypot(ey, ep);
                worstAt = i;
            }
        }
        rms = std::sqrt(sq / (double) ry.size());
        if (rms <= kCalAcceptDeg || ry.size() <= 6)
            break;
        droppedText += std::format("{}target ({:+.0f}, {:+.0f}) read ({:+.1f}, {:+.1f}), {:.1f} deg off",
                                   dropped > 0 ? "; " : "", ty[worstAt], tp[worstAt], ry[worstAt], rp[worstAt], worst);
        ry.erase(ry.begin() + worstAt);
        rp.erase(rp.begin() + worstAt);
        ty.erase(ty.begin() + worstAt);
        tp.erase(tp.begin() + worstAt);
        ++dropped;
        if (!FitAffine(ry, rp, ty, cy) || !FitAffine(ry, rp, tp, cp))
        {
            g_cal.text = "calibration FAILED: the fit broke down after dropping a point";
            LOG_WARN("Varjo gaze: {}", g_cal.text);
            return;
        }
    }
    if (dropped > 0)
        LOG_INFO("Varjo gaze: calibration dropped {} point(s) as outliers: {}", dropped, droppedText);
    // A fit this far off is not a tracker's bias: the targets and the gaze did not agree (one D3D12 game:
    // 14-15 deg typical with a NEGATIVE yaw gain, from targets drawn through a projection with the headset's FOV offsets
    // while the game rendered centred -- saved, it sent the crop the opposite way of the gaze). Nothing is saved.
    if (rms > kCalAcceptDeg)
    {
        g_cal.text = std::format("calibration REJECTED: fit error {:.2f} deg typical, {:.2f} worst (over 4): the targets "
                                 "and the gaze do not agree -- check 'Gaze projection' and that both eyes see the target "
                                 "in one place; nothing saved",
                                 rms, worst);
        LOG_WARN("Varjo gaze: {}; the fit would have been yaw = {:.3f} + {:.4f} raw yaw + {:.4f} raw pitch; pitch = "
                 "{:.3f} + {:.4f} raw yaw + {:.4f} raw pitch",
                 g_cal.text, cy[0], cy[1], cy[2], cp[0], cp[1], cp[2]);
        return;
    }
    auto* cfg = Config::Instance();
    cfg->UpscalerFoveaGazeCal = true;
    cfg->UpscalerFoveaGazeCalYaw0 = (float) cy[0];
    cfg->UpscalerFoveaGazeCalYaw1 = (float) cy[1];
    cfg->UpscalerFoveaGazeCalYaw2 = (float) cy[2];
    cfg->UpscalerFoveaGazeCalPitch0 = (float) cp[0];
    cfg->UpscalerFoveaGazeCalPitch1 = (float) cp[1];
    cfg->UpscalerFoveaGazeCalPitch2 = (float) cp[2];
    // The fit contains the offset and the range: start them over (the offset stays usable as a fine trim).
    cfg->UpscalerFoveaGazeYaw = 0.0f;
    cfg->UpscalerFoveaGazePitch = 0.0f;
    cfg->UpscalerFoveaGazeRangeUp = 1.0f;
    cfg->UpscalerFoveaGazeRangeDown = 1.0f;
    cfg->UpscalerFoveaGazeRangeLeft = 1.0f;
    cfg->UpscalerFoveaGazeRangeRight = 1.0f;
    g_cal.text = std::format("calibrated: {} points{}, fit error {:.2f} deg typical, {:.2f} deg worst", ry.size(),
                             dropped > 0 ? std::format(" ({} dropped)", dropped) : std::string(), rms, worst);
    LOG_INFO("Varjo gaze: {}; yaw = {:.3f} + {:.4f} raw yaw + {:.4f} raw pitch; pitch = {:.3f} + {:.4f} raw yaw + {:.4f} "
             "raw pitch",
             g_cal.text, cy[0], cy[1], cy[2], cp[0], cp[1], cp[2]);
}

// ONE-POINT centring: the median of point 1's samples (the eyes on the centre target for ~1 s) is taken as
// straight ahead; the offset is set so it reads (0, 0), on top of the calibration map if one is saved (whose gains stay).
// The first 'Centre' button read ONE instant while the eyes were on the button on the panel: +-5 deg between presses.
void FinishCentring()
{
    g_cal.active = false;
    if (!g_cal.ok[0])
    {
        g_cal.text = "centring FAILED: too few tracked samples while the target was up; look at it and try again";
        LOG_WARN("Varjo gaze: {}", g_cal.text);
        return;
    }
    double yaw = g_cal.rawYaw[0], pitch = g_cal.rawPitch[0];
    auto* cfg = Config::Instance();
    if (cfg->UpscalerFoveaGazeCal.value_or_default())
    {
        const double y0 = yaw, p0 = pitch;
        yaw = cfg->UpscalerFoveaGazeCalYaw0.value_or_default() + cfg->UpscalerFoveaGazeCalYaw1.value_or_default() * y0 +
              cfg->UpscalerFoveaGazeCalYaw2.value_or_default() * p0;
        pitch = cfg->UpscalerFoveaGazeCalPitch0.value_or_default() +
                cfg->UpscalerFoveaGazeCalPitch1.value_or_default() * y0 +
                cfg->UpscalerFoveaGazeCalPitch2.value_or_default() * p0;
    }
    if (std::abs(yaw) > 20.0 || std::abs(pitch) > 20.0)
    {
        g_cal.text = std::format("centring FAILED: the gaze read ({:+.1f}, {:+.1f}) deg at the centre target, too far for a bias",
                                 yaw, pitch);
        LOG_WARN("Varjo gaze: {}", g_cal.text);
        return;
    }
    VarjoGaze::SetOffset((float) -yaw, (float) -pitch);
    g_cal.text = std::format("centred: the gaze read ({:+.1f}, {:+.1f}) deg at the centre target; offset now ({:+.1f}, {:+.1f})",
                             yaw, pitch, VarjoGaze::OffsetYaw(), VarjoGaze::OffsetPitch());
    LOG_INFO("Varjo gaze: {}", g_cal.text);
}

// Called from Poll with every tracked (not held) sample's raw angles, and once per Poll to advance.
void CalibrationStep(bool haveSample, double rawYaw, double rawPitch)
{
    if (!g_cal.active)
        return;
    const auto now = std::chrono::steady_clock::now();
    const double t = std::chrono::duration<double>(now - g_cal.phaseStart).count();
    const double settle = g_cal.centreOnly ? kCentreSettleS : kSettleS;
    if (haveSample && t >= settle && t < settle + kCollectS)
        g_cal.samples.emplace_back(rawYaw, rawPitch);
    if (t < settle + kCollectS)
        return;
    const int i = g_cal.index;
    std::vector<double> ys, ps;
    for (auto& [y, p] : g_cal.samples)
    {
        ys.push_back(y);
        ps.push_back(p);
    }
    g_cal.ok[i] = ys.size() >= 15;
    if (g_cal.ok[i])
    {
        g_cal.rawYaw[i] = Median(ys);
        g_cal.rawPitch[i] = Median(ps);
    }
    LOG_INFO("Varjo gaze: calibration point {} at ({:+.0f}, {:+.0f}) deg: {} samples -> raw ({:+.2f}, {:+.2f}){}", i + 1,
             kCalYaw[i], kCalPitch[i], ys.size(), g_cal.rawYaw[i], g_cal.rawPitch[i],
             g_cal.ok[i] ? "" : " -- too few, left out");
    g_cal.samples.clear();
    g_cal.phaseStart = now;
    if (g_cal.centreOnly)
        FinishCentring();
    else if (++g_cal.index >= kCalPoints)
        FinishCalibration();
    else
        g_cal.text = std::format("calibrating: look at the yellow target ({} of {})", g_cal.index + 1, kCalPoints);
}

// Raw angles (deg) of a direction, the flips applied.
void RawAngles(const float in[3], double& yaw, double& pitch)
{
    const double x = g_flipX.load() ? -in[0] : in[0];
    const double y = g_flipY.load() ? -in[1] : in[1];
    const double z = std::abs((double) in[2]);
    yaw = std::atan2(x, z) * 57.29578;
    pitch = std::atan2(y, std::sqrt(x * x + z * z)) * 57.29578;
}

void Fail(const std::string& why)
{
    g_failed = true;
    g_status = "off: " + why;
    LOG_ERROR("Varjo gaze: {}", why);
}

// Varjo Base's runtime, from its install folder (the OpenXR runtime's copy first).
bool Load()
{
    const wchar_t* candidates[] = { L"C:\\Program Files\\Varjo\\varjo-openxr\\VarjoLib.dll",
                                    L"C:\\Program Files\\Varjo\\varjo-compositor\\VarjoLib.dll" };
    for (const wchar_t* path : candidates)
    {
        HMODULE lib = LoadLibraryW(path);
        if (lib == nullptr)
            continue;
        p_SessionInit = (PFN_SessionInit) GetProcAddress(lib, "varjo_SessionInit");
        p_IsGazeAllowed = (PFN_IsGazeAllowed) GetProcAddress(lib, "varjo_IsGazeAllowed");
        p_GazeInit = (PFN_GazeInit) GetProcAddress(lib, "varjo_GazeInit");
        p_GetGaze = (PFN_GetGaze) GetProcAddress(lib, "varjo_GetGaze");
        p_GetError = (PFN_GetError) GetProcAddress(lib, "varjo_GetError");
        p_GetErrorDesc = (PFN_GetErrorDesc) GetProcAddress(lib, "varjo_GetErrorDesc");
        p_GetCurrentTime = (PFN_GetCurrentTime) GetProcAddress(lib, "varjo_GetCurrentTime");
        if (p_SessionInit && p_IsGazeAllowed && p_GazeInit && p_GetGaze && p_GetCurrentTime)
        {
            char narrow[MAX_PATH] {};
            WideCharToMultiByte(CP_UTF8, 0, path, -1, narrow, MAX_PATH, nullptr, nullptr);
            LOG_INFO("Varjo gaze: runtime loaded from {}", narrow);
            return true;
        }
        FreeLibrary(lib);
    }
    return false;
}

// The blocking Varjo calls -- a library load, the service handshake, the gaze init -- ran on the PRESENT thread (Poll is
// called from the swapchain's Present, at poll 300) and froze a game at launch the first time Varjo's eye tracker
// was live (the log stopped at exactly that Present, nothing else of ours ran). They run on a worker now,
// outside g_mutex; the result is committed under the lock. A worker that never returns costs the gaze, not the game.
std::atomic<bool> g_starting { false };
std::atomic<int> g_startAttempts { 0 };
int g_startedAtPoll = 0;
bool g_loggedStartHang = false;

void StartOnWorker()
{
    const int attempt = ++g_startAttempts;
    std::string failure;
    void* session = nullptr;
    bool retryLater = false;
    if (p_SessionInit == nullptr && !Load())
        failure = "VarjoLib.dll not found in Varjo Base's folder (or missing the gaze calls)";
    else
    {
        session = p_SessionInit();
        if (session == nullptr)
            retryLater = true;
        else if (p_IsGazeAllowed(session) == 0)
            failure = "eye tracking is not allowed for applications -- enable it in Varjo Base (privacy settings)";
        else
        {
            p_GazeInit(session);
            if (p_GetError != nullptr && p_GetError(session) != 0)
            {
                const int64_t e = p_GetError(session);
                const char* d = (p_GetErrorDesc != nullptr) ? p_GetErrorDesc(e) : nullptr;
                failure = std::format("varjo_GazeInit failed: error {}{}{}", e, d ? ": " : "", d ? d : "");
            }
        }
    }
    std::lock_guard lock(g_mutex);
    g_attempts = attempt;
    if (!failure.empty())
        Fail(failure);
    else if (retryLater)
    {
        g_status = std::format("no Varjo session (attempt {}; is Varjo Base running?)", attempt);
        LOG_WARN("Varjo gaze: varjo_SessionInit returned nothing (attempt {}); retrying later", attempt);
        if (attempt >= 5)
            Fail("no Varjo session after 5 attempts");
    }
    else
    {
        g_session = session;
        g_status = "started, waiting for a tracked sample";
        LOG_INFO("Varjo gaze: session open, eye tracking allowed, gaze started (on a worker thread)");
    }
    g_starting = false;
}

// The first sample the tracker calls valid must look like one, or nothing is trusted.
bool CheckLayout(const GazeBuffer& b)
{
    float l[3], r[3], c[3];
    Direction(b, kLeftForward, l);
    Direction(b, kRightForward, r);
    Direction(b, kBothForward, c);
    const int64_t ls = Field<int64_t>(b, kLeftStatus), rs = Field<int64_t>(b, kRightStatus);
    // The combined gaze must be a unit vector. An eye the tracker lost at this instant (status 0 -- a blink, or the
    // first sample) comes back all zero, which is not a layout error: the first version failed on exactly that
    // (left (0 0 0), status 0, while the gaze and the right eye were unit length).
    auto eyeOk = [](const float* v, int64_t st)
    { return std::abs(Length(v) - 1.0) < 0.05 || (st == 0 && Length(v) < 1e-6); };
    const bool unit = std::abs(Length(c) - 1.0) < 0.05 && eyeOk(l, ls) && eyeOk(r, rs);
    const bool statuses = ls >= 0 && ls <= 3 && rs >= 0 && rs <= 3;
    LOG_INFO("Varjo gaze: first sample marked valid: gaze ({:.4f} {:.4f} {:.4f}), left ({:.4f} {:.4f} {:.4f}), right "
             "({:.4f} {:.4f} {:.4f}), eye statuses {} {}, frame {} -> layout {}",
             c[0], c[1], c[2], l[0], l[1], l[2], r[0], r[1], r[2], ls, rs, Field<int64_t>(b, kFrameNumber),
             unit && statuses ? "OK" : "NOT as expected (directions not unit length or unknown statuses)");
    return unit && statuses;
}
} // namespace

namespace VarjoGaze
{
bool StartedAtLaunch()
{
    static const bool atLaunch = Config::Instance()->UpscalerFoveaGaze.value_or_default();
    return atLaunch;
}

// The tracker is started only by a launch that began with the setting on (a library load and a session: never made
// or torn down in the middle of a run). The overlay's 'Eye tracking' switch turns its USE off and back
// on at once; turned on where the launch began without it, it starts with the next launch.
bool Enabled() { return StartedAtLaunch() && Config::Instance()->UpscalerFoveaGaze.value_or_default(); }

int Source() { return std::clamp(Config::Instance()->UpscalerFoveaGazeSource.value_or_default(), 0, 2); }

void Poll()
{
    if (!Enabled())
        return;
    std::lock_guard lock(g_mutex);
    ++g_polls;
    // The OpenXR layer's gaze first (VarjoGaze.h: Source). Once it has given one, it is the source for good: two trackers
    // in two frames must not take turns.
    const int source = VarjoGaze::Source();
    if (source != 1)
    {
        XrObserverViewsV1 v { sizeof(v) };
        if (StaleEyeShare::LayerViews(&v))
        {
            g_layerLocates = v.locates;
            if (v.gazeState >= 1)
                g_layerOffersGaze = true;
            if (v.gazeValid != 0)
                return TakeLayerSample(v);
        }
        if (g_layerSource)
        {
            ++g_layerInvalid;
            return HoldOrLose(std::format("the OpenXR layer's gaze is not valid now; {}", v.text));
        }
        if (source == 2)
        {
            CalibrationStep(false, 0, 0);
            g_status = std::format("waiting for the OpenXR layer's gaze ({} locates seen; {})", g_layerLocates,
                                   g_layerLocates > 0 ? v.text : "the layer has located no views yet, or is not loaded");
            return;
        }
        if (g_layerOffersGaze && g_polls < kLayerWaitPolls)
        {
            if (!g_loggedLayerWait)
            {
                g_loggedLayerWait = true;
                LOG_INFO("Gaze: the OpenXR layer has the eye gaze extension; Varjo's reader waits for its gaze (up to "
                         "{} polls). Layer: {}",
                         kLayerWaitPolls, v.text);
            }
            CalibrationStep(false, 0, 0);
            g_status = std::format("waiting for the OpenXR layer's gaze ({})", v.text);
            return;
        }
        if (g_layerOffersGaze && !g_loggedLayerGiveUp)
        {
            g_loggedLayerGiveUp = true;
            LOG_WARN("Gaze: no valid gaze from the OpenXR layer after {} polls ({}); Varjo's reader is started after all "
                     "(the layer's gaze still takes over if it comes)",
                     g_polls, v.text);
        }
    }
    if (g_failed)
        return;
    if (g_session == nullptr)
    {
        if (g_starting)
        {
            // The worker is still inside Varjo's calls. The game goes on; say so once if it takes long.
            if (!g_loggedStartHang && g_polls - g_startedAtPoll > 900)
            {
                g_loggedStartHang = true;
                LOG_WARN("Varjo gaze: the session start has not returned after ~900 Presents (Varjo Base?); the game "
                         "goes on without the gaze until it does");
            }
            return;
        }
        // First try after startup has settled, then every ~10 s -- on a worker, never on this (the present) thread.
        if (g_polls >= 300 && (g_polls - 300) % 900 == 0)
        {
            g_starting = true;
            g_startedAtPoll = g_polls;
            std::thread([] { StartOnWorker(); }).detach();
        }
        return;
    }
    const GazeBuffer b = p_GetGaze(g_session);
    const int64_t status = Field<int64_t>(b, kStatus);
    // Every change of status, with the eyes' statuses: a calibration in progress, a headset taken off or a tracker that
    // never starts all look like "status 0" from one line (the second run of this code sat at 0 all session).
    if (status != g_lastStatus && g_statusChanges < 40)
    {
        ++g_statusChanges;
        LOG_INFO("Varjo gaze: status {} -> {} at poll {} (eyes {} {}, {} valid polls so far)", g_lastStatus, status,
                 g_polls, Field<int64_t>(b, kLeftStatus), Field<int64_t>(b, kRightStatus), g_validPolls);
    }
    g_lastStatus = status;
    if (status < 0 || status > 2)
    {
        if (!g_layoutChecked)
            return Fail(std::format("varjo_Gaze.status reads {} -- the struct layout is not the remembered one", status));
    }
    if (status != 2)
    {
        CalibrationStep(false, 0, 0);
        ++g_invalidPolls;
        const double sinceValid =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - g_lastValidAt).count();
        if (g_lastValid.valid && sinceValid <= VarjoGaze::kHoldMs)
        {
            ++g_heldPolls;
            g_latest = g_lastValid;
            g_latest.held = true;
            g_latest.ageMs = g_lastValid.ageMs + sinceValid;
            return;
        }
        ++g_lostPolls;
        g_latest.valid = false;
        g_latest.status = status;
        if (!g_loggedInvalid && g_invalidPolls >= 600 && g_validPolls == 0)
        {
            g_loggedInvalid = true;
            LOG_WARN("Varjo gaze: 600 polls and no valid sample yet (status {}: 0 = not tracking, 1 = adjust the "
                     "headset) -- calibrate in Varjo Base",
                     status);
        }
        g_status = std::format("no valid gaze (status {}; {} valid of {} polls)", status, g_validPolls,
                               g_validPolls + g_invalidPolls);
        return;
    }
    if (!g_layoutChecked)
    {
        g_layoutChecked = true;
        g_layoutOk = CheckLayout(b);
        if (!g_layoutOk)
            return Fail("the first valid sample did not look like one; gaze disabled (see the line above)");
    }
    Sample s;
    s.valid = true;
    s.status = status;
    Direction(b, kLeftForward, s.left);
    Direction(b, kRightForward, s.right);
    Direction(b, kBothForward, s.both);
    s.leftStatus = Field<int64_t>(b, kLeftStatus);
    s.rightStatus = Field<int64_t>(b, kRightStatus);
    // An eye the tracker has: remember where it is. An eye it lost: its ray toward the point both eyes converge on (the
    // combined ray at the focus distance), from that eye's last known position -- not the combined direction itself,
    // which is off by the vergence angle for anything near.
    float origin[3][3];
    Direction(b, kLeftOrigin, origin[0]);
    Direction(b, kRightOrigin, origin[1]);
    Direction(b, kBothOrigin, origin[2]);
    const int64_t eyeStatus[2] = { s.leftStatus, s.rightStatus };
    float* eyeDir[2] = { s.left, s.right };
    double focus = Field<double>(b, kFocusDistance);
    if (!(focus > 0.2 && focus < 50.0))
        focus = 2.0;
    for (int e = 0; e < 2; ++e)
    {
        if (eyeStatus[e] >= 2)
        {
            if (Length(origin[e]) > 1e-4 && Length(origin[e]) < 0.2)
                std::memcpy(g_eyeOrigin[e], origin[e], sizeof(origin[e]));
            continue;
        }
        float p[3], d[3];
        for (int i = 0; i < 3; ++i)
            p[i] = origin[2][i] + s.both[i] * (float) focus;
        for (int i = 0; i < 3; ++i)
            d[i] = p[i] - g_eyeOrigin[e][i];
        const double len = Length(d);
        if (len > 1e-6)
        {
            for (int i = 0; i < 3; ++i)
                eyeDir[e][i] = (float) (d[i] / len);
            ++g_rebuiltEyes;
        }
        else
            std::memcpy(eyeDir[e], s.both, sizeof(s.both));
    }
    s.leftStatus = s.rightStatus = 2; // both rays are usable now (tracked or rebuilt)
    const int64_t now = p_GetCurrentTime(g_session);
    s.ageMs = (double) (now - Field<int64_t>(b, kCaptureTime)) / 1e6;
    const int64_t frame = Field<int64_t>(b, kFrameNumber);
    if (frame != g_lastFrame)
        ++g_freshPolls;
    g_lastFrame = frame;
    ++g_validPolls;
    g_ageSum += s.ageMs;
    Smooth(s.both, s.smooth, std::chrono::steady_clock::now());
    std::memcpy(s.viewDir[0], s.smooth, sizeof(s.smooth)); // Varjo's HMD frame: the same for both eyes
    std::memcpy(s.viewDir[1], s.smooth, sizeof(s.smooth));
    {
        double ry, rp;
        RawAngles(s.both, ry, rp);
        CalibrationStep(true, ry, rp);
    }
    g_latest = s;
    g_lastValid = s;
    g_lastValidAt = std::chrono::steady_clock::now();
    const double all = (double) (g_validPolls + g_invalidPolls);
    g_status = std::format("tracking: {:.0f}% valid, {:.0f}% held over a dropout, {:.0f}% lost; {:.1f} ms old on average; "
                           "{} glitches ignored",
                           100.0 * g_validPolls / all, 100.0 * g_heldPolls / all, 100.0 * g_lostPolls / all,
                           g_ageSum / (double) g_validPolls, g_spikes);
    if (!g_loggedFirstValid)
    {
        g_loggedFirstValid = true;
        LOG_INFO("Varjo gaze: FIRST valid gaze in use ({:.1f} ms old)", s.ageMs);
    }
    if (g_validPolls % 3000 == 0)
        LOG_INFO("Varjo gaze: {}", g_status);
}

Sample Latest()
{
    std::lock_guard lock(g_mutex);
    return g_latest;
}

void SetFlip(bool x, bool y)
{
    g_flipX = x;
    g_flipY = y;
    LOG_INFO("Varjo gaze: flips x {} y {}", x, y);
}

void StartCalibration()
{
    std::lock_guard lock(g_mutex);
    g_cal = Calibration {};
    g_cal.active = true;
    g_cal.phaseStart = std::chrono::steady_clock::now();
    g_cal.text = std::format("calibrating: look at the yellow target (1 of {})", kCalPoints);
    LOG_INFO("Varjo gaze: calibration started ({} points)", kCalPoints);
}

void ClearCalibration()
{
    std::lock_guard lock(g_mutex);
    g_cal = Calibration {};
    Config::Instance()->UpscalerFoveaGazeCal = false;
    LOG_INFO("Varjo gaze: calibration cleared");
}

bool CalibrationTarget(float& yawDeg, float& pitchDeg)
{
    std::lock_guard lock(g_mutex);
    if (!g_cal.active)
        return false;
    yawDeg = kCalYaw[g_cal.index];
    pitchDeg = kCalPitch[g_cal.index];
    return true;
}

std::string CalibrationText()
{
    std::lock_guard lock(g_mutex);
    if (!g_cal.active && g_cal.text == "not calibrated" && Config::Instance()->UpscalerFoveaGazeCal.value_or_default())
        return "calibrated (saved)";
    return g_cal.text;
}

void SetSteadiness(float s) { Config::Instance()->UpscalerFoveaGazeSteadiness = std::clamp(s, 0.0f, 1.0f); }
float Steadiness() { return Config::Instance()->UpscalerFoveaGazeSteadiness.value_or_default(); }

void SetOffset(float yawDeg, float pitchDeg)
{
    Config::Instance()->UpscalerFoveaGazeYaw = std::clamp(yawDeg, -20.0f, 20.0f);
    Config::Instance()->UpscalerFoveaGazePitch = std::clamp(pitchDeg, -20.0f, 20.0f);
}

void StartCentring()
{
    std::lock_guard lock(g_mutex);
    g_cal = Calibration {};
    g_cal.active = true;
    g_cal.centreOnly = true;
    g_cal.phaseStart = std::chrono::steady_clock::now();
    g_cal.text = "centring: close the overlay and look at the yellow target in the middle until it goes";
    LOG_INFO("Varjo gaze: centring started (one target, {:.1f} s)", kCentreSettleS + kCollectS);
}

float OffsetYaw() { return Config::Instance()->UpscalerFoveaGazeYaw.value_or_default(); }
float OffsetPitch() { return Config::Instance()->UpscalerFoveaGazePitch.value_or_default(); }

void SetRange(int which, float gain)
{
    auto* cfg = Config::Instance();
    gain = std::clamp(gain, 0.5f, 2.0f);
    switch (which)
    {
    case 0:
        cfg->UpscalerFoveaGazeRangeUp = gain;
        break;
    case 1:
        cfg->UpscalerFoveaGazeRangeDown = gain;
        break;
    case 2:
        cfg->UpscalerFoveaGazeRangeLeft = gain;
        break;
    default:
        cfg->UpscalerFoveaGazeRangeRight = gain;
        break;
    }
}

float Range(int which)
{
    auto* cfg = Config::Instance();
    switch (which)
    {
    case 0:
        return cfg->UpscalerFoveaGazeRangeUp.value_or_default();
    case 1:
        return cfg->UpscalerFoveaGazeRangeDown.value_or_default();
    case 2:
        return cfg->UpscalerFoveaGazeRangeLeft.value_or_default();
    default:
        return cfg->UpscalerFoveaGazeRangeRight.value_or_default();
    }
}

void Correct(const float in[3], float out[3])
{
    double yaw, pitch;
    RawAngles(in, yaw, pitch);
    // The in-game calibration's map, if there is one; then the offset (+ right, + up, a fine trim after calibrating),
    // then each direction's range around that corrected centre.
    auto* cfg = Config::Instance();
    if (cfg->UpscalerFoveaGazeCal.value_or_default())
    {
        const double y0 = yaw, p0 = pitch;
        yaw = cfg->UpscalerFoveaGazeCalYaw0.value_or_default() + cfg->UpscalerFoveaGazeCalYaw1.value_or_default() * y0 +
              cfg->UpscalerFoveaGazeCalYaw2.value_or_default() * p0;
        pitch = cfg->UpscalerFoveaGazeCalPitch0.value_or_default() +
                cfg->UpscalerFoveaGazeCalPitch1.value_or_default() * y0 +
                cfg->UpscalerFoveaGazeCalPitch2.value_or_default() * p0;
    }
    yaw += OffsetYaw();
    pitch += OffsetPitch();
    yaw *= yaw >= 0.0 ? Range(3) : Range(2);
    pitch *= pitch >= 0.0 ? Range(0) : Range(1);
    yaw = std::clamp(yaw, -85.0, 85.0);
    pitch = std::clamp(pitch, -85.0, 85.0);
    const double yr = yaw / 57.29578, pr = pitch / 57.29578;
    out[0] = (float) (std::sin(yr) * std::cos(pr));
    out[1] = (float) std::sin(pr);
    out[2] = (float) (std::cos(yr) * std::cos(pr));
}

bool FlipX() { return g_flipX.load(); }
bool FlipY() { return g_flipY.load(); }

std::string StatusText()
{
    std::lock_guard lock(g_mutex);
    return g_status;
}
} // namespace VarjoGaze
