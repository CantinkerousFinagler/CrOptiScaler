#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

// THE GAZE'S FILTER, pure (no clock, no Config): a One Euro filter over the combined gaze's angles (degrees), fed one
// sample a poll by VarjoGaze.cpp (Varjo's reader or the OpenXR layer's gaze alike) and a made-up saccade by
// fit12_test case 17. Steady while the eyes hold still, quick on a saccade; a sample this far (kJumpDeg) from the
// filtered gaze is a glitch unless the next one lands near it (kConfirmDeg): blinks and a half-covered pupil produced
// single-sample jumps the speed term let straight through, so a real saccade costs one sample.
//
// THE SNAP (the eye beat the crop to the other side): during a saccade consecutive
// polls lie farther apart than kConfirmDeg, so the filter HOLDS until the eyes have landed, and then eased toward the
// landing point over 2-3 polls with a tail of small steps (712 crop moves in three minutes of looking around) -- the
// crop arrived after the eye. With `snap`, the poll that confirms a jump puts the filter AT the new place at once.
namespace GazeFilter
{
constexpr double kJumpDeg = 3.0, kConfirmDeg = 2.0;
constexpr double kBeta = 0.04, kDCutoff = 1.0;

struct State
{
    bool primed = false;
    double yaw = 0, pitch = 0, dYaw = 0, dPitch = 0;
    bool pending = false; // a jump seen once, waiting for the next sample to confirm it
    double pendYaw = 0, pendPitch = 0;
    uint64_t spikes = 0, snaps = 0;
};

inline double Alpha(double cutoffHz, double dt)
{
    const double tau = 1.0 / (6.283185307 * cutoffHz);
    return 1.0 / (1.0 + tau / dt);
}

// One sample. dt: seconds since the last sample (anything for the first). steadiness 0..1: the cutoff while fixating,
// 4 Hz .. 0.3 Hz (the speed term opens it on a saccade). snap: see above; false = the filter as it was.
inline void Step(State& f, double yaw, double pitch, double dt, double steadiness, bool snap)
{
    if (!f.primed)
    {
        f = State {};
        f.primed = true;
        f.yaw = yaw;
        f.pitch = pitch;
        return;
    }
    const double jump = std::hypot(yaw - f.yaw, pitch - f.pitch);
    if (jump > kJumpDeg && !(f.pending && std::hypot(yaw - f.pendYaw, pitch - f.pendPitch) < kConfirmDeg))
    {
        // Unconfirmed jump: hold the filtered gaze, remember this one.
        f.pending = true;
        f.pendYaw = yaw;
        f.pendPitch = pitch;
        ++f.spikes;
        return;
    }
    const bool confirmed = f.pending && jump > kJumpDeg;
    f.pending = false;
    if (snap && confirmed)
    {
        f.yaw = yaw;
        f.pitch = pitch;
        f.dYaw = f.dPitch = 0; // a speed of 0: the cutoff closes at once, the noise of the new fixation stays out
        ++f.snaps;
        return;
    }
    // After a dropout, carry on from the filtered gaze (dt capped) instead of restarting at the first new sample,
    // which is often a noisy one.
    dt = std::clamp(dt, 0.001, 0.1);
    const double st = std::clamp(steadiness, 0.0, 1.0);
    const double minCutoff = 4.0 * std::pow(0.3 / 4.0, st);
    const double ad = Alpha(kDCutoff, dt);
    f.dYaw += ad * ((yaw - f.yaw) / dt - f.dYaw);
    f.dPitch += ad * ((pitch - f.pitch) / dt - f.dPitch);
    const double speed = std::sqrt(f.dYaw * f.dYaw + f.dPitch * f.dPitch); // deg/s
    const double a = Alpha(minCutoff + kBeta * speed, dt);
    f.yaw += a * (yaw - f.yaw);
    f.pitch += a * (pitch - f.pitch);
}
} // namespace GazeFilter
