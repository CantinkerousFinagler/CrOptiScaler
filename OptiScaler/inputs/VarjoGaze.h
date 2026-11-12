#pragma once

#include <cstdint>
#include <string>

// Eye tracking from a Varjo headset (up to 200 Hz), read through Varjo Base's own runtime (VarjoLib.dll, loaded from
// its install folder) in a gaze-only session: no frames submitted, nothing drawn. For the foveal crop to follow the
// eyes. [Inputs] UpscalerFoveaGaze (startup). Polled once per Present; every failure is logged once and leaves the
// gaze invalid (the crop then stays where it was).
//
// The varjo_Gaze layout is from memory of the SDK (no headers ship with Varjo Base), so the first sample is checked
// before any value is used: unit-length directions and known status values, else the log prints what it saw.
namespace VarjoGaze
{
bool Enabled();         // started at the launch AND the setting on now (the overlay's switch is live for off / on again)
bool StartedAtLaunch(); // [Inputs] UpscalerFoveaGaze as the launch began
void Poll(); // present thread

struct Sample
{
    bool valid = false;  // tracked (or held: see below), and the layout check passed
    float left[3] {}, right[3] {}, both[3] {}; // gaze directions in the HMD frame as Varjo gives them
    int64_t status = 0, leftStatus = 0, rightStatus = 0;
    double ageMs = 0; // capture to poll
    bool held = false; // the tracker lost the eyes for a moment: the last valid sample, up to kHoldMs old
    // Both eyes' gaze through a One Euro filter (steady while the eyes hold still, quick on a saccade): what the crop
    // and the marker use, for both eyes. The raw per-eye rays jittered and alternated between the eyes' frames.
    float smooth[3] {};
    // Where the sample came from: 0 Varjo's VarjoLib (the HMD frame, the same direction for both eyes), 1 the OpenXR
    // layer (XR_EXT_eye_gaze_interaction through tools/xr_observer, any headset: `both` / `smooth` are in VIEW 0's frame,
    // the left eye's, and viewDir[1] is the smoothed gaze turned into view 1's frame by the headset's own cant).
    int source = 0;
    // The smoothed gaze in each EYE's frame (0 left, 1 right; x right, y up, z forward positive): what PlaceGaze
    // projects with that eye's projection. With source 0 both are `smooth`.
    float viewDir[2][3] {};
};
// [Inputs] UpscalerFoveaGazeSource: 0 AUTO (the OpenXR layer's gaze when it gives one -- Varjo's reader is started only
// if the layer has none after ~30 s or says the runtime lacks the extension), 1 Varjo's reader only, 2 the layer only.
int Source();
// Short dropouts are ridden out (a tracker can lose one eye or both for 0.1-0.2 s at a time; 68% of polls were valid in one measurement).
constexpr double kHoldMs = 150.0;
Sample Latest();

// Overlay: live sign flips for the mapping (the axis convention is the part most likely to be wrong).
void SetFlip(bool x, bool y);
// A constant correction, degrees: + right, + up ([Inputs] UpscalerFoveaGazeYaw / Pitch, saved with the ini). The first
// working run put the gaze low and to the right in both eyes -- a tilt between the tracker's frame and the rendered
// view, or calibration bias; zeroed by eye in the overlay.
void SetOffset(float yawDeg, float pitchDeg);
float OffsetYaw();
float OffsetPitch();
// Range: a gain per direction around the corrected centre, so zeroing the centre (the offset) and stretching toward an
// edge are independent. In the first tuned run the cross did not follow that high at the top (trackers
// commonly under-read upward gaze). [Inputs] UpscalerFoveaGazeRangeUp / Down / Left / Right, 1 = as measured.
void SetRange(int which, float gain); // 0 up, 1 down, 2 left, 3 right
float Range(int which);
// Applies the flips, the offset and the range to a direction (x right, y up, z forward taken positive).
void Correct(const float in[3], float out[3]);
// [Inputs] UpscalerFoveaGazeSteadiness, 0..1 (live): higher = steadier while fixating, a little slower to settle.
void SetSteadiness(float s);
float Steadiness();
// In-game calibration: 9 targets drawn at known angles of the rendered view (head-locked), ~1.6 s each (0.7 s to settle,
// 0.9 s of raw samples, median); then per axis an affine map raw (yaw, pitch) -> view (yaw, pitch) by least squares,
// which replaces the offset and range (both reset; the offset sliders remain as a fine trim). Saved with the ini
// ([Inputs] UpscalerFoveaGazeCal + 6 coefficients). Manual offset/range could not be tuned against the tracker's noise.
void StartCalibration();
void ClearCalibration();
// ONE-POINT centring (9-point runs read gains ~1.0 and a constant ~4 deg pitch bias): the centre
// target alone (0.7 s to settle, 0.9 s of samples, median), then the offset is set so that reading is straight ahead,
// on top of any calibration map (its gains stay). The result is in CalibrationText. (A first version took the gaze of
// the instant the button was pressed: the eyes were on the button, +-5 deg between presses.)
void StartCentring();
// While calibrating: the target to draw, in degrees of the rendered view (+ right, + up).
bool CalibrationTarget(float& yawDeg, float& pitchDeg);
std::string CalibrationText();
bool FlipX();
bool FlipY();
std::string StatusText();
} // namespace VarjoGaze
