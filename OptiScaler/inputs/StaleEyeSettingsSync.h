#pragma once
// When the overlay may show the stale-eye layer's controls and copy the layer's state back into the Config.
//
// The ini's settings are handed to the layer ONCE (StaleEyeShare::SyncLayer), and the layer applies a Set on its own
// thread at its next frame. Until then it reports its own start-up defaults -- and the overlay, which copies the
// layer's state into the Config every frame (so that the ini follows the layer's hotkeys), saved those defaults over
// the ini's: the warp came up Off, the distance 1.5, head movement "Headset only" (the menu
// open in the headset before gameplay). So: nothing is copied back before the push, and after it only once the layer
// reports what was sent (it has applied it), or kSettleMs later (a value the layer keeps otherwise must not lock the
// menu for good). Header only, no Config: taa_test case 35 runs it.
#include "StaleEyeShareAbi.h"

#include <cmath>
#include <cstdint>

namespace StaleEyeSettingsSync
{
constexpr uint64_t kSettleMs = 3000;

// What the ini pushes, compared (showBreaks is never pushed; floats within rounding).
inline bool Same(const XrObserverStaleSettingsV1& a, const XrObserverStaleSettingsV1& b)
{
    return a.mode == b.mode && a.pairingOffset == b.pairingOffset &&
           std::fabs(a.nearThreshold - b.nearThreshold) < 1e-3f && a.bothEyes == b.bothEyes &&
           a.headMode == b.headMode && a.treatMode == b.treatMode && a.strafeOnly == b.strafeOnly &&
           a.depthMode == b.depthMode && a.attachDistanceOnly == b.attachDistanceOnly &&
           a.nearWithBody == b.nearWithBody && a.fillFromOtherEye == b.fillFromOtherEye &&
           a.diagnostics == b.diagnostics;
}

class Gate
{
  public:
    // The ini's settings `sent` were handed to the layer at `nowMs`.
    void Pushed(const XrObserverStaleSettingsV1& sent, uint64_t nowMs)
    {
        _pushed = true;
        _sent = sent;
        _at = nowMs;
    }
    bool WasPushed() const { return _pushed; }
    // True once the layer has what was sent (or kSettleMs after the push); stays true from then on.
    bool Ready(const XrObserverStaleSettingsV1& layerNow, uint64_t nowMs)
    {
        if (!_pushed)
            return false;
        if (!_ready && (Same(layerNow, _sent) || nowMs - _at >= kSettleMs))
            _ready = true;
        return _ready;
    }

  private:
    bool _pushed = false, _ready = false;
    XrObserverStaleSettingsV1 _sent {};
    uint64_t _at = 0;
};
} // namespace StaleEyeSettingsSync
