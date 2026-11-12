#pragma once

#include "StaleEyeShareAbi.h"

// Publishing side of the stale-eye camera ABI (see StaleEyeShareAbi.h). Game backends call Publish once per Present
// with the game's current-to-previous clip matrix; the export RvrTools_GetStaleEyeCameraV1 hands out the latest.
// [Inputs] StaleEyeCameraShare (default off) turns it on.
namespace StaleEyeShare
{
bool Enabled();
// tangents: this Present's projection as (left, right, up, down) frustum tangents, or nullptr if unknown.
// projection: this Present's full projection (16 floats, row-major) or nullptr; tangents are derived from it.
void Publish(const float currentToPrevious[16], uint32_t width, uint32_t height, bool reversedDepth,
             const float* tangents, const float* projection = nullptr);
// The latest publication's sequence and size (0 before the first).
uint64_t LatestSequence(uint32_t* width, uint32_t* height);
// Depth share state, set by the game backend.
struct DepthShareInfo
{
    uint64_t generation = 0;
    uint64_t handles[kStaleEyeDepthSlots] = {};
    uint32_t width = 0, height = 0;
    uint64_t writtenFence = 0, readFence = 0; // 0: no fence (see StaleEyeShareAbi.h)
    bool legacyHandles = false;               // the old kind of shared handle
};
void SetDepthShare(const DepthShareInfo& info);
void SetDepthSequence(uint64_t sequence);
// Whether the depth the game shares from now on carries the marks of the player's kind (depthMarks).
void SetDepthMarks(bool marks);
// The highest sequence the layer has reported its reads queued for (0 = none).
uint64_t ReportedRead();
// Whether the layer currently asks for depth (RvrTools_RequestDepthShareV1).
bool DepthRequested();
// The game's device, as an identity for the layer (not held).
void SetGameDevice(const void* device);

// The stale-eye OpenXR layer's controls (xr_observer.dll, when loaded in this process). Each returns false while the
// layer is absent or has not submitted a frame yet.
bool LayerSettings(XrObserverStaleSettingsV1* out);
bool SetLayerSettings(const XrObserverStaleSettingsV1& in);
bool LayerStatus(XrObserverStaleStatusV1* out);
// The headset's views (per-eye FOV, pose) and eye gaze from the layer (XrObserver_GetViewsV1): false while
// the layer is absent, older than that export, or has located no views yet.
bool LayerViews(XrObserverViewsV1* out);
// Called every Present and by the overlay: once the layer is there, pushes the ini's settings to it (once per
// process).
void SyncLayer();
// Whether the overlay may show the layer's controls and copy `layerNow` (its state) into the Config: only once the
// ini's settings have reached the layer (StaleEyeSettingsSync.h). Before that its state is its start-up defaults.
bool LayerSettingsReady(const XrObserverStaleSettingsV1& layerNow);
// Once, at start-up, before anything creates an OpenXR instance: when the layer lies in the game's own folder
// (<the OptiScaler folder>\xr_observer: the DLL and its manifest), the OpenXR loader of THIS process is pointed at it
// through its environment -- no registry entry, no administrator, no other application sees it. A copy registered
// machine-wide is switched off for this process (its manifest's disable variable), so one layer runs, not two.
// [Inputs] StaleEyeLayerInFolder = false leaves everything to the registry.
void OfferLayerFromFolder();
} // namespace StaleEyeShare
