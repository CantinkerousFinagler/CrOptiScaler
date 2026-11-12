#pragma once

// In-process ABI between this DLL (the game's winmm.dll) and the stale-eye OpenXR layer (tools/xr_observer), both
// loaded in the game's process. Standalone on purpose: the layer includes this same file.
//
// Under alternate-eye VR each Present renders one eye; the other eye is submitted to OpenXR with its old image and
// the head pose it was rendered with, so the runtime reprojects it for head rotation only. The game camera also
// turns with the stick and moves with the player, which no head pose contains. Each Present, a game backend
// publishes the game's own clip-space map from THIS Present's view to the PREVIOUS Present's (the other eye's
// render), exactly as its motion vector pass uses it. The layer turns that into the stale eye's orientation in the
// current frame and submits the stale eye with it, so the runtime's own reprojection includes the game's rotation.
#include <cstdint>

// "RVTSEYE1"
constexpr std::uint64_t kStaleEyeCameraMagicV1 = 0x3145594553545652ull;

// Depth slots (slot = sequence % kStaleEyeDepthSlots). The submitted eye image lags the game's Present (FC5: its stale
// image pairs with the depth of Present latest - 3; another game: image lag 4 and 6 for long
// stretches, which six slots refused -- the no-fence gate takes latest copy 1 to
// kStaleEyeDepthSlots - 2 past the stale image, i.e. image lag up to kStaleEyeDepthSlots - 3). Eight (was twelve;
// eight saves ~235 MB of video memory at a 3920x3920 eye): since the pairing by identity was fixed
// the lag has been 2 in every window of every run; eight takes up to 5. A lag beyond it shows the
// plain stale image for those frames, with its reason counted. Changing this changes structSize: winmm and the layer
// must be built and deployed together.
constexpr int kStaleEyeDepthSlots = 8;

struct RvrToolsStaleEyeCameraV1
{
    std::uint32_t structSize;    // caller sets sizeof(RvrToolsStaleEyeCameraV1)
    std::uint32_t version;       // 1
    std::uint64_t magic;         // kStaleEyeCameraMagicV1
    std::uint64_t sequence;      // +1 per publication (one per Present); 0 = nothing published yet
    std::int64_t qpc;            // QueryPerformanceCounter at publication
    // Row-major 4x4, column-vector convention: clipPrevious = currentToPrevious * (ndcX, ndcY, depth, 1). D3D NDC
    // (x right, y up), camera looking along +z. Depth is reversed (0 = far) when reversedDepth is set.
    float currentToPrevious[16];
    std::uint32_t width;         // the render size the matrix belongs to
    std::uint32_t height;
    std::uint32_t reversedDepth; // 1: depth 0 is the far plane
    // The game's own projections, as tangents of the frustum edges (left, right, up, down; left and down negative) in
    // a camera with x right, y up, looking forward. The VR mod may render the game with a projection other than the
    // FOV it submits (FC5: the render's FOV is not the headset's), so
    // the layer must use these, not the submitted FOVs, when it reads the matrix. fresh = this Present's, stale = the
    // previous Present's. Valid when tangentsValid is 1.
    std::uint32_t tangentsValid;
    float freshTangents[4];
    float staleTangents[4];
    // The game's graphics device (ID3D11Device* in a DX11 game) as an IDENTITY only: never dereferenced, never
    // AddRef'd. The layer compares it with the device the VR mod gives OpenXR, to know whether it can read the
    // game's textures directly. 0 = unknown.
    std::uint64_t gameDevice;

    // ---- Depth share (DX11 games; the warp) ----
    // Each Present's scene depth, copied on the game's device into one of kStaleEyeDepthSlots shared R32_FLOAT
    // textures (slot = sequence % kStaleEyeDepthSlots), in the GAME's projection at the matrix's width x height, reversed as reversedDepth says. The
    // handles are NT handles valid in this process (the layer opens them on its own device); they change when the
    // size does, and depthGeneration counts that. depthSequence is the sequence whose depth is in slot
    // depthSequence % kStaleEyeDepthSlots (the latest copied; equal to `sequence` when this Present's copy was made).
    // Sync, GPU-side only: the game's device signals writtenFence with a sequence once that Present's copy is queued;
    // the layer waits on it before reading. The layer signals readFence with a sequence once its reads of that
    // sequence's slot are queued, and REPORTS that through RvrTools_ReportDepthReadV1; the game waits on readFence
    // before reusing a slot, but only for values the layer has reported queued, so a stopped layer never hangs it.
    std::uint64_t depthGeneration;   // 0 = no depth shared
    std::uint64_t depthHandles[kStaleEyeDepthSlots];
    std::uint32_t depthWidth;
    std::uint32_t depthHeight;
    std::uint64_t writtenFenceHandle;
    std::uint64_t readFenceHandle;
    std::uint64_t depthSequence;
    // Full projections (row-major, column-vector, as the game uses them) for the metric warp: fresh = this Present's
    // eye, stale = the previous Present's. Valid when tangentsValid is 1.
    float freshProjection[16];
    float staleProjection[16];
    // ---- A game device without fences or NT handles (no ID3D11Device5, NT handles refused) ----
    // depthLegacyHandles 1: depthHandles are the OLD kind (IDXGIResource::GetSharedHandle), opened with
    // ID3D11Device::OpenSharedResource and never closed by anyone (they end with their textures).
    // writtenFenceHandle 0 with depth shared = NO FENCE: the game never waits and the layer never waits or signals;
    // the layer reads a slot only once depthSequence is PAST the sequence it wants (a later Present has copied, so
    // the Present of the wanted copy has flushed it), and the slots keep a slot from being written again for
    // several Presents after that.
    std::uint32_t depthLegacyHandles;
    // depthMarks 1: the game marks the pixels of the PLAYER'S KIND by storing their depth NEGATED (one game: its
    // characters and their weapons, stencil class 1 -- the game does not tell the player from an enemy). The layer
    // reads every depth by its magnitude whatever this says; with marks it takes for "the player" only a marked pixel
    // nearer than nearThreshold, so the ground at the player's feet stays world. 0: nearer is the player (distance).
    std::uint32_t depthMarks;
};

// Exported by winmm.dll. Fills *out with the latest publication; returns 1 on success, 0 if nothing has been
// published or the struct size does not match.
using PFN_RvrTools_GetStaleEyeCameraV1 = int(__cdecl*)(RvrToolsStaleEyeCameraV1* out);

// Exported by winmm.dll. The publication with this sequence, from the game's own history of the last
// kStaleEyeCameraHistory (with the depth share's state as it is now); 0 if it is not there. The layer's frames and
// the game's Presents run on different clocks, so "the latest, once per frame" misses publications and repeats others.
constexpr std::uint64_t kStaleEyeCameraHistory = 64;
using PFN_RvrTools_GetStaleEyeCameraAtV1 = int(__cdecl*)(std::uint64_t sequence, RvrToolsStaleEyeCameraV1* out);

// Exported by winmm.dll. The layer reports the highest sequence whose depth-slot reads it has QUEUED together with a
// readFence signal of that value; the game only ever waits on readFence for values reported here.
using PFN_RvrTools_ReportDepthReadV1 = void(__cdecl*)(std::uint64_t sequence);

// ---- The layer's controls: exported by the layer (xr_observer.dll), used by the game DLL's overlay ----
// The layer owns the state (its hotkeys change it too); the overlay reads it to show what is active, sets it, and
// the game DLL pushes the saved ini values once when the layer first appears. A Set is applied on the layer's own
// thread at its next xrEndFrame.
struct XrObserverStaleSettingsV1
{
    std::uint32_t structSize;    // caller sets sizeof(XrObserverStaleSettingsV1)
    std::int32_t mode;           // 0 off, 1 pass (copy), 2 depth (edges), 3 warp
    std::int32_t pairingOffset;  // Presents the application's eye image lags the camera publication (0, 2, 4), or
                                 // -1 AUTO: chosen per frame by how well each fits the submitted head poses
    float nearThreshold;         // game units: nearer is attached to the camera (hands, weapon) and left unmoved; 0 off
    std::int32_t bothEyes;       // 1: the fresh eye is re-submitted from the layer's swapchain too
    // Head movement since the stale image was rendered: 0 the headset's (rotation only; near surfaces judder under
    // head TRANSLATION), 1 warped to where the stale eye was when the fresh eye was rendered (both eyes the same
    // instant; default), submitted with that pose. (A "current pose" 2 was removed: the located
    // views and the submitted poses did not agree, so only their DIFFERENCE is usable -- 2 drew an extra
    // upside-down image.) Values above 1 are treated as 1.
    std::int32_t headMode;
    // Stale-eye TREATMENT on top of Pass / Warp (the parked "hide it" idea: the black frames of an old bug showed no
    // strafing judder): 0 none, 1 dim (strength 1 = black), 2 blur (strength 1 = 24 px radius). Gated: scaled by
    // the game's own motion per Present, full at treatFullAt game units (turning: 1 deg counts as 0.03 units).
    std::int32_t treatMode;
    float treatStrength;
    std::int32_t treatGated;
    float treatFullAt;
    // Simpler strafing: every hard WARP problem -- holes, fill, near objects, the weapon -- comes from the
    // TRANSLATION part, which needs per-pixel depth. These trade accuracy for stability, to compare in one launch:
    // strafeOnly 1 leaves the game's own ROTATION (stick turns) out of the warp and corrects only its translation.
    // depthMode 0 = per pixel (the game's depth), 1 = SINGLE DISTANCE: the whole image moved as if everything were
    // singleMetres away (one plane: no depth buffer read, no holes, no tearing; exact only at that distance),
    // 2 = PAST THE PLAYER: the player (nearer than nearThreshold, distance only) left unmoved, the world moved as one
    // plane at an AUTOMATIC distance -- the median depth of the world in the middle of the view, each frame.
    // 3 = SMOOTH: the player left unmoved, the world moved by its own depth on a COARSE, blurred grid (a cell ~74 px):
    // no holes and no tearing; beside near things the picture stretches instead.
    // clampMetres (per pixel only): nothing is corrected as nearer than this (0 = off): far scenery exact, near
    // objects under-corrected instead of tearing, the weapon moves little instead of doubling.
    std::int32_t strafeOnly;
    std::int32_t depthMode;
    float clampMetres;
    float singleMetres;
    // 1: "attached" (left unmoved) = nearer than nearThreshold, by DISTANCE ONLY -- everything closer is left alone,
    // everything farther warped per pixel. 0: near AND unchanged depth + colour since the eye's last frame (which
    // never recognised FC5's real weapon: 0% attached). Distance only freezes walls/NPCs within reach.
    std::int32_t attachDistanceOnly;
    // 1: WARP's diagnostic view -- what each pixel fell back on, tinted: holes red, fill blue, attached / the player
    // green. Never saved in the ini (a view that replaces the picture must not survive a restart).
    std::int32_t showBreaks;
    // 1: the layer logs timing / pairing / warp EVENTS as they happen and a summary every 10 s. 0 (everyday): no
    // event lines and no summaries: nothing is written during play, and the layer's hotkeys (Home, End, Page Up,
    // Page Down, Delete = MARK, Scroll Lock) do nothing. The counters run either way.
    std::int32_t diagnostics;
    // What is nearer than nearThreshold moves with: 0 the HEAD (left exactly where it was in the picture: right for
    // something the view carries along), 1 the BODY (moved by the head's own movement since the stale picture, with
    // its own depth, and by nothing of the game's: a vehicle's cabin, which the game camera carries along while the
    // head turns inside it). FC5: a cabin left alone doubled at every head turn with headMode 1, and
    // warped as world it doubled by the vehicle's speed. The same as 0 with headMode 0 (no head movement is warped
    // then). SMOOTH and PAST THE PLAYER only; per pixel leaves the near things alone whatever this says.
    std::int32_t nearWithBody;
    // BEHIND THE PLAYER (SMOOTH and PAST THE PLAYER): what the stale eye never saw because the player stood in front
    // of it -- a stick turn uncovers it beside the player. 1: taken from the OTHER eye's picture of this frame (same
    // viewing direction, no depth: exact for distant background), unless that eye's depth shows the player there;
    // what is left is BLURRED from the world beside it. 0: stretched from the nearest world pixel (as before).
    std::int32_t fillFromOtherEye;
};
struct XrObserverStaleStatusV1
{
    std::uint32_t structSize;    // caller sets sizeof(XrObserverStaleStatusV1)
    std::uint32_t session;       // 1 while a D3D11 OpenXR session is running
    float worldScale;            // game units per metre; 0 = not measured yet
    // The last second:
    float processedPerSecond;    // stale images processed
    float warpedPerSecond;       // of them warped
    float holePercent;           // pixels whose start depth was a hole after the fill
    float fillPercent;           // pixels shown with the fill colour (surface hidden in the stale image)
    float attachedPercent;       // pixels attached to the camera (left unmoved)
    float gpuMs;                 // mean GPU time of a warp
    float gameMove;              // mean game-only motion per warped Present (game units)
    float treatApplied;          // mean treatment strength applied (0..1)
    float planeMetres;           // PAST THE PLAYER: the automatic plane distance now (0 = none measured yet)
    // Since the session started:
    std::uint32_t unwarpedTotal; // WARP selected, the plain stale image shown (any reason): each one is a visible break
    std::uint32_t wrongEyeTotal; // of them: the paired camera was the other eye's (see the layer's PAIRING check)
    char lastSkip[112];          // why the last stale image was not processed as selected, within the last second
    // COST. gpuMs above is a LOWER BOUND (D3D11 timestamps do not wait for the bracketed work).
    float videoMemoryMb;         // the whole PROCESS's video memory now (game, VR mod, DLSS, this layer); 0 = unknown
    float videoBudgetMb;         // what the system allows the process; above it, textures are paged out
    float ownMemoryMb;           // this layer's textures, by their sizes (its swapchains included)
};
using PFN_XrObserver_GetStaleSettingsV1 = int(__cdecl*)(XrObserverStaleSettingsV1* out);
using PFN_XrObserver_SetStaleSettingsV1 = int(__cdecl*)(const XrObserverStaleSettingsV1* in);
using PFN_XrObserver_GetStaleStatusV1 = int(__cdecl*)(XrObserverStaleStatusV1* out);

// Exported by winmm.dll. The game shares depth ONLY while the layer requests it (1) -- i.e. while an image mode that
// needs depth is selected -- and only once the render size has settled. With the layer idle the game does no extra
// GPU work at all. (The first build shared from the first frame, at the flat 1920x1080 frames of the switch into VR,
// and the GPU hung: the FC5 hazard of work at a new size.)
using PFN_RvrTools_RequestDepthShareV1 = void(__cdecl*)(int on);

// ---- The overlay in the headset ----
// The game DLL's overlay (ImGui) is drawn on the game's flat window, which no headset shows. While the layer asks --
// it calls RvrTools_GetOverlayV1 at every xrEndFrame, and the game DLL counts a layer that has not called for a
// second as gone -- and the overlay has anything to draw, the game DLL draws it a second time, into one of
// kOverlaySlots shared textures of its own, and the layer submits the newest as a flat panel (a quad layer) in front
// of the eyes. With the overlay closed, or no layer, nothing is drawn and nothing is copied.
constexpr int kOverlaySlots = 3;
struct RvrToolsOverlayV1
{
    std::uint32_t structSize;    // caller sets sizeof(RvrToolsOverlayV1)
    std::uint32_t visible;       // 1: the overlay drew something at the game's last Present
    std::uint64_t generation;    // 0 = nothing shared yet; +1 with every new set of textures (a new size)
    // NT handles valid in this process: R8G8B8A8_TYPELESS, width x height, one level. The colour is premultiplied by
    // its alpha (ImGui blended onto transparent black) and its values are sRGB-encoded, as on the flat window.
    std::uint64_t handles[kOverlaySlots];
    std::uint32_t width;
    std::uint32_t height;
    // The part of the image that is the panel, from its top left corner: the overlay's own rectangle in the game's
    // window (FC5's window in VR is 8192 x 8192, the menu a small part of it), which grows while the overlay is up
    // and starts anew when it has been closed. The textures are larger (their sizes change rarely).
    std::uint32_t usedWidth;
    std::uint32_t usedHeight;
    // The game's device signals this fence with a sequence once that image's draw is queued; the layer waits on it
    // (GPU-side) before it copies slot sequence % kOverlaySlots.
    std::uint64_t writtenFenceHandle;
    std::uint64_t sequence;      // the newest image
    std::int64_t qpc;            // QueryPerformanceCounter when it was drawn (the layer hides a panel that is old)
    // Where the panel is: its width in metres (of the used part; the height follows its shape -- the game DLL gives
    // every pixel of the overlay the same size, so a wider menu is a wider panel), how far in front of the eyes,
    // and how far its centre is below eye level.
    float panelWidth;
    float panelDistance;
    float panelDown;
    // A game's device may be a wrapper that knows less than Direct3D 11.4 (one game's has no ID3D11Device5):
    // writtenFenceHandle 0 = no fence; the game DLL then names each image one Present after it was drawn, and the
    // layer copies without waiting. legacyHandles 1 = the handles are the old kind (IDXGIResource::GetSharedHandle,
    // opened with ID3D11Device::OpenSharedResource), for a device that cannot make NT handles.
    std::uint32_t legacyHandles;
};
// Exported by winmm.dll. Fills *out; returns 1, or 0 if the struct size does not match.
using PFN_RvrTools_GetOverlayV1 = int(__cdecl*)(RvrToolsOverlayV1* out);

// ---- The headset's views and gaze, LAYER -> GAME (tools/xr_observer/gaze_share) ----
// The first thing the layer hands the game DLL. The foveal crop places the gaze and its calibration targets through
// the projection of the eye being rendered; DLSS is handed a projection (ViewToClipMatrix) by ray reconstruction only,
// so a game with DLSS SR alone had none. Here: the per-view FOV and pose of the application's latest xrLocateViews
// (the HEADSET's projection -- the VR mod may render the game with another one; the in-game calibration absorbs an
// affine difference), and the eyes read through XR_EXT_eye_gaze_interaction (any runtime that offers it), located in
// the same space and at the same time as the views and given in EACH VIEW'S OWN FRAME.
// View 0 is the left eye, 1 the right (OpenXR's stereo order); which Present is which eye the game DLL knows from its
// own fits (the parallax sign per parity).
struct XrObserverViewsV1
{
    std::uint32_t structSize;      // caller sets sizeof(XrObserverViewsV1)
    std::uint32_t viewCount;       // 0 = no xrLocateViews seen yet; up to 2 are kept
    std::uint64_t locates;         // +1 per xrLocateViews with views: a reader sees whether anything is new
    std::int64_t displayTime;      // XrTime of the locate
    std::uint32_t session;         // 1 while the application's session exists
    // The extension's state: 0 = not in the instance (the runtime lacks it, or XR_OBSERVER_GAZE=0), 1 = in the
    // instance, the action set not attached yet, 2 = attached (located every frame), 3 = a tracked gaze has been seen.
    std::uint32_t gazeState;
    // Per view: the frustum as tangents (left, right, up, down; left and down NEGATIVE), the pose in the application's
    // space (orientation x y z w, position x y z).
    float fov[2][4];
    float orientation[2][4];
    float position[2][3];
    // The gaze of THIS locate: valid 1 = the runtime located it (orientation valid); tracked 1 = and tracked (not
    // predicted from an old sample). Direction per view in that view's frame, UNIT, x right, y up, z FORWARD taken
    // positive (the game DLL's convention; OpenXR's view space looks along -z).
    std::uint32_t gazeValid;
    std::uint32_t gazeTracked;
    float gazeDirection[2][3];
    std::int64_t gazeSampleTime;   // XrEyeGazeSampleTimeEXT's time (0 = not given)
    std::int64_t gazeAgeNs;        // displayTime - sample time (0 when unknown): how old the sample is at display
    // Since the session: locates with a valid gaze, with a tracked gaze, without.
    std::uint64_t gazeValidTotal;
    std::uint64_t gazeTrackedTotal;
    std::uint64_t gazeInvalidTotal;
    char text[128];                // the state in words (for the overlay and the log)
    // What the application SUBMITTED per view at its last xrEndFrame with a projection layer: the FOV it says the image
    // was rendered with (the compositor's truth about the image), the image rectangle and the swapchain's size. submits
    // 0 until one has been seen. (In one D3D12 game the located FOV is asymmetric, -1.58..+0.71 for the left
    // eye; the VR mod may render with a projection other than the one it submits. A projection
    // with the headset's offsets put a straight gaze 38% off centre, toward the nose, in each eye.)
    std::uint64_t submits;
    float submittedFov[2][4];
    std::int32_t imageRect[2][4];      // x, y, width, height
    std::uint32_t swapchainSize[2][2]; // width, height
};
// Exported by the layer (xr_observer.dll). Fills *out; returns 1, or 0 if the struct size does not match or nothing has
// been located yet.
using PFN_XrObserver_GetViewsV1 = int(__cdecl*)(XrObserverViewsV1* out);
