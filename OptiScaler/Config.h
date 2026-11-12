#pragma once

#include "SysUtils.h"
#include "State.h"

#include <optional>
#include <filesystem>

enum HasDefaultValue
{
    WithDefault,
    NoDefault,
    SoftDefault // Change always gets saved to the config
};

template <class T, HasDefaultValue defaultState = WithDefault> class CustomOptional : public std::optional<T>
{
  private:
    T _defaultValue;
    std::optional<T> _configIni;
    bool _volatile;

  public:
    CustomOptional(T defaultValue)
        requires(defaultState != NoDefault)
        : std::optional<T>(), _defaultValue(std::move(defaultValue)), _configIni(std::nullopt), _volatile(false)
    {
    }

    CustomOptional()
        requires(defaultState == NoDefault)
        : std::optional<T>(), _defaultValue(T {}), _configIni(std::nullopt), _volatile(false)
    {
    }

    // Prevents a change from being saved to ini
    constexpr void set_volatile_value(const T& value)
    {
        if (!_volatile)
        { // make sure the previously set value is saved
            if (this->has_value())
                _configIni = this->value();
            else
                _configIni = std::nullopt;
        }
        _volatile = true;
        std::optional<T>::operator=(value);
    }

    // Use this when first setting a CustomOptional
    constexpr void set_from_config(const std::optional<T>& opt)
    {
        if (!this->has_value())
        {
            _configIni = opt;
            std::optional<T>::operator=(opt);
        }
    }

    constexpr CustomOptional& operator=(const T& value)
    {
        _volatile = false;
        std::optional<T>::operator=(value);
        return *this;
    }

    constexpr CustomOptional& operator=(T&& value)
    {
        _volatile = false;
        std::optional<T>::operator=(std::move(value));
        return *this;
    }

    constexpr CustomOptional& operator=(const std::optional<T>& opt)
    {
        _volatile = false;
        std::optional<T>::operator=(opt);
        return *this;
    }

    constexpr CustomOptional& operator=(std::optional<T>&& opt)
    {
        _volatile = false;
        std::optional<T>::operator=(std::move(opt));
        return *this;
    }

    // Needed for string literals for some reason
    constexpr CustomOptional& operator=(const char* value)
        requires std::same_as<T, std::string>
    {
        _volatile = false;
        std::optional<T>::operator=(T(value));
        return *this;
    }

    constexpr T value_or_default() const&
        requires(defaultState != NoDefault)
    {
        return this->has_value() ? this->value() : _defaultValue;
    }

    constexpr T value_or_default() &&
        requires(defaultState != NoDefault) {
            return this->has_value() ? std::move(this->value()) : std::move(_defaultValue);
        }

        constexpr std::optional<T> value_for_config()
            requires(defaultState == WithDefault)
    {
        if (_volatile)
        {
            if (_configIni != _defaultValue)
                return _configIni;

            return std::nullopt;
        }

        if (!this->has_value() || *this == _defaultValue)
            return std::nullopt;

        return this->value();
    }

    constexpr std::optional<T> value_for_config()
        requires(defaultState != WithDefault)
    {
        if (_volatile)
            return _configIni;

        if (this->has_value())
            return this->value();

        return std::nullopt;
    }

    constexpr T value_for_config_or(T other)
    {
        auto option = value_for_config();

        if (option.has_value())
            return option.value();
        else
            return other;
    }
};

constexpr inline int UnboundKey = -1;
constexpr uint32_t NV_PRESET_LATEST = 0x00FFFFFF;

enum FpsOverlayPos : uint32_t
{
    FpsOverlayPos_TopLeft,
    FpsOverlayPos_TopRight,
    FpsOverlayPos_BottomLeft,
    FpsOverlayPos_BottomRight,
    FpsOverlayPos_COUNT,
};

enum FpsOverlay : uint32_t
{
    FpsOverlay_JustFPS,
    FpsOverlay_Simple,
    FpsOverlay_Detailed,
    FpsOverlay_DetailedGraph,
    FpsOverlay_Full,
    FpsOverlay_FullGraph,
    FpsOverlay_ReflexTimings,
    FpsOverlay_COUNT,
};

// Output scaling downscaler
enum class Scaler : uint32_t
{
    FSR1 = 0,
    Bicubic = 1,
    CatmullRom = 2,
    Lanczos2 = 3,
    Lanczos3 = 4,
    Kaiser2 = 5,
    Kaiser3 = 6,
    Magic = 7,
    Count
};

enum class ForceReflex : uint32_t
{
    InGame,
    ForceDisable,
    ForceEnable,
    Count
};

enum class LFXMode : uint32_t
{
    Conservative,
    Aggressive,
    ReflexIDs,
    Count
};

enum class LowLatencyInput : uint32_t
{
    None,
    Auto,
    AntiLag2,
    Reflex,
    XeLL,
    UeLowLatency,
    _
};

enum class LowLatencyMode : uint32_t
{
    None,
    Auto,
    LatencyFlex,
    AntiLag2,
    XeLL,
    AntiLagVk,
    Reflex
};

class Config
{
  public:
    Config();

    // Init flags
    CustomOptional<bool, NoDefault> DepthInverted;
    CustomOptional<bool, NoDefault> AutoExposure;
    CustomOptional<bool, NoDefault> HDR;
    CustomOptional<bool, NoDefault> JitterCancellation;
    CustomOptional<bool, NoDefault> DisplayResolution;
    CustomOptional<bool, NoDefault> DisableReactiveMask;
    CustomOptional<float> DlssReactiveMaskBias { 0.45f };

    // Logging
    CustomOptional<bool> LogToFile { false };
    CustomOptional<bool> LogToConsole { false };
    CustomOptional<bool> LogToDebug { false };
    CustomOptional<bool> LogToNGX { false };
    CustomOptional<bool> OpenConsole { false };
    CustomOptional<bool> DebugWait { false }; // not in ini
    CustomOptional<int> LogLevel { 0 };
    CustomOptional<std::wstring> LogFileName { L"OptiScaler.log" };
    CustomOptional<bool> LogSingleFile { true };
    CustomOptional<bool> LogAsync { false };
    CustomOptional<int> LogAsyncThreads { 4 };

    // XeSS
    CustomOptional<bool> BuildPipelines { true };
    CustomOptional<int32_t> NetworkModel { 0 };
    CustomOptional<bool> CreateHeaps { true };

    // --- DLSS 5 Neural Rendering (OptiScaler/dlssnr) --- removable as one block -----------------
    // DLSS Neural Rendering: a detail-synthesis pass over the upscaler's output. Off by default -- it is
    // not a feature NVIDIA documents.
    CustomOptional<bool> DlssNrEnabled { false };
    // Run the NR pass on the upscaler's colour input, at render resolution, immediately before SR.
    // Off preserves the v0.2.0 post-upscale placement.
    CustomOptional<bool> DlssNrRunBeforeSr { false };
    CustomOptional<bool> DlssNrFinishedPicture { false };
    // Generate NR before SR, upscale its signed contribution with a private DLSS feature,
    // and apply it after the game's upscaler. Takes precedence over RunBeforeSR; opt-in.
    CustomOptional<bool> DlssNrDeferredDlss { false };
    // Experimental: with RunBeforeSR and the game's Ray Reconstruction both on, run NR before SR
    // but leave the colour input untouched, then add the model's edit back onto the RR+SR output
    // so it survives RR's denoise. v2 carries the edit as an MV-reprojected temporal accumulator
    // (the per-frame ray-trace noise term averages to zero; the enhancement persists). Inert
    // unless RunBeforeSR + RR are both active. Opt-in.
    CustomOptional<bool> DlssNrResidualAcrossRr { false };
    // v2 history blend rate for the accumulator above, 0.01..1. Lower = stabler but slower to
    // appear; 1.0 = no accumulation (each frame's raw residual, which flickers). Default 0.08.
    CustomOptional<float> DlssNrResidualAcrossRrBlend { 0.08f };
    CustomOptional<bool> DlssNrResidualFg { false };
    CustomOptional<bool> DlssNrResidualFgApproxCamera { false };
    // Toggles the pass in game. Unbound by default -- a key that does something unexpected is worse
    // than one that does nothing.
    CustomOptional<int> DlssNrToggleKey { UnboundKey };
    CustomOptional<uint32_t> DlssNrPreset { 0 };
    CustomOptional<float> DlssNrIntensity { 1.0f };
    // 0 default (standard), 1 natural, 2 cinematic -- the model's own processing profiles.
    CustomOptional<uint32_t> DlssNrStyle { 0 };
    // Optional per-pass model profiles. Pass 1 uses Preset/Style above; an absent override inherits
    // pass 1. Keeping inheritance explicit preserves every existing configuration and lets changing
    // the base profile update the whole stack unless a later pass was deliberately specialised.
    CustomOptional<uint32_t, NoDefault> DlssNrPass2Preset;
    CustomOptional<uint32_t, NoDefault> DlssNrPass2Style;
    CustomOptional<uint32_t, NoDefault> DlssNrPass3Preset;
    CustomOptional<uint32_t, NoDefault> DlssNrPass3Style;
    CustomOptional<float> DlssNrLocalStructure { 1.0f };
    CustomOptional<float> DlssNrLocalTone { 1.0f };
    // -1 means follow local structure, which is the model's own default. It is not a strength of zero.
    CustomOptional<float> DlssNrSkinStructure { -1.0f };
    CustomOptional<bool> DlssNrAutoMask { true };
    // Optional final-composition filter, not NVIDIA's semantic auto mask.
    CustomOptional<bool> DlssNrSkinProtection { false };
    CustomOptional<bool> DlssNrSkinToneEnabled { true };
    CustomOptional<float> DlssNrSkinDetail { 1.0f };
    CustomOptional<float> DlssNrSkinColour { 1.0f };
    CustomOptional<float> DlssNrEnvironmentDetail { 1.0f };
    CustomOptional<float> DlssNrEnvironmentColour { 1.0f };
    CustomOptional<bool> DlssNrShowSkinMask { false };
    CustomOptional<float, NoDefault> DlssNrPass2Intensity;
    CustomOptional<float, NoDefault> DlssNrPass2LocalStructure;
    CustomOptional<float, NoDefault> DlssNrPass2LocalTone;
    CustomOptional<float, NoDefault> DlssNrPass2SkinStructure;
    CustomOptional<bool, NoDefault> DlssNrPass2AutoMask;
    CustomOptional<float, NoDefault> DlssNrPass3Intensity;
    CustomOptional<float, NoDefault> DlssNrPass3LocalStructure;
    CustomOptional<float, NoDefault> DlssNrPass3LocalTone;
    CustomOptional<float, NoDefault> DlssNrPass3SkinStructure;
    CustomOptional<bool, NoDefault> DlssNrPass3AutoMask;
    CustomOptional<bool> DlssNrUnlockPasses { false };
    struct NrExtraPass
    {
        CustomOptional<uint32_t, NoDefault> style;
        CustomOptional<float, NoDefault> intensity, structure, tone, skin;
        CustomOptional<bool, NoDefault> autoMask;
    };
    NrExtraPass DlssNrExtraPasses[27]; // pass 4..30; legacy pass 2/3 keys stay compatible

    // How much of the model's edit reaches the frame. Separated because detail synthesis is a luminance
    // edit and any colour shift is usually the part you do not want, and allowed past 1.0 because
    // exaggerating an edit is the only honest way to see whether there is one.
    CustomOptional<float> DlssNrTransferStrength { 1.0f };
    CustomOptional<float> DlssNrColourStrength { 1.0f };

    // The RenoDX reversible proxy mode. 0 = today's soft-knee encode + our composition (default,
    // byte-identical); 1 = unclipped Neutwo proxy + our composition; 2 = Neutwo proxy + pure-inverse
    // replace. An in-game A/B and a way back. Default 0 = byte-identical to before.
    CustomOptional<uint32_t> DlssNrReversibleMode { 0 };

    // Whether the model's edit is applied. Off keeps the pass running (so Hold frame works) but shows
    // the clean upscaler frame -- for A/B'ing NR on/off on a frozen frame. Default true.
    CustomOptional<bool> DlssNrApplyModel { true };

    // Frame hold: freeze the NR pass's input so a live setting change re-renders the SAME frame -- the
    // only clean way to A/B our settings. A live testing toggle, not really a saved preference; off by
    // default. See dlssnr/design/frame-hold.md.
    CustomOptional<bool> DlssNrHoldFrame { false };


    // The most the pass may multiply or divide a pixel by. A detail pass has no business restyling a
    // light source, whatever the model returns.
    CustomOptional<float> DlssNrMaxRatio { 2.0f };

    // How a model that worked below the frame's size is brought back. 0 classic, 1 matched
    // residual. Only has an effect when Model resolution is under 100%.
    CustomOptional<uint32_t> DlssNrTransfer { 1 };

    // Measure the white point from the frame instead of taking it from the slider. On a frame the
    // game already tone mapped there is nothing to measure and this has no effect.
    //
    // Off by default, because it is not finished. The pass writes its result back into the same buffer
    // the meter reads, so with the pass running the meter is partly measuring its own output and the
    // two chase each other: Enshrouded, one session, 1545 samples spanning 0.01 to 97.9 with 57 jumps
    // beyond 1.5x in a single frame. Measured in the same spot seconds apart, 41.31 with the pass off
    // against 0.46 with it on. That is visible as the picture pumping and occasionally flickering.
    //
    // The slider is the supported control until the loop is broken. This stays as an opt-in so the
    // behaviour can still be looked at.


    // Take the white point from the game's own exposure texture instead of measuring or guessing.
    // Off by default until it has been seen to work in more than one game.
    CustomOptional<bool> DlssNrWhitePointFromExposure { true };

    // Ask the model, once, whether it will run on Direct3D 11 without the bridge.
    //
    // Off by default and deliberately so. Everything else this pass does reads memory it already owns;
    // this one initialises an NVIDIA subsystem on the game's live D3D11 device, in a process where the
    // D3D12 NGX instance is already running. It should return an error code and nothing more, but
    // "should" is doing work in that sentence and it ships into games nobody can test first.
    CustomOptional<bool> DlssNrProbeD3D11 { false };

    // 0 off, 1 the picture the model was shown, 2 its raw answer, 3 what it changed, amplified.
    CustomOptional<uint32_t> DlssNrDebugView { 0 };

    // Showing the pass against itself, without having to toggle it and remember what the last frame
    // looked like. 0 off, 1 side by side, 2 a wipe.
    //
    // Side by side squeezes the whole frame into each half, so it is a comparison rather than
    // something to play in. The wipe cuts one frame and resamples nothing, so it is; the split is a
    // stored setting and stays where it was put once the menu closes.
    CustomOptional<uint32_t> DlssNrCompare { 0 };
    CustomOptional<float> DlssNrCompareSplit { 0.5f };

    // Side by side only. 1 fits the whole frame at its right shape and accepts the bars; 2 fills
    // the half and crops the sides off instead.
    CustomOptional<float> DlssNrCompareZoom { 1.0f };

    // Which side the edited frame sits on, in both comparison modes.
    CustomOptional<bool> DlssNrCompareSwap { false };

    // Labels drawn onto the two sides of a comparison, so a screenshot still says which is which.
    // Drawn into the frame's own plane with a clip per side: in the wipe they are revealed and hidden
    // by the split exactly as the images are, and there is nothing to drag.
    CustomOptional<bool> DlssNrCompareTags { false };
    CustomOptional<float> DlssNrTagScale { 1.5f };

    // The fraction of the frame's resolution the model works at. The frame itself is never reduced --
    // only the model's contribution is computed small and enlarged, so the picture underneath is
    // untouched whatever this is set to. 1.0 is full resolution and behaves exactly as before.
    CustomOptional<float> DlssNrWorkingScale { 1.0f };
    // On by default: with the foveal crop on, NR works on the crop's box alone.
    CustomOptional<bool> DlssNrFoveated { true };
    CustomOptional<float> DlssNrFoveaWidth { 0.40f };
    CustomOptional<float> DlssNrFoveaHeight { 0.40f };
    CustomOptional<float> DlssNrFoveaRoundness { 0.0f };
    CustomOptional<float> DlssNrFoveaTransition { 0.08f };
    CustomOptional<float> DlssNrFoveaOriginX { 0.5f };
    CustomOptional<float> DlssNrFoveaOriginY { 0.5f };
    CustomOptional<bool> DlssNrFoveaBorder { false };
    // The NR fovea's centre follows the SR crop's glide centre while that crop follows the gaze (PerEyeDlss_Dx12);
    // a move of the region restarts this eye's NR history (the region is a pan the model cannot see).
    CustomOptional<bool> DlssNrFoveaFollowSr { true };

    // Filter used for NR supersampling (working scale > 1): the model runs above native, and this is
    // the downscaler that averages its answer back to native. Independent of OutputScalingDownscaler
    // so NR and Output Scaling can run different filters at once. Lanczos3 is the sharp default.
    CustomOptional<Scaler> DlssNrScalingDownscaler { Scaler::Lanczos3 };

    // Ask the driver's own nvngx.dll whether it will dispatch Neural Rendering, once per session.
    // If it does, the forwarder is unnecessary and users stop needing a 165 MB copy in every game folder.
    //
    // Off by default: it is a diagnostic, not a feature.
    CustomOptional<bool> DlssNrProxyProbe { false };

    // Run Neural Rendering through the driver's own nvngx.dll rather than through the forwarder.
    // This is how DLSS itself is called. OptiScaler already tells the driver where to look, since
    // NVNGX_FeatureInfo_Paths carries the game and OptiScaler folders into Init_Ext.
    //
    // Off until it is shown to produce the same picture. If it does, the forwarder can go.
    CustomOptional<bool> DlssNrUseProxy { false };

    // Look for the exposure the game computed but never handed to the upscaler.
    //
    // Off by default, and it has to be. Reading a resource the game owns means assuming what state
    // it is in, and unlike depth and motion vectors -- where NGX documents the contract -- a buffer
    // found by its shape comes with no promise at all. UNORDERED_ACCESS is the reasonable
    // assumption, since every candidate got here by having a UAV made on it, but it is an
    // assumption, and nobody who has not asked for the scan should be carrying that risk.
    //
    // It decides nothing either way. It watches and it reports, because the last two times a number
    // was inferred here it went straight into the interface and was wrong.
    CustomOptional<bool> DlssNrScanExposure { false };

    // Anchoring the scan: the white point that looked right, and the scan's value at that moment.
    //
    // The absolute white point cannot be derived from a buffer whose units are unknown. What CAN be
    // derived is every value after the first: if the scan's number halves, the scene got twice as
    // bright, and the white point follows -- whatever the number actually means, because only the
    // ratio is used and the units cancel.
    //
    // So it is set once, in one lighting condition, with a button press. After that it stays
    // correct through every cave and every noon without being touched again. Which is also the shape
    // that makes per-game profiles work: a game anchored once gives everybody the number.
    //
    // Zero means not anchored, and then nothing happens at all.
    // A lamp in the corner showing what the scan currently thinks the light is doing: red for dark,
    // green for full light, and the shades between. Off by default; it is for watching the thing
    // work, not for playing with.
    // Where the white point comes from. One control, because there is one answer.
    //
    //   0  the paper white slider, and nothing else
    //   1  the exposure the game hands the upscaler
    //   2  a buffer the scan found, anchored to a white point chosen once
    //
    // This replaces two independent checkboxes that could both be on. They were made exclusive by
    // greying, which deadlocked -- each disabled the other, so once both were set the only way out
    // was a button the notice never mentioned -- and then by clearing, which silently undid a
    // setting that had been made. Both were attempts to stop an illegal state being REACHED. A single
    // choice cannot reach it: there is nothing to keep consistent, because there is only one value.
    CustomOptional<uint32_t> DlssNrWhitePointSource { 1 };

    CustomOptional<bool> DlssNrScanMeter { false };

    CustomOptional<float> DlssNrScanAnchorValue { 0.0f };       // legacy single anchor, migrated then unused
    CustomOptional<float> DlssNrScanAnchorWhitePoint { 0.0f };  // legacy single anchor, migrated then unused

    // The multi-point anchor table, serialised as "scan:white;scan:white;..." ascending. See
    // dlssnr/design/multi-point-anchoring.md. Replaces the single pair above; a pre-existing single
    // anchor is migrated into a one-row table on first load.
    CustomOptional<std::string> DlssNrScanAnchors { std::string() };

    // Whether the scan's number rises or falls with the light.
    //
    // A found buffer carries no contract. Most engines store an exposure -- a multiplier that goes
    // DOWN as the scene gets brighter -- but some store its reciprocal, and nothing in the buffer
    // says which. Rather than guess and be silently wrong in half the games, this is one click: if
    // the picture moves the wrong way, flip it.
    CustomOptional<bool> DlssNrScanInverted { false };







    // The trim on an exposure-derived white point, kept apart from the manual divisor on purpose.
    //
    // These are two different quantities that happened to share one slider: the manual path wants an
    // absolute divisor on an open-ended linear buffer, which in Nioh 3 is about 240, and the exposure
    // path wants a multiplier on a number the game already supplied, where anything far from 1 is
    // a sign the read is wrong rather than a preference. Sharing one stored value meant touching the
    // slider in one mode silently destroyed the number found in the other.
    //
    // 1.0 is the identity: take the game's exposure exactly as given. That is the "safe value", and
    // it is safe by construction rather than by being written down somewhere.
    CustomOptional<float> DlssNrWhitePointTrim { 1.0f };

    // The scan's trim, kept apart from the exposure texture's.
    //
    // They are trims on different things and a value found against one is meaningless against the
    // other. Sharing one slider meant switching source silently carried a number across, so a
    // picture that had been tuned came back wrong for a reason nothing on screen explained.
    CustomOptional<float> DlssNrScanTrim { 1.0f };

    // How many sequential model layers to run between one encode and one final composition. Each extra
    // layer consumes the preceding model output and owns a persistent feature/history. The implementation
    // deliberately caps this at three and never evaluates a feature on the command list that created it.
    //
    // 1 is what the model was trained for and what every published number describes. Above that it
    // is being asked to enhance its own output, which is outside its training distribution: detail
    // compounds, and so does anything it got wrong. Two often looks richer; three is the guarded
    // ceiling because further layers converge while still paying the full cost.
    //
    // The cost is exactly linear -- the model is 98% of the frame's expense and every pass pays it
    // again -- so 3 costs three times, near enough. There is no shortcut and no amortisation: the
    // passes are sequential and each one needs the last one's output.
    CustomOptional<uint32_t> DlssNrPasses { 1 };

    // Which depth convention the model is told the guide uses.
    //
    //   0  what the game's own DLSS feature was created with, which is what it means for the upscaler
    //   1  force normal
    //   2  force inverted
    //
    // Writes one set of matched before/after frames per session, without anyone having to ask. The
    // folder is cleared at the start of each run, so it holds one session's worth and never grows.
    CustomOptional<bool> DlssNrAutoCapture { true };





    // Multiplies the (auto or manual) white point before the encode: what the model considers "white".
    // Higher means highlights sit lower on the curve and the model treats them as less extreme.
    CustomOptional<float> DlssNrWhitePointScale { 1.0f };





    // --- end DLSS 5 Neural Rendering -------------------------------------------------------------

    // DLSS
    CustomOptional<bool> DLSSEnabled { true };
    CustomOptional<bool> RenderPresetOverride { false };
    CustomOptional<uint32_t> RenderPresetForAll { 0 };
    CustomOptional<uint32_t> RenderPresetDLAA { 0 };
    CustomOptional<uint32_t> RenderPresetUltraQuality { 0 };
    CustomOptional<uint32_t> RenderPresetQuality { 0 };
    CustomOptional<uint32_t> RenderPresetBalanced { 0 };
    CustomOptional<uint32_t> RenderPresetPerformance { 0 };
    CustomOptional<uint32_t> RenderPresetUltraPerformance { 0 };

    // DLSSD
    CustomOptional<bool> DLSSDRenderPresetOverride { false };
    CustomOptional<uint32_t> DLSSDRenderPresetForAll { 0 };
    CustomOptional<uint32_t> DLSSDRenderPresetDLAA { 0 };
    CustomOptional<uint32_t> DLSSDRenderPresetUltraQuality { 0 };
    CustomOptional<uint32_t> DLSSDRenderPresetQuality { 0 };
    CustomOptional<uint32_t> DLSSDRenderPresetBalanced { 0 };
    CustomOptional<uint32_t> DLSSDRenderPresetPerformance { 0 };
    CustomOptional<uint32_t> DLSSDRenderPresetUltraPerformance { 0 };

    // Nukems
    CustomOptional<bool> NvngxFGMakeDepthCopy { false };

    // Libraries
    CustomOptional<std::wstring, NoDefault> MainDllPath;
    CustomOptional<std::wstring, NoDefault> FfxDx12Path;
    // Startup only. When false, the FFX / XeSS proxy DLLs are not loaded from disk in DllMain; they are loaded
    // lazily by the first feature that needs them. A game whose exe imports winmm.dll runs our DllMain during
    // process initialisation, and loading the game's own amd_fidelityfx_dx12.dll there initialises its import
    // a wrapper dxgi.dll in the game's folder inside our DllMain, where it crashed (one game: a wait on a critical
    // section inside the wrapper, 2 s after our load).
    CustomOptional<bool> LoadProxiesAtStartup { true };
    CustomOptional<std::wstring, NoDefault> FfxDx12SRPath;
    CustomOptional<std::wstring, NoDefault> FfxDx12FGPath;
    CustomOptional<std::wstring, NoDefault> FfxDx12RRPath;
    CustomOptional<std::wstring, NoDefault> FfxDx12RCPath;
    CustomOptional<std::wstring, NoDefault> FfxVkPath;
    CustomOptional<std::wstring, NoDefault> XeSSLibrary;
    CustomOptional<std::wstring, NoDefault> XeFGLibrary;
    CustomOptional<std::wstring, NoDefault> XeLLLibrary;
    CustomOptional<std::wstring, NoDefault> XeSSDx11Library;
    CustomOptional<std::wstring, NoDefault> NvngxPath;
    CustomOptional<std::wstring, NoDefault> NVNGX_DLSS_Library;
    CustomOptional<std::wstring, NoDefault> DLSSFeaturePath;
    CustomOptional<std::wstring, NoDefault> NvapiDllPath;

    // Sharpness
    CustomOptional<SharpenShader> SharpnessShader { SharpenShader::RCAS };
    CustomOptional<bool> OverrideSharpness { false };
    CustomOptional<float> Sharpness { 0.4f };

    // RCAS
    CustomOptional<bool> RcasEnabled { false };
    CustomOptional<bool> ContrastEnabled { false };
    CustomOptional<float> Contrast { -0.3f };

    // DA Sharpening
    CustomOptional<float, NoDefault> DADepthScale;
    CustomOptional<float, NoDefault> DADepthBias;
    CustomOptional<bool, NoDefault> DAClampOutput;

    // MAS
    CustomOptional<bool> MotionSharpnessEnabled { false };
    CustomOptional<bool> MotionSharpnessDebug { false };
    CustomOptional<float> MotionSharpness { 0.2f };
    CustomOptional<float> MotionThreshold { 0.0f };
    CustomOptional<float> MotionScaleLimit { 10.0f };

    // Magnifier
    CustomOptional<bool> MagnifierEnabled { false };
    CustomOptional<float> MagnifierSize { 15.f }; // % of screen Height
    CustomOptional<int> MagnifierZoomFactor { 4 };
    CustomOptional<float> MagnifierBorderSize { 0.3f };   // % of screen Height
    CustomOptional<float> MagnifierCursorOffsetX { 0.f }; // Pixels
    CustomOptional<float> MagnifierCursorOffsetY { 0.f }; // Pixels
    CustomOptional<float, NoDefault> MagnifierStaticPosX; // % of screen Width, static pos enabled if both are defined
    CustomOptional<float, NoDefault> MagnifierStaticPosY; // % of screen Height

    // Menu
    CustomOptional<float, NoDefault> MenuScale;
    CustomOptional<bool> OverlayMenu { true };
    CustomOptional<int> ShortcutKey { VK_INSERT };
    CustomOptional<bool> ExtendedLimits { false };
    CustomOptional<bool> ShowFps { false };
    /// 0 Top Left, 1 Top Right, 2 Bottom Left, 3 Bottom Right
    CustomOptional<FpsOverlayPos> FpsOverlayPosition { FpsOverlayPos_TopLeft };
    /// 0 Only FPS, 1 +Avg FPS & Upscaler info 2 +Frame Time,
    /// 3 +Upscaler Time, 4 +Frame Time Graph, 5 +Upscaler Time Graph
    /// 6 +Reflex timings
    CustomOptional<FpsOverlay> FpsOverlayType { FpsOverlay_JustFPS };
    CustomOptional<int> FpsShortcutKey { VK_PRIOR };
    CustomOptional<int> FpsCycleShortcutKey { VK_NEXT };
    CustomOptional<bool> FpsOverlayHorizontal { false };
    CustomOptional<float> FpsOverlayAlpha { 0.4f };
    CustomOptional<float, NoDefault> FpsScale; // No value means same as MenuScale
    CustomOptional<bool> UseHQFont { true };
    CustomOptional<bool> DisableSplash { false };
    // A gamepad drives the overlay: the overlay reads the gamepad (XInput) and the game is kept from it while the
    // overlay is open (the XInput hooks). OFF by default: with our XInput hooks in, R.E.A.L. VR's own overlay could
    // not be opened with the gamepad (Avatar, Star Wars Outlaws), and a read of ours is taken by R.E.A.L. VR for the
    // game's. Off = no XInput hook is installed and the overlay never reads the gamepad. Startup only.
    CustomOptional<bool> GamepadInMenu { false };
    // Write the ini whenever the overlay closes, so changes are kept without pressing Save Settings.
    CustomOptional<bool> AutoSaveSettings { true };
    // Where notifications appear: 0 top left, 1 top centre, 2 top right, 3 bottom left, 4 bottom centre,
    // 5 bottom right, 6 centre. Centre by default: in VR the top of the image is outside the view.
    CustomOptional<int> NotificationPosition { 6 };
    CustomOptional<float> FontSize { 14.0f };
    CustomOptional<std::wstring, NoDefault> TTFFontPath;
    CustomOptional<int> FGShortcutKey { VK_END };
    // The foveal crop on / off, and per-eye lighting on / off. Unbound by default: nothing of ours sits on a function
    // key that a game or another tool may own; the overlay has both switches, and its Keybinds section sets a key.
    CustomOptional<int> FoveaCropKey { UnboundKey };
    CustomOptional<int> PerEyeLightingKey { UnboundKey };
    CustomOptional<bool> LightTheme { false };
    CustomOptional<bool> OverlaysUseTheme { false };
    CustomOptional<float> MenuAccentColorR { 0.00f };
    CustomOptional<float> MenuAccentColorG { 0.40f };
    CustomOptional<float> MenuAccentColorB { 0.77f };
    CustomOptional<float> MenuBGColorR { 0.0f };
    CustomOptional<float> MenuBGColorG { 0.0f };
    CustomOptional<float> MenuBGColorB { 0.0f };
    CustomOptional<float> MenuBGColorA { 0.99f };

    // Hooks
    CustomOptional<bool> HookOriginalNvngxOnly { false };
    CustomOptional<bool> EarlyHooking { false };
    CustomOptional<bool> UseNtdllHooks { true };

    // Upscale Ratio Override
    CustomOptional<bool> UpscaleRatioOverrideEnabled { false };
    CustomOptional<float> UpscaleRatioOverrideValue { 1.3f };

    // DRS
    CustomOptional<bool> DrsMinOverrideEnabled { false };
    CustomOptional<bool> DrsMaxOverrideEnabled { false };

    // Quality Overrides
    CustomOptional<bool> QualityRatioOverrideEnabled { false };
    CustomOptional<float> QualityRatio_DLAA { 1.0f };
    CustomOptional<float> QualityRatio_UltraQuality { 1.3f };
    CustomOptional<float> QualityRatio_Quality { 1.5f };
    CustomOptional<float> QualityRatio_Balanced { 1.7f };
    CustomOptional<float> QualityRatio_Performance { 2.0f };
    CustomOptional<float> QualityRatio_UltraPerformance { 3.0f };

    // ProcessFilter
    CustomOptional<std::wstring, NoDefault> TargetProcess;
    CustomOptional<std::wstring> ProcessExclusionList = {
        L"crashpad_handler.exe|crashreport.exe|crashreporter.exe|crs-handler.exe|crs-uploader.exe|crs-video.exe|"
        L"unitycrashhandler64.exe|idtechlauncher.exe|cefviewwing.exe|ace-setup64.exe|ace-service64.exe|"
        L"qtwebengineprocess.exe|platformprocess.exe|bugsplathd64.exe|bssndrpt64.exe|pspcsdkappmgr.exe|pspcsdkcore.exe|"
        L"pspcsdkstttts.exe|pspcsdktelemetry.exe|pspcsdkui.exe|pspcsdkupdatechecker.exe|pspcsdkvoicechat.exe|"
        L"pspcsdkwebview.exe|windhawk.exe|vscodium.exe|crash_reporter.exe|steamerrorreporter64.exe|crashreportclient."
        L"exe|edcefcrashpadprocess.exe|edcefrenderprocess.exe"
    };

    // Hotfixes
    CustomOptional<bool> CheckForUpdate { false }; // never run in this fork (VersionCheck::Start)
    CustomOptional<bool, SoftDefault> DisableOverlays { false };

    CustomOptional<bool> SimulateWaitableObject { false };

    CustomOptional<float, NoDefault> MipmapBiasOverride; // disabled by default
    CustomOptional<bool> MipmapBiasFixedOverride { false };
    CustomOptional<bool> MipmapBiasScaleOverride { false };
    CustomOptional<bool> MipmapBiasOverrideAll { false };

    CustomOptional<int, NoDefault> AnisotropyOverride; // disabled by default
    CustomOptional<bool> OverrideShaderSampler { true };
    CustomOptional<bool> AnisotropyModifyComp { true };
    CustomOptional<bool> AnisotropyModifyMinMax { true };
    CustomOptional<bool> AnisotropySkipPointFilter { true };

    CustomOptional<int, NoDefault> RoundInternalResolution; // disabled by default

    CustomOptional<int, NoDefault> SkipFirstFrames; // disabled by default
    CustomOptional<bool> RestoreComputeSignature { false };
    // Vulkan (a Vulkan game: OptiScaler alone broke VR's start -- a black screen,
    // VR stopped at the pause menu -- and the game ran fine without it). Both default to what OptiScaler always did.
    // false = only the extensions DLSS needs are added to the game's Vulkan instance and device (NGX's; not FFX's,
    // XeSS's, the D3D12 bridge's, NV low latency, Anti-Lag).
    CustomOptional<bool> VulkanExtraExtensions { true };
    // false = no vkQueueSubmit hook (it serves only the upscalers bridged to D3D12, the *_on12 backends).
    CustomOptional<bool> VulkanSubmitHook { true };
    // true = no sleeps in the Vulkan feature create / release, and a released feature is destroyed 8 Presents later
    // instead of behind vkDeviceWaitIdle (inputs/VkRetire.h). false = OptiScaler as it always was.
    CustomOptional<bool> VulkanNoStalls { false };
    // With VulkanNoStalls: true = the released feature is destroyed 8 Presents later; false = at once, in the
    // game's own ReleaseFeature call, with no wait, as without OptiScaler (a Vulkan game: a late
    // release, 6 s on, broke the headset session's restart).
    CustomOptional<bool> VulkanRetireLater { false };
    CustomOptional<bool> RestoreGraphicSignature { false };
    CustomOptional<bool> ExtendedStateRestore { false };

    CustomOptional<bool> UsePrecompiledShaders { true };

    CustomOptional<bool> UseGenericAppIdWithDlss { false };
    CustomOptional<bool> PreferDedicatedGpu { true };
    CustomOptional<bool> PreferFirstDedicatedGpu { false };

    CustomOptional<int32_t, NoDefault> ColorResourceBarrier;    // disabled by default
    CustomOptional<int32_t, NoDefault> MVResourceBarrier;       // disabled by default
    CustomOptional<int32_t, NoDefault> DepthResourceBarrier;    // disabled by default
    CustomOptional<int32_t, NoDefault> ExposureResourceBarrier; // disabled by default
    CustomOptional<int32_t, NoDefault> MaskResourceBarrier;     // disabled by default
    CustomOptional<int32_t, NoDefault> OutputResourceBarrier;   // disabled by default

    CustomOptional<bool> CreateD3D12DeviceForLuma { false };

    // Upscalers
    CustomOptional<Upscaler, SoftDefault> Dx11Upscaler { Upscaler::FSR22 };
    CustomOptional<Upscaler, SoftDefault> Dx12Upscaler { Upscaler::XeSS };
    CustomOptional<Upscaler, SoftDefault> VulkanUpscaler { Upscaler::FSR22 };

    // Output Scaling
    CustomOptional<bool> OutputScalingEnabled { false };
    CustomOptional<float> OutputScalingMultiplier { 1.5f };
    CustomOptional<Scaler> OutputScalingDownscaler { Scaler::FSR1 };

    // FSR
    CustomOptional<bool> FsrDebugView { false };
    CustomOptional<int> FfxUpscalerIndex { 0 };
    CustomOptional<int> FfxFGIndex { 0 };
    CustomOptional<bool> FsrUseMaskForTransparency { true };
    CustomOptional<bool> FsrNonLinearColorSpace { false };
    CustomOptional<bool> FsrNonLinearSRGB { false };
    CustomOptional<bool> FsrNonLinearPQ { false };
    CustomOptional<bool> FsrAgilitySDKUpgrade { false };

    // These default values will be overwritten at upscaler init time with optimized values
    CustomOptional<float> FsrVelocity { 1.0f };
    CustomOptional<float> FsrReactiveScale { 1.0f };
    CustomOptional<float> FsrShadingScale { 1.0f };
    CustomOptional<float> FsrAccAddPerFrame { 0.333f };
    CustomOptional<float> FsrMinDisOccAcc { -0.333f };

    // FSR4
    CustomOptional<FSR4Support> Fsr4ForceModel { FSR4Support::None };
    CustomOptional<uint32_t, NoDefault> Fsr4Preset;
    CustomOptional<bool> Fsr4EnableWatermark { false };
    CustomOptional<bool> Fsr4DoNotLoadAmdxc64 { false };

    // FSR Common
    CustomOptional<float> FsrVerticalFov { 60.0f };
    CustomOptional<float> FsrHorizontalFov { 0.0f }; // off by default
    CustomOptional<float> FsrCameraNear { 0.1f };
    CustomOptional<float> FsrCameraFar { 100000.0f };
    CustomOptional<bool> FsrUseFsrInputValues { true };

    // dx11wdx12
    CustomOptional<bool> Dx11DelayedInit { false };
    CustomOptional<bool> DontUseNTShared { true };
    // The bridge's D3D12 device is made through ID3D12DeviceFactory, not D3D12CreateDevice, so that a hook on the
    // latter (ReShade's, which tells its add-ons of every new device) does not see it. On by itself with the
    // SekiroTSR mod. See with_dx12/DeviceByFactory.h.
    CustomOptional<bool> Dx11w12DeviceByFactory { false };

    // vulkanwdx12
    CustomOptional<bool> VulkanUseCopyForInputs { false };
    CustomOptional<bool> VulkanUseCopyForOutput { false };

    // NVAPI Override
    CustomOptional<bool> DisableFlipMetering { false };

    // Spoofing
    CustomOptional<bool, SoftDefault> DxgiSpoofing { true };
    CustomOptional<bool> DxgiFactoryWrapping { false };
    CustomOptional<bool> StreamlineSpoofing { true };
    CustomOptional<std::string, NoDefault> DxgiBlacklist; // disabled by default
    CustomOptional<int, NoDefault> DxgiVRAM;              // disabled by default
    CustomOptional<bool> VulkanSpoofing { false };
    CustomOptional<bool> VulkanExtensionSpoofing { false };
    CustomOptional<int, NoDefault> VulkanVRAM; // disabled by default
    CustomOptional<bool> SpoofHAGS { false };
    CustomOptional<bool> SpoofFeatureLevel { false };
    CustomOptional<uint32_t> SpoofedVendorId { VendorId::Nvidia };
    CustomOptional<uint32_t> SpoofedDeviceId { 0x2684 };
    CustomOptional<uint32_t, NoDefault> TargetVendorId;
    CustomOptional<uint32_t, NoDefault> TargetDeviceId;
    CustomOptional<std::wstring> SpoofedGPUName { L"NVIDIA GeForce RTX 4090" };
    CustomOptional<bool> UESpoofIntelAtomics64 { false };
    CustomOptional<bool> SpoofRegistry { false };
    CustomOptional<bool> SpoofUser32 { false };
    CustomOptional<std::wstring> SpoofedDriver { L"32.0.15.9155" };

    // Plugins
    CustomOptional<std::wstring, NoDefault> PluginPath;
    CustomOptional<bool> LoadSpecialK { false };
    CustomOptional<bool> LoadReShade { false };
    CustomOptional<bool> LoadCustomAmdxc64OnRdna2 { false };
    CustomOptional<bool> LoadAsiPlugins { false };
    CustomOptional<int> LateAsiPluginsDelay { 30 };
    // Startup only. A DLL name (e.g. reshade64.dll): when set, EVERY .asi in the plugin folder loads right after that
    // DLL has been loaded (the LdrLoadDll hook), instead of at startup / after LateAsiPluginsDelay. In some games
    // ReShade64.dll is already loaded, and NaturalVision Enhanced must register with ReShade before the game's device.
    CustomOptional<std::wstring, NoDefault> LateAsiPluginsAfter;

    // Frame Generation
    CustomOptional<FGInput> FGInput { FGInput::NoFG };
    CustomOptional<bool> ExternalFrameGeneration { false };
    CustomOptional<FGOutput> FGOutput { FGOutput::NoFG };
    CustomOptional<FGNvngxReplacement> FGNvngxReplacement { FGNvngxReplacement::None };
    CustomOptional<bool> FGDrawUIOverFG { false };
    CustomOptional<bool> FGUIPremultipliedAlpha { true };
    CustomOptional<bool> FGDisableHudless { false };
    CustomOptional<bool> FGDisableUI { false };
    CustomOptional<bool> FGSkipReset { false };
    CustomOptional<int> FGAllowedFrameAhead { 1 };
    CustomOptional<bool> FGDepthValidNow { false };
    CustomOptional<bool> FGVelocityValidNow { false };
    CustomOptional<bool> FGHudlessValidNow { false };
    CustomOptional<bool> FGOnlyAcceptFirstHudless { false };
    CustomOptional<bool> FGPreserveSwapChain { true };
    CustomOptional<bool> FGSkipResizeBuffers { false };
    CustomOptional<bool> FGModifyBufferState { false };
    CustomOptional<bool> FGModifySCIndex { false };
    CustomOptional<float> FGHudCutoff { 0.0f };
    CustomOptional<FrameTimeSource> FTInput { FrameTimeSource::Input };

    // OptiFG
    CustomOptional<bool> FGEnabled { false };
    CustomOptional<bool> FGUseMutexForSwapchain { true };
    CustomOptional<bool> FGMakeMVCopy { true };
    CustomOptional<bool> FGMakeDepthCopy { true };
    CustomOptional<bool> FGResourceFlip { false };
    CustomOptional<bool> FGResourceFlipOffset { false };
    CustomOptional<bool> FGAlwaysCaptureFSRFGSwapchain { false };

    CustomOptional<int, NoDefault> FGRectLeft;
    CustomOptional<int, NoDefault> FGRectTop;
    CustomOptional<int, NoDefault> FGRectWidth;
    CustomOptional<int, NoDefault> FGRectHeight;

    // OptiFG - Hudfix
    CustomOptional<bool> FGDisableHUDFix { false };
    CustomOptional<bool> FGHUDFix { false };
    CustomOptional<int> FGHUDLimit { 1 };
    CustomOptional<bool> FGHUDFixExtended { false };
    CustomOptional<bool> FGImmediateCapture { false };
    CustomOptional<bool> FGDontUseSwapchainBuffers { false };
    CustomOptional<bool> FGRelaxedResolutionCheck { false };
    CustomOptional<bool> FGHudfixDisableRTV { false };
    CustomOptional<bool> FGHudfixDisableSRV { false };
    CustomOptional<bool> FGHudfixDisableUAV { false };
    CustomOptional<bool> FGHudfixDisableOM { false };
    CustomOptional<bool> FGHudfixDisableDispatch { false };
    CustomOptional<bool> FGHudfixDisableDI { false };
    CustomOptional<bool> FGHudfixDisableDII { false };
    CustomOptional<bool> FGHudfixDisableSCR { true };
    CustomOptional<bool> FGHudfixDisableSGR { true };

    // OptiFG - Resource Tracking
    CustomOptional<bool> FGAlwaysTrackHeaps { false };
    CustomOptional<bool> FGResourceBlocking { false };
    CustomOptional<bool> FGUseShards { false };

    // OptiFG - DLSS-D Depth scale
    CustomOptional<bool> FGEnableDepthScale { false };
    CustomOptional<float> FGDepthScaleMax { 10000.0f };

    // FSR-FG
    CustomOptional<bool> FGDebugView { false };
    CustomOptional<bool> FGDebugResetLines { false };
    CustomOptional<bool> FGDebugTearLines { false };
    CustomOptional<bool> FGDebugPacingLines { false };
    CustomOptional<bool> FGAsync { false };
    CustomOptional<bool> FGFramePacingTuning { true };
    CustomOptional<float> FGFPTSafetyMarginInMs { 0.01f };
    CustomOptional<float> FGFPTVarianceFactor { 0.3f };
    CustomOptional<bool> FGFPTAllowHybridSpin { false };
    CustomOptional<int> FGFPTHybridSpinTime { 2 };
    CustomOptional<bool> FGFPTAllowWaitForSingleObjectOnFence { false };

    CustomOptional<bool> FSRFGSkipConfigForHudless { false };
    CustomOptional<bool> FSRFGSkipDispatchForHudless { false };
    CustomOptional<bool> FSRFGEnableWatermark { false };

    // XeFG
    CustomOptional<bool> FGXeFGIgnoreInitChecks { false };
    CustomOptional<int> FGXeFGInterpolationCount { 1 };
    CustomOptional<bool> FGXeFGUIComposition { false };
    CustomOptional<bool> FGXeFGDepthInverted { true };
    CustomOptional<bool> FGXeFGJitteredMV { false };
    CustomOptional<bool> FGXeFGHighResMV { false };
    CustomOptional<bool> FGXeFGDebugView { false };
    CustomOptional<bool> FGXeFGForceBorderless { false };

    // DLSSG
    CustomOptional<int> FGDLSSGInterpolationCount { 1 }; // For Opti's own SL instance
    CustomOptional<bool> FGDLSSGUseGamesReflexMarkers { true };
    CustomOptional<int, NoDefault>
        FGDLSSGOverrideInterpolationCount; // For overriding game's value sent to SL, could be Nvngx FG, could be noFG
                                           // but someone just uses real DLSSG
    CustomOptional<bool> FGDLSSGOverrideForceDMFG { false };   // Overrides game's DLSSG mode to Dynamic
    CustomOptional<bool> FGDLSSGForceDMFG { false };           // Overrides Opti's DLSSG mode to Dynamic
    CustomOptional<float> FGDLSSGFramerateTargetDMFG { 0.0f }; // 0.0 means auto-detects the display refresh rate

    // As per
    // https://github.com/artur-graniszewski/dlss-enabler-main/blob/a92464d468eb0d91ae17befa66c6bf6229f20b9f/Utils/DlssgProxy.cpp#L1033
    CustomOptional<uint32_t> NvngxFGDispatchFlags { 0x10000000 }; // IGNORE_UI_TEXTURE
    CustomOptional<bool> NvngxFGShowDebug { false };
    CustomOptional<bool> NvngxFGDisableHudless { false };

    // fakenvapi
    CustomOptional<bool> UseFakenvapi { true };
    CustomOptional<bool> ForceXeLL { false };
    CustomOptional<bool> FN_ForceLatencyFlex { false };
    CustomOptional<LFXMode> FN_LatencyFlexMode { LFXMode::Conservative };
    CustomOptional<ForceReflex> FN_ForceReflex { ForceReflex::InGame };
    CustomOptional<LowLatencyInput> LowLatencyInput { LowLatencyInput::Auto }; // TODO: no reading/saving to config
    CustomOptional<LowLatencyMode> LowLatencyOutput { LowLatencyMode::Auto };

    // Inputs
    CustomOptional<bool> EnableDlssInputs { true };
    CustomOptional<bool> EnableXeSSInputs { true };
    CustomOptional<bool> UseFsr2Inputs { true };
    CustomOptional<bool> UseFsr2Dx11Inputs { false };
    CustomOptional<bool> UseFsr2VulkanInputs { false };
    CustomOptional<bool> Fsr2Pattern { false };
    CustomOptional<bool> UseFsr3Inputs { true };
    CustomOptional<bool> Fsr3Pattern { false };
    CustomOptional<bool> UseFfxInputs { true };
    CustomOptional<bool> EnableHotSwapping { false };
    CustomOptional<bool> EnableFsr2Inputs { true };
    CustomOptional<bool> EnableFsr3Inputs { true };
    CustomOptional<bool> EnableFfxInputs { true };
    CustomOptional<bool> DuniaTaaHost { true };
    // 2 = automatic (default): upscaling when gameplay renders below the eye size (Resolution Scale below
    // 100%), DLAA to carry NR at 100%. 0 forces DLAA, 1 forces upscaling.
    CustomOptional<int> DuniaTaaQuality { 2 };
    // Startup only. FC6: run DLAA+NR on the finished tonemapped 8-bit eye image (just before the VR mod takes
    // it) instead of on the HDR image at the game's upscale. The game then does its own upscale and TAA.
    CustomOptional<bool> DuniaTaaNrOnTonemapped { false };
    // FC6: wait for F6 before creating the DLSS feature and running it, the same habit as FC5 and Days Gone.
    // The feature is created on the first press (a few seconds' freeze) and is never rebuilt, so this leaves
    // room to set the Resolution Scale and the other settings first; F6 afterwards switches between the
    // game's own picture and ours. false = start by itself once gameplay is seen, as it did before.
    CustomOptional<bool> DuniaTaaStartOnKey { true };
    // Ini only. Shows and enables the diagnostics kept for future motion vector work: the settings below
    // marked "diagnostics" do nothing while this is off.
    CustomOptional<bool> DuniaTaaDiagnostics { false };
    // Diagnostics. 0 = no motion vectors, history reset every frame (FC6 has no camera motion vector
    // buffer). 1 = game candidate as uv, 2 = inverted, 3 = as pixels, 4 = pixels inverted.
    CustomOptional<int> DuniaTaaMotionVectors { 0 };
    // Diagnostics. Rank R16G16_UNORM motion candidates above R16G16_FLOAT ones.
    CustomOptional<bool> DuniaTaaMvPreferUnorm { false };
    // Look for the game's motion vector pass by what it writes: the three UAVs it binds together, at half
    // the eye size. Costs the compute descriptor table hook and ResTrack's heap mirroring, but only until
    // the pass is found -- both go quiet after that. Startup-only, because the hook is installed once.
    CustomOptional<bool> DuniaTaaFindMotion { false };
    // Feed DLSS the game's real motion vectors: find the pass, copy what it writes right after its
    // dispatch, decode it at the splice. Implies DuniaTaaFindMotion, and keeps ResTrack's heap mirroring
    // on for the session -- the cbuffer the matrix lives in can only be reached through a descriptor.
    // F4 toggles the whole path live, so its cost can be judged inside one run. Startup-only.
    CustomOptional<bool> DuniaTaaRealMotion { true };
    // Pixels of motion that saturate the DuniaTaaUpscaleView = 3 view. The original 8 was chosen when the
    // only vectors on offer were near zero; a head turn is 60-100 px, which clips to solid colour.
    CustomOptional<int> DuniaTaaMotionViewScale { 64 };
    // Find the game's own TAA resolve and skip its draw. This is the step the FC5 and Days Gone ports both
    // took and FC6 never has: without it our DLAA is a second temporal pass over an image the game has
    // already resolved, which is why real motion vectors changed so little. Startup-only; F8 cycles the
    // states live once the pass has been found. Implies DuniaTaaFindMotion, whose search this reuses.
    CustomOptional<bool> DuniaTaaFindTaa { false };
    // Step 3 of the port: run per-eye DLSS *at* that resolve and skip the game's draw, so DLSS is the
    // temporal resolve rather than a second pass over one. Implies DuniaTaaFindTaa, and turns
    // DuniaTaaNrOnTonemapped off -- the two compete for the one DLSS feature, and it is created once for
    // one size and format. Startup-only; F8 then cycles game TAA / skipped / DLSS spliced.
    CustomOptional<bool> DuniaTaaSpliceTaa { true };
    // FC6, splice mode below 100% Resolution Scale: DLSS render -> eye in place of the game's upscale (true) or
    // DLAA at render size with the game's own upscale on top (false). Read at feature creation.
    CustomOptional<bool> DuniaTaaSpliceUpscale { true };
    // Put the chained-lookup motion mode back into F4's cycle. Off by default and deliberately so: both FC6 and
    // Days Gone rejected it ("lots of smear and quick sand textures"), and leaving it one keypress from the
    // everyday states meant it kept being what was actually on screen. The code stays; the trap does not.
    CustomOptional<bool> DuniaTaaChainedMotion { false };
    // FC6: rewrite the motion pass's reprojection matrix so the game's own vectors are SAME-eye. Adds a state
    // to F4 after "MATRIX + object motion"; off, nothing in the game's memory is ever written.
    CustomOptional<bool> DuniaTaaSameEyeVectors { true };
    // FC6 jitter: 0 DLSS told zero, 1 told the game's own, 2 own per-eye Halton patched into the view constants.
    CustomOptional<int> DuniaTaaJitter { 2 };
    // Diagnostics, overlay button only: log every bind and copy for the next two Presents. Never saved.
    CustomOptional<bool> DuniaTaaTraceFrame { false };
    // Diagnostics. Upscaling A/B: 0 = DLSS, 1 = the game's own upscale, 2 = split (DLSS left, game
    // right), 3 = the motion vectors DLSS receives.
    CustomOptional<int> DuniaTaaUpscaleView { 0 };
    // Diagnostics. The resource state the F7 D3D12 capture assumes a motion vector candidate is in at the splice
    // point: 0 = guess from the resource's flags, 1 = render target, 2 = unordered access, 3 = non-pixel shader
    // resource, 4 = pixel shader resource, 5 = common. A wrong guess garbles that candidate's .dds.
    CustomOptional<int> DuniaTaaCaptureMvState { 0 };
    // Reversed-Z depth. Wrong here breaks disocclusion at depth edges, where AO and shadows live.
    CustomOptional<bool> DuniaTaaDepthInverted { true };
    // Smallest target treated as a scene/eye buffer rather than menu or UI. A target passes on
    // either axis. Lower these if the per-eye resolution drops below them, or nothing is processed.
    CustomOptional<int> DuniaTaaMinEvalWidth { 2560 };
    CustomOptional<int> DuniaTaaMinEvalHeight { 1440 };
    // TEST ONLY: the size learners see every square target of 1024+ px as 4096x4096 (a probe pass that
    // outgrew the eye). See FakeProbeSize in inputs/DuniaTaa_Dx12.cpp. Never on in a shared build.
    CustomOptional<bool> DuniaTaaTestFakeProbe { false };
    // TEST ONLY: apply the square-eye rules (a scaled scene must be confirmed by the TAA resolve's table) on any eye.
    CustomOptional<bool> DuniaTaaTestSquareEye { false };
    // Diagnostics: GPU timestamps around our passes at the splice, logged per 600 Presents (live in the overlay).
    CustomOptional<bool> DuniaTaaGpuTiming { false };
    // Startup only (Dark Souls III): DLAA+NR on the finished render-target-only 8-bit eye image, after the
    // command list (or immediate-context draw) that draws it and before the VR mod copies it. Nothing happens while
    // NR is off.
    CustomOptional<bool> EyeImageNr { false };
    // Startup only, diagnostics: log what reaches OptiScaler's own D3D11 DLSS feature (inputs, jitter pattern) and,
    // once, which exe functions a mod hooked. See inputs/UpscalerInputProbe_Dx11.h.
    CustomOptional<bool> UpscalerInputProbe { false };
    // Startup only, diagnostics (D3D12): hooks for the overlay's "Measure the game's GPU frame" button, which times
    // the game's passes by kind and size for 64 Presents. Idle cost one atomic load per call. See
    // inputs/FrameSplit_Dx12.h.
    CustomOptional<bool> FrameSplitProbe { false };
    // Startup only, diagnostics (D3D12): F7 or the overlay's button saves what the upscaler is given and gives back
    // for 8 Presents in a row (delay: CaptureDelaySeconds). See inputs/EvalCapture_Dx12.h.
    CustomOptional<bool> UpscalerInputCapture { false };
    // Startup only, diagnostics (D3D12), needs FrameSplitProbe: F7 or the overlay's button saves the textures the
    // game's passes write, at the moment each is finished. F7 goes to this over UpscalerInputCapture. See
    // inputs/GiCapture_Dx12.h.
    CustomOptional<bool> GiBufferCapture { false };
    // Startup only (D3D12, Crimson Desert): the same-eye lighting history, F8 or the game's section in the
    // overlay. Brings the hooks GiBufferCapture and FrameSplitProbe bring, none of their captures or test keys. Starts off.
    CustomOptional<bool> PerEyeLighting { false };
    // Startup only: read the eyes' gaze from a Varjo headset (Varjo Base's runtime) for the foveal crop. Step 1: the
    // outline (F8) also draws a crop-sized box where the eyes look. See inputs/VarjoGaze.h.
    CustomOptional<bool> UpscalerFoveaGaze { false };
    // Live (overlay), degrees: a constant correction of the gaze, + right, + up.
    CustomOptional<float> UpscalerFoveaGazeYaw { 0.0f };
    CustomOptional<float> UpscalerFoveaGazePitch { 0.0f };
    // Live (overlay), 0..1: the gaze filter's steadiness while the eyes hold still.
    CustomOptional<float> UpscalerFoveaGazeSteadiness { 0.6f };
    // The smoothed gaze is put where the eyes landed at once after a saccade (GazeFilter.h); false = eased in, as
    // it used to be (the crop arrived after the eye). Live; the Junk Drawer's 'Gaze snaps to where the eyes land'.
    CustomOptional<bool> UpscalerFoveaGazeSnap { true };
    // Live (overlay): the gaze's range per direction around the corrected centre, 1 = as measured.
    CustomOptional<float> UpscalerFoveaGazeRangeUp { 1.0f };
    CustomOptional<float> UpscalerFoveaGazeRangeDown { 1.0f };
    CustomOptional<float> UpscalerFoveaGazeRangeLeft { 1.0f };
    CustomOptional<float> UpscalerFoveaGazeRangeRight { 1.0f };
    // Live (overlay): where the gaze comes from (VarjoGaze.h): 0 auto (the OpenXR layer's XR_EXT_eye_gaze_interaction when
    // it gives one, else Varjo's reader), 1 Varjo's reader only, 2 the OpenXR layer only.
    CustomOptional<int> UpscalerFoveaGazeSource { 0 };
    // Live (overlay): the projection the gaze and the calibration targets are placed with when DLSS hands none
    // (PerEyeDlss_Dx12 GazeProjection): 0 the headset's FOV scale, CENTRED, 1 the headset's FOV as located (asymmetric;
    // in one game a straight gaze sat 38% toward the nose, wrong), 2 the FOV the VR mod submits with the image, scaled from its image
    // rect to the eye's full image (in VR the submitted FOV is the truth
    // about the picture; the default), 3 the submitted FOV centred and scaled.
    CustomOptional<int> UpscalerFoveaGazeProjection { 2 };
    // Live (overlay 'Crop move'): what a move of the foveal crop does to its DLSS history (PerEyeDlss_Dx12 BeginFovea):
    // 0 carried over (the vectors shifted by the move), 1 carried over with the shift's sign flipped, 2 restarted.
    CustomOptional<int> UpscalerFoveaMoveMode { 0 };
    CustomOptional<int> UpscalerFoveaCropMask { 1 }; // the game's bias mask at an SR crop: 0 none, 1 copied, 2 as handed
    CustomOptional<bool> UpscalerFoveaShiftByOffset { false }; // a crop move's shift through DLSS's MV.Offset (test)
    // An RR crop's vectors: 0 ours (the fit's), 1 the ones left behind below us after the
    // full-size pass of the same evaluate (the default: what the crop was validated on under DL(S)SS; a frame with
    // none there takes ours, counted per window).
    CustomOptional<int> UpscalerFoveaRrVectors { 1 };
    // An RR crop's feature made and run on nvngx_dlssd.dll's own entries through the forwarder, past the driver core's
    // exports (needed while DL(S)SS runs).
    // Takes effect when the crop's feature is made. Overlay 'RR crop past DL(S)SS (direct DLL)'.
    CustomOptional<bool> UpscalerFoveaRrDirect { true };
    // The same for an SR crop (DLSSFeatureDx12, on nvngx_dlss.dll's own entries through the forwarder). On by
    // default: validated under DL(S)SS in one D3D12 game (every crop evaluate ok, none failed).
    CustomOptional<bool> UpscalerFoveaSrDirect { true };
    // The per-eye twin (a full DLSS feature at the game's size, for the other eye's Presents while the routing is per
    // eye, F6): made at the FIRST PER-EYE EVALUATE, not at the game's create (in one D3D12 game
    // while DL(S)SS runs, the twin held 2.9 GB and took 0 evaluates all run; a live create took 70-150 ms there, as the crop's
    // and the periphery's do). true = at the game's create, as before. Launch-time.
    CustomOptional<bool> UpscalerTwinAtCreate { false };
    // An SR crop's vectors: 0 ours (the fit's), 1 the ones left behind below us after the DLSS
    // full-size pass / cheap periphery of the same evaluate (none with the own TAA periphery), -1 auto (default) = 1
    // while DL(S)SS is seen at work on that pass, else 0. Forced 1 in one D3D12 game: no shimmer
    // while driving, even in fast motion. Our fit is skipped while nothing takes it.
    CustomOptional<int> UpscalerFoveaSrVectors { -1 };
    // Live (overlay 'Crop sits on whole output pixels'): a placed crop's render origin is chosen (within 3 px) so that
    // its output origin is a whole pixel; false = the nearest render origin, its output origin rounded (up to half a
    // pixel of displacement inside the crop, changing at every step of a glide).
    CustomOptional<bool> UpscalerFoveaPlaceExact { true };
    // Live (overlay): the foveal crop follows the gaze; it recentres when the gaze leaves this fraction of its half size.
    CustomOptional<bool> UpscalerFoveaGazeFollow { false };
    // The in-game gaze calibration (VarjoGaze.h): view angle = c0 + c1 raw yaw + c2 raw pitch, per axis.
    CustomOptional<bool> UpscalerFoveaGazeCal { false };
    CustomOptional<float> UpscalerFoveaGazeCalYaw0 { 0.0f };
    CustomOptional<float> UpscalerFoveaGazeCalYaw1 { 1.0f };
    CustomOptional<float> UpscalerFoveaGazeCalYaw2 { 0.0f };
    CustomOptional<float> UpscalerFoveaGazeCalPitch0 { 0.0f };
    CustomOptional<float> UpscalerFoveaGazeCalPitch1 { 0.0f };
    CustomOptional<float> UpscalerFoveaGazeCalPitch2 { 1.0f };
    CustomOptional<float> UpscalerFoveaGazeDeadZone { 0.5f };
    // Live (overlay): the crop glides after the gaze with this time constant, ms (0 = it jumps, as before).
    CustomOptional<float> UpscalerFoveaGazeGlideMs { 60.0f };
    // Startup only, with FrameSplitProbe: also hook CreateCommandSignature to tell indirect draws from dispatches.
    // Off by default: the first build with it crashed Crimson Desert at start.
    CustomOptional<bool> FrameSplitSignatures { false };
    // Startup only: the per-eye module (D3D11, D3D12, Vulkan): the foveal crop, the cheap periphery and the per-eye
    // routing exist only with it. On by default: this build is made for alternate eye rendering. On D3D11 it also
    // makes a twin DLSS feature for each one a game or mod creates on OptiScaler's own path (one history per eye
    // once the routing is per eye). See inputs/PerEyeDlss_Dx11.h.
    CustomOptional<bool> UpscalerPerEye { true };
    // With UpscalerPerEye on D3D11: the routing starts per eye (the overlay's "On from the start"). Off: it starts
    // shared, the game's or mod's own picture, until F6.
    CustomOptional<bool> UpscalerPerEyeStart { false };
    // Live: GPU timestamps around what runs at a DLSS evaluate on OptiScaler's D3D11 path (the same-eye motion, and
    // on the D3D12 bridge NR before, the upscaler, NR after) and the render thread's time in the evaluate, as means
    // per 600 Presents in the log and in the overlay's section. See gpu_time/StageTimes.h.
    CustomOptional<bool> UpscalerGpuTiming { false };
    // D3D12 per eye, foveal crop (F4 on/off, F8 outline): the crop's fraction of the eye (an exactly aligned size within
    // 0.15 preferred), its DLSS preset (13 = M) and the blended edge in output px. See inputs/PerEyeDlss_Dx12.h.
    CustomOptional<float> UpscalerFoveaCrop { 0.4f };
    // The crop's height as a fraction of the eye; 0 = the same as the width (UpscalerFoveaCrop).
    CustomOptional<float> UpscalerFoveaCropHeight { 0.0f };
    CustomOptional<int> UpscalerFoveaPreset { 13 };
    CustomOptional<int> UpscalerFoveaFeather { 32 };
    // Frames a pixel must have been in the SR crop's box before the crop shows fully there (the periphery shows
    // through a younger one, in proportion); 0 = off (the default: the fade was no
    // help against the eye arriving first, and the RR games were never run with it). D3D12. Overlay 'Fade-in after a
    // move (frames)'.
    CustomOptional<int> UpscalerFoveaAgeFrames { 0 };
    // The crop on from the launch (Vulkan, FoveaCrop_Vk; the overlay's "Crop on from the launch"). D3D12's crop
    // still starts off and is switched by F4 / the overlay.
    CustomOptional<bool> UpscalerFoveaStart { false };
    // The crop's offset in fractions of the eye: x toward the nose (mirrored per eye, per eye mode only), y up.
    CustomOptional<float> UpscalerFoveaOffsetX { 0.0f };
    CustomOptional<float> UpscalerFoveaOffsetY { 0.0f };
    // The crop's sideways offset, the SAME for both eyes (+ = right), added to the nose offset; needs no eye told apart
    // (it just moves the crop horizontally if needed).
    CustomOptional<float> UpscalerFoveaOffsetSideways { 0.0f };
    // The crop's inputs copied into crop-sized textures (default) instead of read through DLSS's input subrects.
    CustomOptional<bool> UpscalerFoveaCopyInputs { true };
    // Per eye: past two rejections in a row, every second same-eye fit starts from no motion instead of the eye's last
    // accepted fit (a start left behind by fast motion is otherwise never given up). Live. false = the start as it was.
    CustomOptional<bool> UpscalerFitGiveUpStale { true };
    // Per eye: the same-eye fit restarts from no motion within the fit when its first round finds the eye's last
    // accepted fit left behind (a flick of the stick). Live. false = as before (two rejected fits first).
    CustomOptional<bool> UpscalerFitRestart { true };
    // Per eye: the same-eye fit leaves out everything nearer than a gap in its depth histogram (a car
    // around a first-person driver: 69-78% of the samples, held the fit for as long as a drive),
    // restarts when at most half of the rest lie within 8 px of its start, and steps 128, 8, 2 px after a restart.
    // Live. Off by default: every other game keeps the fit it was validated with.
    CustomOptional<bool> UpscalerFitLeaveOutNearest { false };
    // Per eye: what the composed same-eye motion adds beyond the fitted camera motion. 0 = nothing (camera only;
    // validated in one D3D12 game: no shimmer or warp with head and character movement). 1 = the game's
    // object motion x2 through FC5's validated 0.1-0.4 px knee (moving things only: the fit's own ~0.1 px residual
    // stays under it). 2 = the same through the 0.02-0.1 px knee (the older default: the D3D11 path's variant,
    // for vectors with no object motion; in one game it doubled the fit's residual everywhere and the picture warped). 3 = the
    // residual averaged over 3x3 taps through a 0.03-0.15 px knee (small animations and distant moving things, which
    // 1 drops: a test). Live.
    CustomOptional<int> UpscalerSameEyeObjectMotion { 0 };
    // Per eye: DLSS's current-colour bias mask (NVSDK_NGX_Parameter_DLSS_Input_Bias_Current_Color_Mask) made by the
    // compose from where the game's vector disagrees with the fitted camera (things that move by themselves). 0 = off,
    // 1 = from the point residual through the 0.1-0.4 px knee, 2 = from the 3x3-averaged residual through 0.03-0.15 px.
    // A test for the smear of small animations. Live.
    CustomOptional<int> UpscalerSameEyeBiasMask { 0 };
    // Per eye: Neural Rendering is told the eye of every evaluate by the Present's parity even while the DLSS routing
    // is shared (the default; before, shared routing named eye 0 for both eyes and NR's one history took
    // the eyes in turn: the picture flickered in stereo and not in mono). false = as before. Live.
    CustomOptional<bool> UpscalerNrEyeByParity { true };
    // Per eye: with the DLSS routing shared, the foveal crop, the cheap periphery and the TAA periphery keep one
    // feature (history) per Present parity = per eye (the default; before, one for both eyes: with
    // DL(S)SS running the crop showed the usual other-eye shimmer while the rest of the eye was clean). Live.
    CustomOptional<bool> UpscalerCropEyeByParity { true };
    // The cheap DLSS periphery as ONE feature for both eyes in shared routing, on the game's own vectors, as the game's
    // feature is (with the crop past the core, the periphery running as a feature per eye
    // taking turns did not work while DL(S)SS runs;
    // one feature stopped it, and the periphery's shimmer and swim went). Without DL(S)SS
    // one feature mixes the eyes' histories. 0 auto (one feature while DL(S)SS is seen at work on the periphery),
    // 1 always one, 2 always one per eye. Live (the periphery's history restarts). Overlay 'Periphery
    // features'.
    CustomOptional<int> UpscalerPeripheryOneFeature { 0 };
    // Vulkan: the features of our own (crop, periphery) made on a worker thread with a queue of its own (one more
    // queue asked for at vkCreateDevice), so a cold DLSS create never holds a Present. OFF BY DEFAULT (a VR restart
    // after a one-time stall is preferred over any crash risk): NVIDIA's guide says the NGX API is
    // not thread-safe, and the worker's create overlaps the game's evaluate; a hang under the periphery's
    // create loop is unexplained. Off: made inline on the game's thread (a cold pair stalls one Present ~1.3 s,
    // which restarts the headset session; warm after that).
    CustomOptional<bool> UpscalerVkCreateWorker { false };
    // Per eye: under DL(S)SS no same-eye work of ours is needed: no same-eye fit or
    // compose of ours runs, NR / the crop / the periphery read them as they are. Off: the game's vectors point at the
    // other eye and ours are made from them. Live (overlay). Wrong either way = the chain is wrong: say so in the log.
    CustomOptional<bool> UpscalerVectorsSameEye { false };
    // Per eye: the depth the same-eye fit takes. 0 = Auto (linear when RR is told DLSS.Use.HW.Depth = 0, e.g.
    // Hogwarts Legacy), 1 = hardware depth, 2 = linear depth (the fit works on 1/depth). Live.
    CustomOptional<int> UpscalerFitDepth { 0 };
    // Cheap periphery: a render-size DLSS pass (its own preset, 5 = E) enlarged over the eye replaces the full pass.
    // On by default: with the crop on, it is what saves the time.
    CustomOptional<bool> UpscalerFoveaCheapPeriphery { true };
    CustomOptional<int> UpscalerFoveaPeripheryPreset { 5 };
    // RR mode: the cheap periphery as SR instead of RR (no denoising outside the crop). On by default: RR's cost
    // follows its input size, so an RR periphery saves little.
    CustomOptional<bool> UpscalerFoveaPeripherySr { true };
    // Per eye: our own TAA over the whole eye at the output size replaces the cheap periphery's DLSS call (and its
    // enlarge); the weight of the current frame, and the jitter's sign (a diagnostic, if the picture shimmers).
    CustomOptional<bool> UpscalerFoveaPeripheryTaa { false };
    CustomOptional<float> UpscalerFoveaTaaBlend { 0.1f };
    // The TAA periphery's own size as a fraction of the eye: 1 = at the output size (as before); below, it runs smaller
    // (its history too) and the Catmull-Rom enlarge fills the eye. Live.
    CustomOptional<float> UpscalerFoveaTaaScale { 1.0f };
    CustomOptional<bool> UpscalerFoveaTaaFlipJitter { false };
    // Startup only, diagnostics: F7 records the constants of the pass writing DLSS's motion vectors, plus depth and
    // motion samples, for finding the camera matrices offline. See inputs/MotionMatrixProbe_Dx11.h.
    CustomOptional<bool> UpscalerMatrixProbe { false };
    // Startup only, any D3D11 game: DLAA+NR at Present on the last eye-sized 8/10-bit colour texture the gameplay
    // context wrote (Days Gone: its tonemap + TAA compute output). Nothing happens while NR is off.
    CustomOptional<bool> FinalImageNr { false };
    // Startup only, diagnostics: decode the motion vectors of a TAA compute pass (Days Gone) into DLSS form at the
    // pass; the F7 capture saves them and F8 shows them in place of the image. See inputs/TaaMotion_Dx11.h.
    CustomOptional<bool> TaaMotionCheck { false };
    // Startup only: DLAA in place of that TAA pass, one DLSS feature per eye (alternate-eye VR); needs FinalImageNr
    // for the depth. Starts on the game's TAA: F6 toggles game TAA / DLAA, F5 the jitter signs. See
    // inputs/TaaDlaa_Dx11.h.
    CustomOptional<bool> TaaDlaa { false };
    // UE4 renders reversed-Z.
    CustomOptional<bool> TaaDlaaDepthInverted { true };
    // Startup only: while DLAA replaces the pass, the game's own TAA dispatch is skipped (saves its GPU time; its
    // history goes stale, so switching back to game TAA may ghost for a moment). False = the game's pass still runs
    // first and DLAA overwrites its output.
    CustomOptional<bool> TaaDlaaSkipGameTaa { true };
    // Startup only, FarCry5.exe: per-eye DLAA in place of the game's TAA draw, with its decoded motion vectors
    // composed to the same eye (see inputs/Fc5Taa_Dx11.h). Starts on the game's TAA; F6 toggles.
    CustomOptional<bool> Fc5TaaDlss { false };
    // With Fc5TaaDlss: while DLAA runs, the game's view constants get our own per-eye jitter sequence (F4 toggles).
    CustomOptional<bool> Fc5TaaOwnJitter { true };
    // With Fc5TaaDlss below 100% Resolution Scale: show the DLSS upscaling output in place of the game's upscale
    // draw. false = upscaling still evaluates but the game's upscale is shown (isolation test).
    CustomOptional<bool> Fc5TaaUpscaleShow { true };
    // With Fc5TaaDlss: hold every Present to a whole number of display refreshes, so the frames reach the VR
    // compositor on a steady cadence (on DX11 in VR the game was measured free-running, and
    // free-running below refresh reprojects unevenly — the judder that goes away when a weapon is drawn). 0 = off,
    // 2 = one Present every second refresh, and so on. Fc5TaaPaceHz is the headset's refresh rate. The two numbers
    // are just the interval, Fc5TaaPaceRefreshes / Fc5TaaPaceHz seconds, so any rate can be dialled in: one Present
    // per eye means 3 / 180 (16.67 ms) is a stereo frame every 3 refreshes, and 6 / 450 (13.33 ms) is the rate FC5
    // settles at with a weapon drawn.
    // F7 frame capture: wait this many seconds before the first capture, then take CaptureRepeat of them
    // CaptureSpacingSeconds apart (a capture reads back ~700 MB on the render thread and writes for seconds,
    // so don't crowd them). 0 = capture as soon as F7 is pressed, as before. For captures that need both hands on
    // the controls: press F7, then ride.
    CustomOptional<int> CaptureDelaySeconds { 0 };
    CustomOptional<int> CaptureRepeat { 1 };
    CustomOptional<int> CaptureSpacingSeconds { 10 };
    // F7 on a TAA draw (FC5): instead of frame 0's inputs, save the TAA output and the scene depth on each of the four
    // captured Presents, with each one's Present number and time -- the offline stale-eye warp test (a same-eye
    // image warped two Presents forward, scored against the real one). ~650 MB at FC5's 3648x3744.
    CustomOptional<bool> CaptureStaleEye { false };
    // Publish the game's current-to-previous camera matrix each Present for the stale-eye OpenXR layer
    // (inputs/StaleEyeShareAbi.h; FC5 with Fc5TaaDlss). Off: nothing is published and the layer does nothing.
    // On by default: it is what makes the overlay's warp section appear, where the warp itself
    // (StaleEyeMode) starts Off and is switched on by hand. Publishing is a copy of a matrix per Present.
    CustomOptional<bool> StaleEyeCameraShare { true };
    // With StaleEyeCameraShare: also copy each Present's scene depth into shared textures for the stale-eye WARP in
    // the OpenXR layer (FC5; ~one full-size R32 copy a Present). Off: no depth is shared. On by default: the copy is
    // made only while the layer asks for it (a warp mode that reads depth is selected), never with the warp Off.
    CustomOptional<bool> StaleEyeDepthShare { true };
    // The stale-eye OpenXR layer's settings (tools/xr_observer), pushed to it once when it first appears and kept in
    // step with it by the overlay (Far Cry 5 section): mode 0 off / 1 pass / 2 depth / 3 warp; the Presents the
    // submitted eye image lags the camera (FC5: 2); the near threshold in game units (nearer = hands and weapon, left
    // unmoved; 0 off); both eyes from the layer's swapchains.
    CustomOptional<int> StaleEyeMode { 0 };
    CustomOptional<int> StaleEyePairing { 2 };
    CustomOptional<float> StaleEyeNear { 0.7f }; // just past FC5's weapon (within 0.5 units)
    CustomOptional<bool> StaleEyeBothEyes { false };
    CustomOptional<int> StaleEyeTreat { 0 };            // 0 none, 1 dim, 2 blur the stale eye
    CustomOptional<float> StaleEyeTreatStrength { 0.5f };
    CustomOptional<bool> StaleEyeTreatGated { true };   // scaled by the game's own motion
    CustomOptional<float> StaleEyeTreatFullAt { 0.03f }; // game units per Present for full strength
    CustomOptional<int> StaleEyeHead { 1 }; // 0 headset only, 1 match the fresh eye (StaleEyePairing -1 = auto lag)
    // Simpler strafing (XrObserverStaleSettingsV1): leave game turns unwarped; depth 0 per pixel / 1 single distance;
    // nearest corrected distance in metres (per pixel, 0 off); the single distance in metres.
    CustomOptional<bool> StaleEyeStrafeOnly { false };
    CustomOptional<int> StaleEyeDepthMode { 3 }; // Smooth
    CustomOptional<float> StaleEyeClamp { 0.0f };
    CustomOptional<float> StaleEyeSingle { 5.0f };
    CustomOptional<bool> StaleEyeAttachDistanceOnly { true }; // hands/weapon = nearer than StaleEyeNear, distance only
    // What is nearer than StaleEyeNear moves with the body (a vehicle's cabin: moved by the head's movement alone)
    // instead of being left where it is. Smooth / Past the player only.
    CustomOptional<bool> StaleEyeNearWithBody { false };
    // A game whose depth marks its characters in the stencil (class 1): the depth shared with the layer carries the
    // marks, and the layer leaves alone only a CHARACTER nearer than StaleEyeNear -- the player, not the ground at
    // his feet. Off: everything nearer (distance only), as in FC5.
    CustomOptional<bool> StaleEyePlayerMarks { true };
    // What the stale eye never saw behind the player: from the other eye's picture (blurred where it cannot help), or
    // stretched from the nearest world pixel (false, as before). XrObserverStaleSettingsV1::fillFromOtherEye.
    CustomOptional<bool> StaleEyeFillOtherEye { true };
    // The overlay in the headset (inputs/OverlayShare): drawn a second time for the stale-eye OpenXR layer, which
    // shows it as a flat panel in front of the eyes. Nothing happens without the layer, or with the overlay closed.
    // The panel's scale (the width in metres of 2048 pixels of the overlay: the panel is the overlay's own
    // rectangle, as wide as that makes it), its distance in metres, and how far its centre is below eye level.
    CustomOptional<bool> HeadsetOverlay { true };
    CustomOptional<float> HeadsetOverlayWidth { 1.6f };
    CustomOptional<float> HeadsetOverlayDistance { 1.5f };
    CustomOptional<float> HeadsetOverlayDown { 0.0f };
    // The layer offered to this process from <the OptiScaler folder>\xr_observer, when it lies there (no registry).
    CustomOptional<bool> StaleEyeLayerInFolder { true };
    // The layer's eye gaze through XR_EXT_eye_gaze_interaction (gaze_share.cpp); false = XR_OBSERVER_GAZE=0 for the
    // process: the layer adds no extension and no action set, the views are still handed on. Startup only.
    CustomOptional<bool> StaleEyeLayerGaze { true };
    CustomOptional<bool> StaleEyeDiagnostics { false }; // the layer's log during play + the overlay's test controls
    CustomOptional<int> Fc5TaaPaceRefreshes { 0 };
    CustomOptional<int> Fc5TaaPaceHz { 90 };

    // Framerate
    CustomOptional<float> FramerateLimit { 0.0f };

    // HDR
    CustomOptional<bool> ForceHDR { false };
    CustomOptional<bool> UseHDR10 { false };
    CustomOptional<bool> SkipColorSpace { false };

    // V-Sync
    CustomOptional<bool> OverrideVsync { false };
    CustomOptional<bool, NoDefault> ForceVsync;
    CustomOptional<UINT> VsyncInterval { 0 };

    // Old configs for compat reasons
    CustomOptional<bool, NoDefault> _DONTUSE_Fsr4ForceEnableInt8;

    bool LoadFromPath(const wchar_t* InPath);
    bool SaveIni();
    bool SaveXeFG();

    void CheckUpscalerFiles();

    std::vector<std::string> GetConfigLog();

    static Config* Instance();

  private:
    inline static Config* _config;
    inline static std::vector<std::string> _log;

    std::filesystem::path absoluteFileName;
    std::wstring fileName = L"OptiScaler.ini";

    bool Reload(std::filesystem::path iniPath);

    std::optional<std::string> readString(std::string section, std::string key, bool lowercase = false);
    std::optional<std::wstring> readWString(std::string section, std::string key, bool lowercase = false);
    std::optional<float> readFloat(std::string section, std::string key);
    std::optional<int> readInt(std::string section, std::string key);
    std::optional<uint32_t> readUInt(std::string section, std::string key);
    std::optional<bool> readBool(std::string section, std::string key);

    template <typename Enum> std::optional<Enum> readEnum(std::string section, std::string key);
};
