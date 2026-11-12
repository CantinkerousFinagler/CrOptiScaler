#pragma once

// What the overlay's "Foveated SR" block on the Vulkan path (FoveaMenu_Vk.h) is given and what it answers. No ImGui
// here: the module (FoveaCrop_Vk) fills the state and applies the answer.

#include <cstdint>

namespace FoveaMenuVk
{
constexpr int kPresets = 6;
// The DLSS presets a crop can be asked for, as the D3D12 block offers them (dlssnr/DlssNr_Menu.cpp).
constexpr const char* const kPresetNames[kPresets] = { "Default", "E", "J", "K", "L", "M" };
constexpr int kPresetValues[kPresets] = { 0, 5, 10, 11, 12, 13 };

inline int PresetIndex(int value)
{
    for (int i = 0; i < kPresets; ++i)
        if (kPresetValues[i] == value)
            return i;
    return 0;
}

struct Status
{
    bool on = false;      // the crop is drawn (switched on, and it has not failed)
    bool outline = false; // its edge drawn in magenta
    bool failed = false;  // it failed once this session and stays off (the log says why)
    bool startOn = false; // saved: the crop is on from the launch ([Inputs] UpscalerFoveaStart)
    int preset = 13;      // the crop's DLSS preset ([Inputs] UpscalerFoveaPreset)
    float width = 0.4f, height = 0.4f; // the box as fractions of the eye (the height resolved: 0 = the width)
    int feather = 32;                  // the blend at the edge, px
    // The box in use (cropOutW == 0: none planned yet).
    uint32_t cropW = 0, cropH = 0, cropOutW = 0, cropOutH = 0, cropX = 0, cropY = 0, eyeW = 0, eyeH = 0;
    bool exact = false;
    // The last whole window of evaluates (windowEvals == 0: none yet), and the crop features alive.
    uint64_t windowEvals = 0, drawn = 0, skipped = 0, failedEvals = 0;
    uint32_t crops = 0;
    bool eyeByParity = false; // shared routing: the Present's parity names the eye
    // The cheap periphery ([Inputs] UpscalerFoveaCheapPeriphery, UpscalerFoveaPeripheryPreset) and its last window.
    bool periphery = false, peripheryFailed = false;
    int peripheryPreset = 5;
    uint64_t peripheryDrawn = 0, peripheryFallbacks = 0;
    uint32_t peripheries = 0;
    // The last window of Presents (0 = none yet), and the create worker.
    double presentsPerSec = 0.0, evalsPerPresent = 0.0;
    // The GPU times of the last window ([Inputs] UpscalerGpuTiming; 0 frames = nothing measured).
    bool gpuTiming = false;
    uint32_t gpuFrames = 0, gpuGameRuns = 0;
    double gpuFrameMs = 0.0, gpuGameMs = 0.0, gpuOursMs = 0.0;
    bool worker = false;  // features of our own are made off the game's thread
    uint32_t making = 0;  // being made on the worker now
    // The crop's inputs: copied at the box (UpscalerFoveaCopyInputs) and the bias mask's mode (UpscalerFoveaCropMask:
    // 0 left out, 1 copied at the box, 2 as the game hands it); evaluates copied in the last window.
    bool copyInputs = true;
    int cropMask = 1;
    uint64_t gathered = 0;
    // The own TAA periphery (UpscalerFoveaPeripheryTaa, TaaBlend, TaaScale, TaaFlipJitter) and its last window.
    bool peripheryTaa = false, taaFlipJitter = false, taaFailed = false;
    float taaBlend = 0.1f, taaScale = 1.0f;
    uint64_t taaEvals = 0, taaResets = 0, taaSkips = 0;
};

struct Answer
{
    bool toggleCrop = false;
    bool toggleOutline = false;
    int startOn = -1;         // 0 or 1 when changed
    int preset = -1;          // the preset VALUE chosen (0, 5, 10, 11, 12, 13), or -1
    float width = -1.0f;      // both set when a size slider was let go
    float height = -1.0f;
    int feather = -1;         // px when changed
    int periphery = -1;       // 0 or 1 when changed
    int peripheryPreset = -1; // the preset VALUE chosen, or -1
    int copyInputs = -1;      // 0 or 1 when changed
    int cropMask = -1;        // 0, 1 or 2 when changed
    int peripheryTaa = -1;    // 0 or 1 when changed
    int taaFlipJitter = -1;   // 0 or 1 when changed
    float taaBlend = -1.0f;   // when let go
    float taaScale = -1.0f;   // when let go
    // Where things are (the offline test clicks them): the crop checkbox, the preset dropdown's box, its lines
    // while open, and whether it is open.
    float cropBoxAt[2] = {}, presetBoxAt[2] = {}, presetAt[kPresets][2] = {};
    bool presetOpen = false;
};
} // namespace FoveaMenuVk
