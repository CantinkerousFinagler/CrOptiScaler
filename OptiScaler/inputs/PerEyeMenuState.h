#pragma once

// What the overlay's per-eye DLSS section (PerEyeMenu.h) is given and what it answers. No ImGui here: the module
// (PerEyeDlss_Dx11) fills the state.

#include "../gpu_time/StageTimes.h"

#include <cstdint>

namespace PerEyeMenu
{
constexpr int kVariants = 4;
constexpr int kViews = 3;

struct Status
{
    bool twin = false;        // a DLSS feature and its twin exist (a mod may create its feature late)
    bool routing = false;     // per eye
    bool startPerEye = false; // saved: per eye from the start
    int variant = 1;
    int view = 0;
    const char* variantNames[kVariants] = {};
    const char* viewNames[kViews] = {};
    // The last whole window of Presents (0 Presents: none yet).
    uint64_t windowPresents = 0, toGame = 0, toTwin = 0, sameEye = 0, chainBreaks = 0, fitsRead = 0, fitsValid = 0;
    double fitResidual = 0.0, fitInliers = 0.0; // px, and % of the samples within 1 px
    // That window was per eye from its first Present to its last. After a switch the counts are still those of the
    // Presents before it, for up to a window's length.
    bool windowPerEye = false;
    // [Inputs] UpscalerGpuTiming and its last window.
    bool timing = false;
    StageTimes::Window times;
};

struct Answer
{
    bool toggleRouting = false;
    int startPerEye = -1; // 0 or 1 when it was changed (the simple dropdown changes it with the routing)
    int variant = -1;     // the one chosen
    int view = -1;
    int timing = -1;      // 0 or 1 when it was changed
    // What was shown (the offline test reads these).
    bool warned = false;         // per eye, but DLSS got no same-eye motion in the last window
    // Where the simple dropdown is (the offline test clicks it): the box that opens it, each of its two lines
    // (Off, On) while it is open, and whether it is open.
    float fixBoxAt[2] = {}, fixAt[2][2] = {};
    bool fixOpen = false;
};
} // namespace PerEyeMenu
