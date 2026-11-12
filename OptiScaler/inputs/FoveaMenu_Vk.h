#pragma once

// The overlay's "Foveated SR" block for the Vulkan crop (FoveaCrop_Vk): the
// crop on / off, the outline, the preset, the box's size, the edge blend, and what the crop is
// doing. Written because the key F4 is one Vulkan game's own (pressed twice it left VR): on Vulkan the crop is
// switched HERE, and F4 does nothing (menu_common.cpp).
//
// Each control on a line of its own, choices as dropdowns: the headset's panel cuts a row of controls off at the
// right, and the offline test cannot see that.
//
// Header only and ImGui only: it is given the state and returns what was asked for, so that the offline test
// (tools/xr_observer/test/overlay_test, "fovea") runs this very code on a real ImGui frame. The caller
// (dlssnr/DlssNr_Menu.cpp) reads the state from FoveaCrop_Vk::GetStatus() and hands the answer to
// FoveaCrop_Vk::Apply().

// ImGui itself is the includer's (as for PerEyeMenu.h): the game DLL and the test name its header differently.
#include "FoveaMenuState_Vk.h"

namespace FoveaMenuVk
{
namespace detail
{
inline void Help(const char* tip)
{
    ImGui::SameLine();
    ImGui::TextDisabled("(?)");
    if (ImGui::IsItemHovered())
    {
        ImGui::BeginTooltip();
        ImGui::PushTextWrapPos(ImGui::GetFontSize() * 40.0f);
        ImGui::TextUnformatted(tip);
        ImGui::PopTextWrapPos();
        ImGui::EndTooltip();
    }
}

inline void ItemCentre(float at[2])
{
    const ImVec2 lo = ImGui::GetItemRectMin(), hi = ImGui::GetItemRectMax();
    at[0] = lo.x + ImGui::GetFrameHeight() * 0.5f;
    at[1] = (lo.y + hi.y) * 0.5f;
}
} // namespace detail

// THE SIMPLE PART (a clean UI: only the options that are relevant and work):
// what a player chooses -- the crop, its preset and size, the periphery and its kind -- and a warning only
// when something of ours is not doing its work. The A/B switches, the TAA's tuning and the counters are DrawJunk's,
// drawn by the caller under a closed "Junk Drawer" tree.
inline Answer Draw(const Status& st, float scale)
{
    Answer answer;
    const ImVec4 yellow(1.0f, 0.8f, 0.2f, 1.0f);

    ImGui::SeparatorText("Foveated SR");
    bool on = st.on;
    if (ImGui::Checkbox("Crop on", &on))
        answer.toggleCrop = true;
    detail::ItemCentre(answer.cropBoxAt);
    detail::Help("A second DLSS pass over a box at the centre of each eye at its own preset, blended over the rest of "
                 "the picture. Its features are made the first time it is turned on (a short hitch). The saving comes "
                 "with 'Cheap periphery' below.");
    bool startOn = st.startOn;
    if (ImGui::Checkbox("Crop on from the launch", &startOn))
        answer.startOn = startOn ? 1 : 0;
    detail::Help("Saved with the settings: the next launch begins with the crop on.");
    bool outline = st.outline;
    if (ImGui::Checkbox("Outline", &outline))
        answer.toggleOutline = true;
    detail::Help("Draws the crop's edge in magenta. Only visible while the crop is on.");

    {
        const int now = PresetIndex(st.preset);
        ImGui::SetNextItemWidth(ImGui::GetFontSize() * 8.0f);
        const ImVec2 box = ImGui::GetCursorScreenPos();
        answer.presetBoxAt[0] = box.x + ImGui::GetFrameHeight();
        answer.presetBoxAt[1] = box.y + ImGui::GetFrameHeight() * 0.5f;
        if (ImGui::BeginCombo("Crop preset", kPresetNames[now]))
        {
            answer.presetOpen = true;
            for (int i = 0; i < kPresets; ++i)
            {
                if (ImGui::Selectable(kPresetNames[i], i == now) && i != now)
                    answer.preset = kPresetValues[i];
                const ImVec2 lo = ImGui::GetItemRectMin(), hi = ImGui::GetItemRectMax();
                answer.presetAt[i][0] = lo.x + ImGui::GetFrameHeight();
                answer.presetAt[i][1] = (lo.y + hi.y) * 0.5f;
            }
            ImGui::EndCombo();
        }
        detail::Help("The crop's own DLSS preset; the game's full pass keeps its own (the DLSS settings' Override "
                     "Preset). A change remakes the crop's features (the crop is missing for a few frames). Saved.");
    }

    ImGui::PushItemWidth(220.0f * scale);
    // Applied when a slider is let go: the crop's features are remade at the new box.
    static float widthEdit = 0.4f, heightEdit = 0.4f;
    static bool widthEditing = false, heightEditing = false;
    if (!widthEditing)
        widthEdit = st.width;
    if (!heightEditing)
        heightEdit = st.height;
    ImGui::SliderFloat("Width", &widthEdit, 0.2f, 0.8f, "%.2f of the eye");
    widthEditing = ImGui::IsItemActive();
    const bool widthDone = ImGui::IsItemDeactivatedAfterEdit();
    detail::Help("Width of the crop as a fraction of the eye. Applied when you let go: the crop's features are "
                 "remade at the new size (a few frames without the crop). Saved.");
    ImGui::SliderFloat("Height", &heightEdit, 0.2f, 0.8f, "%.2f of the eye");
    heightEditing = ImGui::IsItemActive();
    const bool heightDone = ImGui::IsItemDeactivatedAfterEdit();
    detail::Help("Height of the crop as a fraction of the eye, independent of the width. Saved.");
    if (widthDone || heightDone)
    {
        answer.width = widthEdit;
        answer.height = heightEdit;
    }
    int feather = st.feather;
    if (ImGui::SliderInt("Edge blend", &feather, 0, 128, "%d px"))
        answer.feather = feather;
    detail::Help("Width of the blend from the crop into the rest of the picture at the crop's edge, in output pixels. "
                 "Takes effect at once. Saved.");
    ImGui::PopItemWidth();

    // The cheap periphery: the saving.
    bool periphery = st.periphery;
    if (ImGui::Checkbox("Cheap periphery", &periphery))
        answer.periphery = periphery ? 1 : 0;
    detail::Help("With the crop on: outside it, DLSS runs at the render size (no upscaling) and is enlarged to the "
                 "eye, in place of the game's full-size pass -- cheaper, a little softer away from the centre. Its "
                 "features are made the first time (a short hitch). Saved.");
    {
        const int now = PresetIndex(st.peripheryPreset);
        ImGui::SetNextItemWidth(ImGui::GetFontSize() * 8.0f);
        if (ImGui::BeginCombo("Periphery preset", kPresetNames[now]))
        {
            for (int i = 0; i < kPresets; ++i)
                if (ImGui::Selectable(kPresetNames[i], i == now) && i != now)
                    answer.peripheryPreset = kPresetValues[i];
            ImGui::EndCombo();
        }
        detail::Help("The cheap periphery's own DLSS preset (E is the cheapest). A change remakes its features. "
                     "Saved.");
    }
    // Our own TAA in the periphery's place.
    bool peripheryTaa = st.peripheryTaa;
    if (ImGui::Checkbox("Own TAA periphery", &peripheryTaa))
        answer.peripheryTaa = peripheryTaa ? 1 : 0;
    detail::Help("With the crop and Cheap periphery on: outside the crop our own TAA runs instead of any DLSS call -- "
                 "a little cheaper still, softer, and it can shimmer where DLSS does not. Saved.");

    // Only what is NOT doing its work shows here; the counts are in the Junk Drawer.
    if (st.taaFailed)
        ImGui::TextColored(yellow, "The TAA periphery failed; the DLSS periphery runs (see OptiScaler.log).");
    if (st.peripheryFailed)
        ImGui::TextColored(yellow, "The cheap periphery failed; the full pass runs this session (see OptiScaler.log).");
    if (st.failed)
        ImGui::TextColored(yellow, "The crop failed and stays off this session; see OptiScaler.log.");
    else if (st.cropOutW > 0)
        ImGui::Text("Crop: render %ux%u -> %ux%u at (%u, %u) of %ux%u, %.0f%% of the eye%s", st.cropW, st.cropH,
                    st.cropOutW, st.cropOutH, st.cropX, st.cropY, st.eyeW, st.eyeH,
                    st.eyeW != 0 && st.eyeH != 0 ? 100.0 * st.cropOutW * st.cropOutH / ((double) st.eyeW * st.eyeH)
                                                 : 0.0,
                    st.exact ? ", exact" : "");
    else if (st.on)
        ImGui::Text("Crop: no box planned yet (it comes with the next DLSS call)");
    return answer;
}

// THE JUNK DRAWER'S PART: the A/B switches of the crop's inputs, the TAA periphery's tuning, and what a reader of a
// run needs (the frame rate, the GPU times, the windows' counts). Drawn by the caller under a closed tree; what is
// asked for goes into the same Answer as Draw's.
inline void DrawJunk(const Status& st, float scale, Answer& answer)
{
    const ImVec4 yellow(1.0f, 0.8f, 0.2f, 1.0f);
    ImGui::SeparatorText("Foveated SR");
    // The crop's inputs.
    bool copyInputs = st.copyInputs;
    if (ImGui::Checkbox("Crop reads copies", &copyInputs))
        answer.copyInputs = copyInputs ? 1 : 0;
    detail::Help("On: the crop's colour, motion, depth and bias mask are copied at the box into small images first, "
                 "so its DLSS reads them like any ordinary call. Off: it reads the game's full images through DLSS's "
                 "input offsets (subrects) -- presets K and J shimmer that way. Switching restarts the crop's "
                 "history. Saved.");
    {
        static const char* const maskNames[3] = { "Left out (test)", "Copied at the box",
                                                  "As the game hands it (wrong place, test)" };
        const int now = st.cropMask >= 0 && st.cropMask < 3 ? st.cropMask : 1;
        ImGui::SetNextItemWidth(ImGui::GetFontSize() * 16.0f);
        if (ImGui::BeginCombo("Crop's bias mask", maskNames[now]))
        {
            for (int i = 0; i < 3; ++i)
                if (ImGui::Selectable(maskNames[i], i == now) && i != now)
                    answer.cropMask = i;
            ImGui::EndCombo();
        }
        detail::Help("Some games hand DLSS a full-frame mask of where to trust the current frame over its history "
                     "(one Vulkan game does). Copied at the box: the crop gets its own part of it. As handed: "
                     "the crop reads it at the top left of the frame (the fault of before). Left out: none. Saved.");
    }
    // The own TAA periphery's tuning.
    bool flip = st.taaFlipJitter;
    if (ImGui::Checkbox("Flip jitter", &flip))
        answer.taaFlipJitter = flip ? 1 : 0;
    detail::Help("Diagnostic: if the TAA periphery shimmers or crawls, try this. The right sign is the one that looks "
                 "stable. Saved.");
    ImGui::PushItemWidth(220.0f * scale);
    static float blendEdit = 0.1f, scaleEdit = 1.0f;
    static bool blendEditing = false, scaleEditing = false;
    if (!blendEditing)
        blendEdit = st.taaBlend;
    if (!scaleEditing)
        scaleEdit = st.taaScale;
    ImGui::SliderFloat("TAA blend", &blendEdit, 0.03f, 0.5f, "%.2f");
    blendEditing = ImGui::IsItemActive();
    if (ImGui::IsItemDeactivatedAfterEdit())
        answer.taaBlend = blendEdit;
    detail::Help("Weight of the new frame in the TAA periphery. Lower: smoother and steadier, more trailing on moving "
                 "things. Higher: sharper response, more shimmer and noise. Ctrl+click to type. Saved.");
    ImGui::SliderFloat("TAA resolution", &scaleEdit, 0.4f, 1.0f, "%.2f");
    scaleEditing = ImGui::IsItemActive();
    if (ImGui::IsItemDeactivatedAfterEdit())
        answer.taaScale = scaleEdit;
    detail::Help("Size the TAA periphery runs at, as a fraction of the eye; the result is enlarged to fill the eye. "
                 "1.00 = the eye size. About the game's render ratio keeps its detail and costs less; lower is softer. "
                 "Each change restarts its history. Saved.");
    ImGui::PopItemWidth();

    // What a reader of a run needs.
    if (st.peripheryTaa && !st.taaFailed && st.windowEvals != 0)
        ImGui::TextDisabled("TAA periphery, last window: %llu drawn (%llu history resets), %llu skipped to DLSS",
                            (unsigned long long) st.taaEvals, (unsigned long long) st.taaResets,
                            (unsigned long long) st.taaSkips);
    if (st.periphery && !st.peripheryFailed && st.windowEvals != 0)
        ImGui::TextDisabled("Periphery, last window: %llu drawn in place of the full pass, %llu fell back; %u feature(s)",
                            (unsigned long long) st.peripheryDrawn, (unsigned long long) st.peripheryFallbacks,
                            st.peripheries);
    if (st.presentsPerSec > 0.0)
        ImGui::Text("%.1f Presents/s over the last 600 (%.2f DLSS calls per Present)", st.presentsPerSec,
                    st.evalsPerPresent);
    if (st.gpuFrames > 0)
        ImGui::Text("GPU per frame: the game's full DLSS %.2f ms on %u; ours %.2f ms (Present interval %.2f ms)",
                    st.gpuGameMs, st.gpuGameRuns, st.gpuOursMs, st.gpuFrameMs);
    else if (!st.gpuTiming)
        ImGui::TextDisabled("GPU times: off ([Inputs] UpscalerGpuTiming = true to measure the passes)");
    detail::Help("GPU timestamps in the game's command buffer: the game's full DLSS pass where it ran, and our passes "
                 "(periphery or TAA, copies, crop, composite). Under the headset's refresh cap a saving shows here, "
                 "not in the Presents/s.");
    if (st.making > 0)
        ImGui::TextColored(yellow, "%u feature(s) being made on the worker; the full pass runs meanwhile", st.making);
    else if (!st.worker)
        ImGui::TextDisabled("No create worker: features are made on the game's thread (a first make may stall VR)");
    if (st.windowEvals != 0)
        ImGui::TextDisabled("Last window: %llu drawn (%llu with copied inputs), %llu skipped, %llu failed; %u crop "
                            "feature(s); the eye from %s",
                            (unsigned long long) st.drawn, (unsigned long long) st.gathered,
                            (unsigned long long) st.skipped, (unsigned long long) st.failedEvals, st.crops,
                            st.eyeByParity ? "the Present's parity" : "the routed feature");
}
} // namespace FoveaMenuVk
