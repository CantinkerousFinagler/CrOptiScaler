#pragma once

// The overlay's section for per-eye DLSS on OptiScaler's D3D11 path (first user: a mod calling DLSS through
// OptiScaler): what the key F6 does, as a control that shows the state, and one line that says whether it
// runs (a whole run was once judged without the same-eye fix because F6 had been forgotten).
//
// The section is in two parts, to stay simple for players: Draw() is the
// simple part (one dropdown, Off / On, and the warning when it does not work); DrawJunkDrawer()
// holds what only a reader of a run needs (the saved start, the counts, object motion F4, the view F8, the cost on
// the GPU), and the caller draws it under a closed "Junk Drawer" tree.
//
// Header only and ImGui only: it is given the state and returns what was asked for, so that the offline test
// (tools/xr_observer/test/overlay_test, "section") runs this very code on a real ImGui frame. The caller
// (dlssnr/DlssNr_Menu.cpp) reads the state from PerEyeDlss_Dx11 and applies the answer.

// ImGui itself is the includer's (as for OverlayCrop.h): the game DLL and the test name its header differently.
#include "PerEyeMenuState.h"

namespace PerEyeMenu
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

inline const char* Name(const char* const* names, int count, int index)
{
    return index >= 0 && index < count && names[index] != nullptr ? names[index] : "?";
}

// A list to choose one of: the chosen index, or -1.
inline int Choose(const char* label, const char* const* names, int count, int current, float width)
{
    int chosen = -1;
    ImGui::PushItemWidth(width);
    if (ImGui::BeginCombo(label, Name(names, count, current)))
    {
        for (int i = 0; i < count; ++i)
        {
            const bool selected = i == current;
            if (ImGui::Selectable(Name(names, count, i), selected) && !selected)
                chosen = i;
            if (selected)
                ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
    }
    ImGui::PopItemWidth();
    return chosen;
}
} // namespace detail

// The simple part: per eye, per eye from the start, one line of counts.
inline Answer Draw(const Status& st, float scale)
{
    (void) scale;
    Answer answer;
    const ImVec4 yellow(1.0f, 0.8f, 0.2f, 1.0f);

    // "Shimmer fix": short and to the point for players. ONE dropdown
    // (simple dropdowns for DLSS and Warp):
    // a choice switches now AND is the saved start; the counts are in the Junk Drawer, and only a fault shows here.
    ImGui::SeparatorText("Shimmer fix");
    {
        static const char* const names[2] = { "Off", "On" };
        static const char* const tips[2] = {
            "One DLSS history shared by both eyes, with the game's own motion vectors (which point at the other "
            "eye).",
            "Each eye has a DLSS history of its own, and DLSS gets same-eye motion fitted from the game's vectors "
            "and depth. The usual choice."
        };
        const int now = st.routing ? 1 : 0;
        const ImVec2 box = ImGui::GetCursorScreenPos();
        answer.fixBoxAt[0] = box.x + ImGui::GetFrameHeight();
        answer.fixBoxAt[1] = box.y + ImGui::GetFrameHeight() * 0.5f;
        ImGui::SetNextItemWidth(ImGui::GetFontSize() * 13.0f);
        if (ImGui::BeginCombo("##pereyefix", names[now]))
        {
            answer.fixOpen = true;
            for (int i = 0; i < 2; ++i)
            {
                if (ImGui::Selectable(names[i], i == now) && i != now)
                {
                    answer.toggleRouting = true;
                    answer.startPerEye = i;
                }
                const ImVec2 lo = ImGui::GetItemRectMin(), hi = ImGui::GetItemRectMax();
                answer.fixAt[i][0] = lo.x + ImGui::GetFrameHeight();
                answer.fixAt[i][1] = (lo.y + hi.y) * 0.5f;
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("%s", tips[i]);
            }
            ImGui::EndCombo();
        }
        else if (ImGui::IsItemHovered())
            ImGui::SetTooltip("%s", tips[now]);
    }
    detail::Help("Saved: the next launch begins at it. It acts while DLSS makes the picture. Switching resets both "
                 "histories once. F6 switches too, without saving.");

    // Only the fault: per eye, DLSS was called in the last whole window, and none of its calls had same-eye motion.
    if (st.twin && st.windowPresents != 0 && st.routing && st.windowPerEye && st.sameEye == 0 &&
        st.toGame + st.toTwin != 0)
    {
        answer.warned = true;
        ImGui::TextColored(yellow, "Per eye, but NO same-eye motion reached DLSS (%llu history resets)",
                           (unsigned long long) st.chainBreaks);
    }
    return answer;
}

// The Junk Drawer: what a reader of a run needs and a player does not. Drawn by the caller under a closed tree;
// its answers are added to `answer`.
inline void DrawJunkDrawer(const Status& st, float scale, Answer& answer)
{
    const ImVec4 yellow(1.0f, 0.8f, 0.2f, 1.0f);

    bool start = st.startPerEye;
    if (ImGui::Checkbox("Shimmer fix on from the start", &start) && answer.startPerEye < 0)
        answer.startPerEye = start ? 1 : 0;
    detail::Help("Saved with the settings; the simple dropdown sets it too. On: the game starts per eye. Off: every "
                 "start is shared until F6 or the dropdown.");
    if (!st.twin)
        ImGui::Text("No DLSS feature yet: it is made when DLSS first makes the picture");
    else if (st.windowPresents == 0)
        ImGui::Text("DLSS feature and its twin made; nothing counted yet");
    else if (st.routing && !st.windowPerEye)
        ImGui::Text("Per eye since a switch: the counts come within 7 s");
    else if (st.routing)
        ImGui::Text("Per eye: same-eye motion on %llu of %llu DLSS calls, fits valid %llu of %llu",
                    (unsigned long long) st.sameEye, (unsigned long long) (st.toGame + st.toTwin),
                    (unsigned long long) st.fitsValid, (unsigned long long) st.fitsRead);
    else
        ImGui::Text("Shared: one DLSS history for both eyes, %llu DLSS calls in the last %llu frames",
                    (unsigned long long) (st.toGame + st.toTwin), (unsigned long long) st.windowPresents);

    const int variant = detail::Choose("Object motion", st.variantNames, kVariants, st.variant, 330.0f * scale);
    if (variant >= 0)
        answer.variant = variant;
    detail::Help("Per eye only: how the motion of things that move by themselves is carried into the same-eye "
                 "motion. The second is the one chosen in play.");
    const int view = detail::Choose("View", st.viewNames, kViews, st.view, 330.0f * scale);
    if (view >= 0)
        answer.view = view;
    detail::Help("A diagnostic picture in place of the game's: the motion of objects alone, or the motion DLSS is "
                 "given. Off for play.");
    if (st.twin && st.windowPresents != 0)
    {
        ImGui::Text("Last %llu Presents: %llu DLSS calls to the game's feature, %llu to the twin",
                    (unsigned long long) st.windowPresents, (unsigned long long) st.toGame,
                    (unsigned long long) st.toTwin);
        if (st.routing && st.windowPerEye && st.sameEye != 0)
            ImGui::Text("Same-eye motion on %llu of them; fits valid %llu of %llu, %.3f px, %.1f%% within 1 px; %llu "
                        "history resets",
                        (unsigned long long) st.sameEye, (unsigned long long) st.fitsValid,
                        (unsigned long long) st.fitsRead, st.fitResidual, st.fitInliers,
                        (unsigned long long) st.chainBreaks);
    }

    ImGui::Spacing();
    if (st.timing)
    {
        if (st.times.presents == 0)
            ImGui::Text("Measuring: the first numbers come after 600 Presents");
        else
        {
            double gpu = 0.0;
            for (int i = 0; i < StageTimes::Count; ++i)
            {
                const StageTimes::Line& line = st.times.lines[i];
                const std::string text = StageTimes::Text(i, line);
                if (line.samples != 0 && !line.credible)
                    ImGui::TextColored(yellow, "%s", text.c_str());
                else
                    ImGui::TextUnformatted(text.c_str());
                if (line.credible && i != StageTimes::RenderThread)
                    gpu += line.meanMs;
            }
            ImGui::Text("On the GPU per eye, the credible stages together: %.2f ms (a frame at 90 has 11.1)", gpu);
        }
        if (st.times.videoBudgetMb > 0.0)
        {
            const bool tight = st.times.videoUsedMb > 0.9 * st.times.videoBudgetMb;
            if (tight)
                ImGui::TextColored(yellow, "Video memory: %.1f GB of a budget of %.1f GB -- nearly FULL",
                                   st.times.videoUsedMb / 1024.0, st.times.videoBudgetMb / 1024.0);
            else
                ImGui::Text("Video memory: %.1f GB of a budget of %.1f GB", st.times.videoUsedMb / 1024.0,
                            st.times.videoBudgetMb / 1024.0);
        }
    }
}
} // namespace PerEyeMenu
