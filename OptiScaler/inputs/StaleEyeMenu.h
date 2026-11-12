#pragma once

// The overlay's controls for the stale-eye OpenXR layer (R.E.A.L. VR's alternate-eye rendering), in two parts (the
// complicated settings in the Junk Drawer, simple dropdowns for DLSS and Warp in the open):
//   DrawSimple / DrawStatusSimple   ONE dropdown, Off / Warp, and with Warp the player's distance (a
//                                   slider for the distance the warp works at, since the
//                                   camera's distance from the character varies); the line that says how much
//                                   is warped, the warnings
//   DrawJunk / DrawStatusJunk       everything else: the warp's depth, near things, behind the player, Diagnostics
//                                   and its test controls, the counters
// A game's section may draw the second part under its closed "Junk Drawer" tree; Far Cry 5's draws both in place.
//
// Header only and ImGui only, as PerEyeMenu.h: it is given the layer's settings and status and
// changes the settings it is handed; the caller (dlssnr/DlssNr_Menu.cpp) reads them from the layer, hands them back
// when `changed` is set, and keeps the ini in step. So the offline test (tools/xr_observer/test/overlay_test,
// "stale") runs this very code on a real ImGui frame.

// ImGui itself is the includer's: the game DLL and the test name its header differently.
#include "StaleEyeShareAbi.h"

#include <algorithm>
#include <cmath>

namespace StaleEyeMenu
{
constexpr int kSimpleModes = 2; // Off, Warp

struct Answer
{
    bool changed = false; // a setting of the layer's was changed: the caller hands the settings to the layer
    int playerMarks = -1; // "The player is": 0 or 1 when it was changed
    // What was shown and where (the offline test reads these).
    bool skipShown = false;     // the layer's reason for not processing the stale image as selected
    bool nearOffWarned = false; // the player's distance is off: the hands and weapon move with the world
    float modeBoxAt[2] = {}, modeAt[kSimpleModes][2] = {};
    bool modeOpen = false;
    bool distanceShown = false;   // the player's distance was drawn (Warp, and a depth mode that has a player)
    float distancePlusAt[2] = {}; // its + button
};

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

// A slider with - / + buttons in front (the menu's StepSlider: dragging is hard in the headset).
inline bool StepSlider(const char* label, float* v, float lo, float hi, float step, const char* format,
                       float* plusAt = nullptr)
{
    bool changed = false;
    ImGui::PushID(label);
    ImGui::PushItemFlag(ImGuiItemFlags_ButtonRepeat, true);
    if (ImGui::Button("-"))
    {
        *v = (std::max)(lo, std::round((*v - step) / step) * step);
        changed = true;
    }
    ImGui::SameLine(0.0f, 4.0f);
    if (ImGui::Button("+"))
    {
        *v = (std::min)(hi, std::round((*v + step) / step) * step);
        changed = true;
    }
    if (plusAt != nullptr)
    {
        const ImVec2 lo2 = ImGui::GetItemRectMin(), hi2 = ImGui::GetItemRectMax();
        plusAt[0] = (lo2.x + hi2.x) * 0.5f;
        plusAt[1] = (lo2.y + hi2.y) * 0.5f;
    }
    ImGui::PopItemFlag();
    ImGui::PopID();
    ImGui::SameLine(0.0f, 6.0f);
    changed |= ImGui::SliderFloat(label, v, lo, hi, format);
    return changed;
}
} // namespace detail

// The simple part: Off or Warp, and with Warp the player's distance. A mode the dropdown does not offer (Pass,
// Depth: tests) is shown as such. `marksCharacters`: the game marks its characters in the stencil; the words of the
// distance's help are then a character's, not hands and a weapon.
inline void DrawSimple(XrObserverStaleSettingsV1& s, Answer& a, bool marksCharacters)
{
    static const char* const names[kSimpleModes] = { "Off", "Warp" };
    static const char* const tips[kSimpleModes] = {
        "The stale eye's picture is shown as it is.",
        "The stale eye's picture is moved by the game's own motion since it was drawn (strafe, walk, stick), with "
        "its depth."
    };
    const int mode = std::clamp(s.mode, 0, 3);
    const int now = mode == 0 ? 0 : mode == 3 ? 1 : -1;
    const ImVec2 box = ImGui::GetCursorScreenPos();
    a.modeBoxAt[0] = box.x + ImGui::GetFrameHeight();
    a.modeBoxAt[1] = box.y + ImGui::GetFrameHeight() * 0.5f;
    ImGui::SetNextItemWidth(ImGui::GetFontSize() * 13.0f);
    if (ImGui::BeginCombo("##staleeyemode", now >= 0 ? names[now] : "a test state (Junk Drawer)"))
    {
        a.modeOpen = true;
        for (int i = 0; i < kSimpleModes; ++i)
        {
            if (ImGui::Selectable(names[i], i == now) && i != now)
            {
                s.mode = i == 1 ? 3 : 0;
                a.changed = true;
            }
            const ImVec2 lo = ImGui::GetItemRectMin(), hi = ImGui::GetItemRectMax();
            a.modeAt[i][0] = lo.x + ImGui::GetFrameHeight();
            a.modeAt[i][1] = (lo.y + hi.y) * 0.5f;
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("%s", tips[i]);
        }
        ImGui::EndCombo();
    }
    else if (now >= 0 && ImGui::IsItemHovered())
        ImGui::SetTooltip("%s", tips[now]);
    detail::Help("Saved: the next launch begins at it. Under alternate-eye rendering one eye shows an 11 ms old "
                 "picture each frame. Warp moves it by the game's own motion since then (strafe, walk, stick), with "
                 "its depth. It needs the game to hold the headset's refresh rate: lower R.E.A.L. VR's resolution "
                 "if the whole picture judders.");

    // The player's distance: with Warp, in every depth mode that leaves a player alone (Single distance has none).
    if (s.mode != 3 || s.depthMode == 1)
        return;
    a.distanceShown = true;
    float playerDistance = s.nearThreshold;
    ImGui::PushItemWidth(ImGui::GetFontSize() * 13.0f);
    if (detail::StepSlider("Leave alone nearer than", &playerDistance, 0.0f, 8.0f, 0.1f,
                           playerDistance <= 0.0f ? "off" : "%.1f units", a.distancePlusAt))
    {
        s.nearThreshold = playerDistance;
        a.changed = true;
    }
    ImGui::PopItemWidth();
    if (marksCharacters)
        detail::Help("Saved. Your character moves with the camera: a character nearer than this is left exactly as "
                     "he is, the world beyond is moved. Keep it just past him: 5 with the game's own camera (it "
                     "pulls back when you sprint or jump). If your VR camera is set nearer to the character, "
                     "it can be lower: lower it while enemies near you judder, raise it when your character doubles "
                     "as you move.");
    else
        detail::Help("Saved. Your hands and weapon move with the camera: everything nearer than this is left "
                     "exactly as it was, the world beyond is moved. Keep it just past the weapon: 0.7, and raise it "
                     "only if the weapon doubles. (1 unit is about 1.1 m in FC5.) Off: the gun moves with the "
                     "world.");
}

// Under the simple part, while the layer does anything: how much is warped, and what is not as it should be.
inline void DrawStatusSimple(const XrObserverStaleSettingsV1& s, const XrObserverStaleStatusV1& st, Answer& a)
{
    if (s.mode == 0)
        return;
    const ImVec4 yellow(1.0f, 0.8f, 0.2f, 1.0f);
    ImGui::Text("%.0f of %.0f stale images warped per second", st.warpedPerSecond, st.processedPerSecond);
    if (s.mode == 3 && s.depthMode >= 2 && s.nearThreshold <= 0.0f)
    {
        a.nearOffWarned = true;
        ImGui::TextColored(yellow, "Hands/weapon distance is off: the gun moves with the world");
    }
    if (st.lastSkip[0] != 0)
    {
        a.skipShown = true;
        ImGui::TextColored(yellow, "Skipping: %s", st.lastSkip);
    }
}

// The Junk Drawer's part: every other setting of the layer. `marksCharacters`: the game marks its characters ("The player
// is"), `playerMarks` what is chosen there now.
inline void DrawJunk(XrObserverStaleSettingsV1& s, Answer& a, bool marksCharacters, bool playerMarks)
{
    using detail::Help;
    using detail::StepSlider;
    // Everyday: the two depths worth using, the player's distance. The rest are the experiments that led here;
    // they show with Diagnostics ticked, and a value one of them left set is shown whatever the tick
    // says, so nothing acts unseen.
    const bool tests = s.diagnostics != 0;
    int mode = std::clamp(s.mode, 0, 3);
    if (tests || mode == 1 || mode == 2)
    {
        static const char* modeNames[] = { "Off", "Pass (copy)", "Depth (edges)", "Warp" };
        if (ImGui::Combo("Stale eye, all four modes", &mode, modeNames, 4))
        {
            s.mode = mode;
            a.changed = true;
        }
        Help("Pass: the stale image copied through the layer unchanged. Depth: the edges of the game's depth drawn "
             "over it. Both are tests.");
    }
    int depthMode = std::clamp(s.depthMode, 0, 3);
    if (tests || depthMode == 1 || depthMode == 2)
    {
        static const char* depthNames[] = { "Per pixel (game depth)", "Single distance (no depth)",
                                            "Past the player (auto distance)", "Smooth (no tearing)" };
        if (ImGui::Combo("Warp depth", &depthMode, depthNames, 4))
        {
            s.depthMode = depthMode;
            a.changed = true;
        }
        Help("Per pixel: every point moved by its own distance (exact, but holes and tearing beside "
             "near things). Single distance: the whole picture moved as if everything were at one "
             "distance -- no holes, no tearing, no depth needed; exact at that distance, farther "
             "things move too much (doubles the view at a near distance). Past the player: the "
             "player (nearer than the hands/weapon distance) left alone, the world moved as one plane "
             "at the middle distance of what is in front of you, measured every frame. Smooth: "
             "the player left alone, the world moved by its own distance, but read coarsely and "
             "blurred -- the ground, walls and far scenery each move about right, nothing tears "
             "and nothing is left empty; beside near things the picture stretches a little.");
        if (depthMode == 0)
        {
            float clampM = s.clampMetres;
            if (StepSlider("Nearest corrected distance", &clampM, 0.0f, 5.0f, 0.1f, clampM <= 0.0f ? "off" : "%.1f m"))
            {
                s.clampMetres = clampM;
                a.changed = true;
            }
            Help("Anything nearer is corrected as if it were at this distance: far scenery stays "
                 "exact, near objects are under-corrected instead of tearing, the weapon moves a "
                 "little instead of doubling. Off: every distance exact. Try 1.5.");
        }
        else if (depthMode == 1)
        {
            float singleM = s.singleMetres;
            if (StepSlider("Single distance", &singleM, 0.5f, 30.0f, 0.5f, "%.1f m"))
            {
                s.singleMetres = singleM;
                a.changed = true;
            }
            Help("The distance the whole picture is corrected for. Things at this distance are "
                 "exact while strafing; nearer ones still judder a little, farther ones move a little "
                 "too much. Try 5.");
        }
    }
    else
    {
        static const char* simpleDepths[] = { "Smooth (no tearing)", "Per pixel (exact, can tear)" };
        int simple = depthMode == 0 ? 1 : 0;
        if (ImGui::Combo("Warp depth", &simple, simpleDepths, 2))
        {
            s.depthMode = simple == 1 ? 0 : 3;
            a.changed = true;
        }
        Help("Smooth: the world moved by its own distance, read coarsely and blurred -- nothing "
             "tears and nothing is left empty; beside near things the picture stretches a "
             "little. Per pixel: every point moved by its own distance -- exact, but holes and "
             "tearing beside near things.");
    }
    if (s.depthMode >= 2)
    {
        static const char* nearNames[] = { "Left alone (on foot)", "Moved with the body (vehicles)" };
        int nearWith = s.nearWithBody != 0 ? 1 : 0;
        if (ImGui::Combo("Near things", &nearWith, nearNames, 2))
        {
            s.nearWithBody = nearWith;
            a.changed = true;
        }
        Help("What is nearer than \"Leave alone nearer than\". Left alone: it stays exactly where it was in "
             "the picture. Moved with the body: it is moved by your head's own movement and by "
             "nothing else -- for a vehicle's cabin, which doubles when you look around if it is "
             "left alone. In a vehicle, raise that distance until it is past the whole "
             "cabin (about 2). Needs Head movement = Match the fresh eye (the default).");
        static const char* behindNames[] = { "Stretched (as before)", "From the other eye" };
        int behind = s.fillFromOtherEye != 0 ? 1 : 0;
        if (ImGui::Combo("Behind the player", &behind, behindNames, 2))
        {
            s.fillFromOtherEye = behind;
            a.changed = true;
        }
        Help("What the stale eye never saw because the player stood in front of it: a stick turn "
             "uncovers it beside the player. From the other eye: taken from the other eye's "
             "picture of this frame (exact for things far away), and blurred where the other eye "
             "sees the player there too. Stretched: the nearest background pulled across, which "
             "smears. In the colours view: yellow = from the other eye, blue = stretched or "
             "blurred.");
        // A game that marks its characters in the stencil: the depth shared with the layer carries those
        // marks, and the layer then leaves alone only a CHARACTER nearer than the distance -- not the ground at the
        // player's feet (a seam on the ground at 2.5 and more).
        if (marksCharacters)
        {
            static const char* playerNames[] = { "Distance only (everything near)", "The game's characters (marks)" };
            int byMarks = playerMarks ? 1 : 0;
            if (ImGui::Combo("The player is", &byMarks, playerNames, 2))
                a.playerMarks = byMarks;
            Help("What \"Leave alone nearer than\" takes for the player. The game's characters: only "
                 "a character (the player, his sword, or an enemy) nearer than the distance; the "
                 "ground and walls as near are moved with the world. Raise the distance past the "
                 "player: about 5 (the camera pulls back when you sprint or jump). Distance only: everything nearer, "
                 "as in Far Cry 5.");
        }
    }
    bool diagnostics = s.diagnostics != 0;
    if (ImGui::Checkbox("Diagnostics (log + test controls)", &diagnostics))
    {
        s.diagnostics = diagnostics ? 1 : 0;
        a.changed = true;
    }
    Help("Off for everyday play: the layer writes nothing while you play (only what happens "
         "once: start-up, a change of mode, the reason something failed) and only the controls "
         "above show. On: every timing, pairing and warp event is written as it happens, for a "
         "run that is going to be read afterwards, the test controls and counters appear, and "
         "the hotkeys work (Home = mode, Delete = mark a break, Page Up / Page Down / End), and so do the test "
         "keys F2, F4, F7 and F8.");
    if (!tests)
        return;

    ImGui::SeparatorText("Test controls");
    float nearThreshold = s.nearThreshold;
    if (StepSlider("Hands/weapon distance", &nearThreshold, 0.0f, 8.0f, 0.1f,
                   nearThreshold <= 0.0f ? "off" : "%.1f units"))
    {
        s.nearThreshold = nearThreshold;
        a.changed = true;
    }
    Help("The same value as \"Leave alone nearer than\". Hotkey: Page Down.");
    bool distanceOnly = s.attachDistanceOnly != 0;
    if (ImGui::Checkbox("Distance only (leave everything nearer alone)", &distanceOnly))
    {
        s.attachDistanceOnly = distanceOnly ? 1 : 0;
        a.changed = true;
    }
    Help("On (the setting that works): everything nearer than the distance is left exactly as "
         "it was. Off: only what is near AND unchanged since that eye's last frame, which "
         "never recognised FC5's weapon. Per pixel only; Smooth always goes by distance.");
    static const char* pairingNames[] = { "0", "2 (FC5)", "4", "Auto" };
    int pairing = s.pairingOffset < 0 ? 3 : std::clamp(s.pairingOffset / 2, 0, 2);
    if (ImGui::Combo("Image lag (Presents)", &pairing, pairingNames, 4))
    {
        s.pairingOffset = pairing == 3 ? -1 : pairing * 2;
        a.changed = true;
    }
    Help("How many Presents the submitted eye image lags the game's camera. FC5: 2 (Depth's red "
         "edges hold still while moving; holes drop from ~10% to under 1%). Auto: picked each "
         "frame by how well each lag fits your head's movement (for when it is not always 2). "
         "Hotkey: Page Up.");
    static const char* headNames[] = { "Headset only", "Match the fresh eye" };
    int head = std::clamp(s.headMode, 0, 1);
    if (ImGui::Combo("Head movement", &head, headNames, 2))
    {
        s.headMode = head;
        a.changed = true;
    }
    Help("Head movement since the stale picture was drawn. Headset only: the headset turns it but "
         "cannot move it, so near things judder when your head moves. Match the fresh eye: warped "
         "to where that eye was when the other eye's picture was drawn -- both eyes show the same "
         "moment.");
    ImGui::SeparatorText("Warp: simpler strafing (compare live)");
    bool strafeOnly = s.strafeOnly != 0;
    if (ImGui::Checkbox("Strafe only (game turns not warped)", &strafeOnly))
    {
        s.strafeOnly = strafeOnly ? 1 : 0;
        a.changed = true;
    }
    Help("Warp corrects only where the game camera MOVED (strafe, walk), not how it TURNED (stick "
         "turns): turns look as they do without the layer. Removes every artefact that comes from "
         "warping a turn.");
    bool showBreaks = s.showBreaks != 0;
    if (ImGui::Checkbox("Show where the warp breaks (colours)", &showBreaks))
    {
        s.showBreaks = showBreaks ? 1 : 0;
        a.changed = true;
    }
    Help("Tints the stale eye where Warp could not do its normal work: RED = a hole (nothing "
         "landed there), BLUE = fill (background that was hidden: stretched or blurred), "
         "YELLOW = that background taken from the other eye, GREEN = the player / "
         "hands/weapon (left unmoved or moved with the body). A diagnostic; not saved.");
    ImGui::SeparatorText("Stale-eye treatment (hide it instead of / as well as warping)");
    static const char* treatNames[] = { "None", "Dim", "Blur" };
    int treat = std::clamp(s.treatMode, 0, 2);
    if (ImGui::Combo("Treatment", &treat, treatNames, 3))
    {
        s.treatMode = treat;
        a.changed = true;
    }
    Help("Instead of (Pass) or on top of (Warp) correcting the stale eye, make it less visible, "
         "so the eye that is up to date dominates. An old bug that showed the stale eye black had "
         "no strafing judder at all, but flickered. Dim: darker (strength 1 = black). Blur: "
         "softer (strength 1 = 24 px). Needs Stale eye = Pass or Warp.");
    if (treat != 0)
    {
        float strength = s.treatStrength;
        if (StepSlider("Strength", &strength, 0.0f, 1.0f, 0.05f, "%.2f"))
        {
            s.treatStrength = strength;
            a.changed = true;
        }
        bool gated = s.treatGated != 0;
        if (ImGui::Checkbox("Only while the game camera moves", &gated))
        {
            s.treatGated = gated ? 1 : 0;
            a.changed = true;
        }
        Help("Scales the strength by the game's own motion (strafe, walk, stick turn): standing "
             "still shows the normal picture; head movement alone does not count.");
        if (gated)
        {
            float fullAt = s.treatFullAt;
            if (StepSlider("Full strength at", &fullAt, 0.005f, 0.1f, 0.005f, "%.3f units/Present"))
            {
                s.treatFullAt = fullAt;
                a.changed = true;
            }
            Help("How much game motion per Present counts as full strength. Walking is about "
                 "0.04; lower = kicks in sooner.");
        }
    }
    bool both = s.bothEyes != 0;
    if (ImGui::Checkbox("Both eyes from the layer", &both))
    {
        s.bothEyes = both ? 1 : 0;
        a.changed = true;
    }
    Help("Also copies the fresh eye into the layer's swapchain. Not needed since the snapshot fix "
         "(an extra copy per frame). Hotkey: End.");
}

// Under the Junk Drawer's part: the layer's counters (most of them with Diagnostics only) and its memory.
inline void DrawStatusJunk(const XrObserverStaleSettingsV1& s, const XrObserverStaleStatusV1& st)
{
    if (s.mode == 0)
        return;
    const bool tests = s.diagnostics != 0;
    if (tests)
        ImGui::Text("%.0f stale images/s, %.0f warped/s; world scale %.3f", st.processedPerSecond, st.warpedPerSecond,
                    st.worldScale);
    if (tests && s.mode == 3)
        ImGui::Text("holes %.1f%%, fill %.1f%%, hands/weapon %.1f%%; GPU at least %.2f ms; game "
                    "motion %.3f/Present",
                    st.holePercent, st.fillPercent, st.attachedPercent, st.gpuMs, st.gameMove);
    if (st.videoBudgetMb > 0.0f)
        ImGui::Text("video memory: %.1f of %.1f GB used by the game and everything in it; the "
                    "warp's own textures %.2f GB",
                    st.videoMemoryMb / 1024.0f, st.videoBudgetMb / 1024.0f, st.ownMemoryMb / 1024.0f);
    if (tests && s.mode == 3)
        ImGui::Text("breaks since the start: %u shown unwarped, %u of them wrong-eye pairings", st.unwarpedTotal,
                    st.wrongEyeTotal);
    if (s.mode == 3 && s.depthMode == 2)
        ImGui::Text("past the player: plane at %.1f m", st.planeMetres);
    if (s.treatMode != 0)
        ImGui::Text("treatment applied: %.2f", st.treatApplied);
}
} // namespace StaleEyeMenu
