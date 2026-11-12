#include "pch.h"
#include "DlssNrFeature_Vk.h"

#include "DlssNr.h"
#include "DlssNr_ExposureScan.h"
#include <inputs/DuniaTaa_Dx12.h>
#include <inputs/FrameCapture_Dx12.h>
#include <inputs/FrameSplit_Dx12.h>
#include <inputs/EvalCapture_Dx12.h>
#include <inputs/GiCapture_Dx12.h>
#include <inputs/VarjoGaze.h>
#include <inputs/Fc5Dlss_Dx11.h>
#include <inputs/Fc5Taa_Dx11.h>
#include <inputs/Fc5Trace_Dx11.h>
#include <inputs/StaleEyeShare.h>
#include <inputs/OverlayShare.h>
#include <inputs/PerEyeDlss_Dx12.h>
#include <inputs/PerEyeDlss_Vk.h>
#include <inputs/FoveaCrop_Vk.h>
#include <inputs/TaaDlaa_Dx11.h>
#include <inputs/TaaMotion_Dx11.h>


#include <Config.h>
#include <Util.h>
#include <menu/menu_common.h>
#include <menu/input/input_system.h>

#include <imgui/imgui.h>
#include <inputs/PerEyeDlss_Dx11.h>
#include <inputs/PerEyeMenu.h>
#include <inputs/FoveaMenu_Vk.h>
#include <inputs/StaleEyeMenu.h>

#include <string>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace DlssNr
{

// The "(?)" marker every control carries, matching the rest of the menu.
// A slider with - / + buttons in front: each press moves it by `step`, and holding one repeats at the keyboard's repeat
// rate -- for fine settings that are hard to hit by dragging (zeroing the gaze offset, say).
// Eye tracking is not part of this build: its controls are not drawn. The ini's keys ([Inputs]
// UpscalerFoveaGaze...) are read as before, so a setup made with another build keeps working.
static const bool kEyeTrackingUi = false;

static bool StepSlider(const char* label, float* v, float min, float max, float step, const char* format)
{
    bool changed = false;
    ImGui::PushID(label);
    ImGui::PushItemFlag(ImGuiItemFlags_ButtonRepeat, true);
    if (ImGui::Button("-"))
    {
        *v = std::max(min, std::round((*v - step) / step) * step);
        changed = true;
    }
    ImGui::SameLine(0.0f, 4.0f);
    if (ImGui::Button("+"))
    {
        *v = std::min(max, std::round((*v + step) / step) * step);
        changed = true;
    }
    ImGui::PopItemFlag();
    ImGui::PopID();
    ImGui::SameLine(0.0f, 6.0f);
    changed |= ImGui::SliderFloat(label, v, min, max, format);
    return changed;
}

static void HelpMarker(const char* tip)
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

// The stale-eye OpenXR layer (R.E.A.L. VR alternate-eye rendering): the layer owns the state (its hotkeys change it
// too); the menu shows what is active, sets it, and keeps the ini in step (saved on overlay close). The controls
// themselves are StaleEyeMenu.h's. This reads the layer's settings for this frame of the menu: false, with the line
// that says why, while there is nothing to show.
static bool StaleEyeBegin(XrObserverStaleSettingsV1& s)
{
    // The saved settings go to the layer as soon as it is there (the menu may be open before gameplay),
    // and nothing is shown or copied back into the ini before the layer has them: its start-up defaults
    // were once saved over the player's (the warp came up Off).
    StaleEyeShare::SyncLayer();
    if (!StaleEyeShare::LayerSettings(&s))
    {
        ImGui::TextDisabled("The warp's OpenXR layer is not running (it starts with the VR session; its folder "
                            "xr_observer lies beside winmm.dll)");
        return false;
    }
    if (!StaleEyeShare::LayerSettingsReady(s))
    {
        ImGui::TextDisabled("Handing the saved settings to the layer...");
        return false;
    }
    return true;
}

// What the menu changed goes to the layer, and the ini is kept in step with what is active (hotkeys included), for
// the save on overlay close.
static void StaleEyeEnd(const XrObserverStaleSettingsV1& s, const StaleEyeMenu::Answer& asked)
{
    if (asked.changed)
        StaleEyeShare::SetLayerSettings(s);
    auto* cfg = Config::Instance();
    if (asked.playerMarks >= 0)
        cfg->StaleEyePlayerMarks = asked.playerMarks == 1;
    if (cfg->StaleEyeMode.value_or_default() != s.mode)
        cfg->StaleEyeMode = s.mode;
    if (cfg->StaleEyePairing.value_or_default() != s.pairingOffset)
        cfg->StaleEyePairing = s.pairingOffset;
    if (std::fabs(cfg->StaleEyeNear.value_or_default() - s.nearThreshold) > 1e-4f)
        cfg->StaleEyeNear = s.nearThreshold;
    if (cfg->StaleEyeBothEyes.value_or_default() != (s.bothEyes != 0))
        cfg->StaleEyeBothEyes = s.bothEyes != 0;
    if (cfg->StaleEyeHead.value_or_default() != s.headMode)
        cfg->StaleEyeHead = s.headMode;
    if (cfg->StaleEyeTreat.value_or_default() != s.treatMode)
        cfg->StaleEyeTreat = s.treatMode;
    if (std::fabs(cfg->StaleEyeTreatStrength.value_or_default() - s.treatStrength) > 1e-4f)
        cfg->StaleEyeTreatStrength = s.treatStrength;
    if (cfg->StaleEyeTreatGated.value_or_default() != (s.treatGated != 0))
        cfg->StaleEyeTreatGated = s.treatGated != 0;
    if (std::fabs(cfg->StaleEyeTreatFullAt.value_or_default() - s.treatFullAt) > 1e-5f)
        cfg->StaleEyeTreatFullAt = s.treatFullAt;
    if (cfg->StaleEyeStrafeOnly.value_or_default() != (s.strafeOnly != 0))
        cfg->StaleEyeStrafeOnly = s.strafeOnly != 0;
    if (cfg->StaleEyeDepthMode.value_or_default() != s.depthMode)
        cfg->StaleEyeDepthMode = s.depthMode;
    if (std::fabs(cfg->StaleEyeClamp.value_or_default() - s.clampMetres) > 1e-4f)
        cfg->StaleEyeClamp = s.clampMetres;
    if (std::fabs(cfg->StaleEyeSingle.value_or_default() - s.singleMetres) > 1e-4f)
        cfg->StaleEyeSingle = s.singleMetres;
    if (cfg->StaleEyeAttachDistanceOnly.value_or_default() != (s.attachDistanceOnly != 0))
        cfg->StaleEyeAttachDistanceOnly = s.attachDistanceOnly != 0;
    if (cfg->StaleEyeNearWithBody.value_or_default() != (s.nearWithBody != 0))
        cfg->StaleEyeNearWithBody = s.nearWithBody != 0;
    if (cfg->StaleEyeFillOtherEye.value_or_default() != (s.fillFromOtherEye != 0))
        cfg->StaleEyeFillOtherEye = s.fillFromOtherEye != 0;
    if (cfg->StaleEyeDiagnostics.value_or_default() != (s.diagnostics != 0))
        cfg->StaleEyeDiagnostics = s.diagnostics != 0;
}

// A slider that only writes its value when the handle is released.
//
// Some controls -- intensity, the structure and tone strengths -- are read by the model once, when
// the feature is built, so changing one rebuilds the whole feature. Writing on every pixel of a drag
// meant a rebuild per frame, felt as the picture hitching while you scrub. The slider still tracks
// live under the cursor; only the commit that triggers the rebuild waits for release. Cheap controls
// that are just shader constants (detail, colour, paper white) do not use this -- they can afford to
// apply live.
template <typename Option>
static bool DeferredSlider(const char* label, Option* opt, float mn, float mx,
                           float def, const char* fmt = "%.2f", bool inheritReset = false)
{
    static std::unordered_map<ImGuiID, float> pending;
    const ImGuiID id = ImGui::GetID(label);

    auto it = pending.find(id);
    float value = it != pending.end() ? it->second : (opt->has_value() ? opt->value() : def);
    bool changed = false;

    if (ImGui::SliderFloat(label, &value, mn, mx, fmt))
        pending[id] = value;

    if (ImGui::IsItemDeactivatedAfterEdit())
    {
        auto committed = pending.find(id);

        if (committed != pending.end())
        {
            *opt = std::clamp(committed->second, mn, mx);
            pending.erase(committed);
            changed = true;
        }
    }

    ImGui::SameLine();

    const std::string resetId = std::string("Reset##") + label;
    if (ImGui::SmallButton(resetId.c_str()))
    {
        if (inheritReset)
            *opt = std::optional<float> {};
        else
            *opt = def;
        pending.erase(id);
        changed = true;
    }

    if (std::strcmp(label, "Intensity") == 0)
        HelpMarker("Overall enhancement strength for this pass. 1 = default; results depend on the profile.\nValues above 1 are experimental; the runtime may clamp or ignore them.");
    else if (std::strcmp(label, "Local structure") == 0)
        HelpMarker("Fine detail and local contrast requested from the model (high-frequency structure).\n1 = default; values above 1 are experimental.");
    else if (std::strcmp(label, "Local tone") == 0)
        HelpMarker("Broad brightness and lighting changes requested from the model (low-frequency tone).\nLater passes default to 0. Values above 1 are experimental.");
    else if (std::strcmp(label, "Skin structure") == 0)
        HelpMarker("Fine detail for pixels the model identifies as skin. -1 follows Local structure; 0 reduces skin detail.\nSkin colour is controlled separately. Values above 1 are experimental.");
    return changed;
}

// An absent later-pass setting inherits pass 1. The first combo item represents that absence; the
// remaining items map directly to the model's zero-based profile values.
static bool InheritedProfileCombo(const char* label, CustomOptional<uint32_t, NoDefault>* opt,
                                  const char* const* names, int nameCount)
{
    int selected = 0;

    if (opt->has_value())
        selected = std::clamp((int) opt->value(), 0, nameCount - 2) + 1;

    if (!ImGui::Combo(label, &selected, names, nameCount))
        return false;

    if (selected == 0)
        *opt = std::optional<uint32_t> {};
    else
        *opt = (uint32_t) (selected - 1);

    return true;
}

// The overlay as a panel in the headset (inputs/OverlayShare): the game DLL's own settings, read at every Present, so
// they act while they are moved. In the main menu's LEFT column, not folded away, in every game in which the OpenXR
// layer asks for the overlay (it used to be inside the Far Cry 5 section). Nothing is shown in
// a game without the layer.
void RenderHeadsetMenu()
{
    if (!OverlayShare::LayerAsks())
        return;
    auto* config = Config::Instance();
    ImGui::SeparatorText("This menu in the headset");
    bool panel = config->HeadsetOverlay.value_or_default();
    if (ImGui::Checkbox("Show this menu in the headset", &panel))
        config->HeadsetOverlay = panel;
    HelpMarker("The menu and its messages are drawn on the game's flat window, which the headset does not show. On: "
               "they are also shown as a flat panel in front of you, while there is something to show. Nothing is "
               "drawn or copied with the menu closed.");
    if (panel)
    {
        float width = config->HeadsetOverlayWidth.value_or_default();
        if (StepSlider("Panel scale", &width, 0.4f, 4.0f, 0.1f, "%.1f m per 2048 px"))
            config->HeadsetOverlayWidth = width;
        float distance = config->HeadsetOverlayDistance.value_or_default();
        if (StepSlider("Panel distance", &distance, 0.5f, 5.0f, 0.1f, "%.1f m"))
            config->HeadsetOverlayDistance = distance;
        float down = config->HeadsetOverlayDown.value_or_default();
        if (StepSlider("Panel lower by", &down, -1.0f, 1.0f, 0.05f, "%.2f m"))
            config->HeadsetOverlayDown = down;
        HelpMarker("The panel shows the menu, centred in front of your eyes, wherever the menu is in the game's "
                   "window, and stays there as you turn your head. A larger scale or a smaller distance makes "
                   "everything larger, up to what the view takes in; Ctrl + mouse wheel changes the menu's own text "
                   "size.");
        RvrToolsOverlayV1 now {};
        if (OverlayShare::Latest(&now) && now.visible != 0 && now.usedWidth != 0)
            ImGui::Text("Panel now: %.2f x %.2f m, %u x %u pixels", now.panelWidth,
                        now.panelWidth * (float) now.usedHeight / (float) now.usedWidth, now.usedWidth, now.usedHeight);
    }
    // What the gamepad does while this menu is open, as counted: the menu must see it, the game must not. (The
    // counts are since the game started.)
    const auto input = OptiInput::GetDebugState();
    const bool menuSeesGamepad = (ImGui::GetIO().BackendFlags & ImGuiBackendFlags_HasGamepad) != 0;
    ImGui::Text("Gamepad: the menu %s one (D-pad moves, A presses, B goes back)",
                menuSeesGamepad ? "sees" : "does NOT see");
    static bool gamepadLogged = false;
    if (menuSeesGamepad && !gamepadLogged)
    {
        gamepadLogged = true;
        LOG_INFO("Menu gamepad: the menu sees a gamepad ({} reads of its own past the block); XInput hooks {}, window "
                 "in front {}, the game's reads kept from it so far {}, passed on {}",
                 OptiInput::OwnGamepadReadCount(),
                 input.XInputGetStateHookInstalled || input.XInputGetStateExHookInstalled, input.Focused,
                 input.XInputGetStateBlockedCount, input.XInputGetStatePassedCount);
    }
    if (!input.XInputGetStateHookInstalled && !input.XInputGetStateExHookInstalled)
        ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.2f, 1.0f), "The game's gamepad is NOT kept from it (no XInput hook)");
    else if (!input.Focused)
        ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.2f, 1.0f),
                           "The game's window is not in front: its input is NOT kept from it");
    else
        ImGui::Text("Kept from the game: %llu gamepad reads (passed on: %llu)",
                    (unsigned long long) input.XInputGetStateBlockedCount,
                    (unsigned long long) input.XInputGetStatePassedCount);
    ImGui::Spacing();
}

void RenderMenu(Config* config, float menuResScale)
{

    // <Game> DLSS ---------------------------------------
    // One section named for the game that is running: the F6 toggle and state,
    // the live DLSS preset, the status and warnings, the Resolution Scale guidance, and the diagnostics behind
    // the ini key. Only one game runs at a time, so the game's own detection decides both whether the section
    // exists and what it is called; a new port adds its name and its block here, nothing anywhere else.
    // Open by default: for a player it is the whole overlay.
    // The generic D3D12 per-eye path names no game: it serves any game calling DLSS through OptiScaler.
    const bool perEyeDx12 =
        !DuniaTaa_Dx12::Active() && PerEyeDlss_Dx12::Enabled() && State::Instance().api == DX12;
    // The Vulkan port of its core (PerEyeDlss_Vk: the twin, the routing, the same-eye fit; a Vulkan game).
    const bool perEyeVk = PerEyeDlss_Vk::Enabled() && State::Instance().api == Vulkan;
    const char* gameDlssTitle = DuniaTaa_Dx12::Active()                              ? "Far Cry 6 DLSS"
                                : (Fc5Taa_Dx11::Enabled() || Fc5Dlss_Dx11::Enabled()) ? "Far Cry 5 DLSS"
                                : TaaDlaa_Dx11::Enabled()                             ? "Days Gone DLSS"
                                : perEyeDx12                                          ? "Per-eye DLSS"
                                : perEyeVk                                            ? "Per-eye DLSS (Vulkan)"
                                                                                       : nullptr;
    // The D3D11 per-eye path (a mod calls DLSS through OptiScaler): named for no game.
    // Last in the order: the games above own F6, F4 and F8 when they run (menu_common's keys have the same rule).
    // The API is known from the first DLSS feature on; before that (a mod may create its feature late)
    // the swapchain's says it.
    const auto& state = State::Instance();
    const bool perEyeDx11 = gameDlssTitle == nullptr && PerEyeDlss_Dx11::Enabled() &&
                            (state.api == DX11 || (state.api == NotSelected && state.swapchainApi == DX11));
    if (perEyeDx11)
        gameDlssTitle = "Per-eye DLSS";
    if (GiCapture_Dx12::Enabled())
    {
        ImGui::Spacing();
        if (auto ch = ScopedCollapsingHeader("Crimson Desert", ImGuiTreeNodeFlags_DefaultOpen); ch.IsHeaderOpen())
        {
            ScopedIndent indent {};
            ImGui::Spacing();
            const GiCapture_Dx12::HistoryUi ui = GiCapture_Dx12::HistoryUiStatus();
            bool on = ui.step != 0;
            if (ImGui::Checkbox("Per-eye lighting", &on))
                GiCapture_Dx12::ToggleHistory();
            const ImVec4 good(0.4f, 1.0f, 0.4f, 1.0f), bad(1.0f, 0.5f, 0.3f, 1.0f);
            if (ui.step == 0)
                ImGui::TextUnformatted("Off. Change graphics settings now, then turn it on.");
            else
            {
                ImGui::TextColored(bad, "Turn it off before changing graphics settings.");
                if (ui.textures == 0)
                    ImGui::TextColored(bad, "NOT running: looking for its textures (normal in a menu or loading)");
                else
                    ImGui::TextColored(good, "Running: %d textures found", ui.textures);
                if (ui.textures != 0)
                {
                    if (ui.step < 8)
                        ImGui::TextColored(bad, "Motion fix: off on this step");
                    else if (ui.correcting)
                        ImGui::TextColored(good, "Motion fix: running");
                    else
                        ImGui::TextColored(bad, "Motion fix: NOT running");
                    if (ui.step < 12 || ui.limitRunning)
                        ImGui::TextColored(ui.step < 12 ? bad : good, ui.step < 12 ? "Spot fix: off on this step" : "Spot fix: running");
                    else
                        ImGui::TextColored(bad, "Spot fix: NOT running at these settings");
                }
            }
            if (GiCapture_Dx12::Developer() && ImGui::TreeNode("Junk Drawer##crimson"))
            {
                // 5, 4 and 8 are step 2's parts apart, not on F9: only to find which of them brings a fault in.
                static constexpr int kSteps[] = { 0, 5, 4, 8, 9, 12, 13, 15, 17, 18, 16 };
                static const char* const kNames[] = { "1 beep: off", "A: same-eye camera, no swaps", "B: + swaps, no motion fix", "C: + motion fix, no moments", "2: + moments", "3: + spots at 31", "4: + spots at 16", "5: + NPC motion doubled", "6: 5 + slow motion too", "7: 6 + clothing fix", "8: 5 + reflections on motion" };
                int place = -1;
                for (int i = 0; i < 11; ++i)
                    if (kSteps[i] == ui.step)
                        place = i;
                if (ImGui::Combo("Step", &place, kNames, 11) && place >= 0)
                    GiCapture_Dx12::SetHistoryStep(kSteps[place]);
                ImGui::Text("Shaders edited %llu, other versions left alone %llu", (unsigned long long) ui.shadersEdited, (unsigned long long) ui.otherVersions);
                ImGui::TreePop();
            }
            ImGui::Spacing();
        }
    }
    if (gameDlssTitle != nullptr)
    {
        ImGui::Spacing();
        if (auto ch = ScopedCollapsingHeader(gameDlssTitle, ImGuiTreeNodeFlags_DefaultOpen); ch.IsHeaderOpen())
        {
            ScopedIndent indent {};
            ImGui::Spacing();

            // The simple part first (the shimmer fix, per eye), then ONE closed tree "Junk Drawer"
            // with what a reader of a run needs: object motion F4, the view F8, the cost on the GPU.
            PerEyeMenu::Status perEyeStatus;
            PerEyeMenu::Answer perEyeAsked;
            if (perEyeDx11)
            {
                perEyeStatus = PerEyeDlss_Dx11::GetStatus();
                perEyeAsked = PerEyeMenu::Draw(perEyeStatus, menuResScale);
                ImGui::Spacing();
            }
            if (ImGui::TreeNode("Advanced"))
            {
                if (perEyeDx11)
                    PerEyeMenu::DrawJunkDrawer(perEyeStatus, menuResScale, perEyeAsked);
                ImGui::TreePop();
            }
            ImGui::Spacing();

            if (perEyeDx11)
            {
                if (perEyeAsked.toggleRouting)
                    PerEyeDlss_Dx11::ToggleRouting();
                if (perEyeAsked.startPerEye >= 0)
                {
                    config->UpscalerPerEyeStart = perEyeAsked.startPerEye != 0;
                    LOG_INFO("Per-eye DLSS: overlay, [Inputs] UpscalerPerEyeStart = {} (saved when the overlay closes)",
                             perEyeAsked.startPerEye != 0);
                }
                if (perEyeAsked.variant >= 0)
                    PerEyeDlss_Dx11::SetVariant(perEyeAsked.variant);
                if (perEyeAsked.view >= 0)
                    PerEyeDlss_Dx11::SetView(perEyeAsked.view);
                if (perEyeAsked.timing >= 0)
                {
                    config->UpscalerGpuTiming = perEyeAsked.timing != 0;
                    LOG_INFO("GPU timing: overlay, [Inputs] UpscalerGpuTiming = {}", perEyeAsked.timing != 0);
                }
            }

            if (perEyeVk)
            {
                const auto st = PerEyeDlss_Vk::GetStatus();
                // THE SIMPLE PART FIRST (a cleaner UI, so players only see options that are relevant
                // and work for them): the foveal crop and the periphery (FoveaMenu_Vk.h, drawn by overlay_test "fovea" too;
                // the overlay is the crop's switch on Vulkan, F4 being the game's key in one Vulkan game). Then ONE closed tree with the
                // rest: our own per-eye fit (not needed wherever DL(S)SS runs), the crop's A/B
                // switches, the TAA periphery's tuning, the counters.
                const FoveaMenuVk::Status foveaStatus = FoveaCrop_Vk::GetStatus();
                FoveaMenuVk::Answer foveaAsked = FoveaMenuVk::Draw(foveaStatus, menuResScale);
                ImGui::Spacing();
                if (ImGui::TreeNode("Advanced"))
                {
                    ImGui::TextDisabled("Tests, A/B switches and fine tuning. Nothing here needs changing to play.");
                    ImGui::SeparatorText("Per eye (our own same-eye fit)");
                    bool perEye = st.routing;
                    if (ImGui::Checkbox("Per eye", &perEye))
                        PerEyeDlss_Vk::ToggleRouting();
                    HelpMarker("One DLSS history per eye, with same-eye motion fitted from the game's vectors. Off: the "
                               "game's own single DLSS, shared by both eyes.");
                    auto* cfg = Config::Instance();
                    bool leaveOut = cfg->UpscalerFitLeaveOutNearest.value_or_default();
                    if (ImGui::Checkbox("Fit ignores the nearest things (car cabin)", &leaveOut))
                    {
                        cfg->UpscalerFitLeaveOutNearest = leaveOut;
                        LOG_INFO("Per-eye DLSS VK: the same-eye fit {} (overlay)",
                                 leaveOut ? "LEAVES OUT what lies nearer than a depth gap" : "fits every sample");
                    }
                    HelpMarker("Per eye only. On: inside a vehicle (or on a horse, or with anything close that moves "
                               "with you) the camera's motion is fitted from the world beyond the gap in depth only. On "
                               "foot nothing changes.");
                    ImGui::TextDisabled("%s; last window: %llu of %llu fits valid; the eye from the %s",
                                        st.twin ? "twin feature made" : "no twin feature (yet)", st.fitsValid, st.fitsRead,
                                        st.presentClock ? "Presents" : "evaluates");
                    FoveaMenuVk::DrawJunk(foveaStatus, menuResScale, foveaAsked);
                    ImGui::TreePop();
                }
                FoveaCrop_Vk::Apply(foveaAsked);
            }

            if (perEyeDx12)
            {
                const auto st = PerEyeDlss_Dx12::GetStatus();
                // THE SIMPLE PART FIRST (a cleaner UI, so players only see options that are relevant
                // and work for them): the foveal crop, its preset and size, the periphery and its kind, and with an eye
                // tracker the crop following the gaze and its calibration. Each control on a line of its own (the headset's
                // panel cuts a row off at the right). Then ONE closed tree "Junk Drawer" with everything that is a test, an
                // A/B switch, fine tuning or a counter: our own per-eye fit (not needed wherever
                // DL(S)SS runs), the full pass's preset (the DLSS settings' Override Preset is the same setting), the crop's
                // inputs, the TAA periphery's tuning, the gaze's source / projection / manual offsets. The crop's offsets
                // are out of the drawer, with a sideways one for both eyes.
                static const char* presetNames[] = { "Default", "E", "J", "K", "L", "M" };
                static const int presetValues[] = { 0, 5, 10, 11, 12, 13 };
                auto presetIndex = [](int value)
                {
                    for (int i = 0; i < 6; ++i)
                        if (presetValues[i] == value)
                            return i;
                    return 0;
                };
                // ---- Foveated SR ----
                ImGui::SeparatorText("Foveated SR");
                bool fovea = st.fovea;
                if (ImGui::Checkbox("Crop on", &fovea))
                    PerEyeDlss_Dx12::ToggleFovea();
                HelpMarker("A second DLSS pass over a crop of each eye at its own preset, blended over the full pass. "
                           "Its features are made the first time it is turned on (a short hitch).");
                // Said only while it is so: the crop is made for DL(S)SS's vectors, and nobody should be on our
                // own estimate without knowing.
                if (st.fovea && !st.belowSeen)
                {
                    ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.2f, 1.0f),
                                       "DL(S)SS not seen: the crop is using its own motion estimate");
                    HelpMarker("The crop is made and tested for R.E.A.L. VR's DL(S)SS being on: its motion vectors are "
                               "then right for each eye. Without it the crop runs on a motion estimate of its own, "
                               "tested in few games, and it can shimmer. Turn DL(S)SS on in R.E.A.L. VR, or the crop "
                               "off. (This line can show for a moment in menus and pauses.)");
                }
                bool outline = st.outline;
                if (ImGui::Checkbox("Outline", &outline))
                    PerEyeDlss_Dx12::ToggleFoveaOutline();
                HelpMarker("Draws the crop's edge in magenta. Only visible while the crop is on.");

                ImGui::PushItemWidth(100.0f * menuResScale);
                int cropIndex = presetIndex(st.preset);
                if (ImGui::Combo("Crop preset", &cropIndex, presetNames, 6) && presetValues[cropIndex] != st.preset)
                    PerEyeDlss_Dx12::RequestFoveaPreset(presetValues[cropIndex]);
                HelpMarker("Rebuilds only the crop's DLSS features, live: the crop is missing for a few frames. SR "
                           "presets: with ray reconstruction on, the crop is an RR feature with the RR preset of the "
                           "DLSS settings.");
                ImGui::PopItemWidth();
                if (st.cropRr)
                {
                    // What the crop really runs under ray reconstruction: the hint its RR feature was made with.
                    static const char* rrNames[] = { "the DLL's default", "A", "B", "C", "D", "E", "F" };
                    if (st.rrCropPreset < 0)
                        ImGui::TextWrapped("Ray reconstruction is on: the crop preset above is not used.");
                    else if (st.rrCropPreset <= 6)
                        ImGui::TextWrapped("Ray reconstruction is on: the crop runs RR preset %s (the preset above "
                                           "is not used).",
                                           rrNames[st.rrCropPreset]);
                    else
                        ImGui::TextWrapped("Ray reconstruction is on: the crop runs RR preset number %d (the preset "
                                           "above is not used).",
                                           st.rrCropPreset);
                    HelpMarker("Read back from the crop's ray reconstruction feature when it was made. Change it with "
                               "'Render Presets Override' + 'Override Preset' under DLSSD Settings, then Apply.");
                }

                ImGui::PushItemWidth(220.0f * menuResScale);
                // Applied when a slider is let go: the crop's features are rebuilt at the new size.
                static float widthEdit = 0.4f, heightEdit = 0.4f;
                static bool widthEditing = false, heightEditing = false;
                if (!widthEditing)
                    widthEdit = st.size;
                if (!heightEditing)
                    heightEdit = st.sizeH;
                ImGui::SliderFloat("Width", &widthEdit, 0.2f, 0.8f, "%.2f of the eye");
                widthEditing = ImGui::IsItemActive();
                const bool widthDone = ImGui::IsItemDeactivatedAfterEdit();
                HelpMarker("Width of the crop as a fraction of the eye. Applied when you let go: the crop's features "
                           "are rebuilt at the new size (a few frames without the crop). Unless the crop follows the "
                           "gaze, a size that aligns exactly with the eye's pixels is taken when it is within 0.02 "
                           "(only a few exist, e.g. 0.25 / 0.5 / 0.75); otherwise you get the size you ask for.");
                ImGui::SliderFloat("Height", &heightEdit, 0.2f, 0.8f, "%.2f of the eye");
                heightEditing = ImGui::IsItemActive();
                const bool heightDone = ImGui::IsItemDeactivatedAfterEdit();
                HelpMarker("Height of the crop as a fraction of the eye, independent of the width.");
                if (widthDone || heightDone)
                    PerEyeDlss_Dx12::RequestFoveaSize(widthEdit, heightEdit);
                int feather = st.feather;
                if (ImGui::SliderInt("Edge blend", &feather, 0, 128, "%d px"))
                    PerEyeDlss_Dx12::RequestFoveaFeather(feather);
                // The crop's place, in the open (so it can be used if needed), with - / +
                // steps for the headset. While the crop follows the gaze, the gaze places it instead.
                {
                    float offX = st.offsetX, offY = st.offsetY, offSide = st.offsetSideways;
                    bool moved = StepSlider("Left / right", &offSide, -0.25f, 0.25f, 0.005f, "%.3f");
                    HelpMarker("Moves the crop sideways, the same way in both eyes (+ = right), as a fraction of the "
                               "eye. Works at once, without the eyes told apart.");
                    // Order and names: 'Up / down' under 'Left / right', the nose one last.
                    moved |= StepSlider("Up / down", &offY, -0.25f, 0.25f, 0.005f, "%.3f");
                    HelpMarker("Moves the crop up or down, the same for both eyes (+ = up).");
                    moved |= StepSlider("Toward the nose", &offX, -0.25f, 0.25f, 0.005f, "%.3f");
                    HelpMarker("Moves the crop toward the nose in each eye (mirrored between the eyes; the "
                               "straight-ahead point sits toward the nose in most headsets' eye images). Applies once "
                               "the eyes are told apart, a few seconds into play.");
                    if (moved)
                        PerEyeDlss_Dx12::RequestFoveaOffset(offX, offY, offSide);
                    if (ImGui::Button("Centre"))
                        PerEyeDlss_Dx12::RequestFoveaOffset(0.0f, 0.0f, 0.0f);
                    HelpMarker("All three back to 0. Moving the crop restarts its history, so move it, then look.");
                }
                ImGui::PopItemWidth();
                bool periphery = st.periphery;
                if (ImGui::Checkbox("Cheap periphery", &periphery))
                    PerEyeDlss_Dx12::RequestPeriphery(periphery);
                HelpMarker("With the crop on: outside it, DLSS runs at the render size (no upscaling) and is enlarged "
                           "to the eye, instead of the full-size pass -- cheaper, a little softer at the edges. Its "
                           "features are made the first time (a short hitch).");
                ImGui::PushItemWidth(100.0f * menuResScale);
                int periphIndex = presetIndex(st.peripheryPreset);
                if (ImGui::Combo("Periphery preset", &periphIndex, presetNames, 6) &&
                    presetValues[periphIndex] != st.peripheryPreset)
                    PerEyeDlss_Dx12::RequestPeripheryPreset(presetValues[periphIndex]);
                ImGui::PopItemWidth();
                HelpMarker("The cheap periphery's own preset (rebuilt live). The Outer preset still applies whenever "
                           "the full-size pass runs.");
                {
                    static const char* oneNames[] = { "Auto (one with DL(S)SS)", "One for both eyes", "One per eye" };
                    int oneMode = std::clamp(Config::Instance()->UpscalerPeripheryOneFeature.value_or_default(), 0, 2);
                    ImGui::PushItemWidth(170.0f * menuResScale);
                    if (ImGui::Combo("Periphery features", &oneMode, oneNames, 3))
                        PerEyeDlss_Dx12::RequestPeripheryOneFeature(oneMode);
                    ImGui::PopItemWidth();
                    ImGui::SameLine();
                    ImGui::TextUnformatted(PerEyeDlss_Dx12::PeripheryOneFeatureNow() ? "(now: one)" : "(now: per eye)");
                    HelpMarker("Shared routing only. Auto: ONE feature for both eyes on the game's own vectors whenever "
                               "R.E.A.L. VR's DL(S)SS is seen at work on the periphery (10 evaluates in a row; back "
                               "to one per eye after 60 without); else a feature per eye "
                               "on our same-eye motion. One per eye under DL(S)SS shimmers "
                               "in some games; one for both eyes without DL(S)SS mixes the eyes' histories. A change "
                               "restarts the periphery's history.");
                }
                bool peripheryTaa = st.peripheryTaa;
                if (ImGui::Checkbox("Own TAA periphery", &peripheryTaa))
                    PerEyeDlss_Dx12::RequestPeripheryTaa(peripheryTaa);
                HelpMarker("With the crop and Cheap periphery on: outside the crop our own TAA runs instead of any "
                           "DLSS call -- cheaper, softer and noisier than DLSS. Follows Per eye: per eye, one "
                           "history per eye with same-eye motion; shared, one history for both eyes moved by the game's "
                           "own vectors (from the other eye's frame).");
                bool peripherySr = st.peripherySr;
                if (ImGui::Checkbox("SR periphery in RR mode", &peripherySr))
                    PerEyeDlss_Dx12::RequestPeripherySr(peripherySr);
                HelpMarker("With ray reconstruction on: the cheap periphery runs SR (Periphery preset) instead of RR. "
                           "RR's cost follows its input size, so an RR periphery saves little; SR costs far less but "
                           "does not denoise -- the ray traced noise RR removes stays outside the crop. Rebuilds the "
                           "periphery features live.");
                if (st.taaFailed)
                    ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.2f, 1.0f), "The TAA periphery failed; see OptiScaler.log.");
                if (st.peripheryFailed)
                    ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.2f, 1.0f), "The cheap periphery failed; see OptiScaler.log.");

                if (st.cropOutW > 0)
                    ImGui::Text("Crop: render %ux%u -> %ux%u at (%u, %u) of %ux%u, %.0f%% of the eye%s", st.cropW,
                                st.cropH, st.cropOutW, st.cropOutH, st.cropX, st.cropY, st.eyeW, st.eyeH,
                                st.cropPercent, st.exact ? ", exact" : "");
                if (st.failed)
                    ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.2f, 1.0f), "The crop failed to start; see OptiScaler.log.");
                if (kEyeTrackingUi)
                {
                    // The tracker's own switch (the eye tracking crop disabled separately from
                    // the SR crop): off = the tracker is not read, the crop stays in the middle, no gaze box, and the block
                    // below is hidden. Shown wherever a launch could have a tracker (the D3D12 path).
                    bool gaze = Config::Instance()->UpscalerFoveaGaze.value_or_default();
                    if (ImGui::Checkbox("Eye tracking", &gaze))
                    {
                        Config::Instance()->UpscalerFoveaGaze = gaze;
                        LOG_INFO("Eye tracking: overlay, [Inputs] UpscalerFoveaGaze = {} (in use now: {}; started at the launch: {})",
                                 gaze, VarjoGaze::Enabled(), VarjoGaze::StartedAtLaunch());
                    }
                    HelpMarker("Uses the headset's eye tracker so that the crop can follow where you look. Off: the tracker is "
                               "not read at all, the crop stays in the middle of each eye and nothing is drawn at your gaze. "
                               "Turning it off works at once. Turning it on works at once if the game was started with it on; "
                               "otherwise it starts with the next launch. Saved.");
                    if (gaze && !VarjoGaze::StartedAtLaunch())
                        ImGui::TextDisabled("Eye tracking starts with the next launch of the game.");
                }
                if (kEyeTrackingUi && st.gazeOn)
                {
                    ImGui::SeparatorText("Eye tracking");
                    {
                        bool follow = st.follow;
                        if (ImGui::Checkbox("Crop follows gaze", &follow))
                            Config::Instance()->UpscalerFoveaGazeFollow = follow;
                        HelpMarker("The foveal crop moves to where you look. It recentres only when your gaze leaves the "
                                   "inner part of the crop (the dead zone), and a move keeps the crop's history (DLSS "
                                   "sees it as a pan). With the outline a green box shows the raw gaze while this is on.");
                        float dz = st.deadZone;
                        ImGui::PushItemWidth(220.0f);
                        if (StepSlider("Dead zone", &dz, 0.1f, 0.9f, 0.05f, "%.2f"))
                            Config::Instance()->UpscalerFoveaGazeDeadZone = dz;
                        ImGui::PopItemWidth();
                        HelpMarker("How far your gaze may wander from the crop's centre before it moves, as a fraction "
                                   "of half the crop. Higher = the crop moves less often (steadier, and tracker jitter "
                                   "never moves it); lower = it stays centred on your gaze more closely.");
                        float glide = st.glideMs;
                        ImGui::PushItemWidth(220.0f);
                        if (StepSlider("Glide (ms)", &glide, 0.0f, 300.0f, 10.0f, "%.0f"))
                            Config::Instance()->UpscalerFoveaGazeGlideMs = glide;
                        ImGui::PopItemWidth();
                        HelpMarker("How the crop moves once it moves: it glides after your gaze, covering about 63% of "
                                   "the way in this time (and 95% in three times it). 0 = it jumps there in one frame. "
                                   "Longer = smoother but it trails your eyes; each step then brings only a thin strip "
                                   "of new picture into the crop.");
                        float ageFrames = (float) st.ageFrames;
                        ImGui::PushItemWidth(220.0f);
                        if (StepSlider("Fade-in after a move (frames)", &ageFrames, 0.0f, 30.0f, 1.0f, "%.0f"))
                            Config::Instance()->UpscalerFoveaAgeFrames = (int) ageFrames;
                        ImGui::PopItemWidth();
                        HelpMarker("The picture that comes into the crop at a move has no history yet and takes a few "
                                   "frames to settle; until a pixel has been in the crop this many frames, the settled "
                                   "periphery shows through it instead, in proportion. 0 = off. Takes effect at once.");
                        if (st.follow && st.copiesSwappedBelow && st.copyInputsNow)
                            ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.2f, 1.0f),
                                               "R.E.A.L. VR's DL(S)SS is on and the crop reads copies (set by hand):\n"
                                               "the crop shimmers that way. Untick 'Crop reads\n"
                                               "copies' in the Junk Drawer: a move is carried over through subrects too.");
                        if (st.follow && st.followResetsNoCopy > 0)
                            ImGui::TextWrapped("Every move restarts the crop's history (%llu so far): no copies and no "
                                               "vectors of ours to shift (is 'Vectors are same-eye already' ticked?).",
                                               (unsigned long long) st.followResetsNoCopy);
                    }
                    if (ImGui::Button("Centre on my gaze (one yellow target, ~4 s)"))
                        VarjoGaze::StartCentring();
                    HelpMarker("One point instead of nine. Close the overlay after pressing: a yellow target appears in "
                               "the middle for about 4 s (the first 3 s are for closing the overlay and finding it) -- "
                               "look at it until it goes. That reading becomes straight "
                               "ahead (the offset below is set; a saved 9-point calibration keeps its scale under it). "
                               "Enough when the box is only shifted, as the tracker's usual few degrees low. Needs the "
                               "crop on. Repeat any time.");
                    if (ImGui::Button("Calibrate gaze (9 points, ~15 s)"))
                        VarjoGaze::StartCalibration();
                    ImGui::SameLine(0.0f, 12.0f);
                    if (ImGui::Button("Clear calibration"))
                        VarjoGaze::ClearCalibration();
                    HelpMarker("Close the overlay after pressing it. A yellow target appears at 9 places in turn, about "
                               "1.6 s each: look straight at it and keep looking until it moves (keep your head still "
                               "if you can). The fit replaces the offset and range below (they reset; the offset then "
                               "works as a fine trim). Needs the crop on with RR. Saved with the ini.");
                    ImGui::TextWrapped("Calibration: %s", VarjoGaze::CalibrationText().c_str());
                    ImGui::TextWrapped("Gaze: %s", st.gazeText.c_str());
                }
                ImGui::Spacing();
                if (ImGui::TreeNode("Advanced"))
                {
                    ImGui::TextDisabled("Tests, A/B switches and fine tuning. Nothing here needs changing to play.");
                    ImGui::SeparatorText("Per eye (our own same-eye fit)");
                    bool perEye = st.routing;
                    if (ImGui::Checkbox("Per eye", &perEye))
                        PerEyeDlss_Dx12::ToggleRouting();
                    HelpMarker("One DLSS history per eye, with same-eye motion fitted from the game's vectors. Off: the "
                               "game's own single DLSS, shared by both eyes.");
                    bool giveUp = st.giveUpStale;
                    if (ImGui::Checkbox("Fit recovers after fast motion", &giveUp))
                        PerEyeDlss_Dx12::RequestGiveUpStale(giveUp);
                    HelpMarker("Per eye only. On: after two failed fits in a row the fit starts afresh instead of from the "
                               "eye's last good one, which fast motion (sprinting, driving) can leave too far behind to "
                               "ever be found again. Off: as before (for comparison).");
                    bool fitRestart = st.fitRestart;
                    if (ImGui::Checkbox("Sudden turns: fit starts over at once", &fitRestart))
                        PerEyeDlss_Dx12::RequestFitRestart(fitRestart);
                    HelpMarker("Per eye only. On: when a sudden turn (a flick of the stick) leaves the eye's last fit "
                               "behind, the fit starts over in the same frame. Off: as before -- two frames of each eye "
                               "get the old fit first, at the turn's start and at its end (the picture shimmers and "
                               "settles).");
                    bool leaveOutNearest = st.fitLeaveOutNearest;
                    if (ImGui::Checkbox("Fit ignores the nearest things (car cabin)", &leaveOutNearest))
                        PerEyeDlss_Dx12::RequestFitLeaveOutNearest(leaveOutNearest);
                    HelpMarker("Wherever the fit runs (per eye; the crop or NR in shared routing). On: inside a vehicle the camera's motion is fitted from the world outside "
                               "only -- the car around you (found by the gap in depth between it and the world) moves "
                               "with you, not with the world. On foot nothing changes. Off: as before -- the fit sticks "
                               "to the car and the world shimmers while driving.");
                    {
                        static const char* objectNames[] = { "Camera only", "Camera + moving things",
                                                             "Camera + everything (old)", "Camera + small movements" };
                        int objectMode = std::clamp(st.objectMotion, 0, 3);
                        ImGui::PushItemWidth(220.0f * menuResScale);
                        if (ImGui::Combo("Object motion", &objectMode, objectNames, 4) && objectMode != st.objectMotion)
                            PerEyeDlss_Dx12::RequestSameEyeObjectMotion(objectMode);
                        ImGui::PopItemWidth();
                        HelpMarker("Per eye only. Camera only (default): DLSS gets the fitted camera motion alone; people, "
                                   "cars and foliage get the motion of the world behind them. Camera + moving things: what "
                                   "the game's vectors show beyond the camera is added where it is more than a few tenths "
                                   "of a pixel (Far Cry 5's setting). Camera + everything: added from a fiftieth of a pixel "
                                   "up, the older setting (warped in one game). Camera + small movements: the "
                                   "game's motion averaged over 3x3 pixels, added from a few hundredths of a pixel up -- "
                                   "for small animations and distant moving things, a test. Takes effect at once.");
                    }
                    {
                        static const char* maskNames[] = { "Off", "Moving things", "Small movements" };
                        int maskMode = std::clamp(st.biasMask, 0, 2);
                        ImGui::PushItemWidth(220.0f * menuResScale);
                        if (ImGui::Combo("DLSS trusts the current frame on", &maskMode, maskNames, 3) &&
                            maskMode != st.biasMask)
                            PerEyeDlss_Dx12::RequestSameEyeBiasMask(maskMode);
                        ImGui::PopItemWidth();
                        HelpMarker("Per eye only. DLSS has an input that tells it, per pixel, to trust the current frame over "
                                   "its history. Off: not used. Moving things / Small movements: the mask is made where the "
                                   "game's vectors disagree with the camera, with the same two sensitivities as the object "
                                   "motion choices. Less smear on people and faces, some aliasing on them in exchange. A test; "
                                   "takes effect at once.");
                    }
                    bool vectorsSameEye = st.vectorsSameEye;
                    if (ImGui::Checkbox("Vectors are same-eye already (DL(S)SS on)", &vectorsSameEye))
                        PerEyeDlss_Dx12::RequestVectorsSameEye(vectorsSameEye);
                    HelpMarker("Tick when R.E.A.L. VR's DL(S)SS is on: "
                               "nothing of ours is fitted or composed, and NR, the crop and the periphery read them as they "
                               "are. Untick when it is off: the game's vectors point at the other eye and ours are made "
                               "from them. Wrong either way makes the motion wrong. Takes effect at once.");
                    {
                        static const char* depthNames[] = { "Auto", "Hardware", "Linear (1/depth)" };
                        int depthMode = std::clamp(st.fitDepth, 0, 2);
                        ImGui::PushItemWidth(160.0f * menuResScale);
                        if (ImGui::Combo("Fit depth", &depthMode, depthNames, 3) && depthMode != st.fitDepth)
                            PerEyeDlss_Dx12::RequestFitDepth(depthMode);
                        ImGui::PopItemWidth();
                        HelpMarker("Per eye only. The depth the same-eye fit reads. Auto: linear when the game tells Ray "
                                   "Reconstruction its depth is linear (Hogwarts Legacy), otherwise the hardware depth "
                                   "buffer. Change it only to compare.");
                    }
                    ImGui::SeparatorText("Foveated SR");
                    ImGui::PushItemWidth(100.0f * menuResScale);
                    int outerIndex = presetIndex(st.outerPreset);
                    if (ImGui::Combo("Full pass preset (= DLSS Override Preset)", &outerIndex, presetNames, 6) &&
                        presetValues[outerIndex] != st.outerPreset)
                        PerEyeDlss_Dx12::RequestOuterPreset(presetValues[outerIndex]);
                    HelpMarker("The GAME's full DLSS pass (both eyes), NOT the cheap periphery: this is the very same "
                               "setting as Override Preset in the DLSS settings, so changing it here changes the main "
                               "DLSS preset (it was once read as the periphery's and left the main "
                               "pass at E). The cheap periphery has its own 'Periphery preset' below; the crop has 'Crop "
                               "preset'. Rebuilt live, a short hitch. Default = the game's own (Crimson Desert asks for M).");
                    ImGui::PopItemWidth();
                    bool copyInputs = st.copyInputsNow;
                    if (ImGui::Checkbox("Crop reads copies", &copyInputs))
                        PerEyeDlss_Dx12::RequestFoveaCopyInputs(copyInputs);
                    ImGui::SameLine();
                    ImGui::TextDisabled(st.copyInputsByHand         ? "(set by hand)"
                                        : st.copiesSwappedBelow     ? "(automatic: DL(S)SS seen)"
                                                                    : "(automatic)");
                    HelpMarker("On: the crop's colour, motion and depth are copied into small textures first, so its DLSS "
                               "reads them like any ordinary call. Off: it reads the full inputs through DLSS's input "
                               "offsets (subrects) -- presets K and J turned pixelated that way here.");
                    // The RR and the SR rows of a pair differ by one word: only the rows of the crop's own kind show
                    // ('RR crop: vectors' was once changed for an SR crop and did nothing).
                    if (st.cropRr)
                    {
                        static const char* rrVecNames[] = { "Ours (the fit)", "DL(S)SS's (full pass)" };
                        int rrVec = std::clamp(st.rrVectors, 0, 1);
                        ImGui::PushItemWidth(170.0f * menuResScale);
                        if (ImGui::Combo("RR crop: vectors", &rrVec, rrVecNames, 2))
                            Config::Instance()->UpscalerFoveaRrVectors = rrVec;
                        ImGui::PopItemWidth();
                        HelpMarker("Ray reconstruction only. Ours: the crop runs on the vectors of our own fit (NPCs smeared "
                                   "in Crimson Desert). DL(S)SS's: the crop takes its box from the vectors as they stand with "
                                   "DL(S)SS running, after the full pass or the cheap periphery of the same frame (cleared "
                                   "the NPC smear in Crimson Desert and Star Wars Outlaws). Needs DL(S)SS on; without it the "
                                   "crop keeps ours.");
                    }
                    if (st.cropRr)
                    {
                        bool direct = Config::Instance()->UpscalerFoveaRrDirect.value_or_default();
                        if (ImGui::Checkbox("RR crop past DL(S)SS (direct DLL)", &direct))
                            PerEyeDlss_Dx12::RequestFoveaDirect(true, direct);
                        HelpMarker("Ray reconstruction only. Live: the crop's feature is rebuilt (its history starts over). "
                                   "The crop's RR feature is made and run on nvngx_dlssd.dll's own entry through our "
                                   "forwarder, past the driver core's exports: needed while DL(S)SS runs; "
                                   "'DL(S)SS's (full pass)' above then works as meant. If the direct create fails, the crop "
                                   "is made through the core as before (log: 'RR crop: the DIRECT create').");
                    }
                    if (!st.cropRr)
                    {
                        bool direct = Config::Instance()->UpscalerFoveaSrDirect.value_or_default();
                        if (ImGui::Checkbox("SR crop past DL(S)SS (direct DLL)", &direct))
                            PerEyeDlss_Dx12::RequestFoveaDirect(false, direct);
                        HelpMarker("Super resolution only. Live: the crop's feature is rebuilt (its history starts over). "
                                   "The crop's DLSS feature is made and run on nvngx_dlss.dll's own entry through our "
                                   "forwarder, past the driver core's exports: needed while DL(S)SS runs. The "
                                   "cheap periphery does not. With 'SR crop: vectors (test)' = 'DL(S)SS's (full "
                                   "pass)' the crop takes the vectors as they stand after that pass. If the direct create fails, the crop is "
                                   "made through the core as before (log: 'SR crop: the DIRECT create').");
                    }
                    if (!st.cropRr)
                    {
                        // Index 0 auto (-1), 1 ours (0), 2 DL(S)SS's (1).
                        static const char* srVecNames[] = { "Auto (DL(S)SS's when seen)", "Ours (the fit)",
                                                            "DL(S)SS's (full pass)" };
                        int srVec = std::clamp(st.srVectors, -1, 1) + 1;
                        ImGui::PushItemWidth(170.0f * menuResScale);
                        if (ImGui::Combo("SR crop: vectors", &srVec, srVecNames, 3))
                        {
                            Config::Instance()->UpscalerFoveaSrVectors = srVec - 1;
                            LOG_INFO("Overlay: SR crop: vectors -> {}", srVecNames[srVec]);
                        }
                        ImGui::PopItemWidth();
                        ImGui::SameLine();
                        ImGui::TextUnformatted(st.srFromBelowNow ? "(now: DL(S)SS's)" : "(now: ours)");
                        HelpMarker("Super resolution only. Auto (default): DL(S)SS's while R.E.A.L. VR's DL(S)SS is seen "
                                   "at work on the full-size pass (no shimmer even in fast motion), else ours; "
                                   "our fit is skipped while nothing takes it. Ours: the crop runs on our own fit's same-eye "
                                   "vectors. DL(S)SS's: "
                                   "the crop takes its box from the vectors as they stand with DL(S)SS running, after "
                                   "the DLSS full pass or cheap periphery of the same frame. Needs a DLSS periphery: with "
                                   "'[Inputs] UpscalerFoveaPeripheryTaa = true' (our own TAA) no DLSS full pass runs and the "
                                   "crop keeps ours (log: 'vectors from below wanted, but'). The crop's history starts over "
                                   "at each change.");
                    }
                    ImGui::PushItemWidth(220.0f * menuResScale);
                    // The crop's offsets moved to the open part (Foveated SR).
                    bool flipJitter = st.taaFlipJitter;
                    if (ImGui::Checkbox("Flip jitter", &flipJitter))
                        PerEyeDlss_Dx12::RequestTaaFlipJitter(flipJitter);
                    HelpMarker("Diagnostic: if the periphery shimmers or crawls with the TAA on, try this. The right sign "
                               "is the one that looks stable.");
                    float taaBlend = st.taaBlend;
                    if (StepSlider("TAA blend", &taaBlend, 0.03f, 0.5f, 0.01f, "%.2f"))
                        PerEyeDlss_Dx12::RequestTaaBlend(taaBlend);
                    HelpMarker("Weight of the new frame in the TAA periphery. Lower: smoother and steadier, more trailing "
                               "on moving things. Higher: sharper response, more shimmer and noise.");
                    float taaScale = st.taaScale;
                    if (StepSlider("TAA resolution", &taaScale, 0.4f, 1.0f, 0.05f, "%.2f"))
                        PerEyeDlss_Dx12::RequestTaaScale(taaScale);
                    HelpMarker("Size the TAA periphery runs at, as a fraction of the eye; the result is enlarged to fill the "
                               "eye (Catmull-Rom). 1.00 = full size (as before). About the game's render ratio (0.59 at "
                               "Balanced) keeps its detail and costs less; lower is softer. Each change restarts its "
                               "history.");
                    ImGui::PopItemWidth();
                    if (st.leftParity >= 0)
                        ImGui::Text("Eyes: the left eye is on %s frames", st.leftParity == 0 ? "even" : "odd");
                    else
                        ImGui::TextDisabled("Eyes: not told apart yet (needs Per eye on)");
                    if (st.upscalerMs > 0)
                        ImGui::Text("GPU per frame: full pass %.2f ms, crop %.2f ms", st.upscalerMs, st.cropMs);
                    if (kEyeTrackingUi && st.gazeOn)
                    {
                        ImGui::SeparatorText("Eye tracking");
                        {
                            bool boxWhenStill = st.gazeBoxWhenStill;
                            if (ImGui::Checkbox("Gaze box with the outline while the crop stands still", &boxWhenStill))
                                PerEyeDlss_Dx12::RequestGazeBoxWhenStill(boxWhenStill);
                            HelpMarker("The green box at your raw gaze is drawn with the outline only while 'Crop follows gaze' "
                                       "is on. Tick this to see it with a still crop too: for checking the tracker and for the manual "
                                       "offsets below. Not saved.");
                        }
                        {
                            const char* sourceNames[] = { "Auto (OpenXR layer, else Varjo)", "Varjo's reader only",
                                                          "OpenXR layer only" };
                            int source = VarjoGaze::Source();
                            ImGui::PushItemWidth(260.0f);
                            if (ImGui::Combo("Gaze source", &source, sourceNames, 3))
                                Config::Instance()->UpscalerFoveaGazeSource = source;
                            ImGui::PopItemWidth();
                            HelpMarker("Where the eye tracking comes from. The OpenXR layer (xr_observer in the game's folder) "
                                       "reads it through XR_EXT_eye_gaze_interaction -- any headset whose runtime offers it, "
                                       "in each eye's own frame -- and also hands the eye's FOV over, which a game with DLSS "
                                       "SR alone needs to place the gaze. Varjo's reader is the Aero's own library. Auto "
                                       "takes the layer's gaze when it gives one and starts Varjo's reader only if none "
                                       "comes within ~30 s. Once a source has given a gaze it stays the source for the "
                                       "launch.");
                            const char* projectionNames[] = { "Centred, headset's scale", "Headset's FOV as located",
                                                              "Submitted FOV, scaled to the eye",
                                                              "Submitted FOV, centred and scaled" };
                            int projection =
                                std::clamp(Config::Instance()->UpscalerFoveaGazeProjection.value_or_default(), 0, 3);
                            ImGui::PushItemWidth(260.0f);
                            if (ImGui::Combo("Gaze projection", &projection, projectionNames, 4))
                                Config::Instance()->UpscalerFoveaGazeProjection = projection;
                            ImGui::PopItemWidth();
                            HelpMarker("Only without ray reconstruction (DLSS SR hands no projection): how the gaze and the "
                                       "yellow targets are mapped into the picture, from the FOV the OpenXR layer sees. The "
                                       "headset's own FOV is off-centre toward the outside of each eye; if the VR mod renders "
                                       "a centred picture and crops it, that offset must NOT be applied (in one game a straight "
                                       "gaze sat a third off centre toward the nose in each eye). The FOV the VR mod "
                                       "submits with its image is the truth about the picture, "
                                       "scaled from the submitted crop to the eye's full image. A box that sits a little "
                                       "low or high in BOTH eyes with every entry is the tracker's own bias: calibrate.");
                        }
                    }
                    {
                        // The crop's own settings: shown whatever the eye tracking is (they were inside its group).
                        {
                            const char* moveNames[] = { "Carry the history over", "Carry over, shift flipped (test)",
                                                        "Restart the history (test)" };
                            int moveMode = std::clamp(Config::Instance()->UpscalerFoveaMoveMode.value_or_default(), 0, 2);
                            ImGui::PushItemWidth(260.0f);
                            if (ImGui::Combo("Crop move", &moveMode, moveNames, 3))
                                Config::Instance()->UpscalerFoveaMoveMode = moveMode;
                            ImGui::PopItemWidth();
                            HelpMarker("What a move of the crop does to its DLSS history. Carrying it over shifts the "
                                       "motion vectors by the move so DLSS sees a pan. If objects shimmer WHILE the crop "
                                       "moves and settle once it is still, try the two tests in the same place: a flipped "
                                       "shift that cures it means the carry-over's sign is wrong for this game; a restart "
                                       "that cures it means the carry-over itself is at fault.");
                            {
                                const char* maskNames[] = { "None (test)", "The game's, copied at the crop",
                                                            "The game's as handed (wrong place, test)" };
                                int maskMode = std::clamp(Config::Instance()->UpscalerFoveaCropMask.value_or_default(), 0, 2);
                                ImGui::PushItemWidth(260.0f);
                                if (ImGui::Combo("Crop's bias mask", &maskMode, maskNames, 3))
                                    PerEyeDlss_Dx12::RequestFoveaCropMask(maskMode);
                                ImGui::PopItemWidth();
                                HelpMarker("Some games hand DLSS a full-frame mask of where to favour the current frame over "
                                           "its history (some D3D12 games do). The crop's DLSS used to read it at the top left "
                                           "of the full frame -- the wrong place. Copied at the crop: the mask's box, like the "
                                           "other inputs. None: no mask for the crop. As handed: the old way, for comparison. "
                                           "A change restarts the crop's history.");
                            }
                            if (kEyeTrackingUi)
                            {
                                bool byOffset = Config::Instance()->UpscalerFoveaShiftByOffset.value_or_default();
                                if (ImGui::Checkbox("Crop move through MV offset (test)", &byOffset))
                                    Config::Instance()->UpscalerFoveaShiftByOffset = byOffset;
                                HelpMarker("With DL(S)SS running a "
                                           "move carried over in the crop's vectors may not reach DLSS. On: the move goes into "
                                           "DLSS's motion-vector offset parameter instead. If "
                                           "the shimmer while the crop moves goes with this on, keep it; if nothing changes, "
                                           "DLSS ignores that parameter.");
                            }
                            bool placeExact = Config::Instance()->UpscalerFoveaPlaceExact.value_or_default();
                            if (ImGui::Checkbox("Crop sits on whole output pixels", &placeExact))
                                Config::Instance()->UpscalerFoveaPlaceExact = placeExact;
                            HelpMarker("The crop's render origin is chosen, within 3 pixels of where the gaze wants it, "
                                       "so that its output origin is a whole pixel. Off: the nearest origin, rounded -- "
                                       "up to half an output pixel of displacement of everything inside the crop, "
                                       "changing at every step of a glide (the shimmer on objects while the crop moves). "
                                       "Untick for an A/B in the same place.");
                        }
                    }
                    if (kEyeTrackingUi && st.gazeOn)
                    {
                        if (st.follow)
                            ImGui::Text("Crop moves: %llu (%llu carried over, %llu restarted)",
                                        (unsigned long long) st.followMoves, (unsigned long long) st.followShifts,
                                        (unsigned long long) st.followResets);
                        if (st.gazePlaced)
                            ImGui::Text("Looking at (%u, %u) of %ux%u, %s eye; the green box shows it with the outline",
                                        st.gazeX, st.gazeY, st.eyeW, st.eyeH,
                                        st.gazeEye == 0 ? "left" : st.gazeEye == 1 ? "right" : "both (eye not known yet)");
                        else
                            ImGui::TextDisabled("Not placed yet (needs the crop on, RR, and a valid gaze)");
                        bool fx = st.gazeFlipX, fy = st.gazeFlipY;
                        const bool changedX = ImGui::Checkbox("Flip gaze X", &fx);
                        ImGui::SameLine(0.0f, 16.0f);
                        const bool changedY = ImGui::Checkbox("Flip gaze Y", &fy);
                        if (changedX || changedY)
                            VarjoGaze::SetFlip(fx, fy);
                        HelpMarker("If the green box moves the wrong way when you look left/right (X) or up/down (Y), tick "
                                   "that flip. Diagnostic: the axis convention is the part most likely to be wrong.");
                        float yaw = VarjoGaze::OffsetYaw(), pitch = VarjoGaze::OffsetPitch();
                        ImGui::PushItemWidth(220.0f);
                        const bool movedYaw = StepSlider("Gaze offset right (deg)", &yaw, -5.0f, 5.0f, 0.05f, "%.2f");
                        const bool movedPitch = StepSlider("Gaze offset up (deg)", &pitch, -5.0f, 5.0f, 0.05f, "%.2f");
                        ImGui::PopItemWidth();
                        if (movedYaw || movedPitch)
                            VarjoGaze::SetOffset(yaw, pitch);
                        HelpMarker("Zero the cross like a scope: look steadily at a small object and move these until the "
                                   "cross sits on it. Negative right = move it left, positive up = move it up. - / + step by "
                                   "0.05 degrees (hold to repeat); Ctrl+click a slider to type a value. Saved with the ini ([Inputs] UpscalerFoveaGazeYaw / Pitch).");
                        if (ImGui::Button("Reset gaze offset"))
                            VarjoGaze::SetOffset(0.0f, 0.0f);
                        {
                            const char* names[4] = { "Gaze range up", "Gaze range down", "Gaze range left",
                                                     "Gaze range right" };
                            ImGui::PushItemWidth(220.0f);
                            for (int i = 0; i < 4; ++i)
                            {
                                float gain = VarjoGaze::Range(i);
                                if (StepSlider(names[i], &gain, 0.5f, 2.0f, 0.01f, "%.2f"))
                                    VarjoGaze::SetRange(i, gain);
                            }
                            ImGui::PopItemWidth();
                            HelpMarker("After the offset puts the cross right in the centre: look at something near one edge "
                                       "and raise that direction's range until the cross reaches it (below 1 if it "
                                       "overshoots). Each one stretches only its own side of the centre, so the centre "
                                       "stays zeroed. Saved with the ini.");
                            if (ImGui::Button("Reset gaze range"))
                                for (int i = 0; i < 4; ++i)
                                    VarjoGaze::SetRange(i, 1.0f);
                        }
                        float steady = VarjoGaze::Steadiness();
                        ImGui::PushItemWidth(220.0f);
                        if (StepSlider("Gaze steadiness", &steady, 0.0f, 1.0f, 0.05f, "%.2f"))
                            VarjoGaze::SetSteadiness(steady);
                        ImGui::PopItemWidth();
                        HelpMarker("How strongly the gaze is smoothed while your eyes hold still (a jump of the eyes always "
                                   "gets through quickly). Higher = a steadier cross, a little slower to settle after you "
                                   "look somewhere new. Turn it up to zero the offset, then find what feels right.");
                        bool gazeSnap = Config::Instance()->UpscalerFoveaGazeSnap.value_or_default();
                        if (ImGui::Checkbox("Gaze snaps to where the eyes land", &gazeSnap))
                            Config::Instance()->UpscalerFoveaGazeSnap = gazeSnap;
                        HelpMarker("After a saccade, once two samples agree on the new place, the smoothed gaze is put "
                                   "there at once instead of easing in over a few frames with a tail of small steps -- "
                                   "the crop arrived after the eye. Off = as before, the A/B. Live.");
                    }
                    if (EvalCapture_Dx12::Enabled())
                    {
                        const bool capturing = EvalCapture_Dx12::Busy();
                        ImGui::BeginDisabled(capturing);
                        if (ImGui::Button(capturing ? "Capturing..." : "Capture upscaler frames"))
                            EvalCapture_Dx12::Arm();
                        ImGui::EndDisabled();
                        HelpMarker("Diagnostic ([Inputs] UpscalerInputCapture): saves what the upscaler is given and gives "
                                   "back for 8 frames in a row to OptiCapture beside the game exe. Delay: [Inputs] "
                                   "CaptureDelaySeconds. About 1.5 GB in VR.");
                        const std::string status = EvalCapture_Dx12::Status();
                        if (!status.empty())
                            ImGui::TextUnformatted(status.c_str());
                    }
                    if (GiCapture_Dx12::Developer())
                    {
                        const bool capturing = GiCapture_Dx12::Busy();
                        ImGui::BeginDisabled(capturing);
                        if (ImGui::Button(capturing ? "Capturing GI buffers..." : "Capture GI buffers"))
                            GiCapture_Dx12::Arm();
                        ImGui::EndDisabled();
                        HelpMarker("Diagnostic ([Inputs] GiBufferCapture, needs FrameSplitProbe): saves the textures the "
                                   "game's passes write, for 2 frames, to OptiCapture beside the game exe. Delay: [Inputs] "
                                   "CaptureDelaySeconds. Up to 3 GB.");
                        const std::string status = GiCapture_Dx12::Status();
                        if (!status.empty())
                            ImGui::TextUnformatted(status.c_str());
                    }
                    if (Config::Instance()->FrameSplitProbe.value_or_default() && FrameSplit_Dx12::Enabled())
                    {
                        const auto fs = FrameSplit_Dx12::GetStatus();
                        if (ImGui::Button("Measure the game's GPU frame"))
                            FrameSplit_Dx12::Arm();
                        HelpMarker("Diagnostic ([Inputs] FrameSplitProbe): after 5 s -- close this overlay and play -- "
                                   "times the game's GPU work for 64 frames, split by kind (raster at which size, compute, "
                                   "ray tracing, copies, the upscaler with our passes), and logs it with the timeline of "
                                   "two frames. Costs a little while it runs, nothing otherwise.");
                        if (!fs.installed)
                            ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.2f, 1.0f), "Frame split: hooks not installed.");
                        else if (fs.phase == 1)
                            ImGui::Text("Frame split: starting in 5 s -- close the overlay");
                        else if (fs.phase >= 2)
                            ImGui::Text("Frame split: measuring...");
                        for (const auto& line : fs.lines)
                            ImGui::TextUnformatted(line.c_str());
                    }
                    ImGui::TreePop();
                }
                ImGui::Spacing();
            }

            if (DuniaTaa_Dx12::Active())
            {
                {
                    const auto mode = DuniaTaa_Dx12::GetUpscaleStatus();
                    const char* forced = mode.configuredMode == 2 ? "" : " (forced in ini)";
                    if (mode.activeMode == 1)
                        ImGui::Text("FC6 DLSS: upscaling %ux%u -> %ux%u%s", mode.createdInW, mode.createdInH,
                                    mode.createdOutW, mode.createdOutH, forced);
                    else if (mode.activeMode == 0)
                        ImGui::Text("FC6 DLSS: DLAA %ux%u, carrying NR%s", mode.createdOutW, mode.createdOutH, forced);
                    else if (mode.configuredMode == 1)
                        ImGui::TextDisabled("FC6 DLSS: upscaling, waiting for gameplay below 100%% scale%s", forced);
                    else if (mode.configuredMode == 0)
                        ImGui::TextDisabled("FC6 DLSS: DLAA, starts with Neural Rendering on%s", forced);
                    else
                        ImGui::TextDisabled("FC6 DLSS: automatic, picks a mode once gameplay starts");
                    HelpMarker("Chosen automatically when gameplay starts, from the game's Resolution Scale:\n"
                               "- Below 100%: DLSS replaces the game's own upscale to the eye resolution. Pair it with "
                               "Neural Rendering's 'Run before SR', which then runs at the smaller size for a large "
                               "performance gain.\n- At 100%: DLAA, used to carry Neural Rendering (with NR off there "
                               "is nothing to do).\nMenus render at full size, so nothing is decided there. The choice "
                               "follows a Resolution Scale change live in the TAA splice mode (a note appears while it "
                               "rebuilds).\n[Inputs] DuniaTaaQuality forces a mode: 0 DLAA, 1 upscaling.");
                }

                if (DuniaTaa_Dx12::Active())
                {
                    const auto status = DuniaTaa_Dx12::GetUpscaleStatus();
                    const ImVec4 warn(1.0f, 0.8f, 0.2f, 1.0f);
                    switch (status.problem)
                    {
                    case DuniaTaa_Dx12::UpscaleProblem::ScaleChanged:
                        ImGui::TextColored(warn, "Resolution scale changed (DLSS set up for %ux%u, now %ux%u).",
                                           status.createdInW, status.createdInH, status.seenInW, status.seenInH);
                        ImGui::TextColored(warn, "DLSS rebuilds for it in a moment; restart only if this stays.");
                        break;
                    case DuniaTaa_Dx12::UpscaleProblem::UpscaleStopped:
                        ImGui::TextColored(warn, "DLSS upscaling is not running right now. Normal in menus and");
                        ImGui::TextColored(warn, "loading screens; in gameplay after a Resolution Scale change, restart.");
                        break;
                    case DuniaTaa_Dx12::UpscaleProblem::ScaleNotReduced:
                        ImGui::TextColored(warn, "Upscaling has not started: it needs the game's Resolution Scale");
                        ImGui::TextColored(warn, "below 100%%. Menus render at full size, so this is normal there.");
                        break;
                    case DuniaTaa_Dx12::UpscaleProblem::RestartForMode:
                        ImGui::TextColored(warn, "The forced DLSS mode applies after restarting the game.");
                        break;
                    default:
                        break;
                    }
                }
            }

            if (DuniaTaa_Dx12::Active())
            {
                if (ImGui::Button("DLSS on / off"))
                    DuniaTaa_Dx12::ToggleStart();
                ImGui::SameLine();
                ImGui::Text("%s", DuniaTaa_Dx12::StartName());
                HelpMarker("Switches Far Cry 6 between the game's own picture and DLSS: the first switch creates the "
                           "DLSS feature (a few seconds' freeze), then DLSS replaces the game's temporal anti-aliasing "
                           "and its upscale, with per-eye motion. Switch again for the game's own picture.");

                // Live, unlike the stock DLSS preset dropdown, which waits for a feature re-create: this one asks
                // OptiScaler's staged change to rebuild the two per-eye features over their next three frames.
                static const char* presetNames[] = { "Default", "E", "F", "J", "K", "L", "M" };
                static const int presetValues[] = { 0, 5, 6, 10, 11, 12, 13 };
                int selected = 0;
                const int current = DuniaTaa_Dx12::DlssPreset();
                for (int i = 0; i < IM_ARRAYSIZE(presetValues); ++i)
                    if (presetValues[i] == current)
                        selected = i;
                if (ImGui::Combo("DLSS preset (live)", &selected, presetNames, IM_ARRAYSIZE(presetNames)))
                    DuniaTaa_Dx12::RequestDlssPreset(presetValues[selected]);
                HelpMarker("Changes the DLSS render preset while playing: each eye holds its last image for three "
                           "frames and the history starts again. Not saved unless you save the settings.");

                // Developer test switches: never saved, so they cannot leak into a shared ini.
                if (ImGui::TreeNode("Developer tests (not saved)"))
                {
                    bool fakeProbe = Config::Instance()->DuniaTaaTestFakeProbe.value_or_default();
                    if (ImGui::Checkbox("Fake a large probe pass", &fakeProbe))
                        Config::Instance()->DuniaTaaTestFakeProbe = fakeProbe;
                    HelpMarker("TEST ONLY. The size learners see every square target of 1024+ px as 4096x4096, as on "
                               "a large headset resolution where the game's probe pass outgrew the eye and F6 then did "
                               "nothing. Takes effect as targets are learned: turn it on before loading into the game "
                               "(or set [Inputs] DuniaTaaTestFakeProbe=true, which is read but never saved).");
                    bool squareEye = Config::Instance()->DuniaTaaTestSquareEye.value_or_default();
                    if (ImGui::Checkbox("Square-eye rules", &squareEye))
                        Config::Instance()->DuniaTaaTestSquareEye = squareEye;
                    HelpMarker("TEST ONLY. Applies the rules used for square-eye headsets on any headset: below 100% "
                               "Resolution Scale, the scaled scene is only accepted once the game's TAA pass has been "
                               "seen at that size. Turn it on before loading, then change the Resolution Scale.");
                    bool gpuTiming = Config::Instance()->DuniaTaaGpuTiming.value_or_default();
                    if (ImGui::Checkbox("GPU timing (log)", &gpuTiming))
                        Config::Instance()->DuniaTaaGpuTiming = gpuTiming;
                    HelpMarker("Measures what our passes cost on the GPU, per eye, and writes it to OptiScaler.log every "
                               "600 frames: motion reconstruction, DLSS itself, and the copy back into the game. Also "
                               "counts frames where descriptor tracking (heap mirroring) was on, which is expensive.");
                    if (const auto gt = DuniaTaa_Dx12::GetGpuTimes(); gt.on && gt.samples > 0)
                        ImGui::Text("Per eye: motion %.2f + DLSS %.2f + copy %.2f = %.2f ms; tracking on %u/%u frames",
                                    gt.motionMs, gt.dlssMs, gt.copyMs, gt.motionMs + gt.dlssMs + gt.copyMs,
                                    gt.mirrorPresents, gt.windowPresents);
                    else if (gt.on)
                        ImGui::TextDisabled("Per eye: no timed frames yet (needs DLSS on, F6)");
                    bool nrPerEye = DlssNr::PerEyeHistory();
                    if (ImGui::Checkbox("NR history per eye", &nrPerEye))
                        DlssNr::SetPerEyeHistory(nrPerEye);
                    HelpMarker("TEST ONLY, on by default. Neural Rendering keeps one history per eye, as DLSS does. "
                               "Untick to share one history between the eyes again (the old behaviour: the ground "
                               "swims with NR on, worse when aiming) for a side-by-side in one session. Both "
                               "histories restart on every change.");
                    ImGui::TreePop();
                }

                // Live tuning of how the game's object vectors are carried onto the camera motion, to find a value
                // per Resolution Scale. Collapsed and not saved: closing the game restores the
                // validated values. Each settled change is logged with the scale it was judged at.
                if (ImGui::TreeNode("Object motion tuning (test, not saved)"))
                {
                    float scale, floorPx, speed;
                    bool chain;
                    DuniaTaa_Dx12::GetObjectComposition(&scale, &floorPx, &speed, &chain);
                    bool changed = false, settled = false;
                    changed |= ImGui::SliderFloat("Residual scale", &scale, 0.0f, 3.0f, "x%.2f");
                    settled |= ImGui::IsItemDeactivatedAfterEdit();
                    HelpMarker("How much of the game's per-object motion is carried on top of the camera motion. 1 = as "
                               "validated. Too little: moving NPCs and foliage smear or ghost. Too much: they shimmer "
                               "or overshoot.");
                    changed |= ImGui::SliderFloat("Floor (render px)", &floorPx, 0.0f, 4.0f, "%.2f px");
                    settled |= ImGui::IsItemDeactivatedAfterEdit();
                    HelpMarker("Object motion below this is ignored, which hides the game's 8-bit vector steps on things "
                               "that are not really moving. 1 px = as validated. Higher: calmer static edges, but slow "
                               "movers (idle NPCs, swaying leaves) lose their motion. Lower: small movements kept, more "
                               "crawl. In render pixels, so the same value covers more of the eye at lower scales.");
                    changed |= ImGui::SliderFloat("Floor per px of camera motion", &speed, 0.0f, 0.01f, "%.4f");
                    settled |= ImGui::IsItemDeactivatedAfterEdit();
                    HelpMarker("Raises the floor while you turn your head, where the vector steps show most. 0 = as "
                               "validated; Days Gone used 0.0015.");
                    if (ImGui::Checkbox("Fast movers from the chained lookup", &chain))
                        changed = settled = true;
                    HelpMarker("Off = as validated. On: fast-moving objects take the two-hop chained vectors instead of "
                               "doubling one Present of motion.");
                    if (changed)
                        DuniaTaa_Dx12::SetObjectComposition(scale, floorPx, speed, chain, settled);
                    bool nearOnly = DuniaTaa_Dx12::ObjectNearOnly();
                    if (ImGui::Checkbox("Object motion only on the nearest surface", &nearOnly))
                        DuniaTaa_Dx12::SetObjectNearOnly(nearOnly);
                    HelpMarker("On (new default): along silhouettes the game's half-resolution motion carries the NEAR "
                               "object's movement onto the pixels behind it; those are no longer treated as moving. "
                               "Should remove the dotted edge shimmer (the yellow outlines in the view) so the residual "
                               "scale can stay at x1. Off = the older behaviour.");
                    bool bgReject = DuniaTaa_Dx12::ObjectBgReject();
                    if (ImGui::Checkbox("Ignore background motion on near edges", &bgReject))
                        DuniaTaa_Dx12::SetObjectBgReject(bgReject);
                    HelpMarker("On (new default): the reverse case. A thin near edge (palm fronds, roof lines, posts "
                               "against the sky) can carry the BACKGROUND's motion, which read as movement: the yellow "
                               "silhouettes in the view with the box above on or off. Off = the older "
                               "behaviour.");
                    int radius = DuniaTaa_Dx12::ObjectRadius();
                    if (ImGui::SliderInt("Edge search radius (px)", &radius, 1, 6))
                        DuniaTaa_Dx12::SetObjectRadius(radius);
                    HelpMarker("How far the two boxes above look for a nearer or farther surface. 3 = default (tested: "
                               "the yellow silhouette outlines in the view are gone at 3 and back at 2; moving objects "
                               "keep their motion). The game's motion reaches ~3 px past an edge.");
                    bool privateCopy = DuniaTaa_Dx12::PrivateCopyOn();
                    if (ImGui::Checkbox("Same-eye NPC motion (private copy, experimental)", &privateCopy))
                        DuniaTaa_Dx12::SetPrivateCopyOn(privateCopy);
                    int copies = DuniaTaa_Dx12::SameEyeCopies();
                    const char* copyNames[] = { "All three copies (validated)", "First copy only", "Second copy only",
                                                "Third copy only", "None" };
                    if (ImGui::Combo("Same-eye copies (F4 same-eye state)", &copies, copyNames, 5))
                        DuniaTaa_Dx12::SetSameEyeCopies(copies);
                    HelpMarker("For the in-place same-eye state (F4, diagnostics): which of the game's three per-frame "
                               "copies of the camera constants get the same-eye previous view. If one copy lights up "
                               "idle NPCs in the view WITHOUT the shadow (AO) shimmer, that copy alone is the fix.");
                    int scope = DuniaTaa_Dx12::PrivateCopyScope();
                    const char* scopes[] = { "G-buffer: motion at slot 0", "G-buffer: motion at any slot",
                                             "Every graphics pass", "G-buffer (R8G8 target), in place" };
                    if (ImGui::Combo("Private copy scope", &scope, scopes, 4))
                        DuniaTaa_Dx12::SetPrivateCopyScope(scope);
                    HelpMarker("Which passes get the same-eye copy. 'G-buffer (R8G8 target), in place' is the default: "
                               "the pass the pass map found writing NPC motion. The first three were earlier tests."
                               " Older note:"
                               " 'Any slot' was the default. 'Every graphics pass' is "
                               "like the old in-place patch, except compute passes keep the game's -- if AO does not "
                               "shimmer there, it is enough on its own.");
                    HelpMarker("Off by default, live. On: only the G-buffer gets the same eye's previous view, through "
                               "a rebuilt copy of its constant table, so NPC idle motion reaches DLSS while the game's "
                               "AO and temporal filters keep their own history. Check the log's 'private copy' lines. "
                               "Turn it off at once if anything looks wrong.");
                    if (ImGui::Button("Run private-copy feasibility probe (~5 s)"))
                        DuniaTaa_Dx12::ArmPrivateCopyProbe();
                    ImGui::SameLine();
                    ImGui::TextDisabled("%s", DuniaTaa_Dx12::PrivateCopyProbeState());
                    HelpMarker("A diagnostic for the NPC fix: records which passes read the game's per-view camera "
                               "constants, and which copy of them, for about four seconds of gameplay (the frame rate "
                               "drops meanwhile), then writes a verdict to the log: can the geometry get its own "
                               "copy without touching the ambient occlusion? Run it in gameplay with DLSS on, with "
                               "NPCs in view, and keep moving your head a little.");
                    if (ImGui::Button("Reset to validated"))
                        DuniaTaa_Dx12::SetObjectComposition(1.0f, 1.0f, 0.0f, false);
                    ImGui::SameLine();
                    if (ImGui::Button("Days Gone values"))
                        DuniaTaa_Dx12::SetObjectComposition(2.0f, 0.1f, 0.0015f, true);
                    ImGui::TextDisabled("Note the values you like at each Resolution Scale; the log records them too.");
                    bool view = DuniaTaa_Dx12::ResidualView();
                    if (ImGui::Checkbox("Show where object motion is added (view)", &view))
                        DuniaTaa_Dx12::SetResidualView(view);
                    HelpMarker("Replaces the picture with what the object motion is doing, over a dim grey copy of the "
                               "scene. RED = object motion being added (the gate let it through). GREEN = how far the "
                               "game says the pixel moved by itself, full at 4 px, whether or not it was added. "
                               "Look for: red specks on still walls or ground (noise getting through = shimmer); "
                               "yellow/red on a walking NPC or a car (real motion carried); green without red on a "
                               "mover (real motion thrown away). The sliders act on it live. Turn it off to play; "
                               "DLSS's history restarts.");
                    ImGui::TreePop();
                }
            }

            // Tuning and A/B controls only with [Inputs] DuniaTaaDiagnostics: the validated values are the defaults.
            if (DuniaTaa_Dx12::Active() && config->DuniaTaaDiagnostics.value_or_default())
            {
                ImGui::TextDisabled("Depth: %s (startup only, [Inputs] DuniaTaaDepthInverted)",
                                    config->DuniaTaaDepthInverted.value_or_default() ? "reversed-Z" : "normal");
                HelpMarker("Whether the game's depth buffer runs near=1 far=0. Read when the DLSS feature is created, "
                           "so it can only be changed in the ini before launch.");

                {
                    // The Days Gone composition presets that used to sit here (all judged no better) are
                    // now the live sliders in the everyday FC6 section, "Object motion tuning".
                    static const char* jitterNames[] = { "Off: DLSS told zero (old)", "Game's own, reported to DLSS",
                                                         "Own per-eye Halton (patched + reported)",
                                                         "TEST A: x20, not reported (image should SHAKE)",
                                                         "TEST B: x4, reported (image should stand STILL)" };
                    int jitterMode = DuniaTaa_Dx12::JitterMode();
                    if (ImGui::Combo("FC6 jitter (live)", &jitterMode, jitterNames, IM_ARRAYSIZE(jitterNames)))
                        DuniaTaa_Dx12::SetJitterMode(jitterMode);
                    HelpMarker("Needs F4 motion on. The picture DLSS gets is jittered by the game; 'Off' tells DLSS it is "
                               "not. The game's jitter has two phases, so with one eye per frame each eye never moves "
                               "between its samples; 'Own' gives each eye a Halton sequence. Values only, safe live; "
                               "the history restarts.");
                    {
                        static const char* signNames[] = { "+x +y (FC5's)", "-x -y", "+x -y", "-x +y" };
                        int sign = DuniaTaa_Dx12::JitterSign();
                        if (ImGui::Combo("Jitter sign to DLSS", &sign, signNames, IM_ARRAYSIZE(signNames)))
                            DuniaTaa_Dx12::SetJitterSign(sign);
                    }
                    {
                        static const char* baseNames[] = { "New (derived, 0.5 px per eye different)", "Old (as in the afternoon build)" };
                        int baseline = DuniaTaa_Dx12::BaselineOld();
                        if (ImGui::Combo("Object baseline (HALO A/B)", &baseline, baseNames, IM_ARRAYSIZE(baseNames)))
                            DuniaTaa_Dx12::SetBaselineOld(baseline);
                        HelpMarker("The only all-mode change between the afternoon build and this one that could shift edges "
                                   "per eye. Flip it while looking at halos, dirt edges and the hands; F4 on. Not saved.");
                    }
                }

                // Kept for future motion vector work, shown only with [Inputs] DuniaTaaDiagnostics=true.
                if (config->DuniaTaaDiagnostics.value_or_default())
                {
                    ImGui::SeparatorText("FC6 diagnostics");

                    int mvMode = config->DuniaTaaMotionVectors.value_or_default();
                    if (mvMode < 0 || mvMode > 4)
                        mvMode = 0;
                    const char* mvNames[] = { "Off (reset every frame)", "Game candidate (uv space)",
                                              "Game candidate (uv space, inverted)", "Game candidate (pixel space)",
                                              "Game candidate (pixel space, inverted)" };
                    if (ImGui::Combo("FC6 motion vectors", &mvMode, mvNames, IM_ARRAYSIZE(mvNames)))
                        config->DuniaTaaMotionVectors = mvMode;
                    HelpMarker("FC6 has no camera motion vector buffer: the game candidates are other data and make "
                               "DLSS noisy. Off is correct. The other modes remain for testing motion vectors built "
                               "some other way. Per-eye DLSS histories are only created when this is not Off at startup.");

                    bool preferUnorm = config->DuniaTaaMvPreferUnorm.value_or_default();
                    if (ImGui::Checkbox("Prefer UNORM motion candidate", &preferUnorm))
                        config->DuniaTaaMvPreferUnorm = preferUnorm;
                    HelpMarker("Which of the two same-sized candidates the game modes read.");

                    int upscaleView = config->DuniaTaaUpscaleView.value_or_default();
                    if (upscaleView < 0 || upscaleView > 3)
                        upscaleView = 0;
                    const char* upscaleViewNames[] = { "DLSS", "Game upscale", "Split: DLSS left, game right",
                                                       "Motion vectors" };
                    if (ImGui::Combo("FC6 upscale A/B", &upscaleView, upscaleViewNames, IM_ARRAYSIZE(upscaleViewNames)))
                        config->DuniaTaaUpscaleView = upscaleView;
                    HelpMarker("Needs DLSS upscaling running. Game upscale skips DLSS, so framerate compares too. Split "
                               "shows DLSS on the left half of each eye and the game's upscale on the right. With NR "
                               "before SR both halves get NR. Motion vectors draws what DLSS receives: red right, green "
                               "left, blue vertical, full at 8 pixels; held still it should be near black.");

                    if (ImGui::Button("Trace one frame"))
                        config->DuniaTaaTraceFrame = true;
                    HelpMarker("Writes every render target bind and copy for the next two frames (one per eye) to "
                               "OptiScaler.log. Logging only.");

                    const bool capturing = FrameCapture_Dx12::Busy();
                    ImGui::BeginDisabled(capturing);
                    if (ImGui::Button(capturing ? "Capturing..." : "Capture frames"))
                        FrameCapture_Dx12::Arm();
                    ImGui::EndDisabled();
                    HelpMarker("Saves the finished eye image and every motion vector candidate for the next four eye "
                               "frames to OptiCapture\<time> beside the game exe, for the offline scripts in "
                               "tools\capture. Set [Inputs] CaptureDelaySeconds so the overlay can be closed first. "
                               "Expect a hitch of a second or two and a few hundred MB per capture.");
                }
            }

            if (Fc5Taa_Dx11::Enabled())
            {
                if (ImGui::Button("Switch##fc5antialiasing"))
                    Fc5Taa_Dx11::CycleMode();
                ImGui::SameLine();
                ImGui::Text("FC5 antialiasing: %s", Fc5Taa_Dx11::ModeName());
                HelpMarker("The button switches between the game's TAA and DLSS per eye at the game's TAA pass, which carries "
                           "Neural Rendering (before the game's upscale below 100% Resolution Scale). The first switch "
                           "freezes the game for a few seconds. The Resolution Scale and R.E.A.L. VR's resolution can "
                           "be changed during play: DLSS is rebuilt for the new size once it has held for about a "
                           "second of gameplay (a hitch of a second or two; the game's own picture shows meanwhile). "
                           "100 = DLAA; 70 = Quality, which costs much less.");
                ImGui::TextUnformatted(Fc5Taa_Dx11::StatusText().c_str());
                {
                    // The preset both feature pairs are made with (OptiScaler's Render Presets Override; its own DLSS
                    // section does not show for these features). One selector for DLAA and every upscaling level: two
                    // let a whole A/B run on the one that did not apply (Days Gone).
                    static const char* presetNames[] = { "DLSS default", "E", "F", "J", "K", "L", "M" };
                    static const uint32_t presetValues[] = { 0, 5, 6, 10, 11, 12, 13 };
                    const bool overrideOn = config->RenderPresetOverride.value_or_default();
                    int preset = 0;
                    for (int i = 0; overrideOn && i < IM_ARRAYSIZE(presetValues); ++i)
                    {
                        if (presetValues[i] == config->RenderPresetDLAA.value_or_default())
                            preset = i;
                    }
                    ImGui::PushItemWidth(160.0f);
                    const bool presetChanged =
                        ImGui::Combo("DLSS preset", &preset, presetNames, IM_ARRAYSIZE(presetNames));
                    ImGui::PopItemWidth();
                    if (presetChanged)
                    {
                        const uint32_t value = presetValues[preset];
                        config->RenderPresetOverride = value != 0;
                        config->RenderPresetDLAA = value;
                        config->RenderPresetUltraQuality = value;
                        config->RenderPresetQuality = value;
                        config->RenderPresetBalanced = value;
                        config->RenderPresetPerformance = value;
                        config->RenderPresetUltraPerformance = value;
                        LOG_INFO("FC5 overlay: DLSS preset {} ({}) for DLAA and upscaling; rebuilding", value,
                                 presetNames[preset]);
                        Fc5Taa_Dx11::RequestRebuild();
                    }
                    HelpMarker("The DLSS model, for 100% and below. Changing it rebuilds DLSS at once (a hitch of a "
                               "second or two) and is saved. M is the newer transformer model: the clearest, and the "
                               "most expensive. K is the first transformer model (usually DLSS's own default). E and F "
                               "are the older models: cheaper, softer. The log line 'requested preset' confirms what "
                               "each new pair got.");
                    if (ImGui::Button("Rebuild DLSS now"))
                        Fc5Taa_Dx11::RequestRebuild();
                }
            }

            if (Fc5Dlss_Dx11::Enabled())
            {
                const auto fc5 = Fc5Dlss_Dx11::GetInfo();
                switch (fc5.status)
                {
                case Fc5Dlss_Dx11::Status::Dlaa:
                    ImGui::Text("FC5 DLSS: DLAA %ux%u, carrying NR", fc5.outW, fc5.outH);
                    break;
                case Fc5Dlss_Dx11::Status::Upscaling:
                    ImGui::Text("FC5 DLSS: upscaling %ux%u -> %ux%u", fc5.inW, fc5.inH, fc5.outW, fc5.outH);
                    break;
                case Fc5Dlss_Dx11::Status::RenderSizeNr:
                    ImGui::Text("FC5 DLSS: NR at %ux%u, game upscales to %ux%u", fc5.inW, fc5.inH, fc5.outW, fc5.outH);
                    break;
                case Fc5Dlss_Dx11::Status::CreateFailed:
                    ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "FC5 DLSS: could not be created (see log)");
                    break;
                case Fc5Dlss_Dx11::Status::ScaleChanged:
                    ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.2f, 1.0f),
                                       "FC5 DLSS: Resolution Scale changed. Restart the game to match.");
                    break;
                default:
                    ImGui::TextDisabled("FC5 DLSS: starts once gameplay is drawn");
                    break;
                }
                HelpMarker("DLSS runs on the game's final upscale, after its TAA, and carries Neural Rendering. At 100% "
                           "Resolution Scale it runs as DLAA. Below 100%, NR runs at the render size and the game's own "
                           "upscale does the rest ([Inputs] DuniaTaaQuality=1 uses DLSS upscaling instead; pair it with "
                           "'Run before SR'). Set up once, a few seconds into gameplay; changing the Resolution Scale "
                           "needs a restart.");
            }

            // The stale-eye OpenXR layer in Far Cry 5: both parts of its controls here, in place.
            if (Fc5Taa_Dx11::Enabled() && StaleEyeShare::Enabled())
            {
                ImGui::SeparatorText("Stale-eye warp (alternate eyes)");
                XrObserverStaleSettingsV1 s {};
                if (StaleEyeBegin(s))
                {
                    StaleEyeMenu::Answer asked;
                    StaleEyeMenu::DrawSimple(s, asked, false);
                    StaleEyeMenu::DrawJunk(s, asked, false, false);
                    StaleEyeEnd(s, asked);
                    XrObserverStaleStatusV1 st {};
                    if (StaleEyeShare::LayerStatus(&st))
                    {
                        StaleEyeMenu::DrawStatusSimple(s, st, asked);
                        StaleEyeMenu::DrawStatusJunk(s, st);
                    }
                }
            }

            if (TaaDlaa_Dx11::Enabled())
            {
                if (ImGui::Button("Switch##taadlaamode"))
                    TaaDlaa_Dx11::CycleMode();
                ImGui::SameLine();
                ImGui::Text("Mode: %s", TaaDlaa_Dx11::ModeName());
                HelpMarker("Render Scale and R.E.A.L. VR's resolution can be changed during play: DLSS is rebuilt "
                           "for the new size once it has held for about half a second (a hitch of a second or so; "
                           "the game's own picture shows meanwhile). 100 = DLAA.");
                {
                    // The presets the two feature pairs are made with (OptiScaler's Render Presets Override, set here
                    // live and not saved; its own DLSS section is hidden because these features are not its current
                    // one). A change rebuilds both pairs (a hitch of a second or two).
                    static const char* presetNames[] = { "DLSS default", "E", "F", "J", "K", "L", "M" };
                    static const uint32_t presetValues[] = { 0, 5, 6, 10, 11, 12, 13 };
                    auto* config = Config::Instance();
                    const auto indexOf = [&](uint32_t value)
                    {
                        for (int i = 0; i < IM_ARRAYSIZE(presetValues); ++i)
                        {
                            if (presetValues[i] == value)
                                return i;
                        }
                        return 0;
                    };
                    const bool overrideOn = config->RenderPresetOverride.value_or_default();
                    int preset = overrideOn ? indexOf(config->RenderPresetDLAA.value_or_default()) : 0;
                    ImGui::PushItemWidth(160.0f);
                    const bool changed = ImGui::Combo("DLSS preset", &preset, presetNames, IM_ARRAYSIZE(presetNames));
                    ImGui::PopItemWidth();
                    ImGui::TextWrapped("Recommended: F (the default in this ini) -- least smearing on NPCs and the cheapest. M: a "
                                       "clearer picture, also little smearing, but costs more. K (usually DLSS's own default) "
                                       "smears NPCs the most.");
                    if (changed)
                    {
                        // One preset for DLAA and every upscaling level: two selectors let a whole A/B run on the one
                        // that did not apply (six rebuilds, all at DLSS default).
                        const uint32_t value = presetValues[preset];
                        config->RenderPresetOverride.set_volatile_value(true);
                        config->RenderPresetDLAA.set_volatile_value(value);
                        config->RenderPresetUltraQuality.set_volatile_value(value);
                        config->RenderPresetQuality.set_volatile_value(value);
                        config->RenderPresetBalanced.set_volatile_value(value);
                        config->RenderPresetPerformance.set_volatile_value(value);
                        config->RenderPresetUltraPerformance.set_volatile_value(value);
                        LOG_INFO("Days Gone overlay: DLSS preset {} ({}) for DLAA and upscaling; rebuilding", value,
                                 presetNames[preset]);
                        TaaDlaa_Dx11::RequestRebuild();
                    }
                    HelpMarker("The DLSS model both feature pairs are made with (100% and below). Changing it rebuilds DLSS at once (a "
                               "hitch). E and F are the older models (deprecated by NVIDIA: a future DLSS DLL may drop "
                               "them; M is the fallback); F is the only preset that honours a bias mask (DLSS guide "
                               "3.15); K is the first transformer model, M the newer one. Live, not saved; the log line 'requested preset' "
                               "confirms what each new pair got.");
                }
                if (ImGui::Button("Rebuild DLSS now"))
                    TaaDlaa_Dx11::RequestRebuild();
                {
                    float sharpness = TaaDlaa_Dx11::Sharpness();
                    ImGui::PushItemWidth(220.0f);
                    if (StepSlider("Sharpening", &sharpness, 0.0f, 2.0f, 0.05f, "%.2f"))
                        TaaDlaa_Dx11::SetSharpness(sharpness);
                    int radius = TaaDlaa_Dx11::SharpenRadius() - 1;
                    const char* radiusNames[] = { "1 px", "2 px" };
                    if (ImGui::Combo("Sharpen radius", &radius, radiusNames, IM_ARRAYSIZE(radiusNames)))
                        TaaDlaa_Dx11::SetSharpenRadius(radius + 1);
                    ImGui::PopItemWidth();
                    HelpMarker("Our own sharpening (an unsharp mask) on the DLSS output, before the game's grain and HUD. "
                               "The game's TAA and its upscale draw sharpen and DLSS does not. OptiScaler's RCAS was too "
                               "subtle to see in the headset; a 2 px radius survives the headset's resampling better. "
                               "0 = off. Live, not saved; the log records each change.");
                }
                HelpMarker("Frees and re-makes both DLSS feature pairs with the current settings -- e.g. after picking a "
                           "preset in OptiScaler's own DLSS section (Render Presets Override: DLAA for 100%, Quality for "
                           "the upscaling). A hitch of a second or two. The log line 'requested preset' names the preset "
                           "each new pair was made with.");
                ImGui::SeparatorText("Motion");
                if (TaaMotion_Dx11::Enabled())
                {
                    int motionMode = TaaMotion_Dx11::MotionMode();
                    const char* motionNames[] = { "Matrix hybrid", "Matrix, halo as static (validated)",
                                                  "Chained through the other eye (old)",
                                                  "Matrix + chained object motion (new)" };
                    if (ImGui::Combo("Same-eye motion", &motionMode, motionNames, IM_ARRAYSIZE(motionNames)))
                        TaaMotion_Dx11::SetMotionMode(motionMode);
                    HelpMarker("The same states as F4 with diagnostics. 'Chained' is the old motion: blurrier, but "
                               "the reference NPCs smear least with (it shimmers around objects). 'Matrix + chained object "
                               "motion' (new): the camera part exact, a character's own motion summed over both Presents "
                               "where the same surface is found (dark green in the classification view).");
                    float ownScale = TaaMotion_Dx11::OwnMotionScale();
                    ImGui::PushItemWidth(220.0f);
                    if (StepSlider("Object motion", &ownScale, 0.0f, 2.0f, 0.1f, "x%.1f"))
                        TaaMotion_Dx11::SetOwnMotionScale(ownScale);
                    ImGui::PopItemWidth();
                    HelpMarker("How much of the game's per-object motion is added on top of the camera (matrix) "
                               "motion. x1 = as validated. x0 = camera motion only: in Far Cry 6 this term alone "
                               "caused the swim and smear, and only when the GAME moved the camera (stick, riding), "
                               "never head motion. Test for the dark trail: ride at x1, then at x0; then compare head-"
                               "only movement against steering. x0 brings back smear on NPC faces and hair (idle "
                               "animation) -- a test, not the fix. Live, not saved; the log records each value.");
                    {
                        float knee = TaaMotion_Dx11::TuningValue(TaaMotion_Dx11::KneeStart);
                        float rise = TaaMotion_Dx11::TuningValue(TaaMotion_Dx11::KneeRise);
                        float ratio = TaaMotion_Dx11::TuningValue(TaaMotion_Dx11::KneeRatio);
                        float chainStart = TaaMotion_Dx11::TuningValue(TaaMotion_Dx11::ChainStart);
                        float chainEnd = TaaMotion_Dx11::TuningValue(TaaMotion_Dx11::ChainEnd);
                        ImGui::PushItemWidth(220.0f);
                        if (StepSlider("Knee start (px)", &knee, 0.0f, 0.5f, 0.01f, "%.2f"))
                            TaaMotion_Dx11::SetTuningValue(TaaMotion_Dx11::KneeStart, knee);
                        if (StepSlider("Knee rise per px of camera", &rise, 0.0f, 0.005f, 0.0005f, "%.4f"))
                            TaaMotion_Dx11::SetTuningValue(TaaMotion_Dx11::KneeRise, rise);
                        if (StepSlider("Knee full at", &ratio, 1.5f, 6.0f, 0.5f, "x%.1f"))
                            TaaMotion_Dx11::SetTuningValue(TaaMotion_Dx11::KneeRatio, ratio);
                        if (StepSlider("Chained from (px)", &chainStart, 0.05f, 4.0f, 0.05f, "%.2f"))
                            TaaMotion_Dx11::SetTuningValue(TaaMotion_Dx11::ChainStart, chainStart);
                        if (StepSlider("Chained fully at (px)", &chainEnd, 0.1f, 6.0f, 0.1f, "%.1f"))
                            TaaMotion_Dx11::SetTuningValue(TaaMotion_Dx11::ChainEnd, chainEnd);
                        float busy = TaaMotion_Dx11::TuningValue(TaaMotion_Dx11::BusyScale);
                        if (StepSlider("Object motion on foliage", &busy, 0.0f, 1.0f, 0.05f, "x%.2f"))
                            TaaMotion_Dx11::SetTuningValue(TaaMotion_Dx11::BusyScale, busy);
                        ImGui::PopItemWidth();
                        HelpMarker("Object motion x0.1 removed the last shimmer on grass and foliage but brought NPC smear "
                                   "back. This scales the object motion only where the depth is busy "
                                   "(most neighbours at another depth: grass, leaves, branches); smooth surfaces such as "
                                   "an NPC's body keep all of it. Try x0.1 with Object motion x1. Orange in the "
                                   "classification view = damped here: it should cover vegetation, not NPC bodies.");
                        HelpMarker("Idle NPC smear: the game's small per-object motion is only added above "
                                   "the knee (default 0.1 px, rising with camera motion, full at x3 of it); idle motion "
                                   "sits mostly below it. Try knee 0.02, rise 0, and see whether idle faces and bodies "
                                   "smear less. Chained from / fully at (default 1 / 2 px): from where object pixels "
                                   "switch to the old chained motion ('Chained' looked better on NPCs but shimmers "
                                   "around objects); try 0.1 / 0.3 so moving pixels are chained while the static world "
                                   "and the halo around movers keep the matrix. Live, not saved; the log records each set.");
                    }
                    bool nrPerEye = DlssNr::PerEyeHistory();
                    if (ImGui::Checkbox("NR history per eye", &nrPerEye))
                        DlssNr::SetPerEyeHistory(nrPerEye);
                    HelpMarker("On by default (new). Neural Rendering keeps one history per eye, as DLSS does; before "
                               "this build Days Gone's NR shared one between the eyes, which in Far Cry 6 made the "
                               "ground swim with NR on. Untick for the old behaviour, to compare in one session (NR "
                               "on, F9). Both histories restart on every change. Not saved.");
                    int maskMode = TaaMotion_Dx11::MaskMode();
                    const char* maskNames[] = { "Off", "Covers (depth)", "Covers + moved shadows",
                                                "TEST: whole frame" };
                    if (ImGui::Combo("DLSS hint mask", &maskMode, maskNames, IM_ARRAYSIZE(maskNames)))
                        TaaMotion_Dx11::SetMaskMode(maskMode);
                    HelpMarker("OFF by default: DLSS 310.8 ignored it -- TEST (whole frame) changed nothing, "
                               "and the riding trail shows in the game's own TAA too. "
                               "Tells DLSS where to trust the current frame over its history. Covers: ground that was "
                               "hidden behind the bike two Presents ago (cyan in the classification view). Moved "
                               "shadows (default): also where the history is much darker than now -- the bike's shadow "
                               "and AO travel with it over ground (white). TEST: marks the whole frame -- if DLSS follows "
                               "the mask, the picture turns aliased and jittery; if nothing changes, DLSS ignores it and "
                               "this whole approach is moot.");
                    const bool biasOn = maskMode != 0;
                    if (biasOn)
                    {
                        float strength = TaaMotion_Dx11::BiasStrength();
                        ImGui::PushItemWidth(220.0f);
                        if (StepSlider("Hint strength", &strength, 0.1f, 1.0f, 0.1f, "%.1f"))
                            TaaMotion_Dx11::SetBiasStrength(strength);
                        float nearer = TaaMotion_Dx11::BiasNearer();
                        if (StepSlider("Cover threshold", &nearer, 1.02f, 2.0f, 0.02f, "x%.2f"))
                            TaaMotion_Dx11::SetBiasNearer(nearer);
                        ImGui::PopItemWidth();
                        float drop = TaaMotion_Dx11::ShadowDrop();
                        ImGui::PushItemWidth(220.0f);
                        if (StepSlider("Shadow threshold", &drop, 0.01f, 0.2f, 0.01f, "%.2f"))
                            TaaMotion_Dx11::SetShadowDrop(drop);
                        ImGui::PopItemWidth();
                        HelpMarker("Strength: how strongly DLSS favours the current frame there (1 = fully). Cover "
                                   "threshold: how much nearer the surface in the history must be to count (x1.15 = "
                                   "about 13% closer). Shadow threshold: how much darker the history must be (lower "
                                   "catches fainter shadows and AO, but flags more ground: watch the white in the "
                                   "classification view).");
                    }
                    if (ImGui::Button("Reset motion to validated"))
                    {
                        TaaMotion_Dx11::SetMotionMode(1);
                        TaaMotion_Dx11::SetOwnMotionScale(1.0f);
                        TaaMotion_Dx11::ResetTuning();
                    }
                    ImGui::Text("View (F8 with diagnostics): %s", TaaMotion_Dx11::ViewName());
                    if (ImGui::Button("Next motion view"))
                        TaaMotion_Dx11::CycleView();
                    HelpMarker("Replaces the picture with the motion: off, decoded, composed, classification (grey "
                               "chained, blue matrix, green object motion added, amber blending, red chained, magenta "
                               "halo, cyan disocclusion hint).");
                }
                else
                    ImGui::TextDisabled("Motion module off ([Inputs] TaaDlaa or TaaMotionCheck)");
            }

            if (Fc5Trace_Dx11::Installed() && config->DuniaTaaDiagnostics.value_or_default())
            {
                ImGui::SeparatorText("FC5 diagnostics");
                if (ImGui::Button("Trace FC5 frame"))
                    config->DuniaTaaTraceFrame = true;
                HelpMarker("Starts 3 seconds after clicking, so the overlay can be closed first. Writes every render "
                           "target bind, large texture read, copy and compute dispatch for the next two frames (one "
                           "per eye) to OptiScaler.log. Logging only.");
            }

            if (DuniaTaa_Dx12::Active())
            {
                ImGui::SeparatorText("Resolution Scale");
                ImGui::TextWrapped("Set in the game's video options; DLSS follows whatever it renders. 100 = DLAA. "
                                   "67-70 = Quality (validated). 60 = Balanced, the lowest recommended. 50 = Performance: "
                                   "works, but edges of hands, weapons and nearby objects shimmer more (half the samples, "
                                   "and the game's object-motion step doubles). Changing it during play rebuilds DLSS "
                                   "for the new size within a second or two of gameplay (the game's own picture shows "
                                   "meanwhile).");
            }
        }
    }

    // DLSS Neural Rendering -----------------------------
    ImGui::Spacing();
    if (auto ch = ScopedCollapsingHeader("DLSS Neural Rendering"); ch.IsHeaderOpen())
    {
        ScopedIndent indent {};
        ImGui::Spacing();

        bool enabled = config->DlssNrEnabled.value_or_default();
        if (ImGui::Checkbox("Enable Neural Rendering", &enabled))
            config->DlssNrEnabled = enabled;

        HelpMarker("Enhance lighting and material appearance with the NR model. Placement selects before or after upscaling.\nRequires nvngx_dlssnr.dll plus the included nvngx.dll_dlssnr.dll helper.");

        bool finishedPicture = config->DlssNrFinishedPicture.value_or_default();
        if (ImGui::Checkbox("Apply NR to the finished picture", &finishedPicture))
        {
            config->DlssNrFinishedPicture = finishedPicture;
            DlssNr::RetryAfterFailure();
        }
        HelpMarker("Apply NR after the game has finished its lighting and effects. This may help with green noise.\nWorks with frame generation on or off in native DirectX 12 games, including SDR, HDR10 and scRGB.\nIt can also change the HUD and menus. Enable Run the model before Super Resolution to generate the changes earlier.");
        if (finishedPicture && enabled)
        {
            const auto feature = State::Instance().currentFeature;
            if (feature && (feature->Api() != API::DX12 || feature->IsWithDx12()))
                ImGui::TextWrapped("This option needs a native DirectX 12 game.");
            else
                ImGui::TextWrapped("%s", DlssNr::FinishedPictureStatus().c_str());
        }

        {
            bool perEye = DlssNr::PerEyeHistory();
            if (ImGui::Checkbox("NR history per eye (VR)", &perEye))
                DlssNr::SetPerEyeHistory(perEye);
            HelpMarker("Alternate-eye VR shows one eye per frame. On: NR keeps one history per eye, and on the finished "
                       "picture takes same-eye motion from the per-eye path ([Inputs] UpscalerPerEye) while DLSS stays "
                       "the game's own. Off: one history shared by both eyes (the old behaviour). Live, not saved.");
        }

        bool beforeSr = config->DlssNrRunBeforeSr.value_or_default() ||
                        (finishedPicture && config->DlssNrDeferredDlss.value_or_default());
        const auto activeFeature = State::Instance().currentFeature;
        const bool rayReconstruction = activeFeature && activeFeature->GetUpscalerType() == Upscaler::DLSSD;
        const bool deferredActive = !finishedPicture && config->DlssNrDeferredDlss.value_or_default() && !rayReconstruction;
        if (deferredActive)
            ImGui::BeginDisabled();
        if (ImGui::Checkbox(finishedPicture ? "Run the model before Super Resolution" : "Apply before Super Resolution", &beforeSr))
        {
            config->DlssNrRunBeforeSr = beforeSr;
            if (finishedPicture) config->DlssNrDeferredDlss = false;
        }
        if (deferredActive)
            ImGui::EndDisabled();

        HelpMarker(finishedPicture ? "Run the model at the smaller input size, upscale its changes with DLSS, then apply them to the finished picture.\nExperimental: the colour transfer is approximate and may look different. Requires DLSS SR; does not support RR." :
            "On: apply NR before SR or combined RR+SR. Off: apply it afterward.\nBefore RR is experimental. Unsupported input layouts fall back after upscaling.");

        if (!finishedPicture)
        {
            bool residualAcrossRr = config->DlssNrResidualAcrossRr.value_or_default();
            ImGui::BeginDisabled(deferredActive || !beforeSr);
            if (ImGui::Checkbox("Carry the pre-SR edit across RR (experimental)", &residualAcrossRr))
                config->DlssNrResidualAcrossRr = residualAcrossRr;
            ImGui::EndDisabled();
            HelpMarker("Only with Apply before Super Resolution on and the game's Ray Reconstruction active.\nRuns the model before SR but leaves the colour input untouched, then adds its edit back onto the RR+SR output so it survives RR's denoise.\nThe edit is carried as a motion-vector-reprojected temporal accumulator: the per-frame ray-trace noise averages out, the enhancement stays. Inert otherwise.");

            ImGui::BeginDisabled(deferredActive || !beforeSr || !residualAcrossRr);
            float residualBlend = config->DlssNrResidualAcrossRrBlend.value_or_default();
            if (ImGui::SliderFloat("Detail accumulation rate", &residualBlend, 0.01f, 1.0f, "%.2f"))
                config->DlssNrResidualAcrossRrBlend = std::clamp(residualBlend, 0.01f, 1.0f);
            ImGui::EndDisabled();
            HelpMarker("How fast the carried edit builds up. Lower = stabler but slower to appear; 1.0 = no accumulation (each frame's raw residual, which flickers). Default 0.08.");

        }
        bool deferredDlss = config->DlssNrDeferredDlss.value_or_default();
        if (!finishedPicture)
        {
            if (ImGui::Checkbox("Generate before SR, apply after SR (DLSS)", &deferredDlss))
                config->DlssNrDeferredDlss = deferredDlss;
            HelpMarker("Compute NR at input resolution, upscale its changes with DLSS, then apply them after SR.\nExperimental: may flicker and adds GPU cost. Requires DLSS on DX12 or its bridges; does not support RR.\nOverrides Apply before Super Resolution. Disable Hold frame, Compare and Debug view.");
            if (deferredDlss && rayReconstruction)
                ImGui::TextWrapped("Generate before / apply after is unavailable with RR. Apply before Super Resolution "
                                   "controls NR placement.");
            else if (deferredDlss)
                ImGui::TextWrapped("Residual DLSS: %s", DlssNr::DeferredDlssStatus().c_str());
            ImGui::BeginDisabled(finishedPicture || !deferredDlss || rayReconstruction);
            bool residualFg = config->DlssNrResidualFg.value_or_default();
            if (ImGui::Checkbox("NR every second frame (NVIDIA FG, experimental)", &residualFg))
                config->DlssNrResidualFg = residualFg;
            HelpMarker("Run NR every other rendered frame and use NVIDIA FG to interpolate its changes.\nRequires the option above. Adds one rendered frame of latency and may misalign effects or UI.\nIf motion vectors are unavailable, each NR result is reused for two frames.");
            bool approxCamera = config->DlssNrResidualFgApproxCamera.value_or_default();
            if (ImGui::Checkbox("Allow approximate FG camera guides (experimental)", &approxCamera))
                config->DlssNrResidualFgApproxCamera = approxCamera;
            HelpMarker("Use estimated camera data when the game does not provide it. May cause artifacts during camera movement.");
            ImGui::EndDisabled();

        }
        else if (beforeSr)
            ImGui::TextWrapped("Pre-SR changes: %s", DlssNr::DeferredDlssStatus().c_str());

        // The toggle can be bound to a key, and nobody would think to look for it under Keybinds
        // unless told. Dimmed, because it is a note rather than a setting.
        ImGui::TextDisabled("Set the NR toggle shortcut under Keybinds.");

        bool applyModel = config->DlssNrApplyModel.value_or_default();
        if (ImGui::Checkbox("Apply the model", &applyModel))
            config->DlssNrApplyModel = applyModel;

        HelpMarker("Show or hide the NR effect. The model still runs when hidden.\nDisable Enable Neural Rendering to stop its GPU cost.");

        // Either backend. The two keep separate state, and on a native Vulkan game the D3D12 side
        // is never touched -- so asking only that one reports "waiting for the upscaler" over a pass
        // that is demonstrably running.
        const bool vulkan = DlssNr::IsRunningVk();

        // Turning the pass off does not release the model, so the feature handle stays alive and
        // IsRunning keeps answering yes. Reporting a cost from that was wrong in the way that matters
        // most: the toggle is how anyone A/Bs this, so the one moment the number is read is the one
        // moment it describes the frame before last.
        if (!enabled)
        {
            ImGui::TextDisabled("NR off.");
        }
        else if (!DlssNr::IsRunning() && !vulkan)
        {
            const auto feature = State::Instance().currentFeature;
            const bool nativeVk = feature && feature->Api() == API::Vulkan && !feature->IsWithDx12();
            const char* reason = nativeVk ? DlssNr::FailureReasonVk() : DlssNr::FailureReason();

            if (reason[0] != 0)
            {
                ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.35f, 1.0f), "Off for this session: %s.", reason);
                ImGui::SameLine();

                if (nativeVk)
                    ImGui::TextUnformatted("Restart the game to retry native Vulkan NR.");
                else if (ImGui::SmallButton("Retry"))
                    DlssNr::RetryAfterFailure();
            }
            else if (feature && feature->Api() == API::DX11 && !feature->IsWithDx12())
            {
                ImGui::TextWrapped("NR needs the D3D12 bridge on D3D11. Choose an upscaler marked w/Dx12 and restart.");
            }
            else if (nativeVk && config->DlssNrDeferredDlss.value_or_default())
            {
                ImGui::TextWrapped("Disable Generate before SR, apply after SR (DLSS) to use native Vulkan NR.");
            }
            else if (enabled)
                ImGui::TextUnformatted("Waiting for the upscaler to run.");
        }
        else
        {
            // The elapsed time belongs here rather than only in the upscaler's breakdown: that tooltip needs
            // OptiScaler's own upscaler to have run, and with native DLSS passing through there is
            // nothing in it to hang this off.
            // Either backend's timer. They measure the same thing by different means, and only one
            // of them is running.
            const auto ms = vulkan ? DlssNr::LastGpuTimeVk() : DlssNr::LastGpuTime();

            // With "Apply the model" off the pass STILL RUNS (so Hold-frame A/B can toggle its edit on
            // a frozen frame) -- it only outputs the clean frame.
            // Enable Neural Rendering off stops the work.
            const char* runSuffix =
                !config->DlssNrApplyModel.value_or_default() ? "  (model running, edit hidden)" : "";

            if (ms.has_value())
                ImGui::TextColored(ImVec4(0.4f, 0.9f, 0.5f, 1.0f), "Running%s - %.2f ms elapsed%s",
                                   vulkan ? " natively on Vulkan" : "", ms.value(), runSuffix);
            else if (vulkan)
                // Measured but not yet read: the first few frames are still in the query ring.
                ImGui::TextColored(ImVec4(0.4f, 0.9f, 0.5f, 1.0f), "Running natively on Vulkan - %llu frames%s",
                                   DlssNr::FramesVk(), runSuffix);
            else
                ImGui::TextColored(ImVec4(0.4f, 0.9f, 0.5f, 1.0f), "Running.%s", runSuffix);

            ImGui::SameLine();
            ImGui::TextDisabled("(?)");
            if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
                ImGui::SetTooltip("Time between the start and end of NR on the GPU, including delays while other work runs.\nCompare FPS to check the effect on game performance.");
            if (finishedPicture)
                ImGui::TextDisabled("Includes time shared with other GPU work.");
        }

        ImGui::Spacing();
        ImGui::PushItemWidth(220.0f * menuResScale);

        ImGui::SeparatorText("Performance");

        bool unlockPasses = config->DlssNrUnlockPasses.value_or_default();
        if (ImGui::Checkbox("Lift model pass limit (up to 30; expensive)", &unlockPasses))
            config->DlssNrUnlockPasses = unlockPasses;
        HelpMarker("Allow up to 30 passes instead of 3. More passes use more GPU time and VRAM; high values may crash the game.");
        const unsigned int passLimit = unlockPasses ? MaxPassCount : DefaultMaxPassCount;

        {
            int passes = (int) std::clamp(config->DlssNrPasses.value_or_default(), 1u,
                                          passLimit);
            const ImVec4 colour = passes <= 1   ? ImVec4(0.35f, 0.88f, 0.38f, 1.0f)
                                  : passes == 2 ? ImVec4(0.95f, 0.70f, 0.20f, 1.0f)
                                                : ImVec4(0.92f, 0.30f, 0.25f, 1.0f);

            ImGui::PushStyleColor(ImGuiCol_Text, colour);
            ImGui::PushStyleColor(ImGuiCol_SliderGrab, colour);

            if (ImGui::SliderInt("Model passes", &passes, 1, (int) passLimit,
                                 passes == 1 ? "%d (normal)" : "%dx model cost"))
                config->DlssNrPasses = (uint32_t) std::clamp(passes, 1, (int) passLimit);

            ImGui::PopStyleColor(2);

            HelpMarker("Process the image repeatedly. More passes strengthen the effect and increase GPU cost.\nEach pass has its own settings and history. Start with 1.");
        }

        // Any percentage, rather than a handful of steps somebody chose in advance. The lower bound
        // is 25%: below that the model is working on so little of the picture that its answer no
        // longer survives being enlarged onto it.
        // Applied when the handle is let go, not while it is moving.
        //
        // Every distinct value here is a different working size, and a different working size tears
        // down the scratch textures and rebuilds the model. Writing it on each pixel of a drag meant
        // dozens of rebuilds in a second, which is felt as the whole frame hitching. The slider still
        // reads live; only the commit waits.
        static int pendingScale = -1;

        int scalePercent = pendingScale >= 0
                               ? pendingScale
                               : (int) lroundf(config->DlssNrWorkingScale.value_or_default() * 100.0f);

        if (ImGui::SliderInt("Model resolution", &scalePercent, 25, 200, "%d%%"))
            pendingScale = scalePercent;

        if (ImGui::IsItemDeactivatedAfterEdit() && pendingScale >= 0)
        {
            config->DlssNrWorkingScale = std::clamp(pendingScale, 25, 200) / 100.0f;
            pendingScale = -1;
        }

        HelpMarker("NR resolution relative to the image it processes. 50% halves width and height; 100% uses the full size.\nLower values reduce cost and fine detail. Above 100% increases cost. Game output resolution is unchanged.");

        bool foveated = config->DlssNrFoveated.value_or_default();
        if (ImGui::Checkbox("Foveated NR", &foveated))
        {
            config->DlssNrFoveated = foveated;
            LOG_INFO("DLSS-NR: overlay, Foveated NR -> {} (off = the whole frame)", foveated ? "ON" : "off");
        }
        HelpMarker("Run the model only on a centered ellipse/rounded rectangle and blend it back with a soft edge.\nThe rest of the frame stays as the upscaler left it. Off by default. Gaze tracking is not wired yet.\nChanging width or height rebuilds the model, so those sliders commit when you let go.");
        if (foveated)
        {
            static float pendingFoveaW = -1.0f;
            static float pendingFoveaH = -1.0f;
            float foveaW = pendingFoveaW >= 0.0f ? pendingFoveaW : config->DlssNrFoveaWidth.value_or_default();
            float foveaH = pendingFoveaH >= 0.0f ? pendingFoveaH : config->DlssNrFoveaHeight.value_or_default();
            if (ImGui::SliderFloat("Fovea width", &foveaW, 0.15f, 1.0f, "%.2f"))
                pendingFoveaW = foveaW;
            if (ImGui::IsItemDeactivatedAfterEdit() && pendingFoveaW >= 0.0f)
            {
                config->DlssNrFoveaWidth = pendingFoveaW;
                pendingFoveaW = -1.0f;
            }
            if (ImGui::SliderFloat("Fovea height", &foveaH, 0.15f, 1.0f, "%.2f"))
                pendingFoveaH = foveaH;
            if (ImGui::IsItemDeactivatedAfterEdit() && pendingFoveaH >= 0.0f)
            {
                config->DlssNrFoveaHeight = pendingFoveaH;
                pendingFoveaH = -1.0f;
            }
            float roundness = config->DlssNrFoveaRoundness.value_or_default();
            if (ImGui::SliderFloat("Fovea roundness", &roundness, 0.0f, 1.0f, "%.2f"))
                config->DlssNrFoveaRoundness = roundness;
            float transition = config->DlssNrFoveaTransition.value_or_default();
            if (ImGui::SliderFloat("Fovea transition", &transition, 0.0f, 0.35f, "%.2f"))
                config->DlssNrFoveaTransition = transition;
            float originX = config->DlssNrFoveaOriginX.value_or_default();
            float originY = config->DlssNrFoveaOriginY.value_or_default();
            if (ImGui::SliderFloat("Fovea origin X", &originX, 0.0f, 1.0f, "%.2f"))
                config->DlssNrFoveaOriginX = originX;
            if (ImGui::SliderFloat("Fovea origin Y", &originY, 0.0f, 1.0f, "%.2f"))
                config->DlssNrFoveaOriginY = originY;
            bool border = config->DlssNrFoveaBorder.value_or_default();
            if (ImGui::Checkbox("Fovea alignment ring", &border))
                config->DlssNrFoveaBorder = border;
            bool followSr = config->DlssNrFoveaFollowSr.value_or_default();
            if (ImGui::Checkbox("Fovea follows the SR crop", &followSr))
                config->DlssNrFoveaFollowSr = followSr;
            HelpMarker("While the foveated SR / RR crop is on, the NR fovea is the crop's box in each eye: its "
                       "place (gaze, 'Toward the nose', 'Up') and its size (the crop's Width / Height), so both move "
                       "and resize as one; the fovea's own width, height and origin above apply only with the crop "
                       "off. Each move of the region restarts that eye's NR history (the model would otherwise see "
                       "the move as the world sliding); a new size rebuilds NR's model.");
        }

        if (scalePercent > 100)
            ImGui::TextDisabled("NR scale: %.2fx. Higher resolution increases GPU cost.",
                                scalePercent / 100.0f);

        if (scalePercent > 100)
        {
            static const char* dsNames[] = { "FSR1", "Bicubic", "Catmull-Rom", "Lanczos2",
                                             "Lanczos3", "Kaiser2", "Kaiser3", "MAGIC" };
            int ds = (int) config->DlssNrScalingDownscaler.value_or_default();
            if (ds < 0 || ds >= IM_ARRAYSIZE(dsNames))
                ds = (int) Scaler::Lanczos3;

            if (ImGui::Combo("Downscaler (NR)", &ds, dsNames, IM_ARRAYSIZE(dsNames)))
                config->DlssNrScalingDownscaler = (Scaler) ds;

            HelpMarker("Filter used to reduce NR output when Model resolution exceeds 100%.\nSharper filters may introduce ringing around edges.");
        }


        // Meaningful only when the model runs BELOW the frame's size. At 100% -- and above, where
        // supersampling composites its down-legged answer at native -- the residual collapses to the
        // model's own picture and the two modes are identical, so the control says so by going grey.
        {
            const bool reduced = config->DlssNrWorkingScale.value_or_default() < 0.999f;

            if (!reduced)
                ImGui::BeginDisabled();

            static const char* enlargeNames[] = { "Classic", "Matched residual" };
            int enlarge = config->DlssNrTransfer.value_or_default() == 1 ? 1 : 0;

            if (ImGui::Combo("Enlargement", &enlarge, enlargeNames, IM_ARRAYSIZE(enlargeNames)))
                config->DlssNrTransfer = (uint32_t) enlarge;

            if (!reduced)
                ImGui::EndDisabled();

            HelpMarker("Below 100% model resolution: Classic enlarges the model output; Matched residual enlarges only its changes.\nMatched residual can reduce blur and colour shifts. No effect at 100% or above.");
        }

        ImGui::SeparatorText("Effect strength");

        float transfer = config->DlssNrTransferStrength.value_or_default();
        if (ImGui::SliderFloat("Detail strength", &transfer, 0.0f, 2.0f, "%.2f"))
            config->DlssNrTransferStrength = transfer;

        ImGui::SameLine();
        if (ImGui::SmallButton("Reset##detail"))
            config->DlssNrTransferStrength = 1.0f;

        HelpMarker("Overall NR detail strength: 0 = no effect, 1 = normal, above 1 = exaggerated.");

        float colour = config->DlssNrColourStrength.value_or_default();
        if (ImGui::SliderFloat("Colour strength", &colour, 0.0f, 4.0f, "%.2f"))
            config->DlssNrColourStrength = colour;

        ImGui::SameLine();
        if (ImGui::SmallButton("Reset##colour"))
            config->DlssNrColourStrength = 1.0f;

        HelpMarker("NR colour strength: 0 = preserve game colours, 1 = model colours, above 1 = stronger saturation.");

        // Experimental. 0 off (soft knee), 1 Neutwo + our composition, 2 Neutwo + pure-inverse replace,
        // 3 hybrid+composed, 4 hybrid+replace (identity midtones + unclipped highlights). Always shown.
        static const char* reversibleNames[] = { "Off (soft knee)", "Neutwo proxy + composed",
                                                 "Neutwo proxy + replace", "Hybrid proxy + composed",
                                                 "Hybrid proxy + replace" };
        int reversible = (int) config->DlssNrReversibleMode.value_or_default();
        if (reversible < 0 || reversible > 4)
            reversible = 0;
        if (ImGui::Combo("HDR mapping (experimental)", &reversible, reversibleNames,
                         IM_ARRAYSIZE(reversibleNames)))
            config->DlssNrReversibleMode = (uint32_t) reversible;

        HelpMarker("Choose how HDR brightness is mapped for NR.\nSoft knee compresses highlights. Neutwo uses a reversible curve. Hybrid preserves midtones and compresses highlights.\nComposed uses the strength and highlight controls. Replace bypasses them and may flicker.");

        ImGui::SeparatorText("Model passes");
        ImGui::TextWrapped("Settings apply when you release a slider.");
        static const char* styles[] = { "Standard", "Natural", "Cinematic" };
        static const char* inheritedStyles[] = { "Auto (inherit pass 1)", "Standard", "Natural", "Cinematic" };

        if (ImGui::TreeNodeEx("Pass 1", ImGuiTreeNodeFlags_DefaultOpen))
        {
            int style = (int) std::min(config->DlssNrStyle.value_or_default(), 2u);
            if (ImGui::Combo("Style", &style, styles, IM_ARRAYSIZE(styles)))
                config->DlssNrStyle = (uint32_t) style;
            HelpMarker("Select the appearance profile: Standard, Natural or Cinematic. Intensity controls its strength.");
            DeferredSlider("Intensity", &config->DlssNrIntensity, 0.0f, 2.0f, 1.0f);
            DeferredSlider("Local structure", &config->DlssNrLocalStructure, 0.0f, 2.0f, 1.0f);
            DeferredSlider("Local tone", &config->DlssNrLocalTone, 0.0f, 2.0f, 1.0f);
            DeferredSlider("Skin structure", &config->DlssNrSkinStructure, -1.0f, 2.0f, -1.0f);
            bool mask = config->DlssNrAutoMask.value_or_default();
            if (ImGui::Checkbox("Auto skin mask", &mask))
                config->DlssNrAutoMask = mask;
            HelpMarker("Use the model's learned skin selection to apply Skin structure without an authored mask.\nAccuracy varies. This is separate from the colour-based mask below.");
            ImGui::TreePop();
        }

        if (ImGui::TreeNodeEx("Pass 2", ImGuiTreeNodeFlags_DefaultOpen))
        {
            ImGui::TextWrapped("Defaults: inherit Pass 1; Local tone = 0. Reset restores these defaults.");
            InheritedProfileCombo("Style", &config->DlssNrPass2Style, inheritedStyles, IM_ARRAYSIZE(inheritedStyles));
            DeferredSlider("Intensity", &config->DlssNrPass2Intensity, 0.0f, 2.0f, config->DlssNrIntensity.value_or_default(), "%.2f", true);
            DeferredSlider("Local structure", &config->DlssNrPass2LocalStructure, 0.0f, 2.0f, config->DlssNrLocalStructure.value_or_default(), "%.2f", true);
            DeferredSlider("Local tone", &config->DlssNrPass2LocalTone, 0.0f, 2.0f, 0.0f, "%.2f", true);
            DeferredSlider("Skin structure", &config->DlssNrPass2SkinStructure, -1.0f, 2.0f, config->DlssNrSkinStructure.value_or_default(), "%.2f", true);
            bool mask = config->DlssNrPass2AutoMask.has_value() ? config->DlssNrPass2AutoMask.value() : config->DlssNrAutoMask.value_or_default();
            if (ImGui::Checkbox("Auto skin mask", &mask))
                config->DlssNrPass2AutoMask = mask;
            ImGui::SameLine();
            if (ImGui::SmallButton("Reset##mask"))
                config->DlssNrPass2AutoMask = std::optional<bool> {};
            ImGui::TreePop();
        }

        if (ImGui::TreeNodeEx("Pass 3", ImGuiTreeNodeFlags_DefaultOpen))
        {
            ImGui::TextWrapped("Defaults: inherit Pass 1; Local tone = 0. Reset restores these defaults.");
            InheritedProfileCombo("Style", &config->DlssNrPass3Style, inheritedStyles, IM_ARRAYSIZE(inheritedStyles));
            DeferredSlider("Intensity", &config->DlssNrPass3Intensity, 0.0f, 2.0f, config->DlssNrIntensity.value_or_default(), "%.2f", true);
            DeferredSlider("Local structure", &config->DlssNrPass3LocalStructure, 0.0f, 2.0f, config->DlssNrLocalStructure.value_or_default(), "%.2f", true);
            DeferredSlider("Local tone", &config->DlssNrPass3LocalTone, 0.0f, 2.0f, 0.0f, "%.2f", true);
            DeferredSlider("Skin structure", &config->DlssNrPass3SkinStructure, -1.0f, 2.0f, config->DlssNrSkinStructure.value_or_default(), "%.2f", true);
            bool mask = config->DlssNrPass3AutoMask.has_value() ? config->DlssNrPass3AutoMask.value() : config->DlssNrAutoMask.value_or_default();
            if (ImGui::Checkbox("Auto skin mask", &mask))
                config->DlssNrPass3AutoMask = mask;
            ImGui::SameLine();
            if (ImGui::SmallButton("Reset##mask"))
                config->DlssNrPass3AutoMask = std::optional<bool> {};
            ImGui::TreePop();
        }

        const unsigned int visiblePasses = std::clamp(config->DlssNrPasses.value_or_default(), 1u, passLimit);
        for (unsigned int pass = 3; pass < visiblePasses; ++pass)
        {
            auto& settings = config->DlssNrExtraPasses[pass - 3];
            if (!ImGui::TreeNode(std::format("Pass {}", pass + 1).c_str()))
                continue;
            ImGui::TextWrapped("Defaults: inherit Pass 1; Local tone = 0.");
            InheritedProfileCombo("Style", &settings.style, inheritedStyles, IM_ARRAYSIZE(inheritedStyles));
            DeferredSlider("Intensity", &settings.intensity, 0.0f, 2.0f, config->DlssNrIntensity.value_or_default(), "%.2f", true);
            DeferredSlider("Local structure", &settings.structure, 0.0f, 2.0f, config->DlssNrLocalStructure.value_or_default(), "%.2f", true);
            DeferredSlider("Local tone", &settings.tone, 0.0f, 2.0f, 0.0f, "%.2f", true);
            DeferredSlider("Skin structure", &settings.skin, -1.0f, 2.0f, config->DlssNrSkinStructure.value_or_default(), "%.2f", true);
            bool mask = settings.autoMask.value_or(config->DlssNrAutoMask.value_or_default());
            if (ImGui::Checkbox("Auto skin mask", &mask))
                settings.autoMask = mask;
            ImGui::SameLine();
            if (ImGui::SmallButton("Reset##mask"))
                settings.autoMask = std::optional<bool> {};
            ImGui::TreePop();
        }

        if (ImGui::TreeNode("Advanced preset hints (effect unverified)"))
        {
            ImGui::TextWrapped("Experimental model hints; visual effect unverified. Use Style to select a profile.");
            static const char* presets[] = { "Default", "Preset 1", "Preset 2", "Preset 3" };
            static const char* inheritedPresets[] = { "Auto (inherit pass 1)", "Default", "Preset 1", "Preset 2", "Preset 3" };
            int preset = (int) std::min(config->DlssNrPreset.value_or_default(), 3u);
            if (ImGui::Combo("Pass 1 preset hint", &preset, presets, IM_ARRAYSIZE(presets)))
                config->DlssNrPreset = (uint32_t) preset;
            InheritedProfileCombo("Pass 2 preset hint", &config->DlssNrPass2Preset, inheritedPresets, IM_ARRAYSIZE(inheritedPresets));
            InheritedProfileCombo("Pass 3 preset hint", &config->DlssNrPass3Preset, inheritedPresets, IM_ARRAYSIZE(inheritedPresets));
            ImGui::TreePop();
        }
        ImGui::TextWrapped("Pass settings apply to SR and RR on DX12 and native Vulkan. The driver-proxy backend supports one pass.");

        ImGui::SeparatorText("Colour");

        if (ImGui::TreeNode("Skin and environment (final edit)"))
        {
            ImGui::TextWrapped("Select skin by colour and adjust the final NR effect separately for skin and scenery. Selection can be inaccurate; check Preview.");
            bool filter = config->DlssNrSkinProtection.value_or_default();
            if (ImGui::Checkbox("Separate skin / environment controls", &filter))
                config->DlssNrSkinProtection = filter;
            ImGui::BeginDisabled(!filter);
            bool tone = config->DlssNrSkinToneEnabled.value_or_default();
            if (ImGui::Checkbox("Allow skin tone / colour changes", &tone))
                config->DlssNrSkinToneEnabled = tone;
            HelpMarker("Allow NR colour changes in the selected skin region. Turn off to preserve its colour; detail can still change.");
            const auto slider = [](const char* label, auto& option) {
                float v = option.value_or_default();
                if (ImGui::SliderFloat(label, &v, 0.0f, 1.0f, "%.2f"))
                    option = v;
                HelpMarker("NR strength in this region: 0 = no change, 1 = full effect.");
            };
            slider("Skin detail / lighting", config->DlssNrSkinDetail);
            ImGui::BeginDisabled(!tone);
            slider("Skin colour", config->DlssNrSkinColour);
            ImGui::EndDisabled();
            slider("Environment detail / lighting", config->DlssNrEnvironmentDetail);
            slider("Environment colour", config->DlssNrEnvironmentColour);
            bool preview = config->DlssNrShowSkinMask.value_or_default();
            if (ImGui::Checkbox("Preview colour-based mask", &preview))
                config->DlssNrShowSkinMask = preview;
            ImGui::EndDisabled();
            ImGui::TreePop();
        }

        ImGui::TextDisabled("HDR input settings. Adjust the brightness range presented to NR.");

        {
        // Logarithmic, because the useful range is not linear. A quarter to 240: the low end because
        // a frame the game already tone mapped wants roughly 1, the high end because there is no
        // principled ceiling -- this is a divisor on an open-ended linear buffer, and how far up a
        // given game needs to go is a property of that game's exposure, not of anything we can bound.
        // One game was still improving at 100. A linear slider over that span would spend nine
        // tenths of its travel on values nobody needs and never reach the ones they do.
        // One dropdown, because there is one answer.
        //
        // This was two checkboxes that could both be on, and every attempt to stop that was a patch
        // on a shape that should not have existed. Greying deadlocked -- each disabled the other, so
        // once both were set the only way out was a button the notice never mentioned. Clearing
        // worked but silently undid a setting somebody had made. Both were ways to stop an illegal
        // state being REACHED; a single choice cannot reach it, because there is only one value to
        // be in.
        //
        // Each option also says whether it can actually do anything in THIS game, in colour, so the
        // choice is made on what is available rather than on what sounds best.
        {
            const auto ex = DlssNr::GameExposureStatus();
            const bool vk = DlssNr::IsRunningVk();
            const bool haveExposure = vk ? DlssNr::ExposureOfferedVk() : ex.everOffered;

            const float anchorNow = DlssNr::ExposureScan::BestValue();
            const bool haveAnchor = !DlssNr::ExposureScan::Anchors().empty();

            static const char* sourceNames[] = { "Manual paper white", "Game exposure",
                                                 "Scanned exposure (experimental)" };

            int source = (int) config->DlssNrWhitePointSource.value_or_default();

            if (source < 0 || source > 2)
                source = 0;

            if (ImGui::Combo("White point source", &source, sourceNames, IM_ARRAYSIZE(sourceNames)))
            {
                config->DlssNrWhitePointSource = (uint32_t) source;

                // Nothing else to set. The scan asks the source whether it is wanted, so choosing
                // it here is the whole of switching it on -- there is no second flag to keep in
                // step, and so no way for the two to disagree.
            }

            HelpMarker("Manual: use Paper white. Game exposure: use exposure supplied by the game.\nScanned exposure: estimate it from game buffers; requires calibration and may select the wrong buffer.");

            // Availability, in colour, for the option currently chosen.
            if (source == 1)
            {
                if (!vk && ex.seenFrames == 0)
                    ImGui::TextDisabled("Waiting for a frame...");
                else if (!haveExposure)
                    ImGui::TextColored(ImVec4(0.9f, 0.6f, 0.25f, 1.0f),
                                       "No game exposure available. Using manual paper white.");
                else if (vk)
                    ImGui::TextColored(ImVec4(0.45f, 0.8f, 0.45f, 1.0f),
                                       "Using game exposure.");
                else if (ex.exposure > 1e-6f)
                {
                    const float trim =
                        std::clamp(config->DlssNrWhitePointTrim.value_or_default(), 0.25f, 4.0f);
                    ImGui::TextColored(ImVec4(0.45f, 0.8f, 0.45f, 1.0f),
                                       "Game exposure %.4f  ->  white point %.2f%s", ex.exposure,
                                       ex.preExposure / ex.exposure * trim,
                                       ex.offeredNow ? "" : "  (held: absent this frame)");
                }
                else
                    ImGui::TextDisabled("Reading exposure...");
            }
            else if (source == 2)
            {
                // "Nothing found" and "found several, none of them moving" are different states,
                // and this said the first for both. In one game the log carried eight candidates while
                // the panel claimed there were none, which reads as the scan being broken when what
                // it actually needs is for the light to change.
                if (anchorNow <= 0.0f)
                {
                    const unsigned int watching = (unsigned int) DlssNr::ExposureScan::Report().size();

                    if (watching == 0)
                        ImGui::TextColored(ImVec4(0.9f, 0.6f, 0.25f, 1.0f),
                                           "No exposure candidates found.");
                    else
                        ImGui::TextColored(ImVec4(0.9f, 0.6f, 0.25f, 1.0f),
                                           "%u candidates; move between bright and dark areas to test them.",
                                           watching);
                }
                else if (!haveAnchor)
                    ImGui::TextColored(ImVec4(0.9f, 0.6f, 0.25f, 1.0f),
                                       "Exposure candidate found. Adjust Paper white, then select Anchor here.");
                // Once anchored, the scan -> white point readout sits above the sliders below; it is
                // not repeated up here.
            }
            else if (haveExposure)
            {
                ImGui::TextColored(ImVec4(0.45f, 0.8f, 0.45f, 1.0f),
                                   "Game exposure is available.");
            }
        }






        // A measured suggestion for paper white used to sit here and has been withdrawn.
        //
        // It took the 90th percentile of per-tile peak luminance from the untouched frame, which is a
        // statement about scene content rather than about the buffer's scale. In Nioh 3, where the
        // right answer is about 240, it offered 8 -- because most tiles are shadow and the percentile
        // sits wherever most tiles are. The guard meant to catch that compared each tile against the
        // frame's own brightest, which is scale-free and therefore passes on a black screen: the same
        // relative-threshold mistake the white point meter was removed for, made a second time.
        //
        // A wrong number offered confidently is worse than no number, so nothing is offered. What
        // replaces it has to be a measurement of the game's own exposure rather than of its scenery:
        // the exposure texture where a game supplies one, and otherwise the ratio between the
        // scene-referred buffer and the finished frame, which is that exposure by definition.

        // Two controls, not one control with two meanings.
        //
        // These are different quantities. The manual path wants an absolute divisor on an open-ended
        // linear buffer -- Nioh 3 needs about 240 -- and the exposure path wants a multiplier on a
        // number the game already supplied, where 1 is correct and anything far from it says the read
        // is wrong rather than that somebody prefers it.
        //
        // They used to share one stored value, narrowed to 0.25..4 when the toggle was on. That kept
        // a ruinous value unreachable but left two worse problems: moving the slider in one mode
        // silently destroyed the number found in the other, and there was no way back to "just take
        // the game's answer" short of knowing that the number for it was 1. Separate values fix both.
        // Switching modes is now non-destructive in both directions.
        // The trim belongs to both automatic sources, since both end in "the game's number times a
        // little". Only the manual source gets the absolute slider.
        // One slider per source, each remembering its own number.
        //
        // A trim on the game's exposure and a trim on a buffer the scan found are trims on different
        // things, and a value found against one means nothing against the other. Sharing them meant
        // changing source silently carried a number across, so a picture that had been tuned came
        // back wrong for a reason nothing on screen explained.
        //
        // The scan before it is anchored is the exception, and it has to be: anchoring captures an
        // absolute white point, so there must be an absolute slider to set. Showing a trim there
        // asked people to "set paper white below" next to a control that was not paper white.
        const int wpSource = (int) config->DlssNrWhitePointSource.value_or_default();

        // Which anchor row the paper-white slider edits, or -1 for the live unanchored point. Menu-
        // local and not persisted; the anchor block below sets it when a row is clicked. Declared
        // here because both the slider (this block) and the table (below) read it in the same frame.
        static int selectedAnchor = -1;
        auto anchors = DlssNr::ExposureScan::Anchors();
        if (selectedAnchor >= (int) anchors.size())
            selectedAnchor = -1;

        if (wpSource == 2)
        {
            const bool editingRow = selectedAnchor >= 0 && selectedAnchor < (int) anchors.size();

            // The single scan -> white point readout, above the sliders it explains.
            if (!anchors.empty())
            {
                const float liveScan = DlssNr::ExposureScan::BestValue();

                if (liveScan > 0.0f)
                {
                    const float w = DlssNr::ExposureScan::AnchoredWhitePoint(
                        liveScan, config->DlssNrScanInverted.value_or_default(),
                        config->DlssNrScanTrim.value_or_default());

                    ImGui::TextColored(ImVec4(0.45f, 0.8f, 0.45f, 1.0f),
                                       "Scan %.5f  ->  white point %.2f   (%u point%s)", liveScan, w,
                                       (unsigned) anchors.size(), anchors.size() == 1 ? "" : "s");
                }
            }

            // Paper white shows only when there is a point to set: before the first anchor, or when a
            // row is selected to edit. Once points exist and none is selected, the white point is fixed
            // by the anchors and only the trim adjusts the live picture -- so the trim takes the
            // slider's place, the same shape as the game-exposure source.
            const bool showPaperWhite = anchors.empty() || editingRow;

            if (showPaperWhite)
            {
                float pw = editingRow ? anchors[selectedAnchor].white
                                      : config->DlssNrWhitePointScale.value_or_default();

                char lbl[48];
                if (editingRow)
                    snprintf(lbl, sizeof(lbl), "Paper white (editing point %d)", selectedAnchor + 1);
                else
                    snprintf(lbl, sizeof(lbl), "Paper white");

                if (ImGui::SliderFloat(lbl, &pw, 0.25f, 2000.0f, "%.2fx", ImGuiSliderFlags_Logarithmic))
                {
                    if (editingRow)
                    {
                        DlssNr::ExposureScan::AnchorSetWhite(selectedAnchor, pw);
                        config->DlssNrScanAnchors = DlssNr::ExposureScan::SerializeAnchors();
                    }
                    else
                        config->DlssNrWhitePointScale = pw;
                }

                HelpMarker("Adjust the selected calibration point, or set the value for the next point.\nUse Anchor here to save the current lighting condition.");
            }

            // The trim multiplies the interpolated result, and in the steady state it is the control
            // that stands in for paper white: adjust it until the picture looks right in the current
            // light, then Anchor bakes that trimmed value into a new point and resets the trim to 1.
            if (!anchors.empty())
            {
                float trim = config->DlssNrScanTrim.value_or_default();

                if (ImGui::SliderFloat("Trim (x the scan)", &trim, 0.25f, 4.0f, "%.2fx",
                                       ImGuiSliderFlags_Logarithmic))
                    config->DlssNrScanTrim = std::clamp(trim, 0.25f, 4.0f);

                ImGui::SameLine();

                if (ImGui::SmallButton("Reset##scantrim"))
                    config->DlssNrScanTrim = 1.0f;

                HelpMarker("Multiply the calibrated white point. Anchor here saves the adjusted value and resets this multiplier to 1.");
            }
        }
        else if (wpSource == 1)
        {
            const bool ofScan = false;

            float trim = ofScan ? config->DlssNrScanTrim.value_or_default()
                                : config->DlssNrWhitePointTrim.value_or_default();

            if (ImGui::SliderFloat(ofScan ? "Trim (x the scan)" : "Trim (x the game's exposure)", &trim,
                                   0.25f, 4.0f, "%.2fx", ImGuiSliderFlags_Logarithmic))
            {
                if (ofScan)
                    config->DlssNrScanTrim = std::clamp(trim, 0.25f, 4.0f);
                else
                    config->DlssNrWhitePointTrim = std::clamp(trim, 0.25f, 4.0f);
            }

            ImGui::SameLine();

            // Deliberately always present rather than greyed at 1. The point of it is that the safe
            // value is one click away without having to know what the safe value is.
            if (ImGui::SmallButton("Reset##wptrim"))
            {
                if (ofScan)
                    config->DlssNrScanTrim = 1.0f;
                else
                    config->DlssNrWhitePointTrim = 1.0f;
            }

            HelpMarker("Multiply the white point derived from game exposure. 1 = no adjustment.");
        }
        else
        {
            // Logarithmic, because the useful range is not linear. A quarter to 2000: the low end
            // because a frame the game already tone mapped wants roughly 1, the high end because
            // there is no principled ceiling -- this is a divisor on an open-ended linear buffer, and
            // how far up a given game needs to go is a property of that game's exposure rather than
            // of anything that can be bounded here. One game was still improving at 100.
            float wpScale = config->DlssNrWhitePointScale.value_or_default();

            if (ImGui::SliderFloat("Paper white", &wpScale, 0.25f, 2000.0f, "%.2fx",
                                   ImGuiSliderFlags_Logarithmic))
                config->DlssNrWhitePointScale = wpScale;

        HelpMarker("Brightness reference used to prepare HDR colour for NR. Higher values darken the model input; lower values brighten it.\nAdjust if NR loses detail or produces colour shifts.");
        }

        // Highlight guard, directly under the white point / trim -- it bounds the model's edit and
        // belongs with the exposure controls it works alongside.
        float maxRatio = config->DlssNrMaxRatio.value_or_default();
        if (ImGui::SliderFloat("Highlight guard", &maxRatio, 1.0f, unlockPasses ? (float) MaxPassCount : 8.0f, "%.1fx"))
            config->DlssNrMaxRatio = maxRatio;

        ImGui::SameLine();
        if (ImGui::SmallButton("Reset##guard"))
            config->DlssNrMaxRatio = 2.0f;

        HelpMarker("Limit how much NR can brighten or darken a pixel. Lower values restrict highlight changes; higher values allow more.");

        // Directly under the white point, because that is the number it moves and the number the
        // anchor captures. It used to sit under Inspect, a whole section away from the slider it
        // reads, which left "Anchor here" looking like a control for something else entirely.
        {
            // No checkbox here any more.
            //
            // The dropdown above says whether the scan is the white point's source, and that is
            // the only reason anybody using this would want it running. A second control could
            // only agree with the dropdown or contradict it, and both were on offer: it began as
            // a redundant question and became a way to switch off the thing the chosen source
            // depended on.
            //
            // The ini key survives as a developer override for the one case a user has no reason
            // to want -- running the scan in a game that supplies a REAL exposure, so the log can
            // compare the two. That is validation, and validation does not need a widget.
            //
            // Worth keeping written down, since the panel no longer says it: the scan matches
            // buffers by SHAPE, and shape is a weak filter. In one game -- a game that supplies a
            // real exposure, so the right answer sat visible beside it -- the best candidate was
            // a 1x1 R32_FLOAT that climbed in a straight line for seventeen minutes while the
            // true exposure held still. Their ratio moved 14x. That is an accumulator, not an
            // eye adaptation.

                // Only where it means something. The lamp reads the scan, so offering it beside a
                // white point that comes from the game's own exposure is offering a control that
                // cannot light up.
                bool meter = config->DlssNrScanMeter.value_or_default();

                if (config->DlssNrWhitePointSource.value_or_default() == 2 &&
                    ImGui::Checkbox("Show exposure meter", &meter))
                    config->DlssNrScanMeter = meter;

                HelpMarker("Show the scanned exposure value and a colour indicator. Display only; does not change the image.");

            // Shown when the scan is actually running, whichever way it got switched on.
            if (DlssNr::ExposureScan::Scanning())
            {
                // Anchoring: one press, then it never needs touching again.
                //
                // The absolute white point cannot come out of a buffer whose units are unknown.
                // Every value AFTER the first can: only the ratio against the anchor is used, so
                // whatever the number means, it cancels. That is why this is a button and not a
                // measurement -- the one thing a person can supply that no amount of cleverness
                // can is "this looks right to me".
                int which = 0;
                float low = 0.0f, high = 0.0f;
                const float live = DlssNr::ExposureScan::BestValue(&which, &low, &high);

                const bool isSource = config->DlssNrWhitePointSource.value_or_default() == 2;

                // Anchor captures (currentScan, currentPaperWhite) and ADDS a row -- it does not
                // replace. One row is the old single-anchor ratio law; add a second in different
                // light and the white point is interpolated between the points, so it holds across
                // the whole range instead of only near one anchor. Greyed unless the scan is the
                // chosen source and it currently has a value to capture.
                ImGui::BeginDisabled(live <= 0.0f || !isSource);

                if (ImGui::Button("Anchor here"))
                {
                    // What to capture. Before the first point, the paper white above (an absolute value
                    // with the wide range a fresh game needs). After that, the EFFECTIVE white point the
                    // picture is showing right now -- the interpolated value times the Trim the player just
                    // dialed in -- so a second point in different light captures the trimmed look, not a
                    // frozen paper white (which would make two equal whites and a flat, non-tracking
                    // curve). The trim is reset afterwards: the new point, which the picture now passes
                    // through exactly, must not be multiplied by it a second time.
                    const float captureWhite =
                        anchors.empty()
                            ? std::max(0.01f, config->DlssNrWhitePointScale.value_or_default())
                            : std::max(0.01f, DlssNr::ExposureScan::AnchoredWhitePoint(
                                                  live, config->DlssNrScanInverted.value_or_default(),
                                                  config->DlssNrScanTrim.value_or_default()));

                    if (DlssNr::ExposureScan::AnchorAdd(live, captureWhite))
                    {
                        config->DlssNrScanAnchors = DlssNr::ExposureScan::SerializeAnchors();
                        config->DlssNrScanTrim = 1.0f;
                        selectedAnchor = -1;
                    }
                }

                ImGui::EndDisabled();

                HelpMarker("Save the current exposure and white point as a calibration point.\nAdjust Paper white for the first point, then Trim for additional lighting conditions. Up to 8 points.");

                if (!isSource)
                    ImGui::TextDisabled("Scanned exposure is not the selected white point source.");

                if (!anchors.empty())
                {
                    // The row nearest the live scan value (in log space) is the one driving the
                    // picture right now; mark it so the player can see which calibration is in effect.
                    int active = 0;
                    float bestDist = 1e30f;
                    const float liveLog = std::log(std::max(live, 1e-6f));

                    for (size_t i = 0; i < anchors.size(); ++i)
                    {
                        const float d =
                            std::fabs(std::log(std::max(anchors[i].scan, 1e-6f)) - liveLog);
                        if (d < bestDist)
                        {
                            bestDist = d;
                            active = (int) i;
                        }
                    }

                    for (size_t i = 0; i < anchors.size(); ++i)
                    {
                        ImGui::PushID((int) i);

                        // Delete first, so its click is never swallowed by the row-wide Selectable.
                        if (ImGui::SmallButton("x"))
                        {
                            DlssNr::ExposureScan::AnchorRemove((int) i);
                            config->DlssNrScanAnchors = DlssNr::ExposureScan::SerializeAnchors();
                            if (selectedAnchor == (int) i)
                                selectedAnchor = -1;
                            else if (selectedAnchor > (int) i)
                                --selectedAnchor;
                            ImGui::PopID();
                            continue;
                        }

                        ImGui::SameLine();

                        const bool sel = (int) i == selectedAnchor;
                        char row[96];
                        snprintf(row, sizeof(row), "%s scan %.4f  ->  white %.2f%s",
                                 ((int) i == active && isSource) ? ">" : "  ", anchors[i].scan,
                                 anchors[i].white, sel ? "   [editing]" : "");

                        // Click selects the row (slider edits it); click again deselects (slider
                        // returns to the live unanchored point).
                        if (ImGui::Selectable(row, sel))
                            selectedAnchor = sel ? -1 : (int) i;

                        ImGui::PopID();
                    }

                    ImGui::TextDisabled("Select a row to edit it; select it again"
                                        " to deselect. > marks the active point.");
                }

                // The direction flag only means anything with a single point; with two or more the
                // direction the white point moves is already fixed by the data.
                if (anchors.size() == 1)
                {
                    bool inverted = config->DlssNrScanInverted.value_or_default();
                    if (ImGui::Checkbox("Invert exposure tracking", &inverted))
                        config->DlssNrScanInverted = inverted;

                    HelpMarker("Reverse how scanned exposure changes the white point. Only needed with one calibration point.");
                }

                // The scan -> white point readout is shown above the sliders now, not here.

                // Everything below is read-out rather than control: what the scan is looking at and
                // how to tell whether it found the right thing. Folded away because the two decisions
                // that matter -- anchor, and which way the number runs -- are above it.
                if (ImGui::TreeNode("Advanced"))
                {

                    const auto found = DlssNr::ExposureScan::Report();
                    const char* why = DlssNr::ExposureScan::Status();

                    if (found.empty())
                    {
                        ImGui::TextDisabled("%s", why != nullptr && why[0] != 0
                                                      ? why
                                                      : "No exposure candidates found.");
                    }
                    else
                    {
                        for (size_t i = 0; i < found.size(); ++i)
                        {
                            const auto& c = found[i];

                            if (c.reads == 0)
                            {
                                ImGui::TextDisabled("%zu. %s -- not read yet", i + 1, c.shape.c_str());
                                continue;
                            }

                            // Moving is the whole signal, so it is the thing that is coloured.
                            ImGui::TextColored(c.moves ? ImVec4(0.45f, 0.8f, 0.45f, 1.0f)
                                                       : ImVec4(0.6f, 0.6f, 0.6f, 1.0f),
                                               "%zu. %s = %.5f  (seen %.5f..%.5f) %s", i + 1,
                                               c.shape.c_str(), c.latest, c.lowest, c.highest,
                                               c.moves ? "MOVES" : "flat so far");
                        }

                        ImGui::TextDisabled("Move between bright and dark areas to check exposure tracking.");
                        ImGui::TextDisabled("A value that only increases may be a counter.");
                    }

                    ImGui::TreePop();
                }
            }
        }


        }

        ImGui::SeparatorText("Compare");

        // Freeze the frame the model works on, so a setting change re-renders it in place -- the only
        // clean way to A/B our own settings (a moving scene confounds every other comparison). See
        // design/frame-hold.md.
        bool held = config->DlssNrHoldFrame.value_or_default();
        if (ImGui::Checkbox("Hold frame", &held))
            config->DlssNrHoldFrame = held;

        HelpMarker("Freeze NR's input to compare its settings. The game's HUD and later effects may keep updating.\nDoes not re-run SR/RR or show changes to their settings. Turn off to resume.");

        static const char* compareNames[] = { "Off", "Side by side", "Wipe" };
        int compare = (int) config->DlssNrCompare.value_or_default();
        if (ImGui::Combo("Compare", &compare, compareNames, IM_ARRAYSIZE(compareNames)))
            config->DlssNrCompare = (uint32_t) compare;

        HelpMarker("Compare the original and NR result. Side by side fits both images; Wipe divides one full-size image.");

        if (compare != 0)
        {
            bool swap = config->DlssNrCompareSwap.value_or_default();
            if (ImGui::Checkbox("Swap sides", &swap))
                config->DlssNrCompareSwap = swap;
            HelpMarker("Swap the original and NR sides.");

            bool tags = config->DlssNrCompareTags.value_or_default();
            if (ImGui::Checkbox("Label the sides", &tags))
                config->DlssNrCompareTags = tags;

            HelpMarker("Display labels identifying the original and NR sides.");

            if (tags)
            {
                float tagScale = config->DlssNrTagScale.value_or_default();
                if (ImGui::SliderFloat("Label size", &tagScale, 0.5f, 5.0f, "%.1fx"))
                    config->DlssNrTagScale = std::clamp(tagScale, 0.5f, 5.0f);
            }

        }

        if (compare == 1)
        {
            float zoom = config->DlssNrCompareZoom.value_or_default();
            if (ImGui::SliderFloat("Zoom", &zoom, 1.0f, 2.0f, "%.2f"))
                config->DlssNrCompareZoom = std::clamp(zoom, 1.0f, 2.0f);

            HelpMarker("Side-by-side zoom: 1 = fit the whole image, 2 = fill each half by cropping the sides.");
        }

        if (compare == 2)
        {
            float split = config->DlssNrCompareSplit.value_or_default();
            if (ImGui::SliderFloat("Split", &split, 0.0f, 1.0f, "%.2f"))
                config->DlssNrCompareSplit = std::clamp(split, 0.0f, 1.0f);

            HelpMarker("Position of the comparison boundary. Swap sides reverses which image appears on each side.");
        }

        static const char* debugNames[] = { "Off", "Proxy (what the model sees)", "Model output (raw)",
                                            "Difference (amplified)" };
        int debugView = (int) config->DlssNrDebugView.value_or_default();
        if (ImGui::Combo("Debug view", &debugView, debugNames, IM_ARRAYSIZE(debugNames)))
            config->DlssNrDebugView = (uint32_t) debugView;

        HelpMarker("Show the model input, raw output, or a 20x amplified difference. Grey in Difference means no change.");

        ImGui::PopItemWidth();
    }
}

} // namespace DlssNr

