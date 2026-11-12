// Offline test of the game DLL's side of the headset overlay (OptiScaler/inputs/OverlayCrop.h, as OverlayShare uses
// it) on a REAL ImGui frame and a real D3D11 device: the fork's own ImGui, its own D3D11 renderer.
//   overlay_test
// The canvas is as large as FC5's in VR (8192 x 8192). On it: the MENU (its name begins as OptiScaler's does), a
// performance overlay in a far corner (what only informs), and the mouse pointer drawn far from both. After
// ImGui::Render -- where the game DLL is called -- the rectangle must be the menu's with its margin, and the draw
// data rendered through it must show the menu: opaque inside, transparent in the margin and beyond.
//   frames  1- 3  the menu 900 x 600 at 3000, 2500
//   frames  4- 6  the menu MOVED to 5200, 4700 (past the 4096 a panel holds) and 900 x 800: the rectangle follows
//   frames  7- 9  the menu 900 x 600 again: the rectangle keeps the largest size (no jumping back and forth)
//   frames 10-12  the menu 5000 x 3000: larger than a panel holds, so drawn smaller, NOT cut
//   frames 13-15  the menu closed: everything drawn counts (the performance overlay's rectangle)
// Written after the first version crashed the game at its first frame (it called ImGui after the frame's end): this
// runs the very code, so such a call crashes here. A run of the second version is the reason for frames
// 4-6: the rectangle began at an indicator in the canvas's corner and ended 4096 pixels on, and the menu disappeared
// from the panel when it was moved past that.
// Controls, set by hand, each must FAIL: OVERLAY_TEST_POINTER=1 (the pointer counted), OVERLAY_TEST_PASSIVE=1 (the
// performance overlay counted beside the menu: the old rule), OVERLAY_TEST_OLDCALL=1 (the call that crashed).
#include <windows.h>
#include <d3d11.h>

#include "imgui.h"
#include "imgui_impl_dx11.h"
#include "imgui_impl_dx12.h"
#include "../../../OptiScaler/inputs/MainThreadCall.h"
#include "../../../OptiScaler/inputs/OverlayCrop.h"
#include "../../../OptiScaler/inputs/OverlayTextures.h"
#include "../../../OptiScaler/inputs/OverlayTextures12.h"
#include <d3d11_4.h>
#include <d3d12.h>
#include "../../../OptiScaler/inputs/FoveaMenu_Vk.h"
#include "../../../OptiScaler/inputs/PerEyeMenu.h"
#include "../../../OptiScaler/inputs/StaleEyeMenu.h"

#include <d3d11_1.h>

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// ---- "overlay_test section": the per-eye DLSS section of the menu (OptiScaler/inputs/PerEyeMenu.h, "Per-eye DLSS"),
// the very code the game DLL draws, on a real ImGui frame in a window as large as the headset panel's (1000 x 1000). The
// section is a simple part and a "Junk Drawer"; the test draws both (the drawer open). The
// simple part is ONE dropdown (Off / On), clicked in two steps: one click opens it, a second chooses a line.
//   shared, "On" is chosen: exactly one switch of the routing is asked for, and per eye as the saved start
//   per eye, "On" is chosen again: nothing is asked
//   per eye with same-eye motion: nothing asked for, no warning
//   per eye and DLSS not called at all (the game's TAA makes the picture): NO warning
//   per eye and NO same-eye motion in the last window: the warning is shown
//   per eye SINCE A SWITCH, the last window still counted shared (so without same-eye motion): NO warning -- the
//   section used to say "NO same-eye motion reached DLSS" for up to 7 s after every F6, which reads as "it did
//   not work" (while the log had 537 of 537 fits valid)
//   no feature yet, indices out of range, no names at all: drawn, nothing asked for (and no crash)
//   the GPU times shown, one of them under its floor: drawn, nothing asked for
// Control, must FAIL: OVERLAY_TEST_NOCLICK=1 (no click is made, so a pass would not have come from the click).
static int SectionTest()
{
    char value[8] = {};
    const bool noClick = GetEnvironmentVariableA("OVERLAY_TEST_NOCLICK", value, sizeof(value)) > 0 && value[0] == '1';
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
    ID3D11Device* device = nullptr;
    ID3D11DeviceContext* context = nullptr;
    if (FAILED(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, nullptr, 0, D3D11_SDK_VERSION, &device,
                                 nullptr, &context)))
    {
        std::printf("FAILED: no D3D11 device\n");
        return 1;
    }
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.IniFilename = io.LogFilename = nullptr;
    io.DisplaySize = ImVec2(1000.0f, 1000.0f);
    if (!ImGui_ImplDX11_Init(device, context))
    {
        std::printf("FAILED: ImGui's D3D11 renderer\n");
        return 1;
    }
    D3D11_TEXTURE2D_DESC desc {};
    desc.Width = desc.Height = 1024;
    desc.MipLevels = desc.ArraySize = 1;
    desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.SampleDesc.Count = 1;
    desc.BindFlags = D3D11_BIND_RENDER_TARGET;
    ID3D11Texture2D* texture = nullptr;
    ID3D11RenderTargetView* target = nullptr;
    if (FAILED(device->CreateTexture2D(&desc, nullptr, &texture)) ||
        FAILED(device->CreateRenderTargetView(texture, nullptr, &target)))
    {
        std::printf("FAILED: the test's texture\n");
        return 1;
    }

    struct Case
    {
        const char* name;
        PerEyeMenu::Status status;
        bool click;
        int wantToggles;
        bool wantWarned;
        int wantStart = -1; // the saved start asked for with the switch
    };
    PerEyeMenu::Status named;
    named.twin = true;
    named.variantNames[0] = "variant 0";
    named.variantNames[1] = "variant 1";
    named.variantNames[2] = "variant 2";
    named.variantNames[3] = "variant 3";
    named.viewNames[0] = "Off";
    named.viewNames[1] = "view 1";
    named.viewNames[2] = "view 2";
    named.windowPresents = 600;
    Case cases[8] = { { "shared, \"On\" is chosen", named, true, 1, false, 1 },
                      { "per eye with same-eye motion", named, false, 0, false },
                      { "per eye WITHOUT same-eye motion", named, false, 0, true },
                      { "no feature yet, indices out of range, no names", PerEyeMenu::Status {}, false, 0, false },
                      { "the GPU times shown, one of them under its floor", named, false, 0, false },
                      { "per eye since a switch, the last window counted shared", named, false, 0, false },
                      { "per eye, \"On\" is chosen again: nothing is asked", named, true, 0, false },
                      { "per eye and DLSS not called at all (the game's TAA)", named, false, 0, false } };
    cases[0].status.toGame = 600;
    cases[6].status.routing = cases[6].status.windowPerEye = true;
    cases[6].status.toGame = cases[6].status.toTwin = 300;
    cases[6].status.sameEye = cases[6].status.fitsRead = cases[6].status.fitsValid = 600;
    cases[7].status.routing = cases[7].status.windowPerEye = true;
    cases[1].status.routing = cases[2].status.routing = true;
    cases[1].status.windowPerEye = cases[2].status.windowPerEye = true;
    cases[5].status.routing = true;
    cases[5].status.toGame = 600;
    cases[1].status.toGame = cases[1].status.toTwin = cases[2].status.toGame = cases[2].status.toTwin = 300;
    cases[1].status.sameEye = cases[1].status.fitsRead = 600;
    cases[1].status.fitsValid = 590;
    cases[1].status.fitResidual = 0.05;
    cases[1].status.fitInliers = 97.0;
    cases[2].status.chainBreaks = 600;
    cases[3].status.variant = 9;
    cases[3].status.view = -3;
    cases[4].status.toGame = 600;
    cases[4].status.timing = true;
    cases[4].status.times.presents = 600;
    cases[4].status.times.lines[StageTimes::Upscaler] = { 600, 3.2, 0.2, true };
    cases[4].status.times.lines[StageTimes::SameEyeMotion] = { 600, 0.003, 0.1, false };
    cases[4].status.times.lines[StageTimes::RenderThread] = { 600, 9.0, 0.0, true };

    int failures = 0, checked = 0;
    for (const Case& c : cases)
    {
        int toggles = 0, other = 0, vertices = 0, startAsked = -1;
        bool warned = false, opened = false;
        float at[2] = { 950.0f, 950.0f };
        for (int frame = 1; frame <= 16; ++frame)
        {
            io.DeltaTime = 1.0f / 90.0f;
            if (c.click && !noClick && frame >= 2)
                io.AddMousePosEvent(at[0], at[1]);
            else
                io.AddMousePosEvent(950.0f, 950.0f);
            if (c.click && !noClick && (frame == 4 || frame == 12))
                io.AddMouseButtonEvent(0, true);
            if (c.click && !noClick && (frame == 6 || frame == 14))
                io.AddMouseButtonEvent(0, false);
            ImGui_ImplDX11_NewFrame();
            ImGui::NewFrame();
            ImGui::SetNextWindowPos(ImVec2(20.0f, 20.0f));
            ImGui::SetNextWindowSize(ImVec2(900.0f, 700.0f));
            ImGui::Begin("OptiScaler v-test - game.exe", nullptr, ImGuiWindowFlags_NoSavedSettings);
            PerEyeMenu::Answer answer = PerEyeMenu::Draw(c.status, 1.0f);
            // As the game DLL draws it, but open: the drawer's controls are drawn on every frame too.
            ImGui::SetNextItemOpen(true);
            if (ImGui::TreeNode("Junk Drawer"))
            {
                PerEyeMenu::DrawJunkDrawer(c.status, 1.0f, answer);
                ImGui::TreePop();
            }
            ImGui::End();
            ImGui::Render();
            const float transparent[4] = {};
            context->OMSetRenderTargets(1, &target, nullptr);
            context->ClearRenderTargetView(target, transparent);
            ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());
            toggles += answer.toggleRouting ? 1 : 0;
            // The saved start goes with a switch (the dropdown); without one it is another change.
            if (answer.startPerEye >= 0 && answer.toggleRouting)
                startAsked = answer.startPerEye;
            other += (answer.startPerEye >= 0 && !answer.toggleRouting) + (answer.variant >= 0) + (answer.view >= 0) +
                     (answer.timing >= 0);
            warned = warned || answer.warned;
            vertices = ImGui::GetDrawData()->TotalVtxCount;
            opened = opened || answer.fixOpen;
            // The line clicked is "On" (the second).
            const bool choose = frame >= 8 && answer.fixOpen;
            at[0] = choose ? answer.fixAt[1][0] : answer.fixBoxAt[0];
            at[1] = choose ? answer.fixAt[1][1] : answer.fixBoxAt[1];
        }
        const bool inWindow = at[0] > 20.0f && at[0] < 920.0f && at[1] > 20.0f && at[1] < 720.0f;
        const bool ok = toggles == c.wantToggles && startAsked == c.wantStart && other == 0 &&
                        warned == c.wantWarned && vertices > 100 && inWindow && (!c.click || noClick || opened);
        std::printf("section, %s: routing switches asked for %d (wanted %d), the saved start with them %d (wanted "
                    "%d), other changes %d (0), warning %s (wanted %s), %d vertices (> 100), the dropdown opened: "
                    "%s, the click at %.0f,%.0f (in the window: %s) -> %s\n",
                    c.name, toggles, c.wantToggles, startAsked, c.wantStart, other, warned ? "shown" : "not shown",
                    c.wantWarned ? "shown" : "not shown", vertices, !c.click ? "not asked" : opened ? "yes" : "NO",
                    at[0], at[1], inWindow ? "yes" : "NO", ok ? "OK" : "FAIL");
        ++checked;
        if (!ok)
            ++failures;
    }
    ImGui_ImplDX11_Shutdown();
    ImGui::DestroyContext();
    std::printf("%s: %d cases of the section checked, %d failures\n", failures == 0 && checked > 0 ? "PASSED" : "FAILED",
                checked, failures);
    return failures == 0 && checked > 0 ? 0 : 1;
}

// ---- "overlay_test fovea": the Vulkan crop's block of the menu (OptiScaler/inputs/FoveaMenu_Vk.h, "Foveated SR"
// under "Per-eye DLSS (Vulkan)"; a Vulkan game), the very code the game DLL draws, on a real ImGui frame.
// Written with the block: F4 is the game's own key, so the overlay is the crop's only switch there, and a
// block that asked for the wrong thing (or nothing) would cost a headset launch.
//   crop off, "Crop on" clicked: exactly one switch of the crop asked for, nothing else
//   preset E, "M" chosen from the dropdown (one click opens it, a second chooses): preset 13 asked for, no switch
//   crop on with its box and a window of counts: drawn, nothing asked for
//   the crop failed: drawn, nothing asked for
// Control, must FAIL: OVERLAY_TEST_NOCLICK=1 (no click is made, so a pass would not have come from the click).
static int FoveaTest()
{
    char value[8] = {};
    const bool noClick = GetEnvironmentVariableA("OVERLAY_TEST_NOCLICK", value, sizeof(value)) > 0 && value[0] == '1';
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
    ID3D11Device* device = nullptr;
    ID3D11DeviceContext* context = nullptr;
    if (FAILED(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, nullptr, 0, D3D11_SDK_VERSION, &device,
                                 nullptr, &context)))
    {
        std::printf("FAILED: no D3D11 device\n");
        return 1;
    }
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.IniFilename = io.LogFilename = nullptr;
    io.DisplaySize = ImVec2(1000.0f, 1000.0f);
    if (!ImGui_ImplDX11_Init(device, context))
    {
        std::printf("FAILED: ImGui's D3D11 renderer\n");
        return 1;
    }
    D3D11_TEXTURE2D_DESC desc {};
    desc.Width = desc.Height = 1024;
    desc.MipLevels = desc.ArraySize = 1;
    desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.SampleDesc.Count = 1;
    desc.BindFlags = D3D11_BIND_RENDER_TARGET;
    ID3D11Texture2D* texture = nullptr;
    ID3D11RenderTargetView* target = nullptr;
    if (FAILED(device->CreateTexture2D(&desc, nullptr, &texture)) ||
        FAILED(device->CreateRenderTargetView(texture, nullptr, &target)))
    {
        std::printf("FAILED: the test's texture\n");
        return 1;
    }

    enum Click { None, CropBox, PresetM };
    struct Case
    {
        const char* name;
        FoveaMenuVk::Status status;
        Click click;
        int wantToggles;
        int wantPreset; // -1 = none
    };
    FoveaMenuVk::Status running;
    running.on = true;
    running.preset = 13;
    running.cropW = running.cropH = 1044;
    running.cropOutW = running.cropOutH = 1566;
    running.cropX = running.cropY = 1176;
    running.eyeW = running.eyeH = 3920;
    running.windowEvals = 600;
    running.drawn = 600;
    running.crops = 2;
    running.eyeByParity = true;
    FoveaMenuVk::Status atE;
    atE.preset = 5;
    FoveaMenuVk::Status failed;
    failed.failed = true;
    failed.on = false;
    Case cases[4] = { { "crop off, \"Crop on\" clicked", FoveaMenuVk::Status {}, CropBox, 1, -1 },
                      { "preset E, \"M\" chosen from the dropdown", atE, PresetM, 0, 13 },
                      { "crop on with its box and a window of counts", running, None, 0, -1 },
                      { "the crop failed", failed, None, 0, -1 } };

    int failures = 0, checked = 0;
    for (const Case& c : cases)
    {
        int toggles = 0, other = 0, vertices = 0, presetAsked = -1, presetsAsked = 0;
        bool opened = false;
        const bool click = c.click != None && !noClick;
        float at[2] = { 950.0f, 950.0f };
        for (int frame = 1; frame <= 16; ++frame)
        {
            io.DeltaTime = 1.0f / 90.0f;
            if (click && frame >= 2)
                io.AddMousePosEvent(at[0], at[1]);
            else
                io.AddMousePosEvent(950.0f, 950.0f);
            // The checkbox takes one click; the dropdown one to open and one to choose.
            const bool second = c.click == PresetM;
            if (click && (frame == 4 || (second && frame == 12)))
                io.AddMouseButtonEvent(0, true);
            if (click && (frame == 6 || (second && frame == 14)))
                io.AddMouseButtonEvent(0, false);
            ImGui_ImplDX11_NewFrame();
            ImGui::NewFrame();
            ImGui::SetNextWindowPos(ImVec2(20.0f, 20.0f));
            ImGui::SetNextWindowSize(ImVec2(900.0f, 700.0f));
            ImGui::Begin("OptiScaler v-test - game.exe", nullptr, ImGuiWindowFlags_NoSavedSettings);
            FoveaMenuVk::Answer answer = FoveaMenuVk::Draw(c.status, 1.0f);
            // The Junk Drawer's part, drawn as the game DLL draws it (under a tree, open here): the code runs on a
            // real frame, and nothing in it may be asked for by the clicks above.
            ImGui::SetNextItemOpen(true);
            if (ImGui::TreeNode("Junk Drawer"))
            {
                FoveaMenuVk::DrawJunk(c.status, 1.0f, answer);
                ImGui::TreePop();
            }
            ImGui::End();
            ImGui::Render();
            const float transparent[4] = {};
            context->OMSetRenderTargets(1, &target, nullptr);
            context->ClearRenderTargetView(target, transparent);
            ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());
            toggles += answer.toggleCrop ? 1 : 0;
            if (answer.preset >= 0)
            {
                presetAsked = answer.preset;
                ++presetsAsked;
            }
            other += (answer.toggleOutline ? 1 : 0) + (answer.startOn >= 0) + (answer.width > 0.0f) +
                     (answer.feather >= 0) + (answer.periphery >= 0) + (answer.peripheryPreset >= 0) +
                     (answer.copyInputs >= 0) + (answer.cropMask >= 0) + (answer.peripheryTaa >= 0) +
                     (answer.taaFlipJitter >= 0) + (answer.taaBlend > 0.0f) + (answer.taaScale > 0.0f);
            vertices = ImGui::GetDrawData()->TotalVtxCount;
            opened = opened || answer.presetOpen;
            if (c.click == CropBox)
            {
                at[0] = answer.cropBoxAt[0];
                at[1] = answer.cropBoxAt[1];
            }
            else if (c.click == PresetM)
            {
                // The line clicked is "M" (the last).
                const bool choose = frame >= 8 && answer.presetOpen;
                at[0] = choose ? answer.presetAt[FoveaMenuVk::kPresets - 1][0] : answer.presetBoxAt[0];
                at[1] = choose ? answer.presetAt[FoveaMenuVk::kPresets - 1][1] : answer.presetBoxAt[1];
            }
        }
        // The click's place is checked only where a click is made (the pointer rests far off otherwise).
        const bool inWindow = c.click == None || (at[0] > 20.0f && at[0] < 920.0f && at[1] > 20.0f && at[1] < 720.0f);
        const bool ok = toggles == c.wantToggles && presetAsked == c.wantPreset && presetsAsked <= 1 && other == 0 &&
                        vertices > 100 && inWindow && (c.click != PresetM || noClick || opened);
        std::printf("fovea, %s: crop switches asked for %d (wanted %d), preset asked for %d (wanted %d) %d time(s), "
                    "other changes %d (0), %d vertices (> 100), the dropdown opened: %s, the click at %.0f,%.0f (in "
                    "the window: %s) -> %s\n",
                    c.name, toggles, c.wantToggles, presetAsked, c.wantPreset, presetsAsked, other, vertices,
                    c.click != PresetM ? "not asked" : opened ? "yes" : "NO", at[0], at[1], inWindow ? "yes" : "NO",
                    ok ? "OK" : "FAIL");
        ++checked;
        if (!ok)
            ++failures;
    }
    ImGui_ImplDX11_Shutdown();
    ImGui::DestroyContext();
    std::printf("%s: %d cases of the Vulkan crop's block checked, %d failures\n",
                failures == 0 && checked > 0 ? "PASSED" : "FAILED", checked, failures);
    return failures == 0 && checked > 0 ? 0 : 1;
}


// ---- "overlay_test stale": the stale-eye warp's controls (OptiScaler/inputs/StaleEyeMenu.h), the very code the game
// DLL draws, on a real ImGui frame in a window as large as the headset panel's: the simple part (ONE dropdown, Off / Warp, the
// player's distance with Warp, and its status) and the Junk Drawer's part, drawn open. The settings are the test's own (in the game the caller reads
// them from the layer and hands them back when one was changed).
//   Off, "Warp" is chosen: the mode becomes 3 (Warp), changed once, nothing else changed
//   Warp, "Warp" is chosen again: nothing is changed
//   Warp, the distance's + is clicked once: 5.0 becomes 5.1, changed once
//   Off: the distance is NOT shown; Warp: it is
//   Pass (a test mode): the dropdown shows it as a test state, the drawer its four modes; nothing changed
//   Warp, the layer says why it skips: the reason is shown
//   Warp, Smooth, the player's distance off: the warning is shown
//   Diagnostics on: every test control is drawn (many more vertices), nothing changed
//   Far Cry 5's form (no "The player is"), Per pixel: drawn, nothing changed
//   nothing filled in at all: drawn, nothing changed (and no crash)
// Control, must FAIL: OVERLAY_TEST_NOCLICK=1 (no click is made, so a pass would not have come from the click).
static int StaleTest()
{
    char value[8] = {};
    const bool noClick = GetEnvironmentVariableA("OVERLAY_TEST_NOCLICK", value, sizeof(value)) > 0 && value[0] == '1';
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
    ID3D11Device* device = nullptr;
    ID3D11DeviceContext* context = nullptr;
    if (FAILED(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, nullptr, 0, D3D11_SDK_VERSION, &device,
                                 nullptr, &context)))
    {
        std::printf("FAILED: no D3D11 device\n");
        return 1;
    }
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.IniFilename = io.LogFilename = nullptr;
    io.DisplaySize = ImVec2(1000.0f, 1000.0f);
    if (!ImGui_ImplDX11_Init(device, context))
    {
        std::printf("FAILED: ImGui's D3D11 renderer\n");
        return 1;
    }
    D3D11_TEXTURE2D_DESC desc {};
    desc.Width = desc.Height = 1024;
    desc.MipLevels = desc.ArraySize = 1;
    desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.SampleDesc.Count = 1;
    desc.BindFlags = D3D11_BIND_RENDER_TARGET;
    ID3D11Texture2D* texture = nullptr;
    ID3D11RenderTargetView* target = nullptr;
    if (FAILED(device->CreateTexture2D(&desc, nullptr, &texture)) ||
        FAILED(device->CreateRenderTargetView(texture, nullptr, &target)))
    {
        std::printf("FAILED: the test's texture\n");
        return 1;
    }

    struct Case
    {
        const char* name;
        XrObserverStaleSettingsV1 settings;
        XrObserverStaleStatusV1 status;
        bool marksCharacters = true;
        int clickLine = -1;     // the line of the dropdown that is clicked (1 = Warp), -1: no click
        int wantChanges = 0;    // frames in which a setting was changed
        int wantMode = 0;       // the mode at the end
        bool wantSkip = false, wantNearOff = false;
        int minVertices = 100;
        bool clickPlus = false;      // the + of the player's distance is clicked, once
        float wantNear = -1.0f;      // the distance at the end, when it is checked
        bool wantDistance = true;    // the distance is shown
    };
    // The everyday settings of a game that marks its characters: Warp, Smooth, the player by the game's marks at 5.0, behind him from the other eye.
    XrObserverStaleSettingsV1 everyday {};
    everyday.structSize = sizeof(everyday);
    everyday.mode = 3;
    everyday.pairingOffset = 2;
    everyday.nearThreshold = 5.0f;
    everyday.headMode = 1;
    everyday.depthMode = 3;
    everyday.attachDistanceOnly = 1;
    everyday.nearWithBody = 1;
    everyday.fillFromOtherEye = 1;
    XrObserverStaleStatusV1 running {};
    running.structSize = sizeof(running);
    running.session = 1;
    running.worldScale = 1.0f;
    running.processedPerSecond = running.warpedPerSecond = 90.0f;
    running.videoMemoryMb = 9000.0f;
    running.videoBudgetMb = 30000.0f;
    running.ownMemoryMb = 600.0f;

    Case cases[10];
    int n = 0;
    cases[n] = { "Off, \"Warp\" is chosen", everyday, running, true, 1, 1, 3 };
    cases[n++].settings.mode = 0;
    cases[n++] = { "Warp, \"Warp\" is chosen again: nothing is changed", everyday, running, true, 1, 0, 3 };
    cases[n] = { "Warp, the distance's + is clicked once", everyday, running, true, -1, 1, 3 };
    cases[n].clickPlus = true;
    cases[n++].wantNear = 5.1f;
    cases[n] = { "Off: the distance is not shown", everyday, running, true, -1, 0, 0 };
    cases[n].settings.mode = 0;
    cases[n++].wantDistance = false;
    cases[n] = { "Pass (a test mode) shown as a test state", everyday, running, true, -1, 0, 1 };
    cases[n].wantDistance = false;
    cases[n++].settings.mode = 1;
    cases[n] = { "Warp, the layer says why it skips", everyday, running, true, -1, 0, 3 };
    strcpy_s(cases[n].status.lastSkip, "no depth of this Present");
    cases[n++].wantSkip = true;
    cases[n] = { "Warp, Smooth, the player's distance off", everyday, running, true, -1, 0, 3 };
    cases[n].settings.nearThreshold = 0.0f;
    cases[n++].wantNearOff = true;
    cases[n] = { "Diagnostics on: the test controls are drawn", everyday, running, true, -1, 0, 3 };
    cases[n].settings.diagnostics = 1;
    cases[n].settings.treatMode = 2;
    cases[n].settings.treatGated = 1;
    cases[n++].minVertices = 3000;
    cases[n] = { "Far Cry 5's form, Per pixel", everyday, running, false, -1, 0, 3 };
    cases[n++].settings.depthMode = 0;
    cases[n] = { "nothing filled in", XrObserverStaleSettingsV1 {}, XrObserverStaleStatusV1 {}, true, -1, 0, 0 };
    cases[n++].wantDistance = false;
    if (n != 10)
    {
        std::printf("FAILED: %d cases written, 10 declared\n", n);
        return 1;
    }

    int failures = 0, checked = 0;
    for (const Case& c : cases)
    {
        XrObserverStaleSettingsV1 s = c.settings;
        int changes = 0, marksAsked = 0, vertices = 0;
        bool skip = false, nearOff = false, opened = false, distance = false;
        float at[2] = { 950.0f, 950.0f };
        // A dropdown takes two clicks (open, choose); the + one.
        const bool click = (c.clickLine >= 0 || c.clickPlus) && !noClick;
        for (int frame = 1; frame <= 16; ++frame)
        {
            io.DeltaTime = 1.0f / 90.0f;
            if (click && frame >= 2)
                io.AddMousePosEvent(at[0], at[1]);
            else
                io.AddMousePosEvent(950.0f, 950.0f);
            if (click && (frame == 4 || (frame == 12 && !c.clickPlus)))
                io.AddMouseButtonEvent(0, true);
            if (click && (frame == 6 || (frame == 14 && !c.clickPlus)))
                io.AddMouseButtonEvent(0, false);
            ImGui_ImplDX11_NewFrame();
            ImGui::NewFrame();
            ImGui::SetNextWindowPos(ImVec2(20.0f, 20.0f));
            ImGui::SetNextWindowSize(ImVec2(900.0f, 960.0f));
            ImGui::Begin("OptiScaler v-test - game.exe", nullptr, ImGuiWindowFlags_NoSavedSettings);
            StaleEyeMenu::Answer answer;
            ImGui::SeparatorText("Stale-eye warp");
            StaleEyeMenu::DrawSimple(s, answer, c.marksCharacters);
            StaleEyeMenu::DrawStatusSimple(s, c.status, answer);
            // As the game DLL draws it, but open: the drawer's controls are drawn on every frame too.
            ImGui::SetNextItemOpen(true);
            if (ImGui::TreeNode("Junk Drawer"))
            {
                StaleEyeMenu::DrawJunk(s, answer, c.marksCharacters, true);
                StaleEyeMenu::DrawStatusJunk(s, c.status);
                ImGui::TreePop();
            }
            ImGui::End();
            ImGui::Render();
            const float transparent[4] = {};
            context->OMSetRenderTargets(1, &target, nullptr);
            context->ClearRenderTargetView(target, transparent);
            ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());
            changes += answer.changed ? 1 : 0;
            marksAsked += answer.playerMarks >= 0 ? 1 : 0;
            skip = skip || answer.skipShown;
            nearOff = nearOff || answer.nearOffWarned;
            opened = opened || answer.modeOpen;
            distance = answer.distanceShown; // as in the last frame
            vertices = ImGui::GetDrawData()->TotalVtxCount;
            if (c.clickPlus)
            {
                at[0] = answer.distancePlusAt[0];
                at[1] = answer.distancePlusAt[1];
            }
            if (c.clickLine >= 0)
            {
                const bool choose = frame >= 8 && answer.modeOpen;
                at[0] = choose ? answer.modeAt[c.clickLine][0] : answer.modeBoxAt[0];
                at[1] = choose ? answer.modeAt[c.clickLine][1] : answer.modeBoxAt[1];
            }
        }
        const bool inWindow = (c.clickLine < 0 && !c.clickPlus) ||
                              (at[0] > 20.0f && at[0] < 920.0f && at[1] > 20.0f && at[1] < 980.0f);
        const bool nearOk = c.wantNear < 0.0f || std::fabs(s.nearThreshold - c.wantNear) < 0.01f;
        const bool ok = changes == c.wantChanges && s.mode == c.wantMode && marksAsked == 0 && skip == c.wantSkip &&
                        nearOff == c.wantNearOff && vertices > c.minVertices && inWindow && nearOk &&
                        distance == c.wantDistance && (c.clickLine < 0 || noClick || opened);
        std::printf("stale, %s: the distance %s (wanted %s), at the end %.2f (wanted %.2f, -1 = not checked); ", c.name,
                    distance ? "shown" : "not shown", c.wantDistance ? "shown" : "not shown", s.nearThreshold,
                    c.wantNear);
        std::printf("stale, %s: settings changed in %d frames (wanted %d), the mode at the end %d (wanted %d), \"The "
                    "player is\" asked %d times (0), the skip's reason %s (wanted %s), the distance's warning %s "
                    "(wanted %s), %d vertices (> %d), the dropdown opened: %s, the click at %.0f,%.0f (in the "
                    "window: %s) -> %s\n",
                    c.name, changes, c.wantChanges, s.mode, c.wantMode, marksAsked, skip ? "shown" : "not shown",
                    c.wantSkip ? "shown" : "not shown", nearOff ? "shown" : "not shown",
                    c.wantNearOff ? "shown" : "not shown", vertices, c.minVertices,
                    c.clickLine < 0 ? "not asked" : opened ? "yes" : "NO", at[0], at[1], inWindow ? "yes" : "NO",
                    ok ? "OK" : "FAIL");
        ++checked;
        if (!ok)
            ++failures;
    }
    ImGui_ImplDX11_Shutdown();
    ImGui::DestroyContext();
    std::printf("%s: %d cases of the stale-eye warp's controls checked, %d failures\n",
                failures == 0 && checked > 0 ? "PASSED" : "FAILED", checked, failures);
    return failures == 0 && checked > 0 ? 0 : 1;
}

// ---- "overlay_test mainthread": a callback run on a window's thread in message context, asked from another thread
// (OptiScaler/inputs/MainThreadCall.h; a game's engine resize is asked this way). A real window on a thread of its
// own with a message pump; the main thread attaches (subclasses) it, posts three requests and sends the window an
// ordinary message of its own:
//   the three callbacks ran, each on the WINDOW's thread (not the main thread), in order
//   the ordinary message reached the procedure that was there (counted by the window's class)
//   a second Attach of the same window is a yes; of another window a no, with a reason
// Control, must FAIL: OVERLAY_TEST_NOPOST=1 (nothing is posted, so the callbacks cannot have run).
namespace
{
struct MainThreadWindow
{
    HWND window = nullptr;
    DWORD thread = 0;
    HANDLE ready = nullptr;
    std::atomic<int> ordinary { 0 }; // WM_USER + 7 seen by the class's own procedure
};
MainThreadWindow g_mainThreadWindow;
std::atomic<int> g_callbacksRan { 0 };
DWORD g_callbackThreads[3] = {};
int g_callbackOrder[3] = {};

LRESULT CALLBACK MainThreadClassProc(HWND window, UINT message, WPARAM wParam, LPARAM lParam)
{
    if (message == WM_USER + 7)
    {
        ++g_mainThreadWindow.ordinary;
        return 7;
    }
    if (message == WM_DESTROY)
    {
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProcW(window, message, wParam, lParam);
}

DWORD WINAPI MainThreadWindowThread(LPVOID)
{
    WNDCLASSW wc {};
    wc.lpfnWndProc = MainThreadClassProc;
    wc.hInstance = GetModuleHandleW(nullptr);
    wc.lpszClassName = L"OverlayTestMainThreadWindow";
    RegisterClassW(&wc);
    g_mainThreadWindow.window = CreateWindowExW(0, wc.lpszClassName, L"main thread test", WS_OVERLAPPEDWINDOW, 10, 10,
                                                200, 100, nullptr, nullptr, wc.hInstance, nullptr);
    g_mainThreadWindow.thread = GetCurrentThreadId();
    SetEvent(g_mainThreadWindow.ready);
    if (g_mainThreadWindow.window == nullptr)
        return 1;
    MSG msg;
    while (GetMessageW(&msg, nullptr, 0, 0) > 0)
    {
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }
    return 0;
}

void MainThreadCallback(void* context)
{
    const int n = g_callbacksRan.load();
    if (n < 3)
    {
        g_callbackThreads[n] = GetCurrentThreadId();
        g_callbackOrder[n] = (int) (intptr_t) context;
    }
    ++g_callbacksRan;
}
} // namespace

static int MainThreadTest()
{
    char value[8] = {};
    const bool noPost = GetEnvironmentVariableA("OVERLAY_TEST_NOPOST", value, sizeof(value)) > 0 && value[0] == '1';
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
    g_mainThreadWindow.ready = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    HANDLE thread = CreateThread(nullptr, 0, MainThreadWindowThread, nullptr, 0, nullptr);
    if (thread == nullptr || WaitForSingleObject(g_mainThreadWindow.ready, 5000) != WAIT_OBJECT_0 ||
        g_mainThreadWindow.window == nullptr)
    {
        std::printf("FAILED: no window on a thread of its own\n");
        return 1;
    }
    auto& runner = MainThreadCall::Runner::Get();
    const char* why = nullptr;
    const bool attached = runner.Attach(g_mainThreadWindow.window, &why);
    const bool attachedAgain = runner.Attach(g_mainThreadWindow.window, &why);
    const char* otherWhy = nullptr;
    const bool attachedOther = runner.Attach(GetDesktopWindow(), &otherWhy);
    bool posted = true;
    if (!noPost)
        for (int i = 0; i < 3; ++i)
            posted = runner.Post(&MainThreadCallback, (void*) (intptr_t) (i + 1), &why) && posted;
    // An ordinary message of the window's own: it must reach the class's procedure through ours.
    const LRESULT ordinary = SendMessageW(g_mainThreadWindow.window, WM_USER + 7, 0, 0);
    // Wait for the callbacks (they run when the window's thread pumps our message).
    for (int i = 0; i < 200 && g_callbacksRan.load() < 3; ++i)
        Sleep(10);
    const int ran = g_callbacksRan.load();
    const bool onWindowThread = ran == 3 && g_callbackThreads[0] == g_mainThreadWindow.thread &&
                                g_callbackThreads[1] == g_mainThreadWindow.thread &&
                                g_callbackThreads[2] == g_mainThreadWindow.thread &&
                                g_mainThreadWindow.thread != GetCurrentThreadId();
    const bool inOrder = ran == 3 && g_callbackOrder[0] == 1 && g_callbackOrder[1] == 2 && g_callbackOrder[2] == 3;
    const bool passed = ordinary == 7 && g_mainThreadWindow.ordinary.load() == 1 && runner.Passed() >= 1;
    const bool ok = attached && attachedAgain && !attachedOther && otherWhy != nullptr && posted && ran == 3 &&
                    onWindowThread && inOrder && passed && runner.Ran() == 3 &&
                    runner.LastRanOn() == g_mainThreadWindow.thread;
    std::printf("mainthread: attached %s (again %s, another window %s: %s); posted %s; callbacks ran %d (wanted 3), "
                "on the window's thread %u (the main thread %u): %s, in order: %s; the ordinary message answered %lld "
                "(wanted 7) and reached the class's procedure %d times (wanted 1), %llu messages passed on -> %s\n",
                attached ? "yes" : "NO", attachedAgain ? "yes" : "NO", attachedOther ? "YES" : "no",
                otherWhy != nullptr ? otherWhy : "-", posted ? "yes" : "NO", ran, (unsigned) g_mainThreadWindow.thread,
                (unsigned) GetCurrentThreadId(), onWindowThread ? "yes" : "NO", inOrder ? "yes" : "NO",
                (long long) ordinary, g_mainThreadWindow.ordinary.load(), (unsigned long long) runner.Passed(),
                ok ? "OK" : "FAIL");
    PostMessageW(g_mainThreadWindow.window, WM_CLOSE, 0, 0);
    WaitForSingleObject(thread, 5000);
    std::printf("%s: the main-thread runner checked, %d failures\n", ok ? "PASSED" : "FAILED", ok ? 0 : 1);
    return ok ? 0 : 1;
}

// ---- The game DLL's DIRECT3D 12 side (a D3D12 game): overlay_test dx12 ----
// The menu drawn by ImGui's D3D12 renderer into a shared RGBA8 texture made as OverlayShare makes it
// (OverlayTextures12.h: a shared heap, an NT handle, a shared fence signalled behind the draw); then a D3D11 device --
// the layer's, as the OpenXR sessions it serves are D3D11 -- opens the texture and the fence through the handles, waits
// for the fence on its GPU and reads the texture back: opaque in the menu's middle, transparent in the margin, and the
// texture as the layer checks it (RGBA8 typeless, one level, one sample). Control, set by hand (must FAIL):
// OVERLAY_TEST_DX12_NODRAW=1 (the texture cleared, nothing drawn into it).
static int Dx12Test()
{
    char value[8] = {};
    const bool noDraw = GetEnvironmentVariableA("OVERLAY_TEST_DX12_NODRAW", value, sizeof(value)) > 0 && value[0] == '1';
    // The window's format ImGui draws in: 28 RGBA8 (default), 87 BGRA8, 24 R10G10B10A2 (OVERLAY_TEST_DX12_FORMAT).
    char formatText[8] = {};
    const DXGI_FORMAT rtvFormat =
        GetEnvironmentVariableA("OVERLAY_TEST_DX12_FORMAT", formatText, sizeof(formatText)) > 0
            ? (DXGI_FORMAT) std::atoi(formatText)
            : DXGI_FORMAT_R8G8B8A8_UNORM;
    const DXGI_FORMAT family = OverlayTextures12::TypelessFor(rtvFormat);
    const bool tenBit = family == DXGI_FORMAT_R10G10B10A2_TYPELESS;
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
    ID3D12Device* device = nullptr;
    if (FAILED(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device))) || device == nullptr)
    {
        std::printf("FAILED: no D3D12 device\n");
        return 1;
    }
    D3D12_COMMAND_QUEUE_DESC queueDesc {};
    queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    ID3D12CommandQueue* queue = nullptr;
    ID3D12CommandAllocator* allocator = nullptr;
    ID3D12GraphicsCommandList* list = nullptr;
    ID3D12DescriptorHeap* rtvHeap = nullptr;
    ID3D12DescriptorHeap* srvHeap = nullptr;
    D3D12_DESCRIPTOR_HEAP_DESC rtvDesc {};
    rtvDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    rtvDesc.NumDescriptors = 1;
    D3D12_DESCRIPTOR_HEAP_DESC srvDesc {};
    srvDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    srvDesc.NumDescriptors = 8;
    srvDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    if (FAILED(device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&queue))) ||
        FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator))) ||
        FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator, nullptr, IID_PPV_ARGS(&list))) ||
        FAILED(list->Close()) || FAILED(device->CreateDescriptorHeap(&rtvDesc, IID_PPV_ARGS(&rtvHeap))) ||
        FAILED(device->CreateDescriptorHeap(&srvDesc, IID_PPV_ARGS(&srvHeap))))
    {
        std::printf("FAILED: the D3D12 queue, list or heaps\n");
        return 1;
    }
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.IniFilename = io.LogFilename = nullptr;
    io.DisplaySize = ImVec2(8192.0f, 8192.0f);
    // ImGui's D3D12 renderer as the menu initialises it: an SRV heap with a hand-out allocator (here: 1 descriptor,
    // the font; never freed).
    static ID3D12DescriptorHeap* s_srvHeap = nullptr;
    static int s_srvGiven = 0;
    s_srvHeap = srvHeap;
    s_srvGiven = 0;
    ImGui_ImplDX12_InitInfo initInfo {};
    initInfo.Device = device;
    initInfo.CommandQueue = queue;
    initInfo.NumFramesInFlight = 2;
    initInfo.RTVFormat = rtvFormat;
    initInfo.DSVFormat = DXGI_FORMAT_UNKNOWN;
    initInfo.SrvDescriptorHeap = srvHeap;
    initInfo.SrvDescriptorAllocFn = [](ImGui_ImplDX12_InitInfo* info, D3D12_CPU_DESCRIPTOR_HANDLE* cpu,
                                       D3D12_GPU_DESCRIPTOR_HANDLE* gpu)
    {
        const UINT step = info->Device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        *cpu = s_srvHeap->GetCPUDescriptorHandleForHeapStart();
        *gpu = s_srvHeap->GetGPUDescriptorHandleForHeapStart();
        cpu->ptr += (SIZE_T) step * (SIZE_T) s_srvGiven;
        gpu->ptr += (UINT64) step * (UINT64) s_srvGiven;
        ++s_srvGiven;
    };
    initInfo.SrvDescriptorFreeFn = [](ImGui_ImplDX12_InitInfo*, D3D12_CPU_DESCRIPTOR_HANDLE, D3D12_GPU_DESCRIPTOR_HANDLE) {};
    if (!ImGui_ImplDX12_Init(&initInfo))
    {
        std::printf("FAILED: ImGui's D3D12 renderer\n");
        return 1;
    }
    constexpr int kMargin = 16, kMaxSide = 4096, kW = 1024, kH = 1024;
    ID3D12Resource* texture = nullptr;
    HANDLE handle = nullptr;
    HRESULT hrTexture = S_OK, hrFence = S_OK;
    const D3D12_CPU_DESCRIPTOR_HANDLE rtv = rtvHeap->GetCPUDescriptorHandleForHeapStart();
    const bool made = OverlayTextures12::MakeShared(device, kW, kH, rtvFormat, rtv, &texture, &handle, &hrTexture);
    ID3D12Fence* fence = nullptr;
    HANDLE fenceHandle = nullptr;
    const bool fenced = OverlayTextures12::MakeSharedFence(device, &fence, &fenceHandle, &hrFence);
    std::printf("D3D12: shared texture %dx%d for the window format %d (family %d) %s (%08lX), shared fence %s (%08lX)\n",
                kW, kH, (int) rtvFormat, (int) family, made ? "made" : "FAILED", (unsigned long) hrTexture,
                fenced ? "made" : "FAILED", (unsigned long) hrFence);
    if (!made || !fenced)
    {
        std::printf("FAILED: the shared objects\n");
        return 1;
    }
    // One ImGui frame with the menu, as the game DLL sees it after ImGui::Render.
    io.DeltaTime = 1.0f / 90.0f;
    ImGui_ImplDX12_NewFrame();
    ImGui::NewFrame();
    ImGui::SetNextWindowPos(ImVec2(3000.0f, 2500.0f));
    ImGui::SetNextWindowSize(ImVec2(900.0f, 600.0f));
    ImGui::Begin("OptiScaler v-test - game.exe", nullptr, ImGuiWindowFlags_NoSavedSettings);
    ImGui::Text("The menu");
    ImGui::Button("A button");
    ImGui::End();
    ImGui::Render();
    OverlayCrop::Rules rules;
    rules.primary = "OptiScaler v";
    OverlayCrop::State state;
    float scale = 1.0f;
    const bool have = OverlayCrop::Grow(ImGui::GetDrawData(), state, rules, kMargin, kMaxSide, &scale);
    // Recorded as OverlayShare::AfterDrawDx12 records it: COMMON -> render target -> COMMON around the draw.
    allocator->Reset();
    list->Reset(allocator, nullptr);
    D3D12_RESOURCE_BARRIER barrier {};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = texture;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_RENDER_TARGET;
    list->ResourceBarrier(1, &barrier);
    const float transparent[4] = {};
    list->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
    list->ClearRenderTargetView(rtv, transparent, 0, nullptr);
    list->SetDescriptorHeaps(1, &srvHeap);
    if (have && !noDraw)
    {
        ImDrawData through = OverlayCrop::Through(ImGui::GetDrawData(), state.crop, scale);
        ImGui_ImplDX12_RenderDrawData(&through, list);
    }
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COMMON;
    list->ResourceBarrier(1, &barrier);
    list->Close();
    ID3D12CommandList* lists[] = { list };
    queue->ExecuteCommandLists(1, lists);
    queue->Signal(fence, 1);

    // ---- The layer's side: a D3D11 device opens the texture and the fence by their handles, waits, reads ----
    ID3D11Device* other = nullptr;
    ID3D11DeviceContext* otherContext = nullptr;
    D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, nullptr, 0, D3D11_SDK_VERSION, &other, nullptr,
                      &otherContext);
    ID3D11Device1* other1 = nullptr;
    ID3D11Device5* other5 = nullptr;
    ID3D11DeviceContext4* context4 = nullptr;
    if (other != nullptr)
    {
        other->QueryInterface(IID_PPV_ARGS(&other1));
        other->QueryInterface(IID_PPV_ARGS(&other5));
        otherContext->QueryInterface(IID_PPV_ARGS(&context4));
    }
    ID3D11Texture2D* opened = nullptr;
    ID3D11Fence* fence11 = nullptr;
    const HRESULT open = other1 != nullptr ? other1->OpenSharedResource1(handle, IID_PPV_ARGS(&opened)) : E_FAIL;
    const HRESULT openFence = other5 != nullptr ? other5->OpenSharedFence(fenceHandle, IID_PPV_ARGS(&fence11)) : E_FAIL;
    std::printf("D3D11 (the layer's side): the texture opened %s (%08lX), the fence opened %s (%08lX)\n",
                SUCCEEDED(open) && opened != nullptr ? "yes" : "NO", (unsigned long) open,
                SUCCEEDED(openFence) && fence11 != nullptr ? "yes" : "NO", (unsigned long) openFence);
    int inside = -1, margin = -1;
    bool shape = false;
    if (opened != nullptr && fence11 != nullptr && context4 != nullptr)
    {
        D3D11_TEXTURE2D_DESC got {};
        opened->GetDesc(&got);
        shape = got.Width == kW && got.Height == kH && got.MipLevels == 1 && got.SampleDesc.Count == 1 &&
                got.Format == family;
        std::printf("  as the layer checks it: %ux%u, format %d (the family %d wanted), %u levels, %u samples -> %s\n",
                    got.Width, got.Height, (int) got.Format, (int) family, got.MipLevels, got.SampleDesc.Count,
                    shape ? "OK" : "FAIL");
        context4->Wait(fence11, 1); // GPU-side, as the layer waits before its copy
        D3D11_TEXTURE2D_DESC readDesc = got;
        readDesc.Format = rtvFormat; // a typed view of the same family: what the layer's panel swapchain is
        readDesc.Usage = D3D11_USAGE_STAGING;
        readDesc.BindFlags = 0;
        readDesc.MiscFlags = 0;
        readDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        ID3D11Texture2D* readBack = nullptr;
        D3D11_MAPPED_SUBRESOURCE mapped {};
        if (SUCCEEDED(other->CreateTexture2D(&readDesc, nullptr, &readBack)))
        {
            otherContext->CopyResource(readBack, opened);
            if (SUCCEEDED(otherContext->Map(readBack, 0, D3D11_MAP_READ, 0, &mapped)))
            {
                const auto* bytes = static_cast<const uint8_t*>(mapped.pData);
                // Alpha: byte 3 of RGBA8 / BGRA8; the top two bits of a 10-bit pixel (0..3 -> 0..255).
                const auto alphaAt = [&](int x, int y) {
                    const uint8_t* px = bytes + (size_t) y * mapped.RowPitch + (size_t) x * 4;
                    if (!tenBit)
                        return (int) px[3];
                    uint32_t word = 0;
                    std::memcpy(&word, px, 4);
                    return (int) ((word >> 30) * 85);
                };
                inside = alphaAt(kMargin + 450, kMargin + 300);
                margin = alphaAt(4, 4);
                otherContext->Unmap(readBack, 0);
            }
            readBack->Release();
        }
    }
    const bool ok = have && shape && inside >= 200 && margin == 0;
    std::printf("read back on the D3D11 device after its wait on the D3D12 fence: alpha in the menu's middle %d (>= 200), "
                "in the margin %d (0); rectangle %s -> %s\n",
                inside, margin, have ? "found" : "NOT found", ok ? "OK" : "FAIL");
    ImGui_ImplDX12_Shutdown();
    ImGui::DestroyContext();
    std::printf("%s: D3D12 shared texture + fence opened and read by a D3D11 device, %d failures\n",
                ok ? "PASSED" : "FAILED", ok ? 0 : 1);
    return ok ? 0 : 1;
}

int main(int argc, char** argv)
{
    if (argc > 1 && std::strcmp(argv[1], "dx12") == 0)
        return Dx12Test();
    if (argc > 1 && std::strcmp(argv[1], "section") == 0)
        return SectionTest();
    if (argc > 1 && std::strcmp(argv[1], "fovea") == 0)
        return FoveaTest();
    if (argc > 1 && std::strcmp(argv[1], "stale") == 0)
        return StaleTest();
    if (argc > 1 && std::strcmp(argv[1], "mainthread") == 0)
        return MainThreadTest();
    char value[8] = {};
    OverlayCrop::Rules rules;
    rules.primary = "OptiScaler v";
    rules.countPointer = GetEnvironmentVariableA("OVERLAY_TEST_POINTER", value, sizeof(value)) > 0 && value[0] == '1';
    rules.countPassive = GetEnvironmentVariableA("OVERLAY_TEST_PASSIVE", value, sizeof(value)) > 0 && value[0] == '1';
    const bool oldCall = GetEnvironmentVariableA("OVERLAY_TEST_OLDCALL", value, sizeof(value)) > 0 && value[0] == '1';
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX); // a crash ends the run; it must not wait for a click

    ID3D11Device* device = nullptr;
    ID3D11DeviceContext* context = nullptr;
    if (FAILED(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, nullptr, 0, D3D11_SDK_VERSION, &device,
                                 nullptr, &context)))
    {
        std::printf("FAILED: no D3D11 device\n");
        return 1;
    }
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.IniFilename = io.LogFilename = nullptr;
    io.DisplaySize = ImVec2(8192.0f, 8192.0f);
    io.MouseDrawCursor = true;
    if (!ImGui_ImplDX11_Init(device, context))
    {
        std::printf("FAILED: ImGui's D3D11 renderer\n");
        return 1;
    }

    constexpr int kMargin = 16, kMaxSide = 4096, kSide = 4096;
    D3D11_TEXTURE2D_DESC desc {};
    desc.Width = kSide;
    desc.Height = kSide;
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = DXGI_FORMAT_R8G8B8A8_TYPELESS;
    desc.SampleDesc.Count = 1;
    desc.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
    D3D11_RENDER_TARGET_VIEW_DESC view {};
    view.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    view.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2D;
    ID3D11Texture2D* texture = nullptr;
    ID3D11RenderTargetView* target = nullptr;
    ID3D11Texture2D* staging = nullptr;
    D3D11_TEXTURE2D_DESC read = desc;
    read.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    read.Usage = D3D11_USAGE_STAGING;
    read.BindFlags = 0;
    read.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    if (FAILED(device->CreateTexture2D(&desc, nullptr, &texture)) ||
        FAILED(device->CreateRenderTargetView(texture, &view, &target)) ||
        FAILED(device->CreateTexture2D(&read, nullptr, &staging)))
    {
        std::printf("FAILED: the test's textures\n");
        return 1;
    }

    struct Phase
    {
        const char* name;
        bool menu;
        float x, y, w, h;     // the menu's window
        int want[4];          // the rectangle wanted (x0, y0, x1, y1)
        float scale;          // and the scale it is drawn at
        float midX, midY;     // a point of the canvas that must be opaque in the panel
    };
    const float overlayX = 7600.0f, overlayY = 60.0f, overlayW = 400.0f, overlayH = 120.0f;
    const Phase phases[6] = {
        { "the menu", true, 3000, 2500, 900, 600, { 2984, 2484, 3916, 3116 }, 1.0f, 3450, 2800 },
        // Dragged over the top edge: the rectangle keeps its 932 x 632 and is moved back in (the panel's shape must not
        // change while the menu is dragged), the menu's visible part is in it.
        { "the menu dragged above the top (the size kept, moved back in)", true, 3000, -100, 900, 600,
          { 2984, 0, 3916, 632 }, 1.0f, 3450, 250 },
        { "the menu MOVED past 4096 and larger", true, 5200, 4700, 900, 800, { 5184, 4684, 6116, 5516 }, 1.0f, 5650, 5100 },
        { "the menu smaller again (the size is kept)", true, 5200, 4700, 900, 600, { 5184, 4684, 6116, 5516 }, 1.0f, 5650, 5000 },
        { "the menu larger than a panel holds", true, 1000, 2000, 5000, 3000, { 984, 1984, 6016, 5016 },
          4096.0f / 5032.0f, 3500, 3500 },
        { "the menu closed", false, 0, 0, 0, 0,
          { (int) overlayX - kMargin, (int) overlayY - kMargin, (int) (overlayX + overlayW) + kMargin,
            (int) (overlayY + overlayH) + kMargin },
          1.0f, overlayX + overlayW / 2, overlayY + overlayH / 2 },
    };

    int failures = 0, checked = 0;
    OverlayCrop::State state;
    for (int frame = 1; frame <= 18; ++frame)
    {
        const Phase& phase = phases[(frame - 1) / 3];
        io.DeltaTime = 1.0f / 90.0f;
        io.AddMousePosEvent(6000.0f, 7000.0f);
        ImGui_ImplDX11_NewFrame();
        ImGui::NewFrame();
        if (phase.menu)
        {
            ImGui::SetNextWindowPos(ImVec2(phase.x, phase.y));
            ImGui::SetNextWindowSize(ImVec2(phase.w, phase.h));
            ImGui::Begin("OptiScaler v-test - game.exe", nullptr, ImGuiWindowFlags_NoSavedSettings);
            ImGui::Text("The menu");
            ImGui::Button("A button");
            static float slider = 0.5f;
            ImGui::SliderFloat("A slider", &slider, 0.0f, 1.0f);
            ImGui::BeginChild("##CollapsingHeaderChild", ImVec2(0, 120), ImGuiChildFlags_Borders);
            ImGui::Text("A child window of the menu");
            ImGui::EndChild();
            ImGui::End();
        }
        ImGui::SetNextWindowPos(ImVec2(overlayX, overlayY));
        ImGui::SetNextWindowSize(ImVec2(overlayW, overlayH));
        ImGui::Begin("Performance Overlay", nullptr,
                     ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoInputs |
                         ImGuiWindowFlags_NoFocusOnAppearing);
        ImGui::Text("90.0 fps");
        ImGui::End();
        ImGui::Render();

        // ---- where the game DLL is called: after the frame's end ----
        ImDrawData* drawData = ImGui::GetDrawData();
        if (oldCall)
        {
            // What an earlier build did here. It must not come back: this run has to end in a crash, not in PASSED.
            std::printf("frame %d: calling ImGui::GetForegroundDrawList() after the frame's end, as an earlier build did\n", frame);
            std::fflush(stdout);
            const ImDrawList* pointer = ImGui::GetForegroundDrawList();
            std::printf("frame %d: it returned %p\n", frame, (const void*) pointer);
        }
        float scale = 1.0f;
        const bool have = OverlayCrop::Grow(drawData, state, rules, kMargin, kMaxSide, &scale);
        const float transparent[4] = {};
        context->OMSetRenderTargets(1, &target, nullptr);
        context->ClearRenderTargetView(target, transparent);
        if (have)
        {
            ImDrawData through = OverlayCrop::Through(drawData, state.crop, scale);
            ImGui_ImplDX11_RenderDrawData(&through);
        }
        context->CopyResource(staging, texture);
        D3D11_MAPPED_SUBRESOURCE mapped {};
        if (FAILED(context->Map(staging, 0, D3D11_MAP_READ, 0, &mapped)))
        {
            std::printf("frame %d: FAILED to read the texture back\n", frame);
            ++failures;
            continue;
        }
        const auto alphaAt = [&](int x, int y) {
            x = (std::min)((std::max)(x, 0), kSide - 1);
            y = (std::min)((std::max)(y, 0), kSide - 1);
            return (int) static_cast<const uint8_t*>(mapped.pData)[(size_t) y * mapped.RowPitch + (size_t) x * 4 + 3];
        };
        // The rectangle wanted, within 2 pixels (a window's border), and its scale.
        bool rectangle = have && state.menu == phase.menu && std::fabs(scale - phase.scale) < 0.002f;
        for (int i = 0; i < 4; ++i)
            rectangle = rectangle && std::abs(state.crop[i] - phase.want[i]) <= 2;
        // In the texture: the point of the canvas that must show is opaque (a window's background, alpha 0.94); the
        // margin's corner is transparent, and so is what lies beyond the rectangle.
        const int usedW = (int) ((state.crop[2] - state.crop[0]) * scale), usedH = (int) ((state.crop[3] - state.crop[1]) * scale);
        const int inside = alphaAt((int) ((phase.midX - state.crop[0]) * scale), (int) ((phase.midY - state.crop[1]) * scale));
        const int margin = alphaAt(4, 4);
        const bool room = usedW + 8 < kSide || usedH + 8 < kSide;
        const int beyond = room ? alphaAt(usedW + 8 < kSide ? usedW + 8 : 4, usedW + 8 < kSide ? 4 : usedH + 8) : 0;
        context->Unmap(staging, 0);
        const bool picture = inside >= 200 && margin == 0 && beyond == 0;
        const bool ok = rectangle && picture;
        std::printf("frame %2d (%s): %d draw lists; %s rectangle %d,%d .. %d,%d at %.3f (wanted %s %d,%d .. %d,%d at "
                    "%.3f); alpha where it must show %d (>= 200), in the margin %d (0), beyond it %d (0) -> %s\n",
                    frame, phase.name, drawData->CmdListsCount, state.menu ? "the MENU's" : "everything's",
                    state.crop[0], state.crop[1], state.crop[2], state.crop[3], scale,
                    phase.menu ? "the menu's" : "everything's", phase.want[0], phase.want[1], phase.want[2],
                    phase.want[3], phase.scale, inside, margin, beyond, ok ? "OK" : "FAIL");
        // The first frame of all is laid out before the windows' contents are known; scored from the second.
        if (frame >= 2)
        {
            ++checked;
            if (!ok)
                ++failures;
        }
        if (frame >= 2 && drawData->CmdListsCount < (phase.menu ? 4 : 2))
        {
            std::printf("frame %d: FAIL %d draw lists: the menu, its child, the performance overlay or the pointer "
                        "was not drawn, so the test has no hard case\n",
                        frame, drawData->CmdListsCount);
            ++failures;
        }
    }
    // ---- The shared textures, both kinds (OverlayTextures.h), as the game DLL makes them: the menu drawn into
    // each on this device, then read from a SECOND device through its handle, as the layer does. The old kind is what
    // a game's device that makes no NT handles falls back on. ----
    {
        ID3D11Device* other = nullptr;
        ID3D11DeviceContext* otherContext = nullptr;
        D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, nullptr, 0, D3D11_SDK_VERSION, &other, nullptr,
                          &otherContext);
        ID3D11Device1* other1 = nullptr;
        if (other != nullptr)
            other->QueryInterface(IID_PPV_ARGS(&other1));
        D3D11_QUERY_DESC eventDesc { D3D11_QUERY_EVENT, 0 };
        ID3D11Query* done = nullptr;
        device->CreateQuery(&eventDesc, &done);
        for (int kind = 0; kind < 2; ++kind)
        {
            const bool oldKind = kind == 1;
            const char* name = oldKind ? "old-kind handle" : "NT handle";
            ID3D11Texture2D* shared = nullptr;
            ID3D11RenderTargetView* sharedTarget = nullptr;
            HANDLE handle = nullptr;
            bool ok = other != nullptr && other1 != nullptr && done != nullptr &&
                      OverlayTextures::MakeShared(device, 1024, 1024, oldKind, &shared, &sharedTarget, &handle);
            int inside = -1, margin = -1;
            if (ok)
            {
                io.DeltaTime = 1.0f / 90.0f;
                ImGui_ImplDX11_NewFrame();
                ImGui::NewFrame();
                ImGui::SetNextWindowPos(ImVec2(3000.0f, 2500.0f));
                ImGui::SetNextWindowSize(ImVec2(900.0f, 600.0f));
                ImGui::Begin("OptiScaler v-test - game.exe", nullptr, ImGuiWindowFlags_NoSavedSettings);
                ImGui::Text("The menu");
                ImGui::End();
                ImGui::Render();
                OverlayCrop::State fresh;
                float scale = 1.0f;
                ok = OverlayCrop::Grow(ImGui::GetDrawData(), fresh, rules, kMargin, kMaxSide, &scale);
                const float transparent[4] = {};
                context->OMSetRenderTargets(1, &sharedTarget, nullptr);
                context->ClearRenderTargetView(sharedTarget, transparent);
                if (ok)
                {
                    ImDrawData through = OverlayCrop::Through(ImGui::GetDrawData(), fresh.crop, scale);
                    ImGui_ImplDX11_RenderDrawData(&through);
                }
                // No fence here: the draw is known to be done before the other device reads (the game DLL names an
                // image one Present late for the same reason).
                context->End(done);
                context->Flush();
                BOOL finished = FALSE;
                for (int tries = 0; tries < 2000 && !finished; ++tries)
                {
                    if (context->GetData(done, &finished, sizeof(finished), 0) != S_OK)
                        finished = FALSE;
                    if (!finished)
                        Sleep(1);
                }
                ID3D11Texture2D* opened = nullptr;
                const HRESULT open = oldKind ? other->OpenSharedResource(handle, IID_PPV_ARGS(&opened))
                                             : other1->OpenSharedResource1(handle, IID_PPV_ARGS(&opened));
                ok = ok && finished && SUCCEEDED(open) && opened != nullptr;
                if (ok)
                {
                    D3D11_TEXTURE2D_DESC got {};
                    opened->GetDesc(&got);
                    D3D11_TEXTURE2D_DESC readDesc = got;
                    readDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
                    readDesc.Usage = D3D11_USAGE_STAGING;
                    readDesc.BindFlags = 0;
                    readDesc.MiscFlags = 0;
                    readDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
                    ID3D11Texture2D* readBack = nullptr;
                    D3D11_MAPPED_SUBRESOURCE mapped {};
                    if (SUCCEEDED(other->CreateTexture2D(&readDesc, nullptr, &readBack)))
                    {
                        otherContext->CopyResource(readBack, opened);
                        if (SUCCEEDED(otherContext->Map(readBack, 0, D3D11_MAP_READ, 0, &mapped)))
                        {
                            const auto* bytes = static_cast<const uint8_t*>(mapped.pData);
                            inside = bytes[(size_t) (kMargin + 300) * mapped.RowPitch + (size_t) (kMargin + 450) * 4 + 3];
                            margin = bytes[(size_t) 4 * mapped.RowPitch + (size_t) 4 * 4 + 3];
                            otherContext->Unmap(readBack, 0);
                        }
                        readBack->Release();
                    }
                    ok = inside >= 200 && margin == 0;
                }
                if (opened != nullptr)
                    opened->Release();
            }
            std::printf("shared texture, %s: made and opened on a second device %s; alpha there in the menu's middle "
                        "%d (>= 200), in the margin %d (0) -> %s\n",
                        name, ok || inside >= 0 ? "yes" : "NO", inside, margin, ok ? "OK" : "FAIL");
            ++checked;
            if (!ok)
                ++failures;
            if (sharedTarget != nullptr)
                sharedTarget->Release();
            if (shared != nullptr)
                shared->Release();
            if (handle != nullptr && !oldKind)
                CloseHandle(handle);
        }
    }
    ImGui_ImplDX11_Shutdown();
    ImGui::DestroyContext();
    std::printf("%s: %d frames checked, %d failures\n", failures == 0 && checked > 0 ? "PASSED" : "FAILED", checked,
                failures);
    return failures == 0 && checked > 0 ? 0 : 1;
}
