#pragma once

// The overlay's own rectangle in the game's window, for the headset's panel (inputs/OverlayShare). Header only and
// free of everything but ImGui's data types, so that tools/xr_observer/test/overlay_test runs THIS code on a real
// ImGui frame: the first version called ImGui::GetForegroundDrawList() after the frame had ended, which reads the
// current window (null by then), and crashed the game a second after its start, twice.
// NOTHING of ImGui is called here: the draw data is only read.
// Include imgui.h before this file (the game DLL and the test reach it by different paths).
//
// WHICH rectangle: the first rule was "everything the overlay drew, growing
// while it is up". In FC5 in VR the overlay's canvas is 8192 x 8192, a small indicator sits in one corner of it and
// the menu in its middle, so the rectangle began at the indicator and ended at the 4096 a panel holds: the menu
// disappeared from the panel as soon as it was moved past that. Now: while the MENU is up (a draw list whose
// owner's name begins with `primary`) the rectangle is the menu's, with what belongs to it (its child windows,
// popups, combos, tooltips, dialogs), and FOLLOWS it wherever it is on the canvas; what only informs (the splash,
// the performance overlay, the scan indicator, messages) does not count. With the menu closed everything counts,
// so that messages reach the headset. A rectangle larger than a panel holds is shown smaller, not cut.
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstring>

namespace OverlayCrop
{
struct Rules
{
    const char* primary = nullptr; // the menu window's name begins with this; nullptr = there is no menu
    bool countPointer = false;     // the test's control: the mouse pointer's list counts
    bool countPassive = false;     // the test's control: what only informs counts beside the menu
};

// What is kept between two frames (the Present thread's own).
struct State
{
    bool valid = false;
    bool menu = false; // the rectangle is the menu's
    int crop[4] = {};  // pixels of the window from its top left corner: x0, y0, x1, y1
};

inline bool Begins(const char* name, const char* with)
{
    return name != nullptr && with != nullptr && with[0] != 0 && std::strncmp(name, with, std::strlen(with)) == 0;
}
inline bool IsPointer(const ImDrawList* list) { return Begins(list->_OwnerName, "##Foreground"); }
// What only informs: drawn whether the menu is open or not, anywhere on the canvas.
inline bool IsPassive(const ImDrawList* list)
{
    const char* name = list->_OwnerName;
    return Begins(name, "Splash") || Begins(name, "Performance Overlay") || Begins(name, "DlssNrExposureScan") ||
           Begins(name, "##TOAST") || Begins(name, "##Background") || Begins(name, "Debug##Default");
}
inline bool Draws(const ImDrawList* list)
{
    for (const ImDrawCmd& command : list->CmdBuffer)
        if (command.UserCallback == nullptr && command.ElemCount != 0)
            return true;
    return false;
}
inline bool MenuIsUp(const ImDrawData* data, const Rules& rules)
{
    for (int n = 0; n < data->CmdListsCount; ++n)
        if (data->CmdLists[n] != nullptr && Begins(data->CmdLists[n]->_OwnerName, rules.primary) &&
            Draws(data->CmdLists[n]))
            return true;
    return false;
}

// What counts of what the overlay drew, in ImGui's coordinates: every command's triangles, inside its clip
// rectangle. False: nothing.
inline bool ContentBox(const ImDrawData* data, float box[4], const Rules& rules, bool menu)
{
    float x0 = FLT_MAX, y0 = FLT_MAX, x1 = -FLT_MAX, y1 = -FLT_MAX;
    for (int n = 0; n < data->CmdListsCount; ++n)
    {
        const ImDrawList* list = data->CmdLists[n];
        if (list == nullptr)
            continue;
        // The pointer would stretch the rectangle to wherever it is (unless it is all there is).
        if (IsPointer(list) && !rules.countPointer && data->CmdListsCount > 1)
            continue;
        if (menu && !rules.countPassive && IsPassive(list))
            continue;
        for (const ImDrawCmd& command : list->CmdBuffer)
        {
            if (command.UserCallback != nullptr || command.ElemCount == 0)
                continue;
            float cx0 = FLT_MAX, cy0 = FLT_MAX, cx1 = -FLT_MAX, cy1 = -FLT_MAX;
            for (unsigned int i = 0; i < command.ElemCount; ++i)
            {
                const unsigned int at = command.IdxOffset + i;
                if (at >= (unsigned int) list->IdxBuffer.Size)
                    break;
                const unsigned int index = command.VtxOffset + list->IdxBuffer.Data[at];
                if (index >= (unsigned int) list->VtxBuffer.Size)
                    continue;
                const ImVec2& p = list->VtxBuffer.Data[index].pos;
                cx0 = (std::min)(cx0, p.x);
                cy0 = (std::min)(cy0, p.y);
                cx1 = (std::max)(cx1, p.x);
                cy1 = (std::max)(cy1, p.y);
            }
            cx0 = (std::max)(cx0, command.ClipRect.x);
            cy0 = (std::max)(cy0, command.ClipRect.y);
            cx1 = (std::min)(cx1, command.ClipRect.z);
            cy1 = (std::min)(cy1, command.ClipRect.w);
            if (cx1 <= cx0 || cy1 <= cy0)
                continue;
            x0 = (std::min)(x0, cx0);
            y0 = (std::min)(y0, cy0);
            x1 = (std::max)(x1, cx1);
            y1 = (std::max)(y1, cy1);
        }
    }
    if (x1 <= x0 || y1 <= y0)
        return false;
    box[0] = x0;
    box[1] = y0;
    box[2] = x1;
    box[3] = y1;
    return true;
}

// The rectangle to show (state.crop) and the scale it is drawn at (1, or less when it is larger than maxSide pixels
// a side). The menu's rectangle follows the menu and keeps the largest size it has had while it is up (a tooltip or
// an opened list must not make the panel jump back and forth); without the menu the rectangle only grows. Either
// starts anew when the menu opens or closes, and when `state.valid` is cleared (the overlay was not up).
// False: nothing to show.
inline bool Grow(const ImDrawData* data, State& state, const Rules& rules, int margin, int maxSide, float* scale)
{
    const bool menu = MenuIsUp(data, rules);
    float box[4] = {};
    if (!ContentBox(data, box, rules, menu))
        return false;
    if (state.valid && state.menu != menu)
        state.valid = false;
    const float left = data->DisplayPos.x, top = data->DisplayPos.y;
    const int width = (int) data->DisplaySize.x, height = (int) data->DisplaySize.y;
    int now[4] = { (int) std::floor(box[0] - left) - margin, (int) std::floor(box[1] - top) - margin,
                   (int) std::ceil(box[2] - left) + margin, (int) std::ceil(box[3] - top) + margin };
    if (state.valid && menu)
    {
        // Where the menu is now, at the largest size so far -- and the size is KEPT when the menu is dragged over the
        // canvas's edge: the rectangle is moved back in, not cut. A panel whose shape changes while the menu is dragged
        // reads as a panel moving nearer and farther (dragged up, it moved forward and
        // backwards in space).
        const int keptW = (std::max)(now[2] - now[0], state.crop[2] - state.crop[0]);
        const int keptH = (std::max)(now[3] - now[1], state.crop[3] - state.crop[1]);
        now[2] = now[0] + keptW;
        now[3] = now[1] + keptH;
        if (now[0] < 0)
        {
            now[2] -= now[0];
            now[0] = 0;
        }
        if (now[1] < 0)
        {
            now[3] -= now[1];
            now[1] = 0;
        }
        if (now[2] > width)
        {
            now[0] -= now[2] - width;
            now[2] = width;
        }
        if (now[3] > height)
        {
            now[1] -= now[3] - height;
            now[3] = height;
        }
    }
    else if (state.valid)
    {
        now[0] = (std::min)(now[0], state.crop[0]);
        now[1] = (std::min)(now[1], state.crop[1]);
        now[2] = (std::max)(now[2], state.crop[2]);
        now[3] = (std::max)(now[3], state.crop[3]);
    }
    now[0] = std::clamp(now[0], 0, width);
    now[1] = std::clamp(now[1], 0, height);
    now[2] = std::clamp(now[2], 0, width);
    now[3] = std::clamp(now[3], 0, height);
    if (now[2] - now[0] < 8 || now[3] - now[1] < 8)
        return false;
    std::memcpy(state.crop, now, sizeof(now));
    state.valid = true;
    state.menu = menu;
    if (scale != nullptr)
        *scale = (std::min)(1.0f, (std::min)((float) maxSide / (float) (now[2] - now[0]),
                                             (float) maxSide / (float) (now[3] - now[1])));
    return true;
}

// The same draw lists seen through the rectangle, at `scale`: a renderer maps DisplayPos .. DisplayPos + DisplaySize
// onto a viewport of DisplaySize x FramebufferScale at its target's top left corner, and clips to it.
inline ImDrawData Through(const ImDrawData* data, const int crop[4], float scale)
{
    ImDrawData through = *data;
    through.DisplayPos = ImVec2(data->DisplayPos.x + (float) crop[0], data->DisplayPos.y + (float) crop[1]);
    through.DisplaySize = ImVec2((float) (crop[2] - crop[0]), (float) (crop[3] - crop[1]));
    through.FramebufferScale = ImVec2(data->FramebufferScale.x * scale, data->FramebufferScale.y * scale);
    return through;
}
} // namespace OverlayCrop
