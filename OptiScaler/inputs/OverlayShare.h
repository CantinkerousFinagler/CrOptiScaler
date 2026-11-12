#pragma once

#include "StaleEyeShareAbi.h"

#include <d3d11.h>
#include <d3d12.h>

struct ImDrawData;

// The overlay in the headset, the game DLL's side (see StaleEyeShareAbi.h, RvrToolsOverlayV1): the overlay is
// drawn a second time into a shared texture while the stale-eye OpenXR layer asks for it, and the layer shows it as
// a flat panel. D3D11 games (AfterDraw) and D3D12 games (AfterDrawDx12 +
// AfterExecuteDx12): the layer's side is D3D11 either way (the layer sees a D3D11 session in both) and opens
// D3D12's NT handles as it opens D3D11's. [Inputs] HeadsetOverlay (default on; nothing happens without the layer).
namespace OverlayShare
{
// Called at every Present from the D3D11 overlay's render, on the Present thread, after the draw onto the flat
// window. drawData: what ImGui::Render produced at this Present, or nullptr when the overlay drew nothing.
void AfterDraw(ID3D11Device* device, ID3D11DeviceContext* context, ImDrawData* drawData);
// D3D12: called with the menu's command list OPEN, right after ImGui's draw onto the back buffer was recorded into
// it: the same draw lists are recorded once more, into one of our shared textures (COMMON -> RENDER_TARGET -> COMMON),
// and the back buffer's target is bound again after (`backBuffer`). rtvFormat: the format ImGui's D3D12 pipeline was
// made for (the window's swapchain format; RGBA8 UNORM / sRGB only, else the panel is skipped with a logged reason).
// nullptr list / drawData = nothing drawn this Present (the panel hides). Then, after the list was executed:
void AfterDrawDx12(ID3D12Device* device, ID3D12GraphicsCommandList* list, DXGI_FORMAT rtvFormat,
                   D3D12_CPU_DESCRIPTOR_HANDLE backBuffer, ImDrawData* drawData);
// Signals the shared fence with the image's sequence on the queue the list went to, and publishes the image.
void AfterExecuteDx12(ID3D12CommandQueue* queue);
// For the menu. Whether the OpenXR layer has asked for the overlay within the last second (it is running, and of
// this build), and what it was last given (false before the first image).
bool LayerAsks();
bool Latest(::RvrToolsOverlayV1* out);
} // namespace OverlayShare
