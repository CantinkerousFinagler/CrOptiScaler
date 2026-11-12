#pragma once

// DLAA + Neural Rendering at Present on the game's finished 8/10-bit eye image, for D3D11 games that draw their
// frame on the immediate context (Days Gone). The gameplay context is the one that binds two or more eye-sized
// targets with eye-sized depth (the G-buffer, which also gives the depth DLSS reads). On it, the last eye-sized
// 8/10-bit colour texture written by a draw or a compute dispatch before Present is the final image: DLSS runs on
// a copy and the result is written back before Present goes on to the VR layer. No motion vectors: zero vectors
// and a history reset every frame, so one feature serves both eyes.
namespace FinalImageNr_Dx11
{
// [Inputs] FinalImageNr on (startup only).
bool Enabled();

// Immediate context calls, from the D3D11 hooks.
void OnBind(ID3D11DeviceContext* ctx, UINT numRtvs, ID3D11RenderTargetView* const* rtvs,
            ID3D11DepthStencilView* dsv);
void OnDraw(ID3D11DeviceContext* ctx);
void OnCsUavs(ID3D11DeviceContext* ctx, UINT start, UINT count, ID3D11UnorderedAccessView* const* uavs);
void OnDispatch(ID3D11DeviceContext* ctx);

// Before the real Present.
void OnPresent();

// The largest gameplay depth bound so far this frame (the scene depth), or nullptr. Not AddRef'd.
ID3D11Texture2D* FrameDepth();
} // namespace FinalImageNr_Dx11
