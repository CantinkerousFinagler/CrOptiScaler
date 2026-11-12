#pragma once

// DLAA + Neural Rendering on the finished 8-bit eye image of a D3D11 game that records its frame on deferred
// contexts; one list draws the final picture (eye-sized R8G8B8A8, render-target only, no depth) with a single
// full-screen draw. That list is marked while it is recorded, and after the immediate context has run it, DLSS
// runs on the picture and writes the result back, before anything later in the frame copies it out. Dark Souls III
// records the same way but draws the final picture on the immediate context; that draw is processed at the next
// bind on that context, or at Present.
// No motion vectors: zero vectors and a history reset every frame, so one feature serves both eyes.
namespace EyeImageNr_Dx11
{
// [Inputs] EyeImageNr on (startup only). Not tied to an exe: Dark Souls III so far.
bool Enabled();

// Recording on a deferred context.
void OnDeferredBind(ID3D11DeviceContext* ctx, UINT numRtvs, ID3D11RenderTargetView* const* rtvs,
                    ID3D11DepthStencilView* dsv);
void OnDeferredDraw(ID3D11DeviceContext* ctx);
void OnFinishCommandList(ID3D11DeviceContext* ctx, ID3D11CommandList* list);

// Immediate context: binds and draws (a final picture drawn there), and after the game's ExecuteCommandList.
void OnImmediateBind(ID3D11DeviceContext* ctx, UINT numRtvs, ID3D11RenderTargetView* const* rtvs,
                     ID3D11DepthStencilView* dsv);
void OnImmediateDraw(ID3D11DeviceContext* ctx);
void AfterExecuteCommandList(ID3D11DeviceContext* ctx, ID3D11CommandList* list);

void OnPresent();
} // namespace EyeImageNr_Dx11
