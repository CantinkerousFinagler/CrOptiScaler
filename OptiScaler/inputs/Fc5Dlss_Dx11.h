#pragma once

// FC5 DLSS on the game's upscale draw, modelled on the FC6 backend (DuniaTaa_Dx12). The game tonemaps, runs
// its TAA on the 8-bit picture, then draws that into the eye-sized target bound without depth. DLSS runs at
// that bind with the TAA output as input, and its result is copied over the game's draw at the next bind,
// before the eye-resolution passes. At 100% Resolution Scale that draw is 1:1 and DLSS runs as DLAA, which
// carries Neural Rendering through the D3D11 bridge.
namespace Fc5Dlss_Dx11
{
enum class Status
{
    WaitingForGameplay = 0,
    Dlaa = 1,
    Upscaling = 2,
    CreateFailed = 3,
    ScaleChanged = 4,
    RenderSizeNr = 5,
};

struct Info
{
    Status status = Status::WaitingForGameplay;
    unsigned inW = 0;
    unsigned inH = 0;
    unsigned outW = 0;
    unsigned outH = 0;
};

// FarCry5.exe with [Inputs] DuniaTaaHost on and [Inputs] Fc5TaaDlss off.
bool Enabled();
// Called from the OMSetRenderTargets hooks, around the game's call.
void BeforeBind(ID3D11DeviceContext* ctx, UINT numRtvs, ID3D11RenderTargetView* const* rtvs,
                ID3D11DepthStencilView* dsv);
void AfterBind(ID3D11DeviceContext* ctx);
void OnPresent();
Info GetInfo();
} // namespace Fc5Dlss_Dx11
