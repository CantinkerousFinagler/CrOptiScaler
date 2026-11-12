#pragma once

// FC5 D3D11 hooks. The render target bind hooks feed Fc5Dlss_Dx11; with [Inputs] DuniaTaaDiagnostics=true
// the rest log what the game binds, reads, draws and copies for two Presents (one per eye).
// Installed at device creation: FarCry5.exe when its DLSS backend is on, any D3D11 game when EyeImageNr is on, and
// any D3D11 game when diagnostics are on.

namespace Fc5Trace_Dx11
{
void Install(ID3D11Device* device);
bool Installed();
void OnPresent();
} // namespace Fc5Trace_Dx11
