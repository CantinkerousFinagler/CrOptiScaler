#pragma once

// Diagnostics: what a game (or a mod such as SekiroTSR) hands DLSS through OptiScaler's own D3D11 feature path.
// Logs the inputs once on the first evaluate (texture sizes and formats, motion vector scale, jitter, subrects), then
// per 600 evaluations the jitter pattern (distinct positions, range), resets, handles and contexts. Once, at the
// 600th evaluate, a census of the exe's code as it is in memory (some exes are encrypted on disk, so no file compare):
// every jump that leaves the exe for another module, directly or through a MinHook-style relay -- i.e. which
// functions a mod hooked, as exe offsets. Changes nothing.
namespace UpscalerInputProbe_Dx11
{
// [Inputs] UpscalerInputProbe (startup only, default off).
bool Enabled();

// Called at the top of NVSDK_NGX_D3D11_EvaluateFeature for OptiScaler's own features.
void OnEvaluate(ID3D11DeviceContext* ctx, unsigned int handle, NVSDK_NGX_Parameter* params);
} // namespace UpscalerInputProbe_Dx11
