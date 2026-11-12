#pragma once

// Diagnostics: what a DX12 game (Crimson Desert, through Streamline) hands DLSS through OptiScaler's NGX exports.
// Logs every create (feature, path, sizes, flags incl. DepthInverted), the inputs once on the first evaluate per
// feature (texture sizes and formats, MV scale, jitter, subrects, ray-reconstruction guides), then per 600 gameplay
// Presents: evaluates per Present (0 / 1 / 2+), per feature, jitter pattern, mean jitter x by Present parity (an
// even cycle split by eye under AER), resets, MV scale and render size changes. Changes nothing.
// Shares [Inputs] UpscalerInputProbe with the D3D11 probe.
namespace UpscalerInputProbe_Dx12
{
bool Enabled();

// After a successful NVSDK_NGX_D3D12_CreateFeature. `optiPath` = OptiScaler's own feature (else native passthrough).
void OnCreate(int featureId, unsigned int handle, bool optiPath, NVSDK_NGX_Parameter* params);

// At the top of NVSDK_NGX_D3D12_EvaluateFeature. featureId = -1 when the handle was never recorded.
void OnEvaluate(ID3D12GraphicsCommandList* cmdList, unsigned int handle, int featureId, NVSDK_NGX_Parameter* params);

// Every Present, from the swapchain wrapper.
void OnPresent();
} // namespace UpscalerInputProbe_Dx12
