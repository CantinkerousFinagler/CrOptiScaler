#pragma once

#include <d3d12.h>

#include <string>

struct NVSDK_NGX_Parameter;

// Diagnostics ([Inputs] UpscalerInputCapture, F7 or the overlay's button): what a DX12 game hands the upscaler, and what
// comes out, for 8 Presents in a row. Written to answer one question without a video codec in the way: how much noise
// does one-eye-per-Present rendering add, and is it already in the picture before the upscaler?
//
// Saved to OptiCapture\<time>\ beside the game exe as .dds (DX10 header, mip 0) plus index.txt: colour (the render
// subrect) and output on all 8, motion and depth on the first 2. Each frame carries its Present number, so a pair is
// matched by that stamp and a gap in it shows.
//
// Nothing stalls the render thread: the readback buffers are made on a worker when armed, the frames are only copies
// recorded on the game's list, and the fence wait, Map and writing are on a worker. No game resource is ever held.
// Off, each hook is one bool test. Assumes the states NGX asks for at an evaluate: inputs NON_PIXEL_SHADER_RESOURCE,
// output UNORDERED_ACCESS.
namespace EvalCapture_Dx12
{
bool Enabled();
void Arm();
bool Busy();
std::string Status();

// At the top of NVSDK_NGX_D3D12_EvaluateFeature. Returns true when it took this evaluate: then AfterEvaluate must be
// called on the same list once the upscaler has recorded.
bool OnEvaluate(ID3D12GraphicsCommandList* cmdList, int featureId, NVSDK_NGX_Parameter* params);
void AfterEvaluate(ID3D12GraphicsCommandList* cmdList);

// Every Present, from the swapchain wrapper.
void OnPresent(ID3D12CommandQueue* queue);
} // namespace EvalCapture_Dx12
