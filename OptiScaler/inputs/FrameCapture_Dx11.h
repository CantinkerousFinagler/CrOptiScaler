#pragma once

#include <vector>

// Diagnostics ([Inputs] DuniaTaaDiagnostics, F7): saves the textures one D3D11 pass reads and writes as .dds files,
// for finding motion vectors, pre-TAA colour and the like in games with no capture tool running (VR included).
// Armed, it spends two Presents learning the pass: the last compute dispatch on an immediate context whose UAV 0 is
// an eye-sized texture (the output can differ per eye). On the next three Presents it copies that dispatch's
// constant buffers, and on the first also its SRV and UAV textures just before it runs and its UAVs just after;
// frames 0 and 2 are the same eye. The copies are read back at the last Present and written on a background thread
// to OptiCapture\<time>\ beside the game exe: .dds textures, index.txt and constants.txt. Expect a short hitch.
// Games whose TAA is a draw (FC5; preferred there, or wherever no eye-sized dispatch is seen): the pass is instead the
// pixel-shader draw that last wrote the texture the first draw into the eye target (render target only) reads. Its ps
// and vs constants on all three frames; on frame 0 its SRVs, render targets before and after, the scene depth (the
// last depth bound with a target of the TAA size before it) and its pixel shader.
// Draw mode takes four Presents, and from the second on also the constants of the draws that wrote the TAA's t0 on
// the first (the motion vector pass: draws or, in FC5, a compute dispatch; "mv" files: their SRVs and shaders).
namespace FrameCapture_Dx11
{
void Arm();
bool Busy();

// Immediate context, around the game's Dispatch.
void BeforeDispatch(ID3D11DeviceContext* ctx, UINT x, UINT y, UINT z);
void AfterDispatch(ID3D11DeviceContext* ctx);

// Immediate context, around the game's draws: the last draw into an eye-sized render target before Present (an
// upscale after TAA) is captured too, on frame 0 after the pass ("draw" files, its pixel shader and ps constants).
void BeforeDraw(ID3D11DeviceContext* ctx);
void AfterDraw(ID3D11DeviceContext* ctx);

// Before the real Present.
void OnPresent();

// From the device's CreateComputeShader / CreatePixelShader hooks: keeps each shader's bytecode so a capture can
// save the one bound to the pass (disassemble it with D3DDisassemble).
void RememberShader(const void* shader, const void* bytecode, SIZE_T length);
// The bytecode kept for a shader; false when none was kept (the shader was made before the hooks).
bool ShaderCode(const void* shader, std::vector<char>& code);
} // namespace FrameCapture_Dx11
