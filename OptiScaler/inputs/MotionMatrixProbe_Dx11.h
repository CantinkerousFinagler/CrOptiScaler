#pragma once

#include <cstdint>

struct ID3D11DeviceContext;
struct ID3D11Resource;
struct ID3D11RenderTargetView;

// Diagnostics, for matrix (same-eye) motion in a game that hands DLSS its motion vectors through OptiScaler's own
// D3D11 path (a mod calling DLSS): which constants the pass writing those vectors reads, so the camera matrices can be
// found offline.
//
// F7 arms a capture of the next kPresents Presents. During it, every draw into the motion vector texture DLSS gets
// (learned by identity at evaluate) records the bound VS and PS constant buffers' contents -- as the game last wrote
// them with Map/Unmap or UpdateSubresource, on that context or any other -- and, at each evaluate, a 32x32 grid of
// (depth, motion x, motion y) plus the jitter and the motion scale. Everything goes to one text file beside the exe
// (MatrixProbe\probe-<time>.txt); the log gets a summary with what was and was NOT found.
//
// Idle cost: one flag check per bind, draw and Map. Nothing is referenced beyond the call it was seen in.
namespace MotionMatrixProbe_Dx11
{
// [Inputs] UpscalerMatrixProbe (startup only, default off).
bool Enabled();

// Keys (menu_common) and Present.
void Arm();
void OnPresent();
bool Capturing();
uint64_t Present();

// From DLSS's evaluate: the motion vector texture, compared by address only.
void SetMotionTexture(ID3D11Resource* motion);

// From the D3D11 hooks (Fc5Trace_Dx11), on any context and thread.
void OnBind(ID3D11DeviceContext* ctx, unsigned int numRtvs, ID3D11RenderTargetView* const* rtvs);
void OnDraw(ID3D11DeviceContext* ctx);
void OnMap(ID3D11DeviceContext* ctx, ID3D11Resource* resource, unsigned int subresource, int mapType, void* data);
void BeforeUnmap(ID3D11DeviceContext* ctx, ID3D11Resource* resource, unsigned int subresource);
void OnUpdateSubresource(ID3D11DeviceContext* ctx, ID3D11Resource* resource, unsigned int subresource,
                         const void* box, const void* data);

// From PerEyeDlss_Dx11 when a frame strip has been read back: 256x256 RGBA8 of DLSS's output (after any F8 view).
void WriteStrip(uint64_t present, int parity, int viewMode, bool perEye, const uint32_t* pixels);

// From PerEyeDlss_Dx11 when a fit has been read back: rows x, y, w (z scaled by 100), stats, reason, the eye's streak.
void WriteFit(uint64_t present, int slot, const float* fit, float streak);

// From PerEyeDlss_Dx11 when a sample grid has been read back: 1024 x 2 float4, (depth, game mv.x px, game mv.y px, -)
// and (DLSS's mv.x px, DLSS's mv.y px, -, -).
void WriteSamples(uint64_t present, int parity, const float* samples, float mvScaleX, float mvScaleY, float jitterX,
                  float jitterY, unsigned int renderW, unsigned int renderH);
} // namespace MotionMatrixProbe_Dx11
