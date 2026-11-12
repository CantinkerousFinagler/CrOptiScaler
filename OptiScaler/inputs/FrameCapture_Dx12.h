#pragma once

#include <d3d12.h>

// Diagnostics ([Inputs] DuniaTaaDiagnostics, F7): the D3D12 counterpart of FrameCapture_Dx11, cut down to the one
// question FC6 still has open -- are the R16G16 textures the engine hands around real motion vectors?
//
// Far Cry 6 creates several eye-sized R16G16 textures (FLOAT and UNORM, some render-target + UAV, some UAV only).
// Wiring one of them into DLSS smeared or turned to noise, but the pick was by a score, not by evidence, so the
// question is which of them -- if any -- holds camera motion. This saves the candidates and the finished eye image
// for a few consecutive Presents so the warp test can run offline (tools\capture), rather than by relaunching the
// game for every guess.
//
// Armed, it waits [Inputs] CaptureDelaySeconds (the overlay has to be closed first in VR), then over the next six
// eye frames copies:
//   * that finished eye image ("eye"), and, at the first render-target bind of each frame,
//   * every registered motion candidate ("mvbind"), which there still holds what the previous frame wrote and is
//     in a state the game itself put it in rather than one guessed from its flags.
// The copies are read back after the last of those Presents and written on a background thread to
// OptiCapture\<time>\ beside the game exe: .dds files (DX10 header, mip 0) plus index.txt, the same layout the
// DX11 capture writes, so the existing scripts read them unchanged. Expect a hitch of a second or two.
//
// Under one-eye-per-Present VR frames n and n+2 are the same eye, n and n+1 the two eyes. A bind copy holds the
// previous frame's content, so mvbind of frame n goes with eye frames n-1 and n-3; index.txt says so too.
namespace FrameCapture_Dx12
{
void Arm();
bool Busy();

// Every motion-vector candidate the RTV/UAV hooks offer, whether or not it wins the score. Cheap and idempotent:
// each resource is registered once, with a reference held until the capture is done with it. With a command list
// and a render-target bind, this is also where a running capture takes that candidate's copy for the frame.
void NoteCandidate(ID3D12Resource* resource, bool boundAsRenderTarget, ID3D12GraphicsCommandList* cmd = nullptr);

// Eye-shaped and in a format that could hold a motion pair somewhere in it. Wider than DuniaTaa's MotionScore on
// purpose: that one only takes two-channel formats, and FC6 is described as keeping motion in the G-buffer.
bool IsCaptureShape(const D3D12_RESOURCE_DESC& desc);

// How likely this format is to be worth its ~62 MB a frame, highest first; 0 means log it but never copy it.
// Dedicated two-channel targets rank above packed 32-bit G-buffer targets; 8-byte-per-pixel ones are logged only.
int CapturePriority(const D3D12_RESOURCE_DESC& desc);

// Reaching a texture a compute pass writes. Such a texture is never passed to OMSetRenderTargets, so the
// render-target path above can never copy it, and Dunia writes a great deal through compute -- including the
// half-res R8G8 that is the last motion vector candidate. Three calls cooperate, and the order matters:
//
//   NoteUavBound   a candidate appeared as a UAV descriptor in a table bound on this list (BindingCapture_Dx12).
//                  Nothing is copied and no barrier is emitted: a descriptor table bind says nothing about the
//                  state the resource is in. Assuming otherwise at this point hung the game.
//   NoteBarriers   every ResourceBarrier the game records on this list, which is where the true state comes from.
//   AfterDispatch  after the game's Dispatch: the UAVs bound since the previous one are copied, but only those
//                  this same list transitioned into UNORDERED_ACCESS and has not transitioned out again. Within
//                  one command list, recorded order is execution order, so that state is a fact and not a guess.
//                  A candidate whose state this list never set is skipped, not assumed.
void NoteUavBound(ID3D12GraphicsCommandList* cmd, ID3D12Resource* resource);
void NoteBarriers(ID3D12GraphicsCommandList* cmd, UINT count, const D3D12_RESOURCE_BARRIER* barriers);
void AfterDispatch(ID3D12GraphicsCommandList* cmd, ID3D12PipelineState* pipeline);
// Every compute pipeline the game creates, with its DXBC. Kept from load while diagnostics are on, because a
// pipeline is usually created long before the capture that wants to read it. The shader that writes a candidate
// is kept with its reflection data, which says what that texture is for. Dumped as cs_<pipeline>.dxbc beside the .dds
// files.
void NoteComputePipeline(ID3D12PipelineState* pipeline, const void* bytecode, size_t bytes);
// Command lists are recycled; everything tracked for this one is dropped.
void NoteListReset(ID3D12GraphicsCommandList* cmd);
// Whether the Dispatch / ResourceBarrier / Reset hooks are worth installing at all, asked at hook time.
bool WantsComputeHooks();

// At the splice point, on the game's list: `eye` is finished and in `eyeState`. Candidates are not copied here --
// by the end of the frame D3D12's transient allocators have handed their memory to other passes.
void CaptureAt(ID3D12GraphicsCommandList* cmd, ID3D12Resource* eye, D3D12_RESOURCE_STATES eyeState);

// After the real Present, with the queue the frame was submitted on.
void OnPresent(ID3D12CommandQueue* queue);
} // namespace FrameCapture_Dx12
