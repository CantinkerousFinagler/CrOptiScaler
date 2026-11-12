#pragma once

#include <d3d12.h>

#include <string>

// Diagnostics ([Inputs] DuniaTaaDiagnostics, F7): what FC6 actually binds to each draw, resolved through the
// descriptor heaps. This is the in-process answer to the question RenderDoc would have answered, after RenderDoc
// turned out not to attach to FC6 at all.
//
// Why this exists. Two dead ends led here:
//   * the motion buffer was hunted by *shape and format*, which ruled out R16G16_FLOAT, R16G16_UNORM and R8G8 at
//     full resolution and never found it. Guessing by descriptor is the wrong instrument -- what settles it is
//     seeing which texture a pass actually reads.
//   * ConstantCapture_Dx12 came back empty three captures running, and the INFO diagnostics proved why: the root
//     CBV hooks are installed and *no root CBV bind ever arrives*. FC6 binds its constants through descriptor
//     tables, so the constants are reachable only from here.
//
// Both are the same mechanism, so one module gets both. While a capture runs, every
// SetGraphicsRootDescriptorTable / SetComputeRootDescriptorTable is resolved: the base GPU handle finds its heap
// through ResTrack_Dx12, and each descriptor in the table is walked back to the resource behind it. That gives a
// *group* -- the set of textures and constant buffers one pass has bound together -- which is the thing worth
// reading. A TAA resolve binds colour, history, depth and motion in one group, and a group of eye-sized textures
// with a two-channel or packed one among them is the motion buffer, identified by what the engine does with it
// rather than by what its descriptor looks like.
//
// Constant buffers found in a table are dumped inline, in the format tools\capture\parsecb.py already reads, so
// cbmat2.py can go looking for CurrentToPrevMatrix in them without any further work.
//
// Written as bindings.txt beside the capture's .dds files.
//
// Limits: a descriptor table carries no length (only the root signature knows), so each is walked to
// kMaxDescriptorsPerTable or until the descriptors stop resolving -- expect some trailing entries that belong to
// the next table. Groups are deduplicated by their resource set, so a pass that binds the same things every draw
// appears once, and the order of groups is bind order, not draw order.
namespace BindingCapture_Dx12
{
// True when the descriptor table hooks are worth installing at all, asked at hook time by D3D12_Hooks.
bool Wanted();

// Every root descriptor table bind, from the D3D12_Hooks table hooks. Does nothing unless a capture is running.
void NoteTable(bool graphics, UINT rootParameterIndex, D3D12_GPU_DESCRIPTOR_HANDLE base,
               ID3D12GraphicsCommandList* cmd);

// Driven by FrameCapture_Dx12, which owns the capture's lifetime and its folder.
void Start();
void SetFrame(int frame);
void Stop();

// The eye shape, reported from the splice every frame. Groups are only kept when they read something at screen
// shape; without it the whole budget goes on geometry passes and their material textures.
void SetEyeShape(UINT width, UINT height);

// The bindings.txt text for the capture just finished, or empty if nothing was recorded. Called on the present
// thread; the caller writes it on its background thread with the rest of the capture.
std::string Take();
} // namespace BindingCapture_Dx12
