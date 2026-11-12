#pragma once

#include <vector>

#include <d3d12.h>

#include <cstdint>
#include <string>

struct NVSDK_NGX_Parameter;

// Diagnostics ([Inputs] GiBufferCapture, needs FrameSplitProbe for its hooks; F7 or the overlay's button): the textures
// a DX12 game's passes write, saved at the moment each pass finishes with them. Written to answer one question: under
// alternate-eye rendering, is a fault of the final picture already in the game's lighting buffers, and in which?
//
// Works without knowing the game: a texture leaving UNORDERED_ACCESS or RENDER_TARGET has just been written, through a
// legacy transition or an enhanced texture barrier (Crimson Desert's passes use enhanced ones). Armed, it
// watches every barrier for 4 Presents to learn what exists (name, size, format, order, how many writes), keeps the
// textures of a render-size fraction and the 8-bit pictures past the render size, makes readback on a worker, waits [Inputs] CaptureDelaySeconds, then for 6
// Presents copies the first and last write of each, next to the game's own barrier, on the game's list.
//
// Saved to OptiCapture\<time>\ as .dds + index.txt, with the upscaler's colour and depth of the same Presents as
// f00_colour.dds, f00_depth.dds ... so the eye of a buffer can be read from a picture, not assumed from a counter.
//
// Nothing stalls the render thread. No game resource is ever held: a pointer is identity only, checked against the
// size and format at every use. Off, each hook is one bool test. Global barriers carry no texture, so a texture synced only
// by those is not seen: the survey counts them, and lists textures whose barriers never ended a write.
//
// timeline.txt: every texture barrier of the survey in recording order, with the work each list did between its barriers and
// the order of the submits: who writes a texture, in which Present, and what comes right after it is made readable.
//
// OptiCapture\<time>-camera.txt: what the game told Streamline about its camera over the survey, then every mapped buffer
// searched for that projection (up to a minute, on a worker), with 4 KB around each copy saved twice. With this key on, the
// ID3D12Resource::Map hook and a mapping of every buffer the game maps are on from the start of the game.
//
// Who reads the camera: over the survey the timeline also holds every root CBV (C) and root SRV (S) bind, every pipeline set
// (P) and every CBV created (V). Each Present a worker reads what those addresses point at, up to 16 KB from each, and the
// camera file's `block` lines name each place holding an eye's position or the previous-camera row, for that Present.
// Not seen by those: a block reached through a descriptor made earlier, or copied into a GPU buffer. So the timeline also holds
// buffer copies (B: their sources are read too, their destinations are what a bind may point at), descriptor tables (T), root
// constants (K) and the heaps set (H); every buffer CBV and SRV is followed from launch through the descriptor copies, and the
// camera file's `covers` lines name the descriptors over each place holding the two eyes side by side. Only the latest
// descriptor of each slot is known. Not seen: ExecuteIndirect's arguments. Textures and UAVs are followed too, and each table bind's first
// 64 slots go into the timeline as they were at the bind (D lines): which textures and buffers a pass had in reach, its pipeline on the P line above.
//
// The camera-block test (F9, F10; off at launch, a beep for each step): does anything read the 2768-byte camera block the game copies
// into a GPU buffer at each list's head? The game's copy reads our upload slot instead of its own, so nothing of the game's memory is
// written and no later CPU write can undo it, and the current camera's fields are never touched: geometry that moves means the layout
// is not what was read. F9: off > a metre on the previous camera's row > the whole previous camera a metre to its right, coherent > the
// same eye's previous camera > the same eye's clipToPrevClip alone. F8, the per-eye history test: off > the same eye's matrix and both lighting pairs and their map, each eye's own history put back in
// front of the game's barrier > also the two R32_UINT snapshot pairs (likely packed depth and normals), the partner saved and this eye's last put in, in front of the barrier that starts the write. A finder pairs them by place and refuses anything but the expected set. Then the matrix apart from the game's: its head copy goes out untouched on both queues, and the same eye's
// clipToPrevClip and twin (128 bytes) are copied into the block just ahead of the lighting's history barrier, past the motion texture's writes: > snapshots and lighting swapped, left in for the rest of the frame > the same, the
// game's matrix put back at the first UAV > SRV of the lighting's UAV > UAV batch > that bracket with no swaps > the bracket with only the first lighting pair of the batch swapped, the map and snapshots as ever > only the second.
// F10: both queues > direct only > compute only. A block goes out only if its structure holds; CameraTestStatus counts every skip.
namespace GiCapture_Dx12
{
bool Enabled();
void Arm();
bool Busy();
std::string Status();

// After the game's ResourceBarrier has been recorded, from FrameSplit_Dx12's hook.
void OnBarrier(ID3D12GraphicsCommandList* cmdList, UINT count, const D3D12_RESOURCE_BARRIER* barriers);
// BEFORE the game's ID3D12GraphicsCommandList7::Barrier is recorded: the copy goes in front of it.
void OnEnhancedBarrier(ID3D12GraphicsCommandList7* cmdList, UINT32 count, const D3D12_BARRIER_GROUP* groups);

// A draw (0), dispatch (1), DispatchRays (2) or ExecuteIndirect (3), from FrameSplit_Dx12's hooks: the survey's timeline only.
void OnWork(ID3D12GraphicsCommandList* cmdList, int kind);
// The survey's binds, from FrameSplit_Dx12's hooks, before the game's call.
void OnRootView(ID3D12GraphicsCommandList* cmdList, bool graphics, bool srv, UINT rootIndex, D3D12_GPU_VIRTUAL_ADDRESS address);
void OnPipeline(ID3D12GraphicsCommandList* cmdList, void* pipeline);
void OnSetHeaps(ID3D12GraphicsCommandList* cmdList, UINT count, ID3D12DescriptorHeap* const* heaps);
void OnRootTable(ID3D12GraphicsCommandList* cmdList, bool graphics, UINT rootIndex, uint64_t table);
// From launch: what each root signature declares, and each list's current one. So a table's D line is its real range.
void OnCreateRootSignature(void* signature, const void* blob, SIZE_T bytes);
void OnSetRootSignature(ID3D12GraphicsCommandList* cmdList, bool graphics, void* signature);
void OnRootConstants(ID3D12GraphicsCommandList* cmdList, bool graphics, UINT rootIndex, UINT count, const void* values, UINT offset);
// `bytes` 0: the whole of the source (CopyResource). Buffers only.
void OnCopyBuffer(ID3D12GraphicsCommandList* cmdList, ID3D12Resource* dest, UINT64 destOffset, ID3D12Resource* source, UINT64 sourceOffset, UINT64 bytes);
// After OnCopyBuffer, before the game's copy is recorded: may point the copy at another source.
void PatchCameraCopy(ID3D12GraphicsCommandList* cmdList, ID3D12Resource*& source, UINT64& sourceOffset, UINT64 bytes, ID3D12Resource* dest = nullptr, UINT64 destOffset = 0);
void CycleCameraEffect();
void CycleCameraQueues();
std::string CameraTestStatus();
void CycleHistoryTest();
// [Inputs] GiBufferCapture: the captures, the test keys and steps, the saved shaders. [Inputs] PerEyeLighting alone gets F8 and the overlay's section.
bool Developer();
// F8: off, or everything that has held up so far: 1 beep off, 7 on, 3 low beeps if it cannot find its textures.
void ToggleHistory();
// The overlay's section: the step, and what is really running, over the last half second or so of Presents.
struct HistoryUi
{
    int step = 0;
    int textures = 0;       // the pairs found; none while it looks
    bool correcting = false; // the same-eye motion went in
    bool limitRunning = false; // the edited shaders were bound
    uint64_t shadersEdited = 0, otherVersions = 0;
};
HistoryUi HistoryUiStatus();
void SetHistoryStep(int step);
// F11: 180 Presents of a box in the middle of the view, to OptiCapture\<time>-clip: 2 beeps when it starts, 3 when it is written.
void ArmClip();
std::string HistoryTestStatus();
// From launch: which buffer range each descriptor slot holds.
void OnCreateCbv(const D3D12_CONSTANT_BUFFER_VIEW_DESC* desc, uint64_t dest);
void OnCreateSrv(ID3D12Resource* resource, const D3D12_SHADER_RESOURCE_VIEW_DESC* desc, uint64_t dest);
void OnCreateUav(ID3D12Resource* resource, const D3D12_UNORDERED_ACCESS_VIEW_DESC* desc, uint64_t dest);
// A texture copied to a texture: a snapshot made once a frame shows here. Over the survey only.
void OnCopyTexture(ID3D12GraphicsCommandList* cmdList, ID3D12Resource* dest, ID3D12Resource* source);
void OnCopyDescriptors(ID3D12Device* device, UINT destRanges, const D3D12_CPU_DESCRIPTOR_HANDLE* destStarts, const UINT* destSizes, UINT srcRanges,
                       const D3D12_CPU_DESCRIPTOR_HANDLE* srcStarts, const UINT* srcSizes);
// From the slSetConstants hook: the game's camera, for the search. Matrices as 16 floats, as Streamline's rows.
void OnCameraConstants(const float* viewToClip, const float* clipToPrevClip, const float* jitter, const float* pos, const float* up,
                       const float* right, const float* fwd, float nearZ, float farZ, uint32_t frame);
// A compute pipeline the game has just made, with its bytecode: the two shaders of the history limit get their edited twins.
void OnPipelineCreated(ID3D12PipelineState* pipeline, ID3D12RootSignature* root, const void* bytes, size_t length);
// When the game binds a pipeline: the edited twin, on the steps that want it.
ID3D12PipelineState* SwapPipeline(ID3D12PipelineState* pipeline);
// The offline test: an edited shader as the game would get it.
bool PipelineForTest(int shader, int level, const void* bytes, std::vector<uint8_t>& out);
// The offline test: F8 through every step, not only the game's.
void HistoryAllSteps(bool all);
// Before ExecuteCommandLists: the timeline's submit order.
void OnExecute(ID3D12CommandQueue* queue, ID3D12CommandList* const* lists, UINT count);
// The same place: true when our list has to run among these, and out is what to submit instead.
bool SpliceExecute(ID3D12CommandQueue* queue, ID3D12CommandList* const* lists, UINT count, std::vector<ID3D12CommandList*>& out);

// At the top of NVSDK_NGX_D3D12_EvaluateFeature.
void OnEvaluate(ID3D12GraphicsCommandList* cmdList, int featureId, NVSDK_NGX_Parameter* params);

// Every Present, from the swapchain wrapper.
void OnPresent(ID3D12CommandQueue* queue);
} // namespace GiCapture_Dx12
