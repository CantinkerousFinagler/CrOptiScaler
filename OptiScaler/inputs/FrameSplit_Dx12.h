#pragma once

#include <d3d12.h>

#include <string>
#include <vector>

// Diagnostics: where the GAME's GPU frame goes (raster at which size, compute, ray tracing, copies, and the upscaler
// with our passes), for deciding whether foveated shading (VRS) of the game's own raster passes can pay.
//
// Startup only: [Inputs] FrameSplitProbe = true installs the command-list hooks (one atomic load per call while idle).
// Live: the overlay button arms one measurement. After a 5 s delay (time to close the overlay) it writes a GPU
// timestamp into the game's command lists at every change of pass kind for 64 Presents, then reads them back and logs
// the split per Present, and the timeline of one Present. A timestamp and a query resolve are the only things it
// records into the game's lists -- no state is touched.
namespace FrameSplit_Dx12
{
bool Enabled();

// From D3D12 device hooking; installs the command-list hooks once (only when Enabled()).
void Install(ID3D12Device* device);

// Every Present (present thread).
void OnPresent(ID3D12CommandQueue* queue);

// Around NVSDK_NGX_D3D12_EvaluateFeature: everything recorded between them is "upscaler + our passes".
void BeginOurs(ID3D12GraphicsCommandList* cmdList);
void EndOurs(ID3D12GraphicsCommandList* cmdList);

// Overlay.
void Arm();
struct Status
{
    bool installed = false;
    int phase = 0; // 0 idle, 1 waiting, 2 capturing, 3 draining
    std::vector<std::string> lines; // the last result, short form
};
Status GetStatus();
} // namespace FrameSplit_Dx12
