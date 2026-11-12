#pragma once

// Neural Rendering through the driver's own nvngx.dll, instead of through the forwarder: the way
// DLSS itself is called. OptiScaler already tells the driver where to look (NVNGX_FeatureInfo_Paths
// carries the executable folder and OptiScaler's own folder into Init_Ext, the mechanism that finds
// nvngx_dlss.dll and friends), so nothing new is needed to locate the model.
//
// Off by default until it has been shown to produce the same picture as the forwarder path.



#include <d3d12.h>

namespace DlssNr
{
namespace Proxy
{
// True when the driver's nvngx is initialised and exports what this path needs.
bool Available();

// Creates the feature if it does not exist, or if the resolution changed, and evaluates it.
// Returns the NGX result of the evaluate, or 0 when nothing could be attempted.
unsigned int Run(ID3D12GraphicsCommandList* cmdList, ID3D12Device* device, ID3D12Resource* color,
                 ID3D12Resource* depth, ID3D12Resource* motion, ID3D12Resource* output,
                 unsigned int width, unsigned int height, unsigned int guideWidth,
                 unsigned int guideHeight, unsigned int motionWidth, unsigned int motionHeight,
                 unsigned int depthBaseX, unsigned int depthBaseY, unsigned int motionBaseX,
                 unsigned int motionBaseY, bool depthInverted, bool reset, float mvScaleX, float mvScaleY);

// Drops the feature and its parameter block, for a resolution change or shutdown.
void Release();
} // namespace Proxy
} // namespace DlssNr

