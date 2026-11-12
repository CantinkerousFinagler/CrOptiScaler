#pragma once

#include <d3d12.h>

#include "DlssNr_ActiveColor.h"

namespace DlssNr
{
struct FoveaBlitParams
{
    ColorExtent roi;
    float roundness = 0.7f;
    float transition = 0.08f;
    bool showBorder = false;
};

// Lerps the edited compact ROI onto dest (which still holds the untouched frame). Dest needs UAV.
bool BlitFovea(ID3D12Device* device, ID3D12GraphicsCommandList* commands, ID3D12Resource* dest,
               ID3D12Resource* edited, const FoveaBlitParams& params);

void ReleaseFoveaBlit();
} // namespace DlssNr
