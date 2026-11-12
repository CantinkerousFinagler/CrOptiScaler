#pragma once

#include <d3d12.h>
#include <algorithm>
#include <cmath>
#include <optional>

namespace DlssNr
{
struct ColorExtent
{
    unsigned int width;
    unsigned int height;
    unsigned int x = 0;
    unsigned int y = 0;
};

// NGX reports the active image separately from the allocation. No preset names or standard
// resolutions belong here. Non-zero origins still need a separate guide/colour-offset integration.
inline std::optional<ColorExtent> PreSrColorExtent(const D3D12_RESOURCE_DESC& allocation,
                                                  unsigned int renderWidth, unsigned int renderHeight,
                                                  unsigned int baseX = 0, unsigned int baseY = 0)
{
    if (allocation.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D ||
        allocation.SampleDesc.Count != 1 || allocation.DepthOrArraySize != 1 ||
        allocation.Width == 0 || allocation.Width > D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION ||
        allocation.Height == 0 || allocation.Height > D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION ||
        baseX != 0 || baseY != 0)
        return std::nullopt;

    if (renderWidth == 0 && renderHeight == 0)
        return ColorExtent { (unsigned int) allocation.Width, allocation.Height };

    if (renderWidth == 0 || renderHeight == 0 || renderWidth > allocation.Width ||
        renderHeight > allocation.Height)
        return std::nullopt;

    return ColorExtent { renderWidth, renderHeight };
}

inline unsigned int AlignExtent(unsigned int value, unsigned int maximum)
{
    value = (std::max)(32u, value);
    value = (value + 7u) / 8u * 8u;
    return (std::min)(value, maximum);
}

// Centered ellipse/rounded-rect crop inside an origin-zero active image.
inline ColorExtent FoveaInside(ColorExtent full, float widthFrac, float heightFrac, float originX,
                               float originY)
{
    widthFrac = std::clamp(widthFrac, 0.15f, 1.0f);
    heightFrac = std::clamp(heightFrac, 0.15f, 1.0f);
    originX = std::clamp(originX, 0.0f, 1.0f);
    originY = std::clamp(originY, 0.0f, 1.0f);
    if (full.width == 0 || full.height == 0)
        return full;

    unsigned int width = AlignExtent((unsigned int) (full.width * widthFrac + 0.5f), full.width);
    unsigned int height = AlignExtent((unsigned int) (full.height * heightFrac + 0.5f), full.height);
    int x = (int) std::lround(originX * (float) full.width) - (int) (width / 2);
    int y = (int) std::lround(originY * (float) full.height) - (int) (height / 2);
    x = std::clamp(x, 0, (int) (full.width - width));
    y = std::clamp(y, 0, (int) (full.height - height));
    return ColorExtent { width, height, (unsigned int) x, (unsigned int) y };
}

// Both resources must be in COPY_SOURCE/COPY_DEST respectively. An explicit box is essential:
// a whole-resource copy either has mismatched dimensions or overwrites the game's padding.
// toCompact: allocation (origin in active) -> compact 0,0. Otherwise compact 0,0 -> allocation origin.
inline void CopyActiveColor(ID3D12GraphicsCommandList* commands, ID3D12Resource* destination,
                            ID3D12Resource* source, ColorExtent active, bool toCompact = true)
{
    D3D12_TEXTURE_COPY_LOCATION src {};
    src.pResource = source;
    src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    D3D12_TEXTURE_COPY_LOCATION dst {};
    dst.pResource = destination;
    dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    if (toCompact)
    {
        const D3D12_BOX box { active.x, active.y, 0, active.x + active.width, active.y + active.height,
                              1 };
        commands->CopyTextureRegion(&dst, 0, 0, 0, &src, &box);
    }
    else
    {
        const D3D12_BOX box { 0, 0, 0, active.width, active.height, 1 };
        commands->CopyTextureRegion(&dst, active.x, active.y, 0, &src, &box);
    }
}
} // namespace DlssNr
