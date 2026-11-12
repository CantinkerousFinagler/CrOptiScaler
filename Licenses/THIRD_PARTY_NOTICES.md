# Third-party binaries in this repository

The repository carries these prebuilt files so that a clone builds without further downloads. They are the same
files the upstream [OptiScaler](https://github.com/optiscaler/OptiScaler) repository ships (byte-identical to its
v0.9.4 release); none of them is modified here.

| Files | Origin | Licence |
|---|---|---|
| `OptiScaler/library/detours/detours.lib` | [Microsoft Detours](https://github.com/microsoft/Detours) | MIT (below) |
| `OptiScaler/library/fsr2/*.lib`, `fsr2_212/*.lib`, `fsr31/*.lib` | [AMD FidelityFX SDK](https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK) (FSR 2.1 / 2.1.2 / 3.1) | MIT, see `external/FidelityFX-SDK/LICENSE.txt` |
| `OptiScaler/library/vulkan/vulkan-1.lib` | [Vulkan SDK](https://vulkan.lunarg.com/) import library | Apache-2.0, see `external/vulkan/LICENSES/Apache-2.0.txt` |
| `OptiScaler/library/d3dx/d3dx11.lib` | Microsoft DirectX SDK (June 2010) import library | Microsoft DirectX SDK EULA (redistributable import library) |
| `OptiScaler/shaders/shader_tools/dxc.exe`, `dxcompiler.dll`, `dxil.dll`, `dxv.exe` | [DirectX Shader Compiler](https://github.com/microsoft/DirectXShaderCompiler) | University of Illinois / NCSA Open Source License (the LLVM release licence) |
| `OptiScaler/shaders/shader_tools/fxc.exe` | Microsoft Windows SDK (legacy HLSL compiler) | Microsoft Windows SDK licence |
| `external/freetype/freetype.lib` | [FreeType](https://gitlab.freedesktop.org/freetype/freetype) 2.13.3 | [FTL](https://gitlab.freedesktop.org/freetype/freetype/-/blob/master/docs/FTL.TXT) |
| `external/directx_agility_sdk/lib/D3D12Core.dll` | Microsoft DirectX 12 Agility SDK | Microsoft Software License Terms (redistributable) |

The shader tools are only needed to recompile the HLSL in `OptiScaler/shaders`; the precompiled shaders are in the
tree. The submodules under `external/` carry their own licence files.

## Microsoft Detours

MIT License

Copyright (c) Microsoft Corporation.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of
the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
