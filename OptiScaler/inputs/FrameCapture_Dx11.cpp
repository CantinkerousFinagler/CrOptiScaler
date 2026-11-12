#include "pch.h"
#include "FrameCapture_Dx11.h"
#include "TaaMotion_Dx11.h"
#include "Fc5Dlss_Dx11.h"
#include "Fc5Taa_Dx11.h"

#include "Config.h"
#include "Logger.h"
#include "Util.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <format>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace
{
// All of this runs on the thread driving the immediate context and calling Present; only the arm request crosses
// threads (the overlay's hotkey handling).
enum class CaptureState
{
    Idle,
    Learning,
    Capturing,
};

CaptureState g_state = CaptureState::Idle;
std::atomic<bool> g_armRequested { false };
int g_presentsInState = 0;
constexpr int kMaxPresentsPerState = 10;

// The pass: the last dispatch of a Present whose UAV 0 is an eye-sized texture, on this context. Learned over two
// Presents because the output can be a different texture per eye (Days Gone).
ID3D11DeviceContext* g_targetCtx = nullptr;
ID3D11Texture2D* g_frameTarget = nullptr; // this Present's candidate while learning
ID3D11Texture2D* g_targets[2] = {};
UINT g_targetGroups[3] = {};
constexpr int kLearnPresents = 2;

// Three consecutive Presents of the pass (four for a TAA draw): frames 0 and 2 are the same eye, 0 and 1 the two
// eyes. Textures on frame 0 only; constant buffers on all of them.
constexpr int kCaptureFrames = 4;
int g_frame = 0;
bool g_capturedThisFrame = false;
ID3D11Texture2D* g_frameOutput[kCaptureFrames] = {}; // not held; UAV 0 per frame, for the index
// Per captured frame: the Present it came from (counted from the start of capturing) and when, for the index -- the
// stale-eye test needs to know the frames were consecutive Presents and how far apart they were.
int g_framePresent[kCaptureFrames] = {};
int64_t g_frameTicks[kCaptureFrames] = {};

struct CbSnapshot
{
    int frame = 0;
    const char* stage = "cs";
    std::string note; // which draw, for draws other than the pass
    UINT slot = 0;
    ID3D11Buffer* source = nullptr; // not held
    UINT bytes = 0;
    ID3D11Buffer* staging = nullptr;
};
std::vector<CbSnapshot> g_cbSnapshots;

// Shader bytecode by shader object, recorded at creation (shaders are created on several threads). A reused
// address is simply overwritten by the newer shader.
std::mutex g_shaderMutex;
std::unordered_map<const void*, std::vector<char>> g_shaderCode;
// The captured pass's compute shader, copied on frame 0.
const void* g_capturedShader = nullptr;
std::vector<char> g_capturedShaderCode;

// The last draw into an eye-sized render target before Present (Days Gone below 100% render scale: the upscale
// after TAA), learned by its pixel shader and captured once on frame 0 after the pass: pixel shader, constants,
// SRVs, and RT 0 before and after. Shader pointers are compared only, not held.
const void* g_frameDrawShader = nullptr; // this Present's candidate while learning
const void* g_drawShader = nullptr;
bool g_drawDone = false;
bool g_pendingAfterDraw = false;
std::vector<char> g_drawShaderCode;

// The TAA draw (FC5: a pixel-shader pass at render size, no compute): the draw that last wrote the texture the first
// draw into the eye target (render target only, never sampled) reads. Learned by its pixel shader; when it's the pass,
// g_targetCtx, g_capturedShader and g_frameOutput refer to it instead of a dispatch.
std::unordered_map<ID3D11Texture2D*, const void*> g_rtProducers; // this Present: RT 0 -> pixel shader, not held
const void* g_frameTaaShader = nullptr; // this Present's candidate while learning
ID3D11DeviceContext* g_frameTaaCtx = nullptr;
UINT g_frameTaaSize[2] = {};
const void* g_taaShader = nullptr;
ID3D11DeviceContext* g_taaCtx = nullptr;
UINT g_taaSize[2] = {};
bool g_drawPass = false;
// Frame 0: the last depth bound with a render-size target before the TAA draw (the scene depth). Held.
ID3D11Texture2D* g_sceneDepth = nullptr;
// The motion vector pass behind the TAA draw's t0 (FC5: a compute dispatch writing it as UAV 0): the pixel and
// compute shaders that wrote t0 on frame 0; on later frames the constants of their draws and dispatches (up to
// kMaxMvDraws per Present), and the first time each shader is seen its SRVs and bytecode.
std::unordered_map<ID3D11Texture2D*, std::vector<const void*>> g_frameWriters; // this Present, not held
std::vector<const void*> g_mvShaders;
D3D11_TEXTURE2D_DESC g_mvDesc {};
struct MvShader
{
    const void* shader = nullptr;
    const char* stage = "ps";
    std::vector<char> code; // may be empty
};
std::vector<MvShader> g_mvShaderCode;
int g_mvDraws = 0;
constexpr int kMaxMvDraws = 8;
// The first G-buffer draw of each frame (3+ render targets of the TAA size, with depth): its vs constants (the
// projection, for the jitter) and, once, its vertex shader.
bool g_gbufferDone = false;
std::vector<MvShader> g_gbufferShaders;
// Dispatch mode (Days Gone): the pass's R32_UINT motion input (texture desc, learned with the pass). Every draw (RT
// 0-7) or dispatch (UAV 0-7) on the pass's context that writes a texture of that size and format before the pass gets
// its constants saved on each captured frame (up to kMaxMvDraws per Present) and its bytecode once, to find the camera
// matrices behind the motion.
D3D11_TEXTURE2D_DESC g_frameComputeMvDesc {}; // this Present's candidate while learning
D3D11_TEXTURE2D_DESC g_computeMvDesc {};

int CaptureFrames() { return g_drawPass ? 4 : 3; }

// [Inputs] CaptureStaleEye on a TAA draw: the TAA output and scene depth of all four frames, none of frame 0's inputs
// or motion pass textures (they would not fit the budget beside them).
bool StaleEyeMode() { return g_drawPass && Config::Instance()->CaptureStaleEye.value_or_default(); }

struct Snapshot
{
    std::string kind; // "srv t3", "uav u0 before", "uav u0 after"
    ID3D11Texture2D* source = nullptr; // not held; for the index only
    D3D11_TEXTURE2D_DESC desc {};
    ID3D11Texture2D* staging = nullptr;
};
std::vector<Snapshot> g_snapshots;
bool g_pendingAfter = false;
uint64_t g_bytes = 0;
// Read back on the render thread; 2 GB froze the game long enough to stall the VR runtime.
constexpr uint64_t kMaxBytes = 768ull * 1024 * 1024;
// Stale-eye mode: four frames of output + depth are ~790 MB at a 4000x4128 eye.
constexpr uint64_t kMaxStaleEyeBytes = 1024ull * 1024 * 1024;
// Cutscenes use 8192x8192 targets; those aren't eye images.
// Eyes reach 4000x4128 at high VR resolutions; 8192 still excludes the cutscene ones.
constexpr UINT kMaxSide = 6144;

// Same eye-size limits as the NR backends ([Inputs] DuniaTaaMinEvalWidth / DuniaTaaMinEvalHeight).
bool IsEyeSize(UINT width, UINT height)
{
    const auto minW = (UINT) std::max(0, Config::Instance()->DuniaTaaMinEvalWidth.value_or_default());
    const auto minH = (UINT) std::max(0, Config::Instance()->DuniaTaaMinEvalHeight.value_or_default());
    return width >= minW || height >= minH;
}

// Bits per pixel of uncompressed formats seen in game render targets and inputs; 0 for anything else (skipped).
UINT BitsPerPixel(DXGI_FORMAT format)
{
    const UINT f = (UINT) format;
    if (f >= 1 && f <= 4)
        return 128;
    if (f >= 5 && f <= 8)
        return 96;
    if (f >= 9 && f <= 22)
        return 64;
    if (f >= 23 && f <= 47)
        return 32;
    if (f >= 48 && f <= 59)
        return 16;
    if (f >= 60 && f <= 65)
        return 8;
    switch (format)
    {
    case DXGI_FORMAT_R9G9B9E5_SHAREDEXP:
    case DXGI_FORMAT_B8G8R8A8_UNORM:
    case DXGI_FORMAT_B8G8R8X8_UNORM:
    case DXGI_FORMAT_R10G10B10_XR_BIAS_A2_UNORM:
    case DXGI_FORMAT_B8G8R8A8_TYPELESS:
    case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
    case DXGI_FORMAT_B8G8R8X8_TYPELESS:
    case DXGI_FORMAT_B8G8R8X8_UNORM_SRGB:
        return 32;
    case DXGI_FORMAT_B5G6R5_UNORM:
    case DXGI_FORMAT_B5G5R5A1_UNORM:
    case DXGI_FORMAT_B4G4R4A4_UNORM:
        return 16;
    default:
        return 0;
    }
}

// Returns an AddRef'd Texture2D, or nullptr (buffers and other dimensions are skipped).
ID3D11Texture2D* TextureFromView(ID3D11View* view)
{
    if (view == nullptr)
        return nullptr;
    ID3D11Resource* resource = nullptr;
    view->GetResource(&resource);
    if (resource == nullptr)
        return nullptr;
    ID3D11Texture2D* texture = nullptr;
    resource->QueryInterface(IID_PPV_ARGS(&texture));
    resource->Release();
    return texture;
}

void Snap(ID3D11DeviceContext* ctx, ID3D11Texture2D* texture, std::string kind)
{
    D3D11_TEXTURE2D_DESC desc {};
    texture->GetDesc(&desc);
    const UINT bpp = BitsPerPixel(desc.Format);
    uint64_t bytes = (uint64_t) desc.Width * desc.Height * bpp / 8 * desc.ArraySize;
    if (desc.MipLevels > 1)
        bytes = bytes * 4 / 3;
    if (desc.Width > kMaxSide || desc.Height > kMaxSide)
    {
        LOG_INFO("Frame capture: skipping {} ({}x{} fmt {}): larger than {} per side", kind, desc.Width, desc.Height,
                 (UINT) desc.Format, kMaxSide);
        return;
    }
    if (bpp == 0 || desc.SampleDesc.Count != 1)
    {
        LOG_INFO("Frame capture: skipping {} ({}x{} fmt {}, {} samples: format not supported)", kind, desc.Width,
                 desc.Height, (UINT) desc.Format, desc.SampleDesc.Count);
        return;
    }
    const uint64_t budget = StaleEyeMode() ? kMaxStaleEyeBytes : kMaxBytes;
    if (g_bytes + bytes > budget)
    {
        LOG_WARN("Frame capture: skipping {} ({}x{} fmt {}): over the {} MB budget", kind, desc.Width, desc.Height,
                 (UINT) desc.Format, budget >> 20);
        return;
    }

    D3D11_TEXTURE2D_DESC stagingDesc = desc;
    stagingDesc.Usage = D3D11_USAGE_STAGING;
    stagingDesc.BindFlags = 0;
    stagingDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    stagingDesc.MiscFlags = 0;

    ID3D11Device* device = nullptr;
    ctx->GetDevice(&device);
    ID3D11Texture2D* staging = nullptr;
    const HRESULT result = device != nullptr ? device->CreateTexture2D(&stagingDesc, nullptr, &staging) : E_FAIL;
    if (device != nullptr)
        device->Release();
    if (FAILED(result) || staging == nullptr)
    {
        LOG_WARN("Frame capture: staging texture for {} ({}x{} fmt {}) failed: {:X}", kind, desc.Width, desc.Height,
                 (UINT) desc.Format, (UINT) result);
        return;
    }

    ctx->CopyResource(staging, texture);
    g_bytes += bytes;
    g_snapshots.push_back({ std::move(kind), texture, desc, staging });
}

// Snapshots every Texture2D bound as a compute (or pixel) SRV (slots 0-31) and UAV (slots 0-7).
void SnapInputs(ID3D11DeviceContext* ctx, bool pixel = false, const std::string& prefix = "draw ")
{
    ID3D11ShaderResourceView* srvs[32] = {};
    if (pixel)
        ctx->PSGetShaderResources(0, 32, srvs);
    else
        ctx->CSGetShaderResources(0, 32, srvs);
    for (UINT i = 0; i < 32; ++i)
    {
        if (srvs[i] == nullptr)
            continue;
        if (ID3D11Texture2D* texture = TextureFromView(srvs[i]))
        {
            Snap(ctx, texture, std::format("{}srv t{}", pixel || prefix != "draw " ? prefix : std::string {}, i));
            texture->Release();
        }
        srvs[i]->Release();
    }
}

void SnapUavs(ID3D11DeviceContext* ctx, const char* when)
{
    ID3D11UnorderedAccessView* uavs[8] = {};
    ctx->CSGetUnorderedAccessViews(0, 8, uavs);
    for (UINT i = 0; i < 8; ++i)
    {
        if (uavs[i] == nullptr)
            continue;
        if (ID3D11Texture2D* texture = TextureFromView(uavs[i]))
        {
            Snap(ctx, texture, std::format("uav u{} {}", i, when));
            texture->Release();
        }
        uavs[i]->Release();
    }
}

// Every constant buffer bound to one stage ("cs", "ps" or "vs"; slots 0-13).
void SnapConstantBuffers(ID3D11DeviceContext* ctx, int frame, const char* stage = "cs", const std::string& note = {})
{
    ID3D11Buffer* buffers[D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT] = {};
    if (stage[0] == 'p')
        ctx->PSGetConstantBuffers(0, D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT, buffers);
    else if (stage[0] == 'v')
        ctx->VSGetConstantBuffers(0, D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT, buffers);
    else
        ctx->CSGetConstantBuffers(0, D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT, buffers);
    ID3D11Device* device = nullptr;
    ctx->GetDevice(&device);
    for (UINT i = 0; i < D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT; ++i)
    {
        if (buffers[i] == nullptr)
            continue;
        D3D11_BUFFER_DESC desc {};
        buffers[i]->GetDesc(&desc);
        D3D11_BUFFER_DESC stagingDesc {};
        stagingDesc.ByteWidth = desc.ByteWidth;
        stagingDesc.Usage = D3D11_USAGE_STAGING;
        stagingDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        ID3D11Buffer* staging = nullptr;
        if (device != nullptr && SUCCEEDED(device->CreateBuffer(&stagingDesc, nullptr, &staging)) && staging != nullptr)
        {
            ctx->CopyResource(staging, buffers[i]);
            g_cbSnapshots.push_back({ frame, stage, note, i, buffers[i], desc.ByteWidth, staging });
        }
        else
        {
            LOG_WARN("Frame capture: staging buffer for cb{} ({} bytes) failed", i, desc.ByteWidth);
        }
        buffers[i]->Release();
    }
    if (device != nullptr)
        device->Release();
}

// Mip 0 of a read-back texture, rows packed.
struct Pixels
{
    std::filesystem::path path;
    D3D11_TEXTURE2D_DESC desc {};
    std::vector<char> rows;
};

void WriteDds(const Pixels& pixels)
{
    const auto& desc = pixels.desc;
    const UINT rowBytes = desc.Width * BitsPerPixel(desc.Format) / 8;

    uint32_t header[32] = {};
    header[0] = 0x20534444; // "DDS "
    header[1] = 124;
    header[2] = 0x1 | 0x2 | 0x4 | 0x8 | 0x1000; // caps, height, width, pitch, pixel format
    header[3] = desc.Height;
    header[4] = desc.Width;
    header[5] = rowBytes;
    header[7] = 1; // mips
    header[19] = 32; // pixel format size
    header[20] = 0x4; // FOURCC
    header[21] = 0x30315844; // "DX10"
    header[27] = 0x1000; // TEXTURE
    const uint32_t dx10[5] = { (uint32_t) desc.Format, 3 /* TEXTURE2D */, 0, 1, 0 };

    std::ofstream file(pixels.path, std::ios::binary);
    file.write(reinterpret_cast<const char*>(header), sizeof(header));
    file.write(reinterpret_cast<const char*>(dx10), sizeof(dx10));
    file.write(pixels.rows.data(), (std::streamsize) pixels.rows.size());
}

void Reset()
{
    for (auto& snapshot : g_snapshots)
    {
        if (snapshot.staging != nullptr)
            snapshot.staging->Release();
    }
    g_snapshots.clear();
    for (auto& snapshot : g_cbSnapshots)
    {
        if (snapshot.staging != nullptr)
            snapshot.staging->Release();
    }
    g_cbSnapshots.clear();
    g_capturedShader = nullptr;
    g_capturedShaderCode.clear();
    g_frameDrawShader = nullptr;
    g_drawShader = nullptr;
    g_drawDone = false;
    g_pendingAfterDraw = false;
    g_drawShaderCode.clear();
    g_rtProducers.clear();
    g_frameWriters.clear();
    g_mvShaders.clear();
    g_mvShaderCode.clear();
    g_mvDraws = 0;
    g_gbufferDone = false;
    g_gbufferShaders.clear();
    g_frameComputeMvDesc = {};
    g_computeMvDesc = {};
    g_frameTaaShader = nullptr;
    g_frameTaaCtx = nullptr;
    g_taaShader = nullptr;
    g_taaCtx = nullptr;
    g_drawPass = false;
    if (g_sceneDepth != nullptr)
        g_sceneDepth->Release();
    g_sceneDepth = nullptr;
    for (auto*& target : g_targets)
    {
        if (target != nullptr)
            target->Release();
        target = nullptr;
    }
    if (g_frameTarget != nullptr)
        g_frameTarget->Release();
    g_frameTarget = nullptr;
    g_targetCtx = nullptr;
    g_pendingAfter = false;
    g_capturedThisFrame = false;
    g_frame = 0;
    std::fill(std::begin(g_frameOutput), std::end(g_frameOutput), nullptr);
    g_bytes = 0;
    g_presentsInState = 0;
    g_state = CaptureState::Idle;
}

// 16 bytes per line: byte offset, the four words as floats, then in hex.
std::string DescribeConstants(const std::vector<uint32_t>& words)
{
    std::string text;
    for (size_t i = 0; i < words.size(); i += 4)
    {
        float f[4] = {};
        uint32_t w[4] = {};
        for (size_t j = 0; j < 4 && i + j < words.size(); ++j)
        {
            w[j] = words[i + j];
            std::memcpy(&f[j], &w[j], 4);
        }
        text += std::format("  {:4}: {:>14.6g} {:>14.6g} {:>14.6g} {:>14.6g} | {:08X} {:08X} {:08X} {:08X}\n", i * 4,
                            f[0], f[1], f[2], f[3], w[0], w[1], w[2], w[3]);
    }
    return text;
}

// Reads the snapshots back (Map waits for the copies) on the render thread, then writes the files on a background
// thread so the game only stalls for the read-back. Textures: mip 0 and array slice 0 only.
void WriteCapture()
{
    std::time_t now = std::time(nullptr);
    std::tm local {};
    localtime_s(&local, &now);
    char stamp[32] = {};
    std::strftime(stamp, sizeof(stamp), "%Y%m%d-%H%M%S", &local);
    const auto folder = Util::ExePath().parent_path() / "OptiCapture" / stamp;
    std::error_code error;
    std::filesystem::create_directories(folder, error);
    if (error)
    {
        LOG_ERROR("Frame capture: can't create {}: {}", folder.string(), error.message());
        return;
    }

    std::string index =
        g_drawPass ? std::format("TAA draw (pixel shader {:X}) into {}x{} on context {:X}; constants: ps and vs\n",
                                 (size_t) g_taaShader, g_taaSize[0], g_taaSize[1], (size_t) g_targetCtx)
                   : std::format("Dispatch {}x{}x{} on context {:X}; outputs (UAV 0) {:X} {:X}\n", g_targetGroups[0],
                                 g_targetGroups[1], g_targetGroups[2], (size_t) g_targetCtx, (size_t) g_targets[0],
                                 (size_t) g_targets[1]);
    LARGE_INTEGER frequency {};
    QueryPerformanceFrequency(&frequency);
    for (int f = 0; f < g_frame; ++f)
        index += std::format("Frame {}: {} {:X}, Present {}, t {:.3f} ms\n", f, g_drawPass ? "RT 0" : "UAV 0",
                             (size_t) g_frameOutput[f], g_framePresent[f],
                             frequency.QuadPart > 0
                                 ? (g_frameTicks[f] - g_frameTicks[0]) * 1000.0 / (double) frequency.QuadPart
                                 : 0.0);
    if (StaleEyeMode())
        index += "Stale-eye capture ([Inputs] CaptureStaleEye): \"fN scene depth\" and \"rtK after fN\" per frame, "
                 "no frame 0 inputs\n";
    const auto shaderName = std::format("{}_{:X}.dxbc", g_drawPass ? "ps" : "cs", (size_t) g_capturedShader);
    index += g_capturedShaderCode.empty()
                 ? std::format("{} shader: bytecode not recorded (created before the hooks?)\n",
                               g_drawPass ? "Pixel" : "Compute")
                 : std::format("{} shader: {} ({} bytes)\n", g_drawPass ? "Pixel" : "Compute", shaderName,
                               g_capturedShaderCode.size());
    if (g_drawPass)
    {
        index += "TAA draw files: \"draw srv\" inputs, \"rtN before/after\" outputs, \"scene depth\"\n";
        index += std::format("Motion vector pass (wrote t0 on frame 0): {} shaders; constants from frame 1 on, "
                             "\"mv\" files\n",
                             g_mvShaders.size());
        for (const auto& gb : g_gbufferShaders)
            index += gb.code.empty() ? std::format("  G-buffer vs shader {:X}: bytecode not recorded\n", (size_t) gb.shader)
                                     : std::format("  gbuffer_vs_{:X}.dxbc ({} bytes)\n", (size_t) gb.shader, gb.code.size());
        for (const auto& mv : g_mvShaderCode)
            index += mv.code.empty()
                         ? std::format("  mv {} shader {:X}: bytecode not recorded\n", mv.stage, (size_t) mv.shader)
                         : std::format("  mv_{}_{:X}.dxbc ({} bytes)\n", mv.stage, (size_t) mv.shader, mv.code.size());
    }
    else
    {
        index += g_computeMvDesc.Width == 0
                     ? "Motion writers: the pass has no R32_UINT input\n"
                     : std::format("Motion writers (draws/dispatches writing a {}x{} fmt {} texture before the pass; "
                                   "constants noted \"motion writer\"): {} shaders\n",
                                   g_computeMvDesc.Width, g_computeMvDesc.Height, (UINT) g_computeMvDesc.Format,
                                   g_mvShaderCode.size());
        for (const auto& mv : g_mvShaderCode)
            index += mv.code.empty()
                         ? std::format("  mv {} shader {:X}: bytecode not recorded\n", mv.stage, (size_t) mv.shader)
                         : std::format("  mv_{}_{:X}.dxbc ({} bytes)\n", mv.stage, (size_t) mv.shader, mv.code.size());
    }
    const auto drawShaderName = std::format("ps_{:X}.dxbc", (size_t) g_drawShader);
    if (g_drawPass)
        ; // the TAA draw is the pass
    else if (g_drawShader == nullptr)
        index += "Last eye-sized draw: none seen\n";
    else if (!g_drawDone)
        index += std::format("Last eye-sized draw: pixel shader {:X}, didn't run after the pass on frame 0\n",
                             (size_t) g_drawShader);
    else
        index += g_drawShaderCode.empty()
                     ? std::format("Last eye-sized draw: pixel shader {:X}, bytecode not recorded (\"draw\" files)\n",
                                   (size_t) g_drawShader)
                     : std::format("Last eye-sized draw: pixel shader {} ({} bytes) (\"draw\" files)\n",
                                   drawShaderName, g_drawShaderCode.size());
    index += "Textures (frame 0): file | binding | source | width x height | format | bind flags | mips | array\n";

    std::vector<Pixels> images;
    for (size_t i = 0; i < g_snapshots.size(); ++i)
    {
        auto& snapshot = g_snapshots[i];
        std::string kind = snapshot.kind;
        std::replace(kind.begin(), kind.end(), ' ', '-');
        const auto name = std::format("{:02}_{}_{:X}_{}x{}_fmt{}.dds", i, kind, (size_t) snapshot.source,
                                      snapshot.desc.Width, snapshot.desc.Height, (UINT) snapshot.desc.Format);

        D3D11_MAPPED_SUBRESOURCE map {};
        const HRESULT result = g_targetCtx->Map(snapshot.staging, 0, D3D11_MAP_READ, 0, &map);
        if (FAILED(result))
        {
            LOG_WARN("Frame capture: Map of {} failed: {:X}", snapshot.kind, (UINT) result);
            continue;
        }
        Pixels pixels { folder / name, snapshot.desc, {} };
        const size_t rowBytes = (size_t) snapshot.desc.Width * BitsPerPixel(snapshot.desc.Format) / 8;
        pixels.rows.resize(rowBytes * snapshot.desc.Height);
        const auto* data = static_cast<const char*>(map.pData);
        for (UINT y = 0; y < snapshot.desc.Height; ++y)
            std::memcpy(pixels.rows.data() + y * rowBytes, data + (size_t) y * map.RowPitch, rowBytes);
        g_targetCtx->Unmap(snapshot.staging, 0);
        images.push_back(std::move(pixels));

        index += std::format("{} | {} | {:X} | {}x{} | {} | {:X} | {} | {}\n", name, snapshot.kind,
                             (size_t) snapshot.source, snapshot.desc.Width, snapshot.desc.Height,
                             (UINT) snapshot.desc.Format, snapshot.desc.BindFlags, snapshot.desc.MipLevels,
                             snapshot.desc.ArraySize);
    }

    std::string constants;
    for (auto& snapshot : g_cbSnapshots)
    {
        D3D11_MAPPED_SUBRESOURCE map {};
        if (FAILED(g_targetCtx->Map(snapshot.staging, 0, D3D11_MAP_READ, 0, &map)))
            continue;
        std::vector<uint32_t> words(snapshot.bytes / 4);
        std::memcpy(words.data(), map.pData, words.size() * 4);
        g_targetCtx->Unmap(snapshot.staging, 0);
        constants += std::format("frame {} {} cb{} buffer {:X} {} bytes{}\n", snapshot.frame, snapshot.stage,
                                 snapshot.slot, (size_t) snapshot.source, snapshot.bytes, snapshot.note);
        constants += DescribeConstants(words);
    }

    LOG_INFO("Frame capture: read back {} of {} textures ({} MB) and {} constant buffers over {} frames; writing to {}",
             images.size(), g_snapshots.size(), g_bytes >> 20, g_cbSnapshots.size(), g_frame, folder.string());

    std::thread(
        [folder, index = std::move(index), constants = std::move(constants), images = std::move(images),
         shaderName, shader = std::move(g_capturedShaderCode), drawShaderName,
         drawShader = std::move(g_drawShaderCode), mvShaders = std::move(g_mvShaderCode),
         gbufferShaders = std::move(g_gbufferShaders)]()
        {
            std::ofstream(folder / "index.txt") << index;
            std::ofstream(folder / "constants.txt") << constants;
            if (!shader.empty())
                std::ofstream(folder / shaderName, std::ios::binary).write(shader.data(), (std::streamsize) shader.size());
            if (!drawShader.empty())
                std::ofstream(folder / drawShaderName, std::ios::binary)
                    .write(drawShader.data(), (std::streamsize) drawShader.size());
            for (const auto& gb : gbufferShaders)
            {
                if (!gb.code.empty())
                    std::ofstream(folder / std::format("gbuffer_vs_{:X}.dxbc", (size_t) gb.shader), std::ios::binary)
                        .write(gb.code.data(), (std::streamsize) gb.code.size());
            }
            for (const auto& mv : mvShaders)
            {
                if (!mv.code.empty())
                    std::ofstream(folder / std::format("mv_{}_{:X}.dxbc", mv.stage, (size_t) mv.shader),
                                  std::ios::binary)
                        .write(mv.code.data(), (std::streamsize) mv.code.size());
            }
            for (const auto& pixels : images)
                WriteDds(pixels);
            LOG_INFO("Frame capture: {} files written to {}", images.size() + 2, folder.string());
        })
        .detach();
}
// Draw mode: a draw (RT 0) or dispatch (UAV 0) writing `output`. Frame 0 records the writers; later frames capture
// the ones that wrote the TAA's t0.
void WatchMvWriter(ID3D11DeviceContext* ctx, ID3D11Texture2D* output, const void* shader, bool compute)
{
    if (output == nullptr || shader == nullptr)
        return;
    if (g_frame == 0)
    {
        auto& writers = g_frameWriters[output];
        if (std::find(writers.begin(), writers.end(), shader) == writers.end())
            writers.push_back(shader);
        return;
    }
    if (g_mvDraws >= kMaxMvDraws || std::find(g_mvShaders.begin(), g_mvShaders.end(), shader) == g_mvShaders.end())
        return;
    D3D11_TEXTURE2D_DESC desc {};
    output->GetDesc(&desc);
    if (desc.Width != g_mvDesc.Width || desc.Height != g_mvDesc.Height || desc.Format != g_mvDesc.Format)
        return;

    const char* stage = compute ? "cs" : "ps";
    const auto note = std::format(" (motion vector {} {}, {} shader {:X})", compute ? "dispatch" : "draw",
                                  g_mvDraws++, stage, (size_t) shader);
    SnapConstantBuffers(ctx, g_frame, stage, note);
    if (!compute)
        SnapConstantBuffers(ctx, g_frame, "vs", note);
    const bool seen = std::any_of(g_mvShaderCode.begin(), g_mvShaderCode.end(),
                                  [shader](const auto& entry) { return entry.shader == shader; });
    if (seen)
        return;
    std::vector<char> code;
    {
        std::lock_guard lock(g_shaderMutex);
        if (auto it = g_shaderCode.find(shader); it != g_shaderCode.end())
            code = it->second;
    }
    g_mvShaderCode.push_back({ shader, stage, std::move(code) });
    if (!StaleEyeMode()) // its textures would not fit beside four frames; its constants (the matrix) are what's needed
        SnapInputs(ctx, !compute, std::format("mv {:X} ", (size_t) shader));
}

// Dispatch mode: a draw or dispatch on the pass's context before the pass this Present; if one of its outputs matches
// the pass's motion input, its constants (and bytecode once). No textures: the budget goes to the pass's own inputs.
void WatchComputeMvWriters(ID3D11DeviceContext* ctx, bool compute)
{
    if (g_drawPass || g_state != CaptureState::Capturing || ctx != g_targetCtx || g_capturedThisFrame ||
        g_pendingAfter || g_computeMvDesc.Width == 0 || g_mvDraws >= kMaxMvDraws)
        return;
    ID3D11View* views[8] = {};
    if (compute)
        ctx->CSGetUnorderedAccessViews(0, 8, reinterpret_cast<ID3D11UnorderedAccessView**>(views));
    else
        ctx->OMGetRenderTargets(8, reinterpret_cast<ID3D11RenderTargetView**>(views), nullptr);
    int match = -1;
    ID3D11Texture2D* output = nullptr;
    for (int i = 0; i < 8; ++i)
    {
        if (views[i] == nullptr)
            continue;
        if (ID3D11Texture2D* texture = TextureFromView(views[i]); texture != nullptr)
        {
            D3D11_TEXTURE2D_DESC desc {};
            texture->GetDesc(&desc);
            if (match < 0 && desc.Width == g_computeMvDesc.Width && desc.Height == g_computeMvDesc.Height &&
                desc.Format == g_computeMvDesc.Format)
            {
                match = i;
                output = texture; // compared only
            }
            texture->Release();
        }
        views[i]->Release();
    }
    if (match < 0)
        return;

    const void* shader = nullptr;
    if (compute)
    {
        ID3D11ComputeShader* cs = nullptr;
        ctx->CSGetShader(&cs, nullptr, nullptr);
        shader = cs;
        if (cs != nullptr)
            cs->Release();
    }
    else
    {
        ID3D11PixelShader* ps = nullptr;
        ctx->PSGetShader(&ps, nullptr, nullptr);
        shader = ps;
        if (ps != nullptr)
            ps->Release();
    }
    const char* stage = compute ? "cs" : "ps";
    const auto note = std::format(" (motion writer {}, present {}: {} shader {:X}, {} {} = {:X})", g_mvDraws++,
                                  g_presentsInState, stage, (size_t) shader, compute ? "u" : "rt", match,
                                  (size_t) output);
    SnapConstantBuffers(ctx, g_frame, stage, note);
    if (!compute)
        SnapConstantBuffers(ctx, g_frame, "vs", note);
    if (shader == nullptr || std::any_of(g_mvShaderCode.begin(), g_mvShaderCode.end(),
                                         [shader](const auto& entry) { return entry.shader == shader; }))
        return;
    std::vector<char> code;
    {
        std::lock_guard lock(g_shaderMutex);
        if (auto it = g_shaderCode.find(shader); it != g_shaderCode.end())
            code = it->second;
    }
    g_mvShaderCode.push_back({ shader, stage, std::move(code) });
}

// Learning, dispatch mode: the texture behind the first R32_UINT compute SRV (slots 0-7) of the candidate pass.
void LearnComputeMv(ID3D11DeviceContext* ctx)
{
    g_frameComputeMvDesc = {};
    ID3D11ShaderResourceView* srvs[8] = {};
    ctx->CSGetShaderResources(0, 8, srvs);
    for (auto* view : srvs)
    {
        if (view == nullptr)
            continue;
        D3D11_SHADER_RESOURCE_VIEW_DESC viewDesc {};
        view->GetDesc(&viewDesc);
        if (viewDesc.Format == DXGI_FORMAT_R32_UINT && g_frameComputeMvDesc.Width == 0)
        {
            if (ID3D11Texture2D* texture = TextureFromView(view))
            {
                texture->GetDesc(&g_frameComputeMvDesc);
                texture->Release();
            }
        }
        view->Release();
    }
}
} // namespace

void FrameCapture_Dx11::RememberShader(const void* shader, const void* bytecode, SIZE_T length)
{
    if (shader == nullptr || bytecode == nullptr || length == 0)
        return;
    const auto* bytes = static_cast<const char*>(bytecode);
    std::lock_guard lock(g_shaderMutex);
    g_shaderCode[shader].assign(bytes, bytes + length);
}

bool FrameCapture_Dx11::ShaderCode(const void* shader, std::vector<char>& code)
{
    std::lock_guard lock(g_shaderMutex);
    const auto it = g_shaderCode.find(shader);
    if (it == g_shaderCode.end())
        return false;
    code = it->second;
    return true;
}

// Delayed / repeated capture: F7 can start a series that fires while both hands are on the controls (riding).
uint64_t g_pendingAtMs = 0;
int g_pendingLeft = 0;

void FrameCapture_Dx11::Arm()
{
    const int delay = Config::Instance()->CaptureDelaySeconds.value_or_default();
    const int repeat = std::max(1, Config::Instance()->CaptureRepeat.value_or_default());
    if (delay <= 0 && repeat <= 1)
    {
        g_armRequested = true;
        LOG_INFO("Frame capture: armed");
        return;
    }
    if (g_pendingLeft > 0)
    {
        g_pendingLeft = 0;
        LOG_INFO("Frame capture: the pending series was cancelled");
        return;
    }
    g_pendingLeft = repeat;
    g_pendingAtMs = GetTickCount64() + (uint64_t) std::max(0, delay) * 1000;
    LOG_INFO("Frame capture: {} capture(s) armed, the first in {} s, then every {} s (F7 again cancels)", repeat,
             delay, std::max(1, Config::Instance()->CaptureSpacingSeconds.value_or_default()));
}

bool FrameCapture_Dx11::Busy() { return g_state != CaptureState::Idle || g_armRequested.load(); }

void FrameCapture_Dx11::BeforeDispatch(ID3D11DeviceContext* ctx, UINT x, UINT y, UINT z)
{
    if (g_state == CaptureState::Idle || ctx == nullptr)
        return;
    if (g_drawPass)
    {
        if (g_state != CaptureState::Capturing || ctx != g_targetCtx || g_capturedThisFrame)
            return;
        ID3D11UnorderedAccessView* output = nullptr;
        ctx->CSGetUnorderedAccessViews(0, 1, &output);
        ID3D11Texture2D* texture = TextureFromView(output);
        if (output != nullptr)
            output->Release();
        ID3D11ComputeShader* shader = nullptr;
        ctx->CSGetShader(&shader, nullptr, nullptr);
        if (shader != nullptr)
            shader->Release();
        WatchMvWriter(ctx, texture, shader, true);
        if (texture != nullptr)
            texture->Release();
        return;
    }

    ID3D11UnorderedAccessView* uav = nullptr;
    ctx->CSGetUnorderedAccessViews(0, 1, &uav);
    ID3D11Texture2D* texture = TextureFromView(uav);
    if (uav != nullptr)
        uav->Release();
    if (g_state == CaptureState::Capturing && (texture == nullptr || (texture != g_targets[0] && texture != g_targets[1])))
        WatchComputeMvWriters(ctx, true);
    if (texture == nullptr)
        return;

    D3D11_TEXTURE2D_DESC desc {};
    texture->GetDesc(&desc);

    if (g_state == CaptureState::Learning)
    {
        if (IsEyeSize(desc.Width, desc.Height) && (g_targetCtx == nullptr || ctx == g_targetCtx))
        {
            // The last one before Present wins.
            if (g_frameTarget != nullptr)
                g_frameTarget->Release();
            texture->AddRef();
            g_frameTarget = texture;
            g_targetCtx = ctx;
            g_targetGroups[0] = x;
            g_targetGroups[1] = y;
            g_targetGroups[2] = z;
            LearnComputeMv(ctx);
        }
    }
    else if (g_state == CaptureState::Capturing && !g_capturedThisFrame && !g_pendingAfter && ctx == g_targetCtx &&
             (texture == g_targets[0] || texture == g_targets[1]))
    {
        g_frameOutput[g_frame] = texture;
        SnapConstantBuffers(ctx, g_frame);
        if (g_frame == 0)
        {
            ID3D11ComputeShader* shader = nullptr;
            ctx->CSGetShader(&shader, nullptr, nullptr);
            if (shader != nullptr)
            {
                g_capturedShader = shader;
                std::lock_guard lock(g_shaderMutex);
                if (auto it = g_shaderCode.find(shader); it != g_shaderCode.end())
                    g_capturedShaderCode = it->second;
                shader->Release();
            }
        }
        if (g_frame == 0)
        {
            SnapInputs(ctx);
            SnapUavs(ctx, "before");
            // [Inputs] TaaMotionCheck: its decoded and composed motion for this pass, just made, and its copy of the
            // pass's output from two Presents back (the same eye under alternate-eye VR).
            if (ID3D11Texture2D* motion = TaaMotion_Dx11::Decoded())
                Snap(ctx, motion, "decoded motion");
            if (ID3D11Texture2D* motion = TaaMotion_Dx11::Composed())
                Snap(ctx, motion, "composed motion");
            if (ID3D11Texture2D* output = TaaMotion_Dx11::OutputPresentsAgo(2))
                Snap(ctx, output, "output 2 presents ago");
            // What the composition read, to reproduce it offline.
            if (ID3D11Texture2D* colour = TaaMotion_Dx11::ColourPresentsAgo(2))
                Snap(ctx, colour, "colour 2 presents ago");
            if (ID3D11Texture2D* motion = TaaMotion_Dx11::PreviousDecoded())
                Snap(ctx, motion, "previous decoded motion");
            if (ID3D11Texture2D* depth = TaaMotion_Dx11::FrameDepth())
                Snap(ctx, depth, "frame depth");
            if (ID3D11Texture2D* depth = TaaMotion_Dx11::PreviousDepth())
                Snap(ctx, depth, "previous depth");
            g_pendingAfter = true;
        }
        else
        {
            g_capturedThisFrame = true;
        }
    }
    texture->Release();
}

void FrameCapture_Dx11::AfterDispatch(ID3D11DeviceContext* ctx)
{
    if (!g_pendingAfter || ctx != g_targetCtx || g_drawPass)
        return;
    SnapUavs(ctx, "after");
    g_pendingAfter = false;
    g_capturedThisFrame = true;
}

namespace
{
// RT 0 of the bound render targets if it's an eye-sized Texture2D (AddRef'd), else nullptr.
ID3D11Texture2D* EyeSizedTarget(ID3D11DeviceContext* ctx)
{
    ID3D11RenderTargetView* rtv = nullptr;
    ctx->OMGetRenderTargets(1, &rtv, nullptr);
    ID3D11Texture2D* texture = TextureFromView(rtv);
    if (rtv != nullptr)
        rtv->Release();
    if (texture == nullptr)
        return nullptr;
    D3D11_TEXTURE2D_DESC desc {};
    texture->GetDesc(&desc);
    if (!IsEyeSize(desc.Width, desc.Height))
    {
        texture->Release();
        return nullptr;
    }
    return texture;
}

void SnapTarget(ID3D11DeviceContext* ctx, const char* when)
{
    if (ID3D11Texture2D* target = EyeSizedTarget(ctx))
    {
        Snap(ctx, target, std::format("draw rt0 {}", when));
        target->Release();
    }
}

// Every bound render target (the TAA draw may write its history too).
void SnapTargets(ID3D11DeviceContext* ctx, const char* when)
{
    ID3D11RenderTargetView* rtvs[D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT] = {};
    ctx->OMGetRenderTargets(D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT, rtvs, nullptr);
    for (UINT i = 0; i < D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT; ++i)
    {
        if (rtvs[i] == nullptr)
            continue;
        if (ID3D11Texture2D* texture = TextureFromView(rtvs[i]))
        {
            Snap(ctx, texture, std::format("rt{} {}", i, when));
            texture->Release();
        }
        rtvs[i]->Release();
    }
}

// Learning the TAA draw: remembers which pixel shader drew each eye-sized target this Present; at the first draw into
// the eye target (render target only) that reads one of them, that texture's last writer is the TAA candidate.
void LearnTaaDraw(ID3D11DeviceContext* ctx)
{
    if (g_frameTaaShader != nullptr)
        return;
    ID3D11RenderTargetView* rtv = nullptr;
    ctx->OMGetRenderTargets(1, &rtv, nullptr);
    ID3D11Texture2D* target = TextureFromView(rtv);
    if (rtv != nullptr)
        rtv->Release();
    if (target == nullptr)
        return;
    D3D11_TEXTURE2D_DESC desc {};
    target->GetDesc(&desc);
    target->Release();
    if (!IsEyeSize(desc.Width, desc.Height))
        return;

    if ((desc.BindFlags & D3D11_BIND_SHADER_RESOURCE) == 0)
    {
        ID3D11ShaderResourceView* srvs[32] = {};
        ctx->PSGetShaderResources(0, 32, srvs);
        for (UINT i = 0; i < 32; ++i)
        {
            if (srvs[i] == nullptr)
                continue;
            ID3D11Texture2D* texture = TextureFromView(srvs[i]);
            srvs[i]->Release();
            if (texture == nullptr)
                continue;
            if (auto it = g_rtProducers.find(texture); it != g_rtProducers.end() && g_frameTaaShader == nullptr)
            {
                D3D11_TEXTURE2D_DESC sourceDesc {};
                texture->GetDesc(&sourceDesc);
                g_frameTaaShader = it->second;
                g_frameTaaCtx = ctx;
                g_frameTaaSize[0] = sourceDesc.Width;
                g_frameTaaSize[1] = sourceDesc.Height;
            }
            texture->Release();
        }
        return;
    }

    ID3D11PixelShader* shader = nullptr;
    ctx->PSGetShader(&shader, nullptr, nullptr);
    if (shader == nullptr)
        return;
    shader->Release();
    g_rtProducers[target] = shader; // compared only, never dereferenced
}

// Capturing the TAA draw. Before frame 0's TAA draw, keeps the last depth bound with a target of the TAA size.
void CaptureTaaDraw(ID3D11DeviceContext* ctx)
{
    if (ctx != g_targetCtx || g_capturedThisFrame || g_pendingAfter)
        return;
    ID3D11RenderTargetView* rtv = nullptr;
    ID3D11DepthStencilView* dsv = nullptr;
    ctx->OMGetRenderTargets(1, &rtv, &dsv);
    ID3D11Texture2D* target = TextureFromView(rtv);
    ID3D11Texture2D* depth = TextureFromView(dsv);
    if (rtv != nullptr)
        rtv->Release();
    if (dsv != nullptr)
        dsv->Release();

    D3D11_TEXTURE2D_DESC desc {};
    if (target != nullptr)
        target->GetDesc(&desc);
    const bool taaSize = target != nullptr && desc.Width == g_taaSize[0] && desc.Height == g_taaSize[1];

    ID3D11PixelShader* shader = nullptr;
    ctx->PSGetShader(&shader, nullptr, nullptr);
    if (shader != nullptr)
        shader->Release();

    if (!(taaSize && shader == g_taaShader))
        WatchMvWriter(ctx, target, shader, false);

    if (!g_gbufferDone && taaSize && depth != nullptr)
    {
        ID3D11RenderTargetView* rtvs[D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT] = {};
        ctx->OMGetRenderTargets(D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT, rtvs, nullptr);
        int count = 0;
        for (auto* view : rtvs)
        {
            if (view != nullptr)
            {
                ++count;
                view->Release();
            }
        }
        if (count >= 3)
        {
            g_gbufferDone = true;
            ID3D11VertexShader* vertex = nullptr;
            ctx->VSGetShader(&vertex, nullptr, nullptr);
            if (vertex != nullptr)
                vertex->Release();
            SnapConstantBuffers(ctx, g_frame, "vs",
                                std::format(" (first G-buffer draw, {} targets, vs shader {:X})", count, (size_t) vertex));
            const bool seen = std::any_of(g_gbufferShaders.begin(), g_gbufferShaders.end(),
                                          [vertex](const auto& entry) { return entry.shader == vertex; });
            if (vertex != nullptr && !seen)
            {
                std::vector<char> code;
                {
                    std::lock_guard lock(g_shaderMutex);
                    if (auto it = g_shaderCode.find(vertex); it != g_shaderCode.end())
                        code = it->second;
                }
                g_gbufferShaders.push_back({ vertex, "vs", std::move(code) });
            }
        }
    }

    if (taaSize && shader == g_taaShader)
    {
        g_frameOutput[g_frame] = target;
        g_framePresent[g_frame] = g_presentsInState;
        LARGE_INTEGER now {};
        QueryPerformanceCounter(&now);
        g_frameTicks[g_frame] = now.QuadPart;
        SnapConstantBuffers(ctx, g_frame, "ps");
        SnapConstantBuffers(ctx, g_frame, "vs");
        if (StaleEyeMode())
        {
            if (g_frame == 0)
            {
                g_capturedShader = shader;
                {
                    std::lock_guard lock(g_shaderMutex);
                    if (auto it = g_shaderCode.find(shader); it != g_shaderCode.end())
                        g_capturedShaderCode = it->second;
                }
                // The motion pass behind t0, as below: its cb0 holds CurrentToPrevMatrix, saved from frame 1 on.
                ID3D11ShaderResourceView* motionView = nullptr;
                ctx->PSGetShaderResources(0, 1, &motionView);
                if (ID3D11Texture2D* motion = TextureFromView(motionView))
                {
                    if (auto it = g_frameWriters.find(motion); it != g_frameWriters.end())
                        g_mvShaders = it->second;
                    motion->GetDesc(&g_mvDesc);
                    motion->Release();
                }
                if (motionView != nullptr)
                    motionView->Release();
                LOG_INFO("Frame capture: stale-eye, {} shaders wrote the TAA's t0 (motion vectors) this Present",
                         g_mvShaders.size());
            }
            // Copied, then let go at once: no game resource held across a Present (a resize needs them all freed).
            if (g_sceneDepth != nullptr)
            {
                Snap(ctx, g_sceneDepth, std::format("f{} scene depth", g_frame));
                g_sceneDepth->Release();
                g_sceneDepth = nullptr;
            }
            else
                LOG_WARN("Frame capture: stale-eye frame {} has no scene depth", g_frame);
            g_pendingAfter = true; // AfterDraw saves the TAA output and marks the frame captured
        }
        else if (g_frame == 0)
        {
            g_capturedShader = shader;
            {
                std::lock_guard lock(g_shaderMutex);
                if (auto it = g_shaderCode.find(shader); it != g_shaderCode.end())
                    g_capturedShaderCode = it->second;
            }
            SnapInputs(ctx, true);
            SnapTargets(ctx, "before");
            ID3D11ShaderResourceView* motionView = nullptr;
            ctx->PSGetShaderResources(0, 1, &motionView);
            if (ID3D11Texture2D* motion = TextureFromView(motionView))
            {
                if (auto it = g_frameWriters.find(motion); it != g_frameWriters.end())
                    g_mvShaders = it->second;
                motion->GetDesc(&g_mvDesc);
                motion->Release();
            }
            if (motionView != nullptr)
                motionView->Release();
            LOG_INFO("Frame capture: {} shaders wrote the TAA's t0 (motion vectors) this Present",
                     g_mvShaders.size());
            if (g_sceneDepth != nullptr)
                Snap(ctx, g_sceneDepth, "scene depth");
            g_pendingAfter = true;
        }
        else
        {
            g_capturedThisFrame = true;
        }
    }
    else if (taaSize && depth != nullptr && (g_frame == 0 || StaleEyeMode()))
    {
        if (g_sceneDepth != nullptr)
            g_sceneDepth->Release();
        depth->AddRef();
        g_sceneDepth = depth;
    }

    if (target != nullptr)
        target->Release();
    if (depth != nullptr)
        depth->Release();
}
} // namespace

void FrameCapture_Dx11::BeforeDraw(ID3D11DeviceContext* ctx)
{
    if (g_state == CaptureState::Idle || ctx == nullptr)
        return;
    if (g_state == CaptureState::Learning)
        LearnTaaDraw(ctx);
    else if (g_drawPass)
    {
        CaptureTaaDraw(ctx);
        return;
    }
    else
    {
        WatchComputeMvWriters(ctx, false);
    }
    if (ctx != g_targetCtx)
        return;
    const bool learning = g_state == CaptureState::Learning;
    // Frame 0, once the pass has been captured this Present.
    if (!learning && (g_frame != 0 || !g_capturedThisFrame || g_drawDone || g_pendingAfterDraw))
        return;

    ID3D11Texture2D* target = EyeSizedTarget(ctx);
    if (target == nullptr)
        return;
    target->Release();
    ID3D11PixelShader* shader = nullptr;
    ctx->PSGetShader(&shader, nullptr, nullptr);
    if (shader == nullptr)
        return;
    shader->Release();

    if (learning)
    {
        g_frameDrawShader = shader; // the last one before Present wins
        return;
    }
    if (shader != g_drawShader)
        return;
    {
        std::lock_guard lock(g_shaderMutex);
        if (auto it = g_shaderCode.find(shader); it != g_shaderCode.end())
            g_drawShaderCode = it->second;
    }
    SnapConstantBuffers(ctx, 0, "ps");
    SnapInputs(ctx, true);
    SnapTarget(ctx, "before");
    g_pendingAfterDraw = true;
}

void FrameCapture_Dx11::AfterDraw(ID3D11DeviceContext* ctx)
{
    if (g_drawPass)
    {
        if (!g_pendingAfter || ctx != g_targetCtx)
            return;
        SnapTargets(ctx, StaleEyeMode() ? std::format("after f{}", g_frame).c_str() : "after");
        g_pendingAfter = false;
        g_capturedThisFrame = true;
        return;
    }
    if (!g_pendingAfterDraw || ctx != g_targetCtx)
        return;
    SnapTarget(ctx, "after");
    g_pendingAfterDraw = false;
    g_drawDone = true;
}

void FrameCapture_Dx11::OnPresent()
{
    switch (g_state)
    {
    case CaptureState::Idle:
        if (g_armRequested.exchange(false))
        {
            g_state = CaptureState::Learning;
            g_presentsInState = 0;
            g_pendingLeft = 0;
        }
        else if (g_pendingLeft > 0 && GetTickCount64() >= g_pendingAtMs)
        {
            --g_pendingLeft;
            g_pendingAtMs =
                GetTickCount64() +
                (uint64_t) std::max(1, Config::Instance()->CaptureSpacingSeconds.value_or_default()) * 1000;
            g_state = CaptureState::Learning;
            g_presentsInState = 0;
            LOG_INFO("Frame capture: starting a capture from the armed series ({} left after this one)",
                     g_pendingLeft);
        }
        return;

    case CaptureState::Learning:
        if (g_frameTarget != nullptr)
        {
            if (g_frameTarget != g_targets[0] && g_frameTarget != g_targets[1])
            {
                auto& slot = g_targets[0] == nullptr ? g_targets[0] : g_targets[1];
                if (slot != nullptr)
                    slot->Release();
                slot = g_frameTarget;
            }
            else
            {
                g_frameTarget->Release();
            }
            g_frameTarget = nullptr;
        }
        if (g_frameDrawShader != nullptr)
            g_drawShader = g_frameDrawShader;
        g_frameDrawShader = nullptr;
        if (g_frameTaaShader != nullptr)
        {
            g_taaShader = g_frameTaaShader;
            g_taaCtx = g_frameTaaCtx;
            g_taaSize[0] = g_frameTaaSize[0];
            g_taaSize[1] = g_frameTaaSize[1];
        }
        g_frameTaaShader = nullptr;
        g_rtProducers.clear();
        // FC5 is all draws: its TAA draw is the pass there even if an eye-sized dispatch was seen.
        if (++g_presentsInState >= kLearnPresents && g_taaShader != nullptr &&
            (g_targets[0] == nullptr || Fc5Dlss_Dx11::Enabled() || Fc5Taa_Dx11::Enabled()))
        {
            g_drawPass = true;
            g_targetCtx = g_taaCtx;
            g_drawShader = nullptr;
            LOG_INFO("Frame capture: TAA draw found, pixel shader {:X} into {}x{} on context {:X}",
                     (size_t) g_taaShader, g_taaSize[0], g_taaSize[1], (size_t) g_targetCtx);
            g_state = CaptureState::Capturing;
            g_presentsInState = 0;
        }
        else if (g_presentsInState >= kLearnPresents && g_targets[0] != nullptr)
        {
            D3D11_TEXTURE2D_DESC desc {};
            g_targets[0]->GetDesc(&desc);
            LOG_INFO("Frame capture: pass found, dispatch {}x{}x{} writing {}x{} fmt {} ({:X}, {:X}) on context {:X}",
                     g_targetGroups[0], g_targetGroups[1], g_targetGroups[2], desc.Width, desc.Height,
                     (UINT) desc.Format, (size_t) g_targets[0], (size_t) g_targets[1], (size_t) g_targetCtx);
            LOG_INFO("Frame capture: last eye-sized draw before Present: pixel shader {:X}", (size_t) g_drawShader);
            g_computeMvDesc = g_frameComputeMvDesc;
            LOG_INFO("Frame capture: pass motion input {}x{} fmt {}", g_computeMvDesc.Width, g_computeMvDesc.Height,
                     (UINT) g_computeMvDesc.Format);
            g_state = CaptureState::Capturing;
            g_presentsInState = 0;
        }
        else if (g_presentsInState >= kMaxPresentsPerState)
        {
            LOG_WARN("Frame capture: no eye-sized compute pass or TAA draw in {} Presents; nothing captured", kMaxPresentsPerState);
            Reset();
        }
        return;

    case CaptureState::Capturing:
        g_frameWriters.clear();
        g_mvDraws = 0;
        g_gbufferDone = false;
        if (g_capturedThisFrame)
        {
            g_capturedThisFrame = false;
            ++g_frame;
        }
        ++g_presentsInState;
        if (g_frame >= CaptureFrames() || (g_presentsInState >= kMaxPresentsPerState && g_frame > 0))
        {
            WriteCapture();
            Reset();
        }
        else if (g_presentsInState >= kMaxPresentsPerState)
        {
            LOG_WARN("Frame capture: the pass didn't run again in {} Presents; nothing captured",
                     kMaxPresentsPerState);
            Reset();
        }
        return;
    }
}
