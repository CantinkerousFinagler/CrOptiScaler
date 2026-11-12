#include "pch.h"

#include "Config.h"
#include "Logger.h"
#include "Util.h"

#include "MotionMatrixProbe_Dx11.h"

#include <d3d11.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <map>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace
{
constexpr uint64_t kPresents = 16;
constexpr int kDrawsPerContextPresent = 4;
constexpr UINT kMaxCbufferBytes = 64 * 1024;

std::mutex g_mutex;
std::atomic<bool> g_capturing { false };
std::atomic<bool> g_armed { false };
std::atomic<uint64_t> g_present { 0 };
uint64_t g_captureEnd = 0;
std::atomic<ID3D11Resource*> g_motion { nullptr }; // identity only, never dereferenced for its contents
FILE* g_file = nullptr;
std::string g_path;

struct Written
{
    std::vector<uint8_t> bytes;
    uint64_t present = 0;
    void* ctx = nullptr;
};
// Constant buffer contents as the game last wrote them, per context and on any context.
std::map<std::pair<void*, void*>, void*> g_mapped;
std::map<std::pair<void*, void*>, Written> g_written;
std::map<void*, Written> g_latestAny;
// Contexts whose bound render targets include the motion texture (slot + 1).
std::map<void*, int> g_boundMv;
std::map<std::pair<void*, uint64_t>, int> g_drawsLogged;

struct Counters
{
    uint64_t mvBinds = 0, mvDraws = 0, drawsLogged = 0, cbufsFound = 0, cbufsFoundOtherCtx = 0, cbufsMissing = 0,
             mapsRecorded = 0, updatesRecorded = 0, sampleFrames = 0, stripFrames = 0;
};
Counters g_counters;

bool IsConstantBuffer(ID3D11Resource* resource, UINT* size)
{
    if (resource == nullptr)
        return false;
    D3D11_RESOURCE_DIMENSION dim = D3D11_RESOURCE_DIMENSION_UNKNOWN;
    resource->GetType(&dim);
    if (dim != D3D11_RESOURCE_DIMENSION_BUFFER)
        return false;
    D3D11_BUFFER_DESC desc {};
    static_cast<ID3D11Buffer*>(resource)->GetDesc(&desc);
    if ((desc.BindFlags & D3D11_BIND_CONSTANT_BUFFER) == 0 || desc.ByteWidth > kMaxCbufferBytes)
        return false;
    *size = desc.ByteWidth;
    return true;
}

void WriteFloats(const uint8_t* bytes, size_t size)
{
    const size_t floats = size / 4;
    for (size_t i = 0; i < floats; i += 16)
    {
        std::fprintf(g_file, "    +%04zx:", i * 4);
        for (size_t j = i; j < std::min(floats, i + 16); ++j)
        {
            float f;
            std::memcpy(&f, bytes + j * 4, 4);
            std::fprintf(g_file, "%s %.7g", (j - i) % 4 == 0 ? "  |" : "", f);
        }
        std::fprintf(g_file, "\n");
    }
}

void DumpStage(ID3D11DeviceContext* ctx, const char* stage, ID3D11Buffer* const* buffers)
{
    for (UINT slot = 0; slot < D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT; ++slot)
    {
        ID3D11Buffer* buffer = buffers[slot];
        if (buffer == nullptr)
            continue;
        D3D11_BUFFER_DESC desc {};
        buffer->GetDesc(&desc);
        const Written* written = nullptr;
        bool otherCtx = false;
        if (auto it = g_written.find({ ctx, buffer }); it != g_written.end())
            written = &it->second;
        else if (auto any = g_latestAny.find(buffer); any != g_latestAny.end())
        {
            written = &any->second;
            otherCtx = true;
        }
        if (written == nullptr)
        {
            ++g_counters.cbufsMissing;
            std::fprintf(g_file, "  %s cb%u %p size %u usage %d: NOT WRITTEN since the capture was armed\n", stage, slot,
                         (void*) buffer, desc.ByteWidth, (int) desc.Usage);
            continue;
        }
        ++(otherCtx ? g_counters.cbufsFoundOtherCtx : g_counters.cbufsFound);
        std::fprintf(g_file, "  %s cb%u %p size %u usage %d: written at present %llu on %s ctx %p\n", stage, slot,
                     (void*) buffer, desc.ByteWidth, (int) desc.Usage, (unsigned long long) written->present,
                     otherCtx ? "ANOTHER" : "this", written->ctx);
        WriteFloats(written->bytes.data(), written->bytes.size());
    }
}

void Finish()
{
    const auto& c = g_counters;
    LOG_INFO("Matrix probe: capture done -> {}: {} binds of the motion texture, {} draws into it ({} dumped); constant "
             "buffers: {} from the same context, {} from another context, {} NOT written during the capture; {} Maps "
             "and {} UpdateSubresource of constant buffers recorded; {} sample frames, {} frame strips so far (late ones "
             "follow into the _frames folder)",
             g_path, c.mvBinds, c.mvDraws, c.drawsLogged, c.cbufsFound, c.cbufsFoundOtherCtx, c.cbufsMissing,
             c.mapsRecorded, c.updatesRecorded, c.sampleFrames, c.stripFrames);
    if (g_motion.load() == nullptr)
        LOG_WARN("Matrix probe: the motion texture was never learned (no DLSS evaluate before or during the capture)");
    else if (c.mvBinds == 0)
        LOG_WARN("Matrix probe: the motion texture was never bound as a render target -- written another way (copy, "
                 "compute UAV, or a bind these hooks do not see)");
    if (g_file != nullptr)
    {
        std::fclose(g_file);
        g_file = nullptr;
    }
    g_mapped.clear();
    g_written.clear();
    g_latestAny.clear();
    g_boundMv.clear();
    g_drawsLogged.clear();
}
} // namespace

namespace MotionMatrixProbe_Dx11
{
bool Enabled()
{
    static const bool enabled = Config::Instance()->UpscalerMatrixProbe.value_or_default();
    return enabled;
}

void Arm()
{
    if (!Enabled())
        return;
    std::lock_guard lock(g_mutex);
    if (g_capturing.load() || g_armed.load())
    {
        LOG_INFO("Matrix probe: F7 ignored, a capture is already running");
        return;
    }
    const auto dir = Util::ExePath().parent_path() / L"MatrixProbe";
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    const auto now = std::chrono::system_clock::now();
    const auto seconds = std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count();
    const auto file = dir / std::format(L"probe-{}.txt", seconds);
    g_path = wstring_to_string(file.wstring());
    if (_wfopen_s(&g_file, file.c_str(), L"w") != 0 || g_file == nullptr)
    {
        LOG_ERROR("Matrix probe: can't open {}", g_path);
        g_file = nullptr;
        return;
    }
    std::fprintf(g_file, "# Matrix probe. Draws into the DLSS motion texture with their VS/PS constant buffers (floats, "
                         "16 per line, offsets in bytes); per evaluate a 32x32 grid: col row depth, the game's mv and "
                         "the mv DLSS got (render px, DLSS convention = previous - current); per-eye fits (reason 0 "
                         "accepted, 1 solve failed, 2 few depth samples, 3 few within 1 px; z scaled by 100).\n");
    g_counters = Counters {};
    g_armed = true;
    LOG_INFO("Matrix probe: F7, armed; capturing the next {} Presents to {}", kPresents, g_path);
}

void OnPresent()
{
    if (!Enabled())
        return;
    const uint64_t present = ++g_present;
    if (!g_armed.load() && !g_capturing.load())
        return;
    std::lock_guard lock(g_mutex);
    if (g_armed.load())
    {
        g_armed = false;
        g_capturing = true;
        g_captureEnd = present + kPresents;
        std::fprintf(g_file, "# capture from present %llu to %llu\n", (unsigned long long) present,
                     (unsigned long long) g_captureEnd);
        return;
    }
    if (g_file != nullptr)
        std::fprintf(g_file, "present %llu ends (parity %llu)\n", (unsigned long long) present - 1,
                     (unsigned long long) ((present - 1) & 1));
    if (present >= g_captureEnd)
    {
        g_capturing = false;
        Finish();
    }
}

bool Capturing() { return g_capturing.load(std::memory_order_relaxed); }

uint64_t Present() { return g_present.load(); }

void SetMotionTexture(ID3D11Resource* motion)
{
    if (Enabled())
        g_motion = motion;
}

void OnBind(ID3D11DeviceContext* ctx, unsigned int numRtvs, ID3D11RenderTargetView* const* rtvs)
{
    if (!Capturing())
        return;
    ID3D11Resource* motion = g_motion.load();
    if (motion == nullptr)
        return;
    int slot = 0;
    for (unsigned int i = 0; rtvs != nullptr && i < numRtvs && slot == 0; ++i)
    {
        if (rtvs[i] == nullptr)
            continue;
        ID3D11Resource* resource = nullptr;
        rtvs[i]->GetResource(&resource);
        if (resource == motion)
            slot = (int) i + 1;
        if (resource != nullptr)
            resource->Release();
    }
    std::lock_guard lock(g_mutex);
    if (slot == 0)
    {
        g_boundMv.erase(ctx);
        return;
    }
    g_boundMv[ctx] = slot;
    ++g_counters.mvBinds;
    if (g_file != nullptr)
    {
        std::fprintf(g_file, "bind present %llu ctx %p (%s): motion texture at RT slot %d of %u; slots:",
                     (unsigned long long) g_present.load(), (void*) ctx,
                     ctx->GetType() == D3D11_DEVICE_CONTEXT_DEFERRED ? "deferred" : "immediate", slot - 1, numRtvs);
        for (unsigned int i = 0; i < numRtvs; ++i)
        {
            D3D11_RENDER_TARGET_VIEW_DESC desc {};
            if (rtvs[i] != nullptr)
                rtvs[i]->GetDesc(&desc);
            std::fprintf(g_file, " %d", rtvs[i] != nullptr ? (int) desc.Format : -1);
        }
        std::fprintf(g_file, "\n");
    }
}

void OnDraw(ID3D11DeviceContext* ctx)
{
    if (!Capturing())
        return;
    std::lock_guard lock(g_mutex);
    auto bound = g_boundMv.find(ctx);
    if (bound == g_boundMv.end() || g_file == nullptr)
        return;
    ++g_counters.mvDraws;
    const uint64_t present = g_present.load();
    int& logged = g_drawsLogged[{ ctx, present }];
    if (logged >= kDrawsPerContextPresent)
        return;
    ++logged;
    ++g_counters.drawsLogged;

    ID3D11VertexShader* vs = nullptr;
    ID3D11PixelShader* ps = nullptr;
    ctx->VSGetShader(&vs, nullptr, nullptr);
    ctx->PSGetShader(&ps, nullptr, nullptr);
    D3D11_VIEWPORT viewport {};
    UINT viewports = 1;
    ctx->RSGetViewports(&viewports, &viewport);
    std::fprintf(g_file, "draw present %llu ctx %p #%d: vs %p ps %p viewport %.0fx%.0f at (%.0f, %.0f)\n",
                 (unsigned long long) present, (void*) ctx, logged, (void*) vs, (void*) ps, viewport.Width,
                 viewport.Height, viewport.TopLeftX, viewport.TopLeftY);
    if (vs != nullptr)
        vs->Release();
    if (ps != nullptr)
        ps->Release();

    ID3D11Buffer* buffers[D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT] = {};
    ctx->VSGetConstantBuffers(0, D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT, buffers);
    DumpStage(ctx, "vs", buffers);
    for (auto& b : buffers)
        if (b != nullptr)
        {
            b->Release();
            b = nullptr;
        }
    ctx->PSGetConstantBuffers(0, D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT, buffers);
    DumpStage(ctx, "ps", buffers);
    for (auto& b : buffers)
        if (b != nullptr)
            b->Release();
}

void OnMap(ID3D11DeviceContext* ctx, ID3D11Resource* resource, unsigned int subresource, int mapType, void* data)
{
    if ((!Capturing() && !g_armed.load(std::memory_order_relaxed)) || data == nullptr || subresource != 0)
        return;
    if (mapType != D3D11_MAP_WRITE_DISCARD && mapType != D3D11_MAP_WRITE_NO_OVERWRITE && mapType != D3D11_MAP_WRITE)
        return;
    UINT size = 0;
    if (!IsConstantBuffer(resource, &size))
        return;
    std::lock_guard lock(g_mutex);
    g_mapped[{ ctx, resource }] = data;
}

void BeforeUnmap(ID3D11DeviceContext* ctx, ID3D11Resource* resource, unsigned int subresource)
{
    if ((!Capturing() && !g_armed.load(std::memory_order_relaxed)) || subresource != 0)
        return;
    std::lock_guard lock(g_mutex);
    auto it = g_mapped.find({ ctx, resource });
    if (it == g_mapped.end())
        return;
    UINT size = 0;
    if (IsConstantBuffer(resource, &size))
    {
        // The game's bytes, read before its own Unmap while the mapping is still valid.
        Written w;
        w.bytes.assign((const uint8_t*) it->second, (const uint8_t*) it->second + size);
        w.present = g_present.load();
        w.ctx = ctx;
        g_latestAny[resource] = w;
        g_written[{ ctx, resource }] = std::move(w);
        ++g_counters.mapsRecorded;
    }
    g_mapped.erase(it);
}

void OnUpdateSubresource(ID3D11DeviceContext* ctx, ID3D11Resource* resource, unsigned int subresource,
                         const void* box, const void* data)
{
    if ((!Capturing() && !g_armed.load(std::memory_order_relaxed)) || data == nullptr || subresource != 0 ||
        box != nullptr)
        return;
    UINT size = 0;
    if (!IsConstantBuffer(resource, &size))
        return;
    std::lock_guard lock(g_mutex);
    Written w;
    w.bytes.assign((const uint8_t*) data, (const uint8_t*) data + size);
    w.present = g_present.load();
    w.ctx = ctx;
    g_latestAny[resource] = w;
    g_written[{ ctx, resource }] = std::move(w);
    ++g_counters.updatesRecorded;
}

void WriteSamples(uint64_t present, int parity, const float* samples, float mvScaleX, float mvScaleY, float jitterX,
                  float jitterY, unsigned int renderW, unsigned int renderH)
{
    std::lock_guard lock(g_mutex);
    if (g_file == nullptr)
        return;
    ++g_counters.sampleFrames;
    std::fprintf(g_file,
                 "samples present %llu parity %d render %ux%u mvscale %.7g %.7g jitter %.7g %.7g (col row depth game_mvx "
                 "game_mvy dlss_mvx dlss_mvy)\n",
                 (unsigned long long) present, parity, renderW, renderH, mvScaleX, mvScaleY, jitterX, jitterY);
    for (int i = 0; i < 1024; ++i)
        std::fprintf(g_file, "  %d %d %.9g %.5g %.5g %.5g %.5g\n", i % 32, i / 32, samples[i * 8], samples[i * 8 + 1],
                     samples[i * 8 + 2], samples[i * 8 + 4], samples[i * 8 + 5]);
}

void WriteStrip(uint64_t present, int parity, int viewMode, bool perEye, const uint32_t* pixels)
{
    std::lock_guard lock(g_mutex);
    if (g_path.empty())
        return;
    // Beside the text file: <probe>_frames\present_<n>_p<parity>.bmp (256x256, 24-bit, bottom-up).
    std::filesystem::path dir(g_path);
    dir.replace_extension();
    dir += L"_frames";
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    const auto file = dir / std::format(L"present_{}_p{}.bmp", present, parity);
    FILE* f = nullptr;
    double luma = 0;
    if (_wfopen_s(&f, file.c_str(), L"wb") == 0 && f != nullptr)
    {
        const uint32_t w = 256, h = 256, row = w * 3, size = 54 + row * h;
        uint8_t header[54] = { 'B', 'M' };
        auto put32 = [&](int at, uint32_t v) { std::memcpy(header + at, &v, 4); };
        put32(2, size);
        put32(10, 54);
        put32(14, 40);
        put32(18, w);
        put32(22, h);
        header[26] = 1;
        header[28] = 24;
        put32(34, row * h);
        std::fwrite(header, 1, 54, f);
        std::vector<uint8_t> line(row);
        for (int y = (int) h - 1; y >= 0; --y)
        {
            for (uint32_t x = 0; x < w; ++x)
            {
                const uint32_t c = pixels[y * w + x];
                const uint8_t r = c & 0xFF, g = (c >> 8) & 0xFF, b = (c >> 16) & 0xFF;
                line[x * 3 + 0] = b;
                line[x * 3 + 1] = g;
                line[x * 3 + 2] = r;
                luma += 0.299 * r + 0.587 * g + 0.114 * b;
            }
            std::fwrite(line.data(), 1, row, f);
        }
        std::fclose(f);
        luma /= (double) (w * h);
    }
    if (g_file != nullptr)
        std::fprintf(g_file, "strip present %llu parity %d %s view %d mean luma %.2f\n", (unsigned long long) present,
                     parity, perEye ? "per-eye" : "shared", viewMode, luma);
    ++g_counters.stripFrames;
}

void WriteFit(uint64_t present, int slot, const float* fit, float streak)
{
    std::lock_guard lock(g_mutex);
    if (g_file == nullptr)
        return;
    std::fprintf(g_file,
                 "fit present %llu slot %d reason %d streak %g residual %.5g inliers %.4g n %g x %.8g %.8g %.8g %.8g y "
                 "%.8g %.8g %.8g %.8g w %.8g %.8g %.8g %.8g\n",
                 (unsigned long long) present, slot, (int) fit[16], streak, fit[12], fit[13], fit[15], fit[0], fit[1],
                 fit[2], fit[3], fit[4], fit[5], fit[6], fit[7], fit[8], fit[9], fit[10], fit[11]);
}
} // namespace MotionMatrixProbe_Dx11
