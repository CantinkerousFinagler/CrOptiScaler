#include "pch.h"

#include "EvalCapture_Dx12.h"

#include "Config.h"
#include "Logger.h"
#include "NVNGX_Parameter.h"
#include "Util.h"

#include <include/d3dx/d3dx12.h>

#include <atomic>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <format>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace
{
constexpr int kFrames = 8;
constexpr int kFramesWithGuides = 2; // motion and depth: for masks, not for the noise
constexpr uint64_t kByteLimit = 2500ull << 20;
constexpr uint64_t kDrainPresents = 3;
constexpr int kSuperSampling = 1;
constexpr int kRayReconstruction = 13;

enum Phase : int
{
    Idle,
    Requested,  // F7 pressed: the next evaluate plans the capture
    Allocating, // the worker is making the readback buffers
    Waiting,    // [Inputs] CaptureDelaySeconds
    Capturing,
    Draining,   // the last copies are on their way to the GPU
    Writing
};

struct Item
{
    int frame = 0;
    const char* kind = "";
    DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint {};
    UINT rows = 0;
    UINT64 rowBytes = 0;
    UINT64 total = 0;
    ID3D12Resource* readback = nullptr;
    bool recorded = false;
};

std::atomic<int> g_phase { Idle };
std::atomic<uint64_t> g_present { 0 };
std::mutex g_mutex; // g_items, g_frameInfo and the fields below while a frame is recorded
std::vector<Item> g_items;
std::vector<std::string> g_frameInfo;
int g_frame = 0;
uint64_t g_lastPresentTaken = UINT64_MAX;
uint64_t g_startAtTick = 0;
uint64_t g_drainFrom = 0;
ID3D12Resource* g_outputThisEvaluate = nullptr; // valid inside one evaluate only; never held past it
ID3D12Fence* g_fence = nullptr;
HANDLE g_fenceEvent = nullptr;
std::string g_status;

UINT g_boxW = 0, g_boxH = 0;
UINT g_colourX = 0, g_colourY = 0, g_motionX = 0, g_motionY = 0;
DXGI_FORMAT g_colourFormat = DXGI_FORMAT_UNKNOWN, g_outputFormat = DXGI_FORMAT_UNKNOWN;

ID3D12Resource* GetResource(NVSDK_NGX_Parameter* params, const char* name)
{
    ID3D12Resource* resource = nullptr;
    if (params->Get(name, &resource) != NVSDK_NGX_Result_Success)
        params->Get(name, (void**) &resource);
    return resource;
}

float GetFloat(NVSDK_NGX_Parameter* params, const char* name)
{
    float value = 0.0f;
    params->Get(name, &value);
    return value;
}

unsigned int GetUInt(NVSDK_NGX_Parameter* params, const char* name)
{
    unsigned int value = 0;
    params->Get(name, &value);
    return value;
}

void SetStatus(const std::string& text)
{
    std::lock_guard lock(g_mutex);
    g_status = text;
}

// What a reader of the .dds should take the bytes as: the footprint of plane 0 is what was copied.
DXGI_FORMAT ReadableFormat(DXGI_FORMAT format)
{
    switch (format)
    {
    case DXGI_FORMAT_R32G8X24_TYPELESS:
    case DXGI_FORMAT_D32_FLOAT_S8X24_UINT:
    case DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS:
    case DXGI_FORMAT_R32_TYPELESS:
    case DXGI_FORMAT_D32_FLOAT:
        return DXGI_FORMAT_R32_FLOAT;
    case DXGI_FORMAT_R16G16B16A16_TYPELESS:
        return DXGI_FORMAT_R16G16B16A16_FLOAT;
    case DXGI_FORMAT_R32G32_TYPELESS:
        return DXGI_FORMAT_R32G32_FLOAT;
    case DXGI_FORMAT_R16G16_TYPELESS:
        return DXGI_FORMAT_R16G16_FLOAT;
    default:
        return format;
    }
}

bool AddItem(ID3D12Device* device, std::vector<Item>& items, int frame, const char* kind, D3D12_RESOURCE_DESC desc,
             UINT width, UINT height)
{
    desc.Width = width;
    desc.Height = height;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    Item item;
    item.frame = frame;
    item.kind = kind;
    item.format = ReadableFormat(desc.Format);
    device->GetCopyableFootprints(&desc, 0, 1, 0, &item.footprint, &item.rows, &item.rowBytes, &item.total);
    if (item.total == 0 || item.total == UINT64_MAX)
    {
        LOG_WARN("Upscaler capture: no footprint for {} ({}x{} format {})", kind, width, height, (int) desc.Format);
        return false;
    }
    items.push_back(item);
    return true;
}

void ReleaseAll()
{
    for (auto& item : g_items)
    {
        if (item.readback != nullptr)
            item.readback->Release();
        item.readback = nullptr;
    }
    g_items.clear();
}

void AllocateOnWorker(ID3D12Device* device, std::vector<Item> items)
{
    uint64_t bytes = 0;
    bool ok = true;
    for (auto& item : items)
    {
        D3D12_HEAP_PROPERTIES heap = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK);
        auto bufferDesc = CD3DX12_RESOURCE_DESC::Buffer(item.total);
        if (FAILED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &bufferDesc,
                                                   D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                   IID_PPV_ARGS(&item.readback))) ||
            item.readback == nullptr)
        {
            LOG_WARN("Upscaler capture: readback buffer of {} MB for {} failed", item.total >> 20, item.kind);
            ok = false;
            break;
        }
        bytes += item.total;
    }
    if (ok && g_fence == nullptr)
    {
        ok = SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&g_fence)));
        if (ok)
            g_fenceEvent = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        ok = ok && g_fenceEvent != nullptr;
        if (!ok)
            LOG_WARN("Upscaler capture: no fence");
    }
    device->Release();

    std::lock_guard lock(g_mutex);
    g_items = std::move(items);
    if (!ok)
    {
        ReleaseAll();
        g_status = "Upscaler capture: failed making its buffers (see the log)";
        g_phase.store(Idle);
        return;
    }
    g_frame = 0;
    g_frameInfo.clear();
    g_lastPresentTaken = UINT64_MAX;
    const int delay = Config::Instance()->CaptureDelaySeconds.value_or_default();
    g_startAtTick = GetTickCount64() + (uint64_t) (delay > 0 ? delay : 0) * 1000;
    g_status = std::format("Upscaler capture: starting in {} s, hold still", delay);
    LOG_INFO("Upscaler capture: {} MB of readback made on the worker; starting in {} s", bytes >> 20, delay);
    g_phase.store(Waiting);
}

void Plan(ID3D12GraphicsCommandList* cmdList, NVSDK_NGX_Parameter* params)
{
    ID3D12Resource* colour = GetResource(params, NVSDK_NGX_Parameter_Color);
    ID3D12Resource* motion = GetResource(params, NVSDK_NGX_Parameter_MotionVectors);
    ID3D12Resource* depth = GetResource(params, NVSDK_NGX_Parameter_Depth);
    ID3D12Resource* output = GetResource(params, NVSDK_NGX_Parameter_Output);
    if (colour == nullptr || motion == nullptr || depth == nullptr || output == nullptr)
    {
        LOG_WARN("Upscaler capture: an input is missing (colour {:p}, motion {:p}, depth {:p}, output {:p}): not taken",
                 (void*) colour, (void*) motion, (void*) depth, (void*) output);
        SetStatus("Upscaler capture: an input is missing (see the log)");
        g_phase.store(Idle);
        return;
    }

    const D3D12_RESOURCE_DESC cd = colour->GetDesc(), md = motion->GetDesc(), dd = depth->GetDesc(), od = output->GetDesc();
    UINT w = GetUInt(params, NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Width);
    UINT h = GetUInt(params, NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Height);
    if (w == 0 || h == 0)
    {
        w = (UINT) cd.Width;
        h = cd.Height;
    }
    const UINT cx = GetUInt(params, NVSDK_NGX_Parameter_DLSS_Input_Color_Subrect_Base_X);
    const UINT cy = GetUInt(params, NVSDK_NGX_Parameter_DLSS_Input_Color_Subrect_Base_Y);
    const UINT mx = GetUInt(params, NVSDK_NGX_Parameter_DLSS_Input_MV_SubrectBase_X);
    const UINT my = GetUInt(params, NVSDK_NGX_Parameter_DLSS_Input_MV_SubrectBase_Y);
    if (cx + w > cd.Width || cy + h > cd.Height || mx + w > md.Width || my + h > md.Height)
    {
        LOG_WARN("Upscaler capture: box {}x{} at colour {},{} / motion {},{} does not fit colour {}x{} / motion {}x{}: "
                 "not taken",
                 w, h, cx, cy, mx, my, cd.Width, cd.Height, md.Width, md.Height);
        SetStatus("Upscaler capture: the render box does not fit (see the log)");
        g_phase.store(Idle);
        return;
    }

    ID3D12Device* device = nullptr;
    if (FAILED(cmdList->GetDevice(IID_PPV_ARGS(&device))) || device == nullptr)
    {
        LOG_WARN("Upscaler capture: no device from the list: not taken");
        SetStatus("Upscaler capture: no device (see the log)");
        g_phase.store(Idle);
        return;
    }

    std::vector<Item> items;
    uint64_t total = 0;
    bool ok = true;
    for (int f = 0; f < kFrames && ok; ++f)
    {
        ok = AddItem(device, items, f, "colour", cd, w, h) && AddItem(device, items, f, "output", od, (UINT) od.Width, od.Height);
        if (ok && f < kFramesWithGuides)
            ok = AddItem(device, items, f, "motion", md, w, h) &&
                 AddItem(device, items, f, "depth", dd, (UINT) dd.Width, dd.Height);
    }
    for (const auto& item : items)
        total += item.total;
    if (!ok || total > kByteLimit)
    {
        LOG_WARN("Upscaler capture: {} MB over the limit of {} MB, or a footprint failed: not taken", total >> 20,
                 kByteLimit >> 20);
        SetStatus("Upscaler capture: too large (see the log)");
        device->Release();
        g_phase.store(Idle);
        return;
    }

    {
        std::lock_guard lock(g_mutex);
        g_boxW = w;
        g_boxH = h;
        g_colourX = cx;
        g_colourY = cy;
        g_motionX = mx;
        g_motionY = my;
        g_colourFormat = cd.Format;
        g_outputFormat = od.Format;
        g_status = "Upscaler capture: making its buffers";
    }
    LOG_INFO("Upscaler capture: planned {} frames, {} MB: colour {}x{} format {} (box {}x{} at {},{}), motion {}x{} format "
             "{} (at {},{}), depth {}x{} format {}, output {}x{} format {}",
             kFrames, total >> 20, cd.Width, cd.Height, (int) cd.Format, w, h, cx, cy, md.Width, md.Height,
             (int) md.Format, mx, my, dd.Width, dd.Height, (int) dd.Format, od.Width, od.Height, (int) od.Format);
    g_phase.store(Allocating);
    std::thread(AllocateOnWorker, device, std::move(items)).detach();
}

// The caller holds g_mutex.
void Record(ID3D12GraphicsCommandList* cmdList, ID3D12Resource* source, int frame, const char* kind,
            D3D12_RESOURCE_STATES state, const D3D12_BOX* box)
{
    for (auto& item : g_items)
    {
        if (item.frame != frame || strcmp(item.kind, kind) != 0)
            continue;
        const D3D12_TEXTURE_COPY_LOCATION dst = CD3DX12_TEXTURE_COPY_LOCATION(item.readback, item.footprint);
        const D3D12_TEXTURE_COPY_LOCATION src = CD3DX12_TEXTURE_COPY_LOCATION(source, 0);
        D3D12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::Transition(source, state, D3D12_RESOURCE_STATE_COPY_SOURCE);
        cmdList->ResourceBarrier(1, &barrier);
        cmdList->CopyTextureRegion(&dst, 0, 0, 0, &src, box);
        std::swap(barrier.Transition.StateBefore, barrier.Transition.StateAfter);
        cmdList->ResourceBarrier(1, &barrier);
        item.recorded = true;
        return;
    }
}

void WriteOnWorker(uint64_t fenceValue)
{
    const uint64_t startTick = GetTickCount64();
    bool completed = g_fence->GetCompletedValue() >= fenceValue;
    if (!completed && SUCCEEDED(g_fence->SetEventOnCompletion(fenceValue, g_fenceEvent)))
        completed = WaitForSingleObject(g_fenceEvent, 10000) == WAIT_OBJECT_0;
    // The Present's queue may not be the only one the lists went to: a moment more, on a worker, costs nothing.
    Sleep(250);
    if (!completed)
        LOG_WARN("Upscaler capture: the fence did not complete in 10 s; writing anyway, the frames may be incomplete");

    std::vector<Item> items;
    std::vector<std::string> info;
    UINT boxW, boxH;
    {
        std::lock_guard lock(g_mutex);
        items = std::move(g_items);
        g_items.clear();
        info = g_frameInfo;
        boxW = g_boxW;
        boxH = g_boxH;
    }

    char stamp[32] {};
    const std::time_t now = std::time(nullptr);
    std::tm local {};
    localtime_s(&local, &now);
    std::strftime(stamp, sizeof(stamp), "%Y%m%d-%H%M%S", &local);
    const auto folder = Util::ExePath().parent_path() / "OptiCapture" / stamp;
    std::error_code ec;
    std::filesystem::create_directories(folder, ec);

    std::ofstream index(folder / "index.txt");
    index << "upscaler capture, " << kFrames << " Presents, box " << boxW << "x" << boxH << "\n";
    for (const auto& line : info)
        index << line << "\n";

    int written = 0, skipped = 0;
    for (auto& item : items)
    {
        const std::string name = std::format("f{:02}_{}.dds", item.frame, item.kind);
        void* mapped = nullptr;
        const D3D12_RANGE range { 0, (SIZE_T) item.total };
        if (!item.recorded || FAILED(item.readback->Map(0, &range, &mapped)) || mapped == nullptr)
        {
            ++skipped;
            index << "SKIPPED " << name << " recorded " << item.recorded << "\n";
        }
        else
        {
            const UINT width = item.footprint.Footprint.Width, height = item.footprint.Footprint.Height;
            uint32_t header[32] = {};
            header[0] = 0x20534444; // "DDS "
            header[1] = 124;
            header[2] = 0x1 | 0x2 | 0x4 | 0x8 | 0x1000; // caps, height, width, pitch, pixel format
            header[3] = height;
            header[4] = width;
            header[5] = (uint32_t) item.rowBytes;
            header[7] = 1;
            header[19] = 32;
            header[20] = 0x4;        // FOURCC
            header[21] = 0x30315844; // "DX10"
            header[27] = 0x1000;
            const uint32_t dx10[5] = { (uint32_t) item.format, 3 /* TEXTURE2D */, 0, 1, 0 };
            std::ofstream file(folder / name, std::ios::binary);
            file.write(reinterpret_cast<const char*>(header), sizeof(header));
            file.write(reinterpret_cast<const char*>(dx10), sizeof(dx10));
            const char* row = static_cast<const char*>(mapped) + item.footprint.Offset;
            for (UINT y = 0; y < item.rows; ++y, row += item.footprint.Footprint.RowPitch)
                file.write(row, (std::streamsize) item.rowBytes);
            const D3D12_RANGE none { 0, 0 };
            item.readback->Unmap(0, &none);
            index << name << " " << width << "x" << height << " format " << (int) item.format << "\n";
            ++written;
        }
        item.readback->Release();
        item.readback = nullptr;
    }
    index.close();

    LOG_INFO("Upscaler capture: {} files written, {} skipped, {} frames, to {} ({} ms on the worker)", written, skipped,
             info.size(), folder.string(), GetTickCount64() - startTick);
    SetStatus(std::format("Upscaler capture: done, {} files in OptiCapture\\{}", written, stamp));
    g_phase.store(Idle);
}
} // namespace

bool EvalCapture_Dx12::Enabled()
{
    static const bool enabled = Config::Instance()->UpscalerInputCapture.value_or_default();
    return enabled;
}

void EvalCapture_Dx12::Arm()
{
    if (!Enabled())
    {
        LOG_WARN("Upscaler capture: pressed, but [Inputs] UpscalerInputCapture is off");
        return;
    }
    int expected = Idle;
    if (g_phase.compare_exchange_strong(expected, Requested))
    {
        SetStatus("Upscaler capture: armed, waiting for the next evaluate");
        LOG_INFO("Upscaler capture: pressed, armed");
    }
    else
    {
        LOG_INFO("Upscaler capture: pressed while one is running (phase {}): ignored", expected);
    }
}

bool EvalCapture_Dx12::Busy() { return g_phase.load() != Idle; }

std::string EvalCapture_Dx12::Status()
{
    std::lock_guard lock(g_mutex);
    return g_status;
}

bool EvalCapture_Dx12::OnEvaluate(ID3D12GraphicsCommandList* cmdList, int featureId, NVSDK_NGX_Parameter* params)
{
    if (!Enabled() || cmdList == nullptr || params == nullptr ||
        (featureId != kSuperSampling && featureId != kRayReconstruction))
        return false;

    int phase = g_phase.load();
    if (phase == Requested)
    {
        if (cmdList->GetType() != D3D12_COMMAND_LIST_TYPE_DIRECT)
        {
            LOG_WARN("Upscaler capture: the evaluate is on a list of type {}, not direct: not taken", (int) cmdList->GetType());
            SetStatus("Upscaler capture: not a direct list (see the log)");
            g_phase.store(Idle);
            return false;
        }
        Plan(cmdList, params);
        return false;
    }
    if (phase == Waiting && GetTickCount64() >= g_startAtTick)
    {
        SetStatus("Upscaler capture: capturing, hold still");
        g_phase.store(Capturing);
        phase = Capturing;
    }
    if (phase != Capturing)
        return false;

    std::lock_guard lock(g_mutex);
    const uint64_t present = g_present.load();
    if (g_outputThisEvaluate != nullptr || present == g_lastPresentTaken)
        return false; // one evaluate per Present

    ID3D12Resource* colour = GetResource(params, NVSDK_NGX_Parameter_Color);
    ID3D12Resource* motion = GetResource(params, NVSDK_NGX_Parameter_MotionVectors);
    ID3D12Resource* depth = GetResource(params, NVSDK_NGX_Parameter_Depth);
    ID3D12Resource* output = GetResource(params, NVSDK_NGX_Parameter_Output);
    const UINT w = GetUInt(params, NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Width);
    const UINT h = GetUInt(params, NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Height);
    if (colour == nullptr || motion == nullptr || depth == nullptr || output == nullptr ||
        cmdList->GetType() != D3D12_COMMAND_LIST_TYPE_DIRECT || (w != 0 && (w != g_boxW || h != g_boxH)) ||
        colour->GetDesc().Format != g_colourFormat || output->GetDesc().Format != g_outputFormat)
    {
        LOG_WARN("Upscaler capture: this evaluate differs from the plan (box {}x{}, planned {}x{}): frame {} skipped", w, h,
                 g_boxW, g_boxH, g_frame);
        return false;
    }

    g_lastPresentTaken = present;
    const int frame = g_frame;
    const D3D12_BOX colourBox { g_colourX, g_colourY, 0, g_colourX + g_boxW, g_colourY + g_boxH, 1 };
    Record(cmdList, colour, frame, "colour", D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, &colourBox);
    if (frame < kFramesWithGuides)
    {
        const D3D12_BOX motionBox { g_motionX, g_motionY, 0, g_motionX + g_boxW, g_motionY + g_boxH, 1 };
        Record(cmdList, motion, frame, "motion", D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, &motionBox);
        Record(cmdList, depth, frame, "depth", D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, nullptr);
    }
    g_frameInfo.push_back(std::format(
        "frame {} present {} parity {} feature {} jitter {} {} mvscale {} {} preexposure {} reset {}", frame, present,
        present & 1, featureId, GetFloat(params, NVSDK_NGX_Parameter_Jitter_Offset_X),
        GetFloat(params, NVSDK_NGX_Parameter_Jitter_Offset_Y), GetFloat(params, NVSDK_NGX_Parameter_MV_Scale_X),
        GetFloat(params, NVSDK_NGX_Parameter_MV_Scale_Y), GetFloat(params, "DLSS.Pre.Exposure"),
        GetUInt(params, NVSDK_NGX_Parameter_Reset)));
    g_outputThisEvaluate = output;
    return true;
}

void EvalCapture_Dx12::AfterEvaluate(ID3D12GraphicsCommandList* cmdList)
{
    std::lock_guard lock(g_mutex);
    if (g_outputThisEvaluate == nullptr)
        return;
    Record(cmdList, g_outputThisEvaluate, g_frame, "output", D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr);
    g_outputThisEvaluate = nullptr;
    if (++g_frame >= kFrames)
    {
        g_drainFrom = g_present.load();
        g_status = "Upscaler capture: writing";
        g_phase.store(Draining);
    }
}

void EvalCapture_Dx12::OnPresent(ID3D12CommandQueue* queue)
{
    if (!Enabled())
        return;
    const uint64_t present = g_present.fetch_add(1) + 1;
    if (g_phase.load() != Draining || queue == nullptr || present < g_drainFrom + kDrainPresents)
        return;
    static uint64_t fenceValue = 0;
    ++fenceValue;
    if (FAILED(queue->Signal(g_fence, fenceValue)))
        LOG_WARN("Upscaler capture: signalling the fence failed");
    g_phase.store(Writing);
    std::thread(WriteOnWorker, fenceValue).detach();
}
