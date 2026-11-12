// Offline test of the GPU timers the D3D11-to-D3D12 bridge uses (OptiScaler/gpu_time/ListTimes_Dx12.h) and of the
// table their numbers go through (OptiScaler/gpu_time/StageTimes.h), on a real D3D12 device.
//   times_test
// One command list, four marks: 16 copies of a 4096 x 4096 image, 16 copies of a 1024 x 1024 image, nothing. A
// timer's number is a cost only if it (a) is at or above what the bytes WRITTEN need at the card's bandwidth, (b)
// scales with the work (16 times the pixels: more than 4 times the time is asked for), (c) reads next to nothing
// for nothing. Seven lists, the median of each interval.
// The table: means per window, and a mean under its floor said to be NOT CREDIBLE.
// Control, must FAIL: TIMES_TEST_EMPTY=1 (the large copies are left out: the first interval is then a timer's
// number for no work, which is what three timers once showed).
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_4.h>

#include "../../../OptiScaler/gpu_time/ListTimes_Dx12.h"
#include "../../../OptiScaler/gpu_time/StageTimes.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static ID3D12Resource* MakeTexture(ID3D12Device* device, UINT side, D3D12_RESOURCE_STATES state)
{
    D3D12_HEAP_PROPERTIES heap {};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC desc {};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = side;
    desc.Height = side;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.SampleDesc.Count = 1;
    ID3D12Resource* texture = nullptr;
    device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr, IID_PPV_ARGS(&texture));
    return texture;
}

// Noise into a texture made in COPY_DEST, left in COPY_SOURCE. An image that was never written copies for next to
// nothing (the first version of this test read 0.40 ms for 2.1 GB moved): the hard case is an image with content.
static bool FillWithNoise(ID3D12Device* device, ID3D12CommandQueue* queue, ID3D12CommandAllocator* allocator,
                          ID3D12GraphicsCommandList* list, ID3D12Fence* fence, UINT64* fenceValue, HANDLE done,
                          ID3D12Resource* texture, UINT side)
{
    D3D12_RESOURCE_DESC textureDesc = texture->GetDesc();
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint {};
    UINT64 bytes = 0;
    device->GetCopyableFootprints(&textureDesc, 0, 1, 0, &footprint, nullptr, nullptr, &bytes);
    D3D12_HEAP_PROPERTIES heap {};
    heap.Type = D3D12_HEAP_TYPE_UPLOAD;
    D3D12_RESOURCE_DESC desc {};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = bytes;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    ID3D12Resource* upload = nullptr;
    if (FAILED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_GENERIC_READ,
                                               nullptr, IID_PPV_ARGS(&upload))))
        return false;
    uint8_t* data = nullptr;
    if (FAILED(upload->Map(0, nullptr, reinterpret_cast<void**>(&data))))
        return false;
    uint32_t state = 0x12345678u ^ side;
    for (UINT y = 0; y < side; ++y)
    {
        uint32_t* row = reinterpret_cast<uint32_t*>(data + footprint.Offset + (size_t) y * footprint.Footprint.RowPitch);
        for (UINT x = 0; x < side; ++x)
        {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            row[x] = state;
        }
    }
    upload->Unmap(0, nullptr);
    allocator->Reset();
    list->Reset(allocator, nullptr);
    D3D12_TEXTURE_COPY_LOCATION to {}, from {};
    to.pResource = texture;
    to.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    from.pResource = upload;
    from.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    from.PlacedFootprint = footprint;
    list->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
    D3D12_RESOURCE_BARRIER barrier {};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = texture;
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    list->ResourceBarrier(1, &barrier);
    list->Close();
    ID3D12CommandList* lists[] = { list };
    queue->ExecuteCommandLists(1, lists);
    queue->Signal(fence, ++*fenceValue);
    fence->SetEventOnCompletion(*fenceValue, done);
    const bool finished = WaitForSingleObject(done, 10000) == WAIT_OBJECT_0;
    upload->Release();
    return finished;
}

static double Median(std::vector<double> v)
{
    if (v.empty())
        return -1.0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

int main()
{
    char value[8] = {};
    const bool empty = GetEnvironmentVariableA("TIMES_TEST_EMPTY", value, sizeof(value)) > 0 && value[0] == '1';
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);

    ID3D12Device* device = nullptr;
    if (FAILED(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device))))
    {
        std::printf("FAILED: no D3D12 device\n");
        return 1;
    }
    D3D12_COMMAND_QUEUE_DESC queueDesc {};
    queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    ID3D12CommandQueue* queue = nullptr;
    ID3D12CommandAllocator* allocator = nullptr;
    ID3D12GraphicsCommandList* list = nullptr;
    ID3D12Fence* fence = nullptr;
    if (FAILED(device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&queue))) ||
        FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator))) ||
        FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator, nullptr, IID_PPV_ARGS(&list))) ||
        FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence))))
    {
        std::printf("FAILED: the queue, the list or the fence\n");
        return 1;
    }
    list->Close();
    const HANDLE done = CreateEventW(nullptr, FALSE, FALSE, nullptr);

    constexpr UINT kLarge = 4096, kSmall = 1024;
    constexpr int kCopies = 16, kLists = 7;
    ID3D12Resource* largeFrom = MakeTexture(device, kLarge, D3D12_RESOURCE_STATE_COPY_DEST);
    ID3D12Resource* largeTo = MakeTexture(device, kLarge, D3D12_RESOURCE_STATE_COPY_DEST);
    ID3D12Resource* smallFrom = MakeTexture(device, kSmall, D3D12_RESOURCE_STATE_COPY_DEST);
    ID3D12Resource* smallTo = MakeTexture(device, kSmall, D3D12_RESOURCE_STATE_COPY_DEST);
    ListTimes_Dx12 times;
    if (largeFrom == nullptr || largeTo == nullptr || smallFrom == nullptr || smallTo == nullptr || !times.Init(device))
    {
        std::printf("FAILED: the textures or the timer\n");
        return 1;
    }

    std::vector<double> got[3];
    UINT64 fenceValue = 0;
    if (!FillWithNoise(device, queue, allocator, list, fence, &fenceValue, done, largeFrom, kLarge) ||
        !FillWithNoise(device, queue, allocator, list, fence, &fenceValue, done, smallFrom, kSmall))
    {
        std::printf("FAILED: the images could not be filled\n");
        return 1;
    }
    for (int i = 0; i < kLists; ++i)
    {
        allocator->Reset();
        list->Reset(allocator, nullptr);
        const bool began = times.Begin(list);
        if (!empty)
            for (int c = 0; c < kCopies; ++c)
                list->CopyResource(largeTo, largeFrom);
        times.Mark(list, 1);
        for (int c = 0; c < kCopies; ++c)
            list->CopyResource(smallTo, smallFrom);
        times.Mark(list, 2);
        times.Mark(list, 3);
        times.End(list);
        list->Close();
        ID3D12CommandList* lists[] = { list };
        queue->ExecuteCommandLists(1, lists);
        times.Submitted(queue, list);
        queue->Signal(fence, ++fenceValue);
        fence->SetEventOnCompletion(fenceValue, done);
        WaitForSingleObject(done, 5000);
        ListTimes_Dx12::Intervals intervals;
        const bool collected = times.Collect(&intervals);
        std::printf("list %d: begun %s, collected %s; large copies %.4f ms, small copies %.4f ms, nothing %.4f ms\n", i + 1,
                    began ? "yes" : "NO", collected ? "yes" : "NO", intervals.have[0] ? intervals.ms[0] : -1.0,
                    intervals.have[1] ? intervals.ms[1] : -1.0, intervals.have[2] ? intervals.ms[2] : -1.0);
        // The first list pays for the textures' first use: not scored.
        if (i > 0 && collected)
            for (int m = 0; m < 3; ++m)
                if (intervals.have[m])
                    got[m].push_back(intervals.ms[m]);
    }

    int failures = 0, checked = 0;
    const double tLarge = Median(got[0]), tSmall = Median(got[1]), tNothing = Median(got[2]);
    // What is WRITTEN only: counted both ways, the small copies went under the floor (their source is read from
    // the card's cache).
    const double largeFloor = StageTimes::FloorMs(kCopies * (double) kLarge * kLarge * 4.0);
    const double smallFloor = StageTimes::FloorMs(kCopies * (double) kSmall * kSmall * 4.0);
    const bool enough = got[0].size() >= 5 && got[1].size() >= 5 && got[2].size() >= 5;
    const bool aboveFloor = tLarge >= largeFloor && tSmall >= smallFloor;
    const bool scales = tSmall > 0.0 && tLarge / tSmall > 4.0;
    const bool nothingIsLittle = tNothing >= 0.0 && tNothing < tSmall;
    std::printf("the timer, medians of %zu lists: 16 copies of %u^2 %.4f ms (floor %.4f), 16 copies of %u^2 %.4f ms "
                "(floor %.4f), nothing %.4f ms; above the floors %s, ratio %.1f (more than 4 wanted) %s, nothing under "
                "the small copies %s -> %s\n",
                got[0].size(), kLarge, tLarge, largeFloor, kSmall, tSmall, smallFloor, tNothing, aboveFloor ? "yes" : "NO",
                tSmall > 0.0 ? tLarge / tSmall : 0.0, scales ? "yes" : "NO", nothingIsLittle ? "yes" : "NO",
                enough && aboveFloor && scales && nothingIsLittle ? "OK" : "FAIL");
    ++checked;
    if (!(enough && aboveFloor && scales && nothingIsLittle))
        ++failures;

    // ---- The table: what the timer measured goes in with its floor; a number under its floor is said so. ----
    {
        StageTimes::Table table;
        bool ended = false;
        for (int p = 0; p < 600; ++p)
        {
            if (p % 2 == 0)
            {
                table.Add(StageTimes::Upscaler, 3.0, 0.2);
                table.Add(StageTimes::NrAfter, 5.0, 0.1);
                table.Add(StageTimes::SameEyeMotion, 0.003, 0.1); // a timer's number for no work
                table.Add(StageTimes::RenderThread, 9.0, 0.0);
            }
            if (p % 4 == 0)
                table.Add(StageTimes::Upscaler, 5.0, 0.2);
            ended = table.Present(600) || ended;
        }
        const StageTimes::Window w = table.Last();
        const auto& up = w.lines[StageTimes::Upscaler];
        const auto& motion = w.lines[StageTimes::SameEyeMotion];
        const auto& before = w.lines[StageTimes::NrBefore];
        const std::string motionText = StageTimes::Text(StageTimes::SameEyeMotion, motion);
        const std::string beforeText = StageTimes::Text(StageTimes::NrBefore, before);
        const bool means = ended && w.presents == 600 && up.samples == 450 && std::fabs(up.meanMs - 3.6667) < 0.001 &&
                           up.credible && w.lines[StageTimes::RenderThread].credible;
        const bool underFloor = motion.samples == 300 && !motion.credible &&
                                motionText.find("NOT CREDIBLE") != std::string::npos;
        const bool notRun = before.samples == 0 && !before.credible && beforeText.find("did not run") != std::string::npos;
        table.Clear();
        const bool cleared = table.Last().presents == 0;
        const bool ok = means && underFloor && notRun && cleared;
        std::printf("the table: window of %llu Presents; DLSS %.4f ms over %u samples (3.6667 over 450 wanted) %s; a "
                    "mean under its floor \"%s\" %s; a stage that did not run \"%s\" %s; cleared %s -> %s\n",
                    (unsigned long long) w.presents, up.meanMs, up.samples, means ? "yes" : "NO", motionText.c_str(),
                    underFloor ? "yes" : "NO", beforeText.c_str(), notRun ? "yes" : "NO", cleared ? "yes" : "NO",
                    ok ? "OK" : "FAIL");
        ++checked;
        if (!ok)
            ++failures;
    }
    std::printf("%s: %d checks, %d failures\n", failures == 0 ? "PASSED" : "FAILED", checked, failures);
    return failures == 0 ? 0 : 1;
}
