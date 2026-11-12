#pragma once

// GPU timestamps at up to kMarks points of one D3D12 command list, read back when the queue has finished the list.
// For the D3D11-to-D3D12 bridge: its list holds NR before the upscaler, the upscaler and NR after it, and at the
// headset's refresh cap the frame rate says nothing about what each of them costs.
//
// A set of timestamps is one slot of a ring. Begin takes a free slot and writes mark 0; Mark writes the others; End
// resolves them into the readback buffer; Submitted (after ExecuteCommandLists) signals the slot's fence on the
// queue; Collect hands out the intervals of every slot whose fence has passed. A slot that is never submitted is
// given back by the next Begin on the same list.
//
// Header only and D3D12 only, so that the offline test (tools/xr_observer/test/times_test) runs this very code on
// a real device with work of two sizes: a timer's number is not a cost until it has scaled with the work and stayed
// above what the memory's bandwidth allows (three brackets read 0.003-0.031 ms for full-eye copies).
// The caller includes d3d12.h.

#include <cstdint>

class ListTimes_Dx12
{
  public:
    static constexpr int kMarks = 4;
    static constexpr int kSlots = 8;

    struct Intervals
    {
        bool have[kMarks - 1] = {};
        double ms[kMarks - 1] = {}; // mark i to mark i + 1
    };

    ~ListTimes_Dx12() { Release(); }

    bool Init(ID3D12Device* device)
    {
        Release();
        if (device == nullptr)
            return false;
        D3D12_QUERY_HEAP_DESC heapDesc {};
        heapDesc.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
        heapDesc.Count = kSlots * kMarks;
        D3D12_HEAP_PROPERTIES heap {};
        heap.Type = D3D12_HEAP_TYPE_READBACK;
        D3D12_RESOURCE_DESC desc {};
        desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        desc.Width = kSlots * kMarks * sizeof(uint64_t);
        desc.Height = 1;
        desc.DepthOrArraySize = 1;
        desc.MipLevels = 1;
        desc.SampleDesc.Count = 1;
        desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        bool ok = SUCCEEDED(device->CreateQueryHeap(&heapDesc, IID_PPV_ARGS(&_heap))) &&
                  SUCCEEDED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
                                                            D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                            IID_PPV_ARGS(&_readback)));
        for (int i = 0; ok && i < kSlots; ++i)
            ok = SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&_slots[i].fence)));
        if (!ok)
            Release();
        return ok;
    }

    bool Ready() const { return _readback != nullptr; }

    // Mark 0. False: no slot is free (the GPU is more than kSlots lists behind) or nothing was made.
    bool Begin(ID3D12GraphicsCommandList* list)
    {
        _recording = -1;
        if (!Ready() || list == nullptr)
            return false;
        for (int i = 0; i < kSlots; ++i)
        {
            Slot& s = _slots[i];
            if (s.state == Recorded && s.list == list)
                s.state = Free; // recorded on this list and never submitted
            if (s.state != Free || _recording >= 0)
                continue;
            s.state = Recording;
            s.list = list;
            s.marks = 1;
            for (bool& m : s.marked)
                m = false;
            s.marked[0] = true;
            _recording = i;
        }
        if (_recording < 0)
        {
            ++_noSlot;
            return false;
        }
        list->EndQuery(_heap, D3D12_QUERY_TYPE_TIMESTAMP, (UINT) (_recording * kMarks));
        return true;
    }

    void Mark(ID3D12GraphicsCommandList* list, int mark)
    {
        if (_recording < 0 || mark <= 0 || mark >= kMarks || list == nullptr)
            return;
        Slot& s = _slots[_recording];
        if (s.marked[mark])
            return;
        s.marked[mark] = true;
        list->EndQuery(_heap, D3D12_QUERY_TYPE_TIMESTAMP, (UINT) (_recording * kMarks + mark));
    }

    // After the last mark, on the list that will be executed.
    void End(ID3D12GraphicsCommandList* list)
    {
        if (_recording < 0 || list == nullptr)
            return;
        Slot& s = _slots[_recording];
        // Every query of the slot is resolved: one never written in this pass is written here, so that the resolve
        // reads nothing unwritten. Its intervals are not handed out (marked stays false).
        for (int m = 1; m < kMarks; ++m)
            if (!s.marked[m])
                list->EndQuery(_heap, D3D12_QUERY_TYPE_TIMESTAMP, (UINT) (_recording * kMarks + m));
        list->ResolveQueryData(_heap, D3D12_QUERY_TYPE_TIMESTAMP, (UINT) (_recording * kMarks), kMarks, _readback,
                               (UINT64) _recording * kMarks * sizeof(uint64_t));
        s.state = Recorded;
        _recording = -1;
    }

    // After ExecuteCommandLists of that list on that queue.
    void Submitted(ID3D12CommandQueue* queue, ID3D12GraphicsCommandList* list)
    {
        if (queue == nullptr)
            return;
        for (Slot& s : _slots)
        {
            if (s.state != Recorded || s.list != list)
                continue;
            UINT64 frequency = 0;
            if (FAILED(queue->GetTimestampFrequency(&frequency)) || frequency == 0 ||
                FAILED(queue->Signal(s.fence, ++s.value)))
            {
                s.state = Free;
                continue;
            }
            s.frequency = frequency;
            s.state = Submitted_;
        }
    }

    // The intervals of one finished slot (the oldest first); false when none has finished.
    bool Collect(Intervals* out)
    {
        if (!Ready() || out == nullptr)
            return false;
        int oldest = -1;
        for (int i = 0; i < kSlots; ++i)
        {
            Slot& s = _slots[i];
            if (s.state != Submitted_ || s.fence->GetCompletedValue() < s.value)
                continue;
            if (oldest < 0 || s.value < _slots[oldest].value)
                oldest = i;
        }
        if (oldest < 0)
            return false;
        Slot& s = _slots[oldest];
        *out = Intervals {};
        const D3D12_RANGE read { (SIZE_T) oldest * kMarks * sizeof(uint64_t),
                                 (SIZE_T) (oldest + 1) * kMarks * sizeof(uint64_t) };
        uint64_t* data = nullptr;
        if (SUCCEEDED(_readback->Map(0, &read, reinterpret_cast<void**>(&data))) && data != nullptr)
        {
            const uint64_t* t = data + oldest * kMarks;
            for (int m = 0; m + 1 < kMarks; ++m)
            {
                if (!s.marked[m] || !s.marked[m + 1] || t[m + 1] < t[m])
                    continue;
                out->have[m] = true;
                out->ms[m] = (double) (t[m + 1] - t[m]) * 1000.0 / (double) s.frequency;
            }
            const D3D12_RANGE written { 0, 0 };
            _readback->Unmap(0, &written);
        }
        s.state = Free;
        ++_collected;
        return true;
    }

    // The process is ending: nothing is waited for and nothing released (the device may be gone already).
    void Abandon()
    {
        for (Slot& s : _slots)
            s = Slot {};
        _readback = nullptr;
        _heap = nullptr;
        _recording = -1;
    }

    // Before the owner lets go of its lists: see Release.
    void Free_() { Release(); }

    uint64_t Collected() const { return _collected; }
    uint64_t NoSlot() const { return _noSlot; }

  private:
    enum State
    {
        Free,
        Recording,
        Recorded,
        Submitted_
    };
    struct Slot
    {
        ID3D12Fence* fence = nullptr;
        ID3D12GraphicsCommandList* list = nullptr; // identity only
        UINT64 value = 0, frequency = 0;
        State state = Free;
        int marks = 0;
        bool marked[kMarks] = {};
    };

    // The query heap and the readback buffer are named by lists that may still be in flight: each submitted slot's
    // fence is waited for first (half a second at most: a removed device never signals).
    void Release()
    {
        for (Slot& s : _slots)
        {
            if (s.fence != nullptr && s.state == Submitted_ && s.fence->GetCompletedValue() < s.value)
            {
                if (const HANDLE done = CreateEventW(nullptr, FALSE, FALSE, nullptr))
                {
                    if (SUCCEEDED(s.fence->SetEventOnCompletion(s.value, done)))
                        WaitForSingleObject(done, 500);
                    CloseHandle(done);
                }
            }
        }
        for (Slot& s : _slots)
        {
            if (s.fence != nullptr)
                s.fence->Release();
            s = Slot {};
        }
        if (_readback != nullptr)
            _readback->Release();
        _readback = nullptr;
        if (_heap != nullptr)
            _heap->Release();
        _heap = nullptr;
        _recording = -1;
    }

    ID3D12QueryHeap* _heap = nullptr;
    ID3D12Resource* _readback = nullptr;
    Slot _slots[kSlots];
    int _recording = -1;
    uint64_t _collected = 0, _noSlot = 0;
};
