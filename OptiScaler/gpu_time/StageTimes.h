#pragma once

// What our passes cost, by stage, as means over a window of Presents: gathered where the pass runs (the game's render
// thread), read by the overlay and the log (the present thread). Every stage has a FLOOR, the least time its work can
// take by the bytes it moves; a mean under its floor is not a cost and is said to be not credible, never shown as a
// number to be believed (three timers read 0.003-0.031 ms for full-eye copies and were shown for a build).
//
// Header only and standard C++ only, so that the offline test (tools/xr_observer/test/times_test) runs it.

#include <array>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <string>

namespace StageTimes
{
enum Stage
{
    SameEyeMotion, // D3D11: the fit and the composed same-eye motion
    NrBefore,      // D3D12 list: NR in front of the upscaler
    Upscaler,      // D3D12 list: the upscaler (DLSS)
    NrAfter,       // D3D12 list: NR behind the upscaler
    RenderThread,  // CPU: the game's render thread inside the whole evaluate (its waits for the GPU included)
    Count
};

inline const char* Name(int stage)
{
    // In the menu's own words: NR runs "before SR" or "after SR" ([DlssNr] RunBeforeSR), SR being DLSS's upscaling
    // step. "NR before DLSS" read as if DLSS could be off.
    static const char* const names[Count] = { "Same-eye motion (D3D11)", "NR before SR", "DLSS SR (the upscaling)",
                                              "NR after SR", "Render thread in the evaluate (CPU)" };
    return stage >= 0 && stage < Count ? names[stage] : "?";
}

// The floors: the bytes a stage WRITES, over the card's memory bandwidth (an RTX 5090 moves about 1.8 TB/s, and no
// card this runs on moves more). Only what is written: what is read can come from the card's cache, and counting it
// made a floor that real work went under (times_test: 16 copies of a 4 MB image took 0.051 ms, 2.6 TB/s counted
// both ways; 16 copies of a 64 MB image took 1.15 ms, 1.87 TB/s both ways, 0.94 TB/s written).
constexpr double kBytesPerMs = 1.8e9;
inline double FloorMs(double bytesWritten) { return bytesWritten > 0.0 ? bytesWritten / kBytesPerMs : 0.0; }

struct Line
{
    uint32_t samples = 0;
    double meanMs = 0.0, floorMs = 0.0;
    bool credible = false; // samples, and the mean at or above the floor
};

struct Window
{
    uint64_t presents = 0; // 0: no window has ended yet
    std::array<Line, Count> lines {};
    // The process's video memory and its budget (DXGI, local segment), read about once a second: not a mean, the
    // latest. 0 = not read.
    double videoUsedMb = 0.0, videoBudgetMb = 0.0;
};

class Table
{
  public:
    void SetVideoMemory(double usedMb, double budgetMb)
    {
        std::lock_guard lock(_mutex);
        _videoUsedMb = usedMb;
        _videoBudgetMb = budgetMb;
    }

    // From the thread the pass runs on. A floor of 0 = none known (a CPU time).
    void Add(int stage, double ms, double floorMs)
    {
        if (stage < 0 || stage >= Count || !(ms >= 0.0) || ms > 10000.0)
            return;
        std::lock_guard lock(_mutex);
        _sum[stage] += ms;
        _floor[stage] = floorMs;
        ++_samples[stage];
    }

    // Once per Present. True when a window has just ended (the caller logs it).
    bool Present(uint64_t windowPresents = 600)
    {
        std::lock_guard lock(_mutex);
        if (++_presents < windowPresents)
            return false;
        Window w;
        w.presents = _presents;
        for (int i = 0; i < Count; ++i)
        {
            Line& l = w.lines[i];
            l.samples = _samples[i];
            l.floorMs = _floor[i];
            l.meanMs = _samples[i] != 0 ? _sum[i] / (double) _samples[i] : 0.0;
            l.credible = _samples[i] != 0 && l.meanMs >= l.floorMs;
            _sum[i] = 0.0;
            _samples[i] = 0;
        }
        _last = w;
        _presents = 0;
        return true;
    }

    // Timing switched off: nothing of an old window is shown as if it were now.
    void Clear()
    {
        std::lock_guard lock(_mutex);
        _videoUsedMb = _videoBudgetMb = 0.0;
        _last = Window {};
        _presents = 0;
        for (int i = 0; i < Count; ++i)
        {
            _sum[i] = 0.0;
            _samples[i] = 0;
        }
    }

    Window Last()
    {
        std::lock_guard lock(_mutex);
        Window w = _last;
        w.videoUsedMb = _videoUsedMb;
        w.videoBudgetMb = _videoBudgetMb;
        return w;
    }

  private:
    std::mutex _mutex;
    double _videoUsedMb = 0.0, _videoBudgetMb = 0.0;
    std::array<double, Count> _sum {};
    std::array<double, Count> _floor {};
    std::array<uint32_t, Count> _samples {};
    uint64_t _presents = 0;
    Window _last;
};

inline Table& Global()
{
    static Table table;
    return table;
}

// One stage as text, the same in the log and in the overlay.
inline std::string Text(int stage, const Line& line)
{
    char text[200];
    if (line.samples == 0)
        std::snprintf(text, sizeof(text), "%s: did not run", Name(stage));
    else if (!line.credible)
        std::snprintf(text, sizeof(text), "%s: NOT CREDIBLE (%.3f ms read, under the %.3f ms its bytes need; %u samples)",
                      Name(stage), line.meanMs, line.floorMs, line.samples);
    else
        std::snprintf(text, sizeof(text), "%s: %.2f ms (%u samples)", Name(stage), line.meanMs, line.samples);
    return text;
}
} // namespace StageTimes
