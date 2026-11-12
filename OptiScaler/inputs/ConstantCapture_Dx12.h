#pragma once

#include <d3d12.h>

#include <cstdint>
#include <filesystem>
#include <vector>

// Diagnostics ([Inputs] DuniaTaaDiagnostics, F7): the constant buffers FC6 binds while it draws a frame, so the
// camera matrices can be found offline. The D3D12 answer to what Fc5Trace_Dx11 does for Far Cry 5.
//
// The motion question is settled: FC6 has no usable motion vector buffer (both
// candidates proved to be byte-identical across six frames of a fast pan, one cleared to 1.0 and one holding
// uninitialised memory). That leaves FC5's proven route: build camera motion from depth and a
// current-to-previous matrix. This module is how that matrix gets found.
//
// What it does, while a FrameCapture_Dx12 capture is running:
//   * every upload-heap buffer the game creates is recorded by its GPU virtual address range and mapped once,
//     persistently -- upload heaps are CPU-visible, so no GPU copy, no barriers and no readback buffer;
//   * every root CBV the game binds (SetGraphicsRootConstantBufferView / SetComputeRootConstantBufferView) is
//     resolved through that map and its bytes copied straight out of the mapping, there and then. Reading at the
//     bind is what makes it safe: the buffer is alive, the contents are the ones that draw will see, and nothing
//     depends on the resource still existing when the capture is written.
//
// Written as constants.txt beside the capture's .dds files, in the format tools\capture\parsecb.py already reads:
//
//   frame 0 graphics cb1 buffer 7F00A2C0 2048 bytes
//        0:     1.000000     0.000000 ... | 3F800000 00000000 ...
//
// Then `python tools\capture\cbmat2.py tools\capture out.txt <capture dir>` picks out the rotation, projection and
// near-identity 4x4 matrices. FC5's offsets are the first guess: its temporal-effects constant block holds the current-to-previous
// matrix at +16..+64, and a 1984-byte per-viewport block holds projection and jitter.
//
// Limits worth knowing before reading the output:
//   * root CBVs only. Constants reached through a descriptor table are not here yet -- that needs the descriptor
//     heap walked through ResTrack_dx12, and root CBVs are where Dunia's per-view constants have turned up so far.
//     If the matrices are not in this dump, descriptor tables are the next place to look.
//   * a root CBV carries no size, so each one is dumped to kMaxBytesPerCb or the end of its buffer.
//   * several distinct contents are kept per slot per frame (a shadow pass and the main view share slot numbers),
//     so expect repeats. parsecb.py returns them all; cbdiff_lib/cbmat2 keep the last.
namespace ConstantCapture_Dx12
{
// True when the root CBV hooks are worth installing at all. D3D12_Hooks asks this at hook time: without it the
// constant buffer hooks only go on when [Hotfix] ExtendedStateRestore is set, which is off by default.
bool Wanted();

// Every buffer the game creates, from the CreateCommittedResource / CreatePlacedResource hooks. Only CPU-visible
// buffers are kept; everything else returns immediately. Cheap enough to sit in the creation path unconditionally.
// Takes the whole heap properties, not just the type: a CUSTOM heap is CPU-visible or not according to its
// CPUPageProperty, and Dunia's constants turned out not to be in a plain upload heap -- every ReadAt missed,
// so a capture full of CBV addresses produced not one byte of contents.
void NoteBuffer(const D3D12_RESOURCE_DESC* desc, const D3D12_HEAP_PROPERTIES& props, ID3D12Resource* resource);

// How many CPU-visible buffers are mapped, how many address lookups were tried, and how many found nothing.
// Without these "0 constant buffers dumped" has three explanations that look identical.
// Every buffer the game maps, from the ID3D12Resource::Map hook. Version-proof where NoteBuffer is not: the
// creation hooks only cover ID3D12Device::CreateCommittedResource and CreatePlacedResource, and a modern engine
// may use CreateCommittedResource1/2/3 or CreatePlacedResource1/2 instead -- FC6 recorded ZERO buffers through
// the creation path, upload heaps included. Whatever made the buffer, the game has to map it to write constants
// into it, and this takes its own reference-counted mapping rather than trusting the game's pointer to outlive
// the Unmap.
void NoteMappedBuffer(ID3D12Resource* resource);

// Every Unmap, from the ID3D12Resource::Unmap hook, BEFORE the original. If the game passes a written range
// and a watch pattern is set, the range is checked for the pattern there and then -- the game has just
// written those bytes, so a per-view block that is rewritten every frame is caught every frame. Hits are
// kept (deduplicated, capped) for TakeWrittenHits. A null range is counted and otherwise ignored: it means
// "everything", and checking a whole ring on the game's thread is the sweep again.
void NoteWrittenRange(ID3D12Resource* resource, const D3D12_RANGE* written);
void SetWatchPattern(const float* pattern, size_t count, float relativeTolerance, bool (*accept)(const uint8_t*));
size_t TakeWrittenHits(const uint8_t** out, size_t maxOut);
// Called for EVERY raw match of the pattern, in the sweep and at Unmap alike, before the per-view validator and
// before its 512-byte margin test -- with how many bytes of the mapping lie before the quad and from the quad
// to its end, so the observer can bound its own reads. It exists because the viewport quad is also register 7
// of the motion pass's constant block, which the per-view validator rightly rejects: the
// sweep was already walking over those blocks and throwing them away. Runs under this module's lock, on
// whichever thread found the match, so it must be quick and must not call back in here.
void SetRawMatchObserver(void (*observer)(const uint8_t* quad, uint64_t bytesBefore, uint64_t bytesFromQuad));
// Unmap statistics for the log: calls, calls with a range, calls whose range was checked, hits so far, and
// why the rest were not checked -- the range too big, the resource not among the mappings, or the range
// past the mapping's end. Without these "0 hits" has four explanations that look identical.
struct WrittenRangeStatsOut
{
    uint64_t calls = 0, withRange = 0, checked = 0, hits = 0;
    uint64_t tooBig = 0, noMapping = 0, pastEnd = 0, rawMatches = 0;
};
WrittenRangeStatsOut WrittenRangeStats();

// A mapped buffer has died. Called from ResTrack_Dx12::hkRelease on the last release of every resource, while
// it is still alive, so the mapping is dropped BEFORE the pages go -- and so is anything cached into them.
//
// This exists because the alternative does not work: the cached CPU pointers cannot be validated by reading
// them, because reading a freed page IS the crash. A menu is a size change, a size change is when Dunia frees
// and reallocates, and the matrix path had raw pointers into those buffers. Invalidate on the resource's death,
// never on its contents looking wrong. Never AddRef the game's buffer to keep it alive either -- that once
// broke resolution changes.
//
// Returns true when a mapping was dropped. Also hands DuniaTaa_Dx12 the dead CPU range so it can forget any
// candidate inside it -- after releasing this module's lock: ViewConstants::Update holds its own lock while
// calling FindPattern here, so calling out under g_lock is the opposite lock order and would deadlock.
bool ForgetResource(ID3D12Resource* resource);

uint64_t MappedCount();
// How many buffers each route offered, so "0 mapped" says which hook is not firing rather than just failing.
uint64_t BuffersSeenAtCreate();
uint64_t BuffersSeenAtMap();

// Every mapped buffer that holds `count` consecutive floats equal to `pattern` at a 16-byte-aligned offset,
// returned as pointers to the START of that quad. This is how the per-view constants are found at runtime with
// no descriptor and no heap mirroring: Dunia's viewport struct carries [w, h, 1/w, 1/h], and two of those being
// the exact reciprocals of the other two at the eye size is not something another buffer stumbles into.
//
// `accept` is applied to every raw match INSIDE the sweep and only accepted ones count towards `maxOut`. The
// raw quad matches 512+ buffers in FC6 and the sweep used to stop at that many raw hits, so whether the handful
// of real per-view structs were among them depended on where they sat in address order: one run found 6, the
// next found 0 and matrix motion never ran. `rawSeen` reports how many raw matches the sweep looked at, and
// `sweptBytes` how much it read, so the cost of a scan is in the log and not a guess.
// The pointers stay valid until the buffer is released, which ForgetResource reports.
//
// The sweep is RESUMABLE and TIME-BUDGETED. Upload memory is write-combined, and reading it from the CPU is
// uncached: a full sweep of FC6's ~600 mapped buffers (24 MB) measured 1040-1090 ms, which was the one-second
// hitch on leaving a menu. One call does at most `msBudget` of it and records where it stopped in `scan`;
// the caller calls again next Present until `scan.done`. Results are appended to `out` from `scan.found`.
// Each mapping is read in 4 KB chunks into a local buffer first, so the uncached reads are sequential
// streaming loads rather than one 4-byte load per 16-byte stride.
// The order is fixed when the scan starts: mappings that held a per-view struct before (by size, since a menu
// replaces the buffer but not its size), then MOST RECENTLY MAPPED first. The game's per-view constants are
// sub-allocated inside its constant rings, which it re-Maps every frame to write into; texture-streaming
// rings are mapped at loads. Address order found them by luck and then ground through 536 MB of streaming
// rings; size order put the ~6 MB constant rings last (mapping 77 of 537 after 200 slices, never done).
// The budget is checked per 4 KB chunk and the scan resumes mid-buffer.
struct ScanEntry
{
    uint64_t priority = 0;
    uint64_t size = 0;
    D3D12_GPU_VIRTUAL_ADDRESS base = 0;
};
struct PatternScan
{
    std::vector<ScanEntry> order; // empty = not started
    size_t index = 0;      // next entry of `order`
    uint64_t offset = 0;   // resume offset inside that mapping
    bool done = false;
    size_t found = 0;
    size_t raw = 0;
    uint64_t swept = 0;
    uint64_t largest = 0;  // the biggest mapping in the set, for the scan line
    uint64_t total = 0;    // the whole set's bytes, so the log says how far a stalled sweep got
    double ms = 0.0;
    unsigned slices = 0;
};
void FindPattern(const float* pattern, size_t count, float relativeTolerance, const uint8_t** out, size_t maxOut,
                 bool (*accept)(const uint8_t* quad), PatternScan& scan, double msBudget);

// Counters for the scan line. Without them "0 real per-view structs" has at least three explanations that look
// identical: the mappings were never made, they were dropped by a false death, or the live ones were rebound
// under a new resource at an address we already had and were thrown away as duplicates.
uint64_t ForgetCount();
uint64_t RebindCount();

// A pointer the caller committed to as a real per-view struct: the mapping holding it, and its SIZE, go to
// the front of every later sweep. The size survives the mapping's death, which is what a menu is.
void NoteUsefulPointer(const uint8_t* ptr);
// The CPU range of the mapping that holds `ptr`. For a caller that has to look through a whole constant pool
// for a block that moves (FC6's motion pass constants). The range dies with the buffer: ForgetResource hands
// it to DuniaTaa_Dx12::ForgetViewConstants, and the caller must drop it there, not by reading it.
bool MappingOf(const uint8_t* ptr, const uint8_t** begin, uint64_t* size);
// The GPU virtual address of a CPU pointer inside a mapped buffer: what a CBV descriptor would hold for it.
// Compares pointers only, never reads through them.
bool GpuAddressOf(const uint8_t* ptr, uint64_t* gpuAddress);
// Copies `bytes` from a GPU address inside a live mapping, under the lock the resource-death path takes: the only
// safe way to read a buffer we hold no pointer into. False if no live mapping covers the whole range.
bool ReadAtGpuAddress(uint64_t gpuAddress, void* out, size_t bytes);
uint64_t ReadAttempts();
uint64_t ReadMisses();

// Every root CBV bind, from the D3D12_Hooks constant buffer hooks. Does nothing unless a capture is running.
void NoteRootCbv(bool graphics, UINT rootParameterIndex, D3D12_GPU_VIRTUAL_ADDRESS location);

// Reads the bytes at a GPU virtual address out of the persistent mapping of whichever recorded buffer contains
// it, which is how a constant buffer reached through a *descriptor table* becomes readable -- the descriptor
// carries an address, not a resource. Same resolution NoteRootCbv does: the greatest base at or below the
// address, then a range check, because Dunia sub-allocates per-draw constants out of big shared upload buffers.
// `bytes` caps the read; the result is clamped to the end of the buffer and to whole 32-bit words. False when
// the address is in no recorded buffer, which is what a default heap or an unrecorded allocation looks like.
bool ReadAt(D3D12_GPU_VIRTUAL_ADDRESS location, UINT64 bytes, std::vector<uint32_t>& out);

// Driven by FrameCapture_Dx12, which owns the capture's lifetime and its folder.
void Start();
void SetFrame(int frame);
void Stop();

// The constants.txt text for the capture just finished, or an empty string if nothing was recorded. Called on the
// present thread; the caller writes it on its background thread with the rest of the capture.
std::string Take();
} // namespace ConstantCapture_Dx12
