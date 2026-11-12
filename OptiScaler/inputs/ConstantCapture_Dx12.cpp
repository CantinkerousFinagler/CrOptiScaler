#include "pch.h"
#include "ConstantCapture_Dx12.h"

#include <inputs/DuniaTaa_Dx12.h>
#include <cmath>

#include "Config.h"
#include "Logger.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <format>
#include <map>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace
{
// Buffers are created and bound from several threads; the capture itself is driven from the present thread. One
// lock covers both tables -- the bind path is the hot one, but it does a map lookup and a memcpy, not a syscall.
std::mutex g_lock;

// Read before the lock is taken. Every root CBV bind in the frame comes through NoteRootCbv, capture or not, so
// the not-capturing path has to cost nothing more than this load.
std::atomic<bool> g_capturing { false };
int g_frame = 0;

// A root CBV is a bare GPU address, so the only way back to bytes is a map of what lives where. Buffers that the
// CPU cannot see are not recorded at all: reading them would need a copy, a barrier and a fence, and the camera
// constants of a Dunia frame are written by the CPU every frame and so live in upload heaps.
struct Mapping
{
    D3D12_GPU_VIRTUAL_ADDRESS base = 0;
    UINT64 size = 0;
    uint8_t* cpu = nullptr;
    // Identity only, NEVER AddRef'd -- holding a reference to a game resource is what stopped the swapchain
    // resizing and broke resolution changes. This is here so the resource's death
    // can be recognised in ResTrack's Release hook, not so it can be prevented.
    ID3D12Resource* resource = nullptr;
    // Bumped on every Map of this buffer. The per-view constants live in rings the game re-Maps every frame,
    // and that is the property the sweep orders by.
    uint64_t lastMapped = 0;
    bool useful = false;
};
std::atomic<uint64_t> g_mapSerial { 0 };
// Sizes of mappings that held a per-view struct. Survives their death: after a menu the replacement ring has
// the same size, so it is swept first even before it has been mapped enough times to lead on recency.
std::vector<uint64_t> g_usefulSizes;
// Keyed by base address, so a bind resolves with upper_bound and one step back.
std::map<D3D12_GPU_VIRTUAL_ADDRESS, Mapping> g_mappings;
// The other direction, for the death notice: hkRelease has a resource and no address, and it fires for every
// resource the game frees, so that lookup has to be a hash and not a walk of the map above.
std::unordered_map<ID3D12Resource*, D3D12_GPU_VIRTUAL_ADDRESS> g_byResource;
constexpr size_t kMaxMappings = 4096;

// Read before the lock in ForgetResource, which is on the path of every resource release in the game. Zero
// means there is nothing to forget and no lock is worth taking.
std::atomic<size_t> g_mappingCount { 0 };

// One place that records a mapping, because there are two callers and both have to keep the two tables and the
// counter agreeing. Caller holds g_lock. Replacing the entry at an address is the point -- D3D12 recycles
// virtual address ranges, and a stale record of a freed buffer would otherwise answer a lookup for its
// replacement -- so the resource the old entry named is dropped from the reverse index with it.
void RecordMappingLocked(D3D12_GPU_VIRTUAL_ADDRESS base, UINT64 size, uint8_t* cpu, ID3D12Resource* resource)
{
    const auto existing = g_mappings.find(base);
    if (existing != g_mappings.end() && existing->second.resource != nullptr &&
        existing->second.resource != resource)
        g_byResource.erase(existing->second.resource);

    g_mappings[base] = Mapping { base, size, cpu, resource, g_mapSerial.fetch_add(1, std::memory_order_relaxed) + 1 };
    if (resource != nullptr)
        g_byResource[resource] = base;
    g_mappingCount.store(g_mappings.size(), std::memory_order_relaxed);
}

// A root CBV says nothing about how big the buffer behind it is, so each one is dumped to this or to the end of
// its buffer, whichever comes first. FC5's per-viewport constant block is 1984 bytes.
constexpr UINT64 kMaxBytesPerCb = 4096;

// The same slot is bound many times a frame, by passes that do not share a camera: a shadow pass and the main view
// both bind "cb1". Keeping several distinct contents per slot means the main view's matrices are in the dump even
// when a shadow pass binds that slot first.
constexpr size_t kMaxRecordsPerSlot = 8;
constexpr size_t kMaxRecords = 512;

struct Record
{
    int frame = 0;
    bool graphics = true;
    UINT slot = 0;
    D3D12_GPU_VIRTUAL_ADDRESS address = 0;
    std::vector<uint32_t> words;
};
std::vector<Record> g_records;

size_t CountForSlot(int frame, bool graphics, UINT slot)
{
    size_t count = 0;
    for (const auto& record : g_records)
    {
        if (record.frame == frame && record.graphics == graphics && record.slot == slot)
            ++count;
    }
    return count;
}

bool AlreadyHave(int frame, bool graphics, UINT slot, const uint32_t* words, size_t count)
{
    for (const auto& record : g_records)
    {
        if (record.frame != frame || record.graphics != graphics || record.slot != slot)
            continue;
        if (record.words.size() == count && std::memcmp(record.words.data(), words, count * sizeof(uint32_t)) == 0)
            return true;
    }
    return false;
}

float AsFloat(uint32_t word)
{
    float value = 0.0f;
    std::memcpy(&value, &word, sizeof(value));
    return value;
}
} // namespace

// Real motion needs the registry too, not just a capture: the matrix path reads the game's per-view constants
// out of these mappings every frame. Without this the whole thing would only work with diagnostics on, which is
// exactly the "works in the mode it was built for and nowhere else" trap that took F7 out.
bool ConstantCapture_Dx12::Wanted()
{
    return Config::Instance()->DuniaTaaDiagnostics.value_or_default() || DuniaTaa_Dx12::WantsRealMotion() ||
           Config::Instance()->GiBufferCapture.value_or_default() || Config::Instance()->PerEyeLighting.value_or_default();
}

// Told about every raw pattern match, in the sweep below and at Unmap; see SetRawMatchObserver.
static std::atomic<void (*)(const uint8_t*, uint64_t, uint64_t)> g_rawObserver { nullptr };

void ConstantCapture_Dx12::FindPattern(const float* pattern, size_t count, float rel, const uint8_t** out,
                                       size_t maxOut, bool (*accept)(const uint8_t* quad), PatternScan& scan,
                                       double msBudget)
{
    if (scan.done)
        return;
    if (pattern == nullptr || count == 0 || out == nullptr || maxOut == 0 || count * sizeof(float) > 64)
    {
        scan.done = true;
        return;
    }

    const auto t0 = std::chrono::steady_clock::now();
    const auto elapsedMs = [&]
    { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(); };
    ++scan.slices;

    std::lock_guard<std::mutex> lock(g_lock);

    // First slice: fix the order, smallest first. Mappings that die before their turn are simply absent when
    // looked up; ones that arrive after the start are caught by the next scan.
    if (scan.order.empty())
    {
        scan.order.reserve(g_mappings.size());
        const uint64_t serial = g_mapSerial.load(std::memory_order_relaxed);
        for (const auto& entry : g_mappings)
        {
            const Mapping& m = entry.second;
            if (m.cpu == nullptr || m.size < count * sizeof(float))
                continue;
            // Tiers, highest first: held a struct before; same size as one that did; RING-SIZED (1-32 MB),
            // largest first; everything else by recency. The per-view structs are sub-allocated inside the
            // game's constant rings, which at gameplay time are its largest mappings (~6 MB) -- size-ascending
            // put them last (mapping 77 of 537 after 200 slices), and recency does not discriminate because
            // the game re-Maps nearly every buffer every frame (mapping 57 of 491). Two runs, one lesson.
            const bool sizeKnown =
                std::find(g_usefulSizes.begin(), g_usefulSizes.end(), m.size) != g_usefulSizes.end();
            // Within a tier: ADDRESS ASCENDING, i.e. allocation order, startup buffers first. It is the only
            // order that has ever found the per-view structs (three sessions, inside the first 23 MB); size
            // ascending, recency and ring-size-descending each swept 20-50 MB and found nothing.
            const uint64_t tier = m.useful ? 2 : (sizeKnown ? 1 : 0);
            const uint64_t within = ((1ull << 62) - 1) - (m.base & ((1ull << 62) - 1));
            scan.order.push_back(ScanEntry { (tier << 62) | within, m.size, m.base });
            scan.largest = std::max(scan.largest, m.size);
            scan.total += m.size;
        }
        std::sort(scan.order.begin(), scan.order.end(),
                  [](const ScanEntry& a, const ScanEntry& b) { return a.priority > b.priority; });
        scan.index = 0;
        scan.offset = 0;
        (void) serial;
    }

    // Chunked so the uncached reads stream. The chunk overlaps by one stride so a quad that straddles two
    // chunks is still seen whole: a 16-byte-aligned quad of up to 64 bytes lies inside [off, off + 64).
    constexpr size_t kChunk = 4096;
    alignas(16) uint8_t buf[kChunk + 64];

    for (; scan.index < scan.order.size(); ++scan.index, scan.offset = 0)
    {
        const auto it = g_mappings.find(scan.order[scan.index].base);
        if (it == g_mappings.end() || it->second.cpu == nullptr || it->second.size != scan.order[scan.index].size)
            continue; // died, or replaced by something else at that address, since the order was fixed
        const Mapping& m = it->second;
        const UINT64 limit = m.size - count * sizeof(float);
        for (UINT64 start = scan.offset; start <= limit; start += kChunk)
        {
            const size_t len = (size_t) std::min<UINT64>(kChunk + 64, m.size - start);
            std::memcpy(buf, m.cpu + start, len);
            scan.swept += std::min<UINT64>(kChunk, m.size - start);
            // Constant buffers are 16-byte aligned and so is every register in them, so only those
            // offsets are worth testing. Only quads that START inside this chunk, so none is seen twice.
            const size_t inner = (size_t) std::min<UINT64>(kChunk, limit + 1 - start);
            for (size_t off = 0; off < inner; off += 16)
            {
                if (off + count * sizeof(float) > len)
                    break;
                const float* v = (const float*) (buf + off);
                bool ok = true;
                for (size_t i = 0; i < count && ok; ++i)
                {
                    const float a = v[i], b = pattern[i];
                    if (a != a || b == 0.0f)
                        ok = false;
                    else if (std::abs(a - b) > std::abs(b) * rel)
                        ok = false;
                }
                if (!ok)
                    continue;
                ++scan.raw;
                // The validation happens here, not after a capped raw list, so a real struct beyond the
                // 512th raw hit is still found. It reads the mapping itself (the struct extends well
                // beyond the quad), which is live because g_lock is held.
                const uint8_t* quad = m.cpu + start + off;
                if (const auto observer = g_rawObserver.load(std::memory_order_acquire))
                    observer(quad, start + off, m.size - (start + off));
                // The validator reads the struct around the quad: 512 bytes before it, ~64 after. Keep it
                // inside the mapping.
                if (start + off < 512 || start + off + 128 > m.size)
                    continue;
                if (accept != nullptr && !accept(quad))
                    continue;
                out[scan.found++] = quad;
                if (scan.found >= maxOut)
                {
                    scan.ms += elapsedMs();
                    scan.done = true;
                    return;
                }
            }
            // The budget is checked per chunk, so a slice is bounded whatever the buffer's size, and the
            // scan resumes inside this mapping next time.
            if (elapsedMs() >= msBudget)
            {
                scan.offset = start + kChunk;
                if (scan.offset > limit)
                {
                    ++scan.index;
                    scan.offset = 0;
                }
                scan.ms += elapsedMs();
                return;
            }
        }
    }
    scan.ms += elapsedMs();
    scan.done = true;
}

// The watch: set once by DuniaTaa when the eye shape is known. Read on every Unmap, so the pattern itself is
// copied under the lock once and the hot path takes the lock only when a range is actually present.
float g_watch[4] {};
size_t g_watchCount = 0;
float g_watchRel = 0.0f;
bool (*g_watchAccept)(const uint8_t*) = nullptr;
std::atomic<bool> g_watchSet { false };
constexpr size_t kMaxWrittenHits = 32;
std::vector<const uint8_t*> g_writtenHits;
std::atomic<uint64_t> g_unmapCalls { 0 }, g_unmapWithRange { 0 }, g_unmapChecked { 0 }, g_unmapHits { 0 };
std::atomic<uint64_t> g_unmapTooBig { 0 }, g_unmapNoMapping { 0 }, g_unmapPastEnd { 0 }, g_unmapRaw { 0 };
std::atomic<int> g_unmapPeeks { 0 };
// A written range bigger than this is a bulk upload, not a constant block; checking it on the game's thread
// would be the sweep by another name.
constexpr UINT64 kMaxWrittenCheck = 256 * 1024;

void ConstantCapture_Dx12::SetWatchPattern(const float* pattern, size_t count, float rel,
                                           bool (*accept)(const uint8_t*))
{
    std::lock_guard<std::mutex> lock(g_lock);
    g_watchCount = 0;
    if (pattern != nullptr && count > 0 && count <= 4)
    {
        std::memcpy(g_watch, pattern, count * sizeof(float));
        g_watchCount = count;
    }
    g_watchRel = rel;
    g_watchAccept = accept;
    g_writtenHits.clear();
    g_watchSet.store(g_watchCount > 0, std::memory_order_release);
}

void ConstantCapture_Dx12::SetRawMatchObserver(void (*observer)(const uint8_t*, uint64_t, uint64_t))
{
    g_rawObserver.store(observer, std::memory_order_release);
}

void ConstantCapture_Dx12::NoteWrittenRange(ID3D12Resource* resource, const D3D12_RANGE* written)
{
    g_unmapCalls.fetch_add(1, std::memory_order_relaxed);
    if (written == nullptr)
        return;
    g_unmapWithRange.fetch_add(1, std::memory_order_relaxed);
    if (resource == nullptr || written->End <= written->Begin)
        return;
    if (written->End - written->Begin > kMaxWrittenCheck)
    {
        g_unmapTooBig.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    if (!g_watchSet.load(std::memory_order_acquire))
        return;

    std::lock_guard<std::mutex> lock(g_lock);
    const auto it = g_byResource.find(resource);
    const auto mit = it != g_byResource.end() ? g_mappings.find(it->second) : g_mappings.end();
    if (mit == g_mappings.end() || mit->second.cpu == nullptr)
    {
        g_unmapNoMapping.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    const Mapping& m = mit->second;
    if (written->End > m.size)
    {
        g_unmapPastEnd.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    g_unmapChecked.fetch_add(1, std::memory_order_relaxed);

    // The first few checked ranges, in the clear: their size, their offset in the mapping, the mapping's
    // size, and the first four floats. "0 hits" with these in the log is a fact about the bytes; without them
    // it is a fact about nothing.
    if (g_unmapPeeks.load(std::memory_order_relaxed) < 6)
    {
        const int n = g_unmapPeeks.fetch_add(1, std::memory_order_relaxed);
        if (n < 6)
        {
            const float* f = (const float*) (m.cpu + written->Begin);
            const size_t words = (size_t) std::min<UINT64>(4, (written->End - written->Begin) / 4);
            LOG_INFO("Constant capture: Unmap peek {}: written [{:X}, {:X}) of a {} byte mapping (resource {:X}); "
                     "first floats {:.6g} {:.6g} {:.6g} {:.6g}; watching for [{:.6g}, {:.6g}, {:.6g}, {:.6g}]",
                     n + 1, (size_t) written->Begin, (size_t) written->End, m.size, (size_t) resource,
                     words > 0 ? f[0] : 0.0f, words > 1 ? f[1] : 0.0f, words > 2 ? f[2] : 0.0f,
                     words > 3 ? f[3] : 0.0f, g_watch[0], g_watch[1], g_watch[2], g_watch[3]);
        }
    }

    // Only quads that START inside the written range, 16-byte aligned, the same test the sweep makes.
    const UINT64 begin = (written->Begin + 15) & ~(UINT64) 15;
    const UINT64 last = written->End >= g_watchCount * sizeof(float) ? written->End - g_watchCount * sizeof(float) : 0;
    for (UINT64 off = begin; off <= last && off + g_watchCount * sizeof(float) <= m.size; off += 16)
    {
        const float* v = (const float*) (m.cpu + off);
        bool ok = true;
        for (size_t i = 0; i < g_watchCount && ok; ++i)
        {
            const float a = v[i], b = g_watch[i];
            if (a != a || b == 0.0f)
                ok = false;
            else if (std::abs(a - b) > std::abs(b) * g_watchRel)
                ok = false;
        }
        if (!ok)
            continue;
        g_unmapRaw.fetch_add(1, std::memory_order_relaxed);
        const uint8_t* quad = m.cpu + off;
        if (const auto observer = g_rawObserver.load(std::memory_order_acquire))
            observer(quad, off, m.size - off);
        if (off < 512 || off + 128 > m.size)
            continue; // the validator reads 512 bytes before the quad and ~64 after; keep it inside the mapping
        if (g_watchAccept != nullptr && !g_watchAccept(quad))
            continue;
        if (std::find(g_writtenHits.begin(), g_writtenHits.end(), quad) != g_writtenHits.end())
            continue;
        if (g_writtenHits.size() >= kMaxWrittenHits)
            return;
        g_writtenHits.push_back(quad);
        g_unmapHits.fetch_add(1, std::memory_order_relaxed);
    }
}

size_t ConstantCapture_Dx12::TakeWrittenHits(const uint8_t** out, size_t maxOut)
{
    if (out == nullptr || maxOut == 0)
        return 0;
    std::lock_guard<std::mutex> lock(g_lock);
    size_t n = 0;
    for (; n < g_writtenHits.size() && n < maxOut; ++n)
        out[n] = g_writtenHits[n];
    g_writtenHits.clear();
    return n;
}

ConstantCapture_Dx12::WrittenRangeStatsOut ConstantCapture_Dx12::WrittenRangeStats()
{
    WrittenRangeStatsOut s;
    s.calls = g_unmapCalls.load(std::memory_order_relaxed);
    s.withRange = g_unmapWithRange.load(std::memory_order_relaxed);
    s.checked = g_unmapChecked.load(std::memory_order_relaxed);
    s.hits = g_unmapHits.load(std::memory_order_relaxed);
    s.tooBig = g_unmapTooBig.load(std::memory_order_relaxed);
    s.noMapping = g_unmapNoMapping.load(std::memory_order_relaxed);
    s.pastEnd = g_unmapPastEnd.load(std::memory_order_relaxed);
    s.rawMatches = g_unmapRaw.load(std::memory_order_relaxed);
    return s;
}

bool ConstantCapture_Dx12::MappingOf(const uint8_t* ptr, const uint8_t** begin, uint64_t* size)
{
    if (ptr == nullptr || begin == nullptr || size == nullptr)
        return false;
    std::lock_guard<std::mutex> lock(g_lock);
    for (const auto& entry : g_mappings)
    {
        const Mapping& m = entry.second;
        if (m.cpu == nullptr || ptr < m.cpu || ptr >= m.cpu + m.size)
            continue;
        *begin = m.cpu;
        *size = m.size;
        return true;
    }
    return false;
}

bool ConstantCapture_Dx12::GpuAddressOf(const uint8_t* ptr, uint64_t* gpuAddress)
{
    if (ptr == nullptr || gpuAddress == nullptr)
        return false;
    std::lock_guard<std::mutex> lock(g_lock);
    for (const auto& entry : g_mappings)
    {
        const Mapping& m = entry.second;
        if (m.cpu == nullptr || ptr < m.cpu || ptr >= m.cpu + m.size)
            continue;
        *gpuAddress = m.base + (uint64_t) (ptr - m.cpu);
        return true;
    }
    return false;
}

bool ConstantCapture_Dx12::ReadAtGpuAddress(uint64_t gpuAddress, void* out, size_t bytes)
{
    if (gpuAddress == 0 || out == nullptr || bytes == 0)
        return false;
    // Copied out under g_lock, which ForgetResource also takes before the resource dies: no pointer outlives the
    // call, so this can never read a freed page (the fourth menu crash was a cached one).
    std::lock_guard<std::mutex> lock(g_lock);
    auto it = g_mappings.upper_bound(gpuAddress);
    if (it == g_mappings.begin())
        return false;
    --it;
    const Mapping& m = it->second;
    if (m.cpu == nullptr || gpuAddress < m.base || gpuAddress + bytes > m.base + m.size)
        return false;
    std::memcpy(out, m.cpu + (gpuAddress - m.base), bytes);
    return true;
}

void ConstantCapture_Dx12::NoteUsefulPointer(const uint8_t* ptr)
{
    if (ptr == nullptr)
        return;
    std::lock_guard<std::mutex> lock(g_lock);
    // A CPU pointer, not an address, so this is a walk -- once per committed candidate per scan, not hot.
    for (auto& entry : g_mappings)
    {
        Mapping& m = entry.second;
        if (m.cpu == nullptr || ptr < m.cpu || ptr >= m.cpu + m.size)
            continue;
        m.useful = true;
        if (std::find(g_usefulSizes.begin(), g_usefulSizes.end(), m.size) == g_usefulSizes.end() &&
            g_usefulSizes.size() < 16)
            g_usefulSizes.push_back(m.size);
        return;
    }
}

std::atomic<uint64_t> g_forgets { 0 };
std::atomic<uint64_t> g_rebinds { 0 };
uint64_t ConstantCapture_Dx12::ForgetCount() { return g_forgets.load(std::memory_order_relaxed); }
uint64_t ConstantCapture_Dx12::RebindCount() { return g_rebinds.load(std::memory_order_relaxed); }

std::atomic<uint64_t> g_seenAtCreate { 0 };
std::atomic<uint64_t> g_seenAtMap { 0 };
std::atomic<uint64_t> g_readAttempts { 0 };
std::atomic<uint64_t> g_readMisses { 0 };

uint64_t ConstantCapture_Dx12::ReadAttempts() { return g_readAttempts.load(std::memory_order_relaxed); }
uint64_t ConstantCapture_Dx12::ReadMisses() { return g_readMisses.load(std::memory_order_relaxed); }

uint64_t ConstantCapture_Dx12::BuffersSeenAtCreate() { return g_seenAtCreate.load(std::memory_order_relaxed); }
uint64_t ConstantCapture_Dx12::BuffersSeenAtMap() { return g_seenAtMap.load(std::memory_order_relaxed); }

uint64_t ConstantCapture_Dx12::MappedCount()
{
    std::lock_guard<std::mutex> lock(g_lock);
    return (uint64_t) g_mappings.size();
}

// GUARDRAIL, the same shape as EvaluateScope in DuniaTaa_Dx12 and for the same reason: this function is called
// FROM the ID3D12Resource::Map hook and then calls Map itself, so without it the hook re-enters this, which maps
// again, for ever. That is a stack overflow on the first buffer the game maps, and it crashed FC6 before the
// startup logging had finished -- exactly the recursion NR's foveated crop copy caused inside the splice, which
// is why the pattern is already in this codebase and should have been reached for first.
//
// The flag is per thread because other threads mapping their own buffers are the game's business, not ours.
thread_local int g_inOwnMap = 0;

struct OwnMapScope
{
    OwnMapScope() { ++g_inOwnMap; }
    ~OwnMapScope() { --g_inOwnMap; }
};

void ConstantCapture_Dx12::NoteBuffer(const D3D12_RESOURCE_DESC* desc, const D3D12_HEAP_PROPERTIES& props,
                                      ID3D12Resource* resource)
{
    // Cheapest tests first: this sits in the resource creation path, which a level load hammers.
    if (desc == nullptr || resource == nullptr)
        return;
    if (desc->Dimension != D3D12_RESOURCE_DIMENSION_BUFFER || desc->Width == 0)
        return;
    // Only the heaps the CPU can read through a mapping -- but "plain upload heaps are where the constants are"
    // turned out to be wrong for Dunia, and it cost a capture: 3,999 CBV addresses and not one byte of contents,
    // because every one of them resolved to no mapping. A CUSTOM heap is CPU-visible or not according to its
    // CPUPageProperty, which is one field to read rather than a guess.
    const bool cpuVisible = props.Type == D3D12_HEAP_TYPE_UPLOAD || props.Type == D3D12_HEAP_TYPE_READBACK ||
                            (props.Type == D3D12_HEAP_TYPE_CUSTOM &&
                             (props.CPUPageProperty == D3D12_CPU_PAGE_PROPERTY_WRITE_COMBINE ||
                              props.CPUPageProperty == D3D12_CPU_PAGE_PROPERTY_WRITE_BACK));
    if (!cpuVisible)
        return;
    if (!Wanted())
        return;

    g_seenAtCreate.fetch_add(1, std::memory_order_relaxed);

    const auto base = resource->GetGPUVirtualAddress();
    if (base == 0)
        return;

    // The cap is checked before mapping, not after: mapping first and then discarding the result on a full table
    // left a mapping behind with nothing recording it.
    {
        std::lock_guard<std::mutex> lock(g_lock);
        if (g_mappings.size() >= kMaxMappings && g_mappings.find(base) == g_mappings.end())
            return;
    }

    // Mapped once and never unmapped. Map is reference counted and does not keep the resource alive, so this adds
    // a mapping to a buffer the game very likely keeps mapped anyway rather than holding anything open. Reads only
    // ever happen at a bind, when the game itself is using the buffer, so the mapping is live by construction.
    void* cpu = nullptr;
    const D3D12_RANGE readNothing { 0, 0 };
    {
        // Under the same guard as NoteMappedBuffer's: with the Map hook installed, this call would otherwise go
        // straight back through it and map the same buffer a second time.
        OwnMapScope scope;
        if (FAILED(resource->Map(0, &readNothing, &cpu)) || cpu == nullptr)
            return;
    }

    std::lock_guard<std::mutex> lock(g_lock);
    RecordMappingLocked(base, desc->Width, (uint8_t*) cpu, resource);
}

void ConstantCapture_Dx12::NoteRootCbv(bool graphics, UINT rootParameterIndex, D3D12_GPU_VIRTUAL_ADDRESS location)
{
    if (location == 0)
        return;

    // Says once, whether or not a capture is running, that a root CBV bind reached us at all. Without it an empty
    // constants.txt has two very different explanations that look identical in the log: the hooks never installed,
    // or they did and this game binds its constants through descriptor tables instead. One line settles which.
    static std::atomic<bool> announced { false };
    if (!announced.exchange(true))
    {
        LOG_INFO("Constant capture: root CBV binds are reaching the hook ({} root parameter {}), so the hook is "
                 "installed and live",
                 graphics ? "graphics" : "compute", rootParameterIndex);
    }

    if (!g_capturing.load(std::memory_order_relaxed))
        return;

    std::lock_guard<std::mutex> lock(g_lock);
    if (!g_capturing || g_records.size() >= kMaxRecords)
        return;
    if (CountForSlot(g_frame, graphics, rootParameterIndex) >= kMaxRecordsPerSlot)
        return;

    // The greatest base at or below this address, then a range check: a root CBV can point part way into a big
    // shared upload buffer, which is exactly how Dunia sub-allocates its per-draw constants.
    auto it = g_mappings.upper_bound(location);
    if (it == g_mappings.begin())
        return;
    --it;

    const Mapping& mapping = it->second;
    if (location < mapping.base || location >= mapping.base + mapping.size)
        return;

    const UINT64 offset = location - mapping.base;
    UINT64 bytes = mapping.size - offset;
    if (bytes > kMaxBytesPerCb)
        bytes = kMaxBytesPerCb;
    bytes &= ~(UINT64) 3; // whole 32-bit words, which is all the constant buffer layouts here are made of
    if (bytes < 64)
        return; // too small to hold a 4x4 matrix, so of no use to what this capture is for

    std::vector<uint32_t> words((size_t) (bytes / 4));
    std::memcpy(words.data(), mapping.cpu + offset, (size_t) bytes);

    if (AlreadyHave(g_frame, graphics, rootParameterIndex, words.data(), words.size()))
        return;

    g_records.push_back(Record { g_frame, graphics, rootParameterIndex, location, std::move(words) });
}

void ConstantCapture_Dx12::NoteMappedBuffer(ID3D12Resource* resource)
{
    if (g_inOwnMap > 0 || resource == nullptr || !Wanted())
        return;

    const auto desc = resource->GetDesc();
    if (desc.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER || desc.Width < 64)
        return;

    const auto base = resource->GetGPUVirtualAddress();
    if (base == 0)
        return;

    g_seenAtMap.fetch_add(1, std::memory_order_relaxed);

    {
        std::lock_guard<std::mutex> lock(g_lock);
        // Already have this one. Checked before mapping, the same way NoteBuffer does it: mapping first and
        // discarding the result leaves a mapping behind with nothing recording it.
        //
        // But only when it is the SAME resource. A placed resource can be released and a new one created over
        // the same heap range -- same address, same size, different object -- and the game may well map the
        // new one before releasing the old. Treating that as a duplicate meant the old resource's death then
        // erased the mapping and the live buffer had no record until the game happened to Map it again. That
        // is a way to reach "0 real per-view structs" with the scan reading nothing wrong.
        const auto it = g_mappings.find(base);
        if (it != g_mappings.end() && it->second.size == desc.Width)
        {
            if (it->second.resource == resource)
            {
                // A re-Map of a known buffer: the game is about to write into it. That recency is what puts
                // the constant rings at the front of the sweep.
                it->second.lastMapped = g_mapSerial.fetch_add(1, std::memory_order_relaxed) + 1;
                return;
            }
            g_rebinds.fetch_add(1, std::memory_order_relaxed);
        }
        if (g_mappings.size() >= kMaxMappings && it == g_mappings.end())
            return;
    }

    // Our own mapping, not the pointer the game was just handed: Map is reference counted, so this keeps the
    // mapping alive past the game's Unmap and cannot be left dangling by it.
    void* cpu = nullptr;
    const D3D12_RANGE readNothing { 0, 0 };
    {
        // The re-entry the hook would otherwise take straight back into this function.
        OwnMapScope scope;
        if (FAILED(resource->Map(0, &readNothing, &cpu)) || cpu == nullptr)
            return;
    }

    std::lock_guard<std::mutex> lock(g_lock);
    RecordMappingLocked(base, desc.Width, (uint8_t*) cpu, resource);
}

bool ConstantCapture_Dx12::ForgetResource(ID3D12Resource* resource)
{
    // Free when there is nothing mapped, which is every game but this one and every run before F6.
    if (resource == nullptr || g_mappingCount.load(std::memory_order_relaxed) == 0)
        return false;

    const uint8_t* deadBegin = nullptr;
    const uint8_t* deadEnd = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_lock);
        const auto it = g_byResource.find(resource);
        if (it == g_byResource.end())
            return false;

        const auto mapping = g_mappings.find(it->second);
        if (mapping != g_mappings.end())
        {
            deadBegin = mapping->second.cpu;
            deadEnd = deadBegin != nullptr ? deadBegin + mapping->second.size : nullptr;

            // Give back the mapping reference this module took. The resource is still alive at this point --
            // hkRelease calls us before the release that kills it -- so this is an ordinary Unmap, and it is
            // ours alone: Map is reference counted, and the game's own mapping is the game's business.
            if (deadBegin != nullptr)
                resource->Unmap(0, nullptr);
            g_mappings.erase(mapping);
        }
        g_byResource.erase(it);
        g_mappingCount.store(g_mappings.size(), std::memory_order_relaxed);
        g_forgets.fetch_add(1, std::memory_order_relaxed);
    }

    // Outside the lock, deliberately: ViewConstants::Update holds ITS lock while calling FindPattern here, so
    // taking that one while holding g_lock is the reverse order and deadlocks. The range is passed rather than
    // a bare "forget everything" so the sweep is only paid for when a cached pointer really was in this buffer.
    if (deadBegin != nullptr)
        DuniaTaa_Dx12::ForgetViewConstants(deadBegin, deadEnd);
    return true;
}

bool ConstantCapture_Dx12::ReadAt(D3D12_GPU_VIRTUAL_ADDRESS location, UINT64 bytes, std::vector<uint32_t>& out)
{
    if (location == 0)
        return false;

    g_readAttempts.fetch_add(1, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lock(g_lock);

    // The same resolution NoteRootCbv does, and for the same reason: an address can point part way into a big
    // shared upload buffer, which is exactly how Dunia sub-allocates its per-draw constants.
    auto missed = [&](const char* why)
    {
        g_readMisses.fetch_add(1, std::memory_order_relaxed);
        static std::atomic<bool> announced { false };
        if (!announced.exchange(true))
            LOG_WARN("Constant capture: address {:X} could not be read ({}) -- {} CPU-visible buffers are "
                     "mapped. A capture full of CBV addresses and no contents means this, every time",
                     (size_t) location, why, (uint64_t) g_mappings.size());
        return false;
    };

    auto it = g_mappings.upper_bound(location);
    if (it == g_mappings.begin())
        return missed("below every mapped buffer");
    --it;

    const Mapping& mapping = it->second;
    if (location < mapping.base || location >= mapping.base + mapping.size)
        return missed("in no mapped buffer's range");

    const UINT64 offset = location - mapping.base;
    const UINT64 available = mapping.size - offset;
    if (bytes == 0 || bytes > available)
        bytes = available;
    if (bytes > kMaxBytesPerCb)
        bytes = kMaxBytesPerCb;
    bytes &= ~(UINT64) 3; // whole 32-bit words, which is all these layouts are made of
    if (bytes < 64)
        return false; // too small to hold a 4x4 matrix, so of no use to what this capture is for

    out.resize((size_t) (bytes / 4));
    std::memcpy(out.data(), mapping.cpu + offset, (size_t) bytes);
    return true;
}

void ConstantCapture_Dx12::Start()
{
    std::lock_guard<std::mutex> lock(g_lock);
    g_capturing = true;
    g_frame = 0;
    g_records.clear();
}

void ConstantCapture_Dx12::SetFrame(int frame)
{
    std::lock_guard<std::mutex> lock(g_lock);
    g_frame = frame;
}

void ConstantCapture_Dx12::Stop()
{
    std::lock_guard<std::mutex> lock(g_lock);
    g_capturing = false;
    g_frame = 0;
    g_records.clear();
}

std::string ConstantCapture_Dx12::Take()
{
    std::vector<Record> records;
    {
        std::lock_guard<std::mutex> lock(g_lock);
        records.swap(g_records);
    }

    if (records.empty())
    {
        LOG_INFO("Constant capture: no root constant buffers were recorded");
        return {};
    }

    std::string out = "Far Cry 6 (D3D12) root constant buffers, one block per bind kept per slot per frame.\n"
                      "Constants reached through a descriptor table are not here -- root CBVs only.\n"
                      "Read with tools\\capture\\parsecb.py; find the matrices with tools\\capture\\cbmat2.py.\n\n";

    for (const auto& record : records)
    {
        out += std::format("frame {} {} cb{} buffer {:X} {} bytes\n", record.frame,
                           record.graphics ? "graphics" : "compute", record.slot, (size_t) record.address,
                           record.words.size() * 4);

        for (size_t i = 0; i < record.words.size(); i += 4)
        {
            const size_t count = record.words.size() - i < 4 ? record.words.size() - i : 4;

            std::string floats;
            std::string hex;
            for (size_t j = 0; j < count; ++j)
            {
                floats += std::format("{:>13.6g}", AsFloat(record.words[i + j]));
                hex += std::format("{}{:08X}", j == 0 ? "" : " ", record.words[i + j]);
            }

            out += std::format("  {:>5}: {} | {}\n", i * 4, floats, hex);
        }
        out += "\n";
    }

    LOG_INFO("Constant capture: {} constant buffer binds recorded", records.size());
    return out;
}
