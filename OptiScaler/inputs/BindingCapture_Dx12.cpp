#include "pch.h"
#include "BindingCapture_Dx12.h"

#include "Config.h"
#include "Logger.h"

#include <inputs/ConstantCapture_Dx12.h>
#include <inputs/FrameCapture_Dx12.h>
#include <resource_tracking/ResTrack_Dx12.h>

#include <atomic>
#include <cstdint>
#include <cstring>
#include <format>
#include <mutex>
#include <string>
#include <vector>

namespace
{
std::mutex g_lock;

// Read before the lock. Every descriptor table bind in the frame comes through NoteTable, capture or not, and a
// frame has thousands of them, so the not-capturing path must cost one atomic load and nothing else.
std::atomic<bool> g_capturing { false };
int g_frame = 0;

// A table's length lives in the root signature, which is not available here, so each one is walked until the
// descriptors stop resolving or this many have been read. Dunia's tables are far shorter than this; the cost of
// overshooting is a few trailing entries belonging to whatever sits next in the heap.
constexpr UINT kMaxDescriptorsPerTable = 32;

// How many unresolved descriptors in a row end the scan. Anything ResTrack does not track reads as a hole, so
// this has to tolerate a few without running off into the next table's descriptors.
constexpr UINT kMaxConsecutiveHoles = 6;

// Constant buffers are dumped to this or their own size, whichever is smaller. FC5's
// per-viewport constant block is 1984 bytes.
constexpr UINT64 kMaxBytesPerCb = 4096;

constexpr size_t kMaxGroups = 512;
constexpr size_t kMaxCbDumps = 48;

// The eye shape, so a group can be judged against it. Set from the splice every frame.
UINT g_eyeW = 0;
UINT g_eyeH = 0;

// Block compressed formats are asset textures -- BC1..BC5 and BC6H/BC7. A 2048x2048 BC7 is somebody's material,
// never a render target, and the first run of this filled its whole budget with them.
bool IsBlockCompressed(DXGI_FORMAT format)
{
    const UINT f = (UINT) format;
    return (f >= 70 && f <= 84) || (f >= 94 && f <= 99);
}

// "Something drawn at screen shape", which is what every post-processing pass reads and no material texture is.
// Deliberately loose on size: motion and depth are often half or quarter resolution, and FC5's motion was
// half-res. Deliberately strict on aspect and squareness, which is what throws out the shadow and material atlases.
bool IsScreenish(UINT width, UINT height, DXGI_FORMAT format)
{
    if (g_eyeW == 0 || g_eyeH == 0 || width == 0 || height == 0)
        return false;
    if (width == height)
        return false;
    if (IsBlockCompressed(format))
        return false;

    const double aspect = (double) width / height;
    const double eyeAspect = (double) g_eyeW / g_eyeH;
    if (aspect < eyeAspect * 0.85 || aspect > eyeAspect * 1.15)
        return false;

    const double area = (double) width * height;
    const double eyeArea = (double) g_eyeW * g_eyeH;
    return area >= eyeArea / 32.0 && area <= eyeArea * 1.1;
}

struct Slot
{
    UINT index = 0;
    ID3D12Resource* resource = nullptr;
    UINT64 width = 0;
    UINT height = 0;
    DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;
    UINT type = 0;
    std::vector<uint32_t> words; // constant buffer contents, when this slot turned out to be one
    // CBV slots only. A constant buffer view names no resource -- it is a GPU virtual address and a size -- so
    // `resource` stays null and the address goes here.
    UINT64 address = 0;
    // The address this slot held on each frame it was seen, which is the whole question a CBV capture is asked:
    // if Dunia sub-allocates its per-draw constants from a ring, the address moves every frame and the
    // descriptor has to be resolved every frame (which costs the heap mirroring); if it is stable, it can be
    // learned once and read from for free afterwards. Only changes are appended, so a stable address is one row.
    std::vector<std::pair<int, UINT64>> addressByFrame;
};
constexpr size_t kMaxAddressHistory = 12;

struct Group
{
    int frame = 0;
    bool graphics = true;
    UINT rootParameter = 0;
    std::vector<Slot> slots;
};
std::vector<Group> g_groups;
size_t g_cbDumps = 0;
// Addresses whose contents have already been written out. The budget is small and the trailing slots a table
// walk picks up are nearly all the same shared default buffer: in the 2152 capture, 43 of 48 dumps were one
// address, which starved every pass of its own constants. The bytes at one address are the same bytes however
// many tables point at it.
std::unordered_set<UINT64> g_cbDumped;

const char* TypeName(UINT type)
{
    switch (type)
    {
    case SRV:
        return "srv";
    case RTV:
        return "rtv";
    case UAV:
        return "uav";
    case DSV:
        return "dsv";
    case CBV:
        return "cbv";
    default:
        return "?";
    }
}

float AsFloat(uint32_t word)
{
    float value = 0.0f;
    std::memcpy(&value, &word, sizeof(value));
    return value;
}

// Two groups are the same when they name the same resources in the same slots. A pass that binds the same table
// for every draw of the frame would otherwise fill the list on its own.
bool SameResources(const Group& a, const Group& b)
{
    if (a.graphics != b.graphics || a.rootParameter != b.rootParameter || a.slots.size() != b.slots.size())
        return false;
    for (size_t i = 0; i < a.slots.size(); ++i)
    {
        // Deliberately NOT comparing the CBV address: if the game rings its constants, every frame would look
        // like a new group, the budget would go on duplicates of one pass, and the very thing being measured
        // would stop the measurement. Addresses are merged into the group that already matched instead.
        if (a.slots[i].resource != b.slots[i].resource || a.slots[i].index != b.slots[i].index ||
            a.slots[i].type != b.slots[i].type)
            return false;
    }
    return true;
}

// A buffer in an upload heap can be read straight from a mapping. Anything else would need a GPU copy and a
// barrier, which is not worth it for a diagnostic -- the per-view constants of a Dunia frame are CPU-written.
bool ReadConstantBuffer(ID3D12Resource* resource, std::vector<uint32_t>& out)
{
    if (resource == nullptr)
        return false;

    const auto desc = resource->GetDesc();
    if (desc.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER || desc.Width < 64)
        return false;

    D3D12_HEAP_PROPERTIES heap {};
    D3D12_HEAP_FLAGS heapFlags = D3D12_HEAP_FLAG_NONE;
    if (FAILED(resource->GetHeapProperties(&heap, &heapFlags)))
        return false;
    if (heap.Type != D3D12_HEAP_TYPE_UPLOAD && heap.Type != D3D12_HEAP_TYPE_READBACK)
        return false;

    UINT64 bytes = desc.Width < kMaxBytesPerCb ? desc.Width : kMaxBytesPerCb;
    bytes &= ~(UINT64) 3;
    if (bytes < 64)
        return false;

    void* mapped = nullptr;
    const D3D12_RANGE readNothing { 0, 0 };
    if (FAILED(resource->Map(0, &readNothing, &mapped)) || mapped == nullptr)
        return false;

    out.resize((size_t) (bytes / 4));
    std::memcpy(out.data(), mapped, (size_t) bytes);

    const D3D12_RANGE nothingWritten { 0, 0 };
    resource->Unmap(0, &nothingWritten);
    return true;
}
} // namespace

bool BindingCapture_Dx12::Wanted() { return Config::Instance()->DuniaTaaDiagnostics.value_or_default(); }

void BindingCapture_Dx12::NoteTable(bool graphics, UINT rootParameterIndex, D3D12_GPU_DESCRIPTOR_HANDLE base,
                                    ID3D12GraphicsCommandList* cmd)
{
    if (base.ptr == 0 || !g_capturing.load(std::memory_order_relaxed))
        return;

    // Resolved outside our lock: ResTrack takes its own, and holding both would invite a lock order problem.
    auto heap = ResTrack_Dx12::GetTableHeap(graphics, (SIZE_T) base.ptr);
    if (heap == nullptr || heap->increment == 0)
        return;

    Group group;
    group.graphics = graphics;
    group.rootParameter = rootParameterIndex;

    // Holes are walked past, not stopped at. ResTrack only records what CheckResource accepts, and that requires
    // a TEXTURE2D -- so every constant buffer in a table is a hole, and a table that starts with one (most of
    // them) would otherwise come back with nothing at all. Stopping only after a run of holes keeps the scan from
    // dragging in whatever sits further along the heap.
    UINT consecutiveHoles = 0;
    for (UINT i = 0; i < kMaxDescriptorsPerTable; ++i)
    {
        const SIZE_T handle = (SIZE_T) base.ptr + (SIZE_T) i * heap->increment;
        if (handle >= heap->gpuEnd)
            break;

        ResourceInfo info {};
        const bool resolved = heap->GetByGpuHandle(handle, info);
        // A constant buffer view names no resource, so buffer == nullptr is its normal state rather than a hole.
        // Before CreateConstantBufferView was hooked this branch could never be taken and every constant slot in
        // every table was skipped in silence -- which is why no capture ever contained one.
        const bool isCbv = resolved && info.buffer == nullptr && info.type == CBV && info.cbvAddress != 0;
        if (!resolved || (info.buffer == nullptr && !isCbv))
        {
            if (++consecutiveHoles >= kMaxConsecutiveHoles)
                break;
            continue;
        }
        consecutiveHoles = 0;

        Slot slot;
        slot.index = i;
        slot.resource = info.buffer;
        slot.width = info.width;
        slot.height = info.height;
        slot.format = info.format;
        slot.type = (UINT) info.type;
        if (isCbv)
        {
            slot.address = info.cbvAddress;
            slot.addressByFrame.push_back({ g_frame, info.cbvAddress });
        }
        group.slots.push_back(std::move(slot));
    }

    if (group.slots.empty())
        return;

    // Only groups that touch something at screen shape are kept. Without this the budget is spent entirely on the
    // geometry passes -- the first run recorded 256 groups, every one of them from frame 0, dominated by 2048x2048
    // and 4096x4096 BC-compressed material textures, and never reached the post-processing passes late in the
    // frame where a TAA resolve lives. A group that reads no render target cannot be the one we are looking for.
    bool screenish = false;
    for (const auto& slot : group.slots)
    {
        if (slot.height != 0 && IsScreenish((UINT) slot.width, slot.height, slot.format))
        {
            screenish = true;
            break;
        }
    }
    if (!screenish)
        return;

    // A UAV bind is the only chance to reach a texture a compute pass writes -- such a texture never reaches
    // OMSetRenderTargets, so the render-target path cannot see it at all. Only *noted* here: the copy waits
    // for the dispatch that writes it, because a table bind says nothing about the resource's state.
    // Done before this module's own lock is taken: the capture has its own, and holding both would invite a
    // lock order problem.
    for (const auto& slot : group.slots)
    {
        if (slot.type == UAV && slot.height != 0 && IsScreenish((UINT) slot.width, slot.height, slot.format))
            FrameCapture_Dx12::NoteUavBound(cmd, slot.resource);
    }

    std::lock_guard<std::mutex> lock(g_lock);
    if (!g_capturing || g_groups.size() >= kMaxGroups)
        return;

    group.frame = g_frame;
    for (auto& seen : g_groups)
    {
        if (!SameResources(seen, group))
            continue;
        // Already have this pass. Keep one thing from the repeat: whether its constants moved. That is the
        // question the whole CBV capture exists to answer, and it can only be seen across frames.
        for (size_t i = 0; i < seen.slots.size() && i < group.slots.size(); ++i)
        {
            auto& slot = seen.slots[i];
            const UINT64 address = group.slots[i].address;
            if (address == 0 || slot.addressByFrame.size() >= kMaxAddressHistory)
                continue;
            if (!slot.addressByFrame.empty() && slot.addressByFrame.back().second == address)
                continue;
            slot.addressByFrame.push_back({ g_frame, address });
        }
        return;
    }

    // Constant buffers are read only for groups that are actually being kept, and only while there is budget --
    // a Map and a memcpy per slot on a recording thread is not free.
    for (auto& slot : group.slots)
    {
        if (g_cbDumps >= kMaxCbDumps)
            break;
        if (slot.height != 0)
            continue; // a texture, not a buffer
        const UINT64 key = slot.address != 0 ? slot.address : (UINT64) (size_t) slot.resource;
        if (key == 0 || !g_cbDumped.insert(key).second)
            continue; // already written out; the same address holds the same bytes
        // A CBV carries an address, not a resource, so its bytes come out of the persistent mapping
        // ConstantCapture_Dx12 already keeps of every CPU-visible buffer. `width` is the view's SizeInBytes.
        const bool read = slot.address != 0
                              ? ConstantCapture_Dx12::ReadAt(slot.address, slot.width, slot.words)
                              : ReadConstantBuffer(slot.resource, slot.words);
        if (read)
            ++g_cbDumps;
    }

    g_groups.push_back(std::move(group));
}

void BindingCapture_Dx12::Start()
{
    std::lock_guard<std::mutex> lock(g_lock);
    g_capturing = true;
    g_frame = 0;
    g_groups.clear();
    g_cbDumps = 0;
    g_cbDumped.clear();
}

void BindingCapture_Dx12::SetEyeShape(UINT width, UINT height)
{
    // Plain stores of two values the filter reads; a torn pair would cost one misjudged group, not correctness.
    g_eyeW = width;
    g_eyeH = height;
}

void BindingCapture_Dx12::SetFrame(int frame)
{
    std::lock_guard<std::mutex> lock(g_lock);
    g_frame = frame;
    // The budget is per FRAME, not per capture. Deduplicating by address freed 43 of 48 slots and frame 0 simply
    // took all of them anyway -- so the motion pass, whose group turns up on frame 1, still got nothing. A pass
    // that only runs on one eye is invisible to a budget the first frame can exhaust.
    //
    // The dumped-address set is deliberately NOT cleared here: the junk default buffer that trailing table slots
    // point at is the same address every frame and would otherwise eat a slot in each of them.
    g_cbDumps = 0;
}

void BindingCapture_Dx12::Stop()
{
    std::lock_guard<std::mutex> lock(g_lock);
    g_capturing = false;
    g_frame = 0;
    g_groups.clear();
    g_cbDumps = 0;
}

std::string BindingCapture_Dx12::Take()
{
    std::vector<Group> groups;
    size_t cbDumps = 0;
    {
        std::lock_guard<std::mutex> lock(g_lock);
        groups.swap(g_groups);
        // Read before the reset, not after: the summary at the bottom of this function used to print the
        // already-zeroed counter and reported "0 constant buffers dumped" over a file holding 48 of them.
        // One more unreadable zero, in the very line added to make zeros readable.
        cbDumps = g_cbDumps;
        g_cbDumps = 0;
        g_cbDumped.clear();
    }

    if (groups.empty())
    {
        LOG_INFO("Binding capture: no descriptor tables were resolved");
        return {};
    }

    std::string out =
        "Far Cry 6 (D3D12) descriptor table bindings, one block per distinct group of resources bound together.\n"
        "A group is what one pass had bound at once: a TAA resolve shows colour, history, depth and motion\n"
        "together, so a group of eye-sized textures with a two-channel or packed one among them is the motion\n"
        "buffer -- identified by what reads it, not by what its descriptor looks like.\n"
        "Tables carry no length, so each is walked until the descriptors stop resolving; trailing slots may belong\n"
        "to whatever sits next in the heap. Constant buffer slots are dumped inline in the format\n"
        "tools\\capture\\parsecb.py reads, so cbmat2.py can be run straight over this file.\n"
        "\n"
        "A cbv row names a GPU virtual address, not a resource: that is all a constant buffer view holds. The\n"
        "'address by frame' line under it is the point of capturing these -- ONE entry means the address never\n"
        "moved while the pass was seen, so it can be learned once and read from for free afterwards; SEVERAL\n"
        "mean the game rings its constants and the descriptor has to be resolved every frame, which costs the\n"
        "heap mirroring (~45% of the frame rate). If no cbv rows appear at all, check WantsCbvTracking() was\n"
        "true when the hooks went on before concluding the game binds none.\n\n";

    size_t textures = 0;
    size_t buffers = 0;
    size_t cbvs = 0;
    for (const auto& group : groups)
    {
        out += std::format("group frame {} {} root {} ({} slots)\n", group.frame,
                           group.graphics ? "graphics" : "compute", group.rootParameter, group.slots.size());

        for (const auto& slot : group.slots)
        {
            if (slot.height != 0)
            {
                ++textures;
                out += std::format("  [{:>2}] {:X} {}x{} fmt {} {}\n", slot.index, (size_t) slot.resource,
                                   slot.width, slot.height, (UINT) slot.format, TypeName(slot.type));
                continue;
            }

            ++buffers;
            if (slot.address != 0)
            {
                ++cbvs;
                out += std::format("  [{:>2}] cbv va {:X} {} bytes\n", slot.index, slot.address, slot.width);
                // The answer to "can this be learned once, or must the descriptor be resolved every frame":
                // one row means the address never moved across the frames this group was seen on.
                out += std::format("       address by frame ({} distinct):", slot.addressByFrame.size());
                for (const auto& entry : slot.addressByFrame)
                    out += std::format(" f{}={:X}", entry.first, entry.second);
                out += "\n";
            }
            else
            {
                out += std::format("  [{:>2}] {:X} buffer {} bytes {}\n", slot.index, (size_t) slot.resource,
                                   slot.width, TypeName(slot.type));
            }

            if (slot.words.empty())
                continue;

            // The header parsecb.py matches on, so a constant buffer found through a table reads exactly like one
            // found through a root CBV would have.
            out += std::format("frame {} {} cb{} buffer {:X} {} bytes\n", group.frame,
                               group.graphics ? "graphics" : "compute", group.rootParameter,
                               slot.address != 0 ? (size_t) slot.address : (size_t) slot.resource,
                               slot.words.size() * 4);

            for (size_t i = 0; i < slot.words.size(); i += 4)
            {
                const size_t count = slot.words.size() - i < 4 ? slot.words.size() - i : 4;

                std::string floats;
                std::string hex;
                for (size_t j = 0; j < count; ++j)
                {
                    floats += std::format("{:>13.6g}", AsFloat(slot.words[i + j]));
                    hex += std::format("{}{:08X}", j == 0 ? "" : " ", slot.words[i + j]);
                }

                out += std::format("  {:>5}: {} | {}\n", i * 4, floats, hex);
            }
        }
        out += "\n";
    }

    // The three numbers that tell a blind spot from a fact, side by side: how many CBVs the game created at all,
    // how many of those landed in a heap we track, and how many reached a table we walked. Zero in the first
    // column means the game does not use them; zero only in the last means we are not looking where it binds.
    LOG_INFO("Binding capture: {} distinct groups, {} texture slots, {} buffer slots ({} of them constant buffer "
             "views). CreateConstantBufferView called {} times, {} recorded into a tracked heap",
             groups.size(), textures, buffers, cbvs, ResTrack_Dx12::CbvCreateCount(),
             ResTrack_Dx12::CbvRecordedCount());
    LOG_INFO("Binding capture: {} constant buffers dumped; {} CPU-visible buffers mapped, {} address reads "
             "tried, {} found no mapping. Buffers offered: {} at creation, {} at Map",
             cbDumps, ConstantCapture_Dx12::MappedCount(), ConstantCapture_Dx12::ReadAttempts(),
             ConstantCapture_Dx12::ReadMisses(), ConstantCapture_Dx12::BuffersSeenAtCreate(),
             ConstantCapture_Dx12::BuffersSeenAtMap());
    return out;
}
