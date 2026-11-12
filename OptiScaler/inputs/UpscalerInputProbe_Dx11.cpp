#include "pch.h"

#include "Config.h"
#include "Logger.h"
#include "NVNGX_Parameter.h"

#include "UpscalerInputProbe_Dx11.h"

#include <d3d11.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace
{
std::mutex g_mutex;
uint64_t g_evaluations = 0;
bool g_censusStarted = false;

// The per-window pattern. Jitter positions are compared at 1/1024 px.
struct Window
{
    uint64_t evaluations = 0;
    uint64_t resets = 0;
    uint64_t deferred = 0;
    uint64_t zeroJitter = 0;
    std::set<std::pair<int, int>> positions;
    std::set<unsigned int> handles;
    float minX = 1e9f, maxX = -1e9f, minY = 1e9f, maxY = -1e9f;
    float mvScaleX = 0, mvScaleY = 0;
    bool mvScaleChanged = false;
    unsigned int renderW = 0, renderH = 0;
    bool renderChanged = false;
};
Window g_window;

// The first jitter values in order, logged once: the cycle's length and order (odd or even under alternate eyes).
std::vector<std::pair<float, float>> g_sequence;
constexpr size_t kSequence = 48;

ID3D11Resource* GetResource(NVSDK_NGX_Parameter* params, const char* name)
{
    ID3D11Resource* resource = nullptr;
    if (params->Get(name, &resource) != NVSDK_NGX_Result_Success)
        params->Get(name, (void**) &resource);
    return resource;
}

// Size and format of a parameter's texture, for the log. Compared and described only; no reference is kept.
std::string Describe(NVSDK_NGX_Parameter* params, const char* name)
{
    ID3D11Resource* resource = GetResource(params, name);
    if (resource == nullptr)
        return "none";
    ID3D11Texture2D* texture = nullptr;
    if (FAILED(resource->QueryInterface(IID_PPV_ARGS(&texture))) || texture == nullptr)
        return std::format("{:p} (not a Texture2D)", (void*) resource);
    D3D11_TEXTURE2D_DESC desc {};
    texture->GetDesc(&desc);
    texture->Release();
    return std::format("{}x{} fmt {} bind {:#x} ({:p})", desc.Width, desc.Height, (int) desc.Format, desc.BindFlags,
                       (void*) resource);
}

float GetFloat(NVSDK_NGX_Parameter* params, const char* name, float fallback = NAN)
{
    float value = fallback;
    params->Get(name, &value);
    return value;
}

unsigned int GetUInt(NVSDK_NGX_Parameter* params, const char* name, unsigned int fallback = 0)
{
    unsigned int value = fallback;
    params->Get(name, &value);
    return value;
}

void LogFirst(ID3D11DeviceContext* ctx, unsigned int handle, NVSDK_NGX_Parameter* params)
{
    LOG_INFO("Input probe: FIRST evaluate, handle {}, {} context", handle,
             ctx->GetType() == D3D11_DEVICE_CONTEXT_IMMEDIATE ? "immediate" : "DEFERRED");
    LOG_INFO("Input probe:   colour  {}", Describe(params, NVSDK_NGX_Parameter_Color));
    LOG_INFO("Input probe:   motion  {}", Describe(params, NVSDK_NGX_Parameter_MotionVectors));
    LOG_INFO("Input probe:   depth   {}", Describe(params, NVSDK_NGX_Parameter_Depth));
    LOG_INFO("Input probe:   output  {}", Describe(params, NVSDK_NGX_Parameter_Output));
    LOG_INFO("Input probe:   exposure {}, transparency mask {}", Describe(params, NVSDK_NGX_Parameter_ExposureTexture),
             Describe(params, NVSDK_NGX_Parameter_TransparencyMask));
    LOG_INFO("Input probe:   MV scale ({}, {}), jitter ({}, {}), reset {}, pre-exposure {}, exposure scale {}, "
             "sharpness {}",
             GetFloat(params, NVSDK_NGX_Parameter_MV_Scale_X), GetFloat(params, NVSDK_NGX_Parameter_MV_Scale_Y),
             GetFloat(params, NVSDK_NGX_Parameter_Jitter_Offset_X),
             GetFloat(params, NVSDK_NGX_Parameter_Jitter_Offset_Y), GetUInt(params, NVSDK_NGX_Parameter_Reset),
             GetFloat(params, NVSDK_NGX_Parameter_DLSS_Pre_Exposure),
             GetFloat(params, NVSDK_NGX_Parameter_DLSS_Exposure_Scale),
             GetFloat(params, NVSDK_NGX_Parameter_Sharpness));
    LOG_INFO("Input probe:   render subrect {}x{}, colour subrect base ({}, {}), MV subrect base ({}, {}), depth "
             "subrect base ({}, {})",
             GetUInt(params, NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Width),
             GetUInt(params, NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Height),
             GetUInt(params, NVSDK_NGX_Parameter_DLSS_Input_Color_Subrect_Base_X),
             GetUInt(params, NVSDK_NGX_Parameter_DLSS_Input_Color_Subrect_Base_Y),
             GetUInt(params, NVSDK_NGX_Parameter_DLSS_Input_MV_SubrectBase_X),
             GetUInt(params, NVSDK_NGX_Parameter_DLSS_Input_MV_SubrectBase_Y),
             GetUInt(params, NVSDK_NGX_Parameter_DLSS_Input_Depth_Subrect_Base_X),
             GetUInt(params, NVSDK_NGX_Parameter_DLSS_Input_Depth_Subrect_Base_Y));
}

void LogWindow()
{
    auto& w = g_window;
    std::string handles;
    for (auto h : w.handles)
        handles += std::format("{} ", h);
    LOG_INFO("Input probe: {} evaluations: {} distinct jitter positions, x [{:.4f}, {:.4f}] y [{:.4f}, {:.4f}], {} with "
             "zero jitter; {} resets; MV scale ({}, {}){}; render {}x{}{}; handles {}; {} on a deferred context",
             w.evaluations, w.positions.size(), w.minX, w.maxX, w.minY, w.maxY, w.zeroJitter, w.resets, w.mvScaleX,
             w.mvScaleY, w.mvScaleChanged ? " (CHANGED in the window)" : "", w.renderW, w.renderH,
             w.renderChanged ? " (CHANGED in the window)" : "", handles, w.deferred);
    w = Window {};
}

// ---- The census: which functions of the exe now jump somewhere else ----

std::string ModuleOf(const void* address)
{
    HMODULE module = nullptr;
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            (LPCWSTR) address, &module) ||
        module == nullptr)
        return std::format("{:p} (no module)", address);
    wchar_t path[MAX_PATH] {};
    GetModuleFileNameW(module, path, MAX_PATH);
    std::wstring name(path);
    const auto slash = name.find_last_of(L"\\/");
    if (slash != std::wstring::npos)
        name = name.substr(slash + 1);
    return std::format("{}+{:#x}", wstring_to_string(name), (uintptr_t) address - (uintptr_t) module);
}

bool ReadOwn(const void* address, void* out, size_t size)
{
    SIZE_T read = 0;
    return ReadProcessMemory(GetCurrentProcess(), address, out, size, &read) && read == size;
}

// Where a jump at `at` goes: E9 rel32, EB rel8, FF 25 [rip+disp32], or mov rax, imm64; jmp rax. Nothing otherwise.
const uint8_t* JumpTarget(const uint8_t* at)
{
    uint8_t b[16] {};
    if (!ReadOwn(at, b, sizeof(b)))
        return nullptr;
    if (b[0] == 0xE9)
    {
        int32_t rel;
        std::memcpy(&rel, b + 1, 4);
        return at + 5 + rel;
    }
    if (b[0] == 0xFF && b[1] == 0x25)
    {
        int32_t disp;
        std::memcpy(&disp, b + 2, 4);
        const uint8_t* slot = at + 6 + disp;
        const uint8_t* target = nullptr;
        return ReadOwn(slot, &target, sizeof(target)) ? target : nullptr;
    }
    if (b[0] == 0x48 && b[1] == 0xB8 && b[10] == 0xFF && b[11] == 0xE0)
    {
        const uint8_t* target = nullptr;
        std::memcpy(&target, b + 2, 8);
        return target;
    }
    return nullptr;
}

std::string Hex(const uint8_t* bytes, size_t n)
{
    std::string s;
    for (size_t i = 0; i < n; ++i)
        s += std::format("{:02x}", bytes[i]);
    return s;
}

// Executable, committed memory outside every module: where MinHook puts the relay a hooked function jumps to.
bool IsRelayMemory(const uint8_t* at)
{
    MEMORY_BASIC_INFORMATION mbi {};
    if (VirtualQuery(at, &mbi, sizeof(mbi)) != sizeof(mbi) || mbi.State != MEM_COMMIT || mbi.Type == MEM_IMAGE)
        return false;
    return (mbi.Protect & (PAGE_EXECUTE | PAGE_EXECUTE_READ | PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY)) != 0;
}

bool InModule(const void* address, HMODULE* module)
{
    return GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                              (LPCWSTR) address, module) &&
           *module != nullptr;
}

// The exe's code is stored encrypted on disk (some games: the whole .text differs from the file), so it cannot be
// compared with the file. Instead: every E9 rel32 in the exe's executable sections, as the code is now, whose target
// leaves the exe, and either lands in another module or in relay memory (executable, outside every module) that
// itself jumps into another module. Random E9 bytes inside other instructions practically never meet that.
void Census()
{
    const auto base = (const uint8_t*) GetModuleHandleW(nullptr);
    wchar_t path[MAX_PATH] {};
    GetModuleFileNameW(nullptr, path, MAX_PATH);
    const auto dos = (const IMAGE_DOS_HEADER*) base;
    const auto nt = (const IMAGE_NT_HEADERS64*) (base + dos->e_lfanew);
    const uint8_t* imageEnd = base + nt->OptionalHeader.SizeOfImage;
    const HMODULE self = (HMODULE) base;
    LOG_INFO("Input probe: census v2 of {} (loaded at {:p}, image {:#x} bytes): scanning the code as it is now for "
             "jumps out of the exe",
             wstring_to_string(path), (void*) base, nt->OptionalHeader.SizeOfImage);

    auto sections = IMAGE_FIRST_SECTION(nt);
    int reported = 0;
    size_t totalHooks = 0;
    std::vector<std::pair<std::string, size_t>> perModule;
    for (int i = 0; i < nt->FileHeader.NumberOfSections; ++i)
    {
        const auto& s = sections[i];
        if ((s.Characteristics & IMAGE_SCN_MEM_EXECUTE) == 0)
            continue;
        const std::string name((const char*) s.Name, strnlen((const char*) s.Name, 8));
        const size_t size = s.Misc.VirtualSize;
        std::vector<uint8_t> live(size + 16);
        size_t unreadable = 0;
        constexpr size_t kPage = 0x1000;
        for (size_t off = 0; off < size; off += kPage)
        {
            const size_t n = std::min(kPage, size - off);
            if (!ReadOwn(base + s.VirtualAddress + off, live.data() + off, n))
            {
                std::memset(live.data() + off, 0, n);
                unreadable += n;
            }
        }
        size_t candidates = 0, direct = 0, relayed = 0;
        for (size_t off = 0; off + 5 <= size; ++off)
        {
            if (live[off] != 0xE9)
                continue;
            int32_t rel;
            std::memcpy(&rel, live.data() + off + 1, 4);
            const uint8_t* at = base + s.VirtualAddress + off;
            const uint8_t* target = at + 5 + rel;
            if (target >= base && target < imageEnd)
                continue;
            ++candidates;
            HMODULE module = nullptr;
            std::string where;
            if (InModule(target, &module))
            {
                if (module == self)
                    continue;
                ++direct;
                where = ModuleOf(target);
            }
            else if (IsRelayMemory(target))
            {
                const uint8_t* next = JumpTarget(target);
                if (next == nullptr || !InModule(next, &module) || module == self)
                    continue;
                ++relayed;
                where = std::format("relay {:p} -> {}", (const void*) target, ModuleOf(next));
            }
            else
                continue;

            ++totalHooks;
            wchar_t ownerPath[MAX_PATH] {};
            GetModuleFileNameW(module, ownerPath, MAX_PATH);
            std::wstring ownerName(ownerPath);
            ownerName = ownerName.substr(ownerName.find_last_of(L"\\/") + 1);
            const std::string owner = wstring_to_string(ownerName);
            auto it = std::find_if(perModule.begin(), perModule.end(), [&](auto& e) { return e.first == owner; });
            if (it == perModule.end())
                perModule.push_back({ owner, 1 });
            else
                ++it->second;
            if (reported < 200)
            {
                LOG_INFO("Input probe: census: exe+{:#x} ({}) jumps to {}; bytes here {} | before {}",
                         (uintptr_t) (at - base), name, where, Hex(live.data() + off, std::min<size_t>(48, size - off)),
                         Hex(live.data() + (off >= 16 ? off - 16 : 0), off >= 16 ? 16 : off));
                ++reported;
            }
        }
        LOG_INFO("Input probe: census: section {} ({:#x} bytes): {} E9 jumps leave the exe; {} land in another module, "
                 "{} through relay memory{}",
                 name, size, candidates, direct, relayed,
                 unreadable ? std::format(", {} bytes UNREADABLE (not scanned)", unreadable) : std::string());
    }
    std::string summary;
    for (auto& [m, n] : perModule)
        summary += std::format("{} x{}, ", m, n);
    LOG_INFO("Input probe: census done: {} hooks ({} logged) by module: {}", totalHooks, reported,
             summary.empty() ? std::string("none -- if SekiroTSR is loaded this means the census is still blind") : summary);
}
} // namespace

namespace UpscalerInputProbe_Dx11
{
bool Enabled()
{
    static const bool enabled = Config::Instance()->UpscalerInputProbe.value_or_default();
    return enabled;
}

void OnEvaluate(ID3D11DeviceContext* ctx, unsigned int handle, NVSDK_NGX_Parameter* params)
{
    if (!Enabled() || ctx == nullptr || params == nullptr)
        return;
    std::lock_guard lock(g_mutex);
    if (g_evaluations++ == 0)
        LogFirst(ctx, handle, params);

    auto& w = g_window;
    ++w.evaluations;
    w.handles.insert(handle);
    if (ctx->GetType() != D3D11_DEVICE_CONTEXT_IMMEDIATE)
        ++w.deferred;
    if (GetUInt(params, NVSDK_NGX_Parameter_Reset) != 0)
        ++w.resets;

    const float jx = GetFloat(params, NVSDK_NGX_Parameter_Jitter_Offset_X, 0.0f);
    const float jy = GetFloat(params, NVSDK_NGX_Parameter_Jitter_Offset_Y, 0.0f);
    if (jx == 0.0f && jy == 0.0f)
        ++w.zeroJitter;
    w.positions.insert({ (int) std::lround(jx * 1024.0f), (int) std::lround(jy * 1024.0f) });
    w.minX = std::min(w.minX, jx);
    w.maxX = std::max(w.maxX, jx);
    w.minY = std::min(w.minY, jy);
    w.maxY = std::max(w.maxY, jy);
    if (g_sequence.size() < kSequence)
    {
        g_sequence.push_back({ jx, jy });
        if (g_sequence.size() == kSequence)
        {
            std::string s;
            for (auto& [x, y] : g_sequence)
                s += std::format("({:.3f},{:.3f}) ", x, y);
            LOG_INFO("Input probe: the first {} jitter values in order: {}", kSequence, s);
        }
    }

    const float sx = GetFloat(params, NVSDK_NGX_Parameter_MV_Scale_X, 0.0f);
    const float sy = GetFloat(params, NVSDK_NGX_Parameter_MV_Scale_Y, 0.0f);
    if (w.evaluations > 1 && (sx != w.mvScaleX || sy != w.mvScaleY))
        w.mvScaleChanged = true;
    w.mvScaleX = sx;
    w.mvScaleY = sy;

    // The render size at evaluate is the render subrect (the render size comes from the subrect).
    unsigned int rw = GetUInt(params, NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Width);
    unsigned int rh = GetUInt(params, NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Height);
    if (w.evaluations > 1 && (rw != w.renderW || rh != w.renderH))
        w.renderChanged = true;
    w.renderW = rw;
    w.renderH = rh;

    if (w.evaluations >= 600)
        LogWindow();

    // By the 600th evaluate the process has settled. Off the render thread.
    if (g_evaluations == 600 && !g_censusStarted)
    {
        g_censusStarted = true;
        std::thread(Census).detach();
    }
}
} // namespace UpscalerInputProbe_Dx11
