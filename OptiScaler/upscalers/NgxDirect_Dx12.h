#pragma once

// The forwarder's generic NGX D3D12 calls on an already-loaded snippet (nvngx.dll_dlssnr.dll, beside OptiScaler or the
// game): a feature of our own that does not share the game's NGX entry (needed when the VR mod has a DLSS pass of
// its own). Used by a foveal crop's RR feature (DLSSDFeatureDx12,
// nvngx_dlssd.dll) and SR feature (DLSSFeatureDx12, nvngx_dlss.dll). One instance for the process (inline: one static).
namespace NgxDirect
{
using PFN_Ngx12Populate = int (*)(const wchar_t*, void*);
using PFN_Ngx12Create = int (*)(const wchar_t*, ID3D12GraphicsCommandList*, int, void*, void**);
using PFN_Ngx12Evaluate = int (*)(ID3D12GraphicsCommandList*, void*, void*);
using PFN_Ngx12Release = int (*)(void*);

struct Calls
{
    bool tried = false;
    HMODULE module = nullptr;
    PFN_Ngx12Populate populate = nullptr;
    PFN_Ngx12Create create = nullptr;
    PFN_Ngx12Evaluate evaluate = nullptr;
    PFN_Ngx12Release release = nullptr;
    std::string reason; // why it is not available
};

inline Calls& Get()
{
    static Calls d;
    if (d.tried)
        return d;
    d.tried = true;
    auto found = Util::FindFilePath(Util::DllPath().remove_filename(), "nvngx.dll_dlssnr.dll");
    if (!found.has_value())
        found = Util::FindFilePath(Util::ExePath().remove_filename(), "nvngx.dll_dlssnr.dll");
    if (!found.has_value())
    {
        d.reason = "nvngx.dll_dlssnr.dll (the forwarder) is missing beside OptiScaler and the game";
        return d;
    }
    d.module = LoadLibraryW(found.value().wstring().c_str());
    if (d.module == nullptr)
    {
        d.reason = std::format("nvngx.dll_dlssnr.dll would not load (error {})", GetLastError());
        return d;
    }
    d.populate = (PFN_Ngx12Populate) GetProcAddress(d.module, "dlssnr_ngx12_populate");
    d.create = (PFN_Ngx12Create) GetProcAddress(d.module, "dlssnr_ngx12_create");
    d.evaluate = (PFN_Ngx12Evaluate) GetProcAddress(d.module, "dlssnr_ngx12_evaluate");
    d.release = (PFN_Ngx12Release) GetProcAddress(d.module, "dlssnr_ngx12_release");
    if (d.create == nullptr || d.evaluate == nullptr || d.release == nullptr)
    {
        d.reason = "nvngx.dll_dlssnr.dll is an older build without the dlssnr_ngx12_* exports";
        d.create = nullptr;
    }
    return d;
}
} // namespace NgxDirect
