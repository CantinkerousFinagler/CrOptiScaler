// Offline test of the bridge's D3D12 device made through the device factory (OptiScaler/with_dx12/DeviceByFactory.h)
// beside a STAND-IN for ReShade's hook: the first bytes of d3d12.dll's D3D12CreateDevice are replaced by a jump to a
// function of this test that counts the call and fails it. What ReShade does with a device it sees (tell its add-ons,
// SekiroTSR among them) begins with that call being seen.
//   device_test
// The new path must leave the count at 0 and give a device that works: on the adapter of a real D3D11 device, with a
// command queue and a fence, and a texture of the D3D11 device opened on it through a shared handle (what the bridge
// does with every input).
// Control, must FAIL: DEVICE_TEST_OLDPATH=1 (D3D12CreateDevice, what the bridge called before: the stand-in sees it).
#include <windows.h>
#include <d3d11_1.h>
#include <d3d12.h>
#include <dxgi1_4.h>

#include "../../../OptiScaler/with_dx12/DeviceByFactory.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

static int g_seen = 0;

static HRESULT WINAPI StandInForTheHook(IUnknown*, D3D_FEATURE_LEVEL, REFIID, void** device)
{
    ++g_seen;
    if (device != nullptr)
        *device = nullptr;
    return E_FAIL;
}

static bool PutJump(void* at, void* to)
{
    // mov rax, to; jmp rax
    unsigned char code[12] = { 0x48, 0xB8, 0, 0, 0, 0, 0, 0, 0, 0, 0xFF, 0xE0 };
    std::memcpy(code + 2, &to, sizeof(to));
    DWORD before = 0;
    if (!VirtualProtect(at, sizeof(code), PAGE_EXECUTE_READWRITE, &before))
        return false;
    std::memcpy(at, code, sizeof(code));
    VirtualProtect(at, sizeof(code), before, &before);
    FlushInstructionCache(GetCurrentProcess(), at, sizeof(code));
    return true;
}

int main()
{
    char value[8] = {};
    const bool oldPath = GetEnvironmentVariableA("DEVICE_TEST_OLDPATH", value, sizeof(value)) > 0 && value[0] == '1';
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);

    ID3D11Device* device11 = nullptr;
    ID3D11DeviceContext* context11 = nullptr;
    if (FAILED(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, nullptr, 0, D3D11_SDK_VERSION, &device11,
                                 nullptr, &context11)))
    {
        std::printf("FAILED: no D3D11 device\n");
        return 1;
    }
    IDXGIDevice* dxgiDevice = nullptr;
    IDXGIAdapter* adapter = nullptr;
    if (FAILED(device11->QueryInterface(IID_PPV_ARGS(&dxgiDevice))) || FAILED(dxgiDevice->GetAdapter(&adapter)))
    {
        std::printf("FAILED: the D3D11 device's adapter\n");
        return 1;
    }
    DXGI_ADAPTER_DESC adapterDesc {};
    adapter->GetDesc(&adapterDesc);

    const HMODULE d3d12 = LoadLibraryW(L"d3d12.dll");
    void* createDevice = d3d12 != nullptr ? (void*) GetProcAddress(d3d12, "D3D12CreateDevice") : nullptr;
    const auto getInterface =
        d3d12 != nullptr ? (DeviceByFactory::GetInterfaceFn) GetProcAddress(d3d12, "D3D12GetInterface") : nullptr;
    if (createDevice == nullptr || !PutJump(createDevice, (void*) &StandInForTheHook))
    {
        std::printf("FAILED: the stand-in for the hook could not be put on D3D12CreateDevice\n");
        return 1;
    }
    // The stand-in itself must work, or a count of 0 means nothing.
    {
        ID3D12Device* probe = nullptr;
        const HRESULT hr = D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&probe));
        if (g_seen != 1 || SUCCEEDED(hr))
        {
            std::printf("FAILED: the stand-in did not see a plain D3D12CreateDevice (seen %d, result %08lX)\n", g_seen,
                        (unsigned long) hr);
            return 1;
        }
        g_seen = 0;
    }

    int failures = 0;
    ID3D12Device* device12 = nullptr;
    DeviceByFactory::Said said;
    HRESULT made = E_FAIL;
    if (oldPath)
        made = D3D12CreateDevice(adapter, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device12));
    else
        made = DeviceByFactory::Create(getInterface, adapter, D3D_FEATURE_LEVEL_11_0, &device12, &said);
    const bool unseen = g_seen == 0;
    std::printf("%s: result %08lX (factory %08lX, flags %08lX, create %08lX), device %s; D3D12CreateDevice seen by the "
                "stand-in %d time(s) (wanted 0) -> %s\n",
                oldPath ? "D3D12CreateDevice (the OLD path)" : "the device factory", (unsigned long) made,
                (unsigned long) said.factory, (unsigned long) said.flags, (unsigned long) said.create,
                device12 != nullptr ? "made" : "NOT made", g_seen, unseen && device12 != nullptr ? "OK" : "FAIL");
    if (!unseen || device12 == nullptr)
        ++failures;

    if (device12 != nullptr)
    {
        // The same adapter as the D3D11 device's.
        const LUID luid = device12->GetAdapterLuid();
        const bool sameAdapter =
            luid.LowPart == adapterDesc.AdapterLuid.LowPart && luid.HighPart == adapterDesc.AdapterLuid.HighPart;
        // A queue and a fence, as the bridge makes them.
        D3D12_COMMAND_QUEUE_DESC queueDesc {};
        queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        ID3D12CommandQueue* queue = nullptr;
        ID3D12Fence* fence = nullptr;
        const bool queueOk = SUCCEEDED(device12->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&queue))) &&
                             SUCCEEDED(device12->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)));
        // A texture of the D3D11 device, opened on the D3D12 device through an NT handle.
        D3D11_TEXTURE2D_DESC desc {};
        desc.Width = desc.Height = 256;
        desc.MipLevels = desc.ArraySize = 1;
        desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        desc.SampleDesc.Count = 1;
        desc.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
        desc.MiscFlags = D3D11_RESOURCE_MISC_SHARED | D3D11_RESOURCE_MISC_SHARED_NTHANDLE;
        ID3D11Texture2D* texture = nullptr;
        IDXGIResource1* resource = nullptr;
        HANDLE handle = nullptr;
        ID3D12Resource* opened = nullptr;
        bool shared = SUCCEEDED(device11->CreateTexture2D(&desc, nullptr, &texture)) &&
                      SUCCEEDED(texture->QueryInterface(IID_PPV_ARGS(&resource))) &&
                      SUCCEEDED(resource->CreateSharedHandle(nullptr, DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE,
                                                             nullptr, &handle)) &&
                      SUCCEEDED(device12->OpenSharedHandle(handle, IID_PPV_ARGS(&opened))) && opened != nullptr;
        if (shared)
        {
            const D3D12_RESOURCE_DESC got = opened->GetDesc();
            shared = got.Width == 256 && got.Height == 256 && got.Format == DXGI_FORMAT_R8G8B8A8_UNORM;
        }
        const bool ok = sameAdapter && queueOk && shared;
        std::printf("the device: on the D3D11 device's adapter %s, a command queue and a fence %s, a D3D11 texture "
                    "opened on it through a shared handle %s -> %s\n",
                    sameAdapter ? "yes" : "NO", queueOk ? "yes" : "NO", shared ? "yes" : "NO", ok ? "OK" : "FAIL");
        if (!ok)
            ++failures;
        if (opened != nullptr)
            opened->Release();
        if (handle != nullptr)
            CloseHandle(handle);
        if (resource != nullptr)
            resource->Release();
        if (texture != nullptr)
            texture->Release();
        if (fence != nullptr)
            fence->Release();
        if (queue != nullptr)
            queue->Release();
    }
    std::printf("%s: %d checks, %d failures\n", failures == 0 ? "PASSED" : "FAILED", device12 != nullptr ? 2 : 1,
                failures);
    std::fflush(stdout);
    // No releases of the devices: the process ends here, and d3d12.dll's own function stays replaced until then.
    return failures == 0 ? 0 : 1;
}
