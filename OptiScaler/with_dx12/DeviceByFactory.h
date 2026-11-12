#pragma once

// A D3D12 device made WITHOUT a call of D3D12CreateDevice: through ID3D12DeviceFactory, which D3D12GetInterface hands
// out. For the D3D11-to-D3D12 bridge in a process where something watches D3D12CreateDevice and tells others of
// every new device.
//
// Sekiro with the SekiroTSR mod under ReShade: the mod re-initialised on the bridge's device and crashed.
// OptiScaler's own answer to this conflict is to make the device before ReShade is loaded ([Hotfix]
// CreateD3D12DeviceForLuma with LoadReshade); that cannot be used where ReShade is already loaded in these games.
//
// Header only and D3D12 only, so that the offline test (tools/xr_observer/test/device_test) runs this very code
// beside a stand-in for such a hook. The caller includes d3d12.h (the game DLL and the test have their own).

namespace DeviceByFactory
{
using GetInterfaceFn = HRESULT(WINAPI*)(REFCLSID, REFIID, void**);

struct Said
{
    HRESULT factory = E_FAIL; // D3D12GetInterface(CLSID_D3D12DeviceFactory)
    HRESULT flags = E_FAIL;   // SetFlags
    HRESULT create = E_FAIL;  // CreateDevice
};

// As D3D12CreateDevice does, a device the process already has for this adapter is returned in place of a new one.
inline HRESULT Create(GetInterfaceFn getInterface, IUnknown* adapter, D3D_FEATURE_LEVEL featureLevel,
                      ID3D12Device** device, Said* said)
{
    Said local;
    Said& s = said != nullptr ? *said : local;
    s = Said {};
    if (device == nullptr)
        return E_POINTER;
    *device = nullptr;
    if (getInterface == nullptr)
        return s.factory = E_NOINTERFACE;
    ID3D12DeviceFactory* factory = nullptr;
    s.factory = getInterface(CLSID_D3D12DeviceFactory, IID_PPV_ARGS(&factory));
    if (FAILED(s.factory) || factory == nullptr)
        return FAILED(s.factory) ? s.factory : (s.factory = E_NOINTERFACE);
    s.flags = factory->SetFlags(D3D12_DEVICE_FACTORY_FLAG_ALLOW_RETURNING_EXISTING_DEVICE |
                                D3D12_DEVICE_FACTORY_FLAG_ALLOW_RETURNING_INCOMPATIBLE_EXISTING_DEVICE);
    s.create = factory->CreateDevice(adapter, featureLevel, IID_PPV_ARGS(device));
    factory->Release();
    if (SUCCEEDED(s.create) && *device == nullptr)
        s.create = E_FAIL;
    return s.create;
}
} // namespace DeviceByFactory
