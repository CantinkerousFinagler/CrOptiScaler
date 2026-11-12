#pragma once

#include <string_view>

// Late ASI plugins loaded the moment a named DLL is loaded ([Plugins] LateAsiPluginsAfter), instead of after a timer.
// Some plugins must register with ReShade before the game creates its device, and ReShade is
// already loaded in some games (ReShade loaded early as an .asi got in its way).
// So the plugins wait for ReShade64.dll and load right behind it. Defined in dllmain.cpp (LoadAsiPlugins).
namespace LateAsiTrigger
{
// Called by the LdrLoadDll hook after every successful load; does nothing unless armed, and fires once.
void OnModuleLoaded(std::wstring_view loadedName);
} // namespace LateAsiTrigger
