# Puts the build that build_test.ps1 has just built and tested into a game: winmm.dll beside the game's
# exe and the OpenXR layer (DLL + manifest) in <the OptiScaler folder>\xr_observer, from where winmm offers it to the
# process's OpenXR loader (no registry). Builds nothing and tests nothing: run build_test.ps1 first.
#   -Folder    the folder of the game's exe (where its winmm.dll is)
#   -Process   the game's process name, without .exe (nothing is copied while it runs)
#   -Previous  the commit of the winmm.dll being replaced (names the backups)
# The OptiScaler folder is <Folder>\OptiScaler when that exists, else <Folder> itself (as dllmain decides it).
param([Parameter(Mandatory)][string]$Folder, [Parameter(Mandatory)][string]$Process, [Parameter(Mandatory)][string]$Previous)
$ErrorActionPreference = 'Stop'
$fork = (Resolve-Path "$PSScriptRoot\..\..\..").Path
if (Get-Process $Process -ErrorAction SilentlyContinue) { Write-Output "$Process is RUNNING: nothing copied"; exit 2 }
if (-not (Test-Path "$Folder\winmm.dll")) { Write-Output "no winmm.dll in $Folder : not a game this fork is installed in; nothing copied"; exit 3 }
if (git -C $fork status --short) { Write-Output 'the working tree is NOT clean: the DLLs may not be the commit they say; nothing copied'; exit 4 }
$head = (git -C $fork log --oneline -1)
$dll = Get-Item "$fork\x64\Release\a\OptiScaler.dll"
$layer = Get-Item "$fork\tools\xr_observer\bin\menu\xr_observer.dll"
$commitTime = [datetime](git -C $fork log -1 --format=%cI)
Write-Output "fork HEAD: $head ($commitTime); OptiScaler.dll $($dll.LastWriteTime); layer $($layer.LastWriteTime)"
if ($dll.LastWriteTime -lt $commitTime) { Write-Output '*** OptiScaler.dll is OLDER than the commit: build first; nothing copied ***'; exit 5 }

$backups = @(@("$Folder\winmm.dll", "$Folder\winmm.dll.$Previous.bak"), @("$Folder\OptiScaler.ini", "$Folder\OptiScaler.ini.$Previous.bak"))
foreach ($b in $backups) {
    if (-not (Test-Path $b[0])) { continue }
    if (Test-Path $b[1]) { Write-Output "backup exists already, kept: $($b[1])" } else { Copy-Item $b[0] $b[1]; Write-Output "backup: $($b[1])" }
}
Copy-Item $dll.FullName "$Folder\winmm.dll" -Force
$opti = if (Test-Path "$Folder\OptiScaler" -PathType Container) { "$Folder\OptiScaler" } else { $Folder }
$inFolder = "$opti\xr_observer"
New-Item -ItemType Directory -Force $inFolder | Out-Null
Copy-Item $layer.FullName "$inFolder\xr_observer.dll" -Force
$manifest = @'
{
  "file_format_version": "1.0.0",
  "api_layer": {
    "name": "XR_APILAYER_OPTISCALER_RVR_xr_observer",
    "library_path": ".\\xr_observer.dll",
    "api_version": "1.0",
    "implementation_version": "1",
    "description": "The OptiScaler menu as a panel in the headset"
  }
}
'@
Set-Content -Path "$inFolder\XR_APILAYER_OPTISCALER_RVR_xr_observer.json" -Value $manifest -Encoding ascii
foreach ($p in @(@($dll.FullName, "$Folder\winmm.dll"), @($layer.FullName, "$inFolder\xr_observer.dll"))) {
    $same = (Get-FileHash $p[0]).Hash -eq (Get-FileHash $p[1]).Hash
    Write-Output ("{0} -> {1}: {2}" -f (Split-Path $p[0] -Leaf), $p[1], $(if ($same) { 'same hash' } else { '*** DIFFERENT ***' }))
}
Write-Output "DEPLOYED to $Folder : $head"
