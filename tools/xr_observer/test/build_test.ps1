# Builds the game DLL, the layer and the tests, checks that each DLL is newer than its sources and than the commit,
# and runs the offline runs. Nothing is copied anywhere. For a build that goes to ONE game afterwards: commit, run
# this, then deploy_to_game.ps1.
# A running game can hold the PDBs (LNK1201): close it, or build when it has ended.
$ErrorActionPreference = 'Stop'
$fork = (Resolve-Path "$PSScriptRoot\..\..\..").Path
$msb = 'C:\Program Files\Microsoft Visual Studio\18\Community\MSBuild\Current\Bin\MSBuild.exe'
$head = (git -C $fork log --oneline -1)
Write-Output "fork HEAD: $head"
if (git -C $fork status --short) { Write-Output 'the working tree is NOT clean: nothing built'; exit 3 }
# The Vulkan fit's SPIR-V comes from the D3D12 shader source: regenerated here, it must come out as committed.
& python "$fork\tools\xr_observer\test\vk_fit_shaders.py"
if ($LASTEXITCODE -ne 0) { Write-Output '*** vk_fit_shaders.py FAILED ***'; exit 8 }
if (git -C $fork status --short) { Write-Output '*** PerEyeFitShader_Vk.h was stale (the shaders changed without vk_fit_shaders.py): regenerate and commit ***'; exit 9 }
& python "$fork\tools\xr_observer\test\vk_fovea_shaders.py"
if ($LASTEXITCODE -ne 0) { Write-Output '*** vk_fovea_shaders.py FAILED ***'; exit 8 }
if (git -C $fork status --short) { Write-Output '*** FoveaShader_Vk.h was stale (the shaders changed without vk_fovea_shaders.py): regenerate and commit ***'; exit 9 }
$o = & $msb "$fork\OptiScaler.sln" -p:Configuration=Release -p:Platform=x64 -m -v:minimal -nologo 2>&1
if ($LASTEXITCODE -ne 0) { $o | Select-String ' error ' | Select-Object -First 10 | ForEach-Object { $_.Line }; Write-Output '*** OptiScaler BUILD FAILED ***'; exit 4 }
& "$fork\tools\xr_observer\test\build_all.ps1"
if ($LASTEXITCODE -ne 0) { Write-Output '*** layer BUILD FAILED ***'; exit 5 }
$dll = Get-Item "$fork\x64\Release\a\OptiScaler.dll"
$layer = Get-Item "$fork\tools\xr_observer\bin\menu\xr_observer.dll"
$layerSources = @(Get-ChildItem "$fork\tools\xr_observer" -File | Where-Object { $_.Extension -in '.cpp', '.h', '.vcxproj' }) + @(Get-Item "$fork\OptiScaler\inputs\StaleEyeShareAbi.h")
$newestLayerSource = ($layerSources | Sort-Object LastWriteTime -Descending | Select-Object -First 1)
$newestDllSource = (Get-ChildItem "$fork\OptiScaler" -Recurse -File -Include *.cpp, *.h | Sort-Object LastWriteTime -Descending | Select-Object -First 1)
$commitTime = [datetime](git -C $fork log -1 --format=%cI)
Write-Output "commit $commitTime; OptiScaler.dll $($dll.LastWriteTime) (newest source $($newestDllSource.Name) $($newestDllSource.LastWriteTime)); layer $($layer.LastWriteTime) (newest source $($newestLayerSource.Name) $($newestLayerSource.LastWriteTime))"
if ($dll.LastWriteTime -lt $newestDllSource.LastWriteTime -or $layer.LastWriteTime -lt $newestLayerSource.LastWriteTime -or $dll.LastWriteTime -lt $commitTime) {
    Write-Output '*** a DLL is OLDER than its sources or the commit ***'; exit 6
}
$out = & "$fork\tools\xr_observer\test\run_all.ps1"
$out | ForEach-Object { $_ }
$bad = ($out | Select-String 'UNEXPECTED').Count
Write-Output "offline runs: $($out.Count), as expected $(($out | Select-String 'as expected').Count), unexpected $bad"
if ($bad -ne 0 -or $out.Count -lt 48) { Write-Output '*** offline tests not as expected ***'; exit 7 }
Write-Output "BUILT AND TESTED $head"
exit 0
