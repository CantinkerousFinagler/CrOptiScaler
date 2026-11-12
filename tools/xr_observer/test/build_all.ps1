$ErrorActionPreference = 'Continue'
$msb = 'C:\Program Files\Microsoft Visual Studio\18\Community\MSBuild\Current\Bin\MSBuild.exe'
$root = Split-Path -Parent $PSScriptRoot
foreach ($p in @("$root\menu_layer.vcxproj", "$root\test\fake_winmm.vcxproj", "$root\test\panel_test.vcxproj", "$root\test\overlay_test.vcxproj", "$root\test\device_test.vcxproj", "$root\test\times_test.vcxproj", "$root\test\fit_test.vcxproj", "$root\test\fit12_test.vcxproj", "$root\test\vk_fit_test.vcxproj", "$root\test\vk_fovea_test.vcxproj", "$root\test\crash_trace_test.vcxproj")) {
    $out = & $msb $p -p:Configuration=Release -p:Platform=x64 -v:minimal -nologo 2>&1
    $code = $LASTEXITCODE
    $errs = $out | Select-String -Pattern 'error|warning C' | Select-Object -First 20
    Write-Output "== $p exit $code"
    $errs | ForEach-Object { Write-Output $_.Line }
    if ($code -ne 0) {
        # A failed build leaves the previous DLL in place: the tests would run against it and mean nothing.
        Write-Output '*** BUILD FAILED: stopping, the tests would run against the previous DLL ***'
        exit 1
    }
}
