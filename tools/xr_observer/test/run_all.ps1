$root = Split-Path -Parent $PSScriptRoot
$exe = "$root\test\bin\panel\panel_test.exe"
$layer = "$root\bin\menu\xr_observer.dll"
$fake = "$root\test\bin\fake\winmm.dll"
$log = "$root\test\results"
New-Item -ItemType Directory -Force $log | Out-Null
# The headset menu layer (menu_layer.cpp + overlay_panel.cpp) on stub runtime functions (panel_test.cpp): the
# overlay as a quad layer while it is open, the application's frame untouched once it is closed. Mode 1 with a
# fence, 2 without, 3 without and with the old kind of shared handle. The control switches the panel off.
function Run($mode, $name, $envs, $expectPass) {
    Remove-Item env:XR_OBSERVER_PANEL -ErrorAction SilentlyContinue
    foreach ($k in $envs.Keys) { Set-Item "env:$k" $envs[$k] }
    $out = & $exe $layer $fake $mode 2>&1
    $code = $LASTEXITCODE
    $out | Out-File "$log\$name.txt" -Encoding utf8
    $passed = $code -eq 0
    $verdict = if ($passed -eq $expectPass) { 'as expected' } else { '*** UNEXPECTED ***' }
    $last = ($out | Select-Object -Last 1)
    Write-Output ("{0,-28} {1,-7} {2}  | {3}" -f $name, $(if ($passed) {'PASS'} else {'FAIL'}), $verdict, $last)
    Remove-Item env:XR_OBSERVER_PANEL -ErrorAction SilentlyContinue
}
Run 1 'panel_fence'        @{} $true
Run 2 'panel_nofence'      @{} $true
Run 3 'panel_legacy'       @{} $true
Run 1 'ctl_panel_off'      @{XR_OBSERVER_PANEL='0'} $false
# The game DLL's side of the headset overlay, on a real ImGui frame (overlay_test.cpp).
$overlayExe = "$root\test\bin\overlay\overlay_test.exe"
$overlayVars = 'OVERLAY_TEST_POINTER','OVERLAY_TEST_PASSIVE','OVERLAY_TEST_OLDCALL','OVERLAY_TEST_NOCLICK','OVERLAY_TEST_NOPOST','OVERLAY_TEST_DX12_NODRAW','OVERLAY_TEST_DX12_FORMAT'
function RunOverlay($name, $envs, $expectPass, $mode = $null) {
    foreach ($v in $overlayVars) { Remove-Item "env:$v" -ErrorAction SilentlyContinue }
    foreach ($k in $envs.Keys) { Set-Item "env:$k" $envs[$k] }
    $out = if ($mode) { & $overlayExe $mode 2>&1 } else { & $overlayExe 2>&1 }
    $code = $LASTEXITCODE
    $out | Out-File "$log\$name.txt" -Encoding utf8
    $passed = $code -eq 0
    $verdict = if ($passed -eq $expectPass) { 'as expected' } else { '*** UNEXPECTED ***' }
    $last = ($out | Select-Object -Last 1)
    Write-Output ("{0,-28} {1,-7} {2}  | {3} (exit {4})" -f $name, $(if ($passed) {'PASS'} else {'FAIL'}), $verdict, $last, $code)
    foreach ($v in $overlayVars) { Remove-Item "env:$v" -ErrorAction SilentlyContinue }
}
RunOverlay 'overlay'             @{} $true
RunOverlay 'ctl_overlay_pointer' @{OVERLAY_TEST_POINTER='1'} $false
RunOverlay 'ctl_overlay_passive' @{OVERLAY_TEST_PASSIVE='1'} $false
# The D3D12 game side: a D3D12 shared texture + fence drawn by ImGui's D3D12 renderer, opened and read by
# a D3D11 device as the layer does.
RunOverlay 'overlay_dx12'        @{} $true 'dx12'
RunOverlay 'ctl_overlay_dx12_nodraw' @{OVERLAY_TEST_DX12_NODRAW='1'} $false 'dx12'
RunOverlay 'overlay_dx12_bgra8'  @{OVERLAY_TEST_DX12_FORMAT='87'} $true 'dx12'
RunOverlay 'overlay_dx12_10bit'  @{OVERLAY_TEST_DX12_FORMAT='24'} $true 'dx12'
# The bridge's D3D12 device made through the device factory, beside a stand-in for a hook on D3D12CreateDevice
# (device_test.cpp).
$deviceExe = "$root\test\bin\device\device_test.exe"
function RunDevice($name, $envs, $expectPass) {
    Remove-Item env:DEVICE_TEST_OLDPATH -ErrorAction SilentlyContinue
    foreach ($k in $envs.Keys) { Set-Item "env:$k" $envs[$k] }
    $out = & $deviceExe 2>&1
    $code = $LASTEXITCODE
    $out | Out-File "$log\$name.txt" -Encoding utf8
    $passed = $code -eq 0
    $verdict = if ($passed -eq $expectPass) { 'as expected' } else { '*** UNEXPECTED ***' }
    $last = ($out | Select-Object -Last 1)
    Write-Output ("{0,-28} {1,-7} {2}  | {3} (exit {4})" -f $name, $(if ($passed) {'PASS'} else {'FAIL'}), $verdict, $last, $code)
    Remove-Item env:DEVICE_TEST_OLDPATH -ErrorAction SilentlyContinue
}
RunDevice 'device_by_factory'   @{} $true
RunDevice 'ctl_device_oldpath'  @{DEVICE_TEST_OLDPATH='1'} $false
# The GPU timers of the bridge's list and the table of their means (times_test.cpp): work of two sizes, with content.
$timesExe = "$root\test\bin\times\times_test.exe"
function RunTimes($name, $envs, $expectPass) {
    Remove-Item env:TIMES_TEST_EMPTY -ErrorAction SilentlyContinue
    foreach ($k in $envs.Keys) { Set-Item "env:$k" $envs[$k] }
    $out = & $timesExe 2>&1
    $code = $LASTEXITCODE
    $out | Out-File "$log\$name.txt" -Encoding utf8
    $passed = $code -eq 0
    $verdict = if ($passed -eq $expectPass) { 'as expected' } else { '*** UNEXPECTED ***' }
    $last = ($out | Select-Object -Last 1)
    Write-Output ("{0,-28} {1,-7} {2}  | {3} (exit {4})" -f $name, $(if ($passed) {'PASS'} else {'FAIL'}), $verdict, $last, $code)
    Remove-Item env:TIMES_TEST_EMPTY -ErrorAction SilentlyContinue
}
RunTimes 'gpu_times'            @{} $true
RunTimes 'ctl_times_empty'      @{TIMES_TEST_EMPTY='1'} $false
# The same-eye fit (OptiScaler/inputs/PerEyeFitShader.h) on made-up depth and motion: a stale start is given up,
# and the wide fit gives the fits of the fit in one group for under half of its GPU time.
$fitExe = "$root\test\bin\fit\fit_test.exe"
function RunFit($name, $envs, $expectPass) {
    Remove-Item env:FIT_TEST_OLD, env:FIT_TEST_ONE_GROUP -ErrorAction SilentlyContinue
    foreach ($k in $envs.Keys) { Set-Item "env:$k" $envs[$k] }
    $out = & $fitExe 2>&1
    $code = $LASTEXITCODE
    $out | Out-File "$log\$name.txt" -Encoding utf8
    $passed = $code -eq 0
    $verdict = if ($passed -eq $expectPass) { 'as expected' } else { '*** UNEXPECTED ***' }
    $last = ($out | Select-Object -Last 1)
    Write-Output ("{0,-28} {1,-7} {2}  | {3} (exit {4})" -f $name, $(if ($passed) {'PASS'} else {'FAIL'}), $verdict, $last, $code)
    Remove-Item env:FIT_TEST_OLD, env:FIT_TEST_ONE_GROUP -ErrorAction SilentlyContinue
}
RunFit 'same_eye_fit'           @{} $true
RunFit 'ctl_fit_old_shader'     @{FIT_TEST_OLD='1'} $false
RunFit 'ctl_fit_one_group'      @{FIT_TEST_ONE_GROUP='1'} $false
# The D3D12 same-eye fit (OptiScaler/inputs/PerEyeFitShader_Dx12.h, PerEyeDlss_Dx12): a stale start is given up;
# the control is the overlay's switch off (the start as it was).
$fit12Exe = "$root\test\bin\fit12\fit12_test.exe"
function RunFit12($name, $envs, $expectPass) {
    Remove-Item env:FIT12_TEST_OLD, env:FIT12_TEST_LINEAR_OFF, env:FIT12_TEST_NO_RESTART, env:FIT12_TEST_NEAR_KEPT, env:FIT12_TEST_CABIN, env:FIT12_TEST_NO_DUMP, env:FIT12_TEST_OLD_GAP, env:FIT12_TEST_SHIFT_OFF, env:FIT12_TEST_AGE_OFF, env:FIT12_TEST_NO_SNAP -ErrorAction SilentlyContinue
    foreach ($k in $envs.Keys) { Set-Item "env:$k" $envs[$k] }
    $out = & $fit12Exe 2>&1
    $code = $LASTEXITCODE
    $out | Out-File "$log\$name.txt" -Encoding utf8
    $passed = $code -eq 0
    $verdict = if ($passed -eq $expectPass) { 'as expected' } else { '*** UNEXPECTED ***' }
    $last = ($out | Select-Object -Last 1)
    Write-Output ("{0,-28} {1,-7} {2}  | {3} (exit {4})" -f $name, $(if ($passed) {'PASS'} else {'FAIL'}), $verdict, $last, $code)
    Remove-Item env:FIT12_TEST_OLD, env:FIT12_TEST_LINEAR_OFF, env:FIT12_TEST_NO_RESTART, env:FIT12_TEST_NEAR_KEPT, env:FIT12_TEST_CABIN, env:FIT12_TEST_NO_DUMP, env:FIT12_TEST_OLD_GAP, env:FIT12_TEST_SHIFT_OFF, env:FIT12_TEST_AGE_OFF, env:FIT12_TEST_NO_SNAP -ErrorAction SilentlyContinue
}
RunFit12 'same_eye_fit_dx12'      @{} $true
RunFit12 'ctl_fit_dx12_keep_stale' @{FIT12_TEST_OLD='1'} $false
RunFit12 'ctl_fit_dx12_no_restart' @{FIT12_TEST_NO_RESTART='1'} $false
RunFit12 'ctl_fit_dx12_linear_off' @{FIT12_TEST_LINEAR_OFF='1'} $false
# Nothing left out at a depth gap (UpscalerFitLeaveOutNearest off): the cabin cases 12-12d fail.
RunFit12 'ctl_fit_dx12_near_kept'  @{FIT12_TEST_NEAR_KEPT='1'} $false
# The crop's move through subrects (case 15): the shift pass shifting by nothing fails it.
RunFit12 'ctl_fit_dx12_shift_off'  @{FIT12_TEST_SHIFT_OFF='1'} $false
# The crop's age (case 16): the move's pass carrying nothing along fails it.
RunFit12 'ctl_fit_dx12_age_off'    @{FIT12_TEST_AGE_OFF='1'} $false
# The gaze filter without the snap (case 17): the eased-in gaze needs more polls than allowed.
RunFit12 'ctl_fit_dx12_no_snap'    @{FIT12_TEST_NO_SNAP='1'} $false
# The dump's bit not set: case 13 (the samples bit for bit where the DLL copies them) fails.
RunFit12 'ctl_fit_dx12_no_dump'    @{FIT12_TEST_NO_DUMP='1'} $false
# The VULKAN fit and compose (SameEyeFit_Vk, the SPIR-V of the same source, PerEyeDlss_Vk): fit12's cases on a real
# Vulkan device; the control: no dump bit.
$vkFitExe = "$root\test\bin\vk_fit\vk_fit_test.exe"
function RunVkFit($name, $envs, $expectPass) {
    Remove-Item env:VK_FIT_TEST_NEAR_KEPT, env:VK_FIT_TEST_NO_DUMP, env:VK_FIT_TEST_OLD_GAP -ErrorAction SilentlyContinue
    foreach ($k in $envs.Keys) { Set-Item "env:$k" $envs[$k] }
    $out = & $vkFitExe 2>&1
    $code = $LASTEXITCODE
    $out | Out-File "$log\$name.txt" -Encoding utf8
    $passed = $code -eq 0
    $verdict = if ($passed -eq $expectPass) { 'as expected' } else { '*** UNEXPECTED ***' }
    $last = ($out | Select-Object -Last 1)
    Write-Output ("{0,-28} {1,-7} {2}  | {3} (exit {4})" -f $name, $(if ($passed) {'PASS'} else {'FAIL'}), $verdict, $last, $code)
    Remove-Item env:VK_FIT_TEST_NEAR_KEPT, env:VK_FIT_TEST_NO_DUMP, env:VK_FIT_TEST_OLD_GAP -ErrorAction SilentlyContinue
}
RunVkFit 'same_eye_fit_vk'          @{} $true
RunVkFit 'ctl_fit_vk_no_dump'       @{VK_FIT_TEST_NO_DUMP='1'} $false
# The VULKAN foveal crop's passes (FoveaPasses_Vk: COMPOSITE with the feather, ENLARGE) and the box maths (FoveaBox.h)
# on a real device; the controls: no feather where one is expected, the box composited at the wrong place.
$vkFoveaExe = "$root\test\bin\vk_fovea\vk_fovea_test.exe"
function RunVkFovea($name, $envs, $expectPass) {
    Remove-Item env:VK_FOVEA_TEST_NO_FEATHER, env:VK_FOVEA_TEST_BOX_OFF, env:VK_FOVEA_TEST_NO_WORKER, env:VK_FOVEA_TEST_COPY_OFF, env:VK_FOVEA_TEST_TAA_OFF, env:VK_FOVEA_TEST_TIME_OFF, env:VK_FOVEA_TEST_TIME_NO_SKIP -ErrorAction SilentlyContinue
    foreach ($k in $envs.Keys) { Set-Item "env:$k" $envs[$k] }
    $out = & $vkFoveaExe 2>&1
    $code = $LASTEXITCODE
    $out | Out-File "$log\$name.txt" -Encoding utf8
    $passed = $code -eq 0
    $verdict = if ($passed -eq $expectPass) { 'as expected' } else { '*** UNEXPECTED ***' }
    $last = ($out | Select-Object -Last 1)
    Write-Output ("{0,-28} {1,-7} {2}  | {3} (exit {4})" -f $name, $(if ($passed) {'PASS'} else {'FAIL'}), $verdict, $last, $code)
    Remove-Item env:VK_FOVEA_TEST_NO_FEATHER, env:VK_FOVEA_TEST_BOX_OFF, env:VK_FOVEA_TEST_NO_WORKER, env:VK_FOVEA_TEST_COPY_OFF, env:VK_FOVEA_TEST_TAA_OFF, env:VK_FOVEA_TEST_TIME_OFF, env:VK_FOVEA_TEST_TIME_NO_SKIP -ErrorAction SilentlyContinue
}
RunVkFovea 'fovea_passes_vk'        @{} $true
RunVkFovea 'ctl_fovea_vk_no_feather' @{VK_FOVEA_TEST_NO_FEATHER='1'} $false
RunVkFovea 'ctl_fovea_vk_box_off'   @{VK_FOVEA_TEST_BOX_OFF='1'} $false
RunVkFovea 'ctl_fovea_vk_no_worker' @{VK_FOVEA_TEST_NO_WORKER='1'} $false
RunVkFovea 'ctl_fovea_vk_copy_off'  @{VK_FOVEA_TEST_COPY_OFF='1'} $false
RunVkFovea 'ctl_fovea_vk_taa_off'   @{VK_FOVEA_TEST_TAA_OFF='1'} $false
RunVkFovea 'ctl_fovea_vk_time_off'  @{VK_FOVEA_TEST_TIME_OFF='1'} $false
RunVkFovea 'ctl_fovea_vk_time_no_skip' @{VK_FOVEA_TEST_TIME_NO_SKIP='1'} $false
# The crash reporter's trace of the FAULTING thread (OptiScaler/inputs/CrashTrace.h): a call through a null pointer
# names its caller; the control is the old trace from the handler's own stack.
$crashExe = "$root\test\bin\crash\crash_trace_test.exe"
function RunCrash($name, $envs, $expectPass) {
    Remove-Item env:CRASH_TEST_OLD -ErrorAction SilentlyContinue
    foreach ($k in $envs.Keys) { Set-Item "env:$k" $envs[$k] }
    $out = & $crashExe 2>&1
    $code = $LASTEXITCODE
    $out | Out-File "$log\$name.txt" -Encoding utf8
    $passed = $code -eq 0
    $verdict = if ($passed -eq $expectPass) { 'as expected' } else { '*** UNEXPECTED ***' }
    $last = ($out | Select-Object -Last 1)
    Write-Output ("{0,-28} {1,-7} {2}  | {3} (exit {4})" -f $name, $(if ($passed) {'PASS'} else {'FAIL'}), $verdict, $last, $code)
    Remove-Item env:CRASH_TEST_OLD -ErrorAction SilentlyContinue
}
RunCrash 'crash_trace'            @{} $true
RunCrash 'ctl_crash_trace_old'    @{CRASH_TEST_OLD='1'} $false
# The per-eye DLSS section of the menu (OptiScaler/inputs/PerEyeMenu.h), drawn and clicked.
RunOverlay 'overlay_section'     @{} $true 'section'
RunOverlay 'ctl_section_noclick' @{OVERLAY_TEST_NOCLICK='1'} $false 'section'
# The Vulkan crop's block of the menu (OptiScaler/inputs/FoveaMenu_Vk.h; a Vulkan game): the switch and the preset dropdown, clicked.
RunOverlay 'overlay_fovea_vk'    @{} $true 'fovea'
RunOverlay 'ctl_fovea_vk_noclick' @{OVERLAY_TEST_NOCLICK='1'} $false 'fovea'
# The stale-eye warp's controls (OptiScaler/inputs/StaleEyeMenu.h): the Off / Warp dropdown and the Junk Drawer's part.
RunOverlay 'overlay_stale'       @{} $true 'stale'
RunOverlay 'ctl_stale_noclick'   @{OVERLAY_TEST_NOCLICK='1'} $false 'stale'
# A callback run on a window's thread in message context, from another thread (MainThreadCall.h; a game's engine resize).
RunOverlay 'overlay_mainthread'  @{} $true 'mainthread'
RunOverlay 'ctl_mainthread_nopost' @{OVERLAY_TEST_NOPOST='1'} $false 'mainthread'
RunOverlay 'ctl_overlay_oldcall' @{OVERLAY_TEST_OLDCALL='1'} $false

