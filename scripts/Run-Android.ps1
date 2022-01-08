param(
    [switch] $Verbose,
    [switch] $BuildOnly,
    [string] $ClearAspect
)

if (-not(Test-Path "build-android-aarch64")) {
    New-Item "build-android-aarch64" -ItemType Directory
}

$NdkHome = $null
if ($env:ANDROID_NDK -ne $null) {
    $NdkHome = $env:ANDROID_NDK
}
if ($env:ANDROID_NDK_HOME -ne $null) {
    $NdkHome = $env:ANDROID_NDK_HOME
}

if ($NdkHome -eq $null) {
    Write-Host "Couldn't find `ANDROID_NDK` in environment variables. Is NDK installed?"
    return -1
}

Push-Location "build-android-aarch64"
cmake -DCMAKE_TOOLCHAIN_FILE="$NdkHome/build/cmake/android.toolchain.cmake" -DANDROID_ABI="arm64-v8a" -DANDROID_PLATFORM=android-28 -G "Ninja" ..
cmake --build . -t ArchProbe
Pop-Location

if ($BuildOnly) {
    return
}

$Args = ""
if ($Verbose) {
    $Args += "-v "
}
if ($ClearAspect) {
    $Args += "-c $ClearAspect "
}

adb reconnect offline
adb push ./build-android-aarch64/assets/ /data/local/tmp/gpu-testbench/
adb push ./build-android-aarch64/bin/ /data/local/tmp/gpu-testbench/
adb shell chmod 777 /data/local/tmp/gpu-testbench/bin/ArchProbe
adb shell "cd /data/local/tmp/gpu-testbench/bin && ./ArchProbe $Args"
