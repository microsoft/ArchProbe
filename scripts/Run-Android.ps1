param(
    [switch] $Verbose,
    [switch] $BuildOnly,
    [string] $ClearAspect,
    [string] $Arch
)

if (-not $Arch) {
    $Arch = "arm64-v8a"
}

if (-not(Test-Path "build-android-$Arch")) {
    New-Item "build-android-$Arch" -ItemType Directory
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

Push-Location "build-android-$Arch"
cmake -DCMAKE_TOOLCHAIN_FILE="$NdkHome/build/cmake/android.toolchain.cmake" -DANDROID_ABI="$Arch" -DANDROID_PLATFORM=android-28 -G "Ninja" ..
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
adb push ./build-android-$Arch/assets/ /data/local/tmp/archprobe/
adb push ./build-android-$Arch/bin/ /data/local/tmp/archprobe/
adb shell chmod 777 /data/local/tmp/archprobe/bin/ArchProbe
adb shell "cd /data/local/tmp/archprobe/bin && ./ArchProbe $Args"
