if (Test-Path tmp/gpu-testbench) {
    Remove-Item tmp/gpu-testbench -Recurse -Force
}

adb shell rm -r /data/local/tmp/gpu-testbench
