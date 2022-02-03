if (Test-Path tmp/archprobe) {
    Remove-Item tmp/archprobe -Recurse -Force
}

adb shell rm -r /data/local/tmp/archprobe
