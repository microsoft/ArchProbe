if (-not(Test-Path tmp)) {
    New-Item -ItemType Directory tmp
}

adb pull /data/local/tmp/gpu-testbench ./tmp/
