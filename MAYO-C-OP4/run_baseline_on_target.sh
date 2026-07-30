#!/bin/bash
# Run this from the MAYO-C-OP4 project directory on the A100/H100 server.
# Reproduces the same baseline measurements gathered locally on the
# RTX 3070, but with real execution numbers for the target GPU.
set -euo pipefail

OUT=baseline_report
mkdir -p "$OUT"

echo "== 1. Register/spill/shared-mem report (native arch) =="
nvcc -arch=native -O3 -lineinfo -Xptxas=-v -Iinclude -I/usr/local/cuda/include \
    -c src/mayo.cu -o "$OUT/mayo_native.o" 2> "$OUT/registers_native.log"
grep -E "Compiling entry function|Used [0-9]+ registers|spill|smem" "$OUT/registers_native.log" \
    | tee "$OUT/registers_summary.txt"

echo "== 2. Build the real executable with the same flags =="
make clean >/dev/null
make NVCC_FLAGS="-arch=native -O3 -lineinfo -Xptxas=-v -Xptxas -O3 -Xcompiler -O3" 2>&1 | tee "$OUT/build.log"

echo "== 3. Correctness check =="
./mayo-2-gpu-fine 2>&1 | tee "$OUT/run_batch_default.log"

echo "== 4. Nsight Systems kernel-time breakdown =="
nsys profile -o "$OUT/nsys_baseline" --force-overwrite true --trace=cuda ./mayo-2-gpu-fine >/dev/null 2>&1
nsys stats --report cuda_gpu_kern_sum "$OUT/nsys_baseline.nsys-rep" | tee "$OUT/nsys_kernels.txt"
nsys stats --force-export=true --report cuda_api_sum "$OUT/nsys_baseline.nsys-rep" | tee "$OUT/nsys_api.txt"

echo "== 5. Nsight Compute occupancy (needs perf-counter permission; run with sudo if ERR_NVGPUCTRPERM) =="
ncu --set full -f -o "$OUT/ncu_baseline" --launch-count 30 ./mayo-2-gpu-fine > "$OUT/ncu_run.log" 2>&1 || \
    echo "ncu failed -- see $OUT/ncu_run.log (likely needs sudo or NVreg_RestrictProfilingToAdminUsers=0)"

echo "== 6. BATCH sweep: 1, 8, 32, 128, 1024 =="
cp include/mayo.cuh include/mayo.cuh.bak
for B in 1 8 32 128 1024; do
    sed -i "s/#define BATCH (.*)/#define BATCH ($B)/" include/mayo.cuh
    make clean >/dev/null
    make >/dev/null 2>&1
    echo "--- BATCH=$B ---" | tee -a "$OUT/batch_sweep.txt"
    ./mayo-2-gpu-fine 2>&1 | tr -d '\r' | grep -E "Throughput|Sign result" | tee -a "$OUT/batch_sweep.txt"
done
cp include/mayo.cuh.bak include/mayo.cuh
rm -f include/mayo.cuh.bak
make clean >/dev/null
make >/dev/null 2>&1

echo "Done. All output under $OUT/"
