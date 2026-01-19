# CUDA Video Processor — Benchmark Results

## Hardware Info

| Property | Value |
|----------|-------|
| GPU | NVIDIA GeForce RTX 3090 |
| SM Count | 82 |
| Memory | 24126 MB |
| Memory Bus | 384-bit |
| Memory Clock | 9751 MHz |
| L2 Cache | 6.0 MB |
| Peak Memory BW (spec) | 936 GB/s |
| CUDA Runtime | 13.1 |
| CUDA Driver | 13.1 |
| Compute Capability | 8.6 |

**Resolution:** 3840 x 2160 (8294400 pixels, 31.64 MB/frame)

Generating synthetic 3840x2160 RGBA frame...
Running CPU baselines...
CPU baselines: Sobel=70.2ms, Enhance=22.4ms, RGB=2.9ms, Brightness=8.4ms

## Correctness Validation

  Sobel correctness: max_err=0, mean_err=0.000 — PASS
  Enhancement correctness: max_err=0, mean_err=0.000 — PASS
  RGB average: GPU=(127,127,127) CPU=(127,127,127) — PASS
  Brightness histogram: max_bin_err=0 — PASS

Running per-kernel benchmarks (20 warmup, 100 timed, L2 flush between)...

## Per-Kernel Results

| Kernel | Median (ms) | P99 (ms) | BW (GB/s) | % Peak BW | CPU (ms) | Speedup |
|--------|-------------|----------|-----------|-----------|----------|---------|
| Sobel (naive/global) | 0.129 | 0.130 | 514.9 | 55.0% | 70.2 | 544x |
| Sobel (shared mem tiled) | 0.124 | 0.126 | 536.0 | 57.3% | 70.2 | 567x |
| Enhancement | 0.094 | 0.097 | 709.4 | 75.8% | 22.4 | 239x |
| RGB Histogram | 0.147 | 0.154 | 225.9 | 24.1% | 2.9 | 20x |
| Brightness Histogram | 0.140 | 0.152 | 237.0 | 25.3% | 8.4 | 60x |

**Shared memory tiling speedup:** 1.04x over naive Sobel

## Full Pipeline Benchmark (1000 frames, all filters ON, headless)

| Metric | Value |
|--------|-------|
| GPU sustained FPS | 332.1 |
| GPU frame latency | 3.01 ms |
| CPU baseline FPS | 9.92 |
| Pipeline speedup | 33x |
| Frames timed | 1000 |

Benchmark complete.
