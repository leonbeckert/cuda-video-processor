# CUDA Video Processor — Benchmark Results

## Hardware Info

| Property | Value |
|----------|-------|
| GPU | NVIDIA GeForce RTX 4090 |
| SM Count | 128 |
| Memory | 24083 MB |
| Memory Bus | 384-bit |
| Memory Clock | 10501 MHz |
| L2 Cache | 72.0 MB |
| Peak Memory BW (spec) | 1008 GB/s |
| CUDA Runtime | 13.1 |
| CUDA Driver | 13.1 |
| Compute Capability | 8.9 |

**Resolution:** 3840 x 2160 (8294400 pixels, 31.64 MB/frame)

Generating synthetic 3840x2160 RGBA frame...
Running CPU baselines...
CPU baselines: Sobel=70.1ms, Enhance=22.5ms, RGB=2.9ms, Brightness=8.4ms

## Correctness Validation

  Sobel correctness: max_err=0, mean_err=0.000 — PASS
  Enhancement correctness: max_err=0, mean_err=0.000 — PASS
  RGB average: GPU=(127,127,127) CPU=(127,127,127) — PASS
  Brightness histogram: max_bin_err=0 — PASS

Running per-kernel benchmarks (20 warmup, 100 timed, L2 flush between)...

## Per-Kernel Results

| Kernel | Median (ms) | P99 (ms) | BW (GB/s) | % Peak BW | CPU (ms) | Speedup |
|--------|-------------|----------|-----------|-----------|----------|---------|
| Sobel (naive/global) | 0.099 | 0.101 | 667.9 | 66.3% | 70.1 | 706x |
| Sobel (shared mem tiled) | 0.090 | 0.091 | 736.4 | 73.1% | 70.1 | 778x |
| Enhancement | 0.112 | 0.119 | 594.5 | 59.0% | 22.5 | 201x |
| RGB Histogram | 0.163 | 0.183 | 204.0 | 20.2% | 2.9 | 18x |
| Brightness Histogram | 0.142 | 0.152 | 233.9 | 23.2% | 8.4 | 59x |

**Shared memory tiling speedup:** 1.10x over naive Sobel

## Full Pipeline Benchmark (1000 frames, all filters ON, headless)

| Metric | Value |
|--------|-------|
| GPU sustained FPS | 95.9 |
| GPU frame latency | 10.42 ms |
| CPU baseline FPS | 9.95 |
| Pipeline speedup | 10x |
| Frames timed | 1000 |

Benchmark complete.
