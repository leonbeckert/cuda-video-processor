# CUDA Video Processor

Real-time 4K video processing with CUDA-accelerated image filters and OpenGL rendering. Sobel edge detection at 3840x2160: **770x faster** than the single-threaded CPU baseline on an RTX 4090 (0.091 ms vs 70 ms per frame).

## Overview

A GPU-accelerated video player that applies Sobel edge detection, contrast/brightness enhancement, and real-time histogram analysis to 4K RGBA video streams. At 3840x2160, each frame is 33 MB — too large for CPU-based stencil operations to sustain interactive framerates. CUDA kernels with shared memory tiling, pinned memory for async DMA transfers, and a stream-based pipeline deliver sustained processing at hundreds of FPS with all filters enabled.

## Background

Extended from a university GPGPU course assignment. The original project provided a CPU video player skeleton with OpenGL rendering. I implemented all CUDA kernels (Sobel, enhancement, histogram), the shared memory tiling optimization, pinned memory allocation, CUDA streams double-buffering, and the benchmark harness with CPU reference validation.

## Pipeline Architecture

```
FFmpeg Decode ──► cudaMemcpyAsync ──► Sobel Kernel ──► Enhancement ──► Histograms ──► cudaMemcpy D2H ──► glTexSubImage2D ──► Display
    (CPU)         (Pinned H2D)       (Shared Mem      (In-place)     (Shared            (Pinned)          (CPU ► GPU
                   (Stream 1)         Tiling)                         Atomics +                             texture upload)
                                     (Stream 0)       (Stream 0)     Reduction)
```

**Double-buffered streams:** While Stream 0 runs kernels on frame N, Stream 1 uploads frame N+1 via async DMA. The H2D transfer overlaps with compute — effectively free when kernels take longer than the transfer.

## CUDA Optimizations

### Shared Memory Tiling (Sobel)

**Problem:** The 3x3 Sobel stencil causes each pixel to read 8 neighbors from global memory. Neighboring threads redundantly fetch overlapping data.

**Solution:** Each 16x16 thread block cooperatively loads an 18x18 tile of packed RGBA pixels (uint32) into shared memory, including a 1-pixel halo. All channels load in a single pass — the initial per-channel approach required 3x tile loads and 6 `__syncthreads` barriers, which was actually slower than naive global memory access. The single-pass uint32 approach uses only 1296 bytes of shared memory per block.

**Impact:** 1.12x faster than naive on RTX 4090 (728 vs 648 GB/s effective bandwidth). The improvement is modest because 4K frames (33 MB) fit within the 4090's 72 MB L2 cache, reducing the benefit of shared memory. On the RTX 3090 (6 MB L2), the tiling benefit is more sensitive to power/thermal state — at max power (480W) it reaches 1.14x, while at the default 270W limit the benefit is marginal. This demonstrates that shared memory tiling for stencil operations is most effective at larger data sizes or higher stencil radii.

### Pinned Memory for Async Transfers

**Problem:** Standard `malloc` memory requires the CUDA driver to copy data into an internal pinned staging buffer before DMA transfer. This makes `cudaMemcpyAsync` silently synchronous.

**Solution:** `cudaMallocHost` pins pages directly, enabling true DMA transfers that bypass CPU page table walks. The `VideoFrame` struct allocates via `cudaMallocHost` with automatic fallback to pageable memory if pinned allocation fails.

**Impact:** Required prerequisite for CUDA streams to actually overlap transfers with compute.

### CUDA Streams Pipeline

**Problem:** Sequential frame processing wastes time — the GPU's DMA copy engine sits idle during kernel execution, and compute units idle during memory transfers.

**Solution:** Two CUDA streams with double-buffered device memory. Stream 0 (compute) runs kernels on the current frame while Stream 1 (transfer) uploads the next frame. The GPU executes DMA transfers and kernels simultaneously on different hardware engines.

**Impact:** Improves throughput (FPS) without changing individual kernel latency.

### Brightness Histogram — Shared Memory Privatization

Each thread block maintains a private histogram in shared memory using `atomicAdd`. After all pixels are processed, a single `atomicAdd` per bin merges the block result into global memory. This reduces global atomic contention from millions of operations (one per pixel) to thousands (one per bin per block). The kernel uses `template<int NUM_BINS>` for compile-time bin count, enabling register placement and full unrolling of bin-mapping arithmetic.

### RGB Histogram — Parallel Reduction

The RGB average kernel uses a different pattern: block-level parallel reduction with `__syncthreads` barriers to compute per-channel sums, then a final device-to-host copy of the 3-element result. No shared-memory privatization needed since the output is a single average per channel, not a distribution.

## Performance

All benchmarks: 4K RGBA synthetic frame (3840x2160, 33.18 MB), 20 warmup iterations, L2 cache flush between measurements, 100 timed iterations, median reported. GPU clocks locked for stability.

### CPU vs GPU Speedup

| Kernel | CPU (ms) | RTX 4090 (ms) | Speedup | RTX 3090 (ms) | Speedup |
|--------|----------|---------------|---------|---------------|---------|
| Sobel (tiled) | 70.1 | 0.091 | **770x** | 0.132 | **531x** |
| Sobel (naive) | 70.1 | 0.102 | **685x** | 0.150 | **466x** |
| Enhancement | 22.6 | 0.113 | **201x** | 0.094 | **245x** |
| RGB Histogram | 2.9 | 0.163 | **18x** | 0.148 | **20x** |
| Brightness Hist | 8.4 | 0.142 | **59x** | 0.141 | **59x** |

CPU baseline: single-threaded x86 on the same machine. GPU numbers are median of 100 runs with L2 flush.

### Effective Bandwidth

| Kernel | RTX 4090 BW | % Peak | RTX 3090 BW | % Peak |
|--------|-------------|--------|-------------|--------|
| Sobel (tiled) | 728 GB/s | 72% | 502 GB/s | 54% |
| Sobel (naive) | 648 GB/s | 64% | 441 GB/s | 47% |
| Enhancement | 589 GB/s | 58% | 704 GB/s | 75% |
| RGB Histogram | 204 GB/s | 20% | 224 GB/s | 24% |
| Brightness Hist | 233 GB/s | 23% | 235 GB/s | 25% |

### Full Pipeline (all filters enabled, 1000 frames)

| Metric | RTX 3090 | RTX 4090 |
|--------|----------|----------|
| Sustained FPS | 332 | 96 |
| Frame latency | 3.01 ms | 10.41 ms |
| CPU baseline FPS | 9.9 | 9.9 |
| Pipeline speedup | **34x** | **10x** |

### Power Scaling (max power limits removed)

| GPU | Default PL | Max PL | Sobel Tiled (default) | Sobel Tiled (max) | Delta |
|-----|-----------|--------|----------------------|-------------------|-------|
| RTX 3090 | 270W | 480W | 441 GB/s | 502 GB/s | +14% |
| RTX 4090 | 240W | 600W | 728 GB/s | 728 GB/s | +0% |

### Interpretation

**Why don't bandwidth-bound kernels scale with the 4090's 2.3x compute advantage?** These kernels are memory-bound: the Sobel kernel has an arithmetic intensity of ~0.5 FLOPs/byte, placing it firmly on the memory-bound side of the roofline. The 4090's memory bandwidth is only 1.08x higher than the 3090 (1008 vs 936 GB/s), so kernel times scale with bandwidth, not compute.

**Why is the 4090 pipeline FPS lower than the 3090?** The histogram kernels allocate and free GPU memory per frame (`cudaMalloc`/`cudaFree` inside each call). This host-device synchronization overhead dominates the pipeline on the 4090. Pre-allocating histogram buffers would eliminate this bottleneck.

**Why do histogram kernels only reach ~20-25% peak bandwidth?** Atomic contention. Even with shared memory privatization (one histogram per block), the shared-memory `atomicAdd` serializes threads that hash to the same bin. With only 25 bins and 256 threads per block, collisions are frequent.

**Power limit impact is asymmetric.** The RTX 3090 gains +14% effective bandwidth when power is raised from 270W to 480W — at 270W, the memory controller itself is power-constrained, not just the compute cores. The RTX 4090 shows no measurable change because its 240W limit already provides sufficient power for the memory subsystem at this workload.

### Correctness Validation

All kernels pass correctness checks against CPU reference implementations. Sobel and enhancement allow a max per-pixel error of 1 due to floating-point rounding differences between CPU and GPU `sqrtf` implementations. Histogram kernels match exactly.

## Building

### Prerequisites
- CUDA Toolkit 11.8+ (required for SM 89 / Ada Lovelace)
- CMake 3.18+
- FFmpeg development libraries (`libavformat-dev libavcodec-dev libavutil-dev libswscale-dev`)
- OpenGL, GLFW, GLEW (GLFW and GLEW are fetched automatically by CMake)

### Build
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### Build with benchmarks
```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_BENCHMARKS=ON
make -j$(nproc)
```

### Run
```bash
./cuda-video-processor <path-to-video.mp4>
# or headless benchmark:
./benchmark --device 0
```

### Keyboard Controls
- `S` — Toggle Sobel edge detection
- `E` — Toggle contrast/brightness enhancement
- `C` — Toggle RGB color histogram overlay
- `H` — Toggle brightness histogram overlay
- `ESC` — Exit

## Test Environment

| Property | RTX 3090 | RTX 4090 |
|----------|----------|----------|
| GPU | NVIDIA GeForce RTX 3090 | NVIDIA GeForce RTX 4090 |
| SM Count | 82 | 128 |
| Memory | 24 GB GDDR6X, 384-bit | 24 GB GDDR6X, 384-bit |
| L2 Cache | 6 MB | 72 MB |
| Peak BW (spec) | 936 GB/s | 1008 GB/s |
| Compute Capability | 8.6 (Ampere) | 8.9 (Ada Lovelace) |
| CUDA Toolkit | 13.1 | 13.1 |
| Driver | 590.44.01 | 590.44.01 |
| Power Limit | 270W (benchmark) / 480W (max) | 240W (benchmark) / 600W (max) |
| OS | Debian 12 | Debian 12 |

## References

- [NVIDIA CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [How to Implement Performance Metrics in CUDA C/C++](https://developer.nvidia.com/blog/how-implement-performance-metrics-cuda-cc/)
- [Using Shared Memory in CUDA C/C++](https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/)
- [An Efficient Histogram Kernel (CUDA Handbook)](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory)
