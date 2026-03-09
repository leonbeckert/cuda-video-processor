// --- Brightness histogram with shared memory privatization ---
//
// Each block maintains a private histogram in shared memory. Threads accumulate
// via atomicAdd to shared memory (much faster than global atomics), then a
// single atomicAdd per bin per block merges results into the global histogram.
// This reduces global atomic contention from 8.3M operations (one per pixel)
// to ~1024 * 25 bins (one per bin per block, grid capped at 1024).
//
// Template specialization on NUM_BINS lets the compiler size the shared array
// at compile time, enabling register placement when possible and full unrolling
// of bin-mapping arithmetic.
//
// Known limitation: same per-call cudaMalloc/cudaFree as the RGB histogram
// (see histogram.cu for explanation).

#include "brightness_histogram.cuh"
#include "../cuda_check.h"
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

// Fixed-point BT.709 grayscale conversion (deterministic, no float rounding variance)
__device__ __forceinline__ uint8_t gray709_u8(uint8_t r, uint8_t g, uint8_t b) {
    // 0.2126, 0.7152, 0.0722 scaled by 65536
    uint32_t y = 13933u * r + 46871u * g + 4732u * b + 32768u;
    return (uint8_t)(y >> 16);
}

// Per-block shared memory histogram with atomic reduction to global.
// Template on NUM_BINS for compile-time array sizing and loop unrolling.
template<int NUM_BINS>
__global__ void brightness_hist_rgba8_kernel(
    const uint8_t* __restrict__ img,
    int pixel_count,
    int* __restrict__ g_hist
) {
    __shared__ int sh_hist[NUM_BINS];

    for (int i = threadIdx.x; i < NUM_BINS; i += blockDim.x) {
        sh_hist[i] = 0;
    }
    __syncthreads();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < pixel_count; i += stride) {
        int base = i * 4;
        uint8_t r = img[base + 0];
        uint8_t g = img[base + 1];
        uint8_t b = img[base + 2];

        uint8_t gray = gray709_u8(r, g, b);
        int bin = (gray * NUM_BINS) >> 8;
        if (bin >= NUM_BINS) bin = NUM_BINS - 1;

        atomicAdd(&sh_hist[bin], 1);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < NUM_BINS; i += blockDim.x) {
        atomicAdd(&g_hist[i], sh_hist[i]);
    }
}

void launch_brightness_histogram(
    const uint8_t* d_rgba,
    int width,
    int height,
    int num_bins,
    int* h_histogram
) {
    if (!d_rgba || !h_histogram || width <= 0 || height <= 0) return;

    const int pixel_count = width * height;

    int* d_hist = nullptr;
    CUDA_CHECK(cudaMalloc(&d_hist, num_bins * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_hist, 0, num_bins * sizeof(int)));

    dim3 block(256);
    int grid = (pixel_count + block.x - 1) / block.x;
    if (grid > 1024) grid = 1024;

    switch (num_bins) {
        case 25:
            brightness_hist_rgba8_kernel<25><<<grid, block>>>(
                d_rgba, pixel_count, d_hist);
            break;
        default:
            CUDA_CHECK(cudaFree(d_hist));
            return;
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_histogram, d_hist, num_bins * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_hist));
}
