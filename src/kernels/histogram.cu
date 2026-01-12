// --- RGB histogram via block-level parallel reduction ---
//
// Computes per-channel (R, G, B) average values across all pixels. Each thread
// accumulates partial sums in registers via a grid-stride loop, then a shared
// memory parallel reduction combines results within each block. Finally, one
// atomicAdd per channel per block merges into the global accumulator — reducing
// global atomic traffic from millions (one per pixel) to thousands (one per block).
//
// 64-bit accumulators are required: 4K = 8.3M pixels * max 255 = 2.1 billion,
// which overflows uint32.
//
// Known limitation: the launch function allocates and frees device memory on
// every call (cudaMalloc/cudaFree for the 3 accumulators). This was inherited
// from the course skeleton and causes host-device synchronization that
// dominates pipeline time — it's the primary reason the full pipeline benchmark
// shows lower FPS on the 4090 than the 3090 (see README). Pre-allocating these
// buffers would eliminate this bottleneck.

#include "histogram.cuh"
#include "../cuda_check.h"
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

// Block-level reduction for 3 channels (R,G,B) in one pass.
// Accumulates into 64-bit to avoid overflow for large images (4K = 8.3M pixels).
__global__ void rgb_sum_rgba8_kernel(
    const uint8_t* __restrict__ img,
    int pixel_count,
    unsigned long long* __restrict__ out_r,
    unsigned long long* __restrict__ out_g,
    unsigned long long* __restrict__ out_b
) {
    unsigned long long r = 0, g = 0, b = 0;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < pixel_count; i += stride) {
        int base = i * 4;
        r += img[base + 0];
        g += img[base + 1];
        b += img[base + 2];
    }

    // Reduce within block using shared memory
    __shared__ unsigned long long sh_r[256];
    __shared__ unsigned long long sh_g[256];
    __shared__ unsigned long long sh_b[256];

    int lane = threadIdx.x;
    sh_r[lane] = r;
    sh_g[lane] = g;
    sh_b[lane] = b;
    __syncthreads();

    // Parallel reduction (assumes blockDim.x is power of two)
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (lane < offset) {
            sh_r[lane] += sh_r[lane + offset];
            sh_g[lane] += sh_g[lane + offset];
            sh_b[lane] += sh_b[lane + offset];
        }
        __syncthreads();
    }

    // One atomic per channel per block
    if (lane == 0) {
        atomicAdd(out_r, sh_r[0]);
        atomicAdd(out_g, sh_g[0]);
        atomicAdd(out_b, sh_b[0]);
    }
}

void launch_rgb_average(
    const uint8_t* d_rgba,
    int width,
    int height,
    int* red_avg,
    int* green_avg,
    int* blue_avg
) {
    if (!d_rgba || width <= 0 || height <= 0 || !red_avg || !green_avg || !blue_avg) {
        return;
    }

    const int pixel_count = width * height;

    unsigned long long* d_r = nullptr;
    unsigned long long* d_g = nullptr;
    unsigned long long* d_b = nullptr;

    CUDA_CHECK(cudaMalloc(&d_r, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_g, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_b, sizeof(unsigned long long)));

    CUDA_CHECK(cudaMemset(d_r, 0, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_g, 0, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_b, 0, sizeof(unsigned long long)));

    dim3 block(256);
    int max_blocks = 1024;
    int grid = (pixel_count + block.x - 1) / block.x;
    if (grid > max_blocks) grid = max_blocks;

    rgb_sum_rgba8_kernel<<<grid, block>>>(d_rgba, pixel_count, d_r, d_g, d_b);
    CUDA_CHECK(cudaGetLastError());

    unsigned long long h_r = 0, h_g = 0, h_b = 0;
    CUDA_CHECK(cudaMemcpy(&h_r, d_r, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_g, d_g, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_b, d_b, sizeof(unsigned long long), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_r));
    CUDA_CHECK(cudaFree(d_g));
    CUDA_CHECK(cudaFree(d_b));

    *red_avg   = (int)(h_r / (unsigned long long)pixel_count);
    *green_avg = (int)(h_g / (unsigned long long)pixel_count);
    *blue_avg  = (int)(h_b / (unsigned long long)pixel_count);
}
