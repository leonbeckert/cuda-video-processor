#include "sobel.cuh"
#include <cmath>
#include <cuda_runtime.h>
#include <stdint.h>

// --- Shared memory tiling for Sobel edge detection ---
//
// The 3x3 Sobel stencil means each output pixel reads 8 neighbors.
// Without tiling, every thread independently fetches its 3x3 window from global
// memory — massive redundancy since neighboring threads share most inputs.
//
// Strategy: each thread block cooperatively loads a (TILE_W+2) x (TILE_H+2) tile
// of packed RGBA pixels (uint32) into shared memory, including a 1-pixel halo for
// the stencil. All channels are loaded in ONE pass — no per-channel iteration.
// Interior pixels then read all neighbors from shared memory (~20-30x lower latency
// than global DRAM on Ampere/Ada). This reduces global memory traffic from ~9 reads/pixel to ~1 read/pixel.
//
// Shared memory: 18 x 18 x 4 bytes = 1296 bytes per block (trivial vs 48KB limit).

#define TILE_W 16
#define TILE_H 16

__device__ __forceinline__ uint8_t sobel_clamp_u8(int v) {
    return static_cast<uint8_t>(v < 0 ? 0 : (v > 255 ? 255 : v));
}

__global__ void sobel_tiled_rgba8_kernel(
    const uint8_t* __restrict__ in,
    uint8_t* __restrict__ out,
    int width, int height
) {
    // Reinterpret as uint32 for packed RGBA loads (4 bytes per pixel, one transaction)
    const uint32_t* __restrict__ in32 = reinterpret_cast<const uint32_t*>(in);
    uint32_t* __restrict__ out32 = reinterpret_cast<uint32_t*>(out);

    int x = blockIdx.x * TILE_W + threadIdx.x;
    int y = blockIdx.y * TILE_H + threadIdx.y;

    // Shared memory tile: (TILE_W+2) x (TILE_H+2) packed RGBA pixels
    __shared__ uint32_t tile[(TILE_H + 2) * (TILE_W + 2)];

    const int tile_pitch = TILE_W + 2;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // --- Cooperative tile load (single pass, all 4 channels at once) ---
    // Each thread loads its corresponding pixel from the halo-extended region.
    int src_x = blockIdx.x * TILE_W - 1 + tx;
    int src_y = blockIdx.y * TILE_H - 1 + ty;

    // Clamp to image bounds
    int cx = min(max(src_x, 0), width - 1);
    int cy = min(max(src_y, 0), height - 1);
    tile[ty * tile_pitch + tx] = in32[cy * width + cx];

    // Right halo columns (2 extra columns)
    if (tx < 2) {
        int hx = min(max(src_x + TILE_W, 0), width - 1);
        tile[ty * tile_pitch + (tx + TILE_W)] = in32[cy * width + hx];
    }

    // Bottom halo rows (2 extra rows)
    if (ty < 2) {
        int hy = min(max(src_y + TILE_H, 0), height - 1);
        tile[(ty + TILE_H) * tile_pitch + tx] = in32[hy * width + cx];
    }

    // Bottom-right corner (4 pixels)
    if (tx < 2 && ty < 2) {
        int hx = min(max(src_x + TILE_W, 0), width - 1);
        int hy = min(max(src_y + TILE_H, 0), height - 1);
        tile[(ty + TILE_H) * tile_pitch + (tx + TILE_W)] = in32[hy * width + hx];
    }

    __syncthreads();

    if (x >= width || y >= height) return;

    // Border pixels: copy unchanged
    if (x == 0 || y == 0 || x == width - 1 || y == height - 1) {
        out32[y * width + x] = in32[y * width + x];
        return;
    }

    // --- Apply Sobel from shared memory ---
    // Thread position in tile is (tx+1, ty+1) offset by halo
    int t = (ty + 1) * tile_pitch + (tx + 1);

    // Load 8 packed RGBA neighbors
    uint32_t px00 = tile[t - tile_pitch - 1];
    uint32_t px01 = tile[t - tile_pitch    ];
    uint32_t px02 = tile[t - tile_pitch + 1];
    uint32_t px10 = tile[t             - 1];
    uint32_t px12 = tile[t             + 1];
    uint32_t px20 = tile[t + tile_pitch - 1];
    uint32_t px21 = tile[t + tile_pitch    ];
    uint32_t px22 = tile[t + tile_pitch + 1];

    // Process R, G, B channels from packed pixels
    uint32_t result = 0;
    #pragma unroll
    for (int c = 0; c < 3; c++) {
        int shift = c * 8;
        int p00 = (px00 >> shift) & 0xFF;
        int p01 = (px01 >> shift) & 0xFF;
        int p02 = (px02 >> shift) & 0xFF;
        int p10 = (px10 >> shift) & 0xFF;
        int p12 = (px12 >> shift) & 0xFF;
        int p20 = (px20 >> shift) & 0xFF;
        int p21 = (px21 >> shift) & 0xFF;
        int p22 = (px22 >> shift) & 0xFF;

        int gx = -p00 + p02 - 2*p10 + 2*p12 - p20 + p22;
        int gy = -p00 - 2*p01 - p02 + p20 + 2*p21 + p22;

        float mag = sqrtf(static_cast<float>(gx * gx + gy * gy));
        uint32_t val = sobel_clamp_u8(static_cast<int>(mag));
        result |= (val << shift);
    }

    // Alpha channel: copy from input
    result |= (tile[(ty + 1) * tile_pitch + (tx + 1)] & 0xFF000000u);

    out32[y * width + x] = result;
}

void launch_sobel(
    const uint8_t* d_in,
    uint8_t* d_out,
    int width,
    int height,
    cudaStream_t stream
) {
    dim3 block(TILE_W, TILE_H);
    dim3 grid(
        (width + TILE_W - 1) / TILE_W,
        (height + TILE_H - 1) / TILE_H
    );

    sobel_tiled_rgba8_kernel<<<grid, block, 0, stream>>>(d_in, d_out, width, height);
}
