#include "sobel.cuh"
#include <cmath>
#include <cuda_runtime.h>
#include <stdint.h>

// --- Shared memory tiling for Sobel edge detection ---
//
// The 3x3 Sobel stencil means each output pixel reads 8 neighbors from memory.
// Without tiling, every thread independently fetches its 3x3 window from global
// memory — massive redundancy since neighboring threads share most of their inputs.
//
// Strategy: each thread block cooperatively loads a (TILE_W+2) x (TILE_H+2) tile
// into shared memory, including a 1-pixel halo around the edges for the stencil.
// Interior pixels then read all neighbors from shared memory (~100x lower latency
// than global). This reduces global memory traffic from ~9 reads/pixel to ~1 read/pixel
// for interior threads.
//
// We tile per-channel (R, G, B separately) to keep shared memory usage reasonable:
//   18 x 18 x 1 byte = 324 bytes per channel pass, well within the 48KB limit.

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
    // Output pixel coordinates
    int x = blockIdx.x * TILE_W + threadIdx.x;
    int y = blockIdx.y * TILE_H + threadIdx.y;

    int idx = (y * width + x) * 4;

    // Border pixels: copy unchanged (no valid 3x3 neighborhood)
    if (x < width && y < height) {
        if (x == 0 || y == 0 || x == width - 1 || y == height - 1) {
            out[idx + 0] = in[idx + 0];
            out[idx + 1] = in[idx + 1];
            out[idx + 2] = in[idx + 2];
            out[idx + 3] = in[idx + 3];
            // Don't return yet — thread still participates in shared memory loads below
        }
    }

    // Shared memory tile: (TILE_W + 2) x (TILE_H + 2) for one channel at a time.
    // Processing channels sequentially keeps shared memory small (324 bytes).
    __shared__ uint8_t tile[(TILE_H + 2) * (TILE_W + 2)];

    const int tile_pitch = TILE_W + 2;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // Process R, G, B channels with shared memory tiling
    #pragma unroll
    for (int c = 0; c < 3; c++) {
        // --- Cooperative tile load ---
        // Each of the 16x16 = 256 threads loads one interior pixel.
        // Halo cells (border ring) require extra loads by edge threads.

        // Source coordinates for this thread's pixel in the halo-extended tile
        int src_x = blockIdx.x * TILE_W - 1 + tx;
        int src_y = blockIdx.y * TILE_H - 1 + ty;

        // Clamp to image bounds
        int cx = min(max(src_x, 0), width - 1);
        int cy = min(max(src_y, 0), height - 1);
        tile[ty * tile_pitch + tx] = in[(cy * width + cx) * 4 + c];

        // Right halo column: threads with tx < 2 load the extra 2 columns
        if (tx < 2) {
            int hx = min(max(src_x + TILE_W, 0), width - 1);
            tile[ty * tile_pitch + (tx + TILE_W)] = in[(cy * width + hx) * 4 + c];
        }

        // Bottom halo rows: threads with ty < 2 load the extra 2 rows
        if (ty < 2) {
            int hy = min(max(src_y + TILE_H, 0), height - 1);
            tile[(ty + TILE_H) * tile_pitch + tx] = in[(hy * width + cx) * 4 + c];
        }

        // Bottom-right corner: thread (0,0) and (1,0), (0,1), (1,1) load corners
        if (tx < 2 && ty < 2) {
            int hx = min(max(src_x + TILE_W, 0), width - 1);
            int hy = min(max(src_y + TILE_H, 0), height - 1);
            tile[(ty + TILE_H) * tile_pitch + (tx + TILE_W)] = in[(hy * width + hx) * 4 + c];
        }

        __syncthreads();

        // --- Apply Sobel from shared memory ---
        if (x > 0 && y > 0 && x < width - 1 && y < height - 1) {
            // Thread's position in tile is (tx+1, ty+1) — offset by halo
            int t = (ty + 1) * tile_pitch + (tx + 1);

            int p00 = tile[t - tile_pitch - 1];
            int p01 = tile[t - tile_pitch    ];
            int p02 = tile[t - tile_pitch + 1];
            int p10 = tile[t             - 1];
            int p12 = tile[t             + 1];
            int p20 = tile[t + tile_pitch - 1];
            int p21 = tile[t + tile_pitch    ];
            int p22 = tile[t + tile_pitch + 1];

            int gx = -p00 + p02 - 2*p10 + 2*p12 - p20 + p22;
            int gy = -p00 - 2*p01 - p02 + p20 + 2*p21 + p22;

            float mag = sqrtf(static_cast<float>(gx * gx + gy * gy));
            out[idx + c] = sobel_clamp_u8(static_cast<int>(mag));
        }

        __syncthreads();  // Ensure tile is fully consumed before next channel overwrites it
    }

    // Alpha channel: straight copy (no filtering)
    if (x < width && y < height) {
        out[idx + 3] = in[idx + 3];
    }
}

void launch_sobel(
    const uint8_t* d_in,
    uint8_t* d_out,
    int width,
    int height
) {
    dim3 block(TILE_W, TILE_H);
    dim3 grid(
        (width + TILE_W - 1) / TILE_W,
        (height + TILE_H - 1) / TILE_H
    );

    sobel_tiled_rgba8_kernel<<<grid, block>>>(d_in, d_out, width, height);
}
