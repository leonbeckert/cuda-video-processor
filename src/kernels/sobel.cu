#include "sobel.cuh"
#include <cmath>
#include <cuda_runtime.h>
#include <stdint.h>

__device__ __forceinline__ uint8_t sobel_clamp_u8(int v) {
    return static_cast<uint8_t>(v < 0 ? 0 : (v > 255 ? 255 : v));
}

__global__ void sobel_rgba8_kernel(
    const uint8_t* __restrict__ in,
    uint8_t* __restrict__ out,
    int width, int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = (y * width + x) * 4;

    // Border pixels: copy input to output unchanged
    if (x == 0 || y == 0 || x == width - 1 || y == height - 1) {
        out[idx + 0] = in[idx + 0];
        out[idx + 1] = in[idx + 1];
        out[idx + 2] = in[idx + 2];
        out[idx + 3] = in[idx + 3];
        return;
    }

    // Sobel operator: 3x3 convolution for edge detection
    // Gx kernel:        Gy kernel:
    // -1  0  1          -1 -2 -1
    // -2  0  2           0  0  0
    // -1  0  1           1  2  1

    #pragma unroll
    for (int c = 0; c < 3; c++) {
        int gx = 0, gy = 0;

        // Load 3x3 neighborhood from global memory (9 reads per channel)
        int idx00 = ((y - 1) * width + (x - 1)) * 4 + c;
        int idx01 = ((y - 1) * width + (x    )) * 4 + c;
        int idx02 = ((y - 1) * width + (x + 1)) * 4 + c;
        int idx10 = ((y    ) * width + (x - 1)) * 4 + c;
        int idx12 = ((y    ) * width + (x + 1)) * 4 + c;
        int idx20 = ((y + 1) * width + (x - 1)) * 4 + c;
        int idx21 = ((y + 1) * width + (x    )) * 4 + c;
        int idx22 = ((y + 1) * width + (x + 1)) * 4 + c;

        int p00 = in[idx00], p01 = in[idx01], p02 = in[idx02];
        int p10 = in[idx10],                  p12 = in[idx12];
        int p20 = in[idx20], p21 = in[idx21], p22 = in[idx22];

        gx = -p00 + p02 - 2*p10 + 2*p12 - p20 + p22;
        gy = -p00 - 2*p01 - p02 + p20 + 2*p21 + p22;

        float mag = sqrtf(static_cast<float>(gx * gx + gy * gy));
        out[idx + c] = sobel_clamp_u8(static_cast<int>(mag));
    }

    // Alpha channel unchanged
    out[idx + 3] = in[idx + 3];
}

void launch_sobel(
    const uint8_t* d_in,
    uint8_t* d_out,
    int width,
    int height
) {
    dim3 block(16, 16);
    dim3 grid(
        (width + block.x - 1) / block.x,
        (height + block.y - 1) / block.y
    );

    sobel_rgba8_kernel<<<grid, block>>>(d_in, d_out, width, height);
}
