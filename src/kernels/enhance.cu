// --- Per-pixel contrast/brightness enhancement (in-place) ---
//
// Purely memory-bound: each pixel reads and writes a 4-byte RGBA word, only
// modifying R/G/B channels. No data reuse between pixels, so shared memory
// tiling would add overhead without benefit (unlike Sobel where neighboring
// threads read overlapping data).

#include "enhance.cuh"
#include <cuda_runtime.h>
#include <stdint.h>

__device__ __forceinline__ uint8_t enhance_clamp_u8(float v) {
    v = v < 0.f ? 0.f : (v > 255.f ? 255.f : v);
    return (uint8_t)v;
}

__global__ void enhance_rgba8_kernel(
    uint8_t* __restrict__ img,
    int width, int height,
    float contrast, float brightness
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = (y * width + x) * 4;

    #pragma unroll
    for (int c = 0; c < 3; ++c) {
        float p = (float)img[idx + c];
        p = p * contrast + brightness;
        img[idx + c] = enhance_clamp_u8(p);
    }
    // Alpha unchanged
}

void launch_enhancement(
    uint8_t* d_img,
    int width, int height,
    float contrast, float brightness,
    cudaStream_t stream
) {
    dim3 block(16, 16);
    dim3 grid(
        (width  + block.x - 1) / block.x,
        (height + block.y - 1) / block.y
    );
    enhance_rgba8_kernel<<<grid, block, 0, stream>>>(d_img, width, height, contrast, brightness);
}
