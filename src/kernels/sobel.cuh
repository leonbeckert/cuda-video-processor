#pragma once
#include <stdint.h>
#include <cuda_runtime.h>

void launch_sobel(
    const uint8_t* d_in,
    uint8_t* d_out,
    int width,
    int height,
    cudaStream_t stream = 0
);
