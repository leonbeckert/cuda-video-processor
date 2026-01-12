#pragma once
#include <stdint.h>

// Computes average R/G/B over an RGBA8 image on GPU.
// d_rgba: device pointer to width*height*4 bytes (RGBA interleaved)
// Outputs: red_avg, green_avg, blue_avg (host ints)
void launch_rgb_average(
    const uint8_t* d_rgba,
    int width,
    int height,
    int* red_avg,
    int* green_avg,
    int* blue_avg
);
