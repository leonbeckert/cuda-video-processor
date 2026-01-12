#pragma once
#include <stdint.h>

// Computes brightness histogram (NUM_BINS bins) from RGBA8 image on GPU.
// d_rgba: device pointer to width*height*4 bytes
// h_histogram: host pointer to num_bins ints (output)
void launch_brightness_histogram(
    const uint8_t* d_rgba,
    int width,
    int height,
    int num_bins,
    int* h_histogram
);
