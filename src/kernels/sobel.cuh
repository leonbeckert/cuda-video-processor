#pragma once
#include <stdint.h>

void launch_sobel(
    const uint8_t* d_in,
    uint8_t* d_out,
    int width,
    int height
);
