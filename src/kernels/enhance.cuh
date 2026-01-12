#pragma once
#include <stdint.h>

void launch_enhancement(
    uint8_t* __restrict__ img,
    int width, int height,
    float contrast, float brightness
);
