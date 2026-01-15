#ifndef COMMON_H
#define COMMON_H
#include <iostream>
#include <memory>
#include <cstring>

// Define GLEW_STATIC before any OpenGL includes
#ifndef GLEW_STATIC
#define GLEW_STATIC
#endif

// Include GLEW first, before any other OpenGL headers
#include <GL/glew.h>

// Now include CUDA headers
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "cuda_check.h"

// OpenGL error checking macro
#define GL_CHECK(stmt) \
    do { \
        stmt; \
        GLenum error = glGetError(); \
        if (error != GL_NO_ERROR) { \
            std::cerr << "OpenGL error: " << error << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while(0)

// Pinned (page-locked) memory for host-side frame buffers.
// Regular malloc'd memory must be copied to a pinned staging buffer by the
// CUDA driver before DMA transfer to GPU. cudaMallocHost pins pages directly,
// enabling:
//   1. DMA transfers that bypass the CPU page table — higher H2D/D2H bandwidth
//   2. Truly asynchronous cudaMemcpyAsync (with pageable memory, async calls
//      silently fall back to synchronous behavior)
// Tradeoff: pinned memory is a limited OS resource and slower to allocate than
// regular heap memory. Ideally the caller pre-allocates frames and reuses them
// rather than allocating per-frame — see VideoReader::readFrame() which
// currently reallocates on each call (a known inefficiency).

struct VideoFrame {
    int width;
    int height;
    int channels;
    unsigned char* data;
    bool pinned;  // track allocation type for correct deallocation

    VideoFrame() : width(0), height(0), channels(0), data(nullptr), pinned(false) {}

    VideoFrame(int w, int h, int c) : width(w), height(h), channels(c), pinned(false) {
        size_t bytes = (size_t)width * height * channels;
        cudaError_t err = cudaMallocHost((void**)&data, bytes);
        if (err == cudaSuccess) {
            pinned = true;
        } else {
            // Fallback to pageable if pinned allocation fails (e.g., resource exhaustion)
            data = new unsigned char[bytes];
            pinned = false;
        }
    }

    ~VideoFrame() {
        if (data) {
            if (pinned) {
                cudaFreeHost(data);
            } else {
                delete[] data;
            }
        }
    }

    // Copy constructor — deep copy with pinned allocation attempt
    VideoFrame(const VideoFrame& other)
        : width(other.width), height(other.height), channels(other.channels), pinned(false)
    {
        if (other.data) {
            size_t bytes = (size_t)width * height * channels;
            cudaError_t err = cudaMallocHost((void**)&data, bytes);
            if (err == cudaSuccess) {
                pinned = true;
            } else {
                data = new unsigned char[bytes];
                pinned = false;
            }
            memcpy(data, other.data, bytes);
        } else {
            data = nullptr;
        }
    }

    // Move constructor — transfers ownership, no allocation
    VideoFrame(VideoFrame&& other) noexcept
        : width(other.width), height(other.height), channels(other.channels),
          data(other.data), pinned(other.pinned)
    {
        other.data = nullptr;
        other.pinned = false;
    }

    VideoFrame& operator=(const VideoFrame& other) {
        if (this != &other) {
            if (data) {
                if (pinned) cudaFreeHost(data);
                else delete[] data;
            }
            width = other.width;
            height = other.height;
            channels = other.channels;
            pinned = false;
            if (other.data) {
                size_t bytes = (size_t)width * height * channels;
                cudaError_t err = cudaMallocHost((void**)&data, bytes);
                if (err == cudaSuccess) {
                    pinned = true;
                } else {
                    data = new unsigned char[bytes];
                    pinned = false;
                }
                memcpy(data, other.data, bytes);
            } else {
                data = nullptr;
            }
        }
        return *this;
    }

    // Move assignment — transfers ownership, no allocation
    VideoFrame& operator=(VideoFrame&& other) noexcept {
        if (this != &other) {
            if (data) {
                if (pinned) cudaFreeHost(data);
                else delete[] data;
            }
            width = other.width;
            height = other.height;
            channels = other.channels;
            data = other.data;
            pinned = other.pinned;
            other.data = nullptr;
            other.pinned = false;
        }
        return *this;
    }
};

#endif // COMMON_H
