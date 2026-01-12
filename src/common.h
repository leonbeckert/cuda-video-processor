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

// Video frame structure
struct VideoFrame {
    int width;
    int height;
    int channels;
    unsigned char* data;

    VideoFrame() : width(0), height(0), channels(0), data(nullptr) {}

    VideoFrame(int w, int h, int c) : width(w), height(h), channels(c) {
        data = new unsigned char[width * height * channels];
    }

    ~VideoFrame() {
        delete[] data;
    }

    VideoFrame(const VideoFrame& other) : width(other.width), height(other.height), channels(other.channels) {
        if (other.data) {
            data = new unsigned char[width * height * channels];
            memcpy(data, other.data, width * height * channels);
        } else {
            data = nullptr;
        }
    }

    VideoFrame& operator=(const VideoFrame& other) {
        if (this != &other) {
            delete[] data;
            width = other.width;
            height = other.height;
            channels = other.channels;
            if (other.data) {
                data = new unsigned char[width * height * channels];
                memcpy(data, other.data, width * height * channels);
            } else {
                data = nullptr;
            }
        }
        return *this;
    }
};

#endif // COMMON_H
