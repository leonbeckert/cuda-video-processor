#include <iostream>
#include <memory>
#include <chrono>
#include <thread>
#include <vector>
#include <cmath>

#include "common.h"
#include "video_reader.h"
#include "opengl_renderer.h"

#include <cuda_runtime.h>
#include <stdint.h>

#include "kernels/sobel.cuh"
#include "kernels/enhance.cuh"
#include "kernels/histogram.cuh"
#include "kernels/brightness_histogram.cuh"

const int NUM_BINS = 25;

// Toggle state for real-time filter control
static bool enable_sobel = false;
static bool enable_enhancement = false;
static bool enable_color_histogram = true;
static bool enable_brightness_histogram = true;

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (action == GLFW_PRESS) {
        switch (key) {
            case GLFW_KEY_S:
                enable_sobel = !enable_sobel;
                std::cout << "Sobel Filter: " << (enable_sobel ? "ON" : "OFF") << std::endl;
                break;
            case GLFW_KEY_E:
                enable_enhancement = !enable_enhancement;
                std::cout << "Image Enhancement: " << (enable_enhancement ? "ON" : "OFF") << std::endl;
                break;
            case GLFW_KEY_C:
                enable_color_histogram = !enable_color_histogram;
                std::cout << "Color Histogram: " << (enable_color_histogram ? "ON" : "OFF") << std::endl;
                break;
            case GLFW_KEY_H:
                enable_brightness_histogram = !enable_brightness_histogram;
                std::cout << "Brightness Histogram: " << (enable_brightness_histogram ? "ON" : "OFF") << std::endl;
                break;
            case GLFW_KEY_ESCAPE:
                glfwSetWindowShouldClose(window, GLFW_TRUE);
                break;
        }
    }
}

class VideoPlayer {
private:
    std::unique_ptr<VideoReader> video_reader;
    std::unique_ptr<OpenGLRenderer> opengl_renderer;

    int window_width = 1920;
    int window_height = 1080;

    uint8_t* d_in = nullptr;
    uint8_t* d_out = nullptr;
    size_t d_bytes = 0;

public:
    VideoPlayer() {
        video_reader = std::make_unique<VideoReader>();
        opengl_renderer = std::make_unique<OpenGLRenderer>();
    }

    bool initialize(const std::string& video_filename) {
        std::cout << "=== Initializing Video Player ===" << std::endl;

        std::cout << "1. Opening video file: " << video_filename << std::endl;
        if (!video_reader->open(video_filename)) {
            std::cerr << "FAILED: Could not open video file: " << video_filename << std::endl;
            return false;
        }
        std::cout << "   SUCCESS: Video file opened" << std::endl;

        int video_width = video_reader->getWidth();
        int video_height = video_reader->getHeight();
        double fps = video_reader->getFPS();

        std::cout << "   Video properties: " << video_width << "x" << video_height
                  << " @ " << fps << " fps" << std::endl;

        std::cout << "2. Initializing OpenGL renderer..." << std::endl;
        if (!opengl_renderer->initialize(window_width, window_height,
                                        video_width, video_height)) {
            std::cerr << "FAILED: Could not initialize OpenGL renderer" << std::endl;
            return false;
        }
        std::cout << "   SUCCESS: OpenGL renderer initialized" << std::endl;

        glfwSetKeyCallback(opengl_renderer->getWindow(), key_callback);
        std::cout << "   Keyboard controls registered:" << std::endl;
        std::cout << "     [S] Toggle Sobel Filter" << std::endl;
        std::cout << "     [E] Toggle Image Enhancement" << std::endl;
        std::cout << "     [C] Toggle Color Histogram" << std::endl;
        std::cout << "     [H] Toggle Brightness Histogram" << std::endl;
        std::cout << "     [ESC] Exit" << std::endl;
        std::cout << "=== Video Player Initialization Complete ===" << std::endl;
        return true;
    }

    void run() {
        VideoFrame frame;
        double target_frame_time = 1.0 / video_reader->getFPS();

        std::cout << "=== Starting Video Playback ===" << std::endl;
        std::cout << "Target FPS: " << video_reader->getFPS() << std::endl;
        std::cout << "Press ESC or close window to exit" << std::endl;

        int frame_count = 0;
        double target_fps = video_reader->getFPS();
        double actual_fps = 0.0;

        while (!opengl_renderer->shouldClose()) {
            auto frame_start_time = std::chrono::high_resolution_clock::now();

            if (video_reader->readFrame(frame)) {
                int width  = frame.width;
                int height = frame.height;
                size_t bytes = (size_t)width * (size_t)height * 4;

                if (bytes != d_bytes) {
                    if (d_in)  CUDA_CHECK(cudaFree(d_in));
                    if (d_out) CUDA_CHECK(cudaFree(d_out));
                    CUDA_CHECK(cudaMalloc(&d_in, bytes));
                    CUDA_CHECK(cudaMalloc(&d_out, bytes));
                    d_bytes = bytes;
                }

                CUDA_CHECK(cudaMemcpy(d_in, frame.data, bytes, cudaMemcpyHostToDevice));

                if (enable_sobel) {
                    launch_sobel(d_in, d_out, width, height);
                    CUDA_CHECK(cudaGetLastError());
                    std::swap(d_in, d_out);
                }
                if (enable_enhancement) {
                    launch_enhancement(d_in, width, height, 1.2f, 20.0f);
                    CUDA_CHECK(cudaGetLastError());
                }

                int red_avg = 0, green_avg = 0, blue_avg = 0;
                if (enable_color_histogram) {
                    launch_rgb_average(d_in, width, height, &red_avg, &green_avg, &blue_avg);
                }

                int brightness_histogram[NUM_BINS];
                if (enable_brightness_histogram) {
                    launch_brightness_histogram(d_in, width, height, NUM_BINS, brightness_histogram);
                }

                CUDA_CHECK(cudaMemcpy(frame.data, d_in, bytes, cudaMemcpyDeviceToHost));

                glBindTexture(GL_TEXTURE_2D, opengl_renderer->getTextureID());
                glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, frame.width, frame.height,
                                GL_RGBA, GL_UNSIGNED_BYTE, frame.data);
                glBindTexture(GL_TEXTURE_2D, 0);

                opengl_renderer->render();

                if (enable_color_histogram) {
                    opengl_renderer->renderColorHistogram(red_avg, green_avg, blue_avg);
                }
                if (enable_brightness_histogram) {
                    opengl_renderer->renderBrightnessHistogram(brightness_histogram, NUM_BINS);
                }

                opengl_renderer->swapBuffers();

                frame_count++;

                auto frame_end_time = std::chrono::high_resolution_clock::now();
                auto frame_duration = std::chrono::duration<double>(frame_end_time - frame_start_time).count();
                actual_fps = (frame_duration > 0.0) ? 1.0 / frame_duration : 0.0;

                char title[512];
                snprintf(title, sizeof(title),
                         "Video Player | [S]obel:%s [E]nhance:%s [C]olor:%s [H]istogram:%s | FPS: %.1f/%.1f",
                         enable_sobel ? "ON" : "OFF",
                         enable_enhancement ? "ON" : "OFF",
                         enable_color_histogram ? "ON" : "OFF",
                         enable_brightness_histogram ? "ON" : "OFF",
                         actual_fps, target_fps);
                opengl_renderer->setWindowTitle(title);

                if (frame_count % 100 == 0) {
                    std::cout << "Processed " << frame_count << " frames" << std::endl;
                }

                if (frame_duration < target_frame_time) {
                    std::this_thread::sleep_for(
                        std::chrono::duration<double>(target_frame_time - frame_duration));
                }
            } else {
                std::cout << "End of video reached or error occurred, continuing with loop..." << std::endl;
                opengl_renderer->render();
                opengl_renderer->swapBuffers();
                std::this_thread::sleep_for(std::chrono::milliseconds(16));
            }

            opengl_renderer->pollEvents();
        }
    }

    void cleanup() {
        if (opengl_renderer) {
            opengl_renderer->cleanup();
        }
        if (video_reader) {
            video_reader->close();
        }
        if (d_in)  CUDA_CHECK(cudaFree(d_in));
        if (d_out) CUDA_CHECK(cudaFree(d_out));
        d_in = d_out = nullptr;
        d_bytes = 0;
    }
};

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <video_file>" << std::endl;
        std::cerr << "Example: " << argv[0] << " sample_4k_video.mp4" << std::endl;
        return 1;
    }

    std::string video_filename = argv[1];
    std::cout << "CUDA-OpenGL Video Player" << std::endl;
    std::cout << "Loading video: " << video_filename << std::endl;

    VideoPlayer player;

    if (!player.initialize(video_filename)) {
        std::cerr << "Failed to initialize video player" << std::endl;
        return -1;
    }

    try {
        player.run();
    } catch (const std::exception& e) {
        std::cerr << "Error during playback: " << e.what() << std::endl;
        player.cleanup();
        return -1;
    }

    player.cleanup();
    std::cout << "Video player terminated successfully" << std::endl;

    return 0;
}
