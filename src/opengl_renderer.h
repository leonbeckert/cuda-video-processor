#pragma once

#include "common.h"
#include <GLFW/glfw3.h>
#include <string>

class OpenGLRenderer {
private:
    GLFWwindow* window;
    GLuint texture_id;

    int window_width, window_height;
    int texture_width, texture_height;

public:
    OpenGLRenderer();
    ~OpenGLRenderer();

    bool initialize(int window_w, int window_h, int texture_w, int texture_h);
    bool createTexture();
    void render();
    void cleanup();

    bool shouldClose() const;
    void swapBuffers();
    void pollEvents();

    GLuint getTextureID() const { return texture_id; }
    GLFWwindow* getWindow() const { return window; }

    void setWindowTitle(const std::string& title);
    void renderColorHistogram(int red_val, int green_val, int blue_val);
    void renderBrightnessHistogram(const int* brightness_hist, int num_bins);

    static void framebufferSizeCallback(GLFWwindow* window, int width, int height);
};
