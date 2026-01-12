#include "opengl_renderer.h"
#include <iostream>
#include <cstdlib>

OpenGLRenderer::OpenGLRenderer()
    : window(nullptr), texture_id(0), window_width(0), window_height(0),
      texture_width(0), texture_height(0) {
}

OpenGLRenderer::~OpenGLRenderer() {
    cleanup();
}

bool OpenGLRenderer::initialize(int window_w, int window_h, int texture_w, int texture_h) {
    window_width = window_w;
    window_height = window_h;
    texture_width = texture_w;
    texture_height = texture_h;

    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return false;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);

    window = glfwCreateWindow(window_width, window_height, "CUDA-OpenGL Video Player", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return false;
    }

    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);

    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW" << std::endl;
        return false;
    }

    glViewport(0, 0, window_width, window_height);
    glEnable(GL_TEXTURE_2D);

    if (!createTexture()) {
        std::cerr << "Failed to create texture" << std::endl;
        return false;
    }

    std::cout << "OpenGL renderer initialized: " << window_width << "x" << window_height
              << " window, " << texture_width << "x" << texture_height << " texture" << std::endl;

    return true;
}

bool OpenGLRenderer::createTexture() {
    GL_CHECK(glGenTextures(1, &texture_id));
    GL_CHECK(glBindTexture(GL_TEXTURE_2D, texture_id));

    GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
    GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));
    GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
    GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));

    GL_CHECK(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, texture_width, texture_height,
                         0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr));

    GL_CHECK(glBindTexture(GL_TEXTURE_2D, 0));

    std::cout << "Created OpenGL texture " << texture_id << " ("
              << texture_width << "x" << texture_height << ")" << std::endl;

    return true;
}

void OpenGLRenderer::render() {
    GL_CHECK(glClear(GL_COLOR_BUFFER_BIT));

    GL_CHECK(glEnable(GL_TEXTURE_2D));
    GL_CHECK(glBindTexture(GL_TEXTURE_2D, texture_id));

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    static const GLfloat vertices[] = {
        -1.0f, -1.0f,  0.0f, 1.0f,
         1.0f, -1.0f,  1.0f, 1.0f,
         1.0f,  1.0f,  1.0f, 0.0f,
        -1.0f,  1.0f,  0.0f, 0.0f
    };

    static const GLubyte indices[] = {
        0, 1, 2, 3
    };

    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_TEXTURE_COORD_ARRAY);

    glVertexPointer(2, GL_FLOAT, 4 * sizeof(GLfloat), &vertices[0]);
    glTexCoordPointer(2, GL_FLOAT, 4 * sizeof(GLfloat), &vertices[2]);

    glDrawElements(GL_QUADS, 4, GL_UNSIGNED_BYTE, indices);

    glDisableClientState(GL_TEXTURE_COORD_ARRAY);
    glDisableClientState(GL_VERTEX_ARRAY);

    glBindTexture(GL_TEXTURE_2D, 0);
}

void OpenGLRenderer::renderColorHistogram(int red_val, int green_val, int blue_val) {
    glDisable(GL_TEXTURE_2D);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, window_width, 0, window_height, -1, 1);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    const int hist_size = 300;
    const int bar_width = 70;
    const int bar_spacing = (hist_size - 3 * bar_width) / 4;

    int red_bar_height = (red_val * hist_size) / 255;
    int green_bar_height = (green_val * hist_size) / 255;
    int blue_bar_height = (blue_val * hist_size) / 255;

    int red_x = bar_spacing;
    int green_x = bar_spacing + bar_width + bar_spacing;
    int blue_x = bar_spacing + bar_width + bar_spacing + bar_width + bar_spacing;

    GLfloat vertices[] = {
        0.0f, 0.0f,
        (float)hist_size, 0.0f,
        (float)hist_size, (float)hist_size,
        0.0f, (float)hist_size,

        (float)red_x, 0.0f,
        (float)(red_x + bar_width), 0.0f,
        (float)(red_x + bar_width), (float)red_bar_height,
        (float)red_x, (float)red_bar_height,

        (float)green_x, 0.0f,
        (float)(green_x + bar_width), 0.0f,
        (float)(green_x + bar_width), (float)green_bar_height,
        (float)green_x, (float)green_bar_height,

        (float)blue_x, 0.0f,
        (float)(blue_x + bar_width), 0.0f,
        (float)(blue_x + bar_width), (float)blue_bar_height,
        (float)blue_x, (float)blue_bar_height
    };

    static const GLfloat colors[] = {
        1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f,

        1.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.0f,

        0.0f, 1.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 1.0f, 0.0f,

        0.0f, 0.0f, 1.0f,
        0.0f, 0.0f, 1.0f,
        0.0f, 0.0f, 1.0f,
        0.0f, 0.0f, 1.0f
    };

    static const GLubyte indices[] = {
        0, 1, 2, 3,
        4, 5, 6, 7,
        8, 9, 10, 11,
        12, 13, 14, 15
    };

    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);

    glVertexPointer(2, GL_FLOAT, 0, vertices);
    glColorPointer(3, GL_FLOAT, 0, colors);

    glDrawElements(GL_QUADS, 16, GL_UNSIGNED_BYTE, indices);

    glDisableClientState(GL_COLOR_ARRAY);
    glDisableClientState(GL_VERTEX_ARRAY);

    glColor3f(1.0f, 1.0f, 1.0f);
}

void OpenGLRenderer::renderBrightnessHistogram(const int* brightness_hist, int num_bins) {
    glDisable(GL_TEXTURE_2D);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, window_width, 0, window_height, -1, 1);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    const int brightness_hist_width = 400;
    const int brightness_hist_height = 200;
    const int brightness_hist_x = 320;
    const int brightness_hist_y = 0;
    const float bin_width = static_cast<float>(brightness_hist_width) / num_bins;

    int max_count = 1;
    for (int i = 0; i < num_bins; i++) {
        if (brightness_hist[i] > max_count) {
            max_count = brightness_hist[i];
        }
    }

    glColor3f(1.0f, 1.0f, 1.0f);
    glBegin(GL_QUADS);
        glVertex2f(brightness_hist_x, brightness_hist_y);
        glVertex2f(brightness_hist_x + brightness_hist_width, brightness_hist_y);
        glVertex2f(brightness_hist_x + brightness_hist_width, brightness_hist_y + brightness_hist_height);
        glVertex2f(brightness_hist_x, brightness_hist_y + brightness_hist_height);
    glEnd();

    glColor3f(0.3f, 0.3f, 0.3f);
    for (int i = 0; i < num_bins; i++) {
        float x = brightness_hist_x + i * bin_width;
        float bar_height = (static_cast<float>(brightness_hist[i]) / max_count) * brightness_hist_height;

        glBegin(GL_QUADS);
            glVertex2f(x, brightness_hist_y);
            glVertex2f(x + bin_width - 1, brightness_hist_y);
            glVertex2f(x + bin_width - 1, brightness_hist_y + bar_height);
            glVertex2f(x, brightness_hist_y + bar_height);
        glEnd();
    }

    glColor3f(1.0f, 1.0f, 1.0f);
}

void OpenGLRenderer::cleanup() {
    if (texture_id) {
        glDeleteTextures(1, &texture_id);
        texture_id = 0;
    }

    if (window) {
        glfwDestroyWindow(window);
        window = nullptr;
    }

    glfwTerminate();
}

bool OpenGLRenderer::shouldClose() const {
    return window ? glfwWindowShouldClose(window) : true;
}

void OpenGLRenderer::swapBuffers() {
    if (window) {
        glfwSwapBuffers(window);
    }
}

void OpenGLRenderer::pollEvents() {
    glfwPollEvents();
}

void OpenGLRenderer::setWindowTitle(const std::string& title) {
    if (window) {
        glfwSetWindowTitle(window, title.c_str());
    }
}

void OpenGLRenderer::framebufferSizeCallback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}
