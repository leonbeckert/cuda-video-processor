cmake_minimum_required(VERSION 3.18.0)

project(cuda-video-processor VERSION 1.0.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

# Compiler warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-unused-parameter")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda -Xcompiler=-Wall,-Wextra,-Wno-unused-parameter")
# Ampere (3090) + Ada Lovelace (4090) architectures
set(CMAKE_CUDA_ARCHITECTURES "86;89")
find_package(CUDAToolkit REQUIRED)

# OpenGL
find_package(OpenGL REQUIRED)

# FetchContent for GLFW and GLEW
include(FetchContent)

# GLFW
message(STATUS "Fetching GLFW...")
FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG        3.3.8
)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(glfw)

# GLEW
message(STATUS "Fetching GLEW...")
FetchContent_Declare(
    glew
    GIT_REPOSITORY https://github.com/Perlmint/glew-cmake.git
    GIT_TAG        glew-cmake-2.2.0
)
set(glew-cmake_BUILD_SHARED OFF CACHE BOOL "" FORCE)
set(glew-cmake_BUILD_STATIC ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(glew)

# FFmpeg via pkg-config (Linux)
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFMPEG REQUIRED libavformat libavcodec libavutil libswscale)

# Main executable
add_executable(cuda-video-processor
    src/main.cu
    src/kernels/sobel.cu
    src/kernels/enhance.cu
    src/kernels/histogram.cu
    src/kernels/brightness_histogram.cu
    src/opengl_renderer.cpp
    src/video_reader.cpp
)

target_include_directories(cuda-video-processor PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${FFMPEG_INCLUDE_DIRS}
)

target_link_libraries(cuda-video-processor
    CUDA::cudart
    OpenGL::GL
    glfw
    libglew_static
    ${FFMPEG_LIBRARIES}
)

set_target_properties(cuda-video-processor PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

message(STATUS "cuda-video-processor configured")
message(STATUS "  CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  FFmpeg: ${FFMPEG_LIBRARIES}")
