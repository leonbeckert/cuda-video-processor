// Benchmark harness for CUDA Video Processor kernels
// Follows NVIDIA best practices: 20 warmup, L2 flush, 100+ timed, report median + P99
//
// Usage: ./benchmark [--device N] [--width W] [--height H] [--pipeline-frames N]

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <algorithm>
#include <vector>
#include <chrono>
#include <numeric>

// Include kernel headers
#include "../src/kernels/sobel.cuh"
#include "../src/kernels/enhance.cuh"
#include "../src/kernels/histogram.cuh"
#include "../src/kernels/brightness_histogram.cuh"

#include "../src/cuda_check.h"

// ============================================================
// Utility: percentile from sorted array
// ============================================================
static float percentile(std::vector<float>& sorted_data, float p) {
    if (sorted_data.empty()) return 0.0f;
    float idx = p / 100.0f * (sorted_data.size() - 1);
    int lo = (int)idx;
    int hi = lo + 1;
    if (hi >= (int)sorted_data.size()) return sorted_data.back();
    float frac = idx - lo;
    return sorted_data[lo] * (1.0f - frac) + sorted_data[hi] * frac;
}

// ============================================================
// Naive (global memory) Sobel — baseline for comparison
// ============================================================
__device__ __forceinline__ uint8_t bench_clamp_u8(int v) {
    return static_cast<uint8_t>(v < 0 ? 0 : (v > 255 ? 255 : v));
}

__global__ void sobel_naive_rgba8_kernel(
    const uint8_t* __restrict__ in,
    uint8_t* __restrict__ out,
    int width, int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = (y * width + x) * 4;

    if (x == 0 || y == 0 || x == width - 1 || y == height - 1) {
        out[idx+0] = in[idx+0]; out[idx+1] = in[idx+1];
        out[idx+2] = in[idx+2]; out[idx+3] = in[idx+3];
        return;
    }

    #pragma unroll
    for (int c = 0; c < 3; c++) {
        int p00 = in[((y-1)*width+(x-1))*4+c], p01 = in[((y-1)*width+x)*4+c], p02 = in[((y-1)*width+(x+1))*4+c];
        int p10 = in[(y*width+(x-1))*4+c],                                      p12 = in[(y*width+(x+1))*4+c];
        int p20 = in[((y+1)*width+(x-1))*4+c], p21 = in[((y+1)*width+x)*4+c], p22 = in[((y+1)*width+(x+1))*4+c];

        int gx = -p00 + p02 - 2*p10 + 2*p12 - p20 + p22;
        int gy = -p00 - 2*p01 - p02 + p20 + 2*p21 + p22;
        float mag = sqrtf((float)(gx*gx + gy*gy));
        out[idx+c] = bench_clamp_u8((int)mag);
    }
    out[idx+3] = in[idx+3];
}

static void launch_sobel_naive(const uint8_t* d_in, uint8_t* d_out,
                                int width, int height, cudaStream_t stream = 0) {
    dim3 block(16, 16);
    dim3 grid((width+15)/16, (height+15)/16);
    sobel_naive_rgba8_kernel<<<grid, block, 0, stream>>>(d_in, d_out, width, height);
}

// ============================================================
// CPU reference implementations
// ============================================================
static void cpu_sobel(const uint8_t* in, uint8_t* out, int w, int h) {
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int idx = (y * w + x) * 4;
            if (x == 0 || y == 0 || x == w-1 || y == h-1) {
                memcpy(out + idx, in + idx, 4);
                continue;
            }
            for (int c = 0; c < 3; c++) {
                int p00 = in[((y-1)*w+(x-1))*4+c], p01 = in[((y-1)*w+x)*4+c], p02 = in[((y-1)*w+(x+1))*4+c];
                int p10 = in[(y*w+(x-1))*4+c],                                  p12 = in[(y*w+(x+1))*4+c];
                int p20 = in[((y+1)*w+(x-1))*4+c], p21 = in[((y+1)*w+x)*4+c], p22 = in[((y+1)*w+(x+1))*4+c];
                int gx = -p00 + p02 - 2*p10 + 2*p12 - p20 + p22;
                int gy = -p00 - 2*p01 - p02 + p20 + 2*p21 + p22;
                float mag = sqrtf((float)(gx*gx + gy*gy));
                int m = (int)mag;
                out[idx+c] = (uint8_t)(m < 0 ? 0 : (m > 255 ? 255 : m));
            }
            out[idx+3] = in[idx+3];
        }
    }
}

static void cpu_enhance(uint8_t* img, int w, int h, float contrast, float brightness) {
    for (int i = 0; i < w * h; i++) {
        for (int c = 0; c < 3; c++) {
            float p = img[i*4+c] * contrast + brightness;
            p = p < 0.f ? 0.f : (p > 255.f ? 255.f : p);
            img[i*4+c] = (uint8_t)p;
        }
    }
}

static void cpu_rgb_average(const uint8_t* img, int w, int h, int* r, int* g, int* b) {
    unsigned long long sr = 0, sg = 0, sb = 0;
    int n = w * h;
    for (int i = 0; i < n; i++) {
        sr += img[i*4+0]; sg += img[i*4+1]; sb += img[i*4+2];
    }
    *r = (int)(sr / n); *g = (int)(sg / n); *b = (int)(sb / n);
}

static void cpu_brightness_histogram(const uint8_t* img, int w, int h, int num_bins, int* hist) {
    memset(hist, 0, num_bins * sizeof(int));
    for (int i = 0; i < w * h; i++) {
        uint32_t y = 13933u * img[i*4+0] + 46871u * img[i*4+1] + 4732u * img[i*4+2] + 32768u;
        uint8_t gray = (uint8_t)(y >> 16);
        int bin = (gray * num_bins) >> 8;
        if (bin >= num_bins) bin = num_bins - 1;
        hist[bin]++;
    }
}

// ============================================================
// L2 cache flush
// ============================================================
static uint8_t* d_flush_buf = nullptr;
static size_t flush_size = 0;

static void init_flush_buffer(int device) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    flush_size = prop.l2CacheSize * 2;
    if (flush_size == 0) flush_size = 12 * 1024 * 1024; // fallback 12MB
    CUDA_CHECK(cudaMalloc(&d_flush_buf, flush_size));
}

static void flush_l2() {
    CUDA_CHECK(cudaMemset(d_flush_buf, 0, flush_size));
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================
// Benchmark runner
// ============================================================
struct BenchResult {
    float median_ms;
    float p99_ms;
    float stddev_ms;
    float bandwidth_gbs;
    float pct_peak;
    float cpu_time_ms;
    float speedup;
};

typedef void (*KernelFunc)(const uint8_t* d_in, uint8_t* d_out,
                           int width, int height, cudaStream_t stream);

static BenchResult run_kernel_benchmark(
    KernelFunc func,
    const uint8_t* d_in, uint8_t* d_out,
    int width, int height,
    size_t bytes_read, size_t bytes_written,
    float peak_bw_gbs,
    float cpu_time_ms,
    int warmup_iters = 20,
    int bench_iters = 100
) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warmup
    for (int i = 0; i < warmup_iters; i++) {
        func(d_in, d_out, width, height, 0);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed iterations with L2 flush
    std::vector<float> times(bench_iters);
    for (int i = 0; i < bench_iters; i++) {
        flush_l2();
        CUDA_CHECK(cudaEventRecord(start));
        func(d_in, d_out, width, height, 0);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&times[i], start, stop));
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    std::sort(times.begin(), times.end());

    BenchResult r;
    r.median_ms = percentile(times, 50.0f);
    r.p99_ms = percentile(times, 99.0f);

    // Stddev
    float mean = 0;
    for (auto t : times) mean += t;
    mean /= times.size();
    float var = 0;
    for (auto t : times) var += (t - mean) * (t - mean);
    r.stddev_ms = sqrtf(var / times.size());

    r.bandwidth_gbs = (float)(bytes_read + bytes_written) / (r.median_ms * 1e6f);
    r.pct_peak = (r.bandwidth_gbs / peak_bw_gbs) * 100.0f;
    r.cpu_time_ms = cpu_time_ms;
    r.speedup = cpu_time_ms / r.median_ms;

    return r;
}

// Wrapper for in-place kernels (enhance)
typedef void (*InplaceKernelFunc)(uint8_t* d_img, int width, int height,
                                  float contrast, float brightness, cudaStream_t stream);

static BenchResult run_inplace_benchmark(
    InplaceKernelFunc func,
    uint8_t* d_img, const uint8_t* d_src,
    int width, int height,
    size_t bytes_rw,
    float peak_bw_gbs,
    float cpu_time_ms,
    float contrast, float brightness,
    int warmup_iters = 20,
    int bench_iters = 100
) {
    size_t frame_bytes = (size_t)width * height * 4;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warmup
    for (int i = 0; i < warmup_iters; i++) {
        CUDA_CHECK(cudaMemcpy(d_img, d_src, frame_bytes, cudaMemcpyDeviceToDevice));
        func(d_img, width, height, contrast, brightness, 0);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> times(bench_iters);
    for (int i = 0; i < bench_iters; i++) {
        CUDA_CHECK(cudaMemcpy(d_img, d_src, frame_bytes, cudaMemcpyDeviceToDevice));
        flush_l2();
        CUDA_CHECK(cudaEventRecord(start));
        func(d_img, width, height, contrast, brightness, 0);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&times[i], start, stop));
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    std::sort(times.begin(), times.end());

    BenchResult r;
    r.median_ms = percentile(times, 50.0f);
    r.p99_ms = percentile(times, 99.0f);
    float mean = 0;
    for (auto t : times) mean += t;
    mean /= times.size();
    float var = 0;
    for (auto t : times) var += (t - mean) * (t - mean);
    r.stddev_ms = sqrtf(var / times.size());
    r.bandwidth_gbs = (float)bytes_rw / (r.median_ms * 1e6f);
    r.pct_peak = (r.bandwidth_gbs / peak_bw_gbs) * 100.0f;
    r.cpu_time_ms = cpu_time_ms;
    r.speedup = cpu_time_ms / r.median_ms;

    return r;
}

// ============================================================
// Hardware info
// ============================================================

// Peak bandwidth from manufacturer spec (not computed from clock rates,
// which may be unavailable in newer CUDA toolkit versions)
static float spec_peak_bandwidth(const char* gpu_name) {
    if (strstr(gpu_name, "4090")) return 1008.0f;   // RTX 4090: 1008 GB/s
    if (strstr(gpu_name, "3090")) return 936.0f;     // RTX 3090: 936 GB/s
    if (strstr(gpu_name, "4080")) return 716.8f;
    if (strstr(gpu_name, "3080")) return 760.0f;
    if (strstr(gpu_name, "A100")) return 2039.0f;
    if (strstr(gpu_name, "H100")) return 3350.0f;
    return 500.0f;  // conservative fallback
}

static void print_hardware_info(int device) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    int runtime_ver, driver_ver;
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime_ver));
    CUDA_CHECK(cudaDriverGetVersion(&driver_ver));

    float peak_bw = spec_peak_bandwidth(prop.name);

    // Query memory clock via attribute API (survives cudaDeviceProp changes)
    int mem_clock_khz = 0;
    cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, device);

    printf("## Hardware Info\n\n");
    printf("| Property | Value |\n");
    printf("|----------|-------|\n");
    printf("| GPU | %s |\n", prop.name);
    printf("| SM Count | %d |\n", prop.multiProcessorCount);
    printf("| Memory | %.0f MB |\n", prop.totalGlobalMem / 1048576.0f);
    printf("| Memory Bus | %d-bit |\n", prop.memoryBusWidth);
    if (mem_clock_khz > 0) {
        printf("| Memory Clock | %d MHz |\n", mem_clock_khz / 1000);
    }
    printf("| L2 Cache | %.1f MB |\n", prop.l2CacheSize / 1048576.0f);
    printf("| Peak Memory BW (spec) | %.0f GB/s |\n", peak_bw);
    printf("| CUDA Runtime | %d.%d |\n", runtime_ver / 1000, (runtime_ver % 1000) / 10);
    printf("| CUDA Driver | %d.%d |\n", driver_ver / 1000, (driver_ver % 1000) / 10);
    printf("| Compute Capability | %d.%d |\n", prop.major, prop.minor);
    printf("\n");
}

static float get_peak_bandwidth(int device) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    return spec_peak_bandwidth(prop.name);
}

// ============================================================
// Correctness validation
// ============================================================
static bool validate_sobel(const uint8_t* gpu_out, const uint8_t* cpu_out, int w, int h) {
    int max_err = 0;
    long long sum_err = 0;
    int n = w * h * 4;
    for (int i = 0; i < n; i++) {
        int diff = abs((int)gpu_out[i] - (int)cpu_out[i]);
        if (diff > max_err) max_err = diff;
        sum_err += diff;
    }
    float mean_err = (float)sum_err / n;
    // Allow tolerance of 1 for floating-point rounding between CPU/GPU sqrtf
    bool pass = max_err <= 1;
    printf("  Sobel correctness: max_err=%d, mean_err=%.3f — %s\n",
           max_err, mean_err, pass ? "PASS" : "FAIL");
    return pass;
}

static bool validate_enhance(const uint8_t* gpu_out, const uint8_t* cpu_out, int w, int h) {
    int max_err = 0;
    long long sum_err = 0;
    int n = w * h * 4;
    for (int i = 0; i < n; i++) {
        int diff = abs((int)gpu_out[i] - (int)cpu_out[i]);
        if (diff > max_err) max_err = diff;
        sum_err += diff;
    }
    float mean_err = (float)sum_err / n;
    bool pass = max_err <= 1;
    printf("  Enhancement correctness: max_err=%d, mean_err=%.3f — %s\n",
           max_err, mean_err, pass ? "PASS" : "FAIL");
    return pass;
}

// ============================================================
// Main
// ============================================================
int main(int argc, char** argv) {
    int device = 0;
    int width = 3840;
    int height = 2160;
    int pipeline_frames = 1000;

    // Parse args
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--device") == 0 && i+1 < argc) device = atoi(argv[++i]);
        else if (strcmp(argv[i], "--width") == 0 && i+1 < argc) width = atoi(argv[++i]);
        else if (strcmp(argv[i], "--height") == 0 && i+1 < argc) height = atoi(argv[++i]);
        else if (strcmp(argv[i], "--pipeline-frames") == 0 && i+1 < argc) pipeline_frames = atoi(argv[++i]);
    }

    CUDA_CHECK(cudaSetDevice(device));

    size_t frame_bytes = (size_t)width * height * 4;
    int pixel_count = width * height;

    printf("# CUDA Video Processor — Benchmark Results\n\n");
    print_hardware_info(device);
    printf("**Resolution:** %d x %d (%d pixels, %.2f MB/frame)\n\n",
           width, height, pixel_count, frame_bytes / 1048576.0f);

    float peak_bw = get_peak_bandwidth(device);

    // ---- Generate synthetic 4K frame ----
    printf("Generating synthetic %dx%d RGBA frame...\n", width, height);
    uint8_t* h_frame = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_frame, frame_bytes));
    srand(42);
    for (size_t i = 0; i < frame_bytes; i++) {
        h_frame[i] = (uint8_t)(rand() % 256);
    }

    // ---- Allocate device buffers ----
    uint8_t *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, frame_bytes));
    CUDA_CHECK(cudaMalloc(&d_out, frame_bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_frame, frame_bytes, cudaMemcpyHostToDevice));

    init_flush_buffer(device);

    // ============================================================
    // CPU baselines
    // ============================================================
    printf("Running CPU baselines...\n");

    uint8_t* cpu_sobel_out = (uint8_t*)malloc(frame_bytes);
    uint8_t* cpu_enhance_buf = (uint8_t*)malloc(frame_bytes);

    // CPU Sobel
    auto t0 = std::chrono::high_resolution_clock::now();
    cpu_sobel(h_frame, cpu_sobel_out, width, height);
    auto t1 = std::chrono::high_resolution_clock::now();
    float cpu_sobel_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

    // CPU Enhancement
    memcpy(cpu_enhance_buf, h_frame, frame_bytes);
    t0 = std::chrono::high_resolution_clock::now();
    cpu_enhance(cpu_enhance_buf, width, height, 1.2f, 20.0f);
    t1 = std::chrono::high_resolution_clock::now();
    float cpu_enhance_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

    // CPU RGB average
    int cpu_r, cpu_g, cpu_b;
    t0 = std::chrono::high_resolution_clock::now();
    cpu_rgb_average(h_frame, width, height, &cpu_r, &cpu_g, &cpu_b);
    t1 = std::chrono::high_resolution_clock::now();
    float cpu_rgb_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

    // CPU brightness histogram
    const int NUM_BINS = 25;
    int cpu_hist[25];
    t0 = std::chrono::high_resolution_clock::now();
    cpu_brightness_histogram(h_frame, width, height, NUM_BINS, cpu_hist);
    t1 = std::chrono::high_resolution_clock::now();
    float cpu_bright_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

    printf("CPU baselines: Sobel=%.1fms, Enhance=%.1fms, RGB=%.1fms, Brightness=%.1fms\n\n",
           cpu_sobel_ms, cpu_enhance_ms, cpu_rgb_ms, cpu_bright_ms);

    // ============================================================
    // Correctness validation
    // ============================================================
    printf("## Correctness Validation\n\n");

    // GPU Sobel (tiled)
    launch_sobel(d_in, d_out, width, height);
    CUDA_CHECK(cudaDeviceSynchronize());
    uint8_t* gpu_sobel_out = (uint8_t*)malloc(frame_bytes);
    CUDA_CHECK(cudaMemcpy(gpu_sobel_out, d_out, frame_bytes, cudaMemcpyDeviceToHost));
    bool sobel_ok = validate_sobel(gpu_sobel_out, cpu_sobel_out, width, height);

    // GPU Enhancement
    uint8_t* d_enhance_buf;
    CUDA_CHECK(cudaMalloc(&d_enhance_buf, frame_bytes));
    CUDA_CHECK(cudaMemcpy(d_enhance_buf, d_in, frame_bytes, cudaMemcpyDeviceToDevice));
    launch_enhancement(d_enhance_buf, width, height, 1.2f, 20.0f);
    CUDA_CHECK(cudaDeviceSynchronize());
    uint8_t* gpu_enhance_out = (uint8_t*)malloc(frame_bytes);
    CUDA_CHECK(cudaMemcpy(gpu_enhance_out, d_enhance_buf, frame_bytes, cudaMemcpyDeviceToHost));
    bool enhance_ok = validate_enhance(gpu_enhance_out, cpu_enhance_buf, width, height);
    cudaFree(d_enhance_buf);

    // GPU RGB average
    int gpu_r, gpu_g, gpu_b;
    launch_rgb_average(d_in, width, height, &gpu_r, &gpu_g, &gpu_b);
    bool rgb_ok = (abs(gpu_r - cpu_r) <= 1 && abs(gpu_g - cpu_g) <= 1 && abs(gpu_b - cpu_b) <= 1);
    printf("  RGB average: GPU=(%d,%d,%d) CPU=(%d,%d,%d) — %s\n",
           gpu_r, gpu_g, gpu_b, cpu_r, cpu_g, cpu_b, rgb_ok ? "PASS" : "FAIL");

    // GPU brightness histogram
    int gpu_hist[25];
    launch_brightness_histogram(d_in, width, height, NUM_BINS, gpu_hist);
    int hist_max_err = 0;
    for (int i = 0; i < NUM_BINS; i++) {
        int diff = abs(gpu_hist[i] - cpu_hist[i]);
        if (diff > hist_max_err) hist_max_err = diff;
    }
    bool hist_ok = hist_max_err == 0;
    printf("  Brightness histogram: max_bin_err=%d — %s\n", hist_max_err, hist_ok ? "PASS" : "FAIL");

    if (!sobel_ok || !enhance_ok || !rgb_ok || !hist_ok) {
        printf("\n**CORRECTNESS CHECK FAILED — fix kernels before reporting performance**\n");
        // Continue anyway to show numbers, but flag prominently
    }
    printf("\n");

    free(gpu_sobel_out);
    free(gpu_enhance_out);

    // ============================================================
    // Per-kernel benchmarks
    // ============================================================
    printf("Running per-kernel benchmarks (20 warmup, 100 timed, L2 flush between)...\n\n");

    // Effective bandwidth calculation uses minimum data touched (not redundant reads):
    //   Sobel: 1 frame read + 1 frame write = 2 * frame_bytes
    //   Enhancement: read + write in-place = 2 * frame_bytes
    //   Histogram: 1 frame read, negligible write = frame_bytes
    size_t kernel_bytes_rw = frame_bytes + frame_bytes;  // Sobel and Enhancement
    size_t hist_bytes_r = frame_bytes;

    // --- Sobel naive ---
    BenchResult r_sobel_naive = run_kernel_benchmark(
        launch_sobel_naive,
        d_in, d_out, width, height,
        frame_bytes, frame_bytes, peak_bw, cpu_sobel_ms);

    // --- Sobel tiled ---
    BenchResult r_sobel_tiled = run_kernel_benchmark(
        launch_sobel,
        d_in, d_out, width, height,
        frame_bytes, frame_bytes, peak_bw, cpu_sobel_ms);

    // --- Enhancement ---
    // Need to wrap launch_enhancement to match the KernelFunc signature
    CUDA_CHECK(cudaMemcpy(d_out, d_in, frame_bytes, cudaMemcpyDeviceToDevice));
    BenchResult r_enhance = run_inplace_benchmark(
        launch_enhancement,
        d_out, d_in, width, height,
        kernel_bytes_rw, peak_bw, cpu_enhance_ms,
        1.2f, 20.0f);

    // --- RGB Histogram (uses default stream, allocates internally) ---
    {
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        for (int i = 0; i < 20; i++) {
            int rr, gg, bb;
            launch_rgb_average(d_in, width, height, &rr, &gg, &bb);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<float> times(100);
        for (int i = 0; i < 100; i++) {
            flush_l2();
            CUDA_CHECK(cudaEventRecord(start));
            int rr, gg, bb;
            launch_rgb_average(d_in, width, height, &rr, &gg, &bb);
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            CUDA_CHECK(cudaEventElapsedTime(&times[i], start, stop));
        }

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        std::sort(times.begin(), times.end());

        BenchResult r;
        r.median_ms = percentile(times, 50.0f);
        r.p99_ms = percentile(times, 99.0f);
        float mean = 0; for (auto t : times) mean += t; mean /= times.size();
        float var = 0; for (auto t : times) var += (t-mean)*(t-mean);
        r.stddev_ms = sqrtf(var / times.size());
        r.bandwidth_gbs = (float)hist_bytes_r / (r.median_ms * 1e6f);
        r.pct_peak = (r.bandwidth_gbs / peak_bw) * 100.0f;
        r.cpu_time_ms = cpu_rgb_ms;
        r.speedup = cpu_rgb_ms / r.median_ms;

        // Print after all benchmarks
        // Store for later
        BenchResult r_rgb = r;

        // --- Brightness Histogram ---
        for (int i = 0; i < 20; i++) {
            int h[25];
            launch_brightness_histogram(d_in, width, height, NUM_BINS, h);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<float> times2(100);
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        for (int i = 0; i < 100; i++) {
            flush_l2();
            CUDA_CHECK(cudaEventRecord(start));
            int h[25];
            launch_brightness_histogram(d_in, width, height, NUM_BINS, h);
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            CUDA_CHECK(cudaEventElapsedTime(&times2[i], start, stop));
        }
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        std::sort(times2.begin(), times2.end());

        BenchResult r_bright;
        r_bright.median_ms = percentile(times2, 50.0f);
        r_bright.p99_ms = percentile(times2, 99.0f);
        mean = 0; for (auto t : times2) mean += t; mean /= times2.size();
        var = 0; for (auto t : times2) var += (t-mean)*(t-mean);
        r_bright.stddev_ms = sqrtf(var / times2.size());
        r_bright.bandwidth_gbs = (float)hist_bytes_r / (r_bright.median_ms * 1e6f);
        r_bright.pct_peak = (r_bright.bandwidth_gbs / peak_bw) * 100.0f;
        r_bright.cpu_time_ms = cpu_bright_ms;
        r_bright.speedup = cpu_bright_ms / r_bright.median_ms;

        // ============================================================
        // Print per-kernel results table
        // ============================================================
        printf("## Per-Kernel Results\n\n");
        printf("| Kernel | Median (ms) | P99 (ms) | BW (GB/s) | %% Peak BW | CPU (ms) | Speedup |\n");
        printf("|--------|-------------|----------|-----------|-----------|----------|---------|\n");
        printf("| Sobel (naive/global) | %.3f | %.3f | %.1f | %.1f%% | %.1f | %.0fx |\n",
               r_sobel_naive.median_ms, r_sobel_naive.p99_ms,
               r_sobel_naive.bandwidth_gbs, r_sobel_naive.pct_peak,
               r_sobel_naive.cpu_time_ms, r_sobel_naive.speedup);
        printf("| Sobel (shared mem tiled) | %.3f | %.3f | %.1f | %.1f%% | %.1f | %.0fx |\n",
               r_sobel_tiled.median_ms, r_sobel_tiled.p99_ms,
               r_sobel_tiled.bandwidth_gbs, r_sobel_tiled.pct_peak,
               r_sobel_tiled.cpu_time_ms, r_sobel_tiled.speedup);
        printf("| Enhancement | %.3f | %.3f | %.1f | %.1f%% | %.1f | %.0fx |\n",
               r_enhance.median_ms, r_enhance.p99_ms,
               r_enhance.bandwidth_gbs, r_enhance.pct_peak,
               r_enhance.cpu_time_ms, r_enhance.speedup);
        printf("| RGB Histogram | %.3f | %.3f | %.1f | %.1f%% | %.1f | %.0fx |\n",
               r_rgb.median_ms, r_rgb.p99_ms,
               r_rgb.bandwidth_gbs, r_rgb.pct_peak,
               r_rgb.cpu_time_ms, r_rgb.speedup);
        printf("| Brightness Histogram | %.3f | %.3f | %.1f | %.1f%% | %.1f | %.0fx |\n",
               r_bright.median_ms, r_bright.p99_ms,
               r_bright.bandwidth_gbs, r_bright.pct_peak,
               r_bright.cpu_time_ms, r_bright.speedup);
        printf("\n");

        // Tiling improvement
        float tiling_speedup = r_sobel_naive.median_ms / r_sobel_tiled.median_ms;
        printf("**Shared memory tiling speedup:** %.2fx over naive Sobel\n\n", tiling_speedup);
    }

    // ============================================================
    // Full pipeline benchmark (headless simulation)
    // ============================================================
    printf("## Full Pipeline Benchmark (%d frames, all filters ON, headless)\n\n", pipeline_frames);
    {
        // Pre-allocate everything to avoid timing allocation
        uint8_t* d_pipeline_in;
        uint8_t* d_pipeline_out;
        uint8_t* h_out;
        CUDA_CHECK(cudaMalloc(&d_pipeline_in, frame_bytes));
        CUDA_CHECK(cudaMalloc(&d_pipeline_out, frame_bytes));
        CUDA_CHECK(cudaMallocHost(&h_out, frame_bytes));

        cudaStream_t compute_stream;
        CUDA_CHECK(cudaStreamCreate(&compute_stream));

        // Warmup
        for (int i = 0; i < 20; i++) {
            CUDA_CHECK(cudaMemcpyAsync(d_pipeline_in, h_frame, frame_bytes,
                                       cudaMemcpyHostToDevice, compute_stream));
            launch_sobel(d_pipeline_in, d_pipeline_out, width, height, compute_stream);
            std::swap(d_pipeline_in, d_pipeline_out);
            launch_enhancement(d_pipeline_in, width, height, 1.2f, 20.0f, compute_stream);
            CUDA_CHECK(cudaMemcpyAsync(h_out, d_pipeline_in, frame_bytes,
                                       cudaMemcpyDeviceToHost, compute_stream));
            CUDA_CHECK(cudaStreamSynchronize(compute_stream));
        }

        // Timed run
        auto start_time = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < pipeline_frames; i++) {
            CUDA_CHECK(cudaMemcpyAsync(d_pipeline_in, h_frame, frame_bytes,
                                       cudaMemcpyHostToDevice, compute_stream));
            CUDA_CHECK(cudaStreamSynchronize(compute_stream));

            launch_sobel(d_pipeline_in, d_pipeline_out, width, height, compute_stream);
            std::swap(d_pipeline_in, d_pipeline_out);
            launch_enhancement(d_pipeline_in, width, height, 1.2f, 20.0f, compute_stream);

            int rr, gg, bb;
            CUDA_CHECK(cudaStreamSynchronize(compute_stream));
            launch_rgb_average(d_pipeline_in, width, height, &rr, &gg, &bb);

            int hist[25];
            launch_brightness_histogram(d_pipeline_in, width, height, 25, hist);

            CUDA_CHECK(cudaMemcpyAsync(h_out, d_pipeline_in, frame_bytes,
                                       cudaMemcpyDeviceToHost, compute_stream));
            CUDA_CHECK(cudaStreamSynchronize(compute_stream));
        }
        auto end_time = std::chrono::high_resolution_clock::now();

        float total_ms = std::chrono::duration<float, std::milli>(end_time - start_time).count();
        float gpu_fps = pipeline_frames / (total_ms / 1000.0f);
        float frame_latency = total_ms / pipeline_frames;

        // CPU pipeline baseline (fewer frames)
        int cpu_pipeline_frames = 10;
        auto cpu_start = std::chrono::high_resolution_clock::now();
        uint8_t* cpu_pipeline_out = (uint8_t*)malloc(frame_bytes);
        for (int i = 0; i < cpu_pipeline_frames; i++) {
            cpu_sobel(h_frame, cpu_pipeline_out, width, height);
            cpu_enhance(cpu_pipeline_out, width, height, 1.2f, 20.0f);
            int r, g, b;
            cpu_rgb_average(cpu_pipeline_out, width, height, &r, &g, &b);
            int h[25];
            cpu_brightness_histogram(cpu_pipeline_out, width, height, 25, h);
        }
        auto cpu_end = std::chrono::high_resolution_clock::now();
        float cpu_total_ms = std::chrono::duration<float, std::milli>(cpu_end - cpu_start).count();
        float cpu_fps = cpu_pipeline_frames / (cpu_total_ms / 1000.0f);

        printf("| Metric | Value |\n");
        printf("|--------|-------|\n");
        printf("| GPU sustained FPS | %.1f |\n", gpu_fps);
        printf("| GPU frame latency | %.2f ms |\n", frame_latency);
        printf("| CPU baseline FPS | %.2f |\n", cpu_fps);
        printf("| Pipeline speedup | %.0fx |\n", gpu_fps / cpu_fps);
        printf("| Frames timed | %d |\n", pipeline_frames);
        printf("\n");

        free(cpu_pipeline_out);
        CUDA_CHECK(cudaStreamDestroy(compute_stream));
        CUDA_CHECK(cudaFreeHost(h_out));
        cudaFree(d_pipeline_in);
        cudaFree(d_pipeline_out);
    }

    // Cleanup
    CUDA_CHECK(cudaFreeHost(h_frame));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_flush_buf));
    free(cpu_sobel_out);
    free(cpu_enhance_buf);

    printf("Benchmark complete.\n");
    return 0;
}
