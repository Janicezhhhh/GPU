#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#define CUDA_CHECK(call) do { \
    cudaError_t status__ = (call); \
    if (status__ != cudaSuccess) { \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\\n", __FILE__, __LINE__, cudaGetErrorString(status__)); \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

constexpr int kWarmupRuns = 5;
constexpr int kMeasuredRuns = 20;
constexpr float kTolerance = 1e-4f;

inline float median_ms(std::vector<float> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

template <typename Prepare, typename Launch>
float benchmark_ms(Prepare prepare, Launch launch) {
    for (int i = 0; i < kWarmupRuns; ++i) { prepare(); launch(); }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    std::vector<float> samples;
    samples.reserve(kMeasuredRuns);
    for (int i = 0; i < kMeasuredRuns; ++i) {
        prepare();
        CUDA_CHECK(cudaEventRecord(start));
        launch();
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float elapsed = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
        samples.push_back(elapsed);
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return median_ms(samples);
}

template <typename Launch>
float benchmark_ms(Launch launch) {
    return benchmark_ms([] {}, launch);
}

inline void append_result(const char* name, float ms) {
    std::ofstream out("results.md", std::ios::app);
    if (out.tellp() == 0) out << "# Local benchmark results\\n\\n| Program | Median kernel time | Method |\\n| --- | ---: | --- |\\n";
    out << "| " << name << " | " << ms << " ms | 5 warm-up + 20 event samples, median |\\n";
}

inline void require_close(float got, float expected, const char* label) {
    if (std::fabs(got - expected) > kTolerance) {
        std::fprintf(stderr, "%s failed: GPU=%0.8f CPU=%0.8f\\n", label, got, expected);
        std::exit(EXIT_FAILURE);
    }
}

inline void require_equal(const std::vector<float>& got, const std::vector<float>& expected, const char* label) {
    for (size_t i = 0; i < got.size(); ++i) {
        if (std::fabs(got[i] - expected[i]) > kTolerance) {
            std::fprintf(stderr, "%s failed at %zu: GPU=%f CPU=%f\\n", label, i, got[i], expected[i]);
            std::exit(EXIT_FAILURE);
        }
    }
}
