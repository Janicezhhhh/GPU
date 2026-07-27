#include "cuda_common.cuh"

// Baseline: every thread contends for the same global-memory accumulator.
__global__ void reduce_atomic(const float* input, float* result, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(result, input[i]);
}

int main() {
    constexpr int n = (1 << 22) + 37;
    constexpr int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    std::vector<float> h_input(n, 1.0f);
    float cpu = 0.0f;
    for (float x : h_input) cpu += x;

    float *d_input, *d_result;
    CUDA_CHECK(cudaMalloc(&d_input, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    auto prepare = [&] { CUDA_CHECK(cudaMemset(d_result, 0, sizeof(float))); };
    auto launch = [&] { reduce_atomic<<<blocks, threads>>>(d_input, d_result, n); };

    prepare(); launch(); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    float gpu; CUDA_CHECK(cudaMemcpy(&gpu, d_result, sizeof(float), cudaMemcpyDeviceToHost));
    require_close(gpu, cpu, "reduction_01_naive_atomic correctness");
    const float ms = benchmark_ms(prepare, launch);
    std::printf("reduction_01_naive_atomic: PASS, median %.4f ms\\n", ms);
    append_result("reduction_01_naive_atomic", ms);
    CUDA_CHECK(cudaFree(d_input)); CUDA_CHECK(cudaFree(d_result));
}
