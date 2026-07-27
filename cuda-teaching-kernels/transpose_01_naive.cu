#include "cuda_common.cuh"

// Input reads are contiguous across a warp; output writes are separated by height elements.
__global__ void transpose_naive(const float* input, float* output, int width, int height) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < height && col < width) output[col * height + row] = input[row * width + col];
}

int main() {
    constexpr int width = 2048, height = 1536;
    const size_t count = static_cast<size_t>(width) * height;
    std::vector<float> input(count), expected(count), output(count);
    for (size_t i = 0; i < count; ++i) input[i] = static_cast<float>(i % 997) * 0.001f;
    for (int r = 0; r < height; ++r) for (int c = 0; c < width; ++c) expected[c * height + r] = input[r * width + c];
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, count * sizeof(float))); CUDA_CHECK(cudaMalloc(&d_output, count * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input, input.data(), count * sizeof(float), cudaMemcpyHostToDevice));
    dim3 block(32, 32), grid((width + 31) / 32, (height + 31) / 32);
    auto launch = [&] { transpose_naive<<<grid, block>>>(d_input, d_output, width, height); };
    launch(); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(output.data(), d_output, count * sizeof(float), cudaMemcpyDeviceToHost));
    require_equal(output, expected, "transpose_01_naive correctness");
    const float ms = benchmark_ms(launch);
    std::printf("transpose_01_naive: PASS, median %.4f ms\\n", ms);
    append_result("transpose_01_naive", ms);
    CUDA_CHECK(cudaFree(d_input)); CUDA_CHECK(cudaFree(d_output));
}
