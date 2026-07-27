#include "cuda_common.cuh"

// Coalesces global reads and writes. The transposed shared-memory read has a 32-way bank conflict.
__global__ void transpose_shared_tile(const float* input, float* output, int width, int height) {
    __shared__ float tile[32][32];
    const int x = blockIdx.x * 32 + threadIdx.x, y = blockIdx.y * 32 + threadIdx.y;
    if (x < width && y < height) tile[threadIdx.y][threadIdx.x] = input[y * width + x];
    __syncthreads();
    const int out_x = blockIdx.y * 32 + threadIdx.x;
    const int out_y = blockIdx.x * 32 + threadIdx.y;
    if (out_x < height && out_y < width) output[out_y * height + out_x] = tile[threadIdx.x][threadIdx.y];
}

int main() {
    constexpr int width = 2048, height = 1536; const size_t count = static_cast<size_t>(width) * height;
    std::vector<float> input(count), expected(count), output(count);
    for (size_t i = 0; i < count; ++i) input[i] = static_cast<float>(i % 997) * 0.001f;
    for (int r = 0; r < height; ++r) for (int c = 0; c < width; ++c) expected[c * height + r] = input[r * width + c];
    float *d_input, *d_output; CUDA_CHECK(cudaMalloc(&d_input, count * sizeof(float))); CUDA_CHECK(cudaMalloc(&d_output, count * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input, input.data(), count * sizeof(float), cudaMemcpyHostToDevice));
    dim3 block(32, 32), grid((width + 31) / 32, (height + 31) / 32);
    auto launch = [&] { transpose_shared_tile<<<grid, block>>>(d_input, d_output, width, height); };
    launch(); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize()); CUDA_CHECK(cudaMemcpy(output.data(), d_output, count * sizeof(float), cudaMemcpyDeviceToHost));
    require_equal(output, expected, "transpose_02_shared_tile correctness");
    const float ms = benchmark_ms(launch); std::printf("transpose_02_shared_tile: PASS, median %.4f ms\\n", ms); append_result("transpose_02_shared_tile", ms);
    CUDA_CHECK(cudaFree(d_input)); CUDA_CHECK(cudaFree(d_output));
}
