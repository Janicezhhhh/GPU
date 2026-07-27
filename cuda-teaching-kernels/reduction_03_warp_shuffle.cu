#include "cuda_common.cuh"

__device__ float warp_reduce(float value) {
    for (int offset = 16; offset > 0; offset >>= 1)
        value += __shfl_down_sync(0xffffffff, value, offset);
    return value;
}

// Shared memory handles inter-warp work; the last warp finishes without __syncthreads().
__global__ void reduce_warp_shuffle(const float* input, float* block_sums, int n) {
    __shared__ float shared[256];
    const int tid = threadIdx.x, i = blockIdx.x * blockDim.x + tid;
    shared[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (tid < stride) shared[tid] += shared[tid + stride];
        __syncthreads();
    }
    if (tid < 32) {
        float value = shared[tid] + shared[tid + 32];
        value = warp_reduce(value);
        if (tid == 0) block_sums[blockIdx.x] = value;
    }
}

int main() {
    constexpr int n = (1 << 22) + 37, threads = 256;
    const int blocks = (n + threads - 1) / threads;
    std::vector<float> h_input(n, 1.0f), h_blocks(blocks);
    float cpu = 0.0f; for (float x : h_input) cpu += x;
    float *d_input, *d_blocks;
    CUDA_CHECK(cudaMalloc(&d_input, n * sizeof(float))); CUDA_CHECK(cudaMalloc(&d_blocks, blocks * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    auto launch = [&] { reduce_warp_shuffle<<<blocks, threads>>>(d_input, d_blocks, n); };
    launch(); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_blocks.data(), d_blocks, blocks * sizeof(float), cudaMemcpyDeviceToHost));
    float gpu = 0.0f; for (float x : h_blocks) gpu += x;
    require_close(gpu, cpu, "reduction_03_warp_shuffle correctness");
    const float ms = benchmark_ms(launch);
    std::printf("reduction_03_warp_shuffle: PASS, median %.4f ms\\n", ms);
    append_result("reduction_03_warp_shuffle", ms);
    CUDA_CHECK(cudaFree(d_input)); CUDA_CHECK(cudaFree(d_blocks));
}
