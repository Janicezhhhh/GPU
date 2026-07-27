#include "cuda_common.cuh"

// One input per thread; all block-local collaboration occurs in shared memory.
__global__ void reduce_shared_tree(const float* input, float* block_sums, int n) {
    extern __shared__ float shared[];
    const int tid = threadIdx.x;
    const int i = blockIdx.x * blockDim.x + tid;
    shared[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) shared[tid] += shared[tid + stride];
        __syncthreads();
    }
    if (tid == 0) block_sums[blockIdx.x] = shared[0];
}

int main() {
    constexpr int n = (1 << 22) + 37, threads = 256;
    const int blocks = (n + threads - 1) / threads;
    std::vector<float> h_input(n, 1.0f), h_blocks(blocks);
    float cpu = 0.0f; for (float x : h_input) cpu += x;
    float *d_input, *d_blocks;
    CUDA_CHECK(cudaMalloc(&d_input, n * sizeof(float))); CUDA_CHECK(cudaMalloc(&d_blocks, blocks * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    auto launch = [&] { reduce_shared_tree<<<blocks, threads, threads * sizeof(float)>>>(d_input, d_blocks, n); };
    launch(); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_blocks.data(), d_blocks, blocks * sizeof(float), cudaMemcpyDeviceToHost));
    float gpu = 0.0f; for (float x : h_blocks) gpu += x;
    require_close(gpu, cpu, "reduction_02_shared_tree correctness");
    const float ms = benchmark_ms(launch);
    std::printf("reduction_02_shared_tree: PASS, median %.4f ms\\n", ms);
    append_result("reduction_02_shared_tree", ms);
    CUDA_CHECK(cudaFree(d_input)); CUDA_CHECK(cudaFree(d_blocks));
}
