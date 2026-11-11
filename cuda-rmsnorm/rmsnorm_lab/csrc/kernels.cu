#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <optional>

namespace {
constexpr int kThreads = 256;

template <typename T, bool Fused>
__device__ __forceinline__ float value(const T* x, const T* residual, int64_t i) {
  if constexpr (Fused) {
    // Match torch.add: round the residual sum before FP32 accumulation.
    return static_cast<float>(static_cast<T>(static_cast<float>(x[i]) +
                                              static_cast<float>(residual[i])));
  }
  return static_cast<float>(x[i]);
}

// Teaching baseline: one thread per row, serial sum and output loops.
template <typename T, bool Fused>
__global__ void serial_kernel(const T* x, const T* weight, const T* residual,
                              T* out, int64_t rows, int64_t hidden, float eps) {
  const int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= rows) return;
  const int64_t base = row * hidden;
  float sum = 0.0f;
  for (int64_t col = 0; col < hidden; ++col) {
    const float z = value<T, Fused>(x, residual, base + col);
    sum += z * z;
  }
  const float inv = rsqrtf(sum / hidden + eps);
  for (int64_t col = 0; col < hidden; ++col) {
    out[base + col] = static_cast<T>(value<T, Fused>(x, residual, base + col) *
                                    inv * static_cast<float>(weight[col]));
  }
}

__device__ __forceinline__ float warp_sum(float sum) {
  for (int offset = 16; offset > 0; offset /= 2)
    sum += __shfl_down_sync(0xffffffff, sum, offset);
  return sum;
}

// One block per row: coalesced loads, warp reductions, only 8 shared partials.
template <typename T, bool Fused>
__global__ void block_kernel(const T* x, const T* weight, const T* residual,
                             T* out, int64_t hidden, float eps) {
  const int tid = threadIdx.x;
  const int64_t base = static_cast<int64_t>(blockIdx.x) * hidden;
  float sum = 0.0f;
  for (int64_t col = tid; col < hidden; col += blockDim.x) {
    const float z = value<T, Fused>(x, residual, base + col);
    sum += z * z;
  }
  sum = warp_sum(sum);
  __shared__ float partials[kThreads / 32];
  __shared__ float inv;
  if ((tid & 31) == 0) partials[tid / 32] = sum;
  __syncthreads();
  if (tid < 32) {
    sum = tid < kThreads / 32 ? partials[tid] : 0.0f;
    sum = warp_sum(sum);
    if (tid == 0) inv = rsqrtf(sum / hidden + eps);
  }
  __syncthreads();
  for (int64_t col = tid; col < hidden; col += blockDim.x) {
    out[base + col] = static_cast<T>(value<T, Fused>(x, residual, base + col) *
                                    inv * static_cast<float>(weight[col]));
  }
}

template <typename T, bool Fused>
void launch(const torch::Tensor& x, const torch::Tensor& weight,
            const T* residual, torch::Tensor& out, float eps, bool serial,
            cudaStream_t stream) {
  const int64_t hidden = x.size(-1), rows = x.numel() / hidden;
  if (serial) {
    serial_kernel<T, Fused><<<(rows + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
        x.data_ptr<T>(), weight.data_ptr<T>(), residual, out.data_ptr<T>(), rows, hidden, eps);
  } else {
    block_kernel<T, Fused><<<rows, kThreads, 0, stream>>>(
        x.data_ptr<T>(), weight.data_ptr<T>(), residual, out.data_ptr<T>(), hidden, eps);
  }
}
}  // namespace

torch::Tensor rmsnorm_cuda(torch::Tensor x, torch::Tensor weight,
                           std::optional<torch::Tensor> residual,
                           double eps, bool serial) {
  const c10::cuda::CUDAGuard guard(x.device());
  auto out = torch::empty_like(x);
  if (x.numel() == 0) return out;
  const auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
      x.scalar_type(), "rmsnorm_cuda", [&] {
        if (residual.has_value())
          launch<scalar_t, true>(x, weight, residual->data_ptr<scalar_t>(), out,
                                static_cast<float>(eps), serial, stream);
        else
          launch<scalar_t, false>(x, weight, nullptr, out,
                                 static_cast<float>(eps), serial, stream);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
