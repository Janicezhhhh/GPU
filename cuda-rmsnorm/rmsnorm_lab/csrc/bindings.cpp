#include <torch/extension.h>
#include <cmath>
#include <limits>
#include <optional>

torch::Tensor rmsnorm_cuda(torch::Tensor x, torch::Tensor weight,
                           std::optional<torch::Tensor> residual,
                           double eps, bool serial);

torch::Tensor forward(torch::Tensor x, torch::Tensor weight,
                      std::optional<torch::Tensor> residual,
                      double eps, bool serial) {
  TORCH_CHECK(x.is_cuda(), "x must be CUDA");
  TORCH_CHECK(x.dim() >= 1 && x.size(-1) > 0, "invalid hidden dimension");
  TORCH_CHECK(x.scalar_type() == at::kFloat || x.scalar_type() == at::kHalf ||
              x.scalar_type() == at::kBFloat16, "unsupported dtype");
  TORCH_CHECK(std::isfinite(eps) && eps >= std::numeric_limits<float>::min() &&
              eps <= std::numeric_limits<float>::max(),
              "eps must be a positive normal float32 value");
  TORCH_CHECK(weight.dim() == 1 && weight.numel() == x.size(-1), "invalid weight shape");
  auto check = [&](const torch::Tensor& t) {
    TORCH_CHECK(t.device() == x.device() && t.scalar_type() == x.scalar_type(),
                "device and dtype must match x");
    TORCH_CHECK(t.is_contiguous(), "inputs must be contiguous");
    TORCH_CHECK(!t.requires_grad(), "inference only");
  };
  check(x);
  check(weight);
  if (residual.has_value()) {
    check(*residual);
    TORCH_CHECK(residual->sizes() == x.sizes(), "invalid residual shape");
  }
  TORCH_CHECK(x.numel() / x.size(-1) <= std::numeric_limits<int>::max(),
              "too many rows for the launch grid");
  return rmsnorm_cuda(x, weight, residual, eps, serial);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Inference RMSNorm with optional residual fusion");
}
