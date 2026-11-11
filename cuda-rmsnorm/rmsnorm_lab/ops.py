import math
import os
from functools import lru_cache
from pathlib import Path

import torch


def _validate(x, weight, residual, eps):
    if x.ndim < 1 or x.shape[-1] == 0:
        raise ValueError("x must have a nonempty last dimension")
    if x.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise TypeError("supported dtypes: float32, float16, bfloat16")
    if weight.shape != (x.shape[-1],):
        raise ValueError("weight must have shape (x.shape[-1],)")
    for name, tensor in (("x", x), ("weight", weight), ("residual", residual)):
        if tensor is None:
            continue
        if tensor.device != x.device or tensor.dtype != x.dtype:
            raise ValueError(f"{name} must match x's device and dtype")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
        if tensor.requires_grad:
            raise ValueError("inference only: detach tensors requiring gradients")
    if residual is not None and residual.shape != x.shape:
        raise ValueError("residual must have the same shape as x")
    if not math.isfinite(eps) or not 1.1754943508222875e-38 <= eps <= 3.4028234663852886e38:
        raise ValueError("eps must be a positive normal float32 value")


def reference(x, weight, residual=None, eps=1e-6):
    """Eager PyTorch reference; residual addition rounds to the input dtype.

    Square, mean, normalization and weight multiplication use float32;
    the final result is cast to the input dtype. Does not mutate inputs.
    """
    _validate(x, weight, residual, eps)
    z = x if residual is None else x + residual
    z32 = z.float()
    return (z32 * torch.rsqrt(z32.square().mean(dim=-1, keepdim=True) + eps)
            * weight.float()).to(x.dtype)


@lru_cache(maxsize=1)
def _extension():
    from torch.utils.cpp_extension import CUDA_HOME, load

    if CUDA_HOME is None:
        raise RuntimeError("CUDA Toolkit/nvcc not found. See README setup instructions.")
    source = Path(__file__).parent / "csrc"
    return load(
        name="rmsnorm_lab_cuda_v1",
        sources=[str(source / "bindings.cpp"), str(source / "kernels.cu")],
        extra_cflags=["/O2" if os.name == "nt" else "-O3"],
        extra_cuda_cflags=["-O3", "-lineinfo"],
        verbose=os.environ.get("RMSNORM_BUILD_VERBOSE") == "1",
    )


def rmsnorm(x, weight, residual=None, eps=1e-6, implementation="block"):
    """Run serial or block CUDA RMSNorm, optionally fusing residual addition.

    Input: contiguous (..., hidden), matching contiguous (hidden,) weight.
    No autograd or CPU fallback. Use reference() for a CPU oracle.
    """
    _validate(x, weight, residual, eps)
    if implementation not in ("serial", "block"):
        raise ValueError("implementation must be 'serial' or 'block'")
    if not x.is_cuda:
        raise ValueError("rmsnorm requires CUDA tensors; use reference() on CPU")
    return _extension().forward(x, weight, residual, eps, implementation == "serial")
