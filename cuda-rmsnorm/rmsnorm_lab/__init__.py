"""Inference-only RMSNorm kernels. CUDA compilation is lazy."""

from .ops import reference, rmsnorm

__all__ = ["reference", "rmsnorm"]
