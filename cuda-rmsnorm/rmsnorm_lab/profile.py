"""Focused capture for Nsight Compute after extension compilation/warmup."""
import argparse

import torch

from .ops import rmsnorm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--fused", action="store_true")
    parser.add_argument("--implementation", choices=["serial", "block"], default="block")
    args = parser.parse_args()
    if args.rows <= 0 or args.hidden <= 0:
        parser.error("rows and hidden must be positive")
    if not torch.cuda.is_available():
        parser.error("CUDA GPU required")
    torch.manual_seed(42)
    x = torch.randn(args.rows, args.hidden, device="cuda", dtype=torch.float16)
    w = torch.randn(args.hidden, device="cuda", dtype=x.dtype)
    residual = torch.randn_like(x) if args.fused else None
    for _ in range(10):
        rmsnorm(x, w, residual, implementation=args.implementation)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    try:
        rmsnorm(x, w, residual, implementation=args.implementation)
        torch.cuda.synchronize()
    finally:
        torch.cuda.cudart().cudaProfilerStop()


if __name__ == "__main__":
    main()
