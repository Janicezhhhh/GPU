"""A CPU-only numerical walkthrough: python -m rmsnorm_lab.lesson."""
import math

import torch

from .ops import reference


def main():
    x = torch.tensor([[3.0, 4.0], [0.0, 0.0]])
    weight = torch.tensor([2.0, 0.5])
    eps = 1e-6
    mean_square = x.square().mean(-1, keepdim=True)
    inverse_rms = torch.rsqrt(mean_square + eps)
    output = reference(x, weight, eps=eps)
    # Independent scalar answer for a known case, not a GPU emulation.
    expected = torch.tensor([[6.0 / math.sqrt(12.5 + eps),
                              2.0 / math.sqrt(12.5 + eps)], [0.0, 0.0]])
    torch.testing.assert_close(output, expected)
    print("CPU walkthrough (this is not a CUDA performance result)")
    for name, tensor in (("x", x), ("mean(x^2)", mean_square),
                         ("inverse RMS", inverse_rms), ("weight", weight), ("output", output)):
        print(f"{name}: {tensor.tolist()}")
    residual = torch.tensor([[1.0, -1.0], [2.0, -2.0]])
    fused_reference = reference(x, weight, residual)
    torch.testing.assert_close(fused_reference, reference(x + residual, weight))
    print(f"residual sum: {(x + residual).tolist()}")
    print(f"residual RMSNorm: {fused_reference.tolist()}")
    print("Known-answer and residual-semantics checks passed on CPU.")


if __name__ == "__main__":
    main()
