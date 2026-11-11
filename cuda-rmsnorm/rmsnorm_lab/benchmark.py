"""python -m rmsnorm_lab.benchmark --output results/benchmark.json"""
import argparse
import hashlib
import json
import platform
import statistics
import subprocess
from pathlib import Path

import torch

from .ops import reference, rmsnorm


def measure(fn, warmup, iterations, repeats):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / iterations)
    return {"median_us": statistics.median(samples), "samples_us": samples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 32, 512])
    parser.add_argument("--hidden", type=int, nargs="+", default=[127, 1024, 4096])
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float16")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("results/benchmark.json"))
    parser.add_argument("--overwrite", action="store_true", help="explicitly replace an existing result file")
    args = parser.parse_args()
    if min(args.rows + args.hidden + [args.warmup, args.iterations, args.repeats]) <= 0:
        parser.error("sizes and iteration counts must be positive")
    if not torch.cuda.is_available():
        parser.error("CUDA GPU and CUDA-enabled PyTorch are required; no CPU timing fallback")
    if args.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        parser.error("this GPU does not support bfloat16")
    torch.manual_seed(args.seed)
    dtype = getattr(torch, args.dtype)
    tolerance = {torch.float32: (2e-5, 2e-5), torch.float16: (3e-3, 3e-3),
                 torch.bfloat16: (2e-2, 2e-2)}[dtype]
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                           stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unavailable"
    report = {
        "completed": False,
        "environment": {"gpu": torch.cuda.get_device_name(),
                        "compute_capability": torch.cuda.get_device_capability(),
                        "torch": torch.__version__, "cuda": torch.version.cuda,
                        "python": platform.python_version(), "platform": platform.platform(),
                        "git_revision": revision,
                        "source_sha256": {str(p.relative_to(Path(__file__).parent)): hashlib.sha256(p.read_bytes()).hexdigest()
                                          for p in sorted(Path(__file__).parent.rglob("*"))
                                          if p.suffix in (".py", ".cpp", ".cu")}},
        "config": {**vars(args), "output": str(args.output), "eps": 1e-6},
        "method": "CUDA events, repeated eager calls, warm cache; excludes JIT build; includes launch gaps",
        "results": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.output.open("w" if args.overwrite else "x", encoding="utf-8") as output:
            json.dump(report, output, indent=2)
    except FileExistsError:
        parser.error(f"{args.output} already exists; choose a new path or pass --overwrite")
    with torch.inference_mode():
        for rows in args.rows:
            for hidden in args.hidden:
                x = torch.randn(rows, hidden, device="cuda", dtype=dtype)
                weight = torch.randn(hidden, device="cuda", dtype=dtype)
                residual = torch.randn_like(x)
                for fused in (False, True):
                    r = residual if fused else None
                    expected = reference(x, weight, r)
                    methods = {
                        "torch_eager": lambda: reference(x, weight, r),
                        "cuda_serial": lambda: rmsnorm(x, weight, r, implementation="serial"),
                        "cuda_block": lambda: rmsnorm(x, weight, r),
                    }
                    if fused:
                        methods["cuda_unfused"] = lambda: rmsnorm(x + residual, weight)
                    measured = {}
                    for name, fn in methods.items():
                        actual = fn()  # Build and verify outside timed region.
                        torch.testing.assert_close(actual, expected, rtol=tolerance[0], atol=tolerance[1])
                        stats = measure(fn, args.warmup, args.iterations, args.repeats)
                        measured[name] = {**stats, "max_abs_error":
                                          (actual.float() - expected.float()).abs().max().item()}
                    eager_us = measured["torch_eager"]["median_us"]
                    for stats in measured.values():
                        stats["speedup_vs_eager"] = eager_us / stats["median_us"]
                    entry = {"rows": rows, "hidden": hidden, "fused": fused, "methods": measured}
                    report["results"].append(entry)
                    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
                    print(f"{rows=} {hidden=} {fused=}: " + ", ".join(
                        f"{name}={stats['median_us']:.2f} us" for name, stats in measured.items()))
    report["completed"] = True
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
