"""Environment report; works even before PyTorch has been installed."""
import importlib.util
import json
import platform
import shutil
import subprocess


def main():
    report = {"python": platform.python_version(), "platform": platform.platform(),
              "tools": {name: shutil.which(name) for name in ["nvcc", "nvidia-smi", "ninja", "c++", "cl"]}}
    if importlib.util.find_spec("torch"):
        import torch
        from torch.utils.cpp_extension import CUDA_HOME
        report["torch"] = {"version": torch.__version__, "cuda_version": torch.version.cuda,
                           "cuda_available": torch.cuda.is_available(), "cuda_home": CUDA_HOME,
                           "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}
    else:
        report["torch"] = "not installed"
    if report["tools"]["nvidia-smi"]:
        result = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                                 "--format=csv,noheader"], capture_output=True, text=True, timeout=15)
        report["nvidia_smi"] = result.stdout.strip() or result.stderr.strip()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
