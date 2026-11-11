import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def run(*args):
    # These are CPU-entry-point tests even when invoked on a GPU runner.
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1"}
    return subprocess.run([sys.executable, *args], cwd=ROOT, env=env,
                          capture_output=True, text=True, timeout=60)


def test_lesson_known_answer():
    result = run("-m", "rmsnorm_lab.lesson")
    assert result.returncode == 0, result.stderr
    assert "checks passed on CPU" in result.stdout


def test_benchmark_requires_gpu(tmp_path):
    output = tmp_path / "not-a-gpu-result.json"
    result = run("-m", "rmsnorm_lab.benchmark", "--output", str(output))
    assert result.returncode == 2
    assert "no CPU timing fallback" in result.stderr
    assert not output.exists()


def test_invalid_benchmark_size():
    result = run("-m", "rmsnorm_lab.benchmark", "--rows", "0")
    assert result.returncode == 2
    assert "must be positive" in result.stderr
