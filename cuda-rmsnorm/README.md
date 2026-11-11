# CUDA RMSNorm Lab

一个用于学习、测量和复盘的 CUDA 算子项目：从串行 RMSNorm，到块内并行归约，再到 Residual Add + RMSNorm 融合。项目独立放在 GPU 仓库的 cuda-rmsnorm 子目录中，可与现有 CUDA 教学示例配合学习。

**状态：CPU 测试 35 项通过、GPU 测试 123 项跳过，wheel 构建成功；GPU 编译、正确性和加速比尚未验证。不要把预期收益写成实测结果。** 本机验证记录见 [docs/validation.md](docs/validation.md)。

## 计算语义

```text
z = x                            # 普通 RMSNorm
z = cast_to_input_dtype(x + r)    # 残差融合模式
y = cast_to_input_dtype(z_fp32 * rsqrt(mean(z_fp32²) + eps) * weight_fp32)
```

- 输入形状 `(..., hidden)`，weight 为 `(hidden,)`，输出与输入同形状、同精度。
- 支持连续布局的 FP32 / FP16 / BF16；归约和乘法使用 FP32。
- residual 与 x 的形状、设备、精度一致；残差和先舍入到输入精度，与 `torch.add` 保持一致。
- 不修改输入，不额外返回残差和。仅支持推理，不实现反向传播；带 `requires_grad` 的输入会被拒绝。
- 支持非整齐 hidden、非零 storage offset、零行输入与当前 CUDA stream。hidden 不得为零。
- eps 默认 `1e-6`，必须在 FP32 正规正数范围内。
- BF16 实验需要支持该精度的 GPU。极端值可能导致 FP32 平方溢出；该实现没有额外的数值缩放机制。

## 实现与对照

| 名称 | 实现 | 目的 |
|---|---|---|
| `torch_eager` | 显式 FP32 中间计算的 PyTorch 表达式 | 语义参考与 eager 性能基准 |
| `cuda_serial` | 一个线程处理一行 | 教学基线，暴露串行归约与访存问题 |
| `cuda_block` | 每行一个 256 线程 block；warp shuffle + 共享内存归约 | 并行化与连续线程合并访存 |
| `cuda_unfused` | 单独 `torch.add`，再调用 block RMSNorm | 与融合版比较，隔离融合效果 |

普通与融合模式共享模板实现。融合版不分配全局残差中间张量，但目前在归约和输出两个阶段各读取一次输入，并重复计算残差和。是否更快必须实测；小 hidden 或很少行时固定 256 线程可能浪费资源。

当前没有显式向量化加载、形状 autotuning、`torch.compile` 专用集成或生产框架接入。`torch_eager` 不是 PyTorch 所能达到的最佳性能；对它的收益不能宣称为超过所有成熟实现。

## 环境准备

### 当前没有 NVIDIA GPU：先运行 CPU 阶段

在 `cuda-rmsnorm` 目录下运行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements-cpu.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-build-isolation --no-deps
.\.venv\Scripts\python.exe -m rmsnorm_lab.lesson
.\.venv\Scripts\python.exe -m pytest -q -ra
```

Linux 用 `.venv/bin/python` 替换上面的解释器路径。`requirements-cpu.txt` 固定本阶段的直接依赖版本，用于重复测试，不表示已经验证所有 PyTorch 版本。CPU 阶段不需要 CUDA Toolkit，也不会编译 CUDA 扩展。

学习入口：[从公式到线程映射](docs/learning.md)。仓库根目录的 `.github/workflows/cuda-rmsnorm-cpu.yml` 会对该项目的提交运行 CPU 测试与 wheel 打包；它不证明 CUDA 已通过验证。

### 后续 GPU 阶段

建议使用 Linux / WSL2 的 NVIDIA GPU 开发环境。需要 Python 3.10+、CUDA-enabled PyTorch、兼容的 CUDA Toolkit（含 nvcc）、C++ 编译器和 Ninja。PyTorch wheel 不等于已经安装完整 CUDA 编译工具链。

先进入本目录。创建虚拟环境并按 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/) 安装与你的 GPU 驱动和 CUDA 环境兼容的 PyTorch，然后安装本项目：

```bash
python -m venv .venv
source .venv/bin/activate
# 先按上面的官方链接安装适合机器的 PyTorch
python -m pip install -e ".[test]"
python scripts/doctor.py
python -m pytest -q
```

Windows 可用 `.venv\Scripts\Activate.ps1` 激活环境；编译需要 MSVC C++ Build Tools 和 nvcc 可用，建议从对应开发者终端启动。CPU-only PyTorch 只能运行参考实现和 CPU 测试。

第一次调用 CUDA 算子会通过 `torch.utils.cpp_extension.load` 编译，耗时不计入性能测试。构建缓存由 PyTorch 管理。可设置 `RMSNORM_BUILD_VERBOSE=1` 查看日志，内存不足时设置 `MAX_JOBS=1`。

## 最小示例

```python
import torch
from rmsnorm_lab import reference, rmsnorm

x = torch.randn(32, 4096, device="cuda", dtype=torch.float16)
w = torch.ones(4096, device="cuda", dtype=x.dtype)
r = torch.randn_like(x)
y = rmsnorm(x, w, residual=r)
torch.testing.assert_close(y, reference(x, w, r), rtol=3e-3, atol=3e-3)
```

## 正确性与性能测试

```bash
python -m pytest -q
python -m rmsnorm_lab.benchmark --rows 1 32 512 --hidden 127 1024 4096 --dtype float16 --output results/fp16.json
python -m rmsnorm_lab.benchmark --dtype float32 --output results/fp32.json
python -m rmsnorm_lab.benchmark --dtype bfloat16 --output results/bf16.json
```

测试覆盖三种精度、两种 CUDA 实现、融合/非融合、不同维数和非整齐尺寸、不修改输入、偏移视图、自定义 stream；多 GPU 时额外检查 device guard。没有 CUDA 时 GPU 测试明确跳过，不能据此宣布 GPU 测试通过。

benchmark 在每种方法计时前先校验结果，使用 CUDA events、warmup、多轮平均和轮间中位数，保存每轮耗时、最大绝对误差、环境和随机种子。它衡量重复 eager 调用下的设备时间窗口，包含可能的 Python/调度发射空隙，不是纯 kernel 指令时间。输入重复使用，是热缓存场景；不使用 CUDA Graph。

结果文件默认不覆盖；需要重跑到同一路径时显式加 `--overwrite`。实验开始就写入 `completed: false`，只有全部案例完成后才变为 `true`，避免把中断实验误当成完整结果。

性能报告应同时比较：

1. `serial / block`：并行化收益，注意串行版是较弱基线。
2. `torch_eager / block`：相对明确表达式的收益。
3. 残差模式的 `cuda_unfused / cuda_block`：融合收益。

记录 GPU、驱动、软件版本、运行命令和代码版本；空闲机器上重复实验。不要只报告最好的形状，也不要把 kernel 提速当作模型整体提速。脚本 JSON 保存 GPU、PyTorch/CUDA 版本等，驱动信息由 doctor 的 `nvidia-smi` 输出补充。

## Profiling 与下一步

安装 Nsight Compute 后，从本目录运行：

```bash
ncu --profile-from-start off --set full -o results/block python -m rmsnorm_lab.profile --rows 512 --hidden 4096 --fused
ncu --profile-from-start off --set full -o results/serial python -m rmsnorm_lab.profile --rows 512 --hidden 4096 --implementation serial --fused
```

先创建 `results` 目录。profile 脚本在编译和 warmup 完成后才开启捕获。部分设备需要管理员配置性能计数器权限。

按 [实验复盘模板](docs/experiment.md) 记录带宽、occupancy、寄存器、访存与归约开销，再决定是否增加向量化、warp-per-row 或其他形状策略。每个新策略都必须重跑正确性与基准。

## 文件结构

```text
rmsnorm_lab/ops.py              Python API、输入检查、PyTorch 参考实现
rmsnorm_lab/csrc/bindings.cpp   C++ 入口和边界检查
rmsnorm_lab/csrc/kernels.cu     串行 / block / residual 融合 CUDA 实现
rmsnorm_lab/benchmark.py        校验、计时和 JSON 输出
rmsnorm_lab/profile.py          定点 Nsight 捕获
tests/test_ops.py               CPU 与 GPU 测试
scripts/doctor.py              无依赖环境检查入口
docs/experiment.md              实验复盘模板
docs/validation.md              当前验证范围
```

## 发布与简历

项目位于 `Janicezhhhh/GPU` 仓库的普通子目录，阶段成果使用 `cuda-rmsnorm-lab` 分支，不创建嵌套 Git 仓库。发布前实际完成 GPU 测试并保留 benchmark 原始结果。

简历只描述自己理解、能够解释并复现的工作；标明实现和验证范围。数据尚未生成时，不填写加速倍数，不称为生产级实现。

技术参考：[PyTorch C++/CUDA 扩展](https://docs.pytorch.org/docs/stable/cpp_extension.html)、[NVIDIA warp-level primitives](https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/)、[CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)。
