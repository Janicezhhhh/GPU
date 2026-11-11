# 验证记录

日期：2026-11-11。

## 已完成

- 项目已迁入 `D:\Janice\Github\GPU\cuda-rmsnorm`，CI 位于 GPU 仓库根目录；FlyGame 工作树已恢复干净。
- Windows 11 / Python 3.12.9 / PyTorch 2.12.0+cpu / pytest 8.3.5。
- `python -m pytest -q -ra --junitxml=results/cpu-tests.xml`：**35 passed, 123 skipped**。
- 123 项跳过均需 CUDA GPU，其中 1 项需两张 GPU；它们没有被执行，不计为通过。
- CPU 已验证：已知数值答案、FP64 oracle 对比、低精度残差语义、非法输入、推理约束、空行和 CLI 错误处理。
- `python -m rmsnorm_lab.lesson`：已知数值和残差语义检查通过。
- `python -m pip wheel . --no-deps --no-build-isolation --wheel-dir dist`：wheel 构建成功。该 wheel 携带 CUDA 源码，**不代表 CUDA 已编译**。
- Python 语法与 TOML 配置解析通过。

## 本机依赖来源

下载 PyTorch 的尝试曾遇到网络超时和哈希不匹配，未绕过包校验。后来发现本机已有 `D:\Janice\Github\.venv` 中的 CPU PyTorch，于项目自己的 `.venv` 内通过本地 `.pth` 引用该环境，并在项目环境中安装缺少的测试工具，没有修改共享环境。

`.venv`、机器路径配置、测试结果和 wheel 均不纳入 Git。其他机器应按 README 新建环境，不能复制本机虚拟环境。

CI 使用独立安装的 PyTorch 2.6.0 CPU 环境，和本机版本不同；CI 的最终状态以 GitHub Actions 实际运行记录为准，不能从本机结果推断。

## 尚未完成

- CUDA 扩展编译、GPU 正确性、stream / 多卡验证。
- 所有性能测量、Nsight profiling、加速比和模型端到端收益。

本机检测到 Intel UHD Graphics 730，未检测到 NVIDIA GPU；用户确认暂时没有远程 GPU。因此本阶段完成 CPU 可验证部分，保留完整 GPU 验证入口，不新增未经 profiling 支撑的性能策略。
