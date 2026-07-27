# CUDA 基本功：两个教学 kernel

这个目录包含 7 个可以分别编译、分别运行的 CUDA 示例。重点是把线程协作、warp、访存合并和 shared-memory bank conflict 变成可观察的行为，而不是追求绝对性能。

## 前置条件与构建

需要 CUDA Toolkit（含 `nvcc`）和 NVIDIA GPU。Windows PowerShell 中：

```powershell
cd cuda-teaching-kernels
make
```

Makefile 默认使用课程要求的 `nvcc -O3 -arch=native`；可指定架构，例如 `make ARCH=-arch=sm_86`。每个程序亦可单独编译：

```powershell
nvcc -O3 -arch=native reduction_01_naive_atomic.cu -o reduction_01_naive_atomic.exe
```

运行所有示例：

```powershell
make run
```

每个程序在测速前先与 CPU 参考结果比对（绝对误差容差 `1e-4`），随后 warm-up 5 次，测量 20 次，并报告 CUDA event 的中位数。结果会自动追加到 `results.md`，所以此文件是**本机实测记录**，不应提交虚构数据。

## Reduction：数组求和

| 文件 | 递进重点 | 预期相对表现 |
| --- | --- | --- |
| `reduction_01_naive_atomic.cu` | 每线程 `atomicAdd` 到全局结果 | 最慢；大量原子竞争 |
| `reduction_02_shared_tree.cu` | shared-memory 树形块归约、`__syncthreads()` | 显著更快 |
| `reduction_03_warp_shuffle.cu` | 最后一个 warp 用 `__shfl_down_sync` | 消除末段 block 同步 |
| `reduction_04_grid_stride.cu` | grid-stride loop + 每线程多元素累加 | 适配任意规模，是常见工业写法 |

### 自查：为何 `tid % (2*s) == 0` 比 `tid < s` 慢？

在每一步归约中，取模式条件让同一个 warp 的活跃线程呈交错分布；warp 必须串行执行不同分支路径，造成 **warp divergence**，且取模本身也更昂贵。`tid < s` 让活跃线程形成连续前缀，控制流与内存访问都更友好。

## Matrix transpose：访存模式

| 文件 | 重点 | 预期相对表现 |
| --- | --- | --- |
| `transpose_01_naive.cu` | 直接 `out[col][row] = in[row][col]` | 读合并、写跨步且不合并 |
| `transpose_02_shared_tile.cu` | `32x32` shared tile 中转 | 全局读写均合并；但 shared 列访问可能 bank conflict |
| `transpose_03_padded_tile.cu` | tile 为 `[32][33]` | padding 打散 bank 映射，消除列访问的 bank conflict |

建议分别 profiling 后对比 2、3：

```powershell
ncu --set full .\transpose_02_shared_tile.exe
ncu --set full .\transpose_03_padded_tile.exe
```

在 Nsight Compute 中查看：

- **SOL**：判断 kernel 更偏计算受限还是访存受限；
- **Memory Workload Analysis**：查看全局内存带宽利用率；
- shared-memory 的 `shared_load_bank_conflicts`（不同版本界面字段名称可能略有差异）：padding 版应接近零。

> 首次 `ncu` 可能需要管理员权限或 NVIDIA profiling 权限；这是驱动安全策略，不是程序错误。

## 本机结果

执行 `make run` 后在此处或 `results.md` 记录 GPU 型号、CUDA 版本、输入规模和各程序输出的中位数。不同 GPU 的绝对时间不可横向比较，重点看同一机器上的层级趋势。
