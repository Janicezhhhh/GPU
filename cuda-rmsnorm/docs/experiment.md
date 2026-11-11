# 性能实验与优化复盘

此文件是待填写的实验记录，不包含任何虚构测量值。

## 实验环境

- GPU / 显存 / 驱动：待测
- OS / Python / PyTorch / CUDA Toolkit / 编译器：待测
- Git commit 与未提交修改：待记录
- `doctor.py` 输出、完整运行命令、原始 JSON：待附
- 是否有其他 GPU 负载、功耗/时钟限制：待记录

## 正确性

- `pytest` 结果与 skip 原因：待记录
- 各精度容差与最大误差：待记录
- 极端值、长 hidden、不同设备上是否仍适用：待验证

## 结果表

| rows | hidden | dtype | residual | eager μs | serial μs | block μs | unfused μs |
|---|---|---|---|---|---|---|---|
| 待测 | | | | | | | |

同时附每轮样本；比较中位数，并保留没有收益和退化的形状。

## 假设与证据

1. **并行归约**：串行版的访存与并行度为何不理想？block 版改善了什么？
2. **残差融合**：减少一次 kernel launch 和中间张量，但重复读取原始输入；实际收益取决于什么？
3. **尺寸变化**：rows 很少、hidden 很小或很大时，256 线程的策略有什么限制？
4. **数值误差**：FP32 归约顺序、FMA 和低精度舍入如何影响误差？

每条结论附对应 benchmark 或 Nsight 指标。不要仅凭理论猜测就宣布为带宽瓶颈。

## 下一次实验（每次只改一个主要因素）

- 多个 block size / 小 hidden 使用 warp-per-row。
- 对齐的向量化加载及非对齐 fallback。
- 缓存每线程中间值，减少重复读；同时测寄存器压力与 spill。
- 添加 `torch.compile` 或成熟实现作为更强对照，明确各自数值语义。
- 接入真实模型，验证端到端收益及数值影响。

## 可以在面试中解释的内容

画出线程到数据的映射；说明两次 block 同步的必要性；说明 why full warp mask is valid；解释为何融合残差和要先舍入；说明 current stream 和 device guard 的作用；区分微基准收益与真实模型收益。
