# MR-GPTQ MXFP4 低比特量化算法（7月社区任务）

- **任务**：AMCT 低比特量化算法开发（量化算法方向）
- **算法**：MR-GPTQ（Micro-Rotated GPTQ，arXiv:2509.23202），AMCT 仓未适配算法（进阶项）
- **数据类型**：MXFP4（主线 weight-only W4A16，W4A4 进阶）
- **测试模型**：Qwen3-8B、Qwen3.6-35B-A3B　|　**数据集**：WikiText2　|　**指标**：PPL（seq_len=4096, block 粒度）
- **精度目标**：`delta = ppl_quant − ppl_bf16 ≤ 0.4`；量化层（nn.Linear）占比 ≥ 70%
- **提交者**：程逸雷 / gitcode `asp1r1n1`

## 目录

```
mr-gptq-mxfp4_asp1r1n1/
├── README.md              # 本文件
└── docs/
    ├── design.md          # 设计文档（评审对象）
    └── task-book.md       # 任务书原文存档
```

## 当前状态

本分支（`feat/mr-gptq-mxfp4`）为**设计文档评审分支**，仅含设计文档，是交付 #3 的评审对象。
算法代码、测试脚本、自验证报告与运行日志在**代码分支 `feat/mr-gptq-impl`**（交付 #1/#2）。

**算法已实现并完成自验证，8B 与 35B 三项硬指标全部达标：**

| 模型 | bf16 PPL | MR-GPTQ PPL | delta | 覆盖 | 显存降 | 判定 |
|------|----------|-------------|-------|------|--------|------|
| Qwen3-8B（dense） | 8.9893 | 9.2621 | 0.273 | 99.6% | 67.4% | ✅ |
| Qwen3.6-35B-A3B（MoE） | 6.311 | 6.3781 | 0.067 | 71.4% nn.Linear / 全参数 97% | 71.2% | ✅ |

设计文档见 [docs/design.md](docs/design.md)（v0.5，§9 含 35B 实测修正）。

## 方案概述

基于 AMCT 已有 GPTQ（`amct_pytorch/classic/quantize_op/gptq_module.py`）子类化实现 MR-GPTQ，针对 MXFP4 微缩格式做三处增量：① MXFP scale fitting（E8M0 scale 网格重映射，头号增益）② block-wise Hadamard 微旋转 ③ 静态激活重排。以仓内 GPTQ-MXFP4 为 baseline 对照，验证精度提升。详见设计文档。
