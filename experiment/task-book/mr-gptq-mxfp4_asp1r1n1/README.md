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

## 当前阶段

**设计文档评审阶段**。设计文档见 [docs/design.md](docs/design.md)。算法代码、测试脚本、自验证报告将在设计文档评审通过后补充。

## 方案概述

基于 AMCT 已有 GPTQ（`amct_pytorch/classic/quantize_op/gptq_module.py`）子类化实现 MR-GPTQ，针对 MXFP4 微缩格式做三处增量：① MXFP scale fitting（E8M0 scale 网格重映射，头号增益）② block-wise Hadamard 微旋转 ③ 静态激活重排。以仓内 GPTQ-MXFP4 为 baseline 对照，验证精度提升。详见设计文档。
