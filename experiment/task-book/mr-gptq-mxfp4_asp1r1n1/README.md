# MR-GPTQ MXFP4 低比特量化算法（7月社区任务）

- **任务**：AMCT 低比特量化算法开发（量化算法方向，进阶项"未适配新算法"）
- **算法**：MR-GPTQ（Micro-Rotated GPTQ，arXiv:2509.23202），子类化 AMCT `GPTQuant` 实现，运行时注册、**零改 amct 源码**
- **数据类型**：MXFP4（weight-only W4A16）
- **测试模型**：Qwen3-8B（dense）、Qwen3.6-35B-A3B（`qwen3_5_moe` 混合 MoE）　|　**数据集**：WikiText2　|　**指标**：PPL（seq_len=4096）
- **精度目标**：`delta = ppl_quant − ppl_bf16 ≤ 0.4`；量化层（nn.Linear）占比 ≥ 70%；显存降低 ≥ 50%
- **提交者**：程逸雷 / gitcode `asp1r1n1`

## 当前状态：✅ 8B 与 35B 三项硬指标全部达标

| 模型 | bf16 PPL | MR-GPTQ PPL | **delta** | 量化层覆盖 | **显存降低** | 判定 |
|------|----------|-------------|-----------|------------|--------------|------|
| Qwen3-8B（dense） | 8.9893 | 9.2621 | **0.273** | 99.6%（nn.Linear） | **67.4%** | ✅ |
| Qwen3.6-35B-A3B（MoE） | 6.311 | 6.3781 | **0.067** | 71.4% nn.Linear / **全参数 97%** | **71.2%** | ✅ |

要求：delta ≤ 0.4、覆盖 ≥ 70%、显存降 ≥ 50%。详见 [自验证报告](docs/self-verification-report.md)、[实验记录](docs/experiment-log.md)。

## 目录

```
mr-gptq-mxfp4_asp1r1n1/
├── README.md
├── docs/
│   ├── design.md                    # 设计文档（评审对象，v0.5 含实现补记）
│   ├── task-book.md                 # 任务书原文存档
│   ├── experiment-log.md            # 实验流水与踩坑
│   ├── self-verification-report.md  # 自验证报告（数据+截图）
│   └── img/                         # 运行截图
└── src/
    ├── run_eval.py         # 评测入口（bf16 / gptq_mxfp4 / mr_gptq）
    ├── mr_gptq_module.py   # ★ MR-GPTQ 实现（核心交付）
    ├── eval_utils.py       # 加载 / 校准 / PPL / 覆盖率与显存统计
    ├── data.py             # WikiText2 / pile 数据集加载
    └── logs/               # 运行日志
```

## 方案概述

MR-GPTQ = GPTQ 骨架 + 三增量：① **block-wise Hadamard 微旋转**（k=128，正态化权重分布、摊平离群值）② **MXFP scale fitting**（附录 H，E8M0 scale 网格重映射为次幂次）③ 静态激活重排。以仓内 GPTQ-MXFP4 为 baseline 对照证明精度提升。

工程要点：MXFP4 是纯 torch fake-quant（无需 NPU 算子）；GPTQ 补偿移到 CPU 消除 per-column kernel 启动开销（~15× 提速）；MoE 融合路由专家（非 nn.Linear、amct 覆盖不到）额外实现 MXFP4 量化以达显存指标。详见 [design.md](docs/design.md)。

## 运行方式

```bash
cd src
export ASCEND_RT_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1

# --- Qwen3-8B（单卡）---
python3 -u run_eval.py --model_path <Qwen3-8B> --mode bf16     --seq_len 4096
python3 -u run_eval.py --model_path <Qwen3-8B> --mode mr_gptq  --increments hadamard scale_fitting --seq_len 4096

# --- Qwen3.6-35B-A3B（双卡，35B 单卡放不下）---
export ASCEND_RT_VISIBLE_DEVICES=0,1
python3 -u run_eval.py --model_path <35B> --mode bf16    --seq_len 4096 --device_map auto
python3 -u run_eval.py --model_path <35B> --mode mr_gptq --increments hadamard scale_fitting --seq_len 4096 --device_map auto
```

运行即打印：量化层覆盖率（`[coverage]`/`[actual-coverage]`）、融合专家量化（`[fused-experts]`）、显存降低（`[memory-full]`）、WikiText2 PPL（`Score:`）。环境搭建见 [experiment-log.md](docs/experiment-log.md) §2。

## ⚠️ 安全提示

本项目对 `qwen3_5_moe` 等**自定义模型架构**，在 `eval_utils.py` 加载时使用了
`trust_remote_code=True`（该架构未并入 transformers 主干，不开则无法加载）。此参数会让
HuggingFace 模型仓内的 `modeling_*.py` / `tokenization_*.py` 等自定义代码在加载阶段**直接执行**，
等同于以当前进程权限运行任意 Python 代码。若 `--model_path` 指向被篡改或来路不明的模型，可能导致
**主机被控、数据泄露**。

使用建议：
- **仅从受信任来源获取模型**（如官方 ModelScope `Qwen/...`、HuggingFace 官方仓）；
- 尽量**固定到已知安全的版本 / commit hash**，避免加载被篡改的最新快照；
- 不要把 `--model_path` 指向未经核验的第三方模型目录。

## 交付说明

- **设计文档**（交付 #3）：分支 `feat/mr-gptq-mxfp4`，MR 合入 `cann/amct:feature/community-tasks`（评审对象）。
- **算法代码 + 自验证报告**（交付 #1/#2）：本分支 `feat/mr-gptq-impl`，目录 `experiment/task-book/mr-gptq-mxfp4_asp1r1n1/`。
