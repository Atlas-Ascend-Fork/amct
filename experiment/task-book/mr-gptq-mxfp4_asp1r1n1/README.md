# MR-GPTQ MXFP4 低比特量化算法（7月社区任务）

- **任务**：AMCT 低比特量化算法开发（量化算法方向，进阶项"未适配新算法"）
- **算法**：MR-GPTQ（Micro-Rotated GPTQ，arXiv:2509.23202），子类化 AMCT `GPTQuant` 实现，运行时注册、**零改 amct 源码**
- **数据类型**：MXFP4（block=32，E2M1 元素 + E8M0 共享指数）。主线 **W4A16**（weight-only）；按两轮评审意见补充 **W4A4**（权重+激活均 4-bit），**8B 与 35B 均达标**
- **测试模型**：Qwen3-8B（dense）、Qwen3.6-35B-A3B（`qwen3_5_moe` 混合 MoE）　|　**数据集**：WikiText2　|　**指标**：PPL（seq_len=4096）
- **精度目标**：`delta = ppl_quant − ppl_bf16 ≤ 0.4`；量化层（nn.Linear）占比 ≥ 70%；显存降低 ≥ 50%
- **提交者**：程逸雷 / gitcode `asp1r1n1`

## 当前状态：✅ 8B 与 35B 三项硬指标全部达标

**W4A16 主线**

| 模型 | bf16 PPL | MR-GPTQ PPL | **delta** | 量化层覆盖 | **权重压缩率** | 判定 |
|------|----------|-------------|-----------|------------|--------------|------|
| Qwen3-8B（dense） | 8.9893 | 9.2621 | **0.273** | 99.6%（nn.Linear） | **62.3%** | ✅ |
| Qwen3.6-35B-A3B（MoE） | 6.311 | 6.3781 | **0.067** | 71.4% nn.Linear / 全参数 97% | **71.2%** | ✅ |

**W4A4（权重+激活均 4-bit，按两轮评审意见先后补充）**

| 模型 | bf16 PPL | W4A4 PPL | **delta** | **真 4-bit 激活层**（任务书口径） | 判定 |
|------|----------|----------|-----------|------------------|------|
| Qwen3-8B | 8.9893 | 9.3790 | **0.3897** | 180/252 = **71.4%** | ✅ |
| Qwen3.6-35B-A3B | 6.311 | 6.6352 | **0.3242**（三 seed 0.2843–0.3242） | 250/350 = **71.4%**（**无保护层**） | ✅ |

要求：delta ≤ 0.4、量化层（nn.Linear）占比 ≥ 70%、显存降 ≥ 50%。

> ⚠️ **三点口径**（详见报告 §4.2 / §4.4 / §4.3 ⑤）：
> ① 35B 的「全参数 97%」是 **MXFP4 数据类型**的覆盖率，完整 MR-GPTQ 算法只作用于
> 约 4% 的参数（250 个 nn.Linear）；「71.4% 的层跑真 4-bit」也**不等于** 71.4% 的算力，
> 35B 算力大头是 32B 融合专家、激活仍为 fp16。
> ② 权重压缩率为**解析推算**（4.25 bit/参数），非运行时实测——fake-quant 下张量仍以 fp16 承载。
> ③ 报告已按实测的**测量分辨率**（跨校准 seed 极差 0.111 / 同 seed 响应面粗糙度 0.06）
> 重新审视全部归因，主动撤销四项低于该量级的结论。

详见 [自验证报告](docs/self-verification-report.md)、[实验记录](docs/experiment-log.md)。

## 目录

```
mr-gptq-mxfp4_asp1r1n1/
├── README.md
├── docs/
│   ├── design.md                    # 设计文档（评审对象，v0.8）
│   ├── task-book.md                 # 任务书原文存档
│   ├── experiment-log.md            # 实验流水与踩坑
│   ├── self-verification-report.md  # 自验证报告（数据+截图）
│   └── img/                         # 运行截图
└── src/
    ├── run_eval.py         # 评测入口（bf16 / gptq_mxfp4 / mr_gptq）
    ├── mr_gptq_module.py   # ★ MR-GPTQ 实现（核心交付）
    ├── eval_utils.py       # 加载 / 校准 / PPL / 覆盖率与显存统计
    ├── data.py             # WikiText2 / pile 数据集加载
    ├── run_ablations.sh    # 消融批跑（分组 / 断点续跑 / 自动汇总）
    ├── run_night.sh        # 夜间队列（测量分辨率复测与 35B W4A4）
    └── logs/               # 运行日志
```

## 方案概述

MR-GPTQ = GPTQ 骨架 + ① **block-wise Hadamard 微旋转**（正态化权重分布、摊平离群值）② **MXFP scale fitting**（附录 H，E8M0 scale 网格重映射为次幂次）。论文的第三项增量「静态激活重排」**有意未采用**——本实现已对每层施加全宽旋转，旋转后各通道近似可交换、按幅度分组已无物可分（报告 §5）。以仓内 GPTQ-MXFP4 为 baseline 对照证明精度提升。

W4A4 在此基础上另加激活量化与四项优化。经三 seed **配对复测**，其中两项确证、一项证伪：

| 手段 | 效应量 | 判定 |
|---|---|---|
| **混合精度保护**（`o_proj`/`down_proj` 保高精度激活） | **0.224** | ✅ 8B 不可省；**35B 必须弃用**（覆盖率口径，见报告 §4.4） |
| **旋转强度**（k=128 → 全宽；`down_proj` 的 12288 走 Paley H₁₂ ⊗ Sylvester H₁₀₂₄） | **0.147** | ✅ k=128 达 0.4993、**不达标** |
| ~~GPTQ 阻尼调优~~ | +0.0146 | ❌ **证伪**（2/3 seed 变差、方向随数据变号） |

⚠️ 达标依赖全宽旋转，而其在线开销在朴素矩阵乘下约 **+104%**（`k=128` 仅 1.6% 但不达标）——
**可部署性取决于是否实现 FWHT**（O(k log k)，开销 < 1%），本项目未实现。报告 §4.3 ④ 另记录 5 项未采纳的手段。

工程要点：MXFP4 是纯 torch fake-quant（无需 NPU 算子）；GPTQ 补偿移到 CPU 消除 per-column kernel 启动开销（~10× 提速）；MoE 融合路由专家（非 nn.Linear、amct 覆盖不到）额外实现 MXFP4 量化以达显存指标（RTN + scale fitting，无旋转/无补偿——完整 MR-GPTQ 仍只作用于 250 个 nn.Linear，口径见报告 §4.2）。详见 [design.md](docs/design.md)。

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

```bash
# --- W4A4 达标配置（Qwen3-8B，单卡；delta 0.3897）---
python3 -u run_eval.py --model_path <Qwen3-8B> --mode mr_gptq --seq_len 4096     --hadamard_k 4096 --hadamard_full --perc_damp 0.2     --act_protect down_proj o_proj     --increments hadamard scale_fitting act_quant act_scale_fit joint_comp
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

设计文档、算法代码与自验证报告**统一由单个 PR 交付**：分支 `feat/mr-gptq-impl` →
`cann/amct:feature/community-tasks`，目录 `experiment/task-book/mr-gptq-mxfp4_asp1r1n1/`。

- 交付 #1 自测用例 / 测试报告 / 步骤文档 → `docs/self-verification-report.md`（§3 为可复现步骤）、`src/logs/`
- 交付 #2 算法代码 → `src/`（核心为 `mr_gptq_module.py`）
- 交付 #3 设计文档 → `docs/design.md`
