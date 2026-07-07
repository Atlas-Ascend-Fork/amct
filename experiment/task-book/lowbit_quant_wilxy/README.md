# 低比特量化算法 — 社区任务提交

> **提交人**: wilxy  
> **分支**: `feature/community-tasks`  
> **目录**: `experiment/task-book/lowbit_quant_wilxy/`  
> **日期**: 2026-07-07

---

## 概述

本提交为 AMCT 低比特量化算法社区任务，新增两个预置量化配置 + 三个模型验证脚本，覆盖 HiFloat8、MXFP4、INT4 三种低比特格式。

### 新增预置配置

| 配置名 | 格式 | 算法 | 策略 | 用途 |
|--------|------|------|------|------|
| `HIFP8_OFMR_CHANNEL_CFG` | HiFloat8 W+A | OFMR | channel | 权重通道级 OFMR，比 tensor 级精度更高 |
| `MXFP4_GPTQ_WEIGHT_QUANT_CFG` | MXFP4 W-only | GPTQ | group(32) | MXFP4 格式 + GPTQ Hessian 补偿 |

### 示例脚本

| 脚本 | 模型 | 配置 | 说明 |
|------|------|------|------|
| `src/run_qwen3_hifp8.py` | Qwen3-8B | HIFP8_OFMR_CHANNEL_CFG | HiFloat8 通道级 OFMR W+A |
| `src/run_qwen3_mxfp4.py` | Qwen3-8B | MXFP4_GPTQ_WEIGHT_QUANT_CFG | MXFP4 GPTQ W-only |
| `src/run_qwen3_moe_int4.py` | Qwen3.6-35B-A3B | INT4_GPTQ_WEIGHT_QUANT_CFG | MoE INT4 GPTQ 基线 |

---

## 快速开始

### 环境要求

- Ascend 950 (Atlas A5) + CANN 对应版本
- Python 3.10+ + PyTorch 2.9+ + torch_npu
- AMCT PyTorch 已安装（`import amct_pytorch` 可用）

```bash
pip install -r requirements.txt
```

### 运行示例

```bash
# Qwen3-8B HiFloat8 channel OFMR
python src/run_qwen3_hifp8.py --model_path /path/to/Qwen3-8B

# Qwen3-8B MXFP4 GPTQ
python src/run_qwen3_mxfp4.py --model_path /path/to/Qwen3-8B

# Qwen3.6-35B-A3B MoE INT4 GPTQ
python src/run_qwen3_moe_int4.py --model_path /path/to/Qwen3.6-35B-A3B
```

### 直接使用配置

```python
import sys
sys.path.insert(0, "experiment/task-book/lowbit_quant_wilxy")
from config import HIFP8_OFMR_CHANNEL_CFG, MXFP4_GPTQ_WEIGHT_QUANT_CFG

import amct_pytorch as amct
amct.quantize(model, HIFP8_OFMR_CHANNEL_CFG)
```

---

## 算法说明

### HiFloat8 channel OFMR

OFMR 对权重做逐 batch MSE 累积，在候选指数系数 `EC_CAND = (-5, ..., +5)` 中搜索使 MSE 最小的 `scale = 2^ec`。

- **tensor 策略**: 所有权重通道共用同一个 ec → `cout = 1`
- **channel 策略**: 每个输出通道独立选最优 ec → `cout = weight.shape[0]`

当 LLM 各通道数值范围差异大时，channel 级精度显著优于 tensor 级。

### MXFP4 GPTQ

MXFP4 (Microscaling FP4, E2M1) 使用 shared exponent 机制（group_size=32），每个 group 内共享指数位。GPTQ 在此基础上：

1. Hessian 累积：calibration 阶段累积逐层输出二阶信息
2. 逐列量化 + 补偿：量化每列后立即用 Hessian 逆补偿剩余列
3. 格式无关：GPTQ 只关心最小化输出误差

---

## 精度目标

| 编号 | 模型 | 算法 | 格式 | 预期 PPL 劣化 | 预期量化比例 |
|:--:|------|------|------|:--:|:--:|
| TC-01 | Qwen3-8B | OFMR channel | HiFloat8 W+A | ≤ 0.1 | ≥ 85% |
| TC-02 | Qwen3-8B | OFMR channel | HiFloat8 W-only | ≤ 0.1 | ≥ 85% |
| TC-03 | Qwen3-8B | GPTQ | MXFP4 W-only | ≤ 0.4 | ≥ 75% |
| TC-04 | Qwen3.6-35B-A3B | GPTQ | INT4 W-only | ≤ 0.4 | ≥ 70% |
| TC-05 | Qwen3.6-35B-A3B | OFMR channel | HiFloat8 W-only | ≤ 0.1 | ≥ 80% |

---

## 上游合入建议

当前 `config.py` 以独立模块提供。建议将以下配置合入 `amct_pytorch/common/config/config.py` 并在两级 `__init__.py` 注册导出：

```python
HIFP8_OFMR_CHANNEL_CFG = {
    'batch_num': 1,
    'quant_cfg': {
        'weights': {
            'type': 'hifloat8', 'symmetric': True, 'strategy': 'channel',
        },
        'inputs': {
            'type': 'hifloat8', 'symmetric': True, 'strategy': 'tensor',
        },
    },
    'algorithm': {'ofmr'},
    'skip_layers': {'lm_head'}
}

MXFP4_GPTQ_WEIGHT_QUANT_CFG = {
    'batch_num': 1,
    'quant_cfg': {
        'weights': {
            'type': 'mxfp4_e2m1', 'symmetric': True,
            'strategy': 'group', 'group_size': 32,
        },
    },
    'algorithm': {'gptq'},
    'skip_layers': {'lm_head'}
}
```

---

## 文件清单

```
lowbit_quant_wilxy/
├── README.md                     # 本文档（中文）
├── README_en.md                  # English version
├── requirements.txt              # Python 依赖
├── config.py                     # 两个新增预置配置
└── src/
    ├── run_qwen3_hifp8.py        # HiFloat8 channel OFMR 示例
    ├── run_qwen3_mxfp4.py        # MXFP4 GPTQ 示例
    ├── run_qwen3_moe_int4.py     # MoE INT4 GPTQ 基线
    └── utils.py                  # 通用工具函数
```
