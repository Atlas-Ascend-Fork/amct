# Low-Bit Quantization Algorithms — Community Task Submission

> **Author**: wilxy  
> **Branch**: `feature/community-tasks`  
> **Directory**: `experiment/task-book/lowbit_quant_wilxy/`  
> **Date**: 2026-07-07

---

## Overview

This submission adds two new preset quantization configs and three model
verification scripts covering HiFloat8, MXFP4, and INT4 low-bit formats.

### New Preset Configs

| Config | Format | Algorithm | Strategy | Use Case |
|--------|--------|-----------|----------|----------|
| `HIFP8_OFMR_CHANNEL_CFG` | HiFloat8 W+A | OFMR | channel | Per-channel scale search, tighter than tensor-level |
| `MXFP4_GPTQ_WEIGHT_QUANT_CFG` | MXFP4 W-only | GPTQ | group(32) | MXFP4 format + GPTQ Hessian compensation |

### Example Scripts

| Script | Model | Config | Description |
|--------|-------|--------|-------------|
| `src/run_qwen3_hifp8.py` | Qwen3-8B | HIFP8_OFMR_CHANNEL_CFG | HiFloat8 per-channel OFMR W+A |
| `src/run_qwen3_mxfp4.py` | Qwen3-8B | MXFP4_GPTQ_WEIGHT_QUANT_CFG | MXFP4 GPTQ W-only |
| `src/run_qwen3_moe_int4.py` | Qwen3.6-35B-A3B | INT4_GPTQ_WEIGHT_QUANT_CFG | MoE INT4 GPTQ baseline |

---

## Quick Start

### Requirements

- Ascend 950 (Atlas A5) + matching CANN version
- Python 3.10+ + PyTorch 2.9+ + torch_npu
- AMCT PyTorch installed (`import amct_pytorch` works)

```bash
pip install -r requirements.txt
```

### Run Examples

```bash
# Qwen3-8B HiFloat8 channel OFMR
python src/run_qwen3_hifp8.py --model_path /path/to/Qwen3-8B

# Qwen3-8B MXFP4 GPTQ
python src/run_qwen3_mxfp4.py --model_path /path/to/Qwen3-8B

# Qwen3.6-35B-A3B MoE INT4 GPTQ
python src/run_qwen3_moe_int4.py --model_path /path/to/Qwen3.6-35B-A3B
```

### Use configs directly

```python
import sys
sys.path.insert(0, "experiment/task-book/lowbit_quant_wilxy")
from config import HIFP8_OFMR_CHANNEL_CFG, MXFP4_GPTQ_WEIGHT_QUANT_CFG

import amct_pytorch as amct
amct.quantize(model, HIFP8_OFMR_CHANNEL_CFG)
```

---

## Algorithm Notes

### HiFloat8 channel OFMR

OFMR accumulates per-batch MSE on weights and searches for the optimal
exponent coefficient `scale = 2^ec` from `EC_CAND = (-5, ..., +5)`.

- **tensor**: all weight channels share a single `ec` → `cout = 1`
- **channel**: each output channel independently selects the best `ec` → `cout = weight.shape[0]`

Channel-level strategy yields better accuracy when weight channel magnitudes
vary significantly — the typical case for large language models.

### MXFP4 GPTQ

MXFP4 (Microscaling FP4, E2M1) uses a shared exponent per group of 32 elements.
GPTQ adds Hessian-based error compensation on top:

1. Hessian accumulation: layer-wise second-order info from calibration data
2. Column-wise quantization + compensation: quantize each column, then
   compensate remaining columns via inverse Hessian
3. Format-agnostic: GPTQ only cares about minimizing output error

---

## Accuracy Targets

| # | Model | Algorithm | Format | Expected ΔPPL | Expected Quant Ratio |
|:--:|-------|-----------|--------|:--:|:--:|
| TC-01 | Qwen3-8B | OFMR channel | HiFloat8 W+A | ≤ 0.1 | ≥ 85% |
| TC-02 | Qwen3-8B | OFMR channel | HiFloat8 W-only | ≤ 0.1 | ≥ 85% |
| TC-03 | Qwen3-8B | GPTQ | MXFP4 W-only | ≤ 0.4 | ≥ 75% |
| TC-04 | Qwen3.6-35B-A3B | GPTQ | INT4 W-only | ≤ 0.4 | ≥ 70% |
| TC-05 | Qwen3.6-35B-A3B | OFMR channel | HiFloat8 W-only | ≤ 0.1 | ≥ 80% |

---

## Upstream Merge Suggestion

The two configs below should be added to `amct_pytorch/common/config/config.py`
and registered in both levels of `__init__.py`:

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

## File Layout

```
lowbit_quant_wilxy/
├── README.md                     # Chinese documentation
├── README_en.md                  # English documentation
├── requirements.txt              # Python dependencies
├── config.py                     # Two new preset configs
└── src/
    ├── run_qwen3_hifp8.py        # HiFloat8 channel OFMR sample
    ├── run_qwen3_mxfp4.py        # MXFP4 GPTQ sample
    ├── run_qwen3_moe_int4.py     # MoE INT4 GPTQ baseline
    └── utils.py                  # Shared utilities
```
