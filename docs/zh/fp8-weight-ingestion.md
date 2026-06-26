# FP8 原始权重接入设计文档

## 背景

部分模型（如 DeepSeek-V3、Hy3-preview 潜在计划）会以 FP8 格式存储原始权重（`float8_e4m3fn`），
由 HuggingFace compressed-tensors 标准描述。amct 原有流程假设原始权重为 BF16；本文档说明
如何在 eval / ptq 入口透明接入 FP8 原始权重。

**落地分支**：`feat/fp8-weight-ingestion`  
**核心改动**：`amct_pytorch/common/models/llm/common/base.py`

---

## FP8 权重标准（HF compressed-tensors）

### safetensors 布局

```
model.layers.N.mlp.gate_proj.weight        → dtype=float8_e4m3fn
model.layers.N.mlp.gate_proj.weight_scale  → dtype=float32 (or bfloat16)
model.layers.N.self_attn.q_proj.weight     → dtype=float8_e4m3fn
model.layers.N.self_attn.q_proj.weight_scale → dtype=float32
```

### config.json 标志字段

```json
{
  "quantization_config": {
    "format": "float-quantized",
    "quant_method": "compressed-tensors",
    "quantization_status": "compressed",
    "config_groups": {
      "group_0": {
        "weights": { "num_bits": 8, "type": "float" }
      }
    }
  }
}
```

检测条件：`format ∈ {"float-quantized", "naive-quantized"}` 且 `quant_method == "compressed-tensors"`。

### scale 粒度（三种）

| 粒度 | shape | dequant 公式 | 典型使用场景 |
|---|---|---|---|
| per-tensor | `()` 或 `(1,)` | `w * scale` | 轻量模型、实验场景 |
| per-channel | `(out_features,)` | `w * scale[:, None]`（广播） | 主流 LLM（Llama、Qwen 等）|
| per-group | `(out_features, in_features // group_size)` | block 广播 | DeepSeek 块量化（block_size=128）|

**实现选择**：本方案用 `tensor.float() * scale.float()` 直接乘，依赖 PyTorch 广播规则，
三种粒度均可处理，无需分支判断。

---

## 实现说明

### 1. FP8 检测（`_detect_source_fp8`）

在 `BaseModel.__init__` 末尾调用一次，结果存 `self._source_is_fp8 : bool`。
读 `config.json` 而非依赖 safetensors header，避免每次 open 文件。

### 2. dequant 逻辑（`_dequant_fp8_state_dict`）

在 `load_layer_weight` 返回前调用，输入/输出均为 `dict[str, Tensor]`。

**scale 键名查找顺序**：

```
1. {weight_name}_scale        — HF compressed-tensors 标准
2. {weight_name}_scale_inv    — DeepSeek 约定（存倒数，dequant = w / scale_inv）
3. 无 scale → 直接 .to(bfloat16)，打 WARNING
```

**跳过的键**：以 `_scale`、`_scale_inv`、`zero_point` 结尾的 tensor 不写入干净 state_dict，
避免 `load_state_dict(strict=True)` 报 unexpected key。

### 3. 调用点（`load_layer_weight`）

```python
if self._source_is_fp8:
    state_dict = self._dequant_fp8_state_dict(state_dict)
return state_dict
```

`_source_is_fp8=False` 时零开销，BF16 模型路径完全不变。

---

## embed / lm_head 处理

`load_embed_state_dict()` 调用 `load_layer_weight(prefix)`，因此 embed_tokens、norm、lm_head
也会经过相同的 dequant 逻辑。

### 当前状态

- embed_tokens / norm：FP8 量化模型通常不量这两层，权重已是 BF16，`element_size() != 1`，
  dequant 逻辑自动跳过，无影响。
- lm_head：主流 FP8 模型（DeepSeek V3 等）lm_head 也保持 BF16，同上。

### 待支持场景（全量 FP8 包含 lm_head）

如果模型将 lm_head 也量化为 FP8（较少见，但理论存在），当前实现可以正确处理：
`load_layer_weight("lm_head.")` 读到 FP8 weight + weight_scale，经 `_dequant_fp8_state_dict`
还原为 BF16 后传给 `module.load_state_dict()`，`nn.Linear` 正常接收 BF16，无需额外改动。

**唯一潜在问题**：若 lm_head 配置了 `enable_lm_head_fp32=true`（Hy3-preview 有此字段），
模型期望 lm_head.weight 是 float32。当前 dequant 输出 bfloat16，会触发 dtype mismatch。

**处理方案**（待需求确认后实施）：

```python
# _dequant_fp8_state_dict 末尾加一行
tensor = tensor.to(torch.get_default_dtype())  # 跟随全局 dtype 而非写死 BF16
```

或在 `get_embed_load_specs` 返回的 lm_head 条目里加 cast hook。目前保守输出 BF16，
与 `BaseModel.block()` 末尾 `.bfloat16()` 统一。

---

## 不在本次范围内

| 项目 | 说明 |
|---|---|
| MX 格式（block_size=32）FP8 | deploy `_run_tensorwise` 已单独处理，eval/ptq 路径暂不覆盖 |
| hifloat8（华为私有 FP8）| `element_size()` 也是 1，但 scale 语义不同；待 hifloat8 权重格式确认后扩展 |
| FP8 activation 量化配置透传 | 原始模型 activation 是否也 FP8 对 amct eval 无关（amct 重新量化 activation）|
| 适配器级 scale 粒度验证 | 需拿到实际 Hy3 FP8 权重后通过 `weight_scale.shape` 确认 |

---

## 验证计划（待权重到位）

```bash
# 1. 检测是否命中 FP8 路径
python3 -c "
from types import SimpleNamespace
from amct_pytorch.common.models.llm.common.base import BaseModel
args = SimpleNamespace(model='/path/to/hy3-fp8', quant_target=['moe'])
# 只检测，不实例化
import json, os
cfg = json.load(open(os.path.join(args.model, 'config.json')))
print(cfg.get('quantization_config', {}).get('format'))
"

# 2. 跑 BF16 baseline eval，验证加载不报错
python -m amct_pytorch.eval \
  --model /path/to/hy3-fp8 \
  --model_name hy_v3 --device npu:1 \
  --granularity block --eval_mode bf16 \
  --bit_config amct_pytorch/configs/bf16.yaml --seq_len 4096

# 3. 对比 FP8 dequant 后的 BF16 PPL 与原始 BF16 模型 PPL
#    期望：两者接近（dequant 误差 < 0.01 PPL）
```
