# Qwen3-4B

首条 `qwen3` dense 适配案例。adapter：`amct_pytorch/common/models/llm/qwen/qwen3/qwen3.py`。

## 适配重点

- 标准 dense Qwen3 decoder（无 MoE）；attention 沿用官方 `Qwen3Attention`，quant wrapper 拆成 `attn-linear` 与 `attn-cache` 两类目标。
- `qwen3.py` 只用于 dense；`qwen3_moe.py` 独立 adapter，不共享目录语义。
- 复用：下一条 dense Qwen3 优先复用 `qwen3.py`、`quant_module.py`。

## 关键陷阱（可复用）

- **`lm_head` 与 `embed_tokens` tied**：不单独落盘，adapter 里需把 `lm_head` 绑回 `model.embed_tokens`；不要按独立 `lm_head.weight` 找 index。
- **`Catcher` 必须兼容 `attention_type`**：否则换掉 `layer0` 后官方 forward 在 embedding 阶段访问不到该字段而报错。
- **`attn-linear` wrapper 不能替换官方 attention kernel**：只做 q/k/v/o 量化时若把 `attention_interface` 换成自定义 matmul，step8 会明显漂；必须继续复用官方 interface。

## 精度速查表

> ppl 口径 seq_len=4096。MXFP 双值为两次评测口径 a/b。

| 数据类型 | 量化配置 | 量化算法 | ppl |
| --- | --- | --- | --- |
| BF16 | 无 | 无 | 12.377 |
| INT | A4w4: mlp, attn-linear | 无 | 9942.9922 |
| INT | A8w4: mlp, attn-linear | 无 | 19.9143 |
| INT | A8w8: mlp, attn-linear | 无 | 12.4327 |
| MXFP | A4w4: mlp, attn-linear | 无 | 16.043/14.1 |
| MXFP | A8w4: mlp, attn-linear | 无 | 13.808/13.066 |
| MXFP | A8w8: mlp, attn-linear | 无 | 12.6976/12.365 |
| HiFP | A4w4: mlp, attn-linear | 无 |  |
| HiFP | A8w4: mlp, attn-linear | 无 |  |
| HiFP | A8w8: mlp, attn-linear | 无 |  |
