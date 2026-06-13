# Qwen3-4B · 系列 qwen · 结构 dense

> **速览**：仓内首条 `qwen3` dense 适配。adapter `qwen/qwen3/qwen3.py`，纯复用主链（不改 workflow/solver）。首推 `attn-linear + mlp` W8A8-int（block 直转 `delta ≤ 0.2`，可直接 deploy）。

## 结构与适配要点

- 标准 dense Qwen3 decoder（无 MoE）。attention 沿用官方 `Qwen3Attention`；quant wrapper 拆为 `attn-linear`（q/k/v/o 投影）+ `attn-cache`（KV）两类目标。
- **复用**：`BaseModel` blockwise 工作流、`QuantLinear`、`QuantGatedMLP`。**新增**：无（dense 路径纯复用）。
- **起步复用清单**（下一条 dense Qwen3）：从 `qwen3.py`、`quant_module.py` 起步，首轮直接 `attn-linear + mlp` W8A8-int。

## 适配验证结论

- 标准三步闭环（BF16 baseline → 关闭量化浮点等价 → 最小 PTQ smoke）全部通过。
- **模型特定结论**：`attn-linear`、`mlp` 在 `bits=16` 下与 BF16 **逐项对齐** → rebuilt quant module 前向语义正确，量化误差来自数值精度而非 wrapper 结构；`block0/mlp` 单元 PTQ 链（提取→训练→导出→回载）已打通。

## 关键陷阱（模型/框架内在，可复用）

- **`lm_head` 与 `embed_tokens` 权重共享（tied）** —— 现象：找不到独立 `lm_head.weight`。处理：adapter 内把 `lm_head` 绑回 `model.embed_tokens`，勿按独立 key 去 checkpoint index 查找。
- **`Catcher` 未透传 `attention_type`** —— 现象：替换 `layer0` 后官方 forward 在 embedding 阶段报错。处理：`Catcher` 显式透传 `attention_type`。
- **`attn-linear` wrapper 替换了官方 attention kernel** —— 现象：step8 关闭量化等价明显漂移。根因：仅量化 q/k/v/o 时不应改 `attention_interface`。处理：继续复用官方 interface。

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
