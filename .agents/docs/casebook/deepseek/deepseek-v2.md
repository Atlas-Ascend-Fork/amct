# DeepSeek-V2-Lite · 系列 deepseek · 结构 mla-moe（无 DSA）

> **速览**：adapter `deepseek/deepseek_v2/`（`DeepseekV2(BaseModel)`）。结构 = MLA + DeepSeekMoE，**无 DSA Indexer**（比 v3.2 简单一代）。W8A8 `attn-linear+moe` 直转 `delta=0.134`（达标）。本案是「agent 自主适配新模型」的实测产物——三步闭环 + 直转全通。
> **触发信号**：config `kv_lora_rank` 有值（MLA）+ `n_routed_experts>0`（MoE）+ **`q_lora_rank: null`**（Q 不压缩，区别于 v3/v3.2）+ **无 `index_topk`**（无 DSA）。读 [L2 · MLA 类 + MoE 类 + 接非 transformers 自定义 modeling](../family-pitfalls.md) + [L1](../common-pitfalls.md)。

## 结构与适配要点

- 27 层，layer 0 dense（`first_k_dense_replace=1`）、其余 DeepSeekMoE（64 routed + 2 shared，top-6）；MLA `kv_lora_rank=512`、`qk_nope=128`/`qk_rope=64`/`v=128`、`num_heads=16`；YaRN rope。
- **参考与差异**：最近参考 `deepseek-v3.2`（同 `DeepseekV32` 的 MLA+MoE 骨架）；**关键差异 = 去掉 DSA Indexer + Q 不压缩**。
- **复用**：`BaseModel` blockwise、`PtqUnit`、`QuantLinear`、`apply_quant_to_attn`/`apply_quant_to_moe_mlp`。**新增**：`deepseek_v2/{deepseekv2.py, quant_module.py, modeling/}` —— `QuantDeepseekV2Attention`（含 `q_lora_rank is None → q_proj` 分支）、`QuantDeepseekV2MLP`、PreTrainedModel 化的 `DeepseekV2ForCausalLM`。
- **起步复用清单**：下一条 MLA+MoE（无 DSA）从 `deepseek_v2/` 起步；有 DSA 看 `deepseek-v3.2`。

## 适配验证结论

标准三步闭环 + 第一轮直转全部通过（real weights `/mnt/data/models/deepseek-ai/DeepSeek-V2-Lite`，Wikitext PPL seq_len=4096，npu）：

| 步骤 | 结果 |
| --- | --- |
| BF16 baseline | **9.103** |
| 关闭量化等价 `attn-linear`（bits=16） | **9.103**（逐位=BF16）|
| 关闭量化等价 `moe`（bits=16） | **9.103**（逐位=BF16）|
| W8A8 `attn-linear+moe` 直转（无 PTQ） | **9.2374**，`delta=0.134`（≤0.2 达标）|

- 最小 PTQ smoke：extract/ptq 链路打通，autoround 受 env 缺 `triton` 阻断（非模型问题，直转不需要 triton）。
- 关闭量化等价两路逐位等于 BF16 → MLA attn wrapper（含非压缩 Q 分支）与 MoE wrapper 前向语义精确正确。

## 关键陷阱（L3 模型专属；通用见 L1/L2）

- **`q_lora_rank: null` 非压缩 Q（L3，V2-Lite 区别于 v3/v3.2）** —— 现象：直接套 v3.2 的 `QuantDeepseekV3Attention`（假定 `wq_a/wq_b` 压缩 Q）会缺 `q_proj` 路径。处理：wrapper 内 `if q_lora_rank is None: 包 q_proj else: 包 q_b_proj`，activation quant / forward 同步分支。教训：MLA 家族先查 `q_lora_rank` 是否为 None，决定 Q 路径。
- **`empty_weights_model` 必须实例化仓内 modeling，别走 trust_remote_code 动态加载（L3/L2）** —— 现象：`AutoModelForCausalLM.from_config(config, trust_remote_code=True)` 会去读**权重目录里的上游 `modeling_deepseek.py`**，而上游在新 transformers（5.3）下 `from ...import is_torch_fx_available` 直接 ImportError。处理：override `empty_weights_model` 用 `with init_empty_weights(): DeepseekV2ForCausalLM(self.config)`（仓内已 `AutoModelForCausalLM.register` 的类），与权重目录解耦。→ 见 [L2 · 接非 transformers 自定义 modeling](../family-pitfalls.md)（longcat-next 同理）。
- 通用坑引用：blockwise mask/pos、checkpoint 真实结构 → [L1](../common-pitfalls.md)；MoE expert 布局、per-expert 量化性能 → [L2 · MoE 类](../family-pitfalls.md)；MLA attn-linear/attn-cache 分路、cache 更敏感 → [L2 · MLA 类](../family-pitfalls.md)。

## 量化结论

- 首推 `attn-linear + moe` W8A8-int8 直转：`delta=0.134`（≤0.2，无需 PTQ）。若要更紧（如 W8A4 或更低 delta），再上 lac/lwc（需先补 `triton`）。
- attn-cache（KV）未单独测；MLA cache 位宽通常更敏感，量化它先 q/k 16-bit 起步。

## 适配建议（下次同系列/同结构）

- 先参考：本案 + `deepseek-v3.2`（有 DSA 时）。
- 先做什么：先验 `q_lora_rank`（决定 Q 路径）、`first_k_dense_replace`（layer0 dense 路由）、有无 `index_topk`（DSA）；`empty_weights_model` 用仓内 modeling。
- **不建议**：盲套 v3.2 的压缩-Q wrapper；依赖 `trust_remote_code` 让框架读权重目录上游 modeling（版本不兼容 + 不可移植）。

## 精度速查表

> ppl 口径 seq_len=4096。

| 数据类型 | 量化配置 | 量化算法 | ppl |
| --- | --- | --- | --- |
| BF16 | 无 | 无 | 9.103 |
| INT | A8W8: attn-linear（等价校验，bits=16） | 无 | 9.103 |
| INT | A8W8: moe（等价校验，bits=16） | 无 | 9.103 |
| INT | A8W8: attn-linear + moe（直转） | 无 | 9.2374 |
