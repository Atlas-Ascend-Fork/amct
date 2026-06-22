# DeepSeek-V3.2 · 系列 deepseek · 结构 mla-moe（DSA）

> **速览**：adapter `deepseek/deepseek_v3_2/{deepseekv3_2.py,quant_module.py}` = `DeepseekV32(BaseModel)`，MLA + MoE + DSA indexer，**moe-only**（不支持 `quant_target=mlp`）。是 DeepSeek 系列基类（`glm5` 继承它、`deepseekv4` 是更复杂下一代）。**适配实测（验证/陷阱）待补**。精度见下表。
> **触发信号**：config 有 `kv_lora_rank`（MLA）+ `num_experts`（MoE）+ `index_topk`（DSA indexer）。读 [L2 · MLA 类 + MoE 类](../structure-family-pitfalls.md) + [L1](../cross-model-pitfalls.md)。

## 结构与适配要点

- `DeepseekV32(BaseModel)`：MLA attention + MoE，含 **DSA indexer**（`QuantIndexer`）；`quant_target` 走 `moe` / attn 路径，**不支持 `mlp`**。
- **复用**：`BaseModel` blockwise、`apply_quant_to_attn` / `apply_quant_to_moe_mlp`、`QuantGatedMLP`。**新增**：`QuantDeepseekV3Attention`、`QuantDeepseekV3MLP(QuantGatedMLP)`、`QuantIndexer`（DSA）。
- **参考与差异**：本案是 DeepSeek 系列基类——`glm5` 继承它、`deepseekv4` 是更复杂的下一代（MLA + Compressor + Indexer + Hyper-Connections）。
- **起步复用清单**：下一条 MLA+MoE(+DSA) 从 `deepseekv3_2.py`、`quant_module.py` 起步。

## 适配验证结论 / 关键陷阱

- **待补**：实测记录与踩坑待补；DSA indexer 的部署门控见 deploy 侧 `hasattr(config,index_topk)`（deploy↔infer schema 字段级互锁）。

## 精度速查表

> ppl 口径 seq_len=4096。 MXFP 双值为两次评测口径 a/b。

| 数据类型 | 量化配置 | 量化算法 | ppl | mmlu | GPQA | math500 | aime24 | liveCodeBench | gsm8k | drop | mmlu_pro |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BF16 | 无 |  | 3.2133 | 90.8 | 75.85 | 93.2 | 68.96 | 50.37 | 96.44 | 87.57 | 80.94 |
| INT | MLA_A8W8C8 MOE_A8W8 | lac lwc | 3.2666 | 90.62 | 75.9 | 94.4 | 64.17 | 50 | 96.36 | 87.41 | 81.42 |
| INT | MLA_A8W8C8 MOE_A8W4 | lac lwc | 3.2264 | 90.66 | 76.56 | 93.2 | 63.75 | 49.73 | 96.13 | 87.68 | 80.8 |
| INT | MLA_A8W8CMXFP4 MOE_A8W8 | lac lwc | 3.2757 | 90.69 | 75.35 | 93.4 | 62.86 | 50 | 96.44 | 87.64 | 81.32 |
