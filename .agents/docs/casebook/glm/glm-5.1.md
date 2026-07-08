# GLM-5.1 · 系列 glm · 结构 mla-moe（继承 DeepseekV32）

> **速览**：adapter `glm/glm5/glm5.py` = `GLM5(DeepseekV32)`，结构复用 DeepSeek V3.2 的 MLA+MoE 适配；**结构通用坑见 `deepseek/deepseek-v3.2` + L2 · MLA 类**，本案 L3 仅 = "不含 DSA indexer"。适配实测（验证/陷阱）待补。精度见下表。
> **触发信号**：`GLM5(DeepseekV32)`，config 有 `kv_lora_rank`（MLA）+ `num_experts`（MoE），**无 `index_topk`**（不走 DSA）。读 [L2 · MLA 类 + MoE 类](../structure-family-pitfalls.md) + [L1](../cross-model-pitfalls.md)。

## 结构与适配要点

- `GLM5(DeepseekV32)` —— **直接继承 DeepSeek V3.2 adapter**：MLA attention + MoE，量化复用 `QuantDeepseekV3Attention` / `QuantDeepseekV3MLP` / `apply_quant_to_attn` / `apply_quant_to_moe_mlp`，几乎无模型专属新增代码。
- **参考与差异（即本案 L3）**：最近参考 `deepseek-v3.2`（同 `DeepseekV32` 基类）；唯一差异：GLM5 适配器**不含 DSA indexer 代码**（indexer 为 DSv3.2 专属；`llm_deploy` 以 `hasattr(config,index_topk)` 门控 li_cache，glm5 不走该路径，**非缺陷**；deploy↔infer schema 互锁细节见 DeepSeek 系列与 deploy 侧分析）。
- **起步复用清单**：下一条 GLM 从 `glm5.py`（即 DeepseekV32 路径）+ `deepseek_v3_2/quant_module.py` 起步。

## 适配验证结论 / 关键陷阱

- **待补**：源码已适配，BF16 baseline / 关闭量化等价 / 最小 PTQ smoke 的实测记录与踩坑待补；MLA/MoE 通用陷阱见 DeepSeek 系列 README。

## 精度速查表

> ppl 口径 seq_len=4096。 MXFP 双值为两次评测口径 a/b。

| 数据类型 | 量化配置 | 量化算法 | ppl |
| --- | --- | --- | --- |
| BF16 | 无 | 无 | 2.5658 |
| INT | MOE A8W8 | 无 | 2.5725 |
| INT | MOE A8W4 | 无 | 2.8266 |
