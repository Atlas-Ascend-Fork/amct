# Qwen3.6-35B · 系列 qwen · 结构 moe（混合 attention）

> **速览**：adapter `qwen/qwen3_6/qwen3_6_moe.py` = `Qwen3_6Moe(Qwen3_5Moe)`，**近乎纯复用** Qwen3.5-MoE（混合 `linear`+`full` attention + packed MoE）。**适配经验（验证/陷阱）待补**——参考 `qwen3.5-35b-a3b`。精度见下表。

## 结构与适配要点

- `Qwen3_6Moe(Qwen3_5Moe)` —— **直接继承 Qwen3.5-MoE adapter**，结构同：block 内混合 `linear_attention` + `full_attention`、checkpoint MoE 已 packed；当前近乎子类化复用、无显著模型专属新增。
- **参考与差异**：最近参考 `qwen3.5-35b-a3b`（同 `Qwen3_5Moe` 基类）；差异：暂无显著结构差异。
- **起步复用清单**：下一条 Qwen3.5/3.6 MoE 从 `qwen3_5_moe.py` 起步。

## 适配验证结论 / 关键陷阱

- **待补**：实测记录与踩坑待补；混合 attention 的 causal mask 路由陷阱见 `qwen3-next` 与 Qwen 系列 README。

## 精度速查表

> ppl 口径 seq_len=4096。 MXFP 双值为两次评测口径 a/b。

| 数据类型 | 量化配置 | 量化算法 | ppl |
| --- | --- | --- | --- |
| BF16 | 无 | 无 | 6.2799 |
| INT | A4w4: moe, attn-linear | 无 | 3962.8643 |
| INT | A8w4: moe, attn-linear | 无 | 7.3586 |
| INT | A8w8: moe, attn-linear | 无 | 6.3788 |
| MXFP | A4w4: moe, attn-linear | 无 | 7.024/6.931 |
| MXFP | A8w4: moe, attn-linear | 无 | 6.599/6.536 |
| MXFP | A8w8: moe, attn-linear | 无 | 6.3168/6.308 |
| HiFP | A4w4: moe, attn-linear | 无 |  |
| HiFP | A8w4: moe, attn-linear | 无 |  |
| HiFP | A8w8: moe, attn-linear | 无 |  |
