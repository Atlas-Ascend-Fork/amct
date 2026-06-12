# Qwen3.6-35b

> 本模型暂无完整适配叙事，精度数据由 casebook 总表拆分而来；后续可补充结构判断 / 适配要点 / 复用建议。

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
