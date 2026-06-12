# Deepseekv2-lite

> 本模型暂无完整适配叙事，精度数据由 casebook 总表拆分而来；后续可补充结构判断 / 适配要点 / 复用建议。

## 精度速查表

> ppl 口径 seq_len=2048（其余模型默认 4096）。 MXFP 双值为两次评测口径 a/b。

| 数据类型 | 量化配置 | 量化算法 | ppl | mmlu |
| --- | --- | --- | --- | --- |
| BF16 | 无 | 无 | 6.31 | 54.2 |
| INT | 全网 w4 | RTN | 6.82 |  |
| INT | moe experts w4 | RTN | 6.39 |  |
| INT | moe experts a4w4 | 标准 hadamard | 6.5996 |  |
| INT | moe experts a4w4 | flatquant | 6.5504 |  |
| INT | attn a8w8 | RTN | 6.3729 |  |
| MXFP | moe experts a4w4 | RTN | 6.8387 |  |
