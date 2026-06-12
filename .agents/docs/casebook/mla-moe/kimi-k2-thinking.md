# Kimi-K2 Thinking

> 本模型暂无完整适配叙事，精度数据由 casebook 总表拆分而来；后续可补充结构判断 / 适配要点 / 复用建议。

## 精度速查表

> ppl 口径 seq_len=4096。 MXFP 双值为两次评测口径 a/b。

| 数据类型 | 量化配置 | 量化算法 | ppl |
| --- | --- | --- | --- |
| BF16 | 无 | 无 | 2.1896 |
| INT | A4W4 PerChannel: moe | 无 | 2.5487 |
| INT | A4W4 PerToken-PerChannel: moe | 无 | 2.3337 |
