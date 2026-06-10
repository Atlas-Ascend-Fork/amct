# Deepseek-v3.1

> 本模型暂无完整适配叙事，精度数据由 casebook 总表拆分而来；后续可补充结构判断 / 适配要点 / 复用建议。

## 精度速查表

> MXFP 双值为两次评测口径 a/b。

| 数据类型 | 量化配置 | 量化算法 | mmlu | GPQA | math500 | humaneval+ |
| --- | --- | --- | --- | --- | --- | --- |
| BF16 | 无 |  | 87.8 | 77.44 | 95.06 | 92.68 |
| FP8 | MLA C8 静态 per tensor QKV 动态 per tensor | lac | 87.7 | 76.93 | 94.66 | 93.29 |
| FP8 | MLA C8 静态 per tensor Q per token KV 动态 per tensor | lac | 87.6 | 76.77 | 94.67 | 92.68 |
| MXFP4 | Decode C4 PerBlock 32x32 | lac | 87.6 | 76.26 | 94.67 | 91.46 |
| MXFP8 | Decode C8 PerBlock 32x32 | lac | 87.7 | 76.60 | 95.33 | 92.68 |
