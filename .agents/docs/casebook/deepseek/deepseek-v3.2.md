# DeepSeek-V3.2 · 系列 deepseek · 结构 mla-moe（DSA）

> **速览**：`deepseek_v3_2` 已适配（源码就绪），**适配经验待补**——参考 `deepseekv4`（MLA/Compressor/Indexer 设计）。精度见下表。

## 精度速查表

> ppl 口径 seq_len=4096。 MXFP 双值为两次评测口径 a/b。

| 数据类型 | 量化配置 | 量化算法 | ppl | mmlu | GPQA | math500 | aime24 | liveCodeBench | gsm8k | drop | mmlu_pro |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BF16 | 无 |  | 3.2133 | 90.8 | 75.85 | 93.2 | 68.96 | 50.37 | 96.44 | 87.57 | 80.94 |
| INT | MLA_A8W8C8 MOE_A8W8 | lac lwc | 3.2666 | 90.62 | 75.9 | 94.4 | 64.17 | 50 | 96.36 | 87.41 | 81.42 |
| INT | MLA_A8W8C8 MOE_A8W4 | lac lwc | 3.2264 | 90.66 | 76.56 | 93.2 | 63.75 | 49.73 | 96.13 | 87.68 | 80.8 |
| INT | MLA_A8W8CMXFP4 MOE_A8W8 | lac lwc | 3.2757 | 90.69 | 75.35 | 93.4 | 62.86 | 50 | 96.44 | 87.64 | 81.32 |
