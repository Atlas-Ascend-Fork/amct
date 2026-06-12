# Diffusion 视频模型

> 视频扩散模型 attention 量化精度；指标体系与 LLM 不同（subject/motion/dynamic/aesthetic/imaging/overall consistency）。

| Difussion网络 | 视频规格 | 数据类型 | 量化策略 | 量化算法 | subject consistency | motion smoothness | dynamic quality | aesthetic quality | imaging quality | overall consistency |
|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|
| HunyuanVideo | 720x1280x129 | bf16 | - | - | 95.37 | 99.36 | 53.33 | 60.25 | 68.22 | 26.62 |
|              |              | fp8  | Q Block_128 K/V Block_256 P 静态PerTensor softmax_fp32 | - | 95.48 | 99.36 | 53.33 | 60.3 | 67.85 | 26.54 |
|              | 540x960x129  | bf16 | - | - | 95.67 | 99.28 | 58.33 | 61.93 | 65.47 | 26.98 |
|              |              | fp8  | Q Block_128 K/V Block_256 P cast直转 softmax_fp16 | - | 95.77 | 99.3 | 56.66 | 61.91 | 65.29 | 26.94 |
|              |              | fp8  | Q Block_128 K/V Block_256 P 静态PerTensor softmax_fp16 | - | 95.51 | 99.26 | 59.16 | 62.35 | 65.96 | 26.89 |
|              | 320x480x65   | bf16 | - | - | 94.86 | 98.8 | 67.5 | 57.18 | 62.08 | 27.18 |
|              |              | mxfp4/fp8  | Q/K mxfp4 P per_tensor V FP8 block_256      | -           | 93.86 | 98.63 | 74.16 | 55.6 | 58.05 | 26.77 |
|              |              | mxfp4/fp8  | Q/K mxfp4 P per_tensor V FP8 block_256      | RHT 二级量化 | 94.78 | 98.85 | 66.66 | 56.92 | 61.19 | 26.71 |
|              |              | mxfp4/fp8  | ACE + Q/K mxfp4 P per_token V FP8 block_256 | RHT 二级量化 | 94.9  | 98.85 | 66.66 | 57.06 | 61.11 | 26.87 |
| Wan2.2-t2v | 240x432x61 | bf16  | -          | -         | 91.88 | 97.88 | 90.83 | 58.09 | 65    | 26.18 |
|            |            | hif4  | QKPV hif4  | RHT       | 90.96 | 97.95 | 91.66 | 56.53 | 64.11 | 25.68 |
|            |            | hif4  | QKPV hif4  | RHT P two | 91.30 | 97.80 | 93.33 | 57.38 | 64.59 | 25.98 |
|            |            | mxfp4 | QKPV mxfp4 | RHT P two | 87.65 | 97.82 | 93    | 52.99 | 59.99 | 24.68 |
|            |            | nvf4  | QKPV nvf4  | RHT       | 91.32 | 97.83 | 91.66 | 56.31 | 63.62 | 26.18 |
|            |            | nvf4  | QKPV nvf4  | RHT two   | 90.91 | 97.81 | 90.83 | 56.73 | 63.17 | 25.87 |
