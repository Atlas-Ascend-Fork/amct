# Casebook

`casebook/` 用来沉淀模型适配和量化过程中的可复用经验。

## 记录内容

- 模型适配路径
- 量化起步方案与结果
- 典型问题与排查结论
- 成功路径和失败路径

## 组织方式

**主轴 = 网络结构**（量化策略由结构决定：dense→mlp/attn-linear；MoE→moe.routed/shared；MLA→attn-cache；VL→language_model scope）。厂商专属适配经验单独收在 `_vendor-notes/`。

```text
casebook/
  README.md
  dense/          # 标准 dense decoder（attn-linear + mlp）
  moe/            # 标准 MoE（FFN experts，routed/shared）
  mla-moe/        # MLA attention + MoE（deepseek 式）
  vl-moe/         # 多模态 VL + MoE（language_model + visual + mtp）
  _vendor-notes/  # 厂商系列适配经验（adapter + 系列量化经验）
```

### 个案导航（按结构）

- **dense/**：`qwen3-4b`、`qwen3-8b`、`pangu-7b`
- **moe/**：`qwen3-30b-a3b`、`qwen3-235b-a22b`、`qwen3-next-80b-a3b-instruct`、`longcat-flash-lite`、`longcat-next`
- **mla-moe/**：`deepseekv4`
- **vl-moe/**：`qwen3.5-35b-a3b`
- **_vendor-notes/**：`qwen`、`deepseek`、`glm`、`longcat`、`pangu`

## 维护规则

- 新结果与同结构已有经验一致时，优先更新对应结构目录下的个案；厂商通用经验更新 `_vendor-notes/<vendor>.md`
- 出现新的结构模式、量化路径或高价值问题时，再新增 `<structure>/<case>.md`
- 只记录可复用结论，不记录一次性的临时过程

## 当前精度总表

> 这是跨模型/跨方案的**全量精度速查表**（含部分尚无个案的模型，作快速横比用）；按网络结构找具体个案见上方「个案导航」。Diffusion 视频模型在 LLM 表之后单独成表。

可以先把当前已经拿到的稳定精度汇总在这里，方便快速查阅和横向对比。
当某个模型的结论稳定下来后，再同步到对应系列的 `README.md` 或个案文件里。

| 结构 | 网络                 | 数据类型 | 量化配置                                              | 量化算法        | ppl            | mmlu | GPQA  | math500 | aime24 | humaneval+ | liveCodeBench | gsm8k | drop | mmlu_pro |
| --- |--------------------|------|---------------------------------------------------|-------------|----------------|------|-------|---------|--------|-------|-------|-------|-------|-------|
| dense | Pangu-7B           | BF16 | 无                                                 | 无           | 12.716         |  |       |     |      |  
| dense |                    | INT  | A4w4: mlp, attn-linear                            | 无           | 42268.0117     |      |       |     |      |
| dense |                    | INT  | A8w4: mlp, attn-linear                            | 无           | 20.853         |      |       |     |      |
| dense |                    | INT  | A8w8: mlp, attn-linear                            | 无           | 13.23          |      |       |     |      |
| dense |                    | MXFP  | A4w4: mlp, attn-linear                            | 无           | 20.3654/17.69  |      |       |     |      |
| dense |                    | MXFP  | A8w4: mlp, attn-linear                            | 无           | 15.1952/15.41  |      |       |     |      |
| dense |                    | MXFP  | A8w8: mlp, attn-linear                            | 无           | 12.935/13.093  |      |       |     |      |
| dense |                    | HiFP  | A4w4: mlp, attn-linear                            | 无           | 14.444         |      |       |     |      |
| dense |                    | HiFP  | A8w4: mlp, attn-linear                            | 无           | 13.7646        |      |       |     |      |
| dense |                    | HiFP  | A8w8: mlp, attn-linear                            | 无           | 13.542         |      |       |     |      |
| dense | Qwen3.5-4b         | BF16 | 无                                                 | 无           | 8.8445         |  |       |     |      |  
| dense |                    | INT  | A4w4: mlp, attn-linear                            | 无           | 4836.4731      |  |       |     |      | 
| dense |                    | INT  | A8w4: mlp, attn-linear                            | 无           | 12.4741        |  |       |     |      | 
| dense |                    | INT  | A8w8: mlp, attn-linear                            | 无           | 9.022          |  |       |     |      | 
| dense |                    | MXFP  | A4w4: mlp, attn-linear                            | 无           | 11.08/10.357   |  |       |     |      | 
| dense |                    | MXFP  | A8w4: mlp, attn-linear                            | 无           | 9.5715/9.255   |  |       |     |      | 
| dense |                    | MXFP  | A8w8: mlp, attn-linear                            | 无           | 8.972/8.879    |  |       |     |      | 
| dense |                    | HiFP  | A4w4: mlp, attn-linear                            | 无           | 9.84           |  |       |     |      | 
| dense |                    | HiFP  | A8w4: mlp, attn-linear                            | 无           | 9.5166         |  |       |     |      | 
| dense |                    | HiFP  | A8w8: mlp, attn-linear                            | 无           | 9.0552         |  |       |     |      | 
| dense | Qwen3-4b           | BF16 | 无                                                 | 无           | 12.377         |  |       |     |      |  
| dense |                    | INT  | A4w4: mlp, attn-linear                            | 无           | 9942.9922      |  |       |     |      | 
| dense |                    | INT  | A8w4: mlp, attn-linear                            | 无           | 19.9143        |  |       |     |      | 
| dense |                    | INT  | A8w8: mlp, attn-linear                            | 无           | 12.4327        |  |       |     |      | 
| dense |                    | MXFP  | A4w4: mlp, attn-linear                            | 无           | 16.043/14.1    |  |       |     |      | 
| dense |                    | MXFP  | A8w4: mlp, attn-linear                            | 无           | 13.808/13.066  |  |       |     |      | 
| dense |                    | MXFP  | A8w8: mlp, attn-linear                            | 无           | 12.6976/12.365 |  |       |     |      | 
| dense |                    | HiFP  | A4w4: mlp, attn-linear                            | 无           |                |  |       |     |      | 
| dense |                    | HiFP  | A8w4: mlp, attn-linear                            | 无           |                |  |       |     |      | 
| dense |                    | HiFP  | A8w8: mlp, attn-linear                            | 无           |                |  |       |     |      | 
| moe | Qwen3-235B-A22B    | BF16 | 无                                                 | 无           | 5.09           |      |       |     |      |       |       |
| moe |                    | INT  | A4w4: moe, attn-linear                            | 无           | 46.3021        |      |       |     |      |
| moe |                    | INT  | A8w4: moe, attn-linear                            | 无           | 5.71           |      |       |     |      |
| moe |                    | INT  | A8w8: moe, attn-linear                            | 无           | 5.1427         |      |       |     |      |
| moe |                    | MXFP | MoE Expert A8W8                                   | 无           | 5.115          |      |       |     |      |       |       |
| moe |                    |      | Attn A8W8                                         | 无           | 5.12           |      |       |     |      |       |       |
| moe |                    |      | MoE Expert A4W4                                   | 无           | 5.52           |      |       |     |      |       |       |
| moe |                    |      | Attn A4W4                                         | 无           | 5.40           |      |       |     |      |       |       |
| moe |                    | MXFP | A4w4: moe, attn-linear                            | 无           | 5.82/5.656     |      |       |     |      |
| moe |                    | MXFP | A8w4: moe, attn-linear                            | 无           | 5.417/5.368    |      |       |     |      |
| moe |                    | MXFP | A8w8: moe, attn-linear                            | 无           | 5.1267/5.135   |      |       |     |      |
| moe |                    | HiFP | A4w4: moe, attn-linear                            | 无           | 5.3359         |      |       |     |      |
| moe |                    | HiFP | A8w4: moe, attn-linear                            | 无           |                |      |       |     |      |
| moe |                    | HiFP | A8w8: moe, attn-linear                            | 无           |                |      |       |     |      |
| moe | GLM-5.1            | BF16 | 无                                                 | 无           | 2.5658         |      |       |     |      |  
| moe |                    | INT  | MOE A8W8                                          | 无           | 2.5725         |      |       |     |      |  
| moe |                    |      | MOE A8W4                                          | 无           | 2.8266         |      |       |     |      |  
| moe | Qwen3-Next         | BF16 | 无                                                 | 无           | 5.4483         | 85.4 |       |     |      |  
| moe |                    | INT  | A4w4: moe, attn-linear                            | 无           | 561.4258       |      |       |     |      |
| moe |                    | INT  | A8w4: moe, attn-linear                            | 无           | 6.0114         |      |       |     |      |
| moe |                    | INT  | A8w8: moe, attn-linear                            | 无           |                |      |       |     |      |
| moe |                    | INT  | MOE A8W8                                          | 无           |                | 85.5 |       |     |      | 
| moe |                    |      | MOE A8W4                                          | 无           |                |      |       |     |      |
| moe |                    |      | MOE A8W4                                          | LWC         |                |      |       |     |      |
| moe |                    | MXFP  | A4w4: moe, attn-linear                            | 无           | 6.123/5.901    |      |       |     |      |
| moe |                    | MXFP  | A8w4: moe, attn-linear                            | 无           | 5.7337/5.609   |      |       |     |      |
| moe |                    | MXFP  | A8w8: moe, attn-linear                            | 无           | 5.451          |      |       |     |      |
| moe |                    | HiFP  | A4w4: moe, attn-linear                            | 无           | 5.6854         |      |       |     |      |
| moe |                    | HiFP  | A8w4: moe, attn-linear                            | 无           |                |      |       |     |      |
| moe |                    | HiFP  | A8w8: moe, attn-linear                            | 无           |                |      |       |     |      |
| mla-moe | Deepseekv2-lite    | BF16 | 无                                                 | 无           | 6.31           | 54.2 |       |         |        |       |       |
| mla-moe |                    | INT  | 全网 w4                                             | RTN         | 6.82           |      |       |         |        |       |       |
| mla-moe |                    |      | moe experts w4                                    | RTN         | 6.39           |      |       |         |        |       |       |
| mla-moe |                    |      | moe experts a4w4                                  | 标准 hadamard | 6.5996         |      |       |         |        |       |       |
| mla-moe |                    |      | moe experts a4w4                                  | flatquant   | 6.5504         |      |       |         |        |       |       |
| mla-moe |                    |      | attn a8w8                                         | RTN         | 6.3729         |      |       |         |        |       |       |
| mla-moe |                    | MXFP | moe experts a4w4                                  | RTN         | 6.8387         |      |       |         |        |       |       |
| mla-moe | Deepseek-R1        | BF16 | 无                                                 | 无           | 3.3303         | 87.1 | 72.73 | 97.6    | 80     |       |       |
| mla-moe |                    | INT  | 全网 w4                                             | RTN         | 3.86           |      |       |         |        |       |       |
| mla-moe |                    |      | 全网 A8W8C8: 静态per tensor C8                        | lac         | 3.37           | 87.0 | 71.7  | 97      | 73.3   |       |       |
| mla-moe |                    |      | moe experts a4w4                                  | flatquant   | 3.64           | 86.3 |       | 95      | 70.0   |       |       |
| mla-moe | Deepseek-v3.1      | BF16 | 无                                                 |             |                | 87.8 | 77.44 | 95.06 | | 92.68|
| mla-moe |                    | FP8  | MLA C8 静态 per tensor QKV 动态 per tensor            | lac         |                | 87.7 | 76.93 | 94.66 | | 93.29 |
| mla-moe |                    | FP8  | MLA C8 静态 per tensor Q per token KV 动态 per tensor | lac         |                | 87.6 | 76.77 | 94.67 | | 92.68 |
| mla-moe |                    | MXFP4  | Decode C4 PerBlock 32x32                           | lac         |            | 87.6 | 76.26 | 94.67 | | 91.46 |
| mla-moe |                    | MXFP8  | Decode C8 PerBlock 32x32                           | lac         |            | 87.7 | 76.60 | 95.33 | | 92.68 |
| mla-moe | Deepseek-v3.2      | BF16 | 无                                                 |             | 3.2133         | 90.8  | 75.85 | 93.2 | 68.96 | | 50.37 | 96.44 | 87.57 | 80.94 | 
| mla-moe |                    | INT  | MLA_A8W8C8 MOE_A8W8                               | lac lwc     | 3.2666         | 90.62 | 75.9  | 94.4 | 64.17 | | 50    | 96.36 | 87.41 | 81.42 |
| mla-moe |                    | INT  | MLA_A8W8C8 MOE_A8W4                               | lac lwc     | 3.2264         | 90.66 | 76.56 | 93.2 | 63.75 | | 49.73 | 96.13 | 87.68 | 80.8  |
| mla-moe |                    | INT  | MLA_A8W8CMXFP4 MOE_A8W8                           | lac lwc     | 3.2757         | 90.69 | 75.35 | 93.4 | 62.86 | | 50    | 96.44 | 87.64 | 81.32 |  
| mla-moe | longcat-flash      | BF16 | 无                                                 | 无           | 3.1214         | 74.49 | 57.07 | 92.4 | | 82.93 |
| mla-moe |                    | INT  | MOE+MLPs A8W8                                     | 无           | 3.1746         |       |       |     |      |
| mla-moe |                    | MXFP | MLA_A8W8CFP8 MOE_A8W4                             | 无           | 3.291          | 75.4  | 60.1  | 91   | | 82.32 |
| mla-moe | longcat-flash-lite | BF16 | 无                                                 | 无           | 14.6133        |      |       |     |      |  
| mla-moe |                    | INT  | A4w4: mlp, moe, attn-linear                       | 无           | 1053.832       |      |       |     |      |
| mla-moe |                    | INT  | A8w4: mlp, moe, attn-linear                       | 无           | 13.1681        |      |       |     |      |
| mla-moe |                    | INT  | A8w8: mlp, moe, attn-linear                       | 无           | 14.0334        |      |       |     |      |
| mla-moe |                    | MXFP | A4w4: mlp, moe, attn-linear                       | 无           | 13.2715/12.283 |      |       |     |      |
| mla-moe |                    | MXFP | A8w4: mlp, moe, attn-linear                       | 无           | 10.4672/10.061 |      |       |     |      |
| mla-moe |                    | MXFP | A8w8: mlp, moe, attn-linear                       | 无           | 14.8785/15.091 |      |       |     |      |
| mla-moe |                    | HiFP | A4w4: mlp, moe, attn-linear                       | 无           | 14.258         |      |       |     |      |
| mla-moe |                    | HiFP | A8w4: mlp, moe, attn-linear                       | 无           |                |      |       |     |      |
| mla-moe |                    | HiFP | A8w8: mlp, moe, attn-linear                       | 无           |                |      |       |     |      |
| mla-moe | longcat-Next       | BF16 | 无                                                 | 无           | 17.518         |      |       |     |      |  
| mla-moe |                    | INT  | A4w4: mlp, moe, attn-linear                       | 无           | 3302.094       |      |       |     |      |
| mla-moe |                    | INT  | A8w4: mlp, moe, attn-linear                       | 无           | 24.8581        |      |       |     |      |
| mla-moe |                    | INT  | A8w8: mlp, moe, attn-linear                       | 无           | 15.718         |      |       |     |      |
| mla-moe |                    | MXFP  | A4w4: mlp, moe, attn-linear                       | 无           | 19.0125/18.266 |      |       |     |      |
| mla-moe |                    | MXFP  | A8w4: mlp, moe, attn-linear                       | 无           | 17.5067/16.899 |      |       |     |      |
| mla-moe |                    | MXFP  | A8w8: mlp, moe, attn-linear                       | 无           | 18.3902/19.2   |      |       |     |      |
| mla-moe |                    | HiFP  | A4w4: mlp, moe, attn-linear                       | 无           | 16.9826        |      |       |     |      |
| mla-moe |                    | HiFP  | A8w4: mlp, moe, attn-linear                       | 无           |                |      |       |     |      |
| mla-moe |                    | HiFP  | A8w8: mlp, moe, attn-linear                       | 无           |                |      |       |     |      |
| mla-moe | Kimi-K2 Thinking    | BF16 | 无                                                 | 无           | 2.1896         |      |       |     |      |       |       |
| mla-moe |                    | INT  | A4W4 PerChannel: moe                           | 无           | 2.5487         |  |       |     |      | 
| mla-moe |                    | INT  | A4W4 PerToken-PerChannel: moe                            | 无           | 2.3337         |  |       |     |      | 
| vl-moe | Qwen3.5-35b        | BF16 | 无                                                 | 无           | 6.2486         |  |       |     |      |  
| vl-moe |                    | INT  | A4w4: moe, attn-linear                            | 无           | 3839.2734      |  |       |     |      | 
| vl-moe |                    | INT  | A8w4: moe, attn-linear                            | 无           | 7.1928         |  |       |     |      | 
| vl-moe |                    | INT  | A8w8: moe, attn-linear                            | 无           | 6.3781         |  |       |     |      | 
| vl-moe |                    | MXFP  | A4w4: moe, attn-linear                            | 无           | 6.9814/6.826   |  |       |     |      | 
| vl-moe |                    | MXFP  | A8w4: moe, attn-linear                            | 无           | 6.508/6.436    |  |       |     |      | 
| vl-moe |                    | MXFP  | A8w8: moe, attn-linear                            | 无           | 6.3168/6.285   |  |       |     |      | 
| vl-moe |                    | HiFP  | A4w4: moe, attn-linear                            | 无           | 6.617          |  |       |     |      | 
| vl-moe |                    | HiFP  | A8w4: moe, attn-linear                            | 无           |                |  |       |     |      | 
| vl-moe |                    | HiFP  | A8w8: moe, attn-linear                            | 无           |                |  |       |     |      | 
| vl-moe | Qwen3.6-35b        | BF16 | 无                                                 | 无           | 6.2799         |  |       |     |      |  
| vl-moe |                    | INT  | A4w4: moe, attn-linear                            | 无           | 3962.8643      |  |       |     |      | 
| vl-moe |                    | INT  | A8w4: moe, attn-linear                            | 无           | 7.3586         |  |       |     |      | 
| vl-moe |                    | INT  | A8w8: moe, attn-linear                            | 无           | 6.3788         |  |       |     |      | 
| vl-moe |                    | MXFP  | A4w4: moe, attn-linear                            | 无           | 7.024/6.931    |  |       |     |      | 
| vl-moe |                    | MXFP  | A8w4: moe, attn-linear                            | 无           | 6.599/6.536    |  |       |     |      | 
| vl-moe |                    | MXFP  | A8w8: moe, attn-linear                            | 无           | 6.3168/6.308   |  |       |     |      | 
| vl-moe |                    | HiFP  | A4w4: moe, attn-linear                            | 无           |                |  |       |     |      | 
| vl-moe |                    | HiFP  | A8w4: moe, attn-linear                            | 无           |                |  |       |     |      | 
| vl-moe |                    | HiFP  | A8w8: moe, attn-linear                            | 无           |                |  |       |     |      | 

Deepseekv2-lite, Deepseek-R1, Deepseek-v3.1, longcat-flash测试ppl的seq length=2048，后续可以刷新，其余均4096
注：longcat-lite和longcat-next的seq length从2048-> 4096精度还变差


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

## 当前结构分类

按网络结构维护（量化方案按结构复用）：

- `dense/`、`moe/`、`mla-moe/`、`vl-moe/`
- 厂商系列适配经验：`_vendor-notes/`
