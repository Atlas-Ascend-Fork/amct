# Qwen3.6-27B A4W4 MXFP4 Attn-Linear · 系列 qwen · 结构 dense

> **速览**：Qwen3.6-27B Dense 复用 `qwen3_5` adapter，对 64 层的
> `attn-linear` 执行 A4W4 MXFP4 blockwise 量化。Direct、LWC、LAC、
> LWC+LAC、OmniQuant、AutoRound 和 FlatQuant 均完成 PPL、Deploy 与
> clean reload 验证。
> **触发信号**：`model_type=qwen3_5`、
> `architectures=Qwen3_5ForConditionalGeneration`，实际权重为
> Qwen3.6-27B Dense。通用坑见 [L1](../cross-model-pitfalls.md)。

## 1. 实践配置

| 配置项 | 取值 |
| --- | --- |
| 模型 | Qwen3.6-27B Dense（64 层） |
| 位宽与格式 | A4W4 MXFP4 |
| 量化目标 | `attn-linear` |
| 量化粒度 | block |
| Block 范围 | `[0, 64)` |
| Bit 配置 | `amct_pytorch/configs/w4a4.yaml` |
| 校准数据 | PileVal，16 个样本，`seq_len=1024` |
| 精度数据 | WikiText2 test，PPL |
| AMCT 适配名 | `qwen3_5` |

校准数据提取覆盖 64 个 Block，耗时 4003 秒，退出码为 0；七种方案均
生成 64 个 Block 参数。耗时仅作本次工程记录，不作为算法速度排名。

## 2. 结构与适配要点

- 本模型沿用 `qwen3_5` 的 Dense 主链，无需为 Qwen3.6-27B 新增独立
  adapter；模型名称应以实际加载的 Qwen3.6-27B 权重为准。
- 量化目标固定为 `attn-linear`，Eval 和 Deploy 使用
  `--attn_linear_param_dir` 读取对应 PTQ 参数。

## 3. 量化与精度结论

- BF16 baseline 为 `7.782411575317383`；Direct 相对 BF16 增加
  `0.267979`。
- LWC、LAC、LWC+LAC、OmniQuant 和 FlatQuant 均相对 Direct 恢复部分
  PPL；AutoRound 在本次口径下高于 Direct。
- LAC、LWC+LAC 和 FlatQuant 低于 BF16，只表示本次 WikiText2 配置下的
  实测结果，不应解释为量化普遍提升模型质量。
- LAC 和 FlatQuant 独立复测与正式 PPL 结果一致。

## 4. 精度速查表

> PPL 数据集为 WikiText2 test。Eval 为单卡严格串行复测耗时；PTQ 为历史
> 正式运行耗时，未在统一条件下串行复测。

| 方法 | PPL | 相对 BF16 | 相对 Direct 恢复量 | Eval/s | PTQ/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| BF16 | 7.782411575317383 | — | — | 1807 | — |
| Direct | 8.050390243530273 | +0.267979 | 0.000000 | 2164 | 786 |
| LWC | 7.966102123260498 | +0.183691 | +0.084288 | 2021 | 969 |
| LAC | 7.583184719085693 | -0.199227 | +0.467206 | 1975 | 918 |
| LWC+LAC | 7.579893112182617 | -0.202518 | +0.470497 | 1948 | 1196 |
| OmniQuant | 7.799318790435791 | +0.016907 | +0.251071 | 2008 | 991 |
| AutoRound | 8.158552169799805 | +0.376141 | -0.108162 | 1961 | 2583 |
| FlatQuant | 7.207714557647705 | -0.574697 | +0.842676 | 2013 | 1158 |

## 5. Deploy 与 clean reload

| 方法 | Deploy/s | 文件数 | 导出大小 | clean reload |
| --- | ---: | ---: | ---: | --- |
| Direct | 1418 | 83 | 约 42G | PASS |
| LWC | 1399 | 83 | 约 42G | PASS |
| LAC | 1379 | 83 | 约 42G | PASS |
| LWC+LAC | 1383 | 83 | 约 42G | PASS |
| OmniQuant | 1474 | 83 | 约 42G | PASS |
| AutoRound | 1308 | 83 | 约 42G | PASS |
| FlatQuant | 1372 | 83 | 约 42G | PASS |

七种 Deploy 均以退出码 0 完成。clean reload 均无缺失键、非预期键、形状
不匹配或加载错误；共识别 304 个量化模块，第 0 层识别 5 个，Vision 分支
和 `lm_head` 未被错误量化。

## 6. 已知问题与复用建议

- Direct 是不启用额外算法的基线，调用时不传 `--algos`。
- clean reload 需用 BF16 初始化模型骨架和未量化组件；量化配置中不兼容的
  `activation_use_clip` 仅从内存副本移除，不改 Deploy 产物。
- MXFP block size 为 32；Vision、Visual、Image、`lm_head` 及输入维度不
  满足整除约束的 Linear 不属于本次文本量化目标，应从 reload 匹配中排除。
- FLA 与部分 FlatQuant 算子可能回退到 Torch/CPU，影响端到端 Eval 耗时，
  但本次未影响正式 PPL、Deploy 或 clean reload。
- Eval 时间不是在线推理吞吐或延迟；性能比较应保持模型、数据、设备和并发
  条件一致，避免多任务竞争污染结果。
