# Qwen3.6-27B A4W4 MXFP4 MLP · 系列 qwen · 结构 dense

> **速览**：Qwen3.6-27B Dense 复用 `qwen3_5` adapter，对 64 层 MLP
> 执行 A4W4 MXFP4 blockwise 量化。Direct、LWC、LAC、LWC+LAC、
> OmniQuant、AutoRound 和 FlatQuant 均完成 PPL、Deploy 与 clean reload
> 验证。
> **触发信号**：`model_type=qwen3_5`、
> `architectures=Qwen3_5ForConditionalGeneration`，实际权重为
> Qwen3.6-27B Dense。通用坑见 [L1](../cross-model-pitfalls.md)。

## 1. 实践配置

| 配置项 | 取值 |
| --- | --- |
| 模型 | Qwen3.6-27B Dense（64 层） |
| 位宽与格式 | A4W4 MXFP4 |
| 量化目标 | `mlp` |
| 量化粒度 | block |
| Block 范围 | `[0, 64)` |
| Bit 配置 | `amct_pytorch/configs/w4a4.yaml` |
| 校准数据 | PileVal，16 个样本，`seq_len=1024` |
| PTQ 轮数 | 15 |
| 精度数据 | WikiText2 test，PPL |
| AMCT 适配名 | `qwen3_5` |

校准数据提取覆盖 64 个 Block，耗时 4013 秒，退出码为 0；七种方案均
生成 64 个 Block 参数。

## 2. 结构与适配要点

- 本模型沿用 `qwen3_5` 的 Dense 主链，无需为 Qwen3.6-27B 新增独立
  adapter；模型名称应以实际加载的 Qwen3.6-27B 权重为准。
- 每层 MLP 的目标 Linear 为 `gate_proj`、`up_proj` 和 `down_proj`，64 层
  共 192 个量化模块。
- 当前经过验证的 Dense MLP 参数入口仍为 `--moe_mlp_param_dir`；参数名中
  的 `moe` 不表示本模型是 MoE。
- 本轮补充了可选 `data_dir`：显式提供时，从本地 `pileval.txt` 提取校准
  样本、从本地 `wikitext2_test.txt` 构造评测样本；未提供时保留原有数据
  加载路径。
- 本地 PileVal 文本需至少提供一个 `seq_len` 的 token；本地采样使用固定
  随机种子，便于不同算法复用同一校准口径。

## 3. 量化与精度结论

- BF16 baseline 为 `7.782411575317383`；Direct 相对 BF16 增加
  `0.111413`。
- LAC、OmniQuant 和 FlatQuant 相对 Direct 恢复部分 PPL；LWC、LWC+LAC
  和 AutoRound 在本次口径下高于 Direct。
- FlatQuant 的 PPL 低于 BF16，只表示本模型、MLP 目标和 WikiText2 配置下
  的实测结果，不应推广为量化普遍提升模型质量。

## 4. 精度速查表

> PPL 数据集为 WikiText2 test。Eval/PTQ 为正式任务的端到端工程记录；
> Eval 跨越不同服务器环境，不用于严格算法速度排名。

| 方法 | PPL | 相对 BF16 | 相对 Direct 恢复量 | Eval/s | PTQ/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| BF16 | 7.782411575317383 | — | — | — | — |
| Direct | 7.893825054168701 | +0.111413 | 0.000000 | 1695 | 727 |
| LWC | 8.021615028381348 | +0.239203 | -0.127790 | 2042 | 2237 |
| LAC | 7.833244323730469 | +0.050833 | +0.060581 | 2586 | 1713 |
| LWC+LAC | 7.982542037963867 | +0.200130 | -0.088717 | 2215 | 2429 |
| OmniQuant | 7.849410057067871 | +0.066998 | +0.044415 | 2014 | 2301 |
| AutoRound | 7.941936492919922 | +0.159525 | -0.048111 | 1820 | 2439 |
| FlatQuant | 7.695919036865234 | -0.086493 | +0.197906 | 1551 | 2517 |

## 5. Deploy 与 clean reload

| 方法 | Deploy/s | 文件数 | 导出大小 | clean reload | Reload/s |
| --- | ---: | ---: | ---: | --- | ---: |
| Direct | 1396 | 80 | 约 29G | PASS | 17 |
| LWC | 1801 | 80 | 约 29G | PASS | 18 |
| LAC | 1577 | 80 | 约 29G | PASS | 19 |
| LWC+LAC | 1511 | 80 | 约 29G | PASS | 19 |
| OmniQuant | 1418 | 80 | 约 29G | PASS | 18 |
| AutoRound | 1475 | 80 | 约 29G | PASS | 19 |
| FlatQuant | 1445 | 80 | 约 29G | PASS | 18 |

七种 Deploy 均以退出码 0 完成。clean reload 均无缺失键、非预期键、形状
不匹配或加载错误；共识别 192 个 MLP 量化模块，第 0 层识别 3 个，Vision
分支和 `lm_head` 未被错误量化。

## 6. 已知问题与复用建议

- Direct 是不启用额外算法的基线，调用时不传 `--algos`；`direct` 不是已
  注册的算法名。
- Dense MLP 的 PTQ 参数仍通过 `--moe_mlp_param_dir` 传入，不要因参数名
  误判模型结构。
- MXFP block size 为 32；Vision、Visual、Image 和输入维度不满足整除约束
  的 Linear 不属于本次文本 MLP 目标，应从 reload 匹配中排除。
- clean reload 需用 BF16 初始化模型骨架和未量化组件；量化配置中不兼容的
  `activation_use_clip` 仅从内存副本移除，不改 Deploy 产物。
- 本案耗时跨不同运行环境，只保留为任务记录；正式性能比较需统一模型、
  数据、设备和负载条件。
