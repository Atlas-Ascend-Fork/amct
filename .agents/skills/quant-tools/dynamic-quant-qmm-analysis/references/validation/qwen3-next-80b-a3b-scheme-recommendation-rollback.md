# Qwen3-Next-80B-A3B Scheme Recommendation 回退案例

## 目的

验证 `$dynamic-quant-qmm-analysis` 接入 `$scheme-recommendation` 后，如何把 Linear `dynamic quant + qmm` 的模块级整体负收益用于第一轮量化方案回退。

这个案例与 `Qwen3-4B` 接入案例形成对照：`Qwen3-4B` 的 dense `attn-linear + mlp` 部署比较单元均为理论正收益，不触发回退；本案例中 `Qwen3-Next-80B-A3B` 的 full-attention projection linear 子路径为理论正收益，但 MoE expert 模块为整体负收益，因此触发 `moe` 模块级回退。

本案例仍只提供理论性能筛查结果。推荐决策由 `$scheme-recommendation` 按当前仓可执行粒度完成，qmm estimator 不替代精度评测或 runtime validation。

## 输入口径

- 模型：`Qwen3-Next-80B-A3B-Instruct`
- 结构：Qwen3-Next MoE，混合 `linear_attention` / `full_attention`
- 任务：Wikitext PPL
- `seq_len`：`2048`
- 接入前候选方案：`attn + moe`，`a8w8-int`
- 平台：`A2/A3`
- qmm 方案：`int_w8a8`
- `M`：`2048`，来自 Wikitext PPL `seq_len=2048` 口径，用作本案例 qmm validation 的 full-token 评估参数；实际部署评估时应按 workload 重新确认
- routed expert `M_eff`：`2048 * 10 / 512 = 40`，由 `seq_len * num_experts_per_tok / num_experts` 推导，作为平均路由 token 数；真实部署应补 router token count validation
- 权重准备：`offline_prepared`

参考信息：

- 同系列 casebook：`.agents/docs/casebook/moe/qwen3-next-80b-a3b-instruct.md`
- 模型配置：模型目录下的 `config.json`，`hidden_size=2048`，`moe_intermediate_size=512`，`num_experts=512`，`num_experts_per_tok=10`，`num_attention_heads=16`，`num_key_value_heads=2`，`head_dim=256`，`num_hidden_layers=48`
- 调用形态核对：`amct_pytorch/common/models/llm/qwen/qwen3_next/quant_module.py`
- 通用 MLP wrapper：`amct_pytorch/common/models/llm/common/quant_apply.py` 中 `QuantGatedMLP`
- `QuantLinear` 权重 shape 口径：`amct_pytorch/quantization/modules/quant_linear.py` 中 `linear.weight.shape = [N, K]`

Shape 来源：

- full-token `M=2048`：来自 Wikitext PPL `seq_len=2048` 口径，用作 full-attention projection linear 和 shared expert 的评估参数
- routed expert `M_eff=40`：由模型配置和 full-token `M` 推导，标记为 `model_config_derived`
- `K/N`：由模型目录下的 `config.json` 和 wrapper 中实际 Linear 口径推导，标记为 `model_config_derived`

| 比较单元 | M 来源 | Linear shape |
| --- | --- | --- |
| Routed MoE expert module | `M_eff=2048*10/512=40`，`model_config_derived` | `up/gate: M=40,K=2048,N=512`；`down: M=40,K=512,N=2048` |
| Shared expert module | `M=2048`，来自 `seq_len=2048` 口径，`model_config_derived` | `up/gate: M=2048,K=2048,N=512`；`down: M=2048,K=512,N=2048` |
| Full-attention projection linear path | `M=2048`，来自 `seq_len=2048` 口径，`model_config_derived` | `q: M=2048,K=2048,N=8192`；`k/v: M=2048,K=2048,N=512`；`o: M=2048,K=4096,N=2048` |

## 接入前推荐卡

### 当前模型

- 模型名称：`Qwen3-Next-80B-A3B-Instruct`
- 模型系列：Qwen3-Next MoE
- 结构标签：MoE / mixed attention
- 关键结构特征：block 内部混合 `linear_attention` 和 `full_attention`，MoE routed experts 数量多且单 expert 平均 token 数较小

### 任务口径

- 数据集：Wikitext
- 指标：PPL
- `seq_len`：`2048`
- `granularity`：blockwise / first-round direct quant
- 时间预算：低
- 算力预算：低
- 特殊约束：优先复用 Qwen MoE 系列经验

### 相似案例

- 案例 1：Qwen MoE 系列通常可先从 `attn + moe` 的 `a8w8-int` 直转起步
- 案例 2：同模型 casebook 已完成 BF16 baseline、关闭量化验证和最小 PTQ smoke，但未沉淀 `attn + moe` 的整网直转精度结果

### 候选方案

- 候选方案 A（保守）：`attn-only`，`a8w8-int`
- 候选方案 B（首推）：`attn + moe`，`a8w8-int`
- 候选方案 C（激进）：扩大到更多 Linear，后续再看 PTQ

### 首推方案

- 量化模块：`attn + moe`
- `quant_dtype`：`int`
- `a_bits / w_bits`：`8 / 8`
- `algos`：直转
- `granularity`：默认 per-token/per-channel 口径

### 推荐理由

- 同系列经验：Qwen MoE 系列可从 `attn + moe` 的 `a8w8-int` 作为第一轮起点
- 成本与收益平衡：直转成本低，能同时覆盖 attention projection linear 和 MoE expert
- 失败后是否容易定位：可回退到 `attn-only` 或只保留 MoE 侧验证

### 风险点

- `linear_attention` 和 `full_attention` 不能混为同一类 attention 量化路径处理
- MoE expert 数量多，routed expert 平均 token 数小，qmm fixed overhead 和 activation dynamic quant 可能拉低收益
- routed expert token 分布可能不均匀，平均 `M_eff` 只能作为第一轮理论筛查口径

### 回退方案

- `attn + moe -> attn-only`
- `attn + moe -> moe-only`
- 若直转精度失败，再升级 PTQ

### 证据强度

- 中
- 判断理由：结构和候选路径来自同系列经验；接入前尚未筛查 MoE expert 的 qmm 性能方向

## QMM Estimator 结果

本节只使用理论重算结果，不使用 runtime timing。调用方按 wrapper 的实际执行口径组织比较单元：MLP/expert `up_proj/gate_proj` 共享一次 input activation quant，`down_proj` 独立评估；full-attention projection `q_proj/k_proj/v_proj` 共享一次 attention input activation quant，`o_proj` 独立评估。表中模块级结果把同一可回退模块内的相关比较单元聚合后给出。各比较单元 shape 见上方 Shape 来源表。

| 比较单元 | Baseline 口径 | Target 口径 | T_baseline | T_target | 理论 speedup | 方向 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| Routed MoE expert module | `up_mm + gate_mm + down_mm` | `shared DynamicQuant + up_qmm + gate_qmm + DynamicQuant + down_qmm` | `18.60 us` | `53.77 us` | `0.35x` | 负收益 |
| Shared expert module | `up_mm + gate_mm + down_mm` | `shared DynamicQuant + up_qmm + gate_qmm + DynamicQuant + down_qmm` | `54.37 us` | `101.58 us` | `0.54x` | 负收益 |
| Full-attention projection linear path | `q_mm + k_mm + v_mm + o_mm` | `shared DynamicQuant + q_qmm + k_qmm + v_qmm + DynamicQuant + o_qmm` | `408.77 us` | `320.93 us` | `1.27x` | 正收益 |

观察：

- full-attention projection linear 子路径为理论正收益，可以保留为候选量化子路径；该结论不代表完整 attention module 性能。
- routed expert 的平均 `M_eff=40`，模块聚合后为明确负收益，主风险来自 dynamic quant 和 fixed overhead。
- shared expert 使用 full-token `M=2048` 评估，但 `moe_intermediate_size=512` 较窄，模块聚合后仍为明确负收益。
- `moe` 内 routed expert 和 shared expert 均为整体负收益，满足 `$scheme-recommendation` 的模块级回退条件。
- 当前结论是理论估算结果，实际部署仍需补 router token count 和 runtime validation。

## 接入后推荐卡

### 当前模型

- 模型名称：`Qwen3-Next-80B-A3B-Instruct`
- 模型系列：Qwen3-Next MoE
- 结构标签：MoE / mixed attention
- 关键结构特征：full-attention projection linear 子路径为理论正收益；MoE routed/shared expert 模块均为理论负收益，触发 `moe` 模块级回退

### 任务口径

- 数据集：Wikitext
- 指标：PPL
- `seq_len`：`2048`
- `granularity`：blockwise / first-round direct quant
- 时间预算：低
- 算力预算：低
- 特殊约束：优先复用 Qwen MoE 系列经验，并用 qmm estimator 筛查 Linear 性能方向

### 相似案例

- 案例 1：Qwen MoE 系列经验支持先尝试 `attn + moe` 的 `a8w8-int`
- 案例 2：本文件中的 qmm estimator 显示 full-attention projection linear 子路径为理论正收益，但 MoE expert 模块整体负收益

### 候选方案

- 候选方案 A（首推）：`attn-only`，`a8w8-int`
- 候选方案 B（性能风险）：`attn + moe`，`a8w8-int`
- 候选方案 C（激进）：保留 `attn + moe`，但必须补 router token count 与 runtime validation

### 首推方案

- 量化模块：`attn-only`
- `quant_dtype`：`int`
- `a_bits / w_bits`：`8 / 8`
- `algos`：直转
- `granularity`：activation `PER_TOKEN/PER_ROW`，weight `PER_CHANNEL`

### 推荐理由

- qmm estimator：full-attention projection linear path 理论 speedup 为 `1.27x`，可保留为第一轮直转候选子路径；完整 attention 性能仍需 full attention estimator 或 runtime validation 补强
- qmm estimator：routed MoE expert module 理论 speedup 为 `0.35x`，shared expert module 理论 speedup 为 `0.54x`，均为模块级整体负收益
- 成本与收益平衡：当前 qmm 结果触发 `moe` 模块级回退；保留 `attn-only` 能避免 MoE expert 小 `M_eff` 带来的 dynamic quant/fixed overhead 风险
- 失败后是否容易定位：后续可单独补 MoE expert 的 router token count 和 runtime validation，再决定是否重新纳入 `moe`

### 风险点

- routed expert 的 `M_eff=40` 是平均路由口径；真实 token 分布如果高度偏斜，部分 hot expert 可能与平均结论不同
- `linear_attention` 路径没有在本案例中作为同一 attention 比较单元评估，不应把 full-attention projection linear 子路径结论泛化到所有 attention 形态或完整 attention module
- 当前 qmm validation 使用 `seq_len=2048` 口径；实际 workload 的 `M` 或 expert routing 明显变化时需要重新评估

### 升级条件

- 若必须量化 `moe`，下一轮先补 router token count，并按 hot/cold expert 分桶重新评估 `M_eff`
- 若 MoE runtime validation 显示 hot expert 有稳定正收益，可考虑只在可执行粒度允许时纳入对应 MoE 路径
- 若精度优先且必须保留 `attn + moe`，应明确标记 MoE 性能风险，并把 runtime validation 作为阻塞项

### 回退方案

- `attn + moe -> attn-only`
- 若 `attn-only` 精度失败，再评估 `moe-only` 或进入 PTQ
- 若 runtime validation 推翻 MoE 负收益结论，再恢复 `attn + moe`

### 证据强度

- 中
- 判断理由：性能依据来自 qmm estimator 按部署比较单元的理论评估；routed expert 使用平均 `M_eff`，仍需 router token count 和 runtime validation 补强

## 接入前后对比

| 字段 | 接入前 | 接入后 |
| --- | --- | --- |
| 首推方案 | `attn + moe` | `attn-only` |
| 推荐依据 | Qwen MoE 系列经验 | Qwen MoE 系列经验 + qmm 部署比较单元理论评估 |
| 风险点 | mixed attention、MoE expert layout | MoE routed/shared expert 模块级整体负收益；attention 侧仅覆盖 projection linear 子路径 |
| 回退粒度 | 模块级回退 | 触发 `moe` 模块级回退：`attn + moe -> attn-only` |
| 证据强度 | 中等结构经验 | 中等综合证据，MoE 仍需 router token count 与 runtime validation |

## 结论

- qmm skill 接入后，recommendation 不再只按 Qwen MoE 系列经验首推 `attn + moe`。
- 按实际 wrapper 口径组织比较后，full-attention projection linear 子路径为理论正收益，MoE routed/shared expert module 为理论负收益。
- 由于 `moe` 内相关比较单元整体明确负收益，且当前回退粒度支持模块级候选回退，本案例触发 `moe` 模块级回退。
- 接入后首推方案从 `attn + moe` 调整为 `attn-only`。
- 该案例与 `Qwen3-4B` 正收益案例共同说明 qmm estimator 对 `$scheme-recommendation` 的作用：既能补充性能风险项，也能在模块级整体负收益时改变首推方案。
