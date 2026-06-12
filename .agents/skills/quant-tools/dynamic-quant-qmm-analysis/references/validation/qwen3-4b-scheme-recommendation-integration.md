# Qwen3-4B Scheme Recommendation 接入案例

## 目的

验证 `$dynamic-quant-qmm-analysis` 接入 `$scheme-recommendation` 后，如何把 Linear `dynamic quant + qmm` 的收益方向用于第一轮量化方案筛选。

这个案例不改变 qmm estimator 的定位：qmm skill 只提供理论评估、方向验证和关键假设；推荐决策仍由 `$scheme-recommendation` 在自己的语境中完成。

## 输入口径

- 模型：`Qwen3-4B`
- 结构：dense Qwen3 decoder
- 任务：Wikitext PPL
- `seq_len`：`2048`
- 候选方案：`attn-linear + mlp`，`a8w8-int`
- 平台：`A2/A3`
- qmm 方案：`int_w8a8`
- 权重准备：`offline_prepared`

参考信息：

- 同模型 casebook：`.agents/docs/casebook/dense/qwen3-4b.md`
- 模型配置：模型目录下的 `config.json`，`hidden_size=2560`，`intermediate_size=9728`，`num_attention_heads=32`，`num_key_value_heads=8`，`head_dim=128`，`num_hidden_layers=36`
- 已有精度结果：`attn-linear + mlp`，`a8w8-int`，`delta=0.1736`

Shape 来源：

- `M=2048`：来自同模型 casebook 的 Wikitext PPL `seq_len=2048` 口径，用作本案例 qmm validation 的评估参数
- `K/N`：由模型目录下的 `config.json` 推导
- 调用形态核对：`amct_pytorch/common/models/llm/qwen/qwen3/quant_module.py`
- `QuantLinear` 权重 shape 口径：`amct_pytorch/quantization/modules/quant_linear.py` 中 `linear.weight.shape = [N, K]`

| 比较单元 | M 来源 | Linear shape |
| --- | --- | --- |
| MLP `up_proj/gate_proj` group | `M=2048`，来自 casebook `seq_len=2048` 口径，`model_config_derived` | `up/gate: M=2048,K=2560,N=9728` |
| MLP `down_proj` | `M=2048`，来自 casebook `seq_len=2048` 口径，`model_config_derived` | `down: M=2048,K=9728,N=2560` |
| Attention projection `q_proj/k_proj/v_proj` group | `M=2048`，来自 casebook `seq_len=2048` 口径，`model_config_derived` | `q: M=2048,K=2560,N=4096`；`k/v: M=2048,K=2560,N=1024` |
| Attention projection `o_proj` | `M=2048`，来自 casebook `seq_len=2048` 口径，`model_config_derived` | `o: M=2048,K=4096,N=2560` |

## 接入前推荐卡

### 当前模型

- 模型名称：`Qwen3-4B`
- 模型系列：Qwen3 dense
- 结构标签：dense
- 关键结构特征：标准 dense decoder，attention projection linear 和 MLP 都已接入 `QuantLinear`

### 任务口径

- 数据集：Wikitext
- 指标：PPL
- `seq_len`：`2048`
- `granularity`：blockwise / first-round direct quant
- 时间预算：低
- 算力预算：低
- 特殊约束：优先复用 casebook

### 相似案例

- 案例 1：`Qwen3-4B` casebook，`attn-linear + mlp a8w8-int` 已验证 `delta=0.1736`

### 候选方案

- 候选方案 A（保守）：`mlp only`，`a8w8-int`
- 候选方案 B（首推）：`attn-linear + mlp`，`a8w8-int`
- 候选方案 C（激进）：扩大到更多 Linear，后续再看 PTQ

### 首推方案

- 量化模块：`attn-linear + mlp`
- `quant_dtype`：`int`
- `a_bits / w_bits`：`8 / 8`
- `algos`：直转
- `granularity`：默认 per-token/per-channel 口径

### 推荐理由

- 同系列经验：`Qwen3-4B` casebook 已验证可接受
- 结构相似性：当前就是同一模型个案
- 成本与收益平衡：直转成本低，`delta <= 0.2`
- 失败后是否容易定位：可拆 `mlp-only` 和 `attn-linear-only`

### 风险点

- attention wrapper 等价性必须保持
- 若后续模型结构变化，需重新确认 mask/rotary 路径
- 小 shape Linear 的性能收益未筛查

### 升级条件

- `delta > 0.2` 时进入 PTQ 或误差定位
- attention-only 异常时先检查 wrapper 等价
- MLP 异常时先缩到 `mlp-only`

### 回退方案

- `attn-linear + mlp -> mlp only`
- `attn-linear + mlp -> attn-linear only`
- 直转失败后升级 PTQ

### 证据强度

- 强
- 判断理由：同模型 casebook 已有实测精度结果

## QMM Estimator 结果

本节只使用理论重算结果，不使用 runtime timing。调用方按 `amct_pytorch/common/models/llm/qwen/qwen3/quant_module.py` 的实际执行口径组织比较单元：MLP `up_proj/gate_proj` 共享一次 input activation quant，attention projection `q_proj/k_proj/v_proj` 共享一次 attention input activation quant，`down_proj` 和 `o_proj` 分别独立评估。各比较单元 shape 见上方 Shape 来源表。

| 比较单元 | Baseline 口径 | Target 口径 | T_baseline | T_target | 理论 speedup | 方向 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| MLP `up_proj/gate_proj` group | `up_mm + gate_mm` | `shared DynamicQuant + up_qmm + gate_qmm` | `718.25 us` | `448.70 us` | `1.60x` | 正收益 |
| MLP `down_proj` | `down_mm` | `DynamicQuant + down_qmm` | `374.18 us` | `236.94 us` | `1.58x` | 正收益 |
| Attention projection `q_proj/k_proj/v_proj` group | `q_mm + k_mm + v_mm` | `shared DynamicQuant + q_qmm + k_qmm + v_qmm` | `239.55 us` | `182.60 us` | `1.31x` | 正收益 |
| Attention projection `o_proj` | `o_mm` | `DynamicQuant + o_qmm` | `157.55 us` | `120.12 us` | `1.31x` | 正收益 |

观察：

- 按实际 wrapper 口径组织后，4 个部署比较单元均为理论正收益。
- 单独看 `k_proj/v_proj` 时窄 `N=1024` 会呈负收益倾向，但在 attention `q/k/v` group 中 shared activation quant 只计一次，group 级方向转为正收益。
- attention projection linear 结果只代表 projection 子路径，不代表完整 attention module 性能。
- 当前结论是理论估算结果，实际部署仍需结合具体 workload `M` 和 runtime validation 确认。

## 接入后推荐卡

### 当前模型

- 模型名称：`Qwen3-4B`
- 模型系列：Qwen3 dense
- 结构标签：dense
- 关键结构特征：按实际 wrapper 口径组织比较后，MLP 和 attention projection linear 比较单元均为理论正收益；`k_proj/v_proj` 仍是 group 内的窄 `N` 风险来源

### 任务口径

- 数据集：Wikitext
- 指标：PPL
- `seq_len`：`2048`
- `granularity`：blockwise / first-round direct quant
- 时间预算：低
- 算力预算：低
- 特殊约束：优先复用 casebook，并用 qmm estimator 筛查 Linear 性能方向

### 相似案例

- 案例 1：`Qwen3-4B` casebook 支持 `attn-linear + mlp a8w8-int` 精度可接受
- 案例 2：本文件中的 qmm estimator 按部署比较单元显示 MLP 和 attention projection linear 均为理论正收益

### 候选方案

- 候选方案 A（保守）：`mlp only`，`a8w8-int`
- 候选方案 B（首推）：`attn-linear + mlp`，`a8w8-int`
- 候选方案 C（激进）：扩大到更多 Linear，后续再看 PTQ

### 首推方案

- 量化模块：`attn-linear + mlp`
- `quant_dtype`：`int`
- `a_bits / w_bits`：`8 / 8`
- `algos`：直转
- `granularity`：activation `PER_TOKEN/PER_ROW`，weight `PER_CHANNEL`

### 推荐理由

- 同系列经验：原 casebook 支持 `attn-linear + mlp a8w8-int` 精度可接受
- qmm estimator：按部署比较单元重算后，MLP `up/gate` group、MLP `down_proj`、attention projection `q/k/v` group 和 attention projection `o_proj` 均为理论正收益
- 成本与收益平衡：当前 qmm 结果不触发模块级回退；`k_proj/v_proj` 作为 attention group 内的窄 `N` 风险来源保留在风险点
- 失败后是否容易定位：当前仓按模块级回退，优先在下一轮验证 `mlp-only` 或 `attn-linear-only`

### 风险点

- `k_proj/v_proj` 的窄 `N=1024` 仍可能拉低 attention `q/k/v` group 收益，后续 runtime validation 应重点关注
- 当前 qmm validation 使用来自 casebook `seq_len=2048` 口径的 `M=2048`；实际 workload 的 `M` 明显变化时需要重新评估

### 升级条件

- 若必须保持 `attn-linear` 全量量化，下一轮补 attention `q/k/v` group 的更多 workload shape validation
- 若需要执行回退，只有模块内相关 Linear 或调用方整理的比较单元整体明确负收益时才建议模块级回退
- 若精度优先，保留 casebook 全量 `attn-linear + mlp`，但标记性能风险

### 回退方案

- `attn-linear + mlp -> mlp only`
- `attn-linear + mlp -> attn-linear only`
- 若直转精度失败，再升级 PTQ

### 证据强度

- 中
- 判断理由：精度依据来自同模型 casebook；性能依据来自 qmm estimator 按部署比较单元的理论评估，但覆盖 shape 仍是单点 `M=2048` 口径

## 接入前后对比

| 字段 | 接入前 | 接入后 |
| --- | --- | --- |
| 首推方案 | `attn-linear + mlp` | `attn-linear + mlp` |
| 推荐依据 | casebook 精度结果 | casebook 精度结果 + qmm 部署比较单元理论评估 |
| 风险点 | attention wrapper、误差风险 | 新增 attention `q/k/v` group 需按 workload 扩展验证的性能风险 |
| 回退粒度 | 模块级回退 | 仍按模块级回退；仅当模块内比较单元整体明确负收益时回退 |
| 证据强度 | 强精度证据 | 中等综合证据，性能需按 workload 扩展验证 |

## 结论

- qmm skill 接入后，recommendation 在模块经验之外补充部署比较单元的理论性能依据。
- 按实际 wrapper 口径组织比较后，MLP 和 attention projection linear 比较单元均为理论正收益。
- 由于当前没有模块内比较单元整体明确负收益，且当前回退粒度为模块级，本案例不触发模块级回退。
- 该案例证明 qmm estimator 对下游 recommendation 的实际作用是补充性能风险项，而不是替代 recommendation 直接下决策。
