# 子能力：model-adapter-review

本文件用于对已有 LLM adapter/wrapper 做局部非结构修复和跨端约定 review。

## 职责边界

- 模型专属逻辑放在 `amct_pytorch/common/models/llm/<family>/...`。
- workflow 只负责阶段编排和数据交接，不加入模型名、mask、position 或 wrapper 分支。
- solver 只负责优化和参数生命周期，不读取模型结构或离线输入文件。
- 通用 quant module 不判断模型身份；模型差异由 adapter 表达。

## 实现定位

1. 从 `MODEL_REGISTRY` 和目标注册类定位 adapter，确认实际调用方。
2. 按问题搜索 `build_quant_block`、`iter_ptq_units`、embedding/block forward、`iter_deploy_bindings` 及对应测试。
3. 局部 bug 沿“捕获端 -> 离线文件 -> adapter 注入 -> block forward -> wrapper 消费”完整核验。
4. 若只改变下游接口，按约定 owner 追加对应 reference。

## 跨端约定

修改模型层或其消费者时，至少核对：

| 跨端约定 | 必须一致的两端 |
| --- | --- |
| `PtqUnit.kind`、`save_name` | adapter、extract 文件名、PTQ 参数文件和回载逻辑 |
| hook 与 kwargs | adapter 的 embedding/block forward、`llm-cli-dataflow` 的保存与加载 |
| quant block 参数 | adapter、solver 的参数收集/finalize、`PtqParamStore` |
| deploy binding | adapter、deploy 导出的权重映射、ignore layers 和 dtype/algo 配置 |

## 实现规则

### Attention

按真实 blockwise forward 确认 `attention_mask`、`position_ids`、`position_embeddings`、RoPE 和 cache 参数是否参与计算。当前路径不用的 decode/cache 分支不要保留；优先复用同族 wrapper 和 `attention_forward.py`，只替换量化目标内的投影层。确认 SDPA 的 `attn_mask` 与 `is_causal` 组合合法，混合 attention 使用对应 mask，MLA 按实际需求区分 `attn-linear` 与 `attn-cache`。dense/MoE 实现一致时共用一份 wrapper。关闭量化后必须与原始 float block 等价，或在明确容差内接近（容差要写明依据）。

### Wrapper

新增 wrapper 前比较同族实现的初始化、状态和 forward；三者一致就合并，只有模块结构、导出语义、`PtqUnit` 边界或量化路径真实不同才拆分。优先复用 `QuantLinear`、`QuantGatedMLP`、`apply_quant_to_attn()`、`apply_quant_to_moe_mlp()` 和 family helper。packed experts 要确认 PTQ 所需参数已 materialize；shared experts 要和 `build_no_algo_args(args)`、`iter_ptq_units()` 保持一致。`unit.kind` 必须对应 extract 文件，参数必须能保存和回载；loss 为 0 或 NaN 时立即停止检查输入、梯度和 unit 边界。

## 禁止项

- 局部修复引入新的 wrapper 类、模型名分支或注册链改动。
- workflow/solver/通用 quant module 中出现模型专属逻辑。
- unit 约定（`PtqUnit.kind`/`save_name`、hook、deploy binding）单侧改动，未同步另一端。

## 自验

局部修复至少运行目标 adapter 的直接单测；涉及输入捕获、unit 或回载时补对应 workflow mocked 测试。验证应分别说明源码调用链、单测覆盖、集成闭环和真实模型/NPU 验证，不能用 import 成功代替模型适配完成。

### 局部修复完成标准

先用直接断言复现并验证目标 bug，再按实际受影响约定追加下表中的验证；未触及的路径不作为局部修复完成门槛。

| 实际影响 | 追加验证 |
| --- | --- |
| block 构建或权重加载 | 受影响的 float/quant block 可构建，权重名称、shape 和加载结果符合预期 |
| forward、mask/position 或量化开关 | 目标 forward 路径正确；已按「实现规则」取证浮点等价 |
| 输入捕获、kwargs 或 unit 约定 | 对应捕获、保存、加载、unit 枚举及 GT 路径一致；补相关 workflow mocked 测试 |
| PTQ 参数导出或回载 | 受影响状态经实际保存与加载入口恢复，eval/deploy 中受影响的消费者能使用该状态 |
| deploy binding | binding 对应正确权重和模块，导出 payload、tensor route 与 index 一致；仅改 binding 不要求额外验证 GT 路径 |

完成说明列出已验证的跨端约定和仍缺失的证据。缺少真实模型或硬件时明确说明验证范围，不把 mocked 测试通过写成完整模型适配通过。
