# 子能力：llm-model-unit

## 触发

用户提到模型注册、`MODEL_REGISTRY`、新模型 adapter、Qwen/DeepSeek/GLM/LongCat/Hunyuan、attention/MLP/MoE wrapper、packed experts、`PtqUnit`、权重 key、mask/position、tied `lm_head` 时使用。

## 必读文件

- `.agents/docs/repo-map.md`
- `.agents/docs/casebook/cross-model-pitfalls.md`
- `.agents/docs/casebook/structure-family-pitfalls.md`
- 命中模型族的 `.agents/docs/casebook/<family>/...`
- `amct_pytorch/common/models/llm/__init__.py`
- `amct_pytorch/common/models/llm/common/base.py`
- `amct_pytorch/common/models/llm/common/quant_apply.py`
- `amct_pytorch/common/models/llm/common/ptq_units.py`
- `amct_pytorch/common/models/llm/common/attention_forward.py`
- 现有同族 `quant_module.py` / `moe_common.py`
- 涉及 attention 细节时再读 `references/attention-rules.md`
- 涉及 MLP/MoE wrapper 合并时再读 `references/wrapper-rules.md`

## 上下游依赖

- 上游：模型适配依赖真实 checkpoint key、config、modeling 代码、repo-map/casebook 和通用 quant module 能力。
- 平级：adapter 不实现 dtype 或算法细节，只决定哪些模块被替换、哪些 unit 被训练、哪些权重被导出。
- 下游：`llm-data-workflow` 消费 `do_embedding_forward()`、`do_block_forward()`、hook 名和 `PtqUnit.kind`；`ptq-solver-params` 消费 `PtqUnit` 和 quant block；`deploy-export-code` 消费 deploy binding、ignore layers 和 bits/cache scheme。
- 用户要求纯模型适配时还要考虑是否应转 `model-adapter`；若同时改 dtype/algorithm/workflow，则继续使用本 playbook 与对应 playbook。

## 执行步骤

1. 读取 config、checkpoint index 和同族 adapter，确认 block、attention、MLP/MoE、experts、tied weight、mask/position 的真实结构。
2. 注册模型族和模型名，保证 `MODEL_REGISTRY` 能找到 adapter。
3. 实现 float block、quant block 构建和权重加载，优先复用 `QuantLinear`、`QuantGatedMLP`、`apply_quant_to_attn()`、`apply_quant_to_moe_mlp()`。
4. 定义 `iter_ptq_units()`，确保 `unit.kind` 与 extract 产物 `block_<layer>_<target>_in.pkl` 一致。
5. 实现 blockwise forward 所需 kwargs 的捕获、保存和回放，特别是 `attention_mask`、`position_ids`、`position_embeddings`。
6. 若支持 deploy，补 `iter_deploy_bindings()`、ignore layers、bits/cache scheme 和 tied weight 处理。
7. 补注册、block 构建、quant block 等价、unit 枚举、输入命名和 deploy binding 的最小测试。

## 默认落点

新模型和模型专属逻辑默认放：

`amct_pytorch/common/models/llm/<family>/<model>/...`

只在共享抽象确实跨多个模型族复用时，才改 `common/models/llm/common/`。

## 适配清单

编码前先回答：

1. checkpoint 真实 key 模式是什么。
2. block / attention / MLP class 是什么。
3. dense、MoE、MLA、混合 attention、自定义 modeling 分别命中哪些。
4. experts 是磁盘展开还是运行时 packed。
5. `lm_head` 是否 tied 到 `embed_tokens`。
6. blockwise forward 需要哪些 `attention_mask`、`position_ids`、`position_embeddings`、`input_ids`。
7. 要拆哪些 `PtqUnit`，`unit.kind` 是否和 extract 产物命名一致。

## wrapper 规则

- 先复用 `QuantLinear`、`QuantGatedMLP`、`apply_quant_to_attn()`、`apply_quant_to_moe_mlp()`。
- 同系列 attention/MLP wrapper 初始化、状态、forward 一样时必须合并。
- attention wrapper 面向 blockwise BF16/PPL/PTQ/deploy，不默认保留 generate/cache 全分支。
- `attn-linear` 只 wrap q/k/v/o 等投影时，不替换官方 attention kernel。
- MoE shared experts 若 wrapper 走 `build_no_algo_args()`，adapter 不 yield `moe.shared`。

## BLOCKER

- 在 workflow 中写某个模型的 attention、MLP、mask、weight key 特例。
- 在 quant module 中判断模型名。
- `PtqUnit.kind` 与 `load_ptq_inps(data_dir, kind, layer_idx)` 文件名不一致。
- `Catcher` 漏透传模型特有字段，导致 embedding 捕获不等价。
- BF16 PPL 明显异常但继续做 PTQ。
- 关闭量化后 wrapper 与原始 float block 不等价。

## 自验

- 注册：检查 `MODEL_REGISTRY` 能取到新模型名。
- 构建：至少一层 `block()` 和 `build_quant_block()` 成功。
- PTQ unit：至少枚举一个 `PtqUnit`。
- 等价：关闭量化后 quant block 与 float block forward 足够接近。
- 数据：extract 产出的文件名能被该 unit 加载。
- 单测：优先放 `tests/unit_test/common/models/llm/...`，必要时补 workflow mocked 测试。
