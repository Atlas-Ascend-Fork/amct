# 子能力：llm-model-unit（模型层契约与委派）

## 定位

本子能力**不承接模型层实现**。新增模型 adapter、新增 wrapper 类、改 attention/MLP/MoE 结构、注册新模型，一律转 `model-adapter`。

amct-coder 在模型层只保留三件事：

1. **契约对齐检查**：模型层产出的 `PtqUnit.kind`、hook 名、deploy binding 是否与 `llm-data-workflow`、`ptq-solver-params`、`deploy-export-code` 的消费口径一致。
2. **review**：越层改动、注册缺失、关量化不等价、验证缺失。
3. **局部 bug 修复**：不改结构、不新增 wrapper、不动注册链的小范围修复。

## 触发与委派判断

| 用户需求 | 处理方式 |
| --- | --- |
| 新增模型族 / 新模型 adapter / 注册新模型 | 转 `model-adapter` |
| 新增 attention / MLP / MoE wrapper 类，或改其结构 | 转 `model-adapter` |
| 改 `iter_ptq_units()` 的 unit 拆分策略 | 转 `model-adapter` |
| 改动 dtype / 算法 / workflow / solver / deploy，需要确认模型层契约是否还成立 | 本文件（契约对齐） |
| review 已有模型层代码或 model-adapter 的产出 | 本文件 + [validation.md](validation.md) |
| 修 adapter 的局部 bug（权重 key 拼错、kwargs 漏传、命名不一致） | 本文件 |
| 判不清是纯模型适配还是跨层 | 先答下面「适配清单」7 问，全部落在模型层就转 `model-adapter` |

委派时先简短说明原因，例如：“这是新增模型 adapter 和 wrapper，属模型层实现，应使用 `model-adapter`。”其余委派场景见 [delegation.md](delegation.md)。

## 必读文件

做契约对齐或 review 时读：

- `.agents/docs/repo-map.md`
- `amct_pytorch/common/models/llm/common/base.py`
- `amct_pytorch/common/models/llm/common/ptq_units.py`
- `amct_pytorch/common/models/llm/__init__.py`（确认注册链）
- 命中模型的 adapter 文件

review 涉及 attention 或 wrapper 细节时，再读 [attention-rules.md](attention-rules.md) / [wrapper-rules.md](wrapper-rules.md)。这两份文件的必查清单同样适用于检查 `model-adapter` 的产出。

模型个案见 `.agents/docs/casebook/cross-model-pitfalls.md`、`structure-family-pitfalls.md` 和命中模型族的 L3 个案。

## 契约清单

模型层对下游的契约，改动任一侧都要两边同时确认：

| 契约 | 生产者 | 消费者 |
| --- | --- | --- |
| `PtqUnit.kind` | adapter `iter_ptq_units()` | extract 产物 `block_<layer>_<target>_in.pkl`、`load_ptq_inps()` |
| `unit.save_name` | adapter | PTQ 参数文件 `layer_<idx>_<unit.save_name>.pt` |
| hook 名 | adapter `do_embedding_forward()` / `do_block_forward()` | `llm-data-workflow` 的 extract 保存端 |
| blockwise kwargs | adapter 捕获 | `save_ptq_kwargs()` / provider 回放 |
| deploy binding、ignore layers、bits/cache scheme | adapter `iter_deploy_bindings()` | `deploy-export-code` |
| quant block 结构 | adapter `build_quant_block()` | `ptq-solver-params` 的参数收集与回载 |

## 默认落点

模型专属逻辑落 `amct_pytorch/common/models/llm/<family>/<model>/...`；只在共享抽象确实跨多个模型族复用时才动 `common/models/llm/common/`。发现改动要越出这个范围时停下——这通常意味着任务不是纯模型适配，需要回到对应子能力 playbook。

## 适配清单

用于判断任务是否为纯模型层，也用于 review 时核对 `model-adapter` 的输入是否完整：

1. checkpoint 真实 key 模式是什么。
2. block / attention / MLP class 是什么。
3. dense、MoE、MLA、混合 attention、自定义 modeling 分别命中哪些。
4. experts 是磁盘展开还是运行时 packed。
5. `lm_head` 是否 tied 到 `embed_tokens`。
6. blockwise forward 需要哪些 `attention_mask`、`position_ids`、`position_embeddings`、`input_ids`。
7. 要拆哪些 `PtqUnit`，`unit.kind` 是否和 extract 产物命名一致。

7 问全部落在模型层就转 `model-adapter`；出现 dtype、算法、workflow、solver、deploy 侧的改动，那部分留在 amct-coder。

## wrapper 规则

wrapper 是否新增由“结构或导出语义是否真实不同”决定，不由 dense/MoE 标签决定。复用顺序和 MoE/shared experts 策略见 [wrapper-rules.md](wrapper-rules.md)，attention 细节见 [attention-rules.md](attention-rules.md)。本 skill 用它们做 review 判据，不用它们写新 wrapper。

## BLOCKER

- 在本 skill 内新增 adapter、新增 wrapper 类或改模型结构，而不是转 `model-adapter`。
- 在 workflow 中写某个模型的 attention、MLP、mask、weight key 特例。
- 在 quant module 中判断模型名。
- `PtqUnit.kind` 与 `load_ptq_inps(data_dir, kind, layer_idx)` 文件名不一致。
- `Catcher` 漏透传模型特有字段，导致 embedding 捕获不等价。
- BF16 PPL 明显异常但继续做 PTQ。
- 关闭量化后 wrapper 与原始 float block 不等价。

## 自验

完成标准见 [validation.md](validation.md) §LLM adapter / wrapper 完成标准（注册、float block、quant block、PtqUnit、关量化等价、参数导出闭环六项）。review `model-adapter` 产出时用同一套标准。

契约对齐或局部修复的额外要求：extract 产出的文件名能被该 unit 加载；单测优先放 `tests/unit_test/common/models/llm/...`，必要时补 workflow mocked 测试。
