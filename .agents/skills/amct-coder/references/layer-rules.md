# 分层边界与越界检查

## 来源

- `.agents/docs/repo-map.md`
- `.agents/docs/casebook/cross-model-pitfalls.md`
- `.agents/docs/casebook/structure-family-pitfalls.md`
- `amct_pytorch/workflows/llm_ptq.py`
- `amct_pytorch/common/models/llm/common/base.py`
- `amct_pytorch/common/models/llm/common/quant_apply.py`
- `amct_pytorch/quantization/modules/quant_base.py`

## 必读顺序

1. 先读 `.agents/docs/repo-map.md`。
2. 抽查任务相关锚点文件，确认 repo-map 的关键结论仍成立。
3. 再读 L1 casebook：`.agents/docs/casebook/cross-model-pitfalls.md`。
4. 命中结构家族时读 L2 casebook：`.agents/docs/casebook/structure-family-pitfalls.md`。
5. 命中已记录模型族时读 L3 个案。

repo-map 是稳定架构说明，不是实时注册清单。若锚点文件显示注册模型或局部实现已经变化，以代码为准；只有框架边界或核心流程变化时才更新 repo-map。

## 层级职责

层与职责的权威表在 `.agents/docs/repo-map.md` §分层边界，本文件不复制。本文件只写落点判断和 BLOCKER 检查。

`repo-map` 的分层表覆盖 CLI、Workflow、Model adapter、Quant modules、Algorithms、Optimization、Data 七层；deploy 相关落点见 [deploy-export-code.md](deploy-export-code.md)。

## 默认落点判断

- 新模型、新结构、权重 key 特例、tied `lm_head`、position/mask 捕获、packed experts 重组：放 `amct_pytorch/common/models/llm/<family>/...`。
- 同一模型族可共享的 MoE expert 视图、attention helper、wrapper：优先放该 family 目录或 `common/models/llm/common/`。
- 跨多个模型族都稳定复用的能力，才考虑放 `common/models/llm/common/`。
- 与模型无关的量化模块能力，才考虑 `quantization/modules/`。
- 真正改变 PTQ 编排语义，才考虑 `workflows/`。
- 真正改变训练/重建范式，才考虑 `common/optimization/`。

## 按层检查规则

先确定改动落在哪层，再逐条查该层的检查项。「命中即 BLOCKER」列命中时停下改方案，不要直接编辑。

| 改动层 | 检查什么 | 命中即 BLOCKER |
| --- | --- | --- |
| CLI `cli/llm/` | diff 是否出现模型类名 import、按模型名的 `if` 分支、结构计算 | 有任一项 |
| Workflow `workflows/` | diff 是否出现模型类名 import、attention/MLP/MoE 计算、模型专属 hook 名、weight-key 特例、mask/position 构造 | 有任一项 |
| Model adapter `common/models/llm/` | 是新增 adapter/wrapper 类或结构性修改，还是不改结构的局部修复 | 前者——转 `model-adapter`，本 skill 不实现 |
| Quant modules `quantization/modules/` | diff 是否判断具体模型族或模型名；若要扩接口，现有接口是否真的表达不了 | 出现模型族判断，或未证明现有接口不够就改接口 |
| Algorithms `algorithms/` | 是否有 `@ALGO_REGISTRY.register(...)`、是否声明 `targets`、`algorithms/quant/__init__.py` 是否 import、`build_algorithms_by_target()` 能否按 `args.algos` 选中 | 缺任一项，或在 workflow 里用 `if/else` 硬编码分流代替注册 |
| Optimization `common/optimization/` | diff 是否按模型名或 layer 类型分支；模型差异能否改由 `PtqUnit`/adapter 表达 | 出现模型分支 |
| Data `common/datasets/` | 是否为单个模型改通用 provider 行为；能否复用现有 provider | 为单模型改通用行为且未证明其它模型同样需要 |

deploy 相关落点见 [deploy-export-code.md](deploy-export-code.md)。

## 跨层 BLOCKER

不绑定单层，发现即停：

- 为单个模型修改通用 bit policy 或 dtype 行为，却没有证明其它模型也需要。
- 新增算法未通过 `ALGO_REGISTRY.register(...)` 注册。
- 新增算法没有声明 `targets`，或 `targets` 与 `quant_base.py` 的分流不匹配。
- PTQ eval/deploy 加载训练参数时漏掉与训练一致的 `--algos`。
- `build_no_algo_args(args)` 与 adapter 的 `iter_ptq_units()` 对同一 unit 策略不一致。

## 模型适配前结构自检

适配前 7 问见 [llm-model-unit.md](llm-model-unit.md) §适配清单，本文件不复制。答不清时先读 checkpoint index、config 和已有同族 adapter，不要先写 workflow 兜底。

## 算法注册检查

LLM new-path 算法以 `amct_pytorch/algorithms/registry_factory.py` 的 `ALGO_REGISTRY` 为准。classic path 中 `AlgorithmRegistry.register(...)` 的算法名不代表 LLM PTQ 可用。

新增或修改算法必须检查：

- 算法文件中有 `@ALGO_REGISTRY.register(...)`。
- metadata 中有 `targets`。
- `targets` 只能表达真实作用域：`activation`、`weight`、`structure`。
- `amct_pytorch/algorithms/quant/__init__.py` 导入该算法文件。
- `amct_pytorch/quantization/modules/quant_base.py` 中 `build_algorithms_by_target()` 能按 `args.algos` 选中。

## repo-map 写回规则

只有以下变化触发更新 `.agents/docs/repo-map.md`：

- `llm_ptq.py` 改变主流程编排语义。
- `common/models/llm/__init__.py` 改变实际注册链的含义。
- `BaseModel` 改变 block / unit / 导出加载语义。
- `quant_base.py` 改变算法分流方式。
- `blockwise_solver.py` 改变优化职责边界。

这些情况默认不更新 repo-map：

- 新增尚未接入主注册链的模型目录。
- 新增一个算法文件但不改变分流机制。
- 修局部 wrapper bug。
- 改不影响复用边界的模型专属实现细节。
