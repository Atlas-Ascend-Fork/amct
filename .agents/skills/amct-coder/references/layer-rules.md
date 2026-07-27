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

| 层 | 主要位置 | 职责 | 默认规则 |
| --- | --- | --- | --- |
| CLI | `amct_pytorch/cli/llm/` | 参数解析、启动 workflow | 不放模型逻辑 |
| Workflow | `amct_pytorch/workflows/` | 编排 eval、PTQ 数据提取、PTQ 训练、deploy | 不放模型细节 |
| Model adapter | `amct_pytorch/common/models/llm/...` | 模型注册、block 构建、quant block、`PtqUnit`、输入重放、权重加载 | 新模型和模型专属逻辑默认落点 |
| Quant modules | `amct_pytorch/quantization/modules/` | 通用量化 wrapper、量化器、`QuantLinear` | 保持通用，不写模型名分支 |
| Algorithms | `amct_pytorch/algorithms/` | activation / weight / structure 算法 | 通过 LLM new-path registry 和 `targets` 分流 |
| Optimization | `amct_pytorch/common/optimization/` | optimizer、epoch 循环、重建训练、solver | 不混入模型分支 |
| Data | `amct_pytorch/common/datasets/` | PTQ 输入、GT、dataloader | 优先复用现有 provider |

## 默认落点判断

- 新模型、新结构、权重 key 特例、tied `lm_head`、position/mask 捕获、packed experts 重组：放 `amct_pytorch/common/models/llm/<family>/...`。
- 同一模型族可共享的 MoE expert 视图、attention helper、wrapper：优先放该 family 目录或 `common/models/llm/common/`。
- 跨多个模型族都稳定复用的能力，才考虑放 `common/models/llm/common/`。
- 与模型无关的量化模块能力，才考虑 `quantization/modules/`。
- 真正改变 PTQ 编排语义，才考虑 `workflows/`。
- 真正改变训练/重建范式，才考虑 `common/optimization/`。

## BLOCKER 检查

发现下列情况必须先停下，不要直接编辑：

- 在 CLI 或 workflow 中加入某个模型的 attention、MLP、MoE、mask、position、weight-key 特例。
- 在 solver 中通过模型名或 layer 类型写分支。
- 在 `quantization/modules/` 中判断具体模型族。
- 为单个模型修改通用 bit policy、dtype 或 data provider 行为，却没有证明其它模型也需要。
- 新增算法未通过 `ALGO_REGISTRY.register(...)` 注册。
- 新增算法没有声明 `targets`，或 `targets` 与 `quant_base.py` 的分流不匹配。
- PTQ eval/deploy 加载训练参数时漏掉与训练一致的 `--algos`。
- `build_no_algo_args(args)` 与 adapter 的 `iter_ptq_units()` 对同一 unit 策略不一致。

## 模型适配前结构自检

编码前先回答：

1. checkpoint 的真实 key 模式是什么，不要只靠 config 或 `architectures` 推断。
2. block / attention / MLP class 是什么。
3. dense、MoE、MLA、混合 attention、自定义 modeling 分别命中哪些。
4. experts 是磁盘展开还是运行时 packed。
5. `lm_head` 是否 tied 到 `embed_tokens`。
6. blockwise 路径需要哪些 `attention_mask`、`position_ids`、`position_embeddings`、`input_ids`。
7. 要拆哪些 `PtqUnit`，`unit.kind` 是否和 extract 产物命名一致。

这些问题答不清时，先读 checkpoint index、config 和已有同族 adapter，不要先写 workflow 兜底。

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
