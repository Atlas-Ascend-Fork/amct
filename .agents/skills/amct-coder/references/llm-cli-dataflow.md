# 子能力：llm-cli-dataflow

本文直接对应当前 AMCT LLM CLI 源码，用于开发和定位 `extract_ptq_data`、`ptq`、`eval`、`deploy` 的数据问题。

## 职责边界

- 覆盖 `extract_ptq_data`、`ptq`、`eval`、`deploy` 四个 CLI 阶段的 workflow 编排、阶段产物与跨阶段约定：谁生产、谁消费、改一端要同步另一端。
- 组件级职责分工见「组件协作」的组件表。
- 不覆盖 classic eager/graph_based API。

## 组件协作

```text
LLM CLI: eval | extract_ptq_data | ptq | deploy
  -> 参数解析与配置
  -> 对应 workflow（注册、构建组件、调度、管理阶段产物）
       +-- Model adapter：屏蔽模型结构差异，提供 block/unit 与 forward
       |     +-- 模型专属 wrapper：保持模型计算结构
       |           +-- 通用 quant module
       |                 +-- dtype：量化数值语义
       |                 +-- algorithm：量化算法行为与状态
       +-- Data provider：组织 PTQ 输入、参考输出和 batch
       +-- Solver：执行优化，收集并返回 PTQ 结果
       +-- 参数存储与导出辅助：参数保存/回载、部署产物组装
```

这是组件协作图，不是每条命令都会执行的调用序列；workflow 按阶段使用相应能力。

| 组件 | 在架构中的作用 | 不承担的职责 |
| --- | --- | --- |
| CLI | 解析用户选择，启动对应 workflow | 模型结构与算法实现 |
| Workflow | 选择并连接组件，管理执行顺序、状态和产物 | 模型专属 forward、dtype 数值规则 |
| Model adapter | 将具体模型接入通用接口，构建浮点/量化 block，枚举 `PtqUnit`，提供输入捕获及权重绑定 | 通用优化循环 |
| Quant module | 在模型计算中组合 dtype 和 algorithm，控制量化与 observe 行为 | CLI 阶段编排 |
| Dtype / algorithm | 分别定义量化数值规则、算法处理及其状态 | 模型加载与离线目录管理 |
| Data provider | 经 adapter 加载 unit 输入，生成 GT，组织训练 batch | 算法优化策略 |
| Solver | 消费 unit、batch 和算法参数，优化并返回结果 | 解析模型结构、直接读取离线文件 |
| 参数存储与导出辅助 | 由 workflow/adapter 调用，衔接 PTQ 状态的保存回载与部署格式 | 重新定义算法或 dtype 语义 |

`PtqUnit` 是 adapter 暴露给 PTQ 的处理单元，不是独立服务；offline data 是阶段产物，不是组件。kwargs 是 forward 附加输入，可能共享也可能逐样本变化；GT 是用于重建损失的参考输出，生成时的量化/observe 状态由 workflow 与模块约定共同约束。

## 调用链

```text
CLI parser_gen
  -> command workflow
       |
       +-- extract_ptq_data
       |     模型 + Pileval 校准样本
       |       -> embedding forward / block forward
       |       -> hook 捕获 norm 输出
       |       -> data_dir:
       |            block_<layer>_<target>_in.pkl
       |            position_ids.pkl / position_embeddings.pkl / attention_mask.pkl
       |
       +-- ptq
       |     同一模型 + data_dir + 单 quant_target
       |       -> adapter 构建 quant block，拆 PtqUnit
       |       -> provider 加载 unit 输入和 kwargs
       |       -> 关闭量化但开启 observe（算法 calib_forward/observe_input 生效），用原始 module forward 生成 GT
       |       -> DataLoader 组织输入和 GT
       |       -> BlockwiseSolver 求解 -> 保存 unit PTQ 参数
       |
       +-- eval
       |     模型 + Wikitext 输入
       |       -> embedding/block forward -> head -> PPL
       |       -> bf16：原始 block；quant：重建 quant block，按 bit policy 开关
       |
       +-- deploy
             模型权重 + 量化配置 +（按需）PTQ 参数
               -> 重建 adapter/dtype/algorithm 状态
               -> blockwise 或 tensorwise 导出
               -> safetensors + index.json + config.json
```

四个 workflow 是独立 CLI 入口，不是强制串行流水线：BF16/直转 eval 不需要 extract 或 ptq；deploy 也不负责重新生成校准数据。

## 实现定位

### 主要入口

`amct_pytorch/cli/llm/` 与 `amct_pytorch/workflows/`；阶段级稳定入口只保留 `LlmExtractPtqDataWorkflow`、`LlmPtqWorkflow`、`LlmEvalWorkflow`、`LlmDeployWorkflow`。

### 关键符号

从 `parser_gen`、`LlmEvalWorkflow`、`LlmExtractPtqDataWorkflow`、`LlmPtqWorkflow`、`LlmDeployWorkflow` 沿调用定位。组件重点搜索 `BaseModel`、`PtqUnit`、`LlmPtqDataProvider`、`BlockwiseSolver`、`PtqParamStore` 和相应 registry；注册定义必须与实际注册入口一起核对。分层边界以 [repo-map.md](../../../docs/repo-map.md#分层边界) 为准。

### 易变实现检查点

首轮定位四个入口后，再沿当前调用关系查找：

| 要查的问题 | 当前源码检查方向 |
| --- | --- |
| 参数和阶段分派 | `parser_gen`、workflow 的 `run/setup` 和 granularity 分支 |
| 输入保存/加载 | `amct_pytorch/common/models/llm/common/base.py` 与 `amct_pytorch/common/datasets/ptq_io.py` |
| 捕获点和 kwargs | `amct_pytorch/common/models/llm/common/capture.py`、adapter 的 embedding/block forward |
| GT 和 batch | `amct_pytorch/common/datasets/ptq_provider.py`、PTQ workflow 的 unit 准备逻辑 |
| unit 边界 | adapter 的 `build_quant_block`、`iter_ptq_units`、`PtqUnit` |
| 参数求解/回载 | `amct_pytorch/common/optimization/`、`ptq_params.py`、eval/deploy 的加载逻辑 |
| PPL 输入 | `amct_pytorch/common/datasets/preproc.py`、`amct_pytorch/common/evaluate/eval_ppl.py` |
| 导出产物 | `llm_deploy.py` 与 `amct_pytorch/common/models/llm/common/deploy_export.py` |

入口移动后按四个 workflow 和 producer/consumer 关系重定位。

### 验证入口

测试定位与命令见 [validation.md](validation.md#场景测试索引)，覆盖要求见本文件“自验”。

## 当前事实

作为判断依据前沿下表的生产者/消费者符号现查，与源码冲突时以源码为准（判定见 [conditional-rules「当前事实与核对命令纪律」](conditional-rules.md#当前事实与核对命令纪律)）。

### 阶段影响检查

阶段只描述 LLM 数据流影响，不决定代码职责归属。仅在修改 LLM CLI/workflow、阶段产物或跨阶段约定时检查；classic、ops 和纯 public API/packaging 标记为“不适用”。

只做阶段影响检查时读本节即可，需要逐阶段核对生产者/消费者约定再读下表与其后各节。通用能力可能同时影响多个阶段，不要强行选择单阶段；列出需要闭环的受影响阶段即可。

| 阶段 | 生产者 | 数据/状态 | 消费者 | 开发时必须保持 |
| --- | --- | --- | --- | --- |
| extract | `LlmExtractPtqDataWorkflow` + model adapter | 每层 unit 输入文件；attention 相关 kwargs 文件 | PTQ provider / adapter | hook 位置、`PtqUnit.kind`、文件命名、模型 `seq_len` 一致 |
| ptq | `LlmPtqWorkflow` + provider | GT、solver 返回的 unit 参数 | eval、deploy 的参数回载 | unit 枚举、输入 shape、GT 状态、参数命名和回载结构一致 |
| eval | `LlmEvalWorkflow` + Wikitext provider | PPL 和日志 | 精度判读 | BF16/quant 使用相同数据、`seq_len`、模型和评测口径 |
| deploy | `LlmDeployWorkflow` + deploy helpers | safetensors、index、`quantization_config` | 下游推理仓 | 权重替换、索引、配置和 dtype/algo payload 成对更新 |

### extract -> ptq 的真实细节

1. extract 从 Pileval 取得 `nsamples` 个 `seq_len` 样本；首层 `Catcher` 截获 embedding 后的 block 输入，并记录首次出现的 `attention_mask`、`position_ids`、`position_embeddings`。
2. 每层 `do_block_forward(..., hook_name=...)` 对指定 norm 注册 forward hook；hook 保存 norm 输出，文件名由 layer 和 target 组成。保存的是目标 unit 的输入，不是完整 decoder block 输出。
3. ptq 按 adapter 的 `iter_ptq_units()` 枚举 unit，provider 根据 unit 的 kind/layer 读取对应输入；attention 目标额外读取 kwargs 文件。
4. PTQ 当前要求单个 `quant_target`，主流程接通 `block` granularity。输入缺失时 provider 返回缺失状态，workflow/solver 不应静默把它当成功。

### 模型与数据流约定

模型 adapter 是这些数据约定的生产者，workflow/provider 是消费者，修改任一侧都要同步核对：

- `PtqUnit.kind` 必须对应 `block_<layer>_<target>_in.pkl` 的命名和 `load_ptq_inps()` 的读取参数。
- `unit.save_name` 必须与 PTQ 参数文件命名和 eval/deploy 回载结构一致。
- adapter 的 embedding/block hook 必须保存下游需要的 norm 输入及 `attention_mask`、`position_ids`、`position_embeddings` 等 kwargs。
- workflow 只消费稳定约定。

### ptq -> eval/deploy 的真实细节

1. PTQ 对每个 unit 生成 GT 时关闭量化但开启 observe（`set_model_to_observe(unit.module, True)`），算法的统计副作用会在此阶段发生；随后恢复量化状态交给 solver。GT 是 reconstruction 参考，不是 eval 的 PPL 标签；observe 统计是否污染或复用，按 [llm-algorithm](llm-algorithm.md) 的运行方式核对。
2. `DataLoader` 只对输入和 GT 建 batch；当前 provider 传给 solver 的 kwargs 是原对象。新增逐样本 kwargs 时，必须显式设计同步切 batch，不能假定现有逻辑自动完成。
3. solver 返回的结果由 PTQ workflow 按 unit 保存；eval/deploy 必须用兼容的 adapter、量化模块、算法状态和参数命名回载，不能只检查参数文件存在。

## 实现规则

1. 先确定改动属于四个 workflow 的哪一阶段，再画出受影响的生产者、产物和消费者。
2. 改 extract 或离线格式，必须同时检查 hook 捕获、文件写入、provider 读取和 unit 命名；只改一端会造成静态可写、运行不可读。
3. 改 PTQ 数据准备，必须同时核对 unit 枚举、原始输入边界、kwargs、GT 状态、batch 维度和参数保存。
4. 改 eval，先核对 Wikitext 切分、`seq_len`、BF16/quant 状态和 bit policy；不要用 extract 的 Pileval 产物替代 eval 输入。
5. 改 deploy，先确认上游参数/dtype/algo payload 已确定，再检查权重替换、index 和 config 的消费者格式。
6. 不在 workflow 中加入模型专属结构分支；模型输入捕获、norm 名、unit 拆分和 wrapper 逻辑归 adapter。
7. 新增 kwargs 或复合 unit 时，先定义字段所有权、作用域、shape/dtype、缺失行为和旧目录兼容策略，再改容器或 CLI 参数。

## PPL 口径

Wikitext PPL 默认 `seq_len=4096`。历史结果不是同口径时，只能参考，不能直接比较。

BF16 PPL 异常时第一优先级检查 mask/position/position_embeddings 传递链。

## 禁止项

- `extract_ptq_data` 与 PTQ 的 `quant_target`、`seq_len`、block 范围或 `data_dir` 不一致。
- 保存端和加载端的 unit kind/命名不一致，或 unit 输入文件缺失仍继续执行。
- hook 保存的是 norm 输出，却在完整 block 中再次执行同一 norm。
- 逐样本 kwargs 未随输入 batch 同步切分。
- GT 生成时量化未关闭（observe 开启是当前预期行为），或 GT 与量化输出边界不一致。
- BF16 PPL 异常却未检查 mask/position 传递链。
- eval/deploy 只发现参数文件存在，未证明 adapter、量化状态和参数结构可回载。

## 自验

测试定位与命令见 [validation.md](validation.md#场景测试索引)；数据流问题按下面的顺序定位。

### 问题定位顺序

```text
PPL 异常？
  -> 先确认 eval 数据集/seq_len/shift
  -> 再查 BF16 与 quant 的状态和 bit policy
  -> attention 模型再查 mask/position 是否从 adapter 传到 block

PTQ loss/输入异常？
  -> unit 是否被 adapter 枚举
  -> 对应离线文件是否存在且 shape 对齐
  -> kwargs 是否适用于当前 batch
  -> GT 是否在关闭量化（observe 开启）的原始 module 上生成
  -> solver 是否收到非空 batch 并保存结果

Deploy 产物异常？
  -> 参数是否能按同一 unit/target 回载
  -> dtype/algo payload 是否与 quant module 一致
  -> 权重替换、index.json、config.json 是否同时刷新
```

证据必须分级：源码调用链、单测覆盖、真实离线产物、真实模型/NPU 运行不是同一结论。只有前两级时，应写“静态链路一致”或“建议验证”，不能写“PTQ 已跑通”。
