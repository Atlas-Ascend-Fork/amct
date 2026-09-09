# AMCT LLM CLI 架构

用于理解 LLM CLI 的整体组成、职责和协作关系，不是编码步骤或术语词典。不覆盖 classic eager/graph_based API，也不描述量化实验的 skill 编排。

## 组件与职责

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

`PtqUnit` 是 adapter 暴露给 PTQ 的处理单元，不是独立服务；offline data 是阶段产物，不是组件。kwargs 是 forward 附加输入，可能共享也可能逐样本变化；GT 是用于重建损失的参考输出，生成时的量化/observe 状态由 workflow 与模块契约共同约束。

## 命令与产物

```text
模型 + 校准样本 -> extract_ptq_data -> 离线 unit 输入/上下文
模型 + 量化配置 + 离线数据 -> ptq -> PTQ 参数

模型 + 评测数据 + 评测配置 [+ PTQ 参数] -> eval -> PPL/日志
模型 + 量化配置           [+ PTQ 参数] -> deploy -> 部署模型目录
```

- 四条命令是独立入口，不是必须连续执行的流水线；BF16 评测和直转量化无需先运行 PTQ。
- extract 负责捕获和保存输入；PTQ 通过 adapter 构建 unit、provider 准备输入/GT、solver 优化，再由 workflow 保存返回结果。保存与回载是协作契约，不代表当前统一经过同一个存储接口。
- eval/deploy 在需要 PTQ 状态时重建兼容的模型、量化模块和算法结构并回载参数；deploy 产出部署权重和配置，不执行下游推理服务。
- 当前 extract/PTQ 只支持单个 `quant_target`，PTQ 主链接通的是 `block`；不能从 modelwise 占位分支或 solver 文件存在推断能力可用。

## 定位与深入阅读

主要入口是 `amct_pytorch/cli/llm/` 和 `amct_pytorch/workflows/`；从 `parser_gen`、`LlmEvalWorkflow`、`LlmExtractPtqDataWorkflow`、`LlmPtqWorkflow`、`LlmDeployWorkflow` 沿调用定位。组件重点搜索 `BaseModel`、`PtqUnit`、`LlmPtqDataProvider`、`BlockwiseSolver`、`PtqParamStore` 和相应 registry；注册定义必须与实际注册入口一起核对。

- 数据与阶段编排：[llm-data-workflow.md](llm-data-workflow.md)。
- 模型与量化计算：`model-adapter`、[quant-dtype-modules.md](quant-dtype-modules.md)、[llm-algorithm.md](llm-algorithm.md)。
- 优化与产物：[ptq-solver-params.md](ptq-solver-params.md)、[deploy-export-impl.md](deploy-export-impl.md)。

本文件提供粗粒度架构视图；跨 skill 的稳定边界以 [repo-map.md](../../../docs/repo-map.md#分层边界) 为准，当前接通行为仍需核对源码。不复制模型名单、字段清单或测试路径。
