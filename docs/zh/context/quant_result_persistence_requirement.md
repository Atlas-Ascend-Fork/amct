# Classic 量化结果持久化与恢复需求说明

## 1 文档目的

这份文档说明 Classic 量化结果保存与恢复功能解决什么问题、什么时候使用、需要准备什么，以及最终能得到什么。

> 当前状态：本文描述的是 [PR #267](https://gitcode.com/cann/amct/merge_requests/267) 中的候选实现，尚未合入 master，也未随任何已发布 AMCT 版本提供。仓库 master 当前仍不支持 `quant_result_dir`。

面向使用 AMCT 的量化工程师、测试人员和部署人员。具体的接口、文件字段和模块设计见《软件设计说明书》；测试步骤和执行结果见《测试用例及测试报告》。

### 概览

这个功能的核心是：把量化结果做成一个可检查、可恢复的交付包。

1. **生产端。** 浮点模型完成量化后，按层写出 `safetensors`，并记录每个文件的大小、摘要和参数说明。
2. **`quant_manifest.json` 是交付完成清单。** 它只有在所有应保存的层都通过复验后才出现。
3. **消费端。** 新进程拿到同源浮点模型和目录，先检查身份、文件和参数，再一次性恢复量化模块。
4. **恢复后继续走原有 deploy 流程。** artifact 是中间结果，不是最终 NPU 二进制，也不保存原始模型或校准数据。
5. **标准 API 文档同步。** `quantize()`、`convert()` 的参数、目录契约和默认行为以 `docs/zh/api/` 下的正式接口文档为准。

![图 1：Classic artifact 端到端流程](../figures/classic-artifact-overview.png)

图中下方虚线表示不传 `quant_result_dir` 时的旧 API 路径；这条路径仍然需要回归测试。橙色提示表示使用边界，绿色提示表示可以直接验收的结果。

## 2 特性概述

### 2.1 要解决的问题

在原有流程中，量化参数通常只存在于当前 Python 进程的模型对象里。进程结束、换到另一台机器，或者需要重新执行部署转换时，已经完成的校准结果不能直接拿来使用，往往只能重新校准。

这个功能把一次量化得到的结果保存成一个可以检查的目录。之后，用户可以用同一个来源的浮点模型读取这个目录，恢复量化状态，再继续生成部署模型。

### 2.2 功能目标

用户完成一次量化和校准后，可以选择保存结果；另一个进程拿到原始浮点模型和结果目录后，可以恢复结果并完成部署转换。没有选择保存目录时，原有 Classic 调用方式和行为保持不变。

### 2.3 适用范围

- Classic Python API 的量化结果保存、恢复和部署转换。
- 当前版本已支持的 MinMax、AWQ、GPTQ 代表性量化路径。
- 按量化层保存结果，支持多层模型和嵌套层。
- 使用 JSON 记录目录清单，使用 safetensors 保存张量数据，便于完整性检查和跨进程使用。

### 2.4 直接路径和 artifact 路径的区别

| 比较项 | 直接路径 | artifact 路径 |
|------|----------|---------------|
| 输入 | 浮点模型、配置、校准输入 | 生产端的浮点模型、配置、校准输入；消费端再提供同源浮点模型和结果目录 |
| 中间状态 | 只留在当前 Python 进程 | 按层保存到目录，并由 manifest 标记是否完整 |
| 是否可换进程 | 不能直接复用内存中的量化状态 | 可以把目录交给另一个进程继续 reload |
| 失败影响 | 由当前调用栈处理 | Reader 在修改模型前完成校验，失败时拒绝目录 |
| 结果 | 直接进入原有转换流程 | reload 成功后进入同一个原有转换流程 |

这张对照表用于界定需求：本功能增加的是“量化结果可交付、可检查、可恢复”的能力，不改变部署算子本身的计算定义。

### 2.5 关联验证：Qwen3.8-27B 纯 Classic Python API 实测

Classic 文档的主对象是 Classic Python API。为确认该 API 能在真实大模型和两卡 NPU 上工作，另用 `Qwen/Qwen3.8-27B` 做了纯 API 实机验证：不调用 AMCT CLI、`parser_gen` 或 LLM workflow，直接构建完整 `nn.Module`，走 `algorithm_register → quantize → artifact → fresh-process convert → PPL/generation`。这组结果作为大模型关联验证记录，不计入 Classic FR-001～FR-008 的仓库自动化回归数量。

| 验证项 | 实际做法 | 结果 | 结论边界 |
|--------|----------|------|----------|
| 模型完整性和设备分布 | 校验 33 个文件、18 个 safetensors、1,199 个 tensor，加载 27.78B BF16 `nn.Module`；其中 31 个 decoder layer 放 `npu:0`、33 个 decoder layer 放 `npu:1`（共 64 个 decoder layer） | **PASS**；总字节数 `55,586,114,768`，1,199/1,199 tensor，496 个 language Linear；checkpoint 结构记录见实机测试日志 | 证明纯 API 实测使用的 checkpoint 完整、两卡模型结构可构建 |
| BF16 baseline | 同一 31/33 两卡 `nn.Module` 上执行 WikiText-2 72×4096 PPL 和短文本生成 | API BF16 PPL `6.9534372879` | 作为纯 API 量化结果的对照基线 |
| W8A8 量化与恢复 | 通过 `algorithm_register` 和 `quantize` 生成 496 unit artifact；producer 与 deploy/PPL 分进程，fresh process 校验 checkpoint fingerprint 后 reload，再调用 `amct_pytorch.convert` | 496 unit **PASS**；PPL `7.0308233570`；GSM8K batch-8 8/8 **PASS**；496 个 W8 deploy module | 证明两卡纯 API artifact 生产、跨进程恢复和部署转换可运行 |
| mixed W4A8 量化与恢复 | 400 个 MLP/W4 unit + 96 个主 projection/W8 unit；两者合计 496 个 language Linear；fresh process reload 后 convert，并执行 PPL 与 generation gate | 496 unit **PASS**；PPL `7.1221962920`；single-record GSM8K **PASS**；400 packed W4 + 96 W8 deploy module | 证明 mixed policy 可通过纯 API 路线落盘和恢复 |
| 量化算法与运行时边界 | 使用 11 点、1.0-0.5 diagonal-Hessian clipping search；mixed W4A8 采用 packed W4 `npu_weight_quant_batchmatmul`，A8 为数值上的 dynamic per-token quantize/dequantize | W8A8 和 mixed W4A8 均可执行 | 该算法是 diagonal-Hessian GPTQ clipping 近似，不是 full GPTQ；mixed 路径不宣称物理 INT8 x INT4 kernel |
| 正式 accuracy benchmark | 15,268 prompt 的 generation/scoring 流程单独执行 | 未把部分 JSONL 包装成完整结果 | generation gate 只证明机制，不替代正式 accuracy benchmark |

这组验证说明：Classic API 不仅能在 tiny 模型上验证协议，也能在 Qwen3.8-27B 两卡真实模型上完成量化、artifact 保存、跨进程恢复、部署转换、PPL 和 generation gate。需要注意，实测算法是 diagonal-Hessian GPTQ clipping 近似，不是 full GPTQ；正式 15,268 样本 accuracy benchmark 仍需单独完成后才能作为发布结论。

### 2.6 三个关键决策的验收边界

![图 2：Qwen 纯 API 实测的关键决策和边界](../figures/qwen-api-decision-boundaries.png)

#### 为什么选择 mixed W4A8

mixed W4A8 不是“为了低比特而低比特”，而是在显存和精度之间做一个可解释的折中：

| 方案 | 覆盖范围 | PPL | 结论 |
|------|----------|-----|------|
| API BF16 | 全部 language Linear 保持 BF16 | `6.9534372879` | 对照基线 |
| W8A8 | 496 个 language Linear 使用 W8 | `7.0308233570` | 精度接近 BF16，显存仍较高 |
| mixed W4A8 | 400 个 MLP Linear 使用 W4，96 个主 projection Linear 使用 W8 | `7.1221962920` | 相比 W8A8 少约 `9.48 GiB` 部署显存，PPL 增加 `0.091372935` |

选择 400 个 MLP Linear 为 W4、96 个主 projection Linear 为 W8 的 mixed policy，是因为 MLP Linear 数量多，使用 W4 能拿到主要显存收益；主 Q/K/V projection 位于注意力主路径，保留 W8 可以降低精度风险。该策略是本轮实测的选择，不表示所有模型和所有部署环境都应采用同一比例。

#### 为什么不能宣称 physical INT8 x INT4 kernel

mixed W4A8 的权重是 packed W4，activation 使用 dynamic per-token INT8 数值行为；但部署算子的执行顺序是：

```text
activation
  -> dynamic INT8 quantize
  -> dequantize 回 BF16
  -> npu_weight_quant_batchmatmul(BF16 activation, packed W4 weight)
```

因此 matmul 实际收到的 activation operand 是 BF16，不是 INT8。该实现保留 W4 权重存储和 A8 数值行为，但没有使用 physical INT8 x INT4 matmul kernel，也没有 profiling 证据证明这样的 kernel。文档只能写“W4A8 numerical behavior + packed W4 storage”，不能写“physical INT8 x INT4 kernel”。

#### 为什么本轮不使用 full GPTQ

本轮 Qwen 实测的目标是验证 Classic Python API 的量化、artifact 落盘、跨进程恢复和部署链路，不是完成 27B full GPTQ 算法评测。当前实现只收集 diagonal Hessian，搜索每行 `clip_factor`，artifact 只保存 `clip_factor`；它不做 full Hessian inverse，不做逐权重误差补偿，也不保存 full GPTQ 优化后的 weight。

full GPTQ 需要完整 Hessian 或 blockwise Hessian、误差补偿、优化后 weight 持久化和单独的精度/性能验证。把这些混进本轮会引入多个变量，失败时难以判断是 artifact 协议问题还是算法问题。因此本轮将其明确排除，并将实测算法命名为 diagonal-Hessian GPTQ clipping approximation。

#### 测试完整性的判断

| 验收问题 | 当前答案 |
|----------|----------|
| Classic artifact 机制在声明范围内是否完整 | 是；tiny 回归、Qwen 纯 API 496 unit、fresh-process convert、PPL 和 generation gate 均有证据 |
| Qwen 是否证明 27B full GPTQ | 否；本轮是 diagonal-Hessian clipping approximation |
| Qwen 是否证明 physical INT8 x INT4 kernel | 否；activation 在 matmul 前已回到 BF16 |
| Qwen 是否证明正式发布精度 | 否；15,268 样本 accuracy benchmark 仍需单独完成 |

这个边界的意义是：结论可以复核，但不能把小范围机制验证扩大成产品发布结论。


## 3 使用场景

### 场景一：量化完成后换进程部署

用户在校准机器上完成量化，把结果目录交给部署机器。部署机器重新加载同一个浮点模型，读取结果目录后直接生成部署模型，不再重复校准。

### 场景二：量化完成后需要交付结果

用户希望把本次量化结果作为一个明确的交付物保存下来。只有所有应保存的量化层都完成并通过检查，结果目录才对外有效；中途停止或写入不完整的目录不能作为交付物使用。

### 场景三：复现和比较量化结果

用户保留浮点模型、配置和结果目录，在相同输入下重新生成部署模型，检查本次结果是否能够稳定复现。

## 4 功能需求：从量化结果到部署模型的完整流程

本节只有一个端到端功能。下面的 FR 编号是这个功能的验收点，供设计文档和测试文档追踪。

### 4.1 用户输入

量化时，用户提供：

1. 一个尚未量化的浮点模型。
2. 一份量化配置，至少说明要量化的层和使用的算法。
3. 用于校准的输入数据，数据格式和模型原有输入要求一致。
4. 可选的结果目录路径。未提供该路径时，继续使用原有的内存流程。

恢复和部署时，用户提供：

1. 与量化时来源一致的浮点模型。模型类型、数据类型和需要恢复的层应保持一致。
2. 已完成发布的量化结果目录。

### 4.2 用户可观察的处理过程

1. AMCT 根据配置和校准数据完成量化。
2. 如果用户提供了结果目录，AMCT 为每个已完成的量化层保存结果，并在全部结果检查通过后发布目录清单。
3. 用户在新的进程中提供浮点模型和结果目录。
4. AMCT 先检查结果目录是否完整、是否与当前模型匹配，再恢复各量化层的状态。
5. 恢复成功后，AMCT 继续使用原有的部署转换流程生成部署模型。
6. 任一检查失败时，AMCT 明确报告原因，不使用不完整结果，也不把模型留在只替换了一部分层的状态。

### 4.3 用户得到的结果

- 量化阶段：一个可交付的结果目录。目录中包含每层量化结果文件和一份说明目录完整性的清单文件。
- 恢复阶段：当前进程中的模型恢复到已校准状态，可以继续调用原有部署转换能力。
- 部署阶段：与直接量化后转换相同用途的部署模型，部署权重、量化参数和推理结果应保持一致。
- 失败阶段：一个可定位的错误，且原始模型对象不被部分修改。

### 4.4 典型使用方式

用户不需要改变原有的量化步骤，只需在量化时指定一个结果目录：

```text
第一次进程：浮点模型 + 量化配置 + 校准输入
          -> 完成量化，并将结果保存到指定目录

后续进程：同源浮点模型 + 结果目录
          -> 恢复量化结果，再执行原有的部署转换
```

在 Classic API 中，这表示量化调用增加可选的 `quant_result_dir` 目录参数；部署转换调用也可以接收同一个结果目录。这里的目录参数是可选的，不传参数时不启用结果保存和恢复。

不指定结果目录时，仍按原有方式在当前进程中完成量化和转换。

### 4.5 端到端验收点

| 编号 | 验收点 | 用户应看到的结果 |
|------|--------|------------------|
| FR-001 | 原有调用兼容 | 不提供结果目录时，原有量化和转换流程仍可运行，且不会额外生成结果文件。 |
| FR-002 | 分层保存 | 提供结果目录后，每个应保存的量化层都有对应结果文件，结果可用于后续恢复。 |
| FR-003 | 完整发布 | 只有所有应保存的层都写入并检查通过，结果目录才出现有效的完成清单。 |
| FR-004 | 支持的算法可恢复 | MinMax、AWQ、GPTQ 的已保存结果可以从同源浮点模型恢复。 |
| FR-005 | 失败不留下半量化模型 | 目录缺失、损坏、与模型不匹配或恢复中出错时，模型不会只替换一部分层。 |
| FR-006 | 恢复后可以部署 | 读取结果目录后可以继续执行原有部署转换；直接转换和恢复后转换的结果一致。 |
| FR-007 | 输入有明确边界 | 不支持的算法、模型或损坏的结果会被拒绝，并提示用户修正输入，不会静默按另一种算法处理。 |
| FR-008 | 结果可验证 | 交付时有对应的单元测试、集成测试和端到端测试记录，能够说明功能是否真正跑通。 |

### 4.6 每条需求到底要证明什么

下面把验收表翻译成开发者执行时可以直接检查的动作。测试编号对应测试报告中的 TC-001～TC-035。

| 需求 | 说明 | 主要操作 | 通过时应看到 | 关联设计 / 测试 |
|------|----------|----------|--------------|----------------|
| FR-001 | 不开启新功能，旧代码会不会被改变？ | 不传 `quant_result_dir`，执行原来的 `quantize`、`convert`，检查目录和模型签名 | 不创建 artifact，不安装 Writer hook，旧流程继续运行 | `quantize.py`；TC-014～TC-015 |
| FR-002 | 每个量化层的结果有没有真正保存？ | 让 MinMax、AWQ、GPTQ 在各自定稿点导出参数，读取 unit 文件 | 每个预期层都有对应 `safetensors`，参数 key、shape、dtype 可读 | Writer / BaseQuantizeModule；TC-005～TC-008、TC-016～TC-019 |
| FR-003 | 目录什么时候才算交付完成？ | 只写一部分 unit、篡改文件，再调用发布和 Reader | 部分目录没有 manifest；完整目录才发布；摘要不一致会拒绝 | manifest-last；TC-009～TC-013 |
| FR-004 | 换到新进程后，三种算法能否恢复？ | 用 fresh float model 读取同源 artifact，检查模块类型和参数 | MinMax、AWQ、GPTQ 都能重建，且不重新校准或搜索 | Reader / 注册表；TC-020～TC-023 |
| FR-005 | 中途出错会不会留下半量化模型？ | 删除、截断或修改第二层 unit，在多层模型上触发 reload | 抛出可定位错误，所有 live layer 保持原对象和原 state | 两阶段事务；TC-024～TC-030 |
| FR-006 | 恢复之后能不能继续部署？ | 对同一输入比较 direct convert 和 artifact reload 后 convert | 两条路径都能进入原有 deploy pass，输出在声明边界内一致 | ReplaceNpuQuantModulePass；TC-031～TC-034 |
| FR-007 | 不支持的输入是否会被明确拒绝？ | 修改算法名、模块类型、model dtype、文件摘要或目录类型 | 失败原因明确，不静默切换算法、不部分修改模型 | schema / registry；TC-003～TC-004、TC-027～TC-029 |
| FR-008 | 测试结论能否被别人复核？ | 保存命令、环境、commit、实际输出和限制 | 文档能从需求追到测试、证据和未执行项 | 三份文档追踪；TC-035 |

## 5 使用限制和不在范围内的内容

### 5.1 使用限制

- 恢复必须使用与保存结果时相同来源的浮点模型，不能把结果目录直接套到任意结构相似的模型上。
- 结果目录必须完整、可读且未被修改。缺少清单或清单列出的文件损坏时，整个目录视为无效。
- 结果目录保存的是量化后的参数和必要说明，不保存训练数据，也不替代原始浮点模型。
- MinMax、AWQ、GPTQ 以当前版本已注册并经过测试的量化层为准；未声明支持的量化层不能对外承诺。

### 5.2 不在本需求内

- 逐层量化入口和校准数据编排。这些属于后续工作。
- 命令行入口、`parser_gen` 和 LLM workflow 编排。本需求只覆盖 Classic Python API。
- 将 Qwen3.8-27B 实机结果直接计入 Classic FR-001～FR-008 的自动化回归数量；Qwen 的正式 accuracy benchmark、性能调优和发布实验仍单独管理。
- 重新训练模型、重新执行 AWQ 搜索或 GPTQ 优化，以及新增量化算法。
- 修改下游硬件算子或部署运行时。

## 6 基本数据流

```text
浮点模型 + 量化配置 + 校准输入
              |
              v
        Classic 量化与校准
              |
              +-->（可选）量化结果目录
              |
              v
          部署转换
              |
              v
          部署模型

新的浮点模型 + 已完成的量化结果目录
              |
              v
       检查并恢复量化状态
              |
              v
          原有部署转换
              |
              v
          部署模型
```

结果目录是中间交付物，不是新的模型格式。它记录的是让 AMCT 恢复量化状态所需要的结果；部署模型仍由原有部署转换流程生成。

![图 3：需求、设计、测试和证据的对应关系](../figures/requirement-design-test-trace.png)

图中的“证据”必须能回到一个实际命令、一个测试用例或一个可检查文件。只写“测试通过”而没有命令、输入和限制，不足以作为验收记录。

## 7 验收标准

功能达到以下条件后，本需求视为完成：

1. 不带结果目录的旧 Classic 流程通过回归测试，行为没有变化。
2. 带结果目录完成一次量化后，目录中存在每层结果文件和有效完成清单。
3. 使用同源浮点模型在新进程中读取该目录，可以恢复 MinMax、AWQ、GPTQ 代表性结果。
4. 恢复后可以正常执行原有部署转换，不需要重新校准或重新搜索量化参数。
5. 直接量化后转换，与保存、重新加载后再转换，在部署参数和相同输入的输出上保持一致。
6. 缺少文件、文件被修改、模型不匹配或输入不支持时，流程明确失败，模型不发生部分替换。
7. 单元测试、集成测试和端到端测试均有命令、环境、提交版本和实际结果记录。真实 NPU 验证受环境条件限制时，必须在测试报告中单独说明，不能用模拟结果代替 NPU 结论。

### 7.1 明确的使用边界

以下情况不能按本需求的成功路径理解：

- **不能把 artifact 套到任意结构相似的模型。** Reader 至少要检查来源类型、dtype、受管层路径、模块类型、参数 shape 和 dtype。
- **不能把半成品目录当成结果。** 没有 manifest，或者 manifest 列出的文件缺失、截断、摘要不匹配，都必须失败。
- **不能把 artifact 当最终部署文件。** 它保存的是恢复量化状态所需的张量和元数据，最终设备模块仍由原有 deploy pass 生成。
- **不能把已经量化的模型再次传给 fresh-model reload 路径。** 该路径的输入契约是同源、尚未插入量化模块的浮点模型。
- **不能用 tiny 或 mock 结果替代正式大模型精度结论。** 本文只记录已执行的 tiny、协议、事务、可用 NPU 和 Qwen 纯 API baseline 结果；正式 accuracy benchmark 需要单独冻结模型、数据和环境。

## 8 术语

| 术语 | 含义 |
|------|------|
| Classic API | AMCT 面向 Python 模型对象的传统量化调用入口。 |
| 量化结果目录 | 保存一次量化结果的目录，包含每层结果文件和完成清单。 |
| 量化层 | 模型中被量化并需要保存或恢复参数的层。 |
| 浮点模型 | 尚未经过本功能量化替换的原始模型。 |
| 部署模型 | 完成部署转换、供后续推理或下游转换使用的模型。 |
| manifest | 结果目录的完成清单，用来说明目录包含哪些结果以及结果是否完整。 |
| safetensors | 用于保存张量的文件格式；本功能用它保存量化结果，不使用 pickle 执行反序列化。 |
| MinMax | 根据权重或激活值范围计算量化参数的算法。 |
| AWQ | 根据激活信息调整权重量化的算法。 |
| GPTQ | 通过误差补偿优化权重量化的算法。 |

## 9 依据和关联文档

- PR #267 的需求、实现和测试记录。
- 《软件设计说明书_Classic量化结果持久化与恢复》：说明内部接口、数据结构和模块协作。
- 《测试用例及测试报告_Classic量化结果持久化与恢复》：说明每个验收点的测试方法和实际结果。
- Qwen3.8-27B 纯 Classic Python API 两卡实机测试记录：说明真实模型加载、量化、artifact、fresh-process convert、PPL 和 generation gate。
