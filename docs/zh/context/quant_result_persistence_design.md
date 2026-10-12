## 缩略语清单

表 2 缩略语清单

| 缩略语 | 英文全名 | 中文解释 |
|--------|----------|----------|
| AMCT | Ascend Model Compression Toolkit | 昇腾模型压缩工具包 |
| API | Application Programming Interface | Python 程序接口 |
| PTQ | Post-Training Quantization | 训练后量化 |
| NPU | Neural Processing Unit | 昇腾处理器 |
| JSON | JavaScript Object Notation | manifest 文本格式 |
| SHA-256 | Secure Hash Algorithm 256-bit | 文件完整性摘要 |
| UUID | Universally Unique Identifier | artifact 唯一标识 |
| UT | Unit Test | 单元测试 |
| E2E | End to End | 端到端测试 |
| HF | Hugging Face | 模型配置来源之一 |

## 1 简介

### 1.1 目的


1. Classic Python API 如何从量化结果导出、文件协议和恢复边界推导设计约束。
2. 生产端如何在不破坏旧 API 的情况下接入结果写端。
3. 消费端如何从结果目录安全恢复模块并继续部署。
4. 每个代码模块、数据字段、异常和测试如何对应需求。

### 1.2 范围

#### 1.2.1 软件名称

AMCT PyTorch Classic quant result artifact（Classic 量化结果持久化与恢复）。

#### 1.2.2 软件功能

纳入范围：

- `quantize(model, config=None, quant_result_dir=None)` 的可选量化结果落盘。
- MinMax、AWQ、GPTQ 三类已声明能力的量化模块导出和加载。
- 每层 safetensors、canonical JSON、SHA-256、entry digest 和 manifest-last。
- `reload_quant_params(model, quant_result_dir)` 的严格校验和两阶段事务替换。
- `convert(model, quant_result_dir=...)` 先 reload、后复用既有 deploy pass。
- `quant_result_dir=None` 的旧路径兼容。

不纳入范围：

- 逐层量化工作流的 `pipeline="layerwise"`、Adapter、Extract、Solver 和本地 calibration data。
- 命令行入口、`parser_gen`、LLM workflow 编排、Qwen 性能研究和发布实验。

### 1.3 关联验证：Qwen3.8-27B 纯 Classic Python API 实测

Classic artifact 的实现以通用模块和文件协议为边界。Qwen3.8-27B 真实大模型验证只走纯 Classic Python API：不调用 AMCT CLI、`parser_gen` 或 LLM workflow，直接在两卡 NPU 上构建完整 `nn.Module`，走 `algorithm_register → quantize → artifact → fresh-process convert → PPL/generation`。这条路线直接检查 Classic public API 与 artifact 语义，但不计入 Classic 单元测试数量。

![图 11：Qwen3.8-27B 纯 Classic Python API 验证链路](../figures/qwen-real-model-validation.png)

| 验证对象 | 设计关注点 | 实测结果 |
|----------|------------|----------|
| 完整模型和设备设计 | 33 个文件、18 个分片、1,199 个 tensor、27.78B BF16 `nn.Module`；31 层在 `npu:0`，33 层在 `npu:1` | 1,199/1,199 tensor、496 个 language Linear，模型结构可构建 |
| BF16 baseline | 同一 31/33 两卡结构执行 WikiText-2 72×4096 PPL | API BF16 PPL `6.9534372879` |
| W8A8 纯 API 链路 | `algorithm_register` → `quantize` → 496 unit artifact → fresh-process reload → `amct_pytorch.convert` → PPL/generation | 496 unit PASS；PPL `7.0308233570`；GSM8K batch-8 8/8 PASS；496 个 W8 module |
| mixed W4A8 纯 API 链路 | 400 个 W4 unit + 96 个主 Q/K/V projection W8 unit；fresh-process reload 后 convert | 496 unit PASS；PPL `7.1221962920`；single-record GSM8K PASS；400 packed W4 + 96 W8 module |
| 运行时和算法边界 | diagonal-Hessian clipping 近似；packed W4 `npu_weight_quant_batchmatmul`，A8 数值 quantize/dequantize | 可执行，但不是 full GPTQ，也不宣称物理 INT8 x INT4 kernel |
| 正式 accuracy benchmark | 15,268 prompt generation/scoring 单独执行 | 未把部分 JSONL 包装成完整结果 |

Qwen 纯 API 实测说明底层 artifact 协议可以落到真实模型，两卡 public API 链路能够完成量化、落盘、跨进程恢复和部署转换；但该实测算法是 diagonal-Hessian GPTQ clipping 近似，不能据此宣称 full GPTQ、所有算法或正式 accuracy benchmark 已经通过。

### 1.4 设计决策：为什么这么实现

![图 12：Qwen 纯 API 实测的关键决策和边界](../figures/qwen-api-decision-boundaries.png)

#### mixed W4A8 的设计取舍

Qwen 纯 API baseline 覆盖 496 个 language Linear。mixed policy 将 400 个 MLP Linear 设为 W4，将 96 个主 Q/K/V projection 设为 W8。这样设计的理由是：MLP 数量占多数，W4 能带来主要显存收益；主 Q/K/V projection 是注意力主路径，保留 W8 可以降低精度风险。实测结果支持这个取舍：mixed W4A8 相比 W8A8 部署显存少约 `9.48 GiB`，PPL 增加 `0.091372935`。

这不是通用最优策略。换模型、换层分布、换 CANN 版本或换精度目标时，必须重新评估 W4/W8 分配。

#### W4 运行时 ABI 的设计边界

CANN 9.1 在当前 910C 环境拒绝 generic `npu_quant_matmul` 的 int4 dtype 路径。实现因此复用已验证的 packed W4 `npu_weight_quant_batchmatmul` ABI：

1. activation 执行 dynamic per-token INT8 quantize；
2. 立即用 scale dequantize 回 BF16；
3. BF16 activation 与 packed W4 weight 进入 `npu_weight_quant_batchmatmul`。

这个设计能保留 W4 权重存储和 A8 数值行为，但 matmul 的 activation operand 是 BF16。因此设计上不能声明 physical INT8 x INT4 kernel。若未来 CANN 提供真正 INT8 activation × INT4 weight 的 ABI，需要重新实现并重新测试。

#### 为什么本轮不用 full GPTQ

当前实验目标是验证 artifact 协议和 public API 链路。diagonal-Hessian clipping approximation 只需要保存每层 `clip_factor`，变量少、可复现、失败容易定位，适合作为链路验证算法。

full GPTQ 至少需要：

| full GPTQ 要求 | 当前实现是否具备 |
|----------------|------------------|
| full 或 blockwise Hessian | 否，只收集 diagonal Hessian |
| Hessian inverse / 误差补偿 | 否 |
| 逐权重顺序更新 | 否 |
| 保存优化后 weight | 否，只保存 `clip_factor` |
| 独立 27B 精度和性能验证 | 否 |

因此把当前实现称为 full GPTQ 会掩盖设计和测试边界。若后续要支持 full GPTQ，需要新增算法模块、扩展 artifact payload、重新验证 reload 语义，并单独执行 PPL、generation gate 和正式 benchmark。


## 2 第零层设计描述

### 2.1 软件系统上下文定义

图 2 Classic artifact 系统上下文（文字数据流）

```text
用户 Python 进程
  |
  | quantize(model, config, quant_result_dir)
  v
Classic quantize.py
  |-- parse_config -> AlgorithmRegistry
  |-- InsertQuantizeModulePass -> BaseQuantizeModule 子类
  |-- 定稿点 -> QuantResultWriter -> unit.safetensors
  `-- 全部 unit 复验 -> quant_manifest.json（最后发布）

另一进程 / fresh float model
  |
  | convert(model, quant_result_dir)
  v
load manifest -> 校验目录/模型/unit -> staging replacement modules
  -> 统一替换为 calibration quant modules
  -> ReplaceNpuQuantModulePass
  -> 原有 deploy module
```

![图 3：生产端、artifact 和消费端的结构关系](../figures/artifact-directory.png)

这张图补充了文字数据流没有表达出的“完成信号”：单个 unit 写成功，只能说明某一层可读；manifest 原子发布后，Reader 才可以把整个目录当成一个交付物。

表 3 外部实体与系统交互

| 外部实体 | 交互方式 | 输入 | 输出/约束 |
|----------|----------|------|-----------|
| 用户 Python 代码 | 调用 Classic API | model、config、目录 | 原地修改模型；返回值仍为 None 或原模型 |
| 文件系统 | 读写 artifact | 目录、JSON、safetensors | 目录必须可读写；manifest 是完成信号 |
| PyTorch | Module、Tensor、hook、device | nn.Module 和 tensor | 参数恢复到原层 device |
| safetensors | 安全 tensor 序列化 | CPU contiguous tensors | 不使用 pickle；header 只存交叉校验字段 |
| AlgorithmRegistry | 查找量化类 | algorithm、ori_module_type | 只能实例化当前安装中已注册的类 |
| NPU/torch_npu | 可选部署和前向执行 | deploy module、输入 tensor | NPU 测试依赖实际可用硬件 |

### 2.2 设计约束

#### 2.2.1 遵循标准

- 遵守仓库 Apache 2.0 代码头和 Python/ruff 规范。
- 遵守 requirement-development 文档模板。
- manifest 使用 UTF-8、排序 key、紧凑分隔符的 canonical JSON。
- digest 统一使用 `sha256:<64 位小写十六进制>`。

#### 2.2.2 硬件限制

- 量化算法可能在 CPU 或 NPU 上运行，恢复参数必须跟随原量化层 device。
- GPTQ Hessian 计算仍由既有算法负责；artifact 路径不能重新运行 Hessian。
- 真实 NPU 测试受 torch_npu/CANN 版本和设备可用性影响，环境失败必须单独记录。

#### 2.2.3 技术限制

- 不能 pickle load 不可信 artifact，也不能从 manifest 动态 import 任意 Python 类。
- `quant_result_dir` 必须是调用方已经创建的目录；Writer 不负责隐式 mkdir。
- 一个 writer 只允许写预期 layer 一次，不能覆盖已有 AMCT manifest/unit，也不支持弱 resume。
- 旧路径不计算 fingerprint、不创建 Writer、不安装发布 hook。

#### 2.2.4 系统配置

- 当前协议常量集中在 `amct_pytorch/classic/quant_result.py` 顶部。
- 当前 schema version 为整数 `1`。
- 当前 artifact 支持声明在 MinMaxQuant、LinearAWQuant、GPTQuant 类上，Base 默认为 False。
- 当前测试使用 tiny `nn.Linear` 模型、bf16/fp32 参数和仓库已注册算法；NPU 测试由 pytest 条件控制。

### 2.3 设计思路

#### 2.3.1 设计原则

优先级依次为：旧 API 兼容、结果可验证、失败不破坏模型、安全边界、复用既有算法和部署逻辑、最后才是扩展性。

#### 2.3.2 设计方法

采用“能力声明 + 注册表查找 + Writer/Reader 对称协议 + 两阶段事务”的组合：

- 能力声明解决哪些算法允许进入 artifact 路径的问题。
- 注册表解决字符串配置到实际量化类的受信任映射。
- Writer/Reader 共享常量和 schema，避免各写一套格式。
- staging/commit 使模型替换具备全有或全无语义。

#### 2.3.3 方案比较

表 4 关键方案比较

| 问题 | 未采用方案 | 采用方案 | 原因 |
|------|------------|----------|------|
| 参数保存 | `torch.save` 整个模型 | safetensors + JSON manifest | 不依赖 pickle；可逐层审计和校验 |
| 支持算法判断 | 入口维护字符串白名单 | 量化类元数据 + 注册表实际查找 | 避免注册表和白名单漂移 |
| 模型身份 | 完整权重/config 摘要强匹配 | model_type + torch_dtype，受管目标层逐层匹配 | 允许同量化身份的 checkpoint 复用；仍拒绝目标层类型、参数 shape/dtype 不匹配 |
| reload 替换 | 边读边 `setattr` | 全量 staging 后统一 commit | 防止部分量化模型 |
| deploy | 另写 artifact 专用转换器 | 复用 `ReplaceNpuQuantModulePass` | 保障两条路径行为一致 |

#### 2.3.4 软件复用

- 复用 `parse_config` 和 `AlgorithmRegistry.algo`。
- 复用 `InsertQuantizeModulePass` 执行原有模块插入。
- 复用 `BaseModuleFusionPass`/`ModuleHelper` 的模块定位和替换能力。
- 复用既有 MinMax、AWQ、GPTQ forward 和 deploy 计算。
- 复用 `amct_pytorch.common.utils.file_io` 中的目录校验、文件摘要和原子写入工具；`quant_result.py` 只保留协议、Writer 和 Reader 逻辑。
- 复用 `ReplaceNpuQuantModulePass`，不在 artifact 模块中生成第二套 deploy 算法。

#### 2.3.5 可扩展性和兼容性

新算法要接入 artifact，至少需要：

1. 在注册表中有可实例化的量化类。
2. 类明确声明 `supports_quant_result = True`。
3. 类实现非空且稳定的 `quant_param_keys()`。
4. 在自身定稿点调用基类 dump hook。
5. 在 `_on_quant_params_loaded()` 中验证必需参数并清理校准状态。

`quant_result_dir=None` 是旧行为兼容开关；协议演进不能靠悄悄接受未知字段，必须增加 schema version 或显式兼容层。
目录兼容规则同样是协议的一部分：Writer 只拥有本次 expected units 对应的文件和 `quant_manifest.json`；Reader 只消费
manifest 列出的文件。未列举文件可以共存，但不能补足缺失 unit，也不能改变 manifest 的完成语义。

#### 2.3.6 性能和可靠性

tensor 和文件摘要按块处理；Writer 不复制整个模型。reload 只 deep copy 被恢复的原层。manifest-last 使目录在
发布前不可加载；commit rollback 处理极少见的 `setattr` 异常。

#### 2.3.7 并发和进程

当前 P0 不承诺多个进程并发写同一结果目录。临时文件与目标文件同目录，使用进程号生成临时名；已有 managed path
会被拒绝。用户应为每次运行提供独立、版本化目录。

### 2.4 协议一致性问题设计

表 5 协议一致性问题

| 序号 | 协议名 | 章节 | 协议模糊点和可选项 | 目前实现说明 |
|------|--------|------|-------------------|-------------|
| 1 | quant result schema | source_model | 早期草案是否保存完整权重/config 摘要 | 当前只保存 model_type、torch_dtype；层结构和 tensor schema 单独校验 |
| 2 | quant result schema | manifest | producer/library_version 是否属于协议字段 | 当前删除；format/schema_version 是唯一协议来源 |
| 3 | quant result schema | safetensors header | 是否复制 format/schema/config | 当前只保存 artifact_id、layer_name、entry_digest，manifest 为唯一完整配置源 |
| 4 | algorithm capability | support boundary | 注册算法是否自动等于可导出 | 当前不等于；必须有类能力声明和参数声明 |
| 5 | compatibility | schema v1 | 旧三字段 artifact 是否兼容 | 当前不兼容读取；跨版本读取需通过新的 schema version 显式引入 |

## 3 第一层设计描述

### 3.1 总体结构

图 4 生产端与消费端分层

```text
Classic API layer
  quantize.py: 参数检查、生命周期、发布 hook、convert 入口
          |
Artifact protocol layer
  quant_result.py: schema、digest、路径、Writer、Reader、reload transaction
          |
Quant module layer
  BaseQuantizeModule: export/load/dump mechanism
  MinMaxQuant / LinearAWQuant / GPTQuant: algorithm state and finalization
          |
Existing framework layer
  AlgorithmRegistry / parse_config / ModuleHelper / InsertPass / ReplacePass
          |
Runtime layer
  PyTorch Tensor、文件系统、safetensors、可选 torch_npu/NPU
```

#### 3.1.1 模块划分

先用一句话记忆各层职责：**API 负责何时做，协议层负责怎样存，量化模块负责存什么，既有 Pass 负责怎样部署。**

表 6 模块划分

| 模块 | 实现文件 | 职责 | 明确不负责 |
|------|----------|------|------------|
| API 入口 | `classic/quantize.py` | 参数、Writer 生命周期、convert 调用 reload | 不实现算法数学和文件 schema 细节 |
| 结果协议 | `classic/quant_result.py` | 常量、JSON、文件、manifest、验证、reload | 不决定量化公式 |
| 基类机制 | `quantize_op/base_quant_module.py` | 参数声明、导出、加载、dump hook | 不知道具体算法参数名称 |
| MinMax | `classic/quantize_op/minmax_module.py` | scale/offset 计算和定稿时机 | 不负责 manifest 发布 |
| AWQ | `classic/quantize_op/linear_awq_module.py` | scale 搜索、状态恢复 | 不重复保存可重算 weight |
| GPTQ | `classic/quantize_op/gptq_module.py` | Hessian 优化和优化后 weight | 不负责读取 artifact |
| 插入 Pass | `classic/optimizer/insert_quantize_op_pass.py` | 创建并替换 calibration module，注入 Writer | 不决定何时发布 manifest |
| 部署 Pass | 既有 `ReplaceNpuQuantModulePass` | 校准态到 deploy module | 不解析 manifest |

### 3.2 分解描述

#### 3.2.1 生产端模块分解

`quantize()` 的调用顺序：

1. `check_params` 校验 model/config/path 类型。
2. `parse_config` 得到 layer-level config。
3. 有结果目录时，检查 layer、注册量化类能力和目录 managed path。
4. `compute_model_fingerprint` 在任何模块插入前记录 model_type/dtype。
5. 创建 `QuantResultWriter`。
6. `InsertQuantizeModulePass` 创建量化模块、替换原模块并注入 Writer。
7. 各算法在定稿点调用 `maybe_dump_quant_params()`。
8. Writer 写 unit 并在所有 unit 完整时发布 manifest；activation 路径通过 root forward hook 延迟发布。

#### 3.2.2 消费端模块分解

`convert(model, quant_result_dir)` 的调用顺序：

1. 目录非空时调用 `reload_quant_params`。
2. Reader 读取 manifest 并校验精确 schema、目录和 source model。
3. 每个 unit 定位原模块、反查量化类、读取 safetensors、校验 tensor schema。
4. 在原模块 deep copy 上构造 replacement，并调用 `load_quant_params`。
5. 所有 replacement 成功后统一 `setattr`；异常时回滚。
6. `ReplaceNpuQuantModulePass` 继续执行原有部署转换。

#### 3.2.3 数据分解

表 7 artifact 数据实体

| 实体 | 内容 | 来源 | 消费者 |
|------|------|------|--------|
| source_model | model_type、torch_dtype | 原始 float model | reload 身份校验 |
| unit entry | layer/module/config/metadata/schema/digests | 量化模块和 Writer | Reader、reload |
| tensor payload | scale、offset、AWQ scale、GPTQ weight 等 | 子类导出 | `load_quant_params` |
| header | artifact_id、layer_name、entry_digest | Writer | Reader 交叉校验 |
| manifest | 协议版本、artifact_id、source、units | Writer 最后发布 | Reader 唯一入口 |

![图 5：artifact 目录、受管文件和发布清单](../figures/artifact-directory.png)

**说明：** `unit.safetensors` 是“每层货物”，`quant_manifest.json` 是“装箱单和封条”。缺一层、摘要对不上或装箱单没有封上，都不能交付。

### 3.3 依赖性描述

#### 3.3.1 运行设计

量化阶段不需要先创建 manifest；Writer 只在 expected units 全部 dump 后发布。activation 量化未完成校准时，
目录可以暂时存在 safetensors，但没有 manifest，因此任何 Reader 都会拒绝。该目录不支持按文件存在进行 resume。

#### 3.3.2 数据依赖关系

```text
量化 class._quant_param_keys
  -> BaseQuantizeModule.export_quant_params()
  -> Writer.write_unit()
  -> unit entry.tensors + safetensors payload
  -> manifest.units
  -> Reader schema check
  -> BaseQuantizeModule.load_quant_params()
  -> subclass _on_quant_params_loaded()
  -> ReplaceNpuQuantModulePass
```

### 3.4 接口描述

#### 3.4.1 模块/子系统接口

表 8 对外和内部接口

| 接口 | 输入 | 输出 | 异常/限制 |
|------|------|------|-----------|
| `quantize(model, config, quant_result_dir=None)` | nn.Module、dict、已有目录 | None，原地改写 | 类型、目录、算法能力错误；默认 None 保持旧行为 |
| `convert(model, quant_result_dir=None)` | calibration model 或 fresh float + artifact | None，原地部署 | artifact 无效或 reload 失败；不重新校准 |
| `reload_quant_params(model, dir)` | fresh float model、已发布目录 | 同一 model 实例 | 任一校验失败则不替换 |
| `BaseQuantizeModule.export_quant_params()` | 模块属性 | tensors、metadata | 只导出声明 key，None 跳过 |
| `BaseQuantizeModule.load_quant_params()` | tensors、metadata | 模块状态 | 未声明、重复、shape/dtype 错误拒绝 |
| `QuantResultWriter.write_unit()` | layer、payload、配置 | unit entry | 重复 layer、路径冲突、不可序列化值拒绝 |
| `QuantResultWriter.publish_if_complete()` | 无 | bool | 缺 unit 返回 False，不发布 |

#### 3.4.2 进程接口

当前无跨进程 RPC。跨进程边界通过文件系统 artifact：生产进程结束后，消费进程只能依赖 manifest 和 safetensors，
不能依赖生产进程的 Python 对象、内存 cache 或 calibration 数据。

## 4 第二层设计描述

### 4.1 QuantResultWriter 结构

#### 4.1.1 总体结构

表 9 Writer 状态

| 状态 | 条件 | 允许操作 | 禁止操作 |
|------|------|----------|----------|
| Created | expected units 已校验 | write_unit | publish 前覆盖 managed path |
| Partial | 有 unit、仍有 pending | 继续写其它预期 unit | Reader 加载；发布 manifest |
| Complete | pending 为空、文件复验通过 | 原子发布 manifest | 再写 unit |
| Published | manifest 已存在且 `_published=True` | 返回已发布状态 | 任何修改或覆盖 |

#### 4.1.2 Writer 关键处理

- `validate_quant_result_dir` 要求非空、已有目录且具备读写搜索权限。
- `_check_expected_units` 在 Writer 创建阶段拒绝重复、非法和互相嵌套的 layer 名称，避免完整写出后才发现不可 reload。
- `_check_managed_paths_available` 只拒绝 AMCT 自己的目标文件，不删除其它用户文件。
- `unit_file_name` 将非安全字符归一化，并用反向表拒绝碰撞。
- `atomic_write_via` 同目录临时文件、fsync、chmod、replace、目录 fsync。
- `_verify_unit_file` 写后重新读取 header 和 tensor schema。
- `_verify_result_dir` 发布前重新验证文件大小、SHA-256、entry digest 和 unit 内容。

### 4.2 BaseQuantizeModule 结构

#### 4.2.1 参数声明

`_quant_param_keys` 是子类声明的属性名集合，基类不写死算法字段。当前声明如下：

表 10 三算法参数声明

| 类 | `supports_quant_result` | `quant_param_keys()` | 定稿点 |
|----|-------------------------|----------------------|--------|
| `MinMaxQuant` | True | `scale_w`、`offset_w`、`scale_d`、`offset_d`、`scale_w1`、`scale_w2` | weight-only 注入后；activation 第 `batch_num` 次 forward |
| `LinearAWQuant` | True | `scale_w`、`offset_w`、`scale` | scale search 和 apply 完成后 |
| `GPTQuant` | True | `scale_w`、`offset_w`、`weight` | Hessian 优化后、weight 写回后 |
| `BaseQuantizeModule` | False | 默认空 tuple | 不允许直接进入 artifact P0 |

#### 4.2.2 导出和加载规则

![图 6：校准、fake quant 和 deploy 三个阶段](../figures/quantization-three-stages.png)

![图 7：Linear 量化参数和 W4A16 的含义](../figures/linear-quantization.png)

![图 8：per-tensor、per-channel、per-group 粒度](../figures/granularity.png)

**理解：** 校准阶段得到“该用多大的 scale”；fake quant 阶段用它模拟低比特误差；deploy 阶段才把模块变成设备路径。三张图分别回答“什么时候保存”“保存哪些量”“scale 的 shape 为什么不同”。

1. tensor 值执行 detach、CPU、contiguous 后交给 safetensors。
2. 非 tensor 值进入 `non_tensor_params`，None 不写入任何 payload。
3. 加载前检查 incoming key 是声明 key，tensor 和 metadata 不得重叠。
4. 已有 tensor 属性执行 shape/dtype 校验后原地 copy；None 属性按目标 device 创建。
5. 子类 hook 用 `require_loaded_quant_params` 检查算法真正需要的最小参数，并跳过已完成的校准阶段。

### 4.3 Reader 和 reload 结构

#### 4.3.1 manifest 校验顺序

```text
存在 quant_manifest.json
 -> JSON duplicate-key 拒绝
 -> 精确字段集合/format/schema_version/artifact_id
 -> source_model 字段
 -> unit 字段、唯一 layer、安全路径
 -> 结果目录文件、大小、SHA-256、entry digest
 -> header、tensor key/dtype/shape
 -> 模型 model_type/dtype
 -> 定位原模块和量化类
```

#### 4.3.2 量化类反查

当前实现不从 artifact 读取 Python 路径，也不依赖额外的类名全局字典。`_resolve_quant_module_class` 从
`entry.quant_config.algorithm` 读取唯一算法名，再执行：

```python
quant_cls = AlgorithmRegistry.algo[algorithm][entry["ori_module_type"]]
```

随后校验：类存在、类是 `BaseQuantizeModule` 子类、声明 artifact 能力、类名等于 `quant_module_type`。

#### 4.3.3 两阶段事务

![图 9：reload 的 Validate → Stage → Commit 事务](../figures/reload-transaction.png)

**理解：** Reader 先在草稿副本中准备所有新模块，全部成功后才一次性安装到 live model。这个事务保护的是“模块替换”这一步；它不承诺整个外层 `convert` 的所有副作用都能回滚。

表 11 reload 两阶段

| 阶段 | 允许动作 | 不允许动作 | 失败后果 |
|------|----------|------------|----------|
| Validate | 读 manifest、目录、模型和文件 | 修改 live model | 抛错，模型完全不变 |
| Stage | deep copy 原层、构造 replacement、加载参数 | `setattr` 到 live model | 抛错，模型完全不变 |
| Commit | 按 manifest 顺序统一 `setattr` | 再做不受控的 artifact 解析 | `setattr` 异常时逆序恢复 |

### 4.4 开发者检查清单

阅读或修改实现时，可以按下面顺序定位问题：

| 看到的现象 | 先看哪个文件 / 函数 | 应该验证什么 | 对应测试 |
|------------|--------------------|--------------|----------|
| 目录没有 manifest | `classic/quant_result.py` 的 Writer 状态和 `publish_if_complete` | expected units 是否全部写出；写后复验是否通过 | TC-007～TC-013 |
| unit 能读但 reload 失败 | `read_quant_unit`、`load_quant_params` | header、entry digest、key、shape、dtype 是否一致 | TC-006、TC-010、TC-013、TC-026 |
| 某算法没有写结果 | 对应 `minmax_module.py`、`linear_awq_module.py`、`gptq_module.py` 的定稿点 | `supports_quant_result`、`quant_param_keys()` 和 dump hook 是否齐全 | TC-016～TC-019 |
| 错误模型被部分替换 | `reload_quant_params` 的 validate/stage/commit | 所有文件和模块是否在第一次 `setattr` 前验证完成 | TC-027～TC-030 |
| artifact 路径和直接路径不一致 | `convert` 与 `ReplaceNpuQuantModulePass` | 两条路径是否复用同一个 deploy pass，是否误传已量化模型 | TC-031～TC-032 |
| NPU 上设备不对 | `BaseQuantizeModule.load_quant_params` 的 device 处理 | 文件可在 CPU 读取，但参数是否回到原层 device | TC-033～TC-034 |

![图 10：需求、设计、测试和证据追踪](../figures/requirement-design-test-trace.png)

## 5 可靠性分析

### 5.1 SFMEA 分析表

表 12 可靠性失效模式

| 失效模式 | 原因 | 检测点 | 处理 |
|----------|------|--------|------|
| 校准中断 | 进程异常/forward 失败 | manifest 不存在 | Reader 拒绝；使用新目录重试 |
| unit 文件截断 | 写入或磁盘故障 | size/SHA/header/schema | 发布前或 reload 前拒绝 |
| 配置被篡改 | JSON 手工修改 | entry digest/unknown field | 拒绝，不碰模型 |
| 混入其它 unit | 目录复用 | managed path、manifest units | 写端拒绝覆盖；读端只消费列举 unit |
| 错误模型 | checkpoint 或 dtype 不匹配 | model_type/dtype、层类型、shape | staging 前拒绝 |
| 算法类缺失 | 环境版本/注册配置不一致 | AlgorithmRegistry lookup | 明确报错，不动态 import |
| 第二层替换失败 | setattr/构造异常 | stage/commit | commit rollback |
| AWQ scale 未复用 | reload 后重新搜索 | `_on_quant_params_loaded`、calc_done | 标记完成并从参数重建 cache |
| GPTQ weight 丢失 | 只保存 scale | `require_loaded_quant_params` | 拒绝不完整 unit |

### 5.2 SFMEA 问题优先级

表 13 可靠性优先级

| 问题 | 影响 | 优先级 | 当前措施 |
|------|------|--------|----------|
| manifest 过早出现 | 下游加载半产物 | 高 | manifest-last + atomic replace |
| 部分替换模型 | 后续推理结果不可解释 | 高 | 全量 staging + rollback |
| 未知字段静默接受 | 协议漂移 | 高 | exact schema + duplicate key rejection |
| source model 过严 | 合法 checkpoint 无法复用 | 中 | 只比较 model_type/dtype，结构单独检查 |
| 旧 schema 不受支持 | 跨版本加载失败 | 中 | 明确拒绝，不隐式放宽校验 |

## 6 配置和控制

### 6.1 启动

用户先创建空的、版本化的结果目录，再调用 `quantize(..., quant_result_dir=path)`。当前 Writer 不替用户创建目录，
这是为了避免路径拼写错误时悄悄产生不可追踪的目录。

### 6.2 关闭

weight-only 路径在 `quantize` 返回前可直接发布；activation 路径在成功的 root forward hook 中发布。若没有完成
所需 calibration batch 或 forward 抛异常，manifest 不出现。

### 6.3 版本和配置控制

协议版本由 `MANIFEST_FORMAT` 和 `SCHEMA_VERSION` 控制；量化配置只保存 canonical JSON 可表达的标量、容器和 dtype。
不能把 module、tensor 或运行时 Writer 放入 `quant_config`。

## 7 数据库

本功能不使用数据库。artifact 目录是文件系统上的不可部分加载数据实体；manifest 是索引和协议入口，safetensors 是
tensor payload，entry digest 和 SHA-256 是完整性关系。

## 8 QuantResultWriter 详细设计

### 8.1 数据描述

表 14 当前 manifest 顶层字段

| 字段 | 类型 | 语义 |
|------|------|------|
| `format` | string | 固定 `amct-quant-result-manifest` |
| `schema_version` | int | 当前精确值 1 |
| `artifact_id` | UUID string | 绑定所有 unit header |
| `source_model` | object | `model_type`、`torch_dtype` |
| `units` | array | 每个量化层一个 entry |

表 15 当前 unit 字段

| 字段 | 类型 | 语义 |
|------|------|------|
| `layer_name` | string | 模型中的精确子模块路径 |
| `file` | string | 只允许 bare relative 文件名；Writer 生成的 unit 使用 `.safetensors` 后缀 |
| `quant_module_type` | string | 量化类名 |
| `ori_module_type` | string | 原模块类型名 |
| `quant_config` | object | 构造量化类所需完整配置 |
| `non_tensor_params` | object | 非 tensor 参数 |
| `tensors` | array | key/dtype/shape schema |
| `file_size` | non-negative int | 文件字节数 |
| `sha256` | string | 文件摘要 |
| `entry_digest` | string | 排除派生字段后的 entry 摘要 |

### 8.2 函数描述

#### 8.2.1 `QuantResultWriter.create`

- **功能描述：** 验证调用方目录、预期 unit 和 managed path，创建新的 writer。
- **输入参数：** 结果目录、预期 layer 名元组、source model fingerprint。
- **输出参数：** `QuantResultWriter` 实例。
- **返回值：** writer；不创建目录。
- **异常处理：** 目录不存在、不是目录、不可读写、预期 unit 为空/重复、目标文件已存在时抛异常。
- **调用关系：** `quantize -> _create_quant_result_writer -> QuantResultWriter.create`。
- **算法说明：** 用 expected unit 顺序固定 manifest 顺序，并生成 UUID artifact_id。

#### 8.2.2 `QuantResultWriter.write_unit`

- **功能描述：** 将一个 layer 的 tensor 和 metadata 原子写成 safetensors，并登记 entry。
- **输入参数：** layer_name、tensor dict、metadata、完整 quant_config、两种 module type。
- **输出参数：** unit entry 字典。
- **返回值：** 含 schema、file size、SHA-256、entry digest 的 entry。
- **异常处理：** 未知/重复 layer、文件名碰撞、非法 tensor、不可 JSON 序列化配置、路径冲突时失败；失败不会登记为完成。
- **调用关系：** 子类 `maybe_dump_quant_params -> write_unit`。
- **算法说明：** 先生成 entry digest，再写 header；写后读回 header/schema；最后计算派生摘要。

#### 8.2.3 `QuantResultWriter.publish_if_complete`

- **功能描述：** 只有所有 expected unit 完成时发布 manifest。
- **输入参数：** 无。
- **输出参数：** 无。
- **返回值：** 已发布或本次发布成功返回 True；仍 pending 返回 False。
- **异常处理：** 文件集合/摘要/schema 不一致时抛异常，manifest 不发布。
- **调用关系：** `quantize._finalize_quant_result` 立即调用；activation root hook 后再次调用。
- **算法说明：** `_verify_result_dir` 完成全部复验后，manifest 以同目录临时文件原子替换。

## 9 Reader/reload 详细设计

### 9.1 数据描述

Reader 不信任文件名以外的隐含信息。完整配置只来自 manifest；safetensors header 只做 artifact、layer 和 entry digest
的交叉绑定；实际 tensor payload 通过 safe_open 读取。

### 9.2 函数描述

#### 9.2.1 `load_quant_manifest`

- **功能描述：** 读取并严格验证 manifest。
- **输入参数：** artifact 目录。
- **输出参数：** 已验证 manifest 字典。
- **返回值：** manifest。
- **异常处理：** 缺失文件、重复 key、未知字段、类型错误、错误 format/version、非法 UUID、非法 unit/path 时抛异常。
- **调用关系：** `reload_quant_params` 首个步骤，也可由测试直接调用。
- **算法说明：** `json.load(object_pairs_hook=...)` 拒绝重复 key；`_check_required` 同时检查 missing/unknown 字段。

#### 9.2.2 `read_quant_unit`

- **功能描述：** 读取一个 unit 的 tensor 和 metadata，并校验 manifest/header/file。
- **输入参数：** artifact 目录、layer_name、可选已加载 manifest。
- **输出参数：** tensors、non_tensor_params、entry。
- **返回值：** 三元组；tensor 在 CPU 上，随后由模块加载逻辑移动到目标 device。
- **异常处理：** layer 不存在、文件缺失、摘要/header/schema 错误时抛异常。
- **调用关系：** `reload_quant_params` 对每个 entry 调用。
- **算法说明：** 先按 manifest 精确定位 bare filename，再 safe_open 读取，比较实际 key/dtype/shape。

#### 9.2.3 `reload_quant_params`

- **功能描述：** 在 fresh float model 上恢复所有量化 calibration module。
- **输入参数：** `nn.Module`、已发布 artifact 目录。
- **输出参数：** 同一个 model 实例。
- **返回值：** `model` 本身；成功后指定子模块已替换。
- **异常处理：** 所有 validate/stage 错误发生在第一次 `setattr` 前；commit 中异常逆序恢复。
- **调用关系：** `convert` 在提供目录时调用；不负责最终 deploy。
- **算法说明：** manifest 顺序建立 replacement plan；每层 deep copy 原模块，构造注册类并加载参数；计划完整后 commit。

#### 9.2.4 `convert`

- **功能描述：** 统一用户入口，兼容旧 calibration model 路径和新 artifact 路径。
- **输入参数：** calibration model，或 fresh float model + `quant_result_dir`。
- **输出参数：** 无。
- **返回值：** None。
- **异常处理：** reload 错误原样阻断 deploy；目录 None 时走旧逻辑。
- **调用关系：** `convert -> reload_quant_params（可选） -> ReplaceNpuQuantModulePass`。
- **算法说明：** 不解析算法参数、不复制 deploy 逻辑，保持原有转换 Pass 为单一实现。

## 10 错误处理

### 10.1 系统错误

- 目录权限、文件消失、磁盘写入或 fsync 错误：中止当前 artifact，保留无 manifest 的目录供诊断，改用新目录重试。
- NPU/CANN 不可用：测试报告标记 Block 或环境错误，不将其伪装为功能通过。

### 10.2 接口错误

- API 参数类型、空目录、root quantizable module、没有 quantizable layer：入口处抛 `TypeError`/`ValueError`。
- 不支持 artifact 的算法：通过注册类能力和参数声明拒绝，错误信息指导用户去掉目录或选择受支持算法。
- 旧路径 `quant_result_dir=None`：不触发 artifact 分支。

### 10.3 协议错误

- manifest 缺失、未知字段、重复 key、错误版本或非法 UUID：Reader 在读取阶段拒绝。
- 文件路径绝对路径、目录组件或 `..`：拒绝，不允许逃逸 artifact 根目录。
- 文件摘要、header、entry digest、tensor key/dtype/shape 不一致：拒绝，不修改模型。
- model_type/dtype、原模块类型或注册算法不匹配：staging 前拒绝。
- 必需量化参数缺失：子类 `_on_quant_params_loaded` 通过 `require_loaded_quant_params` 拒绝。
