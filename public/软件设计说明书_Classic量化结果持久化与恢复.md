# Classic 量化结果持久化与恢复软件设计说明书

| 拟制 | SWE：sunshine750 | 日期 | 2026-08-31 |
|------|-----------------|------|------------|
| 审核 | PL：待指定 | 日期 | 待评审 |
| 批准 | PL：待指定 | 日期 | 待批准 |

## 修订记录

表 1 修订记录

| 日期 | 修订版本 | CR号 | 修改章节 | 修改描述 | 作者 |
|------|----------|------|----------|----------|------|
| 2026-08-31 | 1.00 | PR #267 | 全文 | 根据 requirement-development 模板补齐 Week 1-4 设计 | sunshine750 |

**关键词：** Classic API、QuantResultWriter、safetensors、manifest-last、reload、事务替换、MinMax、AWQ、GPTQ

**摘要：**

本设计把一次 Classic 量化拆成三个可验证阶段：量化模块在自己的参数定稿点导出状态，Writer 按 layer unit
原子写入并在最后发布 manifest，Reader 在 fresh float model 上完成全量校验和隔离构造后，再一次性替换模块。
设计复用现有的算法注册表、模块替换 Pass 和 deploy Pass，不复用 LLM CLI 的 Workflow.run，也不复制部署逻辑。

## 缩略语清单

表 2 缩略语清单

| 缩略语 | 英文全名 | 中文解释 |
|--------|----------|----------|
| AMCT | Ascend Model Compression Toolkit | 昇腾模型压缩工具包 |
| API | Application Programming Interface | Python 程序接口 |
| CLI | Command-Line Interface | 命令行接口 |
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


1. Week 1-2 学到的 CLI 数据流如何转化为 Classic 设计约束。
2. Week 3 如何在不破坏旧 API 的情况下接入结果写端。
3. Week 4 如何从结果目录安全恢复模块并继续部署。
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

- Week 5-8 的 `pipeline="layerwise"`、Adapter、Extract、Solver 和本地 calibration data。
- LLM CLI 的上层 Workflow 入口、Qwen benchmark、性能研究和实验报告。

## 2 第零层设计描述

### 2.1 软件系统上下文定义

图 1 Classic artifact 系统上下文（文字数据流）

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

表 3 外部实体与系统交互

| 外部实体 | 交互方式 | 输入 | 输出/约束 |
|----------|----------|------|-----------|
| 用户 Python 代码 | 调用 Classic API | model、config、目录 | 原地修改模型；返回值仍为 None 或原模型 |
| 文件系统 | 读写 artifact | 目录、JSON、safetensors | 目录必须可读写；manifest 是完成信号 |
| PyTorch | Module、Tensor、hook、device | nn.Module 和 tensor | 参数恢复到原层 device |
| safetensors | 安全 tensor 序列化 | CPU contiguous tensors | 不使用 pickle；header 只存交叉校验字段 |
| AlgorithmRegistry | 查找量化类 | algorithm、ori_module_type | 只能实例化当前安装中已注册的类 |
| NPU/torch_npu | 可选部署和前向执行 | deploy module、输入 tensor | NPU 测试依赖实际可用硬件 |
| LLM CLI | Week 1-2 学习参照 | Extract/PTQ/Eval/Deploy | 不作为 Classic 上层调用入口 |

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
| 5 | compatibility | schema v1 | 旧三字段 artifact 是否兼容 | 当前不兼容读取，待老师确认是否设计 schema v2 |

## 3 第一层设计描述

### 3.1 总体结构

图 2 生产端与消费端分层

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
- `_check_managed_paths_available` 只拒绝 AMCT 自己的目标文件，不删除其它用户文件。
- `unit_file_name` 将非安全字符归一化，并用反向表拒绝碰撞。
- `_atomic_write_via` 同目录临时文件、fsync、chmod、replace、目录 fsync。
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

表 11 reload 两阶段

| 阶段 | 允许动作 | 不允许动作 | 失败后果 |
|------|----------|------------|----------|
| Validate | 读 manifest、目录、模型和文件 | 修改 live model | 抛错，模型完全不变 |
| Stage | deep copy 原层、构造 replacement、加载参数 | `setattr` 到 live model | 抛错，模型完全不变 |
| Commit | 按 manifest 顺序统一 `setattr` | 再做不受控的 artifact 解析 | `setattr` 异常时逆序恢复 |

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
| 旧 schema 兼容未定 | 跨版本加载失败 | 中 | 明确列入待确认，不隐式放宽校验 |

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

## 11 检视意见

### 11.1 检视结论

| 检视日期 | 检视人 | 检视结论 | 遗留风险 |
|----------|--------|----------|----------|
| 待评审 | 老师/PL | 待评审 | schema v1 旧 artifact 兼容、PR 是否 squash、正式 E2E 验收口径 |

### 11.2 检视问题清单

| 序号 | 问题描述 | 问题类别 | 严重程度 | 责任人 | 计划闭环时间 | 当前状态 |
|------|----------|----------|----------|--------|--------------|----------|
| 1 | 早期设计报告与当前 schema 字段不一致，是否需要同时维护兼容 reader | 协议一致性 | 高 | PL | PR 合入前 | 待确认 |
| 2 | 是否要将 artifact 能力声明提升为 AlgorithmRegistry 的显式元数据对象 | 方案完整性 | 中 | PL | 后续算法扩展前 | 当前采用类元数据 |
| 3 | 当前 tiny/NPU 端到端证据是否满足里程碑签收 | 可验证性 | 中 | PL | Week 4 验收 | 待确认 |

### 11.3 检视问题闭环记录

| 序号 | 闭环说明 | 修改章节 | 闭环确认人 | 确认日期 |
|------|----------|------------|------------|----------|
| 1 | 根据老师决定补 schema v2 或维持当前 v1 严格拒绝 | 2.4、9、10 | 待指定 | 待确认 |
| 2 | 根据老师确认的正式 E2E 模型和设备补充复现命令 | 2.2.4、测试报告 | 待指定 | 待确认 |
