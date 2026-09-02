# Classic 量化结果持久化与恢复软件需求规格说明书

| 拟制 | SWE：sunshine750 | 日期 | 2026-08-31 |
|------|-----------------|------|------------|
| 审核 | PL：待指定 | 日期 | 待评审 |
| 批准 | SE：待指定 | 日期 | 待批准 |



**关键词：** AMCT、Classic API、量化结果、safetensors、manifest、reload、convert、MinMax、AWQ、GPTQ

**摘要：**

Week 1-2 建立量化和 AMCT 工作流的系统认识，明确 Classic API 与 LLM blockwise CLI 的边界；Week 3-4
在 Classic API 中实现量化结果 artifact 的生产、校验、恢复和部署转换。一次校准后，用户可以将每个量化层
的最终参数保存为 safetensors，并在另一个由同源浮点模型构造的进程中，通过 `convert(model,
quant_result_dir=...)` 恢复校准态并继续使用既有部署转换逻辑。结果目录只有在所有 unit 完整写入、校验通过并
最后发布 `quant_manifest.json` 后才有效。

## 0 依据、范围与术语边界

### 0.1 需求依据

| 依据 | 用途 | 证据位置 |
|------|------|----------|
| 导师《实习计划：quantize 逐层量化与量化结果落盘》 | 确定 Week 1-4 的学习门槛、P0 范围和 Week 3/4 DoD | `/mnt/workspace/week3/docs/internship_plan_quantize.md` |
| 导师《quantize() 量化结果落盘 V2》 | 确定 artifact、manifest-last、reload、convert 和安全边界 | `/mnt/workspace/week3/docs/quantize_results_dump_v2.md` |
| PR #267 老师 review | 确定最终能力声明、模型校验范围、manifest 字段和常量组织 | `https://gitcode.com/cann/amct/merge_requests/267` |
| 当前实现 | 确认需求已经落到的函数、字段和异常行为 | `/mnt/workspace/amct-pr267-reviewfix`，commit `d5a2ac8d` |
| 当前测试 | 确认每一项需求的可验证方式和实际执行证据 | `tests/unit_test/classic/` |

### 0.2 Week 1-2 与 Week 3-4 的连续关系

1. Week 1-2 学习 `quantize -> calibration -> convert`，并通过 LLM CLI 的 Extract、PTQ、Eval、Deploy
   数据流理解“参数由谁生产、由谁消费”。这些 CLI 运行和 Qwen 实验是学习证据，不是本 PR 的 Classic
   代码入口。
2. Week 1-2 识别出 Classic 量化参数只在内存中、算法缺少统一导出边界、文件存在不能证明完整、逐层边读边替换
   会产生半量化模型等问题。
3. Week 3 解决可靠写端和 MinMax 最小闭环；Week 4 接入 AWQ/GPTQ、严格 reader、事务式 reload 和
   `convert` 入口，完成一个完整纵向闭环。

### 0.3 明确不属于本需求的内容

- `pipeline="layerwise"`、Adapter、Extract 编排、CalibrationSolver 和本地 calibration data 属于 Week 5-8。
- LLM blockwise CLI 的 `.pt` 参数目录不等同于 Classic artifact；本需求不把 CLI workflow 当作 Classic 上层入口。
- Qwen3.8 benchmark、性能实验、报告、图片和压缩包不属于本需求，不能加入正式代码 PR。
- 未声明 artifact 能力的其它 Classic 量化模块不在当前 P0 承诺中。
- 当前不要求旧的、包含 `producer`、`library_version`、`config_sha256` 或 `state_dict_sha256` 字段的未合入
  artifact 兼容读取；如未来需要，必须通过新的 schema version 或显式兼容分支实现。

### 0.4 术语和缩略语

| 缩略语 | 英文全名 | 中文解释 |
|--------|----------|----------|
| AMCT | Ascend Model Compression Toolkit | 昇腾模型压缩工具包 |
| API | Application Programming Interface | 程序调用接口；本需求指 Classic 的 Python 入口 |
| CLI | Command-Line Interface | 命令行入口；本需求仅作为 Week 1-2 学习对象 |
| PTQ | Post-Training Quantization | 训练后量化 |
| UT | Unit Test | 单元测试，验证一个函数或模块的局部行为 |
| E2E | End to End | 端到端测试，验证从用户入口到最终产物的完整链路 |
| NPU | Neural Processing Unit | 昇腾神经网络处理器 |
| JSON | JavaScript Object Notation | manifest 使用的结构化文本格式 |
| SHA-256 | Secure Hash Algorithm 256-bit | 文件和内容完整性摘要算法 |
| UUID | Universally Unique Identifier | artifact 的唯一标识 |
| AWQ | Activation-aware Weight Quantization | 感知激活的权重量化算法 |
| GPTQ | Generative Pre-trained Transformer Quantization | 基于 Hessian 误差补偿的权重量化算法 |
| MinMax | Min-Max Quantization | 根据最小值/最大值计算量化 scale/offset 的算法 |
| artifact | quantization result artifact | 可校验、可恢复的量化结果中间产物 |
| unit | quantization unit | 一个可独立保存和恢复的量化层 |

### 0.5 要求性质与证据等级


| 类别 | 本文结论 | 证据/处理方式 |
|------|----------|---------------|
| 老师明确要求 | Classic `quantize(..., quant_result_dir=...)` 可选落盘；按层保存量化结果；manifest-last；从原始浮点模型 reload 后 convert；保持默认 Classic API；Week 3/4 覆盖 MinMax/AWQ/GPTQ 代表性路径 | 导师实习计划、量化结果落盘协议、PR #267 review；对应 FR-001~FR-006 |
| 为完成要求必须补充的工程工作 | canonical JSON、safetensors、SHA-256、文件名安全校验、严格 schema、注册表反查、参数导出/加载 hook、staging/commit/rollback、UT/集成/NPU 证据 | 这些是让“可验证、可恢复、失败不破坏模型”能够落地的实现约束；由 V2 协议、当前代码和测试共同确认 |
| 合理推断 | `source_model` 只比较 `model_type`/`torch_dtype`，目标层的类型和参数 shape/dtype 单独校验；无关目录文件可以共存但不被读取；AWQ 重建时不保存可由原始权重和 scale 得到的 weight | 当前 review 后 schema、`validate_source_model`、`validate_result_directory`、三算法参数声明和对应 UT；若老师要求更强约束需另行冻结 |
| 当前明确不做/可选扩展 | 旧字段 artifact 的隐式兼容、其它未声明算法、完整目录集合强相等校验、Week 5-8 layerwise API、Qwen benchmark 和大模型性能实验 | 任务计划的范围边界和 PR #267 当前实现；扩展必须新开需求或 schema version，不在本里程碑加入 |

## 1 功能需求

### 1.1 FR-001：保持 Classic API 兼容并增加可选结果目录

- **需求编号：** FR-001；来源：PR #267，`quantize.py::quantize` 和 `convert`；Issue 链接：未提供，以 PR #267 为追踪来源。
- **需求背景：** 原有 Classic 流程在同一个 Python 模型对象中完成量化、校准和转换，量化参数没有独立的可移交产物。
- **需求描述：** `quantize(model, config=None, quant_result_dir=None)` 增加可选落盘能力；
  `quant_result_dir=None` 时必须保持原有内存路径。`convert(model, quant_result_dir=None)` 在目录非空时
  先恢复量化校准态，再复用既有 deploy pass。
- **输入：** `nn.Module`；可选 config 字典；可选的已存在、可读写的目录；convert 时可选已发布 artifact 目录。
- **处理：** 只有目录参数非空时才校验目录、创建 Writer、计算 source model 身份和安装发布 hook；convert 只有目录
  非空时才调用 `reload_quant_params`。
- **输出：** 函数仍原地修改模型并返回 `None`；默认路径不创建 artifact；落盘路径产生有效 manifest；convert 最终得到原有部署模块。
- **性能要求：** 默认路径不得因为未启用落盘而执行完整模型指纹计算或磁盘扫描；落盘开销只与声明的量化 unit 和
  其 tensor 大小相关。
- **质量要求：** 参数类型错误、空路径、文件路径、缺失目录和根模块量化必须给出稳定的 `TypeError`/`ValueError`；
  不得把 Writer 或路径对象写入量化配置。

### 1.2 FR-002：按量化层保存完整且可重建的 unit artifact

- **需求编号：** FR-002；来源：导师 V2 产物协议；Issue 链接：未提供，以 PR #267 为追踪来源。
- **需求背景：** 不同算法的参数在不同时间定稿，不能由入口函数猜测，也不能把所有层的参数混成一个无结构文件。
- **需求描述：** 每个预期 layer 必须生成一个 safetensors unit；manifest 保存层路径、量化类名、原模块类型、完整
  `quant_config`、非 tensor 参数和 tensor schema。量化类通过 `supports_quant_result` 声明是否参加该协议，
  基类默认不声明；实际参数 key 仍由每个子类的 `quant_param_keys()` 提供。
- **输入：** 量化模块在自身定稿点导出的 tensor 和 metadata；层名；原/量化模块类型；完整 layer config。
- **处理：** `BaseQuantizeModule.export_quant_params()` 将 tensor 与 JSON metadata 分流；
  `maybe_dump_quant_params()` 调用 Writer；Writer 生成 unit entry 和 entry digest。
- **输出：** `<sanitized_layer_name>.safetensors`；内存中的 unit entry；所有 unit 完成后由 FR-003 发布 manifest。
- **性能要求：** tensor 以 CPU、contiguous 形式写入；哈希按固定块读取，不要求把整个模型或文件一次性复制到内存。
- **质量要求：** 基类不能硬编码 MinMax/AWQ/GPTQ 参数名称；None 参数不伪造为已产生参数；GPTQ 必须保存优化后
  `weight`，AWQ 不重复保存可由原始权重和 scale 重算的权重。

### 1.3 FR-003：实现严格 manifest schema、完整性校验和 manifest-last 发布

- **需求编号：** FR-003；来源：导师 V2 协议及 PR #267 最新 review；Issue 链接：未提供。
- **需求背景：** “目录里有几个文件”不能证明量化运行完整。中断或受 manifest 管理的 unit 部分写入时，Reader
  必须拒绝整个 artifact；与 artifact 无关的文件允许共存，但 Reader 不得把它们当作量化结果消费。
- **需求描述：** 当前 schema v1 的 manifest 只包含 `format`、`schema_version`、`artifact_id`、`source_model`、
  `units`。`source_model` 只包含 `model_type` 和 `torch_dtype`。每个 unit 包含 layer、文件、模块类型、配置、
  tensor schema、file size、SHA-256 和 entry digest。manifest 必须最后原子发布。
- **输入：** Writer 的全部 unit entry 和结果目录。
- **处理：** safetensors 先写同目录临时文件，再 fsync 和 `os.replace`；发布前重新校验每个文件；最后以同样方式写 manifest。
  Reader 使用严格字段集合、类型检查、重复 JSON key 拒绝、路径穿越检查和摘要交叉校验。
- **输出：** 成功时一个合法 `quant_manifest.json`；失败时不得出现可被 Reader 误认的 manifest。
- **性能要求：** 使用 safetensors，不使用 pickle；文件摘要按块计算；目录中无关的非 artifact 文件可以共存，但 AMCT 自己
  管理的 manifest/unit 文件不能被覆盖。
- **质量要求：** 缺字段、未知字段、重复字段、错误版本、错误摘要、manifest 列出的受管文件缺失、错误 header、错误
  tensor schema、绝对路径、`..` 和嵌套路径都必须拒绝，并在修改模型前报告错误。目录中未被 manifest 列举且不与
  Writer 目标路径同名的普通文件、隐藏文件或额外 `.safetensors` 文件不属于 artifact，不应导致 reload 失败，也不得被读取。

### 1.4 FR-004：从受信任注册信息重建三类量化模块

- **需求编号：** FR-004；来源：导师 Week 4 DoD、PR #267 最新 review；Issue 链接：未提供。
- **需求背景：** manifest 中只能保存字符串，不能直接序列化 Python class；Reader 需要根据算法名和原模块类型找到实际类。
- **需求描述：** Reader 从 unit 的 `quant_config.algorithm` 取得唯一算法名，再通过 `AlgorithmRegistry.algo[algorithm][ori_module_type]`
  查找当前安装中已注册的量化类。不得根据 manifest 动态 import 任意 Python 路径。MinMax、AWQ、GPTQ 需恢复各自声明的参数。
- **输入：** 合法 unit entry、当前安装的 `AlgorithmRegistry`、fresh float model 中对应的原模块。
- **处理：** 校验算法恰好一个、算法已注册、原模块类型已注册、量化类声明 artifact 能力且类名与 manifest 一致；构造量化模块并调用
  `load_quant_params`。
- **输出：** 已加载参数的校准态量化模块，交给既有 `ReplaceNpuQuantModulePass`。
- **性能要求：** 反查只使用内存注册表；不得执行不受信任的导入或重新运行校准搜索。
- **质量要求：** 未知算法、未知原模块类型、未知量化类、多算法 config、未声明能力和参数 key 不匹配都必须拒绝。

### 1.5 FR-005：实现 reload 的全有或全无替换

- **需求编号：** FR-005；来源：导师 Week 4 DoD；Issue 链接：未提供。
- **需求背景：** 如果第一层已经替换，第二层加载失败，用户会得到无法判断状态的半量化模型。
- **需求描述：** `reload_quant_params(model, quant_result_dir)` 分为 validate/stage 和 commit 两阶段。任何 artifact、模型、构造或加载
  错误都必须发生在第一次替换之前；提交阶段出现异常时逆序恢复已经替换的原模块。
- **输入：** fresh float `nn.Module` 和已发布 artifact 目录。
- **处理：** 先加载 manifest、验证目录和 source model；逐 unit 定位父模块、检查原类型、读取文件、构造 deep copy 上的量化模块并加载参数；
  全部成功后统一 `setattr`。
- **输出：** 成功返回原模型对象，并将 unit 层替换为校准态量化模块；失败时模型结构和原模块对象保持不变。
- **性能要求：** staging 只复制被恢复的层，不复制整个模型；量化参数恢复到原层 device。
- **质量要求：** 嵌套 unit、未知层、原模块类型错误、模型类型/dtype 错误、不同 tensor shape/dtype 和 commit 异常都必须有确定行为。

### 1.6 FR-006：恢复后复用既有 convert 并保持部署等价

- **需求编号：** FR-006；来源：导师 Week 4 里程碑；Issue 链接：未提供。
- **需求背景：** 新增 artifact 路径不应复制一套 deploy 逻辑，否则两条路径会产生不同的部署模型。
- **需求描述：** `convert(fresh_model, quant_result_dir=artifact_dir)` 必须先 reload，再调用原有 `ReplaceNpuQuantModulePass`。直接校准后 convert
  与落盘后 reload 再 convert 的部署权重、scale 和输出应一致。
- **输入：** 同源 fresh float model、合法 artifact；直接路径使用同配置校准态模型。
- **处理：** 通过 FR-005 恢复 MinMax/AWQ/GPTQ 模块，再执行未修改的 deploy pass。
- **输出：** 部署模型；可比较的部署 tensor、scale 和前向输出。
- **性能要求：** artifact 路径不重新读取 calibration data，不重新执行 AWQ 搜索或 GPTQ Hessian 优化。
- **质量要求：** 三种代表性算法至少各有一个往返测试；部署等价失败时必须定位到参数或转换链路，而不是只比较文件存在。

### 1.7 FR-007：保证安全、可维护和可诊断

- **需求编号：** FR-007；来源：V2 协议安全边界及 PR review；Issue 链接：未提供。
- **需求背景：** artifact 是可能来自外部目录的输入，文件路径、JSON 和序列化方式不能扩大可执行范围。
- **需求描述：** 只使用 JSON 和 safetensors；拒绝路径穿越、未知 schema 字段和重复 JSON key；错误信息面向用户说明下一步，不能暴露内部能力 flag 名称。
- **输入：** 不可信或损坏的 manifest、unit 文件和配置。
- **处理：** 先解析和校验，不动态 import、不 pickle load、不部分替换模型。
- **输出：** 合法输入正常恢复；非法输入抛出可定位的异常并保留模型状态。
- **性能要求：** 不因安全校验复制完整 state_dict；文件摘要使用流式块读取。
- **质量要求：** 关键失败路径有负例 UT；日志能够区分写入、发布和 reload 阶段。

### 1.8 FR-008：提供可复现的测试和交付证据

- **需求编号：** FR-008；来源：导师实习计划交付要求；Issue 链接：未提供。
- **需求背景：** 仅有单测通过不能证明完整 API 可用，也不能证明测试对应当前代码版本。
- **需求描述：** 交付必须包含 UT、mock/integration 和实际 NPU/小模型端到端证据，记录 commit、环境、命令、结果、已知环境阻塞和未覆盖边界。
- **输入：** 当前代码 commit、测试环境、tiny model/注册算法和可用 NPU。
- **处理：** 先执行快速 UT，再执行 Classic 目录回归，最后执行端到端和 NPU 测试；独立记录与本次代码无关的 collection error。
- **输出：** 测试报告、命令和实际结果；PR review 可按用例编号追溯到源码和测试函数。
- **性能要求：** 快速 UT 可在开发循环中执行；大模型精度实验不作为每次修改的阻塞条件。
- **质量要求：** 不以跳过失败测试替代修复；每个测试必须对应明确问题、输入、预期和执行状态。

### 1.9 需求跟踪矩阵


| 需求 | 主要实现位置 | 主要验证入口 | 当前状态 |
|------|--------------|--------------|----------|
| FR-001 | `amct_pytorch/classic/quantize.py`、`classic/optimizer/insert_quantize_op_pass.py` | TC-014~TC-017、TC-032 | 已实现，默认路径和 artifact 路径均有覆盖 |
| FR-002 | `quantize_op/base_quant_module.py`、`classic/quantize_op/{minmax,linear_awq,gptq}_module.py` | TC-005~TC-007、TC-016~TC-019 | 已实现，按算法定稿点导出 |
| FR-003 | `amct_pytorch/classic/quant_result.py` 的 schema、digest、原子写和 reader 校验 | TC-001~TC-004、TC-008~TC-013、TC-025~TC-026 | 已实现，manifest-last；无关额外文件规则已明确 |
| FR-004 | `quant_result.py::_resolve_quant_module_class`、`AlgorithmRegistry` | TC-020~TC-023、TC-028~TC-029 | 已实现，受信任注册表反查 |
| FR-005 | `quant_result.py::reload_quant_params`、`_commit_replacements` | TC-024~TC-030 | 已实现，validate/stage/commit/rollback |
| FR-006 | `classic/quantize.py::convert`、既有 `ReplaceNpuQuantModulePass` | TC-031~TC-034 | 已实现；NPU 证据受环境条件约束 |
| FR-007 | `quant_result.py` 的安全路径、JSON、safetensors 和错误处理 | TC-003、TC-010~TC-013、TC-024~TC-030 | 已实现，关键负例覆盖 |
| FR-008 | `docs/week1-4/` 测试报告、测试命令和 PR CI | TC-035、报告“测试执行命令和结果” | 当前 commit 已有可复现证据；完整全仓收集仍受外部依赖约束 |

## 2 待确定问题

| 序号 | 问题描述 | 负责人 | 计划解决时间 |
|------|----------|--------|--------------|
| 1 | 是否需要兼容尚未合入、含旧 `producer/library_version/config_sha256/state_dict_sha256` 的 artifact；若需要，schema 是否升为 2 | 老师/PL | 合入 PR 前确认 |
| 2 | PR #267 是否要求在当前阶段 squash 为单一提交、以及由谁将四条 discussion 标记为 resolved | 老师/仓库维护者 | PR 合入前确认 |
| 3 | 三算法的正式支持矩阵是否保持当前 MinMax/AWQ/GPTQ 代表性路径，还是扩展其它已注册模块 | 老师/PL | Week 5 规划前确认 |
| 4 | 真实 NPU 端到端验收使用哪一套固定模型、配置和输入；当前 tiny/NPU 测试是否满足里程碑签收 | 老师/PL | 里程碑验收时确认 |
| 5 | 需求文档归档应放在 PR #267 还是新建文档 PR；本文当前只写入工作树供审查 | 老师/PL | 提交文档前确认 |

## 3 冲突和解释记录

| 旧材料表述 | 当前采用解释 | 依据 |
|------------|--------------|------|
| 早期 V2 草案把 `producer`、`library_version` 列为必需 | 当前 schema v1 删除这两个冗余字段 | 老师 2026-08-28 review；commit `d5a2ac8d` |
| 早期草案要求完整 config/state_dict digest | 只校验影响量化身份的 `model_type`、`torch_dtype`；层和 tensor 结构仍单独校验 | 老师关于“校验太严格”的 review；当前 `validate_source_model` |
| 早期草案 header 包含 format/schema_version | 当前 header 只保留 artifact_id、layer_name、entry_digest，manifest 是唯一协议来源 | 老师此前 review 和当前 `unit_header` |
| 历史 Week 4 报告描述 `name_to_quant_op` | 当前代码没有依赖该字段，reload 按算法名和原模块类型直接查 `AlgorithmRegistry.algo` | 当前 `quant_result.py::_resolve_quant_module_class` |

## 4 检视意见

### 4.1 检视结论

| 检视日期 | 检视人 | 检视结论 | 遗留风险 |
|----------|--------|----------|----------|
| 待评审 | 老师/PL | 待评审 | 旧 artifact 兼容策略、提交归档位置和正式 NPU 验收口径待确认 |

### 4.2 检视问题清单

| 序号 | 问题描述 | 问题类别 | 严重程度 | 责任人 | 计划闭环时间 | 当前状态 |
|------|----------|----------|----------|--------|--------------|----------|
| 1 | 需求编号是否需要关联独立 issue，而不是只关联 PR #267 | 需求完整性 | 中 | PL | 提交文档前 | 待确认 |
| 2 | 当前 schema v1 是否需要旧字段兼容 | 需求一致性 | 高 | PL | PR 合入前 | 待确认 |
| 3 | Week 1-2 CLI 学习证据是否需要作为正式附件归档 | 可验证性 | 低 | SWE/PL | 文档评审时 | 待确认 |

### 4.3 检视问题闭环记录

| 序号 | 闭环说明 | 修改章节 | 闭环确认人 | 确认日期 |
|------|----------|----------|------------|----------|
| 1 | 待老师确认后补充 issue 链接或明确“PR 追踪” | 0.1、1.x | 待指定 | 待确认 |
| 2 | 待老师选择兼容策略后补充 schema 版本或兼容规则 | 2、3 | 待指定 | 待确认 |
