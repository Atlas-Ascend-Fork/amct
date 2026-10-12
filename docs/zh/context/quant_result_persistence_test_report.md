# Classic 量化结果持久化与恢复测试用例及测试报告

| 测试负责人 | sunshine750 | 测试日期 | 2026-10-19 |
|------------|-----------------|----------|------------|
| 被测版本 | #267 单一提交 | 测试对象 | AMCT PyTorch Classic artifact P0 |
| 测试范围 | Classic artifact P0 机制 | 文档范围 | Classic 测试与 Qwen 实机验证分开记录 |
| 测试环境 | 见测试环境和固定条件 | 当前结论 | #267 当前实现 Classic 全量回归零失败 |


---

## 测试结果

这份报告把每个测试分成四个层次：

- **协议单元测试（UT）**：确认 JSON、摘要、路径、schema 和参数导出/加载这些最小规则正确。
- **tiny 集成测试**：用真实 Classic API、真实 manifest、真实 safetensors 和真实注册表，确认多个模块能协作。
- **NPU / E2E 测试**：确认参数能回到设备、部署前向能执行，以及 direct / artifact 两条路径的结果关系。
- **静态和构建检查**：确认代码能导入、格式一致、编译检查可以完成。

![图 1：Classic 测试金字塔](../figures/test-pyramid.png)


### 状态定义

| 状态 | 在本文中的含义 | 能否算作功能通过 |
|------|----------------|------------------|
| **Pass** | 命令实际执行，且预期断言满足 | 可以，但只覆盖该用例输入和范围 |
| **Historical** | 结果来自文档记录的历史 commit，本轮未重新执行 | 只能作为历史证据，不能当作当前环境新结果 |
| **Block** | 测试逻辑明确，但硬件、依赖或指定数据不具备 | 不能；要记录阻塞原因 |
| **Skip** | pytest 条件未满足而没有执行 | 不能；不能把 skip 改写成 Pass |
| **Fail** | 实际执行且断言失败 | 不能；需要缺陷或修复记录 |

下文表 3 的 Pass 表示自动化测试在当前 #267 实现上执行并通过；重新验收时应同时记录新的提交、环境和日志。

![图 2：需求、设计、测试和证据追踪](../figures/requirement-design-test-trace.png)

## Classic API 当前实现边界

#267 已按“不新增对外接口”的要求收敛：用户仍只使用 `quantize(model, config=None, quant_result_dir=None)` 和 `convert(model, quant_result_dir=None)`。量化结果未完全生成时，内部 root forward hook 会在下一次成功 forward 自动发布 `quant_manifest.json` 并移除自身；如果用户直接把未完成目录传给 `convert()`，接口会列出未发布 unit 并要求重新校准，不要求用户调用额外生命周期函数。

Classic 全量回归结果见“测试执行命令和结果”：458 passed、13 skipped。

## 测试范围和分层

表 1 测试分层

| 层级 | 目标 | 测试位置 | 是否依赖 NPU/大模型 |
|------|------|----------|---------------------|
| UT | 验证 canonical JSON、摘要、schema、路径、Writer、Base 参数机制和单个算法状态 | `tests/unit_test/classic/test_quant_result.py`、`test_quantize_modules.py`、`ptq/test_quantize.py` | 否；NPU 专项除外 |
| Mock/tiny 集成 | 验证 Classic API、真实文件、三算法、模型定位、reload 事务和 convert 链路 | `tests/unit_test/classic/test_quant_reload.py`、`ptq/test_quantize.py` | tiny model；deploy/NPU 分支按条件执行 |
| E2E/NPU | 验证参数回到 NPU、reload 后前向和 deploy 前向与直接路径一致 | `test_quant_reload.py::TestReloadOnNpu`、`TestDeployForwardOnNpu`、`ptq/test_quantize.py` NPU 用例 | 需要实际 torch_npu 和 NPU |
| CI/静态 | 验证代码可导入、格式和编译检查 | ruff、compileall、`git diff --check` | 否；由当前环境执行 |

## 关联验证：Qwen3.8-27B 纯 Classic Python API 实测

Classic 用例验证通用 artifact 协议；Qwen 实测只采用纯 Classic Python API 路线：不调用 AMCT CLI、`parser_gen` 或 LLM workflow，直接完成 `algorithm_register`、`quantize`、artifact 保存、fresh-process convert、PPL 和 generation gate。Qwen 结果与 Classic 36 个用例分开统计，不计入 Classic 全量回归数量。

![图 3：Qwen3.8-27B 纯 Classic Python API 验证链路](../figures/qwen-real-model-validation.png)

| 编号 | 测试了什么 | 怎么测 | 实际结果 | 状态和边界 |
|------|------------|--------|----------|------------|
| QWEN-001 | 模型文件是否完整 | 逐文件核对 size/SHA-256，读取 index 和 18 个 safetensors header；构建完整 BF16 `nn.Module` | 33 个文件、18 个分片、1,199/1,199 tensor、55,586,114,768 bytes、496 个 language Linear（64 层 × 每层 8 个 Linear） | **PASS**；确认实机使用的是记录中的 BF16 checkpoint |
| QWEN-002 | 两卡模型是否能按显式策略构建 | 64 个 decoder layer 分布在两张 NPU（31/33）；vision、embedding、MTP 放 `npu:0`，final norm 和 lm_head 放 `npu:1` | 两卡设备分布成立，无 CPU/meta 残留 | **PASS**；这是设备放置检查，不是吞吐测试 |
| QWEN-003 | BF16 API baseline 是否可运行 | 同一 31/33 两卡 `nn.Module` 执行 WikiText-2 72×4096 PPL 和短文本生成 | API BF16 PPL `6.9534372879`；forward/logits 正常 | **PASS（baseline）**；作为 W8A8 和 mixed W4A8 的对照 |
| QWEN-004 | W8A8 纯 API 链路是否能落盘 | `algorithm_register` 后调用 `quantize(model, config, quant_result_dir)`；496 个 language Linear 使用 W8 和 dynamic per-token A8 | 496 unit artifact **PASS**；只保存 finalized `clip_factor` 与身份/清单信息 | **PASS**；不复制 52GB checkpoint |
| QWEN-005 | W8A8 是否能跨进程恢复并部署 | producer 与 deploy/PPL 分进程；fresh process 校验 checkpoint fingerprint 后 reload，再调用 `amct_pytorch.convert` | 496 个 W8 deploy module；PPL `7.0308233570` | **PASS**；证明 artifact reload 后可继续 public API convert |
| QWEN-006 | W8A8 生成机制是否可用 | greedy `transformers.PreTrainedModel.generate`，GSM8K batch-8 gate | 8/8 predictions **PASS** | **PASS（gate）**；不是 15,268 样本正式 accuracy benchmark |
| QWEN-007 | mixed W4A8 纯 API 链路是否能落盘 | 400 个 MLP/W4 unit（64 层 × 每层 4 个 MLP Linear）+ 96 个主 projection/W8 unit（64 层 × 每层 4 个 projection Linear）；同样走 `algorithm_register`、`quantize` 和 artifact | 496 unit artifact **PASS** | **PASS**；mixed policy 覆盖 496 个 language Linear |
| QWEN-008 | mixed W4A8 是否能跨进程恢复并部署 | fresh process reload 后调用 `amct_pytorch.convert`；400 个 W4 Linear（64 层 × 每层 4 个 MLP Linear）使用 packed W4 `npu_weight_quant_batchmatmul`，A8 为数值 dynamic per-token quantize/dequantize | 400 packed W4 + 96 W8 module；PPL `7.1221962920` | **PASS**；不宣称物理 INT8 x INT4 kernel |
| QWEN-009 | mixed W4A8 生成机制是否可用 | greedy generate，single-record GSM8K gate | 答案 `#### 18`，**PASS** | **PASS（gate）**；不能替代正式 accuracy benchmark |
| QWEN-010 | 算法边界是否记录清楚 | 检查算法描述和运行时 ABI | 11 点、1.0-0.5 diagonal-Hessian clipping search | **PASS（边界记录）**；这是 diagonal-Hessian GPTQ clipping 近似，不是 full GPTQ |
| QWEN-011 | 正式 accuracy benchmark 是否完整 | 15,268 prompt generation/scoring 流程单独执行 | 尚未形成完整正式结果，不把部分 JSONL 包装成完整 benchmark | **NOT_COMPLETE**；generation gate 不等于正式分数 |

Qwen 这一组测试证明：27.78B BF16 checkpoint 可以构建为两卡 `nn.Module`，BF16 baseline 可运行，W8A8 和 mixed W4A8 可以通过纯 Classic Python API 完成 `algorithm_register`、`quantize`、artifact 保存、fresh-process convert、PPL 和 generation gate。尚未证明的是 full GPTQ、物理 INT8 x INT4 kernel，以及 15,268 样本正式 accuracy benchmark 的完整分数。

Qwen 实测环境与 Classic tiny 回归不同：它使用两张 NPU，模型为 27.78B BF16 `nn.Module`，运行时受 torch_npu 和 CANN 9.1 ABI 影响。mixed W4A8 的 packed W4 路径来自该环境的可用算子，不能直接外推到其它 CANN 版本。

### Qwen 决策和边界详解

![图 4：Qwen 纯 API 实测的关键决策和边界](../figures/qwen-api-decision-boundaries.png)

#### 1. 为什么测 mixed W4A8

mixed W4A8 用于验证“显存收益和精度损失是否可以接受”。实测数据如下：

| 方案 | PPL | 部署显存 NPU0/NPU1 | 相对 W8A8 |
|------|-----|---------------------|-----------|
| W8A8 | `7.0308233570` | `15.019730 / 14.064249 GiB` | 基准 |
| mixed W4A8 | `7.1221962920` | `10.415925 / 9.186136 GiB` | PPL 增加 `0.091372935`，显存少约 `9.48 GiB` |

400 个 MLP Linear 使用 W4，96 个主 Q/K/V projection 使用 W8。这个测试回答的不是“mixed 一定最好”，而是“在该模型、该环境、该精度目标下，mixed 是否提供了一个可解释的折中”。

#### 2. 为什么不能写成 physical INT8 x INT4 kernel

测试中没有证明 physical INT8 x INT4 kernel，因为实现路径不是这样的 kernel：

```text
INT8 dynamic quantize
  -> dequantize 回 BF16
  -> BF16 activation × packed W4 weight
```

因此测试结论只能写成：

- packed W4 权重存储：成立。
- dynamic per-token A8 数值行为：成立。
- physical INT8 x INT4 matmul：不成立，也未测试。

如果未来要证明 physical 8×4 kernel，需要新的算子 ABI、正确 dtype 传递和 profiling 证据。

#### 3. 为什么没有测 full GPTQ

本轮没有测 27B full GPTQ，原因是测试目标不同。当前测试验证的是 Classic Python API 的 artifact 链路：

```text
algorithm_register
  -> quantize
  -> 496 unit artifact
  -> fresh-process convert
  -> PPL / generation gate
```

当前算法只收集 diagonal Hessian 并保存 `clip_factor`，不做 full Hessian inverse、误差补偿或优化后 weight 持久化。因此它不能被称为 full GPTQ。

#### 4. 测试是否完整

| 范围 | 是否完整 | 说明 |
|------|----------|------|
| Classic tiny 回归 | 完整 | 36 个 TC，#267 当前实现 Classic 全量回归零失败 |
| Qwen 纯 API 链路 | 完整 | 模型构建、BF16 baseline、W8A8、mixed W4A8、artifact、fresh-process convert、PPL、generation gate 均有结果 |
| 27B full GPTQ | 未覆盖 | 需要完整 Hessian、误差补偿、优化后 weight 和新一轮测试 |
| physical INT8 x INT4 kernel | 未覆盖 | 当前 ABI 中 activation 为 BF16 |
| 正式 accuracy benchmark | 未完成 | 15,268 样本 generation/scoring 不能用 gate 替代 |

所以准确结论是：**对声明的 artifact/API 链路范围，测试完整；对更大的 full GPTQ、physical 8×4 kernel 和正式发布验收，尚未覆盖。**

#### 5. 如果补 full GPTQ，需要新增哪些测试

1. full 或 blockwise Hessian 构建测试。
2. Hessian inverse / 误差补偿数值测试。
3. 优化后 weight 的导出、加载和 shape/dtype 校验。
4. artifact payload 扩展后的 manifest/schema 测试。
5. fresh-process reload 后不重跑 Hessian 的测试。
6. W8A8、mixed W4A8 的 PPL 复测。
7. generation gate 复测。
8. 15,268 样本正式 accuracy benchmark。
9. 与 diagonal-Hessian clipping approximation 的对照报告。



### 测试前后必须固定的条件

为了让结果可以复核，以下条件不能只写“使用默认配置”：模型结构、算法配置、输入 shape、随机种子、Python/PyTorch/torch_npu 版本、commit、pytest 命令和设备状态都需要写入日志。tiny 测试可以不依赖联网下载，但它仍然要使用真实的 API、文件协议和注册表。

## 测试环境和固定条件

表 2 测试环境

| 项目 | 值 |
|------|----|
| Python | 3.12.9 |
| PyTorch | 2.7.1+cpu |
| torch_npu | 本轮 Classic 全量用例在无真实 NPU 的本地环境执行；NPU 专项按条件跳过 |
| safetensors | 0.8.0 |
| pytest | 8.4.3 |
| ruff | 0.3.8 |
| 模型 | tiny `nn.Linear` 组合；真实 NPU 用例使用同构 tiny 模型 |
| 数据 | 固定随机种子和 bf16/int8 测试输入；不依赖联网下载 |
| 测试版本 | #267 当前单一提交 |
| 代码基线 | 最新 master `e4501b0d` 加 #267 单一提交 |

## 测试用例

表 3 测试用例总表

| 测试用例编号 | 测试项 | 测试用例标题 | 责任人 | 重要级别 | 预置条件 | 输入 | 操作步骤 | 预期结果 | 用例执行情况 |
|-------------|--------|--------------|--------|----------|----------|------|----------|----------|-------------|
| TC-001 | FR-002 | canonical JSON 序列化稳定 | sunshine750 | 高 | 可导入 `quant_result.py` | 等价但顺序不同的 dict | 调用 `canonical_json` 比较字符串 | key 排序、紧凑分隔符和 UTF-8 结果唯一 | Pass |
| TC-002 | FR-003 | 文本、tensor 和文件 SHA-256 格式 | sunshine750 | 高 | hashlib 可用 | 标量文本、bf16/非连续 tensor、文件 | 调用 digest helpers 并重复计算 | 结果带 `sha256:` 前缀且重复运行一致 | Pass |
| TC-003 | FR-003 | layer 文件名规整和路径穿越拒绝 | sunshine750 | 高 | 临时目录 | 含点号、斜杠、`..` 的 layer/file name | 调用 `unit_file_name`、`resolve_unit_path` | 安全字符归一化；绝对路径、目录组件和空名被拒绝 | Pass |
| TC-004 | FR-001 | 结果目录类型和权限校验 | sunshine750 | 高 | pytest tmp_path | None、空串、缺失路径、文件路径、目录路径 | 调用 `validate_quant_result_dir` 或 `quantize` | 非法输入抛明确异常；已有目录可用 | Pass |
| TC-005 | FR-002 | Base 导出 tensor/metadata 分流 | sunshine750 | 高 | 构造声明参数的测试模块 | tensor、标量、None、未声明属性 | 调用 `export_quant_params` | tensor 到 tensors，标量到 metadata，None/未声明值不导出 | Pass |
| TC-006 | FR-002 | Base 加载 key、shape、dtype 校验 | sunshine750 | 高 | 测试模块已有声明 | 合法和非法 tensors/metadata | 调用 `load_quant_params` | 合法值恢复；未知 key、重复 key、shape/dtype/类型错误被拒绝 | Pass |
| TC-007 | FR-002 | Writer 写入一个完整 unit | sunshine750 | 高 | 新空结果目录 | layer、tensor、config、module types | 调用 `write_unit` 并读回 | 生成 safetensors，entry 含 schema、size、SHA-256、entry digest | Pass |
| TC-008 | FR-002 | Writer 防止重复层和文件名碰撞 | sunshine750 | 高 | expected units 含多个层 | 重复 layer、规整后相同文件名 | 连续调用 `write_unit` | 第二次写入失败，不覆盖第一份文件 | Pass |
| TC-008a | FR-002 | Writer 创建阶段拒绝嵌套 expected units | sunshine750 | 高 | 已创建空结果目录 | `blk`、`blk.fc` | 调用 `QuantResultWriter.create` | 在写出任何文件前抛 ValueError；目录保持为空 | Pass |
| TC-009 | FR-003 | manifest-last 和发布状态机 | sunshine750 | 高 | 两个 expected units | 只写部分 unit、全部写入、重复 publish | 调用 `publish_if_complete` | 部分写入返回 False 且无 manifest；全部验证后原子发布；发布后不可写 | Pass |
| TC-010 | FR-003 | writer 写后和发布前完整性复验 | sunshine750 | 高 | 已写 unit | 删除、截断、篡改 unit 或修改 entry | 调用 publish/reader | size、SHA、header、schema、entry digest 任一不符即拒绝 | Pass |
| TC-011 | FR-003 | manifest 严格 schema | sunshine750 | 高 | 已发布测试 artifact | 缺失字段、未知字段、错误类型和 schema version | 修改 manifest 后调用 `load_quant_manifest` | 只接受精确 schema；非法 manifest 被拒绝 | Pass |
| TC-012 | FR-003 | 重复 JSON key 拒绝 | sunshine750 | 高 | 合法 manifest | 重复 `format` 或其它 key | 原始 JSON 读取 | `object_pairs_hook` 抛 duplicate key 错误，不静默覆盖 | Pass |
| TC-013 | FR-003 | manifest 和 unit header 绑定 | sunshine750 | 高 | 已发布 artifact | 错误 artifact_id、layer_name、entry_digest | 调用 `read_quant_unit` | header 与 manifest 不一致时拒绝 | Pass |
| TC-014 | FR-001 | 旧 API 不启用 artifact | sunshine750 | 高 | tiny model | `quantize(model, config)`、`convert(model)` | 不传 result dir 执行旧流程 | 不计算 artifact 指纹、不创建目录、不安装 Writer hook；旧转换行为保持 | Pass |
| TC-015 | FR-001 | 非法结果目录在模型修改前失败 | sunshine750 | 高 | tiny model 和空/缺失目录 | 空路径、文件路径、缺失目录 | 调用带 `quant_result_dir` 的 `quantize` | 抛 TypeError/ValueError，原模型保持原样 | Pass |
| TC-016 | FR-002 | MinMax weight-only 定稿即写入 | sunshine750 | 高 | MinMax config、空目录 | tiny Linear，无 calibration forward | quantize 后检查目录 | weight scale/offset unit 写出；无需 forward 即可发布 manifest | Pass |
| TC-017 | FR-002 | MinMax activation 等最后 batch 写入 | sunshine750 | 高 | activation MinMax、`batch_num=2` | 两次 calibration forward | 每次检查 pending/manifest | 第一次仍 pending；达到 batch_num 后写 activation 参数并发布 | Pass |
| TC-018 | FR-002 | AWQ 在 scale search 完成后写入 | sunshine750 | 高 | AWQ config、tiny Linear | calibration inputs | 执行 forward 并读 artifact | 保存 scale 及需要的 weight quant 参数，不保存可重算原始 weight | Pass |
| TC-019 | FR-002 | GPTQ 保存 Hessian 优化后 weight | sunshine750 | 高 | GPTQ config、tiny Linear | 达到 batch_num 的输入 | 执行 calibration forward 并读 unit | unit 包含优化后 weight、scale/offset；reload 不重新运行 Hessian | Pass |
| TC-020 | FR-004 | MinMax unit reload | sunshine750 | 高 | MinMax artifact 和 fresh float model | manifest + safetensors | 调用 `reload_quant_params` | 对应层变为 MinMaxQuant，参数加载成功且 calibration 已完成 | Pass |
| TC-021 | FR-004 | AWQ unit reload | sunshine750 | 高 | AWQ artifact 和 fresh float model | manifest + safetensors | 调用 reload 并 forward | scale 被加载并重新建立 fake-quant cache，不再次 search | Pass |
| TC-022 | FR-004 | GPTQ unit reload | sunshine750 | 高 | GPTQ artifact 和 fresh float model | manifest + safetensors | 调用 reload 并 forward | 优化后 weight 被恢复，直接进入 fake-quant 状态 | Pass |
| TC-023 | FR-004 | 嵌套 layer 路径 reload | sunshine750 | 高 | `Sequential` tiny model | `block.0`、`block.1` units | 调用 reload | 父模块和属性名解析正确，两层都被替换 | Pass |
| TC-024 | FR-005 | 缺 manifest 的 reload | sunshine750 | 高 | 删除 manifest 的 artifact | fresh model + 目录 | 调用 reload | 抛 FileNotFoundError，模型 state_dict 和子模块对象不变 | Pass |
| TC-025 | FR-005 | 受管文件缺失与无关额外文件 | sunshine750 | 高 | 已发布 artifact | 删除 manifest unit；加入额外 `.safetensors` 或隐藏文件 | 调用目录校验和 reload | manifest 列出的 unit 缺失时拒绝且模型不变；未列举的额外/隐藏文件被忽略且保留 | Pass |
| TC-026 | FR-005 | 文件截断和摘要篡改拒绝 | sunshine750 | 高 | GPTQ/MinMax artifact | truncate、改字节、改 entry digest | 调用 reload | 在第一次 `setattr` 前失败，模型未被部分替换 | Pass |
| TC-027 | FR-005 | 错误 model_type/dtype/架构 | sunshine750 | 高 | 合法 artifact | 不同类、不同 dtype、不同结构模型 | 调用 reload | source/层校验失败；错误模型不发生替换 | Pass |
| TC-028 | FR-004 | 未知算法、原模块类型或量化类 | sunshine750 | 高 | 手工修改 manifest/config | 未注册算法、错误 ori/quant module type、多算法 config | 调用 reload | 受信任注册表查找失败并给出可诊断异常 | Pass |
| TC-029 | FR-004 | 算法能力和必需参数校验 | sunshine750 | 高 | 不支持类或缺参数 unit | 未声明能力、缺 scale/offset/weight | 调用 quantize/reload | 进入 artifact 前或子类 load hook 处失败，不暴露内部实现 flag | Pass |
| TC-030 | FR-005 | reload 两阶段原子性 | sunshine750 | 高 | 多层 artifact，第二层制造错误 | fresh model、篡改第二层 | 调用 reload，保存 state/module id | 第一层不会留下替换；所有验证/构造失败后 live model 不变 | Pass |
| TC-031 | FR-006 | 三算法 direct/reload/convert 等价 | sunshine750 | 高 | 相同 seed、相同 batches、三算法 config | direct model、artifact model、fresh model | 两条路径分别 convert 并比较 deploy model | 部署权重、scale 和输出一致或符合算法明确的近似边界 | Pass |
| TC-032 | FR-006 | artifact convert 拒绝已量化输入 | sunshine750 | 高 | 已经 quantize 的 model 和 artifact | calibration model + artifact | 调用 `convert(model, dir)` | 明确提示应传原始 float model，模型不被破坏 | Pass |
| TC-033 | FR-006 | NPU 参数回设备 | sunshine750 | 高 | 实际 NPU 可用 | NPU tiny model、CPU artifact tensor | reload 后检查 device 和 forward | 磁盘 tensor 为 CPU；模块参数回到 NPU；前向成功 | Pass |
| TC-034 | FR-006 | NPU deploy 前向等价 | sunshine750 | 高 | 真实 torch_npu/NPU | 三算法相同输入 | direct convert 与 artifact convert 后在 NPU 前向 | 两个 deploy model 都可执行且输出一致 | Pass |
| TC-035 | FR-008 | 静态检查和编译检查 | sunshine750 | 高 | 当前 commit | ruff、format、compileall、`git diff --check` | 执行命令并检查返回码 | 本地静态检查、格式和编译检查通过 | Pass |

### 解释：每组到底测了什么

| 用例组 | 要防止的具体问题 | 怎么测 | 通过的判据 | 通过后仍不能说明什么 |
|---------|------------------|--------|------------|----------------------|
| TC-001～TC-006：协议和参数基础 | 同一个对象序列化出不同字符串；摘要不稳定；错误 key、shape、dtype 被悄悄接受 | 构造顺序不同的 dict、标量和非连续 tensor，调用 canonical JSON、digest、Base export/load，并注入未知或重复字段 | 相同逻辑输入得到相同文本/摘要；合法参数恢复；非法字段在模型修改前报错 | 不能说明多个层或完整 API 的生命周期正确 |
| TC-007～TC-013：Writer、unit、manifest | 文件写了一半就被当成完整结果；文件被篡改后仍能加载；重复层覆盖旧结果 | 在临时目录写一个或多个 unit，删除/截断/改字节/修改 header，再执行 publish 或 Reader | 所有受管文件复验通过后才出现 manifest；任一 size、SHA、header、entry digest 不符都拒绝 | 不能说明真实 NPU 能执行这些参数 |
| TC-014～TC-019：旧 API 和算法定稿 | 新参数破坏旧调用；算法在错误的时刻导出；AWQ/GPTQ 保存了不应保存或缺少必需的状态 | 分别不传目录、weight-only、activation 多 batch、AWQ search、GPTQ Hessian 后读取目录 | 旧路径不创建 artifact；三算法在各自定稿点写出声明参数，activation 未完成前不发布 | tiny 模型上的写出不等于正式模型精度 |
| TC-020～TC-023：正常 reload | Reader 不能从 manifest 反查正确算法；嵌套层路径解析错误；恢复后仍处于未完成校准状态 | 用 fresh float model 读取三种算法 artifact，并覆盖 `Sequential` 嵌套层，检查模块类型、参数和 calibration 标志 | 全部目标层恢复，参数 shape/dtype/device 正确，AWQ/GPTQ 不重复搜索/Hessian | 不能说明错误目录和多层部分失败安全 |
| TC-024～TC-030：负例和事务 | 缺文件、篡改文件、错误模型或第二层失败后，第一层被留下造成半量化模型 | 在 manifest、unit、source model、算法名和第二层参数上制造错误，记录 state_dict 和 module id 前后变化 | 错误可定位；第一次 `setattr` 前完成校验；失败后 live model 对象和参数保持原样 | 事务只覆盖 reload 替换，不能推导整个外层进程可回滚 |
| TC-031～TC-034：convert、direct/artifact、NPU | artifact 路径绕开既有 deploy 逻辑，或 CPU 文件恢复后设备不对；两条路径行为不一致 | 同 seed、同输入分别走 direct convert 和 reload 后 convert；设备可用时检查 NPU 参数和前向 | 复用同一 deploy pass；输出在声明误差边界内一致；NPU 参数回到原设备 | tiny/NPU 结果不能替代固定大模型 benchmark |
| TC-035：静态和构建检查 | 代码能在本地运行却无法导入、格式不合或编译失败 | 执行 ruff、format、compileall、`git diff --check`，检查返回码 | 本地命令零错误 | 静态通过不等于功能和精度通过 |

![图 4：TC-007～TC-013 检查 artifact 是否完整发布](../figures/artifact-directory.png)

这张图对应 Writer/manifest 组：左侧的 partial 目录只能留下待处理产物，右侧只有清单和受管文件都能复验时才允许 reload。

![图 5：TC-024～TC-030 检查 reload 失败时是否保持原模型](../figures/reload-transaction.png)

这张图对应负例和事务组：测试不仅断言“抛异常”，还要比较失败前后的 `state_dict` 和子模块对象，确认没有留下半量化状态。

### 用例到自动化测试实现的追踪

表 4 测试实现追踪

| 用例 | 分类 | 主要自动化测试实现 | 说明 |
|------|------|--------------------|------|
| TC-001~TC-004 | UT | `test_quant_result.py::TestCanonicalJson`、`TestUnitFileName`、`TestResultDirValidation` | 验证纯函数、路径和目录前置条件 |
| TC-005~TC-006 | UT | `test_quantize_modules.py::TestBaseQuantParamExport`、`TestBaseQuantParamLoad` | 验证 Base 导出/加载协议，不经过完整 API |
| TC-007~TC-013 | UT/文件集成 | `test_quant_result.py::TestWriterUnitWrites`、`TestStreamingWrite`、`TestManifestPublish`、`TestReadBack`、`TestManifestSchemaIsEnforced`、`TestWriterRefusesUnloadableUnits` | 在临时目录真实写读 JSON 与 safetensors |
| TC-008a | UT | `test_quant_result.py::TestWriterUnitWrites::test_rejects_nested_expected_units_before_any_file_is_written` | 在 Writer 创建阶段拦截嵌套 layer，保证不会产出不可 reload 的完整 artifact |
| TC-014~TC-017 | Classic API tiny 集成 | `ptq/test_quantize.py::test_quantize_without_result_dir_creates_nothing`、`test_quantize_without_result_dir_keeps_pass_signature`、`test_weight_only_publishes_without_any_forward`、`test_activation_publishes_only_after_the_last_calibration_batch` | 从公共 `quantize()` 入口验证兼容性和 MinMax 生命周期 |
| TC-018~TC-019 | 算法 tiny 集成 | `ptq/test_quantize.py::test_awq_is_now_accepted_for_export`、`test_quant_reload.py::TestCalibrationStateEquivalence::test_awq_does_not_persist_weight`、`test_gptq_persists_optimized_weight` | 验证 AWQ/GPTQ 的不同持久化边界 |
| TC-020~TC-023 | reload tiny 集成 | `test_quant_reload.py::TestReloadHappyPath::test_reload_rebuilds_quant_modules`、`test_reload_covers_activation_minmax`、`test_reload_handles_nested_layer_names` | 通过真实注册表、manifest 和 safetensors 重建三算法及嵌套层 |
| TC-024~TC-030 | 负例/事务集成 | `test_quant_reload.py::TestReloadRejectsBadArtifact`、`TestReloadRejectsWrongModel`、`TestReloadRejectsBadStructure`、`TestRequiredQuantParamsPerDtype`、`TestRequiredMinMaxParams`、`TestReloadIsTransactional` | 每个失败测试同时断言 live model 未被部分修改 |
| TC-025 | 目录契约集成 | `TestReloadRejectsBadArtifact::test_missing_unit_file_is_rejected`、`test_extra_unit_file_is_ignored`、`test_hidden_extra_file_is_ignored`；`test_quant_result.py::TestReadBack::test_validate_result_directory_ignores_an_unmanaged_file` | 明确区分“受管文件缺失”和“无关文件共存” |
| TC-031~TC-032 | convert tiny/E2E | `test_quant_reload.py::TestConvertFromQuantResult::test_layerwise_equivalence`、`test_convert_rejects_a_calibrated_model` | 三算法比较直接 convert 与 artifact convert；同时覆盖错误输入 |
| TC-033 | NPU E2E | `ptq/test_quantize.py::test_npu_round_trip_restores_params_to_the_device`、`test_quant_reload.py::TestReloadOnNpu` | 验证 CPU artifact tensor 恢复到 NPU 并可前向 |
| TC-034 | NPU E2E | `test_quant_reload.py::TestDeployForwardOnNpu::test_npu_deploy_forward_matches_direct` | 三算法在真实 NPU 上完成部署转换与前向比较 |
| TC-035 | 静态/CI | 下方 ruff、compileall 和 `git diff --check` | 不把静态检查替代为功能测试 |


## 测试执行命令和结果

### 1. 聚焦 UT、tiny 集成和 NPU 回归

这条命令回答的是“协议、Writer/Reader、Classic API、tiny reload/convert 和条件 NPU 分支是否能在一个聚焦集合内共同通过”。它不是正式大模型 benchmark。

命令：

```bash
python -m pytest -q \
  tests/unit_test/classic/test_quant_result.py \
  tests/unit_test/classic/test_quant_reload.py \
  tests/unit_test/classic/ptq/test_quantize.py \
  tests/unit_test/classic/test_quantize_modules.py
```

实际结果：

```text
386 passed, 13 skipped
```

该命令覆盖 Writer/Reader/schema、Base 和三算法参数机制、Classic API、tiny 模型 reload、convert、事务失败路径以及可用时的实际 NPU 分支。
13 个 skip 均由当前环境缺少真实 NPU 或 torch_npu 部署条件触发；`tests/amct_pytorch/test_config.py` 在本机因外部 torchaudio ABI 与 transformers 导入链冲突无法收集，未计入本轮结果。

### 2. Classic 全目录回归

这条命令检查 Classic 目录下的其它回归用例，回答“新增 artifact 代码有没有影响同目录的旧功能”。它覆盖面更宽，但结果仍受当前测试环境和条件 skip 影响。

命令：

```bash
python -m pytest -q tests/unit_test/classic
```

实际结果：

```text
458 passed, 13 skipped
```

### 3. 静态检查

这些检查不执行量化数学或 NPU 前向，作用是尽早发现导入、语法、格式和编译问题。

命令和结果：

```bash
ruff check \
  amct_pytorch/classic/quant_result.py \
  amct_pytorch/classic/quantize.py \
  amct_pytorch/classic/quantize_op/gptq_module.py \
  amct_pytorch/classic/quantize_op/linear_awq_module.py \
  amct_pytorch/classic/quantize_op/minmax_module.py \
  amct_pytorch/quantize_op/base_quant_module.py \
  tests/unit_test/classic/ptq/test_quantize.py \
  tests/unit_test/classic/test_quant_reload.py \
  tests/unit_test/classic/test_quant_result.py
# All checks passed!

ruff format --check \
  amct_pytorch/classic/quant_result.py \
  amct_pytorch/classic/quantize.py \
  amct_pytorch/classic/quantize_op/gptq_module.py \
  amct_pytorch/classic/quantize_op/linear_awq_module.py \
  amct_pytorch/classic/quantize_op/minmax_module.py \
  amct_pytorch/quantize_op/base_quant_module.py \
  tests/unit_test/classic/ptq/test_quantize.py \
  tests/unit_test/classic/test_quant_reload.py \
  tests/unit_test/classic/test_quant_result.py
# 9 files already formatted

python -m compileall -q \
  amct_pytorch/classic/quant_result.py \
  amct_pytorch/classic/quantize.py \
  amct_pytorch/classic/quantize_op/gptq_module.py \
  amct_pytorch/classic/quantize_op/linear_awq_module.py \
  amct_pytorch/classic/quantize_op/minmax_module.py \
  amct_pytorch/quantize_op/base_quant_module.py
# exit code 0

git diff --check
# exit code 0
```
