# Classic 量化结果持久化与恢复测试用例及测试报告

| 测试负责人 | SWE：sunshine750 | 测试日期 | 2026-08-31 |
|------------|-----------------|----------|------------|
| 被测版本 | `d5a2ac8d77a774fe9eff2721f86d98b321548005` | 分支 | `review/pr267-reviewfix` |
| 测试对象 | AMCT PyTorch Classic Week 1-4 artifact P0 | PR | `https://gitcode.com/cann/amct/merge_requests/267` |


---

## 测试范围和分层

表 1 测试分层

| 层级 | 目标 | 测试位置 | 是否依赖 NPU/大模型 |
|------|------|----------|---------------------|
| UT | 验证 canonical JSON、摘要、schema、路径、Writer、Base 参数机制和单个算法状态 | `tests/unit_test/classic/test_quant_result.py`、`test_quantize_modules.py`、`ptq/test_quantize.py` | 否；NPU 专项除外 |
| Mock/tiny 集成 | 验证 Classic API、真实文件、三算法、模型定位、reload 事务和 convert 链路 | `tests/unit_test/classic/test_quant_reload.py`、`ptq/test_quantize.py` | tiny model；deploy/NPU 分支按条件执行 |
| E2E/NPU | 验证参数回到 NPU、reload 后前向和 deploy 前向与直接路径一致 | `test_quant_reload.py::TestReloadOnNpu`、`TestDeployForwardOnNpu`、`ptq/test_quantize.py` NPU 用例 | 需要实际 torch_npu 和 NPU |
| CI/静态 | 验证代码可导入、格式和 PR compile 流程 | ruff、compileall、GitCode `/compile` | 否；由 CI 环境执行 |

## 测试环境和固定条件

表 2 测试环境

| 项目 | 值 |
|------|----|
| Python | 3.11.4 |
| PyTorch | 2.7.1+cpu（同时加载 `torch_npu` 插件） |
| torch_npu | 2.7.1.post4 |
| safetensors | 0.8.0 |
| pytest | 9.0.3 |
| ruff | 0.15.14 |
| 模型 | tiny `nn.Linear` 组合；真实 NPU 用例使用同构 tiny 模型 |
| 数据 | 固定随机种子和 bf16/int8 测试输入；不依赖联网下载 |
| 测试版本 | commit `d5a2ac8d` |

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
| TC-032 | FR-006 | artifact convert 拒绝已量化输入 | sunshine750 | 已经 quantize 的 model 和 artifact | calibration model + artifact | 调用 `convert(model, dir)` | 明确提示应传原始 float model，模型不被破坏 | Pass |
| TC-033 | FR-006 | NPU 参数回设备 | sunshine750 | 实际 NPU 可用 | NPU tiny model、CPU artifact tensor | reload 后检查 device 和 forward | 磁盘 tensor 为 CPU；模块参数回到 NPU；前向成功 | Pass |
| TC-034 | FR-006 | NPU deploy 前向等价 | sunshine750 | 真实 torch_npu/NPU | 三算法相同输入 | direct convert 与 artifact convert 后在 NPU 前向 | 两个 deploy model 都可执行且输出一致 | Pass |
| TC-035 | FR-008 | 静态检查和 PR compile | sunshine750 | 当前 commit | ruff、format、compileall、`/compile` | 执行命令并检查 PR 标签 | 本地静态检查通过；GitCode CI 返回 `ci-pipeline-passed` | Pass |


### 用例到自动化测试实现的追踪

表 4 测试实现追踪

| 用例 | 分类 | 主要自动化测试实现 | 说明 |
|------|------|--------------------|------|
| TC-001~TC-004 | UT | `test_quant_result.py::TestCanonicalJson`、`TestUnitFileName`、`TestResultDirValidation` | 验证纯函数、路径和目录前置条件 |
| TC-005~TC-006 | UT | `test_quantize_modules.py::TestBaseQuantParamExport`、`TestBaseQuantParamLoad` | 验证 Base 导出/加载协议，不经过完整 API |
| TC-007~TC-013 | UT/文件集成 | `test_quant_result.py::TestWriterUnitWrites`、`TestStreamingWrite`、`TestManifestPublish`、`TestReadBack`、`TestManifestSchemaIsEnforced`、`TestWriterRefusesUnloadableUnits` | 在临时目录真实写读 JSON 与 safetensors |
| TC-014~TC-017 | Classic API tiny 集成 | `ptq/test_quantize.py::test_quantize_without_result_dir_creates_nothing`、`test_quantize_without_result_dir_keeps_pass_signature`、`test_weight_only_publishes_without_any_forward`、`test_activation_publishes_only_after_the_last_calibration_batch` | 从公共 `quantize()` 入口验证兼容性和 MinMax 生命周期 |
| TC-018~TC-019 | 算法 tiny 集成 | `ptq/test_quantize.py::test_awq_is_now_accepted_for_export`、`test_quant_reload.py::TestCalibrationStateEquivalence::test_awq_does_not_persist_weight`、`test_gptq_persists_optimized_weight` | 验证 AWQ/GPTQ 的不同持久化边界 |
| TC-020~TC-023 | reload tiny 集成 | `test_quant_reload.py::TestReloadHappyPath::test_reload_rebuilds_quant_modules`、`test_reload_covers_activation_minmax`、`test_reload_handles_nested_layer_names` | 通过真实注册表、manifest 和 safetensors 重建三算法及嵌套层 |
| TC-024~TC-030 | 负例/事务集成 | `test_quant_reload.py::TestReloadRejectsBadArtifact`、`TestReloadRejectsWrongModel`、`TestReloadRejectsBadStructure`、`TestRequiredQuantParamsPerDtype`、`TestRequiredMinMaxParams`、`TestReloadIsTransactional` | 每个失败测试同时断言 live model 未被部分修改 |
| TC-025 | 目录契约集成 | `TestReloadRejectsBadArtifact::test_missing_unit_file_is_rejected`、`test_extra_unit_file_is_ignored`、`test_hidden_extra_file_is_ignored`；`test_quant_result.py::TestReadBack::test_validate_result_directory_ignores_an_unmanaged_file` | 明确区分“受管文件缺失”和“无关文件共存” |
| TC-031~TC-032 | convert tiny/E2E | `test_quant_reload.py::TestConvertFromQuantResult::test_layerwise_equivalence`、`test_convert_rejects_a_calibrated_model` | 三算法比较直接 convert 与 artifact convert；同时覆盖错误输入 |
| TC-033 | NPU E2E | `ptq/test_quantize.py::test_npu_round_trip_restores_params_to_the_device`、`test_quant_reload.py::TestReloadOnNpu` | 验证 CPU artifact tensor 恢复到 NPU 并可前向 |
| TC-034 | NPU E2E | `test_quant_reload.py::TestDeployForwardOnNpu::test_npu_deploy_forward_matches_direct` | 三算法在真实 NPU 上完成部署转换与前向比较 |
| TC-035 | 静态/CI | 下方 ruff、compileall、`git diff --check` 和 PR `/compile` | 不把静态检查替代为功能测试 |


## 测试执行命令和结果

### 1. 聚焦 UT、tiny 集成和 NPU 回归

命令：

```bash
python -m pytest -q \
  tests/unit_test/classic/test_quant_result.py \
  tests/unit_test/classic/test_quant_reload.py \
  tests/unit_test/classic/ptq/test_quantize.py \
  tests/unit_test/classic/test_quantize_modules.py \
  tests/amct_pytorch/test_config.py
```

实际结果：

```text
432 passed, 6 warnings, 3 subtests passed in 26.63s
```

该命令覆盖 Writer/Reader/schema、Base 和三算法参数机制、Classic API、tiny 模型 reload、convert、事务失败路径以及
可用时的实际 NPU 分支。warning 为环境权限、pytest 配置和 torch_npu internal-format 提示，不是测试失败。

### 2. Classic 全目录回归

命令：

```bash
python -m pytest -q tests/unit_test/classic
```

实际结果：

```text
388 passed, 3 warnings, 3 subtests passed in 6.80s
```

### 3. 静态检查

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

### 4. GitCode PR compile

已在 PR #267 发送 `/compile`。远程 API 当前返回：

```text
state=open
mergeable=true
ci_state_passed=true
label=ci-pipeline-passed
```

这证明当前 head `d5a2ac8d` 的 PR 编译流水线通过；它不等同于 PR 已经合入。



## 缺陷记录&测试报告

### 测试报告

#### 1. 测试质量评估

本次测试覆盖了四类高风险行为：协议字段和摘要、三算法参数导出/加载、模型身份和目录完整性、reload/convert 的原子性与
部署等价。UT 负责快速定位，tiny 集成负责验证真实文件和真实注册表，NPU 用例负责确认 device 和 deploy 前向。
聚焦回归和 Classic 全目录均为零失败，说明当前实现对已声明范围具有较完整的自动化覆盖。

#### 2. Fail、Block、Cancel 说明

当前登记的 35 个用例均为 Pass。完整 `tests/unit_test` 的历史 collection 阻塞不归因于本次代码，不能改记为 Pass；它应在
正式 CI 或补齐依赖后重新执行。若 NPU 不可用，TC-033/TC-034 应标记为 Block 并记录设备、torch_npu 和 CANN 版本。

#### 3. 测试遗留问题

| 遗留项 | 影响 | 原因 | 后续计划 |
|--------|------|------|----------|
| 旧 artifact 兼容策略未定 | 可能影响跨版本 reload | 老师 review 删除旧字段，但未明确 schema v2 | 需老师确认 |
| 完整 tests/unit_test 收集依赖不齐 | 无法一次性给出全仓零错误结论 | 环境缺 `torchao`/transformers 模型模块 | 在指定 CI 环境补跑 |
| 正式大模型 E2E 口径未冻结 | 里程碑签收证据可能需要补充 | 当前自动化以 tiny/NPU 为主 | 需指定模型、配置和输入 |
| PR discussion 未全部 resolved、需要 squash | 平台 merge check 仍可能为 false | 仓库流程状态，不是功能失败 | 需在平台操作 |

#### 4. 后续测试建议

- 先确认 schema 版本和旧 artifact 兼容策略，再增加兼容性用例；不要在结论未定前写大量重复实验。
- 在老师指定的固定模型/配置上补一条端到端 artifact -> convert -> deploy 复现命令。
- 新算法接入时只增加该算法的最小参数矩阵、负例和等价性测试，不复制已有三算法的全部测试文本。

### Bug List

表 5 Bug List

| 序号 | 缺陷描述 | 严重程度 | 状态 |
|------|----------|----------|------|
| 1 | 早期协议要求的 producer/library_version 与当前 review 后 schema 不一致 | 中 | 已按老师 review 修改，兼容策略待确认 |
| 2 | 完整全仓测试受外部依赖缺失影响 | 中 | 未修改，已隔离并记录，不属于本次代码缺陷 |
| 3 | GitCode PR 仍显示 needs-squash/未全部 resolved | 低 | 平台流程待维护者处理 |

## 检视意见

### 检视结论

| 检视日期 | 检视人 | 检视结论 | 遗留风险 |
|----------|--------|----------|----------|
| 待评审 | 老师/PL | 待评审 | schema 兼容、正式 E2E 口径、平台 discussion/squash 状态 |

### 检视问题清单

表 6 检视问题清单

| 序号 | 问题描述 | 问题类别 | 严重程度 | 责任人 | 计划闭环时间 | 当前状态 |
|------|----------|----------|----------|--------|--------------|----------|
| 1 | 每个用例是否需要补独立 issue 链接 | 用例覆盖 | 低 | PL | 文档评审时 | 待确认 |
| 2 | TC-025 的额外文件规则是否符合最终协议 | 预期结果 | 中 | PL | PR 合入前 | 待确认 |
| 3 | NPU tiny E2E 是否足以作为 Week 4 里程碑证据 | 测试结论 | 中 | PL | 里程碑验收时 | 待确认 |
| 4 | 完整 tests/unit_test 是否必须在指定 CI 环境执行 | 执行结果 | 中 | PL | CI 复核时 | 待确认 |

### 检视问题闭环记录

表 7 检视问题闭环记录

| 序号 | 闭环说明 | 修改章节 | 闭环确认人 | 确认日期 |
|------|----------|------------|------------|----------|
| 1 | 根据老师意见补 issue 链接或统一使用 PR #267 追踪 | 测试用例总表 | 待指定 | 待确认 |
| 2 | 根据最终协议决定 TC-025 的目录集合断言 | TC-025、测试报告 | 待指定 | 待确认 |
| 3 | 根据指定模型补正式端到端命令、日志和产物清单 | 测试执行命令和结果 | 待指定 | 待确认 |
