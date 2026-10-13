# 里程碑 2：Classic 逐层量化测试用例及测试报告

| 文档属性 | 内容 |
|---|---|
| 版本 / 复核日期 | 2.0 / 2026-09-28 |
| 历史运行记录 | 2026-09-23 交付包；本次未重新运行模型或 pytest |
| 范围 | Week 5–8；layerwise P0；V2 与 Classic full 相关回归 |
| 需求与设计 | [需求规格说明书](classic_layerwise_requirement.md)、[软件设计说明书](classic_layerwise_design.md) |
| 判定原则 | 每条结论写清对象、断言和证据；不能把代码存在、报告摘录、完整运行日志都称为同一种 PASS |

## 1 测试结论与证据等级

里程碑 2 已有本地模型入口、逐层缓存、逐 FFN Linear MinMax 参数发布与 fresh reload 的功能实现。留存 pytest 日志分别记录 79 passed、506 passed/13 skipped、350 passed/13 skipped。历史 Qwen3-0.6B 记录覆盖 28 层、84 units 的 CPU 功能冒烟；真实实验保留了脚本与旧报告摘录，没有完整独立运行日志和仍在原路径的完整临时产物。

因此本报告的结论是“P0 功能链路已有实现与分层证据”，不是“所有算法、正式精度、NPU 部署与性能均通过”。统一 Solver 主调度、严格真实缓存复用、写中失败注入和正式精度/部署/性能仍有独立补测或接入项。

2026-10-13 提交前复核：代码分支重放到 Week 3-4 PR #267 最新基线 `71a7b181` 后，聚焦单测为 `127 passed, 1 skipped`（skip 原因为本机无 Ascend NPU），`pre-commit` 与 `git diff --check` 通过。该复核不改变上文对历史真实模型证据和待验收边界的判断。

![证据种类与结论范围](figures/10-evidence.svg)

### 1.1 状态定义

| 标记 | 含义 | 使用约束 |
|---|---|---|
| L：历史日志支持 | 测试代码的相应断言处于留存 pytest 已通过范围 | 日志多为汇总，不冒充本次逐用例新运行 |
| C：源码可核 | 当前实现中可以读取该行为，缺专门运行断言 | 不能标成独立测试通过 |
| H：历史报告记录 | 有留存脚本与旧报告输出摘录，无完整独立运行日志 | 可说明历史记录及脚本判据，证据强度低于完整日志 |
| O：观测值 | 脚本只记录数值，未断言阈值 | 不据此宣称质量达标 |
| S：SKIPPED | 测试存在，前置环境不满足而跳过 | 不是通过，不计入已执行功能 |
| N：未完成验收 | 没有对应执行记录或断言不足 | 明确补测方案，不用“未发现问题”替代 |

本报告的 TC 是按验收主题重新组织的文档编号，不是 pytest 测试数量。参数化会让一个主题对应多个 pytest case，同一文件也会被不同回归集合重复执行。

### 1.2 三组日志与证据链接

| 集合 | 留存摘要 | 范围 | 证据 |
|---|---|---|---|
| layerwise + solver | 79 passed，30.66s | layerwise 文件与 CalibrationSolver 组件文件 | [focused 日志](../evidence/layerwise_milestone2_lastfocused.log) |
| Classic + optimization | 506 passed，13 skipped，33.89s | 两个测试目录 | [regression 日志](../evidence/layerwise_milestone2_lastregression.log) |
| 算法/V2 选定文件 | 350 passed，13 skipped，27.59s | 5 个指定文件；含协议、reload、模块、入口和算法注册 | [算法/V2 日志](../evidence/final_detailed_algorithm_matrix.log) |

三组有交集，不能相加为 935 个独立测试，更不能称为 350 种算法或 350 个真实模型实验。Ruff 与 `git diff --check` 的通过来自历史报告记录，本次未重新运行源码检查。

## 2 测试范围、环境与固定条件

### 2.1 风险分层

| 层级 | 要防止的问题 | 合适的测试对象 |
|---|---|---|
| 入口与纯配置 | 走错路径、拼写被忽略、样本数矛盾 | 参数化小用例 |
| 数据与缓存 | 输入规模错误、旧缓存串档、文件损坏 | 本地小 Dataset 与 mock Adapter |
| Linear / 算法组件 | 错投影输入、dtype 错误、导出时点错误 | tiny MLP、真实 Linear、Solver 组件 |
| 文件协议与恢复 | 半成品发布、文件篡改、错误模型、部分替换 | 真正 JSON/safetensors 文件与 fresh tiny 模型 |
| 真实模型功能 | Adapter 包装、全层命名、真实权重身份 | Qwen3-0.6B 全 28 层 CPU |
| 精度 / 部署 / 性能 | 数值质量、设备算子、内存与时延 | 代表性数据、真实 NPU、固定 benchmark 条件 |

### 2.2 历史环境

| 项目 | 历史记录 |
|---|---|
| 源码 | `amct-layerwise-dev` |
| 代码基线 | aedab74f 加本地修改；仅基线号不足以唯一重建实际工作树 |
| PyTorch | 2.7.1+cpu |
| torch_npu | 2.7.1.post4；安装包存在不等于真实 NPU 运行可用 |
| transformers / datasets | 5.12.1 / 5.0.1 |
| torchaudio | 2.11.0，与 torch ABI 不匹配，导入报告 undefined symbol: torch_library_impl |
| 真实模型 | `Qwen3-0.6B` |
| 真实模型设备 / dtype | CPU / bfloat16 |
| 静态校准规模 | 1 个样本 × 16 tokens；重复英语短句 |
| 历史临时处理 | `/tmp/layerwise_audio_shim/sitecustomize.py` 绕过音频导入障碍 |

音频 ABI 与 NPU skip 是两个问题。前者是导入障碍，历史通过临时 shim 运行；后者是设备和真实 torch_npu 前置条件不足，shim 不会补出 NPU 能力。2026-09-28 复核时该 shim 与历史 `/tmp/real-*` 产物均不在原位置，不能把历史命令直接称为当前可重放环境。

后续重跑应保存 Python、pip freeze、AMCT 源码补丁/文件摘要、模型 config/权重摘要、tokenizer 版本、dataset 内容身份、设备状态、完整命令和 stdout/stderr。只写基线 commit 和 PASS 不足以支持完全复现。

## 3 Week 5：入口与准备验收

本周独立验收对象是“给定调用是否得到唯一计划或明确拒绝”。这些用例不要求真正执行完整模型，不用模型前向耗时证明配置是否正确。

下列 `test_*` 均位于 [test_layerwise.py](../../tests/unit_test/classic/test_layerwise.py)，类名同时给出以便定位。

| TC | FR / DS | 目的、操作与预期 | 当前证据 |
|---|---|---|---|
| TC-001 | 001 | 不传 pipeline，传 nn.Module；应进入原 full 路径 | L；TestPipelineDispatch::test_default_pipeline_uses_original_full_path |
| TC-002 | 001 | 显式 full + nn.Module；原入口行为保持 | L；test_explicit_full_uses_original_full_path |
| TC-003 | 001 | str 缺省或显式 layerwise；分派到 layerwise | L；test_string_path_infers_layerwise_when_pipeline_is_omitted、test_config_pipeline_selects_layerwise |
| TC-004 | 001 | layerwise + nn.Module、full + str；应明确拒绝，不伪完成 | L；test_layerwise_rejects_module_before_dispatch、test_full_rejects_string_path |
| TC-005 | 001 | 未知值、非字符串 config.pipeline、额外 pipeline 关键字；应报错 | L；test_rejects_unknown_pipeline、test_rejects_non_string_config_pipeline、test_rejects_pipeline_keyword |
| TC-006 | 002 | 空/非法路径、非合法 config、缺 model_type；应有明确错误 | L；TestModelResolution::test_rejects_invalid_local_model_path、test_rejects_invalid_model_config |
| TC-007 | 002 | 合法本地 config 与唯一注册 metadata；应唯一解析 | L；test_loads_local_config_without_transformers、test_resolves_unique_registry_metadata |
| TC-008 | 002 | 零匹配和多匹配；应列可用项或拒绝歧义 | L；test_unknown_model_type_lists_available_adapters、test_duplicate_model_type_metadata_is_rejected |
| TC-009 | 002 | 对未知类型监控 import，对合法类型监控构造；仅处理目标 Adapter | L；test_model_type_specific_registration_is_lazy、test_unknown_model_type_registration_does_not_import、TestPrepareLayerwiseQuantization::test_constructs_exactly_the_resolved_mock_adapter |
| TC-010 | 003 | 翻译 weight-only、无算法、静态 MinMax、int4 group；内部字段符合配置 | L；TestBuildArgsFromConfig::test_maps_weight_only_minmax_without_cli_defaults、test_maps_weight_only_without_calibration_algorithm、test_maps_static_activation_minmax、test_maps_int4_group_weight_config |
| TC-011 | 003 | 参数化样本数关系；缺省/推导合法，矛盾或非正值拒绝 | L；test_calibration_sample_relationship、test_rejects_invalid_calibration_shape |
| TC-012 | 003 | 不支持算法、MoE、dynamic/KV/fuzzy、未知字段；应拒绝 | L；test_rejects_features_outside_week5_p0、test_rejects_moe_outside_layerwise_p0、test_rejects_silently_ignored_config_fields |
| TC-013 | 003 | 错配置不构造 Adapter，工作与结果目录重叠拒绝；prepare 不提前创建目录 | L；test_bad_config_fails_before_adapter_construction、test_rejects_work_dir_overlapping_result_dir、test_defaults_result_directory_without_creating_it |

### 3.1 如何判断早期失败真的成立

TC-013 不应只检查抛错文本，还要记录 Adapter 构造次数；错误出现前没有构造才能证明“配置早失败”。这一结论只覆盖已知配置错误。dataset 实际内容不足、MLP 运行结构错误发生在后续阶段，不能因此声称所有错误均在 Adapter 前完成检查。

### 3.2 本周能证明什么

入口与准备测试可以证明分派规则、参数翻译和拒绝边界与实现一致，不能证明被解析出的所有真实模型均可完整运行。真实 Adapter 的权重布局、包装与 dtype 需要 Week 7/8 继续验证。

## 4 Week 6：数据与缓存验收

本周独立验收对象是固定 token 窗口、是否需要 Extract，以及缓存的身份和完整性。文件数量、文件内容和调用次数是不同断言，不能相互替代。

| TC | FR / DS | 目的、操作与预期 | 当前证据 |
|---|---|---|---|
| TC-014 | 004 | load_from_disk、单 split、text 列；不满足 schema 明确失败 | C；load_calibration_data 源码有检查；未找到覆盖全部 schema 分支的独立参数化用例，不宣称逐分支已测 |
| TC-015 | 004 | 空 text、token 数不足；不得补造样本 | L；TestWeek6CalibrationData::test_rejects_empty_text_and_insufficient_tokens；非字符串/零 token 分支另为 C |
| TC-016 | 004 | 顺序编码、拼接、切窗；样本数/shape 和首窗口内容正确 | L；test_concatenates_and_cuts_exact_token_windows，mock tokenizer 将字节映射为 token |
| TC-017 | 005 | weight-only 无 dataset；Extract 调用为空，仍发布 6-unit mock artifact | L；test_needs_extract_branches、TestLayerwiseEndToEnd::test_weight_only_uses_no_dataset_and_publishes_v2 |
| TC-018 | 005 | 静态配置触发 Extract；记录 block cache 与 manifest | L；test_static_minmax_extracts_and_writes_activation_params；此 mock 用例实际只断言 scale_w，不能以名称证明 scale_d |
| TC-019 | 006 | 审查 cache key 字段；模型/路径/采样进入，bit/算法不进入 | C；_activation_cache_manifest；未把同路径 dataset 内容变化标为已识别，补 BT-01 |
| TC-020 | 007 | 完整缓存后换新 result dir；Extract 调用数不增加且新 manifest 可读 | L；test_complete_activation_cache_is_reused |
| TC-021 | 007 | 把 block_0 pkl 改成 corrupt 后重跑；应增加 Extract 调用 | L；test_corrupt_activation_cache_is_re_extracted |
| TC-022 | 007 | manifest 原子写与预期文件 size/SHA 比较；失败不给完整缓存 | C；_publish_activation_cache_manifest、_has_valid_activation_cache；当前有 mock 损坏重抽，不等于覆盖全部中断位置；合法 JSON 非 mapping 的 AttributeError 尚未防御 |

![缓存检查与已知缺口](figures/05-cache.svg)

### 4.1 TC-020 / TC-021 的详细步骤

前置条件为两层 mock Adapter 与有效静态 config。第一次运行完成后保存 `len(adapter.extract_calls)`。第二次把量化结果目录换成新目录，保持缓存目录和身份一致；预期 Extract 次数不变，新 V2 manifest 可解析。这直接回答“复用是否真的跳过 Extract”。

损坏用例在第二次执行前覆盖一份 pkl，再比较 Extract 次数增加。它验证损坏不会被静默消费，但没有分别注入每个文件、每个写入字节位置或每一种身份字段变化。后续可根据风险补齐，不将单例损坏测试夸大为完整故障矩阵。

### 4.2 数据内容与文件内容是两个层次

cache files 的 SHA-256 检查中间激活有没有被改坏；dataset 路径字段检查数据从哪个目录选择。这并不证明同目录 text 没有被覆盖。同样，tokenizer.json 的摘要也不覆盖所有 tokenizer 配置及库版本。BT-01 必须为来源身份补内容规则，不能再加一遍 pkl hash 就声称解决来源问题。

### 4.3 本周能证明什么

mock 缓存调用计数能证明对应逻辑在受控模型上跳过/重抽。真实模型第二轮文件数未变只证明目标文件集合仍存在，无法排除重新生成了同名文件。真实复用验收见 BT-02。

## 5 Week 7：逐 Linear、MinMax 与发布验收

本周独立验收对象是正确的 unit 与输入、参数导出、完成发布和恢复身份。需要把“调用发生”“参数正确”“异常发生在目标阶段”分别断言。

| TC | FR / DS | 目的、操作与预期 | 当前证据 |
|---|---|---|---|
| TC-023 | 008 | 两层 dense MLP；manifest 恰好列出 gate/up/down 共 6 个全局路径 | L；test_weight_only_uses_no_dataset_and_publishes_v2 检查完整有序层名 |
| TC-024 | 008 | 以 QuantLinear 风格包装 Linear；应解析原模块并记录 ori_module_type=Linear | L；test_plan_accepts_quant_linear_wrapped_dense_mlp |
| TC-025 | 009 | 捕获三个投影的真实输入并对齐 dtype；down 应等于门控结果 | C；_capture_linear_inputs、_forward_mlp_for_capture、_quantize_linear 可核；历史真实链路可执行，独立数值比较待 BT-03 |
| TC-026 | 010 | 对称 weight-only 导出仅 scale_w，六个 unit 完整发布 | L；test_weight_only_uses_no_dataset_and_publishes_v2 |
| TC-027 | 010 | 对称静态 MinMax 导出 scale_d+scale_w，批数达到要求 | H；真实 Qwen 脚本逐 unit 直接断言两个 key；组件源码有定稿逻辑。不能用 TC-018 的 mock 名称替代该证据 |
| TC-028 | 011 | Solver 对两 batch 逐次 forward、透传 kwargs；拒绝多输入 batch | L；test_calibration_solver_runs_each_batch_without_gradient_state、test_calibration_solver_rejects_multi_input_batches |
| TC-029 | 011 | Solver finalize 的 export_ptq_params fallback 返回参数 | L；test_calibration_solver_exports_ptq_params_on_finalize；优先 export_quant_params 分支为源码可核，主路径接入待 BT-04 |
| TC-030 | 012 | Writer 只在全部 expected units 完成并复验后发布；重复/缺失拒绝 | L；V2 相关 test_quant_result.py 的 TestManifestPublish / TestWriterUnitWrites / TestReadBack 范围 |
| TC-031 | 012 | 参数文件和 manifest 的 schema、size、摘要、header 绑定 | L；test_quant_result.py 与 test_quant_reload.py 对应协议负例；本报告引用既有回归，不称新增 layerwise 独立用例 |
| TC-032 | 012 | 无 manifest 时 Reader 拒绝完整加载 | L；test_quant_reload.py::TestReloadRejectsBadArtifact；layerwise 已写部分 unit 后的定点失败待 BT-05 |
| TC-033 | 013 | 流式 checkpoint 指纹与同权重 full 模型指纹一致 | L；TestSourceFingerprint::test_streamed_checkpoint_digest_matches_full_model |
| TC-034 | 013 | fresh dense 模型恢复为 MinMaxQuant；错误恢复不部分修改 | L；test_artifact_reloads_into_fresh_dense_model；事务负例来自 test_quant_reload.py::TestReloadIsTransactional |
| TC-035 | 014 | 第二层 build_quant_block 抛异常，结果无 manifest | L；test_failure_leaves_no_manifest；异常可能在前置规划阶段，不能称已验证 mid-write failure |

### 5.1 TC-025：为何要求独立数值对照

正例构造已知权重与非平凡输入 x，独立计算 `gate=gate_proj(x)`、`up=up_proj(x)`、`middle=act_fn(gate)*up`。捕获后分别比较 gate_input 与 x、up_input 与 x、down_input 与 middle。测试应选择 middle 与 x 不相同的情况；仅断言三个 key 存在或 shape 合法容易掩盖错误路由。

对真实包装再重复一遍，覆盖 `.linear` 的解析及 wrapper 的 forward。当前实现路径可核，旧模型冒烟说明它可执行；完整数值 oracle 尚待 BT-03，不伪装成已有用例。

![真实 Linear 输入的数值对照对象](figures/06-mlp.svg)

### 5.2 TC-028 / TC-029：Solver 测试不能代替调度测试

组件测试构造记录 calls 的模块，输入两个 batch 并透传 multiplier=3，断言每批内容和 kwargs。finalize 测试验证 export_ptq_params fallback；多输入 batch 测试要求 ValueError。

源码在 solver 内部使用 no_grad，但现有测试外部也包了 no_grad，因此该用例不是“关闭外层保护后仍不建图”的独立证明。BT-04 可以加入外层启用梯度、输入 requires_grad、模块记录 `torch.is_grad_enabled()` 的检查。更关键的是当前 layerwise 没调用 solve/finalize，组件通过不代表统一调度已经接入。

### 5.3 TC-035：必须检查故障注入位置

旧用例把第二层 `build_quant_block` 替换成抛错函数，最终断言没有 quant_manifest。主流程在写参数之前会遍历所有层规划 units，因此异常可能先在规划发生；“第二层失败”并不自动等于“第一层参数写完后失败”。

该用例继续作为运行失败不发布的有效保护。要检验写中状态，应在 Writer 已成功写出第一个或第 k 个 unit 后注入异常，确认磁盘上确有部分 unit、无 manifest、Reader 拒绝，随后用新目录正常重跑。对应 BT-05。

![写中失败必须发生在 unit 文件产生之后](figures/08-publish.svg)

### 5.4 本周能证明什么

已有代码与日志说明真实 Linear 单位、基础导出、V2 完整性和 fresh reload 有保护；尚不能把所有 hook 数值、全部写中故障和 Solver 主调用路径列为已完成验收。每个缺口都有可执行补测定义，而不是泛称“继续加强测试”。

## 6 Week 8：真实 Qwen3-0.6B 独立报告

真实模型证据与 pytest 数量分开记录。运行脚本为 [test_real_layerwise_matrix.py](../evidence/test_real_layerwise_matrix.py)，结果摘录来源为 [历史测试报告](../evidence/历史测试报告_20260923.md)。本节状态均为 H，说明历史报告记录与当前脚本判据，不冒充 2026-09-28 新运行。

### 6.1 为什么选择 Qwen3-0.6B

该模型在当前 CPU 环境可完成完整层遍历，同时使用真实 tokenizer、真实 safetensors、28 层 Adapter 与门控 MLP 包装，能覆盖 mock 难以发现的层路径、dtype、源指纹和包装差异。每层三个 FFN Linear，84 units 足以检查全层命名及发布完整性。

模型规模选择是功能验收取舍。单个 16-token 样本刻意缩小校准计算，适合验证执行链路；它不具备代表性数据覆盖，不能支持 PPL、准确率、生成质量或长期稳定性结论。

![真实 Qwen 的层与参数单元覆盖](figures/11-qwen-coverage.svg)

### 6.2 QW-01：weight-only 参数生成

**前置与配置：** 本地 Qwen3-0.6B、CPU、1 batch、batch size 1、1 样本、seq_len 16；权重 int8、对称、channel；算法 MinMax；不提供 dataset。

**操作：** prepare，断言 `_needs_extract=False`，运行 layerwise，读取 manifest。

**脚本断言：** units 数为 84；每个 unit 的 tensor key 恰好为 scale_w。

**历史记录：** `weight-only: PASS, units=84, reload=MinMaxQuant, forward=finite`。

**说明：** 使用真实权重且所有 FFN 参数单元完成；“无 dataset”不意味着 Adapter 无 tokenizer 初始化依赖。也没有比较权重量化误差分布或速度。

### 6.3 QW-02：weight-only fresh reload 与前向

**操作：** 从相同源目录新加载完整 bf16 模型；比较 `compute_model_fingerprint(model)` 与 manifest.source_model；恢复参数。

**脚本断言：** 首层 gate 和第 27 层 down 的类型为 MinMaxQuant；输入 `[[151643,1,2,…,15]]`；logits shape 为 `(1,16,151936)`；所有输出为有限数。

**证明：** 参数能绑定对应源模型、恢复目标路径可用、完整模型前向可执行。

**不证明：** 相对原 FP 模型误差小；文本生成正确；正式数据集精度达标；NPU 部署成功。fresh 是重新加载对象，不能仅凭这段同进程脚本声称已经做了 fresh-process 验证。

### 6.4 QW-03：静态 MinMax 参数与缓存

**数据：** `Dataset.from_dict` 保存两条相同英语短句，拼接后只取 1×16 tokens。权重 int8 对称 channel；输入 int8 对称 tensor；其余规模同 QW-01。

**操作与断言：** `_needs_extract=True`；数据加载得到 1 个 `(1,16)` 样本；run 后 84 units；每个 unit 的 tensor key 恰好为 `['scale_d','scale_w']`；cache manifest 的 files 数为 28。

**历史记录：** `static-minmax: PASS, units=84, cache_files=28, reuse=PASS, reload+forward=PASS`。其中 reuse 字样需按 QW-05 的实际断言解读，不能直接作为严格零 Extract 证明。

### 6.5 QW-04：静态 MinMax fresh reload 与前向

重新加载同源 bf16 模型，比较 fingerprint，执行 reload 和相同输入前向；脚本断言 shape 与 finite。该路径把激活参数生成、参数文件读取和完整前向连接起来，是功能证据。

没有原 FP baseline 的 logits 误差、交叉熵、PPL 或任务分数。因此“静态 MinMax 参数存在且可运行”与“此校准数据产生合格精度”必须区分。

### 6.6 QW-05：第二轮生成与算法拒绝

第一次静态运行后换成 result-reuse 目录，再次执行同一 Plan。脚本比较前后 pkl 数量均为 28、第二份 manifest.source_model 与第一份相同。它证明第二轮产物生成完成，未检查 Extract 调用、文件 mtime 或内容是否被重写，所以严格真实复用状态为 N，待 BT-02。

脚本还分别传 AWQ、GPTQ、SmoothQuant，预期错误包含 `only no algorithm or algorithm 'minmax'`；这证明拒绝边界，不能反过来标成算法支持。

### 6.7 真实结果总表

| 项目 | weight-only | 静态 MinMax | 证据能支持的表述 |
|---|---|---|---|
| 模型范围 | 28 层、84 units | 28 层、84 units | 全 FFN 参数单元功能覆盖 |
| 对称参数 | scale_w | scale_w + scale_d | 指定配置参数集合成立 |
| 校准数据 | 不读取 dataset | 1×16 token | 功能输入规模明确 |
| block cache | 不适用 | 28 文件 | 缓存集合有 28 项 |
| fresh fingerprint | 脚本相等断言 | 脚本相等断言 | 同源模型身份可匹配 |
| forward | shape 正确、finite | shape 正确、finite | 完整前向可执行 |
| 原模型质量差异 | 未验收 | 未验收 | 不能给精度结论 |
| 实际 NPU 部署 | 未执行 | 未执行 | 不宣称整数算子运行 |

## 7 Classic full 对照与通用检查

| TC | FR / DS | 对象、操作、预期 | 当前证据 |
|---|---|---|---|
| TC-036 | 015 | 两个 1024×1024 Linear，MinMax/GPTQ；落盘前后量化输出 bit exact | H；[full 脚本](../evidence/test_real_full_algorithms2.py) 中 torch.equal；不是 Qwen 算法实验 |
| TC-037 | 015 | 同双 Linear 的 AWQ；恢复模块正确、输出 finite，记录误差 | H+O；历史 max_abs_diff=0.005859375；未设阈值，不判误差验收通过 |
| TC-038 | 015 | 独立小型 AWQ reload，用 assert_close 比较 | L；test_quant_reload.py::TestCalibrationStateEquivalence 对应 AWQ case，rtol=0.05、atol=0.05；与 TC-037 分开 |
| TC-039 | 016 | 三组回归结果和静态检查可追踪 | 三组为 L；Ruff/diff 为 H；不可将重叠集合相加 |
| TC-040 | 016 | 真实 NPU 转换、设备恢复和前向 | S；历史 13 个 skip。设备可用后按 BT-07 重跑 |

### 7.1 双 Linear 实验完整条件

网络为两个随机初始化 1024×1024 Linear，中间 SiLU，bf16；源模型 seed=123，两个 `(1,16,1024)` 校准 batch；探针 seed=321，同 shape。脚本 deepcopy 源模型用于 direct 量化，再把 artifact 恢复到源模型副本，检查两个子模块类型与输出。

MinMax 和 GPTQ 的 `torch.equal` 比较的是 direct 量化模型与 reloaded 量化模型，不能说量化后等于未量化原模型。AWQ 的最大差值只被打印，未依据算法误差预算断言；也没有逐输出量化步长推导，因此不能使用“符合一个舍入步合同”的说法。

### 7.2 为什么既需要 tiny 回归也需要真实模型

tiny 方便精确注入缺文件、错 key、错 dtype、第二层失败等情形，是协议与事务保护的主要工具。真实模型验证 Adapter、结构、dtype 和全层路径，但执行成本高且定位困难。两者互补，任一层通过都不代表另一层已验证。

## 8 历史执行命令与复现前提

下列为已归档测试集合对应的命令形式，不是本次新运行结果。历史使用过 `/tmp/layerwise_audio_shim`，该目录当前不存在；重跑前应修复或建立可记录的兼容环境，而不是照抄已失效的 PYTHONPATH。

在源码根目录、确保当前 AMCT 可导入后运行：

```bash
python -m pytest tests/unit_test/classic/test_layerwise.py \
  tests/unit_test/common/optimization/test_calibration_solver.py

python -m pytest tests/unit_test/classic \
  tests/unit_test/common/optimization

python -m pytest tests/unit_test/classic/test_quant_result.py \
  tests/unit_test/classic/test_quant_reload.py \
  tests/unit_test/classic/test_quantize_modules.py \
  tests/unit_test/classic/ptq/test_quantize.py \
  tests/unit_test/algorithms/test_quant_algorithm_base.py
```

静态检查历史命令：

```bash
ruff check amct_pytorch/classic/layerwise.py \
  amct_pytorch/common/optimization/calibration_solver.py \
  amct_pytorch/algorithms/quant/minmax.py \
  tests/unit_test/classic/test_layerwise.py \
  tests/unit_test/common/optimization/test_calibration_solver.py

git diff --check
```

真实脚本保存在 evidence，可审阅其 config、断言和历史 `/tmp` 输出路径。脚本包含重置实验目录的逻辑；新一轮应先把 ROOT 指向本轮独立归档目录，再执行并保存完整日志，避免清掉需要保留的前轮证据。原脚本作为历史证据不在此改写。

## 9 Skip 明细与未执行项

### 9.1 13 个 skip 的真实原因

| 数量 | 日志原因 | 对应能力 |
|---|---|---|
| 7 | deploy conversion requires the real torch_npu | 真实部署转换相关检查 |
| 5 | requires an NPU device | 参数回设备与 NPU 前向相关检查 |
| 1 | no Ascend NPU available | NPU PTQ round trip |

不能把这 13 项统称为 torchaudio ABI 引起，也不能因 CPU 上可导入 torch_npu 就把它们改为通过。详见留存 regression 与 algorithm/V2 日志的 short test summary。

### 9.2 当前不支持与未测要区分

AWQ/GPTQ/SmoothQuant layerwise、MoE、多 target、动态激活、KV cache 属于明确拒绝的 P0 范围外功能；逐层续抽和 per-unit PTQ resume 属于没有实现的能力；PPL、正式准确率、峰值内存、吞吐与真实 NPU 部署属于独立验收未完成。三种状态不能都模糊写成“测试未通过”。

## 10 补测与接入计划：每项都有可判定标准

以下编号 BT 是后续工作，不属于历史通过数。新执行时应保存独立日志并将状态从 N/C 更新为实际结果。

### BT-01：数据与 tokenizer 内容身份（FR-006 / DS-006）

**操作：** 在相同 dataset 路径先保存文本 A 并生成缓存，再覆盖成 token 序列明显不同的文本 B；保持全部量化配置不变。另构造 tokenizer.json 不变、相关 tokenizer 配置变化的实验。

**要求：** 先实现并冻结来源身份规则；变更影响 token 时必须改变缓存身份或明确强制重抽。相同数据拷贝的是否复用策略应有文档，不允许由路径巧合决定。正例与负例同时断言 key、Extract 调用和 token 内容。当前实现预计不能自动识别同路径数据覆盖，这是一项功能补强，不只是补写测试。

补充负例：将 cache manifest 改成合法 JSON 的 list、null、数值和字符串；类型防御应把它们明确判为无效或给出可诊断 schema 错误，不能产生未经设计的 AttributeError。该项随缓存身份与 schema 补强一并完成。

### BT-02：真实 Qwen 严格缓存复用（FR-007 / DS-007）

**操作：** 第一次运行后记录 28 个 pkl 和 cache manifest 的 size、SHA-256、mtime_ns；对 embedding/block Extract 入口记录调用。换新结果目录再运行。

**通过：** 第二轮 Extract 调用数为 0；缓存摘要和 mtime 均不变；新 artifact 84 units 完整且可 reload。再损坏一个 pkl，要求重抽发生并重新得到可读完整 cache。不能只检查文件数量。

### BT-03：gate/up/down 语义 oracle（FR-009 / DS-009）

**操作：** 给 tiny 原始 MLP 和 QuantLinear 风格包装各设非平凡固定权重与 x，独立计算 gated middle；比较三个 captured 输入。覆盖 bf16 与 float32 输入对齐、缺一个投影和 hook 抛错。

**通过：** gate/up 与 x 相等，down 与独立 middle 在 dtype 明确的容差内一致；输入缺失明确报错；失败后 hooks 已移除。容差按具体 dtype 预先写入测试，而不在失败后放宽。

### BT-04：统一 Solver 接入（FR-011 / DS-011）

**操作：** 先实现主流程真正调用 solve/finalize；spy 记录调用次数及 batch 次数，使用可导出和不可导出模块验证分支。外层启用梯度，solver 内记录 grad 状态。

**通过：** 每个需要量化的 unit 只有一次 solve/finalize 生命周期，无重复校准；no_grad 由 solver 自身保证；导出与原 Classic 路径同配置参数一致；异常不发布 manifest。仅新增调用但结果为空或重复批统计不能通过。

### BT-05：真实写中故障注入（FR-012 / FR-014）

**操作：** 让第一项 `writer.write_unit()` 成功，随后在第 2 项导出、文件替换或最终发布前分别注入异常。记录已完成写入计数和磁盘文件。

**通过：** 故障前确实有 unit 写出；故障后无完成 manifest；Reader 拒绝；fresh model 未被部分替换；同源配置使用新结果目录能完整重跑。每个注入点单独报告，不能用规划失败代替。

### BT-06：精度、AWQ 阈值与代表性数据

**操作：** 固定 FP baseline、模型权重、tokenizer、代表性校准集和独立评测集；分别运行 weight-only 与 static MinMax。报告 logits 误差、PPL/任务指标及不同文本长度；保留完整配置与结果。

**通过标准：** 功能硬标准为无 NaN/Inf、shape 正确、参数完整；质量阈值应在运行前由任务需求冻结，例如允许的 PPL 增幅、任务分数降幅，不能事后根据结果选择阈值。AWQ 双 Linear 另加入显式 rtol/atol 或根据算法推导的误差界，记录超过阈值的元素比例。尚未冻结阈值时，结果只能记观测，不记质量通过。

### BT-07：补齐 NPU 部署验收（TC-040）

**操作：** 固定 torch、torch_npu、CANN、驱动与设备型号，先验证真实 NPU 可用；逐项执行原 13 个 skip；同 seed/同 batch 比较 direct convert 与 artifact reload 后 convert。覆盖参数 device、量化模块/部署模块类型、输出 shape 与数值容差。

**通过：** 13 项全部实际执行且通过，不能仍有环境 skip；真实 Qwen 在 NPU 上完成目标链路并保存日志。若某算法或 kernel 不支持，明确标出支持范围，不能以 fake quant 输出代替整数部署结果。

### BT-08：性能及完整归档

**操作：** 固定模型、配置、数据、batch、seq_len、线程/设备、预热次数；同条件比较 full 与 layerwise 的 Extract/PTQ 阶段耗时、峰值 RSS/设备内存，并分别测部署前向吞吐/时延。冷缓存与热缓存分组，加载、hash、Extract、PTQ、reload、推理阶段分别计时。

**通过：** 至少重复多轮并报告中位数、离散程度与样本数；资源曲线与计时来自实测，不从结构图推导百分比。性能目标在实验前冻结；未设目标只能记录结果。归档脚本、源文件摘要、环境、数据身份、配置、全日志、manifest 和所有受管文件摘要，完成一次 fresh-process 恢复作为独立检查。

## 11 缺陷修复记录与本次审查发现

### 11.1 历史报告中的修复

| 问题 | 原影响 | 当前代码中的处理 |
|---|---|---|
| registry 期望算法集合缺 minmax | 注册测试不一致 | 增加 minmax 期望 |
| Qwen QuantLinear 包装 | 无法按真实 Linear 建 unit | `.linear` 解析 |
| bf16 权重与 float 输入 | forward dtype 不匹配 | 按原 Linear 权重 dtype 转换 |
| config 指纹序列化不同 | 同源 fresh 模型身份不匹配 | 统一规范 JSON 与 tensor-byte 规则 |
| QuantGatedMLP 捕获方式 | down 输入或 hook 行为错误 | 显式 gated forward 语义 |

这些属于旧报告修复记录，当前代码可定位相应处理；本次没有重新制造并运行每个历史缺陷。

### 11.2 本次必须保留的审查结论

- Solver 有组件实现，主流程没有调用 solve/finalize。
- dataset 只有路径身份，tokenizer 摘要只覆盖可选 JSON。
- mock 静态测试名字含 activation params，实际仅断言 scale_w。
- 真实第二轮文件数量不足以证明零 Extract。
- 第二层 build 失败可能发生在前置规划，不证明写中中断。
- 双 Linear AWQ 最大误差未设置阈值；旧“一个舍入步”表述撤回。
- 真实 Qwen 为 CPU 全层功能冒烟，缺完整独立日志与现存原临时产物。

因此不保留旧稿“当前无已知未修复缺陷”“全部需求均已有充分自动化证据”的绝对表述。当前功能可讲解、可追踪，同时仍有明确实现补强和验收缺口。

## 12 追踪索引与证据保存规则

| 需求/设计 | 文档用例 | 独立真实实验 / 补测 |
|---|---|---|
| FR/DS-001～003 | TC-001～013 | 入口与配置参数化 |
| FR/DS-004～007 | TC-014～022 | QW-03/05；BT-01/02 |
| FR/DS-008～011 | TC-023～029 | QW-01～04；BT-03/04 |
| FR/DS-012～014 | TC-030～035 | QW-02/04；BT-05 |
| FR/DS-015～016 | TC-036～040 | QW-01～05；BT-06～08 |

证据引用优先使用本交付包 `evidence/` 中的原始副本。原脚本、日志和旧报告保持历史内容；新解释放在本报告，不回写历史文件把弱断言改成曾经通过的强断言。新运行应另建日期目录，并记录“此次运行使用的代码”而不是只引用旧基线。

源码定位：[layerwise 测试](../../tests/unit_test/classic/test_layerwise.py)、[Solver 测试](../../tests/unit_test/common/optimization/test_calibration_solver.py)、[V2 Writer/Reader 测试](../../tests/unit_test/classic/test_quant_result.py)、[reload 测试](../../tests/unit_test/classic/test_quant_reload.py)。本轮技术事实复核见 [technical_audit.md](../qa/technical_audit.md)。
