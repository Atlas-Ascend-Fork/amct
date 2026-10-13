# 里程碑 2 讲解材料技术事实审查

审查日期：2026-09-28。审查方式：只读当前源码、既有测试代码、2026-09-23 交付包与留存日志；当日未运行真实模型、未修改实现、未提交代码。先读取了代码仓库 AGENTS.md。2026-10-13 提交前，在 Week 3-4 PR #267 最新代码基线 71a7b181 上复核聚焦单测与 pre-commit；真实模型实验未重新运行。

代码根目录：`amct-layerwise-dev`。

历史交付根目录：`docs/zh/evidence`。

## 一、必须修正的表达

| 旧版可能造成的理解 | 审查结论 | 新版可直接使用的措辞 |
|---|---|---|
| CalibrationSolver 已经串起主流程的 solve → finalize | 主流程选择并实例化 solver，但未调用 `solve()` 或 `finalize()`；当前主流程直接构造和驱动 Classic MinMaxQuant | “CalibrationSolver 已实现并有组件测试；当前 layerwise 主路径由 Classic MinMaxQuant 完成统计与参数导出，统一 solve/finalize 调度仍待接入。” |
| 数据内容变化必定使缓存失效 | cache key 中 dataset 只有规范化路径，没有 dataset 内容摘要 | “缓存绑定模型指纹、数据路径、tokenizer.json 摘要及采样配置，并检查每个缓存文件的大小和摘要。同路径数据被覆盖尚不能自动识别。” |
| tokenizer 的所有变化都可被识别 | 只 hash tokenizer.json，不覆盖所有 tokenizer 相关文件、运行版本或编码逻辑 | “tokenizer 身份当前取模型目录路径和 tokenizer.json 的可选 SHA-256。” |
| 真实 Qwen3 验证已证明模型精度、部署性能和完整生产能力 | 真实权重、全 28 层 FFN，但 CPU、1 个 16-token 样本，检查 artifact/reload/logits shape/finite；没有 PPL、准确率、峰值内存、吞吐或 NPU 实测 | “真实 Qwen3-0.6B 的全层功能冒烟：84 个 FFN Linear 单元、fresh reload 与有限输出成立。精度、性能和 NPU 部署另行验收。” |
| 真实 Qwen 第二轮缓存测试证明完全没有重新 Extract | 真实脚本只断言前后均有 28 个文件及 source_model 相同，不检查调用、mtime 或内容是否被重写 | “真实脚本完成第二轮产物生成；不重抽行为由 mock 单测的 extract_calls 断言证明。” |
| AWQ 的 0.005859375 已证明满足一个舍入步合同 | 1024×1024 双 Linear 脚本只打印差值，只断言 finite；没有阈值断言，也没有推导每个输出的量化步长 | “历史双 Linear 脚本记录 max_abs_diff=0.005859375，但未做误差阈值验收。小型 AWQ 回归另有 assert_close(rtol=0.05, atol=0.05)。” |
| 350 个测试代表每一种算法全部真实模型测试 | 350 是 5 个特定测试文件的总用例数，包含 artifact、reload、模块与注册测试，不是 350 个算法或 350 个真实模型实验 | “算法与 V2 artifact 相关回归：350 passed、13 skipped（历史日志）。” |
| 可以把 79、506、350 相加当独立覆盖量 | 三组测试集合重叠 | “并列报告三组回归结果，不求和。” |
| 所有错误都在 Adapter 构造前失败 | 配置结构、算法范围等可早失败；dataset 不存在、tokens 不足、运行期模块结构错误发生于后续阶段 | “配置与已知支持边界在 Adapter 构造前校验；数据内容和运行时结构在相应阶段校验。” |
| NPU convert 已通过或 reload 后能够可靠继续 convert | 留存日志有 13 个 skip；无真实 NPU 部署证据 | “产物沿用 V2 reload 契约；本轮 CPU 功能证据不替代 NPU convert 验收。” |

## 二、已核实的技术事实及定位

### Week 5：入口与合同

- `config.pipeline` 控制分派。未传时字符串默认 layerwise，nn.Module 默认 full；`full + str` 明确未实现。来源：[quantize.py](../../amct_pytorch/classic/quantize.py:60)，分派规则在第 78–93 行。
- 准备阶段先加载本地 config，再翻译和校验配置，最后构造 Adapter。来源：[layerwise.py](../../amct_pytorch/classic/layerwise.py:339)。
- `LayerwiseArgs` 是 Adapter 消费的内部数据合同，不是新的公共 API。来源：[LayerwiseArgs](../../amct_pytorch/classic/layerwise.py:83)。
- P0 接受无算法或 MinMax、int4/int8 权重、静态 int8 激活；拒绝 AWQ/GPTQ/SmoothQuant、动态激活和 MoE。来源：[P0 配置校验](../../amct_pytorch/classic/layerwise.py:1023)、[MoE 拒绝](../../amct_pytorch/classic/layerwise.py:1073)。
- “已注册 dense 模型”还需要兼容当前 Adapter 合同与 gate/up/down 结构。实现通过 `_planned_mlp` 和 `_linear_child` 再验证。不要泛称任意 dense 网络均支持。来源：[unit plan](../../amct_pytorch/classic/layerwise.py:437)。

### Week 6：本地数据与 Extract

- `datasets.load_from_disk` 加载本地 dataset；只允许一个 split；要求 text 字符串列；tokenizer 编码后拼接并切固定窗口，每个样本 `[1, seq_len]`。来源：[load_calibration_data](../../amct_pytorch/classic/layerwise.py:608)。
- `_needs_extract` 的条件为 `a_bits < 16 and not activation_dynamic and "minmax" in algos`。weight-only 不读取校准 dataset、不 Extract。来源：[判断](../../amct_pytorch/classic/layerwise.py:663)、[主调度](../../amct_pytorch/classic/layerwise.py:386)。
- Extract 调用 embedding forward，再逐 block forward；使用 FFN norm hook 产生每 block 的 MLP 输入缓存。来源：[Extract](../../amct_pytorch/classic/layerwise.py:667)。
- cache key 包含模型路径/指纹/model_type、dataset 路径、tokenizer 路径/tokenizer.json 可选摘要、样本数、batch size、序列长度、target、hook 与固定采样描述。没有 dataset 内容摘要，没有量化 bit 或算法。来源：[cache manifest](../../amct_pytorch/classic/layerwise.py:691)。
- 每个缓存 pkl 有文件名、大小、SHA-256。校验时重新构建预期 manifest 并比较 key 和 files，损坏、缺失或已覆盖字段不一致时重抽。来源：[完整性检查](../../amct_pytorch/classic/layerwise.py:739)。
- 不完整 cache 是全量重抽；不是断点续抽。缓存 manifest 用临时文件、flush、fsync、replace 发布。来源：[重抽清理](../../amct_pytorch/classic/layerwise.py:680)、[发布](../../amct_pytorch/classic/layerwise.py:728)。

### Week 7：实际运行链路

- 每层 gate_proj、up_proj、down_proj 三个 Linear 为持久化 unit；包装类型经 `.linear` 取出真实 `nn.Linear`。来源：[unit plan](../../amct_pytorch/classic/layerwise.py:437)、[wrapper 解析](../../amct_pytorch/classic/layerwise.py:480)。
- 静态路径读取 MLP 输入后挂 hook 捕获各投影输入。QuantGatedMLP 特殊路径显式执行 gate、up、`act_fn(gate) * up`、down，保证 down 收到中间激活。来源：[capture](../../amct_pytorch/classic/layerwise.py:820)、[gated forward](../../amct_pytorch/classic/layerwise.py:857)。
- `_select_solver_cls` 返回 CalibrationSolver。实例化时 BaseSolver 仅赋值 `self.model = model`，不会量化或调用 solve。来源：[solver 选择](../../amct_pytorch/classic/layerwise.py:756)、[BaseSolver](../../amct_pytorch/common/optimization/base_solver.py:25)。
- 主流程实际执行：解析 Linear → 构建 UnitBatch → 实例化 solver → 判定 solver.model → 构造 Classic MinMaxQuant → 遍历 batch 调用 quant_module → `export_quant_params()` → `writer.write_unit()`。代码没有 `solver.solve()` 或 `solver.finalize()` 调用。来源：[实际 PTQ](../../amct_pytorch/classic/layerwise.py:867)。
- CalibrationSolver 类的 `solve()` 确实使用 no_grad 逐 batch forward；`finalize()` 优先调用 export_quant_params，再尝试 export_ptq_params。它是存在并单测过的组件能力。来源：[CalibrationSolver](../../amct_pytorch/common/optimization/calibration_solver.py:32)、[组件测试](../../tests/unit_test/common/optimization/test_calibration_solver.py:42)。
- 注册算法 `MinMax` 和 Classic `MinMaxQuant` 不是同一个类。前者只是观测 min/max 并提供 registry 元数据；P0 真正写量化参数的是后者。来源：[MinMax registry](../../amct_pytorch/algorithms/quant/minmax.py:24)、[Classic MinMaxQuant](../../amct_pytorch/classic/quantize_op/minmax_module.py:42)。
- MinMaxQuant 在初始化计算权重 scale，静态激活则累计 batch min/max，在第 batch_num 个 batch 得出 scale_d。对称案例是 scale_w 或 scale_w+scale_d；非对称配置还可能有 offset_w/offset_d。图示若只画 scale，应标注“对称量化示例”。来源：[权重统计](../../amct_pytorch/classic/quantize_op/minmax_module.py:99)、[激活统计](../../amct_pytorch/classic/quantize_op/minmax_module.py:129)。
- reload 后 MinMaxQuant 的 forward 是 fake quant 路径；这不等价于已部署 NPU 整数算子。来源：[fake_quant_forward](../../amct_pytorch/classic/quantize_op/minmax_module.py:198)。
- 所有 expected unit 完成后才 publish manifest；结果是可 reload 的量化参数 artifact，不是完整低比特模型权重包。来源：[主流程 writer 与 publish](../../amct_pytorch/classic/layerwise.py:378)。

### Week 8：真实权重实验的证明范围

留存脚本：[test_real_layerwise_matrix.py](../evidence/test_real_layerwise_matrix.py)。

| 项目 | 脚本中的事实 | 可以证明 | 不能证明 |
|---|---|---|---|
| 模型 | 本地 Qwen3-0.6B，28 层，84 FFN unit | 使用真实模型权重与 Adapter | 任意模型通用性 |
| weight-only | CPU，int8 对称 channel，84 个 scale_w；fresh bf16 模型 fingerprint 相等；首尾包装检查；shape=(1,16,151936)；finite | 全层 artifact + fresh reload + forward 功能 | 精度保持、速度收益、NPU 支持 |
| static MinMax | CPU，文本为重复英语短句，1 样本×16 tokens；int8 对称 tensor 激活；84 组 scale_w+scale_d；28 缓存文件 | 激活参数存在、全层闭环可运行 | 有代表性的校准集精度 |
| 第二轮 PTQ | 换结果目录，再运行；前后 cache 文件数均 28；source_model 相同 | 第二轮可完成产物生成 | 单靠该脚本不能证明 Extract 从未执行 |
| fresh reload | 重新加载同一源模型并比 fingerprint，reload 后 forward shape/finite | 源身份与 reload 可执行 | 与未量化模型误差界限 |

定位：weight-only 第 32–69 行；static 第 72–128 行；第二轮检查第 110–118 行；算法拒绝第 131–152 行。

缓存不重抽的更直接证据位于 mock 单测：[test_complete_activation_cache_is_reused](../../tests/unit_test/classic/test_layerwise.py:956)，明确比较 `extract_calls` 不增加；损坏重抽在第 966–975 行。

## 三、测试数字与证据强度

### 留存 pytest 日志确实存在

| 测试组 | 历史结果 | 证据 |
|---|---|---|
| Layerwise + CalibrationSolver 定向 | 79 passed，30.66s | `evidence/layerwise_milestone2_lastfocused.log` |
| Classic + optimization | 506 passed，13 skipped，33.89s | `evidence/layerwise_milestone2_lastregression.log` |
| 算法 / V2 相关选定文件 | 350 passed，13 skipped，27.59s | `evidence/final_detailed_algorithm_matrix.log` |

这些是 2026-09-23 交付中留存的 pytest 摘要；本次没有重新运行。不同组互相重叠，不能相加为唯一测试数。350 那组在旧手册列出的命令包含 `test_quant_result.py`、`test_quant_reload.py`、`test_quantize_modules.py`、`classic/ptq/test_quantize.py`、`test_quant_algorithm_base.py`，不代表全仓或每算法全覆盖。

### 真实模型证据为脚本加旧报告摘录

- 旧交付 evidence 有两个真实测试脚本，但没有对应完整独立运行日志。
- 旧测试报告第 183–198 行保存了 PASS 输出摘录。新 PPT 应标注“历史报告记录”，避免当成本次新跑结果。
- 2026-09-28 只读核查：`/tmp/real-layerwise-algorithm-matrix`、`/tmp/real-full-algorithm-matrix` 和 `/tmp/layerwise_audio_shim/sitecustomize.py` 已不存在。
- 因此旧报告第 202–208 行的 /tmp 产物路径本次不可直接复核，不应继续称“当前可复核完整产物”。

### Classic full 三算法需独立说明

脚本：[test_real_full_algorithms2.py](../evidence/test_real_full_algorithms2.py)。

- 实验对象是随机初始化的两个 1024×1024 Linear，中间 SiLU，bf16，2 个校准 batch。不是 Qwen3 上的 AWQ/GPTQ 测试。
- MinMax/GPTQ 第 61 行以 `torch.equal(direct, reloaded)` 检验量化模型的落盘前后输出一致；不代表与 FP 原模型输出一致。
- AWQ 第 56–59 行仅算误差、断言有限并打印最大误差。旧报告记录 `0.005859375`，不能据此声称阈值断言通过。
- 独立的小型 AWQ reload 单测确有 `torch.testing.assert_close(..., rtol=0.05, atol=0.05)`。来源：[AWQ 回归](../../tests/unit_test/classic/test_quant_reload.py:746)，断言在第 764 行。可以介绍它，但必须与 1024×1024 双 Linear 实验区分。

### 当前测试报告的两处额外表述偏强

- `test_static_minmax_extracts_and_writes_activation_params` 的 mock 测试第 947–948 行实际上只断言 `scale_w`，不能用其名称或通过状态证明真实 scale_d；scale_d 的直接断言来自 Qwen 脚本第 103–105 行。
- `test_failure_leaves_no_manifest` 通过第二层 `build_quant_block` 抛异常。主流程会在写入前的 `_plan_linear_units` 遍历构建每层，因此该测试可能在规划阶段即中断，不能单独声称证明“已经写出第一层后第二层 PTQ 失败的事务行为”。可以准确称“运行失败不发布 manifest 的回归”；如果要证明 mid-write failure，还需定点故障注入。

## 四、CPU / NPU 的边界

留存 13 个 skip 的明确原因是：

- 7 个 deploy conversion 用例要求真实 torch_npu。
- 5 个 reload/部署用例要求 NPU device。
- 1 个 PTQ round-trip 用例无可用 Ascend NPU。

skip 原因来自留存日志，而不是笼统的“torchaudio ABI 不兼容”。torchaudio ABI 是旧报告记录的导入障碍，历史运行通过临时 shim 绕过；它与缺少 NPU/真实 torch_npu 的部署验收缺口应分开写。

“逐层降低设备峰值占用”属于设计动机与结构预期；没有 RSS/显存峰值曲线或基线测量时，不给百分比、不写已证明的内存收益。CPU 真实模型功能冒烟、NPU 部署、模型精度、内存与吞吐性能应是不同状态格。

## 五、建议新版结论

> Classic layerwise P0 已有本地模型入口、校准数据与缓存、逐 FFN Linear 的 MinMax 参数持久化及 fresh reload 功能链路。历史 Qwen3-0.6B 记录覆盖 28 层、84 个 unit 的 CPU 冒烟；主路径仍直接调用 Classic MinMaxQuant，统一 Solver 调度、数据内容指纹、严格真实缓存复用断言以及正式精度/NPU 性能验收仍需补齐。三组历史回归日志分别为 79 passed、506 passed/13 skipped、350 passed/13 skipped，不相加。

新版可继续完整讲解 Week 5–8 的已实现设计和关键算法原理，同时用清晰图例区分“代码事实”“历史测试证据”“尚待接入或验证”。
