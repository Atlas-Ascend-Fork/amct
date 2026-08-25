# AMCT 量化 Agent 测试集

## 测试 1：BATQuant 与 MXFP4 Group-wise Quantization

测试语：

> 在 AMCT 量化框架中开发 BATQuant 算法，接入现有 PTQ pipeline，并支持 MXFP4 group-wise quantization。请跟踪并说明 Agent 从需求解析、代码定位、方案设计到验证设计的处理数据流，检查其是否符合 AMCT 现有分层和调用链预期。本次只输出任务规划，不修改代码，不执行量化，不写入任何文件。

**核心方案**：注册为 `structure` 算法，复用 FlatQuant 协议，`--algos batquant` 触发。

| 项目 | 内容 |
|---|---|
| 注册 | `@ALGO_REGISTRY.register(name="batquant", targets=("structure",))` |
| 构造 | `__init__(self, args, ctx)` — g₁=8 / g₂=4 / block_size=32 默认值 |
| forward | `forward(x, inv_t=False, name=None)` — 按 name 分发 input_gpk / hidden_gpk |
| 训练 | `trainable_params()` → GPK + clip → `BlockwiseSolver` (AdamW, lr=2e-3, 5 epochs) |
| 导出 | weight 侧 P⁻¹ 融合进权重，activation 侧在线保留 |

**改动面**：

| 改 | 文件 |
|---|---|
| 新建 | `algorithms/quant/batquant.py` |
| 加一行导入 | `algorithms/quant/__init__.py` → `register_algorithms()` |
| 新建测试 | `tests/unit_test/algorithms/test_batquant.py` |

**分层合规性**：

- ✅ 算法层：不写模型名分支，通过 `targets` 路由
- ✅ quant module：`QuantGatedMLP` 的 `input_transform` / `hidden_transform` 已预留 structure 算法调用点
- ✅ solver：`BlockwiseSolver` 通过 `trainable_params()` 消费，不感知算法身份
- ✅ CLI：`--algos` 无硬编码 choices，registry 自动可选
- ✅ 不混用 classic `AlgorithmRegistry`

**待冻结契约**：

| # | 问题 | 影响 |
|---|---|---|
| C1 | `QuantGatedMLP.forward()` 调 transform 时是否传 `name` 参数？ | 算法区分 input/hidden |
| C2 | input_transform 和 hidden_transform 是同一实例还是两个？ | 参数共享策略 |
| C3 | `BlockwiseSolver` 是否收集 structure 算法的 `trainable_params()`？ | 训练闭环 |
| C4 | Deploy 时 weight 融合由谁触发？ | 融合路径 |

**验证矩阵**：

| 层 | 断言 |
|---|---|
| 注册 | `ALGO_REGISTRY.has("batquant")` |
| GPK 数值 | reshape → einsum → reshape 数值正确性 |
| 导出回载 | `export_ptq_params()` → `load_ptq_params()` roundtrip |
| solver 训练 | trainable_params 进入 optimizer，loss 下降 |
| 端到端 | `--algos batquant --quant_dtype mxfp --w_bits 4 --a_bits 4` PPL 达标 |

**数据流**：

```
CLI: --quant_dtype mxfp --w_bits 4 --a_bits 4 --algos batquant
  → register_algorithms() → 导入 batquant.py → 注册
  → for layer_idx: build_quant_block → QuantGatedMLP
      → input_transform / hidden_transform = BATQuant 实例
  → BlockwiseSolver.solve() → trainable_params() → AdamW
  → PtqParamHandler.export_unit() → layer_X_batquant.pt
```


token 消耗：`6w`

## 测试 2：基于 INT4 扩展 HIF4 路径

测试语：

> 基于 AMCT 现有 INT4 权重量化实现扩展 HIF4 路径，要求原有 INT4 行为、导出格式和测试结果不受影响。请跟踪并说明 Agent 从现有实现分析、兼容性边界识别、改动范围设计到回归验证设计的处理数据流，检查其是否满足向后兼容预期。本次只输出任务规划，不修改代码，不执行测试，不写入任何文件。

**核心方案**：新建 `hifloat4.py` 注册独立 dtype，复用 INT4 的 `pack_4bit`，HiFloat4 编码/量化逻辑独立实现，与 INT4 代码路径零交叉。

**改动面**：

| 改 | 文件 | 说明 |
|---|---|---|
| 新建 | `quantization/dtypes/hifloat4.py` | `QuantDequantHiF4` 注册为 `"hifloat4"` |
| 新建 | `quantization/dtypes/hif4_impl.py` | HiFloat4 编码/解码、per-channel quant、fake quant |
| 加一行导入 | `quantization/dtypes/__init__.py` → `register_dtype()` | 缺导入 → registry 不可达 |
| 新建测试 | `tests/unit_test/quantization/test_hifloat4.py` | 数值正确性、与 INT4 对比、导出 roundtrip |

**不改的文件**（INT4 隔离保证）：`int.py` / `int_impl.py` / `bit_policy.py` / 所有现有 INT4 测试。

**dtype 契约冻结状态**：

| # | 契约项 | INT4 现状 | HIF4 待冻结 |
|---|---|---|---|
| 1 | registry 名 | `"int"` | `"hifloat4"` — 不与 classic path 冲突 |
| 2 | 支持 bit 集合 | `{4, 8, 16}` | HIF4 固定 4-bit；`bits=16` 直通 |
| 3 | 适用范围 | weight + activation | ⚠️ 仅 weight-only？还是 weight+activation？ |
| 4 | 编码格式 | 整数对称 `[−7, 7]` | ⚠️ HiFloat4 精确格式全仓无定义 |
| 5 | 量化粒度 | per-channel | ⚠️ 是否沿用 per-channel？还是 per-group？ |
| 6 | deploy payload | qweight + uint64 scale + fp32 bias | ⚠️ scale 编码方式、是否需要 assistance bias |
| 7 | NPU kernel | 无需 | ⚠️ 是否需要 NPU 加速？ |

**条件分支**：

| 分支 | 前提 | 落点 |
|---|---|---|
| A: weight-only | C3 冻结为 weight-only | `hif4_impl.py` 只实现 `weight_quant()` + `export_deploy()` |
| B: weight+activation | C3 冻结为 weight+activation | 额外实现 `activation_quant()` |
| C: per-group | C5 冻结为 per-group | 新增 group_size 参数，reshape → per-group scale → pack |

**数据流**：

```
CLI: --quant_dtype hifloat4 --w_bits 4
  → DTYPE_REGISTRY.get("hifloat4")(bits=4) → QuantDequantHiF4
  → forward(x):  x → scale → HiFloat4 fake quant (Python LUT) → dequant
  → export_deploy(x):  x → HiFloat4 real quant → pack_4bit(复用) + HiF4 scale
```


token 消耗：`1.5w`

## 测试 3：BATQuant 敏感层自动识别

测试语：

> 为 BATQuant 设计敏感层自动识别逻辑，根据 activation error 或 Hessian 近似自动决定跳过量化的层。请跟踪并说明 Agent 从指标选择、数据采集、阈值或排序策略、跳层决策到效果验证设计的处理数据流，检查其是否符合现有 PTQ pipeline 的职责边界。本次只输出任务规划，不修改代码，不运行实验，不写入任何文件。

**核心设计**：两阶段分离 — 算法内度量 sensitivity score，workflow 内按 score 跳层。

| 阶段 | 职责 | 落点 |
|---|---|---|
| 度量（算法） | `forward()` / `observe_input()` 中收集 activation error / Hessian 近似 | `batquant.py` 内部 |
| 决策（workflow） | `_prepare_unit_batch()` 后按 score 判断 skip，不调 solver | `llm_ptq.py` ~5 行 |

**改动面**：

| 改 | 文件 |
|---|---|
| 新增 | `algorithms/quant/batquant.py` — `_compute_sensitivity()` |
| 修改 | `workflows/llm_ptq.py` — 跳层判断 |
| 新增 | `tests/unit_test/algorithms/test_batquant.py` — 度量正确性 |
| 可选 | `cli/llm/args.py` — `--skip_threshold` 参数 |

**度量方案**：

| 方案 | 度量方式 | 优势 | 劣势 |
|---|---|---|---|
| A: Activation MSE | `MSE(quant_out, fp32_out)` per unit | 实现简单 | 只反映单层误差 |
| B: Hessian 近似 | `diag(H) ≈ E[g²]` (Fisher/GN) | 二阶信息 | 需存梯度，内存开销 |
| C: A+B 加权 | 先 error 再少量 batch 估 Hessian | 平衡 | 需冻结加权策略 |

**跳层决策**：推荐阈值法 + `--skip_threshold`（默认 None = 不跳）。

**数据流**：

```
PTQ 主循环:
  for layer_idx:
    ① build_quant_block(layer_idx)
    ② iter_ptq_units() → units
    ③ for unit in units:
       a. _prepare_unit_batch(unit)        ← materialize_gt() 已产出 FP32 gts
       b. [NEW] _compute_sensitivity()    ← 再跑一次量化 forward 对比
       c. [NEW] if score > threshold: skip
       d. 否则 solver.solve()
```

**分层合规性**：

| 检查项 | 结论 |
|---|---|
| 算法内度量违反"不读取 dataset"？ | ✅ 不违反 — 在 `forward()` 中自然获得 |
| workflow 内跳层违反"不写模型名分支"？ | ✅ 不违反 — 按 score 过滤 |
| 跳层后参数缺失影响 eval/deploy 回载？ | ⚠️ 需闭环 — 跳过层必须有明确行为 |

**待冻结契约**：

| # | 问题 | 影响 |
|---|---|---|
| C1 | 度量用 activation MSE 还是 Hessian 近似？ | 实现复杂度 |
| C2 | 阈值全局统一还是 per-layer 自适应？ | CLI 参数设计 |
| C3 | 跳过的层 eval/deploy 时保持 FP32 还是报错？ | 回载路径 |
| C4 | 度量 pre-pass 还是边训练边收集？ | 内存开销 |
| C5 | `--skip_threshold` 是否暴露为 CLI 参数？ | CLI 改动面 |


token 消耗：`不到1w`

## 测试 4：AMCT CLI 增加 GPTQ

测试语：

> 基于 AMCT CLI 框架增加 GPTQ 量化算法，支持 INT4、INT8、MXFP4 等数据类型，并将优化后的权重落盘到文件中。请跟踪并说明 Agent处理数据流，检查其是否符合现有 CLI、workflow、algorithm 和 deploy 的分层预期。本次只输出任务规划，不修改代码，不执行量化，不生成权重文件，也不写入任何文件。

**核心方案**：注册为 `weight` 算法，`--algos gptq` 触发，实现 `quantize()` hook 完全控制权重量化路径。

| 项目 | 内容 |
|---|---|
| 注册 | `@ALGO_REGISTRY.register(name="gptq", targets=("weight",))` |
| 构造 | `__init__(self, args, w_bits)` — 累积 Hessian，block_size=128，perc_damp=0.01 |
| 统计 | `observe_input(x, weight)` — calibration 阶段累积 Hessian (`x^T @ x`) |
| 量化 | `quantize(weight, quant_obj)` — Cholesky 求逆 → block-wise 逐列优化 → fake quant |
| 导出 | `export_ptq_params()` / `load_ptq_params()` 持久化 Hessian + 优化状态 |
| deploy | `export_deploy(weight, quant_obj)` — 输出优化后 qweight + scale |

**改动面**：

| 改 | 文件 | 说明 |
|---|---|---|
| 新建 | `algorithms/quant/gptq.py` | `GPTQ` 类，复用 classic path 的 Hessian + block-wise 优化逻辑 |
| 加导入 | `algorithms/quant/__init__.py` | `register_algorithms()` 中导入 |
| 修改 | `workflows/llm_ptq.py` | `solver.solve()` 前插入 calibration pre-pass（~10 行） |
| 新建测试 | `tests/unit_test/algorithms/test_gptq.py` | Hessian 正确性、权重优化、导出回载 |

**关键设计难点**：GPTQ 是统计型 weight 算法，需要先看输入数据累积 Hessian 再做优化。当前 LLM new-path 的 `BlockwiseSolver.solve()` 直接开始训练，无独立 calibration 阶段。需在 `_run_blockwise()` 中 solver.solve 前插入 calibration pre-pass，让 `observe_input()` 被调用。

**分层合规性**：

- ✅ 算法层：不写模型名分支，通过 `targets=("weight",)` 路由
- ✅ workflow：calibration pre-pass 按算法接口（`observe_input`）分发，不按模型名
- ✅ 不混用 classic `AlgorithmRegistry`
- ✅ `quantize()` hook 与 AutoRound 模式一致（`WeightQuantizer.algo_forward()` 已支持）

**数据流**：

```
CLI: --algos gptq --quant_dtype int --w_bits 4
  ↓
llm_ptq.py._run_blockwise():
  for layer_idx:
    block = build_quant_block → WeightQuantizer 内含 GPTQ 实例
    for unit:
      ① _prepare_unit_batch(unit)           ← 加载 inps，生成 FP32 gts
      ② [NEW] _run_calibration(unit, inps)  ← GPTQ.observe_input() 累积 Hessian
      ③ solver.solve(data_loader)           ← GPTQ.quantize() 执行权重优化
      ④ solver.finalize() → layer_X_gptq.pt
```

**classic path 复用**：核心 Hessian 累积（`update_hessian`）、Cholesky 求逆（`cal_hessian_inverse`）、block-wise 优化（`get_opt_weight_and_quant_factor`）可从 `classic/quantize_op/gptq_module.py` 移植，但需适配新协议（`observe_input` / `quantize` / `export_deploy` 接口）。

**待冻结契约**：

| # | 问题 | 影响 |
|---|---|---|
| C1 | calibration 用多少 batch？固定 N 还是走 `args.batch_num`？ | Hessian 精度 vs 内存 |
| C2 | MXFP4 场景下 scale/offset 计算走 `DTYPE_REGISTRY["mxfp"]` 还是自定义？ | 与 classic path `quant_dequant_tensor` 对齐 |
| C3 | GPTQ 与 AutoRound 能否同时作用于同一 weight？ | `WeightQuantizer` 只允许一个 `quantize()` hook |
| C4 | deploy 时 weight 优化已融合进权重，`export_deploy` 是否仍需调 `quant_obj`？ | payload 格式 |

**验证矩阵**：

| 层 | 断言 |
|---|---|
| 注册 | `ALGO_REGISTRY.has("gptq")` |
| Hessian | 累积值与 classic path 一致（同输入同结果） |
| 权重优化 | 优化后 MSE 低于直接量化 |
| 导出回载 | `export_ptq_params()` → `load_ptq_params()` roundtrip |
| 端到端 | `--algos gptq --quant_dtype int --w_bits 4` PPL 达标 |


token 消耗：`2wtokens`




## 测试 5：Attention 与 MLP 联合校准

测试语：

> 为 AMCT CLI 框架增加 Attention 模块和 MLP 模块一起进行量化校准的数据通路，使用 Qwen3.5-9B 进行测试，并给出联合量化与分开量化的精度比较。请跟踪并说明 Agent 处理数据流。本次只输出任务规划，不修改代码，不运行模型或量化实验，不写入任何文件。

**现状瓶颈**：`llm_ptq.py:45` 强制 `quant_target` 单元素，`base.py:263` `iter_ptq_units()` 用排他分派（进了 attn 就 return，不会到 MLP）。每次 PTQ 只处理一种 unit。联合校准需同时改数据通路和 unit 划分。

**推荐路径**：整体 block — `PtqUnit(kind="attn+mlp", module=block)`，整个 decoder layer 作为 unit。

| 项目 | 内容 |
|---|---|
| unit 划分 | `iter_ptq_units()` 新增 `"attn+mlp"` 分支，yield `PtqUnit(kind="attn+mlp", module=block)` |
| 数据 capture | block 输入（pre-attn layernorm 之后的 hidden_states），存 `block_{i}_attn+mlp_in.pkl` |
| 数据 load | `load_ptq_inps()` 新增 `"attn+mlp"` kind 分支 |
| GT 生成 | `_prepare_unit_batch()` 中 `materialize_gt` 跑整个 block forward，输出作为 GT |
| solver | `BlockwiseSolver` 不改 — 已能处理任意 module，`_collect_trainable_param_groups` 扫描所有子模块 |
| finalize | 不改 — `export_ptq_params()` 已递归调用 attn wrapper 和 MLP wrapper 的 `export_ptq_params()` |

**改动面**：

| # | 文件 | 改动 | 层 |
|---|---|---|---|
| 1 | `llm_ptq.py:45` | 放宽 `quant_target` 校验，允许 `["attn+mlp"]` | workflow |
| 2 | `base.py:263` `iter_ptq_units()` | 新增分支：`quant_target[0] == "attn+mlp"` 时 yield `PtqUnit(kind="attn+mlp", module=block)` | model adapter |
| 3 | `llm_extract_ptq_data.py:72-88` | capture hook 新增 `"attn+mlp"` 分支，hook block 输入（hidden_states） | data |
| 4 | `ptq_io.py:40` `load_ptq_inps()` | 新增 `"attn+mlp"` 分支，加载 block 输入文件 | data |
| 5 | `llm_ptq.py:157` `_prepare_unit_batch()` | unit.kind="attn+mlp" 时 block-level forward 生成 GT | workflow |

**不需要改的**：

- `BlockwiseSolver` — 已能处理任意 module
- `finalize()` — 已递归调 `export_ptq_params()`
- `_collect_trainable_param_groups()` — 已扫描所有子模块
- Qwen3.5 adapter — `super().iter_ptq_units()` 直接走 base 分支
- model adapter/wrapper/attention/MLP — 不新增模型专属分支

**数据通路：capture 策略**

attn 输入是 `input_layernorm(x)`，MLP 输入是 `post_attention_layernorm(x + attn(x))`，MLP 依赖 attn 输出。

| | 策略 1：Block 输入（推荐） | 策略 2：合并两个 hook |
|---|---|---|
| capture 点 | pre-attn layernorm 之前的 block 输入 `hidden_states` | 同时 hook attn_norm 和 ffn_norm |
| 优点 | solver 直接跑 block forward，GT 是 block 输出，天然对齐 | 复用已有 hook 机制 |
| 缺点 | 需确认 block 输入 hook 位置（不同模型可能不一致） | MLP 依赖 attn 输出，分开 hook 时序需保证正确 |
| 推荐 | ✅ 更简单，与 solver block forward 对齐 | 依赖两个 hook 时序正确 |

**分层合规性**：

- ✅ 不在 CLI/workflow/algorithm 中加模型专属分支
- ✅ unit 划分通过 `quant_target` 参数路由，不按模型名
- ✅ solver/optimizer 不感知 unit 类型
- ✅ 不混用 classic `AlgorithmRegistry`
- ✅ `PtqUnit.module` 本身就是 `Any`，不改类型契约

**数据流**：

```
CLI: --quant_target attn+mlp --algos lwc --w_bits 8
  ↓
llm_ptq.py._run_blockwise():
  for layer_idx:
    block = build_quant_block(layer_idx)
    units = list(iter_ptq_units(layer_idx, block))
      → yield PtqUnit(kind="attn+mlp", module=block)  ← 整个 decoder layer
    for unit:
      ① _prepare_unit_batch(unit)
          → load_unit_inputs(unit)  ← 加载 block 输入 hidden_states
          → materialize_gt(block)  ← block forward 生成 block 输出作为 GT
      ② solver = build_solver(layer_idx, unit.module)  ← module = block
      ③ solver.solve(data_loader)  ← block forward + MSE loss
      ④ solver.finalize()
          → block.export_ptq_params()  ← 递归发现 attn wrapper + MLP wrapper
          → layer_X_attn+mlp.ptq_param
```

**精度对比实验设计**：

| 实验 | quant_target | PTQ unit | 说明 |
|---|---|---|---|
| 分开量化-attention | `["attn-linear"]` | self_attn | 只量化 attn |
| 分开量化-mlp | `["mlp"]` | mlp | 只量化 MLP |
| 联合量化 | `["attn+mlp"]` | block | 同时量化 attn+MLP |

- 模型：Qwen3.5-9B，共用参数 `--algos lwc`，`--w_bits 8`，`--a_bits 8`，`--batch_num 5`
- 评测：lm-eval-harness 输出 ppl
- 对比：联合 vs 分开的 ppl delta

**预期**：联合量化精度可能略优（solver 同时优化能找到更好的联合最优解），但显存占用更大（block 级 forward+backward 同时持有 attn+MLP 权重+梯度）。

**待冻结契约**：

| # | 问题 | 影响 |
|---|---|---|
| C1 | block 输入 hook 位置在哪？不同模型（Qwen3.5 vs DeepSeek）是否一致？ | capture 实现 |
| C2 | 联合 unit 的 `export_ptq_params` / `load_ptq_params` 是否需要新文件命名？ | `layer_X_attn+mlp.ptq_param` vs `layer_X_block.ptq_param` |
| C3 | GT 生成方式：跑整个 block forward 还是分别跑 attn+MLP 再拼接？ | GT 准确性 vs 实现复杂度 |
| C4 | 联合 unit 是否需要新的 `quant_apply` hook，还是复用已有 attn+MLP 独立 hook？ | 模型适配 |
| C5 | 显存够不够？block 级 unit 的 forward+backward 同时持有 attn+MLP 权重 | 实际可行性 |

**验证矩阵**：

| 层 | 断言 |
|---|---|
| 注册 | `quant_target=["attn+mlp"]` 通过校验 |
| unit 划分 | `iter_ptq_units()` yield 一个 kind="attn+mlp" 的 unit，module 是整个 block |
| 数据 | capture hook 捕获 block 输入，`load_ptq_inps` 正确加载 |
| solver | `BlockwiseSolver` 收到 block module，`_collect_trainable_param_groups` 发现 attn+MLP 的 trainable params |
| finalize | `export_ptq_params()` 递归调用 attn wrapper 和 MLP wrapper 的 `export_ptq_params()` |
| 端到端 | Qwen3.5-9B 联合量化 ppl 达标 |
| 对比 | 输出联合 vs 分开的 ppl delta 对比表 |

**实施步骤**：

1. 读取 block 输入 hook 位置，确认 Qwen3.5 decoder layer 的 block 输入来源
2. 修改 `iter_ptq_units`：新增 `"attn+mlp"` 分支
3. 修改 capture：新增 block 输入 hook，存合并输入文件
4. 修改 load：新增 `"attn+mlp"` 分支
5. 修改 `_prepare_unit_batch`：适配 block-level GT 生成
6. 写测试：unit 划分、数据通路、端到端
7. 实验：Qwen3.5-9B 分开量化 vs 联合量化，输出 ppl 对比表

token 消耗：`2w`

## 测试 6：Kimi-K3 PTQ 量化流程

测试语：

> 实现 Kimi-K3 的 PTQ 量化流程，支持 FP16 baseline、FlatQuant 加 INT4 权重量化，并输出精度对比结果。请跟踪并说明 Agent 处理数据流，检查其是否符合 AMCT 端到端量化流程及人工确认门预期。本次只输出任务规划，不修改代码，不运行模型或量化实验，不写入任何文件。

使用了amct-coder，**核心方案**：从零适配 Kimi-K3 模型 adapter，复用已有 FlatQuant 结构算法 + INT4 bit_config，不改 workflow/solver/algorithm。

| 项目 | 内容 |
|---|---|
| adapter 注册 | `@MODEL_REGISTRY.register(name="kimi_k3", task="llm", family="moonshot")` |
| adapter 继承 | `class KimiK3(BaseModel)` |
| FlatQuant 集成 | 自动 — `--algos flatquant` 触发 quant wrapper 的 `_init_structure_transforms()` |
| INT4 配置 | `--bit_config w4a4.yaml` 或 `--w_bits 4 --a_bits 4` |
| FP16 baseline | `llm_eval --eval_mode bf16`（纯 HuggingFace forward，不重建 quant block） |

**改动面**：

| # | 文件 | 说明 |
|---|---|---|
| 1 | `models/llm/moonshot/kimi_k3/kimi_k3.py` | 新建 adapter：`__init__` / `get_layer_weight_prefix` / `build_quant_block` / `iter_ptq_units` / `_embed_base_prefix` |
| 2 | `models/llm/moonshot/kimi_k3/quant_module.py` | 新建 `QuantKimiK3Attn`（继承 `QuantAttentionBase`）+ `QuantKimiK3MLP`（继承 `QuantGatedMLP`） |
| 3 | `models/llm/__init__.py` | `register_llm_models()` 中加一行导入 |

**不需要改的**：workflow、solver、algorithm、dtype、CLI args、deploy。FlatQuant 通过 quant wrapper 的 `_init_structure_transforms()` 自动接入。

**分层合规性**：

- ✅ adapter 收在 `models/llm/moonshot/` 下，不改 workflow/solver/algorithm
- ✅ FlatQuant 通过 `ALGO_REGISTRY` 自动路由，不按模型名分派
- ✅ 不混用 classic `AlgorithmRegistry`
- ✅ `iter_ptq_units` 复用 base 分支（attn OR mlp 排他分派）
- ✅ INT4 通过 `bit_config` / `bit_policy` 路由，不按模型名分派

**数据流**：

```
CLI: --model /path/to/Kimi-K3 --model_name kimi_k3 --quant_target attn-linear
     --algos flatquant --bit_config w4a4.yaml
  ↓
workflow:
  setup() → register_llm_models() → MODEL_REGISTRY.get("kimi_k3") → KimiK3
  _run_blockwise():
    for layer_idx:
      block = build_quant_block(layer_idx)
        → QuantKimiK3Attn._init_structure_transforms()  ← FlatQuant 自动创建
        → QuantKimiK3MLP._init_structure_transforms()   ← FlatQuant 自动创建
      unit = iter_ptq_units(layer_idx, block)             ← 复用 base 分支
      solver.solve()                                      ← FlatQuant trainable_params + QuantLinear fake quant
      finalize() → flatquant params + quant weights 落盘
```

**FP16 baseline vs PTQ 对比流程**：

| 步骤 | 命令 | 产出 |
|---|---|---|
| FP16 baseline | `llm_eval --model /path/to/Kimi-K3 --eval_mode bf16` | ppl_bf16 |
| extract_ptq_data | `llm_extract_ptq_data --model_name kimi_k3 --quant_target attn-linear` | 校准数据 |
| PTQ (FlatQuant+INT4) | `llm_ptq --model_name kimi_k3 --quant_target attn-linear --algos flatquant --bit_config w4a4.yaml` | ptq_param 文件 |
| PTQ eval | `llm_eval --model_name kimi_k3 --quant_target attn-linear --eval_mode quant --algos flatquant --attn_param_dir ...` | ppl_quant |
| 对比 | delta = ppl_quant - ppl_bf16 | delta ≤ 0.2 为达标 |

**待冻结契约（BLOCKER）**：

| # | 问题 | 影响 | 能否自查 |
|---|---|---|---|
| C1 | Kimi-K3 的 decoder layer 类名？`KimiK3DecoderLayer` / `MoonshotK3DecoderLayer` / 其他？ | adapter 的 `self.cls` | ❌ 需用户确认 |
| C2 | checkpoint 嵌套路径？顶层 key 模式？ | `get_layer_weight_prefix()` | ❌ 需拉 index.json |
| C3 | Attention 类型？full / linear / MLA / 混合？ | `QuantKimiK3Attn` attention interface | ❌ 需确认 |
| C4 | MLP 结构？标准 gated MLP（gate+up+down+SiLU）？ | 能否直接继承 `QuantGatedMLP` | ❌ 需确认 |
| C5 | lm_head 是否 tied embed_tokens？ | adapter 是否需单独处理 lm_head | ⚠️ 可自查 |
| C6 | 模型权重文件位置？ | 测试时 `--model` 参数 | ❌ 需用户确认 |

**与 test 1（BATQuant）对比**：

| | test 1（BATQuant） | test 6（Kimi-K3） |
|---|---|---|
| 核心难点 | 算法实现（GPK 分解） | 模型适配（结构未知） |
| 改动层 | algorithm 层 | model adapter 层 |
| FlatQuant 角色 | 无 | 核心（`--algos flatquant` 激活） |
| BLOCKER | 算法参数冻结 | **模型结构确认**（C1-C4） |

**验证矩阵**：

| 层 | 断言 |
|---|---|
| 注册 | `MODEL_REGISTRY.has("kimi_k3")` |
| adapter | `get_layer_weight_prefix()` 正确加载 layer 权重 |
| quant block | `build_quant_block()` 前向不报错 |
| FlatQuant | `--algos flatquant` 时 `_init_structure_transforms()` 创建实例 |
| INT4 | `--bit_config w4a4.yaml` 时 weight bits=4 |
| BF16 baseline | `--eval_mode bf16` ppl 在合理量级 |
| PTQ smoke | `extract_ptq_data` → `ptq` → 落盘 ptq_param |
| 端到端 | PTQ eval ppl 达标（delta ≤ 0.2） |
| 对比 | 输出 FP16 vs FlatQuant+INT4 ppl 对比表 |


token 消耗：`1wtokens`

