# amct-coder 参考文档架构说明

> 本文件是**维护者文档**，不参与编码任务的上下文。执行编码任务时不要读；改 amct-coder 结构时才读。
> 它原先放在 `skills/amct-coder/references/README.md`，因为对编码动作零价值且体量最大，已移出 references 目录。

## 零、内容边界（改文档前先看这条）

| 位置 | 存什么 | 不存什么 |
| --- | --- | --- |
| `docs/repo-map.md` | 仓是什么样：分层职责表、主路径、适配原则、wrapper 合并原则 | 编码动作、验证命令 |
| `docs/casebook/` | 已踩的坑、模型个案、精度结论 | 通用原则 |
| `skills/amct-coder/references/` | 编码时做什么 / 不能做什么：执行步骤、必查清单、BLOCKER、自验命令 | 原则性表述、模型个案 |

references 需要引用原则或个案时，写一行指针指向 repo-map / casebook，不复制正文。同一条规则只允许有一个正文位置。

## 一、架构决策：单 Skill + 按需加载参考文档

**当前选择：一个主 SKILL.md 文件做路由和全局规则，具体编码规则拆分到 `references/` 目录下的 14 个参考文档中，使用时按场景按需读取。**

### 为什么选择单 Skill 而不是拆成多个 Skill

| 方案 | 优点 | 缺点 |
| --- | --- | --- |
| **多 Skill**（每个子能力一个 SKILL.md） | 每个 skill 职责清晰，可独立触发 | 全局 BLOCKER 规则散落在多个入口；跨子能力协调需要额外委派逻辑；OpenCode skill 数量膨胀 |
| **单 Skill + 按需读参考文档** | 全局 BLOCKER 和路由集中维护；跨场景只需在主文件写清读取顺序；参考文档可独立演进不改主文件 | 主文件需要承载路由表；单个参考文档可能过长 |

**skill 的 description 全量进上下文，body 才按需加载。** 拆成 多个coding skill 的真实代价不是文件数，而是 多个 description 同时参与匹配；而 AMCT 编码任务天然跨子能力（"给 qwen3.6 加 attn-cache 量化" 同时命中 llm-model-unit + dtype-quant-module + deploy-export-code），需要一个先分类再排序的中心，客户端的 description 匹配做不了这件事。子能力能被用户单独说出来时拆 skill 才划算，这里做不到。

另外当前 9 个子能力是按**契约边界**切的（dtype 定义数值语义、adapter 定义结构、solver 定义优化），不是按目录切的。这让"读上游确认契约、读下游确认闭环"这个动作有意义；按目录切会退化成索引。

**结论**：当前 9 个子能力 + 5 个辅助参考文档共 14 个文件（主 SKILL.md ~120 行，单个参考文档最大 ~100 行）。保持当前架构。

### 读取预算

按需加载只在读得少时才有收益。SKILL.md §读取预算 规定默认软预算为 3 个 reference（主 playbook + 最多 1 个辅助规则 + validation），上下游 playbook 改为条件补读且只读「上下游依赖」「BLOCKER」两节。这个数字不是硬上限：任务确实命中 3 个以上子能力时允许分阶段补读，但不能一开始全量加载所有 reference。改 SKILL.md 路由时要保住这条，否则退化成"全量读 1250 行"，等于没拆。

### 触发重构的可观测信号

不用"子能力数量"这类不可观测指标。命中下面任一条时再动架构：

- 路由表超过 12 行。
- 单个任务平均读取超过 4 个 reference。
- 某个 reference 超过 150 行。

演进方向：

1. 将高频共用的规则（validation、layer-rules）提升为独立 skill（如 `amct-validation`）。
2. 将 LLM new-path 的 6 个子能力打包为一个 `amct-llm-path` mega-skill，内部再做按需加载。
3. classic / ops 保持独立 skill 不变。

---

## 二、文件清单与职责

### 主入口

| 文件 | 行数 | 职责 |
| --- | --- | --- |
| `SKILL.md` | 122 | 场景分类、读取预算、子能力路由表、全局依赖链、全局边界、全局 BLOCKER、验证要求 |

### 子能力参考文档（按依赖链顺序）

| 序号 | 文件名 | 对应子能力 | 行数 | 覆盖范围 |
| --- | --- | --- | --- | --- |
| 1 | `repo-scenarios.md` | 全局（条件读） | 102 | 三条代码体系总览、依赖拓扑、入口文件清单。场景落点已收敛到 SKILL.md，不在此重复 |
| 2 | `dtype-quant-module.md` | `dtype-quant-module` | 70 | `quantization/dtypes/`、`quantization/modules/`、bit policy、deploy payload |
| 3 | `llm-algorithm.md` | `llm-algorithm` | 75 | `algorithms/quant/`、`ALGO_REGISTRY`、`targets`、可训练参数/统计导出 |
| 4 | `llm-model-unit.md` | `llm-model-unit` | 76 | `MODEL_REGISTRY`、adapter、PtqUnit、权重 key、mask/position、tied lm_head、适配清单 7 问（唯一正文位置） |
| 5 | `llm-data-workflow.md` | `llm-data-workflow` | 86 | eval / extract_ptq_data / PTQ workflow、离线输入、GT、PPL、dataloader |
| 6 | `ptq-solver-params.md` | `ptq-solver-params` | 69 | solver、optimizer、scheduler、PTQ 参数保存/加载/导出 |
| 7 | `deploy-export-code.md` | `deploy-export-code` | 76 | deploy workflow、safetensors、weight index、`quantization_config` |
| 8 | `classic-graph.md` | `classic-graph` | 76 | classic eager、graph_based、QAT、校准、剪枝、蒸馏、tensor decomposition |
| 9 | `ops-boundary.md` | `ops-boundary` | 75 | `amct_pytorch` 与 `amct_ops` 的调用边界、构建、Meta backend |
| 10 | `public-api-tests.md` | `public-api-tests` | 88 | `__all__`、懒加载、setup/build、测试选择、docs/examples 同步 |

### 辅助参考文档（按需读取，不是独立子能力）

| 文件名 | 行数 | 职责 | 何时读取 |
| --- | --- | --- | --- |
| `validation.md` | 70 | 验证规则、场景测试索引、adapter/wrapper 完成标准（唯一正文位置）、PTQ smoke 信号 | 每次编码结束前 |
| `layer-rules.md` | 82 | 默认落点判断、BLOCKER 检查、算法注册检查、repo-map 写回规则。分层职责表指向 repo-map | 涉及跨层改动时 |
| `wrapper-rules.md` | 77 | 复用顺序、MoE 检查、shared experts 策略、PTQ unit 命名、禁止模式。合并原则指向 repo-map | 涉及 wrapper 新增/修改时 |
| `attention-rules.md` | 58 | attention 编码动作、必查清单、禁止模式。适配原则指向 repo-map | 涉及 attention 改动时 |
| `delegation.md` | 48 | 委派表、与 model-adapter 关系、不委派场景 | 需要判断是否交给其它 skill 时 |

---

## 三、覆盖范围矩阵

### 已覆盖的编码能力

代码仓中所有涉及编码的目录与对应的参考文档映射关系如下：

```
amct_pytorch/
├── quantization/
│   ├── dtypes/          → dtype-quant-module.md    量化数据类型（INT、MXFP 的 fake quant / dequant / export_deploy）
│   ├── modules/         → dtype-quant-module.md    通用量化模块（ActivationQuantizer、WeightQuantizer、QuantLinear、QuantizedMatmul）
│   └── bit_policy.py    → dtype-quant-module.md    位宽策略（根据 dtype 和 bits 决定量化行为）
├── algorithms/
│   ├── quant/           → llm-algorithm.md         LLM PTQ 算法（LWC、LAC、AutoRound、OmniQuant、FlatQuant）
│   ├── registry_factory → llm-algorithm.md         ALGO_REGISTRY 注册机制
│   └── register_algo    → classic-graph.md         classic AlgorithmRegistry 注册机制（GPTQ、AWQ、SmoothQuant 等 8 种）
├── common/
│   ├── models/llm/      → llm-model-unit.md + wrapper-rules.md + attention-rules.md
│   │                                      模型适配器（13 个已注册模型）、wrapper、PtqUnit、attention、MoE
│   ├── datasets/        → llm-data-workflow.md     离线输入保存/加载、GT 生成、dataloader
│   ├── evaluate/        → llm-data-workflow.md     PPL 评测（Wikitext）
│   ├── optimization/    → ptq-solver-params.md     solver（BlockwiseSolver）、optimizer、scheduler、PTQ 参数管理
│   ├── config/          → classic-graph.md         classic 量化配置解析（parse_config、InsertQuantizeModulePass 等）
│   └── utils/           → 散布在多个参考文档中     注册工厂、日志、安全加载、参数检查等通用工具
├── workflows/           → llm-data-workflow.md     4 个工作流入口（eval、extract_ptq_data、ptq、deploy）
├── cli/                 → public-api-tests.md      CLI 参数解析（args.py）和模块入口
├── classic/             → classic-graph.md         classic 量化全流程（quantize → calibrate → convert → deploy）
│   ├── quantize_op/     → classic-graph.md         10 个 classic 量化算子（GPTQuant、SmoothQuant 等）
│   ├── deploy_op/       → classic-graph.md         8 个 NPU 部署算子（NpuQuantizationLinear 等）
│   ├── optimizer/       → classic-graph.md         4 个 optimizer pass（InsertQuantize、ReplaceNpu 等）
│   └── graph_based/     → classic-graph.md         graph_based 图压缩子系统（~48 个 pass、custom_op、proto、configuration）
├── deploy.py            → deploy-export-code.md    顶层 deploy 入口
├── eval.py              → llm-data-workflow.md     顶层 eval 入口
├── ptq.py               → llm-data-workflow.md     顶层 ptq 入口
├── extract_ptq_data.py  → llm-data-workflow.md     顶层 extract 入口
├── __init__.py          → public-api-tests.md      顶层公共 API 导出（__all__、懒加载）
└── nn/                  → classic-graph.md         QAT 模块（Conv2dQAT、LinearQAT 等，从 graph_based 再导出）

amct_ops/                → ops-boundary.md          NPU 自定义算子调用边界，具体实现委派 amct-ops-dev

tests/                   → public-api-tests.md + validation.md   测试结构和验证规则
setup.py / build.sh      → public-api-tests.md      打包构建逻辑
```

---

### 未覆盖区域及原因说明

以下是代码仓中存在但当前参考文档尚未专门覆盖的区域。每个区域都附有**为什么暂不覆盖**和**何时需要补充**的说明。

#### 1. experimental 试验特性

**位置**：`amct_pytorch/experimental/`

**包含内容**：
- `flatquant/` — FlatQuant 量化模块的独立实现（FlatQuantAttention、FlatQuantMLP），与 LLM new-path 的 `algorithms/quant/flatquant.py` 配合使用，但有自己独立的 quant_module 和 NPU 部署模块
- `hifloat8/` — HiFloat8 C++ 扩展（float_to_hifloat8、hifloat8_to_float32），通过 setup.py 独立构建
- `quantization/DeepSeekV3.2/` — DeepSeekV3.2 的独立量化路径（有自己的 calibrator、quantizer、modeling），不走 LLM new-path
- `quantization/MiniMax-M2.5/` — MiniMax-M2.5 的独立量化路径（mxfp4_quantizer、stage1/stage2 脚本）

**为什么不单独写参考文档**：
- FlatQuant 的算法注册已经在 `llm-algorithm.md` 中覆盖（`ALGO_REGISTRY` 注册了 `flatquant`），其 quant_module 细节属于模型适配范畴，由 `llm-model-unit.md` 覆盖
- HiFloat8 是纯 C++ 扩展，属于 `amct-ops-dev` skill 的范畴，`ops-boundary.md` 已经覆盖了调用边界
- DeepSeekV3.2 和 MiniMax-M2.5 的独立量化路径当前由 `quant-workflow` skill 端到端处理，不需要在编码 skill 中单独写参考文档

**何时需要补充**：当 experimental 中的某个特性要接入 LLM new-path 主链（例如 FlatQuant 从 experimental 迁移到 `algorithms/quant/`）时，需要在对应子能力参考文档中补充说明。

#### 2. 预定义配置文件（configs/）

**位置**：`amct_pytorch/configs/`（w4a8.yaml、w8a8.yaml、w4a4.yaml、bf16.yaml、example_w4a8.yaml）

**包含内容**：用户通过 CLI `--config` 加载的预定义量化配置，定义了 weight/activation 的 bit width 和分组策略。

**为什么不单独写参考文档**：
- 这些 YAML 文件是 CLI 参数的预设值，结构简单（几个字段），不需要专门的编码规则
- CLI 参数解析在 `public-api-tests.md` 中已经覆盖（CLI 改动要同步 choices、help、解析测试）
- 新增预定义配置只需要加 YAML 文件 + 更新 CLI choices，不涉及复杂编码逻辑

**何时需要补充**：当预定义配置的结构变复杂（例如引入 per-layer override、conditional quant 策略）时，需要在 `public-api-tests.md` 或 `dtype-quant-module.md` 中补充说明。

#### 3. auto_calibration 和 auto_channel_prune

**位置**：`amct_pytorch/common/auto_calibration/`、`amct_pytorch/common/auto_channel_prune/`

**包含内容**：
- `auto_calibration/` — 从 `classic.graph_based` 再导出的自动校准基类（AccuracyBasedAutoCalibrationBase、BinarySearchStrategy 等）
- `auto_channel_prune/` — 自动通道剪枝的敏感度分析和搜索基类（SensitivityBase、SearchChannelBase）

**为什么不单独写参考文档**：
- 这两个模块是 classic graph_based 的共享基础设施，从 `classic.graph_based` 再导出
- 它们在 `classic-graph.md` 中已经有入口级覆盖（"auto calibration"、"auto channel prune"）
- 这些模块的使用场景（graph_based 图压缩）开发频率低，不需要独立的编码规则

**何时需要补充**：当 LLM new-path 需要用到 auto_calibration 或 auto_channel_prune 时（目前没有），需要在对应子能力参考文档中补充说明。

#### 4. graph_based 子系统的内部细节

**位置**：`amct_pytorch/classic/graph_based/amct_pytorch/` 下约 48 个 optimizer pass、custom_op（~15 个）、proto（6 个）、configuration（~10 个）

**包含内容**：graph_based 图压缩的完整内部实现，包括 ONNX 图解析、各种 pass（insert/replace/delete/fusion）、自定义算子（fake_quant、retrain、distill 等）、protobuf 配置。

**为什么不单独写参考文档**：
- `classic-graph.md` 已经覆盖了入口级的开发规则（quantize/convert 流程、AlgorithmRegistry 注册、public API）
- graph_based 内部的 pass 级开发规则非常庞大（48 个 pass），且当前开发频率很低
- 这些内部细节属于 classic graph_based 专家领域，一般编码任务不需要了解全部 pass 的实现

**何时需要补充**：当有具体的 graph_based 开发需求（例如新增一个 optimizer pass、新增 custom_op）时，再按需新增 `references/graph-based-internals.md`。

#### 5. QAT 模块（nn/module/quantization/）

**位置**：`amct_pytorch/nn/module/quantization/`（Conv2dQAT、Conv3dQAT、ConvTranspose2dQAT、LinearQAT、QuantCalibrationOp）

**包含内容**：QAT（Quantization-Aware Training）的 PyTorch 模块封装，从 `classic.graph_based` 再导出。

**为什么不单独写参考文档**：
- 这些模块是从 graph_based 再导出的，属于 classic 体系的一部分
- `classic-graph.md` 已经覆盖了 QAT 的入口级规则
- QAT 模块的编码规则与 classic quantize_op 类似，不需要独立参考文档

#### 6. tensor_decompose（张量分解）

**位置**：`amct_pytorch/tensor_decompose/`（再导出 `classic.graph_based.amct_pytorch.tensor_decompose` 的 `auto_decomposition`、`decompose_network`）

**为什么不单独写参考文档**：
- 这是从 graph_based 再导出的顶层 API
- `classic-graph.md` 已经覆盖了 tensor decomposition 的入口

#### 7. CMake 构建系统

**位置**：`cmake/`、顶层 `CMakeLists.txt`、`amct_pytorch/CMakeLists.txt`

**包含内容**：CMake 构建配置，用于编译 C++ 扩展和 amct_ops。

**为什么不单独写参考文档**：
- `public-api-tests.md` 已经覆盖了 `bash build.sh` 的构建选项和打包逻辑
- CMake 细节仅在涉及 C++ 扩展开发时需要，而 C++ 扩展开发属于 `amct-ops-dev` skill 的范畴
- amct_pytorch 本身的 CMakeLists.txt 很简单（主要是 copy 文件），不需要专门的编码规则

---

## 四、子能力依赖链

依赖链的权威表述在 `SKILL.md` §依赖关系（文字形式，逐条说明谁消费谁）。这里不再画 ASCII 图：原图的箭头方向语义不自洽，且把 `public-api-tests` 只挂在 deploy 下游，而它实际是所有子能力的共同下游。

跨子能力读取顺序见 `SKILL.md` §依赖关系末段，受 §读取预算 的软约束（上下游为条件补读，不是必读；命中 3 个以上子能力时分阶段补读）。

---

## 五、辅助参考文档的层级关系

```
SKILL.md（全局路由 + BLOCKER）
  ├── layer-rules.md（分层边界，跨所有子能力）
  ├── wrapper-rules.md（wrapper 合并，属于 llm-model-unit 的子规则）
  ├── attention-rules.md（attention 适配，属于 llm-model-unit 的子规则）
  ├── validation.md（验证，跨所有子能力）
  └── delegation.md（委派，跨所有子能力）
```

`wrapper-rules.md` 和 `attention-rules.md` 本质上是 `llm-model-unit` 的细化规则。它们被提升为独立文件的原因是：
- attention 和 MoE wrapper 是高频改动点，独立文件便于快速定位
- 行为独立，不依赖 llm-model-unit 的全部内容

`layer-rules.md` 和 `repo-scenarios.md` 的分工：
- `repo-scenarios.md`：三条代码体系的拓扑和入口文件清单（告诉 agent 代码在哪）
- `layer-rules.md`：落点判断和 BLOCKER 检查（告诉 agent 什么不能做）

以下规则各自只有一个正文位置，其余位置只允许写指针：

| 规则 | 唯一正文位置 |
| --- | --- |
| 分层职责表 | `docs/repo-map.md` §分层边界 |
| Attention 适配原则 | `docs/repo-map.md` §Attention 适配原则 |
| Wrapper 合并原则 | `docs/repo-map.md` §Wrapper 合并原则 |
| 适配前 7 问 | `references/llm-model-unit.md` §适配清单 |
| adapter/wrapper 完成标准 | `references/validation.md` |
| 场景默认落点 | `SKILL.md` §全局边界 + §子能力路由 |

---

## 六、更新维护指南

### 何时更新参考文档

| 事件 | 更新目标 |
| --- | --- |
| 新增模型适配器接入主注册链 | `repo-scenarios.md`（入口文件列表） |
| 新增量化 dtype 或 bit config | `dtype-quant-module.md` + `public-api-tests.md` |
| 新增 LLM PTQ 算法 | `llm-algorithm.md` |
| 新增 solver 或改变优化范式 | `ptq-solver-params.md` |
| deploy 路径变更 | `deploy-export-code.md` |
| classic 注册体系变更 | `classic-graph.md` |
| 新增 NPU 算子 | `ops-boundary.md` |
| 新增 public API / CLI 参数 | `public-api-tests.md` |
| 框架边界变化（见 layer-rules.md 写回规则） | `repo-scenarios.md` + `layer-rules.md` |
| experimental 特性接入主链 | 相关子能力参考文档 + `public-api-tests.md` |

### 何时不更新

- 新增模型目录但未接入主注册链
- 新增算法文件但不改变分流机制
- 修局部 wrapper bug
- 改不影响复用边界的模型专属实现细节

---

## 七、与其它 Skill 的边界

| Skill | 与 amct-coder 的关系 |
| --- | --- |
| `amct-ops-dev` | amct-coder 识别到需要写 NPU kernel / C++ extension / Meta kernel 时委派 |
| `quant-workflow` | 端到端量化编排，amct-coder 只负责编码实现 |
| `quant-run` | 跑量化实验命令，amct-coder 不执行实验 |
| `model-adapter` | 纯模型适配时委派；涉及 dtype/algorithm/workflow/deploy 时 amct-coder 自行处理 |
| `direct-quant-eval` / `algorithm-validation` | 只读结果判读，amct-coder 不做精度判断 |
| `scheme-recommendation` / `algorithm-recommendation` | 方案推荐，amct-coder 不下方案结论 |
| `deploy-export` | 交付导出，amct-coder 不执行交付 |
| `gitcode-pr` / `gitcode-issue` | 仓库协作，必须使用对应 skill |

---

## 八、代码仓注册体系速查

当前代码仓共有 5 个注册表，分属两条体系：

### LLM new-path 注册表（4 个）

| 注册表 | 位置 | 已注册条目数 | 说明 |
| --- | --- | --- | --- |
| `DTYPE_REGISTRY` | `quantization/dtypes/__init__.py` | 2（int、mxfp） | 量化数据类型，通过 `register_dtype()` 懒加载注册 |
| `ALGO_REGISTRY` | `algorithms/registry_factory.py` | 5（lwc、lac、autoround、omniquant、flatquant） | LLM PTQ 算法，通过装饰器 `@ALGO_REGISTRY.register()` 注册 |
| `MODEL_REGISTRY` | `common/models/__init__.py` | 13 个模型（deepseek_v3_2、deepseek_v4、glm5、glm5_2、hy_v3、longcat_lite、longcat_next、qwen3、qwen3_moe、qwen3_5、qwen3_5_moe、qwen3_6_moe、qwen3_next） | LLM 模型适配器，通过 `register_llm_models()` 懒加载注册 |
| `SOLVER_REGISTRY` | `common/optimization/__init__.py` | 1（block） | PTQ solver，通过 `register_solvers()` 懒加载注册 |

### classic 注册表（1 个）

| 注册表 | 位置 | 已注册条目数 | 说明 |
| --- | --- | --- | --- |
| `AlgorithmRegistry` | `algorithms/__init__.py` | 8 种算法 / 12 条注册 | classic 量化算法，模块级直接注册，通过 `register(name, src_op, quant_op, deploy_op)` 四元组映射 |

**关键边界**：LLM new-path 的 4 个 `Registry("...")` 实例和 classic 的 `AlgorithmRegistry` 是两套独立体系，不能混用。这是全局 BLOCKER 规则。

---

## 九、description 写法约束（触发失败复盘）

2026-07-29 在 opencode 上用「在 workflow 里面加一段 attention 的计算」测试，没有触发 amct-coder。排查结论如下，改 description 前必须先看这节。

### 客户端行为（决定了改哪里）

- `opencode debug skill` 可列出所有被发现的 skill 及 location，是判断"发现问题"还是"匹配问题"的唯一手段。本次 amct-coder 正常出现在列表里，`.agents/` → `.opencode/skills` 的 symlink 投影正常。
- opencode 只把 `name + description` 拼进 system prompt（binary 内 `SystemPrompt.skills`，格式 `- **name**: description`），body 要等 skill 工具调用才加载。所以**触发失败一律是 description 的问题**，改 SKILL.md 正文没用。
- opencode 的项目级扫描 glob 是 `skills/**/SKILL.md`，支持任意深度，`quant-tools/<name>/SKILL.md` 这种嵌套没问题。
- 改完 description 要重启客户端，config 和 skill 都不热加载。

### 根因一：排除项按关键词写，不按落点写

原 description 的「不适用于」写的是「新增或修改模型 adapter、wrapper、attention/MLP/MoE 结构（→ model-adapter）」。而 model-adapter 的 description 把 attention 写成肯定项（"补模型专属 attention·MLP wrapper"）。同一个词在一边是排除项、在另一边是包含项，模型看到 "attention" 只会走向 model-adapter。

但「在 workflow 里加 attention 计算」恰恰是应该被 amct-coder 拦住的越层请求——正确响应是命中 amct-coder 然后报 BLOCKER，不是转走。

**约束**：排除项必须写落点（路径 + 动作），不能只写结构名。凡是"结构名出现但落点不在模型层"的请求，description 里要有一条显式反例把它拉回来。

### 根因二：description 用元语言，没有用户原话

原文写的是"场景识别、分层边界判断、实现路径选择和最小自验"——这是维护者视角。用户实际说的是"加一段""这段放哪""这样写行不行"。仓内触发最稳的 `gitcode-pr` / `gitcode-issue` 都带显式的「必须触发此 skill 的场景」触发词清单。

**约束**：description 用 YAML `|` 块标量写多行，首行一句定位，其后是「必须触发此 skill 的场景」分组触发词清单，末行是按落点写的排除项。与仓内其它 skill 保持同一体例。

### 配套改动

SKILL.md §子能力路由 末尾新增越层请求对照表，把「workflow 里加 attention」「solver 里按模型名分支」「通用 quant module 里判 MoE」「给 attention 加新 wrapper 类」四条列出落点判断和正确响应。前三条留在本 skill 报 BLOCKER，只有第四条转 model-adapter。

### 触发验收用例（选项二下的正确期望）

触发正确性必须成对验证：同样出现 attention / wrapper 字样，区分依据是**落点路径**，不是关键词。单测任一条都测不出 description 是否写对。

| 测试语 | 落点 | 期望行为 |
| --- | --- | --- |
| 在 `amct_pytorch/workflows/llm_ptq.py` 里加一段 Qwen3 的 attention 计算 | `workflows/` | 命中 amct-coder，报越层 BLOCKER，把计算下沉到模型层 wrapper，workflow 只做编排。**不转 model-adapter** |
| 给 Qwen3-Next 加一个新 wrapper | `common/models/llm/qwen/qwen3_next/` | 命中 amct-coder，判出落点在模型层，转 `model-adapter`；实现完成后回到 amct-coder 做契约对齐检查 |

第二条的原始表述是"期望 agent 先读 repo-map 分层边界再读 casebook 个案，给出适配步骤"。选项二下这半段不成立：给出适配步骤是 `model-adapter` 的职责，其 SKILL.md 固定流程第 1、2 步已经要求先读 repo-map、再查 casebook 复用经验（`casebook/qwen/qwen3-next-80b-a3b-instruct.md` 已存在）。amct-coder 在这条用例里唯一要验的是**转对了**，不是转过去之后做了什么——否则等于要求 amct-coder 复制 model-adapter 的流程，把刚拆掉的重复又装回来。

量化门槛（≥5 越层场景 100% 拦截 + ≥3 合法改动 ≥90% 放行）里，越层样本至少覆盖：workflow 加结构计算、solver 按模型名分支、通用 quant module 判 MoE、CLI 加模型分支、为单模型改通用 bit policy。合法样本至少覆盖：新增 dtype、新增 PTQ 算法、改 deploy safetensors 导出。

### 后续风险

- description 已压缩为三行：定位、触发入口、委派边界。后续再改时不要重新枚举完整子能力表；子能力细节留在 `SKILL.md` 正文的路由表中，description 只保留能影响触发和委派判断的关键词。
- 反向不对称仍在：`model-adapter` 的 description 没写"在 workflow/solver 里加模型结构计算 → amct-coder"。受改动范围约束没动，靠 amct-coder 侧的正向声明兜。
- 本次没能跑通端到端验证：`opencode providers list` 显示 0 credentials，`opencode run` 直接 Unauthorized。只验证了 skill 仍被发现、YAML 仍可解析。真实触发效果需要用户在有凭证的环境复测。

### 2026-07-29 补齐状态

已在 `SKILL.md` 补回三类入口信息：

1. `执行流程`：说明用户请求如何先委派判断、再判代码体系、再命中主子能力。
2. `子能力路由` 与 `子能力编排`：把 9 个主子能力和 5 个辅助 reference 的触发场景、读取顺序、依赖方向集中到主入口。
3. `常见映射` 与 `越层请求对照`：用具体用户请求区分“同样出现 attention/wrapper/MoE 词时，按落点路径判断是拦截还是转 model-adapter”。

后续仍要用触发验收样本复测真实客户端行为；没有凭证时只做 YAML/frontmatter 和 skill 发现校验。
