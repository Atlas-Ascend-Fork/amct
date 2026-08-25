# 子能力：dtype-quant-module

## 简介

解决一种量化数据类型应该怎样定义、注册并接入通用量化模块的问题，保证数值格式、支持位宽、权重与激活用法、量化粒度以及导出数据格式形成一致契约。本子能力处理跨模型通用能力，不承载具体模型结构。

## 触发与职责

新增或修改 `quant_dtype`、INT/MXFP/HiFloat 类数值格式、fake/real quant、activation/weight quantizer、`QuantLinear`/`QuantizedMatmul`、bit policy 或 dtype deploy payload 时使用。

OWN：`amct_pytorch/quantization/dtypes/`、`amct_pytorch/quantization/modules/`、`amct_pytorch/quantization/bit_policy.py`。本层定义跨模型通用的数值语义和 quantizer 契约，不读取模型名。

PRODUCE：`DTYPE_REGISTRY` 条目、quantizer 实例化接口、`export_deploy()` payload 和 bit policy。CLI/public config、deploy、algorithm 兼容和可选 NPU 加速是下游闭环，不自动成为主修改面。

## 先冻结 dtype 契约

信息不足时先填写下述状态模型，只输出待确认项和条件式计划，不给默认 FP4/FP8 语义：

1. registry 名和 CLI 值，是否与现有 `int`/`mxfp` 冲突。
2. 支持 bit 集合，以及 `bits == 16` 在 fake quant、quant module 和 deploy 中的语义。
3. weight、activation、cache 的适用范围；不能从格式名推断 weight-only 或 weight+activation。
4. 编码格式、符号位/指数/尾数、共享指数、对称性、round/clamp、NaN/Inf/零、STE 梯度。
5. per-token/per-channel/group/block 粒度，量化维度、block size 和非整除行为。
6. real quant layout、packing 顺序、scale/bias dtype 与 shape、payload 字段名和反量化契约。
7. algorithm hook 是否支持 rounding offset `v` 或自定义 `quantize()`/`export_deploy()`。
8. deploy config 的 qtype/format/block size、blockwise/tensorwise 支持范围和下游消费者。
9. 是否需要 NPU kernel；需要时保留 Python 语义基线并转 `amct-ops-dev` 实现硬件算子。

### 契约状态模型

把当前仓库事实与新 dtype 的目标决策分开记录，不能用参考实现的现状替目标做决定：

| 维度 | 允许状态 | 证据要求 |
| --- | --- | --- |
| 当前仓库事实 | `confirmed` / `not found` | 给出当前 commit 的定义、调用方或直接测试；`not found` 表示本轮已查锚点未找到，不表示目标必须拒绝 |
| 目标契约 | `pending` / `supported` / `rejected` / `not applicable` | 给出用户输入、已冻结需求或兼容性裁决；`pending` 时写明缺失输入和条件分支 |

按以下规则转换目标状态：

1. 只读规划允许 role、bit、粒度、CLI 和 deploy 等目标项保持 `pending`。
2. 只有用户明确输入、已冻结需求、兼容性裁决或可直接决定目标语义的契约证据，才能把 `pending` 转为终态。
3. 当前 INT、MXFP 或 experimental 实现的 role、bit、粒度和 deploy 范围只进入“当前仓库事实”，不能单独触发目标状态转换。
4. 同一回答中没有新增证据时，后文必须保持前文的 `pending`，不得静默改写为 `supported`、`rejected` 或 `not applicable`。
5. 目标契约仍为 `pending` 时，只列候选锚点和条件分支，不给出唯一类名、实现文件、CLI 值、payload 字段或最终改动清单。
6. 实际编码遇到会产出不兼容实现的 `pending` 时，完成不依赖该项的事实核验后一次性询问用户，不落盘投机分支。

### 既有语义复用门

冻结新名字前，先用数值格式、block/group 粒度、packing layout、scale 语义和 weight/activation 适用范围，与现有 INT、MXFP、experimental dtype 逐项比较。若已冻结的目标数值语义被现有实现完整覆盖，优先选择零改动、公开别名或复用现有 registry/消费分支；只有至少一项真实语义或用户契约不同，才新增 dtype 类和 deploy format。名称不同、产品称谓不同或题目说“基于 INT4 扩展”都不能单独证明需要新 dtype，也不能把被参考 dtype 的适用范围复制成目标契约。

## 按动作选择源码锚点

| 当前动作 | 首轮锚点 | 契约成立后再读 |
| --- | --- | --- |
| 注册或新 dtype 类 | `dtypes/__init__.py`、目标 dtype 类/impl、`test_dtypes.py` | CLI choices；deploy/algorithm 仅按实际支持范围扩展 |
| fake quant 数值 | dtype `forward()`/`fake_quant()`/impl 与数值测试 | `ActivationQuantizer` 或 `WeightQuantizer` 中实际消费的一侧 |
| 通用 quant module | `quant_base.py` 与命中的 `quant_linear.py`/`quant_matmul.py`、直接测试 | algorithm hook；模型 wrapper 仅做契约 review，结构实现转 `model-adapter` |
| bit policy/config | `bit_policy.py`、命中 YAML、policy 测试 | eval/deploy 对 bit policy 的消费者 |
| deploy payload | dtype `export_deploy()`、`WeightQuantizer.export_deploy()`、deploy payload 消费点 | `deploy-export-code`、下游 config schema |
| NPU 加速 | Python dtype/impl 的确定语义、上层调用点 | `ops-boundary`；kernel/schema/Meta/CMake 转 `amct-ops-dev` |

`rg quant_dtype` 得到的模型或算法分支只是兼容性审计信号，不是默认必改列表。只有新 dtype 真实进入该路径且现有 fallback 语义不成立时才扩展；模型结构性变化转 `model-adapter`。

## 当前接通事实

- `register_dtype()` 通过 import 注册 `int` 和 `mxfp`；只有新增文件而没有进入该 import 链时，registry 不可达。
- `ActivationQuantizer` 用 `is_act=True` 实例化 dtype；`WeightQuantizer` 使用 weight 默认语义。两者 disabled 时直接返回输入。
- 当前 bit policy 允许 4、8、16；INT/MXFP 的 `forward()` 在 16 bit 时直通。
- INT activation 当前是 dynamic per-token；INT weight 当前是 per-row scale。INT4 real quant 还会 packing 并生成辅助 bias。
- MXFP 当前 block size 为 32；8 bit 输出 float8 payload，4 bit 输出 packed E2M1 和共享指数 scale。
- CLI `--quant_dtype` 当前 choices 为 `int`、`mxfp`。
- deploy 同时存在 `startswith("mx"/"int"/"hif")` 分类和 tensorwise `quant_dtype in ["int", "mxfp"]` 精确分支；新名字必须逐处判断，不能只更新 CLI。
- `generate_quant_config()` 当前以 `is_mx` 二分 float/int 格式；新类别不能默认套入任一侧。

## 分层实施顺序

1. 填写当前事实和目标契约状态；检查现有 LLM、classic、experimental 实现是否已经覆盖已冻结的目标语义，不同代码体系不能直接复用注册表。
2. 在 dtype/impl 层实现 fake quant、real quant、packing/dequant 和输入约束；保持 `bits == 16` 直通路径。
3. 接入 `register_dtype()`，验证 registry 重复注册、未知名称和 import 顺序。
4. 只在通用接口不兼容时修改 quant module；registry 构造已经满足时不重复增加 dtype 分支。
5. 同步 bit policy、CLI choice 和配置样例；仅允许真实支持的 bit/activation/weight 组合进入入口。
6. 若参与 deploy，按已冻结 payload 更新 blockwise/tensorwise 消费和 `quantization_config`；此阶段完整读取 `deploy-export-code`。
7. 若现有 algorithm 显式区分 dtype，逐个确认支持、拒绝或 fallback；不要无依据扩展所有算法。
8. 若模型侧显式依赖 dtype 选择 cache/layout，先做兼容性 review；需要模型专属结构实现时转 `model-adapter`。
9. 若需要 NPU 加速，Python 路径继续作为语义参考或 fallback，上层软依赖由 `ops-boundary` 处理。

## 消费面强制收口

新 dtype 的计划、实现 review 和最终说明必须逐项填写下表。每项分别记录“当前事实状态 + 源码/测试证据”和“目标契约状态 + 决策依据或缺失输入”；不得用“已注册”代替下游消费证据。

只读规划允许目标状态为 `pending`。实际编码前，所有会改变实现或公开契约的项目必须转为 `supported`、`rejected` 或 `not applicable`；仍为 `pending` 的项目按“契约状态模型”停止对应实现，不阻塞其它事实核验。

| 消费面 | 当前事实核验 | 目标契约决策 |
| --- | --- | --- |
| registry/import | 现有注册名、导入链、重复注册和未知名称行为 | 新名字是新增、别名、复用还是 `pending` |
| CLI | 当前 `--quant_dtype` choices/help 与 registry 的关系 | 是否公开新值、显式拒绝或保持 `pending` |
| bit policy/config | 当前 weight、activation、cache 各 role 的合法 bit/粒度 | 目标 role、bit、粒度是支持、拒绝、不涉及还是 `pending` |
| weight quantizer | 当前 fake/real quant、algorithm hook、16-bit 直通和 payload | 目标 weight 语义与所需扩展；未指定时保持 `pending` |
| activation quantizer | 当前 dynamic/static、shape/粒度和直通行为 | 目标 activation 语义；不支持时是否在稳定入口拒绝 |
| blockwise deploy | 当前 dtype payload、tensor 命名、index 和 config 消费者 | 目标是否参与及 payload 契约；未冻结时保持 `pending` |
| tensorwise deploy | 当前精确 dtype 分支、packing/layout、scale/bias 消费者 | 目标是否参与及布局契约；不得从 blockwise 支持自动推导 |
| algorithm compatibility | 当前显式 dtype 分支的支持、拒绝或 fallback | 目标算法组合的逐项决定；不按搜索命中盲改 |
| model-specific branches | 当前精确名称判断及 fallback | 是否真实受影响；没有目标证据时保持 `pending`，不是当然 `not applicable` |

若任一目标 `supported` 项只有定义文件而找不到活跃调用方，将当前事实记为 `not found`，并把目标实现列为尚未接通；不能伪装成已支持。若一个消费面按产品契约不支持新 dtype，应在最上游稳定入口拒绝并补测试，不能依赖下游偶然报错。

## 测试矩阵

| 层 | 最小断言 |
| --- | --- |
| dtype 数值 | shape/dtype、零和边界值、误差界、round/clamp、STE、非法 bit/shape/block |
| 直通 | weight 与 activation 的 16-bit 行为不改输入 |
| real quant | payload 必含 `qweight`，scale/bias shape 与 dtype，pack/unpack roundtrip，`v` 生效 |
| registry | `register_dtype()` 后可取、新名字不冲突、重复调用幂等 |
| quant module | activation/weight 正确实例化、enable/disable、algorithm hook、export_deploy |
| bit/CLI | 支持组合可解析，不支持组合显式拒绝，CLI choice 有最小 parser 测试 |
| deploy | safetensors 字段、tensor route、index、config 和反量化 layout 一致 |

优先运行 `tests/unit_test/quantization/test_dtypes.py`；quant module 变化再跑 `tests/unit_test/quantization/modules/`；CLI、deploy、algorithm 或 adapter 只在对应契约确实变化时追加其测试。

## BLOCKER

- 数值格式或适用范围未冻结，却按名称臆造实现语义或最终落点。
- 前文把 role、bit、粒度、CLI 或 deploy 范围标为 `pending`，后文没有新增证据却自行选择终态。
- 把 INT、MXFP、experimental 或其它参考实现的当前行为写入目标决策，借此默认 weight-only、weight+activation、固定 bit/粒度或 deploy 范围。
- 未比较现有 INT/MXFP/experimental 数值语义，仅因新名称重复创建等价 dtype。
- 新 dtype 未进入 `register_dtype()`，或 CLI/bit policy 允许框架实际不支持的组合。
- activation 误走 weight real-quant/packing 逻辑，或 `bits == 16` 不再直通。
- `export_deploy()` 字段、packing/layout 与 deploy 写入、index/config 或下游反量化不一致。
- 消费面状态写为 `supported`，但只有类/函数存在证据，没有实际 registry、CLI、quantizer 或 deploy caller。
- 仅因 `rg quant_dtype` 命中就修改所有 algorithm/adapter，或在通用 dtype/module 中写模型名分支。
- NPU kernel 没有 Python 语义基线/软依赖边界，或把 kernel/C++ 实现留在本子能力。

## 自验

- `pytest tests/unit_test/quantization/test_dtypes.py`
- quant module：`pytest tests/unit_test/quantization/modules/`
- CLI：`pytest tests/unit_test/cli/test_llm_module_entrypoints.py` 或新增的 parser 测试。
- deploy：`pytest tests/unit_test/workflows/test_llm_deploy.py tests/unit_test/common/models/llm/common/test_deploy_export.py`
- Skill 行为回归：用不含既有 dtype 名和 role 暗示的隔离规划题，验证未给出的目标项全程保持 `pending`，且不产生唯一实现落点。
- 真实 eval/deploy 或 NPU 数值未执行时，分别说明缺模型/数据/硬件环境。
