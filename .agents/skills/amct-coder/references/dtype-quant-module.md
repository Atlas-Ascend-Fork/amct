# 子能力：dtype-quant-module

## 简介

解决一种量化数据类型应该怎样定义、注册并接入通用量化模块的问题，保证数值格式、支持位宽、权重与激活用法、量化粒度以及导出数据格式形成一致契约。本子能力处理跨模型通用能力，不承载具体模型结构。

## 触发与职责

新增或修改 `quant_dtype`、INT/MXFP/HiFloat 类数值格式、fake/real quant、activation/weight quantizer、`QuantLinear`/`QuantizedMatmul`、bit policy 或 dtype deploy payload 时使用。

OWN：`amct_pytorch/quantization/dtypes/`、`amct_pytorch/quantization/modules/`、`amct_pytorch/quantization/bit_policy.py`。本层定义跨模型通用的数值语义和 quantizer 契约，不读取模型名。

PRODUCE：`DTYPE_REGISTRY` 条目、quantizer 实例化接口、`export_deploy()` payload 和 bit policy。CLI/public config、deploy、algorithm 兼容和可选 NPU 加速是下游闭环，不自动成为主修改面。

## 先冻结 dtype 契约

信息不足时只输出待确认项和条件式计划，不给默认 FP4/FP8 语义：

1. registry 名和 CLI 值，是否与现有 `int`/`mxfp` 冲突。
2. 支持 bit 集合，以及 `bits == 16` 在 fake quant、quant module 和 deploy 中的语义。
3. weight、activation、cache 的适用范围；不能从格式名推断 weight-only 或 weight+activation。
4. 编码格式、符号位/指数/尾数、共享指数、对称性、round/clamp、NaN/Inf/零、STE 梯度。
5. per-token/per-channel/group/block 粒度，量化维度、block size 和非整除行为。
6. real quant layout、packing 顺序、scale/bias dtype 与 shape、payload 字段名和反量化契约。
7. algorithm hook 是否支持 rounding offset `v` 或自定义 `quantize()`/`export_deploy()`。
8. deploy config 的 qtype/format/block size、blockwise/tensorwise 支持范围和下游消费者。
9. 是否需要 NPU kernel；需要时保留 Python 语义基线并转 `amct-ops-dev` 实现硬件算子。

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

1. 冻结上述 dtype 契约，并检查现有 LLM、classic、experimental 实现是否已经覆盖相同格式；不同代码体系不能直接复用注册表。
2. 在 dtype/impl 层实现 fake quant、real quant、packing/dequant 和输入约束；保持 `bits == 16` 直通路径。
3. 接入 `register_dtype()`，验证 registry 重复注册、未知名称和 import 顺序。
4. 只在通用接口不兼容时修改 quant module；registry 构造已经满足时不重复增加 dtype 分支。
5. 同步 bit policy、CLI choice 和配置样例；仅允许真实支持的 bit/activation/weight 组合进入入口。
6. 若参与 deploy，按已冻结 payload 更新 blockwise/tensorwise 消费和 `quantization_config`；此阶段完整读取 `deploy-export-code`。
7. 若现有 algorithm 显式区分 dtype，逐个确认支持、拒绝或 fallback；不要无依据扩展所有算法。
8. 若模型侧显式依赖 dtype 选择 cache/layout，先做兼容性 review；需要模型专属结构实现时转 `model-adapter`。
9. 若需要 NPU 加速，Python 路径继续作为语义参考或 fallback，上层软依赖由 `ops-boundary` 处理。

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
- 新 dtype 未进入 `register_dtype()`，或 CLI/bit policy 允许框架实际不支持的组合。
- activation 误走 weight real-quant/packing 逻辑，或 `bits == 16` 不再直通。
- `export_deploy()` 字段、packing/layout 与 deploy 写入、index/config 或下游反量化不一致。
- 仅因 `rg quant_dtype` 命中就修改所有 algorithm/adapter，或在通用 dtype/module 中写模型名分支。
- NPU kernel 没有 Python 语义基线/软依赖边界，或把 kernel/C++ 实现留在本子能力。

## 自验

- `pytest tests/unit_test/quantization/test_dtypes.py`
- quant module：`pytest tests/unit_test/quantization/modules/`
- CLI：`pytest tests/unit_test/cli/test_llm_module_entrypoints.py` 或新增的 parser 测试。
- deploy：`pytest tests/unit_test/workflows/test_llm_deploy.py tests/unit_test/common/models/llm/common/test_deploy_export.py`
- 真实 eval/deploy 或 NPU 数值未执行时，分别说明缺模型/数据/硬件环境。
