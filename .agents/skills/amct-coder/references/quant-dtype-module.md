# 子能力：quant-dtype-module

解决一种量化数据类型应该怎样定义、注册并接入通用量化模块的问题，保证数值格式、支持位宽、权重与激活用法、量化粒度以及导出数据格式形成一致的跨端约定。本子能力处理跨模型通用能力。

## 职责边界

本子能力只处理 `amct_pytorch/quantization/dtypes/` 与 `amct_pytorch/quantization/modules/` 两层（含 `amct_pytorch/quantization/bit_policy.py`）：跨模型通用的数值语义、quantizer 约定和 bit policy。

## 调用链

```text
dtype registry -> quant module -> role/bit policy
  -> algorithm hook -> fake/real quant -> deploy payload
```

## 实现定位

### 主要入口

- `amct_pytorch/quantization/dtypes/`：数值格式与 dtype 注册。
- `amct_pytorch/quantization/modules/`：通用量化模块。

本层定义跨模型通用数值语义和 quantizer 约定，不读取模型名。

### 关键符号

按动作搜索 `DTYPE_REGISTRY`、`register_dtype`、`BitPolicy`、`ActivationQuantizer`、`WeightQuantizer`、`QuantLinear`、`QuantizedMatmul`、`fake_quant` 和 `export_deploy`；确认定义、注册以及实际消费侧。

### 验证入口

核对数值边界、位宽/role/粒度、关量化行为和部署 payload。测试定位与命令见 [validation.md](validation.md#场景测试索引)，覆盖目标见本文件“测试矩阵”。

PRODUCE（本子能力的产出物）：`DTYPE_REGISTRY` 条目、quantizer 实例化接口、`export_deploy()` payload 和 bit policy。CLI/public config、deploy、algorithm 兼容和可选 NPU 加速是下游闭环，不自动成为主修改面。

## 当前事实

每条事实附一行「核对」命令；作为判断依据前先执行对应命令，结果与断言冲突时以命令结果为准；命令搜不到目标符号本身就是过期信号。发现不符时不改本节，把差异写进交付说明（判定见 [conditional-rules「当前事实与核对命令纪律」](conditional-rules.md#当前事实与核对命令纪律)）。

- `register_dtype()` 通过 import 注册 `int`、`mxfp` 和 `hifp`。只有新增文件而没有进入该 import 链时，registry 不可达。（核对：`rg -n "from \. import|import" amct_pytorch/quantization/dtypes/__init__.py` 中的 `register_dtype()` 函数体与 `rg -n 'name="' amct_pytorch/quantization/dtypes/int.py`）
- `ActivationQuantizer` 用 `is_act=True` 实例化 dtype；`WeightQuantizer` 使用 weight 默认语义。两者 disabled 时直接返回输入。（核对：`rg -n "is_act=True|def fake_quant" amct_pytorch/quantization/modules/quant_base.py`）
- 当前 bit policy 允许 4、8、16；INT/MXFP 的 `forward()` 在 16 bit 时直通。（核对：`rg -n "_ALLOWED_BITS" amct_pytorch/quantization/bit_policy.py` 与 `rg -n -A2 "bits == 16" amct_pytorch/quantization/dtypes/int.py amct_pytorch/quantization/dtypes/mxfp.py`）
- INT activation 当前是 dynamic per-token；INT weight 当前是 per-row scale。INT4 real quant 还会 packing 并生成辅助 bias。（核对：`rg -n "dynamic_per_token_quant|weight_quant|real_quant=True" amct_pytorch/quantization/dtypes/int.py`）
- MXFP 当前 block size 为 32；8 bit 输出 float8 payload，4 bit 输出 packed E2M1 和共享指数 scale。（核对：`rg -n "block_size|float8|f32_to_f4" amct_pytorch/quantization/dtypes/mxfp.py`）
- HiFp 当前只支持 4/8/16 bit：16 直通，8/4 走 fake quant；`export_deploy()` 只实现 4 bit（value + scale payload），8 bit 抛 `NotImplementedError`。`__init__` 接受 `is_act` 但不消费，weight 与 activation 是同一套 fake quant 语义，不像 INT 有 `is_act` 分流。（核对：`rg -n "is_act|bits == 16|export_deploy|NotImplementedError" amct_pytorch/quantization/dtypes/hifp.py`）
- CLI `--quant_dtype` 当前 choices 为 `int`、`mxfp`、`hifp`。（核对：`rg -n -A4 "quant_dtype" amct_pytorch/cli/llm/args.py`，`-A4` 才能覆盖到 `choices` 行）
- 算法侧存在 hifp 分支：AutoRound 以 `quant_dtype == "hifp"` 进入 HiF4 分支（仅 4 bit，group size 整除约束），GPTQ 以 `wts_type == 'hifloat4'` 判断；这些是精确名称匹配，不是 `startswith` 前缀类。int 类精确匹配点当前只接受 `"int"`（AutoRound `_get_group_size`、GPTQ `__init__`）。（核对：`rg -n '"hifp"|hifloat4|quant_dtype' amct_pytorch/algorithms/quant/auto_round.py amct_pytorch/algorithms/quant/gptq.py`）
- deploy 同时存在 `startswith("mx"/"int"/"hif")` 分类和 tensorwise `quant_dtype in ["int", "mxfp"]` 精确分支；新名字必须逐处判断，不能只更新 CLI。（核对：`rg -n 'startswith\(|quant_dtype in' amct_pytorch/workflows/llm_deploy.py`）
- `generate_quant_config()` 当前以 `is_mx` 二分 float/int 格式；新类别不能默认套入任一侧。（核对：`rg -n -A3 "def generate_quant_config" amct_pytorch/common/models/llm/common/deploy_export.py`）

## 前置检查

### dtype 约定

信息不足时先填写下述状态模型，只输出待确认项和条件式计划，不给默认 FP4/FP8 语义：

1. registry 名和 CLI 值，是否与现有 `int`/`mxfp` 冲突。
2. 支持 bit 集合，以及 `bits == 16` 在 fake quant、quant module 和 deploy 中的语义。
3. weight、activation、cache 的适用范围；不能从格式名推断 weight-only 或 weight+activation。
4. 编码格式、符号位/指数/尾数、共享指数、对称性、round/clamp、NaN/Inf/零、STE 梯度。
5. per-token/per-channel/group/block 粒度，量化维度、block size 和非整除行为。
6. real quant layout、packing 顺序、scale/bias dtype 与 shape、payload 字段名和反量化约定。
7. algorithm hook 是否支持 rounding offset `v` 或自定义 `quantize()`/`export_deploy()`。
8. deploy config 的 qtype/format/block size、blockwise/tensorwise 支持范围和下游消费者。
9. 是否需要 NPU kernel；需要时保留 Python 语义基线。

### 跨端约定状态表

本节只给 dtype 领域的记账机器：

| 维度 | 允许状态 | 证据要求 |
| --- | --- | --- |
| 当前仓库事实 | `confirmed` / `not found` | 给出当前 commit 的定义、调用方或直接测试；`not found` 表示本轮已查锚点未找到，不表示目标必须拒绝 |
| 目标约定 | `pending` / `supported` / `rejected` / `not applicable` | 给出用户输入、已确定需求或兼容性裁决；`pending` 时写明缺失输入和条件分支 |

领域补充：当前 INT、MXFP 或 experimental 实现的 role、bit、粒度和 deploy 范围只进入“当前仓库事实”，不能单独触发目标状态转换。

### 命名约定与动作一致性

产品称谓、CLI 值、registry key、Python 文件/类标识符和可选 ops schema 名分别记录为目标约定；它们可以相同，也可以映射，但仓库惯例只能形成候选映射，公开值或兼容别名仍需用户输入、已确定需求或兼容性证据。待定项与当前动作的对应关系按下表执行：

| 目标项仍为 `pending` | 不能写成当前立即动作 |
| --- | --- |
| 新增、别名或复用 | 创建唯一 dtype 类/文件、registry key 或导入项 |
| CLI 是否公开 | 修改 `choices`、help、配置或样例 |
| deploy 参与范围 | 修改 payload、packing、index 或 `quantization_config` |
| NPU 加速与接口 | 创建确定的 kernel、schema、Meta、构建项或算子目录 |

交付前逐项比较目标状态与修改动作、修改位置、实施顺序和修改后测试。

### 既有实现复用

确定新名字前，先用数值格式、block/group 粒度、packing layout、scale 语义和 weight/activation 适用范围，与现有 INT、MXFP、experimental dtype 逐项比较。若已确定的目标数值语义被现有实现完整覆盖，优先选择零改动、公开别名或复用现有 registry/消费分支；只有至少一项真实语义或用户跨端约定不同，才新增 dtype 类和 deploy format。名称不同、产品称谓不同或题目说“基于 INT4 扩展”都不能单独证明需要新 dtype，也不能把被参考 dtype 的适用范围复制成目标约定。

### 下游接入点

新 dtype 的计划、实现 review 和最终说明必须逐项填写下表。每项分别记录“当前事实状态 + 源码/测试证据”和“目标约定状态 + 决策依据或缺失输入”；不得用“已注册”代替下游消费证据。`pending` 项的处理遵循跨端约定状态表的对应关系，不阻塞其它事实核验。

| 下游接入点 | 当前事实核验 | 目标约定决策 |
| --- | --- | --- |
| registry/import | 现有注册名、导入链、重复注册和未知名称行为 | 新名字是新增、别名、复用还是 `pending` |
| CLI | 当前 `--quant_dtype` choices/help 与 registry 的关系 | 是否公开新值、显式拒绝或保持 `pending` |
| bit policy/config | 当前 weight、activation、cache 各 role 的合法 bit/粒度 | 目标 role、bit、粒度是支持、拒绝、不涉及还是 `pending` |
| weight quantizer | 当前 fake/real quant、algorithm hook、16-bit 直通和 payload | 目标 weight 语义与所需扩展；未指定时保持 `pending` |
| activation quantizer | 当前 dynamic/static、shape/粒度和直通行为 | 目标 activation 语义；不支持时是否在稳定入口拒绝 |
| blockwise deploy | 当前 dtype payload、tensor 命名、index 和 config 消费者 | 目标是否参与及 payload 约定；未确定时保持 `pending` |
| tensorwise deploy | 当前精确 dtype 分支、packing/layout、scale/bias 消费者 | 目标是否参与及布局约定；不得从 blockwise 支持自动推导 |
| algorithm compatibility | 当前显式 dtype 分支的支持、拒绝或 fallback | 目标算法组合的逐项决定；不按搜索命中盲改 |
| model-specific branches | 当前精确名称判断及 fallback | 是否真实受影响；没有目标证据时保持 `pending`，不是当然 `not applicable` |

若任一目标 `supported` 项只有定义文件而找不到活跃调用方，将当前事实记为 `not found`，并把目标实现列为尚未接通；不能伪装成已支持。若一个下游接入点按产品约定不支持新 dtype，应在最上游稳定入口拒绝并补测试，不能依赖下游偶然报错。

## 按动作选择锚点

| 当前动作 | 首轮锚点 | 约定成立后再读 |
| --- | --- | --- |
| 注册或新 dtype 类 | `DTYPE_REGISTRY`、`register_dtype()`、目标 dtype 类/impl | CLI choices；deploy/algorithm 仅按实际支持范围扩展 |
| fake quant 数值 | dtype `forward()`/`fake_quant()`/impl | `ActivationQuantizer` 或 `WeightQuantizer` 中实际消费的一侧 |
| 通用 quant module | 目标 quantizer、`QuantLinear` 或 `QuantizedMatmul` | algorithm hook；模型 wrapper 的跨端约定 |
| bit policy/config | `BitPolicy`、命中 YAML 及其加载方 | eval/deploy 对 bit policy 的消费者 |
| deploy payload | dtype `export_deploy()`、`WeightQuantizer.export_deploy()`、deploy payload 消费点 | `llm-deploy`、下游 config schema |
| NPU 加速 | Python dtype/impl 的确定语义、上层调用点、可选 import、fallback 与报错时机 | 已确定的算子接口（schema、算子目录） |

`rg quant_dtype` 得到的模型或算法分支只是兼容性审计信号，不是默认必改列表。只有新 dtype 真实进入该路径且现有 fallback 语义不成立时才扩展。

## 分层实施顺序

以下步骤只对相应目标约定已经转为 `supported` 的部分成立；仍为 `pending` 的步骤保留为条件分支，不进入当前改动清单。

1. 填写当前事实和目标约定状态；检查现有 LLM、classic、experimental 实现是否已经覆盖已确定的目标语义，不同代码体系不能直接复用注册表。
2. 在 dtype/impl 层实现 fake quant、real quant、packing/dequant 和输入约束；保持 `bits == 16` 直通路径。
3. 接入 `register_dtype()`，验证 registry 重复注册、未知名称和 import 顺序。
4. 只在通用接口不兼容时修改 quant module；registry 构造已经满足时不重复增加 dtype 分支。
5. 仅在目标 CLI/config 参与状态为 `supported` 时同步 bit policy、CLI choice 和配置样例；只允许真实支持的 bit/activation/weight 组合进入入口。
6. 仅在目标 deploy 参与状态为 `supported` 时，按已确定 payload 更新 blockwise/tensorwise 消费和 `quantization_config`；确认存在真实跨端影响后再读取 `llm-deploy`。
7. 若现有 algorithm 显式区分 dtype，逐个确认支持、拒绝或 fallback；不要无依据扩展所有算法。
8. 若模型侧显式依赖 dtype 选择 cache/layout，先做兼容性 review。
9. 仅在目标 NPU 加速状态为 `supported` 时，以 Python 路径作为语义参考或 fallback；上层软依赖、延迟 import 和报错时机属本子能力，schema 与算子目录以已确定的算子接口为准。

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

优先验证 dtype 数值约定；quant module 变化再验证模块消费行为；CLI、deploy、algorithm 或 adapter 只在对应约定确实变化时追加其测试。

## 禁止项

- 数值格式或适用范围未确定，却按名称臆造实现语义或最终修改位置。
- 把 INT、MXFP、experimental 或其它参考实现的当前行为写入目标决策，借此默认 weight-only、weight+activation、固定 bit/粒度或 deploy 范围。
- 未比较现有 INT/MXFP/experimental 数值语义，仅因新名称重复创建等价 dtype。
- 新 dtype 未进入 `register_dtype()`，或 CLI/bit policy 允许框架实际不支持的组合。
- activation 误走 weight real-quant/packing 逻辑，或 `bits == 16` 不再直通。
- `export_deploy()` 字段、packing/layout 与 deploy 写入、index/config 或下游反量化不一致。
- 下游接入点状态写为 `supported`，但只有类/函数存在证据，没有实际 registry、CLI、quantizer 或 deploy caller。
- 仅因 `rg quant_dtype` 命中就修改所有 algorithm/adapter，或在通用 dtype/module 中写模型名分支。
- NPU kernel 没有 Python 语义基线/软依赖边界。

## 自验

- 测试定位与命令见 [validation.md](validation.md#场景测试索引)。
- dtype/quant module：覆盖上面的数值与模块测试矩阵。
- CLI：验证配置解析、支持值和 role 位宽生效。
- deploy：分别验证 payload、binding 和 workflow 产物消费。
- 真实 eval/deploy 或 NPU 数值未执行时，分别说明缺模型/数据/硬件环境。
