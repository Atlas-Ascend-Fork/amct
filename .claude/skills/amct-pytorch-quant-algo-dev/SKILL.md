---
name: amct-pytorch-quant-algo-dev
description: |
  面向 amct_pytorch 量化算法的选型推荐与开发的技能。当用户给出一个待量化模型、问「推荐用哪个量化算法 / 量化方案」「该选 awq 还是 gptq」「W4 / W8 / W8A8 / MX / HIF8 该怎么配」「skip 哪些层」「group_size / strategy / symmetric 怎么设」「精度不达标怎么调优」时，用此 skill 给算法选型、量化策略和配置骨架建议；以及用户要新增 / 修改 PyTorch 量化算法、量化算子模块、部署模块、算法注册、量化配置、校准流程、convert 部署转换或相关单测、或直接用 amct.quantize / convert 量化一个现成模型时也用。适用于 AMCT PyTorch 中 awq、gptq、minmax、smoothquant、ofmr、cast、quantile、mxquant、自定义 algorithm_register 等路径。涉及关键词：量化算法推荐、量化方案选型、选哪个 algo、skip_layers、调优 ladder、预置 CFG、quantize/convert 使用、量化校准。
---

# AMCT PyTorch 量化算法开发

本 skill 用于在 `amct_pytorch` 中开发或调整量化算法。优先遵循现有注册、配置、模块替换和测试模式，避免引入旁路框架。

> **目录结构提示**：`amct_pytorch` 已重构为分层结构，量化算法实现集中在三个包：
> - `amct_pytorch/classic/` —— 经典 PTQ 主链路（`quantize.py`、`quantize_op/`、`deploy_op/`、`optimizer/`）
> - `amct_pytorch/algorithms/` —— 算法注册表与纯算法函数（`register_algo.py`、`__init__.py`、`quant/`）
> - `amct_pytorch/common/` —— 跨链路共享的配置与工具（`config/`、`utils/`）
>
> 注意 `BaseQuantizeModule` 基类从顶层 shim `amct_pytorch.quantize_op.base_quant_module` 导入（`classic/quantize_op/` 下的各算法模块也是这样引用它的），这是规范写法，不要改成 `classic.quantize_op.base_quant_module`。公开接口统一从顶层导出：`from amct_pytorch import quantize, convert, algorithm_register`、`from amct_pytorch.algorithms import AlgorithmRegistry`。

## 先读上下文

按任务范围读取最小必要文件：

```bash
rg -n "AlgorithmRegistry|algorithm_register|def quantize|def convert" amct_pytorch tests/amct_pytorch -g "*.py"
sed -n '1,220p' amct_pytorch/classic/quantize.py
sed -n '1,180p' amct_pytorch/algorithms/__init__.py
sed -n '1,260p' amct_pytorch/common/config/fields.py
sed -n '1,260p' amct_pytorch/common/config/parser.py
sed -n '1,220p' tests/amct_pytorch/test_customized_algo.py
```

常用参考：

- 注册入口（`quantize` / `convert` / `algorithm_register`）：`amct_pytorch/classic/quantize.py`
- 全局注册表类 `Algorithm` + 单例 `AlgorithmRegistry`：`amct_pytorch/algorithms/register_algo.py`
- 内置算法注册 + `BUILT_IN_ALGORITHM`：`amct_pytorch/algorithms/__init__.py`
- 算法纯函数（awq / auto_round / omniquant 等搜索逻辑）：`amct_pytorch/algorithms/quant/`
- 配置字段校验：`amct_pytorch/common/config/fields.py`
- 分层配置生成 + `parse_config` / `check_skip_layer`：`amct_pytorch/common/config/parser.py`
- 校准量化模块：`amct_pytorch/classic/quantize_op/`（基类 `BaseQuantizeModule` 从顶层 shim `amct_pytorch.quantize_op.base_quant_module` 导入）
- 部署模块：`amct_pytorch/classic/deploy_op/`
- 模块替换 pass：`amct_pytorch/classic/optimizer/insert_quantize_op_pass.py`、`amct_pytorch/classic/optimizer/replace_npu_quant_pass.py`
- 最小自定义算法样例：`tests/amct_pytorch/test_customized_algo.py`

## 调用链

`quantize(model, cfg)` 的主链路：

1. `parse_config(model, cfg, AlgorithmRegistry)` 解析配置，筛选支持层。
2. `InsertQuantizeModulePass` 按 `AlgorithmRegistry.algo[algo][src_op]` 将原始模块替换为校准量化模块。
3. 用户执行若干 batch 前向，量化模块收集或计算量化参数。
4. `convert(model)` 通过 `ReplaceNpuQuantModulePass` 按 `AlgorithmRegistry.quant_to_deploy[type(quant_module)]` 替换成部署模块。

开发时先判断改动属于哪一段：配置解析、校准量化模块、部署模块、算法工具函数，还是替换 pass。

## 直接使用：quantize → 校准 → convert（端到端配方）

当用户不是要改算法、而是想**用 amct 量化一个现成模型**时，给这套"拿来即用"的调用配方，不要让他去翻内部 Pass。公开接口只有三个：`quantize`、`convert`、`algorithm_register`，全部从顶层导入。

```python
import amct_pytorch as amct

# 1. 量化：用预置 CFG（推荐）或自己拼 dict。免校准算法（如 cast）到这步就出结果
amct.quantize(model, amct.HIFP8_CAST_CFG)        # quantize(model, config=None)

# 2. 校准（仅需要统计激活分布的算法，如 minmax/smoothquant/W4A8 等）：
#    喂少量有代表性的校准数据做前向，量化模块在前向里收集 scale/offset
with torch.no_grad():
    model(calib_data.to(device))                 # 一次或若干 batch，无需 backward

# 3. 转部署：把校准量化模块替换成 NPU 部署模块
amct.convert(model)                              # convert(model)
```

关键点（决定要不要第 2 步）：

- **免校准算法**（`cast`）：`quantize` 后直接 `convert`，无第 2 步。
- **需校准算法**（`minmax` / `smoothquant` / `ofmr` / W4A8 等）：必须在 `quantize` 和 `convert` 之间跑校准前向。校准数据量小即可（几十到一两百条有代表性样本，如 pileval/wikitext 片段），且**只前向、不反传**——量化模块在 forward 里累积统计量（参见各 `classic/quantize_op/*_module.py` 的 `forward`）。
- **校准样本组织**：常见做法是把校准文本 tokenize 后切成固定长度 block 拼成一个 batch，整体喂一次前向。`gptq` 这类按 `batch_num` 累积的算法，要喂够配置里要求的 batch 数才会触发权重重构。
- 真实端到端样例（含 pileval 校准的完整写法）见 benchmark skill 的 `scripts/ppl_eval_amct.py`（`fp8fp4` 分支）。

## 算法推荐

当用户给出一个待量化模型并询问「该用哪个算法」「推荐量化方案」「选哪个 algo」时，按以下流程给出建议，不要直接抛全部算法。

### 推荐输入

先尽量从对话或代码中确认四个事实，缺失就直接向用户问清楚，不要凭空假设：

1. **模型结构**：以 Linear 为主的 LLM / Transformer 类，是当前推荐覆盖的范围。Conv2d 主导的 CNN 场景目前实际使用很少，不在主推荐范围内；遇到时建议先与用户对齐是否一定要走 amct_pytorch。
2. **量化目标**：weight-only / weight+activation；目标位宽（W4 / W8 / W8A8 / MX 块量化 / HIF8）。
3. **可用数据**：是否有校准数据、batch 数量；能否做多轮前向；是否能反传。
4. **部署侧硬件**：是否走 NPU 部署模块（决定可选的 `deploy_op`）。

### 决策表

| 场景 | 推荐算法 | 说明 |
| --- | --- | --- |
| LLM，weight-only，W4/W8，重精度 | `gptq` | 逐层重构：用少量校准数据做若干次前向，累积输入激活的 Hessian（XᵀX），再用其逆做逐列权重补偿。纯前向，无需反向传播 |
| LLM，weight-only，W4/W8，重速度且对激活敏感 | `awq` | 基于激活幅度搜索 per-channel 缩放，校准成本较低 |
| LLM，weight+activation，W8A8 | `smoothquant` | 把激活离群值迁移到权重，配合 `NpuQuantizationLinear` 部署 |
| LLM，块量化 / 低位宽（MX 格式） | `mxquant` | NPU MX 部署模块，注意 group size 与 dtype 约束 |
| HIF8 weight cast | `cast` | 仅做 HIF8 浮点 cast，无校准过程 |
| 激活分布有显著离群值，需要鲁棒统计 | `quantile` | 用分位数代替 min/max，适合长尾激活 |
| Linear 层逐层重构（权重或权重+激活） | `ofmr` | 通过输出特征重构最小化误差，校准开销中等 |
| 第一次 baseline / 配置打通 / 回归对比 | `minmax` | 最小依赖，做精度地板和回归基线 |

`src_op` 与 `deploy_op` 的真实对应以 `amct_pytorch/algorithms/__init__.py` 中 `AlgorithmRegistry.register(...)` 为准；推荐前先核对该文件，避免推荐了未注册的算子组合。注意几个事实：`ofmr` 同时注册了 `Linear` 和 `Conv2d`（Conv2d 路径现已较少使用，不作为推荐入口）；`cast` 除 `Linear` 外还注册了 `FP8Linear`；`quantile` 额外注册了 `DeepseekV3Attention` 和 `LongcatFlashMLA` 两个注意力源算子（MLA / 注意力层量化能力，需要时核对注册表）。

### 输出格式

回答时按这个结构组织，避免发散：

1. **场景判断**：一两句话复述识别到的模型类型、目标位宽和约束。
2. **首选算法 + 备选**：给一个首选和最多两个备选，每个写一行选择理由。
3. **配置骨架**：基于 `## 配置规则` 中的结构，针对首选算法给出最小可跑的 `cfg`，标注关键字段（`weights.type`、`inputs`、`algorithm`、`group_size` 等）。
4. **下一步**：指明该跑哪个 `tests/amct_pytorch/test_<algo>.py` 或 `examples/algorithms/<algo>/` 样例做验证。

如果用户的需求落在内置算法都不满足的位置（例如新数据类型、新源算子），改走 `### 外部自定义算法` 流程，先用 `algorithm_register` 接入，再回到本节复核。

## 量化策略推荐

选定算法后，用户接下来问的往往是「该跳哪些层 / 用什么 dtype / 粒度怎么设」。回答时优先复用 `amct_pytorch/common/config/config.py` 中的预置（`INT4_AWQ_WEIGHT_QUANT_CFG`、`INT8_SMOOTHQUANT_CFG`、`MXFP4_AWQ_WEIGHT_QUANT_CFG` 等），再按场景做最小化调整，不要凭空捏新组合 —— 预置已经覆盖了 NPU 部署需要的 dtype/strategy 约束。

### skip_layers 建议

`skip_layers` 是模糊匹配（见 `amct_pytorch/common/config/parser.py:check_skip_layer`）：写 `'down_proj'` 会匹配所有 `*.down_proj`。基于现有预置和 example 的事实模式：

| 场景 | 推荐 skip | 依据 |
| --- | --- | --- |
| 任何 LLM 量化（默认基线） | `{'lm_head'}` | 8 个预置 cfg 全部默认 skip，输出层小误差会被放大成 PPL/logit 偏差，量化收益小 |
| W8A8 且激活离群严重（FFN 后） | `{'lm_head', 'down_proj'}` | `examples/algorithms/smoothquant/src/run_qwen_samples.py:52`、Qwen README 的实战配置 |
| 离群仍重（W8A8 难达标） | 再加 `o_proj` 或 `gate_proj/up_proj` | attention 输出和 SiLU 前线性层是常见离群源 |
| 多模态 / 特殊 head（vision_tower、mm_proj 等） | skip 这些前缀 | 这些层往往不是 Linear 主体或 NPU 不支持 |

不要默认 skip `q_proj/k_proj/v_proj` —— 它们参与量化收益最大，只在精度调优 ladder 的最后再考虑。

### dtype 选择

`weights.type` / `inputs.type` 的可选值以 `amct_pytorch/common/config/fields.py` 的字段校验为准。常见对应：

| 目标 | weights.type | inputs.type | 配套算法 / 预置 |
| --- | --- | --- | --- |
| W4 weight-only（最常见 LLM 压缩） | `int4` | 省略 | awq / gptq，`INT4_AWQ_WEIGHT_QUANT_CFG` / `INT4_GPTQ_WEIGHT_QUANT_CFG` |
| W8 weight-only | `int8` | 省略 | minmax，`INT8_MINMAX_WEIGHT_QUANT_CFG` |
| W8A8 | `int8` | `int8` | smoothquant，`INT8_SMOOTHQUANT_CFG` |
| MX 块量化（W4/W8） | `mxfp4_e2m1` / `mxfp8_e4m3fn` / `mxint8` | 同 weights（或省略） | mxquant / awq，`MXFP4_AWQ_WEIGHT_QUANT_CFG`、`MXFP8_QUANT_CFG` |
| FP8 | `float8_e4m3fn` | `float8_e4m3fn` | ofmr，`FP8_OFMR_CFG` |
| HiF8 weight + activation | `hifloat8` | `hifloat8` | cast / quantile / ofmr，`HIFP8_CAST_CFG`（免校准直转）、`HIFP8_QUANTILE_CFG`（分位校准）、`HIFP8_OFMR_CFG`（离群迁移） |
| HiF8 KV cache 量化 | （kvcache 字段） | — | quantile，`HIFP8_KVCACHE_CFG`（量化 KV cache，省 KV 显存） |

### strategy / group_size / symmetric

| 字段 | 默认值 | 何时切换 | 注意 |
| --- | --- | --- | --- |
| `weights.strategy` | `channel`（int8 baseline） | int4 / MX 切 `group` 提精度；FP8/HiF8 部署侧常用 `tensor` | `group` 必须配 `group_size`，且 `group_size` 整除 `in_features` |
| `inputs.strategy` | `tensor`（静态 per-tensor scale） | W8A8 + 激活离群严重，可尝试 `token`（per-token 动态） | per-token 通常要配 `symmetric: True` |
| `*.symmetric` | `True`（所有预置） | weights 几乎永远 True（zero-point=0 部署最高效）；inputs 在非对称分布场景才试 `False` | 改 `inputs.symmetric=False` 时确认对应 `deploy_op` 是否支持 |
| `group_size` | 32（MX）/ 128（INT4 group） | 精度敏感场景缩到 64/32 | 在 down_proj 这种 `in_features=11008` 的位置 group_size=128 不整除，得改值或 skip 该层 |

### 调优 ladder

精度不达标时按这个顺序调整，逐步加成本，每步都要重新校准 + 评估：

1. **加 skip_layers**：先 `down_proj`，再 `o_proj`、`gate_proj/up_proj`。
2. **细化 strategy**：weights `channel` → `group` (group_size 128 → 64 → 32)。
3. **算法参数微调**：smoothquant 调 `smooth_strength`（0.5 → 0.77 → 0.85），awq 调 `grids_num`（20 → 40）。
4. **换算法**：weight-only 场景 awq → gptq；激活离群场景 smoothquant → quantile。
5. **降量化目标**：W4A8 退回 W4A16；W8A8 退回 W8A16。

把这个 ladder 直接写进推荐回复的「下一步」一节，让用户知道精度不达标时第一步该改什么、不是无头苍蝇地全调。

### 输出格式补充

在「## 算法推荐 → 输出格式」原有四段基础上，**第 3 段「配置骨架」要做到**：

- 起点引用预置 cfg 名（如「基于 `INT8_SMOOTHQUANT_CFG`」），避免重复人工 dtype 校验；
- 显式列出 `skip_layers`，即使只是默认的 `{'lm_head'}`；
- 若改了 `strategy` 或 `symmetric` 的默认值，注明原因（如「激活离群严重，inputs 改 token + symmetric=True」）。

## 开发模式

### 外部自定义算法

优先使用公开接口 `algorithm_register(name, src_op, quant_op, deploy_op=None)`，不要修改内置注册表。

要求：

- `quant_op` 继承 `BaseQuantizeModule`，构造签名保持 `(ori_module, layer_name, quant_config)`。
- `deploy_op` 继承 `torch.nn.Module`，构造签名通常为 `(quant_module)`；没有部署转换时传 `None`，但要确认 `convert()` 行为是否符合预期。
- `src_op` 使用 `type(module).__name__` 能匹配的名称，例如 `Linear`、`Conv2d`。
- 自定义算法配置写入 `cfg["algorithm"]`，例如 `{"my_algo": {"param": 1}}`；自定义算法参数不会走内置字段强校验。

最小验证参考 `tests/amct_pytorch/test_customized_algo.py`：注册后断言 `AlgorithmRegistry.algo`、`AlgorithmRegistry.quant_to_deploy`，再验证 `quantize()` 和 `convert()` 后模块类型。

### 内置算法

新增内置算法时通常需要同时处理：

- 在 `amct_pytorch/classic/quantize_op/` 新增或复用校准量化模块。
- 必要时在 `amct_pytorch/algorithms/quant/` 放算法搜索、统计、缩放等纯函数。
- 在 `amct_pytorch/classic/deploy_op/` 新增或复用部署模块。
- 在 `amct_pytorch/algorithms/__init__.py` 注册算法、源算子、量化模块、部署模块，并更新 `BUILT_IN_ALGORITHM`。
- 如有专用算法参数，在 `amct_pytorch/common/config/fields.py` 增加字段类并接入 `AlgorithmField`。
- 如有预置配置，在 `amct_pytorch/common/config/config.py` 和 `amct_pytorch/common/config/__init__.py` 暴露。
- 如需公开给用户，在 `amct_pytorch/__init__.py` 更新 `__all__` 和导入。

量化模块中保留原模块必要状态，例如 `weight`、`bias`、`ori_module`、`ori_module_type`、`layer_name`、`scale_w`、`offset_w`、`scale_d`、`offset_d`、`group_size`。部署模块依赖这些属性时，必须在单测覆盖。

## 配置规则

配置结构以现有字段为准：

```python
cfg = {
    "batch_num": 1,
    "quant_cfg": {
        "weights": {"type": "int8", "symmetric": True, "strategy": "channel"},
        "inputs": {"type": "int8", "symmetric": True, "strategy": "tensor"},
    },
    "algorithm": {"minmax"},
    "skip_layers": {"lm_head"},
}
```

注意：

- `weights` 或模糊匹配的 `*.weights` 必须存在。
- `inputs` 省略时表示仅权重量化；若显式关闭输入量化，使用 `{"enable_quant": False}`。
- 内置算法会检查数据类型组合、粒度、group size、权重 dtype 和部分 NPU 部署约束。
- 自定义算法不会走内置算法约束，但仍会经过基础配置字段校验。
- 同一个 `src_op` 在一次配置中只能对应一个算法。

## 实现注意事项

- 优先复用 `amct_pytorch/classic/quantize_op/utils.py`、`amct_pytorch/common/utils/quant_util.py`、`amct_pytorch/common/utils/vars.py` 中已有量化工具和常量。
- 不要在算法实现里偷偷改全局注册表，注册应集中在公开入口或 `algorithm/__init__.py`。
- 不要为单个算法改动通用 pass，除非问题确实在模块替换通用逻辑。
- 部署模块涉及 `torch_npu` 时，测试中优先参考 `tests/amct_pytorch/mock_torch_npu.py` 和 `unittest.mock.patch`。
- 改内置算法时同步检查预置配置、导出符号、README/API 文档是否需要更新。
- 保持现有 Python 风格和版权头；新增文件参考邻近文件格式。

## 验证

优先运行受影响算法的定向测试：

```bash
pytest tests/amct_pytorch/test_customized_algo.py
pytest tests/amct_pytorch/test_config.py
pytest tests/amct_pytorch/test_minmax.py
pytest tests/amct_pytorch/test_awq.py
pytest tests/amct_pytorch/test_gptq.py
pytest tests/amct_pytorch/test_smoothquant.py
pytest tests/amct_pytorch/test_ofmr.py
pytest tests/amct_pytorch/test_cast.py
pytest tests/amct_pytorch/test_quantile.py
```

选择规则：

- 注册或自定义算法：至少跑 `test_customized_algo.py` 和相关配置测试。
- 配置字段或 parser：跑 `test_config.py`、`test_fuzzy_matching.py` 和目标算法测试。
- 部署模块或 `torch_npu` 调用：跑目标算法测试，并确认 mock 覆盖了新增 NPU API。
- 改共享量化工具：跑所有受影响算法测试，必要时扩到 `tests/amct_pytorch`。

验证后在回复中说明已运行的命令、结果，以及未覆盖的硬件/NPU 实机风险。
