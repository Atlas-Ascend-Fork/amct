# Fake Quant Forward 与 Bias 量化 设计文档

| 项 | 内容 |
|---|---|
| 特性名称 | Fake quant forward 与 bias 量化 |
| 关联分支 | feature/fake-quant-forward-v2 |
| 关联 MR | annqr/amct_open!（fake-quant forward + bias 量化） |
| 设计依据 | docs/api/quantize.md「全量化支持的层」 |

---

## 1. 简述方案

本特性面向训练后量化（PTQ）的精度评测场景，提供两项能力：

1. **Fake quant forward**：校准模块在完成 `batch_num` 轮校准后，其 `forward()` 不再返回原始浮点结果，而是用校准得到的量化因子对权重、激活（及 bias）做 quant→dequant，从而在 CPU/GPU 上近似量化引入的数值误差，无需经过 `convert()` 与 NPU 硬件即可评估量化后精度。

2. **Bias 量化**：在 fake quant 评测路径与 deploy 部署路径两端，按数据类型组合对 bias 施加量化。fake quant 侧的目标并非逐位复刻 NPU 的真实 bias 加法实现，而是在已有的 weight/activation fake quant 之外，让 bias 也经过相应的量化处理，从而使 bias 同样反映量化影响；deploy 侧则负责产出 NPU 算子期望的 bias 形态。按组合区分：INT8 per-tensor 路径将 bias 量化为 INT32（fake quant 用 quant→dequant 模拟，deploy 产出 INT32 buffer）；FP8/HIFLOAT8 对称组合保持浮点 bias；fp8*fp4 组合不支持 bias。

实现层面，每个校准子类自行实现 `fake_quant_forward(inputs)`：基类 `BaseQuantizeModule` 不再承载任何 fake quant 钩子，避免对 attention 类等不适配 Linear/Conv2d 假设的子类施加无意义约束；具体的 quant→dequant 调用与 bias 量化逻辑直接内联在每个 Linear-like 子类内，子类按需选择 `quant_dequant_tensor` 或 `apply_progressive_quant_dequant`。

---

## 2. 规格

### 2.1 数据类型组合与 bias 处理矩阵

| 量化组合 | 激活粒度 | Fake quant bias | Deploy bias 形态 | Deploy 是否传入 NPU 算子 |
|---|---|---|---|---|
| INT8 + INT8 | per-tensor，非 dynamic | 量化为 INT32（quant→dequant 模拟） | INT32 buffer | 是（算子内累加） |
| INT8 + INT8 | per-token | 不量化（保持浮点） | 浮点 | 是 |
| INT8 + INT8 | dynamic | 不量化 | 浮点 | 是 |
| FP8 + FP8 | per-tensor（唯一） | 不量化 | 浮点 | 否（算子外浮点相加） |
| HIFLOAT8 + HIFLOAT8 | per-tensor（唯一） | 不量化 | 浮点 | 否（算子外浮点相加） |
| fp8*fp4 | per-tensor | 不适用（bias 恒为 None） | None / offset_bias | 视 offset_bias |
| 无 bias | 任意 | None | None / offset_bias | 视 offset_bias |

### 2.2 适用模块

- 校准模块：`MinMaxQuant`、`SmoothQuant`、`OfmrQuant`、`LinearAWQuant`、`GPTQuant`、`QuantileQuant`（均继承 `BaseQuantizeModule`，各自实现 `fake_quant_forward`）。
- attention 类（`DeepseekV3AttentionQuant` / `LongcatFlashMLAQuant`）也继承 `BaseQuantizeModule`，但不参与 fake quant，对本特性透明。
- 部署模块：`NpuQuantizationLinear`。
- 算子类型：`torch.nn.Linear`（全部组合）、`torch.nn.Conv2d`（仅 OfmrQuant，且全量化组合仅 FP8×FP8 / HIFLOAT8×HIFLOAT8）。

### 2.3 数学规格

INT8 per-tensor 路径，记 `deq_scale = scale_d × scale_w`：

- Fake quant bias：`bias_q = round(bias / deq_scale).clamp(INT32_MIN, INT32_MAX)`，乘回 `deq_scale` 得到浮点 bias，传入 `F.linear` / `F.conv2d`。该 quant→dequant 让 bias 也带上 INT8 量化误差，与 weight/activation 的 fake quant 方式一致；目的是在评测中体现 bias 量化的影响，并非逐位复刻 NPU 的整数 bias 加法实现。
- 不显式模拟 NPU 整数累加器的 INT32 溢出截断（常规矩阵规模 K ≤ 16384 不触发）。

---

## 3. 输入

### 3.1 Fake quant 路径（校准模块 forward）

| 输入 | 来源 | 说明 |
|---|---|---|
| inputs | 调用方 | 推理输入张量；Linear 为 2~6 维，Conv2d 为 NCHW |
| self.weight / self.bias | ori_module | 原始权重与 bias |
| self.scale_w / scale_w1 / scale_w2 | 校准结果 | 权重量化因子（fp8*fp4 用 scale_w1/scale_w2） |
| self.scale_d / offset_d | 校准结果 | 激活量化因子（仅激活量化启用时存在） |
| self.scale | 校准结果 | SmoothQuant smooth_factor 或 AWQ pre-scale |
| self.act_type / wts_type / act_granularity / group_size | 配置 | 量化类型与粒度 |

### 3.2 Deploy 路径（NpuQuantizationLinear）

| 输入 | 来源 | 说明 |
|---|---|---|
| quant_module | 校准模块 | 提供 bias、scale_d、scale_w(_tensor)、act_type、wts_type、act_granularity、dynamic |
| self.scale_w_tensor | _init_weight_quant | 权重量化因子张量（在 _init_bias 之前已设置） |
| self.offset_bias | _init_weight_quant | 非对称量化的 INT32 偏置修正（可能为 None） |

---

## 4. 输出

### 4.1 Fake quant 路径

- 输出量化误差已注入的浮点张量，dtype 与原始权重一致；形状与对应 `F.linear` / `F.conv2d` 的浮点输出一致。
- 校准未完成时（`cur_batch <= batch_num` 等），输出仍等于原始浮点结果，不引入量化误差。

### 4.2 Deploy 路径

- `_init_bias` 产出注册到 buffer 的 `self.bias`（INT32 或浮点或 None）与布尔属性 `self.bias_for_npu_op`。
- `forward` 输出经 `npu_quant_matmul` 计算并按 `bias_for_npu_op` 决定 bias 加法位置后的张量，dtype 为 `output_dtype`。

---

## 5. 限制

1. **算子覆盖**：bias 量化仅在 `NpuQuantizationLinear` 范围内；MXFP8 走 `NpuMXQuantizationLinear`、仅权重量化走 `NpuWeightQuantizedLinear`，均不在本特性范围。
2. **fp8*fp4 bias**：fp8*fp4 组合不支持带 bias 的 Linear，bias 由上游保证为 None，代码不做额外防御。
3. **Conv2d 全量化**：仅 FP8×FP8 / HIFLOAT8×HIFLOAT8，均走 bias 浮点外加路径。
4. **FP8/HIFLOAT8 量化算子依赖 NPU**：fake quant 中 FP8/HIFLOAT8 的 `quant_tensor` 调用 `torch_npu` 算子，纯 CPU 环境仅 INT8/INT4 路径可独立运行；deploy 路径整体依赖 NPU 硬件。
5. **缓存有效性**：每个子类的 `fake_quant_forward` 假设校准完成后权重、scale、bias 不再变化，首次调用时把 dequantized weight 与（INT8 per-tensor 时的）量化后 bias 缓存为实例属性 `cached_dq_w` / `cached_bias`，并置 `fake_quant_cache_ready=True`；若调用方在首次 `fake_quant_forward` 之后修改这些属性，需手动清除缓存标志。
6. **INT32 累加溢出**：不显式模拟 NPU 整数累加器溢出截断（常规矩阵规模 K ≤ 16384 不触发）。

---

## 6. 模块设计

### 6.1 总体结构

```
BaseQuantizeModule (amct_pytorch/quantize_op/base_quant_module.py)
  仅持有通用属性（act_type/wts_type/scale_w/scale_d/offset_w/offset_d/group_size/scale）。
  不承载任何 fake quant 钩子；attention 子类对 fake quant 完全透明。

amct_pytorch/classic/quantize_op/<algo>_module.py
  ├─ MinMaxQuant.fake_quant_forward(inputs)
  ├─ SmoothQuant.fake_quant_forward(inputs)        # 折 smooth_factor
  ├─ OfmrQuant.fake_quant_forward(inputs)          # Linear/Conv2d 分发
  ├─ LinearAWQuant.fake_quant_forward(inputs)      # 折 awq pre-scale
  ├─ GPTQuant.fake_quant_forward(inputs)           # 仅权重量化
  └─ QuantileQuant.fake_quant_forward(inputs)      # weight-only / dynamic / static 三分支
  每个子类首次调用时缓存 cached_dq_w（及 INT8 per-tensor 时的 cached_bias），
  设置 fake_quant_cache_ready=True，后续仅重算激活 dequant。

amct_pytorch/classic/quantize_op/utils.py
  └─ apply_progressive_quant_dequant()  fp8*fp4 两级 quant→dequant（per-channel FP8 → per-group FP4）

amct_pytorch/classic/deploy_op/npu_quantization_linear.py
  ├─ _init_bias(module)  决定 bias 形态并设置 bias_for_npu_op
  └─ forward(x)          按 bias_for_npu_op 派发 bias 加法位置

amct_pytorch/common/utils/vars.py
  └─ INT32_MAX / INT32_MIN 常量
```

### 6.2 关键设计点

- **基类零包袱**：`BaseQuantizeModule` 不放 fake quant 钩子，attention 类等不适配 Linear/Conv2d 假设的子类无需面对它们；新增的算法子类自由决定是否实现 `fake_quant_forward`。
- **每子类自治**：每个 Linear-like 子类直接调用 `quant_dequant_tensor` / `apply_progressive_quant_dequant`，无需通过基类 helper 中转，让每个子类的 fake quant 行为一目了然。
- **缓存策略**：与 batch 无关的量（dequantized weight、INT8 per-tensor 时的量化后 bias）首次调用时计算并缓存为实例属性，后续 batch 仅重算激活 dequant，降低大权重矩阵的重复开销。
- **派发标志**：deploy 端 bias「算子内加 / 算子外加」的决策在 `_init_bias` 一次性确定并记录于 `bias_for_npu_op`，`forward` 仅读取标志，避免两处条件判断不同步。
- **粒度即区分**：fake quant 端的 INT8 PER_TENSOR bias 量化条件不显式判断 `dynamic`，因为 dynamic per-token 走 `act_granularity == 'token'`，已被 per-tensor 判断排除。

---

## 7. 异常与 DFX 设计

### 7.1 显式异常

| 触发点 | 条件 | 行为 |
|---|---|---|
| apply_progressive_quant_dequant | scale_w1 形状 ≠ [n] | 抛 RuntimeError，提示期望形状与实际形状 |
| apply_progressive_quant_dequant | scale_w2 形状 ≠ [n*k/g, 1] | 抛 RuntimeError，提示期望形状与实际形状 |

### 7.2 隐式保护与容错

- **bias 为 None**：子类在 `fake_quant_forward` 中仅当 `bias is not None` 时才进入 INT32 量化分支，`F.linear` / `F.conv2d` 接受 `bias=None`，不报错。
- **scale 缺失**：`scale_d` 为 None（仅权重量化）或 `scale_w` 为 None（progressive 路径）时跳过 bias INT32 量化，bias 透传原始浮点张量。
- **校准未完成**：`fake_quant_forward` 仅在校准完成后被调用，调用时机由各子类 forward 控制（`cur_batch > batch_num` / `calc_done` / `weight_compress_only` / `dynamic is True`）。
- **设备一致性**：SmoothQuant 与 AWQ 在 fake quant 内对 `self.scale` 调用 `.to(device=..., dtype=...)`，避免跨设备/跨 dtype 运算异常。

### 7.3 DFX（可定位/可维护）

- fp8*fp4 形状校验的 RuntimeError 携带期望值与实际值，便于定位 scale 配置错误。
- `bias_for_npu_op` 作为单一信号源，使 bias 派发路径可在 `_init_bias` 处一次性审阅。
- 缓存属性命名规范：`cached_dq_w`（权重 dequant）、`cached_bias`（INT8 per-tensor 量化后 bias）、`fake_quant_cache_ready`（就绪标志）。状态可读、可在调试器中直接观察。

---

## 8. 核心函数接口

仅列接口签名与语义，不含实现细节。

### 8.1 各校准子类

每个 Linear-like 子类实现自己的 `fake_quant_forward`；签名一致，差别在于子类如何准备 input/weight。

| 子类 | 函数 | 关键差异 |
|---|---|---|
| MinMaxQuant | `fake_quant_forward(self, inputs)` | 直接对 weight、input 做 quant→dequant；INT8 per-tensor 时缓存量化后 bias |
| SmoothQuant | `fake_quant_forward(self, inputs)` | 先把 `weight × smooth_factor`、`inputs / smooth_factor`，再 quant→dequant |
| OfmrQuant | `fake_quant_forward(self, inputs)` | Linear/Conv2d 派发，HIFLOAT8/FP8 路径不做 bias INT32 量化 |
| LinearAWQuant | `fake_quant_forward(self, inputs)` | `inputs × self.scale`（self.scale = 1/scale_awq），仅权重量化 |
| GPTQuant | `fake_quant_forward(self, inputs)` | 直接对 GPTQ 已优化的权重做 quant→dequant |
| QuantileQuant | `fake_quant_forward(self, inputs)` | 三分支：weight-only/dynamic 实时算 scale_d/static |

返回值统一为浮点 Tensor，形状等同对应的 `F.linear` / `F.conv2d` 输出。

### 8.2 amct_pytorch/classic/quantize_op/utils.py

| 函数 | 签名 | 返回 | 语义 |
|---|---|---|---|
| apply_progressive_quant_dequant | `apply_progressive_quant_dequant(weight, scale_w1, scale_w2, group_size=32)` | Tensor | per-channel FP8 → per-group FP4 两级 quant→dequant，返回与 weight 同形浮点张量 |

### 8.3 NpuQuantizationLinear

| 函数 | 签名 | 返回 | 语义 |
|---|---|---|---|
| _init_bias | `_init_bias(self, module)` | None | 按数据类型组合确定 bias 形态（INT32 / 浮点 / None），注册到 buffer，并设置 `bias_for_npu_op`；FP8/HIFLOAT8 路径在 fp 域加 offset_bias 后早 return，op 外浮点相加 |
| forward | `forward(self, x)` | Tensor | 量化激活后调用 npu_quant_matmul；`is_new_torch_npu` 分支按 `bias_for_npu_op` 决定 bias 算子内传入或算子外相加 |

---

## 9. 测试用例设计

测试文件：`tests/amct_pytorch/test_fake_quant_forward.py`，共 10 个用例，端到端走 `quantize` + `forward`，依赖 `tests/amct_pytorch/conftest.py`（注入项目根路径、mock graph_based_compression 与 transformers）。执行：`cd tests/amct_pytorch && python -m pytest test_fake_quant_forward.py -v`。

「测试项」指被验证的功能点（一个测试项对应一段需要被验证的能力）；每个测试项下有一个或多个「测试用例」，用例名称即实际的 pytest 标识符，便于按名执行与回溯。

### 9.1 测试项总览

| 测试项编号 | 测试项（被验证的功能点） | 用例数 | 用例类型 |
|---|---|---|---|
| TI-01 | MinMax 校准完成后 forward 引入量化误差，覆盖 INT8 全量化与 INT8 仅权重 | 2 | 功能 |
| TI-02 | SmoothQuant 校准完成后 forward 引入量化误差，覆盖 INT8 全量化与 fp8*fp4 progressive 路径 | 2 | 功能 |
| TI-03 | OfmrQuant 校准完成后 fake quant 路径可正常执行（HIFLOAT8 全量化） | 1 | 功能 |
| TI-04 | LinearAWQuant 校准完成后 forward 引入量化误差并应用 awq pre-scale | 1 | 功能 |
| TI-05 | GPTQuant 校准完成后 forward 引入量化误差（仅权重量化） | 1 | 功能 |
| TI-06 | QuantileQuant 三条 fake quant 分支（静态激活、dynamic per-token、仅权重）端到端可用 | 3 | 功能 |

测试基础设施：`FakeQuantTestBase.setUp` 把内嵌 `mock_torch_npu` 函数注入 `sys.modules['torch_npu']`，让 `quant_dequant_tensor` 等内部函数走 mock；测试模型为本文件内的 `_Model`/`_ModelBias`（避开 `tests/amct_pytorch/utils.py` 顶部对 `transformers.models.deepseek_v3` 的硬依赖）。

### 9.2 TI-01 MinMax 校准完成后 forward 引入量化误差

验证目标：`MinMaxQuant` 在校准完成后，forward 输出与原始浮点输出不再相同——即 `fake_quant_forward` 路径生效，量化误差被注入到输出。覆盖 INT8 全量化与 INT8 仅权重两条路径。

| 用例名称（pytest） | 输入 | 预期结果 | 类型 | 操作步骤 |
|---|---|---|---|---|
| TestMinMaxFakeQuant::test_int8_int8_post_calibration_diverges_from_fp | INT8 + INT8 per-tensor，跑完 1 个 calibration batch | 第二次 forward 输出 ≠ 原模型浮点输出 | 功能 | 1) `quantize(model, cfg)`；2) `model(x)` 触发校准完成；3) 取浮点参考 `fp_ref = self.test_model(x)`；4) `fake_out = model(x)`；5) 断言 `not torch.allclose(fp_ref, fake_out, atol=0)` |
| TestMinMaxFakeQuant::test_weight_only_int8_diverges_from_fp | 仅权重 INT8 per-channel | 首次 forward 输出 ≠ 浮点输出（weight_compress_only 立即生效） | 功能 | 1) `quantize`；2) 浮点参考；3) `model(x)`；4) 断言不 allclose |

### 9.3 TI-02 SmoothQuant 校准完成后 forward 引入量化误差

验证目标：`SmoothQuant` 在两条主路径（INT8 全量化、fp8*fp4 progressive）下，校准完成后 forward 输出 ≠ 原始浮点输出，确认 smooth_factor 折叠 + 权重/激活/bias 的 quant→dequant 链路被触发。

| 用例名称（pytest） | 输入 | 预期结果 | 类型 | 操作步骤 |
|---|---|---|---|---|
| TestSmoothQuantFakeQuant::test_int8_int8_diverges_from_fp | INT8 channel + INT8 per-tensor + smoothquant(0.5) | 校准后 forward 输出 ≠ 浮点 | 功能 | 1) `quantize`；2) 跑一次校准；3) 浮点参考；4) `model(x)`；5) 断言不 allclose |
| TestSmoothQuantFakeQuant::test_fp8_fp4_progressive_diverges_from_fp | fp8 per-tensor + fp4_e2m1 group(32) + smoothquant，使用 `_ModelBias`（中间 layer bias=False 满足 fp8*fp4 限制） | 校准后 forward 输出 ≠ 浮点 | 功能 | 1) 用 `_ModelBias` 构造模型；2) `quantize`；3) 一次校准；4) 浮点参考；5) `model(x)`；6) 断言不 allclose |

### 9.4 TI-03 OfmrQuant 校准完成后 fake quant 可执行（HIFLOAT8）

验证目标：`OfmrQuant` 在 HIFLOAT8×HIFLOAT8 配置下校准完成后 `fake_quant_forward` 可正常执行并产出非空输出。HIFLOAT8 路径在 mock 环境下接近 identity，故不断言"输出 ≠ 浮点"，仅断言路径接通。

| 用例名称（pytest） | 输入 | 预期结果 | 类型 | 操作步骤 |
|---|---|---|---|---|
| TestOfmrFakeQuant::test_hif8_hif8_post_calibration_runs | HIFLOAT8 channel + HIFLOAT8 per-tensor，ofmr 算法 | 类型为 `OfmrQuant`，post-calibration forward 返回非空 | 功能 | 1) `quantize`；2) 一次校准；3) `out = model(x)`；4) 断言 `type(model.linear1).__name__ == 'OfmrQuant'`；5) 断言 `out is not None` |

### 9.5 TI-04 LinearAWQuant 校准完成后 forward 引入量化误差

验证目标：AWQ 模块在仅权重 INT8 配置下校准完成后，forward 输出 ≠ 原始浮点输出。AWQ 通过 `inputs * self.scale` 折叠 awq pre-scale + `quant_dequant_tensor(ori_module.weight)`，二者共同导致输出与浮点结果不等。

| 用例名称（pytest） | 输入 | 预期结果 | 类型 | 操作步骤 |
|---|---|---|---|---|
| TestAWQFakeQuant::test_awq_int8_diverges_from_fp | 仅权重 INT8 channel + awq(grids_num=20, duo_scaling=False) | 校准完成后 forward 输出 ≠ 浮点 | 功能 | 1) `quantize`；2) 一次 forward 触发 calibration_done；3) 浮点参考；4) 再次 `model(x)`；5) 断言不 allclose |

### 9.6 TI-05 GPTQuant 校准完成后 forward 引入量化误差

验证目标：GPTQ 模块在仅权重 INT8 配置下，calibration（含 hessian 累计 + GPTQ 优化权重写回）完成后，forward 走 `fake_quant_forward` 并输出 ≠ 浮点。

| 用例名称（pytest） | 输入 | 预期结果 | 类型 | 操作步骤 |
|---|---|---|---|---|
| TestGPTQFakeQuant::test_gptq_int8_diverges_from_fp | 仅权重 INT8 channel + gptq | 校准完成后 forward 输出 ≠ 浮点 | 功能 | 1) `quantize`；2) `model(x)` 累计 hessian；3) `model(x)` 触发 GPTQ 优化；4) 浮点参考；5) 再次 `model(x)`；6) 断言不 allclose |

### 9.7 TI-06 QuantileQuant 三条 fake quant 分支端到端可用

验证目标：`QuantileQuant.fake_quant_forward` 内部三条分支——`weight_compress_only`、`dynamic is True`（实时算 scale_d）、静态激活量化——都能在端到端流程下被正确触发并产出非空输出。HIFLOAT8 路径在 mock 环境下不便于做"输出 ≠ 浮点"的断言，故仅断言路径接通。

| 用例名称（pytest） | 输入 | 预期结果 | 类型 | 操作步骤 |
|---|---|---|---|---|
| TestQuantileFakeQuant::test_static_hif8_post_calibration_runs | HIFLOAT8 channel + HIFLOAT8 per-tensor 静态量化 | 校准完成后 forward 返回非空，类型为 `QuantileQuant` | 功能 | 1) `quantize`；2) 一次校准；3) `out = model(x)`；4) 断言类型与非空 |
| TestQuantileFakeQuant::test_dynamic_hif8_post_calibration_runs | HIFLOAT8 tensor + HIFLOAT8 token + dynamic=True | 首次 forward 走 dynamic 分支（实时计算 scale_d），返回非空 | 功能 | 1) `quantize`；2) 直接 `out = model(x)`；3) 断言类型与非空 |
| TestQuantileFakeQuant::test_weight_only_hif8_post_calibration_runs | 仅权重 HIFLOAT8 channel | 首次 forward 走 weight-only 分支，返回非空 | 功能 | 1) `quantize`；2) 直接 `out = model(x)`；3) 断言类型与非空 |

---

## 10. 附：变更文件清单

| 文件 | 变更 |
|---|---|
| amct_pytorch/common/utils/vars.py | 新增 INT32_MAX / INT32_MIN |
| amct_pytorch/classic/quantize_op/utils.py | 新增 apply_progressive_quant_dequant |
| amct_pytorch/classic/quantize_op/{minmax,smooth_quant,ofmr_quant,linear_awq,gptq,quantile}_module.py | 新增 fake_quant_forward；forward 校准完成后路由到该方法 |
| amct_pytorch/classic/deploy_op/npu_quantization_linear.py | 改写 _init_bias（FP8/HIFLOAT8 路径 fp 域加 offset_bias 后早 return）；新增 bias_for_npu_op；forward 按标志派发 bias |
| tests/amct_pytorch/test_fake_quant_forward.py | 新增 10 项端到端测试 |
