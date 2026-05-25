# Fake Quant Forward 与 Bias 量化 设计文档

| 项 | 内容 |
|---|---|
| 特性名称 | Fake quant forward 与 bias 量化 |
| 关联分支 | feature/fake-quant-forward |
| 关联 MR | annqr/amct_open!（fake-quant forward + bias 量化） |
| 设计依据 | docs/api/quantize.md「全量化支持的层」 |

---

## 1. 简述方案

本特性面向训练后量化（PTQ）的精度评测场景，提供两项能力：

1. **Fake quant forward**：校准模块在完成 `batch_num` 轮校准后，其 `forward()` 不再返回原始浮点结果，而是用校准得到的量化因子对权重、激活（及 bias）做 quant→dequant，从而在 CPU/GPU 上近似量化引入的数值误差，无需经过 `convert()` 与 NPU 硬件即可评估量化后精度。

2. **Bias 量化**：在 fake quant 评测路径与 deploy 部署路径两端，按数据类型组合对 bias 施加量化。fake quant 侧的目标并非逐位复刻 NPU 的真实 bias 加法实现，而是在已有的 weight/activation fake quant 之外，让 bias 也经过相应的量化处理，从而使 bias 同样反映量化影响；deploy 侧则负责产出 NPU 算子期望的 bias 形态。按组合区分：INT8 per-tensor 路径将 bias 量化为 INT32（fake quant 用 quant→dequant 模拟，deploy 产出 INT32 buffer）；FP8/HIFLOAT8 对称组合保持浮点 bias；fp8*fp4 组合不支持 bias。

两项能力共享同一套 `BaseQuantizeModule` 基础设施，bias 量化是 fake quant forward 在 bias 维度上的补全。

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

- 校准模块：`MinMaxQuant`、`SmoothQuant`、`OfmrQuant`、`LinearAWQuant`（均继承 `BaseQuantizeModule`）。
- 部署模块：`NpuQuantizationLinear`。
- 算子类型：`torch.nn.Linear`（全部组合）、`torch.nn.Conv2d`（仅 OfmrQuant，且全量化组合仅 FP8×FP8 / HIFLOAT8×HIFLOAT8）。

### 2.3 数学规格

INT8 per-tensor 路径，记 `deq_scale = scale_d × scale_w`：

- Fake quant bias：`bias_q = round(bias / deq_scale)`，截断到 INT32 区间后乘回 `deq_scale` 得到浮点 bias，传入 `F.linear` / `F.conv2d`。该 quant→dequant 让 bias 也带上 INT8 量化误差，与 weight/activation 的 fake quant 方式一致；目的是在评测中体现 bias 量化的影响，并非逐位复刻 NPU 的整数 bias 加法实现。
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

- `_init_bias` 产出注册到 buffer 的 `self.bias`（INT32 或浮点或 None）与布尔属性 `self._bias_for_npu_op`。
- `forward` 输出经 `npu_quant_matmul` 计算并按 `_bias_for_npu_op` 决定 bias 加法位置后的张量，dtype 为 `output_dtype`。

---

## 5. 限制

1. **算子覆盖**：bias 量化仅在 `NpuQuantizationLinear` 范围内；MXFP8 走 `NpuMXQuantizationLinear`、仅权重量化走 `NpuWeightQuantizedLinear`，均不在本特性范围。
2. **fp8*fp4 bias**：fp8*fp4 组合不支持带 bias 的 Linear，bias 由上游保证为 None，代码不做额外防御。
3. **Conv2d 全量化**：仅 FP8×FP8 / HIFLOAT8×HIFLOAT8，均走 bias 浮点外加路径。
4. **FP8/HIFLOAT8 量化算子依赖 NPU**：fake quant 中 FP8/HIFLOAT8 的 `quant_tensor` 调用 `torch_npu` 算子，纯 CPU 环境仅 INT8/INT4 路径可独立运行；deploy 路径整体依赖 NPU 硬件。
5. **缓存有效性**：`_fake_quant_forward` 的缓存假设校准完成后权重/scale/bias 不再变化；若调用方在首次 `_fake_quant_forward` 之后修改这些属性，需手动清除缓存标志（设计上不暴露失效接口）。
6. **INT32 累加溢出**：不显式模拟 NPU 整数累加器溢出截断（常规矩阵规模 K ≤ 16384 不触发）。

---

## 6. 模块设计

### 6.1 总体结构

```
BaseQuantizeModule (amct_pytorch/quantize_op/base_quant_module.py)
  ├─ _quantizable_weight()        权重预变换（SmoothQuant smooth_factor）
  ├─ _quantizable_input(inputs)   激活预变换
  ├─ _fake_quant_bias(deq_scale)  bias 的 quant→dequant 模拟
  ├─ _build_fake_quant_cache()    惰性缓存权重 dequant / deq_scale / bias
  └─ _fake_quant_forward(inputs)  组装并执行 fake quant 前向

子类（MinMaxQuant / SmoothQuant / OfmrQuant / LinearAWQuant）
  └─ forward()  校准完成后 return self._fake_quant_forward(...)
     LinearAWQuant 额外覆盖 _quantizable_weight()

amct_pytorch/quantize_op/utils.py
  └─ apply_progressive_quant_dequant()  fp8*fp4 两级 quant→dequant（per-channel FP8 → per-group FP4）

NpuQuantizationLinear (amct_pytorch/deploy_op/npu_quantization_linear.py)
  ├─ _init_bias(module)  决定 bias 形态并设置 _bias_for_npu_op
  └─ forward(x)          按 _bias_for_npu_op 派发 bias 加法位置

amct_pytorch/utils/vars.py
  └─ INT32_MAX / INT32_MIN 常量
```

### 6.2 关键设计点

- **职责单一**：`_fake_quant_bias` 只负责 bias 量化决策；`_build_fake_quant_cache` 只负责缓存构建；`_fake_quant_forward` 只负责组装前向。
- **缓存策略**：与 batch 无关的量（权重 dequant、deq_scale、量化后 bias）首次调用时计算一次并存为实例属性，后续 batch 仅重算激活 dequant，降低大权重矩阵的重复开销。
- **派发标志**：deploy 端 bias「算子内加 / 算子外加」的决策在 `_init_bias` 一次性确定并记录于 `_bias_for_npu_op`，`forward` 仅读取标志，避免两处条件判断不同步。
- **粒度即区分**：fake quant 端不显式判断 `dynamic`，因为 dynamic per-token 走 `act_granularity == 'token'`，已被 per-tensor 判断排除。

---

## 7. 异常与 DFX 设计

### 7.1 显式异常

| 触发点 | 条件 | 行为 |
|---|---|---|
| apply_progressive_quant_dequant | scale_w1 形状 ≠ [n] | 抛 RuntimeError，提示期望形状与实际形状 |
| apply_progressive_quant_dequant | scale_w2 形状 ≠ [n*k/g, 1] | 抛 RuntimeError，提示期望形状与实际形状 |

### 7.2 隐式保护与容错

- **bias 为 None**：`_fake_quant_bias` 返回 None，`F.linear` / `F.conv2d` 接受 `bias=None`，不报错。
- **scale 缺失**：`deq_scale` 为 None（仅权重量化或校准未完成）时，`_fake_quant_bias` 返回原始 bias，不做量化。
- **校准未完成**：`_fake_quant_forward` 仅在校准完成后被调用，调用时机由各子类 forward 控制（`cur_batch > batch_num` / `calc_done` / `weight_compress_only`）。
- **设备一致性**：`_quantizable_weight` / `_quantizable_input` 将 scale 迁移到张量所在 device 与 dtype，避免跨设备运算异常。

### 7.3 DFX（可定位/可维护）

- fp8*fp4 形状校验的 RuntimeError 携带期望值与实际值，便于定位 scale 配置错误。
- `_bias_for_npu_op` 作为单一信号源，使 bias 派发路径可在 `_init_bias` 处一次性审阅。
- 缓存属性命名带 `_cached_` 前缀（`_cached_dequant_w` / `_cached_deq_scale` / `_cached_bias_for_linear`）与就绪标志 `_fake_quant_cache_ready`，状态可读、易排查。

---

## 8. 核心函数接口

仅列接口签名与语义，不含实现细节。

### 8.1 BaseQuantizeModule

| 函数 | 签名 | 返回 | 语义 |
|---|---|---|---|
| _quantizable_weight | `_quantizable_weight(self)` | Tensor | 返回送入 quant_dequant 的权重；SmoothQuant 预乘 smooth_factor |
| _quantizable_input | `_quantizable_input(self, inputs)` | Tensor | 返回送入 quant_dequant 的激活；SmoothQuant 预除 smooth_factor |
| _fake_quant_bias | `_fake_quant_bias(self, deq_scale)` | Tensor / None | INT8 per-tensor 时返回 INT32 round-trip 后的浮点 bias，否则返回原始 bias 或 None |
| _build_fake_quant_cache | `_build_fake_quant_cache(self)` | None | 计算并缓存 `_cached_dequant_w` / `_cached_deq_scale` / `_cached_bias_for_linear` |
| _fake_quant_forward | `_fake_quant_forward(self, inputs)` | Tensor | 首次构建缓存；每 batch 重算激活 dequant；按 ori_module 类型派发 F.linear / F.conv2d |

### 8.2 quantize_op/utils.py

| 函数 | 签名 | 返回 | 语义 |
|---|---|---|---|
| apply_progressive_quant_dequant | `apply_progressive_quant_dequant(weight, scale_w1, scale_w2, group_size=32)` | Tensor | per-channel FP8 → per-group FP4 两级 quant→dequant，返回与 weight 同形浮点张量 |

### 8.3 NpuQuantizationLinear

| 函数 | 签名 | 返回 | 语义 |
|---|---|---|---|
| _init_bias | `_init_bias(self, module)` | None | 按数据类型组合确定 bias 形态（INT32 / 浮点 / None），注册到 buffer，并设置 `_bias_for_npu_op` |
| forward | `forward(self, x)` | Tensor | 量化激活后调用 npu_quant_matmul；`is_new_torch_npu` 分支按 `_bias_for_npu_op` 决定 bias 算子内传入或算子外相加 |

---

## 9. 测试用例设计

测试文件：`tests/amct_pytorch/test_fake_quant_forward.py`，共 28 个用例，依赖本地 `conftest.py`（注入项目根路径、mock graph_based_compression 与 transformers）。执行：`cd tests/amct_pytorch && python -m pytest test_fake_quant_forward.py -v`。

「测试项」指被验证的功能点（一个测试项对应一段需要被验证的能力）；每个测试项下有一个或多个「测试用例」，用例名称即实际的 pytest 标识符，便于按名执行与回溯。

### 9.1 测试项总览

| 测试项编号 | 测试项（被验证的功能点） | 用例数 | 用例类型 |
|---|---|---|---|
| TI-01 | bias 量化判定：`_fake_quant_bias` 是否仅在「INT8 + per-tensor」时把 bias 量化为 INT32，其余情况原样返回 | 5 | 功能 |
| TI-02 | 校准完成后 forward 施加 fake quant：输出等于「对权重(/激活/bias) 做 quant→dequant 后再 linear」的手工结果 | 7 | 功能 |
| TI-03 | 校准进行中保持浮点：`cur_batch ≤ batch_num` 时 forward 输出等于原始浮点，不引入量化误差 | 2 | 功能 |
| TI-04 | 权重预变换正确性：`_quantizable_weight` 对 SmoothQuant 返回 weight×smooth_factor、对 AWQ 返回 ori_module.weight | 2 | 功能 |
| TI-05 | fp8*fp4 两级量化：`apply_progressive_quant_dequant` 两级 quant→dequant 的派发与 scale 形状校验 | 4 | 功能 3 / 异常 1（含两条形状异常即共 2 条异常） |
| TI-06 | Conv2d 派发：`ori_module` 为 Conv2d 时 forward 走 `F.conv2d` 并传卷积参数 | 1 | 功能 |
| TI-07 | 缓存机制：与 batch 无关的量（权重 dequant 等）首次构建后缓存，后续 forward 不重复计算 | 2 | 功能 |
| TI-08 | bias 派发结果（端到端）：各 dtype/粒度组合下 `_build_fake_quant_cache` 产出的 bias 是被量化还是原样透传 | 5 | 功能 4 / 鲁棒性 1 |
| TI-09 | Deploy 端 bias 形态与派发（NPU CI）：`_init_bias` 的 bias 形态与 `_bias_for_npu_op`、forward 的 bias 加法位置 | 3（建议） | 功能 |

> 注：TI-05 共 4 个用例，其中 2 个为形状非法的异常用例、2 个为正常派发的功能用例。

### 9.2 TI-01 bias 量化判定（`_fake_quant_bias`）

验证目标：`_fake_quant_bias(deq_scale)` 只在「`act_type==INT8` 且 `act_granularity=='tensor'`」时把 bias 做 INT32 的 quant→dequant，其余条件一律原样返回（None 或原始 bias 对象）。

| 用例名称（pytest） | 输入 | 预期结果 | 类型 | 操作步骤 |
|---|---|---|---|---|
| TestFakeQuantBias::test_returns_none_when_bias_none | 模块 bias 置为 None，传入任意 deq_scale | 返回 None | 功能 | 1) 构造 INT8 per-tensor 模块；2) 置 mod.bias=None；3) 调用 `_fake_quant_bias`；4) 断言返回 None |
| TestFakeQuantBias::test_returns_raw_bias_when_deq_scale_none | deq_scale=None | 返回原始 bias（同一对象） | 功能 | 1) 构造模块；2) 调用 `_fake_quant_bias(None)`；3) 用 assertIs 断言返回的就是 mod.bias |
| TestFakeQuantBias::test_returns_raw_bias_for_per_token_granularity | act_granularity='token'，给定 deq_scale | 返回原始 bias | 功能 | 1) 构造模块；2) 改 act_granularity='token'；3) 调用；4) assertIs 断言原样返回 |
| TestFakeQuantBias::test_returns_raw_bias_for_non_int8_act | act_type='float8_e4m3fn'，给定 deq_scale | 返回原始 bias | 功能 | 1) 构造模块；2) 改 act_type 为非 INT8；3) 调用；4) assertIs 断言原样返回 |
| TestFakeQuantBias::test_int8_per_tensor_round_trip_through_int32 | INT8 per-tensor，给定具体 deq_scale 张量 | 返回 `round(bias/deq_scale).clamp(INT32)×deq_scale` | 功能 | 1) 构造 INT8 per-tensor 模块；2) 调用 `_fake_quant_bias`；3) 手工按公式算期望；4) allclose 断言相等 |

### 9.3 TI-02 校准完成后 forward 施加 fake quant

验证目标：各校准模块在校准完成后，`forward`/`_fake_quant_forward` 的输出等于「对权重（及激活、bias）手工做 quant→dequant 后再 `F.linear`」的结果。

| 用例名称（pytest） | 输入 | 预期结果 | 类型 | 操作步骤 |
|---|---|---|---|---|
| TestBaseQuantizeFakeQuant::test_fake_quant_forward_differs_from_fp | INT8 仅权重 MinMaxQuant，随机输入 x | fake quant 输出与原始浮点输出不相等 | 功能 | 1) 构造模块；2) 调用 `_fake_quant_forward(x)`；3) 与 linear(x) 浮点输出比较；4) 断言不 allclose |
| TestBaseQuantizeFakeQuant::test_fake_quant_forward_matches_manual_quant_dequant | 同上 | 输出等于「quant_dequant 权重后 F.linear(x)」 | 功能 | 1) 调用 `_fake_quant_forward(x)`；2) 手工 quant_dequant 权重并 F.linear；3) allclose 断言相等 |
| TestMinMaxFakeQuant::test_weight_compress_only_applies_fake_quant_immediately | INT8 仅权重 MinMaxQuant | 输出等于「quant_dequant 权重后 F.linear」 | 功能 | 1) 构造仅权重模块；2) forward(x)；3) 手工算期望；4) allclose 断言 |
| TestMinMaxFakeQuant::test_after_calibration_weight_act_fake_quant | INT8 权重+激活，跑到 cur_batch>batch_num | 输出等于「dq_x、dq_w 与 INT32 round-trip bias 的 F.linear」 | 功能 | 1) 多次 forward 触发校准完成；2) 再 forward(x) 取输出；3) 手工算 dq_x/dq_w 与量化后 bias 的期望；4) allclose 断言 |
| TestSmoothQuantFakeQuant::test_after_calibration_uses_smooth_factor | INT8 + smoothquant，校准完成 | 输出等于「weight×smooth、x/smooth、INT32 bias 的 F.linear」 | 功能 | 1) 跑完校准；2) forward(x)；3) 用 smooth_factor 手工算期望；4) allclose 断言 |
| TestOfmrFakeQuant::test_after_calibration_applies_fake_quant | INT8 OfmrQuant，cur_batch>batch_num | 输出等于「quant_dequant 权重后 F.linear」 | 功能 | 1) 跑完校准；2) forward(x)；3) 手工算期望；4) allclose 断言 |
| TestAWQFakeQuant::test_after_calibration_output_matches_manual | AWQ 仅权重，校准完成 | 输出等于「quant_dequant(ori_module.weight) 后 F.linear」 | 功能 | 1) 跑完校准；2) forward(x)；3) 用 ori_module.weight 手工算期望；4) allclose 断言 |

### 9.4 TI-03 校准进行中保持浮点

验证目标：校准未完成（`cur_batch ≤ batch_num`）时，forward 仍返回原始浮点结果，不引入量化误差。

| 用例名称（pytest） | 输入 | 预期结果 | 类型 | 操作步骤 |
|---|---|---|---|---|
| TestMinMaxFakeQuant::test_during_calibration_output_equals_fp | INT8 权重+激活，cur_batch ≤ batch_num | 输出等于原始浮点参考 | 功能 | 1) 构造模块；2) 单次 forward(x)；3) 与 linear(x) 比较；4) allclose 断言相等 |
| TestOfmrFakeQuant::test_during_calibration_output_equals_fp | INT8 OfmrQuant，cur_batch==batch_num（校准当批） | 输出等于原始浮点参考 | 功能 | 1) 构造模块；2) 单次 forward(x)；3) 与浮点比较；4) allclose 断言相等 |

### 9.5 TI-04 权重预变换正确性（`_quantizable_weight`）

验证目标：`_quantizable_weight` 对 SmoothQuant 返回 weight×smooth_factor，对 AWQ 返回 AWQ 缩放后的 ori_module.weight。

| 用例名称（pytest） | 输入 | 预期结果 | 类型 | 操作步骤 |
|---|---|---|---|---|
| TestSmoothQuantFakeQuant::test_quantizable_weight_applies_smooth_factor | SmoothQuant，校准完成（已得 smooth_factor） | `_quantizable_weight()` 返回 weight×smooth_factor | 功能 | 1) 跑完校准得 mod.scale=smooth_factor；2) 调用 `_quantizable_weight`；3) 与 weight×smooth 比较；4) allclose 断言 |
| TestAWQFakeQuant::test_quantizable_weight_override_returns_ori_module_weight | LinearAWQuant，校准完成 | `_quantizable_weight()` 返回 ori_module.weight.data（AWQ 缩放后） | 功能 | 1) 跑完校准；2) 调用 `_quantizable_weight`；3) 与 ori_module.weight.data 比较；4) assertIs/allclose 断言 |

### 9.6 TI-05 fp8*fp4 两级量化（`apply_progressive_quant_dequant`）

验证目标：fp8*fp4 的两级 quant→dequant（per-channel FP8 → per-group FP4）派发顺序与形状正确，且对非法 scale 形状抛出明确异常。

| 用例名称（pytest） | 输入 | 预期结果 | 类型 | 操作步骤 |
|---|---|---|---|---|
| TestApplyProgressiveQuantDequant::test_dispatches_to_quant_dequant_tensor_twice | weight[8,64]、scale_w1[8]、scale_w2[16,1]，mock quant_dequant_tensor | 调用两次：先 FP8（weight 转置形状）后 FP4（[16,32] 形状），输出与 weight 同形 | 功能 | 1) mock quant_dequant_tensor 记录调用；2) 调用 `apply_progressive_quant_dequant`；3) 断言调用 2 次且各次 dtype/形状正确；4) 断言输出形状 |
| TestApplyProgressiveQuantDequant::test_rejects_wrong_scale_w1_shape | scale_w1 形状不等于 [n] | 抛 RuntimeError | 异常 | 1) 传入错误形状的 scale_w1；2) 断言抛 RuntimeError |
| TestApplyProgressiveQuantDequant::test_rejects_wrong_scale_w2_shape | scale_w2 形状不等于 [n*k/g, 1] | 抛 RuntimeError | 异常 | 1) 传入错误形状的 scale_w2；2) 断言抛 RuntimeError |
| TestFakeQuantForwardProgressiveDispatch::test_progressive_branch_calls_progressive_helper | scale_w=None 且注入 scale_w1/w2，mock helper 返回哨兵权重 | `_fake_quant_forward` 调用 `apply_progressive_quant_dequant`，输出等于 F.linear(x, 哨兵权重) | 功能 | 1) 注入 fp8*fp4 的 scale；2) mock helper；3) `_fake_quant_forward(x)`；4) 断言调用 helper 且输出匹配 |

### 9.7 TI-06 Conv2d 派发

验证目标：`ori_module` 为 `nn.Conv2d` 时，`_fake_quant_forward` 走 `F.conv2d` 并正确传入 stride/padding/dilation/groups。

| 用例名称（pytest） | 输入 | 预期结果 | 类型 | 操作步骤 |
|---|---|---|---|---|
| TestFakeQuantForwardConv2dDispatch::test_conv2d_dispatch_uses_F_conv2d_with_conv_attrs | OfmrQuant 包装的 Conv2d，mock quant_dequant_tensor | 走 F.conv2d，传入 conv 的 stride/padding/dilation/groups，输出形状与直接 F.conv2d 一致 | 功能 | 1) 构造 Conv2d 的 OfmrQuant 模块；2) mock quant_dequant_tensor；3) `_fake_quant_forward(x)`；4) 与手工 F.conv2d 输出比较形状/数值 |

### 9.8 TI-07 缓存机制（`_build_fake_quant_cache`）

验证目标：与 batch 无关的量（权重 dequant、fp8*fp4 两级 dequant）首次构建后缓存，后续多次 forward 不再重复计算。

| 用例名称（pytest） | 输入 | 预期结果 | 类型 | 操作步骤 |
|---|---|---|---|---|
| TestFakeQuantCache::test_weight_dequant_computed_once_across_calls | INT8 仅权重，连续 3 次 forward，用 wraps 方式 mock quant_dequant_tensor 计数 | 权重 dequant 仅被调用 1 次（仅权重路径下激活 dequant 0 次） | 功能 | 1) wraps-mock quant_dequant_tensor；2) 连续 3 次 `_fake_quant_forward`；3) 断言 call_count==1 |
| TestFakeQuantCache::test_progressive_dequant_computed_once_across_calls | 注入 scale_w1/w2，连续 2 次 forward，mock helper | `apply_progressive_quant_dequant` 仅被调用 1 次 | 功能 | 1) 注入 fp8*fp4 scale；2) mock helper；3) 连续 2 次 forward；4) 断言 call_count==1 |

### 9.9 TI-08 bias 派发结果（端到端）

验证目标：经 `_build_fake_quant_cache` 后，各 dtype/粒度组合下缓存的 bias 是被量化（INT8 per-tensor）还是原样透传（其余），无 bias 时不报错。

| 用例名称（pytest） | 输入 | 预期结果 | 类型 | 操作步骤 |
|---|---|---|---|---|
| TestFakeQuantBiasDispatch::test_fp8_per_tensor_keeps_bias_fp | act_type='float8_e4m3fn'，per-tensor | 缓存 bias 为原始 bias（同一对象） | 功能 | 1) 构造模块并改 act_type=FP8；2) `_build_fake_quant_cache`；3) assertIs 断言缓存 bias 即原始 bias |
| TestFakeQuantBiasDispatch::test_hifloat8_per_tensor_keeps_bias_fp | act_type='hifloat8'，per-tensor | 缓存 bias 为原始 bias | 功能 | 1) 改 act_type=HIFLOAT8；2) 构建缓存；3) assertIs 断言 |
| TestFakeQuantBiasDispatch::test_int8_per_token_skips_bias_quantization | INT8，per-token | 缓存 bias 为原始 bias（不量化） | 功能 | 1) 构造 INT8 per-token 模块；2) 构建缓存；3) assertIs 断言 |
| TestFakeQuantBiasDispatch::test_none_bias_path_does_not_error | 构造无 bias 模块（bias=False） | `_fake_quant_forward` 正常输出形状 (2,4)，不报错 | 鲁棒性 | 1) 构造无 bias 模块；2) `_fake_quant_forward(x)`；3) 断言输出形状且无异常 |
| TestFakeQuantBiasDispatch::test_progressive_path_passes_bias_unchanged | 无 bias，注入 scale_w1/w2，mock helper | 输出形状 (2,4)，缓存 bias 为 None | 功能 | 1) 注入 fp8*fp4 scale；2) mock helper；3) forward；4) 断言输出形状与缓存 bias 为 None |

### 9.10 TI-09 Deploy 端 bias 形态与派发（NPU CI）

`NpuQuantizationLinear._init_bias` 与 forward 的 bias 派发依赖 NPU 硬件，开发环境无 NPU，相关用例在 cann CI（NPU 环境）执行，本仓库 fake quant 测试不覆盖。建议 CI 覆盖项：

| 用例（建议） | 输入 | 预期结果 | 类型 | 操作步骤 |
|---|---|---|---|---|
| INT8 per-tensor bias 量化为 INT32 并算子内相加 | INT8 per-tensor 量化模块 | self.bias dtype 为 int32，_bias_for_npu_op 为 True，forward 走算子内 bias | 功能 | 1) 构造 INT8 per-tensor 部署模块；2) 检查 self.bias.dtype 与 _bias_for_npu_op；3) forward 验证走 bias= 参数 |
| FP8×FP8 bias 浮点算子外相加 | FP8×FP8 量化模块 | self.bias 为浮点，_bias_for_npu_op 为 False，forward 走算子外相加 | 功能 | 1) 构造 FP8×FP8 部署模块；2) 检查 bias 为浮点、_bias_for_npu_op 为 False；3) forward 验证算子外相加 |
| 无 bias + offset_bias | 非对称量化、原模块无 bias | self.bias 取 offset_bias，_bias_for_npu_op 为 True | 功能 | 1) 构造非对称、无 bias 模块；2) 检查 self.bias 即 offset_bias、_bias_for_npu_op 为 True |

---

## 10. 附：变更文件清单

| 文件 | 变更 |
|---|---|
| amct_pytorch/utils/vars.py | 新增 INT32_MAX / INT32_MIN |
| amct_pytorch/quantize_op/base_quant_module.py | 新增 _fake_quant_bias / _build_fake_quant_cache；改写 _fake_quant_forward |
| amct_pytorch/quantize_op/utils.py | 新增 apply_progressive_quant_dequant |
| amct_pytorch/quantize_op/{minmax,smooth_quant,ofmr_quant,linear_awq}_module.py | forward 校准后改为调用 _fake_quant_forward；LinearAWQuant 覆盖 _quantizable_weight |
| amct_pytorch/deploy_op/npu_quantization_linear.py | 改写 _init_bias，新增 _bias_for_npu_op；forward 按标志派发 bias |
| tests/amct_pytorch/test_fake_quant_forward.py | 新增 28 项测试 |
