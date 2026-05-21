# Bias 量化设计

> **状态**: spec draft, awaiting implementation
> **依赖**: 已合入的 fake quant forward 功能（branch feature/fake-quant-forward）

## 1. 背景与目标

`amct_pytorch` 的 fake quant forward 已支持 weight + activation 的 quant→dequant 模拟，但 bias 一直保持原 fp 形式。NPU 实际推理时 bias 在不同 dtype 路径下有不同处理：

- **INT8 PER_TENSOR**：bias 被预除以 `deq_scale = scale_d × scale_w` 并 round 到 INT32，由 NPU op 内部加在整数累加器上
- **FP8×FP8 / HIFLOAT8×HIFLOAT8 PER_TENSOR**：bias 保持 fp，在 NPU op 之外做 fp 加（cann/master 现状）

要让 CPU/GPU 评测与 NPU 真实推理对齐，fake quant 端必须模拟 bias 经过的量化误差；deploy 端的 `_init_bias` 也必须按真实 NPU op 期望的 bias 形式产出。两端的量化逻辑严格对齐。

## 2. 范围与约束

### 2.1 涉及的模块

| 模块 | 角色 |
|---|---|
| `amct_pytorch/quantize_op/base_quant_module.py` | Fake quant `_fake_quant_forward` / `_fake_quant_bias` / `_build_fake_quant_cache` |
| `amct_pytorch/deploy_op/npu_quantization_linear.py` | Deploy `_init_bias` + forward 分支 |
| `amct_pytorch/utils/vars.py` | 引入 `INT32_MIN` / `INT32_MAX`（若未存在则新增） |
| `tests/amct_pytorch/test_fake_quant_forward.py` | 新增 bias 量化与缓存测试 |

### 2.2 数据类型组合范围

依据 `docs/api/quantize.md`「全量化支持的层」章节，`NpuQuantizationLinear` 涉及：

| 组合 | granularity 约束 | 算法 | bias |
|---|---|---|---|
| INT8 + INT8 | PER_TENSOR (sym/asym) 或 PER_TOKEN (sym) | minmax / smoothquant | PER_TENSOR 量化为 INT32；PER_TOKEN 不量化 |
| FP8 + FP8 (FLOAT8_E4M3FN×FLOAT8_E4M3FN) | act 仅 PER_TENSOR，wts PER_TENSOR/PER_CHANNEL | ofmr | 不传 NPU op，op 外 fp 加 |
| HIFLOAT8 + HIFLOAT8 | act 仅 PER_TENSOR，wts PER_TENSOR/PER_CHANNEL | ofmr / quantile | 不传 NPU op，op 外 fp 加 |
| FP8 + FP4 (progressive) | act PER_TENSOR + wts PER_GROUP | minmax / smoothquant | bias=False（不支持） |

**不在范围内**：
- MXFP8+MXFP8（走 `NpuMXQuantizationLinear`）
- Weight-only 量化（走 `NpuWeightQuantizedLinear`）
- INT4 act：文档明确 INT4 仅出现在 weight-only 场景

### 2.3 完整触发矩阵

| 路径 | Fake quant `_fake_quant_bias` | Deploy bias 形式 | `_bias_for_npu_op` | Forward bias 加法位置 |
|---|---|---|---|---|
| INT8 + PER_TENSOR + 非 dynamic | `(bias/deq_scale).round().clamp(INT32) × deq_scale` | INT32 buffer | True | NPU op 内部 (`bias=` 参数) |
| INT8 + PER_TOKEN | 不量化（原 fp） | 原 fp | True | NPU op 内部 |
| INT8 + dynamic | 不量化 | 原 fp | True | NPU op 内部 |
| FP8 + FP8 PER_TENSOR | 不量化 | 原 fp | **False** | op 外 fp 加 |
| HIFLOAT8 + HIFLOAT8 PER_TENSOR | 不量化 | 原 fp | **False** | op 外 fp 加 |
| FP8 + FP4 progressive | n/a（约束下 bias=None） | None / offset_bias | True iff offset_bias 存在 | 视情况 |
| `module.bias is None` | None | None / offset_bias | True iff offset_bias 存在 | 视情况 |

## 3. Fake quant 端：`base_quant_module.py`

### 3.1 引入

```python
import torch
import torch.nn as nn
from amct_pytorch.utils.vars import INT8, INT32_MIN, INT32_MAX
```

### 3.2 辅助方法 `_quantizable_weight` / `_quantizable_input`（已有，保留）

```python
def _quantizable_weight(self):
    """Weight tensor to pass into quant_dequant_tensor.
    SmoothQuant pre-multiplies by self.scale (smooth_factor) so the
    fake-quant matches scale_w, which was computed from weight * smooth_factor.
    """
    if self.scale is not None:
        return self.weight.data * self.scale.to(device=self.weight.device,
                                                dtype=self.weight.dtype)
    return self.weight.data


def _quantizable_input(self, inputs):
    """Activation to pass into quant_dequant_tensor.
    SmoothQuant divides by self.scale (smooth_factor) so it matches scale_d.
    """
    if self.scale is not None:
        return inputs / self.scale.to(device=inputs.device, dtype=inputs.dtype)
    return inputs
```

### 3.3 新增 `_fake_quant_bias`

```python
def _fake_quant_bias(self, deq_scale):
    """Simulate NPU integer bias addition by quant→dequant.
    Only INT8 + per-tensor activation path quantizes bias to INT32 (matching
    the NPU integer accumulator); all other paths return the raw fp bias.
    """
    if self.bias is None:
        return None
    if (deq_scale is None
            or self.act_type != INT8
            or getattr(self, 'act_granularity', None) != 'tensor'):
        return self.bias
    bias_q = (self.bias.data / deq_scale).round().clamp(INT32_MIN, INT32_MAX)
    return bias_q * deq_scale
```

### 3.4 新增 `_build_fake_quant_cache`

```python
def _build_fake_quant_cache(self):
    """Compute weight dequant tensor, deq_scale and quantized bias once.
    These derive from calibration outputs (scale_w/scale_w1/scale_w2/scale_d/
    offset/bias) and do not change across batches once calibration finishes,
    so they are cached on the first _fake_quant_forward call.
    """
    from amct_pytorch.utils.quant_util import quant_dequant_tensor

    if self.scale_w is None and getattr(self, 'scale_w1', None) is not None \
            and getattr(self, 'scale_w2', None) is not None:
        from amct_pytorch.quantize_op.utils import apply_progressive_quant_dequant
        self._cached_dequant_w = apply_progressive_quant_dequant(
            self._quantizable_weight(), self.scale_w1, self.scale_w2)
        self._cached_deq_scale = None
    else:
        self._cached_dequant_w = quant_dequant_tensor(
            self._quantizable_weight(), self.wts_type,
            self.scale_w, self.offset_w, self.group_size)
        if self.scale_d is not None and self.scale_w is not None:
            ref_dtype = self.bias.dtype if self.bias is not None \
                else self._cached_dequant_w.dtype
            self._cached_deq_scale = (self.scale_d.to(ref_dtype)
                                      * self.scale_w.to(ref_dtype)).reshape(-1)
        else:
            self._cached_deq_scale = None

    self._cached_bias_for_linear = self._fake_quant_bias(self._cached_deq_scale)
```

### 3.5 改写 `_fake_quant_forward`

```python
@torch.no_grad()
def _fake_quant_forward(self, inputs):
    """Fake-quantize inputs and run linear/conv with the cached dequantized
    weight and bias. Activation dequant is recomputed per batch; everything
    else comes from the cache built on the first call.
    """
    import torch.nn.functional as F
    from amct_pytorch.utils.quant_util import quant_dequant_tensor

    if not getattr(self, '_fake_quant_cache_ready', False):
        self._build_fake_quant_cache()
        self._fake_quant_cache_ready = True

    if self.scale_d is not None:
        dequant_x = quant_dequant_tensor(
            self._quantizable_input(inputs), self.act_type,
            self.scale_d, self.offset_d)
    else:
        dequant_x = inputs

    if isinstance(getattr(self, 'ori_module', None), nn.Conv2d):
        conv = self.ori_module
        return F.conv2d(dequant_x, self._cached_dequant_w,
                        self._cached_bias_for_linear,
                        conv.stride, conv.padding, conv.dilation, conv.groups)
    return F.linear(dequant_x, self._cached_dequant_w,
                    self._cached_bias_for_linear)
```

### 3.6 缓存语义

| 缓存项 | 依赖 | 在 forward 间变化？ |
|---|---|---|
| `_cached_dequant_w` | `scale_w` / `scale_w1` / `scale_w2`、`offset_w`、`group_size`、weight | 否 |
| `_cached_deq_scale` | `scale_d × scale_w` | 否 |
| `_cached_bias_for_linear` | bias、deq_scale、`act_type`、`act_granularity` | 否 |

`_fake_quant_forward` 被调用的时机（`cur_batch > batch_num` / `calc_done` / `weight_compress_only`）即 calibration 完成。calibration 完成后所有依赖项固定，缓存语义正确。

## 4. 新增 `apply_progressive_quant_dequant`

`amct_pytorch/quantize_op/utils.py`：

```python
def apply_progressive_quant_dequant(weight, scale_w1, scale_w2, group_size=32):
    """Fake-quant counterpart of apply_progressive_quant: two-level quant+dequant
    (per-channel FP8 → per-group FP4) returning a float tensor of the same
    shape as `weight`, so calibration modules can evaluate CPU/GPU accuracy
    without the NPU pack op.

    Args:
        weight: [n, k] tensor
        scale_w1: [n] FP8 per-channel scale
        scale_w2: [n*k/g, 1] FP4 per-group scale
    """
    if list(scale_w1.shape) != [weight.shape[0]]:
        raise RuntimeError("scale_w1.shape should be [{}] current shape is {}"
                        .format(weight.shape[0], list(scale_w1.shape)))
    group = int(weight.shape[0] * weight.shape[1] / group_size)
    if list(scale_w2.shape) != [group, 1]:
        raise RuntimeError("scale_w2.shape should be [{}, 1] current shape is {}"
                        .format(group, list(scale_w2.shape)))

    dq_w_fp8 = quant_dequant_tensor(weight.transpose(-1, -2), FLOAT8_E4M3FN, scale_w1)
    dq_w_fp8 = dq_w_fp8.transpose(-1, -2).to(weight.dtype)
    dq_w_grouped = dq_w_fp8.reshape(-1, group_size)
    dq_w_fp4 = quant_dequant_tensor(dq_w_grouped, FLOAT4_E2M1, scale_w2)
    return dq_w_fp4.reshape(weight.shape).to(weight.dtype)
```

## 5. Deploy 端：`npu_quantization_linear.py`

### 5.1 改写 `_init_bias`

```python
def _init_bias(self, module):
    """Decide bias form and whether forward should pass it to npu_quant_matmul:
      - INT8 PER_TENSOR + non-dynamic: quantize bias to INT32, pass to op
      - FP8*FP8 / HIFLOAT8*HIFLOAT8 PER_TENSOR: keep fp, add outside op
        (preserves cann/master behavior)
      - INT8 PER_TOKEN / dynamic: keep fp, pass to op
      - module.bias is None: handle offset_bias if present
    Sets self._bias_for_npu_op accordingly.
    """
    self._bias_for_npu_op = False
    if module.bias is None:
        if self.offset_bias is None:
            self.bias = None
        else:
            self.bias = self.offset_bias
            self._bias_for_npu_op = True
        return

    is_fp8_fp8 = (self.act_type == FLOAT8_E4M3FN
                  and self.wts_type == FLOAT8_E4M3FN)
    is_hif8_hif8 = (self.act_type == HIFLOAT8
                    and self.wts_type == HIFLOAT8)

    if is_fp8_fp8 or is_hif8_hif8:
        self.register_buffer('bias', module.bias)
    elif (self.act_granularity == 'tensor'
            and self.dynamic is not True
            and self.act_type == INT8):
        deq_scale = module.scale_d * self.scale_w_tensor
        bias_tensor = (module.bias.data / deq_scale).round().to(torch.int32)
        self.register_buffer('bias', bias_tensor)
        self._bias_for_npu_op = True
    else:
        self.register_buffer('bias', module.bias)
        self._bias_for_npu_op = True

    if self.offset_bias is not None:
        self.bias = self.bias + self.offset_bias
        self._bias_for_npu_op = True
```

### 5.2 修改 `forward` 的 `is_new_torch_npu` 分支

```python
if self.is_new_torch_npu:
    if self._bias_for_npu_op and self.bias is not None:
        output = torch_npu.npu_quant_matmul(
            quant_x, self.quantized_weight,
            scale=self.deq_scale, pertoken_scale=self.pertoken_scale,
            bias=self.bias, output_dtype=self.output_dtype,
            x1_dtype=self.x1_dtype, x2_dtype=self.x2_dtype,
            group_sizes=self.group_sizes, y_scale=self.y_scale)
        if self.act_type == FLOAT8_E4M3FN and self.wts_type == FLOAT4_E2M1:
            output = output.reshape(*ori_shape[:-1], -1)
    else:
        output = torch_npu.npu_quant_matmul(
            quant_x, self.quantized_weight,
            scale=self.deq_scale, pertoken_scale=self.pertoken_scale,
            output_dtype=self.output_dtype,
            x1_dtype=self.x1_dtype, x2_dtype=self.x2_dtype,
            group_sizes=self.group_sizes, y_scale=self.y_scale)
        if self.act_type == FLOAT8_E4M3FN and self.wts_type == FLOAT4_E2M1:
            output = output.reshape(*ori_shape[:-1], -1)
        if self.bias is not None:
            output = output + self.bias
else:
    # 旧 NPU path 不变：始终把 bias 作为参数传给 NPU op
    output = torch_npu.npu_quant_matmul(quant_x, self.quantized_weight,
        scale=self.deq_scale, pertoken_scale=self.pertoken_scale,
        bias=self.bias, output_dtype=self.output_dtype)
```

### 5.3 `_bias_for_npu_op` flag 设计动机

- **"做决定一次、用决定多处"**：`_init_bias` 根据 dtype/granularity 一次性决定 bias 处理方式，forward 只读 flag 而非重复条件判断，避免两处条件同步出错
- **dtype 不足以区分**：fp dtype 既可能是 "`bias / deq_scale`"（INT8 path 之外的、传 NPU op 的 fp bias）也可能是 "原 fp"（FP8/HIFLOAT8 path 的 op 外加 bias），需要 flag 显式表达决策

## 6. 测试 (`tests/amct_pytorch/test_fake_quant_forward.py`)

新增以下测试：

| 测试 | 验证点 |
|---|---|
| `test_int8_per_tensor_quantizes_bias_int32_roundtrip` | INT8 PER_TENSOR：输出等于"bias 经 `(bias/deq_scale).round() × deq_scale` 后 F.linear" |
| `test_int8_per_token_skips_bias_quantization` | INT8 PER_TOKEN：bias 原样传给 F.linear |
| `test_fp8_per_tensor_keeps_bias_fp` | act/wts FP8：bias 透传不量化（mock `quant_dequant_tensor`，因 FP8 quant 需 NPU） |
| `test_hifloat8_per_tensor_keeps_bias_fp` | act/wts HIFLOAT8：同上 |
| `test_none_bias_path_does_not_error` | `module.bias=None`：F.linear/F.conv2d 收到 `bias=None` 不报错 |
| `test_progressive_path_passes_bias_unchanged` | Progressive 路径 bias 直接透传（验证 None 路径） |
| `test_weight_dequant_computed_once_across_calls` | mock `quant_dequant_tensor`，连续多次 `_fake_quant_forward`，weight 路径只调一次 |
| `test_progressive_dequant_computed_once_across_calls` | mock `apply_progressive_quant_dequant`，验证只调一次 |

Deploy 端不加新测试（开发环境无 NPU），由 cann/master CI 验证。

## 7. 文件改动汇总

| 文件 | 改动 |
|---|---|
| `amct_pytorch/quantize_op/base_quant_module.py` | 新增 `_fake_quant_bias` / `_build_fake_quant_cache`；改写 `_fake_quant_forward` 用缓存；新增 `INT8` / `INT32_MIN` / `INT32_MAX` 引入 |
| `amct_pytorch/quantize_op/utils.py` | 新增 `apply_progressive_quant_dequant`（已存在则确认行为符合） |
| `amct_pytorch/deploy_op/npu_quantization_linear.py` | 改写 `_init_bias`；新增 `_bias_for_npu_op` attribute；forward `is_new_torch_npu` 分支按 flag dispatch |
| `amct_pytorch/utils/vars.py` | 若 `INT32_MIN` / `INT32_MAX` 未存在则新增 |
| `tests/amct_pytorch/test_fake_quant_forward.py` | 新增 8 个测试用例 |

## 8. 不变约束与边界

| 约束 | 说明 |
|---|---|
| LinearAWQuant `_quantizable_weight` 覆盖不变 | bias 量化用 `self.scale_w`（AWQ 已正确设置） |
| Progressive FP8×FP4 不支持 bias | 上游保证 `module.bias is None`；fake quant / deploy 代码不做额外防御 |
| 旧 NPU path（`is_new_torch_npu=False`）保持现状 | 始终接受 `bias=` 参数，无需按 flag dispatch |
| INT4 act 不存在 | 文档明确 INT4 仅在 weight-only 场景；`act_type == INT4` 永远不会触发 bias 量化 |
| MXFP8 走 `NpuMXQuantizationLinear` | 不在本 design 范围 |
| Conv2d 全量化 dtype 范围 | 只有 FP8×FP8 / HIFLOAT8×HIFLOAT8，都走"bias 不量化 + op 外加"分支 |
| 缓存不主动 invalidate | calibration 完成后 weight/scale/bias 不会再变；若调用方在 `_fake_quant_forward` 之后修改这些 attribute（罕见），需手动清除 `_fake_quant_cache_ready`。YAGNI：不暴露 invalidate 接口 |

## 9. 数学等价性（INT8 PER_TENSOR）

NPU 真实推理：
```
matmul_int = quant_x_int @ quant_w_int.T            (整数累加)
out_int    = matmul_int + bias_int32                (INT32 累加)
out_fp     = out_int * deq_scale
```

Fake quant 实现：
```
dequant_x   = quant_x_int * scale_d
dequant_w   = quant_w_int * scale_w
bias_fp     = bias_int32 * deq_scale                (来自 _fake_quant_bias)
out_linear  = F.linear(dequant_x, dequant_w, bias_fp)
            = (quant_x_int @ quant_w_int.T) * deq_scale + bias_int32 * deq_scale
            = (matmul_int + bias_int32) * deq_scale
```

两者数学等价（fp 算术 ≤ 2 ulp 差异）。NPU INT32 累加器溢出 clamp 在常规 LLM/CV 矩阵规模下（K ≤ 16384）不触发，本设计不主动模拟该极端边缘行为。

## 10. 实施顺序建议

1. `amct_pytorch/utils/vars.py` 加 `INT32_MIN` / `INT32_MAX`（若需要）
2. `amct_pytorch/quantize_op/utils.py` 确认 `apply_progressive_quant_dequant` 已存在
3. `amct_pytorch/quantize_op/base_quant_module.py` 新增 `_fake_quant_bias` / `_build_fake_quant_cache` / 改写 `_fake_quant_forward`
4. `tests/amct_pytorch/test_fake_quant_forward.py` 新增 fake quant 端测试，确保全部通过
5. `amct_pytorch/deploy_op/npu_quantization_linear.py` 改写 `_init_bias` 与 forward 分支
6. Commit, push, force-with-lease
