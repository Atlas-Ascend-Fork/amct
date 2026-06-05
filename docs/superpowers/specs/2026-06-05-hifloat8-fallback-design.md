# HiFloat8 量化阶段 Fallback 设计

日期：2026-06-05
分支：quant_skills

## 1. 背景与目标

amct_pytorch 的量化（校准 / 伪量化）阶段在处理 `hifloat8` 数据类型时，依赖 `torch_npu` 提供的原生算子（`torch_npu.hifloat8` dtype、`npu_dtype_cast`、`npu_quantize`、`npu_dynamic_quant`）。在 `torch_npu` 不提供 hifloat8 能力的环境上，这些算法的伪量化路径直接不可用。

amct_ops 子项目提供了纯类型编解码算子（运行在昇腾 NPU 上）：

- `amct_ops.hifloat8_cast.encode_to_hifloat8(x)`：FP16/BF16 → uint8（HiF8 编码）
- `amct_ops.hifloat8_cast.decode_from_hifloat8(x, dtype)`：uint8 → FP16/BF16（HiF8 解码）

**目标**：当 `torch_npu` 不支持 hifloat8 时，**仅在量化阶段（quantize_op）**用「FP → HiF8 → FP」的伪量化往返（即 amct_ops 的 encode + decode）替代原生 cast，使 ofmr / gptq / quantile 等算法的伪量化路径在缺硬件原生支持的环境下也能跑通。

**非目标**：部署阶段（deploy_op）的 matmul、`npu_quantize` 真量化导出、KV-Cache 部署算子等不在本次范围内。

## 2. 范围分析

### 2.1 涉及 hifloat8 的量化算法

依据 `common/utils/vars.py` 的权威映射：

- `hifloat8 hifloat8`（W+A 量化）：`ofmr`、`cast`、`quantile`
- `NOT_QUANTIZE hifloat8`（weight-only）：`ofmr`、`gptq`、`cast`、`quantile`
- KV-Cache（`hifloat8`）：`quantile`（DeepseekV3Attention、LongcatFlashMLA）

### 2.2 量化阶段实际碰 `torch_npu.hifloat8` 的 cast 点

调用方分析（grep 全仓）确认：

- `quant_dequant_tensor`（伪量化往返，输出 FP）的调用方**全部**在 `quantize_op` / `algorithms`（awq、gptq、smoothquant、minmax、quantile、ofmr、linear_awq）——属量化阶段。
- `quant_tensor` / `quant_weight`（纯量化，输出真量化值）的 hifloat8 调用方**全部**在 `deploy_op`（weight_npu_quant_module、npu_quantization_linear、npu_quantization_conv2d、npu_hif8_cast_quantization_linear）——属部署阶段，不在范围内。`quantize_op/utils.py` 中对 `quant_tensor` 的调用只针对 FLOAT8_E4M3FN / FLOAT4_E2M1，不碰 hifloat8。
- `convert_dtype`（quant_util.py:108）的唯一调用方在 `deploy_op/npu_hif8_cast_quantization_linear.py:93`——属部署阶段，不在范围内。

**结论**：量化阶段真正需要改造的 hifloat8 cast 点只有两个：

| 改造点 | 文件:行 | 当前实现 |
|--------|---------|----------|
| A | `common/utils/quant_util.py` 的 `quant_dequant_tensor` hif8 分支（约 249/296-298）| 经 `quant_tensor` 的 `npu_quantize`(÷scale→hif8) + `npu_dtype_cast`(hif8→FP) + `*scale` |
| B | `classic/quantize_op/quantile_module.py` 动态 act（124-128）| `npu_dynamic_quant`(per-token, dst_type_max=15) + `npu_dtype_cast`(hif8→FP) + `*pertoken_scale` |

## 3. 方案选型

**方案 1（采用）**：在 `quant_util.py` 集中新增探测 helper 与伪量化 helper；改造点 B 在 Python 侧复现 per-token scale。改动集中、风险最低，torch_npu 支持时保持原生路径。

**方案 2（弃用）**：完全弃用 torch_npu 统一走 amct_ops——绕开硬件原生路径，性能/一致性受损，amct_ops 必须常驻。

**方案 3（弃用）**：monkey-patch `torch_npu`——全局副作用、难调试。

## 4. 详细设计

### 4.1 新增 helper（`common/utils/quant_util.py`）

探测沿用代码库既有规范——`common/utils/check_params.py` 的 `check_parameters_in_schema`，与 `npu_hif8_quantization_linear.py:39`、`npu_quantization_linear.py:185`、`weight_npu_quant_module.py:76` 保持一致（不自创 `hasattr` 探测）。

```python
import functools
from amct_pytorch.common.utils.check_params import check_parameters_in_schema

@functools.lru_cache(maxsize=1)
def _hifloat8_supported():
    """探测 torch_npu 是否提供 hifloat8 能力，进程内缓存一次。

    沿用代码库既有判据：npu_quant_matmul 的 schema 是否含 x1_dtype 参数
    （新版 torch_npu 才有，对应 hifloat8 量化 matmul 能力）。
    """
    try:
        import torch_npu
    except ImportError:
        return False
    return check_parameters_in_schema(torch_npu.npu_quant_matmul, 'x1_dtype')


@torch.no_grad()
def _hifloat8_fake_quant(fp_tensor):
    """FP → HiF8 → FP 伪量化往返。

    torch_npu 支持 hifloat8 时走原生 cast；否则走 amct_ops 的 encode/decode。
    amct_ops 仅支持 float16/bfloat16，float32 输入先降到 bfloat16 往返再还原回原 dtype。
    """
    if _hifloat8_supported():
        import torch_npu
        hif8 = torch_npu.npu_dtype_cast(fp_tensor, torch_npu.hifloat8)
        return torch_npu.npu_dtype_cast(hif8, fp_tensor.dtype, input_dtype=torch_npu.hifloat8)
    from amct_ops.hifloat8_cast import encode_to_hifloat8, decode_from_hifloat8
    work_dtype = fp_tensor.dtype if fp_tensor.dtype in (torch.float16, torch.bfloat16) else torch.bfloat16
    codes = encode_to_hifloat8(fp_tensor.to(work_dtype).npu())
    return decode_from_hifloat8(codes, work_dtype).to(fp_tensor.dtype)
```

设计要点：
- `_hifloat8_supported` 复用 `check_parameters_in_schema`，与现有 deploy_op 探测口径统一；用 `lru_cache` 缓存结果，避免每次 cast 重复反射 schema。
- `_hifloat8_fake_quant` 只负责「cast 往返」这一语义单元；scale 的乘除留在各调用点，职责清晰、可独立测试。
- float32：amct_ops 算子不接受 fp32（`ALLOWED_WEIGHT_DTYPES` 中 `hifloat8 hifloat8` 允许 fp32），故先降 bf16 往返再还原。这是 fallback 分支特有的精度近似，原生分支不受影响。

### 4.2 改造点 A：`quant_dequant_tensor` 的 hif8 分支（自包含）

`quant_dequant_tensor` 的 hifloat8 分支当前依赖 `quant_tensor`（内部 `npu_quantize` 完成 ÷scale 并 cast 到 hif8）+ 本函数的 `npu_dtype_cast`（hif8→FP）+ `*scale`。改造为**自包含**，fallback 时不再触碰 `npu_quantize`：

```python
# quant_dequant_tensor 内，hif8 分支改为：
if dst_dtype == HIFLOAT8:
    scaled = tensor / scale.to(tensor.device)          # 复现 npu_quantize 的 ÷scale
    dq = _hifloat8_fake_quant(scaled)                  # FP → HiF8 → FP 往返
    dequantize_tensor = dq.to(ori_dtype) * scale.to(ori_dtype)
    return dequantize_tensor.to(ori_dtype)
```

注意：原 `quant_dequant_tensor` 先调用 `quant_tensor`（其中含 hif8 的 `npu_quantize` 分支）再处理。改造后，hif8 分支在进入 `quant_tensor` 之前就被本分支拦截并自包含完成，因此 `quant_tensor` 内的 hif8 `npu_quantize` 分支在量化阶段不再被 `quant_dequant_tensor` 触发（它仍被 deploy_op 的 `quant_weight`/`quant_tensor` 调用，保持原样不动）。

**等价性论证**：原链路 = `npu_dtype_cast⁻¹(npu_quantize(t, scale, hif8)) * scale` = `decode(encode(t/scale)) * scale`。改造后 = `_hifloat8_fake_quant(t/scale) * scale`，在 torch_npu 支持分支下逐算子等价；在 amct_ops 分支下用等价的 encode/decode 实现同一数学语义。

### 4.3 改造点 B：`quantile_module.py` 动态 act（124-128）

`npu_dynamic_quant(inputs, dst_type=hif8, dst_type_max=15)` 做 per-token 动态量化：每 token 计算 `scale = amax(|x|)/15`，`x/scale` cast 到 hif8，返回 `(codes, pertoken_scale)`。改造为先探测，fallback 时 Python 侧复现：

```python
@torch.no_grad()
def fake_quant_forward(self, inputs):
    ...
    elif self.dynamic is True:
        from amct_pytorch.common.utils.quant_util import _hifloat8_supported, _hifloat8_fake_quant
        if _hifloat8_supported():
            import torch_npu
            quant_x, pertoken_scale = torch_npu.npu_dynamic_quant(
                inputs.npu(), dst_type=torch_npu.hifloat8, dst_type_max=15)
            quant_x_fp = torch_npu.npu_dtype_cast(quant_x, inputs.dtype, input_dtype=torch_npu.hifloat8)
            dq_x = quant_x_fp * pertoken_scale.unsqueeze(-1).to(quant_x_fp.dtype)
        else:
            amax = inputs.abs().amax(dim=-1, keepdim=True)
            pertoken_scale = (amax / 15).to(torch.float32)
            dq_x = _hifloat8_fake_quant(inputs / pertoken_scale).to(inputs.dtype) * pertoken_scale.to(inputs.dtype)
    ...
```

per-token 维度：`dst_type_max=15` 对应 HiF8 高精度范围上界（与 `calculate_hif8_scale` 的 `/16` 缩放约定一致，动态量化采用 15 作为上界）。scale 在最后一维（token 维）上规约。

## 5. 错误处理

- amct_ops 导入失败（既不支持 torch_npu hifloat8、又没装 amct_ops）：让 `ImportError` 自然抛出，错误信息已能指明缺少 amct_ops。不静默吞掉。
- amct_ops 算子要求输入在 NPU 上：helper 内 `.npu()` 已处理；输出 device 由 decode 决定，调用点的 `* scale` 会按 `to(tensor.device)` 对齐。

## 6. 测试策略

新增单测（沿用现有 `tests/ut` 布局与风格）：

1. **`_hifloat8_supported` 探测**：mock `torch_npu.npu_quant_matmul` 的 schema 含/不含 `x1_dtype`、mock `import torch_npu` 抛 `ImportError`，验证三种返回；并验证 `lru_cache` 只探测一次。
2. **`_hifloat8_fake_quant` 分支选择**：
   - mock `_hifloat8_supported=True`，断言走 `npu_dtype_cast` 往返。
   - mock `_hifloat8_supported=False` 且 mock amct_ops 的 encode/decode，断言被调用且 fp32 输入按 bf16 往返。
3. **改造点 A 等价性**：在 mock 下验证 `quant_dequant_tensor` hif8 分支的 `÷scale → fake_quant → ×scale` 调用次序与张量形状。
4. **改造点 B per-token scale**：构造已知输入，断言 fallback 分支 `pertoken_scale = amax/15` 计算正确、形状与原生路径一致。

真机 NPU 端到端验证（encode/decode 数值精度）依赖昇腾环境，在具备硬件的环境单独跑，不纳入 CI 单测。

## 7. 影响面

- 修改文件：`common/utils/quant_util.py`（新增 2 helper + 改 1 分支）、`classic/quantize_op/quantile_module.py`（改 1 分支）。
- 受益算法：ofmr、gptq、quantile、cast（其 weight 伪量化经 `quant_dequant_tensor`/`quant_dequant_weight`），以及 quantile 的动态 act。
- 不动：deploy_op 全部、`quant_tensor`/`quant_weight`/`convert_dtype` 的 hif8 路径、vars/config 常量。
- 向后兼容：torch_npu 支持 hifloat8 的环境行为完全不变（走原生分支）。
