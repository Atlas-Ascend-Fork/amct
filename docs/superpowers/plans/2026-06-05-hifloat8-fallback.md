# HiFloat8 量化阶段 Fallback 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 当 torch_npu 不支持 hifloat8 时，量化阶段（quantize_op）的 hifloat8 cast 改用 amct_ops 的 encode/decode 做「FP→HiF8→FP」伪量化往返。

**Architecture:** 在 `common/utils/quant_util.py` 新增两个 helper（`_hifloat8_supported` 复用 `check_parameters_in_schema` 探测、`_hifloat8_fake_quant` 做 cast 往返）；改造 `quant_dequant_tensor` 的 hif8 分支为自包含（÷scale→fake_quant→×scale），以及 `quantile_module.py` 动态 act 分支（fallback 时 Python 侧复现 per-token scale）。

**Tech Stack:** Python, PyTorch, torch_npu, amct_ops.hifloat8_cast, pytest。

**Spec:** `docs/superpowers/specs/2026-06-05-hifloat8-fallback-design.md`

---

## File Structure

- **Modify** `amct_pytorch/common/utils/quant_util.py`：新增 `_hifloat8_supported`、`_hifloat8_fake_quant`；改造 `quant_dequant_tensor` 的 HIFLOAT8 分支。
- **Modify** `amct_pytorch/classic/quantize_op/quantile_module.py`：改造 `fake_quant_forward` 的 `dynamic is True` 分支。
- **Modify** `tests/unit_test/common/utils/test_quant_util.py`：新增 helper 与 hif8 分支单测。
- **Create** `tests/unit_test/classic/quantize_op/test_quantile_fallback.py`：quantile 动态 act fallback 单测（若目录无 `__init__.py` 需按现有 unit_test 约定补齐）。

测试约束：单测在 CPU CI 跑，不能真调 torch_npu / amct_ops / `.npu()`。所有 NPU 路径通过 monkeypatch 替换。

---

## Task 1: 新增 `_hifloat8_supported` 探测 helper

**Files:**
- Modify: `amct_pytorch/common/utils/quant_util.py`
- Test: `tests/unit_test/common/utils/test_quant_util.py`

- [ ] **Step 1: Write the failing test**

在 `test_quant_util.py` 末尾追加：

```python
# ---- _hifloat8_supported / _hifloat8_fake_quant ---------------------------

import sys
from unittest import mock


def test_hifloat8_supported_true_when_schema_has_x1_dtype(monkeypatch):
    qu._hifloat8_supported.cache_clear()
    fake_npu = mock.MagicMock()
    monkeypatch.setitem(sys.modules, "torch_npu", fake_npu)
    monkeypatch.setattr(qu, "check_parameters_in_schema", lambda func, name: True)
    assert qu._hifloat8_supported() is True


def test_hifloat8_supported_false_when_schema_missing(monkeypatch):
    qu._hifloat8_supported.cache_clear()
    fake_npu = mock.MagicMock()
    monkeypatch.setitem(sys.modules, "torch_npu", fake_npu)
    monkeypatch.setattr(qu, "check_parameters_in_schema", lambda func, name: False)
    assert qu._hifloat8_supported() is False


def test_hifloat8_supported_false_when_torch_npu_missing(monkeypatch):
    qu._hifloat8_supported.cache_clear()
    monkeypatch.setitem(sys.modules, "torch_npu", None)  # import 触发 ImportError
    assert qu._hifloat8_supported() is False
```

注：`monkeypatch.setitem(sys.modules, "torch_npu", None)` 会让 `import torch_npu` 抛 `ImportError`，符合 helper 的 except 分支。

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit_test/common/utils/test_quant_util.py -k hifloat8_supported -v`
Expected: FAIL — `AttributeError: module ... has no attribute '_hifloat8_supported'`

- [ ] **Step 3: Write minimal implementation**

在 `quant_util.py` 顶部 import 区（现有 `import math` / `import torch` 之后）加：

```python
import functools

from amct_pytorch.common.utils.check_params import check_parameters_in_schema
```

并在文件中（建议紧跟 import、`FP4E2M1_MAX_EXP` 常量附近）新增：

```python
@functools.lru_cache(maxsize=1)
def _hifloat8_supported():
    """探测 torch_npu 是否提供 hifloat8 能力，进程内缓存一次。

    沿用代码库既有判据（与 deploy_op 一致）：npu_quant_matmul 的 schema
    是否含 x1_dtype 参数（新版 torch_npu 才有，对应 hifloat8 量化 matmul 能力）。
    """
    try:
        import torch_npu
    except ImportError:
        return False
    return check_parameters_in_schema(torch_npu.npu_quant_matmul, 'x1_dtype')
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit_test/common/utils/test_quant_util.py -k hifloat8_supported -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add amct_pytorch/common/utils/quant_util.py tests/unit_test/common/utils/test_quant_util.py
git commit -m "feat: add _hifloat8_supported detection helper in quant_util"
```

---

## Task 2: 新增 `_hifloat8_fake_quant` 往返 helper

**Files:**
- Modify: `amct_pytorch/common/utils/quant_util.py`
- Test: `tests/unit_test/common/utils/test_quant_util.py`

- [ ] **Step 1: Write the failing test**

在 `test_quant_util.py` 追加：

```python
def test_fake_quant_native_path_uses_npu_dtype_cast(monkeypatch):
    qu._hifloat8_supported.cache_clear()
    monkeypatch.setattr(qu, "_hifloat8_supported", lambda: True)

    calls = []
    fake_npu = mock.MagicMock()
    fake_npu.hifloat8 = "HIF8"

    def fake_cast(t, dtype, input_dtype=None):
        calls.append((dtype, input_dtype))
        return t  # 透传，验证调用次序

    fake_npu.npu_dtype_cast = fake_cast
    monkeypatch.setitem(sys.modules, "torch_npu", fake_npu)

    x = torch.randn(4, 8, dtype=torch.float16)
    out = qu._hifloat8_fake_quant(x)
    # 先 FP->HIF8，再 HIF8->FP
    assert calls[0] == ("HIF8", None)
    assert calls[1] == (torch.float16, "HIF8")
    assert out.shape == x.shape


def test_fake_quant_fallback_uses_amct_ops_and_bf16_for_fp32(monkeypatch):
    qu._hifloat8_supported.cache_clear()
    monkeypatch.setattr(qu, "_hifloat8_supported", lambda: False)

    seen = {}

    def fake_encode(t):
        seen["encode_dtype"] = t.dtype
        return t.to(torch.uint8)

    def fake_decode(codes, dtype):
        seen["decode_dtype"] = dtype
        return codes.to(dtype)

    fake_mod = mock.MagicMock()
    fake_mod.encode_to_hifloat8 = fake_encode
    fake_mod.decode_from_hifloat8 = fake_decode
    monkeypatch.setitem(sys.modules, "amct_ops", mock.MagicMock())
    monkeypatch.setitem(sys.modules, "amct_ops.hifloat8_cast", fake_mod)
    # CPU CI 上 .npu() 不可用，替换为 no-op
    monkeypatch.setattr(torch.Tensor, "npu", lambda self: self, raising=False)

    x = torch.randn(4, 8, dtype=torch.float32)
    out = qu._hifloat8_fake_quant(x)
    # fp32 输入应降到 bf16 往返
    assert seen["encode_dtype"] == torch.bfloat16
    assert seen["decode_dtype"] == torch.bfloat16
    assert out.dtype == torch.float32  # 还原回原 dtype
    assert out.shape == x.shape
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit_test/common/utils/test_quant_util.py -k fake_quant -v`
Expected: FAIL — `AttributeError: ... has no attribute '_hifloat8_fake_quant'`

- [ ] **Step 3: Write minimal implementation**

在 `quant_util.py` 的 `_hifloat8_supported` 之后新增：

```python
@torch.no_grad()
def _hifloat8_fake_quant(fp_tensor):
    """FP -> HiF8 -> FP 伪量化往返。

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

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit_test/common/utils/test_quant_util.py -k fake_quant -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add amct_pytorch/common/utils/quant_util.py tests/unit_test/common/utils/test_quant_util.py
git commit -m "feat: add _hifloat8_fake_quant round-trip helper in quant_util"
```

---

## Task 3: 改造 `quant_dequant_tensor` 的 HIFLOAT8 分支（自包含）

**Files:**
- Modify: `amct_pytorch/common/utils/quant_util.py`（`quant_dequant_tensor`，约 279-318 行）
- Test: `tests/unit_test/common/utils/test_quant_util.py`

背景：当前 `quant_dequant_tensor` 先调用 `quant_tensor`（其 hif8 分支用 `npu_quantize`），再在 296-298 行做 `npu_dtype_cast` + `*scale`。改造后让 HIFLOAT8 在进入 `quant_tensor` 之前被本函数拦截、自包含完成 `÷scale → _hifloat8_fake_quant → ×scale`，fallback 时不再依赖 `npu_quantize`。`quant_tensor` 自身的 hif8 分支保持不动（仍供 deploy_op 使用）。

- [ ] **Step 1: Write the failing test**

在 `test_quant_util.py` 追加：

```python
from amct_pytorch.common.utils.vars import HIFLOAT8


def test_quant_dequant_tensor_hifloat8_self_contained(monkeypatch):
    # hif8 分支应自包含：÷scale -> _hifloat8_fake_quant -> ×scale，
    # 不调用 quant_tensor 的 npu_quantize。
    captured = {}

    def fake_fake_quant(t):
        captured["scaled"] = t.clone()
        return t  # 透传，等价于无损 cast，便于断言数值

    monkeypatch.setattr(qu, "_hifloat8_fake_quant", fake_fake_quant)
    # 若 hif8 分支错误地走了 quant_tensor，会触发 import torch_npu 失败
    monkeypatch.setattr(qu, "quant_tensor",
                        mock.MagicMock(side_effect=AssertionError("should not call quant_tensor")))

    t = torch.tensor([[2.0, 4.0, 8.0, 16.0]])
    scale = torch.tensor([[2.0]])
    out = qu.quant_dequant_tensor(t, HIFLOAT8, scale=scale)

    # 透传 fake_quant 下：out = (t/scale)*scale = t
    assert torch.allclose(captured["scaled"], t / scale)
    assert torch.allclose(out, t)
    assert out.dtype == t.dtype
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit_test/common/utils/test_quant_util.py -k hifloat8_self_contained -v`
Expected: FAIL — 当前实现会调用 `quant_tensor`（被 mock 成 AssertionError），或走原 npu_quantize 路径。

- [ ] **Step 3: Write minimal implementation**

在 `quant_dequant_tensor` 函数体内，`quantized_tensor, shared_exponent = quant_tensor(...)` 这一行**之前**插入 HIFLOAT8 拦截分支：

```python
@torch.no_grad()
def quant_dequant_tensor(tensor, dst_dtype, scale=None, offset=None, group_size=None):
    """..."""
    ori_dtype = tensor.dtype
    ori_shape = tensor.shape

    if dst_dtype == HIFLOAT8:
        # 自包含 hif8 伪量化：÷scale -> FP->HiF8->FP 往返 -> ×scale。
        # 不依赖 quant_tensor 的 npu_quantize，torch_npu 缺 hif8 支持时走 amct_ops。
        scaled = tensor / scale.to(tensor.device)
        dq = _hifloat8_fake_quant(scaled)
        dequantize_tensor = dq.to(ori_dtype) * scale.to(ori_dtype).to(dq.device)
        return dequantize_tensor.to(ori_dtype)

    quantized_tensor, shared_exponent = quant_tensor(tensor, dst_dtype, scale, offset, group_size)
    ...
```

并删除原 296-298 行 `if dst_dtype == HIFLOAT8:` 那段（已被上面的早返回取代）。确认 `import torch_npu` 仍保留给其他分支（FLOAT8_E4M3FN/MXFP8 等）使用。

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit_test/common/utils/test_quant_util.py -k hifloat8_self_contained -v`
Expected: PASS

并跑回归确认未破坏其他 dtype：
Run: `pytest tests/unit_test/common/utils/test_quant_util.py -v`
Expected: 全部 PASS（含原有 int8/int4/mxfp4/fp4 用例）

- [ ] **Step 5: Commit**

```bash
git add amct_pytorch/common/utils/quant_util.py tests/unit_test/common/utils/test_quant_util.py
git commit -m "feat: make quant_dequant_tensor hifloat8 branch self-contained via fake-quant"
```

---

## Task 4: 改造 `quantile_module.py` 动态 act 分支

**Files:**
- Modify: `amct_pytorch/classic/quantize_op/quantile_module.py`（`fake_quant_forward`，123-128 行的 `elif self.dynamic is True:` 分支）
- Test: `tests/unit_test/classic/quantize_op/test_quantile_fallback.py`（新建）

- [ ] **Step 1: Write the failing test**

先确认 `tests/unit_test/classic/quantize_op/` 是否存在 `__init__.py`，无则按相邻 unit_test 目录约定补齐（多数 pytest 布局无需 `__init__.py`，遵循现有 `tests/unit_test/common/utils/` 是否有即可）。新建 `test_quantile_fallback.py`：

```python
#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
import sys
from unittest import mock

import torch

from amct_pytorch.common.utils import quant_util as qu


def _make_quantile_dynamic_act(monkeypatch):
    """构造一个只跑 fake_quant_forward dynamic 分支的最小 QuantileQuant 替身。"""
    from amct_pytorch.classic.quantize_op import quantile_module as qm

    inst = qm.QuantileQuant.__new__(qm.QuantileQuant)
    inst.weight_compress_only = False
    inst.dynamic = True
    inst.bias = None
    inst.cached_dq_w = torch.eye(8)
    inst.fake_quant_cache_ready = True
    return inst, qm


def test_quantile_dynamic_fallback_pertoken_scale(monkeypatch):
    qu._hifloat8_supported.cache_clear()
    monkeypatch.setattr(qu, "_hifloat8_supported", lambda: False)

    captured = {}

    def fake_fake_quant(t):
        captured["scaled"] = t.clone()
        return t  # 透传

    monkeypatch.setattr(qu, "_hifloat8_fake_quant", fake_fake_quant)

    inst, qm = _make_quantile_dynamic_act(monkeypatch)
    # weight 为单位阵，bias=None：F.linear(dq_x, I) == dq_x
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]])
    out = inst.fake_quant_forward(x)

    amax = x.abs().amax(dim=-1, keepdim=True)
    expected_scale = amax / 15
    # 透传 fake_quant：dq_x = (x/scale)*scale = x
    assert torch.allclose(captured["scaled"], x / expected_scale)
    assert torch.allclose(out, x, atol=1e-4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit_test/classic/quantize_op/test_quantile_fallback.py -v`
Expected: FAIL — 当前 dynamic 分支直接 `import torch_npu` 调 `npu_dynamic_quant`，在 CPU 上无 fallback，触发错误。

- [ ] **Step 3: Write minimal implementation**

把 `quantile_module.py` 的 `fake_quant_forward` 中 `elif self.dynamic is True:` 分支改为：

```python
        elif self.dynamic is True:
            from amct_pytorch.common.utils.quant_util import (
                _hifloat8_supported, _hifloat8_fake_quant)
            if _hifloat8_supported():
                import torch_npu
                quant_x, pertoken_scale = torch_npu.npu_dynamic_quant(
                    inputs.npu(), dst_type=torch_npu.hifloat8, dst_type_max=15)
                quant_x_fp = torch_npu.npu_dtype_cast(
                    quant_x, inputs.dtype, input_dtype=torch_npu.hifloat8)
                dq_x = quant_x_fp * pertoken_scale.unsqueeze(-1).to(quant_x_fp.dtype)
            else:
                amax = inputs.abs().amax(dim=-1, keepdim=True)
                pertoken_scale = (amax / 15).to(torch.float32)
                dq_x = _hifloat8_fake_quant(inputs / pertoken_scale).to(inputs.dtype) \
                    * pertoken_scale.to(inputs.dtype)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit_test/classic/quantize_op/test_quantile_fallback.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add amct_pytorch/classic/quantize_op/quantile_module.py tests/unit_test/classic/quantize_op/test_quantile_fallback.py
git commit -m "feat: add hifloat8 fallback for quantile dynamic per-token activation"
```

---

## Task 5: 回归与全量单测

**Files:** 无新增，仅验证。

- [ ] **Step 1: 跑量化相关单测全集**

Run: `pytest tests/unit_test/common/utils/test_quant_util.py tests/unit_test/classic/quantize_op/ -v`
Expected: 全部 PASS。

- [ ] **Step 2: 跑既有 hifloat8 集成测试（若环境具备 mock_torch_npu）**

Run: `pytest tests/amct_pytorch/test_quantile.py -v`
Expected: 全部 PASS（确认原生路径行为未变；该用例 mock 了 torch_npu，`_hifloat8_supported` 经 schema 探测返回的分支需与 mock 兼容——若 mock 的 `npu_quant_matmul` 无 `_schemas`，`check_parameters_in_schema` 会抛错，此时应在该测试栈中令 `_hifloat8_supported` 返回 True 的既有 patch 生效；如发现冲突，记录并在执行阶段处理）。

- [ ] **Step 3: 无提交（纯验证任务）**

---

## Self-Review

**Spec coverage:**
- 4.1 helper（`_hifloat8_supported` + `_hifloat8_fake_quant`，复用 `check_parameters_in_schema`）→ Task 1、2 ✓
- 4.2 改造点 A（`quant_dequant_tensor` 自包含）→ Task 3 ✓
- 4.3 改造点 B（quantile 动态 act per-token=amax/15）→ Task 4 ✓
- 6 测试策略 4 条 → Task 1（探测）、Task 2（分支选择+fp32 降 bf16）、Task 3（A 等价性）、Task 4（B per-token scale）✓
- 真机 NPU 端到端不纳入 CI → Task 5 Step 2 仅跑 mock 集成测试 ✓

**Placeholder scan:** 无 TBD/TODO；每个 code step 均含完整代码。Task 5 Step 2 标注了一个执行期可能的 mock 冲突需观察——这是真实风险提示，非占位。

**Type consistency:** `_hifloat8_supported()`、`_hifloat8_fake_quant(fp_tensor)` 在 Task 1-4 命名/签名一致；`quant_dequant_tensor`/`quant_tensor`/`HIFLOAT8` 与现有代码一致。
