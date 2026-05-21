# Bias Quantization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add bias quantization to the fake quant forward path (CPU/GPU evaluation) and the NPU deploy module so both routes match real NPU integer bias addition behavior for INT8 PER_TENSOR, while preserving cann/master fp-add behavior for FP8/HIFLOAT8.

**Architecture:**
- Fake quant: add `_fake_quant_bias(deq_scale)` (quant→dequant simulation, INT8 PER_TENSOR only) + `_build_fake_quant_cache()` (lazy cache for weight dequant / deq_scale / quantized bias) inside `BaseQuantizeModule`. Rewrite `_fake_quant_forward` to compute activation dequant per batch and reuse the cache for everything else.
- Deploy: rewrite `_init_bias` to quantize bias to INT32 for INT8 PER_TENSOR + non-dynamic, keep fp for FP8×FP8 / HIFLOAT8×HIFLOAT8, and keep fp + pass to NPU op for INT8 PER_TOKEN / dynamic. Use a new `_bias_for_npu_op` flag for forward dispatch.

**Tech Stack:** Python 3, PyTorch, `amct_pytorch` (internal), unittest

**Spec:** `docs/superpowers/specs/2026-05-21-bias-quant-design.md`

**Starting point:** branch `feature/fake-quant-forward` at HEAD (spec already committed).

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Modify | `amct_pytorch/utils/vars.py` | Add `INT32_MIN` / `INT32_MAX` constants |
| Modify | `amct_pytorch/quantize_op/base_quant_module.py` | New `_fake_quant_bias`, `_build_fake_quant_cache`; rewrite `_fake_quant_forward` to use cache + bias quantization |
| Modify | `amct_pytorch/deploy_op/npu_quantization_linear.py` | Rewrite `_init_bias` with `_bias_for_npu_op` flag; forward `is_new_torch_npu` branch dispatches on flag |
| Modify | `tests/amct_pytorch/test_fake_quant_forward.py` | Adapt 2 existing tests whose modules carry non-None bias on INT8 PER_TENSOR path; add 8 new tests |

`amct_pytorch/quantize_op/utils.py:apply_progressive_quant_dequant` already exists (no change needed).

---

### Task 1: Add INT32 constants to `vars.py`

**Files:**
- Modify: `amct_pytorch/utils/vars.py`

- [ ] **Step 1: Add INT32_MIN / INT32_MAX after INT8_MIN/INT8_MAX**

Open `amct_pytorch/utils/vars.py` and locate:

```python
INT8_MAX = torch.iinfo(torch.int8).max
INT8_MIN = torch.iinfo(torch.int8).min
INT4_MAX = 7
INT4_MIN = -8
```

Replace with:

```python
INT8_MAX = torch.iinfo(torch.int8).max
INT8_MIN = torch.iinfo(torch.int8).min
INT4_MAX = 7
INT4_MIN = -8
INT32_MAX = torch.iinfo(torch.int32).max
INT32_MIN = torch.iinfo(torch.int32).min
```

- [ ] **Step 2: Smoke-check the import**

```bash
cd /mnt/workspace/gitCode/annqr/amct_open
python -c "from amct_pytorch.utils.vars import INT32_MIN, INT32_MAX; print(INT32_MIN, INT32_MAX)"
```

Expected output: `-2147483648 2147483647`

- [ ] **Step 3: Commit**

```bash
git add amct_pytorch/utils/vars.py
SKIP=ruff-check,ruff-format PRE_COMMIT_ALLOW_NO_CONFIG=1 git commit -m "feat: add INT32_MIN/INT32_MAX constants"
```

---

### Task 2: Add `_fake_quant_bias` helper (fake-quant side, no integration yet)

**Why first:** Keep the new helper pure and independently testable before wiring it into `_fake_quant_forward`.

**Files:**
- Modify: `amct_pytorch/quantize_op/base_quant_module.py`
- Test: `tests/amct_pytorch/test_fake_quant_forward.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/amct_pytorch/test_fake_quant_forward.py`:

```python
class TestFakeQuantBias(unittest.TestCase):
    """_fake_quant_bias quantizes bias only on INT8 + PER_TENSOR; otherwise raw."""

    def _make_int8_per_tensor_module(self, weight_only=False):
        from amct_pytorch.quantize_op.minmax_module import MinMaxQuant
        linear = nn.Linear(8, 4, bias=True).to(torch.bfloat16)
        cfg = {
            'batch_num': 1,
            'weights_cfg': {'quant_type': 'int8', 'symmetric': True,
                            'strategy': 'channel', 'group_size': None},
            'inputs_cfg': {'quant_type': 'int8', 'symmetric': True,
                           'strategy': 'tensor', 'enable_quant': not weight_only},
        }
        return MinMaxQuant(linear, 'l', cfg)

    def test_returns_none_when_bias_none(self):
        mod = self._make_int8_per_tensor_module()
        mod.bias = None
        deq_scale = torch.ones(4).to(torch.bfloat16)
        self.assertIsNone(mod._fake_quant_bias(deq_scale))

    def test_returns_raw_bias_when_deq_scale_none(self):
        mod = self._make_int8_per_tensor_module()
        self.assertIs(mod._fake_quant_bias(None), mod.bias)

    def test_returns_raw_bias_for_per_token_granularity(self):
        mod = self._make_int8_per_tensor_module()
        mod.act_granularity = 'token'
        deq_scale = torch.ones(4).to(torch.bfloat16)
        self.assertIs(mod._fake_quant_bias(deq_scale), mod.bias)

    def test_returns_raw_bias_for_non_int8_act(self):
        mod = self._make_int8_per_tensor_module()
        mod.act_type = 'float8_e4m3fn'
        deq_scale = torch.ones(4).to(torch.bfloat16)
        self.assertIs(mod._fake_quant_bias(deq_scale), mod.bias)

    def test_int8_per_tensor_round_trip_through_int32(self):
        from amct_pytorch.utils.vars import INT32_MIN, INT32_MAX
        mod = self._make_int8_per_tensor_module()
        deq_scale = torch.tensor([0.25, 0.5, 0.125, 1.0]).to(mod.bias.dtype)
        result = mod._fake_quant_bias(deq_scale)
        expected_int = (mod.bias.data / deq_scale).round().clamp(INT32_MIN, INT32_MAX)
        expected = expected_int * deq_scale
        self.assertTrue(torch.allclose(result, expected),
                        f'max diff: {(result - expected).abs().max()}')
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /mnt/workspace/gitCode/annqr/amct_open/tests/amct_pytorch
python -m pytest test_fake_quant_forward.py::TestFakeQuantBias -v
```

Expected: 5 failures, all `AttributeError: 'MinMaxQuant' object has no attribute '_fake_quant_bias'`.

- [ ] **Step 3: Implement `_fake_quant_bias` on `BaseQuantizeModule`**

Open `amct_pytorch/quantize_op/base_quant_module.py`. Replace the top import block and `BaseQuantizeModule.__init__`-through-`_quantizable_input` region with:

```python
import torch
import torch.nn as nn
from amct_pytorch.utils.vars import INT8, INT32_MAX, INT32_MIN


class BaseQuantizeModule(nn.Module):
    def __init__(self, ori_module, layer_name, quant_config):
        super().__init__()
        self.ori_module_type = None
        self.act_type = None
        self.wts_type = None
        self.scale_w = None
        self.scale_d = None
        self.offset_w = None
        self.offset_d = None
        self.group_size = None

        self.scale = None
        self.clip_max = None

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

Leave `_fake_quant_forward` and `forward` below unchanged for now.

- [ ] **Step 4: Run the new tests to verify they pass**

```bash
python -m pytest test_fake_quant_forward.py::TestFakeQuantBias -v
```

Expected: all 5 PASS.

- [ ] **Step 5: Run the full fake-quant test file to make sure nothing else broke**

```bash
python -m pytest test_fake_quant_forward.py -v
```

Expected: all previous tests still pass + 5 new tests pass.

- [ ] **Step 6: Commit**

```bash
cd /mnt/workspace/gitCode/annqr/amct_open
git add amct_pytorch/quantize_op/base_quant_module.py tests/amct_pytorch/test_fake_quant_forward.py
SKIP=ruff-check,ruff-format PRE_COMMIT_ALLOW_NO_CONFIG=1 git commit -m "feat: add _fake_quant_bias helper for INT8 PER_TENSOR bias quantization"
```

---

### Task 3: Add `_build_fake_quant_cache` and rewrite `_fake_quant_forward`

**Why:** Cache the calibration-dependent artifacts (weight dequant, deq_scale, quantized bias) so per-batch forward only computes activation dequant. Also wire `_fake_quant_bias` into the linear/conv2d call.

**Files:**
- Modify: `amct_pytorch/quantize_op/base_quant_module.py`
- Modify: `tests/amct_pytorch/test_fake_quant_forward.py` (adapt 2 existing tests, add cache tests)

- [ ] **Step 1: Write the failing cache tests**

Append to `tests/amct_pytorch/test_fake_quant_forward.py`:

```python
class TestFakeQuantCache(unittest.TestCase):
    """_build_fake_quant_cache runs once; subsequent forward calls reuse the cache."""

    def _make_int8_weight_only_module(self):
        from amct_pytorch.quantize_op.minmax_module import MinMaxQuant
        linear = nn.Linear(8, 4, bias=True).to(torch.bfloat16)
        cfg = {
            'batch_num': 1,
            'weights_cfg': {'quant_type': 'int8', 'symmetric': True,
                            'strategy': 'channel', 'group_size': None},
            'inputs_cfg': {'quant_type': 'int8', 'symmetric': True,
                           'strategy': 'tensor', 'enable_quant': False},
        }
        return MinMaxQuant(linear, 'l', cfg)

    def test_weight_dequant_computed_once_across_calls(self):
        from amct_pytorch.utils import quant_util
        mod = self._make_int8_weight_only_module()
        x = torch.randn(2, 8).to(torch.bfloat16)
        with patch.object(quant_util, 'quant_dequant_tensor',
                          wraps=quant_util.quant_dequant_tensor) as m:
            mod._fake_quant_forward(x)
            mod._fake_quant_forward(x)
            mod._fake_quant_forward(x)
        # weight dequant: 1 call total (cached); activation dequant: 0 calls
        # because this module is weight-only (scale_d is None).
        self.assertEqual(m.call_count, 1)

    def test_progressive_dequant_computed_once_across_calls(self):
        from amct_pytorch.quantize_op.minmax_module import MinMaxQuant
        from amct_pytorch.quantize_op import utils as quant_utils
        linear = nn.Linear(8, 4, bias=True).to(torch.bfloat16)
        cfg = {
            'batch_num': 1,
            'weights_cfg': {'quant_type': 'int8', 'symmetric': True,
                            'strategy': 'channel', 'group_size': None},
            'inputs_cfg': {'quant_type': 'int8', 'symmetric': True,
                           'strategy': 'tensor', 'enable_quant': False},
        }
        mod = MinMaxQuant(linear, 'l', cfg)
        mod.scale_w = None
        mod.scale_w1 = torch.ones(4).to(torch.bfloat16)
        mod.scale_w2 = torch.ones(1, 1).to(torch.bfloat16)
        sentinel_dq_w = torch.randn(4, 8).to(torch.bfloat16)
        x = torch.randn(2, 8).to(torch.bfloat16)
        with patch.object(quant_utils, 'apply_progressive_quant_dequant',
                          return_value=sentinel_dq_w) as m:
            mod._fake_quant_forward(x)
            mod._fake_quant_forward(x)
        self.assertEqual(m.call_count, 1)
```

- [ ] **Step 2: Run cache tests to verify they fail**

```bash
cd /mnt/workspace/gitCode/annqr/amct_open/tests/amct_pytorch
python -m pytest test_fake_quant_forward.py::TestFakeQuantCache -v
```

Expected: both FAIL because the current implementation calls `quant_dequant_tensor` / `apply_progressive_quant_dequant` once per forward (3 / 2 calls respectively).

- [ ] **Step 3: Add `_build_fake_quant_cache` and rewrite `_fake_quant_forward`**

In `amct_pytorch/quantize_op/base_quant_module.py`, replace the existing `_fake_quant_forward` (with the surrounding `@torch.no_grad()` decorator and `forward(self, inputs): pass`) with:

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

    def forward(self, inputs):
        pass
```

- [ ] **Step 4: Run cache tests to verify they pass**

```bash
python -m pytest test_fake_quant_forward.py::TestFakeQuantCache -v
```

Expected: both PASS.

- [ ] **Step 5: Run the full fake-quant test file to find adapted-test breakages**

```bash
python -m pytest test_fake_quant_forward.py -v
```

Expected failures: 2 tests that had non-None bias on INT8 PER_TENSOR path now produce a different output:
- `test_after_calibration_weight_act_fake_quant` (MinMax)
- `test_after_calibration_uses_smooth_factor` (SmoothQuant)

Other tests should pass.

- [ ] **Step 6: Adapt the 2 affected existing tests**

Open `tests/amct_pytorch/test_fake_quant_forward.py`. Locate `TestMinMaxFakeQuant.test_after_calibration_weight_act_fake_quant`:

```python
    def test_after_calibration_weight_act_fake_quant(self):
        from amct_pytorch.quantize_op.minmax_module import MinMaxQuant
        from amct_pytorch.utils.quant_util import quant_dequant_tensor
        linear = nn.Linear(8, 4).to(torch.bfloat16)
        mod = MinMaxQuant(linear, 'l', self._cfg(weight_only=False))
        x = torch.randn(2, 8).to(torch.bfloat16)
        mod(x)
        mod(x)
        out = mod(x)
        dq_w = quant_dequant_tensor(mod.weight.data, 'int8', mod.scale_w, mod.offset_w)
        dq_x = quant_dequant_tensor(x, 'int8', mod.scale_d, mod.offset_d)
        expected = F.linear(dq_x, dq_w, mod.bias)
        self.assertTrue(torch.allclose(out, expected),
                        f'max diff: {(out - expected).abs().max()}')
```

Replace with the bias-quantized expected:

```python
    def test_after_calibration_weight_act_fake_quant(self):
        from amct_pytorch.quantize_op.minmax_module import MinMaxQuant
        from amct_pytorch.utils.quant_util import quant_dequant_tensor
        from amct_pytorch.utils.vars import INT32_MIN, INT32_MAX
        linear = nn.Linear(8, 4).to(torch.bfloat16)
        mod = MinMaxQuant(linear, 'l', self._cfg(weight_only=False))
        x = torch.randn(2, 8).to(torch.bfloat16)
        mod(x)
        mod(x)
        out = mod(x)
        dq_w = quant_dequant_tensor(mod.weight.data, 'int8', mod.scale_w, mod.offset_w)
        dq_x = quant_dequant_tensor(x, 'int8', mod.scale_d, mod.offset_d)
        deq_scale = (mod.scale_d.to(mod.bias.dtype)
                     * mod.scale_w.to(mod.bias.dtype)).reshape(-1)
        bias_q = (mod.bias.data / deq_scale).round().clamp(INT32_MIN, INT32_MAX)
        expected_bias = bias_q * deq_scale
        expected = F.linear(dq_x, dq_w, expected_bias)
        self.assertTrue(torch.allclose(out, expected),
                        f'max diff: {(out - expected).abs().max()}')
```

Locate `TestSmoothQuantFakeQuant.test_after_calibration_uses_smooth_factor`:

```python
    def test_after_calibration_uses_smooth_factor(self):
        from amct_pytorch.quantize_op.smooth_quant_module import SmoothQuant
        from amct_pytorch.utils.quant_util import quant_dequant_tensor
        linear = nn.Linear(8, 4).to(torch.bfloat16)
        mod = SmoothQuant(linear, 'l', self._cfg())
        x = torch.randn(2, 8).to(torch.bfloat16)
        mod(x)
        out = mod(x)

        smooth = mod.scale
        dq_w = quant_dequant_tensor(
            mod.weight.data * smooth.to(dtype=mod.weight.dtype),
            'int8', mod.scale_w, mod.offset_w)
        dq_x = quant_dequant_tensor(
            x / smooth.to(dtype=x.dtype),
            'int8', mod.scale_d, mod.offset_d)
        expected = F.linear(dq_x, dq_w, mod.bias)
        self.assertTrue(torch.allclose(out, expected),
                        f'max diff: {(out - expected).abs().max()}')
```

Replace with:

```python
    def test_after_calibration_uses_smooth_factor(self):
        from amct_pytorch.quantize_op.smooth_quant_module import SmoothQuant
        from amct_pytorch.utils.quant_util import quant_dequant_tensor
        from amct_pytorch.utils.vars import INT32_MIN, INT32_MAX
        linear = nn.Linear(8, 4).to(torch.bfloat16)
        mod = SmoothQuant(linear, 'l', self._cfg())
        x = torch.randn(2, 8).to(torch.bfloat16)
        mod(x)
        out = mod(x)

        smooth = mod.scale
        dq_w = quant_dequant_tensor(
            mod.weight.data * smooth.to(dtype=mod.weight.dtype),
            'int8', mod.scale_w, mod.offset_w)
        dq_x = quant_dequant_tensor(
            x / smooth.to(dtype=x.dtype),
            'int8', mod.scale_d, mod.offset_d)
        deq_scale = (mod.scale_d.to(mod.bias.dtype)
                     * mod.scale_w.to(mod.bias.dtype)).reshape(-1)
        bias_q = (mod.bias.data / deq_scale).round().clamp(INT32_MIN, INT32_MAX)
        expected_bias = bias_q * deq_scale
        expected = F.linear(dq_x, dq_w, expected_bias)
        self.assertTrue(torch.allclose(out, expected),
                        f'max diff: {(out - expected).abs().max()}')
```

- [ ] **Step 7: Run the full fake-quant test file again**

```bash
python -m pytest test_fake_quant_forward.py -v
```

Expected: all tests PASS.

- [ ] **Step 8: Commit**

```bash
cd /mnt/workspace/gitCode/annqr/amct_open
git add amct_pytorch/quantize_op/base_quant_module.py tests/amct_pytorch/test_fake_quant_forward.py
SKIP=ruff-check,ruff-format PRE_COMMIT_ALLOW_NO_CONFIG=1 git commit -m "feat: cache fake-quant artifacts and wire bias quantization into _fake_quant_forward"
```

---

### Task 4: Add remaining bias-route tests

**Why:** Lock in dispatch behavior for FP8/HIFLOAT8 PER_TENSOR (mock-based, since the FP8 ops need NPU), `bias=None` path, progressive path, and PER_TOKEN.

**Files:**
- Modify: `tests/amct_pytorch/test_fake_quant_forward.py`

- [ ] **Step 1: Append the new tests**

Append to `tests/amct_pytorch/test_fake_quant_forward.py`:

```python
class TestFakeQuantBiasDispatch(unittest.TestCase):
    """End-to-end dispatch: bias only quantized on INT8 PER_TENSOR; others raw."""

    def _make_module(self, act_type='int8', granularity='tensor', enable_quant=True,
                     bias=True):
        from amct_pytorch.quantize_op.minmax_module import MinMaxQuant
        linear = nn.Linear(8, 4, bias=bias).to(torch.bfloat16)
        cfg = {
            'batch_num': 1,
            'weights_cfg': {'quant_type': 'int8', 'symmetric': True,
                            'strategy': 'channel', 'group_size': None},
            'inputs_cfg': {'quant_type': act_type, 'symmetric': True,
                           'strategy': granularity, 'enable_quant': enable_quant},
        }
        return MinMaxQuant(linear, 'l', cfg)

    def test_fp8_per_tensor_keeps_bias_fp(self):
        from amct_pytorch.utils import quant_util
        mod = self._make_module(act_type='int8', granularity='tensor')
        mod.act_type = 'float8_e4m3fn'  # override to FP8 after calibration
        x = torch.randn(2, 8).to(torch.bfloat16)
        # Pre-build the cache to lock the dispatch decision.
        with patch.object(quant_util, 'quant_dequant_tensor',
                          wraps=quant_util.quant_dequant_tensor):
            mod._build_fake_quant_cache()
        self.assertIs(mod._cached_bias_for_linear, mod.bias)

    def test_hifloat8_per_tensor_keeps_bias_fp(self):
        from amct_pytorch.utils import quant_util
        mod = self._make_module(act_type='int8', granularity='tensor')
        mod.act_type = 'hifloat8'
        with patch.object(quant_util, 'quant_dequant_tensor',
                          wraps=quant_util.quant_dequant_tensor):
            mod._build_fake_quant_cache()
        self.assertIs(mod._cached_bias_for_linear, mod.bias)

    def test_int8_per_token_skips_bias_quantization(self):
        mod = self._make_module(act_type='int8', granularity='token')
        mod._build_fake_quant_cache()
        self.assertIs(mod._cached_bias_for_linear, mod.bias)

    def test_none_bias_path_does_not_error(self):
        mod = self._make_module(bias=False)
        x = torch.randn(2, 8).to(torch.bfloat16)
        out = mod._fake_quant_forward(x)
        self.assertEqual(out.shape, (2, 4))

    def test_progressive_path_passes_bias_unchanged(self):
        from amct_pytorch.quantize_op import utils as quant_utils
        mod = self._make_module(bias=False)
        mod.scale_w = None
        mod.scale_w1 = torch.ones(4).to(torch.bfloat16)
        mod.scale_w2 = torch.ones(1, 1).to(torch.bfloat16)
        sentinel_dq_w = torch.randn(4, 8).to(torch.bfloat16)
        x = torch.randn(2, 8).to(torch.bfloat16)
        with patch.object(quant_utils, 'apply_progressive_quant_dequant',
                          return_value=sentinel_dq_w):
            out = mod._fake_quant_forward(x)
        # bias=None in this setup -> F.linear bias arg is None -> no error and shape matches
        self.assertEqual(out.shape, (2, 4))
        self.assertIsNone(mod._cached_bias_for_linear)
```

- [ ] **Step 2: Run the new tests**

```bash
cd /mnt/workspace/gitCode/annqr/amct_open/tests/amct_pytorch
python -m pytest test_fake_quant_forward.py::TestFakeQuantBiasDispatch -v
```

Expected: all 5 PASS.

- [ ] **Step 3: Run the entire test file as a sanity check**

```bash
python -m pytest test_fake_quant_forward.py -v
```

Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
cd /mnt/workspace/gitCode/annqr/amct_open
git add tests/amct_pytorch/test_fake_quant_forward.py
SKIP=ruff-check,ruff-format PRE_COMMIT_ALLOW_NO_CONFIG=1 git commit -m "test: cover FP8/HIFLOAT8/per-token/none-bias/progressive bias dispatch paths"
```

---

### Task 5: Rewrite `_init_bias` in `npu_quantization_linear.py`

**Why:** Quantize bias to INT32 for INT8 PER_TENSOR + non-dynamic; preserve cann/master behavior (no NPU op bias) for FP8×FP8 / HIFLOAT8×HIFLOAT8; otherwise pass fp bias to NPU op. Record the dispatch decision in `self._bias_for_npu_op`.

**Files:**
- Modify: `amct_pytorch/deploy_op/npu_quantization_linear.py`

- [ ] **Step 1: Verify the current `_init_bias` matches cann/master**

```bash
cd /mnt/workspace/gitCode/annqr/amct_open
git diff cann/master HEAD -- amct_pytorch/deploy_op/npu_quantization_linear.py
```

Expected output: empty (file unchanged by the fake-quant feature branch).

- [ ] **Step 2: Replace `_init_bias`**

Open `amct_pytorch/deploy_op/npu_quantization_linear.py`. Find the existing method:

```python
    def _init_bias(self, module):
        """
        Function: init bias for npu op
        Args:
            module: quant module 
        """
        if module.bias is not None:
            self.register_buffer('bias', module.bias)
            if self.offset_bias is not None:
                self.bias = self.bias + self.offset_bias
        else:
            if self.offset_bias is None:
                self.bias = None 
            else:
                self.bias = self.offset_bias
```

Replace with:

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

- [ ] **Step 3: Smoke import**

```bash
cd /mnt/workspace/gitCode/annqr/amct_open
python -c "from amct_pytorch.deploy_op.npu_quantization_linear import NpuQuantizationLinear; print('import OK')"
```

Expected output: `import OK`. (Run from a shell that has `torch_npu` installed; the harness env does.)

- [ ] **Step 4: Commit**

```bash
git add amct_pytorch/deploy_op/npu_quantization_linear.py
SKIP=ruff-check,ruff-format PRE_COMMIT_ALLOW_NO_CONFIG=1 git commit -m "feat: deploy _init_bias quantizes INT8 PER_TENSOR bias to INT32"
```

---

### Task 6: Update deploy forward to dispatch on `_bias_for_npu_op`

**Why:** When `_init_bias` decides to pass bias to the NPU op, the new-torch_npu path must include it as the `bias=` argument; otherwise keep the existing op-外 fp 加 behavior.

**Files:**
- Modify: `amct_pytorch/deploy_op/npu_quantization_linear.py`

- [ ] **Step 1: Replace the `is_new_torch_npu` branch in `forward`**

In `amct_pytorch/deploy_op/npu_quantization_linear.py`, locate the `forward` method's `is_new_torch_npu` branch:

```python
        if self.is_new_torch_npu:
            if self.act_type == FLOAT8_E4M3FN and self.wts_type == FLOAT4_E2M1:
                ori_shape = quant_x.shape
                quant_x = quant_x.reshape(-1, x.shape[-1])
            output = torch_npu.npu_quant_matmul(quant_x, self.quantized_weight,
                scale=self.deq_scale, pertoken_scale=self.pertoken_scale,
                output_dtype=self.output_dtype,
                x1_dtype=self.x1_dtype, x2_dtype=self.x2_dtype,
                group_sizes=self.group_sizes, y_scale=self.y_scale)
            if self.act_type == FLOAT8_E4M3FN and self.wts_type == FLOAT4_E2M1:
                output = output.reshape(*ori_shape[:-1], -1)
            if self.bias is not None:
                output = output + self.bias
        else:
            output = torch_npu.npu_quant_matmul(quant_x, self.quantized_weight,
                scale=self.deq_scale, pertoken_scale=self.pertoken_scale,
                bias=self.bias, output_dtype=self.output_dtype)
```

Replace with:

```python
        if self.is_new_torch_npu:
            if self.act_type == FLOAT8_E4M3FN and self.wts_type == FLOAT4_E2M1:
                ori_shape = quant_x.shape
                quant_x = quant_x.reshape(-1, x.shape[-1])
            if self._bias_for_npu_op and self.bias is not None:
                output = torch_npu.npu_quant_matmul(quant_x, self.quantized_weight,
                    scale=self.deq_scale, pertoken_scale=self.pertoken_scale,
                    bias=self.bias, output_dtype=self.output_dtype,
                    x1_dtype=self.x1_dtype, x2_dtype=self.x2_dtype,
                    group_sizes=self.group_sizes, y_scale=self.y_scale)
                if self.act_type == FLOAT8_E4M3FN and self.wts_type == FLOAT4_E2M1:
                    output = output.reshape(*ori_shape[:-1], -1)
            else:
                output = torch_npu.npu_quant_matmul(quant_x, self.quantized_weight,
                    scale=self.deq_scale, pertoken_scale=self.pertoken_scale,
                    output_dtype=self.output_dtype,
                    x1_dtype=self.x1_dtype, x2_dtype=self.x2_dtype,
                    group_sizes=self.group_sizes, y_scale=self.y_scale)
                if self.act_type == FLOAT8_E4M3FN and self.wts_type == FLOAT4_E2M1:
                    output = output.reshape(*ori_shape[:-1], -1)
                if self.bias is not None:
                    output = output + self.bias
        else:
            output = torch_npu.npu_quant_matmul(quant_x, self.quantized_weight,
                scale=self.deq_scale, pertoken_scale=self.pertoken_scale,
                bias=self.bias, output_dtype=self.output_dtype)
```

- [ ] **Step 2: Smoke import**

```bash
python -c "from amct_pytorch.deploy_op.npu_quantization_linear import NpuQuantizationLinear; print('import OK')"
```

Expected output: `import OK`.

- [ ] **Step 3: Commit**

```bash
git add amct_pytorch/deploy_op/npu_quantization_linear.py
SKIP=ruff-check,ruff-format PRE_COMMIT_ALLOW_NO_CONFIG=1 git commit -m "feat: forward dispatches bias to NPU op based on _bias_for_npu_op flag"
```

---

### Task 7: Final regression

**Files:** all fake-quant tests + the deploy module (import smoke)

- [ ] **Step 1: Run the full fake-quant test suite**

```bash
cd /mnt/workspace/gitCode/annqr/amct_open/tests/amct_pytorch
python -m pytest test_fake_quant_forward.py -v
```

Expected: all tests pass (16 from prior work + 5 `TestFakeQuantBias` + 2 `TestFakeQuantCache` + 5 `TestFakeQuantBiasDispatch` = 28 total; 2 of the prior 16 are adapted in Task 3 and remain passing).

- [ ] **Step 2: Confirm no unintended changes to other files**

```bash
cd /mnt/workspace/gitCode/annqr/amct_open
git diff --stat cann/master HEAD
```

Expected: changes confined to
- `amct_pytorch/quantize_op/base_quant_module.py`
- `amct_pytorch/quantize_op/linear_awq_module.py` (from prior fake-quant work)
- `amct_pytorch/quantize_op/minmax_module.py`
- `amct_pytorch/quantize_op/ofmr_quant_module.py`
- `amct_pytorch/quantize_op/smooth_quant_module.py`
- `amct_pytorch/quantize_op/utils.py`
- `amct_pytorch/deploy_op/npu_quantization_linear.py`
- `amct_pytorch/utils/vars.py`
- `tests/amct_pytorch/test_fake_quant_forward.py`
- `docs/superpowers/specs/2026-05-21-bias-quant-design.md`
- `docs/superpowers/plans/2026-05-21-bias-quant.md`

- [ ] **Step 3: No extra commit if Tasks 1-6 already committed everything**

Otherwise:

```bash
git status
# If anything is uncommitted that belongs to this feature, add it and commit.
```

---

## Self-review

**Spec coverage:**
- Touch points in `base_quant_module.py` (3.2 / 3.3 / 3.4 / 3.5) → Task 2, Task 3 ✓
- `INT32_MIN` / `INT32_MAX` introduction → Task 1 ✓
- `apply_progressive_quant_dequant` (Section 4) → already exists, referenced in Task 3 ✓
- Deploy `_init_bias` rewrite (Section 5.1) → Task 5 ✓
- Deploy forward dispatch (Section 5.2) → Task 6 ✓
- 8 new tests (Section 6) → distributed across Tasks 2, 3, 4
- Cache safety / `_bias_for_npu_op` motivation / data-type matrix / math equivalence → covered by code matching spec, no separate task needed

**Placeholder scan:** no TBD / TODO / "implement appropriate" / "similar to Task N" / vague steps. ✓

**Type consistency:** `_fake_quant_bias`, `_build_fake_quant_cache`, `_cached_dequant_w`, `_cached_deq_scale`, `_cached_bias_for_linear`, `_fake_quant_cache_ready`, `_bias_for_npu_op` — names match across all tasks. ✓

**Constants:** `INT8`, `INT32_MAX`, `INT32_MIN`, `FLOAT8_E4M3FN`, `HIFLOAT8`, `FLOAT4_E2M1` — all from `amct_pytorch.utils.vars`. `INT8`, `FLOAT8_E4M3FN`, `HIFLOAT8`, `FLOAT4_E2M1` are already imported in `npu_quantization_linear.py`; `INT32_MAX`/`INT32_MIN` introduced in Task 1.
