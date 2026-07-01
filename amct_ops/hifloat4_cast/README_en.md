# hifloat4_cast — HiFloat4 FP→HiF4→FP fake-quant operator

FP16/BF16 → HiFloat4 → FP16/BF16 simulation (fake-quant) for Ascend NPU. HiFloat4 (hifx4) is a
4-bit block-scaling float: per element **S1P2** (1 sign + 2-bit mantissa, 8 magnitudes) plus a
per-64-element **3-level scale** (L1 `scale_factor` E6M2 over the 64-block, L2 over each 8, L3
over each 4). This op does a one-shot FP→FP round trip (it does **not** pack real 4-bit); it is
the NPU backend for `amct_pytorch`'s `hifloat4_fake_quant`, used for off-NPU-equivalent accuracy
/ PPL evaluation.

Unlike `hifloat8_cast`, there is **no lookup table**: the scale is an arithmetic per-block
reduction and S1P2 has only 8 magnitudes, so a LUT buys nothing (see design doc P1.7).

## API

The device op `torch.ops.amct.hifloat4_fake_quant(Tensor input, int qdim=-1) -> Tensor` takes
FP16/BF16, returns the same shape/dtype, and forms 64-element shared-scale blocks along `qdim`.
`qdim` is a schema arg; the permute/pad/slice layout work lives in the **C++ host** (python is a
thin passthrough):

```python
from amct_ops.hifloat4_cast import hifloat4_fake_quant
y = hifloat4_fake_quant(x)                      # blocks along the last dim
y = hifloat4_fake_quant(w, qdim=1)              # e.g. Linear weight [out, in] along in
y = torch.ops.amct.hifloat4_fake_quant(x, -1)   # equivalent
```

The host moves `qdim` to the last axis, zero-pads to a multiple of 64 (zeros don't affect the
block max), calls the kernel, then slices off the pad and restores the dim order.

## Correctness (P1.6 — the acceptance red-line)

The kernel reproduces the pure-python reference exactly. Validate on device against the bundled
oracle:

```python
import torch
from amct_ops.hifloat4_cast import hifloat4_fake_quant
from amct_ops.hifloat4_cast.reference import hifloat4_reference

x = torch.randn(4, 512, dtype=torch.bfloat16)
y_ref = hifloat4_reference(x.float()).to(x.dtype)      # CPU oracle
y_npu = hifloat4_fake_quant(x.npu()).cpu()             # AscendC kernel
assert (y_ref.float() - y_npu.float()).abs().max() < 1e-6
# zeros must not produce NaN
z = torch.zeros(1, 64, dtype=torch.bfloat16)
assert not torch.isnan(hifloat4_fake_quant(z.npu())).any()
```

`tests/amct_ops/test_hifloat4_cast.py` cross-checks `reference.py` against the vendored
`quant_hifx` on CPU (runs anywhere) and, when an NPU is present, against the kernel.

## Build

```bash
# source CANN set_env.sh first (sets ASCEND_HOME_PATH)
bash amct_ops/ops_build.sh hifloat4_cast                    # A2 (ascend910b), this op only
bash amct_ops/ops_build.sh --soc ascend910_93 hifloat4_cast # A3
bash amct_ops/ops_build.sh                                  # build + package all ops into one wheel
pip install amct_ops/dist/amct_ops-*.whl
```

## Status / notes

- Simulation only (stage 2A). Real 4-bit packed codec + low-bit matmul (stage 2C) is future work.
- The kernel computes in fp32 with bf16 rounding replicated at the reference's rounding points.
  The scalar float bit-tricks (`Pow2i` / `FloorLog2Pos` / `Bf16Round`) avoid libm; if any device
  math intrinsic differs, adjust only those helpers and re-check against `reference.py`.
