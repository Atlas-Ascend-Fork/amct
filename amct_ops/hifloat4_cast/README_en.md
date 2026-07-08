# hifloat4_cast — HiFloat4 FP→HiF4→FP fake-quant operator

FP16/BF16 → HiFloat4 → FP16/BF16 simulation (fake-quant) for Ascend NPU. HiFloat4 (hifx4) is a
4-bit block-scaling float: per element **S1P2** (1 sign + 2-bit mantissa, 8 magnitudes) plus a
per-64-element **3-level scale** (L1 `scale_factor` E6M2 over the 64-block, L2 over each 8, L3
over each 4). This op does a one-shot FP→FP round trip (it does **not** pack real 4-bit); it is
the NPU backend for `amct_pytorch`'s `hifloat4_fake_quant`, used for off-NPU-equivalent accuracy
/ PPL evaluation.

Unlike `hifloat8_cast`, there is **no lookup table**: the scale is an arithmetic per-block
reduction and S1P2 has only 8 magnitudes, so a LUT buys nothing.

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

## Correctness (acceptance red-line)

The kernel reproduces amct's pure-python reference exactly. Validate on device against it:

```python
import torch
from amct_ops.hifloat4_cast import hifloat4_fake_quant
from amct_pytorch.common.utils.hifx4_quant import QType, quant_dequant_float

x = torch.randn(4, 512, dtype=torch.bfloat16)
y_ref = quant_dequant_float(x, QType('hifx4').dim(-1), force_py=True, force_fp32=True)  # CPU reference
y_npu = hifloat4_fake_quant(x.npu()).cpu()                                              # AscendC kernel
assert (y_ref.float() - y_npu.float()).abs().max() < 1e-6
# zeros must not produce NaN
z = torch.zeros(1, 64, dtype=torch.bfloat16)
assert not torch.isnan(hifloat4_fake_quant(z.npu())).any()
```

`tests/amct_ops/test_hifloat4_cast.py` cross-checks the kernel against this CPU reference when an NPU is present.

## Build

```bash
# source CANN set_env.sh first (sets ASCEND_HOME_PATH)
bash amct_ops/ops_build.sh hifloat4_cast                    # A2 (ascend910b), this op only
bash amct_ops/ops_build.sh --soc ascend910_93 hifloat4_cast # A3
bash amct_ops/ops_build.sh                                  # build + package all ops into one wheel
pip install amct_ops/dist/amct_ops-*.whl
```

## Status / notes

- Simulation only. Real 4-bit packed codec + low-bit matmul is future work.
- The kernel does the block reduction and quant in fp32 (bf16 / E6M2 rounding via the hardware
  `vconv`). The kernel body is guarded by `__DAV_C220_VEC__` (A2/A3, dav-2201 = C220, covering
  910B/910C); other arches compile to a no-op, so validate on the target NPU.
