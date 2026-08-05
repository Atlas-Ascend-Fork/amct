# hifloat4_cast — HiFloat4 quantization operator

HiFloat4 (hifx4) is a 4-bit block-scaling float format: per element **S1P2** (1 sign + 2-bit
mantissa, 8 magnitudes) plus a per-64-element **3-level scale** (L1 `scale_factor` E6M2 over the
64-block, L2 over each 8, L3 over each 4).

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  CLI (amct_pytorch/cli/llm/)                            │
│  --quant_dtype hifp  →  bits=4 (HiF4) / bits=8 (HiF8)   │
│    ├─ PTQ:    fake-quant (inject HiF4 error)            │
│    └─ Deploy: blockwise → HiF4 packed bytes            │
│              tensorwise → fp32 unpack                   │
├─────────────────────────────────────────────────────────┤
│  Python (amct_pytorch/quantization/dtypes/hifp_impl.py)  │
│  ├─ hif4_pack()              float → HiF4 uint8 pack    │
│  ├─ hif4_unpack()            HiF4 uint8 → fp32 unpack   │
│  └─ hifloat4_fake_quant()    NPU kernel / CPU fallback   │
├─────────────────────────────────────────────────────────┤
│  NPU Kernel (op_kernel/hifloat4_cast_kernel.cpp)         │
│  └─ hifx_kernel / hifx_kernel_bf16   FP→HiF4→FP round   │
│     trip (fake-quant only; pack/unpack is in Python)     │
└─────────────────────────────────────────────────────────┘
```

- **1 NPU kernel**: `hifx_kernel` (fp32) / `hifx_kernel_bf16` (bf16), performs a single FP→HiF4→FP
  simulation round trip
- **Pack/unpack**: pure PyTorch (`hifp_impl.py`), `hif4_pack` encodes to uint8 HiF4 bytes,
  `hif4_unpack` decodes back to fp32
- **Shared encode/decode**: fake-quant and pack share the same `hif4_encode` / `hif4_decode` logic,
  guaranteeing the evaluation path and deployed weight never drift

## CLI workflow

```bash
# PTQ: fake-quant simulation (injects HiF4 error, no file output)
python -m amct_pytorch.cli.llm.ptq \
    --model /path/to/model --quant_dtype hifp --w_bits 4 --output_dir ./output

# Deploy blockwise: fp16/bf16 → HiF4 packed bytes
python -m amct_pytorch.cli.llm.deploy \
    --model /path/to/model --quant_dtype hifp --granularity block --output_dir ./output

# Deploy tensorwise: HiF4 packed weights → fp32
python -m amct_pytorch.cli.llm.deploy \
    --model /path/to/model --quant_dtype hifp --granularity tensor --output_dir ./output
```

## API

The device op `torch.ops.amct.hifloat4_fake_quant(Tensor input, int qdim=-1) -> Tensor` takes
FP16/BF16, returns the same shape/dtype, and forms 64-element shared-scale blocks along `qdim`.
`qdim` is a schema arg; the permute/pad/slice layout work lives in the **C++ host** (Python is a
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

The kernel reproduces AMCT's pure-python reference exactly. Validate on device against it:

```python
import torch
from amct_ops.hifloat4_cast import hifloat4_fake_quant
from amct_pytorch.quantization.dtypes.hifp_impl import hif4_pack, hif4_unpack

# pack/unpack round-trip
x = torch.randn(2, 128, dtype=torch.bfloat16)
scale, value = hif4_pack(x)
y = hif4_unpack(scale, value)
assert y.shape == x.shape

# NPU kernel vs CPU fallback (~0 error)
y_npu = hifloat4_fake_quant(x.npu()).cpu()
assert (y_npu.float() - y).abs().max() < 1e-6
```

`tests/amct_ops/test_hifloat4_cast.py` cross-checks the kernel against this CPU reference when an NPU is present.

## Packed format

```
scale : uint8 [M, N // 64, 4]     one 4-byte record per 64-group
    byte 0 : E6M2                 level-1 scale factor, E6M2 = 8 bits
    byte 1 : E1_8  (8 × 1 bit)   per-8-subgroup exponent bit
    byte 2 : E1_16 high 8 bits   per-4-subgroup exponent bit, blocks 0..7
    byte 3 : E1_16 low  8 bits   blocks 8..15
value : uint8 [M, N // 2]        two adjacent 4-bit codes packed into one byte
    each nibble = (sign << 3) | code,  code = round(|in_grp| * 4) in 0..7
    low-bit-first: element 0 in low nibble, element 1 in high nibble
```

## Build

```bash
# source CANN set_env.sh first (sets ASCEND_HOME_PATH)
bash amct_ops/ops_build.sh hifloat4_cast                    # A2 (ascend910b), this op only
bash amct_ops/ops_build.sh --soc ascend910_93 hifloat4_cast # A3
bash amct_ops/ops_build.sh                                  # build + package all ops into one wheel
pip install amct_ops/dist/amct_ops-*.whl
```

## Status / notes

- **NPU kernel**: does the block reduction and quant in fp32 (bf16 / E6M2 rounding via the hardware
  `vconv`). The kernel body is guarded by `__DAV_C220_VEC__` (A2/A3, dav-2201 = C220, covering
  910B/910C); other arches compile to a no-op, so validate on the target NPU.
- **Pack/unpack**: done in Python, shares the same encode/decode math (`_hif4_encode_decode`) as the
  NPU kernel, guaranteeing bit-identical results.
