# hifloat4_cast — HiFloat4 quantization operator

HiFloat4 (hifx4) is a 4-bit block-scaling float format: per element **S1P2** (1 sign + 2-bit
mantissa, 8 magnitudes) plus a per-64-element **3-level scale** (L1 `scale_factor` E6M2 over the
64-block, L2 over each 8, L3 over each 4).

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  CLI (amct_pytorch/cli/llm/)                            │
│  --quant_dtype hifp  →  auto-detect source format       │
│    ├─ PTQ:   fake-quant + unpack (HiF4 → float)         │
│    └─ Deploy: pack (float → HiF4 bytes)                 │
├─────────────────────────────────────────────────────────┤
│  Python (amct_pytorch/common/utils/hifloat4_quant/)      │
│  ├─ _hif4_encode_decode.py  shared encode/decode math    │
│  ├─ hif4_pack_unpack.py     float ↔ HiF4 uint8 packer   │
│  └─ quant_dequant.py        fake-quant entry point       │
├─────────────────────────────────────────────────────────┤
│  NPU Kernel (op_kernel/hifloat4_cast_kernel.cpp)         │
│  └─ hifx_kernel / hifx_kernel_bf16   FP→HiF4→FP round   │
│     trip (fake-quant only; pack/unpack is in Python)     │
└─────────────────────────────────────────────────────────┘
```

- **1 NPU kernel**: `hifx_kernel` (fp32) / `hifx_kernel_bf16` (bf16), performs a single FP→HiF4→FP
  simulation round trip
- **Pack/unpack**: pure PyTorch (`hif4_pack_unpack.py`), encodes float weights to uint8 HiF4 bytes
  or decodes bytes back to float
- **Shared encode/decode**: fake-quant and pack share the same `_hif4_encode_decode` logic,
  guaranteeing the evaluation path and deployed weight never drift

## CLI workflow

```bash
# PTQ: fake-quant simulation or HiF4 unpack
python -m amct_pytorch.cli.llm.ptq \
    --model /path/to/model --quant_dtype hifp --output_dir ./output

# Deploy: float → HiF4 uint8 bytes, outputs safetensors
python -m amct_pytorch.cli.llm.deploy \
    --model /path/to/model --quant_dtype hifp --output_dir ./output
```

When `--quant_dtype hifp`, the CLI **auto-detects** the input model format:
- **hif4 packed** → PTQ directly unpacks to a float checkpoint (no model build needed)
- **bf16/fp16 raw weights** → PTQ runs fake-quant evaluation, or deploy runs real packing

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
from amct_pytorch.common.utils.hifloat4_quant import QType, quant_dequant_float

x = torch.randn(4, 512, dtype=torch.bfloat16)
y_ref = quant_dequant_float(x, QType('hifx4').dim(-1), force_py=True, force_fp32=True)  # CPU reference
y_npu = hifloat4_fake_quant(x.npu()).cpu()                                              # AscendC kernel
assert (y_ref.float() - y_npu.float()).abs().max() < 1e-6
# zeros must not produce NaN
z = torch.zeros(1, 64, dtype=torch.bfloat16)
assert not torch.isnan(hifloat4_fake_quant(z.npu())).any()
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

## Changes from reference

Based on [HiFloat4](https://github.com/global-computing-consortium/HiFloat4), with the following
updates (the rest of the algorithm is identical):

**NPU fake-quant kernel** (`hifloat4_cast_kernel.cpp`):
- Added tail masking: when N is not a multiple of BATCH=512, the kernel walks backward from
  BATCH-64 zeroing all tail elements (the original code only zeroed the last 64-block repeatedly,
  leaving stale data in between)
- Otherwise identical to the reference: 3-level shared scale (L1 E6M2 / L2 E1 8x / L3 E1 16x)
  + S1P2 mantissa, fp32 internal precision, bf16/E6M2 rounding via hardware `vconv`

**Python encode/decode** (`_hif4_encode_decode.py`):
- **`keepdim` + explicit `qdim`**: block max reduction uses `amax(dim=qdim, keepdim=True)` instead
  of `amax(dim=-1)` chains, matching the reference — all intermediate tensors share the same ndim,
  broadcasting is automatic, no manual permute needed
- After removing the block scale and computing lv2/lv3 scaling, **round to bf16 before quantizing
  to S1P2**
- When a 64-block contains NaN / +Inf / -Inf: **whole block → NaN** (block-poison). Scale byte 0
  is stored as `0xFF`; reconstruction restores the entire block to NaN

**Pack/unpack** (`hif4_pack_unpack.py`):
- 4-bit packing uses **low-bit-first**: element 0 in low nibble, element 1 in high nibble
- Scale packed as 4 bytes: `[E6M2, E1_8, E1_16_hi, E1_16_lo]`
- Non-finite blocks: scale byte 0 = `0xFF` → `hif4_bit_reconstruct` restores the whole block as NaN

## Status / notes

- **NPU kernel**: does the block reduction and quant in fp32 (bf16 / E6M2 rounding via the hardware
  `vconv`). The kernel body is guarded by `__DAV_C220_VEC__` (A2/A3, dav-2201 = C220, covering
  910B/910C); other arches compile to a no-op, so validate on the target NPU.
- **Pack/unpack**: done in Python, shares the same encode/decode math (`_hif4_encode_decode`) as the
  NPU kernel, guaranteeing bit-identical results.
