# MXFP4 Ascend-C Accelerated Operator (experimental)

An Ascend-C custom kernel that implements the MXFP4 (Microscaling FP4 E2M1) fake-quant operator on Ascend NPU. The directory layout is aligned with `amct_ops/hifloat8_cast` (`op_kernel` / `op_extension` / `python`). The implementation remains under `amct_pytorch/experimental/fakequant/` (experimental; not part of `amct_ops`).

Compared with the torch_npu software path: **3.3x speedup** (large matrices), **18x speedup** (small matrices). Correctness is bit-exact against the PyTorch reference implementation.

## Runtime Environment

| Component | Version |
|-----------|---------|
| Hardware | Ascend 910B3 / compatible SoC |
| CANN | 8.2.RC1+ |
| Python | 3.10 (aarch64) |
| PyTorch | 2.6.0 |
| torch_npu | 2.6.0.post4 |

> The open-source repository does not ship prebuilt `.so` files. Compile locally against your SoC / CANN / Python ABI.

## Directory Structure

```
mxfp4_ascendc/
├── op_kernel/
│   ├── mxfp4_kernel.cpp      # Ascend-C device kernel
│   └── mxfp4_tiling.h        # Host/device shared tiling constants and structs
├── op_extension/
│   ├── mxfp4_torch.cpp       # PyTorch host: tiling + ACLRT_LAUNCH_KERNEL
│   ├── ops.h                 # C++ host API declaration (namespace AscendKernel)
│   └── register.cpp          # TORCH_LIBRARY_FRAGMENT(amct, ...) + Meta
├── python/
│   └── mxfp4/
│       ├── __init__.py       # load .so, self-check, re-export
│       └── ops.py            # thin Python wrapper (pad / dtype)
├── reference/
│   └── mxfp4_ref.py          # pure PyTorch reference implementation
├── CMakeLists.txt            # build entry
├── build.sh                  # one-shot build and stage .so into python/mxfp4/
├── tests/                    # correctness / inv_scale / benchmark
├── README.md
└── README_en.md
```

## Build from Source

```bash
cd /path/to/mxfp4_ascendc

# Build (takes a few minutes); on success, .so files are copied to python/mxfp4/
bash build.sh

# Correctness + performance tests
python tests/test_mxfp4.py
# python tests/test_inv_scale.py   # inv_scale parameter correctness
# python tests/bench_qdq.py       # extra performance comparison
```

Specify SoC:

```bash
SOC_VERSION=Ascend910_9392 bash build.sh
```

## Quick Start

```python
import sys
sys.path.insert(0, "/path/to/mxfp4_ascendc/python")

from mxfp4 import quant_dequant_mxfp4

x_npu = x.npu()
result = quant_dequant_mxfp4(x_npu)

# Equivalent low-level call (input must already be float32 flat, numel a multiple of 32)
# result = torch.ops.amct.quant_dequant_mxfp4(x_flat, 1.0)
```

### API

```python
quant_dequant_mxfp4(
    x: torch.Tensor,                 # any shape, float32 recommended, on NPU
    block_size: int = 32,            # quantization block width (must be 32)
    inv_scale_factor_scale: float = 1.0,
) -> torch.Tensor                    # same shape / dtype / device
```

AIV core count is queried at runtime on the host via `PlatformAscendC::GetCoreNumAiv()`; you do not need to set it manually.

For training use (differentiable fake quant with STE), see `../mxfp4_qat/`. Its `backend="auto"` automatically calls this operator on NPU tensors.

### Performance

| Shape | torch_npu | Ascend-C | Speedup |
|-------|-----------|----------|---------|
| (64, 4096) | 0.69 ms | 0.038 ms | **18.1x** |
| (256, 4096) | 0.72 ms | 0.059 ms | **12.3x** |
| (1024, 4096) | 0.72 ms | 0.219 ms | **3.28x** |
