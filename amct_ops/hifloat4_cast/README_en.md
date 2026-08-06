# hifloat4_cast — HiFloat4 Data Type Conversion Operator

HiFloat4 data type conversion operator, supporting FP16/BF16 → HiFloat4 → FP16/BF16 fake-quant simulation (64-element block scaling, S1P2 mantissa).

## Feature Introduction

HiFloat4 (hifx4) is a 4-bit block-scaling float format: per element **S1P2** (1 sign + 2-bit mantissa, 8 magnitudes) plus a per-64-element **3-level scale** (L1 `scale_factor` E6M2 over the 64-block, L2 over each 8, L3 over each 4).

At the underlying level, `hifx_kernel` (fp32) / `hifx_kernel_bf16` (bf16) performs a single FP→HiF4→FP simulation round trip. Pack/unpack is done in Python (`hif4_pack` / `hif4_unpack` in `hifp_impl.py`); fake-quant and pack share the same encode/decode logic, guaranteeing the evaluation path and deployed weight are always consistent.

## Interface Description

### hifloat4_fake_quant(x: Tensor, qdim: int = -1) -> Tensor

Perform a FP→HiF4→FP simulation round trip along `qdim` in 64-element shared-scale blocks.

**Parameters**:
- `x`: Input tensor (NPU device), dtype is `torch.float16` or `torch.bfloat16`, any non-empty shape
- `qdim`: Block dimension, default `-1` (last dimension)

**Returns**:
- Tensor of same shape/dtype (with HiFloat4 quantization error injected)

**Exceptions**:
- `RuntimeError`: dtype is not float16 / bfloat16, or input is not on NPU

**Example**:
```python
from amct_ops.hifloat4_cast import hifloat4_fake_quant
y = hifloat4_fake_quant(x)                      # blocks along the last dim
y = hifloat4_fake_quant(w, qdim=1)              # Linear weight [out, in] along in
y = torch.ops.amct.hifloat4_fake_quant(x, -1)   # equivalent torch.ops call
```

The host moves `qdim` to the last axis, zero-pads to a multiple of 64 (zeros don't affect the block max), calls the kernel, then slices off the pad and restores the dim order.

## Directory Structure

```
hifloat4_cast/
├── op_kernel/
│   ├── hifloat4_cast_kernel.cpp   # Device-side kernel (FP→HiF4→FP simulation round trip)
│   └── tensorutils.h              # Tensor utility functions (padding, slicing, dim transforms)
├── op_extension/
│   ├── hifloat4_cast_torch.cpp    # PyTorch host implementation (qdim preprocessing, kernel call)
│   ├── ops.h                      # PyTorch C++ extension function declaration
│   └── register.cpp               # TORCH_LIBRARY registration
├── python/
│   └── hifloat4_cast/
│       ├── __init__.py            # Python package entry (load .so, register operator)
│       └── ops.py                 # Python interface implementation
├── CMakeLists.txt                 # CMake build configuration
├── README.md                      # Operator documentation (Chinese)
└── README_en.md                   # Operator documentation (English)
```

## Packed Format

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

## Environment Dependencies

| SOC | Platform | Nominal UB / core | CANN Measured Available UB |
|-----|------|--------------|----------------|
| `ascend910b` | Atlas A2 series products | 256 KB | ~192 KB |
| `ascend910_93` | Atlas A3 series products | 512 KB | ~384 KB |
| `ascend950` | Ascend950PR/Ascend950DT | 512 KB+ | — |

> CANN runtime reserves part of UB. `tileLength` is calculated by querying actual available amount using `GetCoreMemSize()` at each call, no manual specification needed.

- CANN 9.0.0
- Python ≥ 3.9
- PyTorch + torch_npu (matching corresponding CANN version)

## Compilation

### Method 1: Unified Packaging (Recommended)

```bash
cd amct_ops
bash ops_build.sh [--soc <soc>] [hifloat4_cast]
```

| `--soc` | Platform | `--npu-arch` | Default |
|---------|------|-------------|------|
| `ascend910b` | A2 (910B1/B2/B3, UB 256 KB) | `dav-2201` | ✓ |
| `ascend910_93` | A3 (910_93, UB 512 KB) | `dav-2201` | |
| `ascend950` | A5 (UB 512 KB+) | `dav-3510` | |

> A2/A3 share the same ISA (`dav-2201`), build artifacts are the same; UB size differences are distinguished by runtime platform API, automatically selecting optimal `tileLength`.
> A5 build requires current CANN compiler to support `dav-3510`. If `bisheng` reports `Unsupported NPU architecture or soc`, need to switch to a CANN compilation environment that supports A5 targets.

```bash
bash ops_build.sh                                   # All operators, default platform
bash ops_build.sh --soc ascend910_93                # All operators, specified platform
bash ops_build.sh hifloat4_cast                     # Specified operator, default platform
bash ops_build.sh --soc ascend950 hifloat4_cast     # Specified operator, specified platform
pip install dist/amct_ops-*.whl
```

### Method 2: Single Operator Independent Compilation (Development Debugging)

Independent compilation for a single operator, suitable for development and debugging scenarios.

Specify target platform through `-DNPU_ARCH` (default A2):

| Platform | `-DNPU_ARCH` |
|------|-------------|
| Atlas A2 series products | `dav-2201` (default) |
| Atlas A3 series products | `dav-2201` (same as A2) |
| Ascend950PR/Ascend950DT | `dav-3510` |

```bash
cd amct_ops/hifloat4_cast
source $ASCEND_HOME_PATH/set_env.sh
mkdir -p build && cd build

cmake .. -DNPU_ARCH=dav-2201   # A2 / A3 (default)
# cmake .. -DNPU_ARCH=dav-3510  # A5

make -j8
```

Build artifact: `build/libhifloat4_cast_ops.so`

#### Known CMake Warnings

When building with pip-installed PyTorch, `find_package(Torch)` may output the following warning:

```text
static library kineto_LIBRARY-NOTFOUND not found.
```

This warning comes from PyTorch's built-in `TorchConfig.cmake`, indicating that the static library for Kineto profiler was not found. `hifloat4_cast` does not depend on PyTorch profiler / Kineto capabilities; as long as CMake configure, compilation, and linking succeed, this warning can be ignored.

## Usage Example

```python
import torch
import torch_npu
from amct_ops.hifloat4_cast import hifloat4_fake_quant

# Fake-quant simulation (FP16/BF16 → HiF4 → FP16/BF16)
x = torch.randn(1024, 256, dtype=torch.bfloat16, device='npu')
y = hifloat4_fake_quant(x)                     # shape [1024, 256], dtype bfloat16

# Specify block dimension
w = torch.randn(4096, 1024, dtype=torch.float16, device='npu')
y = hifloat4_fake_quant(w, qdim=1)             # blocks along in_features
```

Can also call underlying operators directly through `torch.ops.amct`:

```python
y = torch.ops.amct.hifloat4_fake_quant(x, -1)
```

## Accuracy Verification

Accuracy verification is performed through the following methods:
- **NPU API comprehensive testing**: Type inference, boundary values, special values, subnormal numbers, see `tests/amct_ops/test_hifloat4_cast.py`
- **kernel vs CPU reference**: Bit-exact comparison between NPU kernel and Python CPU implementation (`hif4_pack` / `hif4_unpack`), ~0 error

| Test Scenario | Result |
|---------|------|
| Random FP16/BF16 fake-quant roundtrip | ✓ Relative error within HiFloat4 precision range |
| pack/unpack round-trip | ✓ Shape consistent, bit-exact recovery |
| NPU kernel vs CPU fallback | ✓ Bit-exact match (max abs diff < 1e-6) |

## Test Method

```bash
# Execute in repository root directory. First build amct_ops, then run tests through staging.
bash amct_ops/ops_build.sh hifloat4_cast

# Python API comprehensive testing (type inference, boundary values, special values, subnormal numbers)
PYTHONPATH=amct_ops/staging python3 -m unittest tests.amct_ops.test_hifloat4_cast
```

Can also install wheel first then run tests:

```bash
pip install amct_ops/dist/amct_ops-*.whl
python3 -m unittest tests.amct_ops.test_hifloat4_cast
```

For more test execution instructions, see `tests/amct_ops/README.md`.
