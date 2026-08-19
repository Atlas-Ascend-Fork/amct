# hifloat4_cast — HiFloat4 Data Type Conversion Operator

HiFloat4 data type conversion operator, supporting FP16/BF16 → HiFloat4 → FP16/BF16 fake-quant simulation (64-element block scaling, S1P2 mantissa).

## Feature Introduction

HiFloat4 (HiF4) is a 4-bit block-scaling float format: each element uses an S1P2 representation (1-bit sign + 2-bit mantissa, 8 magnitudes in total) and a per-64-element three-level scale (L1 `scale_factor` in E6M2 over the 64-block, L2 per 8 elements, L3 per 4 elements).

This operator only implements fake-quant simulation: a single FP → HiF4 → FP quantization and dequantization pass. The output is a floating-point tensor with the same shape and dtype as the input.

The corresponding underlying kernel is invoked based on the input dtype; both kernels compute internally in fp32:

| Input dtype | Underlying kernel | Internal computation |
|-------------|-------------------|---------------------|
| BF16 | `hifx_kernel_bf16` | fp32: converted to fp32 after loading, bf16 rounding only at the specified computation points (scale factor, reciprocal, in-group value), converted back to bf16 on output |
| FP16 | `hifx_kernel` | fp32: converted to fp32 on the host side first, output converted back to fp16 after computation |

## Interface Description

### hifloat4_fake_quant(x: Tensor, qdim: int = -1) -> Tensor

Blocks the FP16/BF16 tensor along `qdim`, with one shared scale per 64 elements, and performs a single FP → HiF4 → FP fake-quant simulation.

Parameters:
- `x`: Input tensor on an NPU device, dtype is `torch.float16` or `torch.bfloat16`, any non-empty shape
- `qdim`: Block dimension, default `-1` (the last dimension)

Returns:
- Tensor with the same shape and dtype, containing HiFloat4 quantization error

Exceptions:
- `RuntimeError`: dtype is not float16 / bfloat16, or input tensor is not on an NPU device
- `RuntimeError`: the `qdim` length is not a multiple of 64

Example:
```python
from amct_ops.hifloat4_cast import hifloat4_fake_quant
y = hifloat4_fake_quant(x)                      # blocks along the last dim
y = hifloat4_fake_quant(w, qdim=1)              # Linear weight [out, in] along in
y = torch.ops.amct.hifloat4_fake_quant(x, -1)   # equivalent torch.ops call
```

The host moves `qdim` to the last axis; its length must be a multiple of 64 (otherwise the op raises). It then zero-pads to a multiple of 512 (zeros don't affect the block max), calls the kernel, and slices back to the original length, restoring the dimension order.

## Directory Structure

```
hifloat4_cast/
├── op_kernel/
│   ├── hifloat4_cast_kernel.cpp   # Device-side kernel (FP→HiF4→FP simulation round trip)
│   └── tensorutils.h              # Ascend C utility functions (kernel-side tensor operations)
├── op_extension/
│   ├── hifloat4_cast_torch.cpp    # PyTorch host implementation (qdim preprocessing and kernel call)
│   ├── ops.h                      # PyTorch C++ extension function declaration
│   └── register.cpp               # TORCH_LIBRARY registration
├── python/
│   └── hifloat4_cast/
│       ├── __init__.py            # Python package entry (loads .so and registers the operator)
│       └── ops.py                 # Python interface implementation
├── CMakeLists.txt                 # CMake build configuration
├── README.md                      # Operator documentation (Chinese)
└── README_en.md                   # Operator documentation (English)
```

## Environment Dependencies

- CANN 9.0.0
- Python ≥ 3.9
- PyTorch + torch_npu (matching the corresponding CANN version)

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

> A2/A3 share the same ISA (`dav-2201`), build artifacts are the same.
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

The build artifact is located at:

```text
build/libhifloat4_cast_ops.so
```

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

The underlying operator can also be called directly through `torch.ops.amct`:

```python
y = torch.ops.amct.hifloat4_fake_quant(x, -1)
```

## Accuracy Verification

Accuracy verification methods are as follows, see `tests/amct_ops/test_hifloat4_cast.py`:

| Test Scenario | Result |
| --- | --- |
| Random BF16/FP16 fake-quant round-trip simulation | ✓ Element-wise match with the CPU reference implementation (max abs diff < 1e-6) |
| Non-64-aligned dims | ✓ Raises RuntimeError (fake-quant and pack behave the same) |
| Zero input | ✓ No NaN, output all zeros |

## Test Method

```bash
# Execute in repository root directory. First build amct_ops, then run tests through staging.
bash amct_ops/ops_build.sh hifloat4_cast

# NPU kernel vs CPU reference implementation element-wise consistency verification + shape regression tests
PYTHONPATH=amct_ops/staging python3 -m unittest tests.amct_ops.test_hifloat4_cast
```

Can also install wheel first then run tests:

```bash
pip install amct_ops/dist/amct_ops-*.whl
python3 -m unittest tests.amct_ops.test_hifloat4_cast
```

For more test execution instructions, see `tests/amct_ops/README.md`.
