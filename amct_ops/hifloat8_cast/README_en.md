# hifloat8_cast — HiFloat8 Data Type Conversion Operator

HiFloat8 data type conversion operator, supporting FP16/BF16 ↔ HiFloat8 bidirectional conversion.

## Feature Introduction

Supports 4 conversion modes, all implemented by `KernelHiFloat8CastLut` at the underlying level:

| castMode | Conversion | Underlying LUT |
|----------|------|---------|
| 0 | FP16 → HiFloat8 | 32768-entry half-space UB-LUT (32 KB) |
| 1 | BF16 → HiFloat8 | 32768-entry half-space UB-LUT (32 KB) |
| 2 | HiFloat8 → FP16 | 256-entry UB-LUT (512 B) |
| 3 | HiFloat8 → BF16 | 256-entry UB-LUT (512 B) |

> Python interface automatically selects castMode based on input dtype, no manual specification needed.

## Interface Description

### encode_to_hifloat8(x: Tensor) -> Tensor

Encode FP16/BF16 tensors to HiFloat8 format.

**Parameters**:
- `x`: Input tensor (NPU device), dtype is `torch.float16` or `torch.bfloat16`, any non-empty shape

**Returns**:
- `torch.uint8` tensor, same shape as input

**Exceptions**:
- `RuntimeError`: dtype is not float16 / bfloat16, or input is not on NPU

**Example**:
```python
y = encode_to_hifloat8(x_fp16)
y = encode_to_hifloat8(x_bf16)
```

---

### decode_from_hifloat8(x: Tensor, dtype: dtype = torch.bfloat16) -> Tensor

Decode HiFloat8 format to FP16/BF16 tensors.

**Parameters**:
- `x`: HiFloat8 encoded tensor (`torch.uint8`), any non-empty shape
- `dtype`: Output type, default `torch.bfloat16`, supports `float16` / `bfloat16`

**Returns**:
- Decoded tensor, same shape as input

**Exceptions**:
- `RuntimeError`: Input dtype is not uint8, or output dtype is not supported

**Example**:
```python
z = decode_from_hifloat8(y)                   # Default output BF16
z = decode_from_hifloat8(y, torch.float16)    # Specify output FP16
```

## Directory Structure

```
hifloat8_cast/
├── op_kernel/
│   ├── hifloat8_cast_kernel.cpp   # Device-side kernel (LUT encode + decode, half-space LUT optimization)
│   └── hifloat8_cast_tiling.h     # TilingData structure definition (tileLength filled by host runtime)
├── op_extension/
│   ├── hifloat8_cast_torch.cpp    # PyTorch host implementation (CPU-side LUT pre-computation and caching, calls ASC-generated host stub)
│   ├── ops.h                      # PyTorch C++ extension function declaration
│   └── register.cpp               # TORCH_LIBRARY registration
├── python/
│   └── hifloat8_cast/
│       ├── __init__.py            # Python package entry (load .so, register operator)
│       └── ops.py                 # Python interface implementation
├── CMakeLists.txt                 # CMake build configuration
└── README.md                      # Operator documentation
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
bash ops_build.sh [--soc <soc>] [hifloat8_cast]
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
bash ops_build.sh hifloat8_cast                     # Specified operator, default platform
bash ops_build.sh --soc ascend950 hifloat8_cast     # Specified operator, specified platform
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
cd amct_ops/hifloat8_cast
source $ASCEND_HOME_PATH/set_env.sh
mkdir -p build && cd build

cmake .. -DNPU_ARCH=dav-2201   # A2 / A3 (default)
# cmake .. -DNPU_ARCH=dav-3510  # A5

make -j8
```

Build artifact: `build/libhifloat8_cast_ops.so`

#### Known CMake Warnings

When building with pip-installed PyTorch, `find_package(Torch)` may output the following warning:

```text
static library kineto_LIBRARY-NOTFOUND not found.
```

This warning comes from PyTorch's built-in `TorchConfig.cmake`, indicating that the static library for Kineto profiler was not found. `hifloat8_cast` does not depend on PyTorch profiler / Kineto capabilities; as long as CMake configure, compilation, and linking succeed, this warning can be ignored.

## Usage Example

```python
import torch
import torch_npu
from amct_ops.hifloat8_cast import encode_to_hifloat8, decode_from_hifloat8

# Encode (FP16/BF16 → HiFloat8)
x = torch.randn(1024, 256, dtype=torch.bfloat16, device='npu')
y = encode_to_hifloat8(x)      # shape [1024, 256], dtype uint8

# Decode (HiFloat8 → FP16/BF16)
z = decode_from_hifloat8(y, torch.bfloat16)  # shape [1024, 256], dtype bfloat16
```

Can also call underlying operators directly through `torch.ops.amct`:

```python
y = torch.ops.amct.encode_to_hifloat8(x)
z = torch.ops.amct.decode_from_hifloat8(y, torch.bfloat16)
```

## Performance Verification

**Test Platform**: Ascend 910B3 (A2, ascend910b), CANN 9.0.0  
**Throughput Definition**: (Input bytes + Output bytes) / Elapsed time (100 iteration average, 10 warm-up, NPU synchronize timing)  
**Recommended Scenario**: NPU advantage significant when data volume ≥ 256K

### BF16 ↔ HiFloat8

| Data Size | Encode (ms) | Throughput (MB/s) | Decode (ms) | Throughput (MB/s) |
|---------|------------|------------|------------|------------|
| 1K      | 0.207      | 14.9       | 0.228      | 13.5       |
| 4K      | 0.215      | 57.2       | 0.219      | 56.0       |
| 16K     | 0.215      | 228.4      | 0.255      | 192.9      |
| 64K     | 0.215      | 914.0      | 0.290      | 677.6      |
| 256K    | 0.289      | 2723.9     | 0.293      | 2679.8     |
| 1M      | 0.867      | 3627.2     | 0.863      | 3645.7     |
| 4M      | 3.133      | 4016.7     | 3.111      | 4044.3     |
| 16M     | 12.236     | 4113.4     | 12.152     | 4141.7     |

### FP16 ↔ HiFloat8

| Data Size | Encode (ms) | Throughput (MB/s) | Decode (ms) | Throughput (MB/s) |
|---------|------------|------------|------------|------------|
| 1K      | 0.208      | 14.8       | 0.224      | 13.7       |
| 4K      | 0.216      | 56.9       | 0.223      | 55.1       |
| 16K     | 0.215      | 228.6      | 0.257      | 191.6      |
| 64K     | 0.217      | 904.1      | 0.289      | 680.9      |
| 256K    | 0.288      | 2734.8     | 0.293      | 2687.8     |
| 1M      | 0.867      | 3628.1     | 0.863      | 3645.2     |
| 4M      | 3.130      | 4019.6     | 3.121      | 4032.2     |
| 16M     | 12.236     | 4113.4     | 12.148     | 4143.1     |

### Roundtrip (Encode + Decode)

| Data Size | FP16 (ms) | Throughput (MB/s) | BF16 (ms) | Throughput (MB/s) |
|---------|----------|------------|----------|------------|
| 1K      | 0.389    | 10.5       | 0.392    | 10.5       |
| 4K      | 0.392    | 41.8       | 0.393    | 41.7       |
| 16K     | 0.394    | 166.2      | 0.390    | 168.1      |
| 64K     | 0.388    | 674.8      | 0.389    | 673.7      |
| 256K    | 0.548    | 1911.9     | 0.554    | 1891.6     |
| 1M      | 1.714    | 2446.7     | 1.716    | 2444.9     |
| 4M      | 6.246    | 2686.2     | 6.249    | 2684.8     |
| 16M     | 24.393   | 2751.1     | 24.401   | 2750.3     |

> Small data (< 256K) throughput is relatively low, but has been optimized through core count priority strategy to avoid excessive core loading LUT overhead.  
> Large data (≥ 4M) encode/decode throughput exceeds 4 GB/s, approaching HBM bandwidth limit.  
> tileLength is dynamically calculated by runtime based on actual UB size and core count on the platform.

## Accuracy Verification

Accuracy verification is performed through the following methods:
- **NPU API comprehensive testing**: Type inference, boundary values, special values, subnormal numbers, full 256 HiFloat8 decode completeness, see `tests/amct_ops/test_hifloat8_cast.py`
- **Boundary/Special values**: ±0, ±Inf, NaN, max value, min subnormal number and other boundary cases

| Test Scenario | Result |
|---------|------|
| Random FP16/BF16 roundtrip | ✓ Relative error within HiFloat8 precision range |
| Boundary value roundtrip | ✓ Covers 0, positive/negative numbers, decimals, larger values |
| Special values: ±0 / ±Inf / NaN | ✓ Encoded bytes and decoded semantics conform to specification |
| Full 256 HiF8 value decode (FP16 + BF16) | ✓ Only 0x80 is NaN encoding |

> Note: FP16 subnormal encoding/decoding (including HiFloat8 subnormal values mapping to FP16 subnormal values) has been fully verified correctly.

## Test Method

```bash
# Execute in repository root directory. First build amct_ops, then run tests through staging.
bash amct_ops/ops_build.sh hifloat8_cast

# Python API comprehensive testing (type inference, boundary values, special values, subnormal numbers)
PYTHONPATH=amct_ops/staging python3 -m unittest tests.amct_ops.test_hifloat8_cast
```

Can also install wheel first then run tests:

```bash
pip install amct_ops/dist/amct_ops-*.whl
python3 -m unittest tests.amct_ops.test_hifloat8_cast
```

For more test execution instructions, see `tests/amct_ops/README.md`.