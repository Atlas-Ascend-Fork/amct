# amct_ops — AMCT NPU Custom Operators

## Introduction

**Positioning**:
- `amct_ops` is the NPU custom operator layer of AMCT, responsible for carrying hardware-level operators such as low-bit quantization and data type conversion that PyTorch / torch_npu has not yet covered.
- Unlike `amct_pytorch/` which focuses on quantization algorithms and compression process orchestration, `amct_ops` is closer to the underlying hardware implementation.

**Independent Advantages**:
- **Clear Responsibilities**: `amct_pytorch` calls operators through Python interfaces or `torch.ops.amct` without needing to pay attention to details such as Ascend C kernel, C++ extension, and CMake compilation
- **Independent Development**: Operators can be developed, built, and tested as independent modules, avoiding mixing kernel and build logic in the main algorithm directory
- **Flexible Extension**: When adding new low-bit types, quantization auxiliary operators, or adjusting NPU implementations, they do not interfere with each other

**Division of Labor with `amct_pytorch/`**:

| Dimension | `amct_pytorch/` | `amct_ops/` |
|------|-----------------|------------|
| Focus | Compression algorithms and process orchestration | Operator underlying implementation |
| Language | Python | Ascend C kernel & C++ binding & Python interface |
| Product | .tar.gz package (source code compression package) | wheel package (containing .so & Python interface) |
| Reusability | Bound to AMCT process | Independent PyTorch extension, not strongly dependent on AMCT main process |

**Usage**:
- Independently installable PyTorch extension package (wheel format)
- Supports two interfaces: `amct_ops.<op>` and `torch.ops.amct.<op>`

## Supported Operators

| Operator | Description | Python Interface |
|------|------|-------------|
| [`hifloat8_cast`](./hifloat8_cast/) | FP16 / BF16 ↔ HiFloat8 bidirectional conversion | `encode_to_hifloat8(x)`<br>`decode_from_hifloat8(x, dtype)` |

## Directory Structure

```
amct_ops/
├── hifloat8_cast/          # HiFloat8 conversion operator source code (kernel + binding + Python interface)
├── ops_build.sh            # Unified build entry
├── setup.py                # wheel packaging configuration
└── ops_init.py             # Copied as __init__.py during packaging, providing package interface and documentation
```

During the build process, directories such as `build/`, `dist/`, `staging/`, and `<op>/build/` will be generated. These are local build artifacts and do not need to be submitted.

## Build and Packaging

All operators are compiled and packaged as wheel through the unified build script in the amct_ops root directory.

### Dependency Requirements

| Dependency | Version |
|------|------|
| Python | >=3.9 |
| PyTorch | 2.7.1 or 2.1.0 (requires matching `torch_npu`) |
| GCC / CMake | ≥ 7.3 / ≥ 3.16 (recommended 3.20) |
| CANN (Toolkit & Ops) | ≥ 9.0.0 (requires pre-installed NPU driver / firmware) |

For complete environment deployment, please refer to [Quick Installation](../docs/zh/quick_install.md)

### Build Commands

```bash
cd amct_ops/
bash ops_build.sh [--soc <soc>] [<operator>]

# --soc specifies target platform (default ascend910b):
#   ascend910b    A2 (Ascend 910B1/B2/B3)
#   ascend910_93  A3 (Ascend 910_93)
#   ascend950     A5 (Ascend 950, requires CANN compiler support for dav-3510)

bash ops_build.sh                                # All operators, default platform
bash ops_build.sh --soc ascend910_93             # All operators, specified platform
bash ops_build.sh hifloat8_cast                  # Specified operator, default platform
bash ops_build.sh --soc ascend950 hifloat8_cast  # Specified operator, specified platform
```

### Build Artifacts

Build artifacts are located at `dist/amct_ops-1.0.0-cp*-cp*-linux_<arch>.whl`, where `<arch>` is automatically generated as `x86_64` or `aarch64` based on the build host, containing the Python package and compiled `.so` for all operators.

The two `cp*` in the wheel file name represent the Python implementation/version tag and ABI tag respectively. For example, `cp311-cp311` indicates that this wheel is for CPython 3.11 and depends on the CPython 3.11 ABI; `linux_<arch>` indicates the build host platform architecture.

### Installation

```bash
pip install dist/amct_ops-*.whl
```

### Usage Example

Two import methods after installation:

```python
# Method 1: Module import (has IDE completion and docstrings)
from amct_ops.hifloat8_cast import encode_to_hifloat8, decode_from_hifloat8

y = encode_to_hifloat8(x)                    # FP16/BF16 → uint8
z = decode_from_hifloat8(y)                  # → bfloat16 (default)
z = decode_from_hifloat8(y, torch.float16)   # → float16

# Method 2: torch.ops.amct (consistent with other NPU operator styles)
import amct_ops.hifloat8_cast                 # Trigger .so loading
torch.ops.amct.encode_to_hifloat8(x)
torch.ops.amct.decode_from_hifloat8(y, torch.float16)
```

#### Python Introspection

```python
import amct_ops
help(amct_ops)                               # View all submodules and interface lists

import amct_ops.hifloat8_cast
help(amct_ops.hifloat8_cast.encode_to_hifloat8)   # View single function signature and documentation
```

#### Known CMake Warnings

When building with pip-installed PyTorch, `find_package(Torch)` may output the following warning:

```text
static library kineto_LIBRARY-NOTFOUND not found.
```

This warning comes from PyTorch's built-in `TorchConfig.cmake`, indicating that the static library for the Kineto profiler was not found. Current `amct_ops` operators do not depend on PyTorch profiler / Kineto capabilities; as long as CMake configure, compilation, and linking succeed, this warning can be ignored.

## Adding New Operators

### Operator Directory Structure

Each operator occupies a separate subdirectory, with structure following `hifloat8_cast/`:

```
<new_operator>/
├── op_kernel/              # Ascend C kernel (.cpp + tiling.h)
├── op_extension/           # PyTorch C++ binding (host stub call + TORCH_LIBRARY registration)
├── python/<pkg>/           # Python interface (__init__.py)
├── CMakeLists.txt          # Independent compilation entry
└── README.md               # Operator documentation
```

After adding, no need to modify `ops_build.sh` or `setup.py`; the build script will automatically discover `<op>/python/<pkg>/` directories and package them. Formal testing should be placed under `tests/amct_ops/` to avoid putting build artifacts, performance scripts, or comparison tools that depend on additional reference implementations into the operator source directory.

### Namespace Constraints

**All operators must be registered in the `amct` namespace**—consistent with the `amct_ops` package name, making it easy for callers to distinguish AMCT custom operators from `torch_npu` upstream operators.

- C++ side: `TORCH_LIBRARY_FRAGMENT(amct, m)` + `TORCH_LIBRARY_IMPL(amct, PrivateUse1, m)`
- Python side: `torch.ops.amct.<operator_name>` or module import `from amct_ops.xxx import ...`
- Operator names must be unique within `amct`. Before adding, check whether `torch.ops.amct` already has an operator with the same name.

## References
[cann/ops-nn Operator Development Guide](https://gitcode.com/cann/ops-nn/blob/master/docs/zh/develop/torch_extension_develop_guide.md)