# AGENTS.md

This file provides guidance for agents working in this code repository.

## Project Overview

AMCT (Ascend Model Compression Toolkit) is a deep learning model compression toolkit tailored for Ascend AI processors, providing multiple model quantization compression features (W8A16, W8A8, W4A16, and so on). After compression, the model size becomes smaller, and deployment on Ascend AI processors enables low-bit operations, improving computational efficiency.

Main Features:
- **Quantization Compression**: Supports multiple quantization algorithms (AWQ, GPTQ, SmoothQuant, Minmax, and so on)
- **HiFloat8**: Supports HiFloat8/FP8/FP4 data quantization and quantile quantization
- **Graph Compression**: Graph-based compression optimization (tensor decomposition and so on)
- **NPU Operators**: Quantization-related Ascend NPU operator implementations
- **Experimental Features**: Quantization support for frontier models such as DeepSeekV3.2/V4

## Build Commands

### Basic Build
```bash
# Build (Release mode)
bash build.sh

# View build options
bash build.sh --help

# Build package (generate distributable tar.gz)
bash build.sh --pkg

# Debug mode build
bash build.sh --build-type=Debug

# Build with specified thread count
bash build.sh -j8

# Build with specified third-party library path
bash build.sh --cann_3rd_lib_path=/path/to/3rd_party
```

### Unit Testing
```bash
# Build and run unit tests
bash build.sh -u

# Enable coverage
bash build.sh -u --cov

# Enable AddressSanitizer
bash build.sh --asan
```

### Python Environment Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Install pre-commit code checking tool
pip install pre-commit
pre-commit install
```

## Directory Structure

| Directory | Purpose |
|------|------|
| `amct_pytorch/` | PyTorch quantization compression core source code |
| `amct_pytorch/experimental/` | Experimental features (HiFloat8, DeepSeek, and so on) |
| `amct_pytorch/graph_based_compression/` | Graph-based compression optimization (tensor decomposition and so on) |
| `npu_ops/` | Quantization-related Ascend NPU operator code |
| `tests/` | Unit tests |
| `examples/` | End-to-end sample development and usage examples |
| `docs/` | Tool documentation (concept introduction, API documentation, algorithm introduction, and so on) |
| `cmake/` | CMake build configuration |
| `build.sh` | Project compilation script |
| `setup.py` | Python package packaging entry |
| `requirements.txt` | Python dependency packages |
| `.clang-format` | C/C++ code formatting configuration |

## Development Standards

### GitCode PR/Issue Operations
@.claude/skills/default-skills/SKILL.md

### Code Style
- Follow Google open source code standards (based on `.clang-format`)
- Use `clang-format` to format C/C++ code, line width limit 120 characters, 4-space indentation
- Use `pre-commit` to automatically check code standards before submission
- Python code follows PEP8 standards

### Contribution Process
- Simple bug fixes: Submit PR directly
- New features/interface changes: First discuss the solution through an Issue, reach consensus, then submit PR
- PR needs to be associated with the relevant Issue and include feature code, optimization documentation, and README

### License
- Apache 2.0 agreement, need to mark copyright information in code

## Phrases
Use Chinese