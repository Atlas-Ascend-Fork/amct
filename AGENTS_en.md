# AGENTS.md

This file provides guidance for agents working in this code repository.

## Project Overview

AMCT (Ascend Model Compression Toolkit) is a deep learning model compression toolkit with affinity for Ascend AI processors. It provides multiple model quantization compression features (W8A8, W4A8, W4A4, and so on). After compression, the model size decreases. Deploying the model on Ascend AI processors enables low-bit operations and improves computation efficiency.

Main features:

- **Quantization compression**: Supports multiple quantization algorithms (AWQ, GPTQ, SmoothQuant, Minmax, and so on)
- **HiFloat8**: Supports HiFloat8/FP8/FP4 data quantization and quantile quantization
- **Structured pruning**: Prunes dense FFN intermediate dimensions / CNN channels / MoE experts in eager mode
- **Graph compression**: Compression optimization based on compute graphs (for example, tensor decomposition)
- **NPU operators**: Implementations of quantization-related Ascend NPU operators
- **Experimental features**: Quantization support for cutting-edge models such as DeepSeekV3.2/V4

## Agent Skills

This repository has Agent Skills infrastructure built in (`.agents/`, automatically projected to each AI client through git-tracked symlinks, ready to use after cloning). It encapsulates domain-specific capabilities with each skill as a unit, triggered by natural language and executed autonomously by agents. Currently built-in skills include **quant-workflow** (the main entry for orchestrating the entire LLM quantization workflow), **amct-ops-dev** (NPU operator development), and `gitcode-pr`/`gitcode-issue` (repository collaboration operations). More skills will be added as needed. For details, see [.agents/README.md](.agents/README.md) (quick start) and [.agents/docs/architecture.md](.agents/docs/architecture.md) (architecture design).

## Build Commands

### Basic Build
>
> ⚠️ Note: Running `bash build.sh` without any packaging options **does not produce any distribution package** — the script
> performs an actual build only when `--torch`, `--pkg`, or `-u/--utest` is specified. To build an installable
> package, you must use `--torch` or `--pkg`.

```bash
# View build options
bash build.sh --help

# Build only the amct_pytorch package (output in build_out/, most commonly used)
bash build.sh --torch

# Build the full amct package (amct_pytorch + graph, output in build_out/)
bash build.sh --pkg

# Build the amct_pytorch package with experimental features (including amct_pytorch/experimental/)
bash build.sh --torch --experimental

# Build the full amct package with experimental features
bash build.sh --pkg --experimental

# Build in Debug mode (must be used together with --torch or --pkg)
bash build.sh --torch --build-type=Debug

# Build with a specified number of threads
bash build.sh --torch -j8

# Specify the third-party library path
bash build.sh --torch --cann_3rd_lib_path=/path/to/3rd_party
```

### Unit Tests

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

# Install the pre-commit code checking tool
pip install pre-commit
pre-commit install
```

### Install AMCT into the Python Environment

Build commands such as `bash build.sh --torch` / `--pkg` only generate distribution packages, with artifacts in the `build_out/` directory.
They are not automatically installed into the current Python environment. To make `import amct_pytorch` available, you must install the build artifacts with pip:

```bash
# 1. Build the amct_pytorch distribution package (output in build_out/)
bash build.sh --torch

# 2. Install the build artifacts
#    Obtain ${version} from the file name in the build_out/ directory, for example amct_pytorch-1.1.0-py3-none-linux_aarch64.tar.gz
#    ${arch}    is the CPU architecture, for example x86_64 or aarch64
pip3 install build_out/amct_pytorch-${version}-py3-none-linux_${arch}.tar.gz --user

# 3. Verify the installation
python3 -c "import amct_pytorch as amct; print('successfully installed AMCT')"
```

> ⚠️ Note: If the pip version is greater than 25.2, you need to add `--no-build-isolation` to the installation command.
> Otherwise, `ModuleNotFoundError: No module named 'torch'` may occur.
>
> In addition, AMCT requires CANN (Toolkit & Ops) ≥ 8.5.0 and the corresponding NPU drivers/firmware to run.
> For complete environment deployment, see `docs/zh/quick_install.md`.

## Directory Structure

| Directory | Description |
| ------ | ------ |
| `amct_pytorch/` | Core source code of PyTorch quantization compression |
| `amct_pytorch/experimental/` | Experimental features (HiFloat8, DeepSeek, and so on) |
| `amct_pytorch/classic/graph_based/` | Compression optimization based on compute graphs (for example, tensor decomposition) |
| `amct_pytorch/pruning/` | Structured pruning in eager mode (dense FFN / CNN channels / MoE experts) |
| `amct_ops/` | Code of quantization-related Ascend NPU operators |
| `tests/` | Unit tests |
| `examples/` | End-to-end sample development and invocation examples |
| `docs/` | Tool documentation (concept introductions, API documentation, algorithm introductions, and so on) |
| `cmake/` | CMake build configuration |
| `build.sh` | Project build script |
| `setup.py` | Python packaging entry point |
| `requirements.txt` | Python dependency packages |
| `.clang-format` | C/C++ code formatting configuration |

## Development Guidelines

### GitCode PR/Issue Operations

@.claude/skills/gitcode-pr/SKILL.md
@.claude/skills/gitcode-issue/SKILL.md

If the preceding skills are missing, install them automatically as described in @.claude/skills/default-skills/SKILL.md.

### Code Style

- Follow the Google open source code style guide (based on `.clang-format`)
- Use `clang-format` to format C/C++ code, with a line width limit of 120 characters and 4-space indentation
- Use `pre-commit` to automatically check code style before each commit
- Use `ruff` to check and format Python code (following PEP8)

### Contribution Process

- Simple bug fixes: Submit a PR directly
- New features/API changes: Discuss the solution through an Issue first, and submit a PR after reaching consensus
- A PR needs to be associated with the related Issue and include the feature code, documentation improvements, and README

### Issue / PR Templates

When submitting an Issue or PR, you must fill in the content by referring to the corresponding template in this repository:

- **PR template**: `.gitcode/PULL_REQUEST_TEMPLATE.zh-CN.md`
- **Issue templates** (located in `.gitcode/ISSUE_TEMPLATE/`, select by type):
  - `bug_report.yaml` — Bug report
  - `feature_request.yaml` — New feature request
  - `documentation.yaml` — Documentation issue
  - `question.yaml` — Usage consultation/questions
  - `experience_request.yaml` — Experience request

### License

- Apache 2.0 license; copyright information must be included in the code

## Phrases

Use Chinese
