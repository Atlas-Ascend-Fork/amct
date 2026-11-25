# Contributing Guide

This project welcomes developers to experience and contribute. Before participating in community contributions, please refer to [cann-community](https://gitcode.com/cann/community) to learn the [code of conduct](https://gitcode.com/cann/community/blob/master/contributor/code-of-conduct.md), complete the [CLA agreement signing](https://gitcode.com/cann/infrastructure/blob/master/docs/cla/cla使用指南.md), and understand the contribution process of source repositories. Beyond code contributions, the community also welcomes contributions in various forms such as meetings, emails, and testing. For details, see the community [contribution overview](https://gitcode.com/cann/community#参与贡献).

If you wish to make advanced contributions to the community beyond this repository (such as creating a new SIG, becoming a core contributor, creating a new repository, or introducing open-source third-party components), please refer to the [CANN Community Contribution Guide](https://gitcode.com/cann/community#进阶贡献).

## Contribution Scenarios

Developer contribution scenarios mainly include:

- [Contributing New Features](#1-contributing-new-features)
- [Bug Fix](#2-bug-fix)
- [Documentation Correction](#3-documentation-correction)
- [Experience Feedback](#4-experience-feedback)
- [Help Solve Others' Issues](#5-help-solve-others-issues)

---

## 1. Contributing New Features

Thank you for participating in AMCT development and bringing more value to model compression! The complete contribution process is as follows:

> **Create an Issue** → **Solution Review** → **Development Environment Setup** → **PR Submission** → **CI Gate** → **Committer Review and Maintainer Merge**

### 1. Create an Issue (Solution Discussion)

If your contribution is not a simple bug fix but involves new features, new interfaces, new configuration parameters, or modifications to the code flow, please discuss the solution through an Issue first to avoid your code being rejected. If you are unsure whether your modification can be classified as a "simple bug fix," you can also discuss it by submitting an Issue.

Please create a `Requirement` type Issue to propose your idea. The Issue should include at least the following:

- **Background and motivation of the feature**: for example, what scenario requirements this feature addresses.
- **Core design solution**: briefly describe the technical approach for implementing this feature and the design of key modules.
- **Expected functional and performance goals**: for example, the core functions this feature must implement, and the expected accuracy metrics or performance.
- **Estimated completion time**: this helps the community understand development progress and facilitates subsequent collaborative support.

For specific steps, refer to the [Issue Operation Guide](https://gitcode.com/cann/community/blob/master/contributor/issue-operation.md).

### 2. Solution Review

After creating a requirement Issue, you need to submit a SIG topic and attend the [Tools SIG](https://gitcode.com/cann/community/blob/master/CANN/sigs/tools/README.md) review. For meeting arrangements, topic application methods, and Maintainer contact information, see the [Tools SIG description](https://gitcode.com/cann/community/blob/master/CANN/sigs/tools/README.md).

**Urgent requirements**: If your requirement is urgent, contact the Maintainer according to the Tools SIG description to request a temporary SIG review.

**Requirement acceptance**: If the requirement is accepted, [SIG members](https://gitcode.com/cann/community/blob/master/CANN/sigs/tools/sig-info.yaml) will assign you an appropriate integration path (such as `amct_pytorch/experimental/`). Please submit the feature code to the corresponding directory.

### 3. Development Environment Setup

AMCT depends on the Ascend CANN software stack. Please complete environment setup and source code building first to ensure that results can be reproduced and verified locally:

**Basic dependencies** (for complete environment deployment, see [Quick Installation](docs/en/quick_install.md)):

| Dependency | Version Requirement |
| ---------- | ------------------- |
| Python | >= 3.9 |
| PyTorch | 2.7.1 or 2.1.0 (the NPU acceleration backend requires the CPU build of PyTorch, used together with `TorchNPU`) |
| GCC / CMake / patch | >= 7.3 / >= 3.16 (3.20 recommended) / >= 2.7 |
| CANN (Toolkit & Ops) | >= 8.5.0 (NPU driver / firmware must be installed in advance) |

**Source code build and installation verification**:

```shell
# 1. Install dependencies and enable local code checks (see Appendix B)
pip3 install -r requirements.txt
pip install pre-commit
pre-commit install

# 2. Build the distribution package (artifacts are located in build_out/)
bash build.sh --torch    # Build only the amct_pytorch package (most common)
# or bash build.sh --pkg # Build the full amct package (amct_pytorch + graph). The artifact
#                        # is a CANN full package (cann-amct_${version}_${os}-${arch}.tar.gz)
#                        # and cannot be installed directly with pip
# Note: The amct_pytorch/experimental/ directory is excluded from compilation by default.
#       If your change involves this directory, append --experimental,
#       e.g. bash build.sh --torch --experimental

# 3. Install the built package and verify (using the --torch artifact as an example)
#    ${version} is obtained from the file name in the build_out/ directory,
#    e.g. amct_pytorch-1.1.0-py3-none-linux_aarch64.tar.gz
#    ${arch} is the CPU architecture, e.g. x86_64, aarch64
pip3 install build_out/amct_pytorch-${version}-py3-none-linux_${arch}.tar.gz --user
python3 -c "import amct_pytorch as amct; print('successfully installed AMCT')"
```

> **Note**: If your pip version is > 25.2, append `--no-build-isolation` to the installation command; otherwise, `ModuleNotFoundError: No module named 'torch'` may occur.

**Unit tests** (recommended to run locally before submitting a PR):

```shell
bash build.sh -u    # Build and run unit tests
```

### 4. PR Submission

#### Delivery Requirements

| Scenario | Delivery Requirements | Reference |
| -------- | --------------------- | --------- |
| New features such as quantization / compression algorithms | Feature code + optimization documentation + README documentation | [Algorithm Introduction](docs/en/algorithm_brief.md) |
| NPU operators | Operator code (`op_kernel` / `op_extension`) + UT tests + README documentation | [amct_ops](amct_ops/README_en.md) |
| New model adaptation | Adaptation code / quantization scripts + one-stop sample documentation | [examples/models](examples/models/) |

README documentation is key to helping other developers get started quickly, and needs to include two parts:

- **Brief description**: the core content of the current feature, such as the model used, supported execution devices, and verified accuracy / performance information.
- **Operation steps**: describe the full process from environment preparation to feature execution in as much detail as possible, so that other developers can smoothly reproduce the results.

If your feature has not undergone strong generalization testing, it is recommended to merge the feature code into the `amct_pytorch/experimental/` directory. For directory organization, refer to existing features (such as [hifloat8](amct_pytorch/experimental/hifloat8/)). For the complete structure, see [Appendix A](#appendix-a-complete-delivery-structure).

> **Note**: If feature development involves adding or modifying operators, please first merge the corresponding operator changes into the operator repository (for the `amct_ops/` process, see [amct_ops](amct_ops/README_en.md)), and then proceed with the compression feature integration.

#### Compliance Checklist

Before submitting, please perform a self-check on each item:

- [ ] Does the code comply with the [C++ Coding Standards](https://gitcode.com/cann/community/blob/master/contributor/coding-standards/C++%20Coding%20standards.md) or the Python standards (for the tool matrix of this repository, see [Appendix B](#appendix-b-coding-standards-and-local-checks))?
- [ ] Does the code compile successfully (`bash build.sh --torch`), and do the UTs pass?
- [ ] Does the Markdown documentation syntax comply with the specifications?
- [ ] Is the LICENSE used compliant? Licenses such as Apache 2.0 are recommended, with copyright information marked according to actual conditions.
- [ ] The merged code contains no binary files (except for images needed in README or documentation).
- [ ] If third-party datasets are involved, only describe the download and usage methods in the documentation; there is no need to provide the datasets directly.
- [ ] If the modification involves public code (not code internal to the feature), it can pass CI validation.

#### Submission Conventions

- **Target branch**: Submit PRs to the `master` branch.
- **PR title**: Follow the Conventional Commits specification (`feat:` / `fix:` / `docs:` / `perf:` / `refactor:` etc.). It is recommended to associate the Issue number, such as `feat: add XX quantization algorithm(#123)`.
- **PR description**: Please carefully fill in the business background, purpose, solution, and other information of this PR according to the PR template, and link the preceding requirement Issue.

For specific operations such as forking, cloning, synchronization, and conflict handling, refer to the [PR Operation Guide](https://gitcode.com/cann/community/blob/master/contributor/pull_request_operation.md) and the [GitCode Workflow description](https://gitcode.com/cann/community/blob/master/contributor/gitcode-workflow.md).

### 5. CI Gate

Trigger the open-source repository gate by entering the `/compile` command in the PR comment box, and make modifications according to the CI results. Currently, the CI gate mainly includes the following checks:

- Code compilation (x86 / aarch64)
- Static checks (for codecheck false positives, please submit them to SIG members for suppression)
- UT tests
- Smoke tests

After the gate passes, please @ the Committer in the associated Issue.

### 6. Committer Review and Maintainer Merge

After the Committer review, review comments will be provided. Please make modifications based on the comments, and @ the Committer when the modifications are complete.

After the Committer review passes, the `/lgtm` label will be added. The Maintainer then performs the final review and, after confirming that there are no problems, adds the `/approve` label to merge the PR.

---

## 2. Bug Fix

If you discover bugs in this project and wish to fix them, you are welcome to create an Issue for feedback and tracking.

You can follow the [Submit Issue/Process Issue Task](https://gitcode.com/cann/community#提交Issue处理Issue任务) guide to create a `Bug-Report` type Issue to describe the bug. For simple bug fixes, you can also submit a PR directly (please still describe the background and verification method according to the PR template).

## 3. Documentation Correction

If you discover documentation description errors in this project, you are welcome to create an Issue for feedback and correction. For documentation writing specifications, refer to the [Documentation Contribution Guide](https://gitcode.com/cann/community/blob/master/contributor/docs/document_writing_specs.md).

You can follow the [Submit Issue/Process Issue Task](https://gitcode.com/cann/community#提交Issue处理Issue任务) guide to create a `Documentation` type Issue to point out the problems in the corresponding documentation. Then enter `/assign` or `/assign @yourself` in the comment box to assign the Issue to yourself to correct the documentation description.

## 4. Experience Feedback

If you find usability problems during the use of AMCT in areas such as documentation, APIs and configuration, examples, the build toolchain, or runtime feedback (such as being hard to learn, hard to use, or unclear error messages), you are welcome to create an Issue for feedback to help us continuously improve usability.

You can follow the [Submit Issue/Process Issue Task](https://gitcode.com/cann/community#提交Issue处理Issue任务) guide to create an `Experience Request` type Issue, describing the feature chain and compression process stage the problem belongs to, the problem you actually encountered, and the improvement you expect. If you would like to work on the improvement yourself, you can also enter `/assign` or `/assign @yourself` in the comment box to assign the Issue to yourself for processing.

## 5. Help Solve Others' Issues

If you have suitable solutions for problems encountered by others in the community, you are welcome to publish comments in the Issue to communicate, helping others solve problems and pain points and jointly improving usability.

If the corresponding Issue requires code modification, you can enter `/assign` or `/assign @yourself` in the Issue comment box to assign the Issue to yourself, and track the assistance in solving the problem.

---

## Appendix A: Complete Delivery Structure

### Quantization / Compression New Features (`amct_pytorch/experimental/`)

```text
├── experimental                             # Experimental features
│   ├── ${feature_name}                      # Feature name (such as hifloat8, flatquant)
│   │   ├── README.md                        # Feature documentation (README_en.md is the English version)
│   │   ├── ${feature_name}.py               # Feature implementation code
│   │   ├── test.py                          # Feature test file
│   │   ├── build.sh / setup.py              # Build configuration (if involved)
│   │   └── ...                              # Other necessary files
│   └── ...
```

### NPU Operators (`amct_ops/`)

```text
├── ${op_name}                               # Operator name (such as hifloat8_cast)
│   ├── op_kernel                            # Operator kernel implementation
│   ├── op_extension                         # Operator extension implementation
│   ├── python                               # Python-side interfaces
│   ├── CMakeLists.txt                       # Operator build configuration file
│   └── README.md                            # Operator documentation
```

### New Model Adaptation (`examples/models/`)

```text
├── ${model_name}                            # Model name (such as deepseekv4.1, qwen3.6)
│   ├── README.md                            # Quantization operation steps (README_en.md is the English version)
│   ├── ${Model-Walkthrough}.md              # One-stop sample documentation (full process from environment preparation to result reproduction)
│   └── assets                               # Resources such as documentation images
```

## Appendix B: Coding Standards and Local Checks

This repository uses `pre-commit` to automatically run code checks on commits. The main checks include:

| Check | Description |
| ----- | ----------- |
| clang-format | C/C++ code formatting (follows `.clang-format` in the repository root: 120-character line width, 4-space indentation) |
| ruff | Python code checking and formatting (follows PEP8) |
| codespell | Spelling check (the CANN-specific word whitelist is configured in the `[tool.codespell]` section of `pyproject.toml`) |
| Text / security checks | Trailing whitespace, end-of-file newline, large files (>1 MB), private key detection, etc. |

```shell
pip install pre-commit
pre-commit install
```

For more standards, refer to the community documentation:

- [C++ Coding Standards](https://gitcode.com/cann/community/blob/master/contributor/coding-standards/C++%20Coding%20standards.md)
- [Secure Coding Standards](https://gitcode.com/cann/community/blob/master/contributor/coding-standards/SecureCoding.md)
- [Experimental Feature Conventions](https://gitcode.com/cann/community/blob/master/contributor/experimental-feature-conventions.md) (including the `AMCT_EXPERIMENTAL` macro convention)
- [Open-Source Code Snippet Reference Guideline](https://gitcode.com/cann/community/blob/master/contributor/third_party/snippet-reference-guideline.md)

If you have any questions during the contribution process, feel free to communicate in the community at any time. Thank you again for your support, and we look forward to your wonderful contributions!
