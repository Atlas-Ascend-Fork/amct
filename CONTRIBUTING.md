# 贡献指南

本项目欢迎广大开发者体验并参与贡献。在参与社区贡献之前，请参见 [cann-community](https://gitcode.com/cann/community) 了解[行为准则](https://gitcode.com/cann/community/blob/master/contributor/code-of-conduct.md)、完成 [CLA 协议签署](https://gitcode.com/cann/infrastructure/blob/master/docs/cla/cla使用指南.md)，并了解源码仓的贡献流程。除代码贡献外，社区同样欢迎会议、邮件、测试等多种形式的贡献，详见社区[参与贡献](https://gitcode.com/cann/community#参与贡献)总览。

若您希望参与社区非本仓的进阶贡献（如新建 SIG、成为核心贡献者、新建仓库、引入开源三方件等），请参阅 [CANN 社区贡献指南](https://gitcode.com/cann/community#进阶贡献)。

## 贡献场景

开发者贡献场景主要包括：

- [贡献新特性](#一贡献新特性)
- [Bug 修复](#二bug-修复)
- [文档纠错](#三文档纠错)
- [体验反馈](#四体验反馈)
- [帮助解决他人 Issue](#五帮助解决他人-issue)

---

## 一、贡献新特性

感谢您参与 AMCT 建设，为模型压缩注入更多价值！完整的贡献流程如下：

> **创建 Issue** → **方案评审** → **开发环境准备** → **PR 提交** → **CI 门禁** → **Committer 检视与 Maintainer 合入**

### 1. 创建 Issue（方案讨论）

若您的贡献不是简单的 Bug 修复，而是涉及新增特性、新增接口、新增配置参数或修改代码流程等，请务必先通过 Issue 进行方案讨论，以避免您的代码被拒绝合入。若您不确定本次修改是否可归为"简单的 Bug 修复"，亦可通过提交 Issue 进行讨论。

请新建 `Requirement|需求建议` 类 Issue 提出您的想法，Issue 中请至少包含以下内容：

- **特性的背景和动机**：比如该特性解决了什么场景的需求。
- **核心设计方案**：简要说明该特性实现的技术思路、关键模块的设计方案。
- **预期达成的功能性能目标**：例如该特性需实现的核心功能、预期的精度指标或性能表现。
- **预计完成时间**：方便社区了解开发进度，也便于后续协同支持。

具体操作步骤请参阅：[Issue 操作指南](https://gitcode.com/cann/community/blob/master/contributor/issue-operation.md)。

### 2. 方案评审

创建需求类 Issue 后，需申报 SIG 议题，并参加 [Tools SIG](https://gitcode.com/cann/community/blob/master/CANN/sigs/tools/README.md) 评审。会议安排、议题申请方式及 Maintainer 联系方式请查阅 [Tools SIG 说明](https://gitcode.com/cann/community/blob/master/CANN/sigs/tools/README.md)。

**紧急需求处理**：若需求紧急，可根据 Tools SIG 说明联系 Maintainer，申请临时 SIG 评审。

**需求接纳**：若需求被接纳，[SIG 成员](https://gitcode.com/cann/community/blob/master/CANN/sigs/tools/sig-info.yaml)将为您分配合适的合入路径（如 `amct_pytorch/experimental/`），请将特性代码提交至对应目录。

### 3. 开发环境准备

AMCT 依赖昇腾 CANN 软件栈，请先完成环境搭建与源码构建，确保本地可复现验证：

**基础依赖**（完整环境部署请参见[快速安装](docs/zh/quick_install.md)）：

| 依赖 | 版本要求 |
| ---- | -------- |
| Python | ≥ 3.9 |
| PyTorch | 2.7.1 或 2.1.0（NPU 加速后端需使用 CPU 版 PyTorch，并配套 `TorchNPU`） |
| GCC / CMake / patch | ≥ 7.3 / ≥ 3.16（推荐 3.20） / ≥ 2.7 |
| CANN（Toolkit & Ops） | ≥ 8.5.0（需提前安装 NPU 驱动 / 固件） |

**源码构建与安装验证**：

```shell
# 1. 安装依赖并启用本地代码检查（见附录 B）
pip3 install -r requirements.txt
pip install pre-commit
pre-commit install

# 2. 构建分发包（产物位于 build_out/）
bash build.sh --torch    # 仅构建 amct_pytorch 包（最常用）
# 或 bash build.sh --pkg # 构建完整 amct 包（amct_pytorch + graph），产物为 CANN 整包
#                        # （cann-amct_${version}_${os}-${arch}.tar.gz），不支持 pip 直接安装
# 注意：amct_pytorch/experimental/ 目录默认不在编译范围内，若改动涉及该目录，
#       请追加 --experimental 参数，如 bash build.sh --torch --experimental

# 3. 安装构建产物并验证（以 --torch 产物为例）
#    ${version} 从 build_out/ 目录的文件名获取，如 amct_pytorch-1.1.0-py3-none-linux_aarch64.tar.gz
#    ${arch}    为 CPU 架构，如 x86_64、aarch64
pip3 install build_out/amct_pytorch-${version}-py3-none-linux_${arch}.tar.gz --user
python3 -c "import amct_pytorch as amct; print('successfully installed AMCT')"
```

> **注意**：若 pip 版本 > 25.2，安装命令需追加 `--no-build-isolation`，否则可能出现 `ModuleNotFoundError: No module named 'torch'`。

**单元测试**（建议提交 PR 前本地自测）：

```shell
bash build.sh -u    # 构建并运行单元测试
```

### 4. PR 提交

#### 交付件要求

| 场景 | 交付件要求 | 参考资料 |
| ---- | ---------- | -------- |
| 量化 / 压缩算法等新特性 | 特性代码 + 优化文档 + README 文档 | [算法介绍](docs/zh/algorithm_brief.md) |
| NPU 算子 | 算子代码（`op_kernel` / `op_extension`） + UT 测试 + README 文档 | [amct_ops](amct_ops/README.md) |
| 新模型适配 | 适配代码 / 量化脚本 + 一站式样例文档 | [examples/models](examples/models/) |

其中 README 文档是帮助其他开发者快速上手的关键，需包含两部分：

- **简述**：当前特性的核心内容，比如用到的模型、支持的执行设备、已验证的精度 / 性能信息等。
- **操作步骤**：请尽量详细地描述从环境准备到特性执行的全流程，确保其他开发者能顺利复现结果。

若您的特性尚未经过较强的泛化性测试，建议将特性代码合入到 `amct_pytorch/experimental/` 目录下，目录组织可参考已有特性（如 [hifloat8](amct_pytorch/experimental/hifloat8/)），完整结构参见[附录 A](#附录-a完整交付件结构)。

> **注意**：如果特性开发过程中涉及算子的新增或修改，请先将对应的算子变更合入到算子仓（`amct_ops/` 相关流程参见 [amct_ops](amct_ops/README.md)），再推进压缩特性合入。

#### 合规检查

提交前请逐项自查：

- [ ] 代码是否符合 [C++ 编程规范](https://gitcode.com/cann/community/blob/master/contributor/coding-standards/C++%20Coding%20standards.md) 或 Python 规范（本仓工具矩阵见[附录 B](#附录-b代码规范与本地检查)）
- [ ] 代码是否编译通过（`bash build.sh --torch`），UT 是否通过
- [ ] Markdown 文档语法是否符合规范
- [ ] 所使用的 LICENSE 是否合规，建议使用 Apache 2.0 等协议，并按照实际情况标注版权信息
- [ ] 合入的代码里不包含二进制文件（README 或文档中需要的配图除外）
- [ ] 若涉及第三方数据集，只需在文档中说明下载方式、使用方法，无需直接提供数据集
- [ ] 若修改涉及公共代码（非该特性内部代码），能通过 CI 验证

#### 提交规范

- **目标分支**：向 `master` 分支提交 PR。
- **PR 标题**：遵循 Conventional Commits 规范（`feat:` / `fix:` / `docs:` / `perf:` / `refactor:` 等），建议关联 Issue 编号，如 `feat: 新增XX量化算法(#123)`。
- **PR 描述**：请按照 PR 模板仔细填写本次 PR 的业务背景、目的、方案等信息，并关联前述需求 Issue。

Fork、克隆、同步、冲突处理等具体操作步骤，请参阅 [PR 操作指南](https://gitcode.com/cann/community/blob/master/contributor/pull_request_operation.md)与 [GitCode 工作流说明](https://gitcode.com/cann/community/blob/master/contributor/gitcode-workflow.md)。

### 5. CI 门禁

通过在 PR 评论框输入 `/compile` 指令触发开源仓门禁，并依据 CI 检测结果进行修改。目前 CI 门禁主要包含以下检查项：

- 代码编译（x86 / aarch64）
- 静态检查（如涉及 codecheck 误报，请提交给 SIG 成员屏蔽）
- UT 测试
- 冒烟测试

门禁通过后，请在关联的 Issue 中 @ Committer。

### 6. Committer 检视与 Maintainer 合入

Committer 检视后将反馈检视意见，请根据意见修改，完成后 @ Committer。

Committer 检视通过后，将标注 `/lgtm` 标签；Maintainer 最终审核，确认无问题后，将标注 `/approve` 标签合入 PR。

---

## 二、Bug 修复

如果您在本项目中发现了某些 Bug，希望对其进行修复，欢迎您新建 Issue 进行反馈和跟踪处理。

您可以按照[提交 Issue / 处理 Issue 任务](https://gitcode.com/cann/community#提交Issue处理Issue任务)指引新建 `Bug-Report|缺陷反馈` 类 Issue 对 Bug 进行描述。若为简单 Bug 修复，亦可直接提交 PR（仍请按 PR 模板说明背景与验证方式）。

## 三、文档纠错

如果您在本项目中发现某些文档描述错误，欢迎您新建 Issue 进行反馈和修复，文档写作规范可参考[文档贡献指南](https://gitcode.com/cann/community/blob/master/contributor/docs/document_writing_specs.md)。

您可以按照[提交 Issue / 处理 Issue 任务](https://gitcode.com/cann/community#提交Issue处理Issue任务)指引新建 `Documentation|文档反馈` 类 Issue 指出对应文档的问题，然后在评论框中输入 `/assign` 或 `/assign @yourself`，将该 Issue 分配给您纠正对应文档描述。

## 四、体验反馈

如果您在使用 AMCT 的过程中，发现文档、API 与配置、样例、构建工具链或运行时反馈等环节存在体验问题（如不易学习、不易使用、报错信息不清晰等），欢迎您新建 Issue 进行反馈，帮助我们持续优化易用性。

您可以按照[提交 Issue / 处理 Issue 任务](https://gitcode.com/cann/community#提交Issue处理Issue任务)指引新建 `Experience Request|体验需求` 类 Issue，说明体验问题所属的功能链路与压缩流程阶段、实际遇到的问题以及期望的改进方式。若您希望直接参与改进，也可以在评论框中输入 `/assign` 或 `/assign @yourself`，将该 Issue 分配给您进行处理。

## 五、帮助解决他人 Issue

如果社区中他人遇到的问题您有合适的解决方法，欢迎您在 Issue 中发表评论交流，帮助他人解决问题和痛点，共同优化易用性。

如果对应 Issue 需要进行代码修改，您可以在 Issue 评论框中输入 `/assign` 或 `/assign @yourself`，将该 Issue 分配给您，跟踪协助解决问题。

---

## 附录 A：完整交付件结构

### 量化 / 压缩新特性（`amct_pytorch/experimental/`）

```text
├── experimental                             # 实验特性
│   ├── ${feature_name}                      # 特性名称（如 hifloat8、flatquant）
│   │   ├── README.md                        # 特性说明文档（README_en.md 为英文版）
│   │   ├── ${feature_name}.py               # 特性实现代码
│   │   ├── test.py                          # 特性测试文件
│   │   ├── build.sh / setup.py              # 构建配置（如涉及）
│   │   └── ...                              # 其他必要文件
│   └── ...
```

### NPU 算子（`amct_ops/`）

```text
├── ${op_name}                               # 算子名称（如 hifloat8_cast）
│   ├── op_kernel                            # 算子 Kernel 实现
│   ├── op_extension                         # 算子扩展实现
│   ├── python                               # Python 侧接口
│   ├── CMakeLists.txt                        # 算子编译配置文件
│   └── README.md                            # 算子说明文档
```

### 新模型适配（`examples/models/`）

```text
├── ${model_name}                            # 模型名称（如 deepseekv4.1、qwen3.6）
│   ├── README.md                            # 量化操作步骤（README_en.md 为英文版）
│   ├── ${Model-Walkthrough}.md              # 一站式样例文档（含环境准备到结果复现全流程）
│   └── assets                               # 文档配图等资源
```

## 附录 B：代码规范与本地检查

本仓通过 `pre-commit` 在提交时自动执行代码检查，主要检查项包括：

| 检查项 | 说明 |
| ------ | ---- |
| clang-format | C/C++ 代码格式化（遵循仓库根目录 `.clang-format`，行宽 120 字符、4 空格缩进） |
| ruff | Python 代码检查与格式化（遵循 PEP8 规范） |
| codespell | 拼写检查（CANN 专用词典白名单见 `pyproject.toml` 的 `[tool.codespell]` 段） |
| 文本 / 安全检查 | 行尾空白、文件末尾换行、大文件（>1MB）、私有密钥检测等 |

```shell
pip install pre-commit
pre-commit install
```

更多规范请参考社区文档：

- [C++ 编程规范](https://gitcode.com/cann/community/blob/master/contributor/coding-standards/C++%20Coding%20standards.md)
- [安全编码规范](https://gitcode.com/cann/community/blob/master/contributor/coding-standards/SecureCoding.md)
- [试验特性规范](https://gitcode.com/cann/community/blob/master/contributor/experimental-feature-conventions.md)（含 `AMCT_EXPERIMENTAL` 宏约定）
- [开源代码片段引用指导](https://gitcode.com/cann/community/blob/master/contributor/third_party/snippet-reference-guideline.md)

如果在贡献过程中有任何疑问，随时欢迎在社区中交流；再次感谢您的支持，期待您的精彩贡献！
