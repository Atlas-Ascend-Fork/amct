# AGENTS.md

本文件为 agent 在此代码仓库中工作时提供指导。

## 项目概述

AMCT（Ascend Model Compression Toolkit）是一款昇腾 AI 处理器亲和的深度学习模型压缩工具包，提供多种模型量化压缩特性（W8A8、W4A8、W4A4 等）。压缩后模型体积变小，部署到昇腾 AI 处理器可使能低比特运算，提高计算效率。

主要功能：
- **量化压缩**：支持多种量化算法（AWQ、GPTQ、SmoothQuant、Minmax 等）
- **HiFloat8**：支持 HiFloat8/FP8/FP4 数据量化和分位量化
- **图压缩**：基于计算图的压缩优化（张量分解等）
- **NPU 算子**：量化相关的昇腾 NPU 算子实现
- **试验特性**：DeepSeekV3.2/V4 等前沿模型的量化支持

## 构建命令

### 基础构建
> ⚠️ 注意：不带任何打包选项直接执行 `bash build.sh` **不会产出任何分发包**——脚本仅在
> 指定 `--torch`、`--pkg` 或 `-u/--utest` 时才会真正执行构建。要构建可安装的包，必须使用
> `--torch` 或 `--pkg`。

```bash
# 查看构建选项
bash build.sh --help

# 仅构建 amct_pytorch 包（产物位于 build_out/，最常用）
bash build.sh --torch

# 构建完整 amct 包（amct_pytorch + graph，产物位于 build_out/）
bash build.sh --pkg

# 构建包含试验特性的 amct_pytorch 包（含 amct_pytorch/experimental/）
bash build.sh --torch --experimental

# 构建包含试验特性的完整 amct 包
bash build.sh --pkg --experimental

# Debug 模式构建（需配合 --torch 或 --pkg）
bash build.sh --torch --build-type=Debug

# 指定线程数构建
bash build.sh --torch -j8

# 指定第三方库路径
bash build.sh --torch --cann_3rd_lib_path=/path/to/3rd_party
```

### 单元测试
```bash
# 构建并运行单元测试
bash build.sh -u

# 启用覆盖率
bash build.sh -u --cov

# 启用 AddressSanitizer
bash build.sh --asan
```

### Python 环境安装
```bash
# 安装依赖
pip install -r requirements.txt

# 安装 pre-commit 代码检查工具
pip install pre-commit
pre-commit install
```

### 安装 AMCT 到 Python 环境
`bash build.sh --torch` / `--pkg` 等构建命令只生成分发包，产物位于 `build_out/` 目录下，
并不会自动装入当前 Python 环境。要让 `import amct_pytorch` 可用，必须再用 pip 安装构建产物：
```bash
# 1. 构建 amct_pytorch 分发包（产物位于 build_out/）
bash build.sh --torch

# 2. 安装构建产物
#    ${version} 从 build_out/ 目录的文件名获取，如 amct_pytorch-1.1.0-py3-none-linux_aarch64.tar.gz
#    ${arch}    为 CPU 架构，如 x86_64、aarch64
pip3 install build_out/amct_pytorch-${version}-py3-none-linux_${arch}.tar.gz --user

# 3. 验证安装
python3 -c "import amct_pytorch as amct; print('successfully installed AMCT')"
```

> ⚠️ 注意：若 pip 版本 > 25.2，安装命令需追加 `--no-build-isolation`，
> 否则可能出现 `ModuleNotFoundError: No module named 'torch'`。
>
> 此外 AMCT 运行依赖 CANN（Toolkit & Ops）≥ 8.5.0 及对应的 NPU 驱动/固件，
> 完整环境部署参见 `docs/zh/quick_install.md`。

## 目录结构

| 目录 | 用途 |
|------|------|
| `amct_pytorch/` | PyTorch 量化压缩核心源码 |
| `amct_pytorch/experimental/` | 试验特性（HiFloat8、DeepSeek 等） |
| `amct_pytorch/classic/graph_based/` | 基于计算图的压缩优化（张量分解等） |
| `amct_ops/` | 量化相关的昇腾 NPU 算子代码 |
| `tests/` | 单元测试 |
| `examples/` | 端到端样例开发和调用示例 |
| `docs/` | 工具文档（概念介绍、API 文档、算法介绍等） |
| `cmake/` | CMake 构建配置 |
| `build.sh` | 工程编译脚本 |
| `setup.py` | Python 包打包入口 |
| `requirements.txt` | Python 依赖包 |
| `.clang-format` | C/C++ 代码格式化配置 |

## 开发规范

### gitcode pr/issue 操作
@.claude/skills/default-skills/SKILL.md

### 代码风格
- 遵循 Google 开源代码规范（基于 `.clang-format`）
- 使用 `clang-format` 格式化 C/C++ 代码，行宽限制 120 字符，4 空格缩进
- 使用 `pre-commit` 在提交前自动检查代码规范
- Python 代码使用 `ruff` 检查与格式化（遵循 PEP8 规范）

### 贡献流程
- 简单 bug 修复：直接提交 PR
- 新特性/接口变更：先通过 Issue 讨论方案，达成共识后再提交 PR
- PR 需要关联相关 Issue，并包含特性代码、优化文档和 README

### Issue / PR 模板
提交 Issue 和 PR 时必须参照本仓的对应模板填写：
- **PR 模板**：`.gitcode/PULL_REQUEST_TEMPLATE.zh-CN.md`
- **Issue 模板**（位于 `.gitcode/ISSUE_TEMPLATE/`，按类型选择）：
  - `bug_report.yaml` — Bug 报告
  - `feature_requeset.yaml` — 新特性需求
  - `documentaion.yaml` — 文档问题
  - `question.yaml` — 使用咨询/提问

### 许可证
- Apache 2.0 协议，需在代码中标注版权信息

## 短语
使用中文
