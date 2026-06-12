# AGENTS.md

本文件为 agent 在此代码仓库中工作时提供指导。

## 项目概述

AMCT（Ascend Model Compression Toolkit）是一款昇腾 AI 处理器亲和的深度学习模型压缩工具包，提供多种模型量化压缩特性（W8A16、W8A8、W4A16 等）。压缩后模型体积变小，部署到昇腾 AI 处理器可使能低比特运算，提高计算效率。

主要功能：
- **量化压缩**：支持多种量化算法（AWQ、GPTQ、SmoothQuant、Minmax 等）
- **HiFloat8**：支持 HiFloat8/FP8/FP4 数据量化和分位量化
- **图压缩**：基于计算图的压缩优化（张量分解等）
- **NPU 算子**：量化相关的昇腾 NPU 算子实现
- **实验特性**：DeepSeekV3.2/V4 等前沿模型的量化支持

## 构建命令

### 基础构建
```bash
# 构建（Release 模式）
bash build.sh

# 查看构建选项
bash build.sh --help

# 构建打包（生成可分发的 tar.gz）
bash build.sh --pkg

# Debug 模式构建
bash build.sh --build-type=Debug

# 指定线程数构建
bash build.sh -j8

# 指定第三方库路径
bash build.sh --cann_3rd_lib_path=/path/to/3rd_party
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

## 目录结构

| 目录 | 用途 |
|------|------|
| `amct_pytorch/` | PyTorch 量化压缩核心源码 |
| `amct_pytorch/experimental/` | 实验特性（HiFloat8、DeepSeek 等） |
| `amct_pytorch/graph_based_compression/` | 基于计算图的压缩优化（张量分解等） |
| `npu_ops/` | 量化相关的昇腾 NPU 算子代码 |
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
- Python 代码遵循 PEP8 规范

### 贡献流程
- 简单 bug 修复：直接提交 PR
- 新特性/接口变更：先通过 Issue 讨论方案，达成共识后再提交 PR
- PR 需要关联相关 Issue，并包含特性代码、优化文档和 README

### 许可证
- Apache 2.0 协议，需在代码中标注版权信息

## 短语
使用中文
