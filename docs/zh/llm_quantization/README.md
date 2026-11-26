# AMCT LLM 文档中心

本文档是 AMCT LLM 文档体系的标准入口，帮助用户和开发者根据任务选择对应内容。

## 能力范围

AMCT LLM 文档围绕以下能力组织：

- LLM 量化能力和使用入口；
- 量化目标、数据类型、bit 配置和算法；
- 从模型评估、PTQ 到部署验证的工作流；
- 模型适配和新模型接入；
- 量化组件扩展和底层算子；
- CLI、Python API、配置文件和输出产物参考；
- 常见问题和故障排查。

具体支持范围、模型和部署能力以当前 GitCode 版本及对应示例为准。

## 文档目录

| 主题 | 目录 | 主要内容 | 适合读者 | 入口 |
|---|---|---|---|---|
| 新手入门 | [01_getting_started](01_getting_started/) | 入门概览、环境准备和 Quick Start | 初次使用者 | [入门概览](01_getting_started/01_overview.md) |
| 量化方案指南 | [02_quantization_guide](02_quantization_guide/) | 量化概念、方案和算法 | 量化使用者 | [量化概览](02_quantization_guide/01_overview.md) |
| 量化工作流 | [03_workflow](03_workflow/) | 模型评估、PTQ、部署和验证 | 使用者和工程师 | [工作流总览](03_workflow/01_overview.md) |
| 模型适配 | [04_model_adaptation](04_model_adaptation/) | 模型适配器、新模型接入和支持矩阵 | 模型开发者 | [适配概览](04_model_adaptation/01_overview.md) |
| 量化扩展开发 | [05_algorithm_development](05_algorithm_development/) | 量化组件、扩展和算子 | 框架开发者 | [扩展概览](05_algorithm_development/01_overview.md) |
| 接口与产物参考 | [06_reference](06_reference/) | CLI、Python API、配置和输出产物 | 使用者和维护者 | [参考概览](06_reference/01_overview.md) |
| 问题排查 | [07_troubleshooting](07_troubleshooting/) | 环境、模型、配置、PTQ 和部署排错 | 所有使用者 | [问题排查](07_troubleshooting/01_troubleshooting.md) |

## 推荐阅读路径

### 用户使用

```text
01_getting_started
  ↓
02_quantization_guide
  ↓
03_workflow
  ↓
06_reference / 07_troubleshooting
```

### 模型适配

```text
04_model_adaptation/01_overview.md
  ↓
04_model_adaptation/02_new_model.md
  ↓
04_model_adaptation/03_support_matrix.md
```

### 量化扩展

```text
05_algorithm_development/01_overview.md
  ↓
05_algorithm_development/02_extension.md
  ↓
05_algorithm_development/03_operators.md
```
