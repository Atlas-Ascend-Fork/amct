<div align="center">

# AMCT

**Ascend Model Compression Toolkit**

_昇腾 NPU 原生模型压缩工具包_

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![CANN](https://img.shields.io/badge/CANN-%E2%89%A58.5.0-green.svg)](docs/zh/quick_install.md)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0%20%7C%202.7.1-orange.svg)](requirements.txt)

[快速开始](#-快速开始) · [特性](#-核心特性) · [样例](#-文档样例) · [FAQ](#-常见问题) · [贡献](#-参与贡献)

</div>

---

## 🔥 最新动态

- **[2026/05/28]** 新增当前主流 LLM 网络量化、PTQ 算法支持，并提供 [DeepSeek-V4](./examples/models/deepseekv4/DeepSeekV4-Flash-Walkthrough.md) 和 [Qwen3.6-MoE](./examples/models/qwen3.6/Qwen3.6-Moe.md) 的一站式样例
- **[2026/04/24]** 新增 [DeepSeek-V4](./amct_pytorch/experimental/deepseek-v4/README.md) 模型 INT8 量化支持
- **[2026/04/17]** 新增 HiFloat8 分位量化（Quantile）算法
- **[2026/03/02]** 新增 HiFloat8 数据直转（Cast）算法
- **[2026/02/02]** 新增 HiFloat8 / MXFP8 / MXFP4 数据量化
- **[2025/12/22]** AMCT 项目首次上线 🎉

## 🚀 概述

AMCT 是昇腾 NPU 原生的模型量化压缩工具。量化后模型体积减小，在昇腾 NPU 上启用低比特运算，显著提升推理性能。部署架构如下：

<div align="center">
  <img src="docs/zh/figures/amct_architecture.png" alt="AMCT 架构" width="720">
</div>

**亮点为：**

- **🎯 硬件亲和** —— 量化结果直接对接昇腾 NPU 低比特运算单元
- **🔢 多精度全栈** —— INT8 / INT4 / MXFP8 / MXFP4 / HiFloat8 任选
- **🚀 大模型就绪** —— 原生支持 DeepSeek-V3.2 / V4 等前沿模型


## ✨ 核心特性

| 特性类别 | 简介 |
|----------|------|
| **PTQ 量化算法** | Min-Max / AWQ / GPTQ / SmoothQuant 等训练后量化算法，详见 [算法介绍](docs/zh/algorithm_brief.md) |
| **HiFloat8 量化** | 华为自研 8-bit 浮点格式，锥形精度 + 大动态范围，详见 [HiFloat8 介绍](docs/zh/context/hifloat8_quantization.md) |
| **NPU 自定义算子** | 基于NPU的自研算子，Ascend C kernel 实现，详见 [amct_ops](amct_ops/README.md) |
| **大模型量化** | DeepSeek-V3.2 / V4 量化方案，详见 [DeepSeek-V4](./amct_pytorch/experimental/deepseek-v4/README.md) |


## 📊 性能收益

量化显著降低部署成本：

| 精度格式 | 仅权重（W） | 全量化（W+A） | 收益 |
|---------|------------|--------------|------|
| **INT8** | ✅ Min-Max / AWQ / GPTQ | ✅ Min-Max / SmoothQuant | 体积 **↓50%** · 吞吐 ↑ |
| **INT4** | ✅ AWQ / GPTQ | ✅ FlatQuant | 体积 **↓75%** · 低带宽友好 |
| **HiFloat8** | ✅ Cast / Quantile / OFMR | ✅ Cast / Quantile / OFMR | 体积 **↓50%** · 大动态范围 |
| **MXFP8** | ✅ MXQuant | ✅ MXQuant | 体积 **↓50%** · 高精度 |
| **MXFP4** | ✅ MXQuant | ✅ MXQuant | 体积 **↓75%** · 微缩浮点 |


## 📦 快速开始

### 环境要求

| 依赖 | 版本 |
|------|------|
| Python | >=3.9 |
| PyTorch | 2.7.1 或 2.1.0（NPU 加速后端需使用 CPU 版 PyTorch，并配套 `torch_npu`） |
| GCC / CMake / patch | ≥ 7.3 / ≥ 3.16（推荐 3.20） / ≥ 2.7 |
| CANN（Toolkit & Ops） | ≥ 8.5.0（需提前安装 NPU 驱动 / 固件） |

> ⚠️ 使用 NPU 加速后端时，请安装 CPU 版 PyTorch 后再安装配套 `torch_npu`，不要在同一环境中混用非 CPU 版 PyTorch 与 `torch_npu`。

完整环境部署请参见 [快速安装](docs/zh/quick_install.md)。

### 安装&验证

```bash
# 1. 拉取源码并安装依赖
git clone https://gitcode.com/cann/amct.git
cd amct
pip3 install -r requirements.txt

# 2. 源码构建打包
bash build.sh --torch

# 3. 安装（产物位于 build_out/）
#    ${version} 从 build_out/ 目录中的文件名获取，如 amct_pytorch-1.1.0-py3-none-linux_aarch64.tar.gz
#    ${arch}    为 CPU 架构，如 x86_64、aarch64
pip3 install build_out/amct_pytorch-${version}-py3-none-linux_${arch}.tar.gz --user
```

> ⚠️ **注意**：若使用 `--no-build-isolation` 安装，pip 不会自动安装构建依赖；
> 请先执行 `pip install wheel`，再追加 `--no-build-isolation` 安装产物包，否则可能出现
> `error: invalid command 'bdist_wheel'`。

```bash
# 验证 AMCT 安装
python3 -c "import torch; assert '+cpu' in torch.__version__, 'must use CPU torch for NPU path'"
python3 -c "import amct_pytorch as amct; print(f'successfully installed AMCT ')"
```

更多构建选项与本地验证请参见 [构建指南](docs/zh/build.md)。

### 🏃 一站式平台快速体验

「一站式平台」是为开发者提供的 NPU 环境，内部已集成完整的 CANN 环境，可以直接使用。AMCT 针对该平台在相应样例 README 中提供了简化的「快速启动」路径，帮助用户最小步骤完成 NPU 推理体验。当前支持的模型正在持续扩展中，敬请关注：

| 实践                                                                           | 简介                                                                                             |
|------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------|
| [Qwen3.6-MoE](./examples/models/qwen3.6/Qwen3.6-Moe.md)                      | 在 Atlas A3 环境中完成 Qwen3.6-MoE 模型的量化、数据提取以及 PTQ，针对一站式平台场景提供标准启动流程和相关配置，帮助用户快速上手完成一次端到端 NPU 推理体验。 |
| [DeepSeek-V4](examples/models/deepseekv4/DeepSeekV4-Flash-Walkthrough.md) | 在 Atlas A3 环境中完成 DeepSeek-V4 Flash 模型的单卡推理，针对一站式平台场景提供标准启动流程和相关配置，帮助用户快速上手完成一次端到端 NPU 推理体验。    |

## 📖 文档样例

| 主题 | 内容 |
|------|------|
| [压缩概念](docs/zh/compression_concepts.md) | 量化、稀疏、蒸馏等基础概念 |
| [LLM量化](docs/zh/AMCT_Pytorch_LLM.md) | 面向大语言模型（LLM）的量化特性 |
| [压缩特性](docs/README.md) | AMCT 支持的基础压缩特性 |
| [API 文档](docs/zh/api/README.md) | 接口使用说明 |
| [算法介绍](docs/zh/algorithm_brief.md) | AWQ、GPTQ、SmoothQuant等算法原理 |

## 🔍 目录结构

```text
amct/
├── amct_pytorch/                  # PyTorch 量化压缩核心源码
│   ├── algorithms/                 # 量化算法实现
│   ├── cli/                        # 命令行入口
│   ├── common/                     # 通用工具、模型和数据处理
│   ├── configs/                    # 量化配置模板
│   ├── experimental/               # 实验特性（HiFloat8、DeepSeek 等）
│   ├── quantization/               # 量化数据类型与基础模块
│   └── workflows/                  # LLM 量化、评估和部署流程
├── amct_ops/                      # AMCT 自定义 NPU 算子
├── examples/                      # 端到端样例与调用示例
├── tests/                         # 单元测试
├── docs/                          # 工具文档（概念、API、算法等）
├── cmake/                         # CMake 构建配置
├── build.sh                       # 工程编译脚本
├── setup.py                       # Python 包打包入口
└── requirements.txt               # Python 第三方依赖
```

## ❓ 常见问题

<details>
<summary><strong>算法选择：何时使用 AWQ / GPTQ / SmoothQuant？</strong></summary>

| 算法 | 适用场景 | 核心思路 |
|------|---------|---------|
| **AWQ** | 大模型 PTQ，追求低量化误差 | 感知激活的权重量化，保护 ~1% 显著权重 |
| **GPTQ** | 大模型 PTQ，强调逐层优化 | 基于海森矩阵的权重微调，最小化量化误差 |
| **SmoothQuant** | 激活分布困难场景 | 将激活量化难度迁移至权重，平滑激活异常值 |
| **Min-Max** | 入门场景，简单快速 | 直接取最大最小值计算量化因子 |

**建议**：大模型权重量化首选 AWQ 或 GPTQ；W8A8 全量化场景推荐 SmoothQuant；入门学习推荐 Min-Max。

</details>

<details>
<summary><strong>量化后精度下降如何处理？</strong></summary>

**处理路径（按优先级）：**

1. **调整校准数据量**：增大 `batch_num`（推荐 batch_num × batch_size = 16 或 32）
2. **回退敏感层**：识别量化敏感层（首层、尾层、参数量少的层），在配置中设置 `quant_enable: false`
3. **调整量化算法**：分析模型数据分布特点，使用合适的量化算法
4. **尝试量化感知训练（QAT）**：若 PTQ 无法满足精度，使用 QAT 重训练

</details>

<details>
<summary><strong>安装时报 "ModuleNotFoundError: No module named 'torch'"？</strong></summary>

**原因**：pip 版本 > 25.2，构建隔离导致 torch 未被识别。

**解决方案**：

```bash
# 方案 1：降低 pip 版本
pip install pip==25.2

# 方案 2：先安装 wheel，再添加 --no-build-isolation
pip install wheel
pip3 install build_out/amct_pytorch-${version}-py3-none-linux_${arch}.tar.gz --user --no-build-isolation
```

</details>

更多问题请查阅 [压缩特性文档](docs/README.md) 或在 [Issue](https://gitcode.com/cann/amct/issues) 中提问。

## 💬 社区讨论

欢迎加入 AMCT 社区，参与讨论与交流：

| 平台 | 用途 |
|------|------|
| [GitCode Issue](https://gitcode.com/cann/amct/issues) | 问题反馈、功能建议、技术讨论 |
| [GitCode Discussions](https://gitcode.com/cann/amct/discussions) | 经验分享、最佳实践、社区互动 |
| [SIG Discussions](https://gitcode.com/cann/community/blob/master/CANN/sigs/tools/README.md) | 技术决策、问题处理、项目落地 |

## 🤝 参与贡献

欢迎贡献代码、算法与文档，详见 [贡献指南](CONTRIBUTING.md)：

- **简单 bug 修复**：直接提交 PR
- **新特性 / 接口变更**：先在 [Issue](https://gitcode.com/cann/amct/issues) 中讨论方案，达成共识后再提交 PR
- **代码风格**：C/C++ 遵循 Google 规范（基于 `.clang-format`），Python 遵循 PEP8；提交前启用 `pre-commit`

## 🙏 致谢

感谢所有为 AMCT 做出贡献的开发者！

本项目受启发于以下开源项目：

- [AWQ](https://github.com/mit-han-lab/llm-awq) - 激活感知权重量化
- [GPTQ](https://github.com/IST-DASLab/gptq) - GPTQ 实现参考
- [SmoothQuant](https://github.com/mit-han-lab/smoothquant) - 平滑激活量化
- [FlatQuant](https://github.com/ruikangliu/FlatQuant) - 矩阵平坦量化

## 📝 许可证

本项目基于 [Apache 2.0](LICENSE) 协议开源。使用前请阅读 [安全声明](SECURITY.md) 与 [免责声明](DISCLAIMER.md)。
