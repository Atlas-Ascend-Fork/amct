# 简介

本项目提供了不同场景的调用样例，搭建完环境后，可以根据实际场景尝试运行：

| 样例 | 算法 | 说明 |
| ------ |------ |------ |
| 使用MIN-MAX算法量化模型 | [Min-Max](algorithms/minmax/README.md) | 基于极值的简单量化，入门首选 |
| 使用AWQ算法量化模型 | [AWQ](algorithms/awq/README.md) | 激活感知的权重量化，适合大模型 PTQ |
| 使用GPTQ算法量化模型 | [GPTQ](algorithms/gptq/README.md) | 基于二阶信息的权重量化，逐层优化 |
| 使用SmoothQuant算法量化模型 | [SmoothQuant](algorithms/smoothquant/README.md) | 平滑激活分布的 W8A8 量化 |
|使用Cast直转算法量化模型 | [Cast](algorithms/cast/README.md) | HiFloat8 数据直转 |
| 使用Quantile算法量化模型 | [Quantile](algorithms/quantile/README.md) | HiFloat8 分位量化 |
| 使用ofmr算法量化模型 | [OFMR](algorithms/ofmr/README.md) | 输出特征 Min-Max 量化 |
| 使用mxquant算法量化模型 | [MXQuant](algorithms/mxquant/README.md) | 微缩浮点量化（MXFP8/MXFP4） |
| 使用FlatQuant算法量化模型（试验特性） | [FlatQuant](algorithms/flatquant/README.md) | 通过仿射变换平整化分布的量化 |

> **注意**：标注"试验特性"的样例依赖 `amct_pytorch/experimental/` 目录下的内容，需使用
> `bash build.sh --torch --experimental`（或 `--pkg --experimental`）构建安装包后方可使用。
