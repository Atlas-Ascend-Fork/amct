# 简介

本项目提供了不同场景的调用样例，搭建完环境后，可以根据实际场景尝试运行：

| 样例 | 算法 | 说明 |
| ------ |------ |------ |
| 使用MIN-MAX算法量化模型 | [Min-Max](algorithms/minmax/README_CN.md) | 基于极值的简单量化，入门首选 |
| 使用AWQ算法量化模型 | [AWQ](algorithms/awq/README_CN.md) | 激活感知的权重量化，适合大模型 PTQ |
| 使用GPTQ算法量化模型 | [GPTQ](algorithms/gptq/README_CN.md) | 基于二阶信息的权重量化，逐层优化 |
| 使用SmoothQuant算法量化模型 | [SmoothQuant](algorithms/smoothquant/README_CN.md) | 平滑激活分布的 W8A8 量化 |
|使用Cast直转算法量化模型 | [Cast](algorithms/cast/README_CN.md) | HiFloat8 数据直转 |
| 使用Quantile算法量化模型 | [Quantile](algorithms/quantile/README_CN.md) | HiFloat8 分位量化 |
| 使用ofmr算法量化模型 | [OFMR](algorithms/ofmr/README_CN.md) | 输出特征 Min-Max 量化 |
| 使用mxquant算法量化模型 | [MXQuant](algorithms/mxquant/README_CN.md) | 微缩浮点量化（MXFP8/MXFP4） |
| 使用FlatQuant算法量化模型 | [FlatQuant](algorithms/flatquant/README_CN.md) | 通过仿射变换平整化分布的量化 |
