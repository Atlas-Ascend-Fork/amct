# AMCT

## 🔥Latest News

-   [2025/12/22] AMCT项目首次上线。

## 🚀概述

AMCT（Ascend Model Compression Toolkit，简称AMCT）是一款昇腾AI处理器亲和的深度学习模型压缩工具包，提供多种模型量化压缩特性。压缩后模型体积变小，部署到昇腾AI处理器可使能低比特运算，提高计算效率，达到性能提升的目标。AMCT部署架构图如下：

![](docs/figures/部署架构.png)

## 🔍目录结构

```
├── docs                           # AMCT工具文档介绍
├── README.md
├── build.sh                       # 工程编译脚本
├── requirements.txt               # 环境搭建需要的第三方依赖包
├── amct_pytorch                   # 源码
├── npu_ops                        # 算法对应的算子代码
├── examples                       # 端到端样例开发和调用示例
└── tests                          # 测试工程目录
```

## ⚡️快速开始

若您希望快速体验该工具的使用和样例执行，请访问如下文档获取简易教程：

-   [构建验证](docs/build.md)：介绍工具的完整构建、测试样例运行流程。
-   [样例执行](examples/README.md)：介绍如何端到端执行样例代码。

## 📖学习教程

若您希望深入了解AMCT工具的使用方法，请参见介绍：

-   [压缩概念介绍](docs/压缩概念.md)：介绍压缩特性相关的概念，比如什么是量化，什么是稀疏等。
-   [压缩特性介绍](docs/README.md)：介绍AMCT工具支持的压缩特性。
-   [压缩接口介绍](docs/api/README.md)：介绍使用的压缩接口。
-   [压缩算法介绍](docs/算法介绍.md)：介绍AMCT压缩过程中使用的算法，比如AWQ、GPTQ等。

## 🅿️ Pre-commit
pre-commit 是一个用于管理和维护 Git 预提交钩子（hooks）的框架，通过在代码提交前自动化执行代码检查、格式化和安全扫描，确保代码质量并统一团队规范，显著减少 CI/CD 流水线失败并提升协作效率。
本仓已配置pre-commit，用户可以参考CANN社区的[pre-commit配置指导书中第3章节](https://gitcode.com/cann/infrastructure/blob/main/docs/SC/pre-commit/pre-commit%E9%85%8D%E7%BD%AE%E6%8C%87%E5%AF%BC%E4%B9%A6.md#3-%E7%A4%BE%E5%8C%BA%E8%B4%A1%E7%8C%AE%E8%80%85%E4%BD%BF%E7%94%A8pre-commit%E8%83%BD%E5%8A%9B)安装pre-commit, 首次由于需要配置java，maven环境以及构建jar包，需要的时间比较长。

## 📝相关信息

-   [贡献指南](CONTRIBUTING.md)
-   [安全声明](SECURITY.md)
-   [许可证](LICENSE)

