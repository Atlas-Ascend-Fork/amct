# Vendored & adapted: Torch-Pruning

本目录是 **Torch-Pruning** 的 static 内置(vendored)副本，按其原始许可证（MIT）
改编并随附于 AMCT，用于提供依赖图(DepGraph)结构化剪枝能力，**不引入任何外部
运行时依赖**。

- 上游项目: Torch-Pruning — https://github.com/VainF/Torch-Pruning
- 版本: 1.6.1（API 报告 1.6.0）
- 版权: Copyright (c) 2019 Gongfan Fang（见同目录 `LICENSE`、`AUTHORS`）
- 许可证: MIT（保留原始版权与许可声明，见 `LICENSE`）

## 改编说明

- 以 static 方式整体内置该包，未改动其内部实现；仅置于
  `amct_pytorch.experimental.pruning._vendor.torch_pruning` 命名空间下，
  包内全部为相对 import，可原样工作。
- AMCT 通过 `methods/cnn_depgraph.py` 调用本副本的 `DependencyGraph` /
  `pruner.MetaPruner` / `importance.*` 完成 CNN 通道剪枝；AMCT 自身代码版权
  归 Huawei Technologies Co., Ltd.，与本目录的上游版权相互独立。
- 原始 MIT 许可证文本完整保留于 `LICENSE`，分发时一并保留。
