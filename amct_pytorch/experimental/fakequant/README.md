# FakeQuant 模拟伪量化工具包

本目录提供模拟伪量化（Fake Quantization）工具包，用于在硬件尚不原生支持某些量化格式时，在软件侧模拟对应格式的量化精度，以便进行精度验证与算法评估。

典型场景包括：目标格式尚未合入 NPU 算子栈、或当前环境无法启用对应低比特计算单元时，仍可通过伪量化路径复现近似的量化数值行为。

## 目录说明

```
fakequant/
├── mxfp4_ascendc/   # MXFP4 Ascend-C 伪量化算子（目录对齐 amct_ops/hifloat8_cast）
│   ├── op_kernel/   # device kernel + tiling
│   ├── op_extension/# Torch host + TORCH_LIBRARY 注册
│   ├── python/mxfp4/# Python 包装
│   ├── reference/   # 纯 PyTorch 参考实现
│   └── tests/
└── README.md
```

## 说明

- 本模块属于试验特性（`experimental`），接口与实现可能随硬件能力演进而调整。
- 伪量化结果用于精度对齐与方案验证，不等同于目标硬件上的真实低比特算子性能表现。
- MXFP4 算子实现参考 `amct_ops/hifloat8_cast` 的三层结构，但因试验阶段暂不迁入 `amct_ops/`。
