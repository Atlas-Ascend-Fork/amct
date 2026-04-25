# amct_llm 目录重构说明

## 重构日期
2026-04-24

## 重构目标
按照《AMCT-Q融入amct_open重构方案.md》的架构设计，将 amct_llm 仓库重构为分层架构，以便后续 AMCT-Q 的内容可以融合进来。

## 重构后的目录结构

```
amct_pytorch/
├── common/                      # Layer 0: 共享基础
│   ├── config/                  # 配置系统（从原 config/ 迁移）
│   ├── utils/                   # 工具函数（从原 utils/ 迁移）
│   ├── datasets/                # 数据集模块（新增基类）
│   ├── evaluate/                # 评测模块（新增基类）
│   ├── models/                  # 模型适配层（新增基类）
│   │   └── classic/             # Classic模型适配器
│   └── types/                   # 类型定义（新增）
│
├── quantization/                # Layer 1: 量化核心
│   ├── dtypes/                  # 数据类型（待AMCT-Q迁移）
│   └── modules/                 # 量化模块（待AMCT-Q迁移）
│
├── algorithms/                  # Layer 2: 统一算法中心
│   ├── common/                  # 算法基类
│   ├── quant/                   # PTQ量化算法
│   │   └── solver/              # PTQ求解器
│   └── workflow/                # 工作流基类
│
├── classic/                     # Layer 3: 通用模型压缩
│   └── ptq/
│       ├── quantize_op/         # 校准算子（从原 quantize_op/ 迁移）
│       ├── deploy_op/           # 部署算子（从原 deploy_op/ 迁移）
│       ├── optimizer/           # 图优化（从原 optimizer/ 迁移）
│       ├── graph_based/         # 图压缩（从原 graph_based_compression/ 迁移）
│       └── quantize.py          # classic PTQ入口
│
├── experimental/                # 实验性功能（保持不变）
│
└── __init__.py                  # 统一入口
```

## 文件迁移映射

| 原路径 | 新路径 |
|-------|-------|
| `amct_pytorch/config/` | `amct_pytorch/common/config/` |
| `amct_pytorch/utils/` | `amct_pytorch/common/utils/` |
| `amct_pytorch/quantize_op/` | `amct_pytorch/classic/ptq/quantize_op/` |
| `amct_pytorch/deploy_op/` | `amct_pytorch/classic/ptq/deploy_op/` |
| `amct_pytorch/optimizer/` | `amct_pytorch/classic/ptq/optimizer/` |
| `amct_pytorch/graph_based_compression/` | `amct_pytorch/classic/ptq/graph_based/` |
| `amct_pytorch/quantize.py` | `amct_pytorch/classic/ptq/quantize.py` |

## 新增模块

### common/datasets/
- `base_provider.py` - 数据提供器基类
- `calibration_data.py` - 校准数据提供器

### common/evaluate/
- `base_evaluator.py` - 评测器基类

### common/models/
- `base_adapter.py` - 模型适配器基类
- `model_registry.py` - 模型注册中心（ClassicAdapterRegistry）

### common/types/
- `quant_config.py` - 量化配置类型

### quantization/
- `dtypes/` - 数据类型（待AMCT-Q迁移）
- `modules/` - 量化模块（待AMCT-Q迁移）

### algorithms/
- `common/base_algo.py` - 算法基类
- `quant/solver/` - 求解器（BaseSolver, GlobalSolver）
- `workflow/base_workflow.py` - 工作流基类

## Import 路径变更

| 原路径 | 新路径 | 影响文件数 |
|-------|-------|-----------|
| `from amct_pytorch.config` | `from amct_pytorch.common.config` | 7 |
| `from amct_pytorch.utils` | `from amct_pytorch.common.utils` | 54 |
| `from amct_pytorch.quantize_op` | `from amct_pytorch.classic.ptq.quantize_op` | 25 |
| `from amct_pytorch.deploy_op` | `from amct_pytorch.classic.ptq.deploy_op` | 8 |
| `from amct_pytorch.optimizer` | `from amct_pytorch.classic.ptq.optimizer` | 6 |
| `from amct_pytorch.graph_based_compression` | `from amct_pytorch.classic.ptq.graph_based` | 2 |

**验证结果**：旧路径数量为 0，所有导入已成功替换。

## 分层架构约束

```
Layer 0: common          （无外部依赖）
    ├── config/
    ├── utils/
    ├── datasets/
    ├── evaluate/
    ├── models/
    └── types/

Layer 1: quantization    （依赖 common）
    ├── dtypes/
    └── modules/

Layer 2: algorithms      （依赖 common, quantization）
    ├── quant/
    ├── common/
    └── workflow/

Layer 3: classic         （依赖 algorithms, quantization, common）
```

**约束规则**：
- 低层模块不可依赖高层模块
- 同层模块之间可相互依赖
- 使用 `import-linter` 在 CI 中检查层级约束

## 后续工作

1. **迁移 AMCT-Q 内容**：
   - LLM模型适配器 → `common/models/llm/`
   - 量化算法 → `algorithms/quant/`
   - quantization/dtypes → `quantization/dtypes/`
   - quantization/modules → `quantization/modules/`
   - LLM工作流 → 新增 `llm/` 模块

2. **创建兼容性测试**：确保原有接口 `quantize()`, `convert()`, `algorithm_register()` 正常工作

3. **配置 import-linter**：确保分层约束不被破坏