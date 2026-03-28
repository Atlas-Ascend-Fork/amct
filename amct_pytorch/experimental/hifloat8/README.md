# HiFloat8 伪量化模块

本模块提供了基于 HiFloat8 格式的 PyTorch 模型伪量化功能，包括 HiFloat8 与 FP32/FP16/BF16 格式的转换、伪量化线性模块以及测试脚本。

## 文件说明

```
├── hifloat8_cast.cpp               # HiFloat8 与 FP32/FP16/BF16 格式转换的 C++ 实现
├── setup.py                        # Python 编译脚本
├── build.sh                        # Shell 编译脚本
├── hifloat8_fakequant_linear.py    # HiFloat8 伪量化线性模块
├── test.py                         # 测试脚本
├── utils.py                        # 工具函数
└── README.md
```

## 1、HiFloat8 格式转换

### 1.1 性能说明

**重要提示**：当前 HiFloat8 格式转换完全基于 CPU 实现，使用 OpenMP 进行并行加速。相比于 NPU 硬件加速，CPU 实现的性能会有一定差距。建议：
- 在小批量数据处理时使用
- 用于模型精度验证和调试
- 生产环境建议使用 NPU 硬件加速版本

### 1.2 编译 C++ 扩展

首先需要编译 C++ 扩展模块：

```bash
# 设置并行编译任务数（可选，默认为 8）
export AMCT_NUM_BUILD_JOBS=8

# 编译扩展
./build.sh
```

或直接使用 Python：

```bash
python setup.py build_ext --inplace
```

### 1.3 转换函数

编译成功后，可以使用以下转换函数：

```python
import torch
import hifloat8_cast

# FP32/FP16/BF16 转 HiFloat8
float32_tensor = torch.randn(10, 10, dtype=torch.float32)
hifloat8_tensor = hifloat8_cast.float_to_hifloat8(float32_tensor)

# HiFloat8 转 FP32
float32_tensor = hifloat8_cast.hifloat8_to_float32(hifloat8_tensor)

# HiFloat8 转 FP16
float16_tensor = hifloat8_cast.hifloat8_to_float16(hifloat8_tensor)

# HiFloat8 转 BFloat16
bfloat16_tensor = hifloat8_cast.hifloat8_to_bfloat16(hifloat8_tensor)
```

## 2、算法注册机制

### 2.1 算法注册概述

amct_pytorch 提供了灵活的算法注册机制，允许用户自定义量化算法并将其集成到量化流程中。通过 `algorithm_register` 函数，可以将自定义的量化模块注册到系统中。

### 2.2 算法注册函数

```python
amct.algorithm_register(algorithm_name, layer_type, quant_module, deploy_module)
```

**参数说明：**

- `algorithm_name` (str) - 算法名称，用于在配置文件中引用
- `layer_type` (str) - 目标层类型，如 'Linear', 'Conv2d' 等
- `quant_module` (class) - 量化模块类，继承自 `BaseQuantizeModule`
- `deploy_module` (class/None) - 部署模块

### 2.3 自定义量化模块要求

自定义量化模块需要满足以下要求：

1. **继承基类**：必须继承自 `BaseQuantizeModule`
2. **实现构造函数**：接收原始模块、层名称和量化配置
3. **实现前向传播**：定义 `forward` 方法实现量化逻辑

**示例结构：**

```python
from amct_pytorch.quantize_op.base_quant_module import BaseQuantizeModule

class CustomQuantModule(BaseQuantizeModule):
    def __init__(self, ori_module, layer_name, quant_config):
        super().__init__(ori_module, layer_name, quant_config)
        # 初始化量化参数
        self.layer_name = layer_name
        self.device = ori_module.weight.device
        
        # 处理权重量化
        self.fakequant_weight = self._quantize_weight(ori_module.weight)
        
        # 处理偏置
        self.bias = ori_module.bias
    
    def _quantize_weight(self, weight):
        # 实现权重量化逻辑
        return quantized_weight
    
    def forward(self, x):
        # 实现前向传播逻辑
        quantized_x = self._quantize_input(x)
        return self._compute_output(quantized_x)
```

### 2.4 HiFloat8 伪量化模块

### 2.4.1 模块介绍

`Hifloat8FakequantLinear` 是一个基于 HiFloat8 格式的伪量化线性层模块，继承自 `BaseQuantizeModule`。

### 2.4.2 量化策略

- **权重量化**：采用 per-tensor 量化策略
- **输入量化**：采用 tensor 量化策略
- **量化范围**：HiFloat8 高精度范围为 16

### 2.4.3 注册和使用示例

```python
import torch
from hifloat8_fakequant_linear import Hifloat8FakequantLinear
import amct_pytorch as amct

# 注册 HiFloat8 伪量化算法
amct.algorithm_register('hifloat8_fakequant', 'Linear', Hifloat8FakequantLinear, None)

# 配置量化参数
hifloat8_fakequant_cfg = {
    'quant_cfg': {
        'weights': {
            'type': 'hifloat8',
            'symmetric': True,
            'strategy': 'channel',
        },
        'inputs': {
            'type': 'hifloat8',
            'symmetric': True,
            'strategy': 'tensor',
        },
    },
    'algorithm': {'hifloat8_fakequant'},
    'skip_layers': {'lm_head'}
}

# 对模型进行量化
amct.quantize(model, hifloat8_fakequant_cfg)
```

### 2.5 配置文件说明

量化配置文件包含以下关键部分：

```python
{
    'quant_cfg': {
        'weights': {
            'type': 'hifloat8',           # 量化类型
            'symmetric': True,             #   是否对称量化
            'strategy': 'channel',         # 量化策略：'channel' 或 'tensor'
        },
        'inputs': {
            'type': 'hifloat8',           # 输入量化类型
            'symmetric': True,             # 是否对称量化
            'strategy': 'tensor',          # 量化策略
        },
    },
    'algorithm': {'hifloat8_fakequant'},  # 使用的算法名称（与注册时一致）
    'skip_layers': {'lm_head'}            # 跳过量化的层名称集合
}
```

## 3、测试脚本

### 3.1 测试脚本说明

`test.py` 提供了完整的测试流程，包括：
1. 加载预训练模型（支持 Qwen 等模型）
2. 对模型进行 HiFloat8 伪量化
3. 在 Wikitext-2 数据集上评估困惑度（Perplexity）

### 3.2 运行测试

```bash
python test.py --model_path /path/to/model
```

### 3.3 测试流程

测试脚本执行以下步骤：

1. **加载模型**：从指定路径加载预训练模型和分词器
2. **注册算法**：注册 HiFloat8 伪量化算法
3. **模型量化**：使用 HiFloat8 格式对模型进行伪量化
4. **数据集加载**：加载 Wikitext-2 测试数据集
5. **性能评估**：计算量化后模型的困惑度

### 3.4 工具函数

`utils.py` 提供了以下工具函数：

- `get_qwen(model_path, seqlen=2048)` - 加载 Qwen 模型
- `get_test_dataset(enc)` - 加载测试数据集
- `test_ppl(model, testenc)` - 计算模型困惑度

## 4、完整使用流程

### 4.1 准备环境

```bash
# 安装依赖
pip install torch transformers datasets amct_pytorch

# 编译 HiFloat8 扩展
cd amct_pytorch/experimental/hifloat8
./build.sh
```

### 4.2 运行测试

```bash
# 运行测试脚本
python test.py --model_path /path/to/qwen/model
```

## 5、技术细节

### 5.1 HiFloat8 格式特点

- 8 位浮点格式
- 动态指数和尾数位分配
- 适用于深度学习模型量化
- 支持正负无穷和 NaN

### 5.2 量化过程

1. **权重量化**：
   - 计算权重最大值
   - 根据量化范围计算缩放因子
   - 将权重转换为 HiFloat8 格式
   - 反向转换用于前向传播

2. **输入量化**：
   - 在前向传播时将输入转换为 HiFloat8
   - 反向转换用于计算

### 5.3 性能优势

- 相比传统 8 位量化，HiFloat8 提供更好的数值精度
- 动态位分配优化了不同数值范围的表示
- 适合大语言模型等对精度要求较高的场景

## 6、注意事项

1. **性能考虑**：HiFloat8 转换基于 CPU 实现，相比 NPU 硬件加速性能较慢，适合精度验证和调试
2. 确保已正确编译 C++ 扩展模块
3. 测试前需要安装 transformers 和 datasets 库
4. 模型路径需要指向正确的预训练模型
5. 量化过程可能需要较多内存，建议使用 NPU

## 7、依赖项

- Python 3.x
- PyTorch
- transformers
- datasets
- amct_pytorch
- C++ 编译器（支持 OpenMP）
