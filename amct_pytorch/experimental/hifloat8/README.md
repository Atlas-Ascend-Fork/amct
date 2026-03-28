# HiFloat8 伪量化模块

本模块提供了基于 HiFloat8 格式的 PyTorch 模型伪量化功能，包括 HiFloat8 与 FP32/FP16/BF16 格式的转换、伪量化线性模块以及测试脚本。

*注意* 
当前HiFloat8格式转换完全基于 CPU 实现，使用OpenMP进行并行加速。相比于NPU硬件加速，其性能会有一定差距。建议：
- 在小批量数据处理时使用
- 用于模型精度验证和调试
- 生产环境建议使用NPU硬件加速版本

## 文件说明

```
├── hifloat8_cast.cpp               # HiFloat8与FP32/FP16/BF16格式转换的C++实现
├── setup.py                        # Python 编译脚本
├── build.sh                        # Shell 编译脚本
├── hifloat8_fakequant_linear.py    # HiFloat8 伪量化线性模块
├── test.py                         # 测试脚本
├── utils.py                        # 工具函数
└── README.md
```

## 功能说明
### 1、HiFloat8格式转换

#### 1.1 编译 C++ 扩展

通过shell编译 C++ 扩展模块：

```bash
# 设置并行编译任务数（可选，默认为 8）
export AMCT_NUM_BUILD_JOBS=8

# 编译扩展
./build.sh
```

或直接使用Python编译：

```bash
python setup.py build_ext --inplace
```

#### 1.2 转换函数说明

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

### 2、伪量化模块

`Hifloat8FakequantLinear` 是一个基于 HiFloat8 格式的伪量化线性层模块，继承自 `BaseQuantizeModule`。

- **权重量化**：采用 per-tensor 量化策略
- **量化范围**：HiFloat8 高精度范围为 16
- **激活量化**：采用数据直转策略

### 3、 算法注册

amct_pytorch 提供了灵活的算法注册机制，允许用户自定义量化算法并将其集成到量化流程中。
通过 [`algorithm_register`](../../../docs/api/algorithm_register.md) 函数，可以将自定义的量化模块注册到系统中。

```python
import torch
from hifloat8_fakequant_linear import Hifloat8FakequantLinear
import amct_pytorch as amct

# 注册 HiFloat8 伪量化算法
amct.algorithm_register('hifloat8_fakequant', 'Linear', Hifloat8FakequantLinear, None)
```

### 4、测试脚本

测试脚本执行以下步骤：
1. **加载模型**：从指定路径加载预训练模型和分词器
2. **注册算法**：注册 HiFloat8 伪量化算法
3. **模型量化**：使用 HiFloat8 格式对模型进行伪量化
4. **数据集加载**：加载 Wikitext-2 测试数据集
5. **性能评估**：计算量化后模型的困惑度

### 5、完整使用流程

#### 5.1 准备环境
1. 安装amct工具，参考[工具构建](../../../docs/build.md)
2. 安装其他依赖
```bash
pip install transformers datasets
```
3. 编译 HiFloat8 扩展
```bash
cd amct_pytorch/experimental/hifloat8
./build.sh
```

#### 5.2 运行测试
```bash
# 运行测试脚本
python test.py --model_path /path/to/qwen/model
```
