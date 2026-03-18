# HIF8 Cast 功能说明

## 概述

HIF8 Cast功能实现了BF16/FP16网络到HIF8网络的直接转换，提供以下特性：

1. **离线权重量化**: 权重在量化阶段直接转换为目标HIF8数据类型
2. **在线激活转换**: 激活数据通过clip+cast算子实现在线直转功能
3. **无需校准数据**: Cast模式不需要额外的校准数据
4. **部署即量化**: quantize后的模型即为部署模型

## 架构设计

### 1. 自定义部署算子: HIF8CastLinear

**文件**: `amct_pytorch/deploy_op/hif8_castra_linear.py`

**功能**:
- 权重离线量化为HIF8格式
- 激活数据在线clip+cast到HIF8
- 使用NPU算子进行高效计算

**关键实现**:
```python
class HIF8CastLinear(nn.Module):
    def __init__(self, quant_module):
        # 离线权重量化
        weight_tensor = quant_weight(weight, HIFLOAT8, scale_w, offset_w)
        self.register_buffer(''quantized_weight'', weight_tensor)
    
    def forward(self, inputs):
        # 在线激活clip+cast
        clipped_inputs = torch.clamp(inputs, -240.0, 240.0)
        cast_inputs = torch_npu.npu_dtype_cast(clipped_inputs, dtype=torch_npu.hifloat8)
        # NPU计算
        output = torch_npu.npu_weight_quant_batchmatmul(...)
```

### 2. 量化算子: HIF8CastQuant

**文件**: `amct_pytorch/quantize_op/hif8_cast_quant.py`

**功能**:
- 计算权重量化因子(scale/offset)
- 支持tensor和channel级别的量化策略
- 在校准阶段演示clip+cast操作

### 3. 算法注册

**文件**: `amct_pytorch/algorithm/__init__.py`

```python
AlgorithmRegistry.register(''cast'', ''Linear'', HIF8CastQuant, HIF8CastLinear)
```

- src_op: Linear
- quant_op: HIF8CastQuant (校准算子)
- deploy_op: HIF8CastLinear (部署算子)

### 4. 配置模板

**文件**: `amct_pytorch/config/config.py`

```python
HIF8_CAST_CFG = {
    ''batch_num'': 1,
    ''quant_cfg'': {
        ''weights'': {
            ''type'': ''hifloat8'',
            ''symmetric'': True,
            ''strategy'': ''channel'',
        },
        ''inputs'': {
            ''type'': ''hifloat8'',
            ''symmetric'': True,
            ''strategy'': ''tensor'',
        },
    },
    ''algorithm'': {''cast''}
}
```

### 5. convert接口增强

**文件**: `amct_pytorch/quantize.py`

**功能**: 检测cast模式，提示用户无需额外转换

```python
def convert(model):
    # 检测是否为cast模式
    for name, module in model.named_modules():
        if isinstance(module, HIF8CastQuant):
            LOGGER.logi(''The source network is already a deployment model...'')
            return
    # 正常转换流程
    ...
```

## 使用方法

### 基本用法

```python
import torch
import torch.nn as nn
import amct_pytorch as amct

# 1. 创建原始模型
model = MyModel()
model.eval()

# 2. 使用HIF8_CAST_CFG配置
cfg = amct.HIF8_CAST_CFG

# 3. 调用quantize完成转换
amct.quantize(model, config=cfg)

# 4. 模型已为部署状态，可直接使用
output = model(input_data)

# 5. 可选：调用convert（会提示无需转换）
amct.convert(model)
```

### 自定义配置

```python
custom_cfg = {
    ''batch_num'': 1,
    ''quant_cfg'': {
        ''weights'': {
            ''type'': ''hifloat8'',
            ''symmetric'': True,
            ''strategy'': ''channel'',  # 或 ''tensor''
        },
        ''inputs'': {
            ''type'': ''hifloat8'',
            ''symmetric'': True,
            ''strategy'': ''tensor'',
        },
    },
    ''algorithm'': {''cast''},
    ''skip_layers'': {''lm_head''}  # 可选：跳过某些层
}

amct.quantize(model, config=custom_cfg)
```

## 技术细节

### Clip范围

HIF8的clip范围设置为[-240.0, 240.0]，这是基于HIF8数据类型的动态范围特性。

### 权重量化策略

- **tensor策略**: 整个权重张量使用统一的scale/offset
- **channel策略**: 每个输出通道使用独立的scale/offset（推荐）

### 激活量化策略

- **tensor策略**: 整个激活张量使用统一的scale/offset

### NPU算子使用

- `torch_npu.npu_dtype_cast`: 数据类型转换
- `torch_npu.npu_weight_quant_batchmatmul`: 权重量化矩阵乘法

## 测试

运行测试脚本：

```bash
python test_hif8_cast.py
```

测试内容：
1. 模型量化前后对比
2. 前向传播测试
3. 数据类型验证
4. convert接口测试

## 优势

1. **简化流程**: quantize即完成部署，无需额外convert步骤
2. **高效转换**: 利用NPU硬件加速的cast操作
3. **精度保证**: 通过clip操作保证数据在有效范围内
4. **零拷贝**: 权重离线量化，运行时无额外开销
5. **易于使用**: 统一的API接口，与现有算法保持一致

## 限制

1. 仅支持Linear层
2. 仅支持HIF8数据类型
3. 需要NPU硬件支持
4. 输入数据应为BF16或FP16格式

## 文件清单

新增文件：
- `amct_pytorch/deploy_op/hif8_cast_linear.py` - HIF8部署算子
- `amct_pytorch/quantize_op/hif8_cast_quant.py` - HIF8量化算子
- `test_hif8_cast.py` - 测试脚本
- `HIF8_CAST_README.md` - 本文档

修改文件：
- `amct_pytorch/algorithm/__init__.py` - 注册cast算法
- `amct_pytorch/config/config.py` - 添加HIF8_CAST_CFG
- `amct_pytorch/config/__init__.py` -导出配置
- `amct_pytorch/deploy_op/__init__.py` - 导出部署算子
- `amct_pytorch/quantize_op/__init__.py` - 导出量化算子
- `amct_pytorch/quantize.py` - 增强convert接口
- `amct_pytorch/__init__.py` - 导出公共API