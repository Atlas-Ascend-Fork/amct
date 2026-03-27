# Quantile量化算法实现总结

## 功能概述

成功实现了Quantile分位量化算法，支持HIF8数据类型，提供多种量化粒度组合。

## 实现的功能

### 1. Quantile量化算子 (amct_pytorch/quantize_op/quantile_module.py)

**核心特性**:
- **权重最大幅值缩放到16**: 权重最大值除以16作为HIF8的scale
- **激活量化因子计算**: 激活取迭代数据最大值的0.99，叠加之前数据0.01作为max值
- **支持HIF8数据类型**: 专门针对HIF8格式优化
- **量化结果处理不变**: 权重量化因子和激活量化因子保存到量化部署模型

**实现原理**:
```python
class QuantileQuant(BaseQuantizeModule):
    """
    Quantile量化算子，实现：
    1. 权重最大值缩放到16计算scale
    2. 激活量化：max = 0.99 * current_max + 0.01 * previous_max
    3. 支持HIF8数据类型
    4. 支持多种量化粒度组合
    """
```

**关键功能**:
- `calculate_hif8_scale()`: HIF8专用scale计算，权重最大值/16
- `calculate_scale_offset()`: 通用scale和offset计算
- 支持激活量化时的累积最大值计算
- 支持仅权重量化模式

### 2. NPU部署算子 (amct_pytorch/deploy_op/npu_quantile_quantized_linear.py)

**核心特性**:
- **量化因子保存**: 权重量化因子和激活量化因子保存到部署模型
- **支持HIF8数据类型**: 使用torch_npu.hifloat8
- **支持多种量化粒度**: weight per-tensor/per-channel, activation per-tensor/static per-token/dynamic per-token
- **NPU算子调用**: 使用npu_weight_quant_batchmatmul进行高效计算

**实现原理**:
```python
class NpuQuantileQuantizedLinear(nn.Module):
    """
    NPU部署算子，实现：
    1. 保存量化因子到buffer
    2. 支持权重量化和激活量化
    3. 支持多种量化粒度
    4. 使用NPU专用算子进行推理
    """
```

### 3. 算法注册 (amct_pytorch/algorithm/__init__.py)

**注册信息**:
```python
AlgorithmRegistry.register('quantile', 'Linear', QuantileQuant, 
    [NpuWeightQuantizedLinear, NpuQuantileQuantizedLinear])
```

**参数说明**:
- `name='quantile'`: 算法名称
- `src_ops=['Linear']`: 支持的源算子类型
- `quant_op=QuantileQuant`: 量化算子
- `deploy_op=[NpuWeightQuantizedLinear, NpuQuantileQuantizedLinear]`: 部署算子（权重量化和全量化）

### 4. 配置定义 (amct_pytorch/config/config.py)

**新增配置**:
```python
HIFP8_QUANTILE_CFG = {
    'batch_num': 1,
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
    'algorithm': {'quantile'},
    'skip_layers': {'lm_head'}
}
```

### 5. 支持的量化粒度组合

根据描述要求，支持以下组合：

| 权重量化粒度 | 激活量化粒度 | 支持状态 |
|:--| :-: | :-: |
| per-tensor | per-tensor | ✓ |
| per-channel | per-tensor | ✓ |
| per-tensor | static per-token | ✓ |
| per-channel | static per-token | ✓ |
| per-tensor | dynamic per-token | ✓ |
| per-channel | dynamic per-token | ✓ |

**仅权重量化支持**:
- weight per-tensor
- weight per-channel

### 6. 变量支持更新 (amct_pytorch/utils/vars.py)

**更新内容**:
- 在`ALGORITHM_SUPPORTED_QUANT_TYPE_COMB`中添加quantile算法支持
- 支持组合：`'hifloat8 hifloat8': ['ofmr', 'cast', 'quantile']`
- 支持组合：`'NOT_QUANTIZE hifloat8': ['ofmr', 'gptq', 'cast', 'quantile']`

## 技术细节

### 权重量化流程

1. **获取权重最大值**: 根据量化粒度获取权重最大值
2. **计算HIF8 scale**: scale = weight_max / 16.0
3. **量化权重**: 使用scale对权重进行量化
4. **保存量化因子**: scale保存到量化算子

### 激活量化流程

1. **收集激活数据**: 在forward中校准批次内收集激活数据
2. **计算最大值**: 计算当前批次的最大值
3. **累积最大值**: quantile_max = 0.99 * current_max + 0.01 * previous_max
4. **计算scale**:根据quantile_max计算scale
5. **保存量化因子**: scale保存到量化算子

### HIF8量化公式

**权重量化**:
```
scale = weight_max / 16.0
quantized_weight = round(weight / scale)
```

**激活量化**:
```
quantile_max = 0.99 * current_max + 0.01 * previous_max
scale = quantile_max / 16.0
quantized_activation = round(activation / scale)
```

## 使用示例

### 基本使用

```python
from amct_pytorch import quantize, convert, HIFP8_QUANTILE_CFG

# 原始模型
model = YourModel().to(torch.float16)

# 使用Quantile量化
quantize(model, HIFP8_QUANTILE_CFG)

# 转换为部署模型
convert(model)
```

### 权重量化

```python
config = {
    'batch_num': 1,
    'quant_cfg': {
        'weights': {
            'type': 'hifloat8',
            'symmetric': True,
            'strategy': 'channel',  # per-channel
        },
    },
    'algorithm': {'quantile'}
}

quantize(model, config)
```

### 全量化

```python
config = {
    'batch_num': 1,
    'quant_cfg': {
        'weights': {
            'type': 'hifloat8',
            'symmetric': True,
            'strategy': 'channel',
        },
        'inputs': {
            'type': 'hifloat8',
            'symmetric': True,
            'strategy': 'tensor',  # per-tensor
        },
    },
    'algorithm': {'quantile'}
}

quantize(model, config)
```

### 激活per-token量化

```python
config = {
    'batch_num': 1,
    'quant_cfg': {
        'weights': {
            'type': 'hifloat8',
            'symmetric': True,
            'strategy': 'channel',
        },
        'inputs': {
            'type': 'hifloat8',
            'symmetric': True,
            'strategy': 'token',  # per-token
        },
    },
    'algorithm': {'quantile'}
}

quantize(model, config)
```

## 修改的文件

1. **amct_pytorch/quantize_op/quantile_module.py**
   - 新增QuantileQuant量化算子
   - 实现权重和激活量化逻辑
   - 支持HIF8数据类型

2. **amct_pytorch/deploy_op/npu_quantile_quantized_linear.py**
   - 新增NPU部署算子
   - 保存量化因子到buffer
   - 支持多种量化粒度

3. **amct_pytorch/algorithm/__init__.py**
   - 注册quantile算法
   - 添加到BUILT_IN_ALGORITHM列表

4. **amct_pytorch/config/config.py**
   - 新增HIFP8_QUANTILE_CFG配置
   - 提供内置配置模板

5. **amct_pytorch/config/__init__.py**
   - 导出HIFP8_QUANTILE_CFG配置

6. **amct_pytorch/utils/vars.py**
   - 更新ALGORITHM_SUPPORTED_QUANT_TYPE_COMB
   - 添加quantile算法支持

7. **tests/amct_pytorch/test_quantile.py**
   - 新增测试用例
   - 测试权重量化和全量化
   - 测试不同量化粒度

## 与算法描述对应

根据提供的描述，实现的功能包括：

✅ **1. quantize接口新增quantile_quant算法选项，支持HIF8数据类型**
   - 在algorithm/__init__.py中注册quantile算法
   - 支持HIF8数据类型量化

✅ **2. 权重最大幅值缩放到16，激活取迭代数据最大值的0.99，叠加之前数据0.01作为max值，计算量化scale**
   - 在quantile_module.py中实现calculate_hif8_scale()
   - 权重scale = weight_max / 16.0
   - 激活quantile_max = 0.99 * current_max + 0.01 * previous_max

✅ **3. 量化结果处理不变，权重量化因子，激活量化因子保存量化部署模型**
   - 在npu_quantile_quantized_linear.py中实现
   - 量化因子保存到buffer
   - 支持权重量化和全量化

✅ **4. 支持activation per-tensor, weight per-tensor/per-channel，支持activation static per-token, weight per-tensor/per-channel，支持activation dynamic per-token, weight per-tensor/per-channel，支持仅权重量化，weight per-tensor/per-channel**
   - 支持weight per-tensor和per-channel
   - 支持activation per-tensor、static per-token、dynamic per-token
   - 支持仅权重量化模式

## 优势

1. **灵活的量化粒度**: 支持多种量化粒度组合
2. **HIF8优化**: 专门针对HIF8格式优化
3. **累积最大值**: 激活量化使用累积策略提高精度
4. **简单易用**: 提供内置配置，使用简单
5. **完全兼容**: 与现有量化流程完全兼容

## 注意事项

1. **数据类型要求**: 源模型建议使用BF16或FP16格式
2. **NPU环境**: 需要安装CANN工具包和torch_npu
3. **HIF8专用**: 当前主要针对HIF8数据类型优化
4. **校准数据**: 需要校准数据收集激活统计信息
5. **部署限制**: 量化后的模型只能在昇腾NPU上部署

## 总结

成功实现了Quantile分位量化算法，支持HIF8数据类型，提供多种量化粒度组合。该算法通过累积最大值策略计算激活量化因子，权重最大值缩放到16计算权重量化因子，实现了与描述要求完全一致的功能。

**主要特性**:
- ✓ 支持HIF8数据类型
- ✓ 权重最大值缩放到16
- ✓ 激活累积最大值策略
- ✓ 支持多种量化粒度组合
- ✓ 支持仅权重量化和全量化
- ✓ 量化因子保存到部署模型

**适用场景**:
- HIF8数据类型量化
- 需要灵活量化粒度的场景
- 大模型量化优化
- 昇腾NPU平台部署
