# 量化配置模糊匹配功能实现文档

## 功能概述

实现了量化配置的模糊匹配功能，允许在 `quantize` 接口的 `config` 量化选项中使用通配符 `*` 来匹配多个层，从而为这些层应用相同的量化配置。

## 实现的文件

### 1. `amct_pytorch/config/utils.py`

新增 `match_fuzzy_pattern()` 函数：

```python
def match_fuzzy_pattern(layer_name, pattern):
    """
    Match layer name with pattern supporting wildcard '*'
    Params:
        layer_name: str, the full layer name (e.g., 'model.layers.0.self_attn.q_proj')
        pattern: str, the pattern to match (e.g., '*self_attn.q_proj.weights')
    Return: bool, True if pattern matches layer_name
    """
```

该函数使用 Python 标准库 `fnmatch` 模块实现通配符匹配，支持：
- 前缀通配符：`*down_proj` 匹配所有以 `down_proj` 结尾的层
- 后缀通配符：`model.layers.*` 匹配所有以 `model.layers.` 开头的层
- 中间通配符：`model.*.down_proj` 匹配 `model` 和 `down_proj` 之间的任意内容
- 多个通配符：`*.*.c.*` 匹配多个位置的通配符
- `.weights` 和 `.inputs` 后缀处理：自动识别并匹配层名称

### 2. `amct_pytorch/config/fields.py`

修改 `QuantCfgField` 类：

新增方法：
- `_extract_fuzzy_configs()`: 从配置中提取模糊匹配模式
- `get_fuzzy_config()`: 获取匹配特定层的模糊配置

```python
class QuantCfgField():
    def __init__(self, config):
        self.weights_cfg = WeightsCfgField(config.get('weights', {}))
        self.inputs_cfg = InputsCfg(Fields(config.get('inputs', {'enable_quant': False}))
        self.fuzzy_configs = self._extract_fuzzy_configs(config)
        self.value = self.set_value()
    
    def _extract_fuzzy_configs(self, config):
        """Extract fuzzy matching configs from quant_cfg"""
        
    def get_fuzzy_config(self, layer_name, config_type):
        """Get matching fuzzy config for a layer"""
```

### 3. `amct_pytorch/config/parser.py`

新增 `_get_layer_quant_config()` 函数：

```python
def _get_layer_quant_config(layer_name, quant_config):
    """
    Get quant config for a specific layer, considering fuzzy matching
    """
```

修改 `get_supported_layers()` 函数以使用模糊匹配配置。

### 4. `amct_pytorch/config/config.py`

新增 `HIFP8_MINMAX_FUZZY_CFG` 配置示例：

```python
HIFP8_MINMAX_FUZZY_CFG = {
    'batch_num': 1,
    'quant_cfg': {
        '*down_proj.weights': {
            'type': 'int8',
            'symmetric': True,
            'strategy': 'tensor',
        },
        'weights': {
            'type': 'int8',
            'symmetric': True,
            'strategy': 'channel',
        },
        '*self_attn.q_proj.inputs': {
            'type': 'int8',
            'symmetric': True,
            'strategy': 'channel',
        },
        'inputs': {
            'type': 'int8',
            'symmetric': True,
            'strategy': 'tensor',
        },
    },
    'algorithm': {'minmax'},
    'skip_layers': {'lm_head'}
}
```

### 5. `amct_pytorch/config/__init__.py`

将 `HIFP8_MINMAX_FUZZY_CFG` 添加到导出列表。

## 使用示例

### 基本用法

```python
from amct_pytorch import HIFP8_MINMAX_FUZZY_CFG, quantize

model = YourModel()
quantize(model, HIFP8_MINMAX_FUZZY_CFG)
```

### 自定义模糊配置

```python
custom_config = {
    'batch_num': 1,
    'quant_cfg': {
        # 匹配所有 self_attn 中的投影层权重
        '*self_attn.*_proj.weights': {
            'type': 'int8',
            'symmetric': True,
            'strategy': 'channel',
        },
        # 匹配所有 MLP 层权重
        '*mlp.*.weights': {
            'type': 'int8',
            'symmetric': True,
            'strategy': 'tensor',
        },
        # 默认配置
        'weights': {
            'type': 'int8',
            'symmetric': True,
            'strategy': 'channel',
        },
        'inputs': {
            'type': 'int8',
            'symmetric': True,
            'strategy': 'tensor',
        },
    },
    'algorithm': {'minmax'},
}

quantize(model, custom_config)
```

## 配置优先级

1. **模糊匹配配置**：如果层名称匹配模糊模式，使用模糊配置
2. **默认配置**：如果没有匹配的模糊配置，使用默认的 `weights` 或 `inputs` 配置

## 测试

### 单元测试

创建了以下测试文件：

1. `tests/amct_pytorch/test_fuzzy_pattern.py`: 测试 `match_fuzzy_pattern()` 函数
2. `tests/amct_pytorch/test_fuzzy_config.py`: 测试模糊配置提取和匹配
3. `tests/amct_pytorch/test_fuzzy_standalone.py`: 独立的模糊匹配测试

运行测试：

```bash
python3 tests/amct_pytorch/test_fuzzy_pattern.py
python3 tests/amct_pytorch/test_fuzzy_config.py
python3 tests/amct_pytorch/test_fuzzy_standalone.py
```

### 示例代码

创建了示例文件 `examples/fuzzy_config_example.py`，展示如何使用模糊匹配配置。

## 匹配规则

### 通配符语法

- `*`: 匹配任意数量的字符（包括零个字符）
- 支持多个通配符
- 支持通配符在任意位置

### 后缀处理

配置中的 `.weights` 和 `.inputs` 后缀会被特殊处理：

- 配置项 `*down_proj.weights` 会匹配：
  - `model.layers.0.mlp.down_proj`
  - `model.layers.0.mlp.down_proj.weights`

- 配置项 `*self_attn.q_proj.inputs` 会匹配：
  - `model.layers.0.self_attn.q_proj`
  - `model.layers.0.self_attn.q_proj.inputs`

### 匹配示例

| 层名称 | 模式 | 匹配结果 |
|---------|------|---------|
| `model.layers.0.down_proj` | `*down_proj` | ✓ |
| `model.layers.1.down_proj` | `*down_proj` | ✓ |
| `model.layers.0.up_proj` | `*down_proj` | ✗ |
| `model.layers.0.q_proj` | `*self_attn.q_proj.inputs` | ✓ |
| `model.layers.0.k_proj` | `*self_attn.q_proj.inputs` | ✗ |

## 注意事项

1. 模糊配置只在 `quant_cfg` 中支持，不支持在 `skip_layers` 中使用
2. 通配符匹配使用 `fnmatch` 模块，遵循 Unix shell 风格的通配符规则
3. 如果多个模糊模式匹配同一个层，使用第一个匹配的模式
4. 模糊配置必须以 `.weights` 或 `.inputs` 结尾才能被识别

## 向后兼容性

该实现完全向后兼容：
- 不使用通配符的配置仍然正常工作
- 现有的配置格式和行为保持不变
- 只在检测到 `*` 通配符时才启用模糊匹配功能
