# 量化配置模糊匹配功能实现总结

## 功能概述

已成功为 AMCT 实现量化配置的模糊匹配功能，支持使用通配符 `*` 来匹配多个层，从而为这些层应用相同的量化配置。

## 实现的功能

### 1. 模糊匹配工具函数 (amct_pytorch/config/utils.py)

添加了以下工具函数：

- `is_fuzzy_pattern(pattern)`: 检查模式是否为模糊匹配模式（以 `*` 开头）
- `get_fuzzy_suffix(pattern)`: 从模糊模式中提取后缀
- `match_fuzzy_pattern(layer_name, pattern)`: 检查层名是否匹配模糊模式
- `find_matching_pattern(layer_name, patterns)`: 查找第一个匹配的模式

### 2. 配置字段扩展 (amct_pytorch/config/fields.py)

新增 `OverrideLayerConfigsField` 类：
- 支持解析 `override_layer_configs` 配置字段
- 验证模糊匹配模式和配置的有效性

### 3. 配置解析器增强 (amct_pytorch/config/parser.py)

增强功能：
- `merge_layer_config()`: 合并基础配置和覆盖配置
- `get_supported_layers()`: 应用模糊匹配逻辑，为匹配的层应用覆盖配置

### 4. 测试用例 (tests/amct_pytorch/test_fuzzy_matching_utils.py)

完整的测试覆盖：
- 模糊匹配工具函数测试
- 模式匹配测试
- 多模式匹配测试

### 5. 使用文档 (docs/fuzzy_matching_guide.md)

详细的使用指南，包含：
- 功能概述和规则
- 多种使用示例（权重、激活、KV Cache 量化）
- Transformer 模型实际应用场景
- 注意事项和最佳实践

### 6. 演示程序 (examples/fuzzy_matching_demo.py)

交互式演示程序，展示：
- 模糊匹配工具函数使用
- 模式匹配测试
- 实际 Transformer 模型层名匹配示例

## 使用示例

### 基本用法

```python
from amct_pytorch import quantize

config = {
    'batch_num': 1,
    'quant_cfg': {
        'weights': {
            'type': 'int8',
            'symmetric': True,
            'strategy': 'channel',
        },
    },
    'algorithm': {'minmax'},
    'override_layer_configs': {
        # 所有以 down_proj 结尾的层使用 int4 权重量化
        '*down_proj': {
            'quant_cfg': {
                'weights': {
                    'type': 'int4',
                    'symmetric': True,
                    'strategy': 'channel',
                }
            }
        },
        # 所有以 q_proj 结尾的层启用激活量化
        '*q_proj': {
            'quant_cfg': {
                'inputs': {
                    'type': 'int8',
                    'symmetric': True,
                    'strategy': 'tensor',
                    'enable_quant': True
                }
            }
        }
    }
}

quantize(model, config)
```

### Transformer 模型完整配置

```python
config = {
    'batch_num': 1,
    'quant_cfg': {
        'weights': {
            'type': 'int8',
            'symmetric': True,
            'strategy': 'channel',
        },
        'inputs': {
            'type': 'int8',
            'symmetric': True,
            'strategy': 'tensor',
            'enable_quant': False,
        },
    },
    'algorithm': {'minmax'},
    'override_layer_configs': {
        # FFN 层使用 int4 权重量化
        '*down_proj': {
            'quant_cfg': {
                'weights': {
                    'type': 'int4',
                    'symmetric': True,
                    'strategy': 'channel',
                }
            }
        },
        '*gate_proj': {
            'quant_cfg': {
                'weights': {
                    'type': 'int4',
                    'symmetric': True,
                    'strategy': 'channel',
                }
            }
        },
        '*up_proj': {
            'quant_cfg': {
                'weights': {
                    'type': 'int4',
                    'symmetric': True,
                    'strategy': 'channel',
                }
            }
        },
        # Attention 层启用激活量化
        '*q_proj': {
            'quant_cfg': {
                'inputs': {
                    'type': 'int8',
                    'symmetric': True,
                    'strategy': 'tensor',
                    'enable_quant': True
                }
            }
        },
        '*k_proj': {
            'quant_cfg': {
                'inputs': {
                    'type': 'int8',
                    'symmetric': True,
                    'strategy': 'tensor',
                    'enable_quant': True
                }
            }
        },
        '*v_proj': {
            'quant_cfg': {
                'inputs': {
                    'type': 'int8',
                    'symmetric': True,
                    'strategy': 'tensor',
                    'enable_quant': True
                }
            }
        }
    }
}
```

## 技术特性

### 1. 模糊匹配规则
- 模糊匹配模式必须以 `*` 开头
- `*` 表示匹配任意前缀，后面的字符串必须是层名的后缀
- 例如：`*self.atten.q_proj` 匹配 `model.layers.0.self.atten.q_proj`

### 2. 匹配优先级
1. 精确匹配优先于模糊匹配
2. 多个模糊匹配模式时，使用第一个匹配的模式
3. `override_layer_configs` 中的配置优先于基础配置

### 3. 配置合并策略
- 覆盖配置会替换基础配置中的相应字段
- 支持的覆盖字段：
  - `batch_num`: 批次大小
  - `quant_cfg.weights`: 权重量化配置
  - `quant_cfg.inputs`: 激活量化配置
  - `algorithm`: 算法配置

## 测试验证

### 运行单元测试

```bash
python3 -m unittest tests.amct_pytorch.test_fuzzy_matching_utils.TestFuzzyMatchingUtils -v
```

测试结果：
```
test_find_matching_pattern ... ok
test_get_fuzzy_suffix ... ok
test_is_fuzzy_pattern ... ok
test_match_fuzzy_pattern ... ok

----------------------------------------------------------------------
Ran 4 tests in 0.001s

OK
```

### 运行演示程序

```bash
python3 examples/fuzzy_matching_demo.py
```

演示程序展示了：
- 模糊匹配工具函数的使用
- 各种模式匹配场景
- 实际 Transformer 模型层名匹配示例

## 修改的文件

1. **amct_pytorch/config/utils.py**
   - 新增模糊匹配工具函数

2. **amct_pytorch/config/fields.py**
   - 新增 `OverrideLayerConfigsField` 类
   - 修改 `QuantConfig` 类支持 override_layer_configs

3. **amct_pytorch/config/parser.py**
   - 新增 `merge_layer_config()` 函数
   - 修改 `get_supported_layers()` 函数应用模糊匹配

4. **tests/amct_pytorch/test_fuzzy_matching_utils.py**
   - 新增模糊匹配工具函数测试用例

5. **docs/fuzzy_matching_guide.md**
   - 新增详细使用指南

6. **examples/fuzzy_matching_demo.py**
   - 新增演示程序

## 优势

1. **简化配置**: 无需为每个层单独指定配置
2. **提高可维护性**: 修改模式即可批量更新配置
3. **减少错误**: 避免重复配置导致的不一致
4. **灵活性**: 支持精确匹配和模糊匹配混合使用
5. **向后兼容**: 不影响现有的精确匹配配置

## 注意事项

1. 模糊匹配模式必须以 `*` 开头
2. 匹配是基于层名的后缀匹配
3. 一个层只能匹配一个模糊模式（第一个匹配的模式）
4. 建议在日志中查看实际匹配情况，确认配置应用正确
5. 模糊匹配模式之间没有优先级，按照配置文件中的顺序匹配

## 日志输出

当模糊匹配成功时，会在日志中看到类似输出：

```
[INFO] [AMCT]: [Parser]: Layer model.layers.0.down_proj matches fuzzy pattern *down_proj, applying override config
[INFO] [AMCT]: [Parser]: Layer transformer.layers.1.self_attn.q_proj matches fuzzy pattern *q_proj, applying override config
```

## 后续优化建议

1. 可以考虑支持更复杂的匹配模式（如正则表达式）
2. 可以添加模式优先级配置
3. 可以提供配置验证工具，检查模式是否能匹配预期的层
4. 可以添加模式冲突检测，避免多个模式匹配同一层

## 总结

成功实现了量化配置的模糊匹配功能，大幅简化了大型模型（如 Transformer）的量化配置过程。该功能完全向后兼容，不会影响现有配置，同时提供了更灵活、更易维护的配置方式。
