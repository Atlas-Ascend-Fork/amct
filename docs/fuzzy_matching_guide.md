# 量化配置模糊匹配功能使用指南

## 功能概述

AMCT 现在支持量化配置的模糊匹配功能，允许使用通配符 `*` 来匹配多个层，从而为这些层应用相同的量化配置。

## 模糊匹配规则

- 模糊匹配模式以 `*` 开头
- `*` 表示匹配任意前缀，后面的字符串必须是层名的后缀
- 例如：`*self.atten.q_proj` 匹配所有以 `self.atten.q_proj` 结尾的层名

## 使用示例

### 1. 权重量化配置模糊匹配

```python
import torch
import torch.nn as nn
from amct_pytorch import quantize, convert

class TransformerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(768, 768),
            nn.Linear(768, 3072),
            nn.Linear(3072, 768)
        ])
        self.down_proj = nn.Linear(768, 768)
        self.gate_proj = nn.Linear(768, 768)
        self.up_proj = nn.Linear(768, 768)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

model = TransformerModel()

# 使用模糊匹配配置
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
        # 所有以 gate_proj 结尾的层使用 int4 权重量化
        '*gate_proj': {
            'quant_cfg': {
                'weights': {
                    'type': 'int4',
                    'symmetric': True,
                    'strategy': 'channel',
                }
            }
        }
    }
}

quantize(model, config)
```

### 2. 激活量化配置模糊匹配

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
            'enable_quant': False  # 默认不进行激活量化
        },
    },
    'algorithm': {'minmax'},
    'override_layer_configs': {
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
        },
        # 所有以 k_proj 结尾的层启用激活量化
        '*k_proj': {
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

### 3. KV Cache 量化配置模糊匹配

```python
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
        # 所有以 self_attn 结尾的层应用 KV Cache 量化配置
        '*self_attn': {
            'quant_cfg': {
                'kvcache': {
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

### 4. 混合使用多个模糊匹配模式

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
        # 权重量化模糊匹配
        '*down_proj': {
            'quant_cfg': {
                'weights': {
                    'type': 'int4',
                    'symmetric': True,
                    'strategy': 'channel',
                }
            }
        },
        # 激活量化量化模糊匹配
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
        'k_proj': {
            'quant_cfg': {
                'inputs': {
                    'type': 'int8',
                    'symmetric': True,
                    'strategy': 'tensor',
                    'enable_quant': True
                }
            }
        },
        # KV Cache 量化模糊匹配
        '*self_attn': {
            'quant_cfg': {
                'kvcache': {
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

## 匹配优先级

1. 精确匹配优先于模糊匹配
2. 多个模糊匹配模式时，使用第一个匹配的模式
3. override_layer_configs 中的配置优先于基础配置

## 注意事项

1. 模糊匹配模式必须以 `*` 开头
2. 匹配是基于层名的后缀匹配
3. 一个层只能匹配一个模糊模式（第一个匹配的模式）
4. 建议在日志中查看实际匹配情况，确认配置应用正确

## 日志示例

当模糊匹配成功时，会在日志中看到类似输出：

```
[INFO] [AMCT]: [Parser]: Layer model.layers.0.down_proj matches fuzzy pattern *down_proj, applying override config
[INFO] [AMCT]: [Parser]: Layer transformer.layers.1.self_attn.q_proj matches fuzzy pattern *q_proj, applying override config
```

## 常见使用场景

### Transformer 模型量化

```python
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

这样可以大大简化配置，避免为每个层单独指定配置。
