# 量化配置模糊匹配功能 - 异常场景测试设计文档

## 1. 测试概述

### 1.1 测试目标
验证量化配置模糊匹配功能在各种异常场景下的行为，确保功能的健壮性和正确性。

### 1.2 测试范围
- 无效的通配符模式
- 配置冲突和优先级
- 边界条件
- 特殊字符处理
- 与现有功能的交互

---

## 2. 异常场景分类

### 2.1 通配符模式异常

#### 2.1.1 无效的通配符模式

| 场景ID | 测试场景 | 输入示例 | 预期行为 | 优先级 |
|---------|---------|-----------|---------|--------|
| WC-001 | 空通配符模式 | `''` | 不匹配任何层 | P0 |
| WC-002 | 只有通配符 | `'*'` | 匹配所有层 | P0 |
| WC-003 | 连续多个通配符 | `'**'` | 按照fnmatch规则匹配 | P1 |
| WC-004 | 通配符在开头 | `'*layer'` | 匹配以layer结尾的层 | P0 |
| WC-005 | 通配符在中间 | `'layer*name'` | 匹配包含的层 | P0 |
| WC-006 | 通配符在结尾 | `'layer*'` | 匹配以layer开头的层 | P0 |
| WC-007 | 多个通配符组合 | `'*layer*name*'` | 匹配包含layer和name的层 | P1 |
| WC-008 | 特殊字符与通配符 | `'*layer@name'` | 正确处理特殊字符 | P2 |

#### 2.1.2 后缀处理异常

| 场景ID | 测试场景 | 输入示例 | 预期行为 | 优先级 |
|---------|---------|-----------|---------|--------|
| SF-001 | 只有后缀 | `'.weights'` | 不匹配任何层 | P1 |
| SF-002 | 通配符加后缀 | `'*.weights'` | 匹配所有层的权重配置 | P0 |
| SF-003 | 多个后缀 | `'*layer.weights.weights'` | 只处理最后一个后缀 | P1 |
| SF-004 | 后缀大小写 | `'*.Weights'` | 不匹配（大小写敏感） | P2 |
| SF-005 | 无效后缀 | `'*.invalid'` | 不识别为模糊配置 | P1 |

### 2.2 配置冲突和优先级

#### 2.2.1 多个模式匹配同一层

| 场景ID | 测试场景 | 配置示例 | 预期行为 | 优先级 |
|---------|---------|-----------|---------|--------|
| CF-001 | 两个权重模式匹配 | `'*layer.weights'` 和 `'*specific.weights'` | 使用第一个匹配的模式 | P0 |
| CF-002 | 权重和输入模式同时匹配 | `'*layer.weights'` 和 `'*layer.inputs'` | 分别应用各自的配置 | P0 |
| CF-003 | 模糊配置和默认配置 | `'*layer.weights'` 和 `'weights'` | 模糊配置优先 | P0 |
| CF-004 | 精确配置和模糊配置 | `'exact.weights'` 和 `'*exact.weights'` | 精确配置优先 | P1 |

#### 2.2.2 配置优先级顺序

优先级从高到低：
1. 精确匹配的层配置（如 `'layer1.weights'`）
2. 模糊匹配的层配置（如 `'*layer.weights'`）
3. 默认配置（`'weights'` 或 `'inputs'`）

### 2.3 边界条件

#### 2.3.1 空配置和None值

| 场景ID | 测试场景 | 输入示例 | 预期行为 | 优先级 |
|---------|---------|-----------|---------|--------|
| BC-001 | quant_cfg为空字典 | `{}` | 使用默认配置 | P0 |
| BC-002 | quant_cfg为None | `None` | 使用默认配置 | P0 |
| BC-003 | 模糊配置值为空 | `'*layer.weights': {}` | 跳过该配置 | P1 |
| BC-004 | 配置值为None | `'*layer.weights': None` | 跳过该配置 | P1 |

#### 2.3.2 层名称边界

| 场景ID | 测试场景 | 层名称示例 | 预期行为 | 优先级 |
|---------|---------|-----------|---------|--------|
| LN-001 | 空层名称 | `''` | 不匹配任何模式 | P1 |
| LN-002 | 非常长的层名称 | `'a.' * 100 + 'layer'` | 正确处理 | P2 |
| LN-003 | 层名称包含数字 | `'layer0', 'layer1'` | 正确匹配 | P0 |
| LN-004 | 层名称包含下划线 | `'layer_name'` | 正确匹配 | P0 |
| LN-005 | 层名称包含点号 | `'model.layer.name'` | 正确匹配 | P0 |

### 2.4 特殊字符处理

| 场景ID | 测试场景 | 字符示例 | 预期行为 | 优先级 |
|---------|---------|-----------|---------|--------|
| SC-001 | 空格字符 | `'layer name'` | 正确处理 | P1 |
| SC-002 | 中文字符 | `'层名称'` | 正确处理 | P2 |
| SC-003 | 特殊符号 | `'layer@#$'` | 正确处理 | P2 |
| SC-004 | Unicode字符 | `'layerαβγ'` | 正确处理 | P2 |
| SC-005 | 转义字符 | `'layer\nname'` | 正确处理 | P2 |

### 2.5 与现有功能的交互

#### 2.5.1 与skip_layers的交互

| 场景ID | 测试场景 | 配置示例 | 预期行为 | 优先级 |
|---------|---------|-----------|---------|--------|
| SL-001 | 模糊匹配层被跳过 | 模糊匹配层在skip_layers中 | 跳过量化 | P0 |
| SL-002 | 部分匹配层被跳过 | 模糊匹配部分层在skip_layers中 | 只量化未跳过的层 | P0 |
| SL-003 | skip_layers包含通配符 | `skip_layers={'*layer'}` | 按现有逻辑处理 | P1 |

#### 2.5.2 与算法验证的交互

| 场景ID | 测试场景 | 配置示例 | 预期行为 | 优先级 |
|---------|---------|-----------|---------|--------|
| AL-001 | 模糊配置不支持算法 | 模糊配置使用不支持的量化类型 | 抛出异常 | P0 |
| AL-002 | 模糊配置参数无效 | 模糊配置包含无效参数 | 抛出异常 | P0 |
| AL-003 | 混合配置算法冲突 | 不同层配置不同算法 | 允许（每层独立配置） | P1 |

#### 2.5.3 与层约束检查的交互

| 场景ID | 测试场景 | 配置示例 | 预期行为 | 优先级 |
|---------|---------|-----------|---------|--------|
| LC-001 | 模糊匹配层不满足约束 | 模糊匹配层形状不满足要求 | 跳过该层 | P0 |
| LC-002 | 部分匹配层不满足约束 | 模糊匹配部分层不满足约束 | 只量化满足约束的层 | P0 |

---

## 3. 测试用例设计

### 3.1 单元测试用例

#### 测试类: TestFuzzyPatternExceptions

```python
def test_empty_pattern(self):
    """测试空模式"""
    result = match_fuzzy_pattern('layer1', '')
    self.assertFalse(result)

def test_wildcard_only(self):
    """测试只有通配符"""
    result = match_fuzzy_pattern('any.layer.name', '*')
    self.assertTrue(result)

def test_multiple_wildcards(self):
    """测试多个连续通配符"""
    result = match_fuzzy_pattern('layer', '**')
    self.assertTrue(result)

def test_special_characters(self):
    """测试特殊字符"""
    result = match_fuzzy_pattern('layer@name', '*@name')
    self.assertTrue(result)

def test_suffix_case_sensitive(self):
    """测试后缀大小写敏感"""
    result = match_fuzzy_pattern('layer', '*.Weights')
    self.assertFalse(result)

def test_invalid_suffix(self):
    """测试无效后缀"""
    result = match_fuzzy_pattern('layer', '*.invalid')
    self.assertFalse(result)
```

#### 测试类: TestFuzzyConfigConflicts

```python
def test_multiple_patterns_same_layer(self):
    """测试多个模式匹配同一层"""
    config = {
        '*layer.weights': {'type': 'int8', 'strategy': 'tensor'},
        '*specific.weights': {'type': 'int4', 'strategy': 'channel'},
        'weights': {'type': 'int8', 'strategy': 'channel'},
    }
    # 验证使用第一个匹配的模式

def test_fuzzy_vs_default_priority(self):
    """测试模糊配置优先级高于默认配置"""
    config = {
        '*down_proj.weights': {'type': 'int8', 'strategy': 'tensor'},
        'weights': {'type': 'int8', 'strategy': 'channel'},
    }
    # 验证down_proj使用tensor策略

def test_exact_vs_fuzzy_priority(self):
    """测试精确配置优先级高于模糊配置"""
    config = {
        'exact.weights': {'type': 'int4', 'strategy': 'channel'},
        '*exact.weights': {'type': 'int8', 'strategy': 'tensor'},
    }
    # 验证exact使用int4
```

#### 测试类: TestFuzzyConfigBoundaryConditions

```python
def test_empty_quant_cfg(self):
    """测试空的quant_cfg"""
    config = {}
    # 验证使用默认配置

def test_none_quant_cfg(self):
    """测试None的quant_cfg"""
    config = None
    # 验证使用默认配置

def test_empty_fuzzy_config_value(self):
    """测试空的模糊配置值"""
    config = {
        '*layer.weights': {},
        'weights': {'type': 'int8', 'strategy': 'channel'},
    }
    # 验证跳过空配置

def test_long_layer_name(self):
    """测试非常长的层名称"""
    long_name = 'a.' * 100 + 'layer'
    pattern = '*layer'
    result = match_fuzzy_pattern(long_name, pattern)
    self.assertTrue(result)
```

#### 测试类: TestFuzzyConfigSpecialCharacters

```python
def test_layer_name_with_spaces(self):
    """测试包含空格的层名称"""
    result = match_fuzzy_pattern('layer name', '*name')
    self.assertTrue(result)

def test_layer_name_with_chinese(self):
    """测试包含中文的层名称"""
    result = match_fuzzy_pattern('层名称', '*名称')
    self.assertTrue(result)

def test_layer_name_with_unicode(self):
    """测试包含Unicode的层名称"""
    result = match_fuzzy_pattern('layerαβγ', '*αβγ')
    self.assertTrue(result)
```

#### 测试类: TestFuzzyConfigWithSkipLayers

```python
def test_fuzzy_layer_in_skip_layers(self):
    """测试模糊匹配的层在skip_layers中"""
    model = TestModel()
    config = {
        '*layer.weights': {'type': 'int8', 'strategy': 'tensor'},
        'weights': {'type': 'int8', 'strategy': 'channel'},
    }
    skip_layers = {'specific.layer'}
    # 验证specific.layer被跳过

def test_partial_fuzzy_layers_in_skip_layers(self):
    """测试部分模糊匹配的层在skip_layers中"""
    # 验证只有未跳过的层被量化
```

#### 测试类: TestFuzzyConfigWithAlgorithmValidation

```python
def test_fuzzy_config_unsupported_quant_type(self):
    """测试模糊配置使用不支持的量化类型"""
    config = {
        '*layer.weights': {'type': 'invalid_type', 'strategy': 'channel'},
    }
    # 验证抛出异常

def test_fuzzy_config_invalid_parameters(self):
    """测试模糊配置包含无效参数"""
    config = {
        '*layer.weights': {
            'type': 'int8',
            'strategy': 'invalid_strategy',
        },
    }
    # 验证抛出异常
```

### 3.2 集成测试用例

#### 测试类: TestFuzzyConfigIntegration

```python
def test_fuzzy_config_with_real_model(self):
    """测试模糊配置与真实模型的集成"""
    model = TransformerModel(num_layers=4, hidden_size=64)
    config = HIFP8_MINMAX_FUZZY_CFG
    detail_config = parse_config(model, config, AlgorithmRegistry)
    # 验证所有层的配置正确

def test_fuzzy_config_mixed_with_exact_config(self):
    """测试模糊配置与精确配置混合使用"""
    config = {
        '*layer.weights': {'type': 'int8', 'strategy': 'tensor'},
        'exact.layer.weights': {'type': 'int4', 'strategy': 'channel'},
        'weights': {'type': 'int8', 'strategy': 'channel'},
    }
    # 验证配置优先级正确

def test_fuzzy_config_with_complex_patterns(self):
    """测试复杂的模糊配置模式"""
    config = {
        '*self_attn.*_proj.weights': {'type': 'int8', 'strategy': 'channel'},
        '*mlp.*.weights': {'type': 'int8', 'strategy': 'tensor'},
        'weights': {'type': 'int8', 'strategy': 'channel'},
    }
    # 验证复杂模式正确匹配
```

---

## 4. 性能测试

### 4.1 性能指标

| 指标 | 目标值 | 测试方法 |
|-------|--------|---------|
| 单次匹配时间 | < 1ms | 测试单次match_fuzzy_pattern调用 |
| 1000层配置解析时间 | < 100ms | 测试parse_config处理1000层 |
| 内存占用增量 | < 10MB | 测试模糊配置的内存占用 |

### 4.2 性能测试用例

```python
def test_fuzzy_pattern_performance(self):
    """测试模糊匹配性能"""
    import time
    
    pattern = '*layer.weights'
    layer_names = [f'model.layers.{i}.down_proj' for i in range(1000)]
    
    start_time = time.time()
    for layer_name in layer_names:
        match_fuzzy_pattern(layer_name, pattern)
    elapsed_time = time.time() - start_time
    
    self.assertLess(elapsed_time, 0.1)  # 100ms

def test_large_model_parse_performance(self):
    """测试大模型配置解析性能"""
    import time
    
    model = LargeModel(num_layers=1000)
    config = HIFP8_MINMAX_FUZZY_CFG
    
    start_time = time.time()
    detail_config = parse_config(model, config, AlgorithmRegistry)
    elapsed_time = time.time() - start_time
    
    self.assertLess(elapsed_time, 1.0)  # 1s
```

---

## 5. 测试执行计划

### 5.1 测试优先级

| 优先级 | 测试场景数量 | 执行顺序 |
|--------|-------------|---------|
| P0 | 15 | 第一批执行 |
| P1 | 10 | 第二批执行 |
| P2 | 8 | 第三批执行 |

### 5.2 测试环境

- Python版本: 3.11
- PyTorch版本: 2.7.1
- 测试框架: unittest
- 覆盖率目标: > 90%

### 5.3 测试执行命令

```bash
# 执行所有异常场景测试
python3 -m unittest tests.amct_pytorch.test_fuzzy_exceptions -v

# 执行特定优先级测试
python3 -m unittest tests.amct_pytorch.test_fuzzy_exceptions.TestFuzzyPatternExceptions -v

# 执行性能测试
python3 -m unittest tests.amct_pytorch.test_fuzzy_exceptions.TestFuzzyConfigPerformance -v

# 生成覆盖率报告
python3 -m coverage run -m unittest tests.amct_pytorch.test_fuzzy_exceptions
python3 -m coverage report -m amct_pytorch/config/utils.py amct_pytorch/config/fields.py amct_pytorch/config/parser.py
```

---

## 6. 预期结果

### 6.1 功能测试结果

- 所有P0测试用例通过率: 100%
- 所有P1测试用例通过率: 100%
- 所有P2测试用例通过率: 100%

### 6.2 性能测试结果

- 单次匹配时间: < 1ms ✓
- 1000层配置解析时间: < 100ms ✓
- 内存占用增量: < 10MB ✓

### 6.3 代码覆盖率

- `amct_pytorch/config/utils.py`: > 95%
- `amct_pytorch/config/fields.py`: > 90%
- `amct_pytorch/config/parser.py`: > 90%

---

## 7. 已知限制和注意事项

### 7.1 功能限制

1. **skip_layers不支持通配符**: 当前skip_layers不支持通配符匹配，只支持精确匹配和子字符串匹配
2. **配置优先级固定**: 无法自定义配置优先级顺序
3. **大小写敏感**: 匹配是大小写敏感的

### 7.2 性能注意事项

1. **复杂模式影响性能**: 使用多个通配符的复杂模式会降低匹配速度
2. **大量模糊配置**: 配置中包含大量模糊模式会增加解析时间

### 7.3 使用建议

1. 避免使用过于宽泛的通配符模式（如`*`）
2. 优先使用精确配置，仅在必要时使用模糊配置
3. 对于大型模型，建议使用更具体的模式以提高性能

---

## 8. 附录

### 8.1 测试数据示例

```python
# 测试用模型
class TestFuzzyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'self_attn': nn.ModuleDict({
                    'q_proj': nn.Linear(64, 64),
                    'k_proj': nn.Linear(64, 64),
                    'v_proj': nn.Linear(64, 64),
                }),
                'mlp': nn.ModuleDict({
                    'up_proj': nn.Linear(64, 64),
                    'down_proj': nn.Linear(64, 64),
                })
            })
            for _ in range(4)
        ])
        self.lm_head = nn.Linear(64, 64)

# 异常配置示例
EXCEPTION_CONFIGS = {
    'empty_pattern': {
        '': {'type': 'int8', 'strategy': 'channel'},
    },
    'wildcard_only': {
        '*': {'type': 'int8', 'strategy': 'channel'},
    },
    'multiple_wildcards': {
        '**': {'type': 'int8', 'strategy': 'channel'},
    },
    'invalid_suffix': {
        '*.invalid': {'type': 'int8', 'strategy': 'channel'},
    },
}
```

### 8.2 测试工具函数

```python
def create_test_model(num_layers=4, hidden_size=64):
    """创建测试模型"""
    pass

def generate_test_configs():
    """生成测试配置"""
    pass

def verify_config_priority(detail_config, expected_config):
    """验证配置优先级"""
    pass

def measure_performance(func, *args, **kwargs):
    """测量函数执行时间"""
    import time
    start = time.time()
    result = func(*args, **kwargs)
    elapsed = time.time() - start
    return result, elapsed
```

---

## 9. 文档版本历史

| 版本 | 日期 | 作者 | 修改说明 |
|------|------|------|---------|
| 1.0 | 2026-03-31 | opencode | 初始版本 |

---

## 10. 参考文档

- [fuzzy_matching_implementation.md](./fuzzy_matching_implementation.md) - 模糊匹配功能实现文档
- [amct_pytorch/config/utils.py](../amct_pytorch/config/utils.py) - 工具函数实现
- [amct_pytorch/config/fields.py](../amct_pytorch/config/fields.py) - 配置字段实现
- [amct_pytorch/config/parser.py](../amct_pytorch/config/parser.py) - 配置解析实现
