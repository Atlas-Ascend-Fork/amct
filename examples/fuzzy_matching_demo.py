# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def is_fuzzy_pattern(pattern):
    """
    Check if a pattern is a fuzzy matching pattern (starts with *)
    """
    return isinstance(pattern, str) and pattern.startswith('*')


def get_fuzzy_suffix(pattern):
    """
    Extract suffix from fuzzy pattern
    Example: "*self.atten.q_proj" -> "self.atten.q_proj"
    """
    if not is_fuzzy_pattern(pattern):
        return None
    return pattern[1:]


def match_fuzzy_pattern(layer_name, pattern):
    """
    Check if layer_name matches fuzzy pattern
    Example: "model.layers.0.self.atten.q_proj" matches "*self.atten.q_proj"
    """
    if not is_fuzzy_pattern(pattern):
        return layer_name == pattern
    
    suffix = get_fuzzy_suffix(pattern)
    return layer_name.endswith(suffix)


def find_matching_pattern(layer_name, patterns):
    """
    Find first matching pattern for a given layer name
    Returns:
        pattern that matches, or None if no match found
    """
    for pattern in patterns:
        if match_fuzzy_pattern(layer_name, pattern):
            return pattern
    return None

print("=" * 60)
print("AMCT 量化配置模糊匹配功能演示")
print("=" * 60)

print("\n1. 模糊匹配工具函数演示")
print("-" * 60)

test_patterns = [
    '*self.atten.q_proj',
    '*down_proj.weight',
    '*self_attn.kvcache',
    'exact.match'
]

for pattern in test_patterns:
    is_fuzzy = is_fuzzy_pattern(pattern)
    suffix = get_fuzzy_suffix(pattern) if is_fuzzy else "N/A"
    print(f"模式: {pattern:30s} | 是否模糊匹配: {is_fuzzy} | 后缀: {suffix}")

print("\n2. 模糊匹配测试")
print("-" * 60)

test_cases = [
    ('model.layers.0.self.atten.q_proj', '*self.atten.q_proj'),
    ('transformer.layers.1.down_proj.weight', '*down_proj.weight'),
    ('encoder.self_attn.kvcache', '*self_attn.kvcache'),
    ('model.layers.0.self.atten.q_proj', '*down_proj.weight'),
    ('exact.match', 'exact.match')
]

for layer_name, pattern in test_cases:
    matches = match_fuzzy_pattern(layer_name, pattern)
    status = "✓ 匹配" if matches else "✗ 不匹配"
    print(f"{status} | 层名: {layer_name:40s} | 模式: {pattern}")

print("\n3. 多模式匹配测试")
print("-" * 60)

patterns = ['*self.atten.q_proj', '*down_proj.weight', '*self_attn.kvcache']
test_layers = [
    'model.layers.0.self.atten.q_proj',
    'transformer.layers.1.down_proj.weight',
    'encoder.self_attn.kvcache',
    'no.match.layer',
    'exact.match'
]

for layer_name in test_layers:
    matched_pattern = find_matching_pattern(layer_name, patterns)
    if matched_pattern:
        print(f"✓ 层名: {layer_name:40s} 匹配模式: {matched_pattern}")
    else:
        print(f"✗ 层名: {layer_name:40s} 无匹配模式")

print("\n4. 实际 Transformer 模型层名匹配示例")
print("-" * 60)

transformer_layer_names = [
    'transformer.h.0.attn.c_attn.weight',
    'transformer.h.0.attn.q_proj.weight',
    'transformer.h.0.attn.k_proj.weight',
    'transformer.h.0.attn.v_proj.weight',
    'transformer.h.0.mlp.c_fc.weight',
    'transformer.h.0.mlp.c_proj.weight',
    'transformer.h.1.attn.c_attn.weight',
    'transformer.h.1.mlp.c_fc.weight',
    'transformer.ln_f.weight'
]

common_patterns = [
    ('*q_proj.weight', 'Query投影层'),
    ('*k_proj.weight', 'Key投影层'),
    ('*v_proj.weight', 'Value投影层'),
    ('*c_proj.weight', 'MLP输出投影层'),
    ('*c_fc.weight', 'MLP前馈层')
]

print("\n层名匹配结果:")
for layer_name in transformer_layer_names:
    for pattern, description in common_patterns:
        if match_fuzzy_pattern(layer_name, pattern):
            print(f"  {layer_name:50s} -> {description} (模式: {pattern})")
            break
    else:
        print(f"  {layer_name:50s} -> 未匹配特定模式")

print("\n" + "=" * 60)
print("模糊匹配功能演示完成！")
print("=" * 60)
print("\n使用建议:")
print("1. 在配置中使用 'override_layer_configs' 字段")
print("2. 模糊匹配模式以 '*' 开头，后面跟后缀字符串")
print("3. 一个层只会匹配第一个符合的模式")
print("4. 精确匹配优先于模糊匹配")
print("5. 查看日志确认配置应用情况")
