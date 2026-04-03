#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, '/mnt/workspace/gitCode/fujun19/amct_open')

import fnmatch
import torch
import torch.nn as nn


def match_fuzzy_pattern(layer_name, pattern):
    """
    Match layer name with pattern supporting wildcard '*'
    """
    if '*' not in pattern:
        return layer_name == pattern
    
    suffix = None
    if pattern.endswith('.weights'):
        suffix = '.weights'
        pattern = pattern[:-8]
    elif pattern.endswith('.inputs'):
        suffix = '.inputs'
        pattern = pattern[:-7]
    
    if suffix and layer_name.endswith(suffix):
        layer_name = layer_name[:-len(suffix)]
    
    return fnmatch.fnmatch(layer_name, pattern)


def _extract_fuzzy_configs(config):
    """
    Extract fuzzy matching configs from quant_cfg
    """
    fuzzy_configs = {'weights': [], 'inputs': []}
    
    for key, value in config.items():
        if '*' in key:
            if key.endswith('.weights'):
                fuzzy_configs['weights'].append({
                    'pattern': key,
                    'config': value
                })
            elif key.endswith('.inputs'):
                fuzzy_configs['inputs'].append({
                    'pattern': key,
                    'config': value
                })
    
    return fuzzy_configs


def get_fuzzy_config(layer_name, fuzzy_configs, config_type):
    """
    Get matching fuzzy config for a layer
    """
    for fuzzy_cfg in fuzzy_configs[config_type]:
        if match_fuzzy_pattern(layer_name, fuzzy_cfg['pattern']):
            return fuzzy_cfg['config']
    return None


class TestModelBias(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(64, 64)
        self.linear2 = nn.Linear(64, 64)
        self.linear3 = nn.Linear(64, 64)


def test_fp8_fp4_group_sym_minmax_success_config():
    """
    测试类似 test_fp8_fp4_group_sym_minmax_success_config 的模糊配置
    """
    print("=" * 60)
    print("Testing FP8 FP4 Group Sym Minmax Fuzzy Config")
    print("=" * 60)
    
    cfg = {
        'batch_num': 1,
        'quant_cfg': {
            '*linear1.weights': {
                'type': 'int8',
                'symmetric': True,
                'strategy': 'tensor',
            },
            'weights': {
                'type': 'int8',
                'symmetric': True,
                'strategy': 'channel',
            },
            '*linear1.inputs': {
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
        'skip_layers': {'linear3'}
    }
    
    # 提取模糊配置
    fuzzy_configs = _extract_fuzzy_configs(cfg['quant_cfg'])
    
    print("\nExtracted fuzzy configs:")
    print(f"  Weights: {len(fuzzy_configs['weights'])} patterns")
    print(f"  Inputs: {len(fuzzy_configs['inputs'])} patterns")
    
    # 测试每个层的配置
    model = TestModelBias()
    test_layers = ['linear1', 'linear2', 'linear3']
    
    print("\nTesting layer configurations:")
    for layer_name in test_layers:
        # 跳过层
        if layer_name in cfg['skip_layers']:
            print(f"  {layer_name}: SKIPPED")
            continue
        
        # 获取权重配置
        weights_cfg = get_fuzzy_config(layer_name, fuzzy_configs, 'weights')
        if weights_cfg:
            weights_strategy = weights_cfg['strategy']
        else:
            weights_strategy = cfg['quant_cfg']['weights']['strategy']
        
        # 获取输入配置
        inputs_cfg = get_fuzzy_config(layer_name, fuzzy_configs, 'inputs')
        if inputs_cfg:
            inputs_strategy = inputs_cfg['strategy']
        else:
            inputs_strategy = cfg['quant_cfg']['inputs']['strategy']
        
        print(f"  {layer_name}:")
        print(f"    weights: {weights_strategy}")
        print(f"    inputs: {inputs_strategy}")
    
    # 验证结果
    print("\nVerifying results:")
    
    # linear1 应该使用模糊配置
    weights_cfg = get_fuzzy_config('linear1', fuzzy_configs, 'weights')
    inputs_cfg = get_fuzzy_config('linear1', fuzzy_configs, 'inputs')
    
    assert weights_cfg is not None, "linear1 should have fuzzy weights config"
    assert weights_cfg['strategy'] == 'tensor', "linear1 weights should use tensor strategy"
    assert inputs_cfg is not None, "linear1 should have fuzzy inputs config"
    assert inputs_cfg['strategy'] == 'channel', "linear1 inputs should use channel strategy"
    print("  ✓ linear1 uses fuzzy config (weights: tensor, inputs: channel)")
    
    # linear2 应该使用默认配置
    weights_cfg = get_fuzzy_config('linear2', fuzzy_configs, 'weights')
    inputs_cfg = get_fuzzy_config('linear2', fuzzy_configs, 'inputs')
    
    assert weights_cfg is None, "linear2 should not have fuzzy weights config"
    assert inputs_cfg is None, "linear2 should not have fuzzy inputs config"
    print("  ✓ linear2 uses default config (weights: channel, inputs: tensor)")
    
    # linear3 应该被跳过
    assert 'linear3' in cfg['skip_layers'], "linear3 should be in skip_layers"
    print("  ✓ linear3 is skipped")
    
    print("\n" + "=" * 60)
    print("All tests PASSED!")
    print("=" * 60)
    
    return True


if __name__ == '__main__':
    try:
        test_fp8_fp4_group_sym_minmax_success_config()
        sys.exit(0)
    except Exception as e:
        print(f"\nTest FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
