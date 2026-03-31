#!/usr/bin/env python3
import sys
import os

sys.path.insert(0, '/mnt/workspace/gitCode/fujun19/amct_open')

import fnmatch


def match_fuzzy_pattern(layer_name, pattern):
    """
    Match layer name with pattern supporting wildcard '*'
    Params:
        layer_name: str, full layer name (e.g., 'model.layers.0.self_attn.q_proj')
        pattern: str, pattern to match (e.g., '*self_attn.q_proj.weights')
    Return: bool, True if pattern matches layer_name
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
    Patterns like '*down_proj.weights' or '*self_attn.q_proj.inputs'
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
    Params:
        layer_name: str, layer name
        fuzzy_configs: dict, fuzzy configs for weights and inputs
        config_type: str, 'weights' or 'inputs'
    Return: dict or None, matching config or None
    """
    for fuzzy_cfg in fuzzy_configs[config_type]:
        if match_fuzzy_pattern(layer_name, fuzzy_cfg['pattern']):
            return fuzzy_cfg['config']
    return None


def test_hifp8_fuzzy_cfg():
    print("Testing HIFP8_MINMAX_FUZZY_CFG...")
    
    config = {
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
    }
    
    fuzzy_configs = _extract_fuzzy_configs(config)
    
    print(f"  Extracted fuzzy configs:")
    print(f"    Weights: {len(fuzzy_configs['weights'])} patterns")
    print(f"    Inputs: {len(fuzzy_configs['inputs'])} patterns")
    
    test_layers = [
        'model.layers.0.self_attn.q_proj',
        'model.layers.0.self_attn.k_proj',
        'model.layers.0.mlp.down_proj',
        'model.layers.0.mlp.up_proj',
        'model.layers.1.self_attn.q_proj',
        'model.layers.1.mlp.down_proj',
    ]
    
    print("\n  Testing layer config matching:")
    for layer_name in test_layers:
        weights_cfg = get_fuzzy_config(layer_name, fuzzy_configs, 'weights')
        inputs_cfg = get_fuzzy_config(layer_name, fuzzy_configs, 'inputs')
        
        weights_strategy = weights_cfg['strategy'] if weights_cfg else 'default (channel)'
        inputs_strategy = inputs_cfg['strategy'] if inputs_cfg else 'default (tensor)'
        
        print(f"    {layer_name}:")
        print(f"      weights: {weights_strategy}")
        print(f"      inputs: {inputs_strategy}")
    
    print("\n  Verifying expected results:")
    
    expected_results = [
        ('model.layers.0.self_attn.q_proj', 'default (channel)', 'channel'),
        ('model.layers.0.self_attn.k_proj', 'default (channel)', 'default (tensor)'),
        ('model.layers.0.mlp.down_proj', 'tensor', 'default (tensor)'),
        ('model.layers.0.mlp.up_proj', 'default (channel)', 'default (tensor)'),
        ('model.layers.1.self_attn.q_proj', 'default (channel)', 'channel'),
        ('model.layers.1.mlp.down_proj', 'tensor', 'default (tensor)'),
    ]
    
    passed = 0
    failed = 0
    for layer_name, expected_weights, expected_inputs in expected_results:
        weights_cfg = get_fuzzy_config(layer_name, fuzzy_configs, 'weights')
        inputs_cfg = get_fuzzy_config(layer_name, fuzzy_configs, 'inputs')
        
        actual_weights = weights_cfg['strategy'] if weights_cfg else 'default (channel)'
        actual_inputs = inputs_cfg['strategy'] if inputs_cfg else 'default (tensor)'
        
        if actual_weights == expected_weights and actual_inputs == expected_inputs:
            passed += 1
        else:
            failed += 1
            print(f"    FAILED: {layer_name}")
            print(f"      Expected: weights={expected_weights}, inputs={expected_inputs}")
            print(f"      Actual:   weights={actual_weights}, inputs={actual_inputs}")
    
    print(f"\n  Passed: {passed}/{len(expected_results)}")
    print(f"  Failed: {failed}/{len(expected_results)}")
    
    return failed == 0


if __name__ == '__main__':
    print("=" * 60)
    print("HIFP8_MINMAX_FUZZY_CFG Test")
    print("=" * 60)
    
    result = test_hifp8_fuzzy_cfg()
    
    print("\n" + "=" * 60)
    if result:
        print("All tests PASSED!")
        sys.exit(0)
    else:
        print("Some tests FAILED!")
        sys.exit(1)
