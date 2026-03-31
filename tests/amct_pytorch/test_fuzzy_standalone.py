#!/usr/bin/env python3
import sys
sys.path.insert(0, '/mnt/workspace/gitCode/fujun19/amct_open')

import fnmatch


def match_fuzzy_pattern(layer_name, pattern):
    """
    Match layer name with pattern supporting wildcard '*'
    Params:
        layer_name: str, the full layer name (e.g., 'model.layers.0.self_attn.q_proj')
        pattern: str, the pattern to match (e.g., '*self_attn.q_proj.weights')
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


def test_basic_patterns():
    print("Testing basic patterns...")
    
    tests = [
        ('layer1', 'layer1', True),
        ('layer1', 'layer2', False),
        ('model.layers.0.down_proj', '*down_proj', True),
        ('model.layers.1.down_proj', '*down_proj', True),
        ('model.layers.0.up_proj', '*down_proj', False),
        ('model.layers.0.down_proj', 'model.layers.*', True),
        ('model.layers.1.down_proj', 'model.layers.*', True),
        ('model.layers.0', 'model.layers.*', True),
        ('model.layers', 'model.layers.*', False),
        ('model.layers.0.down_proj', 'model.*.down_proj', True),
        ('model.layers.1.down_proj', 'model.*.down_proj', True),
        ('model.layers.0.up_proj', 'model.*.down_proj', False),
        ('model.layers.0.down_proj', '*down_proj.weights', True),
        ('model.layers.0.down_proj.weights', '*down_proj.weights', True),
        ('model.layers.0.down_proj.inputs', '*down_proj.weights', False),
        ('model.layers.0.q_proj', '*q_proj.inputs', True),
        ('model.layers.0.q_proj.inputs', '*q_proj.inputs', True),
        ('model.layers.0.q_proj.weights', '*q_proj.inputs', False),
        ('a.b.c.d', '*.*.c.*', True),
        ('x.y.c.z', '*.*.c.*', True),
        ('a.b.x.d', '*.*.c.*', False),
    ]
    
    passed = 0
    failed = 0
    for layer_name, pattern, expected in tests:
        result = match_fuzzy_pattern(layer_name, pattern)
        if result == expected:
            passed += 1
        else:
            failed += 1
            print(f"  FAILED: match_fuzzy_pattern('{layer_name}', '{pattern}') = {result}, expected {expected}")
    
    print(f"  Passed: {passed}/{len(tests)}")
    print(f"  Failed: {failed}/{len(tests)}")
    return failed == 0


def test_hifp8_fuzzy_cfg_patterns():
    print("\nTesting HIFP8_MINMAX_FUZZY_CFG patterns...")
    
    tests = [
        ('model.layers.0.self_attn.q_proj', '*self_attn.q_proj.inputs', True),
        ('model.layers.0.self_attn.q_proj.inputs', '*self_attn.q_proj.inputs', True),
        ('model.layers.0.self_attn.k_proj', '*self_attn.q_proj.inputs', False),
        ('model.layers.0.mlp.down_proj', '*down_proj.weights', True),
        ('model.layers.0.mlp.down_proj.weights', '*down_proj.weights', True),
        ('model.layers.0.mlp.up_proj', '*down_proj.weights', False),
    ]
    
    passed = 0
    failed = 0
    for layer_name, pattern, expected in tests:
        result = match_fuzzy_pattern(layer_name, pattern)
        if result == expected:
            passed += 1
        else:
            failed += 1
            print(f"  FAILED: match_fuzzy_pattern('{layer_name}', '{pattern}') = {result}, expected {expected}")
    
    print(f"  Passed: {passed}/{len(tests)}")
    print(f"  Failed: {failed}/{len(tests)}")
    return failed == 0


if __name__ == '__main__':
    print("=" * 60)
    print("Fuzzy Pattern Matching Test")
    print("=" * 60)
    
    result1 = test_basic_patterns()
    result2 = test_hifp8_fuzzy_cfg_patterns()
    
    print("\n" + "=" * 60)
    if result1 and result2:
        print("All tests PASSED!")
        sys.exit(0)
    else:
        print("Some tests FAILED!")
        sys.exit(1)
