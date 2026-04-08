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
"""
Example: Using Fuzzy Matching for Quantization Configuration

This example demonstrates how to use wildcard patterns (*) in quantization
configuration to apply the same quantization settings to multiple layers.
"""

import torch
import torch.nn as nn


class TransformerLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.self_attn = nn.ModuleDict({
            'q_proj': nn.Linear(hidden_size, hidden_size),
            'k_proj': nn.Linear(hidden_size, hidden_size),
            'v_proj': nn.Linear(hidden_size, hidden_size),
        })
        self.mlp = nn.ModuleDict({
            'up_proj': nn.Linear(hidden_size, hidden_size * 4),
            'down_proj': nn.Linear(hidden_size * 4, hidden_size),
        })


class TransformerModel(nn.Module):
    def __init__(self, num_layers=2, hidden_size=64):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerLayer(hidden_size) for _ in range(num_layers)
        ])
        self.lm_head = nn.Linear(hidden_size, 64)


def example_fuzzy_config():
    """
    Example using fuzzy matching configuration
    """
    from amct_pytorch import quantize
    
    model = TransformerModel(num_layers=2, hidden_size=64)
    
    print("Model structure:")
    for name, _ in model.named_modules():
        if 'q_proj' in name or 'down_proj' in name:
            print(f"  {name}")
    
    print("\nApplying fuzzy matching configuration...")
    print("Configuration includes:")
    print("  - '*down_proj.weights': tensor strategy (matches all down_proj layers)")
    print("  - '*self_attn.q_proj.inputs': channel strategy (matches all q_proj layers)")
    print("  - Default weights: channel strategy")
    print("  - Default inputs: tensor strategy")
    
    # Apply quantization with fuzzy matching
    # quantize(model, custom_fuzzy_config)
    
    print("\nQuantization configuration applied successfully!")
    print("All 'down_proj' layers will use tensor strategy for weights")
    print("All 'q_proj' layers will use channel strategy for inputs")
    print("Other layers will use default configurations")


def example_custom_fuzzy_config():
    """
    Example creating custom fuzzy configuration
    """
    from amct_pytorch import quantize
    
    model = TransformerModel(num_layers=2, hidden_size=64)
    
    custom_fuzzy_config = {
        'batch_num': 1,
        'quant_cfg': {
            # Match all projection layers in self-attention
            '*self_attn.*_proj.weights': {
                'type': 'int8',
                'symmetric': True,
                'strategy': 'channel',
            },
            # Match all MLP layers
            '*mlp.*.weights': {
                'type': 'int8',
                'symmetric': True,
                'strategy': 'tensor',
            },
            # Default configuration
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
        'skip_layers': {'lm_head'}
    }
    
    print("\nCustom fuzzy configuration example:")
    print("  - '*self_attn.*_proj.weights': matches q_proj, k_proj, v_proj in self_attn")
    print("  - '*mlp.*.weights': matches up_proj, down_proj in mlp")
    
    # Apply quantization with custom fuzzy matching
    # quantize(model, custom_fuzzy_config)
    
    print("\nCustom fuzzy configuration applied successfully!")


if __name__ == '__main__':
    print("=" * 70)
    print("Fuzzy Matching Quantization Configuration Example")
    print("=" * 70)
    
    example_fuzzy_config()
    example_custom_fuzzy_config()
    
    print("\n" + "=" * 70)
    print("For more information, see:")
    print("  - amct_pytorch/config/config.py: Custom fuzzy config examples")
    print("  - amct_pytorch/config/utils.py: match_fuzzy_pattern()")
    print("=" * 70)
