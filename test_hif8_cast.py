#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
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

import torch
import torch.nn as nn
import amct_pytorch as amct


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(128, 256, bias=True)
        self.linear2 = nn.Linear(256, 128, bias=True)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


def test_hif8_cast():
    print("=== HIF8 Cast Functionality Test ===")
    
    model = SimpleModel()
    model.eval()
    
    print(f"Original model: {model}")
    print(f"Linear1 weight dtype: {model.linear1.weight.dtype}")
    print(f"Linear2 weight dtype: {model.linear2.weight.dtype}")
    
    cfg = amct.HIF8_CAST_CFG
    print(f"\nUsing HIF8_CAST_CFG: {cfg}")
    
    print("\n1. Calling quantize()...")
    amct.quantize(model, config=cfg)
    
    print(f"\nModel after quantize: {model}")
    print(f"Linear1 type: {type(model.linear1)}")
    print(f"Linear2 type: {type(model.linear2)}")
    
    if hasattr(model.linear1, 'scale_w'):
        print(f"Linear1 scale_w: {model.linear1.scale_w}")
        print(f"Linear1 weight dtype: {model.linear1.weight.dtype}")
    
    print("\n2. Testing forward pass...")
    dummy_input = torch.randn(1, 128, dtype=torch.bfloat16)
    
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Output shape: {output.shape}")
    print(f"Output dtype: {output.dtype}")
    
    print("\n3. Calling convert()...")
    amct.convert(model)
    
    print(f"\nModel after convert: {model}")
    print(f"Linear1 type: {type(model.linear1)}")
    print(f"Linear2 type: {type(model.linear2)}")
    
    print("\n4. Testing forward pass after conversion...")
    with torch.no_grad():
        output_converted = model(dummy_input)
    
    print(f"Output shape: {output_converted.shape}")
    print(f"Output dtype: {output_converted.dtype}")
    
    print("\n=== Test Completed Successfully ===")


if __name__ == "__main__":
    test_hif8_cast()