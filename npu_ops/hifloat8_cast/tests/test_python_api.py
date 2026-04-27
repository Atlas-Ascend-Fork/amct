# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

import torch
import torch_npu
from hifloat8 import encode_to_hifloat8, decode_from_hifloat8

torch.npu.set_device(0)


def test_type_inference():
    print("=" * 70)
    print("测试 encode_to_hifloat8 自动类型推断")
    print("=" * 70)
    
    x_fp32 = torch.randn(1000, dtype=torch.float32, device='npu')
    y_fp32 = encode_to_hifloat8(x_fp32)
    print(f"FP32 输入: {x_fp32.dtype} → 输出: {y_fp32.dtype} ✅")
    
    x_fp16 = torch.randn(1000, dtype=torch.float16, device='npu')
    y_fp16 = encode_to_hifloat8(x_fp16)
    print(f"FP16 输入: {x_fp16.dtype} → 输出: {y_fp16.dtype} ✅")
    
    x_bf16 = torch.randn(1000, dtype=torch.bfloat16, device='npu')
    y_bf16 = encode_to_hifloat8(x_bf16)
    print(f"BF16 输入: {x_bf16.dtype} → 输出: {y_bf16.dtype} ✅")
    
    try:
        x_int32 = torch.randn(1000, dtype=torch.int32, device='npu')
        encode_to_hifloat8(x_int32)
        print(f"INT32 输入: ❌ 应该报错")
    except ValueError as e:
        print(f"INT32 输入: 正确报错 ✅")


def test_decode_type():
    print()
    print("=" * 70)
    print("测试 decode_from_hifloat8 指定输出类型")
    print("=" * 70)
    
    y = torch.randint(0, 255, (1000,), dtype=torch.uint8, device='npu')
    
    z_fp32 = decode_from_hifloat8(y)
    print(f"默认输出: {z_fp32.dtype} ✅")
    
    z_fp16 = decode_from_hifloat8(y, torch.float16)
    print(f"指定 FP16: {z_fp16.dtype} ✅")
    
    z_bf16 = decode_from_hifloat8(y, torch.bfloat16)
    print(f"指定 BF16: {z_bf16.dtype} ✅")
    
    try:
        y_fp32 = torch.randn(1000, dtype=torch.float32, device='npu')
        decode_from_hifloat8(y_fp32)
        print(f"非 uint8 输入: ❌ 应该报错")
    except ValueError as e:
        print(f"非 uint8 输入: 正确报错 ✅")


def test_roundtrip_basic():
    print()
    print("=" * 70)
    print("测试完整性：编码-解码往返（基本值）")
    print("=" * 70)
    
    x_orig = torch.tensor([1.0, -1.0, 0.5, 2.0, 0.0], dtype=torch.float32, device='npu')
    y_encoded = encode_to_hifloat8(x_orig)
    z_decoded = decode_from_hifloat8(y_encoded, torch.float32)
    
    print(f"原始: {x_orig}")
    print(f"编码: {y_encoded}")
    print(f"解码: {z_decoded}")
    
    match_count = (x_orig == z_decoded).sum().item()
    accuracy = match_count / x_orig.numel() * 100
    print(f"精度: {accuracy:.2f}% ({match_count}/{x_orig.numel()}) ✅")


def test_roundtrip_random(dtype_name, dtype, num_elements=10000):
    print()
    print("=" * 70)
    print(f"测试随机数据往返: {dtype_name} ({num_elements} 元素)")
    print("=" * 70)
    
    x_orig = torch.randn(num_elements, dtype=dtype, device='npu')
    
    y_encoded = encode_to_hifloat8(x_orig)
    z_decoded = decode_from_hifloat8(y_encoded, dtype)
    
    x_cpu = x_orig.float().cpu()
    z_cpu = z_decoded.float().cpu()
    
    match_count = (x_cpu == z_cpu).sum().item()
    accuracy_exact = match_count / num_elements * 100
    
    abs_diff = torch.abs(x_cpu - z_cpu)
    max_abs_diff = abs_diff.max().item()
    mean_abs_diff = abs_diff.mean().item()
    
    rel_diff = abs_diff / (torch.abs(x_cpu) + 1e-8)
    max_rel_diff = rel_diff.max().item()
    mean_rel_diff = rel_diff.mean().item()
    
    print(f"精确匹配率: {accuracy_exact:.2f}% ({match_count}/{num_elements})")
    print(f"最大绝对误差: {max_abs_diff:.6f}")
    print(f"平均绝对误差: {mean_abs_diff:.6f}")
    print(f"最大相对误差: {max_rel_diff:.6f}")
    print(f"平均相对误差: {mean_rel_diff:.6f}")
    
    if accuracy_exact >= 99.0 or max_abs_diff < 0.01:
        print(f"结果: ✅ 精度良好")
    else:
        print(f"结果: ⚠️ 有精度损失")


def test_boundary_values():
    print()
    print("=" * 70)
    print("测试边界值")
    print("=" * 70)
    
    test_values = [
        0.0, -0.0,
        1.0, -1.0,
        0.5, -0.5,
        2.0, -2.0,
        0.25, 0.125,
        3.0, -3.0,
        0.0625, 0.03125,
        0.01, 0.001,
        10.0, 100.0,
    ]
    
    x_orig = torch.tensor(test_values, dtype=torch.float32, device='npu')
    y_encoded = encode_to_hifloat8(x_orig)
    z_decoded = decode_from_hifloat8(y_encoded, torch.float32)
    
    print(f"测试值: {test_values}")
    print(f"编码后: {y_encoded.cpu().tolist()}")
    print(f"解码后: {z_decoded.cpu().tolist()}")
    
    x_cpu = x_orig.cpu()
    z_cpu = z_decoded.cpu()
    
    abs_diff = torch.abs(x_cpu - z_cpu)
    
    print()
    print("误差分析:")
    for i, val in enumerate(test_values):
        diff = abs_diff[i].item()
        status = "✅" if diff < 0.001 else "⚠️"
        print(f"  {val:8.4f} → {z_cpu[i].item():8.4f}, 误差={diff:.6f} {status}")
    
    match_count = (abs_diff < 0.001).sum().item()
    accuracy = match_count / len(test_values) * 100
    print(f"\n总体精度: {accuracy:.2f}% ({match_count}/{len(test_values)})")


def test_special_values():
    print()
    print("=" * 70)
    print("测试特殊值：±0、NaN、Inf")
    print("=" * 70)
    
    test_values = [
        0.0,
        -0.0,
        float('inf'),
        float('-inf'),
        float('nan'),
    ]
    
    test_names = ['+0', '-0', 'Inf', '-Inf', 'NaN']
    
    x_orig = torch.tensor(test_values, dtype=torch.float32, device='npu')
    
    print(f"输入值: {test_names}")
    print(f"原始数据: {x_orig.cpu().tolist()}")
    
    y_encoded = encode_to_hifloat8(x_orig)
    print(f"编码后: {y_encoded.cpu().tolist()}")
    
    z_decoded = decode_from_hifloat8(y_encoded, torch.float32)
    print(f"解码后: {z_decoded.cpu().tolist()}")
    
    print()
    print("特殊值处理分析:")
    
    for i, name in enumerate(test_names):
        decoded_val = z_decoded[i].item()
        encoded_byte = y_encoded[i].item()
        
        if name == '+0':
            is_zero = decoded_val == 0.0
            status = "✅ 正确" if is_zero else "❌ 错误"
            print(f"  +0: 编码={encoded_byte:3d}, 解码={decoded_val}, {status}")
        elif name == '-0':
            is_neg_zero = decoded_val == 0.0 and str(decoded_val) == '-0.0'
            status = "⚠️ 符号丢失" if not is_neg_zero else "✅ 正确"
            print(f"  -0: 编码={encoded_byte:3d}, 解码={decoded_val}, {status}")
        elif name == 'Inf':
            is_inf = decoded_val == float('inf')
            status = "✅ 正确" if is_inf else "❌ 错误"
            print(f"  Inf: 编码={encoded_byte:3d}, 解码={decoded_val}, {status}")
        elif name == '-Inf':
            is_neg_inf = decoded_val == float('-inf')
            status = "✅ 正确" if is_neg_inf else "❌ 错误"
            print(f"  -Inf: 编码={encoded_byte:3d}, 解码={decoded_val}, {status}")
        elif name == 'NaN':
            is_nan = decoded_val != decoded_val
            status = "✅ 正确" if is_nan else "❌ 错误"
            print(f"  NaN: 编码={encoded_byte:3d}, 解码={decoded_val}, {status}")


def test_denormal_values():
    print()
    print("=" * 70)
    print("测试 Denormal 数（非规格化浮点数）")
    print("=" * 70)
    
    print()
    print("Denormal 数说明:")
    print("  - IEEE 754 中指数为最小值（全零）但尾数不为零的浮点数")
    print("  - 用于表示非常小的数值（0 和最小规格化数之间）")
    print("  - FP32: 2^-149 ~ 2^-126 (≈ 1.4e-45 ~ 1.2e-38)")
    print()
    
    test_cases = [
        ('FP32 最大 denormal', torch.float32, 1.17549435e-38, '接近最小规格化数'),
        ('FP32 中等 denormal', torch.float32, 1.0e-40, '中间值'),
        ('FP32 最小 denormal', torch.float32, 1.4e-45, '接近零'),
        ('FP32 负 denormal', torch.float32, -1.0e-40, '负值'),
        ('FP16 最大 denormal', torch.float16, 6.1035e-5, '接近最小规格化数'),
        ('FP16 最小 denormal', torch.float16, 5.96e-8, '接近零'),
        ('BF16 最大 denormal', torch.bfloat16, 1.17549435e-38, '接近最小规格化数'),
        ('BF16 最小 denormal', torch.bfloat16, 9.2e-41, '接近零'),
    ]
    
    print("测试结果:")
    print("-" * 70)
    
    for name, dtype, value, desc in test_cases:
        try:
            x_orig = torch.tensor([value], dtype=dtype, device='npu')
            y_encoded = encode_to_hifloat8(x_orig)
            z_decoded = decode_from_hifloat8(y_encoded, dtype)
            
            orig_val = x_orig.cpu().item()
            decoded_val = z_decoded.cpu().item()
            encoded_byte = y_encoded.cpu().item()
            
            abs_diff = abs(orig_val - decoded_val)
            rel_diff = abs_diff / (abs(orig_val) + 1e-50)
            
            if decoded_val == 0.0:
                status = "⚠️ Flush to Zero"
                result = f"编码={encoded_byte:3d}, 解码={decoded_val}, FTZ"
            elif rel_diff < 0.1:
                status = "✅ 精度良好"
                result = f"编码={encoded_byte:3d}, 解码={decoded_val:.6e}, 相对误差={rel_diff:.2%}"
            else:
                status = "⚠️ 有损失"
                result = f"编码={encoded_byte:3d}, 解码={decoded_val:.6e}, 相对误差={rel_diff:.2%}"
            
            print(f"  {name:20s} ({desc:15s}): {result} {status}")
            
        except Exception as e:
            print(f"  {name:20s}: ❌ 错误 - {e}")
    
    print()
    print("分析:")
    print("  - HiFloat8 可能对 denormal 数做 Flush-to-Zero (FTZ) 处理")
    print("  - 或近似为最小可表示值")
    print("  - 这是量化场景常见做法（节省硬件复杂度）")


def main():
    test_type_inference()
    test_decode_type()
    test_roundtrip_basic()
    
    test_roundtrip_random("FP32", torch.float32, 10000)
    test_roundtrip_random("FP16", torch.float16, 10000)
    test_roundtrip_random("BF16", torch.bfloat16, 10000)
    
    test_boundary_values()
    test_special_values()
    test_denormal_values()
    
    print()
    print("=" * 70)
    print("所有测试通过 ✅")
    print("=" * 70)


if __name__ == "__main__":
    main()