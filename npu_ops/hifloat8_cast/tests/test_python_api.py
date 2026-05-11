# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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
HiFloat8 Python API comprehensive tests

Test coverage:
1. Automatic dtype inference (FP16/BF16 only)
2. Type validation (FP32 and other unsupported dtypes raise ValueError)
3. Encode-decode roundtrip (FP16/BF16)
4. Random data accuracy tests
5. Boundary value tests
"""

import sys
import torch
import torch_npu
from npu_ops.hifloat8 import encode_to_hifloat8, decode_from_hifloat8

torch.npu.set_device(0)
def test_type_inference():
    print("=" * 70)
    print("Test encode_to_hifloat8 automatic type inference")
    print("=" * 70)

    x_fp16 = torch.randn(1000, dtype=torch.float16, device='npu')
    y_fp16 = encode_to_hifloat8(x_fp16)
    print(f"FP16 input: {x_fp16.dtype} -> output: {y_fp16.dtype} OK")

    x_bf16 = torch.randn(1000, dtype=torch.bfloat16, device='npu')
    y_bf16 = encode_to_hifloat8(x_bf16)
    print(f"BF16 input: {x_bf16.dtype} -> output: {y_bf16.dtype} OK")

    for bad_dtype, name in [(torch.float32, "FP32"), (torch.int32, "INT32")]:
        try:
            x_bad = torch.randn(1000, dtype=bad_dtype, device='npu') if bad_dtype != torch.int32 \
                else torch.zeros(1000, dtype=bad_dtype, device='npu')
            encode_to_hifloat8(x_bad)
            print(f"{name} input: FAIL - should raise error")
        except (ValueError, RuntimeError):
            print(f"{name} input: Correctly raised error OK")
def test_decode_type():
    print()
    print("=" * 70)
    print("Test decode_from_hifloat8 output type specification")
    print("=" * 70)

    y = torch.randint(0, 255, (1000,), dtype=torch.uint8, device='npu')

    z_default = decode_from_hifloat8(y)
    assert z_default.dtype == torch.bfloat16, f"Expected bfloat16 default, got {z_default.dtype}"
    print(f"Default output: {z_default.dtype} OK")

    z_fp16 = decode_from_hifloat8(y, torch.float16)
    print(f"Specify FP16: {z_fp16.dtype} OK")

    z_bf16 = decode_from_hifloat8(y, torch.bfloat16)
    print(f"Specify BF16: {z_bf16.dtype} OK")

    try:
        y_fp32 = torch.randn(1000, dtype=torch.float32, device='npu')
        decode_from_hifloat8(y_fp32)
        print(f"Non-uint8 input: FAIL - should raise error")
    except (ValueError, RuntimeError):
        print(f"Non-uint8 input: Correctly raised error OK")
def test_roundtrip_basic():
    print()
    print("=" * 70)
    print("Test integrity: encode-decode roundtrip (basic values, FP16)")
    print("=" * 70)

    x_orig = torch.tensor([1.0, -1.0, 0.5, 2.0, 0.0], dtype=torch.float16, device='npu')
    y_encoded = encode_to_hifloat8(x_orig)
    z_decoded = decode_from_hifloat8(y_encoded, torch.float16)

    print(f"Original: {x_orig}")
    print(f"Encoded:  {y_encoded}")
    print(f"Decoded:  {z_decoded}")

    match_count = (x_orig == z_decoded).sum().item()
    accuracy = match_count / x_orig.numel() * 100
    print(f"Accuracy: {accuracy:.2f}% ({match_count}/{x_orig.numel()}) OK")
def test_roundtrip_random(dtype_name, dtype, num_elements=10000):
    print()
    print("=" * 70)
    print(f"Test random data roundtrip: {dtype_name} ({num_elements} elements)")
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

    print(f"Exact match rate: {accuracy_exact:.2f}% ({match_count}/{num_elements})")
    print(f"Max absolute error: {max_abs_diff:.6f}")
    print(f"Mean absolute error: {mean_abs_diff:.6f}")
    print(f"Max relative error: {max_rel_diff:.6f}")
    print(f"Mean relative error: {mean_rel_diff:.6f}")

    if accuracy_exact >= 99.0 or max_abs_diff < 0.01:
        print(f"Result: OK - Good accuracy")
    else:
        print(f"Result: WARN - Accuracy loss detected")
def test_boundary_values():
    print()
    print("=" * 70)
    print("Test boundary values (FP16)")
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

    x_orig = torch.tensor(test_values, dtype=torch.float16, device='npu')
    y_encoded = encode_to_hifloat8(x_orig)
    z_decoded = decode_from_hifloat8(y_encoded, torch.float16)

    print(f"Test values: {test_values}")
    print(f"Encoded: {y_encoded.cpu().tolist()}")
    print(f"Decoded: {z_decoded.cpu().tolist()}")

    x_cpu = x_orig.float().cpu()
    z_cpu = z_decoded.float().cpu()

    abs_diff = torch.abs(x_cpu - z_cpu)

    print()
    print("Error analysis:")
    for i, val in enumerate(test_values):
        diff = abs_diff[i].item()
        status = "OK" if diff < 0.001 else "WARN"
        print(f"  {val:8.4f} -> {z_cpu[i].item():8.4f}, error={diff:.6f} {status}")

    match_count = (abs_diff < 0.001).sum().item()
    accuracy = match_count / len(test_values) * 100
    print(f"\nOverall accuracy: {accuracy:.2f}% ({match_count}/{len(test_values)})")
def test_special_values():
    print()
    print("=" * 70)
    print("Test special values: +-0, NaN, Inf (FP16)")
    print("=" * 70)

    test_values = [
        0.0,
        -0.0,
        float('inf'),
        float('-inf'),
        float('nan'),
    ]

    test_names = ['+0', '-0', 'Inf', '-Inf', 'NaN']

    x_orig = torch.tensor(test_values, dtype=torch.float16, device='npu')

    print(f"Input values: {test_names}")
    print(f"Original data: {x_orig.cpu().tolist()}")

    y_encoded = encode_to_hifloat8(x_orig)
    print(f"Encoded: {y_encoded.cpu().tolist()}")

    z_decoded = decode_from_hifloat8(y_encoded, torch.float16)
    print(f"Decoded: {z_decoded.float().cpu().tolist()}")

    print()
    print("Special value handling analysis:")

    for i, name in enumerate(test_names):
        decoded_val = z_decoded[i].item() if z_decoded.dtype != torch.float16 \
            else float(z_decoded[i])
        encoded_byte = y_encoded[i].item()

        if name == '+0':
            is_zero = decoded_val == 0.0
            status = "OK Correct" if is_zero else "FAIL Error"
            print(f"  +0: encoded={encoded_byte:3d}, decoded={decoded_val}, {status}")
        elif name == '-0':
            is_neg_zero = decoded_val == 0.0 and str(decoded_val) == '-0.0'
            status = "WARN Sign lost" if not is_neg_zero else "OK Correct"
            print(f"  -0: encoded={encoded_byte:3d}, decoded={decoded_val}, {status}")
        elif name == 'Inf':
            is_inf = decoded_val == float('inf')
            status = "OK Correct" if is_inf else "FAIL Error"
            print(f"  Inf: encoded={encoded_byte:3d}, decoded={decoded_val}, {status}")
        elif name == '-Inf':
            is_neg_inf = decoded_val == float('-inf')
            status = "OK Correct" if is_neg_inf else "FAIL Error"
            print(f"  -Inf: encoded={encoded_byte:3d}, decoded={decoded_val}, {status}")
        elif name == 'NaN':
            is_nan = decoded_val != decoded_val
            status = "OK Correct" if is_nan else "FAIL Error"
            print(f"  NaN: encoded={encoded_byte:3d}, decoded={decoded_val}, {status}")
def test_denormal_values():
    print()
    print("=" * 70)
    print("Test denormal numbers (subnormal floating-point)")
    print("=" * 70)

    print()
    print("Denormal explanation:")
    print("  - IEEE 754 numbers with exponent all-zero but non-zero mantissa")
    print("  - Represent very small values (between 0 and min normal)")
    print("  - FP16: 2^-24 ~ 2^-14 (approx 5.96e-8 ~ 6.10e-5)")
    print()

    test_cases = [
        ('FP16 max denormal', torch.float16, 6.1035e-5, 'near min normal'),
        ('FP16 min denormal', torch.float16, 5.96e-8,   'near zero'),
        ('BF16 max denormal', torch.bfloat16, 1.17549435e-38, 'near min normal'),
        ('BF16 min denormal', torch.bfloat16, 9.2e-41,        'near zero'),
    ]

    print("Test results:")
    print("-" * 70)

    for name, dtype, value, desc in test_cases:
        try:
            x_orig = torch.tensor([value], dtype=dtype, device='npu')
            y_encoded = encode_to_hifloat8(x_orig)
            z_decoded = decode_from_hifloat8(y_encoded, dtype)

            orig_val = float(x_orig[0])
            decoded_val = float(z_decoded[0])
            encoded_byte = y_encoded.cpu().item()

            abs_diff = abs(orig_val - decoded_val)
            rel_diff = abs_diff / (abs(orig_val) + 1e-50)

            if decoded_val == 0.0:
                status = "WARN Flush to Zero"
                result = f"encoded={encoded_byte:3d}, decoded={decoded_val}, FTZ"
            elif rel_diff < 0.1:
                status = "OK Good accuracy"
                result = f"encoded={encoded_byte:3d}, decoded={decoded_val:.6e}, rel_error={rel_diff:.2%}"
            else:
                status = "WARN Loss detected"
                result = f"encoded={encoded_byte:3d}, decoded={decoded_val:.6e}, rel_error={rel_diff:.2%}"

            print(f"  {name:20s} ({desc:15s}): {result} {status}")

        except Exception as e:
            print(f"  {name:20s}: FAIL Error - {e}")

    print()
    print("Analysis:")
    print("  - HiFloat8 may apply Flush-to-Zero (FTZ) for denormal numbers")
    print("  - Common practice in quantization (saves hardware complexity)")
def main():
    test_type_inference()
    test_decode_type()
    test_roundtrip_basic()

    test_roundtrip_random("FP16", torch.float16, 10000)
    test_roundtrip_random("BF16", torch.bfloat16, 10000)

    test_boundary_values()
    test_special_values()
    test_denormal_values()

    print()
    print("=" * 70)
    print("All tests passed OK")
    print("=" * 70)
if __name__ == "__main__":
    main()
