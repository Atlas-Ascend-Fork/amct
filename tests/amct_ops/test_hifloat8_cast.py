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
import unittest

import torch
import torch_npu

from amct_ops.hifloat8_cast import decode_from_hifloat8, encode_to_hifloat8


EXPECTED_ENCODE = {
    '+0': 0x00,
    '-0': 0x00,
    '+Inf': 0x6F,
    '-Inf': 0xEF,
    'NaN': 0x80,
}
DEVICE = 'npu'


class TestHiFloat8Cast(unittest.TestCase):
    @staticmethod
    def _is_decoded_special_value_valid(name, decoded_val):
        if name in ('+0', '-0'):
            return decoded_val == 0.0
        if name == '+Inf':
            return decoded_val == float('inf')
        if name == '-Inf':
            return decoded_val == float('-inf')
        if name == 'NaN':
            return decoded_val != decoded_val
        return False

    @classmethod
    def setUpClass(cls):
        torch.npu.set_device(0)

    def test_type_inference(self):
        x_fp16 = torch.randn(1000, dtype=torch.float16, device=DEVICE)
        self.assertEqual(encode_to_hifloat8(x_fp16).dtype, torch.uint8)

        x_bf16 = torch.randn(1000, dtype=torch.bfloat16, device=DEVICE)
        self.assertEqual(encode_to_hifloat8(x_bf16).dtype, torch.uint8)

        for dtype in (torch.float32, torch.int32):
            if dtype == torch.int32:
                x_bad = torch.zeros(1000, dtype=dtype, device=DEVICE)
            else:
                x_bad = torch.randn(1000, dtype=dtype, device=DEVICE)
            with self.assertRaises((ValueError, RuntimeError)):
                encode_to_hifloat8(x_bad)

    def test_decode_output_type(self):
        y = torch.randint(0, 255, (1000,), dtype=torch.uint8, device=DEVICE)

        self.assertEqual(decode_from_hifloat8(y).dtype, torch.bfloat16)
        self.assertEqual(decode_from_hifloat8(y, torch.float16).dtype, torch.float16)
        self.assertEqual(decode_from_hifloat8(y, torch.bfloat16).dtype, torch.bfloat16)

        y_fp32 = torch.randn(1000, dtype=torch.float32, device=DEVICE)
        with self.assertRaises((ValueError, RuntimeError)):
            decode_from_hifloat8(y_fp32)

    def test_roundtrip_basic(self):
        x_orig = torch.tensor([1.0, -1.0, 0.5, 2.0, 0.0], dtype=torch.float16, device=DEVICE)
        y_encoded = encode_to_hifloat8(x_orig)
        z_decoded = decode_from_hifloat8(y_encoded, torch.float16)
        self.assertTrue(bool((x_orig == z_decoded).all().item()))

    def test_roundtrip_random_fp16(self):
        self._check_roundtrip_random(torch.float16)

    def test_roundtrip_random_bf16(self):
        self._check_roundtrip_random(torch.bfloat16)

    def test_boundary_values(self):
        test_values = [
            0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 0.25, 0.125,
            3.0, -3.0, 0.0625, 0.03125, 0.01, 0.001, 10.0, 100.0,
        ]
        x_orig = torch.tensor(test_values, dtype=torch.float16, device=DEVICE)
        y_encoded = encode_to_hifloat8(x_orig)
        z_decoded = decode_from_hifloat8(y_encoded, torch.float16)

        abs_diff = torch.abs(x_orig.float().cpu() - z_decoded.float().cpu())
        ok_count = int((abs_diff < 0.001).sum().item())
        self.assertGreaterEqual(ok_count, int(len(test_values) * 0.9))

    def test_special_values(self):
        fp16_encoded = self._check_special_one_dtype(torch.float16)
        bf16_encoded = self._check_special_one_dtype(torch.bfloat16)
        self.assertTrue(bool((fp16_encoded == bf16_encoded).all().item()))

    def test_denormal_values(self):
        test_cases = [
            (torch.float16, 6.097555e-5),
            (torch.float16, 5.96e-8),
            (torch.bfloat16, 1.1663108e-38),
            (torch.bfloat16, 9.2e-41),
        ]
        for dtype, value in test_cases:
            x_orig = torch.tensor([value], dtype=dtype, device=DEVICE)
            y_encoded = encode_to_hifloat8(x_orig)
            z_decoded = decode_from_hifloat8(y_encoded, dtype)
            decoded_val = z_decoded[0].item()
            if decoded_val == 0.0:
                continue
            rel_diff = abs(value - decoded_val) / abs(value)
            self.assertLess(rel_diff, 0.2)

    def test_fp16_subnormal_rounding_regression(self):
        value = 2.7180e-05
        expected = 0x7E
        x_orig = torch.tensor([value], dtype=torch.float16, device=DEVICE)
        y_encoded = encode_to_hifloat8(x_orig)
        self.assertEqual(y_encoded.cpu().item(), expected)

    def test_full_256_decode(self):
        all_hif8 = torch.arange(256, dtype=torch.int32, device=DEVICE).to(torch.uint8)

        z_fp16 = decode_from_hifloat8(all_hif8, torch.float16)
        z_bf16 = decode_from_hifloat8(all_hif8, torch.bfloat16)

        self.assertEqual((~z_fp16.isnan()).sum().item(), z_fp16.numel() - 1)
        self.assertEqual((~z_bf16.isnan()).sum().item(), z_bf16.numel() - 1)

    def _check_roundtrip_random(self, dtype, num_elements=10000):
        x_orig = torch.randn(num_elements, dtype=dtype, device=DEVICE)
        y_encoded = encode_to_hifloat8(x_orig)
        z_decoded = decode_from_hifloat8(y_encoded, dtype)

        abs_diff = torch.abs(x_orig.float().cpu() - z_decoded.float().cpu())
        rel_diff = abs_diff / (torch.abs(x_orig.float().cpu()) + 1e-8)
        nonzero_mask = torch.abs(x_orig.float().cpu()) > 1e-4
        max_rel_diff = rel_diff[nonzero_mask].max().item() if nonzero_mask.any() else 0.0
        self.assertLess(max_rel_diff, 0.25)

    def _check_special_one_dtype(self, dtype):
        test_names = ['+0', '-0', '+Inf', '-Inf', 'NaN']
        test_values = [0.0, -0.0, float('inf'), float('-inf'), float('nan')]

        x_orig = torch.tensor(test_values, dtype=dtype, device=DEVICE)
        y_encoded = encode_to_hifloat8(x_orig)
        z_decoded = decode_from_hifloat8(y_encoded, dtype)

        enc_cpu = y_encoded.cpu()
        dec_cpu = z_decoded.float().cpu()
        for i, name in enumerate(test_names):
            self.assertEqual(enc_cpu[i].item(), EXPECTED_ENCODE[name])
            self.assertTrue(self._is_decoded_special_value_valid(name, dec_cpu[i].item()))
        return enc_cpu


if __name__ == '__main__':
    unittest.main()
