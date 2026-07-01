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
"""HiFloat4 cast op tests.

Two layers:
  1. CPU (runs anywhere): the standalone op reference (amct_ops.hifloat4_cast.reference)
     must match the vendored amct_pytorch quant_hifx bit-for-bit. This guards the oracle.
  2. NPU (skipped without hardware): the AscendC kernel must match the reference to ~0 error
     (< 1e-6), including zero-input blocks (P1.6 acceptance red-line).
"""
import unittest

import torch

from amct_ops.hifloat4_cast.reference import hifloat4_reference


def _npu_available():
    try:
        import torch_npu  # noqa: F401
        return torch.npu.is_available()
    except Exception:
        return False


class TestHiFloat4Reference(unittest.TestCase):
    """CPU-only: op reference == amct_pytorch vendored quant_hifx."""

    def _assert_matches_vendored(self, x, qdim=-1):
        from amct_pytorch.common.utils.hifx4_quant import QType, quant_dequant_float
        vendored = quant_dequant_float(x, QType('hifx4').dim(qdim), force_py=True, force_fp32=True)
        ours = hifloat4_reference(x, qdim=qdim)
        self.assertEqual(vendored.shape, ours.shape)
        self.assertTrue(torch.equal(vendored.float(), ours.float()),
                        msg=f"reference diverges from quant_hifx (qdim={qdim})")

    def test_reference_matches_vendored_exact_block(self):
        torch.manual_seed(0)
        x = torch.randn(4, 128, dtype=torch.float32)  # 128 = 2 * 64, no padding
        self._assert_matches_vendored(x)

    def test_reference_matches_vendored_padded(self):
        torch.manual_seed(1)
        x = torch.randn(3, 100, dtype=torch.float32)  # 100 -> padded to 128
        self._assert_matches_vendored(x)

    def test_reference_matches_vendored_qdim(self):
        torch.manual_seed(2)
        x = torch.randn(80, 5, dtype=torch.float32)   # reduce along dim 0
        self._assert_matches_vendored(x, qdim=0)

    def test_reference_zero_is_nan_free(self):
        z = torch.zeros(2, 64, dtype=torch.float32)
        out = hifloat4_reference(z)
        self.assertFalse(torch.isnan(out).any())
        self.assertTrue(torch.equal(out, torch.zeros_like(out)))


@unittest.skipUnless(_npu_available(), "requires an Ascend NPU")
class TestHiFloat4Kernel(unittest.TestCase):
    """NPU: AscendC kernel == reference (< 1e-6), incl. zeros. The P1.6 red-line."""

    TOL = 1e-6

    @classmethod
    def setUpClass(cls):
        torch.npu.set_device(0)

    def _assert_kernel_matches_reference(self, x, qdim=-1):
        from amct_ops.hifloat4_cast import hifloat4_fake_quant
        y_ref = hifloat4_reference(x.float(), qdim=qdim).float()
        y_npu = hifloat4_fake_quant(x.npu(), qdim=qdim).cpu().float()
        self.assertEqual(y_ref.shape, y_npu.shape)
        self.assertLess((y_ref - y_npu).abs().max().item(), self.TOL)

    def test_kernel_bf16(self):
        torch.manual_seed(3)
        self._assert_kernel_matches_reference(torch.randn(4, 512, dtype=torch.bfloat16))

    def test_kernel_fp16(self):
        torch.manual_seed(4)
        self._assert_kernel_matches_reference(torch.randn(8, 256, dtype=torch.float16))

    def test_kernel_padded(self):
        torch.manual_seed(5)
        self._assert_kernel_matches_reference(torch.randn(3, 100, dtype=torch.bfloat16))

    def test_kernel_qdim(self):
        torch.manual_seed(6)
        self._assert_kernel_matches_reference(torch.randn(200, 7, dtype=torch.bfloat16), qdim=0)

    def test_kernel_zero_is_nan_free(self):
        from amct_ops.hifloat4_cast import hifloat4_fake_quant
        z = torch.zeros(2, 64, dtype=torch.bfloat16).npu()
        out = hifloat4_fake_quant(z)
        self.assertFalse(torch.isnan(out).any())
        self.assertEqual(out.abs().max().item(), 0.0)


if __name__ == '__main__':
    unittest.main()
