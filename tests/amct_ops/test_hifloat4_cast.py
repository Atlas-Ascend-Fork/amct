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
"""HiFloat4 cast op test.

Compares the AscendC kernel against amct's CPU reference (quant_dequant_float)
to ~0 error (< 1e-6), including zero-input blocks. Requires an Ascend NPU; skipped otherwise.
"""
import unittest

import torch

from amct_pytorch.common.utils.hifx4_quant import QType, quant_dequant_float


def _npu_available():
    try:
        import torch_npu  # noqa: F401
        return torch.npu.is_available()
    except Exception:
        return False


def _oracle(x, qdim=-1):
    """CPU reference: the pure-python hifx4 quant-dequant the kernel is built to match."""
    return quant_dequant_float(x, QType('hifx4').dim(qdim), force_py=True, force_fp32=True)


@unittest.skipUnless(_npu_available(), "requires an Ascend NPU")
class TestHiFloat4Kernel(unittest.TestCase):
    TOL = 1e-6

    @classmethod
    def setUpClass(cls):
        torch.npu.set_device(0)

    def _assert_kernel_matches_reference(self, x, qdim=-1):
        from amct_ops.hifloat4_cast import hifloat4_fake_quant
        y_ref = _oracle(x.cpu(), qdim).float()
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
