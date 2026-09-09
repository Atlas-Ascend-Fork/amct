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

Compares the AscendC kernel against the pure-torch hif4_cast dequant reference
to ~0 error (< 1e-6), including zero-input blocks. Requires an Ascend NPU; skipped
otherwise. Also includes CPU-only shape regression tests for hif4_cast pack/dequant.
"""

import unittest

import torch

from amct_pytorch.quantization.dtypes.hifp_impl import hif4_pack, hifloat4_fake_quant


def _npu_available():
    try:
        import torch_npu  # noqa: F401

        return torch.npu.is_available()
    except Exception:
        return False


def _oracle(x):
    """CPU reference: hif4_cast dequant round-trip."""
    return hifloat4_fake_quant(x).float()


@unittest.skipUnless(_npu_available(), "requires an Ascend NPU")
class TestHiFloat4Kernel(unittest.TestCase):
    TOL = 1e-6

    @classmethod
    def setUpClass(cls):
        torch.npu.set_device(0)

    def test_kernel_bf16(self):
        torch.manual_seed(3)
        self._assert_kernel_matches_reference(torch.randn(4, 512, dtype=torch.bfloat16))

    def test_kernel_fp16(self):
        torch.manual_seed(4)
        self._assert_kernel_matches_reference(torch.randn(8, 256, dtype=torch.float16))

    def test_kernel_unaligned_raises(self):
        from amct_ops.hifloat4_cast import hifloat4_fake_quant as npu_fake_quant

        torch.manual_seed(5)
        x = torch.randn(3, 100, dtype=torch.bfloat16)
        with self.assertRaises(RuntimeError):
            npu_fake_quant(x.npu())

    def test_kernel_zero_is_nan_free(self):
        from amct_ops.hifloat4_cast import hifloat4_fake_quant as npu_fake_quant

        z = torch.zeros(2, 64, dtype=torch.bfloat16).npu()
        out = npu_fake_quant(z)
        self.assertFalse(torch.isnan(out).any())
        self.assertEqual(out.abs().max().item(), 0.0)

    def test_kernel_nan_block_poison_keeps_device_healthy(self):
        """Regression test: NaN/Inf blocks are poisoned to NaN as a whole, and the
        device must still launch normally afterwards.

        The earlier hand-written kernel injected NaN into the scale; NaN flowing
        through vdiv/vconv triggered an AI Core exception (507057), and after one
        run with NaN-containing input every subsequent launch failed.
        """
        from amct_ops.hifloat4_cast import hifloat4_fake_quant as npu_fake_quant

        torch.manual_seed(6)
        x = torch.randn(4, 512, dtype=torch.bfloat16)
        x[:, :64] = float("nan")
        x[:, 64:128] = float("inf")

        out = npu_fake_quant(x.npu()).cpu()

        # Poison contract: blocks containing non-finite elements are NaN as a whole;
        # clean blocks contain no NaN.
        self.assertTrue(torch.isnan(out[:, :128]).all())
        self.assertFalse(torch.isnan(out[:, 128:]).any())
        # Clean blocks are bit-identical to the CPU reference (NaN block payload bits
        # may differ).
        ref = _oracle(x)
        self.assertTrue(
            torch.equal(
                out[:, 128:].float().view(torch.uint32),
                ref[:, 128:].view(torch.uint32),
            )
        )

        # Device-health regression: a subsequent launch with finite input must work.
        x2 = torch.randn(2, 128, dtype=torch.bfloat16).npu()
        out2 = npu_fake_quant(x2)
        torch.npu.synchronize()
        self.assertFalse(torch.isnan(out2.cpu()).any())

    def _assert_kernel_matches_reference(self, x):
        from amct_ops.hifloat4_cast import hifloat4_fake_quant as npu_fake_quant

        y_ref = _oracle(x.cpu()).float()
        y_npu = npu_fake_quant(x.npu()).cpu().float()
        self.assertEqual(y_ref.shape, y_npu.shape)
        self.assertLess((y_ref - y_npu).abs().max().item(), self.TOL)


class TestHiFloat4CastShapes(unittest.TestCase):
    """CPU-only shape regression tests for hif4_cast.

    Guards against the squeeze index-shifting bug: the two consecutive keepdim-1
    dims on e1_8 must be collapsed from the right (squeeze(qdim) twice), otherwise
    the second squeeze lands on the wrong dimension and causes a shape mismatch.
    """

    def test_packed_shape_2x128(self):
        x = torch.randn(2, 128, dtype=torch.bfloat16)
        scale, value = hif4_pack(x)
        self.assertEqual(scale.shape, (2, 2, 4))
        self.assertEqual(value.shape, (2, 64))
        self.assertEqual(scale.dtype, torch.uint8)
        self.assertEqual(value.dtype, torch.uint8)

    def test_dequant_shape_2x128(self):
        x = torch.randn(2, 128, dtype=torch.bfloat16)
        out = hifloat4_fake_quant(x)
        self.assertEqual(out.shape, x.shape)
        self.assertEqual(out.dtype, x.dtype)

    def test_packed_shape_3x512(self):
        x = torch.randn(3, 512, dtype=torch.bfloat16)
        scale, value = hif4_pack(x)
        self.assertEqual(scale.shape, (3, 8, 4))
        self.assertEqual(value.shape, (3, 256))

    def test_dequant_shape_3x512(self):
        x = torch.randn(3, 512, dtype=torch.bfloat16)
        out = hifloat4_fake_quant(x)
        self.assertEqual(out.shape, x.shape)

    def test_packed_shape_1x64(self):
        x = torch.randn(1, 64, dtype=torch.bfloat16)
        scale, value = hif4_pack(x)
        self.assertEqual(scale.shape, (1, 1, 4))
        self.assertEqual(value.shape, (1, 32))

    def test_fake_quant_unaligned_raises(self):
        x = torch.randn(3, 100, dtype=torch.bfloat16)
        with self.assertRaises(RuntimeError):
            hifloat4_fake_quant(x)


if __name__ == '__main__':
    unittest.main()
