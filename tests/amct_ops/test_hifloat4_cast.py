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
import json
import os
import shutil
import tempfile
import unittest

import torch

from amct_pytorch.common.utils.hifloat4_quant import QType, quant_dequant_float
from amct_pytorch.common.utils.hifloat4_quant.hif4_pack_unpack import hif4_cast
from amct_pytorch.common.utils.decode.unpack_hif4 import unpack_hif4_checkpoint


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


class TestHiFloat4CastShapes(unittest.TestCase):
    """CPU-only shape regression tests for hif4_cast (no NPU required).

    Guards against the squeeze index-shifting bug: the two consecutive keepdim-1
    dims on e1_8 must be collapsed from the right (squeeze(qdim) twice), otherwise
    the second squeeze lands on the wrong dimension and causes a shape mismatch.
    """

    def test_packed_shape_2x128(self):
        """Exact regression case from review: hif4_cast(randn(2,128), output='packed')."""
        x = torch.randn(2, 128, dtype=torch.bfloat16)
        scale, value = hif4_cast(x, output="packed")
        self.assertEqual(scale.shape, (2, 2, 4))
        self.assertEqual(value.shape, (2, 64))
        self.assertEqual(scale.dtype, torch.uint8)
        self.assertEqual(value.dtype, torch.uint8)

    def test_dequant_shape_2x128(self):
        x = torch.randn(2, 128, dtype=torch.bfloat16)
        out = hif4_cast(x, output="dequant")
        self.assertEqual(out.shape, x.shape)
        self.assertEqual(out.dtype, x.dtype)

    def test_packed_shape_3x512(self):
        x = torch.randn(3, 512, dtype=torch.bfloat16)
        scale, value = hif4_cast(x, output="packed")
        self.assertEqual(scale.shape, (3, 8, 4))
        self.assertEqual(value.shape, (3, 256))

    def test_dequant_shape_3x512(self):
        x = torch.randn(3, 512, dtype=torch.bfloat16)
        out = hif4_cast(x, output="dequant")
        self.assertEqual(out.shape, x.shape)

    def test_packed_shape_1x64(self):
        x = torch.randn(1, 64, dtype=torch.bfloat16)
        scale, value = hif4_cast(x, output="packed")
        self.assertEqual(scale.shape, (1, 1, 4))
        self.assertEqual(value.shape, (1, 32))


class TestUnpackConfigCleanup(unittest.TestCase):
    """Verify config.json is cleaned up after unpack: quantization_config removed,
    torch_dtype updated to match the unpack output dtype."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.outdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        shutil.rmtree(self.outdir, ignore_errors=True)

    def _make_checkpoint(self, torch_dtype="float32", with_quant_config=True):
        """Create a minimal HiF4 packed checkpoint with the given config.json."""
        import numpy as np
        from safetensors.torch import save_file

        # Create two packed weights
        w1 = torch.randn(4, 64, dtype=torch.bfloat16)
        w2 = torch.randn(8, 128, dtype=torch.bfloat16)
        scale1, val1 = hif4_cast(w1, output="packed")
        scale2, val2 = hif4_cast(w2, output="packed")

        save_file({"model.layers.0.weight": val1,
                    "model.layers.0.weight_scale": scale1},
                   os.path.join(self.tmpdir, "model-00001-of-00002.safetensors"))
        save_file({"model.layers.1.weight": val2,
                    "model.layers.1.weight_scale": scale2},
                   os.path.join(self.tmpdir, "model-00002-of-00002.safetensors"))

        weight_map = {
            "model.layers.0.weight": "model-00001-of-00002.safetensors",
            "model.layers.0.weight_scale": "model-00001-of-00002.safetensors",
            "model.layers.1.weight": "model-00002-of-00002.safetensors",
            "model.layers.1.weight_scale": "model-00002-of-00002.safetensors",
        }
        index = {"metadata": {"total_size": 0}, "weight_map": weight_map}
        with open(os.path.join(self.tmpdir, "model.safetensors.index.json"), "w") as f:
            json.dump(index, f)

        cfg = {"torch_dtype": torch_dtype, "hidden_size": 64}
        if with_quant_config:
            cfg["quantization_config"] = {"quant_method": "hif4"}
        with open(os.path.join(self.tmpdir, "config.json"), "w") as f:
            json.dump(cfg, f)

    def _read_config(self):
        with open(os.path.join(self.outdir, "config.json"), "r") as f:
            return json.load(f)

    def test_fp16_removes_quant_config_and_updates_dtype(self):
        self._make_checkpoint(torch_dtype="float32", with_quant_config=True)
        unpack_hif4_checkpoint(self.tmpdir, self.outdir, out_dtype="fp16", quiet=True)
        cfg = self._read_config()
        self.assertNotIn("quantization_config", cfg)
        self.assertEqual(cfg["torch_dtype"], "float16")

    def test_bf16_removes_quant_config_and_updates_dtype(self):
        self._make_checkpoint(torch_dtype="float32", with_quant_config=True)
        unpack_hif4_checkpoint(self.tmpdir, self.outdir, out_dtype="bf16", quiet=True)
        cfg = self._read_config()
        self.assertNotIn("quantization_config", cfg)
        self.assertEqual(cfg["torch_dtype"], "bfloat16")

    def test_no_quant_config_still_updates_dtype(self):
        self._make_checkpoint(torch_dtype="float32", with_quant_config=False)
        unpack_hif4_checkpoint(self.tmpdir, self.outdir, out_dtype="bf16", quiet=True)
        cfg = self._read_config()
        self.assertNotIn("quantization_config", cfg)
        self.assertEqual(cfg["torch_dtype"], "bfloat16")


if __name__ == '__main__':
    unittest.main()
