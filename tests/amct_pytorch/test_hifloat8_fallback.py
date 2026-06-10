# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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
"""UT for the HiFloat8 amct_ops fallback path.

These tests exercise the code that the mocked-npu_quantize tests in test_gptq.py
and test_cast.py cannot reach: the hifloat8_supported() liveness probe and the
hifloat8_fake_quant amct_ops fallback (quant_util.py). The fallback is the whole
point of PR #147 -- run hifloat8 fake-quant when torch_npu lacks a native cast.
"""
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

import torch

from amct_pytorch.common.utils import quant_util


def _install_fake_torch_npu(with_hif8=True, cast_raises=False):
    """Build a fake torch_npu module and register it in sys.modules."""
    mod = types.ModuleType('torch_npu')
    if with_hif8:
        mod.hifloat8 = 'hifloat8_enum'

        def _cast(tensor, dtype, input_dtype=None):
            if cast_raises:
                raise RuntimeError('aclnnCast 161002: DT_HIFLOAT8 unsupported')
            return tensor
        mod.npu_dtype_cast = _cast
    sys.modules['torch_npu'] = mod
    return mod


def _install_fake_amct_ops(record):
    """Inject a stub amct_ops.hifloat8_cast with encode/decode that record calls."""
    pkg = types.ModuleType('amct_ops')
    sub = types.ModuleType('amct_ops.hifloat8_cast')

    def encode_to_hifloat8(tensor):
        record['encoded_dtype'] = tensor.dtype
        record['encode_called'] = True
        return tensor

    def decode_from_hifloat8(codes, dtype):
        record['decode_called'] = True
        return codes.to(dtype)

    sub.encode_to_hifloat8 = encode_to_hifloat8
    sub.decode_from_hifloat8 = decode_from_hifloat8
    pkg.hifloat8_cast = sub
    sys.modules['amct_ops'] = pkg
    sys.modules['amct_ops.hifloat8_cast'] = sub


class TestHifloat8Supported(unittest.TestCase):
    """quant_util.hifloat8_supported() probe branches."""

    def setUp(self):
        quant_util.hifloat8_supported.cache_clear()
        self._saved_npu = torch.Tensor.npu
        torch.Tensor.npu = lambda self: self

    def tearDown(self):
        quant_util.hifloat8_supported.cache_clear()
        torch.Tensor.npu = self._saved_npu
        sys.modules.pop('torch_npu', None)

    def test_returns_false_when_torch_npu_missing(self):
        sys.modules['torch_npu'] = None  # force ImportError on `import torch_npu`
        self.assertFalse(quant_util.hifloat8_supported())

    def test_returns_false_when_hif8_attr_absent(self):
        _install_fake_torch_npu(with_hif8=False)
        self.assertFalse(quant_util.hifloat8_supported())

    def test_returns_false_when_probe_cast_raises(self):
        _install_fake_torch_npu(with_hif8=True, cast_raises=True)
        self.assertFalse(quant_util.hifloat8_supported())

    def test_returns_true_when_probe_round_trip_ok(self):
        _install_fake_torch_npu(with_hif8=True, cast_raises=False)
        self.assertTrue(quant_util.hifloat8_supported())


class TestHifloat8FakeQuant(unittest.TestCase):
    """quant_util.hifloat8_fake_quant() native vs amct_ops fallback."""

    def setUp(self):
        self._saved_npu = torch.Tensor.npu
        torch.Tensor.npu = lambda self: self

    def tearDown(self):
        torch.Tensor.npu = self._saved_npu
        for name in ('torch_npu', 'amct_ops', 'amct_ops.hifloat8_cast'):
            sys.modules.pop(name, None)

    @patch.object(quant_util, 'hifloat8_supported', return_value=True)
    def test_uses_native_cast_when_supported(self, _):
        mod = _install_fake_torch_npu(with_hif8=True)
        x = torch.randn(4, 8, dtype=torch.float16)
        out = quant_util.hifloat8_fake_quant(x)
        # native path: returned via torch_npu.npu_dtype_cast (identity stub here)
        self.assertEqual(out.shape, x.shape)
        self.assertTrue(hasattr(mod, 'npu_dtype_cast'))

    @patch.object(quant_util, 'hifloat8_supported', return_value=False)
    def test_falls_back_to_amct_ops_when_unsupported(self, _):
        record = {}
        _install_fake_amct_ops(record)
        x = torch.randn(4, 8, dtype=torch.bfloat16)
        out = quant_util.hifloat8_fake_quant(x)
        self.assertTrue(record.get('encode_called'))
        self.assertTrue(record.get('decode_called'))
        self.assertEqual(out.dtype, torch.bfloat16)
        self.assertEqual(out.shape, x.shape)

    @patch.object(quant_util, 'hifloat8_supported', return_value=False)
    def test_fp32_input_downcast_to_bf16_for_amct_ops(self, _):
        record = {}
        _install_fake_amct_ops(record)
        x = torch.randn(4, 8, dtype=torch.float32)
        out = quant_util.hifloat8_fake_quant(x)
        # amct_ops only supports fp16/bf16: fp32 must be cast to bf16 for the round trip
        self.assertEqual(record.get('encoded_dtype'), torch.bfloat16)
        # ...then restored to the original fp32 dtype on return
        self.assertEqual(out.dtype, torch.float32)

    @patch.object(quant_util, 'hifloat8_supported', return_value=False)
    def test_raises_importerror_when_amct_ops_missing(self, _):
        sys.modules['amct_ops'] = None  # force ImportError on the fallback import
        sys.modules['amct_ops.hifloat8_cast'] = None
        x = torch.randn(2, 4, dtype=torch.bfloat16)
        with self.assertRaises(ImportError):
            quant_util.hifloat8_fake_quant(x)


if __name__ == '__main__':
    unittest.main()
