#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
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
"""Unit tests for the NPU KV-cache deploy quant modules (torch_npu mocked)."""
import sys
import types
import unittest
from unittest.mock import MagicMock

import torch
import torch.nn as nn


def _install_fake_torch_npu():
    """Register a fake torch_npu whose quant ops are identity-ish stubs."""
    mod = types.ModuleType("torch_npu")
    mod.hifloat8 = "hifloat8_enum"
    mod.npu_quantize = lambda t, s, z, dtype=None, div_mode=None: t
    mod.npu_anti_quant = lambda t, s, src_dtype=None, dst_dtype=None: t.to(
        dst_dtype if dst_dtype is not None else t.dtype
    )
    sys.modules["torch_npu"] = mod
    return mod


def _make_quant_module():
    """Build a stand-in quant_module with the attributes the deploy op reads."""
    qm = MagicMock()
    inner = nn.Linear(4, 4)
    qm.ori_module = MagicMock()
    qm.ori_module.q_a_proj = inner
    # ori_module is callable; return a marker so forward() can be asserted.
    qm.ori_module.return_value = "fp_out_marker"
    qm.scale_k = torch.ones(1)
    qm.scale_v = torch.ones(1)
    return qm


class _KvDeployTestMixin(unittest.TestCase):
    """Shared assertions for the two structurally identical deploy modules."""

    __test__ = False  # base mixin: not collected directly, only its subclasses
    module_cls = None

    def setUp(self):
        if self.module_cls is None:
            self.skipTest("base mixin, no module under test")
        self._saved_npu = getattr(torch.Tensor, "npu", None)
        self._saved_torch_npu = sys.modules.get("torch_npu")
        torch.Tensor.npu = lambda self: self
        _install_fake_torch_npu()

    def tearDown(self):
        if getattr(self, "_saved_npu", None) is not None:
            torch.Tensor.npu = self._saved_npu
        elif hasattr(torch.Tensor, "npu"):
            delattr(torch.Tensor, "npu")
        # Restore whatever torch_npu was in sys.modules before this test
        # (e.g. the conftest mock), rather than removing it outright.
        if getattr(self, "_saved_torch_npu", None) is not None:
            sys.modules["torch_npu"] = self._saved_torch_npu
        else:
            sys.modules.pop("torch_npu", None)

    def test_init_reads_scales_and_dtype(self):
        qm = _make_quant_module()
        mod = self.module_cls(qm)
        self.assertIsNotNone(mod.scale_k)
        self.assertIsNotNone(mod.scale_v)
        self.assertIs(mod.ori_module, qm.ori_module)

    def test_forward_creates_cache_when_none_and_delegates(self):
        qm = _make_quant_module()
        mod = self.module_cls(qm)
        out = mod.forward(
            hidden_states=torch.randn(1, 2, 4),
            position_embeddings=None,
            attention_mask=None,
            past_key_values=None,
        )
        self.assertEqual(out, "fp_out_marker")
        # ori_module called with a freshly-created DynamicCache (not None)
        _, kwargs = qm.ori_module.call_args
        self.assertIsNotNone(kwargs["past_key_values"])

    def test_hook_kv_states_installs_hook_and_round_trips(self):
        qm = _make_quant_module()
        mod = self.module_cls(qm)

        captured = {}

        def base_update(k, v, layer_idx, **kwargs):
            captured["k"] = k
            captured["v"] = v
            return k, v

        cache = MagicMock()
        cache.update = base_update
        # start un-hooked
        cache.is_hooked = False

        hook_installer = getattr(mod, "_hook_kv_states")
        hook_installer(cache)
        self.assertTrue(cache.is_hooked)

        # exercise the wrapped update: quantize -> update -> anti_quant
        key = torch.randn(1, 2, 4)
        val = torch.randn(1, 2, 4)
        dq_k, dq_v = cache.update(key, val, 0)
        self.assertEqual(dq_k.shape, key.shape)
        self.assertEqual(dq_v.shape, val.shape)
        # base_update saw the quantized states (identity stub keeps shape)
        self.assertIn("k", captured)

    def test_hook_kv_states_noop_when_already_hooked(self):
        qm = _make_quant_module()
        mod = self.module_cls(qm)
        cache = MagicMock()
        original_update = cache.update
        cache.is_hooked = True
        hook_installer = getattr(mod, "_hook_kv_states")
        hook_installer(cache)
        # update should be unchanged (early return)
        self.assertIs(cache.update, original_update)


class TestDeepseekV3AttentionDeploy(_KvDeployTestMixin):
    __test__ = True

    @classmethod
    def setUpClass(cls):
        _install_fake_torch_npu()
        from amct_pytorch.classic.deploy_op.npu_quantization_deepseekv3_attention import (
            NpuDeepseekV3AttentionQuant,
        )

        cls.module_cls = NpuDeepseekV3AttentionQuant


class TestLongcatFlashMLADeploy(_KvDeployTestMixin):
    __test__ = True

    @classmethod
    def setUpClass(cls):
        _install_fake_torch_npu()
        from amct_pytorch.classic.deploy_op.npu_quantization_longcat_flashmla import (
            NpuLongcatFlashMLA,
        )

        cls.module_cls = NpuLongcatFlashMLA


if __name__ == "__main__":
    unittest.main()
