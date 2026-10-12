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
"""Unit tests for OfmrQuant and LinearAWQuant modules (CPU, no NPU required)."""

import unittest
from unittest.mock import patch

import torch
import torch.nn as nn

from amct_pytorch.classic.quantize_op.ofmr_quant_module import OfmrQuant
from amct_pytorch.classic.quantize_op.gptq_module import GPTQuant
from amct_pytorch.classic.quantize_op.linear_awq_module import LinearAWQuant
from amct_pytorch.classic.quantize_op.minmax_module import MinMaxQuant
from amct_pytorch.classic.quantize_op.smooth_quant_module import SmoothQuant
from amct_pytorch.classic.quantize_op.quantile_module import QuantileQuant
from amct_pytorch.classic.quantize_op.utils import calculate_hifloat8_weight_scale
from amct_pytorch.common.utils.quant_util import quant_dequant_tensor
from amct_pytorch.common.utils.vars import HIFLOAT8, INT8
from amct_pytorch.quantize_op import BaseQuantizeModule
from amct_pytorch.classic.quantize_op import QuantResultSupportMixin


def _make_ofmr_config(batch_num=2, strategy="tensor", weight_compress_only=False):
    return {
        "batch_num": batch_num,
        "weights_cfg": {"quant_type": INT8, "strategy": strategy},
        "inputs_cfg": {
            "quant_type": INT8,
            "enable_quant": not weight_compress_only,
        },
    }


def _make_awq_config(group_size=None):
    cfg = {
        "weights_cfg": {
            "quant_type": INT8,
            "symmetric": True,
            "group_size": group_size,
        },
        "inputs_cfg": {"enable_quant": True, "strategy": "tensor"},
    }
    return cfg


def _make_linear(in_f=8, out_f=4, bias=True):
    return nn.Linear(in_f, out_f, bias=bias)


class TestOfmrQuantInit(unittest.TestCase):
    """Test OfmrQuant __init__ paths."""

    def test_init_per_tensor_weight(self):
        mod = _make_linear()
        cfg = _make_ofmr_config(strategy="tensor")
        q = OfmrQuant(mod, "fc", cfg)
        self.assertEqual(q.cout, 1)
        self.assertEqual(q.batch_num, 2)

    def test_init_per_channel_weight(self):
        mod = _make_linear(8, 4)
        cfg = _make_ofmr_config(strategy="channel")
        q = OfmrQuant(mod, "fc", cfg)
        self.assertEqual(q.cout, mod.weight.shape[0])

    def test_init_weight_compress_only(self):
        mod = _make_linear()
        cfg = _make_ofmr_config(weight_compress_only=True)
        q = OfmrQuant(mod, "fc", cfg)
        self.assertTrue(q.weight_compress_only)


class TestSmoothQuantCalculateSmooth(unittest.TestCase):
    """Test SmoothQuant smooth factor sanitization."""

    def test_calculate_smooth_replaces_non_finite_factor_with_one(self):
        quant_module = object.__new__(SmoothQuant)
        quant_module.weight = torch.zeros((2, 2), dtype=torch.float32)
        quant_module.quant_config = {
            "algorithm": {"smoothquant": {"smooth_strength": 0.5}}
        }

        smooth_factor = quant_module.calculate_smooth(
            torch.ones((1, 2), dtype=torch.float32)
        )

        self.assertTrue(torch.equal(smooth_factor, torch.ones_like(smooth_factor)))


class TestOfmrQuantForward(unittest.TestCase):
    """Test OfmrQuant forward and calibration paths."""

    def setUp(self):
        self.mod = nn.Linear(8, 4, bias=False)
        self.cfg = _make_ofmr_config(batch_num=2, strategy="tensor")
        self.q = OfmrQuant(self.mod, "fc", self.cfg)

    def test_forward_calibration_batches(self):
        x = torch.randn(2, 8)
        out1 = self.q(x)
        self.assertEqual(out1.shape, (2, 4))
        out2 = self.q(x)
        self.assertEqual(out2.shape, (2, 4))

    def test_forward_after_calibration_uses_fake_quant(self):
        x = torch.randn(2, 8)
        self.q(x)
        self.q(x)
        # after batch_num batches, scale_w is set; next call uses fake_quant_forward
        out = self.q(x)
        self.assertEqual(out.shape, (2, 4))

    def test_compute_mse_no_reduce(self):
        a = torch.ones(4)
        b = torch.zeros(4)
        loss = OfmrQuant.compute_mse(a, b)
        self.assertAlmostEqual(loss.item(), 1.0, places=5)

    def test_compute_mse_with_reduce_axis(self):
        a = torch.ones(2, 3, 4)
        b = torch.zeros(2, 3, 4)
        loss = OfmrQuant.compute_mse(a, b, reduce_axis=(0, 2))
        self.assertEqual(loss.shape, (3,))

    def test_raises_on_invalid_linear_input_dim(self):
        x = torch.randn(2)  # 1D input — invalid for Linear
        with self.assertRaises(RuntimeError):
            self.q(x)

    def test_weight_compress_only_skips_act_loss(self):
        cfg = _make_ofmr_config(batch_num=1, weight_compress_only=True)
        q = OfmrQuant(self.mod, "fc", cfg)
        x = torch.randn(2, 8)
        out = q(x)
        self.assertEqual(out.shape, (2, 4))
        # scale_d should remain None (no act quant)
        self.assertIsNone(q.scale_d)


class TestOfmrQuantFakeQuantForward(unittest.TestCase):
    """Test fake_quant_forward caching."""

    def test_linear_fake_quant_cached(self):
        mod = nn.Linear(4, 2, bias=True)
        cfg = _make_ofmr_config(batch_num=1)
        q = OfmrQuant(mod, "fc", cfg)
        x = torch.randn(2, 4)
        q(x)  # calibration batch
        # directly exercise fake_quant_forward
        q.scale_w = torch.tensor([1.0])
        q.offset_w = None
        q.scale_d = None
        out = q.fake_quant_forward(x)
        self.assertEqual(out.shape, (2, 2))
        # second call hits cache
        out2 = q.fake_quant_forward(x)
        self.assertEqual(out2.shape, (2, 2))

    def test_conv2d_fake_quant_path(self):
        """Exercise the Conv2d branch in fake_quant_forward (line 137)."""
        mod = nn.Conv2d(2, 4, kernel_size=3, padding=1, bias=True)
        cfg = _make_ofmr_config(batch_num=1, strategy="tensor")
        q = OfmrQuant(mod, "conv", cfg)
        x = torch.randn(1, 2, 8, 8)
        q(x)  # calibration
        q.scale_w = torch.ones(1)
        q.offset_w = None
        q.scale_d = None
        out = q.fake_quant_forward(x)
        self.assertEqual(out.shape, (1, 4, 8, 8))

    def test_conv2d_calibration_with_act_quant(self):
        """Exercise Conv2d branches in _calc_weight/act_quant_loss (lines 202,211,214,240)."""
        mod = nn.Conv2d(2, 4, kernel_size=3, padding=1, bias=False)
        cfg = _make_ofmr_config(
            batch_num=2, strategy="tensor", weight_compress_only=False
        )
        q = OfmrQuant(mod, "conv", cfg)
        x = torch.randn(1, 2, 8, 8)
        q(x)
        q(x)  # both batches done → scale_w and scale_d computed
        self.assertIsNotNone(q.scale_w)
        self.assertIsNotNone(q.scale_d)

    def test_conv2d_per_channel_calibration_cout_axis(self):
        """Per-channel Conv2d hits the cout>1 axis-reduction path (lines 214-218)."""
        mod = nn.Conv2d(2, 4, kernel_size=3, padding=1, bias=False)
        cfg = _make_ofmr_config(
            batch_num=1, strategy="channel", weight_compress_only=True
        )
        q = OfmrQuant(mod, "conv", cfg)
        self.assertEqual(q.cout, 4)
        x = torch.randn(1, 2, 8, 8)
        q(x)  # single batch → _calc_weight_quant_loss with cout>1 → scale_w computed
        self.assertIsNotNone(q.scale_w)

    def test_conv2d_per_channel_fake_quant_uses_output_channel_scales(self):
        mod = nn.Conv2d(2, 4, kernel_size=3, padding=1, bias=False)
        cfg = _make_ofmr_config(
            batch_num=1, strategy="channel", weight_compress_only=True
        )
        q = OfmrQuant(mod, "conv", cfg)
        x = torch.randn(1, 2, 8, 8)
        q(x)  # calibration
        out = q(x)  # post-calibration fake-quant forward

        expected_weight = quant_dequant_tensor(
            q.weight.data, INT8, q.scale_w.reshape(-1, 1, 1, 1)
        )
        self.assertEqual(out.shape, (1, 4, 8, 8))
        self.assertTrue(torch.equal(q.cached_dq_w, expected_weight))


class TestLinearAWQuantInit(unittest.TestCase):
    """Test LinearAWQuant __init__ paths."""

    def test_init_with_group_size(self):
        mod = _make_linear(bias=False)
        cfg = _make_awq_config(group_size=4)
        q = LinearAWQuant(mod, "fc", cfg)
        self.assertEqual(q.group_size, 4)

    def test_init_without_group_size(self):
        mod = _make_linear(bias=False)
        cfg = _make_awq_config(group_size=None)
        q = LinearAWQuant(mod, "fc", cfg)
        self.assertFalse(hasattr(q, "group_size") and q.group_size is not None)

    def test_init_act_quant_disabled(self):
        mod = _make_linear(bias=False)
        cfg = {
            "weights_cfg": {"quant_type": INT8, "symmetric": True, "group_size": None},
            "inputs_cfg": {"enable_quant": False, "strategy": "tensor"},
        }
        q = LinearAWQuant(mod, "fc", cfg)
        self.assertFalse(hasattr(q, "act_granularity"))


class TestLinearAWQuantForward(unittest.TestCase):
    """Test LinearAWQuant forward and fake_quant_forward."""

    def setUp(self):
        self.mod = nn.Linear(8, 4, bias=False)
        self.cfg = _make_awq_config()

    @patch("amct_pytorch.classic.quantize_op.linear_awq_module.search_scale")
    @patch("amct_pytorch.classic.quantize_op.linear_awq_module.apply_scale")
    @patch(
        "amct_pytorch.classic.quantize_op.linear_awq_module.calculate_scale_offset_by_granularity"
    )
    def test_forward_first_pass(self, mock_calc, mock_apply, mock_search):
        mock_search.return_value = torch.ones(8)
        mock_calc.return_value = (torch.ones(4, 1), torch.zeros(4, 1))
        q = LinearAWQuant(self.mod, "fc", self.cfg)
        x = torch.randn(2, 8)
        out = q(x)
        self.assertEqual(out.shape, (2, 4))
        self.assertTrue(q.calc_done)

    @patch("amct_pytorch.classic.quantize_op.linear_awq_module.search_scale")
    @patch("amct_pytorch.classic.quantize_op.linear_awq_module.apply_scale")
    @patch(
        "amct_pytorch.classic.quantize_op.linear_awq_module.calculate_scale_offset_by_granularity"
    )
    def test_forward_second_pass_uses_fake_quant(
        self, mock_calc, mock_apply, mock_search
    ):
        mock_search.return_value = torch.ones(8)
        mock_calc.return_value = (torch.ones(4, 1), torch.zeros(4, 1))
        q = LinearAWQuant(self.mod, "fc", self.cfg)
        x = torch.randn(2, 8)
        q(x)  # first pass sets calc_done=True
        out = q(x)  # second pass uses fake_quant_forward
        self.assertEqual(out.shape, (2, 4))

    @patch("amct_pytorch.classic.quantize_op.linear_awq_module.search_scale")
    @patch("amct_pytorch.classic.quantize_op.linear_awq_module.apply_scale")
    @patch(
        "amct_pytorch.classic.quantize_op.linear_awq_module.calculate_scale_offset_by_granularity"
    )
    def test_fake_quant_forward_cached(self, mock_calc, mock_apply, mock_search):
        mock_search.return_value = torch.ones(8)
        mock_calc.return_value = (torch.ones(4, 1), torch.zeros(4, 1))
        q = LinearAWQuant(self.mod, "fc", self.cfg)
        x = torch.randn(2, 8)
        q(x)
        # call fake_quant_forward directly twice to test cache path
        out1 = q.fake_quant_forward(x)
        out2 = q.fake_quant_forward(x)  # hits cache
        self.assertEqual(out1.shape, out2.shape)


def _make_hif8_weight_only_config(strategy="channel"):
    """Weight-only HIF8 config consumed by QuantileQuant.__init__ (no calibration)."""
    return {
        "batch_num": 1,
        "inputs_cfg": {"enable_quant": False},
        "weights_cfg": {"quant_type": HIFLOAT8, "strategy": strategy},
    }


def _make_linear_with_weight(weight):
    out_f, in_f = weight.shape
    mod = nn.Linear(in_f, out_f, bias=False)
    mod.weight = nn.Parameter(weight.clone())
    return mod


class TestHifloat8WeightScaleAbsMax(unittest.TestCase):
    """HIF8 symmetric weight scale must cover the negative absolute extreme."""

    def test_cast_channel_negative_dominant_row(self):
        # Row 0 has its largest magnitude on the negative side: scale is 6.25.
        weight = torch.tensor([[-100.0, 1.0], [1.0, 100.0]])
        scale_w = calculate_hifloat8_weight_scale(weight, "channel")
        self.assertEqual(scale_w.tolist(), [6.25, 6.25])

    def test_cast_tensor_negative_dominant(self):
        # tensor granularity: the global positive max is 1 while abs max is 100.
        weight = torch.tensor([[-100.0, -1.0], [-2.0, -3.0]])
        scale_w = calculate_hifloat8_weight_scale(weight, "tensor")
        self.assertEqual(scale_w.tolist(), [6.25, 6.25])

    def test_cast_positive_dominant(self):
        # Positive-dominant groups keep max(abs(weight)) / 16.
        weight = torch.tensor([[100.0, -1.0], [1.0, 64.0]])
        scale_w = calculate_hifloat8_weight_scale(weight, "channel")
        self.assertEqual(scale_w.tolist(), [6.25, 4.0])

    def test_cast_all_negative_group_positive_scale(self):
        # An all-negative group still yields a positive per-channel scale.
        weight = torch.tensor([[-0.5, -0.1], [-8.0, -2.0], [1.0, 2.0]])
        scale_w = calculate_hifloat8_weight_scale(weight, "channel")
        self.assertEqual(scale_w.tolist(), [0.03125, 0.5, 0.125])
        self.assertNotIn(1.0, scale_w.tolist())

    def test_quantile_channel_negative_dominant_row(self):
        weight = torch.tensor([[-100.0, 1.0], [1.0, 100.0]])
        mod = _make_linear_with_weight(weight)
        q = QuantileQuant(mod, "fc", _make_hif8_weight_only_config("channel"))
        self.assertEqual(q.scale_w.flatten().tolist(), [6.25, 6.25])

    def test_quantile_tensor_negative_dominant(self):
        weight = torch.tensor([[-100.0, -1.0], [-2.0, -3.0]])
        mod = _make_linear_with_weight(weight)
        q = QuantileQuant(mod, "fc", _make_hif8_weight_only_config("tensor"))
        self.assertEqual(q.scale_w.flatten().tolist(), [6.25])


class _StubWriter:
    """Minimal QuantResultWriter stand-in recording write_unit calls.

    Mirrors the real keyword-only writer signature, so a drift in the call site
    fails here rather than silently passing a wrong field name through.
    """

    def __init__(self):
        self.calls = []

    def write_unit(
        self,
        layer_name,
        tensors,
        non_tensor_params=None,
        quant_config=None,
        quant_module_type=None,
        ori_module_type=None,
    ):
        self.calls.append(
            {
                'layer_name': layer_name,
                'tensors': tensors,
                'non_tensor_params': non_tensor_params,
                'quant_config': quant_config,
                'quant_module_type': quant_module_type,
                'ori_module_type': ori_module_type,
            }
        )
        return {'layer_name': layer_name}


class _DeclaredParamsModule(QuantResultSupportMixin, BaseQuantizeModule):
    """Exercises the generic export/load plumbing with mixed value kinds."""

    _quant_param_keys = ('scale_w', 'offset_w', 'group_size', 'scale_d')

    def __init__(self, layer_name='fc', quant_config=None):
        super().__init__(None, layer_name, quant_config or {})
        self.weight = nn.Parameter(torch.zeros(4, 8))
        self.ori_module_type = 'Linear'
        self.scale_w = torch.ones(4, 1)
        self.offset_w = None
        self.group_size = 64
        self.loaded_hook_calls = 0

    def _on_quant_params_loaded(self):
        self.loaded_hook_calls += 1


class TestBaseQuantParamExport(unittest.TestCase):
    """export_quant_params splits declared params by kind and skips None."""

    def test_splits_tensor_and_non_tensor_and_drops_none(self):
        module = _DeclaredParamsModule()
        tensors, metadata = module.export_quant_params()
        self.assertEqual(sorted(tensors), ['scale_w'])
        self.assertEqual(metadata, {'group_size': 64})
        # offset_w and scale_d are None, so they appear in neither payload.
        self.assertNotIn('offset_w', tensors)
        self.assertNotIn('scale_d', metadata)

    def test_exported_tensors_are_cpu_contiguous_and_detached(self):
        module = _DeclaredParamsModule()
        source = torch.zeros(4, 6, requires_grad=True)
        module.scale_w = source.transpose(0, 1)
        self.assertFalse(module.scale_w.is_contiguous())
        tensors, _ = module.export_quant_params()
        exported = tensors['scale_w']
        self.assertTrue(exported.is_contiguous())
        self.assertFalse(exported.requires_grad)
        self.assertIsNone(exported.grad_fn)
        self.assertEqual(exported.device.type, 'cpu')

    def test_ignores_undeclared_attributes(self):
        module = _DeclaredParamsModule()
        module.scale = torch.ones(3)  # set by the base ctor, not declared
        tensors, metadata = module.export_quant_params()
        self.assertNotIn('scale', tensors)
        self.assertNotIn('scale', metadata)

    def test_rejects_malformed_key_declaration(self):
        class _BadKeys(QuantResultSupportMixin, BaseQuantizeModule):
            _quant_param_keys = ('scale_w', 'scale_w')

        with self.assertRaises(ValueError) as duplicate_ctx:
            _BadKeys.quant_param_keys()
        self.assertNotIn('_quant_param_keys', str(duplicate_ctx.exception))

        class _NonStrKeys(QuantResultSupportMixin, BaseQuantizeModule):
            _quant_param_keys = (1,)

        with self.assertRaises(TypeError) as type_ctx:
            _NonStrKeys.quant_param_keys()
        self.assertNotIn('_quant_param_keys', str(type_ctx.exception))


class TestBaseQuantParamLoad(unittest.TestCase):
    """load_quant_params validates keys and preserves device/dtype/shape."""

    def test_round_trip_restores_values(self):
        module = _DeclaredParamsModule()
        module.scale_w = torch.tensor([[2.0], [3.0], [4.0], [5.0]])
        module.group_size = 32
        tensors, metadata = module.export_quant_params()

        target = _DeclaredParamsModule()
        target.load_quant_params(tensors, metadata)
        self.assertTrue(torch.equal(target.scale_w, module.scale_w))
        self.assertEqual(target.group_size, 32)
        self.assertEqual(target.loaded_hook_calls, 1)

    def test_copies_in_place_keeping_the_existing_tensor_object(self):
        module = _DeclaredParamsModule()
        original = module.scale_w
        module.load_quant_params({'scale_w': torch.full((4, 1), 7.0)})
        self.assertIs(module.scale_w, original)
        self.assertTrue(torch.equal(module.scale_w, torch.full((4, 1), 7.0)))

    def test_none_attribute_adopts_persisted_dtype_and_weight_device(self):
        module = _DeclaredParamsModule()
        persisted = torch.tensor([1.5], dtype=torch.float64)
        module.load_quant_params({'scale_d': persisted})
        self.assertEqual(module.scale_d.dtype, torch.float64)
        self.assertEqual(module.scale_d.device, module.weight.device)

    def test_rejects_undeclared_key(self):
        module = _DeclaredParamsModule()
        with self.assertRaises(ValueError) as ctx:
            module.load_quant_params({'scale_bogus': torch.ones(1)})
        self.assertIn('not declared', str(ctx.exception))
        self.assertNotIn('_quant_param_keys', str(ctx.exception))
        with self.assertRaises(ValueError):
            module.load_quant_params({}, {'not_declared': 1})

    def test_rejects_key_in_both_payloads(self):
        module = _DeclaredParamsModule()
        with self.assertRaises(ValueError) as ctx:
            module.load_quant_params({'scale_w': torch.ones(4, 1)}, {'scale_w': 1})
        self.assertIn('both tensor and non-tensor', str(ctx.exception))

    def test_rejects_shape_mismatch(self):
        module = _DeclaredParamsModule()
        with self.assertRaises(ValueError) as ctx:
            module.load_quant_params({'scale_w': torch.ones(9, 1)})
        self.assertIn('shape', str(ctx.exception))

    def test_rejects_dtype_mismatch(self):
        module = _DeclaredParamsModule()
        with self.assertRaises(ValueError) as ctx:
            module.load_quant_params({'scale_w': torch.ones(4, 1, dtype=torch.float64)})
        self.assertIn('dtype', str(ctx.exception))

    def test_rejects_non_tensor_in_tensor_payload(self):
        module = _DeclaredParamsModule()
        with self.assertRaises(TypeError):
            module.load_quant_params({'scale_w': [1.0]})


class TestBaseQuantResultDump(unittest.TestCase):
    """maybe_dump_quant_params dumps once, and only with a writer attached."""

    def test_no_writer_means_no_dump(self):
        module = _DeclaredParamsModule()
        self.assertFalse(module.maybe_dump_quant_params())
        self.assertFalse(hasattr(module, 'quant_result_writer'))
        self.assertFalse(hasattr(module, 'quant_params_dumped'))

    def test_dumps_once_with_full_provenance(self):
        module = _DeclaredParamsModule(
            layer_name='block.0.fc1',
            quant_config={'batch_num': 1, 'algorithm': {'minmax': {}}},
        )
        writer = _StubWriter()
        module.attach_quant_result_writer(writer)

        self.assertTrue(module.maybe_dump_quant_params())
        self.assertFalse(module.maybe_dump_quant_params())
        self.assertEqual(len(writer.calls), 1)
        call = writer.calls[0]
        self.assertEqual(call['layer_name'], 'block.0.fc1')
        self.assertEqual(sorted(call['tensors']), ['scale_w'])
        self.assertEqual(call['non_tensor_params'], {'group_size': 64})
        self.assertEqual(call['quant_module_type'], '_DeclaredParamsModule')
        self.assertEqual(call['ori_module_type'], 'Linear')
        self.assertEqual(call['quant_config']['batch_num'], 1)
        # The algorithm is inside quant_config, not a separate field.
        self.assertEqual(call['quant_config']['algorithm'], {'minmax': {}})

    def test_failed_write_leaves_the_unit_pending(self):
        class _FailingWriter(_StubWriter):
            def write_unit(self, *args, **kwargs):
                raise OSError('disk full')

        module = _DeclaredParamsModule()
        module.attach_quant_result_writer(_FailingWriter())
        with self.assertRaises(OSError):
            module.maybe_dump_quant_params()
        retry_writer = _StubWriter()
        module.attach_quant_result_writer(retry_writer)
        self.assertTrue(module.maybe_dump_quant_params())
        self.assertEqual(len(retry_writer.calls), 1)

    def test_rejects_invalid_layer_name_during_construction(self):
        for layer_name in ('', None, 1):
            with self.subTest(layer_name=layer_name):
                with self.assertRaisesRegex(ValueError, 'layer_name'):
                    _DeclaredParamsModule(layer_name=layer_name)

    def test_rejects_a_quant_config_reload_could_not_use(self):
        module = _DeclaredParamsModule(quant_config={'mod': nn.Linear(2, 2)})
        module.attach_quant_result_writer(_StubWriter())
        with self.assertRaises(TypeError):
            module.maybe_dump_quant_params()


class TestQuantResultParamDeclaration(unittest.TestCase):
    """Each wired algorithm declares the params its implementation produces."""

    def test_wired_algorithms_declare_params(self):
        for quant_cls in (MinMaxQuant, LinearAWQuant, GPTQuant):
            self.assertTrue(quant_cls.quant_param_keys())

    def test_gptq_persists_its_optimized_weight(self):
        # Unlike AWQ, GPTQ rewrites the weight, so the weight must be declared:
        # it cannot be recomputed from the original weight and the scales.
        self.assertIn('weight', GPTQuant.quant_param_keys())
        self.assertNotIn('weight', LinearAWQuant.quant_param_keys())
        self.assertIn('scale', LinearAWQuant.quant_param_keys())

    def test_algorithms_not_yet_wired_do_not_expose_artifact_methods(self):
        for quant_cls in (SmoothQuant, OfmrQuant):
            self.assertFalse(hasattr(quant_cls, 'quant_param_keys'))


def _make_minmax_config(batch_num=2, quant_act=True):
    cfg = {
        'batch_num': batch_num,
        'weights_cfg': {
            'quant_type': INT8,
            'symmetric': False,
            'strategy': 'channel',
            'group_size': None,
        },
        'inputs_cfg': {'enable_quant': False},
        'algorithm': {'minmax': {}},
    }
    if quant_act:
        cfg['inputs_cfg'] = {
            'enable_quant': True,
            'quant_type': INT8,
            'symmetric': False,
            'strategy': 'tensor',
        }
    return cfg


class TestMinMaxQuantResultDump(unittest.TestCase):
    """MinMax has two distinct finalization points; each dumps exactly once."""

    def test_weight_only_dumps_at_writer_attachment(self):
        module = MinMaxQuant(_make_linear(), 'fc', _make_minmax_config(quant_act=False))
        writer = _StubWriter()
        module.attach_quant_result_writer(writer)
        # scale_w/offset_w are already final after __init__: no forward needed.
        self.assertEqual(len(writer.calls), 1)
        self.assertEqual(sorted(writer.calls[0]['tensors']), ['offset_w', 'scale_w'])
        self.assertEqual(writer.calls[0]['quant_config']['algorithm'], {'minmax': {}})

    def test_activation_waits_for_the_last_calibration_batch(self):
        module = MinMaxQuant(_make_linear(), 'fc', _make_minmax_config(batch_num=2))
        writer = _StubWriter()
        module.attach_quant_result_writer(writer)
        self.assertEqual(writer.calls, [])

        inputs = torch.randn(3, 8)
        module(inputs)
        self.assertEqual(writer.calls, [], 'must not dump before batch_num')
        module(inputs)
        self.assertEqual(len(writer.calls), 1)
        self.assertEqual(
            sorted(writer.calls[0]['tensors']),
            ['offset_d', 'offset_w', 'scale_d', 'scale_w'],
        )

        module(inputs)
        self.assertEqual(len(writer.calls), 1, 'must dump only once')

    def test_declares_every_minmax_quant_param(self):
        self.assertEqual(
            MinMaxQuant.quant_param_keys(),
            ('scale_w', 'offset_w', 'scale_d', 'offset_d', 'scale_w1', 'scale_w2'),
        )

    def test_exports_the_progressive_scale_pair_instead_of_scale_w(self):
        # The fp8-act/fp4-weight path fills scale_w1/scale_w2 and leaves
        # scale_w unset. Building it for real needs an NPU kernel, so drive the
        # export contract directly: declared-but-None params stay absent.
        module = MinMaxQuant(_make_linear(), 'fc', _make_minmax_config(quant_act=False))
        module.scale_w = None
        module.offset_w = None
        module.scale_w1 = torch.ones(4, 1)
        module.scale_w2 = torch.full((4, 1), 0.5)
        tensors, _ = module.export_quant_params()
        self.assertEqual(sorted(tensors), ['scale_w1', 'scale_w2'])

        target = MinMaxQuant(_make_linear(), 'fc', _make_minmax_config(quant_act=False))
        target.scale_w = None
        target.load_quant_params(tensors)
        self.assertTrue(torch.equal(target.scale_w1, module.scale_w1))
        self.assertTrue(torch.equal(target.scale_w2, module.scale_w2))
        self.assertIsNone(target.scale_w)

    def test_records_the_full_layer_config_not_just_the_algo_subconfig(self):
        config = _make_minmax_config(quant_act=False)
        module = MinMaxQuant(_make_linear(), 'fc', config)
        writer = _StubWriter()
        module.attach_quant_result_writer(writer)
        recorded = writer.calls[0]['quant_config']
        self.assertEqual(recorded['batch_num'], config['batch_num'])
        self.assertIn('weights_cfg', recorded)
        self.assertIn('inputs_cfg', recorded)
        self.assertIn('algorithm', recorded)


class TestMinMaxQuantParamRoundTrip(unittest.TestCase):
    """A loaded module must fake-quant identically, without recalibrating."""

    def _calibrated_module(self, batch_num=2, quant_act=True):
        module = MinMaxQuant(
            _make_linear(), 'fc', _make_minmax_config(batch_num, quant_act)
        )
        inputs = torch.randn(3, 8)
        for _ in range(batch_num):
            module(inputs)
        return module, inputs

    def test_round_trip_preserves_scale_offset_and_output(self):
        source, inputs = self._calibrated_module()
        expected = source(inputs)
        tensors, metadata = source.export_quant_params()

        target = MinMaxQuant(_make_linear(), 'fc', _make_minmax_config())
        target.weight = source.weight
        target.bias = source.bias
        target.ori_module = source.ori_module
        target.load_quant_params(tensors, metadata)

        self.assertTrue(torch.equal(target.scale_w, source.scale_w))
        self.assertTrue(torch.equal(target.offset_w, source.offset_w))
        self.assertTrue(torch.equal(target.scale_d, source.scale_d))
        self.assertTrue(torch.equal(target.offset_d, source.offset_d))
        self.assertTrue(torch.allclose(target(inputs), expected))

    def test_load_marks_calibration_complete_and_drops_caches(self):
        source, inputs = self._calibrated_module()
        source(inputs)  # populate the fake-quant cache
        self.assertTrue(source.fake_quant_cache_ready)

        target = MinMaxQuant(_make_linear(), 'fc', _make_minmax_config())
        target.weight = source.weight
        target.bias = source.bias
        target.ori_module = source.ori_module
        target(inputs)  # one calibration batch, then discard it via load
        tensors, metadata = source.export_quant_params()
        target.load_quant_params(tensors, metadata)

        self.assertFalse(target.fake_quant_cache_ready)
        self.assertIsNone(target.cached_dq_w)
        self.assertIsNone(target.cached_bias)
        self.assertEqual(target.cur_batch, target.batch_num)
        # The next forward must fake-quant rather than re-collect min/max.
        stale_max = target.data_max.clone()
        target(inputs * 100)
        self.assertTrue(torch.equal(target.data_max, stale_max))

    def test_weight_only_round_trip(self):
        source = MinMaxQuant(_make_linear(), 'fc', _make_minmax_config(quant_act=False))
        inputs = torch.randn(3, 8)
        expected = source(inputs)
        tensors, metadata = source.export_quant_params()
        self.assertNotIn('scale_d', tensors)

        target = MinMaxQuant(_make_linear(), 'fc', _make_minmax_config(quant_act=False))
        target.weight = source.weight
        target.bias = source.bias
        target.ori_module = source.ori_module
        target.load_quant_params(tensors, metadata)
        self.assertIsNone(target.scale_d)
        self.assertTrue(torch.allclose(target(inputs), expected))


if __name__ == "__main__":
    unittest.main()
