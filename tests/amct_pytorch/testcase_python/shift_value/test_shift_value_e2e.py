#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
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
"""End to end tests of the shift value feature (graph mode PTQ full flow).

Scenarios:
- S1 auto + skip_layers: skipped layer fp16_fp16 (searched, SSE argmin),
  quantized layer untouched, fc gets a really searched
  fixed_shift_value from the fp16fixed kernel;
- S2 auto + weight-only (arq_quantize{}, no acts -- the activation is
  carried unquantized): fp16_int8 does NOT search (the board kernel
  pins sv=13 and ignores the passed value), conv gets the pinned
  regime default 13, the record carries no activation factors and the
  deploy graph feeds the raw input to the int8-weight operator;
- S2b auto + act num_bits=16: the activation IS quantized, the layer
  never joins the sv calibration;
- S3 off: regression, no wrapper and no attribute;
- S4 fp16fixed kernel monkeypatched: the fp16_fp16 argmin selection
  takes effect;
- S5 auto + INT6 weight-only: the fp16_int8 entry requires an 8-bit
  weight, an INT6 layer never joins the sv calibration -- no
  fixed_shift_value in the record, no sv attribute in the deploy graph;
- S6 auto + AdaRound weight-only ConvTranspose2d: the rebuilt int8
  weight is bit-exact with the deploy ONNX weight (the weight-only
  deploy premise, guarded end to end);
- S7 all: every target module is forced onto the FP16 non-quantized
  path and searched, the Linear exports as Gemm;
- S8 name mapping: after BN fusion the fused Conv keeps the module
  name; a Linear applied to a 3D input exports as MatMul + Add and the
  MatMul node still carries the module name;
- S13/S14/S15/S16: shift-layer-first calibration order, Conv1d
  weight-only rejection, the enable_shift_value entry gate and the
  three-way (weight-only + skipped fp16_fp16 + fully quantized) record
  flush regression.
"""

import os
import re
import shutil
import unittest

import pytest
from unittest.mock import patch

import onnx
import torch

import amct_pytorch
from amct_pytorch.classic.graph_based.amct_pytorch.nn.module.quantization import (
    shift_value_kernel,
)
from amct_pytorch.classic.graph_based.amct_pytorch.nn.module.quantization.shift_calibration import (
    ShiftCalibrationModule,
)

pytestmark = pytest.mark.cpu


CUR_DIR = os.path.split(os.path.realpath(__file__))[0]


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, padding=1)
        self.fc = torch.nn.Linear(8 * 8 * 8, 4)

    def forward(self, x):
        return self.fc(torch.flatten(self.conv(x), 1))


class TinyDeconvModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.deconv = torch.nn.ConvTranspose2d(3, 8, 3, padding=1)
        self.fc = torch.nn.Linear(8 * 8 * 8, 4)

    def forward(self, x):
        return self.fc(torch.flatten(self.deconv(x), 1))


class TinyConvBnModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, padding=1)
        self.bn = torch.nn.BatchNorm2d(8)
        self.fc = torch.nn.Linear(8 * 8 * 8, 4)

    def forward(self, x):
        return self.fc(torch.flatten(self.bn(self.conv(x)), 1))


class TinyLinear3dModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, padding=1)
        self.fc = torch.nn.Linear(8, 4)

    def forward(self, x):
        # the Linear runs on a 3D input: it exports as MatMul + Add
        y = self.fc(self.conv(x).permute(0, 2, 3, 1))
        return torch.flatten(y, 1)


class TinyConv1dModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1d = torch.nn.Conv1d(3, 8, 3, padding=1)

    def forward(self, x):
        return self.conv1d(x)


class TinyTwoConvModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 8, 3, padding=1)
        self.fc = torch.nn.Linear(8 * 8 * 8, 4)

    def forward(self, x):
        return self.fc(torch.flatten(self.conv2(self.conv1(x)), 1))


def _deploy_attrs(onnx_file):
    attrs = {}
    model = onnx.load(onnx_file)
    for node in model.graph.node:
        for attr in node.attribute:
            if attr.name == 'fixed_shift_value':
                attrs[node.name] = attr.i
    return attrs


def _deploy_int8_weight(onnx_file, node_name):
    """The int8 weight initializer of the deploy node (torch layout)."""
    model = onnx.load(onnx_file)
    weight_name = None
    for node in model.graph.node:
        if node.name == node_name:
            weight_name = node.input[1]
            break
    if weight_name is None:
        return None
    for initializer in model.graph.initializer:
        if initializer.name == weight_name:
            return torch.from_numpy(onnx.numpy_helper.to_array(initializer))
    return None


def _sv_deploy_topology(onnx_file, node_name):
    """The sv deploy-node topology check inputs as a dict:
    input_op  -- the producer op type of the node's first input
    output_op -- the consumer op type of the node's first output"""
    model = onnx.load(onnx_file)
    producers = {out: n.op_type for n in model.graph.node for out in n.output}
    consumers = {}
    for n in model.graph.node:
        for inp in n.input:
            consumers.setdefault(inp, []).append(n.op_type)
    for node in model.graph.node:
        if node.name == node_name:
            input_op = producers.get(node.input[0])
            output_op = consumers.get(node.output[0], [None])[0]
            return {'input_op': input_op, 'output_op': output_op}
    return None


class TestShiftValueEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_folder = os.path.join(CUR_DIR, 'test_shift_value_e2e')
        if not os.path.isdir(cls.temp_folder):
            os.makedirs(cls.temp_folder)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_folder, ignore_errors=True)

    def _run_flow(self, tag, simple_cfg_text, model=None):
        work_dir = os.path.join(self.temp_folder, tag)
        if not os.path.isdir(work_dir):
            os.makedirs(work_dir)

        model = TinyModel().eval() if model is None else model.eval()
        input_data = torch.randn(2, 3, 8, 8)
        simple_cfg = os.path.join(work_dir, 'simple.cfg')
        with open(simple_cfg, 'w') as fid:
            fid.write(simple_cfg_text)
        config_file = os.path.join(work_dir, 'config.json')
        amct_pytorch.create_quant_config(
            config_file=config_file,
            model=model,
            input_data=input_data,
            config_defination=simple_cfg,
        )

        record_file = os.path.join(work_dir, 'record.txt')
        modified_onnx = os.path.join(work_dir, 'modified.onnx')
        calib_model = amct_pytorch.quantize_model(
            config_file=config_file,
            modfied_onnx_file=modified_onnx,
            record_file=record_file,
            model=model,
            input_data=input_data,
        )
        calib_model.eval()
        with torch.no_grad():
            for _ in range(2):
                calib_model(torch.randn(2, 3, 8, 8))

        with open(record_file) as fid:
            record_text = fid.read()

        amct_pytorch.save_model(
            modfied_onnx_file=modified_onnx,
            record_file=record_file,
            save_path=os.path.join(work_dir, tag),
        )
        deploy_file = os.path.join(work_dir, tag + '_deploy_model.onnx')
        fakequant_file = os.path.join(work_dir, tag + '_fake_quant_model.onnx')
        return {
            'calib_model': calib_model,
            'record_text': record_text,
            'deploy_file': deploy_file,
            'deploy_attrs': _deploy_attrs(deploy_file),
            'fakequant_attrs': _deploy_attrs(fakequant_file),
        }

    def test_s1_auto_skip(self):
        result = self._run_flow(
            's1_auto_skip',
            'batch_num: 2\nenable_shift_value: "auto"\n'
            'skip_layers: "fc"\ncommon_config {\n}\n',
        )
        calib_model = result['calib_model']
        record_text = result['record_text']

        self.assertEqual(type(calib_model.conv).__name__, 'CaliQuant')
        self.assertIsInstance(calib_model.fc, ShiftCalibrationModule)
        self.assertEqual(calib_model.fc.quant_regime, 'fp16_fp16')

        self.assertIn('key: "conv"', record_text)
        self.assertIn('scale_w', record_text)
        self.assertIn('scale_d', record_text)

        # the fp16fixed kernel is plugged in: fc gets a really searched
        # fixed_shift_value within the fp16_fp16 candidate set {34..43}
        match = re.search(r'key: "fc".*?fixed_shift_value: (\d+)', record_text, re.S)
        self.assertIsNotNone(match)
        fc_shift_value = int(match.group(1))
        self.assertIn(fc_shift_value, range(34, 44))
        self.assertEqual(result['deploy_attrs'], {'fc': fc_shift_value})
        self.assertEqual(result['fakequant_attrs'], {})

    def test_s2_auto_weight_only(self):
        result = self._run_flow(
            's2_auto_wonly',
            'batch_num: 2\nenable_shift_value: "auto"\n'
            'override_layer_configs {\n  layer_name: "conv"\n'
            '  calibration_config {\n'
            '    arq_quantize { }\n  }\n}\ncommon_config {\n}\n',
        )
        calib_model = result['calib_model']
        record_text = result['record_text']

        # weight-only: no activation calibration wrapper, the module IS
        # the bare ShiftCalibrationModule (no CaliQuant shell)
        self.assertEqual(type(calib_model.conv).__name__, 'ShiftCalibrationModule')
        self.assertEqual(calib_model.conv.quant_regime, 'fp16_int8')
        self.assertEqual(type(calib_model.fc).__name__, 'CaliQuant')

        # weight-only conv: no activation factors in ITS record entry
        # (the fully quantized fc entry does carry scale_d -- its flush
        # is guarded by test_s13_mixed_order_shift_layer_first)
        conv_block = re.search(r'key: "conv"(.*?)\n\}', record_text, re.S)
        self.assertIsNotNone(conv_block, 'conv record entry missing')
        self.assertNotIn('act_type: "INT16"', conv_block.group(1))
        self.assertNotIn('scale_d', conv_block.group(1))
        self.assertNotIn('act_type', conv_block.group(1))
        # weight-only: record carries int8 weight factors only
        self.assertIn('wts_type: "INT8"', record_text)
        # fp16_int8 deploy graph contract:
        #   conv input: NOT AscendQuant (input stays unquantized)
        #   conv weight: int8 dtype with integer values (weight quantized)
        topology = _sv_deploy_topology(result['deploy_file'], 'conv')
        self.assertIsNotNone(topology)
        self.assertNotEqual(
            topology['input_op'],
            'AscendQuant',
            'the fp16_int8 operator input must not be quantized '
            '(no AscendQuant producer)',
        )
        deploy_weight = _deploy_int8_weight(result['deploy_file'], 'conv')
        self.assertIsNotNone(deploy_weight, 'conv weight initializer missing')
        self.assertEqual(
            deploy_weight.dtype,
            torch.int8,
            'the fp16_int8 operator weight must be int8-quantized',
        )
        self.assertTrue(
            torch.equal(deploy_weight, deploy_weight.round()),
            'the fp16_int8 operator weight values must be integers',
        )
        # the fp16_int8 search is disabled (the board kernel pins
        # sv=13 and ignores the passed value): conv gets the regime
        # default directly
        match = re.search(r'key: "conv".*?fixed_shift_value: (\d+)', record_text, re.S)
        self.assertIsNotNone(match)
        conv_shift_value = int(match.group(1))
        self.assertEqual(conv_shift_value, 13)
        self.assertNotEqual(result['deploy_attrs'], {})
        self.assertEqual(result['deploy_attrs'], {'conv': conv_shift_value})
        self.assertEqual(result['fakequant_attrs'], {})

    def test_s3_off_regression(self):
        result = self._run_flow(
            's3_off',
            'batch_num: 2\ncommon_config {\n}\n',
        )
        calib_model = result['calib_model']
        record_text = result['record_text']

        self.assertEqual(type(calib_model.conv).__name__, 'CaliQuant')
        self.assertEqual(type(calib_model.fc).__name__, 'CaliQuant')
        self.assertNotIn('fixed_shift_value', record_text)
        self.assertEqual(result['deploy_attrs'], {})
        self.assertEqual(result['fakequant_attrs'], {})

    def test_s12_mixed_order_act_factors_survive(self):
        """Regression of the singleton CopyFrom wipe (PR review Major):
        the Recorder keeps the activation factors of not yet flushed
        layers in the shared singleton; a shift value write that
        replaced the singleton content wholesale lost them. A plain
        quantized layer calibrated BEFORE a shift wrapped layer must
        keep its scale_d / offset_d in the final record."""
        result = self._run_flow(
            's12',
            'batch_num: 2\nenable_shift_value: "auto"\n'
            'override_layer_configs {\n  layer_name: "fc"\n'
            '  calibration_config {\n'
            '    arq_quantize { }\n  }\n}\ncommon_config {\n}\n',
        )
        record_text = result['record_text']

        # conv runs first (plain quantized): its activation factors are
        # accumulated in the singleton memory when the fc shift write
        # happens, they must survive to the final record
        conv_block = re.search(r'key: "conv"(.*?)\n\}', record_text, re.S)
        self.assertIsNotNone(conv_block, 'conv record entry missing')
        self.assertIn('scale_d', conv_block.group(1))
        self.assertIn('offset_d', conv_block.group(1))
        self.assertIn('act_type', conv_block.group(1))
        self.assertIn('scale_w', conv_block.group(1))

        # fc (shift wrapped, weight-only -- arq_quantize, no acts): complete as well
        fc_block = re.search(r'key: "fc"(.*?)\n\}', record_text, re.S)
        self.assertIsNotNone(fc_block, 'fc record entry missing')
        self.assertIn('fixed_shift_value', fc_block.group(1))
        self.assertNotIn('scale_d', fc_block.group(1))

        # deploy keeps the shift attribute of fc
        fc_shift = int(
            re.search(r'key: "fc".*?fixed_shift_value: (\d+)', record_text, re.S).group(
                1
            )
        )
        self.assertEqual(result['deploy_attrs'].get('fc'), fc_shift)

    def test_s13_mixed_order_shift_layer_first(self):
        """The opposite calibration order of s12: the shift wrapped layer
        runs BEFORE the plain quantized layer (conv weight-only, fc fully
        quantized). The shift calibration writes the record file directly
        and never goes through the Recorder counters -- the Recorder's
        flush criterion must therefore derive the shift layer count from
        the record content, otherwise fc's activation factors stay in
        memory and the deploy graph loses AscendQuant / AscendDequant."""
        result = self._run_flow(
            's13',
            'batch_num: 2\nenable_shift_value: "auto"\n'
            'override_layer_configs {\n  layer_name: "conv"\n'
            '  calibration_config {\n'
            '    arq_quantize { }\n  }\n}\ncommon_config {\n}\n',
        )
        record_text = result['record_text']

        # conv (shift wrapped, weight-only) first: fixed_shift_value in
        # the record, no activation factors
        conv_block = re.search(r'key: "conv"(.*?)\n\}', record_text, re.S)
        self.assertIsNotNone(conv_block, 'conv record entry missing')
        self.assertIn('fixed_shift_value', conv_block.group(1))
        self.assertNotIn('scale_d', conv_block.group(1))

        # fc (plain quantized) after: its activation factors must have
        # been flushed to the record file -- without the content-derived
        # shift count the mixed flush criterion never fires
        fc_block = re.search(r'key: "fc"(.*?)\n\}', record_text, re.S)
        self.assertIsNotNone(fc_block, 'fc record entry missing')
        self.assertIn('scale_d', fc_block.group(1))
        self.assertIn('offset_d', fc_block.group(1))
        self.assertIn('act_type', fc_block.group(1))

        # deploy: the fully quantized fc keeps its quantized topology
        # (AscendQuant producer / AscendDequant consumer)
        fc_topology = _sv_deploy_topology(result['deploy_file'], 'fc')
        self.assertIsNotNone(fc_topology, 'fc deploy node missing')
        self.assertEqual(
            fc_topology['input_op'],
            'AscendQuant',
            'the fully quantized fc input must stay quantized (AscendQuant producer)',
        )
        self.assertEqual(
            fc_topology['output_op'],
            'AscendDequant',
            'the fully quantized fc output must stay dequantized '
            '(AscendDequant consumer)',
        )
        # the weight-only conv keeps its searched shift attribute
        conv_shift = int(
            re.search(
                r'key: "conv".*?fixed_shift_value: (\d+)', record_text, re.S
            ).group(1)
        )
        self.assertEqual(result['deploy_attrs'].get('conv'), conv_shift)

    def _create_config_only(self, tag, simple_cfg_text, model, input_data):
        """create_quant_config only; returns the config file path."""
        work_dir = os.path.join(self.temp_folder, tag)
        if not os.path.isdir(work_dir):
            os.makedirs(work_dir)
        simple_cfg = os.path.join(work_dir, 'simple.cfg')
        with open(simple_cfg, 'w') as fid:
            fid.write(simple_cfg_text)
        config_file = os.path.join(work_dir, 'config.json')
        amct_pytorch.create_quant_config(
            config_file=config_file,
            model=model,
            input_data=input_data,
            config_defination=simple_cfg,
        )
        return config_file

    def test_s14_conv1d_weight_only_rejected(self):
        """Conv1d weight-only must be rejected: the module type comes
        from GraphQuerier.get_name_type_dict (torch class names), so
        Conv1d / Conv3d no longer fold onto Conv2d and silently export
        an int8 weight without any fixed_shift_value."""
        with self.assertRaises(ValueError) as ctx:
            self._create_config_only(
                's14_conv1d',
                'batch_num: 2\nenable_shift_value: "auto"\n'
                'override_layer_configs {\n  layer_name: "conv1d"\n'
                '  calibration_config {\n'
                '    arq_quantize { }\n  }\n}\ncommon_config {\n}\n',
                TinyConv1dModel().eval(),
                torch.randn(2, 3, 16),
            )
        self.assertIn('FIXED_SHIFT_VALUE_TYPES', str(ctx.exception))
        self.assertIn('conv1d', str(ctx.exception))

    def test_s15_weight_only_requires_enable_shift_value(self):
        """Weight-only on a supported type but WITHOUT enable_shift_value
        must be rejected up front: without the fp16fixed search the layer
        never gets a fixed_shift_value and the int8 weight would feed the
        raw operator (structurally valid, numerically wrong)."""
        with self.assertRaises(ValueError) as ctx:
            self._create_config_only(
                's15_no_esv',
                'batch_num: 2\n'
                'override_layer_configs {\n  layer_name: "conv"\n'
                '  calibration_config {\n'
                '    arq_quantize { }\n  }\n}\ncommon_config {\n}\n',
                TinyModel().eval(),
                torch.randn(2, 3, 8, 8),
            )
        self.assertIn("enable_shift_value", str(ctx.exception))

    def test_s16_weight_only_skip_fp16_and_quantized_mixed(self):
        """Three-way calibration order regression: a weight-only layer
        (conv1, fp16_int8), a skipped fp16_fp16 shift layer (conv2, pure
        fixed_shift_value record entry with no weight factors) and a
        fully quantized layer (fc) mixed in one model. The Recorder's
        mixed flush criterion and the layers_num fallback count must use
        the SAME pure-FP16-entry predicate: the skipped layer is counted
        on neither side, so fc's activation factors still flush and the
        deploy graph keeps fc's AscendQuant / AscendDequant."""
        result = self._run_flow(
            's16',
            'batch_num: 2\nenable_shift_value: "auto"\nskip_layers: "conv2"\n'
            'override_layer_configs {\n  layer_name: "conv1"\n'
            '  calibration_config {\n'
            '    arq_quantize { }\n  }\n}\ncommon_config {\n}\n',
            model=TinyTwoConvModel(),
        )
        record_text = result['record_text']

        # conv1: weight-only, fixed_shift_value + int8 weight factors
        conv1_block = re.search(r'key: "conv1"(.*?)\n\}', record_text, re.S)
        self.assertIsNotNone(conv1_block, 'conv1 record entry missing')
        self.assertIn('fixed_shift_value', conv1_block.group(1))
        self.assertIn('scale_w', conv1_block.group(1))
        self.assertNotIn('scale_d', conv1_block.group(1))

        # conv2: skipped fp16_fp16 shift layer, pure fixed_shift_value
        # entry (no weight factors, no activation factors)
        conv2_block = re.search(r'key: "conv2"(.*?)\n\}', record_text, re.S)
        self.assertIsNotNone(conv2_block, 'conv2 record entry missing')
        self.assertIn('fixed_shift_value', conv2_block.group(1))
        self.assertNotIn('scale_w', conv2_block.group(1))
        self.assertNotIn('scale_d', conv2_block.group(1))

        # fc: fully quantized, activation factors must have flushed to
        # the record file (the three-way mixed criterion regression)
        fc_block = re.search(r'key: "fc"(.*?)\n\}', record_text, re.S)
        self.assertIsNotNone(fc_block, 'fc record entry missing')
        self.assertIn('scale_d', fc_block.group(1))
        self.assertIn('offset_d', fc_block.group(1))
        self.assertIn('act_type', fc_block.group(1))

        # deploy: both shift layers carry the attribute, fc keeps its
        # quantized topology (AscendQuant producer / AscendDequant
        # consumer)
        conv1_shift = int(
            re.search(
                r'key: "conv1".*?fixed_shift_value: (\d+)', record_text, re.S
            ).group(1)
        )
        conv2_shift = int(
            re.search(
                r'key: "conv2".*?fixed_shift_value: (\d+)', record_text, re.S
            ).group(1)
        )
        # the fp16_int8 search is disabled (board kernel pins sv=13):
        # conv1 gets the regime default directly; conv2 (fp16_fp16)
        # keeps its SSE search
        self.assertEqual(conv1_shift, 13)
        self.assertIn(conv2_shift, range(34, 44))
        self.assertEqual(
            result['deploy_attrs'], {'conv1': conv1_shift, 'conv2': conv2_shift}
        )
        fc_topology = _sv_deploy_topology(result['deploy_file'], 'fc')
        self.assertIsNotNone(fc_topology, 'fc deploy node missing')
        self.assertEqual(
            fc_topology['input_op'],
            'AscendQuant',
            'the fully quantized fc input must stay quantized',
        )
        self.assertEqual(
            fc_topology['output_op'],
            'AscendDequant',
            'the fully quantized fc output must stay dequantized',
        )

    def test_s4_kernel_selection(self):
        def fake_compute(
            input_tensor, weight, quant_regime, module, weight_quant_info=None
        ):
            sse = torch.full((10,), 10.0)
            # candidate 37 is at index 3 of [34, ..., 43]
            sse[3] = 1.0
            return sse

        with patch.object(shift_value_kernel, 'compute_shift_value_sse', fake_compute):
            result = self._run_flow(
                's4_kernel',
                'batch_num: 2\nenable_shift_value: "auto"\n'
                'skip_layers: "fc"\ncommon_config {\n}\n',
            )

        self.assertIsInstance(result['calib_model'].fc, ShiftCalibrationModule)
        self.assertIn('fixed_shift_value: 37', result['record_text'])
        self.assertEqual(result['deploy_attrs'], {'fc': 37})
        self.assertEqual(result['fakequant_attrs'], {})

    def test_s5_int6_weight_only_never_searched(self):
        """Weight-only (arq_quantize, no acts) with an INT6 weight: the
        fp16_int8 entry requires an 8-bit weight, an INT6 layer never
        joins the sv calibration -- no fixed_shift_value at all."""
        result = self._run_flow(
            's5_int6_fallback',
            'batch_num: 2\nenable_shift_value: "auto"\n'
            'override_layer_configs {\n  layer_name: "conv"\n'
            '  calibration_config {\n'
            '    arq_quantize { quant_bits: 6 }\n  }\n}\ncommon_config {\n}\n',
        )
        record_text = result['record_text']

        self.assertIn('wts_type: "INT6"', record_text)
        self.assertNotIn('fixed_shift_value', record_text)
        self.assertEqual(result['deploy_attrs'], {})
        self.assertEqual(result['fakequant_attrs'], {})

    def test_s6_auto_weight_only_adaround_convt(self):
        """Weight-only ConvTranspose2d (AdaRound): rebuild == deploy int8.

        Guards the weight-only deploy premise end to end: after the
        AdaRound finetuning writes the fake-quant weight back, the int8
        weight rebuilt by the shift value kernel must be bit-exact with
        the weight the deploy ONNX actually carries.
        """
        tag = 's6_ada_convt'
        result = self._run_flow(
            tag,
            'batch_num: 2\nenable_shift_value: "auto"\n'
            'override_layer_configs {\n  layer_name: "deconv"\n'
            '  calibration_config {\n'
            '    ada_quantize { num_iteration: 100 }\n  }\n}\n'
            'common_config {\n}\n',
            model=TinyDeconvModel(),
        )
        record_text = result['record_text']
        calib_model = result['calib_model']

        # weight-only: no activation factors in the record
        self.assertNotIn('act_type: "INT16"', record_text)
        # the fp16_int8 search is disabled (board kernel pins sv=13):
        # deconv gets the regime default directly
        match = re.search(
            r'key: "deconv".*?fixed_shift_value: (\d+)', record_text, re.S
        )
        self.assertIsNotNone(match)
        deconv_shift_value = int(match.group(1))
        self.assertEqual(deconv_shift_value, 13)
        self.assertEqual(result['deploy_attrs'], {'deconv': deconv_shift_value})

        # the rebuilt int8 weight is bit-exact with the deploy weight
        # weight-only: the module IS the bare ShiftCalibrationModule
        shift_module = calib_model.deconv
        self.assertEqual(shift_module.quant_regime, 'fp16_int8')
        weight_quant_info = shift_module._get_weight_quant_info()
        self.assertIsNotNone(weight_quant_info)
        rebuilt, _ = shift_value_kernel._rebuild_int8_weight(
            shift_module.object_module, weight_quant_info
        )
        deploy_file = os.path.join(self.temp_folder, tag, tag + '_deploy_model.onnx')
        deploy_weight = _deploy_int8_weight(deploy_file, 'deconv')
        self.assertIsNotNone(deploy_weight)
        self.assertEqual(deploy_weight.dtype, torch.int8)
        self.assertEqual(tuple(deploy_weight.shape), tuple(rebuilt.shape))
        self.assertTrue(
            torch.equal(deploy_weight, rebuilt),
            'the rebuilt int8 weight differs from the deploy weight',
        )

    def test_s2b_act16_activation_is_quantized_not_searched(self):
        """auto + act INT16: the activation IS quantized there (AscendQuant
        in the deploy graph), it does not match the unquantized-FP16-input
        contract of fp16_int8 and must NOT join the search."""
        result = self._run_flow(
            's2b_act16',
            'batch_num: 2\nenable_shift_value: "auto"\n'
            'override_layer_configs {\n  layer_name: "conv"\n'
            '  calibration_config {\n    ifmr_quantize { dst_type: INT16 }\n'
            '    arq_quantize { }\n  }\n}\ncommon_config {\n}\n',
        )
        calib_model = result['calib_model']
        record_text = result['record_text']

        # conv stays a plain activation-calibrated layer: no shift wrapper
        self.assertEqual(type(calib_model.conv).__name__, 'CaliQuant')
        self.assertNotIsInstance(
            getattr(calib_model.conv, 'sub_module', None), ShiftCalibrationModule
        )
        self.assertNotIn('fixed_shift_value', record_text)
        self.assertEqual(result['deploy_attrs'], {})
        self.assertEqual(result['fakequant_attrs'], {})

    def test_s7_all_mode(self):
        """enable_shift_value='all': force the FP16 path on every target."""
        tag = 's7_all_mode'
        result = self._run_flow(
            tag,
            'batch_num: 2\nenable_shift_value: "all"\ncommon_config {\n}\n',
        )
        record_text = result['record_text']
        calib_model = result['calib_model']

        # the config generation dropped the target layers from the quant
        # config: no CaliQuant wrapper, both run the FP16 non-quantized
        # path (fp16_fp16 regime)
        self.assertIsInstance(calib_model.conv, ShiftCalibrationModule)
        self.assertIsInstance(calib_model.fc, ShiftCalibrationModule)
        self.assertEqual(calib_model.conv.quant_regime, 'fp16_fp16')
        self.assertEqual(calib_model.fc.quant_regime, 'fp16_fp16')

        # both layers get really searched values within {34, ..., 43}
        for layer in ('conv', 'fc'):
            match = re.search(
                r'key: "{}".*?fixed_shift_value: (\d+)'.format(layer),
                record_text,
                re.S,
            )
            self.assertIsNotNone(match, layer)
            value = int(match.group(1))
            self.assertIn(value, range(34, 44))
            self.assertEqual(result['deploy_attrs'][layer], value)
        self.assertEqual(result['fakequant_attrs'], {})

        # the module -> node name mapping survives: Linear exports as a
        # Gemm node named after the module
        deploy_file = os.path.join(self.temp_folder, tag, tag + '_deploy_model.onnx')
        nodes = {node.name: node for node in onnx.load(deploy_file).graph.node}
        self.assertEqual(nodes['conv'].op_type, 'Conv')
        self.assertEqual(nodes['fc'].op_type, 'Gemm')

        # fp16_fp16 deploy graph contract: neither side may carry any
        # quant node -- the operator consumes the raw input and produces
        # the raw output (both operands stay unquantized FP16)
        for target in ('conv', 'fc'):
            topology = _sv_deploy_topology(result['deploy_file'], target)
            self.assertIsNotNone(topology, target)
            self.assertNotEqual(
                topology['input_op'],
                'AscendQuant',
                '{}: the fp16_fp16 operator input must stay unquantized'.format(target),
            )
            self.assertNotEqual(
                topology['output_op'],
                'AscendDequant',
                '{}: the fp16_fp16 operator output must not be dequantized'
                '(the weight stays FP16)'.format(target),
            )

    def test_s8_name_mapping_bn_fusion(self):
        """After BN fusion the fused Conv keeps the module name."""
        tag = 's8_bn_fusion'
        result = self._run_flow(
            tag,
            'batch_num: 2\nenable_shift_value: "auto"\nskip_layers: "conv"\n'
            'common_config {\n}\n',
            model=TinyConvBnModel(),
        )
        record_text = result['record_text']

        # conv (skipped, FP16 non-quantized) is searched while the BN is
        # fused into it (do_fusion defaults to true)
        match = re.search(r'key: "conv".*?fixed_shift_value: (\d+)', record_text, re.S)
        self.assertIsNotNone(match)
        conv_shift_value = int(match.group(1))
        self.assertIn(conv_shift_value, range(34, 44))

        deploy_file = os.path.join(self.temp_folder, tag, tag + '_deploy_model.onnx')
        nodes = {node.name: node for node in onnx.load(deploy_file).graph.node}
        # the fusion really happened: no BatchNormalization node is left
        self.assertNotIn(
            'bn',
            [
                name
                for name, node in nodes.items()
                if 'BatchNormalization' in node.op_type
            ],
        )
        # the fused Conv node keeps the module name and carries the attr
        self.assertEqual(nodes['conv'].op_type, 'Conv')
        self.assertEqual(result['deploy_attrs'], {'conv': conv_shift_value})

    def test_s8_name_mapping_linear_matmul_add(self):
        """A 3D-input Linear exports as MatMul + Add, name still matches."""
        tag = 's8_matmul_add'
        result = self._run_flow(
            tag,
            'batch_num: 2\nenable_shift_value: "auto"\nskip_layers: "fc"\n'
            'common_config {\n}\n',
            model=TinyLinear3dModel(),
        )
        record_text = result['record_text']

        match = re.search(r'key: "fc".*?fixed_shift_value: (\d+)', record_text, re.S)
        self.assertIsNotNone(match)
        fc_shift_value = int(match.group(1))
        self.assertIn(fc_shift_value, range(34, 44))

        # the MatMul node carries the module name and the attribute; the
        # bias Add node is not a shift value target and stays clean
        deploy_file = os.path.join(self.temp_folder, tag, tag + '_deploy_model.onnx')
        nodes = {node.name: node for node in onnx.load(deploy_file).graph.node}
        self.assertEqual(nodes['fc'].op_type, 'MatMul')
        self.assertEqual(result['deploy_attrs'], {'fc': fc_shift_value})
        add_nodes = [
            node for node in nodes.values() if node.op_type == 'Add' and node.name != ''
        ]
        for node in add_nodes:
            self.assertNotIn(
                'fixed_shift_value',
                [attr.name for attr in node.attribute],
            )


if __name__ == '__main__':
    unittest.main()
