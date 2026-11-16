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
- S1 auto + skip_layers: skipped layer fp16_fp16, quantized layer untouched,
  fc gets a really searched fixed_shift_value from the fp16fixed kernel;
- S2 auto + act num_bits=16 (A16 weight-only): fp16_int8, act INT16
  calibration + weight INT8 + fixed_shift_value (really searched from the
  rebuilt deployed int8 weight, within {5, ..., 14}) coexist;
- S3 off: regression, no wrapper and no attribute;
- S4 fp16fixed kernel monkeypatched: the argmin selection takes effect;
- S5 auto + INT6 weight-only: the S8 kernel constraint (INT8 only) sends
  the layer to the regime default value 13;
- S6 auto + AdaRound weight-only ConvTranspose2d: the rebuilt int8
  weight is bit-exact with the deploy ONNX weight (the premise of the
  fp16_int8 search, guarded end to end);
- S7 all: every target module is forced onto the FP16 non-quantized
  path and searched, the Linear exports as Gemm;
- S8 name mapping: after BN fusion the fused Conv keeps the module
  name; a Linear applied to a 3D input exports as MatMul + Add and the
  MatMul node still carries the module name.
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
            '  calibration_config {\n    ifmr_quantize { dst_type: INT16 }\n'
            '    arq_quantize { }\n  }\n}\ncommon_config {\n}\n',
        )
        calib_model = result['calib_model']
        record_text = result['record_text']

        self.assertEqual(type(calib_model.conv).__name__, 'CaliQuant')
        # conv is wrapped inside the CaliQuant wrapper
        self.assertIsInstance(calib_model.conv.sub_module, ShiftCalibrationModule)
        self.assertEqual(calib_model.conv.sub_module.quant_regime, 'fp16_int8')
        self.assertEqual(type(calib_model.fc).__name__, 'CaliQuant')

        self.assertIn('act_type: "INT16"', record_text)
        # the int8 regime kernel is plugged in: conv gets a really
        # searched fixed_shift_value within {5, ..., 14}, rebuilt from
        # the fake-quant weight and the record scale_w
        match = re.search(r'key: "conv".*?fixed_shift_value: (\d+)', record_text, re.S)
        self.assertIsNotNone(match)
        conv_shift_value = int(match.group(1))
        self.assertIn(conv_shift_value, range(5, 15))
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
            '  calibration_config {\n    ifmr_quantize { dst_type: INT16 }\n'
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

        # fc (shift wrapped, INT16 weight-only): complete as well
        fc_block = re.search(r'key: "fc"(.*?)\n\}', record_text, re.S)
        self.assertIsNotNone(fc_block, 'fc record entry missing')
        self.assertIn('fixed_shift_value', fc_block.group(1))
        self.assertIn('scale_d', fc_block.group(1))

        # deploy keeps the shift attribute of fc
        fc_shift = int(
            re.search(r'key: "fc".*?fixed_shift_value: (\d+)', record_text, re.S).group(
                1
            )
        )
        self.assertEqual(result['deploy_attrs'].get('fc'), fc_shift)

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

    def test_s5_auto_weight_only_int6_fallback(self):
        """Weight-only with an INT6 weight: the S8 kernel falls back."""
        result = self._run_flow(
            's5_int6_fallback',
            'batch_num: 2\nenable_shift_value: "auto"\n'
            'override_layer_configs {\n  layer_name: "conv"\n'
            '  calibration_config {\n    ifmr_quantize { dst_type: INT16 }\n'
            '    arq_quantize { quant_bits: 6 }\n  }\n}\ncommon_config {\n}\n',
        )
        record_text = result['record_text']

        self.assertIn('act_type: "INT16"', record_text)
        self.assertIn('wts_type: "INT6"', record_text)
        # the F16xS8 path models an INT8 weight only: default value 13
        self.assertIn('fixed_shift_value: 13', record_text)
        self.assertEqual(result['deploy_attrs'], {'conv': 13})
        self.assertEqual(result['fakequant_attrs'], {})

    def test_s6_auto_weight_only_adaround_convt(self):
        """Weight-only ConvTranspose2d (AdaRound): rebuild == deploy int8.

        Guards the premise of the fp16_int8 search end to end: after the
        AdaRound finetuning writes the fake-quant weight back, the int8
        weight rebuilt by the shift value kernel must be bit-exact with
        the weight the deploy ONNX actually carries.
        """
        tag = 's6_ada_convt'
        result = self._run_flow(
            tag,
            'batch_num: 2\nenable_shift_value: "auto"\n'
            'override_layer_configs {\n  layer_name: "deconv"\n'
            '  calibration_config {\n    ifmr_quantize { dst_type: INT16 }\n'
            '    ada_quantize { num_iteration: 100 }\n  }\n}\n'
            'common_config {\n}\n',
            model=TinyDeconvModel(),
        )
        record_text = result['record_text']
        calib_model = result['calib_model']

        self.assertIn('act_type: "INT16"', record_text)
        # deconv gets a really searched value within {5, ..., 14}
        match = re.search(
            r'key: "deconv".*?fixed_shift_value: (\d+)', record_text, re.S
        )
        self.assertIsNotNone(match)
        deconv_shift_value = int(match.group(1))
        self.assertIn(deconv_shift_value, range(5, 15))
        self.assertEqual(result['deploy_attrs'], {'deconv': deconv_shift_value})

        # the rebuilt int8 weight is bit-exact with the deploy weight
        shift_module = calib_model.deconv.sub_module
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
