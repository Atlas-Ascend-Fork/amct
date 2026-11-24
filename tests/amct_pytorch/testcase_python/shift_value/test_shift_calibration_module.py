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
"""Unit tests of ShiftCalibrationModule."""

import os
import shutil
import unittest

import pytest
from unittest.mock import patch

import torch
from google.protobuf import text_format

from amct_pytorch.classic.graph_based.amct_pytorch.nn.module.quantization import (
    shift_value_kernel,
)
from amct_pytorch.classic.graph_based.amct_pytorch.nn.module.quantization.shift_calibration import (
    ShiftCalibrationModule,
)
from amct_pytorch.classic.graph_based.amct_pytorch.proto import (
    scale_offset_record_pb2,
)

pytestmark = pytest.mark.cpu


CUR_DIR = os.path.split(os.path.realpath(__file__))[0]

PRESET_RECORD = '''
record {
  key: "conv"
  value {
    scale_w: 0.001
    scale_w: 0.002
    wts_type: "INT8"
  }
}
'''


def _read_record(record_file):
    records = scale_offset_record_pb2.ScaleOffsetRecord()
    with open(record_file, 'r') as fid:
        text_format.Merge(fid.read(), records)
    return records


class TestShiftCalibrationModule(unittest.TestCase):
    def setUp(self):
        self.temp_folder = os.path.join(CUR_DIR, 'test_shift_calibration_module')
        if not os.path.isdir(self.temp_folder):
            os.makedirs(self.temp_folder)
        self.record_file = os.path.join(self.temp_folder, 'record.txt')
        with open(self.record_file, 'w') as fid:
            fid.write(PRESET_RECORD)
        self.fc = torch.nn.Linear(8 * 8 * 8, 4)
        self.inputs = torch.randn(2, 8 * 8 * 8)

    def tearDown(self):
        shutil.rmtree(self.temp_folder, ignore_errors=True)

    def _make_module(self, batch_num=2, quant_regime='fp16_fp16', object_module=None):
        return ShiftCalibrationModule(
            object_module=object_module if object_module is not None else self.fc,
            layer_name='fc',
            record_file=self.record_file,
            batch_num=batch_num,
            quant_regime=quant_regime,
        )

    def test_conv_groups_supported(self):
        """groups != 1 Conv2d is calibrated (engine has the full group loop)."""
        conv = torch.nn.Conv2d(6, 8, 3, padding=1, groups=2)
        self.assertTrue(ShiftCalibrationModule._check_module_supported(conv))
        module = self._make_module(object_module=conv)
        self.assertFalse(module.use_default_value)
        with torch.no_grad():
            module(torch.randn(2, 6, 8, 8))
            module(torch.randn(2, 6, 8, 8))
        self.assertIn('fixed_shift_value', open(self.record_file).read())

    def test_convt_groups_and_output_padding_supported(self):
        """groups != 1 / output_padding != 0 ConvTranspose2d are calibrated."""
        convt = torch.nn.ConvTranspose2d(
            6, 8, 3, stride=2, padding=1, output_padding=(1, 0), groups=2
        )
        self.assertTrue(ShiftCalibrationModule._check_module_supported(convt))
        module = self._make_module(object_module=convt)
        self.assertFalse(module.use_default_value)
        with torch.no_grad():
            module(torch.randn(2, 6, 8, 8))
            module(torch.randn(2, 6, 8, 8))
        self.assertIn('fixed_shift_value', open(self.record_file).read())

    def test_invalid_quant_regime(self):
        with self.assertRaises(ValueError):
            self._make_module(quant_regime='int8_int8')

    def test_forward_output_untouched(self):
        """The wrapper keeps the user model numerics untouched."""
        module = self._make_module()
        with torch.no_grad():
            expect = self.fc(self.inputs)
            actual = module(self.inputs)
        self.assertTrue(torch.equal(actual, expect))

    def test_default_value_written_without_kernel(self):
        """Kernel missing (NotImplementedError): the regime default value."""
        with patch.object(
            shift_value_kernel,
            'compute_shift_value_sse',
            side_effect=NotImplementedError,
        ):
            module = self._make_module(batch_num=2)
            with torch.no_grad():
                module(self.inputs)
                # the layer entry is not written before batch_num is reached
                self.assertNotIn(
                    'fc',
                    [item.key for item in _read_record(self.record_file).record],
                )
                module(self.inputs)
        records = _read_record(self.record_file)
        record_map = {item.key: item.value for item in records.record}
        self.assertEqual(record_map['fc'].fixed_shift_value, 42)
        # the preset entry of the other layer is kept untouched
        for actual, expect in zip(record_map['conv'].scale_w, [0.001, 0.002]):
            self.assertAlmostEqual(actual, expect, places=6)
        self.assertEqual(record_map['conv'].wts_type, 'INT8')

    def test_kernel_searched_value_written(self):
        """Kernel plugged in: the argmin of the accumulated SSE is written."""

        def fake_compute(
            input_tensor, weight, quant_regime, calib_module, weight_quant_info=None
        ):
            sse = torch.full((10,), 9.0)
            sse[6] = 1.0
            return sse

        with patch.object(shift_value_kernel, 'compute_shift_value_sse', fake_compute):
            module = self._make_module(batch_num=2)
            with torch.no_grad():
                module(self.inputs)
                module(self.inputs)
        records = _read_record(self.record_file)
        record_map = {item.key: item.value for item in records.record}
        # candidate list is [34, ..., 43], argmin falls on index 6 -> 40
        self.assertEqual(record_map['fc'].fixed_shift_value, 40)

    def test_fp16_int8_default_value(self):
        """fp16_int8 without a record entry for this layer: default 13.

        The preset record only carries the 'conv' entry, so 'fc' has no
        weight quant factors and the kernel falls back.
        """
        module = self._make_module(batch_num=1, quant_regime='fp16_int8')
        with torch.no_grad():
            module(self.inputs)
        records = _read_record(self.record_file)
        record_map = {item.key: item.value for item in records.record}
        self.assertEqual(record_map['fc'].fixed_shift_value, 13)

    def test_fp16_int8_pinned_default_written(self):
        """fp16_int8: the search is disabled (the board kernel pins
        sv=13 and ignores the passed value), the regime default 13 is
        written without running any probe -- with or without weight
        quant factors, symmetric or asymmetric."""

        def _write_fc_record(offset_w=0):
            weight = self.fc.weight.detach()
            scale = (weight.abs().amax(dim=1) / 127.0).clamp_min(2.0**-30)
            # leave the fake-quant weight behind, as WeightsCalibrationPass
            # does, so the rebuild recovers the deployed int8 weight
            int8 = torch.round(weight / scale.reshape(-1, 1)).clamp(-127, 127)
            self.fc.weight.data = int8.to(torch.float32) * scale.reshape(-1, 1)
            with open(self.record_file, 'w') as fid:
                fid.write('record {\n  key: "fc"\n  value {\n')
                for entry in scale.tolist():
                    fid.write('    scale_w: {}\n'.format(entry))
                for _ in range(scale.numel()):
                    fid.write('    offset_w: {}\n'.format(offset_w))
                fid.write('    wts_type: "INT8"\n  }\n}\n')

        module = self._make_module(batch_num=1, quant_regime='fp16_int8')
        self.assertTrue(module.use_default_value)
        # the probe is never invoked: whatever the record factors look
        # like, the pinned default is written
        with patch.object(
            shift_value_kernel,
            'compute_shift_value_sse',
            side_effect=AssertionError('fp16_int8 search must not run'),
        ) as probe_mock:
            _write_fc_record()
            module = self._make_module(batch_num=1, quant_regime='fp16_int8')
            with torch.no_grad():
                module(self.inputs)
        probe_mock.assert_not_called()
        records = _read_record(self.record_file)
        record_map = {item.key: item.value for item in records.record}
        self.assertEqual(record_map['fc'].fixed_shift_value, 13)

        # an asymmetric weight quantization keeps the same pinned default
        _write_fc_record(offset_w=1)
        self.fc.weight.data = torch.randn_like(self.fc.weight.data)
        module = self._make_module(batch_num=1, quant_regime='fp16_int8')
        with torch.no_grad():
            module(self.inputs)
        records = _read_record(self.record_file)
        record_map = {item.key: item.value for item in records.record}
        self.assertEqual(record_map['fc'].fixed_shift_value, 13)

    def test_non_finite_input_falls_back(self):
        """inf / nan in the calibration data: the regime default value.

        The fixed point pipeline has no representation for inf / nan
        (raw exponent 31), so the layer falls back instead of searching
        on garbage. A finite fp32 value beyond the fp16 range becomes
        inf in the search domain and falls back too.
        """
        # fp16_fp16: inf input
        module = self._make_module(batch_num=1)
        with torch.no_grad():
            module(torch.full((2, 8 * 8 * 8), float('inf')))
        records = _read_record(self.record_file)
        record_map = {item.key: item.value for item in records.record}
        self.assertEqual(record_map['fc'].fixed_shift_value, 42)

        # fp32 input beyond the fp16 range: inf after the fp16 cast
        module = self._make_module(batch_num=1)
        with torch.no_grad():
            module(torch.full((2, 8 * 8 * 8), 1e5))
        records = _read_record(self.record_file)
        record_map = {item.key: item.value for item in records.record}
        self.assertEqual(record_map['fc'].fixed_shift_value, 42)

        # fp16_int8: nan input
        module = self._make_module(batch_num=1, quant_regime='fp16_int8')
        with torch.no_grad():
            module(torch.full((2, 8 * 8 * 8), float('nan')))
        records = _read_record(self.record_file)
        record_map = {item.key: item.value for item in records.record}
        self.assertEqual(record_map['fc'].fixed_shift_value, 13)

    def test_non_finite_weight_falls_back(self):
        """fp16_fp16 with a non-finite weight: the regime default value."""
        self.fc.weight.data[0] = float('inf')
        module = self._make_module(batch_num=1)
        with torch.no_grad():
            module(self.inputs)
        records = _read_record(self.record_file)
        record_map = {item.key: item.value for item in records.record}
        self.assertEqual(record_map['fc'].fixed_shift_value, 42)

    def test_tie_breaks_to_first_candidate(self):
        """Equal SSE across candidates: the first (smallest) sv is written.

        A bias-free module with an all-zero input yields identical SSE
        for every candidate; ``_select_value`` takes the argmin, whose
        tie rule is the first index, i.e. the smallest candidate (34
        for fp16_fp16 -- the coarsest grid / most headroom).
        """
        no_bias_fc = torch.nn.Linear(8 * 8 * 8, 4, bias=False)
        module = self._make_module(batch_num=1, object_module=no_bias_fc)
        with torch.no_grad():
            module(torch.zeros(2, 8 * 8 * 8))
        records = _read_record(self.record_file)
        record_map = {item.key: item.value for item in records.record}
        self.assertEqual(record_map['fc'].fixed_shift_value, 34)

    def test_no_write_before_batch_num_reached(self):
        module = self._make_module(batch_num=3)
        with torch.no_grad():
            module(self.inputs)
            module(self.inputs)
        self.assertEqual(module.current_batch, 2)
        records = _read_record(self.record_file)
        self.assertNotIn('fc', [item.key for item in records.record])

    def test_write_once_across_extra_batches(self):
        """Extra batches beyond batch_num neither re-write nor duplicate."""
        module = self._make_module(batch_num=1)
        with torch.no_grad():
            module(self.inputs)
            module(self.inputs)
            module(self.inputs)
        records = _read_record(self.record_file)
        keys = [item.key for item in records.record]
        self.assertEqual(keys.count('fc'), 1)

    def test_kernel_selection_by_accumulated_sse(self):
        """The candidate with the minimum accumulated SSE is selected."""
        module = self._make_module(batch_num=2)
        sse_values = [
            torch.full((10,), 5.0),
            torch.full((10,), 5.0),
        ]
        # after two batches candidate index 0 sums to 6.0 (argmin)
        sse_values[0][1] = 1.0
        sse_values[1][1] = 3.0
        # candidate index 2 sums to 8.0
        sse_values[0][2] = 4.0
        sse_values[1][2] = 4.0

        def fake_compute(
            input_tensor, weight, quant_regime, calib_module, weight_quant_info=None
        ):
            return sse_values[module.current_batch]

        with patch.object(shift_value_kernel, 'compute_shift_value_sse', fake_compute):
            with torch.no_grad():
                module(self.inputs)
                module(self.inputs)
        records = _read_record(self.record_file)
        record_map = {item.key: item.value for item in records.record}
        # candidate list is [34, ..., 43], argmin falls on index 1 -> 35
        self.assertEqual(record_map['fc'].fixed_shift_value, 35)

    def test_kernel_sse_length_mismatch(self):
        module = self._make_module(batch_num=1)

        def fake_compute(
            input_tensor, weight, quant_regime, calib_module, weight_quant_info=None
        ):
            return torch.zeros(3)

        with patch.object(shift_value_kernel, 'compute_shift_value_sse', fake_compute):
            with torch.no_grad():
                with self.assertRaises(RuntimeError):
                    module(self.inputs)

    def test_module_support_constraints(self):
        """Support constraints: the fp16fixed engines implement the full group
        loop and the conv-transpose output_padding canvas, so those
        first-stage restrictions are lifted (verified against
        F.conv2d / F.conv_transpose2d with groups {1,2,4} and
        output_padding {(0,1),(1,1),(1,0)})."""
        group_conv = torch.nn.Conv2d(4, 8, 3, groups=2)
        self.assertTrue(
            ShiftCalibrationModule._check_module_supported(torch.nn.Linear(4, 2))
        )
        self.assertTrue(
            ShiftCalibrationModule._check_module_supported(torch.nn.Conv2d(4, 8, 3))
        )
        self.assertTrue(ShiftCalibrationModule._check_module_supported(group_conv))
        self.assertTrue(
            ShiftCalibrationModule._check_module_supported(
                torch.nn.ConvTranspose2d(4, 8, 3, output_padding=(1, 0))
            )
        )
        self.assertTrue(
            ShiftCalibrationModule._check_module_supported(
                torch.nn.ConvTranspose2d(4, 8, 3)
            )
        )
        self.assertFalse(
            ShiftCalibrationModule._check_module_supported(torch.nn.BatchNorm1d(4))
        )
        # unsupported module falls back to the default value
        module = self._make_module(batch_num=1, object_module=torch.nn.BatchNorm2d(4))
        self.assertTrue(module.use_default_value)
        with torch.no_grad():
            module(torch.randn(1, 4, 8, 8))
        records = _read_record(self.record_file)
        record_map = {item.key: item.value for item in records.record}
        self.assertEqual(record_map['fc'].fixed_shift_value, 42)

    def test_record_field_isolation(self):
        """Only fixed_shift_value is added, other preset fields survive."""
        with open(self.record_file, 'w') as fid:
            fid.write(
                'record {\n  key: "fc"\n  value {\n'
                '    scale_w: 0.5\n    wts_type: "INT8"\n  }\n}\n'
            )
        with patch.object(
            shift_value_kernel,
            'compute_shift_value_sse',
            side_effect=NotImplementedError,
        ):
            module = self._make_module(batch_num=1)
            with torch.no_grad():
                module(self.inputs)
        records = _read_record(self.record_file)
        record_map = {item.key: item.value for item in records.record}
        self.assertEqual(record_map['fc'].fixed_shift_value, 42)
        self.assertEqual(list(record_map['fc'].scale_w), [0.5])
        self.assertEqual(record_map['fc'].wts_type, 'INT8')


if __name__ == '__main__':
    unittest.main()
