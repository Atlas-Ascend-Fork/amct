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
import logging
import unittest

from amct_pytorch.classic.graph_based.amct_pytorch.capacity import CAPACITY
from amct_pytorch.classic.graph_based.amct_pytorch.common.config import field as F

logger = logging.getLogger(__name__)


def _f(cls):
    return cls(CAPACITY)


class TestLeafFieldChecks(unittest.TestCase):
    def test_version_field(self):
        f = _f(F.VersionField)
        self.assertRaises(TypeError, f.check, 'version', '1')
        self.assertRaises(ValueError, f.check, 'version', 2)
        f.check('version', 1)

    def test_batch_num_field(self):
        f = _f(F.BatchNumField)
        self.assertRaises(TypeError, f.check, 'batch_num', '1')
        self.assertRaises(ValueError, f.check, 'batch_num', 0)
        f.check('batch_num', 4)
        self.assertEqual(f.default_value(), F.BATCH_NUM)

    def test_bool_fields(self):
        for cls in (F.ActOffsetField, F.LayerActOffsetField, F.JointQuantField,
                    F.WtsOffsetField, F.DoFusionField, F.QuantEnableField):
            f = _f(cls)
            self.assertRaises(TypeError, f.check, 'x', 'not_bool')
            f.check('x', True)
        # has_default coverage
        self.assertFalse(_f(F.LayerActOffsetField).has_default())

    def test_act_num_bits_field(self):
        f = _f(F.ActNumBitsField)
        self.assertRaises(TypeError, f.check, 'num_bits', '8')
        self.assertRaises(ValueError, f.check, 'num_bits', 3)
        f.check('num_bits', 8)

    def test_wts_num_bits_field(self):
        f = _f(F.WtsNumBitsField)
        self.assertRaises(ValueError, f.check, 'num_bits', 5)
        f.check('num_bits', 4)

    def test_fakequant_precision_mode_field(self):
        f = _f(F.FakequantPrecisionModeField)
        self.assertRaises(TypeError, f.check, 'fp', 1)
        self.assertRaises(ValueError, f.check, 'fp', 'BAD')
        f.check('fp', 'DEFAULT')
        self.assertEqual(f.default_value(), 'DEFAULT')

    def test_wts_algo_field(self):
        f = _f(F.WtsAlgoField)
        self.assertRaises(ValueError, f.check, 'wts_algo', 'bad')
        f.check('wts_algo', 'arq_quantize')

    def test_act_algo_field(self):
        f = _f(F.ActAlgoField)
        self.assertRaises(ValueError, f.check, 'act_algo', 'bad')
        f.check('act_algo', 'ifmr')

    def test_approximate_algo_field(self):
        f = _f(F.ApproximateAlgoField)
        self.assertRaises(ValueError, f.check, 'algo', 'bad')
        f.check('algo', 'FastSoftmax')
        self.assertEqual(f.default_value(), 'FastSoftmax')

    def test_num_of_bins_field(self):
        f = _f(F.NumOfBinsField)
        self.assertRaises(TypeError, f.check, 'n', '1024')
        self.assertRaises(ValueError, f.check, 'n', 999)
        f.check('n', 1024)

    def test_dmq_balancer_param_field(self):
        f = _f(F.DMQBalancerParamField)
        self.assertFalse(f.has_default())
        self.assertRaises(TypeError, f.check, 'dmq', 1)
        self.assertRaises(ValueError, f.check, 'dmq', 0.9)
        f.check('dmq', 0.5)

    def test_beta_range_field(self):
        f = _f(F.BetaRangeField)
        self.assertRaises(TypeError, f.check, 'beta', 1.0)
        self.assertRaises(ValueError, f.check, 'beta', [1.0])
        self.assertRaises(ValueError, f.check, 'beta', [1.0, 2.0])  # start<=end
        self.assertRaises(ValueError, f.check, 'beta', [2.0, -1.0])  # end<=0
        f.check('beta', [2.0, 1.0])

    def test_warm_start_field(self):
        f = _f(F.WarmStartField)
        self.assertRaises(ValueError, f.check, 'ws', 1)
        self.assertRaises(ValueError, f.check, 'ws', 0)
        f.check('ws', 0.5)

    def test_num_iteration_field(self):
        f = _f(F.NumIterationField)
        self.assertRaises(ValueError, f.check, 'ni', -1)
        f.check('ni', 10)

    def test_reg_param_field(self):
        f = _f(F.RegParamField)
        self.assertRaises(ValueError, f.check, 'rp', 1)
        f.check('rp', 0.01)

    def test_max_percentile_field(self):
        f = _f(F.MaxPercentileField)
        self.assertRaises(ValueError, f.check, 'mp', 0.4)
        self.assertRaises(ValueError, f.check, 'mp', 1.5)
        f.check('mp', 0.999)

    def test_search_step_field(self):
        f = _f(F.SearchStepField)
        self.assertRaises(ValueError, f.check, 'ss', 0)
        f.check('ss', 0.01)

    def test_search_range_field(self):
        f = _f(F.SearchRangeField)
        self.assertRaises(TypeError, f.check, 'sr', 1.0)
        self.assertRaises(ValueError, f.check, 'sr', [1.0])
        self.assertRaises(ValueError, f.check, 'sr', [0, 1.0])      # start<=0
        self.assertRaises(ValueError, f.check, 'sr', [1.3, 0.7])    # start>=end
        f.check('sr', [0.7, 1.3])

    def test_layer_name_and_input_index_field(self):
        ln = _f(F.LayerNameField)
        self.assertFalse(ln.has_default())
        self.assertRaises(TypeError, ln.check, 'layer_name', 1)
        ln.check('layer_name', 'conv1')
        ii = _f(F.InputIndexField)
        self.assertFalse(ii.has_default())
        self.assertRaises(TypeError, ii.check, 'input_index', '0')
        ii.check('input_index', 0)


class TestUnimplementedBase(unittest.TestCase):
    def test_field_base_unimplemented(self):
        f = F.Field(CAPACITY)
        self.assertTrue(f.has_default())
        self.assertRaises(RuntimeError, f.is_leaf)
        self.assertRaises(RuntimeError, f.check, 'x', 1)

    def test_leaf_field_default_value_unimplemented(self):
        lf = F.LeafField(CAPACITY)
        self.assertTrue(lf.is_leaf())
        self.assertRaises(RuntimeError, lf.default_value)

    def test_placeholder_match_unimplemented(self):
        plh = F.PlaceholderField(CAPACITY)
        self.assertRaises(RuntimeError, plh.match, 'x')


class TestParamPool(unittest.TestCase):
    def test_set_get_clear(self):
        pool = F.ParamPool()
        pool.set_quant_layers(['a'])
        pool.set_supported_layers(['a', 'b'])
        pool.set_skip_layers(['s'])
        pool.set_layer_type({'a': 'Conv2d'})
        pool.set_layer_name('a')
        pool.set_wts_algo('arq_quantize')
        pool.set_act_algo('ifmr')
        self.assertEqual(pool.get_quant_layers(), ['a'])
        self.assertEqual(pool.get_supported_layers(), ['a', 'b'])
        self.assertEqual(pool.get_skip_layers(), ['s'])
        self.assertEqual(pool.get_layer_type(), {'a': 'Conv2d'})
        self.assertEqual(pool.get_layer_name(), 'a')
        self.assertEqual(pool.get_wts_algo(), 'arq_quantize')
        self.assertEqual(pool.get_act_algo(), 'ifmr')
        pool.clear()
        self.assertIsNone(pool.get_quant_layers())
        self.assertIsNone(pool.get_act_algo())


class TestContainerFields(unittest.TestCase):
    def test_skip_fusion_layers_field(self):
        F.PARAM_POOL.clear()
        F.PARAM_POOL.set_layer_type({'conv1': 'Conv2d', 'relu1': 'ReLU'})
        f = _f(F.SkipFusionLayersField)
        self.assertRaises(TypeError, f.check, 'skip', 'notalist')
        self.assertRaises(TypeError, f.check, 'skip', [1])
        self.assertRaises(ValueError, f.check, 'skip', ['ghost'])   # not in graph
        self.assertRaises(ValueError, f.check, 'skip', ['relu1'])   # type not fusible
        f.check('skip', ['conv1'])
        F.PARAM_POOL.clear()

    def test_act_quant_params_conflict(self):
        f = _f(F.ActQuantParamsField)
        # ifmr-only and hfmg-only params at same time
        self.assertRaises(ValueError, f.check, 'act',
                          {'act_algo': 'ifmr', 'num_of_bins': 1024,
                           'search_range': [0.7, 1.3]})

    def test_wgt_quant_params_invalid_param(self):
        f = _f(F.WgtQuantParamsField)
        # reg_param is not valid for arq_quantize
        self.assertRaises(ValueError, f.check, 'wgt',
                          {'wts_algo': 'arq_quantize', 'reg_param': 0.01})

    def test_tensor_quantize_field_type(self):
        f = _f(F.TensorQuantizeField)
        self.assertRaises(TypeError, f.check, 'tq', 'notalist')
        self.assertEqual(f.init_container(), [])
        self.assertEqual(f.fill_default(None), None)


if __name__ == '__main__':
    unittest.main()
