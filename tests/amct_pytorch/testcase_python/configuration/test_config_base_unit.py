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
from types import SimpleNamespace
from unittest.mock import MagicMock

from amct_pytorch.classic.graph_based.amct_pytorch.capacity import CAPACITY
from amct_pytorch.classic.graph_based.amct_pytorch.common.config.config_base import (
    ConfigBase,
    GraphObjects,
    check_config_quant_enable,
    check_config_dmq_balancer,
)
from amct_pytorch.classic.graph_based.amct_pytorch.common.config.proto_config import (
    ProtoConfig,
)

logger = logging.getLogger(__name__)

ACT = 'activation_quant_params'
WGT = 'weight_quant_params'


def _make_config(enable_quant=True, enable_approximate=False, querier=None):
    querier = querier or MagicMock()
    go = GraphObjects(graph_querier=querier, graph_checker=MagicMock())
    return ConfigBase(
        go, CAPACITY, enable_quant=enable_quant, enable_approximate=enable_approximate
    )


class TestConfigBaseStatic(unittest.TestCase):
    def test_init_conflict_raises(self):
        self.assertRaises(RuntimeError, ConfigBase, None, None, True, True)

    def test_init_approximate_tree(self):
        cfg = _make_config(enable_quant=False, enable_approximate=True)
        self.assertFalse(cfg.enable_quant)
        self.assertTrue(cfg.enable_approximate)

    def test_get_common_activation_quant_config_empty(self):
        self.assertEqual(
            ConfigBase.get_common_activation_quant_config(None), [None, None]
        )

    def test_get_common_activation_quant_config_value(self):
        common = {ACT: {'act_algo': 'ifmr', 'asymmetric': True}}
        ret = ConfigBase.get_common_activation_quant_config(common)
        self.assertEqual(ret, ['ifmr', True])

    def test_set_global_asymmetric_match(self):
        algo = 'ifmr'
        config = {'act_algo': algo}
        ConfigBase.set_global_asymmetric(config, [algo, True], False)
        self.assertTrue(config.get('asymmetric'))

    def test_set_global_asymmetric_fallback(self):
        config = {'act_algo': 'ifmr'}
        # asymmetric is None -> falls back to activation_offset
        ConfigBase.set_global_asymmetric(config, ['hfmg', None], True)
        self.assertTrue(config.get('asymmetric'))

    def test_add_global_to_layer(self):
        quant_config = {
            'batch_num': 4,
            'fakequant_precision_mode': 'FORCE_FP16',
            'layer1': {
                ACT: {
                    'act_algo': 'ifmr',
                    'search_range': [0.7, 1.3],
                    'asymmetric': None,
                },
                WGT: {},
            },
        }
        ConfigBase.add_global_to_layer(
            quant_config, num_bits=8, wts_algo='arq_quantize'
        )
        layer = quant_config.get('layer1')
        act = layer.get(ACT)
        wgt = layer.get(WGT)
        self.assertEqual(act.get('search_range_start'), 0.7)
        self.assertEqual(act.get('search_range_end'), 1.3)
        self.assertNotIn('search_range', act)
        self.assertEqual(act.get('batch_num'), 4)
        self.assertEqual(act.get('fakequant_precision_mode'), 'FORCE_FP16')
        self.assertEqual(wgt.get('num_bits'), 8)
        self.assertEqual(wgt.get('wts_algo'), 'arq_quantize')

    def test_legacy_fc_config_still_applies_to_linear(self):
        proto_config = ProtoConfig.__new__(ProtoConfig)
        proto_config.proto_config = SimpleNamespace(
            conv_calibration_config=object(), fc_calibration_config=object()
        )
        proto_config.quantizable_type = ['Conv2d', 'Linear']
        proto_config.channel_wise_types = ['Conv2d', 'Linear']
        proto_config._get_global_config = MagicMock(return_value={})
        proto_config._get_override_layer_configs = MagicMock(return_value={})
        proto_config._get_override_layer_types = MagicMock(return_value={})
        proto_config._get_common_config = MagicMock(return_value={})
        proto_config._get_conv_calibration_config = MagicMock(
            return_value={'config': 'conv'}
        )
        proto_config._get_fc_calibration_config = MagicMock(
            return_value={'config': 'fc'}
        )

        config = proto_config.get_proto_config()

        self.assertEqual(config.type_config['Linear'], {'config': 'fc'})


class TestConfigBaseModuleFuncs(unittest.TestCase):
    def test_check_config_quant_enable_ok(self):
        check_config_quant_enable({'l1': {'quant_enable': True}})

    def test_check_config_quant_enable_tensor(self):
        check_config_quant_enable({'tensor_quantize': [{'x': 1}], 'v': 1})

    def test_check_config_quant_enable_raises(self):
        self.assertRaises(
            RuntimeError,
            check_config_quant_enable,
            {'l1': {'quant_enable': False}, 'scalar': 1},
        )

    def test_check_config_dmq_balancer_ok(self):
        check_config_dmq_balancer({'l1': {'dmq_balancer_param': {'a': 1}}})

    def test_check_config_dmq_balancer_raises(self):
        self.assertRaises(
            RuntimeError,
            check_config_dmq_balancer,
            {'l1': {'quant_enable': True}, 'scalar': 1},
        )


if __name__ == '__main__':
    unittest.main()
