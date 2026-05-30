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
from unittest.mock import MagicMock

from amct_pytorch.classic.graph_based.amct_pytorch.capacity import CAPACITY
from amct_pytorch.classic.graph_based.amct_pytorch.common.config.config_base import (
    ConfigBase,
    GraphObjects,
)

logger = logging.getLogger(__name__)

ACT = 'activation_quant_params'
WGT = 'weight_quant_params'
GRAPH = object()


def _cfg(enable_quant=True, enable_approximate=False):
    querier = MagicMock()
    go = GraphObjects(graph_querier=querier, graph_checker=MagicMock())
    obj = ConfigBase(go, CAPACITY, enable_quant=enable_quant,
                     enable_approximate=enable_approximate)
    return obj, querier


class TestConfigBaseChecks(unittest.TestCase):
    def test_check_skip_layers_ok(self):
        obj, q = _cfg()
        q.get_name_type_dict.return_value = {'conv1': 'Conv2d', 'relu1': 'ReLU'}
        obj.check_skip_layers(GRAPH, ['conv1'])

    def test_check_skip_layers_not_in_graph(self):
        obj, q = _cfg()
        q.get_name_type_dict.return_value = {'conv1': 'Conv2d'}
        self.assertRaises(ValueError, obj.check_skip_layers, GRAPH, ['ghost'])

    def test_check_skip_layers_type_unsupported(self):
        obj, q = _cfg()
        q.get_name_type_dict.return_value = {'relu1': 'ReLU'}
        self.assertRaises(ValueError, obj.check_skip_layers, GRAPH, ['relu1'])

    def test_check_skip_layers_approximate_branch(self):
        obj, q = _cfg(enable_quant=False, enable_approximate=True)
        q.get_name_type_dict.return_value = {'conv1': 'Conv2d'}
        # empty skip list reaches approximation branch without the final loop
        obj.check_skip_layers(GRAPH, [])

    def test_check_skip_types_unsupported(self):
        obj, _ = _cfg()
        self.assertRaises(ValueError, obj.check_skip_types, ['NotAType'])

    def test_check_skip_types_empty_result(self):
        obj, _ = _cfg()
        all_types = list(obj.quantizable_type)
        self.assertRaises(ValueError, obj.check_skip_types, all_types)

    def test_get_supported_layers_none(self):
        obj, q = _cfg()
        q.get_support_quant_layers.return_value = []
        self.assertRaises(ValueError, obj.get_supported_layers, GRAPH)

    def test_get_supported_layers_global_name(self):
        obj, q = _cfg()
        q.get_support_quant_layers.return_value = ['version']
        self.assertRaises(ValueError, obj.get_supported_layers, GRAPH)

    def test_get_supported_layers_approximate(self):
        obj, q = _cfg(enable_quant=False, enable_approximate=True)
        q.get_support_approximate_layers.return_value = ['op1']
        self.assertEqual(obj.get_supported_layers(GRAPH), ['op1'])

    def test_check_ada_quantize_layers(self):
        obj, q = _cfg()
        q.get_ada_quant_layers.return_value = []
        config = {'l1': {WGT: {'wts_algo': 'ada_quantize'}}}
        obj.check_ada_quantize_layers(GRAPH, config, ['l1'])
        self.assertNotIn(WGT, config['l1'])

    def test_check_common_config_dmq_layers(self):
        obj, q = _cfg()
        q.get_name_type_dict.return_value = {'l1': 'Conv2d'}
        q.get_support_quant_layers.return_value = ['l1']
        q.get_support_dmq_balancer_layers.return_value = []
        obj.set_param_pool(['l1'], GRAPH)
        # no quant layer supports dmq_balancer -> warning branch
        obj.check_common_config_dmq_layers(GRAPH)

    def test_check_int16_quantize_layers_downgrade(self):
        obj, q = _cfg()
        q.get_support_int16_quantizable_layers.return_value = []
        config = {'l1': {ACT: {'num_bits': 16}}}
        obj.check_int16_quantize_layers(GRAPH, config, ['l1'])
        self.assertEqual(config['l1'][ACT]['num_bits'], 8)

    def test_check_and_down_grade_winograd_num_bits(self):
        obj, q = _cfg()
        q.get_support_winograd_quant_layers.return_value = []
        config = {'l1': {WGT: {'num_bits': 6}}}
        obj.check_and_down_grade_winograd_num_bits(GRAPH, config, ['l1'])
        self.assertEqual(config['l1'][WGT]['num_bits'], 8)

    def test_check_activation_symmetric_valid_raises(self):
        obj, q = _cfg()
        q.get_act_symmetric_limit_layers.return_value = ['l1']
        config = {ACT: {'asymmetric': True}}
        self.assertRaises(ValueError, obj.check_activation_symmetric_valid,
                          GRAPH, config, 'l1')

    def test_check_int16_quantize_valid_raises(self):
        obj, q = _cfg()
        q.get_support_int16_quantizable_layers.return_value = []
        self.assertRaises(ValueError, obj.check_int16_quantize_valid, GRAPH, 'l1')

    def test_set_skip_layers(self):
        obj, q = _cfg()
        q.get_skip_quant_layers.return_value = ['s1']
        obj.set_skip_layers(GRAPH)

    def test_set_param_pool_approximate(self):
        obj, q = _cfg(enable_quant=False, enable_approximate=True)
        q.get_support_approximate_layers.return_value = ['op1']
        q.get_name_type_dict.return_value = {'op1': 'Conv2d'}
        obj.set_param_pool(['op1'], GRAPH)

    def test_check_quant_tensor_valid(self):
        obj, _ = _cfg()
        # graph_checker is a MagicMock, so check_tensor_quant exists and is called
        obj.check_quant_tensor_valid(GRAPH, [{'layer_name': 'x'}])

    def test_create_quant_config_no_layer_quant(self):
        obj, q = _cfg()
        q.get_name_type_dict.return_value = {'l1': 'Conv2d'}
        q.get_support_quant_layers.return_value = ['l1']
        self.assertRaises(ValueError, obj.create_quant_config,
                          '/tmp/x.json', GRAPH, skip_layers=['l1'])


if __name__ == '__main__':
    unittest.main()
