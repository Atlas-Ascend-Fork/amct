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
from amct_pytorch.classic.graph_based.amct_pytorch.common.retrain_config import (
    retrain_config_base as rcb,
)

logger = logging.getLogger(__name__)

GRAPH = object()


def _make(enable_retrain=True, enable_prune=False):
    querier = MagicMock()
    checker = MagicMock()
    go = rcb.GraphObjects(graph_querier=querier, graph_checker=checker)
    obj = rcb.RetrainConfigBase(go, CAPACITY)
    obj.set_ability(enable_retrain, enable_prune)
    return obj, querier, checker


class TestRetrainConfigBase(unittest.TestCase):
    def test_init_defaults(self):
        obj, _, _ = _make()
        self.assertEqual(obj.prune_type, rcb.NO_PRUNE)
        self.assertTrue(obj.enable_retrain)
        self.assertFalse(obj.enable_prune)

    def test_set_ability(self):
        obj, _, _ = _make()
        obj.set_ability(False, True)
        self.assertFalse(obj.enable_retrain)
        self.assertTrue(obj.enable_prune)

    def test_get_supported_layers_qat_branch(self):
        obj, q, _ = _make()
        q.get_support_qat_layer2type.return_value = {'l1': 'Conv2d'}
        self.assertEqual(obj.get_supported_layers(GRAPH), {'l1': 'Conv2d'})

    def test_get_supported_layers_ptq_branch(self):
        obj, q, _ = _make()
        del q.get_support_qat_layer2type
        q.get_name_type_dict.return_value = {'l1': 'Conv2d', 'l2': 'ReLU'}
        q.get_support_quant_layers.return_value = ['l1', 'l2']
        # ReLU not in RETRAIN_TYPES -> filtered out
        self.assertEqual(obj.get_supported_layers(GRAPH), {'l1': 'Conv2d'})

    def test_check_quant_layers_valid_passthrough(self):
        obj, _, checker = _make()
        # strip all optional checker hooks -> returns input unchanged
        for attr in (
            'check_data_type',
            'check_gradient_op',
            'check_matmul_transpose',
            'check_quantize_placeholder',
        ):
            if hasattr(checker, attr):
                delattr(checker, attr)
        layers = ['l1', 'l2']
        self.assertEqual(obj.check_quant_layers_valid(GRAPH, layers), layers)

    def test_check_quant_layers_valid_with_hooks(self):
        obj, _, checker = _make()
        for attr in (
            'check_gradient_op',
            'check_matmul_transpose',
            'check_quantize_placeholder',
        ):
            if hasattr(checker, attr):
                delattr(checker, attr)
        checker.check_data_type.return_value = ['l1']
        self.assertEqual(obj.check_quant_layers_valid(GRAPH, ['l1', 'l2']), ['l1'])

    def test_create_default_config_prune_error(self):
        obj, _, _ = _make(enable_retrain=True, enable_prune=True)
        self.assertRaises(RuntimeError, obj.create_default_config, '/tmp/x.json', GRAPH)

    def test_create_default_config_no_retrain_error(self):
        obj, _, _ = _make(enable_retrain=False, enable_prune=False)
        self.assertRaises(RuntimeError, obj.create_default_config, '/tmp/x.json', GRAPH)

    def test_create_default_config_no_supported_layers(self):
        obj, q, _ = _make()
        q.get_support_qat_layer2type.return_value = {}
        self.assertRaises(ValueError, obj.create_default_config, '/tmp/x.json', GRAPH)

    def test_get_support_layers_retrain_only(self):
        obj, q, _ = _make()
        q.get_support_qat_layer2type.return_value = {'l1': 'Conv2d'}
        supported, retrain, prune = obj.get_support_layers(GRAPH)
        self.assertEqual(retrain, {'l1': 'Conv2d'})
        self.assertEqual(supported, {'l1': 'Conv2d'})
        self.assertIn(rcb.NO_PRUNE, prune)

    def test_set_config_by_graph_construct_no_hook(self):
        obj, _, checker = _make()
        if hasattr(checker, 'set_softmax_channelwise'):
            delattr(checker, 'set_softmax_channelwise')
        # no hook -> no-op, should not raise
        obj.set_config_by_graph_construct({}, GRAPH)


class TestDowngradeInt4UnpackableLayers(unittest.TestCase):
    def test_downgrade_unpackable_int4_layer_to_int8(self):
        obj, querier, _ = _make()
        # conv1 不支持沿 Cin pack -> 降级；conv2 支持 -> 保持 INT4
        querier.check_int4_cin_pack_supported.side_effect = lambda graph, layer: (
            layer != 'conv1'
        )
        config = {
            'conv1': {'retrain_weight_config': {'dst_type': 'INT4'}},
            'conv2': {'retrain_weight_config': {'dst_type': 'INT4'}},
            'version': 'v1',  # 非 dict 之外的全局项也应被安全跳过
        }
        obj.downgrade_int4_unpackable_layers(None, config)
        self.assertEqual(config['conv1']['retrain_weight_config']['dst_type'], 'INT8')
        self.assertEqual(config['conv2']['retrain_weight_config']['dst_type'], 'INT4')

    def test_non_int4_layer_untouched(self):
        obj, querier, _ = _make()
        querier.check_int4_cin_pack_supported.return_value = False
        config = {'fc': {'retrain_weight_config': {'dst_type': 'INT8'}}}
        obj.downgrade_int4_unpackable_layers(None, config)
        # 非 INT4 层即便 not supported 也不动
        self.assertEqual(config['fc']['retrain_weight_config']['dst_type'], 'INT8')

    def test_no_querier_method_is_noop(self):
        obj, _, _ = _make()
        # graph_querier 无 check_int4_cin_pack_supported 时直接返回，不报错
        obj.graph_querier = object()
        config = {'conv': {'retrain_weight_config': {'dst_type': 'INT4'}}}
        obj.downgrade_int4_unpackable_layers(None, config)
        self.assertEqual(config['conv']['retrain_weight_config']['dst_type'], 'INT4')


if __name__ == '__main__':
    unittest.main()
