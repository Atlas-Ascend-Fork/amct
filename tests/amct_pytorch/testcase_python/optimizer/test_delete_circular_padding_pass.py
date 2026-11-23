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
import unittest
from unittest.mock import MagicMock

from amct_pytorch.classic.graph_based.amct_pytorch.optimizer import (
    DeleteCircularPaddingPass,
)


class TestDeleteCircularPaddingPass(unittest.TestCase):
    def _make_conv_node(self, name):
        conv_node = MagicMock()
        conv_node.name = name
        conv_node.type = 'Conv'
        conv_node.has_attr = lambda key: key == 'padding_circle'
        conv_node.get_attr = lambda key: True
        return conv_node

    def test_do_pass_skips_when_no_pad_registered(self):
        """Conv with padding_circle but no registered pad ops skips safely."""
        pass_obj = DeleteCircularPaddingPass()
        conv_node = self._make_conv_node('conv_0')
        graph = MagicMock()

        self.assertIsNone(pass_obj.do_pass(graph, conv_node))

    def test_do_pass_skips_when_producer_all_registered(self):
        """No external padding producer found skips safely instead of crashing."""
        pass_obj = DeleteCircularPaddingPass()
        conv_node = self._make_conv_node('conv_0')
        graph = MagicMock()
        pad_node = MagicMock()
        producer = MagicMock()
        producer.name = 'pad_0'
        pad_node.get_producer.return_value = (producer, 0)
        graph.get_node_by_name.return_value = pad_node
        pass_obj.circular_pad_op_dict['conv_0'] = ['pad_0']

        self.assertIsNone(pass_obj.do_pass(graph, conv_node))


if __name__ == '__main__':
    unittest.main()
