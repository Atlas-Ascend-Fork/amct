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
import torch
from ...amct_pytorch.optimizer.base_fusion_pass import BaseFusionPass
from ...amct_pytorch.utils.log import LOGGER
from ...amct_pytorch.utils.model_util import ModuleHelper


class DeleteLinearAddPass(BaseFusionPass):
    """
    Function: Delete 'Add' node in graph if 'Add' is from Linear.
    APIs: match_pattern, do_pass
    """
    def __init__(self):
        """
        Function: init object
        Parameter: None
        Return: None
        """
        BaseFusionPass.__init__(self)

    @staticmethod
    def match_pattern(node):
        """
        Function: Match the node to be quantized in graph
        Parameters: node: node in graph to be matched
        Return: True: matched / False: mismatch
        """
        if node.type != 'Add':
            return False

        input_types = []
        for input_anchor in node.input_anchors:
            producer, _ = node.get_producer(input_anchor.index)
            input_types.append(producer.type)
        if input_types.count('MatMul') != 1:
            return False

        return True

    @staticmethod
    def do_pass(graph, object_node):
        """
        Function: Do actually delete Add from torch.nn.Linear.
        Parameters:
        graph: graph structure
        object_node: node to process
        Return: None
        """
        LOGGER.logd("Doing: delete node {} in graph.".format(object_node.name),
                    'DeleteLinearAddPass')

        producer_name = ''
        input_anchor_index = 0
        for input_anchor in object_node.input_anchors:
            producer, _ = object_node.get_producer(input_anchor.index)
            if producer.type == 'MatMul':
                producer_name = producer.name
                input_anchor_index = input_anchor.index
                break

        model_helper = ModuleHelper(graph.model)
        producer_module = model_helper.get_module(producer_name)
        if not isinstance(producer_module, torch.nn.Linear):
            return
        if producer_module.bias is None:
            return

        graph.delete_node(object_node, input_anchor_index, 0)
        graph.remove_node(object_node)

        LOGGER.logd("Finished: delete node {} in graph.".format(object_node.name),
                    'DeleteLinearAddPass')
