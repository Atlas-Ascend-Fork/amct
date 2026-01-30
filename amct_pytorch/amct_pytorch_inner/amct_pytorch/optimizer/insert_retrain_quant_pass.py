#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

Insert some mudule about retrain quantization.

"""
import torch # pylint: disable=E0401

from amct_pytorch.common.utils.vars_util import RNN_LAYER_TYPE
from amct_pytorch.configuration.retrain_config import RetrainConfig as \
    Configuration
from amct_pytorch.optimizer.base_fusion_pass import BaseFusionPass
from amct_pytorch.custom_op.retrain_quant import RetrainQuant
from amct_pytorch.custom_op.rnn_retrain_quant import RNNRetrainQuant
from amct_pytorch.utils.model_util import ModuleHelper
from amct_pytorch.utils.vars import RETRAIN_ONNX_TYPES
from amct_pytorch.utils.log import LOGGER


class InsertRetrainQuantPass(BaseFusionPass):
    """
    Function: Insert some mudule about retrain quantization.
    APIs: match_pattern, do_pass
    """
    def __init__(self, torch_recorder, device='cpu'):
        """
        Function: init object
        Parameter: None
        Return: None
        """
        BaseFusionPass.__init__(self)
        self.conf = Configuration()
        self.record_module = torch_recorder
        self.device = device

    def match_pattern(self, node):
        """
        Function: Match the node to be retrain quantized in graph
        Parameters: node: node in graph to be matched
        Return: True: matched
                False: mismatch
        """
        if node.type not in RETRAIN_ONNX_TYPES:
            return False
        if not self.conf.retrain_enable(node.name):
            return False

        return True

    def do_pass(self, graph, object_node, model=None):
        """
        Function: Insert some mudule about retrain quantization.
        Parameters: graph: graph structure
                    object_node: node to process
                    model: torch.nn.Module, the model to be modified. if it's
                        None, the gaph will be modified.
        Return: None
        """
        model_helper = ModuleHelper(model)
        object_node_name = object_node.name
        object_module = model_helper.get_module(object_node_name)
        parent_module = model_helper.get_parent_module(object_node_name)

        common_config = object_module.common_config

        is_next_node_bn = False
        bn_module = None
        bn_module_name = None
        if len(object_node.output_anchors) == 1:
            output_anchor = object_node.get_output_anchor(0)
            if len(output_anchor.get_peer_input_anchor()) == 1:
                next_node = output_anchor.get_peer_input_anchor()[0].node
                if next_node.type == 'BatchNormalization' and \
                    object_node.type == 'Conv':
                    is_next_node_bn = True
                    bn_module_name = next_node.name
                    bn_module = model_helper.get_module(bn_module_name)
        if object_node.type in RNN_LAYER_TYPE:
            new_module = RNNRetrainQuant(object_module, self.record_module)
        else:
            new_module = RetrainQuant(object_module, self.record_module, bn_module, bn_module_name)
        setattr(parent_module, object_node_name.split('.')[-1], new_module)
        if is_next_node_bn:
            bn_parent_module = model_helper.get_parent_module(next_node.name)
            setattr(bn_parent_module, next_node.name.split('.')[-1],
                    torch.nn.Identity())

        LOGGER.logd(
            "Insert RetrainQuant module to '{}' "
            "success!".format(object_node.name), 'InsertRetrainQuantPass')
