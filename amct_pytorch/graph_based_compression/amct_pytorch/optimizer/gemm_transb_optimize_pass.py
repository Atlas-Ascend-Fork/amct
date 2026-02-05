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
import numpy as np

from ...amct_pytorch.optimizer.base_fusion_pass import BaseFusionPass

from ...amct_pytorch.common.utils.onnx_node_util import AttributeProtoHelper
from ...amct_pytorch.utils.log import LOGGER
from ...amct_pytorch.utils.quant_node import QuantOpInfo
from ...amct_pytorch.utils.onnx_initializer_util import TensorProtoHelper


class GemmTransBOptimizePass(BaseFusionPass):
    """
    Function: If gemm transB is true, do transpose offline, and set it to false
    APIs: match_pattern, do_pass
    """
    def __init__(self, records):
        """
        Function: init object
        Parameter:
            records: dict including quant factors such as scale_w
        Return: None
        """
        BaseFusionPass.__init__(self)
        self.records = records

    def match_pattern(self, node):
        """
        Function: Match pattern of node to do gemm transb optimize
        Parameters: node: node in graph to be matched
        Return: True: matched
                False: mismatch
        """
        if node.type != 'Gemm' or node.name not in self.records:
            return False
        attr_helper = AttributeProtoHelper(node.proto)
        if not attr_helper.has_attr('transB') or \
                attr_helper.get_attr_value('transB') != 1:
            return False

        return True

    def do_pass(self, graph, object_node, model=None):
        """
        Function: Do actually gemm transb optimize.
        Parameters:
            graph: graph structure
            object_node: node to process
            model: torch.nn.Module, the model to be modified. if it's
                None, the gaph will be modified.
        Return: None
        """
        if len(object_node.input_anchors) < 2:
            LOGGER.logd('Cannot find weights of "%s".' % (object_node.name),
                        'GemmTransBOptimizePass')
            return
        weights_param = QuantOpInfo.get_weight_node(object_node)
        weight_helper = TensorProtoHelper(weights_param.proto, weights_param.model_path)

        # get data
        weights = weight_helper.get_data()
        weight_helper.clear_data()

        # do transpose
        if weights.ndim != 2:
            raise RuntimeError(
                'The shape of onnx Gemm operator\'s input '
                'tensor B should be (K, N) if transB is 0, or (N, K) if '
                'transB is non-zero, but got from "{}" is {}'.format(
                    object_node.name, weights.shape))
        weights = np.transpose(weights, (1, 0))
        weight_helper.set_data(weights.flatten(),
                               type_string='FLOAT',
                               dims=weights.shape)

        attr_helper = AttributeProtoHelper(object_node.proto)
        attr_helper.set_attr_value('transB', 'INT', 0)
