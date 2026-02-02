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
from ...amct_pytorch.utils.log import LOGGER
from ...amct_pytorch.utils.onnx_initializer_util import TensorProtoHelper
from ...amct_pytorch.utils.weight_quant_api import get_deconv_group
from ...amct_pytorch.utils.weight_quant_api import adjust_deconv_weight_shape


class ReplaceWeightQuantPass(BaseFusionPass):
    """
    Function: Fakequant weight from int8 to int9
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
        Function: Match pattern of node to be quantized in graph
        Parameters: node: node in graph to be matched
        Return: True: matched
                False: mismatch
        """
        if node.type != 'AscendWeightQuant':
            return False

        return True

    def do_pass(self, graph, object_node, model=None):
        """
        Function: Do actual quantization and node's weight is changed to int9.
        Parameters:
            graph: graph structure
            object_node: node to process
            model: torch.nn.Module, the model to be modified. if it's
                None, the gaph will be modified.
        Return: None
        """
        # remove input links
        weight_node = object_node.get_input_anchor(0).get_peer_output_anchor().node
        offset_node = object_node.get_input_anchor(1).get_peer_output_anchor().node
        graph.remove_edge(weight_node, 0, object_node, 0)
        graph.remove_edge(offset_node, 0, object_node, 1)

        # remove output links
        peer_input_anchors = []
        # input_anchors of object_node
        for input_anchor in object_node.get_output_anchor(0).get_peer_input_anchor():
            peer_input_anchors.append(input_anchor)

        for input_anchor in peer_input_anchors:
            peer_input_node = input_anchor.node
            input_index = input_anchor.index
            graph.remove_edge(object_node, 0, peer_input_node, input_index)
            graph.add_edge(weight_node, 0, peer_input_node, input_index)

        if weight_node.type == 'Transpose':
            weight_node = weight_node.get_input_anchor(0).get_peer_output_anchor().node
        weight_helper = TensorProtoHelper(weight_node.proto, weight_node.model_path)
        offset_helper = TensorProtoHelper(offset_node.proto.attribute[0].t, offset_node.model_path)
        # get data
        int8_weight = weight_helper.get_data()
        int8_offset = offset_helper.get_data()
        weight_helper.clear_data()
        offset_helper.clear_data()

        for input_anchor in peer_input_anchors:
            peer_input_node = input_anchor.node
            if peer_input_node.type == "ConvTranspose" and get_deconv_group(peer_input_node) > 1:
                group = get_deconv_group(peer_input_node)
                if len(peer_input_anchors) > 1:
                    raise RuntimeError("Cannot replace weight quant layer '{}' to fake weight quant layer"\
                        "in the case of group > 1 weight sharing of ConvTranspose".format(object_node.name))
                else:
                    int8_weight = adjust_deconv_weight_shape(group, int8_weight)
                    trans_axes = (1, 0, 2, 3, 4)[:len(int8_weight.shape)]
                    int8_weight = np.transpose(int8_weight, trans_axes)

        int9_weight = int8_weight.astype(np.float32) - \
                      int8_offset.astype(np.float32)

        for input_anchor in peer_input_anchors:
            peer_input_node = input_anchor.node
            if peer_input_node.type == "ConvTranspose" and get_deconv_group(peer_input_node) > 1:
                group = get_deconv_group(peer_input_node)
                int9_weight = np.transpose(int9_weight, (1, 0, 2, 3))
                int9_weight = adjust_deconv_weight_shape(group, int9_weight)

        int9_weight = int9_weight.reshape([-1])

        weight_helper.set_data(int9_weight, 'FLOAT')

        graph.remove_node(object_node)
        graph.remove_node(offset_node)

        LOGGER.logd(
            "Replace weight quant layer '{}' to fake weight quant layer success!".
                format(object_node.name), 'ReplaceWeightQuantPass')
