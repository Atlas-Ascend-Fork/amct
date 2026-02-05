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

from ...amct_pytorch.common.utils.vars_util import RNN_LAYER_TYPE
from ...amct_pytorch.common.utils.vars_util import LSTM_OUTPUT_NUMS, GRU_OUTPUT_NUMS
from ...amct_pytorch.configuration.retrain_config import RetrainConfig as \
    Configuration
from ...amct_pytorch.optimizer.base_fusion_pass import BaseFusionPass
from ...amct_pytorch.utils.model_util import ModuleHelper
from ...amct_pytorch.utils.vars import RETRAIN_ONNX_TYPES
from ...amct_pytorch.utils.log import LOGGER

MEMBERS = 'members'


class ShareActCompPass(BaseFusionPass):
    """
    Function: Share the ActQAT layer and parameters.
    APIs: match_pattern, do_pass
    """
    def __init__(self):
        """
        Function: init object
        Parameter: None
        Return: None
        """
        BaseFusionPass.__init__(self)
        self.conf = Configuration()
        self.op_group_map = {}
        self.group_info = {}
        self.group_num = 0

    def match_pattern(self, node):
        """
        Function: Match pattern of single output multi reference structure
            in graph
        Parameters: node: node in graph to be matched
        Return: True: matched
                False: mismatch
        """
        if node.type not in RETRAIN_ONNX_TYPES:
            return False
        if not self.conf.retrain_enable(node.name):
            return False

        output_anchor = node.get_input_anchor(0).get_peer_output_anchor()
        output_num = 1
        if node.type == 'LSTM':
            output_num = LSTM_OUTPUT_NUMS
        elif node.type == 'GRU':
            output_num = GRU_OUTPUT_NUMS
        if len(output_anchor.get_peer_input_anchor()) <= output_num:
            return False

        lgroups = []
        for peer_input_anchor in output_anchor.get_peer_input_anchor():
            consumer = peer_input_anchor.node
            if consumer.type in RETRAIN_ONNX_TYPES and \
                self.conf.retrain_enable(consumer.name):
                for l_g in lgroups:
                    name = self.group_info.get(l_g).get(MEMBERS)[0]
                    # if already has group
                    if self.conf.get_quant_config()[consumer.name] ==\
                            self.conf.get_quant_config()[name]:
                        self.op_group_map[consumer.name] = l_g
                        self.group_info.get(l_g).get(MEMBERS).append(consumer.name)
                        break
                if consumer.name not in self.op_group_map.keys():
                    # build new group
                    self.group_num += 1
                    self.op_group_map[consumer.name] = self.group_num
                    self.group_info[self.group_num] = {}
                    self.group_info.get(self.group_num)['done'] = False
                    self.group_info.get(self.group_num)[MEMBERS] = [consumer.name]
                    lgroups.append(self.group_num)
        if len(self.group_info.get(self.op_group_map.get(node.name)).get(MEMBERS)) > 1:
            return True

        return False

    def do_pass(self, graph, object_node, model=None):
        """
        Function: Share the ActQAT layer and parameters.
        Parameters: graph: graph structure
                    object_node: node to process
                    model: torch.nn.Module, the model to be modified. if it's
                        None, the gaph will be modified.
        Return: None
        """
        object_node_name = object_node.name
        model_helper = ModuleHelper(model)
        object_module = model_helper.get_module(object_node_name)
        main_module_names = \
            self.group_info.get(self.op_group_map.get(object_node_name)).get(MEMBERS)
        for item in model.named_modules():
            if item[0] in main_module_names:
                main_module_name = item[0]
                break
        main_module = model_helper.get_module(main_module_name)
        if main_module != object_module:
            object_module.acts_comp_reuse = main_module
            del object_module.acts_clip_max
            del object_module.acts_clip_min
            del object_module.acts_clip_max_pre
            del object_module.acts_clip_min_pre
            del object_module.acts_scale
            del object_module.acts_offset
            if object_node.type in RNN_LAYER_TYPE:
                del object_module.acts_h_clip_max
                del object_module.acts_h_clip_min
                del object_module.acts_h_clip_max_pre
                del object_module.acts_h_clip_min_pre
                del object_module.acts_h_scale
                del object_module.acts_h_offset

        LOGGER.logd(
            "Share ActivationQAT module to '{}' "
            "success!".format(object_node.name), 'ShareActCompPass')
