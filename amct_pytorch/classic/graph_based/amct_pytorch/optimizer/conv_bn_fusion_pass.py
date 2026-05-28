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
import numpy as np
from onnx import onnx_pb

from ...amct_pytorch.optimizer.base_module_fusion_pass import BaseModuleFusionPass
from ...amct_pytorch.utils.log import LOGGER
from ...amct_pytorch.utils.model_util import ModuleHelper
from ...amct_pytorch.utils.quant_node import QuantOpInfo
from ...amct_pytorch.utils.onnx_initializer_util import TensorProtoHelper
from ...amct_pytorch.common.utils.util import version_higher_than

LOW_TORCH_VERSION = '1.6.0'
FLOAT = 'FLOAT'
FLOAT16 = 'FLOAT16'


class ConvBnFusionPass(BaseModuleFusionPass):
    """
    Function: Do "Conv2d" and "BatchNorm2d" fusion operation
    APIs: match_pattern, do_pass
    """
    def __init__(self, config=None, record_helper=None):
        """
        Function: init object
        Parameter: None
        Return: None
        """
        BaseModuleFusionPass.__init__(self)

        if config:
            self.config = config()
        self.structure = {}
        self.record_helper = record_helper

    @staticmethod
    def is_fusionable_conv(conv_module, conv_name):
        """
        Function: check if the conv module can be fused
        Parameters:
        conv_module: conv module
        conv_name: name of the module
        Return: bool value which specifies wether the module can be fused
        """
        if type(conv_module).__name__ not in ('Conv2d', 'Conv3d', 'Conv1d'):
            return False
        if conv_module.padding_mode != 'zeros':
            LOGGER.logd('Can only do Conv+BN fusion of Conv that padding_mode '
                        '= "zeros", but "%s" is "%s"' % (
                            conv_name, conv_module.padding_mode),
                        'ConvBnFusionPass')
            return False
        return True

    @staticmethod
    def set_fused_data_back(fused_conv_module, conv_node, graph):
        """
        Function:
            After fusing "Conv" and "BatchNorm" layer, the fused weight and bias
            are set back to the graph
        Parameters: 
            fused_conv_module: torch.nn.Module, the fused module contains the fused data.
            conv_node: the ori conv node in the graph
            graph: graph structure
        Return: None
        """
        fused_weight_raw = fused_conv_module.weight.cpu().detach().numpy().flatten()
        weight_param = QuantOpInfo.get_weight_node(conv_node)
        weight_helper = TensorProtoHelper(weight_param.proto, weight_param.model_path)
        # set weights data to graph
        weight_helper.clear_data()
        if fused_conv_module.weight.dtype == torch.float16:
            data_type = FLOAT16
        else:
            data_type = FLOAT
        weight_helper.set_data(fused_weight_raw, data_type)

        # set bias data to graph
        fused_bias_tensor = fused_conv_module.bias.data
        fused_bias_raw = fused_bias_tensor.cpu().detach().numpy().flatten()
        bias_index = QuantOpInfo.get_quant_index(conv_node).get('bias_index')
        if len(conv_node.input_anchors) <= bias_index:
            # bias is None
            bias_tensor = onnx_pb.TensorProto()
            bias_tensor.name = f'{conv_node.name}.bias'
            bias_helper = TensorProtoHelper(bias_tensor)
            bias_helper.set_data(fused_bias_raw,
                                 type_string=data_type,
                                 dims=[fused_bias_tensor.numel()])
            conv_bias = graph.add_node(bias_tensor)
            conv_node.add_input_anchor(f'{conv_node.name}.sub_module.bias')
            graph.add_edge(conv_bias, 0, conv_node, 2)
        else:
            bias_input_anchor = conv_node.get_input_anchor(bias_index)
            bias_param = bias_input_anchor.get_peer_output_anchor().node
            bias_helper = TensorProtoHelper(bias_param.proto, bias_param.model_path)
            bias_helper.clear_data()
            bias_helper.set_data(fused_bias_raw, data_type)

    def match_pattern(self, module, name, graph=None):
        """
        Function:Match the bn module to be fused in model
        Parameters:
            module: module to be matched
            name: module's name
            graph: graph structure
        Return: True: matched
                False: mismatch
        """
        # limit mod type and status
        if not isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
            return False
        if isinstance(module, (torch.nn.BatchNorm3d, torch.nn.BatchNorm1d)) and \
            torch.__version__ != LOW_TORCH_VERSION and \
            not version_higher_than(torch.__version__, LOW_TORCH_VERSION):
            return False
        if not module.affine or not module.track_running_stats:
            LOGGER.logd(
                'Cannot do conv + bn:\'{}\' fuison for affine and ' \
                'track_running_stats must be True!'.format(name),
                'ConvBnFusionPass')
            return False

        # limit graph link
        try:
            bn_node = graph.get_node_by_name(name)
        except RuntimeError:
            LOGGER.logd('Cannot find module "%s" in graph, means it not used '
                        'in forward.' % (name))
            return False

        if len(bn_node.input_anchors) != 5:
            raise RuntimeError('BatchNorm node should only have 5 input ' \
                'actually have {}'.format(len(bn_node.input_anchors)))

        peer_out_anchor = bn_node.get_input_anchor(0).get_peer_output_anchor()
        conv_node = peer_out_anchor.node

        # return false if moudle is in training mode
        if module.training:
            LOGGER.logw(
                'Cannot do conv:\'{}\' + bn:\'{}\' fuison for training must be False! ' \
                'Conv + bn fusion process will be skipped.'.format(conv_node.name, name), \
                'ConvBnFusionPass')
            return False

        # If do Conv + BN fusion, conv must can only output to bn
        if len(conv_node.output_anchors) != 1 or \
            len(conv_node.get_output_anchor(0).get_peer_input_anchor()) != 1:
            return False

        skip_fusion_layers = [] if self.record_helper is not None else self.config.get_skip_fusion_layers()
        if conv_node.type != 'Conv' or \
            conv_node.name in skip_fusion_layers:
            return False

        self.structure[name] = {'conv': conv_node.name}
        return True

    def do_pass(self, model, object_module, object_name, graph=None):
        """
        Function: Do actual "Convolution" layer and "BatchNorm" layer
                  fusion operation
        Parameters: model: torch.nn.Module, the model to be modified.
                    object_module: module to process
                    object_name: name of object_module
                    graph: graph structure
        Return: None
        """
        conv_name = self.structure.get(object_name).get('conv')
        conv_node = graph.get_node_by_name(conv_name)
        # conv and bn must be in eval state
        try:
            conv_module = ModuleHelper(model).get_module(conv_name)
        except RuntimeError:
            LOGGER.logd('Cannot find "%s" in model, cannot do Conv+BN '
                        'fuison.' % (conv_name))
            return
        if not self.is_fusionable_conv(conv_module, conv_name):
            return
        checked_model = torch.quantization.fuse_modules(model, [[conv_name, object_name]])
        checked_module = ModuleHelper(checked_model).get_module(conv_name)
        if torch.isinf(checked_module.weight).any() or torch.isnan(checked_module.weight).any() or \
            torch.isinf(checked_module.bias).any() or torch.isnan(checked_module.bias).any():
            LOGGER.logw(f'Conv+BN fusion for {conv_name} skipped because data is abnormal after fusion.')
            return
        # delete bn in graph
        _delete_bn_from_graph(graph, conv_name, object_name)

        # fuse "bn + conv" in model
        torch.quantization.fuse_modules(model, [[conv_name, object_name]], inplace=True)

        # if do fusion after qat, update quant factors in record
        if self.record_helper is not None and self.record_helper.has_key(conv_name):
            self._update_record_for_fusion(conv_node, object_module)
            LOGGER.logd(
            'Update record for conv:\'{}\' + bn:\'{}\' fuison!'.format(
                conv_name, object_name), 'ConvBnFusionPass')

        # Set fused weights and bias back
        fused_conv_module = ModuleHelper(model).get_module(conv_name)
        self.set_fused_data_back(fused_conv_module, conv_node, graph)

        LOGGER.logd(
            'Do conv:\'{}\' + bn:\'{}\' fuison success!'.format(conv_name, object_name), 'ConvBnFusionPass')

    def _update_record_for_fusion(self, conv_node, bn_module):
        """
        Function: update quant factor function for conv_bn fusion
        Parameters:
            conv_node: conv node
            bn_module: bn module
        Return: None
        """
        scale_w, offset_w = self.record_helper.read_weights_scale_offset(conv_node.name)
        if len(scale_w) != len(offset_w):
            raise RuntimeError('scale_w must be same length with offset_w.')
        # if channel-wise False & BN. scale_w and offset_w expand.
        if len(scale_w) == 1:
            scale_w = np.ones((bn_module.num_features), np.float32) * scale_w[0]
            offset_w = np.ones((bn_module.num_features), np.int8) * offset_w[0]
        scale_w = torch.tensor(scale_w, requires_grad=False)

        bn_var_rsqrt = torch.rsqrt(bn_module.running_var + bn_module.eps)
        scale = bn_module.weight * bn_var_rsqrt
        fused_scale_w = scale_w.to(scale.device).abs() * abs(scale).detach()
        self.record_helper.record_weights_scale_offset(
            conv_node.name, fused_scale_w.cpu().tolist(), offset_w)


def _delete_bn_from_graph(graph, conv_name, bn_name):
    """Function: delete bn from graph"""
    # Step1: remove edge of BatchNorm
    conv_node = graph.get_node_by_name(conv_name)
    bn_node = graph.get_node_by_name(bn_name)
    node_scale = bn_node.get_input_anchor(1).get_peer_output_anchor().node
    node_b = bn_node.get_input_anchor(2).get_peer_output_anchor().node
    node_mean = bn_node.get_input_anchor(3).get_peer_output_anchor().node
    node_var = bn_node.get_input_anchor(4).get_peer_output_anchor().node
    # remove input links
    graph.remove_edge(conv_node, 0, bn_node, 0)
    graph.remove_edge(node_scale, 0, bn_node, 1)
    graph.remove_edge(node_b, 0, bn_node, 2)
    graph.remove_edge(node_mean, 0, bn_node, 3)
    graph.remove_edge(node_var, 0, bn_node, 4)
    # remove output links
    bn_peer_in_anchors = \
        bn_node.get_output_anchor(0).get_peer_input_anchor()
    bn_peer_input_nodes = []
    bn_peer_input_indexes = []
    for input_anchor in bn_peer_in_anchors:
        bn_peer_input_nodes.append(input_anchor.node)
        bn_peer_input_indexes.append(input_anchor.index)
    for index, element in enumerate(bn_peer_input_nodes):
        graph.remove_edge(bn_node, 0,
                          element, bn_peer_input_indexes[index])

    # Step2: Add link from conv to bn_peer_input_node
    for index, element in enumerate(bn_peer_input_nodes):
        graph.add_edge(conv_node, 0, element, bn_peer_input_indexes[index])
        if element.type == 'graph_anchor':
            conv_node.get_output_anchor(0).set_name(element.name)

    # Step3: Remove node from graph
    graph.remove_node(bn_node)

    # Step4: Remove initializer from graph
    graph.remove_initializer(node_scale)
    graph.remove_initializer(node_b)
    graph.remove_initializer(node_mean)
    graph.remove_initializer(node_var)
