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
from ...amct_pytorch.common.utils.vars_util import LSTM_ATTRS_LIMIT_MAP, GRU_ATTRS_LIMIT_MAP
from ...amct_pytorch.common.utils.vars_util import RNN_SEQ_LENS_INDEX, RNN_H_INDEX
from ...amct_pytorch.common.utils.vars_util import LSTM_C_INDEX, LSTM_P_INDEX
from ...amct_pytorch.common.utils.vars_util import LSTM_OUTPUT_NUMS, GRU_OUTPUT_NUMS
from ...amct_pytorch.utils.log import LOGGER
from ...amct_pytorch.utils.vars import ADA_ROUND_TYPES
from ...amct_pytorch.utils.vars import QUANTIZABLE_TYPES
from ...amct_pytorch.utils.vars import QUANTIZABLE_ONNX_TYPES
from ...amct_pytorch.utils.vars import INT16_QUANTIZABLE_TYPES
from ...amct_pytorch.utils.vars import INT16_QUANTIZABLE_ONNX_TYPES
from ...amct_pytorch.utils.vars import RETRAIN_TYPES
from ...amct_pytorch.utils.vars import RETRAIN_ONNX_TYPES
from ...amct_pytorch.utils.vars import DISTILL_TYPES
from ...amct_pytorch.utils.vars import AMCT_OPERATIONS
from ...amct_pytorch.utils.vars import PRUNABLE_TYPES
from ...amct_pytorch.utils.vars import PRUNABLE_ONNX_TYPES
from ...amct_pytorch.utils.vars import SELECTIVE_PRUNABLE_TYPES
from ...amct_pytorch.common.utils.onnx_node_util import AttributeProtoHelper
from ...amct_pytorch.utils.quant_node import QuantOpInfo
from ...amct_pytorch.utils.onnx_initializer_util import TensorProtoHelper
from ...amct_pytorch.utils.vars import QUANT_LAYER_SUFFIX
from ...amct_pytorch.utils.onnx_conv_util import OnnxConvUtil
from ...amct_pytorch.capacity import CAPACITY

SYMMETRIC_LIMIT_TYPES = ['Conv3d']


class GraphQuerier():
    '''provide some APIs to query the model'''
    @staticmethod
    def get_name_type_dict(graph):
        '''get all layer name to type dict'''
        layer_type = {}
        if graph.model is None:
            for node in graph.nodes:
                layer_type[node.name] = node.type
            LOGGER.logd("get name type dict from graph",
                        module_name="Configuration")
        else:
            for name, mod in graph.model.named_modules():
                layer_type[name] = type(mod).__name__
            LOGGER.logd("get name type dict from model",
                        module_name="Configuration")
        return layer_type

    @staticmethod
    def get_support_quant_layer2type(graph):
        '''return supported layer to type map in model'''
        layers = {}
        if graph.model is None:
            for node in graph.nodes:
                if GraphChecker.check_graph_quantize_type(node):
                    layers[node.name] = node.type
            LOGGER.logd("get support quant layer2type from graph",
                        module_name="Configuration")
        else:
            for name, mod in graph.model.named_modules():
                if GraphChecker.check_quantize_type(name, mod, graph):
                    layers[name] = type(mod).__name__
            LOGGER.logd("get support quant layer2type from model",
                        module_name="Configuration")
        return layers

    @staticmethod
    def get_support_qat_layer2type(graph):
        """
        Function: return supported retrain layer to type map in model.
        """
        layer2type = {}
        if graph.model is None:
            for node in graph.nodes:
                if GraphChecker.check_graph_retrain_type(node):
                    layer2type[node.name] = node.type
            LOGGER.logd("get support retrain layer2type from graph",
                        module_name="Configuration")
        else:
            for name, mod in graph.model.named_modules():
                if GraphChecker.check_retrain_type(name, mod, graph):
                    layer2type[name] = type(mod).__name__
            LOGGER.logd("get support retrain layer2type from model",
                        module_name="Configuration")
        return layer2type

    @staticmethod
    def get_support_distill_layer2type(model, graph=None):
        """
        Function: return supported distill layer to type map in model.
        """
        layer2type = {}
        for name, mod in model.named_modules():
            if GraphChecker.check_distill_type(name, mod, graph):
                layer2type[name] = type(mod).__name__
        LOGGER.logd("get support distill layer2type from model",
                    module_name="Configuration")
        return layer2type

    @staticmethod
    def get_ada_quant_layers(graph):
        '''get ada quant layers'''
        layers = []
        for name, mod in graph.model.named_modules():
            mod_type = type(mod).__name__
            if mod_type in ADA_ROUND_TYPES and GraphChecker.check_shared_type(mod_type, name, mod, graph):
                layers.append(name)
        return layers

    @staticmethod
    def get_support_quant_layers(graph):
        '''return supported quant layers in model'''
        layers = []
        if graph.model is None:
            for node in graph.nodes:
                if GraphChecker.check_graph_quantize_type(node):
                    layers.append(node.name)
            LOGGER.logd("get support quant nodes from graph",
                        module_name="Configuration")
        else:
            for name, mod in graph.model.named_modules():
                if GraphChecker.check_quantize_type(name, mod, graph):
                    layers.append(name)
            LOGGER.logd("get support quant modules from model",
                        module_name="Configuration")
        return layers

    @staticmethod
    def get_support_int16_quantizable_layers(graph):
        '''return supported int16 quantize layers in model'''
        layers = []
        if graph.model is None:
            for node in graph.nodes:
                if GraphChecker.check_graph_int16_quantize_type(node):
                    layers.append(node.name)
            LOGGER.logd("get support int16 quantize nodes from graph",
                        module_name="Configuration")
        else:
            for name, mod in graph.model.named_modules():
                if GraphChecker.check_int16_quantize_type(name, mod, graph):
                    layers.append(name)
            LOGGER.logd("get support int16 quantize modules from model",
                        module_name="Configuration")
        return layers


    @staticmethod
    def get_support_prune_layer2type(graph):
        '''
        Function: return supported layer to type map in model.
        Parameters: graph.
        Return: layers.
        '''
        layers = {}
        if graph.model is None:
            for node in graph.nodes:
                if GraphChecker.check_graph_prune_type(node):
                    layers[node.name] = node.type
            LOGGER.logd("get support prune layer2type from graph",
                        module_name="Configuration")
        else:
            for name, mod in graph.model.named_modules():
                if GraphChecker.check_prune_type(name, mod, graph):
                    layers[name] = type(mod).__name__
            LOGGER.logd("get support prune layer2type from model",
                        module_name="Configuration")
        return layers

    @staticmethod
    def get_support_selective_prune_layer2type(graph):
        '''return supported layer to type map in model'''
        layers = {}
        if graph.model is None:
            for node in graph.nodes:
                if GraphChecker.check_graph_selective_prune_type(node):
                    layers[node.name] = node.type
        else:
            for name, mod in graph.model.named_modules():
                if GraphChecker.check_selective_prune_type(name, mod, graph):
                    layers[name] = type(mod).__name__
        LOGGER.logd("get support selective prune layer2type from graph",
                    module_name="Configuration")
        return layers

    @staticmethod
    def check_op_matching(graph, fused_op_list):
        """
        Function: Check whether the ops in json the ops in the original graph
        Inputs:
            graph: Graph, to be quantized
            fused_op_list: list, the ops parsed from the json file
        """
        # check whether the ops in json the ops in the original graph
        original_graph_ids = graph.node_ids
        for json_op in fused_op_list:
            is_quant_op = False
            for quant_suffix in QUANT_LAYER_SUFFIX:
                if quant_suffix in json_op:
                    is_quant_op = True
            if is_quant_op:
                continue
            if json_op not in original_graph_ids:
                LOGGER.logd(
                    "Op '{}' in the given mapping_file does not exist in the original graph. "\
                    "The mapping_file may not match the original graph, please check!".format(json_op))

    @staticmethod
    def get_act_symmetric_limit_types():
        """
        Function: get type only support activation symmetric
        """
        types = SYMMETRIC_LIMIT_TYPES
        return types

    @staticmethod
    def get_act_symmetric_limit_layers(graph):
        '''get layer only support activation symmetric'''
        layers = []
        if graph.model is None:
            for node in graph.nodes:
                if node.type == 'Conv' and check_kernel_shape(node, [3]):
                    layers.append(node.name)
            LOGGER.logd("get act symmetric limit layers from graph",
                        module_name="Configuration")
        else:
            for name, mod in graph.model.named_modules():
                mod_type = type(mod).__name__
                if mod_type in SYMMETRIC_LIMIT_TYPES:
                    layers.append(name)
            LOGGER.logd("get act symmetric limit layers from model",
                        module_name="Configuration")

        return layers

    @staticmethod
    def get_support_dmq_balancer_types():
        '''get type only support dmq_balancer'''
        quant_types = QUANTIZABLE_TYPES + QUANTIZABLE_ONNX_TYPES
        quant_types.remove('AvgPool2d')
        quant_types.remove('AveragePool')

        return quant_types

    @staticmethod
    def get_support_dmq_balancer_layers(graph):
        '''get layers only support dmq_balancer'''
        layers = []
        if graph.model is None:
            for node in graph.nodes:
                if GraphChecker.check_graph_quantize_type(node) and node.type != 'AveragePool':
                    layers.append(node.name)
            LOGGER.logd("get support dmq_balancer nodes from graph",
                        module_name="Configuration")
        else:
            for name, mod in graph.model.named_modules():
                mod_type = type(mod).__name__
                if GraphChecker.check_quantize_type(name, mod, graph) and mod_type != 'AvgPool2d':
                    layers.append(name)
            LOGGER.logd("get support dmq_balancer modules from model",
                        module_name="Configuration")

        return layers

    @staticmethod
    def get_support_winograd_quant_layers(graph):
        """get layers support int6 int7 quant"""
        layers = list()
        if graph.model is None:
            for node in graph.nodes:
                # only conv2D support int6 int7 wts quant
                if node.type in ['Conv'] and check_kernel_shape(node, [2]):
                    layers.append(node.name)
        else:
            for name, mod in graph.model.named_modules():
                if type(mod).__name__ in ['Conv2d']:
                    layers.append(name)
        return layers

    @staticmethod
    def get_support_winograd_layer_types():
        """get layer types support int6 int7 quant"""
        return ['Conv2d']


class GraphChecker():
    """Check the model."""
    @staticmethod
    def check_retrain_type(mod_name, mod, graph=None):
        """
        Function: check if mod in model can be retrain or not.
        """
        mod_type = type(mod).__name__
        # check type
        if mod_type not in RETRAIN_TYPES:
            return False

        return GraphChecker.check_shared_type(mod_type, mod_name, mod, graph)

    @staticmethod
    def check_distill_type(mod_name, mod, graph=None):
        """
        Function: check if mod in model can be distill or not.
        """
        mod_type = type(mod).__name__
        if mod_type not in DISTILL_TYPES:
            return False

        return GraphChecker.check_shared_type(mod_type, mod_name, mod, graph)

    @staticmethod
    def check_quantize_type(mod_name, mod, graph=None):
        """ check if mod in model can be quantized or not."""
        mod_type = type(mod).__name__
        # check type
        if mod_type not in QUANTIZABLE_TYPES:
            return False

        return GraphChecker.check_shared_type(mod_type, mod_name, mod, graph)

    @staticmethod
    def check_int16_quantize_type(mod_name, mod, graph=None):
        """ check if mod in model can be quantized or not."""
        mod_type = type(mod).__name__
        # check type
        if mod_type not in INT16_QUANTIZABLE_TYPES:
            return False

        # only Gemm support int16
        if mod_type in ['Linear']:
            if mod.bias is None:
                return False

        return GraphChecker.check_shared_type(mod_type, mod_name, mod, graph)

    @staticmethod
    def check_shared_type(mod_type, mod_name, mod, graph):
        """
        Function: check the PTQ & QAT some common check.
        """
        padding_mode_flag = GraphChecker.check_padding_mode(mod_type, mod_name, mod)
        if not padding_mode_flag:
            return False

        dilation_flag = GraphChecker.check_dilation(mod_type, mod_name, mod)
        if not dilation_flag:
            return False

        graph_flag = GraphChecker.check_graph(graph, mod_name)
        if not graph_flag:
            return False

        shared_flag = GraphChecker.check_shared_weight(mod_type, mod, graph)
        if not shared_flag:
            return False

        if not GraphChecker.check_rnn_limit(mod_type, mod_name, mod):
            return False

        return True

    @staticmethod
    def check_padding_mode(mod_type, mod_name, mod):
        """
        Function: padding mode check. PTQ & QAT common.
        """
        if mod_type in ['Conv1d', 'Conv2d', 'Conv3d', 'ConvTranspose1d', 'ConvTranspose2d', 'ConvTranspose3d']:
            if mod.padding_mode != 'zeros':
                LOGGER.logd(f"Layer {mod_name}'s padding_mode is {mod.padding_mode}.",
                            module_name="Configuration")
                return False
        return True

    @staticmethod
    def check_dilation(mod_type, mod_name, mod):
        """
        Function: dilation check. PTQ & QAT common.
        """
        if mod_type in ['Conv3d']:
            if isinstance(mod.dilation, (list, tuple)):
                if len(mod.dilation) != 3 or mod.dilation[0] != 1:
                    LOGGER.logd(
                        f"Layer {mod_name}'s dilation is {mod.dilation}.",
                        module_name="Configuration")
                    return False
            else:
                return False
        return True

    @staticmethod
    def check_graph(graph, mod_name):
        """
        Function: check graph. PTQ & QAT common.
        """
        if graph is not None:
            try:
                node = graph.get_node_by_name(mod_name)
                return GraphChecker.check_special_limit(node)
            except RuntimeError:
                # the mod is not in graph
                return False
        return True

    @staticmethod
    def check_rnn_limit(mod_type, mod_name, mod):
        """
        Function: check if rnn module can be retrain or not.
        """
        if mod_type not in RNN_LAYER_TYPE:
            return True
        if mod.num_layers != 1:
            LOGGER.logd("Layer {} cannot be quantized for num_layers isn't 1.".format(mod_name))
            return False
        if mod.bidirectional:
            LOGGER.logd("Layer {} cannot be quantized for bidirectional isn't False.".format(mod_name))
            return False
        if mod.dropout != 0:
            LOGGER.logd("Layer {} cannot be quantized for dropout isn't 0.".format(mod_name))
            return False
        if hasattr(mod, 'proj_size') and mod.proj_size != 0:
            LOGGER.logd("Layer {} cannot be quantized for proj_size isn't 0.".format(mod_name))
            return False
        return True

    @staticmethod
    def check_graph_retrain_type(node):
        """
        Function: check if node in graph can be retrain or not.
        """
        # check type
        if node.type not in RETRAIN_ONNX_TYPES:
            return False

        return GraphChecker.check_graph_shared_type(node)

    @staticmethod
    def check_graph_quantize_type(node):
        """ check if node in graph can be quantized or not."""
        # check type
        if node.type not in QUANTIZABLE_ONNX_TYPES:
            return False

        return GraphChecker.check_graph_shared_type(node)

    @staticmethod
    def check_graph_int16_quantize_type(node):
        """ check if node in graph can be quantized or not."""
        # check type
        if node.type not in INT16_QUANTIZABLE_ONNX_TYPES:
            return False
        if node.type in ['Conv']:
            # only conv2d and conv1d support int16
            if not check_kernel_shape(node, [1, 2]):
                return False
        elif node.type in ['ConvTranspose']:
            # convtranspose1d and convtranspose2d support int16
            if not check_kernel_shape(node, [1, 2]):
                return False

        return GraphChecker.check_graph_shared_type(node)

    @staticmethod
    def check_graph_shared_type(node):
        """
        Function: check common, if node in graph can be quant / retrain.
        """
        if node.type in ['Conv']:
            # conv1d & conv2d & conv3d
            if not check_kernel_shape(node, [1, 2, 3]):
                return False
        elif node.type in ['ConvTranspose']:
            # convtranspose1d & convtranspose2d & convtranspose3d
            if not check_kernel_shape(node, [1, 2, 3]):
                return False
        elif node.type == 'LSTM':
            if not check_lstm_limit(node):
                return False
        elif node.type == 'GRU':
            if not check_gru_limit(node):
                return False

        return GraphChecker.check_special_limit(node)

    @staticmethod
    def check_special_limit(node):
        """ Check if the node in graph satisfy special limits ro be quantized,
            limits include:
                1. Gemm's transA must be false
                2. reused module is not support
                3. MatMul shape must be 2
        """
        # Check not support Gemm with transA:True
        attr_helper = AttributeProtoHelper(node.proto)
        if node.type == 'Gemm':
            if attr_helper.has_attr('transA') and \
                    attr_helper.get_attr_value('transA') == 1:
                LOGGER.logw('Cannot support quantize "Gemm" layer "{}" with '
                            'transA:True.'.format(node.name))
                return False
        if node.type == 'GlobalAveragePool':
            LOGGER.logw('Cannot support quantize global-average-pooling '
                        'layer "{}".'.format(node.name))
            return False
        # Check not support reused node do quantize
        if node.has_attr('is_reuse') and node.get_attr('is_reuse'):
            LOGGER.logw(f'Not support do reused module "{node.name}" do quantize.')
            return False

        # check MatMul weights' dim is 1 or 2
        if node.type == 'MatMul':
            if not _check_matmul(node):
                return False

        # Check not support input_dimension_reduction node do quantize
        if node.has_attr('input_dimension_reduction') and node.get_attr('input_dimension_reduction'):
            LOGGER.logw(f'This module {node.name} is not supported for quantization '
                        'because its input data is dimensionality reduced.')
            return False

        return True


    @staticmethod
    def check_quant_behaviours(graph):
        """
        Function: Check whether there're quant behaviours in the model.
            If there're layers defined by AMCT, raise error.
        Inputs: None
        Return: None
        """

        quant_defined_layers = []
        if graph.model is None:
            # find all layers whose type is in AMCT_OPERATIONS in onnx graph
            for node in graph.nodes:
                if node.type in AMCT_OPERATIONS:
                    quant_defined_layers.append(node.name)
        else:
            # find all the layers whose type is in AMCT_OPERATIONS in model
            for name, mod in graph.model.named_modules():
                if type(mod).__name__ in AMCT_OPERATIONS:
                    quant_defined_layers.append(name)

        if quant_defined_layers:
            raise RuntimeError("The model cannot be quantized for following "\
                "quant layers are in the model {}".format(quant_defined_layers))

    @staticmethod
    def check_prune_type(mod_name, mod, graph):
        """
        Function: check if mod in model can be pruned or not.
        Return: bool, True if the module can be pruned.
        """
        mod_type = type(mod).__name__
        # check type
        if mod_type not in PRUNABLE_TYPES:
            return False

        if mod_type == 'Conv2d' and mod.groups == mod.in_channels:
            return False

        return GraphChecker.check_mod_prune_limit(mod_name, mod, graph)

    @staticmethod
    def check_mod_prune_limit(mod_name, mod, graph):
        try:
            node = graph.get_node_by_name(mod_name)
        except RuntimeError:
            # the mod is not in graph
            return False
        else:
            return GraphChecker.check_prune_limit(node)

        return True

    @staticmethod
    def check_selective_prune_type(mod_name, mod, graph):
        mod_type = type(mod).__name__
        # check type
        if mod_type not in SELECTIVE_PRUNABLE_TYPES:
            return False
        return GraphChecker.check_mod_prune_limit(mod_name, mod, graph)

    @staticmethod
    def check_graph_prune_type(node):
        """
        Function: check if graph node can be pruned or not.
        Return: bool, True if the module can be pruned
        """
        # check type
        if node.type not in PRUNABLE_ONNX_TYPES:
            return False

        if node.type in ['Conv']:
            # must be conv2D
            if not check_kernel_shape(node, [2]):
                return False
            # not support depthwise
            conv_util = OnnxConvUtil(node)
            if conv_util.is_depthwise_conv():
                return False

        return GraphChecker.check_prune_limit(node)

    @staticmethod
    def check_graph_selective_prune_type(node):
        """
        Function: check if graph node can be selective pruned or not.
        Return: bool, True if the module can be selective pruned
        """
        # check type
        if node.type not in ['Conv', 'ConvTranspose', 'Gemm', 'MatMul']:
            return False

        if node.type in ['Conv', 'ConvTranspose']:
            if not check_kernel_shape(node, [2]):
                return False

        return GraphChecker.check_prune_limit(node)

    @staticmethod
    def check_prune_limit(node):
        # Check not support reused node do prune
        if node.has_attr('is_reuse') and node.get_attr('is_reuse'):
            LOGGER.logw(f'Not support do reused module {node.name} do prune.')
            return False

        # Check not support input_dimension_reduction node do quantize
        if node.has_attr('input_dimension_reduction') and node.get_attr('input_dimension_reduction'):
            LOGGER.logw(f'This module {node.name} is not supported for quantization '
                        'because its input data is dimensionality reduced.')
            return False
        return True

    @staticmethod
    def check_shared_weight(mod_type, mod, graph):
        """
        Function: shared weight check.
        """
        if not graph or not graph.model:
            return True

        if mod_type not in ['Conv1d', 'Conv2d', 'Conv3d', 'ConvTranspose1d', 'ConvTranspose2d',
            'ConvTranspose3d', 'Linear']:
            return True

        weight_shared_count = 0
        for _, graph_mod in graph.model.named_modules():
            graph_mod_type = type(graph_mod).__name__
            if graph_mod_type not in \
                ['Conv2d', 'Conv3d', 'ConvTranspose1d', 'ConvTranspose2d', 'ConvTranspose3d', 'Linear']:
                continue
            if id(mod.weight) == id(graph_mod.weight):
                weight_shared_count += 1
            if weight_shared_count > 1:
                LOGGER.logw(f'Not support shared weight "{mod_type}" do quant.')
                return False
        return True


def check_kernel_shape(node, kernel_shape_range):
    """ Check whether the node's kernel_shape satisfy kernel_shape_range"""
    kernel_shape = AttributeProtoHelper(
        node.proto).get_attr_value('kernel_shape')
    if len(kernel_shape) not in kernel_shape_range:
        LOGGER.logd(f"Layer {node.name}'s kernel_shape is {kernel_shape}.",
                    module_name="Configuration")
        return False

    return True


def _check_matmul(node):
    """
    Function: Check whether Gemm can be quantized
    """
    try:
        weight_tensor = QuantOpInfo.get_weight_tensor(node)
    except RuntimeError:
        LOGGER.logd('Not support quantization for "MatMul" layer "{}" with no constant weight'.format(node.name))
        return False
    shape = weight_tensor.dims
    if len(shape) != 2:
        LOGGER.logd('Not support quantization for "MatMul" layer "{}" with weight '
                    'dim({}) not 2.'.format(node.name, len(shape)),
                    "Configuration")
        return False
    return True


def _check_group(node, group_range):
    """
    Function: Check whether the node's group satisfy group_range
    """
    try:
        group = AttributeProtoHelper(node.proto).get_attr_value('group')
    except RuntimeError:
        group = 1
    if group != group_range:
        LOGGER.logd("Layer %s's group is %s." % (node.name, group))
        return False
    return True


def _check_dilation(node, dilation_range):
    """
    Function:Check whether the node's dilation satisfy dilation_range
    """
    dilation = AttributeProtoHelper(node.proto).get_attr_value('dilations')
    if dilation not in dilation_range:
        LOGGER.logd("Layer %s's dilation is %s." % (node.name, dilation),
                    module_name="Configuration")
        return False
    return True


def check_lstm_limit(node):
    """ Check whether LSTM is quantizable. """
    if not _check_lstm_input_limit(node):
        return False
    if not _check_output_num_limit(node, LSTM_OUTPUT_NUMS):
        return False
    if not _check_rnn_weight_limit(node):
        return False
    if not _check_attrs_limit(node, LSTM_ATTRS_LIMIT_MAP):
        return False
    return True


def _check_lstm_input_limit(node):
    """ Check whether LSTM input meet quantization limit """
    node_input_anchors = node.input_anchors
    if len(node_input_anchors) < LSTM_C_INDEX + 1 or \
        node_input_anchors[LSTM_C_INDEX].get_peer_output_anchor() is None:
        LOGGER.logd('Node {} cannot be quantized '
                    'for it has no initial_c input'.format(node.name))
        return False

    initial_c = node_input_anchors[LSTM_C_INDEX].get_peer_output_anchor().node
    if not _check_node_linked_to_outside_input(initial_c):
        LOGGER.logd('Node {} cannot be quantized '
                    'for its initial_c is not linked to graph input'.format(node.name))
        return False

    if node_input_anchors[RNN_H_INDEX].get_peer_output_anchor() is None:
        LOGGER.logd('Node {} cannot be quantized '
                    'for it has no initial_h input'.format(node.name))
        return False

    initial_h = node_input_anchors[RNN_H_INDEX].get_peer_output_anchor().node
    if not _check_node_linked_to_outside_input(initial_h):
        LOGGER.logd('Node {} cannot be quantized '
                    'for its initial_h is not linked to graph input'.format(node.name))
        return False

    if node_input_anchors[RNN_SEQ_LENS_INDEX].get_peer_output_anchor() is not None:
        LOGGER.logd('Node {} cannot be quantized '
                    'for it has sequence_lens input'.format(node.name))
        return False
    if len(node_input_anchors) > LSTM_P_INDEX and \
        node_input_anchors[LSTM_P_INDEX].get_peer_output_anchor() is not None:
        LOGGER.logd('Node {} cannot be quantized for it has P input'.format(node.name))
        return False
    return True


def _check_output_num_limit(node, output_num):
    """ Check whether LSTM output meet quantization limit """
    node_output_anchors = node.output_anchors
    if len(node_output_anchors) < output_num:
        LOGGER.logd('Node {} cannot be quantized for its output is less than {}.'.format(node.name, output_num))
        return False
    for idx, output_anchor in enumerate(node_output_anchors):
        if output_anchor.get_peer_input_anchor is None:
            LOGGER.logd('Node {} cannot be quantized for its {} output is None'.format(node.name, idx))
            return False
    return True


def _check_rnn_weight_limit(node):
    """ Check whether RNN weight meet quantization limit """
    weights_node = QuantOpInfo.get_weight_node(node)
    weight_tensor = QuantOpInfo.get_node_tensor(weights_node)
    weight_shape = weight_tensor.dims
    if weight_shape[0] != 1:
        LOGGER.logd('Node {} cannot be quantized '
                    'for its num_directions is not equal to 1'.format(node.name))
        return False

    recurrence_weights_node = QuantOpInfo.get_recurrence_weight_node(node)
    recurrence_weight_tensor = QuantOpInfo.get_node_tensor(recurrence_weights_node)
    recurrence_weight_shape = recurrence_weight_tensor.dims
    if recurrence_weight_shape[0] != 1:
        LOGGER.logd('Node {} cannot be quantized '
                    'for its num_directions is not equal to 1'.format(node.name))
        return False
    return True


def _check_attrs_limit(node, attrs_limit_map):
    """ Check whether operator's attributes in quant limit on map"""
    attrs_helper = AttributeProtoHelper(node.proto)

    for attr_name, attr_limit in attrs_limit_map.items():
        if not attrs_helper.has_attr(attr_name):
            continue
        attr_value = attrs_helper.get_attr_value(attr_name)
        if attr_limit is not None:
            if isinstance(attr_value, bytes):
                attr_value = attr_value.decode('utf-8')
            if attr_value != attr_limit:
                LOGGER.logd("Node {}'s {} is {}, while limit is {}".format(
                    node.name, attr_name, attr_value, attr_limit))
                return False
        else:
            if attr_value is not None:
                LOGGER.logd("Node {} can not be quantized for "
                            "its {} is set.".format(node.name, attr_name))
                return False
    return True


def check_gru_limit(node):
    """ Check whether GRU op is quantizable"""
    if not _check_rnn_weight_limit(node):
        return False
    if not _check_gru_input_limit(node):
        return False
    if not _check_attrs_limit(node, GRU_ATTRS_LIMIT_MAP):
        return False
    if not _check_output_num_limit(node, GRU_OUTPUT_NUMS):
        return False
    attrs_helper = AttributeProtoHelper(node.proto)
    linear_before_reset = 0
    if attrs_helper.has_attr('linear_before_reset'):
        linear_before_reset = attrs_helper.get_attr_value('linear_before_reset')
    if linear_before_reset == 0:
        return False
    return True


def _check_gru_input_limit(node):
    node_input_anchors = node.input_anchors
    if len(node_input_anchors) < RNN_H_INDEX + 1 or \
        node_input_anchors[RNN_H_INDEX].get_peer_output_anchor() is None:
        LOGGER.logd('Node {} cannot be quantized '
                    'for it has no initial_h input'.format(node.name))
        return False

    if node_input_anchors[RNN_SEQ_LENS_INDEX].get_peer_output_anchor() is not None:
        LOGGER.logd('Node {} cannot be quantized '
                    'for it has sequence_lens input'.format(node.name))
        return False

    initial_h = node_input_anchors[RNN_H_INDEX].get_peer_output_anchor().node
    if not _check_node_linked_to_outside_input(initial_h):
        LOGGER.logd('Node {} cannot be quantized '
                    'for its initial_h is not linked to graph input'.format(node.name))
        return False
    return True


def _check_node_linked_to_outside_input(node):
    """ Check whether node is linked to graph input directly or indirectly"""
    if node.type == 'graph_anchor':
        return True
    flag = False
    for input_anchor in node.input_anchors:
        input_node = input_anchor.get_peer_output_anchor().node
        if _check_node_linked_to_outside_input(input_node):
            flag = True
            break
    return flag
