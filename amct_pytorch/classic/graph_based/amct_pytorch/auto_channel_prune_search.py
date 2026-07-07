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


import os
import copy
from io import BytesIO
import torch

from ..amct_pytorch.utils.log import LOGGER
from ..amct_pytorch.utils.model_util import ModuleHelper
from ..amct_pytorch.parser.parser import Parser
from ..amct_pytorch.common.auto_channel_prune.auto_channel_prune_config_helper \
    import AutoChannelPruneConfigHelper
from ..amct_pytorch.common.utils.check_params import check_params
from ..amct_pytorch.common.auto_channel_prune.auto_channel_prune_search_base \
    import AutoChannelPruneSearchBase
from ..amct_pytorch.common.auto_channel_prune.search_channel_base import SearchChannelBase
from ..amct_pytorch.common.auto_channel_prune.sensitivity_base import SensitivityBase
from ..amct_pytorch.common.utils.net_params import ParamsHelper
from ..amct_pytorch.common.utils.prune_record_attr_util import AttrProtoHelper
from ..amct_pytorch.configuration.retrain_config import RetrainConfig
from ..amct_pytorch.configuration.check import GraphQuerier
from ..amct_pytorch.capacity import CAPACITY
from ..amct_pytorch.prune.pruner_helper import PruneHelper
from ..amct_pytorch.configuration.check import check_kernel_shape
from ..amct_pytorch.utils.module_info import ModuleInfo
from ..amct_pytorch.utils.output_shape import OutputShape
from ..amct_pytorch.common.auto_channel_prune.search_channel_base import GreedySearch
from ..amct_pytorch.utils.singleton_record import SingletonScaleOffsetRecord


class AutoChannelPruneSearch(AutoChannelPruneSearchBase):
    """ """
    def __init__(self, graph, input_data, config_helper, sensitivity, search_alg, output_cfg):
        super().__init__(graph, input_data, config_helper, sensitivity, search_alg, output_cfg)

    def get_search_ops(self, graph, prune_config):
        RetrainConfig.amc_init(graph, self.config_item.config_file)
        SingletonScaleOffsetRecord().reset_record()
        prune_helper = PruneHelper(graph, self.input_data[0], None)
        prune_helper.preprocess_graph()
        prune_helper.simplify_graph()
        prune_helper.find_prune_consumers()
        record = prune_helper.record.prune_record
        return record

    def get_graph_bitops(self, graph, input_data=None):
        graph_info = {}
        model_helper = ModuleHelper(graph.model)
        for node in graph.nodes:
            if node.type not in CAPACITY.get_value('PRUNABLE_ONNX_TYPES'):
                continue
            module = model_helper.get_module(node.name)
            if node.type == 'Conv' and check_kernel_shape(node, [2]):
                in_channel, out_channel, bitops = self._cal_conv2d_flops(module)

            elif node.type in ['MatMul', 'Gemm']:
                in_channel, out_channel, bitops = self._cal_matmul_flops(module)
            else:
                continue
            graph_info[node.name] = {}
            graph_info.get(node.name)['cin'] = in_channel
            graph_info.get(node.name)['cout'] = out_channel
            graph_info.get(node.name)['ori_cout'] = out_channel
            graph_info.get(node.name)['bitops'] = bitops

        return graph_info

    def _cal_conv2d_flops(self, module):
        weight_shape = module.weight.shape
        output_shape = OutputShape(module)(weight_shape)
        # batch_size, out_channel, out_h, out_w
        out_channel, out_h, out_w = output_shape[1], output_shape[2], output_shape[3]
        group = module.groups
        in_channel, k_h, k_w = weight_shape[1], weight_shape[2], weight_shape[3]
        in_channel *= group
        # not module.bias cause to boolean value of Tensor with more than one value is ambiguous
        has_bias = module.bias is not None
        flops = ParamsHelper.calc_conv_flops(in_channel=in_channel,
                                            out_channel=out_channel,
                                            k_h=k_h,
                                            k_w=k_w,
                                            out_h=out_h,
                                            out_w=out_w,
                                            group=1,
                                            has_bias=has_bias)
        flops *= output_shape[0]
        bitops = flops * ((module.weight.element_size() * 8) ** 2)
        return in_channel, out_channel, bitops


    def _cal_matmul_flops(self, module):
        input_shape = module.weight.shape
        output_shape = OutputShape(module)(input_shape)
        has_bias = module.bias is not None
        flops = ParamsHelper.calc_matmul_flops(input_shape, output_shape, has_bias=has_bias)
        flops *= output_shape[0]
        bitops = flops * ((module.weight.element_size() * 8) ** 2)
        return input_shape[-1], output_shape[-1], bitops


class TaylorLossSensitivity(SensitivityBase):
    """ TaylorLossSensitivity """
    def __init__(self):
        super(TaylorLossSensitivity, self).__init__()
        self.set_loss_func()
        self.grads = None
        self.weights = None
        self.graph_info = None
        self.graph = None

    def set_loss_func(self, loss_func=torch.nn.L1Loss()):
        self.loss_func = loss_func

    def setup_initialization(self, graph_tuple, input_data, test_iteration, output_nodes=None):
        """
        Function: setup initialization
        Param: graph_tuple (graph, graph_info)
        Return: None
        """
        self.graph, self.graph_info = graph_tuple
        self.grads, self.weights = self.get_backward_grad(input_data, test_iteration)

    def get_sensitivity(self, search_records):
        """
        Function: get sensitivity
        Param: search_records
        Return: None, revise record
        """
        model_helper = ModuleHelper(self.graph.model)
        for prune_record in search_records:
            producer_list = []
            module_list = []
            ch_info = {}
            for producer in prune_record.producer:
                attr_helper = AttrProtoHelper(producer)
                node_type = attr_helper.get_attr_value('type')
                if node_type not in CAPACITY.get_value('PRUNABLE_ONNX_TYPES'):
                    continue
                ch_info['begin'] = attr_helper.get_attr_value('begin')
                ch_info['end'] = attr_helper.get_attr_value('end')
                producer_list.append(producer.name)
                module_list.append(model_helper.get_module(producer.name))

            consumer_list = []
            for consumer in prune_record.consumer:
                attr_helper = AttrProtoHelper(consumer)
                node_type = attr_helper.get_attr_value('type')
                if node_type not in CAPACITY.get_value('PRUNABLE_ONNX_TYPES'):
                    continue
                parent_module = model_helper.get_parent_module(consumer.name)
                if parent_module not in module_list:
                    continue
                consumer_list.append(consumer.name)

            sensitivity = \
                sum(self.compute_taylor_by_channel(x, ch_info) for x in producer_list) + \
                sum(self.compute_taylor_by_channel(x, ch_info, is_consumer=True) for x in consumer_list)

            attr_helper = AttrProtoHelper(prune_record.producer[0])
            attr_helper.set_attr_value('sensitivity', 'FLOATS', sensitivity.tolist())

    def compute_taylor_by_channel(self, layer_name, ch_info, is_consumer=False):
        """
        Function: compute, taylor
        Param: layer_name, ch_info, is_consumer
        Return: list, taylor score by channel
        """
        taylor = self.weights.get(layer_name) * self.grads.get(layer_name)
        model_helper = ModuleHelper(self.graph.model)
        module = model_helper.get_module(layer_name)
        cout_axis, cin_axis = ModuleInfo.get_wts_cout_cin(module)
        taylor = taylor.split(self.graph_info.get(layer_name).get('cout'), cout_axis)[0]
        taylor = taylor.split(self.graph_info.get(layer_name).get('cin'), cin_axis)[0]
        ch_axis = cin_axis if is_consumer else cout_axis
        sum_dims = [_ for _ in range(len(self.weights.get(layer_name).shape)) if _ != ch_axis]
        taylor_arr = taylor.norm(p=1, dim=sum_dims).cpu()
        taylor_arr = taylor_arr if is_consumer else taylor_arr[ch_info.get('begin'): ch_info.get('end')]
        return taylor_arr

    def get_backward_grad(self, input_data, test_iteration, batch_num=1):
        """
        Function: feed data, foward and backward, get backward grad
        Param: input_data, test_iteration, batch_num
        Return: grads, weights
        """
        if input_data[0].shape[0] < test_iteration * batch_num:
            raise RuntimeError('insufficient input data for iterative testing : ' + str(test_iteration))
        device = next(iter(self.graph.model.parameters())).device
        grads_collection = {}
        wts_collection = {}
        model_helper = ModuleHelper(self.graph.model)
        self.graph.model.zero_grad()
        self.graph.model.eval()
        x_data, y_label = input_data
        for i in range(test_iteration):
            y_pred = self.graph.model(x_data[i * batch_num: (i + 1) * batch_num])
            loss = self.loss_func(y_pred, y_label[i * batch_num: (i + 1) * batch_num].to(device))
            loss.backward()
            # reduce and clear grads
            for layer_name in self.graph_info.keys():
                with torch.no_grad():
                    module = model_helper.get_module(layer_name)
                    if grads_collection.get(layer_name):
                        grads_collection[layer_name] += copy.deepcopy(module.weight.grad)
                    else:
                        grads_collection[layer_name] = copy.deepcopy(module.weight.grad)
            self.graph.model.zero_grad()

        for layer_name in self.graph_info.keys():
            module = model_helper.get_module(layer_name)
            wts_collection[layer_name] = copy.deepcopy(module.weight)

        return grads_collection, wts_collection


@check_params(model=torch.nn.Module,
              config=str,
              input_data=list,
              output_cfg=str,
              sensitivity=(str, SensitivityBase),
              search_alg=(str, SearchChannelBase))
def auto_channel_prune_search(model, config, input_data, output_cfg,
                              sensitivity='TaylorLossSensitivity', search_alg='GreedySearch'):
    """ Auto search quant bit for a model based calibration.

    Args:
        model (torch.nn.Module): model to be processed.
        config (string): file from AutoChannelPruneConfig, indicating how to do search.
        input_data (list of dict of data): feed dict for calibration.
        output_cfg (string): path of output channel prune config file.
        sensitivity (union [str, SensitivityBase]): the way to measure the quant sensitivity.
        search_alg (union [str, SearchChannelBase]): algorithm to do search.
    """
    # check
    config = os.path.realpath(config)
    output_cfg = os.path.realpath(output_cfg)

    try:
        model = ModuleHelper.deep_copy(model)
    except RuntimeError as exception:
        LOGGER.logw(exception, "auto_channel_prune_search")

    ModuleHelper(model).check_amct_op()

    # parse to graph
    tmp_onnx = BytesIO()
    Parser.export_onnx(model, input_data[0], tmp_onnx)
    graph = Parser.parse_net_to_graph(tmp_onnx)
    graph.add_model(model)

    if isinstance(search_alg, str):
        if search_alg == 'GreedySearch':
            search_alg = GreedySearch()
        else:
            raise ValueError("search_alg not support.")
    if isinstance(sensitivity, str):
        if sensitivity == 'TaylorLossSensitivity':
            sensitivity = TaylorLossSensitivity()
        else:
            raise ValueError("sensitivity not support.")

    config_helper = AutoChannelPruneConfigHelper(graph, config, GraphQuerier, CAPACITY)
    amc = AutoChannelPruneSearch(graph, input_data, config_helper, sensitivity, search_alg, output_cfg)
    amc.run(input_data)
