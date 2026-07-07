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

from abc import ABC, abstractmethod, abstractstaticmethod
from ..utils.prune_record_attr_util import AttrProtoHelper
from ...utils.log import LOGGER

BITOPS = 'bitops'


class AutoChannelPruneSearchBase(ABC):
    def __init__(self,
                 graph,
                 input_data,
                 config_item,
                 sesitivity,
                 search_alg,
                 output_config):
        self.graph = graph
        self.input_data = input_data
        self.config_item = config_item
        self.search_records = self.get_search_ops(graph, config_item.create_prune_config())
        self.graph_info = self.get_graph_bitops(graph, input_data)
        self.sensitivity = sesitivity
        self.search_alg = search_alg
        self.output_config = output_config

    @staticmethod
    def cal_channel_bitops(graph_info, search_records):
        for record in search_records:
            bitops = 0
            for producer in record.producer:
                if graph_info.get(producer.name):
                    bitops += graph_info[producer.name][BITOPS] / graph_info[producer.name]['cin']
            for consumer in record.consumer:
                if graph_info.get(consumer.name):
                    bitops += graph_info[consumer.name][BITOPS] / graph_info[consumer.name]['cout']
            attr_helper = AttrProtoHelper(record.producer[0])
            attr_helper.set_attr_value(BITOPS, 'FLOAT', bitops)

    @abstractmethod
    def get_search_ops(self, graph, prune_config):
        ''' get search range and save as records '''
        pass

    @abstractmethod
    def get_graph_bitops(self, graph, input_data):
        ''' calculate bitops and save as graph_info '''
        pass

    def get_sensitivity(self, input_data, test_iteration, search_records, output_nodes=None):
        ''' sensitivity write to record'''
        search_graph_info = self.graph_info
        self.sensitivity.setup_initialization((self.graph, search_graph_info), input_data, test_iteration, output_nodes)
        self.sensitivity.get_sensitivity(search_records)

    def run(self, input_data, output_nodes=None):
        if not self.search_records:
            LOGGER.logw("No channel prunable structure detected in layers to be searched.")
            prune_channel = {}
        else:
            self.get_sensitivity(input_data, self.config_item.test_iteration, self.search_records, output_nodes)
            self.cal_channel_bitops(self.graph_info, self.search_records)
            prune_channel = self.search_alg.channel_prune_search(self.graph_info, self.search_records, \
                (self.config_item.compress_ratio, self.config_item.ascend_optimized, self.config_item.max_prune_ratio))
        self.config_item.create_final_config(prune_channel, self.output_config)
