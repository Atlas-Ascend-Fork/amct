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

from abc import ABC, abstractmethod
import numpy as np
from google.protobuf import text_format
from ..utils.log_base import LOG_FILE_DIR
from ...utils.log import LOGGER

from ...proto import scale_offset_record_pb2
from ..utils.prune_record_attr_util import AttrProtoHelper

from .auto_channel_prune_search_base import AutoChannelPruneSearchBase


class SearchChannelBase(ABC):
    """ search algorithm implementation for auto_channel_prune_search """
    @abstractmethod
    def channel_prune_search(self, graph_info, search_records, prune_config):
        pass

    def update_graph(self, prune_channel_num, graph_info, search_records):
        for record_idx, prune_num in prune_channel_num.items():
            for producer in search_records[record_idx].producer:
                op_name = producer.name
                if graph_info.get(op_name):
                    cout_ori = graph_info[op_name]['cout']
                    cout_new = cout_ori - prune_num
                    graph_info[op_name]['bitops'] *= (cout_new / cout_ori)
                    graph_info[op_name]['cout'] = cout_new
            for consumer in search_records[record_idx].consumer:
                op_name = consumer.name
                if graph_info.get(op_name):
                    cin_ori = graph_info[op_name]['cin']
                    cin_new = cin_ori - prune_num
                    graph_info[op_name]['bitops'] *= (cin_new / cin_ori)
                    graph_info[op_name]['cin'] = cin_new

        AutoChannelPruneSearchBase.cal_channel_bitops(graph_info, search_records)


class GreedySearch(SearchChannelBase):
    def __init__(self, search_step=100):
        self.search_step = search_step

    def channel_prune_search(self, graph_info, search_records, prune_config):
        """
        Function: using greedy algorithm to search the prune channel of layer
        Param:
            graph_info: Information about each layer in graph
            search_records: Information about each record to be pruned
            target_compress_ratio: Customer-specified target compression ratio
            ascend_optimized(bool): Whether the remaining channels of the record are 16-aligned.
            max_prune_ratio: Maximum pruning ratio of a single record
        Return: prune_channel, a dict
        """
        target_compress_ratio, ascend_optimized, max_prune_ratio = prune_config
        ori_bit = self._sum_bit(graph_info)
        self._set_prune_record(search_records)
        compress_ratio = 1
        prune_channel = {}
        for record in search_records:
            for producer in record.producer:
                ori_cout = graph_info.get(producer.name).get('ori_cout')
                prune_channel[producer.name] = [1] * ori_cout
        prune_channel_num = {}
        prune_channel_amount = 0
        while compress_ratio < target_compress_ratio:
            min_record, min_record_idx, min_record_prune_step = \
                self._find_min_record(ascend_optimized, search_records, max_prune_ratio)

            if min_record is None:
                new_bit = self._sum_bit(graph_info)
                compress_ratio = ori_bit / new_bit
                if compress_ratio < target_compress_ratio:
                    LOGGER.logw("No more channels to prune, maybe your acceleration target is too high.")
                break

            prune_num, prune_channel = \
                self._get_prune_record(graph_info, min_record, min_record_prune_step, prune_channel)
            if min_record_idx in prune_channel_num:
                prune_channel_num[min_record_idx] += prune_num
            else:
                prune_channel_num[min_record_idx] = prune_num
            prune_channel_amount += prune_num
            if prune_channel_amount >= self.search_step:
                self.update_graph(prune_channel_num, graph_info, search_records)
                prune_channel_amount = 0
                prune_channel_num = {}
                new_bit = self._sum_bit(graph_info)
                compress_ratio = ori_bit / new_bit

        return prune_channel

    def _set_prune_record(self, search_records):
        """
        Function: set cout, sorted sensitivity & idx in search_records
        Param:
            search_records:Information about each record to be pruned
        Return: None
        """
        for record in search_records:
            attr_helper = AttrProtoHelper(record.producer[0])
            record_sensitivity = attr_helper.get_attr_value('sensitivity')
            record_sensitivity_idx = np.argsort(record_sensitivity)[::-1]
            record_sensitivity.sort(reverse=True)

            for producer in record.producer:
                attr_helper = AttrProtoHelper(producer)
                begin = attr_helper.get_attr_value('begin')
                end = attr_helper.get_attr_value('end')
                cout = end - begin
                attr_helper.set_attr_value("cout", 'INT', cout)
                record_sensitivity_idx = [i + begin for i in record_sensitivity_idx]
                attr_helper.set_attr_value("record_sensitivity_idx", 'INTS', record_sensitivity_idx)
                attr_helper.set_attr_value("record_sensitivity", 'FLOATS', record_sensitivity)

    def _sum_bit(self, graph_info):
        """
        Function: calculating the bitpos of the graph
        Param: graph_info: information about each layer in graph
        Return: sum_bit
        """
        sum_bit = 0
        for layer in graph_info.values():
            sum_bit += layer.get('bitops')
        return sum_bit

    def _find_min_record(self, ascend_optimized, search_records, max_prune_ratio):
        """
        Function: find the record with the minimum value density
        Param:
            ascend_optimized(bool): whether the remaining channels of the record are 16-aligned.
            search_records:Information about each record to be pruned
        Return: min_record, min_record_idx, prune_step
        """
        if ascend_optimized:
            channel_group_size = 16
        else:
            channel_group_size = 1

        min_value = float("inf")
        min_record = None
        min_record_idx = None
        for index, record in enumerate(search_records):
            attr_helper = AttrProtoHelper(record.producer[0])
            record_sensitivity = attr_helper.get_attr_value('record_sensitivity')
            bitops = attr_helper.get_attr_value('bitops')

            prune_step = len(record_sensitivity) % channel_group_size
            if prune_step == 0:
                prune_step = channel_group_size

            cout = attr_helper.get_attr_value('cout')
            if len(record_sensitivity) - prune_step >= cout * (1 - max_prune_ratio) and \
                len(record_sensitivity) - prune_step > 0:
                value_density = [(sens / bitops) for sens in record_sensitivity]
                value = sum(value_density[-prune_step:])
                if value < min_value:
                    min_value = value
                    min_record = record
                    min_record_idx = index
        return min_record, min_record_idx, prune_step

    def _get_prune_record(self, graph_info, min_record, prune_step, prune_channel):
        """
        Function: prune the min_record and get the prune channel info
        Param:
            graph_info: Information about each layer in graph
            min_record: The record with the minimum value density
            prune_step: Number for channels to be pruned each time for a record
            prune_channel: Channel reservation at each layer in the graph
        Return: prune_num, prune_channel
        """
        prune_num = 0
        for producer in min_record.producer:
            attr_helper = AttrProtoHelper(producer)
            prune_layer_name = producer.name
            record_sensitivity = attr_helper.get_attr_value('record_sensitivity')
            record_sensitivity_idx = attr_helper.get_attr_value('record_sensitivity_idx')

            prune_idx = record_sensitivity_idx[-prune_step:]
            record_sensitivity = record_sensitivity[:-prune_step]
            record_sensitivity_idx = record_sensitivity_idx[:-prune_step]

            attr_helper.set_attr_value("record_sensitivity", 'FLOATS', record_sensitivity)
            attr_helper.set_attr_value("record_sensitivity_idx", 'INTS', record_sensitivity_idx)

            for i in prune_idx:
                prune_channel[prune_layer_name][i] = 0
        prune_num += prune_step
        return prune_num, prune_channel

