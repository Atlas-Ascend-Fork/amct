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
from ...amct_pytorch.common.prune.active_prune_helper_base import ActivePruneHelperBase
from ...amct_pytorch.common.prune.passive_prune_helper_base import PassivePruneHelperBase
from ...amct_pytorch.common.prune.eltwise_prune_helper_base import EltwisePruneHelperBase
from ...amct_pytorch.common.prune.concat_prune_helper_base import ConcatPruneHelperBase
from ...amct_pytorch.common.prune.disable_prune_helper_base import DisablePruneHelperBase
from ...amct_pytorch.common.utils.prune_record_attr_util import AttrProtoHelper
from ...amct_pytorch.common.prune.prune_recorder_helper import PruneRecordHelper

from ...amct_pytorch.utils.vars import PRUNABLE_ONNX_TYPES
from ...amct_pytorch.utils.vars import PASSIVE_PRUNABLE_ONNX_TYPES
from ...amct_pytorch.utils.vars import ELTWISE_ONNX_TYPES
from ...amct_pytorch.utils.log import LOGGER
from ...amct_pytorch.utils.quant_node import QuantOpInfo
from ...amct_pytorch.common.utils.onnx_node_util import AttributeProtoHelper
from ...amct_pytorch.utils.onnx_conv_util import OnnxConvUtil
from ...amct_pytorch.configuration.check import GraphChecker
from ...amct_pytorch.configuration import retrain_config as conf

PASSIVE_PRUNE_RECORDS = 'passive_prune_records'


def create_filter_prune_helper(node):
    """
    Function: create helper for node to do filter prune
    Params:node Node object
    Ruturns: prune_helper
    """
    if ActivePruneHelper.match_pattern(node):
        return ActivePruneHelper(node)

    if PassivePruneHelper.match_pattern(node):
        return PassivePruneHelper(node)

    if EltwisePruneHelper.match_pattern(node):
        return EltwisePruneHelper(node)

    if ConcatPruneHelper.match_pattern(node):
        return ConcatPruneHelper(node)

    return DisablePruneHelper(node)


def get_prune_group(node):
    """
    Function: get the prune group of node
    Param: None
    Return: a number
    """
    if node.type not in ['Conv', 'ConvTranspose']:
        return 1
    conv_util = OnnxConvUtil(node)
    prune_group = conv_util.get_group()
    if conv_util.is_depthwise_conv():
        prune_group = 1
    return prune_group


class ActivePruneHelper(ActivePruneHelperBase):
    """FilterPruneHelper for active type node"""
    prunable_types = PRUNABLE_ONNX_TYPES

    @staticmethod
    def match_pattern(node):
        """
        Function: Match pattern of active prune for node in graph
        Parameter: None
        Return: bool, matched or not
        """
        if not GraphChecker.check_graph_prune_type(node):
            return False
        if not conf.RetrainConfig().filter_prune_enable(node.name):
            return False
        return True

    def get_group(self):
        """
        Function: get the prune group of self.node
        Param: None
        Return: a number
        """
        return get_prune_group(self.node)

    def get_prune_axis(self):
        """
        Function: get the prune group of self.node
        Param: None
        Return: a number
        """
        if self.node.type in ['Conv']:
            return 1
        if self.node.type in ['MatMul', 'Gemm']:
            return -1
        raise RuntimeError('unexpected node type {}'.foramt(self.node.type))

    def get_cout_length(self):
        """
        Function: get the length of cout of self.node
        Param: None
        Return: a number
        """
        return QuantOpInfo.get_cout_length(self.node)

    def set_producer(self, record_helper):
        """
        Function: set the prune producer and add it to prune record when node is a prune consumer
        Param: record_helper, PruneRecordHelper, help to process record
        Return: None
        """
        super().set_producer(record_helper)

        PassivePruneHelper.set_producer_further(self.node, record_helper)


class PassivePruneHelper(PassivePruneHelperBase):
    """FilterPruneHelper for passive type node"""
    prunable_types = PRUNABLE_ONNX_TYPES

    def __init__(self, node):
        """
        Function: init object
        Param: node
        Return: None
        """
        super().__init__(node)
        if not GraphChecker.check_graph_prune_type(node):
            node.set_attr('unable_active', True)

    @staticmethod
    def get_prune_axis(node):
        """
        Function: get the prune group of self.node
        Param: None
        Return: a number
        """
        if node.type in ['Conv', 'ConvTranspose', 'BatchNormalization']:
            return 1
        if node.type in ['MatMul', 'Gemm']:
            return -1
        raise RuntimeError('unexpected node type {}'.foramt(node.type))

    @staticmethod
    def match_pattern(node):
        """
        Function: Match pattern of passive prune for node in graph
        Parameter: None
        Return: bool, matched or not
        """
        if node.type not in PASSIVE_PRUNABLE_ONNX_TYPES + ['D4toD2']:
            return False
        return True

    @staticmethod
    def set_producer_further(node, record_helper):
        """
        Function: supplement for node's set_producer
        Param: node, Node to process
        Param: record_helper, PruneRecordHelper, help to process record
        Return: None
        """
        if not node.has_attr(PASSIVE_PRUNE_RECORDS):
            return

        PassivePruneHelper._set_producer_further_ending(node, record_helper)
        PassivePruneHelper._set_producer_further_bn(node, record_helper)

    @staticmethod
    def _set_producer_further_ending(node, record_helper):
        """
        Function: supplement for node's set_producer, the ending has some requirements
        Param: node, Node to process
        Param: record_helper, PruneRecordHelper, help to process record
        Return: None
        """
        # only Linear-Linear, Conv-Conv is supported
        if node.type not in ['Conv', 'ConvTranspose', 'MatMul', 'Gemm']:
            return
        prune_axis = PassivePruneHelper.get_prune_axis(node)
        prune_records = node.get_attr(PASSIVE_PRUNE_RECORDS)
        disable = False
        for prune_record in prune_records:
            if prune_axis == record_helper.get_prune_axis(prune_record):
                continue
            axis_equal = False
            if prune_axis == -1 and record_helper.get_prune_axis(prune_record) == 1:
                for consumer in prune_record.consumer:
                    attr_helper = AttrProtoHelper(consumer)
                    consumer_type = attr_helper.get_attr_value('type')
                    if consumer_type == 'D4toD2':
                        axis_equal = True
                        break
            if not axis_equal:
                disable = True
                break

        if disable:
            while prune_records:
                record_helper.delete_record(prune_records[0])
            LOGGER.logd("Disable node {} for only support Linear-Linear and Conv-Conv, not support Linear-Conv "
                        "and Conv-Linear".format(node.name), "PassivePruneHelper")

    @staticmethod
    def _set_producer_further_bn(node, record_helper):
        """
        Function: supplement for node's set_producer, when the node is BatchNormalization
        Param: node, Node to process
        Param: record_helper, PruneRecordHelper, help to process record
        Return: None
        """
        if node.type != 'BatchNormalization':
            return
        torch_type = node.get_attr('torch_type')
        if torch_type == 'BatchNorm1d':
            # disable BatchNormalization when "MatMul+BatchNorm1d",
            prune_records = node.get_attr(PASSIVE_PRUNE_RECORDS)
            disable = True
            for prune_record in prune_records:
                for producer in prune_record.producer:
                    attr_helper = AttrProtoHelper(producer)
                    producer_type = attr_helper.get_attr_value('type')
                    if producer_type != 'MatMul':
                        disable = False
                        break
            if disable:
                while prune_records:
                    record_helper.delete_record(prune_records[0])
                LOGGER.logd('Disable BatchNormalization {} for "MatMul+BatchNorm1d"'.format(node.name))
        if torch_type == 'BatchNorm2d':
            prune_records = node.get_attr(PASSIVE_PRUNE_RECORDS)
            by_pass = False
            for prune_record in prune_records:
                for producer in prune_record.producer:
                    attr_helper = AttrProtoHelper(producer)
                    producer_type = attr_helper.get_attr_value('type')
                    if producer_type == 'MatMul':
                        by_pass = True
                        break
            if by_pass:
                PruneRecordHelper.delete_consumer_from_record(prune_record, node.name)
                node.set_attr(PASSIVE_PRUNE_RECORDS, PassivePruneHelper.get_producer_record(node, 0))
                LOGGER.logd('Bypass BatchNormalization {} for "MatMul+BatchNorm2d"'.format(node.name))

    def get_group(self):
        """
        Function: get the prune group of self.node
        Param: None
        Return: a number
        """
        return get_prune_group(self.node)

    def set_producer(self, record_helper):
        """
        Function: set the prune producer and add it to prune record when node is a prune consumer
        Param: record_helper, PruneRecordHelper, help to process record
        Return: None
        """
        super().set_producer(record_helper)
        self.set_producer_further(self.node, record_helper)


class EltwisePruneHelper(EltwisePruneHelperBase):
    """FilterPruneHelper for eltwise type node"""
    prunable_types = PRUNABLE_ONNX_TYPES

    @staticmethod
    def match_pattern(node):
        """
        Function: Match pattern of eltwise prune for node in graph
        Parameter: None
        Return: bool, matched or not
        """
        if node.type not in ELTWISE_ONNX_TYPES:
            return False
        return True


class ConcatPruneHelper(ConcatPruneHelperBase):
    """FilterPruneHelper for concat type node"""

    prunable_types = PRUNABLE_ONNX_TYPES

    @staticmethod
    def match_pattern(node):
        """
        Function: Match pattern of concat prune for node in graph
        Parameter: None
        Return: bool, matched or not
        """
        if node.type not in ['Concat', ]:
            return False
        attr_helper = AttributeProtoHelper(node.proto)
        axis = 0
        if attr_helper.has_attr('axis'):
            axis = attr_helper.get_attr_value('axis')
        if axis != 1:
            return False
        return True

    def _get_branch_cout_len(self, idx):
        """
        Function: Get the input[idx]'s cout length for concat op
        Param: idx, the input index for concat op
        Return: a number
        """
        producer, _ = self.node.get_producer(idx)
        if producer.type in PASSIVE_PRUNABLE_ONNX_TYPES:
            return QuantOpInfo.get_cout_length(producer)
        return None


class DisablePruneHelper(DisablePruneHelperBase):
    """FilterPruneHelper for disable type node"""
    prunable_types = PRUNABLE_ONNX_TYPES

    def __init__(self, node):
        """
        Function: init object
        Param: node
        Return: None
        """
        super().__init__(node)
        self.known_types = ['graph_anchor', 'Split', 'Concat', 'Reshape', 'Flatten', 'Expand']
        self.unknown_types = []

    def process(self, record_helper):
        """
        Function: process node's prune relationship and modify record
        Param: record_helper, PruneRecordHelper, help to process record
        Return: None
        """
        if self.node.type not in self.known_types:
            self.unknown_types.append(self.node.type)
        super().process(record_helper)
