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
from ..utils.prune_record_attr_util import AttrProtoHelper
from .filter_prune_helper_base import PruneHelperBase
from ...utils.log import LOGGER

MODULE_NAME = 'EltwisePruneHelperBase'


class EltwisePruneHelperBase(PruneHelperBase):
    """ base class of EltwisePruneHelperBase"""

    def set_producer(self, record_helper):
        """
        Function: set the prune producer and add it to prune record when node is a prune consumer
        Param: record_helper, PruneRecordHelper, help to process record
        Return: None
        """
        prune_record_in0 = self.get_producer_record(self.node, 0)
        prune_record_in1 = self.get_producer_record(self.node, 1)

        # input0 and input1 both have no prune
        if prune_record_in0 is None and prune_record_in1 is None:
            return
        # only input1 has prune
        if prune_record_in0 is None:
            record_helper.delete_record_list(prune_record_in1)
            LOGGER.logd("disable Add {} for only input[1] is to do prune".format(self.node.name),
                        MODULE_NAME)
            return
        # only input0 has prune
        if prune_record_in1 is None:
            record_helper.delete_record_list(prune_record_in0)
            LOGGER.logd("disable Add {} for only input[0] is to do prune".format(self.node.name),
                        MODULE_NAME)
            return
        # input0 and input1 both have prune
        if len(prune_record_in0) > 1 or len(prune_record_in1) > 1:
            record_helper.delete_record_list(prune_record_in1)
            record_helper.delete_record_list(prune_record_in0)
            LOGGER.logd("disable Add {} for input[0] or input[1] has more than one prune_record"
                        .format(self.node.name), MODULE_NAME)
            return
        # the prune_axis is different
        prune_axis_in0 = record_helper.get_prune_axis(prune_record_in0[0])
        prune_axis_in1 = record_helper.get_prune_axis(prune_record_in1[0])
        if prune_axis_in0 != prune_axis_in1:
            record_helper.delete_record(prune_record_in0[0])
            record_helper.delete_record(prune_record_in1[0])
            LOGGER.logd("disable Add {} for input[0] or input[1] has different prune axis".format(self.node.name),
                        MODULE_NAME)
            return

        prune_record = record_helper.merge_record(prune_record_in0[0], prune_record_in1[0])
        cout_len = record_helper.get_record_cout(prune_record)
        consumer_record = prune_record.consumer.add()
        consumer_record.name = self.node.name
        attr_helper = AttrProtoHelper(consumer_record)
        attr_helper.set_attr_value('type', 'STRING', self.node.type)
        attr_helper.set_attr_value('begin', 'INT', 0)
        attr_helper.set_attr_value('end', 'INT', cout_len)

        self.node.set_attr('passive_prune_records', [prune_record])

    def process(self, record_helper):
        """
        Function: process node's prune relationship and modify record
        Param: record_helper, PruneRecordHelper, help to process record
        Return: None
        """
        self.set_producer(record_helper)
