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


class PassivePruneHelperBase(PruneHelperBase):
    """ base class of PassivePruneHelperBase"""

    @staticmethod
    def set_producer_for_passive(node, group, record_helper, prune_records):
        """
        Function: add it to prune record when node is a prune consumer
        Param:
            node, a Node
            group, node's prune group
            record_helper, PruneRecordHelper, help to process record
            prune_records, node producer's record
        Return: None
        """
        if prune_records is None:
            return

        if group not in [1, None]:
            # cannot support "concat+group conv"
            if len(prune_records) > 1:
                record_helper.delete_record_list(prune_records)
                LOGGER.logd("disable {} {} for cannot set prune group for several passive prune_records"
                            .format(node.type, node.name), "PassivePruneHelperBase")
                return

            success = record_helper.set_prune_group(prune_records[0], group)
            # group_num1 + group_num2
            if not success:
                record_helper.delete_record(prune_records[0])
                LOGGER.logd("disable {} {} for prune group is unmatched".format(node.type, node.name),
                            "PassivePruneHelperBase")
                return

        output_ranges = [0, 0]
        for prune_record in prune_records:
            consumer_record = prune_record.consumer.add()
            consumer_record.name = node.name

            cout_len = record_helper.get_record_cout(prune_record)
            output_ranges = [output_ranges[1], output_ranges[1] + cout_len]
            attr_helper = AttrProtoHelper(consumer_record)
            attr_helper.set_attr_value('type', 'STRING', node.type)
            attr_helper.set_attr_value('begin', 'INT', output_ranges[0])
            attr_helper.set_attr_value('end', 'INT', output_ranges[1])
            if group not in [1, None]:
                attr_helper.set_attr_value('group', 'INT', group)
        # the passive node has different list with active node, but the item is same one
        node.set_attr('passive_prune_records', [record for record in prune_records])

    def get_group(self):
        """
        Function: get the prune group of self.node
        Param: None
        Return: a number
        """
        return 1

    def set_producer(self, record_helper):
        """
        Function: set the prune producer and add it to prune record when node is a prune consumer
        Param: record_helper, PruneRecordHelper, help to process record
        Return: None
        """
        prune_records = self.get_producer_record(self.node, 0)
        group = self.get_group()
        self.set_producer_for_passive(self.node, group, record_helper, prune_records)

    def process(self, record_helper):
        """
        Function: process node's prune relationship and modify record
        Param: record_helper, PruneRecordHelper, help to process record
        Return: None
        """
        self.set_producer(record_helper)
