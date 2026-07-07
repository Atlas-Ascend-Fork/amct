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
from .passive_prune_helper_base import PassivePruneHelperBase


class ActivePruneHelperBase(PruneHelperBase):
    """ base class of active prune helper"""

    def get_group(self):
        """
        Function: get the prune group of self.node
        Param: None
        Return: a number
        """
        return 1

    def get_cout_length(self):
        """
        Function: get the length of cout of self.node
        Param: None
        Return: a number
        """
        raise NotImplementedError("This function has not implemented.")

    def get_prune_axis(self):
        """
        Function: get the prune group of self.node
        Param: None
        Return: a number
        """
        return None

    def set_producer(self, record_helper):
        """
        Function: set the prune producer and add it to prune record when node is a prune consumer
        Param: record_helper, PruneRecordHelper, help to process record
        Return: None
        """
        prune_records = self.get_producer_record(self.node, 0)
        group = self.get_group()
        PassivePruneHelperBase.set_producer_for_passive(self.node, group, record_helper, prune_records)

    def create_record(self, record_helper):
        """
        Function: create a prune record when node is a prune producer
        Param: record_helper, PruneRecordHelper, help to process record
        Return: None
        """
        prune_record = record_helper.add_record()
        producer_record = prune_record.producer.add()
        producer_record.name = self.node.name
        cout_len = self.get_cout_length()

        attr_helper = AttrProtoHelper(producer_record)
        attr_helper.set_attr_value('type', 'STRING', self.node.type)
        attr_helper.set_attr_value('begin', 'INT', 0)
        attr_helper.set_attr_value('end', 'INT', cout_len)
        prune_axis = self.get_prune_axis()
        if prune_axis:
            attr_helper.set_attr_value('prune_axis', 'INT', prune_axis)
        self.node.set_attr('active_prune_records', [prune_record])
        group = self.get_group()
        if group not in [1, None]:
            attr_helper.set_attr_value('group', 'INT', group)
            success = record_helper.set_prune_group(prune_record, group)
            if not success:
                record_helper.delete_record(prune_record)

    def process(self, record_helper):
        """
        Function: process node's prune relationship and modify record
        Param: record_helper, PruneRecordHelper, help to process record
        Return: None
        """
        self.set_producer(record_helper)
        self.create_record(record_helper)
