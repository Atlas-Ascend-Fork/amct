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


class SplitPruneHelperBase(PruneHelperBase):
    """ base class of SplitPruneHelperBase
    """
    def get_split_info(self):
        """
        Function: get the split info for self.node like [10, 20, 30]
        Param: None
        Return: a list
        """
        raise NotImplementedError("not implemented.")

    def get_split_input_axis(self):
        """
        Function: get split node data input axis
        Param: None
        Return: a list
        """
        return 0

    def set_producer(self, record_helper):
        """
        Function: set the prune producer and add it to prune record when node is a prune consumer
        Param: record_helper, PruneRecordHelper, help to process record
        Return: None
        """
        # step1: get the producer's prune records
        split_input_axis = self.get_split_input_axis()
        prune_records = self.get_producer_record(self.node, split_input_axis)
        if prune_records is None:
            return

        # step2: disable prune in some cases
        # step2_1: disable prune if split_op's input has been split to several parts to split.
        # 'concat-split' will be like this.
        if len(prune_records) > 1:
            record_helper.delete_record_list(prune_records)
            LOGGER.logd("disable Split {} for it has more than one prune_record".format(self.node.name),
                        "SplitPruneHelperBase")
            return
        # step2_2: disable prune if split_op's input will be pruned with group
        prune_group = record_helper.get_prune_group(prune_records[0])
        if prune_group != 1:
            record_helper.delete_record(prune_records[0])
            LOGGER.logd("disable Split {} for its prune_record has prune group more than 1".format(self.node.name),
                        "SplitPruneHelperBase")
            return

        # step3: split the record and add split op to records
        split_info = self.get_split_info()
        new_prune_records = record_helper.split_record(prune_records[0], split_info)

        output_range = [0, 0]
        for idx, split_size in enumerate(split_info):
            consumer_record = new_prune_records[idx].consumer.add()
            consumer_record.name = self.node.name
            attr_helper = AttrProtoHelper(consumer_record)
            output_range = [output_range[1], output_range[1] + split_size]
            attr_helper.set_attr_value('type', 'STRING', self.node.type)
            attr_helper.set_attr_value('begin', 'INT', output_range[0])
            attr_helper.set_attr_value('end', 'INT', output_range[1])
            attr_helper.set_attr_value('branch_idx', 'INT', idx)
            if idx == 0:
                self.node.set_attr('passive_prune_records', [new_prune_records[idx]])
            else:
                self.node.set_attr('passive_prune_records:{}'.format(idx), [new_prune_records[idx]])

    def process(self, record_helper):
        """
        Function: process node's prune relationship and modify record
        Param: record_helper, PruneRecordHelper, help to process record
        Return: None
        """
        self.set_producer(record_helper)
