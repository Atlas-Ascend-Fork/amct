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


class ConcatPruneHelperBase(PruneHelperBase):
    """ base class of concate prune helper"""

    @staticmethod
    def _set_producer_not_prune(prune_records, record_helper, do_prune):
        """
        Function: part of set_producer when prune is not support
        Param:prune_records, a list;
        Param:record_helper, PruneRecordHelper, help to process record
        Param:do_prune, a bool
        Return: a number
        """
        prune_axis = record_helper.get_prune_axis(prune_records[0])
        for prune_record in prune_records[1:]:
            if prune_axis != record_helper.get_prune_axis(prune_record):
                do_prune = False
                break
        # only support concat conv now
        if prune_axis != 1:
            do_prune = False
        if not do_prune:
            record_helper.delete_record_list(prune_records)
        return do_prune

    def get_invalid_input_idx(self):
        """
        Function: get the invalid input index when this input is not data input
        Param: None
        Return: a list
        """
        invalid_input_idx = []
        return invalid_input_idx

    def set_producer(self, record_helper):
        """
        Function: set the prune producer and add it to prune record when node is a prune consumer
        Param: record_helper, PruneRecordHelper, help to process record
        Return: None
        """
        # Step1: find the input prune info
        do_prune = True
        last_end = 0
        output_ranges = []
        prune_records = []
        for idx, _ in enumerate(self.node.input_anchors):
            if idx in self.get_invalid_input_idx():
                continue
            in_prune_records = self.get_producer_record(self.node, idx)
            if in_prune_records is None:
                cout_len = self._get_branch_cout_len(idx)
                # cout_len is None, cannot find the cout of input[idx]
                if cout_len is None:
                    do_prune = False
                else:
                    last_end += cout_len
                continue
            prune_records.extend(in_prune_records)
            for in_prune_record in in_prune_records:
                cout_len = record_helper.get_record_cout(in_prune_record)
                output_ranges.append([last_end, last_end + cout_len])
                last_end += cout_len

        # Step2: process for concat op
        # case1: has no prune
        if not prune_records:
            return
        # case2: not do prune
        do_prune = self._set_producer_not_prune(prune_records, record_helper, do_prune)
        if not do_prune:
            LOGGER.logd("disable Concat {} for input range is not determinate or prune axis differs"
                        .format(self.node.name), "ConcatPruneHelperBase")
            return
        # case3: do prune
        for prune_record, output_range in zip(prune_records, output_ranges):
            consumer_record = prune_record.consumer.add()
            consumer_record.name = self.node.name
            attr_helper = AttrProtoHelper(consumer_record)
            attr_helper.set_attr_value('type', 'STRING', self.node.type)
            attr_helper.set_attr_value('begin', 'INT', output_range[0])
            attr_helper.set_attr_value('end', 'INT', output_range[1])
        # the passive node has different list with active node, but the item is same one
        self.node.set_attr('passive_prune_records', [record for record in prune_records])

    def process(self, record_helper):
        """
        Function: process node's prune relationship and modify record
        Param: record_helper, PruneRecordHelper, help to process record
        Return: None
        """
        self.set_producer(record_helper)

    def _get_branch_cout_len(self, idx):
        """
        Function: Get the input[idx]'s cout length for concat op
        Param: idx, the input index for concat op
        Return: a number
        """
        raise NotImplementedError("This case has not implement way to process.")
