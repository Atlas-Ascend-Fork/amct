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


class SelectivePruneHelperBase(PruneHelperBase):
    """ base class of selective prune helper"""

    def get_mask_shape(self):
        """
        Function: get the prune mask shape of self.node
        Param: None
        Return: a list
        """
        return None

    def create_record(self, record_helper):
        """
        Function: create a prune record when node is a selective prune producer
        Param: record_helper, PruneRecordHelper, help to process record
        Return: None
        """
        prune_record = record_helper.add_record()
        record = prune_record.selective_prune
        record.name = self.node.name

        attr_helper = AttrProtoHelper(record)
        prune_mask_shape = self.get_mask_shape()
        if prune_mask_shape:
            attr_helper.set_attr_value('mask_shape', 'INTS', prune_mask_shape)

        self.node.set_attr('selective_prune_records', [prune_record])

    def process(self, record_helper):
        """
        Function: process node's prune relationship and modify record
        Param: record_helper, PruneRecordHelper, help to process record
        Return: None
        """
        self.create_record(record_helper)
