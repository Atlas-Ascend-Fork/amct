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


class PruneHelperBase:
    """ base class of prune helper"""
    prunable_types = []

    def __init__(self, node):
        """
        Function: init object
        Param: node
        Return: None
        """
        self.node = node

    @staticmethod
    def match_pattern(node):
        """
        Function: Match pattern of prune for node in graph
        Param: None
        Return: bool, matched or not
        """
        return False
    
    @classmethod
    def get_producer_record(cls, node, in_idx):
        """
        Function: get prune records of node input[idx]'s producer
        Param:
            node, a Node
            in_idx, a number
        Return: prune_records, a list
        """
        pre_node, out_idx = node.get_producer(in_idx)
        if not pre_node:
            return None
        if out_idx == 0:
            tail = ''
        else:
            tail = ':{}'.format(out_idx)
        if pre_node.type in cls.prunable_types and not pre_node.has_attr('unable_active'):
            if not pre_node.has_attr('active_prune_records{}'.format(tail)):
                return None
            prune_records = pre_node.get_attr('active_prune_records{}'.format(tail))
        else:
            if not pre_node.has_attr('passive_prune_records{}'.format(tail)):
                return None
            prune_records = pre_node.get_attr('passive_prune_records{}'.format(tail))
        return prune_records

    def process(self, record_helper):
        """
        Function: process node's prune relationship and modify record
        Param: record_helper, PruneRecordHelper, help to process record
        Return: None
        """
        pass
