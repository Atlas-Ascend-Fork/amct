#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

Top APIs of AMCT_TORCH prune tool for user

"""


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

    @staticmethod
    def match_pattern(node):
        """
        Function: Match pattern of prune for node in graph
        Param: None
        Return: bool, matched or not
        """
        return False

    def process(self, record_helper):
        """
        Function: process node's prune relationship and modify record
        Param: record_helper, PruneRecordHelper, help to process record
        Return: None
        """
        pass
