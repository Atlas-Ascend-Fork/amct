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

from .filter_prune_helper_base import PruneHelperBase
from ...utils.log import LOGGER


class DisablePruneHelperBase(PruneHelperBase):
    """ base class of DisablePruneHelperBase"""

    @staticmethod
    def match_pattern(node):
        """ match pattern """
        raise NotImplementedError(
            "The match_pattern is not implemented for DisablePruneHelperBase for there's"
            "no blacklist.")

    def process(self, record_helper):
        """
        Function: process node's prune relationship and modify record
        Param: record_helper, PruneRecordHelper, help to process record
        Return: None
        """
        if not self.node.input_anchors:
            return
        for idx, _ in enumerate(self.node.input_anchors):
            prune_records = self.get_producer_record(self.node, idx)
            if prune_records is None:
                continue
            record_helper.delete_record_list(prune_records)
            LOGGER.logd("disable {} for it is not in whitelist.".format(self.node.name), "DisablePruneHelperBase")
