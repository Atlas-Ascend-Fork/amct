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
