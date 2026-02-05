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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__all__ = ["RecordFileParser"]

from ...amct_pytorch.proto import scale_offset_record_pb2  # pylint: disable=E0611
from ...amct_pytorch.capacity import CAPACITY
from ...amct_pytorch.common.utils.parse_record_file import RecordFileParserBase
from ...amct_pytorch.utils.quant_node import QuantOpInfo
from ...amct_pytorch.configuration.check import GraphQuerier


class RecordFileParser(RecordFileParserBase):
    """
    Function: Parse the information of compression from record_file.
    APIs: read_record_file, parse
    """
    def __init__(self, record_file, graph, model_name, enable_quant=True,
                 enable_prune=False):
        """
        Function: init object
        Inputs:
            record_file: a string, the file to parse.
            graph: Graph, the graph corresponding to record_file.
            model_name: a string, the model's name.
        """
        capacity = {
            'FUSE_TYPES':
            CAPACITY.get_value('FUSE_ONNX_TYPES'),
            'QUANTIZABLE_TYPES':
            CAPACITY.get_value('QUANTIZABLE_ONNX_TYPES'),
            'NO_WEIGHT_QUANT_TYPES':
            CAPACITY.get_value('NO_WEIGHT_QUANT_ONNX_TYPES'),
        }
        config = {
            "capacity": capacity,
            "records_pb2": scale_offset_record_pb2,
            "op_quirer": QuantOpInfo,
            "graph_querier": GraphQuerier,
            "graph_checker": None
        }
        RecordFileParserBase.__init__(self, record_file, graph, model_name,
                                      config, enable_quant=enable_quant, enable_prune=enable_prune)
