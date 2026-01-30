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

Parse record file.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from amct_pytorch.proto import scale_offset_record_pb2  # pylint: disable=E0611
from amct_pytorch.capacity import CAPACITY
from amct_pytorch.common.utils.parse_record_file import RecordFileParserBase
from amct_pytorch.utils.quant_node import QuantOpInfo
from amct_pytorch.configuration.check import GraphQuerier

__all__ = ["RecordFileParser"]


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
