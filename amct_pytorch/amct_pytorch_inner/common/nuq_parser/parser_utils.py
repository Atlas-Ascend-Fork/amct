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

utils for amct_parse_fusion_json.py

"""
VECTOR_COMPARISON_NONE_ERROR = 0
VECTOR_COMPARISON_UNKNOWN_ERROR = 1
VECTOR_COMPARISON_NO_DUMP_FILE_ERROR = 2
VECTOR_COMPARISON_INVALID_PATH_ERROR = 3
VECTOR_COMPARISON_PARSE_DUMP_FILE_ERROR = 4
VECTOR_COMPARISON_MALLOC_ERROR = 5
VECTOR_COMPARISON_INVALID_PARAM_ERROR = 6
VECTOR_COMPARISON_OPEN_FILE_ERROR = 7
VECTOR_COMPARISON_CLOSE_FILE_ERROR = 8
VECTOR_COMPARISON_OPEN_DIR_ERROR = 9
VECTOR_COMPARISON_CLOSE_DIR_ERROR = 10
VECTOR_COMPARISON_INVALID_DUMP_DATA_ERROR = 11
VECTOR_COMPARISON_INDEX_OUT_OF_BOUNDS_ERROR = 12
VECTOR_COMPARISON_PARSER_JSON_FILE_ERROR = 13
VECTOR_COMPARISON_MEMSET_ERROR = 14
VECTOR_COMPARISON_LEFT_DUMP_ZERO_ERROR = 15
VECTOR_COMPARISON_RIGHT_DUMP_ZERO_ERROR = 16
VECTOR_COMPARISON_MATCH_MORE_FILE_ERROR = 17
VECTOR_COMPARISON_DUMP_FILE_ERROR = 18
VECTOR_COMPARISON_WRITE_FILE_ERROR = 19
VECTOR_COMPARISON_INVALID_DUMP_TYPE_ERROR = 20
VECTOR_COMPARISON_INVALID_DATA_TYPE_ERROR = 21
VECTOR_COMPARISON_INVALID_FORMAT_ERROR = 22
VECTOR_COMPARISON_INVALID_SHAPE_ERROR = 23
RESERVED_DIGITS = 6


class OutputDesc: # pylint: disable=R0903,R0913
    """
    The class for fusion op output desc
    """

    def __init__(self, origin_name, origin_output_index, origin_format,
                 origin_shape, sim_file_op_name):
        self.origin_name = origin_name
        self.origin_output_index = origin_output_index
        self.origin_format = origin_format
        self.origin_shape = origin_shape
        # just use for simulator find file
        self.sim_file_op_name = sim_file_op_name


class FusionOp: # pylint: disable=R0903,R0913,R0902
    """
    The class for fusion op
    """

    def __init__(self, op_id, op_name, original_op_names, input_list, op_type,
                 output_desc, l1_fusion_no, is_multi_op, is_compress):
        self.op_id = op_id
        self.op_name = op_name
        self.original_op_names = original_op_names
        self.input_list = input_list
        self.op_type = op_type
        self.output_desc = output_desc
        self.l1_fusion_no = l1_fusion_no
        self.is_multi_op = is_multi_op
        self.is_compress = is_compress


class CompareError(Exception):
    """
    The class for compare error
    """

    def __init__(self, error_info): # pylint: disable=W0231
        self.error_info = error_info
