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
import os
import json
import hashlib
from . import parser_utils as utils
from ...utils.log import LOGGER # pylint: disable=E0402


MEMORY_BOUND = 'memory_bound'

# constant key string
GRAPH_OBJECT = "graph"
OP_OBJECT = "op"
NAME_OBJECT = "name"
TYPE_OBJECT = "type"
INPUT_OBJECT = "input"
ATTR_OBJECT = "attr"
L1_FUSION_SUB_GRAPH_NO_OBJECT = "_L1_fusion_sub_graph_no"
GROUP_OP_NAME_OBJECT = "_datadump_group_op_name"
ORIGINAL_OP_NAMES_OBJECT = "_datadump_original_op_names"
SUB_SPLITTER_INDEX_OBJECT = "_datadump_sub_splitter_index"
OUTPUT_DESC_OBJECT = "output_desc"
ORIGIN_NAME_OBJECT = "_datadump_origin_name"
ORIGIN_OUTPUT_INDEX_OBJECT = "_datadump_origin_output_index"
ORIGIN_FORMAT_OBJECT = "_datadump_origin_format"
GE_ORIGIN_SHAPE_OBJECT = "origin_shape"
IS_MULTI_OP = "_datadump_is_multiop"
FE_WEIGHT_COMPRESS = "_fe_weight_compress"
USE_CUDNN_ON_GPU = 'use_cudnn_on_gpu'
GE_ORIGIN_FORMAT_OBJECT = "origin_format"
KEY_OBJECT = "key"
VALUE_OBJECT = "value"
STRING_TYPE_OBJECT = "s"
INT_TYPE_OBJECT = "i"
BOOL_TYPE_OBJECT = 'b'
LIST_TYPE_OBJECT = "list"



class FusionRuleParser:
    """
    the class for parse fusion rule.
    """

    def __init__(self, path):
        self.json_path = path
        self.json_object = None
        self.fusion_op_name_to_op_map = {}
        self.op_name_to_fusion_op_name_map = {}
        self.compress_op_list = []
        self.original_op_list = []

    @staticmethod
    def _generate_names_hash(original_op_names):
        sha = hashlib.sha256()
        for original_op_name in original_op_names:
            sha.update(original_op_name.encode("utf-8"))
        return sha.hexdigest()

    @staticmethod
    def _check_key_exist(json_object, key):
        if key not in json_object:
            LOGGER.loge(
                'There is no "' + key + '" element in fusion rule file.')
            raise utils.CompareError(
                utils.VECTOR_COMPARISON_PARSER_JSON_FILE_ERROR)

    @staticmethod
    def _check_attr_exist(attr_list, key):
        for attr in attr_list:
            if attr['key'] == key:
                return True
        return False

    @staticmethod
    def _load_json_file(json_file):
        try:
            with open(json_file, 'r') as model_json_file:
                try:
                    return json.load(model_json_file)
                except Exception as e:
                    LOGGER.loge(
                        'Failed to load json object. '
                        'The content of the json file "'
                        + json_file + '" is invalid.')
                    raise utils.CompareError(
                        utils.VECTOR_COMPARISON_PARSER_JSON_FILE_ERROR) from e

        except IOError as io_error:
            LOGGER.loge(
                'Failed to open "' + json_file + '" ' + str(io_error) + '.')
            raise utils.CompareError(
                utils.VECTOR_COMPARISON_OPEN_FILE_ERROR) from io_error

    def analysis_fusion_rule(self):
        """
        Function Description:
            analysis fusion json file
        Exception Description:
            when analysis error throw exception
        """
        self.json_object = self._load_json_file(self.json_path)
        self._parse_fusion_op_json_object()

    def make_fusion_op_name(self, name, l1_fusion_no, original_op_names):
        """
        Function Description:
            make fusion op name by group op name and original op names
        Return:
            the fusion op name
        """
        # the fusion op name priority:
        # l1_fusion_no -> original_op_names -> name
        if l1_fusion_no != "":
            # the l1_fusion_no is not empty,
            # the fusion op name is the l1_fusion_no
            self.op_name_to_fusion_op_name_map[name] = l1_fusion_no
            return

        if original_op_names:
            if len(original_op_names) == 1:
                # There is one original op name
                if original_op_names[0] == '':
                    # the original name is empty, the fusion op name is op name
                    self.op_name_to_fusion_op_name_map[name] = name
                else:
                    # the original name is not empty,
                    # the fusion op name is original op name
                    self.op_name_to_fusion_op_name_map[name] = \
                        original_op_names[0]
            else:
                # The original op name more than one,
                # the fusion op name is hashed names
                self.op_name_to_fusion_op_name_map[name] = \
                    self._generate_names_hash(original_op_names)
        else:
            self.op_name_to_fusion_op_name_map[name] = name

    def _parse_fusion_op_json_object(self):
        # check graph element in json file
        self._check_array_object_valid(self.json_object, GRAPH_OBJECT)
        for graph in self.json_object[GRAPH_OBJECT]:
            # check op element in graph value
            self._check_array_object_valid(graph, OP_OBJECT)
            for op_object in graph[OP_OBJECT]:
                self._parse_op_object(op_object)

    def _parse_input(self, op_object):
        input_list = []
        # data layer has no input layer
        if INPUT_OBJECT in op_object:
            if not isinstance(op_object[INPUT_OBJECT], list):
                LOGGER.loge(
                    'The content of the json file "' + self.json_path
                    + '" is invalid. The "' + INPUT_OBJECT
                    + '" element is not an array.')
                raise utils.CompareError(
                    utils.VECTOR_COMPARISON_PARSER_JSON_FILE_ERROR)
            for item in op_object[INPUT_OBJECT]:
                if item == "" and len(op_object[INPUT_OBJECT]) == 1:
                    break
                # skip control edge
                if not item.endswith(':-1'):
                    input_list.append(item)
        return input_list

    def _parse_output_desc(self, op_object):
        output_desc_list = []
        # get output desc
        if OUTPUT_DESC_OBJECT in op_object:
            default_index = 0
            for output_desc_object in op_object[OUTPUT_DESC_OBJECT]:
                if ATTR_OBJECT in output_desc_object:
                    origin_name = self._get_string_value_in_attr(
                        output_desc_object[ATTR_OBJECT], ORIGIN_NAME_OBJECT)
                    origin_output_index = self._get_int_value_in_attr(
                        output_desc_object[ATTR_OBJECT],
                        ORIGIN_OUTPUT_INDEX_OBJECT)
                    if origin_output_index is None:
                        origin_output_index = default_index
                    origin_output_format = self._get_string_value_in_attr(
                        output_desc_object[ATTR_OBJECT], ORIGIN_FORMAT_OBJECT)
                    if origin_output_format == '':
                        origin_output_format = self._get_string_value_in_attr(
                            output_desc_object[ATTR_OBJECT],
                            GE_ORIGIN_FORMAT_OBJECT)
                    origin_output_shape = self._get_origin_shape_in_attr(
                        output_desc_object[ATTR_OBJECT])
                    output_desc = utils.OutputDesc(origin_name,
                                                   origin_output_index,
                                                   origin_output_format,
                                                   origin_output_shape,
                                                   origin_name)
                    output_desc_list.append(output_desc)
                default_index += 1
        return output_desc_list

    def _parse_op_object(self, op_object): # pylint: disable=R0914,R0912
        # check name element is valid
        self._check_string_object_valid(op_object, NAME_OBJECT)
        name = op_object[NAME_OBJECT]
        # check type element is valid
        self._check_string_object_valid(op_object, TYPE_OBJECT)

        input_list = self._parse_input(op_object)

        # check attr element is valid
        if ATTR_OBJECT not in op_object:
            attr_array = []
        else:
            self._check_array_object_valid(op_object, ATTR_OBJECT)
            attr_array = op_object[ATTR_OBJECT]
        is_multiop = self._get_bool_value_in_attr(attr_array, IS_MULTI_OP)
        # get l1_fusion_sub_graph_no
        l1_fusion_no = self._get_string_value_in_attr(
            attr_array, L1_FUSION_SUB_GRAPH_NO_OBJECT)
        # get original op names
        original_op_names, have_origin = self._get_original_op_names_in_attr(
            attr_array, name)
        if have_origin:
            if original_op_names != [""]:
                self.original_op_list.append(original_op_names)

        # get output desc
        output_desc_list = self._parse_output_desc(op_object)
        if not have_origin:
            if len(output_desc_list) == 0:
                output_desc = utils.OutputDesc(name, None, "", [], name)
                output_desc_list.append(output_desc)
            else:
                for (index, _) in enumerate(output_desc_list):
                    if output_desc_list[index].origin_name == "":
                        output_desc_list[index].origin_name = name
                        output_desc_list[index].sim_file_op_name = name

        # get whether need compress
        if FusionRuleParser._check_attr_exist(attr_array, FE_WEIGHT_COMPRESS):
            is_compress = self._get_bool_value_in_attr(
                attr_array, FE_WEIGHT_COMPRESS)
            if is_compress:
                self.compress_op_list.append(original_op_names)
        else:
            is_compress = False
        self.make_fusion_op_name(name, l1_fusion_no, original_op_names)
        fusion_op_name = self.op_name_to_fusion_op_name_map.get(name)
        fusion_op_info = utils.FusionOp(
            0, name, original_op_names, input_list,
            op_object[TYPE_OBJECT], output_desc_list, l1_fusion_no,
            is_multiop, is_compress)
        if fusion_op_name in self.fusion_op_name_to_op_map:
            fusion_op_info.op_id = \
                self.fusion_op_name_to_op_map[fusion_op_name][0].op_id
            self.fusion_op_name_to_op_map[fusion_op_name].append(
                fusion_op_info)
        else:
            fusion_op_info.op_id = len(self.fusion_op_name_to_op_map)
            self.fusion_op_name_to_op_map[fusion_op_name] = [fusion_op_info]

    def _get_string_value_in_attr(self, attr_array, key):
        value = ""
        for attr in attr_array:
            self._check_string_object_valid(attr, KEY_OBJECT)
            key_value = attr[KEY_OBJECT]
            if key_value == key:
                self._check_key_exist(attr, VALUE_OBJECT)
                value_value = attr[VALUE_OBJECT]
                self._check_string_object_valid(
                    value_value, STRING_TYPE_OBJECT)
                value = value_value[STRING_TYPE_OBJECT]
                break
        return value

    def _get_int_value_in_attr(self, attr_array, key):
        value = None
        for attr in attr_array:
            self._check_string_object_valid(attr, KEY_OBJECT)
            key_value = attr[KEY_OBJECT]
            if key_value == key:
                self._check_key_exist(attr, VALUE_OBJECT)
                value_value = attr[VALUE_OBJECT]
                self._check_int_object_valid(value_value, INT_TYPE_OBJECT)
                value = value_value[INT_TYPE_OBJECT]
                break
        return value

    def _get_origin_shape_in_attr(self, attr_array):
        value = []
        for attr in attr_array:
            self._check_string_object_valid(attr, KEY_OBJECT)
            key_value = attr[KEY_OBJECT]
            if key_value == GE_ORIGIN_SHAPE_OBJECT:
                self._check_key_exist(attr, VALUE_OBJECT)
                value_value = attr[VALUE_OBJECT]
                self._check_key_exist(value_value, LIST_TYPE_OBJECT)
                if INT_TYPE_OBJECT in value_value[LIST_TYPE_OBJECT]:
                    self._check_array_object_valid(
                        value_value[LIST_TYPE_OBJECT], INT_TYPE_OBJECT)
                    value = value_value[LIST_TYPE_OBJECT][INT_TYPE_OBJECT]
                break
        return value

    def _get_bool_value_in_attr(self, attr_array, key):
        value = False
        for attr in attr_array:
            self._check_string_object_valid(attr, KEY_OBJECT)
            key_value = attr[KEY_OBJECT]
            if key_value == key:
                self._check_key_exist(attr, VALUE_OBJECT)
                value_value = attr[VALUE_OBJECT]
                self._check_bool_object_valid(value_value, BOOL_TYPE_OBJECT)
                value = value_value[BOOL_TYPE_OBJECT]
                break
        return value

    def _get_original_op_names_in_attr(self, attr_array, op_name):
        array = []
        match = False
        for attr in attr_array:
            self._check_string_object_valid(attr, KEY_OBJECT)
            key_value = attr[KEY_OBJECT]
            if key_value == ORIGINAL_OP_NAMES_OBJECT:
                self._check_key_exist(attr, VALUE_OBJECT)
                value = attr[VALUE_OBJECT]
                self._check_key_exist(value, LIST_TYPE_OBJECT)
                if STRING_TYPE_OBJECT not in value[LIST_TYPE_OBJECT]:
                    array = ['']
                else:
                    self._check_array_object_valid(value[LIST_TYPE_OBJECT],
                                                   STRING_TYPE_OBJECT)
                    array = value[LIST_TYPE_OBJECT][STRING_TYPE_OBJECT]
                match = True
                break
        if not match:
            array.append(op_name)
        return array, match

    def _check_array_object_valid(self, json_object, key):
        self._check_key_exist(json_object, key)
        if not isinstance(json_object[key], list):
            LOGGER.loge(
                'The content of the json file "'
                + self.json_path + '" is invalid. The "'
                + key + '" element is not an array.')
            raise utils.CompareError(
                utils.VECTOR_COMPARISON_PARSER_JSON_FILE_ERROR)

    def _check_string_object_valid(self, json_object, key):
        self._check_key_exist(json_object, key)
        if not isinstance(json_object[key], str):
            LOGGER.loge(
                'The content of the json file "'
                + self.json_path + '" is invalid. The "'
                + key + '" element is not a string.')
            raise utils.CompareError(
                utils.VECTOR_COMPARISON_PARSER_JSON_FILE_ERROR)

    def _check_int_object_valid(self, json_object, key):
        self._check_key_exist(json_object, key)
        if not isinstance(json_object[key], int):
            LOGGER.loge(
                'The content of the json file "'
                + self.json_path + '" is invalid. The "'
                + key + '" element is not a integer.')
            raise utils.CompareError(
                utils.VECTOR_COMPARISON_PARSER_JSON_FILE_ERROR)

    def _check_bool_object_valid(self, json_object, key):
        self._check_key_exist(json_object, key)
        if not isinstance(json_object[key], bool):
            LOGGER.loge(
                'The content of the json file "'
                + self.json_path + '" is invalid. The "'
                + key + '" element is not a bool.')
            raise utils.CompareError(
                utils.VECTOR_COMPARISON_PARSER_JSON_FILE_ERROR)


def get_weight_compress_layers(json_file_path):
    """
    Function Description:
    get the layers that need compress, and the list of all the ops parsed from the json file
    Return:
    unzip_compress_op_list: the ops need to compress
    unzip_original_op_list: the ops parsed from the json file
    """
    real_json_file_path = os.path.realpath(json_file_path)
    if not os.path.exists(real_json_file_path):
        raise RuntimeError('the file {} do not exists!'.format(
            json_file_path))

    parser = FusionRuleParser(json_file_path)
    parser.analysis_fusion_rule()
    compress_op_list = parser.compress_op_list
    original_op_list = parser.original_op_list

    unzip_compress_op_list = []
    unzip_original_op_list = []
    for ori_ops in compress_op_list:
        unzip_compress_op_list.extend(ori_ops)
    for ops in original_op_list:
        unzip_original_op_list.extend(ops)
    # remove the properly duplicate ops
    unzip_compress_op_list = list(set(unzip_compress_op_list))
    unzip_original_op_list = list(set(unzip_original_op_list))
    return unzip_compress_op_list, unzip_original_op_list
