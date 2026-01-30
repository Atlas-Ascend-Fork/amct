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

the ascend performance sampling file

"""

import json

from .performance_helper import CsvManager
from ..utils.files import create_empty_file
from ...utils.log import LOGGER  # pylint: disable=E0402
from .executer.atc_executer import AtcExecuter
from .executer.ncx_executer import NcxExecuter

GRAPH_OBJECT = 'graph'
OP_OBJECT = 'op'
ATTR_OBJECT = 'attr'
NAME = 'name'
KEY = 'key'
ORIGINAL_OP = '_datadump_original_op_names'
VALUE_OBJECT = 'value'
LIST_OBJECT = 'list'
S_OBJECT = 's'
OM_FILE = 'om_file'


class AscendParser():  # pylint: disable=R0903
    """Ascend parsing class, used to parse om's JSON file and profiling data.
    """
    def __init__(self):
        pass

    @staticmethod
    def _check_key_exist(json_object, key):
        if key not in json_object:
            LOGGER.push_error_message(
                "The key '{}' doesn't exist in object.".format(key))
            raise RuntimeError(
                "The key '{}' doesn't exist in object.".format(key))

    @staticmethod
    def _create_csv_file(recorder):
        csv_manager = CsvManager()
        csv_reader_dict = csv_manager.read(
            recorder['prof_csv_file'],
            'Op Name', 'Task Duration(us)', 'Total Time(us)')

        csv_heads = list(csv_reader_dict.keys())
        csv_rows = []
        for index, _ in enumerate(csv_reader_dict[csv_heads[0]]):
            csv_row_dict = {}
            for head in csv_heads:
                try:
                    csv_row_dict[head] = csv_reader_dict[head][index]
                except IndexError:
                    pass
            csv_rows.append(csv_row_dict)

        csv_manager.write(recorder['parser_csv_file'], csv_heads, csv_rows)

    def pre_parser_profiling(self, recorder, quant_info):
        """Parsing the JSON file and profiling data of OM."""
        AscendParser._create_csv_file(recorder)
        self._create_json_file(recorder, quant_info)

    def _check_list_object_valid(self, json_object, key):
        self._check_key_exist(json_object, key)
        if not isinstance(json_object[key], list):
            LOGGER.push_error_message(
                "The type of '{}' value is not a list.".format(key))
            raise RuntimeError(
                "The type of '{}' value is not a list.".format(key))

    def _check_list_object_onlyone(self, json_object, key):
        self._check_key_exist(json_object, key)
        if not isinstance(json_object[key], list):
            LOGGER.push_error_message(
                "The type of '{}' value is not a list.".format(key))
            raise RuntimeError(
                "The type of '{}' value is not a list.".format(key))
        if len(json_object[key]) != 1:
            LOGGER.push_error_message(
                "The length of list[{}] is not equal to one!".format(key))
            raise RuntimeError(
                "The length of list[{}] is not equal to one!".format(key))

    def _parse_om_json(self, om_json):
        op_mapping = {}
        self._check_list_object_onlyone(om_json, GRAPH_OBJECT)
        graph_object = om_json[GRAPH_OBJECT][0]
        self._check_list_object_valid(graph_object, OP_OBJECT)

        for op_object in graph_object[OP_OBJECT]:
            flag = False
            self._check_list_object_valid(op_object, ATTR_OBJECT)
            for attr in op_object[ATTR_OBJECT]:
                if attr[KEY] == ORIGINAL_OP:
                    if LIST_OBJECT in attr[VALUE_OBJECT].keys():
                        if S_OBJECT in attr[VALUE_OBJECT][LIST_OBJECT].keys():
                            op_mapping[op_object[NAME]] = \
                                attr[VALUE_OBJECT][LIST_OBJECT][S_OBJECT]
                            flag = True
                            break
            if not flag:
                op_mapping[op_object[NAME]] = [op_object[NAME]]

        return op_mapping

    def _create_json_file(self, recorder, quant_info):
        with open(recorder['prof_json_file']) as fid:
            om_json = json.loads(fid.read())
            om_map_pb = self._parse_om_json(om_json)

        group_layers = quant_info.get_group_layers()
        for group_layer in group_layers:
            if len(group_layer) > 1:
                is_searched = False
                for group_layer_item in group_layer:
                    for key, value in om_map_pb.items():
                        if group_layer_item in value:
                            om_map_pb.get(key).extend(group_layer)
                            is_searched = True
                            break
                    if is_searched:
                        break

        json_file = create_empty_file(recorder['parser_json_file'])
        with open(json_file, 'w') as fid:
            json.dump(om_map_pb, fid, indent=4, separators=(',', ':'))


class AscendSampler():
    """
    Ascend performance sampling class is used to obtain the board performance
    data of OM model.
    """
    def __init__(self, sampler_config):
        self.sampler_config = sampler_config

        self.atc_executer = AtcExecuter()
        self.ascend_parser = AscendParser()

    def sample_profiling(self, iter_recorder, iter_num, quant_info):
        """Run the model on the board and get the performance data."""
        iter_recorder.add_items(iter_num, OM_FILE, 'result_path', '.om')
        iter_recorder.add_items(iter_num, 'prof_csv_file', 'profiling_path',
                                '.csv')
        iter_recorder.add_items(iter_num, 'prof_json_file', 'profiling_path',
                                '.json')
        iter_recorder.add_items(iter_num, 'parser_csv_file', 'parser_path',
                                '.csv')
        iter_recorder.add_items(iter_num, 'parser_json_file', 'parser_path',
                                '.json')
        recorder = iter_recorder.recorders[iter_num]

        self._get_om_model(recorder['pb_file'], recorder[OM_FILE])
        self._get_om_profiling(recorder[OM_FILE], recorder['prof_csv_file'])
        self._get_om_json(recorder[OM_FILE], recorder['prof_json_file'])

        self.ascend_parser.pre_parser_profiling(recorder, quant_info)

    def get_om_output(self, om_model, csv_file, inputs, outputs):
        """Run the model on the board and get the output data of the model."""
        NcxExecuter.get_om_output(self.sampler_config.get('ncx'),
                                  om_model, csv_file, inputs, outputs)

    def _get_om_model(self, pb_model, om_model):
        self.atc_executer.get_om_model(self.sampler_config.get('atc'),
                                       pb_model, om_model)

    def _get_om_profiling(self, om_model, csv_file):
        NcxExecuter.get_om_profiling(self.sampler_config.get('ncx'),
                                     om_model, csv_file)

    def _get_om_json(self, om_model, json_file):
        self.atc_executer.get_om_json(self.sampler_config.get('atc'), om_model,
                                      json_file)
