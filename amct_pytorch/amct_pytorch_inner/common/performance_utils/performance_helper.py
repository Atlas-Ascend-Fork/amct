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

the common auxiliary files for performance fallback

"""

import os
import csv

from ..utils.files import create_path
from ..utils.files import create_empty_file


class CsvManager():
    """the csv file management class."""
    def __init__(self):
        pass

    @staticmethod
    def write(output_file_name, heads, rows):
        """Writing data to a CSV file."""
        output_file_name = create_empty_file(output_file_name)
        with open(output_file_name, 'w') as out_csv:
            writer = csv.DictWriter(out_csv, heads)
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def read(input_file_name, *heads):
        """Reading data from a CSV file."""
        input_file_name = os.path.realpath(input_file_name)
        csv_read_dict = {}
        for head in heads:
            csv_read_dict[head] = []
        with open(input_file_name) as in_csv:
            reader = csv.DictReader(in_csv)
            for row in reader:
                for head in heads:
                    try:
                        csv_read_dict[head].append(row[head])
                    except KeyError:
                        pass
        return csv_read_dict


class IterationRecorder():
    """Iterative recorder design class."""
    def __init__(self, amct_log_dir, prefix_name):
        self.prefix_name = prefix_name
        self.amct_log_dir = os.path.realpath(amct_log_dir)
        self.recorders = {
            0: {
                'iter_num': 0,
                'root': self.amct_log_dir,
                'model_path': None,
                'profiling_path': None,
                'result_path': None,
                'parser_path': None,
                'total_time': None,
                'roll_back_layers': None,
                'not_quantized_layers': None
            }
        }

    def update(self, iter_num):
        """Update iteration recorder."""
        iteration_path = os.path.join(self.amct_log_dir,
                                      'iteration_num_{}'.format(iter_num))
        model_path = os.path.join(iteration_path, 'model_path')
        profiling_path = os.path.join(iteration_path, 'profiling_path')
        result_path = os.path.join(iteration_path, 'result_path')
        parser_path = os.path.join(iteration_path, 'parser_path')
        create_path(iteration_path)
        create_path(model_path)
        create_path(profiling_path)
        create_path(result_path)
        create_path(parser_path)
        self.recorders[iter_num] = {
            'iter_num': iter_num,
            'root': self.amct_log_dir,
            'model_path': model_path,
            'profiling_path': profiling_path,
            'result_path': result_path,
            'parser_path': parser_path,
            'total_time': 0,
            'roll_back_layers': [],
            'not_quantized_layers': []
        }

    def add_items(self, iter_num, item_key, parent_key, suffix_name):
        """Add elements to the iteration logger."""
        self.recorders.get(iter_num)[item_key] = os.path.join(
            self.recorders.get(iter_num).get(parent_key), '{}_iter{}{}'.format(
                self.prefix_name, iter_num, suffix_name))
