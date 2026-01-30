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

the class of atc model transformation execution

"""

import os
import subprocess

from ....utils.log import LOGGER  # pylint: disable=E0402

MODULE_NAME = 'AtcExecuter'


class AtcExecuter():
    """ATC model transformation execution class."""
    def __init__(self):
        self.config = {}
        self.atc_command_list = []

    @staticmethod
    def _execute_command(command_list):
        LOGGER.logd(
            ' '.join(AtcExecuter._add_quotation_to_command(command_list)),
            MODULE_NAME)
        if len(command_list) >= 2:
            if command_list[0] in ('bash', 'cmd', '/bin/sh') or \
                command_list[1] in ('-c',):
                LOGGER.loge(
                    'There is a risk of unsafe execution of commands[{}]!'.format(
                        command_list), MODULE_NAME)
                raise RuntimeError(
                    'There is a risk of unsafe execution of commands[{}]!'.format(
                        command_list), MODULE_NAME)
            try:
                command_process = subprocess.Popen(command_list, shell=False)
            except FileNotFoundError as e:
                LOGGER.loge(
                    'ATC is not installed locally or the bin path of ATC is '
                    'not set to the environment variable(PATH)!',
                    MODULE_NAME)
                raise RuntimeError(
                    'ATC is not installed locally or the bin path of ATC is '
                    'not set to the environment variable(PATH)!') from e
            command_process.communicate()

    @staticmethod
    def _add_quotation_to_command(command_list):
        command_list_copy = command_list[:]
        for index, command_item in enumerate(command_list_copy):
            if command_item == '--input_shape':
                command_list_copy[index + 1] = "\"{}\"".format(command_list_copy[index + 1])
        return command_list_copy

    def get_om_model(self, config, pb_model, om_model):
        """Getting om model through ATC."""
        self.config = config
        self.config['model'] = os.path.realpath(pb_model)
        om_model = os.path.realpath(om_model)
        self.config['output'] = '.'.join(om_model.split('.')[:-1])
        self._get_model_command()
        AtcExecuter._execute_command(self.atc_command_list)

        if not os.path.exists(om_model):
            LOGGER.loge('Failed to generate the om model!', MODULE_NAME)
            LOGGER.logi(
                'The atc command is: {}'.format(' '.join(
                    AtcExecuter._add_quotation_to_command(self.atc_command_list))),
                MODULE_NAME)
            raise RuntimeError('Failed to generate the om model!')

    def get_om_json(self, config, om_model, json_file):
        """Get the corresponding JSON file of OM model through ATC."""
        self.config = config
        self.config['om'] = os.path.realpath(om_model)
        self.config['json'] = os.path.realpath(json_file)
        self._get_json_command()
        AtcExecuter._execute_command(self.atc_command_list)

        if not os.path.exists(json_file):
            LOGGER.loge(
                'Failed to generate the om\'s json model!', MODULE_NAME)
            LOGGER.logi(
                'The atc command is: {}'.format(' '.join(
                    AtcExecuter._add_quotation_to_command(self.atc_command_list))),
                MODULE_NAME)
            raise RuntimeError('Failed to generate the om\'s json model!')

    def _get_model_command(self):
        self.atc_command_list = ['atc']
        black_key_list = ['mode', 'json', 'om']
        for key, value in self.config.items():
            if key not in black_key_list and value is not None:
                if value != '':
                    self.atc_command_list.append('--{}'.format(key))
                    self.atc_command_list.append(value)
                else:
                    self.atc_command_list.append('--{}'.format(key))

    def _get_json_command(self):
        self.atc_command_list = [
            'atc', '--mode', '1', '--om',
            self.config.get('om'), '--json',
            self.config.get('json'), '--log',
            self.config.get('log')]
