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

import os
from collections import OrderedDict
from google.protobuf import text_format

from ....proto import sampler_config_pb2  # pylint: disable=E0402
from ....utils.log import LOGGER  # pylint: disable=E0402

SOC_VERSION_CONSTRAINT = ['Ascend610']
FRAMEWORK_DICT = {'Tensorflow': '3', 'Onnx': '5'}
ASCEND_SAMPLER_CONFIG = 'ascend_sampler_config'


class SamplerProtoConfig(): # pylint: disable=R0903
    """
    Function: Cope with simple config file from proto.
    APIs: get_sampler_config.
    """
    def __init__(self, config_file, framework):
        self.config_file = os.path.realpath(config_file)
        if framework not in FRAMEWORK_DICT.keys():
            LOGGER.loge(
                'framework[{}] is not valid! Please choose from {}'.format(
                    framework, list(FRAMEWORK_DICT.keys())))
            raise TypeError(
                'framework[{}] is not valid! Please choose from {}'.format(
                    framework, list(FRAMEWORK_DICT.keys())))
        self.framework = framework
        self.proto_config = self._read()

    @staticmethod
    def _get_optional_options(config_item):
        optional_options = {}
        for option in config_item.options:
            optional_options[option.key] = option.value
        return optional_options

    @staticmethod
    def _get_distribute_config(config_item):
        if not config_item.HasField('distribute_config'):
            LOGGER.loge('Not set distribute_config!')
            raise AttributeError('Must set distribute_config!')
        distribute_config = {}
        distribute_config['ip'] = config_item.distribute_config.ip
        distribute_config['port'] = config_item.distribute_config.port

        return distribute_config

    @staticmethod
    def _get_soc_version(config_item):
        if not config_item.HasField('soc_version'):
            LOGGER.loge('Not set soc_version!')
            raise AttributeError('Must set soc_version!')
        if config_item.soc_version not in SOC_VERSION_CONSTRAINT:
            LOGGER.loge('Unknown soc_version!')
            raise AttributeError('Unknown soc_version been used!')
        return config_item.soc_version

    @staticmethod
    def _get_input_shape(config_item):
        if not config_item.HasField('input_shape'):
            return None
        return config_item.input_shape

    def get_sampler_config(self):
        """get sampler's config information."""
        sampler_config = OrderedDict()
        if self.proto_config.HasField(ASCEND_SAMPLER_CONFIG):
            self.proto_config = self.proto_config.ascend_sampler_config
            sampler_config[ASCEND_SAMPLER_CONFIG] = OrderedDict()
            sampler_config[ASCEND_SAMPLER_CONFIG][
                'atc'] = self._get_atc_config()
            sampler_config[ASCEND_SAMPLER_CONFIG][
                'ncx'] = self._get_ncx_config()
        else:
            raise AttributeError('Vaild sampler config not found!')
        return sampler_config

    def _read(self):
        """Read config from config_file. """
        proto_config = sampler_config_pb2.SamplerConfig()
        with open(self.config_file, 'rb') as cfg_file:
            pbtxt_string = cfg_file.read()
            text_format.Merge(pbtxt_string, proto_config)

        return proto_config

    def _get_atc_config(self):
        atc_config = OrderedDict()
        atc_config['soc_version'] = SamplerProtoConfig._get_soc_version(
            self.proto_config)
        atc_config['input_shape'] = SamplerProtoConfig._get_input_shape(
            self.proto_config)
        atc_config['framework'] = FRAMEWORK_DICT.get(self.framework)
        atc_config['log'] = 'null'

        if self.proto_config.HasField('atc'):
            optional_options = SamplerProtoConfig._get_optional_options(
                self.proto_config.atc)
            atc_config.update(optional_options)
        return atc_config

    def _get_ncx_config(self):
        ncx_config = OrderedDict()
        ncx_config['soc_version'] = SamplerProtoConfig._get_soc_version(
            self.proto_config)
        ncx_config['distribute_config'] = \
            SamplerProtoConfig._get_distribute_config(self.proto_config)
        ncx_config['loop_count'] = 10

        return ncx_config
