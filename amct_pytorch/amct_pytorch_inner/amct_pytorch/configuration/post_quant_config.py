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

from amct_pytorch.utils.log import LOGGER
from amct_pytorch.capacity import POST_CAPACITY
from amct_pytorch.configuration.post_quant_config_base.post_quant_configurer \
    import PostQuantConfigurer


def inner_create_post_quant_config(config_file,
                                   model,
                                   config_defination):
    """ inner method to create quant calibration config """
    configurer = PostQuantConfigurer(POST_CAPACITY)
    if config_defination is None:
        configurer.create_default_config(config_file, model)
        LOGGER.logd('Create post quant config file {} without config defination success'.format(config_file))
        return
    config_defination = os.path.realpath(config_defination)
    if not os.path.isfile(config_defination):
        raise RuntimeError('Invalid argument config_defination. '\
                            'Please check whether the file {} exist'.format(config_defination))
    configurer.create_config_from_proto(config_file, model, config_defination)
    LOGGER.logd('Create post quant config file {} '\
                'with config defination {} success'.format(config_file, config_defination))


def parse_post_quant_config(config_file, model):
    """ parse quant calibration config from config file based on model """
    return PostQuantConfigurer(POST_CAPACITY).parse_config_file(config_file, model)
