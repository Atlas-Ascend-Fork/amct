#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

Config middle layer for post quant.

"""
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
