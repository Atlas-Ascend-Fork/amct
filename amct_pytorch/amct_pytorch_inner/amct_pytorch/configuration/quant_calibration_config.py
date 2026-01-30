#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

Config middle layer for quant calibration.

"""

import os

import torch

from amct_pytorch.utils.log import LOGGER
from amct_pytorch.common.utils.check_params import check_params
from amct_pytorch.configuration.quant_calibration_config_base.quant_calibration_config_base \
    import QuantCalibrationConfigBase


CONFIGURER = QuantCalibrationConfigBase()


def inner_create_quant_calibration_config(config_file,
                                          model,
                                          quant_layers,
                                          config_defination):
    """ inner method to create quant calibration config """
    if config_defination is None:
        CONFIGURER.create_default_config(config_file, model, quant_layers)
        LOGGER.logi('Create quant calibration config file {} without config defination success'.format(config_file))
        return
    config_defination = os.path.realpath(config_defination)
    if not os.path.isfile(config_defination):
        raise RuntimeError('Invalid argument config_defination. '\
                            'Please check whether the file {} exist'.format(config_defination))
    CONFIGURER.create_config_from_proto(config_file, model, config_defination)
    LOGGER.logi('Create quant calibration config file {} '\
                'with config defination {} success'.format(config_file, config_defination))


def parse_cali_quant_config(config_file, model):
    """ parse quant calibration config from config file based on model """
    return CONFIGURER.parse_quant_config(config_file, model)


def get_quant_layer_config(layer_name, quant_calibration_config):
    """ get quantized layer names on quant calibration config file """
    return QuantCalibrationConfigBase.get_quant_layer_config(layer_name, quant_calibration_config)


def get_kv_cache_quant_layers(quant_calibration_config):
    """ get kv quant layer name """
    return CONFIGURER.get_quant_layers(quant_calibration_config, 'kv_quant')
