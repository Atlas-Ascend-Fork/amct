#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

Create quantize config file and parse config file.

"""
import torch

from amct_pytorch.capacity import CAPACITY
from amct_pytorch.common.utils.check_params import check_params
from amct_pytorch.configuration.distill_config_base.distill_config_base import DistillConfigBase
from amct_pytorch.configuration.distill_config_base.distill_config_base import GraphObjects
from amct_pytorch.configuration.check import GraphQuerier
from amct_pytorch.configuration.check import GraphChecker
from amct_pytorch.graph.graph import Graph
from amct_pytorch.utils.log import LOGGER


CONFIGURER = DistillConfigBase(
    GraphObjects(graph_querier=GraphQuerier, graph_checker=None), CAPACITY
)


@check_params(config_file=str, graph=Graph)
def create_default_distill_config(config_file, graph):
    '''
    Function: create default distill config by graph
    Parameter: config_file: the path of json file to save distill config
                graph: IR Graph
    Return: None
    '''
    CONFIGURER.create_default_config(config_file, graph)


@check_params(config_file=str, graph=Graph, config_proto_file=str)
def create_distill_config_from_proto(config_file, graph, config_proto_file):
    '''
    Function: create distill config by graph and simple config file
    Parameter: config_file: the path of json file to save distill config
                graph: IR Graph
                config_proto_file: the path of user's simple config file
    Return: None
    '''
    CONFIGURER.create_config_from_proto(config_file, graph, config_proto_file)


def parse_distill_config(config_file, model):
    '''
    Function: parse distill config
        if graph is None, will not check op specification
    Parameter: config_file: the path of config json file
                model: torch.nn.Module
                graph: None, IR Graph
    Return: dict, distill config
    '''
    distill_config = CONFIGURER.parse_distill_config(config_file, model)
    return distill_config


def get_enable_quant_layers(distill_config):
    '''
    Function: get all enable quant layers
    Parameter: distill_config: dict, parse result from config json file
    Return: list, all enable quant layers
    '''
    quant_layers = CONFIGURER.get_enable_quant_layers(distill_config)
    return quant_layers


def get_quant_layer_config(layer_name, distill_config):
    '''
    Function: get quant layer config by layer name
    Parameter: layer_name: the name of layer
            distill_config: dict, parse result from config json file
    Return: dict, quant config of layer
    '''
    quant_layers = CONFIGURER.get_enable_quant_layers(distill_config)
    if layer_name not in quant_layers:
        LOGGER.logd("layer {} is disabled for quantization.".format(layer_name))
        return None
    layer_config = distill_config.get(layer_name)
    return layer_config
