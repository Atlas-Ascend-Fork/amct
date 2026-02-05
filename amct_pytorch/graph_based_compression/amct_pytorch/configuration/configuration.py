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

from ...amct_pytorch.utils.log import LOGGER
from ...amct_pytorch.graph.graph import Graph
from ...amct_pytorch.configuration.check import GraphChecker
from ...amct_pytorch.configuration.check import GraphQuerier
from ...amct_pytorch.common.utils.check_params import check_params
from ...amct_pytorch.common.utils.files import is_valid_name
from ...amct_pytorch.common.config.config_base import ConfigBase
from ...amct_pytorch.common.config.config_base import GraphObjects
from ...amct_pytorch.common.config.config_base import check_config_quant_enable
from ...amct_pytorch.common.config.config_base import check_config_dmq_balancer

from ...amct_pytorch.capacity import CAPACITY

CONFIGURER = ConfigBase(
    GraphObjects(graph_querier=GraphQuerier, graph_checker=None), CAPACITY)


class Configuration:
    """
    Function: manage configuration of project including quant_config
              and record_file.
    APIs: get_quant_config, get_layer_config, get_record_file_path;
        create_quant_config, parse_quant_config
    """
    __instance = None
    __is_init = False

    def __new__(cls, *args, **kw):
        if cls.__instance is None:
            cls.__instance = object.__new__(cls, *args, **kw)
        return cls.__instance

    def __init__(self):
        if not self.__is_init:
            self.__quant_config = None
            self.__record_file_path = None
            self.__skip_fusion_layers = None
            self.__initialized = False
            self.__is_init = True

    @staticmethod
    @check_params(config_file=str,
                  graph=Graph,
                  skip_modules=(list, type(None)),
                  batch_num=int,
                  activation_offset=bool,
                  config_defination=(type(None), str))
    def create_quant_config(
            config_file,
            graph,
            skip_modules=None,
            batch_num=1,
            activation_offset=True,
            config_defination=None):
        """
        Function: Create quant config.
        Inputs:
            config_file: a string, the file(including path information) to
                save quant config.
            graph: IR Graph, the graph to be quantized.
            skip_modules: a list, cell names which would not apply quantize,
                the skip layer type should be in ['Conv2D', 'MatMul'].
            batch_num: an integer indicating how many batch of data are used
                for calibration.
            activation_offset: a bool indicating whether there's offset or not
                in quantize activation.
            config_defination: a string, the simple config file path,
                containing the simple quant config according to proto.
        Returns: None
        """
        is_valid_name(config_file, 'config_file')
        GraphChecker.check_quant_behaviours(graph)

        if skip_modules is None:
            skip_modules = []
        if config_defination is not None:
            if len(skip_modules) != 0:
                LOGGER.logw(
                    "When setting 'config_defination' param of "
                    "'create_quant_config' API, 'skip_modules' need to be set "
                    "in simple quant config file!",
                    module_name="Configuration")
            CONFIGURER.create_config_from_proto(config_file, graph,
                                                config_defination)
        else:
            CONFIGURER.create_quant_config(config_file, graph, skip_modules,
                                           batch_num, activation_offset)

    @staticmethod
    def add_global_to_layer(quant_config):
        """add global quantize parameter to each layer"""
        CONFIGURER.add_global_to_layer(quant_config)
        LOGGER.logd("Add global params to layer's config success!",
                    "Configuration")

    @staticmethod
    @check_params(file_path=str, graph=Graph)
    def parse_quant_config(file_path, graph):
        """parse quantize configuration from config json file"""
        quant_config_torch = CONFIGURER.parse_config_file(file_path, graph)
        check_config_quant_enable(quant_config_torch)
        Configuration.add_global_to_layer(quant_config_torch)
        return quant_config_torch

    @staticmethod
    def check_quant_config_dmq_balancer(quant_config):
        """check whether no dmq_balancer enable layer"""
        check_config_dmq_balancer(quant_config)

    @staticmethod
    def check_dmq_balancer_enable(quant_config):
        """check whether no dmq_balancer enable layer"""
        for key, _ in quant_config.items():
            if not isinstance(quant_config[key], dict):
                continue
            if quant_config[key].get("dmq_balancer_param"):
                return True
        else:
            return False

    @staticmethod
    def get_layers_name(quant_config):
        """ Get all layers' name from quant_config """
        layers_name = list(quant_config.keys())
        # contains other keys: version, activation_offset,etc
        for item in CONFIGURER.root.get_keys():
            layers_name.remove(item)
        return layers_name

    @check_params(config_file=str, record_file=str, graph=Graph)
    def init(self, config_file, record_file, graph):
        """
        Function: init the Configuration.
        Inputs:
            config_file: a string, the file containing the quant config.
            record_file: a string, the file containing the scale and offset.
            graph: IR Graph
        Returns: None
        """
        self.__quant_config = self.parse_quant_config(
            os.path.realpath(config_file), graph)
        self.__record_file_path = os.path.realpath(record_file)
        self.__skip_fusion_layers = \
            self.__quant_config.get('skip_fusion_layers')
        self.__initialized = True

    def uninit(self):
        '''uninit Configuration Class'''
        self.__skip_fusion_layers = None
        self.__quant_config = None
        self.__initialized = False

    def get_quant_config(self):
        """
        Function: get quant config.
        """
        if not self.__initialized:
            raise RuntimeError('Must init Configuration before access it.')
        return self.__quant_config

    def get_layer_config(self, layer_name):
        """
        Function: get one lsyer's quant config.
        """
        if not self.__initialized:
            raise RuntimeError('Must init Configuration before access it.')
        return self.__quant_config.get(layer_name)

    def get_record_file_path(self):
        """
        Function: get record_file_path.
        """
        if not self.__initialized:
            raise RuntimeError('Must init Configuration before access it.')
        return self.__record_file_path

    def get_skip_fusion_layers(self):
        """
        Function: Get layers that need to skip do fusion
        """
        if not self.__initialized:
            raise RuntimeError('Must init Configuration before access it.')
        return self.__skip_fusion_layers

    def get_global_config(self, global_params_name):
        """
        Function: get global quant config.
        """
        if not self.__initialized:
            raise RuntimeError('Must init Configuration before access it.')
        return self.__quant_config.get(global_params_name)

    def get_fusion_switch(self):
        """
        Function: get global fusion switch state
        """
        if not self.__initialized:
            raise RuntimeError('Must init Configuration before access it.')
        return self.__quant_config.get('do_fusion')
