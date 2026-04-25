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
import copy

from ...amct_pytorch.graph.graph import Graph
from ...amct_pytorch.utils.log import LOGGER
from ...amct_pytorch.configuration.check import GraphChecker
from ...amct_pytorch.configuration.check import GraphQuerier
from ...amct_pytorch.common.utils.check_params import check_params
from ...amct_pytorch.common.retrain_config.retrain_config_base import \
        RetrainConfigBase
from ...amct_pytorch.common.config.config_base import GraphObjects
from ...amct_pytorch.common.auto_channel_prune.auto_channel_prune_config_helper import AutoChannelPruneConfigHelper
from ...amct_pytorch.capacity import CAPACITY

WEIGHT_QUANT_PARAMS = 'weight_quant_params'
CONFIGURER = RetrainConfigBase(
    GraphObjects(graph_querier=GraphQuerier, graph_checker=None), CAPACITY
)


class RetrainConfig():
    """
    Function: manage configuration of project including quant_config
              and record_file.
    APIs: init, get_quant_config, get_layer_config, get_record_file_path;
        create_quant_config, parse_quant_config
    """
    __instance = None
    __initialized = False
    __record_file_path = None

    def __new__(cls, *args, **kw):
        if cls.__instance is None:
            cls.__instance = object.__new__(cls, *args, **kw)
        return cls.__instance

    def __init__(self):
        super(RetrainConfig, self).__init__()
        if not RetrainConfig.__initialized:
            raise RuntimeError(
                "Classmethod RetrainConfig.init() should be called firstly.")
        if self.enable_retrain:
            self.__retrain_config = RetrainConfig.retrain_config
            self.__skip_fusion_layers = []
        if self.enable_prune:
            self.__retrain_config = RetrainConfig.retrain_config

    @staticmethod
    @check_params(file_name=str, graph=Graph)
    def parse_retrain_config(file_name, graph):
        """parse quantize configuration from config json file"""
        GraphChecker.check_quant_behaviours(graph)
        quant_config = CONFIGURER.parse_config_file(file_name, graph)
        return quant_config

    @staticmethod
    @check_params(config_name=str, graph=Graph, config_defination=(str))
    def create_quant_retrain_config(config_name, graph, config_defination):
        """
        Function: Create retrain config.
        Inputs:
            config_name: a string, the file(including path information) to
                save retrain config.
            graph: IR Graph, the graph to be retrained.
            config_defination: the file containing the simple retrain config
                according to proto.
        Returns: None
        """
        if isinstance(config_name, str):
            config_name = os.path.realpath(config_name)
        config_file = os.path.realpath(config_defination)

        LOGGER.logi(
            f"Create {config_name} according to {config_file}, " \
                "other configuration will be ignored.", module_name='RetrainConfig')
        CONFIGURER.set_ability(enable_retrain=True, enable_prune=False)
        CONFIGURER.create_config_from_proto(config_name, graph, config_file)

        LOGGER.logi(
            "Create quant config success!", module_name='RetrainConfig')

    @staticmethod
    @check_params(file_name=str, graph=Graph)
    def create_default_retrain_config(file_name, graph):
        """create deafult config"""
        CONFIGURER.set_ability(enable_retrain=True, enable_prune=False)
        CONFIGURER.create_default_config(
            file_name,
            graph)
        LOGGER.logd("Create retrain config by file success!",
                    module_name='RetrainConfig')

    @staticmethod
    @check_params(graph=Graph, config_defination=(str, type(None)))
    def create_prune_config(graph, config_defination):
        """
        Function: create pruning configs based on config_defination for the given graph.
        Parameters:
        graph: model graph
        config_defination: path to the config file
        Return: created retrain_config
        """
        retrain_config = {}
        config_file = os.path.realpath(config_defination)
        CONFIGURER.create_config_from_proto(retrain_config, graph, config_file)

        LOGGER.logi(
            "Create quant config success!", module_name='RetrainConfig')
        return retrain_config

    @classmethod
    def un_init(cls):
        """uninitialize class"""
        cls.__initialized = False

    @classmethod
    @check_params(config_file=str, record_file=str, graph=Graph)
    def init_retrain(cls, config_file, record_file, graph):
        """
        Function: init the RetrainConfig.
        Inputs:
            config_file: a string, the file containing the quant config.
            record_file: a string, the file storing scale and offset.
            graph: IR Graph
        Returns: None
        """
        cls.set_ability(enable_retrain=True, enable_prune=False)
        cls.retrain_config = cls.parse_retrain_config(
            os.path.realpath(config_file), graph)
        RetrainConfig.__initialized = True
        RetrainConfig.__record_file_path = os.path.realpath(record_file)

    @classmethod
    def init(cls, graph, config_defination, enable_retrain=False, enable_prune=True):
        """
        Function: initialize pruning based on config_defination.
        """
        cls.set_ability(enable_retrain=enable_retrain, enable_prune=enable_prune)
        cls.retrain_config = cls.create_prune_config(graph, config_defination)
        if enable_retrain and enable_prune:
            cls.enable_retrain = CONFIGURER.enable_retrain
            cls.enable_prune = CONFIGURER.enable_prune
        cls.__initialized = True

    @classmethod
    def amc_init(cls, graph, config_defination, enable_retrain=False, enable_prune=True):
        """
        Function: initialize pruning based on amc config.
        """
        cls.set_ability(enable_retrain=enable_retrain, enable_prune=enable_prune)
        config_helper = AutoChannelPruneConfigHelper(graph, config_defination, GraphQuerier, CAPACITY)
        cls.retrain_config = config_helper.create_prune_config()
        cls.__initialized = True

    @classmethod
    def set_ability(cls, enable_retrain, enable_prune):
        cls.enable_retrain = enable_retrain
        cls.enable_prune = enable_prune
        CONFIGURER.set_ability(enable_retrain, enable_prune)
        LOGGER.logi('enable_retrain is {}, enable_prune is {}'.format(enable_retrain, enable_prune),
                    'RetrainConfig')

    def get_retrain_config(self, layer):
        """get retrain config"""
        if not self.retrain_enable(layer):
            return None
        return self.__retrain_config.get(layer)

    def get_layer_prune_config(self, layer):
        """
        Function: get prune config for the given layer
        """
        if not self.filter_prune_enable(layer) and not self.selective_prune_enable(layer):
            return None
        return self.__retrain_config.get(layer).get("regular_prune_config")

    def retrain_enable(self, layer):
        """if retrain enable"""
        if layer not in self.__retrain_config.keys():
            return False
        # avoid KeyError
        if self.__retrain_config.get(layer).get('retrain_enable') is None:
            return False
        return self.__retrain_config.get(layer).get('retrain_enable')

    def filter_prune_enable(self, layer):
        """
        Function: check if filter pruning is enable for the given layer.
        """
        if layer not in self.__retrain_config.keys():
            return False
        if not self.__retrain_config.get(layer).get('regular_prune_enable'):
            return False
        # is filter prune
        if self.__retrain_config.get(layer).get('regular_prune_config').get('algo') \
            in ["balanced_l2_norm_filter_prune"]:
            return True
        return False

    def selective_prune_enable(self, layer):
        """
        Function: check if pruning is enable for the given layer.
        """
        if layer not in self.__retrain_config.keys():
            return False
        if not self.__retrain_config.get(layer).get('regular_prune_enable'):
            return False
        # is selective prune
        if self.__retrain_config.get(layer).get('regular_prune_config').get('algo') in ["l1_selective_prune"]:
            return True
        return False

    def get_layer_config(self, layer):
        """get retrain layer weight config"""
        layer_config = self.get_retrain_config(layer)
        if layer_config is None:
            return None

        config = {}
        config[WEIGHT_QUANT_PARAMS] = {}
        config.get(WEIGHT_QUANT_PARAMS)['num_bits'] = 8
        config.get(WEIGHT_QUANT_PARAMS)['with_offset'] = False
        config.get(WEIGHT_QUANT_PARAMS)['wts_algo'] = 'arq_quantize'
        config.get(WEIGHT_QUANT_PARAMS)['channel_wise'] = \
                layer_config.get('retrain_weight_config')['channel_wise']

        return config

    def get_quant_config(self):
        """get retrain config"""
        config = copy.deepcopy(self.__retrain_config)
        config['activation_offset'] = True

        return config

    def get_skip_fusion_layers(self):
        """Get layers that need to skip do fusion"""
        return self.__skip_fusion_layers
