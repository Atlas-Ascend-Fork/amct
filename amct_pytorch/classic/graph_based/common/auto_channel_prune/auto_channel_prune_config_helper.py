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
import stat
from google.protobuf import text_format

from ...proto.basic_info_pb2 import AutoChannelPruneConfig
from ...proto import retrain_config_pb2
from ..utils import util
from ..retrain_config.retrain_proto import RetrainProtoConfig
from ..utils import files as files_util


FILTER_PRUNE = 'filter_prune'
OVERRIDE_FIELDS = ['skip_layers', 'skip_layer_types', 'regular_prune_skip_layers', 'regular_prune_skip_types',
    'override_layer_configs', 'override_layer_types']


class AutoChannelPruneConfigHelper:
    """ help to parse and check auto_channel_prune_search config """
    def __init__(self, graph, config_file, graph_querier, capacity):
        """ Init function. """
        self.config_file = config_file
        self.config_proto = self._read_file(config_file)
        self.support_types = capacity.get_value('PRUNABLE_TYPES')
        self.support_layers = graph_querier.get_support_prune_layer2type(graph)
        self._check_params()
        self.search_layers = self.support_layers.keys()
        self.override_proto_config = retrain_config_pb2.AMCTRetrainConfig()
        self.capacity = capacity
        if self.override_prune_cfg:
            self._check_override_prune_cfg()

    @property
    def compress_ratio(self):
        """ get compress_ratio. """
        return util.proto_float_to_python_float(self.config_proto.compress_ratio)

    @property
    def ascend_optimized(self):
        """ get ascend_optimized. """
        return self.config_proto.ascend_optimized

    @property
    def max_prune_ratio(self):
        """ get max_prune_ratio. """
        return self.config_proto.max_prune_ratio

    @property
    def test_iteration(self):
        """ get test_iteration. """
        return self.config_proto.test_iteration

    @property
    def override_prune_cfg(self):
        """ get override_prune_cfg. """
        if not self.config_proto.HasField('override_prune_cfg'):
            return None
        return os.path.realpath(self.config_proto.override_prune_cfg)

    @classmethod
    def _read_file(cls, config_file):
        """ Read one file to parse proto.

        Args:
            config_file (string): file to read and parse.
            proto_init_func (function): function to initialize proto.

        Raises:
            RuntimeError: content in file doesn't match proto.

        Returns:
            proto parsed.
        """
        config_file = os.path.realpath(config_file)
        config_proto = AutoChannelPruneConfig()
        with open(config_file, 'r') as fid:
            pbtxt_string = fid.read()
            try:
                text_format.Merge(pbtxt_string, config_proto)
            except text_format.ParseError as e:
                raise RuntimeError(
                    "the config_file {} cannot be parsered, please ensure "\
                    "it matches with {}!"
                    .format(config_file, 'AutoChannelPruneConfig')) from e

        return config_proto

    def create_prune_config(self):
        """ create prune_config for layers to be searched. """
        default_prune_config = dict()
        default_prune_config['prune_type'] = FILTER_PRUNE
        default_prune_config['prune_ratio'] = None
        default_prune_config['ascend_optimized'] = self.ascend_optimized
        default_prune_config['algo'] = 'balanced_l2_norm_filter_prune'
        default_config = dict()
        default_config['regular_prune_enable'] = True
        default_config['regular_prune_config'] = default_prune_config
        config = dict()
        for layer in self.search_layers:
            config[layer] = default_config

        return config

    def create_final_config(self, search_prune_config, output_config):
        """ create auto prune channel result config.

        Args:
            config_file (string): user file to read.
            search_prune_config (dict): prune config <layer_name: >.
            output_config (string): output config file

        Returns:
            None.
        """

        files_util.create_empty_file(output_config, check_exist=True)

        proto_config = self._parser_prune_config(search_prune_config)
        file_flags = os.O_WRONLY + os.O_CREAT + os.O_TRUNC
        file_mode = stat.S_IRUSR + stat.S_IWUSR + stat.S_IRGRP
        with os.fdopen(os.open(output_config, file_flags, file_mode), 'w',
                    encoding='UTF-8', newline='') as fid:
            fid.write(text_format.MessageToString(proto_config, as_utf8=True))

    def _check_params(self):
        """ check config in config_proto """
        if not self.config_proto.ListFields():
            raise ValueError("config file is empty.")
        if not self.support_layers:
            raise ValueError("graph has no layer support channel prune.")
        if self.config_proto.compress_ratio <= 1:
            raise ValueError("compress_ratio not supported. compress_ratio should be larger than 1.")
        if self.max_prune_ratio <= 0 or self.max_prune_ratio > 1:
            raise ValueError("max_prune_ratio not supported. max_prune_ratio should be in (0, 1].")
        if self.config_proto.test_iteration < 1:
            raise ValueError("test_iteration not supported. test_iteration should be no smaller than 1.")

    def _check_override_prune_cfg(self):
        """ check override_prune_cfg and update search_layers:
        1. file exists
        2. config fields only contain override or skip config
        3. override configs only contain filter_pruner config
        4. layers in override or skip config are supported filter_pruner layer
        5. search_layers is not empty
        """
        if not os.path.exists(self.override_prune_cfg):
            raise ValueError("The {} in AutoChannelPruneConfig does not exist, please check the file path."
                             .format(self.override_prune_cfg))
        proto = RetrainProtoConfig(self.override_prune_cfg, self.capacity)
        self.override_proto_config = proto.proto_config

        fields = [desc.name for desc, val in proto.proto_config.ListFields()]
        if not set(fields).issubset(set(OVERRIDE_FIELDS)):
            raise ValueError("override_prune_cfg should contain only override or skip config.")

        _, retrain_override_layers, prune_override_layers = proto.get_override_layers()
        if retrain_override_layers or not \
            set(prune_type for prune_type, layer in prune_override_layers.items() if layer).issubset([FILTER_PRUNE]):
            raise ValueError("override_layer_configs in override_prune_cfg should contain only filter_pruner config.")
        if not set(prune_override_layers.get(FILTER_PRUNE)).issubset(set(self.support_layers.keys())):
            raise ValueError("some override_layer not in valid_layers for filter prune")

        _, retrain_override_types, prune_override_types = proto.get_override_layer_types()
        if retrain_override_types or not \
            set(prune_type for prune_type, types in prune_override_types.items() if types).issubset([FILTER_PRUNE]):
            raise ValueError("override_layer_types in override_prune_cfg should contain only filter_pruner config.")
        if not set(prune_override_types.get(FILTER_PRUNE)).issubset(set(self.support_types)):
            raise ValueError("some override_types not supported for filter prune")

        regular_prune_skip_layers = proto.get_regular_prune_skip_layers()
        if not set(regular_prune_skip_layers).issubset(set(self.support_layers.keys())):
            raise ValueError("some regular_prune_skip_layers not in valid_layers")

        regular_prune_skip_types = proto.get_regular_prune_skip_types()
        if not set(regular_prune_skip_types).issubset(set(self.support_types)):
            raise ValueError("some regular_prune_skip_types not supported")

        self.search_layers = []
        for layer, layer_type in self.support_layers.items():
            if layer not in prune_override_layers.get(FILTER_PRUNE) + regular_prune_skip_layers and \
                layer_type not in prune_override_types.get(FILTER_PRUNE) + regular_prune_skip_types:
                self.search_layers.append(layer)
        if not self.search_layers:
            raise ValueError("no layer to be searched, please make sure not all layers are overrided or skipped.")

    def _parser_prune_config(self, search_prune_config):
        """parser prune config
        Args:
            config_file: user file to read.
            search_prune_config: prune config <layer_name: prune_config list (0-prune, 1-remain)>.

        Returns:
            proto parsed.
        """
        # global prune config
        self.override_proto_config.prune_config.filter_pruner.balanced_l2_norm_filter_prune.prune_ratio = 0.3
        self.override_proto_config.prune_config.filter_pruner.balanced_l2_norm_filter_prune.ascend_optimized = \
            self.ascend_optimized

        # search override/skip layer config
        for layer_name, prune_config in search_prune_config.items():
            sum_prune = sum(prune_config)
            len_prune = len(prune_config)

            # skip layer
            if sum_prune == len_prune:
                self.override_proto_config.regular_prune_skip_layers.append(layer_name)
                continue

            prune_ratio = 1 - sum_prune / len_prune
            proto_override = retrain_config_pb2.RetrainOverrideLayer()
            proto_override.layer_name = layer_name
            proto_override.prune_config.filter_pruner.balanced_l2_norm_filter_prune.prune_ratio = prune_ratio
            proto_override.prune_config.filter_pruner.balanced_l2_norm_filter_prune.ascend_optimized = \
                self.ascend_optimized
            self.override_proto_config.override_layer_configs.append(proto_override)

        return self.override_proto_config
