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

import copy
from google.protobuf import text_format

from ...proto import retrain_config_pb2


class ConfigProtoModifier:
    """ class of Modifying config proto. """
    def __init__(self, cfg_file, override_cfg_file, bit_config, layer_types):
        """ Init func.

        With quant config(AMCTConfig or AMCTRetrainConfig) and bit config, to generate a new quant config. For new
        quant config, each layer's num_bit info is from bit_config and others is from override_cfg_file. Finally save
        new quant config to a file.

        Args:
            cfg_file (string): file to save new quant config.
            override_cfg_file (string): file whose content is a quant config.
            bit_config (dict): key is layer and value is num_bit for quant.
            quant_config_proto (proto): proto format, it has AMCTConfig or AMCTRetrainConfig.
            layer_types (dict): key is layer and value is node's type.
        """
        self.cfg_file = cfg_file
        self.override_cfg_file = override_cfg_file
        self.bit_config = bit_config
        self.layer_types = layer_types

    @staticmethod
    def parse(override_cfg_file):
        """ Parse proto config file to proto

        Args:
            override_cfg_file (string): config file from proto.
        """
        pass

    @staticmethod
    def save(cfg_proto, cfg_file):
        """ Save cfg_proto to file.

        Args:
            cfg_proto (AMCTConfig/AMCTRetrainConfig): proto format.
            cfg_file (string): file to save cfg_proto.
        """
        with open(cfg_file, 'w') as fid:
            fid.write(text_format.MessageToString(cfg_proto, as_utf8=True))

    def process(self):
        """ Do modify really. """
        pass


class QatConfigProtoModifier(ConfigProtoModifier):
    """ class of Modifying AMCTRetrainConfig proto. """
    def __init__(self, cfg_file, override_cfg_file, bit_config, layer_types):
        """ Init func.

        With quant config(AMCTConfig or AMCTRetrainConfig) and bit config, to generate a new quant config. For new
        quant config, each layer's num_bit info is from bit_config and others is from override_cfg_file. Finally save
        new quant config to a file.

        Args:
            cfg_file (string): file to save new quant config.
            override_cfg_file (string): file whose content is a quant config.
            bit_config (dict): key is layer and value is num_bit for quant.
            layer_types (dict): key is layer and value is node's type.
        """
        super(QatConfigProtoModifier, self).__init__(cfg_file, override_cfg_file, bit_config, layer_types)

    @staticmethod
    def set_skip(qat_cfg_proto, bit_config):
        """ Set skip layers for qat. Layer will be skiped if it's 16 in bit_config.

        Args:
            qat_cfg_proto (AMCTRetrainConfig): proto format.
            bit_config (dict): key is layer and value is num_bit for qat.
        """
        skip_layers = [layer for layer, num_bit in bit_config.items() if num_bit == 16]
        qat_cfg_proto.quant_skip_layers.extend(skip_layers)
        for layer in skip_layers:
            del bit_config[layer]

    @staticmethod
    def add_override_layer(qat_cfg_proto, copy_proto, layer_name, num_bit):
        """ Add one override_layer_configs in qat_cfg_proto with layer_name, num_bit, copy_proto.

        Args:
            qat_cfg_proto (AMCTRetrainConfig): [description]
            copy_proto (proto): include retrain_data_quant_config and retrain_weight_quant_config, from which algo from
            layer_name (string): name for new override_layer_configs.
            num_bit (int): bit for new override_layer_configs.
        """
        layer_proto = qat_cfg_proto.override_layer_configs.add()
        layer_proto.layer_name = layer_name
        layer_proto.retrain_data_quant_config.CopyFrom(copy_proto.retrain_data_quant_config)
        layer_proto.retrain_weight_quant_config.CopyFrom(copy_proto.retrain_weight_quant_config)
        QatConfigProtoModifier.modify_weight_quant_config(num_bit, layer_proto.retrain_weight_quant_config)
        QatConfigProtoModifier.modify_data_quant_config(num_bit, layer_proto.retrain_data_quant_config)

    @staticmethod
    def add_override_type(qat_cfg_proto, copy_proto, num_bit):
        """ Add one override_layer_types in qat_cfg_proto with num_bit, copy_proto.

        Args:
            qat_cfg_proto (AMCTRetrainConfig): [description]
            copy_proto (proto): include retrain_data_quant_config and retrain_weight_quant_config, from which algo from
            num_bit (int): bit for new override_layer_configs.
        """
        layer_proto = qat_cfg_proto.override_layer_types.add()
        layer_proto.CopyFrom(copy_proto)
        QatConfigProtoModifier.modify_weight_quant_config(num_bit, layer_proto.retrain_weight_quant_config)
        QatConfigProtoModifier.modify_data_quant_config(num_bit, layer_proto.retrain_data_quant_config)

    @staticmethod
    def parse(override_cfg_file):
        """ Parse proto config file to proto

        Args:
            override_cfg_file (string): config file from proto.

        Returns:
            AMCTRetrainConfig: config proto parsed from file.
        """
        qat_cfg_proto = retrain_config_pb2.AMCTRetrainConfig()
        if override_cfg_file:
            with open(override_cfg_file, 'rb') as cfg_file:
                pbtxt_string = cfg_file.read()
                text_format.Merge(pbtxt_string, qat_cfg_proto)
            for item in ['skip_layers', 'skip_layer_types', 'quant_skip_layers', 'quant_skip_types']:
                qat_cfg_proto.ClearField(item)
        return qat_cfg_proto

    @staticmethod
    def modify_data_quant_config(num_bit, data_config):
        """ Modify data_config's num_bit info.

        Args:
            num_bit (int): bit for data_config.
            data_config (RetrainDataQuantConfig): config to modify num_bit info.
        """
        mapping = {'4': 'INT4', '8': 'INT8', '16': 'INT16'}
        # if algo is uncertain, use default
        data_config.ulq_quantize.dst_type = getattr(retrain_config_pb2.DataType, mapping.get(str(num_bit)))

    @staticmethod
    def modify_weight_quant_config(num_bit, weight_config):
        """ Modify weight_config's num_bit info.

        Args:
            num_bit (int): bit for weight_config.
            weight_config (RetrainDataQuantConfig): config to modify num_bit info.
        """
        mapping = {'4': 'INT4', '8': 'INT8'}
        if weight_config.HasField('ulq_retrain'):
            weight_config.ulq_retrain.dst_type = getattr(retrain_config_pb2.DataType, mapping.get(str(num_bit)))
            return
        # if algo is uncertain, use default
        weight_config.arq_retrain.dst_type = getattr(retrain_config_pb2.DataType, mapping.get(str(num_bit)))

    @staticmethod
    def cmp_count(count):
        """ compare count of layer in each quant bit.

        Args:
            count (dict): key id quant bit, value is list of layer name
        """
        if len(count[8]) >= len(count[4]):
            max_bit = 8
            min_bit = 4
        else:
            max_bit = 4
            min_bit = 8
        return max_bit, min_bit

    def process(self):
        """ Do modify really. """
        # step0: prepare proto
        override_cfg_proto = self.parse(self.override_cfg_file)
        bit_config = copy.deepcopy(self.bit_config)
        qat_cfg_proto = retrain_config_pb2.AMCTRetrainConfig()

        # step1: skip layer not for quant
        self.set_skip(qat_cfg_proto, bit_config)
        # step2: override layer with override_layer_proto
        self.process_override_layer(qat_cfg_proto, override_cfg_proto, bit_config)
        # step3: override layers with override_type_proto
        self.process_override_type(qat_cfg_proto, override_cfg_proto, bit_config)
        # step4: process layer with common config
        self.process_common(qat_cfg_proto, override_cfg_proto, bit_config)

        # step5: save to file
        self.save(qat_cfg_proto, self.cfg_file)

    def process_override_layer(self, qat_cfg_proto, override_cfg_proto, bit_config):
        """ Process override_layer in override_cfg_proto to generate new config in qat_cfg_proto.

        For each layer's override_layer_configs, the num_bit is from bit_config and others is from override_cfg_proto.

        Args:
            qat_cfg_proto (AMCTRetrainConfig): new config to generate.
            override_cfg_proto (AMCTRetrainConfig): including original config, according to which to generate new one.
            bit_config (dict): key is layer and value is num_bit for qat.
        """
        override_layer_protos = {}
        for proto in override_cfg_proto.override_layer_configs:
            override_layer_protos[proto.layer_name] = proto

        for layer in override_layer_protos:
            if layer not in bit_config:
                continue
            self.add_override_layer(qat_cfg_proto, override_layer_protos[layer], layer, bit_config[layer])
            del bit_config[layer]

    def process_override_type(self, qat_cfg_proto, override_cfg_proto, bit_config):
        """ Process override_layer_types in override_cfg_proto to generate new config in qat_cfg_proto.

        Args:
            qat_cfg_proto (AMCTRetrainConfig): new config to generate.
            override_cfg_proto (AMCTRetrainConfig): including original config, according to which to generate new one.
            bit_config (dict): key is layer and value is num_bit for qat.
        """
        override_type_protos = {}
        for proto in override_cfg_proto.override_layer_types:
            override_type_protos[proto.layer_type] = proto

        for node_type in override_type_protos:
            count = {4: [], 8: []}
            for layer, num_bit in bit_config.items():
                if self.layer_types.get(layer) == node_type:
                    count.get(num_bit).append(layer)
            type_bit, layer_bit = QatConfigProtoModifier.cmp_count(count)
            # some of layers is set by override_type
            if not count.get(type_bit):
                continue
            self.add_override_type(qat_cfg_proto, override_type_protos.get(node_type), type_bit)
            for layer in count.get(type_bit):
                del bit_config[layer]
            # some of layers is set by override_layer
            if not count.get(layer_bit):
                continue
            for layer in count.get(layer_bit):
                self.add_override_layer(qat_cfg_proto, override_type_protos.get(node_type), layer, layer_bit)
                del bit_config[layer]

    def process_common(self, qat_cfg_proto, override_cfg_proto, bit_config):
        """ Process common in override_cfg_proto to generate new config in qat_cfg_proto.

        Args:
            qat_cfg_proto (AMCTRetrainConfig): new config to generate.
            override_cfg_proto (AMCTRetrainConfig): including original config, according to which to generate new one.
            bit_config (dict): key is layer and value is num_bit for qat.
        """
        count = {4: [], 8: []}
        for layer, num_bit in bit_config.items():
            count.get(num_bit).append(layer)
        common_bit, layer_bit = QatConfigProtoModifier.cmp_count(count)
        # some of common is set by common
        qat_cfg_proto.retrain_data_quant_config.CopyFrom(override_cfg_proto.retrain_data_quant_config)
        self.modify_data_quant_config(common_bit, qat_cfg_proto.retrain_data_quant_config)
        qat_cfg_proto.retrain_weight_quant_config.CopyFrom(override_cfg_proto.retrain_weight_quant_config)
        self.modify_weight_quant_config(common_bit, qat_cfg_proto.retrain_weight_quant_config)
        for layer in count.get(common_bit):
            del bit_config[layer]
        # some of common is set by override_layer
        for layer in count.get(layer_bit):
            self.add_override_layer(qat_cfg_proto, override_cfg_proto, layer, layer_bit)
            del bit_config[layer]

        # copy global param to new config
        qat_cfg_proto.batch_num = override_cfg_proto.batch_num
