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

the file of information structure class

"""

import numpy as np

STATUS = 'status'
LAYER_NAME = 'layer_name'
INT8 = 'int8'


class PerformanceInfo(): # pylint: disable=R0902
    """
    The performance information class is used to store the relevant
    information of the model performance before and after quantization.
    """
    def __init__(self):
        self._main_layers = []
        self._auxi_layers = []
        self._main_types = {}
        self._fp_names = {}
        self._fp_times = {}
        self._int8_names = {}
        self._int8_times = {}
        self._benefits = {}
        self._status = {}

    def update_benefits_status(self, quant_info, iter_num):
        """Update earnings status before and after quantification."""
        fp_times = np.array(self._fp_times.get(iter_num))
        int8_times = np.array(self._int8_times.get(iter_num))

        self.set_benefits(iter_num, fp_times - int8_times)
        status = []
        for layer_name in self._main_layers:
            if quant_info.get_quant_config()[layer_name].get('quant_enable'):
                status.append(INT8)
            else:
                if iter_num == 1 and layer_name not in \
                    quant_info.get_not_quant_layers():
                    status.append(INT8)
                else:
                    status.append('float')
        self.set_status(iter_num, status)

    def get_main_layers(self):
        """Get a quantifiable list of main layers."""
        return self._main_layers

    def get_auxi_layers(self):
        """
        Gets the list of auxiliary quantization layers of the quantifiable
        main layer.
        """
        return self._auxi_layers

    def get_main_types(self):
        """Get quantifiable type dictionary of main layer."""
        return self._main_types

    def get_fp_names(self):
        """Get the unquantified layer name dictionary."""
        return self._fp_names

    def get_fp_times(self):
        """Get the unquantified layer time dictionary."""
        return self._fp_times

    def get_int8_names(self):
        """Get the quantified layer name dictionary."""
        return self._int8_names

    def get_int8_times(self):
        """Get the quantified layer time dictionary."""
        return self._int8_times

    def get_benefits(self):
        """Get the layer name dictionary of quantified income."""
        return self._benefits

    def get_status(self):
        """Get the quantization status of each layer of the current network."""
        return self._status

    def set_main_layers(self, main_layers):
        """Set a quantifiable list of main layers."""
        self._main_layers = main_layers

    def set_auxi_layers(self, auxi_layers):
        """
        Sets the list of auxiliary quantization layers of the quantifiable
        main layer.
        """
        self._auxi_layers = auxi_layers

    def set_main_types(self, main_types):
        """Set quantifiable type dictionary of main layer."""
        self._main_types = main_types

    def set_fp_names(self, iter_num, fp_names):
        """Set the unquantified layer name dictionary."""
        self._fp_names[iter_num] = fp_names

    def set_fp_times(self, iter_num, fp_times):
        """Set the unquantified layer time dictionary."""
        self._fp_times[iter_num] = fp_times

    def set_int8_names(self, iter_num, int8_names):
        """Set the quantified layer time dictionary."""
        self._int8_names[iter_num] = int8_names

    def set_int8_times(self, iter_num, int8_times):
        """Set the quantified layer name dictionary."""
        self._int8_times[iter_num] = int8_times

    def set_benefits(self, iter_num, benefits):
        """Set the layer name dictionary of quantified income."""
        self._benefits[iter_num] = benefits

    def set_status(self, iter_num, status):
        """Set the quantization status of each layer of the current network."""
        self._status[iter_num] = status


class QuantInfo(): # pylint: disable=R0902,R0904
    """The quantitative information class, save quantitative information.
    """
    def __init__(self):
        self._main_layers = []
        self._auxi_layers = []
        self._main_types = {}
        self._not_quant_layers = []
        self._total_records = {}
        self._quant_config = {'version': 1, 'skip_fusion_layers': []}

        self._group_layers = []
        self._search_layers = {}
        self._pre_fallback_layers = []

        self._roll_back_layers = []

        self._cin_cout_info = {}

    def add_items(self, # pylint: disable=R0913
                  layer_info,
                  quant_enable,
                  skip_fusion=False):
        """
        Add elements to the quantitative information class.
        layer_info: a dict, recording layer_name, auxi_layer, layer_type
        """
        self._main_layers.append(layer_info.get(LAYER_NAME))
        self._auxi_layers.append(layer_info.get('auxi_layer'))
        self._main_types[layer_info.get(LAYER_NAME)] = layer_info.get('layer_type')

        if not quant_enable:
            self._not_quant_layers.append(layer_info.get(LAYER_NAME))

        if skip_fusion:
            self._quant_config.get('skip_fusion_layers').append(layer_info.get(LAYER_NAME))
        self._quant_config[layer_info.get(LAYER_NAME)] = {
            'quant_enable': quant_enable,
            'weight_quant_params': {
                'channel_wise': False
            }
        }

    def add_search_layers(self, group_layer):
        """Add quantization layer to quantization fallback search space."""
        self._search_layers[tuple(group_layer)] = {
            STATUS: 'int8',
            'performance_benefits': 0
        }

    def change_search_layers_status(self, group_layer):
        """Modifying the state in the quantization backoff search space."""
        if self._search_layers.get(tuple(group_layer)).get(STATUS) == INT8:
            self._search_layers.get(tuple(group_layer))[STATUS] = 'float'
        else:
            self._search_layers.get(tuple(group_layer))[STATUS] = INT8

    def change_search_layers_benefits(self, group_layer, performance_benefits):
        """Modifying performance gains in quantitative backoff search space."""
        self._search_layers.get(tuple(group_layer))['performance_benefits'] = performance_benefits

    def add_pre_fallback_layers(self, fallback_layer):
        """Add pre fallback layer."""
        self._pre_fallback_layers.append(fallback_layer)

    def add_cin_cout_info(self, layer_name, cin_cout_info):
        """
        Add the CIN / cout information of the specified layer of the network.
        """
        self._cin_cout_info[layer_name] = cin_cout_info

    def add_total_records(self, layer_name, record):
        """Add network quantization factor information."""
        self._total_records[layer_name] = record

    def add_group_layers(self, group_layer):
        """Add network quantitative group information."""
        self._group_layers.append(group_layer)

    def get_pre_fallback_layers(self):
        """Get pre fallback layer."""
        return self._pre_fallback_layers

    def get_search_layers(self):
        """Get the search space of fallback layer."""
        return self._search_layers

    def get_cin_cout_info(self):
        """
        Get the CIN / cout information of the specified layer of the network.
        """
        return self._cin_cout_info

    def get_group_layers(self):
        """Get network quantitative group information."""
        return self._group_layers

    def get_main_layers(self):
        """Get quantifiable type dictionary of main layer."""
        return self._main_layers

    def get_auxi_layers(self):
        """
        Gets the list of auxiliary quantization layers of the quantifiable
        main layer.
        """
        return self._auxi_layers

    def get_main_types(self):
        """Get quantifiable type dictionary of main layer."""
        return self._main_types

    def get_total_records(self):
        """Get network quantization factor information."""
        return self._total_records

    def get_quant_config(self):
        """Get the quantitative configuration information of the network."""
        return self._quant_config

    def get_not_quant_layers(self):
        """Get the list of layer names that are not quantified in the network.
        """
        return self._not_quant_layers

    def get_roll_back_layers(self):
        """Get roll back layers of the whole net."""
        roll_back_layers = []
        for layer_name in self._main_layers:
            if not self._quant_config.get(layer_name).get('quant_enable'):
                roll_back_layers.append(layer_name)
        roll_back_layers = list(
            set(roll_back_layers) - set(self._not_quant_layers))
        return roll_back_layers

    def set_total_records(self, layer_name, weight_scale, weight_offset,
                          shift_n):
        """Set network quantization factor information."""
        self._total_records.get(layer_name)['weight_scale'] = weight_scale
        self._total_records.get(layer_name)['weight_offset'] = weight_offset
        self._total_records.get(layer_name)['shift_n'] = shift_n

    def set_config_enable(self, layer_name, quant_enable):
        """
        Set the quantitative enabling information of the specified layer of
        the network.
        """
        if self._quant_config.get(layer_name) is None:
            raise TypeError(
                'self._quant_config[{}] is None, except isn\'t None.'.format(
                    layer_name))
        self._quant_config.get(layer_name)['quant_enable'] = quant_enable

    def set_config_channelwise(self, layer_name, channel_wise):
        """
        Set the channel quantization information of the specified layer of the
        network.
        """
        if self._quant_config.get(layer_name) is None:
            raise TypeError(
                'self._quant_config[{}] is None, except isn\'t None.'.format(
                    layer_name))
        self._quant_config.get(layer_name).get('weight_quant_params')[
            'channel_wise'] = channel_wise
