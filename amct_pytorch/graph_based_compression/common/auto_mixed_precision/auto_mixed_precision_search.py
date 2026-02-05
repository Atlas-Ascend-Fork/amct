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
import json
import shutil
from datetime import datetime
from datetime import timezone
from datetime import timedelta
import numpy as np

from ..utils import struct_helper
from ..utils import files as files_util
from ..utils import vars_util
from .interger_programming import BranchBound
from .modify_retrain_proto import QatConfigProtoModifier

from ...utils.log import LOGGER


class AutoSearchMixedPrecisionBase:
    """ Superclass class of auto search quant bit
    """
    def __init__(self, graph_info, cali_config_info, auto_quant_info):
        """ Init to add essential info.

        Args:
            graph_info (struct_helper.GraphInfoBase): struct including graph/model/net info
            cali_config_info (struct_helper.CalibrationConfigInfo): struct including calibration info
            auto_quant_info (struct_helper.AutoSearchMixedPrecisionInfo): struct including auto quant info
        """
        self.graph_info = graph_info
        self.cali_config_info = cali_config_info
        self.auto_quant_info = auto_quant_info
        self.temp_info = None

    @staticmethod
    def union_search_range(bit_search_range):
        """ Union search range, so several layers can has same choice.

        Args:
            bit_search_range (dict): key is layer and value is search_range.

        Returns:
            dict: new search search_range
            dict: key is union_name and value is list of layer to union.
        """
        search_range = bit_search_range
        union_info = {}
        return search_range, union_info

    @staticmethod
    def unravel_search_range(best_path, union_info):
        """ Unravel search range. Layers to union has same choice/config with union_name.

        Args:
            best_path (dict): the choice for each layer
            union_info (dict): key is union_name and value is list of layer to union.

        Returns:
            dict: the choice for each layer, adding layers to union and delete union_name.
        """
        for key in union_info:
            for map_key in union_info[key]:
                best_path[map_key] = best_path[key]
            del best_path[key]
        return best_path

    @staticmethod
    def sort_dict(order_keys, dict_to_sort):
        """ Sort a dict according a sequence.

        Args:
            order_keys (list): sequence the new dict will in its order.
            dict_to_sort (dict): dict to sort.

        Returns:
            dict: new dict in order.
        """
        order_dict = {}
        for key in order_keys:
            if key in dict_to_sort:
                order_dict[key] = dict_to_sort[key]
        for key in dict_to_sort:
            if key not in order_dict:
                order_dict[key] = dict_to_sort[key]
        return order_dict

    @staticmethod
    def trans_config_to_qat_cfg(qat_cfg_file, override_qat_cfg_file, bit_config_json, layer_types):
        """Trans bits config info to qat config file.

        According to bits config json and qat config file(from retrain proto), to generate a new qat config file(from
        retrain proto). In which, the bit info is from bits config json and other info like algo is from qat config
        file.

        Args:
            qat_cfg_file (string): to save new generated qat proto config.
            override_qat_cfg_file (string): According to it generate qat file, including algo info
            bit_config_json (string): According to it generate qat file, including bit info
            layer_types (dict): layer name and its type info
        """
        LOGGER.logi("generate config file for model is quantized in quantization aware training.",
                    'AutoSearchMixedPrecisionBase')
        with open(bit_config_json, 'r') as fid:
            bit_config = json.load(fid)
        simple_bit_config = {}
        for layer in bit_config:
            simple_bit_config[layer] = bit_config[layer]['num_bits']

        modifier = QatConfigProtoModifier(qat_cfg_file, override_qat_cfg_file, simple_bit_config, layer_types)
        modifier.process()

    @staticmethod
    def _add_info_for_union(union_name, to_union_names, search_range, acc_decay, computes_ops):
        """ Add search info(search_range, acc_decay, computes_ops) for one union.

        Args:
            union_name (string): new 'layer' to represent several layers.
            to_union_names (list of string): layers to be represented.
            search_range (dict): key is layer and value is its search_range.
            acc_decay (dict): key is layer and value is its acc decay in different bit.
            computes_ops (dict): key is layer and value is its computes_ops(like bitops)) in different bit.
        """
        # add search range for union_name and delete search for layer to union.
        search_range[union_name] = search_range[to_union_names[0]]
        for key in to_union_names:
            del search_range[key]
        # add acc decay for union_name, use add operation considerating all layers are added.
        acc_decay[union_name] = copy.deepcopy(acc_decay[to_union_names[0]])
        for name in to_union_names[1:]:
            for choice in acc_decay[name]:
                acc_decay[union_name][choice] += acc_decay[name][choice]
        # add computes ops for union_name, use add operation considerating gropu_conv is such.
        computes_ops[union_name] = copy.deepcopy(computes_ops[to_union_names[0]])
        for name in to_union_names[1:]:
            for choice in computes_ops[name]:
                computes_ops[union_name][choice] += computes_ops[name][choice]

    def cal_node_bitops(self):
        """ Calculate bitops of quantizable layers in graph.

        Raises:
            NotImplementedError: function hasn't been implemented yet.
        """
        raise NotImplementedError("Method or function hasn't been implemented yet.")

    def layer_sensitivity_analysis(self):
        """[summary]

        Args:
            quant_mix_bits ([type]): [description]

        Raises:
            NotImplementedError: function hasn't been implemented yet.
        """
        raise NotImplementedError("Method or function hasn't been implemented yet.")

    def prepare_calibration_config(self, config_defination):
        """ prepare calibration config.

        Raises:
            NotImplementedError: function hasn't been implemented yet.
        """
        raise NotImplementedError("Method or function hasn't been implemented yet.")

    def prepare_qat_config(self, config_defination):
        """ prepare quant retrain config.

        Raises:
            NotImplementedError: function hasn't been implemented yet.
        """
        raise NotImplementedError("Method or function hasn't been implemented yet.")

    def do_global_calibration(self, num_bits):
        """ Do calibration to generate record with num_bit for whole graph.

        Args:
            num_bits (int): in which bit to do quant.

        Raises:
            NotImplementedError: function hasn't been implemented yet.
        """
        raise NotImplementedError("Method or function hasn't been implemented yet.")

    def get_support_layer2type(self, mode):
        """ Get support quant layer's name and type info.

        Args:
            mode (string): get in which quant scene. 'ptq' and 'qat' is valid.

        Raises:
            NotImplementedError: function hasn't been implemented yet.
        """
        raise NotImplementedError("Method or function hasn't been implemented yet.")

    def run(self):
        """ Main control process to search best quant bit config. """
        self.prepare()
        # prepare acc_decay and performance_data
        self.generate_record_files()
        self.generate_acc_decay()
        self.generate_computes_ops()
        # search qat config, ptq is as follows.
        self.search(mode='qat')
        self.trans_config_to_qat_cfg(self.auto_quant_info.qat_cfg_file,
                                     self.auto_quant_info.cfg_helper.override_qat_cfg,
                                     self.auto_quant_info.qat_precision_file,
                                     self.get_support_layer2type('qat'))

        self.tear_down()

    def prepare(self):
        """ Do preparation at first, including preparing some variables, creating file folder. """
        # make temp folder and prepare temp variable
        LOGGER.logi("prepare before strating searching.", 'AutoSearchMixedPrecisionBase')
        temp_dir = self._make_temp_dir()
        self.temp_info = struct_helper.AutoSearchMixedPrecisionTempInfo(temp_dir)
        # prepare search range, which layer is valid and which bit is valid
        ptq_layer2type = self.get_support_layer2type(mode='ptq')
        self.auto_quant_info.quant_layers = list(ptq_layer2type.keys())
        # if need qat, please add ptq_search_range as follow
        self.auto_quant_info.qat_search_range = self.prepare_search_range(mode='qat')
        # if giving quant config file, check it first.
        # if not giving quant config file, check model has PTQ or QAT layers.
        try:
            self.prepare_calibration_config(self.auto_quant_info.cfg_helper.ptq_cfg)
        except Exception as e:
            raise RuntimeError("Fail to create calibration config."
                               "Please check the error info for more information."
                               "Maybe wrong ptq_cfg or no supported layers for PTQ.") from e
        try:
            self.prepare_qat_config(self.auto_quant_info.cfg_helper.override_qat_cfg)
        except Exception as e:
            raise RuntimeError("Fail to create retrain config."
                               "Please check the error info for more information."
                               "Maybe wrong override_qat_cfg or no supported layers for QAT.") from e

    def tear_down(self):
        """ do some thing in the end. """
        LOGGER.logi("delete redundant files.", 'AutoSearchMixedPrecisionBase')
        # delete dump file
        shutil.rmtree(self.temp_info.dump_dir)
        # delete temp info in debug
        if not LOGGER.is_file_debug_level():
            shutil.rmtree(self.temp_info.temp_dir)

    def generate_record_files(self):
        """ Generate record files, including int4 and int8 for whole net. """
        LOGGER.logi("quantize model to generate scale and offset in different dest_type [4, 8].",
                    'AutoSearchMixedPrecisionBase')
        # generate a general config
        self.prepare_calibration_config(self.cali_config_info.config_defination)
        # do the global int4/int8 calibration to get records file
        for quant_bit in self.auto_quant_info.quant_mix_bits:
            if quant_bit == vars_util.FP16_BIT:
                continue
            self.set_global_num_bits(quant_bit)
            self.do_global_calibration(quant_bit)

    def generate_acc_decay(self):
        """ Generate acc_decay info, indicating the acc loss for each layer in each bit. """
        LOGGER.logi("analyze accurancy decay for model in different dest_type {}."
                    .format(self.auto_quant_info.quant_mix_bits), 'AutoSearchMixedPrecisionBase')
        self.analyze_sensitivity()
        self.auto_quant_info.acc_decay = self.sort_dict(self.auto_quant_info.quant_layers,
                                                        self.auto_quant_info.acc_decay)
        acc_decay = self.auto_quant_info.acc_decay

        # 添加类型转换函数
        def default_converter(obj):
            if hasattr(obj, 'dtype') and obj.dtype == np.float32:
                return float(obj)
            raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

        with open(self.temp_info.loss_info_file, 'w') as fid:
            json.dump(acc_decay, fid, indent=4, separators=(',', ':'), default=default_converter)

        self.auto_quant_info.shape_info = self.sort_dict(self.auto_quant_info.quant_layers,
                                                         self.auto_quant_info.shape_info)
        with open(self.temp_info.shape_info_file, 'w') as fid:
            json.dump(self.auto_quant_info.shape_info, fid, indent=4, separators=(',', ':'), default=default_converter)

    def generate_computes_ops(self):
        """ Generate performance info, indicating the performance for each layer in each bit. """
        LOGGER.logi("analyze computational cost for model in different dest_type {}."
                    .format(self.auto_quant_info.quant_mix_bits), 'AutoSearchMixedPrecisionBase')
        self.auto_quant_info.computes_ops = self.cal_node_bitops()
        self.auto_quant_info.computes_ops = self.sort_dict(self.auto_quant_info.quant_layers,
                                                           self.auto_quant_info.computes_ops)
        with open(self.temp_info.bitops_info_file, 'w') as fid:
            json.dump(self.auto_quant_info.computes_ops, fid, indent=4, separators=(',', ':'))

    def search(self, mode):
        """ Search the num bits config.

        Args:
            mode (string): search for which quant scene. 'ptq' and 'qat' is valid.
        """
        if mode == 'qat':
            bit_search_range = self.auto_quant_info.qat_search_range
        else:
            raise ValueError('only support "qat", but get {}'.format(mode))
        mode = "post training quantization" if mode == 'ptq' else "quantization aware training"
        LOGGER.logi("search best mixed precision config for model is quantized in {}.".format(mode),
                    'AutoSearchMixedPrecisionBase')
        # step2: prepare constrant preformance
        net_compute_ops = 0
        for layer in bit_search_range:
            net_compute_ops += self.auto_quant_info.computes_ops[layer][str(vars_util.FP16_BIT)]
        constraint = net_compute_ops / self.auto_quant_info.compression_ratio
        # step3: search the best config
        search_range, union_info = self.union_search_range(bit_search_range)
        search_helper = BranchBound(self.auto_quant_info.computes_ops, self.auto_quant_info.acc_decay,
                                    constraint_weight=constraint)
        best_path, sum_compute_ops, _ = search_helper.search(search_range)

        if not best_path:
            raise RuntimeError("Cannot search a config with constrant.")
        real_compress_ratio = net_compute_ops / sum_compute_ops
        LOGGER.logi('when configured compression_ratio is {}, the searched compression_ratio is {} for {}'
                    .format(self.auto_quant_info.compression_ratio, real_compress_ratio, mode),
                    'AutoSearchMixedPrecisionBase')
        best_path = self.unravel_search_range(best_path, union_info)
        # step4: save the config
        self.auto_quant_info.bits_config = {}
        for layer, value in best_path.items():
            self.auto_quant_info.bits_config[layer] = {"num_bits": int(value)}
        # sort the config th keep in order
        self.auto_quant_info.bits_config = self.sort_dict(self.auto_quant_info.quant_layers,
                                                          self.auto_quant_info.bits_config)
        # save the config to file
        with open(self.auto_quant_info.qat_precision_file, 'w') as fid:
            json.dump(self.auto_quant_info.bits_config, fid, indent=4, separators=(',', ':'))

    def set_global_num_bits(self, num_bits):
        """ Generate a new config file, enable all layers to do quant and set num_bits.

        Args:
            num_bits (int): which bit to do quant.
        """
        config_file = self.temp_info.config_file
        with open(config_file, 'r') as fid:
            quant_config = json.load(fid)

        for item in quant_config:
            if isinstance(quant_config[item], dict):
                # enable to do quant
                quant_config[item]['quant_enable'] = True
                # add bit info
                act_config = quant_config[item]["activation_quant_params"]
                weight_config = quant_config[item]["weight_quant_params"]
                act_config["num_bits"] = num_bits
                weight_config["num_bits"] = num_bits

        files_util.save_to_json(self.map_file(str(num_bits), 'config_file'), quant_config)

    def prepare_search_range(self, mode):
        """ Preapre search range for diffirent quant behaviours.

        Args:
            mode (string): preapre for which quant scene. 'ptq' and 'qat' is valid.

        Raises:
            ValueError: the mode'value is invalid.
            RuntimeError: the support quant layer is not in self.quant_layers, which means it has no acc_decay info
            and performance info.

        Returns:
            dict: bit_search_range, indicating each layer's bit range.
        """
        if mode not in ["ptq", "qat"]:
            raise ValueError('only support ["ptq", "qat"], but get {}'.format(mode))
        support_layer2type = self.get_support_layer2type(mode)
        search_limit = self.auto_quant_info.cfg_helper.quant_bit_limit
        self.auto_quant_info.cfg_helper.check_search_limit(search_limit, support_layer2type, mode)

        bit_search_range = {}
        for layer in support_layer2type:
            if layer not in self.auto_quant_info.quant_layers:
                raise RuntimeError('cannot search layer {} for it cannot do calibration.'.format(layer))
            if layer in search_limit:
                bit_search_range[layer] = search_limit[layer]

            elif support_layer2type[layer] in ['AvgPool', 'AvgPool2d']:
                bit_search_range[layer] = [str(vars_util.INT8_BIT), str(vars_util.FP16_BIT)]
            else:
                bit_search_range[layer] = [str(vars_util.INT4_BIT), str(vars_util.INT8_BIT), str(vars_util.FP16_BIT)]
        bit_search_range = self.sort_dict(self.auto_quant_info.quant_layers, bit_search_range)

        return bit_search_range

    def map_file(self, quant_bit, file_kind):
        """ Map file by quant_bit and file_kind info.

        Args:
            quant_bit (string): which bit, 4 and 8 is valid.
            file_kind (string): which kind, config_file and record_file is valid.

        Returns:
            string: the file to find.
        """
        mapping = {
            '4': {
                'config_file': self.temp_info.int4_config_file,
                'record_file': self.temp_info.int4_record_file
            },
            '8': {
                'config_file': self.temp_info.int8_config_file,
                'record_file': self.temp_info.int8_record_file
            }
        }
        return mapping.get(str(quant_bit)).get(file_kind)

    def analyze_sensitivity(self):
        """ Analyze acc sensitivity for each layer in different quant bit.

        This is the first head function for analyzing acc sensitivity. If there'are more methods, please add from here.
        """
        mappding_func = {
            'LAYER_WISE': 'layer_sensitivity_analysis',
        }
        strategy = 'LAYER_WISE'
        getattr(self, mappding_func.get(strategy))()

    def _make_temp_dir(self):
        """ Make temp folder.

        Returns:
            string: path of temp folder
        """
        time_stamp = datetime.now(tz=timezone(offset=timedelta(hours=8))).strftime('%Y%m%d%H%M%S%f')
        if not os.path.isdir(self.auto_quant_info.save_dir):
            files_util.create_path(self.auto_quant_info.save_dir)
        temp_dir = os.path.join(self.auto_quant_info.save_dir, 'temp{}'.format(time_stamp))
        files_util.create_path(temp_dir)
        return temp_dir
