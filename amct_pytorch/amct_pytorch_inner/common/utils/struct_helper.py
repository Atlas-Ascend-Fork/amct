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

from ..auto_calibration.sensitivity_base import SensitivityBase
from ..auto_calibration.sensitivity_base import MseSensitivity
from . import files as files_util
from . import vars_util
from ..config.auto_mix_quant_config import AutoMixedPrecisionConfigHelper


class DumpConfig:
    """ Config info for dump data. """
    def __init__(self, dump_dir, batch_num):
        """ Init func.

        Args:
            dump_dir (string): the path to save dump file.
            batch_num (int): batch to dump data, if -1, will dump every batch.
        """
        self.dump_dir = dump_dir
        self.batch_num = batch_num

        self.check_params()

    def check_params(self):
        """ check params"""
        self.dump_dir = os.path.realpath(self.dump_dir)


class GraphInfoBase:
    """ Config info for graph. """
    def __init__(self, graph):
        """ Init func """
        self.graph = graph

        self.check_params()

    def check_params(self):
        """ check params"""
        pass


class CalibrationConfigInfo:
    """ Config info for Calibration. """
    def __init__(self, config_defination):
        """ Init func """
        self.config_defination = config_defination
        self.batch_num = None
        self.check_params()

    def check_params(self):
        """ check params"""
        if self.config_defination is not None and not os.path.exists(self.config_defination):
            raise RuntimeError("The config_defination {} does not exist, please check the file path.".format(
                self.config_defination))


class AutoSearchMixedPrecisionInfo:
    """ Config info for AutoSearchMixedPrecision. """
    def __init__(self, cfg_file, save_dir, sensitivity):
        """ Init func.

        Args:
            cfg_file (string): file from AutoMixedPrecisionConfig, indicating how to do search.
            save_dir (string): the path where to store model and model's name.
        """
        # set variable by config
        self.cfg_helper = AutoMixedPrecisionConfigHelper(AutoMixedPrecisionConfigHelper.read_file(cfg_file))
        self.save_dir, self.save_prefix = files_util.split_save_path(save_dir)
        self.sensitivity = sensitivity
        self.check_params()

        # set inner variable
        self.quant_mix_bits = [vars_util.FP16_BIT, vars_util.INT8_BIT, vars_util.INT4_BIT]
        self.acc_decay = {}
        self.computes_ops = {}
        self.computes_ops_constraint = 0
        self.bit_config = {}
        self.quant_layers = []
        self.shape_info = {}
        self.qat_search_range = {}
        self.qat_precision_file = files_util.concat_name(self.save_dir, self.save_prefix, 'qat_mixed_precision.json')
        self.qat_cfg_file = files_util.concat_name(self.save_dir, self.save_prefix, 'qat_mixed_precision.cfg')

        files_util.create_file_path(self.save_dir)

    def check_params(self):
        """ check params """
        self.cfg_helper.parse()
        self.compression_ratio = self.cfg_helper.compress_ratio

        if isinstance(self.sensitivity, SensitivityBase):
            pass
        elif self.sensitivity != 'MseSimilarity':
            raise ValueError("{} is not support. Only support ['MseSimilarity']."
                    .format(self.sensitivity))
        else:
            self.sensitivity = MseSensitivity(do_normalization=True)


class AutoSearchMixedPrecisionTempInfo:
    """ Config info for AutoSearchMixedPrecision tmp using. """
    def __init__(self, temp_dir):
        """ Init func.

        Args:
            temp_dir (string): path of temp folder.
        """
        self.temp_dir = temp_dir

        self.dump_dir = os.path.join(self.temp_dir, "dump")
        files_util.create_path(self.dump_dir)

        self.config_file = os.path.join(self.temp_dir, "calibration_config.json")
        self.int8_config_file = os.path.join(self.temp_dir, "int8_calibration_config.json")
        self.int8_record_file = os.path.join(self.temp_dir, "int8_record.txt")
        self.int4_config_file = os.path.join(self.temp_dir, "int4_calibration_config.json")
        self.int4_record_file = os.path.join(self.temp_dir, "int4_record.txt")
        self.loss_info_file = os.path.join(self.temp_dir, "loss_info.txt")
        self.bitops_info_file = os.path.join(self.temp_dir, "bitops_info.txt")
        self.shape_info_file = os.path.join(self.temp_dir, "shape_info.txt")

        self.fused_file = os.path.join(self.temp_dir, "fused.pb")
        self.modified_onnx_file = os.path.join(self.temp_dir, 'modified_model.onnx')
