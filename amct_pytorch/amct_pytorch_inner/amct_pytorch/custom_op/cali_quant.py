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

from torch import nn
import torch
import numpy as np

from ...amct_pytorch.custom_op.ifmr.ifmr import IFMR
from ...amct_pytorch.custom_op.hfmg.hfmg import HFMG
from ...amct_pytorch.custom_op.dump.dump import DUMP

SCALE_D = 'scale_d'


class CaliQuantBase(nn.Module):
    """
    Function: Customized torch.nn.Module of the calibration operator base class.
    APIs: forward.
    """
    def __init__(self,
                sub_module,
                record_module,
                layers_name,
                num_bits=8,
                batch_num=2,
                with_offset=False,
                dump_config=None,
                mode='cali_dump',
                tensor_balance_factor=None,
                fakequant_precision_mode='DEFAULT'):
        """
        Function: init base function.
        Init some common param both in IFMR and HFMG.

        Args:
        sub_module: torch module. Quantized_type.
        record_module: customed record module. To read and write.
        layers_name: sub_module's name.
        num_bits: controls the cali quant's num bits.
        batch_num: IFMR and HFMG's accumulate data.
        mode: controls the data quant.
        1) cali: ONLY do data calibration.
        2) dump: ONLY do dump the input data.
        3) cali_dump: First dump the input data, secondly do data calibration.
        tensor_balance_factor: the quant result of DMQbalancer op.
        """
        super().__init__()
        self.sub_module = sub_module
        self.record_module = record_module
        self.layers_name = layers_name
        self.num_bits = num_bits
        self.batch_num = batch_num
        self.with_offset = with_offset
        self.dump_config = dump_config
        self.mode = mode
        self.tensor_balance_factor = tensor_balance_factor
        self.fakequant_precision_mode = fakequant_precision_mode

        self.cur_batch = 0

        self._init_dump_mode()
        self._init_cali_param()

        self.scale_d = None

    def forward(self, inputs):
        """
        Function: IFMR / HFMG foward funtion.

        Args:
        inputs: data used for calibration in torch.tensor.
        """
        # step 0. cali / dump / cali_dump
        with torch.no_grad():
            if 'dump' in self.mode and self.dump_config is not None:
                # directly pass the inputs.
                inputs = self.dump_module.forward(inputs)
            if self.tensor_balance_factor is not None:
                device = inputs.device
                input_dtype = inputs.dtype
                inputs = inputs.cpu()
                inputs = inputs / self.tensor_balance_factor
                inputs = inputs.to(dtype=input_dtype).to(device)
            sub_out = self.sub_module(inputs)

            if 'cali' not in self.mode:
                return sub_out

            # step 1. ifmr / hfmg
            self.cur_batch += 1
            if self.cur_batch <= self.cali_algo_param.get('batch_num'):
                self.cali_process(inputs)

            return sub_out

    def cali_process(self, inputs):
        """
        Function: data cali process.

        Args:
        inputs: the data before input into the sub_module.
        """
        # do ifmr / hfmg
        quant_info = self.cali_quant_module.forward(inputs)
        calibration_flag = quant_info.flag
        scale_d = quant_info.scale[0]
        offset_d = quant_info.offset[0]
        self.scale_d = scale_d
        if self.fakequant_precision_mode == 'FORCE_FP16_QUANT':
            self.record_module.fakequant_precision_mode = 'FORCE_FP16_QUANT'
        if calibration_flag:
            # save scale_d and offset_d to record_module
            self.record_module(self.layers_name, self.cali_algo_name,
                               {SCALE_D: scale_d.cpu().tolist(),
                                'offset_d': int(offset_d.cpu().tolist()),
                                'num_bits': self.cali_algo_param['num_bits']})

    def _init_dump_mode(self):
        """
        Function: check mode and init dump_config.
        """
        if self.dump_config is not None:
            if self.dump_config.batch_num is None:
                self.dump_config.batch_num = self.batch_num
            self.dump_module = DUMP(self.layers_name, self.dump_config)
        if self.mode not in ['cali', 'dump', 'cali_dump']:
            raise ValueError("param mode only support ['cali', 'dump', 'cali_dump'], but get {}"
            .format(self.mode))

    def _init_cali_param(self):
        """
        Function: init cali params. IFMR and HFMG's common param.
        """
        self.cali_algo_param = {}
        self.cali_algo_param['layers_name'] = self.layers_name
        self.cali_algo_param['num_bits'] = self.num_bits
        self.cali_algo_param['batch_num'] = self.batch_num
        self.cali_algo_param['with_offset'] = self.with_offset


class CaliQuant(CaliQuantBase):
    """
    Function: Customized torch.nn.Module of the calibration operator. IFMR.
    APIs: forward
    """
    def __init__(self,
                 sub_module,
                 record_module,
                 layers_name,
                 num_bits=8,
                 batch_num=2,
                 with_offset=False,
                 max_percentile=0.999999,
                 min_percentile=0.999999,
                 search_start=0.7,
                 search_end=1.3,
                 search_step=0.01,
                 dump_config=None,
                 mode='cali_dump',
                 tensor_balance_factor=None,
                 fakequant_precision_mode='DEFAULT'):
        """
        Function: IFMR init function.

        Args:
        max_percentile: IFMR param.
        min_percentile: IFMR param.
        search_start: IFMR param.
        search_end: IFMR param.
        search_step: IFMR param.
        """
        super().__init__(
            sub_module=sub_module,
            record_module=record_module,
            layers_name=layers_name,
            num_bits=num_bits,
            batch_num=batch_num,
            with_offset=with_offset,
            dump_config=dump_config,
            mode=mode,
            tensor_balance_factor=tensor_balance_factor,
            fakequant_precision_mode=fakequant_precision_mode,
        )
        # IFMR params init
        self.cali_algo_param['max_percentile'] = max_percentile
        self.cali_algo_param['min_percentile'] = min_percentile
        self.cali_algo_param['search_start'] = search_start
        self.cali_algo_param['search_end'] = search_end
        self.cali_algo_param['search_step'] = search_step
        # IFMR module init.
        self.cali_quant_module = IFMR(**self.cali_algo_param)
        # algo name and search name
        self.cali_algo_name = 'ifmr'


class CaliQuantHfmg(CaliQuantBase):
    """
    Function: Customized torch.nn.Module of the calibration operator. HFMG.
    APIs: forward.
    """
    def __init__(self,
                 sub_module,
                 record_module,
                 layers_name,
                 num_bits=8,
                 batch_num=2,
                 with_offset=False,
                 nbins=4096,
                 dump_config=None,
                 mode='cali_dump',
                 tensor_balance_factor=None,
                 fakequant_precision_mode='DEFAULT'):
        """
        Function: HFMG init function.

        Args:
        nbins: HFMG param.
        """
        super().__init__(
            sub_module=sub_module,
            record_module=record_module,
            layers_name=layers_name,
            num_bits=num_bits,
            batch_num=batch_num,
            with_offset=with_offset,
            dump_config=dump_config,
            mode=mode,
            tensor_balance_factor=tensor_balance_factor,
            fakequant_precision_mode=fakequant_precision_mode,
        )
        # HFMG param init.
        self.cali_algo_param['nbins'] = nbins
        # HFMG module init.
        self.cali_quant_module = HFMG(**self.cali_algo_param)
        # algo name and search name
        self.cali_algo_name = 'hfmg'
