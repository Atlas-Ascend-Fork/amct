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
import math
import torch
from torch import nn
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache
from amct_pytorch.quantize_op.base_quant_module import BaseQuantizeModule
from amct_pytorch.classic.quantize_op.utils import calculate_quantile_ema_scale
from amct_pytorch.common.utils.vars import HIFLOAT8
from amct_pytorch.common.utils.log import LOGGER


class LongcatFlashMLAQuant(BaseQuantizeModule):
    """
    Function: Customized torch.nn.Module of the LongcatFlashMLAQuant class.
    APIs: forward.
    """
    def __init__(self,
                ori_module,
                layer_name,
                quant_config):
        """
        Function: init objective.
        Args:
        ori_module: torch module. Linear.
        layer_name: ori_module's name.
        quant_config: quantization parameters.
        """
        super().__init__(ori_module, layer_name, quant_config)
        self.ori_module = ori_module
        self.kvcache_enable = quant_config.get('kvcache_cfg').get('enable_quant', False)
        self.layer_name = layer_name
        if not self.kvcache_enable:
            LOGGER.logd("kvcache is not enable of layer '{}'!".format(self.layer_name), 'LongcatFlashMLAQuant')
        self.quant_type = quant_config.get('kvcache_cfg').get('quant_type')
        self.symmetric = quant_config.get('kvcache_cfg').get('symmetric')
        self.granularity = quant_config.get('kvcache_cfg').get('strategy')
        self.batch_num = quant_config.get('batch_num')
        self.cur_batch = 0
        self.previous_max_k = None
        self.previous_max_v = None
        self.scale_k = None
        self.scale_v = None
        self.offset_k = None
        self.offset_v = None

    @torch.no_grad()
    def forward(self, hidden_states, position_embeddings, attention_mask, past_key_values, **kwargs):
        """
        Function: LongcatFlashMLAQuant forward function.
        Args:
        inputs: data used for calibration in torch.tensor.
        """
        kv_states = self._hook_kv_states(past_key_values)
        fp_out = self.ori_module(hidden_states, position_embeddings, attention_mask,
                                past_key_values=past_key_values, **kwargs)
        
        self.cur_batch += 1
        if self.cur_batch > self.batch_num:
            return fp_out

        batch_max_k = torch.max(torch.abs(kv_states.get('k'))).reshape(-1)
        batch_max_v = torch.max(torch.abs(kv_states.get('v'))).reshape(-1)
        self.previous_max_k = batch_max_k if self.previous_max_k is None else \
            calculate_quantile_ema_scale(self.previous_max_k, batch_max_k)
        self.previous_max_v = batch_max_v if self.previous_max_v is None else \
            calculate_quantile_ema_scale(self.previous_max_v, batch_max_v)
        if self.cur_batch == self.batch_num:
            self.scale_k = (self.previous_max_k / 16.0).to(torch.float32)
            self.scale_v = (self.previous_max_v / 16.0).to(torch.float32)
            LOGGER.logd("Calculate quantile activation quant params of layer '{}' success!".format(self.layer_name),
            'LongcatFlashMLAQuant')
        return fp_out

    def _hook_kv_states(self, past_key_value):
        update_function = past_key_value.update
        capture_data = {}

        def hook_update(key_states, value_states, layer_idx, **kwargs):
            capture_data['k'] = torch.max(torch.abs(key_states)).reshape(-1)
            capture_data['v'] = torch.max(torch.abs(value_states)).reshape(-1)
            LOGGER.logd("Capture data from layer '{}' success!".format(self.layer_name),
            'LongcatFlashMLAQuant')
            return update_function(key_states, value_states, layer_idx, **kwargs)
        
        past_key_value.update = hook_update
        
        return capture_data
