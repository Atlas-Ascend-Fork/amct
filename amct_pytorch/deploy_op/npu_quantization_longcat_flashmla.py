# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
from amct_pytorch.utils.vars import HIFLOAT8
from amct_pytorch.utils.log import LOGGER
from amct_pytorch.quantize_op.utils import calculate_quantile_ema_scale

import torch_npu


class NpuLongcatFlashMLA(torch.nn.Module):
    """
    Function: Customized torch.nn.Module of the GPTQuant class.
    APIs: forward.
    """
    def __init__(self, quant_module):
        """
        Function: init objective.
        Args:
        ori_module: torch module. Linear.
        layer_name: ori_module's name.
        quant_config: quantization parameters.
        """
        super().__init__()
        self.output_dtype = quant_module.q_proj.dtype
        self.device = quant_module.weight.device
        self.scale_k = quant_module.scale_k
        self.scale_v = quant_module.scale_v


    @torch.no_grad()
    def forward(self, hidden_states, position_embeddings, attention_mask, past_key_values, **kwargs):
        """
        Function: NpuDeepseekV3AttentionQuant forward function.
        Args:
        inputs: data used for calibration in torch.tensor.
        """
        if past_key_values is None:
            past_key_values = DynamicCache()
        kv_states = self._hook_kv_states(past_key_values)
        fp_out = self.ori_module(hidden_states, position_embeddings, attention_mask, past_key_values=past_key_values, **kwargs)
        return fp_out

    def _hook_kv_states(self, past_key_values):
        update_function = past_key_values.update
        def hook_update(key_states, value_states, layer_idx, **kwargs):
            quantized_key_states = torch_npu.npu_quantize(key_states, self.scale_k, dtype=HIFLOAT8, div_mode=False)
            quantized_value_states = torch_npu.npu_quantize(value_states, self.scale_v, dtype=HIFLOAT8, div_mode=False)
            key_states, value_states = update_function(quantized_key_states, quantized_value_states, layer_idx, **kwargs)
            dequantized_key_states = torch_npu.npu_anti_quantize(key_states, self.scale_k, dtype=HIFLOAT8, div_mode=False)
            dequantized_value_states = torch_npu.npu_anti_quantize(value_states, self.scale_v, dtype=HIFLOAT8, div_mode=False)
            
            return dequantized_key_states, dequantized_value_states
        
        past_key_values.update = hook_update



