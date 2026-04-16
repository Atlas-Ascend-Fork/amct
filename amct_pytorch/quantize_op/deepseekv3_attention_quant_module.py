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
import transfermer
from amct_pytorch.utils.vars import HIFLOAT8
from amct_pytorch.utils.log import LOGGER
from amct_pytorch.quantize_op.utils import calculate_quantile_ema_scale, get_weight_min_max_by_granularity


class DeepseekV3AttentionQuant(BaseQuantizeModule):
    """
    Function: Customized torch.nn.Module of the GPTQuant class.
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
        self.config = ori_module.config
        self.layer_idx = ori_module.layer_idx
        self.num_key_value_groups = ori_module.num_key_value_groups
        self.attention_dropout = ori_module.attention_dropout
        self.num_heads = ori_module.num_heads

        self.q_lora_rank = ori_module.q_lora_rank
        self.qk_rope_head_dim = ori_module.qk_rope_head_dim
        self.kv_lora_rank = ori_module.kv_lora_rank
        self.v_head_dim = ori_module.v_head_dim
        self.qk_nope_head_dim = ori_module.qk_nope_head_dim
        self.qk_head_dim = ori_module.qk_head_dim

        self.is_causal = ori_module.is_causal

        self.q_proj = ori_module.q_proj if ori_module.q_proj else None
        self.q_a_proj = ori_module.q_a_proj if ori_module.q_a_proj else None
        self.q_a_layernorm = ori_module.q_a_layernorm if ori_module.q_a_layernorm else None
        self.q_b_proj = ori_module.q_b_proj if ori_module.q_b_proj else None

        self.kv_a_proj_with_mqa = ori_module.kv_a_proj_with_mqa
        self.kv_a_layernorm = ori_module.kv_a_layernorm
        self.kv_b_proj = ori_module.kv_b_proj
        
        self.o_proj = ori_module.o_proj

        self.scaling = ori_module.scaling

        self.kvcache_enable = quant_config.get('kvcache_cfg', {}).get('enable_quant', False)
        if self.kvcache_enable:
            self.kvcache_type = quant_config.get('kvcache_cfg', {}).get('quant_type')
            self.kvcache_symmetric = quant_config.get('kvcache_cfg', {}).get('symmetric', True)
            self.kvcache_granularity = quant_config.get('kvcache_cfg', {}).get('strategy', 'tensor')
            self.kvcache_batch_num = quant_config.get('kvcache_cfg', {}).get('batch_num', 32)
            self.kvcache_cur_batch = 0
            self.kvcache_previous_max_k = None
            self.kvcache_previous_max_v = None
            self.scale_k_cache = None
            self.scale_v_cache = None
            self.offset_k_cache = None
            self.offset_v_cache = None

    @torch.no_grad()
    def forward(
            self,
            hidden_states,
            position_embeddings,
            attention_mask,
            past_key_values,
            cu_seq_lens_q,
            cu_seq_lens_k,
            max_length_q,
            max_length_k,
        ):
        batch_size, seq_length = hidden_states.shape[:-1]
        query_shape = (batch_size, seq_length, -1, self.qk_head_dim)
        key_shape = (batch_size, seq_length, -1, self.qk_nope_head_dim + self.v_head_dim)

        if self.q_lora_rank is None:
            q_states = self.q_proj(hidden_states)
        else:
            q_states = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q_states = q_states.view(query_shape).transpose(1, 2)
        q_pass, q_rot = torch.split(q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        k_pass, k_rot = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)

        k_pass = self.kv_b_proj(self.kv_a_layernorm(k_pass)).view(key_shape).transpose(1, 2)
        k_pass, value_states = torch.split(k_pass, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        k_rot = k_rot.view(batch_size, 1, seq_length, self.qk_rope_head_dim)

        cos, sin = position_embeddings
        if self.config.rope_interleave:  # support using interleaved weights for efficiency
            q_rot, k_rot = apply_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin)
        else:
            q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot, cos, sin)
        k_rot = k_rot.expand(*k_pass.shape[:-1], -1)

        query_states = torch.cat((q_pass, q_rot), dim=-1)
        key_states = torch.cat((k_pass, k_rot), dim=-1)

        if past_key_values is not None:
            if self.kvcache_enable and self.kvcache_cur_batch < self.kvcache_batch_num:
                quantized_key, quantized_value = self._quantize_kv_cache(key_states, value_states)
                self.kvcache_cur_batch += 1
            else:
                quantized_key = key_states
                quantized_value = value_states
            key_states, value_states = past_key_values.update(quantized_key, quantized_value, self.layer_idx)

        if is_flash_attention_requested(self.config) and self.qk_head_dim != self.v_head_dim:
            value_states = F.pad(value_states, [0, self.qk_head_dim - self.v_head_dim])

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        if is_flash_attention_requested(self.config) and self.qk_head_dim != self.v_head_dim:
            attn_output = attn_output[:, :, :, : self.v_head_dim]

        attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    def _quantize_kv_cache(self, key_states, value_states):
        """
        Function: quantize key and value states for KV cache using quantile algorithm
        Args:
            key_states: torch.Tensor, key states to quantize
            value_states: torch.Tensor, value states to quantize
        Returns:
            quantized_key: quantized key states
            quantized_value: quantized value states
        """
        batch_max_k = self._compute_kv_batch_max(key_states)
        batch_max_v = self._compute_kv_batch_max(value_states)

        self._update_kv_scale_tensor(batch_max_k, batch_max_v)

        if self.kvcache_cur_batch == self.kvcache_batch_num - 1:
            self._finalize_kv_calibration()

        quantized_key = self._apply_kv_quantization(key_states, self.scale_k_cache, self.offset_k_cache)
        quantized_value = self._apply_kv_quantization(value_states, self.scale_v_cache, self.offset_v_cache)

        return quantized_key, quantized_value

    def _compute_kv_batch_max(self, tensor):
        """
        Function: compute batch max for KV cache quantization
        Args:
            tensor: input tensor
        Returns:
            batch_max: max value of tensor
        """
        if self.kvcache_granularity == 'tensor':
            return torch.max(torch.abs(tensor)).reshape(-1)
        elif self.kvcache_granularity == 'token':
            cout_axis = -2
            axis_list = list(range(0, tensor.dim()))
            axis_list.pop(cout_axis)
            axis = tuple(axis_list)
            return torch.amax(torch.abs(tensor), dim=axis, keepdim=True)
        return None

    def _update_kv_scale_tensor(self, batch_max_k, batch_max_v):
        """
        Function: update KV cache scale tensor using EMA
        Args:
            batch_max_k: current batch max for key
            batch_max_v: current batch max for value
        """
        if self.kvcache_previous_max_k is None:
            self.kvcache_previous_max_k = batch_max_k
        else:
            self.kvcache_previous_max_k = calculate_quantile_ema_scale(self.kvcache_previous_max_k, batch_max_k)

        if self.kvcache_previous_max_v is None:
            self.kvcache_previous_max_v = batch_max_v
        else:
            self.kvcache_previous_max_v = calculate_quantile_ema_scale(self.kvcache_previous_max_v, batch_max_v)

    def _finalize_kv_calibration(self):
        """
        Function: finalize KV cache calibration by computing final scale factors
        """
        self.scale_k_cache = self._calculate_hif8_scale(self.kvcache_previous_max_k)
        self.scale_v_cache = self._calculate_hif8_scale(self.kvcache_previous_max_v)
        self.offset_k_cache = None
        self.offset_v_cache = None
        LOGGER.logd("Calculate quantile KV cache quant params of layer '{}' success!".format(self.layer_name),
            'DeepseekV3AttentionQuant')

    def _calculate_hif8_scale(self, tensor_max):
        """
        Function: calculate HIF8 scale by scaling tensor max to 16
        Args:
            tensor_max: max value of tensor
        Returns:
            scale: HIF8 scale factor
        """
        if tensor_max is not None:
            return (tensor_max / 16.0).to(torch.float32)
        else:
            return tensor_max

    def _apply_kv_quantization(self, tensor, scale, offset):
        """
        Function: apply quantization to KV cache tensor
        Args:
            tensor: input tensor
            scale: scale factor
            offset: offset factor (None for symmetric quantization)
        Returns:
            quantized_tensor: quantized and dequantized tensor
        """
        if scale is None:
            return tensor

        import torch_npu
        quantized_tensor = torch_npu.npu_quantize(tensor.npu(), scale.npu(), None, dtype=torch_npu.hifloat8)
        dequantized_tensor = torch_npu.npu_dtype_cast(quantized_tensor, tensor.dtype, input_dtype=torch_npu.hifloat8)
        return dequantized_tensor * scale