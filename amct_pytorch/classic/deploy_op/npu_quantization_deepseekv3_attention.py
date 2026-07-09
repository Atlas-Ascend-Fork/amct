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
import torch
from transformers.cache_utils import DynamicCache


class NpuDeepseekV3AttentionQuant(torch.nn.Module):
    """
    Function: Customized torch.nn.Module of the NpuDeepseekV3AttentionQuant class.
    APIs: forward.
    """

    def __init__(self, quant_module):
        """
        Function: init objective.
        Args:
        ori_module: torch module. DeepseekV3Attention.
        layer_name: ori_module's name.
        quant_config: quantization parameters.
        """
        super().__init__()
        self.output_dtype = quant_module.ori_module.q_a_proj.weight.dtype
        self.device = quant_module.ori_module.q_a_proj.weight.device
        self.ori_module = quant_module.ori_module
        self.scale_k = quant_module.scale_k.to(self.device).to(self.output_dtype)
        self.scale_v = quant_module.scale_v.to(self.device).to(self.output_dtype)

    @torch.no_grad()
    def forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask,
        past_key_values,
        **kwargs,
    ):
        """
        Function: NpuDeepseekV3AttentionQuant forward function.
        Args:
        inputs: data used for calibration in torch.tensor.
        """
        if past_key_values is None:
            past_key_values = DynamicCache()
        # Side effect: monkey-patches past_key_values.update to inject KV quantization hook
        self._hook_kv_states(past_key_values)
        fp_out = self.ori_module(
            hidden_states,
            position_embeddings,
            attention_mask,
            past_key_values=past_key_values,
            **kwargs,
        )
        return fp_out

    def _hook_kv_states(self, past_key_values):
        update_function = past_key_values.update
        if hasattr(past_key_values, "is_hooked") and past_key_values.is_hooked:
            return

        def hook_update(key_states, value_states, layer_idx, **kwargs):
            import torch_npu

            device = key_states.device
            quantized_key_states = torch_npu.npu_quantize(
                key_states.npu(),
                self.scale_k.npu(),
                None,
                dtype=torch_npu.hifloat8,
                div_mode=False,
            )
            quantized_value_states = torch_npu.npu_quantize(
                value_states.npu(),
                self.scale_v.npu(),
                None,
                dtype=torch_npu.hifloat8,
                div_mode=False,
            )
            key_states, value_states = update_function(
                quantized_key_states.to(device),
                quantized_value_states.to(device),
                layer_idx,
                **kwargs,
            )
            dequantized_key_states = torch_npu.npu_anti_quant(
                key_states.npu(),
                self.scale_k.to(torch.float32).npu(),
                src_dtype=torch_npu.hifloat8,
                dst_dtype=self.output_dtype,
            )
            dequantized_value_states = torch_npu.npu_anti_quant(
                value_states.npu(),
                self.scale_v.to(torch.float32).npu(),
                src_dtype=torch_npu.hifloat8,
                dst_dtype=self.output_dtype,
            )

            return dequantized_key_states.to(device), dequantized_value_states.to(
                device
            )

        past_key_values.update = hook_update
        past_key_values.is_hooked = True
