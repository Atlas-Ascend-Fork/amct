#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

DecoderAWQuant module and Function

"""
from collections import defaultdict
import functools
import torch
from torch import nn

from amct_pytorch.algo.awq import search_scale, apply_scale, search_clip, apply_clip_by_weight_granularity, \
    get_quant_scale
from amct_pytorch.custom_op.awq.linear_awq import LinearAWQuant
from amct_pytorch.utils.data_utils import check_data_type
from amct_pytorch.utils.model_util import ModuleHelper
from amct_pytorch.utils.vars import MXFP4_E2M1, FLOAT4_E2M1, FLOAT4_E1M2


class DecoderAWQuant(nn.Module):
    """
    Function: Customized torch.nn.Module of the DecoderAWQuant class.
    APIs: forward.
    """
    def __init__(self,
                 ori_module,
                 record_module,
                 layer_name,
                 quant_config):
        """
        Function: init objective.
        Args:
        ori_module: torch module. LlamaDecoderLayer.
        record_module: customed record module. To read and write.
        layer_name: ori_module's name.
        quant_config: calibration algorithm parameters.
        """
        super().__init__()
        # replace linear
        self.named_linears = dict()
        for name, mod in ori_module.named_modules():
            if isinstance(mod, nn.Linear):
                linear_awq = LinearAWQuant(mod, record_module, [layer_name[0] + '.' + name], quant_config, do_awq=False)
                ModuleHelper.replace_module_by_name(ori_module, name, linear_awq)
                self.named_linears[name] = linear_awq

        self.decoder_layer = ori_module
        self.post_quant_record_module = record_module
        self.layer_name = layer_name
        self.quant_config = quant_config


    def forward(self, inputs, **kwargs):
        """
        Function: DecoderAWQuant forward function.
        Args:
        inputs: data used for calibration in torch.tensor.
        """
        def cache_input_hook(module, inputs, outputs, name, feat_dict):
            feat_dict[name].append(inputs[0].detach())

        check_data_type(inputs.dtype, (torch.float16, torch.bfloat16))
        # get linear input data
        input_feat = defaultdict(list)
        hooks = list()
        for name in self.named_linears:
            hooks.append(self.named_linears[name].register_forward_hook(
                functools.partial(cache_input_hook, name=name, feat_dict=input_feat)))
        outputs = self.decoder_layer(inputs, **kwargs)
        for hook in hooks:
            hook.remove()
        input_feat = {name: torch.cat(tensor, dim=0) for name, tensor in input_feat.items()}

        qkv_scale = search_scale(input_feat['self_attn.q_proj'], [self.decoder_layer.self_attn.q_proj,
            self.decoder_layer.self_attn.k_proj, self.decoder_layer.self_attn.v_proj],
            self.decoder_layer.self_attn, self.quant_config, kwargs)

        o_scale = search_scale(input_feat['self_attn.o_proj'], [self.decoder_layer.self_attn.o_proj],
            self.decoder_layer.self_attn.o_proj, self.quant_config)

        gate_up_scale = search_scale(input_feat['mlp.gate_proj'], [self.decoder_layer.mlp.gate_proj,
            self.decoder_layer.mlp.up_proj], self.decoder_layer.mlp, self.quant_config)

        down_scale = search_scale(input_feat['mlp.down_proj'], [self.decoder_layer.mlp.down_proj],
            self.decoder_layer.mlp.down_proj, self.quant_config)

        scale_list = {'self_attn.q_proj': qkv_scale, 'self_attn.k_proj': qkv_scale, 'self_attn.v_proj': qkv_scale,
                      'self_attn.o_proj': o_scale, 'mlp.gate_proj': gate_up_scale, 'mlp.up_proj': gate_up_scale,
                      'mlp.down_proj': down_scale}

        apply_scale(scale_list, self.named_linears, input_feat)

        clip_max_list = search_clip(self.named_linears, input_feat, self.quant_config)

        quant_params = {
            self.layer_name[0] + '.' + name: {
                "awq_quantize":{
                    'scale': 1 / scale_list[name].detach().cpu(),
                    'clip_max': clip_max_list[name].detach().cpu() if name in clip_max_list else None
                }
            } for name in scale_list
        }

        quant_info = {'wts_type': self.quant_config.get("wts_type"),
                      'weight_compress_only': True,
                      'group_size': self.quant_config.get("group_size"), 
                      'round_mode': 'RINT'}

        if self.quant_config.get('wts_type') in (FLOAT4_E2M1, FLOAT4_E1M2):
            quant_params = self.generate_FP4_quant_params(quant_params, clip_max_list, scale_list)
        
        self.post_quant_record_module(list(quant_params.keys()), quant_info, quant_params)
        return outputs

    def generate_FP4_quant_params(self, quant_params, clip_max_list, scale_list):
        cliped_weight = apply_clip_by_weight_granularity(clip_max_list, self.named_linears, \
            self.quant_config.get("weight_granularity"), self.quant_config.get("group_size"))
        scale_group_list, offset_group_list = get_quant_scale(cliped_weight, self.quant_config)
        for name in scale_list:
            quant_factor = {
                "quant_factors": {
                        'scale_w': scale_group_list[name].detach().cpu(),
                }
            }
            if self.quant_config.get("wts_asymmetric"):
                quant_factor["quant_factors"]["offset_w"] = \
                    offset_group_list[name].reshape(-1).detach().cpu()
            quant_params[self.layer_name[0] + '.' + name].update(quant_factor)
        return quant_params
