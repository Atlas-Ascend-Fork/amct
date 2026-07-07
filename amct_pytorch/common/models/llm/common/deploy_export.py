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


import torch.nn as nn

from amct_pytorch.quantization.modules.quant_linear import QuantLinear


def generate_quant_group(a_num_bits=8, w_num_bits=8, qtype="float", activation_use_clip=False):
    quant_group = {"input_activations": {"actorder": None, "block_structure": None, "dynamic": True,
                                         "group_size": None, "num_bits": a_num_bits,
                                         "observer": "memoryless", "observer_kwargs": {},
                                         "strategy": "token", "symmetric": True, "type": qtype},
                   "activation_use_clip": activation_use_clip,
                   "output_activations": None,
                   "weights": {"actorder": None, "block_structure": None, "dynamic": False,
                               "group_size": None, "num_bits": w_num_bits,
                               "observer": "minmax", "observer_kwargs": {},
                               "strategy": "channel", "symmetric": True, "type": qtype}}
    return quant_group


def generate_quant_config(cache_scheme, ignores, w4a8=False, w4a4=False, is_mx=False):
    """
    Generate a quantization configuration dictionary based on the specified parameters.
    """
    config_groups = {"group_0": {"targets": ["Linear"]}}
    if is_mx:
        if w4a4:
            config_groups.update({"group_1": {"targets": ["MoEGMMUpGate"]}})
            config_groups.update({"group_2": {"targets": ["MoEGMMDown"]}})
        else:
            config_groups.update({"group_1": {"targets": ["MoEGMM"]}})
    quant_config = {"config_groups": config_groups,
                    "format": "float-quantized" if is_mx else "int-quantized",
                    "global_compression_ratio": 1,
                    "ignore": ignores,
                    "quant_method": "compressed-tensors",
                    "quantization_status": "compressed"}
    quant_config.update(cache_scheme)
    qtype = "float" if is_mx else "int"
    quant_config["config_groups"]["group_0"].update(generate_quant_group(a_num_bits=8, w_num_bits=8, qtype=qtype))
    if is_mx:
        if w4a4:
            quant_config["config_groups"]["group_1"].update(
                generate_quant_group(a_num_bits=4, w_num_bits=4, qtype=qtype))
            quant_config["config_groups"]["group_2"].update(
                generate_quant_group(a_num_bits=8, w_num_bits=4, qtype=qtype))
        else:
            quant_config["config_groups"]["group_1"].update(generate_quant_group(
                a_num_bits=8, w_num_bits=4 if w4a8 else 8, qtype=qtype))
        quant_config["weight_block_size"] = [1, 32]
    return quant_config


def get_quant_ignore_linear_names(block, weight_prefix):
    quant_linear_prefixes = tuple(
        f"{name}."
        for name, module in block.named_modules()
        if isinstance(module, QuantLinear)
    )

    names = []
    for name, module in block.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if any(name.startswith(prefix) for prefix in quant_linear_prefixes):
            continue
        names.append(f"{weight_prefix}{name}")
    return names


def export_block_deploy(pipeline, layer_idx: int, quant_ignore_layers: list):
    block = pipeline.build_quant_block(layer_idx).to(pipeline.args.device).eval()
    pipeline.load_selected_layer_ptq_params(layer_idx, block, strict=False)
    deploy_tensors = {}
    tensor_routes = {}
    weight_prefix = pipeline.get_layer_weight_prefix(layer_idx)
    quant_ignore_layers.extend(get_quant_ignore_linear_names(block, weight_prefix))
    for weight_key, module in pipeline.iter_deploy_bindings(layer_idx, block):
        payload = module.export_deploy()
        deploy_tensors[weight_key] = payload["qweight"]
        tensor_routes[weight_key] = weight_key
        for extra_name, extra_tensor in payload.items():
            if extra_name == "qweight" or extra_tensor is None:
                continue
            extra_key = weight_key.replace(".weight", f".{extra_name}")
            deploy_tensors[extra_key] = extra_tensor
            tensor_routes[extra_key] = weight_key
    return deploy_tensors, tensor_routes

    
