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
from loguru import logger
import torch
import torch.nn as nn
from safetensors.torch import load_file

from amct_pytorch.quantization.modules.quant_linear import QuantLinear
from amct_pytorch.quantization.dtypes.mxfp_impl import weight_dequant


def generate_quant_group(a_bits=8, w_bits=8, qtype="float", activation_use_clip=False):
    observer = "minmax" if qtype == "float" else "memoryless"
    act_strategy = "group" if qtype == "float" else "token"
    w_strategy = "group" if qtype == "float" else "channel"
    group_size = 32 if qtype == "float" else None
    quant_group = {"input_activations": {"actorder": None, "block_structure": None, "dynamic": True,
                                         "group_size": group_size, "num_bits": a_bits,
                                         "observer": observer, "observer_kwargs": {},
                                         "strategy": act_strategy, "symmetric": True, "type": qtype},
                   "activation_use_clip": activation_use_clip,
                   "output_activations": None,
                   "weights": {"actorder": None, "block_structure": None, "dynamic": False,
                               "group_size": group_size, "num_bits": w_bits,
                               "observer": observer, "observer_kwargs": {},
                               "strategy": w_strategy, "symmetric": True, "type": qtype}}
    return quant_group


def _default_bits_scheme():
    return [
        {"targets": ["Linear"], "w_bits": 8, "a_bits": 8},
        {"targets": ["MoEGMM"], "w_bits": 8, "a_bits": 8},
    ]


def generate_quant_config(cache_scheme, ignores, is_mx=False, bits_scheme=None):
    """
    Generate a quantization configuration dictionary based on the specified parameters.
    """
    if bits_scheme is None:
        bits_scheme = _default_bits_scheme()
    qtype = "float" if is_mx else "int"
    config_groups = {}
    for idx, group in enumerate(bits_scheme):
        entry = {"targets": group["targets"]}
        entry.update(generate_quant_group(a_bits=group["a_bits"], w_bits=group["w_bits"], qtype=qtype))
        config_groups[f"group_{idx}"] = entry
    quant_config = {"config_groups": config_groups,
                    "format": "float-quantized" if is_mx else "int-quantized",
                    "global_compression_ratio": 1,
                    "ignore": ignores,
                    "quant_method": "compressed-tensors",
                    "quantization_status": "compressed"}
    quant_config.update(cache_scheme)
    if is_mx:
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


def convert_state_dict(weight, weight_name, scale_inv_name, original_weight_map, model_dir, loaded_files, block_size):
    if weight.element_size() == 1:
        # FP8 weight
        try:
            # Get scale_inv from the correct file
            file_name = original_weight_map[scale_inv_name]
            if file_name not in loaded_files:
                file_path = model_dir / file_name
                loaded_files[file_name] = load_file(file_path, device="cpu")
            scale_inv = loaded_files[file_name][scale_inv_name]
            if weight.dtype == torch.int8:
                weight = weight_dequant(weight, scale_inv, block_size=block_size, is_mx=True, is_packed=True)
            else:
                weight = weight_dequant(weight, scale_inv, block_size=block_size)
        except KeyError:
            logger.warning(
                f"Warning: Missing scale_inv tensor for {weight_name}, skipping conversion")
    return weight


def quant_payload(quant_cls, weight_name, weight, bit):
    quant_obj = quant_cls(bits=int(bit))
    payload = quant_obj.export_deploy(weight)
    tensors = {weight_name: payload["qweight"]}
    for extra_name, extra_tensor in payload.items():
        if extra_name == "qweight" or extra_tensor is None:
            continue
        extra_key = weight_name.replace(".weight", f".{extra_name}")
        tensors[extra_key] = extra_tensor
    return tensors
