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

from amct_pytorch.quantization.bit_policy import LayerBits
from amct_pytorch.quantization.modules.quant_linear import QuantLinear


# MoE "Down" kernel requires 8-bit activations even in W4A4 mixed-precision;
# only "UpGate" carries the configured 4-bit activations.
_MOE_DOWN_A_BITS = 8


def generate_quant_group(a_num_bits=8, w_num_bits=8, qtype="int", activation_use_clip=False):
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


def generate_quant_config(cache_scheme, ignores, *, is_mx=False,
                          linear_bits: LayerBits = (8, 8),
                          moe_bits: LayerBits | None = None,
                          moe_split=False):
    """
    Build a compressed-tensors quantization_config.

    Per-target bit-widths are explicit so one deploy can mix bit-widths
    (e.g. attn "Linear" at W8A8 + MoE "MoEGMM" at W4A8):
      - linear_bits=(w, a): the "Linear" group (attn / dense; MoE experts also
        fall back to this target when no separate MoE group is emitted).
      - moe_bits=(w, a): emit a separate "MoEGMM" group; None => MoE shares "Linear".
      - moe_split=True: split MoE into "MoEGMMUpGate" (uses moe_bits.a activations)
        and "MoEGMMDown" (always 8-bit activations). Used for mx W4A4 mixed-precision MoE
        where the fused UpGate kernel requires 4-bit activations but Down uses 8-bit.

    Examples:
      MoE-only W4A8        : linear_bits=(4, 8)
      attn W8A8 + MoE W4A8 : linear_bits=(8, 8), moe_bits=(4, 8)
      mx W4A4 split        : is_mx=True, linear_bits=(8, 8), moe_bits=(4, 4), moe_split=True
    """
    qtype = "float" if is_mx else "int"

    def _group(targets, w_bits, a_bits):
        return {"targets": list(targets),
                **generate_quant_group(a_num_bits=a_bits, w_num_bits=w_bits, qtype=qtype)}

    config_groups = {"group_0": _group(["Linear"], *linear_bits)}
    if moe_bits is not None:
        moe_w, moe_a = moe_bits
        if moe_split:
            config_groups["group_1"] = _group(["MoEGMMUpGate"], moe_w, moe_a)
            config_groups["group_2"] = _group(["MoEGMMDown"], moe_w, _MOE_DOWN_A_BITS)
        else:
            config_groups["group_1"] = _group(["MoEGMM"], moe_w, moe_a)

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
