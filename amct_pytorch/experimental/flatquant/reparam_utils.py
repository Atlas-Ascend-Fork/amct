import re

from torch import nn
from .npu_flat_quant_module.flat_quant_module import NpuFlatQuantAttention, NpuFlatQuantMLP
from .flat_quant_module.flat_utils import reparameterize_ln


def get_layer_from_submodule(model: nn.Module, object_name: str):
    object_name_tks = object_name.split(".")
    layer_module = model
    for i in range(len(object_name_tks) - 1):
        layer_module = getattr(layer_module, object_name_tks[i])
    return layer_module


def get_replacement_module(model, object_type_name, object_name, object_module):
    if object_type_name != 'FlatQuantAttention' and object_type_name != 'FlatQuantMLP':
        raise ValueError(f'object_type_name {object_type_name} is invalid')
    if object_type_name == 'FlatQuantAttention':
        matched = re.match(r".*\.layers\..*\.self_attn", object_name)
        if not matched:
            raise ValueError(f"object_name {object_name} not matched with required structure for self_attn")
    else:
        matched = re.match(r".*\.layers\..*\.mlp", object_name)
        if not matched:
            raise ValueError(f"object_name {object_name} not matched with required structure for mlp")

    object_module.reparameterize()
    layer_module = get_layer_from_submodule(model, object_name)

    if object_type_name == 'FlatQuantAttention':
        if object_module.ln_trans is not None and object_module.ln_trans.add_diag:
            reparameterize_ln(layer_module.input_layernorm, object_module.ln_trans)
        npu_module = NpuFlatQuantAttention.from_quant_module(object_module)

    else:
        if object_module.up_gate_trans is not None and object_module.up_gate_trans.add_diag:
            reparameterize_ln(layer_module.post_attention_layernorm, object_module.up_gate_trans)
        npu_module = NpuFlatQuantMLP.from_quant_module(object_module)

    return npu_module
