from copy import deepcopy

import torch.nn as nn

from amct_pytorch.common.models.llm.common.ptq_params import PtqParamHandler
from amct_pytorch.algorithms.quant import AlgoBuildContext
from amct_pytorch.quantization.modules.quant_base import (
    ActivationQuantizer,
    WeightQuantizer,
    build_algorithms_by_target,
)
from amct_pytorch.quantization.modules.quant_linear import QuantLinear


class PlainLinear(nn.Module):
    """Wrap a plain nn.Linear so it accepts and ignores a ``structure_transform`` kwarg,
    matching the call signature of QuantLinear at non-quantized attn projections."""

    def __init__(self, linear: nn.Module):
        super().__init__()
        self.linear = linear

    def forward(self, x, structure_transform=None):
        return self.linear(x)


def set_model_act_quant_state(model, flag):
    for name, mod in model.named_modules():
        if isinstance(mod, ActivationQuantizer):
            mod.enable = flag
            mod.is_observe = not flag


def set_model_weight_quant_state(model, flag):
    for name, mod in model.named_modules():
        if isinstance(mod, WeightQuantizer):
            mod.enable = flag

def set_model_to_observe(model, flag):
    for name, mod in model.named_modules():
        if hasattr(mod, "is_observe"):
            mod.is_observe = flag


def build_direct_quant_args(args, bits=8):
    """Build a direct-quant config that skips PTQ-specific algorithms."""
    new_args = deepcopy(args)
    new_args.w_bits = bits
    new_args.a_bits = bits
    new_args.algos = []
    return new_args


def apply_quant_to_moe_mlp(args, model, shared_expert_bits=None, routed_expert_bits=None, cls=None):
    for name, mod in model.named_children():
        if name in ["mlp"] and hasattr(mod, "experts") is False:
            new_args = args
            setattr(model, name, cls(new_args, mod))
            break

        if name in ["experts"]:
            for i in range(len(mod)):
                if mod[i] is None:
                    continue
                new_args = args
                if routed_expert_bits is not None:
                    _new_args = deepcopy(args)
                    _new_args.w_bits = routed_expert_bits
                    new_args = _new_args
                mod[i] = cls(new_args, mod[i])

        if name in ["shared_experts"]:
            new_args = args
            if shared_expert_bits is not None:
                new_args = build_direct_quant_args(args, bits=shared_expert_bits)
            setattr(model, name, cls(new_args, mod))

        if len(list(mod.children())) > 0:
            apply_quant_to_moe_mlp(args, mod, shared_expert_bits=shared_expert_bits, routed_expert_bits=routed_expert_bits, cls=cls)

    return model


def apply_quant_to_attn(args, model, cls):
    # skip_initialization()
    for name, mod in model.named_children():
        if name in ["self_attn", "linear_attn"]:
            setattr(model, name, cls(args, mod))
        if len(list(mod.children())) > 0:
            apply_quant_to_attn(args, mod, cls)
    return model


class QuantGatedMLP(nn.Module):
    """Reusable gated-MLP quant wrapper for fan-out/fan-in feed-forward blocks."""

    def __init__(self, quant_args, mlp_module):
        super().__init__()
        self.quant_args = quant_args
        self.hidden_size = mlp_module.hidden_size
        self.intermediate_size = mlp_module.intermediate_size
        self.act_fn = mlp_module.act_fn
        self.input_transform = None
        self.hidden_transform = None
        self._init_structure_transforms()
        self.up_proj = QuantLinear(quant_args, mlp_module.up_proj, name="up_proj")
        self.gate_proj = QuantLinear(quant_args, mlp_module.gate_proj, name="gate_proj")
        self.down_proj = QuantLinear(quant_args, mlp_module.down_proj, name="down_proj")

        self.input_quant = ActivationQuantizer(quant_args, self.quant_args.a_bits)
        self.hidden_quant = ActivationQuantizer(quant_args, self.quant_args.a_bits)

    def _init_structure_transforms(self):
        ctx = AlgoBuildContext(matrix_size=128, dim_size=self.hidden_size)
        self.input_transform = build_algorithms_by_target(self.quant_args, "structure", ctx)
        ctx = AlgoBuildContext(matrix_size=128, dim_size=self.intermediate_size)
        self.hidden_transform = build_algorithms_by_target(self.quant_args, "structure", ctx)

    def forward(self, input_states):
        if self.input_transform is not None:
            input_states = self.input_transform(input_states)
        x_q = self.input_quant(input_states)
        up = self.up_proj(x_q, structure_transform=self.input_transform)
        gate = self.gate_proj(x_q, structure_transform=self.input_transform)

        hidden = self.act_fn(gate) * up
        if self.hidden_transform is not None:
            hidden = self.hidden_transform(hidden)

        hidden_q = self.hidden_quant(hidden)
        down_states = self.down_proj(hidden_q, structure_transform=self.hidden_transform)
        return down_states

    def export_ptq_params(self):
        params = PtqParamHandler.export_module(self)
        if params:
            return params
        return PtqParamHandler.export_trainable_module(self)

    def load_ptq_params(self, params):
        if isinstance(params, dict) and params and all(isinstance(v, dict) for v in params.values()):
            PtqParamHandler.load_module(self, params)
            return
        PtqParamHandler.load_trainable_module(self, params)
