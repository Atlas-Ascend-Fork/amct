from copy import deepcopy
from inspect import Parameter, signature

import torch.nn as nn

from amct_pytorch.common.models.llm.common.ptq_params import PtqParamHandler
from amct_pytorch.algorithms.quant import AlgoBuildContext
from amct_pytorch.quantization.modules.quant_base import (
    ActivationQuantizer,
    WeightQuantizer,
    build_algorithms_by_target,
)
from amct_pytorch.quantization.modules.quant_linear import QuantLinear
from amct_pytorch.quantization.bit_policy import ensure_bit_policy


class PlainLinear(nn.Module):
    """
    Wrap a plain nn.Linear so it accepts and ignores a ``structure_transform`` kwarg,
    matching the call signature of QuantLinear at non-quantized attn projections.
    """

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


def build_no_algo_args(args):
    """Return a shallow-copied args with ``algos`` cleared.

    Historically used for the MoE shared-expert path, which is wrapped with the
    same gated-MLP class but skips PTQ-specific algorithms. Bit-widths come
    from ``args.bit_policy`` based on the role passed at construction time.
    """
    new_args = deepcopy(args)
    new_args.algos = []
    return new_args


def build_direct_quant_args(args, bits):
    """Return args for a direct quantized path that bypasses PTQ algorithms."""
    new_args = build_no_algo_args(args)
    new_args.w_bits = bits
    new_args.a_bits = bits
    return new_args


def _build_quant_wrapper(cls, args, module, group):
    init_params = signature(cls.__init__).parameters
    accepts_group = (
        "group" in init_params
        or any(param.kind == Parameter.VAR_KEYWORD for param in init_params.values())
    )
    if accepts_group:
        return cls(args, module, group=group)
    return cls(args, module)


def apply_quant_to_moe_mlp(args, model, cls=None):
    """Wrap dense MLPs and per-expert MoE blocks with their quant counterparts.

    The ``group`` is assigned by location so the bit policy can target each path:
      - top-level ``mlp`` (no ``experts`` attr) -> ``mlp``
      - per-expert wrappers under ``experts``    -> ``moe.routed``
      - shared experts                           -> ``moe.shared`` (skips algos
        via :func:`build_no_algo_args`, matching prior behavior)
    """
    for name, mod in model.named_children():
        if name in ["mlp"] and hasattr(mod, "experts") is False:
            setattr(model, name, _build_quant_wrapper(cls, args, mod, group="mlp"))
            break

        if name in ["experts"]:
            for idx, cur_mod in enumerate(mod):
                if cur_mod is None:
                    continue
                mod[idx] = _build_quant_wrapper(cls, args, cur_mod, group="moe.routed")

        if name in ["shared_experts"]:
            shared_args = build_no_algo_args(args)
            setattr(model, name, _build_quant_wrapper(cls, shared_args, mod, group="moe.shared"))

        if len(list(mod.children())) > 0:
            apply_quant_to_moe_mlp(args, mod, cls=cls)

    return model


def apply_quant_to_attn(args, model, cls):
    for name, mod in model.named_children():
        if name in ["self_attn", "linear_attn"]:
            setattr(model, name, cls(args, mod))
        if len(list(mod.children())) > 0:
            apply_quant_to_attn(args, mod, cls)
    return model


class QuantGatedMLP(nn.Module):
    """Reusable gated-MLP quant wrapper for fan-out/fan-in feed-forward blocks.

    ``group`` selects which yaml block applies:
      - ``"mlp"``        — dense MLP (default)
      - ``"moe.routed"`` — per-expert wrapper inside a routed MoE
      - ``"moe.shared"`` — shared-expert wrapper

    Per-projection bits come from yaml entries keyed by ``gate_proj`` /
    ``up_proj`` / ``down_proj`` under that group. Activation quantizers reuse
    the ``a_bits`` of the linear they feed.
    """

    def __init__(self, quant_args, mlp_module, group: str = "mlp"):
        super().__init__()
        self.quant_args = quant_args
        ensure_bit_policy(quant_args)
        self.group = group
        self.hidden_size = mlp_module.hidden_size
        self.intermediate_size = mlp_module.intermediate_size
        self.act_fn = mlp_module.act_fn
        self.input_transform = None
        self.hidden_transform = None
        self._init_structure_transforms()

        bits = quant_args.bit_policy[group]
        gate, up, down = bits["gate_proj"], bits["up_proj"], bits["down_proj"]
        self.gate_proj = QuantLinear(quant_args, mlp_module.gate_proj, w_bits=gate.w, name="gate_proj")
        self.up_proj = QuantLinear(quant_args, mlp_module.up_proj, w_bits=up.w, name="up_proj")
        self.down_proj = QuantLinear(quant_args, mlp_module.down_proj, w_bits=down.w, name="down_proj")
        self.input_quant = ActivationQuantizer(quant_args, gate.a)
        self.hidden_quant = ActivationQuantizer(quant_args, down.a)

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

    def _init_structure_transforms(self):
        ctx = AlgoBuildContext(matrix_size=128, dim_size=self.hidden_size)
        self.input_transform = build_algorithms_by_target(self.quant_args, "structure", ctx)
        ctx = AlgoBuildContext(matrix_size=128, dim_size=self.intermediate_size)
        self.hidden_transform = build_algorithms_by_target(self.quant_args, "structure", ctx)
