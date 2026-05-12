import sys
import torch
import torch.nn as nn

from transformers import AutoModelForCausalLM
from amct_pytorch.common.models.llm.common.attention_forward import scaled_dot_product_attention
from amct_pytorch.algorithms.quant import AlgoBuildContext
from amct_pytorch.quantization.modules.quant_base import ActivationQuantizer, build_algorithms_by_target
from amct_pytorch.quantization.modules.quant_linear import QuantLinear
from amct_pytorch.quantization.modules.quant_matmul import QuantizedMatmul
from amct_pytorch.quantization.bit_policy import ensure_bit_policy
from amct_pytorch.common.models.llm.common.quant_apply import QuantGatedMLP, PlainLinear


def _ensure_transformers_compat():
    import transformers.utils as transformers_utils
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    from typing import TypedDict

    if not hasattr(transformers_utils, "LossKwargs"):
        class LossKwargs(TypedDict, total=False):
            pass

        transformers_utils.LossKwargs = LossKwargs

    if not hasattr(transformers_utils, "auto_docstring"):
        def auto_docstring(*decorator_args, **decorator_kwargs):
            if decorator_args and callable(decorator_args[0]) and len(decorator_args) == 1 and not decorator_kwargs:
                return decorator_args[0]

            def decorator(obj):
                return obj

            return decorator

        transformers_utils.auto_docstring = auto_docstring

    if not hasattr(transformers_utils, "can_return_tuple"):
        def can_return_tuple(obj):
            return obj

        transformers_utils.can_return_tuple = can_return_tuple

    if "default" not in ROPE_INIT_FUNCTIONS:
        def _compute_default_rope_parameters(config, device=None, seq_len=None, **kwargs):
            head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
            base = getattr(config, "rope_theta", 10000.0)
            inv_freq = 1.0 / (
                base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
            )
            return inv_freq, 1.0

        ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters


def _disable_npu_fused_attention(model):
    module_names = {
        model.__class__.__module__,
        model.model.__class__.__module__,
        type(model.model.layers[0]).__module__,
    }
    for module_name in module_names:
        module = sys.modules.get(module_name)
        if module is not None and hasattr(module, "NPU_ATTN_INFR"):
            module.NPU_ATTN_INFR = False


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class QuantPanguMLP(QuantGatedMLP):
    pass


class QuantPanguAttn(nn.Module):
    def __init__(self, quant_args, attn_module):
        super().__init__()
        self.config = attn_module.config
        self.layer_idx = attn_module.layer_idx
        self.head_dim = attn_module.head_dim
        self.num_heads = attn_module.num_heads
        self.num_key_value_heads = attn_module.num_key_value_heads
        self.num_key_value_groups = attn_module.num_key_value_groups
        self.scaling = attn_module.scaling
        self.attention_dropout = attn_module.attention_dropout
        self.is_causal = attn_module.is_causal
        self.quant_args = quant_args
        ensure_bit_policy(quant_args)
        self.enable_attn_linear = "attn-linear" in self.quant_args.quant_target
        self.hidden_size = attn_module.hidden_size
        self.value_dim = getattr(attn_module, "value_dim", self.num_heads * self.head_dim)

        bits = quant_args.bit_policy["attn-linear"]
        q, k, v, o = bits["q_proj"], bits["k_proj"], bits["v_proj"], bits["o_proj"]
        self.q_proj = QuantLinear(quant_args, attn_module.q_proj, w_bits=q.w, name="q_proj") if self.enable_attn_linear else PlainLinear(attn_module.q_proj)
        self.k_proj = QuantLinear(quant_args, attn_module.k_proj, w_bits=k.w, name="k_proj") if self.enable_attn_linear else PlainLinear(attn_module.k_proj)
        self.v_proj = QuantLinear(quant_args, attn_module.v_proj, w_bits=v.w, name="v_proj") if self.enable_attn_linear else PlainLinear(attn_module.v_proj)
        self.o_proj = QuantLinear(quant_args, attn_module.o_proj, w_bits=o.w, name="o_proj") if self.enable_attn_linear else PlainLinear(attn_module.o_proj)
        self.qk_matmul = QuantizedMatmul(
            self.quant_args,
            l_bits=quant_args.bit_policy.cache_bits("q"),
            r_bits=quant_args.bit_policy.cache_bits("k"),
            left_dim=self.head_dim,
            right_dim=self.head_dim,
        )
        self.pv_matmul = QuantizedMatmul(
            self.quant_args,
            l_bits=quant_args.bit_policy.cache_bits("p"),
            r_bits=quant_args.bit_policy.cache_bits("v"),
            right_dim=self.head_dim,
        )
        self.input_transform = None
        self.out_transform = None
        if self.enable_attn_linear:
            self._init_structure_transforms()
            self.inp_afq = ActivationQuantizer(quant_args, q.a)
            self.o_proj_afq = ActivationQuantizer(quant_args, o.a)
        else:
            self.inp_afq = nn.Identity()
            self.o_proj_afq = nn.Identity()

    def _init_structure_transforms(self):
        ctx = AlgoBuildContext(matrix_size=128, dim_size=self.hidden_size)
        self.input_transform = build_algorithms_by_target(self.quant_args, "structure", ctx)
        ctx = AlgoBuildContext(matrix_size=128, dim_size=self.value_dim)
        self.out_transform = build_algorithms_by_target(self.quant_args, "structure", ctx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_value=None,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        if self.input_transform is not None:
            hidden_states = self.input_transform(hidden_states)
        hidden_states = self.inp_afq(hidden_states)

        query_states = self.q_proj(hidden_states, structure_transform=self.input_transform).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states, structure_transform=self.input_transform).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states, structure_transform=self.input_transform).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attn_output = scaled_dot_product_attention(
            self,
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            is_causal=False,
            dropout_p=0.0,
            scale=self.scaling,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        if self.out_transform is not None:
            attn_output = self.out_transform(attn_output)
        attn_output = self.o_proj_afq(attn_output)
        attn_output = self.o_proj(attn_output, structure_transform=self.out_transform)
        return attn_output, None
