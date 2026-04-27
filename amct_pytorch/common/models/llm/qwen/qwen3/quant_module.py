import torch
import torch.nn as nn

from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3MLP,
    apply_rotary_pos_emb as apply_rotary_pos_emb_qwen3,
)

from amct_pytorch.common.models.llm.common.attention_forward import scaled_dot_product_attention
from amct_pytorch.common.models.llm.common.quant_apply import QuantGatedMLP, PlainLinear
from amct_pytorch.algorithms.quant import AlgoBuildContext
from amct_pytorch.quantization.modules.quant_base import ActivationQuantizer, build_algorithms_by_target
from amct_pytorch.quantization.modules.quant_linear import QuantLinear
from amct_pytorch.quantization.modules.quant_matmul import QuantizedMatmul


class QuantQwen3MLP(QuantGatedMLP):
    def __init__(self, quant_args, mlp_module):
        super().__init__(quant_args, mlp_module)


class QuantQwen3Attn(Qwen3Attention):
    def __init__(self, quant_args, attn_module):
        super().__init__(attn_module.config, attn_module.layer_idx)
        self.quant_args = quant_args
        self.enable_attn_linear = "attn-linear" in self.quant_args.quant_target
        self.hidden_size = getattr(attn_module, "hidden_size", attn_module.config.hidden_size)
        self.q_proj = QuantLinear(quant_args, attn_module.q_proj, name="q_proj") if self.enable_attn_linear else PlainLinear(attn_module.q_proj)
        self.k_proj = QuantLinear(quant_args, attn_module.k_proj, name="k_proj") if self.enable_attn_linear else PlainLinear(attn_module.k_proj)
        self.v_proj = QuantLinear(quant_args, attn_module.v_proj, name="v_proj") if self.enable_attn_linear else PlainLinear(attn_module.v_proj)
        self.o_proj = QuantLinear(quant_args, attn_module.o_proj, name="o_proj") if self.enable_attn_linear else PlainLinear(attn_module.o_proj)
        self.q_norm = attn_module.q_norm
        self.k_norm = attn_module.k_norm
        self.qk_matmul = QuantizedMatmul(
            self.quant_args,
            l_bits=quant_args.q_bits,
            r_bits=quant_args.k_bits,
            left_dim=self.head_dim,
            right_dim=self.head_dim,
        )
        self.pv_matmul = QuantizedMatmul(
            self.quant_args,
            l_bits=quant_args.p_bits,
            r_bits=quant_args.v_bits,
            right_dim=self.head_dim,
        )
        self.input_transform = None
        self.out_transform = None
        if self.enable_attn_linear:
            self._init_structure_transforms()
            self.inp_afq = ActivationQuantizer(quant_args, self.quant_args.a_bits)
            self.o_proj_afq = ActivationQuantizer(quant_args, self.quant_args.a_bits)
        else:
            self.inp_afq = nn.Identity()
            self.o_proj_afq = nn.Identity()

    def _init_structure_transforms(self):
        ctx = AlgoBuildContext(matrix_size=128, dim_size=self.hidden_size)
        self.input_transform = build_algorithms_by_target(self.quant_args, "structure", ctx)
        ctx = AlgoBuildContext(matrix_size=128, dim_size=self.hidden_size)
        self.out_transform = build_algorithms_by_target(self.quant_args, "structure", ctx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        if self.input_transform is not None:
            hidden_states = self.input_transform(hidden_states)
        hidden_states = self.inp_afq(hidden_states)

        query_states = self.q_norm(
            self.q_proj(hidden_states, structure_transform=self.input_transform).view(hidden_shape)
        ).transpose(1, 2)
        key_states = self.k_norm(
            self.k_proj(hidden_states, structure_transform=self.input_transform).view(hidden_shape)
        ).transpose(1, 2)
        value_states = self.v_proj(
            hidden_states, structure_transform=self.input_transform
        ).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_qwen3(query_states, key_states, cos, sin)

        attn_output = scaled_dot_product_attention(
            self,
            query_states,
            key_states,
            value_states,
            attn_mask=None,
            is_causal=True,
            dropout_p=0.0,
            scale=self.scaling,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        if self.out_transform is not None:
            attn_output = self.out_transform(attn_output)
        attn_output = self.o_proj_afq(attn_output)
        attn_output = self.o_proj(attn_output, structure_transform=self.out_transform)
        return attn_output, None
