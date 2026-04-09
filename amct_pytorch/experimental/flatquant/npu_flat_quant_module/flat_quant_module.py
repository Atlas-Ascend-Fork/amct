from typing import Callable
import torch
import torch.nn.functional as F

from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, eager_attention_forward, Qwen3RMSNorm
from transformers.activations import ACT2FN
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from .quant_utils import AscendW4A4FlatQuantDynamicLinearMethod, pack_int4_weights, dynamic_w4a4


class NpuFlatQuantAttention(torch.nn.Module):
    def __init__(self, module_config, layer_idx, quant_config):
        '''
        construct init npu attention for flatquant
        config: module config
        layer_idx: layer index
        '''
        super().__init__()
        self.config = module_config
        self.layer_idx = layer_idx
        self.head_dim = getattr(self.config, "head_dim", self.config.hidden_size // self.config.num_attention_heads)
        self.num_key_value_groups = self.config.num_attention_heads // self.config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = self.config.attention_dropout
        self.is_causal = True
        self.quant_config = quant_config

        if type(self.config).__name__ == 'Qwen3Config':
            # qwen3 use q_norm/k_norm and sliding_window
            self.q_norm = Qwen3RMSNorm(self.head_dim, eps=self.config.rms_norm_eps)
            self.k_norm = Qwen3RMSNorm(self.head_dim, eps=self.config.rms_norm_eps)
            self.sliding_window = self.config.sliding_window if hasattr(self.config, 'sliding_window') else None
            if not (
                self.config.use_sliding_window
                and getattr(self.config, "sliding_window", None) is not None
                and self.layer_idx >= self.config.max_window_layers
            ):
                self.sliding_window = None

        # kv-cache flatquant params, not support for now

        # qkvo proj flatquant params
        # weight scale
        self.register_parameter('scale_q_proj', torch.nn.Parameter(requires_grad=False))
        self.register_parameter('scale_k_proj', torch.nn.Parameter(requires_grad=False))
        self.register_parameter('scale_v_proj', torch.nn.Parameter(requires_grad=False))

        # packed weight
        self.register_parameter('wt_q_packed', torch.nn.Parameter(requires_grad=False))
        self.register_parameter('wt_k_packed', torch.nn.Parameter(requires_grad=False))
        self.register_parameter('wt_v_packed', torch.nn.Parameter(requires_grad=False))

        # trans matrix
        self.register_parameter('left_trans', torch.nn.Parameter(requires_grad=False))
        self.register_parameter('right_trans', torch.nn.Parameter(requires_grad=False))

        if self.quant_config.use_o_quant:
            self.register_parameter('scale_o_proj', torch.nn.Parameter(requires_grad=False))
            self.register_parameter('wt_o_packed', torch.nn.Parameter(requires_grad=False))
            self.register_parameter('o_trans', torch.nn.Parameter(requires_grad=False))
        else:
            self.o_proj = torch.nn.Linear(
                self.config.hidden_size, self.config.hidden_size, bias=self.config.attention_bias
                )
        self.method = AscendW4A4FlatQuantDynamicLinearMethod()

    @classmethod
    def from_quant_module(cls, quant_module) -> 'NpuFlatQuantAttention':
        '''
        construct npu attention for flatquant from quantified calibration module
        '''
        if type(quant_module).__name__ != 'FlatQuantAttention':
            raise ValueError(f'not support construct "NpuFlatQuantAttention" from module {type(quant_module).__name__}')

        flat_attn = cls(quant_module.module_config, quant_module.layer_idx, quant_module.flat_config)

        if hasattr(quant_module, 'sliding_window'):
            flat_attn.sliding_window = quant_module.sliding_window
        flat_attn.q_norm = quant_module.q_norm if hasattr(quant_module, 'q_norm') else None
        flat_attn.k_norm = quant_module.k_norm if hasattr(quant_module, 'k_norm') else None

        quant_module.q_proj.weight_quantizer.find_params(quant_module.q_proj.linear.weight)
        quant_module.k_proj.weight_quantizer.find_params(quant_module.k_proj.linear.weight)
        quant_module.v_proj.weight_quantizer.find_params(quant_module.v_proj.linear.weight)

        wt_q_proj, scale_q_proj = quant_module.q_proj.weight_quantizer.quantize(
            quant_module.q_proj.linear.weight, quantonly=True)
        wt_k_proj, scale_k_proj = quant_module.k_proj.weight_quantizer.quantize(
            quant_module.k_proj.linear.weight, quantonly=True)
        wt_v_proj, scale_v_proj = quant_module.v_proj.weight_quantizer.quantize(
            quant_module.v_proj.linear.weight, quantonly=True)

        flat_attn.scale_q_proj = torch.nn.Parameter(scale_q_proj, requires_grad=False)
        flat_attn.scale_k_proj = torch.nn.Parameter(scale_k_proj, requires_grad=False)
        flat_attn.scale_v_proj = torch.nn.Parameter(scale_v_proj, requires_grad=False)
        flat_attn.wt_q_packed = torch.nn.Parameter(pack_int4_weights(wt_q_proj.data), requires_grad=False)
        flat_attn.wt_k_packed = torch.nn.Parameter(pack_int4_weights(wt_k_proj.data), requires_grad=False)
        flat_attn.wt_v_packed = torch.nn.Parameter(pack_int4_weights(wt_v_proj.data), requires_grad=False)
        flat_attn.q_lac = torch.nn.Parameter(quant_module.q_proj.lac_ratio, requires_grad=False)
        flat_attn.k_lac = torch.nn.Parameter(quant_module.k_proj.lac_ratio, requires_grad=False)
        flat_attn.v_lac = torch.nn.Parameter(quant_module.v_proj.lac_ratio, requires_grad=False)
        flat_attn.left_trans = torch.nn.Parameter(quant_module.ln_trans.matrix_left.T, requires_grad=False)
        flat_attn.right_trans = torch.nn.Parameter(quant_module.ln_trans.matrix_right, requires_grad=False)

        if quant_module.flat_config.use_o_quant:
            quant_module.o_proj.weight_quantizer.find_params(quant_module.o_proj.linear.weight)
            wt_o_proj, scale_o_proj = quant_module.o_proj.weight_quantizer.quantize(
                quant_module.o_proj.linear.weight, quantonly=True)
            flat_attn.scale_o_proj = torch.nn.Parameter(scale_o_proj, requires_grad=False)
            flat_attn.wt_o_packed = torch.nn.Parameter(pack_int4_weights(wt_o_proj.data), requires_grad=False)
            flat_attn.o_trans = torch.nn.Parameter(quant_module.o_trans.get_matrix().T, requires_grad=False)
        else:
            flat_attn.o_proj = quant_module.o_proj.linear

        return flat_attn

    def forward(self, hidden_states, **kwargs):
        '''
        hidden_states: torch.Tensor
        kwargs:
            position_embeddings: tuple[torch.Tensor, torch.Tensor]
            attention_mask: Optional[torch.Tensor]
            past_key_values/past_key_value: Optional, past_key_value is deprecated in transformers 4.58
            cache_position: Optional
            other from TransformerKwargs/FlashAttentionKwargs
        '''
        bsz, q_len, _ = hidden_states.size()

        # NPU operator torch_npu.npu_kronecker_quant requires that total data length (bsz * q_len)
        # is divisible by 16, so we pad q_len if it is not
        # TODO: Remove if this is not required in the future
        q_len_pad_size_after = 0
        if bsz % 16 and q_len % 16:
            q_len_pad_size_after = (q_len // 16 + 1) * 16 - q_len
            padding = (0, 0) * (hidden_states.dim() - 2) + (0, q_len_pad_size_after) + (0, 0)
            hidden_states = F.pad(hidden_states, padding, mode='constant', value=0)

        query_states = self.method.apply(
            self.left_trans, self.right_trans, self.wt_q_packed, self.scale_q_proj, hidden_states, self.q_lac)
        key_states = self.method.apply(
             self.left_trans, self.right_trans, self.wt_k_packed, self.scale_k_proj, hidden_states, self.k_lac)
        value_states = self.method.apply(
             self.left_trans, self.right_trans, self.wt_v_packed, self.scale_v_proj, hidden_states, self.v_lac)
        if q_len_pad_size_after:
            query_states = query_states[:, :q_len]
            key_states = key_states[:, :q_len]
            value_states = value_states[:, :q_len]
        return self.forward_qkv(
            query_states, key_states, value_states, bsz, q_len, **kwargs
        )

    def forward_qkv(self, query_states, key_states, value_states, bsz, q_len, **kwargs):
        hidden_shape = (bsz, q_len, -1, self.head_dim)

        if self.q_norm is None or self.k_norm is None:
            query_states = query_states.view(hidden_shape).transpose(1, 2)
            key_states = key_states.view(hidden_shape).transpose(1, 2)
        else:
            query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
            key_states = self.k_norm(key_states.view(hidden_shape)).transpose(1, 2)
        value_states = value_states.view(hidden_shape).transpose(1, 2)

        cos, sin = kwargs.pop('position_embeddings')
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # kv states are not quantized for now
        past_key_values = None
        if kwargs.get('past_key_value') is not None:
            past_key_values = kwargs.pop('past_key_value')
        if kwargs.get('past_key_values') is not None:
            past_key_values = kwargs.pop('past_key_values')
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, 'cache_position': kwargs.pop('cache_position')}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != 'eager':
            if self.config._attn_implementation == 'sdpa' and kwargs.get('output_attentions', False):
                print(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        if hasattr(self, 'sliding_window'):
            kwargs['sliding_window'] = self.sliding_window
        attention_mask = kwargs.pop('attention_mask')
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        # o_proj quantization
        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        if self.quant_config.use_o_quant:
            init_shape = attn_output.shape
            attn_shape = [-1, self.config.num_attention_heads, self.head_dim]
            attn_output = attn_output.reshape(attn_shape)
            attn_output = torch.matmul(self.o_trans.to(attn_output), attn_output).reshape(init_shape)
            attn_output = dynamic_w4a4(attn_output, self.wt_o_packed, self.scale_o_proj)
        else:
            attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights


class NpuFlatQuantMLP(torch.nn.Module):
    def __init__(self, config, quant_config):
        '''
        construct init npu mlp for flatquant
        '''
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.act_fn = ACT2FN[config.hidden_act]
        self.quant_config = quant_config

        # flat quant activation scale factor
        self.register_parameter('scale_up_proj', torch.nn.Parameter(requires_grad=False))
        self.register_parameter('scale_gate_proj', torch.nn.Parameter(requires_grad=False))
        # flat quant packed weight for npu_kronecker_quant func
        self.register_parameter('wt_up_packed', torch.nn.Parameter(requires_grad=False))
        self.register_parameter('wt_gate_packed', torch.nn.Parameter(requires_grad=False))
        # flat quant trans matrix
        self.register_parameter('ug_left_trans', torch.nn.Parameter(requires_grad=False))
        self.register_parameter('ug_right_trans', torch.nn.Parameter(requires_grad=False))

        if self.quant_config.use_down_quant:
            self.register_parameter('scale_down_proj', torch.nn.Parameter(requires_grad=False))
            self.register_parameter('wt_down_packed', torch.nn.Parameter(requires_grad=False))
            self.register_parameter('down_left_trans', torch.nn.Parameter(requires_grad=False))
            self.register_parameter('down_right_trans', torch.nn.Parameter(requires_grad=False))
        else:
            self.down_proj = torch.nn.Linear(self.hidden_size, self.intermediate_size, bias=False)

        # flat quant mm func
        self.method = AscendW4A4FlatQuantDynamicLinearMethod()

    @classmethod
    def from_quant_module(cls, quant_module) -> 'NpuFlatQuantMLP':
        '''
        construct npu mlp for flatquant from quantified calibration module
        '''
        if type(quant_module).__name__ != 'FlatQuantMLP':
            raise ValueError(f'not support construct "NpuFlatQuantMLP" from module {type(quant_module).__name__}')

        flat_mlp = cls(quant_module.module_config, quant_module.flat_config)
        flat_mlp.act_fn = quant_module.act_fn

        quant_module.up_proj.weight_quantizer.find_params(quant_module.up_proj.linear.weight)
        quant_module.gate_proj.weight_quantizer.find_params(quant_module.gate_proj.linear.weight)

        wt_up_proj, scale_up_proj = quant_module.up_proj.weight_quantizer.quantize(
            quant_module.up_proj.linear.weight, quantonly=True)
        flat_mlp.wt_up_packed = torch.nn.Parameter(pack_int4_weights(wt_up_proj.data), requires_grad=False)
        flat_mlp.scale_up_proj = torch.nn.Parameter(scale_up_proj, requires_grad=False)

        wt_gate_proj, scale_gate_proj = quant_module.gate_proj.weight_quantizer.quantize(
            quant_module.gate_proj.linear.weight, quantonly=True)
        flat_mlp.wt_gate_packed = torch.nn.Parameter(pack_int4_weights(wt_gate_proj.data), requires_grad=False)
        flat_mlp.scale_gate_proj = torch.nn.Parameter(scale_gate_proj, requires_grad=False)
        flat_mlp.ug_left_trans = torch.nn.Parameter(quant_module.up_gate_trans.matrix_left.T, requires_grad=False)
        flat_mlp.ug_right_trans = torch.nn.Parameter(quant_module.up_gate_trans.matrix_right, requires_grad=False)
        flat_mlp.up_lac = torch.nn.Parameter(quant_module.up_proj.lac_ratio, requires_grad=False)
        flat_mlp.gate_lac = torch.nn.Parameter(quant_module.gate_proj.lac_ratio, requires_grad=False)

        if flat_mlp.quant_config.use_down_quant:
            quant_module.down_proj.weight_quantizer.find_params(quant_module.down_proj.linear.weight)
            wt_down_proj, scale_down_proj = quant_module.down_proj.weight_quantizer.quantize(
                quant_module.down_proj.linear.weight, quantonly=True)
            flat_mlp.wt_down_packed = torch.nn.Parameter(pack_int4_weights(wt_down_proj.data), requires_grad=False)
            flat_mlp.scale_down_proj = torch.nn.Parameter(scale_down_proj, requires_grad=False)
            flat_mlp.down_left_trans = torch.nn.Parameter(quant_module.down_trans.matrix_left.T, requires_grad=False)
            flat_mlp.down_right_trans = torch.nn.Parameter(quant_module.down_trans.matrix_right, requires_grad=False)
            flat_mlp.down_lac = torch.nn.Parameter(quant_module.down_proj.lac_ratio, requires_grad=False)
        else:
            flat_mlp.down_proj = quant_module.down_proj.linear

        return flat_mlp

    def forward(self, x):
        # treat indivisible by 16 cases; same as the attention part
        # TODO: remove once allowed
        bsz, q_len, _ = x.size()
        q_len_pad_size_after = 0
        if bsz % 16 and q_len % 16:
            q_len_pad_size_after = (q_len // 16 + 1) * 16 - q_len
            padding = (0, 0) * (x.dim() - 2) + (0, q_len_pad_size_after) + (0, 0)
            x = F.pad(x, padding, mode='constant', value=0)

        up_states = self.method.apply(
            self.ug_left_trans, self.ug_right_trans, self.wt_up_packed, self.scale_up_proj, x, self.up_lac)
        gate_states = self.method.apply(
            self.ug_left_trans, self.ug_right_trans, self.wt_gate_packed, self.scale_gate_proj, x, self.gate_lac)
        x_act_fn = self.act_fn(gate_states) * up_states
        if self.quant_config.use_down_quant:
            down_states = self.method.apply(
                self.down_left_trans, self.down_right_trans, self.wt_down_packed, self.scale_down_proj, x_act_fn,
                self.down_lac.data)
        else:
            down_states = self.down_proj(x_act_fn)

        if q_len_pad_size_after:
            down_states = down_states[:, :q_len]

        return down_states
