from typing import Callable
import torch
import torch.nn as nn

from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, eager_attention_forward
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS


from .quant_utils import ActivationQuantizer
from .function_utils import get_init_scale, get_decompose_dim
from .trans_utils import SVDSingleTransMatrix, SVDDecomposeTransMatrix
from .flat_linear import FlatQuantizedLinear
from .config_utils import FlatConfig


def flat_config_parser(layer_config):
    flatquant_config = layer_config.get('algorithm').get('flatquant')

    flat_config = FlatConfig()
    # quant config
    flat_config.a_bits = int(layer_config.get('inputs_cfg').get('quant_type').replace('int', ''))
    flat_config.w_bits = int(layer_config.get('weights_cfg').get('quant_type').replace('int', ''))
    flat_config.a_sym = layer_config.get('inputs_cfg').get('symmetric')
    flat_config.w_sym = layer_config.get('weights_cfg').get('symmetric')
    flat_config.lac = flatquant_config.get('lac', FlatConfig.lac)
    flat_config.lwc = flatquant_config.get('lwc', FlatConfig.lwc)
    flat_config.use_o_quant = flatquant_config.get('use_o_quant', FlatConfig.use_o_quant)
    flat_config.use_down_quant = flatquant_config.get('use_down_quant', FlatConfig.use_down_quant)

    # kv config
    flat_config.use_kcache_quant = flatquant_config.get('use_kcache_quant', FlatConfig.use_kcache_quant)
    flat_config.k_bits = flatquant_config.get('k_bits', FlatConfig.k_bits)
    flat_config.k_sym = flatquant_config.get('k_sym', FlatConfig.k_sym)
    flat_config.use_vcache_quant = flatquant_config.get('use_vcache_quant', FlatConfig.use_vcache_quant)
    flat_config.v_bits = flatquant_config.get('v_bits', FlatConfig.v_bits)
    flat_config.v_sym = flatquant_config.get('v_sym', FlatConfig.v_sym)
    
    # cali config
    flat_config.add_diag = flatquant_config.get('add_diag', FlatConfig.add_diag)
    flat_config.epochs = flatquant_config.get('epochs', FlatConfig.epochs)
    flat_config.cali_bsz = flatquant_config.get('cali_bsz', FlatConfig.cali_bsz)
    flat_config.flat_lr = flatquant_config.get('flat_lr', FlatConfig.flat_lr)
    flat_config.cali_trans = flatquant_config.get('cali_trans', FlatConfig.cali_trans)

    return flat_config


class FlatQuantAttention(nn.Module):
    def __init__(self, module: nn.Module, layer_name: str="", layer_config = None):
        super().__init__()

        self.flat_config = flat_config_parser(layer_config)

        # module
        self.module_config = module.config
        self.layer_idx = module.layer_idx
        self.head_dim = module.head_dim
        self.num_key_value_groups = module.num_key_value_groups
        self.scaling = module.scaling
        self.attention_dropout = module.attention_dropout
        self.is_causal = True

        self.q_norm = module.q_norm if hasattr(module, 'q_norm') else None
        self.k_norm = module.k_norm if hasattr(module, 'k_norm') else None
        if hasattr(module, 'sliding_window'):
            self.sliding_window = module.sliding_window

        self.q_proj = FlatQuantizedLinear(module.q_proj, self.flat_config)
        self.k_proj = FlatQuantizedLinear(module.k_proj, self.flat_config)
        self.v_proj = FlatQuantizedLinear(module.v_proj, self.flat_config)
        self.o_proj = FlatQuantizedLinear(module.o_proj, self.flat_config)
        self.add_fq_trans()

        if self.flat_config.use_kcache_quant and self.flat_config.k_bits < 16:
            self.k_cache_quantizer = ActivationQuantizer(bits=self.flat_config.k_bits,
                sym=self.flat_config.k_sym, lac=self.flat_config.lac, groupsize=-1, )
        if self.flat_config.use_vcache_quant and self.flat_config.v_bits < 16:
            self.v_cache_quantizer = ActivationQuantizer(bits=self.flat_config.v_bits,
                sym=self.flat_config.v_sym, lac=self.flat_config.lac, groupsize=-1, )

        self._ori_mode = False
        self._eval_mode = False
        self.ln_smax = torch.ones_like(self.q_proj.linear.weight.abs().max(dim=0)[0]).npu() * 1e-5

    def add_fq_trans(self):
        '''
        addd the low rank trans matrix before the ln
        '''
        if self.flat_config.a_bits < 16 or self.flat_config.w_bits < 16:
            ln_dim_left, ln_dim_right = get_decompose_dim(self.q_proj.linear.weight.shape[1])
            self.ln_trans = SVDDecomposeTransMatrix(ln_dim_left, ln_dim_right, add_diag=self.flat_config.add_diag)
            if self.flat_config.use_o_quant:
                self.o_trans = SVDSingleTransMatrix(self.module_config.num_attention_heads)
            else:
                self.o_trans = None
        else:
            self.ln_trans, self.o_trans = None, None

        head_dim = self.module_config.hidden_size // self.module_config.num_attention_heads
        if self.flat_config.use_kcache_quant and self.flat_config.k_bits < 16:
            self.kcache_trans = SVDSingleTransMatrix(head_dim)
        else:
            self.kcache_trans = None

        if self.flat_config.use_vcache_quant and \
            (self.flat_config.v_bits < 16 or self.flat_config.w_bits < 16 or self.flat_config.a_bits < 16):
            self.vcache_trans = SVDSingleTransMatrix(head_dim)
        else:
            self.vcache_trans = None

    def _trans_forward_after_ln(self, hidden_states):
        if self.ln_trans is not None:
            hidden_states = self.ln_trans(hidden_states)
        query_states = self.q_proj(hidden_states, qa_trans=self.ln_trans)
        key_states = self.k_proj(hidden_states, qa_trans=self.ln_trans)
        value_states = self.v_proj(hidden_states, qa_trans=self.ln_trans, out_trans=self.vcache_trans)
        return query_states, key_states, value_states

    def _ori_forward_after_ln(self, hidden_states):
        if hasattr(self, "ln_smax"):
            self.ln_smax = torch.maximum(self.ln_smax, \
                hidden_states.reshape(-1, hidden_states.shape[-1]).abs().max(0)[0].clone().detach())
        query_states = self.q_proj._ori_forward(hidden_states)
        key_states = self.k_proj._ori_forward(hidden_states)
        value_states = self.v_proj._ori_forward(hidden_states)
        return query_states, key_states, value_states

    def quant_vcache(self, value_states):
        if self.v_bits < 16:
            value_states = self.v_cache_quantizer(value_states)
        return value_states

    def quant_kcache(self, q, k):
        if not self.flat_config.k_bits < 16:
            return q, k
        # Q/K transform
        if self.kcache_trans is not None:
            q = self.kcache_trans(q, inv_t=True)
            k = self.kcache_trans(k)
        # TODO: by default do the per-head quantizaion for k-v-cache
        if self.flat_config.k_bits < 16:
            k = self.k_cache_quantizer(k).to(q)
        return q, k

    def forward_qkv(
            self, query_states, key_states, value_states, bsz, q_len, **kwargs
        ):
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

        # ---- here do the kv cache quantization ----
        if not self._ori_mode:
            if self.flat_config.use_kcache_quant:
                query_states, key_states = self.quant_kcache(query_states, key_states)
            if self.flat_config.use_vcache_quant:
                value_states = self.quant_vcache(value_states)

        past_key_values = None
        if kwargs.get('past_key_value') is not None:
            past_key_values = kwargs.pop('past_key_value')
        if kwargs.get('past_key_values') is not None:
            past_key_values = kwargs.pop('past_key_values')
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, 'cache_position': kwargs.pop('cache_position')}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.module_config._attn_implementation != 'eager':
            if self.module_config._attn_implementation == 'sdpa' and kwargs.get('output_attentions', False):
                print(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.module_config._attn_implementation]

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

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()

        if self._ori_mode or (self.o_trans is None and self.vcache_trans is None):
            attn_output = self.o_proj._ori_forward(attn_output)
        else:
            init_shape = attn_output.shape
            head_dim = self.module_config.hidden_size // self.module_config.num_attention_heads
            attn_shape = [-1, self.module_config.num_attention_heads, head_dim]
            attn_output = attn_output.reshape(attn_shape)
            if self.o_trans is None and self.vcache_trans is not None:
                attn_output = torch.matmul(attn_output, self.vcache_trans.get_matrix(inv_t=True).T.to(attn_output)).reshape(init_shape)
                attn_output = self.o_proj(attn_output)
            else:
                attn_output = torch.matmul(self.o_trans.get_matrix().T.to(attn_output), attn_output).reshape(init_shape)
                if not self._eval_mode:
                    attn_o_og_it = self.o_trans.get_matrix(inv_t=True)
                    attn_v_og_it = self.vcache_trans.get_matrix(inv_t=True)
                    attn_output = self.o_proj(attn_output, qa_trans=[attn_o_og_it, attn_v_og_it])
                else:
                    attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights

    def forward(self, hidden_states, **kwargs):
        bsz, q_len, _ = hidden_states.size()
        
        if self._ori_mode:
            query_states, key_states, value_states = self._ori_forward_after_ln(hidden_states)
        else:
            query_states, key_states, value_states = self._trans_forward_after_ln(hidden_states)
        
        attn_res = self.forward_qkv(query_states, key_states, value_states, bsz, q_len, **kwargs)
        return attn_res

    def reparameterize(self):
        if self.ln_trans is not None:
            self.ln_trans.to_eval_mode()
        if self.kcache_trans is not None:
            self.kcache_trans.to_eval_mode()
        if self.vcache_trans is not None:
            self.vcache_trans.to_eval_mode()
        if self.o_trans is not None:
            self.o_trans.to_eval_mode()
        self.q_proj.reparameterize(qa_trans=self.ln_trans)
        self.k_proj.reparameterize(qa_trans=self.ln_trans)
        self.v_proj.reparameterize(qa_trans=self.ln_trans, out_trans=self.vcache_trans)
        if self.o_trans is not None and self.vcache_trans is not None:
            attn_o_og_it = self.o_trans.get_matrix(inv_t=True)
            attn_v_og_it = self.vcache_trans.get_matrix(inv_t=True)
            self.o_proj.reparameterize(qa_trans=[attn_o_og_it, attn_v_og_it])
        self._eval_mode = True

    def init_diag_scale(self, alpha=0.5):
        assert hasattr(self, "ln_smax")
        qkvw_smax = torch.cat([self.q_proj.linear.weight, self.k_proj.linear.weight,
                                self.v_proj.linear.weight], dim=0).abs().max(dim=0)[0]
        if self.ln_trans is not None:
            self.ln_trans.diag_scale.data = get_init_scale(qkvw_smax, self.ln_smax, alpha)
        del self.ln_smax

    def rep_matrix_only(self, ):
        if self.ln_trans is not None:
            self.ln_trans.to_eval_mode()
        if self.kcache_trans is not None:
            self.kcache_trans.to_eval_mode()
        if self.vcache_trans is not None:
            self.vcache_trans.to_eval_mode()
        if self.o_trans is not None:
            self.o_trans.to_eval_mode()


class FlatQuantMLP(nn.Module):
    def __init__(self, module: nn.Module, layer_name: str="", layer_config = None):
        super().__init__()

        self.flat_config = flat_config_parser(layer_config)
        self.module_config = module.config

        # quant module
        self.hidden_size = module.config.hidden_size
        self.intermediate_size = module.config.intermediate_size
        self.act_fn = module.act_fn if hasattr(module, 'act_fn') else None

        self.up_proj = FlatQuantizedLinear(module.up_proj, self.flat_config)
        self.gate_proj = FlatQuantizedLinear(module.gate_proj, self.flat_config)
        self.down_proj = FlatQuantizedLinear(module.down_proj, self.flat_config)
        self.add_fq_trans()

        self._ori_mode = False
        self.up_smax = torch.ones_like(self.up_proj.linear.weight.abs().max(dim=0)[0]).npu() * 1e-5
        self.down_smax = torch.ones_like(self.down_proj.linear.weight.abs().max(dim=0)[0]).npu() * 1e-5
        
    def add_fq_trans(self):
        if self.flat_config.a_bits < 16 or self.flat_config.w_bits < 16:
            up_dim_left, up_dim_right = get_decompose_dim(self.up_proj.linear.weight.shape[1])
            self.up_gate_trans = SVDDecomposeTransMatrix(up_dim_left, up_dim_right, 
                                                        add_diag=self.flat_config.add_diag)
            
            if self.flat_config.use_down_quant:
                down_dim_left, down_dim_right = get_decompose_dim(self.down_proj.linear.weight.shape[1])
                self.down_trans = SVDDecomposeTransMatrix(down_dim_left,
                                                        down_dim_right, add_diag=self.flat_config.add_diag)
            else:
                self.down_trans = None
        else:
            self.up_gate_trans, self.down_trans = None, None

    def _trans_forward(self, x):
        # up & gate trans quant
        x_ts = self.up_gate_trans(x) if self.up_gate_trans is not None else x
        up_states = self.up_proj(x_ts, qa_trans=self.up_gate_trans)
        gate_states = self.gate_proj(x_ts, qa_trans=self.up_gate_trans)
        x_act_fn = self.act_fn(gate_states) * up_states

        # not quantize down_proj
        if not self.flat_config.use_down_quant:
            return self.down_proj._ori_forward(x_act_fn)

        # down_proj trans quant
        x_ts_2 = self.down_trans(x_act_fn) if self.down_trans is not None else x_act_fn
        down_states = self.down_proj(x_ts_2, qa_trans=self.down_trans)
        return down_states

    def _ori_forward(self, x):
        '''origin implement: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))'''
        self.up_smax = torch.maximum(self.up_smax, x.reshape(-1, x.shape[-1]).abs().max(0)[0].clone().detach())
        x = self.act_fn(self.gate_proj._ori_forward(x)) * self.up_proj._ori_forward(x)
        self.down_smax = torch.maximum(self.down_smax, x.reshape(-1, x.shape[-1]).abs().max(0)[0].clone().detach())
        down_states = self.down_proj._ori_forward(x)
        return down_states

    def forward(self, x):
        y = self._ori_forward(x) if self._ori_mode else self._trans_forward(x)
        return y

    def reparameterize(self):
        if self.up_gate_trans is not None:
            self.up_gate_trans.to_eval_mode()
        if self.down_trans is not None:
            self.down_trans.to_eval_mode()
        self.gate_proj.reparameterize(qa_trans=self.up_gate_trans)
        self.up_proj.reparameterize(qa_trans=self.up_gate_trans)
        if self.flat_config.use_down_quant:
            self.down_proj.reparameterize(qa_trans=self.down_trans)
        if self.up_gate_trans is not None:
            self.up_gate_trans.use_diag = False
        # merge trans's diag scale
        if self.down_trans is not None and self.down_trans.add_diag:
            up_weight = self.up_proj.linear.weight
            ori_dtype = up_weight.dtype
            up_weight = up_weight.to(torch.float64).T.mul(self.down_trans.diag_scale.to(torch.float64)).T
            self.up_proj.linear.weight.data = up_weight.to(ori_dtype)
            self.down_trans.use_diag = False

    def init_diag_scale(self, alpha=0.5):
        assert hasattr(self, "up_smax") and hasattr(self, "down_smax")
        upw_smax = torch.cat([self.up_proj.linear.weight, self.gate_proj.linear.weight], dim=0).abs().max(dim=0)[0]
        downw_smax = self.down_proj.linear.weight.abs().max(dim=0)[0]
        if self.up_gate_trans is not None:
            self.up_gate_trans.diag_scale.data = get_init_scale(upw_smax, self.up_smax, alpha)
        if self.down_trans is not None:
            self.down_trans.diag_scale.data = get_init_scale(downw_smax, self.down_smax, alpha)
        del self.down_smax
        del self.up_smax

    def rep_matrix_only(self, ):
        if self.up_gate_trans is not None:
            self.up_gate_trans.to_eval_mode()
        if self.down_trans is not None:
            self.down_trans.to_eval_mode()
