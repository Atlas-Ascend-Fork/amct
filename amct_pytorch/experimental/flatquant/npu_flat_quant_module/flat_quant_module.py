import torch
import torch.nn.functional as F

from torch import nn
from amct_pytorch.experimental.flatquant.flat_quant_module.default_model_utils import FlatQuantAttention, FlatQuantMLP
from .quant_utils import (
    AscendW4A4FlatQuantDynamicLinearMethod, pack_int4_weights)


class NpuFlatQuantAttention(nn.Module):
    def __init__(self, quant_module: FlatQuantAttention):
        super().__init__()
        self.quant_module = quant_module

        quant_module.q_proj.weight_quantizer.find_params(quant_module.q_proj.linear.weight)
        quant_module.k_proj.weight_quantizer.find_params(quant_module.k_proj.linear.weight)
        quant_module.v_proj.weight_quantizer.find_params(quant_module.v_proj.linear.weight)

        wt_q_proj, self.scale_q_proj = quant_module.q_proj.weight_quantizer.quantize(
            quant_module.q_proj.linear.weight, quantonly=True)
        wt_k_proj, self.scale_k_proj = quant_module.k_proj.weight_quantizer.quantize(
            quant_module.k_proj.linear.weight, quantonly=True)
        wt_v_proj, self.scale_v_proj = quant_module.v_proj.weight_quantizer.quantize(
            quant_module.v_proj.linear.weight, quantonly=True)
        
        self.register_parameter('wt_q_packed', torch.nn.Parameter(pack_int4_weights(wt_q_proj.data),
                                                                  requires_grad=False))
        self.register_parameter('wt_k_packed', torch.nn.Parameter(pack_int4_weights(wt_k_proj.data),
                                                                  requires_grad=False))
        self.register_parameter('wt_v_packed', torch.nn.Parameter(pack_int4_weights(wt_v_proj.data),
                                                                  requires_grad=False))
        
        self.left_trans = quant_module.ln_trans.matrix_left.T # transpose
        self.right_trans = quant_module.ln_trans.matrix_right
        self.method = AscendW4A4FlatQuantDynamicLinearMethod()

        del self.quant_module.q_proj
        del self.quant_module.k_proj
        del self.quant_module.v_proj
        del self.quant_module.ln_trans
        del wt_q_proj, wt_k_proj, wt_v_proj


    def forward(self, hidden_states, position_embeddings, attention_mask,
                    past_key_value, cache_position, **kwargs):
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
            self.left_trans, self.right_trans, self.wt_q_packed, self.scale_q_proj, hidden_states)
        key_states = self.method.apply(
             self.left_trans, self.right_trans, self.wt_k_packed, self.scale_k_proj, hidden_states)
        value_states = self.method.apply(
             self.left_trans, self.right_trans, self.wt_v_packed, self.scale_v_proj, hidden_states)
        if q_len_pad_size_after:
            query_states = query_states[:, :q_len]
            key_states = key_states[:, :q_len]
            value_states = value_states[:, :q_len]
        
        return self.quant_module.forward_qkv(
            query_states, key_states, value_states, bsz, q_len,
            position_embeddings, cache_position, attention_mask, past_key_value, **kwargs
        )


class NpuFlatQuantMLP(nn.Module):
    def __init__(self, quant_module: FlatQuantMLP):
        super().__init__()
        self.quant_module = quant_module

        # if down_proj not quant, use original down_proj linear
        if not quant_module.flat_config.use_down_quant:
            self.down_proj = quant_module.down_proj.linear
        else:
            self.down_proj = None

        quant_module.up_proj.weight_quantizer.find_params(quant_module.up_proj.linear.weight)
        quant_module.gate_proj.weight_quantizer.find_params(quant_module.gate_proj.linear.weight)
        quant_module.down_proj.weight_quantizer.find_params(quant_module.down_proj.linear.weight)

        wt_up_proj, self.scale_up_proj = quant_module.up_proj.weight_quantizer.quantize(
            quant_module.up_proj.linear.weight, quantonly=True)
        wt_gate_proj, self.scale_gate_proj = quant_module.gate_proj.weight_quantizer.quantize(
            quant_module.gate_proj.linear.weight, quantonly=True)
        if not self.down_proj:
            wt_down_proj, self.scale_down_proj = quant_module.down_proj.weight_quantizer.quantize(
                quant_module.down_proj.linear.weight, quantonly=True)

        self.register_parameter('wt_up_packed', torch.nn.Parameter(pack_int4_weights(wt_up_proj.data),
                                                                      requires_grad=False))
        self.register_parameter('wt_gate_packed', torch.nn.Parameter(pack_int4_weights(wt_gate_proj.data),
                                                                      requires_grad=False))
        if not self.down_proj:
            self.register_parameter('wt_down_packed', torch.nn.Parameter(pack_int4_weights(wt_down_proj.data),
                                                                      requires_grad=False))
            del wt_down_proj

        self.ug_left_trans = quant_module.up_gate_trans.matrix_left.T
        self.ug_right_trans = quant_module.up_gate_trans.matrix_right
        if not self.down_proj:
            self.down_left_trans = quant_module.down_trans.matrix_left.T
            self.down_right_trans = quant_module.down_trans.matrix_right
        self.method = AscendW4A4FlatQuantDynamicLinearMethod()

        del self.quant_module.up_proj
        del self.quant_module.gate_proj
        del self.quant_module.up_gate_trans
        del self.quant_module.down_proj
        del self.quant_module.down_trans
        del wt_up_proj, wt_gate_proj


    def forward(self, x):
        # treat indivisible by 16 cases; same as the attention part
        # TODO: remove once allowed
        bsz, q_len, _ = x.size()
        q_len_pad_size_after = 0
        if bsz % 16 and q_len % 16:
            q_len_pad_size_after = (q_len // 16 + 1) * 16 - q_len
            padding = (0, 0) * (x.dim() - 2) + (0, q_len_pad_size_after) + (0, 0)
            x = F.pad(x, padding, mode='constant', value=0)

        up_states = self.method.apply(self.ug_left_trans, self.ug_right_trans, self.wt_up_packed, self.scale_up_proj, x)
        gate_states = self.method.apply(
            self.ug_left_trans, self.ug_right_trans, self.wt_gate_packed, self.scale_gate_proj, x)
        x_act_fn = self.quant_module.act_fn(gate_states) * up_states

        if not self.down_proj:
            down_states = self.method.apply(
                self.down_left_trans, self.down_right_trans, self.wt_down_packed, self.scale_down_proj, x_act_fn)
        else:
            down_states = self.down_proj(x_act_fn)

        if q_len_pad_size_after:
            down_states = down_states[:, :q_len]

        return down_states