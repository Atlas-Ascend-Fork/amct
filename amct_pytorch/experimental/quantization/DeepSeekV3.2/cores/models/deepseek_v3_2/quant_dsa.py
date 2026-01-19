# coding=utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
from torch.nn import Linear
from torch.nn import LayerNorm
from einops import rearrange
from cores.models.deepseek_v3_2.indexer import rotate_activation, fp32_index
from cores.quantization.node import ActivationQuantizer
from cores.quantization.matmul import QuantizedMatmul
from cores.quantization.linear import QuantLinear
from cores.models.deepseek_v3_2.modeling_deepseek_v3_2 import DeepseekV3Attention, apply_rotary_pos_emb


class QuantIndexer(torch.nn.Module):
    def __init__(self, module, args, rotary_emb, bit):
        super().__init__()
        self.dim: int = args.model_args.dim
        self.n_heads: int = args.model_args.index_n_heads
        self.n_local_heads = args.model_args.index_n_heads
        self.head_dim: int = args.model_args.index_head_dim
        self.rope_head_dim: int = args.model_args.qk_rope_head_dim
        self.index_topk: int = args.model_args.index_topk
        self.q_lora_rank: int = args.model_args.q_lora_rank
        self.bit: int = bit
        self.wq_b = QuantLinear(args, module.wq_b, w_bits=self.bit, lwc=False)
        self.wk = Linear(self.dim, self.head_dim, dtype=torch.get_default_dtype(), bias=False)
        self.k_norm = LayerNorm(self.head_dim)
        self.weights_proj = Linear(self.dim, self.n_heads, dtype=torch.get_default_dtype(), bias=False)
        self.softmax_scale = self.head_dim ** -0.5
        self.scale_fmt = args.model_args.scale_fmt
        self.rotary_emb = rotary_emb
        self.afq_rot_q = ActivationQuantizer(bits=self.bit, \
                                             sym=True, lac=False, groupsize=-1, per_tensor=False)
        self.afq_rot_k = ActivationQuantizer(bits=self.bit, \
                                             sym=True, lac=False, groupsize=-1, per_tensor=False)

        self.wk.weight.data[:] = module.wk.weight.data.clone()

        self.k_norm.weight.data = module.k_norm.weight.data.clone()
        self.k_norm.bias.data = module.k_norm.bias.data.clone()

        self.weights_proj.weight.data = module.weights_proj.weight.data.clone()

    def forward(self, x: torch.Tensor, qr: torch.Tensor, mask, position_ids):
        bsz, seqlen, _ = x.size()
        q = self.wq_b(qr)
        q = rearrange(q, 'b s (h d) -> b s h d', d=self.head_dim)
        q_pe, q_nope = torch.split(q, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1)
        cos, sin = self.rotary_emb(q, seq_len=seqlen)
        q = torch.cat([q_pe, q_nope], dim=-1)
        k = self.wk(x)
        k = self.k_norm(k)
        k_pe, k_nope = torch.split(k, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1)
        q_pe = q_pe.permute(0, 2, 1, 3)
        k_pe = k_pe.unsqueeze(2).permute(0, 2, 1, 3)
        q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)
        q_pe = q_pe.permute(0, 2, 1, 3)
        k_pe = k_pe.permute(0, 2, 1, 3).squeeze(2)
        k = torch.cat([k_pe, k_nope], dim=-1)

        q = rotate_activation(q)
        k = rotate_activation(k)

        q = q.half()
        k = k.half()

        q_fq = self.afq_rot_q(q)
        k_fq = self.afq_rot_k(k)

        weights = self.weights_proj(x).half() * self.n_heads ** -0.5
        weights = weights.unsqueeze(-1) * self.softmax_scale
        index_score = fp32_index(q_fq, k_fq.unsqueeze(2), weights)
        if mask is not None:
            index_score += mask.squeeze(0)
        topk_indices = index_score.topk(min(self.index_topk, seqlen), dim=-1)[1]
        topk_indices_ = topk_indices.clone()

        return topk_indices_


class QuantDSA(DeepseekV3Attention):
    def __init__(self, args, module: DeepseekV3Attention):
        super().__init__(module.config, module.layer_idx)
        self.bit = 8 if args.cls == 'c8' else 16
        if self.q_lora_rank is None:
            self.q_proj = module.q_proj
        else:
            self.q_a_proj = module.q_a_proj
            self.q_a_layernorm = module.q_a_layernorm
            self.q_b_proj = module.q_b_proj

        self.kv_a_proj_with_mqa = module.kv_a_proj_with_mqa

        self.kv_a_layernorm = module.kv_a_layernorm
        self.kv_b_proj = module.kv_b_proj

        self.o_proj = QuantLinear(args, module.o_proj, w_bits=self.bit, lwc=False)

        self.qnope_matmul = QuantizedMatmul(args, bits=self.bit, lwc=False)
        self.qr_matmul = QuantizedMatmul(args, bits=self.bit, lwc=False)

        self.q_cache_quantizer = ActivationQuantizer(bits=16, \
                                                     sym=not (args.q_asym), lac=False, groupsize=-1)

        self.k_cache_quantizer = ActivationQuantizer(bits=self.bit, \
                                                     sym=not (args.k_asym), lac=True, groupsize=-1, per_tensor=False)

        self.afq1 = ActivationQuantizer(bits=self.bit, \
                                        sym=True, lac=False, groupsize=-1, per_tensor=False)
        self.afq2 = ActivationQuantizer(bits=self.bit, \
                                        sym=True, lac=False, groupsize=-1, per_tensor=False)
        self.indexer = QuantIndexer(module.indexer, args, self.rotary_emb, self.bit)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask=None,
            position_ids=None,
            past_key_value=None,
            output_attentions: bool = False,
            use_cache: bool = False,
            **kwargs,
    ):

        kv_seq_len = hidden_states.shape[-2]
        if self.q_lora_rank is not None:
            # get Wuq and Wqr
            _, q_b_proj_inc = self.q_b_proj.weight.size()
            q_b_proj = self.q_b_proj.weight.t().reshape(q_b_proj_inc, self.num_heads, self.q_head_dim)
            q_b_proj_c_weight, q_b_proj_r_weight = torch.split(q_b_proj, [self.qk_nope_head_dim, self.qk_rope_head_dim],
                                                               dim=-1)

        # get Wdkv and Wkr
        dkv_weight, kr_weight = torch.split(self.kv_a_proj_with_mqa.weight.t(),
                                            [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)

        # get Wuk and Wuv
        _, kv_b_proj_inc = self.kv_b_proj.weight.size()
        ukv_weight = self.kv_b_proj.weight.t().reshape(kv_b_proj_inc, self.num_heads,
                                                       (self.qk_nope_head_dim + self.v_head_dim))
        uk_weight, uv_weight = torch.split(ukv_weight, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        uk_weight = uk_weight.reshape(kv_b_proj_inc, -1).t().reshape(self.num_heads, self.qk_nope_head_dim,
                                                                     kv_b_proj_inc)

        # start calculate
        bsz, q_len, _ = hidden_states.size()

        if self.q_lora_rank is not None:
            q_norm = self.q_a_layernorm(self.q_a_proj(hidden_states))
            q_norm = self.afq1(q_norm)
            q = q_norm
        else:
            q = self.q_proj(hidden_states)
            q_norm = q
        ckv = torch.matmul(hidden_states, dkv_weight)  # compressed_kv
        ckv = self.kv_a_layernorm(ckv)

        cos, sin = self.rotary_emb(ckv, seq_len=kv_seq_len)
        if self.q_lora_rank is not None:
            # wqr
            qr = self.qr_matmul(q, q_b_proj_r_weight.reshape(q_b_proj_inc, -1)).view(bsz, q_len, self.num_heads,
                                                                                     self.qk_rope_head_dim).transpose(1,
                                                                                                                      2)
            # wuq
            q_nope = self.qnope_matmul(q, q_b_proj_c_weight.reshape(q_b_proj_inc, -1)).view(bsz, q_len, self.num_heads,
                                                                                            self.qk_nope_head_dim).transpose(
                1, 2)

        else:
            q = q.view(bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.qk_rope_head_dim).transpose(1, 2)
            q_nope, qr = torch.split(
                q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
            )
        kr = torch.matmul(hidden_states, kr_weight)

        kr = kr.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        qr, kr = apply_rotary_pos_emb(qr, kr, cos, sin, position_ids)

        qn = torch.matmul(q_nope, uk_weight)

        qn = self.q_cache_quantizer(qn)
        q_n_r = torch.cat((qn, qr), dim=-1)

        ckv = self.k_cache_quantizer(ckv.unsqueeze(1))
        k_c_r = torch.cat((ckv, kr), dim=-1)

        attn_weights = torch.matmul(q_n_r, k_c_r.transpose(2, 3))

        fix_k_c_r = k_c_r

        attn_weights = attn_weights * self.softmax_scale

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f'Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)},'
                f'but is {attn_weights.size()}'
            )
        assert attention_mask is not None
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f'Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)},'
                    f'but is {attention_mask.size()}'
                )

            topk_indices = self.indexer(hidden_states, q_norm, attention_mask, position_ids)
            index_mask = torch.full((bsz, kv_seq_len, kv_seq_len), float("-inf"), device=hidden_states.device).scatter_(
                -1, topk_indices, 0).unsqueeze(0)
            index_mask += attention_mask
            attn_weights = attn_weights + index_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            k_c_r.dtype
        )
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )

        attn_output = torch.matmul(attn_weights, fix_k_c_r.squeeze(1)[:, :, :kv_b_proj_inc])

        attn_output = torch.matmul(attn_output, uv_weight.transpose(0, 1))

        if attn_output.size() != (bsz, self.num_heads, q_len, self.v_head_dim):
            raise ValueError(
                f'`attn_output` should be of size {(bsz, self.num_heads, q_len, self.v_head_dim)},'
                f' but is {attn_output.size()}'
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim)

        attn_output = self.afq2(attn_output)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_n_r = None

        return attn_output, attn_n_r, past_key_value
