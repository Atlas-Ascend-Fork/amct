import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from amct_pytorch.common.models.llm.common.base import BaseModel
from amct_pytorch.common.models.llm.common.ptq_units import iter_indexed_units, make_ptq_unit
from amct_pytorch.common.models.llm.common.quant_apply import QuantGatedMLP
from amct_pytorch.common.models.llm.qwen.moe_common import QuantGatedExperts, is_packed_experts, pack_gated_expert_weights
from amct_pytorch.algorithms.quant import AlgoBuildContext
from amct_pytorch.quantization.modules.quant_base import ActivationQuantizer, build_algorithms_by_target
from amct_pytorch.quantization.modules.quant_linear import QuantLinear
from amct_pytorch.quantization.bit_policy import ensure_bit_policy
from amct_pytorch.common.models import MODEL_REGISTRY
from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (
    ALL_ATTENTION_FUNCTIONS,
    GlmMoeDsaAttention,
    GlmMoeDsaDecoderLayer,
    apply_rotary_pos_emb,
    eager_attention_forward,
    is_flash_attention_requested,
)


def _linear_with_structure(hidden_states, linear, structure_transform=None, name=None):
    weight = linear.weight
    if structure_transform is not None:
        weight = structure_transform(weight, inv_t=True, name=name)
    return F.linear(hidden_states, weight, linear.bias)


class QuantGLM5Indexer(nn.Module):
    def __init__(self, args, module):
        super().__init__()
        ensure_bit_policy(args)
        self.config = module.config
        self.layer_idx = module.layer_idx
        self.hidden_size = self.config.hidden_size
        self.n_heads = self.config.index_n_heads
        self.head_dim = self.config.index_head_dim
        self.qk_rope_head_dim = self.config.qk_rope_head_dim
        self.index_topk = self.config.index_topk
        self.q_lora_rank = self.config.q_lora_rank
        self.wq_b = QuantLinear(args, module.wq_b, w_bits=args.bit_policy["attn-linear"]["wq_b"].w, name="wq_b")
        self.wk = module.wk
        self.k_norm = module.k_norm
        self.weights_proj = module.weights_proj
        self.softmax_scale = module.softmax_scale
        self._cached_keys = None

    @torch.no_grad()
    def forward(self, hidden_states, q_resid, position_embeddings, attention_mask, use_cache=False, structure_transform=None):
        batch_size, seq_len, _ = hidden_states.shape
        cos, sin = position_embeddings

        q = self.wq_b(q_resid)
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim)
        q_pe, q_nope = torch.split(q, [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim], dim=-1)
        q_pe = apply_rotary_pos_emb(q_pe, cos, sin, unsqueeze_dim=2)
        q = torch.cat([q_pe, q_nope], dim=-1)

        k = self.k_norm(_linear_with_structure(hidden_states, self.wk, structure_transform=structure_transform, name="wk"))
        k_pe, k_nope = torch.split(k, [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim], dim=-1)
        k_pe = apply_rotary_pos_emb(k_pe.unsqueeze(2), cos, sin, unsqueeze_dim=2).squeeze(2)
        k = torch.cat([k_pe, k_nope], dim=-1)

        if seq_len > 1:
            self._cached_keys = None

        if use_cache:
            if self._cached_keys is not None:
                k_cached = torch.cat([self._cached_keys, k], dim=1)
            else:
                k_cached = k
            self._cached_keys = k_cached
        else:
            k_cached = k

        weights = _linear_with_structure(
            hidden_states,
            self.weights_proj,
            structure_transform=structure_transform,
            name="weights_proj",
        ).float() * (self.n_heads ** -0.5)
        scores = torch.einsum("bshd,btd->bsht", q.float(), k_cached.float()) * self.softmax_scale
        index_scores = torch.einsum("bsht,bsh->bst", scores, weights)

        if attention_mask is not None:
            index_scores = index_scores + attention_mask

        total_len = index_scores.shape[-1]
        topk = min(self.index_topk, total_len)
        return index_scores.topk(topk, dim=-1).indices


class QuantGLM5Attn(GlmMoeDsaAttention):
    def __init__(self, args, module: GlmMoeDsaAttention):
        super().__init__(module.config, module.layer_idx)
        self.quant_args = args
        ensure_bit_policy(args)
        self.enable_attn_linear = "attn-linear" in args.quant_target
        self.hidden_size = getattr(module, "hidden_size", module.config.hidden_size)
        bits = args.bit_policy["attn-linear"]
        kv_a, kv_b, o = bits["kv_a_proj_with_mqa"], bits["kv_b_proj"], bits["o_proj"]
        if self.q_lora_rank is None:
            q = bits["q_proj"]
            self.q_proj = QuantLinear(args, module.q_proj, w_bits=q.w, name="q_proj")
        else:
            q_a, q_b = bits["q_a_proj"], bits["q_b_proj"]
            self.q_a_proj = QuantLinear(args, module.q_a_proj, w_bits=q_a.w, name="q_a_proj")
            self.q_a_layernorm = module.q_a_layernorm
            self.q_b_proj = QuantLinear(args, module.q_b_proj, w_bits=q_b.w, name="q_b_proj")

        self.kv_a_proj_with_mqa = QuantLinear(args, module.kv_a_proj_with_mqa, w_bits=kv_a.w, name="kv_a_proj_with_mqa")
        self.kv_a_layernorm = module.kv_a_layernorm
        self.kv_b_proj = QuantLinear(args, module.kv_b_proj, w_bits=kv_b.w, name="kv_b_proj")
        self.o_proj = QuantLinear(args, module.o_proj, w_bits=o.w, name="o_proj")
        self.scaling = module.scaling
        self.indexer = QuantGLM5Indexer(args, module.indexer)
        self.input_transform = None
        self.out_transform = None
        if self.enable_attn_linear:
            self._init_structure_transforms()
            self.inp_afq = ActivationQuantizer(args, bits=kv_a.a)
            self.o_proj_afq = ActivationQuantizer(args, bits=o.a)
        else:
            self.inp_afq = nn.Identity()
            self.o_proj_afq = nn.Identity()

    def _init_structure_transforms(self):
        ctx = AlgoBuildContext(matrix_size=128, dim_size=self.hidden_size)
        self.input_transform = build_algorithms_by_target(self.quant_args, "structure", ctx)
        ctx = AlgoBuildContext(matrix_size=128, dim_size=self.num_heads * self.v_head_dim)
        self.out_transform = build_algorithms_by_target(self.quant_args, "structure", ctx)

    def forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask=None,
        past_key_values=None,
        cache_position=None,
        **kwargs,
    ):
        batch_size, seq_length = hidden_states.shape[:-1]
        cos, sin = position_embeddings
        if self.input_transform is not None:
            hidden_states = self.input_transform(hidden_states)
        hidden_states = self.inp_afq(hidden_states)

        if self.q_lora_rank is None:
            query_states = self.q_proj(hidden_states, structure_transform=self.input_transform)
            q_resid = None
        else:
            q_resid = self.q_a_layernorm(self.q_a_proj(hidden_states, structure_transform=self.input_transform))
            query_states = self.q_b_proj(q_resid)
        query_states = query_states.view(batch_size, seq_length, -1, self.qk_head_dim).transpose(1, 2)
        q_nope, q_pe = torch.split(query_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_pe = apply_rotary_pos_emb(q_pe, cos, sin, unsqueeze_dim=1)

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states, structure_transform=self.input_transform)
        k_compressed, k_pe = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_compressed = self.kv_a_layernorm(k_compressed)

        kv_expanded = self.kv_b_proj(k_compressed)
        kv_expanded = kv_expanded.view(batch_size, seq_length, -1, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, value_states = torch.split(kv_expanded, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k_nope = k_nope.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        k_pe = k_pe.view(batch_size, 1, seq_length, self.qk_rope_head_dim)
        k_pe = apply_rotary_pos_emb(k_pe, cos, sin, unsqueeze_dim=1)
        k_pe = k_pe.expand(-1, k_nope.shape[1], -1, -1)

        query_states = torch.cat([q_nope, q_pe], dim=-1)
        key_states = torch.cat([k_nope, k_pe], dim=-1)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        indexer_mask = (
            attention_mask[:, 0, :, :]
            if attention_mask is not None and attention_mask.dim() == 4
            else attention_mask.unsqueeze(1)
            if attention_mask is not None
            else None
        )
        topk_indices = self.indexer(
            hidden_states,
            q_resid,
            position_embeddings,
            indexer_mask,
            use_cache=past_key_values is not None,
            structure_transform=self.input_transform,
        )

        total_len = key_states.shape[2]
        index_mask = torch.full(
            (batch_size, seq_length, total_len),
            float("-inf"),
            device=hidden_states.device,
            dtype=query_states.dtype,
        )
        index_mask.scatter_(-1, topk_indices, 0.0)
        index_mask = index_mask.unsqueeze(1)
        if attention_mask is not None and attention_mask.dim() == 4:
            causal_mask = attention_mask[..., :total_len]
            combined_mask = index_mask + causal_mask
        else:
            combined_mask = (
                attention_mask.masked_fill(index_mask == float("-inf"), float("-inf"))
                if attention_mask is not None
                else index_mask
            )

        if is_flash_attention_requested(self.config) and self.qk_head_dim != self.v_head_dim:
            value_states = F.pad(value_states, [0, self.qk_head_dim - self.v_head_dim])

        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation,
            eager_attention_forward,
        )
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            combined_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            indices=topk_indices,
            **kwargs,
        )

        if is_flash_attention_requested(self.config) and self.qk_head_dim != self.v_head_dim:
            attn_output = attn_output[:, :, :, : self.v_head_dim]

        attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
        if self.out_transform is not None:
            attn_output = self.out_transform(attn_output)
        attn_output = self.o_proj_afq(attn_output)
        attn_output = self.o_proj(attn_output, structure_transform=self.out_transform)
        return attn_output, attn_weights


class QuantGLM5MLP(QuantGatedMLP):
    pass


@MODEL_REGISTRY.register(
    name="glm5",
    task="llm",
    family="glm",
    description="GLM-5.1 model adapter",
)
class GLM5(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        self._ensure_glm5_config()
        self.cls = GlmMoeDsaDecoderLayer
        self.model = self.empty_weights_model()
        self.num_layers = self.config.num_hidden_layers

    def float_model(self):
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
        )

    def empty_weights_model(self):
        return super().empty_weights_model()

    def get_layer_weight_prefix(self, layer_idx: int) -> str:
        return f"model.layers.{layer_idx}."

    def _is_moe_layer(self, layer_idx: int) -> bool:
        first_dense = self.config.first_k_dense_replace or 0
        moe_freq = self.config.moe_layer_freq or 1
        return layer_idx >= first_dense and ((layer_idx - first_dense) % moe_freq == 0)

    def _ensure_glm5_config(self):
        self.config._attn_implementation = "eager"

        if getattr(self.config, "mlp_layer_types", None):
            return

        num_layers = self.config.num_hidden_layers
        first_dense = getattr(self.config, "first_k_dense_replace", 3)
        moe_freq = getattr(self.config, "moe_layer_freq", 1)
        mlp_layer_types = []
        for layer_idx in range(num_layers):
            is_sparse = layer_idx >= first_dense and ((layer_idx - first_dense) % moe_freq == 0)
            mlp_layer_types.append("sparse" if is_sparse else "dense")
        self.config.mlp_layer_types = mlp_layer_types

    def load_layer_weight(self, prefix):
        return super().load_layer_weight(prefix)

    def load_embed_state_dict(self):
        return super().load_embed_state_dict()

    def block(self, layer_idx):
        decoder_layer = self.cls(self.config, layer_idx)
        state_dict = self.load_layer_weight(self.get_layer_weight_prefix(layer_idx))
        if self._is_moe_layer(layer_idx):
            state_dict = pack_gated_expert_weights(state_dict, expert_prefix="mlp.experts")
        decoder_layer.load_state_dict(state_dict, strict=True)
        decoder_layer.eval().bfloat16()
        return decoder_layer

    def do_embedding_forward(self, samples, dtype=torch.bfloat16, hook_name=None):
        return super().do_embedding_forward(samples, dtype=dtype, hook_name=hook_name)

    def do_block_forward(self, layer_idx, samples, hook_name=None, use_quant_block=False, enable_quant=False):
        return super().do_block_forward(
            layer_idx,
            samples,
            hook_name=hook_name,
            use_quant_block=use_quant_block,
            enable_quant=enable_quant,
        )

    def do_head_forward(self, inps):
        return super().do_head_forward(inps)

    def build_quant_block(self, layer_idx):
        decoder_layer = self.block(layer_idx)

        if "attn-cache" in self.quant_target:
            raise NotImplementedError("glm5 does not support quant_target='attn-cache' yet.")
        if "attn-linear" in self.quant_target:
            decoder_layer.self_attn = QuantGLM5Attn(self.args, decoder_layer.self_attn)

        mlp = getattr(decoder_layer, "mlp", None)
        if mlp is None:
            return decoder_layer

        if hasattr(mlp, "experts"):
            if "moe" in self.quant_target:
                if not is_packed_experts(mlp.experts):
                    raise TypeError(f"Expected packed experts, got {type(mlp.experts).__name__}")
                mlp.experts = QuantGatedExperts(self.args, mlp.experts)
            if "mlp" in self.quant_target and hasattr(mlp, "shared_experts"):
                mlp.shared_experts = QuantGLM5MLP(self.args, mlp.shared_experts)
            return decoder_layer

        if "mlp" in self.quant_target:
            decoder_layer.mlp = QuantGLM5MLP(self.args, mlp)
        return decoder_layer

    def iter_ptq_units(self, layer_idx, block):
        if "attn-linear" in self.quant_target or "attn-cache" in self.quant_target:
            attn = getattr(block, "self_attn", None)
            if attn is None:
                logger.warning("Layer {} has no self_attn, fallback to block PTQ.", layer_idx)
                yield from super().iter_ptq_units(layer_idx, block)
                return
            yield make_ptq_unit("attn", "self_attn", layer_idx, attn)
            return

        mlp = getattr(block, "mlp", None)
        if mlp is None:
            raise ValueError(f"Unsupported quant target '{self.quant_target}'.")

        if "moe" in self.quant_target:
            experts = getattr(mlp, "experts", None)
            if experts is None:
                return
            if not isinstance(experts, QuantGatedExperts):
                raise TypeError(f"Expected QuantGatedExperts, got {type(experts).__name__}")
            yield from iter_indexed_units(
                kind="moe",
                name_prefix="expert",
                layer_idx=layer_idx,
                items=experts.expert_modules,
                metadata_fn=lambda expert_idx, _: {"expert_idx": expert_idx},
            )
            if "mlp" in self.quant_target:
                shared_experts = getattr(mlp, "shared_experts", None)
                if shared_experts is not None:
                    yield make_ptq_unit("mlp", "shared_experts", layer_idx, shared_experts)
            return

        if "mlp" in self.quant_target:
            shared_experts = getattr(mlp, "shared_experts", None)
            if shared_experts is not None:
                yield make_ptq_unit("mlp", "shared_experts", layer_idx, shared_experts)
                return
            yield make_ptq_unit("mlp", "mlp", layer_idx, mlp)

    def iter_deploy_bindings(self, layer_idx, block):
        weight_prefix = self.get_layer_weight_prefix(layer_idx)
        for name, module in block.named_modules():
            if not isinstance(module, QuantLinear):
                continue

            if name.startswith("mlp.experts.expert_modules."):
                parts = name.split(".")
                if len(parts) != 5:
                    raise ValueError(f"Unexpected GLM5 expert module name: {name}")
                _, _, _, expert_idx, proj_name = parts
                yield f"{weight_prefix}mlp.experts.{expert_idx}.{proj_name}.weight", module
                continue

            yield f"{weight_prefix}{name}.weight", module

    def load_unit_inputs(self, data_dir, unit):
        return super().load_unit_inputs(data_dir, unit)
