import torch
from loguru import logger
from safetensors import safe_open
from compressed_tensors.utils.safetensors_load import get_weight_mappings
from transformers import AutoModelForCausalLM
from transformers.models.qwen3_next.modeling_qwen3_next import (
    Qwen3NextConfig,
    Qwen3NextDecoderLayer,
    Qwen3NextMLP,
)
from amct_pytorch.common.models.llm.common.base import BaseModel, PtqUnit
from amct_pytorch.common.models.llm.common.moe_unpack import find_moe_module
from amct_pytorch.common.models.llm.common.quant_apply import (
    apply_quant_to_attn,
    build_no_algo_args,
)
from amct_pytorch.common.models.llm.qwen.moe_common import (
    QuantGatedExperts,
    is_packed_experts,
    pack_gated_expert_weights,
)
from amct_pytorch.quantization.modules.quant_linear import QuantLinear
from amct_pytorch.common.models import MODEL_REGISTRY
from amct_pytorch.common.models.llm.qwen.qwen3_next.quant_module import (
    QuantQwen3NextAttn, QuantQwen3NextLinearAttn, QuantQwen3NextMLP,
)


@MODEL_REGISTRY.register(
    name="qwen3_next",
    task="llm",
    family="qwen",
    description="Qwen3 Next model adapter",
)
class Qwen3Next(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        self._weight_map = get_weight_mappings(self.model_path)
        self.num_layers = self.config.num_hidden_layers
        self.cls = Qwen3NextDecoderLayer
        self.model = self.empty_weights_model()
        self.parse_quant_mode()

    def parse_quant_mode(self):
        if "mlp" in self.quant_target:
            raise ValueError("Qwen3-next is a moe model and does not support quant_target='mlp'.")

    def float_model(self):
        return super().float_model()

    def empty_weights_model(self):
        return super().empty_weights_model()

    def get_layer_weight_prefix(self, layer_idx: int) -> str:
        return f"model.layers.{layer_idx}."

    def load_layer_weight(self, prefix):
        state_dict = super().load_layer_weight(prefix)
        state_dict = pack_gated_expert_weights(state_dict, expert_prefix="mlp.experts")
        state_dict.pop("linear_attn.rotary_emb.inv_freq", None)
        return state_dict

    def load_embed_state_dict(self):
        return super().load_embed_state_dict()

    def block(self, layer_idx):
        decoder_layer = self.cls(self.config, layer_idx)
        state_dict = self.load_layer_weight(self.get_layer_weight_prefix(layer_idx))
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
        if "attn-linear" in self.quant_target or "attn-cache" in self.quant_target:
            self.apply_quant_attn(decoder_layer)

        if "moe" in self.quant_target:
            quant_moe = find_moe_module(decoder_layer)
            if quant_moe is not None:
                experts = getattr(quant_moe, "experts", None)
                if experts is not None and is_packed_experts(experts):
                    quant_moe.experts = QuantGatedExperts(self.args, experts, group="moe.routed")
                shared_expert = getattr(quant_moe, "shared_expert", None)
                if shared_expert is not None and not isinstance(shared_expert, QuantQwen3NextMLP):
                    shared_expert_args = build_no_algo_args(self.args)
                    quant_moe.shared_expert = QuantQwen3NextMLP(shared_expert_args, shared_expert, group="moe.shared")
        return decoder_layer

    def apply_quant_attn(self, decoder_layer):
        layer_type = decoder_layer.layer_type
        if layer_type == "linear_attention":
            attn_cls = QuantQwen3NextLinearAttn
            quant_attn = getattr(decoder_layer, "linear_attn", None)
            quant_attn.config = self.config
        else:
            attn_cls = QuantQwen3NextAttn
        return apply_quant_to_attn(self.args, decoder_layer, attn_cls)

    def iter_ptq_units(self, layer_idx, block):
        yield from super().iter_ptq_units(layer_idx, block)

    def iter_deploy_bindings(self, layer_idx, block):
        weight_prefix = self.get_layer_weight_prefix(layer_idx)
        for name, module in block.named_modules():
            if not isinstance(module, QuantLinear):
                continue

            if name.startswith("mlp.experts.expert_modules."):
                parts = name.split(".")
                if len(parts) != 5:
                    raise ValueError(f"Unexpected Qwen3-Next expert module name: {name}")
                _, _, _, expert_idx, proj_name = parts
                yield f"{weight_prefix}mlp.experts.{expert_idx}.{proj_name}.weight", module
                continue

            yield f"{weight_prefix}{name}.weight", module

    def load_unit_inputs(self, data_dir, unit: PtqUnit):
        return super().load_unit_inputs(data_dir, unit)
