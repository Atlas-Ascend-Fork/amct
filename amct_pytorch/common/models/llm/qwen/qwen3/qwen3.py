import torch
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3Config,
    Qwen3DecoderLayer,
    Qwen3MLP,
)

from amct_pytorch.common.models.llm.common.base import BaseModel, PtqUnit
from amct_pytorch.common.models.llm.common.quant_apply import apply_quant_to_attn, apply_quant_to_moe_mlp
from amct_pytorch.common.models.llm.qwen.qwen3.quant_module import QuantQwen3Attn, QuantQwen3MLP
from amct_pytorch.common.models import MODEL_REGISTRY


@MODEL_REGISTRY.register(
    name="qwen3",
    task="llm",
    family="qwen",
    description="Qwen3 dense model adapter",
)
class Qwen3(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.textconfig = Qwen3Config
        self.num_layers = self.config.num_hidden_layers
        self.cls = Qwen3DecoderLayer
        self.model = self.empty_weights_model()
        self.parse_quant_mode()

    def parse_quant_mode(self):
        if "moe" in self.quant_target:
            raise ValueError("Qwen3 dense model does not support quant_target='moe'.")

    def float_model(self):
        return super().float_model()

    def empty_weights_model(self):
        return super().empty_weights_model()

    def get_layer_weight_prefix(self, layer_idx: int) -> str:
        return f"model.layers.{layer_idx}."

    def load_layer_weight(self, prefix):
        return super().load_layer_weight(prefix)

    def load_embed_state_dict(self):
        return super().load_embed_state_dict()

    def block(self, layer_idx):
        return super().block(layer_idx)

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
            apply_quant_to_attn(self.args, decoder_layer, QuantQwen3Attn)
        if "mlp" in self.quant_target:
            apply_quant_to_moe_mlp(self.args, decoder_layer, cls=QuantQwen3MLP)
        return decoder_layer

    def iter_ptq_units(self, layer_idx, block):
        yield from super().iter_ptq_units(layer_idx, block)

    def iter_deploy_bindings(self, layer_idx, block):
        yield from super().iter_deploy_bindings(layer_idx, block)

    def load_unit_inputs(self, data_dir, unit: PtqUnit):
        return super().load_unit_inputs(data_dir, unit)
