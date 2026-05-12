import torch
import torch.nn as nn
from loguru import logger
from transformers import AutoModelForCausalLM


from amct_pytorch.common.models.llm.common.attention_forward import scaled_dot_product_attention
from amct_pytorch.common.models.llm.common.base import BaseModel, PtqUnit
from amct_pytorch.common.models.llm.common.ptq_units import make_ptq_unit
from amct_pytorch.common.models.llm.common.quant_apply import (
    apply_quant_to_attn,
    apply_quant_to_moe_mlp,
    QuantGatedMLP,
)
from amct_pytorch.common.models import MODEL_REGISTRY
from amct_pytorch.common.models.llm.pangu.pangu_7b.quant_module import _ensure_transformers_compat, _disable_npu_fused_attention, QuantPanguAttn, QuantPanguMLP


@MODEL_REGISTRY.register(
    name="pangu_7b",
    task="llm",
    family="pangu",
    description="openPangu Embedded 7B model adapter",
)
class Pangu7B(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.quant_target = args.quant_target
        self._update_config()
        _ensure_transformers_compat()
        self.model = self.empty_weights_model()
        _disable_npu_fused_attention(self.model)
        self.cls = type(self.model.model.layers[0])
        self.num_layers = self.config.num_hidden_layers
        self.parse_quant_mode()

    def parse_quant_mode(self):
        if "moe" in self.quant_target:
            raise ValueError("pangu_7b is a dense model and does not support quant_target='moe'.")

    def _update_config(self):
        self.config._attn_implementation = "eager"

    def float_model(self):
        _ensure_transformers_compat()
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            config=self.config,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        _disable_npu_fused_attention(model)
        return model

    def empty_weights_model(self):
        return super().empty_weights_model()

    def get_layer_weight_prefix(self, layer_idx: int) -> str:
        return f"model.layers.{layer_idx}."

    def load_layer_weight(self, prefix):
        return super().load_layer_weight(prefix)

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
            apply_quant_to_attn(self.args, decoder_layer, QuantPanguAttn)
        if "mlp" in self.quant_target:
            apply_quant_to_moe_mlp(self.args, decoder_layer, cls=QuantPanguMLP)
        return decoder_layer

    def iter_ptq_units(self, layer_idx, block):
        if "attn-linear" in self.quant_target or "attn-cache" in self.quant_target:
            yield make_ptq_unit("attn", "self_attn", layer_idx, block.self_attn)
            return
        if "mlp" in self.quant_target:
            yield make_ptq_unit("mlp", "mlp", layer_idx, block.mlp)
            return

    def iter_deploy_bindings(self, layer_idx, block):
        yield from super().iter_deploy_bindings(layer_idx, block)

    def load_unit_inputs(self, data_dir, unit: PtqUnit):
        return super().load_unit_inputs(data_dir, unit)
