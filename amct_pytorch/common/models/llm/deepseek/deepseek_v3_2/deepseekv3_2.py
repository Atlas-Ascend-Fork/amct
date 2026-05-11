import os
import json
from pathlib import Path
import torch
from loguru import logger
from safetensors import safe_open
from tqdm import tqdm
from amct_pytorch.common.models import MODEL_REGISTRY
from amct_pytorch.common.models.llm.common.base import BaseModel
from amct_pytorch.common.models.llm.common.quant_apply import apply_quant_to_attn, apply_quant_to_moe_mlp
from amct_pytorch.common.models.llm.deepseek.deepseek_v3_2.modeling.modeling_deepseek_v3_2 import DeepseekV3DecoderLayer
from amct_pytorch.common.models.llm.deepseek.deepseek_v3_2.modeling.indexer import ModelArgs
from amct_pytorch.common.models.llm.deepseek.deepseek_v3_2.quant_module import QuantDeepseekV3MLP, QuantDSA


@MODEL_REGISTRY.register(
    name="deepseekv32",
    task="llm",
    family="deepseek",
    description="DeepSeek V3.2 model adapter",
)
class DeepseekV32(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.quant_target = args.quant_target
        self.local_args()
        self.init_cls()
        self.cls = DeepseekV3DecoderLayer
        self.model = self.empty_weights_model()
        self.num_layers = self.args.model_args.n_layers

    def local_args(self):
        config = Path(__file__).parent / "modeling" / "config_671B_v3.2.json"
        with open(config) as f:
            model_args = ModelArgs(**json.load(f))
        self.args.model_args = model_args

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
        if not hasattr(self.config, 'model_args'):
            self.config.model_args = self.args.model_args
        decoder_layer = self.cls(self.config, layer_idx, is_nextn=layer_idx == 61)
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
            self.apply_quant_moe_mlp(decoder_layer)
        if "mlp" in self.quant_target:
            raise ValueError("Deepseek V32 is a moe model and does not support quant_target='mlp'.")

        return decoder_layer

    def apply_quant_attn(self, decoder_layer):
        return apply_quant_to_attn(self.args, decoder_layer, QuantDSA)

    def apply_quant_moe_mlp(self, decoder_layer):
        return apply_quant_to_moe_mlp(
            self.args,
            decoder_layer,
            cls=QuantDeepseekV3MLP,
        )

    def iter_ptq_units(self, layer_idx, block):
        yield from super().iter_ptq_units(layer_idx, block)

    def iter_deploy_bindings(self, layer_idx, block):
        yield from super().iter_deploy_bindings(layer_idx, block)

    def load_unit_inputs(self, data_dir, unit):
        return super().load_unit_inputs(data_dir, unit)
