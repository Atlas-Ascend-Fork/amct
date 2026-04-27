import os
import torch
from abc import ABCMeta
from tqdm import tqdm
from accelerate import init_empty_weights
from loguru import logger
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from safetensors import safe_open
from compressed_tensors.utils.safetensors_load import get_weight_mappings
from amct_pytorch.common.datasets.ptq_io import save_ptq_inps, save_ptq_kwargs, load_ptq_inps
from amct_pytorch.common.models.llm.common.capture import Catcher, register_forward_hooks
from amct_pytorch.common.models.llm.common.ptq_params import PtqParamHandler, PtqParamStore
from amct_pytorch.common.models.llm.common.ptq_units import iter_indexed_units, make_ptq_unit, PtqUnit




class BaseModel(metaclass=ABCMeta):
    def __init__(self, args):
        self.args = args
        self.quant_target = args.quant_target
        self.position_ids = None
        self.attention_mask = None
        self.position_embeddings = None
        self.model_path = self.args.model
        self.config = AutoConfig.from_pretrained(
            self.model_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True)
        self.ptq_param_handler = PtqParamHandler()
        self.ptq_param_store = PtqParamStore(self.ptq_param_handler, self.iter_ptq_units)

    def float_model(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16)
        return model

    def empty_weights_model(self):
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(
                self.config,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16)
        return model


    def init_cls(self):
        pass

    def _embed_base_prefix(self) -> str:
        """Checkpoint-side public prefix for `embed_tokens` / `norm`.

        Override in adapters whose checkpoint nests the text backbone under a
        multimodal wrapper (e.g. return `"model.language_model."`).
        """
        return "model."

    def get_embed_load_specs(self):
        """Default (embed_tokens, norm, lm_head) load specs.

        The `lm_head` prefix is chosen from `config.tie_word_embeddings`:
        - tied:   read from `<base>embed_tokens.` (lm_head shares weight with
                  embed_tokens; the checkpoint has no separate `lm_head.weight`).
        - untied: read from `lm_head.`.
        Falls back to untied when the field is missing (a strict-load mismatch
        is safer than silently loading the wrong tensor).
        Adapters that need extra entries can extend `super().get_embed_load_specs()`.
        """
        base = self._embed_base_prefix()
        tie = bool(getattr(self.config, "tie_word_embeddings", False))
        lm_head_prefix = f"{base}embed_tokens." if tie else "lm_head."
        return [
            (self.model.model.embed_tokens, f"{base}embed_tokens."),
            (self.model.model.norm,         f"{base}norm."),
            (self.model.lm_head,            lm_head_prefix),
        ]

    def load_embed_state_dict(self):
        for module, prefix in self.get_embed_load_specs():
            module.to_empty(device="cpu")
            module.load_state_dict(self.load_layer_weight(prefix), strict=True)


    def load_layer_weight(self, prefix):
        weight_map = getattr(self, "_weight_map", None)
        if weight_map is None:
            weight_map = get_weight_mappings(self.model_path)
        # Find files containing this layer's weights
        file_list = set()
        for weight_name, file_name in weight_map.items():
            if weight_name.startswith(prefix):
                file_list.add(file_name)
        # Load weights from files
        state_dict = {}
        for file_path in file_list:
            full_path = os.path.join(self.model_path, file_path)
            with safe_open(full_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.startswith(prefix):
                        # Remove prefix from key
                        new_key = key[len(prefix):]
                        state_dict[new_key] = f.get_tensor(key)
        # Remove rotary_emb.inv_freq if present (not needed for inference)
        state_dict.pop('self_attn.rotary_emb.inv_freq', None)
        return state_dict


    def block(self, layer_idx):
        decoder_layer = self.cls(self.config, layer_idx)
        state_dict = self.load_layer_weight(self.get_layer_weight_prefix(layer_idx))
        decoder_layer.load_state_dict(state_dict, strict=True)
        decoder_layer.eval().bfloat16()
        return decoder_layer

    def _build_block_for_forward(self, layer_idx, use_quant_block=False):
        if use_quant_block:
            block = self.build_quant_block(layer_idx)
            self.load_selected_layer_ptq_params(layer_idx, block, strict=False)
            return block
        return self.block(layer_idx)

    def get_block_forward_kwargs(self):
        kwargs = {}
        if self.position_ids is not None:
            kwargs["position_ids"] = self.position_ids.to(self.args.device)
        if self.position_embeddings is not None:
            kwargs["position_embeddings"] = (
                self.position_embeddings[0].to(self.args.device),
                self.position_embeddings[1].to(self.args.device),
            )
        if self.attention_mask is not None:
            kwargs["attention_mask"] = self.attention_mask.to(self.args.device)
        return kwargs

    def register_block_forward_hooks(self, block, hook_name, hooks, act_stat):
        if hook_name is None:
            return
        register_forward_hooks(block, hook_name, hooks, act_stat)

    def save_block_hook_inputs(self, act_stat, hook_name, layer_idx):
        if hook_name is None:
            raise ValueError(f"hook_name cannot be None")
        save_target = (
            "attn"
            if "attn-linear" in self.quant_target or "attn-cache" in self.quant_target
            else self.quant_target[0]
        )
        save_ptq_inps(act_stat, hook_name, save_target, layer_idx, self.args.data_dir)

    def do_embedding_forward(self, samples, dtype=torch.bfloat16, hook_name=None):
        outs = []
        self.load_embed_state_dict()
        layers = self.model.model.layers
        layers[0] = self.block(0)
        layers[0] = layers[0].bfloat16()
        layers[0] = Catcher(layers[0], outs)
        with torch.no_grad():
            # Loop through each batch
            for bs, inputs in tqdm(enumerate(samples), total=len(samples), desc=f"Embedding Processing..."):
                try:
                    self.model(inputs)
                except ValueError:
                    pass
        self.position_ids = layers[0].position_ids
        self.position_embeddings = layers[0].position_embeddings
        self.attention_mask = layers[0].attention_mask
        if hook_name is not None:
            save_ptq_kwargs(self.position_ids, self.position_embeddings, self.attention_mask, self.args.data_dir)
        layers[0] = layers[0].module
        return outs

    def do_head_forward(self, inps):
        self.model.model.norm.to(self.args.device)
        self.model.lm_head.to(self.args.device)
        preds = []
        with torch.no_grad():
            for idx, inp in tqdm(enumerate(inps), total=len(inps), desc='Head Processing...'):
                inp = inp.to(self.args.device)
                out = self.model.model.norm(inp)
                out = self.model.lm_head(out)[:, :-1, :].contiguous()
                preds.append(out.to('cpu'))
        return preds

    def build_quant_block(self, layer_idx):
        return self.block(layer_idx)

    def do_block_forward(self, layer_idx, samples, hook_name=None, use_quant_block=False, enable_quant=False):
        act_stat = {}
        outs, hooks = [], []
        block = self._build_block_for_forward(layer_idx, use_quant_block=use_quant_block)
        if use_quant_block:
            from amct_pytorch.common.models.llm.common.quant_apply import set_model_act_quant_state, set_model_weight_quant_state
            set_model_weight_quant_state(block, enable_quant)
            set_model_act_quant_state(block, enable_quant)

        block = block.to(self.args.device).eval()
        if use_quant_block and hook_name is None:
            from amct_pytorch.quantization.modules.quant_linear import QuantLinear
            for mod in block.modules():
                if isinstance(mod, QuantLinear):
                    mod._eval_mode = True
                    mod._cached_eval_weight = None
        if hook_name is not None:
            self.register_block_forward_hooks(block, hook_name, hooks, act_stat)

        block_kwargs = self.get_block_forward_kwargs()
        with torch.no_grad():
            for sample in samples:
                sample = sample.to(self.args.device)
                out = block(sample, **block_kwargs)
                out = out[0] if isinstance(out, (tuple, list)) else out
                outs.append(out.to("cpu"))

        if hook_name is not None:
            for hook in hooks:
                hook.remove()
            self.save_block_hook_inputs(act_stat, hook_name, layer_idx)
        return outs

    def get_layer_weight_prefix(self, layer_idx: int) -> str:
        pass

    def iter_ptq_units(self, layer_idx, block):
        if "attn-linear" in self.quant_target or "attn-cache" in self.quant_target:
            layer_type = getattr(block, "layer_type", None)
            attn_name = "linear_attn" if layer_type == "linear_attention" else "self_attn"
            quant_attn = getattr(block, attn_name, None)
            yield make_ptq_unit("attn", attn_name, layer_idx, quant_attn)
            return

        mlp = getattr(block, "mlp", None)
        if mlp is None:
            raise ValueError(f"Unsupported quant target '{self.quant_target}'.")
        if "moe" in self.quant_target:
            experts = getattr(mlp, "experts", [])
            if hasattr(experts, "iter_ptq_expert_modules"):
                experts = experts.iter_ptq_expert_modules()
            elif hasattr(experts, "expert_modules"):
                experts = experts.expert_modules
            yield from iter_indexed_units(
                kind="moe",
                name_prefix="expert",
                layer_idx=layer_idx,
                items=experts,
                metadata_fn=lambda expert_idx, _: {"expert_idx": expert_idx},
            )
            return
        if "mlp" in self.quant_target:
            yield make_ptq_unit("mlp", "mlp", layer_idx, mlp)
            return

    def iter_deploy_bindings(self, layer_idx, block):
        from amct_pytorch.quantization.modules.quant_linear import QuantLinear

        weight_prefix = self.get_layer_weight_prefix(layer_idx)
        for name, module in block.named_modules():
            if not isinstance(module, QuantLinear):
                continue
            yield f"{weight_prefix}{name}.weight", module

    def load_unit_inputs(self, data_dir, unit: PtqUnit):
        return load_ptq_inps(data_dir, unit.kind, unit.layer_idx)

    def load_layer_ptq_params(self, layer_idx: int, block, param_dir: str, strict: bool = False):
        return self.ptq_param_store.load_layer(layer_idx, block, param_dir, strict=strict)

    def load_selected_layer_ptq_params(self, layer_idx: int, block, strict: bool = False):
        results = {}
        quant_targets = self.quant_target if isinstance(self.quant_target, (list, tuple, set)) else [self.quant_target]

        target_specs = []
        if "attn-linear" in quant_targets:
            target_specs.append(("attn-linear", self.args.attn_linear_param_dir))
        if "attn-cache" in quant_targets:
            target_specs.append(("attn-cache", self.args.attn_cache_param_dir))
        if "mlp" in quant_targets:
            target_specs.append(("mlp", self.args.moe_mlp_param_dir))
        if "moe" in quant_targets:
            target_specs.append(("moe", self.args.moe_mlp_param_dir))

        original_quant_target = self.quant_target
        try:
            for target_name, param_dir in target_specs:
                if not param_dir:
                    logger.warning(
                        "No PTQ param dir specified for target '{}' at layer {}; fallback to direct quant for this target.",
                        target_name,
                        layer_idx,
                    )
                    results[target_name] = {"loaded": [], "missing": []}
                    continue

                self.quant_target = [target_name]
                result = self.load_layer_ptq_params(layer_idx, block, param_dir, strict=strict)
                results[target_name] = result

                if result["loaded"] and result["missing"]:
                    logger.warning(
                        "Partially loaded PTQ params for layer {} target '{}' from '{}'; missing units: {}.",
                        layer_idx,
                        target_name,
                        param_dir,
                        ", ".join(result["missing"]),
                    )
                elif not result["loaded"]:
                    logger.warning(
                        "No PTQ params loaded for layer {} target '{}' from '{}'; fallback to direct quant for this target.",
                        layer_idx,
                        target_name,
                        param_dir,
                    )
        finally:
            self.quant_target = original_quant_target
        return results
