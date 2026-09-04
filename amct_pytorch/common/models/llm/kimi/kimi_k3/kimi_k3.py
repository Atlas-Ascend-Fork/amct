# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

import gc
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import torch
from accelerate import init_empty_weights
from safetensors import safe_open
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

from amct_pytorch.common.models import MODEL_REGISTRY
from amct_pytorch.common.models.llm.common.base import BaseModel
from amct_pytorch.common.models.llm.common.capture import register_forward_hooks
from amct_pytorch.common.models.llm.common.ptq_params import PtqParamHandler, PtqParamStore
from amct_pytorch.common.models.llm.kimi.kimi_k3.modeling.configuration_kimi_k3 import (
    KimiK3Config,
    KimiLinearConfig,
)
from amct_pytorch.quantization.modules.quant_linear import QuantLinear


# Temporary eval diagnostics; remove after locating the performance bottleneck.
def _perf_sync(device):
    device = torch.device(device)
    if device.type == "npu":
        torch.npu.synchronize(device)
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def _attach_perf_timer(module, name, device, handles):
    state = {"count": 0}

    def before(_module, _args):
        if state["count"] < 2:
            print(f"[PERF] BEGIN {name} sample={state['count']}", flush=True)
            _perf_sync(device)
            state["start"] = time.perf_counter()

    def after(_module, _args, _output):
        if state["count"] < 2:
            _perf_sync(device)
            elapsed = time.perf_counter() - state["start"]
            print(f"[PERF] END {name} sample={state['count']}: {elapsed:.3f}s", flush=True)
        state["count"] += 1

    handles.append(module.register_forward_pre_hook(before))
    handles.append(module.register_forward_hook(after))


@MODEL_REGISTRY.register(
    name="kimi_k3",
    task="llm",
    family="kimi",
    description="Kimi K3 placeholder adapter",
)
class KimiK3(BaseModel):
    """Kimi K3 adapter with text-only BF16 eval and tensor-wise deploy."""

    _MOE_WEIGHT_RE = re.compile(
        r"^language_model\.model\.layers\.\d+\.block_sparse_moe\.experts\.\d+\.w[123]\.weight_packed$"
    )

    def __init__(self, args):
        if getattr(args, "granularity", None) == "tensor":
            self.args = args
            self.model_path = args.model
            self.quant_target = getattr(args, "quant_target", [])
            self.quant_dtype = getattr(args, "quant_dtype", "")
            self._weight_map = None
            return

        self._register_kimi_config()
        self.args = args
        self.quant_target = args.quant_target
        self.quant_dtype = args.quant_dtype
        self.position_ids = None
        self.attention_mask = None
        self.position_embeddings = None
        self.input_ids = None
        self._block_residuals = None
        self.model_path = self.args.model
        self.config = KimiK3Config.from_pretrained(self.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True)
        self.ptq_param_handler = PtqParamHandler()
        self.ptq_param_store = PtqParamStore(self.ptq_param_handler, self.iter_ptq_units)
        self.kimi_config = self.config
        self.text_config = self.kimi_config.text_config
        self.config = self.text_config
        self.num_layers = self.config.num_hidden_layers
        self.cls = self._load_text_modeling().KimiDecoderLayer
        self.model = self.empty_weights_model()
        self._weight_map = None

    @staticmethod
    def block_size(weight):
        return 32

    @staticmethod
    def _register_kimi_config():
        AutoConfig.register("kimi_k3", KimiK3Config, exist_ok=True)
        AutoConfig.register("kimi_linear", KimiLinearConfig, exist_ok=True)

    @staticmethod
    def _load_text_modeling():
        from amct_pytorch.common.models.llm.kimi.kimi_k3.modeling import modeling_kimi
        return modeling_kimi

    @staticmethod
    def _is_kimi_moe_weight_packed(weight_name: str) -> bool:
        return bool(KimiK3._MOE_WEIGHT_RE.match(weight_name))

    def _load_weight_map(self):
        if self._weight_map is not None:
            return self._weight_map

        index_path = Path(self.model_path) / "model.safetensors.index.json"
        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)
        self._weight_map = dict(index.get("weight_map", {}))
        return self._weight_map

    def get_layer_weight_prefix(self, layer_idx: int) -> str:
        return f"language_model.model.layers.{layer_idx}."

    def _embed_base_prefix(self) -> str:
        return "language_model.model."

    def get_embed_load_specs(self):
        specs = [
            (self.model.model.embed_tokens, "language_model.model.embed_tokens."),
            (self.model.model.norm, "language_model.model.norm."),
            (self.model.lm_head, "language_model.lm_head."),
        ]
        if hasattr(self.model.model, "output_attn_res_norm"):
            specs.extend([
                (self.model.model.output_attn_res_norm, "language_model.model.output_attn_res_norm."),
                (self.model.model.output_attn_res_proj, "language_model.model.output_attn_res_proj."),
            ])
        return specs

    def float_model(self):
        text_modeling = self._load_text_modeling()
        model = text_modeling.KimiLinearForCausalLM(self.text_config)
        model.load_state_dict(self._load_text_state_dict(), strict=True)
        return model.bfloat16()

    def empty_weights_model(self):
        text_modeling = self._load_text_modeling()
        with init_empty_weights():
            return text_modeling.KimiLinearForCausalLM(self.text_config)

    def _load_text_state_dict(self):
        return self.load_layer_weight("language_model.")

    def do_embedding_forward(self, samples, dtype=torch.bfloat16, hook_name=None):
        if hook_name is not None:
            raise NotImplementedError("Kimi K3 PTQ data extraction is not implemented yet.")

        outs = []
        block_residuals = []
        self.load_embed_state_dict()
        # The legacy eval caller passes output_dir as the second argument;
        # keep BF16 explicit instead of using the dtype argument here.
        embed_tokens = self.model.model.embed_tokens.to(
            device=self.args.device, dtype=torch.bfloat16,
        )
        use_attn_residuals = bool(getattr(self.model.model, "use_attn_residuals", False))

        with torch.no_grad():
            for inputs in tqdm(samples, total=len(samples), desc="Embedding Processing..."):
                hidden_states = embed_tokens(inputs.to(self.args.device))
                outs.append(hidden_states.to("cpu"))
                if use_attn_residuals:
                    batch_size, seq_len, hidden_size = hidden_states.shape
                    block_residuals.append(
                        hidden_states.new_zeros(batch_size * seq_len, 0, hidden_size).to("cpu")
                    )

        self.input_ids = list(samples)
        self.position_ids = None
        self.position_embeddings = None
        self.attention_mask = None
        self._block_residuals = block_residuals if use_attn_residuals else None
        return outs

    def _build_layer_forward_kwargs(self, block, sample):
        cache_position = torch.arange(0, sample.shape[1], device=sample.device)
        position_ids = cache_position.unsqueeze(0)
        attention_mask = None

        if not getattr(block, "is_linear_attn", False):
            text_modeling = self._load_text_modeling()
            attention_mask = text_modeling.create_causal_mask(
                config=self.config,
                input_embeds=sample,
                attention_mask=None,
                cache_position=cache_position,
                past_key_values=None,
                position_ids=position_ids,
            )
        elif hasattr(self.model.model, "_update_linear_attn_mask"):
            attention_mask = self.model.model._update_linear_attn_mask(None, cache_position)

        return {
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "cache_position": cache_position,
        }

    def do_block_forward(self, layer_idx, samples, hook_name=None, use_quant_block=False, enable_quant=False):
        act_stat = {}
        outs, hooks = [], []
        block = self._build_block_for_forward(layer_idx, use_quant_block=use_quant_block)
        if use_quant_block:
            from amct_pytorch.common.models.llm.common.quant_apply import (
                set_model_act_quant_state,
                set_model_weight_quant_state,
            )
            set_model_weight_quant_state(block, enable_quant)
            set_model_act_quant_state(block, enable_quant)

        block = self._dispatch_block(block).eval()

        perf_handles = []
        module = getattr(block, "block_sparse_moe", None)
        if module is not None:
            _attach_perf_timer(module, f"layer={layer_idx} block_sparse_moe", self.args.device, perf_handles)
        if use_quant_block and hook_name is None:
            for mod in block.modules():
                if isinstance(mod, QuantLinear):
                    mod.eval_mode = True
                    mod.cached_eval_weight = None
        if hook_name is not None:
            register_forward_hooks(block, hook_name, hooks, act_stat)

        use_attn_residuals = bool(getattr(block, "use_attn_residuals", False))
        if use_attn_residuals and self._block_residuals is None:
            raise RuntimeError("Kimi K3 block residuals are not initialized; run do_embedding_forward first.")

        next_residuals = []
        residuals = self._block_residuals if use_attn_residuals else [None] * len(samples)
        with torch.no_grad():
            for sample, block_residual in zip(samples, residuals):
                sample = sample.to(self.args.device)
                call_kwargs = self._build_layer_forward_kwargs(block, sample)
                if use_attn_residuals:
                    call_kwargs["block_residual"] = block_residual.to(self.args.device)
                out = block(sample, **call_kwargs)
                if use_attn_residuals:
                    out, block_residual = out
                    next_residuals.append(block_residual.to("cpu"))
                else:
                    out = out[0] if isinstance(out, (tuple, list)) else out
                outs.append(out.to("cpu"))

        for handle in perf_handles:
            handle.remove()
        if use_attn_residuals:
            self._block_residuals = next_residuals

        if hook_name is not None:
            for hook in hooks:
                hook.remove()
            self.save_block_hook_inputs(act_stat, hook_name, layer_idx)
        try:
            from accelerate.hooks import remove_hook_from_module
            remove_hook_from_module(block, recurse=True)
        except Exception:
            pass
        block = None
        gc.collect()
        if hasattr(torch, "npu"):
            torch.npu.empty_cache()
        return outs

    def do_head_forward(self, inps):
        device = self.args.device
        text_model = self.model.model
        text_model.norm.to(device)
        self.model.lm_head.to(device)
        if getattr(text_model, "use_attn_residuals", False):
            text_model.output_attn_res_norm.to(device)
            text_model.output_attn_res_proj.to(device)
            if self._block_residuals is None:
                raise RuntimeError("Kimi K3 block residuals are not initialized; run block forward first.")

        preds = []
        residuals = self._block_residuals if getattr(text_model, "use_attn_residuals", False) else [None] * len(inps)
        with torch.no_grad():
            for inp, block_residual in tqdm(zip(inps, residuals), total=len(inps), desc="Head Processing..."):
                inp = inp.to(device)
                if block_residual is not None:
                    inp = text_model._apply_output_attn_res(inp, block_residual.to(device))
                out = text_model.norm(inp)
                out = self.model.lm_head(out)[:, :-1, :].contiguous()
                preds.append(out.to("cpu"))
        return preds

    def load_layer_weight(self, prefix):
        weight_map = self._load_weight_map()
        raw_tensors = {}
        weights_by_file = defaultdict(list)
        for weight_name in weight_map:
            if not weight_name.startswith(prefix):
                continue
            weights_by_file[weight_map[weight_name]].append(weight_name)

        for file_name, weight_names in weights_by_file.items():
            with safe_open(str(Path(self.model_path) / file_name), framework="pt", device="cpu") as f:
                for weight_name in weight_names:
                    raw_tensors[weight_name] = f.get_tensor(weight_name)

        state_dict = {}
        for weight_name, tensor in raw_tensors.items():
            if weight_name.endswith(".weight_packed") or weight_name.endswith(".weight_scale"):
                continue
            state_dict[weight_name[len(prefix):]] = tensor

        a_log_key = "self_attn.A_log"
        a_log = state_dict.get(a_log_key)
        if a_log is not None:
            linear_attn_config = self.config.linear_attn_config
            num_heads = linear_attn_config["num_heads"]
            head_dim = linear_attn_config["head_dim"]
            if a_log.shape == (head_dim,):
                padding = a_log[num_heads:]
                if torch.count_nonzero(padding).item() != 0:
                    raise RuntimeError("Kimi K3 self_attn.A_log contains non-zero padding")
                state_dict[a_log_key] = a_log[:num_heads].contiguous()
        return state_dict

    def get_scale_name(self, weight_name):
        if weight_name.endswith(".weight_packed"):
            return ".weight_scale", weight_name.replace(".weight_packed", ".weight_scale")
        if weight_name.endswith(".weight"):
            return ".weight_scale", weight_name.replace(".weight", ".weight_scale")
        return ".weight_scale", f"{weight_name}.weight_scale"

    def tensorwise_output_weight_name(self, weight_name):
        if weight_name.endswith(".weight_packed"):
            return weight_name.replace(".weight_packed", ".weight")
        return weight_name

    def generate_tensorwise_quant_layers(self):
        if self.quant_target and "moe" not in self.quant_target:
            return {}
        quant_layers = {}
        for weight_name in self._load_weight_map():
            if not self._is_kimi_moe_weight_packed(weight_name):
                continue
            output_name = self.tensorwise_output_weight_name(weight_name)
            quant_layers[output_name.rsplit(".", 1)[0]] = 4
        return quant_layers

    def generate_tensorwise_ignore_layers(self):
        return []

    def bits_scheme(self):
        return [
            {"targets": ["Linear"], "w_bits": 8, "a_bits": 8},
            {"targets": ["MoEGMM"], "w_bits": 4, "a_bits": 8},
        ]
