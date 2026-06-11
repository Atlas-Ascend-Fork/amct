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

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import init_empty_weights, dispatch_model
from accelerate.hooks import AlignDevicesHook, add_hook_to_module
from accelerate.utils import set_module_tensor_to_device
from loguru import logger
from safetensors import safe_open
from transformers import AutoModelForCausalLM
from compressed_tensors.utils.safetensors_load import get_weight_mappings
from tqdm import tqdm

from amct_pytorch.common.models import MODEL_REGISTRY
from amct_pytorch.common.models.llm.common.base import BaseModel
from amct_pytorch.common.models.llm.common.capture import Catcher
from amct_pytorch.common.models.llm.common.quant_apply import apply_quant_to_moe_mlp
from amct_pytorch.common.models.llm.common.ptq_units import iter_indexed_units, make_ptq_unit
from amct_pytorch.common.models.llm.deepseek.deepseek_v4.modeling.configuration_deepseek_v4 import DeepseekV4Config
from amct_pytorch.common.models.llm.deepseek.deepseek_v4.modeling.modeling_deepseek_v4 import Block
from amct_pytorch.common.models.llm.deepseek.deepseek_v4.quant_module import (
    QuantV4Attention,
    QuantV4Expert,
)


class _NoMoveAlignDevicesHook(AlignDevicesHook):
    def init_hook(self, module):
        return module


@MODEL_REGISTRY.register(
    name="deepseek_v4",
    task="llm",
    family="deepseek",
    description="DeepSeek V4 model adapter",
)
class DeepseekV4(BaseModel):
    attn_norm_name = "attn_norm"
    ffn_norm_name = "ffn_norm"

    def __init__(self, args):
        super().__init__(args)
        if not isinstance(self.config, DeepseekV4Config):
            self.config = DeepseekV4Config(**self.config.to_dict())
        ptq_seq = getattr(args, "seq_len", None)
        if ptq_seq and ptq_seq < self.config.max_seq_len:
            self.config.max_seq_len = ptq_seq
            self.config.max_position_embeddings = ptq_seq
        self.cls = Block
        self.model = self.empty_weights_model()
        self.num_layers = self.config.n_layers
        self.sharded_block = False

    @staticmethod
    def block_size(weight):
        bs = 32 if weight.dtype == torch.int8 else 128
        return bs

    @staticmethod
    def _resolve_param_device(local_name, device_map):
        best_path = None
        for path in device_map:
            if path and (local_name == path or local_name.startswith(path + ".")):
                if best_path is None or len(path) > len(best_path):
                    best_path = path
        if best_path is not None:
            return device_map[best_path]
        return device_map.get("", "cpu")

    @staticmethod
    def _retag_descendant_hooks(root, device_map):
        sorted_entries = sorted(
            ((p, d) for p, d in device_map.items() if p),
            key=lambda kv: kv[0].count("."),
        )
        for path, dev in sorted_entries:
            target = root
            for part in path.split("."):
                target = getattr(target, part)
            for sub in target.modules():
                hook = getattr(sub, "_hf_hook", None)
                if hook is None:
                    continue
                hook.execution_device = dev
                if sub is target:
                    hook.io_same_device = True

    def apply_quant_attn(self, decoder_layer):
        decoder_layer.attn = QuantV4Attention(self.args, decoder_layer.attn)
        return decoder_layer

    def apply_quant_moe_mlp(self, decoder_layer):
        return apply_quant_to_moe_mlp(
            self.args,
            decoder_layer.ffn,
            cls=QuantV4Expert,
        )

    def block(self, layer_idx):
        """Dispatch: sharded multi-NPU load for eval, CPU staging (via
        BaseModel) for PTQ. PTQ needs CPU staging so it can move one expert
        at a time onto the NPU during per-MLP learning; the sharded path
        pins every expert to a fixed NPU via AlignDevicesHook, which fights
        PTQ's ``.to(device)`` / ``.cpu()`` cycle.
        """
        if self.sharded_block:
            return self._block_sharded(layer_idx)
        return super().block(layer_idx)

    def build_quant_block(self, layer_idx):
        decoder_layer = self.block(layer_idx)
        if "attn-linear" in self.quant_target or "attn-cache" in self.quant_target:
            self.apply_quant_attn(decoder_layer)
        if "moe" in self.quant_target:
            self.apply_quant_moe_mlp(decoder_layer)
        if "mlp" in self.quant_target:
            raise ValueError(
                "Deepseek V4 is a moe model and does not support quant_target='mlp'."
            )
        return decoder_layer

    def do_block_forward(self, layer_idx, samples, hook_name=None, use_quant_block=False, enable_quant=False):
        return super().do_block_forward(
            layer_idx,
            samples,
            hook_name=hook_name,
            use_quant_block=use_quant_block,
            enable_quant=enable_quant,
        )

    def do_embedding_forward(self, samples, dtype=torch.bfloat16, hook_name=None):
        outs = []
        self.load_embed_state_dict()

        layers = self.model.layers
        layers[0] = self.block(0).bfloat16()
        layers[0] = Catcher(layers[0], outs)
        embed = self.model.embed
        hc_mult = self.model.hc_mult

        with torch.no_grad():
            for inputs in tqdm(samples, total=len(samples), desc="Embedding Processing..."):
                try:
                    h = embed(inputs)
                    h = h.unsqueeze(2).repeat(1, 1, hc_mult, 1)
                    layers[0](h, start_pos=0, input_ids=inputs)
                except ValueError:
                    pass

        layers[0] = layers[0].module
        self.input_ids = list(samples)
        return outs

    def do_head_forward(self, inps):
        device = self.args.device
        head = self.model.head.to(device)
        norm = self.model.norm.to(device)
        for name in ("hc_head_fn", "hc_head_scale", "hc_head_base"):
            param = getattr(self.model, name)
            if param.is_meta:
                raise RuntimeError(
                    f"v4 top-level param '{name}' is still on meta; "
                    "load_embed_state_dict must run before do_head_forward."
                )
            setattr(self.model, name, nn.Parameter(param.to(device)))

        preds = []
        with torch.no_grad():
            for inp in tqdm(inps, total=len(inps), desc="Head Processing..."):
                inp = inp.to(device)  # [b, s, hc_mult, d]
                x = head.hc_head(
                    inp,
                    self.model.hc_head_fn,
                    self.model.hc_head_scale,
                    self.model.hc_head_base,
                )  # [b, s, d]
                x = norm(x)
                logits = F.linear(x.float(), head.weight)  # [b, s, vocab]
                preds.append(logits[:, :-1, :].contiguous().to("cpu"))
        return preds

    def empty_weights_model(self):
        with init_empty_weights(include_buffers=True):
            model = AutoModelForCausalLM.from_config(
                self.config,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            )
        from amct_pytorch.common.models.llm.deepseek.deepseek_v4.modeling.modeling_deepseek_v4 import (
            precompute_freqs_cis)
        precompute_freqs_cis.cache_clear()
        return model

    def get_embed_load_specs(self):
        tie = bool(getattr(self.config, "tie_word_embeddings", False))
        head_prefix = "embed." if tie else "head."
        return [
            (self.model.embed, "embed."),
            (self.model.norm, "norm."),
            (self.model.head, head_prefix),
        ]

    def get_layer_weight_prefix(self, layer_idx: int) -> str:
        return f"layers.{layer_idx}."

    def get_scale_name(self, weight_name):
        scale_prefix = ".scale"
        scale_inv_name = weight_name.replace('.weight', '.scale')
        return scale_prefix, scale_inv_name

    def iter_deploy_bindings(self, layer_idx, block):
        yield from super().iter_deploy_bindings(layer_idx, block)

    def iter_ptq_units(self, layer_idx, block):
        if "attn-linear" in self.quant_target or "attn-cache" in self.quant_target:
            attn = getattr(block, "attn", None)
            yield make_ptq_unit("attn", "attn", layer_idx, attn)
            return

        ffn = getattr(block, "ffn", None)
        if ffn is None:
            raise ValueError(f"Unsupported quant target '{self.quant_target}'.")
        if "moe" in self.quant_target:
            experts = getattr(ffn, "experts", [])
            yield from iter_indexed_units(
                kind="moe",
                name_prefix="expert",
                layer_idx=layer_idx,
                items=experts,
                metadata_fn=lambda expert_idx, _: {"expert_idx": expert_idx},
            )

    def load_embed_state_dict(self):
        super().load_embed_state_dict()
        self._load_top_level_hc_head_params()

    def cache_scheme(self):
        cache_type = "int" if self.quant_dtype == "int" else "float"
        scheme = {
        "kv_cache_scheme": {"num_bits": 8, "type": "float"},
        "li_cache_scheme": {
            "type": cache_type,
            "num_bits": 8,
        }}
        return scheme

    def _block_sharded(self, layer_idx):
        """Meta build + per-tensor direct send to target NPU. Avoids the
        ~35 GB CPU staging + 75s Module construction of BaseModel.block."""
        with init_empty_weights():
            block = self.cls(self.config, layer_idx)

        device_map = self._build_block_device_map(block)
        if device_map is None:
            device_map = {"": str(self.args.device)}

        weight_prefix = self.get_layer_weight_prefix(layer_idx)
        weight_map = self._weight_map_cached()
        expected = set(name for name, _ in block.named_parameters())
        expected.update(name for name, _ in block.named_buffers())

        file_to_keys = {}
        for full_name, file_name in weight_map.items():
            if not full_name.startswith(weight_prefix):
                continue
            local_name = full_name[len(weight_prefix):]
            if local_name not in expected:
                continue
            file_to_keys.setdefault(file_name, []).append((full_name, local_name))

        for file_name, keys in file_to_keys.items():
            full_path = os.path.join(self.model_path, file_name)
            with safe_open(full_path, framework="pt", device="cpu") as f:
                for full_name, local_name in keys:
                    tensor = f.get_tensor(full_name)
                    if tensor.is_floating_point() and tensor.dtype != torch.bfloat16:
                        tensor = tensor.to(torch.bfloat16)
                    target_dev = self._resolve_param_device(local_name, device_map)
                    set_module_tensor_to_device(block, local_name, target_dev, value=tensor)

        missing = [name for name, p in block.named_parameters() if p.is_meta]
        if missing:
            raise RuntimeError(
                f"Layer {layer_idx}: {len(missing)} param(s) still on meta after direct load "
                f"(e.g. {missing[:3]}); checkpoint key missing under '{weight_prefix}'."
            )

        for name, buf in list(block.named_buffers()):
            target_dev = self._resolve_param_device(name, device_map)
            if str(buf.device) == target_dev:
                continue
            set_module_tensor_to_device(block, name, target_dev, value=buf.to(target_dev))

        block.eval()
        return block

    def _build_block_device_map(self, block):
        """npu:0 = stem; npu:1 = attn proper; npu:2 = attn.indexer (if present);
        experts round-robin on remaining cards. Returns None for single-NPU."""
        n_npu = max(1, torch.npu.device_count())
        if n_npu <= 1:
            return None
        device_map = {"": "npu:0"}
        attn_dev = 1 % n_npu
        device_map["attn"] = f"npu:{attn_dev}"
        reserved = {attn_dev}
        indexer = getattr(block.attn, "indexer", None)
        if isinstance(indexer, torch.nn.Module):
            indexer_dev = 2 % n_npu
            device_map["attn.indexer"] = f"npu:{indexer_dev}"
            reserved.add(indexer_dev)
        expert_devices = [d for d in range(n_npu) if d not in reserved] or list(range(n_npu))
        for i in range(len(block.ffn.experts)):
            device_map[f"ffn.experts.{i}"] = f"npu:{expert_devices[i % len(expert_devices)]}"
        return device_map

    def _dispatch_block(self, module):
        device_map = self._build_block_device_map(module)
        if device_map is None:
            return super()._dispatch_block(module)
        for path, dev in sorted(device_map.items(), key=lambda kv: kv[0].count(".")):
            target = module if path == "" else module
            if path:
                for part in path.split("."):
                    target = getattr(target, part)
            hook = _NoMoveAlignDevicesHook(
                execution_device=dev,
                io_same_device=(path == ""),
                place_submodules=True,
            )
            add_hook_to_module(target, hook)
        self._retag_descendant_hooks(module, device_map)
        return module

    def _load_top_level_hc_head_params(self):
        weight_map = getattr(self, "_weight_map", None) or get_weight_mappings(self.model_path)
        self._weight_map = weight_map
        for name in ("hc_head_fn", "hc_head_scale", "hc_head_base"):
            key = next((k for k in (name, f"model.{name}") if k in weight_map), None)
            if key is None:
                logger.warning(
                    "Top-level v4 param '{}' not found in checkpoint weight_map; staying meta.",
                    name,
                )
                continue
            file_path = os.path.join(self.model_path, weight_map[key])
            with safe_open(file_path, framework="pt", device="cpu") as f:
                tensor = f.get_tensor(key)
            setattr(self.model, name, nn.Parameter(tensor.to(torch.float32)))

    def _weight_map_cached(self):
        wm = getattr(self, "_weight_map", None)
        if wm is None:
            wm = get_weight_mappings(self.model_path)
            self._weight_map = wm
        return wm