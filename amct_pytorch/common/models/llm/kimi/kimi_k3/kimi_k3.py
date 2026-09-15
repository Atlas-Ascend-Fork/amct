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
from collections import defaultdict
from pathlib import Path

import torch
from accelerate import init_empty_weights
from accelerate.hooks import AlignDevicesHook, add_hook_to_module
from accelerate.utils import set_module_tensor_to_device
from safetensors import safe_open
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

from amct_pytorch.common.models import MODEL_REGISTRY
from amct_pytorch.common.models.llm.common.base import BaseModel
from amct_pytorch.common.models.llm.common.capture import register_forward_hooks
from amct_pytorch.common.models.llm.common.ptq_params import (
    PtqParamHandler,
    PtqParamStore,
)
from amct_pytorch.common.models.llm.common.ptq_units import (
    iter_indexed_units,
    make_ptq_unit,
)
from amct_pytorch.common.models.llm.common.quant_apply import build_no_algo_args
from amct_pytorch.common.models.llm.kimi.kimi_k3.modeling.configuration_kimi_k3 import (
    KimiK3Config,
    KimiLinearConfig,
)
from amct_pytorch.common.models.llm.kimi.kimi_k3.quant_module import (
    QuantKimiLinear,
    QuantKimiMLAAttention,
    QuantKimiMLP,
    quantize_kimi_kda_linears,
)
from amct_pytorch.quantization.bit_policy import ensure_bit_policy
from amct_pytorch.quantization.modules.quant_linear import QuantLinear


class _NoMoveAlignDevicesHook(AlignDevicesHook):
    """Move module inputs/outputs without relocating pre-sharded weights."""

    def init_hook(self, module):
        return module


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
    _MLA_WEIGHT_RE = re.compile(
        r"^language_model\.model\.layers\.\d+\.self_attn\."
        r"(q_a_proj|q_b_proj|kv_a_proj_with_mqa)\.weight$"
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
        # Existing eval/extraction workflows enable this shared V4 switch.
        # K3 only uses it for streaming onto args.device, not multi-NPU sharding.
        # PTQ retains the CPU-staged BaseModel.block path.
        self.sharded_block = False

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
            ((path, device) for path, device in device_map.items() if path),
            key=lambda item: item[0].count("."),
        )
        for path, device in sorted_entries:
            target = root
            for part in path.split("."):
                target = getattr(target, part)
            for submodule in target.modules():
                hook = getattr(submodule, "_hf_hook", None)
                if hook is None:
                    continue
                hook.execution_device = device
                if submodule is target:
                    hook.io_same_device = True

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

    def block(self, layer_idx):
        if getattr(self, "sharded_block", False):
            return self._block_sharded(layer_idx)
        return super().block(layer_idx)

    def _block_sharded(self, layer_idx):
        """Build on meta and stream a floating checkpoint onto target NPUs.

        Unlike CPU staging, this never holds a full layer state_dict or a
        randomly initialized CPU copy of all experts. Routed experts are
        evenly distributed across visible NPUs; all other modules stay on the
        root device. Buffers stay real during construction so non-persistent
        derived values can be preserved.
        """
        with init_empty_weights(include_buffers=False):
            block = self.cls(self.config, layer_idx)

        device_map = self._build_block_device_map(block)
        if device_map is None:
            device_map = {"": str(self.args.device)}

        prefix = self.get_layer_weight_prefix(layer_idx)
        # state_dict includes persistent buffers but excludes derived buffers.
        expected = {name: (value.shape, value.dtype) for name, value in block.state_dict().items()}
        weights_by_file = defaultdict(list)
        present = set()
        for full_name, file_name in self._load_weight_map().items():
            if not full_name.startswith(prefix):
                continue
            local_name = full_name[len(prefix):]
            if local_name.endswith((".weight_packed", ".weight_scale")):
                raise RuntimeError(
                    f"Layer {layer_idx}: {full_name} is a quantized checkpoint entry; "
                    "direct BF16 loading requires ordinary .weight tensors without scales."
                )
            present.add(local_name)
            weights_by_file[file_name].append((full_name, local_name))
        missing = sorted(expected.keys() - present)
        unexpected = sorted(present - expected.keys())
        if missing or unexpected:
            raise RuntimeError(
                f"Layer {layer_idx}: checkpoint key mismatch under {prefix}; "
                f"missing ({len(missing)}): {missing[:5]}; "
                f"unexpected ({len(unexpected)}): {unexpected[:5]}"
            )

        for file_name, names in weights_by_file.items():
            with safe_open(str(Path(self.model_path) / file_name), framework="pt", device="cpu") as f:
                for full_name, local_name in names:
                    tensor = self._prepare_layer_tensor(local_name, f.get_tensor(full_name))
                    shape, dtype = expected[local_name]
                    if tensor.shape != shape:
                        raise RuntimeError(
                            f"Layer {layer_idx}: shape mismatch for {full_name}: "
                            f"checkpoint {tuple(tensor.shape)}, model {tuple(shape)}"
                        )
                    if dtype.is_floating_point:
                        if not tensor.is_floating_point():
                            raise RuntimeError(f"Layer {layer_idx}: {full_name} must be a floating tensor")
                        dtype = torch.bfloat16
                    tensor = tensor.to(dtype=dtype)
                    target_device = self._resolve_param_device(local_name, device_map)
                    # Explicit dtype is essential: meta parameters default to
                    # FP32 and Accelerate otherwise casts the value back to it.
                    set_module_tensor_to_device(
                        block, local_name, target_device, value=tensor, dtype=dtype,
                    )
                    del tensor

        unresolved = [
            name for name, value in list(block.named_parameters()) + list(block.named_buffers())
            if value.is_meta
        ]
        if unresolved:
            raise RuntimeError(f"Layer {layer_idx}: tensors still on meta after loading: {unresolved[:5]}")
        # Includes non-persistent buffers initialized outside meta above.
        for name, buffer in list(block.named_buffers()):
            dtype = torch.bfloat16 if buffer.is_floating_point() else buffer.dtype
            target_device = self._resolve_param_device(name, device_map)
            set_module_tensor_to_device(
                block, name, target_device, value=buffer, dtype=dtype,
            )
        return block.eval()

    def _build_block_device_map(self, block):
        """Keep non-expert modules on the root and shard routed experts."""
        n_npu = max(1, int(torch.npu.device_count()))
        moe = getattr(block, "block_sparse_moe", None)
        experts = getattr(moe, "experts", None) if moe is not None else None
        if n_npu <= 1 or experts is None or len(experts) == 0:
            return None

        device_map = {"": str(self.args.device)}
        for expert_idx in range(len(experts)):
            device_map[
                f"block_sparse_moe.experts.{expert_idx}"
            ] = f"npu:{expert_idx % n_npu}"
        return device_map

    def _dispatch_block(self, module):
        device_map = self._build_block_device_map(module)
        if device_map is None:
            return super()._dispatch_block(module)

        # Quant wrappers are created after the sharded checkpoint load. Move
        # their newly created quantizers/algorithm state to the same device as
        # the wrapped projection without ever moving the whole MoE container.
        for name, submodule in module.named_modules():
            if isinstance(
                submodule,
                (QuantKimiMLP, QuantKimiLinear, QuantKimiMLAAttention),
            ):
                device = self._resolve_param_device(name, device_map)
                submodule.to(device)

        for path, device in sorted(
            device_map.items(), key=lambda item: item[0].count("."),
        ):
            target = module
            if path:
                for part in path.split("."):
                    target = getattr(target, part)
            hook = _NoMoveAlignDevicesHook(
                execution_device=device,
                io_same_device=(path == ""),
                place_submodules=True,
            )
            add_hook_to_module(target, hook)
        self._retag_descendant_hooks(module, device_map)
        return module

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
            local_name = weight_name[len(prefix):]
            state_dict[local_name] = self._prepare_layer_tensor(local_name, tensor)
        return state_dict

    def _prepare_layer_tensor(self, local_name, tensor):
        """Apply the same K3 checkpoint normalization in both loading paths."""
        if local_name == "self_attn.A_log":
            linear_attn_config = self.config.linear_attn_config
            num_heads = linear_attn_config["num_heads"]
            head_dim = linear_attn_config["head_dim"]
            if tensor.shape == (head_dim,):
                padding = tensor[num_heads:]
                if torch.count_nonzero(padding).item() != 0:
                    raise RuntimeError("Kimi K3 self_attn.A_log contains non-zero padding")
                tensor = tensor[:num_heads].contiguous()
        return tensor

    def build_quant_block(self, layer_idx):
        decoder_layer = self.block(layer_idx)
        quantize_attention = any(
            target in self.quant_target
            for target in ("attn-linear", "attn-cache")
        )
        is_linear_attn = getattr(decoder_layer, "is_linear_attn", False)
        if quantize_attention and not is_linear_attn:
            decoder_layer.self_attn = QuantKimiMLAAttention(
                self.args, decoder_layer.self_attn,
            )
        elif "attn-linear" in self.quant_target and is_linear_attn:
            decoder_layer.self_attn = quantize_kimi_kda_linears(
                self.args, decoder_layer.self_attn,
            )

        if "mlp" in self.quant_target and hasattr(decoder_layer, "mlp"):
            decoder_layer.mlp = QuantKimiMLP(
                self.args, decoder_layer.mlp, group="mlp",
            )

        if "moe" in self.quant_target and hasattr(decoder_layer, "block_sparse_moe"):
            moe = decoder_layer.block_sparse_moe
            for expert_idx, expert in enumerate(moe.experts):
                if expert is not None and not isinstance(expert, QuantKimiMLP):
                    moe.experts[expert_idx] = QuantKimiMLP(
                        self.args, expert, group="moe.routed",
                    )
            shared_experts = getattr(moe, "shared_experts", None)
            if shared_experts is not None and not isinstance(shared_experts, QuantKimiMLP):
                moe.shared_experts = QuantKimiMLP(
                    build_no_algo_args(self.args),
                    shared_experts,
                    group="moe.shared",
                )
            for name in (
                "routed_expert_down_proj",
                "routed_expert_up_proj",
            ):
                linear = getattr(moe, name, None)
                if linear is not None and not isinstance(linear, QuantKimiLinear):
                    setattr(
                        moe,
                        name,
                        QuantKimiLinear(
                            self.args,
                            linear,
                            name=name,
                            group="moe.routed",
                        ),
                    )
        return decoder_layer

    def iter_ptq_units(self, layer_idx, block):
        attention = getattr(block, "self_attn", None)
        has_quant_kda_linear = (
            "attn-linear" in self.quant_target
            and attention is not None
            and any(
                isinstance(getattr(attention, name, None), QuantKimiLinear)
                for name in ("q_proj", "k_proj", "v_proj", "o_proj")
            )
        )
        if any(
            target in self.quant_target
            for target in ("attn-linear", "attn-cache")
        ) and (
            isinstance(attention, QuantKimiMLAAttention)
            or has_quant_kda_linear
        ):
            yield make_ptq_unit(
                "attn", "self_attn", layer_idx, attention,
            )

        if "mlp" in self.quant_target:
            mlp = getattr(block, "mlp", None)
            if mlp is not None:
                yield make_ptq_unit("mlp", "mlp", layer_idx, mlp)

        if "moe" in self.quant_target:
            moe = getattr(block, "block_sparse_moe", None)
            experts = getattr(moe, "experts", []) if moe is not None else []
            yield from iter_indexed_units(
                kind="moe",
                name_prefix="expert",
                layer_idx=layer_idx,
                items=experts,
                metadata_fn=lambda expert_idx, _: {"expert_idx": expert_idx},
            )

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
        quant_layers = {}
        quantize_moe = not self.quant_target or "moe" in self.quant_target
        quantize_mla = "attn-linear" in self.quant_target
        attn_bits = (
            ensure_bit_policy(self.args)["attn-linear"]
            if quantize_mla else None
        )
        for weight_name in self._load_weight_map():
            if quantize_moe and self._is_kimi_moe_weight_packed(weight_name):
                output_name = self.tensorwise_output_weight_name(weight_name)
                quant_layers[output_name.rsplit(".", 1)[0]] = 4
                continue

            mla_match = self._MLA_WEIGHT_RE.match(weight_name)
            if quantize_mla and mla_match is not None:
                projection_name = mla_match.group(1)
                quant_layers[weight_name.rsplit(".", 1)[0]] = (
                    attn_bits[projection_name].w
                )
        return quant_layers

    def generate_tensorwise_ignore_layers(self):
        return []

    def bits_scheme(self):
        return [
            {"targets": ["Linear"], "w_bits": 8, "a_bits": 8},
            {"targets": ["MoEGMM"], "w_bits": 4, "a_bits": 8},
        ]
