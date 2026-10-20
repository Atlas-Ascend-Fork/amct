# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Kimi-K3 checkpoint conversion without constructing the model."""

import json
import os
import re
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from amct_pytorch.common.models import MODEL_REGISTRY
from amct_pytorch.common.models.llm.common.deploy_export import (
    generate_quant_config,
)
from amct_pytorch.common.models.llm.common.weight_path_validation import (
    collect_safetensors_files,
    resolve_safetensors_path,
)
from amct_pytorch.quantization.dtypes.fp_impl import weight_dequant


@MODEL_REGISTRY.register(
    name="kimi_k3", task="llm", family="kimi", description="Kimi-K3 tensorwise deploy"
)
class KimiK3:
    """Fixed MXFP4 routed experts / MXFP8 latent and full MLA projections."""

    _MOE_WEIGHT_RE = re.compile(
        r"^language_model\.model\.layers\.\d+\.block_sparse_moe\.experts\.\d+\.w[123]\.weight_packed$"
    )
    _ROUTED_PROJECTIONS = ("w1", "w2", "w3")
    _LATENT_PROJECTIONS = ("routed_expert_down_proj", "routed_expert_up_proj")
    _MLA_PROJECTIONS = ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "o_proj")

    _UNQUANTIZED_ATTN_PROJECTIONS = (
        "q_proj",
        "k_proj",
        "v_proj",
        "f_a_proj",
        "f_b_proj",
        "b_proj",
        "g_proj",
        "g_a_proj",
        "g_b_proj",
        "kv_b_proj",
    )
    _FLOAT_MLP_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")

    def __init__(self, args):
        if getattr(args, "granularity", None) != "tensor":
            raise ValueError("Kimi-K3 supports tensorwise deploy only")
        if getattr(args, "quant_dtype", "") not in ("", None, "mxfp", "bf16"):
            raise ValueError("Kimi-K3 supports mxfp or BF16 tensorwise deploy only")
        self.args = args
        self.model_path = args.model
        self.quant_dtype = getattr(args, "quant_dtype", "")
        self.quant_target = getattr(args, "quant_target", []) or []
        self.config = json.loads(
            (Path(self.model_path) / "config.json").read_text(encoding="utf-8")
        )
        self._weight_map = None
        self._tensorwise_mla_layer_indices = set()
        if "attn-linear" in self.quant_target:
            layers = (
                self.config.get("text_config", self.config)
                .get("linear_attn_config", {})
                .get("full_attn_layers")
            )
            self._tensorwise_mla_layer_indices = {i - 1 for i in layers}

    def _load_weight_map(self):
        if self._weight_map is None:
            root = Path(self.model_path)
            index = root / "model.safetensors.index.json"
            if index.exists():
                self._weight_map = json.loads(index.read_text(encoding="utf-8"))[
                    "weight_map"
                ]
            else:
                files = collect_safetensors_files(root)
                if len(files) != 1:
                    raise ValueError("Kimi-K3 multiple shards require a weight index")
                path = next(iter(files))
                self._weight_map = dict.fromkeys(load_file(str(path)), path.name)
        return self._weight_map

    def generate_tensorwise_quant_layers(self):
        weight_map = self._load_weight_map()
        text_config = self.config.get("text_config", self.config)
        layers = range(text_config["num_hidden_layers"])
        expert_count = text_config["num_experts"]
        selected = {}

        def add_weight(module, suffix, bit):
            if f"{module}.{suffix}" in weight_map:
                selected[module] = bit

        for layer_idx in layers:
            prefix = f"language_model.model.layers.{layer_idx}"
            if not self.quant_target or "moe" in self.quant_target:
                for expert_idx in range(expert_count):
                    for projection in self._ROUTED_PROJECTIONS:
                        add_weight(
                            f"{prefix}.block_sparse_moe.experts.{expert_idx}.{projection}",
                            "weight_packed",
                            4,
                        )
                for projection in self._LATENT_PROJECTIONS:
                    add_weight(f"{prefix}.block_sparse_moe.{projection}", "weight", 8)
            if (
                "attn-linear" in self.quant_target
                and layer_idx in self._tensorwise_mla_layer_indices
            ):
                for projection in self._MLA_PROJECTIONS:
                    add_weight(f"{prefix}.self_attn.{projection}", "weight", 8)
        return selected

    @staticmethod
    def _runtime_name(name):
        return name.removeprefix("language_model.")

    def generate_tensorwise_ignore_layers(self):
        weight_map = self._load_weight_map()
        ignore = set()

        def add_float(module, suffix="weight"):
            if f"{module}.{suffix}" in weight_map:
                ignore.add(self._runtime_name(module))

        def indexed_ids(prefix):
            return sorted(
                {
                    name.removeprefix(prefix).split(".", 1)[0]
                    for name in weight_map
                    if name.startswith(prefix)
                    and name.removeprefix(prefix).split(".", 1)[0].isdigit()
                },
                key=int,
            )

        quantize_moe = not self.quant_target or "moe" in self.quant_target
        quantize_attn = "attn-linear" in self.quant_target
        for layer_idx in indexed_ids("language_model.model.layers."):
            prefix = f"language_model.model.layers.{layer_idx}"
            for projection in self._UNQUANTIZED_ATTN_PROJECTIONS:
                add_float(f"{prefix}.self_attn.{projection}")
            # o_proj belongs to quantized MLA only on configured full-attention layers.
            if (
                not quantize_attn
                or int(layer_idx) not in self._tensorwise_mla_layer_indices
            ):
                for projection in self._MLA_PROJECTIONS:
                    add_float(f"{prefix}.self_attn.{projection}")
            for group in ("mlp", "block_sparse_moe.shared_experts"):
                for projection in self._FLOAT_MLP_PROJECTIONS:
                    add_float(f"{prefix}.{group}.{projection}")
            for projection in ("self_attention_res_proj", "mlp_res_proj"):
                add_float(f"{prefix}.{projection}")
            if not quantize_moe:
                for projection in self._LATENT_PROJECTIONS:
                    add_float(f"{prefix}.block_sparse_moe.{projection}")
                expert_prefix = f"{prefix}.block_sparse_moe.experts."
                for expert_idx in indexed_ids(expert_prefix):
                    for projection in self._ROUTED_PROJECTIONS:
                        add_float(
                            f"{expert_prefix}{expert_idx}.{projection}", "weight_packed"
                        )
        for module in (
            "language_model.model.output_attn_res_proj",
            "language_model.lm_head",
        ):
            add_float(module)
        for block_idx in indexed_ids("vision_tower.encoder.blocks."):
            for projection in ("wqkv", "wo", "mlp.fc0", "mlp.fc1"):
                add_float(f"vision_tower.encoder.blocks.{block_idx}.{projection}")
        for projection_idx in indexed_ids("mm_projector.proj."):
            add_float(f"mm_projector.proj.{projection_idx}")
        return sorted(ignore)

    @staticmethod
    def bits_scheme():
        return [
            {"targets": ["Linear"], "w_bits": 8, "a_bits": 8},
            {"targets": ["MoEGMM"], "w_bits": 4, "a_bits": 8},
        ]

    @staticmethod
    def cache_scheme():
        return {"kv_cache_scheme": None}

    @staticmethod
    def get_scale_name(weight_name):
        module = weight_name.rsplit(".", 1)[0]
        return ".weight_scale", module + ".weight_scale"

    @staticmethod
    def block_size(weight):
        return 32

    def convert_tensorwise_tensor(
        self, name, weight, model_dir, original_weight_map, quant_layers
    ):
        """Handle packed experts and auxiliaries before common source decoding."""
        if name.endswith(".weight_scale"):
            return {}
        if not name.endswith((".weight", ".weight_packed")):
            return {name: weight}
        if not name.endswith(".weight_packed"):
            return None

        module = name.rsplit(".", 1)[0]
        scale_name = module + ".weight_scale"
        if scale_name not in original_weight_map:
            raise ValueError(f"Missing Kimi weight/scale in index: {scale_name}")
        files = collect_safetensors_files(model_dir)
        path = resolve_safetensors_path(
            model_dir, original_weight_map[scale_name], files
        )
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            if scale_name not in handle.keys():  # noqa: SIM118
                raise ValueError(f"Missing Kimi weight/scale in shard: {scale_name}")
            scale = handle.get_tensor(scale_name)
        if (
            self.quant_dtype == "mxfp"
            and module in quant_layers
            and self._MOE_WEIGHT_RE.match(name)
        ):
            return {name: weight, scale_name: scale}
        dequant_weight = weight_dequant(
            weight, scale, block_size=32, is_mx=True, is_packed=True
        )
        return {module + ".weight": dequant_weight.to(torch.bfloat16)}

    def refresh_config(self, ignored):
        path = Path(self.args.output_dir) / "config.json"
        config = json.loads(path.read_text(encoding="utf-8"))
        text = config.get("text_config")
        if self.quant_dtype == "mxfp":
            quant = generate_quant_config(
                self.cache_scheme(), ignored, is_mx=True, bits_scheme=self.bits_scheme()
            )
            config["quantization_config"] = quant
            if isinstance(text, dict):
                text["quantization_config"] = quant
        else:
            config.pop("quantization_config", None)
            if isinstance(text, dict):
                text.pop("quantization_config", None)
        temporary = path.with_name(".config.json.tmp")
        temporary.write_text(json.dumps(config, indent=2), encoding="utf-8")
        os.replace(temporary, path)
