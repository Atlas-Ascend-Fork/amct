# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Kimi-K3 checkpoint conversion without constructing the model."""

import json
import os
import re
from pathlib import Path

import torch
import yaml
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from amct_pytorch.common.models import MODEL_REGISTRY
from amct_pytorch.common.models.llm.common.deploy_export import (
    generate_quant_config,
    quant_payload,
)
from amct_pytorch.common.models.llm.common.weight_path_validation import (
    collect_safetensors_files,
    resolve_safetensors_path,
    validate_weight_map,
)
from amct_pytorch.quantization.dtypes.fp_impl import weight_dequant
from amct_pytorch.quantization.dtypes.mxfp import QuantDequantMx


@MODEL_REGISTRY.register(
    name="kimi_k3", task="llm", family="kimi", description="Kimi-K3 tensorwise deploy"
)
class KimiK3:
    """Fixed MXFP4 routed experts / MXFP8 latent and full MLA projections."""

    _MOE_WEIGHT_RE = re.compile(
        r"^language_model\.model\.layers\.\d+\.block_sparse_moe\.experts\.\d+\.w[123]\.weight_packed$"
    )
    _MLA_WEIGHT_RE = re.compile(
        r"^language_model\.model\.layers\.(?P<layer_idx>\d+)\.self_attn\.(?:q_a_proj|q_b_proj|kv_a_proj_with_mqa|o_proj)\.weight$"
    )
    _LATENT_MOE_WEIGHT_RE = re.compile(
        r"^language_model\.model\.layers\.\d+\.block_sparse_moe\.(?:routed_expert_down_proj|routed_expert_up_proj)\.weight$"
    )

    _TENSORWISE_LINEAR_WEIGHT_RES = (
        re.compile(
            r"^language_model\.model\.layers\.\d+\.self_attn\."
            r"(?:q_proj|k_proj|v_proj|f_a_proj|f_b_proj|b_proj|g_proj|g_a_proj|g_b_proj|o_proj|"
            r"q_a_proj|q_b_proj|kv_a_proj_with_mqa|kv_b_proj)\.weight$"
        ),
        re.compile(
            r"^language_model\.model\.layers\.\d+\.mlp\."
            r"(?:gate_proj|up_proj|down_proj)\.weight$"
        ),
        re.compile(
            r"^language_model\.model\.layers\.\d+\.block_sparse_moe\.shared_experts\."
            r"(?:gate_proj|up_proj|down_proj)\.weight$"
        ),
        re.compile(
            r"^language_model\.model\.layers\.\d+\.block_sparse_moe\."
            r"(?:routed_expert_down_proj|routed_expert_up_proj)\.weight$"
        ),
        re.compile(
            r"^language_model\.model\.layers\.\d+\."
            r"(?:self_attention_res_proj|mlp_res_proj)\.weight$"
        ),
        re.compile(r"^language_model\.model\.output_attn_res_proj\.weight$"),
        re.compile(r"^language_model\.lm_head\.weight$"),
        re.compile(
            r"^vision_tower\.encoder\.blocks\.\d+\."
            r"(?:wqkv|wo|mlp\.fc0|mlp\.fc1)\.weight$"
        ),
        re.compile(r"^mm_projector\.proj\.\d+\.weight$"),
    )

    def __init__(self, args):
        if getattr(args, "granularity", None) != "tensor":
            raise ValueError("Kimi-K3 supports tensorwise deploy only")
        if (
            getattr(args, "deploy_format", "legacy") != "legacy"
            or getattr(args, "deploy_selection", None) is not None
            or getattr(args, "quant_layers_config", None)
        ):
            raise ValueError(
                "Kimi-K3 supports legacy deploy without JSON selection only"
            )
        if getattr(args, "quant_dtype", "") not in ("", None, "mxfp", "bf16"):
            raise ValueError("Kimi-K3 supports mxfp or BF16 tensorwise deploy only")
        self.args = args
        self.model_path = args.model
        self.quant_dtype = getattr(args, "quant_dtype", "")
        self.quant_target = getattr(args, "quant_target", []) or []
        output = Path(args.output_dir).resolve()
        source = Path(self.model_path).resolve()
        if (
            output == source
            or source.is_relative_to(output)
            or output.is_relative_to(source)
        ):
            raise ValueError("Kimi source and output directories must not overlap")
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            raise ValueError("Kimi output directory must be new or empty")
        bit_config = getattr(args, "bit_config", None)
        if bit_config:
            fixed = Path(__file__).resolve().parents[5] / "configs" / "infer_0918.yaml"
            if yaml.safe_load(
                Path(bit_config).read_text(encoding="utf-8")
            ) != yaml.safe_load(fixed.read_text(encoding="utf-8")):
                raise ValueError(
                    "Kimi-K3 requires the fixed infer_0918.yaml bit_config"
                )
        if set(self.quant_target) - {"moe", "attn-linear"}:
            raise ValueError("Kimi-K3 tensorwise targets are moe and attn-linear")
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
            if not isinstance(layers, list) or any(
                type(i) is not int or i <= 0 for i in layers
            ):
                raise ValueError(
                    "Kimi-K3 config must define positive full_attn_layers (one-based)"
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
        selected = {}
        for name in self._load_weight_map():
            if not self.quant_target or "moe" in self.quant_target:
                if self._MOE_WEIGHT_RE.match(name):
                    selected[name.rsplit(".", 1)[0]] = 4
                elif self._LATENT_MOE_WEIGHT_RE.match(name):
                    selected[name.rsplit(".", 1)[0]] = 8
            match = self._MLA_WEIGHT_RE.match(name)
            if (
                "attn-linear" in self.quant_target
                and match
                and int(match.group("layer_idx")) in self._tensorwise_mla_layer_indices
            ):
                selected[name.rsplit(".", 1)[0]] = 8
        return selected

    @staticmethod
    def _runtime_name(name):
        return name.removeprefix("language_model.")

    def generate_tensorwise_ignore_layers(self):
        selected = self.generate_tensorwise_quant_layers()
        return sorted(
            {
                self._runtime_name(name.rsplit(".", 1)[0])
                for name in self._load_weight_map()
                if (
                    any(
                        pattern.match(name)
                        for pattern in self._TENSORWISE_LINEAR_WEIGHT_RES
                    )
                    or self._MOE_WEIGHT_RE.match(name)
                )
                and name.rsplit(".", 1)[0] not in selected
            }
        )

    @staticmethod
    def bits_scheme():
        return [
            {"targets": ["Linear"], "w_bits": 8, "a_bits": 8},
            {"targets": ["MoEGMM"], "w_bits": 4, "a_bits": 8},
        ]

    @staticmethod
    def cache_scheme():
        return {"kv_cache_scheme": None}

    def convert_tensorwise_shard(
        self, source_file, model_dir, original_weight_map, quant_layers, loaded_files
    ):
        files = collect_safetensors_files(model_dir)
        validate_weight_map(model_dir, original_weight_map, files)
        for name in original_weight_map:
            if (
                name.endswith(".weight_packed")
                and name.removesuffix(".weight_packed") + ".weight"
                in original_weight_map
            ):
                raise ValueError(f"Duplicate Kimi weight/weight_packed module: {name}")

        def load(name):
            if name not in original_weight_map:
                raise ValueError(f"Missing Kimi weight/scale in index: {name}")
            shard = original_weight_map[name]
            with safe_open(
                str(resolve_safetensors_path(model_dir, shard, files)),
                framework="pt",
                device="cpu",
            ) as handle:
                if name not in handle.keys():  # noqa: SIM118 (safe_open is not a dict)
                    raise ValueError(f"Missing Kimi weight/scale in shard: {name}")
                return handle.get_tensor(name)

        result = {}
        for name, shard in original_weight_map.items():
            if shard != source_file:
                continue
            if name.endswith(".weight_scale"):
                owner = name.removesuffix(".weight_scale")
                if (
                    owner + ".weight" in original_weight_map
                    or owner + ".weight_packed" in original_weight_map
                ):
                    continue
            weight = load(name)
            if not name.endswith((".weight", ".weight_packed")):
                result[name] = weight
                continue
            module = name.rsplit(".", 1)[0]
            scale_name = module + ".weight_scale"
            packed = name.endswith(".weight_packed")
            lowbit = packed or weight.element_size() == 1
            scale = None
            if lowbit:
                scale = load(scale_name)
                if weight.ndim != 2:
                    raise ValueError(f"Invalid Kimi MX weight shape: {name}")
                width = weight.shape[-1] * (2 if packed else 1)
                if (
                    weight.ndim != 2
                    or scale.dtype != torch.uint8
                    or scale.shape != (weight.shape[0], width // 32)
                    or width % 32
                ):
                    raise ValueError(
                        f"Invalid Kimi MX weight/scale shape or dtype: {name}"
                    )
                if packed and weight.dtype not in (torch.uint8, torch.int8):
                    raise ValueError(f"Kimi packed weight must be INT8/UINT8: {name}")
                if not packed and weight.dtype not in (
                    torch.float8_e4m3fn,
                    torch.float8_e5m2,
                ):
                    raise ValueError(f"Kimi MXFP8 weight must have FP8 dtype: {name}")
            if (
                self.quant_dtype == "mxfp"
                and module in quant_layers
                and packed
                and self._MOE_WEIGHT_RE.match(name)
            ):
                result[name] = weight
                result[scale_name] = scale
                continue
            if lowbit:
                weight = weight_dequant(
                    weight, scale, block_size=32, is_mx=True, is_packed=packed
                )
            output_name = module + ".weight"
            if self.quant_dtype == "mxfp" and module in quant_layers:
                if (
                    weight.ndim != 2
                    or not weight.is_floating_point()
                    or weight.numel() == 0
                    or weight.shape[-1] % 32
                ):
                    raise ValueError(
                        f"Kimi selected weight must be nonempty floating 2-D with groups of 32: {name}"
                    )
                if not torch.isfinite(weight).all():
                    raise ValueError(f"Kimi selected weight must be finite: {name}")
                payload = quant_payload(
                    QuantDequantMx, output_name, weight, quant_layers[module]
                )
                result.update(payload)
            else:
                result[output_name] = (
                    weight.to(torch.bfloat16)
                    if lowbit
                    or self.quant_dtype != "mxfp"
                    and weight.is_floating_point()
                    else weight
                )
        output = Path(self.args.output_dir).resolve()
        path = output / source_file
        if Path(source_file).name != source_file or path.resolve().parent != output:
            raise ValueError("Kimi output shard must be a local filename")
        if result:
            output.mkdir(parents=True, exist_ok=True)
            temporary = output / f".{source_file}.tmp"
            save_file({k: v.contiguous() for k, v in result.items()}, str(temporary))
            os.replace(temporary, path)
        loaded_files.clear()
        return dict.fromkeys(result, source_file)

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
