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

from __future__ import annotations

import copy
import json
import os
import re
from fnmatch import fnmatchcase
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from amct_pytorch.common.models import MODEL_REGISTRY
from amct_pytorch.common.models.llm.common.deployment.deploy_export import (
    adapt_payload,
    convert_state_dict,
    quant_payload,
)
from amct_pytorch.common.models.llm.common.deployment.deploy_selection import (
    MtpSelectionReport,
    QuantSpec,
    ResolvedDeployPlan,
    get_structured_quant_spec,
    match_deploy_modules,
)
from amct_pytorch.common.models.llm.common.deployment.schemes import (
    QuantSchemeKey,
    get_quant_scheme,
)
from amct_pytorch.common.models.llm.common.weight_path_validation import (
    collect_safetensors_files,
    resolve_safetensors_path,
)
from amct_pytorch.quantization.dtypes import DTYPE_REGISTRY, register_dtype

_LAYER_RE = re.compile(r"^model\.language_model\.layers\.(\d+)\.(.+)$")
_WEIGHT_SCALE_INV = ".weight_scale_inv"
_BLOCK_SIZE = 128

_GLM53_QUANT_DTYPES = {
    "A3": frozenset({"int"}),
    "ascend950": frozenset({"int", "mxfp"}),
}


@MODEL_REGISTRY.register(
    name="glm5_3",
    task="llm",
    family="glm",
    description="GLM-5.3 FP8 tensor deployment only",
)
class GLM5_3:
    """Lightweight GLM5.3 deployment-only adapter."""

    def __init__(self, args):
        self.validate_deploy_args(args)
        self.args = args
        self.model_path = Path(args.model)
        self.output_dir = Path(args.output_dir)
        source, output = self.model_path.resolve(), self.output_dir.resolve()
        if source == output or source in output.parents or output in source.parents:
            raise ValueError(
                "GLM5.3 deployment output and source directories must not overlap"
            )
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            raise ValueError("GLM5.3 deployment output directory must be new or empty")
        self.config = json.loads(
            (self.model_path / "config.json").read_text(encoding="utf-8")
        )
        self.safetensors_files = collect_safetensors_files(self.model_path)
        self.processed_modules: set[str] = set()
        self._planned_specs: dict[str, QuantSpec] = {}
        self.output_tensor_bytes = 0
        self.converted_names: set[str] = set()

    @staticmethod
    def validate_deploy_args(args):
        if getattr(args, "granularity", None) != "tensor":
            raise NotImplementedError("GLM5.3 supports tensor deployment only")
        for key in (
            "algos",
            "attn_linear_param_dir",
            "attn_cache_param_dir",
            "moe_mlp_param_dir",
        ):
            if getattr(args, key, None):
                raise ValueError(f"GLM5.3 tensor deployment does not consume --{key}")
        selection = getattr(args, "deploy_selection", None)
        if not getattr(selection, "uses_structured_rules", False):
            raise ValueError(
                "GLM5.3 tensor deployment requires structured quant_layers rules"
            )
        for rule in selection.quant_layers:
            spec = rule.quant_spec
            scheme = get_quant_scheme(
                QuantSchemeKey(spec.quant_dtype, spec.weight_bits, spec.activation_bits)
            )
            if scheme.quant_dtype == "hifp":
                raise NotImplementedError(
                    "GLM5.3 tensor deployment does not support hifp yet"
                )

    @classmethod
    def validate_deploy_source(cls, args):
        source = Path(args.model)
        try:
            config = json.loads((source / "config.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        quant = config.get("quantization_config")
        markers = (
            "quant_model_description.json",
            "quant_model_weights.safetensors.index.json",
            "rot.safetensors",
        )
        return (
            config.get("model_type") == "glm5_next"
            and isinstance(quant, dict)
            and quant.get("quant_method") == "fp8"
            and not config.get("is_rot_used")
            and not any((source / marker).exists() for marker in markers)
        )

    @staticmethod
    def get_deploy_capabilities():
        return _GLM53_QUANT_DTYPES

    def _text_config(self):
        return self.config.get("text_config", self.config)

    def _num_layers(self):
        return int(self._text_config().get("num_hidden_layers", 0))

    def _first_dense(self):
        return int(self._text_config().get("first_k_dense_replace", 0))

    def _is_mtp(self, module):
        match = _LAYER_RE.match(module)
        return bool(match and int(match[1]) >= self._num_layers()) or module.startswith(
            "mtp."
        )

    def _is_full_attention_layer(self, layer_idx):
        text_config = self._text_config()
        full = text_config.get("linear_attn_config", {}).get("full_attn_layers", ())
        if layer_idx in full:
            return True
        layer_types = text_config.get("layer_types", ())
        return (
            layer_idx < len(layer_types)
            and layer_types[layer_idx] == "deepseek_sparse_attention"
        )

    def _deploy_role(self, name: str) -> tuple[str, str] | None:
        match = _LAYER_RE.fullmatch(name)
        if match is None:
            return None
        layer_idx, suffix = int(match[1]), match[2]
        if layer_idx >= self._num_layers():
            return None
        if self._is_full_attention_layer(layer_idx) and suffix in {
            "self_attn.q_a_proj",
            "self_attn.q_b_proj",
            "self_attn.kv_a_proj_with_mqa",
            "self_attn.o_proj",
        }:
            return "attn-linear", suffix.rsplit(".", 1)[1]
        dense = layer_idx < self._first_dense()
        if dense and suffix in {"mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"}:
            return "mlp", suffix.rsplit(".", 1)[1]
        if not dense:
            if self._text_config().get("n_shared_experts", 0) and suffix in {
                "mlp.shared_experts.gate_proj",
                "mlp.shared_experts.up_proj",
                "mlp.shared_experts.down_proj",
            }:
                return "moe.shared", suffix.rsplit(".", 1)[1]
            expert = re.fullmatch(
                r"mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)", suffix
            )
            if expert and int(expert[1]) < int(
                self._text_config().get("n_routed_experts", 0)
            ):
                return "moe.routed", expert[2]
        return None

    def get_deploy_candidates(self, module_names: set[str]) -> set[str]:
        return {name for name in module_names if self._deploy_role(name) is not None}

    def validate_deploy_selection(self, selected: set[str]) -> None:
        unsupported = sorted(
            name for name in selected if self._deploy_role(name) is None
        )
        if unsupported:
            raise ValueError(
                f"Unsupported GLM5.3 deployment modules: {unsupported[:20]}"
            )

        def require_pair(first, second):
            if (first in selected) != (second in selected):
                raise ValueError(
                    f"Fused modules must share quantization: {first}, {second}"
                )

        for layer_idx in range(self._num_layers()):
            prefix = f"model.language_model.layers.{layer_idx}."
            if self._is_full_attention_layer(layer_idx):
                require_pair(
                    prefix + "self_attn.q_a_proj",
                    prefix + "self_attn.kv_a_proj_with_mqa",
                )
            mlp = prefix + "mlp."
            if layer_idx < self._first_dense():
                require_pair(mlp + "gate_proj", mlp + "up_proj")
                continue
            require_pair(
                mlp + "shared_experts.gate_proj", mlp + "shared_experts.up_proj"
            )
            expert_prefix = mlp + "experts."
            expected = {
                expert_prefix + f"{expert}.{projection}"
                for expert in range(int(self._text_config().get("n_routed_experts", 0)))
                for projection in ("gate_proj", "up_proj", "down_proj")
            }
            if selected & expected and not expected <= selected:
                raise ValueError(
                    f"Deployment requires all experts and projections in {expert_prefix}"
                )

    def prepare_tensor_deploy_plan(self, weight_map):
        names = {
            key.removesuffix(".weight") for key in weight_map if key.endswith(".weight")
        }
        selected, counts = match_deploy_modules(names, self.args.deploy_selection)
        ignored = frozenset(
            name
            for name in names
            if any(
                fnmatchcase(name, pattern)
                for pattern in self.args.deploy_selection.ignore_layers
            )
        )
        selected -= set(ignored)
        specs = {
            name: get_structured_quant_spec(self.args.deploy_selection, name)
            for name in sorted(selected)
        }
        self.validate_deploy_selection(set(specs))
        self._planned_specs = specs
        self.processed_modules.clear()
        mtp = frozenset(name for name in names if self._is_mtp(name))
        return ResolvedDeployPlan(
            module_quant_specs=specs,
            selected_weight_keys=frozenset(name + ".weight" for name in specs),
            ignored_modules=ignored,
            unselected_weight_modules=frozenset(names - set(specs)),
            match_counts=counts,
            mtp=MtpSelectionReport(mtp, mtp & ignored, mtp & set(specs)),
            unmatched_ignore_patterns=frozenset(
                pattern
                for pattern in self.args.deploy_selection.ignore_layers
                if counts.get(pattern, 0) == 0
            ),
        )

    def convert_tensorwise_shard(
        self, source_file, model_dir, original_weight_map, quant_layers, loaded_files
    ):
        source_path = resolve_safetensors_path(
            model_dir, source_file, self.safetensors_files
        )
        source = load_file(str(source_path), device="cpu")
        register_dtype()
        output = {}
        processed = set()
        for name, tensor in source.items():
            if name.endswith(_WEIGHT_SCALE_INV):
                continue
            module = name.removesuffix(".weight") if name.endswith(".weight") else None
            weight = tensor
            if module is not None and tensor.dtype == torch.float8_e4m3fn:
                scale_name = module + _WEIGHT_SCALE_INV
                weight = convert_state_dict(
                    tensor,
                    name,
                    scale_name,
                    original_weight_map,
                    model_dir,
                    loaded_files,
                    block_size=_BLOCK_SIZE,
                    safetensors_files=self.safetensors_files,
                ).to(torch.bfloat16)
            if module in quant_layers:
                spec = self._planned_specs[module]
                payload = quant_payload(
                    DTYPE_REGISTRY.get(spec.quant_dtype),
                    name,
                    weight,
                    spec.weight_bits,
                )
                output.update(
                    adapt_payload(
                        name,
                        payload,
                        spec.quant_dtype,
                        spec.weight_bits,
                        spec.activation_bits,
                    )
                )
                processed.add(module)
            else:
                output[name] = weight
        destination = self.output_dir / source_file
        temporary = destination.with_name(f".{destination.name}.tmp")
        save_file(output, str(temporary), metadata={"format": "pt"})
        os.replace(temporary, destination)
        self.processed_modules.update(processed)
        self.converted_names.update(output)
        self.output_tensor_bytes += sum(
            value.numel() * value.element_size() for value in output.values()
        )
        return {name: source_file for name in output}

    def validate_tensorwise_result(self, plan):
        if self.processed_modules != plan.selected_modules:
            raise ValueError(
                "Actual GLM5.3 converted modules differ from the deployment plan"
            )
        return self.processed_modules

    def refresh_config(self, quant_ignore_layers, *, tensor_plan=None):
        del quant_ignore_layers, tensor_plan
        config = copy.deepcopy(self.config)
        config.pop("quantization_config", None)
        if isinstance(config.get("text_config"), dict):
            config["text_config"].pop("quantization_config", None)
        temporary = self.output_dir / ".config.json.tmp"
        temporary.write_text(
            json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        os.replace(temporary, self.output_dir / "config.json")
