# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the Apache License, Version 2.0.

"""Kimi-K3 tensorwise deployment adapter.

Kimi-K3 ships routed experts in the native MXFP4 ``weight_packed`` format.
The adapter quantizes the requested BF16 MLA/latent matrices to MXFP8 and
renames the native expert payloads to vLLM's logical ``weight`` contract.
It intentionally does not construct a model or expose blockwise/PTQ paths.
"""

from __future__ import annotations

import copy
import json
import os
import re
from fnmatch import fnmatchcase
from pathlib import Path

from safetensors.torch import load_file, save_file

from amct_pytorch.common.models import MODEL_REGISTRY
from amct_pytorch.common.models.llm.common.deployment.deploy_export import (
    adapt_payload,
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


_LAYER_RE = re.compile(r"^language_model\.model\.layers\.(\d+)\.(.+)$")
_PACKED_SUFFIX = ".weight_packed"
_SCALE_SUFFIX = ".weight_scale"


@MODEL_REGISTRY.register(
    name="kimi_k3",
    task="llm",
    family="kimi",
    description="Kimi-K3 tensorwise deployment only",
)
class KimiK3:
    """Lightweight Kimi-K3 deployment-only registry entry."""

    def __init__(self, args):
        self.validate_deploy_args(args)
        self.args = args
        self.model_path = Path(args.model)
        self.output_dir = Path(args.output_dir)
        self.config = json.loads(
            (self.model_path / "config.json").read_text(encoding="utf-8")
        )
        self.safetensors_files = collect_safetensors_files(self.model_path)
        self.processed_modules: set[str] = set()
        self._planned_specs: dict[str, QuantSpec] = {}
        self.output_tensor_bytes = 0

    @staticmethod
    def validate_deploy_args(args):
        if getattr(args, "granularity", None) != "tensor":
            raise NotImplementedError("Kimi-K3 supports tensor deployment only")
        if getattr(args, "deploy_platform", None) != "ascend950":
            raise ValueError(
                "Kimi-K3 MXFP tensor deployment requires --deploy_platform ascend950"
            )
        if getattr(args, "quant_dtype", None) not in (None, "", "mxfp"):
            raise ValueError(
                "Kimi-K3 tensor deployment is controlled by quant_layers_config"
            )
        for key in (
            "algos",
            "attn_linear_param_dir",
            "attn_cache_param_dir",
            "moe_mlp_param_dir",
        ):
            if getattr(args, key, None):
                raise ValueError(f"Kimi-K3 tensor deployment does not consume --{key}")
        args.quant_dtype = "mxfp"

    @classmethod
    def validate_deploy_source(cls, args):
        source = Path(args.model)
        try:
            config = json.loads((source / "config.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        return config.get("model_type") == "kimi_k3" and not any(
            (source / marker).exists()
            for marker in ("quant_model_description.json", "rot.safetensors")
        )

    @staticmethod
    def get_deploy_capabilities():
        return {"ascend950": frozenset({"mxfp"})}

    @staticmethod
    def _canonical_module(name: str) -> str | None:
        for suffix in (_PACKED_SUFFIX, _SCALE_SUFFIX, ".weight"):
            if name.endswith(suffix):
                return name[: -len(suffix)]
        return None

    @classmethod
    def _canonical_weight_modules(cls, weight_map):
        return {
            module
            for module in (cls._canonical_module(name) for name in weight_map)
            if module is not None
        }

    @staticmethod
    def _is_expert(module: str) -> bool:
        return ".block_sparse_moe.experts." in module and module.endswith(
            (".w1", ".w2", ".w3")
        )

    @staticmethod
    def _is_supported_module(module: str) -> bool:
        match = _LAYER_RE.match(module)
        if match is None:
            return False
        suffix = match[2]
        if suffix in {
            "self_attn.q_a_proj",
            "self_attn.q_b_proj",
            "self_attn.kv_a_proj_with_mqa",
            "block_sparse_moe.routed_expert_up_proj",
            "block_sparse_moe.routed_expert_down_proj",
        }:
            return True
        if suffix == "self_attn.o_proj":
            return int(match[1]) in {3 + 4 * i for i in range(23)} | {92}
        return KimiK3._is_expert(module)

    def get_deploy_candidates(self, module_names: set[str]) -> set[str]:
        return {name for name in module_names if self._is_supported_module(name)}

    def validate_deploy_selection(self, selected: set[str]) -> None:
        unsupported = sorted(
            name for name in selected if not self._is_supported_module(name)
        )
        if unsupported:
            raise ValueError(
                f"Unsupported Kimi-K3 Deployment modules: {unsupported[:20]}"
            )
        experts = {name for name in selected if self._is_expert(name)}
        if experts:
            expected = {
                "language_model.model.layers."
                f"{layer}.block_sparse_moe.experts.{expert}.w{projection}"
                for layer in range(1, 93)
                for expert in range(self._num_experts())
                for projection in (1, 2, 3)
            }
            if experts != expected:
                raise ValueError(
                    "Kimi-K3 routed experts must select all native MXFP4 "
                    "w1/w2/w3 tensors"
                )

    def _num_experts(self):
        text_config = self.config.get("text_config", self.config)
        return int(text_config.get("num_experts", 0))

    def prepare_tensor_deploy_plan(self, weight_map):
        selection = self.args.deploy_selection
        names = self._canonical_weight_modules(weight_map)
        selected, counts = match_deploy_modules(names, selection)
        ignored = frozenset(
            name
            for name in names
            if any(fnmatchcase(name, pattern) for pattern in selection.ignore_layers)
        )
        selected -= set(ignored)
        specs = {
            name: get_structured_quant_spec(selection, name)
            for name in sorted(selected)
        }
        for name, spec in specs.items():
            get_quant_scheme(
                QuantSchemeKey(spec.quant_dtype, spec.weight_bits, spec.activation_bits)
            )
        self.validate_deploy_selection(set(specs))
        available = names
        if missing := set(specs) - available:
            raise ValueError(
                "Selected Kimi-K3 modules are absent from checkpoint: "
                f"{sorted(missing)[:20]}"
            )
        packed = {name for name in names if self._is_expert(name)}
        if packed - set(specs):
            raise ValueError(
                "All native Kimi-K3 routed expert tensors must be selected in "
                "quant_layers_config"
            )
        self._planned_specs = specs
        self.processed_modules.clear()
        weights = frozenset(names)
        mtp = frozenset(
            name for name in names if ".mtp." in name or name.startswith("mtp.")
        )
        return ResolvedDeployPlan(
            module_quant_specs=specs,
            selected_weight_keys=frozenset(name + ".weight" for name in specs),
            ignored_modules=ignored,
            unselected_weight_modules=frozenset(weights - set(specs)),
            match_counts=counts,
            mtp=MtpSelectionReport(mtp, mtp & ignored, mtp & set(specs)),
            unmatched_ignore_patterns=frozenset(
                pattern
                for pattern in selection.ignore_layers
                if counts.get(pattern, 0) == 0
            ),
        )

    def convert_tensorwise_shard(
        self, source_file, model_dir, original_weight_map, quant_layers, loaded_files
    ):
        del loaded_files
        path = resolve_safetensors_path(model_dir, source_file, self.safetensors_files)
        source = load_file(str(path), device="cpu")
        register_dtype()
        quant_device = getattr(self.args, "device", None) or "cpu"
        output = {}
        processed = set()
        for name, tensor in source.items():
            module = self._canonical_module(name)
            if name.endswith(_PACKED_SUFFIX):
                if module not in quant_layers:
                    raise ValueError(
                        f"Native MXFP4 expert is missing from deployment plan: {module}"
                    )
                spec = quant_layers[module]
                if not isinstance(spec, QuantSpec):
                    spec = self._planned_specs[module]
                if not isinstance(spec, QuantSpec) or (
                    spec.quant_dtype,
                    spec.weight_bits,
                    spec.activation_bits,
                ) != ("mxfp", 4, 8):
                    raise ValueError(
                        f"Kimi-K3 native expert {module} must use mxfp W4A8"
                    )
                output[module + ".weight"] = tensor
                processed.add(module)
            elif name.endswith(_SCALE_SUFFIX):
                if module not in quant_layers:
                    raise ValueError(
                        "Native MXFP4 expert scale is missing from deployment plan: "
                        f"{module}"
                    )
                output[module + ".weight_scale"] = tensor
            elif name.endswith(".weight") and module in quant_layers:
                spec = quant_layers[module]
                if not isinstance(spec, QuantSpec):
                    spec = self._planned_specs[module]
                if (
                    spec.quant_dtype != "mxfp"
                    or spec.weight_bits != 8
                    or spec.activation_bits != 8
                ):
                    raise ValueError(f"Kimi-K3 BF16 module {module} must use mxfp W8A8")
                # Safetensors is read from CPU, but the actual MXFP8 quantization
                # follows the requested deployment device. Quantized payloads are
                # moved back to CPU because safetensors files cannot store NPU tensors.
                quantized_tensor = tensor.to(device=quant_device)
                payload = quant_payload(
                    DTYPE_REGISTRY.get("mxfp"), name, quantized_tensor, 8
                )
                payload = {
                    payload_name: payload_tensor.detach().cpu()
                    for payload_name, payload_tensor in payload.items()
                }
                output.update(adapt_payload(name, payload, "mxfp", 8, 8))
                processed.add(module)
            else:
                output[name] = tensor
        destination = self.output_dir / source_file
        temporary = destination.with_name(f".{destination.name}.tmp")
        save_file(output, str(temporary), metadata={"format": "pt"})
        os.replace(temporary, destination)
        self.processed_modules.update(processed)
        self.output_tensor_bytes += sum(
            value.numel() * value.element_size() for value in output.values()
        )
        return {name: source_file for name in output}

    def validate_tensorwise_result(self, plan):
        if self.processed_modules != plan.selected_modules:
            raise ValueError(
                "Actual Kimi-K3 converted modules differ from the deployment plan"
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

    def make_description(self, output_tensor_names, selected_modules):
        description = {"group_size": 32}
        expected = set()
        for module in selected_modules:
            spec = self._planned_specs[module]
            scheme = get_quant_scheme(
                QuantSchemeKey(spec.quant_dtype, spec.weight_bits, spec.activation_bits)
            )
            for suffix in scheme.tensor_suffixes:
                name = module + suffix
                expected.add(name)
                if name not in output_tensor_names:
                    raise ValueError(f"Kimi-K3 deployment output is missing {name}")
                description[name] = scheme.model_quant_type
        description.update(
            {name: "FLOAT" for name in sorted(output_tensor_names - expected)}
        )
        return description
