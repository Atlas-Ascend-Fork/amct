# Copyright (c) 2026 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared tensorwise deployment implementation for DeepSeek adapters."""

from __future__ import annotations

import copy
import json
import os
from fnmatch import fnmatchcase
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from amct_pytorch.common.models.llm.common.deployment.deploy_export import (
    adapt_payload,
    convert_state_dict,
    quant_payload,
)
from amct_pytorch.common.models.llm.common.weight_path_validation import (
    resolve_safetensors_path,
)
from amct_pytorch.quantization.dtypes import DTYPE_REGISTRY, register_dtype
from amct_pytorch.quantization.dtypes.hifp_impl import hif8_pack

BLOCK = 32
MAX_OPEN_SCALE_FILES = 8
GROUPED_SUFFIX = "attn.wo_a"


class DeepSeekTensorwiseDeployCommon:
    """Mixin containing the shared DeepSeek tensorwise deploy contract."""

    def _is_hif8(self):
        return getattr(self.args, "quant_dtype", None) == "hifp"

    @classmethod
    def validate_deploy_source(cls, args):
        """Allow the official FP8/FP4 source for INT deployment on either platform."""
        if getattr(args, "deploy_platform", None) not in {"A3", "ascend950"} or getattr(
            args, "quant_dtype", None
        ) not in {"int", "hifp"}:
            return False
        source = Path(args.model)
        try:
            config = json.loads((source / "config.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        markers = (
            "quant_model_description.json",
            "quant_model_weights.safetensors.index.json",
            "rot.safetensors",
        )
        quant = config.get("quantization_config")
        return (
            isinstance(quant, dict)
            and quant.get("quant_method") == "fp8"
            and not config.get("is_rot_used")
            and not any((source / name).exists() for name in markers)
        )

    @classmethod
    def validate_deploy_args(cls, args):
        model_name = getattr(args, "model_name", "")
        if not model_name:
            raise ValueError("--model_name is required")
        cli_dtype = getattr(args, "quant_dtype", None)
        if (
            getattr(args, "granularity", None) == "tensor"
            and cli_dtype
            and cli_dtype not in {"int", "hifp"}
        ):
            raise NotImplementedError(
                f"{model_name} tensor deployment with quant_dtype '{cli_dtype}' "
                "is temporarily unsupported"
            )

        from amct_pytorch.common.models.llm.common.deployment.deploy_selection import (
            DeploySelection,
            load_deploy_selection,
        )
        from amct_pytorch.quantization.bit_policy import BitPolicy

        if getattr(args, "granularity", None) != "tensor":
            raise NotImplementedError(
                f"{model_name} supports tensor deployment only; PTQ is not implemented"
            )
        for key in (
            "algos",
            "attn_linear_param_dir",
            "attn_cache_param_dir",
            "moe_mlp_param_dir",
        ):
            if getattr(args, key, None):
                raise ValueError(f"{model_name} deployment does not consume --{key}")
        path = getattr(args, "quant_layers_config", None)
        if path:
            args.deploy_selection = load_deploy_selection(path)
        elif not isinstance(getattr(args, "deploy_selection", None), DeploySelection):
            raise ValueError(f"{model_name} requires --quant_layers_config")
        selection = args.deploy_selection
        if selection.uses_structured_rules:
            from amct_pytorch.common.models.llm.common.deployment.schemes import (
                validate_selection_quant_scheme,
            )

            scheme = validate_selection_quant_scheme(selection)
            if cli_dtype and cli_dtype != scheme.quant_dtype:
                raise ValueError(
                    f"{model_name} --quant_dtype does not match "
                    "quant_layers_config: "
                    f"{cli_dtype!r} != {scheme.quant_dtype!r}"
                )
            selected_dtype = scheme.quant_dtype
        else:
            selected_dtype = cli_dtype
        args.quant_dtype = selected_dtype
        if selected_dtype not in {"int", "hifp"}:
            raise NotImplementedError(
                f"{model_name} tensor deployment with quant_dtype "
                f"'{selected_dtype}' is temporarily unsupported"
            )
        args.bit_policy = BitPolicy()

    def prepare_tensor_deploy_plan(self, weight_map):
        """Inspect shard headers before writing, then select actual work."""
        from amct_pytorch.common.models.llm.common.deployment.deploy_selection import (
            MtpSelectionReport,
            QuantSpec,
            ResolvedDeployPlan,
            get_structured_quant_spec,
            match_deploy_modules,
        )
        from amct_pytorch.common.models.llm.common.deployment.schemes import (
            validate_platform_quant_scheme,
        )

        names = {name.rsplit(".", 1)[0] for name in weight_map}
        selection = self.args.deploy_selection
        selected, counts = match_deploy_modules(names, selection)
        candidates = set()
        selected = set(selected)
        platform = getattr(self.args, "deploy_platform", None)
        is_int = self.args.quant_dtype == "int"
        if selection.uses_structured_rules:
            validate_platform_quant_scheme(selection, platform)
        weight_keys = {key for key in weight_map if key.endswith(".weight")}
        header_keys = weight_keys | {
            key.removesuffix(".weight") + ".scale"
            for key in weight_keys
            if key.removesuffix(".weight") + ".scale" in weight_map
        }
        header_keys_by_file = {}
        for key in header_keys:
            header_keys_by_file.setdefault(weight_map[key], set()).add(key)
        headers = {}
        for filename in sorted(set(weight_map.values())):
            path = resolve_safetensors_path(
                self.model_path, filename, self.safetensors_files
            )
            with safe_open(str(path), framework="pt", device="cpu") as shard:
                present_keys = set(shard.keys()) & header_keys_by_file.get(
                    filename, set()
                )
                for key in present_keys:
                    view = shard.get_slice(key)
                    headers[key] = (view.get_dtype(), view.get_shape())
        for key in sorted(weight_keys):
            base = key.removesuffix(".weight")
            weight_header = headers.get(key)
            if weight_header is None:
                continue
            if is_int and base in selected:
                weight_shape = weight_header[1]
                if len(weight_shape) != 2 or any(size <= 0 for size in weight_shape):
                    raise ValueError(
                        f"Selected INT module is not a nonempty 2-D weight: {base}"
                    )
                candidates.add(base)
                continue
            if (
                self.args.quant_dtype == "hifp"
                and base in selected
                and len(weight_header[1]) == 2
                and all(size > 0 for size in weight_header[1])
            ):
                candidates.add(base)
        if is_int and (missing := selected - candidates):
            raise ValueError(
                f"Selected INT modules are not nonempty 2-D weights: {sorted(missing)}"
            )
        if self._is_hif8() and (missing := selected - candidates):
            raise ValueError(
                "Selected HiFloat8 modules are not nonempty 2-D weights: "
                f"{sorted(missing)}"
            )
        self.processed_modules.clear()
        ignored = frozenset(
            name
            for name in names
            if any(fnmatchcase(name, pattern) for pattern in selection.ignore_layers)
        )
        weights = {
            key.removesuffix(".weight") for key in weight_map if key.endswith(".weight")
        }
        mtp = frozenset(
            name for name in names if name.startswith("mtp.") or ".mtp." in name
        )
        if selection.uses_structured_rules:
            module_quant_specs = {
                name: get_structured_quant_spec(selection, name) for name in selected
            }
        else:
            module_quant_specs = {
                name: QuantSpec(self.args.quant_dtype, 8, 8) for name in selected
            }
        return ResolvedDeployPlan(
            module_quant_specs=module_quant_specs,
            selected_weight_keys=frozenset(name + ".weight" for name in selected),
            ignored_modules=ignored,
            unselected_weight_modules=frozenset(weights - selected),
            match_counts=counts,
            mtp=MtpSelectionReport(mtp, mtp & ignored, mtp & candidates),
            unmatched_ignore_patterns=frozenset(
                p for p in selection.ignore_layers if counts[p] == 0
            ),
        )

    def convert_tensorwise_shard(
        self, source_file, model_dir, original_weight_map, quant_layers, loaded_files
    ):
        target_dtype = self.args.quant_dtype
        if target_dtype not in {"int", "hifp"}:
            model_name = getattr(self.args, "model_name", type(self).__name__)
            raise NotImplementedError(
                f"{model_name} tensor deployment does not support quant_dtype "
                f"{target_dtype!r}"
            )
        return self._convert_tensorwise_shard_common(
            source_file,
            model_dir,
            original_weight_map,
            quant_layers,
            loaded_files,
            target_dtype,
        )

    def _convert_tensorwise_shard_common(
        self,
        source_file,
        model_dir,
        original_weight_map,
        quant_layers,
        loaded_files,
        target_dtype,
    ):
        del loaded_files
        source_path = resolve_safetensors_path(
            model_dir, source_file, self.safetensors_files
        )
        current_state_dict = load_file(str(source_path), device="cpu")
        scale_files = {}
        new_state_dict = {}
        processed = set()

        for weight_name, source_weight in current_state_dict.items():
            if weight_name.endswith(".scale"):
                if (
                    weight_name.removesuffix(".scale") + ".weight"
                    in original_weight_map
                ):
                    continue
                new_state_dict[weight_name] = source_weight
                continue

            weight = source_weight
            if weight_name.endswith(".weight") and weight.ndim == 2:
                scale_name = weight_name.removesuffix(".weight") + ".scale"
                if scale_name in original_weight_map:
                    scale_file = original_weight_map[scale_name]
                    scale = self._read_source_scale(
                        model_dir, scale_file, scale_name, scale_files
                    )
                    prepared_scale = self._prepare_int_scale(weight, scale)
                    weight = self._decode_source_weight(
                        weight,
                        weight_name,
                        prepared_scale,
                        original_weight_map,
                        model_dir,
                    )
                    if (
                        target_dtype == "int"
                        and weight.is_floating_point()
                        and weight.dtype != torch.bfloat16
                    ):
                        weight = weight.to(torch.bfloat16)
                elif not weight.is_floating_point() or weight.dtype in {
                    torch.float8_e4m3fn,
                    torch.float8_e5m2,
                }:
                    raise ValueError(
                        f"Missing source scale tensor: {scale_name} for weight "
                        f"{weight_name} (dtype={weight.dtype}). FP8 and non-floating-point "
                        "source weights require a scale tensor for dequantization, "
                        "but no corresponding scale entry was found in original_weight_map."
                    )

            module_name = (
                weight_name.removesuffix(".weight")
                if weight_name.endswith(".weight")
                else None
            )
            if module_name in quant_layers:
                if target_dtype == "int":
                    state_dict = self._export_int_payload(
                        weight_name, weight, quant_layers[module_name]
                    )
                else:
                    state_dict = self._export_hif8_payload(
                        weight_name, weight.to(torch.float32)
                    )
                new_state_dict.update(state_dict)
                processed.add(module_name)
            else:
                new_state_dict[weight_name] = weight

        destination = self.output_dir / source_file
        temporary = destination.with_name(f".{destination.name}.tmp")
        save_file(new_state_dict, str(temporary), metadata={"format": "pt"})
        os.replace(temporary, destination)
        self.processed_modules.update(processed)
        self.output_tensor_bytes = getattr(self, "output_tensor_bytes", 0) + sum(
            tensor.numel() * tensor.element_size() for tensor in new_state_dict.values()
        )
        return {name: source_file for name in new_state_dict}

    def _decode_source_weight(
        self,
        weight,
        weight_name,
        scale,
        original_weight_map,
        model_dir,
    ):
        return self._dequantize_source_weight(
            weight,
            scale,
            weight_name,
            original_weight_map,
            model_dir,
        )

    @staticmethod
    def _dequantize_source_weight(
        weight, scale, weight_name, original_weight_map, model_dir
    ):
        # fp8, v4_block=128, v41_block=32
        if weight.dtype == torch.float8_e4m3fn and not torch.is_floating_point(scale):
            if scale.ndim != 2 or weight.ndim != 2:
                raise ValueError(
                    f"DSV4 FP8 source {weight_name} requires 2-D weight and scale"
                )
            rows, cols = weight.shape
            if rows % scale.shape[0] or cols % scale.shape[1]:
                raise ValueError(
                    f"DSV4 FP8 source scale {weight_name} does not tile weight shape "
                    f"{tuple(weight.shape)} with scale shape {tuple(scale.shape)}"
                )
            row_block = rows // scale.shape[0]
            col_block = cols // scale.shape[1]
            scale_float = torch.exp2(scale.to(torch.float32) - 127)
            dense_scale = scale_float.repeat_interleave(
                row_block, dim=0
            ).repeat_interleave(col_block, dim=1)
            return (weight.to(torch.float32) * dense_scale).to(torch.bfloat16)
        # mxfp4 (moe)
        return convert_state_dict(
            weight,
            weight_name,
            weight_name.removesuffix(".weight") + ".scale",
            original_weight_map,
            model_dir,
            {},
            block_size=BLOCK,
            safetensors_files=None,
        )

    def _read_source_scale(self, model_dir, scale_file, scale_name, open_files):
        entry = open_files.pop(scale_file, None)
        if entry is None:
            scale_path = resolve_safetensors_path(
                model_dir, scale_file, self.safetensors_files
            )
            handle = safe_open(str(scale_path), framework="pt", device="cpu")
            entry = (handle, frozenset(handle.keys()))
            while len(open_files) >= MAX_OPEN_SCALE_FILES:
                open_files.pop(next(iter(open_files)))
        open_files[scale_file] = entry
        handle, names = entry
        if scale_name not in names:
            raise ValueError(f"Missing source scale tensor: {scale_name}")
        return handle.get_tensor(scale_name)

    @staticmethod
    def _prepare_int_scale(weight, scale):
        del weight
        if scale.dtype == torch.float8_e8m0fnu:
            scale = scale.view(torch.uint8)
        return scale

    @staticmethod
    def _export_int_payload(weight_name, weight, bit):
        register_dtype()
        payload = quant_payload(DTYPE_REGISTRY.get("int"), weight_name, weight, bit)
        return adapt_payload(weight_name, payload)

    def _export_hif8_payload(self, weight_name, weight):
        module_name = weight_name.removesuffix(".weight")
        scale_name = module_name + ".weight_scale"
        text_config = self.config.get("text_config", self.config)
        grouped = module_name.endswith(GROUPED_SUFFIX)
        if grouped:
            groups = self._required_int(text_config, "o_groups")
            rank = self._required_int(text_config, "o_lora_rank")
            rows, columns = weight.shape
            if rows != groups * rank:
                raise ValueError(
                    f"{module_name}: {rows} rows, expected o_groups * o_lora_rank"
                    f" = {groups} * {rank}"
                )
            weight = (
                weight.view(groups, rank, columns)
                .transpose(0, 1)
                .reshape(rank, groups * columns)
            )
        codes, scale = hif8_pack(weight)
        payload = {weight_name: codes, scale_name: scale}
        if grouped:
            payload[weight_name] = (
                payload[weight_name]
                .view(rank, groups, columns)
                .transpose(0, 1)
                .reshape(rows, columns)
                .contiguous()
            )
            payload[scale_name] = (
                payload[scale_name]
                .view(1, rank)
                .expand(groups, rank)
                .reshape(rows, 1)
                .contiguous()
            )
        return payload

    @staticmethod
    def _required_int(text_config, key):
        value = text_config.get(key)
        if type(value) is not int:
            raise ValueError(
                f"config.json text_config requires integer '{key}' for HiFloat8 "
                f"export, got {value!r}"
            )
        return value

    def validate_tensorwise_result(self, plan):
        if self.processed_modules != plan.selected_modules:
            raise ValueError("Actual converted modules differ from the deployment plan")
        return self.processed_modules

    def refresh_config(self, quant_ignore_layers, *, tensor_plan=None):
        del quant_ignore_layers, tensor_plan
        config = copy.deepcopy(self.config)
        config.pop("quantization_config", None)
        temporary = self.output_dir / ".config.json.tmp"
        temporary.write_text(json.dumps(config, indent=2), encoding="utf-8")
        os.replace(temporary, self.output_dir / "config.json")
