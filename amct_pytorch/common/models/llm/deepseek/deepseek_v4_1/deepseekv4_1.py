# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""DeepSeek V4.1 deployment adapter; no model construction or PTQ support.

The legacy 950 format follows cann-recipes-infer's deepseek_v4_1/utils
converter: preserve weights, expand UE8M0 scales from 32x32 to 1x32.
Only weight/scale pairs in the same shard are considered.
"""

import copy
import json
import os
from fnmatch import fnmatchcase
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from amct_pytorch.common.models import MODEL_REGISTRY
from amct_pytorch.common.models.llm.common.weight_path_validation import (
    collect_safetensors_files,
    resolve_safetensors_path,
)

BLOCK = 32


@MODEL_REGISTRY.register(
    name="deepseek_v4_1",
    task="llm",
    family="deepseek",
    description="DeepSeek V4.1 tensor deployment only",
)
class DeepseekV41:
    """Lightweight registry entry, deliberately independent of BaseModel/PTQ."""

    def __init__(self, args):
        self.validate_deploy_args(args)
        self.args = args
        self.quant_dtype = args.quant_dtype
        self.model_path = Path(args.model)
        self.output_dir = Path(args.output_dir)
        source, output = self.model_path.resolve(), self.output_dir.resolve()
        if source == output or source in output.parents or output in source.parents:
            raise ValueError(
                "DeepSeek V4.1 output and source directories must not overlap"
            )
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            raise ValueError("DeepSeek V4.1 output directory must be new or empty")
        with open(self.model_path / "config.json", encoding="utf-8") as stream:
            self.config = json.load(stream)
        self.safetensors_files = collect_safetensors_files(self.model_path)
        self.processed_modules = set()

    @staticmethod
    def _validate_platform(args):
        platform = getattr(args, "deploy_platform", None)
        expected = {"A3": "int", "ascend950": "mxfp"}
        if platform not in expected:
            raise ValueError("DeepSeek V4.1 requires --deploy_platform A3 or ascend950")
        if getattr(args, "quant_dtype", None) != expected[platform]:
            raise ValueError(
                f"DeepSeek V4.1 {platform} requires --quant_dtype {expected[platform]}"
            )
        return platform

    @classmethod
    def validate_deploy_args(cls, args):
        from amct_pytorch.common.models.llm.common.deploy_selection import (
            DeploySelection,
            load_deploy_selection,
        )
        from amct_pytorch.quantization.bit_policy import BitPolicy

        platform = cls._validate_platform(args)
        if getattr(args, "granularity", None) != "tensor":
            raise NotImplementedError(
                "DeepSeek V4.1 supports tensor deployment only; PTQ is not implemented"
            )
        deploy_format = getattr(args, "deploy_format", "legacy")
        if deploy_format == "ascend":
            raise NotImplementedError(
                "DeepSeek V4.1 ascend deployment is not implemented"
            )
        if deploy_format != "legacy":
            raise ValueError(f"Unsupported deploy_format: {deploy_format}")
        if platform == "A3":
            raise NotImplementedError(
                "DeepSeek V4.1 A3 INT conversion is not implemented"
            )
        if getattr(args, "bit_config", None):
            raise ValueError(
                "DeepSeek V4.1 950 preserves source bits; --bit_config is not supported"
            )
        for key in (
            "algos",
            "attn_linear_param_dir",
            "attn_cache_param_dir",
            "moe_mlp_param_dir",
        ):
            if getattr(args, key, None):
                raise ValueError(f"DeepSeek V4.1 deployment does not consume --{key}")
        path = getattr(args, "quant_layers_config", None)
        if path:
            args.deploy_selection = load_deploy_selection(path)
        elif not isinstance(getattr(args, "deploy_selection", None), DeploySelection):
            raise ValueError("DeepSeek V4.1 requires --quant_layers_config")
        args.bit_policy = BitPolicy()

    @staticmethod
    def _is_block_fp8(weight_dtype, weight_shape, scale_dtype, scale_shape):
        return (
            weight_dtype == "F8_E4M3"
            and scale_dtype == "F8_E8M0"
            and len(weight_shape) == 2
            and len(scale_shape) == 2
            and all(size > 0 and size % BLOCK == 0 for size in weight_shape)
            and tuple(scale_shape) == tuple(size // BLOCK for size in weight_shape)
        )

    def prepare_tensor_deploy_plan(self, weight_map):
        """Inspect shard headers before writing, then select only actual FP8 work."""
        from amct_pytorch.common.models.llm.common.deploy_selection import (
            MtpSelectionReport,
            ResolvedDeployPlan,
            match_deploy_modules,
        )

        names = {name.rsplit(".", 1)[0] for name in weight_map}
        selection = self.args.deploy_selection
        selected, counts = match_deploy_modules(names, selection)
        candidates = set()
        for filename in sorted(set(weight_map.values())):
            path = resolve_safetensors_path(
                self.model_path, filename, self.safetensors_files
            )
            with safe_open(str(path), framework="pt", device="cpu") as shard:
                keys = set(shard.keys())
                for key in keys:
                    if not key.endswith(".scale"):
                        continue
                    base = key.removesuffix(".scale")
                    if base + ".weight" not in keys:
                        continue
                    weight = shard.get_slice(base + ".weight")
                    scale = shard.get_slice(key)
                    if self._is_block_fp8(
                        weight.get_dtype(),
                        weight.get_shape(),
                        scale.get_dtype(),
                        scale.get_shape(),
                    ):
                        candidates.add(base)
        if missing := candidates - selected:
            raise ValueError(
                f"FP8 32x32 modules not selected by JSON: {sorted(missing)}"
            )
        # Fail on invalid config before any checkpoint output is created.
        self._make_legacy_quant_config()
        self.processed_modules.clear()
        ignored = frozenset(
            name
            for name in names
            if any(fnmatchcase(name, p) for p in selection.ignore_layers)
        )
        weights = {
            key.removesuffix(".weight") for key in weight_map if key.endswith(".weight")
        }
        mtp = frozenset(
            name for name in names if name.startswith("mtp.") or ".mtp." in name
        )
        return ResolvedDeployPlan(
            selected_modules=frozenset(candidates),
            selected_weight_keys=frozenset(name + ".weight" for name in candidates),
            module_bits={name: (8, 8) for name in candidates},
            ignored_modules=ignored,
            unselected_weight_modules=frozenset(weights - candidates),
            match_counts=counts,
            mtp=MtpSelectionReport(mtp, mtp & ignored, mtp & candidates),
            unmatched_ignore_patterns=frozenset(
                p for p in selection.ignore_layers if counts[p] == 0
            ),
        )

    def convert_tensorwise_shard(
        self, source_file, model_dir, original_weight_map, quant_layers, loaded_files
    ):
        """Same arguments and {tensor_name: shard_name} return as the workflow."""
        platform = self._validate_platform(self.args)
        convert = (
            self._convert_tensorwise_shard_a3
            if platform == "A3"
            else self._convert_tensorwise_shard_950
        )
        return convert(
            source_file, model_dir, original_weight_map, quant_layers, loaded_files
        )

    def _convert_tensorwise_shard_a3(
        self, source_file, model_dir, original_weight_map, quant_layers, loaded_files
    ):
        raise NotImplementedError("DeepSeek V4.1 A3 INT conversion is not implemented")

    def _convert_tensorwise_shard_950(
        self, source_file, model_dir, original_weight_map, quant_layers, loaded_files
    ):
        path = resolve_safetensors_path(model_dir, source_file, self.safetensors_files)
        tensors = load_file(str(path), device="cpu")
        processed = set()
        for name, scale in list(tensors.items()):
            if not name.endswith(".scale"):
                continue
            base = name.removesuffix(".scale")
            weight = tensors.get(base + ".weight")
            if weight is None or base not in quant_layers:
                continue
            if (
                weight.dtype == torch.float8_e4m3fn
                and scale.dtype == torch.float8_e8m0fnu
                and self._is_block_fp8("F8_E4M3", weight.shape, "F8_E8M0", scale.shape)
            ):
                tensors[name] = (
                    scale.view(torch.uint8)
                    .repeat_interleave(BLOCK, dim=0)
                    .view(scale.dtype)
                )
                processed.add(base)
        destination = self.output_dir / source_file
        temporary = destination.with_name(f".{destination.name}.tmp")
        save_file(tensors, str(temporary), metadata={"format": "pt"})
        os.replace(temporary, destination)
        self.processed_modules.update(processed)
        return {name: source_file for name in tensors}

    def validate_tensorwise_result(self, plan):
        """Return actual converted modules for the workflow's completion check."""
        if self.processed_modules != plan.selected_modules:
            raise ValueError(
                "Actual FP8 scale conversions differ from the deployment plan"
            )
        return self.processed_modules

    @staticmethod
    def _source_layers(fields, short_name, id_name):
        # 临时兼容参考转换脚本与官方模型 config 的字段命名差异：
        # kv_source_layers / kv_source_layer_ids，以及
        # index_source_layers / index_source_layer_ids。
        # TODO: 待字段命名统一后，删除此兼容接口并直接读取统一字段。
        short, ids = fields.get(short_name), fields.get(id_name)
        if short is not None and ids is not None and short != ids:
            raise ValueError(f"Conflicting config fields: {short_name} / {id_name}")
        value = short if short is not None else ids
        if not isinstance(value, list) or any(
            type(i) is not int or i < 0 for i in value
        ):
            raise ValueError(
                f"config.json requires {short_name} or {id_name} as a list of layer indices"
            )
        return value

    def _make_legacy_quant_config(self):
        from amct_pytorch.common.models.llm.common.deploy_export import (
            generate_quant_config,
        )

        fields = self.config.get("text_config") or self.config
        kv = self._source_layers(fields, "kv_source_layers", "kv_source_layer_ids")
        index = self._source_layers(
            fields, "index_source_layers", "index_source_layer_ids"
        )
        ignores = []
        for i in index:
            ignores.extend(
                f"layers.{i}.attn.indexer.{name}"
                for name in ("weights_proj", "compressor.wgate", "compressor.wkv")
            )
        for i in kv:
            ignores.extend(
                f"layers.{i}.attn.compressor.{name}" for name in ("wgate", "wkv")
            )
        ignores.append("head")
        config = generate_quant_config(
            {
                name: {"num_bits": bits, "type": "float"}
                for name, bits in (
                    ("kv_cache_scheme", 8),
                    ("comp_cache_scheme", 4),
                    ("li_cache_scheme", 4),
                )
            },
            ignores,
            is_mx=True,
            bits_scheme=[
                {"targets": ["Linear"], "w_bits": 8, "a_bits": 8},
                {"targets": ["MoEGMM"], "w_bits": 4, "a_bits": 8},
            ],
        )
        # Match the reference converter's activation observer exactly.
        for group in config["config_groups"].values():
            group["input_activations"]["observer"] = "memoryless"
        return config

    def refresh_config(self, quant_ignore_layers, *, tensor_plan=None):
        platform = self._validate_platform(self.args)
        refresh = (
            self._refresh_config_a3 if platform == "A3" else self._refresh_config_950
        )
        return refresh(quant_ignore_layers, tensor_plan=tensor_plan)

    def _refresh_config_a3(self, quant_ignore_layers, *, tensor_plan=None):
        raise NotImplementedError("DeepSeek V4.1 A3 config refresh is not implemented")

    def _refresh_config_950(self, quant_ignore_layers, *, tensor_plan=None):
        # JSON selects transformations, not FP4/FP8 quantization status. In
        # particular unprocessed FP4 experts must NOT become config ignores.
        config = copy.deepcopy(self.config)
        # Reinsert at the end, matching the reference converter's JSON order.
        config.pop("quantization_config", None)
        config["quantization_config"] = self._make_legacy_quant_config()
        temporary = self.output_dir / ".config.json.tmp"
        temporary.write_text(json.dumps(config, indent=2), encoding="utf-8")
        os.replace(temporary, self.output_dir / "config.json")

    def make_ascend_description(self, output_tensor_names, selected_modules):
        platform = self._validate_platform(self.args)
        describe = (
            self._make_ascend_description_a3
            if platform == "A3"
            else self._make_ascend_description_950
        )
        return describe(output_tensor_names, selected_modules)

    def _make_ascend_description_a3(self, output_tensor_names, selected_modules):
        raise NotImplementedError(
            "DeepSeek V4.1 A3 ascend description is not implemented"
        )

    def _make_ascend_description_950(self, output_tensor_names, selected_modules):
        raise NotImplementedError(
            "DeepSeek V4.1 950 ascend description is not implemented"
        )
