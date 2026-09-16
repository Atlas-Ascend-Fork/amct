# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Deploy-only module selection; independent of model construction and PTQ."""

import json
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path


@dataclass(frozen=True)
class DeploySelection:
    """Case-sensitive checkpoint module globs, without the .weight suffix."""

    quant_layers: tuple[str, ...]
    ignore_layers: tuple[str, ...] = ()


@dataclass(frozen=True)
class AscendDeployPlan:
    """Resolved checkpoint selection shared by payload and metadata export."""

    selected_modules: frozenset[str]
    selected_weight_keys: frozenset[str]
    match_counts: dict[str, int]
    excluded_mtp: frozenset[str]


def build_ascend_deploy_plan(
    weight_map: dict[str, str], selection: DeploySelection, adapter
) -> AscendDeployPlan:
    """Resolve against actual tensor names, then validate model constraints.

    Inventory includes modules with non-weight parameters so unsupported
    explicit selections are diagnosed rather than silently discarded.
    excluded_mtp records matches removed by the mandatory MTP exclusion, after
    the user's ignore rules have been applied.
    """
    if hasattr(adapter, "expand_ascend_deploy_weight_map"):
        weight_map = adapter.expand_ascend_deploy_weight_map(weight_map)
    names = {key.rsplit(".", 1)[0] for key in weight_map}
    selected, counts = match_deploy_modules(names, selection)
    excluded_mtp = adapter.get_ascend_deploy_mtp_modules(selected)
    selected -= excluded_mtp
    if not selected:
        raise ValueError("No modules selected after mandatory MTP exclusion")
    candidates = adapter.get_ascend_deploy_candidates(
        {key.removesuffix(".weight") for key in weight_map if key.endswith(".weight")}
    )
    unsupported = selected - candidates
    if unsupported:
        raise ValueError(
            f"Unsupported Ascend deployment modules: {sorted(unsupported)}"
        )
    adapter.validate_ascend_deploy_selection(selected)
    return AscendDeployPlan(
        selected_modules=frozenset(selected),
        selected_weight_keys=frozenset(f"{name}.weight" for name in selected),
        match_counts=counts,
        excluded_mtp=frozenset(excluded_mtp),
    )


def load_deploy_selection(path: Path) -> DeploySelection:
    """Read a selection JSON, rejecting typos and invalid rule types.

    Duplicate patterns are collapsed in first-occurrence order. Module-name
    matching and unmatched-pattern errors require the checkpoint inventory and
    are handled by match_deploy_modules instead.
    """
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read quant_layers_config {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path}: selection must be a JSON object")
    unknown = data.keys() - {"quant_layers", "ignore_layers"}
    if unknown:
        raise ValueError(f"{path}: unknown selection fields: {sorted(unknown)}")
    if not data.get("quant_layers"):
        raise ValueError(f"{path}: quant_layers must be a non-empty list")

    rules = {}
    for key in ("quant_layers", "ignore_layers"):
        values = data.get(key, [])
        if not isinstance(values, list) or any(
            not isinstance(value, str) or not value.strip() for value in values
        ):
            raise ValueError(f"{path}: {key} must be a list of non-empty strings")
        rules[key] = tuple(dict.fromkeys(values))
    return DeploySelection(**rules)


def match_deploy_modules(
    names: set[str], selection: DeploySelection
) -> tuple[set[str], dict[str, int]]:
    """Resolve full-name globs with ignore precedence and raw hit counts.

    Counts include ignored matches and are keyed by unique pattern. Ignore
    patterns may match non-quantizable modules (e.g. norm/head). Candidate and
    fusion validation belongs to the model adapter, not this matcher.
    """
    counts = {}
    selected = set()
    ignored = set()
    for patterns, matches in (
        (selection.quant_layers, selected),
        (selection.ignore_layers, ignored),
    ):
        for pattern in dict.fromkeys(patterns):
            found = {name for name in names if fnmatchcase(name, pattern)}
            counts[pattern] = len(found)
            matches.update(found)
    unmatched = sorted(pattern for pattern, count in counts.items() if count == 0)
    if unmatched:
        raise ValueError(f"Selection patterns matched no modules: {unmatched}")
    selected -= ignored
    if not selected:
        raise ValueError(
            "No modules selected after applying quant_layers and ignore_layers"
        )
    return selected, counts


def validate_ascend_deploy_args(args) -> None:
    """Validate deploy mode without loading weights or altering legacy options.

    This checks the W8A8 default; the model adapter must additionally validate
    effective per-module bit-policy overrides when resolving selected modules.
    Source tensors/config and output-directory checks belong to export preflight.
    """
    deploy_format = getattr(args, "deploy_format", "legacy")
    selection_path = getattr(args, "quant_layers_config", None)
    if deploy_format not in {"legacy", "ascend"}:
        raise ValueError(f"Unsupported deploy_format: {deploy_format}")
    if deploy_format == "legacy":
        if selection_path is not None:
            raise ValueError("quant_layers_config requires --deploy_format ascend")
        return
    if not selection_path:
        raise ValueError("Ascend deployment requires --quant_layers_config")
    if getattr(args, "model_name", None) not in {"glm5_2", "qwen3_6_moe"}:
        raise ValueError(
            "Ascend deployment requires --model_name one of ['glm5_2', 'qwen3_6_moe']"
        )
    for key, expected in (
        ("granularity", "tensor"),
        ("quant_dtype", "int"),
    ):
        if getattr(args, key, None) != expected:
            raise ValueError(f"Ascend deployment requires --{key} {expected}")
    policy = getattr(args, "bit_policy", None)
    if policy is None or (policy.w_bits, policy.a_bits) != (8, 8):
        raise ValueError("Ascend deployment requires W8A8 defaults in --bit_config")
    for key in (
        "algos",
        "attn_linear_param_dir",
        "attn_cache_param_dir",
        "moe_mlp_param_dir",
    ):
        if getattr(args, key, None):
            raise ValueError(f"Ascend tensor deployment does not consume --{key}")
