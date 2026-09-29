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
import re

from loguru import logger
import yaml

from amct_pytorch.quantization.bit_policy import BitPolicy


# Transitional allowlist while adapters migrate to JSON tensor deployment.
TENSOR_DEPLOY_CONFIG_MODELS = frozenset({"deepseek_v4_1", "glm5_2", "qwen3_6_moe"})


def requires_tensor_deploy_config(args):
    """Return whether the model uses the JSON tensor deployment contract."""
    return getattr(args, "model_name", None) in TENSOR_DEPLOY_CONFIG_MODELS


@dataclass(frozen=True)
class DeploySelection:
    """Case-sensitive checkpoint module globs, without the .weight suffix."""

    quant_layers: tuple[str, ...]
    ignore_layers: tuple[str, ...] = ()


@dataclass(frozen=True)
class MtpSelectionReport:
    """Observed MTP selection; reporting never changes the selected set."""

    modules: frozenset[str]
    ignored: frozenset[str]
    selected: frozenset[str]

    @property
    def status(self):
        if not self.modules:
            return "not_present"
        if self.ignored == self.modules:
            return "fully_ignored"
        if self.ignored:
            return "partially_ignored"
        return "selected" if self.selected else "not_selected"


@dataclass(frozen=True)
class ResolvedDeployPlan:
    """Resolved checkpoint selection shared by conversion and export metadata.

    Module names are full checkpoint paths without the final parameter suffix
    (for example, ``model.layers.0.self_attn.o_proj``).

    Attributes:
        selected_modules: Modules matched by quant_layers after ignore_layers
            takes precedence. Every selected module has a .weight tensor.
        selected_weight_keys: Exact checkpoint keys to quantize, formed by
            appending .weight to each selected module name. Biases are excluded.
        module_bits: Selected module name to (w_bits, a_bits), resolved from
            BitPolicy. The current tensor export contract requires (8, 8).
        ignored_modules: Existing parameter-bearing modules matched explicitly
            by JSON ignore_layers, including matches outside quant_layers.
            This is not the set of all modules left unquantized.
        unselected_weight_modules: All checkpoint modules with .weight tensors
            minus selected_modules. Legacy metadata combines this set with
            ignored_modules so unselected weights are not advertised as quantized.
        match_counts: Raw checkpoint module hit count for each unique JSON
            pattern, before ignore precedence. Zero-hit ignore rules remain here.
        mtp: Observed MTP modules and their ignored/selected subsets, used for
            validation and logging without changing selection.
        unmatched_ignore_patterns: JSON ignore patterns with zero checkpoint
            module matches; retained for warning logs and configuration auditing.
    """

    selected_modules: frozenset[str]
    selected_weight_keys: frozenset[str]
    module_bits: dict[str, tuple[int, int]]
    ignored_modules: frozenset[str]
    unselected_weight_modules: frozenset[str]
    match_counts: dict[str, int]
    mtp: MtpSelectionReport
    unmatched_ignore_patterns: frozenset[str] = frozenset()

    @property
    def quant_layers(self):
        """Return selected module names mapped to weight bits for conversion."""
        return {name: bits[0] for name, bits in self.module_bits.items()}


def validate_tensor_bit_config(cfg):
    """Strict tensor-only contract; do not change shared BitPolicy defaults."""
    if not isinstance(cfg, dict) or not cfg:
        raise ValueError("Tensor W8A8 deployment requires a nonempty --bit_config")
    allowed = {"attn-linear", "mlp", "moe", "w_bits", "a_bits"}
    unknown = cfg.keys() - allowed
    if unknown:
        raise ValueError(f"Unknown tensor bit_config fields: {sorted(unknown)}")

    def check(node, path):
        if not isinstance(node, dict):
            raise ValueError(f"{path}: expected a bit_config mapping")
        for key, value in node.items():
            full = f"{path}.{key}" if path else key
            if key in {"w_bits", "a_bits"}:
                if type(value) is not int or value != 8:
                    raise ValueError(
                        f"{full}={value}: tensor deployment currently supports only W8A8; "
                        "W4 export is not implemented in this path"
                    )
            elif key in {"w_bit", "a_bit"} or not isinstance(value, dict):
                raise ValueError(f"Unknown bit_config field {full}; use w_bits/a_bits")
            else:
                check(value, full)
        if ("w_bits" in node) != ("a_bits" in node):
            raise ValueError(f"{path}: set both w_bits and a_bits")

    check(cfg, "")


def _validate_model_deploy_args(args):
    """Let deploy-only adapters validate their own source-format contract."""
    from amct_pytorch.common.models import MODEL_REGISTRY

    # Argument validation precedes workflow setup: register the lightweight
    # adapter here without importing all model implementations.
    from ..deepseek.deepseek_v4_1.deepseekv4_1 import DeepseekV41  # noqa: F401

    name = getattr(args, "model_name", None)
    if name in MODEL_REGISTRY:
        validator = getattr(MODEL_REGISTRY.get(name), "validate_deploy_args", None)
        if callable(validator):
            validator(args)
            return True
    return False


def validate_tensor_deploy_args(args):
    """Preflight for both CLI and programmatic tensor workflows."""
    if not requires_tensor_deploy_config(args):
        return
    if _validate_model_deploy_args(args):
        return
    if getattr(args, "quant_dtype", None) != "int":
        raise ValueError(
            "Tensor deployment currently requires --quant_dtype int (W8A8)"
        )
    path = getattr(args, "bit_config", None)
    if path:
        try:
            cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError, yaml.YAMLError) as exc:
            raise ValueError(f"Cannot read bit_config {path}: {exc}") from exc
    else:
        policy = getattr(args, "bit_policy", None)
        cfg = policy.cfg if isinstance(policy, BitPolicy) else None
    validate_tensor_bit_config(cfg)
    args.bit_policy = BitPolicy(cfg)
    selection_path = getattr(args, "quant_layers_config", None)
    if selection_path:
        args.deploy_selection = load_deploy_selection(selection_path)
    elif not isinstance(getattr(args, "deploy_selection", None), DeploySelection):
        raise ValueError("Tensor deployment requires --quant_layers_config")
    if getattr(args, "quant_target", None):
        logger.warning(
            "Tensor JSON selection replaces quant_target; quant_target is not used"
        )


def tensor_module_role(name):
    """Map common checkpoint paths to existing coarse BitPolicy roles.

    This describes names, not quantization selection. Unsupported naming fails
    explicitly instead of silently falling back to the global/default 16 bits.
    """
    if ".self_attn." in name:
        return "attn-linear", name.rsplit(".", 1)[-1]
    if re.search(r"\.mlp\.experts\.\d+\.", name):
        return "moe.routed", name.rsplit(".", 1)[-1]
    if ".mlp.shared_experts." in name or ".mlp.shared_mlp." in name:
        return "moe.shared", name.rsplit(".", 1)[-1]
    if ".mlp." in name:
        return "mlp", name.rsplit(".", 1)[-1]
    raise ValueError(f"No tensor bit-policy role for selected module: {name}")


def _resolve_bits(name, policy):
    group, projection = tensor_module_role(name)
    node = policy.cfg
    explicit = False
    for part in (*group.split("."), projection):
        node = node.get(part)
        if not isinstance(node, dict):
            break
        explicit |= "w_bits" in node and "a_bits" in node
    if not explicit:
        raise ValueError(f"Missing explicit {group} bit configuration for {name}")
    return policy.linear_bits(name=projection, group=group)


def _mtp_report(names, ignored, selected, config):
    start = getattr(config, "num_hidden_layers", 0)
    count = getattr(config, "num_nextn_predict_layers", 0)
    prefixes = tuple(f"model.layers.{i}." for i in range(start, start + count))
    modules = frozenset(name for name in names if name.startswith(prefixes))
    return MtpSelectionReport(modules, modules & ignored, modules & selected)


def log_mtp_selection(
    report, *, phase="planned", quantized_modules=(), module_bits=None
):
    log = (
        logger.warning
        if report.modules and report.ignored != report.modules
        else logger.info
    )
    log(
        "MTP {}: status={}, modules={}, ignored={}, selected={}, quantized={}, selected_sample={}",
        phase,
        report.status,
        len(report.modules),
        len(report.ignored),
        len(report.selected),
        len(report.modules & set(quantized_modules)),
        [
            (name, (module_bits or {}).get(name))
            for name in sorted(report.selected)[:20]
        ],
    )


def build_tensor_deploy_plan(weight_map, selection, policy, *, config=None):
    validate_tensor_bit_config(policy.cfg)
    names = {key.rsplit(".", 1)[0] for key in weight_map}
    selected, counts = match_deploy_modules(names, selection)
    ignored = frozenset(
        name
        for name in names
        if any(fnmatchcase(name, p) for p in selection.ignore_layers)
    )
    weights = {
        key.removesuffix(".weight") for key in weight_map if key.endswith(".weight")
    }
    report = _mtp_report(names, ignored, selected, config)
    log_mtp_selection(report)
    if missing := selected - weights:
        raise ValueError(f"Selected modules have no .weight tensor: {sorted(missing)}")
    bits = {}
    for name in sorted(selected):
        try:
            bits[name] = _resolve_bits(name, policy)
        except ValueError as exc:
            prefix = "MTP " if name in report.selected else ""
            raise ValueError(f"{prefix}{exc}") from exc
    return ResolvedDeployPlan(
        frozenset(selected),
        frozenset(name + ".weight" for name in selected),
        bits,
        ignored,
        frozenset(weights - selected),
        counts,
        report,
        frozenset(p for p in selection.ignore_layers if counts[p] == 0),
    )


def validate_ascend_deploy_plan(plan, weight_map, adapter):
    """Validate Ascend capability without altering JSON selection."""
    required = ("get_ascend_deploy_candidates", "validate_ascend_deploy_selection")
    missing = [name for name in required if not callable(getattr(adapter, name, None))]
    if missing:
        raise ValueError(
            f"Ascend deployment adapter {type(adapter).__name__} lacks required capabilities: {missing}"
        )
    if plan.mtp.selected:
        raise ValueError(
            f"Ascend MTP quantization is not supported: {sorted(plan.mtp.selected)}; use ignore_layers"
        )
    candidates = adapter.get_ascend_deploy_candidates(
        {key.removesuffix(".weight") for key in weight_map if key.endswith(".weight")}
    )
    if unsupported := plan.selected_modules - candidates:
        raise ValueError(
            f"Unsupported Ascend deployment modules: {sorted(unsupported)}"
        )
    adapter.validate_ascend_deploy_selection(plan.selected_modules)
    return plan


def build_ascend_deploy_plan(weight_map, selection, adapter):
    plan = build_tensor_deploy_plan(
        weight_map, selection, adapter.args.bit_policy, config=adapter.config
    )
    return validate_ascend_deploy_plan(plan, weight_map, adapter)


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
    patterns may match non-quantizable modules (e.g. norm/head). Unmatched quant
    rules fail; unmatched ignore rules warn, with their zero counts retained.
    Candidate and fusion validation belongs to the format adapter.
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
    unmatched = sorted({p for p in selection.quant_layers if counts[p] == 0})
    if unmatched:
        raise ValueError(f"quant_layers patterns matched no modules: {unmatched}")
    unmatched_ignore = sorted({p for p in selection.ignore_layers if counts[p] == 0})
    if unmatched_ignore:
        logger.warning(
            "ignore_layers patterns matched no modules: {}", unmatched_ignore
        )
    selected -= ignored
    if not selected:
        raise ValueError(
            "No modules selected after applying quant_layers and ignore_layers"
        )
    return selected, counts


def validate_ascend_deploy_args(args) -> None:
    """Validate deploy mode without loading weights or altering legacy options.

    W8A8 configuration is checked by the shared tensor argument validator.
    Model fusion constraints and source/output checks remain format-specific.
    """
    if _validate_model_deploy_args(args):
        return
    deploy_format = getattr(args, "deploy_format", "legacy")
    selection_path = getattr(args, "quant_layers_config", None)
    if deploy_format not in {"legacy", "ascend"}:
        raise ValueError(f"Unsupported deploy_format: {deploy_format}")
    if deploy_format == "legacy":
        if (
            selection_path is not None
            and getattr(args, "granularity", None) != "tensor"
        ):
            raise ValueError("quant_layers_config requires --granularity tensor")
        return
    if not selection_path and not isinstance(
        getattr(args, "deploy_selection", None), DeploySelection
    ):
        raise ValueError("Ascend deployment requires --quant_layers_config")
    for key, expected in (
        ("granularity", "tensor"),
        ("quant_dtype", "int"),
    ):
        if getattr(args, key, None) != expected:
            raise ValueError(f"Ascend deployment requires --{key} {expected}")
    for key in (
        "algos",
        "attn_linear_param_dir",
        "attn_cache_param_dir",
        "moe_mlp_param_dir",
    ):
        if getattr(args, key, None):
            raise ValueError(f"Ascend tensor deployment does not consume --{key}")
