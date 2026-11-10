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

from __future__ import annotations

import json
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path

from loguru import logger

from amct_pytorch.quantization.bit_policy import BitPolicy

from .schemes import (
    QuantSchemeKey,
    get_quant_scheme,
    resolve_deploy_platform,
    validate_platform_quant_scheme,
    validate_selection_quant_scheme,
)


@dataclass(frozen=True)
class QuantSpec:
    """Quantization scheme assigned to one checkpoint module."""

    quant_dtype: str
    weight_bits: int
    activation_bits: int


@dataclass(frozen=True)
class QuantLayerRule:
    """Case-sensitive checkpoint module glob and its quantization scheme."""

    pattern: str
    quant_spec: QuantSpec


@dataclass(frozen=True)
class DeploySelection:
    """Checkpoint quantization and ignore rules without .weight suffixes."""

    quant_layers: tuple[str | QuantLayerRule, ...]
    ignore_layers: tuple[str, ...] = ()

    @property
    def uses_structured_rules(self):
        """Return whether every quant layer carries an explicit QuantSpec."""
        return bool(self.quant_layers) and all(
            isinstance(rule, QuantLayerRule) for rule in self.quant_layers
        )


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
        module_quant_specs: Selected module name to explicit QuantSpec. Modules
            are matched by quant_layers after ignore_layers takes precedence.
            Every selected module has a .weight tensor.
        selected_weight_keys: Exact checkpoint keys to quantize, formed by
            appending .weight to each selected module name. Biases are excluded.
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

    module_quant_specs: dict[str, QuantSpec]
    selected_weight_keys: frozenset[str]
    ignored_modules: frozenset[str]
    unselected_weight_modules: frozenset[str]
    match_counts: dict[str, int]
    mtp: MtpSelectionReport
    unmatched_ignore_patterns: frozenset[str] = frozenset()

    @property
    def selected_modules(self):
        """Return modules with an explicit quantization plan."""
        return frozenset(self.module_quant_specs)

    @property
    def module_bits(self):
        """Compatibility view for callers not migrated to QuantSpec yet."""
        return {
            name: (spec.weight_bits, spec.activation_bits)
            for name, spec in self.module_quant_specs.items()
        }

    @property
    def quant_layers(self):
        """Return selected module names mapped to weight bits for conversion."""
        return {
            name: spec.weight_bits for name, spec in self.module_quant_specs.items()
        }


def _prepare_tensor_deploy_args(args):
    """Load the structured tensor JSON contract for migrated models."""
    selection_path = getattr(args, "quant_layers_config", None)
    if selection_path:
        args.deploy_selection = load_deploy_selection(selection_path)
    selection = getattr(args, "deploy_selection", None)
    if not isinstance(selection, DeploySelection):
        raise ValueError(  # noqa: TRY004
            "Configured tensor deployment requires --quant_layers_config"
        )
    if not selection.uses_structured_rules:
        raise ValueError(
            "Configured tensor deployment JSON must use structured rules with "
            "pattern, quant_dtype, w_bits, and a_bits"
        )
    validate_selection_quant_scheme(selection)
    if getattr(args, "quant_target", None):
        logger.warning(
            "Tensor JSON selection replaces quant_target; quant_target is not used"
        )


def validate_deploy_cli_args(args):
    """Validate deploy-only CLI shape before loading a model or checkpoint."""
    selection_path = getattr(args, "quant_layers_config", None)
    if selection_path is not None and not Path(selection_path).is_file():
        raise ValueError(
            f"quant_layers_config must be an existing file: {selection_path}"
        )

    if getattr(args, "granularity", None) != "tensor":
        raise ValueError("Deployment requires --granularity tensor")
    if not selection_path:
        raise ValueError("Deployment requires --quant_layers_config")


def _quant_pattern(rule):
    return rule.pattern if isinstance(rule, QuantLayerRule) else rule


def get_structured_quant_spec(selection, module_name):
    """Return the explicit JSON scheme matching one resolved module."""
    if not getattr(selection, "uses_structured_rules", False):
        raise ValueError("Tensor deployment requires structured quant_layers rules")
    matches = [
        rule.quant_spec
        for rule in selection.quant_layers
        if isinstance(rule, QuantLayerRule) and fnmatchcase(module_name, rule.pattern)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Structured JSON must match exactly one quant_layers rule for "
            f"{module_name}, got {len(matches)}"
        )
    return matches[0]


def _validate_quant_spec(spec, path):
    if spec.quant_dtype == "hifp":
        raise NotImplementedError(
            f"{path}: Tensor deployment with quant_dtype 'hifp' is temporarily unsupported"
        )
    if not isinstance(spec.quant_dtype, str) or spec.quant_dtype not in {
        "int",
        "mxfp",
    }:
        raise ValueError(
            f"{path}.quant_dtype={spec.quant_dtype!r}: expected 'int' or 'mxfp'"
        )
    for field, value in (
        ("w_bits", spec.weight_bits),
        ("a_bits", spec.activation_bits),
    ):
        if type(value) is not int or value not in {4, 8}:
            raise ValueError(f"{path}.{field}={value!r}: expected 4 or 8")
    if (spec.weight_bits, spec.activation_bits) not in {(8, 8), (4, 8), (4, 4)}:
        raise ValueError(
            f"{path}: Quantization configuration compatibility issue: "
            f"bit combination W{spec.weight_bits}A{spec.activation_bits} "
            "is outside the AMCT selection schema; "
            "expected W8A8, W4A8, or W4A4"
        )


def _validate_executable_quant_spec(spec, path):
    _validate_quant_spec(spec, path)
    try:
        get_quant_scheme(
            QuantSchemeKey(spec.quant_dtype, spec.weight_bits, spec.activation_bits)
        )
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from exc


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
        "MTP {}: status={}, modules={}, ignored={}, selected={}, quantized={}, "
        "selected_sample={}",
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


def build_tensor_deploy_plan(
    weight_map, selection, policy=None, *, config=None, quant_dtype="int"
):
    if not selection.uses_structured_rules:
        raise ValueError("Tensor deployment requires structured quant_layers rules")
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
    specs = {}
    for name in sorted(selected):
        try:
            rule = next(
                rule
                for rule in selection.quant_layers
                if fnmatchcase(name, rule.pattern)
            )
            _validate_executable_quant_spec(rule.quant_spec, rule.pattern)
            specs[name] = rule.quant_spec
        except ValueError as exc:
            prefix = "MTP " if name in report.selected else ""
            raise ValueError(f"{prefix}{exc}") from exc
    return ResolvedDeployPlan(
        specs,
        frozenset(name + ".weight" for name in selected),
        ignored,
        frozenset(weights - selected),
        counts,
        report,
        frozenset(p for p in selection.ignore_layers if counts[p] == 0),
    )


def validate_tensor_deploy_plan(plan, weight_map, adapter):
    """Validate Deployment capability without altering JSON selection."""
    required = ("get_deploy_candidates", "validate_deploy_selection")
    missing = [name for name in required if not callable(getattr(adapter, name, None))]
    if missing:
        raise ValueError(
            f"Deployment adapter {type(adapter).__name__} lacks required "
            f"capabilities: {missing}"
        )
    if plan.mtp.selected:
        raise ValueError(
            "Deployment MTP quantization is not supported: "
            f"{sorted(plan.mtp.selected)}; use ignore_layers"
        )
    scheme_keys = {
        QuantSchemeKey(spec.quant_dtype, spec.weight_bits, spec.activation_bits)
        for spec in plan.module_quant_specs.values()
    }
    if len(scheme_keys) > 1:
        raise ValueError(
            "Tensorwise export currently requires one quant_dtype and "
            "one quantization scheme "
            f"per model, got {sorted(scheme_keys, key=lambda key: (key.quant_dtype, key.weight_bits, key.activation_bits))}"
        )
    scheme = get_quant_scheme(next(iter(scheme_keys)))
    capabilities = getattr(adapter, "get_deploy_capabilities", None)
    if callable(capabilities):
        platform = resolve_deploy_platform(
            getattr(adapter.args, "deploy_platform", None)
        )
        platform_capabilities = capabilities()
        if platform not in platform_capabilities:
            raise ValueError(
                f"{type(adapter).__name__} does not support Deployment platform {platform}"
            )
        if scheme.quant_dtype not in set(platform_capabilities[platform]):
            raise ValueError(
                f"{type(adapter).__name__} on {platform} does not support "
                f"quant_dtype value: {scheme.quant_dtype!r}"
            )
    candidates = adapter.get_deploy_candidates(
        {key.removesuffix(".weight") for key in weight_map if key.endswith(".weight")}
    )
    if unsupported := plan.selected_modules - candidates:
        raise ValueError(f"Unsupported Deployment modules: {sorted(unsupported)}")
    adapter.validate_deploy_selection(plan.selected_modules)
    return plan


def load_deploy_selection(path: Path) -> DeploySelection:
    """Read a selection JSON, rejecting typos and invalid rule types.

    Duplicate legacy string patterns are collapsed in first-occurrence order.
    Structured rules are preserved so matching can reject overlaps. Module-name
    matching and unmatched-pattern errors require the checkpoint inventory.
    """
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read quant_layers_config {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(  # noqa: TRY004
            f"{path}: selection must be a JSON object"
        )
    unknown = data.keys() - {"quant_layers", "ignore_layers"}
    if unknown:
        raise ValueError(f"{path}: unknown selection fields: {sorted(unknown)}")
    if not data.get("quant_layers"):
        raise ValueError(f"{path}: quant_layers must be a non-empty list")

    quant_layers = data["quant_layers"]
    if not isinstance(quant_layers, list):
        raise ValueError(  # noqa: TRY004
            f"{path}: quant_layers must be a non-empty list"
        )
    if all(isinstance(value, dict) for value in quant_layers):
        quant_rules = tuple(
            _load_quant_layer_rule(value, path, index)
            for index, value in enumerate(quant_layers)
        )
    elif all(isinstance(value, str) for value in quant_layers):
        if any(not value.strip() for value in quant_layers):
            raise ValueError(
                f"{path}: quant_layers must be a list of non-empty strings"
            )
        quant_rules = tuple(dict.fromkeys(quant_layers))
    else:
        raise ValueError(f"{path}: quant_layers entries must all use the same schema")

    ignore_layers = data.get("ignore_layers", [])
    if not isinstance(ignore_layers, list) or any(
        not isinstance(value, str) or not value.strip() for value in ignore_layers
    ):
        raise ValueError(f"{path}: ignore_layers must be a list of non-empty strings")
    return DeploySelection(quant_rules, tuple(dict.fromkeys(ignore_layers)))


def _load_quant_layer_rule(data, path, index):
    fields = {"pattern", "quant_dtype", "w_bits", "a_bits"}
    location = f"{path}: quant_layers[{index}]"
    if unknown := data.keys() - fields:
        raise ValueError(f"{location}: unknown fields: {sorted(unknown)}")
    if missing := fields - data.keys():
        raise ValueError(f"{location}: missing fields: {sorted(missing)}")
    pattern = data["pattern"]
    if not isinstance(pattern, str) or not pattern.strip():
        raise ValueError(f"{location}.pattern must be a non-empty string")
    if pattern.endswith(".weight"):
        raise ValueError(f"{location}.pattern must not include the .weight suffix")
    spec = QuantSpec(data["quant_dtype"], data["w_bits"], data["a_bits"])
    _validate_quant_spec(spec, location)
    return QuantLayerRule(pattern, spec)


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
    quant_matches = {}
    for rules, matches in (
        (selection.quant_layers, selected),
        (selection.ignore_layers, ignored),
    ):
        for rule in rules:
            pattern = _quant_pattern(rule)
            found = {name for name in names if fnmatchcase(name, pattern)}
            counts[pattern] = len(found)
            matches.update(found)
            if isinstance(rule, QuantLayerRule):
                for name in found:
                    quant_matches.setdefault(name, []).append(pattern)
    quant_patterns = {_quant_pattern(rule) for rule in selection.quant_layers}
    unmatched = sorted({p for p in quant_patterns if counts[p] == 0})
    if unmatched:
        raise ValueError(f"quant_layers patterns matched no modules: {unmatched}")
    unmatched_ignore = sorted({p for p in selection.ignore_layers if counts[p] == 0})
    if unmatched_ignore:
        logger.warning(
            "ignore_layers patterns matched no modules: {}", unmatched_ignore
        )
    selected -= ignored
    overlaps = {
        name: patterns
        for name, patterns in quant_matches.items()
        if name in selected and len(patterns) > 1
    }
    if overlaps:
        raise ValueError(
            "Modules matched multiple quant_layers patterns: "
            f"{dict(sorted(overlaps.items()))}"
        )
    if not selected:
        raise ValueError(
            "No modules selected after applying quant_layers and ignore_layers"
        )
    return selected, counts


def prepare_deploy_args(args) -> None:
    """Validate and materialize deploy arguments exactly once.

    CLI parsing and programmatic workflow construction share this entry point.
    It performs shape checks, loads the single JSON selection, validates the
    supported quantization contract, and checks the requested platform. Weight
    and model-structure checks happen later in the deploy workflow.
    """
    if (
        getattr(args, "granularity", None) == "tensor"
        and getattr(args, "quant_dtype", None) == "hifp"
    ):
        raise NotImplementedError(
            f"Tensor deployment with quant_dtype '{args.quant_dtype}' is temporarily unsupported"
        )
    validate_deploy_cli_args(args)
    if getattr(args, "granularity", None) == "tensor":
        _prepare_tensor_deploy_args(args)
    elif getattr(args, "bit_config", None):
        args.bit_policy = BitPolicy.from_yaml(args.bit_config)
    else:
        args.bit_policy = BitPolicy()

    if getattr(args, "granularity", None) == "tensor":
        selection = getattr(args, "deploy_selection", None)
        if not isinstance(selection, DeploySelection):
            raise ValueError("Tensor deployment requires --quant_layers_config")
        platform = resolve_deploy_platform(getattr(args, "deploy_platform", None))
        validate_platform_quant_scheme(selection, platform)
        for key in (
            "algos",
            "attn_linear_param_dir",
            "attn_cache_param_dir",
            "moe_mlp_param_dir",
        ):
            if getattr(args, key, None):
                raise ValueError(f"Tensor deployment does not consume --{key}")
    args._deploy_args_prepared = True
