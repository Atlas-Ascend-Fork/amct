# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Independent checkpoint fixtures compare JSON selection with retained adapters.

These are legacy-plan equivalence tests, not Ascend runtime compatibility tests.
Exact oracle entries must exist in the fixture: never intersect two outputs to
hide names that the old interface generated but the checkpoint does not contain.
"""

import itertools
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from amct_pytorch.common.models.llm.common.deploy_selection import (
    build_tensor_deploy_plan,
    load_deploy_selection,
)
from amct_pytorch.common.models.llm.glm.glm5_2.glm5_2 import GLM5_2
from amct_pytorch.common.models.llm.hyv3.hyv3 import HyV3
from amct_pytorch.quantization.bit_policy import BitPolicy

ROOT = Path(__file__).resolve().parents[6]
CONFIG_DIR = ROOT / "amct_pytorch/configs/deploy"
FFN = ("gate_proj", "up_proj", "down_proj")
HY_ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")
TARGETS = ("attn-linear", "mlp", "moe")


def _policy():
    return BitPolicy.from_yaml(str(CONFIG_DIR / "tensor_w8a8.yaml"))


def _adapter(cls, config, targets=TARGETS):
    # Bypass model loading, but execute the original methods without overriding them.
    model = cls.__new__(cls)
    model.config = config
    model.num_layers = config.num_hidden_layers
    model.quant_target = list(targets)
    model.args = SimpleNamespace(quant_dtype="int", bit_policy=_policy())
    return model


def _checkpoint(config, family):
    """Describe small transformer structures independently of both selectors."""
    names = {"model.embed_tokens", "model.norm", "lm_head"}
    for layer in range(config.num_hidden_layers + config.num_nextn_predict_layers):
        prefix = f"model.layers.{layer}"
        names.update(
            f"{prefix}.{norm}"
            for norm in ("input_layernorm", "post_attention_layernorm")
        )
        if family == "glm":
            projections = (
                "q_a_proj",
                "q_b_proj",
                "kv_a_proj_with_mqa",
                "kv_b_proj",
                "o_proj",
            )
            types = config.indexer_types
            owns_indexer = (
                layer >= config.num_hidden_layers
                or not types
                or layer >= len(types)
                or types[layer] == "full"
            )
            if owns_indexer:
                names.update(
                    f"{prefix}.self_attn.indexer.{name}"
                    for name in ("wq_b", "wk", "weights_proj")
                )
            dense = layer < config.first_k_dense_replace
            count = config.n_routed_experts
            shared = "shared_experts"
        else:
            projections = HY_ATTN
            dense = layer == 0
            count = config.num_experts
            shared = "shared_mlp"
        names.update(f"{prefix}.self_attn.{name}" for name in projections)
        if dense:
            names.update(f"{prefix}.mlp.{name}" for name in FFN)
        else:
            names.add(f"{prefix}.mlp.gate")
            names.update(
                f"{prefix}.mlp.experts.{expert}.{name}"
                for expert in range(count)
                for name in FFN
            )
            names.update(f"{prefix}.mlp.{shared}.{name}" for name in FFN)
        if layer >= config.num_hidden_layers:
            names.update((f"{prefix}.eh_proj", f"{prefix}.shared_head.head"))
    weight_map = {
        f"{name}.weight": "model-00001-of-00001.safetensors" for name in names
    }
    # Include non-weight keys: selectors must never quantize biases.
    weight_map["model.layers.0.self_attn.o_proj.bias"] = (
        "model-00001-of-00001.safetensors"
    )
    # Runtime MoE containers have no checkpoint tensor; regex equivalence is
    # evaluated against parameter-bearing modules, not runtime containers.
    return weight_map, set(names)


def _expanded_oracle_ignore(rules, modules):
    result = set()
    for rule in rules:
        if rule.startswith("re:"):
            matches = {name for name in modules if re.match(rule[3:], name)}
            assert matches, f"Oracle regex has no fixture matches: {rule}"
            result.update(matches)
        else:
            assert rule in modules, (
                f"Oracle exact ignore missing from checkpoint: {rule}"
            )
            result.add(rule)
    return result


def _assert_equivalent(adapter, selection, weight_map, modules):
    expected_quant = adapter.generate_tensorwise_quant_layers()
    missing = set(expected_quant) - {
        key[:-7] for key in weight_map if key.endswith(".weight")
    }
    assert not missing, (
        f"Oracle quant entries absent from checkpoint: {sorted(missing)}"
    )
    expected_ignore = _expanded_oracle_ignore(
        adapter.generate_tensorwise_ignore_layers(), modules
    )
    plan = build_tensor_deploy_plan(
        weight_map, selection, adapter.args.bit_policy, config=adapter.config
    )
    actual_quant = dict(plan.quant_layers)
    assert plan.selected_modules == frozenset(expected_quant)
    assert actual_quant == expected_quant, {
        "unexpected": sorted(set(actual_quant) - set(expected_quant)),
        "missing": sorted(set(expected_quant) - set(actual_quant)),
        "bits": {
            name: (actual_quant[name], expected_quant[name])
            for name in actual_quant.keys() & expected_quant.keys()
            if actual_quant[name] != expected_quant[name]
        },
    }
    assert set(plan.ignored_modules) == expected_ignore, {
        "unexpected": sorted(set(plan.ignored_modules) - expected_ignore),
        "missing": sorted(expected_ignore - set(plan.ignored_modules)),
    }
    assert set(plan.selected_weight_keys) == {
        f"{name}.weight" for name in expected_quant
    }
    assert all(tuple(bits) == (8, 8) for bits in plan.module_bits.values())
    return plan


@pytest.mark.parametrize(
    "dense,experts,indexers,nextn,counts",
    [
        (1, 2, ["full", "shared", "full"], 1, (41, 24)),
        (2, 3, None, 2, (57, 33)),
        (1, 1, ["full"], 1, (33, 26)),
    ],
)
def test_glm_delivered_json_matches_retained_interfaces(
    dense, experts, indexers, nextn, counts
):
    config = SimpleNamespace(
        num_hidden_layers=3,
        num_nextn_predict_layers=nextn,
        n_routed_experts=experts,
        first_k_dense_replace=dense,
        indexer_types=indexers,
    )
    adapter = _adapter(GLM5_2, config)
    weights, modules = _checkpoint(config, "glm")
    plan = _assert_equivalent(
        adapter,
        load_deploy_selection(CONFIG_DIR / "glm5_2_tensor_w8a8.json"),
        weights,
        modules,
    )
    assert (len(plan.quant_layers), len(plan.ignored_modules)) == counts
    # The legacy GLM plan intentionally quantizes MTP attention and experts.
    assert "model.layers.3.self_attn.o_proj" in plan.selected_modules
    assert "model.layers.3.eh_proj" in plan.ignored_modules


def test_glm_without_mtp_uses_explicit_json_variant(tmp_path):
    config = SimpleNamespace(
        num_hidden_layers=3,
        num_nextn_predict_layers=0,
        n_routed_experts=2,
        first_k_dense_replace=1,
        indexer_types=["full", "shared", "full"],
    )
    data = json.loads((CONFIG_DIR / "glm5_2_tensor_w8a8.json").read_text())
    data["ignore_layers"] = [
        rule
        for rule in data["ignore_layers"]
        if not rule.endswith((".eh_proj", ".shared_head.head"))
    ]
    path = tmp_path / "no_mtp.json"
    path.write_text(json.dumps(data))
    weights, modules = _checkpoint(config, "glm")
    _assert_equivalent(
        _adapter(GLM5_2, config), load_deploy_selection(path), weights, modules
    )


@pytest.mark.parametrize(
    "targets",
    [combo for size in range(1, 4) for combo in itertools.combinations(TARGETS, size)],
)
def test_hyv3_all_target_combinations(tmp_path, targets):
    config = SimpleNamespace(
        num_hidden_layers=3, num_nextn_predict_layers=1, num_experts=2
    )
    adapter = _adapter(HyV3, config, targets)
    weights, modules = _checkpoint(config, "hy")
    # All enabled exercises the delivered file itself; subsets are user-authored
    # JSON selections, never an additional production quant_target filter.
    path = CONFIG_DIR / "hyv3_tensor_w8a8.json"
    if set(targets) != set(TARGETS):
        data = json.loads(path.read_text())
        enabled = []
        for rule in data["quant_layers"]:
            group = (
                "attn-linear"
                if ".self_attn." in rule
                else "moe"
                if ".experts." in rule or ".shared_mlp." in rule
                else "mlp"
            )
            (enabled if group in targets else data["ignore_layers"]).append(rule)
        data["quant_layers"] = enabled
        path = tmp_path / "subset.json"
        path.write_text(json.dumps(data))
    _assert_equivalent(adapter, load_deploy_selection(path), weights, modules)


def test_missing_oracle_weight_is_not_hidden_by_intersection():
    config = SimpleNamespace(
        num_hidden_layers=2, num_nextn_predict_layers=1, num_experts=1
    )
    weights, modules = _checkpoint(config, "hy")
    del weights["model.layers.0.self_attn.q_proj.weight"]
    with pytest.raises(AssertionError, match="Oracle quant entries absent"):
        _assert_equivalent(
            _adapter(HyV3, config),
            load_deploy_selection(CONFIG_DIR / "hyv3_tensor_w8a8.json"),
            weights,
            modules,
        )


def test_missing_oracle_exact_ignore_is_reported():
    config = SimpleNamespace(
        num_hidden_layers=2, num_nextn_predict_layers=1, num_experts=1
    )
    weights, modules = _checkpoint(config, "hy")
    del weights["lm_head.weight"]
    modules.remove("lm_head")
    with pytest.raises(AssertionError, match="Oracle exact ignore missing"):
        _assert_equivalent(
            _adapter(HyV3, config),
            load_deploy_selection(CONFIG_DIR / "hyv3_tensor_w8a8.json"),
            weights,
            modules,
        )


def test_glm_full_checkpoint_equivalence_with_explicit_absent_ignore():
    import os

    reference = os.environ.get("AMCT_GLM52_REFERENCE_DIR")
    if not reference:
        pytest.skip("Set AMCT_GLM52_REFERENCE_DIR for full checkpoint name comparison")
    root = Path(reference) / "GLM-5.2"
    config = SimpleNamespace(**json.loads((root / "config.json").read_text()))
    weights = json.loads((root / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    model = _adapter(GLM5_2, config)
    rules = load_deploy_selection(CONFIG_DIR / "glm5_2_tensor_w8a8.json")
    plan = build_tensor_deploy_plan(weights, rules, _policy(), config=config)
    expected_quant = model.generate_tensorwise_quant_layers()
    expected_ignore = set(model.generate_tensorwise_ignore_layers())
    names = {key.rsplit(".", 1)[0] for key in weights}
    # Report and assert the exact known absence, never silently hide differences.
    missing_ignore = expected_ignore - names
    assert missing_ignore == {"model.layers.78.shared_head.head"}
    assert plan.unmatched_ignore_patterns == {"model.layers.*.shared_head.head"}
    assert plan.quant_layers == expected_quant
    assert plan.selected_modules == frozenset(expected_quant)
    assert plan.ignored_modules | missing_ignore == expected_ignore
    assert len(plan.quant_layers) == 58785
    assert len(plan.ignored_modules) == 359
