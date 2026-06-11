# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
# ----------------------------------------------------------------------------
"""Predefined prune configs, in the same style as the quant presets in
amct_pytorch.common.config.
"""

CNN_VARIANCE_PRUNE_CFG = {
    "methods": {
        "cnn": {"name": "variance_channel", "kwargs": {"prune_ratio": 0.30}},
    },
    "missing_data_policy": "warn_skip",
}


# Output-preserving (reconstruct) channel pruning: low-damage; consumer weights are
# least-squares reconstructed after pruning to recover accuracy.
CNN_RECONSTRUCT_PRUNE_CFG = {
    "methods": {
        "cnn": {
            "name": "reconstruct",
            "kwargs": {"prune_ratio": 0.30},
        },
    },
    "missing_data_policy": "warn_skip",
}


DENSE_LOWVAR_PRUNE_CFG = {
    "methods": {
        "dense": {"name": "low_variance", "kwargs": {"prune_ratio": 0.50}},
    },
    "missing_data_policy": "warn_skip",
}


MOE_MASSVAR_PRUNE_CFG = {
    "methods": {
        "moe": {
            "name": "mass_variance",
            "kwargs": {"prune_ratio": 0.50, "boundary": 10},
        },
    },
    "missing_data_policy": "warn_skip",
}


# Per-layer sensitivity allocation: same total reduction as uniform, but prunes
# robust layers more and sensitive layers less; calib_nll guard auto-reverts to
# uniform and warns if worse. Requires calibration data (data=).
SENSITIVITY_ALLOC_PRUNE_CFG = {
    "methods": {
        "dense": {"name": "low_variance", "kwargs": {"prune_ratio": 0.50}},
    },
    "allocation": {
        "strategy": "sensitivity",
        "ref_ratio": 0.50,
        "min_ratio": 0.05,
        "max_ratio": 0.90,
        "guard": "calib_nll",
    },
    "missing_data_policy": "warn_skip",
}


# MoE output-space expert merge: LS-residual greedy pairing + drop fallback, with a
# calib-NLL selector guaranteeing no worse than the activation_count baseline (the
# choice is written to the report, never silent).
MOE_OUTPUT_MERGE_PRUNE_CFG = {
    "methods": {
        "moe": {
            "name": "output_merge",
            "kwargs": {"keep_ratio": 0.50, "selector": "calib_nll"},
        },
    },
    "missing_data_policy": "warn_skip",
}


# --------------------------------------------------------------------------- #
# Advanced: config-driven "menu mode" for accuracy_based_auto_prune. A method
# spec carries a "menu" = list of (display_name, method_spec); item 0 is the
# fallback. accuracy_based_auto_prune prunes one deepcopy per menu item at the
# fixed prune_ratio, evaluates each on a validation set, and applies the best --
# only switching off the item-0 fallback on a strict improvement (guarded), so
# the result is never worse than the fallback on validation. The chosen variant
# is recorded in the report (prune(report=PruneReport()) sink).
#   accuracy_based_auto_prune(model, MOE_VARIANCE_MENU_CFG, data=calib,
#                             eval_data=val, evaluator=ev)
# --------------------------------------------------------------------------- #

#: MoE expert-selection criterion menu (item 0 = mass fallback; the three variance
#: entries use boundary=-1 so every layer scores by variance). The domain kwargs
#: (prune_ratio, top_k) are shared by every menu item.
_MOE_CRITERION_MENU = (
    ("mass", {"name": "activation_count", "kwargs": {}}),
    ("cond_var", {"name": "mass_variance", "kwargs": {"boundary": -1, "variance_score": "cond"}}),
    ("peak", {"name": "mass_variance", "kwargs": {"boundary": -1, "variance_score": "peak"}}),
    ("cvxpeak", {"name": "mass_variance", "kwargs": {"boundary": -1, "variance_score": "cvxpeak"}}),
)


MOE_VARIANCE_MENU_CFG = {
    "methods": {
        "moe": {
            "name": "activation_count",
            "kwargs": {"prune_ratio": 0.50, "top_k": 8},
            "menu": _MOE_CRITERION_MENU,
        },
    },
    "missing_data_policy": "warn_skip",
}


#: Post-prune compensation (recovery) menu for dense FFN reconstruct; item 0 "none"
#: (naive slice) is the fallback. Each item is a reconstruct spec carrying a recovery.
_DENSE_RECOVERY_MENU = (
    ("none", {"name": "reconstruct", "kwargs": {"recovery": "none"}}),
    ("bias", {"name": "reconstruct", "kwargs": {"recovery": "bias"}}),
    ("ls", {"name": "reconstruct", "kwargs": {"recovery": "ls"}}),
)


DENSE_RECOVERY_MENU_CFG = {
    "methods": {
        "dense": {
            "name": "reconstruct",
            "kwargs": {"prune_ratio": 0.50, "ridge": 1e-2},
            "menu": _DENSE_RECOVERY_MENU,
        },
    },
    "missing_data_policy": "warn_skip",
}


FULL_STRUCTURED_PRUNE_CFG = {
    "methods": {
        "cnn": {"name": "variance_channel", "kwargs": {"prune_ratio": 0.30}},
        "dense": {"name": "low_variance", "kwargs": {"prune_ratio": 0.50}},
        "moe": {
            "name": "mass_variance",
            "kwargs": {"prune_ratio": 0.50, "boundary": 10},
        },
    },
    "missing_data_policy": "warn_skip",
}
