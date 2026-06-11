# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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
