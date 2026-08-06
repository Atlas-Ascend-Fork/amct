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
CNN_VARIANCE_PRUNE_CFG = {
    "methods": {
        "cnn": {"name": "variance_channel", "kwargs": {"prune_ratio": 0.30}},
    },
    "missing_data_policy": "warn_skip",
}


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


MOE_OUTPUT_MERGE_PRUNE_CFG = {
    "methods": {
        "moe": {
            "name": "output_merge",
            "kwargs": {"keep_ratio": 0.50, "selector": "calib_nll"},
        },
    },
    "missing_data_policy": "warn_skip",
}


_MOE_CRITERION_MENU = (
    ("mass", {"name": "activation_count", "kwargs": {}}),
    (
        "cond_var",
        {"name": "mass_variance", "kwargs": {"boundary": -1, "variance_score": "cond"}},
    ),
    (
        "peak",
        {"name": "mass_variance", "kwargs": {"boundary": -1, "variance_score": "peak"}},
    ),
    (
        "cvxpeak",
        {
            "name": "mass_variance",
            "kwargs": {"boundary": -1, "variance_score": "cvxpeak"},
        },
    ),
)


MOE_VARIANCE_MENU_CFG = {
    "methods": {
        "moe": {
            "name": "activation_count",
            "kwargs": {"prune_ratio": 0.50},
            "menu": _MOE_CRITERION_MENU,
        },
    },
    "missing_data_policy": "warn_skip",
}


_RECOVERY_MENU = (
    ("none", {"name": "reconstruct", "kwargs": {"recovery": "none"}}),
    ("bias", {"name": "reconstruct", "kwargs": {"recovery": "bias"}}),
    ("ls", {"name": "reconstruct", "kwargs": {"recovery": "ls"}}),
)

DENSE_RECOVERY_MENU_CFG = {
    "methods": {
        "dense": {
            "name": "reconstruct",
            "kwargs": {"prune_ratio": 0.50, "ridge": 1e-2},
            "menu": _RECOVERY_MENU,
        },
    },
    "missing_data_policy": "warn_skip",
}


CNN_RECOVERY_MENU_CFG = {
    "methods": {
        "cnn": {
            "name": "reconstruct",
            "kwargs": {"prune_ratio": 0.30, "ridge": 1e-2},
            "menu": _RECOVERY_MENU,
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
