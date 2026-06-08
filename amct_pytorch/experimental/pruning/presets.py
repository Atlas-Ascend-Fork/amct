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
"""预定义剪枝配置，与 amct_pytorch.common.config 中量化预设保持同一风格。"""

CNN_VARIANCE_PRUNE_CFG = {
    "methods": {
        "cnn": {"name": "variance_channel", "kwargs": {"prune_ratio": 0.30}},
    },
    "missing_data_policy": "warn_skip",
}


# 输出保留(重构)通道剪枝：自研、低破坏，剪枝后用最小二乘重构 consumer 权重补偿精度。
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
