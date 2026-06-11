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
"""CNN 通道剪枝样例（纯 CPU、随机权重、不下载）。

 - ``variance_channel``：按激活方差朴素切片（对照）；
 - ``reconstruct``：剪掉低显著度通道后用最小二乘重构 consumer 权重补偿，破坏更低；
 - 守卫式 recovery 菜单搜索：accuracy_based_auto_prune 在 recovery MENU {none, bias, ls} 上守卫式择优，不劣于 none。
cnn 域自动识别 (producer conv → bn → consumer) 顺序通道目标，并避开输出头。
运行：python3 src/run_cnn_samples.py
"""

import copy

from utils import make_cnn, cnn_data, count_params
import amct_pytorch as amct
from amct_pytorch.pruning import accuracy_based_auto_prune

#: CNN recovery 菜单（菜单模式）：item 0 "none"（朴素切片）为 fallback；每项为带 recovery 的 reconstruct。
_CNN_RECOVERY_MENU_CFG = {
    "methods": {
        "cnn": {
            "name": "reconstruct",
            "kwargs": {"prune_ratio": 0.3, "ridge": 1e-2},
            "menu": (
                ("none", {"name": "reconstruct", "kwargs": {"recovery": "none"}}),
                ("bias", {"name": "reconstruct", "kwargs": {"recovery": "bias"}}),
                ("ls", {"name": "reconstruct", "kwargs": {"recovery": "ls"}}),
            ),
        },
    },
    "missing_data_policy": "warn_skip",
}


def fixed_ratio_channel_prune():
    base = make_cnn()
    calib = cnn_data()
    before = count_params(base)
    for name in ("variance_channel", "reconstruct"):
        model = copy.deepcopy(base)
        cfg = {"methods": {"cnn": {"name": name, "kwargs": {"prune_ratio": 0.3}}}}
        amct.prune(model, cfg, data=calib)
        after = count_params(model)
        _ = model(calib[0])  # 前向校验
        print(
            f"[cnn.{name:16s}] 通道剪枝 0.3：参数 {before:,} → {after:,} "
            f"(削减 {100 * (1 - after / before):.1f}%) 前向OK"
        )
    # 也可直接用内置预设（重构方法）
    model = copy.deepcopy(base)
    amct.prune(model, amct.CNN_RECONSTRUCT_PRUNE_CFG, data=calib)
    print(f"[cnn.preset] CNN_RECONSTRUCT_PRUNE_CFG：参数 → {count_params(model):,}")


def recovery_hybrid_channel_prune():
    model = make_cnn()
    calib, val = cnn_data(n=4), cnn_data(n=2)
    before = count_params(model)
    res = accuracy_based_auto_prune(
        model, _CNN_RECOVERY_MENU_CFG, data=calib, eval_data=val, apply=True
    )
    _ = model(val[0])
    print("[cnn.recovery_hybrid]", res.summary())
    assert count_params(model) < before


def main():
    fixed_ratio_channel_prune()
    recovery_hybrid_channel_prune()


if __name__ == "__main__":
    main()
