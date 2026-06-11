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
"""MoE 专家剪枝样例（gate + ModuleList experts，纯 CPU、随机权重、不下载）。覆盖：

 - 固定率专家剪枝：activation_count（按激活频次）/ mass_variance（按质量方差），
   裁掉贡献最小的专家并同步收缩 gate；
 - 守卫式 variance-menu：一次校准 → 在判据 MENU {mass, cond_var, peak, cvxpeak} 上
   各剪一份，用独立小验证集实测择优，仅当某方差判据*严格优于* fallback(mass) 时切换。
注意：只剪 MoE 专家时把 cnn/dense 置 0，避免默认把专家内部 dense FFN
也一起剪（见 README 配置说明）。
运行：python3 src/run_moe_samples.py
"""

import copy

from utils import make_moe, moe_data, count_params, num_experts
import amct_pytorch as amct
from amct_pytorch.pruning import accuracy_based_auto_prune, MOE_VARIANCE_MENU_CFG


def fixed_ratio_expert_prune():
    base = make_moe()
    calib = moe_data()
    before = count_params(base)
    print(f"[moe] 原始：参数 {before:,}，每层专家数 {num_experts(base)}")
    for method in ("activation_count", "mass_variance"):
        model = copy.deepcopy(base)
        kwargs = {"prune_ratio": 0.5}
        if method == "mass_variance":
            kwargs["boundary"] = 10
        cfg = {"methods": {
            "cnn": {"name": "variance_channel", "kwargs": {"prune_ratio": 0.0}},
            "dense": {"name": "low_variance", "kwargs": {"prune_ratio": 0.0}},
            "moe": {"name": method, "kwargs": kwargs},
        }}
        amct.prune(model, cfg, data=calib)
        _ = model(calib[0])
        print(
            f"[moe.{method:16s}] 剪 50% 专家：参数 → {count_params(model):,}，"
            f"每层专家数 {num_experts(model)} 前向OK"
        )


def variance_menu_expert_prune():
    model = make_moe()
    calib = moe_data(n=8)            # 校准集
    val = moe_data(n=3, batch=2)     # 守卫用的验证集（与校准不同）
    # 守卫式 variance 菜单搜索（菜单模式）：accuracy_based_auto_prune 在判据 MENU 上各剪一份，
    # 实测择优，仅当严格优于 fallback(mass) 时切换；返回标准 AutoTuneResult。
    cfg = copy.deepcopy(MOE_VARIANCE_MENU_CFG)
    cfg["methods"]["moe"]["kwargs"] = {"prune_ratio": 0.5, "top_k": 2}
    res = accuracy_based_auto_prune(
        model, cfg, data=calib, eval_data=val, apply=True
    )
    _ = model(val[0])
    print("[moe.variance_menu]", res.summary())
    # 选中判据记录在 report 里（也可传 report=PruneReport() sink 获取）
    chosen = next(
        (e.detail for e in res.report.events if e.stage == "menu-select"), None
    ) if res.report else None
    print(
        f"[moe.variance_menu] {chosen}，每层专家数 {num_experts(model)} 前向OK"
    )


def main():
    fixed_ratio_expert_prune()
    variance_menu_expert_prune()


if __name__ == "__main__":
    main()
