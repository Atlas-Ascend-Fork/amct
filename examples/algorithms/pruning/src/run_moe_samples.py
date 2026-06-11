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
"""MoE expert-pruning samples (gate + ModuleList experts, CPU-only, random weights, no download). Covers:

 - fixed-ratio expert pruning: activation_count (by activation frequency) / mass_variance
   (by mass variance), dropping the least-contributing experts and shrinking the gate in step;
 - guarded variance-menu: calibrate once -> prune one copy per criterion MENU
   {mass, cond_var, peak, cvxpeak}, measure on a separate small validation set, and switch
   only when a variance criterion is *strictly better* than the fallback (mass).
Note: when pruning only MoE experts, set cnn/dense to 0 so the default does not also prune
the experts' inner dense FFN (see the README config notes).
Run: python3 src/run_moe_samples.py
"""

import copy

from utils import make_moe, moe_data, count_params, num_experts
import amct_pytorch as amct
from amct_pytorch.pruning import accuracy_based_auto_prune, MOE_VARIANCE_MENU_CFG


def fixed_ratio_expert_prune():
    base = make_moe()
    calib = moe_data()
    before = count_params(base)
    print(f"[moe] original: params {before:,}, experts per layer {num_experts(base)}")
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
            f"[moe.{method:16s}] pruned 50% of experts: params -> {count_params(model):,}, "
            f"experts per layer {num_experts(model)} forward OK"
        )


def variance_menu_expert_prune():
    model = make_moe()
    calib = moe_data(n=8)
    val = moe_data(n=3, batch=2)
    cfg = copy.deepcopy(MOE_VARIANCE_MENU_CFG)
    cfg["methods"]["moe"]["kwargs"] = {"prune_ratio": 0.5, "top_k": 2}
    res = accuracy_based_auto_prune(
        model, cfg, data=calib, eval_data=val, apply=True
    )
    _ = model(val[0])
    print("[moe.variance_menu]", res.summary())
    chosen = next(
        (e.detail for e in res.report.events if e.stage == "menu-select"), None
    ) if res.report else None
    print(
        f"[moe.variance_menu] {chosen}, experts per layer {num_experts(model)} forward OK"
    )


def main():
    fixed_ratio_expert_prune()
    variance_menu_expert_prune()


if __name__ == "__main__":
    main()
