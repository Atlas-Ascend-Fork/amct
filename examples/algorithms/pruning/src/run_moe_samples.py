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
 - 守卫式 variance-HYBRID：一次校准 → 在判据 MENU {mass, cond_var, peak, cvxpeak} 上
   各剪一份，用独立小验证集实测择优，仅当某方差判据*严格优于* fallback(mass) 时切换。
注意：只剪 MoE 专家时把 cnn/dense 置 0，避免默认把专家内部 dense FFN
也一起剪（见 README 配置说明）。
运行：python3 src/run_moe_samples.py
"""

import copy

from utils import make_moe, moe_data, count_params, num_experts
import amct_pytorch.experimental.pruning as P
from amct_pytorch.experimental.pruning import guarded_variance_hybrid_prune


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
        P.prune(model, cfg, data=calib)
        _ = model(calib[0])
        print(
            f"[moe.{method:16s}] 剪 50% 专家：参数 → {count_params(model):,}，"
            f"每层专家数 {num_experts(model)} 前向OK"
        )


def variance_hybrid_expert_prune():
    model = make_moe()
    calib = moe_data(n=8)            # 校准集
    val = moe_data(n=3, batch=2)     # 守卫用的验证集（与校准不同）
    res = guarded_variance_hybrid_prune(
        model, data=calib, prune_ratio=0.5, top_k=2, eval_data=val, apply=True
    )
    _ = model(val[0])
    print("[moe.variance_hybrid]", res.summary())
    print(
        f"[moe.variance_hybrid] 选中判据={res.chosen_criterion}"
        f"（fallback={res.fallback_criterion}），每层专家数 {num_experts(model)} 前向OK"
    )
    # 守卫不变量：验证集上绝不劣于 fallback(mass)
    assert res.gain_over_fallback >= 0.0


def main():
    fixed_ratio_expert_prune()
    variance_hybrid_expert_prune()


if __name__ == "__main__":
    main()
