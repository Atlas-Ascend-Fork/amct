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
"""dense FFN 结构化剪枝样例（纯 CPU、随机权重、不下载）。覆盖：

 - 固定剪枝率（dict 配置直传 prune()，与 amct.quantize 同风格）；
 - 容差驱动自动剪枝（只给 tolerance，二分查找满足容差的最大剪枝率）；
 - 守卫式 recovery-HYBRID（recovery MENU {none, bias, ls} 上择优，不劣于 none）；
 - 剪枝 → 量化 → 转换 联合压缩（正交，体积收益相乘；先剪后量）；
 - 自定义精度抽象（evaluator / quality_fn）与可剪性诊断 diagnose()。
运行：python3 src/run_dense_samples.py
"""

import torch

from utils import make_mlp, mlp_data, count_params
import amct_pytorch.pruning as P
from amct_pytorch.pruning import guarded_recovery_hybrid_prune

try:
    import amct_pytorch as amct

    HAS_QUANT = hasattr(amct, "quantize") and hasattr(amct, "convert")
except Exception:
    HAS_QUANT = False


def fixed_ratio_prune():
    model = make_mlp()
    calib = mlp_data()
    before = count_params(model)
    cfg = {"methods": {"dense": {"name": "low_variance", "kwargs": {"prune_ratio": 0.5}}}}
    P.prune(model, cfg, data=calib)  # in-place
    after = count_params(model)
    print(
        f"[dense.fixed] FFN 固定率 0.5：参数 {before:,} → {after:,} "
        f"(削减 {100 * (1 - after / before):.1f}%)"
    )
    print("[dense.fixed] report:", P.get_last_report().as_dict())
    _ = model(calib[0])  # 剪后前向通过


def tolerance_auto_prune():
    for tol in (0.05, 0.20):
        model = make_mlp()
        calib = mlp_data()
        P.prune(model, data=calib, tolerance=tol)  # 仅给容差，其余全自动
        res = P.get_last_auto_prune()
        print(
            f"[dense.tolerance={tol}] baseline={res.baseline_quality:.3f} "
            f"chosen_ratio={res.chosen_ratio} applied={res.applied}"
        )


def recovery_hybrid_prune():
    model = make_mlp()
    calib, val = mlp_data(n=8), mlp_data(n=3, batch=2)
    before = count_params(model)
    res = guarded_recovery_hybrid_prune(
        model, data=calib, prune_ratio=0.5, domain="dense", eval_data=val, apply=True
    )
    _ = model(val[0])
    print("[dense.recovery_hybrid]", res.summary())
    assert res.gain_over_fallback >= 0.0 and count_params(model) < before


def prune_then_quantize():
    model = make_mlp()
    calib = mlp_data()
    before = count_params(model)
    P.prune(
        model,
        {"methods": {"dense": {"name": "low_variance", "kwargs": {"prune_ratio": 0.5}}}},
        data=calib,
    )
    print(f"[dense.prune+quant] 剪枝后参数 {before:,} → {count_params(model):,}")
    if HAS_QUANT:
        amct.quantize(model)  # 量化须在剪枝后的权重上 calibration
        amct.convert(model)
        _ = model(calib[0])
        print("[dense.prune+quant] quantize→convert 完成，前向OK")
    else:
        print("[dense.prune+quant] 环境无 amct.quantize/convert，跳过量化（剪枝部分已验证）")


def evaluator_and_diagnose():
    model = make_mlp()
    calib = mlp_data()

    # quality_fn：直接传自定义精度函数（优先级 quality_fn > evaluator > 默认 top-1 保真度）
    ref = model(calib[0]).argmax(-1)

    def quality_fn(m):
        with torch.no_grad():
            pred = m(calib[0]).argmax(-1)
        return (pred == ref).float().mean().item()

    res = P.accuracy_based_auto_prune(model, data=calib, tolerance=0.05, quality_fn=quality_fn)
    print("[dense.quality_fn]", res.summary())

    # diagnose()：不改模型，提前报告各域可剪目标（无目标/不支持早发现）
    rep = P.prune_diagnose(make_mlp(), data=calib)
    print("[dense.diagnose]", rep.as_dict() if hasattr(rep, "as_dict") else rep)


def main():
    fixed_ratio_prune()
    tolerance_auto_prune()
    recovery_hybrid_prune()
    prune_then_quantize()
    evaluator_and_diagnose()


if __name__ == "__main__":
    main()
