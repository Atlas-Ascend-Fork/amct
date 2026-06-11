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
"""dense FFN structured-pruning samples (CPU-only, random weights, no download). Covers:

 - fixed prune ratio (dict config passed straight to prune(), same style as amct.quantize);
 - tolerance-driven auto pruning (give only a tolerance; binary-search the largest ratio
   that still meets it);
 - guarded recovery-menu (pick the best of the recovery MENU {none, bias, ls}, falling back to none);
 - prune -> quantize -> convert joint compression (orthogonal, size gains multiply; prune first);
 - a custom accuracy abstraction (evaluator) and prunability diagnosis via diagnose().
Run: python3 src/run_dense_samples.py
"""

import copy

import torch

from utils import make_mlp, mlp_data, count_params
import amct_pytorch as amct
from amct_pytorch.pruning import (
    accuracy_based_auto_prune,
    DENSE_RECOVERY_MENU_CFG,
    PruneReport,
    prune_diagnose,
)

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
    report = PruneReport()
    amct.prune(model, cfg, data=calib, report=report)
    after = count_params(model)
    print(
        f"[dense.fixed] FFN fixed ratio 0.5: params {before:,} -> {after:,} "
        f"(-{100 * (1 - after / before):.1f}%)"
    )
    print("[dense.fixed] report:", report.as_dict())
    _ = model(calib[0])


def tolerance_auto_prune():
    for tol in (0.05, 0.20):
        model = make_mlp()
        calib = mlp_data()
        res = amct.accuracy_based_auto_prune(model, data=calib, tolerance=tol)
        print(
            f"[dense.tolerance={tol}] baseline={res.baseline_quality:.3f} "
            f"chosen_ratio={res.chosen_ratio} applied={res.applied}"
        )


def recovery_menu_prune():
    model = make_mlp()
    calib, val = mlp_data(n=8), mlp_data(n=3, batch=2)
    before = count_params(model)
    cfg = copy.deepcopy(DENSE_RECOVERY_MENU_CFG)
    cfg["methods"]["dense"]["kwargs"] = {"prune_ratio": 0.5, "ridge": 1e-2}
    res = accuracy_based_auto_prune(
        model, cfg, data=calib, eval_data=val, apply=True
    )
    _ = model(val[0])
    print("[dense.recovery_menu]", res.summary())
    assert count_params(model) < before


def prune_then_quantize():
    model = make_mlp()
    calib = mlp_data()
    before = count_params(model)
    amct.prune(
        model,
        {"methods": {"dense": {"name": "low_variance", "kwargs": {"prune_ratio": 0.5}}}},
        data=calib,
    )
    print(f"[dense.prune+quant] params after pruning {before:,} -> {count_params(model):,}")
    if HAS_QUANT:
        amct.quantize(model)
        amct.convert(model)
        _ = model(calib[0])
        print("[dense.prune+quant] quantize->convert done, forward OK")
    else:
        print(
            "[dense.prune+quant] amct.quantize/convert unavailable in this env, "
            "skipping quantization (pruning part already verified)"
        )


def evaluator_and_diagnose():
    model = make_mlp()
    calib = mlp_data()

    ref = model(calib[0]).argmax(-1)

    def acc_fn(m):
        with torch.no_grad():
            pred = m(calib[0]).argmax(-1)
        return (pred == ref).float().mean().item()

    res = amct.accuracy_based_auto_prune(model, data=calib, tolerance=0.05, evaluator=acc_fn)
    print("[dense.evaluator]", res.summary())

    rep = prune_diagnose(make_mlp(), data=calib)
    print("[dense.diagnose]", rep.as_dict() if hasattr(rep, "as_dict") else rep)


def main():
    fixed_ratio_prune()
    tolerance_auto_prune()
    recovery_menu_prune()
    prune_then_quantize()
    evaluator_and_diagnose()


if __name__ == "__main__":
    main()
