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
"""dense FFN structured-pruning samples (CPU-only, random weights, no download). Covers:

 - fixed prune ratio (dict config passed straight to prune(), same style as amct.quantize);
 - tolerance-driven auto pruning (give only a tolerance; binary-search the largest ratio
   that still meets it);
 - guarded recovery-menu (pick the best of the recovery MENU {none, bias, ls}, falling back to none);
 - prune -> quantize -> convert joint compression (orthogonal, size gains multiply; prune first);
 - a custom accuracy abstraction (evaluator) and prunability diagnosis via prune_diagnose().
Run: python3 src/run_dense_samples.py
"""

import copy

import torch

from utils import make_mlp, mlp_data, count_params
import amct_pytorch as amct
from amct_pytorch.pruning import (
    DENSE_RECOVERY_MENU_CFG,
    PruneReport,
    prune_diagnose,
)

HAS_QUANT = hasattr(amct, "quantize") and hasattr(amct, "convert")


def fixed_ratio_prune():
    model = make_mlp()
    calib = mlp_data()
    before = count_params(model)
    cfg = {
        "methods": {"dense": {"name": "low_variance", "kwargs": {"prune_ratio": 0.5}}}
    }
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
        report = PruneReport()
        amct.prune(model, data=calib, tolerance=tol, report=report)
        cut = 1 - report.params_after / report.params_before
        note = report.warnings[0] if report.warnings else "met the tolerance"
        print(
            f"[dense.tolerance={tol}] weights {report.params_before:,} -> "
            f"{report.params_after:,} (cut {100 * cut:.1f}%), {note}"
        )


def recovery_menu_prune():
    model = make_mlp()
    calib, val = mlp_data(n=8), mlp_data(n=3, batch=2)
    before = count_params(model)
    cfg = copy.deepcopy(DENSE_RECOVERY_MENU_CFG)
    cfg["methods"]["dense"]["kwargs"] = {"prune_ratio": 0.5, "ridge": 1e-2}
    report = PruneReport()
    amct.prune(model, cfg, data=calib, eval_data=val, report=report)
    _ = model(val[0])
    chosen = next((e.detail for e in report.events if e.stage == "menu-select"), None)
    cut = 1 - report.params_after / report.params_before
    print(
        f"[dense.recovery_menu] weights {report.params_before:,} -> "
        f"{report.params_after:,} (cut {100 * cut:.1f}%), {chosen}"
    )
    assert count_params(model) < before


def prune_then_quantize():
    model = make_mlp()
    calib = mlp_data()
    before = count_params(model)
    amct.prune(
        model,
        {
            "methods": {
                "dense": {"name": "low_variance", "kwargs": {"prune_ratio": 0.5}}
            }
        },
        data=calib,
    )
    print(
        f"[dense.prune+quant] params after pruning {before:,} -> {count_params(model):,}"
    )
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

    report = PruneReport()
    amct.prune(model, data=calib, tolerance=0.05, evaluator=acc_fn, report=report)
    cut = 1 - report.params_after / report.params_before
    note = report.warnings[0] if report.warnings else "met the tolerance"
    print(
        f"[dense.evaluator] weights {report.params_before:,} -> {report.params_after:,} "
        f"(cut {100 * cut:.1f}%), {note}"
    )

    rep = prune_diagnose(make_mlp(), data=calib)
    print("[dense.diagnose]", rep.summary())


def main():
    fixed_ratio_prune()
    tolerance_auto_prune()
    recovery_menu_prune()
    prune_then_quantize()
    evaluator_and_diagnose()


if __name__ == "__main__":
    main()
