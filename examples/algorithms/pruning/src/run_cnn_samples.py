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
"""CNN channel-pruning samples (CPU-only, random weights, no download).

 - ``variance_channel``: naive slicing by activation variance (baseline);
 - ``reconstruct``: after dropping low-saliency channels, compensate the consumer weights
   via least squares, lowering the damage;
 - guarded recovery-menu search: a config carrying a ``menu`` makes prune() pick the best of
   the recovery MENU {none, bias, ls}, falling back to none.
The cnn domain auto-detects sequential (producer conv -> bn -> consumer) channel targets
and avoids the output head.
Run: python3 src/run_cnn_samples.py
"""

import copy

from utils import make_cnn, cnn_data, count_params
import amct_pytorch as amct
from amct_pytorch.pruning import CNN_RECOVERY_MENU_CFG, PruneReport


def fixed_ratio_channel_prune():
    base = make_cnn()
    calib = cnn_data()
    before = count_params(base)
    for name in ("variance_channel", "reconstruct"):
        model = copy.deepcopy(base)
        cfg = {"methods": {"cnn": {"name": name, "kwargs": {"prune_ratio": 0.3}}}}
        amct.prune(model, cfg, data=calib)
        after = count_params(model)
        _ = model(calib[0])
        print(
            f"[cnn.{name:16s}] channel prune 0.3: params {before:,} -> {after:,} "
            f"(-{100 * (1 - after / before):.1f}%) forward OK"
        )
    model = copy.deepcopy(base)
    amct.prune(model, amct.CNN_RECONSTRUCT_PRUNE_CFG, data=calib)
    print(f"[cnn.preset] CNN_RECONSTRUCT_PRUNE_CFG: params -> {count_params(model):,}")


def recovery_menu_channel_prune():
    model = make_cnn()
    calib, val = cnn_data(n=4), cnn_data(n=2)
    before = count_params(model)
    report = PruneReport()
    amct.prune(model, CNN_RECOVERY_MENU_CFG, data=calib, eval_data=val, report=report)
    _ = model(val[0])
    chosen = next((e.detail for e in report.events if e.stage == "menu-select"), None)
    cut = 1 - report.params_after / report.params_before
    print(
        f"[cnn.recovery_menu] weights {report.params_before:,} -> {report.params_after:,} "
        f"(cut {100 * cut:.1f}%), {chosen}"
    )
    assert count_params(model) < before


def main():
    fixed_ratio_channel_prune()
    recovery_menu_channel_prune()


if __name__ == "__main__":
    main()
