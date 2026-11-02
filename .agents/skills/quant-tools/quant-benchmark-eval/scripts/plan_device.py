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
"""Answer the card-count question before anything is loaded.

`--device_map` and `ASCEND_RT_VISIBLE_DEVICES` must be fixed at launch, yet the
`[device]` line that reveals the right answer only prints while the checkpoint
is being read: guessing wrong costs one full load and then a rejection from
`_check_no_offload`. This script reads `config.json` only -- no weights, no
torch, no NPU -- and shares the size formula with the two evaluation entry
points (`utils.plan_devices`, whose `infer_device_map` wrapper they call), so
the formula exists exactly once and the card list printed here is the one the
run derives.

It is step 1 of the deployment sequence in env-and-portability.md #6.
"""

import argparse
import os
import sys

SKILL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SKILL_DIR)

from utils import plan_devices  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        required=True,
        help="model directory holding config.json (weights are never read)",
    )
    a = ap.parse_args()
    try:
        device_map, cards = plan_devices(a.model)
    except RuntimeError as exc:
        # Size unknowable; the message already says what to pass explicitly.
        print(f"[plan] {exc}")
        return 2
    print(
        f"[plan] pass --device_map {device_map!r} and export "
        f"ASCEND_RT_VISIBLE_DEVICES={cards} before launching"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
