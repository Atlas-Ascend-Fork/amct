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
"""Check quant-benchmark-eval runtime dependencies."""

import argparse
import os
import sys

SKILL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SKILL_DIR)

from utils import check_environment  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--skip_lm_eval", action="store_true", help="do not require lm-eval==0.4.12"
    )
    ap.add_argument(
        "--livecodebench_score",
        action="store_true",
        help="also require lcb_runner for official scoring",
    )
    args = ap.parse_args()

    check_environment(
        require_lm_eval=not args.skip_lm_eval,
        require_lcb_runner=args.livecodebench_score,
    )
    print("[env] ok")


if __name__ == "__main__":
    main()
