# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

import os

from loguru import logger


def ensure_log_dir(args):
    log_dir = getattr(args, "log_dir", "") or os.path.join(args.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    args.log_dir = log_dir
    return log_dir


def setup_run_logging(args, task_name: str):
    log_dir = ensure_log_dir(args)
    log_path = os.path.join(log_dir, f"{task_name}.log")
    sink_id = logger.add(
        log_path,
        level="INFO",
        encoding="utf-8",
        backtrace=False,
        diagnose=False,
        enqueue=False,
    )
    return sink_id, log_path
