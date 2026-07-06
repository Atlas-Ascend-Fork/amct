#!/usr/bin/env python3
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
"""SubagentStop hook: quant-implementer 自验证检查 + 外循环重试限制（amct 量化 agent）"""

import glob
import json
import logging
import os
import re
import sys
import tempfile
import time

logging.basicConfig(stream=sys.stderr, level=logging.WARNING, format="%(message)s")
logger = logging.getLogger(__name__)

# implementer 的「### 自验证结果」必须覆盖的项（对齐 quant-implementer.md progress 格式）
SELF_VERIFY_KEYWORDS = ["参考 skill", "命令", "指标", "产物"]
MAX_RETRY_COUNT = 10  # implementer + reviewer 各 5 轮

SELF_VERIFY_TEMPLATE = """自验证不完整：progress.md 的「### 自验证结果」缺少：{missing}。
请真正完成对应验证（而非仅补文字记录），结果写入 progress.md 再结束。格式参考：
### 自验证结果
- 参考 skill：<编排指定的 skill>
- 命令：通过 / crash（错误信息）
- 指标：ppl_bf16 / ppl_quant / delta
- 产物：index 0 missing？config.json num_bits/strategy 是否符合方案？"""

RETRY_LIMIT_MSG = ("重试上限：当前阶段已执行 {n} 轮 implementer/reviewer 循环，超过 5 轮上限。"
                   "请回退当前阶段改动，向用户报告阻塞点。")


COUNTER_TTL_SECONDS = 48 * 3600


def cleanup_stale_counters():
    """删除临时目录下超过 48 小时的 hook_retry_*.count 文件（懒清理，静默失败）。"""
    cutoff = time.time() - COUNTER_TTL_SECONDS
    pattern = os.path.join(tempfile.gettempdir(), "hook_retry_*.count")
    for path in glob.glob(pattern):
        try:
            if os.path.getmtime(path) < cutoff:
                os.remove(path)
        except OSError:
            pass


def find_progress_md(cwd):
    candidates = []
    try:
        for dirpath, _dirs, filenames in os.walk(cwd):
            if "/.git/" in dirpath or dirpath.endswith("/.git"):
                continue
            if "progress.md" in filenames:
                candidates.append(os.path.join(dirpath, "progress.md"))
    except OSError:
        return None
    if not candidates:
        return None
    candidates.sort(key=lambda f: os.path.getmtime(f), reverse=True)
    return candidates[0]


def get_current_stage(content):
    """优先取机读状态块 STAGE，回退 '## 阶段 N'"""
    m = re.findall(r"STAGE:\s*(\d+)", content) or re.findall(r"## 阶段\s*(\d+)", content)
    return m[-1] if m else "0"


def check_self_verification(data, content):
    """检查 1：implementer 自验证检查（阻断）"""
    if data.get("agent_type") != "quant-implementer":
        return None
    if "### 自验证结果" not in content:
        return SELF_VERIFY_TEMPLATE.format(missing="、".join(SELF_VERIFY_KEYWORDS))
    start = content.index("### 自验证结果")
    section = content[start:]
    nxt = section.find("\n### ", 1)
    if nxt > 0:
        section = section[:nxt]
    missing = [kw for kw in SELF_VERIFY_KEYWORDS if kw not in section]
    if missing:
        return SELF_VERIFY_TEMPLATE.format(missing="、".join(missing))
    return None


def check_retry_limit(data, content):
    """检查 2：外循环重试限制（阻断）"""
    import fcntl
    if data.get("agent_type", "") not in ("quant-implementer", "quant-reviewer"):
        return None
    stage = get_current_stage(content)
    if stage == "0":
        # 无 STAGE 标记（编排初始化前 / 格式异常）：不计入共享 "0" 计数器，
        # 避免跨阶段累加误触发上限阻断
        return None
    session_id = data.get("session_id")
    if not session_id:
        # 无 session_id：跳过计数，不退化到共享 "unknown" key
        # （否则不同会话/任务的重试计数互相累加，误触发上限阻断）
        return None
    counter_file = os.path.join(tempfile.gettempdir(), f"hook_retry_{session_id}_{stage}.count")
    # flock 保护读-改-写原子性（防极端并发下计数丢失）
    with open(counter_file, "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        try:
            count = int(f.read().strip() or "0")
        except ValueError:
            count = 0
        count += 1
        f.seek(0)
        f.truncate()
        f.write(str(count))
        fcntl.flock(f, fcntl.LOCK_UN)
    if count > MAX_RETRY_COUNT:
        return RETRY_LIMIT_MSG.format(n=count // 2)
    return None


def main():
    cleanup_stale_counters()
    data = json.load(sys.stdin)
    cwd = data.get("cwd", ".")
    progress_path = find_progress_md(cwd)
    if not progress_path:
        sys.exit(0)
    try:
        with open(progress_path) as f:
            content = f.read()
    except OSError:
        sys.exit(0)
    for check in [
        lambda: check_self_verification(data, content),
        lambda: check_retry_limit(data, content),
    ]:
        reason = check()
        if reason:
            logger.error(reason)
            sys.exit(2)
    sys.exit(0)


if __name__ == "__main__":
    main()
