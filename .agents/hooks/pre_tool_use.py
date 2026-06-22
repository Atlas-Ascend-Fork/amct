#!/usr/bin/env python3
"""PreToolUse hook: 角色越界文件保护 + 改 adapter 代码前必读 progress.md（amct 量化 agent）"""

import fnmatch
import json
import logging
import os
import sys

logging.basicConfig(stream=sys.stderr, level=logging.WARNING, format="%(message)s")
logger = logging.getLogger(__name__)

# agent_type -> 不允许修改的文件模式：analyzer/reviewer 不碰 adapter 代码 / bit_config / CLI / 算法实现
PROTECTED_PATTERNS = {
    "quant-analyzer": ["*.yaml", "*/common/models/llm/*", "*/amct_pytorch/cli/*", "*/algorithms/quant/*"],
    "quant-reviewer": ["*.yaml", "*/common/models/llm/*", "*/amct_pytorch/cli/*", "*/algorithms/quant/*"],
}
# 改 adapter 代码前必读 progress.md 的文件模式
CODE_PATTERNS = ["*/common/models/llm/*"]


def matches_any(basename, full_path, patterns):
    for pat in patterns:
        if fnmatch.fnmatch(basename, pat) or fnmatch.fnmatch(full_path, pat):
            return True
    return False


def check_role_protection(data):
    """检查 1：角色越界文件保护（阻断）"""
    agent_type = data.get("agent_type", "")
    agent_id = data.get("agent_id", "")
    file_path = data.get("tool_input", {}).get("file_path", "")
    if not agent_id or not file_path:
        return None
    patterns = PROTECTED_PATTERNS.get(agent_type)
    if not patterns:
        return None
    if matches_any(os.path.basename(file_path), file_path, patterns):
        return (f"禁止：{agent_type} 不可修改 {file_path}"
                "（adapter 代码 / bit_config yaml / CLI / 算法实现）。改代码请走 quant-implementer。")
    return None


def check_read_progress_first(data):
    """检查 2：改 adapter 代码前必读 progress.md（阻断）"""
    agent_id = data.get("agent_id", "")
    file_path = data.get("tool_input", {}).get("file_path", "")
    if not agent_id or not file_path:
        return None
    if not matches_any(os.path.basename(file_path), file_path, CODE_PATTERNS):
        return None
    marker = f"/tmp/hook_read_progress_{agent_id}.marker"
    if os.path.exists(marker):
        return None
    return "禁止：修改 adapter 代码前必须先读 progress.md（了解当前阶段方案与实施记录）后再改。"


def main():
    data = json.load(sys.stdin)
    for check in [check_role_protection, check_read_progress_first]:
        reason = check(data)
        if reason:
            logger.error(reason)
            sys.exit(2)
    sys.exit(0)


if __name__ == "__main__":
    main()
