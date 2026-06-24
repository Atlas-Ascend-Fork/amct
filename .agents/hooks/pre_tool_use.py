#!/usr/bin/env python3
"""PreToolUse hook: 角色越界文件保护（analyzer/reviewer 不改 adapter/bit_config/CLI/算法）。amct 量化 agent。"""

import fnmatch
import json
import logging
import os
import sys

logging.basicConfig(stream=sys.stderr, level=logging.WARNING, format="%(message)s")
logger = logging.getLogger(__name__)

# agent_type -> 不允许修改的文件模式
PROTECTED_PATTERNS = {
    "quant-analyzer": ["*.yaml", "*/common/models/llm/*", "*/amct_pytorch/cli/*", "*/algorithms/quant/*"],
    "quant-reviewer": ["*.yaml", "*/common/models/llm/*", "*/amct_pytorch/cli/*", "*/algorithms/quant/*"],
}


def matches_any(basename, full_path, patterns):
    for pat in patterns:
        if fnmatch.fnmatch(basename, pat) or fnmatch.fnmatch(full_path, pat):
            return True
    return False


def main():
    try:
        data = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        # stdin JSON 异常（管道截断/编码错误）：放行，避免误阻断正常操作
        sys.exit(0)
    agent_type = data.get("agent_type", "")
    file_path = data.get("tool_input", {}).get("file_path", "")
    patterns = PROTECTED_PATTERNS.get(agent_type)
    if patterns and file_path and matches_any(os.path.basename(file_path), file_path, patterns):
        logger.error(
            f"禁止：{agent_type} 不可修改 {file_path}"
            "（adapter 代码 / bit_config yaml / CLI / 算法实现）。改代码请走 quant-implementer。"
        )
        sys.exit(2)
    sys.exit(0)


if __name__ == "__main__":
    main()
