#!/usr/bin/env python3
"""归档 progress.md 的工作区内容到 progress_history.md（阶段推进时调用）。

常驻区与工作区用以下标记分隔：
    <!-- ===== 以上为常驻区，不清除 ===== -->
    <!-- ===== 以下为工作区，阶段推进时归档并清空 ===== -->
常驻区（机读状态块 + 产物契约 + 阶段概览）保留；工作区（各角色本轮记录）归档清空。
"""

import logging
import os
import sys
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

SEPARATOR_END = "<!-- ===== 以上为常驻区，不清除 ===== -->"
SEPARATOR_START = "<!-- ===== 以下为工作区，阶段推进时归档并清空 ===== -->"


def archive_progress(progress_path: str) -> None:
    if not os.path.exists(progress_path):
        logger.error("文件不存在: %s", progress_path)
        sys.exit(1)

    with open(progress_path, "r", encoding="utf-8") as f:
        content = f.read()

    if SEPARATOR_END not in content:
        logger.error("未找到常驻区结束标记，跳过归档。请确保 progress.md 含：%s", SEPARATOR_END)
        sys.exit(1)

    parts = content.split(SEPARATOR_END, 1)
    persistent_section = parts[0] + SEPARATOR_END

    work_section = parts[1] if len(parts) > 1 else ""
    if SEPARATOR_START in work_section:
        work_section = work_section.split(SEPARATOR_START, 1)[1]
    work_section = work_section.strip()

    if not work_section:
        logger.info("工作区为空，无需归档。")
        return

    progress_dir = os.path.dirname(os.path.abspath(progress_path))
    history_path = os.path.join(progress_dir, "progress_history.md")
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    archive_entry = f"\n\n---\n\n## 归档于 {timestamp}\n\n{work_section}\n"

    # 幂等：若该工作区内容已在历史中（上次 progress 清空失败后的重跑），跳过重复归档
    already_archived = False
    if os.path.exists(history_path):
        with open(history_path, "r", encoding="utf-8") as f:
            if work_section in f.read():
                already_archived = True
                logger.info("工作区内容已在历史中，跳过重复归档。")
    if not already_archived:
        if not os.path.exists(history_path):
            header = (
                "<!-- 本文件默认禁止全文读取。需要历史信息时请用 Grep 按关键字查找。 -->\n"
                "# 进度历史归档\n"
            )
            with open(history_path, "w", encoding="utf-8") as f:
                f.write(header)
        with open(history_path, "a", encoding="utf-8") as f:
            f.write(archive_entry)

    # 原子写 progress.md：先写临时文件再 os.replace，崩溃（磁盘满/OOM）不会丢失常驻区/工作区
    new_content = f"{persistent_section}\n\n{SEPARATOR_START}\n\n"
    tmp_path = progress_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(new_content)
    os.replace(tmp_path, progress_path)

    logger.info("归档完成：%d 行 → %s", len(work_section.splitlines()), history_path)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        logger.error("Usage: python3 archive_progress.py <progress.md>")
        sys.exit(1)
    archive_progress(sys.argv[1])
