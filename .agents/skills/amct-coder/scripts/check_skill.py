#!/usr/bin/env python3
"""amct-coder skill 自检脚本。

集中校验三类内容（对应 validation.md「按变更类型选验证」中 skill 修改的门槛）：
  1. 仓库相对链接的目标文件存在；
  2. 带锚点的链接，锚点在目标文件的标题中可达（GitHub 锚点归一化）；
  3. 快照核对命令：命令中的文件路径存在、搜索的符号在目标文件中可命中。

用法：
  python .agents/skills/amct-coder/scripts/check_skill.py [--fix-hint]

退出码：0 = 全部通过；1 = 有失败项（逐项打印）。外链（https://）不检查。
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = SKILL_DIR.parents[2]


def gh_anchor(heading: str) -> str:
    """GitHub 风格锚点：小写、移除标点（保留中英文与连字符）、空格转连字符。"""
    return re.sub(r"[^\w一-鿿\- ]", "", heading.strip().lower()).replace(" ", "-")


def check_links(files: list[Path]) -> list[str]:
    """检查 markdown 链接的路径存在性与锚点可达性。"""
    issues = []
    for f in files:
        text = f.read_text(encoding="utf-8")
        for m in re.finditer(r"\]\((?!https?://)([^)#\s]+)(?:#([^)\s]+))?\)", text):
            path, anchor = m.group(1), m.group(2)
            target = (f.parent / path).resolve()
            if not target.exists():
                issues.append(f"{f.name}: broken link {path}")
                continue
            if anchor:
                headings = [
                    gh_anchor(h)
                    for h in re.findall(r"^#+ (.+)$", target.read_text(encoding="utf-8"), re.M)
                ]
                if anchor not in headings:
                    issues.append(f"{f.name}: broken anchor {path}#{anchor}")
    return issues


CMD_RE = re.compile(r"（核对：`((?:rg|grep)\s+[^`]*)`）")
FLAG_RE = re.compile(r"^-+\w*$")


def _tokenize(cmd: str) -> list[str] | None:
    """按引号规则分词（引号内含空格算一个 token，引号剥离）；未闭合引号返回 None。"""
    toks, cur, in_q, q = [], [], False, None
    for ch in cmd:
        if in_q:
            if ch == q:
                in_q = False
            else:
                cur.append(ch)
        elif ch in "\"'":
            in_q, q = True, ch
        elif ch.isspace():
            if cur:
                toks.append("".join(cur))
                cur = []
        else:
            cur.append(ch)
    if cur:
        toks.append("".join(cur))
    return None if in_q else toks


def _parse_cmd(cmd_text: str) -> tuple[str, list[str]] | None:
    """解析 rg/grep 命令为 (pattern, paths)；无法可靠解析时返回 None（跳过）。"""
    toks = _tokenize(cmd_text)
    if not toks:
        return None
    # 剥离命令词（rg/grep）与选项
    rest = [t for t in toks[1:] if not FLAG_RE.match(t)]
    if len(rest) < 2:
        return None  # pattern 或 path 缺失（含被 | 截断的命令）
    return rest[0], rest[1:]


def check_snapshot_cmds(files: list[Path]) -> list[str]:
    """执行各手册快照节的核对命令，报告不可命中的。"""
    issues = []
    for f in files:
        for m in CMD_RE.finditer(f.read_text(encoding="utf-8")):
            cmd_text = m.group(1).split("|")[0].strip()  # 去管道与说明文字
            if not cmd_text.startswith(("rg ", "grep ")):
                continue
            parsed = _parse_cmd(cmd_text)
            if parsed is None:
                continue
            pattern, paths = parsed
            resolved = [REPO_ROOT / p for p in paths]
            missing = [str(p) for p in resolved if not p.exists()]
            if missing:
                issues.append(f"{f.name}: snapshot cmd target missing: {missing}")
                continue
            try:
                # 用 rg 执行（与原文命令同一种 regex 方言）；rg 不可用时回退 grep -E
                grep_cmd = ["grep", "-rnE", "-e", pattern, *[str(p) for p in resolved]]
                if subprocess.run(["which", "rg"], capture_output=True).returncode == 0:
                    grep_cmd = ["rg", "-n", pattern, *[str(p) for p in resolved]]
                result = subprocess.run(grep_cmd, capture_output=True, text=True, timeout=30)
                if result.returncode != 0:
                    issues.append(f"{f.name}: snapshot symbol not found: {pattern!r}")
            except (subprocess.TimeoutExpired, OSError) as e:
                issues.append(f"{f.name}: snapshot cmd failed to run: {e}")
    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description="amct-coder skill self-check")
    parser.add_argument("--fix-hint", action="store_true",
                        help="打印修复提示（快照命令失败时提醒以语义为准先核对是否正常重构）")
    args = parser.parse_args()

    md_files = sorted(SKILL_DIR.rglob("*.md"))
    issues = check_links(md_files)
    issues += check_snapshot_cmds(md_files)

    if issues:
        print(f"FAILED: {len(issues)} issue(s)")
        for i in issues:
            print(f"  - {i}")
        if args.fix_hint:
            print(
                "\n提示：快照命令失败时，先核对源码行为与断言语义是否仍一致"
                "（可能只是正常重构挪动了符号/控制流）——语义一致则以语义为准更新"
                "命令，不算快照过期；行为变了才更新断言。见 process-rules「渐进披"
                "露与读取预算」的误报处理。"
            )
        return 1
    print(f"OK: {len(md_files)} files, links/anchors/snapshot-symbols all pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
