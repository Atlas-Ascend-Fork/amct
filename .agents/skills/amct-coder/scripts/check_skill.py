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
"""amct-coder skill 自检脚本。

集中校验四类内容：
  1. 仓库相对链接的目标文件存在；
  2. 带锚点的链接，锚点在目标文件的标题中可达（GitHub 锚点归一化）；
  3. 纯文本节名引用（「X」）与本 skill 内标题的一致性——没有锚点可查的散文式引用
     一旦标题改名就会静默失效，用 difflib 近似度兜住；
  4. 核对命令：覆盖文中全部「核对：`cmd`」标记，按断言方向判定——
     必须命中 / 必须无命中 / 命中受路径限制 / 需上下文佐证。
     工具白名单先于断言方向判定：非 rg|grep 命令一律显式列为「待人工执行」，
     不落 argv、不执行——带「应无命中」注解的脚本命令同样不例外；
     rg 不在 PATH 时降级为等价 GNU grep；工具完全不可执行、或一条都没判定成
     （零覆盖）都判失败，不静默跳过。

用法：
  python .agents/skills/amct-coder/scripts/check_skill.py [--fix-hint] [--verbose]
  python .agents/skills/amct-coder/scripts/check_skill.py --self-test   # 验证判定方向本身

退出码：0 = 全部通过；1 = 有失败项（逐项打印）。外链（https://）不检查。
"""

from __future__ import annotations

import argparse
import difflib
import re
import shutil
import subprocess
import sys
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = SKILL_DIR.parents[2]
RUN_TIMEOUT = 60

# rg 在部分环境是 shell function 而非 PATH 上的可执行文件，subprocess 看不到它
# （`command -v rg` 显示 function、`shutil.which("rg")` 返回 None）。按 which 解析，
# 缺失时降级为等价 GNU grep；两者都没有才判失败。
RG_BIN = shutil.which("rg")
GREP_BIN = shutil.which("grep")


# --------------------------------------------------------------------------
# 1/2. 链接与锚点
# --------------------------------------------------------------------------
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
                    for h in re.findall(
                        r"^#+ (.+)$", target.read_text(encoding="utf-8"), re.M
                    )
                ]
                if anchor not in headings:
                    issues.append(f"{f.name}: broken anchor {path}#{anchor}")
    return issues


# --------------------------------------------------------------------------
# 3. 纯文本节名引用（「X」）
# --------------------------------------------------------------------------
QUOTED_RE = re.compile(r"「([^」\n]{2,40})」")
PAREN_RE = re.compile(r"[（(][^）)]*[）)]")
HEADING_NUM_RE = re.compile(r"^[〇一二三四五六七八九十]+、")
SIM_THRESHOLD = 0.72
# 近似某标题、但本来就不是在引用该节的散文用词，逐条给出豁免理由。
# 新增豁免必须写理由，否则等于给未来的漂移开口子。
QUOTED_WHITELIST = {
    # validation.md「场景测试索引」的表行标签，同名标题是另一节「公共接口与打包检查」
    "公共接口与打包": "表行标签（validation.md 场景测试索引），非节名引用",
    # 完整示范（walkthrough.md）里的散文措辞
    "已验证": "散文措辞，与标题「验证」无关",
}


def _heading_keys(files: list[Path]) -> dict[str, str]:
    """收集全部标题的可引用形态：原文、去括注、去中文序号前缀。"""
    keys: dict[str, str] = {}
    for f in files:
        for h in re.findall(r"^#+ (.+)$", f.read_text(encoding="utf-8"), re.M):
            h = h.strip()
            no_paren = PAREN_RE.sub("", h).strip()
            for variant in {
                h,
                no_paren,
                HEADING_NUM_RE.sub("", h).strip(),
                HEADING_NUM_RE.sub("", no_paren).strip(),
            }:
                if variant:
                    keys.setdefault(variant, f"{f.name}「{h}」")
    return keys


def check_quoted_refs(files: list[Path]) -> list[str]:
    """散文里的节名引用必须与实际标题精确一致。

    check_links 只覆盖 `[文本](path#anchor)`；没有锚点的纯文本「X」改名后不会报错，
    读者按「X」找不到节。近似度 >= SIM_THRESHOLD 却不精确相等即判漂移。
    """
    keys = _heading_keys(files)
    names = list(keys)
    issues = []
    for f in files:
        for m in QUOTED_RE.finditer(f.read_text(encoding="utf-8")):
            ref = m.group(1).strip()
            if ref in keys or ref in QUOTED_WHITELIST:
                continue
            best = max(
                names, key=lambda n: difflib.SequenceMatcher(None, ref, n).ratio()
            )
            ratio = difflib.SequenceMatcher(None, ref, best).ratio()
            if ratio >= SIM_THRESHOLD:
                issues.append(
                    f"{f.name}: 节名引用「{ref}」近似 {keys[best]}（相似度 {ratio:.2f}）"
                    f"但不一致——改成精确节名，或确认是散文用词后加进 QUOTED_WHITELIST"
                )
    return issues


# --------------------------------------------------------------------------
# 4. 核对命令
# --------------------------------------------------------------------------
MARKER_RE = re.compile(r"核对[:：]")
CMD_SPAN_RE = re.compile(r"`((?:rg|grep|bash|sh)\b[^`]*)`")
IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
FLAG_RE = re.compile(r"^-+\w*$")
# 断言方向
ABSENT_RE = re.compile(r"应无命中|不应命中|无命中即|没有命中")
ONLY_RE = re.compile(r"应只命中\s*`([^`]+)`|只命中\s*`([^`]+)`\s*下")
# 需要上下文佐证的结构性断言（仅单命令组适用，避免跨命令误归属）
STRUCT_RE = re.compile(r"应看到|确认|即证据|位于|分支|结构|两支|紧跟")


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


def _split_pipeline(cmd: str) -> list[str]:
    """在引号外按 `|` 切分管道；引号内的 `|`（如 rg 的 alternation）不切。"""
    segs, cur, in_q, q = [], [], False, None
    for ch in cmd:
        if in_q:
            cur.append(ch)
            if ch == q:
                in_q = False
        elif ch in "\"'":
            in_q, q = True, ch
            cur.append(ch)
        elif ch == "|":
            segs.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    segs.append("".join(cur).strip())
    return [s for s in segs if s]


_CTX_FLAGS = frozenset("ABC")


def _rg_to_grep_flags(flags: list[str]) -> list[str]:
    """rg flags -> GNU grep 等价项：补 `-r -E`（ERE 才认 alternation），`-A8` 拆成 `-A 8`。

    只覆盖核对命令实际用到的 flag（-n / -A / -B / -C）；其余原样透传——真不被 grep
    接受时 rc>=2 会被判为「命令执行错误」，不会静默通过。
    """
    out = ["-n", "-r", "-E"]
    for fl in flags:
        body = fl.lstrip("-")
        if not body or fl == "-n":
            continue  # -n 已在 out 里；空 flag 忽略
        if body[0] in _CTX_FLAGS:
            out.append(f"-{body[0]}")
            if body[1:].isdigit():
                out.append(body[1:])
        else:
            out.append(fl)
    return out


def _line_bounds(text: str, pos: int) -> tuple[int, int]:
    ls = text.rfind("\n", 0, pos) + 1
    le = text.find("\n", pos)
    return ls, (len(text) if le == -1 else le)


def _closing_paren(text: str, start: int, limit: int) -> int:
    """从 start 扫到第一个未被反引号包裹的 `）`；同行内找不到则退到行尾。"""
    in_q = False
    for i in range(start, limit):
        ch = text[i]
        if ch == "`":
            in_q = not in_q
        elif ch == "）" and not in_q:
            return i + 1
    return limit


def _marker_spans(text: str) -> tuple[list[tuple[int, int]], int]:
    """定位每个「核对」组范围；返回 (spans, 散文式标记数)。"""
    spans, prose = [], 0
    for m in MARKER_RE.finditer(text):
        ls, le = _line_bounds(text, m.start())
        if "`" not in text[m.end() : le]:
            prose += 1  # 散文（如「至少核对：」后接列表），不是命令标记
            continue
        open_idx = text.rfind("（", ls, m.start())
        start = open_idx if open_idx != -1 else m.start()
        spans.append((start, _closing_paren(text, start, le)))
    # 去重并按起点排序（同一 （ 被多个 marker 命中时）
    uniq = sorted(set(spans))
    merged: list[tuple[int, int]] = []
    for s, e in uniq:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged, prose


class Snapshot:
    """一条待判定的核对命令。"""

    def __init__(self, file: Path, raw: str, ann: str):
        self.file = file
        self.raw = raw
        self.ann = ann
        self.segments = _split_pipeline(raw)
        self.toks = _tokenize(self.segments[0]) or []
        self.tool = self.toks[0] if self.toks else ""
        rest = self.toks[1:]
        self.flags = [t for t in rest if FLAG_RE.match(t)]
        body = [t for t in rest if not FLAG_RE.match(t)]
        self.pattern = body[0] if body else ""
        self.paths = body[1:] if len(body) > 1 else []
        self.filters = self.segments[1:]
        # 工具白名单先判：非 rg/grep 一律 unsupported，绝不进 absent/restricted/present。
        # 若把断言方向排在前面，一条带「应无命中」的 bash/sh 命令会被判为 absent，
        # 绕过人工执行跳过，被 argv() 拼好后在 REPO_ROOT 下真的执行。
        if self.tool not in ("rg", "grep"):
            self.kind = "unsupported"
        elif ABSENT_RE.search(ann):
            self.kind = "absent"
        elif ONLY_RE.search(ann) or self.filters:
            self.kind = "restricted"
        else:
            self.kind = "present"
        m = ONLY_RE.search(ann)
        self.only_prefix = next((g for g in (m.groups() if m else ()) if g), "")
        # 结构性断言的佐证 token：仅单命令组，避免把邻接断言的符号误归属
        self.evidence: list[str] = []
        if self.kind == "present" and not self.filters and STRUCT_RE.search(ann):
            ann_no_cmd = CMD_SPAN_RE.sub("", ann)
            for span in re.findall(r"`([^`]+)`", ann_no_cmd):
                self.evidence.extend(IDENT_RE.findall(span))
            self.evidence = sorted(set(self.evidence))
            if self.evidence:
                self.kind = "present+evidence"

    @property
    def valid(self) -> bool:
        return bool(self.pattern) and bool(self.paths)

    def argv(self) -> tuple[list[str], bool]:
        """解析成可执行 argv；返回 (argv, 是否由 rg 降级为 grep)。

        工具不在 PATH 时原样返回 argv，让 _run 抛 OSError，由调用方判为失败——
        「环境跑不了」不能等同于「断言通过」。
        只接受 rg/grep：非白名单工具（bash/sh 等）一律不落 argv，把「静默执行
        任意命令」变成显式报错，即使将来断言方向的分派被改错也不会执行。
        """
        if self.tool not in ("rg", "grep"):
            raise ValueError(f"拒绝为非白名单工具构造 argv: {self.tool!r}")
        if self.tool == "rg":
            if RG_BIN:
                flags = list(dict.fromkeys(["-n", *self.flags]))
                return [RG_BIN, *flags, "-e", self.pattern, *self.paths], False
            if GREP_BIN:
                flags = _rg_to_grep_flags(self.flags)
                return [GREP_BIN, *flags, "-e", self.pattern, *self.paths], True
        elif self.tool == "grep":
            bin_ = GREP_BIN or self.tool
            return [bin_, *self.flags, "-e", self.pattern, *self.paths], False
        return [self.tool, *self.flags, "-e", self.pattern, *self.paths], False


def _run(argv: list[str]) -> tuple[int, str, str]:
    r = subprocess.run(
        argv, cwd=REPO_ROOT, capture_output=True, text=True, timeout=RUN_TIMEOUT
    )
    return r.returncode, r.stdout, r.stderr


def _apply_filters(lines: list[str], filters: list[str]) -> list[str]:
    """按管道后续 grep 段过滤命中行（-v 反选，其余正选）。"""
    out = lines
    for seg in filters:
        toks = _tokenize(seg) or []
        if not toks or toks[0] != "grep":
            continue
        negate = any("v" in t for t in toks[1:] if FLAG_RE.match(t))
        body = [t for t in toks[1:] if not FLAG_RE.match(t)]
        if not body:
            continue
        try:
            rx = re.compile(body[0])
            hit = lambda ln: rx.search(ln) is not None  # noqa: E731
        except re.error:
            hit = lambda ln, s=body[0]: s in ln  # noqa: E731
        out = [ln for ln in out if hit(ln) is not negate]
    return out


def _hit_paths(lines: list[str]) -> list[str]:
    return [ln.split(":", 1)[0] for ln in lines if ":" in ln]


def check_verify_cmds(
    files: list[Path], verbose: bool = False
) -> tuple[list[str], list[str], dict]:
    """执行全部核对命令。返回 (失败项, 跳过项, 统计)。"""
    issues, skipped = [], []
    stats = {
        "markers": 0,
        "prose": 0,
        "cmds": 0,
        "kinds": {},
        "verified": 0,
        "degraded": 0,
    }

    for f in files:
        text = f.read_text(encoding="utf-8")
        spans, prose = _marker_spans(text)
        stats["markers"] += len(spans)
        stats["prose"] += prose
        for s, e in spans:
            seg = text[s:e]
            cmds = CMD_SPAN_RE.findall(seg)
            if not cmds:
                issues.append(f"{f.name}: 核对标记内未解析出命令 -> {seg[:70]}...")
                continue
            ann = CMD_SPAN_RE.sub("", seg)
            for raw in cmds:
                snap = Snapshot(f, raw, ann if len(cmds) == 1 else "")
                stats["cmds"] += 1
                stats["kinds"][snap.kind] = stats["kinds"].get(snap.kind, 0) + 1
                tag = f"{f.name}: [{snap.kind}] {raw[:88]}"

                if snap.kind == "unsupported":
                    skipped.append(f"{tag} —— 非 rg/grep，需人工执行")
                    continue
                if not snap.valid:
                    issues.append(f"{tag} —— 无法解析 pattern/paths")
                    continue
                missing = [p for p in snap.paths if not (REPO_ROOT / p).exists()]
                if missing:
                    issues.append(f"{tag} —— 搜索路径不存在: {missing}")
                    continue
                argv, degraded = snap.argv()
                if degraded:
                    stats["degraded"] += 1
                try:
                    rc, out, err = _run(argv)
                except subprocess.TimeoutExpired as exc:
                    skipped.append(f"{tag} —— 执行超时（>{RUN_TIMEOUT}s）: {exc}")
                    continue
                except OSError as exc:
                    issues.append(
                        f"{tag} —— 搜索工具不可执行（{exc}）；环境缺 rg/grep 属阻塞，不算通过"
                    )
                    continue
                if rc >= 2:
                    issues.append(f"{tag} —— 命令执行错误 rc={rc}: {err.strip()[:120]}")
                    continue
                stats["verified"] += 1  # 命令真跑起来了，下面的方向判定才算数

                lines = [ln for ln in out.splitlines() if ln.strip()]
                if snap.kind == "absent":
                    if rc == 0:
                        issues.append(
                            f"{tag} —— 断言为「应无命中」，实际命中 {len(lines)} 行（论断已过期）"
                        )
                    elif verbose:
                        print(f"  ok   {tag} (无命中，符合断言)")
                elif snap.kind == "restricted":
                    kept = _apply_filters(lines, snap.filters)
                    if not kept:
                        issues.append(
                            f"{tag} —— 过滤后无命中，断言「应只命中 {snap.only_prefix}」失去证据"
                        )
                        continue
                    if snap.only_prefix:
                        bad = [
                            p
                            for p in _hit_paths(kept)
                            if not p.startswith(snap.only_prefix)
                        ]
                        if bad:
                            issues.append(
                                f"{tag} —— 命中越出 {snap.only_prefix}: {sorted(set(bad))[:3]}（论断已过期）"
                            )
                            continue
                    if verbose:
                        print(
                            f"  ok   {tag} (命中 {len(kept)} 行，限于 {snap.only_prefix or '过滤条件'})"
                        )
                else:  # present / present+evidence
                    if rc != 0 or not lines:
                        issues.append(f"{tag} —— 符号未命中: {snap.pattern!r}")
                        continue
                    if snap.kind == "present+evidence":
                        miss = [t for t in snap.evidence if t not in out]
                        if miss:
                            issues.append(
                                f"{tag} —— 上下文佐证缺失 {miss}；"
                                f"先判断是正常重构（语义仍一致则扩大 -A/-B 或改佐证词）还是行为已变"
                            )
                            continue
                    if verbose:
                        extra = f"，佐证 {snap.evidence}" if snap.evidence else ""
                        print(f"  ok   {tag} (命中 {len(lines)} 行{extra})")
    return issues, skipped, stats


# --------------------------------------------------------------------------
# 自检：验证判定方向本身没退化（负向用例必须被检出）
# --------------------------------------------------------------------------
SELF_TEST_CASES = [
    # (说明, 核对标记文本, 期望是否被检出为 issue)
    (
        "absent 断言实际有命中",
        "（核对：`grep -n \"def solve\" amct_pytorch/common/optimization/blockwise_solver.py` 应无命中）",
        True,
    ),
    (
        "restricted 断言命中越出限定目录",
        "（核对：`grep -rn \"def finalize\" amct_pytorch/ tests/ | grep -v \"zzz\"` 应只命中 `tests/` 下文件）",
        True,
    ),
    (
        "present 断言符号不存在",
        "（核对：`rg -n \"SYMBOL_SHOULD_NOT_EXIST_XYZ\" amct_pytorch/cli/llm/args.py`）",
        True,
    ),
    (
        "present+evidence 佐证 token 缺失",
        "（核对：`rg -n -A8 \"def finalize\" amct_pytorch/common/optimization/base_solver.py`，"
        "应看到 `ZZZ_missing_evidence` 分派）",
        True,
    ),
    (
        "搜索路径不存在",
        "（核对：`rg -n \"import\" amct_pytorch/no_such_dir/x.py`）",
        True,
    ),
    (
        "absent 断言确实无命中",
        "（核对：`grep -n \"self\\._convert_tensor\\b\" amct_pytorch/workflows/llm_deploy.py` 应无命中）",
        False,
    ),
    (
        "restricted 断言命中落在限定目录内",
        "（核对：`grep -rn \"\\.export_unit(\" amct_pytorch/ tests/ | grep -v \"def export_unit\"` "
        "应只命中 `tests/` 下文件）",
        False,
    ),
    ("present 断言符号存在", "（核对：`rg -n \"AMCT_EXPERIMENTAL\" setup.py`）", False),
    ("非 rg/grep 命令应跳过而非静默通过", "（核对：`bash build.sh --help`）", False),
]


def self_test() -> int:
    """跑合成用例，确认四类断言方向与跳过语义都没退化。"""
    import tempfile

    failures = []
    with tempfile.TemporaryDirectory() as td:
        for idx, (name, marker, expect_issue) in enumerate(SELF_TEST_CASES):
            f = Path(td) / f"case{idx}.md"
            f.write_text(f"论断。{marker}\n", encoding="utf-8")
            issues, skipped, _ = check_verify_cmds([f])
            got_issue = bool(issues)
            if got_issue != expect_issue:
                failures.append(
                    f"{name}: 期望 issue={expect_issue}，实际 issue={got_issue} "
                    f"(issues={issues[:1]}, skipped={skipped[:1]})"
                )
        # 节名引用：漂移必须被检出，精确一致与白名单必须放过
        head = Path(td) / "quoted_head.md"
        head.write_text(
            "## 跨端约定缺失的处理形态\n\n## 验证\n\n正文。\n", encoding="utf-8"
        )
        quoted_cases = [
            # 历史上真实发生过的漂移：多一个「时」字，锚点检查查不出来
            ("跨端约定缺失时的处理形态", True),
            ("跨端约定缺失的处理形态", False),  # 精确一致
            ("已验证", False),  # 白名单散文用词（近似标题「验证」）
        ]
        for idx, (ref, expect_issue) in enumerate(quoted_cases):
            user = Path(td) / f"quoted_ref{idx}.md"
            user.write_text(f"见「{ref}」。\n", encoding="utf-8")
            got = bool(check_quoted_refs([head, user]))
            if got != expect_issue:
                failures.append(
                    f"节名引用「{ref}」: 期望 issue={expect_issue}，实际 issue={got}"
                )

        # 跳过语义单独确认：非 rg/grep 必须进 skipped，不能算通过也不能算失败
        f = Path(td) / "skip.md"
        f.write_text("论断。（核对：`bash build.sh --help`）\n", encoding="utf-8")
        _, skipped, stats = check_verify_cmds([f])
        if not skipped or stats["kinds"].get("unsupported") != 1:
            failures.append(
                f"unsupported 命令未被显式跳过: skipped={skipped}, kinds={stats['kinds']}"
            )

        # 工具白名单优先级守卫：否定式注解不得让非 rg/grep 命令绕过跳过而被执行。
        # 三重断言——必须进 skipped、kind 必须是 unsupported、argv() 必须拒绝构造；
        # 任一条被改回「断言方向优先」都会在此暴露（脚本路径故意指向不存在的文件，
        # 即使守卫失效也不会执行到任何真命令）。
        guard_raw = "sh ./no_such_probe_script.sh setup.py"
        guard_file = Path(td) / "skip_absent.md"
        guard_file.write_text(
            f"论断。（核对：`{guard_raw}` 应无命中）\n", encoding="utf-8"
        )
        g_issues, g_skipped, g_stats = check_verify_cmds([guard_file])
        if g_issues or not g_skipped or g_stats["kinds"].get("unsupported") != 1:
            failures.append(
                "带「应无命中」的非 rg/grep 命令未被跳过（可能已被真实执行）: "
                f"issues={g_issues}, skipped={g_skipped}, kinds={g_stats['kinds']}"
            )
        snap = Snapshot(guard_file, guard_raw, "（核对： 应无命中）")
        if snap.kind != "unsupported":
            failures.append(
                f"非白名单工具 + 否定式注解未判为 unsupported: kind={snap.kind}"
            )
        try:
            snap.argv()
        except ValueError:
            pass
        else:
            failures.append("argv() 未拒绝非白名单工具，存在静默执行任意命令的风险")

    if failures:
        print(f"SELF-TEST FAILED: {len(failures)} case(s)")
        for x in failures:
            print(f"  - {x}")
        return 1
    print(
        f"SELF-TEST OK: {len(SELF_TEST_CASES)} 个核对命令用例 + 3 个节名引用用例"
        "（含 6 个负向用例均被检出）+ 工具白名单优先级守卫 3 项"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="amct-coder skill self-check")
    parser.add_argument(
        "--fix-hint",
        action="store_true",
        help="打印修复提示（核对命令失败时提醒以语义为准先核对是否正常重构）",
    )
    parser.add_argument("--verbose", action="store_true", help="逐条打印通过的核对命令")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="只跑合成用例，验证四类断言方向与跳过语义未退化",
    )
    args = parser.parse_args()

    if args.self_test:
        return self_test()

    md_files = sorted(SKILL_DIR.rglob("*.md"))
    issues = check_links(md_files)
    issues += check_quoted_refs(md_files)
    verify_issues, skipped, stats = check_verify_cmds(md_files, verbose=args.verbose)
    issues += verify_issues

    kinds = " ".join(f"{k}={v}" for k, v in sorted(stats["kinds"].items()))
    # unsupported = 设计上就要人工执行的（如 bash build.sh --help），不计入机器可判定分母
    executable = stats["cmds"] - stats["kinds"].get("unsupported", 0)
    degrade_note = (
        f"；rg 不在 PATH，降级为 grep {stats['degraded']} 条"
        if stats["degraded"]
        else ""
    )
    print(
        f"coverage: {stats['markers']} 个核对标记 -> {stats['cmds']} 条命令 "
        f"({kinds})；散文式标记 {stats['prose']} 个（非命令）；"
        f"已实际判定 {stats['verified']}/{executable} 条；未自动判定 {len(skipped)} 条{degrade_note}"
    )
    for s in skipped:
        print(f"  SKIP {s}")

    # 零覆盖守卫：一条都没判定成就报 OK，等价于用「没测」冒充「测过」
    if executable and stats["verified"] == 0:
        issues.append(
            f"零覆盖：{executable} 条可机器判定的核对命令一条都没跑成，不能报 OK"
            "（确认 rg 或 grep 在 PATH 上可执行；rg 是 shell function 时 subprocess 看不到）"
        )

    if issues:
        print(f"FAILED: {len(issues)} issue(s)")
        for i in issues:
            print(f"  - {i}")
        if args.fix_hint:
            print(
                "\n提示：核对命令失败时，先核对源码行为与断言语义是否仍一致"
                "（可能只是正常重构挪动了符号/控制流）——语义一致则以语义为准更新"
                "命令，不算「当前事实」过期；行为变了才更新断言。见 conditional-rules"
                "「当前事实与核对命令纪律」的误报处理。反向断言（应无命中）失败说明该 helper "
                "已接通，要改的是手册论断本身。"
            )
        return 1
    print(
        f"OK: {len(md_files)} files；links/anchors/节名引用 全通过；"
        f"核对断言实际判定 {stats['verified']}/{executable} 条"
        f"（{stats['kinds'].get('unsupported', 0)} 条需人工执行）"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
