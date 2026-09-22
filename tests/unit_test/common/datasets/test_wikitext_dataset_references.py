#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import re
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]


def test_wikitext_loads_use_namespaced_dataset_id():
    # 仅扫描 git 跟踪的 .py 文件，避免误入虚拟环境（.venv*/site-packages）、
    # 构建产物等非仓库源码目录，这些目录下的第三方包可能使用短数据集 ID。
    short_id = re.compile(r"load_dataset\(\s*['\"]wikitext['\"]")
    references = []
    result = subprocess.run(
        ["git", "ls-files", "-z", "--", "amct_pytorch/*.py", "examples/*.py"],
        cwd=REPO_ROOT,
        capture_output=True,
        check=True,
    )
    for raw in result.stdout.split(b"\0"):
        rel = raw.decode("utf-8").strip()
        if not rel:
            continue
        path = REPO_ROOT / rel
        if short_id.search(path.read_text(encoding="utf-8")):
            references.append(path.relative_to(REPO_ROOT))

    assert not references, f"Wikitext uses a non-namespaced dataset ID: {references}"
