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

import re
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]


def test_wikitext_loads_use_namespaced_dataset_id():
    # 仅扫描 git 跟踪的 .py 文件，避免误入虚拟环境（.venv*/site-packages）、
    # 构建产物等非仓库源码目录，这些目录下的第三方包可能使用短数据集 ID。
    short_id = re.compile(r"load_dataset\(\s*['\"]wikitext['\"]")
    references = []
    scanned = []
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
        scanned.append(path)
        if short_id.search(path.read_text(encoding="utf-8")):
            references.append(path.relative_to(REPO_ROOT))

    assert scanned, "git ls-files 未扫描到任何文件，请检查 pathspec 或仓库路径是否正确"
    assert not references, f"Wikitext uses a non-namespaced dataset ID: {references}"
