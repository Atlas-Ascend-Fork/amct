#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]


def test_wikitext_loads_use_namespaced_dataset_id():
    source_roots = (REPO_ROOT / "amct_pytorch", REPO_ROOT / "examples")
    short_id = re.compile(r"load_dataset\(\s*['\"]wikitext['\"]")
    references = []
    for root in source_roots:
        assert root.is_dir(), f"source root does not exist: {root}"
        for path in root.rglob("*.py"):
            content = path.read_text(encoding="utf-8")
            if short_id.search(content):
                references.append(path.relative_to(REPO_ROOT))

    assert not references, f"Wikitext uses a non-namespaced dataset ID: {references}"
