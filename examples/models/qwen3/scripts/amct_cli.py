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
"""Launch an amct_pytorch CLI module after masking a broken torchaudio import.

Some images ship a torchaudio build that crashes while transformers imports
RNNT losses. Qwen3 does not need torchaudio. Usage:

    python3 scripts/amct_cli.py amct_pytorch.eval --trust_remote_code ...
"""

from __future__ import annotations

import importlib.util
import runpy
import sys


def _disable_torchaudio():
    if getattr(importlib.util.find_spec, "_amct_torchaudio_patched", False):
        return
    original_find_spec = importlib.util.find_spec

    def _find_spec(name, package=None):
        if name == "torchaudio":
            return None
        return original_find_spec(name, package)

    _find_spec._amct_torchaudio_patched = True  # type: ignore[attr-defined]
    importlib.util.find_spec = _find_spec


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: amct_cli.py <module> [args...]")
    _disable_torchaudio()
    module = sys.argv[1]
    sys.argv = [module, *sys.argv[2:]]
    runpy.run_module(module, run_name="__main__")


if __name__ == "__main__":
    main()
