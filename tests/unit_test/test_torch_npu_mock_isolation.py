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
"""Regression tests for issue #310: torch_npu mock pollution.

``tests/conftest.py`` must import the real torch_npu only when an Ascend NPU
is attached (/dev/davinciN nodes visible) and must fall back to a MagicMock
otherwise, and test modules must not replace ``sys.modules["torch_npu"]``
at import (collection) time without restoring it.
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFTEST = REPO_ROOT / "tests" / "conftest.py"
BLOCKWISE_TEST = (
    REPO_ROOT
    / "tests"
    / "unit_test"
    / "common"
    / "optimization"
    / "test_blockwise_solver.py"
)


def _glob_prologue(davinci_nodes):
    return f"""
        import glob
        _real_glob = glob.glob
        def _patched_glob(pattern, *args, **kwargs):
            if 'davinci' in pattern:
                return {davinci_nodes!r}
            return _real_glob(pattern, *args, **kwargs)
        glob.glob = _patched_glob
    """


def _run_child(code, extra_pythonpath=None):
    env = dict(os.environ)
    if extra_pythonpath is not None:
        env["PYTHONPATH"] = (
            str(extra_pythonpath) + os.pathsep + env.get("PYTHONPATH", "")
        )
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        env=env,
        cwd=REPO_ROOT,
        check=False,
    )


def _make_fake_torch_npu(root, marker):
    pkg = root / "torch_npu"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text(f"FAKE_TORCH_NPU = {marker!r}\n")
    return root


def _exec_conftest():
    return f"""
        src = open({str(CONFTEST)!r}).read()
        exec(compile(src, 'conftest.py', 'exec'))
    """


def test_conftest_keeps_importable_torch_npu_with_device(tmp_path):
    fake_root = _make_fake_torch_npu(tmp_path / "good", "good")
    code = (
        _glob_prologue(["/dev/davinci0"])
        + _exec_conftest()
        + """
        import sys
        import torch_npu
        assert getattr(torch_npu, 'FAKE_TORCH_NPU', None) == 'good', repr(torch_npu)
        print('OK')
    """
    )
    result = _run_child(code, extra_pythonpath=fake_root)
    assert result.returncode == 0, result.stdout + result.stderr


def test_conftest_falls_back_to_mock_when_import_fails(tmp_path):
    fake_root = tmp_path / "broken"
    pkg = fake_root / "torch_npu"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text(
        "raise ImportError('simulated broken torch_npu')\n"
    )
    code = (
        _glob_prologue(["/dev/davinci0"])
        + _exec_conftest()
        + """
        import sys
        import torch_npu
        import torch
        from unittest.mock import MagicMock
        assert isinstance(sys.modules['torch_npu'], MagicMock), repr(sys.modules['torch_npu'])
        assert sys.modules['torch_npu'].__spec__ is not None
        assert hasattr(torch, 'npu')
        print('OK')
    """
    )
    result = _run_child(code, extra_pythonpath=fake_root)
    assert result.returncode == 0, result.stdout + result.stderr


def test_conftest_mocks_importable_torch_npu_without_device(tmp_path):
    fake_root = _make_fake_torch_npu(tmp_path / "unused", "unused")
    code = (
        _glob_prologue([])
        + _exec_conftest()
        + """
        import sys
        import torch_npu
        from unittest.mock import MagicMock
        assert isinstance(sys.modules['torch_npu'], MagicMock), repr(sys.modules['torch_npu'])
        assert getattr(torch_npu, 'FAKE_TORCH_NPU', None) != 'unused'
        print('OK')
    """
    )
    result = _run_child(code, extra_pythonpath=fake_root)
    assert result.returncode == 0, result.stdout + result.stderr


def test_blockwise_solver_module_preserves_torch_npu():
    code = f"""
        import os
        import sys
        import types
        os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
        sentinel = types.ModuleType('torch_npu')
        sentinel.MARKER = 'sentinel'
        sys.modules['torch_npu'] = sentinel
        src = open({str(BLOCKWISE_TEST)!r}).read()
        exec(compile(src, {str(BLOCKWISE_TEST)!r}, 'exec'))
        assert sys.modules['torch_npu'] is sentinel, (
            'module-level assignment replaced sys.modules torch_npu: '
            + repr(sys.modules['torch_npu'])
        )
        print('OK')
    """
    result = _run_child(code)
    assert result.returncode == 0, result.stdout + result.stderr
