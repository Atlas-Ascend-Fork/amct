# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Smoke CLI must select a real case before loading a model or reporting PASS."""

import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "smoke/run_smoke.py"
BASH = shutil.which("bash")


class SmokeCaseSelectionTest(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location("smoke_selection_test", SCRIPT)
        self.module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"torch": ModuleType("torch")}):
            spec.loader.exec_module(self.module)
        self.amct = ModuleType("amct_pytorch")
        self.amct.HIFP8_CAST_CFG = {}

    def assert_empty_selection_rejected(self, cases):
        argv = [str(SCRIPT), "--device", "cpu", "--cases", cases]
        with (
            patch.dict(sys.modules, {"amct_pytorch": self.amct}),
            patch.object(sys, "argv", argv),
            patch.object(self.module.logging, "basicConfig"),
            patch.object(
                self.module, "_load_base_model", return_value=object()
            ) as loader,
        ):
            with self.assertRaises(SystemExit) as caught:
                self.module.main()
            self.assertEqual(caught.exception.code, 1)
            loader.assert_not_called()

    def test_empty_string_rejected(self):
        self.assert_empty_selection_rejected("")

    def test_whitespace_rejected(self):
        self.assert_empty_selection_rejected("   ")

    def test_only_separators_rejected(self):
        self.assert_empty_selection_rejected(", ,")

    def test_valid_case_selection_unchanged(self):
        argv = [str(SCRIPT), "--device", "cpu", "--cases", " minmax "]
        with (
            patch.dict(sys.modules, {"amct_pytorch": self.amct}),
            patch.object(sys, "argv", argv),
            patch.object(self.module.logging, "basicConfig"),
            patch.object(self.module, "_run_all", return_value=True) as runner,
        ):
            with self.assertRaises(SystemExit) as caught:
                self.module.main()
            self.assertEqual(caught.exception.code, 0)
            self.assertEqual(
                [case["name"] for case in runner.call_args.args[0]], ["minmax"]
            )


@unittest.skipUnless(BASH, "Bash is required")
class SmokeShellCaseSelectionTest(unittest.TestCase):
    def run_wrapper(self, cases):
        env = os.environ.copy()
        with tempfile.TemporaryDirectory(
            dir=os.getenv("AMCT_SMOKE_TEST_TMPDIR")
        ) as directory:
            root = Path(directory)
            script = root / "run_smoke.sh"
            startup = root / "startup.sh"
            calls = root / "python-calls.txt"
            with script.open("w", encoding="utf-8", newline="\n") as stream:
                stream.write(SCRIPT.with_suffix(".sh").read_text(encoding="utf-8"))
            with startup.open("w", encoding="utf-8", newline="\n") as stream:
                stream.write(
                    'python3() { printf "%s\\n" "$*" >> "$AMCT_SMOKE_CALL_LOG"; '
                    'printf "%s\\n" "$AMCT_SMOKE_FAKE_MODEL_PATH"; }\n'
                )
            env.update(
                {
                    "BASH_ENV": startup.as_posix(),
                    "AMCT_SMOKE_CALL_LOG": calls.as_posix(),
                    "AMCT_SMOKE_FAKE_MODEL_PATH": (root / "model").as_posix(),
                }
            )
            result = subprocess.run(
                [
                    BASH,
                    script.as_posix(),
                    "--model_dir",
                    (root / "cache").as_posix(),
                    "--device",
                    "cpu",
                    "--cases",
                    cases,
                ],
                env=env,
                capture_output=True,
                encoding="utf-8",
                timeout=10,
                check=False,
            )
            return result, calls.read_text(encoding="utf-8") if calls.exists() else ""

    def assert_empty_selection_rejected(self, cases):
        result, calls = self.run_wrapper(cases)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertEqual(calls, "")

    def test_empty_string_rejected_before_python(self):
        self.assert_empty_selection_rejected("")

    def test_whitespace_rejected_before_python(self):
        self.assert_empty_selection_rejected("   ")

    def test_separators_rejected_before_python(self):
        self.assert_empty_selection_rejected(", ,")

    def test_valid_selection_is_forwarded(self):
        result, calls = self.run_wrapper("minmax")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("--cases minmax", calls)


if __name__ == "__main__":
    unittest.main()
