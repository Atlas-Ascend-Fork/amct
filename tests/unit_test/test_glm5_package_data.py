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
"""Check the real source distribution without importing model dependencies."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_sdist_contains_glm5_runtime_configuration(tmp_path):
    """A successful package build must retain GLM5's required local config."""
    source = tmp_path / "source"
    source.mkdir()
    shutil.copytree(
        REPO_ROOT / "amct_pytorch",
        source / "amct_pytorch",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info"),
    )
    for name in ("setup.py", "pyproject.toml", "README.md", "LICENSE"):
        shutil.copyfile(REPO_ROOT / name, source / name)

    config_relative = "amct_pytorch/common/models/llm/glm/glm5/config.json"
    expected_config = (source / config_relative).read_bytes()
    env = os.environ.copy()
    # Match the metadata supplied by the normal CMake packaging target.
    env["AMCT_PYTORCH_PLATFORM"] = "linux_x86_64"
    env["AMCT_EXPERIMENTAL"] = "FALSE"
    result = subprocess.run(
        [sys.executable, "-B", "setup.py", "sdist", "--formats=gztar"],
        cwd=source,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    archives = list((source / "dist").glob("amct_pytorch-*.tar.gz"))
    assert len(archives) == 1
    module_relative = "amct_pytorch/common/models/llm/glm/glm5/glm5.py"
    with tarfile.open(archives[0], "r:gz") as archive:
        members = set(archive.getnames())
        modules = [name for name in members if name.endswith("/" + module_relative)]
        assert len(modules) == 1, "The real sdist must contain the GLM5 adapter"
        config_member = modules[0].rsplit("/", 1)[0] + "/config.json"
        assert config_member in members, (
            "GLM5 loads config.json next to glm5.py, but the real sdist omits it"
        )
        payload = archive.extractfile(config_member)
        assert payload is not None
        assert payload.read() == expected_config
