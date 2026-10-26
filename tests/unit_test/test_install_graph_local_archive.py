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
"""Run the real graph installer against local archives, without downloads."""

import io
import os
from pathlib import Path
import shutil
import subprocess
import tarfile

import pytest


BASH = shutil.which("bash")
if os.name == "nt" and shutil.which("git"):
    git_bash = Path(shutil.which("git")).resolve().parents[1] / "bin/bash.exe"
    if git_bash.is_file():
        BASH = str(git_bash)
pytestmark = pytest.mark.skipif(BASH is None, reason="Bash is required")
ROOT = Path(__file__).resolve().parents[2]


def prepare_installer(tmp_path, directory="repo"):
    root = (tmp_path / directory).resolve()
    assert root.is_relative_to(tmp_path.resolve())
    root.mkdir()
    shutil.copyfile(ROOT / "install_graph.sh", root / "install_graph.sh")
    architecture = subprocess.check_output([BASH, "-c", "uname -m"], text=True).strip()
    return root, architecture


def make_archive(root, relative, architecture, version="9.2.0"):
    folder = root / "build" / relative
    folder.mkdir(parents=True, exist_ok=True)
    archive = folder / f"cann-amct-graph_{version}_linux-{architecture}.tar.gz"
    content = b"local graph package\n"
    with tarfile.open(archive, "w:gz") as bundle:
        member = tarfile.TarInfo("local-marker.txt")
        member.size = len(content)
        bundle.addfile(member, io.BytesIO(content))
    return archive, content


def run_installer(root, path_prefix=None):
    env = os.environ.copy()
    env.pop("BASH_ENV", None)
    env.pop("SHELLOPTS", None)
    env["PATH"] = str(Path(BASH).parent) + os.pathsep + env.get("PATH", "")
    if path_prefix is not None:
        env["PATH"] = str(path_prefix) + os.pathsep + env["PATH"]
    return subprocess.run(
        [BASH, "install_graph.sh"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
        check=False,
    )


@pytest.mark.parametrize(
    "directory,relative",
    [
        ("repo", "."),
        ("repo", "nested/output"),
        ("repo", "nested output"),
        ("repo with spaces", "."),
    ],
)
def test_reuses_the_discovered_local_archive(tmp_path, directory, relative):
    root, architecture = prepare_installer(tmp_path, directory)
    archive, expected = make_archive(root, relative, architecture)
    result = run_installer(root)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Downloading" not in result.stdout
    assert (root / "amctgraph/local-marker.txt").read_bytes() == expected
    assert archive.is_file()


def test_ambiguous_local_archives_fail_without_downloading(tmp_path):
    root, architecture = prepare_installer(tmp_path)
    make_archive(root, "first", architecture)
    make_archive(root, "second", architecture, "9.1.0")
    result = run_installer(root)
    assert result.returncode != 0
    assert "Multiple local graph packages" in result.stdout
    assert "Downloading" not in result.stdout


def test_reuses_archive_from_symlinked_build_directory(tmp_path):
    root, architecture = prepare_installer(tmp_path)
    archive, expected = make_archive(root, "nested output", architecture)
    target = root / "local build"
    (root / "build").rename(target)
    env = os.environ.copy()
    if os.name == "nt":
        # Git Bash supports its own symlinks without Windows symlink privileges.
        env["MSYS"] = "winsymlinks"
    subprocess.run(
        [
            BASH,
            "-c",
            'ln -s -- "$1" build && test -L build',
            "create-build-link",
            target.name,
        ],
        cwd=root,
        env=env,
        capture_output=True,
        check=True,
        timeout=10,
    )
    guard = root / "no-download"
    guard.mkdir()
    wget = guard / "wget"
    wget.write_text("#!/bin/sh\nexit 99\n", encoding="utf-8")
    wget.chmod(0o755)
    result = run_installer(root, path_prefix=guard)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Downloading" not in result.stdout
    assert (root / "amctgraph/local-marker.txt").read_bytes() == expected
    assert (target / archive.relative_to(root / "build")).is_file()
