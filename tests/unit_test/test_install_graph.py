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
import subprocess
import tarfile
import os
from pathlib import Path
from platform import machine


REPO_ROOT = Path(__file__).resolve().parents[2]
INSTALL_GRAPH_CMAKE = REPO_ROOT / "cmake" / "install_graph.cmake"


def _make_tarball(path: Path, marker_text: str) -> None:
    payload = path.parent / f"{path.stem}_payload"
    payload.mkdir(parents=True)
    marker = payload / "marker.txt"
    marker.write_text(marker_text, encoding="utf-8")
    with tarfile.open(path, "w:gz") as tar:
        tar.add(marker, arcname="marker.txt")


def _run_install_graph(
    work_dir: Path,
    *extra_args: str,
    env: dict[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess:
    command = [
        "cmake",
        f"-DAMCT_OPEN_DIR={work_dir}",
        "-DAMCT_GRAPH_ARCH=x86_64",
        *extra_args,
        "-P",
        str(INSTALL_GRAPH_CMAKE),
    ]
    return subprocess.run(command, check=check, text=True, capture_output=True, env=env)


def _env_without_ascend_home() -> dict[str, str]:
    env = os.environ.copy()
    env.pop("ASCEND_HOME_PATH", None)
    return env


def test_cmake_install_graph_uses_cann_version_for_newest_weekly_package(tmp_path):
    ascend_home = tmp_path / "Ascend" / "cann-10.4.0"
    ascend_home.mkdir(parents=True)
    obs_dir = tmp_path / "obs" / "CANN" / "20260528_newest"
    obs_dir.mkdir(parents=True)
    tarball_name = "cann-amct-graph_10.4.0_linux-x86_64.tar.gz"
    _make_tarball(obs_dir / tarball_name, "remote-newest")

    work_dir = tmp_path / "work"
    work_dir.mkdir()
    _run_install_graph(
        work_dir,
        f"-DAMCT_GRAPH_BASE_URL=file://{tmp_path / 'obs' / 'CANN'}",
        f"-DASCEND_HOME_PATH={ascend_home}",
        "-DAMCT_GRAPH_TODAY=20260601",
        "-DAMCT_GRAPH_LOOKBACK_DAYS=10",
        env=_env_without_ascend_home(),
    )

    assert (work_dir / "amctgraph" / "marker.txt").read_text(encoding="utf-8") == "remote-newest"
    assert not (work_dir / "amctgraph" / tarball_name).exists()


def test_cmake_install_graph_prefers_local_build_tarball(tmp_path):
    work_dir = tmp_path / "work"
    local_tar_dir = work_dir / "build"
    local_tar_dir.mkdir(parents=True)
    tarball_name = "cann-amct-graph_10.4.0_linux-x86_64.tar.gz"
    _make_tarball(local_tar_dir / tarball_name, "local-build")

    obs_dir = tmp_path / "obs" / "CANN" / "20260528_newest"
    obs_dir.mkdir(parents=True)
    _make_tarball(obs_dir / tarball_name, "remote-newest")

    _run_install_graph(
        work_dir,
        f"-DAMCT_GRAPH_BASE_URL=file://{tmp_path / 'obs' / 'CANN'}",
        "-DAMCT_GRAPH_VERSION=10.4.0",
        "-DAMCT_GRAPH_TODAY=20260601",
        "-DAMCT_GRAPH_LOOKBACK_DAYS=10",
        env=_env_without_ascend_home(),
    )

    assert (work_dir / "amctgraph" / "marker.txt").read_text(encoding="utf-8") == "local-build"


def test_cmake_install_graph_defaults_arch_from_uname_in_script_mode(tmp_path):
    work_dir = tmp_path / "work"
    local_tar_dir = work_dir / "build"
    local_tar_dir.mkdir(parents=True)
    tarball_name = f"cann-amct-graph_10.4.0_linux-{machine()}.tar.gz"
    _make_tarball(local_tar_dir / tarball_name, "default-arch")

    subprocess.run(
        [
            "cmake",
            f"-DAMCT_OPEN_DIR={work_dir}",
            "-DAMCT_GRAPH_VERSION=10.4.0",
            "-P",
            str(INSTALL_GRAPH_CMAKE),
        ],
        check=True,
        text=True,
        capture_output=True,
        env=_env_without_ascend_home(),
    )

    assert (work_dir / "amctgraph" / "marker.txt").read_text(encoding="utf-8") == "default-arch"


def test_cmake_install_graph_falls_back_to_version_range_without_version_source(tmp_path):
    obs_dir = tmp_path / "obs" / "CANN" / "20260528_newest"
    obs_dir.mkdir(parents=True)
    tarball_name = "cann-amct-graph_10.4.0_linux-x86_64.tar.gz"
    _make_tarball(obs_dir / tarball_name, "fallback-version")

    work_dir = tmp_path / "work"
    work_dir.mkdir()

    _run_install_graph(
        work_dir,
        f"-DAMCT_GRAPH_BASE_URL=file://{tmp_path / 'obs' / 'CANN'}",
        "-DASCEND_HOME_PATH=",
        "-DAMCT_GRAPH_TODAY=20260601",
        "-DAMCT_GRAPH_LOOKBACK_DAYS=10",
        "-DAMCT_GRAPH_FALLBACK_MIN_VERSION=10.3.0",
        "-DAMCT_GRAPH_FALLBACK_MAX_VERSION=10.5.0",
        env=_env_without_ascend_home(),
    )

    assert (work_dir / "amctgraph" / "marker.txt").read_text(encoding="utf-8") == "fallback-version"


def test_cmake_install_graph_reads_version_from_cann_install_info(tmp_path):
    ascend_home = tmp_path / "Ascend" / "latest"
    install_info_dir = ascend_home / "x86_64"
    install_info_dir.mkdir(parents=True)
    (install_info_dir / "ascend_toolkit_install.info").write_text(
        "package_name=Ascend-cann-toolkit\nversion=10.5.0\n",
        encoding="utf-8",
    )

    obs_dir = tmp_path / "obs" / "CANN" / "20260528_newest"
    obs_dir.mkdir(parents=True)
    tarball_name = "cann-amct-graph_10.5.0_linux-x86_64.tar.gz"
    _make_tarball(obs_dir / tarball_name, "install-info-version")

    work_dir = tmp_path / "work"
    work_dir.mkdir()
    _run_install_graph(
        work_dir,
        f"-DAMCT_GRAPH_BASE_URL=file://{tmp_path / 'obs' / 'CANN'}",
        f"-DASCEND_HOME_PATH={ascend_home}",
        "-DAMCT_GRAPH_TODAY=20260601",
        "-DAMCT_GRAPH_LOOKBACK_DAYS=10",
        env=_env_without_ascend_home(),
    )

    assert (work_dir / "amctgraph" / "marker.txt").read_text(encoding="utf-8") == "install-info-version"
