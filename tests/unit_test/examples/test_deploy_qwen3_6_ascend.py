# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

import os
import shutil
import subprocess
import sys
from pathlib import Path


def test_export_shell_forwards_qwen36_ascend_tensorwise_options(tmp_path):
    capture = tmp_path / "arguments.json"
    launcher = tmp_path / "python launcher.sh"
    launcher.write_text(
        '#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "${CAPTURE_ARGS}"\n',
        newline="\n",
    )
    launcher.chmod(0o755)
    repo_dir = Path(__file__).resolve().parents[3]
    script = repo_dir / "examples/deploy_qwen3_6_ascend.sh"
    assert script.is_file()
    env = dict(
        os.environ,
        PYTHON_BIN=launcher.as_posix(),
        CAPTURE_ARGS=capture.as_posix(),
        QWEN36_BF16_DIR=str(tmp_path / "BF16 model"),
        QWEN36_ASCEND_DIR=str(tmp_path / "INT8 model"),
    )

    if sys.platform == "win32":
        git_path = shutil.which("git")
        assert git_path is not None
        git = Path(git_path)
        bash = git.parent.parent / "bin/bash.exe"
    else:
        bash_path = shutil.which("bash")
        assert bash_path is not None
        bash = Path(bash_path)
    assert bash.is_file()
    subprocess.run([bash, script], cwd=tmp_path, env=env, check=True)

    args = capture.read_text().splitlines()
    assert args[args.index("--model") + 1] == env["QWEN36_BF16_DIR"]
    assert args[args.index("--output_dir") + 1] == env["QWEN36_ASCEND_DIR"]
    assert args[args.index("--model_name") + 1] == "qwen3_6_moe"
    assert args[args.index("--device") + 1] == "cpu"
    assert args[args.index("--granularity") + 1] == "tensor"
    assert args[args.index("--quant_dtype") + 1] == "int"
    assert args[args.index("--deploy_format") + 1] == "ascend"
    bit_config = args[args.index("--bit_config") + 1].replace("\\", "/")
    quant_layers_config = args[args.index("--quant_layers_config") + 1].replace(
        "\\", "/"
    )
    assert bit_config.endswith("/amct_pytorch/configs/w8a8.yaml")
    assert quant_layers_config.endswith(
        "/amct_pytorch/configs/deploy/qwen3_6_ascend_w8a8.json"
    )
    assert (repo_dir / "amct_pytorch/configs/w8a8.yaml").is_file()
    assert (
        repo_dir / "amct_pytorch/configs/deploy/qwen3_6_ascend_w8a8.json"
    ).is_file()
