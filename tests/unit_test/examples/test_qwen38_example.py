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

import argparse
from pathlib import Path
import re
import shlex
from types import SimpleNamespace

from amct_pytorch.cli.llm.args import parser_gen
from amct_pytorch.quantization.bit_policy import BitPolicy


REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_DIR = REPO_ROOT / "examples" / "models" / "qwen3.8"
ZH_GUIDE = EXAMPLE_DIR / "Qwen3.8-27B.md"
EN_GUIDE = EXAMPLE_DIR / "Qwen3.8-27B_en.md"
BIT_CONFIG = EXAMPLE_DIR / "qwen3_8_27b_mixed_w4a8.yaml"


def _module_commands(document):
    commands = []
    for block in re.findall(r"```bash\n(.*?)```", document, flags=re.DOTALL):
        normalized = block.replace("\\\n", " ")
        for command in re.findall(
            r"python -m amct_pytorch\.(?:eval|extract_ptq_data|ptq|deploy).*?(?=\n\n|$)",
            normalized,
            flags=re.DOTALL,
        ):
            commands.append(shlex.split(command))
    return commands


def _parser_options(monkeypatch):
    captured = {}

    def capture_parse_args(parser):
        captured["parser"] = parser
        return SimpleNamespace(bit_config=None)

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", capture_parse_args)
    parser_gen()
    return {
        option
        for action in captured["parser"]._actions
        for option in action.option_strings
    }


def test_qwen38_mixed_policy_has_documented_module_counts():
    policy = BitPolicy.from_yaml(str(BIT_CONFIG))
    role_counts = {
        "mlp": {"gate_proj": 64, "up_proj": 64, "down_proj": 64},
        "attn-linear": {
            "in_proj_qkv": 48,
            "in_proj_z": 48,
            "in_proj_b": 48,
            "in_proj_a": 48,
            "out_proj": 48,
            "q_proj": 16,
            "k_proj": 16,
            "v_proj": 16,
            "o_proj": 16,
        },
    }

    counts = {4: 0, 8: 0}
    for group, roles in role_counts.items():
        for role, count in roles.items():
            w_bits, a_bits = policy.linear_bits(name=role, group=group)
            assert a_bits == 8
            counts[w_bits] += count

    assert counts == {4: 400, 8: 96}


def test_qwen38_guides_keep_commands_in_sync_and_use_real_cli_options(monkeypatch):
    zh_commands = _module_commands(ZH_GUIDE.read_text(encoding="utf-8"))
    en_commands = _module_commands(EN_GUIDE.read_text(encoding="utf-8"))
    assert zh_commands == en_commands
    assert [command[2] for command in zh_commands] == [
        "amct_pytorch.eval",
        "amct_pytorch.extract_ptq_data",
        "amct_pytorch.extract_ptq_data",
        "amct_pytorch.ptq",
        "amct_pytorch.ptq",
        "amct_pytorch.eval",
        "amct_pytorch.deploy",
    ]

    known_options = _parser_options(monkeypatch)
    for command in zh_commands:
        used_options = {token for token in command if token.startswith("--")}
        assert used_options <= known_options


def test_qwen38_readme_links_and_guides_reference_the_policy():
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    readme_en = (REPO_ROOT / "README_en.md").read_text(encoding="utf-8")
    assert "examples/models/qwen3.8/Qwen3.8-27B.md" in readme
    assert "examples/models/qwen3.8/Qwen3.8-27B_en.md" in readme_en
    for guide in (ZH_GUIDE, EN_GUIDE):
        text = guide.read_text(encoding="utf-8")
        assert BIT_CONFIG.name in text
        assert "--algos gptq_diag" in text
        assert re.search(r"(?m)(?:^|[ =`])/(?:mnt|home)/", text) is None
