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
"""剪枝运行时行为：缺失/空校准数据策略、运行时环境采集、tokenizer 落盘工具。

针对 HuggingFace / ModelScope 适配器与 BoolQ 评估的能力按 AMCT 流水线约定放置在
examples/algorithms/prune/src/utils.py。
"""

from __future__ import annotations

import unittest
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory

from mock_torch_npu import *  # noqa: F401,F403

import torch
import torch.nn as nn

from amct_pytorch.pruning import (
    AutoPruner,
    MethodSpec,
    PruneConfig,
    collect_runtime_environment,
    save_tokenizer,
)
from amct_pytorch.pruning.toy_models import TinyTransformerModel
from amct_pytorch.pruning.utils import count_parameters


class FakeSavableTokenizer:
    @classmethod
    def from_pretrained(cls, *_args, **_kwargs):
        return cls()

    def save_pretrained(self, save_dir: str) -> None:
        path = Path(save_dir) / "tokenizer.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")


class FakeUnsavableTokenizer:
    pass


class RuntimeBehaviorTest(unittest.TestCase):
    def test_missing_prune_data_warns_and_skips_dense_stage(self) -> None:
        model = TinyTransformerModel(hidden_size=32, intermediate_size=64, vocab_size=50)
        before = count_parameters(model)
        pruner = AutoPruner(
            PruneConfig(
                methods={
                    "cnn": MethodSpec("tp_variance_channel", {"prune_ratio": 0.0}),
                    "dense": MethodSpec("low_variance", {"prune_ratio": 0.50}),
                    "moe": MethodSpec("activation_count", {"prune_ratio": 0.0}),
                },
                missing_data_policy="warn_skip",
            )
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pruned = pruner(model)
        self.assertEqual(before, count_parameters(pruned))
        self.assertTrue(any("Skipping pruning method 'low_variance'" in str(item.message) for item in caught))
        self.assertIsNotNone(pruner.last_report)
        assert pruner.last_report is not None
        self.assertTrue(any("Skipping pruning method 'low_variance'" in message for message in pruner.last_report.warnings))

    def test_empty_prune_data_warns_and_skips_dense_stage(self) -> None:
        model = TinyTransformerModel(hidden_size=32, intermediate_size=64, vocab_size=50)
        before = count_parameters(model)
        pruner = AutoPruner(
            PruneConfig(
                methods={
                    "cnn": MethodSpec("tp_variance_channel", {"prune_ratio": 0.0}),
                    "dense": MethodSpec("low_variance", {"prune_ratio": 0.50}),
                    "moe": MethodSpec("activation_count", {"prune_ratio": 0.0}),
                }
            )
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pruned = pruner(model, data=[])
        self.assertEqual(before, count_parameters(pruned))
        self.assertTrue(any("empty pruning dataset" in str(item.message) for item in caught))

    def test_collect_runtime_environment(self) -> None:
        report = collect_runtime_environment()
        self.assertIn("python_version", report)
        self.assertIn("packages", report)
        self.assertIn("torch", report["packages"])

    def test_save_tokenizer_success(self) -> None:
        with TemporaryDirectory() as tmpdir:
            saved, message = save_tokenizer(FakeSavableTokenizer(), tmpdir)
            self.assertTrue(saved)
            self.assertIsNone(message)
            self.assertTrue((Path(tmpdir) / "tokenizer.json").exists())

    def test_save_tokenizer_missing_interface_warns(self) -> None:
        with TemporaryDirectory() as tmpdir:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                saved, message = save_tokenizer(FakeUnsavableTokenizer(), tmpdir)
            self.assertFalse(saved)
            self.assertIsNotNone(message)
            self.assertTrue(any("save_pretrained" in str(item.message) for item in caught))


if __name__ == "__main__":
    unittest.main()
