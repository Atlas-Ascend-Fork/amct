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

from __future__ import annotations

import os
import pickle
import re
from pathlib import Path

import torch
from loguru import logger

from amct_pytorch.common.utils.safe_load import safe_torch_load


class EvalCheckpointManager:
    """Keep the latest block output, selected by model name and dtype."""

    def __init__(self, args, num_layers: int):
        self.num_layers = num_layers
        self.model_name = args.model_name
        self.quant_dtype = (
            "bf16" if args.eval_mode == "bf16" else (args.quant_dtype or "none")
        )
        self.run_dir = Path(args.output_dir) / "temp" / "eval"
        self.prefix = f"{self.model_name}_{self.quant_dtype}_"
        self.pattern = re.compile(re.escape(self.prefix) + r"(\d+)\.pt")

    def _checkpoints(self):
        if not self.run_dir.exists():
            return []
        return [
            (int(match.group(1)), path)
            for path in self.run_dir.iterdir()
            if path.is_file() and (match := self.pattern.fullmatch(path.name))
        ]

    def save(self, completed_block_idx: int, inter_io, model_state=None):
        if not 0 <= completed_block_idx < self.num_layers:
            raise ValueError(
                f"completed block index {completed_block_idx} is outside "
                f"[0, {self.num_layers})."
            )
        self.run_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = self.run_dir / f"{self.prefix}{completed_block_idx}.pt"
        temporary_path = checkpoint_path.with_suffix(".pt.tmp")
        with temporary_path.open("wb") as file_obj:
            torch.save(
                {
                    "inter_io": inter_io,
                    "model_state": model_state,
                },
                file_obj,
            )
            file_obj.flush()
            os.fsync(file_obj.fileno())
        os.replace(temporary_path, checkpoint_path)
        for _, path in self._checkpoints():
            if path != checkpoint_path:
                path.unlink()

    def load(self):
        self.run_dir.mkdir(parents=True, exist_ok=True)
        for path in self.run_dir.iterdir():
            if (
                path.is_file()
                and path.name.endswith(".tmp")
                and self.pattern.fullmatch(path.name.removesuffix(".tmp"))
            ):
                path.unlink()
        for block_idx, path in sorted(self._checkpoints(), reverse=True):
            try:
                payload = safe_torch_load(path, map_location="cpu")
            except (EOFError, pickle.UnpicklingError, RuntimeError):
                payload = None
            if isinstance(payload, dict) and 0 <= block_idx < self.num_layers:
                inter_io = payload.get("inter_io")
                if (
                    isinstance(inter_io, (list, tuple))
                    and inter_io
                    and all(isinstance(item, torch.Tensor) for item in inter_io)
                ):
                    return block_idx, inter_io, payload.get("model_state")
            logger.warning("Discard invalid eval checkpoint '{}'.", path)
            path.unlink()
        return None

    def cleanup(self):
        for _, path in self._checkpoints():
            path.unlink()
