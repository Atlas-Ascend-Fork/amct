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

"""DeepSeek V4.1 tensor deployment; no model construction or PTQ support."""

import json
from pathlib import Path

from amct_pytorch.common.models import MODEL_REGISTRY
from amct_pytorch.common.models.llm.common.weight_path_validation import (
    collect_safetensors_files,
)
from amct_pytorch.common.models.llm.deepseek.deepseek_deploy_common import (
    DeepSeekTensorwiseDeployCommon,
)


@MODEL_REGISTRY.register(
    name="deepseek_v4_1",
    task="llm",
    family="deepseek",
    description="DeepSeek V4.1 tensor deployment only",
)
class DeepseekV41(DeepSeekTensorwiseDeployCommon):
    """Lightweight registry entry, deliberately independent of BaseModel/PTQ."""

    def __init__(self, args):
        self.validate_deploy_args(args)
        self.args = args
        self.quant_dtype = args.quant_dtype
        self.model_path = Path(args.model)
        self.output_dir = Path(args.output_dir)
        source, output = self.model_path.resolve(), self.output_dir.resolve()
        if source == output or source in output.parents or output in source.parents:
            raise ValueError(
                "DeepSeek V4.1 output and source directories must not overlap"
            )
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            raise ValueError("DeepSeek V4.1 output directory must be new or empty")
        with open(self.model_path / "config.json", encoding="utf-8") as stream:
            self.config = json.load(stream)
        self.safetensors_files = collect_safetensors_files(self.model_path)
        self.processed_modules = set()
