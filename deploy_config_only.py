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

"""Config-only deploy script.

Subclasses ``LlmDeployWorkflow`` and overrides ``_run_blockwise`` to skip all
weight I/O, reusing ``setup`` / ``_copy_support_files`` / ``_refresh_config``
from the parent.  Blocks are built with ``init_empty_weights`` so no
safetensors files are needed — only config.json + tokenizer files are
required in --model.

Usage (same args as the real deploy CLI)::

    python deploy_config_only.py \
        --model /path/to/GLM-5.2 \
        --model_name glm5_2 \
        --seq_len 4096 \
        --granularity block \
        --device npu:0 \
        --quant_dtype int \
        --quant_target moe \
        --bit_config amct_pytorch/configs/w8a8.yaml \
        --output_dir /path/to/output
"""

import gc
import types

from accelerate import init_empty_weights
from tqdm import tqdm

from amct_pytorch.cli.llm.args import parser_gen
from amct_pytorch.common.models.llm.common.deploy_export import (
    get_quant_ignore_linear_names,
)
from amct_pytorch.workflows.llm_deploy import LlmDeployWorkflow


class ConfigOnlyDeployWorkflow(LlmDeployWorkflow):
    """Deploy workflow that only generates config.json, skipping all weight I/O.

    Reuses from parent:
      - ``__init__``      — arg parsing, field setup
      - ``run``           — dispatch to ``_run_blockwise``
      - ``setup``         — registration, pipeline construction, logging
      - ``_copy_support_files`` — copy config.json + tokenizer to output_dir
      - ``_refresh_config``     — write quantization_config into config.json

    Overrides:
      - ``setup``          — adds two patches before/after super().setup()
      - ``_run_blockwise`` — collect ignore names from block structure only
    """

    def setup(self):
        # Patch 1: get_weight_mappings must be stubbed BEFORE pipeline
        # construction, because GLM5_2.__init__ calls it to read the
        # safetensors index — which doesn't exist in a config-only model dir.
        import amct_pytorch.common.models.llm.glm.glm5_2 as _glm5_2_mod
        setattr(_glm5_2_mod, "get_weight_mappings", lambda _path: {})

        sink_id = super().setup()

        # Patch 2: pipeline.block normally loads real weights from safetensors.
        # Replace with init_empty_weights so only module *structure* is built
        # (enough for isinstance checks in get_quant_ignore_linear_names).
        def _block_empty(pipeline_self, layer_idx):
            with init_empty_weights():
                return pipeline_self.cls(pipeline_self.config, layer_idx)

        self.pipeline.block = types.MethodType(_block_empty, self.pipeline)
        return sink_id

    def _run_blockwise(self):
        """Collect ignore names from block traversal, then write config.json.

        Same loop structure as ``LlmDeployWorkflow._run_blockwise``, but calls
        ``get_quant_ignore_linear_names`` directly instead of
        ``export_block_deploy`` (which also exports weights).  Skips
        ``_write_block_file``, ``_write_remaining_original_weights``, and
        ``_refresh_weight_index`` entirely.
        """
        self._copy_support_files()
        quant_ignore_layers = []
        for layer_idx in tqdm(range(self.pipeline.num_layers), desc="Block Processing..."):
            block = self.pipeline.build_quant_block(layer_idx)
            weight_prefix = self.pipeline.get_layer_weight_prefix(layer_idx)
            quant_ignore_layers.extend(
                get_quant_ignore_linear_names(block, weight_prefix)
            )
            del block
            gc.collect()

        # Deduplicate preserving order (export_block_deploy would otherwise
        # append duplicates per layer without dedup).
        seen = set()
        quant_ignore_layers = [
            x for x in quant_ignore_layers if not (x in seen or seen.add(x))
        ]
        self._refresh_config(quant_ignore_layers)
        return {"ignore_count": len(quant_ignore_layers)}


def main():
    args = parser_gen(command="deploy")
    workflow = ConfigOnlyDeployWorkflow(args)
    workflow.run()


if __name__ == "__main__":
    main()
