# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

from __future__ import annotations

from tqdm import tqdm
from loguru import logger

from amct_pytorch.common.datasets.preproc import get_pileval
from amct_pytorch.common.models.llm import register_llm_models
from amct_pytorch.common.models import MODEL_REGISTRY
from amct_pytorch.common.utils.run_logging import ensure_log_dir, setup_run_logging


class LlmExtractPtqDataWorkflow:

    def __init__(self, args):
        self.args = args
        self.seq_len = args.seq_len
        self.nsamples = args.nsamples
        self.granularity = args.granularity
        if len(args.quant_target) != 1:
            raise ValueError("extract_ptq_data only supports a single quant_target.")
        self.quant_target = args.quant_target[0]
        self.device = args.device
        self.pipeline = None
        self.data_provider = None
        self.model_name = self.args.model_name

    @staticmethod
    def _register_components():
        register_llm_models()

    def setup(self):
        ensure_log_dir(self.args)
        self._register_components()
        self.pipeline = self._build_pipeline()
        sink_id, _ = setup_run_logging(self.args, "extract_ptq_data")
        return sink_id

    def run(self):
        sink_id = self.setup()
        if self.granularity == "block":
            results = self._run_blockwise()
        else:
            raise ValueError(
                f"Currently unsupported granularity '{self.granularity}' for extract ptq data."
            )
        logger.remove(sink_id)
        return results

    def _build_pipeline(self):
        model_cls = MODEL_REGISTRY.get(self.model_name)
        return model_cls(self.args)

    def _run_blockwise(self):
        tokenizer = self.pipeline.tokenizer
        attn_hook_name = getattr(self.pipeline, "attn_norm_name", "input_layernorm")
        ffn_hook_name = getattr(self.pipeline, "ffn_norm_name", "post_attention_layernorm")
        hook_name = (
            attn_hook_name
            if self.quant_target in ("attn", "attn-linear", "attn-cache")
            else ffn_hook_name
        )
        samples = get_pileval(tokenizer, self.nsamples, seq_len=self.seq_len)
        logger.info(
            "Loaded {} calibration samples for extract_ptq_data.",
            len(samples),
        )
        inter_io = self.pipeline.do_embedding_forward(samples, hook_name=hook_name)
        for layer_idx in tqdm(range(self.pipeline.num_layers), desc="Block Processing..."):
            inter_io = self.pipeline.do_block_forward(layer_idx, inter_io, hook_name=hook_name)
