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

import inspect
import json
import os
from typing import Any
from tqdm import tqdm
import torch
from loguru import logger

from amct_pytorch.common.datasets.preproc import get_wiki_inputs
from amct_pytorch.common.evaluate.eval_ppl import wikitext2_ppl
from amct_pytorch.common.models.llm import register_llm_models
from amct_pytorch.common.optimization import register_solvers
from amct_pytorch.quantization.dtypes import register_dtype
from amct_pytorch.algorithms.quant import register_algorithms
from amct_pytorch.common.models import MODEL_REGISTRY
from amct_pytorch.common.optimization import SOLVER_REGISTRY
from amct_pytorch.common.utils.run_logging import ensure_log_dir, setup_run_logging
from amct_pytorch.quantization.bit_policy import ensure_bit_policy


class LlmEvalWorkflow:
    """Orchestrate LLM Eval without hard-coding solver granularity."""

    def __init__(self, args):
        self.args = args
        self.seq_len = args.seq_len
        self.device = args.device
        self.granularity = args.granularity
        self.eval_mode = args.eval_mode
        self.pipeline = None
        self.data_provider = None
        self.model_name = args.model_name
        self.quant_target = args.quant_target
        self.quant_dtype = args.quant_dtype
        self.bit_policy = ensure_bit_policy(args)

    @staticmethod
    def _register_components():
        register_llm_models()
        register_dtype()
        register_algorithms()

    def setup(self):
        self._register_components()
        self.pipeline = self._build_pipeline()
        if hasattr(self.pipeline, "sharded_block"):
            self.pipeline.sharded_block = True
        sink_id, _ = setup_run_logging(
            self.args,
            f"eval_{self.model_name}_{self.eval_mode}",
        )
        return sink_id

    def run(self):
        sink_id = self.setup()
        if self.granularity == "block":
            ppl = self._run_blockwise()
        elif self.granularity == "model":
            ppl = self._run_modelwise()
        else:
            raise ValueError(
                f"Unsupported granularity '{self.granularity}' for eval."
            )
        logger.info(
            f"Model: {self.model_name}; Quant target: {self.quant_target}; Quant dtype: {self.quant_dtype}; "
            f"PPL: {ppl}\n{self.bit_policy.summary()}"
        )
        logger.remove(sink_id)

    def _build_pipeline(self):
        model_cls = MODEL_REGISTRY.get(self.model_name)
        return model_cls(self.args)

    def _has_relevant_quant(self):
        if not hasattr(self, "bit_policy"):
            self.bit_policy = ensure_bit_policy(self.args)
        quant_targets = set(self.args.quant_target)
        if quant_targets.intersection({"mlp", "moe", "attn-linear"}):
            if self.bit_policy.has_quant_linear():
                return True
        if "attn-cache" in quant_targets:
            if self.bit_policy.has_quant_cache():
                return True
        return False

    def _get_relevant_quant_bits(self):
        if not hasattr(self, "bit_policy"):
            self.bit_policy = ensure_bit_policy(self.args)
        quant_targets = set(self.args.quant_target)
        bits = []
        if quant_targets.intersection({"mlp", "moe", "attn-linear"}):
            bits.extend([self.bit_policy.w_bits, self.bit_policy.a_bits])
        if "attn-cache" in quant_targets:
            bits.extend(self.bit_policy.cache_bits(key) for key in ("q", "k", "p", "v"))
        return bits

    def _resolve_eval_states(self):
        if not hasattr(self, "bit_policy"):
            self.bit_policy = ensure_bit_policy(self.args)
        if self.eval_mode == "bf16":
            return False, False, "Eval mode=bf16: use original BF16 blocks."

        if self._has_relevant_quant():
            return True, True, "Eval mode=quant: run quant modules with quantization enabled."
        return True, False, (
            "Eval mode=quant with no <16-bit entries in policy: "
            "run rebuilt quant modules with quantization disabled."
        )

    def _run_blockwise(self,):
        tokenizer = self.pipeline.tokenizer
        samples = get_wiki_inputs(tokenizer, self.seq_len)
        logger.info("Loaded {} eval samples for blockwise eval.", len(samples))
        inter_io = self.pipeline.do_embedding_forward(samples, self.args.output_dir)
        use_quant_block, enable_quant, eval_message = self._resolve_eval_states()
        logger.info(eval_message)
        for layer_idx in tqdm(range(self.pipeline.num_layers), desc="Block Processing..."):
            inter_io = self.pipeline.do_block_forward(
                layer_idx,
                inter_io,
                use_quant_block=use_quant_block,
                enable_quant=enable_quant,
            )
            if layer_idx == self.pipeline.num_layers - 1:
                preds = self.pipeline.do_head_forward(inter_io)
        ppl = wikitext2_ppl(preds, samples, seq_len=self.seq_len)
        return ppl


    def _run_modelwise(self, ):
        model = self.pipeline.float_model().eval().to(self.device)
        tokenizer = self.pipeline.tokenizer
        samples = get_wiki_inputs(tokenizer, self.seq_len)
        logger.info("Loaded {} eval samples for modelwise eval.", len(samples))
        preds = []
        with torch.no_grad():
            for i, sample in tqdm(enumerate(samples), desc="Evaluating"):
                lm_logits = model(sample.to(self.device)).logits
                shift_logits = lm_logits[:, :-1, :].contiguous()
                preds.append(shift_logits.to("cpu"))
        ppl = wikitext2_ppl(preds, samples, seq_len=self.seq_len)
        return ppl

    def _save_inter_result(self, result, name):
        save_path = os.path.join(self.args.output_dir, f"{name}.pkl")
        torch.save(result, save_path)
