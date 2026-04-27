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
# from amct_pytorch.common.models.llm.common import PtqUnit
from amct_pytorch.common.models.llm import register_llm_models
from amct_pytorch.common.optimization import register_solvers
from amct_pytorch.quantization.dtypes import register_dtype
from amct_pytorch.algorithms.quant import register_algorithms
from amct_pytorch.common.models import MODEL_REGISTRY
from amct_pytorch.common.optimization import SOLVER_REGISTRY
from amct_pytorch.common.utils.run_logging import ensure_log_dir, setup_run_logging


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
        self.a_bits, self.w_bits = args.a_bits, args.w_bits
        self.q_bits, self.k_bits, self.p_bits, self.v_bits = args.q_bits, args.k_bits, args.p_bits, args.v_bits


    def setup(self):
        self._register_components()
        self.pipeline = self._build_pipeline()
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
        logger.info(f"Model: {self.model_name}; Quant target: {self.quant_target}; "
                    f"a_bits: {self.a_bits}, w_bits: {self.w_bits} "
                    f"q_bits: {self.q_bits}, k_bits: {self.k_bits}, p_bits: {self.p_bits}, v_bits: {self.v_bits}; "
                    f"PPL: {ppl}")
        logger.remove(sink_id)

    def _register_components(self):
        register_llm_models()
        register_dtype()
        register_algorithms()

    def _build_pipeline(self):
        model_cls = MODEL_REGISTRY.get(self.model_name)
        return model_cls(self.args)

    def _get_relevant_quant_bits(self):
        quant_targets = set(self.args.quant_target)
        bits = []
        if quant_targets.intersection({"mlp", "moe", "attn-linear"}):
            bits.extend([self.args.a_bits, self.args.w_bits])
        if "attn-cache" in quant_targets:
            bits.extend([self.args.q_bits, self.args.k_bits, self.args.p_bits, self.args.v_bits])
        return bits

    def _resolve_eval_states(self):
        if self.eval_mode == "bf16":
            return False, False, "Eval mode=bf16: use original BF16 blocks."

        relevant_bits = self._get_relevant_quant_bits()
        enable_quant = any(bit < 16 for bit in relevant_bits)
        if enable_quant:
            return True, True, "Eval mode=quant: run quant modules with quantization enabled."
        return True, False, "Eval mode=quant with all relevant bits=16: run rebuilt quant modules with quantization disabled."

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
        model =  self.pipeline.float_model().eval().to(self.device)
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
