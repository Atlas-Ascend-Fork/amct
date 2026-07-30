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
"""Shared whole-model export workflow base for deploy (pack) and ptq-convert (fake-quant).

Both deploy and ptq-convert load a float model, auto-detect/decode the source format, run a
per-block export, and write a whole checkpoint (shards + refreshed index). That machinery lives
here; subclasses supply only the two things that differ:

* ``_export_block``   -- per-layer export (deploy packs to HiF4/int/mxfp; convert fake-quants to float)
* ``_refresh_config`` -- output config.json rewrite (deploy keeps quantization_config; convert strips it)

Deploy additionally implements the tensor-granularity path (``_run_tensorwise``); block granularity
is shared here and is what ptq-convert uses.
"""
from __future__ import annotations

import os

from loguru import logger
from tqdm import tqdm

from amct_pytorch.algorithms.quant import register_algorithms
from amct_pytorch.common.models import MODEL_REGISTRY
from amct_pytorch.common.models.llm import register_llm_models
from amct_pytorch.common.utils.checkpoint_io import CheckpointShardWriter
from amct_pytorch.common.utils.run_logging import ensure_log_dir, setup_run_logging
from amct_pytorch.quantization.dtypes import register_dtype


class LlmExportWorkflow:
    """Whole-model per-block checkpoint export with source-format auto-detect.

    Common base for ``LlmDeployWorkflow`` (pack) and ``LlmPtqWorkflow`` (calibrate / convert).
    Subclasses override ``_export_block`` and ``_refresh_config`` (and, for tensor granularity,
    ``_tensorwise_export``). This base owns setup, the detect/decode front-end, the block loop, and
    the shard/index writing."""

    def __init__(self, args):
        self.args = args
        self.granularity = args.granularity
        self.pipeline = None
        self.model_name = args.model_name
        self.model_path = args.model
        self.quant_dtype = args.quant_dtype
        self.output_dir = self.args.output_dir
        self.is_mx = self.quant_dtype.startswith("mx")
        self.is_int = self.quant_dtype.startswith("int")
        self.is_hif = self.quant_dtype.startswith("hif")

    @staticmethod
    def _register_components():
        register_llm_models()
        register_dtype()
        register_algorithms()

    def run(self):
        sink_id = self.setup()
        if self.granularity == "block":
            results = self._blockwise_export()
        elif self.granularity == "tensor":
            results = self._tensorwise_export()
        else:
            raise ValueError(
                f"Unsupported granularity '{self.granularity}' for export."
            )
        logger.remove(sink_id)
        return results

    def setup(self):
        os.makedirs(self.output_dir, exist_ok=True)
        ensure_log_dir(self.args)
        self._register_components()
        self._detect_and_decode_source()
        self.pipeline = self._build_pipeline()
        sink_id, _ = setup_run_logging(self.args, "deploy")
        return sink_id

    def _detect_and_decode_source(self):
        """Auto-detect a foreign FP4/FP8 source and decode it to BF16 before export.

        On this baseline branch ``FOREIGN_FORMATS`` is empty, so this is a detect-and-log no-op;
        downstream branches register nvfp4/mxfp4/mxfp8 and this decodes them to a temp BF16
        checkpoint first, pointing the pipeline at it. fp16/bf16 pass through unchanged.
        """
        from amct_pytorch.common.utils.decode.detect import (
            detect_source_format, decode_to_bf16, FOREIGN_FORMATS,
        )
        fmt = detect_source_format(self.model_path)
        self.source_fmt = fmt
        logger.info("Export source format detected: {}", fmt)
        if fmt in FOREIGN_FORMATS:
            decoded = os.path.join(self.output_dir, "_decoded_bf16")
            logger.info("Decoding {} -> BF16 at {} before export", fmt, decoded)
            decode_to_bf16(self.model_path, decoded, fmt)
            self.model_path = decoded
            self.args.model = decoded

    def _build_pipeline(self):
        model_cls = MODEL_REGISTRY.get(self.model_name)
        return model_cls(self.args)

    # ------------------------------------------------------------------ hooks
    def _export_block(self, layer_idx, quant_ignore_layers):
        """Per-layer export hook. Deploy packs (``export_block_deploy``); ptq-convert fake-quants
        to float (``export_block_fakequant``)."""
        raise NotImplementedError

    def _refresh_config(self, quant_ignore_layers):
        """Rewrite output config.json. Deploy writes quantization_config; ptq-convert strips it."""
        raise NotImplementedError

    def _tensorwise_export(self):
        """Tensor-granularity export -- implemented by deploy only."""
        raise NotImplementedError(
            "tensor granularity is not supported by this workflow"
        )

    # -------------------------------------------------------------- machinery
    def _blockwise_export(self):
        # Local writer bound to the *current* model_path (post foreign-format decode) / output_dir.
        writer = CheckpointShardWriter(self.model_path, self.output_dir)
        writer.copy_support_files()
        quant_ignore_layers = []
        original_index = writer.load_weight_index()
        original_weight_map = dict(original_index.get("weight_map", {}))
        updated_weight_map = {}
        replaced_original_weights = set()
        for layer_idx in tqdm(range(self.pipeline.num_layers), desc="Block Processing..."):
            layer_tensors, tensor_routes = self._export_block(layer_idx, quant_ignore_layers)
            if not layer_tensors:
                continue
            updated_weight_map.update(
                writer.write_block_file(layer_idx, self.pipeline.num_layers, layer_tensors)
            )
            replaced_original_weights.update(
                writer.collect_replaced_original_weights(
                    layer_tensors,
                    tensor_routes,
                    original_weight_map,
                )
            )

        updated_weight_map.update(
            writer.write_remaining_original_weights(
                original_weight_map,
                replaced_original_weights,
            )
        )
        index_path = writer.refresh_weight_index(original_index, updated_weight_map)
        self._refresh_config(quant_ignore_layers)
        logger.info("Exported model to {}", self.output_dir)
        logger.info("Refreshed weight index at {}", index_path)
        return {
            "index_path": index_path,
            "num_output_files": len(set(updated_weight_map.values())),
        }
