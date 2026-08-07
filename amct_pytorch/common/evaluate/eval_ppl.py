# coding=utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
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

import time

import torch.nn as nn
from loguru import logger
import torch
from tqdm import tqdm


_GIB = 1024**3


def wikitext2_ppl(preds, samples, seq_len=4096):
    nlls = []
    nsamples = len(samples)
    for i, sample in tqdm(enumerate(samples), desc="Evaluating"):
        shift_logits = preds[i]
        shift_labels = sample[:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
        )
        if i == 0:
            logger.info(f"iter {i}:loss {loss}")
        neg_log_likelihood = loss.float() * seq_len
        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seq_len))
    torch.npu.empty_cache()
    return ppl.item()


def _same_device(actual, expected):
    return torch.device(actual) == torch.device(expected)


def _is_npu_device(device):
    return str(device).split(":", maxsplit=1)[0] == "npu"


def _reset_npu_peak(device):
    torch.npu.reset_peak_memory_stats(device)


def _read_npu_memory(device):
    return {
        "allocated": torch.npu.memory_allocated(device),
        "reserved": torch.npu.memory_reserved(device),
        "peak_allocated": torch.npu.max_memory_allocated(device),
        "peak_reserved": torch.npu.max_memory_reserved(device),
    }


def _warn_without_raising(message, error):
    try:
        logger.warning(message, error)
    except Exception:
        pass


def _empty_npu_cache(device):
    if not _is_npu_device(device):
        return
    try:
        torch.npu.empty_cache()
    except Exception as error:
        _warn_without_raising(
            "Failed to empty NPU cache after streaming PPL: {}",
            error,
        )


def wikitext2_ppl_streaming(logits_iter, samples, device, seq_len=4096):
    nsamples = len(samples)
    if nsamples == 0:
        raise ValueError("samples must not be empty")

    iterator = iter(logits_iter)
    loss_fct = nn.CrossEntropyLoss()
    nll_sum = None
    consumed = 0
    compute_start = time.perf_counter()
    memory_query_total = 0.0
    memory_records = []
    monitor_enabled = _is_npu_device(device)
    monitor_complete = True
    if monitor_enabled:
        try:
            _reset_npu_peak(device)
        except Exception as error:
            _warn_without_raising(
                "NPU memory monitoring is unavailable for streaming PPL: {}",
                error,
            )
            monitor_enabled = False
            monitor_complete = False

    try:
        for i, shift_logits in enumerate(iterator):
            if i >= nsamples:
                raise ValueError("logits count exceeds samples count")
            if not _same_device(shift_logits.device, device):
                raise ValueError(
                    f"logits are on {shift_logits.device}; expected device {device}"
                )
            shift_labels = samples[i][:, 1:].to(
                device=shift_logits.device,
                dtype=torch.long,
            )
            loss = loss_fct(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
            )
            nll = loss.float() * seq_len
            nll_sum = nll if nll_sum is None else nll_sum + nll
            consumed += 1

            if monitor_enabled:
                query_start = time.perf_counter()
                try:
                    memory = _read_npu_memory(device)
                except Exception as error:
                    _warn_without_raising(
                        "NPU memory monitoring is unavailable for streaming PPL: {}",
                        error,
                    )
                    monitor_enabled = False
                    monitor_complete = False
                else:
                    memory_records.append(
                        {
                            "index": i,
                            "shape": tuple(shift_logits.shape),
                            "dtype": str(shift_logits.dtype),
                            "device": str(shift_logits.device),
                            **memory,
                        }
                    )
                finally:
                    memory_query_total += time.perf_counter() - query_start

            del shift_logits, shift_labels, loss, nll

        if consumed != nsamples:
            raise ValueError(
                f"logits count {consumed} does not match samples count {nsamples}"
            )
        ppl_value = torch.exp(nll_sum / (nsamples * seq_len)).item()
        compute_total = time.perf_counter() - compute_start

        log_start = time.perf_counter()
        try:
            for record in memory_records:
                logger.info(
                    "[EvalEvidence] stage=streaming_ppl_sample index={} shape={} "
                    "dtype={} device={} allocated_gib={:.6f} "
                    "reserved_gib={:.6f} peak_allocated_gib={:.6f} "
                    "peak_reserved_gib={:.6f}",
                    record["index"],
                    record["shape"],
                    record["dtype"],
                    record["device"],
                    record["allocated"] / _GIB,
                    record["reserved"] / _GIB,
                    record["peak_allocated"] / _GIB,
                    record["peak_reserved"] / _GIB,
                )
        except Exception as error:
            _warn_without_raising(
                "Failed to emit streaming PPL memory records: {}",
                error,
            )
        memory_log_emit = time.perf_counter() - log_start

        if memory_records and monitor_complete:
            max_allocated = max(record["peak_allocated"] for record in memory_records)
            max_reserved = max(record["peak_reserved"] for record in memory_records)
            peak_allocated = f"{max_allocated / _GIB:.6f}"
            peak_reserved = f"{max_reserved / _GIB:.6f}"
        else:
            peak_allocated = "n/a"
            peak_reserved = "n/a"
        try:
            logger.info(
                "[EvalEvidence] stage=streaming_ppl_complete ppl={:.6f} "
                "compute_total_s={:.6f} memory_query_total_s={:.6f} "
                "memory_log_emit_s={:.6f} peak_allocated_gib={} "
                "peak_reserved_gib={}",
                ppl_value,
                compute_total,
                memory_query_total,
                memory_log_emit,
                peak_allocated,
                peak_reserved,
            )
        except Exception as error:
            _warn_without_raising(
                "Failed to emit streaming PPL completion record: {}",
                error,
            )
        return ppl_value
    finally:
        try:
            close = getattr(iterator, "close", None)
            if close is not None:
                close()
        except Exception as error:
            _warn_without_raising(
                "Failed to close streaming PPL logits iterator: {}",
                error,
            )
        try:
            _empty_npu_cache(device)
        except Exception as error:
            _warn_without_raising(
                "Failed to empty NPU cache after streaming PPL: {}",
                error,
            )
