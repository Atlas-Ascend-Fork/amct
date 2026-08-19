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

import torch
import torch.nn as nn
from loguru import logger
from tqdm import tqdm


def _same_device(actual, expected):
    return torch.device(actual) == torch.device(expected)


def _is_npu_device(device):
    return str(device).split(":", maxsplit=1)[0] == "npu"


def _cleanup_ppl_resources(iterator, device):
    """Release the logits iterator and cached NPU memory after PPL evaluation.

    Args:
        iterator: Iterator that provides logits during PPL evaluation.
        device: Device used for PPL evaluation.
    """
    close = getattr(iterator, "close", None)
    if close is not None:
        close()
    if _is_npu_device(device):
        torch.npu.empty_cache()


def wikitext2_ppl(logits_iter, samples, device, seq_len=4096):
    nsamples = len(samples)
    if nsamples == 0:
        raise ValueError("samples must not be empty")

    iterator = iter(logits_iter)
    nll_sum = None
    consumed = 0
    loss_fct = nn.CrossEntropyLoss()
    with tqdm(
        total=nsamples,
        desc="PPL Evaluating...",
    ) as progress:
        for shift_logits in iterator:
            if consumed >= nsamples:
                raise ValueError("logits count exceeds samples count")
            if not _same_device(shift_logits.device, device):
                raise ValueError(
                    f"logits are on {shift_logits.device}; expected device {device}"
                )
            shift_labels = samples[consumed][:, 1:].to(
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
            del shift_logits, shift_labels, loss, nll
            progress.update(1)

    if consumed != nsamples:
        raise ValueError(
            f"logits count {consumed} does not match samples count {nsamples}"
        )

    ppl_value = torch.exp(nll_sum / (nsamples * seq_len)).item()
    logger.info("PPL evaluation completed: {:.6f}", ppl_value)
    _cleanup_ppl_resources(iterator, device)
    return ppl_value
