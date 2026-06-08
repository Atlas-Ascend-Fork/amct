# -*- coding: UTF-8 -*-
"""Canonical WikiText-2 PPL evaluation — single source of truth.

This is THE implementation of the AMCT PPL protocol (chunked, seqlen-weighted NLL,
exp(sum_nll / (nsamples * seqlen))). All eval scripts (ppl_eval.py,
ppl_eval_amct.py, ppl_eval_flatquant.py) import eval_ppl from here so the formula
lives in exactly one place — change it once, every script stays consistent.

Protocol matches amct_pytorch/experimental/hifloat8/utils.py:test_ppl.

Import this at the TOP of a script, before any sys.path manipulation: the scripts
directory path contains 'gitCode/cann/amct', so the protobuf-fallback path strip in
ppl_eval_amct.py / ppl_eval_flatquant.py would remove this module's directory if the
import ran after it.
"""
import json
import time

import torch
import torch.nn as nn
import tqdm


def eval_ppl(model, testenc, seqlen, device,
             start_chunk=0, max_chunks=None, label="ppl",
             nll_dump_path=None):
    """Compute PPL over chunks [start_chunk, start_chunk + max_chunks).

    Args:
        model: a causal LM returning .logits.
        testenc: tokenized eval set, shape [1, total_tokens].
        seqlen: chunk length (2048 for the AMCT standard).
        device: torch device string.
        start_chunk / max_chunks: chunk window — lets a long eval be split across
            multiple Bash invocations (each capped at 600s in subagent context).
        label: tqdm description.
        nll_dump_path: if given, dump raw per-chunk NLLs as JSON so partial runs can
            be merged later via merge_ppl.py.

    Returns:
        (ppl, elapsed_seconds, nsamples_used)
    """
    nlls = []
    total_chunks = testenc.numel() // seqlen
    end_chunk = total_chunks if max_chunks is None else min(total_chunks, start_chunk + max_chunks)
    t0 = time.time()
    for i in tqdm.tqdm(range(start_chunk, end_chunk), desc=label):
        batch = testenc[:, i * seqlen : (i + 1) * seqlen].to(device)
        with torch.no_grad():
            logits = model(batch).logits
        shift_logits = logits[:, :-1, :].contiguous().float()
        shift_labels = batch[:, 1:]
        loss = nn.CrossEntropyLoss()(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        nlls.append(loss.float() * seqlen)
    dt = time.time() - t0
    if nll_dump_path is not None:
        with open(nll_dump_path, "w") as f:
            json.dump({
                "start_chunk": start_chunk,
                "end_chunk": end_chunk,
                "seqlen": seqlen,
                "total_chunks": total_chunks,
                "nlls": [n.item() for n in nlls],
            }, f)
    nsamples = end_chunk - start_chunk
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))
    return ppl.item(), dt, nsamples
