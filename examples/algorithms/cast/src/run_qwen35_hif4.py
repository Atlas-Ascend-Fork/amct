"""Test Qwen3.5-9B with HiFloat4 weight-only quant on wikitext-2 PPL."""
import argparse
import torch
import torch_npu
from transformers import AutoModelForCausalLM, AutoTokenizer
import tqdm
import torch.nn as nn

import amct_pytorch as amct

MODEL_PATH = '/run/z84450661/modelscope/models/Qwen--Qwen3.5-9B/snapshots/master'
WIKI_DIR = '/run/z84450661/wikitext-2'


def load_wikitext_test(tokenizer, seqlen=2048):
    """Load wikitext-2 test from raw files and tokenize."""
    with open(f'{WIKI_DIR}/wiki.test.raw') as f:
        text = f.read()
    # Split into paragraphs and filter empty ones
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    # Join with double newlines (matching HF wikitext format)
    joined = "\n\n".join(paragraphs)
    enc = tokenizer(joined, return_tensors="pt")
    # Truncate to max model length
    max_len = getattr(tokenizer, 'model_max_length', 262144)
    if enc.input_ids.numel() > max_len:
        enc['input_ids'] = enc.input_ids[:, :max_len]
    return enc


def test_ppl(model, testenc, seqlen=2048):
    nlls = []
    nsamples = testenc.numel() // seqlen
    for i in tqdm.tqdm(range(nsamples), desc="evaluating..."):
        batch = testenc[:, (i * seqlen):((i + 1) * seqlen)].to(model.device)
        with torch.no_grad():
            lm_logits = model(batch).logits
        shift_logits = lm_logits[:, :-1, :].contiguous().float()
        shift_labels = testenc[:, (i * seqlen):((i + 1) * seqlen)][:, 1:].to(model.device)
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * seqlen
        nlls.append(neg_log_likelihood)

    import math
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))
    print(f'PPL: {ppl.item():.4f}')


if __name__ == '__main__':
    SEQLEN = 2048

    # Load model
    print('Loading Qwen3.5-9B...')
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, device_map='npu'
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False, trust_remote_code=True)

    # Quantize with HiFloat4 (weight-only)
    print('Applying HiF4 quantization...')
    cfg = {
        "batch_num": 1,
        "skip_layers": [],
        "weights": {"type": "hifloat4", "symmetric": True, "strategy": "channel"},
        "inputs": {"type": "hifloat4", "symmetric": True, "strategy": "tensor"},
        "algorithm": "cast",
    }
    amct.quantize(model, cfg)
    amct.convert(model)

    # Test PPL
    print('Loading wikitext-2 test...')
    testenc = load_wikitext_test(tokenizer, seqlen=SEQLEN)
    testenc = testenc.input_ids  # extract tensor from BatchEncoding
    print(f'Test tokens: {testenc.numel()}, seqlen={SEQLEN}')
    testenc = testenc.npu()
    test_ppl(model, testenc, seqlen=SEQLEN)
