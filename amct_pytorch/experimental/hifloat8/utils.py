import time
import tqdm
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


def get_qwen(model_path, seqlen=2048):
    print(f'Getting official pretrained {model_path}')
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16)
    model.seqlen = seqlen
    tokenizer = AutoTokenizer.from_pretrained(
            model_path, use_fast=False, trust_remote_code=True
    )
    return model, tokenizer


def get_test_dataset(enc):
    print('Loading dataset: Wikitext2')
    testenc = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    testenc = enc("\n\n".join(testenc["text"]), return_tensors="pt")
    return testenc


def test_ppl(model, testenc):
    nlls = []
    nsamples = testenc.numel() // model.seqlen
    test_start_time = time.time()
    for i in tqdm.tqdm(range(nsamples), desc="evaluating..."):
        batch = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)].to(
            model.device
        )
        with torch.no_grad():
            lm_logits = model(batch).logits
        shift_logits = lm_logits[:, :-1, :].contiguous().float()
        shift_labels = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    test_end_time = time.time()

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))

    total_time = test_end_time - test_start_time
    print('Test time taken: ', total_time // 60, 'min ', total_time % 60, 's')
    print('Score: ', ppl.item())