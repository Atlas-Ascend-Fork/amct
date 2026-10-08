# -*- coding: UTF-8 -*-
"""Assemble a full MLP-only amct fake-quant model and run a generative benchmark.

Why MLP-only: quantized attention (QuantQwen3Attn) has no KV-cache support, so
quantizing attn forces generate(use_cache=False) → ~7 min/question. Quantizing
only MLP keeps attn native (KV cache on) → ~6 s/question. See references/constraints.md #1.

Assembly reuses amct blockwise building blocks (float_model + build_quant_block +
load_selected_layer_ptq_params) but assembles ALL layers into one generatable model
(amct's own eval never does this — it goes block-by-block for PPL).

--bf16 skips quant entirely and runs the original model through the SAME harness,
for an apples-to-apples baseline. Always run both and report the delta.
"""

import argparse
import os
import sys
import torch

SKILL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SKILL_DIR)  # for bench_offline / config
from config import DEFAULT_BIT_CONFIG  # noqa: E402
from utils import (  # noqa: E402
    check_environment,
    ensure_supported_quant_target,
    has_attention_quant,
    split_quant_targets,
)

# quant_target that keeps attn native (KV cache works): dense→mlp, MoE→moe.
_KVSAFE_TARGET = {
    "qwen3": "mlp",
    "qwen3_5": "mlp",
    "qwen3_5_moe": "moe",
    "qwen3_6_moe": "moe",
}
_TASK_ALIASES = {
    "math500": "hendrycks_math500",
}


def input_device(model):
    """Device the model's input_ids should go to (embedding's card, multi-card safe)."""
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return getattr(model, "device", torch.device("npu:0"))


def _check_no_offload(model):
    dmap = getattr(model, "hf_device_map", {}) or {}
    off = sorted({str(v) for v in dmap.values() if str(v) in ("cpu", "disk")})
    if off:
        raise RuntimeError(
            f"device_map offload to {off}: insufficient HBM. amct quant can't load "
            f"offloaded weights → wrong results. Add cards (ASCEND_RT_VISIBLE_DEVICES) "
            f"or lower precision."
        )
    used = sorted({str(v) for v in dmap.values()}) if dmap else ["npu:0"]
    print(f"[device] model spans: {used}")


def _get_layers(model):
    """Locate the decoder layer ModuleList (dense: model.model.layers;
    VL-nested MoE: model.model.language_model.layers)."""
    for getter in (lambda m: m.model.layers, lambda m: m.model.language_model.layers):
        try:
            return getter(model)
        except AttributeError:
            continue
    raise RuntimeError("cannot locate decoder layers on this model")


def _load_float(model_path, device_map):
    import torch_npu  # noqa: F401
    from transformers import AutoModelForCausalLM

    if device_map:
        m = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map=device_map,
            trust_remote_code=True,
        )
        _check_no_offload(m)
    else:
        m = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16, trust_remote_code=True
        ).to("npu:0")
    return m.eval()


def build_pipeline_args(
    model, model_name, param_dir, quant_target, bit_cfg, seq_len, output_dir
):
    from amct_pytorch.quantization.bit_policy import BitPolicy

    quant_targets = split_quant_targets(quant_target)
    ns = argparse.Namespace(
        model=model,
        model_name=model_name,
        device="npu:0",
        granularity="block",
        seed=0,
        quant_target=quant_targets,
        seq_len=seq_len,
        data_dir="",
        output_dir=output_dir,
        nsamples=128,
        eval_mode="quant",
        quant_dtype="int",
        cali_bsz=4,
        base_lr=1e-5,
        optimizer="adamw",
        weight_decay=0.0,
        momentum=0.9,
        lr_scheduler="cosine",
        min_lr=0.0,
        lr_step_size=1,
        lr_gamma=0.1,
        epochs=15,
        algos=["lwc", "lac"],
        is_per_tensor=False,
        k_size=128,
        start_block_idx=0,
        end_block_idx=61,
        bit_config=bit_cfg,
        attn_linear_param_dir=param_dir if "attn-linear" in quant_targets else "",
        attn_cache_param_dir=param_dir if "attn-cache" in quant_targets else "",
        moe_mlp_param_dir=param_dir
        if {"mlp", "moe"}.intersection(quant_targets)
        else "",
        wikitext_final_out="",
    )
    ns.bit_policy = BitPolicy.from_yaml(bit_cfg)
    return ns


def assemble_quant_model(
    model,
    model_name,
    param_dir,
    bit_cfg,
    seq_len,
    device_map=None,
    quant_target=None,
    output_dir="/tmp/qbe_out",
    allow_attn_quant=False,
):
    from amct_pytorch.common.models.llm import register_llm_models
    from amct_pytorch.quantization.dtypes import register_dtype
    from amct_pytorch.algorithms.quant import register_algorithms
    from amct_pytorch.common.models import MODEL_REGISTRY
    from amct_pytorch.quantization.modules.quant_linear import QuantLinear
    from amct_pytorch.common.models.llm.common.quant_apply import set_model_to_observe

    register_llm_models()
    register_dtype()
    register_algorithms()

    qt = quant_target or _KVSAFE_TARGET.get(model_name, "mlp")
    ensure_supported_quant_target(qt, allow_attn_quant=allow_attn_quant)
    args = build_pipeline_args(
        model, model_name, param_dir, qt, bit_cfg, seq_len, output_dir
    )
    pipe = MODEL_REGISTRY.get(model_name)(args)
    print(
        f"[assemble] loading float_model (bf16, device_map={device_map}, target={qt})..."
    )
    m = _load_float(model, device_map)
    layers = _get_layers(m)
    n = pipe.num_layers
    # Multi-card: device_map=auto attaches accelerate AlignDevicesHooks that move
    # activations to each layer's card as data flows layer→layer. Replacing a layer
    # object drops its hook → cross-card activations no longer realigned → matmul
    # "tensors on npu:0 and npu:1" crash at generate. So we re-attach the original
    # layer's hook (carries execution_device) onto the swapped-in quant block.
    try:
        from accelerate.hooks import add_hook_to_module
    except Exception:
        add_hook_to_module = None
    for i in range(n):
        # place quant block on the SAME card as the original layer (multi-card safe)
        dev = next(layers[i].parameters()).device
        old_hook = getattr(layers[i], "_hf_hook", None)
        qblock = pipe.build_quant_block(i)
        pipe.load_selected_layer_ptq_params(i, qblock, strict=False)
        set_model_to_observe(qblock, False)
        for mod in qblock.modules():
            if isinstance(mod, QuantLinear):
                mod.eval_mode = True
                mod.cached_eval_weight = None
        qblock = qblock.to(dev).eval().bfloat16()
        if old_hook is not None and add_hook_to_module is not None:
            add_hook_to_module(qblock, old_hook)  # preserve cross-card activation move
        layers[i] = qblock
        if (i + 1) % 6 == 0:
            print(
                f"[assemble] {i + 1}/{n} layers quantized"
                f"{' (+hook)' if old_hook is not None else ''}"
            )
    import torch_npu

    torch_npu.npu.empty_cache()
    if has_attention_quant(qt):
        m.config.use_cache = False
        print(
            "[warn] attention quant enabled; model.config.use_cache=False for generate"
        )
    print("[assemble] done.")
    return m, pipe.tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="model dir")
    ap.add_argument(
        "--model_name",
        default="qwen3",
        help="amct pipeline key (qwen3, qwen3_5, qwen3_5_moe, qwen3_6_moe)",
    )
    ap.add_argument(
        "--param_dir",
        default="",
        help="ptq params dir (mlp for dense, moe for MoE); ignored with --bf16",
    )
    ap.add_argument(
        "--quant_target",
        default="",
        help="override KV-safe target (dense→mlp, MoE→moe)",
    )
    ap.add_argument(
        "--bit_config",
        default=DEFAULT_BIT_CONFIG,
        help="default derived from $AMCT_HOME",
    )
    ap.add_argument(
        "--output_dir", default="/tmp/qbe_out", help="amct temporary output dir"
    )
    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument(
        "--device_map",
        default="",
        help="'auto' for multi-card (35B); empty=single npu:0 (8B)",
    )
    ap.add_argument("--task", default="mmlu_pro")
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="sample cap; omit for full benchmark",
    )
    ap.add_argument("--max_gen_toks", type=int, default=512)
    ap.add_argument(
        "--bf16", action="store_true", help="baseline: original model, no quant"
    )
    ap.add_argument(
        "--allow_attn_quant",
        action="store_true",
        help="explicitly allow attn-linear/attn-cache quant targets",
    )
    ap.add_argument("--smoke", action="store_true", help="only generate smoke test")
    ap.add_argument("--log_samples", action="store_true")
    a = ap.parse_args()

    check_environment(require_lm_eval=not a.smoke)
    dmap = a.device_map or None
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model, use_fast=False, trust_remote_code=True)
    if a.bf16:
        print(f"[bf16] loading original model (no quant, device_map={dmap})...")
        model = _load_float(a.model, dmap)
    else:
        if not a.param_dir:
            sys.exit("--param_dir required unless --bf16")
        model, tok = assemble_quant_model(
            a.model,
            a.model_name,
            a.param_dir,
            a.bit_config,
            a.seq_len,
            device_map=dmap,
            quant_target=(a.quant_target or None),
            output_dir=a.output_dir,
            allow_attn_quant=a.allow_attn_quant,
        )

    generate_use_cache = not (not a.bf16 and has_attention_quant(a.quant_target))
    print(f"[smoke] generate (use_cache={generate_use_cache})...")
    ids = tok("The capital of France is", return_tensors="pt").input_ids.to(
        input_device(model)
    )
    with torch.no_grad():
        out = model.generate(
            ids, max_new_tokens=16, do_sample=False, use_cache=generate_use_cache
        )
    print(
        "[smoke] ->", repr(tok.decode(out[0][ids.shape[1] :], skip_special_tokens=True))
    )
    if a.smoke:
        return

    import bench_offline

    bench_offline.install()
    from lm_eval.models.huggingface import HFLM
    from lm_eval import simple_evaluate

    lm = HFLM(
        pretrained=model, tokenizer=tok, batch_size=1, max_gen_toks=a.max_gen_toks
    )
    tag = "bf16" if a.bf16 else "quant"
    task = _TASK_ALIASES.get(a.task, a.task)
    print(f"[eval] {tag} task={task} limit={a.limit} max_gen_toks={a.max_gen_toks}")
    res = simple_evaluate(
        model=lm,
        tasks=[task],
        limit=a.limit,
        bootstrap_iters=0,
        log_samples=a.log_samples,
    )
    for t, mtr in res["results"].items():
        print(f"[RESULT] {tag} {t}: {mtr}")


if __name__ == "__main__":
    main()
