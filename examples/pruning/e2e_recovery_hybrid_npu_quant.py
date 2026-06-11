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
"""真·端到端（NPU）：守卫式 recovery-HYBRID 剪枝 → 真实 int8 量化发火 → convert，测真实 ppl。

验证 HYBRID 剪枝后的模型能走完整部署流水线：guarded_recovery_hybrid_prune 选最优 recovery
（none/bias/ls）就地剪 FFN → amct.quantize 在 fp16 上真正把 Linear 换成量化模块 → amct.convert。
CPU 上量化是 no-op（见 tests/.../test_pruning_quantize_compose.py 的 CPU 组合用例），故真实发火须 NPU+fp16。

运行：ASCEND_RT_VISIBLE_DEVICES=<id> HF_HUB_DISABLE_XET=1 python examples/pruning/e2e_recovery_hybrid_npu_quant.py
"""

import json
import os
import time

import torch
import torch_npu
import amct_pytorch as amct
from amct_pytorch.experimental.pruning import guarded_recovery_hybrid_prune

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
_NPU_AVAILABLE = torch_npu is not None and torch.npu.is_available()
MODEL = "HuggingFaceTB/SmolLM-135M"
DEV = "npu:0"  # ASCEND_RT_VISIBLE_DEVICES=<id> 选物理卡
RESULT = os.path.join(
    os.environ.get(
        "AMCT_PRUNE_RESULT_DIR",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"),
    ),
    "e2e_recovery_hybrid_npu_quant.json",
)

TEXT = (
    "Machine learning is a field of study in artificial intelligence concerned with the development "
    "of statistical algorithms that can learn from data and generalize to unseen examples. Deep neural "
    "networks have driven much of the recent progress in vision, language, and speech. Model compression "
    "techniques such as pruning and quantization reduce the size and cost of these networks."
) * 6


@torch.no_grad()
def ppl(model, ids):
    out = model(ids, labels=ids)
    return float(torch.exp(out.loss.float()))


def nparams(m):
    return sum(p.numel() for p in m.parameters())


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    base = (
        AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to(DEV).eval()
    )
    ids = tok(TEXT, return_tensors="pt").input_ids[:, :512].to(DEV)
    n = int(ids.shape[1])
    chunks = [ids[:, i : i + 64] for i in range(0, n - 63, 64)]  # full 64-token chunks
    half = max(1, len(chunks) // 2)
    calib = chunks[:half]
    val = chunks[half:] or chunks[:1]  # 保证验证集非空（与 calib disjoint）
    val_ids = torch.cat(val, dim=1)  # 守卫在此 held 集上按 ppl(=上报指标)择优，而非默认 top-1 fidelity

    base_ppl = ppl(base, ids)
    base_params = nparams(base)
    print(f"[NPU-E2E] baseline(fp16): params={base_params:,} ppl={base_ppl:.4f}")

    # (1) recovery-HYBRID 剪 FFN（守卫自动择优 none/bias/ls，就地应用）
    t0 = time.time()
    res = guarded_recovery_hybrid_prune(
        base, data=calib, prune_ratio=0.3, domain="dense", eval_data=val,
        quality_fn=lambda m: -ppl(m, val_ids), apply=True  # 门控 == 上报(ppl)，非默认 fidelity
    )
    ppl_pruned = ppl(base, ids)
    print(
        f"[NPU-E2E] recovery-HYBRID prune: pick={res.chosen_recovery} "
        f"(gain_over_none={res.gain_over_fallback:+.4f}) ppl {base_ppl:.4f} → {ppl_pruned:.4f} "
        f"params {base_params:,} → {nparams(base):,} (−{res.weight_reduction * 100:.1f}%) {time.time() - t0:.1f}s"
    )

    # (2) 在剪枝后的模型上真实 int8 量化 → convert（NPU+fp16 真正发火）
    t1 = time.time()
    amct.quantize(base)
    amct.convert(base)
    quant_types = sorted(
        {type(m).__name__ for m in base.modules() if "Quant" in type(m).__name__}
    )
    ppl_pq = ppl(base, ids)
    fired = bool(quant_types)
    print(
        f"[NPU-E2E] +int8 quant→convert: ppl {ppl_pruned:.4f} → {ppl_pq:.4f} "
        f"quant_fired={fired} quant_modules={quant_types} {time.time() - t1:.1f}s"
    )

    rec = {
        "model": MODEL,
        "device": DEV,
        "recovery_pick": res.chosen_recovery,
        "recovery_quality": res.recovery_quality,
        "gain_over_none": res.gain_over_fallback,
        "base_ppl": base_ppl,
        "pruned_ppl": ppl_pruned,
        "prune_quant_ppl": ppl_pq,
        "base_params": base_params,
        "pruned_params": nparams(base),
        "weight_reduction": res.weight_reduction,
        "quant_fired": fired,
        "quant_module_types": quant_types,
    }
    os.makedirs(os.path.dirname(RESULT), exist_ok=True)
    with open(RESULT, "w") as f:
        json.dump(rec, f, indent=1)
    print(f"[NPU-E2E] 已保存: {RESULT}")
    if not fired:
        print("[NPU-E2E] 警告: 量化未发火（需 NPU+fp16；CPU/fp32 为 no-op）。")


if __name__ == "__main__":
    if not _NPU_AVAILABLE:
        raise SystemExit("[NPU-E2E] 需要真实 NPU + torch_npu（CPU 上量化是 no-op）。")
    main()
