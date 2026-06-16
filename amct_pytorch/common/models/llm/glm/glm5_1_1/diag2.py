#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""精确诊断：追踪 GlmMoeDsaAttention 构造过程中的每个维度值。

用法：
    python3 amct_pytorch/common/models/llm/glm/glm5_1_1/diag2.py /path/to/glm-5_1
"""
import json
import os
import sys
import types

import torch
from transformers import AutoConfig
from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaAttention


def main(model_path):
    config_file = os.path.join(model_path, "config.json")
    with open(config_file, encoding="utf-8") as f:
        raw = json.load(f)

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    # --- 在构造前记录 config 属性 ---
    keys = ("kv_lora_rank", "qk_rope_head_dim", "qk_nope_head_dim",
            "q_lora_rank", "v_head_dim", "hidden_size", "qk_head_dim",
            "num_attention_heads", "num_key_value_heads",
            "index_head_dim")
    print("=" * 60)
    print("config 对象属性 (构造 attention 前)")
    for k in keys:
        v = getattr(config, k, "MISSING")
        print(f"  config.{k} = {v}")

    # --- 用 monkey-patch 拦截 GlmMoeDsaAttention.__init__ 内的赋值 ---
    orig_init = GlmMoeDsaAttention.__init__
    captured = {}

    def patched_init(self, cfg, layer_idx):
        # 在调用原始 __init__ 前，先记录传入 cfg 的值
        captured["pre_kv_lora"] = getattr(cfg, "kv_lora_rank", "?")
        captured["pre_qk_rope"] = getattr(cfg, "qk_rope_head_dim", "?")
        captured["pre_qk_nope"] = getattr(cfg, "qk_nope_head_dim", "?")
        captured["pre_hidden"] = getattr(cfg, "hidden_size", "?")

        # 劫持 self 的属性赋值，记录 self.xxx = yyy 的时刻
        real_setattr = self.__setattr__
        trace_log = []

        def tracing_setattr(name, value):
            if name in ("kv_lora_rank", "qk_rope_head_dim", "qk_nope_head_dim",
                         "q_lora_rank", "v_head_dim", "qk_head_dim", "hidden_size",
                         "num_heads", "num_key_value_groups"):
                trace_log.append(f"  self.{name} = {value}")
            real_setattr(name, value)

        self.__setattr__ = tracing_setattr
        try:
            orig_init(self, cfg, layer_idx)
        finally:
            self.__setattr__ = real_setattr

        captured["trace"] = trace_log
        captured["final_kv_lora"] = self.kv_lora_rank
        captured["final_qk_rope"] = self.qk_rope_head_dim
        captured["final_qk_nope"] = self.qk_nope_head_dim
        captured["kv_proj_shape"] = self.kv_a_proj_with_mqa.weight.shape
        captured["expected"] = self.kv_lora_rank + self.qk_rope_head_dim

    GlmMoeDsaAttention.__init__ = patched_init

    attn = GlmMoeDsaAttention(config, 0)
    GlmMoeDsaAttention.__init__ = orig_init

    print()
    print("=" * 60)
    print("__init__ 执行追踪")
    print(f"  传入 cfg.kv_lora_rank      = {captured['pre_kv_lora']}")
    print(f"  传入 cfg.qk_rope_head_dim   = {captured['pre_qk_rope']}")
    print(f"  传入 cfg.qk_nope_head_dim   = {captured['pre_qk_nope']}")
    print()
    print("  self 赋值顺序:")
    for line in captured["trace"]:
        print(line)
    print()
    print(f"  最终 self.kv_lora_rank     = {captured['final_kv_lora']}")
    print(f"  最终 self.qk_rope_head_dim  = {captured['final_qk_rope']}")
    print(f"  最终 self.qk_nope_head_dim  = {captured['final_qk_nope']}")
    print()
    print(f"  kv_a_proj_with_mqa.weight  = {captured['kv_proj_shape']}")
    print(f"  期望 (kv_lora + qk_rope)   = {captured['expected']}")

    if captured["kv_proj_shape"][0] != captured["expected"]:
        diff = captured["kv_proj_shape"][0] - captured["expected"]
        print()
        print("  *** 维度不匹配! ***")
        print(f"  差值: {diff}")
        # 尝试匹配
        final = captured["final_kv_lora"]
        rope = captured["final_qk_rope"]
        nope = captured["final_qk_nope"]
        actual = captured["kv_proj_shape"][0]
        candidates = {
            f"kv_lora + qk_nope ({final}+{nope})": final + nope,
            f"kv_lora + qk_rope + index_hd ({final}+{rope}+{config.index_head_dim})": final + rope + config.index_head_dim,
            f"kv_lora + 2*qk_rope ({final}+{2*rope})": final + 2 * rope,
        }
        for desc, val in candidates.items():
            if val == actual:
                print(f"  匹配: {desc} = {val}")
                break
    else:
        print()
        print("  *** 维度正确 ✓ ***")

    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python3 diag2.py <model_path>")
        sys.exit(1)
    main(sys.argv[1])
