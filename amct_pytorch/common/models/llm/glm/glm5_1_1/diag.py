#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""临时诊断脚本：检查 GLM-5.1 config 加载后 kv_a_proj_with_mqa 维度是否正确。

用法：
    python3 amct_pytorch/common/models/llm/glm/glm5_1_1/diag.py /path/to/glm-5_1
"""
import json
import os
import sys

import torch
from transformers import AutoConfig
from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaAttention


def main(model_path):
    if not os.path.isdir(model_path):
        print(f"错误：模型路径不存在: {model_path}")
        sys.exit(1)

    config_file = os.path.join(model_path, "config.json")
    if not os.path.isfile(config_file):
        print(f"错误：config.json 不存在: {config_file}")
        sys.exit(1)

    # 1) 读取原始 config.json
    with open(config_file, encoding="utf-8") as f:
        raw = json.load(f)

    print("=" * 60)
    print("1) config.json 原始关键字段")
    for key in ("kv_lora_rank", "qk_rope_head_dim", "qk_nope_head_dim",
                "head_dim", "q_lora_rank", "v_head_dim", "hidden_size",
                "num_hidden_layers", "index_head_dim", "index_topk",
                "rope_interleave", "transformers_version"):
        print(f"   {key}: {raw.get(key, 'MISSING')}")

    # 2) AutoConfig.from_pretrained
    print()
    print("=" * 60)
    print("2) AutoConfig.from_pretrained 加载后的值")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    for key in ("kv_lora_rank", "qk_rope_head_dim", "qk_nope_head_dim",
                "v_head_dim", "q_lora_rank", "hidden_size",
                "num_hidden_layers"):
        val = getattr(config, key, "NOT FOUND")
        flag = ""
        if key == "qk_rope_head_dim" and val != 64:
            flag = "  ← 错误！应为 64"
        elif key == "kv_lora_rank" and val != 512:
            flag = "  ← 错误！应为 512"
        print(f"   {key}: {val}{flag}")

    # 3) 创建 attention 层，检查实际维度
    print()
    print("=" * 60)
    print("3) GlmMoeDsaAttention 实例化后 kv_a_proj_with_mqa 形状")
    attn = GlmMoeDsaAttention(config, 0)
    w = attn.kv_a_proj_with_mqa.weight
    expected = config.kv_lora_rank + config.qk_rope_head_dim
    actual_out = w.shape[0]
    actual_in = w.shape[1]

    status = "正确 ✓" if actual_out == expected else f"错误 ✗ (期望 {expected})"
    print(f"   kv_a_proj_with_mqa.weight: {w.shape}")
    print(f"   期望输出维度: kv_lora_rank + qk_rope_head_dim = {config.kv_lora_rank} + {config.qk_rope_head_dim} = {expected}")
    print(f"   实际: {actual_out}  ← {status}")
    if actual_out != expected:
        diff = actual_out - expected
        print(f"   差值: {diff}")
        if actual_out == config.kv_lora_rank + config.qk_nope_head_dim:
            print(f"   匹配: kv_lora_rank + qk_nope_head_dim = {config.kv_lora_rank + config.qk_nope_head_dim}")
            print(f"   → 结论: qk_rope_head_dim 被错误赋值为 qk_nope_head_dim({config.qk_nope_head_dim})")
        elif actual_out == config.kv_lora_rank + config.qk_rope_head_dim + config.index_head_dim:
            print(f"   匹配: kv_lora_rank + qk_rope_head_dim + index_head_dim = "
                  f"{config.kv_lora_rank + config.qk_rope_head_dim + config.index_head_dim}")
            print(f"   → 结论: 多加了 index_head_dim({config.index_head_dim})")

    # 4) 其他 attention 投影维度
    print()
    print("=" * 60)
    print("4) 其他 Attention 投影维度参考")
    projs = [
        ("q_a_proj",       attn.q_a_proj),
        ("q_b_proj",       attn.q_b_proj),
        ("kv_a_proj_with_mqa", attn.kv_a_proj_with_mqa),
        ("kv_b_proj",      attn.kv_b_proj),
        ("o_proj",         attn.o_proj),
    ]
    for name, mod in projs:
        print(f"   {name}: {mod.weight.shape}")

    print()
    print("=" * 60)
    print(f"结论: {'配置正常' if actual_out == expected else '需要修复 config.json 或升级 transformers'}")
    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python3 diag.py <model_path>")
        print("示例: python3 diag.py /mnt/workspace/gitCode/fujun19/model/glm-5_1")
        sys.exit(1)
    main(sys.argv[1])
