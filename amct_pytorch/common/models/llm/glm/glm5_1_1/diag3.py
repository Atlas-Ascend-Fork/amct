#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""最底层诊断：绕过 AutoConfig，直接调用 GlmMoeDsaConfig 构造链。

找出 qk_rope_head_dim 从 64 变成 192 的确切位置。

用法：
    python3 amct_pytorch/common/models/llm/glm/glm5_1_1/diag3.py /path/to/glm-5_1
"""
import json
import os
import sys

from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig
from transformers import AutoConfig


def main(model_path):
    config_file = os.path.join(model_path, "config.json")
    with open(config_file, encoding="utf-8") as f:
        raw = json.load(f)

    print("=" * 60)
    print("A) 原始 config.json → GlmMoeDsaConfig.from_dict()")
    print()
    c_from_dict = GlmMoeDsaConfig.from_dict(raw)
    print(f"   qk_rope_head_dim = {c_from_dict.qk_rope_head_dim}")
    print(f"   qk_nope_head_dim = {c_from_dict.qk_nope_head_dim}")
    print(f"   kv_lora_rank     = {c_from_dict.kv_lora_rank}")

    print()
    print("=" * 60)
    print("B) AutoConfig.from_pretrained()")
    print()
    c_auto = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    print(f"   qk_rope_head_dim = {c_auto.qk_rope_head_dim}")
    print(f"   qk_nope_head_dim = {c_auto.qk_nope_head_dim}")
    print(f"   kv_lora_rank     = {c_auto.kv_lora_rank}")

    print()
    print("=" * 60)
    print("C) 手动模拟 attribute_map 处理逻辑")
    print()
    attr_map = GlmMoeDsaConfig.attribute_map
    print(f"   attribute_map = {attr_map}")
    print()

    # 模拟 from_dict 第一步：attribute_map 重命名
    remapped = dict(raw)
    for old_key, new_key in attr_map.items():
        if old_key in remapped:
            removed = remapped.pop(old_key)
            # 如果目标 key 已存在，可能会覆盖
            collision = new_key in remapped
            remapped[new_key] = removed
            print(f"   remap: '{old_key}' ({removed}) → '{new_key}'  {'(覆盖已有值!)' if collision else ''}")

    print()
    print("   处理后 dict 中 qk_rope_head_dim 相关键:")
    for k, v in sorted(remapped.items()):
        if "rope" in k.lower() or "nope" in k.lower() or "head_dim" in k.lower() or "lora" in k.lower():
            print(f"     {k} = {v}")

    # 构造实例
    c_manual = GlmMoeDsaConfig(**remapped)
    print()
    print(f"   构造后 qk_rope_head_dim = {c_manual.qk_rope_head_dim}")
    print(f"   构造后 qk_nope_head_dim = {c_manual.qk_nope_head_dim}")

    print()
    print("=" * 60)
    print("D) 对比结论")
    print()
    results = {
        "A from_dict(raw)": c_from_dict.qk_rope_head_dim,
        "B AutoConfig     ": c_auto.qk_rope_head_dim,
        "C manual remap   ": c_manual.qk_rope_head_dim,
    }
    for label, val in results.items():
        flag = "✓" if val == 64 else f"✗ (应为 64)"
        print(f"   {label}: {val} {flag}")

    if c_from_dict.qk_rope_head_dim != 64:
        print()
        print("   → from_dict() 就错了，问题在 GlmMoeDsaConfig.from_dict 内部")
    elif c_auto.qk_rope_head_dim != 64:
        print()
        print("   → from_dict() 正确但 AutoConfig 错了，问题在 get_config_dict 或缓存")

    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python3 diag3.py <model_path>")
        sys.exit(1)
    main(sys.argv[1])
