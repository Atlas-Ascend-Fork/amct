#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""终极追踪：hook setattr 记录构造过程中 qk_rope_head_dim 的每一次赋值。

用法：
    python3 amct_pytorch/common/models/llm/glm/glm5_1_1/diag4.py /path/to/glm-5_1
"""
import json
import os
import sys
import traceback

from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig


def main(model_path):
    config_file = os.path.join(model_path, "config.json")
    with open(config_file, encoding="utf-8") as f:
        raw = json.load(f)

    # Hook setattr on GlmMoeDsaConfig to trace qk_rope_head_dim
    real_setattr = GlmMoeDsaConfig.__setattr__
    trace = []

    def tracing_setattr(self, name, value):
        if name == "qk_rope_head_dim":
            tb = traceback.extract_stack(limit=12)
            caller = tb[-3] if len(tb) >= 3 else tb[-1]
            trace.append(f"  setattr(qk_rope_head_dim, {value})  ← {caller.filename.split('/')[-1]}:{caller.lineno} in {caller.name}")
        real_setattr(self, name, value)

    GlmMoeDsaConfig.__setattr__ = tracing_setattr

    print("=" * 60)
    print("from_dict() 构造期间 qk_rope_head_dim 的每次 setattr:")
    print()
    c = GlmMoeDsaConfig.from_dict(raw)
    print()
    for entry in trace:
        print(entry)

    print()
    print(f"最终 qk_rope_head_dim = {c.qk_rope_head_dim}")

    GlmMoeDsaConfig.__setattr__ = real_setattr


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python3 diag4.py <model_path>")
        sys.exit(1)
    main(sys.argv[1])
