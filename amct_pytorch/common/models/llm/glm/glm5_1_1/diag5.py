#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""终极追踪：hook PreTrainedConfig.__setattr__ + __post_init__ 全链路。

用法：
    python3 amct_pytorch/common/models/llm/glm/glm5_1_1/diag5.py /path/to/glm-5_1
"""
import json
import os
import sys
import traceback

from transformers.configuration_utils import PreTrainedConfig
from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig


def main(model_path):
    config_file = os.path.join(model_path, "config.json")
    with open(config_file, encoding="utf-8") as f:
        raw = json.load(f)

    # ----- Hook PreTrainedConfig.__setattr__ -----
    real_ptc_setattr = PreTrainedConfig.__setattr__
    trace = []

    def tracing_ptc_setattr(self, key, value):
        attr_map = object.__getattribute__(self, 'attribute_map')
        if key in attr_map:
            mapped_key = attr_map[key]
        else:
            mapped_key = key

        if mapped_key == "qk_rope_head_dim" or key == "qk_rope_head_dim":
            tb = traceback.extract_stack(limit=10)
            trace.append(
                f"  PTC.__setattr__({key!r}={value!r}) → remap → {mapped_key!r}={value!r}"
                f"  ← {tb[-3].filename.split('/')[-1]}:{tb[-3].lineno}"
            )

        real_ptc_setattr(self, key, value)

    PreTrainedConfig.__setattr__ = tracing_ptc_setattr

    # ----- Hook GlmMoeDsaConfig.__post_init__ -----
    real_glm_post = GlmMoeDsaConfig.__post_init__

    def tracing_glm_post(self, **kwargs):
        rope_kwargs = {k: v for k, v in kwargs.items()
                       if any(w in k.lower() for w in ('rope', 'head_dim', 'lora', 'nope', 'qk_'))}
        trace.append(f"  GlmMoeDsa.__post_init__ kwargs: {rope_kwargs}")
        result = real_glm_post(self, **kwargs)
        trace.append(f"  after GlmMoeDsa.__post_init__: qk_rope_head_dim = {self.qk_rope_head_dim}")
        return result

    GlmMoeDsaConfig.__post_init__ = tracing_glm_post

    # ----- Hook PreTrainedConfig.__post_init__ -----
    real_ptc_post = PreTrainedConfig.__post_init__

    def tracing_ptc_post(self, **kwargs):
        if hasattr(self, 'qk_rope_head_dim'):
            rope_kwargs = {k: v for k, v in kwargs.items()
                           if any(w in k.lower() for w in ('rope', 'head_dim', 'lora', 'nope', 'qk_'))}
            trace.append(f"  PTC.__post_init__ kwargs (rope): {rope_kwargs}")
            result = real_ptc_post(self, **kwargs)
            trace.append(f"  after PTC.__post_init__: qk_rope_head_dim = {self.qk_rope_head_dim}")
            return result
        return real_ptc_post(self, **kwargs)

    PreTrainedConfig.__post_init__ = tracing_ptc_post

    print("=" * 60)
    print("from_dict() 全链路追踪:")
    print()
    c = GlmMoeDsaConfig.from_dict(raw)
    print()
    for entry in trace:
        print(entry)

    print()
    print(f"最终 qk_rope_head_dim = {c.qk_rope_head_dim}")

    # Restore
    PreTrainedConfig.__setattr__ = real_ptc_setattr
    GlmMoeDsaConfig.__post_init__ = real_glm_post
    PreTrainedConfig.__post_init__ = real_ptc_post


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python3 diag5.py <model_path>")
        sys.exit(1)
    main(sys.argv[1])
