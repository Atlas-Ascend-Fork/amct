# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

__all__ = ["register_llm_models"]

import importlib

_REGISTERED = False

# (module suffix under this package, label) per LLM adapter. Importing the
# module triggers its @register decorator (classes are import-for-side-effect).
_LLM_ADAPTERS = (
    (".deepseek.deepseek_v3_2.deepseekv3_2", "DeepseekV32"),
    (".deepseek.deepseek_v4.deepseekv4", "DeepseekV4"),
    (".longcat.longcat_lite.longcat_lite", "LongcatLite"),
    (".longcat.longcat_next.longcat_next", "LongcatNext"),
    (".qwen.qwen3_5.qwen3_5", "Qwen3_5"),
    (".qwen.qwen3_5.qwen3_5_moe", "Qwen3_5Moe"),
    (".qwen.qwen3_6.qwen3_6_moe", "Qwen3_6Moe"),
    (".qwen.qwen3.qwen3", "Qwen3"),
    (".qwen.qwen3_next.qwen3_next", "Qwen3Next"),
    (".qwen.qwen3.qwen3_moe", "Qwen3Moe"),
    (".glm.glm5.glm5", "GLM5"),
)


def register_llm_models():
    global _REGISTERED
    if _REGISTERED:
        return

    for module_suffix, label in _LLM_ADAPTERS:
        try:
            importlib.import_module(module_suffix, __name__)
        except ModuleNotFoundError as exc:
            # An optional adapter dependency is missing; skip just this adapter
            # rather than taking down registration of every other model.
            print(f"[WARN] Skip LLM model adapter {label!r}: {exc}")

    _REGISTERED = True
