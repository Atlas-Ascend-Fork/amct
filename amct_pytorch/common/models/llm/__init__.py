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

from importlib import import_module


_REGISTERED = False

_MODEL_TYPE_REGISTRATIONS = {
    "deepseek_v32": (
        "deepseek_v3_2",
        ".deepseek.deepseek_v3_2.deepseekv3_2",
    ),
    "deepseek_v4": ("deepseek_v4", ".deepseek.deepseek_v4.deepseekv4"),
    "longcat_flash": ("longcat_lite", ".longcat.longcat_lite.longcat_lite"),
    "longcat_next": ("longcat_next", ".longcat.longcat_next.longcat_next"),
    "qwen3": ("qwen3", ".qwen.qwen3.qwen3"),
    "qwen3_moe": ("qwen3_moe", ".qwen.qwen3.qwen3_moe"),
    "qwen3_5": ("qwen3_5", ".qwen.qwen3_5.qwen3_5"),
    "qwen3_5_moe": ("qwen3_5_moe", ".qwen.qwen3_5.qwen3_5_moe"),
    "qwen3_6_moe": ("qwen3_6_moe", ".qwen.qwen3_6.qwen3_6_moe"),
    "qwen3_next": ("qwen3_next", ".qwen.qwen3_next.qwen3_next"),
    "glm5": ("glm5", ".glm.glm5.glm5"),
    "hy_v3": ("hy_v3", ".hyv3.hyv3"),
    "glm_moe_dsa": ("glm5_2", ".glm.glm5_2.glm5_2"),
}


def register_llm_models(model_type=None):
    if model_type is not None:
        registration = _MODEL_TYPE_REGISTRATIONS.get(model_type)
        if registration is not None:
            import_module(registration[1], package=__name__)
        return

    global _REGISTERED
    if _REGISTERED:
        return

    from .deepseek.deepseek_v4_1.deepseekv4_1 import DeepseekV41  # noqa: F401
    from .deepseek.deepseek_v3_2.deepseekv3_2 import DeepseekV32  # noqa: F401
    from .deepseek.deepseek_v4.deepseekv4 import DeepseekV4  # noqa: F401
    from .longcat.longcat_lite.longcat_lite import LongcatLite  # noqa: F401
    from .longcat.longcat_next.longcat_next import LongcatNext  # noqa: F401
    from .qwen.qwen3_5.qwen3_5 import Qwen3_5  # noqa: F401
    from .qwen.qwen3_5.qwen3_5_moe import Qwen3_5Moe  # noqa: F401
    from .qwen.qwen3_6.qwen3_6_moe import Qwen3_6Moe  # noqa: F401
    from .qwen.qwen3.qwen3 import Qwen3  # noqa: F401
    from .qwen.qwen3_next.qwen3_next import Qwen3Next  # noqa: F401
    from .qwen.qwen3.qwen3_moe import Qwen3Moe  # noqa: F401
    from .glm.glm5.glm5 import GLM5  # noqa: F401
    from .hyv3.hyv3 import HyV3  # noqa: F401
    from .glm.glm5_2.glm5_2 import GLM5_2  # noqa: F401

    _REGISTERED = True


def available_llm_adapters():
    """Return adapter names keyed by HuggingFace model_type without importing them."""
    return {
        adapter_name: model_type
        for model_type, (adapter_name, _) in _MODEL_TYPE_REGISTRATIONS.items()
    }


__all__ = ["available_llm_adapters", "register_llm_models"]
