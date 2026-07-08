#!/usr/bin/env python3
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
try:
    import transformers.models.glm_moe_dsa.modeling_glm_moe_dsa as _glm_dsa_mod
    if not hasattr(_glm_dsa_mod, "apply_rotary_pos_emb_interleave"):
        _glm_dsa_mod.apply_rotary_pos_emb_interleave = lambda *a, **kw: (a[0], a[1])
    if not hasattr(_glm_dsa_mod, "eager_attention_forward"):
        _glm_dsa_mod.eager_attention_forward = lambda *a, **kw: (None, None)
except ImportError:
    pass
