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

__all__ = [
    "DeepseekV4Config",
    "DeepseekV4ForCausalLM",
    "DeepseekV4PreTrainedModel",
]

from transformers import AutoConfig, AutoModelForCausalLM

from amct_pytorch.common.models.llm.deepseek.deepseek_v4.modeling.configuration_deepseek_v4 import DeepseekV4Config
from amct_pytorch.common.models.llm.deepseek.deepseek_v4.modeling.modeling_deepseek_v4 import (
    DeepseekV4ForCausalLM,
    DeepseekV4PreTrainedModel,
)

AutoConfig.register("deepseek_v4", DeepseekV4Config, exist_ok=True)
AutoModelForCausalLM.register(DeepseekV4Config, DeepseekV4ForCausalLM, exist_ok=True)
