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
    "DeepseekV2Config",
    "DeepseekV2ForCausalLM",
    "DeepseekV2PreTrainedModel",
    "DeepseekV2",
]

from transformers import AutoConfig, AutoModelForCausalLM

from amct_pytorch.common.models.llm.deepseek.deepseek_v2.modeling.configuration_deepseek import DeepseekV2Config
from amct_pytorch.common.models.llm.deepseek.deepseek_v2.modeling.modeling_deepseek import (
    DeepseekV2ForCausalLM,
    DeepseekV2PreTrainedModel,
)
from amct_pytorch.common.models.llm.deepseek.deepseek_v2.deepseekv2 import DeepseekV2

AutoConfig.register("deepseek_v2", DeepseekV2Config, exist_ok=True)
AutoModelForCausalLM.register(DeepseekV2Config, DeepseekV2ForCausalLM, exist_ok=True)