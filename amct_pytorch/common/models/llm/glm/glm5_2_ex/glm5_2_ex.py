# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

import os

from transformers import AutoConfig

from amct_pytorch.common.models import MODEL_REGISTRY
from amct_pytorch.common.models.llm.deepseek.deepseek_v3_2.deepseekv3_2 import DeepseekV32


CUR_DIR = os.path.dirname(__file__)


@MODEL_REGISTRY.register(
    name="glm_5_2_ex",
    task="llm",
    family="glm",
    description="GLM-5.2 adapter via DeepseekV32",
)
class GLM5_2_EX(DeepseekV32):
    """GLM-5.2 adapter inheriting DeepseekV32.  Uses a translated config.json."""
    def __init__(self, args):
        super().__init__(args)
        self.config = AutoConfig.from_pretrained(os.path.join(CUR_DIR, "config.json"))
        self.model = self.empty_weights_model()
