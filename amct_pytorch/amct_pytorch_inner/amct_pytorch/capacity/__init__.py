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

import os
from amct_pytorch.amct_pytorch_inner.amct_pytorch.common.capacity.query_capacity import Capacity

__all__ = ['CAPACITY', 'POST_CAPACITY']

_CUR_DIR = os.path.split(os.path.realpath(__file__))[0]
CAPACITY = Capacity(os.path.join(_CUR_DIR, 'capacity_config.csv'))
POST_CAPACITY = Capacity(os.path.join(_CUR_DIR, 'capacity_post_config.csv'))
