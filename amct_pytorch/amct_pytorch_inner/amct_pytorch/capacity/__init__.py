#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

init capacity.

"""

import os
from amct_pytorch.common.capacity.query_capacity import Capacity

__all__ = ['CAPACITY', 'POST_CAPACITY']

_CUR_DIR = os.path.split(os.path.realpath(__file__))[0]
CAPACITY = Capacity(os.path.join(_CUR_DIR, 'capacity_config.csv'))
POST_CAPACITY = Capacity(os.path.join(_CUR_DIR, 'capacity_post_config.csv'))
