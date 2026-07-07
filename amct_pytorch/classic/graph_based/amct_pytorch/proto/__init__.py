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

from . import calibration_config_pytorch_pb2 as calibration_config_pb2 # pylint: disable=E0611, W0406
from . import retrain_config_pytorch_pb2 as retrain_config_pb2 # pylint: disable=E0611, W0406
from . import distill_config_pytorch_pb2 as distill_config_pb2
from . import scale_offset_record_pytorch_pb2 as scale_offset_record_pb2 # pylint: disable=E0611, W0406
from . import quant_calibration_config_pytorch_pb2 as quant_calibration_config_pb2