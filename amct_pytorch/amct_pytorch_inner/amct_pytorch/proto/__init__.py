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
"""

from . import calibration_config_pytorch_pb2 as calibration_config_pb2 # pylint: disable=E0611, W0406
from . import retrain_config_pytorch_pb2 as retrain_config_pb2 # pylint: disable=E0611, W0406
from . import distill_config_pytorch_pb2 as distill_config_pb2
from . import scale_offset_record_pytorch_pb2 as scale_offset_record_pb2 # pylint: disable=E0611, W0406
from . import quant_calibration_config_pytorch_pb2 as quant_calibration_config_pb2
from . import post_quant_config_pytorch_pb2 as post_quant_config_pb2
from . import post_quant_record_pytorch_pb2 as post_quant_record_pb2