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

from torch import nn
from ....amct_pytorch.custom_op import dump_forward
from ....amct_pytorch.utils.log import LOGGER

LAYER_NAME = 'layer_name'


class DUMP(nn.Module):
    """
    The class to dump the inputs data. and directly pass the inputs.
    """
    def __init__(self, layers_name, dump_config):
        """
        Function: Init Function.

        Args:
        layers_name: list of string, length 1. The name of dump data's prefix.
        dump_config: DumpConfig class, contains `dump_dir` and `batch_num`.
        """
        super().__init__()
        self.params = {}
        self.params['dump_dir'] = dump_config.dump_dir
        self.params['layer_name'] = layers_name[0]

        self.batch_num = dump_config.batch_num

        self.cur_batch = 0

    def forward(self, inputs):
        """
        Function: DUMP foward funtion.

        Args:
        inputs: tensor. Dump support float and double and int,
                kFloat32/kFloat64/kInt32 in C++.

        Return:
        inputs: tensor, do not process, just dump.
        """
        self.cur_batch += 1

        if self.batch_num == -1 or \
            self.cur_batch <= self.batch_num:

            dump_param = self.params

            name_prefix = "{}_activation".format(dump_param.get(LAYER_NAME))
            status = dump_forward(inputs,
                        dump_param.get('dump_dir'),
                        name_prefix,
                        self.cur_batch)
            if status == 0:
                LOGGER.logi("Do layer [{}] data dump {} / {} succeeded!"
                    .format(self.params.get(LAYER_NAME), self.cur_batch, self.batch_num), 'DUMP')
            elif status == -65519:
                raise RuntimeError("NOT SUPPORT THE DATA TYPE!")
            else:
                raise RuntimeError("Do layer {} data dump {} / {} failed!"
                .format(self.params.get(LAYER_NAME), self.cur_batch, self.batch_num))

        return inputs
