# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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


from amct_pytorch.utils.log import LOGGER


class Algorithm:
    def __init__(self):
        self.algo = dict()
        self.quant_to_deploy = dict()
        self.quant_op = []

    def register(self, name, src_op, quant_op, deploy_op):
        if self.algo.get(name) is None:
            self.algo[name] = {}
        self.algo[name][src_op] = quant_op

        if self.quant_to_deploy.get(quant_op) is None:
            if isinstance(deploy_op, list):
                self.quant_to_deploy[quant_op] = deploy_op
            else:
                self.quant_to_deploy[quant_op] = []

        if isinstance(deploy_op, list):
            self.quant_to_deploy[quant_op] += deploy_op
        else:
            self.quant_to_deploy[quant_op].append(deploy_op)
        self.quant_to_deploy[quant_op] = list(set(self.quant_to_deploy[quant_op]))
        LOGGER.logd(f'register algorithm {name} success.')