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


class ModuleHelper():
    """
    Funtion: Helper for torch.nn.module
    APIS: get_module, get_parent_module
    """
    def __init__(self, model):
        ''' init function '''
        self.named_module_dict = {}
        for name, mod in model.named_modules():
            self.named_module_dict[name] = mod

    @staticmethod
    def replace_module_by_name(model, name, mod):
        """ replace module in model by a new mod according to module name """
        tokens = name.split('.')
        sub_tokens = tokens[:-1]
        cur_mod = model
        for s in sub_tokens:
            cur_mod = getattr(cur_mod, s)
        setattr(cur_mod, tokens[-1], mod)
