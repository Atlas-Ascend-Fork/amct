# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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

from amct_pytorch.common.utils.model_util import ModuleHelper


def replace_module_aliases(model, old_module, new_module):
    """Replace every registered path that references the same module."""
    alias_names = [
        name
        for name, module in model.named_modules(remove_duplicate=False)
        if name and module is old_module
    ]
    for name in alias_names:
        ModuleHelper.replace_module_by_name(model, name, new_module)
