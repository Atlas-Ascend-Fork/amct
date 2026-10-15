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


class ModuleAliasIndex:
    """Index registered module paths and keep them current during replacements."""

    def __init__(self, model):
        self.named_module_dict = {}
        self.alias_names_by_id = {}
        self._module_by_name = {}

        seen_module_ids = set()
        for name, module in model.named_modules(remove_duplicate=False):
            if name:
                self._add(name, module)
            module_id = id(module)
            if module_id not in seen_module_ids:
                self.named_module_dict[name] = module
                seen_module_ids.add(module_id)

    def _add(self, name, module):
        self._module_by_name[name] = module
        self.alias_names_by_id.setdefault(id(module), set()).add(name)

    def _remove(self, name):
        module = self._module_by_name.pop(name, None)
        if module is None:
            return

        alias_names = self.alias_names_by_id.get(id(module))
        if alias_names is None:
            return
        alias_names.discard(name)
        if not alias_names:
            del self.alias_names_by_id[id(module)]

    def replace(self, model, old_module, new_module):
        """Replace all current aliases and refresh paths under those aliases."""
        alias_names = sorted(self.alias_names_by_id.get(id(old_module), ()))
        if not alias_names:
            return

        old_children = [
            (name, module)
            for name, module in old_module.named_modules(remove_duplicate=False)
            if name
        ]
        new_submodules = list(new_module.named_modules(remove_duplicate=False))

        for alias_name in alias_names:
            self._remove(alias_name)
            for child_name, _ in old_children:
                self._remove(f"{alias_name}.{child_name}")

        for alias_name in alias_names:
            ModuleHelper.replace_module_by_name(model, alias_name, new_module)

        for alias_name in alias_names:
            self._add(alias_name, new_module)
            for child_name, module in new_submodules:
                if child_name:
                    self._add(f"{alias_name}.{child_name}", module)


def replace_module_aliases(model, old_module, new_module, alias_index=None):
    """Replace every registered path that references the same module."""
    if alias_index is None:
        alias_index = ModuleAliasIndex(model)
    alias_index.replace(model, old_module, new_module)
