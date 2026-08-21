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


from amct_pytorch.common.utils.log import LOGGER


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

    def get_quant_op_by_name(self, quant_module_type):
        """Resolve a registered quantize module class by class name.

        Raises rather than returning ``None``: an unknown name in a manifest
        means the artifact cannot be rebuilt in this installation.
        """
        if not isinstance(quant_module_type, str) or not quant_module_type:
            raise ValueError(
                f'quant_module_type must be a non-empty str, got {quant_module_type!r}.'
            )

        known_modules = set()
        matches = []
        for algorithm in self.algo.values():
            for quant_op in algorithm.values():
                if not isinstance(quant_op, type):
                    continue
                known_modules.add(quant_op.__name__)
                if quant_op.__name__ != quant_module_type:
                    continue
                if all(registered is not quant_op for registered in matches):
                    matches.append(quant_op)

        if not matches:
            raise ValueError(
                f"quantize module '{quant_module_type}' is not registered in this "
                'installation; known modules are '
                f'{sorted(known_modules)}.'
            )
        if len(matches) > 1:
            candidates = [
                f'{quant_op.__module__}.{quant_op.__qualname__}' for quant_op in matches
            ]
            raise ValueError(
                f"quantize module name '{quant_module_type}' is ambiguous; "
                f'registered classes are {candidates}.'
            )
        return matches[0]
