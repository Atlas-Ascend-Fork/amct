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
        # Class name -> quantize module class, so a quant result manifest that
        # only records ``quant_module_type`` as a string can be resolved back to
        # the class needed to rebuild the module. Registered classes are the
        # only trusted source: reload never imports a name from the artifact.
        self.name_to_quant_op = {}

    def register(self, name, src_op, quant_op, deploy_op):
        # Checked before any registry is touched: a conflicting registration
        # must not leave the registries half updated.
        self._check_quant_op_name(quant_op)

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
        self._record_quant_op_name(quant_op)
        LOGGER.logd(f'register algorithm {name} success.')

    def _check_quant_op_name(self, quant_op):
        """Reject two different classes claiming the same ``__name__``.

        Re-registering the very same class is normal: one class is registered for
        several algorithm/source-op pairs. Two distinct classes sharing a name
        would make manifest lookup ambiguous, so that is refused outright.
        """
        if not isinstance(quant_op, type):
            return
        quant_name = quant_op.__name__
        existing = self.name_to_quant_op.get(quant_name)
        if existing is not None and existing is not quant_op:
            raise ValueError(
                "quantize module name '{}' is already registered to {}.{}; "
                "refusing to register {}.{} under the same name because quant "
                "result reload resolves modules by class name.".format(
                    quant_name,
                    existing.__module__,
                    existing.__qualname__,
                    quant_op.__module__,
                    quant_op.__qualname__,
                )
            )

    def _record_quant_op_name(self, quant_op):
        """Index a registered quantize module class by its ``__name__``.

        Some entries register a *string* placeholder instead of a class (a module
        resolved elsewhere at deploy time); those have no class to rebuild from,
        so they are skipped rather than indexed.
        """
        if not isinstance(quant_op, type):
            LOGGER.logd(
                f'quant_op {quant_op!r} is not a class, '
                'skipped in the quant module name index.'
            )
            return
        self.name_to_quant_op[quant_op.__name__] = quant_op

    def get_quant_op_by_name(self, quant_module_type):
        """Resolve a registered quantize module class by class name.

        Raises rather than returning ``None``: an unknown name in a manifest
        means the artifact cannot be rebuilt in this installation.
        """
        if not isinstance(quant_module_type, str) or not quant_module_type:
            raise ValueError(
                f'quant_module_type must be a non-empty str, got {quant_module_type!r}.'
            )
        quant_op = self.name_to_quant_op.get(quant_module_type)
        if quant_op is None:
            raise ValueError(
                f"quantize module '{quant_module_type}' is not registered in this "
                'installation; known modules are '
                f'{sorted(self.name_to_quant_op)}.'
            )
        return quant_op
