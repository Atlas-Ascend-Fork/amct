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

__all__ = ["register_solvers", "SOLVER_REGISTRY"]

from amct_pytorch.common.utils.registry_factory import Registry

SOLVER_REGISTRY = Registry("solver")

_REGISTERED = False


def register_solvers():
    global _REGISTERED
    if _REGISTERED:
        return

    from .blockwise_solver import BlockwiseSolver  # noqa: F401

    _REGISTERED = True
