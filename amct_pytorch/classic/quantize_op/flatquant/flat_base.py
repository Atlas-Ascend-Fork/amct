# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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
#
# The code snippet comes from FlatQuant (https://github.com/ruikangliu/FlatQuant).
#
# Copyright (c) 2024 ruikangliu. All rights reserved.
#
# This file is part of FlatQuant, which is released under the MIT License.
# See the LICENSE file in the root directory of this source tree
# or at https://opensource.org/licenses/MIT for details.
#
# --------------------------------------------------------------------------------
"""Shared base for the FlatQuant reconstruction modules.

Carries the module contract consumed by the generic block-calibration
framework.

- ``quant_config``:      the layer's resolved detail config dict. The
                         framework's matcher requires a truthy value so
                         non-quantized wrappers holding nn.Parameter are
                         never mistaken for gradient-calibratable modules.
- ``trainable_params()``: the learnable terms this module contributes to
                         the block training (transforms / diag scales /
                         clip factors). Non-empty is what routes the module
                         to the 'gradient' block calibrator.
- ``block_trainer``:    the class-level training loop entry declared to the
                         gradient block calibrator (contract documented in
                         amct_pytorch.common.block_calibrator).

Subclasses keep the model-specific parts (forward, set_fake_quant,
init_diag_scale, reparameterize, get_external_reparam)."""

import torch.nn as nn

from .cali_flatquant import train_block


class FlatQuantModuleBase(nn.Module):
    """Base class for FlatQuantQwen3Attention/MLP and CLIP variants."""

    #: gradient-kernel contract: the per-block training loop this
    #: algorithm contributes. staticmethod so every subclass shares one
    #: function object (the kernel requires a single distinct trainer
    #: across a block); the loop itself stays algorithm-private.
    block_trainer = staticmethod(train_block)

    def __init__(self, module=None, layer_name='', layer_config=None):
        super().__init__()
        # module contract: consumed by _is_gradient_module and the gradient
        # block calibrator; truthy whenever the module was built by quantize()
        self.quant_config = layer_config if layer_config is not None else {}
        self.layer_name = layer_name

    def trainable_params(self):
        """Learnable parameters owned by this reconstruction module.

        Called by the framework matcher (must stay non-empty for gradient
        dispatch) and by the training loop when assembling param groups.
        """
        return list(self.parameters())

    def set_fake_quant(self, enabled):
        """Enable/disable fake-quant training; each subclass implements."""
        raise NotImplementedError

    def init_diag_scale(self, alpha=0.5):
        """Init diag scales from weight/activation stats; optional."""
        pass

    def reparameterize(self):
        """Convert-time solidification; each subclass implements."""
        raise NotImplementedError

    def get_external_reparam(self, model=None, name=''):
        """Declare out-of-module convert side effects."""
        return []
