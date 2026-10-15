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
"""Insert shared-input pre-op calibration modules after projection quantization."""

from amct_pytorch.classic.optimizer.base_module_fusion_pass import BaseModuleFusionPass
from amct_pytorch.classic.optimizer.model_structure import build_layer_norm_quant_specs
from amct_pytorch.classic.optimizer.layer_norm_quant import LayerNormQuant
from amct_pytorch.common.utils.log import LOGGER
from amct_pytorch.common.utils.model_util import ModuleHelper


class InsertLayerNormQuantPass(BaseModuleFusionPass):
    """Match whitelisted pre-ops and replace them with calibration modules."""

    def __init__(self, structures=None):
        super().__init__()
        self.structures = structures
        self.specs = {}

    def run(self, model):
        self.specs = {
            spec.pre_op_ref: spec
            for spec in build_layer_norm_quant_specs(model, self.structures)
        }
        super().run(model)

    def match_pattern(self, module, name):
        return name in self.specs

    def do_pass(self, model, object_module, object_name):
        spec = self.specs[object_name]
        module = LayerNormQuant(
            pre_op=object_module,
            scaling_layers=tuple(model.get_submodule(ref) for ref in spec.member_refs),
            member_refs=spec.member_refs,
            target_ref=spec.target_ref,
            adapter=spec.adapter,
            algorithm=spec.algorithm,
            act_granularity=spec.act_granularity,
            batch_num=spec.batch_num,
            dynamic_per_token=spec.dynamic_per_token,
        )
        ModuleHelper.replace_module_by_name(model, object_name, module)
        LOGGER.logi(
            f'Inserted LayerNormQuant {object_name}, '
            f'members={list(spec.member_refs)}, '
            f'dynamic_per_token={spec.dynamic_per_token}',
            'InsertLayerNormQuantPass',
        )
