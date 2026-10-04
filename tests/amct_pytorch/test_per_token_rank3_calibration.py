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
import unittest

import torch
from torch import nn

from amct_pytorch.classic.quantize_op.minmax_module import MinMaxQuant
from amct_pytorch.classic.quantize_op.quantile_module import QuantileQuant
from amct_pytorch.classic.quantize_op.smooth_quant_module import SmoothQuant
from amct_pytorch.classic.quantize_op.utils import calculate_scale_offset


class TestPerTokenRankThreeCalibration(unittest.TestCase):
    def setUp(self):
        self.inputs = torch.tensor(
            [
                [
                    [0.10, -0.12, 0.08, -0.09],
                    [0.15, -0.17, 0.12, -0.13],
                    [0.06, -0.08, 0.09, -0.07],
                ],
                [
                    [100.0, -80.0, 60.0, -40.0],
                    [70.0, -90.0, 50.0, -60.0],
                    [90.0, -60.0, 80.0, -70.0],
                ],
            ],
            dtype=torch.float32,
        )

    def _module(self, module_type, act_type, weight_type, algorithm=None):
        linear = nn.Linear(4, 3, bias=False)
        linear.weight.data.fill_(1.0)
        config = {
            "batch_num": 1,
            "inputs_cfg": {
                "enable_quant": True,
                "quant_type": act_type,
                "symmetric": True,
                "strategy": "token",
                "dynamic": False,
            },
            "weights_cfg": {
                "quant_type": weight_type,
                "symmetric": True,
                "strategy": "channel",
                "group_size": None,
            },
            "algorithm": algorithm or {},
        }
        return module_type(linear, "linear", config)

    def _assert_rank2_control(self, module_type, act_type, weight_type, algorithm=None):
        module = self._module(module_type, act_type, weight_type, algorithm)
        inputs = self.inputs.reshape(-1, self.inputs.shape[-1])
        module(inputs)
        self.assertEqual(tuple(module.scale_d.shape), (inputs.shape[0], 1))

    def test_minmax_keeps_batch_and_sequence_token_axes(self):
        module = self._module(MinMaxQuant, "int8", "int8")
        module(self.inputs)
        expected, _ = calculate_scale_offset(
            self.inputs.amax(dim=-1, keepdim=True),
            self.inputs.amin(dim=-1, keepdim=True),
            True,
            "int8",
        )
        self.assertEqual(tuple(module.scale_d.shape), (2, 3, 1))
        torch.testing.assert_close(module.scale_d, expected)
        self._assert_rank2_control(MinMaxQuant, "int8", "int8")

    def test_quantile_keeps_batch_and_sequence_token_axes(self):
        module = self._module(QuantileQuant, "hifloat8", "hifloat8")
        module(self.inputs)
        expected = module.calculate_hif8_scale(
            self.inputs.abs().amax(dim=-1, keepdim=True)
        )
        self.assertEqual(tuple(module.scale_d.shape), (2, 3, 1))
        torch.testing.assert_close(module.scale_d, expected)
        self._assert_rank2_control(QuantileQuant, "hifloat8", "hifloat8")

    def test_smoothquant_keeps_batch_and_sequence_token_axes(self):
        algorithm = {"smoothquant": {"smooth_strength": 0.5}}
        module = self._module(SmoothQuant, "int8", "int8", algorithm)
        module(self.inputs)
        smooth_input = module.batch_input / module.scale.cpu().repeat(
            1, module.batch_num
        )
        expected, _ = calculate_scale_offset(
            smooth_input.amax(dim=-1, keepdim=True),
            smooth_input.amin(dim=-1, keepdim=True),
            True,
            "int8",
        )
        self.assertEqual(tuple(module.scale_d.shape), (2, 3, 1))
        torch.testing.assert_close(module.scale_d, expected)
        self._assert_rank2_control(SmoothQuant, "int8", "int8", algorithm)
