#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
import unittest

import torch

from amct_pytorch.classic.graph_based.amct_pytorch.custom_op.cali_quant import (
    CaliQuantBase,
)


class CaliQuantForTest(CaliQuantBase):
    """Capture the hidden-state calibration input without running IFMR/HFMG."""

    def cali_process(self, inputs, hx=None):
        self.hidden_inputs = hx


class TestCaliQuantRNN(unittest.TestCase):
    def _test_hidden_inputs(self, rnn_type, batch_first):
        input_size = 3
        hidden_size = 5
        sequence_length = 4
        batch_size = 2
        module = rnn_type(
            input_size,
            hidden_size,
            num_layers=1,
            batch_first=batch_first,
        )
        cali_quant = CaliQuantForTest(
            module,
            record_module=None,
            layers_name=['rnn'],
            batch_num=1,
            mode='cali',
        )

        if batch_first:
            inputs = torch.randn(batch_size, sequence_length, input_size)
        else:
            inputs = torch.randn(sequence_length, batch_size, input_size)
        h0 = torch.randn(1, batch_size, hidden_size)
        hx = (h0, torch.randn_like(h0)) if rnn_type is torch.nn.LSTM else h0

        outputs = cali_quant(inputs, hx)
        hidden_inputs = cali_quant.hidden_inputs
        if batch_first:
            torch.testing.assert_close(hidden_inputs[:, :1, :], h0.permute(1, 0, 2))
            torch.testing.assert_close(hidden_inputs[:, 1:, :], outputs[0][:, :-1, :])
        else:
            torch.testing.assert_close(hidden_inputs[:1], h0)
            torch.testing.assert_close(hidden_inputs[1:], outputs[0][:-1])
        self.assertEqual(
            tuple(hidden_inputs.shape),
            tuple(inputs.shape[:-1]) + (hidden_size,),
        )

    def test_lstm_hidden_inputs(self):
        for batch_first in (False, True):
            with self.subTest(batch_first=batch_first):
                self._test_hidden_inputs(torch.nn.LSTM, batch_first)

    def test_gru_hidden_inputs(self):
        for batch_first in (False, True):
            with self.subTest(batch_first=batch_first):
                self._test_hidden_inputs(torch.nn.GRU, batch_first)


if __name__ == '__main__':
    unittest.main()
