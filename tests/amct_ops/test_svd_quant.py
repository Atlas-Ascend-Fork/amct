# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import unittest
import torch
import torch_npu
import numpy as np

from amct_ops import svd_quant  # noqa: F401 — Lazy loading of SvdQuant extension
from .quantize_ref import py_quantize_mx4, py_dequantize_mx4

DEVICE_ID = 0
torch_npu.npu.set_device(int(DEVICE_ID))

SVD_QUANT_TEST_CASES = [
    # Prefill, Rank: 32
    (1, 32 * 1024, 10944, 2048, 32),
    # Prefill, Rank: 128
    (1, 32 * 1024, 10944, 2048, 128),
    (1, 32 * 1024, 2816, 2048, 128),
    (1, 32 * 1024, 1408, 2048, 128),
    (1, 32 * 1024, 2048, 10944, 128),
    (1, 32 * 1024, 2048, 2816, 128),
    (1, 32 * 1024, 2048, 1408, 128),
    (1, 32 * 1024, 2048, 2048, 128),
    (1, 32 * 1024, 2048, 3072, 128),
    (1, 32 * 1024, 512, 4096, 128),
    (4, 32 * 1024, 10944, 2048, 128),
    (4, 32 * 1024, 2816, 2048, 128),
    (4, 32 * 1024, 1408, 2048, 128),
    (4, 32 * 1024, 2048, 10944, 128),
    (4, 32 * 1024, 2048, 2816, 128),
    (4, 32 * 1024, 2048, 1408, 128),
    (4, 32 * 1024, 2048, 2048, 128),
    (4, 32 * 1024, 2048, 3072, 128),
    (4, 32 * 1024, 512, 4096, 128),
    # Decode, Rank: 128
    (1, 1, 10944, 2048, 128),
    (1, 1, 2816, 2048, 128),
    (1, 1, 1408, 2048, 128),
    (1, 1, 2048, 10944, 128),
    (1, 1, 2048, 2816, 128),
    (1, 1, 2048, 1408, 128),
    (1, 1, 2048, 2048, 128),
    (1, 1, 2048, 3072, 128),
    (1, 1, 512, 4096, 128),
    (4, 1, 10944, 2048, 128),
    (4, 1, 2816, 2048, 128),
    (4, 1, 1408, 2048, 128),
    (4, 1, 2048, 10944, 128),
    (4, 1, 2048, 2816, 128),
    (4, 1, 2048, 1408, 128),
    (4, 1, 2048, 2048, 128),
    (4, 1, 2048, 3072, 128),
    (4, 1, 512, 4096, 128),
    # Prefill, Rank: 64
    (1, 32 * 1024, 10944, 2048, 64),
    (1, 32 * 1024, 2816, 2048, 64),
    (1, 32 * 1024, 1408, 2048, 64),
    (1, 32 * 1024, 2048, 10944, 64),
    (1, 32 * 1024, 2048, 2816, 64),
    (1, 32 * 1024, 2048, 1408, 64),
    (1, 32 * 1024, 2048, 2048, 64),
    (1, 32 * 1024, 2048, 3072, 64),
    (1, 32 * 1024, 512, 4096, 64),
    (4, 32 * 1024, 10944, 2048, 64),
    (4, 32 * 1024, 2816, 2048, 64),
    (4, 32 * 1024, 1408, 2048, 64),
    (4, 32 * 1024, 2048, 10944, 64),
    (4, 32 * 1024, 2048, 2816, 64),
    (4, 32 * 1024, 2048, 1408, 64),
    (4, 32 * 1024, 2048, 2048, 64),
    (4, 32 * 1024, 2048, 3072, 64),
    (4, 32 * 1024, 512, 4096, 64),
    # Decode, Rank: 64
    (1, 1, 10944, 2048, 64),
    (1, 1, 2816, 2048, 64),
    (1, 1, 1408, 2048, 64),
    (1, 1, 2048, 10944, 64),
    (1, 1, 2048, 2816, 64),
    (1, 1, 2048, 1408, 64),
    (1, 1, 2048, 2048, 64),
    (1, 1, 2048, 3072, 64),
    (1, 1, 512, 4096, 64),
    (4, 1, 10944, 2048, 64),
    (4, 1, 2816, 2048, 64),
    (4, 1, 1408, 2048, 64),
    (4, 1, 2048, 10944, 64),
    (4, 1, 2048, 2816, 64),
    (4, 1, 2048, 1408, 64),
    (4, 1, 2048, 2048, 64),
    (4, 1, 2048, 3072, 64),
    (4, 1, 512, 4096, 64),
]


def quantize(data):
    data_q = py_quantize_mx4(data.float(), group_size=32, rounding_mode=1)
    data_deq = py_dequantize_mx4(data_q, group_size=32)
    return data_deq.to(torch.bfloat16)


class GoldenSubGraph(torch.nn.Module):
    def __init__(self, w, dp, up):
        super(GoldenSubGraph, self).__init__()
        self.down_proj = torch.nn.Linear(dp.shape[0], dp.shape[1], bias=False)
        self.down_proj.weight.data = dp.transpose(1, 0)
        self.up_proj = torch.nn.Linear(up.shape[0], up.shape[1], bias=False)
        self.up_proj.weight.data = up.transpose(1, 0)
        self.mx_w = torch.nn.Linear(w.shape[1], w.shape[0], bias=False)
        self.mx_w.weight.data = w

    def forward(self, x, x_deq):
        dp_o = self.down_proj(x)
        up_o = self.up_proj(dp_o)
        mx_o = self.mx_w(x_deq)
        return up_o + mx_o


class TestSvdQuant(unittest.TestCase):
    def test_svd_quant(self):
        for shape in SVD_QUANT_TEST_CASES:
            test_name = f"{shape}"
            with self.subTest(msg=test_name, shape=shape):
                self.__run_svd_quant(shape)

    def __run_svd_quant(self, shape):
        bs, seq_len, n, k, rank = shape
        a_shape = (bs, seq_len, k)
        w_shape = (n, k)
        dp_shape = (k, rank)
        up_shape = (rank, n)

        np.random.seed(0)

        x = torch.tensor(np.random.uniform(-10, 10, a_shape), dtype=torch.bfloat16)
        w = torch.tensor(np.random.uniform(-10, 10, w_shape), dtype=torch.bfloat16)
        dp = torch.tensor(
            np.random.uniform(-10, 10, dp_shape), dtype=torch.bfloat16
        ).npu()
        up = torch.tensor(
            np.random.uniform(-10, 10, up_shape), dtype=torch.bfloat16
        ).npu()

        # Golden Graph
        x_deq = quantize(x)
        w_deq = quantize(w)
        golden_model = GoldenSubGraph(w_deq.npu(), dp, up)
        golden_out = golden_model(x.npu(), x_deq.npu())
        np_golden_out = golden_out.cpu().float().detach().numpy()

        # SVDQuant Graph
        w_quant, scale = torch_npu.npu_dynamic_mx_quant(
            w.npu(), block_size=32, round_mode="round"
        )
        svd_quant_out = torch.ops.amct.svd_quant(x.npu(), w_quant, scale, dp, up)
        np_svd_out = svd_quant_out.cpu().float().detach().numpy()
        result = np.allclose(np_svd_out, np_golden_out, atol=1e-2, rtol=1e-02)
        self.assertTrue(result)


if __name__ == '__main__':
    unittest.main()
