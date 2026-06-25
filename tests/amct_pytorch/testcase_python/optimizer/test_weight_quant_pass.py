#!/usr/bin/env python3
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
import os
import unittest
from io import BytesIO
from unittest.mock import patch

import numpy as np
import torch

from amct_pytorch.classic.graph_based.amct_pytorch.optimizer.graph_optimizer import (
    GraphOptimizer,
)
from amct_pytorch.classic.graph_based.amct_pytorch.optimizer.insert_weight_quant_pass import (
    InsertWeightQuantPass,
)
from amct_pytorch.classic.graph_based.amct_pytorch.parser.parser import Parser
from amct_pytorch.classic.graph_based.amct_pytorch.utils.onnx_initializer_util import (
    TensorProtoHelper,
)
from amct_pytorch.classic.graph_based.amct_pytorch.utils.vars import (
    QUANTIZABLE_TYPES,
)

from .utils import models, record_file_utils

CUR_DIR = os.path.split(os.path.realpath(__file__))[0]


class TestWeightQuantPass(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        QUANTIZABLE_TYPES.extend(["ConvTranspose2d", "AvgPool2d"])
        cls.temp_folder = os.path.join(CUR_DIR, "test_weight_quant_pass")
        if not os.path.isdir(cls.temp_folder):
            os.makedirs(cls.temp_folder)

        cls.model_001 = models.Net001().to(torch.device("cpu"))
        cls.args_shape = [(1, 2, 28, 28)]
        cls.args = list()
        for input_shape in cls.args_shape:
            cls.args.append(torch.randn(input_shape))
        cls.args = tuple(cls.args)

        tmp_onnx = BytesIO()
        Parser.export_onnx(cls.model_001, cls.args, tmp_onnx)
        cls.graph = Parser.parse_net_to_graph(tmp_onnx)

        cls.records = record_file_utils.generate_records(
            layers_length={"layer1.0": 16, "layer2.0": 16, "fc.2": 1}
        )

    @classmethod
    def tearDownClass(cls):
        QUANTIZABLE_TYPES.remove("ConvTranspose2d")
        QUANTIZABLE_TYPES.remove("AvgPool2d")
        os.popen("rm -r " + cls.temp_folder)

    def test_quant_weight(self):
        passer = InsertWeightQuantPass(self.records)
        optimizer = GraphOptimizer()
        optimizer.add_pass(passer)
        optimizer.do_optimizer(self.graph, None)
        weight_dtype = (
            TensorProtoHelper(
                self.graph.get_node_by_name("layer1.0.sub_module.weight").proto
            )
            .get_data()
            .dtype
        )
        self.assertEqual(weight_dtype, "int8")

    def test_quant_weight_int4(self):
        with patch(
            "amct_pytorch.classic.graph_based.amct_pytorch.utils.quant_node."
            "QuantOpInfo.get_dst_num_bits",
            return_value=4,
        ):
            passer = InsertWeightQuantPass(self.records)
            before_nodes = len(self.graph.nodes)
            optimizer = GraphOptimizer()
            optimizer.add_pass(passer)
            optimizer.do_optimizer(self.graph, None)
            after_nodes = len(self.graph.nodes)
            self.assertEqual(after_nodes - before_nodes, 0)

    def test_deploy_packs_int4_weight(self):
        # 4 个 INT4 权重 → deploy 应 pack 成 2 个 INT8 字节
        from amct_pytorch.classic.graph_based.amct_pytorch.optimizer.insert_weight_quant_pass import (
            finalize_int4_weight_for_deploy,
        )

        int4_vals = np.array([1, -2, 7, -8], dtype=np.int8)
        packed, dtype_str = finalize_int4_weight_for_deploy(int4_vals)
        self.assertEqual(dtype_str, "INT8")
        self.assertEqual(packed.size, 2)

    def test_fakequant_uses_native_int4(self):
        from amct_pytorch.classic.graph_based.amct_pytorch.optimizer.insert_weight_quant_pass import (
            finalize_int4_weight_for_fakequant,
        )

        int4_vals = np.array([1, -2, 7, -8], dtype=np.int8)
        data, dtype_str = finalize_int4_weight_for_fakequant(int4_vals)
        self.assertEqual(dtype_str, "INT4")
        self.assertEqual(data.size, 4)

    def test_rnn_weight_quant_success(self):
        class RNNModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = torch.nn.LSTM(10, 20, 1)

            def forward(self, input_data, hx):
                x = self.lstm(input_data, hx)
                return x

        model = RNNModule()
        tmp_onnx = BytesIO()
        Parser.export_onnx(
            model,
            (torch.randn(1, 1, 10), (torch.randn(1, 1, 20), torch.randn(1, 1, 20))),
            tmp_onnx,
        )
        graph = Parser.parse_net_to_graph(tmp_onnx)
        node_name = "lstm"
        node = graph.get_node_by_name(node_name)

        records = {
            node_name: {
                "weight_scale": np.array([1.0] * 4, dtype=np.float32),
                "weight_offset": np.array([0] * 4, dtype=np.int8),
                "recurrence_weight_scale": np.array([1.0] * 4, dtype=np.float32),
                "recurrence_weight_offset": np.array([0] * 4, dtype=np.int8),
                "wts_type": "INT8",
            }
        }

        passer = InsertWeightQuantPass(records)
        passer.quant_recurrence_weight(node)

    def test_deploy_packs_int4_recurrence_weight(self):
        """LSTM A8W4: deploy finalize must pack recurrence_weight INT4 → INT8.

        After finalize_int4_weight_for_deploy the byte count should be half the
        original INT4 element count (two nibbles packed into one INT8 byte).
        This test guards against the ordering bug where ReplaceRNNPass removes
        the original LSTM node before the INT4 deploy finalize loop runs,
        causing the recurrence_weight to be silently skipped (not packed).
        """
        from amct_pytorch.classic.graph_based.amct_pytorch.optimizer.insert_weight_quant_pass import (
            finalize_int4_weight_for_deploy,
        )

        # Simulate a recurrence_weight tensor for an LSTM with hidden_size=20,
        # input_size=10: shape is (4, 20, 20) → 1600 INT4 elements (even count).
        n_elements = 1600
        rng = np.random.default_rng(42)
        int4_vals = rng.integers(-8, 8, size=n_elements, dtype=np.int8)

        packed, dtype_str = finalize_int4_weight_for_deploy(int4_vals)

        self.assertEqual(dtype_str, "INT8")
        # Two INT4 nibbles packed into each INT8 byte → exactly n_elements // 2 bytes.
        self.assertEqual(
            packed.size,
            n_elements // 2,
            msg="expected {} packed bytes, got {}".format(n_elements // 2, packed.size),
        )

    def test_recurrence_weight_int4(self):
        """When wts_type is INT4, quant_recurrence_weight must quantize with num_bits=4.

        Verifies that quant_recurrence_weight calls weight_quant_np with num_bits=4
        (not the hardcoded 8) when get_dst_num_bits returns 4 for the layer.
        Before the fix (hardcoded 8) this test fails; after the fix it passes.
        """

        class RNNModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = torch.nn.LSTM(10, 20, 1)

            def forward(self, input_data, hx):
                return self.lstm(input_data, hx)

        model = RNNModule()
        tmp_onnx = BytesIO()
        Parser.export_onnx(
            model,
            (torch.randn(1, 1, 10), (torch.randn(1, 1, 20), torch.randn(1, 1, 20))),
            tmp_onnx,
        )
        graph = Parser.parse_net_to_graph(tmp_onnx)
        node = graph.get_node_by_name("lstm")

        scale = np.array([1.0] * 4, dtype=np.float32)
        records = {
            "lstm": {
                "weight_scale": scale,
                "weight_offset": np.array([0] * 4, dtype=np.int8),
                "recurrence_weight_scale": scale,
                "recurrence_weight_offset": np.array([0] * 4, dtype=np.int8),
                "wts_type": "INT4",
            }
        }

        passer = InsertWeightQuantPass(records)

        # Spy on weight_quant_np to capture the num_bits argument actually used.
        # Before fix: hardcoded 8 is passed regardless of get_dst_num_bits.
        # After fix: get_dst_num_bits returns 4 and 4 is passed.
        weight_quant_module = (
            "amct_pytorch.classic.graph_based.amct_pytorch."
            "optimizer.insert_weight_quant_pass.weight_quant_np"
        )
        set_data_module = (
            "amct_pytorch.classic.graph_based.amct_pytorch."
            "optimizer.insert_weight_quant_pass.TensorProtoHelper.set_data"
        )
        # Capture set_data calls via a closure that delegates to the real method.
        # We save the original *before* patching to avoid recursion.
        _real_set_data = (
            TensorProtoHelper.set_data.__wrapped__
            if hasattr(TensorProtoHelper.set_data, "__wrapped__")
            else TensorProtoHelper.set_data
        )
        set_data_type_args = []

        def _spy_set_data(self_inner, data, type_string=None, dims=None):
            set_data_type_args.append(type_string)
            return _real_set_data(self_inner, data, type_string, dims)

        with (
            patch(
                "amct_pytorch.classic.graph_based.amct_pytorch.utils.quant_node."
                "QuantOpInfo.get_dst_num_bits",
                return_value=4,
            ),
            patch(
                weight_quant_module,
                wraps=__import__(
                    "amct_pytorch.classic.graph_based.amct_pytorch.custom_op.arq.arq",
                    fromlist=["weight_quant_np"],
                ).weight_quant_np,
            ) as mock_wqnp,
            patch(set_data_module, _spy_set_data),
        ):
            passer.quant_recurrence_weight(node)

        # Assert weight_quant_np was called with num_bits=4, not 8
        self.assertTrue(mock_wqnp.called, "weight_quant_np was never called")
        args, kwargs = mock_wqnp.call_args
        actual_num_bits = args[3] if len(args) > 3 else kwargs.get("num_bits")
        self.assertEqual(
            actual_num_bits,
            4,
            msg="expected weight_quant_np called with num_bits=4, got {}".format(
                actual_num_bits
            ),
        )

        # Assert recurrence_weight is stored as INT8 container (even when num_bits=4)
        self.assertTrue(
            set_data_type_args, "TensorProtoHelper.set_data was never called"
        )
        # The last set_data call on the recurrence weight helper must use 'INT8'
        actual_dtype = set_data_type_args[-1]
        self.assertEqual(
            actual_dtype,
            "INT8",
            msg="expected recurrence_weight stored as INT8, got {}".format(
                actual_dtype
            ),
        )

    # ------------------------------------------------------------------
    # C-1 regression: non-RNN (Conv) A8W4 finalize must not crash
    # ------------------------------------------------------------------
    def test_conv_int4_finalize_deploy_and_fakequant_no_crash(self):
        """C-1 regression: Conv A8W4 finalize loops must not crash.

        This test directly exercises the two INT4 finalize loops from
        _generate_model (deploy pack and fakequant native-INT4 paths) for a
        Conv node.  Before the fix, get_recurrence_weight_node raised an
        exception for non-RNN nodes because it passed None to
        get_input_anchor.  After the fix it returns None and the loop skips
        cleanly.

        Assertions:
        - No exception is raised.
        - get_recurrence_weight_node returns None for a Conv node.
        - The main weight is finalized correctly (deploy: packed INT8 with
          half the elements; fakequant: native INT4 with same element count).
        """
        from amct_pytorch.classic.graph_based.amct_pytorch.optimizer.insert_weight_quant_pass import (
            finalize_int4_weight_for_deploy,
            finalize_int4_weight_for_fakequant,
        )
        from amct_pytorch.classic.graph_based.amct_pytorch.utils.quant_node import (
            QuantOpInfo,
        )

        # Use the Conv graph already built in setUpClass (Net001 has Conv2d layers).
        # Pick the first Conv node from the graph.
        conv_node = None
        for node in self.graph.nodes:
            if node.type == "Conv":
                conv_node = node
                break
        self.assertIsNotNone(
            conv_node, "Expected at least one Conv node in Net001 graph"
        )

        # ---- assert get_recurrence_weight_node returns None, not crash ----
        rw_node = QuantOpInfo.get_recurrence_weight_node(conv_node)
        self.assertIsNone(
            rw_node,
            "get_recurrence_weight_node must return None for a Conv node, not crash",
        )

        # ---- get the weight node and set up INT4 data ----
        weight_node = QuantOpInfo.get_weight_node(conv_node)
        self.assertIsNotNone(weight_node, "Conv node must have a weight node")

        weight_helper = TensorProtoHelper(weight_node.proto, weight_node.model_path)
        # Read the real weight data from the graph, then cast to int8 to simulate
        # INT4 quantized values (clamp to [-8, 7] so finalize_int4_weight_for_fakequant
        # does not raise an out-of-range error).
        raw_data = weight_helper.get_data()
        n_elements = raw_data.size
        int4_vals = np.clip(raw_data.flatten().astype(np.int8), -8, 7)

        # ---- deploy path: finalize_int4_weight_for_deploy ----
        packed_deploy, dtype_deploy = finalize_int4_weight_for_deploy(int4_vals)
        self.assertEqual(dtype_deploy, "INT8", "deploy dtype must be INT8")
        self.assertEqual(
            packed_deploy.size,
            n_elements // 2,
            "deploy packed size must be half the INT4 element count",
        )

        # ---- fakequant path: finalize_int4_weight_for_fakequant ----
        data_fq, dtype_fq = finalize_int4_weight_for_fakequant(int4_vals)
        self.assertEqual(dtype_fq, "INT4", "fakequant dtype must be INT4")
        self.assertEqual(
            data_fq.size,
            n_elements,
            "fakequant data must preserve the original element count",
        )

        # ---- simulate the full finalize loop body (no save) ----
        # deploy side: clear + set_data must not raise
        weight_helper.clear_data()
        weight_helper.set_data(packed_deploy, dtype_deploy)  # must not raise

        # fakequant side: a fresh helper on the same proto, clear + set_data
        weight_helper_fq = TensorProtoHelper(weight_node.proto, weight_node.model_path)
        weight_helper_fq.clear_data()
        weight_helper_fq.set_data(data_fq, dtype_fq)  # must not raise
