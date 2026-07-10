#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import unittest

import numpy as np
from onnx.onnx_pb import TensorProto

from amct_pytorch.classic.graph_based.amct_pytorch.utils.onnx_initializer_util import (
    TensorProtoHelper,
    pack_int4_to_int8,
    unpack_int8_to_int4,
)


class TestInt4Pack(unittest.TestCase):
    def test_pack_unpack_roundtrip_even(self):
        arr = np.array([1, -2, 7, -8], dtype=np.int8)
        packed = pack_int4_to_int8(arr)
        self.assertEqual(packed.dtype, np.int8)
        self.assertEqual(packed.size, 2)  # 4 个 INT4 → 2 个 INT8
        out = unpack_int8_to_int4(packed, count=4)
        np.testing.assert_array_equal(out, arr)

    def test_pack_odd_length_pads(self):
        arr = np.array([3, -4, 5], dtype=np.int8)
        packed = pack_int4_to_int8(arr)
        self.assertEqual(packed.size, 2)  # 3 个 → 补成 4 → 2 个 INT8
        out = unpack_int8_to_int4(packed, count=3)
        np.testing.assert_array_equal(out, arr)


class TestNativeInt4(unittest.TestCase):
    @unittest.skipUnless(
        hasattr(TensorProto, 'INT4'), 'onnx 版本过旧，不支持原生 INT4'
    )
    def test_set_get_native_int4(self):
        t = TensorProto()
        helper = TensorProtoHelper(t)
        data = np.array([1, -2, 7, -8], dtype=np.int8)
        helper.set_data(data, "INT4", dims=[4])
        self.assertEqual(t.data_type, TensorProto.INT4)
        self.assertEqual(len(t.raw_data), 2)  # 4 个 INT4 → nibble-packed → 2 字节
        out = helper.get_data()
        np.testing.assert_array_equal(out.astype(np.int8), data)


if __name__ == "__main__":
    unittest.main()
