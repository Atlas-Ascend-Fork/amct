#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import os
import tempfile
import unittest
from unittest import mock

import numpy as np
from onnx.onnx_pb import TensorProto

from amct_pytorch.classic.graph_based.amct_pytorch.utils.onnx_initializer_util import (
    TensorProtoHelper,
    pack_int4_to_int8,
    parse_external_data_meta,
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
    @unittest.skipUnless(hasattr(TensorProto, 'INT4'), 'onnx 版本过旧，不支持原生 INT4')
    def test_set_get_native_int4(self):
        t = TensorProto()
        helper = TensorProtoHelper(t)
        data = np.array([1, -2, 7, -8], dtype=np.int8)
        helper.set_data(data, "INT4", dims=[4])
        self.assertEqual(t.data_type, TensorProto.INT4)
        self.assertEqual(len(t.raw_data), 2)  # 4 个 INT4 → nibble-packed → 2 字节
        out = helper.get_data()
        np.testing.assert_array_equal(out.astype(np.int8), data)

    @unittest.skipUnless(hasattr(TensorProto, 'INT4'), 'onnx 版本过旧，不支持原生 INT4')
    def test_external_int4_roundtrip(self):
        # external data 路径下 INT4 需 pack/unpack 对称，且不因 numpy 无 int4 崩溃
        t = TensorProto()
        t.data_type = TensorProto.INT4
        t.dims.extend([4, 2])  # 8 个 INT4
        t.data_location = 1  # EXTERNAL
        data = np.array([1, -2, 7, -8, 3, -4, 0, 5], dtype=np.int8)
        TensorProtoHelper(t).set_external_data(data)
        self.assertEqual(len(t.raw_data), 4)  # 8 个 INT4 → packed → 4 字节
        out, _ = TensorProtoHelper(t).get_external_data()
        self.assertEqual(list(out.shape), [4, 2])
        np.testing.assert_array_equal(out.astype(np.int8).flatten(), data)


class TestParseExternalDataMeta(unittest.TestCase):
    def test_parse_all_fields(self):
        entries = []
        for k, v in [
            ('location', 'w.bin'),
            ('offset', '128'),
            ('length', '256'),
            ('amct_quantized_raw_data', '1'),
        ]:
            e = mock.MagicMock()
            e.key, e.value = k, v
            entries.append(e)
        file_name, offset, length, quantized = parse_external_data_meta(entries)
        self.assertEqual(file_name, 'w.bin')
        self.assertEqual(offset, 128)
        self.assertEqual(length, 256)
        self.assertTrue(quantized)

    def test_parse_defaults_when_empty(self):
        file_name, offset, length, quantized = parse_external_data_meta([])
        self.assertIsNone(file_name)
        self.assertEqual(offset, 0)
        self.assertEqual(length, -1)
        self.assertIsNone(quantized)


class TestExternalDataIO(unittest.TestCase):
    """external data 的 INT8/float 读取路径，不依赖 onnx 原生 INT4。"""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        for f in os.listdir(self.tmp):
            os.remove(os.path.join(self.tmp, f))
        os.rmdir(self.tmp)

    def test_read_int8_external_file(self):
        data = np.arange(12, dtype=np.int8).reshape(3, 4)
        t = self._build_external_tensor(data, TensorProto.INT8)
        out, _ = TensorProtoHelper(t, model_path=self.tmp).get_external_data()
        self.assertEqual(list(out.shape), [3, 4])
        np.testing.assert_array_equal(out, data)

    def test_read_shape_mismatch_raises(self):
        data = np.arange(12, dtype=np.int8)
        t = self._build_external_tensor(data, TensorProto.INT8)
        t.ClearField('dims')
        t.dims.extend([3, 5])  # 与实际 12 个元素不符
        with self.assertRaises(ValueError):
            TensorProtoHelper(t, model_path=self.tmp).get_external_data()

    def _build_external_tensor(self, np_value, data_type):
        ext_file = os.path.join(self.tmp, 'w.bin')
        np_value.tofile(ext_file)
        t = TensorProto()
        t.data_type = data_type
        t.dims.extend(list(np_value.shape))
        t.data_location = 1  # EXTERNAL
        for k, v in [
            ('location', 'w.bin'),
            ('offset', '0'),
            ('length', str(np_value.nbytes)),
        ]:
            e = t.external_data.add()
            e.key, e.value = k, str(v)
        return t


class TestInt4BranchesViaMock(unittest.TestCase):
    """
    覆盖 INT4 分支：这些分支靠字符串 'int4'/data_type_maps 判断，用 mock
    传参即可覆盖，不依赖 onnx 原生 INT4 枚举。
    """

    _MAP = (
        'amct_pytorch.classic.graph_based.amct_pytorch.utils.'
        'onnx_initializer_util.TensorProtoHelper.data_type_maps'
    )

    def test_resolve_np_dtype_int4(self):
        # 'int4' -> ml_dtypes.int4（ml_dtypes 为独立包，不依赖 onnx）
        import ml_dtypes

        self.assertIs(TensorProtoHelper.resolve_np_dtype('int4'), ml_dtypes.int4)

    def test_resolve_np_dtype_int4_without_ml_dtypes(self):
        mod = (
            'amct_pytorch.classic.graph_based.amct_pytorch.utils.'
            'onnx_initializer_util.ml_dtypes'
        )
        with mock.patch(mod, None):
            with self.assertRaises(ImportError):
                TensorProtoHelper.resolve_np_dtype('int4')

    def test_get_external_quantized_int4(self):
        # 经公有入口 get_external_data 覆盖 quantized raw_data 的 int4 分支
        data = np.array([1, -2, 7, -8], dtype=np.int8)
        packed = pack_int4_to_int8(data)
        t = TensorProto()
        t.data_type = TensorProto.INT8
        t.dims.extend([4])
        t.data_location = 1
        setattr(t, 'raw_data', bytes(packed.astype('uint8')))
        e = t.external_data.add()
        e.key, e.value = 'amct_quantized_raw_data', '1'
        helper = TensorProtoHelper(t)
        with mock.patch.object(helper, 'map_np_type', return_value='int4'):
            out, _ = helper.get_external_data()
        np.testing.assert_array_equal(out.astype(np.int8), data)

    def test_set_data_unsupported_type_raises(self):
        t = TensorProto()
        with self.assertRaises(ValueError):
            TensorProtoHelper(t).set_data(np.array([1], np.int8), 'NO_SUCH_TYPE')

    def test_set_data_int4_packs(self):
        # patch data_type_maps 注入 INT4 条目，使 int4 pack 分支可达
        maps = dict(TensorProtoHelper.data_type_maps)
        maps['INT4'] = [22, 'raw_data', 'int4']  # 22 = TensorProto.INT4 值
        t = TensorProto()
        with mock.patch(self._MAP, maps):
            data = np.array([1, -2, 7, -8], dtype=np.int8)
            TensorProtoHelper(t).set_data(data, 'INT4')
        # 4 个 INT4 -> nibble-packed -> 2 字节
        self.assertEqual(len(t.raw_data), 2)

    def test_get_data_int4(self):
        # patch map_np_type 返回 'int4'，覆盖 get_data 的 int4 unpack 分支
        data = np.array([1, -2, 7, -8], dtype=np.int8)
        packed = pack_int4_to_int8(data)
        t = TensorProto()
        t.data_type = TensorProto.INT8  # 非 UNDEFINED，才会走到 map_np_type
        t.dims.extend([4])
        setattr(t, 'raw_data', bytes(packed.astype('uint8')))
        helper = TensorProtoHelper(t)
        with mock.patch.object(helper, 'map_np_type', return_value='int4'):
            out = helper.get_data()
        np.testing.assert_array_equal(np.asarray(out).astype(np.int8), data)

    def test_read_external_file_int4(self):
        # 经公有入口 get_external_data 覆盖外部文件的 int4 unpack 分支
        tmp = tempfile.mkdtemp()
        try:
            data = np.array([1, -2, 7, -8, 3, -4, 0, 5], dtype=np.int8)
            packed = pack_int4_to_int8(data)
            packed.astype('uint8').tofile(os.path.join(tmp, 'w.bin'))
            t = TensorProto()
            t.data_type = TensorProto.INT8
            t.dims.extend([8])
            t.data_location = 1
            for k, v in [
                ('location', 'w.bin'),
                ('offset', '0'),
                ('length', str(packed.nbytes)),
            ]:
                e = t.external_data.add()
                e.key, e.value = k, v
            helper = TensorProtoHelper(t, model_path=tmp)
            with mock.patch.object(helper, 'map_np_type', return_value='int4'):
                out, _ = helper.get_external_data()
            np.testing.assert_array_equal(out.astype(np.int8), data)
        finally:
            for f in os.listdir(tmp):
                os.remove(os.path.join(tmp, f))
            os.rmdir(tmp)


if __name__ == "__main__":
    unittest.main()
