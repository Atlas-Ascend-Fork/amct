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
import json
import logging
import os
import shutil
import tempfile
import unittest

import numpy as np

from amct_pytorch.classic.graph_based.amct_pytorch.common.utils import files

logger = logging.getLogger(__name__)


class TestFiles(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix='amct_test_files_')

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_create_path_idempotent(self):
        target = self._path('a', 'b', 'c')
        files.create_path(target)
        self.assertTrue(os.path.isdir(target))
        # second call hits exist_ok / FileExistsError tolerance branch
        files.create_path(target)
        self.assertTrue(os.path.isdir(target))

    def test_create_path_existing_file_tolerated(self):
        # makedirs on a path that is an existing *file* raises FileExistsError,
        # which create_path swallows.
        existing = files.create_empty_file(self._path('a_file'))
        files.create_path(existing)
        self.assertTrue(os.path.isfile(existing))

    def test_create_file_path_and_empty_file(self):
        file_name = self._path('sub', 'record.txt')
        real = files.create_empty_file(file_name)
        self.assertTrue(os.path.isfile(real))
        self.assertEqual(os.path.getsize(real), 0)
        # permission bits set to FILE_MODE (640)
        mode = oct(os.stat(real).st_mode & 0o777)
        self.assertEqual(mode, oct(0o640))

    def test_create_file_path_check_exist(self):
        file_name = self._path('exist_check.txt')
        # check_exist on a non-existing file just warns, no raise
        files.create_file_path(file_name, check_exist=True)
        self.assertTrue(os.path.isdir(os.path.dirname(file_name)))

    def test_is_valid_name_ok(self):
        # valid name should not raise
        files.is_valid_name('model.onnx', 'model')

    def test_is_valid_name_empty(self):
        self.assertRaises(ValueError, files.is_valid_name, '', 'model')

    def test_is_valid_name_is_path(self):
        self.assertRaises(ValueError, files.is_valid_name, 'some/dir/', 'model')

    def test_is_valid_name_too_long(self):
        long_name = 'a' * 255
        self.assertRaises(ValueError, files.is_valid_name, long_name, 'model')

    def test_is_valid_save_prefix_ok(self):
        files.is_valid_save_prefix('prefix')
        files.is_valid_save_prefix('')

    def test_is_valid_save_prefix_invalid(self):
        self.assertRaises(ValueError, files.is_valid_save_prefix, 'a' * 242)

    def test_check_file_path_missing(self):
        self.assertRaises(RuntimeError, files.check_file_path,
                          self._path('not_exist.bin'), 'feature')

    def test_check_file_path_ok(self):
        file_name = files.create_empty_file(self._path('ok.bin'))
        files.check_file_path(file_name, 'feature')

    def test_delete_dir(self):
        target = self._path('to_delete')
        os.makedirs(target)
        files.delete_dir(target)
        self.assertFalse(os.path.exists(target))
        # delete a non-existing dir tolerated (ignore_errors=True)
        files.delete_dir(target)

    def test_check_files_exist_warn(self):
        file_name = files.create_empty_file(self._path('dup.txt'))
        # existing file path triggers the overwrite warning branch
        files.check_files_exist([file_name])

    def test_split_save_path_empty(self):
        save_dir, prefix = files.split_save_path('')
        self.assertEqual(prefix, '')
        self.assertEqual(save_dir, os.path.realpath(''))

    def test_split_save_path_trailing_slash(self):
        save_dir, prefix = files.split_save_path(self.tmp_dir + '/')
        self.assertEqual(prefix, '')
        self.assertEqual(save_dir, os.path.realpath(self.tmp_dir))

    def test_split_save_path_with_prefix(self):
        save_dir, prefix = files.split_save_path(self._path('myprefix'))
        self.assertEqual(prefix, 'myprefix')
        self.assertEqual(save_dir, os.path.realpath(self.tmp_dir))

    def test_concat_name_with_prefix(self):
        name = files.concat_name('/tmp/out', 'pre', 'weight.bin')
        self.assertEqual(name, os.path.join('/tmp/out', 'pre_weight.bin'))

    def test_concat_name_without_prefix(self):
        name = files.concat_name('/tmp/out', '', 'weight.bin')
        self.assertEqual(name, os.path.join('/tmp/out', 'weight.bin'))

    def test_find_dump_file_ok(self):
        files.create_empty_file(self._path('fm_layer1.bin'))
        files.create_empty_file(self._path('fm_layer2.bin'))
        os.makedirs(self._path('a_subdir'))
        found = files.find_dump_file(self.tmp_dir, 'fm_')
        self.assertEqual(len(found), 2)
        for f in found:
            self.assertTrue(os.path.basename(f).startswith('fm_'))

    def test_find_dump_file_none(self):
        self.assertRaises(RuntimeError, files.find_dump_file, self.tmp_dir, 'nope_')

    def test_parse_dump_data_without_type(self):
        # layout (no type): | dim, shape... | data |
        dim = np.array([1.0], np.float32)
        shape = np.array([3.0], np.float32)
        data = np.array([1.0, 2.0, 3.0], np.float32)
        raw = dim.tobytes() + shape.tobytes() + data.tobytes()
        bin_path = self._path('dump_no_type.bin')
        with open(bin_path, 'wb') as fid:
            fid.write(raw)
        out = files.parse_dump_data(bin_path, with_type=False)
        self.assertEqual(list(out), [1.0, 2.0, 3.0])

    def test_parse_dump_data_with_type(self):
        # layout (with type): | type, dim, shape... | data |
        head = np.array([0.0, 1.0, 2.0], np.float32)  # type=0(float32), dim=1, shape=[2]
        data = np.array([4.0, 5.0], np.float32)
        raw = head.tobytes() + data.tobytes()
        bin_path = self._path('dump_with_type.bin')
        with open(bin_path, 'wb') as fid:
            fid.write(raw)
        out = files.parse_dump_data(bin_path, with_type=True)
        self.assertEqual(list(out), [4.0, 5.0])

    def test_save_to_json(self):
        json_path = self._path('out', 'content.json')
        content = {'a': 1, 'b': [2, 3], 'c': 'x'}
        files.save_to_json(json_path, content)
        self.assertTrue(os.path.isfile(json_path))
        with open(json_path) as fid:
            self.assertEqual(json.load(fid), content)

    def _path(self, *parts):
        return os.path.join(self.tmp_dir, *parts)


if __name__ == '__main__':
    unittest.main()
