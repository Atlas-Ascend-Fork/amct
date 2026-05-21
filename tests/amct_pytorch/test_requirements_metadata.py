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
import re
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _read_requirements():
    requirements = {}
    for raw_line in (PROJECT_ROOT / 'requirements.txt').read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#'):
            continue
        match = re.match(r'([A-Za-z0-9_.-]+)', line)
        if match is not None:
            requirements[match.group(1).lower()] = line
    return requirements


class TestRequirementsMetadata(unittest.TestCase):
    def test_runtime_dependencies_match_supported_versions(self):
        requirements = _read_requirements()

        self.assertEqual(requirements.get('onnx'), 'onnx==1.21.0')
        self.assertEqual(requirements.get('onnxruntime'), 'onnxruntime==1.25.1')
        self.assertIn('protobuf>=4.25.1', requirements.get('protobuf', ''))
        self.assertIn('numpy>=1.26.4', requirements.get('numpy', ''))
        self.assertNotIn('<=1.24.0', requirements.get('numpy', ''))
        self.assertIn('scipy', requirements)

    def test_build_document_lists_test_dependencies(self):
        build_doc = (PROJECT_ROOT / 'docs' / 'build.md').read_text()

        self.assertIn('pip3 install coverage triton', build_doc)


if __name__ == '__main__':
    unittest.main()
