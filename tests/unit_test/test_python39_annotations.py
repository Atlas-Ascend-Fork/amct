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
"""Dependency-light guards for the documented Python 3.9 baseline.

Run directly with Python 3.9 and PyYAML to exercise the real helper imports
without initializing the optional Torch/NPU/model backends.
"""

import ast
from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys
from typing import Union
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]


def load_source_module(relative_path, name):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


class EagerAnnotationVisitor(ast.NodeVisitor):
    def __init__(self):
        self.function_depth = 0
        self.invalid_lines = set()

    def check_annotation(self, annotation):
        if annotation is not None and any(
            isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr)
            for node in ast.walk(annotation)
        ):
            self.invalid_lines.add(annotation.lineno)

    def visit_FunctionDef(self, node):
        arguments = node.args
        for arg in arguments.posonlyargs + arguments.args + arguments.kwonlyargs:
            self.check_annotation(arg.annotation)
        for arg in (arguments.vararg, arguments.kwarg):
            if arg is not None:
                self.check_annotation(arg.annotation)
        self.check_annotation(node.returns)
        self.function_depth += 1
        self.generic_visit(node)
        self.function_depth -= 1

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node):
        depth = self.function_depth
        self.function_depth = 0
        self.generic_visit(node)
        self.function_depth = depth

    def visit_AnnAssign(self, node):
        if self.function_depth == 0:
            self.check_annotation(node.annotation)
        self.generic_visit(node)


class Python39AnnotationTests(unittest.TestCase):
    maxDiff = None

    def test_default_package_has_no_eager_pep604_annotations(self):
        failures = []
        for path in sorted((REPO_ROOT / "amct_pytorch").rglob("*.py")):
            if "experimental" in path.relative_to(REPO_ROOT).parts:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
            postponed = any(
                isinstance(node, ast.ImportFrom)
                and node.module == "__future__"
                and any(alias.name == "annotations" for alias in node.names)
                for node in tree.body
            )
            if postponed:
                continue
            visitor = EagerAnnotationVisitor()
            visitor.visit(tree)
            failures.extend(
                f"{path.relative_to(REPO_ROOT)}:{line}"
                for line in sorted(visitor.invalid_lines)
            )
        self.assertEqual(failures, [], "Python 3.9 cannot evaluate these unions")

    def test_bit_policy_import_and_resolution(self):
        module = load_source_module(
            "amct_pytorch/quantization/bit_policy.py", "_python39_bit_policy"
        )
        self.assertEqual(module.BitPolicy().linear_bits(), (16, 16))
        policy = module.BitPolicy({"w_bits": 4, "a_bits": 8})
        self.assertEqual(policy.linear_bits(name="q_proj", group="attn-linear"), (4, 8))
        self.assertTrue(policy.has_quant_linear())

    def test_ptq_units_import_and_iteration(self):
        module = load_source_module(
            "amct_pytorch/common/models/llm/common/ptq_units.py", "_python39_ptq_units"
        )
        unit = module.make_ptq_unit("attn", "layer.attn", None, object())
        self.assertIsNone(unit.layer_idx)
        self.assertEqual(unit.save_name, "layer_attn")
        units = list(
            module.iter_indexed_units("moe", "expert", None, ["first", None, "last"])
        )
        self.assertEqual([unit.name for unit in units], ["expert_0", "expert_2"])
        self.assertEqual([unit.module for unit in units], ["first", "last"])

    def test_model_structure_relation_definition(self):
        # Execute the exact dataclass definition without loading quantization
        # backends that are unrelated to this annotation compatibility check.
        source = REPO_ROOT / "amct_pytorch/classic/optimizer/model_structure.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        relation = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "PreOpRelation"
        )
        definition = ast.Module(body=[relation], type_ignores=[])
        namespace = {"dataclass": dataclass, "Union": Union}
        exec(compile(definition, str(source), "exec"), namespace)
        cls = namespace["PreOpRelation"]
        instance = cls("norm", ("q", ("k", "v")), "adapter")
        self.assertEqual(instance.scaling_layers, ("q", ("k", "v")))


if __name__ == "__main__":
    unittest.main()
