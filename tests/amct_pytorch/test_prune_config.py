from __future__ import annotations

import unittest

from mock_torch_npu import *  # noqa: F401,F403

from amct_pytorch.pruning import MethodSpec, PruneConfig


class ConfigValidationTest(unittest.TestCase):
    def test_invalid_method_prune_ratio_raises(self) -> None:
        with self.assertRaises(ValueError):
            PruneConfig(methods={"dense": MethodSpec("low_variance", {"prune_ratio": 1.5})}).validate()

    def test_invalid_method_seed_raises(self) -> None:
        with self.assertRaises(ValueError):
            PruneConfig(methods={"cnn": MethodSpec("tp_variance_channel", {"seed": "bad"})}).validate()


if __name__ == "__main__":
    unittest.main()
