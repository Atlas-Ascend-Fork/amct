# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# ----------------------------------------------------------------------------
"""Regression: mapping passed directly to data or eval_data must be treated as
one batch, not iterated over its keys (which are strings).

See #245.
"""
import unittest

from amct_pytorch.pruning.context import default_batch_adapter, PruneContext


class TestMappingBatchNormalization(unittest.TestCase):

    def test_dict_data_yields_one_batch(self):
        """PruneContext with a dict data should yield exactly one batch."""
        batch = {"input_ids": [1, 2, 3], "labels": [0]}
        ctx = PruneContext(data=batch)
        out = list(ctx.iter_model_inputs())
        self.assertEqual(len(out), 1)
        kwargs = out[0][1]
        self.assertEqual(kwargs["input_ids"], [1, 2, 3])
        self.assertEqual(kwargs["labels"], [0])

    def test_default_batch_adapter_dict(self):
        """A plain dict batch should return () positional and dict kwargs."""
        batch = {"input_ids": [1, 2, 3], "labels": [0]}
        pos, kwargs = default_batch_adapter(batch)
        self.assertEqual(pos, ())
        self.assertEqual(kwargs["input_ids"], [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
