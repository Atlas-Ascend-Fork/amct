import unittest
from unittest import mock

import torch

from amct_pytorch.classic.graph_based.amct_pytorch.custom_op.ulq_retrain.ulq_retrain import (
    UlqRetrainFunction,
)


class TestUlqRetrainDynamo(unittest.TestCase):
    def test_forward_dynamo_bypasses_eager_quantization(self):
        params = {
            'acts_scale': torch.ones(1),
            'acts_offset': torch.zeros(1),
            'num_bits': 8,
        }
        with (
            mock.patch(
                'amct_pytorch.classic.graph_based.amct_pytorch.custom_op.ulq_retrain.ulq_retrain.is_dynamo_export',
                return_value=True,
            ),
            mock.patch(
                'amct_pytorch.classic.graph_based.amct_pytorch.custom_op.ulq_retrain.ulq_retrain.add_qdq_dynamo',
                return_value=torch.randn(2, 2),
            ) as add_qdq,
        ):
            result = UlqRetrainFunction.forward(
                None,
                torch.randn(2, 2),
                torch.ones(1),
                torch.ones(1),
                torch.ones(1),
                torch.ones(1),
                params,
                None,
                False,
                None,
                1,
            )
        self.assertEqual(len(result), 5)
        add_qdq.assert_called_once()


if __name__ == '__main__':
    unittest.main()
