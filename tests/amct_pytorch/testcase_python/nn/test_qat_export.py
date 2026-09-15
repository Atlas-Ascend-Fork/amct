import unittest
from unittest import mock

import torch

from amct_pytorch.classic.graph_based.amct_pytorch.nn.module.quantization.linear import (
    LinearQAT,
)


class TestQatExport(unittest.TestCase):
    def _make_module(self, channel_wise=False):
        return LinearQAT(
            2,
            4,
            config={
                'retrain_data_config': {'clip_min': -1.0, 'clip_max': 1.0},
                'retrain_weight_config': {
                    'dst_type': 'INT4',
                    'channel_wise': channel_wise,
                },
            },
        )

    def test_forward_qat_dynamo_builds_qdq_without_mutation(self):
        module = self._make_module()
        module.do_init = False
        with (
            mock.patch(
                'amct_pytorch.classic.graph_based.amct_pytorch.nn.module.quantization.qat_base.is_dynamo_export',
                return_value=True,
            ),
            mock.patch(
                'amct_pytorch.classic.graph_based.amct_pytorch.nn.module.quantization.qat_base.add_qdq_dynamo',
                return_value=torch.randn(1, 2),
            ) as add_act,
            mock.patch(
                'amct_pytorch.classic.graph_based.amct_pytorch.nn.module.quantization.qat_base.add_weight_qdq_dynamo',
                return_value=torch.randn(4, 2),
            ) as add_weight,
        ):
            result = module.forward_qat(torch.randn(1, 2))
        self.assertEqual(len(result), 2)
        add_act.assert_called_once()
        add_weight.assert_called_once()
        self.assertIs(add_weight.call_args.args[2], module.wts_offsets_deploy)

    def test_forward_qat_dynamo_requires_initialized_model(self):
        module = self._make_module()
        module.do_init = True
        with (
            mock.patch(
                'amct_pytorch.classic.graph_based.amct_pytorch.nn.module.quantization.qat_base.is_dynamo_export',
                return_value=True,
            ),
            mock.patch(
                'amct_pytorch.classic.graph_based.amct_pytorch.nn.module.quantization.qat_base.add_qdq_dynamo'
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, 'initialized'):
                module.forward_qat(torch.randn(1, 2))


if __name__ == '__main__':
    unittest.main()
