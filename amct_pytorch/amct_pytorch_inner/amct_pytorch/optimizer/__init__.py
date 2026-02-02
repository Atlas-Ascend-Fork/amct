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
__all__ = [
    'BaseFusionPass',
    'BaseModuleFusionPass',
    'ConvBnFusionPass',
    'DeleteCircularPaddingPass',
    'DeleteRetrainPass',
    'DeleteRetrainPrunePass',
    'DeleteResizePass',
    'DeleteIsolatedNodePass',
    'DeleteLinearAddPass',
    'GemmTransBOptimizePass',
    'GraphOptimizer',
    'InsertCaliQuantPass',
    'InsertDequantPass',
    'InsertQuantPass',
    'InsertWeightQuantPass',
    'InsertBiasQuantPass',
    'InsertRetrainPass',
    'InsertRetrainQuantPass',
    'InsertRetrainPrunePass',
    'InsertKVCacheQuantPass',
    'InsertRNNFakeQuantPass',
    'ModelOptimizer',
    'MultQuantOptimizerPass',
    'QuantFusionPass',
    'ReplaceAntiQuantPass',
    'ReplaceAvgpoolFlattenPass',
    'ReplaceAvgpoolReshapePass',
    'ReplaceDequantPass',
    'ReplaceQuantPass',
    'ReplaceWeightQuantPass',
    'ReplaceBiasQuantPass',
    'RepalceSyncBNPass',
    'ReplaceRNNPass',
    'SetRecorderPass',
    'ShareActCompPass',
    'WeightsCalibrationPass',
    'InsertFakequantConvPass',
    'InsertFakequantConvTransposePass',
    'InsertFakequantLinearPass',
    'InsertFakequantAvgPool2dPass',
    'WeightFakequantModulePass',
    'BiasFakequantModulePass',
    'InsertDMQBalancerPass',
    'ApplyDMQBalancerPass',
    'InsertQatPass',
    'DeleteQatPass'
]

from . import base_fusion_pass
from . import base_module_fusion_pass
from . import conv_bn_fusion_pass
from . import delete_circular_padding_pass
from . import delete_retrain_pass
from . import delete_linear_add_pass
from . import delete_resize_pass
from . import delete_isolated_node_pass
from . import gemm_transb_optimize_pass
from . import graph_optimizer
from . import insert_cali_quant
from . import insert_dequant_pass
from . import insert_quant_pass
from . import insert_weight_quant_pass
from . import insert_bias_quant_pass
from . import insert_retrain_pass
from . import insert_retrain_quant_pass
from . import insert_kv_cache_quant_pass
from . import model_optimizer
from . import mult_output_with_quant_optimizer
from . import quant_fusion_pass
from . import replace_anti_quant_pass
from . import replace_avgpool_flatten_pass
from . import replace_avgpool_reshape_pass
from . import replace_dequant_pass
from . import replace_quant_pass
from . import replace_weight_quant_pass
from . import replace_bias_quant_pass
from . import replace_sync_bn_pass
from . import set_recorder_pass
from . import share_act_comp_pass
from . import weight_calibration
from . import insert_fakequant_conv_pass
from . import insert_fakequant_convtranspose_pass
from . import insert_fakequant_linear_pass
from . import insert_fakequant_avgpool2d_pass
from . import weight_fakequant_module_pass
from . import bias_fakequant_module_pass
from . import insert_dmq_balancer_pass
from . import apply_dmq_balancer_pass
from . import insert_qat_pass
from . import delete_qat_pass

from .base_fusion_pass import BaseFusionPass
from .base_module_fusion_pass import BaseModuleFusionPass
from .conv_bn_fusion_pass import ConvBnFusionPass
from .delete_circular_padding_pass import DeleteCircularPaddingPass
from .delete_retrain_pass import DeleteRetrainPass
from .delete_retrain_prune_pass import DeleteRetrainPrunePass
from .delete_linear_add_pass import DeleteLinearAddPass
from .delete_resize_pass import DeleteResizePass
from .delete_isolated_node_pass import DeleteIsolatedNodePass
from .gemm_transb_optimize_pass import GemmTransBOptimizePass
from .graph_optimizer import GraphOptimizer
from .insert_cali_quant import InsertCaliQuantPass
from .insert_dequant_pass import InsertDequantPass
from .insert_quant_pass import InsertQuantPass
from .insert_weight_quant_pass import InsertWeightQuantPass
from .weight_fakequant_pass import WeightFakequantPass
from .insert_bias_quant_pass import InsertBiasQuantPass
from .insert_retrain_pass import InsertRetrainPass
from .insert_retrain_quant_pass import InsertRetrainQuantPass
from .insert_retrain_prune_pass import InsertRetrainPrunePass
from .insert_kv_cache_quant_pass import InsertKVCacheQuantPass
from .insert_rnn_fake_quant_pass import InsertRNNFakeQuantPass
from .model_optimizer import ModelOptimizer
from .mult_output_with_quant_optimizer import MultQuantOptimizerPass
from .quant_fusion_pass import QuantFusionPass
from .replace_anti_quant_pass import ReplaceAntiQuantPass
from .replace_avgpool_flatten_pass import ReplaceAvgpoolFlattenPass
from .replace_avgpool_reshape_pass import ReplaceAvgpoolReshapePass
from .replace_dequant_pass import ReplaceDequantPass
from .replace_quant_pass import ReplaceQuantPass
from .replace_weight_quant_pass import ReplaceWeightQuantPass
from .replace_bias_quant_pass import ReplaceBiasQuantPass
from .replace_sync_bn_pass import RepalceSyncBNPass
from .replace_rnn_pass import ReplaceRNNPass
from .set_recorder_pass import SetRecorderPass
from .share_act_comp_pass import ShareActCompPass
from .weight_calibration import WeightsCalibrationPass
from .insert_fakequant_conv_pass import InsertFakequantConvPass
from .insert_fakequant_convtranspose_pass import InsertFakequantConvTransposePass
from .insert_fakequant_linear_pass import InsertFakequantLinearPass
from .insert_fakequant_avgpool2d_pass import InsertFakequantAvgPool2dPass
from .weight_fakequant_module_pass import WeightFakequantModulePass
from .bias_fakequant_module_pass import BiasFakequantModulePass
from .insert_dmq_balancer_pass import InsertDMQBalancerPass
from .apply_dmq_balancer_pass import ApplyDMQBalancerPass
from .insert_qat_pass import InsertQatPass
from .delete_qat_pass import DeleteQatPass
