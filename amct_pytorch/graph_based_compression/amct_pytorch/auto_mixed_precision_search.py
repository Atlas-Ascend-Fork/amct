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
import torch

from ..amct_pytorch.quantize_tool import create_quant_config
from ..amct_pytorch.quantize_tool import create_quant_retrain_config
from ..amct_pytorch.quantize_tool import inner_fuse_bn
from ..amct_pytorch.quantize_tool import inner_quantize_model
from ..amct_pytorch.quantize_tool import generate_fakequant_module
from ..amct_pytorch.quantize_tool import add_dump_operations
from ..amct_pytorch.parser.parser import Parser
from ..amct_pytorch.common.utils.check_params import check_params
from ..amct_pytorch.common.utils import struct_helper
from ..amct_pytorch.common.auto_calibration import AutoCalibrationEvaluatorBase
from ..amct_pytorch.common.auto_mixed_precision.auto_mixed_precision_search \
    import AutoSearchMixedPrecisionBase
from ..amct_pytorch.common.auto_calibration.sensitivity_base import SensitivityBase
from ..amct_pytorch.configuration.check import GraphQuerier
from ..amct_pytorch.utils.model_util import ModuleHelper
from ..amct_pytorch.utils.net_params import ParamsHelperTorch
from ..amct_pytorch.utils.auto_calibration_helper import AutoCalibrationHelper
from ..amct_pytorch.utils.log import LOGGER
from ..amct_pytorch.common.utils.vars_util import FP16_BIT


class GraphInfo(struct_helper.GraphInfoBase):
    """
    Function:
    Graph info to manage model's variables.
    """
    def __init__(self, model, input_data):
        super().__init__(model)
        self.model = model
        self.input_data = input_data


class AutoSearchMixedPrecision(AutoSearchMixedPrecisionBase):
    """
    the class for quick automatic calibration API
    """
    @check_params(model=torch.nn.Module,
                  amc_config=str,
                  save_dir=str,
                  evaluator=AutoCalibrationEvaluatorBase,
                  sensitivity=(str, SensitivityBase)
                  )
    def __init__(self,
                 model,
                 input_data,
                 amc_config,
                 save_dir,
                 evaluator,
                 sensitivity):
        '''
        Function:
        init three classes for infomation management.
        Class GraphInfo: manage graph's info based on GraphInfoBase. E.g. torch's export onnx info.
        Class CalibrationConfigInfo: manage calibration's info. E.g. calibration's data.
        Class AutoQuantInfo: amc's infomation. E.g. config file, proto.

        Arguments:
        model: torch.nn.Module, users's model to search.
        input_data: used to compile model, can be random data.
        amc_config: amc config file contains amc config(MUST).
                    calibration & quant config(OPTIONAL).
        save_dir: path to save file.
                  quant config contrains num_bits override.
                  And some temporary intermediate file.
        evaluator: user-defined class based on AutoCalibrationEvaluatorBase.
                  MUST have calibration function to infer the model.

        Return: None
        '''
        graph_info = GraphInfo(model, input_data)
        auto_quant_info = struct_helper.AutoSearchMixedPrecisionInfo(amc_config, save_dir, sensitivity)
        cali_config_info = struct_helper.CalibrationConfigInfo(auto_quant_info.cfg_helper.ptq_cfg)

        super().__init__(graph_info, cali_config_info, auto_quant_info)

        self.evaluator = evaluator

    def get_original_model(self):
        """
        Function: get the original model.
        """
        try:
            model = ModuleHelper.deep_copy(self.graph_info.model)
        except RuntimeError as exception:
            LOGGER.logw(
                exception, "auto_search_quant_bit deep_copy model")
        return model

    def cal_node_bitops(self):
        """
        Function:
        Calculate bitops of quantizable layer in model, according its `layer_name`,

        Args:
        None

        Return:
        constraint_bitops: dict, bitops info of layers in each bit.
        {'layer_name': {'FLOPs': flops, 'num_bits': bitops}}
        """

        params_helper_torch = ParamsHelperTorch(
            original_model=self.get_original_model()
            )

        constraint_bitops = {}
        for layer_name in self.auto_quant_info.quant_layers:

            flops = params_helper_torch.get_flops(layer_name, self.auto_quant_info.shape_info[layer_name])

            constraint_bitops[layer_name] = {}

            constraint_bitops.get(layer_name)['FLOPs'] = flops

            for quant_bit in self.auto_quant_info.quant_mix_bits:
                constraint_bitops.get(layer_name)[str(quant_bit)] = flops * quant_bit * quant_bit

        return constraint_bitops

    def layer_sensitivity_analysis(self):
        """
        Function:
        Analyze sensitivity by analyzing layer's output difference before and after quant.
        Assign `acc_decay` dict to `auto_quant_info.acc_decay`.
        {'layer_name': {'num_bits': 'sensitivity'}}
        Assign `shape_info` dict to `auto_quant_info.shape_info`.
        {'layer_name': {'input_shape': [shape0, shape1...], 'output_shape': [shape0, shape1...]}}

        Args:
        None

        Return:
        None
        """
        def analysis_original_quant(num_bits):
            """
            Function:
            Analyze sensitivity with specific num_bits.
            step 1: do original model quant based on record file.
            step 2: calculate sensitivity using data which has dumped.

            Args:
            num_bits: int, bit to do quant. 4 / 8

            Return:
            None
            """
            config_file = self.map_file(num_bits, 'config_file')
            record_file = self.map_file(num_bits, 'record_file')

            fake_quant_module = generate_fakequant_module(
                model=self.get_original_model(),
                config_file=config_file,
                record_file=record_file,
                input_data=self.graph_info.input_data
            )
            auto_calibration_helper = AutoCalibrationHelper(
                fused_module=fused_model,
                fake_quant_module=fake_quant_module,
                quant_layers=self.auto_quant_info.quant_layers,
                record_file=record_file,
                temp_dir=self.temp_info.dump_dir,
                sensitivity=self.auto_quant_info.sensitivity
            )
            ranking_info, shape_info = auto_calibration_helper.calc_ranking_info()
            for layer in ranking_info:
                acc_decay[layer] = {} if layer not in acc_decay.keys() else acc_decay.get(layer)
                acc_decay.get(layer)[str(num_bits)] = ranking_info.get(layer)
            LOGGER.logi(
                info_message="================ num bit is {} ================".format(num_bits),
                module_name="AMCT_PYTORCH")
            LOGGER.logi(info_message='sensitive of quantized layer:', module_name='auto_mixed_precision_search')
            for layer_name, sensitivity_info in acc_decay.items():
                LOGGER.logi(info_message='layer: {}, sensitivity: {}'.format(layer_name, sensitivity_info),
                            module_name='auto_mixed_precision_search')
            self.auto_quant_info.shape_info = shape_info

        acc_decay = {}
        # note, num_bits to get the config and record file.
        # generate fuse bn model.
        fused_model = self.prepare_fuse_bn_model(num_bits=4)
        # generate the dump data.
        self.dump_data()

        for num_bits in self.auto_quant_info.quant_mix_bits:
            if num_bits == FP16_BIT:
                for quant_layer in self.auto_quant_info.quant_layers:
                    acc_decay[quant_layer] = {} if quant_layer not in acc_decay.keys() else acc_decay.get(quant_layer)
                    acc_decay.get(quant_layer)[str(FP16_BIT)] = 0
                continue
            # first time to dump data, and prepare fused bn model.
            analysis_original_quant(num_bits)
        self.auto_quant_info.acc_decay = acc_decay

    def prepare_fuse_bn_model(self, num_bits):
        """
        Function:
        Prepare the fuse bn model at first, duplicate generation is not required.

        Args:
        num_bits: int, used to get the config file and record file.

        Return:
        None.
        """
        config_file = self.map_file(num_bits, 'config_file')
        record_file = self.map_file(num_bits, 'record_file')
        fused_model = inner_fuse_bn(
                model=self.get_original_model(),
                config_file=config_file,
                record_file=record_file,
                input_data=self.graph_info.input_data
                )
        return fused_model

    def dump_data(self):
        """
        Function:
        Dump data, used to calculate the sensitivity and shape info.
        Dump data into `dump_dir` with `test_iteration` batch_num.
        """
        dump_config = struct_helper.DumpConfig(dump_dir=self.temp_info.dump_dir,
                                batch_num=self.auto_quant_info.cfg_helper.test_iteration)
        original_model = self.get_original_model()
        add_dump_operations(original_model, dump_config)
        self.evaluator.evaluate(original_model, self.auto_quant_info.cfg_helper.test_iteration)

    def prepare_calibration_config(self, config_defination):
        """
        Function:
        generate the global calibration config file,
        based on this, different num_bits config file will generate.

        Args:
        None.

        Return:
        None.
        """
        create_quant_config(self.temp_info.config_file, self.graph_info.model, self.graph_info.input_data,
                            config_defination=config_defination)

    def prepare_qat_config(self, config_defination):
        """ prepare quant retrain config. """
        create_quant_retrain_config(self.temp_info.config_file, self.graph_info.model, self.graph_info.input_data,
                                    config_defination=config_defination)

    def do_global_calibration(self, num_bits):
        """
        Function: Do calibration to generate record with num_bit for whole graph.
        This calibration not dump data for dump_config is None.

        Args:
        num_bits (int): in which bit to do quant.

        Return:
        None.
        """
        config_file = self.map_file(num_bits, 'config_file')
        record_file = self.map_file(num_bits, 'record_file')

        model = self.get_original_model()
        # generate a quantize model
        # this quantize model will not dump data for dump_config is None.
        # in order to generate the record file, dump data when first analysis.
        try:
            modified_module = inner_quantize_model(
                config_file=config_file,
                modfied_onnx_file=self.temp_info.modified_onnx_file,
                record_file=record_file,
                model=model,
                input_data=self.graph_info.input_data,
                weight_fakequant=False)
        except Exception as e:
            raise RuntimeError("fail to do quantize_model when model is quant to {0} bit with algo in 'ptq_cfg' in "
                               "AutoMixedPrecisionConfig in post training quantization. Please check the error info "
                               "for more information. Maybe the algo cannot support {0}".format(num_bits)) from e
        # activation calibration process
        with open(config_file) as fid:
            quant_config = json.load(fid)
        try:
            self.evaluator.calibration(modified_module, batch_num=quant_config.get('batch_num'))
        except Exception as e:
            raise RuntimeError("fail to do calibration when model is quant to {} bit in post training quantization. "
                               "Please check the error info for more information.".format(num_bits)) from e

    def get_support_layer2type(self, mode):
        """
        Function:
        Get the corresponding support mode's dict.

        Args:
        mode: string, get in which quant scene. 'ptq' and 'qat' is valid.

        Return:
        support_layer2type: dict, key is layer name and value is layer type.
        """
        if mode not in ["ptq", "qat"]:
            raise ValueError('only support ["ptq", "qat"], but get {}'.format(mode))
        # get original model
        model = self.get_original_model()
        # get inner graph
        tmp_onnx = self.temp_info.modified_onnx_file
        Parser.export_onnx(model, self.graph_info.input_data, tmp_onnx)
        graph = Parser.parse_net_to_graph(tmp_onnx)
        graph.add_model(model)
        if mode == 'qat':
            support_layer2type = GraphQuerier.get_support_qat_layer2type(graph=graph)
        if mode == 'ptq':
            support_layer2type = GraphQuerier.get_support_quant_layer2type(graph=graph)
        return support_layer2type


@check_params(
    model=torch.nn.Module,
    config=str,
    save_dir=str,
    evaluator=AutoCalibrationEvaluatorBase,
    sensitivity=(str, SensitivityBase)
)
def auto_mixed_precision_search(
    model,
    input_data,
    config,
    save_dir,
    evaluator,
    sensitivity='MseSimilarity'):
    """
    Function: Auto Mix QUANT.
    Auto search a feasible suboptimal num bits config.

    Parameters:
    model: user's torch.nn.Module
    input_data: used to compile model, can be random data.
    config: amc config file contains amc config(MUST).
                calibration & quant config(OPTIONAL).
    save_dir: path to save file.
              quant config contrains num_bits override.
              And some temporary intermediate file.
    evaluator: user-defined class based on AutoCalibrationEvaluatorBase.
               MUST have calibration function to infer the model.
    Return:
    None.
    """
    amc = AutoSearchMixedPrecision(
        model=model,
        input_data=input_data,
        amc_config=config,
        save_dir=save_dir,
        evaluator=evaluator,
        sensitivity=sensitivity
    )
    amc.run()
