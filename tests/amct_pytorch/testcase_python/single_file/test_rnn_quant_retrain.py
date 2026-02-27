import sys
import os
import unittest
import json
import numpy as np
import torch

from .utils import rnn_model
from .utils import record_file_utils
from amct_pytorch.graph_based_compression.amct_pytorch.parser.parser import Parser
from amct_pytorch.graph_based_compression.amct_pytorch.proto import scale_offset_record_pb2
from google.protobuf import text_format

from amct_pytorch.graph_based_compression.amct_pytorch.quantize_tool import create_quant_config
from amct_pytorch.graph_based_compression.amct_pytorch.quantize_tool import quantize_model
from amct_pytorch.graph_based_compression.amct_pytorch.quantize_tool import save_model

from amct_pytorch.graph_based_compression.amct_pytorch.quantize_tool import create_quant_retrain_config
from amct_pytorch.graph_based_compression.amct_pytorch.quantize_tool import create_quant_retrain_model
from amct_pytorch.graph_based_compression.amct_pytorch.quantize_tool import restore_quant_retrain_model
from amct_pytorch.graph_based_compression.amct_pytorch.quantize_tool import save_quant_retrain_model

torch.manual_seed(0)
CUR_DIR = os.path.split(os.path.realpath(__file__))[0]

class TestQuantizeTool(unittest.TestCase):
    """
    The UT for QuantizeTool
    """
    @classmethod
    def setUpClass(cls):
        batch_size = 32
        time_steps = 3
        channels = 3
        height = 64
        width = 64
        num_class = 10
        num_epochs = 10
        learning_rate = 0.001
        conv1d_kernel_size = 3
        conv1d_out_channels = 16
        gru_hidden_size = 64
        num_gru_layers = 1

        cls.model = rnn_model.Conv1dGRU(input_channels=channels,
                            conv1d_kernel_size=conv1d_kernel_size,
                            conv1d_out_channels=conv1d_out_channels,
                            gru_hidden_size=gru_hidden_size,
                            num_gru_layers=num_gru_layers,
                            dropout=0.1)

        cls.temp_folder = os.path.join(CUR_DIR, 'test_rnn')
        if not os.path.isdir(cls.temp_folder):
            os.makedirs(cls.temp_folder)
        
        cls.input = torch.randn(1, time_steps, channels, height, width)
        cls.h0 = torch.zeros(1, 1, gru_hidden_size)

    @classmethod
    def tearDownClass(cls):
        os.popen('rm -r ' + cls.temp_folder)
        pass

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_create_quant_config(self):
        config_file = os.path.join(self.temp_folder, 'config.json')
        create_quant_config(
            config_file=config_file,
            model=self.model,
            input_data=(self.input, self.h0))

        self.assertTrue(os.path.exists(config_file))