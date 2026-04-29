import sys
import os
import unittest
import json
import numpy as np
import torch
import onnx
import onnxruntime as ort
from google.protobuf import text_format

from amct_pytorch.graph_based_compression.amct_pytorch.parser.parser import Parser
from amct_pytorch.graph_based_compression.amct_pytorch.proto import scale_offset_record_pb2

from amct_pytorch.graph_based_compression.amct_pytorch.quantize_tool import create_quant_config
from amct_pytorch.graph_based_compression.amct_pytorch.quantize_tool import quantize_model
from amct_pytorch.graph_based_compression.amct_pytorch.quantize_tool import save_model

from amct_pytorch.graph_based_compression.amct_pytorch.quantize_tool import create_quant_retrain_config
from amct_pytorch.graph_based_compression.amct_pytorch.quantize_tool import create_quant_retrain_model
from amct_pytorch.graph_based_compression.amct_pytorch.quantize_tool import restore_quant_retrain_model
from amct_pytorch.graph_based_compression.amct_pytorch.quantize_tool import save_quant_retrain_model

from .utils import rnn_model
from .utils import record_file_utils

torch.manual_seed(0)
CUR_DIR = os.path.split(os.path.realpath(__file__))[0]


class TestGRUPTQ(unittest.TestCase):
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
                            num_classes=num_class,
                            num_gru_layers=num_gru_layers,
                            dropout=0.1)

        cls.temp_folder = os.path.join(CUR_DIR, 'test_rnn')
        if not os.path.isdir(cls.temp_folder):
            os.makedirs(cls.temp_folder)
        
        cls.input = torch.randn(1, time_steps, channels, height, width)
        cls.h0 = torch.zeros(1, 1, gru_hidden_size)
        
        cls.ori_out = cls.model(cls.input, cls.h0)

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
    
    def test_quantize_model(self):
        config_file = os.path.join(self.temp_folder, 'config.json')
        record_file = os.path.join(self.temp_folder, 'record.txt')
        modified_model = os.path.join(self.temp_folder, 'modified_model.onnx')
        new_model = quantize_model(config_file, modified_model, record_file,
            self.model, (self.input, self.h0))

        self.assertTrue(os.path.exists(modified_model))
        output = new_model(self.input, self.h0)
        
        self.assertTrue(os.path.exists(modified_model))
        self.assertIsNotNone(output)
    
    def test_save_model(self):
        record_file = os.path.join(self.temp_folder, 'record.txt')
        modified_model = os.path.join(self.temp_folder, 'modified_model.onnx')
        save_path = os.path.join(self.temp_folder, 'res')
        save_model(modified_model, record_file, save_path)
        
        fakequant = os.path.join(self.temp_folder, 'res_fake_quant_model.onnx')
        deploy = os.path.join(self.temp_folder, 'res_deploy_model.onnx')

        self.assertTrue(os.path.exists(fakequant))
        self.assertTrue(os.path.exists(deploy))

    
class TestGRUQAT(unittest.TestCase):
    """
    The UT for QuantizeTool
    """
    @classmethod
    def setUpClass(cls):
        cls.batch_size = 1
        time_steps = 3
        channels = 3
        height = 64
        width = 64
        cls.num_class = 10
        num_epochs = 10
        cls.learning_rate = 0.001
        conv1d_kernel_size = 3
        conv1d_out_channels = 16
        gru_hidden_size = 64
        num_gru_layers = 1

        cls.model = rnn_model.Conv1dGRU(input_channels=channels,
                            conv1d_kernel_size=conv1d_kernel_size,
                            conv1d_out_channels=conv1d_out_channels,
                            gru_hidden_size=gru_hidden_size,
                            num_classes=cls.num_class,
                            num_gru_layers=num_gru_layers,
                            dropout=0.1)

        cls.temp_folder = os.path.join(CUR_DIR, 'test_rnn')
        if not os.path.isdir(cls.temp_folder):
            os.makedirs(cls.temp_folder)
        
        cls.input = torch.randn(1, time_steps, channels, height, width)
        cls.h0 = torch.zeros(1, 1, gru_hidden_size)
        
        cls.ori_out = cls.model(cls.input, cls.h0)

        cls.new_model = None

    @classmethod
    def tearDownClass(cls):
        os.popen('rm -r ' + cls.temp_folder)
        pass

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_create_quant_retrain_config(self):
        config_file = os.path.join(self.temp_folder, 'config.json')
        create_quant_retrain_config(
            config_file=config_file,
            model=self.model,
            input_data=(self.input, self.h0))

        self.assertTrue(os.path.exists(config_file))
    
    def test_create_quant_retrain_model(self):
        config_file = os.path.join(self.temp_folder, 'config.json')
        record_file = os.path.join(self.temp_folder, 'record.txt')
        self.new_model = create_quant_retrain_model(config_file, self.model, record_file,
            (self.input, self.h0))
        
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.new_model.parameters(), lr=self.learning_rate)
        label = torch.randint(0, self.num_class, (self.batch_size,))

        self.assertIsNotNone(self.new_model)
        output, _ = self.new_model(self.input, self.h0)
        self.assertIsNotNone(output)
        
        loss = criterion(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        self.new_model.eval()
        with torch.no_grad():
            output, _ = self.new_model(self.input, self.h0)

        save_path = os.path.join(self.temp_folder, 'res')
        fakequant = os.path.join(self.temp_folder, 'res_fake_quant_model.onnx')
        deploy = os.path.join(self.temp_folder, 'res_deploy_model.onnx')
        
        save_quant_retrain_model(model=self.new_model, 
                                 input_data=(self.input, self.h0), 
                                 config_file=config_file, 
                                 record_file=record_file, 
                                 save_path=save_path)

        self.assertTrue(os.path.exists(fakequant))
        self.assertTrue(os.path.exists(deploy))