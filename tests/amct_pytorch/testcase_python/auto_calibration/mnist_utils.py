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
import os
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import onnx
import onnxruntime
import numpy as np

CUR_DIR = os.path.split(os.path.realpath(__file__))[0]
DATASETS_DIR = os.path.realpath(os.path.join(CUR_DIR, '../../../../../../../../build/bin/llt/toolchain/dmct_datasets'))
DATA_PATH = os.path.join(DATASETS_DIR, 'pytorch/data')


class CustomDataset(Dataset):
    def __init__(self, num_samples):
        """
        初始化数据集。
        
        参数:
            num_samples (int): 数据集中的样本数量。
        """
        self.num_samples = num_samples
        self.data = []
        self.labels = []

        # 生成数据和标签
        for _ in range(num_samples):
            # 生成一个形状为 (1, 28, 28) 的张量，均值为 0.1307，方差为 0.3081
            tensor = torch.randn(1, 28, 28) * np.sqrt(0.3081) + 0.1307
            # 生成一个范围在 1 到 10 之间的随机标签
            label = torch.randint(0, 10, (1,)).item()
            self.data.append(tensor)
            self.labels.append(label)
    
    def __len__(self):
        """返回数据集的长度。"""
        return self.num_samples
    
    def __getitem__(self, idx):
        """
        根据索引获取数据和标签。
        
        参数:
            idx (int): 索引值。
            
        返回:
            tuple: 包含张量和标签的元组。
        """
        return self.data[idx], self.labels[idx]


def run_inference_model(model, iterations=None):
    batch_size = 16
    torch.manual_seed(1)
    device = torch.device("cpu")
    kwargs = {'batch_size': batch_size}

    dataset2 = CustomDataset(6000)
    test_loader = torch.utils.data.DataLoader(dataset2, **kwargs)
    if iterations is None:
        iterations = len(test_loader.dataset) // batch_size + 1
        print('-'*20, 'iterations', iterations, '-'*20)

    model.eval()
    test_loss = 0
    correct = 0
    iter_num = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            iter_num = iter_num + 1
            if iter_num == iterations:
                break

    print('iter_num', iter_num)
    data_length = iter_num * batch_size
    test_loss /= data_length
    acc = 100. * correct / data_length

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, data_length,
        100. * correct / data_length))

    return test_loss, acc

def run_inference_model_auto_cali(model, iterations=2):
    batch_size = 16
    torch.manual_seed(1)
    device = torch.device("cpu")
    kwargs = {'batch_size': batch_size}

    model.eval()
    test_loss = 0
    correct = 0
    iter_num = 0
    with torch.no_grad():
        for i in range(iterations):
            data = torch.tensor(np.random.uniform(0, 10, (32,1,28,28)).astype(np.float32))
            data = data.to(device)
            output = model(data)
            iter_num = iter_num + 1
            if iter_num == iterations:
                break


def run_inference_onnx(onnx_file, iterations=None):
    # prepare model
    print('onnx_file', onnx_file)
    onnx_model = onnx.load(onnx_file)
    onnx.checker.check_model(onnx_model)

    ort_session = onnxruntime.InferenceSession(onnx_file, providers=['CPUExecutionProvider'])
    input_names = [input_onnx.name for input_onnx in ort_session.get_inputs()]
    output_names = [output_onnx.name for output_onnx in ort_session.get_outputs()]
    print('inputs:', input_names)
    print('otputs:', output_names)

    def to_numpy(tensor):
       data_numpy = tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
       return data_numpy
    # prepare data
    batch_size = 16
    torch.manual_seed(1)
    device = torch.device("cpu")
    kwargs = {'batch_size': batch_size}

    dataset2 = CustomDataset(6000)
    test_loader = torch.utils.data.DataLoader(dataset2, **kwargs)
    if iterations is None:
        iterations = len(test_loader.dataset) // batch_size + 1
        print('-'*20, 'iterations', iterations, '-'*20)

    # run model
    test_loss = 0
    correct = 0
    iter_num = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            # run in onnxtime
            ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(data)}
            ort_outs = ort_session.run(output_names, ort_inputs)
            output = torch.Tensor(ort_outs[0])
            # cal acc
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            iter_num = iter_num + 1
            if iter_num == iterations:
                break

    print('iter_num', iter_num)
    data_length = iter_num * batch_size
    test_loss /= data_length
    acc = 100. * correct / data_length

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, data_length,
        100. * correct / data_length))

    return test_loss, acc
