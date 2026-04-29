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
"""
Generate model for ut.
"""
from __future__ import print_function
import argparse  #Python 命令行解析工具
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy

def create_onnx(model, args_shapes, onnx_file, mode='eval'):
    """ save onnx """
    args = list()
    for input_shape in args_shapes:
        args.append(torch.randn(input_shape))

    args = tuple(args)
    if mode == 'eval':
        # set the model to inference mode
        model.eval()
    elif mode == 'train':
        model.train()
    else:
        raise RuntimeError("param mode error!")

    torch_in = args[0]
    torch_out = model(torch_in)
    torch.onnx.export(
        model,
        args,
        onnx_file,
        opset_version=11,
        do_constant_folding=False,   # 是否执行常量折叠优化
        )
        # input_names=["input"],  # 输入名
        # output_names=["output"],    # 输出名
        # dynamic_axes={"input":{0:"batch_size"},  # 批处理变量
        #               "output":{0:"batch_size"}})

    return torch_in, torch_out



def save_state_dict(model, name):
    torch.save(model.state_dict(), name)

def restore_model(model, state_dict_path):
    model.load_state_dict(torch.load(state_dict_path))

class Net001(nn.Module):
    """ args_shape: [(1, 2, 28, 28)]
    conv + bn
    conv + relu
    conv + bn + relu
    """
    def __init__(self):
        super(Net001,self).__init__()
        # conv + bn
        self.conv1 = nn.Conv2d(2, 16, kernel_size=3, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        # conv + relu
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, bias=False)
        self.relu1 = nn.ReLU()
        # conv + bn + relu
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        # conv_transpose
        self.conv_transpose = nn.ConvTranspose2d(32, 32, kernel_size=3, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.relu1(x)
        x = self.conv3(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv_transpose(x)

        return x

