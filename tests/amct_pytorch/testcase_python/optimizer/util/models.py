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
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

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
    conv(with bias) + bn
    depthwise_conv + bn
    depthwise_conv(with bais) + bn
    group_conv + bn
    group_conv(bias) + bn
    fc + bn
    fc(bias) + bn
    """
    def __init__(self):
        super(Net001,self).__init__()
        # conv + bn
        self.layer1 = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, bias=False),
            nn.BatchNorm2d(16))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, bias=True),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True))
        # depthwise_conv + bn
        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, groups=16),
            nn.BatchNorm2d(16))
        self.layer4 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, groups=16),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True))
        # group_conv + bn
        self.layer5 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, groups=4),
            nn.BatchNorm2d(32))
        self.layer6 = nn.Sequential(
            nn.Conv2d(32, 8, kernel_size=3, groups=8),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True))
        # fc
        self.fc = nn.Sequential(
            nn.Linear(8 * 16 * 16, 1024, bias=True),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128,10, bias=True))
        self.avg_pool = nn.AvgPool2d(kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)

        return x

class Quant(nn.Module):
    """ args_shape: [(1, 2, 28, 28)]
    """
    def __init__(self, scale, offset, quant_bit):
        super(Quant,self).__init__()
        self.scale = scale
        self.offset = offset
        self.quant_bit = quant_bit
        self.min_value = -2**(quant_bit-1)
        self.max_value = 2**(quant_bit-1) - 1

    def forward(self, data):
        data = torch.mul(data, self.scale)
        data = torch.round(data)
        data = torch.add(data, self.offset)
        data = torch.clamp(
            data,
            torch.tensor(self.min_value, dtype=torch.int64),
            torch.tensor(self.max_value, dtype=torch.int64))
        data = torch.sub(data, self.offset)

        return data


class Net3d(nn.Module):
    """ args_shape: [(1, 2, 4, 14, 14)]
    """
    def __init__(self):
        super(Net3d,self).__init__()
        # conv + bn
        self.layer1 = nn.Sequential(
            nn.Conv3d(2, 4, kernel_size=3, bias=False),
            nn.BatchNorm3d(4))


    def forward(self, x):
        x = self.layer1(x)

        return x
class MobilenetTailModel(torch.nn.Module):
    """ [2, 32, 28, 28]
    """
    def __init__(self):
        super(MobilenetTailModel,self).__init__()
        self.int_channels = 32
        self.last_channel = 64
        self.num_classes = 100
        self.features = nn.Sequential(
            nn.Conv2d(self.int_channels, self.last_channel, kernel_size=3, bias=False),
            nn.BatchNorm2d(self.last_channel),
            nn.ReLU6(inplace=False))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, self.num_classes),
        )

    def forward(self, x):
        x_dim4 = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x_dim4, (1, 1))
        x = torch.reshape(x, (x_dim4.shape[0], -1))
        # x = torch.reshape(x, (x.shape[0], -1))
        x = self.classifier(x)
        return x

class Yolov3ResizeModel(torch.nn.Module):
    """ [2, 32, 28, 28]
    """
    def __init__(self):
        super(Yolov3ResizeModel,self).__init__()
        self.int_channels = 32
        self.last_channel = 64
        self.num_classes = 100
        self.features = nn.Sequential(
            nn.Conv2d(self.int_channels, self.last_channel, kernel_size=3, bias=False),
            nn.BatchNorm2d(self.last_channel),
            nn.ReLU6(inplace=False))

    def forward(self, x):
        x = self.features(x)
        # x = nn.functional.interpolate(x, size=[52, 52])
        x = nn.functional.interpolate(x, scale_factor=2, mode='bicubic')
        return x

class ConvPadLong(torch.nn.Module):
    """ [2, 32, 28, 28]
    """
    def __init__(self):
        super(ConvPadLong,self).__init__()
        self.int_channels = 32
        self.last_channel = 64
        self.num_classes = 100
        self.features = nn.Sequential(
            nn.Conv2d(self.int_channels, self.last_channel, kernel_size=3, bias=False, padding_mode='replicate'),
            nn.BatchNorm2d(self.last_channel),
            nn.ReLU6(inplace=False))

        # circular to graph error

    def forward(self, x):
        x = self.features(x)
        return x