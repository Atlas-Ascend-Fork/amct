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
        self.avg_pool = nn.AvgPool2d(kernel_size=1, stride=1, padding=0)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # fc
        self.fc = nn.Sequential(
            nn.Linear(8 * 16 * 16, 1024, bias=True),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128,10, bias=True))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)
        x = self.layer5(x)
        x = self.layer6(x)
        y = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)

        return x, y

class Net001Sub(nn.Module):
    """ args_shape: [(1, 2, 28, 28)]
    conv + bn
    conv(with bias) + bn
    depthwise_conv + bn
    """
    def __init__(self):
        super(Net001Sub,self).__init__()
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

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

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

class Net3d001(nn.Module):
    """ args_shape: [(1, 2, 4, 14, 14)]
    """
    def __init__(self):
        super(Net3d001,self).__init__()
        self.args_shape = [(1, 2, 4, 14, 14)]
        # conv + bn
        self.layer1 = nn.Sequential(
            nn.Conv3d(2, 4, kernel_size=[1, 1, 1], bias=False, dilation=[1, 1, 1]),
            nn.BatchNorm3d(4))

    def forward(self, x):
        x = self.layer1(x)

        return x

class Net3d002(nn.Module):
    """ args_shape: [(1, 2, 4, 14, 14)]
    """
    def __init__(self):
        super(Net3d002,self).__init__()
        self.args_shape = [(1, 2, 4, 14, 14)]
        # conv + bn
        self.layer1 = nn.Sequential(
            nn.Conv3d(2, 4, kernel_size=[1, 1, 1], bias=False, dilation=1),
            nn.BatchNorm3d(4))

    def forward(self, x):
        x = self.layer1(x)

        return x

class Net3d003(nn.Module):
    """ args_shape: [(1, 2, 4, 14, 14)]
    """
    def __init__(self):
        super(Net3d003,self).__init__()
        self.args_shape = [(1, 2, 4, 14, 14)]
        # conv + bn
        self.layer1 = nn.Sequential(
            nn.Conv3d(2, 4, kernel_size=[1, 1, 1], bias=False, dilation=[3, 1, 1]),
            nn.BatchNorm3d(4))

    def forward(self, x):
        x = self.layer1(x)

        return x

class Net3d004(nn.Module):
    """ args_shape: [(1, 2, 4, 14, 14)]
    """
    def __init__(self):
        super(Net3d004,self).__init__()
        self.args_shape = [(1, 2, 4, 14, 14)]
        # conv + bn
        self.layer1 = nn.Sequential(
            nn.Conv3d(2, 4, kernel_size=[1, 1, 1], bias=False, dilation=3),
            nn.BatchNorm3d(4))

    def forward(self, x):
        x = self.layer1(x)

        return x

class Net002(nn.Module):
    """ args_shape: [(1, 2, 28, 28)]
    special Conv2d
    conv(dilation =2) + bn
    conv(dilation = [3, 3]) + bn
    """
    def __init__(self):
        super(Net002,self).__init__()
        # conv + bn
        self.layer1 = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, bias=False, dilation=2),
            nn.BatchNorm2d(16))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, bias=True, dilation=[3, 3]),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True))


    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)

        return x

class SingleConv(nn.Module):
    def __init__(self):
        super(SingleConv, self).__init__()
        self.args_shape = [(1, 16, 17, 17)]
        self.layer1 = nn.Conv2d(16, 16, kernel_size=[1, 7], stride=[1, 1],
            padding_mode='replicate', padding=[0, 3], dilation=(1, 1),
            groups=1, bias=False)
        self.layer2 = nn.ConvTranspose2d(16, 8, kernel_size=[3, 3],
            padding_mode='zeros')

    def forward(self, x):
        x = self.layer1(x)
        output = self.layer2(x)
        return output

class MatmulDim(nn.Module):
    def __init__(self):
        super(MatmulDim, self).__init__()
        self.args_shape = [(1, 4, 16, 16)]
        self.layer1 = nn.Linear(16, 8, bias=False)
        self.layer2 = nn.Linear(8, 1, bias=False)
        if "1.10" not in torch.__version__:
            self.weight1 = torch.randn([1])
        else:
            self.weight1 = torch.randn([1, 1])

    def forward(self, in_data):
        x = self.layer1(in_data)
        x = self.layer2(x)
        output = torch.nn.functional.linear(x, self.weight1)

        return output

class TorchConv3dShareWeightModel(nn.Module):
    '''conv3d|conv3d  shared weight'''
    def __init__(self,in_channels=3,out_channels=3, kernel_size=(2,3,3), \
        stride=(1,2,2), padding=(0,1,1), dilation=(1,1,1), groups=1,bias=True):
        super(TorchConv3dShareWeightModel, self).__init__()
        self.conv1 =  nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, \
            stride=stride, padding=padding,dilation=dilation, groups=groups,bias=bias)
        self.conv2 =  nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, \
            stride=stride, padding=padding,dilation=dilation, groups=groups,bias=bias)
        self.conv3 = copy.copy(self.conv2) #浅拷贝 指向相同 weight共享
        # self.conv3 = copy.deepcopy(self.conv2) #浅拷贝 指向相同 weight共享

    def forward(self, x):
        x = self.conv1(x)
        y1 = self.conv2(x)
        y2= self.conv3(x)
        z = torch.cat((y1,y2),0)

        return z

class LinearNet(nn.Module):
    def __init__(self):
        super(LinearNet, self).__init__()
        self.linear0 = nn.Linear(16, 8, bias=False)
        self.linear1 = nn.Linear(8, 2, bias=True)

    def forward(self, x):
        x1 = self.linear0(x)
        x2 = self.linear1(x1)
        output = x2

        return output
