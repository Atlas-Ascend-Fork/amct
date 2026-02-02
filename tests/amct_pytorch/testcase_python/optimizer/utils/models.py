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
from amct_pytorch.amct_pytorch_inner.amct_pytorch.common.utils.util import version_higher_than

def create_onnx(model, args_shapes, onnx_file, mode='eval'):
    """ save onnx """
    args = list()
    for input_shape in args_shapes:
        args.append(torch.randn(input_shape))
    args = tuple(args)

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
        affine = version_higher_than(torch.__version__, '2.1.0')
        # conv + bn
        self.layer1 = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, bias=False),
            nn.BatchNorm2d(16))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, bias=True),
            nn.BatchNorm2d(16, affine=affine, track_running_stats=True),
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


class Conv2dLinear(nn.Module):
    """ not do prune"""
    def __init__(self):
        super().__init__()
        # fc
        self.layer1 = nn.Conv2d(3, 160, kernel_size=3, bias=True)
        self.layer2 = nn.BatchNorm2d(160)
        self.layer3 = nn.Linear(14, 80, bias=False)
        self.layer4 = nn.BatchNorm2d(160)
        self.layer5 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x


class Net002(nn.Module):
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
        super(Net002, self).__init__()

        self.branch1 = nn.Conv2d(2, 16, kernel_size=3, bias=False)
        self.branch2 = nn.Conv2d(2, 16, kernel_size=3, bias=False)
        self.branch3 = nn.Conv2d(2, 16, kernel_size=3, bias=False)
        self.branch4 = nn.Conv2d(2, 16, kernel_size=3, bias=False)

        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(64, 16, kernel_size=3, bias=False)
        self.bn = nn.BatchNorm2d(16)
        self.linear = nn.Linear(384 * 24, 10, bias=True)

    def forward(self, x):
        branch_1 = self.branch1(x)
        branch_2 = self.branch2(x)
        branch_3 = self.branch3(x)
        branch_4 = self.branch4(x)
        x = torch.cat([branch_1, branch_2, branch_3, branch_4], 1)
        x = self.relu(x)
        x = self.conv(x)
        x = self.bn(x)
        x = x.view(x.size(0),-1)
        x = self.linear(x)
        return x


class Net003(nn.Module):
    """ args_shape: [(1, 2, 28, 28)]
    """
    def __init__(self):
        super(Net003, self).__init__()

        self.conv = nn.Conv2d(2, 16, kernel_size=3, bias=False)
        self.bn = nn.SyncBatchNorm(16)
        self.linear = nn.Linear(384 * 24, 10, bias=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = x.view(x.size(0),-1)
        x = self.linear(x)
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
    def __init__(self):
        super(Net3d001,self).__init__()
        # conv + bn
        self.layer1 = nn.Sequential(
            nn.Conv3d(2, 4, kernel_size=3, bias=False),
            nn.BatchNorm3d(4))
        self.layer2 = nn.ConvTranspose3d(4, 4, kernel_size=3,
            padding_mode='zeros')

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)

        return x

class Net1d(nn.Module):
    """ args_shape: [(1, 2, 14)]
    """
    def __init__(self):
        super(Net1d,self).__init__()
        self.args_shape = [(1, 2, 14)]
        # conv + bn
        self.layer1 = nn.Sequential(
            nn.Conv1d(2, 2, kernel_size=1, bias=False),
            nn.BatchNorm1d(2))

    def forward(self, x):
        x = self.layer1(x)

        return x