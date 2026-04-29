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


class Net4dMatmul(nn.Module):
    """ args_shape: [(1, 2, 28, 28)]
    fc + bn
    fc(bias) + bn
    """
    def __init__(self):
        super(Net4dMatmul,self).__init__()
        # fc
        self.fc = nn.Sequential(
            nn.Linear(28, 1024, bias=True),
            nn.Linear(1024, 128, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(128,10, bias=True))

    def forward(self, x):
        # x = x.view(-1, 28)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)

        return x

class EltwiseConv(nn.Module):
    """args_shape: [(1, 16, 28, 28)]
    """
    def __init__(self):
        super(EltwiseConv, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=160, groups=2, kernel_size=3, bias=True, padding_mode='circular'),
            nn.BatchNorm2d(160))
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=160, groups=2, kernel_size=3, bias=True),
            nn.BatchNorm2d(160))
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=160, groups=2, kernel_size=3, bias=True),
            nn.BatchNorm2d(160))
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=160, groups=2, kernel_size=3, bias=True),
            nn.BatchNorm2d(160))
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=160, out_channels=80, groups=2, kernel_size=3, bias=True),
            nn.BatchNorm2d(80))
        self.layer6 = nn.Sequential(
            nn.Conv2d(in_channels=160, out_channels=80, groups=2, kernel_size=3, bias=True),
            nn.BatchNorm2d(80))

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x)
        x_1_2 = x1+x2
        x3 = self.layer3(x)
        x_1_2_3 = x_1_2 + x3
        x4 = self.layer4(x)
        x_1_2_4 = x_1_2+x4
        y1 = self.layer5(x_1_2_3)
        y2 = self.layer6(x_1_2_4)
        return y1, y2


class LinearIn2(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Linear(16, 1024, bias=True)
        self.layer2 = nn.BatchNorm1d(1024)
        self.layer3 = nn.Linear(1024, 128, bias=False)
        self.layer4 = nn.BatchNorm1d(128)
        self.layer5 = nn.ReLU(inplace=True)
        self.layer6 = nn.Linear(128,10, bias=True)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)

        return x

class LinearIn3(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(16, 1024, bias=True)
        self.layer2 = nn.BatchNorm1d(16)
        self.layer3 = nn.Linear(1024, 128, bias=False)
        self.layer4 = nn.BatchNorm1d(16)
        self.layer5 = nn.ReLU(inplace=True)
        self.layer6 = nn.Linear(128,10, bias=True)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)

        return x

class LinearIn4(nn.Module):
    def __init__(self):
        super().__init__()
        # fc
        self.layer1 = nn.Linear(16, 1024, bias=True)
        self.layer2 = nn.BatchNorm2d(3)
        self.layer3 = nn.Linear(1024, 128, bias=False)
        self.layer4 = nn.BatchNorm2d(3)
        self.layer5 = nn.ReLU(inplace=True)
        self.layer6 = nn.Linear(128,10, bias=True)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)

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

class LinearConv2d(nn.Module):
    """ not do prune"""
    def __init__(self):
        super().__init__()
        # fc
        self.layer1 = nn.Linear(16, 80, bias=False)
        self.layer2 = nn.BatchNorm2d(3)
        self.layer3 = nn.Conv2d(3, 160, kernel_size=3, bias=True)
        self.layer4 = nn.BatchNorm2d(160)
        self.layer5 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x

class LinearAddConv2d(nn.Module):
    """ not do prune"""
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(64, 64, bias=False)
        self.layer2 = nn.BatchNorm2d(32)
        self.layer3 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=True)
        self.layer4 = nn.BatchNorm2d(32)
        self.layer5 = nn.ReLU(inplace=True)

    def forward(self, x):
        x_1 = self.layer1(x)
        x_1 = self.layer2(x_1)
        x_2 = self.layer3(x)
        x_2 = self.layer4(x_2)
        x_2 = self.layer5(x_2)
        out = x_1 + x_2
        return out

class LinearConcatConv2d(nn.Module):
    """ not do prune"""
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(64, 64, bias=False)
        self.layer2 = nn.BatchNorm2d(32)
        self.layer3 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=True)
        self.layer4 = nn.BatchNorm2d(32)
        self.layer5 = nn.ReLU(inplace=True)

    def forward(self, x):
        x_1 = self.layer1(x)
        x_1 = self.layer2(x_1)
        x_2 = self.layer3(x)
        x_2 = self.layer4(x_2)
        x_2 = self.layer5(x_2)
        out = torch.cat([x_1, x_2], 1)

        return out

class ConcatDim0Conv(nn.Module):
    """args_shape: [(1, 16, 28, 28)]
    """
    def __init__(self):
        super(ConcatDim0Conv, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=160, groups=1, kernel_size=3, bias=True),
            nn.BatchNorm2d(160))
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=160, groups=2, kernel_size=3, bias=True),
            nn.BatchNorm2d(160))
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=160, out_channels=8, groups=1, kernel_size=3, bias=True),
            nn.BatchNorm2d(8))

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x)
        x_1_2 = torch.cat((x1, x2))
        x3 = self.layer3(x_1_2)
        return x3


class GroupConv(nn.Module):
    """args_shape: [(1, 16, 28, 28)]
    """
    def __init__(self):
        super(GroupConv, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=64, groups=1, kernel_size=3, bias=True),
            nn.BatchNorm2d(64))
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, groups=2, kernel_size=3, bias=True),
            nn.BatchNorm2d(64))
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, groups=32, kernel_size=3, bias=True),
            nn.BatchNorm2d(64))
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=8, groups=1, kernel_size=3, bias=True),
            nn.BatchNorm2d(8))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

class NetConvDeconv(nn.Module):
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
        super(NetConvDeconv,self).__init__()
        # conv + bn
        self.layer1 = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, bias=False),
            nn.BatchNorm2d(16))
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(16, 16, kernel_size=3, bias=True),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True))
        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, bias=True),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True))


    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x


class DefaultNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=6, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.conv2 = copy.deepcopy(self.conv1) 
        self.conv3 = nn.Conv2d(in_channels=6, out_channels=6, kernel_size=3, stride=1, padding=1, dilation=2, groups=1, bias=True, padding_mode='zeros')
        self.conv4 = nn.Conv2d(in_channels=6, out_channels=6, kernel_size=3, stride=1, padding=1, dilation=1, groups=2, bias=True, padding_mode='zeros')
        self.conv5 = nn.Conv2d(in_channels=6, out_channels=6, kernel_size=3, stride=1, padding=1, dilation=1, groups=3, bias=True, padding_mode='zeros')
        self.conv6 = nn.Conv2d(in_channels=6, out_channels=6, kernel_size=3, stride=1, padding=1, dilation=1, groups=6, bias=True, padding_mode='zeros')
        self.conv7 = copy.deepcopy(self.conv1)
        self.conv8 = copy.deepcopy(self.conv1)
        self.conv9 = nn.Conv2d(in_channels=6, out_channels=6, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True, padding_mode='replicate')
        self.conv10 = copy.deepcopy(self.conv1)

        self.bn1 = nn.BatchNorm2d(6)
        self.bn2 = copy.deepcopy(self.bn1)
        self.bn3 = copy.deepcopy(self.bn1)
        self.bn4 = copy.deepcopy(self.bn1)
        self.bn5 = copy.deepcopy(self.bn1)
        self.bn6 = copy.deepcopy(self.bn1)
        self.bn7 = copy.deepcopy(self.bn1)
        self.bn8 = copy.deepcopy(self.bn1)
        self.bn9 = copy.deepcopy(self.bn1)
        self.bn10 = copy.deepcopy(self.bn1)
        self.bn11 = copy.deepcopy(self.bn1)

        self.linear1 = nn.Linear(in_features=32, out_features=32, bias=False)
        self.linear2 = copy.deepcopy(self.linear1)
        self.linear3 = copy.deepcopy(self.linear1)
        self.linear4 = copy.deepcopy(self.linear1)

        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None)
        self.avgpool2 = copy.deepcopy(self.avgpool1)
        self.avgpool3 = copy.deepcopy(self.avgpool1)
        self.avgpool4 = nn.AvgPool2d(kernel_size=8, stride=1, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None)
        self.avgpool5 = nn.AvgPool2d(kernel_size=1, stride=1, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None)

        self.deconv1 = nn.ConvTranspose2d(6, 6, kernel_size=3, stride=1, padding=1, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros')
        self.deconv2 = nn.ConvTranspose2d(6, 6, kernel_size=3, stride=1, padding=1, output_padding=0, groups=1, bias=True, dilation=2, padding_mode='zeros')
        self.deconv3 = nn.ConvTranspose2d(6, 6, kernel_size=3, stride=1, padding=1, output_padding=0, groups=2, bias=True, dilation=1, padding_mode='zeros')
        self.deconv4 = nn.ConvTranspose2d(6, 6, kernel_size=3, stride=1, padding=1, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros')
        self.deconv5 = nn.ConvTranspose2d(6, 6, kernel_size=3, stride=1, padding=1, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros')
        self.deconv6 = nn.ConvTranspose2d(6, 6, kernel_size=3, stride=1, padding=1, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros')

    def forward(self, x0):
        x = self.conv1(x0)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.conv6(x)
        x = self.bn6(x)
        m = self.conv7(x)
        m = self.bn7(m)
        n = self.conv8(x)
        n = self.bn8(n)
        x = torch.cat((m, n), 0)
        x = self.conv9(x)
        x = self.bn9(x)
        x = self.conv10(x)
        y1 = self.bn10(x)

        x = self.linear1(x0)
        x = self.bn11(x)
        m = self.linear2(x)
        n = self.linear3(x)
        x = torch.cat((m, n), 0)
        y2 = self.linear4(x)

        x = self.avgpool1(x0)
        m = self.avgpool2(x)
        n = self.avgpool3(x)
        x = torch.cat((m, n), 0)
        x = self.avgpool4(x)
        y3 = self.avgpool5(x)

        x = self.deconv1(x0)
        x = self.deconv2(x)
        x = self.deconv3(x)
        m = self.deconv4(x)
        n = self.deconv5(x)
        x = torch.cat((m, n), 0)
        y4 = self.deconv6(x)

        return y1, y2, y3, y4