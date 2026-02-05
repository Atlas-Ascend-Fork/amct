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
import torch
import torch.nn as nn

class SingleLinear(nn.Module):
    """args_shape: [(1, 16, 28, 28)]
    """
    def __init__(self):
        super(SingleConv, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_features=16, out_features=64, bias=True),
            nn.BatchNorm2d(64))
        self.layer2 = nn.Sequential(
            nn.Linear(in_features=64, out_features=64, ),
            nn.BatchNorm2d(64))
        self.layer3 = nn.Sequential(
            nn.Linear(in_features=64, out_features=48,),
            nn.BatchNorm2d(48))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class SingleConv(nn.Module):
    """args_shape: [(1, 16, 28, 28)]
    """
    def __init__(self):
        super(SingleConv, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=64, groups=1, kernel_size=3, bias=True),
            nn.BatchNorm2d(64))
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=48, groups=1, kernel_size=3, bias=True),
            nn.BatchNorm2d(48))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

class SingleDepthwsieConv(nn.Module):
    """args_shape: [(1, 16, 28, 28)]
    """
    def __init__(self):
        super(SingleDepthwsieConv, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, groups=16, kernel_size=3, bias=True),
            nn.BatchNorm2d(16))
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, groups=1, kernel_size=3, bias=True),
            nn.BatchNorm2d(8))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class GroupConv(nn.Module):
    """args_shape: [(1, 16, 28, 28)]
    """
    def __init__(self):
        super(GroupConv, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=64, groups=1, kernel_size=3, bias=True),
            nn.BatchNorm2d(64))
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, groups=2, kernel_size=3, bias=True),
            nn.BatchNorm2d(128))
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=8, groups=1, kernel_size=3, bias=True),
            nn.BatchNorm2d(8))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

class SingleGroupConv(nn.Module):
    """args_shape: [(1, 16, 28, 28)]
    """
    def __init__(self):
        super(SingleGroupConv, self).__init__()
        self.layer1 = nn.Conv2d(in_channels=16, out_channels=64, groups=4, kernel_size=3, bias=True)


    def forward(self, x):
        x = self.layer1(x)
        return x

class SingleDeconv(nn.Module):
    """args_shape: [(1, 16, 28, 28)]
    """
    def __init__(self):
        super(SingleDeconv, self).__init__()
        self.layer1 = nn.ConvTranspose2d(in_channels=16, out_channels=16, groups=4, kernel_size=3, bias=False)

    def forward(self, x):
        x = self.layer1(x)
        return x

class MultiDeconv(nn.Module):
    """args_shape: [(1, 16, 28, 28)]
    """
    def __init__(self):
        super(MultiDeconv, self).__init__()
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=16, out_channels=64, groups=1, kernel_size=3, bias=True),
            nn.BatchNorm2d(64))
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, groups=32, kernel_size=3, bias=True),
            nn.BatchNorm2d(32))
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=8, groups=1, kernel_size=3, bias=True),
            nn.BatchNorm2d(8))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

class ConcatConv(nn.Module):
    """args_shape: [(1, 16, 28, 28)]
    """
    def __init__(self):
        super(ConcatConv, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=48, groups=1, kernel_size=3, bias=True),
            nn.BatchNorm2d(48))
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=128, groups=2, kernel_size=3, bias=True),
            nn.BatchNorm2d(128))
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=176, out_channels=8, groups=1, kernel_size=3, bias=True),
            nn.BatchNorm2d(8))

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x)
        x_1_2 = torch.cat((x1, x2), dim=1)
        x3 = self.layer3(x_1_2)
        return x3

class EltwiseConv(nn.Module):
    """args_shape: [(1, 16, 28, 28)]
    """
    def __init__(self):
        super(EltwiseConv, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=160, groups=2, kernel_size=3, bias=True),
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

class SplitConv(nn.Module):
    """args_shape: [(1, 16, 28, 28)]
    """
    def __init__(self):
        super(SplitConv, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=160, groups=2, kernel_size=3, bias=True),
            nn.BatchNorm2d(160))
        self.layer2_1 = nn.Sequential(
            nn.Conv2d(in_channels=80, out_channels=160, groups=2, kernel_size=3, bias=True),
            nn.BatchNorm2d(160))
        self.layer2_2 = nn.Sequential(
            nn.Conv2d(in_channels=80, out_channels=80, groups=2, kernel_size=3, bias=True),
            nn.BatchNorm2d(80))


    def forward(self, x):
        x1 = self.layer1(x)
        print(x1.shape)
        x1 = torch.split(x1, 80, dim=1)
        print(x1[0].shape, x1[0].shape)
        x2_1 = self.layer2_1(x1[0])
        x2_2 = self.layer2_2(x1[1])

        return x2_1, x2_2

class SplitConcatGroupConv(nn.Module):
    """args_shape: [(1, 16, 28, 28)]
    """
    def __init__(self):
        super(SplitConcatGroupConv, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=160, groups=1, kernel_size=3, bias=True),
            nn.BatchNorm2d(160))
        self.layer2_1 = nn.Sequential(
            nn.Conv2d(in_channels=80, out_channels=160, groups=2, kernel_size=3, bias=True),
            nn.BatchNorm2d(160))
        self.layer2_2 = nn.Sequential(
            nn.Conv2d(in_channels=80, out_channels=80, groups=2, kernel_size=3, bias=True),
            nn.BatchNorm2d(80))
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=240, out_channels=16, groups=2, kernel_size=3, bias=True),
            nn.BatchNorm2d(16))

    def forward(self, x):
        x1 = self.layer1(x)
        print(x1.shape)
        x1 = torch.split(x1, 80, dim=1)
        print(x1[0].shape, x1[0].shape)
        x2_1 = self.layer2_1(x1[0])
        x2_2 = self.layer2_2(x1[1])
        x2 = torch.cat([x2_1, x2_2], 1)
        x3 = self.layer3(x2)

        return x3

class SplitConcatConv(nn.Module):
    """args_shape: [(1, 16, 28, 28)]
    """
    def __init__(self):
        super(SplitConcatConv, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=160, groups=1, kernel_size=3, bias=True),
            nn.BatchNorm2d(160))
        self.layer2_1 = nn.Sequential(
            nn.Conv2d(in_channels=80, out_channels=160, groups=2, kernel_size=3, bias=True),
            nn.BatchNorm2d(160))
        self.layer2_2 = nn.Sequential(
            nn.Conv2d(in_channels=80, out_channels=80, groups=2, kernel_size=3, bias=True),
            nn.BatchNorm2d(80))
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=240, out_channels=16, groups=1, kernel_size=3, bias=True),
            nn.BatchNorm2d(16))

    def forward(self, x):
        x1 = self.layer1(x)
        print(x1.shape)
        x1 = torch.split(x1, 80, dim=1)
        print(x1[0].shape, x1[0].shape)
        x2_1 = self.layer2_1(x1[0])
        x2_2 = self.layer2_2(x1[1])
        x2 = torch.cat([x2_1, x2_2], 1)
        x3 = self.layer3(x2)

        return x3
class AvgpoolFlatten(nn.Module):
    """args_shape: [(1, 16, 28, 28)]
    """
    def __init__(self):
        super(AvgpoolFlatten, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, groups=1, kernel_size=3, bias=True),
            nn.BatchNorm2d(32))
        self.layer2 = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.layer3 = nn.Linear(in_features=32, out_features=10, bias=True)

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x2 = torch.flatten(x2, start_dim=1)
        x3 = self.layer3(x2)

        return x3

class NetTrainBranch(nn.Module):
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
        super(NetTrainBranch,self).__init__()
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
            nn.Conv2d(16, 16, kernel_size=3, bias=True),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True))


    def forward(self, x):
        x = self.layer1(x)
        if self.training:
            x = self.layer2(x)
        else:
            x = self.layer3(x)

        return x