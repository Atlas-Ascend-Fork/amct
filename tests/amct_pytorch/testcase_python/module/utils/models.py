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
import torch
import torch.nn as nn
import torch.nn.functional as F

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
