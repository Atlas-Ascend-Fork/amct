# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
import torch.nn as nn


class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(64, 64)
        self.linear2 = nn.Linear(64, 32, bias=True)
        self.linear3 = nn.Linear(32, 1)

    def forward(self, inputs):
        x = self.linear1(inputs)
        x = self.linear2(x)
        x = self.linear3(x)
        return x


class TestModelBias(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(64, 64)
        self.linear2 = nn.Linear(64, 32, bias=False)
        self.linear3 = nn.Linear(32, 1)

    def forward(self, inputs):
        x = self.linear1(inputs)
        x = self.linear2(x)
        x = self.linear3(x)
        return x


class TestModelConv2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d1 = nn.Conv2d(32, 32, kernel_size=3, padding_mode='zeros')
        self.conv2d2 = nn.Conv2d(32, 64, kernel_size=3, padding_mode='zeros')
        self.conv2d3 = nn.Conv2d(64, 64, kernel_size=6, padding_mode='zeros')

    def forward(self, inputs):
        x = self.conv2d1(inputs)
        x = self.conv2d2(x)
        x = self.conv2d3(x)
        return x
