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

import functools

import torch.nn as nn


class Catcher(nn.Module):
    def __init__(self, module, dataset):
        super().__init__()
        self.module = module
        self.dataset = dataset
        self.index = 0
        self.layer_type = getattr(module, "layer_type", "linear_attention")
        self.attention_type = getattr(module, "attention_type", None)
        self.attention_mask = None
        self.position_ids = None
        self.position_embeddings = None

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

    def forward(self, inp, **kwargs):
        self.dataset.append(inp.to("cpu"))
        self.index += 1
        if self.attention_mask is None and "attention_mask" in kwargs:
            self.attention_mask = kwargs["attention_mask"]
        if self.position_ids is None and "position_ids" in kwargs:
            self.position_ids = kwargs["position_ids"]
        if self.position_embeddings is None and "position_embeddings" in kwargs:
            self.position_embeddings = kwargs["position_embeddings"]
        raise ValueError


def _append_capture_tensor(act_stat, name, tensor, tensor_type):
    tensor = tensor.detach().cpu()
    key_name = f"{name}_{tensor_type}"
    if key_name in act_stat:
        act_stat[key_name].append(tensor)
    else:
        act_stat[key_name] = [tensor]


def _stat_input_hook(module, inputs, output, *, name, act_stat):
    _append_capture_tensor(act_stat, name, output, "out")


def register_forward_hooks(block, target_name, hooks, act_stat):
    for name, module in block.named_modules():
        if target_name in name:
            hooks.append(
                module.register_forward_hook(
                    functools.partial(_stat_input_hook, name=name, act_stat=act_stat)
                )
            )
