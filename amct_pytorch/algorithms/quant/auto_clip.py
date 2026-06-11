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

from amct_pytorch.algorithms.registry_factory import ALGO_REGISTRY


@ALGO_REGISTRY.register(
    name="lwc",
    description="Learnable weight clipping",
    targets=("weight",),
)
class LWC(torch.nn.Module):
    def __init__(self, args, w_bits=None):
        super().__init__()
        self.args = args
        self.w_size = args.w_size
        self.quant_dtype = args.quant_dtype
        self.init_value = 4.
        self._update_clip_dim()
        self.sigmoid = torch.nn.Sigmoid()
        self.clip_factor_max = torch.nn.Parameter(torch.ones((self.clip_dim, 1)) * self.init_value, requires_grad=True)
        self.clip_factor_min = torch.nn.Parameter(torch.ones((self.clip_dim, 1)) * self.init_value, requires_grad=True)

    def trainable_params(self):
        return [self.clip_factor_min, self.clip_factor_max]

    def export_ptq_params(self):
        return {
            "clip_factor_min": self.clip_factor_min.detach().cpu(),
            "clip_factor_max": self.clip_factor_max.detach().cpu(),
        }

    def load_ptq_params(self, params):
        self.clip_factor_min.data.copy_(
            params["clip_factor_min"].to(device=self.clip_factor_min.device, dtype=self.clip_factor_min.dtype)
        )
        self.clip_factor_max.data.copy_(
            params["clip_factor_max"].to(device=self.clip_factor_max.device, dtype=self.clip_factor_max.dtype)
        )

    def apply_clip(self, x):
        ori_shape = x.shape
        if self.quant_dtype == "mxfp":
            x = x.view(-1, 32)
        cur_min, cur_max = x.min(1, keepdim=True)[0], x.max(1, keepdim=True)[0]
        cur_max *= self.sigmoid(self.clip_factor_max.to(x.device))
        cur_min *= self.sigmoid(self.clip_factor_min.to(x.device))
        x = torch.clamp(x, min=cur_min, max=cur_max)
        return x.view(*ori_shape)

    def forward(self, x):
        return self.apply_clip(x)

    def _update_clip_dim(self):
        if self.quant_dtype == "mxfp":
            self.clip_dim = self.w_size[0] * self.w_size[1] // 32
        else:
            self.clip_dim = self.w_size[0]


@ALGO_REGISTRY.register(
    name="lac",
    description="Learnable activation clipping",
    targets=("activation",),
)
class LAC(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.is_observe = False
        self.is_per_tensor = args.is_per_tensor
        self.init_value = 4.
        self.sigmoid = torch.nn.Sigmoid()
        self.clip_factor_max = torch.nn.Parameter(torch.ones((1,)) * self.init_value, requires_grad=True)
        self.clip_factor_min = torch.nn.Parameter(torch.ones((1,)) * self.init_value, requires_grad=True)
        self.register_buffer(f'maxval', torch.zeros((1)))
        self.register_buffer(f'minval', torch.zeros((1)))

    def trainable_params(self):
        return [self.clip_factor_min, self.clip_factor_max]

    def export_ptq_params(self):
        return {
            "clip_factor_min": self.clip_factor_min.detach().cpu(),
            "clip_factor_max": self.clip_factor_max.detach().cpu(),
            "maxval": self.maxval.detach().cpu(),
            "minval": self.minval.detach().cpu(),
        }

    def load_ptq_params(self, params):
        self.clip_factor_min.data.copy_(
            params["clip_factor_min"].to(device=self.clip_factor_min.device, dtype=self.clip_factor_min.dtype)
        )
        self.clip_factor_max.data.copy_(
            params["clip_factor_max"].to(device=self.clip_factor_max.device, dtype=self.clip_factor_max.dtype)
        )
        self.maxval.copy_(params["maxval"].to(device=self.maxval.device, dtype=self.maxval.dtype))
        self.minval.copy_(params["minval"].to(device=self.minval.device, dtype=self.minval.dtype))

    def apply_clip(self, x):
        if self.is_per_tensor:
            init_shape = x.shape
            cur_max = self.maxval.clone()
            cur_min = self.minval.clone()
        else:
            init_shape = x.shape
            x = x.reshape((-1, x.shape[-1]))
            cur_max, cur_min = x.amax(1, keepdim=True), x.amin(1, keepdim=True)
            tmp = torch.zeros_like(cur_max)
            cur_max, cur_min = torch.maximum(cur_max, tmp), torch.minimum(cur_min, tmp)
        cur_max *= self.sigmoid(self.clip_factor_max.to(x.device))
        cur_min *= self.sigmoid(self.clip_factor_min.to(x.device))
        x = torch.clamp(x, min=cur_min, max=cur_max)
        x = x.reshape(init_shape)
        return x

    def forward(self, x):
        if self.is_observe:
            if x.max() > self.maxval.to(x.device):
                self.maxval.data = x.max()
            if x.min() < self.minval.to(x.device):
                self.minval.data = x.min()
            return x
        return self.apply_clip(x)
