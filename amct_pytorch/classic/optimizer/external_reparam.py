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
"""External reparameterization executor for the convert Pass.

A calibration module may declare out-of-module convert side effects via
``get_external_reparam()`` — a list of ``(host_name, trans)`` pairs, e.g.
flatquant's diag scale merging into the preceding Norm's weight. The Pass
(replace_npu_quant_pass) walks to the parent layer, resolves the declared
host attribute and calls :func:`reparameterize_ln` here.

Transform-object contract for ``reparameterize_ln`` (duck-typed; FlatQuant's
``SVDDecomposeTransMatrix`` is the current implementation):

- ``add_diag`` (bool): whether the transform carries a per-channel diag scale
  to merge; transforms without it are a no-op;
- ``use_diag`` (bool): still-active flag — the merge runs while it is True
  and the module-side weight bake (its ``reparameterize``) turns it off only
  AFTER baking, so the deploy identity ``(W/d @ M^-T) @ (M @ (x*c)) == W @ x``
  holds. This executor therefore must NOT flip it;
- ``diag_scale`` (Tensor, shape (n,)): the per-channel scale to merge.
"""

import torch


def reparameterize_ln(ln, trans):
    """Merge the transform's diag scale into the preceding Norm.

    gamma <- gamma * diag(c); exact identity (elementwise associativity):
    (x*gamma)*c == x*(gamma*c). fp64 intermediate precision so the merge is
    lossless in bf16/fp16 carriers.
    """
    if trans is None or not getattr(trans, 'add_diag', False) or not trans.use_diag:
        return
    ln_weight = ln.weight.data
    ori_dtype = ln_weight.dtype
    ln_weight = ln_weight.to(torch.float64)
    ln_weight = ln_weight * trans.diag_scale.to(torch.float64)
    ln.weight.data = ln_weight.to(ori_dtype)
    # The bias must scale with the weight or the merge is inexact:
    # LN' = ((x-mu)/sigma)*(gamma*c) + (beta*c) == c * LN(x), while a
    # weight-only merge leaves ((x-mu)/sigma*gamma)*c + beta and breaks
    # the deploy identity whenever beta != 0.
    ln_bias = getattr(ln, 'bias', None)
    if ln_bias is not None:
        ori_dtype = ln_bias.dtype
        ln_bias = ln_bias.to(torch.float64)
        ln_bias = ln_bias * trans.diag_scale.to(torch.float64)
        ln.bias.data = ln_bias.to(ori_dtype)
