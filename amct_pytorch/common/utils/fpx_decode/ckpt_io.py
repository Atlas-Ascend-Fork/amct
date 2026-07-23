# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
"""Back-compat shim -- the generic checkpoint I/O layer moved to a shared location.

All checkpoint shard I/O now lives in ``amct_pytorch.common.utils.checkpoint_io`` (shared by
deploy, ptq unpack, and these fpx_decode converters). This module re-exports it so existing
``from ...fpx_decode.ckpt_io import X`` imports keep working; new code should import from
``amct_pytorch.common.utils.checkpoint_io`` directly.
"""
from amct_pytorch.common.utils.checkpoint_io import (  # noqa: F401
    load_index,
    resolve_index,
    invert_weight_map,
    write_index,
    save_shard,
    save_convert_output,
    parse_spec,
    load_tensor_from_spec,
    ShardCache,
    walk_checkpoint,
)
