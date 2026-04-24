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
from ...amct_pytorch.common.graph_base.anchor_base import InputAnchorBase
from ...amct_pytorch.common.graph_base.anchor_base import OutputAnchorBase


class InputAnchor(InputAnchorBase):
    """
    Function: Data structure of anchor which contains graph topologic info
    APIs: node, index, name, set_name, add_link, del_link,
          get_peer_output_anchor
    """
    def __repr__(self):
        anchor_info = '< index: {}, name: {} >'.format(
            self._index, self._anchor_name)
        return anchor_info


class OutputAnchor(OutputAnchorBase):
    """
    Function: Data structure of anchor which contains graph topologic info
    APIs: node, index, name, set_name, add_link, del_link,
          get_peer_input_anchor, get_reused_info
    """
    def __repr__(self):
        anchor_info = '< index: {}, name: {} >'.format(
            self._index, self._anchor_name)
        return anchor_info
