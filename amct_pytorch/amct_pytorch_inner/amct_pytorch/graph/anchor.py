#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

Data structure of anchor which contains graph topologic info

"""
from amct_pytorch.common.graph_base.anchor_base import InputAnchorBase
from amct_pytorch.common.graph_base.anchor_base import OutputAnchorBase


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
