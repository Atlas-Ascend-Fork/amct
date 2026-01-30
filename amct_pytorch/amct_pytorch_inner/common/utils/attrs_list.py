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

Define attributes

"""

# Attribute that record equivalent node after quantized graph
ATTR_NODE_EQUIVALENT_OBJECT_LAYER = 'attr_node_equivalent_object_layer'
# Attribute that record equivalent input info after quantized graph
ATTR_NODE_EQUIVALENT_INPUT = 'attr_node_equivalent_input'
# Attribute that record equivalent output info after quantized graph
ATTR_NODE_EQUIVALENT_OUTPUT = 'attr_node_equivalent_output'
# Attribute that record fused nodes info
ATTR_NODE_FUSION_INFO = 'attr_node_fused_info'
# Attribute that record after fused, equivalent output of original node
ATTR_NODE_OUTPUT_NODE = 'attr_node_output_node'
