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

the class of prior knowledge base design

"""


class KnowledgeLib():  # pylint: disable=too-few-public-methods
    """Prior knowledge base design class."""
    def __init__(self):
        self.quant_info = None

    def pre_fallback(self, quant_info):
        """Prior knowledge base pre rollback."""
        self.quant_info = quant_info
        for group_layer in self.quant_info.get_group_layers():
            cin_num = 0
            conv_layer_type = ('Conv2D', 'Conv')
            for layer_name in group_layer:
                if self.quant_info.get_cin_cout_info().get(layer_name):
                    cin_num += self.quant_info.get_cin_cout_info(
                    )[layer_name][0]
                else:
                    cin_num = 1e6
                    break
            if cin_num <= 16 and len(
                    group_layer) == 1 and self.quant_info.get_main_types()[
                        group_layer[0]] in conv_layer_type:
                for layer_name in group_layer:
                    self.quant_info.set_config_enable(layer_name, False)
                self.quant_info.add_pre_fallback_layers(group_layer)
            else:
                self.quant_info.add_search_layers(group_layer)
