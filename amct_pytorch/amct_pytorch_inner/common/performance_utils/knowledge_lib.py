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
