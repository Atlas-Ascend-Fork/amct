# -*- coding: UTF-8 -*-
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

def get_alg_name_from_config(alg):
    if isinstance(alg, dict):
        return list(alg.keys()), [alg.get(key) for key in list(alg.keys())]
    elif isinstance(alg, set):
        return list(alg), [None] * len(alg)
    elif isinstance(alg, str):
        return [alg], [None]
    else:
        raise ValueError(f'invalid algo name, pls check config')


def is_fuzzy_pattern(pattern):
    """
    Check if a pattern is a fuzzy matching pattern (starts with *)
    """
    return isinstance(pattern, str) and pattern.startswith('*')


def get_fuzzy_suffix(pattern):
    """
    Extract suffix from fuzzy pattern
    Example: "*self.atten.q_proj" -> "self.atten.q_proj"
    """
    if not is_fuzzy_pattern(pattern):
        return None
    return pattern[1:]


def match_fuzzy_pattern(layer_name, pattern):
    """
    Check if layer_name matches fuzzy pattern
    Example: "model.layers.0.self.atten.q_proj" matches "*self.atten.q_proj"
    """
    if not is_fuzzy_pattern(pattern):
        return layer_name == pattern
    
    suffix = get_fuzzy_suffix(pattern)
    return layer_name.endswith(suffix)


def find_matching_pattern(layer_name, patterns):
    """
    Find the first matching pattern for a given layer name
    Returns the pattern that matches, or None if no match found
    """
    for pattern in patterns:
        if match_fuzzy_pattern(layer_name, pattern):
            return pattern
    return None