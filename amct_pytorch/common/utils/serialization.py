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
"""Strict conversion of quantization configuration values to JSON data."""

import json

import torch

__all__ = ['dtype_name', 'tensor_schema_list', 'to_jsonable', 'strict_mapping']


def dtype_name(dtype):
    """Return the artifact dtype spelling, for example float32."""
    text = str(dtype)
    return text[len('torch.') :] if text.startswith('torch.') else text


def _canonical_json(value):
    """Stable JSON form used to order set values."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def tensor_schema_list(tensors):
    """Tensor schema: a key-sorted list of key, dtype and shape dictionaries."""
    return [
        {
            'key': key,
            'dtype': dtype_name(tensors[key].dtype),
            'shape': list(tensors[key].shape),
        }
        for key in sorted(tensors)
    ]


def to_jsonable(value, path='quant_config'):
    """Strictly coerce a value for a persisted manifest, or raise.

    Scalars, torch.dtype and containers of those values are converted.
    Unknown objects are rejected because their string form cannot reconstruct a
    quantization module during reload.
    """
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if value != value or value in (float('inf'), float('-inf')):
            raise ValueError(
                '{} is the non-finite float {!r}, which canonical JSON cannot '
                'represent.'.format(path, value)
            )
        return value
    if isinstance(value, torch.dtype):
        return dtype_name(value)
    if isinstance(value, dict):
        return strict_mapping(value, path)
    if isinstance(value, (set, frozenset)):
        return sorted(
            (to_jsonable(item, '{}[]'.format(path)) for item in value),
            key=_canonical_json,
        )
    if isinstance(value, (list, tuple)):
        return [
            to_jsonable(item, '{}[{}]'.format(path, index))
            for index, item in enumerate(value)
        ]
    raise TypeError(
        '{} holds {} ({!r}), which cannot be serialized into a quant result '
        'manifest. Only scalars, torch.dtype and containers of those are '
        'allowed, so that reload can reconstruct the module.'.format(
            path, type(value).__name__, value
        )
    )


def strict_mapping(mapping, path):
    """Strictly convert a mapping; keys must be str, bool or int."""
    result = {}
    for key, item in mapping.items():
        if not isinstance(key, (str, bool, int)):
            raise TypeError(
                '{} has a {} key ({!r}); quant result config keys must be '
                'str, bool or int.'.format(path, type(key).__name__, key)
            )
        json_key = key if isinstance(key, str) else str(key)
        if json_key in result:
            raise ValueError(
                '{} key {!r} collides with another key after string '
                'normalization.'.format(path, json_key)
            )
        result[json_key] = to_jsonable(item, '{}.{}'.format(path, json_key))
    return result
