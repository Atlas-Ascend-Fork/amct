# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
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

import torch
import torch.nn as nn

from amct_pytorch.common.utils.log import LOGGER

LOG_TAG = 'BaseQuantizeModule'


class BaseQuantizeModule(nn.Module):
    """Base class of calibration modules with quant param export and load.

    Each subclass declares its own quant param attribute names so the base class
    does not hard-code the algorithm-specific names. The generic export, load
    and dump plumbing can then be shared by MinMax, AWQ and GPTQ.
    """

    # Attribute names holding this algorithm's quant params. Values may be
    # tensors, JSON-serializable scalars, or None (skipped when exporting).
    _quant_param_keys = ()

    def __init__(self, ori_module, layer_name, quant_config):
        super().__init__()
        if not isinstance(layer_name, str) or not layer_name:
            raise ValueError(
                'layer_name must be a non-empty string, got {!r}.'.format(layer_name)
            )
        self.ori_module_type = None
        self.act_type = None
        self.wts_type = None
        self.scale_w = None
        self.scale_d = None
        self.offset_w = None
        self.offset_d = None
        self.group_size = None

        self.scale = None
        self.clip_max = None

        self.layer_name = layer_name
        self.quant_config = quant_config
        self._quant_result_writer = None
        self._quant_params_dumped = False

    def forward(self, inputs):
        pass

    @classmethod
    def quant_param_keys(cls):
        """Validated tuple of declared quant param attribute names."""
        keys = tuple(cls._quant_param_keys)
        for key in keys:
            if not isinstance(key, str) or not key:
                raise TypeError(
                    "quant param names declared by {} must be non-empty strings, "
                    "got {!r}.".format(cls.__name__, key)
                )
        if len(set(keys)) != len(keys):
            raise ValueError(
                '{} declares duplicate quant param names.'.format(cls.__name__)
            )
        return keys

    def export_quant_params(self):
        """Split declared quant params into ``(tensors, metadata)``.

        Tensors come back detached, on CPU and contiguous so they are safe to
        serialize; non-tensor values go to metadata; ``None`` values are absent
        from both, which is how "this algorithm/config never produced it" is
        represented.
        """
        tensors = {}
        metadata = {}
        for key in self.quant_param_keys():
            value = getattr(self, key, None)
            if value is None:
                continue
            if isinstance(value, torch.Tensor):
                tensors[key] = value.detach().cpu().contiguous()
            else:
                metadata[key] = value
        return tensors, metadata

    def load_quant_params(self, tensors, metadata=None):
        """Restore quant params previously exported for this module.

        Rejects undeclared keys and keys present in both payloads. An existing
        tensor attribute is copied into in place (preserving its device, dtype
        and shape); an attribute currently ``None`` adopts the persisted dtype
        and moves to this module's device.
        """
        tensors = dict(tensors or {})
        metadata = dict(metadata or {})
        declared = set(self.quant_param_keys())
        self._check_loadable_keys(tensors, metadata, declared)

        device = self._quant_param_device()
        for key, value in tensors.items():
            self._load_tensor_param(key, value, device)
        for key, value in metadata.items():
            setattr(self, key, value)

        # Recorded so a subclass hook can tell "the artifact carried this" from
        # "the constructor happened to leave an attribute here", which matters for
        # params that always pre-exist, such as GPTQ's weight.
        self._loaded_quant_param_keys = frozenset(tensors) | frozenset(metadata)
        self._on_quant_params_loaded()
        LOGGER.logd(
            "Loaded quant params of layer '{}': {} tensor(s), {} scalar(s)".format(
                self.layer_name, len(tensors), len(metadata)
            ),
            LOG_TAG,
        )

    def _check_loadable_keys(self, tensors, metadata, declared):
        """Validate the incoming key sets against the declared params."""
        unknown = sorted((set(tensors) | set(metadata)) - declared)
        if unknown:
            raise ValueError(
                "quant params {} are not declared by {}; refusing to load them "
                "onto layer '{}'.".format(unknown, type(self).__name__, self.layer_name)
            )
        overlap = sorted(set(tensors) & set(metadata))
        if overlap:
            raise ValueError(
                "quant params {} appear as both tensor and non-tensor values "
                "for layer '{}'.".format(overlap, self.layer_name)
            )

    @torch.no_grad()
    def _load_tensor_param(self, key, value, device):
        """Restore one tensor quant param, in place when one already exists."""
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                "quant param '{}' of layer '{}' must be a torch.Tensor, got {}.".format(
                    key, self.layer_name, type(value).__name__
                )
            )
        current = getattr(self, key, None)
        if not isinstance(current, torch.Tensor):
            setattr(self, key, value.detach().to(device))
            return
        if tuple(current.shape) != tuple(value.shape):
            raise ValueError(
                "quant param '{}' of layer '{}' has shape {} but the persisted "
                "value has shape {}.".format(
                    key, self.layer_name, tuple(current.shape), tuple(value.shape)
                )
            )
        if current.dtype != value.dtype:
            raise ValueError(
                "quant param '{}' of layer '{}' has dtype {} but the persisted "
                "value has dtype {}.".format(
                    key, self.layer_name, current.dtype, value.dtype
                )
            )
        current.copy_(value.to(current.device))

    def _quant_param_device(self):
        """Device a restored quant param should live on: follow the weight."""
        weight = getattr(self, 'weight', None)
        if isinstance(weight, torch.Tensor):
            return weight.device
        for parameter in self.parameters(recurse=True):
            return parameter.device
        return torch.device('cpu')

    def attach_quant_result_writer(self, writer):
        """Attach the run's result writer, then run the subclass hook.

        Injected right after this module replaces the original one, which is
        also the point where an algorithm whose params are already final (pure
        weight-only) can dump immediately.
        """
        self._quant_result_writer = writer
        self._on_quant_result_writer_attached()

    def maybe_dump_quant_params(self):
        """Dump this layer's quant params once, if a writer is attached.

        Returns True when this call performed the dump. A failed write leaves
        the module undumped so the unit stays pending and no manifest appears.
        """
        if self._quant_result_writer is None or self._quant_params_dumped:
            return False
        # Imported lazily: quant_result lives under classic/, which this module
        # must not depend on at import time.
        from amct_pytorch.classic.quant_result import to_jsonable

        tensors, non_tensor_params = self.export_quant_params()
        self._quant_result_writer.write_unit(
            layer_name=self.layer_name,
            tensors=tensors,
            non_tensor_params=non_tensor_params,
            quant_config=to_jsonable(
                self.quant_config,
                "quant_config of layer '{}'".format(self.layer_name),
            ),
            quant_module_type=type(self).__name__,
            ori_module_type=self.ori_module_type,
        )
        self._quant_params_dumped = True
        return True

    def require_loaded_quant_params(self, *keys):
        """Fail unless the loaded unit carried every one of ``keys``."""
        loaded_keys = getattr(self, '_loaded_quant_param_keys', frozenset())
        missing = sorted(key for key in keys if key not in loaded_keys)
        if missing:
            raise ValueError(
                "quant result of layer '{}' is missing required quant param(s) {} "
                "for {}; refusing to reload an incomplete unit.".format(
                    self.layer_name, missing, type(self).__name__
                )
            )

    def _on_quant_params_loaded(self):
        """Hook: subclasses drop caches / restore progress after a load."""

    def _on_quant_result_writer_attached(self):
        """Hook: subclasses may dump right away if their params are final."""
