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
"""Shift value calibration module.

``ShiftCalibrationModule`` wraps a target module (``torch.nn.Linear`` /
``torch.nn.Conv2d`` / ``torch.nn.ConvTranspose2d``) that runs on the FP16
non-quantized or the FP16*INT8 weight-only-quantized path. During user
calibration forward passes it accumulates the per-candidate SSE of the fp16fixed
fixed point pipeline against the FP32 reference, and writes the best
``fixed_shift_value`` into the record file after ``batch_num`` batches.

The wrapper keeps the user model numerics untouched: ``forward`` always
returns the original module output.
"""

import os

import torch
from google.protobuf import message as _proto_message
from google.protobuf import text_format
from torch import nn

from .....amct_pytorch.common.utils.record_file_operator import (
    record_fixed_shift_value,
)
from .....amct_pytorch.nn.module.quantization import shift_value_kernel
from .....amct_pytorch.proto import scale_offset_record_pb2
from .....amct_pytorch.utils.log import LOGGER
from .....amct_pytorch.utils.singleton_record import SingletonScaleOffsetRecord

FP16_FP16 = 'fp16_fp16'
FP16_INT8 = 'fp16_int8'

# repeated fields of SingleLayerRecord (see scale_offset_record_pytorch.proto):
# the merge below branches on this explicit set instead of the descriptor
# label attribute -- protobuf 7.x (upb) removed FieldDescriptor.label while
# 4.x lacks is_repeated, so any descriptor-based probe breaks one of the
# two; keep the set in sync when the proto gains a repeated field
_REPEATED_RECORD_FIELDS = frozenset(
    (
        'shift_bit',
        'tensor_balance_factor',
        'scale_w',
        'offset_w',
        'scale_r',
        'offset_r',
    )
)


def _fill_missing_fields(dst_value, src_value):
    """Fill the record fields ``dst_value`` does not carry yet from
    ``src_value``; the destination always wins on conflicts.

    Scalar optional fields are copied only when unset in the
    destination (``HasField``), repeated fields only when the
    destination list is empty. Used to fold the persisted record file
    content into the fresher in-memory singleton view without
    duplicating repeated entries. Portable across protobuf 4.x to 7.x:
    only ``field.name``, ``HasField`` and value types are touched (the
    descriptor label attribute is gone in the 7.x upb implementation).
    """
    for field in src_value.DESCRIPTOR.fields:
        name = field.name
        src_field = getattr(src_value, name)
        if name in _REPEATED_RECORD_FIELDS:
            dst_field = getattr(dst_value, name)
            if not len(dst_field) and len(src_field):
                dst_field.extend(src_field)
        elif not dst_value.HasField(name) and src_value.HasField(name):
            if isinstance(src_field, _proto_message.Message):
                getattr(dst_value, name).CopyFrom(src_field)
            else:
                setattr(dst_value, name, src_field)


class ShiftCalibrationModule(nn.Module):
    """Wrap a FP16-path module and calibrate its fixed_shift_value."""

    def __init__(
        self,
        object_module,
        layer_name,
        record_file,
        batch_num,
        quant_regime=FP16_FP16,
    ):
        """
        Args:
            object_module: the wrapped Linear / Conv2d / ConvTranspose2d.
            layer_name: record key, the pytorch module name.
            record_file: scale offset record file path.
            batch_num: number of calibration batches.
            quant_regime: 'fp16_fp16' or 'fp16_int8'.
        """
        super().__init__()
        if quant_regime not in (FP16_FP16, FP16_INT8):
            raise ValueError(
                "quant_regime should be 'fp16_fp16' or 'fp16_int8', but is {}".format(
                    quant_regime
                )
            )
        self.object_module = object_module
        self.layer_name = layer_name
        self.record_file = os.path.realpath(record_file)
        self.batch_num = batch_num
        self.quant_regime = quant_regime
        self.candidates = shift_value_kernel.get_candidates(quant_regime)
        self.default_value = shift_value_kernel.get_default_value(quant_regime)

        self.current_batch = 0
        self.sse_sum = torch.zeros(len(self.candidates), dtype=torch.float64)
        # set True when the fp16fixed kernel is missing or kernel constraints are
        # not satisfied, the default value is written instead of searching.
        self.use_default_value = not self._check_module_supported(object_module)
        # fp16_int8: the board kernel pins the dequant domain at sv=13
        # (the AscendDequant parameters are compile-time constants and
        # the passed fixed_shift_value is IGNORED -- confirmed with the
        # fp16s8 kernel developers), so whatever sv the search writes,
        # the deployed accuracy is the sv=13 behaviour. The search is
        # disabled: the regime default value (13) is written directly,
        # the probe is never invoked. Reopening the search is a one-line
        # change once the board supports a dynamic sv. The fp16_fp16
        # regime keeps searching (SSE argmin, its sv is the free
        # quantity positioning the accumulator grid).
        if quant_regime == FP16_INT8:
            self.use_default_value = True
        # weight quant factors of this layer, lazily read from the record
        # file by the fp16_int8 regime (None = not read yet)
        self._weight_quant_info = None
        self._weight_quant_info_read = False

    @staticmethod
    def _check_module_supported(object_module):
        """Check the module against the support constraints.

        The fp16fixed engines implement the full group loop (conv: per
        (batch, group) im2col slice; conv transpose: per-group input
        block -> output channel slice) and the conv-transpose
        output_padding canvas, both verified against
        ``F.conv2d`` / ``F.conv_transpose2d`` with groups in {1, 2, 4}
        and output_padding in {(0,1), (1,1), (1,0)} -- so the former
        first-stage restrictions (groups == 1, output_padding == 0) are
        lifted. The amct_onnx ShiftCalibration kernel keeps its own
        group == 1 / output_padding == 0 constraints (its slice builder
        has no group handling); see the ONNX design doc.
        """
        module_type = type(object_module).__name__
        return module_type in ('Linear', 'Conv2d', 'ConvTranspose2d')

    def forward(self, inputs):
        """Run the original module and accumulate the shift calibration."""
        sub_out = self.object_module(inputs)
        with torch.no_grad():
            if self.current_batch < self.batch_num:
                self._accumulate_sse(inputs)
                self.current_batch += 1
                if self.current_batch == self.batch_num:
                    self._write_record()
        return sub_out

    def _accumulate_sse(self, inputs):
        """Accumulate per-candidate SSE of this batch."""
        if self.use_default_value:
            return
        if self._has_non_finite_data(inputs):
            LOGGER.logi(
                "Layer '{}' falls back to the default fixed_shift_value {}, "
                'because the calibration data entering the fixed point '
                'search contains inf / nan.'.format(
                    self.layer_name, self.default_value
                ),
                'ShiftCalibrationModule',
            )
            self.use_default_value = True
            return
        try:
            weight = self._get_second_operand()
            sse = shift_value_kernel.compute_shift_value_sse(
                inputs,
                weight,
                self.quant_regime,
                self.object_module,
                weight_quant_info=self._get_weight_quant_info(),
            )
        except NotImplementedError as exception:
            LOGGER.logi(
                "Layer '{}' falls back to the default fixed_shift_value {}, "
                'because the fp16fixed shift value kernel is not usable '
                '({}).'.format(self.layer_name, self.default_value, exception),
                'ShiftCalibrationModule',
            )
            self.use_default_value = True
            return
        if sse is None:
            self.use_default_value = True
            return
        if sse.numel() != len(self.candidates):
            raise RuntimeError(
                "Layer '{}': shift value kernel returned {} sse values, "
                'but {} candidates are expected.'.format(
                    self.layer_name, sse.numel(), len(self.candidates)
                )
            )
        self.sse_sum += sse.detach().double().cpu()

    def _has_non_finite_data(self, inputs):
        """Whether the data entering the fixed point search is non-finite.

        The ``ShiftCalibration`` kernel spec forbids inf / nan in the
        calibration data: the fixed point pipeline has no representation
        for them (the raw fp16 exponent 31 marks inf / nan), so the
        layer falls back to the regime default value instead of
        searching on garbage. The check covers the fp16 cast as well --
        a finite fp32 value beyond the fp16 range becomes inf in the
        search domain. The fp16_fp16 regime also checks the second
        operand (the fp16 module weight); the fp16_int8 weight is int8
        and always finite.
        """
        if not torch.isfinite(inputs.to(torch.float16)).all():
            return True
        if self.quant_regime == FP16_FP16:
            weight = self._get_second_operand()
            if not torch.isfinite(weight.to(torch.float16)).all():
                return True
        return False

    def _get_weight_quant_info(self):
        """Weight quant factors of this layer, fp16_int8 regime only.

        The weight calibration (``WeightsCalibrationPass``) runs before
        the shift calibration modules are inserted, so ``scale_w`` /
        ``offset_w`` / ``wts_type`` are already in the record file when
        the calibration forwards start. The record is read once and
        cached: the entries do not change during data calibration.
        Returns ``None`` for the fp16_fp16 regime and when this layer
        has no record entry (the kernel then falls back to the default
        value).
        """
        if self.quant_regime != FP16_INT8:
            return None
        if not self._weight_quant_info_read:
            self._weight_quant_info = self._read_weight_quant_info()
            self._weight_quant_info_read = True
        return self._weight_quant_info

    def _read_weight_quant_info(self):
        """Read this layer's weight quant factors from the record file."""
        records = scale_offset_record_pb2.ScaleOffsetRecord()
        if os.path.exists(self.record_file) and (
            os.path.getsize(self.record_file) != 0
        ):
            with open(self.record_file, 'r') as fid:
                text_format.Merge(fid.read(), records)
        for record in records.record:
            if record.key != self.layer_name:
                continue
            if len(record.value.scale_w) == 0:
                return None
            return {
                'scale_w': list(record.value.scale_w),
                'offset_w': list(record.value.offset_w),
                'wts_type': record.value.wts_type,
            }
        return None

    def _get_second_operand(self):
        """Get the second operand of the multiplication."""
        return self.object_module.weight

    def _select_value(self):
        """Select the candidate with the minimum accumulated SSE."""
        if self.use_default_value:
            return self.default_value
        best_index = int(torch.argmin(self.sse_sum).item())
        return int(self.candidates[best_index])

    def _write_record(self):
        """Write the calibrated fixed_shift_value into the record file.

        Only the fixed_shift_value field of this layer's record is added or
        updated; other fields (scale_w / offset_w ...) of the same record
        entry are kept untouched.

        The record file and the shared singleton carry complementary
        state: the Recorder keeps the not yet flushed activation factors
        of other layers in the singleton, while the file holds what an
        earlier pass (weights calibration) already persisted. The file
        content is therefore rebuilt as the UNION of both: the singleton
        forms the base (it is the freshest view), the file fills the
        keys and the individual fields the singleton does not carry
        yet, and repeated entries never merge in append mode. The
        singleton itself is only upserted in place with this layer's
        value -- a wholesale CopyFrom would wipe the Recorder's unsaved
        activation factors of other layers (two layer calibration with
        a plain quantized layer ahead of a shift wrapped layer would
        silently lose the first layer's scale_d / offset_d).
        """
        value = self._select_value()
        singleton = SingletonScaleOffsetRecord()
        singleton.lock.acquire()
        try:
            merged = scale_offset_record_pb2.ScaleOffsetRecord()
            merged.CopyFrom(singleton.record)
            singleton_keys = {item.key for item in merged.record}
            if os.path.exists(self.record_file) and (
                os.path.getsize(self.record_file) != 0
            ):
                file_records = scale_offset_record_pb2.ScaleOffsetRecord()
                with open(self.record_file, 'r') as fid:
                    text_format.Merge(fid.read(), file_records)
                for file_item in file_records.record:
                    if file_item.key not in singleton_keys:
                        merged.record.add().CopyFrom(file_item)
                    else:
                        for merged_item in merged.record:
                            if merged_item.key == file_item.key:
                                _fill_missing_fields(merged_item.value, file_item.value)
                                break
            record_fixed_shift_value(merged, self.layer_name, value)
            with open(self.record_file, 'w') as fid:
                fid.write(text_format.MessageToString(merged, as_utf8=True))
            # in place upsert: never replace the singleton content, the
            # Recorder still needs the unsaved factors of other layers
            record_fixed_shift_value(singleton.record, self.layer_name, value)
        finally:
            singleton.lock.release()
        LOGGER.logi(
            "Layer '{}' fixed_shift_value is {} (regime: {}).".format(
                self.layer_name, value, self.quant_regime
            ),
            'ShiftCalibrationModule',
        )
