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
import torch.nn.functional as F

from amct_pytorch.classic.quantize_op.quant_result_mixin import (
    QuantResultSupportMixin,
)
from amct_pytorch.classic.quantize_op.utils import (
    apply_progressive_quant_dequant,
    calculate_progressive_weights_scale_factor,
    calculate_scale_offset,
)
from amct_pytorch.common.utils.data_utils import check_linear_input_dim
from amct_pytorch.common.utils.log import LOGGER
from amct_pytorch.common.utils.quant_util import (
    quant_dequant_tensor,
    quant_dequant_weight,
)
from amct_pytorch.common.utils.vars import (
    FLOAT4_E2M1,
    FLOAT8_E4M3FN,
    INT8,
    INT32_MAX,
    INT32_MIN,
)
from amct_pytorch.quantize_op.base_quant_module import BaseQuantizeModule

SMOOTH_STRENGTH_RANGE = [0.1, 0.99]
GRID_NUM = 9


class SmoothQuant(QuantResultSupportMixin, BaseQuantizeModule):
    """
    Function: Customized torch.nn.Module of the SmoothQuant class.

    It supports two smooth-strength modes selected by the per-layer resolved
    `algorithm` config:

    - fixed (`smoothquant`): the smoothing strength comes from
      `algorithm.smoothquant.smooth_strength`; the calibration forward finalizes
      automatically once `batch_num` is reached.
    - search (`auto_smooth`): the smoothing strength is not fixed and is
      searched block-wise by `sequential_calibrate` via `finalize(alpha)` (the
      module exposes the `calib_param_grid`/`apply_calib_param`/`reset_calib`
      contract for the gradient-free calibrator).

    APIs: forward, finalize.
    """

    _quant_param_keys = (
        'scale',
        'scale_w',
        'offset_w',
        'scale_d',
        'offset_d',
        'scale_w1',
        'scale_w2',
        'smooth_strength',
    )
    supports_quant_result = True

    def __init__(self, ori_module, layer_name, quant_config):
        """
        Function: init objective.
        Args:
        ori_module: torch module. Linear.
        layer_name: ori_module's name.
        quant_config: calibration algorithm parameters.
        """
        super().__init__(ori_module, layer_name, quant_config)
        self.ori_module_type = type(ori_module).__name__
        self.weight = ori_module.weight
        self.bias = ori_module.bias
        self.layer_name = layer_name
        self.quant_config = quant_config
        self.batch_num = quant_config.get('batch_num')
        self.act_granularity = quant_config.get('inputs_cfg').get('strategy')
        self.dynamic = quant_config.get('inputs_cfg').get('dynamic')
        self.wts_granularity = quant_config.get('weights_cfg').get('strategy')
        self.act_symmetric = quant_config.get('inputs_cfg').get('symmetric')
        self.wts_symmetric = quant_config.get('weights_cfg').get('symmetric')
        self.act_type = quant_config.get('inputs_cfg').get('quant_type')
        self.wts_type = quant_config.get('weights_cfg').get('quant_type')
        self.group_size = quant_config.get('weights_cfg').get('group_size', None)
        self.cur_batch = 0
        self.device = ori_module.weight.device

        algorithm = quant_config.get('algorithm') or {}
        self.is_auto_smooth = 'auto_smooth' in algorithm
        self.smooth_strength = None
        self.finalized = False
        self.fake_quant_cache_ready = False
        self.shared_param_mode = None
        self._stats_finished = False

        self.data_max = (
            torch.ones(
                (1, self.weight.shape[-1]),
                device=self.weight.device,
                dtype=self.weight.dtype,
            )
            * torch.inf
            * -1
        )
        self.data_min = (
            torch.ones(
                (1, self.weight.shape[-1]),
                device=self.weight.device,
                dtype=self.weight.dtype,
            )
            * torch.inf
        )
        self.batch_input = None
        self.scale_w1 = None
        self.scale_w2 = None

    @torch.no_grad()
    def forward(self, inputs):
        """
        Function: SmoothQuant forward function.

        During calibration it accumulates the per-channel activation range and
        returns the original linear output. Once finalized (fixed mode:
        automatically at batch_num; search mode: by `finalize(alpha)`) it
        switches to the fake-quantized forward.
        """
        check_linear_input_dim(inputs)
        inputs = inputs.to(self.weight.device)
        output = F.linear(inputs, self.weight, self.bias)

        if self.finalized and self.is_auto_smooth:
            return self.fake_quant_forward(inputs)

        if self.shared_param_mode is not None:
            # Shared-input mode: activation quantization is owned by the
            # LayerNormQuant pre_op, so this module does not collect stats.
            if not self.is_auto_smooth:
                # Fixed smoothquant keeps the base behaviour: the shared
                # projection returns the original linear output during the
                # calibration batch and only fake-quantizes once the batch
                # budget is exceeded.
                self.cur_batch += 1
                if self.cur_batch > self.batch_num:
                    return self.fake_quant_forward(inputs)
            return output

        if self._stats_finished:
            # The block grid search finished the calibration pass and replays the
            # block to reconstruct MSE. Skip statistics so the replay does not
            # keep bumping cur_batch / concatenating batch_input.
            return output

        self.cur_batch += 1
        if self.cur_batch > self.batch_num:
            if self.is_auto_smooth:
                return output
            return self.fake_quant_forward(inputs)

        batch_max = inputs.reshape(-1, inputs.shape[-1]).amax(dim=0, keepdim=True)
        batch_min = inputs.reshape(-1, inputs.shape[-1]).amin(dim=0, keepdim=True)
        if self.act_granularity == 'token':
            if self.batch_input is None:
                self.batch_input = inputs.cpu()
            else:
                self.batch_input = torch.cat((self.batch_input, inputs.cpu()), dim=-1)

        # update data_max and data_min to calculate smooth factor
        self.data_max = self.data_max.to(inputs.device)
        self.data_min = self.data_min.to(inputs.device)
        self.data_max = torch.where(batch_max > self.data_max, batch_max, self.data_max)
        self.data_min = torch.where(batch_min < self.data_min, batch_min, self.data_min)

        if not self.is_auto_smooth and self.cur_batch == self.batch_num:
            self.finalize()
            # Fixed mode finalizes exactly once here, so this is its dump point.
            self.maybe_dump_quant_params()

        return output

    def get_act_max(self):
        """Return the per-channel activation magnitude used by smooth factor."""
        return torch.max(self.data_max.abs(), self.data_min.abs())

    def get_wts_max(self):
        """Return the per-channel weight magnitude used by smooth factor."""
        return self.weight.abs().amax(dim=0, keepdim=True)

    @torch.no_grad()
    def finalize(self, alpha=None):
        """
        Compute the smooth factor and the quant params, then switch the module
        to the fake-quant mode.

        Args:
            alpha: optional smoothing strength in (0, 1). When None, the fixed
                `smoothquant.smooth_strength` from the config is used.
        """
        if alpha is None:
            alpha = (
                self.quant_config.get('algorithm')
                .get('smoothquant')
                .get('smooth_strength')
            )
        self.smooth_strength = alpha
        act_max = self.get_act_max()
        wts_max = self.get_wts_max()
        smooth_factor = self.calculate_smooth(act_max, wts_max, alpha)
        weight = self.weight * smooth_factor

        # only FP8 * FP4 do progressive scale
        if self.act_type == FLOAT8_E4M3FN and self.wts_type == FLOAT4_E2M1:
            self.scale_w1, self.scale_w2 = calculate_progressive_weights_scale_factor(
                weight.data, group_size=self.group_size
            )
        else:
            self.scale_w, self.offset_w = self.calculate_weights_scale_factor(
                weight, self.wts_granularity
            )

        if self.act_granularity == 'tensor':
            scale_d, offset_d = self._calculate_per_tensor_params(smooth_factor)
        elif self.act_granularity == 'token':
            scale_d, offset_d = self._calculate_per_token_params(smooth_factor)

        self.scale_d = scale_d
        self.offset_d = offset_d
        self.scale = smooth_factor
        self.finalized = True
        self.fake_quant_cache_ready = False

        LOGGER.logd(
            f"Calculate smooth quant params of layer '{self.layer_name}' success!",
            'SmoothQuant',
        )

    def _on_quant_params_loaded(self):
        """Re-arm the fake-quant path after a persisted quant result is loaded."""
        self.require_loaded_quant_params('scale')
        if self.scale_w1 is not None or self.scale_w2 is not None:
            self.require_loaded_quant_params('scale_w1', 'scale_w2')
        else:
            self.require_loaded_quant_params('scale_w')
            if self.wts_symmetric is False:
                self.require_loaded_quant_params('offset_w')

        if self.act_granularity is not None:
            self.require_loaded_quant_params('scale_d')
            if self.act_symmetric is False:
                self.require_loaded_quant_params('offset_d')

        self.fake_quant_cache_ready = False
        self.cached_dq_w = None
        self.cached_bias = None
        self.finalized = True

    def calculate_smooth(self, act_max, wts_max=None, alpha=None):
        """
        Function: smooth the input tensor by the smoothing strength.
        Parameters:
            act_max: max activation per channel, shape [1, cin]
            wts_max: max weight per channel, shape [1, cin]. Defaults to the
                current weight magnitude.
            alpha: smoothing strength in (0, 1). Defaults to the fixed
                `smoothquant.smooth_strength` from the config.
        Return: smooth factor, shape [1, cin]
        """
        if wts_max is None:
            wts_max = self.get_wts_max()
        if alpha is None:
            alpha = (
                self.quant_config.get('algorithm')
                .get('smoothquant')
                .get('smooth_strength')
            )
        smooth_factor = (act_max**alpha) / (wts_max ** (1 - alpha))
        zero_mask = smooth_factor < torch.finfo(act_max.dtype).eps
        invalid_mask = ~torch.isfinite(smooth_factor)
        if zero_mask.any().item() or invalid_mask.any().item():
            smooth_factor = torch.where(
                zero_mask | invalid_mask,
                torch.tensor(
                    1.0, dtype=smooth_factor.dtype, device=smooth_factor.device
                ),
                smooth_factor,
            )
            LOGGER.logd("The smooth factor is non-finite or zero and is set to 1")

        return smooth_factor

    def calculate_weights_scale_factor(self, weight, weight_granularity):
        """
        Function: calculate weights's quant factor
        Parameters:
            weight: weight data
            weight_granularity: weight granularity
        """
        weight_data = weight.data

        if weight_granularity == 'channel':
            weight_max = weight_data.max(dim=1, keepdim=True).values
            weight_min = weight_data.min(dim=1, keepdim=True).values
        elif weight_granularity == 'tensor':
            weight_max = weight_data.max().reshape(1, 1)
            weight_min = weight_data.min().reshape(1, 1)
        scale_w, offset_w = calculate_scale_offset(
            weight_max, weight_min, self.wts_symmetric, self.wts_type
        )

        return scale_w, offset_w

    def accept_shared_params(
        self, mode, smooth_factor=None, activation_scale=None, activation_offset=None
    ):
        """
        Receive shared-input quantization params from the LayerNormQuant pre_op.

        The shared-input path owns the activation scale/offset (and, for fixed
        smoothquant, the smooth factor). The weight scale is computed here from
        the received smooth factor so the module can switch to fake-quant.
        """
        self.shared_param_mode = mode
        self.scale = smooth_factor
        self.scale_d = activation_scale
        self.offset_d = activation_offset
        weight = (
            self.weight * smooth_factor if smooth_factor is not None else self.weight
        )
        if self.act_type == FLOAT8_E4M3FN and self.wts_type == FLOAT4_E2M1:
            self.scale_w1, self.scale_w2 = calculate_progressive_weights_scale_factor(
                weight.data, group_size=self.group_size
            )
        else:
            self.scale_w, self.offset_w = self.calculate_weights_scale_factor(
                weight, self.wts_granularity
            )
        self.fake_quant_cache_ready = False
        self.data_max = None
        self.data_min = None
        self.finalized = True

    @torch.no_grad()
    def fake_quant_forward(self, inputs):
        if not self.fake_quant_cache_ready:
            if self.scale is not None:
                smooth_w = self.scale.to(
                    device=self.weight.device, dtype=self.weight.dtype
                )
                w = self.weight.data * smooth_w
            else:
                w = self.weight.data
            if self.scale_w1 is not None:
                self.cached_dq_w = apply_progressive_quant_dequant(
                    w, self.scale_w1, self.scale_w2, self.group_size
                )
            else:
                self.cached_dq_w = quant_dequant_weight(
                    w, self.wts_type, self.scale_w, self.offset_w, self.group_size
                )
            bias = self.bias
            if (
                bias is not None
                and self.scale_d is not None
                and self.scale_w is not None
                and self.act_type == INT8
                and self.act_granularity == 'tensor'
            ):
                deq_scale = (self.scale_d * self.scale_w).reshape(-1)
                bias_q = (bias.data / deq_scale).round().clamp(INT32_MIN, INT32_MAX)
                bias = (bias_q * deq_scale).to(bias.dtype)
            self.cached_bias = bias
            self.fake_quant_cache_ready = True

        if self.scale is not None:
            smooth_x = self.scale.to(device=inputs.device, dtype=inputs.dtype)
            x = inputs / smooth_x
        else:
            x = inputs
        if self.scale_d is not None:
            dq_x = quant_dequant_tensor(x, self.act_type, self.scale_d, self.offset_d)
        else:
            dq_x = x
        return F.linear(dq_x, self.cached_dq_w, self.cached_bias)

    def _calculate_per_token_params(self, smooth_factor):
        """
        Function: calculate per token scale_d and offset_d
        Parameters:
            smooth_factor: smooth factor
        """
        smooth_factor = smooth_factor.cpu()
        repeat = self.batch_input.shape[-1] // smooth_factor.shape[-1]
        smooth_input = self.batch_input / smooth_factor.repeat(1, repeat)
        cout_axis = -2
        axis_list = list(range(smooth_input.dim()))
        axis_list.pop(cout_axis)
        axis = tuple(axis_list)
        per_token_max = torch.amax(smooth_input, dim=axis, keepdim=True)
        per_token_min = torch.amin(smooth_input, dim=axis, keepdim=True)
        return calculate_scale_offset(
            per_token_max, per_token_min, self.act_symmetric, self.act_type
        )

    def _calculate_per_tensor_params(self, smooth_factor):
        """
        Function: calculate per tensor scale_d and offset_d
        Parameters:
            smooth_factor: smooth factor
        """
        max_val = (self.data_max / smooth_factor).max().reshape(-1)
        min_val = (self.data_min / smooth_factor).min().reshape(-1)
        return calculate_scale_offset(
            max_val, min_val, self.act_symmetric, self.act_type
        )

    def calib_param_grid(self):
        """
        Return the candidate smoothing strengths for the block-wise gradient-free
        grid search (read from the `auto_smooth` config, with defaults).
        """
        algo_cfg = (self.quant_config or {}).get('algorithm', {}) or {}
        algo_cfg = algo_cfg.get('auto_smooth', {}) or {}
        grid_num = algo_cfg.get('grid_num', GRID_NUM)
        smooth_range = algo_cfg.get('smooth_range', SMOOTH_STRENGTH_RANGE)
        return torch.linspace(
            smooth_range[0], smooth_range[1], grid_num, dtype=torch.float64
        ).tolist()

    def reset_calib(self):
        """Switch the module back to the original (unquantized) forward."""
        self.finalized = False
        self._stats_finished = True

    def apply_calib_param(self, param):
        """Apply a smoothing strength and switch to the fake-quant forward."""
        self.finalize(float(param))
