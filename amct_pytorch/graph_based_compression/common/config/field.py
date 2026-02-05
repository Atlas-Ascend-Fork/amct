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
from decimal import Decimal
from collections import OrderedDict
from enum import Enum, unique

from ..utils.util import find_repeated_items
from ..utils.util import check_no_repeated
from ...utils.log import LOGGER # pylint: disable=relative-beyond-top-level
from ..utils.vars_util import WINOGRAD_NUM_BITS

# default parameters for quant config
# global parameters
VERSION = 1
BATCH_NUM = 1
WEIGHT_OFFSET = False
ACTIVATION_OFFSET = True
JOINT_QUANT = False
DO_FUSION = True
SKIP_FUSION_LAYERS = []
DEFAULT_NUM_BITS = 8
ACT_SUPPORT_NUM_BITS = [4, 8, 16]
WTS_SUPPORT_NUM_BITS = [4, 6, 7, 8]

# layer's activation quant parameters
MAX_PERCENTILE = 0.999999
MIN_PERCENTILE = 0.999999
SEARCH_RANGE_START = 0.7
SEARCH_RANGE_END = 1.3
SEARCH_STEP = 0.01
NUM_BINS = 128

# layer's weights quant parameters
NUM_STEPS = 32
NUM_STEPS_RANGE = [16, 32]
NUM_OF_ITERATION = 1
NUM_OF_ITERATION_RANGE = [0, 1, 2, 3, 4, 5]
NUM_OF_BINS = 4096
NUM_OF_BINS_RANGE = [1024, 2048, 4096, 8192]
NUM_ITERATION = 10000
REG_PARAM = 0.01
BETA_RANGE_START = 20
BETA_RANGE_END = 2
WART_START = 0.2

# layer's dmq balancer parameters
DMQ_BALANCER_RANGE_START = 0.2
DMQ_BALANCER_RANGE_END = 0.8

ACT_ALGO = 'act_algo'
WTS_ALGO = 'wts_algo'
SEARCH_RANGE = 'search_range'
FAKEQUANT_PRECISION_MODE = ['DEFAULT', 'FORCE_FP16_QUANT']


def _check_type(name, variable, typeinfo):
    if not isinstance(variable, typeinfo):
        raise TypeError("Type of %s should be %s, but is %s" \
                        % (name, typeinfo, type(variable)))


@unique
class FakequantPrecisionMode(Enum):
    '''enumeration of scale precision'''
    DEFAULT = 'DEFAULT'
    FORCE_FP16_QUANT = 'FORCE_FP16_QUANT'


class ParamPool():
    '''store parameters used by fields'''
    def __init__(self):
        self._quant_layers = None
        self._supported_layers = None
        self._skip_layers = []
        self._layer_type = None
        self._layer_name = None
        self._wts_algo = None
        self._act_algo = None

    def set_quant_layers(self, quant_layers):
        '''set quant layers used by field'''
        self._quant_layers = quant_layers

    def get_quant_layers(self):
        '''get quant layers'''
        return self._quant_layers

    def set_supported_layers(self, supported_layers):
        '''set supported layers used by field'''
        self._supported_layers = supported_layers

    def get_supported_layers(self):
        '''get supported layers'''
        return self._supported_layers

    def set_skip_layers(self, skip_layers):
        '''set skip layers used by field'''
        self._skip_layers = skip_layers

    def get_skip_layers(self):
        '''get supported layers'''
        return self._skip_layers

    def set_layer_type(self, layer_type):
        '''set layer type used by field'''
        self._layer_type = layer_type

    def get_layer_type(self):
        '''get layer type'''
        return self._layer_type

    def set_layer_name(self, layer_name):
        '''set layer name used by child field'''
        self._layer_name = layer_name

    def get_layer_name(self):
        '''get layer name'''
        return self._layer_name

    def set_wts_algo(self, wts_algo):
        '''set wgt alg used by child field'''
        self._wts_algo = wts_algo

    def get_wts_algo(self):
        '''get wgt alg'''
        return self._wts_algo

    def set_act_algo(self, act_algo):
        '''set act alg used by child field'''
        self._act_algo = act_algo

    def get_act_algo(self):
        '''get act alg'''
        return self._act_algo

    # Attention! All parameters should be cleared here.
    def clear(self):
        '''clear all parameters'''
        self._quant_layers = None
        self._supported_layers = None
        self._layer_type = None
        self._layer_name = None
        self._wts_algo = None
        self._act_algo = None


PARAM_POOL = ParamPool()


class Field():
    '''base class for field'''
    def __init__(self, capacity):
        self.capacity = capacity
        self.parent = None

    def has_default(self): # pylint: disable=no-self-use
        '''indicate whether this field has default value'''
        return True

    def is_leaf(self): # pylint: disable=no-self-use
        '''indicate whether it is a leaf field'''
        raise RuntimeError('Unimplemented function.')

    def check(self, name, value): # pylint: disable=no-self-use
        '''check the value of this field'''
        raise RuntimeError('Unimplemented function.')

    def set_parent(self, parent):
        '''point to parent node of this field'''
        self.parent = parent


class LeafField(Field):
    """leaf node have no child"""
    def is_leaf(self):
        return True

    def default_value(self): # pylint: disable=no-self-use
        '''the default value of this field if it has default value'''
        raise RuntimeError('Unimplemented function.')


class ContainerField(LeafField):
    """container node can have (container/leaf) child node"""
    def __init__(self, capacity):
        super(ContainerField, self).__init__(capacity)
        self.child = OrderedDict()
        self.child_placeholder = []

    @staticmethod
    def init_container():
        '''Initialise containers in config'''
        return {}

    def is_leaf(self):
        return False

    def add_child(self, name, item):
        '''add child node to this container node'''
        self.child[name] = item
        item.set_parent(self)
        return item

    def get_child(self, name):
        '''get a child from all children'''
        return self.child[name]

    def get_keys(self):
        '''get all children names'''
        return list(self.child)

    def add_placeholder(self, item):
        '''add a specifial container node as child'''
        self.child_placeholder.append(item)
        item.set_parent(self)
        return item

    def fill_default(self, config):
        '''fill default value for all children'''
        for item in self.get_keys():
            child = self.get_child(item)
            if child.is_leaf() and item in config:
                continue

            if child.has_default():
                if child.is_leaf():
                    config[item] = child.default_value()
                    LOGGER.logd('fill default value {} for field {}'.format(
                        config[item], item))
                else:
                    if config.get(item) is None:
                        config[item] = child.init_container()
                    child.fill_default(config[item])
        for item in self.child_placeholder:
            item.fill_default(config)

    def check_not_found_msg(self, name, item): # pylint: disable=no-self-use
        '''raise an error if this node is not found'''
        raise ValueError(
            "{} is an invalid layer name or parameter.".format(item))

    def check(self, name, value):
        '''recursive check all children'''
        config = value
        for item in config:
            if item in self.get_keys():
                child = self.get_child(item)
                child.check(item, config[item])
                continue
            for plh in self.child_placeholder:
                if plh.match(item):
                    plh.check(item, config[item])
                    break
            else:
                self.check_not_found_msg(name, item)

    def sort(self, config):
        '''sort all child field'''
        ordered_config = OrderedDict()
        for item in self.get_keys():
            if item in config:
                child = self.get_child(item)
                if child.is_leaf():
                    ordered_config[item] = config[item]
                else:
                    ordered_config[item] = child.sort(config[item])

        for plh in self.child_placeholder:
            match_list = []
            for item in config:
                if item in ordered_config:
                    continue
                if plh.match(item):
                    match_list.append(item)

            match_list.sort()
            for item in match_list:
                ordered_config[item] = plh.sort(config[item])
        return ordered_config


class PlaceholderField(ContainerField):
    '''a specifial container node'''
    def match(self, name): # pylint: disable=no-self-use
        '''indicate whether input name match this placeholder'''
        raise RuntimeError('Unimplemented function.')


class VersionField(LeafField):
    '''an object for version field'''
    def default_value(self):
        return VERSION

    def check(self, name, value):
        _check_type(name, value, int)
        if value != 1:
            raise ValueError("The %s must be 1." % (name))


class BatchNumField(LeafField):
    '''an object for batch_num field'''
    def default_value(self):
        return BATCH_NUM

    def check(self, name, value):
        _check_type(name, value, int)
        if value <= 0:
            raise ValueError("The %s must be greater than zero." % (name))


class ActOffsetField(LeafField):
    '''an object for activation_offset field'''
    def default_value(self):
        return ACTIVATION_OFFSET

    def check(self, name, value):
        _check_type(name, value, bool)


class LayerActOffsetField(LeafField):
    '''an object for single_layer_activation_offset field'''
    def has_default(self):
        return False

    def check(self, name, value):
        _check_type(name, value, bool)


class JointQuantField(LeafField):
    '''an object for join_quant field'''
    def default_value(self):
        return JOINT_QUANT

    def check(self, name, value):
        _check_type(name, value, bool)


class WtsOffsetField(LeafField):
    '''an object for weight_offset field'''
    def default_value(self):
        return WEIGHT_OFFSET

    def check(self, name, value):
        _check_type(name, value, bool)


class DoFusionField(LeafField):
    '''an object for do_fusion field'''
    def default_value(self):
        return DO_FUSION

    def check(self, name, value):
        _check_type(name, value, bool)


class ActNumBitsField(LeafField):
    '''an object for act num bits field'''
    def default_value(self):
        return DEFAULT_NUM_BITS

    def check(self, name, value):
        _check_type(name, value, int)
        if value not in ACT_SUPPORT_NUM_BITS:
            raise ValueError("activation num_bits is not in {}".format(ACT_SUPPORT_NUM_BITS))


class FakequantPrecisionModeField(LeafField):
    '''an object for fakequant_precision_mode field'''
    def default_value(self):
        return FakequantPrecisionMode.DEFAULT.value

    def check(self, name, value):
        _check_type(name, value, str)
        if value not in FAKEQUANT_PRECISION_MODE:
            raise ValueError("Type of {} should be {}, but is {}".format(name, FAKEQUANT_PRECISION_MODE, value))


class WtsNumBitsField(LeafField):
    '''an object for wts num bits field'''
    def default_value(self):
        return DEFAULT_NUM_BITS

    def check(self, name, value):
        _check_type(name, value, int)
        if value not in WTS_SUPPORT_NUM_BITS:
            raise ValueError("weight num_bits is not in {}".format(WTS_SUPPORT_NUM_BITS))


class SkipFusionLayersField(LeafField):
    '''an object for skip_fusion_layers field'''
    def default_value(self):
        return SKIP_FUSION_LAYERS

    def check(self, name, value):
        _check_type(name, value, list)

        for layer in value:
            _check_type('layer %s in ' % (layer) + name, layer, str)

        repeated_items = find_repeated_items(value)
        check_no_repeated(repeated_items, 'skip_fusion_layers')

        fuse_types = self.capacity.get_value('FUSE_TYPES')
        layer_type = PARAM_POOL.get_layer_type()
        for layer in value:
            if layer not in layer_type:
                raise ValueError('Layer "{}" in skip_fusion_layers does not '
                                 'exist in the graph.'.format(layer))
            if layer_type[layer] not in fuse_types:
                raise ValueError('Skip fusion layer "{}"\'s type not in '
                                 'supported list {}'.format(layer, fuse_types))


class LayerPlhField(PlaceholderField):
    '''a container field for layer params'''
    def match(self, name):
        if name in PARAM_POOL.get_layer_type():
            return True
        return False

    def check_not_found_msg(self, name, item):
        '''raise an error if a field is not found'''
        raise ValueError("Unknown parameter {}".format(item))

    def check(self, name, value):
        config = value
        if name not in PARAM_POOL.get_skip_layers() and \
                name not in PARAM_POOL.get_supported_layers():
            raise ValueError("layer {} does not support quant.".format(name))
        PARAM_POOL.set_layer_name(name)
        try:
            super(LayerPlhField, self).check(name, config)
        except ValueError as error:
            error_info = str(error.args[0]) + " in layer " + name
            raise ValueError(error_info) from None
        except TypeError as error:
            error_info = str(error.args[0]) + " in layer " + name
            raise TypeError(error_info) from None

    def fill_default(self, config):
        for item in PARAM_POOL.get_supported_layers():
            PARAM_POOL.set_layer_name(item)
            LOGGER.logd('fill default value for layer {}'.format(item))
            if config.get(item) is None:
                config[item] = {}
            super(LayerPlhField, self).fill_default(config[item])


class QuantEnableField(LeafField):
    '''an object for quant_enable field'''
    def default_value(self):
        layer_name = PARAM_POOL.get_layer_name()
        quant_layers = PARAM_POOL.get_quant_layers()
        return layer_name in quant_layers

    def check(self, name, value):
        _check_type(name, value, bool)


class DMQBalancerParamField(LeafField):
    '''an object for dmq_balancer_param field'''
    def has_default(self):
        return False

    def check(self, name, value):
        _check_type(name, value, float)
        if value < DMQ_BALANCER_RANGE_START or value > DMQ_BALANCER_RANGE_END:
            raise ValueError("The {} must be in range [{}, {}]".format(
                name, DMQ_BALANCER_RANGE_START, DMQ_BALANCER_RANGE_END))


class ActQuantParamsField(ContainerField):
    '''an object for activation_quant_params field'''
    @staticmethod
    def _check_params(config):
        ifmr_params = []
        hfmg_params = []
        for item in config:
            if item in ['search_step', SEARCH_RANGE, 'min_percentile', 'max_percentile']:
                ifmr_params.append(item)
            if item in ['num_of_bins']:
                hfmg_params.append(item)
        if config.get(ACT_ALGO) == 'ifmr':
            ifmr_params.append(ACT_ALGO)
        if config.get(ACT_ALGO) == 'hfmg':
            hfmg_params.append(ACT_ALGO)

        if ifmr_params and hfmg_params:
            raise ValueError('{} and {} can not appear at same time'.format(
                ifmr_params, hfmg_params))
        if hfmg_params:
            PARAM_POOL.set_act_algo('hfmg')
        else:
            PARAM_POOL.set_act_algo('ifmr')

    def check_not_found_msg(self, name, item):
        '''raise an error if this node is not found'''
        raise ValueError("Unknown parameter %s in %s" % (item, name))

    def fill_default(self, config):
        self._check_params(config)
        super(ActQuantParamsField, self).fill_default(config)

    def check(self, name, value):
        self._check_params(value)
        config = value
        super(ActQuantParamsField, self).check(name, config)
        if 'search_step' in config:
            if SEARCH_RANGE in config:
                search_range = config.get(SEARCH_RANGE)
            else:
                search_range = self.get_child(SEARCH_RANGE).default_value()
            search_step = Decimal(str(config.get('search_step')))
            search_len = Decimal(str(search_range[1])) - \
                         Decimal(str(search_range[0]))
            if search_step > search_len:
                raise ValueError('The search_step must be less than or equal'
                                 ' to (search_range[1] - search_range[0])')


class AdaAlgField(LeafField):
    '''a base object for ada field'''
    def has_default(self):
        if PARAM_POOL.get_wts_algo() == 'arq_quantize':
            return False
        return True


class BetaRangeField(AdaAlgField):
    '''an object for beta_range field'''
    def default_value(self):
        return [BETA_RANGE_START, BETA_RANGE_END]

    def check(self, name, value):
        _check_type(name, value, list)
        if len(value) != 2:
            raise ValueError('The {} must have two floating-point number'.format(name))
        if value[0] <= value[1]:
            raise ValueError('{} beta_range_start must large than beta_range_end'.format(name))
        if value[1] <= 0:
            raise ValueError('{} beta_range_end must large than 0'.format(name))


class WarmStartField(AdaAlgField):
    '''an object for warm_start field'''

    def default_value(self):
        return WART_START

    def check(self, name, value):
        if value >= 1 or value <= 0:
            raise ValueError('{} must be less than 1 and larger than 0' .format(name))


class NumIterationField(AdaAlgField):
    '''an object for num_iteration field'''
    def default_value(self):
        return NUM_ITERATION
    
    def check(self, name, value):
        if value < 0:
            raise ValueError('{} must be larger or equal to 0'.format(name))


class RegParamField(AdaAlgField):
    '''an object for reg_param field'''
    def default_value(self):
        return REG_PARAM

    def check(self, name, value):
        if value >= 1 or value <= 0:
            raise ValueError("%s must be less than 1 and larger than 0" % (name))


class MaxPercentileField(LeafField):
    '''an object for max_percentile field'''
    def has_default(self):
        if PARAM_POOL.get_act_algo() == 'hfmg':
            return False
        return True

    def default_value(self):
        return MAX_PERCENTILE

    def check(self, name, value):
        value = float(value)
        if value <= 0.5 or value > 1.0:
            raise ValueError("The %s must be greater than 0.5 "\
                "and less than or equal to 1.0" % (name))


class MinPercentileField(MaxPercentileField):
    '''an object for min_percentile field'''
    def has_default(self):
        if PARAM_POOL.get_act_algo() == 'hfmg':
            return False
        return True

    def default_value(self):
        return MIN_PERCENTILE


class SearchRangeField(LeafField):
    '''an object for search_range field'''
    def has_default(self):
        if PARAM_POOL.get_act_algo() == 'hfmg':
            return False
        return True

    def default_value(self):
        return [SEARCH_RANGE_START, SEARCH_RANGE_END]

    def check(self, name, value):
        _check_type(name, value, list)
        if len(value) != 2:
            raise ValueError("The %s must have two floating-point number"\
                             % (name))

        search_start = value[0]
        search_end = value[1]
        if search_start <= 0:
            raise ValueError("%s[0] must be greater than zero" % (name))
        if search_start >= search_end:
            raise ValueError("%s[0] must be less than %s[1]" % (name, name))


class SearchStepField(LeafField):
    '''an object for search_step field'''
    def has_default(self):
        if PARAM_POOL.get_act_algo() == 'hfmg':
            return False
        return True

    def default_value(self):
        return SEARCH_STEP

    def check(self, name, value):
        value = float(value)
        if value <= 0:
            raise ValueError("The %s must be greater than zero" % (name))

 
class WgtQuantParamsField(ContainerField):
    '''an object for weight_quant_params field'''
    @staticmethod
    def _check_params(config):
        param_map = {"arq_quantize": ['channel_wise'], 
                     "ada_quantize": ['num_iteration', 'reg_param', 'warm_start', 'beta_range', 'channel_wise']}
        wts_algo = config.get(WTS_ALGO, 'arq_quantize')
        for item in config:
            if item in ['wts_algo', 'num_bits']:
                continue
            if item not in param_map.get(wts_algo):
                raise ValueError('{} is not valid param of quantize {}'.format(
                item, wts_algo))

        PARAM_POOL.set_wts_algo(wts_algo)

    def check_not_found_msg(self, name, item):
        '''raise an error if this node is not found'''
        raise ValueError("Unknown parameter %s in %s" % (item, name))

    def fill_default(self, config):
        self._check_params(config)
        super(WgtQuantParamsField, self).fill_default(config)

    def check(self, name, value):
        self._check_params(value)
        super(WgtQuantParamsField, self).check(name, value)


class ChannelWiseField(LeafField):
    '''an object for channel_wise field'''
    def has_default(self):
        return True

    def default_value(self):
        layer_name = PARAM_POOL.get_layer_name()
        layer_type = PARAM_POOL.get_layer_type()
        channel_wise_types = self.capacity.get_value('CHANNEL_WISE_TYPES')
        return layer_type[layer_name] in channel_wise_types

    def check(self, name, value):
        _check_type(name, value, bool)
        layer_name = PARAM_POOL.get_layer_name()
        layer_type = PARAM_POOL.get_layer_type()
        channel_wise_types = self.capacity.get_value('CHANNEL_WISE_TYPES')
        if layer_type[layer_name] not in channel_wise_types and value:
            raise ValueError('The {} must be False for layer type {}'.format(
                name, layer_type[layer_name]))


class WtsAlgoField(LeafField):
    '''an object for wts_algo field'''
    def default_value(self):
        return PARAM_POOL.get_wts_algo()

    def check(self, name, value):
        if value not in ('arq_quantize', 'ada_quantize'):
            raise ValueError('wts_algo must be arq_quantize or ada_quantize')


class ActAlgoField(LeafField):
    '''an object for act_algo field'''
    def default_value(self):
        return PARAM_POOL.get_act_algo()

    def check(self, name, value):
        if value not in ('ifmr', 'hfmg'):
            raise ValueError('act_algo must be ifmr or hfmg')


class ApproximateAlgoField(LeafField):
    '''an object for approximate_algo field'''
    def default_value(self):
        return "FastSoftmax"

    def check(self, name, value):
        if value != "FastSoftmax":
            raise ValueError('unknwon op approximation algorithm')


class HfmgField(LeafField):
    '''a base object for hfmg field'''
    def has_default(self):
        if PARAM_POOL.get_act_algo() == 'ifmr':
            return False
        return True


class NumOfBinsField(HfmgField):
    '''an object for num_of_iteration field'''
    def default_value(self):
        return NUM_OF_BINS

    def check(self, name, value):
        _check_type(name, value, int)
        if value not in NUM_OF_BINS_RANGE:
            raise ValueError("num_of_bins {} must be in {}".format(
                value, NUM_OF_BINS_RANGE))


class TensorQuantizeField(ContainerField):
    '''container field for tensor quantization'''
    @staticmethod
    def init_container():
        '''Initialise containers in config'''
        return []

    def check(self, name, value):
        '''check tensor quant field'''
        _check_type(name, value, list)
        config = value
        for tensor_config in config:
            super(TensorQuantizeField, self).check(name, tensor_config)

    def sort(self, config):
        '''sort configs'''
        sorted_configs = []
        for tensor_config in config:
            sorted_configs.append(super(TensorQuantizeField, self).sort(tensor_config))
        return sorted_configs

    def fill_default(self, config):
        '''fill default value for container'''
        if config:
            for tensor_config in config:
                super(TensorQuantizeField, self).fill_default(tensor_config)


class LayerNameField(LeafField):
    '''leaf field for op's name'''
    def has_default(self):
        return False

    def check(self, name, value):
        '''check op name field'''
        _check_type(name, value, str)


class InputIndexField(LeafField):
    '''leaf field for op's input index'''
    def has_default(self):
        return False

    def check(self, name, value):
        '''check tensor index field'''
        _check_type(name, value, int)
