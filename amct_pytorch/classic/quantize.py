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
__all__ = ['quantize', 'convert', 'algorithm_register']

from torch import nn

from amct_pytorch.algorithms import AlgorithmRegistry
from amct_pytorch.classic.optimizer import (
    InsertQuantizeModulePass,
    ModelOptimizer,
    ReplaceNpuQuantModulePass,
)
from amct_pytorch.quantize_op.base_quant_module import BaseQuantizeModule
from amct_pytorch.classic.quant_result import (
    QuantResultWriter,
    compute_model_fingerprint,
    reload_quant_params,
    validate_quant_result_dir,
)
from amct_pytorch.common.config import parse_config, set_default_config
from amct_pytorch.common.config.utils import get_alg_name_from_config
from amct_pytorch.common.utils.check_params import check_params
from amct_pytorch.common.utils.log import LOGGER


@check_params(
    model=nn.Module, config=(dict, type(None)), quant_result_dir=(str, type(None))
)
def quantize(model, config=None, quant_result_dir=None):
    """
    Function: Modify user's model for quantization
    Parameter: model: user mode instance of Torch.nn.Module
               config: simply config dict from user to set
               quant_result_dir: optional path of an existing directory to persist
                   the quant result artifact into. Unrelated files may coexist,
                   but AMCT never overwrites its own result files. None (default)
                   keeps the model in-memory only and writes nothing.
    Return: None
    """
    if config is None:
        config = set_default_config()

    layer_config = parse_config(model, config, AlgorithmRegistry)

    writer = None
    if quant_result_dir is not None:
        writer = _create_quant_result_writer(model, layer_config, quant_result_dir)

    optimizer = ModelOptimizer()
    if writer is None:
        optimizer.add_pass(InsertQuantizeModulePass(layer_config))
    else:
        optimizer.add_pass(
            InsertQuantizeModulePass(layer_config, quant_result_writer=writer)
        )
    optimizer.do_optimizer(model)

    if writer is not None:
        _finalize_quant_result(model, writer)


def _check_algorithms_support_export(layer_config):
    """Reject algorithms that cannot complete a quant result unit.

    Deriving from BaseQuantizeModule is not sufficient: a quantize op also has
    to declare its ``_quant_param_keys`` and dump at its own finalization point.
    An op that does neither would return from quantize() normally and then leave
    an empty directory whose manifest never publishes, so refuse up front --
    before anything is written -- and name the algorithm at fault.
    """
    unsupported = {}
    for layer_name, cfg in layer_config.items():
        algorithm = cfg.get('algorithm') if isinstance(cfg, dict) else None
        alg_names, _ = get_alg_name_from_config(algorithm)
        alg_name = alg_names[0]  # each module corresponds to only one algo
        for quant_op in (AlgorithmRegistry.algo.get(alg_name) or {}).values():
            supports = getattr(quant_op, 'supports_quant_result_export', None)
            if callable(supports) and supports():
                continue
            unsupported.setdefault(alg_name, layer_name)

    if unsupported:
        details = ', '.join(
            "'{}' (e.g. layer '{}')".format(name, layer)
            for name, layer in sorted(unsupported.items())
        )
        raise ValueError(
            "quant_result_dir is not supported yet for algorithm(s) {}. Only "
            "algorithms whose quantize op declares "
            "_supports_quant_result_export can produce a complete quant "
            "result; run without quant_result_dir, or pick a supported "
            "algorithm.".format(details)
        )


def _create_quant_result_writer(model, layer_config, quant_result_dir):
    """Validate the target dir, fingerprint the source model, build the writer.

    Ordering matters: the path and the algorithms are checked before the
    (potentially expensive) fingerprint, and the fingerprint is taken before any
    quant module is inserted, so it always describes the original model.
    """
    validate_quant_result_dir(quant_result_dir)
    if not layer_config:
        raise ValueError(
            "no quantizable layer was found for this model and config, so "
            "there is no quant result to write to '{}'.".format(quant_result_dir)
        )
    if '' in layer_config:
        # The model root itself matched the config. Module replacement is keyed
        # by name and cannot replace the root, so that layer would never be
        # quantized and its unit could never be dumped, leaving the manifest
        # unpublishable. Fail now with the fix rather than hang pending.
        raise ValueError(
            "the model passed to quantize() is itself a quantizable layer ({}); "
            "quant result export needs each quantized layer to be a named "
            "submodule. Wrap it in a parent nn.Module.".format(type(model).__name__)
        )
    _check_algorithms_support_export(layer_config)
    model_fingerprint = compute_model_fingerprint(model)
    return QuantResultWriter.create(
        quant_result_dir, tuple(layer_config.keys()), model_fingerprint
    )


def _finalize_quant_result(model, writer):
    """Publish now if the artifact is already complete, else arm a hook.

    Weight-only algorithms finalize during module insertion, so the manifest can
    be published as soon as the optimizer succeeds. Activation calibration only
    finalizes during the user's forward passes, so publication is deferred to a
    root forward hook that fires after a successful top-level forward; if that
    forward raises, the hook never runs and no manifest is written.
    """
    if writer.publish_if_complete():
        return

    handle_holder = {}

    def _publish_hook(module, args, output):
        if writer.publish_if_complete():
            handle = handle_holder.pop('handle', None)
            if handle is not None:
                # Artifact is immutable once published; stop checking.
                handle.remove()
        return None

    handle_holder['handle'] = model.register_forward_hook(_publish_hook)
    LOGGER.logd(
        "Quant result '{}' has {} pending unit(s); the manifest will be "
        "published after calibration forward completes.".format(
            writer.result_dir, len(writer.pending_units)
        ),
        'quantize',
    )


@check_params(model=nn.Module, quant_result_dir=(str, type(None)))
def convert(model, quant_result_dir=None):
    """
    Function: Convert quantized calibration model to quantized deployment model
    Parameter: model: quantized calibration model instance of Torch.nn.Module.
                      When quant_result_dir is given, pass the original float
                      model instead: it is restored to its calibrated state from
                      the quant result first, then converted.
               quant_result_dir: str or None, directory of a published quant
                      result produced by quantize(..., quant_result_dir=...).
                      None keeps the original in-memory behaviour.
    Return: None
    """
    if quant_result_dir is not None:
        # Restores calibration state only; which deploy module each layer becomes
        # is still decided by ReplaceNpuQuantModulePass below.
        reload_quant_params(model, quant_result_dir)

    optimizer = ModelOptimizer()
    optimizer.add_pass(ReplaceNpuQuantModulePass())
    optimizer.do_optimizer(model)


@check_params(
    name=str, src_op=str, quant_op=BaseQuantizeModule, deploy_op=(nn.Module, type(None))
)
def algorithm_register(name, src_op, quant_op, deploy_op=None):
    """
    Function: quantization algorithm registration
    Parameter: name: str, algorithm's name
               src_op: str, Source operators corresponding to the algorithm
               quant_op: nn.module or type, Quantized operators corresponding to the algorithm
               deploy_op: nn.module or type, Deployment operators corresponding to the algorithm
    Return: None
    """
    AlgorithmRegistry.register(name, src_op, quant_op, deploy_op=deploy_op)
