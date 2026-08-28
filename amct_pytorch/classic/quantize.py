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
    is_quant_result_supported,
    reload_quant_params,
    validate_quant_result_dir,
)
from amct_pytorch.common.config import parse_config, set_default_config
from amct_pytorch.common.config.utils import get_alg_name_from_config
from amct_pytorch.common.utils.check_params import check_params
from amct_pytorch.common.utils.log import LOGGER


@check_params(
    model=(nn.Module, str),
    config=(dict, type(None)),
    quant_result_dir=(str, type(None)),
)
def quantize(model, config=None, quant_result_dir=None):
    """
    Function: Modify user's model for quantization
    Parameter: model: user model instance for the full pipeline, or a local model
                      directory path for the layerwise pipeline
               config: simply config dict from user to set. config.pipeline may
                   be "full" or "layerwise"; when omitted, model type selects
                   "full" for nn.Module and "layerwise" for str.
               quant_result_dir: full mode accepts an optional existing directory;
                   None keeps the model in memory only. Layerwise mode treats it
                   as the future artifact destination and defaults to
                   "./quant_result"; later execution stages create the directory.
    Return: None
    """
    selected_pipeline = _resolve_pipeline(model, config)
    if selected_pipeline == 'layerwise':
        if isinstance(model, nn.Module):
            raise TypeError(
                "pipeline='layerwise' requires model to be a local directory "
                "path str, not an instantiated nn.Module."
            )
        return _quantize_layerwise(model, config, quant_result_dir)

    if not isinstance(model, nn.Module):
        raise TypeError(
            "pipeline='full' with a model path is part of the final layerwise "
            "design but is not implemented; currently pass an instantiated "
            "nn.Module."
        )
    return _quantize_full(model, config, quant_result_dir)


def _resolve_pipeline(model, config):
    """Resolve config.pipeline, inferring the default from the model type."""
    pipeline = None if config is None else config.get('pipeline')
    if pipeline is not None and not isinstance(pipeline, str):
        raise TypeError(
            "config.pipeline must be 'full' or 'layerwise', got {}.".format(
                type(pipeline).__name__
            )
        )
    if pipeline is None:
        return 'layerwise' if isinstance(model, str) else 'full'
    if pipeline not in ('full', 'layerwise'):
        raise ValueError(
            "config.pipeline must be 'full' or 'layerwise', got {!r}.".format(pipeline)
        )
    return pipeline


def _quantize_full(model, config, quant_result_dir):
    """Original Classic implementation, isolated so its call path stays stable."""
    if config is None:
        config = set_default_config()

    layer_config = parse_config(model, config, AlgorithmRegistry)
    if '' in layer_config:
        # Module replacement is keyed by child name and cannot replace the root.
        raise ValueError(
            "the model passed to quantize() is itself a quantizable layer ({}); "
            "each quantized layer must be a named submodule. Wrap it in a parent "
            'nn.Module.'.format(type(model).__name__)
        )

    writer = None
    if quant_result_dir is not None:
        writer = _create_quant_result_writer(model, layer_config, quant_result_dir)

    optimizer = ModelOptimizer()
    optimizer.add_pass(
        InsertQuantizeModulePass(layer_config, quant_result_writer=writer)
    )
    optimizer.do_optimizer(model)

    if writer is not None:
        _finalize_quant_result(model, writer)



def _quantize_layerwise(model_path, config, quant_result_dir):
    """Build the Week 5 hand-off; Extract and PTQ are added in later slices."""
    from amct_pytorch.classic.layerwise import prepare_layerwise_quantization

    plan = prepare_layerwise_quantization(model_path, config, quant_result_dir)
    raise NotImplementedError(
        "pipeline='layerwise' preparation succeeded for adapter '{}' and target "
        "'{}', but Extract/PTQ execution is not connected to the Classic entry "
        "yet.".format(plan.args.model_name, plan.args.quant_target[0])
    )


D
def _check_quant_result_algorithms(model, layer_config):
    """Screen selected registered quant classes before writing any files.

    Artifact support is declared by the class in ``AlgorithmRegistry.algo``;
    there is no second algorithm-name allowlist to drift out of sync.  The
    writer still validates concrete values when each layer reaches finalization.
    """
    unsupported = {}
    modules = dict(model.named_modules())
    for layer_name, cfg in layer_config.items():
        algorithm = cfg.get('algorithm') if isinstance(cfg, dict) else None
        alg_names, _ = get_alg_name_from_config(algorithm)
        for alg_name in alg_names:
            module = modules.get(layer_name)
            source_type = type(module).__name__ if module is not None else None
            quant_cls = AlgorithmRegistry.algo.get(alg_name, {}).get(source_type)
            if not is_quant_result_supported(quant_cls):
                unsupported.setdefault(alg_name, layer_name)

    if unsupported:
        details = ', '.join(
            "'{}' (e.g. layer '{}')".format(name, layer)
            for name, layer in sorted(unsupported.items())
        )
        raise ValueError(
            "quant_result_dir is not supported yet for algorithm(s) {}. "
            "The selected registered quantize class has no artifact contract; "
            "run without quant_result_dir, or choose a class that declares "
            "artifact support.".format(details)
        )


def _create_quant_result_writer(model, layer_config, quant_result_dir):
    """Validate the target dir, fingerprint the source model, build the writer.

    Ordering matters: the path and the algorithm declarations are checked before
    fingerprinting, and the fingerprint is taken before any quant module is
    inserted, so it always describes the original model.
    """
    validate_quant_result_dir(quant_result_dir)
    if len(layer_config) == 0:
        raise ValueError(
            "no quantizable layer was found for this model and config, so "
            "there is no quant result to write to '{}'.".format(quant_result_dir)
        )
    _check_quant_result_algorithms(model, layer_config)
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
