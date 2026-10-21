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
__all__ = [
    'quantize',
    'convert',
    'algorithm_register',
]

from torch import nn

from amct_pytorch.algorithms import AlgorithmRegistry
from amct_pytorch.classic.optimizer import (
    InsertQuantizeModulePass,
    InsertLayerNormQuantPass,
    ModelOptimizer,
    ReplaceNpuQuantModulePass,
)
from amct_pytorch.classic.optimizer.apply_scaling_fusion_pass import (
    ApplyScalingFusionPass,
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
    optimizer.add_pass(InsertLayerNormQuantPass())
    if writer is not None:
        _reject_shared_input_quantization(model)
    optimizer.do_optimizer(model)

    if writer is not None:
        _arm_quant_result_publish(model, writer)


def _reject_shared_input_quantization(model):
    """Reject fused shared-input calibration until artifacts can represent it."""
    if any(
        getattr(module, 'shared_param_mode', None) is not None
        for module in model.modules()
    ):
        raise ValueError(
            'quant_result_dir is not supported yet for shared-input fused '
            'quantization (for example qwen3 / ViT LayerNormQuant structures). '
            'Run without quant_result_dir, or disable shared-input fusion.'
        )


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


def _publish_quant_result_if_complete(writer):
    """Return a root-forward hook that publishes and removes itself once complete."""

    def hook(module, args, output):
        if writer.publish_if_complete():
            for hook_id, callback in list(module._forward_hooks.items()):
                if callback is hook:
                    del module._forward_hooks[hook_id]

    return hook


def _arm_quant_result_publish(model, writer):
    """Publish now if complete, otherwise arm the deferred publish hook."""
    if writer.publish_if_complete():
        return
    model.register_forward_hook(_publish_quant_result_if_complete(writer))
    pending = writer.pending_units
    LOGGER.logw(
        "Quant result '{}' has {} pending unit(s) (e.g. '{}'); the manifest "
        'will be published after the next successful forward pass.'.format(
            writer.result_dir, len(pending), pending[0]
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
    optimizer.add_pass(ApplyScalingFusionPass())
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
