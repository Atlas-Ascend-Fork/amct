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
__all__ = ['algorithm_register', 'convert', 'quantize', 'sequential_calibrate']

import torch
from torch import nn

from amct_pytorch.algorithms import AlgorithmRegistry
from amct_pytorch.classic.optimizer import (
    InsertLayerNormQuantPass,
    InsertQuantizeModulePass,
    ModelOptimizer,
    ReplaceNpuQuantModulePass,
)
from amct_pytorch.classic.optimizer.apply_scaling_fusion_pass import (
    ApplyScalingFusionPass,
)
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
from amct_pytorch.quantize_op.base_quant_module import BaseQuantizeModule


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
            f"the model passed to quantize() is itself a quantizable layer ({type(model).__name__}); "
            "each quantized layer must be a named submodule. Wrap it in a parent "
            'nn.Module.'
        )

    writer = None
    if quant_result_dir is not None:
        _reject_auto_smooth_quant_result(layer_config)
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
            f"'{name}' (e.g. layer '{layer}')"
            for name, layer in sorted(unsupported.items())
        )
        raise ValueError(
            f"quant_result_dir is not supported yet for algorithm(s) {details}. "
            "The selected registered quantize class has no artifact contract; "
            "run without quant_result_dir, or choose a class that declares "
            "artifact support."
        )


def _reject_auto_smooth_quant_result(layer_config):
    """Route auto_smooth persistence to sequential_calibrate.

    auto_smooth layers finalize (and dump their params) only through the block
    grid search in sequential_calibrate(); the calibration forward never
    finalizes them, so quantize(..., quant_result_dir=...) would arm a writer
    that can never publish its manifest. Reject it with a pointer to the correct
    API rather than leaving a manifest that never appears.
    """
    auto_smooth_layers = []
    for layer_name, cfg in layer_config.items():
        algorithm = cfg.get('algorithm') if isinstance(cfg, dict) else None
        alg_names, _ = get_alg_name_from_config(algorithm)
        if 'auto_smooth' in alg_names:
            auto_smooth_layers.append(layer_name)
    if not auto_smooth_layers:
        return
    preview = ', '.join(repr(name) for name in auto_smooth_layers[:3])
    if len(auto_smooth_layers) > 3:
        preview += f', ... ({len(auto_smooth_layers)} layers total)'
    raise ValueError(
        'quantize(..., quant_result_dir=...) is not supported for the '
        f'auto_smooth algorithm (layers: {preview}). auto_smooth publishes its '
        'quant result only from sequential_calibrate(model, calib_set, '
        'quant_result_dir=...). Pass quant_result_dir to sequential_calibrate '
        'instead, or run without quant_result_dir.'
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
            f"there is no quant result to write to '{quant_result_dir}'."
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
        f"Quant result '{writer.result_dir}' has {len(pending)} pending unit(s) (e.g. '{pending[0]}'); the manifest "
        'will be published after the next successful forward pass.',
        'quantize',
    )


def _check_auto_smooth_calibrated(model):
    """Reject convert() while auto_smooth layers are still un-finalized.

    auto_smooth finalizes its smoothing strength only through the block grid
    search in sequential_calibrate(); the legacy quantize -> forward -> convert
    flow leaves scale/scale_w/scale_d as None, which would otherwise surface as
    a cryptic 'NoneType' failure in the deployment layer. Fail fast instead.
    """
    from amct_pytorch.classic.quantize_op import SmoothQuant

    pending = [
        name
        for name, module in model.named_modules()
        if isinstance(module, SmoothQuant)
        and module.is_auto_smooth
        and not module.finalized
    ]
    if not pending:
        return
    preview = ', '.join(repr(name) for name in pending[:3])
    if len(pending) > 3:
        preview += f', ... ({len(pending)} layers total)'
    raise ValueError(
        f'auto_smooth layers are not calibrated: {preview}. The auto_smooth '
        'algorithm determines its smoothing strength through the block grid '
        'search in sequential_calibrate(model, calib_set), so it must be called '
        'after quantize() and before convert(). The legacy quantize -> forward '
        '-> convert flow only finalizes fixed smoothquant layers.'
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
                      result produced by quantize(..., quant_result_dir=...) or
                      sequential_calibrate(..., quant_result_dir=...). None keeps
                      the original in-memory behaviour.
    Return: None
    """
    if quant_result_dir is not None:
        # Restores calibration state only; which deploy module each layer becomes
        # is still decided by ReplaceNpuQuantModulePass below.
        reload_quant_params(model, quant_result_dir)

    _check_auto_smooth_calibrated(model)

    optimizer = ModelOptimizer()
    optimizer.add_pass(ApplyScalingFusionPass())
    optimizer.add_pass(ReplaceNpuQuantModulePass())
    optimizer.do_optimizer(model)


def _get_submodule(model, name):
    if name == '':
        return model
    try:
        return model.get_submodule(name)
    except AttributeError as exc:
        raise ValueError(f"block '{name}' is not a submodule of the model") from exc


def _find_block_name(name):
    parts = name.split('.')
    for i in range(len(parts) - 1, 0, -1):
        if parts[i].isdigit():
            return '.'.join(parts[: i + 1])
    return name.rsplit('.', 1)[0] if '.' in name else name


def _detect_blocks(quant_names):
    groups = {}
    for name in quant_names:
        block_name = _find_block_name(name)
        groups.setdefault(block_name, []).append(name)
    return groups


def _resolve_blocks(quant_names, blocks):
    if blocks is None:
        return _detect_blocks(quant_names)
    if isinstance(blocks, dict):
        return dict(blocks)
    result = {}
    for block_name in blocks:
        members = [
            name
            for name in quant_names
            if name == block_name or name.startswith(block_name + '.')
        ]
        result[block_name] = members
    return result


def _resolve_calib_kind(module):
    """
    Resolve the calibrator kind of a quantized module.

    Dispatch is algorithm-agnostic: it only consults the matchers declared by
    the registered calibrators and returns the calibrator kind ('forward' or
    'gradient') whose matcher recognizes the module. Returns None if the module
    is not calibratable by sequential_calibrate.
    """
    from amct_pytorch.common.block_calibrator import iter_registered_matchers
    from amct_pytorch.common.utils.log import LOGGER

    for kind, matcher in iter_registered_matchers():
        try:
            matched = matcher(module)
        except Exception as exc:
            # A matcher must never raise (contract: matcher(module) -> bool).
            # Swallowing the exception would silently drop this module from
            # block calibration, so surface it instead of hiding the failure.
            LOGGER.loge(
                f"block calibrator matcher '{kind}' raised {type(exc).__name__} "
                f"while resolving module kind: {exc}",
                'quantize',
            )
            raise
        if matched:
            return kind
    return None


def _register_block_calibrators():
    # Import for registration side effects. Each module below decorates its
    # calibrator with @register_block_calibrator(kind, matcher=...).
    import amct_pytorch.algorithms.quant.forward
    import amct_pytorch.algorithms.quant.gradient  # noqa: F401


def _arm_sequential_quant_result(model, quant_result_dir, logger):
    """Attach a quant result writer to every artifact-capable quantized module.

    Called before the calibration forward so every module dumps its params at
    its own finalization point (MinMax/AWQ/GPTQ during forward, SmoothQuant in
    finalize). Returns the writer, or None when persistence is disabled.
    """
    if quant_result_dir is None:
        return None

    # Shared-input quantization (LayerNormQuant) cannot be represented by the
    # artifact yet; reject it consistently with the quantize() path.
    _reject_shared_input_quantization(model)

    persist_units = {
        name: module
        for name, module in model.named_modules()
        if is_quant_result_supported(type(module))
    }
    if not persist_units:
        raise RuntimeError(
            f"no quantized module supports quant result persistence for '{quant_result_dir}'."
        )

    writer = QuantResultWriter.create(
        quant_result_dir,
        tuple(persist_units),
        compute_model_fingerprint(model),
    )
    for module in persist_units.values():
        module.attach_quant_result_writer(writer)

    logger.logi(
        f"Armed quant result persistence for {len(persist_units)} layer(s) to '{quant_result_dir}'.",
        'SequentialCalibrate',
    )
    return writer


@check_params(
    model=nn.Module,
    blocks=(list, dict, type(None)),
    quant_result_dir=(str, type(None)),
)
def sequential_calibrate(
    model, calib_set, blocks=None, quant_result_dir=None, **kwargs
):
    """
    Function: Generic per-block sequential calibration entry.

    It only orchestrates the common machinery: partition the model into blocks,
    capture each block's input, iterate the calibration data, then dispatch the
    per-block calibration to a registered block calibrator. Dispatch is
    algorithm-agnostic: a block is routed by calibrator kind ('forward' for
    gradient-free optimization, 'gradient' for gradient-based optimization), so
    any algorithm is supported as long as a calibrator for its kind is
    registered via `register_block_calibrator(kind, matcher=...)`.

    Parameter:
        model: the quantized calibration model produced by quantize().
        calib_set: iterable of calibration inputs; each item is either a
            tensor, a dict passed as keyword arguments, or a tuple/list
            expanded into positional arguments (e.g. (input_ids, attention_mask)).
            The iterable is materialized into a list on entry, so one-shot
            iterables such as generators are replayed once per block.
        blocks: optional list of block module names, or a dict mapping block
            name to its quantized layer names. When None, blocks are inferred
            from the nearest ancestor module with a numeric trailing segment.
        quant_result_dir: optional path of an existing directory to persist the
            quant result artifact into. Independent of quantize(): the writer is
            created here and published once every quantized module supporting
            the artifact contract has dumped its params.
        kwargs: passthrough params for the calibrator, e.g. 'forward_kwargs'
            dict for model.forward, or algorithm-specific 'args'/'data_loader'
            consumed by the 'gradient' calibrator. Two extra logging controls
            are recognized by sequential_calibrate itself:
            'log_blocks' (list of block names) prints the full per-block
            grid-search log only for those blocks; 'log_block_num' (int) prints
            it only for the first N blocks. Other blocks print a one-line
            summary.
    Return: None
    """
    from amct_pytorch.common.block_calibrator import (
        get_block_calibrator,
        registered_kinds,
    )
    from amct_pytorch.common.utils.log import LOGGER

    _register_block_calibrators()

    # Materialize the calibration set so one-shot iterables (generators, an
    # exhausted DataLoader iterator, ...) can be replayed once per block.
    # Otherwise the first block consumes the iterable and every later block sees
    # an empty captured_inputs, silently skipping calibration.
    calib_set = list(calib_set)

    quant_modules = {}
    for name, module in model.named_modules():
        kind = _resolve_calib_kind(module)
        if kind is not None and get_block_calibrator(kind) is not None:
            quant_modules[name] = module

    if not quant_modules:
        raise RuntimeError(
            "sequential_calibrate found no block-calibratable quantized module. "
            "please invoke quantize(model, config) with a supported algorithm "
            "first, or register a block calibrator for kinds ({}).".format(
                ", ".join(registered_kinds())
            )
        )

    writer = _arm_sequential_quant_result(model, quant_result_dir, LOGGER)

    # Per-block verbose logging control. When `log_blocks` is given, only those
    # blocks print the full grid-search log; when `log_block_num` is given, only
    # the first N blocks print it. Other blocks only print a one-line summary.
    log_blocks = kwargs.pop('log_blocks', None)
    log_block_num = kwargs.pop('log_block_num', None)
    log_counter = [0]

    def _block_verbose(block_name):
        if log_blocks is not None:
            return block_name in log_blocks
        if log_block_num is not None:
            if log_counter[0] < log_block_num:
                log_counter[0] += 1
                return True
            return False
        return True

    forward_kwargs = kwargs.get('forward_kwargs', {})
    block_assign = _resolve_blocks(list(quant_modules.keys()), blocks)

    if blocks is not None:
        covered = {
            name for layer_names in block_assign.values() for name in layer_names
        }
        uncovered = [name for name in quant_modules if name not in covered]
        if uncovered:
            preview = ', '.join(repr(name) for name in uncovered[:3])
            if len(uncovered) > 3:
                preview += f', ... ({len(uncovered)} layers total)'
            LOGGER.logw(
                f"sequential_calibrate will not calibrate {len(uncovered)} "
                f"calibratable layer(s) left uncovered by the 'blocks' argument "
                f"(e.g. {preview}); they keep their pre-calibration state.",
                'SequentialCalibrate',
            )

    def _pre_hook_factory(captured_inputs):
        def pre_hook(module, args, kwargs=None):
            if kwargs is None:
                kwargs = {}
            # Captured block input is always an (args, kwargs) pair so the
            # calibrator can replay both position-dependant and keyword inputs
            # (e.g. CLIP's attention_mask as positional, Qwen3's position_embeddings
            # as keyword).
            captured_inputs.append((tuple(args), dict(kwargs)))

        return pre_hook

    # Classic two-pass per-block calibration: only the current block's inputs
    # are retained in memory. Each block hooks itself, the calibration set runs
    # through the whole model once, the block is calibrated immediately and its
    # captured inputs are released before the next block. This trades N_blocks
    # extra calibration passes for bounded memory (one block at a time).
    model.eval()
    for block_name, layer_names in block_assign.items():
        block = _get_submodule(model, block_name)

        captured_inputs = []
        hook = block.register_forward_pre_hook(
            _pre_hook_factory(captured_inputs), with_kwargs=True
        )
        try:
            with torch.no_grad():
                for data in calib_set:
                    if isinstance(data, (tuple, list)):
                        model(*data, **forward_kwargs)
                    elif torch.is_tensor(data):
                        model(data, **forward_kwargs)
                    else:
                        model(**data)
        finally:
            hook.remove()

        if not captured_inputs:
            LOGGER.logw(
                f"sequential_calibrate captured no input for block '{block_name}'; "
                "the block will not be calibrated. This usually means the "
                "calibration set is empty, or the block's forward path was not "
                "executed (e.g. an unused branch).",
                'SequentialCalibrate',
            )
            continue

        missing = [name for name in layer_names if name not in quant_modules]
        if missing:
            preview = ', '.join(repr(name) for name in missing[:3])
            if len(missing) > 3:
                preview += f', ... ({len(missing)} layers total)'
            raise ValueError(
                f"block '{block_name}' references layer(s) not in the "
                f"calibratable set: {preview}. A layer is calibratable only when "
                "it is recognized by a block calibrator's matcher (for example "
                "shared-input members with shared_param_mode is not None are "
                "excluded). Check the 'blocks' argument for misspelled or "
                "excluded layer names."
            )
        members = [quant_modules[name] for name in layer_names]
        members_by_kind = {}
        for module in members:
            kind = _resolve_calib_kind(module)
            members_by_kind.setdefault(kind, []).append(module)
        verbose = _block_verbose(block_name)
        for kind, kind_members in members_by_kind.items():
            calibrator = get_block_calibrator(kind)
            if calibrator is None:
                LOGGER.logw(
                    f"No block calibrator registered for kind '{kind}', skip block "
                    f"'{block_name}'.",
                    'SequentialCalibrate',
                )
                continue
            calibrator(
                model,
                block_name,
                block,
                kind_members,
                captured_inputs,
                _verbose=verbose,
                **kwargs,
            )

    if writer is not None:
        published = writer.publish_if_complete()
        if not published:
            pending = writer.pending_units
            LOGGER.logw(
                f"Quant result '{quant_result_dir}' was NOT published: "
                f"{len(pending)} unit(s) never dumped (e.g. {pending[:10]}); "
                "the manifest is missing, so the quant result cannot be "
                "reloaded by convert(model, quant_result_dir).",
                'SequentialCalibrate',
            )


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
