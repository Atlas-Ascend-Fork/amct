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
"""Quantization result artifact: writer, integrity digests and manifest.

Implements the fixed quant result wire format.
Directory protocol produced by a successful run::

    quant_result_dir/
    |-- quant_manifest.json
    |-- model_layers_0_self_attn_q_proj.safetensors
    `-- ...

``quant_manifest.json`` is written last and only after every expected unit has
been written and re-verified on disk. A directory without the manifest is not a
valid artifact: it is never partially loaded, never resumed and never
overwritten. The manifest is the single source of truth for the full config; the
safetensors header carries only the minimal cross-check fields, so two copies of
the config can never drift.

Field names, the ``sha256:`` digest prefix and the header key set here are part
of the cross-stage contract consumed by ``reload_quant_params()``; do not rename
them for style.
"""

__all__ = [
    'MANIFEST_FORMAT',
    'MANIFEST_NAME',
    'PARAM_FORMAT',
    'SCHEMA_VERSION',
    'UNIT_SUFFIX',
    'QuantResultWriter',
    'canonical_json',
    'compute_model_fingerprint',
    'load_quant_manifest',
    'read_quant_unit',
    'reload_quant_params',
    'resolve_layer_parent',
    'resolve_unit_path',
    'to_jsonable',
    'to_jsonable_lossy',
    'unit_file_name',
    'unit_header',
    'validate_quant_result_dir',
    'validate_result_directory',
    'validate_source_model',
]

import hashlib
import json
import os
import re
import stat
import uuid

import copy

import torch
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import save_file

from amct_pytorch.common.utils.log import LOGGER

MANIFEST_NAME = 'quant_manifest.json'
# Frozen schema constants. MANIFEST_FORMAT/PARAM_FORMAT identify the two artifacts;
# SCHEMA_VERSION is an int in the manifest and its str form in the header.
MANIFEST_FORMAT = 'amct-quant-result-manifest'
PARAM_FORMAT = 'amct-quant-param'
SCHEMA_VERSION = 1
UNIT_SUFFIX = '.safetensors'
LIBRARY = 'amct_pytorch'
LOG_TAG = 'QuantResultWriter'
DIGEST_PREFIX = 'sha256:'

# 640 for files, matching amct_log conventions.
_FILE_MODE = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP

_UNSAFE_NAME_CHARS = re.compile(r'[^0-9a-zA-Z_]')
_HASH_CHUNK_SIZE = 1024 * 1024
# Fields derived from the file itself, therefore excluded from entry_digest.
_DERIVED_ENTRY_FIELDS = ('file_size', 'sha256', 'entry_digest')


def canonical_json(data):
    """Serialize to the one canonical JSON form used by every digest."""
    return json.dumps(
        data,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _prefixed(hexdigest):
    """Wire form of a digest: ``sha256:<hex>``."""
    return DIGEST_PREFIX + hexdigest


def sha256_of_text(text):
    """Prefixed SHA-256 of a str, encoded as UTF-8."""
    return _prefixed(hashlib.sha256(text.encode('utf-8')).hexdigest())


def digest_of(data):
    """Prefixed SHA-256 over the canonical JSON form of ``data``."""
    return sha256_of_text(canonical_json(data))


def sha256_of_file(file_path):
    """Prefixed SHA-256 of a file, read in bounded chunks."""
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as file_handle:
        for chunk in iter(lambda: file_handle.read(_HASH_CHUNK_SIZE), b''):
            hasher.update(chunk)
    return _prefixed(hasher.hexdigest())


def _byte_view(tensor):
    """Flat contiguous uint8 view of a CPU tensor, dtype-agnostic.

    ``Tensor.numpy()`` refuses bfloat16/float8 dtypes, so reinterpret the raw
    storage as uint8 first. Flattening before the view keeps non-contiguous and
    0-d inputs working.
    """
    return tensor.detach().reshape(-1).contiguous().view(torch.uint8)


def update_hash_with_tensor(hasher, tensor, chunk_elems=_HASH_CHUNK_SIZE):
    """Fold a tensor's raw bytes into ``hasher`` without materializing a copy."""
    byte_view = _byte_view(tensor.cpu())
    total = byte_view.numel()
    for start in range(0, total, chunk_elems):
        hasher.update(byte_view[start : start + chunk_elems].numpy().tobytes())


def dtype_name(dtype):
    """Artifact dtype spelling: ``float32``, not ``torch.float32``."""
    text = str(dtype)
    return text[len('torch.') :] if text.startswith('torch.') else text


def tensor_schema_list(tensors):
    """Tensor schema: a key-sorted list of ``{key, dtype, shape}``."""
    return [
        {
            'key': key,
            'dtype': dtype_name(tensors[key].dtype),
            'shape': list(tensors[key].shape),
        }
        for key in sorted(tensors)
    ]


def to_jsonable(value, path='quant_config'):
    """Strictly coerce a value for the manifest, or raise.

    Only the types the artifact contract admits are converted: str/bool/int/float,
    None, ``torch.dtype``, and list/tuple/set/dict of those. Anything else --
    a module, a tensor, an arbitrary object -- is rejected rather than
    stringified, because a stringified object cannot reconstruct a module on
    reload and would silently produce an unusable artifact.
    """
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if value != value or value in (float('inf'), float('-inf')):
            raise ValueError(
                "{} is the non-finite float {!r}, which canonical JSON cannot "
                "represent.".format(path, value)
            )
        return value
    if isinstance(value, torch.dtype):
        return dtype_name(value)
    if isinstance(value, dict):
        return _strict_mapping(value, path)
    if isinstance(value, (set, frozenset)):
        items = [to_jsonable(item, '{}[]'.format(path)) for item in value]
        return sorted(items, key=canonical_json)
    if isinstance(value, (list, tuple)):
        return [
            to_jsonable(item, '{}[{}]'.format(path, index))
            for index, item in enumerate(value)
        ]
    raise TypeError(
        "{} holds {} ({!r}), which cannot be serialized into a quant result "
        "manifest. Only scalars, torch.dtype and containers of those are "
        "allowed, so that reload can reconstruct the module.".format(
            path, type(value).__name__, value
        )
    )


def _strict_mapping(mapping, path):
    """Strictly convert a dict; keys must be str, bool or int."""
    result = {}
    for key, item in mapping.items():
        if not isinstance(key, (str, bool, int)):
            raise TypeError(
                "{} has a {} key ({!r}); quant result config keys must be "
                "str, bool or int.".format(path, type(key).__name__, key)
            )
        json_key = key if isinstance(key, str) else str(key)
        if json_key in result:
            raise ValueError(
                "{} key {!r} collides with another key after string "
                "normalization.".format(path, json_key)
            )
        result[json_key] = to_jsonable(item, '{}.{}'.format(path, json_key))
    return result


_ADDRESS_PATTERN = re.compile(r'0x[0-9a-fA-F]{6,}')


def to_jsonable_lossy(value):
    """Coerce anything into a stable digestible form, stringifying unknowns.

    Only for the *fingerprint* of a source model config: a HuggingFace
    ``config.to_dict()`` may legitimately carry arbitrary objects, and the
    fingerprint only needs to be deterministic and change-detecting, never
    reversible. Never use this for ``quant_config`` or ``non_tensor_params`` --
    those must survive a reload, so they go through the strict
    :func:`to_jsonable`.
    """
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if value != value or value in (float('inf'), float('-inf')):
            return repr(value)
        return value
    if isinstance(value, torch.dtype):
        return dtype_name(value)
    if isinstance(value, torch.Tensor):
        return {
            'dtype': dtype_name(value.dtype),
            'shape': list(value.shape),
            'sha256': tensor_bytes_sha256(value),
        }
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).hex()
    if isinstance(value, dict):
        return {
            (key if isinstance(key, str) else _stringify_unknown(key)): (
                to_jsonable_lossy(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, (set, frozenset)):
        return sorted((to_jsonable_lossy(item) for item in value), key=canonical_json)
    if isinstance(value, (list, tuple)):
        return [to_jsonable_lossy(item) for item in value]
    return _stringify_unknown(value)


def _stringify_unknown(value):
    """Stable string form for a type JSON cannot represent.

    ``str()`` of many objects embeds the memory address, which would make the
    fingerprint differ between runs; fall back to the qualified type name then.
    """
    text = str(value)
    if _ADDRESS_PATTERN.search(text):
        value_type = type(value)
        return '{}.{}'.format(value_type.__module__, value_type.__qualname__)
    return text


def tensor_bytes_sha256(tensor):
    """Prefixed SHA-256 over the raw bytes of a tensor."""
    hasher = hashlib.sha256()
    update_hash_with_tensor(hasher, tensor)
    return _prefixed(hasher.hexdigest())


def unit_file_name(layer_name):
    """Map a layer name to its unit file name.

    Every character outside ``[0-9a-zA-Z_]`` is replaced, so the result can
    never contain a path separator, ``..`` or any other traversal construct.
    Distinct layer names can normalize to the same file name (``a.b`` and
    ``a-b`` both give ``a_b``); the writer keeps a reverse map and rejects such
    collisions instead of silently overwriting.
    """
    if not isinstance(layer_name, str) or not layer_name:
        raise ValueError(
            "layer name for a quant result unit must be a non-empty str, "
            "got {!r}.".format(layer_name)
        )
    sanitized = _UNSAFE_NAME_CHARS.sub('_', layer_name)
    if sanitized.strip('_') == '':
        raise ValueError(
            "layer name '{}' normalizes to an empty file name.".format(layer_name)
        )
    return sanitized + UNIT_SUFFIX


def resolve_unit_path(result_dir, file_name):
    """Absolute path of a unit file, rejecting anything outside result_dir.

    Also enforced on read, since a manifest is untrusted input: an absolute
    path, a ``..`` segment or a nested path must never escape the artifact.
    """
    if not isinstance(file_name, str) or not file_name:
        raise ValueError("quant result unit file name must be a non-empty str.")
    if os.path.isabs(file_name) or os.path.dirname(file_name):
        raise ValueError(
            "quant result unit file '{}' must be a bare relative file name; "
            "absolute paths and directory components are rejected.".format(file_name)
        )
    root = os.path.realpath(result_dir)
    candidate = os.path.realpath(os.path.join(root, file_name))
    if os.path.dirname(candidate) != root:
        raise ValueError(
            "quant result unit '{}' resolves outside the result directory '{}'.".format(
                file_name, result_dir
            )
        )
    return candidate


def validate_quant_result_dir(quant_result_dir):
    """Validate a caller-created directory before writing quant results.

    Only ``None`` means "do not persist". Everything else must be a usable,
    existing empty directory. Files, missing paths, non-empty directories and
    inaccessible directories are rejected, so an existing result is never
    overwritten or resumed.
    """
    if not isinstance(quant_result_dir, str):
        raise TypeError(
            "quant_result_dir must be a str path or None, got {}.".format(
                type(quant_result_dir).__name__
            )
        )
    if quant_result_dir.strip() == '':
        raise ValueError(
            "quant_result_dir must be a non-empty path; pass None to disable "
            "quant result persistence."
        )

    target = os.path.abspath(os.path.expanduser(quant_result_dir))
    if not os.path.exists(target):
        raise ValueError(
            "quant_result_dir '{}' must be an existing directory; create it "
            "before calling quantize().".format(target)
        )
    if not os.path.isdir(target):
        raise ValueError(
            "quant_result_dir '{}' must be an existing directory, not a file.".format(
                target
            )
        )
    if not os.access(target, os.R_OK | os.W_OK | os.X_OK):
        raise ValueError(
            "quant_result_dir '{}' must be readable, writable, and searchable.".format(
                target
            )
        )
    if os.listdir(target):
        raise ValueError(
            "quant_result_dir '{}' must be empty; AMCT never overwrites or "
            "resumes an existing quant result.".format(target)
        )
    return target


def temp_name_for(file_path):
    """Same-directory temp name for an atomic write of ``file_path``."""
    return os.path.join(
        os.path.dirname(file_path),
        '.{}.{}.tmp'.format(os.path.basename(file_path), os.getpid()),
    )


def _atomic_write_via(file_path, write_payload):
    """Atomically produce file_path: temp file -> fsync -> os.replace.

    ``write_payload(tmp_path)`` does the actual writing, so a large tensor
    payload can stream straight to disk instead of being serialized into a bytes
    object first. Temp file and target share a directory, hence a filesystem.
    """
    tmp_path = temp_name_for(file_path)
    try:
        write_payload(tmp_path)
        _fsync_file(tmp_path)
        os.chmod(tmp_path, _FILE_MODE)
        os.replace(tmp_path, file_path)
    except BaseException:
        # Leave no partial file behind, for any failure including interrupts.
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
    _fsync_dir(os.path.dirname(file_path))


def _fsync_file(file_path):
    """Flush a just-written file's contents to stable storage."""
    file_descriptor = os.open(file_path, os.O_RDONLY)
    try:
        os.fsync(file_descriptor)
    finally:
        os.close(file_descriptor)


def _fsync_dir(directory):
    """fsync a directory so a rename is durable; best effort by platform."""
    try:
        dir_fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(dir_fd)
    except OSError:
        pass
    finally:
        os.close(dir_fd)


def _write_text_atomically(file_path, text):
    """Atomically write UTF-8 text (used for the manifest)."""

    def _write(tmp_path):
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        file_descriptor = os.open(tmp_path, flags, _FILE_MODE)
        with os.fdopen(file_descriptor, 'w', encoding='utf-8') as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())

    _atomic_write_via(file_path, _write)


def _model_structure_config(model):
    """Structural description of a plain ``nn.Module`` (no HF config).

    Records the model class plus every submodule name and type, which is what
    makes the fingerprint sensitive to architecture rather than only weights.
    """
    model_type = type(model)
    return {
        'model_class': '{}.{}'.format(model_type.__module__, model_type.__qualname__),
        'modules': [
            {'name': name, 'type': type(module).__name__}
            for name, module in sorted(model.named_modules(), key=lambda kv: kv[0])
        ],
        # Parameter shapes make this digest architecture-sensitive: two models
        # with the same module types but different widths must not share a
        # config digest, so a mismatch reported at reload distinguishes "wrong
        # architecture" from "wrong weights".
        'parameters': [
            {'name': name, 'dtype': dtype_name(param.dtype), 'shape': list(param.shape)}
            for name, param in sorted(model.named_parameters(), key=lambda kv: kv[0])
        ],
    }


def _source_model_type(model):
    """Artifact ``source_model.model_type``: HF model_type, else class name."""
    config = getattr(model, 'config', None)
    model_type = getattr(config, 'model_type', None)
    if isinstance(model_type, str) and model_type:
        return model_type
    return type(model).__name__


def _model_config_digest(model):
    """Digest of the model config: HF ``to_dict()`` when present, else structure."""
    config = getattr(model, 'config', None)
    to_dict = getattr(config, 'to_dict', None)
    if callable(to_dict):
        return digest_of(to_jsonable_lossy(to_dict()))
    return digest_of(to_jsonable_lossy(_model_structure_config(model)))


def _state_dict_digest(model):
    """Deterministic fingerprint of the state_dict: schema plus raw bytes.

    Streams each tensor through the hash on CPU, so an LLM state_dict is never
    copied in full.
    """
    state_dict = model.state_dict(keep_vars=False)
    hasher = hashlib.sha256()
    for key in sorted(state_dict.keys()):
        value = state_dict[key]
        if not isinstance(value, torch.Tensor):
            hasher.update(
                canonical_json({'key': key, 'value': to_jsonable_lossy(value)}).encode(
                    'utf-8'
                )
            )
            continue
        hasher.update(
            canonical_json(
                {
                    'key': key,
                    'dtype': dtype_name(value.dtype),
                    'shape': list(value.shape),
                }
            ).encode('utf-8')
        )
        update_hash_with_tensor(hasher, value)
    return _prefixed(hasher.hexdigest())


def compute_model_fingerprint(model):
    """Build the source_model block before any quant module is inserted.

    Uses the HuggingFace config when present, and otherwise a structural config
    derived from the model class and submodule types, so plain ``nn.Module``
    models are supported too.
    """
    return {
        'model_type': _source_model_type(model),
        'config_sha256': _model_config_digest(model),
        'state_dict_sha256': _state_dict_digest(model),
    }


def _library_version():
    """Version string shipped in ``amct_pytorch/.version``."""
    version_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.version')
    try:
        with open(version_file, encoding='utf-8') as file_handle:
            return file_handle.readline().strip() or 'unknown'
    except OSError:
        return 'unknown'


def unit_header(artifact_id, layer_name, entry_digest):
    """The minimal safetensors header: five ``str -> str`` fields.

    Deliberately excludes quant_module_type/ori_module_type/quant_config/
    non_tensor_params/tensor schema/file digests -- the manifest is their only
    home, so the two copies cannot drift.
    """
    return {
        'format': PARAM_FORMAT,
        'schema_version': str(SCHEMA_VERSION),
        'artifact_id': artifact_id,
        'layer_name': layer_name,
        'entry_digest': entry_digest,
    }


class QuantResultWriter:
    """Collects per-layer quant params and publishes a quant result artifact.

    Lifecycle: one ``write_unit()`` per expected layer at that layer's own
    finalization point, then ``publish_if_complete()``. The manifest appears
    only when every expected unit has been written and re-verified, so an
    interrupted run leaves an unpublished (invalid) directory rather than a
    half-usable one.
    """

    def __init__(
        self,
        result_dir,
        expected_units,
        source_model,
        artifact_id=None,
    ):
        self._result_dir = os.path.abspath(result_dir)
        self._expected_units = self._check_expected_units(expected_units)
        self._source_model = source_model
        self._artifact_id = artifact_id or str(uuid.uuid4())
        self._library_version = _library_version()
        self._entries = {}
        self._file_names = {}
        self._published = False

    @staticmethod
    def _check_expected_units(expected_units):
        """Validate the expected unit set: non-empty, unique, all str."""
        units = list(expected_units)
        if not units:
            raise ValueError(
                "no quantizable layer was found for this model and config, so "
                "there is no quant result to persist."
            )
        for unit in units:
            if not isinstance(unit, str) or not unit:
                raise ValueError(
                    "expected quant result unit names must be non-empty str, "
                    "got {!r}.".format(unit)
                )
        if len(set(units)) != len(units):
            raise ValueError("expected quant result unit names contain duplicates.")
        return tuple(units)

    @classmethod
    def create(cls, quant_result_dir, expected_units, source_model):
        """Validate inputs and return a writer for a caller-created directory."""
        units = cls._check_expected_units(expected_units)
        result_dir = validate_quant_result_dir(quant_result_dir)
        return cls(result_dir, units, source_model)

    @property
    def result_dir(self):
        return self._result_dir

    @property
    def artifact_id(self):
        return self._artifact_id

    @property
    def manifest_path(self):
        return os.path.join(self._result_dir, MANIFEST_NAME)

    @property
    def published(self):
        return self._published

    @property
    def expected_units(self):
        return self._expected_units

    @property
    def written_units(self):
        return tuple(sorted(self._entries))

    @property
    def pending_units(self):
        """Expected units that still have no entry, in declaration order."""
        return tuple(unit for unit in self._expected_units if unit not in self._entries)

    def write_unit(
        self,
        layer_name,
        tensors,
        non_tensor_params=None,
        quant_config=None,
        quant_module_type=None,
        ori_module_type=None,
    ):
        """Persist one layer's quant params as an atomic safetensors unit.

        Registers the unit in memory only; the manifest is published separately
        once every expected unit has landed.
        """
        self._check_writable(layer_name)
        file_name = self._reserve_file_name(layer_name)
        tensors = self._check_tensors(layer_name, tensors)
        # Required by the schema the reader enforces: writing None here would
        # publish an artifact that cannot be reloaded.
        _non_empty_str(
            quant_module_type,
            "quant_module_type of layer '{}'".format(layer_name),
        )
        _non_empty_str(
            ori_module_type, "ori_module_type of layer '{}'".format(layer_name)
        )
        quant_config = _mapping_or_empty(
            quant_config, "quant_config of layer '{}'".format(layer_name)
        )
        non_tensor_params = _mapping_or_empty(
            non_tensor_params, "non_tensor_params of layer '{}'".format(layer_name)
        )
        entry = {
            'layer_name': layer_name,
            'file': file_name,
            'quant_module_type': quant_module_type,
            'ori_module_type': ori_module_type,
            'quant_config': to_jsonable(
                quant_config, "quant_config of layer '{}'".format(layer_name)
            ),
            'non_tensor_params': to_jsonable(
                non_tensor_params,
                "non_tensor_params of layer '{}'".format(layer_name),
            ),
            'tensors': tensor_schema_list(tensors),
        }
        self._check_jsonable(layer_name, entry)
        entry['entry_digest'] = digest_of(entry)

        file_path = resolve_unit_path(self._result_dir, file_name)
        header = unit_header(self._artifact_id, layer_name, entry['entry_digest'])
        # save_file streams straight to the temp path: a large tensor payload is
        # never serialized into an in-memory bytes object first.
        _atomic_write_via(
            file_path,
            lambda tmp_path: save_file(tensors, tmp_path, metadata=header),
        )
        # Read the unit back before trusting it, then bind size and digest.
        self._verify_unit_file(file_path, entry, header)
        entry['file_size'] = os.path.getsize(file_path)
        entry['sha256'] = sha256_of_file(file_path)

        self._entries[layer_name] = entry
        LOGGER.logd(
            "Wrote quant result unit for layer '{}' to '{}'".format(
                layer_name, file_name
            ),
            LOG_TAG,
        )
        return entry

    def _check_writable(self, layer_name):
        """Reject writes after publish, for unknown layers, or duplicates."""
        if self._published:
            raise RuntimeError(
                "quant result '{}' is already published; it is immutable.".format(
                    self._result_dir
                )
            )
        if layer_name not in self._expected_units:
            raise ValueError(
                "layer '{}' is not part of the expected quant result units; "
                "refusing to write an unexpected unit.".format(layer_name)
            )
        if layer_name in self._entries:
            raise ValueError(
                "quant result unit for layer '{}' was already written; each "
                "layer may be dumped only once.".format(layer_name)
            )

    def _reserve_file_name(self, layer_name):
        """Reserve the unit file name, rejecting post-sanitization collisions."""
        file_name = unit_file_name(layer_name)
        owner = self._file_names.get(file_name)
        if owner is not None and owner != layer_name:
            raise ValueError(
                "layers '{}' and '{}' both map to quant result file '{}'; "
                "refusing to overwrite.".format(owner, layer_name, file_name)
            )
        self._file_names[file_name] = layer_name
        return file_name

    @staticmethod
    def _check_tensors(layer_name, tensors):
        """Validate the tensor payload: str keys, CPU, contiguous, detached."""
        if not isinstance(tensors, dict):
            raise TypeError(
                "quant param tensors of layer '{}' must be a dict, got {}.".format(
                    layer_name, type(tensors).__name__
                )
            )
        checked = {}
        for key, value in tensors.items():
            if not isinstance(key, str) or not key:
                raise ValueError(
                    "quant param tensor keys of layer '{}' must be non-empty "
                    "str, got {!r}.".format(layer_name, key)
                )
            if not isinstance(value, torch.Tensor):
                raise TypeError(
                    "quant param '{}' of layer '{}' must be a torch.Tensor, "
                    "got {}.".format(key, layer_name, type(value).__name__)
                )
            checked[key] = value.detach().cpu().contiguous()
        return checked

    @staticmethod
    def _check_jsonable(layer_name, entry):
        """Fail early if the entry cannot be canonically serialized."""
        try:
            canonical_json(entry)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "quant result entry of layer '{}' is not JSON serializable: {}".format(
                    layer_name, error
                )
            ) from error

    @staticmethod
    def _verify_unit_file(file_path, entry, header):
        """Re-read a just-written unit and check its header and tensor schema."""
        with safe_open(file_path, framework='pt') as file_handle:
            actual_header = file_handle.metadata() or {}
            actual = {key: file_handle.get_tensor(key) for key in file_handle.keys()}
        if actual_header != header:
            raise RuntimeError(
                "quant result unit '{}' header mismatch after write: expected "
                "{}, found {}.".format(file_path, header, actual_header)
            )
        actual_schema = tensor_schema_list(actual)
        if actual_schema != entry['tensors']:
            raise RuntimeError(
                "quant result unit '{}' tensor schema mismatch after write: "
                "expected {}, found {}.".format(
                    file_path, entry['tensors'], actual_schema
                )
            )

    def publish_if_complete(self):
        """Publish the manifest iff every expected unit has been written.

        Returns True when the artifact is published (including a previous
        publish), False when units are still pending. Called at each candidate
        finalization point, so the manifest lands as soon as -- and only when --
        the artifact is complete.
        """
        if self._published:
            return True
        pending = self.pending_units
        if pending:
            LOGGER.logd(
                "Quant result not published yet, {} unit(s) pending, e.g. '{}'".format(
                    len(pending), pending[0]
                ),
                LOG_TAG,
            )
            return False
        self._publish()
        return True

    def _publish(self):
        """Verify the whole file set, then atomically write the manifest last."""
        self._verify_result_dir()
        manifest = {
            'format': MANIFEST_FORMAT,
            'schema_version': SCHEMA_VERSION,
            'artifact_id': self._artifact_id,
            'producer': {
                'library': LIBRARY,
                'library_version': self._library_version,
            },
            'source_model': self._source_model,
            'units': [self._entries[unit] for unit in self._expected_units],
        }
        _write_text_atomically(self.manifest_path, canonical_json(manifest) + '\n')
        self._published = True
        LOGGER.logi(
            "Published quant result manifest '{}' with {} unit(s).".format(
                self.manifest_path, len(manifest['units'])
            ),
            LOG_TAG,
        )

    def _verify_result_dir(self):
        """Check the directory holds exactly the expected units, each intact.

        The file set is compared verbatim -- dot-prefixed entries included -- so
        a stray hidden ``.safetensors`` cannot slip past the check. Only this
        process's own in-flight temp files are exempt, and by this point there
        are none, since every write either replaced or removed its temp file.
        """
        expected_files = {entry['file'] for entry in self._entries.values()}
        actual_files = set(os.listdir(self._result_dir))
        missing = sorted(expected_files - actual_files)
        if missing:
            raise RuntimeError(
                "quant result directory '{}' is missing unit file(s) {}; "
                "refusing to publish an incomplete artifact.".format(
                    self._result_dir, missing
                )
            )
        unexpected = sorted(actual_files - expected_files - {MANIFEST_NAME})
        if unexpected:
            raise RuntimeError(
                "quant result directory '{}' contains file(s) not listed in the "
                "manifest: {}. Refusing to publish, since the artifact requires the "
                "directory to hold exactly the manifest and its units.".format(
                    self._result_dir, unexpected
                )
            )
        for entry in self._entries.values():
            self._verify_entry(entry)

    def _verify_entry(self, entry):
        """Re-check one entry against the file on disk before publishing."""
        file_path = resolve_unit_path(self._result_dir, entry['file'])
        actual_size = os.path.getsize(file_path)
        if actual_size != entry['file_size']:
            raise RuntimeError(
                "quant result unit '{}' changed size after write: expected {} "
                "bytes, found {}.".format(file_path, entry['file_size'], actual_size)
            )
        if sha256_of_file(file_path) != entry['sha256']:
            raise RuntimeError(
                "quant result unit '{}' failed its SHA-256 check before "
                "publish.".format(file_path)
            )
        digest_input = {
            key: value
            for key, value in entry.items()
            if key not in _DERIVED_ENTRY_FIELDS
        }
        if digest_of(digest_input) != entry['entry_digest']:
            raise RuntimeError(
                "quant result entry of layer '{}' failed its entry_digest "
                "check before publish.".format(entry['layer_name'])
            )
        self._verify_unit_file(
            file_path,
            entry,
            unit_header(self._artifact_id, entry['layer_name'], entry['entry_digest']),
        )


_REQUIRED_MANIFEST_FIELDS = (
    'format',
    'schema_version',
    'artifact_id',
    'producer',
    'source_model',
    'units',
)
_REQUIRED_UNIT_FIELDS = (
    'layer_name',
    'file',
    'quant_module_type',
    'ori_module_type',
    'quant_config',
    'non_tensor_params',
    'tensors',
    'file_size',
    'sha256',
    'entry_digest',
)
_REQUIRED_SOURCE_MODEL_FIELDS = ('model_type', 'config_sha256', 'state_dict_sha256')
_REQUIRED_PRODUCER_FIELDS = ('library', 'library_version')
_REQUIRED_TENSOR_FIELDS = ('key', 'dtype', 'shape')
_DIGEST_PATTERN = re.compile(r'sha256:[0-9a-f]{64}')


def load_quant_manifest(quant_result_dir):
    """Load and validate ``quant_manifest.json`` of a published artifact.

    A directory without the manifest is not a valid quant result, so its absence
    is an error rather than an empty read. Parsed with ``json`` only -- never
    pickle -- and every required field, the unit uniqueness rule and the safe
    relative-path rule are enforced before anything is handed back.
    """
    result_dir = os.path.abspath(os.path.expanduser(quant_result_dir))
    manifest_path = os.path.join(result_dir, MANIFEST_NAME)
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(
            "'{}' is not a valid quant result: {} is missing, so the run that "
            "produced it did not complete.".format(quant_result_dir, MANIFEST_NAME)
        )
    with open(manifest_path, encoding='utf-8') as file_handle:
        manifest = json.load(file_handle)
    if not isinstance(manifest, dict):
        raise ValueError(
            "quant manifest '{}' is not a JSON object.".format(manifest_path)
        )

    _check_required(manifest, _REQUIRED_MANIFEST_FIELDS, 'quant manifest')
    if manifest['format'] != MANIFEST_FORMAT:
        raise ValueError(
            "unsupported quant manifest format '{}', expected '{}'.".format(
                manifest['format'], MANIFEST_FORMAT
            )
        )
    # bool is an int subclass and 1.0 == 1, so an exact int is demanded rather
    # than mere equality: the wire format says this field is an integer.
    version = manifest['schema_version']
    if (
        isinstance(version, bool)
        or not isinstance(version, int)
        or version != SCHEMA_VERSION
    ):
        raise ValueError(
            'unsupported quant manifest schema_version {!r}, expected the int '
            '{}.'.format(version, SCHEMA_VERSION)
        )
    _validate_artifact_id(manifest['artifact_id'])
    _validate_producer(manifest['producer'])
    _validate_source_model_fields(manifest['source_model'])
    _validate_units(manifest['units'], result_dir)
    return manifest


def _non_empty_str(value, what):
    """Every identifier in the manifest must be a non-empty string."""
    if not isinstance(value, str) or not value:
        raise ValueError('{} must be a non-empty string, got {!r}.'.format(what, value))
    return value


def _mapping_or_empty(value, what):
    """Normalize an optional mapping field, rejecting anything else.

    ``None`` means "nothing to record" and becomes ``{}``. Deliberately not
    ``value or {}``: that also swallows ``[]``, ``''`` and ``False`` into an empty
    config, and a non-mapping would otherwise be written into a manifest the
    reader is obliged to reject.
    """
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(
            '{} must be a mapping or None, got {}.'.format(what, type(value).__name__)
        )
    return value


def _validate_artifact_id(artifact_id):
    """``artifact_id`` is the UUID the unit headers are cross-checked against."""
    _non_empty_str(artifact_id, 'quant manifest artifact_id')
    try:
        uuid.UUID(artifact_id)
    except (ValueError, AttributeError, TypeError):
        raise ValueError(
            'quant manifest artifact_id {!r} is not a UUID.'.format(artifact_id)
        ) from None


def _validate_producer(producer):
    """``producer`` identifies the library that wrote the artifact."""
    _check_required(producer, _REQUIRED_PRODUCER_FIELDS, 'quant manifest producer')
    for field in _REQUIRED_PRODUCER_FIELDS:
        _non_empty_str(producer[field], 'quant manifest producer.{}'.format(field))
    if producer['library'] != LIBRARY:
        raise ValueError(
            "quant manifest was produced by library '{}', but only '{}' artifacts "
            'can be reloaded here.'.format(producer['library'], LIBRARY)
        )


def _validate_digest(value, what):
    """Digests are always ``sha256:`` followed by 64 lowercase hex chars."""
    _non_empty_str(value, what)
    if not _DIGEST_PATTERN.fullmatch(value):
        raise ValueError(
            '{} must look like "{}<64 hex chars>", got {!r}.'.format(
                what, DIGEST_PREFIX, value
            )
        )


def _validate_source_model_fields(source_model):
    """``source_model`` is the fingerprint reload matches the input model on."""
    _check_required(source_model, _REQUIRED_SOURCE_MODEL_FIELDS, 'source_model')
    _non_empty_str(source_model['model_type'], 'source_model.model_type')
    for field in ('config_sha256', 'state_dict_sha256'):
        _validate_digest(source_model[field], 'source_model.{}'.format(field))


def _check_required(mapping, fields, what):
    """Every required field must be present."""
    if not isinstance(mapping, dict):
        raise ValueError("{} must be a JSON object.".format(what))
    missing = [field for field in fields if field not in mapping]
    if missing:
        raise ValueError("{} is missing required field(s) {}.".format(what, missing))


def _validate_unit_fields(entry):
    """Type-check one unit entry against the frozen artifact schema.

    Every field reload depends on is checked here rather than where it is used, so
    a malformed manifest cannot get as far as touching the model.
    """
    layer_name = entry['layer_name']
    for field in ('layer_name', 'file', 'quant_module_type', 'ori_module_type'):
        _non_empty_str(entry[field], "quant manifest unit {}".format(field))
    for field in ('quant_config', 'non_tensor_params'):
        if not isinstance(entry[field], dict):
            raise ValueError(
                "quant manifest unit '{}' field {} must be a JSON object, got "
                '{}.'.format(layer_name, field, type(entry[field]).__name__)
            )
    # bool is an int subclass, so it has to be excluded explicitly.
    file_size = entry['file_size']
    if isinstance(file_size, bool) or not isinstance(file_size, int) or file_size < 0:
        raise ValueError(
            "quant manifest unit '{}' file_size must be a non-negative int, got "
            '{!r}.'.format(layer_name, file_size)
        )
    _validate_digest(
        entry['sha256'], "quant manifest unit '{}' sha256".format(layer_name)
    )
    _validate_digest(
        entry['entry_digest'],
        "quant manifest unit '{}' entry_digest".format(layer_name),
    )
    _validate_tensor_schema(entry['tensors'], layer_name)


def _validate_tensor_schema(tensors, layer_name):
    """The tensor schema is a list of ``{key, dtype, shape}``, keys unique."""
    if not isinstance(tensors, list):
        raise ValueError(
            "quant manifest unit '{}' tensors must be a list, got {}.".format(
                layer_name, type(tensors).__name__
            )
        )
    seen = set()
    for item in tensors:
        _check_required(
            item,
            _REQUIRED_TENSOR_FIELDS,
            "quant manifest unit '{}' tensor".format(layer_name),
        )
        key = _non_empty_str(
            item['key'], "quant manifest unit '{}' tensor key".format(layer_name)
        )
        _non_empty_str(
            item['dtype'],
            "quant manifest unit '{}' tensor '{}' dtype".format(layer_name, key),
        )
        shape = item['shape']
        if not isinstance(shape, list) or any(
            isinstance(dim, bool) or not isinstance(dim, int) or dim < 0
            for dim in shape
        ):
            raise ValueError(
                "quant manifest unit '{}' tensor '{}' shape must be a list of "
                'non-negative ints, got {!r}.'.format(layer_name, key, shape)
            )
        if key in seen:
            raise ValueError(
                "quant manifest unit '{}' lists tensor '{}' more than once.".format(
                    layer_name, key
                )
            )
        seen.add(key)


def _validate_units(units, result_dir):
    """Validate the unit list: shape, required fields, uniqueness, safe paths."""
    if not isinstance(units, list) or not units:
        raise ValueError("quant manifest units must be a non-empty list.")
    seen_layers = set()
    seen_files = set()
    for entry in units:
        _check_required(entry, _REQUIRED_UNIT_FIELDS, 'quant manifest unit')
        _validate_unit_fields(entry)
        layer_name = entry['layer_name']
        if layer_name in seen_layers:
            raise ValueError(
                "quant manifest lists layer '{}' more than once.".format(layer_name)
            )
        seen_layers.add(layer_name)
        # Raises on absolute paths, directory components and traversal.
        resolve_unit_path(result_dir, entry['file'])
        if entry['file'] in seen_files:
            raise ValueError(
                "quant manifest lists file '{}' more than once.".format(entry['file'])
            )
        seen_files.add(entry['file'])


def validate_result_directory(quant_result_dir, manifest):
    """The directory must hold exactly the manifest and its listed units.

    No ``.safetensors`` may exist outside the manifest, including hidden files.
    """
    result_dir = os.path.abspath(os.path.expanduser(quant_result_dir))
    expected = {entry['file'] for entry in manifest['units']} | {MANIFEST_NAME}
    actual = set(os.listdir(result_dir))
    missing = sorted(expected - actual)
    if missing:
        raise ValueError(
            "quant result '{}' is missing file(s) listed in its manifest: {}.".format(
                result_dir, missing
            )
        )
    unexpected = sorted(actual - expected)
    if unexpected:
        raise ValueError(
            "quant result '{}' contains file(s) not listed in its manifest: {}.".format(
                result_dir, unexpected
            )
        )


def read_quant_unit(quant_result_dir, layer_name, manifest=None):
    """Read one verified unit back as ``(tensors, non_tensor_params, entry)``.

    Checks file size, SHA-256, header, entry digest and tensor schema before
    handing anything back, so a tampered or truncated artifact fails loudly.
    """
    result_dir = os.path.abspath(os.path.expanduser(quant_result_dir))
    manifest = manifest or load_quant_manifest(result_dir)
    entries = {entry['layer_name']: entry for entry in manifest['units']}
    entry = entries.get(layer_name)
    if entry is None:
        raise KeyError(
            "quant result '{}' has no unit for layer '{}'.".format(
                result_dir, layer_name
            )
        )

    file_path = resolve_unit_path(result_dir, entry['file'])
    if not os.path.isfile(file_path):
        raise ValueError("quant result unit '{}' is missing.".format(file_path))
    if os.path.getsize(file_path) != entry['file_size']:
        raise ValueError("quant result unit '{}' has wrong size.".format(file_path))
    if sha256_of_file(file_path) != entry['sha256']:
        raise ValueError(
            "quant result unit '{}' failed its SHA-256 check.".format(file_path)
        )
    digest_input = {
        key: value for key, value in entry.items() if key not in _DERIVED_ENTRY_FIELDS
    }
    if digest_of(digest_input) != entry['entry_digest']:
        raise ValueError(
            "quant result entry of layer '{}' failed its entry_digest check.".format(
                layer_name
            )
        )

    with safe_open(file_path, framework='pt') as file_handle:
        header = file_handle.metadata() or {}
        tensors = {key: file_handle.get_tensor(key) for key in file_handle.keys()}
    expected_header = unit_header(
        manifest['artifact_id'], layer_name, entry['entry_digest']
    )
    if header != expected_header:
        raise ValueError(
            "quant result unit '{}' header does not cross-check with the "
            "manifest.".format(file_path)
        )
    actual_schema = tensor_schema_list(tensors)
    if actual_schema != entry['tensors']:
        raise ValueError(
            "quant result unit '{}' tensor schema does not match the manifest.".format(
                file_path
            )
        )
    return tensors, dict(entry['non_tensor_params']), entry


def validate_source_model(model, source_model):
    """The float model handed to reload must be the one that was calibrated.

    Compared on model type, structure/config digest and full state_dict digest,
    so a different architecture, a changed dtype or shape, or any altered weight
    is refused before a single module is touched.
    """
    # Shape first: a non-mapping must fail as a validation error, not as an
    # AttributeError from inside the strict conversion.
    _check_required(source_model, _REQUIRED_SOURCE_MODEL_FIELDS, 'source_model')
    _strict_mapping(source_model, 'source_model')
    actual = compute_model_fingerprint(model)
    mismatched = [
        (field, source_model[field], actual[field])
        for field in _REQUIRED_SOURCE_MODEL_FIELDS
        if source_model[field] != actual.get(field)
    ]
    if mismatched:
        details = '; '.join(
            '{}: artifact has {!r} but this model has {!r}'.format(*item)
            for item in mismatched
        )
        raise ValueError(
            'the model passed in is not the model this quant result was produced '
            'from ({}). Reload expects the same original float model that was '
            'calibrated.'.format(details)
        )


RELOAD_LOG_TAG = 'reload_quant_params'


def resolve_layer_parent(model, layer_name):
    """Resolve ``layer_name`` to ``(parent_module, attr_name, child_module)``.

    Every failure is a hard error: an empty name, an unreachable path, or a path
    that runs through something that is not a module all mean the artifact does
    not describe this model.
    """
    if not isinstance(layer_name, str) or not layer_name:
        raise ValueError('quant result unit has an empty layer_name.')
    parts = layer_name.split('.')
    if any(not part for part in parts):
        raise ValueError(
            "quant result unit layer_name '{}' has an empty path component.".format(
                layer_name
            )
        )
    parent = model
    for depth, part in enumerate(parts[:-1]):
        child = getattr(parent, part, None)
        if not isinstance(child, nn.Module):
            raise ValueError(
                "cannot reach layer '{}': '{}' is not a submodule of the model "
                'passed in.'.format(layer_name, '.'.join(parts[: depth + 1]))
            )
        parent = child
    attr_name = parts[-1]
    child = getattr(parent, attr_name, None)
    if not isinstance(child, nn.Module):
        raise ValueError(
            "the model passed in has no submodule '{}', which this quant result "
            'expects.'.format(layer_name)
        )
    return parent, attr_name, child


def _check_no_nested_units(layer_names):
    """Refuse units whose paths nest, e.g. ``blk`` and ``blk.fc``.

    Replacing the outer module would throw away the inner replacement (or make
    the result depend on ordering), so this is rejected up front.
    """
    ordered = sorted(layer_names)
    for outer in ordered:
        prefix = outer + '.'
        nested = [name for name in ordered if name.startswith(prefix)]
        if nested:
            raise ValueError(
                "quant result cannot be reloaded: layer '{}' contains layer(s) {}, "
                'so replacing it would discard them.'.format(outer, nested)
            )


def _resolve_quant_module_class(entry):
    """Resolve a unit's ``quant_module_type`` to a registered class.

    The name is only ever looked up in the algorithm registry -- nothing is
    imported from the artifact -- and the resolved class must be a quantize
    module, so an artifact cannot name an arbitrary class to instantiate.
    """
    # Imported lazily to keep this module free of an import-time dependency on
    # the algorithm registry.
    from amct_pytorch.algorithms import AlgorithmRegistry
    from amct_pytorch.quantize_op.base_quant_module import BaseQuantizeModule

    quant_cls = AlgorithmRegistry.get_quant_op_by_name(entry['quant_module_type'])
    if not issubclass(quant_cls, BaseQuantizeModule):
        raise ValueError(
            "quantize module '{}' of layer '{}' is not a BaseQuantizeModule "
            'subclass.'.format(entry['quant_module_type'], entry['layer_name'])
        )
    if not quant_cls.supports_quant_result_export():
        raise ValueError(
            "quantize module '{}' of layer '{}' does not support quant result "
            'reload in this installation.'.format(
                entry['quant_module_type'], entry['layer_name']
            )
        )
    return quant_cls


def _check_ori_module_type(entry, ori_module):
    """The layer in this model must still be the type that was calibrated."""
    actual = type(ori_module).__name__
    if actual != entry['ori_module_type']:
        raise ValueError(
            "layer '{}' is a {} in the model passed in but the quant result was "
            'produced from a {}.'.format(
                entry['layer_name'], actual, entry['ori_module_type']
            )
        )


def _check_quant_config(entry, quant_cls):
    """Cross-check the persisted config against the resolved quantize class.

    The config carries the algorithm name that selected the module, so a unit
    whose config names a different algorithm than the class it asks for is
    inconsistent and must not be rebuilt.
    """
    from amct_pytorch.algorithms import AlgorithmRegistry
    from amct_pytorch.common.config.utils import get_alg_name_from_config

    quant_config = entry['quant_config']
    _strict_mapping(
        quant_config, "quant_config of layer '{}'".format(entry['layer_name'])
    )
    algorithm = quant_config.get('algorithm')
    if algorithm is None:
        return
    names, _ = get_alg_name_from_config(algorithm)
    for name in names:
        registered = AlgorithmRegistry.algo.get(name)
        if registered is None:
            raise ValueError(
                "quant config of layer '{}' names algorithm '{}', which is not "
                'registered in this installation.'.format(entry['layer_name'], name)
            )
        if quant_cls in registered.values():
            return
    raise ValueError(
        "quant config of layer '{}' names algorithm(s) {}, but its "
        "quant_module_type is '{}', which none of them use.".format(
            entry['layer_name'], names, quant_cls.__name__
        )
    )


def _module_device(module):
    """Device of a module's own tensors, defaulting to CPU when it has none."""
    for tensor in list(module.parameters(recurse=True)) + list(module.buffers()):
        return tensor.device
    return torch.device('cpu')


def _build_staged_module(entry, ori_module, tensors, non_tensor_params):
    """Build one replacement module without touching the live model.

    The quantize module is constructed on an isolated ``deepcopy`` of the layer on
    purpose: some modules keep a reference to the original ``Parameter`` (GPTQ
    does), and loading a persisted weight into that would mutate the model before
    the transaction commits.
    """
    staged_source = copy.deepcopy(ori_module)
    quant_cls = _resolve_quant_module_class(entry)
    _check_quant_config(entry, quant_cls)
    replacement = quant_cls(staged_source, entry['layer_name'], entry['quant_config'])
    replacement.load_quant_params(tensors, non_tensor_params)
    replacement.to(_module_device(ori_module))
    return replacement


def reload_quant_params(model, quant_result_dir):
    """Rebuild a calibrated model from a published quant result, atomically.

    Takes the same original float model the artifact was produced from and puts it
    back into its post-calibration state. Everything is validated and every
    replacement module is built and loaded before the first ``setattr``, so a bad
    artifact leaves the model exactly as it was -- no partially quantized model.

    Args:
        model: the original float ``nn.Module`` that was calibrated.
        quant_result_dir: directory holding the published quant result.
    Returns:
        The same model instance, modified in place.
    """
    result_dir = os.path.abspath(os.path.expanduser(quant_result_dir))
    # Phase 1: validate the artifact and stage every replacement.
    manifest = load_quant_manifest(result_dir)
    validate_result_directory(result_dir, manifest)
    validate_source_model(model, manifest['source_model'])

    entries = list(manifest['units'])
    _check_no_nested_units([entry['layer_name'] for entry in entries])

    replacement_plan = []
    for entry in entries:
        layer_name = entry['layer_name']
        parent, attr_name, ori_module = resolve_layer_parent(model, layer_name)
        _check_ori_module_type(entry, ori_module)
        tensors, non_tensor_params, _ = read_quant_unit(
            result_dir, layer_name, manifest
        )
        replacement = _build_staged_module(
            entry, ori_module, tensors, non_tensor_params
        )
        replacement_plan.append((parent, attr_name, replacement, ori_module))

    # Phase 2: commit. Nothing here can fail on artifact content any more; an
    # unexpected setattr failure is rolled back so the model stays untouched.
    _commit_replacements(replacement_plan, result_dir, manifest)
    return model


def _commit_replacements(replacement_plan, result_dir, manifest):
    """Install the staged modules, rolling back if any ``setattr`` fails."""
    committed = []
    try:
        for parent, attr_name, replacement, ori_module in replacement_plan:
            setattr(parent, attr_name, replacement)
            committed.append((parent, attr_name, ori_module))
    except Exception:
        for parent, attr_name, ori_module in reversed(committed):
            setattr(parent, attr_name, ori_module)
        raise
    LOGGER.logd(
        "Reloaded {} quant unit(s) of artifact '{}' from '{}'.".format(
            len(replacement_plan), manifest['artifact_id'], result_dir
        ),
        RELOAD_LOG_TAG,
    )
