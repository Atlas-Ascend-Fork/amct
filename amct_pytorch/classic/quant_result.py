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

import hashlib
import json
import os
import re
import uuid

import copy

import torch
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import save_file

from amct_pytorch.common.utils.file_io import (
    atomic_write_via,
    file_sha256_hex,
    validate_writable_directory,
    write_text_atomically,
)
from amct_pytorch.common.utils.log import LOGGER
from amct_pytorch.common.utils.serialization import (
    dtype_name,
    strict_mapping,
    tensor_schema_list,
    to_jsonable,
)

MANIFEST_NAME = 'quant_manifest.json'
# Frozen manifest schema constants.  Keep every wire-format constant together;
# readers and writers must derive their behavior from this single block.
MANIFEST_FORMAT = 'amct-quant-result-manifest'
SCHEMA_VERSION = 1
UNIT_SUFFIX = '.safetensors'
LOG_TAG = 'QuantResultWriter'
RELOAD_LOG_TAG = 'reload_quant_params'
DIGEST_PREFIX = 'sha256:'
SAFETENSORS_DTYPES = {
    'BOOL': 'bool',
    'U8': 'uint8',
    'I8': 'int8',
    'I16': 'int16',
    'I32': 'int32',
    'I64': 'int64',
    'F16': 'float16',
    'BF16': 'bfloat16',
    'F32': 'float32',
    'F64': 'float64',
}

_REQUIRED_MANIFEST_FIELDS = (
    'format',
    'schema_version',
    'artifact_id',
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
_REQUIRED_SOURCE_MODEL_FIELDS = ('model_type', 'torch_dtype')
_REQUIRED_TENSOR_FIELDS = ('key', 'dtype', 'shape')
_DIGEST_PATTERN = re.compile(r'sha256:[0-9a-f]{64}')

_UNSAFE_NAME_CHARS = re.compile(r'[^0-9a-zA-Z_]')
# Fields derived from the file itself, therefore excluded from entry_digest.
_DERIVED_ENTRY_FIELDS = ('file_size', 'sha256', 'entry_digest')


def is_quant_result_supported(quant_cls):
    """Return the class-level artifact capability declaration.

    The declaration is only an eligibility signal.  ``write_unit`` and the
    reload validators still verify the actual tensor and metadata payload, so a
    class cannot claim a valid artifact merely by setting this flag.
    """
    if not isinstance(quant_cls, type):
        return False
    from amct_pytorch.classic.quantize_op.quant_result_mixin import (
        QuantResultSupportMixin,
    )

    if not issubclass(quant_cls, QuantResultSupportMixin):
        return False
    if not bool(getattr(quant_cls, 'supports_quant_result', False)):
        return False
    quant_param_keys = getattr(quant_cls, 'quant_param_keys', None)
    if not callable(quant_param_keys):
        return False
    try:
        return bool(quant_param_keys())
    except (TypeError, ValueError):
        return False


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
    return _prefixed(file_sha256_hex(file_path))


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
    existing directory. Files, missing paths and inaccessible directories are
    rejected. Other directory entries may coexist with the artifact; the writer
    checks its own manifest and unit paths separately and never overwrites them.
    """
    return validate_writable_directory(
        quant_result_dir,
        name='quant_result_dir',
        type_hint='str path or None',
        empty_hint='pass None to disable quant result persistence',
    )


def _source_model_type(model):
    """Artifact ``source_model.model_type``: HF model_type, else class name."""
    config = getattr(model, 'config', None)
    model_type = getattr(config, 'model_type', None)
    if isinstance(model_type, str) and model_type:
        return model_type
    return type(model).__name__


def _source_model_dtype(model):
    """Return the dtype that quantization treats as the source model dtype.

    HuggingFace models declare it on ``config.torch_dtype``.  Plain PyTorch
    modules do not, so use their first parameter or buffer as the equivalent
    declaration.  The artifact intentionally records this semantic input,
    rather than hashing every weight byte or unrelated configuration field.
    """
    config = getattr(model, 'config', None)
    configured = getattr(config, 'torch_dtype', None)
    if configured is None:
        to_dict = getattr(config, 'to_dict', None)
        if callable(to_dict):
            config_dict = to_dict()
            if isinstance(config_dict, dict):
                configured = config_dict.get('torch_dtype')
    if isinstance(configured, torch.dtype):
        return dtype_name(configured)
    if isinstance(configured, str) and configured:
        return dtype_name(configured)
    for tensor in model.parameters():
        return dtype_name(tensor.dtype)
    for tensor in model.buffers():
        return dtype_name(tensor.dtype)
    return 'unknown'


def compute_model_fingerprint(model):
    """Build the source-model identity before any quant module is inserted.

    Only model type and source dtype are compared during reload.  Quantization
    results are intentionally reusable across checkpoints with the same
    quantization identity; layer names, module types and tensor shapes remain
    independently validated while staging each replacement module.
    """
    return {
        'model_type': _source_model_type(model),
        'torch_dtype': _source_model_dtype(model),
    }


def unit_header(artifact_id, layer_name, entry_digest):
    """The minimal safetensors header: three ``str -> str`` cross-check fields.

    The manifest is the only place that declares the artifact format and schema
    version. A unit only binds itself to that manifest and its corresponding
    entry, so global schema metadata is not duplicated in every file.
    """
    return {
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
        model_fingerprint,
        artifact_id=None,
    ):
        self._result_dir = os.path.abspath(result_dir)
        self._expected_units = self._check_expected_units(expected_units)
        _validate_source_model_fields(model_fingerprint)
        self._model_fingerprint = dict(model_fingerprint)
        self._artifact_id = artifact_id or str(uuid.uuid4())
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
        _check_no_nested_units(units)
        return tuple(units)

    @classmethod
    def create(cls, quant_result_dir, expected_units, model_fingerprint):
        """Validate inputs and return a writer for a caller-created directory."""
        units = cls._check_expected_units(expected_units)
        result_dir = validate_quant_result_dir(quant_result_dir)
        cls._check_managed_paths_available(result_dir, units)
        return cls(result_dir, units, model_fingerprint)

    @staticmethod
    def _check_managed_paths_available(result_dir, expected_units):
        """Reject only paths this writer owns; leave unrelated entries alone."""
        managed_names = [MANIFEST_NAME]
        managed_names.extend(unit_file_name(unit) for unit in expected_units)
        existing = sorted(
            set(
                name
                for name in managed_names
                if os.path.lexists(os.path.join(result_dir, name))
            )
        )
        if existing:
            raise ValueError(
                "quant_result_dir '{}' already contains AMCT quant result file(s) "
                "{}; refusing to overwrite or resume them.".format(result_dir, existing)
            )

    @property
    def result_dir(self):
        return self._result_dir

    @property
    def manifest_path(self):
        return os.path.join(self._result_dir, MANIFEST_NAME)

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
        if os.path.lexists(file_path):
            raise FileExistsError(
                "quant result unit '{}' already exists; refusing to overwrite it.".format(
                    file_path
                )
            )
        header = unit_header(self._artifact_id, layer_name, entry['entry_digest'])
        # save_file streams straight to the temp path: a large tensor payload is
        # never serialized into an in-memory bytes object first.
        atomic_write_via(
            file_path,
            lambda tmp_path: save_file(tensors, tmp_path, metadata=header),
        )
        # Read metadata back before trusting the unit, then bind size and digest.
        # Any failure after the atomic write removes the orphan file and rolls
        # back this layer's file-name reservation, so the same writer can retry.
        try:
            self._verify_unit_file(file_path, entry, header)
            entry['file_size'] = os.path.getsize(file_path)
            entry['sha256'] = sha256_of_file(file_path)
        except Exception:
            try:
                if os.path.lexists(file_path):
                    os.remove(file_path)
            except OSError as cleanup_error:
                LOGGER.logw(
                    "Failed to remove incomplete quant result unit '{}': {}".format(
                        file_path, cleanup_error
                    ),
                    LOG_TAG,
                )
            finally:
                if self._file_names.get(file_name) == layer_name:
                    del self._file_names[file_name]
            raise

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
        """Check a unit header and tensor schema without loading tensor data."""
        with safe_open(file_path, framework='pt') as file_handle:
            actual_header = file_handle.metadata() or {}
            actual_schema = []
            for key in sorted(file_handle.keys()):
                tensor = file_handle.get_slice(key)
                safetensors_dtype = str(tensor.get_dtype())
                dtype = SAFETENSORS_DTYPES.get(safetensors_dtype)
                if dtype is None:
                    raise RuntimeError(
                        "quant result unit '{}' tensor '{}' has unsupported "
                        'safetensors dtype {}.'.format(
                            file_path, key, safetensors_dtype
                        )
                    )
                actual_schema.append(
                    {'key': key, 'dtype': dtype, 'shape': list(tensor.get_shape())}
                )
        if actual_header != header:
            raise RuntimeError(
                "quant result unit '{}' header mismatch after write: expected "
                "{}, found {}.".format(file_path, header, actual_header)
            )
        if actual_schema != entry['tensors']:
            raise RuntimeError(
                "quant result unit '{}' tensor schema mismatch after write: "
                "expected {}, found {}.".format(
                    file_path, entry['tensors'], actual_schema
                )
            )

    def is_complete(self):
        """Return whether every expected unit has been written."""
        return len(self._entries) == len(self._expected_units)

    def cancel(self):
        """Remove this writer's unpublished units and release reservations."""
        if self._published:
            raise RuntimeError(
                "quant result '{}' is already published; it is immutable.".format(
                    self._result_dir
                )
            )
        failures = []
        for entry in self._entries.values():
            try:
                file_path = resolve_unit_path(self._result_dir, entry['file'])
                if os.path.lexists(file_path):
                    os.remove(file_path)
            except OSError as error:
                failures.append(str(error))
        if failures:
            raise RuntimeError(
                "failed to remove all quant result units from '{}': {}.".format(
                    self._result_dir, '; '.join(failures)
                )
            )
        self._entries.clear()
        self._file_names.clear()

    def publish_if_complete(self):
        """Publish the manifest iff every expected unit has been written."""
        if self._published:
            return True
        if not self.is_complete():
            pending = self.pending_units
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
        if os.path.lexists(self.manifest_path):
            raise FileExistsError(
                "quant result manifest '{}' already exists; refusing to overwrite "
                "or resume it.".format(self.manifest_path)
            )
        self._verify_result_dir()
        manifest = {
            'format': MANIFEST_FORMAT,
            'schema_version': SCHEMA_VERSION,
            'artifact_id': self._artifact_id,
            'source_model': self._model_fingerprint,
            'units': [self._entries[unit] for unit in self._expected_units],
        }
        write_text_atomically(self.manifest_path, canonical_json(manifest) + '\n')
        self._published = True
        LOGGER.logi(
            "Published quant result manifest '{}' with {} unit(s).".format(
                self.manifest_path, len(manifest['units'])
            ),
            LOG_TAG,
        )

    def _verify_result_dir(self):
        """Check every unit managed by this writer is present and intact."""
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


def load_quant_manifest(quant_result_dir):
    """Load and validate ``quant_manifest.json`` of a published artifact.

    A directory without the manifest is not a valid quant result, so its absence
    is an error rather than an empty read. Parsed with ``json`` only -- never
    pickle -- and every exact schema field, duplicate-key rule, unit uniqueness
    rule and safe relative-path rule are enforced before anything is handed back.
    """
    result_dir = os.path.abspath(os.path.expanduser(quant_result_dir))
    manifest_path = os.path.join(result_dir, MANIFEST_NAME)
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(
            "'{}' is not a valid quant result: {} is missing, so the run that "
            "produced it did not complete.".format(quant_result_dir, MANIFEST_NAME)
        )
    with open(manifest_path, encoding='utf-8') as file_handle:
        manifest = json.load(file_handle, object_pairs_hook=_reject_duplicate_keys)
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
    _non_empty_str(source_model['torch_dtype'], 'source_model.torch_dtype')


def _reject_duplicate_keys(pairs):
    """Reject duplicate JSON object keys instead of silently keeping the last."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(
                "quant manifest contains duplicate JSON key {!r}.".format(key)
            )
        result[key] = value
    return result


def _check_required(mapping, fields, what):
    """Require the exact frozen field set for a protocol object."""
    if not isinstance(mapping, dict):
        raise ValueError("{} must be a JSON object.".format(what))
    missing = [field for field in fields if field not in mapping]
    if missing:
        raise ValueError("{} is missing required field(s) {}.".format(what, missing))
    unexpected = sorted(set(mapping) - set(fields))
    if unexpected:
        raise ValueError("{} contains unknown field(s) {}.".format(what, unexpected))


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
    _check_no_nested_units(seen_layers)


def validate_result_directory(quant_result_dir, manifest):
    """Check that the manifest and every unit it manages are present."""
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
    """Validate the source model's quantization identity before staging.

    Only model type and source dtype are compared here. The artifact is
    intentionally reusable across checkpoints that share this quantization
    identity. Layer paths, original module types, and tensor dtypes and shapes
    are validated while each replacement is staged, before any live module is
    modified.
    """
    # Shape first: a non-mapping must fail as a validation error, not as an
    # AttributeError from inside the strict conversion.
    _check_required(source_model, _REQUIRED_SOURCE_MODEL_FIELDS, 'source_model')
    strict_mapping(source_model, 'source_model')
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
                "quant result cannot contain nested units: layer '{}' contains "
                "layer(s) {}, "
                'so replacing it would discard them.'.format(outer, nested)
            )


def _resolve_quant_module_class(entry):
    """Resolve a unit through the registered algorithm and source module type."""
    # Imported lazily to keep this module free of an import-time dependency on
    # the algorithm registry.
    from amct_pytorch.algorithms import AlgorithmRegistry
    from amct_pytorch.common.config.utils import get_alg_name_from_config
    from amct_pytorch.quantize_op.base_quant_module import BaseQuantizeModule

    layer_name = entry['layer_name']
    quant_config = entry['quant_config']
    strict_mapping(quant_config, "quant_config of layer '{}'".format(layer_name))
    try:
        algorithms, _ = get_alg_name_from_config(quant_config.get('algorithm'))
    except ValueError as error:
        raise ValueError(
            "quant config of layer '{}' must name exactly one algorithm.".format(
                layer_name
            )
        ) from error
    if len(algorithms) != 1:
        raise ValueError(
            "quant config of layer '{}' must name exactly one algorithm, got {}.".format(
                layer_name, algorithms
            )
        )

    algorithm = algorithms[0]
    registered_sources = AlgorithmRegistry.algo.get(algorithm)
    if registered_sources is None:
        raise ValueError(
            "quant config of layer '{}' names algorithm '{}', which is not "
            'registered in this installation.'.format(layer_name, algorithm)
        )
    ori_module_type = entry['ori_module_type']
    quant_cls = registered_sources.get(ori_module_type)
    if quant_cls is None:
        raise ValueError(
            "algorithm '{}' has no registered quantize module for source type "
            "'{}', required by layer '{}'.".format(
                algorithm, ori_module_type, layer_name
            )
        )
    if not is_quant_result_supported(quant_cls):
        raise ValueError(
            "algorithm '{}' of layer '{}' does not support quant result reload "
            'because its registered quantize class does not declare the artifact '
            'contract.'.format(algorithm, layer_name)
        )
    if not isinstance(quant_cls, type):
        raise ValueError(
            "registered quantize module for algorithm '{}' and source type '{}' "
            'is not a class.'.format(algorithm, ori_module_type)
        )
    from amct_pytorch.classic.quantize_op.quant_result_mixin import (
        QuantResultSupportMixin,
    )

    if not issubclass(quant_cls, QuantResultSupportMixin):
        raise ValueError(
            "quantize module '{}' of layer '{}' is not a "
            'QuantResultSupportMixin subclass.'.format(quant_cls.__name__, layer_name)
        )
    if not issubclass(quant_cls, BaseQuantizeModule):
        raise ValueError(
            "quantize module '{}' of layer '{}' is not a BaseQuantizeModule "
            'subclass.'.format(quant_cls.__name__, layer_name)
        )
    if entry['quant_module_type'] != quant_cls.__name__:
        raise ValueError(
            "quant_module_type '{}' of layer '{}' does not match the registered "
            "class '{}' selected by algorithm '{}' and source type '{}'.".format(
                entry['quant_module_type'],
                layer_name,
                quant_cls.__name__,
                algorithm,
                ori_module_type,
            )
        )
    return quant_cls


def _check_ori_module_type(entry, ori_module):
    """The layer in this model must still be the type that was calibrated."""
    actual = type(ori_module).__name__
    if actual != entry['ori_module_type']:
        raise ValueError(
            "the model passed in is not the model expected by quant result layer "
            "'{}': it is a {}, but the artifact was produced from a {}.".format(
                entry['layer_name'], actual, entry['ori_module_type']
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
