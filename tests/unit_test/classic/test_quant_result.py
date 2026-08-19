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
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
"""Unit tests for the V2 quant result artifact (writer, digests, manifest).

The protocol these tests pin down: a quant result directory is valid only when
``quant_manifest.json`` is present, which happens only after every expected unit
has been written and re-verified. Interrupted runs, colliding file names,
missing or extra files and tampered content must all fail loudly rather than
yield a partially loadable artifact.
"""

import json
import os

import pytest
import torch
import torch.nn as nn

from amct_pytorch.classic.quant_result import (
    MANIFEST_FORMAT,
    MANIFEST_NAME,
    PARAM_FORMAT,
    SCHEMA_VERSION,
    QuantResultWriter,
    canonical_json,
    compute_model_fingerprint,
    create_quant_result_dir,
    load_quant_manifest,
    read_quant_unit,
    resolve_unit_path,
    to_jsonable,
    to_jsonable_lossy,
    unit_file_name,
    validate_quant_result_dir,
    validate_result_directory,
)

SOURCE_MODEL = {
    'model_type': 'tiny',
    'config_sha256': 'sha256:' + '0' * 64,
    'state_dict_sha256': 'sha256:' + '1' * 64,
}


def _writer(tmp_path, units=('fc1', 'fc2'), name='out'):
    """Writer over a fresh result dir under tmp_path."""
    return QuantResultWriter.create(str(tmp_path / name), units, SOURCE_MODEL)


def _write(writer, layer, tensors=None, **kwargs):
    """Write one unit with a small default tensor payload.

    quant_module_type/ori_module_type default to real names because the V2 schema
    requires them: a unit without them cannot be reloaded.
    """
    if tensors is None:
        tensors = {'scale_w': torch.tensor([0.5, 0.25])}
    kwargs.setdefault('quant_module_type', 'MinMaxQuant')
    kwargs.setdefault('ori_module_type', 'Linear')
    return writer.write_unit(layer, tensors, **kwargs)


class TestCanonicalJson:
    """canonical_json / to_jsonable must be stable and digest-friendly."""

    def test_canonical_json_is_key_order_independent(self):
        assert canonical_json({'b': 1, 'a': 2}) == canonical_json({'a': 2, 'b': 1})
        assert canonical_json({'a': 1}) == '{"a":1}'

    def test_canonical_json_rejects_nan(self):
        with pytest.raises(ValueError):
            canonical_json({'a': float('nan')})

    def test_to_jsonable_sorts_sets_deterministically(self):
        assert to_jsonable({'minmax', 'awq'}) == ['awq', 'minmax']
        assert to_jsonable(frozenset({3, 1, 2})) == [1, 2, 3]

    def test_to_jsonable_converts_dtype_without_the_torch_prefix(self):
        assert to_jsonable(torch.float16) == 'float16'
        assert to_jsonable({'w': torch.bfloat16}) == {'w': 'bfloat16'}

    def test_to_jsonable_keeps_tuples_as_lists(self):
        assert to_jsonable((1, 2, 3)) == [1, 2, 3]
        assert to_jsonable({'shape': (4096, 1)}) == {'shape': [4096, 1]}

    def test_to_jsonable_normalizes_non_str_keys(self):
        assert to_jsonable({1: 'a', 'b': 2}) == {'1': 'a', 'b': 2}

    def test_to_jsonable_rejects_colliding_keys(self):
        with pytest.raises(ValueError, match='collides'):
            to_jsonable({1: 'a', '1': 'b'})

    @pytest.mark.parametrize(
        'value',
        [
            nn.Linear(2, 2),
            torch.zeros(2),
            torch.device('cpu'),
            object(),
            lambda x: x,
        ],
    )
    def test_to_jsonable_rejects_unknown_objects(self, value):
        # A stringified object cannot reconstruct a module on reload, so the
        # strict converter must refuse rather than silently degrade it.
        with pytest.raises((TypeError, ValueError)):
            to_jsonable({'bad': value})

    def test_to_jsonable_rejects_non_str_keys_that_are_not_scalars(self):
        with pytest.raises(TypeError, match='config keys must be'):
            to_jsonable({(1, 2): 'a'})

    @pytest.mark.parametrize('value', [float('nan'), float('inf'), float('-inf')])
    def test_to_jsonable_rejects_non_finite_floats(self, value):
        with pytest.raises(ValueError, match='non-finite'):
            to_jsonable({'bad': value})

    def test_to_jsonable_error_names_the_offending_path(self):
        with pytest.raises(TypeError, match=r'quant_config\.a\.b'):
            to_jsonable({'a': {'b': object()}})

    def test_lossy_converter_stringifies_unknowns_for_fingerprints_only(self):
        payload = to_jsonable_lossy(
            {'obj': nn.Linear(2, 2), 'dev': torch.device('cpu'), 'inf': float('inf')}
        )
        # Only needs to be deterministic and digestible, never reversible.
        assert canonical_json(payload)
        assert to_jsonable_lossy(torch.zeros(2, 3))['shape'] == [2, 3]

    def test_lossy_converter_is_stable_across_instances(self):
        first = to_jsonable_lossy({'m': nn.Linear(2, 2)})
        second = to_jsonable_lossy({'m': nn.Linear(2, 2)})
        # Distinct objects at distinct addresses must digest identically.
        assert canonical_json(first) == canonical_json(second)


class TestUnitFileName:
    """File names must be traversal-proof and collision-detectable."""

    @pytest.mark.parametrize(
        'layer,expected',
        [
            ('linear1', 'linear1.safetensors'),
            ('block.0.fc1', 'block_0_fc1.safetensors'),
            ('a/../../etc/passwd', 'a_______etc_passwd.safetensors'),
            (
                'model.layers.0.self_attn.q_proj',
                'model_layers_0_self_attn_q_proj.safetensors',
            ),
        ],
    )
    def test_sanitizes_to_a_flat_safe_name(self, layer, expected):
        name = unit_file_name(layer)
        assert name == expected
        assert os.sep not in name and '..' not in name

    @pytest.mark.parametrize('layer', ['', None, 123, '...', '/'])
    def test_rejects_unusable_layer_names(self, layer):
        with pytest.raises(ValueError):
            unit_file_name(layer)


class TestResultDirValidation:
    """Only None disables persistence; every other input must be usable."""

    def test_accepts_new_path_without_creating_it(self, tmp_path):
        target = tmp_path / 'fresh'
        assert validate_quant_result_dir(str(target)) == str(target)
        assert not target.exists()

    def test_create_makes_dir_with_restricted_mode(self, tmp_path):
        target = create_quant_result_dir(str(tmp_path / 'made'))
        assert os.path.isdir(target)
        assert os.stat(target).st_mode & 0o777 == 0o750

    @pytest.mark.parametrize('bad', ['', '   '])
    def test_rejects_empty_path(self, bad):
        with pytest.raises(ValueError, match='non-empty path'):
            validate_quant_result_dir(bad)

    @pytest.mark.parametrize('bad', [None, 1, 1.5, b'x', object()])
    def test_rejects_non_str_path(self, bad):
        with pytest.raises(TypeError):
            validate_quant_result_dir(bad)

    def test_rejects_existing_dir(self, tmp_path):
        existing = tmp_path / 'already'
        existing.mkdir()
        with pytest.raises(ValueError, match='already exists'):
            validate_quant_result_dir(str(existing))

    def test_rejects_existing_file(self, tmp_path):
        existing = tmp_path / 'a_file'
        existing.write_text('x', encoding='utf-8')
        with pytest.raises(ValueError, match='already exists'):
            validate_quant_result_dir(str(existing))

    def test_rejects_dangling_symlink(self, tmp_path):
        link = tmp_path / 'link'
        link.symlink_to(tmp_path / 'nowhere')
        with pytest.raises(ValueError, match='already exists'):
            validate_quant_result_dir(str(link))

    def test_rejects_missing_parent(self, tmp_path):
        with pytest.raises(ValueError, match='does not exist'):
            validate_quant_result_dir(str(tmp_path / 'no' / 'such' / 'parent'))

    def test_rejects_unwritable_parent(self, tmp_path):
        parent = tmp_path / 'ro'
        parent.mkdir()
        os.chmod(parent, 0o500)
        try:
            if os.access(parent, os.W_OK):
                pytest.skip('running as a user that bypasses write permissions')
            with pytest.raises(ValueError, match='not writable'):
                validate_quant_result_dir(str(parent / 'out'))
        finally:
            os.chmod(parent, 0o700)


class TestWriterUnitWrites:
    """write_unit registers a verified unit but never publishes on its own."""

    def test_write_unit_creates_file_and_defers_manifest(self, tmp_path):
        writer = _writer(tmp_path)
        entry = _write(writer, 'fc1')
        assert entry['file'] == 'fc1.safetensors'
        assert os.path.isfile(os.path.join(writer.result_dir, 'fc1.safetensors'))
        assert not os.path.exists(writer.manifest_path)
        assert writer.pending_units == ('fc2',)
        assert not writer.published

    def test_entry_carries_size_hash_digest_and_schema(self, tmp_path):
        writer = _writer(tmp_path)
        entry = _write(writer, 'fc1', {'scale_w': torch.zeros(3, dtype=torch.float32)})
        path = os.path.join(writer.result_dir, entry['file'])
        assert entry['file_size'] == os.path.getsize(path)
        # V2 spells digests with an explicit algorithm prefix.
        assert entry['sha256'].startswith('sha256:')
        assert entry['entry_digest'].startswith('sha256:')
        assert len(entry['sha256']) == len('sha256:') + 64
        # V2 tensor schema is an ordered list, dtype without the torch. prefix.
        assert entry['tensors'] == [
            {'key': 'scale_w', 'dtype': 'float32', 'shape': [3]}
        ]

    def test_written_tensors_are_detached_cpu_contiguous(self, tmp_path):
        writer = _writer(tmp_path, units=('fc1',))
        source = torch.zeros(4, 6, requires_grad=True)
        non_contiguous = source.transpose(0, 1)
        assert not non_contiguous.is_contiguous()
        _write(writer, 'fc1', {'scale_w': non_contiguous})
        tensors, _, _ = _publish_and_read(writer, 'fc1')
        assert tensors['scale_w'].is_contiguous()
        assert not tensors['scale_w'].requires_grad
        assert tensors['scale_w'].device.type == 'cpu'

    def test_records_module_type_and_algorithm(self, tmp_path):
        writer = _writer(tmp_path, units=('fc1',))
        entry = _write(
            writer,
            'fc1',
            quant_module_type='MinMaxQuant',
            ori_module_type='Linear',
            quant_config={'batch_num': 1, 'algorithm': {'minmax': {}}},
            non_tensor_params={'group_size': 64},
        )
        assert entry['quant_module_type'] == 'MinMaxQuant'
        assert entry['ori_module_type'] == 'Linear'
        assert entry['quant_config'] == {'batch_num': 1, 'algorithm': {'minmax': {}}}
        assert entry['non_tensor_params'] == {'group_size': 64}
        # The algorithm lives inside quant_config, never as a sibling field.
        assert 'algorithm' not in entry

    def test_accepts_unit_without_tensors(self, tmp_path):
        writer = _writer(tmp_path, units=('fc1',))
        entry = _write(writer, 'fc1', {})
        assert entry['tensors'] == []
        assert writer.publish_if_complete()

    def test_rejects_unexpected_layer(self, tmp_path):
        writer = _writer(tmp_path)
        with pytest.raises(ValueError, match='not part of the expected'):
            _write(writer, 'fc3')

    def test_rejects_duplicate_layer(self, tmp_path):
        writer = _writer(tmp_path)
        _write(writer, 'fc1')
        with pytest.raises(ValueError, match='already written'):
            _write(writer, 'fc1')

    def test_rejects_file_name_collision_between_layers(self, tmp_path):
        # 'a.b' and 'a-b' both sanitize to 'a_b.safetensors'.
        writer = _writer(tmp_path, units=('a.b', 'a-b'))
        _write(writer, 'a.b')
        with pytest.raises(ValueError, match='both map to quant result file'):
            _write(writer, 'a-b')
        assert sorted(os.listdir(writer.result_dir)) == ['a_b.safetensors']

    def test_rejects_non_tensor_and_bad_keys(self, tmp_path):
        writer = _writer(tmp_path, units=('fc1',))
        with pytest.raises(TypeError, match='must be a torch.Tensor'):
            _write(writer, 'fc1', {'scale_w': [1.0, 2.0]})
        with pytest.raises(ValueError, match='non-empty'):
            _write(writer, 'fc1', {'': torch.zeros(1)})
        with pytest.raises(TypeError, match='must be a dict'):
            _write(writer, 'fc1', [torch.zeros(1)])

    def test_converts_the_config_types_v2_admits(self, tmp_path):
        writer = _writer(tmp_path, units=('fc1',))
        entry = _write(
            writer,
            'fc1',
            quant_config={
                'algorithm': {'minmax'},
                'dtype': torch.bfloat16,
                'shape': (4096, 1),
                'symmetric': True,
                'group_size': None,
            },
        )
        assert entry['quant_config'] == {
            'algorithm': ['minmax'],
            'dtype': 'bfloat16',
            'shape': [4096, 1],
            'symmetric': True,
            'group_size': None,
        }

    def test_rejects_a_quant_config_reload_could_not_use(self, tmp_path):
        # quant_config is the sole input for reconstructing the module on
        # reload, so an unknown object must fail the write rather than be
        # stringified into something unusable.
        writer = _writer(tmp_path, units=('fc1',))
        with pytest.raises(TypeError, match='cannot be serialized'):
            _write(writer, 'fc1', quant_config={'mod': nn.Linear(2, 2)})
        assert os.listdir(writer.result_dir) == []

    def test_rejects_unserializable_non_tensor_params(self, tmp_path):
        writer = _writer(tmp_path, units=('fc1',))
        with pytest.raises(TypeError, match='cannot be serialized'):
            _write(writer, 'fc1', non_tensor_params={'obj': object()})
        assert os.listdir(writer.result_dir) == []

    def test_rejects_empty_expected_units(self, tmp_path):
        with pytest.raises(ValueError, match='no quantizable layer'):
            QuantResultWriter.create(str(tmp_path / 'none'), (), SOURCE_MODEL)
        assert not (tmp_path / 'none').exists()

    @pytest.mark.parametrize('units', [('fc1', 'fc1'), ('fc1', ''), ('fc1', None)])
    def test_rejects_invalid_expected_units(self, tmp_path, units):
        with pytest.raises(ValueError):
            QuantResultWriter.create(str(tmp_path / 'bad'), units, SOURCE_MODEL)

    def test_no_temp_file_survives_a_failed_write(self, tmp_path, monkeypatch):
        writer = _writer(tmp_path, units=('fc1',))
        import amct_pytorch.classic.quant_result as quant_result

        def boom(*_args, **_kwargs):
            raise OSError('disk full')

        monkeypatch.setattr(quant_result, 'save_file', boom)
        with pytest.raises(OSError):
            _write(writer, 'fc1')
        assert os.listdir(writer.result_dir) == []
        assert not os.path.exists(writer.manifest_path)


class TestStreamingWrite:
    """Tensors stream to disk; the payload is never buffered as bytes."""

    def test_unit_is_written_straight_to_a_temp_path(self, tmp_path, monkeypatch):
        import amct_pytorch.classic.quant_result as quant_result

        seen = {}
        real_save_file = quant_result.save_file

        def spy(tensors, path, metadata=None):
            seen['path'] = path
            return real_save_file(tensors, path, metadata=metadata)

        monkeypatch.setattr(quant_result, 'save_file', spy)
        writer = _writer(tmp_path, units=('fc1',))
        entry = _write(writer, 'fc1')

        # save_file(tensors, path) streams to disk. The earlier save(...) -> bytes
        # form buffered the whole payload, which for a GPTQ weight is hundreds of
        # MiB of avoidable peak memory.
        assert seen['path'] == quant_result.temp_name_for(
            os.path.join(writer.result_dir, entry['file'])
        )
        assert os.path.basename(seen['path']).startswith('.')

    def test_no_temp_file_remains_after_a_successful_write(self, tmp_path):
        writer = _writer(tmp_path, units=('fc1',))
        _write(writer, 'fc1')
        assert os.listdir(writer.result_dir) == ['fc1.safetensors']


def _publish_and_read(writer, layer):
    """Publish then read one unit back through the verifying reader."""
    assert writer.publish_if_complete()
    return read_quant_unit(writer.result_dir, layer)


class TestManifestPublish:
    """The manifest is written last, and only for a complete, intact set."""

    def test_publish_requires_every_expected_unit(self, tmp_path):
        writer = _writer(tmp_path)
        _write(writer, 'fc1')
        assert writer.publish_if_complete() is False
        assert not os.path.exists(writer.manifest_path)
        _write(writer, 'fc2')
        assert writer.publish_if_complete() is True
        assert os.path.isfile(writer.manifest_path)

    def test_publish_is_idempotent(self, tmp_path):
        writer = _writer(tmp_path, units=('fc1',))
        _write(writer, 'fc1')
        assert writer.publish_if_complete()
        first = load_quant_manifest(writer.result_dir)
        assert writer.publish_if_complete()
        assert load_quant_manifest(writer.result_dir) == first

    def test_writes_are_rejected_after_publish(self, tmp_path):
        writer = _writer(tmp_path, units=('fc1', 'fc2'))
        _write(writer, 'fc1')
        _write(writer, 'fc2')
        writer.publish_if_complete()
        with pytest.raises(RuntimeError, match='already published'):
            writer.write_unit('fc1', {}, non_tensor_params={})

    def test_manifest_matches_the_frozen_v2_schema(self, tmp_path):
        writer = _writer(tmp_path)
        _write(writer, 'fc1', quant_module_type='MinMaxQuant', ori_module_type='Linear')
        _write(writer, 'fc2', quant_module_type='MinMaxQuant', ori_module_type='Linear')
        writer.publish_if_complete()
        manifest = load_quant_manifest(writer.result_dir)

        assert set(manifest) == {
            'format',
            'schema_version',
            'artifact_id',
            'producer',
            'source_model',
            'units',
        }
        assert manifest['format'] == MANIFEST_FORMAT == 'amct-quant-result-manifest'
        assert manifest['schema_version'] == SCHEMA_VERSION == 1
        assert manifest['artifact_id'] == writer.artifact_id
        assert set(manifest['producer']) == {'library', 'library_version'}
        assert manifest['producer']['library'] == 'amct_pytorch'
        assert manifest['producer']['library_version']
        assert manifest['source_model'] == SOURCE_MODEL
        assert [unit['layer_name'] for unit in manifest['units']] == ['fc1', 'fc2']

    def test_manifest_unit_has_exactly_the_v2_fields(self, tmp_path):
        writer = _writer(tmp_path, units=('fc1',))
        _write(writer, 'fc1', quant_module_type='MinMaxQuant', ori_module_type='Linear')
        writer.publish_if_complete()
        unit = load_quant_manifest(writer.result_dir)['units'][0]
        assert set(unit) == {
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
        }

    def test_artifact_id_is_a_uuid(self, tmp_path):
        import uuid as uuid_module

        writer = _writer(tmp_path, units=('fc1',))
        # Must parse as a UUID so foreign files can be rejected by identity.
        assert str(uuid_module.UUID(writer.artifact_id)) == writer.artifact_id

    def test_manifest_is_canonical_json(self, tmp_path):
        writer = _writer(tmp_path, units=('fc1',))
        _write(writer, 'fc1')
        writer.publish_if_complete()
        raw = open(writer.manifest_path, encoding='utf-8').read()
        assert raw.endswith('\n')
        assert raw.strip() == canonical_json(json.loads(raw))

    def test_publish_fails_on_a_hidden_extra_safetensors(self, tmp_path):
        # A dot-prefixed file must not be waved through: V2 requires the
        # directory to contain no .safetensors outside the manifest.
        writer = _writer(tmp_path, units=('fc1',))
        _write(writer, 'fc1')
        (tmp_path / 'out' / '.unlisted.safetensors').write_bytes(b'junk')
        with pytest.raises(RuntimeError, match='not listed in the manifest'):
            writer.publish_if_complete()
        assert not os.path.exists(writer.manifest_path)

    def test_publish_fails_on_any_hidden_stray_file(self, tmp_path):
        writer = _writer(tmp_path, units=('fc1',))
        _write(writer, 'fc1')
        (tmp_path / 'out' / '.hidden_note').write_text('x', encoding='utf-8')
        with pytest.raises(RuntimeError, match='not listed in the manifest'):
            writer.publish_if_complete()

    def test_publish_fails_when_a_unit_file_disappeared(self, tmp_path):
        writer = _writer(tmp_path, units=('fc1',))
        _write(writer, 'fc1')
        os.remove(os.path.join(writer.result_dir, 'fc1.safetensors'))
        with pytest.raises(RuntimeError, match='missing unit file'):
            writer.publish_if_complete()
        assert not os.path.exists(writer.manifest_path)

    def test_publish_fails_on_unexpected_extra_file(self, tmp_path):
        writer = _writer(tmp_path, units=('fc1',))
        _write(writer, 'fc1')
        (tmp_path / 'out' / 'stray.safetensors').write_bytes(b'junk')
        with pytest.raises(RuntimeError, match='not listed in the manifest'):
            writer.publish_if_complete()
        assert not os.path.exists(writer.manifest_path)

    def test_publish_fails_when_a_unit_file_changed(self, tmp_path):
        writer = _writer(tmp_path, units=('fc1',))
        entry = _write(writer, 'fc1')
        path = os.path.join(writer.result_dir, entry['file'])
        with open(path, 'r+b') as handle:
            handle.seek(0, os.SEEK_END)
            handle.write(b'\x00')
        with pytest.raises(RuntimeError, match='changed size'):
            writer.publish_if_complete()
        assert not os.path.exists(writer.manifest_path)

    def test_publish_fails_when_content_is_tampered_in_place(self, tmp_path):
        writer = _writer(tmp_path, units=('fc1',))
        entry = _write(writer, 'fc1')
        path = os.path.join(writer.result_dir, entry['file'])
        size = os.path.getsize(path)
        with open(path, 'r+b') as handle:
            handle.seek(size - 1)
            flipped = bytes([handle.read(1)[0] ^ 0xFF])
            handle.seek(size - 1)
            handle.write(flipped)
        assert os.path.getsize(path) == size
        with pytest.raises(RuntimeError, match='SHA-256'):
            writer.publish_if_complete()
        assert not os.path.exists(writer.manifest_path)

    def test_publish_fails_when_entry_digest_no_longer_matches(self, tmp_path):
        writer = _writer(tmp_path, units=('fc1',))
        entry = _write(writer, 'fc1')
        entry['quant_module_type'] = 'Tampered'
        with pytest.raises(RuntimeError, match='entry_digest'):
            writer.publish_if_complete()
        assert not os.path.exists(writer.manifest_path)


class TestReadBack:
    """The reader cross-checks header, digests and schema before returning."""

    def test_round_trip_preserves_tensors_and_metadata(self, tmp_path):
        writer = _writer(tmp_path, units=('block.0.fc1',))
        scale = torch.tensor([[0.5], [0.25]], dtype=torch.float32)
        offset = torch.tensor([3], dtype=torch.int32)
        _write(
            writer,
            'block.0.fc1',
            {'scale_w': scale, 'offset_w': offset},
            non_tensor_params={'group_size': 32},
        )
        tensors, non_tensor_params, entry = _publish_and_read(writer, 'block.0.fc1')
        assert torch.equal(tensors['scale_w'], scale)
        assert torch.equal(tensors['offset_w'], offset)
        assert tensors['offset_w'].dtype == torch.int32
        assert non_tensor_params == {'group_size': 32}
        assert entry['file'] == 'block_0_fc1.safetensors'

    def test_round_trip_preserves_bfloat16(self, tmp_path):
        writer = _writer(tmp_path, units=('fc1',))
        scale = torch.randn(4, dtype=torch.bfloat16)
        _write(writer, 'fc1', {'scale_w': scale})
        tensors, _, _ = _publish_and_read(writer, 'fc1')
        assert tensors['scale_w'].dtype == torch.bfloat16
        assert torch.equal(tensors['scale_w'], scale)

    def test_unit_header_cross_checks_the_manifest(self, tmp_path):
        from safetensors import safe_open

        writer = _writer(tmp_path, units=('fc1',))
        entry = _write(
            writer, 'fc1', quant_module_type='MinMaxQuant', ori_module_type='Linear'
        )
        writer.publish_if_complete()
        path = os.path.join(writer.result_dir, entry['file'])
        with safe_open(path, framework='pt') as handle:
            header = handle.metadata()
        # Exactly the five frozen V2 header fields, all str -> str.
        assert header == {
            'format': PARAM_FORMAT,
            'schema_version': str(SCHEMA_VERSION),
            'artifact_id': writer.artifact_id,
            'layer_name': 'fc1',
            'entry_digest': entry['entry_digest'],
        }
        assert all(isinstance(value, str) for value in header.values())
        # The header must not carry a second copy of the config, or it could
        # drift from the manifest.
        for forbidden in (
            'quant_module_type',
            'ori_module_type',
            'quant_config',
            'non_tensor_params',
            'tensors',
            'file_size',
            'sha256',
        ):
            assert forbidden not in header

    def test_missing_manifest_is_not_a_valid_result(self, tmp_path):
        writer = _writer(tmp_path, units=('fc1', 'fc2'))
        _write(writer, 'fc1')
        with pytest.raises(FileNotFoundError, match='not a valid quant result'):
            load_quant_manifest(writer.result_dir)
        with pytest.raises(FileNotFoundError):
            read_quant_unit(writer.result_dir, 'fc1')

    def test_reader_rejects_unknown_layer(self, tmp_path):
        writer = _writer(tmp_path, units=('fc1',))
        _write(writer, 'fc1')
        writer.publish_if_complete()
        with pytest.raises(KeyError):
            read_quant_unit(writer.result_dir, 'nope')

    def _repoint(self, writer, mutate):
        """Rewrite a published manifest through ``mutate`` in place."""
        manifest = json.loads(open(writer.manifest_path, encoding='utf-8').read())
        mutate(manifest)
        with open(writer.manifest_path, 'w', encoding='utf-8') as handle:
            handle.write(canonical_json(manifest))
        return manifest

    def test_reader_rejects_a_tampered_unit_entry(self, tmp_path):
        writer = _writer(tmp_path, units=('fc1',))
        _write(writer, 'fc1')
        writer.publish_if_complete()
        self._repoint(
            writer,
            lambda m: m['units'][0].update(quant_module_type='Tampered'),
        )
        with pytest.raises(ValueError, match='entry_digest'):
            read_quant_unit(writer.result_dir, 'fc1')

    def test_reader_rejects_unsupported_format_and_version(self, tmp_path):
        writer = _writer(tmp_path, units=('fc1',))
        _write(writer, 'fc1')
        writer.publish_if_complete()
        self._repoint(writer, lambda m: m.update(format='something-else'))
        with pytest.raises(ValueError, match='format'):
            load_quant_manifest(writer.result_dir)
        self._repoint(
            writer, lambda m: m.update(format=MANIFEST_FORMAT, schema_version=99)
        )
        with pytest.raises(ValueError, match='schema_version'):
            load_quant_manifest(writer.result_dir)

    @pytest.mark.parametrize(
        'field',
        [
            'format',
            'schema_version',
            'artifact_id',
            'producer',
            'source_model',
            'units',
        ],
    )
    def test_reader_rejects_a_manifest_missing_a_required_field(self, tmp_path, field):
        writer = _writer(tmp_path, units=('fc1',), name='out_' + field)
        _write(writer, 'fc1')
        writer.publish_if_complete()
        self._repoint(writer, lambda m: m.pop(field))
        with pytest.raises(ValueError, match='missing required field'):
            load_quant_manifest(writer.result_dir)

    @pytest.mark.parametrize(
        'field',
        [
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
        ],
    )
    def test_reader_rejects_a_unit_missing_a_required_field(self, tmp_path, field):
        writer = _writer(tmp_path, units=('fc1',), name='unit_' + field)
        _write(writer, 'fc1')
        writer.publish_if_complete()
        self._repoint(writer, lambda m: m['units'][0].pop(field))
        with pytest.raises(ValueError, match='missing required field'):
            load_quant_manifest(writer.result_dir)

    def test_reader_rejects_missing_source_model_fields(self, tmp_path):
        writer = _writer(tmp_path, units=('fc1',))
        _write(writer, 'fc1')
        writer.publish_if_complete()
        self._repoint(writer, lambda m: m['source_model'].pop('state_dict_sha256'))
        with pytest.raises(ValueError, match='missing required field'):
            load_quant_manifest(writer.result_dir)

    def test_reader_rejects_duplicate_layer_in_manifest(self, tmp_path):
        writer = _writer(tmp_path, units=('fc1',))
        _write(writer, 'fc1')
        writer.publish_if_complete()
        self._repoint(writer, lambda m: m['units'].append(dict(m['units'][0])))
        with pytest.raises(ValueError, match='more than once'):
            load_quant_manifest(writer.result_dir)

    @pytest.mark.parametrize(
        'bad_file',
        ['/etc/passwd', '../escape.safetensors', 'nested/unit.safetensors'],
    )
    def test_reader_rejects_unsafe_unit_paths(self, tmp_path, bad_file):
        writer = _writer(tmp_path, units=('fc1',), name='path_out')
        _write(writer, 'fc1')
        writer.publish_if_complete()
        self._repoint(writer, lambda m: m['units'][0].update(file=bad_file))
        with pytest.raises(ValueError):
            load_quant_manifest(writer.result_dir)

    def test_validate_result_directory_rejects_a_hidden_extra_file(self, tmp_path):
        writer = _writer(tmp_path, units=('fc1',))
        _write(writer, 'fc1')
        writer.publish_if_complete()
        manifest = load_quant_manifest(writer.result_dir)
        validate_result_directory(writer.result_dir, manifest)
        (tmp_path / 'out' / '.sneaky.safetensors').write_bytes(b'junk')
        with pytest.raises(ValueError, match='not listed in its manifest'):
            validate_result_directory(writer.result_dir, manifest)

    def test_resolve_unit_path_rejects_traversal(self, tmp_path):
        for bad in ('/abs.safetensors', '../up.safetensors', 'a/b.safetensors'):
            with pytest.raises(ValueError):
                resolve_unit_path(str(tmp_path), bad)

    def test_reader_rejects_truncated_unit(self, tmp_path):
        writer = _writer(tmp_path, units=('fc1',))
        entry = _write(writer, 'fc1')
        writer.publish_if_complete()
        path = os.path.join(writer.result_dir, entry['file'])
        with open(path, 'r+b') as handle:
            handle.truncate(os.path.getsize(path) - 2)
        with pytest.raises(ValueError, match='wrong size'):
            read_quant_unit(writer.result_dir, 'fc1')


class _TinyModel(nn.Module):
    def __init__(self, hidden=8):
        super().__init__()
        self.fc1 = nn.Linear(8, hidden)
        self.fc2 = nn.Linear(hidden, 4)

    def forward(self, inputs):
        return self.fc2(self.fc1(inputs))


class _FakeHfConfig:
    model_type = 'fake_hf'

    def __init__(self, hidden_size):
        self.hidden_size = hidden_size

    def to_dict(self):
        return {
            'model_type': self.model_type,
            'hidden_size': self.hidden_size,
            'torch_dtype': torch.bfloat16,
        }


class TestModelFingerprint:
    """source_model must be stable, and sensitive to weights and structure."""

    def test_returns_exactly_the_v2_source_model_fields(self):
        fingerprint = compute_model_fingerprint(_TinyModel())
        assert set(fingerprint) == {
            'model_type',
            'config_sha256',
            'state_dict_sha256',
        }
        assert fingerprint['model_type'] == '_TinyModel'
        for field in ('config_sha256', 'state_dict_sha256'):
            assert fingerprint[field].startswith('sha256:')
            assert len(fingerprint[field]) == len('sha256:') + 64

    def test_same_weights_give_the_same_fingerprint(self):
        model = _TinyModel()
        clone = _TinyModel()
        clone.load_state_dict(model.state_dict())
        assert compute_model_fingerprint(model) == compute_model_fingerprint(clone)

    def test_changed_weight_changes_the_fingerprint(self):
        model = _TinyModel()
        before = compute_model_fingerprint(model)
        with torch.no_grad():
            model.fc1.weight[0, 0] += 1.0
        after = compute_model_fingerprint(model)
        assert after['state_dict_sha256'] != before['state_dict_sha256']
        # Structure is unchanged, so the config digest must stay put.
        assert after['config_sha256'] == before['config_sha256']

    def test_changed_structure_changes_both_digests(self):
        small = compute_model_fingerprint(_TinyModel(hidden=8))
        large = compute_model_fingerprint(_TinyModel(hidden=16))
        assert small['state_dict_sha256'] != large['state_dict_sha256']
        assert small['config_sha256'] != large['config_sha256']

    def test_dtype_change_changes_the_state_dict_digest(self):
        model = _TinyModel()
        before = compute_model_fingerprint(model)
        after = compute_model_fingerprint(model.to(torch.bfloat16))
        assert after['state_dict_sha256'] != before['state_dict_sha256']

    def test_hf_config_supplies_model_type_and_config_digest(self):
        model = _TinyModel()
        model.config = _FakeHfConfig(8)
        fingerprint = compute_model_fingerprint(model)
        assert fingerprint['model_type'] == 'fake_hf'
        model.config = _FakeHfConfig(16)
        assert (
            compute_model_fingerprint(model)['config_sha256']
            != fingerprint['config_sha256']
        )

    def test_hf_config_with_a_torch_dtype_does_not_break_fingerprinting(self):
        # A real HF config carries torch dtypes and other non-JSON objects; the
        # fingerprint path tolerates them, unlike the strict quant_config path.
        model = _TinyModel()
        model.config = _FakeHfConfig(8)
        assert canonical_json(compute_model_fingerprint(model))

    def test_fingerprint_is_json_serializable(self):
        assert canonical_json(compute_model_fingerprint(_TinyModel()))


def test_protocol_constants_are_the_frozen_v2_values():
    assert MANIFEST_NAME == 'quant_manifest.json'
    assert MANIFEST_FORMAT == 'amct-quant-result-manifest'
    assert PARAM_FORMAT == 'amct-quant-param'
    assert SCHEMA_VERSION == 1


class TestManifestSchemaIsEnforced:
    """The reader must reject every malformed manifest the V2 spec forbids.

    These are the fields reload trusts before it touches the model, so each one
    gets an explicit negative case rather than being covered only in passing.
    """

    @staticmethod
    def _published(tmp_path, name='schema'):
        writer = _writer(tmp_path, units=('fc1',), name=name)
        _write(writer, 'fc1')
        writer.publish_if_complete()
        return writer.result_dir

    def _forge(self, tmp_path, mutate, name='schema'):
        result_dir = self._published(tmp_path, name=name)
        path = os.path.join(result_dir, MANIFEST_NAME)
        with open(path, encoding='utf-8') as handle:
            manifest = json.load(handle)
        mutate(manifest)
        with open(path, 'w', encoding='utf-8') as handle:
            json.dump(manifest, handle)
        return result_dir

    @pytest.mark.parametrize('version', [True, False, 1.0, '1', 2, 0, None, [1]])
    def test_schema_version_must_be_the_exact_int(self, tmp_path, version):
        # bool is an int subclass and 1.0 == 1, so equality alone is not enough.
        result_dir = self._forge(
            tmp_path, lambda man: man.update({'schema_version': version})
        )
        with pytest.raises(ValueError, match='schema_version'):
            load_quant_manifest(result_dir)

    @pytest.mark.parametrize(
        'producer',
        [
            [],
            'amct_pytorch',
            None,
            {},
            {'library': 'amct_pytorch'},
            {'library': '', 'library_version': '1'},
            {'library': 'amct_pytorch', 'library_version': ''},
            {'library': 'other_lib', 'library_version': '1'},
            {'library': 1, 'library_version': '1'},
        ],
    )
    def test_producer_must_identify_this_library(self, tmp_path, producer):
        result_dir = self._forge(
            tmp_path, lambda man: man.update({'producer': producer})
        )
        with pytest.raises(ValueError, match='producer|library'):
            load_quant_manifest(result_dir)

    @pytest.mark.parametrize(
        'artifact_id', ['', 'not-a-uuid', None, 123, '1234', ['uuid']]
    )
    def test_artifact_id_must_be_a_uuid(self, tmp_path, artifact_id):
        result_dir = self._forge(
            tmp_path, lambda man: man.update({'artifact_id': artifact_id})
        )
        with pytest.raises(ValueError, match='artifact_id'):
            load_quant_manifest(result_dir)

    @pytest.mark.parametrize('field', ['config_sha256', 'state_dict_sha256'])
    @pytest.mark.parametrize(
        'digest',
        [
            '',
            'deadbeef',
            'sha256:' + 'x' * 64,
            'sha256:' + 'A' * 64,
            'sha1:' + 'a' * 64,
            None,
        ],
    )
    def test_source_model_digests_must_be_prefixed_sha256(
        self, tmp_path, field, digest
    ):
        result_dir = self._forge(
            tmp_path, lambda man: man['source_model'].update({field: digest})
        )
        with pytest.raises(ValueError, match=field):
            load_quant_manifest(result_dir)

    def test_source_model_type_must_be_a_non_empty_string(self, tmp_path):
        result_dir = self._forge(
            tmp_path, lambda man: man['source_model'].update({'model_type': ''})
        )
        with pytest.raises(ValueError, match='model_type'):
            load_quant_manifest(result_dir)

    @pytest.mark.parametrize(
        'field', ['layer_name', 'file', 'quant_module_type', 'ori_module_type']
    )
    @pytest.mark.parametrize('value', ['', None, 1, ['x']])
    def test_unit_name_fields_must_be_non_empty_strings(self, tmp_path, field, value):
        result_dir = self._forge(
            tmp_path, lambda man: man['units'][0].update({field: value})
        )
        with pytest.raises(ValueError):
            load_quant_manifest(result_dir)

    @pytest.mark.parametrize('field', ['quant_config', 'non_tensor_params'])
    @pytest.mark.parametrize('value', [None, [], 'x', 1, True])
    def test_unit_config_fields_must_be_objects(self, tmp_path, field, value):
        result_dir = self._forge(
            tmp_path, lambda man: man['units'][0].update({field: value})
        )
        with pytest.raises(ValueError, match=field):
            load_quant_manifest(result_dir)

    @pytest.mark.parametrize('size', [True, False, -1, 1.0, '10', None])
    def test_unit_file_size_must_be_a_non_negative_int(self, tmp_path, size):
        result_dir = self._forge(
            tmp_path, lambda man: man['units'][0].update({'file_size': size})
        )
        with pytest.raises(ValueError, match='file_size'):
            load_quant_manifest(result_dir)

    @pytest.mark.parametrize('field', ['sha256', 'entry_digest'])
    @pytest.mark.parametrize('digest', ['', 'deadbeef', 'sha256:zz', None, 1])
    def test_unit_digests_must_be_prefixed_sha256(self, tmp_path, field, digest):
        result_dir = self._forge(
            tmp_path, lambda man: man['units'][0].update({field: digest})
        )
        with pytest.raises(ValueError, match=field):
            load_quant_manifest(result_dir)

    @pytest.mark.parametrize('tensors', [None, {}, 'x', 1])
    def test_unit_tensors_must_be_a_list(self, tmp_path, tensors):
        result_dir = self._forge(
            tmp_path, lambda man: man['units'][0].update({'tensors': tensors})
        )
        with pytest.raises(ValueError, match='tensors'):
            load_quant_manifest(result_dir)

    @pytest.mark.parametrize(
        'item',
        [
            {},
            {'key': 'scale_w'},
            {'key': '', 'dtype': 'float32', 'shape': [2]},
            {'key': 'scale_w', 'dtype': '', 'shape': [2]},
            {'key': 'scale_w', 'dtype': 'float32', 'shape': 2},
            {'key': 'scale_w', 'dtype': 'float32', 'shape': [-1]},
            {'key': 'scale_w', 'dtype': 'float32', 'shape': [True]},
        ],
    )
    def test_tensor_schema_entries_are_validated(self, tmp_path, item):
        result_dir = self._forge(
            tmp_path, lambda man: man['units'][0].update({'tensors': [item]})
        )
        with pytest.raises(ValueError):
            load_quant_manifest(result_dir)

    def test_duplicate_tensor_keys_are_rejected(self, tmp_path):
        entry = {'key': 'scale_w', 'dtype': 'float32', 'shape': [2]}
        result_dir = self._forge(
            tmp_path,
            lambda man: man['units'][0].update({'tensors': [dict(entry), dict(entry)]}),
        )
        with pytest.raises(ValueError, match='more than once'):
            load_quant_manifest(result_dir)


class TestWriterRefusesUnloadableUnits:
    """The writer must not be able to publish what the reader has to reject."""

    @pytest.mark.parametrize('field', ['quant_module_type', 'ori_module_type'])
    @pytest.mark.parametrize('value', [None, '', 1])
    def test_module_type_names_are_required(self, tmp_path, field, value):
        writer = _writer(tmp_path, units=('fc1',), name='req_' + field + str(value))
        kwargs = {'quant_module_type': 'MinMaxQuant', 'ori_module_type': 'Linear'}
        kwargs[field] = value
        with pytest.raises(ValueError, match=field):
            writer.write_unit('fc1', {'scale_w': torch.tensor([0.5])}, **kwargs)

    @pytest.mark.parametrize('field', ['quant_config', 'non_tensor_params'])
    @pytest.mark.parametrize('value', [[], [1, 2], '', 'x', False, 0])
    def test_non_mapping_config_is_refused(self, tmp_path, field, value):
        # `value or {}` would have silently turned [] / '' / False into {} and
        # written a non-object for the rest, which the reader must reject.
        writer = _writer(tmp_path, units=('fc1',), name='cfg{}{}'.format(field, value))
        kwargs = {'quant_module_type': 'MinMaxQuant', 'ori_module_type': 'Linear'}
        kwargs[field] = value
        with pytest.raises(TypeError, match=field):
            writer.write_unit('fc1', {'scale_w': torch.tensor([0.5])}, **kwargs)

    @pytest.mark.parametrize('field', ['quant_config', 'non_tensor_params'])
    def test_none_still_means_empty(self, tmp_path, field):
        writer = _writer(tmp_path, units=('fc1',), name='none' + field)
        kwargs = {'quant_module_type': 'MinMaxQuant', 'ori_module_type': 'Linear'}
        kwargs[field] = None
        entry = writer.write_unit('fc1', {'scale_w': torch.tensor([0.5])}, **kwargs)
        assert entry[field] == {}

    def test_every_published_unit_survives_the_reader(self, tmp_path):
        writer = _writer(tmp_path, units=('fc1',), name='roundtrip')
        _write(writer, 'fc1', quant_config={'algorithm': {'minmax': {}}})
        writer.publish_if_complete()
        manifest = load_quant_manifest(writer.result_dir)
        assert manifest['units'][0]['quant_module_type'] == 'MinMaxQuant'
