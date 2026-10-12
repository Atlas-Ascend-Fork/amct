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
"""Reusable filesystem helpers for atomic writes and directory validation."""

import hashlib
import os
import stat

_HASH_CHUNK_SIZE = 1024 * 1024
_FILE_MODE = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP


def file_sha256_hex(file_path):
    """Return the unprefixed SHA-256 hex digest of a file."""
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as file_handle:
        for chunk in iter(lambda: file_handle.read(_HASH_CHUNK_SIZE), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def validate_writable_directory(path, *, name, type_hint='str path', empty_hint=None):
    """Validate an existing directory that a caller wants to write into."""
    if not isinstance(path, str):
        raise TypeError(
            '{} must be a {}, got {}.'.format(name, type_hint, type(path).__name__)
        )
    if path.strip() == '':
        message = '{} must be a non-empty path'.format(name)
        if empty_hint:
            message += '; {}'.format(empty_hint)
        raise ValueError(message + '.')

    target = os.path.abspath(os.path.expanduser(path))
    if not os.path.exists(target):
        raise ValueError("{} '{}' must be an existing directory.".format(name, target))
    if not os.path.isdir(target):
        raise ValueError(
            "{} '{}' must be an existing directory, not a file.".format(name, target)
        )
    if not os.access(target, os.R_OK | os.W_OK | os.X_OK):
        raise ValueError(
            "{} '{}' must be readable, writable, and searchable.".format(name, target)
        )
    return target


def temp_name_for(file_path):
    """Same-directory temp name for an atomic write of file_path."""
    return os.path.join(
        os.path.dirname(file_path),
        '.{}.{}.tmp'.format(os.path.basename(file_path), os.getpid()),
    )


def atomic_write_via(file_path, write_payload, file_mode=_FILE_MODE):
    """Atomically produce file_path: temp file, fsync, then replace."""
    tmp_path = temp_name_for(file_path)
    try:
        write_payload(tmp_path)
        _fsync_file(tmp_path)
        os.chmod(tmp_path, file_mode)
        os.replace(tmp_path, file_path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
    _fsync_dir(os.path.dirname(file_path))


def write_text_atomically(file_path, text, file_mode=_FILE_MODE):
    """Atomically write UTF-8 text using the shared atomic-file protocol."""

    def _write(tmp_path):
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        file_descriptor = os.open(tmp_path, flags, file_mode)
        with os.fdopen(file_descriptor, 'w', encoding='utf-8') as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())

    atomic_write_via(file_path, _write, file_mode=file_mode)


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
