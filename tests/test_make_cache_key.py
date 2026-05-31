# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``_make_cache_key`` (prefix-cache composite key encoding).

GPU-free: ``_make_cache_key`` is a module-level pure function in
``kvcached.integration.vllm.patches`` and needs neither the ``vmm_ops``
extension, a GPU, nor an installed vLLM. It guards the str-hash fix: some vLLM
versions pass a ``str`` (hex digest) where ``bytes(block_hash)`` would raise
``TypeError``.
"""
import pytest

from kvcached.integration.vllm.patches import _make_cache_key


def _gid(group_id: int) -> bytes:
    return group_id.to_bytes(4, "big", signed=False)


def test_bytes_hash_roundtrip():
    assert _make_cache_key(b"deadbeef", 0) == b"deadbeef" + _gid(0)


def test_str_hash_encoded_like_bytes():
    """A str hash must not raise and must produce the same key as bytes."""
    assert _make_cache_key("deadbeef", 3) == b"deadbeef" + _gid(3)
    assert _make_cache_key("deadbeef", 3) == _make_cache_key(b"deadbeef", 3)


def test_str_does_not_raise_regression():
    """Before the fix, bytes(str) raised TypeError; guard against regression."""
    _make_cache_key("0a1b2c3d" * 8, 0)  # 64-char hex digest, must not raise


def test_group_id_distinguishes_keys():
    h = b"samehash"
    assert _make_cache_key(h, 0) != _make_cache_key(h, 1)
    # group_id is the 4-byte big-endian suffix.
    assert _make_cache_key(h, 1)[-4:] == _gid(1)


@pytest.mark.parametrize("group_id", [0, 1, 255, 256, 65535, 2 ** 31 - 1])
def test_group_id_encoding_width(group_id):
    key = _make_cache_key(b"h", group_id)
    assert key == b"h" + _gid(group_id)
    assert len(key) == len(b"h") + 4
