# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Dependency-free ownership checks for the native metadata adapter."""

import importlib.util
import os
import types
from collections import OrderedDict
from pathlib import Path
from unittest.mock import Mock

# The helper itself is dependency-free until initialization; load it without the
# package's torch requirement so its lifetime checks also run on a plain CPU host.
_spec = importlib.util.spec_from_file_location(
    "_native_block_pool_under_test",
    Path(os.environ.get("ENGINE_COMPAT_SOURCE", Path(__file__).parents[1]))
    / "kvcached/integration/vllm/native_block_pool.py",
)
assert _spec is not None and _spec.loader is not None
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
Mixin = _module.NativeBlockPoolMixin


class Block:
    def __init__(self, block_id, ref_cnt=0, is_null=False):
        self.block_id = block_id
        self.ref_cnt = ref_cnt
        self.is_null = is_null
        self.block_hash = None

    def reset_hash(self):
        self.block_hash = None


def bare_pool():
    pool = object.__new__(Mixin)
    pool.blocks = [Block(0, is_null=True), Block(1, ref_cnt=1)]
    pool.enable_prefix_cache = False
    pool._evictable_blocks = OrderedDict()
    pool.kv_cache_manager = types.SimpleNamespace(free=Mock())
    pool._block_id_to_key = {}
    return pool


def test_no_apc_retained_copy_waits_for_last_reference():
    pool = bare_pool()
    null, block = pool.blocks
    pool.touch([null, block])
    assert block.ref_cnt == 2 and null.ref_cnt == 0

    pool.free_blocks([null, block])
    assert block.ref_cnt == 1
    pool.kv_cache_manager.free.assert_not_called()

    pool.free_blocks([block])
    assert block.ref_cnt == 0
    pool.kv_cache_manager.free.assert_called_once_with([1])


def test_elastic_eviction_removes_native_aliases_before_releasing_marker():
    pool = bare_pool()
    block = pool.blocks[1]
    pool._block_id_to_key[1] = b"primary"
    remove = Mock(return_value=[b"primary", b"alias"])
    pool._native_block_pool = types.SimpleNamespace(_remove_cached_block_hashes=remove)

    assert pool._remove_cached_block(b"primary", 1) is block
    remove.assert_called_once_with(pool, block)
    assert 1 not in pool._block_id_to_key


def test_reset_refuses_a_pending_copy_without_mutating_cache():
    pool = bare_pool()
    pool.enable_prefix_cache = True
    pool.blocks[1].block_hash = b"primary"
    pool._block_id_to_key[1] = b"primary"
    pool._evict_blocks_from_pool = Mock()

    assert pool.reset_prefix_cache() is False
    pool._evict_blocks_from_pool.assert_not_called()
    assert pool.blocks[1].block_hash == b"primary"
    assert pool._block_id_to_key == {1: b"primary"}
