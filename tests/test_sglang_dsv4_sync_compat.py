# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free source-slice regressions; these do not execute GPU kernels."""

import ast
import functools
import logging
import math
import types
import unittest
from pathlib import Path
from typing import Any, Callable, Optional, cast
from unittest import mock

ROOT = Path(__file__).parents[1]


def load_source_slices():
    tree = ast.parse((ROOT / "kvcached/integration/sglang/patches.py").read_text(encoding="utf-8"))
    names = {
        "_new_tp_scoped_kvcached_allocator",
        "_is_kvcached_managed_pool", "ElasticSWAAllocatorPatch",
        "DeepSeekV4SWAAllocatorPatch",
    }

    class PatchBase:
        logger = logging.getLogger(__name__)

        def _is_already_patched(self, obj, marker):
            return getattr(obj, marker, False)

        def _mark_as_patched(self, obj, marker="__patched__"):
            setattr(obj, marker, True)

    ns = dict(
        Any=Any, Callable=Callable, Optional=Optional, cast=cast,
        types=types, functools=functools, math=math,
        VersionAwarePatch=PatchBase, BasePatch=object,
        version_range=lambda *_: lambda fn: fn,
    )
    nodes = [node for node in tree.body if getattr(node, "name", None) in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "source-slices", "exec"), ns)
    return ns


class TestDeepSeekV4SyncCompatibility(unittest.TestCase):
    def test_each_physical_tp_rank_constructs_local_manager(self):
        create = load_source_slices()["_new_tp_scoped_kvcached_allocator"]
        for rank in (0, 1, 3):
            with self.subTest(rank=rank):
                kvi = types.SimpleNamespace(get_kv_cache_manager=mock.Mock())
                kwargs = dict(num_blocks=65, block_size=1, cell_size=576,
                              num_layers=2, reserve_null_block=True, num_kv_buffers=1,
                              group_id=20000, pool_name="dsv4_swa_kv")
                result = create(kvi, tp_rank=rank, tp_size=4, **kwargs)
                kvi.get_kv_cache_manager.assert_called_once_with(**kwargs)
                self.assertIs(result, kvi.get_kv_cache_manager.return_value)

    def test_native_fallback_after_registered_alias_order(self):
        ns = load_source_slices()
        for page_size in (1, 4):
            for alias_first in (False, True):
                with self.subTest(page_size=page_size, alias_first=alias_first):
                    self.check_dispatch(ns, page_size, alias_first)

    def check_dispatch(self, ns, page_size, alias_first):
        class Native:
            def __init__(self, *args):
                self.pool = args[-1]

        class Elastic:
            _native_allocator_cls = Native

            def __init__(self, *args):
                self.pool = args[-1]
                if not hasattr(self.pool, "kvcached_allocator"):
                    raise ValueError("elastic allocator requires managed pool")

        allocator: Any = types.ModuleType("sglang.srt.mem_cache.allocator")
        allocator.ElasticTokenToKVPoolAllocator = Elastic
        allocator.ElasticPagedTokenToKVPoolAllocator = Elastic
        mem_cache: Any = types.ModuleType("sglang.srt.mem_cache")
        mem_cache.allocator = allocator
        swa = types.SimpleNamespace(TokenToKVPoolAllocator=Native,
                                    PagedTokenToKVPoolAllocator=Native)

        class SWA:
            def __init__(self, size, size_swa, page_size, dtype, device, kvcache,
                         need_sort):
                def alloc(pool):
                    if page_size == 1:
                        return swa.TokenToKVPoolAllocator(size, dtype, device, pool)
                    return swa.PagedTokenToKVPoolAllocator(
                        size, page_size, dtype, device, pool)
                self.full = alloc(getattr(kvcache, "full_kv_pool", None))
                self.swa = alloc(kvcache.swa_kv_pool)

        swa.SWATokenToKVPoolAllocator = SWA
        with mock.patch.dict("sys.modules", {"sglang.srt.mem_cache": mem_cache}), mock.patch(
            "kvcached.integration.sglang.dsv4_backing.bind_native_swa"
        ) as bind:
            if alias_first:
                self.assertTrue(ns["ElasticSWAAllocatorPatch"]().alias_swa_sub_allocators(swa))
            patch = ns["DeepSeekV4SWAAllocatorPatch"]()
            self.assertTrue(patch.patch_swa_allocator(swa))
            self.assertTrue(patch.patch_swa_allocator(swa))  # idempotent
            native_pool = types.SimpleNamespace(full_kv_pool=object(), swa_kv_pool=object())
            native = SWA(64, 32, page_size, object(), "cuda:0", native_pool, False)
            self.assertIs(type(native.full), Native)
            self.assertIs(type(native.swa), Native)
            # Existing elastic MHA pools predate the DeepSeek-V4 marker.
            mha_full = types.SimpleNamespace(kvcached_allocator=object())
            mha_swa = types.SimpleNamespace(kvcached_allocator=object())
            mha_pool = types.SimpleNamespace(full_kv_pool=mha_full,
                                            swa_kv_pool=mha_swa)
            mha = SWA(64, 32, page_size, object(), "cuda:0", mha_pool, False)
            self.assertIs(type(mha.full), Elastic)
            self.assertIs(type(mha.swa), Elastic)
            self.assertIs(mha.full.pool, mha_full)
            self.assertIs(mha.swa.pool, mha_swa)
            # Zero-attention-layer elastic pools deliberately have no manager.
            empty = types.SimpleNamespace(kvcached_allocator=None)
            empty_pool = types.SimpleNamespace(full_kv_pool=empty, swa_kv_pool=empty)
            empty_alloc = SWA(64, 32, page_size, object(), "cuda:0", empty_pool, False)
            self.assertIs(type(empty_alloc.full), Native)
            self.assertIs(type(empty_alloc.swa), Native)
            physical = types.SimpleNamespace(kvcached_allocator=object(),
                                             _kvcached_managed=True,
                                             _kvcached_dsv4_backing=object())
            managed_pool = types.SimpleNamespace(swa_kv_pool=physical)
            managed = SWA(64, 32, page_size, object(), "cuda:0", managed_pool, False)
            self.assertIs(type(managed.full), Native)
            self.assertIs(type(managed.swa), Native)
            self.assertIs(managed.swa.pool, physical)
            self.assertIsNone(managed.full.pool)
            self.assertFalse(hasattr(managed_pool, "full_kv_pool"))
            bind.assert_called_once_with(managed, physical._kvcached_dsv4_backing, None)


if __name__ == "__main__":
    unittest.main()
