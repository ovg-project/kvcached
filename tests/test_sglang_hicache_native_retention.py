# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Exercise the pinned SGLang HiCache write-through completion paths on CPU."""

from __future__ import annotations

import ast
import dataclasses
import heapq
import logging
import os
import sys
import time
from array import array
from collections import defaultdict
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from kvcached import prefix_cache_retention as retention
from kvcached.integration.patch_base import PatchManager
from kvcached.integration.sglang import autopatch
from kvcached.integration.version_utils import VersionManager


@pytest.fixture
def native(monkeypatch):
    root = Path(os.environ.get("SGLANG_SOURCE", "engine_integration/sglang"))
    source = root / "python/sglang/srt/mem_cache"
    if "SGLANG_SOURCE" not in os.environ and not (source / "hiradix_cache.py").is_file():
        pytest.skip("native SGLang source unavailable")

    # Load the upstream classes without importing the GPU runtime. Postponed
    # annotations let the CPU test omit dependencies only used in type hints.
    # Only one rank participates. SGLang's optional consensus instrumentation
    # is a passthrough in this configuration.
    def rank_consensus(fn=None, **kwargs):
        return (lambda method: method) if fn is None else fn

    ns = dict(__name__=__name__, BasePrefixCache=object, dataclasses=dataclasses,
              BASE_COMPONENT_TYPE="full", rank_consensus=rank_consensus,
              defaultdict=defaultdict, heapq=heapq, time=time, torch=torch, array=array,
              logger=logging.getLogger(__name__), StorageMedium=SimpleNamespace(CPU="cpu", GPU="gpu"))
    for filename, names in (
        ("base_prefix_cache.py", {"EvictParams", "EvictResult", "IncLockRefResult", "DecLockRefResult"}),
        ("radix_cache.py", {"RadixCache", "RadixKey", "TreeNode"}),
        ("hiradix_cache.py", {"HiRadixCache"}),
        ("unified_radix_cache.py", {"UnifiedRadixCache"}),
    ):
        path = source / filename
        tree = ast.parse(path.read_text(encoding="utf-8"))
        tree.body = [node for node in tree.body
                     if isinstance(node, ast.ClassDef) and node.name in names]
        exec(compile(tree, str(path), "exec"), ns)
    native = SimpleNamespace(**ns)
    monkeypatch.setitem(sys.modules, "sglang.srt.mem_cache.base_prefix_cache", native)

    # Exercise registration as well as each adapter's implementation.
    def apply_cache_patches(manager):
        return {patch.patch_name: patch.apply(native) for patch, _ in manager.patches
                if patch.target_class in {"RadixCache", "HiRadixCache", "UnifiedRadixCache"}}

    monkeypatch.setenv("ENABLE_KVCACHED", "1")
    monkeypatch.setenv("KVCACHED_AUTOPATCH", "1")
    monkeypatch.setenv("KVCACHED_RANK_COORDINATION_ENABLE", "0")
    monkeypatch.setattr(VersionManager.get_instance(), "detect_version", lambda library: "0.5.20")
    monkeypatch.setattr(PatchManager, "apply_all_patches", apply_cache_patches)
    monkeypatch.setattr(retention, "_policy", retention._PrefixCacheRetentionPolicy(-1))
    autopatch._patch_sglang(ModuleType("sglang"))
    return native


@pytest.mark.parametrize("bigram,change_scope,splits_to_make", [
    (False, False, 1), (True, False, 1), (False, True, 1), (True, True, 1), (False, False, 510),
])
def test_shared_ballot_revalidates_prefix_after_native_parent_split(native, bigram, change_scope, splits_to_make,
                                                                    monkeypatch):
    from kvcached.integration.sglang import hicache_retention as parallel

    root, parent, leaf = (native.TreeNode() for _ in range(3))
    parent.parent, leaf.parent = root, parent
    end = splits_to_make + 2
    parent.key = native.RadixKey(array("q", range(1, end + 1)), is_bigram=bigram)
    leaf.key = native.RadixKey(array("q", [end, end + 1] if bigram else [end + 1]), is_bigram=bigram)
    root.children[parent.key.child_key()] = parent
    parent.children[leaf.key.child_key()] = leaf
    # No store or event hashes have been generated for this prefix.
    monkeypatch.setitem(native.HiRadixCache._split_node.__globals__, "split_node_hash_value",
                        lambda value, *_: (None, None) if value is None else pytest.fail("unexpected hash"))
    accepted = []
    splits: list[object] = []
    width = 32 * parallel._VICTIMS_PER_BATCH

    def vote(data, op):
        if data.numel() == 2 * width + 4:
            data[0] = data[width] = 0  # The peer proposes a different first victim.
        elif data.numel() == width + 4 and not splits:
            for _ in range(splits_to_make):
                splits.append(native.HiRadixCache._split_node(cache, parent.key, parent, 1))
            if change_scope:
                parent.key.cache_salt = "other"

    cache = SimpleNamespace(
        pp_size=1, tp_world_size=2, page_size=1, root_node=root, evictable_leaves={leaf},
        eviction_strategy=SimpleNamespace(get_priority=lambda n: n.priority),
        evictable_size_=1, protected_size_=0, cache_controller=SimpleNamespace(write_policy="write_through"),
        _all_reduce_attn_groups=vote, _all_reduce=lambda *_: None,
        _enforce_prefix_cache_retention=lambda *_: None, update_eviction_metrics=lambda *_: None)
    def evict(node):
        accepted.append(node)
        cache.evictable_leaves.remove(node)
        cache.evictable_size_ -= len(node.key)
        return len(node.key)

    cache._evict_regular = evict
    cache.check_hicache_events = lambda: cache._all_reduce(torch.zeros(2), torch.distributed.ReduceOp.MIN)
    scheduler = SimpleNamespace(tree_cache=cache, enable_hierarchical_cache=True, _engine_paused=False,
                                process_input_requests=lambda _: None)
    monkeypatch.setattr(retention, "_policy", retention._PrefixCacheRetentionPolicy(0))
    parallel.install_parallel_hicache_retention(scheduler)
    extend = Mock(wraps=parallel._extend_path)
    monkeypatch.setattr(parallel, "_extend_path", extend)
    cache.check_hicache_events()
    assert len(splits) == splits_to_make and splits[-1] is parent.parent
    if splits_to_make > parallel._INDEX_PER_POLL:
        assert not accepted and extend.call_count <= parallel._INDEX_PER_POLL
        for _ in range(10):
            extend.reset_mock()
            cache.check_hicache_events()
            assert extend.call_count <= parallel._INDEX_PER_POLL
            if accepted:
                break
    assert accepted == ([] if change_scope else [leaf])


def test_write_through_ack_trims_unlocked_device_kv(native):
    # Bypass GPU pool construction. The cache owns the real tree, lock counts,
    # completion queues and eviction methods; the controller stands in for DMA.
    cache = native.HiRadixCache.__new__(native.HiRadixCache)
    cache.root_node = native.TreeNode()
    cache.evictable_leaves, cache.evictable_host_leaves = set(), set()
    cache.evictable_size_ = cache.protected_size_ = 0
    cache.disable = cache.enable_storage = cache.enable_storage_metrics = False
    cache.metrics_collector = None
    cache.pp_size = cache.tp_world_size = 1
    cache.pp_rank = 0
    cache.attn_cp_group = cache.attn_tp_group = None
    cache.work_list = []
    cache.ongoing_write_through = {}
    cache.kv_events = Mock()
    cache.update_eviction_metrics = Mock()
    cache.eviction_strategy = SimpleNamespace(get_priority=lambda node: node.last_access_time)
    cache._get_extra_pools = lambda: {}
    controller = Mock(write_policy="write_through", ack_write_queue=[], ack_load_queue=[])
    cache.cache_controller = controller
    retention._policy.register(cache)

    def start_backup(start):
        node = native.TreeNode()
        node.parent = cache.root_node
        node.key = range(start, start + 8)
        node.value = torch.arange(start, start + 8)
        cache.root_node.children[start] = node
        cache.evictable_size_ += len(node.key)
        cache._update_leaf_status(node)
        controller.write.return_value = node.value + 100
        assert cache.write_backup(node) == len(node.key)
        return node

    idle, active = start_backup(0), start_backup(8)
    cache.inc_lock_ref(active)
    retention._policy.set_limit(0)
    event = Mock()
    ack = SimpleNamespace(node_ids=[idle.id, active.id], finish_event=event)
    controller.ack_write_queue.append(ack)
    event.query.return_value = False
    cache.check_hicache_events()
    assert cache.evictable_size() == 0 and not idle.evicted
    controller.evict_device.assert_not_called()

    event.query.return_value = True
    cache.check_hicache_events()
    event.synchronize.assert_called_once()
    assert not active.evicted and active.lock_ref == 1
    assert idle.host_value.tolist() == list(range(100, 108))
    assert cache.evictable_size() == 0
    controller.evict_device.assert_called_once()
    torch.testing.assert_close(controller.evict_device.call_args.args[0], torch.arange(8))
    assert idle.evicted


def test_unified_write_through_ack_requests_native_eviction(native):
    cache = native.UnifiedRadixCache.__new__(native.UnifiedRadixCache)
    cache.tree_core = Mock(enable_storage=False, write_back_duplicate_reclaim_digest=0)
    cache.session = Mock()
    cache.session.try_dec_lock_ref.return_value = None
    cache.linker = cache.buffer_pipeline = cache.metrics_collector = None
    cache.disable = cache.enable_storage_metrics = False
    cache.pp_size = cache.tp_world_size = 1
    cache.pp_rank = 0
    cache.attn_cp_group = cache.attn_tp_group = None
    cache.work_list = []
    event, receipt = Mock(), object()
    cache.cache_controller = SimpleNamespace(
        ack_write_queue=[SimpleNamespace(finish_event=event, node_ids=[42])], ack_load_queue=[],
    )
    cache.ongoing_write_through = {42: (7, receipt, [7])}
    # The tree core owns lock accounting and physical eviction. Check the
    # native facade dispatches the precise excess only after releasing the lock.
    def evictable_tokens():
        return 8 if cache.tree_core.dec_lock_ref.called else 0

    cache.tree_core.full_evictable_size.side_effect = evictable_tokens
    cache._evict = Mock()
    retention._policy.register(cache)
    retention._policy.set_limit(3)

    event.query.return_value = False
    cache.check_hicache_events()
    cache.tree_core.dec_lock_ref.assert_not_called()
    cache._evict.assert_not_called()

    event.query.return_value = True
    cache.check_hicache_events()
    event.synchronize.assert_called_once()
    cache.tree_core.finish_write_through.assert_called_once_with([7], 42)
    cache.tree_core.dec_lock_ref.assert_called_once_with(7, receipt, False)
    cache._evict.assert_called_once_with(native.EvictParams(num_tokens=5))


def test_retention_hook_skipped_when_kvcached_disabled(native, monkeypatch):
    cache = native.HiRadixCache.__new__(native.HiRadixCache)
    cache._enforce_prefix_cache_retention = Mock()
    monkeypatch.setenv("ENABLE_KVCACHED", "0")
    cache.loading_check(finish_count=0)
    cache._enforce_prefix_cache_retention.assert_not_called()
    assert not hasattr(cache, "_kvcached_hicache_retention_depth")
