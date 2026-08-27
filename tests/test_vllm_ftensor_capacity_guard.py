# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the KVCacheManager capacity guard (issue #437).

GPU-free: stubs torch and the compiled extension, then exercises
``_resolve_manager_num_blocks`` and the ``get_kv_cache_manager`` wiring in
``kvcached.integration.vllm.interfaces``.

Background (diagnosed by @rob-9 on #437): ``alloc_kv_cache`` clamps the
created tensor capacity to the device memory budget (65,536 requested ->
45,056 created on an L4), but ``get_kv_cache_manager`` trusted the caller's
``num_blocks`` unchanged. The manager then exposed page ids beyond the
FTensor's reserved virtual range and the first map past the reservation
aborted the process inside ``FTensor::map``. The guard turns that into a
ValueError at manager construction.
"""
import importlib
import sys
from unittest import mock

import pytest


@pytest.fixture
def iface(monkeypatch):
    torch = mock.MagicMock()
    torch.__version__ = "2.6.0"
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.cuda", torch.cuda)
    monkeypatch.setitem(sys.modules, "torch.utils", torch.utils)
    monkeypatch.setitem(
        sys.modules, "torch.utils.cpp_extension", torch.utils.cpp_extension
    )
    monkeypatch.setitem(sys.modules, "posix_ipc", mock.MagicMock())
    monkeypatch.setitem(sys.modules, "kvcached.vmm_ops", mock.MagicMock())

    def _kvcached_modules():
        return {
            name: module
            for name, module in sys.modules.items()
            if name == "kvcached" or name.startswith("kvcached.")
        }

    saved = _kvcached_modules()
    monkeypatch.delitem(
        sys.modules, "kvcached.integration.vllm.interfaces", raising=False
    )
    module = importlib.import_module("kvcached.integration.vllm.interfaces")
    yield module

    # Importing interfaces above also (re)imported kvcached modules bound to
    # the stubbed torch/vmm_ops and rebound them as parent-package
    # attributes, which outlive monkeypatch's sys.modules restoration. Drop
    # everything created under the stubs and restore the pre-test bindings so
    # later tests in the same process rebuild against their own dependencies.
    for name, mod in _kvcached_modules().items():
        if saved.get(name) is mod:
            continue
        del sys.modules[name]
        parent_name, _, child = name.rpartition(".")
        parent = sys.modules.get(parent_name) or saved.get(parent_name)
        if parent is not None and getattr(parent, child, None) is mod:
            if name in saved:
                setattr(parent, child, saved[name])
            else:
                delattr(parent, child)
    for name, mod in saved.items():
        sys.modules.setdefault(name, mod)


# The #437 reproducer geometry: (2, 65536, 16, 8, 64) float16 on an L4.
BLOCK_SIZE = 16
CELL_SIZE = 1024  # 8 heads * 64 head_dim * 2 bytes
BLOCK_MEM = BLOCK_SIZE * CELL_SIZE
CREATED_BLOCKS = 45_056
REQUESTED_BLOCKS = 65_536
# FTensor reserves K and V halves for MHA (num_kv_buffers=2).
FTENSOR_BYTES = CREATED_BLOCKS * BLOCK_MEM * 2


def _record(iface, group_id=0, num_blocks=CREATED_BLOCKS,
            ftensor_bytes=FTENSOR_BYTES, num_layers=16):
    iface._created_kv_tensor_capacity[group_id] = {
        "num_blocks": num_blocks,
        "ftensor_bytes_per_layer": ftensor_bytes,
        "num_layers": num_layers,
    }


class TestResolveManagerNumBlocks:

    def test_no_record_trusts_explicit_value(self, iface):
        # The vLLM engine/worker split: the manager lives in a process that
        # never called alloc_kv_cache. Behavior must be unchanged there.
        assert iface._resolve_manager_num_blocks(
            REQUESTED_BLOCKS, BLOCK_SIZE, CELL_SIZE, 2, 0) == REQUESTED_BLOCKS

    def test_no_record_rejects_none(self, iface):
        with pytest.raises(ValueError, match="no allocation is recorded"):
            iface._resolve_manager_num_blocks(None, BLOCK_SIZE, CELL_SIZE, 2, 0)

    def test_within_capacity_passes_through(self, iface):
        _record(iface)
        assert iface._resolve_manager_num_blocks(
            1000, BLOCK_SIZE, CELL_SIZE, 2, 0) == 1000

    def test_exact_capacity_passes_through(self, iface):
        _record(iface)
        assert iface._resolve_manager_num_blocks(
            CREATED_BLOCKS, BLOCK_SIZE, CELL_SIZE, 2, 0) == CREATED_BLOCKS

    def test_reproducer_mismatch_raises_naming_both_numbers(self, iface):
        _record(iface)
        with pytest.raises(ValueError) as exc_info:
            iface._resolve_manager_num_blocks(
                REQUESTED_BLOCKS, BLOCK_SIZE, CELL_SIZE, 2, 0)
        message = str(exc_info.value)
        assert str(REQUESTED_BLOCKS) in message
        assert str(CREATED_BLOCKS) in message
        assert "#437" in message

    def test_none_derives_created_capacity(self, iface):
        _record(iface)
        assert iface._resolve_manager_num_blocks(
            None, BLOCK_SIZE, CELL_SIZE, 2, 0) == CREATED_BLOCKS

    def test_mla_single_buffer_units(self, iface):
        # MLA: num_kv_buffers=1, the FTensor holds the single combined KV.
        _record(iface, ftensor_bytes=CREATED_BLOCKS * BLOCK_MEM)
        assert iface._resolve_manager_num_blocks(
            None, BLOCK_SIZE, CELL_SIZE, 1, 0) == CREATED_BLOCKS
        with pytest.raises(ValueError):
            iface._resolve_manager_num_blocks(
                CREATED_BLOCKS + 1, BLOCK_SIZE, CELL_SIZE, 1, 0)

    def test_records_are_per_group(self, iface):
        _record(iface, group_id=0)
        # No record for group 1: explicit value trusted, None rejected.
        assert iface._resolve_manager_num_blocks(
            REQUESTED_BLOCKS, BLOCK_SIZE, CELL_SIZE, 2, 1) == REQUESTED_BLOCKS
        with pytest.raises(ValueError, match="group 1"):
            iface._resolve_manager_num_blocks(None, BLOCK_SIZE, CELL_SIZE, 2, 1)


class TestGetKvCacheManagerGuard:

    def _init(self, iface, monkeypatch):
        manager_calls = []

        class FakeManager:
            def __init__(self, num_blocks, *args, **kwargs):
                manager_calls.append(num_blocks)

        monkeypatch.setattr(iface, "KVCacheManager", FakeManager)
        monkeypatch.setattr(iface, "register_kv_cache_pool",
                            lambda *args, **kwargs: None)
        monkeypatch.setattr(iface, "_kvcached_initialized", True)
        return manager_calls

    def test_oversized_num_blocks_is_refused_before_construction(
            self, iface, monkeypatch):
        manager_calls = self._init(iface, monkeypatch)
        _record(iface)
        with pytest.raises(ValueError, match="exceeds the capacity"):
            iface.get_kv_cache_manager(
                REQUESTED_BLOCKS, BLOCK_SIZE, CELL_SIZE, 16)
        assert manager_calls == []

    def test_valid_num_blocks_reaches_manager_unchanged(
            self, iface, monkeypatch):
        manager_calls = self._init(iface, monkeypatch)
        _record(iface)
        iface.get_kv_cache_manager(CREATED_BLOCKS, BLOCK_SIZE, CELL_SIZE, 16)
        assert manager_calls == [CREATED_BLOCKS]

    def test_none_builds_manager_with_derived_capacity(
            self, iface, monkeypatch):
        manager_calls = self._init(iface, monkeypatch)
        _record(iface)
        iface.get_kv_cache_manager(None, BLOCK_SIZE, CELL_SIZE, 16)
        assert manager_calls == [CREATED_BLOCKS]

    def test_shutdown_clears_the_capacity_record(self, iface, monkeypatch):
        monkeypatch.setattr(iface, "_shutdown_kvcached_impl",
                            lambda: None, raising=False)
        monkeypatch.setattr(iface, "clear_registered_kv_cache_pools",
                            lambda **kwargs: None)
        _record(iface)
        iface.shutdown_kvcached()
        assert iface._created_kv_tensor_capacity == {}
