# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the KVCacheManager capacity guard (issue #437).

GPU-free: stubs torch and the compiled extension, then exercises
``_resolve_manager_num_blocks``, the ``get_kv_cache_manager`` wiring and the
manager-first check in ``alloc_kv_cache`` in
``kvcached.integration.vllm.interfaces``.

Background (diagnosed by @rob-9 on #437): ``alloc_kv_cache`` clamps the
created tensor capacity to the device memory budget (65,536 requested ->
45,056 created on an L4), but ``get_kv_cache_manager`` trusted the caller's
``num_blocks`` unchanged. The manager then exposed page ids beyond the
FTensor's reserved virtual range and the first map past the reservation
aborted the process inside ``FTensor::map``. The guard turns that into a
ValueError at whichever call comes second: ``get_kv_cache_manager`` after
``alloc_kv_cache``, or ``alloc_kv_cache`` after ``get_kv_cache_manager``
(raised before the tensors exist, since the manager starts mapping as soon as
they do). It also requires the manager's ``num_layers`` and
``num_kv_buffers`` to match the allocation: in the contiguous layout they set
the compound page stride on both sides.
"""
import gc
import importlib
import sys
import types
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

    # Managers built through get_kv_cache_manager() live in the process-wide
    # pool registry; forget them before the module graph is restored.
    module.clear_registered_kv_cache_pools(integration="vllm")

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


# The #437 reproducer geometry: (2, 65536, 16, 8, 64) float16, 16 layers.
NUM_LAYERS = 16
BLOCK_SIZE = 16
NUM_KV_HEADS = 8
HEAD_DIM = 64
CELL_SIZE = NUM_KV_HEADS * HEAD_DIM * 2  # bytes per token per K or V (fp16)
BLOCK_MEM = BLOCK_SIZE * CELL_SIZE
CREATED_BLOCKS = 45_056
REQUESTED_BLOCKS = 65_536
KV_SHAPE = (2, REQUESTED_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM)
# FTensor reserves K and V halves for MHA (num_kv_buffers=2).
NUM_KV_BUFFERS = 2
FTENSOR_BYTES = CREATED_BLOCKS * BLOCK_MEM * NUM_KV_BUFFERS
# alloc_kv_cache() budgets total_memory // num_layers // num_kv_buffers per
# layer per K-or-V and aligns it down to PAGE_SIZE. This total lands exactly
# on 352 pages of 2 MiB per layer per K-or-V: the 45,056 blocks of the
# report, with the requested 65,536 clamped.
PAGE_SIZE = 2 * 1024 * 1024
GPU_TOTAL_MEMORY = FTENSOR_BYTES * NUM_LAYERS
FP16 = types.SimpleNamespace(itemsize=2)


def _record(iface, group_id=0, num_blocks=CREATED_BLOCKS,
            ftensor_bytes=FTENSOR_BYTES, num_layers=NUM_LAYERS,
            num_kv_buffers=NUM_KV_BUFFERS):
    iface._created_kv_tensor_capacity[group_id] = {
        "num_blocks": num_blocks,
        "ftensor_bytes_per_layer": ftensor_bytes,
        "num_layers": num_layers,
        "num_kv_buffers": num_kv_buffers,
    }


class FakeManager:
    """Stands in for KVCacheManager with the attributes the guard reads."""

    def __init__(self, num_blocks, block_size, cell_size, num_layers, *args,
                 num_kv_buffers=2, group_id=0, **kwargs):
        self.num_blocks = num_blocks
        self.block_mem_size = block_size * cell_size
        self.num_layers = num_layers
        self.num_kv_buffers = num_kv_buffers
        self.group_id = group_id


def _enable_manager_factory(iface, monkeypatch):
    """Let get_kv_cache_manager() build FakeManagers; returns those built.

    Registration stays real so alloc_kv_cache() finds the managers in the
    pool registry, as it does in production.
    """
    managers = []

    class RecordingManager(FakeManager):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            managers.append(self)

    monkeypatch.setattr(iface, "KVCacheManager", RecordingManager)
    monkeypatch.setattr(iface, "_kvcached_initialized", True)
    return managers


def _enable_alloc(iface, monkeypatch):
    """Let alloc_kv_cache() run under the stubs; returns create_kv_tensors calls.

    The stubbed device reports GPU_TOTAL_MEMORY, so the reproducer shape's
    65,536 requested blocks clamp to 45,056 created blocks.
    """
    create_calls = []

    def create_kv_tensors(*args, **kwargs):
        create_calls.append((args, kwargs))
        return [mock.MagicMock()]

    monkeypatch.setattr(iface, "create_kv_tensors", create_kv_tensors)
    monkeypatch.setattr(iface, "_kvcached_initialized", True)
    monkeypatch.setattr(iface, "_contiguous_layout", True)
    monkeypatch.setattr(iface, "PAGE_SIZE", PAGE_SIZE)
    device_properties = iface.torch.cuda.get_device_properties.return_value
    device_properties.total_memory = GPU_TOTAL_MEMORY
    return create_calls


def _alloc(iface, num_layers=NUM_LAYERS, group_id=0):
    return iface.alloc_kv_cache(
        KV_SHAPE, BLOCK_SIZE, FP16, "cuda:0", num_layers, group_id=group_id)


class TestResolveManagerNumBlocks:

    def test_no_record_trusts_explicit_value(self, iface):
        # The vLLM engine/worker split: the manager lives in a process that
        # never called alloc_kv_cache. Behavior must be unchanged there.
        assert iface._resolve_manager_num_blocks(
            REQUESTED_BLOCKS, BLOCK_SIZE, CELL_SIZE, NUM_LAYERS, 2, 0
        ) == REQUESTED_BLOCKS

    def test_no_record_rejects_none(self, iface):
        with pytest.raises(ValueError, match="no allocation is recorded"):
            iface._resolve_manager_num_blocks(
                None, BLOCK_SIZE, CELL_SIZE, NUM_LAYERS, 2, 0)

    def test_within_capacity_passes_through(self, iface):
        _record(iface)
        assert iface._resolve_manager_num_blocks(
            1000, BLOCK_SIZE, CELL_SIZE, NUM_LAYERS, 2, 0) == 1000

    def test_exact_capacity_passes_through(self, iface):
        _record(iface)
        assert iface._resolve_manager_num_blocks(
            CREATED_BLOCKS, BLOCK_SIZE, CELL_SIZE, NUM_LAYERS, 2, 0
        ) == CREATED_BLOCKS

    def test_reproducer_mismatch_raises_naming_both_numbers(self, iface):
        _record(iface)
        with pytest.raises(ValueError) as exc_info:
            iface._resolve_manager_num_blocks(
                REQUESTED_BLOCKS, BLOCK_SIZE, CELL_SIZE, NUM_LAYERS, 2, 0)
        message = str(exc_info.value)
        assert str(REQUESTED_BLOCKS) in message
        assert str(CREATED_BLOCKS) in message
        assert "#437" in message

    def test_none_derives_created_capacity(self, iface):
        _record(iface)
        assert iface._resolve_manager_num_blocks(
            None, BLOCK_SIZE, CELL_SIZE, NUM_LAYERS, 2, 0) == CREATED_BLOCKS

    def test_mla_single_buffer_units(self, iface):
        # MLA: num_kv_buffers=1, the FTensor holds the single combined KV.
        _record(iface, ftensor_bytes=CREATED_BLOCKS * BLOCK_MEM,
                num_kv_buffers=1)
        assert iface._resolve_manager_num_blocks(
            None, BLOCK_SIZE, CELL_SIZE, NUM_LAYERS, 1, 0) == CREATED_BLOCKS
        with pytest.raises(ValueError):
            iface._resolve_manager_num_blocks(
                CREATED_BLOCKS + 1, BLOCK_SIZE, CELL_SIZE, NUM_LAYERS, 1, 0)

    def test_records_are_per_group(self, iface):
        _record(iface, group_id=0)
        # No record for group 1: explicit value trusted, None rejected.
        assert iface._resolve_manager_num_blocks(
            REQUESTED_BLOCKS, BLOCK_SIZE, CELL_SIZE, NUM_LAYERS, 2, 1
        ) == REQUESTED_BLOCKS
        with pytest.raises(ValueError, match="group 1"):
            iface._resolve_manager_num_blocks(
                None, BLOCK_SIZE, CELL_SIZE, NUM_LAYERS, 2, 1)

    def test_num_layers_mismatch_raises_naming_both(self, iface):
        _record(iface)
        with pytest.raises(
                ValueError, match="num_layers=32 does not match num_layers=16"):
            iface._resolve_manager_num_blocks(
                CREATED_BLOCKS, BLOCK_SIZE, CELL_SIZE, 32, 2, 0)

    def test_num_kv_buffers_mismatch_raises_naming_both(self, iface):
        _record(iface)
        with pytest.raises(
                ValueError,
                match="num_kv_buffers=1 does not match num_kv_buffers=2"):
            iface._resolve_manager_num_blocks(
                CREATED_BLOCKS, BLOCK_SIZE, CELL_SIZE, NUM_LAYERS, 1, 0)

    def test_none_derivation_still_checks_layout(self, iface):
        _record(iface)
        with pytest.raises(ValueError, match="num_layers=32"):
            iface._resolve_manager_num_blocks(
                None, BLOCK_SIZE, CELL_SIZE, 32, 2, 0)


class TestGetKvCacheManagerGuard:

    def test_oversized_num_blocks_is_refused_before_construction(
            self, iface, monkeypatch):
        managers = _enable_manager_factory(iface, monkeypatch)
        _record(iface)
        with pytest.raises(ValueError, match="exceeds the capacity"):
            iface.get_kv_cache_manager(
                REQUESTED_BLOCKS, BLOCK_SIZE, CELL_SIZE, NUM_LAYERS)
        assert managers == []

    def test_valid_num_blocks_reaches_manager_unchanged(
            self, iface, monkeypatch):
        managers = _enable_manager_factory(iface, monkeypatch)
        _record(iface)
        iface.get_kv_cache_manager(
            CREATED_BLOCKS, BLOCK_SIZE, CELL_SIZE, NUM_LAYERS)
        assert [m.num_blocks for m in managers] == [CREATED_BLOCKS]

    def test_none_builds_manager_with_derived_capacity(
            self, iface, monkeypatch):
        managers = _enable_manager_factory(iface, monkeypatch)
        _record(iface)
        iface.get_kv_cache_manager(None, BLOCK_SIZE, CELL_SIZE, NUM_LAYERS)
        assert [m.num_blocks for m in managers] == [CREATED_BLOCKS]

    def test_num_layers_mismatch_is_refused_before_construction(
            self, iface, monkeypatch):
        managers = _enable_manager_factory(iface, monkeypatch)
        _record(iface)
        with pytest.raises(
                ValueError, match="num_layers=32 does not match num_layers=16"):
            iface.get_kv_cache_manager(CREATED_BLOCKS, BLOCK_SIZE, CELL_SIZE, 32)
        assert managers == []

    def test_num_kv_buffers_mismatch_is_refused_before_construction(
            self, iface, monkeypatch):
        managers = _enable_manager_factory(iface, monkeypatch)
        _record(iface)
        with pytest.raises(
                ValueError,
                match="num_kv_buffers=1 does not match num_kv_buffers=2"):
            iface.get_kv_cache_manager(
                CREATED_BLOCKS, BLOCK_SIZE, CELL_SIZE, NUM_LAYERS,
                num_kv_buffers=1)
        assert managers == []

    def test_shutdown_clears_the_capacity_record(self, iface, monkeypatch):
        monkeypatch.setattr(iface, "_shutdown_kvcached_impl",
                            lambda: None, raising=False)
        monkeypatch.setattr(iface, "clear_registered_kv_cache_pools",
                            lambda **kwargs: None)
        _record(iface)
        iface.shutdown_kvcached()
        assert iface._created_kv_tensor_capacity == {}


class TestManagerFirstOrder:
    """get_kv_cache_manager() before alloc_kv_cache(): alloc revalidates."""

    def test_manager_beyond_clamped_capacity_refuses_alloc(
            self, iface, monkeypatch):
        managers = _enable_manager_factory(iface, monkeypatch)
        create_calls = _enable_alloc(iface, monkeypatch)
        # Nothing is recorded yet, so the reproducer's 65,536 is trusted...
        iface.get_kv_cache_manager(
            REQUESTED_BLOCKS, BLOCK_SIZE, CELL_SIZE, NUM_LAYERS)
        assert [m.num_blocks for m in managers] == [REQUESTED_BLOCKS]
        # ...and alloc_kv_cache(), which clamps to 45,056, must refuse it.
        with pytest.raises(ValueError) as exc_info:
            _alloc(iface)
        message = str(exc_info.value)
        assert str(REQUESTED_BLOCKS) in message
        assert str(CREATED_BLOCKS) in message
        # Refused before the tensors existed: nothing became mappable.
        assert create_calls == []
        assert iface._created_kv_tensor_capacity == {}

    def test_num_layers_mismatch_refuses_alloc(self, iface, monkeypatch):
        _enable_manager_factory(iface, monkeypatch)
        create_calls = _enable_alloc(iface, monkeypatch)
        iface.get_kv_cache_manager(CREATED_BLOCKS, BLOCK_SIZE, CELL_SIZE, 32)
        with pytest.raises(
                ValueError, match="num_layers=32 does not match num_layers=16"):
            _alloc(iface, num_layers=NUM_LAYERS)
        assert create_calls == []

    def test_num_kv_buffers_mismatch_refuses_alloc(self, iface, monkeypatch):
        _enable_manager_factory(iface, monkeypatch)
        create_calls = _enable_alloc(iface, monkeypatch)
        # An MLA-style single-buffer manager over MHA tensors (K and V halves).
        iface.get_kv_cache_manager(
            CREATED_BLOCKS, BLOCK_SIZE, CELL_SIZE, NUM_LAYERS, num_kv_buffers=1)
        with pytest.raises(
                ValueError,
                match="num_kv_buffers=1 does not match num_kv_buffers=2"):
            _alloc(iface)
        assert create_calls == []

    def test_matching_manager_lets_alloc_proceed_and_record(
            self, iface, monkeypatch):
        _enable_manager_factory(iface, monkeypatch)
        create_calls = _enable_alloc(iface, monkeypatch)
        iface.get_kv_cache_manager(
            CREATED_BLOCKS, BLOCK_SIZE, CELL_SIZE, NUM_LAYERS)
        kv_tensors = _alloc(iface)
        assert len(kv_tensors) == NUM_LAYERS
        assert len(create_calls) == 1
        assert iface._created_kv_tensor_capacity[0] == {
            "num_blocks": CREATED_BLOCKS,
            "ftensor_bytes_per_layer": FTENSOR_BYTES,
            "num_layers": NUM_LAYERS,
            "num_kv_buffers": NUM_KV_BUFFERS,
        }

    def test_alloc_first_then_none_derives_created_capacity(
            self, iface, monkeypatch):
        # The other order end to end: the record written by alloc_kv_cache()
        # feeds get_kv_cache_manager(num_blocks=None) and refuses the
        # reproducer's original request.
        managers = _enable_manager_factory(iface, monkeypatch)
        _enable_alloc(iface, monkeypatch)
        _alloc(iface)
        iface.get_kv_cache_manager(None, BLOCK_SIZE, CELL_SIZE, NUM_LAYERS)
        assert [m.num_blocks for m in managers] == [CREATED_BLOCKS]
        with pytest.raises(ValueError, match="exceeds the capacity"):
            iface.get_kv_cache_manager(
                REQUESTED_BLOCKS, BLOCK_SIZE, CELL_SIZE, NUM_LAYERS)

    def test_manager_of_another_group_is_not_compared(
            self, iface, monkeypatch):
        _enable_manager_factory(iface, monkeypatch)
        create_calls = _enable_alloc(iface, monkeypatch)
        iface.get_kv_cache_manager(
            REQUESTED_BLOCKS, BLOCK_SIZE, CELL_SIZE, NUM_LAYERS, group_id=1)
        _alloc(iface, group_id=0)
        assert len(create_calls) == 1

    def test_collected_manager_is_not_compared(self, iface, monkeypatch):
        managers = _enable_manager_factory(iface, monkeypatch)
        create_calls = _enable_alloc(iface, monkeypatch)
        iface.get_kv_cache_manager(
            REQUESTED_BLOCKS, BLOCK_SIZE, CELL_SIZE, NUM_LAYERS)
        # The pool registry holds weak references: a manager that is gone
        # (e.g. from a torn-down engine) cannot veto a new allocation.
        managers.clear()
        gc.collect()
        _alloc(iface)
        assert len(create_calls) == 1
