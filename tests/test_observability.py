# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import gc
import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import Any

if "torch" not in sys.modules and importlib.util.find_spec("torch") is None:
    sys.modules.setdefault("torch", types.ModuleType("torch"))

from kvcached.observability import (  # noqa: E402
    build_kv_cache_pool_snapshot,
    build_runtime_snapshot,
    get_capabilities,
    get_registered_kv_cache_pool_snapshot_dicts,
)
from kvcached.pool_registry import (  # noqa: E402
    clear_registered_kv_cache_pools,
    register_kv_cache_pool,
)


class FakePageAllocator:
    page_state_calls = 0

    def get_page_state(self):
        self.page_state_calls += 1
        return {
            "total_pages": 20,
            "free_pages": 10,
            "inuse_pages": 10,
            "reserved_pages": 2,
        }

    def get_num_free_pages(self):
        return 10

    def get_num_inuse_pages(self):
        return 6

    def get_num_total_pages(self):
        return 20

    def get_num_reserved_pages(self):
        return 2

    def get_avail_physical_pages(self):
        return 4

    def get_resize_target(self):
        return 0


class FakeManager:
    num_blocks = 128
    block_mem_size = 4096
    num_layers = 8
    num_kv_buffers = 2
    group_id = 3
    pool_name = "full_attention"
    page_size = 2 * 1024 * 1024
    mem_size = num_blocks * block_mem_size
    reserved_blocks = [0, 7]
    null_block = [0]
    in_shrink = False
    target_num_blocks = None
    page_allocator = FakePageAllocator()

    def available_size(self):
        return 64

    def _get_num_alloced_blocks(self):
        return 16

    def get_mapped_memory_size(self, unit="bytes"):
        assert unit == "bytes"
        return 6 * self.num_layers * self.page_size * self.num_kv_buffers


def test_capabilities_are_json_serializable():
    capabilities = get_capabilities()

    assert capabilities["schema_version"] == "kvcached.observability.v1"
    assert capabilities["features"]["read_only"] is True
    assert capabilities["features"]["policy_control"] is False
    json.dumps(capabilities)


def test_runtime_snapshot_dict():
    snapshot = build_runtime_snapshot(
        engine="vllm",
        initialized=True,
        device="cuda:0",
        world_size=2,
        pp_rank=1,
        async_sched=True,
        contiguous_layout=False,
        is_worker=True,
    )

    assert snapshot.to_dict() == {
        "schema_version": "kvcached.observability.v1",
        "engine": "vllm",
        "initialized": True,
        "device": "cuda:0",
        "world_size": 2,
        "pp_rank": 1,
        "async_sched": True,
        "contiguous_layout": False,
        "is_worker": True,
    }


def test_kv_cache_pool_snapshot_from_manager_like_object():
    FakeManager.page_allocator.page_state_calls = 0
    snapshot = build_kv_cache_pool_snapshot(
        FakeManager(),
        integration="sglang",
    )
    data = snapshot.to_dict()

    assert data["pool_type"] == "kv_cache"
    assert data["integration"] == "sglang"
    assert data["pool_name"] == "full_attention"
    assert data["group_id"] == 3
    assert data["available_blocks"] == 64
    assert data["allocated_blocks"] == 16
    assert data["reserved_blocks"] == 2
    bytes_per_block = 4096 * 8 * 2
    assert data["available_bytes"] == 64 * bytes_per_block
    assert data["allocated_bytes"] == 16 * bytes_per_block
    assert data["reserved_bytes"] == 2 * bytes_per_block
    assert data["null_block_reserved"] is True
    assert data["virtual_per_layer_bytes"] == 128 * 4096 * 2
    assert data["virtual_total_bytes"] == 128 * 4096 * 8 * 2
    assert data["mapped_bytes"] == 10 * 8 * (2 * 1024 * 1024) * 2
    assert data["total_pages"] == 20
    assert data["free_pages"] == 10
    assert data["inuse_pages"] == 10
    assert data["reserved_pages"] == 2
    assert data["available_physical_pages"] == 4
    assert data["effective_free_pages"] == 6
    assert data["resize_target_bytes"] == 0
    assert FakeManager.page_allocator.page_state_calls == 1
    json.dumps(data)


def test_pool_snapshot_falls_back_for_older_page_allocator():
    class LegacyPageAllocator:
        def get_num_free_pages(self):
            return 7

        def get_num_inuse_pages(self):
            return 5

        def get_num_total_pages(self):
            return 12

        def get_num_reserved_pages(self):
            return 1

        def get_avail_physical_pages(self):
            return 3

        def get_resize_target(self):
            return -1

    class LegacyManager(FakeManager):
        page_allocator: Any = LegacyPageAllocator()

    data = build_kv_cache_pool_snapshot(LegacyManager()).to_dict()

    assert data["total_pages"] == 12
    assert data["free_pages"] == 7
    assert data["inuse_pages"] == 5
    assert data["reserved_pages"] == 1


def test_pool_snapshot_clamps_negative_block_gauges():
    class NegativeBlockManager(FakeManager):
        def available_size(self):
            return -127

        def _get_num_alloced_blocks(self):
            return -1

    data = build_kv_cache_pool_snapshot(NegativeBlockManager()).to_dict()

    assert data["available_blocks"] == 0
    assert data["available_bytes"] == 0
    assert data["allocated_blocks"] == 0
    assert data["allocated_bytes"] == 0


def test_registered_pool_snapshot_uses_manager_snapshot_entrypoint():
    clear_registered_kv_cache_pools()

    class SynchronizedManager(FakeManager):
        snapshot_calls = 0

        def observability_snapshot(self, *, integration=None):
            self.snapshot_calls += 1
            return build_kv_cache_pool_snapshot(self, integration=integration)

    manager = SynchronizedManager()
    register_kv_cache_pool(manager, integration="vllm")

    snapshots = get_registered_kv_cache_pool_snapshot_dicts(integration="vllm")

    assert len(snapshots) == 1
    assert manager.snapshot_calls == 1
    clear_registered_kv_cache_pools()


def test_registered_pool_snapshots_are_filtered_and_do_not_keep_managers_alive():
    clear_registered_kv_cache_pools()
    sglang_manager = FakeManager()
    other_manager = FakeManager()
    sglang_manager.pool_name = "mha"
    other_manager.pool_name = "unified"
    register_kv_cache_pool(
        sglang_manager,
        integration="sglang",
    )
    register_kv_cache_pool(
        other_manager,
        integration="vllm",
    )

    snapshots = get_registered_kv_cache_pool_snapshot_dicts(integration="sglang")

    assert len(snapshots) == 1
    assert snapshots[0]["integration"] == "sglang"
    assert snapshots[0]["pool_name"] == "mha"

    del sglang_manager
    gc.collect()
    assert get_registered_kv_cache_pool_snapshot_dicts(integration="sglang") == []
    assert len(get_registered_kv_cache_pool_snapshot_dicts(integration="vllm")) == 1
    clear_registered_kv_cache_pools()


def test_sglang_manager_factory_registers_and_shutdown_clears_pool(monkeypatch):
    clear_registered_kv_cache_pools()

    torch = types.ModuleType("torch")
    setattr(torch, "dtype", object)
    setattr(torch, "Tensor", object)
    setattr(torch, "cuda", types.SimpleNamespace(current_device=lambda: 0))

    manager_module = types.ModuleType("kvcached.kv_cache_manager")

    class FakeKVCacheManager(FakeManager):
        def __init__(self, num_blocks, block_size, cell_size, num_layers, **kwargs):
            self.num_blocks = num_blocks
            self.block_mem_size = block_size * cell_size
            self.num_layers = num_layers
            self.num_kv_buffers = kwargs["num_kv_buffers"]
            self.group_id = kwargs["group_id"]
            self.pool_name = kwargs["pool_name"]
            self.mem_size = num_blocks * self.block_mem_size
            self.reserved_blocks = []
            self.page_allocator = FakePageAllocator()

    setattr(manager_module, "KVCacheManager", FakeKVCacheManager)

    tp_ipc_module = types.ModuleType("kvcached.tp_ipc_util")
    setattr(tp_ipc_module, "start_worker_listener_thread", lambda *args: None)

    utils_module = types.ModuleType("kvcached.utils")
    setattr(utils_module, "CONTIGUOUS_LAYOUT", False)
    setattr(utils_module, "PAGE_SIZE", 2 * 1024 * 1024)
    setattr(utils_module, "get_kvcached_logger", lambda: types.SimpleNamespace())
    setattr(utils_module, "normalize_gpu_device", lambda device: device)

    vmm_ops_module = types.ModuleType("kvcached.vmm_ops")
    setattr(vmm_ops_module, "create_kv_tensors", lambda *args, **kwargs: [])
    setattr(vmm_ops_module, "init_kvcached", lambda *args, **kwargs: None)
    setattr(vmm_ops_module, "shutdown_kvcached", lambda: None)

    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "kvcached.kv_cache_manager", manager_module)
    monkeypatch.setitem(sys.modules, "kvcached.tp_ipc_util", tp_ipc_module)
    monkeypatch.setitem(sys.modules, "kvcached.utils", utils_module)
    monkeypatch.setitem(sys.modules, "kvcached.vmm_ops", vmm_ops_module)

    module_path = (
        Path(__file__).parents[1]
        / "kvcached"
        / "integration"
        / "sglang"
        / "interfaces.py"
    )
    spec = importlib.util.spec_from_file_location("_test_sglang_interfaces", module_path)
    assert spec is not None and spec.loader is not None
    interfaces = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(interfaces)
    setattr(interfaces, "_kvcached_initialized", True)

    manager = interfaces.get_kv_cache_manager(
        128,
        16,
        256,
        8,
        group_id=4,
        pool_name="mha",
    )
    snapshots = interfaces.kv_cache_pool_snapshot_dicts()

    assert manager.group_id == 4
    assert manager.pool_name == "mha"
    assert len(snapshots) == 1
    assert snapshots[0]["integration"] == "sglang"
    assert snapshots[0]["pool_name"] == "mha"
    assert snapshots[0]["group_id"] == 4

    interfaces.shutdown_kvcached()
    assert interfaces.kv_cache_pool_snapshot_dicts() == []
def test_vllm_manager_factory_registers_and_shutdown_clears_pool(monkeypatch):
    clear_registered_kv_cache_pools()

    torch = types.ModuleType("torch")
    setattr(torch, "dtype", object)
    setattr(torch, "Tensor", object)
    setattr(torch, "cuda", types.SimpleNamespace(current_device=lambda: 0))

    manager_module = types.ModuleType("kvcached.kv_cache_manager")

    class FakeKVCacheManager(FakeManager):
        def __init__(
            self,
            num_blocks,
            block_size,
            cell_size,
            num_layers,
            world_size,
            **kwargs,
        ):
            self.num_blocks = num_blocks
            self.block_mem_size = block_size * cell_size
            self.num_layers = num_layers
            self.num_kv_buffers = kwargs["num_kv_buffers"]
            self.group_id = kwargs["group_id"]
            self.pool_name = kwargs["pool_name"]
            self.mem_size = num_blocks * self.block_mem_size
            self.reserved_blocks = []
            self.page_allocator = FakePageAllocator()
            self.world_size = world_size

    setattr(manager_module, "KVCacheManager", FakeKVCacheManager)

    tp_ipc_module = types.ModuleType("kvcached.tp_ipc_util")
    setattr(tp_ipc_module, "start_worker_listener_thread", lambda *args: None)

    utils_module = types.ModuleType("kvcached.utils")
    setattr(utils_module, "CONTIGUOUS_LAYOUT", False)
    setattr(utils_module, "PAGE_SIZE", 2 * 1024 * 1024)
    setattr(utils_module, "get_kvcached_logger", lambda: types.SimpleNamespace())
    setattr(utils_module, "normalize_gpu_device", lambda device: device)

    vmm_ops_module = types.ModuleType("kvcached.vmm_ops")
    setattr(vmm_ops_module, "create_kv_tensors", lambda *args, **kwargs: [])
    setattr(vmm_ops_module, "init_kvcached", lambda *args, **kwargs: None)
    setattr(vmm_ops_module, "shutdown_kvcached", lambda: None)

    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "kvcached.kv_cache_manager", manager_module)
    monkeypatch.setitem(sys.modules, "kvcached.tp_ipc_util", tp_ipc_module)
    monkeypatch.setitem(sys.modules, "kvcached.utils", utils_module)
    monkeypatch.setitem(sys.modules, "kvcached.vmm_ops", vmm_ops_module)

    module_path = (
        Path(__file__).parents[1]
        / "kvcached"
        / "integration"
        / "vllm"
        / "interfaces.py"
    )
    spec = importlib.util.spec_from_file_location("_test_vllm_interfaces", module_path)
    assert spec is not None and spec.loader is not None
    interfaces = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(interfaces)
    setattr(interfaces, "_kvcached_initialized", True)

    manager = interfaces.get_kv_cache_manager(
        128,
        16,
        256,
        8,
        group_id=5,
        pool_name="unified",
    )
    snapshots = interfaces.kv_cache_pool_snapshot_dicts()

    assert manager.group_id == 5
    assert manager.pool_name == "unified"
    assert manager.world_size == 1
    assert len(snapshots) == 1
    assert snapshots[0]["integration"] == "vllm"
    assert snapshots[0]["pool_name"] == "unified"
    assert snapshots[0]["group_id"] == 5

    interfaces.shutdown_kvcached()
    assert interfaces.kv_cache_pool_snapshot_dicts() == []
