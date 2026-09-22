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
    KVCachePoolSnapshot,
    RuntimeSnapshot,
    build_kv_cache_pool_snapshot,
    build_runtime_snapshot,
    get_capabilities,
    get_registered_kv_cache_pool_snapshot_dicts,
)
from kvcached.pool_registry import (  # noqa: E402
    clear_registered_kv_cache_pools,
    register_kv_cache_pool,
)
from kvcached.utils import PAGE_SIZE  # noqa: E402


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
    setattr(tp_ipc_module, "resolve_gpu_device_index", lambda device: 0)
    setattr(tp_ipc_module, "start_worker_listener_thread", lambda *args: None)
    setattr(tp_ipc_module, "stop_worker_listener_threads", lambda: True)

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
    setattr(tp_ipc_module, "resolve_gpu_device_index", lambda device: 0)
    setattr(tp_ipc_module, "start_worker_listener_thread", lambda *args: None)
    setattr(tp_ipc_module, "stop_worker_listener_threads", lambda: True)

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


def test_capabilities_report_planned_surfaces_as_unsupported():
    """Unlanded surfaces are reported False, never omitted.

    A consumer writes the detection once against a build that predates the
    surface; the same code starts returning True when it ships.
    """
    features = get_capabilities()["features"]

    assert features["operation_counters"] is False
    assert features["runtime_reservation_reporting"] is False
    # Landed in #414: the one write path on the surface.
    assert features["instance_memory_limit"] is True


def test_capabilities_expose_backend_and_integration_records():
    capabilities = get_capabilities()

    backends = capabilities["backends"]
    assert backends["kv_pooling"] is True
    assert backends["elastic_capacity"] is True
    # kvcached accounts for non-KV memory but never manages it.
    assert backends["non_kv_memory_management"] is False
    # Named "default_" because it is this process's import-time env value, not
    # a live engine's page size; consumers needing the runtime value read
    # KVCachePoolSnapshot.page_size_bytes.
    assert backends["default_page_size_bytes"] == PAGE_SIZE
    assert "page_size_bytes" not in backends

    integrations = capabilities["integrations"]
    assert set(integrations) == {"vllm", "sglang"}
    for entry in integrations.values():
        assert "MHA" in entry["attention_types"]
        assert "MLA" in entry["attention_types"]
        assert entry["kv_layouts"] == ["NHD"]

    # A real, code-level distinction between the two shims: only the vLLM
    # integration accepts HYBRID_LINEAR through alloc_kv_cache(); SGLang
    # allocates mamba state through a separate entry point.
    assert "HYBRID_LINEAR" in integrations["vllm"]["attention_types"]
    assert "HYBRID_LINEAR" not in integrations["sglang"]["attention_types"]


def test_hybrid_linear_pooling_mode_distinguishes_the_two_shapes():
    """The bool alone cannot answer "is that state in the pool snapshot?".

    Both shims report hybrid_linear_state_pooling True, but vLLM carves the
    state out of the KV pool (so it shows up in KVCachePoolSnapshot) while
    SGLang allocates it separately (so it does not). Consumers branch on the
    mode rather than parsing comments.
    """
    integrations = get_capabilities()["integrations"]

    for entry in integrations.values():
        assert entry["hybrid_linear_state_pooling"] is True

    assert integrations["vllm"]["hybrid_linear_state_pooling_mode"] == "unified_pool"
    assert (
        integrations["sglang"]["hybrid_linear_state_pooling_mode"]
        == "separate_allocation"
    )


def test_operation_counter_names_stay_coupled_to_their_feature_flag():
    """One flip when #410 lands, not two that can drift apart."""
    capabilities = get_capabilities()

    flag = capabilities["features"]["operation_counters"]
    names = capabilities["operation_counter_names"]

    assert bool(names) == flag


def test_capabilities_enumerate_snapshot_fields_for_feature_detection():
    """Field lists must match the dataclasses consumers actually receive."""
    capabilities = get_capabilities()

    pool_fields = capabilities["pool_snapshot_fields"]
    runtime_fields = capabilities["runtime_snapshot_fields"]

    assert pool_fields == list(KVCachePoolSnapshot.__dataclass_fields__.keys())
    assert runtime_fields == list(RuntimeSnapshot.__dataclass_fields__.keys())

    snapshot = build_kv_cache_pool_snapshot(FakeManager(), integration="vllm")
    assert set(snapshot.to_dict()) == set(pool_fields)

    # No counters until operation observability lands.
    assert capabilities["operation_counter_names"] == []


def test_capabilities_record_is_json_serializable_and_stable():
    """The whole record must survive an exporter round-trip unchanged."""
    capabilities = get_capabilities()

    assert json.loads(json.dumps(capabilities)) == capabilities
    assert get_capabilities() == capabilities


def test_capabilities_need_no_private_field_access():
    """A consumer reads the record through public keys only.

    Guards the contract in #375: integrations must not have to reach into
    allocator internals or applied-patch attributes to learn what is supported.
    """
    capabilities = get_capabilities()

    def assert_public(node):
        if isinstance(node, dict):
            for key, value in node.items():
                assert not key.startswith("_"), f"private key exposed: {key}"
                assert_public(value)
        elif isinstance(node, list):
            for item in node:
                assert_public(item)

    assert_public(capabilities)
    for field_name in capabilities["pool_snapshot_fields"]:
        assert not field_name.startswith("_")



def _load_shim_under_stubs(engine, monkeypatch):
    """Import an engine shim with its heavy deps stubbed out.

    Same approach as the factory tests above, reduced to what module import
    needs: the shim's module-level constants, not a working engine.
    """
    torch = types.ModuleType("torch")
    setattr(torch, "dtype", object)
    setattr(torch, "Tensor", object)
    setattr(torch, "cuda", types.SimpleNamespace(current_device=lambda: 0))

    manager_module = types.ModuleType("kvcached.kv_cache_manager")
    setattr(manager_module, "KVCacheManager", object)

    tp_ipc_module = types.ModuleType("kvcached.tp_ipc_util")
    setattr(tp_ipc_module, "resolve_gpu_device_index", lambda device: 0)
    setattr(tp_ipc_module, "start_worker_listener_thread", lambda *args: None)
    setattr(tp_ipc_module, "stop_worker_listener_threads", lambda: True)

    vmm_ops_module = types.ModuleType("kvcached.vmm_ops")
    setattr(vmm_ops_module, "create_kv_tensors", lambda *args, **kwargs: [])
    setattr(vmm_ops_module, "init_kvcached", lambda *args, **kwargs: None)
    setattr(vmm_ops_module, "shutdown_kvcached", lambda: None)

    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "kvcached.kv_cache_manager", manager_module)
    monkeypatch.setitem(sys.modules, "kvcached.tp_ipc_util", tp_ipc_module)
    monkeypatch.setitem(sys.modules, "kvcached.vmm_ops", vmm_ops_module)

    module_path = (
        Path(__file__).parents[1] / "kvcached" / "integration" / engine / "interfaces.py"
    )
    spec = importlib.util.spec_from_file_location(
        f"_test_{engine}_interfaces_constants", module_path
    )
    assert spec is not None and spec.loader is not None
    shim = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shim)
    return shim


def test_reported_attention_types_match_the_shim_guards(monkeypatch):
    """The record must not drift from the guards it claims to describe.

    Both are derived from the shim's SUPPORTED_* constants, so adding an
    attention type to a shim without updating the other side fails here
    instead of silently shipping a stale record.
    """
    integrations = get_capabilities()["integrations"]

    for engine in ("vllm", "sglang"):
        shim = _load_shim_under_stubs(engine, monkeypatch)
        assert integrations[engine]["attention_types"] == list(
            shim.SUPPORTED_ATTENTION_TYPES
        )
        assert integrations[engine]["kv_layouts"] == list(shim.SUPPORTED_KV_LAYOUTS)
