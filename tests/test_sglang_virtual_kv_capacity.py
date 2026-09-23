# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import math
import sys
import types
from typing import Any, Optional

from kvcached.integration.sglang.patches import (
    SGLangLegacyVirtualKVCapacityPatch,
    SGLangVirtualKVCapacityPatch,
)


def _make_kv_cache_configurator_module():
    class FakeKVCacheConfigurator:
        device = "cuda"
        gpu_id = 0
        server_args = types.SimpleNamespace(mem_fraction_static=0.75)
        mambaish_config = None
        post_capture_kv_active = False

        def _profile_available_bytes(self, pre_model_load_memory: int) -> int:
            return pre_model_load_memory

    module: Any = types.ModuleType("sglang.srt.mem_cache.kv_cache_configurator")
    module.KVCacheConfigurator = FakeKVCacheConfigurator
    return module


def _make_model_runner_module():
    class FakeModelRunner:
        device = "cuda"
        gpu_id = 0
        mem_fraction_static = 0.75
        mambaish_config = None

        def _profile_available_bytes(self, pre_model_load_memory: int) -> int:
            return pre_model_load_memory

    module: Any = types.ModuleType("sglang.srt.model_executor.model_runner")
    module.ModelRunner = FakeModelRunner
    return module


def _install_fake_torch(
    monkeypatch,
    total_memory: int,
    reserved_memory: int = 0,
    *,
    device_capacity_mb: Optional[float] = None,
    world_size: int = 1,
    reduced_capacity: Optional[int] = None,
):
    monkeypatch.setenv("ENABLE_KVCACHED", "true")

    class FakeTensor:
        def __init__(self, value):
            self.value = value

        def item(self):
            return self.value

    def all_reduce(tensor, *, op, group):
        assert op == "min"
        assert group == "cpu-group"
        if reduced_capacity is not None:
            tensor.value = min(tensor.value, reduced_capacity)

    torch: Any = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(
        get_device_properties=lambda _device: types.SimpleNamespace(
            total_memory=total_memory
        ),
        memory_reserved=lambda _device: reserved_memory,
    )
    torch.int64 = "int64"
    torch.tensor = lambda value, dtype: FakeTensor(value)
    torch.distributed = types.SimpleNamespace(
        ReduceOp=types.SimpleNamespace(MIN="min"),
        all_reduce=all_reduce,
    )
    monkeypatch.setitem(sys.modules, "torch", torch)

    world_group = types.SimpleNamespace(
        world_size=world_size,
        cpu_group="cpu-group",
    )
    if device_capacity_mb is None:
        device_capacity_mb = total_memory / 1024**2
    modules: dict[str, dict[str, Any]] = {
        "sglang": {},
        "sglang.srt": {},
        "sglang.srt.distributed": {},
        "sglang.srt.distributed.parallel_state": {
            "get_world_group": lambda: world_group,
        },
        "sglang.srt.utils": {},
        "sglang.srt.utils.common": {
            "get_device_memory_capacity": lambda _device: device_capacity_mb,
        },
    }
    for name, attributes in modules.items():
        module = types.ModuleType(name)
        for key, value in attributes.items():
            setattr(module, key, value)
        monkeypatch.setitem(sys.modules, name, module)


def test_virtual_capacity_deducts_process_local_reserved_memory(
    monkeypatch,
):
    total_memory = 16 * 1024**3
    reserved_memory = 3 * 1024**3
    _install_fake_torch(monkeypatch, total_memory, reserved_memory)
    module = _make_kv_cache_configurator_module()

    patch = SGLangVirtualKVCapacityPatch()
    assert patch.patch_profile_available_bytes(module) is True

    configurator = module.KVCacheConfigurator()
    expected = (
        math.ceil(total_memory * configurator.server_args.mem_fraction_static)
        - reserved_memory
    )
    assert configurator._profile_available_bytes(14) == expected
    assert configurator._profile_available_bytes(7) == expected


def test_virtual_capacity_preserves_mamba_reservation(monkeypatch):
    total_memory = 16 * 1024**3
    reserved_memory = 2 * 1024**3
    _install_fake_torch(monkeypatch, total_memory, reserved_memory)
    module = _make_kv_cache_configurator_module()

    patch = SGLangVirtualKVCapacityPatch()
    assert patch.patch_profile_available_bytes(module) is True

    configurator = module.KVCacheConfigurator()
    configurator.mambaish_config = object()
    configurator._handle_max_mamba_cache = lambda capacity_gib: capacity_gib - 1

    expected = (
        math.ceil(total_memory * configurator.server_args.mem_fraction_static)
        - reserved_memory
        - 1024**3
    )
    assert configurator._profile_available_bytes(3) == expected


def test_virtual_capacity_preserves_post_capture_mamba_reservation(monkeypatch):
    """Post-capture sizing retains the larger SGLang Mamba safety margin."""
    total_memory = 80 * 1024**3
    reserved_memory = 10 * 1024**3
    device_capacity_mb = 40 * 1024
    mamba_reserve_mb = 1024
    _install_fake_torch(
        monkeypatch,
        total_memory,
        reserved_memory,
        device_capacity_mb=device_capacity_mb,
    )
    module = _make_kv_cache_configurator_module()

    patch = SGLangVirtualKVCapacityPatch()
    assert patch.patch_profile_available_bytes(module) is True

    configurator = module.KVCacheConfigurator()
    configurator.server_args.mem_fraction_static = 0.99

    def mamba_pre_capture_reserve_mb(gpu_mem):
        assert gpu_mem == device_capacity_mb
        return mamba_reserve_mb

    configurator.server_args.mamba_pre_capture_reserve_mb = mamba_pre_capture_reserve_mb
    configurator.mambaish_config = object()
    configurator.post_capture_kv_active = True
    configurator._handle_max_mamba_cache = lambda capacity_gib: capacity_gib

    expected = total_memory - reserved_memory - mamba_reserve_mb * 1024**2
    assert configurator._profile_available_bytes(3) == expected

    configurator.server_args.mem_fraction_static = 0.75
    expected = math.ceil(total_memory * 0.75) - reserved_memory
    assert configurator._profile_available_bytes(3) == expected


def test_virtual_capacity_uses_world_group_minimum(monkeypatch):
    total_memory = 16 * 1024**3
    reserved_memory = 2 * 1024**3
    peer_capacity = 7 * 1024**3
    _install_fake_torch(
        monkeypatch,
        total_memory,
        reserved_memory,
        world_size=2,
        reduced_capacity=peer_capacity,
    )
    module = _make_kv_cache_configurator_module()

    patch = SGLangVirtualKVCapacityPatch()
    assert patch.patch_profile_available_bytes(module) is True

    configurator = module.KVCacheConfigurator()
    assert configurator._profile_available_bytes(3) == peer_capacity


def test_virtual_capacity_falls_back_when_peer_query_fails(monkeypatch):
    total_memory = 16 * 1024**3
    _install_fake_torch(
        monkeypatch,
        total_memory,
        world_size=2,
        reduced_capacity=-(1 << 63),
    )
    module = _make_kv_cache_configurator_module()

    patch = SGLangVirtualKVCapacityPatch()
    assert patch.patch_profile_available_bytes(module) is True

    configurator = module.KVCacheConfigurator()
    assert configurator._profile_available_bytes(1234) == 1234


def test_virtual_capacity_falls_back_when_reserved_memory_query_fails(
    monkeypatch,
):
    total_memory = 16 * 1024**3
    _install_fake_torch(monkeypatch, total_memory)
    torch = sys.modules["torch"]

    def _raise_memory_query_error(_device):
        raise RuntimeError("memory query failed")

    torch.cuda.memory_reserved = _raise_memory_query_error
    module = _make_kv_cache_configurator_module()

    patch = SGLangVirtualKVCapacityPatch()
    assert patch.patch_profile_available_bytes(module) is True

    configurator = module.KVCacheConfigurator()
    assert configurator._profile_available_bytes(1234) == 1234


def test_virtual_capacity_falls_back_for_non_gpu_device(monkeypatch):
    _install_fake_torch(monkeypatch, 16 * 1024**3)
    module = _make_kv_cache_configurator_module()

    patch = SGLangVirtualKVCapacityPatch()
    assert patch.patch_profile_available_bytes(module) is True

    configurator = module.KVCacheConfigurator()
    configurator.device = "cpu"
    assert configurator._profile_available_bytes(1234) == 1234


def test_virtual_capacity_patch_is_idempotent(monkeypatch):
    _install_fake_torch(monkeypatch, 16 * 1024**3)
    module = _make_kv_cache_configurator_module()

    patch = SGLangVirtualKVCapacityPatch()
    assert patch.patch_profile_available_bytes(module) is True
    first = module.KVCacheConfigurator._profile_available_bytes
    assert patch.patch_profile_available_bytes(module) is True
    assert module.KVCacheConfigurator._profile_available_bytes is first


def test_legacy_virtual_capacity_still_patches_model_runner(monkeypatch):
    """SGLang 0.5.11-0.5.15 still profiles capacity on ModelRunner."""
    total_memory = 16 * 1024**3
    reserved_memory = 3 * 1024**3
    _install_fake_torch(monkeypatch, total_memory, reserved_memory)
    module = _make_model_runner_module()

    patch = SGLangLegacyVirtualKVCapacityPatch()
    assert patch.patch_profile_available_bytes(module) is True

    runner = module.ModelRunner()
    expected = math.ceil(total_memory * runner.mem_fraction_static) - reserved_memory
    assert runner._profile_available_bytes(14) == expected
