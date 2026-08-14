# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import math
import sys
import types
from typing import Any, Optional

from kvcached.integration.sglang.patches import (
    SGLangVirtualKVCapacityPatch,
)


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
    modules = {
        "sglang": {},
        "sglang.srt": {},
        "sglang.srt.distributed": {},
        "sglang.srt.distributed.parallel_state": {
            "get_world_group": lambda: world_group,
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
    module = _make_model_runner_module()

    patch = SGLangVirtualKVCapacityPatch()
    assert patch.patch_profile_available_bytes(module) is True

    runner = module.ModelRunner()
    expected = (
        math.ceil(total_memory * runner.mem_fraction_static) - reserved_memory
    )
    assert runner._profile_available_bytes(14) == expected
    assert runner._profile_available_bytes(7) == expected


def test_virtual_capacity_preserves_mamba_reservation(monkeypatch):
    total_memory = 16 * 1024**3
    reserved_memory = 2 * 1024**3
    _install_fake_torch(monkeypatch, total_memory, reserved_memory)
    module = _make_model_runner_module()

    patch = SGLangVirtualKVCapacityPatch()
    assert patch.patch_profile_available_bytes(module) is True

    runner = module.ModelRunner()
    runner.mambaish_config = object()
    runner.handle_max_mamba_cache = lambda capacity_gib: capacity_gib - 1

    expected = (
        math.ceil(total_memory * runner.mem_fraction_static)
        - reserved_memory
        - 1024**3
    )
    assert runner._profile_available_bytes(3) == expected


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
    module = _make_model_runner_module()

    patch = SGLangVirtualKVCapacityPatch()
    assert patch.patch_profile_available_bytes(module) is True

    runner = module.ModelRunner()
    assert runner._profile_available_bytes(3) == peer_capacity


def test_virtual_capacity_falls_back_when_peer_query_fails(monkeypatch):
    total_memory = 16 * 1024**3
    _install_fake_torch(
        monkeypatch,
        total_memory,
        world_size=2,
        reduced_capacity=-(1 << 63),
    )
    module = _make_model_runner_module()

    patch = SGLangVirtualKVCapacityPatch()
    assert patch.patch_profile_available_bytes(module) is True

    runner = module.ModelRunner()
    assert runner._profile_available_bytes(1234) == 1234


def test_virtual_capacity_falls_back_when_reserved_memory_query_fails(
    monkeypatch,
):
    total_memory = 16 * 1024**3
    _install_fake_torch(monkeypatch, total_memory)
    torch = sys.modules["torch"]

    def _raise_memory_query_error(_device):
        raise RuntimeError("memory query failed")

    torch.cuda.memory_reserved = _raise_memory_query_error
    module = _make_model_runner_module()

    patch = SGLangVirtualKVCapacityPatch()
    assert patch.patch_profile_available_bytes(module) is True

    runner = module.ModelRunner()
    assert runner._profile_available_bytes(1234) == 1234


def test_virtual_capacity_falls_back_for_non_gpu_device(monkeypatch):
    _install_fake_torch(monkeypatch, 16 * 1024**3)
    module = _make_model_runner_module()

    patch = SGLangVirtualKVCapacityPatch()
    assert patch.patch_profile_available_bytes(module) is True

    runner = module.ModelRunner()
    runner.device = "cpu"
    assert runner._profile_available_bytes(1234) == 1234


def test_virtual_capacity_patch_is_idempotent(monkeypatch):
    _install_fake_torch(monkeypatch, 16 * 1024**3)
    module = _make_model_runner_module()

    patch = SGLangVirtualKVCapacityPatch()
    assert patch.patch_profile_available_bytes(module) is True
    first = module.ModelRunner._profile_available_bytes
    assert patch.patch_profile_available_bytes(module) is True
    assert module.ModelRunner._profile_available_bytes is first
