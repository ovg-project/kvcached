# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import importlib
import sys
import types
from typing import Any
from unittest import mock

import pytest


@pytest.fixture(autouse=True)
def _remove_cached_sglang_submodules():
    yield
    package = sys.modules.get("kvcached.integration.sglang")
    if package is not None:
        package.__dict__.pop("interfaces", None)
        package.__dict__.pop("patches", None)


class FakeTensor:
    def __init__(self, nbytes=0):
        self.nbytes = nbytes
        self.views = []

    def view(self, *args, **kwargs):
        self.views.append((args, kwargs))
        return self

    def __getitem__(self, _key):
        return self


def _load_sglang_modules(monkeypatch):
    torch_mock = mock.MagicMock()
    torch_mock.__version__ = "2.6.0"
    torch_mock.cuda.is_available.return_value = True
    torch_mock.cuda.current_device.return_value = 0
    torch_mock.cuda.get_device_properties.return_value = types.SimpleNamespace(
        total_memory=64 * 1024 * 1024
    )
    monkeypatch.setitem(sys.modules, "torch", torch_mock)
    monkeypatch.setitem(sys.modules, "torch.cuda", torch_mock.cuda)
    monkeypatch.setitem(sys.modules, "posix_ipc", mock.MagicMock())
    monkeypatch.setitem(sys.modules, "kvcached.vmm_ops", mock.MagicMock())

    from kvcached.integration.sglang import interfaces, patches

    importlib.reload(interfaces)
    importlib.reload(patches)
    return interfaces, patches, torch_mock


def test_dsv4_runtime_reservation_patch_registers_per_pool_breakdown(monkeypatch):
    interfaces, patches, _torch_mock = _load_sglang_modules(monkeypatch)
    monkeypatch.setenv("KVCACHED_SGLANG_DSV4_RUNTIME_RESERVATION", "1")
    fake_mod: Any = types.SimpleNamespace()
    registrations: list[tuple[str, str, int]] = []

    class DeepSeekV4TokenToKVPool:
        def __init__(self):
            self.device = "cuda:0"
            self.swa_kv_pool = types.SimpleNamespace(
                kv_buffer=[FakeTensor(11), FakeTensor(13)]
            )
            self.c4_kv_pool = types.SimpleNamespace(kv_buffer=[FakeTensor(17)])
            self.c128_kv_pool = types.SimpleNamespace(kv_buffer=[FakeTensor(19)])
            self.c4_indexer_kv_pool = types.SimpleNamespace(
                index_k_with_scale_buffer=[FakeTensor(23)]
            )
            self.compress_state_pools = [
                types.SimpleNamespace(
                    kv_score_buffer=types.SimpleNamespace(kv_score=FakeTensor(29))
                ),
                None,
            ]
            self.indexer_compress_state_pools = [
                types.SimpleNamespace(
                    kv_score_buffer=types.SimpleNamespace(kv_score=FakeTensor(31))
                )
            ]

    fake_mod.DeepSeekV4TokenToKVPool = DeepSeekV4TokenToKVPool
    monkeypatch.setattr(
        patches.DeepSeekV4RuntimeReservationPatch,
        "initialize_version_info",
        lambda self: True,
    )
    monkeypatch.setattr(
        interfaces,
        "register_runtime_owned_reservation",
        lambda device, pool, num_bytes: registrations.append((device, pool, num_bytes)),
    )

    patch = patches.DeepSeekV4RuntimeReservationPatch()
    assert patch.apply(fake_mod)

    pool: Any = fake_mod.DeepSeekV4TokenToKVPool()

    assert registrations == [
        ("cuda:0", "dsv4.swa_kv_pool", 24),
        ("cuda:0", "dsv4.c4_kv_pool", 17),
        ("cuda:0", "dsv4.c128_kv_pool", 19),
        ("cuda:0", "dsv4.c4_indexer_kv_pool", 23),
        ("cuda:0", "dsv4.compress_state_pools", 29),
        ("cuda:0", "dsv4.indexer_compress_state_pools", 31),
    ]
    assert pool._kvcached_runtime_reservation_breakdown == {
        "dsv4.swa_kv_pool": 24,
        "dsv4.c4_kv_pool": 17,
        "dsv4.c128_kv_pool": 19,
        "dsv4.c4_indexer_kv_pool": 23,
        "dsv4.compress_state_pools": 29,
        "dsv4.indexer_compress_state_pools": 31,
    }


def test_dsv4_runtime_reservation_patch_is_opt_in(monkeypatch):
    _interfaces, patches, _torch_mock = _load_sglang_modules(monkeypatch)
    fake_mod: Any = types.SimpleNamespace()
    calls: list[str] = []

    class DeepSeekV4TokenToKVPool:
        def __init__(self):
            calls.append("original")

    fake_mod.DeepSeekV4TokenToKVPool = DeepSeekV4TokenToKVPool
    monkeypatch.setattr(
        patches.DeepSeekV4RuntimeReservationPatch,
        "initialize_version_info",
        lambda self: (_ for _ in ()).throw(
            AssertionError("version check should not run when env is unset")
        ),
    )

    patch = patches.DeepSeekV4RuntimeReservationPatch()
    assert patch.apply(fake_mod)

    pool = fake_mod.DeepSeekV4TokenToKVPool()
    assert calls == ["original"]
    assert not hasattr(pool, "_kvcached_runtime_reservation_breakdown")


def test_sglang_alloc_kv_cache_subtracts_runtime_owned_reservations(monkeypatch):
    interfaces, _patches, torch_mock = _load_sglang_modules(monkeypatch)
    page_size = interfaces.PAGE_SIZE
    captured: dict[str, Any] = {}

    torch_mock.cuda.get_device_properties.return_value = types.SimpleNamespace(
        total_memory=16 * page_size
    )

    def fake_create_kv_tensors(mem_size, *args, **kwargs):
        captured["mem_size"] = mem_size
        return [FakeTensor()]

    monkeypatch.setattr(interfaces, "create_kv_tensors", fake_create_kv_tensors)
    monkeypatch.setattr(interfaces, "_kvcached_initialized", True)
    monkeypatch.setattr(interfaces, "_contiguous_layout", False)
    interfaces._runtime_owned_reservations.clear()
    interfaces.register_runtime_owned_reservation("cuda:0", "dsv4.swa_kv_pool", 3 * page_size)

    interfaces.alloc_kv_cache(
        kvcache_shape=(1, 1, 1),
        dtype=types.SimpleNamespace(itemsize=1),
        device="cuda:0",
        num_layers=1,
        page_size=1,
        attention_type="MLA",
    )

    # MLA allocations are aligned to 2 * PAGE_SIZE.  (16P - 3P) rounds down to 12P.
    assert captured["mem_size"] == 12 * page_size


def test_sglang_shutdown_clears_runtime_owned_reservations(monkeypatch):
    interfaces, _patches, _torch_mock = _load_sglang_modules(monkeypatch)
    interfaces.register_runtime_owned_reservation(
        "cuda:0", "dsv4.swa_kv_pool", 4096
    )

    assert interfaces.get_runtime_owned_reservation_bytes("cuda:0") == 4096

    interfaces.shutdown_kvcached()

    assert interfaces.get_runtime_owned_reservation_bytes("cuda:0") == 0
