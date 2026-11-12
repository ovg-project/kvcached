# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import gc
import importlib.util
import json
import sys
import threading
import types
import weakref
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from kvcached import runtime_reservations as reports


class Owner:
    pass


@pytest.fixture(autouse=True)
def clean_reports():
    reports.clear_runtime_owned_reservations()
    yield
    reports.clear_runtime_owned_reservations()


def register(owner, size, *, engine="vllm", device="cuda:0", pool="workspace"):
    reports.register_runtime_owned_reservation(
        device, pool, size, owner=owner, integration=engine)


def total(engine="vllm", device="cuda:0"):
    return reports.get_runtime_owned_reservation_bytes(device, integration=engine)


def test_registration_replaces_one_owner_and_sums_distinct_owners():
    target, draft = Owner(), Owner()
    register(target, 100)
    register(draft, 30)
    register(target, 80)
    snapshot, = reports.get_runtime_reservation_snapshots()
    assert (snapshot.num_bytes, snapshot.owner_count) == (110, 2)
    register(target, 0)
    assert total() == 30
    assert reports.get_runtime_reservation_snapshots()[0].owner_count == 1


def test_owner_identity_does_not_depend_on_hash_or_equality():
    class EqualOwner:
        def __eq__(self, other):
            raise AssertionError("owner equality must not be called")

    first, second = EqualOwner(), EqualOwner()
    register(first, 9)
    register(second, 11)
    assert total() == 20
    register(first, 0)
    assert total() == 11


def test_reports_and_old_snapshots_do_not_retain_owners():
    owner = Owner()
    reference = weakref.ref(owner)
    register(owner, 42)
    old = reports.get_runtime_reservation_snapshots()
    del owner
    gc.collect()
    assert reference() is None
    assert reports.get_runtime_reservation_snapshots() == []
    assert old[0].num_bytes == 42
    with pytest.raises(FrozenInstanceError):
        setattr(old[0], "num_bytes", 1)


def test_cyclic_owner_collection_during_snapshot_does_not_invalidate_iteration():
    class CyclicOwner:
        def __init__(self):
            self.cycle = self

    was_enabled = gc.isenabled()
    thresholds = gc.get_threshold()
    gc.collect()
    gc.disable()
    try:
        # Leave owners alive only through unreachable cycles. Trigger a
        # collection while snapshot gathering allocates temporary tuples;
        # list(registry.items()) failed here on CPython 3.10 and 3.11.
        for _ in range(5000):
            owner = CyclicOwner()
            register(owner, 1)
        del owner
        gc.set_threshold(gc.get_count()[0] + 50, 1000000, 1000000)
        gc.enable()
        reports.get_runtime_reservation_snapshots()
    finally:
        gc.disable()
        gc.set_threshold(*thresholds)
        gc.collect()
        if was_enabled:
            gc.enable()
    assert reports.get_runtime_reservation_snapshots() == []


def test_device_aliases_and_engine_categories_are_isolated():
    owner = Owner()
    register(owner, 10, device="HIP:00")
    register(owner, 20, device="cuda:1")
    register(owner, 30, engine="sglang")
    register(owner, 40, pool="graph")
    assert total(device="hip:0") == 50
    assert total(device="cuda:1") == 20
    assert total(engine="sglang") == 30
    assert reports.get_runtime_owned_reservation_breakdown(
        "cuda:0", integration="vllm") == {"graph": 40, "workspace": 10}
    reports.clear_runtime_owned_reservations(integration="sglang")
    assert total(engine="sglang") == 0
    assert total() == 50


@pytest.mark.parametrize("size,error", [
    (-1, ValueError), (1.5, TypeError), (True, TypeError), ("10", TypeError),
])
def test_invalid_size_does_not_replace_an_existing_report(size, error):
    owner = Owner()
    register(owner, 8)
    with pytest.raises(error):
        register(owner, size)
    assert total() == 8


@pytest.mark.parametrize("device", ["cuda", "cuda:-1", "", "cuda:one"])
def test_ambiguous_or_invalid_device_is_rejected_without_querying_cuda(device):
    with pytest.raises(ValueError, match="explicit ordinal"):
        register(Owner(), 1, device=device)
    assert reports.get_runtime_reservation_snapshots() == []


def test_owner_must_support_weak_references():
    with pytest.raises(TypeError):
        register(object(), 1)
    assert reports.get_runtime_reservation_snapshots() == []


@pytest.mark.parametrize("arguments", [{"pool": ""}, {"engine": " vllm"}])
def test_empty_or_ambiguous_category_is_rejected(arguments):
    with pytest.raises(ValueError):
        register(Owner(), 1, **arguments)


def test_snapshot_has_stable_order_and_json_contract():
    owner = Owner()
    register(owner, 20, pool="workspace")
    register(owner, 5, pool="graph")
    data = reports.get_runtime_reservation_snapshot_dicts()
    assert json.loads(json.dumps(data)) == data
    assert [item["pool_name"] for item in data] == ["graph", "workspace"]
    assert data[0] == {
        "schema_version": "kvcached.observability.v1", "integration": "vllm",
        "device": "cuda:0", "pool_name": "graph", "num_bytes": 5, "owner_count": 1,
    }


def test_concurrent_owner_updates_and_collection_do_not_lose_reports():
    owners = [Owner() for _ in range(4)]
    errors = []
    barrier = threading.Barrier(4)

    def write(owner):
        try:
            barrier.wait(timeout=5)
            for size in range(1, 101):
                register(owner, size)
                snapshots = reports.get_runtime_reservation_snapshots()
                assert all(item.owner_count <= 4 and item.num_bytes > 0 for item in snapshots)
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=write, args=(owner,)) for owner in owners]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    assert not errors
    assert all(not thread.is_alive() for thread in threads)
    assert total() == 400


@pytest.fixture(params=["sglang", "vllm"])
def integration(request, monkeypatch):
    # Load the actual public shim while replacing GPU/IPC dependencies only.
    torch: Any = types.ModuleType("torch")
    torch.dtype = object
    torch.Tensor = object
    monkeypatch.setitem(sys.modules, "torch", torch)
    for module in ("kvcached.kv_cache_manager", "kvcached.vmm_ops", "kvcached.tp_ipc_util"):
        monkeypatch.setitem(sys.modules, module, mock.MagicMock())
    engine = request.param
    path = Path(__file__).parents[1] / "kvcached" / "integration" / engine / "interfaces.py"
    spec = importlib.util.spec_from_file_location(f"_reservation_{engine}", path)
    assert spec and spec.loader
    shim: Any = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shim)
    shim._kvcached_initialized = False
    return engine, shim


def test_both_engine_interfaces_share_the_public_reporting_contract(integration):
    engine, shim = integration
    owner = Owner()
    shim.register_runtime_owned_reservation("cuda:0", "workspace", 128, owner=owner)
    assert total(engine=engine) == 128
    assert shim.get_runtime_owned_reservation_bytes("hip:0") == 128
    assert shim.get_runtime_owned_reservation_breakdown("cuda:0") == {"workspace": 128}
    assert shim.runtime_reservation_snapshots()[0].integration == engine
    assert shim.runtime_reservation_snapshot_dicts() == reports.get_runtime_reservation_snapshot_dicts(
        integration=engine)
    shim.create_kv_tensors.assert_not_called()
    shim._init_kvcached_impl.assert_not_called()


@pytest.mark.parametrize("initialized", [False, True])
def test_shutdown_clears_only_its_integration_reports(integration, initialized, monkeypatch):
    engine, shim = integration
    owner = Owner()
    register(owner, 20, engine=engine)
    other = "vllm" if engine == "sglang" else "sglang"
    register(owner, 30, engine=other)
    shim._kvcached_initialized = initialized
    shim.stop_worker_listener_threads.return_value = True
    monkeypatch.setattr(shim, "clear_registered_kv_cache_pools", lambda **kwargs: None)
    if engine == "vllm":
        monkeypatch.setattr(shim, "get_registered_kv_cache_pools", lambda **kwargs: [])
    assert shim.shutdown_kvcached() is True
    assert total(engine=engine) == 0
    assert total(engine=other) == 30


def test_incomplete_shutdown_preserves_reports_for_retry(integration):
    engine, shim = integration
    owner = Owner()
    register(owner, 20, engine=engine)
    shim._kvcached_initialized = True
    shim.stop_worker_listener_threads.return_value = False
    assert shim.shutdown_kvcached() is False
    assert total(engine=engine) == 20
    shim._shutdown_kvcached_impl.assert_not_called()


def test_failed_native_shutdown_preserves_reports(integration):
    engine, shim = integration
    owner = Owner()
    register(owner, 20, engine=engine)
    shim._kvcached_initialized = True
    shim.stop_worker_listener_threads.return_value = True
    shim._shutdown_kvcached_impl.side_effect = RuntimeError("shutdown incomplete")
    if engine == "vllm":
        shim.get_registered_kv_cache_pools = lambda **kwargs: []
    with pytest.raises(RuntimeError, match="shutdown incomplete"):
        shim.shutdown_kvcached()
    assert total(engine=engine) == 20
