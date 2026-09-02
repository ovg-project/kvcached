# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""The /dev/shm segment must be unlinked when vLLM's EngineCore shuts down
(issue #477).

On the vLLM path the segment belongs to the C++ MemInfoTracker, whose only
cleanup is its destructor. The forked EngineCore leaves through os._exit
after run_engine_core()'s finally block calls EngineCore.shutdown(), so no
destructor ever ran and the segment outlived the engine. These tests pin
down the fix: EngineCore.shutdown() now ends in shutdown_kvcached(), which
shuts down every registered pool, and KVCacheManager.shutdown() stops the
prealloc thread and unlinks the segment itself.

CPU-only: torch, posix_ipc and the compiled extension are stubbed.
"""

import importlib
import sys
import types
from typing import Any
from unittest import mock

import pytest

from kvcached.pool_registry import (
    clear_registered_kv_cache_pools,
    get_registered_kv_cache_pools,
    register_kv_cache_pool,
)


@pytest.fixture
def vllm_modules(monkeypatch):
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
    monkeypatch.delitem(
        sys.modules, "kvcached.integration.vllm.interfaces", raising=False
    )
    monkeypatch.delitem(
        sys.modules, "kvcached.integration.vllm.patches", raising=False
    )

    interfaces: Any = importlib.import_module(
        "kvcached.integration.vllm.interfaces"
    )
    patches: Any = importlib.import_module("kvcached.integration.vllm.patches")
    clear_registered_kv_cache_pools()
    yield interfaces, patches
    clear_registered_kv_cache_pools()


def _fake_engine_module(shutdown=None):
    engine_mod = types.ModuleType("mock_engine_mod")

    class FakeEngineCore:
        pass

    if shutdown is not None:
        FakeEngineCore.shutdown = shutdown  # type: ignore[attr-defined]
    setattr(engine_mod, "EngineCore", FakeEngineCore)
    return engine_mod


def test_engine_core_shutdown_releases_kvcached_after_vllm_teardown(
    monkeypatch, vllm_modules
):
    interfaces, patches = vllm_modules
    monkeypatch.setattr(patches, "enable_kvcached", lambda: True)
    calls = []
    monkeypatch.setattr(
        interfaces, "shutdown_kvcached", lambda: calls.append("kvcached")
    )
    engine_mod = _fake_engine_module(lambda self: calls.append("vllm"))

    assert patches.EngineCorePatch().patch_engine_shutdown(engine_mod)
    assert patches.EngineCorePatch().patch_engine_shutdown(engine_mod)  # idempotent
    engine_mod.EngineCore().shutdown()

    assert calls == ["vllm", "kvcached"]


def test_engine_core_shutdown_releases_kvcached_even_if_vllm_teardown_raises(
    monkeypatch, vllm_modules
):
    interfaces, patches = vllm_modules
    monkeypatch.setattr(patches, "enable_kvcached", lambda: True)
    shutdown_kvcached = mock.Mock()
    monkeypatch.setattr(interfaces, "shutdown_kvcached", shutdown_kvcached)

    def failing_shutdown(self):
        raise RuntimeError("executor teardown failed")

    engine_mod = _fake_engine_module(failing_shutdown)
    assert patches.EngineCorePatch().patch_engine_shutdown(engine_mod)

    with pytest.raises(RuntimeError, match="executor teardown"):
        engine_mod.EngineCore().shutdown()
    shutdown_kvcached.assert_called_once_with()


def test_engine_core_shutdown_does_not_mask_vllm_result_when_kvcached_fails(
    monkeypatch, vllm_modules
):
    interfaces, patches = vllm_modules
    monkeypatch.setattr(patches, "enable_kvcached", lambda: True)
    monkeypatch.setattr(
        interfaces, "shutdown_kvcached",
        mock.Mock(side_effect=RuntimeError("segment busy")),
    )
    engine_mod = _fake_engine_module(lambda self: "done")
    assert patches.EngineCorePatch().patch_engine_shutdown(engine_mod)

    assert engine_mod.EngineCore().shutdown() == "done"


def test_engine_core_shutdown_patch_is_inert_when_kvcached_is_disabled(
    monkeypatch, vllm_modules
):
    interfaces, patches = vllm_modules
    monkeypatch.setattr(patches, "enable_kvcached", lambda: False)
    shutdown_kvcached = mock.Mock()
    monkeypatch.setattr(interfaces, "shutdown_kvcached", shutdown_kvcached)
    engine_mod = _fake_engine_module(lambda self: None)
    assert patches.EngineCorePatch().patch_engine_shutdown(engine_mod)

    engine_mod.EngineCore().shutdown()

    shutdown_kvcached.assert_not_called()


def test_engine_core_without_shutdown_is_left_alone(vllm_modules):
    _, patches = vllm_modules
    engine_mod = _fake_engine_module()

    assert patches.EngineCorePatch().patch_engine_shutdown(engine_mod)
    assert not hasattr(engine_mod.EngineCore, "shutdown")


def test_shutdown_kvcached_shuts_down_registered_pools_before_the_allocator(
    monkeypatch, vllm_modules
):
    interfaces, _ = vllm_modules
    monkeypatch.setattr(interfaces, "_kvcached_initialized", True)
    calls = []
    monkeypatch.setattr(
        interfaces, "_shutdown_kvcached_impl", lambda: calls.append("allocator")
    )

    class Pool:
        pool_name = "block_pool"

        def shutdown(self):
            calls.append("pool")

    pool = Pool()
    register_kv_cache_pool(pool, integration="vllm")

    interfaces.shutdown_kvcached()

    assert calls == ["pool", "allocator"]
    assert get_registered_kv_cache_pools(integration="vllm") == []
    assert interfaces._kvcached_initialized is False


def test_shutdown_kvcached_keeps_going_when_a_pool_fails(monkeypatch, vllm_modules):
    interfaces, _ = vllm_modules
    monkeypatch.setattr(interfaces, "_kvcached_initialized", True)
    allocator_shutdown = mock.Mock()
    monkeypatch.setattr(interfaces, "_shutdown_kvcached_impl", allocator_shutdown)

    class BrokenPool:
        pool_name = "broken"

        def shutdown(self):
            raise RuntimeError("cannot stop prealloc thread")

    healthy = mock.Mock()
    broken = BrokenPool()
    register_kv_cache_pool(broken, integration="vllm")
    register_kv_cache_pool(healthy, integration="vllm")

    interfaces.shutdown_kvcached()

    healthy.shutdown.assert_called_once_with()
    allocator_shutdown.assert_called_once_with()
    assert get_registered_kv_cache_pools(integration="vllm") == []


def _install_vmm_ops_stub() -> None:
    stub = types.ModuleType("kvcached.vmm_ops")
    stub.PageAllocator = object  # type: ignore[attr-defined]
    stub.InternalPage = object  # type: ignore[attr-defined]
    stub.kv_tensors_created = lambda group_id=0: True  # type: ignore[attr-defined]
    stub.map_to_kv_tensors = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    stub.unmap_from_kv_tensors = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    sys.modules["kvcached.vmm_ops"] = stub


def _manager_module():
    """kvcached.kv_cache_manager with the compiled extension stubbed if absent
    (the same arrangement tests/test_alloc_rollback.py uses)."""
    try:
        import kvcached.vmm_ops  # noqa: F401
    except ImportError:
        _install_vmm_ops_stub()
    return importlib.import_module("kvcached.kv_cache_manager")


def _make_manager(page_allocator, ipc_name="kvcached_test_477"):
    """A KVCacheManager around a fake allocator, without running __init__
    (which needs the C++ extension, KV tensors, and background threads)."""
    module = _manager_module()
    manager = object.__new__(module.KVCacheManager)
    manager.page_allocator = page_allocator
    manager.ipc_name = ipc_name
    manager._shut_down = False
    return manager


@pytest.fixture
def shm_dir(monkeypatch, tmp_path):
    module = _manager_module()
    monkeypatch.setattr(module, "SHM_DIR", str(tmp_path))
    return tmp_path


def test_manager_shutdown_stops_prealloc_and_unlinks_the_segment(shm_dir):
    allocator = mock.Mock()
    manager = _make_manager(allocator)
    segment = shm_dir / manager.ipc_name
    segment.write_bytes(b"\0" * 24)

    manager.shutdown()

    allocator.stop_prealloc_thread.assert_called_once_with()
    assert not segment.exists()


def test_manager_shutdown_is_idempotent(shm_dir):
    allocator = mock.Mock()
    manager = _make_manager(allocator)
    (shm_dir / manager.ipc_name).write_bytes(b"\0" * 24)

    manager.shutdown()
    manager.shutdown()

    allocator.stop_prealloc_thread.assert_called_once_with()


def test_manager_shutdown_tolerates_a_missing_segment(shm_dir):
    manager = _make_manager(mock.Mock())

    manager.shutdown()  # nothing to unlink, nothing raised


def test_manager_shutdown_unlinks_even_if_the_prealloc_thread_will_not_stop(shm_dir):
    allocator = mock.Mock()
    allocator.stop_prealloc_thread.side_effect = RuntimeError("join timed out")
    manager = _make_manager(allocator)
    segment = shm_dir / manager.ipc_name
    segment.write_bytes(b"\0" * 24)

    manager.shutdown()

    assert not segment.exists()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
