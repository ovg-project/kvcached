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
prealloc thread and asks the native creator to release its retained identity.

A server-level SIGTERM usually kills the engine before that unlink runs:
run_engine_core() restores SIGTERM to SIG_DFL before EngineCore.shutdown(),
so the terminate() from MPClient.shutdown()'s process manager ends the
engine mid-teardown (with --shutdown-timeout 0 a SIGKILL follows). The
client outlives the engines, so MPClient.shutdown() now also removes
    the original segment they left behind, through IPCSegmentCleanup.

Under --api-server-count > 1 no frontend's client owns the engines: the
supervisor launches them through a CoreEngineProcManager and calls its
shutdown directly, with no MPClient in that process at all. That manager
is the final owner boundary, so its shutdown now removes the segment the
killed engines leave, the same capture-then-unlink as the client patch.

CPU-only: torch, posix_ipc and the compiled extension are stubbed.
"""

import importlib
import os
import sys
import threading
import types
from typing import Any
from unittest import mock

import pytest

from kvcached import utils as kv_utils
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
    # Other test files leave never-stopped listeners registered in
    # kvcached.tp_ipc_util (see test_tp_socket_cleanup.socket_root); import
    # it here, under the stubs, and drop leftovers so the listener gate in
    # shutdown_kvcached() only meets listeners these tests start.
    tp_ipc_util = importlib.import_module("kvcached.tp_ipc_util")
    tp_ipc_util._listeners.clear()
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


def _fake_client_module(shutdown=None, resources="owner"):
    """A mock core_client module.

    The default FakeMPClient owns its engines (resources.engine_manager
    set), like the single-API MPClient that launched them. Pass any object
    to use it as the client's resources, or None for a client without the
    attribute.
    """
    client_mod = types.ModuleType("mock_client_mod")

    class FakeMPClient:
        def __init__(self):
            self._test_children = ()
            if resources == "owner":
                self.resources = types.SimpleNamespace(engine_manager=types.SimpleNamespace(
                    processes=[types.SimpleNamespace(exitcode=None, join=mock.Mock())]))
                self._test_children = tuple(self.resources.engine_manager.processes)
            elif resources is not None:
                self.resources = resources

    if shutdown is not None:
        def stop(self, *args, **kwargs):
            children = self._test_children
            try:
                return shutdown(self, *args, **kwargs)
            finally:
                for child in children:
                    child.exitcode = 0

        FakeMPClient.shutdown = stop  # type: ignore[attr-defined]
    setattr(client_mod, "MPClient", FakeMPClient)
    return client_mod


def test_mp_client_shutdown_unlinks_the_segment_once_the_engines_are_stopped(
    monkeypatch, vllm_modules
):
    _, patches = vllm_modules
    monkeypatch.setattr(patches, "enable_kvcached", lambda: True)
    calls = []
    monkeypatch.setattr(
        kv_utils, "IPCSegmentCleanup",
        lambda path: types.SimpleNamespace(unlink=lambda: calls.append("unlink")),
    )
    client_mod = _fake_client_module(lambda self: calls.append("vllm"))

    assert patches.MPClientPatch().patch_client_shutdown(client_mod)
    assert patches.MPClientPatch().patch_client_shutdown(client_mod)  # idempotent
    client_mod.MPClient().shutdown()

    assert calls == ["vllm", "unlink"]


def test_non_owning_client_shutdown_does_not_unlink(monkeypatch, vllm_modules):
    """The --api-server-count 2 case: a frontend's MPClient has
    resources.engine_manager None because the supervisor owns the engines,
    and its shutdown stops nothing, so it must not remove the segment the
    live engines still use."""
    _, patches = vllm_modules
    monkeypatch.setattr(patches, "enable_kvcached", lambda: True)
    calls = []
    monkeypatch.setattr(
        kv_utils, "IPCSegmentCleanup",
        lambda path: types.SimpleNamespace(unlink=lambda: calls.append("unlink")),
    )
    client_mod = _fake_client_module(
        lambda self: calls.append("vllm"),
        resources=types.SimpleNamespace(engine_manager=None),
    )
    assert patches.MPClientPatch().patch_client_shutdown(client_mod)
    client_mod.MPClient().shutdown()
    assert calls == ["vllm"]


def test_legacy_client_with_proc_handles_still_unlinks(monkeypatch, vllm_modules):
    """Older supported vLLM has no resources.engine_manager; there the
    owning client carries per-engine process handles on
    resources.core_engines."""
    _, patches = vllm_modules
    monkeypatch.setattr(patches, "enable_kvcached", lambda: True)
    calls = []
    monkeypatch.setattr(
        kv_utils, "IPCSegmentCleanup",
        lambda path: types.SimpleNamespace(unlink=lambda: calls.append("unlink")),
    )
    process = types.SimpleNamespace(exitcode=None, join=mock.Mock())

    def stop(self):
        calls.append("vllm")
        process.exitcode = 0

    client_mod = _fake_client_module(stop, resources=types.SimpleNamespace(
        core_engines=[types.SimpleNamespace(proc_handle=process)]))
    assert patches.MPClientPatch().patch_client_shutdown(client_mod)
    client_mod.MPClient().shutdown()
    assert calls == ["vllm", "unlink"]


def test_legacy_client_without_proc_handles_does_not_unlink(
    monkeypatch, vllm_modules
):
    _, patches = vllm_modules
    monkeypatch.setattr(patches, "enable_kvcached", lambda: True)
    calls = []
    monkeypatch.setattr(
        kv_utils, "IPCSegmentCleanup",
        lambda path: types.SimpleNamespace(unlink=lambda: calls.append("unlink")),
    )
    client_mod = _fake_client_module(
        lambda self: calls.append("vllm"),
        resources=types.SimpleNamespace(
            core_engines=[types.SimpleNamespace(proc_handle=None)]),
    )
    assert patches.MPClientPatch().patch_client_shutdown(client_mod)
    client_mod.MPClient().shutdown()
    assert calls == ["vllm"]


def test_client_without_resources_is_treated_as_a_non_owner(
    monkeypatch, vllm_modules
):
    """Unknown client structure: leaking a segment is recoverable with
    kvctl delete, removing a live one is not, so no resources means no
    unlink."""
    _, patches = vllm_modules
    monkeypatch.setattr(patches, "enable_kvcached", lambda: True)
    calls = []
    monkeypatch.setattr(
        kv_utils, "IPCSegmentCleanup",
        lambda path: types.SimpleNamespace(unlink=lambda: calls.append("unlink")),
    )
    client_mod = _fake_client_module(
        lambda self: calls.append("vllm"), resources=None)
    assert patches.MPClientPatch().patch_client_shutdown(client_mod)
    client_mod.MPClient().shutdown()
    assert calls == ["vllm"]


def test_mp_client_shutdown_unlinks_after_exit_even_if_vllm_shutdown_raises(
    monkeypatch, vllm_modules
):
    _, patches = vllm_modules
    monkeypatch.setattr(patches, "enable_kvcached", lambda: True)
    unlink = mock.Mock()
    monkeypatch.setattr(
        kv_utils, "IPCSegmentCleanup", lambda path: types.SimpleNamespace(unlink=unlink)
    )

    def failing_shutdown(self):
        raise RuntimeError("engine manager close failed")

    client_mod = _fake_client_module(failing_shutdown)
    assert patches.MPClientPatch().patch_client_shutdown(client_mod)

    with pytest.raises(RuntimeError, match="engine manager"):
        client_mod.MPClient().shutdown()
    unlink.assert_called_once_with()


def test_mp_client_shutdown_does_not_mask_vllm_result_when_the_unlink_fails(
    monkeypatch, vllm_modules
):
    _, patches = vllm_modules
    monkeypatch.setattr(patches, "enable_kvcached", lambda: True)
    monkeypatch.setattr(
        kv_utils, "IPCSegmentCleanup",
        lambda path: types.SimpleNamespace(
            unlink=mock.Mock(side_effect=RuntimeError("segment busy"))),
    )
    client_mod = _fake_client_module(lambda self: "done")
    assert patches.MPClientPatch().patch_client_shutdown(client_mod)

    assert client_mod.MPClient().shutdown() == "done"


def test_mp_client_shutdown_patch_is_inert_when_kvcached_is_disabled(
    monkeypatch, vllm_modules
):
    _, patches = vllm_modules
    monkeypatch.setattr(patches, "enable_kvcached", lambda: False)
    unlink = mock.Mock()
    monkeypatch.setattr(kv_utils, "IPCSegmentCleanup", unlink)
    client_mod = _fake_client_module(lambda self: None)
    assert patches.MPClientPatch().patch_client_shutdown(client_mod)

    client_mod.MPClient().shutdown()

    unlink.assert_not_called()


def test_mp_client_without_shutdown_is_left_alone(vllm_modules):
    _, patches = vllm_modules
    client_mod = _fake_client_module()

    assert patches.MPClientPatch().patch_client_shutdown(client_mod)
    assert not hasattr(client_mod.MPClient, "shutdown")


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
        fail = True

        def shutdown(self):
            if self.fail:
                raise RuntimeError("cannot stop prealloc thread")

    healthy = mock.Mock()
    broken = BrokenPool()
    register_kv_cache_pool(broken, integration="vllm")
    register_kv_cache_pool(healthy, integration="vllm")

    interfaces.shutdown_kvcached()

    healthy.shutdown.assert_called_once_with()
    allocator_shutdown.assert_not_called()
    assert interfaces._kvcached_initialized is True
    assert len(get_registered_kv_cache_pools(integration="vllm")) == 2

    broken.fail = False
    interfaces.shutdown_kvcached()
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
    manager._shutdown_lock = threading.Lock()
    manager._shutdown_requested = threading.Event()
    manager._lifecycle = module.LifecycleState(ipc_name)
    manager._post_init_done = threading.Event()
    manager._post_init_done.set()
    manager._prealloc_stopped = False
    # Model the native constructor's retained identity. Production release is
    # tested separately with the compiled extension, including shared owners.
    path = os.path.join(kv_utils.SHM_DIR, ipc_name)
    with open(path, "wb") as segment:
        segment.write(b"\0" * 24)
    cleanup = kv_utils.IPCSegmentCleanup(path)
    page_allocator.release_shared_segment.side_effect = cleanup.unlink
    return manager


@pytest.fixture
def shm_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(kv_utils, "SHM_DIR", str(tmp_path))
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


def test_manager_shutdown_retries_stop_before_unlinking(shm_dir):
    allocator = mock.Mock()
    allocator.stop_prealloc_thread.side_effect = [RuntimeError("join timed out"), None]
    manager = _make_manager(allocator)
    segment = shm_dir / manager.ipc_name
    segment.write_bytes(b"\0" * 24)

    assert manager.shutdown() is False
    assert segment.exists()

    assert manager.shutdown() is True
    assert not segment.exists()
    assert allocator.stop_prealloc_thread.call_count == 2


@pytest.mark.parametrize("phase", ["readiness", "reservation", "thread_start"])
def test_shutdown_waits_for_inflight_post_init(
    phase, shm_dir, monkeypatch, vllm_modules
):
    interfaces, _ = vllm_modules
    monkeypatch.setattr(interfaces, "should_use_worker_ipc", lambda: False)
    module = _manager_module()
    allocator = mock.Mock()
    manager = _make_manager(allocator)
    manager._post_init_done.clear()
    manager.null_block = None
    manager.world_size = 1
    manager.pp_rank = 0
    manager.group_id = 0
    manager._reserve_null_block = mock.Mock()
    segment = shm_dir / manager.ipc_name
    segment.write_bytes(b"engine")
    entered = threading.Event()
    release = threading.Event()

    def hold_init(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return True

    monkeypatch.setattr(module, "kv_tensors_created", lambda **kwargs: True)
    if phase == "readiness":
        monkeypatch.setattr(module, "kv_tensors_created", hold_init)
    elif phase == "reservation":
        manager._reserve_null_block.side_effect = hold_init
    else:
        allocator.start_prealloc_thread.side_effect = hold_init

    thread = threading.Thread(target=manager._post_init)
    thread.start()
    try:
        assert entered.wait(5)
        assert manager.shutdown() is False
        assert segment.read_bytes() == b"engine"
        allocator.stop_prealloc_thread.assert_not_called()
    finally:
        release.set()
        thread.join(timeout=5)
    assert not thread.is_alive()
    assert manager.shutdown() is True
    allocator.stop_prealloc_thread.assert_called_once_with()
    assert allocator.start_prealloc_thread.call_count == (phase == "thread_start")
    assert not segment.exists()


@pytest.mark.parametrize("waiting_for", ["tensors", "capacity", "allocation"])
def test_shutdown_cancels_initialization_waits(
    waiting_for, shm_dir, monkeypatch, vllm_modules
):
    interfaces, _ = vllm_modules
    monkeypatch.setattr(interfaces, "should_use_worker_ipc", lambda: False)
    module = _manager_module()
    allocator = mock.Mock()
    manager = _make_manager(allocator)
    manager._post_init_done.clear()
    manager.null_block = None
    manager.reserve_null_block = True
    manager.world_size = 1
    manager.pp_rank = 0
    manager.group_id = 0
    manager._alloc = mock.Mock(return_value=None)
    manager.available_size = mock.Mock(return_value=0)
    entered = threading.Event()

    def check_tensors(**kwargs):
        if waiting_for == "tensors":
            entered.set()
            return False
        return True

    def capacity():
        entered.set()
        return 1 if waiting_for == "allocation" else 0

    monkeypatch.setattr(module, "kv_tensors_created", check_tensors)
    manager.available_size.side_effect = capacity
    segment = shm_dir / manager.ipc_name
    segment.write_bytes(b"engine")
    thread = threading.Thread(target=manager._post_init)
    thread.start()
    try:
        assert entered.wait(5)
        assert manager.shutdown() is True
    finally:
        manager._shutdown_requested.set()
        thread.join(timeout=5)
    assert not thread.is_alive()
    assert manager._post_init_done.is_set()
    allocator.start_prealloc_thread.assert_not_called()
    allocator.stop_prealloc_thread.assert_called_once_with()
    assert manager.null_block is None
    assert not segment.exists()


def test_ipc_segment_cleanup_removes_the_segment(tmp_path):
    segment = tmp_path / "kvcached_test_477"
    segment.write_bytes(b"\0" * 24)

    cleanup = kv_utils.IPCSegmentCleanup(str(segment))
    assert cleanup.unlink() is True
    assert not segment.exists()
    assert cleanup.unlink() is True  # already done


def test_ipc_segment_cleanup_warns_but_does_not_raise_on_os_error(
    monkeypatch, tmp_path
):
    segment = tmp_path / "kvcached_test_477"
    segment.write_bytes(b"engine")
    cleanup = kv_utils.IPCSegmentCleanup(str(segment))
    monkeypatch.setattr(
        kv_utils.os, "unlink", mock.Mock(side_effect=OSError("permission denied"))
    )

    assert cleanup.unlink() is False


def test_old_client_shutdown_preserves_replacement_segment(
    monkeypatch, tmp_path, vllm_modules
):
    _, patches = vllm_modules
    monkeypatch.setattr(patches, "enable_kvcached", lambda: True)
    monkeypatch.setattr(kv_utils, "SHM_DIR", str(tmp_path))
    monkeypatch.setattr(kv_utils, "DEFAULT_IPC_NAME", "reused_segment")
    segment = tmp_path / "reused_segment"
    segment.write_bytes(b"old engine")
    upstream_calls = []

    def shutdown_once(self):
        if not getattr(self, "closed", False):
            self.closed = True
            upstream_calls.append("shutdown")

    client_mod = _fake_client_module(shutdown_once)
    assert patches.MPClientPatch().patch_client_shutdown(client_mod)
    client = client_mod.MPClient()
    client.shutdown()
    assert not segment.exists()

    segment.write_bytes(b"replacement engine")
    client.shutdown()

    assert upstream_calls == ["shutdown"]
    assert segment.read_bytes() == b"replacement engine"


def test_manager_shutdown_retries_failed_unlink_without_stopping_twice(
    monkeypatch, shm_dir
):
    allocator = mock.Mock()
    manager = _make_manager(allocator)
    segment = shm_dir / manager.ipc_name
    segment.write_bytes(b"engine")
    real_unlink = kv_utils.os.unlink
    attempts = []

    def fail_once(path):
        attempts.append(path)
        if len(attempts) == 1:
            raise PermissionError("injected unlink failure")
        return real_unlink(path)

    monkeypatch.setattr(kv_utils.os, "unlink", fail_once)
    manager.shutdown()
    assert segment.exists()
    manager.shutdown()

    assert not segment.exists()
    assert len(attempts) == 2
    allocator.stop_prealloc_thread.assert_called_once_with()


@pytest.mark.parametrize("owner", ["client", "manager"])
def test_cleanup_retry_preserves_a_replaced_file(
    owner, monkeypatch, shm_dir, vllm_modules
):
    _, patches = vllm_modules
    segment = shm_dir / "kvcached_test_477"
    segment.write_bytes(b"old engine")
    if owner == "manager":
        instance = _make_manager(mock.Mock())
    else:
        monkeypatch.setattr(patches, "enable_kvcached", lambda: True)
        monkeypatch.setattr(kv_utils, "SHM_DIR", str(shm_dir))
        monkeypatch.setattr(kv_utils, "DEFAULT_IPC_NAME", segment.name)
        client_mod = _fake_client_module(lambda self: None)
        assert patches.MPClientPatch().patch_client_shutdown(client_mod)
        instance = client_mod.MPClient()

    with mock.patch.object(kv_utils.os, "unlink", side_effect=PermissionError("injected")):
        instance.shutdown()
    assert segment.exists()

    segment.unlink()
    segment.write_bytes(b"replacement engine")
    instance.shutdown()
    assert segment.read_bytes() == b"replacement engine"


def test_client_captures_segment_before_upstream_teardown(
    monkeypatch, tmp_path, vllm_modules
):
    _, patches = vllm_modules
    monkeypatch.setattr(patches, "enable_kvcached", lambda: True)
    monkeypatch.setattr(kv_utils, "SHM_DIR", str(tmp_path))
    monkeypatch.setattr(kv_utils, "DEFAULT_IPC_NAME", "reused_segment")
    segment = tmp_path / "reused_segment"
    segment.write_bytes(b"old engine")

    def teardown(self):
        segment.unlink()
        segment.write_bytes(b"replacement engine")

    client_mod = _fake_client_module(teardown)
    assert patches.MPClientPatch().patch_client_shutdown(client_mod)
    client_mod.MPClient().shutdown()

    assert segment.read_bytes() == b"replacement engine"


def test_client_retries_failed_unlink(monkeypatch, tmp_path, vllm_modules):
    _, patches = vllm_modules
    monkeypatch.setattr(patches, "enable_kvcached", lambda: True)
    monkeypatch.setattr(kv_utils, "SHM_DIR", str(tmp_path))
    monkeypatch.setattr(kv_utils, "DEFAULT_IPC_NAME", "retry_segment")
    segment = tmp_path / "retry_segment"
    segment.write_bytes(b"engine")
    client_mod = _fake_client_module(lambda self: None)
    assert patches.MPClientPatch().patch_client_shutdown(client_mod)
    client = client_mod.MPClient()

    with mock.patch.object(kv_utils.os, "unlink", side_effect=PermissionError("injected")):
        client.shutdown()
    assert segment.exists()

    client.shutdown()
    assert not segment.exists()


def test_missing_segment_is_not_claimed_by_later_shutdown(
    monkeypatch, tmp_path, vllm_modules
):
    _, patches = vllm_modules
    monkeypatch.setattr(patches, "enable_kvcached", lambda: True)
    monkeypatch.setattr(kv_utils, "SHM_DIR", str(tmp_path))
    monkeypatch.setattr(kv_utils, "DEFAULT_IPC_NAME", "absent_segment")
    segment = tmp_path / "absent_segment"
    client_mod = _fake_client_module(lambda self: None)
    assert patches.MPClientPatch().patch_client_shutdown(client_mod)
    client = client_mod.MPClient()
    client.shutdown()

    segment.write_bytes(b"new engine")
    client.shutdown()
    assert segment.read_bytes() == b"new engine"


def test_frontend_shutdown_preserves_the_segment_until_the_owner_exits(
    monkeypatch, tmp_path, vllm_modules
):
    """The multi-API end-to-end shape: one frontend goes down first and the
    segment must survive for EngineCore and the other frontend; the owning
    client's shutdown still removes it at the end."""
    _, patches = vllm_modules
    monkeypatch.setattr(patches, "enable_kvcached", lambda: True)
    monkeypatch.setattr(kv_utils, "SHM_DIR", str(tmp_path))
    monkeypatch.setattr(kv_utils, "DEFAULT_IPC_NAME", "shared_segment")
    segment = tmp_path / "shared_segment"
    segment.write_bytes(b"live engines")
    client_mod = _fake_client_module(lambda self: None)
    assert patches.MPClientPatch().patch_client_shutdown(client_mod)

    frontend_mod = _fake_client_module(lambda self: None,
                                       resources=types.SimpleNamespace(engine_manager=None))
    assert patches.MPClientPatch().patch_client_shutdown(frontend_mod)
    frontend = frontend_mod.MPClient()
    frontend.shutdown()
    assert segment.read_bytes() == b"live engines"

    owner = client_mod.MPClient()
    owner.shutdown()
    assert not segment.exists()


def test_unlink_retry_survives_shutdown_clearing_ownership(
    monkeypatch, tmp_path, vllm_modules
):
    """The cleanup is captured before the original shutdown runs and kept
    on the client, so a failed unlink still retries on the next call even
    if vLLM's teardown cleared resources.engine_manager meanwhile."""
    _, patches = vllm_modules
    monkeypatch.setattr(patches, "enable_kvcached", lambda: True)
    monkeypatch.setattr(kv_utils, "SHM_DIR", str(tmp_path))
    monkeypatch.setattr(kv_utils, "DEFAULT_IPC_NAME", "cleared_owner_segment")
    segment = tmp_path / "cleared_owner_segment"
    segment.write_bytes(b"engine")

    def teardown(self):
        self.resources.engine_manager = None

    client_mod = _fake_client_module(teardown)
    assert patches.MPClientPatch().patch_client_shutdown(client_mod)
    client = client_mod.MPClient()

    with mock.patch.object(kv_utils.os, "unlink", side_effect=PermissionError("injected")):
        client.shutdown()
    assert segment.exists()

    client.shutdown()
    assert not segment.exists()


def test_interface_shutdown_retries_pool_unlink(monkeypatch, shm_dir, vllm_modules):
    interfaces, _ = vllm_modules
    monkeypatch.setattr(interfaces, "_kvcached_initialized", True)
    native_shutdown = mock.Mock()
    monkeypatch.setattr(interfaces, "_shutdown_kvcached_impl", native_shutdown)
    allocator = mock.Mock()
    manager = _make_manager(allocator)
    segment = shm_dir / manager.ipc_name
    segment.write_bytes(b"engine")
    register_kv_cache_pool(manager, integration="vllm")

    with mock.patch.object(kv_utils.os, "unlink", side_effect=PermissionError("injected")):
        interfaces.shutdown_kvcached()
    native_shutdown.assert_not_called()
    assert segment.exists()
    assert interfaces._kvcached_initialized is True
    assert get_registered_kv_cache_pools(integration="vllm") == [(manager, "vllm")]

    interfaces.shutdown_kvcached()
    assert not segment.exists()
    native_shutdown.assert_called_once_with()
    allocator.stop_prealloc_thread.assert_called_once_with()
    assert interfaces._kvcached_initialized is False
    assert get_registered_kv_cache_pools(integration="vllm") == []


def _fake_engine_utils_module(shutdown=None):
    """A mock vllm.v1.engine.utils module.

    FakeCoreEngineProcManager stands in for the manager that spawned the
    EngineCore processes. In the supervisor under --api-server-count > 1
    and in headless mode there is no owning MPClient, and vLLM calls this
    manager's shutdown directly on exit.
    """
    utils_mod = types.ModuleType("mock_engine_utils_mod")

    class FakeCoreEngineProcManager:
        def __init__(self, processes=None):
            self._stop_test_children = processes is None
            self.processes = processes if processes is not None else [
                types.SimpleNamespace(exitcode=None, join=mock.Mock())]
            # Simulate launch_core_engines completing its ready handshake, not
            # just the real manager constructor (which returns before READY).
            utils_mod.wait_for_engine_startup(proc_manager=self)

    if shutdown is not None:
        def stop(self, *args, **kwargs):
            children = tuple(self.processes) if self._stop_test_children else ()
            try:
                return shutdown(self, *args, **kwargs)
            finally:
                for child in children:
                    child.exitcode = 0

        FakeCoreEngineProcManager.shutdown = stop  # type: ignore[attr-defined]
    setattr(utils_mod, "CoreEngineProcManager", FakeCoreEngineProcManager)
    setattr(utils_mod, "wait_for_engine_startup", lambda proc_manager: None)
    return utils_mod


def test_engine_manager_shutdown_unlinks_the_segment_once_the_engines_are_stopped(
    monkeypatch, vllm_modules
):
    _, patches = vllm_modules
    monkeypatch.setattr(patches, "enable_kvcached", lambda: True)
    calls = []
    monkeypatch.setattr(
        kv_utils, "IPCSegmentCleanup",
        lambda path: types.SimpleNamespace(unlink=lambda: calls.append("unlink")),
    )
    utils_mod = _fake_engine_utils_module(
        lambda self, timeout=None: calls.append("vllm"))

    patch = patches.CoreEngineProcManagerPatch()
    assert patch.patch_manager_shutdown(utils_mod)
    assert patch.patch_manager_shutdown(utils_mod)  # idempotent
    utils_mod.CoreEngineProcManager().shutdown(timeout=0.0)

    assert calls == ["vllm", "unlink"]


def test_two_frontend_final_exit_removes_the_segment_via_the_supervisor(
    monkeypatch, tmp_path, vllm_modules
):
    """The --api-server-count 2 process-group SIGTERM shape: both frontends
    are non-owners and exit without touching the segment, and the killed
    engines never ran their own unlink. The supervisor owns the engines
    through its CoreEngineProcManager, so its direct shutdown call is the
    last exit and must remove the segment."""
    _, patches = vllm_modules
    monkeypatch.setattr(patches, "enable_kvcached", lambda: True)
    monkeypatch.setattr(kv_utils, "SHM_DIR", str(tmp_path))
    monkeypatch.setattr(kv_utils, "DEFAULT_IPC_NAME", "supervisor_segment")
    segment = tmp_path / "supervisor_segment"
    segment.write_bytes(b"killed engines")

    client_mod = _fake_client_module(lambda self: None,
                                    resources=types.SimpleNamespace(engine_manager=None))
    assert patches.MPClientPatch().patch_client_shutdown(client_mod)
    for _ in range(2):
        frontend = client_mod.MPClient()
        frontend.shutdown()
    assert segment.read_bytes() == b"killed engines"

    utils_mod = _fake_engine_utils_module(lambda self, timeout=None: None)
    assert patches.CoreEngineProcManagerPatch().patch_manager_shutdown(utils_mod)
    utils_mod.CoreEngineProcManager().shutdown(timeout=0.0)

    assert not segment.exists()


def test_engine_manager_shutdown_unlinks_after_exit_even_if_vllm_shutdown_raises(
    monkeypatch, vllm_modules
):
    _, patches = vllm_modules
    monkeypatch.setattr(patches, "enable_kvcached", lambda: True)
    unlink = mock.Mock()
    monkeypatch.setattr(
        kv_utils, "IPCSegmentCleanup", lambda path: types.SimpleNamespace(unlink=unlink)
    )

    def failing_shutdown(self, timeout=None):
        raise RuntimeError("engine terminate failed")

    utils_mod = _fake_engine_utils_module(failing_shutdown)
    assert patches.CoreEngineProcManagerPatch().patch_manager_shutdown(utils_mod)

    with pytest.raises(RuntimeError, match="engine terminate"):
        utils_mod.CoreEngineProcManager().shutdown()
    unlink.assert_called_once_with()


def test_engine_manager_shutdown_does_not_mask_vllm_result_when_the_unlink_fails(
    monkeypatch, vllm_modules
):
    _, patches = vllm_modules
    monkeypatch.setattr(patches, "enable_kvcached", lambda: True)
    monkeypatch.setattr(
        kv_utils, "IPCSegmentCleanup",
        lambda path: types.SimpleNamespace(
            unlink=mock.Mock(side_effect=RuntimeError("segment busy"))),
    )
    utils_mod = _fake_engine_utils_module(lambda self, timeout=None: "done")
    assert patches.CoreEngineProcManagerPatch().patch_manager_shutdown(utils_mod)

    assert utils_mod.CoreEngineProcManager().shutdown() == "done"


def test_engine_manager_shutdown_patch_is_inert_when_kvcached_is_disabled(
    monkeypatch, vllm_modules
):
    _, patches = vllm_modules
    monkeypatch.setattr(patches, "enable_kvcached", lambda: False)
    cleanup = mock.Mock()
    monkeypatch.setattr(kv_utils, "IPCSegmentCleanup", cleanup)
    utils_mod = _fake_engine_utils_module(lambda self, timeout=None: None)
    assert patches.CoreEngineProcManagerPatch().patch_manager_shutdown(utils_mod)

    utils_mod.CoreEngineProcManager().shutdown()

    cleanup.assert_not_called()


def test_engine_manager_captures_segment_before_upstream_teardown(
    monkeypatch, tmp_path, vllm_modules
):
    """Same replacement protection as the client patch: the segment is
    captured before the original shutdown runs, so a file that replaces
    it during teardown is preserved."""
    _, patches = vllm_modules
    monkeypatch.setattr(patches, "enable_kvcached", lambda: True)
    monkeypatch.setattr(kv_utils, "SHM_DIR", str(tmp_path))
    monkeypatch.setattr(kv_utils, "DEFAULT_IPC_NAME", "reused_segment")
    segment = tmp_path / "reused_segment"
    segment.write_bytes(b"old engine")

    def teardown(self, timeout=None):
        segment.unlink()
        segment.write_bytes(b"replacement engine")

    utils_mod = _fake_engine_utils_module(teardown)
    assert patches.CoreEngineProcManagerPatch().patch_manager_shutdown(utils_mod)
    utils_mod.CoreEngineProcManager().shutdown()

    assert segment.read_bytes() == b"replacement engine"


def test_engine_manager_retries_failed_unlink(monkeypatch, tmp_path, vllm_modules):
    _, patches = vllm_modules
    monkeypatch.setattr(patches, "enable_kvcached", lambda: True)
    monkeypatch.setattr(kv_utils, "SHM_DIR", str(tmp_path))
    monkeypatch.setattr(kv_utils, "DEFAULT_IPC_NAME", "retry_segment")
    segment = tmp_path / "retry_segment"
    segment.write_bytes(b"engine")
    utils_mod = _fake_engine_utils_module(lambda self, timeout=None: None)
    assert patches.CoreEngineProcManagerPatch().patch_manager_shutdown(utils_mod)
    manager = utils_mod.CoreEngineProcManager()

    with mock.patch.object(kv_utils.os, "unlink", side_effect=PermissionError("injected")):
        manager.shutdown()
    assert segment.exists()

    manager.shutdown()
    assert not segment.exists()


def test_engine_manager_without_shutdown_is_left_alone(vllm_modules):
    _, patches = vllm_modules
    utils_mod = _fake_engine_utils_module()

    assert patches.CoreEngineProcManagerPatch().patch_manager_shutdown(utils_mod)
    assert not hasattr(utils_mod.CoreEngineProcManager, "shutdown")


@pytest.mark.parametrize("boundary", ["client", "manager", "legacy_client"])
@pytest.mark.parametrize("raise_error", [False, True])
def test_shutdown_preserves_live_segment_and_retries_after_engine_exit(
    monkeypatch, tmp_path, vllm_modules, boundary, raise_error
):
    _, patches = vllm_modules
    monkeypatch.setattr(patches, "enable_kvcached", lambda: True)
    monkeypatch.setattr(kv_utils, "SHM_DIR", str(tmp_path))
    monkeypatch.setattr(kv_utils, "DEFAULT_IPC_NAME", "live_segment")
    segment = tmp_path / "live_segment"
    segment.write_bytes(b"still in use")
    process = types.SimpleNamespace(exitcode=None)
    error = RuntimeError("injected terminate failure")

    def shutdown(self):
        # Real teardown can clear ownership even when it failed to stop a child.
        if boundary != "manager":
            self.resources = None
        else:
            self.processes = []
        if raise_error:
            raise error
        return "done"

    if boundary == "manager":
        mod = _fake_engine_utils_module(shutdown)
        assert patches.CoreEngineProcManagerPatch().patch_manager_shutdown(mod)
        owner = mod.CoreEngineProcManager(processes=[process])
    else:
        if boundary == "client":
            resources = types.SimpleNamespace(engine_manager=types.SimpleNamespace(
                processes=[process]))
        else:
            resources = types.SimpleNamespace(core_engines=[types.SimpleNamespace(
                proc_handle=process)])
        mod = _fake_client_module(shutdown, resources=resources)
        assert patches.MPClientPatch().patch_client_shutdown(mod)
        owner = mod.MPClient()

    def stop():
        if raise_error:
            with pytest.raises(RuntimeError) as exc:
                owner.shutdown()
            assert exc.value is error
        else:
            assert owner.shutdown() == "done"

    stop()
    assert segment.read_bytes() == b"still in use"
    stop()  # A repeated/no-op upstream shutdown still cannot unlink a live segment.
    assert segment.exists()
    process.exitcode = -9
    stop()
    assert not segment.exists()


@pytest.mark.parametrize("boundary", ["client", "manager"])
@pytest.mark.parametrize("state", ["missing", "raises", "empty"])
def test_shutdown_preserves_segment_when_engine_exit_cannot_be_checked(
    monkeypatch, vllm_modules, boundary, state
):
    _, patches = vllm_modules
    monkeypatch.setattr(patches, "enable_kvcached", lambda: True)
    unlink = mock.Mock()
    monkeypatch.setattr(kv_utils, "IPCSegmentCleanup",
                        lambda path: types.SimpleNamespace(unlink=unlink))

    class UnknownProcess:
        @property
        def exitcode(self):
            raise ValueError("process handle closed")

    processes = [] if state == "empty" else [
        UnknownProcess() if state == "raises" else object()]
    if boundary == "manager":
        mod = _fake_engine_utils_module(lambda self: "done")
        assert patches.CoreEngineProcManagerPatch().patch_manager_shutdown(mod)
        owner = mod.CoreEngineProcManager(processes=processes)
    else:
        resources = types.SimpleNamespace(engine_manager=types.SimpleNamespace(
            processes=processes))
        mod = _fake_client_module(lambda self: "done", resources=resources)
        assert patches.MPClientPatch().patch_client_shutdown(mod)
        owner = mod.MPClient()
    assert owner.shutdown() == "done"
    unlink.assert_not_called()


def test_segment_cleanup_reaps_exiting_children_with_one_group_deadline(
    monkeypatch, vllm_modules
):
    _, patches = vllm_modules
    cleanup = mock.Mock()
    monkeypatch.setattr(patches.time, "monotonic", mock.Mock(side_effect=[10.0, 10.0, 10.75]))
    children = [types.SimpleNamespace(exitcode=None) for _ in range(2)]
    calls = []

    def join(index, timeout):
        calls.append((index, timeout))
        cleanup.unlink.assert_not_called()
        children[index].exitcode = -9

    children[0].join = lambda timeout: join(0, timeout)
    children[1].join = lambda timeout: join(1, timeout)
    patches._unlink_stopped_engine_segment(cleanup, tuple(children))
    assert calls == [(0, 1.0), (1, 0.25)]
    cleanup.unlink.assert_called_once_with()


def test_segment_cleanup_keeps_live_child_after_bounded_join(vllm_modules):
    _, patches = vllm_modules
    cleanup = mock.Mock()
    child = types.SimpleNamespace(exitcode=None, join=mock.Mock())
    patches._unlink_stopped_engine_segment(cleanup, (child,))
    timeout = child.join.call_args.kwargs["timeout"]
    assert 0 <= timeout <= 1.0
    cleanup.unlink.assert_not_called()


@pytest.mark.parametrize("startup_api", ["proc_manager", "launch"])
def test_supervisor_captures_only_after_ready_handshake(
    startup_api, monkeypatch, tmp_path, vllm_modules,
):
    _, patches = vllm_modules
    monkeypatch.setattr(patches, "enable_kvcached", lambda: True)
    monkeypatch.setattr(kv_utils, "SHM_DIR", str(tmp_path))
    monkeypatch.setattr(kv_utils, "DEFAULT_IPC_NAME", "handshake_segment")
    segment = tmp_path / "handshake_segment"
    mod = types.ModuleType("engine_utils")

    class Manager:
        def __init__(self):
            self.processes = [types.SimpleNamespace(exitcode=None)]

        def shutdown(self):
            self.processes[0].exitcode = 0

    setattr(mod, "CoreEngineProcManager", Manager)

    def ready(manager):
        assert not hasattr(manager, "_kvcached_ipc_cleanup")
        segment.write_bytes(b"original ready engine")
        return "ready"

    if startup_api == "proc_manager":
        setattr(mod, "wait_for_engine_startup", lambda proc_manager: ready(proc_manager))
    else:
        setattr(mod, "wait_for_engine_startup", lambda launch: ready(launch.engine_manager))
    assert patches.CoreEngineProcManagerPatch().patch_manager_shutdown(mod)
    owner = mod.CoreEngineProcManager()
    assert not hasattr(owner, "_kvcached_ipc_cleanup")
    argument = owner if startup_api == "proc_manager" else types.SimpleNamespace(engine_manager=owner)
    assert mod.wait_for_engine_startup(argument) == "ready"
    owner.processes[0].exitcode = 0
    segment.unlink()
    segment.write_bytes(b"replacement")
    owner.shutdown()
    assert segment.read_bytes() == b"replacement"


def test_supervisor_without_ready_handshake_does_not_claim_later_file(
    monkeypatch, tmp_path, vllm_modules,
):
    _, patches = vllm_modules
    monkeypatch.setattr(patches, "enable_kvcached", lambda: True)
    monkeypatch.setattr(kv_utils, "SHM_DIR", str(tmp_path))
    monkeypatch.setattr(kv_utils, "DEFAULT_IPC_NAME", "startup_failure")
    mod = _fake_engine_utils_module(lambda self: "stopped")
    mod.wait_for_engine_startup = mock.Mock(side_effect=RuntimeError("not ready"))
    # Real manager construction returns before the handshake in launch_core_engines.
    mod.CoreEngineProcManager.__init__ = lambda self: setattr(self, "processes", [])
    assert patches.CoreEngineProcManagerPatch().patch_manager_shutdown(mod)
    owner = mod.CoreEngineProcManager()
    owner._stop_test_children = False
    with pytest.raises(RuntimeError, match="not ready"):
        mod.wait_for_engine_startup(proc_manager=owner)
    segment = tmp_path / "startup_failure"
    segment.write_bytes(b"unowned")
    assert owner.shutdown() == "stopped"
    assert segment.read_bytes() == b"unowned"


def test_client_reuses_supervisor_identity_without_recapturing(monkeypatch, vllm_modules):
    _, patches = vllm_modules
    monkeypatch.setattr(patches, "enable_kvcached", lambda: True)
    child = types.SimpleNamespace(exitcode=None)
    cleanup = mock.Mock()
    manager = types.SimpleNamespace(
        processes=[child], _kvcached_engine_processes=(child,), _kvcached_ipc_cleanup=cleanup,
    )
    capture = mock.Mock(side_effect=AssertionError("must not recapture"))
    monkeypatch.setattr(kv_utils, "IPCSegmentCleanup", capture)
    mod = _fake_client_module(lambda self: setattr(child, "exitcode", 0),
                              resources=types.SimpleNamespace(engine_manager=manager))
    assert patches.MPClientPatch().patch_client_shutdown(mod)
    client = mod.MPClient()
    assert client._kvcached_ipc_cleanup is cleanup
    client.shutdown()
    cleanup.unlink.assert_called_once_with()
    capture.assert_not_called()


@pytest.mark.parametrize("timing", [
    "during_constructor", "during_monitor", "preexisting", "after_exit",
    "no_segment", "monitor_error", "thread_start_error",
])
def test_headless_supervisor_without_ready_handshake(
    timing, monkeypatch, tmp_path, vllm_modules,
):
    _, patches = vllm_modules
    monkeypatch.setattr(patches, "enable_kvcached", lambda: True)
    monkeypatch.setattr(kv_utils, "SHM_DIR", str(tmp_path))
    monkeypatch.setattr(kv_utils, "DEFAULT_IPC_NAME", "headless_segment")
    path = tmp_path / "headless_segment"
    entered = threading.Event()
    finish = threading.Event()
    captured = threading.Event()
    original_capture = kv_utils.IPCSegmentCleanup
    monitor_error = RuntimeError("original monitor failed")

    def capture(segment):
        result = original_capture(segment)
        captured.set()
        return result

    monkeypatch.setattr(kv_utils, "IPCSegmentCleanup", capture)
    if timing == "preexisting":
        path.write_bytes(b"unconfirmed owner")

    class Manager:
        def __init__(self, local_client=False):
            self.processes = [types.SimpleNamespace(exitcode=None, join=mock.Mock())]
            if timing == "during_constructor":
                path.write_bytes(b"owned segment")

        def monitor_engine_liveness(self):
            entered.set()
            assert finish.wait(5)
            if timing == "monitor_error":
                raise monitor_error
            self.shutdown()

        def shutdown(self):
            self.processes[0].exitcode = 0

    mod = types.ModuleType("headless_utils")
    setattr(mod, "CoreEngineProcManager", Manager)
    # run_headless uses the manager directly, without wait_for_engine_startup.
    assert patches.CoreEngineProcManagerPatch().patch_manager_shutdown(mod)
    owner = mod.CoreEngineProcManager(local_client=False)
    if timing == "thread_start_error":
        original_start = threading.Thread.start

        def start(thread):
            if thread.name == "kvcached-segment-owner":
                raise RuntimeError("injected thread start failure")
            return original_start(thread)

        monkeypatch.setattr(threading.Thread, "start", start)
    if timing == "after_exit":
        owner.processes[0].exitcode = 0
        path.write_bytes(b"replacement")
    errors = []

    def monitor():
        try:
            owner.monitor_engine_liveness()
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=monitor)
    thread.start()
    try:
        assert entered.wait(5)
        if timing == "during_monitor":
            path.write_bytes(b"owned segment")
        if timing in ("during_constructor", "during_monitor"):
            assert captured.wait(5), "headless launch never retained its segment"
    finally:
        finish.set()
        thread.join(5)
    assert not thread.is_alive()
    assert errors == ([monitor_error] if timing == "monitor_error" else [])
    if timing in ("during_constructor", "during_monitor"):
        assert not path.exists()
    elif timing in ("preexisting", "after_exit"):
        assert path.exists()
        assert not captured.is_set()
    else:
        assert not path.exists()
        assert not captured.is_set()
    watch = getattr(owner, "_kvcached_segment_watch", None)
    if watch is not None:
        assert not watch[1].is_alive()


def test_headless_watch_stop_failure_keeps_upstream_shutdown(monkeypatch, vllm_modules):
    _, patches = vllm_modules
    monkeypatch.setattr(patches, "enable_kvcached", lambda: True)
    original_error = RuntimeError("upstream shutdown failed")

    class Manager:
        def __init__(self):
            self.processes = []

        def shutdown(self):
            raise original_error

    mod = types.ModuleType("headless_utils")
    setattr(mod, "CoreEngineProcManager", Manager)
    assert patches.CoreEngineProcManagerPatch().patch_manager_shutdown(mod)
    owner = mod.CoreEngineProcManager()
    thread = mock.Mock()
    thread.join.side_effect = RuntimeError("injected join failure")
    owner._kvcached_segment_watch = (threading.Event(), thread)
    with pytest.raises(RuntimeError) as exc:
        owner.shutdown()
    assert exc.value is original_error


@pytest.fixture(params=["client", "legacy_client", "supervisor"])
def lifecycle_owner(request, monkeypatch, tmp_path, vllm_modules):
    """One ready owner, mutable child states, and a real original segment.

    The fixture does not populate private cleanup fields. Production hooks must
    establish ownership themselves; otherwise late first cleanup stays red.
    """
    _, patches = vllm_modules
    monkeypatch.setattr(patches, "enable_kvcached", lambda: True)
    monkeypatch.setattr(kv_utils, "SHM_DIR", str(tmp_path))
    monkeypatch.setattr(kv_utils, "DEFAULT_IPC_NAME", "lifecycle_segment")
    segment = tmp_path / "lifecycle_segment"
    segment.write_bytes(b"original generation")
    children = [types.SimpleNamespace(exitcode=None, join=mock.Mock()) for _ in range(4)]
    state = types.SimpleNamespace(
        segment=segment, children=children, stop_children=True,
        during_shutdown=lambda: None, upstream_error=None, calls=0,
    )

    def shutdown(self, *args, **kwargs):
        state.calls += 1
        if state.stop_children:
            for child in children:
                child.exitcode = 0
        state.during_shutdown()
        if state.upstream_error is not None:
            raise state.upstream_error
        return "upstream result"

    if request.param == "supervisor":
        mod = _fake_engine_utils_module(shutdown)
        assert patches.CoreEngineProcManagerPatch().patch_manager_shutdown(mod)
        def owner_class():
            return mod.CoreEngineProcManager(processes=children)
    else:
        if request.param == "client":
            resources = types.SimpleNamespace(
                engine_manager=types.SimpleNamespace(processes=children))
        else:
            resources = types.SimpleNamespace(core_engines=[
                types.SimpleNamespace(proc_handle=child) for child in children])
        mod = _fake_client_module(shutdown, resources=resources)
        assert patches.MPClientPatch().patch_client_shutdown(mod)
        owner_class = mod.MPClient

    # Pin the test's original inode independently of the production cleanup.
    # Without this descriptor the OS may immediately reuse the old inode.
    with segment.open("rb") as original:
        state.original_identity = os.fstat(original.fileno())
        state.owner_class = owner_class
        state.owner = owner_class()
        yield state


def _replace_lifecycle_segment(state):
    state.segment.unlink(missing_ok=True)
    state.segment.write_bytes(b"replacement generation")
    state.replacement_identity = state.segment.stat()
    assert not os.path.samestat(state.original_identity, state.replacement_identity)


def _assert_replacement_preserved(state):
    assert state.segment.exists(), "old owner removed a replacement generation"
    assert os.path.samestat(state.segment.stat(), state.replacement_identity)
    assert state.segment.read_bytes() == b"replacement generation"


@pytest.mark.parametrize("exit_before_call", [False, True])
def test_lifecycle_owned_original_is_eventually_removed(lifecycle_owner, exit_before_call):
    state = lifecycle_owner
    if exit_before_call:
        for child in state.children:
            child.exitcode = 0
    assert state.owner.shutdown() == "upstream result"
    assert not state.segment.exists(), "preserving every segment is not a cleanup fix"
    assert state.owner.shutdown() == "upstream result"
    assert not state.segment.exists()


@pytest.mark.parametrize("timing", ["before_first_call", "during_call", "after_success", "after_unlink_failure"])
def test_lifecycle_replacement_is_never_claimed(lifecycle_owner, timing):
    state = lifecycle_owner
    if timing == "before_first_call":
        for child in state.children:
            child.exitcode = 0
        _replace_lifecycle_segment(state)
    elif timing == "during_call":
        state.during_shutdown = lambda: _replace_lifecycle_segment(state)
    elif timing == "after_success":
        state.owner.shutdown()
        assert not state.segment.exists()
        _replace_lifecycle_segment(state)
    else:
        with mock.patch.object(kv_utils.os, "unlink", side_effect=PermissionError("injected")):
            state.owner.shutdown()
        assert state.segment.exists()
        _replace_lifecycle_segment(state)

    assert state.owner.shutdown() == "upstream result"
    _assert_replacement_preserved(state)
    state.during_shutdown = lambda: None
    state.owner.shutdown()
    _assert_replacement_preserved(state)


def test_lifecycle_capture_failure_cannot_claim_replacement_on_retry(lifecycle_owner):
    state = lifecycle_owner
    with mock.patch("builtins.open", side_effect=PermissionError("injected capture failure")):
        state.owner = state.owner_class()
        assert state.owner.shutdown() == "upstream result"
    assert state.segment.exists()
    _replace_lifecycle_segment(state)
    assert state.owner.shutdown() == "upstream result"
    _assert_replacement_preserved(state)


@pytest.mark.parametrize("failed_call", ["stat", "fstat", "unlink"])
def test_lifecycle_identity_or_unlink_failure_preserves_retry(lifecycle_owner, failed_call):
    state = lifecycle_owner
    with mock.patch.object(kv_utils.os, failed_call, side_effect=PermissionError("injected")):
        assert state.owner.shutdown() == "upstream result"
    assert state.segment.read_bytes() == b"original generation"
    assert os.path.samestat(state.segment.stat(), state.original_identity)
    assert state.owner.shutdown() == "upstream result"
    assert not state.segment.exists()


@pytest.mark.parametrize("upstream_raises", [False, True])
def test_lifecycle_cleanup_failure_preserves_upstream_contract(lifecycle_owner, upstream_raises):
    state = lifecycle_owner
    error = RuntimeError("upstream shutdown failed")
    state.upstream_error = error if upstream_raises else None
    with mock.patch.object(kv_utils.os, "unlink", side_effect=PermissionError("injected")):
        if upstream_raises:
            with pytest.raises(RuntimeError) as caught:
                state.owner.shutdown()
            assert caught.value is error
        else:
            assert state.owner.shutdown() == "upstream result"
    assert state.segment.read_bytes() == b"original generation"
    state.upstream_error = None
    state.owner.shutdown()
    assert not state.segment.exists()


@pytest.mark.parametrize("last_to_exit", [0, 1, 2, 3])
def test_lifecycle_waits_for_every_owned_process(lifecycle_owner, last_to_exit):
    """Process-state coverage, not a claim of TP/PP GPU coverage."""
    state = lifecycle_owner
    for child in state.children:
        child.exitcode = 0
    state.children[last_to_exit].exitcode = None
    state.stop_children = False

    assert state.owner.shutdown() == "upstream result"
    assert state.segment.read_bytes() == b"original generation"
    state.children[last_to_exit].exitcode = -9
    assert state.owner.shutdown() == "upstream result"
    assert not state.segment.exists()


@pytest.mark.parametrize("failed_call", ["stat", "fstat"])
def test_identity_error_warns_and_does_not_unlink(monkeypatch, tmp_path, failed_call):
    segment = tmp_path / "identity_failure"
    segment.write_bytes(b"original")
    cleanup = kv_utils.IPCSegmentCleanup(str(segment))
    logger = mock.Mock()
    monkeypatch.setattr(kv_utils, "get_kvcached_logger", lambda: logger)
    with mock.patch.object(kv_utils.os, failed_call, side_effect=PermissionError("injected")):
        assert cleanup.unlink() is False
    logger.warning.assert_called_once()
    assert segment.read_bytes() == b"original"
    assert cleanup.unlink() is True
    assert not segment.exists()


def test_absent_identity_never_claims_later_file(tmp_path):
    segment = tmp_path / "initially_absent"
    cleanup = kv_utils.IPCSegmentCleanup(str(segment))
    segment.write_bytes(b"later generation")
    assert cleanup.unlink() is True
    assert segment.read_bytes() == b"later generation"


@pytest.mark.skipif(sys.platform != "linux", reason="Linux descriptor lifecycle")
def test_cleanup_releases_pinned_descriptor_without_inheriting_it(tmp_path):
    segment = tmp_path / "descriptor"
    segment.write_bytes(b"original")
    cleanup = kv_utils.IPCSegmentCleanup(str(segment))
    assert cleanup._file is not None
    fd = cleanup._file.fileno()
    assert not os.get_inheritable(fd)
    assert cleanup.unlink() is True
    with pytest.raises(OSError):
        os.fstat(fd)
    assert cleanup.unlink() is True


def test_concurrent_cleanup_of_one_identity_is_idempotent(monkeypatch, tmp_path):
    segment = tmp_path / "concurrent"
    segment.write_bytes(b"original")
    cleanup = kv_utils.IPCSegmentCleanup(str(segment))
    entered = threading.Event()
    release = threading.Event()
    errors = []
    results = []
    real_unlink = kv_utils.os.unlink

    def unlink(path):
        if threading.current_thread() is first:
            entered.set()
            assert release.wait(5)
        return real_unlink(path)

    def run():
        try:
            results.append(cleanup.unlink())
        except Exception as error:
            errors.append(error)

    monkeypatch.setattr(kv_utils.os, "unlink", unlink)
    first = threading.Thread(target=run)
    second = threading.Thread(target=run)
    first.start()
    try:
        assert entered.wait(5)
        second.start()
        # Give an unguarded second cleanup a deterministic chance to finish.
        # A serialized implementation may instead wait for the first cleanup.
        second.join(timeout=0.2)
    finally:
        release.set()
        first.join(timeout=5)
        if second.ident is not None:
            second.join(timeout=5)
    assert not first.is_alive() and not second.is_alive()
    assert not errors, errors
    assert results == [True, True]
    assert not segment.exists()


@pytest.fixture
def ready_pool_factory(monkeypatch, tmp_path, vllm_modules):
    """Run the real Python constructor/post-init; stub only native allocation."""
    interfaces, _ = vllm_modules
    module = _manager_module()
    monkeypatch.setattr(kv_utils, "SHM_DIR", str(tmp_path))
    monkeypatch.setattr(module, "DEFAULT_IPC_NAME", "pool_lifecycle_segment")
    monkeypatch.setattr(module, "kv_tensors_created", lambda **kwargs: True)
    monkeypatch.setattr(interfaces, "should_use_worker_ipc", lambda: False)
    segment = tmp_path / "pool_lifecycle_segment"
    pools = []
    active = set()
    cleanup = None

    def native_allocator(*args, **kwargs):
        nonlocal cleanup
        # The native constructor uses an explicit ipc_name unchanged, even for
        # nonzero group_id. Shared pool names must be handled as shared ownership.
        assert kwargs["ipc_name"] == segment.name
        segment.write_bytes(b"pool generation")
        if cleanup is None:
            cleanup = kv_utils.IPCSegmentCleanup(str(segment))
        allocator = mock.Mock()
        active.add(allocator)

        def release():
            active.discard(allocator)
            return bool(active) or cleanup.unlink()

        allocator.release_shared_segment.side_effect = release
        return allocator

    monkeypatch.setattr(module, "PageAllocator", native_allocator)

    def create(group_id=0):
        manager = module.KVCacheManager(
            num_blocks=128, block_size=16, cell_size=128, num_layers=1,
            reserve_null_block=False, group_id=group_id,
        )
        pools.append(manager)
        assert manager._post_init_done.wait(5)
        manager.page_allocator.start_prealloc_thread.assert_called_once_with()
        return manager, segment

    yield create, interfaces
    for pool in pools:
        pool.page_allocator.stop_prealloc_thread.side_effect = None
        pool.shutdown()


def test_ready_pool_first_shutdown_preserves_replacement(ready_pool_factory):
    create, _ = ready_pool_factory
    pool, segment = create()
    with segment.open("rb") as original:
        segment.unlink()
        segment.write_bytes(b"replacement pool")
        replacement = segment.stat()
        assert not os.path.samestat(os.fstat(original.fileno()), replacement)
        assert pool.shutdown() is True
        assert segment.exists(), "pool captured a replacement at first shutdown"
        assert os.path.samestat(segment.stat(), replacement)
        assert segment.read_bytes() == b"replacement pool"


def test_registry_keeps_shared_segment_until_every_pool_stops(
    ready_pool_factory, monkeypatch,
):
    create, interfaces = ready_pool_factory
    first, segment = create(group_id=0)
    second, second_segment = create(group_id=1)
    assert segment == second_segment
    monkeypatch.setattr(interfaces, "_kvcached_initialized", True)
    native_shutdown = mock.Mock()
    monkeypatch.setattr(interfaces, "_shutdown_kvcached_impl", native_shutdown)
    register_kv_cache_pool(first, integration="vllm")
    register_kv_cache_pool(second, integration="vllm")
    second.page_allocator.stop_prealloc_thread.side_effect = [RuntimeError("busy pool"), None]

    interfaces.shutdown_kvcached()
    native_shutdown.assert_not_called()
    assert interfaces._kvcached_initialized is True
    assert segment.exists(), "one pool removed a segment another pool still uses"
    assert segment.read_bytes() == b"pool generation"
    assert len(get_registered_kv_cache_pools(integration="vllm")) == 2

    interfaces.shutdown_kvcached()
    assert not segment.exists()
    native_shutdown.assert_called_once_with()
    first.page_allocator.stop_prealloc_thread.assert_called_once_with()
    assert second.page_allocator.stop_prealloc_thread.call_count == 2
    assert not get_registered_kv_cache_pools(integration="vllm")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
