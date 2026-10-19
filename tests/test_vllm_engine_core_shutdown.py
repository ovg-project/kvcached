# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Resource teardown and socket ownership survive control-hook removal.
Real control-file lifetime lives in test_control_segment; engine API doubles
here preserve #482/#543 exit and resource-order regressions.
"""

import importlib
import os
import shutil
import socket
import sys
import tempfile
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


def _parallel_config(tp_size, pp_size=1):
    """The part of vllm_config the socket capture reads."""
    return types.SimpleNamespace(parallel_config=types.SimpleNamespace(
        tensor_parallel_size=tp_size, pipeline_parallel_size=pp_size))


def _fake_client_module(shutdown=None, resources="owner", vllm_config=None):
    """A mock core_client module.

    The default FakeMPClient owns its engines (resources.engine_manager
    set), like the single-API MPClient that launched them. Pass any object
    to use it as the client's resources, or None for a client without the
    attribute. vllm_config, when given, is stored like the real client does.
    """
    client_mod = types.ModuleType("mock_client_mod")

    class FakeMPClient:
        def __init__(self):
            self._test_children = ()
            if vllm_config is not None:
                self.vllm_config = vllm_config
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
    page_allocator.release_shared_segment.return_value = True
    return manager


@pytest.fixture
def shm_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(kv_utils, "SHM_DIR", str(tmp_path))
    return tmp_path


def test_manager_shutdown_stops_prealloc_and_releases_usage(shm_dir):
    allocator = mock.Mock()
    manager = _make_manager(allocator)
    segment = shm_dir / manager.ipc_name
    segment.write_bytes(b"\0" * 24)

    manager.shutdown()

    allocator.stop_prealloc_thread.assert_called_once_with()
    assert segment.exists()


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


def test_manager_shutdown_retries_stop_before_releasing_usage(shm_dir):
    allocator = mock.Mock()
    allocator.stop_prealloc_thread.side_effect = [RuntimeError("join timed out"), None]
    manager = _make_manager(allocator)
    segment = shm_dir / manager.ipc_name
    segment.write_bytes(b"\0" * 24)

    assert manager.shutdown() is False
    assert segment.exists()

    assert manager.shutdown() is True
    assert segment.exists()
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
    assert segment.exists()


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
    assert segment.exists()
























def _fake_engine_utils_module(shutdown=None):
    """A mock vllm.v1.engine.utils module.

    FakeCoreEngineProcManager stands in for the manager that spawned the
    EngineCore processes. In the supervisor under --api-server-count > 1
    and in headless mode there is no owning MPClient, and vLLM calls this
    manager's shutdown directly on exit.
    """
    utils_mod = types.ModuleType("mock_engine_utils_mod")

    class FakeCoreEngineProcManager:
        def __init__(self, processes=None, vllm_config=None):
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
















def test_engine_manager_without_shutdown_is_left_alone(vllm_modules):
    _, patches = vllm_modules
    utils_mod = _fake_engine_utils_module()

    assert patches.CoreEngineProcManagerPatch().patch_manager_shutdown(utils_mod)
    assert not hasattr(utils_mod.CoreEngineProcManager, "shutdown")






















def _replace_lifecycle_segment(state):
    state.segment.unlink(missing_ok=True)
    state.segment.write_bytes(b"replacement generation")
    state.replacement_identity = state.segment.stat()
    assert not os.path.samestat(state.original_identity, state.replacement_identity)


def _assert_replacement_preserved(state):
    assert state.segment.exists(), "old owner removed a replacement generation"
    assert os.path.samestat(state.segment.stat(), state.replacement_identity)
    assert state.segment.read_bytes() == b"replacement generation"




























def _bind_worker_socket(path):
    """A listening socket at *path*, as a live worker binds it."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(path)
    server.listen()
    return server


def _replace_worker_socket(path):
    """Bind a same-name replacement over *path* with a different inode: the
    new node is created beside the old one, so the OS cannot hand it the
    old inode number, and renamed into place like a restart's unlink+bind."""
    replacement = _bind_worker_socket(path + ".new")
    os.rename(path + ".new", path)
    return replacement


def _accepts_connection(path):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as probe:
        probe.settimeout(5)
        try:
            probe.connect(path)
        except ConnectionRefusedError:
            return False
    return True


@pytest.fixture(params=["client", "legacy_client", "supervisor"])
def socket_owner(request, monkeypatch, tmp_path, vllm_modules):
    """One ready owner of two engines whose TP=2 workers bound their IPC
    sockets before READY (issue #510). state.workers holds the listening
    sockets; closing one is what a killed worker leaves behind: the node
    stays, nobody accepts on it."""
    _, patches = vllm_modules
    monkeypatch.setattr(patches, "enable_kvcached", lambda: True)
    monkeypatch.setattr(kv_utils, "SHM_DIR", str(tmp_path))
    monkeypatch.setattr(kv_utils, "DEFAULT_IPC_NAME", "socket_segment")
    # Not tmp_path: unix socket paths are limited to ~104 characters on macOS.
    root = tempfile.mkdtemp(prefix="kvcached-test-", dir="/tmp")
    monkeypatch.setattr(kv_utils, "TP_SOCKET_DIR_ROOT", root)
    segment = tmp_path / "socket_segment"
    segment.write_bytes(b"engines")
    socket_dir = kv_utils.get_tp_socket_dir()
    paths = [os.path.join(socket_dir, f"w{rank}.sock") for rank in range(2)]
    children = [types.SimpleNamespace(exitcode=None, join=mock.Mock()) for _ in range(2)]
    state = types.SimpleNamespace(
        segment=segment, socket_dir=socket_dir, paths=paths, children=children,
        workers={}, stop_children=True, exitcode=-9, calls=0,
    )

    def bind_workers():
        for path in paths:
            state.workers[path] = _bind_worker_socket(path)

    def kill_workers(*which):
        for path in which or tuple(paths):
            state.workers.pop(path).close()

    state.bind_workers = bind_workers
    state.kill_workers = kill_workers

    def shutdown(self, *args, **kwargs):
        state.calls += 1
        if state.stop_children:
            for child in children:
                child.exitcode = state.exitcode
        return "upstream result"

    config = _parallel_config(2)
    if request.param == "supervisor":
        mod = _fake_engine_utils_module(shutdown)
        assert patches.CoreEngineProcManagerPatch().patch_manager_shutdown(mod)
        def owner_class():
            return mod.CoreEngineProcManager(processes=children, vllm_config=config)
    else:
        if request.param == "client":
            resources = types.SimpleNamespace(
                engine_manager=types.SimpleNamespace(processes=children))
        else:
            resources = types.SimpleNamespace(core_engines=[
                types.SimpleNamespace(proc_handle=child) for child in children])
        mod = _fake_client_module(shutdown, resources=resources, vllm_config=config)
        assert patches.MPClientPatch().patch_client_shutdown(mod)
        owner_class = mod.MPClient
    state.owner_class = owner_class
    try:
        yield state
    finally:
        for server in state.workers.values():
            server.close()
        shutil.rmtree(root, ignore_errors=True)


def test_group_sigterm_leaves_no_worker_socket_directory(socket_owner):
    """The process-group SIGTERM shape from the #482 two-GPU run: vLLM's
    process manager kills the engine tree before Worker.shutdown() runs,
    so every socket node stays with nobody listening. The owner that
    removes the segment must also remove them once the engines are gone."""
    state = socket_owner
    state.bind_workers()
    owner = state.owner_class()
    state.kill_workers()
    for path in state.paths:
        assert os.path.exists(path)

    assert owner.shutdown() == "upstream result"

    assert state.segment.exists()
    for path in state.paths:
        assert not os.path.exists(path), "a killed worker's socket was left behind"
    assert not os.path.exists(state.socket_dir)


def test_delayed_worker_keeps_its_socket_until_it_exits(socket_owner):
    """Engine exit alone is not enough (issue #510): a worker that is still
    alive after its engine died, stopped or delayed, keeps its socket, and
    the retry removes it once the worker is gone."""
    state = socket_owner
    state.bind_workers()
    owner = state.owner_class()
    state.kill_workers(state.paths[0])

    owner.shutdown()

    assert not os.path.exists(state.paths[0])
    assert os.path.exists(state.paths[1])
    assert _accepts_connection(state.paths[1])
    assert os.path.isdir(state.socket_dir)

    state.kill_workers(state.paths[1])
    owner.shutdown()
    assert not os.path.exists(state.paths[1])
    assert not os.path.exists(state.socket_dir)


@pytest.mark.parametrize("replacement_alive", [True, False])
def test_same_name_replacement_socket_is_never_claimed(socket_owner, replacement_alive):
    state = socket_owner
    state.bind_workers()
    owner = state.owner_class()
    state.kill_workers()
    replacement = _replace_worker_socket(state.paths[0])
    try:
        if not replacement_alive:
            replacement.close()
        owner.shutdown()
        assert not os.path.exists(state.paths[1])
        assert os.path.exists(state.paths[0]), "the replacement's socket was removed"
        assert _accepts_connection(state.paths[0]) is replacement_alive
        assert os.path.isdir(state.socket_dir)
        owner.shutdown()
        assert os.path.exists(state.paths[0])
    finally:
        replacement.close()


def test_live_engines_keep_worker_sockets_until_their_exit_is_confirmed(socket_owner):
    state = socket_owner
    state.bind_workers()
    owner = state.owner_class()
    state.kill_workers()
    state.stop_children = False

    owner.shutdown()
    for path in state.paths:
        assert os.path.exists(path)

    for child in state.children:
        child.exitcode = -9
    owner.shutdown()
    for path in state.paths:
        assert not os.path.exists(path)
    assert not os.path.exists(state.socket_dir)


def test_sockets_absent_at_ready_are_never_claimed(socket_owner):
    """Ownership is established at READY only: nodes that appear later
    belong to another generation, even when nobody serves them."""
    state = socket_owner
    owner = state.owner_class()
    state.bind_workers()
    state.kill_workers()

    owner.shutdown()

    assert state.segment.exists()
    for path in state.paths:
        assert os.path.exists(path)


def test_repeated_owner_shutdown_leaves_a_replacement_directory(socket_owner):
    """vLLM 0.30 reaches the owner's shutdown three times per teardown:
    MPClient.shutdown() calls the manager, BackgroundResources calls it
    again, and the client's own wrapper runs last. Once the first call has
    removed everything, a same-name replacement that created the directory
    again, before its bind, must keep it."""
    state = socket_owner
    state.bind_workers()
    owner = state.owner_class()
    state.kill_workers()
    owner.shutdown()
    assert not os.path.exists(state.socket_dir)

    os.makedirs(state.socket_dir)  # the replacement, before its bind
    owner.shutdown()
    owner.shutdown()

    assert os.path.isdir(state.socket_dir)
    state.bind_workers()  # the replacement's bind succeeds


def test_non_owning_frontend_leaves_worker_sockets_alone(monkeypatch, tmp_path, vllm_modules):
    """--api-server-count 2: a frontend's client stops no engines and must
    not touch the sockets, served or not."""
    _, patches = vllm_modules
    monkeypatch.setattr(patches, "enable_kvcached", lambda: True)
    monkeypatch.setattr(kv_utils, "SHM_DIR", str(tmp_path))
    monkeypatch.setattr(kv_utils, "DEFAULT_IPC_NAME", "frontend_segment")
    root = tempfile.mkdtemp(prefix="kvcached-test-", dir="/tmp")
    monkeypatch.setattr(kv_utils, "TP_SOCKET_DIR_ROOT", root)
    path = os.path.join(kv_utils.get_tp_socket_dir(), "w0.sock")
    try:
        _bind_worker_socket(path).close()
        mod = _fake_client_module(
            lambda self: None, resources=types.SimpleNamespace(engine_manager=None),
            vllm_config=_parallel_config(1))
        assert patches.MPClientPatch().patch_client_shutdown(mod)
        frontend = mod.MPClient()
        frontend.shutdown()
        assert os.path.exists(path)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_owner_without_parallel_config_preserves_control_and_sockets(
    monkeypatch, tmp_path, vllm_modules
):
    _, patches = vllm_modules
    monkeypatch.setattr(patches, "enable_kvcached", lambda: True)
    monkeypatch.setattr(kv_utils, "SHM_DIR", str(tmp_path))
    monkeypatch.setattr(kv_utils, "DEFAULT_IPC_NAME", "no_config_segment")
    root = tempfile.mkdtemp(prefix="kvcached-test-", dir="/tmp")
    monkeypatch.setattr(kv_utils, "TP_SOCKET_DIR_ROOT", root)
    segment = tmp_path / "no_config_segment"
    segment.write_bytes(b"engine")
    path = os.path.join(kv_utils.get_tp_socket_dir(), "w0.sock")
    try:
        _bind_worker_socket(path).close()
        mod = _fake_client_module(lambda self: None)
        assert patches.MPClientPatch().patch_client_shutdown(mod)
        owner = mod.MPClient()
        assert owner._kvcached_worker_sockets is None
        owner.shutdown()
        assert segment.exists()
        assert os.path.exists(path)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_client_reuses_supervisor_socket_capture(monkeypatch, vllm_modules):
    _, patches = vllm_modules
    monkeypatch.setattr(patches, "enable_kvcached", lambda: True)
    child = types.SimpleNamespace(exitcode=None)
    sockets = mock.Mock()
    manager = types.SimpleNamespace(
        processes=[child], _kvcached_engine_processes=(child,),
        _kvcached_owner_captured=True, _kvcached_worker_sockets=sockets,
    )
    mod = _fake_client_module(lambda self: setattr(child, "exitcode", 0),
                              resources=types.SimpleNamespace(engine_manager=manager))
    assert patches.MPClientPatch().patch_client_shutdown(mod)
    client = mod.MPClient()
    assert client._kvcached_worker_sockets is sockets
    client.shutdown()
    sockets.unlink.assert_called_once_with()



@pytest.mark.skipif(sys.platform == "win32", reason="POSIX fork and SIGTERM")
@pytest.mark.parametrize("boundary", ["client", "manager", "headless"])
@pytest.mark.parametrize("replacement", [False, True])
def test_default_sigterm_stops_real_owned_child_and_preserves_generation(
    monkeypatch, tmp_path, vllm_modules, boundary, replacement
):
    """Exercise real TERM and process handles through the actual owner patches.

    External vLLM constructors remain doubles. Headless capture really runs on
    its observer thread, so registration there alone cannot pass this test.
    """
    import multiprocessing as mp
    import signal
    import time

    _, patches = vllm_modules
    monkeypatch.setattr(patches, "enable_kvcached", lambda: True)
    monkeypatch.setattr(kv_utils, "SHM_DIR", str(tmp_path))
    monkeypatch.setattr(kv_utils, "DEFAULT_IPC_NAME", "signal_segment")
    segment = tmp_path / "signal_segment"
    context = mp.get_context("fork")
    reader, writer = context.Pipe(duplex=False)

    def run_worker():
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        time.sleep(60)

    def run_owner():
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        worker = context.Process(target=run_worker)
        worker.start()

        def shutdown(self):
            worker.kill()
            worker.join(5)

        def ready():
            if replacement:
                segment.unlink()
                segment.write_bytes(b"replacement")
            writer.send(worker.pid)
            writer.close()
            while True:
                signal.pause()

        if boundary == "headless":
            class Manager:
                def __init__(self, local_client=False):
                    self.processes = [worker]

                def monitor_engine_liveness(self):
                    ready()

            setattr(Manager, "shutdown", shutdown)
            mod: Any = types.SimpleNamespace(CoreEngineProcManager=Manager)
            assert patches.CoreEngineProcManagerPatch().patch_manager_shutdown(mod)
            owner = mod.CoreEngineProcManager(local_client=False)
            segment.write_bytes(b"original")
            owner.monitor_engine_liveness()
        else:
            segment.write_bytes(b"original")
            if boundary == "manager":
                mod = _fake_engine_utils_module(shutdown)
                assert patches.CoreEngineProcManagerPatch().patch_manager_shutdown(mod)
                owner = mod.CoreEngineProcManager(processes=[worker])
            else:
                resources = types.SimpleNamespace(
                    engine_manager=types.SimpleNamespace(processes=[worker]))
                mod = _fake_client_module(shutdown, resources=resources)
                assert patches.MPClientPatch().patch_client_shutdown(mod)
                owner = mod.MPClient()
            assert owner._kvcached_owner_captured is not None
            ready()

    process = context.Process(target=run_owner)
    process.start()
    writer.close()
    try:
        assert reader.poll(15), "owner never became ready"
        worker_pid = reader.recv()
        assert process.pid is not None
        os.kill(process.pid, signal.SIGTERM)
        process.join(15)
        assert process.exitcode == -signal.SIGTERM
        with pytest.raises(ProcessLookupError):
            os.kill(worker_pid, 0)
        assert segment.exists()
        if replacement:
            assert segment.read_bytes() == b"replacement"
    finally:
        reader.close()
        if process.is_alive():
            process.kill()
            process.join(5)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))


def test_socket_capture_failure_keeps_owner_sigterm(monkeypatch):
    import kvcached.utils as utils
    from kvcached.integration.vllm import patches
    monkeypatch.setenv("ENABLE_KVCACHED", "true")
    callbacks: list[Any] = []
    monkeypatch.setattr(utils, "register_owner_sigterm_cleanup", callbacks.append)
    def fail_capture(config):
        raise OSError("socket discovery failed")
    monkeypatch.setattr(patches, "_capture_worker_sockets", fail_capture)
    class Owner:
        def shutdown(self):
            pass
    owner: Any = Owner()
    process = types.SimpleNamespace(exitcode=None)
    patches._capture_engine_resources(owner, (process,))
    assert callbacks == [owner.shutdown]
    assert owner._kvcached_owner_captured
    assert owner._kvcached_worker_sockets is None
