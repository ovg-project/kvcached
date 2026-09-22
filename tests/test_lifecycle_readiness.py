# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Lifecycle readiness and failure propagation, phase 1 (issue #375, item 5).

``KVCacheManager._post_init()`` runs in a daemon thread. Before this surface
an exception there died with the thread while ``_post_init_done`` still
opened the gate, so callers walked into a pool with no null block and no
prealloc thread. These tests pin the poll-only surface: the phase machine,
the ``wait_ready()`` gate that re-raises the background error, the broadcast
rule as shipped in phase 1 (an unmap broadcast failure, a timeout included,
is an unknown cross-rank outcome => DEGRADED; a map broadcast failure never
transitions, because it may be the expected recoverable co-tenancy capacity
miss that ``_alloc()`` rolls back and reports as a scheduling miss, #453),
the ``clear()`` window, the snapshot field, and the capability flag.

CPU-only: ``kvcached.vmm_ops`` is stubbed when the compiled extension is
unavailable, and ``PageAllocator`` is swapped for a fake in the tests that
run the real ``__init__``.
"""
from __future__ import annotations

import contextlib
import gc
import json
import os
import shutil
import socket
import sys
import tempfile
import threading
import time
import traceback
import types
import weakref
from typing import Any, Callable, Dict, Iterator, List, Optional

import pytest


def _install_vmm_ops_stub() -> None:
    stub = types.ModuleType("kvcached.vmm_ops")
    stub.PageAllocator = object  # type: ignore[attr-defined]
    stub.InternalPage = object  # type: ignore[attr-defined]
    stub.kv_tensors_created = lambda group_id=0: True  # type: ignore[attr-defined]
    stub.map_to_kv_tensors = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    stub.unmap_from_kv_tensors = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    sys.modules["kvcached.vmm_ops"] = stub


try:
    import kvcached.vmm_ops  # noqa: F401
except ImportError:
    _install_vmm_ops_stub()

import kvcached.kv_cache_manager as kcm  # noqa: E402
from kvcached import tp_ipc_util  # noqa: E402
from kvcached.errors import StateConsistencyError  # noqa: E402
from kvcached.lifecycle import LifecyclePhase, LifecycleState  # noqa: E402
from kvcached.locks import NoOpLock  # noqa: E402
from kvcached.observability import get_capabilities  # noqa: E402


class FakePageAllocator:
    """Records what __init__ and clear() hand it; never touches a device.

    The read-only getters describe an empty pool so the observability
    snapshot can be built against it.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        self.map_callback: Optional[Callable[..., None]] = None
        self.unmap_callback: Optional[Callable[..., None]] = None
        self.calls: List[str] = []
        self.on_call: Dict[str, Callable[[], None]] = {}

    def _record(self, name: str) -> None:
        self.calls.append(name)
        hook = self.on_call.get(name)
        if hook is not None:
            hook()

    def set_use_worker_ipc(self, value: bool) -> None:
        self._record("set_use_worker_ipc")

    def set_broadcast_map_callback(self, callback: Callable[..., None]) -> None:
        self.map_callback = callback

    def set_broadcast_unmap_callback(self, callback: Callable[..., None]) -> None:
        self.unmap_callback = callback

    def start_prealloc_thread(self) -> None:
        self._record("start_prealloc_thread")

    def stop_prealloc_thread(self) -> None:
        self._record("stop_prealloc_thread")

    def free_pages(self, page_ids: List[int]) -> None:
        self._record("free_pages")

    def trim(self) -> None:
        self._record("trim")

    def reset_free_page_order(self) -> None:
        self._record("reset_free_page_order")

    def get_page_state(self) -> Dict[str, int]:
        return {"total_pages": 4, "free_pages": 4, "inuse_pages": 0,
                "reserved_pages": 0}

    def get_num_free_pages(self) -> int:
        return 4

    def get_num_reserved_pages(self) -> int:
        return 0

    def get_avail_physical_pages(self) -> int:
        return 4

    def get_resize_target(self) -> int:
        return 0


class FakeInternalPage:

    @staticmethod
    def get_num_blocks(page_size: int, block_mem_size: int) -> int:
        return page_size // block_mem_size


class FakePage:
    """Just enough of the C++ InternalPage for _alloc()'s page loop."""

    def __init__(self, page_id: int, num_blocks: int):
        self.page_id = page_id
        self._free = [page_id * num_blocks + i for i in range(num_blocks)]

    def init(self, block_mem_size: int) -> None:
        pass

    def num_free_blocks(self) -> int:
        return len(self._free)

    def alloc(self, need: int) -> List[int]:
        taken, self._free = self._free[:need], self._free[need:]
        return taken

    def full(self) -> bool:
        return not self._free


class MapThroughPageAllocator(FakePageAllocator):
    """alloc_page() with the C++ contract (csrc/page_allocator.cpp): run the
    registered map broadcast callback and, when it raises, put the page back
    and rethrow as the RuntimeError that ``_alloc()`` classifies. Since #418
    the native side passes ``StateConsistencyError`` through typed
    (csrc/ftensor.cpp keeps the failed page and rethrows), so the fake
    does the same."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.next_page_id = 0

    def alloc_page(self) -> FakePage:
        page_id, self.next_page_id = self.next_page_id, self.next_page_id + 1
        try:
            assert self.map_callback is not None
            self.map_callback(2, [page_id], 0, 0)
        except StateConsistencyError:
            raise
        except Exception as e:
            self.next_page_id = page_id  # the page goes back on the free list
            raise RuntimeError(f"Failed to map page {page_id}: {e}")
        return FakePage(page_id, num_blocks=2)


def _make_manager(
    monkeypatch: pytest.MonkeyPatch,
    *,
    broadcast_map: Optional[Callable[..., None]] = None,
    broadcast_unmap: Optional[Callable[..., None]] = None,
    allocator: type = FakePageAllocator,
) -> kcm.KVCacheManager:
    """Run the real __init__ (and its _post_init thread) against fakes.

    world_size=2 registers the broadcast callbacks and routes the KV-tensor
    check through ``broadcast_kv_tensors_created``, which is patched here.
    The broadcast functions are patched on ``tp_ipc_util`` before
    construction because __init__ binds them by local import.
    """
    monkeypatch.setattr(kcm, "PageAllocator", allocator)
    monkeypatch.setattr(kcm, "InternalPage", FakeInternalPage)
    monkeypatch.setattr(kcm, "broadcast_kv_tensors_created",
                        lambda *args, **kwargs: True)
    monkeypatch.setattr(kcm, "kv_tensors_created", lambda group_id=0: True)
    if broadcast_map is not None:
        monkeypatch.setattr(tp_ipc_util, "broadcast_map_to_kv_tensors", broadcast_map)
    if broadcast_unmap is not None:
        monkeypatch.setattr(tp_ipc_util, "broadcast_unmap_from_kv_tensors", broadcast_unmap)
    return kcm.KVCacheManager(
        num_blocks=4,
        block_size=1,
        cell_size=16,
        num_layers=1,
        world_size=2,
        pool_name="unified",
    )


def _bare_manager() -> kcm.KVCacheManager:
    """A manager without __init__: only what _post_init/wait_ready read."""
    manager = object.__new__(kcm.KVCacheManager)
    manager.world_size = 2
    manager.pp_rank = 0
    manager.group_id = 0
    manager.reserve_null_block = False
    manager.null_block = None
    manager.page_allocator = FakePageAllocator()
    manager._lock = NoOpLock()
    manager._post_init_done = threading.Event()
    manager._lifecycle = LifecycleState("bare")
    return manager


def _run_post_init(manager: kcm.KVCacheManager) -> threading.Thread:
    """Mirror __init__'s daemon thread. The error is swallowed here exactly
    as the thread machinery swallows it in production, minus the excepthook
    noise pytest would otherwise report."""

    def target() -> None:
        try:
            manager._post_init()
        except Exception:
            pass

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    return thread


def _raise_broadcast(message: str) -> Callable[..., None]:
    def broadcast(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError(message)

    return broadcast


@contextlib.contextmanager
def _silent_worker() -> Iterator[str]:
    """A unix socket standing in for a TP worker that accepts and never
    answers: alive but stuck, the KVCACHED_IPC_TIMEOUT case. The socket lives
    directly under /tmp because pytest's tmp_path can exceed the AF_UNIX path
    limit on macOS."""
    sock_dir = tempfile.mkdtemp(prefix="kvl-", dir="/tmp")
    sock_path = os.path.join(sock_dir, "w0.sock")
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(sock_path)
    server.listen(4)
    server.settimeout(0.1)
    stop = threading.Event()
    held: List[socket.socket] = []

    def serve() -> None:
        while not stop.is_set():
            try:
                conn, _ = server.accept()
            except socket.timeout:
                continue
            held.append(conn)

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    try:
        yield sock_path
    finally:
        stop.set()
        thread.join(timeout=2)
        for conn in held:
            conn.close()
        server.close()
        shutil.rmtree(sock_dir, ignore_errors=True)


# --------------------------------------------------------------------------
# Post-init and the wait_ready() gate (rule 2)
# --------------------------------------------------------------------------


def test_post_init_success_reaches_ready(monkeypatch):
    manager = _make_manager(monkeypatch)
    manager.wait_ready(timeout=5)
    assert manager.lifecycle_phase is LifecyclePhase.READY
    assert manager.lifecycle_error is None
    # Settled implies the legacy soft gate is already open.
    assert manager._post_init_done.is_set()
    assert "start_prealloc_thread" in manager.page_allocator.calls
    assert manager.page_allocator.map_callback is not None


def test_post_init_failure_is_re_raised_by_wait_ready(monkeypatch):
    monkeypatch.setattr(kcm, "KV_TENSOR_WAIT_TIMEOUT", 0.02)
    monkeypatch.setattr(kcm, "broadcast_kv_tensors_created",
                        lambda *args, **kwargs: False)
    manager = _bare_manager()
    thread = _run_post_init(manager)

    with pytest.raises(TimeoutError, match="KV tensors not created") as excinfo:
        manager.wait_ready(timeout=5)
    thread.join(timeout=5)

    assert manager.lifecycle_phase is LifecyclePhase.FAILED
    # The very object raised in the background thread, traceback included.
    assert excinfo.value is manager.lifecycle_error
    frames = [frame.name for frame in traceback.extract_tb(excinfo.value.__traceback__)]
    assert "_post_init" in frames
    # Record-only: the soft gate still opens for the existing entry points.
    assert manager._post_init_done.is_set()
    manager._wait_post_init()
    # And the gate keeps raising for every later caller.
    with pytest.raises(TimeoutError, match="KV tensors not created"):
        manager.wait_ready()
    assert "start_prealloc_thread" not in manager.page_allocator.calls


def test_post_init_ipc_error_is_re_raised_by_wait_ready(monkeypatch):
    """Issue #471's shape: the worker IPC check itself raises.

    Since #373 the wait loop retries a raising check instead of dying on
    the first raise, so the failure surfaces as the wait TimeoutError
    naming the last IPC error, and that is what FAILED records.
    """

    def unreachable(*args: Any, **kwargs: Any) -> bool:
        raise RuntimeError(
            "Worker pp0/rank0 failed to check KV tensors created: [Errno 2] "
            "No such file or directory")

    monkeypatch.setattr(kcm, "KV_TENSOR_WAIT_TIMEOUT", 0.02)
    monkeypatch.setattr(kcm, "broadcast_kv_tensors_created", unreachable)
    manager = _bare_manager()
    thread = _run_post_init(manager)

    with pytest.raises(TimeoutError,
                       match="failed to check KV tensors created") as excinfo:
        manager.wait_ready(timeout=5)
    thread.join(timeout=5)
    assert manager.lifecycle_phase is LifecyclePhase.FAILED
    assert excinfo.value is manager.lifecycle_error
    assert manager._post_init_done.is_set()


def test_repeated_wait_ready_on_a_failed_pool_keeps_the_traceback_flat(monkeypatch):
    """A poller that keeps gating on a failed pool re-raises the same stored
    exception every time, and every raise used to append the caller's frames
    to its traceback (1 -> 201 frames over 100 reads), retaining those
    frames' locals for as long as FAILED holds the error. The re-raise now
    rewinds to the traceback recorded at failure time first."""
    monkeypatch.setattr(kcm, "KV_TENSOR_WAIT_TIMEOUT", 0.02)
    monkeypatch.setattr(kcm, "broadcast_kv_tensors_created",
                        lambda *args, **kwargs: False)
    manager = _bare_manager()
    thread = _run_post_init(manager)
    with pytest.raises(TimeoutError, match="KV tensors not created"):
        manager.wait_ready(timeout=5)
    thread.join(timeout=5)

    def stored_frames() -> int:
        error = manager.lifecycle_error
        assert error is not None
        return len(traceback.extract_tb(error.__traceback__))

    after_first = stored_frames()
    for _ in range(100):
        with pytest.raises(TimeoutError) as excinfo:
            manager.wait_ready()
        assert excinfo.value is manager.lifecycle_error
    assert stored_frames() == after_first
    # The rewound traceback still names the original failure site.
    names = [
        frame.name
        for frame in traceback.extract_tb(manager.lifecycle_error.__traceback__)
    ]
    assert "_post_init" in names


def test_wait_ready_times_out_while_initializing():
    manager = _bare_manager()
    with pytest.raises(TimeoutError, match="still initializing"):
        manager.wait_ready(timeout=0.02)
    assert manager.lifecycle_phase is LifecyclePhase.INITIALIZING


def test_wait_ready_does_not_take_the_manager_lock():
    """The init thread holds the manager lock while reserving the null block,
    so a gate that took it would deadlock a caller in async-sched mode."""
    manager = _bare_manager()
    lock = threading.RLock()
    manager._lock = lock
    manager._lifecycle.mark_ready()
    passed = threading.Event()

    def gate() -> None:
        manager.wait_ready(timeout=2)
        passed.set()

    with lock:
        threading.Thread(target=gate, daemon=True).start()
        assert passed.wait(timeout=2)


# --------------------------------------------------------------------------
# Broadcast failures through the registered callbacks (rule 1)
# --------------------------------------------------------------------------


def test_map_broadcast_failure_does_not_change_lifecycle(monkeypatch):
    """A failed map broadcast records nothing in phase 1: at this layer it
    is indistinguishable from the expected recoverable capacity miss (#453),
    so classification waits for #373's per-rank results (phase 2). The error
    still reaches the C++ caller unchanged."""
    manager = _make_manager(
        monkeypatch,
        broadcast_map=_raise_broadcast("Worker 1 failed to map: did not answer"))
    manager.wait_ready(timeout=5)
    callback = manager.page_allocator.map_callback
    assert callback is not None

    with pytest.raises(RuntimeError, match="did not answer"):
        callback(2, [0], 0, 0)

    assert manager.lifecycle_phase is LifecyclePhase.READY
    assert manager.lifecycle_error is None
    manager.wait_ready()  # no sticky state


def test_capacity_miss_through_alloc_stays_ready_and_recovers(monkeypatch):
    """The #453 contract through the real _alloc() path: a colocated
    instance consuming the remaining physical pool surfaces as a worker map
    failure inside alloc_page(). _alloc() rolls back and reports a
    scheduling miss (None), the phase stays READY, and the next attempt can
    allocate once pressure clears."""
    pressure = {"on": True}

    def broadcast_map(*args: Any, **kwargs: Any) -> None:
        if pressure["on"]:
            raise RuntimeError(
                "Worker 0 failed to map: {'status': 'error', "
                "'message': 'CUDA error: out of memory'}")

    manager = _make_manager(monkeypatch, broadcast_map=broadcast_map,
                            allocator=MapThroughPageAllocator)
    manager.wait_ready(timeout=5)

    assert manager.alloc(1) is None  # rolled back, reported as a miss
    assert manager.lifecycle_phase is LifecyclePhase.READY
    assert manager.lifecycle_error is None

    pressure["on"] = False  # the colocated instance released capacity
    blocks = manager.alloc(1)
    assert blocks is not None and len(blocks) == 1
    assert manager.lifecycle_phase is LifecyclePhase.READY


def test_unmap_broadcast_failure_degrades(monkeypatch):
    manager = _make_manager(
        monkeypatch,
        broadcast_unmap=_raise_broadcast("Worker 1 failed to unmap: boom"))
    manager.wait_ready(timeout=5)
    callback = manager.page_allocator.unmap_callback
    assert callback is not None
    with pytest.raises(RuntimeError, match="boom") as excinfo:
        callback(2, [0])
    assert manager.lifecycle_phase is LifecyclePhase.DEGRADED
    assert manager.lifecycle_error is excinfo.value
    manager.wait_ready()  # still serving


def test_unmap_state_consistency_error_fails_the_pool(monkeypatch):
    """#418's transactions give the unmap broadcast one typed outcome
    stronger than unknown: StateConsistencyError means containment could
    not be established and the affected engine must stop, so the pool
    records FAILED rather than the still-serving DEGRADED."""
    error = StateConsistencyError(
        "state_consistency_unknown: KV unmap commit could not be confirmed")

    def broadcast_unmap(*args: Any, **kwargs: Any) -> None:
        raise error

    manager = _make_manager(monkeypatch, broadcast_unmap=broadcast_unmap)
    manager.wait_ready(timeout=5)
    callback = manager.page_allocator.unmap_callback
    assert callback is not None
    with pytest.raises(StateConsistencyError) as excinfo:
        callback(2, [0])
    assert excinfo.value is error
    assert manager.lifecycle_phase is LifecyclePhase.FAILED
    assert manager.lifecycle_error is error
    with pytest.raises(StateConsistencyError):
        manager.wait_ready()


def test_alloc_state_consistency_error_fails_the_pool(monkeypatch):
    """Unlike the recoverable capacity miss above, StateConsistencyError
    out of alloc_page() is #418's definitive unsafe verdict. _alloc() must
    not roll it back into a scheduling miss: the pool records FAILED and
    the error stays fail-loud to the engine."""

    def broadcast_map(*args: Any, **kwargs: Any) -> None:
        raise StateConsistencyError(
            "state_consistency_unknown for workers with lost responses: "
            "pp0/rank1")

    manager = _make_manager(monkeypatch, broadcast_map=broadcast_map,
                            allocator=MapThroughPageAllocator)
    manager.wait_ready(timeout=5)

    with pytest.raises(StateConsistencyError) as excinfo:
        manager.alloc(1)
    assert manager.lifecycle_phase is LifecyclePhase.FAILED
    assert manager.lifecycle_error is excinfo.value
    with pytest.raises(StateConsistencyError):
        manager.wait_ready()


def _wait_collected(ref):
    # The post-init thread may still be returning after it opened the gate.
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        gc.collect()
        if ref() is None:
            return
        time.sleep(0.01)
    assert ref() is None


def test_retained_unmap_callback_does_not_own_failed_manager(monkeypatch):
    def invoke(manager):
        # Keep the manager in a frame reached by the saved error traceback.
        manager.page_allocator.unmap_callback(2, [0])

    def make_failed_manager():
        manager = _make_manager(
            monkeypatch, broadcast_unmap=_raise_broadcast("unmap lifetime fault"))
        manager.wait_ready(timeout=5)
        try:
            invoke(manager)
        except RuntimeError:
            pass
        assert manager.lifecycle_phase is LifecyclePhase.DEGRADED
        error = manager.lifecycle_error
        assert error is not None and error.__traceback__ is not None
        return (weakref.ref(manager), weakref.ref(manager._lifecycle),
                manager.page_allocator)

    manager_ref, lifecycle_ref, allocator = make_failed_manager()
    # An external callback owner models the strong reference hidden in the
    # native allocator. It must not root the lifecycle/error/manager cycle.
    _wait_collected(manager_ref)
    assert lifecycle_ref() is None
    assert allocator.unmap_callback is not None


def test_unmap_callback_still_propagates_after_lifecycle_is_gone(monkeypatch):
    calls = []

    def broadcast(*args):
        calls.append(args)
        if len(calls) > 1:
            raise RuntimeError("unmap after lifecycle collection")

    manager = _make_manager(monkeypatch, broadcast_unmap=broadcast)
    manager.wait_ready(timeout=5)
    callback = manager.page_allocator.unmap_callback
    manager_ref = weakref.ref(manager)
    del manager
    _wait_collected(manager_ref)

    callback(2, [0])
    with pytest.raises(RuntimeError, match="unmap after lifecycle collection"):
        callback(2, [0])
    assert calls == [(2, [0], 0, 0), (2, [0], 0, 0)]


def test_ipc_timeout_fails_unmap_but_not_map(monkeypatch):
    """The canonical rule-1 trigger, end to end: a worker that is alive but
    not answering (KVCACHED_IPC_TIMEOUT) through the real broadcast path.
    Since #418 that path is transactional, and an unmap whose outcome the
    prepare/abort/commit exchange cannot confirm surfaces as
    StateConsistencyError, the definitive stop-the-engine verdict, so the
    pool records FAILED. The unknown-outcome DEGRADED classification still
    covers untyped unmap failures (test_unmap_broadcast_failure_degrades).
    On map, the callback deliberately records nothing (indistinguishable
    from the #453 capacity miss at this layer); the typed verdict is
    classified where it is caught, in _alloc()
    (test_alloc_state_consistency_error_fails_the_pool)."""
    monkeypatch.setattr(tp_ipc_util, "IPC_TIMEOUT_S", 0.5)
    with _silent_worker() as sock_path:
        monkeypatch.setattr(tp_ipc_util, "get_worker_socket_path",
                            lambda rank, pp_rank=0: sock_path)
        manager = _make_manager(monkeypatch)
        manager.wait_ready(timeout=5)
        map_callback = manager.page_allocator.map_callback
        unmap_callback = manager.page_allocator.unmap_callback
        assert map_callback is not None and unmap_callback is not None

        with pytest.raises(RuntimeError, match="did not answer"):
            map_callback(1, [0], 0, 0)
        assert manager.lifecycle_phase is LifecyclePhase.READY

        with pytest.raises(StateConsistencyError,
                           match="did not answer") as excinfo:
            unmap_callback(1, [0])

    assert manager.lifecycle_phase is LifecyclePhase.FAILED
    assert manager.lifecycle_error is excinfo.value
    with pytest.raises(StateConsistencyError):
        manager.wait_ready()


# --------------------------------------------------------------------------
# clear() window
# --------------------------------------------------------------------------


def test_clear_reenters_initializing_and_returns_to_ready(monkeypatch):
    manager = _make_manager(monkeypatch)
    manager.wait_ready(timeout=5)
    seen: List[LifecyclePhase] = []
    manager.page_allocator.on_call["reset_free_page_order"] = (
        lambda: seen.append(manager.lifecycle_phase))

    manager.clear()

    assert seen == [LifecyclePhase.INITIALIZING]
    assert manager.lifecycle_phase is LifecyclePhase.READY
    assert manager.page_allocator.calls[-1] == "start_prealloc_thread"


def test_clear_failure_moves_to_failed(monkeypatch):
    manager = _make_manager(monkeypatch)
    manager.wait_ready(timeout=5)

    def explode() -> None:
        raise RuntimeError("prealloc thread did not start")

    manager.page_allocator.on_call["start_prealloc_thread"] = explode
    with pytest.raises(RuntimeError, match="prealloc thread did not start"):
        manager.clear()
    assert manager.lifecycle_phase is LifecyclePhase.FAILED
    with pytest.raises(RuntimeError, match="prealloc thread did not start"):
        manager.wait_ready()


def test_clear_keeps_a_degraded_pool_degraded(monkeypatch):
    """clear() on a DEGRADED pool re-enters INITIALIZING for the teardown
    window (wait_ready() must hold: the prealloc thread is stopped and
    mappings are being released), then settles back to DEGRADED with the
    original cause, not READY."""
    manager = _make_manager(
        monkeypatch, broadcast_unmap=_raise_broadcast("Worker 1 failed to unmap"))
    manager.wait_ready(timeout=5)
    callback = manager.page_allocator.unmap_callback
    assert callback is not None
    with pytest.raises(RuntimeError):
        callback(2, [0])
    assert manager.lifecycle_phase is LifecyclePhase.DEGRADED
    cause = manager.lifecycle_error
    assert cause is not None

    inside_clear = threading.Event()
    release = threading.Event()

    def hold() -> None:
        inside_clear.set()
        assert release.wait(timeout=5)

    manager.page_allocator.on_call["reset_free_page_order"] = hold
    settled: List[LifecyclePhase] = []

    def gated_consumer() -> None:
        manager.wait_ready(timeout=5)
        settled.append(manager.lifecycle_phase)

    clearer = threading.Thread(target=manager.clear, daemon=True)
    clearer.start()
    assert inside_clear.wait(timeout=5)

    # Blocked inside _clear_locked(): the readiness gate holds.
    assert manager.lifecycle_phase is LifecyclePhase.INITIALIZING
    with pytest.raises(TimeoutError):
        manager.wait_ready(timeout=0.05)
    waiter = threading.Thread(target=gated_consumer, daemon=True)
    waiter.start()
    waiter.join(timeout=0.2)
    assert waiter.is_alive()
    assert settled == []

    release.set()
    clearer.join(timeout=5)
    waiter.join(timeout=5)
    assert not clearer.is_alive() and not waiter.is_alive()

    # Settles back to DEGRADED, cause preserved.
    assert settled == [LifecyclePhase.DEGRADED]
    assert manager.lifecycle_phase is LifecyclePhase.DEGRADED
    assert manager.lifecycle_error is cause
    assert manager.page_allocator.calls[-1] == "start_prealloc_thread"


# --------------------------------------------------------------------------
# LifecycleState on its own
# --------------------------------------------------------------------------


def test_degraded_is_sticky_and_failed_wins():
    state = LifecycleState("t")
    first, second, fatal = RuntimeError("a"), RuntimeError("b"), RuntimeError("c")
    state.mark_ready()
    assert state.phase is LifecyclePhase.READY

    state.mark_degraded("first", first)
    state.mark_degraded("second", second)
    assert state.phase is LifecyclePhase.DEGRADED
    assert state.error is first
    assert state.reason == "first"

    state.mark_ready()
    assert state.phase is LifecyclePhase.DEGRADED

    state.mark_failed("fatal", fatal)
    state.mark_failed("later")
    state.mark_degraded("ignored")
    state.mark_ready()
    assert state.phase is LifecyclePhase.FAILED
    assert state.error is fatal
    with pytest.raises(RuntimeError, match="^c$"):
        state.raise_if_failed()


def test_failed_without_an_error_object_still_raises():
    state = LifecycleState("t")
    state.mark_failed("no exception recorded")
    with pytest.raises(RuntimeError, match="no exception recorded"):
        state.raise_if_failed()


def test_repeated_raise_if_failed_does_not_grow_the_stored_traceback():
    state = LifecycleState("t")
    try:
        raise RuntimeError("boom during init")
    except RuntimeError as exc:
        state.mark_failed("init failed", exc)

    def stored_frames() -> int:
        error = state.error
        assert error is not None
        return len(traceback.extract_tb(error.__traceback__))

    recorded = stored_frames()
    with pytest.raises(RuntimeError, match="boom during init"):
        state.raise_if_failed()
    after_first = stored_frames()
    # One read appends only its own raise and call frames on top of the
    # recorded failure frame; without the rewind, 100 reads grew this
    # to recorded + 200.
    assert after_first == recorded + 2
    for _ in range(100):
        with pytest.raises(RuntimeError, match="boom during init") as excinfo:
            state.raise_if_failed()
        assert excinfo.value is state.error
    assert stored_frames() == after_first


def test_record_broadcast_failure_degrades_with_the_cause():
    state = LifecycleState("t")
    state.mark_ready()
    error = RuntimeError("Worker 1 failed to unmap: did not answer")
    state.record_broadcast_failure("unmap", error)
    assert state.phase is LifecyclePhase.DEGRADED
    assert state.error is error
    assert state.reason.startswith("unmap broadcast failed")


def test_record_broadcast_failure_fails_on_state_consistency():
    state = LifecycleState("t")
    state.mark_ready()
    error = StateConsistencyError("KV unmap commit could not be confirmed")
    state.record_broadcast_failure("unmap", error)
    assert state.phase is LifecyclePhase.FAILED
    assert state.error is error
    assert state.reason.startswith("unmap transaction unsafe")
    with pytest.raises(StateConsistencyError):
        state.raise_if_failed()


def test_degradation_during_initializing_lands_when_init_completes():
    """E.g. an unmap racing into the clear() window from another thread."""
    state = LifecycleState("t")
    error = RuntimeError("Worker 0 failed to unmap: did not answer")
    state.record_broadcast_failure("unmap", error)
    assert state.phase is LifecyclePhase.INITIALIZING
    assert not state.wait_settled(timeout=0.01)

    state.mark_ready()

    assert state.phase is LifecyclePhase.DEGRADED
    assert state.error is error
    assert state.wait_settled(timeout=0.01)


def test_begin_reinit_no_ops_while_initializing_or_failed():
    state = LifecycleState("t")
    state.begin_reinit()
    assert state.phase is LifecyclePhase.INITIALIZING  # unchanged
    state.mark_ready()
    state.begin_reinit()
    assert state.phase is LifecyclePhase.INITIALIZING
    state.mark_ready()
    state.mark_failed("x")
    state.begin_reinit()
    assert state.phase is LifecyclePhase.FAILED


def test_begin_reinit_carries_a_degraded_cause_through_the_window():
    state = LifecycleState("t")
    state.mark_ready()
    first = RuntimeError("Worker 1 failed to unmap")
    state.mark_degraded("unmap broadcast failed", first)

    state.begin_reinit()

    assert state.phase is LifecyclePhase.INITIALIZING
    assert not state.wait_settled(timeout=0.01)
    # A new degradation inside the window does not displace the first cause.
    state.mark_degraded("second", RuntimeError("later"))

    state.mark_ready()

    assert state.phase is LifecyclePhase.DEGRADED
    assert state.error is first
    assert state.reason == "unmap broadcast failed"
    assert state.wait_settled(timeout=0.01)


def test_wait_settled_wakes_waiters():
    state = LifecycleState("t")
    results: List[bool] = []
    waiter = threading.Thread(
        target=lambda: results.append(state.wait_settled(timeout=5)), daemon=True)
    waiter.start()
    state.mark_ready()
    waiter.join(timeout=5)
    assert results == [True]


def test_concurrent_degradations_keep_exactly_one_cause():
    state = LifecycleState("t")
    state.mark_ready()
    errors = [RuntimeError(str(i)) for i in range(16)]
    start = threading.Barrier(len(errors))

    def degrade(error: RuntimeError) -> None:
        start.wait()
        state.mark_degraded(str(error), error)

    threads = [threading.Thread(target=degrade, args=(e,)) for e in errors]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
    assert state.phase is LifecyclePhase.DEGRADED
    assert state.error in errors
    assert state.reason == str(state.error)


def test_phase_values_are_json_strings():
    assert json.dumps({"phase": LifecyclePhase.READY}) == '{"phase": "ready"}'
    assert LifecyclePhase("degraded") is LifecyclePhase.DEGRADED


# --------------------------------------------------------------------------
# Snapshot field and capability record
# --------------------------------------------------------------------------


def test_pool_snapshot_carries_lifecycle_phase(monkeypatch):
    manager = _make_manager(monkeypatch)
    manager.wait_ready(timeout=5)

    data = manager.observability_snapshot_dict(integration="vllm")
    assert data["lifecycle_phase"] == "ready"
    json.dumps(data)
    assert "lifecycle_phase" in get_capabilities()["pool_snapshot_fields"]

    manager._lifecycle.mark_degraded("unmap broadcast failed")
    assert manager.observability_snapshot_dict()["lifecycle_phase"] == "degraded"
    manager._lifecycle.mark_failed("post-initialization failed", RuntimeError("x"))
    assert manager.observability_snapshot_dict()["lifecycle_phase"] == "failed"


def test_capabilities_advertise_lifecycle_readiness():
    capabilities = get_capabilities()
    assert capabilities["schema_version"] == "kvcached.observability.v1"  # no bump
    assert capabilities["features"]["lifecycle_readiness"] is True
    json.dumps(capabilities)
