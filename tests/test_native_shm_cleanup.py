# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Exercise the real PageAllocator/MemInfoTracker destructor, not a mock."""

import gc
import os
import threading
import uuid
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("requires the GPU native extension", allow_module_level=True)

from kvcached.utils import IPCSegmentCleanup  # noqa: E402
from kvcached.vmm_ops import PageAllocator  # noqa: E402


def _allocator(ipc_name, group_id, pages=8):
    return PageAllocator(
        num_layers=1,
        mem_size_per_layer=pages * 2 * 1024 * 1024,
        page_size=2 * 1024 * 1024,
        world_size=1,
        pp_rank=0,
        async_sched=False,
        contiguous_layout=True,
        enable_page_prealloc=False,
        num_kv_buffers=1,
        group_id=group_id,
        ipc_name=ipc_name,
    )


@pytest.fixture(params=["relative", "absolute", "group"])
def segment(request, tmp_path, monkeypatch):
    name = f"kvcached-cleanup-{os.getpid()}-{uuid.uuid4().hex}"
    group_id = 0
    ipc_name = name
    path = Path("/dev/shm") / name
    if request.param == "absolute":
        path = tmp_path / name
        ipc_name = str(path)
    elif request.param == "group":
        monkeypatch.setenv("KVCACHED_IPC_NAME", name)
        ipc_name = ""
        group_id = 3
        path = path.with_name(f"{name}_g3")
    yield path, ipc_name, group_id
    path.unlink(missing_ok=True)


def test_native_cleanup_removes_original_segment(segment):
    path, ipc_name, group_id = segment
    allocator = _allocator(ipc_name, group_id)
    assert path.exists()
    del allocator
    gc.collect()
    assert not path.exists()


def test_delayed_native_cleanup_preserves_replacement(segment):
    path, ipc_name, group_id = segment
    old_allocator = _allocator(ipc_name, group_id)
    original = path.stat()
    cleanup = IPCSegmentCleanup(str(path))
    assert cleanup.unlink()
    assert not path.exists()

    replacement = _allocator(ipc_name, group_id, pages=16)
    replacement_stat = path.stat()
    contents = path.read_bytes()
    del old_allocator
    gc.collect()
    assert os.path.samestat(path.stat(), replacement_stat)
    assert path.read_bytes() == contents
    assert not os.path.samestat(original, replacement_stat)

    del replacement
    gc.collect()
    assert not path.exists()


def test_native_cleanup_tolerates_already_removed_segment(segment):
    path, ipc_name, group_id = segment
    allocator = _allocator(ipc_name, group_id)
    path.unlink()
    del allocator
    gc.collect()
    assert not path.exists()


def test_failed_initialization_does_not_claim_later_segment(tmp_path):
    path = tmp_path / "not-created-yet" / "segment"
    allocator = _allocator(str(path), 0)
    assert not path.exists()
    path.parent.mkdir()
    path.write_bytes(b"replacement")
    del allocator
    gc.collect()
    assert path.read_bytes() == b"replacement"


def test_native_identity_fd_is_closed_and_close_on_exec(tmp_path):
    path = tmp_path / "segment"
    allocator = _allocator(str(path), 0)
    identity = path.stat()
    owned_fds = []
    for entry in Path("/proc/self/fd").iterdir():
        try:
            fd = int(entry.name)
            if os.path.samestat(os.fstat(fd), identity):
                owned_fds.append(fd)
        except OSError:
            continue
    assert len(owned_fds) == 1
    assert not os.get_inheritable(owned_fds[0])
    del allocator
    gc.collect()
    with pytest.raises(OSError):
        os.fstat(owned_fds[0])
    assert not path.exists()


def test_shutdown_waits_for_native_pool_initialization(monkeypatch):
    from kvcached import kv_cache_manager as module
    from kvcached.integration.vllm import interfaces

    name = f"kvcached-init-cleanup-{os.getpid()}-{uuid.uuid4().hex}"
    path = Path("/dev/shm") / name
    monkeypatch.setattr(module, "DEFAULT_IPC_NAME", name)
    monkeypatch.setattr(interfaces, "should_use_worker_ipc", lambda: False)
    entered = threading.Event()
    release = threading.Event()

    def hold_readiness(**kwargs):
        entered.set()
        assert release.wait(5)
        return True

    monkeypatch.setattr(module, "kv_tensors_created", hold_readiness)
    manager = module.KVCacheManager(
        num_blocks=2048, block_size=16, cell_size=128, num_layers=1,
        async_sched=False, reserve_null_block=False,
    )
    try:
        assert entered.wait(5)
        assert manager.shutdown() is False
        assert path.exists()
        release.set()
        assert manager._post_init_done.wait(5)
        assert manager.shutdown() is True
        assert not path.exists()
    finally:
        release.set()
        manager._post_init_done.wait(5)
        manager.shutdown()
        path.unlink(missing_ok=True)
