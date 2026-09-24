# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Exercise the real PageAllocator/MemInfoTracker destructor, not a mock."""

import gc
import os
import shutil
import subprocess
import threading
import uuid
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("requires the GPU native extension", allow_module_level=True)

from kvcached.utils import IPCSegmentCleanup  # noqa: E402
from kvcached.vmm_ops import PageAllocator  # noqa: E402


def test_native_identity_syscall_failures_and_retries(tmp_path):
    from torch.utils.cpp_extension import CUDA_HOME, ROCM_HOME

    compiler = shutil.which("c++")
    toolkit = ROCM_HOME if torch.version.hip else CUDA_HOME
    if compiler is None or toolkit is None:
        pytest.skip("native fault injection requires the extension build toolchain")
    assert compiler is not None and toolkit is not None
    root = Path(__file__).resolve().parents[1]
    binary = tmp_path / "shm-cleanup-faults"
    subprocess.run([
        compiler, "-std=c++17", "-pthread",
        "-DKVCACHED_USE_ROCM" if torch.version.hip else "-DKVCACHED_USE_CUDA",
        "-I" + str(Path(toolkit) / "include"), "-I" + str(root / "csrc/inc"),
        str(root / "tests/native/shm_cleanup_faults.cpp"),
        "-Wl,--wrap=fcntl", "-Wl,--wrap=stat", "-Wl,--wrap=fstat", "-Wl,--wrap=unlink",
        "-o", str(binary),
    ], check=True, timeout=60)
    result = subprocess.run([str(binary), str(tmp_path / "segment")],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "6 native syscall faults" in result.stdout


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


def test_explicit_native_release_is_idempotent(segment):
    path, ipc_name, group_id = segment
    allocator = _allocator(ipc_name, group_id)
    allocator.stop_prealloc_thread()
    assert allocator.release_shared_segment() is True
    assert not path.exists()
    replacement = _allocator(ipc_name, group_id)
    identity = path.stat()
    assert allocator.release_shared_segment() is True
    del allocator
    gc.collect()
    assert os.path.samestat(path.stat(), identity)
    del replacement
    gc.collect()
    assert not path.exists()


@pytest.mark.parametrize("explicit_release", [False, True])
def test_native_shared_name_waits_for_last_pool(tmp_path, explicit_release):
    path = tmp_path / "shared-pools"
    first = _allocator(str(path), 0)
    second = _allocator(str(path), 1)
    identity = path.stat()
    if explicit_release:
        first.stop_prealloc_thread()
        assert first.release_shared_segment() is True
    del first
    gc.collect()
    assert os.path.samestat(path.stat(), identity)
    second.stop_prealloc_thread()
    assert second.release_shared_segment() is True
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
