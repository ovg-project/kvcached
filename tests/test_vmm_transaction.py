# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import pytest


@pytest.mark.parametrize(("rollback_failure", "background"), [(False, False), (True, False), (False, True)])
def test_native_map_exception_is_terminal_and_rolls_back_prior_targets(tmp_path, rollback_failure, background):
    import os
    import subprocess
    import sys

    _compiled_vmm_ops()
    if sys.platform != "linux":
        pytest.skip("CUDA driver fault injection uses Linux LD_PRELOAD")
    source = tmp_path / "map_fault.c"
    library = tmp_path / "map_fault.so"
    source.write_text(r'''
#include <dlfcn.h>
#include <stddef.h>
static int remaining = -1, fail_rollback = 0, map_failed = 0;
void arm_map_failure(int rollback) {
    remaining = 1; fail_rollback = rollback; map_failed = 0;
}
int cuMemMap(unsigned long long ptr, size_t size, size_t offset,
             unsigned long long handle, unsigned long long flags) {
    if (remaining == 0) { remaining = -1; map_failed = 1; return 2; }
    if (remaining > 0) --remaining;
    void *cuda = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    int (*real_map)(unsigned long long, size_t, size_t, unsigned long long,
                    unsigned long long) = dlsym(cuda, "cuMemMap");
    return real_map(ptr, size, offset, handle, flags);
}
int cuMemUnmap(unsigned long long ptr, size_t size) {
    if (map_failed && fail_rollback) { fail_rollback = 0; return 1; }
    void *cuda = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    int (*real_unmap)(unsigned long long, size_t) = dlsym(cuda, "cuMemUnmap");
    return real_unmap(ptr, size);
}
''', encoding="utf-8")
    subprocess.run(["cc", "-shared", "-fPIC", str(source), "-ldl", "-o", str(library)], check=True)
    scenario = r'''
import ctypes, sys, time, uuid
from kvcached import vmm_ops as v
page = 2 * 1024 * 1024
v.init_kvcached("cuda:0", page, False)
v.create_kv_tensors(8 * page, 2, "cuda:0", 2, 2, 0, False)
a = v.PageAllocator(2, 4 * page, page, 1, pp_rank=0, async_sched=False,
    contiguous_layout=False, enable_page_prealloc=bool(int(sys.argv[2])), num_kv_buffers=2,
    group_id=0, ipc_name="NATIVE_" + uuid.uuid4().hex[:8])
before = a.get_num_free_pages()
ctypes.CDLL(None).arm_map_failure(int(sys.argv[1]))
try:
    if int(sys.argv[2]):
        a.start_prealloc_thread()
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            a.get_avail_physical_pages()
            time.sleep(0.01)
    else:
        a.alloc_page()
except v.KVMappingStateError as error:
    assert not isinstance(error, RuntimeError)
    if int(sys.argv[1]):
        assert "rollback failed" in str(error), str(error)
else:
    raise AssertionError("native mapping failure was downgraded")
a.stop_prealloc_thread()
assert a.get_num_free_pages() == before
if not int(sys.argv[1]):
    assert v.map_to_kv_tensors_with_result([0], 0) == (True, [0])
del a
v.shutdown_kvcached()
print("NATIVE_MAP_FAILURE_OK", flush=True)
'''
    env = dict(os.environ, LD_PRELOAD=str(library))
    result = subprocess.run(
        [sys.executable, "-c", scenario, str(int(rollback_failure)), str(int(background))],
        env=env, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "NATIVE_MAP_FAILURE_OK" in result.stdout


@pytest.mark.parametrize("phase", ["prepare", "commit"])
def test_unknown_map_outcome_survives_native_callback_without_unmap(monkeypatch, phase):
    """Exercise Python -> C++ alloc_page -> Python IPC -> C++ -> Python."""
    import uuid

    from kvcached import tp_ipc_util as ipc

    vmm_ops = _compiled_vmm_ops()
    offsets, _ = _create_layout(vmm_ops, False, False)
    allocator = vmm_ops.PageAllocator(
        num_layers=2, mem_size_per_layer=8 * 1024 * 1024,
        page_size=2 * 1024 * 1024, world_size=1, pp_rank=0,
        async_sched=False, contiguous_layout=False, enable_page_prealloc=False,
        num_kv_buffers=2, group_id=0, ipc_name="UNKNOWN_" + uuid.uuid4().hex[:8],
    )
    allocator.set_use_worker_ipc(True)
    allocator.set_broadcast_map_callback(ipc.broadcast_map_to_kv_tensors)
    monkeypatch.setattr(ipc, "_UNRESOLVED_PHYSICAL_GROWTH_TRANSACTIONS", {})
    monkeypatch.setattr(ipc, "_PHYSICAL_DEVICE_ID_CACHE", {(0, 0): "gpu-0"})
    transactions = []
    commands = []

    async def exchange(rank, message, pp_rank=0):
        command = message["cmd"]
        commands.append(command)
        transaction = message["transaction_id"]
        if command == "prepare_map_to_kv_tensors":
            transactions.append(transaction)
            assert vmm_ops.prepare_map_to_kv_tensors(transaction, message["offsets"], 0)["success"]
            if phase == "prepare":
                raise ConnectionError("lost prepare response")
            return {"status": "success", "transaction_state": "reserved"}
        if command == "commit_prepared_map":
            assert vmm_ops.commit_prepared_map(transaction, 0)["success"]
            raise ConnectionError("lost commit response")
        if command == "get_map_transaction_state":
            raise ConnectionError("lost state query response")
        pytest.fail(f"unexpected cleanup or retry: {command}")

    monkeypatch.setattr(ipc, "_send_and_receive_message", exchange)
    try:
        before = allocator.get_num_free_pages()
        with pytest.raises(ipc.MapTransactionOutcomeUnknownError, match="restart"):
            allocator.alloc_page()
        assert allocator.get_num_free_pages() == before
        assert allocator.get_num_inuse_pages() == 0
        assert len(transactions) == 1
        assert commands.count("get_map_transaction_state") == 2
        calls = len(commands)
        with pytest.raises(ipc.MapTransactionOutcomeUnknownError):
            allocator.alloc_page()
        assert len(commands) == calls
        if phase == "prepare":
            assert vmm_ops.has_prepared_map(transactions[0], 0)
        else:
            # Idempotent native map reports no newly mapped offsets: the
            # original mapping was retained, not silently rolled back.
            assert vmm_ops.map_to_kv_tensors_with_result([offsets[0]], 0) == (True, [])
    finally:
        del allocator
        vmm_ops.shutdown_kvcached()


def test_native_allocator_preserves_recoverable_callback_error():
    import uuid

    vmm_ops = _compiled_vmm_ops()
    allocator = vmm_ops.PageAllocator(
        num_layers=1, mem_size_per_layer=8 * 1024 * 1024,
        page_size=2 * 1024 * 1024, world_size=1, pp_rank=0,
        async_sched=False, contiguous_layout=False, enable_page_prealloc=False,
        num_kv_buffers=2, group_id=0, ipc_name="MISS_" + uuid.uuid4().hex[:8],
    )
    allocator.set_use_worker_ipc(True)

    def fail(*args):
        raise RuntimeError("capacity_exhausted")

    allocator.set_broadcast_map_callback(fail)
    before = allocator.get_num_free_pages()
    with pytest.raises(RuntimeError, match="capacity_exhausted"):
        allocator.alloc_page()
    assert allocator.get_num_free_pages() == before
    assert allocator.get_num_inuse_pages() == 0


def test_background_unknown_outcome_reaches_manager_capacity_check(monkeypatch):
    import threading
    import uuid

    from kvcached import tp_ipc_util as ipc
    from kvcached.kv_cache_manager import KVCacheManager
    from kvcached.locks import NoOpLock

    vmm_ops = _compiled_vmm_ops()
    monkeypatch.setattr(ipc, "_UNRESOLVED_PHYSICAL_GROWTH_TRANSACTIONS", {})
    entered = threading.Event()
    allocator = vmm_ops.PageAllocator(
        num_layers=1, mem_size_per_layer=16 * 1024 * 1024,
        page_size=2 * 1024 * 1024, world_size=1, pp_rank=0,
        async_sched=False, contiguous_layout=False, enable_page_prealloc=True,
        num_kv_buffers=2, group_id=0, ipc_name="BACKGROUND_" + uuid.uuid4().hex[:8],
    )
    allocator.set_use_worker_ipc(True)

    def fail(*args):
        try:
            ipc._fail_unresolved_map_transaction("background", [0], 0, "prepare", ["lost"])
        finally:
            entered.set()

    allocator.set_broadcast_map_callback(fail)
    manager = object.__new__(KVCacheManager)
    manager._lock = NoOpLock()
    try:
        allocator.start_prealloc_thread()
        assert entered.wait(10), "preallocator did not exercise the callback"
        with pytest.raises(ipc.MapTransactionOutcomeUnknownError, match="background"):
            manager.available_size()
    finally:
        allocator.stop_prealloc_thread()


def _compiled_vmm_ops():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for VMM transaction validation")

    try:
        from kvcached import vmm_ops
    except ImportError:
        pytest.skip("kvcached.vmm_ops extension is not built")
    required_operations = (
        "init_kvcached",
        "create_kv_tensors",
        "map_to_kv_tensors_with_result",
        "prepare_map_to_kv_tensors",
        "commit_prepared_map",
        "abort_prepared_map",
        "has_prepared_map",
        "unmap_from_kv_tensors",
        "prepare_unmap_from_kv_tensors",
        "commit_unmap_from_kv_tensors",
        "abort_unmap_from_kv_tensors",
        "current_device_pci_bus_id",
        "shutdown_kvcached",
    )
    if any(not hasattr(vmm_ops, name) for name in required_operations):
        pytest.skip("a test stub replaced the compiled kvcached.vmm_ops module")
    return vmm_ops


def _create_layout(vmm_ops, contiguous_layout, unified_pool):
    page_size = 2 * 1024 * 1024
    num_layers = 2
    num_kv_buffers = 2
    vmm_ops.init_kvcached("cuda:0", page_size, contiguous_layout)
    vmm_ops.create_kv_tensors(
        8 * 1024 * 1024,
        2,
        "cuda:0",
        num_layers,
        num_kv_buffers,
        0,
        unified_pool,
    )
    stride = page_size * num_layers * num_kv_buffers if contiguous_layout else page_size
    return [0, stride], 32 * 1024 * 1024


def test_unified_direct_mapping_uses_actual_physical_bytes(tmp_path):
    import os
    import subprocess
    import sys

    _compiled_vmm_ops()
    if sys.platform != "linux":
        pytest.skip("driver memory-query injection requires Linux")
    source = tmp_path / "capacity.c"
    library = tmp_path / "capacity.so"
    source.write_text(r'''
#include <stddef.h>
#include <dlfcn.h>
static int enabled = 0;
void enable_capacity_limit(void) { enabled = 1; }
int cuMemGetInfo_v2(size_t *free_bytes, size_t *total_bytes) {
    if (enabled) {
        *free_bytes = 6 * 1024 * 1024;
        *total_bytes = 100 * 1024 * 1024;
        return 0;
    }
    void *cuda = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    int (*original)(size_t *, size_t *) = dlsym(cuda, "cuMemGetInfo_v2");
    return original(free_bytes, total_bytes);
}
''', encoding="utf-8")
    subprocess.run(["cc", "-shared", "-fPIC", str(source), "-ldl", "-o", str(library)], check=True)
    scenario = r'''
import ctypes
from kvcached import vmm_ops as v
page = 2 * 1024 * 1024
v.init_kvcached("cuda:0", page, False)
v.create_kv_tensors(4 * page, 2, "cuda:0", 2, 2, 0, True)
ctypes.CDLL(None).enable_capacity_limit()
assert v.map_to_kv_tensors_with_result([0], 0) == (True, [0])
assert v.unmap_from_kv_tensors([0], 0)
assert v.map_to_kv_tensors([page], 0)
assert v.unmap_from_kv_tensors([page], 0)
v.shutdown_kvcached()
'''
    env = dict(os.environ, LD_PRELOAD=str(library), KVCACHED_GPU_UTILIZATION="0.99")
    result = subprocess.run([sys.executable, "-c", scenario], env=env,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    ("contiguous_layout", "unified_pool"),
    [(True, False), (False, True), (False, False)],
    ids=["contiguous", "unified", "per-layer-kv"],
)
@pytest.mark.parametrize("failure_position", [0, 1, 2])
def test_cuda_map_batch_rolls_back_at_each_operation_position(
    contiguous_layout, unified_pool, failure_position
):
    vmm_ops = _compiled_vmm_ops()
    valid_offsets, invalid_offset = _create_layout(vmm_ops, contiguous_layout, unified_pool)
    offsets = list(valid_offsets)
    offsets.insert(failure_position, invalid_offset)

    try:
        with pytest.raises(RuntimeError, match="outside the reserved virtual address"):
            vmm_ops.map_to_kv_tensors_with_result(offsets, 0)

        for offset in valid_offsets:
            assert vmm_ops.map_to_kv_tensors_with_result([offset], 0) == (
                True,
                [offset],
            )
        assert vmm_ops.unmap_from_kv_tensors(valid_offsets, 0)
    finally:
        vmm_ops.shutdown_kvcached()


@pytest.mark.parametrize(
    ("contiguous_layout", "unified_pool"),
    [(True, False), (False, True), (False, False)],
    ids=["contiguous", "unified", "per-layer-kv"],
)
def test_map_rollback_preserves_preexisting_mapping(contiguous_layout, unified_pool):
    vmm_ops = _compiled_vmm_ops()
    valid_offsets, invalid_offset = _create_layout(vmm_ops, contiguous_layout, unified_pool)
    first, second = valid_offsets

    try:
        assert vmm_ops.map_to_kv_tensors_with_result([first], 0) == (True, [first])
        with pytest.raises(RuntimeError, match="outside the reserved virtual address"):
            vmm_ops.map_to_kv_tensors_with_result([first, second, invalid_offset], 0)

        assert vmm_ops.map_to_kv_tensors_with_result([first], 0) == (True, [])
        assert vmm_ops.map_to_kv_tensors_with_result([second], 0) == (True, [second])
        assert vmm_ops.unmap_from_kv_tensors(valid_offsets, 0)
    finally:
        vmm_ops.shutdown_kvcached()


@pytest.mark.parametrize(
    ("contiguous_layout", "unified_pool"),
    [(True, False), (False, True), (False, False)],
    ids=["contiguous", "unified", "per-layer-kv"],
)
def test_prepared_unmap_can_abort_or_commit(contiguous_layout, unified_pool):
    vmm_ops = _compiled_vmm_ops()
    valid_offsets, _invalid_offset = _create_layout(vmm_ops, contiguous_layout, unified_pool)
    first = valid_offsets[0]

    try:
        with pytest.raises(ValueError, match="transaction id must not be empty"):
            vmm_ops.prepare_unmap_from_kv_tensors([first], "", 0)
        with pytest.raises(ValueError, match="transaction id must not be empty"):
            vmm_ops.commit_unmap_from_kv_tensors("", 0)
        with pytest.raises(ValueError, match="transaction id must not be empty"):
            vmm_ops.abort_unmap_from_kv_tensors("", 0)

        assert vmm_ops.map_to_kv_tensors_with_result([first], 0) == (True, [first])
        assert vmm_ops.prepare_unmap_from_kv_tensors([first], "abort-me", 0)
        with pytest.raises(RuntimeError, match="unmap transaction is pending"):
            vmm_ops.map_to_kv_tensors_with_result([valid_offsets[1]], 0)
        assert vmm_ops.abort_unmap_from_kv_tensors("abort-me", 0)
        assert vmm_ops.abort_unmap_from_kv_tensors("abort-me", 0)
        assert vmm_ops.map_to_kv_tensors_with_result([first], 0) == (True, [])

        assert vmm_ops.prepare_unmap_from_kv_tensors([first], "commit-me", 0)
        assert vmm_ops.commit_unmap_from_kv_tensors("commit-me", 0)
        assert vmm_ops.commit_unmap_from_kv_tensors("commit-me", 0)
        with pytest.raises(RuntimeError, match="cannot abort a committed"):
            vmm_ops.abort_unmap_from_kv_tensors("commit-me", 0)
        assert vmm_ops.map_to_kv_tensors_with_result([first], 0) == (True, [first])
        assert vmm_ops.unmap_from_kv_tensors([first], 0)
    finally:
        vmm_ops.shutdown_kvcached()


@pytest.mark.parametrize(
    ("contiguous_layout", "unified_pool"),
    [(True, False), (False, True), (False, False)],
    ids=["contiguous", "unified", "per-layer-kv"],
)
def test_prepared_map_can_abort_or_commit(contiguous_layout, unified_pool):
    vmm_ops = _compiled_vmm_ops()
    valid_offsets, _invalid_offset = _create_layout(
        vmm_ops, contiguous_layout, unified_pool
    )
    first = valid_offsets[0]

    try:
        prepared = vmm_ops.prepare_map_to_kv_tensors("abort-me", [first], 0)
        assert prepared["success"]
        assert vmm_ops.has_prepared_map("abort-me", 0)
        assert vmm_ops.abort_prepared_map("abort-me", 0)
        assert not vmm_ops.has_prepared_map("abort-me", 0)

        assert vmm_ops.map_to_kv_tensors_with_result([first], 0) == (True, [first])
        assert vmm_ops.unmap_from_kv_tensors([first], 0)

        prepared = vmm_ops.prepare_map_to_kv_tensors("commit-me", [first], 0)
        assert prepared["success"]
        committed = vmm_ops.commit_prepared_map("commit-me", 0)
        assert committed["success"]
        assert not vmm_ops.has_prepared_map("commit-me", 0)
        assert vmm_ops.map_to_kv_tensors_with_result([first], 0) == (True, [])
        assert vmm_ops.unmap_from_kv_tensors([first], 0)
    finally:
        vmm_ops.shutdown_kvcached()


@pytest.mark.parametrize("ordered", [False, True])
def test_unmap_can_release_disjoint_offset_during_background_prepare(ordered):
    vmm_ops = _compiled_vmm_ops()
    offsets, _ = _create_layout(vmm_ops, False, False)
    first, second = offsets
    try:
        assert vmm_ops.map_to_kv_tensors_with_result([first], 0) == (True, [first])
        assert vmm_ops.prepare_map_to_kv_tensors("background", [second], 0)["success"]
        with pytest.raises(RuntimeError, match="prepared map"):
            vmm_ops.unmap_from_kv_tensors([second], 0)
        if ordered:
            assert vmm_ops.prepare_unmap_from_kv_tensors([first], "release", 0)
            assert vmm_ops.commit_unmap_from_kv_tensors("release", 0)
        else:
            assert vmm_ops.unmap_from_kv_tensors([first], 0)
        assert vmm_ops.commit_prepared_map("background", 0)["success"]
        assert vmm_ops.map_to_kv_tensors_with_result([second], 0) == (True, [])
        assert vmm_ops.unmap_from_kv_tensors([second], 0)
    finally:
        vmm_ops.shutdown_kvcached()


def test_prepared_map_adopts_existing_mapping_without_capacity_check():
    vmm_ops = _compiled_vmm_ops()
    valid_offsets, _invalid_offset = _create_layout(vmm_ops, False, False)
    first = valid_offsets[0]

    try:
        assert vmm_ops.map_to_kv_tensors_with_result([first], 0) == (True, [first])
        prepared = vmm_ops.prepare_map_to_kv_tensors("adopt-existing", [first], 0)
        assert prepared["success"]
        assert prepared["capacity_checks"] == 0
        assert prepared["required_bytes"] == 0
        assert vmm_ops.commit_prepared_map("adopt-existing", 0)["success"]
        assert vmm_ops.unmap_from_kv_tensors([first], 0)
    finally:
        vmm_ops.shutdown_kvcached()


def test_physical_growth_rejects_non_finite_utilization(monkeypatch):
    vmm_ops = _compiled_vmm_ops()
    valid_offsets, _invalid_offset = _create_layout(vmm_ops, False, False)
    monkeypatch.setenv("KVCACHED_GPU_UTILIZATION", "nan")

    try:
        prepared = vmm_ops.prepare_map_to_kv_tensors(
            "invalid-limit", [valid_offsets[0]], 0
        )
        assert not prepared["success"]
        assert not vmm_ops.has_prepared_map("invalid-limit", 0)
    finally:
        vmm_ops.shutdown_kvcached()


def test_physical_growth_lock_rejects_symlink(monkeypatch, tmp_path):
    vmm_ops = _compiled_vmm_ops()
    valid_offsets, _invalid_offset = _create_layout(vmm_ops, False, False)
    monkeypatch.setenv("KVCACHED_PHYSICAL_GROWTH_LOCK_DIR", str(tmp_path))
    device_id = vmm_ops.current_device_pci_bus_id()
    safe_device_id = "".join(char if char.isalnum() else "_" for char in device_id)
    lock_path = tmp_path / f"kvcached-physical-growth-{safe_device_id}.lock"
    lock_path.symlink_to(tmp_path / "unrelated")

    try:
        prepared = vmm_ops.prepare_map_to_kv_tensors(
            "symlink-lock", [valid_offsets[0]], 0
        )
        assert not prepared["success"]
        assert not vmm_ops.has_prepared_map("symlink-lock", 0)
    finally:
        vmm_ops.shutdown_kvcached()
