# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""A native callback's hidden reference must not retain a failed KV pool."""

import os
import subprocess
import sys

import pytest

SCENARIO = r"""
import gc
import time
import weakref

import torch
from kvcached import kv_cache_manager as kcm, tp_ipc_util as ipc
from kvcached import vmm_ops as native
from kvcached.lifecycle import LifecyclePhase

PAGE = 2 * 1024 * 1024
torch.cuda.set_device(0)
native.init_kvcached("cuda:0", PAGE, False)
kcm.broadcast_kv_tensors_created = (
    lambda world_size, pp_rank=0, group_id=0: native.kv_tensors_created(group_id))
inject = False
hits = 0

def map_pages(world_size, offsets, pp_rank=0, group_id=0):
    assert native.map_to_kv_tensors(offsets, group_id)

def unmap_pages(world_size, offsets, pp_rank=0, group_id=0):
    global hits
    if inject:
        hits += 1
        raise RuntimeError("injected native unmap failure")
    assert native.unmap_from_kv_tensors(offsets, group_id)

ipc.broadcast_map_to_kv_tensors = map_pages
ipc.broadcast_unmap_from_kv_tensors = unmap_pages

def run_case(group_id, fail):
    global inject
    tensors = native.create_kv_tensors(PAGE * 4, 1, "cuda:0", 1, 1, group_id)
    # world_size=2 selects the production C++ -> Python broadcast callback.
    # Its body maps a single local device; this is not a distributed TP test.
    manager = kcm.KVCacheManager(
        16, 1, PAGE // 4, 1, world_size=2, num_kv_buffers=1,
        group_id=group_id, pool_name="native-lifetime")
    manager.wait_ready(timeout=5)
    blocks = manager.alloc(1)
    assert blocks and len(blocks) == 1
    view = tensors[0].reshape(-1)[blocks[0] * (PAGE // 4):][:16]
    view.fill_(7)
    torch.cuda.synchronize()
    assert torch.all(view == 7).item()
    inject = fail
    try:
        manager.free(blocks)
        assert not fail, "fault did not reach the native callback"
    except RuntimeError as exc:
        assert fail and "injected native unmap failure" in str(exc)
        assert manager.lifecycle_phase is LifecyclePhase.DEGRADED
        assert manager.lifecycle_error is not None
        assert manager.lifecycle_error.__traceback__ is not None
    finally:
        inject = False
    manager.clear()
    expected = LifecyclePhase.DEGRADED if fail else LifecyclePhase.READY
    assert manager.lifecycle_phase is expected
    return weakref.ref(manager), weakref.ref(manager._lifecycle), tensors

for fail in (False, True):
    refs = []
    buffers = []
    for index in range(3):
        manager_ref, lifecycle_ref, tensors = run_case(
            100 + int(fail) * 10 + index, fail)
        refs.append((manager_ref, lifecycle_ref))
        buffers.append(tensors)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        gc.collect()
        if all(ref() is None for pair in refs for ref in pair):
            break
        time.sleep(0.01)
    assert all(ref() is None for pair in refs for ref in pair), (
        "native callback retained the lifecycle/traceback/manager cycle", fail)
    del tensors, buffers
    gc.collect()
assert hits == 3, hits
native.shutdown_kvcached()
print("native lifetime: healthy=3/3, failed=3/3, injections=3", flush=True)
"""


def test_native_unmap_error_does_not_retain_manager():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA VMM and the native PageAllocator")

    env = dict(os.environ)
    env.update(
        ENABLE_KVCACHED="false",
        KVCACHED_AUTOPATCH="0",
        KVCACHED_PAGE_SIZE_MB="2",
        KVCACHED_MIN_RESERVED_PAGES="0",
        KVCACHED_MAX_RESERVED_PAGES="0",
        KVCACHED_PAGE_PREALLOC_ENABLED="false",
        KVCACHED_CONTIGUOUS_LAYOUT="false",
        KVCACHED_IPC_NAME=f"lifecycle-native-{os.getpid()}",
    )
    result = subprocess.run(
        [sys.executable, "-c", SCENARIO], env=env,
        capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "healthy=3/3, failed=3/3, injections=3" in result.stdout
