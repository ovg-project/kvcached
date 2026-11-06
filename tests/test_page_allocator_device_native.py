# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Native budget integration and device queries, including two-GPU cases."""

import os
import subprocess
import sys

import pytest


def test_native_standalone_allocator_without_a_tensor_backend(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.version.hip:
        pytest.skip("CUDA required")
    result = subprocess.run([sys.executable, "-c", _STANDALONE, str(tmp_path / "pool")],
                            env=dict(os.environ, ENABLE_KVCACHED="false", KVCACHED_AUTOPATCH="0"),
                            capture_output=True, text=True, check=False, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("mode", ["query", "prealloc"])
@pytest.mark.parametrize("implicit,group", [(False, 0), (True, 0), (False, 7), (True, 7)])
def test_native_pool_uses_its_gpu(tmp_path, mode, implicit, group):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("two visible CUDA devices required")
    if torch.version.hip:
        pytest.skip("CUDA runtime regression")
    env = dict(os.environ, ENABLE_KVCACHED="false", KVCACHED_AUTOPATCH="0",
               KVCACHED_GPU_UTILIZATION="0.08", KVCACHED_MIN_RESERVED_PAGES="4",
               KVCACHED_MAX_RESERVED_PAGES="4",
               KVCACHED_PHYSICAL_GROWTH_LOCK_DIR=str(tmp_path))
    result = subprocess.run([sys.executable, "-c", _SCENARIO, mode, str(int(implicit)),
                             str(group), str(tmp_path / "pool")], env=env,
                            capture_output=True, text=True, check=False, timeout=45)
    if result.returncode == 77:
        pytest.skip("requires idle GPUs with room for the controlled pressure allocation")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS owner-device" in result.stdout


@pytest.mark.parametrize("implicit,group", [(False, 0), (True, 0), (False, 7), (True, 7)])
def test_native_capacity_honors_instance_budget(tmp_path, implicit, group):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.version.hip:
        pytest.skip("CUDA required")
    env = dict(os.environ, ENABLE_KVCACHED="false", KVCACHED_AUTOPATCH="0",
               KVCACHED_GPU_UTILIZATION="0.95",
               KVCACHED_PHYSICAL_GROWTH_LOCK_DIR=str(tmp_path))
    result = subprocess.run([sys.executable, "-c", _BUDGET, str(int(implicit)),
                             str(group), str(tmp_path / "pool")], env=env,
                            capture_output=True, text=True, check=False, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


_SCENARIO = r'''
import sys
import threading
import time
import torch
from kvcached import vmm_ops as native

mode, implicit, group, ipc = sys.argv[1:]
group = int(group)
page = 2 * 1024**2
pool = None
tensors = []
pressure = None
for device in (0, 1):
    torch.cuda.set_device(device)
    torch.cuda.synchronize(device)
free0, total0 = torch.cuda.mem_get_info(0)
pressure_bytes = int(total0 * .12)
if free0 < pressure_bytes + 64 * 1024**2:
    sys.exit(77)
try:
    pressure = torch.empty(pressure_bytes, dtype=torch.uint8, device="cuda:0")
    torch.cuda.set_device(1)
    native.init_kvcached("cuda" if implicit == "1" else "cuda:1", page, True, ipc)
    # A lazy nonzero group must retain the initialized GPU after this switch.
    if group:
        torch.cuda.set_device(0)
    tensors = native.create_kv_tensors(32 * page, 1, "cuda:1", 1, 1, group, False, page)
    assert tensors[0].device.index == 1
    torch.cuda.set_device(0)
    free1, total1 = torch.cuda.mem_get_info(1)
    headroom0, headroom1 = int(total0 * .92), int(total1 * .92)
    free0, _ = torch.cuda.mem_get_info(0)
    if free0 >= headroom0 or free1 < headroom1 + 8 * page:
        sys.exit(77)
    pool = native.PageAllocator(1, 32 * page, page, contiguous_layout=True,
                                enable_page_prealloc=mode == "prealloc",
                                num_kv_buffers=1, group_id=group, ipc_name=ipc)
    if mode == "query":
        observed = []
        def query():
            before = torch.cuda.current_device()
            pages = pool.get_avail_physical_pages()
            observed.append((before, torch.cuda.current_device(), pages))
        worker = threading.Thread(target=query)
        worker.start()
        worker.join(timeout=5)
        assert not worker.is_alive() and len(observed) == 1
        free_after, _ = torch.cuda.mem_get_info(1)
        low, high = sorted(max(0, free - headroom1) // page for free in (free1, free_after))
        before, after, pages = observed[0]
        assert before == after == 0
        assert low <= pages <= high, (pages, low, high)
    else:
        pool.start_prealloc_thread()
        deadline = time.monotonic() + 3
        while pool.get_num_reserved_pages() != 4 and time.monotonic() < deadline:
            time.sleep(.01)
        pool.stop_prealloc_thread()
        assert pool.get_num_reserved_pages() == 4
        torch.cuda.set_device(1)
        allocated = pool.alloc_page()
        view = tensors[0].view(-1)[allocated.page_id * page:allocated.page_id * page + 16]
        view.fill_(37)
        torch.cuda.synchronize(1)
        assert torch.all(view == 37).item()
        pool.free_page(allocated.page_id)
        del allocated, view
    print("PASS owner-device", flush=True)
finally:
    torch.cuda.set_device(1)
    if pool is not None:
        pool.stop_prealloc_thread()
        pool.trim()
        pool.release_shared_segment()
        pool = None
    torch.cuda.synchronize(1)
    tensors.clear()
    native.shutdown_kvcached()
    pressure = None
    torch.cuda.empty_cache()
'''

_STANDALONE = r'''
import sys
import torch
from kvcached import vmm_ops as native
torch.cuda.set_device(0)
pool = native.PageAllocator(1, 8 * 1024**2, 2 * 1024**2,
                            enable_page_prealloc=False, ipc_name=sys.argv[1])
try:
    assert pool.get_avail_physical_pages() >= 0
    assert torch.cuda.current_device() == 0
finally:
    pool.release_shared_segment()
'''

_BUDGET = r'''
import sys
import threading
import torch
from kvcached import vmm_ops as native

implicit, group, ipc = sys.argv[1:]
group = int(group)
page = 2 * 1024**2
torch.cuda.set_device(0)
native.init_kvcached("cuda" if implicit == "1" else "cuda:0", page, True, ipc)
pool = None
tensors = []
try:
    tensors = native.create_kv_tensors(8 * page, 1, "cuda:0", 1, 1, group, False, page)
    pool = native.PageAllocator(1, 8 * page, page, contiguous_layout=True,
                                enable_page_prealloc=False, num_kv_buffers=1,
                                group_id=group, ipc_name=ipc)
    state = native.physical_budget_state(group)
    native.set_instance_budget_limit(state["instance_charged_bytes"] + 3 * page, 1, group)
    assert pool.get_avail_physical_pages() == 3
    native.set_instance_budget_limit(state["instance_charged_bytes"], 2, group)
    observed = []
    def query():
        observed.append((pool.get_avail_physical_pages(), torch.cuda.current_device()))
    worker = threading.Thread(target=query)
    worker.start()
    worker.join(timeout=5)
    assert not worker.is_alive() and observed == [(0, 0)]
finally:
    if pool is not None:
        pool.release_shared_segment()
        pool = None
    tensors.clear()
    native.shutdown_kvcached()
'''
