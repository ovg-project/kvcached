# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Regression test for the prealloc/GIL deadlock of issue #371.

The cycle: a thread calls alloc_page() through a binding that holds the GIL
and blocks in cond_.wait waiting for the prealloc worker to refill the
reserve; the prealloc worker, inside the Python broadcast callback, needs the
GIL to run. Before the fix (GIL released in the blocking bindings) the pair
deadlocks every time this interleaving is forced; captured stacks showed the
worker in take_gil and the caller in pthread_cond_wait.

The scenario is forced deterministically: a pool as small as
min_reserved_pages, so the worker's first reserve drains the free list; a
broadcast callback that sleeps, so the worker predictably sits inside Python
while every page is in flight; then one alloc_page() from the main thread.
It runs in a subprocess so that, on a regression, the deadlock kills the
subprocess via timeout instead of wedging pytest itself.

Needs a GPU: the prealloc worker calls cudaMemGetInfo.
"""

import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("needs a GPU (prealloc worker calls cudaMemGetInfo)",
                allow_module_level=True)

SCENARIO = r"""
import threading, time
import kvcached.vmm_ops as v

PAGE = 2 * 1024 * 1024
POOL_PAGES = 5  # == min_reserved_pages, so the first reserve drains the pool

alloc = v.PageAllocator(num_layers=1, mem_size_per_layer=POOL_PAGES * PAGE,
                        page_size=PAGE, world_size=1, pp_rank=0,
                        async_sched=False, contiguous_layout=True,
                        enable_page_prealloc=True, num_kv_buffers=2,
                        group_id=0, ipc_name="GILTEST")
alloc.set_use_worker_ipc(True)  # force the Python broadcast path

worker_inside = threading.Event()

def slow_map(world_size, offsets):
    worker_inside.set()
    time.sleep(2)  # sleep releases the GIL; resuming afterwards needs it back

alloc.set_broadcast_map_callback(slow_map)
alloc.set_broadcast_unmap_callback(lambda ws, offs: None)
alloc.start_prealloc_thread()
assert worker_inside.wait(timeout=15), "prealloc worker never entered Python"
time.sleep(0.2)  # let the worker settle into its sleep

page = alloc.alloc_page()  # pre-fix: deadlocks here, holding the GIL
print("OK", page.page_id, flush=True)
alloc.stop_prealloc_thread()
"""


def test_alloc_page_does_not_deadlock_against_prealloc_worker():
    proc = subprocess.run([sys.executable, "-c", SCENARIO],
                          capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, (
        f"scenario failed\nstdout: {proc.stdout}\nstderr: {proc.stderr}")
    assert "OK" in proc.stdout, (
        "alloc_page never returned -- the GIL deadlock is back\n"
        f"stdout: {proc.stdout}\nstderr: {proc.stderr}")
