# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import sys
import types

import pytest

pytest.importorskip("torch")

try:
    import kvcached.vmm_ops as vmm_ops
except Exception:  # noqa: BLE001 - any extension import failure uses the CPU stub
    vmm_ops = types.ModuleType("kvcached.vmm_ops")
    sys.modules["kvcached.vmm_ops"] = vmm_ops

for name, value in {
    "PageAllocator": object,
    "InternalPage": object,
    "create_kv_tensors": lambda *args, **kwargs: [],
    "init_kvcached": lambda *args, **kwargs: None,
    "shutdown_kvcached": lambda: None,
    "kv_tensors_created": lambda *args, **kwargs: True,
    "map_to_kv_tensors": lambda *args, **kwargs: None,
    "unmap_from_kv_tensors": lambda *args, **kwargs: None,
}.items():
    if not hasattr(vmm_ops, name):
        setattr(vmm_ops, name, value)

from kvcached.integration.sglang import interfaces  # noqa: E402


@pytest.mark.parametrize(
    "tp_rank,world_size,pp_rank,device,device_index",
    [(2, 4, 1, "cuda:3", 3), (2, 4, 1, "hip:3", 3), (0, 1, 0, "cuda:0", 0)],
)
def test_sglang_keeps_real_tp_size_for_ipc_but_owns_pool_locally(
    monkeypatch, tp_rank, world_size, pp_rank, device, device_index
):
    initialized = []
    listeners = []
    manager_args = []

    class FakeManager:
        def __init__(self, *args, **kwargs):
            manager_args.append((args, kwargs))

    monkeypatch.setattr(interfaces, "_kvcached_initialized", False)
    monkeypatch.setattr(interfaces, "_kvcached_device", None)
    monkeypatch.setattr(interfaces, "_async_sched", False)
    monkeypatch.setattr(interfaces, "_world_size", 1)
    monkeypatch.setattr(interfaces, "_pp_rank", 0)
    monkeypatch.setattr(interfaces, "_init_kvcached_impl", lambda *args: initialized.append(args))
    monkeypatch.setattr(
        interfaces,
        "start_worker_listener_thread",
        lambda tp_rank, pp_rank, *, device_index: listeners.append(
            (tp_rank, pp_rank, device_index)
        ),
    )
    monkeypatch.setattr(interfaces, "KVCacheManager", FakeManager)

    interfaces.init_kvcached(
        tp_rank=tp_rank,
        world_size=world_size,
        pp_rank=pp_rank,
        device=device,
        async_sched=True,
    )
    interfaces.get_kv_cache_manager(128, 16, 64, 8, group_id=3)

    assert initialized == [
        (f"cuda:{device_index}", interfaces.PAGE_SIZE, interfaces._contiguous_layout)
    ]
    assert listeners == ([(tp_rank, pp_rank, device_index)] if world_size > 1 else [])
    assert interfaces._world_size == world_size
    assert manager_args[0][1] == {
        "world_size": 1,
        "pp_rank": pp_rank,
        "async_sched": True,
        "reserve_null_block": True,
        "num_kv_buffers": 2,
        "group_id": 3,
        "pool_name": None,
    }
