# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import sys
import types

try:
    import torch
except ImportError:
    torch = types.ModuleType("torch")
    sys.modules["torch"] = torch

if not hasattr(torch, "Tensor"):
    torch.Tensor = object
if not hasattr(torch, "dtype"):
    torch.dtype = object
if not hasattr(torch, "cuda"):
    torch.cuda = types.SimpleNamespace(current_device=lambda: 0)

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


def test_sglang_keeps_real_tp_size_for_ipc_but_owns_pool_locally(monkeypatch):
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
        lambda tp_rank, pp_rank: listeners.append((tp_rank, pp_rank)),
    )
    monkeypatch.setattr(interfaces, "KVCacheManager", FakeManager)

    interfaces.init_kvcached(
        tp_rank=2,
        world_size=4,
        pp_rank=1,
        device="cuda:2",
        async_sched=True,
    )
    interfaces.get_kv_cache_manager(128, 16, 64, 8, group_id=3)

    assert initialized
    assert listeners == [(2, 1)]
    assert interfaces._world_size == 4
    assert manager_args[0][1] == {
        "world_size": 1,
        "pp_rank": 1,
        "async_sched": True,
        "reserve_null_block": True,
        "num_kv_buffers": 2,
        "group_id": 3,
        "pool_name": None,
    }
