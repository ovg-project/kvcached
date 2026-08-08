# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import sys
import types

try:
    import torch  # noqa: F401
except ImportError:
    torch = types.ModuleType("torch")
    torch.Tensor = object
    torch.dtype = object
    torch.cuda = types.SimpleNamespace(current_device=lambda: 0)
    sys.modules["torch"] = torch

try:
    import kvcached.vmm_ops  # noqa: F401
except Exception:  # noqa: BLE001 - any extension import failure uses the CPU stub
    vmm_ops = types.ModuleType("kvcached.vmm_ops")
    setattr(vmm_ops, "PageAllocator", object)
    setattr(vmm_ops, "InternalPage", object)
    setattr(vmm_ops, "create_kv_tensors", lambda *args, **kwargs: [])
    setattr(vmm_ops, "init_kvcached", lambda *args, **kwargs: None)
    setattr(vmm_ops, "shutdown_kvcached", lambda: None)
    setattr(vmm_ops, "kv_tensors_created", lambda *args, **kwargs: True)
    setattr(vmm_ops, "map_to_kv_tensors", lambda *args, **kwargs: None)
    setattr(vmm_ops, "unmap_from_kv_tensors", lambda *args, **kwargs: None)
    sys.modules["kvcached.vmm_ops"] = vmm_ops

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
    }
