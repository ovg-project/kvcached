# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Real single-GPU controls; these do not validate TP/PP engine execution."""
import gc

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

from kvcached import _C, control, vmm_ops as vmm  # noqa: E402
from kvcached.kv_cache_manager import KVCacheManager  # noqa: E402
from kvcached.runtime_control import RuntimeControl  # noqa: E402

PAGE = 2 * 1024 * 1024

@pytest.fixture(params=[False, True], ids=["split", "contiguous"])
def pools(monkeypatch, tmp_path, request):
    monkeypatch.setenv("KVCACHED_PHYSICAL_GROWTH_LOCK_DIR", str(tmp_path))
    monkeypatch.setattr("kvcached.kv_cache_manager.PAGE_PREALLOC_ENABLED", False)
    monkeypatch.setattr("kvcached.kv_cache_manager.CONTIGUOUS_LAYOUT", request.param)
    vmm.init_kvcached("cuda:0", PAGE, request.param)
    managers, tensors = [], []
    def create(buffers=2, group=0, null=False):
        tensors.extend(vmm.create_kv_tensors(8 * PAGE * buffers, 1, "cuda:0", 2,
                                            buffers, group_id=group, page_size=PAGE))
        manager = KVCacheManager(num_blocks=8, block_size=1, cell_size=PAGE,
                                 num_layers=2, num_kv_buffers=buffers, group_id=group,
                                 page_size=PAGE, reserve_null_block=null)
        manager.wait_ready(timeout=10)
        managers.append(manager)
        return manager
    yield create, managers
    for manager in managers:
        manager.shutdown()
    tensors.clear()
    gc.collect()
    vmm.shutdown_kvcached()

def configure(managers, supported=True):
    for manager in managers:
        manager._runtime_control = RuntimeControl(manager, worker_ipc=False,
            can_shrink=supported, expected_participants=len(managers))

def apply(manager):
    manager._runtime_control.poll(force=True)
    manager.apply_resize_target()

@pytest.mark.parametrize("buffers", [1, 2])
def test_real_geometry_revision_shrink_regrow(pools, buffers):
    create, managers = pools
    manager = create(buffers)
    configure(managers)
    source = manager._runtime_control.sources[0]
    bundle = PAGE * 2 * buffers
    assert source["pool_capacity_bytes"] == 8 * bundle
    blocks = manager.alloc(4)
    assert blocks is not None
    result = manager.set_memory_limit(2 * bundle)
    assert result["accepted"] and not result["converged"]
    apply(manager)
    assert manager.in_shrink
    assert not control.get_instance_memory_limit()["converged"]
    manager.free(blocks)
    apply(manager)
    assert control.get_instance_memory_limit()["converged"]
    old_revision = result["gpus"][0]["revision"]
    same = manager.set_memory_limit(2 * bundle)
    assert same["gpus"][0]["revision"] == old_revision + 1
    assert not _C.acknowledge_instance_control(0, old_revision)
    manager.set_memory_limit(100 * bundle)
    apply(manager)
    assert manager.page_allocator.get_num_total_pages() == 8
    assert manager.page_allocator.get_resize_target() == -1

def test_real_mixed_pools_share_cap(pools):
    create, managers = pools
    create(1, 0)
    create(2, 1)
    configure(managers)
    assert control.set_instance_memory_limit(9 * PAGE + 1)["accepted"]
    for manager in managers:
        apply(manager)
    state = control.get_instance_memory_limit()
    assert len(state["gpus"]) == 1
    assert state["gpus"][0]["effective_limit_bytes"] <= 9 * PAGE + 1
    assert [m.page_allocator.get_num_total_pages() for m in managers] == [1, 1]
    assert state["converged"]

def test_real_zero_clear_preserves_pending_revision(pools):
    create, managers = pools
    manager = create(null=True)
    configure(managers)
    result = manager.set_memory_limit(0)
    apply(manager)
    assert manager.in_shrink
    manager.clear()
    assert manager.null_block == [0] and manager.in_shrink
    state = control.get_instance_memory_limit()
    assert not state["converged"]
    assert state["gpus"][0]["revision"] == result["gpus"][0]["revision"]
    assert state["gpus"][0]["charged_bytes"] > 0
    manager.set_memory_limit(16 * PAGE)
    apply(manager)
    blocks = manager.alloc(1)
    assert blocks and 0 not in blocks
    manager.free(blocks)

def test_real_unsupported_shrink_blocks_growth(pools):
    create, managers = pools
    manager = create()
    configure(managers, supported=False)
    result = manager.set_memory_limit(0)
    assert result["gpus"][0]["convergence"] == "unsupported"
    apply(manager)
    assert not manager.in_shrink
    assert manager.alloc(1) is None


def test_real_cli_api_share_publication_and_observation(pools, capsys):
    import json

    from kvcached.cli.kvctl import cmd_list
    from kvcached.cli.utils import update_kv_cache_limit
    from kvcached.utils import DEFAULT_IPC_NAME
    create, managers = pools
    manager = create()
    configure(managers)
    first = update_kv_cache_limit(DEFAULT_IPC_NAME, 8 * PAGE)
    second = manager.set_memory_limit(8 * PAGE)
    assert second["gpus"][0]["revision"] == first["gpus"][0]["revision"] + 1
    apply(manager)
    capsys.readouterr()
    cmd_list([DEFAULT_IPC_NAME], json_out=True)
    rows = json.loads(capsys.readouterr().out)
    assert rows[0]["gpus"][0]["revision"] == second["gpus"][0]["revision"]
    assert rows[0]["limit_bytes"] == 8 * PAGE
