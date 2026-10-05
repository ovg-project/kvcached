# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Real CUDA allocation, readback and reclamation with SGLang-selected pages."""

import types

import pytest
from test_vmm_transaction import _compiled_vmm_ops

MIB = 1024 * 1024


@pytest.fixture(params=[False, True], ids=["per-layer", "contiguous"])
def pools(monkeypatch, request):
    _compiled_vmm_ops()
    import torch

    from kvcached import kv_cache_manager
    from kvcached.integration.sglang import interfaces

    torch.cuda.init()
    # Other GPU tests can initialize the shim during collection and later
    # shut down native VMM directly. Reset both before choosing the layout.
    assert interfaces.shutdown_kvcached()
    monkeypatch.delenv("KVCACHED_PAGE_SIZE_MB", raising=False)
    monkeypatch.setattr(interfaces, "PAGE_SIZE", 2 * MIB)
    monkeypatch.setattr(interfaces, "_contiguous_layout", request.param)
    monkeypatch.setattr(kv_cache_manager, "CONTIGUOUS_LAYOUT", request.param)
    monkeypatch.setattr(kv_cache_manager, "PAGE_PREALLOC_ENABLED", False)
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda device: types.SimpleNamespace(total_memory=256 * MIB))
    managers: list[kv_cache_manager.KVCacheManager] = []
    interfaces.init_kvcached(device="cuda:0")
    try:
        yield interfaces, managers
    finally:
        torch.cuda.synchronize()
        for manager in managers:
            assert manager.shutdown()
        assert interfaces.shutdown_kvcached()


def _manager(pools, count, tokens, cell, buffers, page_mb, group):
    iface, managers = pools
    manager = iface.get_kv_cache_manager(
        count, tokens, cell, 2, num_kv_buffers=buffers, group_id=group)
    managers.append(manager)
    manager.wait_ready(timeout=10)
    assert manager.page_size == page_mb * MIB
    return manager


def _exercise(manager, views):
    import torch

    available = manager.available_size()
    initial_pages = manager.page_allocator.get_num_inuse_pages()
    for cycle in range(3):
        blocks = manager.alloc(5)
        assert blocks is not None and len(set(blocks)) == 5
        checks = []
        pages = set()
        for i, block in enumerate(blocks):
            start = block * manager.block_mem_size
            end = start + manager.block_mem_size - 1
            assert start // manager.page_size == end // manager.page_size
            pages.add(start // manager.page_size)
            for j, view in enumerate(views(block)):
                value = cycle * 20 + i + j + 1
                view.fill_(value)
                checks.append((view, value))
        assert len(pages) >= 3
        torch.cuda.synchronize()
        for view, value in checks:
            assert torch.all(view == value).item()
        torch.cuda.synchronize()
        manager.free(blocks)
        assert manager.available_size() == available
        assert manager.page_allocator.get_num_inuse_pages() == initial_pages


@pytest.mark.parametrize("attention,buffers", [("MHA", 2), ("MLA", 1)])
@pytest.mark.parametrize("block_bytes,explicit_mb,page_mb", [
    (4 * MIB, None, 4),
    (5 * MIB // 2, None, 6),
    (5 * MIB // 2, 6, 6),
    (5 * MIB // 2, 4, None),
    (4 * MIB, 2, None),
])
def test_attention_page_readback(pools, monkeypatch, attention, buffers,
                                block_bytes, explicit_mb, page_mb):
    import torch

    from kvcached.utils import KVCachedConfigError

    iface, _ = pools
    if explicit_mb is not None:
        monkeypatch.setenv("KVCACHED_PAGE_SIZE_MB", str(explicit_mb))
        monkeypatch.setattr(iface, "PAGE_SIZE", explicit_mb * MIB)
    tokens = 16
    cell = block_bytes // tokens
    raw = iface.alloc_kv_cache(
        (16 * tokens, 1, cell // 2), torch.bfloat16, "cuda:0", 2,
        page_size=tokens, attention_type=attention)
    if page_mb is None:
        with pytest.raises(KVCachedConfigError, match="cannot manage this KV geometry"):
            iface.get_kv_cache_manager(16, tokens, cell, 2, num_kv_buffers=buffers)
        return
    tensors = raw if attention == "MLA" else raw[0] + raw[1]
    manager = _manager(pools, 16, tokens, cell, buffers, page_mb, 0)
    _exercise(manager, lambda block: [t[block * tokens:(block + 1) * tokens] for t in tensors])


@pytest.mark.parametrize("explicit_mb,page_mb", [(None, 4), (4, 4), (6, 6), (2, None)])
def test_mamba_and_attention_keep_independent_pages(pools, monkeypatch, explicit_mb, page_mb):
    import torch

    iface, _ = pools
    if explicit_mb is not None:
        monkeypatch.setenv("KVCACHED_PAGE_SIZE_MB", str(explicit_mb))
        monkeypatch.setattr(iface, "PAGE_SIZE", explicit_mb * MIB)
    # A 2.5 MiB raw slot needs a 4 MiB padded cell, while attention keeps
    # 2 MiB pages when the environment is unset.
    params = types.SimpleNamespace(
        shape=types.SimpleNamespace(conv=[(512, 512)], temporal=(512, 1024)),
        dtype=types.SimpleNamespace(conv=torch.bfloat16, temporal=torch.float32))
    kwargs = dict(num_slots=16, num_mamba_layers=2, cache_params=params,
                  device="cuda:0", group_id=1)
    if page_mb is None:
        with pytest.raises(RuntimeError, match="exceeds.*PAGE_SIZE"):
            iface.alloc_mamba_states(**kwargs)
        return
    conv, temporal, info = iface.alloc_mamba_states(**kwargs)
    manager = _manager(pools, 16, 1, info["cell_size"], 1, page_mb, 1)
    k, v = iface.alloc_kv_cache((8192, 8, 128), torch.bfloat16, "cuda:0", 2,
                              page_size=16, group_id=2)
    attention = _manager(pools, 512, 16, 2048, 2, explicit_mb or 2, 2)
    attention_blocks = attention.alloc(1)
    assert attention_blocks is not None
    block = attention_blocks[0]
    attention_views = [t[block * 16:(block + 1) * 16] for t in k + v]
    for view in attention_views:
        view.fill_(42)

    def views(slot):
        if iface._contiguous_layout:
            return [conv[0][layer, slot] for layer in range(2)] + [
                temporal[layer, slot] for layer in range(2)]
        return [conv[0][layer][slot] for layer in range(2)] + [
            temporal[layer][slot] for layer in range(2)]

    _exercise(manager, views)
    for view in attention_views:
        assert torch.all(view == 42).item()
    torch.cuda.synchronize()
    attention.free(attention_blocks)
