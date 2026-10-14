# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""C4 compressor-state storage, addressed by native SWA page/ring IDs."""

import functools
import importlib
from contextvars import ContextVar
from typing import Any, Callable

from kvcached.integration.sglang.dsv4_backing import SWABacking


def install_c4_state_backing(pool_module: Any, top_class: Any,
                             next_group_id: Callable[[], int]) -> None:
    original_state = getattr(pool_module, "CompressStatePool", None)
    original_factory = getattr(top_class, "_make_compress_state_pool", None)
    original_alloc = getattr(original_state, "_alloc_kv_score_buffer", None)
    if original_state is None or original_factory is None or original_alloc is None:
        return
    context: ContextVar[Any] = ContextVar("dsv4_c4_state_owner", default=None)

    @functools.wraps(original_alloc)
    def allocate(state: Any, *, dtype: Any, device: str, enable_memory_saver: bool) -> None:
        owner = context.get()
        swa = getattr(getattr(owner, "swa_kv_pool", None), "_kvcached_dsv4_backing", None)
        # Request-addressed C128 and unified C4 need their own request-row
        # transaction. Their native ownership and reservation accounting stay intact.
        if (swa is None or state.ratio != 4 or state.online or enable_memory_saver
                or getattr(owner, "_unified_kv", False) or state.page_size != 1):
            return original_alloc(state, dtype=dtype, device=device,
                                  enable_memory_saver=enable_memory_saver)

        import kvcached.integration.sglang.interfaces as kvi

        row_bytes = state.last_dim * dtype.itemsize
        page_bytes = state.ring_size * row_bytes
        native_pages = (state._size + state.ring_size - 1) // state.ring_size
        physical_bytes = kvi.get_page_size_for_block(page_bytes, kvi.PAGE_SIZE)
        physical_pages = (native_pages * page_bytes + physical_bytes - 1) // physical_bytes
        # Native construction and each set_state clear row -1. Keep every
        # physical page touched by that row resident before the constructor runs.
        sentinel = range((state._size - 1) * row_bytes // physical_bytes,
                         (state._size * row_bytes - 1) // physical_bytes + 1)
        permanent = frozenset({0, *sentinel})
        if (native_pages < swa.native_pages or physical_pages <= len(permanent)
                or page_bytes > physical_bytes):
            return original_alloc(state, dtype=dtype, device=device,
                                  enable_memory_saver=enable_memory_saver)

        groups = getattr(owner, "_kvcached_c4_state_groups", None)
        if groups is None:
            groups = owner._kvcached_c4_state_groups = {}
        key = (state._size, state.ring_size, state.last_dim, dtype, str(device))
        group: Any = groups.get(key)
        if group is None:
            layers = owner.compression_ratios[owner._stage_start:owner._stage_end].count(4)
            group_id = next_group_id()
            buffers, raw = kvi.alloc_dsv4_swa_cache(
                num_pages=native_pages, bytes_per_page=page_bytes, num_layers=layers,
                device=device, group_id=group_id,
            )
            manager = kvi.get_kv_cache_manager(
                num_blocks=physical_pages, block_size=1, cell_size=physical_bytes,
                num_layers=layers, reserve_null_block=True, num_kv_buffers=1,
                group_id=group_id, pool_name="dsv4_c4_state",
            )
            backing = SWABacking(manager, native_pages=native_pages,
                                 native_page_bytes=page_bytes, physical_page_bytes=physical_bytes,
                                 permanent_pages=permanent)
            # Padding and sentinel backing must settle before native clear/write.
            manager.wait_ready()
            group = groups[key] = dict(buffers=buffers, raw=raw, backing=backing, used=0)
        index = group["used"]
        if index >= len(group["buffers"]):
            raise RuntimeError("C4 state factory exceeded the native stage layer count")
        group["used"] += 1
        state_module = importlib.import_module(original_state.__module__)
        state.memory_saver_adapter = state_module.TorchMemorySaverAdapter.create(enable=False)
        state.enable_custom_mem_pool, state.custom_mem_pool = False, None
        state.kv_score_buffer = state_module.KVAndScore(
            group["buffers"][index].view(dtype=dtype).view(-1)[:state._size * state.last_dim]
            .view(state._size, state.last_dim)
        )
        state._kvcached_raw_tensors = group["raw"]
        state._kvcached_dsv4_backing = group["backing"]
        state._kvcached_managed = True

    @functools.wraps(original_factory)
    def make_state(owner: Any, *args: Any, **kwargs: Any) -> Any:
        token = context.set(owner)
        try:
            return original_factory(owner, *args, **kwargs)
        finally:
            context.reset(token)

    original_state._alloc_kv_score_buffer = allocate
    top_class._make_compress_state_pool = make_state
