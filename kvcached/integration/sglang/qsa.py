# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Elastic backing for SGLang's dense, layer-major QSA compressed keys."""

import functools
import itertools
import weakref
from typing import Any, Iterable

import torch
from torch.overrides import TorchFunctionMode

from kvcached.integration.patch_base import enable_kvcached
from kvcached.integration.sglang.dsv4_backing import SWABacking, plan_reserve_recovery
from kvcached.utils import PAGE_SIZE, KVCachedConfigError, get_page_size_for_block

_group_ids = itertools.count(30_000)


class QSAIndexBacking:
    def __init__(self, pool, shape, dtype, device, page_size, num_pages):
        from kvcached.integration.sglang import interfaces as kvi

        self.layers = shape[0]
        self.pages_per_layer = num_pages
        self.full_manager = weakref.proxy(pool.full_kv_pool.kvcached_allocator)
        self.failed_prefill: tuple[tuple[int, ...], tuple[int, ...]] = ((), ())
        self.failed_decode: tuple[int, ...] = ()
        self._decode_attempt: tuple[int, ...] | None = None
        self.forward_stream = None
        self.posted_bytes = 0
        native_bytes = (page_size // pool.qsa_compress_ratio * pool.qsa_index_kv_heads
                        * pool.qsa_index_head_dim * dtype.itemsize)
        if shape[1] * dtype.itemsize != num_pages * native_bytes:
            raise KVCachedConfigError("QSA compressed rows must cover whole SGLang pages")
        physical = get_page_size_for_block(native_bytes, PAGE_SIZE)
        if native_bytes > physical:
            raise KVCachedConfigError("A QSA index page exceeds the configured VMM page size")
        total = self.layers * num_pages * native_bytes
        count = (total + physical - 1) // physical
        # The CUDA writer addresses row i as base + i * head_dim. Keep a
        # single dense arena, even when the main KV pool interleaves layers.
        group = next(_group_ids)
        raw, = kvi.create_kv_tensors(count * physical, 1, str(device), 1,
                                     num_kv_buffers=1, group_id=group,
                                     unified_pool=True, page_size=physical)
        self.raw_pages = raw.view(torch.uint8)[:count * physical].view(count, physical)
        self.flat = raw.view(torch.uint8)[:total].view(dtype).view(shape)
        # Every layer has a padding page, including graph replay's dump slot.
        pins: set[int] = set()
        for layer in range(self.layers):
            start = layer * num_pages * native_bytes
            pins.update(range(start // physical, (start + native_bytes - 1) // physical + 1))
        manager = kvi.get_kv_cache_manager(count, 1, physical, 1, num_kv_buffers=1,
                                           group_id=group, pool_name="qsa_compressed_k",
                                           max_block_id=count, permanent_block_ids=tuple(sorted(pins - {0})),
                                           enable_page_prealloc=False)
        manager.wait_ready()
        self.leases = SWABacking(manager, native_pages=num_pages * self.layers,
                                 native_page_bytes=native_bytes, physical_page_bytes=physical,
                                 permanent_pages=frozenset(pins), padding_slots=self.layers)
        self._zero(pins)

    def _zero(self, pages: set[int]) -> None:
        if pages:
            # Reading or filling the entire virtual tensor would touch unmapped pages.
            indices = torch.tensor(sorted(pages), device=self.raw_pages.device)
            stream = torch.cuda.current_stream(self.raw_pages.device)
            if self.forward_stream is None:
                self.raw_pages.index_fill_(0, indices, 0)
                stream.synchronize()
            else:
                self.forward_stream.wait_stream(stream)
                with torch.cuda.stream(self.forward_stream):
                    self.raw_pages.index_fill_(0, indices, 0)
                indices.record_stream(self.forward_stream)

    def _pages(self, ids: Iterable[int]) -> set[int]:
        ids = set(ids)
        if any(i <= 0 or i >= self.pages_per_layer for i in ids):
            raise KVCachedConfigError("Full KV page ID exceeds the QSA index")
        return {layer * self.pages_per_layer + i for i in ids for layer in range(self.layers)}

    def acquire(self, ids: Iterable[int], *, placement: list[int] | None = None) -> bool:
        from kvcached.integration.sglang.patches import _SGLANG_BATCH_ALLOCATION, _SGLANG_DEMAND

        ids = tuple(ids)
        pages = self._pages(ids)
        reqs = getattr(_SGLANG_BATCH_ALLOCATION, "requests", ())
        missing = self.leases.missing_physical_pages(pages) if pages else set()
        if missing:
            self.leases.manager.invalidate_physical_capacity()
        if pages and not self.leases.acquire(pages):
            if not self._recover(missing, []) or not self.leases.acquire(pages):
                if getattr(_SGLANG_DEMAND, "open", False):
                    attempt = tuple(placement) if placement is not None else ids
                    if reqs:
                        self.failed_prefill = (tuple(map(id, reqs)), attempt)
                    elif self._decode_attempt is not None:
                        self._decode_attempt = attempt
                return False
        if reqs:
            self.failed_prefill = ((), ())
        elif self._decode_attempt is not None:
            self._decode_attempt = ()
        self._zero(missing)
        return True

    def idle_bytes(self) -> int:
        return self.leases.manager.page_allocator.get_num_reserved_pages() * self.leases.physical_page_bytes

    def reserve_decode(self, count: int, *, packed: bool) -> bool:
        # Only remember a final rejected reservation. A retry inside alloc()
        # can recover, and the scheduler can try smaller batches after a miss.
        self._decode_attempt = ()
        try:
            ok = self.full_manager.try_to_reserve(count, packed=packed)
            if not ok and not self.failed_decode:
                self.failed_decode = self._decode_attempt or ()
            return ok
        finally:
            self._decode_attempt = None

    def _plan(self, missing: set[int], full_ids: list[int] | None, count: int = 0):
        full = self.full_manager
        index = self.leases.manager
        full_quote = full.exact_recovery_quote(full_ids or [])
        if full_ids is None and full_quote is not None:
            # Ordinary full allocation consumes arbitrary full reserves first.
            wanted = full.demand_pages(count, include_reserves=False)
            idle = full_quote[1]
            full_quote = (max(0, wanted - len(idle)) * full._memory_limit_page_bundle_bytes(), idle[wanted:], 0)
        return plan_reserve_recovery(((full, full_quote), (index, index.exact_recovery_quote(sorted(missing)))),
                                     index.page_allocator.get_avail_physical_bytes())

    def _recover(self, missing: set[int], full_ids: list[int] | None, count: int = 0) -> bool:
        if not (self.idle_bytes() or self.full_manager.page_allocator.get_num_reserved_pages()):
            return False
        plan = self._plan(missing, full_ids, count)
        if not plan:
            return False
        for manager, idle in plan:
            manager.trim_selected_reserves(idle)
        self.full_manager.invalidate_physical_capacity()
        self.leases.manager.invalidate_physical_capacity()
        return True

    def recover_full(self, count: int, ids: list[int] | None) -> bool:
        missing = self.leases.missing_physical_pages(self._pages(ids)) if ids else set()
        return self._recover(missing, ids, count)

    def can_acquire(self, ids: Iterable[int]) -> bool:
        full_ids = list(ids)
        missing = self.leases.missing_physical_pages(self._pages(full_ids))
        return self._plan(missing, full_ids) is not None

    def release(self, ids: Iterable[int]) -> None:
        self.leases.release(self._pages(ids))

    def clear(self) -> None:
        self.leases.release(set(self.leases.live))
        self.leases.manager.trim()
        self.failed_prefill = ((), ())
        self.failed_decode = ()

    def available_size(self) -> int:
        idle = self.full_manager.page_allocator.get_num_reserved_pages()
        reclaimable = idle * self.full_manager._memory_limit_page_bundle_bytes()
        return min(self.pages_per_layer - 1 - len(self.leases.live) // self.layers,
                   self.leases.available_pages(reclaimable) // self.layers)

    def _placement_quote(self, ids: tuple[int, ...], full_idle: int, index_idle: int, fresh_full: bool):
        full = self.full_manager
        index = self.leases.manager
        missing = self.leases.missing_physical_pages(self._pages(ids))
        full_quote = full.exact_recovery_quote(list(ids))
        index_quote = index.exact_recovery_quote(sorted(missing))
        full_cost = None
        maps = 0
        if full_quote is not None:
            maps = full_quote[0]
            full_cost = maps - min(full_idle, len(full_quote[1])) * full._memory_limit_page_bundle_bytes()
        index_cost = (len(missing) - index_idle) * self.leases.physical_page_bytes
        if index_quote is not None:
            maps += index_quote[0]
            index_cost = index_quote[0] - len(index_quote[1]) * self.leases.physical_page_bytes
        else:
            maps += len(missing) * self.leases.physical_page_bytes
        deficit = max(0, -index.page_allocator.get_avail_physical_bytes()) if maps or fresh_full else 0
        return full_cost, index_cost, deficit

    def _demand_bytes(self, need: int, ids: tuple[int, ...] = ()) -> int:
        full = self.full_manager
        fresh = max(0, need - len(full.reserved_blocks))
        tiles = self.leases.physical_demand(fresh * self.layers)
        full_pages = full.demand_pages(need, include_reserves=False)
        idle = max(0, full.page_allocator.get_num_reserved_pages() - getattr(full, "_priority_reserve_debt", 0))
        index_idle = self.leases.manager.page_allocator.get_num_reserved_pages()
        full_cost = (full_pages - idle) * full._memory_limit_page_bundle_bytes()
        index_cost = (tiles - index_idle) * self.leases.physical_page_bytes
        deficit = 0
        if ids:
            exact_full, exact_index, deficit = self._placement_quote(ids, idle, index_idle, full_pages > idle)
            # Decode can take a rolled-back full ID without mapping all its
            # index tiles. Price those tiles even if full placement has changed.
            if exact_full is not None:
                full_cost = max(full_cost, exact_full)
            index_cost = max(index_cost, exact_index)
        return max(0, full_cost + index_cost + deficit)

    def post_demand(self, need: int, *, allocated: int = 0, failed=None, prefill_need: int = 0) -> None:
        from kvcached.vmm_ops import set_kv_demand

        full = self.full_manager
        total = self._demand_bytes(need)
        if failed is not None:
            owners, placement = self.failed_prefill
            ids = placement if owners == tuple(map(id, failed.requests)) else ()
            # The adder's forecast includes the inputs it selected, even if
            # those inputs subsequently failed to acquire their index tiles.
            blocks = max(prefill_need, allocated + failed.requested)
            total = max(total, self._demand_bytes(blocks, ids))
        if self.failed_decode:
            # Decode is a fallback after a rolled-back prefill. Either can be
            # retried next, so publish the larger of their physical shortfalls.
            total = max(total, self._demand_bytes(allocated + len(self.failed_decode), self.failed_decode))
        self.failed_prefill = ((), ())
        self.failed_decode = ()
        if total != self.posted_bytes:
            set_kv_demand(total, full.group_id)
            self.posted_bytes = total


class _CompressedAllocation(TorchFunctionMode):
    """0.5.20 allocates the flat index inline, just after the request RoPE ring."""

    def __init__(self, pool, options):
        self.pool = pool
        self.options = options
        self.backing: Any = None

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        pool = self.pool
        if (func is torch.zeros and self.backing is None and args
                and kwargs.get("dtype") == pool.index_state_dtype
                and getattr(getattr(pool, "qsa_rope_position_buffer", None), "ndim", 0) == 2):
            layers = len(self.options["full_attention_layer_ids"])
            shape = (layers, pool.qsa_compressed_capacity * pool.qsa_index_kv_heads * pool.qsa_index_head_dim)
            if tuple(args[0]) == shape and layers:
                full = pool.full_kv_pool
                if not hasattr(full, "kvcached_allocator") or full.enable_custom_mem_pool:
                    raise KVCachedConfigError("QSA needs elastic full KV with ordinary CUDA VMM")
                page_size = self.options["page_size"]
                self.backing = QSAIndexBacking(pool, shape, pool.index_state_dtype,
                                               full.k_buffer[0].device, page_size,
                                               self.options["size"] // page_size + 1)
                return self.backing.flat
        return func(*args, **kwargs)


def qsa_initializer(cls):
    original = cls.__init__

    from sglang.srt.mem_cache.memory_pool import ElasticMHATokenToKVPool

    class QSAFullKVPool(ElasticMHATokenToKVPool):
        _kvcached_qsa_full_pool = True

    def unsupported_transfer(*args, **kwargs):
        raise KVCachedConfigError(
            "Elastic QSA supports colocated serving. Host copies, KV relocation and "
            "disaggregated transfer require coordinated compressed-index and pending-ring state.")

    @functools.wraps(original)
    def initialize(pool, **options):
        from kvcached.integration.sglang import interfaces as kvi
        from kvcached.integration.sglang.patches import _sglang_pipeline_stages

        if not enable_kvcached() or not str(options["device"]).startswith("cuda"):
            return original(pool, **options)
        if _sglang_pipeline_stages() > 1:
            raise KVCachedConfigError("Elastic QSA requires pipeline parallel size 1 for coordinated admission")
        # The 0.5.20 configurator enables KV copy for every speculative
        # algorithm. Its draft pool shares target page IDs, not target backing.
        if options.get("enable_kv_cache_copy"):
            raise KVCachedConfigError("Elastic QSA does not support speculative decoding with shared draft page IDs")
        if not options["full_attention_layer_ids"]:
            original(pool, **options)
            kvi.register_runtime_owned_reservation(str(pool.qsa_rope_position_buffer.device),
                                                   "qsa.pending_ring", pool.qsa_rope_position_buffer.nbytes,
                                                   owner=pool)
            return
        if options["page_size"] <= 0 or options["size"] % options["page_size"]:
            raise KVCachedConfigError("QSA capacity must contain whole full-KV pages")
        if options.get("full_kv_pool_class") not in (None, ElasticMHATokenToKVPool):
            raise KVCachedConfigError("QSA index backing requires the elastic MHA full-KV pool")
        options["full_kv_pool_class"] = QSAFullKVPool
        with _CompressedAllocation(pool, options) as allocation:
            original(pool, **options)
        backing = allocation.backing
        if backing is None or pool.qsa_compressed_flat is not backing.flat:
            raise KVCachedConfigError("SGLang did not bind the expected QSA compressed index")
        pool.full_kv_pool.kvcached_allocator.bind_block_backing(backing)
        pool._kvcached_qsa_index = backing
        # HybridLinearKVPool copies only full KV and Mamba. RDMA buffer
        # registration also assumes the entire advertised VA is mapped.
        for name in ("get_cpu_copy", "load_cpu_copy", "move_kv_cache", "get_qsa_compressed_state_buf_infos"):
            setattr(pool, name, unsupported_transfer)
        ring_bytes = sum(x.nbytes for x in pool.qsa_key_state_buffer_pool) + pool.qsa_rope_position_buffer.nbytes
        kvi.register_runtime_owned_reservation(str(backing.flat.device), "qsa.pending_ring",
                                               ring_bytes, owner=pool)

    return initialize
