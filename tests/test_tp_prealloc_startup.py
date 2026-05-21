# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import importlib
import sys
import types
from typing import Any

kcm: Any = None


def _load_kv_cache_manager(monkeypatch, *, use_worker_ipc: bool = False):
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))

    fake_vmm_ops: Any = types.ModuleType("kvcached.vmm_ops")
    fake_vmm_ops.PageAllocator = object
    fake_vmm_ops.InternalPage = object
    fake_vmm_ops.kv_tensors_created = lambda *args, **kwargs: True
    fake_vmm_ops.map_to_kv_tensors = lambda *args, **kwargs: True
    fake_vmm_ops.unmap_from_kv_tensors = lambda *args, **kwargs: True
    monkeypatch.setitem(sys.modules, "kvcached.vmm_ops", fake_vmm_ops)

    fake_interfaces: Any = types.ModuleType("kvcached.integration.vllm.interfaces")
    fake_interfaces.should_use_worker_ipc = lambda: use_worker_ipc
    monkeypatch.setitem(
        sys.modules,
        "kvcached.integration.vllm.interfaces",
        fake_interfaces,
    )

    kcm = importlib.import_module("kvcached.kv_cache_manager")
    return importlib.reload(kcm)


class FakeInternalPage:
    def __init__(self, page_id: int, page_size: int):
        self.page_id = page_id
        self.page_size = page_size
        self._free_blocks: list[int] = []

    def init(self, block_mem_size: int):
        blocks_per_page = self.page_size // block_mem_size
        self._free_blocks = list(range(self.page_id * blocks_per_page,
                                       (self.page_id + 1) * blocks_per_page))

    def num_free_blocks(self) -> int:
        return len(self._free_blocks)

    def alloc(self, num_blocks: int = 1):
        allocated = self._free_blocks[:num_blocks]
        self._free_blocks = self._free_blocks[num_blocks:]
        return allocated

    def full(self) -> bool:
        return not self._free_blocks

    @staticmethod
    def get_num_blocks(page_size: int, block_mem_size: int) -> int:
        return page_size // block_mem_size


class FakePageAllocator:
    def __init__(self, *args, **kwargs):
        self.start_prealloc_calls = 0
        self.stop_prealloc_calls = 0
        self._free_pages = [0, 1, 2]

    def set_should_use_worker_ipc_callback(self, callback):
        self.should_use_worker_ipc_callback = callback

    def set_broadcast_map_callback(self, callback):
        self.broadcast_map_callback = callback

    def set_broadcast_unmap_callback(self, callback):
        self.broadcast_unmap_callback = callback

    def start_prealloc_thread(self):
        self.start_prealloc_calls += 1

    def _stop_prealloc_thread(self, timeout=None):
        self.stop_prealloc_calls += 1

    def alloc_page(self):
        return FakeInternalPage(self._free_pages.pop(0), kcm.PAGE_SIZE)

    def free_pages(self, page_ids):
        self._free_pages.extend(sorted(page_ids))

    def trim(self):
        return None

    def reset_free_page_order(self):
        self._free_pages = sorted(self._free_pages)

    def get_num_free_pages(self):
        return len(self._free_pages)

    def get_avail_physical_pages(self):
        return len(self._free_pages)

    def get_num_reserved_pages(self):
        return 0


def _build_manager(monkeypatch, *, world_size: int):
    global kcm
    kcm = _load_kv_cache_manager(monkeypatch)
    monkeypatch.setattr(kcm, "PageAllocator", FakePageAllocator)
    monkeypatch.setattr(kcm, "InternalPage", FakeInternalPage)
    monkeypatch.setattr(kcm, "broadcast_kv_tensors_created",
                        lambda *args, **kwargs: True)

    manager = kcm.KVCacheManager(
        num_blocks=1024,
        block_size=16,
        cell_size=1024,
        num_layers=16,
        world_size=world_size,
        async_sched=True,
    )
    assert manager._post_init_done.wait(timeout=1)
    return manager


def test_multi_process_prealloc_waits_until_first_alloc(monkeypatch):
    manager = _build_manager(monkeypatch, world_size=2)

    assert manager.page_allocator.start_prealloc_calls == 0

    block_ids = manager.alloc(1)

    assert block_ids == [0]
    assert manager.page_allocator.start_prealloc_calls == 1


def test_single_process_keeps_eager_prealloc(monkeypatch):
    manager = _build_manager(monkeypatch, world_size=1)

    assert manager.page_allocator.start_prealloc_calls == 1


def test_clear_restarts_prealloc_thread(monkeypatch):
    manager = _build_manager(monkeypatch, world_size=1)

    assert manager.page_allocator.start_prealloc_calls == 1

    manager.clear()

    assert manager.page_allocator.stop_prealloc_calls == 1
    assert manager.page_allocator.start_prealloc_calls == 2