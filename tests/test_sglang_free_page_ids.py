# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Elastic allocator free_page_ids and free-group protocol coverage.

SGLang 0.5.19 replaced the ``is_not_in_free_group`` flag with
``free_group is None`` as the not-in-group sentinel, and 0.5.20 added
``free_page_ids()``, which the SWA composite calls on its sub-allocators.
Both elastic allocators must free correctly under either protocol.
"""

import inspect
import sys
import types
from typing import Any

import pytest

from kvcached.integration.sglang.patches import ElasticAllocatorPatch


class FakeTensor:
    def __init__(self, data):
        self.data = list(data)

    def numel(self):
        return len(self.data)

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def __floordiv__(self, divisor):
        return FakeTensor([value // divisor for value in self.data])

    def cpu(self):
        return self

    def numpy(self):
        return self

    def tolist(self):
        return list(self.data)

    def clone(self):
        return FakeTensor(self.data)


def _fake_tensor(data, dtype=None, device=None):
    return FakeTensor(data)


def _fake_empty(shape, dtype=None, device=None):
    return FakeTensor([])


def _fake_unique(tensor):
    return FakeTensor(sorted(set(tensor.data)))


def _fake_cat(tensors):
    merged = []
    for tensor in tensors:
        merged.extend(tensor.data)
    return FakeTensor(merged)


def _install_fake_torch(monkeypatch):
    torch: Any = types.ModuleType("torch")
    torch.Tensor = FakeTensor
    torch.int64 = "int64"
    torch.tensor = _fake_tensor
    torch.empty = _fake_empty
    torch.unique = _fake_unique
    torch.cat = _fake_cat
    monkeypatch.setitem(sys.modules, "torch", torch)


def _install_fake_sglang_utils(monkeypatch):
    sglang: Any = types.ModuleType("sglang")
    srt: Any = types.ModuleType("sglang.srt")
    utils: Any = types.ModuleType("sglang.srt.utils")

    utils.get_num_new_pages = lambda **kwargs: 0
    utils.next_power_of_2 = lambda value: 1 << (max(value, 1) - 1).bit_length()
    sglang.srt = srt
    srt.utils = utils

    monkeypatch.setitem(sys.modules, "sglang", sglang)
    monkeypatch.setitem(sys.modules, "sglang.srt", srt)
    monkeypatch.setitem(sys.modules, "sglang.srt.utils", utils)


class FakeKVCachedAllocator:
    def __init__(self):
        self.free_calls = []
        self.clear_calls = 0

    def alloc(self, num_pages):
        return list(range(num_pages))

    def free(self, ids):
        self.free_calls.append(list(ids))

    def clear(self):
        self.clear_calls += 1


class FakeKVCache:
    def __init__(self):
        self.kvcached_allocator = FakeKVCachedAllocator()


class FakeLegacyBase:
    """SGLang <=0.5.18 free-group protocol (is_not_in_free_group flag)."""

    def __init__(self, size, page_size, dtype, device, kvcache, *args, **kwargs):
        self.size = size
        self.page_size = page_size
        self.dtype = dtype
        self.device = device
        self.is_not_in_free_group = True
        self.free_group = []

    def free(self, free_index):
        # The injected elastic subclass overrides this.
        raise NotImplementedError

    def free_group_begin(self):
        self.is_not_in_free_group = False
        self.free_group = []

    def free_group_end(self):
        self.is_not_in_free_group = True
        if self.free_group:
            self.free(_fake_cat(self.free_group))


class FakeModernBase:
    """SGLang 0.5.19+ free-group protocol (free_group None sentinel)."""

    def __init__(self, size, page_size, dtype, device, kvcache, *args, **kwargs):
        self.size = size
        self.page_size = page_size
        self.dtype = dtype
        self.device = device
        self.free_group = None

    def free(self, free_index):
        # The injected elastic subclass overrides this.
        raise NotImplementedError

    def free_group_begin(self):
        assert self.free_group is None, "free groups cannot be nested"
        self.free_group = []

    def free_group_end(self):
        pending, self.free_group = self.free_group, None
        if pending:
            self.free(_fake_cat(pending))

    @staticmethod
    def _copy_for_free_group(free_index):
        return free_index.clone()


class _FakeKernelFn:
    def __init__(self):
        self.__signature__ = inspect.Signature(
            [
                inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                for name in (
                    "pre_lens_ptr",
                    "seq_lens_ptr",
                    "last_loc_ptr",
                    "free_page_ptr",
                    "out_indices",
                    "bs_upper",
                    "page_size",
                )
            ]
        )

    def __call__(self, *args, **kwargs):
        pass


class _FakeKernel:
    def __init__(self):
        self.fn = _FakeKernelFn()

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            pass

        return launch


@pytest.fixture(params=["legacy", "0.5.19"])
def alloc_mod(monkeypatch, request):
    _install_fake_torch(monkeypatch)
    _install_fake_sglang_utils(monkeypatch)

    base = FakeLegacyBase if request.param == "legacy" else FakeModernBase
    module: Any = types.ModuleType("sglang.srt.mem_cache.allocator")
    module.BaseTokenToKVPoolAllocator = base
    module.alloc_extend_kernel = _FakeKernel()
    module.alloc_decode_kernel = _FakeKernel()

    patch = ElasticAllocatorPatch()
    assert patch.inject_elastic_allocator(module) is True
    assert patch.inject_elastic_paged_allocator(module) is True
    return module


def _make_token_allocator(alloc_mod):
    return alloc_mod.ElasticTokenToKVPoolAllocator(
        size=16, dtype=object(), device="cuda:0", kvcache=FakeKVCache()
    )


def _make_paged_allocator(alloc_mod):
    return alloc_mod.ElasticPagedTokenToKVPoolAllocator(
        size=64, page_size=4, dtype=object(), device="cuda:0",
        kvcache=FakeKVCache(),
    )


def test_token_free_page_ids_releases_token_ids(alloc_mod):
    allocator = _make_token_allocator(alloc_mod)
    manager = allocator.kvcached_allocator

    allocator.free_page_ids(FakeTensor([3, 5]))

    assert manager.free_calls == [[3, 5]]


def test_token_free_page_ids_defers_inside_group(alloc_mod):
    allocator = _make_token_allocator(alloc_mod)
    manager = allocator.kvcached_allocator

    allocator.free_group_begin()
    allocator.free_page_ids(FakeTensor([3]))
    allocator.free_page_ids(FakeTensor([5]))
    assert manager.free_calls == []
    allocator.free_group_end()

    assert manager.free_calls == [[3, 5]]


def test_token_free_still_works_outside_group(alloc_mod):
    allocator = _make_token_allocator(alloc_mod)
    manager = allocator.kvcached_allocator

    allocator.free(FakeTensor([7, 8]))

    assert manager.free_calls == [[7, 8]]


def test_paged_free_page_ids_releases_exact_page_ids(alloc_mod):
    allocator = _make_paged_allocator(alloc_mod)
    manager = allocator.kvcached_allocator

    # Page ids arrive pre-reduced (SGLang's contract: no page twice, no
    # dedup), so no division by page_size may happen here.
    allocator.free_page_ids(FakeTensor([2, 7]))

    assert manager.free_calls == [[2, 7]]


def test_paged_free_page_ids_empty_is_noop(alloc_mod):
    allocator = _make_paged_allocator(alloc_mod)
    manager = allocator.kvcached_allocator

    allocator.free_page_ids(FakeTensor([]))

    assert manager.free_calls == []


def test_paged_group_defers_page_ids_and_token_indices(alloc_mod):
    allocator = _make_paged_allocator(alloc_mod)
    manager = allocator.kvcached_allocator

    allocator.free_group_begin()
    allocator.free_page_ids(FakeTensor([2]))
    allocator.free_page_ids(FakeTensor([7]))
    allocator.free(FakeTensor([12, 13, 14, 15]))
    assert manager.free_calls == []
    allocator.free_group_end()

    # The token-index pile flushes first (page 3 via unique // page_size),
    # then the page-id pile flushes as-is, mirroring the native order.
    assert manager.free_calls == [[3], [2, 7]]


def test_paged_group_end_without_frees_releases_nothing(alloc_mod):
    allocator = _make_paged_allocator(alloc_mod)
    manager = allocator.kvcached_allocator

    allocator.free_group_begin()
    allocator.free_group_end()

    assert manager.free_calls == []


def test_paged_clear_resets_group_state(alloc_mod):
    allocator = _make_paged_allocator(alloc_mod)
    manager = allocator.kvcached_allocator

    allocator.free_group_begin()
    allocator.free_page_ids(FakeTensor([2]))
    allocator.clear()

    # clear() drops the pending group like the native allocator and returns
    # to the not-in-group state, so the next free releases immediately and a
    # fresh group can start.
    allocator.free_page_ids(FakeTensor([5]))
    assert manager.free_calls == [[5]]

    allocator.free_group_begin()
    allocator.free_group_end()
    assert manager.free_calls == [[5]]


def test_modern_group_defers_a_copy_of_the_tensor(monkeypatch):
    _install_fake_torch(monkeypatch)
    _install_fake_sglang_utils(monkeypatch)

    module: Any = types.ModuleType("sglang.srt.mem_cache.allocator")
    module.BaseTokenToKVPoolAllocator = FakeModernBase
    module.alloc_extend_kernel = _FakeKernel()
    module.alloc_decode_kernel = _FakeKernel()
    patch = ElasticAllocatorPatch()
    assert patch.inject_elastic_paged_allocator(module) is True

    allocator = module.ElasticPagedTokenToKVPoolAllocator(
        size=64, page_size=4, dtype=object(), device="cuda:0",
        kvcache=FakeKVCache(),
    )
    manager = allocator.kvcached_allocator

    deferred = FakeTensor([2])
    allocator.free_group_begin()
    allocator.free_page_ids(deferred)
    deferred.data[0] = 9  # caller mutates its tensor after the deferred free
    allocator.free_group_end()

    assert manager.free_calls == [[2]]
