# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import importlib
import importlib.machinery
import sys
import types
from typing import Any
from unittest import mock


class FakeTensor(list):
    def numel(self):
        return len(self)

    def detach(self):
        return self

    def __floordiv__(self, value):
        return FakeTensor([item // value for item in self])

    def numpy(self):
        raise AssertionError("free page ids should use tolist() directly")

    def tolist(self):
        return list(self)


class TrackingTensor(FakeTensor):
    def __init__(self, values, reported_numel=None):
        super().__init__(values)
        self.to_calls = []
        self.reported_numel = reported_numel

    def numel(self):
        if self.reported_numel is not None:
            return self.reported_numel
        return super().numel()

    def to(self, *args, **kwargs):
        self.to_calls.append((args, kwargs))
        return self


def _load_sglang_patches(monkeypatch):
    torch_mock = mock.MagicMock()
    torch_mock.__version__ = "2.6.0"
    torch_mock.int64 = "int64"
    torch_mock.empty.side_effect = lambda shape, **_kwargs: FakeTensor(
        [0] * (shape if isinstance(shape, int) else shape[0])
    )
    torch_mock.unique.side_effect = lambda values: FakeTensor(sorted(set(values)))
    monkeypatch.setitem(sys.modules, "torch", torch_mock)

    utils_mod: Any = types.ModuleType("sglang.srt.utils")
    utils_mod.next_power_of_2 = lambda value: 1 << (int(value) - 1).bit_length()
    utils_mod.get_num_new_pages = lambda **_kwargs: 0

    for name in ("sglang", "sglang.srt"):
        module = types.ModuleType(name)
        module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setitem(sys.modules, "sglang.srt.utils", utils_mod)

    from kvcached.integration.sglang import patches

    importlib.reload(patches)
    return patches


def _make_paged_allocator(patches):
    class BaseTokenToKVPoolAllocator:
        def __init__(self, size, page_size, _dtype, device, _kvcache):
            self.size = size
            self.page_size = page_size
            self.device = device
            self.free_group = []
            self.is_not_in_free_group = True

    class Kernel:
        def __call__(self, **_kwargs):
            return None

        def __getitem__(self, _grid):
            def run(**_kwargs):
                return None

            return run

    alloc_mod = types.SimpleNamespace(
        BaseTokenToKVPoolAllocator=BaseTokenToKVPoolAllocator,
        alloc_extend_kernel=Kernel(),
        alloc_decode_kernel=Kernel(),
    )

    assert patches.ElasticAllocatorPatch().inject_elastic_paged_allocator(alloc_mod)

    freed = []
    allocator = types.SimpleNamespace(
        alloc=lambda need_size: list(range(need_size)),
        free=lambda indices: freed.append(indices),
    )
    kvcache = types.SimpleNamespace(kvcached_allocator=allocator)
    paged = alloc_mod.ElasticPagedTokenToKVPoolAllocator(
        128,
        16,
        object(),
        "cuda:0",
        kvcache,
    )

    return paged, freed


def test_elastic_paged_allocator_moves_small_free_to_cpu(monkeypatch):
    patches = _load_sglang_patches(monkeypatch)
    paged, freed = _make_paged_allocator(patches)

    free_index = TrackingTensor([0, 16, 32, 48, 48])
    paged.free(free_index)

    assert free_index.to_calls == [((), {"device": "cpu", "non_blocking": False})]
    assert freed == [[0, 1, 2, 3]]


def test_elastic_paged_allocator_reduces_large_free_before_cpu_copy(monkeypatch):
    patches = _load_sglang_patches(monkeypatch)
    paged, freed = _make_paged_allocator(patches)

    free_index = TrackingTensor(
        [0, 16, 32, 48, 48],
        reported_numel=patches.PAGED_FREE_CPU_UNIQUE_MAX_INDICES + 1,
    )
    paged.free(free_index)

    assert free_index.to_calls == []
    assert freed == [[0, 1, 2, 3]]
