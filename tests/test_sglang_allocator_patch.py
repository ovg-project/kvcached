# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import inspect
import sys
import types
from typing import Any

import pytest

from kvcached.integration.sglang.patches import ElasticAllocatorPatch


class FakeTensor:
    def __init__(self, data=None, shape=None, dtype=None, device=None):
        self.data = data
        self.shape = shape
        self.dtype = dtype
        self.device = device

    def __len__(self):
        if self.shape:
            return self.shape[0]
        return len(self.data)


class FakeKVCachedAllocator:
    def alloc(self, num_pages):
        return list(range(num_pages))


class FakeKVCache:
    def __init__(self):
        self.kvcached_allocator = FakeKVCachedAllocator()


class FakeBaseTokenToKVPoolAllocator:
    def __init__(self, size, page_size, dtype, device, kvcache, *args, **kwargs):
        self.size = size
        self.page_size = page_size
        self.dtype = dtype
        self.device = device
        self.kvcache = kvcache
        self.is_not_in_free_group = True
        self.free_group = []


class FakeTritonKernel:
    def __init__(self, fn):
        self.fn = fn
        self.calls = []

    def __getitem__(self, grid):
        def launch(**kwargs):
            expected_names = tuple(inspect.signature(self.fn).parameters)
            if set(kwargs) != set(expected_names):
                missing = set(expected_names) - set(kwargs)
                unexpected = set(kwargs) - set(expected_names)
                raise AssertionError(
                    f"kernel kwargs mismatch: missing={missing}, unexpected={unexpected}"
                )
            self.calls.append({"grid": grid, "kwargs": kwargs})

        return launch


class FakeKernelFn:
    def __init__(self, parameter_names):
        self.__signature__ = inspect.Signature(
            [
                inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                for name in parameter_names
            ]
        )

    def __call__(self, *args, **kwargs):
        pass


def _install_fake_torch(monkeypatch):
    torch: Any = types.ModuleType("torch")
    torch.Tensor = FakeTensor
    torch.int64 = "int64"

    def empty(shape, dtype=None, device=None):
        return FakeTensor(shape=shape, dtype=dtype, device=device)

    def tensor(data, dtype=None, device=None):
        return FakeTensor(data=list(data), shape=(len(data),), dtype=dtype, device=device)

    torch.empty = empty
    torch.tensor = tensor
    monkeypatch.setitem(sys.modules, "torch", torch)


def _install_fake_sglang_utils(monkeypatch):
    sglang: Any = types.ModuleType("sglang")
    srt: Any = types.ModuleType("sglang.srt")
    utils: Any = types.ModuleType("sglang.srt.utils")

    def next_power_of_2(value):
        return 1 << (value - 1).bit_length()

    utils.get_num_new_pages = lambda **kwargs: 0
    utils.next_power_of_2 = next_power_of_2
    sglang.srt = srt
    srt.utils = utils

    monkeypatch.setitem(sys.modules, "sglang", sglang)
    monkeypatch.setitem(sys.modules, "sglang.srt", srt)
    monkeypatch.setitem(sys.modules, "sglang.srt.utils", utils)


def _make_allocator_module(alloc_extend_kernel):
    alloc_mod: Any = types.ModuleType("sglang.srt.mem_cache.allocator")
    alloc_mod.BaseTokenToKVPoolAllocator = FakeBaseTokenToKVPoolAllocator
    alloc_mod.alloc_extend_kernel = alloc_extend_kernel
    alloc_mod.alloc_decode_kernel = FakeTritonKernel(
        FakeKernelFn(
            (
                "seq_lens_ptr",
                "last_loc_ptr",
                "free_page_ptr",
                "out_indices",
                "bs_upper",
                "page_size",
            )
        )
    )
    return alloc_mod


@pytest.mark.parametrize(
    ("parameter_names", "expected_optional_kwargs"),
    [
        (
            (
                "pre_lens_ptr",
                "seq_lens_ptr",
                "last_loc_ptr",
                "free_page_ptr",
                "out_indices",
                "ret_values",
                "bs_upper",
                "page_size",
                "max_num_extend_tokens",
            ),
            {"ret_values", "max_num_extend_tokens"},
        ),
        (
            (
                "pre_lens_ptr",
                "seq_lens_ptr",
                "last_loc_ptr",
                "free_page_ptr",
                "out_indices",
                "bs_upper",
                "page_size",
                "max_num_extend_tokens",
            ),
            {"max_num_extend_tokens"},
        ),
        (
            (
                "pre_lens_ptr",
                "seq_lens_ptr",
                "last_loc_ptr",
                "free_page_ptr",
                "out_indices",
                "bs_upper",
                "page_size",
            ),
            set(),
        ),
    ],
    ids=["sglang-0.4.9-0.5.2", "sglang-0.5.5-0.5.9", "sglang-0.5.10-0.5.15"],
)
def test_alloc_extend_kernel(
    monkeypatch, parameter_names, expected_optional_kwargs
):
    """Regression: supported SGLang versions use 9-, 8-, and 7-argument
    alloc_extend_kernel signatures, so the patch must pass only the optional
    names present in the installed kernel."""
    _install_fake_torch(monkeypatch)
    _install_fake_sglang_utils(monkeypatch)
    alloc_extend_kernel = FakeTritonKernel(FakeKernelFn(parameter_names))
    alloc_mod = _make_allocator_module(alloc_extend_kernel)

    assert ElasticAllocatorPatch().inject_elastic_paged_allocator(alloc_mod) is True

    allocator = alloc_mod.ElasticPagedTokenToKVPoolAllocator(
        size=64,
        page_size=4,
        dtype=object(),
        device="cuda:0",
        kvcache=FakeKVCache(),
    )
    prefix_lens = FakeTensor(shape=(2,))
    seq_lens = FakeTensor(shape=(2,))
    out_indices = allocator.alloc_extend(
        prefix_lens=prefix_lens,
        prefix_lens_cpu=prefix_lens,
        seq_lens=seq_lens,
        seq_lens_cpu=seq_lens,
        last_loc=FakeTensor(shape=(2,)),
        extend_num_tokens=5,
        num_new_pages=2,
    )

    assert isinstance(out_indices, FakeTensor)
    assert len(alloc_extend_kernel.calls) == 1
    call = alloc_extend_kernel.calls[0]
    kwargs = call["kwargs"]
    assert call["grid"] == (2,)
    assert kwargs["bs_upper"] == 2
    assert kwargs["page_size"] == 4
    assert kwargs["free_page_ptr"].data == [0, 1]
    assert kwargs["out_indices"] is out_indices
    assert expected_optional_kwargs <= set(kwargs)
    assert ("ret_values" in kwargs) == ("ret_values" in expected_optional_kwargs)
    assert ("max_num_extend_tokens" in kwargs) == (
        "max_num_extend_tokens" in expected_optional_kwargs
    )
    if "max_num_extend_tokens" in kwargs:
        assert kwargs["max_num_extend_tokens"] == 8
