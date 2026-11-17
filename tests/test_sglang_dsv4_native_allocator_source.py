# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Execute native allocator page selection/free code on CPU, without CUDA kernels."""

import abc
import ast
import os
from pathlib import Path
from typing import Any

import pytest
import torch
from test_sglang_dsv4_native_backing import MODULE, Manager

SOURCE = Path(os.getenv("SGLANG_SOURCE", "engine_integration/sglang")) / "python/sglang/srt/mem_cache/allocator"


def native_paged_allocator():
    namespace: dict[str, Any] = {
        "abc": abc, "torch": torch, "get_bool_env_var": lambda _: False, "_is_hip": False,
    }
    for filename, name in (("base.py", "BaseTokenToKVPoolAllocator"),
                           ("paged.py", "PagedTokenToKVPoolAllocator")):
        source = ast.parse((SOURCE / filename).read_text(encoding="utf-8"))
        definition = next(node for node in source.body
                          if isinstance(node, ast.ClassDef) and node.name == name)
        module = ast.Module(body=[ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0), definition],
            type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), filename, "exec"), namespace)
    return namespace["PagedTokenToKVPoolAllocator"]


@pytest.mark.skipif(not (SOURCE / "paged.py").is_file(), reason="native SGLang source unavailable")
@pytest.mark.parametrize("need_sort", [False, True])
@pytest.mark.parametrize("grouped", [False, True])
def test_native_free_order_keeps_decode_reservation_backed(need_sort, grouped):
    native = native_paged_allocator()
    full = native(48, 4, torch.int64, "cpu", None, need_sort)
    swa = native(48, 4, torch.int64, "cpu", None, need_sort)
    full.alloc(12)
    swa.alloc(12)
    backing = MODULE.SWABacking(Manager(), native_pages=13, native_page_bytes=6,
                                physical_page_bytes=8)
    assert backing.acquire({1, 2, 3})

    class Composite:
        page_size = 4
        full_attn_allocator = full
        swa_attn_allocator = swa

        def check_decode_capacity(self, *, num_tokens):
            return min(full.available_size(), swa.available_size()) >= num_tokens

        def alloc(self):
            # The public CPU allocation consumes the same native page prefix
            # as the CUDA decode kernel; this test does not execute that kernel.
            return full.alloc(4), swa.alloc(4)

    allocator = Composite()
    MODULE.bind_native_swa(allocator, backing)
    assert allocator.check_decode_capacity(num_tokens=4)
    claims = list(backing.manager.claims)
    backing.manager.budget = len(backing.manager.live)
    if grouped:
        swa.free_group_begin()
    swa.free_page_ids(torch.tensor([1]))
    if grouped:
        swa.free_group_end()
    assert backing.live == {1, 2, 3, 4}
    _, selected = allocator.alloc()
    page = 4 if need_sort else 1
    assert selected.tolist() == list(range(page * 4, page * 4 + 4))
    assert backing.live == {2, 3, page}
    assert backing.manager.claims == claims


@pytest.mark.skipif(not (SOURCE / "paged.py").is_file(), reason="native SGLang source unavailable")
@pytest.mark.parametrize("recovery", ["capacity", "prefix"])
def test_failed_native_prefix_waits_for_placement_progress(recovery):
    native = native_paged_allocator()
    full = native(256, 4, torch.int64, "cpu", None, False)
    swa = native(256, 4, torch.int64, "cpu", None, False)
    for child in (full, swa):
        child.alloc(80)
        child.free_page_ids(torch.arange(2, 20))
        child.free_page_ids(torch.tensor([20]))
    manager = Manager(budget=0)
    backing = MODULE.SWABacking(manager, native_pages=65, native_page_bytes=6,
                                physical_page_bytes=32)
    assert backing.acquire({1})  # This live page shares permanent padding.

    class Composite:
        page_size = 4
        full_attn_allocator = full
        swa_attn_allocator = swa

        def alloc(self):
            return full.alloc(4), swa.alloc(4)

    allocator = Composite()
    MODULE.bind_native_swa(allocator, backing)
    assert swa.available_size() > 0  # Aggregate capacity is initially optimistic.
    assert allocator.alloc() is None
    for _ in range(3):
        assert swa.available_size() == 0
        assert swa.get_all_free_pages()[0].item() == 20
        assert not manager.claims
    if recovery == "capacity":
        manager.budget = 1
        expected = 20
    else:
        swa.free_page_ids(torch.tensor([1]))
        expected = 1
    assert swa.available_size() > 0
    assert not manager.claims  # Availability did not reserve or reorder anything.
    _, selected = allocator.alloc()
    assert selected.tolist() == list(range(expected * 4, expected * 4 + 4))
