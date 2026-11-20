# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Exercise the production clear operation with a CPU physical-page model."""

import ast
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import List

import pytest


def _clear_locked():
    source = Path(__file__).parents[1] / "kvcached" / "kv_cache_manager.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body
               if isinstance(node, ast.ClassDef) and node.name == "KVCacheManager")
    method = next(node for node in cls.body
                  if isinstance(node, ast.FunctionDef) and node.name == "_clear_locked")
    scope = {"List": List, "StateConsistencyError": RuntimeError}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), scope)
    return scope["_clear_locked"]


class Page:
    def __init__(self, page_id):
        self.page_id = page_id
        self.payload = bytearray(b"padding contents")
        self.init(1)

    def init(self, block_mem_size, max_block_id=None):
        assert block_mem_size == 1
        self.free = list(range(4 * self.page_id, min(
            4 * (self.page_id + 1), max_block_id or 16)))

    def alloc(self, count):
        result, self.free = self.free[:count], self.free[count:]
        return result

    def num_free_blocks(self):
        return len(self.free)


@pytest.mark.parametrize("controlled", [False, True])
@pytest.mark.parametrize("max_block_id", [None, 1, 3])
def test_clear_retains_padding_without_a_new_physical_grant(controlled, max_block_id):
    padding, live = Page(0), Page(1)
    padding.alloc(4)
    live.alloc(1)
    mapped = {0: padding, 1: live, 2: Page(2)}
    freed = []
    events = []

    def free_pages(ids):
        assert 0 not in ids, "flush must not unmap padding"
        assert len(ids) == len(set(ids))
        for page_id in ids:
            mapped.pop(page_id)
        freed.extend(ids)

    def trim():
        # A background reservation is also returned, but live padding stays.
        mapped.pop(2, None)

    def reserve():
        pytest.fail("flush must not request physical capacity for existing padding")

    manager = SimpleNamespace(
        page_allocator=SimpleNamespace(
            stop_prealloc_thread=lambda: events.append("stop"),
            free_pages=free_pages,
            reset_free_page_order=lambda: events.append("reset"),
        ),
        free_reserved=lambda: None, trim=trim, _reserve_null_block=reserve,
        _increment_operation_counter=lambda *args: None,
        _apply_resize_target=lambda: events.append("resize"),
        _shutdown_requested=threading.Event(),
        null_block=[0], avail_pages={1: live}, full_pages={0: padding},
        block_mem_size=1, max_block_id=max_block_id, num_avail_blocks=3,
        _retired_pages=[(1, [1])], _logical_allocated_blocks=5,
    )
    if controlled:
        manager._runtime_control = SimpleNamespace(executing=(1,), acknowledged=(1,))

    clear = _clear_locked()
    for _ in range(2):
        clear(manager)
        assert mapped == {0: padding}
        assert padding.payload == b"padding contents"
        assert manager.null_block == [0]
        assert manager._logical_allocated_blocks == 1
        assert manager.num_avail_blocks == (max_block_id or 4) - 1
        assert manager.target_num_pages is None
        assert not manager.in_shrink
        assert manager._retired_pages == []
        assert manager._avail_physical_pages_cache is None
        if max_block_id == 1:
            assert manager.full_pages == {0: padding}
            assert manager.avail_pages == {}
        else:
            assert manager.avail_pages == {0: padding}
            assert manager.full_pages == {}
    assert freed == [1]
    assert events == ["stop", "reset", "resize"] * 2
    if controlled:
        assert manager._runtime_control.executing == ()
        assert manager._runtime_control.acknowledged == ()


def test_clear_without_padding_still_uses_normal_reservation():
    calls = []
    manager = SimpleNamespace(
        page_allocator=SimpleNamespace(
            stop_prealloc_thread=lambda: None,
            reset_free_page_order=lambda: None,
        ),
        free_reserved=lambda: None, trim=lambda: None,
        _reserve_null_block=lambda: calls.append("reserve"),
        _apply_resize_target=lambda: None,
        _shutdown_requested=threading.Event(),
        null_block=None, avail_pages={}, full_pages={},
    )
    _clear_locked()(manager)
    assert calls == ["reserve"]
    assert manager._logical_allocated_blocks == 0
